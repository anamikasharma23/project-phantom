"""Signal DNA: how much should WALL-E trust each signal source?

Each long-range source (e.g. "beacon", "relay") gets a Beta(a, b) counter.
Think of it as "a = times it agreed with independent evidence,
b = times it disagreed", starting from a small prior. Its mean a/(a+b) is the
reliability r used by the belief model:

    likelihood of a bearing = r * (bearing points at this cell)
                            + (1 - r) * (bearing is random noise)

AVOIDING CIRCULAR LOGIC
If we judged a source against a belief built from that same source, it would
always look trustworthy ("I agree with myself"). Judging sources against EACH
OTHER is also unsafe: a confident decoy can make an honest source look wrong.
So readings are judged only against an INDEPENDENT REFERENCE belief built
from evidence that no long-range signal source produced: empty local scans,
physical clues (tracks, residue) and the timestamped initial signal (weighted
down by its age). The judgment is a likelihood ratio:

    q_true  = how well the bearing fits the reference belief
    q_noise = how well a random bearing would fit (1 / 2pi)
    agreement = q_true / (q_true + q_noise)        (between 0 and 1)

    a += agreement,  b += 1 - agreement

A reading is only judged once the reference belief has a real directional
opinion as seen from where the reading was taken (the spread of its
probability over 36 compass directions is below 85% of the maximum).
Until then it waits as "pending": grading a bearing against a map that
"points everywhere" would tell us nothing.

HARD ANCHOR
When a local scan actually sees EVE, every past reading is re-scored
against that confirmed cell (agreement = bearing error <= 2.5 noise std-devs).
That is the strongest evidence available, so it replaces the soft scores.
"""
import math

import numpy as np

from . import sensors
from .geometry import angle_diff_deg, bearing_deg

INFORMATIVE_FRACTION = 0.85   # judge a reading once the reference's bearing spread < 85% of max


def beta_quantile(a, b, q):
    """Quantile of Beta(a, b) by numeric integration (no scipy needed)."""
    xs = np.linspace(1e-6, 1 - 1e-6, 2001)
    logpdf = (a - 1) * np.log(xs) + (b - 1) * np.log(1 - xs)
    pdf = np.exp(logpdf - logpdf.max())
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, xs))


class SignalDNA:
    def __init__(self, source_specs, prior=(2.0, 2.0), enabled=True, fixed_reliability=0.8):
        self.specs = source_specs
        self.enabled = enabled
        self.fixed = fixed_reliability
        self.prior = tuple(prior)
        self.a = {s: float(prior[0]) for s in source_specs}
        self.b = {s: float(prior[1]) for s in source_specs}
        self.judgments = {s: [] for s in source_specs}     # per reading: dict(evidence, agreement, method)
        self.pending = []                                  # evidence ids waiting for an informative check
        self.anchored = False

    # ------------------------------------------------------------------
    def reliability(self, source):
        if not self.enabled:
            return self.fixed
        return self.a[source] / (self.a[source] + self.b[source])

    def reliabilities(self):
        return {s: self.reliability(s) for s in self.specs}

    def add_pending(self, ev):
        if ev.source in self.specs:
            self.pending.append(ev)

    # ------------------------------------------------------------------
    def reference_belief(self, belief, current_rel):
        """Belief from non-signal evidence only (scans, clues, initial signal)."""
        return belief.compute(current_rel, exclude_sources=set(self.specs))

    def soft_update(self, belief, current_rel):
        """Judge pending readings against the independent reference belief."""
        if not self.enabled or self.anchored:
            return []
        made = []
        ref = self.reference_belief(belief, current_rel)
        loc = ref.sum(axis=0)
        still = []
        for ev in self.pending:
            spread = belief.bearing_entropy_fraction(tuple(ev.summary["origin"]), loc)
            if spread > INFORMATIVE_FRACTION:
                still.append(ev)                       # reference has no directional opinion yet
                continue
            q_true = float((loc * ev.vm).sum())
            q_noise = sensors.UNIFORM_BEARING_PDF
            agree = q_true / (q_true + q_noise)
            self.a[ev.source] += agree
            self.b[ev.source] += 1 - agree
            j = {"evidence": ev.id, "step": ev.step, "agreement": round(agree, 3),
                 "q_true": round(q_true, 4), "q_noise": round(q_noise, 4),
                 "reference_bearing_spread": round(spread, 3),
                 "method": "vs independent reference (scans, clues, initial signal)"}
            self.judgments[ev.source].append(j)
            made.append(dict(j, source=ev.source))
        self.pending = still
        return made

    def anchor_update(self, evidence_list, eve_cell):
        """EVE was seen: re-score every past reading against her confirmed cell."""
        if not self.enabled:
            return []
        self.anchored = True
        made = []
        for s in self.specs:
            self.a[s], self.b[s] = self.prior
            self.judgments[s] = []
        for ev in evidence_list:
            if ev.kind != "bearing" or ev.source not in self.specs:
                continue
            origin = tuple(ev.summary["origin"])
            if origin == tuple(eve_cell):
                continue
            err = abs(angle_diff_deg(ev.summary["bearing_deg"], bearing_deg(origin, eve_cell)))
            tol = 2.5 * self.specs[ev.source]["noise_deg"]
            agree = 1.0 if err <= tol else 0.0
            self.a[ev.source] += agree
            self.b[ev.source] += 1 - agree
            j = {"evidence": ev.id, "step": ev.step, "agreement": agree,
                 "bearing_error_deg": round(err, 1), "tolerance_deg": round(tol, 1),
                 "method": "anchored to confirmed EVE position"}
            self.judgments[ev.source].append(j)
            made.append(dict(j, source=ev.source))
        self.pending = []
        return made

    # ------------------------------------------------------------------
    def label(self, source):
        """Plain-English trust label. 'Conflicting' needs strong evidence."""
        a, b = self.a[source], self.b[source]
        n = a + b - sum(self.prior)
        mean = a / (a + b)
        lo, hi = beta_quantile(a, b, 0.05), beta_quantile(a, b, 0.95)
        if not self.enabled:
            return "fixed (DNA off)"
        if n < 3:
            return "uncertain"
        if lo > 0.6:
            return "trusted"
        if hi < 0.45 and n >= 4:
            return "conflicting"
        if mean < 0.5:
            return "doubtful"
        return "uncertain"

    def snapshot(self):
        out = {}
        for s in self.specs:
            a, b = self.a[s], self.b[s]
            out[s] = {"a": round(a, 3), "b": round(b, 3),
                      "reliability": round(self.reliability(s), 4),
                      "interval_90": [round(beta_quantile(a, b, 0.05), 3), round(beta_quantile(a, b, 0.95), 3)],
                      "judged": len(self.judgments[s]),
                      "pending": sum(1 for e in self.pending if e.source == s),
                      "label": self.label(s), "anchored": self.anchored}
        return out
