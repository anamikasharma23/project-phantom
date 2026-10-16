"""WALL-E's belief about EVE: a probability grid.

We keep one probability for every (status, cell) pair:
    P[status, y, x] = "how likely EVE is at (x, y) AND has this status"
with status 0 = active, 1 = dormant. The whole array always sums to 1.

How it updates (Bayes' rule, the textbook "discrete Bayes filter"):
    new belief  ∝  prior  ×  likelihood(evidence 1)  ×  likelihood(evidence 2) ...
Each piece of evidence (a signal bearing, an empty scan, a clue) says, for
every cell, "how likely would I be to see this if EVE were here?".

We store all evidence and recompute the product from scratch. That is cheap
on this grid size and lets the agent re-weigh old signals when it learns a
source is more or less trustworthy (Signal DNA), or leave one source out to
check that source against the others.
"""
import math

import numpy as np

from . import sensors
from .geometry import ROCK

PROB_FLOOR = 1e-12      # no state is ever set to exactly zero (recovers from noise)


def entropy_bits(p):
    q = p[p > 0]
    return float(-(q * np.log2(q)).sum())


class Evidence:
    """One stored piece of evidence with its cached likelihood pieces."""

    def __init__(self, eid, step, kind, source, summary, fixed_loglik=None,
                 vm=None, heard_loglik=None, reliability=None):
        self.id = eid
        self.step = step
        self.kind = kind              # bearing, beacon_silence, scan_empty, scan_detect, status, clue_*, initial
        self.source = source          # signal source name, or "local_scan" / "clue" / "initial"
        self.summary = summary        # small JSON-able dict for the trace
        self.fixed_loglik = fixed_loglik   # (2,H,W) or None
        self.vm = vm                       # (H,W) von Mises density, for reliability-weighted bearings
        self.heard_loglik = heard_loglik   # (2,H,W) beacon detection + strength term
        self.reliability = reliability     # fixed reliability (clues/initial) or None = use Signal DNA

    def loglik(self, rel):
        """Log-likelihood grid (2,H,W). `rel` is the reliability to use."""
        out = np.zeros((2,) + (self.vm.shape if self.vm is not None else self.fixed_loglik.shape[1:]))
        if self.fixed_loglik is not None:
            out = out + self.fixed_loglik
        if self.vm is not None:
            r = self.reliability if self.reliability is not None else rel
            mix = r * self.vm + (1.0 - r) * sensors.UNIFORM_BEARING_PDF
            out = out + np.log(mix)[None, :, :]
        if self.heard_loglik is not None:
            out = out + self.heard_loglik
        return out


class BeliefModel:
    def __init__(self, info):
        self.info = info
        self.W, self.H = info["width"], info["height"]
        terrain = np.array(info["terrain"])
        self.passable = terrain != ROCK
        ys, xs = np.mgrid[0:self.H, 0:self.W]
        self.X, self.Y = xs.astype(float), ys.astype(float)
        p_act = float(info["agent"]["status_prior_active"])
        n = self.passable.sum()
        prior = np.zeros((2, self.H, self.W))
        prior[0][self.passable] = p_act / n
        prior[1][self.passable] = (1 - p_act) / n
        with np.errstate(divide="ignore"):
            self.log_prior = np.log(prior)
        self.evidence = []
        self.P = prior.copy()

    # ---------------- geometry grids ----------------
    def dist_from(self, origin):
        return np.hypot(self.X - origin[0], self.Y - origin[1])

    def bearing_grid_from(self, origin):
        """Compass bearing from origin to every cell (deg)."""
        return np.degrees(np.arctan2(self.X - origin[0], -(self.Y - origin[1]))) % 360.0

    def _vm_grid(self, origin, bearing, noise_deg):
        delta = (self.bearing_grid_from(origin) - bearing + 180.0) % 360.0 - 180.0
        vm = sensors.von_mises_pdf(delta, sensors.kappa_from_noise(noise_deg))
        d = self.dist_from(origin)
        vm[d < 0.5] = sensors.UNIFORM_BEARING_PDF   # bearing is meaningless at the origin cell
        return vm

    # ---------------- evidence constructors ----------------
    def _add(self, ev):
        self.evidence.append(ev)
        return ev

    def add_bearing(self, step, reading, spec):
        """A long-range reading with a bearing (beacon heard, or relay report)."""
        origin = tuple(reading["origin"])
        vm = self._vm_grid(origin, reading["bearing_deg"], spec["noise_deg"])
        heard = None
        if reading["kind"] == "beacon":
            d = self.dist_from(origin)
            p_a = np.clip(sensors.beacon_hear_prob(d, "active", spec), 1e-9, 1)
            p_d = np.clip(sensors.beacon_hear_prob(d, "dormant", spec), 1e-9, 1)
            heard = np.stack([np.log(p_a), np.log(p_d)])
            if reading.get("strength") is not None:
                mu = sensors.beacon_strength_mean(d, spec)
                s = sensors.STRENGTH_NOISE
                ll = -0.5 * ((reading["strength"] - mu) / s) ** 2
                heard = heard + ll[None, :, :]
        summ = {"source": reading["source"], "origin": list(origin),
                "bearing_deg": reading["bearing_deg"], "strength": reading.get("strength")}
        return self._add(Evidence(f"e{len(self.evidence)}", step, "bearing", reading["source"], summ,
                                  vm=vm, heard_loglik=heard))

    def add_beacon_silence(self, step, reading, spec):
        origin = tuple(reading["origin"])
        d = self.dist_from(origin)
        miss_a = np.log(np.clip(1 - sensors.beacon_hear_prob(d, "active", spec), 1e-9, 1))
        miss_d = np.log(np.clip(1 - sensors.beacon_hear_prob(d, "dormant", spec), 1e-9, 1))
        return self._add(Evidence(f"e{len(self.evidence)}", step, "beacon_silence", reading["source"],
                                  {"source": reading["source"], "origin": list(origin)},
                                  fixed_loglik=np.stack([miss_a, miss_d])))

    def add_initial_signal(self, sig, trust, beacon_spec):
        origin = tuple(sig["origin"])
        fixed = np.zeros((2, self.H, self.W))
        vm = None
        if "bearing_deg" in sig and sig["bearing_deg"] is not None:
            vm = self._vm_grid(origin, sig["bearing_deg"], sig.get("noise_deg", 15.0))
        if sig.get("coordinates"):
            cx, cy = sig["coordinates"]
            sd = sig.get("coord_sigma", 3.0)
            g = np.exp(-0.5 * ((self.X - cx) ** 2 + (self.Y - cy) ** 2) / sd ** 2)
            g = trust * g / g.max() + (1 - trust)      # reliability-weighted mixture
            fixed = fixed + np.log(g)[None, :, :]
        if sig.get("strength") is not None and beacon_spec is not None:
            d = self.dist_from(origin)
            mu = sensors.beacon_strength_mean(d, beacon_spec)
            lik = np.exp(-0.5 * ((sig["strength"] - mu) / 0.12) ** 2)
            fixed = fixed + np.log(trust * lik + (1 - trust) * lik.mean() + 1e-12)[None, :, :]
        summ = {"source": "initial_signal", "origin": list(origin), "bearing_deg": sig.get("bearing_deg"),
                "coordinates": sig.get("coordinates"), "strength": sig.get("strength"),
                "trust_used": round(trust, 3)}
        return self._add(Evidence(f"e{len(self.evidence)}", 0, "initial", "initial_signal", summ,
                                  fixed_loglik=fixed, vm=vm, reliability=trust))

    def add_scan(self, step, scan, local_spec):
        center = tuple(scan["center"])
        inside = self.dist_from(center) <= scan["radius"] + 1e-9
        if scan["eve_detected"]:
            ex, ey = scan["eve_position"]
            grid = np.full((self.H, self.W), math.log(1e-9))
            grid[ey, ex] = 0.0
            ev = self._add(Evidence(f"e{len(self.evidence)}", step, "scan_detect", "local_scan",
                                    {"center": list(center), "eve_position": [ex, ey]},
                                    fixed_loglik=np.stack([grid, grid])))
            acc = local_spec["status_accuracy"]
            s_idx = 0 if scan["status_reading"] == "active" else 1
            st = np.zeros((2, self.H, self.W))
            st[s_idx] += math.log(acc)
            st[1 - s_idx] += math.log(1 - acc)
            self._add(Evidence(f"e{len(self.evidence)}", step, "status", "local_scan",
                               {"status_reading": scan["status_reading"], "accuracy": acc},
                               fixed_loglik=st))
            return ev
        grid = np.where(inside, math.log(1 - local_spec["detect_prob"]), 0.0)
        return self._add(Evidence(f"e{len(self.evidence)}", step, "scan_empty", "local_scan",
                                  {"center": list(center), "radius": scan["radius"]},
                                  fixed_loglik=np.stack([grid, grid])))

    def add_clue(self, step, clue, clue_specs):
        pos = tuple(clue["position"])
        if clue["type"] == "track":
            vm = self._vm_grid(pos, clue["bearing_deg"], clue_specs["track_noise_deg"])
            return self._add(Evidence(f"e{len(self.evidence)}", step, "clue_track", "clue",
                                      {"clue_id": clue["id"], "position": list(pos),
                                       "bearing_deg": clue["bearing_deg"]},
                                      vm=vm, reliability=clue_specs["track_reliability"]))
        acc = clue_specs["residue_accuracy"]
        st = np.zeros((2, self.H, self.W))
        s_idx = 0 if clue["freshness"] == "fresh" else 1
        st[s_idx] += math.log(acc)
        st[1 - s_idx] += math.log(1 - acc)
        near = self.dist_from(pos) <= clue_specs["residue_near_radius"]
        if clue["proximity"] == "near":
            prox = np.where(near, math.log(acc), math.log(1 - acc))
        else:
            prox = np.where(near, math.log(1 - acc), math.log(acc))
        return self._add(Evidence(f"e{len(self.evidence)}", step, "clue_residue", "clue",
                                  {"clue_id": clue["id"], "position": list(pos),
                                   "freshness": clue["freshness"], "proximity": clue["proximity"]},
                                  fixed_loglik=st + prox[None, :, :]))

    # ---------------- inference ----------------
    def compute(self, reliability, exclude_sources=(), exclude_ids=()):
        """Posterior (2,H,W) from all evidence. `reliability` maps source -> r.

        `exclude_sources` leaves whole sources out (used by Signal DNA to build
        a reference belief that does not depend on the sources being judged).
        """
        total = self.log_prior.copy()
        for ev in self.evidence:
            if ev.source in exclude_sources or ev.id in exclude_ids:
                continue
            total = total + ev.loglik(reliability.get(ev.source, 0.5))
        return normalise(total, self.passable)

    def update(self, reliability):
        self.P = self.compute(reliability)
        return self.P

    # ---------------- summaries ----------------
    def location(self, P=None):
        P = self.P if P is None else P
        return P.sum(axis=0)

    def p_active(self, P=None):
        P = self.P if P is None else P
        return float(P[0].sum())

    def entropy(self, P=None):
        return entropy_bits(self.P if P is None else P)

    def top_cells(self, k=5, P=None):
        loc = self.location(P)
        flat = np.argsort(loc.ravel())[::-1][:k]
        return [{"cell": [int(i % self.W), int(i // self.W)], "p": round(float(loc.ravel()[i]), 4)}
                for i in flat]

    def mass_within(self, center, radius, P=None):
        loc = self.location(P)
        return float(loc[self.dist_from(center) <= radius + 1e-9].sum())

    def bearing_entropy_fraction(self, origin, loc, bins=36):
        """How 'directional' a location belief looks from `origin`.

        1.0 = equally likely in every direction (useless for judging a bearing),
        lower = the belief points somewhere.
        """
        b = self.bearing_grid_from(origin)
        d = self.dist_from(origin)
        w = np.where(d >= 0.5, loc, 0.0)
        hist, _ = np.histogram(b.ravel(), bins=bins, range=(0, 360), weights=w.ravel())
        if hist.sum() <= 0:
            return 1.0
        return entropy_bits(hist / hist.sum()) / math.log2(bins)

    def scan_coverage_mass(self, radius, P=None):
        """For every cell c: probability mass within `radius` of c (a disk filter)."""
        loc = self.location(P)
        out = np.zeros_like(loc)
        r = int(math.floor(radius))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > radius * radius + 1e-9:
                    continue
                shifted = np.zeros_like(loc)
                ys = slice(max(0, -dy), self.H - max(0, dy))
                xs = slice(max(0, -dx), self.W - max(0, dx))
                yd = slice(max(0, dy), self.H - max(0, -dy))
                xd = slice(max(0, dx), self.W - max(0, -dx))
                shifted[ys, xs] = loc[yd, xd]
                out += shifted
        return out


def normalise(log_grid, passable):
    """exp + normalise a (2,H,W) log grid, with a tiny floor on passable cells."""
    m = np.max(log_grid[np.isfinite(log_grid)])
    p = np.exp(log_grid - m)
    p[~np.isfinite(log_grid)] = 0.0
    p = p / p.sum()
    mask = np.stack([passable, passable])
    p = np.where(mask, np.maximum(p, PROB_FLOOR), 0.0)
    return p / p.sum()
