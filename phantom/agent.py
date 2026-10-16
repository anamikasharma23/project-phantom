"""PHANTOM, the WALL-E agent.

The agent only ever sees:
  * `info`  - the public mission briefing (from Environment.public_info)
  * `Observation` objects returned after each action
It never receives the Environment object itself, so it cannot read EVE's
true position even by accident (a test checks this module never imports
the environment).

Loop:   decide() -> environment.step(action) -> observe(observation) -> ...
"""
import math

import numpy as np

from .belief import BeliefModel
from .config import parse_time
from .geometry import dijkstra
from .planner import Planner
from .signal_dna import SignalDNA

ACTION_WORDS = {"N": "north", "E": "east", "S": "south", "W": "west"}


def cell(c):
    return f"({c[0]}, {c[1]})"


class PhantomAgent:
    kind = "phantom"

    def __init__(self, info, use_dna=None, seed=0):
        self.info = info
        if use_dna is None:
            use_dna = bool(info["agent"].get("use_signal_dna", True))
        self.use_dna = use_dna
        if not use_dna:
            self.kind = "phantom_no_dna"
        self.terrain = np.array(info["terrain"])
        self.position = tuple(info["start"])
        self.battery = float(info["start_battery"])
        self.time_left = int(info["time_budget"])
        self.rng = np.random.default_rng(seed * 7 + 11)
        self.belief = BeliefModel(info)
        self.dna = SignalDNA(info["sources"], prior=info["agent"]["dna_prior"], enabled=use_dna)
        home_cost, home_parent = dijkstra(self.terrain, tuple(info["base"]), info["costs"])
        self.planner = Planner(info, home_cost, home_parent)
        self.confirmed = None          # EVE's cell once a local scan sees her
        self.status_scans = 0
        self.prev_target = None
        self.step = 0
        self.initial_trust, self.signal_age_min = self._initial_trust()
        beacon = next((s for s in info["sources"].values() if s["kind"] == "beacon"), None)
        self.belief.add_initial_signal(info["initial_signal"], self.initial_trust, beacon)
        self.belief.update(self.dna.reliabilities())

    def _initial_trust(self):
        """Older signals are trusted less: trust halves every `half_life` minutes."""
        sig = self.info["initial_signal"]
        a = self.info["agent"]
        age = (parse_time(self.info["start_time"], "start") - parse_time(sig["timestamp"], "ts")).total_seconds() / 60
        trust = a["initial_signal_trust"] * 0.5 ** (age / a["staleness_half_life_min"])
        return float(trust), float(age)

    # ------------------------------------------------------------------
    def decide(self):
        """Score candidates and return (action dict, decision record)."""
        reach_cost, reach_parent = dijkstra(self.terrain, self.position, self.info["costs"])
        decision = self.planner.plan(self.belief, self.belief.P, self.position, self.battery,
                                     self.time_left, reach_cost, reach_parent,
                                     self.dna.reliabilities(), self.rng, confirmed=self.confirmed,
                                     status_scans=self.status_scans, prev_target=self.prev_target)
        if decision["chosen_id"].startswith("goto_"):
            self.prev_target = tuple(decision["target"])
        elif decision["chosen_id"] in ("scan_here", "return_base", "recharge"):
            self.prev_target = None
        decision["plain_english"] = self.explain(decision)
        return decision["action"], decision

    def explain(self, d):
        """One beginner-friendly sentence built from the real decision numbers."""
        chosen = next(c for c in d["candidates"] if c["chosen"])
        act = d["action"]
        if chosen["type"] == "goto_scan":
            cov = chosen.get("p_detect", 0) / max(self.info["local_scan"]["detect_prob"], 1e-9)
            return (f"Driving {ACTION_WORDS[act['direction']]} toward {cell(chosen['target'])}, where there is a "
                    f"{cov:.0%} chance EVE is within scan range.")
        if chosen["id"] == "scan_here":
            cov = chosen.get("p_detect", 0) / max(self.info["local_scan"]["detect_prob"], 1e-9)
            return f"Scanning the area around WALL-E ({cov:.0%} chance EVE is within range)."
        if chosen["id"] == "long_scan":
            return "Listening for long-range signals to get new directions toward EVE."
        if chosen["id"] == "status_scan":
            return "EVE is found. Scanning her again to work out whether she is active or dormant."
        if chosen["id"] == "return_base":
            return f"Heading back to base: {d['reason']}."
        if chosen["id"] == "recharge":
            return "Recharging at base so the next search trip is safe."
        if self.confirmed:
            return "EVE has been located. Ending the mission and writing the report."
        return "No remaining option is both safe and worth the battery, so the mission ends here."

    # ------------------------------------------------------------------
    def observe(self, obs):
        """Fold an Observation into the belief and Signal DNA. Returns a trace record."""
        self.step = obs.step
        self.position = tuple(obs.position)
        self.battery = obs.battery
        self.time_left = obs.time_left
        h_before = self.belief.entropy()
        top_before = self.belief.top_cells(1)[0]
        new_ev = []
        info = self.info
        for r in obs.readings:
            spec = info["sources"][r["source"]]
            if r["kind"] == "beacon" and not r.get("heard"):
                new_ev.append(self.belief.add_beacon_silence(obs.step, r, spec))
            else:
                ev = self.belief.add_bearing(obs.step, r, spec)
                self.dna.add_pending(ev)
                new_ev.append(ev)
        if obs.scan:
            new_ev.append(self.belief.add_scan(obs.step, obs.scan, info["local_scan"]))
            if obs.scan["eve_detected"]:
                if self.confirmed is not None:
                    self.status_scans += 1
                    new_ev.append(self.belief.evidence[-1])      # the status reading
                else:
                    new_ev.append(self.belief.evidence[-1])
                self.confirmed = tuple(obs.scan["eve_position"])
        for clue in obs.clues:
            if clue.get("first_time", True):
                new_ev.append(self.belief.add_clue(obs.step, clue, info["clue_specs"]))

        judgments = []
        if obs.scan and obs.scan["eve_detected"] and not self.dna.anchored:
            judgments = self.dna.anchor_update(self.belief.evidence, self.confirmed)
        elif new_ev or self.dna.pending:
            # first update with current trust, then judge pending readings, then re-update
            self.belief.update(self.dna.reliabilities())
            judgments = self.dna.soft_update(self.belief, self.dna.reliabilities())
        self.belief.update(self.dna.reliabilities())

        h_after = self.belief.entropy()
        top_after = self.belief.top_cells(1)[0]
        return {
            "new_evidence": [dict(e.summary, id=e.id, kind=e.kind) for e in new_ev],
            "entropy_before": round(h_before, 4), "entropy_after": round(h_after, 4),
            "info_gained_bits": round(h_before - h_after, 4),
            "top_cell_before": top_before, "top_cell_after": top_after,
            "dna_judgments": judgments,
        }

    # ------------------------------------------------------------------
    def source_view(self):
        """Per signal source: where it points on its own.

        solo_top = most likely cell using ONLY this source's readings (plus the flat prior),
        weighted by the reliability WALL-E currently believes. The UI compares this with
        the overall estimate to show which sources agree. Nothing here uses hidden truth.
        """
        rel = self.dna.reliabilities()
        names = list(self.info["sources"])
        out = {}
        for s in names:
            n = sum(1 for e in self.belief.evidence if e.source == s and e.kind == "bearing")
            if not n:
                out[s] = {"readings": 0, "solo_top": None}
                continue
            others = set(names) - {s}
            solo = self.belief.compute(rel, exclude_sources=others | {"initial_signal", "local_scan", "clue"})
            out[s] = {"readings": n, "solo_top": self.belief.top_cells(1, solo)[0]}
        return out

    def snapshot(self):
        """Belief + DNA state for the trace (what the UI replays)."""
        loc = self.belief.location()
        return {
            "location": [round(float(v), 6) for v in loc.ravel()],
            "p_active": round(self.belief.p_active(), 4),
            "entropy_bits": round(self.belief.entropy(), 4),
            "top_cells": self.belief.top_cells(5),
            "confirmed": list(self.confirmed) if self.confirmed else None,
            "dna": self.dna.snapshot(),
            "reliability_used": {k: round(v, 4) for k, v in self.dna.reliabilities().items()},
            "initial_signal_trust": round(self.initial_trust, 4),
            "source_view": self.source_view(),
        }
