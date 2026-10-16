"""Simple baseline agents for comparison.

Both use the SAME safety rules as PHANTOM (return home before the battery
gets too low, recharge, turn back when time runs out) so the comparison is
about search strategy, not about who is allowed to cheat on safety. They
also keep a belief grid (fixed trust 0.8 for every source) purely so the
final report has an estimate; their movement ignores it.

* SweepAgent ("naive follower"): checks scan spots in the direction of the
  initial signal first, then sweeps the rest of the map in a lawnmower
  pattern, local-scanning at each spot.
* RandomAgent: random legal moves, a local scan every 4 moves.
"""
import math
import random

import numpy as np

from .belief import BeliefModel
from .geometry import (DIRECTIONS, ROCK, angle_diff_deg, bearing_deg, dijkstra,
                       direction_between, path_from_parents)

FIXED_REL = 0.8


class _SimpleAgent:
    kind = "simple"

    def __init__(self, info, seed=0):
        self.info = info
        self.terrain = np.array(info["terrain"])
        self.base = tuple(info["base"])
        self.position = tuple(info["start"])
        self.battery = float(info["start_battery"])
        self.time_left = int(info["time_budget"])
        self.costs = info["costs"]
        self.rng = random.Random(seed * 13 + 5)
        self.home_cost, self.home_parent = dijkstra(self.terrain, self.base, self.costs)
        self.belief = BeliefModel(info)
        beacon = next((s for s in info["sources"].values() if s["kind"] == "beacon"), None)
        self.belief.add_initial_signal(info["initial_signal"], 0.7, beacon)
        self.rel = {s: FIXED_REL for s in info["sources"]}
        self.belief.update(self.rel)
        self.confirmed = None
        self.mode = "search"

    def _home_step(self):
        path = path_from_parents(self.home_parent, self.base, self.position)
        return {"type": "move", "direction": direction_between(self.position, path[-2])}

    def _home_steps(self):
        path = path_from_parents(self.home_parent, self.base, self.position)
        return len(path) - 1

    def _safe_after(self, cost, cell):
        return self.battery - cost - self.home_cost[cell[1], cell[0]] >= self.info["reserve"]

    def _decision(self, action, label, reason, forced=None):
        return action, {"candidates": [{"id": "rule", "label": label, "type": "rule", "chosen": True,
                                        "feasible": True, "veto": None, "score": None, "eig_bits": None}],
                        "chosen_id": "rule", "chosen_label": label, "action": action,
                        "reason": reason, "forced_by": forced, "plain_english": reason}

    def _safety(self):
        """Shared safety rules. Returns (action, decision) or None."""
        if self.confirmed:
            return self._decision({"type": "end_mission"}, "End mission", "EVE detected by local scan",
                                  "mission goal reached")
        if self.position != self.base and self.time_left <= self._home_steps() + 2:
            return self._decision(self._home_step(), "Return to base", "time nearly out", "time safety rule")
        if self.mode == "recharge":
            if self.position != self.base:
                return self._decision(self._home_step(), "Return to base", "battery low", "battery safety rule")
            if self.battery < self.info["battery_capacity"] - 1e-6 and self.time_left > 5:
                return self._decision({"type": "recharge"}, "Recharge", "recharging to full", "recharge rule")
            self.mode = "search"
        return None

    def observe(self, obs):
        self.position = tuple(obs.position)
        self.battery = obs.battery
        self.time_left = obs.time_left
        h0 = self.belief.entropy()
        for r in obs.readings:
            spec = self.info["sources"][r["source"]]
            if r["kind"] == "beacon" and not r.get("heard"):
                self.belief.add_beacon_silence(obs.step, r, spec)
            else:
                self.belief.add_bearing(obs.step, r, spec)
        if obs.scan:
            self.belief.add_scan(obs.step, obs.scan, self.info["local_scan"])
            if obs.scan["eve_detected"]:
                self.confirmed = tuple(obs.scan["eve_position"])
        for clue in obs.clues:
            if clue.get("first_time", True):
                self.belief.add_clue(obs.step, clue, self.info["clue_specs"])
        self.belief.update(self.rel)
        return {"new_evidence": [], "entropy_before": round(h0, 4),
                "entropy_after": round(self.belief.entropy(), 4),
                "info_gained_bits": round(h0 - self.belief.entropy(), 4), "dna_judgments": []}

    def snapshot(self):
        return {"location": [round(float(v), 6) for v in self.belief.location().ravel()],
                "p_active": round(self.belief.p_active(), 4),
                "entropy_bits": round(self.belief.entropy(), 4),
                "top_cells": self.belief.top_cells(5),
                "confirmed": list(self.confirmed) if self.confirmed else None,
                "dna": {}, "reliability_used": dict(self.rel), "initial_signal_trust": 0.7}


class SweepAgent(_SimpleAgent):
    kind = "sweep"

    def __init__(self, info, seed=0):
        super().__init__(info, seed)
        r = info["local_scan"]["radius"]
        step = int(2 * r + 1)
        W, H = info["width"], info["height"]
        spots = []
        for y in range(int(r), H, step):
            for x in range(int(r), W, step):
                c = self._nearest_open((x, y))
                if c and c not in spots:
                    spots.append(c)
        b0 = info["initial_signal"].get("bearing_deg")
        origin = tuple(info["initial_signal"]["origin"])
        if b0 is not None:
            cone = [s for s in spots if s != origin and abs(angle_diff_deg(bearing_deg(origin, s), b0)) <= 30]
            cone.sort(key=lambda s: math.hypot(s[0] - origin[0], s[1] - origin[1]))
            rest = [s for s in spots if s not in cone]
        else:
            cone, rest = [], spots
        ordered = list(cone)
        cur = ordered[-1] if ordered else self.base
        while rest:                                    # nearest-neighbour order for the sweep
            nxt = min(rest, key=lambda s: math.hypot(s[0] - cur[0], s[1] - cur[1]))
            ordered.append(nxt)
            rest.remove(nxt)
            cur = nxt
        self.spots = ordered
        self.idx = 0

    def _nearest_open(self, c):
        best = None
        for y in range(self.terrain.shape[0]):
            for x in range(self.terrain.shape[1]):
                if self.terrain[y, x] != ROCK and np.isfinite(self.home_cost[y, x]):
                    d = math.hypot(x - c[0], y - c[1])
                    if d <= 2 and (best is None or d < best[0]):
                        best = (d, (x, y))
        return best[1] if best else None

    def decide(self):
        s = self._safety()
        if s:
            return s
        while self.idx < len(self.spots):
            target = self.spots[self.idx]
            reach, parent = dijkstra(self.terrain, self.position, self.costs)
            if not np.isfinite(reach[target[1], target[0]]):
                self.idx += 1
                continue
            if self.position == target:
                if self._safe_after(self.costs["local_scan"], target):
                    self.idx += 1
                    return self._decision({"type": "local_scan"}, "Local scan", f"sweep spot {list(target)}")
            else:
                cost = reach[target[1], target[0]] + self.costs["local_scan"]
                if self._safe_after(cost, target):
                    path = path_from_parents(parent, self.position, target)
                    return self._decision({"type": "move", "direction": direction_between(path[0], path[1])},
                                          "Drive to sweep spot", f"heading to sweep spot {list(target)}")
            # not safe: go recharge, unless already full at base (then this spot is out of reach)
            if self.position == self.base and self.battery >= self.info["battery_capacity"] - 1e-6:
                self.idx += 1
                continue
            self.mode = "recharge"
            return self._safety() or self._decision({"type": "end_mission"}, "End", "cannot continue safely")
        if self.position != self.base:
            return self._decision(self._home_step(), "Return to base", "sweep finished")
        return self._decision({"type": "end_mission"}, "End mission", "sweep finished, EVE not found",
                              "nothing useful left")


class RandomAgent(_SimpleAgent):
    kind = "random"

    def __init__(self, info, seed=0):
        super().__init__(info, seed)
        self.moves = 0

    def decide(self):
        s = self._safety()
        if s:
            return s
        if self.moves >= 4 and self._safe_after(self.costs["local_scan"], self.position):
            self.moves = 0
            return self._decision({"type": "local_scan"}, "Local scan", "random walk: periodic scan")
        x, y = self.position
        opts = []
        for d, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.terrain.shape[1] and 0 <= ny < self.terrain.shape[0] and self.terrain[ny, nx] != ROCK:
                c = self.costs["move_rough"] if self.terrain[ny, nx] == 1 else self.costs["move_plain"]
                if self._safe_after(c, (nx, ny)):
                    opts.append(d)
        if not opts:
            if self.position == self.base:
                if self.battery < self.info["battery_capacity"] - 1e-6 and self.time_left > 5:
                    return self._decision({"type": "recharge"}, "Recharge", "battery low", "recharge rule")
                return self._decision({"type": "end_mission"}, "End mission", "no safe moves", "nothing useful left")
            self.mode = "recharge"
            return self._safety()
        self.moves += 1
        return self._decision({"type": "move", "direction": self.rng.choice(opts)}, "Random move", "random walk")
