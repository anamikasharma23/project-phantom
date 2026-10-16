"""The simulated world (the "environment").

This is the ONLY module that knows where EVE is. The agent talks to it
through two doors:

* `public_info()`   - what WALL-E knows before the mission (map, costs, specs)
* `step(action)`    - do one action, get back an `Observation`

Neither door ever contains EVE's position, her status, clue positions, or the
true reliability of any signal source. `evaluation()` exposes the truth, but
it is only called by the mission runner AFTER the mission ends, to score it.
"""
import copy
import math
import random
from dataclasses import dataclass, field

import numpy as np

from . import sensors
from .config import parse_time
from .geometry import (DIRECTIONS, ROCK, bearing_deg, dijkstra, distance,
                       move_cost)

ACTION_TYPES = ("move", "long_scan", "local_scan", "recharge", "end_mission")


@dataclass(frozen=True)
class Action:
    type: str
    direction: str = None       # only for "move": N, E, S or W

    def to_dict(self):
        d = {"type": self.type}
        if self.direction:
            d["direction"] = self.direction
        return d


@dataclass
class Observation:
    """Everything WALL-E perceives after one action. No hidden truth here."""
    step: int
    action: dict
    ok: bool
    error: str = None
    position: tuple = None
    battery: float = 0.0
    battery_delta: float = 0.0
    time_used: int = 0
    time_left: int = 0
    readings: list = field(default_factory=list)     # long-range signal readings
    scan: dict = None                                # local scan result
    clues: list = field(default_factory=list)        # clues found by this scan
    mission_over: bool = False
    end_reason: str = None

    def to_dict(self):
        return {
            "step": self.step, "action": self.action, "ok": self.ok, "error": self.error,
            "position": list(self.position), "battery": round(self.battery, 3),
            "battery_delta": round(self.battery_delta, 3), "time_used": self.time_used,
            "time_left": self.time_left, "readings": self.readings, "scan": self.scan,
            "clues": self.clues, "mission_over": self.mission_over, "end_reason": self.end_reason,
        }


# Fields the agent must never receive. Used by tests as a tripwire.
HIDDEN_KEYS = {"true_reliability", "decoy", "eve", "clue_positions", "eve_position_truth"}


class Environment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = random.Random(cfg["mission"]["seed"] * 1009 + 7)
        m = cfg["map"]
        self.terrain = np.array(m["grid"], dtype=np.int8)
        self.height, self.width = self.terrain.shape
        self.base = tuple(m["base"])
        self.costs = cfg["resources"]["costs"]
        self.capacity = float(cfg["resources"]["battery_capacity"])
        self.time_budget = int(cfg["resources"]["time_budget"])

        # ---- hidden truth --------------------------------------------------
        self._eve = tuple(cfg["eve"]["position"])
        self._status = cfg["eve"]["status"]
        self._sources = copy.deepcopy(cfg["sensors"]["sources"])
        self._clues = []
        for c in cfg["clues"]["items"]:
            self._clues.append(self._make_clue(c))

        # ---- state the agent can also see -----------------------------------
        self.position = self.base
        self.battery = float(cfg["resources"]["start_battery"])
        self.time_used = 0
        self.step_count = 0
        self.over = False
        self.end_reason = None
        self.found_clue_ids = set()
        self.eve_detected = False

    # ------------------------------------------------------------------
    def _make_clue(self, c):
        """A clue's reading is fixed when the world is built (re-scans agree)."""
        pos = tuple(c["position"])
        clue = {"id": f"clue-{pos[0]}-{pos[1]}", "type": c["type"], "position": list(pos)}
        if c["type"] == "track":
            clue["bearing_deg"] = round((bearing_deg(pos, self._eve)
                                         + self.rng.gauss(0, sensors.CLUE_TRACK_NOISE_DEG)) % 360, 1)
        else:  # residue: status hint + proximity hint, each right 85% of the time
            fresh = (self._status == "active")
            if self.rng.random() > sensors.RESIDUE_ACCURACY:
                fresh = not fresh
            near = distance(pos, self._eve) <= sensors.RESIDUE_NEAR_RADIUS
            if self.rng.random() > sensors.RESIDUE_ACCURACY:
                near = not near
            clue["freshness"] = "fresh" if fresh else "cold"
            clue["proximity"] = "near" if near else "far"
        return clue

    # ------------------------------------------------------------------
    def public_info(self):
        """What WALL-E knows before starting. Hidden truth is stripped out."""
        specs = {}
        for name, s in self._sources.items():
            spec = {k: v for k, v in s.items() if k not in ("true_reliability", "decoy")}
            specs[name] = spec
        sig = dict(self.cfg["initial_signal"])
        return {
            "mission_name": self.cfg["mission"]["name"],
            "seed": self.cfg["mission"]["seed"],
            "start_time": self.cfg["mission"]["start_time"],
            "width": self.width, "height": self.height,
            "terrain": self.terrain.tolist(),
            "base": list(self.base), "start": list(self.base),
            "battery_capacity": self.capacity,
            "start_battery": self.battery,
            "time_budget": self.time_budget,
            "reserve": float(self.cfg["resources"]["reserve"]),
            "costs": dict(self.costs),
            "local_scan": dict(self.cfg["sensors"]["local_scan"]),
            "sources": specs,
            "initial_signal": sig,
            "clue_specs": {"track_noise_deg": sensors.CLUE_TRACK_NOISE_DEG,
                           "track_reliability": sensors.CLUE_TRACK_RELIABILITY,
                           "residue_accuracy": sensors.RESIDUE_ACCURACY,
                           "residue_near_radius": sensors.RESIDUE_NEAR_RADIUS},
            "agent": dict(self.cfg["agent"]),
        }

    # ------------------------------------------------------------------
    def legal_actions(self):
        """List of actions the environment will accept right now."""
        if self.over:
            return []
        acts = []
        x, y = self.position
        for d, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.terrain[ny, nx] != ROCK:
                if self.battery >= move_cost(self.terrain[ny, nx], self.costs):
                    acts.append(Action("move", d))
        if self.battery >= self.costs["long_scan"]:
            acts.append(Action("long_scan"))
        if self.battery >= self.costs["local_scan"]:
            acts.append(Action("local_scan"))
        if self.position == self.base and self.battery < self.capacity:
            acts.append(Action("recharge"))
        acts.append(Action("end_mission"))
        return acts

    # ------------------------------------------------------------------
    def step(self, action):
        if self.over:
            raise RuntimeError("mission is already over")
        if not isinstance(action, Action) or action.type not in ACTION_TYPES:
            raise ValueError(f"unknown action: {action!r}")
        self.step_count += 1
        before = self.battery
        obs = Observation(step=self.step_count, action=action.to_dict(), ok=True)

        if action.type == "move":
            self._do_move(action, obs)
        elif action.type == "long_scan":
            self._do_long_scan(obs)
        elif action.type == "local_scan":
            self._do_local_scan(obs)
        elif action.type == "recharge":
            self._do_recharge(obs)
        elif action.type == "end_mission":
            self.over, self.end_reason = True, "agent_ended"

        self.time_used += 1                              # every action takes one tick
        if not self.over:
            # stranded = away from base without enough charge for even one move
            if self.position != self.base and self.battery + 1e-9 < self.costs["move_plain"]:
                self.over, self.end_reason = True, "stranded"
            elif self.time_used >= self.time_budget:
                self.over, self.end_reason = True, "time_exhausted"

        obs.position = self.position
        obs.battery = self.battery
        obs.battery_delta = self.battery - before
        obs.time_used = self.time_used
        obs.time_left = self.time_budget - self.time_used
        obs.mission_over = self.over
        obs.end_reason = self.end_reason
        return obs

    def _spend(self, cost, obs):
        if self.battery + 1e-9 < cost:
            obs.ok, obs.error = False, f"not enough battery ({self.battery:.1f} < {cost:.1f})"
            return False
        self.battery = max(0.0, self.battery - cost)
        return True

    def _do_move(self, action, obs):
        if action.direction not in DIRECTIONS:
            obs.ok, obs.error = False, f"invalid direction {action.direction!r}"
            return
        dx, dy = DIRECTIONS[action.direction]
        nx, ny = self.position[0] + dx, self.position[1] + dy
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            obs.ok, obs.error = False, "blocked: edge of map"
            return
        cost = move_cost(self.terrain[ny, nx], self.costs)
        if cost is None:
            obs.ok, obs.error = False, "blocked: rock"
            return
        if self._spend(cost, obs):
            self.position = (nx, ny)

    def _do_long_scan(self, obs):
        if not self._spend(self.costs["long_scan"], obs):
            return
        here = self.position
        d_eve = distance(here, self._eve)
        true_b = bearing_deg(here, self._eve)
        for name, s in self._sources.items():
            reading = {"source": name, "kind": s["kind"], "origin": list(here)}
            if s["kind"] == "beacon":
                p_hear = float(sensors.beacon_hear_prob(d_eve, self._status, s))
                if self.rng.random() >= p_hear:
                    reading["heard"] = False
                    obs.readings.append(reading)
                    continue
                reading["heard"] = True
                reading["strength"] = round(min(1.0, max(0.0, float(sensors.beacon_strength_mean(d_eve, s))
                                                         + self.rng.gauss(0, sensors.STRENGTH_NOISE))), 3)
            genuine = self.rng.random() < s["true_reliability"]
            if genuine:
                b = true_b + self.rng.gauss(0, s["noise_deg"])
            elif "decoy" in s:
                b = bearing_deg(here, s["decoy"]) + self.rng.gauss(0, s["noise_deg"])
            else:
                b = self.rng.uniform(0, 360)
            reading["bearing_deg"] = round(b % 360, 1)
            obs.readings.append(reading)

    def _do_local_scan(self, obs):
        if not self._spend(self.costs["local_scan"], obs):
            return
        spec = self.cfg["sensors"]["local_scan"]
        r = spec["radius"]
        here = self.position
        scanned = [[x, y] for y in range(self.height) for x in range(self.width)
                   if distance((x, y), here) <= r + 1e-9]
        result = {"center": list(here), "radius": r, "cells": scanned, "eve_detected": False}
        if distance(here, self._eve) <= r + 1e-9 and self.rng.random() < spec["detect_prob"]:
            result["eve_detected"] = True
            result["eve_position"] = list(self._eve)       # the agent SAW her: legit evidence
            correct = self.rng.random() < spec["status_accuracy"]
            other = "dormant" if self._status == "active" else "active"
            result["status_reading"] = self._status if correct else other
            self.eve_detected = True
        obs.scan = result
        for clue in self._clues:
            if distance(tuple(clue["position"]), here) <= r + 1e-9:
                obs.clues.append(dict(clue, first_time=clue["id"] not in self.found_clue_ids))
                self.found_clue_ids.add(clue["id"])

    def _do_recharge(self, obs):
        if self.position != self.base:
            obs.ok, obs.error = False, "can only recharge at base"
            return
        self.battery = min(self.capacity, self.battery + self.costs["recharge_rate"])

    # ------------------------------------------------------------------
    def evaluation(self, estimate=None):
        """Ground truth for scoring. Never passed to the agent."""
        out = {"eve_position": list(self._eve), "eve_status": self._status,
               "clue_positions": [c["position"] for c in self._clues],
               "source_true_reliability": {k: v["true_reliability"] for k, v in self._sources.items()},
               "decoys": {k: v["decoy"] for k, v in self._sources.items() if "decoy" in v},
               "eve_detected_by_scan": self.eve_detected}
        if estimate is not None:
            out["estimate_error_cells"] = round(distance(tuple(estimate), self._eve), 3)
        dist_home, _ = dijkstra(self.terrain, self.base, self.costs)
        out["return_cost_from_final_position"] = float(dist_home[self.position[1], self.position[0]])
        return out
