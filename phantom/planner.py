"""Ghost Decisions: score a handful of candidate actions, pick one, keep the rest.

Every step the planner builds candidates such as
    * local scan right here
    * long-range scan right here
    * drive to cell X and local-scan there   (for the few most promising X)
    * return to base / recharge / end mission  (safety and housekeeping)

Each information-gathering candidate gets:
    eig_bits     expected information gain: how much, on average, the scan
                 would shrink uncertainty (entropy, in bits) about EVE
    battery_cost battery to drive there + do the scan
    time_cost    ticks to drive there + do the scan
    risk         0..1, how close the plan leaves WALL-E to the safety reserve
and the score is

    score = eig_bits / (battery_cost + 0.5 * time_cost) * (1 - 0.5 * risk)

i.e. "bits of information per unit of effort, discounted for risk". This is
a hand-designed heuristic utility, not machine learning. A plan is VETOED
(cannot be chosen) if afterwards WALL-E could not get back to base with the
reserve still in the tank. The rejected candidates are the "ghosts" logged
in the trace.
"""
import math

import numpy as np

from . import sensors
from .belief import entropy_bits, normalise
from .geometry import bearing_deg, direction_between, path_from_parents

MIN_USEFUL_SCORE = 0.004     # below this, information is not worth the battery
COMMIT_BONUS = 1.10          # small preference to keep the current target (prevents dithering)
TIME_WEIGHT = 0.5
LONG_SCAN_SAMPLES = 24


def cell(c):
    return f"({c[0]}, {c[1]})"


def h2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def local_scan_eig(belief, P, center, spec):
    """Exact expected information gain (bits) of a local scan at `center`."""
    inside = belief.dist_from(center) <= spec["radius"] + 1e-9
    d = spec["detect_prob"]
    h_now = entropy_bits(P)
    m = float(P[:, inside].sum())
    p_det = d * m
    # if detected: location becomes known; remaining uncertainty is EVE's status
    loc = P.sum(axis=0)
    h_det = 0.0
    if m > 1e-12:
        q = np.where(inside, loc, 0.0) / m
        pa = np.divide(P[0], loc, out=np.full_like(loc, 0.5), where=loc > 0)
        h_det = float((q * h2(pa)).sum())
    # if not detected: down-weight cells inside the radius
    post = P * np.where(inside, 1 - d, 1.0)[None, :, :]
    post = post / post.sum()
    h_miss = entropy_bits(post)
    return max(0.0, h_now - (p_det * h_det + (1 - p_det) * h_miss)), p_det


def long_scan_eig(belief, P, here, specs, rel, rng):
    """Monte Carlo estimate of a long-range scan's information gain (bits).

    Imagine EVE at a state sampled from the current belief, simulate what the
    sensors would report (with the reliabilities WALL-E currently believes),
    update a copy of the belief, measure entropy. Average over samples.
    """
    h_now = entropy_bits(P)
    flat = P.ravel()
    idx = rng.choice(flat.size, size=LONG_SCAN_SAMPLES, p=flat / flat.sum())
    logP = np.log(np.maximum(P, 1e-300))
    dist = belief.dist_from(here)
    bgrid = belief.bearing_grid_from(here)
    hs = []
    for k in idx:
        s, rem = divmod(int(k), belief.H * belief.W)
        y, x = divmod(rem, belief.W)
        status = sensors.STATUSES[s]
        d_true = math.hypot(x - here[0], y - here[1])
        total = logP.copy()
        for name, spec in specs.items():
            r = rel.get(name, 0.5)
            if spec["kind"] == "beacon":
                p_hear = float(sensors.beacon_hear_prob(d_true, status, spec))
                hear_a = np.clip(sensors.beacon_hear_prob(dist, "active", spec), 1e-9, 1)
                hear_d = np.clip(sensors.beacon_hear_prob(dist, "dormant", spec), 1e-9, 1)
                if rng.random() >= p_hear:
                    total += np.log(np.clip(np.stack([1 - hear_a, 1 - hear_d]), 1e-9, 1))
                    continue
                total += np.log(np.stack([hear_a, hear_d]))
            if rng.random() < r:
                b = bearing_deg(here, (x, y)) + rng.normal(0, spec["noise_deg"])
            else:
                b = rng.uniform(0, 360)
            delta = (bgrid - b + 180.0) % 360.0 - 180.0
            vm = sensors.von_mises_pdf(delta, sensors.kappa_from_noise(spec["noise_deg"]))
            vm[dist < 0.5] = sensors.UNIFORM_BEARING_PDF
            total += np.log(r * vm + (1 - r) * sensors.UNIFORM_BEARING_PDF)[None, :, :]
        hs.append(entropy_bits(normalise(total, belief.passable)))
    return max(0.0, h_now - float(np.mean(hs)))


def pick_scan_targets(belief, P, here, reach_cost, radius, k=4):
    """Promising places to drive to and local-scan.

    Two lists merged: cells covering the most probability, and cells with the
    best probability-per-battery. Picks are kept at least one scan radius apart.
    """
    cov = belief.scan_coverage_mass(radius, P)
    cov = np.where(np.isfinite(reach_cost) & belief.passable, cov, -1.0)
    eff = np.where(cov > 0, cov / (reach_cost + 2.0), -1.0)
    targets = []
    for grid in (cov, eff):
        order = np.argsort(grid.ravel())[::-1]
        picked = 0
        for i in order:
            if grid.ravel()[i] <= 0 or picked >= k:
                break
            c = (int(i % belief.W), int(i // belief.W))
            if c == tuple(here):
                continue
            if all(math.hypot(c[0] - t[0], c[1] - t[1]) > radius for t in targets):
                targets.append(c)
                picked += 1
    return targets


class Planner:
    def __init__(self, info, home_cost, home_parent):
        self.info = info
        self.home_cost = home_cost          # battery cost from every cell back to base
        self.home_parent = home_parent      # Dijkstra tree rooted at base
        self.capacity = info["battery_capacity"]
        self.reserve = info["reserve"]
        self.costs = info["costs"]

    def _risk(self, slack):
        return float(np.clip(1.0 - slack / (0.25 * self.capacity), 0.0, 1.0))

    def _info_candidate(self, cid, label, ctype, target, first_action, eig, bat, tim, battery, time_left,
                        extra=None):
        after = battery - bat
        home = float(self.home_cost[target[1], target[0]])
        slack = after - home - self.reserve
        cand = {"id": cid, "label": label, "type": ctype, "target": list(target),
                "first_action": first_action, "eig_bits": round(eig, 4),
                "battery_cost": round(bat, 2), "time_cost": int(tim),
                "battery_after": round(after, 2), "return_cost_after": round(home, 2),
                "slack": round(slack, 2), "risk": round(self._risk(slack), 3),
                "feasible": True, "veto": None, "score": None}
        if extra:
            cand.update(extra)
        if slack < 0:
            cand["feasible"], cand["veto"] = False, (
                f"would leave {after:.1f} battery but returning needs {home:.1f} + reserve {self.reserve:.0f}")
        elif tim > time_left:
            cand["feasible"], cand["veto"] = False, f"needs {tim} ticks, only {time_left} left"
        effort = bat + TIME_WEIGHT * tim
        cand["score"] = round(eig / effort * (1 - 0.5 * cand["risk"]), 5) if effort > 0 else 0.0
        return cand

    def plan(self, belief, P, here, battery, time_left, reach_cost, reach_parent, rel, rng,
             confirmed=None, status_scans=0, prev_target=None):
        here = tuple(here)
        base = tuple(self.info["base"])
        spec = self.info["local_scan"]
        cands = []

        # ---------- after EVE is confirmed: settle status, then end ----------
        if confirmed is not None:
            pa = belief.p_active(P)
            status_conf = max(pa, 1 - pa)
            # exact expected entropy drop of one more status reading
            acc = spec["status_accuracy"]
            p_says_active = pa * acc + (1 - pa) * (1 - acc)
            post_a = pa * acc / p_says_active
            post_d = pa * (1 - acc) / (1 - p_says_active)
            eig_status = spec["detect_prob"] * float(
                h2(np.array(pa)) - (p_says_active * h2(np.array(post_a)) + (1 - p_says_active) * h2(np.array(post_d))))
            c = self._info_candidate("status_scan", "Local scan again to read EVE's status", "local_scan",
                                     here, {"type": "local_scan"}, eig_status,
                                     self.costs["local_scan"], 1, battery, time_left)
            within = math.hypot(here[0] - confirmed[0], here[1] - confirmed[1]) <= spec["radius"]
            if status_conf >= 0.8 or status_scans >= 2 or not within:
                c["feasible"], c["veto"] = False, ("status already clear" if status_conf >= 0.8
                                                   else "status scan limit reached")
            cands.append(c)
            end = {"id": "end", "label": "End mission and report", "type": "end_mission",
                   "first_action": {"type": "end_mission"}, "feasible": True, "veto": None,
                   "score": None, "eig_bits": 0.0}
            chosen = c if c["feasible"] else end
            reason = ("EVE confirmed; one more local scan should clarify her status "
                      f"(currently {status_conf:.0%} sure)") if chosen is c else (
                      f"EVE confirmed at ({confirmed[0]}, {confirmed[1]}); status {status_conf:.0%} certain, ending mission")
            cands.append(end)
            return self._finish(cands, chosen, reason, forced="mission goal reached" if chosen is end else None)

        # ---------- information candidates ----------
        eig_here, pdet = local_scan_eig(belief, P, here, spec)
        cands.append(self._info_candidate("scan_here", f"Local scan here {cell(here)}", "local_scan", here,
                                          {"type": "local_scan"}, eig_here, self.costs["local_scan"], 1,
                                          battery, time_left, {"p_detect": round(pdet, 4)}))
        eig_long = long_scan_eig(belief, P, here, self.info["sources"], rel, rng)
        cands.append(self._info_candidate("long_scan", f"Long-range scan from {cell(here)}", "long_scan", here,
                                          {"type": "long_scan"}, eig_long, self.costs["long_scan"], 1,
                                          battery, time_left))
        for t in pick_scan_targets(belief, P, here, reach_cost, spec["radius"]):
            path = path_from_parents(reach_parent, here, t)
            if not path or len(path) < 2:
                continue
            eig_t, pdet_t = local_scan_eig(belief, P, t, spec)
            bat = float(reach_cost[t[1], t[0]]) + self.costs["local_scan"]
            c = self._info_candidate(f"goto_{t[0]}_{t[1]}", f"Drive to {cell(t)} and local scan", "goto_scan",
                                     t, {"type": "move", "direction": direction_between(path[0], path[1])},
                                     eig_t, bat, len(path), battery, time_left,
                                     {"path_len": len(path) - 1, "p_detect": round(pdet_t, 4),
                                      "next_cell": list(path[1])})
            cands.append(c)

        for c in cands:
            if prev_target is not None and c["feasible"] and c["target"] == list(prev_target) \
                    and c["type"] == "goto_scan":
                c["score"] = round(c["score"] * COMMIT_BONUS, 5)
                c["commit_bonus"] = True

        # ---------- safety & housekeeping candidates ----------
        home_cost_here = float(self.home_cost[here[1], here[0]])
        home_steps = self._home_steps(here)
        ret = {"id": "return_base", "label": "Return to base", "type": "return_base",
               "target": list(base), "eig_bits": 0.0, "battery_cost": round(home_cost_here, 2),
               "time_cost": home_steps, "feasible": here != base, "veto": None if here != base else "already at base",
               "score": None}
        if here != base:
            nxt = path_from_parents(self.home_parent, base, here)   # path base -> here
            step_to = nxt[-2] if nxt and len(nxt) >= 2 else base
            ret["first_action"] = {"type": "move", "direction": direction_between(here, step_to)}
        cands.append(ret)
        rech = {"id": "recharge", "label": "Recharge at base", "type": "recharge", "target": list(base),
                "eig_bits": 0.0, "battery_cost": -self.costs["recharge_rate"], "time_cost": 1,
                "first_action": {"type": "recharge"},
                "feasible": here == base and battery < self.capacity - 1e-6, "veto": None, "score": None}
        if not rech["feasible"]:
            rech["veto"] = "not at base" if here != base else "battery full"
        cands.append(rech)
        end = {"id": "end", "label": "End mission", "type": "end_mission", "eig_bits": 0.0,
               "first_action": {"type": "end_mission"}, "feasible": True, "veto": None, "score": None}
        cands.append(end)

        info_c = [c for c in cands if c["type"] in ("local_scan", "long_scan", "goto_scan")]
        feasible = [c for c in info_c if c["feasible"] and c["score"] >= MIN_USEFUL_SCORE]
        best_any = max(info_c, key=lambda c: c["score"] or 0)

        # rule 1: turn back if time is about to run out
        if here != base and time_left <= home_steps + 2:
            return self._finish(cands, ret, f"only {time_left} ticks left and base is {home_steps} moves away: "
                                            "turning back", forced="time safety rule")
        # rule 2: opportunity cost. If a plan blocked only by battery is clearly better
        # than anything affordable, recharge for it instead of spending on scraps.
        # only plans that would fit a FULL battery count (otherwise recharging cannot help)
        blocked = [c for c in info_c if not c["feasible"] and "returning needs" in (c["veto"] or "")
                   and c["score"] >= MIN_USEFUL_SCORE and self._fits_full_battery(c, here)]
        best_blocked = max(blocked, key=lambda c: c["score"]) if blocked else None
        best_feasible = max(feasible, key=lambda c: c["score"]) if feasible else None
        if best_blocked is not None and time_left > home_steps + 5:
            if here == base and rech["feasible"] and (best_feasible is None
                                                       or best_blocked["score"] > best_feasible["score"]):
                return self._finish(cands, rech, f"at base; recharging so '{best_blocked['label']}' "
                                                 f"(score {best_blocked['score']:.4f}) becomes safe",
                                    forced="recharge rule")
            if here != base and best_feasible is not None \
                    and best_blocked["score"] > 2.0 * best_feasible["score"]:
                return self._finish(cands, ret, f"'{best_blocked['label']}' scores {best_blocked['score']:.4f} "
                                                f"but needs more battery; affordable options score at most "
                                                f"{best_feasible['score']:.4f}: returning to recharge",
                                    forced="battery safety rule")
        if feasible:
            best = max(feasible, key=lambda c: c["score"])
            ranked = sorted(feasible, key=lambda c: -c["score"])
            reason = (f"{best['label']}: expects {best['eig_bits']:.3f} bits of information for "
                      f"{best['battery_cost']:.1f} battery and {best['time_cost']} ticks "
                      f"(score {best['score']:.4f})")
            if len(ranked) > 1:
                reason += f"; runner-up '{ranked[1]['label']}' scored {ranked[1]['score']:.4f}"
            return self._finish(cands, best, reason)
        # rule 3: nothing useful is safe from here
        if here != base:
            why = (f"best option '{best_any['label']}' is vetoed ({best_any['veto']})" if best_any["veto"]
                   else "no remaining option is worth its battery")
            return self._finish(cands, ret, f"{why}: returning to base", forced="battery safety rule")
        # rule 4: at base. Recharge if that would unlock a worthwhile plan.
        blocked_by_battery = [c for c in info_c if not c["feasible"] and "battery" in (c["veto"] or "")
                              and c["score"] >= MIN_USEFUL_SCORE and self._fits_full_battery(c, here)]
        if rech["feasible"] and blocked_by_battery and time_left > 5:
            b = max(blocked_by_battery, key=lambda c: c["score"])
            return self._finish(cands, rech, f"at base; recharging so '{b['label']}' becomes safe",
                                forced="recharge rule")
        return self._finish(cands, end, "at base and no remaining option is both safe and worth its cost",
                            forced="nothing useful left")

    def _fits_full_battery(self, c, here):
        """Could this plan be done safely if WALL-E started it from base with a full battery?"""
        to_start = float(self.home_cost[here[1], here[0]])      # base -> here, roughly symmetric
        return to_start + c["battery_cost"] + c["return_cost_after"] + self.reserve <= self.capacity

    def _home_steps(self, here):
        path = path_from_parents(self.home_parent, tuple(self.info["base"]), here)
        return len(path) - 1 if path else 0

    @staticmethod
    def _finish(cands, chosen, reason, forced=None):
        for c in cands:
            c["chosen"] = c is chosen
        return {"candidates": cands, "chosen_id": chosen["id"], "chosen_label": chosen["label"],
                "action": chosen["first_action"], "reason": reason, "forced_by": forced,
                "target": chosen.get("target")}
