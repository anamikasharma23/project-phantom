"""Final mission report, built ONLY from the trace.

Because the report reads nothing but the trace, anyone can re-generate it
from a saved trace file and check that every claim is backed by a logged
event. Ground truth (the `evaluation` block) is never used here.
"""
import math

import numpy as np

from .geometry import ROCK, dijkstra, path_from_parents


def cell(c):
    return f"({c[0]}, {c[1]})"


def build_report(trace):
    pub = trace["public"]
    events = trace["events"]
    last = events[-1]
    snap = last["belief"]
    W, H = pub["width"], pub["height"]
    loc = np.array(snap["location"]).reshape(H, W)
    top = snap["top_cells"][0]
    est = tuple(top["cell"])
    radius = pub["local_scan"]["radius"]
    ys, xs = np.mgrid[0:H, 0:W]
    near_mass = float(loc[np.hypot(xs - est[0], ys - est[1]) <= radius + 1e-9].sum())
    p_active = snap["p_active"]
    status = "active" if p_active >= 0.5 else "dormant"
    status_conf = max(p_active, 1 - p_active)
    confirmed = snap["confirmed"]

    # ---- evidence gathered (from logged observations) ----
    clues, detections, per_source, long_scans, local_scans = [], [], {}, 0, 0
    moves, recharged, spent = 0, 0.0, 0.0
    for e in events[1:]:
        obs = e["observation"]
        a = obs["action"]["type"]
        if obs["battery_delta"] < 0:
            spent += -obs["battery_delta"]
        else:
            recharged += obs["battery_delta"]
        if not obs["ok"]:
            continue
        if a == "move":
            moves += 1
        elif a == "long_scan":
            long_scans += 1
        elif a == "local_scan":
            local_scans += 1
        for r in obs["readings"]:
            s = per_source.setdefault(r["source"], {"readings": 0, "heard": 0, "silent": 0})
            if r["kind"] == "beacon" and not r.get("heard"):
                s["silent"] += 1
            else:
                s["readings"] += 1
                s["heard"] += 1
        if obs["scan"] and obs["scan"]["eve_detected"]:
            detections.append({"step": obs["step"], "cell": obs["scan"]["eve_position"],
                               "status_reading": obs["scan"]["status_reading"]})
        for c in obs["clues"]:
            if c.get("first_time", True):
                clues.append({k: c[k] for k in c if k != "first_time"} | {"found_at_step": obs["step"]})

    dna = snap.get("dna") or {}
    trust = {s: {"reliability": v["reliability"], "interval_90": v["interval_90"], "label": v["label"],
                 "judged": v["judged"]} for s, v in dna.items()}

    # ---- next step ----
    terrain = np.array(pub["terrain"])
    pos = tuple(last["observation"]["position"]) if last.get("observation") else tuple(pub["start"])
    battery = last["observation"]["battery"] if last.get("observation") else pub["start_battery"]
    dist, parent = dijkstra(terrain, pos, pub["costs"])
    route_cost = float(dist[est[1], est[0]])
    path = path_from_parents(parent, pos, est)
    if confirmed:
        tgt = tuple(confirmed)
        route_cost = float(dist[tgt[1], tgt[0]])
        path = path_from_parents(parent, pos, tgt)
        if status == "active":
            action = "approach EVE and attempt contact"
        else:
            action = "approach EVE and deliver charge to wake her (she appears dormant)"
        n_moves = len(path) - 1 if path else 0
        next_step = (f"Drive to EVE at {cell(tgt)} ({n_moves} move{'' if n_moves == 1 else 's'}, about "
                     f"{route_cost:.1f} battery from WALL-E's position; battery now {battery:.1f}) and {action}.")
        if route_cost > battery - pub["reserve"]:
            next_step += " Recharge at base first: the trip does not fit the current battery with reserve."
    else:
        base_dist, _ = dijkstra(terrain, tuple(pub["base"]), pub["costs"])
        round_trip = 2 * float(base_dist[est[1], est[0]]) + pub["costs"]["local_scan"] + pub["reserve"]
        next_step = (f"EVE not confirmed. Recharge at base, then local-scan around {cell(est)} "
                     f"({near_mass:.0%} of the probability lies within scan range of it).")
        if round_trip > pub["battery_capacity"]:
            next_step += (f" Note: that area needs about {round_trip:.0f} battery for a safe round trip, more than "
                          f"the {pub['battery_capacity']:.0f} capacity, so it needs a larger battery or a forward "
                          "charging point.")

    end_reason = last["observation"]["end_reason"] if last.get("observation") else None
    report = {
        "mission": pub["mission_name"], "agent": trace["meta"]["agent"], "seed": pub["seed"],
        "outcome": "EVE located" if confirmed else "EVE not located",
        "end_reason": end_reason,
        "estimated_location": list(confirmed) if confirmed else list(est),
        "location_confidence": {"p_estimated_cell": round(float(loc[est[1], est[0]]), 4),
                                "p_within_scan_radius": round(near_mass, 4),
                                "confirmed_by_local_scan": bool(confirmed)},
        "inferred_status": status, "status_confidence": round(status_conf, 4),
        "evidence": {"eve_detections": detections, "clues_found": clues,
                     "signal_readings": per_source, "long_scans": long_scans, "local_scans": local_scans},
        "signal_trust": trust,
        "resources": {"steps": len(events) - 1, "ticks_used": last["observation"]["time_used"] if last.get("observation") else 0,
                      "time_budget": pub["time_budget"], "battery_spent": round(spent, 2),
                      "battery_recharged": round(recharged, 2), "battery_left": round(battery, 2),
                      "moves": moves, "final_position": list(pos)},
        "remaining_uncertainty": {"entropy_bits": snap["entropy_bits"],
                                  "top_alternatives": snap["top_cells"][1:4]},
        "recommended_next_step": next_step,
    }
    return report


def report_markdown(r):
    lc = r["location_confidence"]
    lines = [f"# Mission report: {r['mission']}", "",
             f"Agent: `{r['agent']}`  ·  seed {r['seed']}  ·  outcome: **{r['outcome']}**  ·  ended: {r['end_reason']}", "",
             "## Where is EVE?", "",
             f"Estimated cell: {cell(r['estimated_location'])}  ",
             f"Probability of that exact cell: {lc['p_estimated_cell']:.1%}  ",
             f"Probability within scan radius: {lc['p_within_scan_radius']:.1%}  ",
             f"Confirmed by a local scan: {'yes' if lc['confirmed_by_local_scan'] else 'no'}", "",
             f"Inferred status: **{r['inferred_status']}** ({r['status_confidence']:.0%} confidence)", "",
             "## Evidence", ""]
    ev = r["evidence"]
    for d in ev["eve_detections"]:
        lines.append(f"- Step {d['step']}: local scan detected EVE at {cell(d['cell'])} (status reading: {d['status_reading']})")
    for c in ev["clues_found"]:
        if c["type"] == "track":
            lines.append(f"- Step {c['found_at_step']}: track clue at {cell(c['position'])} pointing {c['bearing_deg']}°")
        else:
            lines.append(f"- Step {c['found_at_step']}: residue clue at {cell(c['position'])} ({c['freshness']}, EVE {c['proximity']})")
    for s, v in ev["signal_readings"].items():
        lines.append(f"- Source `{s}`: {v['readings']} bearing readings" + (f", {v['silent']} silent scans" if v["silent"] else ""))
    lines.append(f"- {ev['long_scans']} long-range scans, {ev['local_scans']} local scans")
    if r["signal_trust"]:
        lines += ["", "## Signal DNA (learned trust)", "", "| Source | Reliability | 90% interval | Readings judged | Label |",
                  "| --- | --- | --- | --- | --- |"]
        for s, t in r["signal_trust"].items():
            lines.append(f"| {s} | {t['reliability']:.2f} | {t['interval_90'][0]:.2f}–{t['interval_90'][1]:.2f} | {t['judged']} | {t['label']} |")
    res = r["resources"]
    lines += ["", "## Resources", "",
              f"{res['steps']} actions, {res['ticks_used']}/{res['time_budget']} ticks, {res['moves']} moves. "
              f"Battery spent {res['battery_spent']}, recharged {res['battery_recharged']}, left {res['battery_left']}. "
              f"Final position {cell(res['final_position'])}.", "",
              "## Remaining uncertainty", "",
              f"Belief entropy {r['remaining_uncertainty']['entropy_bits']:.2f} bits. Next most likely cells: "
              + ", ".join(f"{cell(a['cell'])} ({a['p']:.1%})" for a in r["remaining_uncertainty"]["top_alternatives"]), "",
              "## Recommended next step", "", r["recommended_next_step"], ""]
    return "\n".join(lines)
