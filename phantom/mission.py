"""Runs one mission: config -> environment + agent -> trace -> report.

The loop is the whole project in five lines:

    action, decision = agent.decide()          # agent thinks (sees only its own belief)
    observation = env.step(action)             # world reacts (only place truth lives)
    update = agent.observe(observation)        # agent learns
    trace.append(...)                          # everything is logged
    stop when observation.mission_over
"""
import json
from pathlib import Path

from .agent import PhantomAgent
from .baselines import RandomAgent, SweepAgent
from .report import build_report, report_markdown
from .world import Action, Environment

AGENTS = ("phantom", "phantom_no_dna", "sweep", "random")


def make_agent(kind, info, seed):
    if kind == "phantom":
        return PhantomAgent(info, use_dna=True, seed=seed)
    if kind == "phantom_no_dna":
        return PhantomAgent(info, use_dna=False, seed=seed)
    if kind == "sweep":
        return SweepAgent(info, seed=seed)
    if kind == "random":
        return RandomAgent(info, seed=seed)
    raise ValueError(f"unknown agent '{kind}', choose from {AGENTS}")


class MissionRunner:
    """Runs a mission one action at a time (used by the live UI and by run_mission)."""

    def __init__(self, cfg, agent_kind="phantom"):
        self.env = Environment(cfg)
        info = self.env.public_info()
        seed = cfg["mission"]["seed"]
        self.agent = make_agent(agent_kind, info, seed)
        self.max_steps = info["time_budget"] + 5
        self.trace = {"format": "phantom-trace/1",
                      "meta": {"agent": agent_kind, "seed": seed, "mission": info["mission_name"]},
                      "public": info,
                      "events": [{"step": 0, "type": "start", "position": list(info["start"]),
                                  "battery": info["start_battery"], "time_left": info["time_budget"],
                                  "decision": None, "observation": None, "belief_update": None,
                                  "belief": self.agent.snapshot()}]}
        self.done = False

    def step(self):
        """One decide -> act -> observe cycle. Returns the new trace event."""
        if self.done:
            raise RuntimeError("mission is already over")
        action_dict, decision = self.agent.decide()
        action = Action(action_dict["type"], action_dict.get("direction"))
        obs = self.env.step(action)
        update = self.agent.observe(obs)
        event = {"step": obs.step, "type": "action", "position": list(obs.position),
                 "battery": round(obs.battery, 3), "time_left": obs.time_left,
                 "decision": decision, "observation": obs.to_dict(),
                 "belief_update": update, "belief": self.agent.snapshot()}
        self.trace["events"].append(event)
        if obs.mission_over or len(self.trace["events"]) > self.max_steps:
            self.finish()
        return event

    def finish(self):
        self.done = True
        self.trace["report"] = build_report(self.trace)
        # ground truth is added only now, after the agent is finished, for scoring
        ev = self.env.evaluation(self.trace["report"]["estimated_location"])
        self.trace["evaluation"] = ev | score_mission(self.trace, ev)

    def run(self):
        while not self.done:
            self.step()
        return self.trace


def run_mission(cfg, agent_kind="phantom"):
    return MissionRunner(cfg, agent_kind).run()


def score_mission(trace, ev):
    """Benchmark metrics. Uses ground truth, so only ever computed after a run."""
    r = trace["report"]
    pub = trace["public"]
    W = pub["width"]
    loc = trace["events"][-1]["belief"]["location"]
    tx, ty = ev["eve_position"]
    return {"metrics": {
        "success": bool(r["location_confidence"]["confirmed_by_local_scan"]
                        and r["estimated_location"] == ev["eve_position"]),
        "stranded": r["end_reason"] == "stranded",
        "steps": r["resources"]["steps"],
        "battery_spent": r["resources"]["battery_spent"],
        "error_cells": ev["estimate_error_cells"],
        "p_true_cell": round(loc[ty * W + tx], 6),
        "status_correct": r["inferred_status"] == ev["eve_status"],
    }}


def write_outputs(trace, out_dir, ui_dir=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "trace.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "header", "format": trace["format"], "meta": trace["meta"],
                            "public": trace["public"]}) + "\n")
        for e in trace["events"]:
            f.write(json.dumps(e) + "\n")
        f.write(json.dumps({"type": "report", "report": trace["report"]}) + "\n")
        f.write(json.dumps({"type": "evaluation", "evaluation": trace["evaluation"]}) + "\n")
    (out / "report.json").write_text(json.dumps(trace["report"], indent=2), encoding="utf-8")
    (out / "report.md").write_text(report_markdown(trace["report"]), encoding="utf-8")
    (out / "mission.json").write_text(json.dumps(trace), encoding="utf-8")
    if ui_dir:
        ui = Path(ui_dir)
        ui.mkdir(parents=True, exist_ok=True)
        (ui / "mission.js").write_text("window.PHANTOM_MISSION = " + json.dumps(trace) + ";\n", encoding="utf-8")
    return out
