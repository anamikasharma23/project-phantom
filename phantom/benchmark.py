"""Benchmark: PHANTOM vs baselines on seeded random missions.

For each scenario and seed we generate one mission and run every agent on
the SAME mission. Metrics come from the trace plus ground truth (which is
only looked at after each run finishes).

What this measures: success rate (EVE confirmed by a local scan at her true
cell), actions used, battery spent, strandings, probability the final belief
puts on EVE's true cell, and whether the status guess was right.
What it does NOT prove: performance on other map sizes, sensor models or
the organisers' own configs; 20 seeds per scenario gives rough estimates
with wide uncertainty (Wilson 95% intervals are reported).
"""
import json
import math
from pathlib import Path
from statistics import mean, median

from .config import SCENARIOS, generate_mission
from .mission import AGENTS, run_mission

ROOT = Path(__file__).resolve().parent.parent


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - r) / d), min(1.0, (c + r) / d))


def run_benchmark(seeds=20, scenarios=None, agents=AGENTS, out_dir=None, quiet=False):
    scenarios = scenarios or list(SCENARIOS)
    rows = []
    for sc in scenarios:
        for seed in range(seeds):
            cfg = generate_mission(1000 + seed, sc)
            for ag in agents:
                m = run_mission(cfg, ag)["evaluation"]["metrics"]
                rows.append({"scenario": sc, "seed": 1000 + seed, "agent": ag, **m})
            if not quiet:
                print(f"  {sc:14s} seed {1000 + seed} done", flush=True)
    summary = []
    for sc in scenarios:
        for ag in agents:
            rs = [r for r in rows if r["scenario"] == sc and r["agent"] == ag]
            k = sum(r["success"] for r in rs)
            lo, hi = wilson(k, len(rs))
            ok = [r for r in rs if r["success"]]
            summary.append({
                "scenario": sc, "agent": ag, "runs": len(rs),
                "success_rate": round(k / len(rs), 3), "success_ci95": [round(lo, 3), round(hi, 3)],
                "mean_steps_when_found": round(mean(r["steps"] for r in ok), 1) if ok else None,
                "mean_battery_spent": round(mean(r["battery_spent"] for r in rs), 1),
                "stranded": sum(r["stranded"] for r in rs),
                "mean_p_true_cell": round(mean(r["p_true_cell"] for r in rs), 3),
                "median_error_cells": round(median(r["error_cells"] for r in rs), 2),
                "status_correct_rate": round(mean(r["status_correct"] for r in rs), 3),
            })
    result = {"seeds_per_scenario": seeds, "scenarios": {s: SCENARIOS[s] for s in scenarios},
              "agents": list(agents), "summary": summary, "runs": rows}
    out = Path(out_dir) if out_dir else ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out / "benchmark.md").write_text(benchmark_markdown(result), encoding="utf-8")
    ui = ROOT / "ui" / "data"
    ui.mkdir(parents=True, exist_ok=True)
    (ui / "benchmark.js").write_text("window.PHANTOM_BENCH = " + json.dumps({k: v for k, v in result.items()
                                                                           if k != "runs"}) + ";\n", encoding="utf-8")
    if not quiet:
        print(benchmark_markdown(result))
    return result


def benchmark_markdown(res):
    lines = [f"# Benchmark ({res['seeds_per_scenario']} seeded missions per scenario)", "",
             "| Scenario | Agent | Success | 95% CI | Steps when found | Battery spent | Stranded | P(true cell) | Status right |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in res["summary"]:
        lines.append(f"| {s['scenario']} | {s['agent']} | {s['success_rate']:.0%} | "
                     f"{s['success_ci95'][0]:.0%}–{s['success_ci95'][1]:.0%} | {s['mean_steps_when_found'] or '–'} | "
                     f"{s['mean_battery_spent']} | {s['stranded']} | {s['mean_p_true_cell']:.2f} | "
                     f"{s['status_correct_rate']:.0%} |")
    return "\n".join(lines) + "\n"
