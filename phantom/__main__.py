"""Command line entry point.

    python -m phantom run configs/sample_mission.yaml          # run + write outputs + UI data
    python -m phantom run configs/decoy_mission.yaml --agent sweep
    python -m phantom validate configs/sample_mission.yaml
    python -m phantom bench --seeds 20                          # baseline comparison
    python -m phantom serve                                     # live console at http://127.0.0.1:8765
"""
import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .mission import AGENTS, run_mission, write_outputs

ROOT = Path(__file__).resolve().parent.parent


def ascii_map(trace):
    """Terminal picture of the final state (bonus ASCII visualisation)."""
    pub = trace["public"]
    W, H = pub["width"], pub["height"]
    last = trace["events"][-1]["belief"]
    loc = last["location"]
    route = {tuple(e["position"]) for e in trace["events"]}
    est = tuple(trace["report"]["estimated_location"])
    shades = " .:-=+*#%@"
    mx = max(loc) or 1
    lines = []
    for y in range(H):
        row = []
        for x in range(W):
            t = pub["terrain"][y][x]
            if (x, y) == tuple(trace["events"][-1]["position"]):
                ch = "W"
            elif (x, y) == tuple(pub["base"]):
                ch = "B"
            elif (x, y) == est:
                ch = "E"
            elif t == 2:
                ch = "█"
            elif (x, y) in route:
                ch = "o"
            else:
                ch = shades[min(9, int(9 * loc[y * W + x] / mx))] if loc[y * W + x] / mx > 0.05 else ("~" if t == 1 else "·")
            row.append(ch)
        lines.append("".join(row))
    lines.append("W = WALL-E   B = base   E = estimated EVE   o = route   █ = rock   .:-=+*#%@ = belief")
    return "\n".join(lines)


def _safe_console():
    """Windows consoles and redirected output may not be UTF-8. Never crash on a map character."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None):
    _safe_console()
    p = argparse.ArgumentParser(prog="phantom", description="PROJECT PHANTOM: WALL-E search agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one mission")
    r.add_argument("config")
    r.add_argument("--agent", default="phantom", choices=AGENTS)
    r.add_argument("--out", default=None, help="output folder (default outputs/<config name>)")
    r.add_argument("--no-ui", action="store_true", help="do not refresh ui/data/mission.js")
    v = sub.add_parser("validate", help="check a config file")
    v.add_argument("config")
    sv = sub.add_parser("serve", help="start the live mission console at http://127.0.0.1:8765")
    sv.add_argument("--port", type=int, default=8765)
    b = sub.add_parser("bench", help="benchmark agents on seeded random missions")
    b.add_argument("--seeds", type=int, default=20)
    b.add_argument("--scenarios", nargs="*", default=None)
    args = p.parse_args(argv)

    if args.cmd in ("run", "validate"):
        try:
            cfg = load_config(args.config)
        except ConfigError as exc:
            print(f"CONFIG ERROR: {exc}", file=sys.stderr)
            return 2
        if args.cmd == "validate":
            print(f"OK: '{cfg['mission']['name']}' {cfg['map']['width']}x{cfg['map']['height']}, "
                  f"{len(cfg['sensors']['sources'])} signal sources, {len(cfg['clues']['items'])} clues")
            return 0
        trace = run_mission(cfg, args.agent)
        out = Path(args.out) if args.out else ROOT / "outputs" / f"{Path(args.config).stem}_{args.agent}"
        write_outputs(trace, out, None if args.no_ui else ROOT / "ui" / "data")
        rep = trace["report"]
        print(ascii_map(trace))
        print()
        print(f"Outcome: {rep['outcome']} ({rep['end_reason']}) in {rep['resources']['steps']} actions")
        est = rep["estimated_location"]
        print(f"Estimate: ({est[0]}, {est[1]})  status: {rep['inferred_status']} "
              f"({rep['status_confidence']:.0%})")
        print(f"Next step: {rep['recommended_next_step']}")
        print(f"Wrote trace, report and mission bundle to {out}")
        if not args.no_ui:
            print("UI data refreshed: open ui/index.html")
        return 0
    if args.cmd == "serve":
        from .server import serve
        serve(args.port)
        return 0
    if args.cmd == "bench":
        from .benchmark import run_benchmark
        run_benchmark(args.seeds, args.scenarios)
        return 0


if __name__ == "__main__":
    sys.exit(main())
