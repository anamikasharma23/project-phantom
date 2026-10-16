"""Local server for the live mission console.

    python -m phantom serve            # then open http://127.0.0.1:8765

It serves the UI and a tiny JSON API so the UI's Start / Pause / Step / Reset
controls and the configuration screen drive the REAL simulation:

    GET  /api/info        presets, agents, defaults
    POST /api/session     build a mission from a preset + overrides, returns the start of the trace
    POST /api/step        run N actions, returns the new trace events
    POST /api/finish      run to the end, returns the remaining events + report
    GET  /api/benchmark   saved benchmark summary (if `phantom bench` was run)

Only the Python standard library is used. The server listens on 127.0.0.1 only.
Ground truth never leaves the server until a mission is finished, and then only
inside the trace's separate `evaluation` block (used by the report's evaluation section).
"""
import json
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from .config import SCENARIOS, ConfigError, generate_mission, normalise_config
from .mission import AGENTS, MissionRunner, write_outputs

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "ui"
CONFIG_DIR = ROOT / "configs"
OUTPUT_DIR = ROOT / "outputs"          # finished live missions are saved here (tests point it elsewhere)
MAX_SESSIONS = 20
LIMITS = {"battery_capacity": (20, 400), "time_budget": (20, 1000), "seed": (0, 10_000_000)}

_sessions = {}
_lock = threading.Lock()


def presets():
    out = []
    for p in sorted(CONFIG_DIR.glob("*.y*ml")):
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
            name = raw.get("mission", {}).get("name", p.stem)
            res = raw.get("resources", {})
            out.append({"id": f"file:{p.stem}", "label": name, "kind": "file",
                        "defaults": {"seed": raw.get("mission", {}).get("seed", 0),
                                     "battery_capacity": res.get("battery_capacity", 120),
                                     "time_budget": res.get("time_budget", 260)}})
        except (yaml.YAMLError, OSError, AttributeError):
            continue
    for name, p in SCENARIOS.items():
        out.append({"id": f"scenario:{name}", "label": f"Random map: {name.replace('_', ' ')}", "kind": "scenario",
                    "defaults": {"seed": 1000, "battery_capacity": p["battery"], "time_budget": 260}})
    return out


def build_config(req):
    """Turn a UI request into a validated config. Raises ConfigError with a readable message."""
    preset = req.get("preset", "file:sample_mission")
    for key, (lo, hi) in LIMITS.items():
        v = req.get(key)
        if v is None or v == "":
            continue
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ConfigError(f"{key.replace('_', ' ')} must be a number")
        if not lo <= v <= hi:
            raise ConfigError(f"{key.replace('_', ' ')} must be between {lo} and {hi} (got {v})")
    if preset.startswith("file:"):
        path = CONFIG_DIR / f"{preset[5:]}.yaml"
        if not path.exists() or path.parent != CONFIG_DIR:
            raise ConfigError(f"unknown preset {preset}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif preset.startswith("scenario:") and preset[9:] in SCENARIOS:
        seed = int(req["seed"]) if req.get("seed") not in (None, "") else 1000
        raw = generate_mission(seed, preset[9:])
    else:
        raise ConfigError(f"unknown preset {preset}")
    if req.get("seed") not in (None, ""):
        raw.setdefault("mission", {})["seed"] = int(req["seed"])
    res = raw.setdefault("resources", {})
    if req.get("battery_capacity") not in (None, ""):
        res["battery_capacity"] = float(req["battery_capacity"])
        res["start_battery"] = float(req["battery_capacity"])
    if req.get("time_budget") not in (None, ""):
        res["time_budget"] = int(req["time_budget"])
    return normalise_config(raw)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(UI_DIR), **kw)

    def log_message(self, fmt, *args):      # keep the terminal quiet
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 100_000:
            raise ConfigError("request too large")
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError as exc:
            raise ConfigError(f"request is not valid JSON: {exc}") from exc

    def do_GET(self):
        if self.path == "/api/info":
            return self._json(200, {"live": True, "presets": presets(), "agents": list(AGENTS),
                                    "limits": LIMITS})
        if self.path == "/api/benchmark":
            p = ROOT / "outputs" / "benchmark.json"
            if not p.exists():
                return self._json(404, {"error": "no benchmark yet: run python -m phantom bench"})
            data = json.loads(p.read_text(encoding="utf-8"))
            data.pop("runs", None)
            return self._json(200, data)
        return super().do_GET()

    def do_POST(self):
        try:
            req = self._body()
            if self.path == "/api/session":
                cfg = build_config(req)
                agent = req.get("agent", "phantom")
                if agent not in AGENTS:
                    raise ConfigError(f"unknown agent {agent}")
                runner = MissionRunner(cfg, agent)
                sid = uuid.uuid4().hex[:12]
                with _lock:
                    if len(_sessions) >= MAX_SESSIONS:
                        _sessions.pop(next(iter(_sessions)))
                    _sessions[sid] = {"runner": runner, "preset": req.get("preset", "file:sample_mission")}
                    t = runner.trace
                    return self._json(200, {"session": sid, "trace": {k: t[k] for k in ("format", "meta", "public", "events")}})
            if self.path in ("/api/step", "/api/finish"):
                with _lock:
                    s = _sessions.get(req.get("session"))
                    if not s:
                        return self._json(404, {"error": "mission session not found (the server may have restarted); press Reset"})
                    runner = s["runner"]
                    count = 10_000 if self.path == "/api/finish" else max(1, min(int(req.get("count", 1)), 50))
                    new = []
                    while not runner.done and len(new) < count:
                        new.append(runner.step())
                    out = {"events": new, "done": runner.done}
                    if runner.done:
                        if "saved_to" not in s:
                            name = s["preset"].split(":", 1)[1]
                            dest = OUTPUT_DIR / f"live_{name}_{runner.trace['meta']['agent']}"
                            write_outputs(runner.trace, dest)
                            try:
                                s["saved_to"] = str(dest.relative_to(ROOT))
                            except ValueError:
                                s["saved_to"] = str(dest)
                        out.update(report=runner.trace["report"], evaluation=runner.trace["evaluation"],
                                   saved_to=s["saved_to"])
                    return self._json(200, out)
            return self._json(404, {"error": "unknown endpoint"})
        except ConfigError as exc:
            return self._json(400, {"error": str(exc)})
        except (ValueError, TypeError) as exc:
            return self._json(400, {"error": f"invalid request: {exc}"})


def serve(port=8765):
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        print(f"Could not start on port {port}: {exc.strerror or exc}.\n"
              f"Another program (or an earlier phantom serve) is probably using it.\n"
              f"Close that window, or pick another port:  python -m phantom serve --port {port + 1}")
        raise SystemExit(1) from None
    print(f"PROJECT PHANTOM console running at http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
