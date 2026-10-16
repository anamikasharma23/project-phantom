"""Mission config loading, validation and seeded generation.

A mission config (YAML or JSON) has two kinds of information:

* PUBLIC: map, base, budgets, action costs, sensor specs, the initial
  signal. The agent is allowed to know these.
* HIDDEN (environment only): EVE's position and status, clue positions,
  the true reliability of each signal source and any decoy position.

`load_config` returns one validated dict. The environment then decides what
the agent may see (see `Environment.public_info`).
"""
import copy
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .geometry import PLAIN, ROCK, ROUGH, bearing_deg, distance

MAP_CHARS = {".": PLAIN, "~": ROUGH, "#": ROCK, "B": PLAIN}

DEFAULT_COSTS = {"move_plain": 1.0, "move_rough": 2.5, "long_scan": 4.0,
                 "local_scan": 2.0, "recharge_rate": 25.0}
DEFAULT_LOCAL_SCAN = {"radius": 2.0, "detect_prob": 0.9, "status_accuracy": 0.85}
DEFAULT_AGENT = {"status_prior_active": 0.5, "dna_prior": [2.0, 2.0],
                 "initial_signal_trust": 0.9, "staleness_half_life_min": 120.0,
                 "use_signal_dna": True}


class ConfigError(ValueError):
    """Raised with a human-readable message when a config is invalid."""


def _require(cond, msg):
    if not cond:
        raise ConfigError(msg)


def _is_cell(v):
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(i, int) for i in v)


def parse_time(text, field):
    try:
        t = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConfigError(f"{field}: '{text}' is not an ISO-8601 timestamp") from exc
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t


def load_config(path):
    """Read a .yaml/.yml/.json file and return a validated, normalised dict."""
    path = Path(path)
    _require(path.exists(), f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not parse {path.name}: {exc}") from exc
    return normalise_config(raw)


# ---------------------------------------------------------------------------
# Terrain generation
# ---------------------------------------------------------------------------

def generate_terrain(width, height, seed, rock_density=0.12, rough_density=0.18):
    """Blob-style terrain from a seed: grows a few rock ridges and rough patches."""
    rng = random.Random(seed)
    grid = [[PLAIN] * width for _ in range(height)]

    def grow(code, target_cells):
        placed = 0
        while placed < target_cells:
            x, y = rng.randrange(width), rng.randrange(height)
            for _ in range(rng.randint(3, 9)):          # a short random walk = a blob
                if grid[y][x] == PLAIN:
                    grid[y][x] = code
                    placed += 1
                    if placed >= target_cells:
                        break
                dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                x = min(max(x + dx, 0), width - 1)
                y = min(max(y + dy, 0), height - 1)

    total = width * height
    grow(ROUGH, int(total * rough_density))
    grow(ROCK, int(total * rock_density))
    return grid


def _reachable(grid, start):
    h, w = len(grid), len(grid[0])
    seen = {tuple(start)}
    stack = [tuple(start)]
    while stack:
        x, y = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen and grid[ny][nx] != ROCK:
                seen.add((nx, ny))
                stack.append((nx, ny))
    return seen


# ---------------------------------------------------------------------------
# Validation + normalisation
# ---------------------------------------------------------------------------

def normalise_config(raw):
    _require(isinstance(raw, dict), "config must be a mapping (YAML/JSON object)")
    cfg = copy.deepcopy(raw)

    # --- mission -----------------------------------------------------------
    mission = cfg.setdefault("mission", {})
    _require(isinstance(mission, dict), "mission must be a mapping")
    mission.setdefault("name", "Unnamed mission")
    seed = mission.setdefault("seed", 0)
    _require(isinstance(seed, int), "mission.seed must be an integer")
    mission.setdefault("start_time", "2026-10-16T08:00:00Z")
    parse_time(mission["start_time"], "mission.start_time")

    # --- map ---------------------------------------------------------------
    m = cfg.get("map")
    _require(isinstance(m, dict), "map section is required")
    if "rows" in m:
        rows = m["rows"]
        _require(isinstance(rows, list) and rows and all(isinstance(r, str) for r in rows),
                 "map.rows must be a non-empty list of strings")
        width = len(rows[0])
        _require(all(len(r) == width for r in rows), "map.rows: all rows must have the same length")
        grid, base_from_rows = [], None
        for y, row in enumerate(rows):
            line = []
            for x, ch in enumerate(row):
                _require(ch in MAP_CHARS, f"map.rows[{y}][{x}]: unknown terrain '{ch}' (use . ~ # B)")
                line.append(MAP_CHARS[ch])
                if ch == "B":
                    base_from_rows = [x, y]
            grid.append(line)
        if "base" not in m and base_from_rows:
            m["base"] = base_from_rows
    else:
        width, height = m.get("width"), m.get("height")
        _require(isinstance(width, int) and isinstance(height, int),
                 "map needs either 'rows' or integer 'width' and 'height'")
        _require(6 <= width <= 80 and 6 <= height <= 80, "map width/height must be between 6 and 80")
        gen = m.get("generate", {}) or {}
        grid = generate_terrain(width, height, gen.get("seed", seed),
                                gen.get("rock_density", 0.12), gen.get("rough_density", 0.18))
    height, width = len(grid), len(grid[0])
    _require(width >= 6 and height >= 6, "map must be at least 6x6")
    base = m.get("base")
    _require(_is_cell(base), "map.base must be [x, y] (or put a 'B' in map.rows)")
    _require(0 <= base[0] < width and 0 <= base[1] < height, f"map.base {base} is outside the map")
    grid[base[1]][base[0]] = PLAIN                      # the base is always drivable
    reachable = _reachable(grid, base)
    m["grid"] = grid
    m["width"], m["height"] = width, height

    # --- resources ---------------------------------------------------------
    res = cfg.setdefault("resources", {})
    costs = {**DEFAULT_COSTS, **(res.get("costs") or {})}
    for k, v in costs.items():
        _require(isinstance(v, (int, float)) and v >= 0, f"resources.costs.{k} must be a number >= 0")
    _require(costs["move_plain"] > 0 and costs["move_rough"] > 0, "move costs must be > 0")
    res["costs"] = costs
    res.setdefault("battery_capacity", 120.0)
    res.setdefault("start_battery", res["battery_capacity"])
    res.setdefault("time_budget", 260)
    res.setdefault("reserve", 6.0)
    _require(res["battery_capacity"] > 0, "resources.battery_capacity must be > 0")
    _require(0 < res["start_battery"] <= res["battery_capacity"],
             "resources.start_battery must be in (0, battery_capacity]")
    _require(isinstance(res["time_budget"], int) and res["time_budget"] > 0,
             "resources.time_budget must be a positive integer (ticks)")

    # --- sensors -----------------------------------------------------------
    sens = cfg.setdefault("sensors", {})
    sens["local_scan"] = {**DEFAULT_LOCAL_SCAN, **(sens.get("local_scan") or {})}
    sources = sens.get("sources")
    _require(isinstance(sources, dict) and sources, "sensors.sources must define at least one signal source")
    for name, s in sources.items():
        _require(isinstance(s, dict), f"sensors.sources.{name} must be a mapping")
        kind = s.get("kind")
        _require(kind in ("beacon", "relay"), f"sensors.sources.{name}.kind must be 'beacon' or 'relay'")
        s.setdefault("noise_deg", 12.0)
        s.setdefault("true_reliability", 1.0)
        _require(0.0 <= s["true_reliability"] <= 1.0,
                 f"sensors.sources.{name}.true_reliability must be in [0, 1]")
        if kind == "beacon":
            s.setdefault("range", 14.0)
            s.setdefault("emit_active", 0.75)
            s.setdefault("emit_dormant", 0.2)
        if "decoy" in s:
            _require(_is_cell(s["decoy"]), f"sensors.sources.{name}.decoy must be [x, y]")

    # --- hidden truth: EVE ---------------------------------------------------
    rng = random.Random(seed * 7919 + 17)
    eve = cfg.setdefault("eve", {})
    if "position" not in eve:
        candidates = [c for c in sorted(reachable) if distance(c, base) >= 0.45 * max(width, height)]
        _require(candidates, "map too small to place EVE away from base")
        eve["position"] = list(rng.choice(candidates))
    _require(_is_cell(eve["position"]), "eve.position must be [x, y]")
    _require(tuple(eve["position"]) in reachable,
             f"eve.position {eve['position']} is on rock or unreachable from base")
    eve.setdefault("status", rng.choice(["active", "dormant"]))
    _require(eve["status"] in ("active", "dormant"), "eve.status must be 'active' or 'dormant'")

    # --- hidden truth: clues -------------------------------------------------
    clues = cfg.setdefault("clues", {})
    if "items" not in clues:
        gen = clues.get("generate", {}) or {}
        crng = random.Random(gen.get("seed", seed) * 104729 + 3)
        count = gen.get("count", 6)
        pool = [c for c in sorted(reachable) if c != tuple(base) and c != tuple(eve["position"])]
        # bias clues toward EVE's region so they are worth finding
        pool.sort(key=lambda c: distance(c, eve["position"]))
        near = pool[: max(count * 6, 1)]
        crng.shuffle(near)
        items = []
        for i, c in enumerate(near[:count]):
            items.append({"type": "track" if i % 2 == 0 else "residue", "position": list(c)})
        clues["items"] = items
    for i, item in enumerate(clues["items"]):
        _require(item.get("type") in ("track", "residue"), f"clues.items[{i}].type must be track or residue")
        _require(_is_cell(item.get("position")), f"clues.items[{i}].position must be [x, y]")

    # --- initial signal ------------------------------------------------------
    sig = cfg.get("initial_signal")
    _require(isinstance(sig, dict), "initial_signal section is required")
    sig.setdefault("origin", list(base))
    _require(_is_cell(sig["origin"]), "initial_signal.origin must be [x, y]")
    _require("bearing_deg" in sig or "coordinates" in sig,
             "initial_signal needs 'bearing_deg' (direction) or 'coordinates'")
    if "coordinates" in sig:
        _require(_is_cell(sig["coordinates"]), "initial_signal.coordinates must be [x, y]")
    if "bearing_deg" in sig:
        _require(isinstance(sig["bearing_deg"], (int, float)), "initial_signal.bearing_deg must be a number")
    sig.setdefault("strength", None)
    _require("timestamp" in sig, "initial_signal.timestamp is required")
    ts = parse_time(sig["timestamp"], "initial_signal.timestamp")
    _require(ts <= parse_time(mission["start_time"], "mission.start_time"),
             "initial_signal.timestamp cannot be after mission.start_time")
    sig.setdefault("noise_deg", 15.0)
    sig.setdefault("coord_sigma", 3.0)

    cfg["agent"] = {**DEFAULT_AGENT, **(cfg.get("agent") or {})}
    return cfg


# ---------------------------------------------------------------------------
# Seeded scenario generator (used by the benchmark)
# ---------------------------------------------------------------------------

SCENARIOS = {
    "clean":          {"relay_rel": 0.85, "relay_noise": 10, "decoy": False, "battery": 120, "beacon_noise": 10},
    "noisy":          {"relay_rel": 0.75, "relay_noise": 22, "decoy": False, "battery": 120, "beacon_noise": 22},
    "unreliable":     {"relay_rel": 0.35, "relay_noise": 12, "decoy": False, "battery": 120, "beacon_noise": 12},
    "decoy":          {"relay_rel": 0.8, "relay_noise": 10, "decoy": True, "battery": 120, "beacon_noise": 12},
    "tight_battery":  {"relay_rel": 0.8, "relay_noise": 12, "decoy": False, "battery": 70, "beacon_noise": 12},
}


def generate_mission(seed, scenario="clean", width=24, height=18):
    """Build a complete random mission dict for `scenario` from `seed`."""
    p = SCENARIOS[scenario]
    rng = random.Random(seed * 31 + len(scenario))
    base = [1 + rng.randrange(3), height - 2 - rng.randrange(3)]
    raw = {
        "mission": {"name": f"{scenario} #{seed}", "seed": seed, "start_time": "2026-10-16T08:00:00Z"},
        "map": {"width": width, "height": height, "base": base,
                "generate": {"seed": seed, "rock_density": 0.12, "rough_density": 0.18}},
        "resources": {"battery_capacity": p["battery"], "time_budget": 260},
        "sensors": {"sources": {
            "beacon": {"kind": "beacon", "noise_deg": p["beacon_noise"], "range": 14.0, "true_reliability": 1.0},
            "relay": {"kind": "relay", "noise_deg": p["relay_noise"], "true_reliability": p["relay_rel"]},
        }},
        "clues": {"generate": {"seed": seed, "count": 6}},
        "initial_signal": {"bearing_deg": 0, "timestamp": "2026-10-16T07:00:00Z"},
    }
    cfg = normalise_config(raw)
    eve = cfg["eve"]["position"]
    # initial signal: true bearing from base + noise, stale by 60 minutes
    true_b = bearing_deg(base, eve)
    cfg["initial_signal"]["bearing_deg"] = round((true_b + rng.gauss(0, 12)) % 360, 1)
    cfg["initial_signal"]["strength"] = round(max(0.02, math.exp(-distance(base, eve) / 14.0)
                                                   + rng.gauss(0, 0.05)), 3)
    if p["decoy"]:
        # decoy: a third source that confidently points at a fake location
        far = [(x, y) for y in range(height) for x in range(width)
               if cfg["map"]["grid"][y][x] != ROCK and distance((x, y), eve) > 10]
        decoy = list(rng.choice(far))
        cfg["sensors"]["sources"]["ghost_echo"] = {"kind": "relay", "noise_deg": 6.0,
                                                   "true_reliability": 0.0, "decoy": decoy}
    return cfg
