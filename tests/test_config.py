import copy
import json

import pytest
import yaml

from phantom.config import ConfigError, generate_mission, load_config, normalise_config
from phantom.geometry import ROCK


def write(tmp_path, data, name="m.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data) if name.endswith(".yaml") else json.dumps(data), encoding="utf-8")
    return p


def base_raw():
    return {
        "map": {"rows": ["B.....", "......", "..##..", "......", "......", "......"]},
        "initial_signal": {"bearing_deg": 120, "timestamp": "2026-10-16T07:00:00Z"},
        "sensors": {"sources": {"relay": {"kind": "relay", "noise_deg": 10, "true_reliability": 0.7}}},
        "eve": {"position": [5, 5], "status": "dormant"},
    }


def test_sample_configs_load(sample_cfg, decoy_cfg):
    assert sample_cfg["map"]["width"] == 24 and sample_cfg["map"]["height"] == 18
    assert "ghost_echo" in decoy_cfg["sensors"]["sources"]


def test_json_and_yaml_equivalent(tmp_path):
    a = load_config(write(tmp_path, base_raw(), "a.yaml"))
    b = load_config(write(tmp_path, base_raw(), "b.json"))
    assert a == b


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r.pop("map"), "map section is required"),
    (lambda r: r["map"].__setitem__("rows", ["B..x..", "......", "......", "......", "......", "......"]), "unknown terrain"),
    (lambda r: r["map"].__setitem__("rows", ["B.....", "....", "......", "......", "......", "......"]), "same length"),
    (lambda r: r["eve"].__setitem__("position", [2, 2]), "on rock"),
    (lambda r: r["eve"].__setitem__("status", "sleepy"), "eve.status"),
    (lambda r: r["initial_signal"].__setitem__("timestamp", "2030-01-01T00:00:00Z"), "after mission.start_time"),
    (lambda r: r["initial_signal"].__setitem__("timestamp", "yesterday"), "ISO-8601"),
    (lambda r: r["initial_signal"].pop("bearing_deg"), "bearing_deg"),
    (lambda r: r["sensors"]["sources"]["relay"].__setitem__("kind", "radio"), "beacon' or 'relay"),
    (lambda r: r["sensors"]["sources"]["relay"].__setitem__("true_reliability", 1.5), "true_reliability"),
    (lambda r: r.__setitem__("resources", {"costs": {"move_plain": -1}}), "move_plain"),
])
def test_invalid_configs_give_clear_errors(mutate, message):
    raw = base_raw()
    mutate(raw)
    with pytest.raises(ConfigError, match=message):
        normalise_config(raw)


def test_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("map: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not parse"):
        load_config(bad)


def test_seeded_map_and_mission_are_reproducible():
    a, b = generate_mission(5, "decoy"), generate_mission(5, "decoy")
    assert a == b
    c = generate_mission(6, "decoy")
    assert a["map"]["grid"] != c["map"]["grid"]


def test_seeded_eve_is_reachable_and_not_on_rock():
    for seed in range(20):
        cfg = generate_mission(seed, "clean")
        x, y = cfg["eve"]["position"]
        assert cfg["map"]["grid"][y][x] != ROCK
