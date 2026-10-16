"""Environment: legal actions, costs, observations and hidden-truth separation."""
import json

import pytest

from phantom.config import normalise_config
from phantom.world import Action, Environment


def tiny_env(**over):
    raw = {
        "mission": {"seed": 1},
        "map": {"rows": ["B.~#..", "......", "......", "......", "......", "......"]},
        "resources": {"battery_capacity": 20, "time_budget": 50, "reserve": 2},
        "initial_signal": {"bearing_deg": 120, "timestamp": "2026-10-16T07:00:00Z"},
        "sensors": {"sources": {"relay": {"kind": "relay", "noise_deg": 10, "true_reliability": 0.5},
                                "beacon": {"kind": "beacon"}},
                    "local_scan": {"radius": 1, "detect_prob": 1.0, "status_accuracy": 1.0}},
        "eve": {"position": [5, 5], "status": "active"},
        "clues": {"items": [{"type": "track", "position": [1, 1]}]},
    }
    raw.update(over)
    return Environment(normalise_config(raw))


def test_move_costs_plain_and_rough():
    env = tiny_env()
    o = env.step(Action("move", "E"))          # plain
    assert o.ok and o.position == (1, 0) and o.battery_delta == pytest.approx(-1.0)
    o = env.step(Action("move", "E"))          # rough
    assert o.ok and o.position == (2, 0) and o.battery_delta == pytest.approx(-2.5)


def test_rock_and_edge_are_blocked_without_cost():
    env = tiny_env()
    o = env.step(Action("move", "N"))
    assert not o.ok and "edge" in o.error and o.position == (0, 0) and o.battery_delta == 0
    env.step(Action("move", "E"))
    env.step(Action("move", "E"))
    o = env.step(Action("move", "E"))          # rock at (3,0)
    assert not o.ok and "rock" in o.error and o.position == (2, 0)


def test_scan_costs_and_recharge_rules():
    env = tiny_env()
    assert env.step(Action("long_scan")).battery_delta == pytest.approx(-4.0)
    assert env.step(Action("local_scan")).battery_delta == pytest.approx(-2.0)
    o = env.step(Action("recharge"))
    assert o.ok and o.battery == pytest.approx(20.0)        # capped at capacity
    env.step(Action("move", "S"))
    o = env.step(Action("recharge"))
    assert not o.ok and "base" in o.error


def test_every_action_takes_one_tick_and_time_runs_out():
    env = tiny_env(resources={"battery_capacity": 20, "time_budget": 3})
    env.step(Action("long_scan"))
    env.step(Action("long_scan"))
    o = env.step(Action("local_scan"))
    assert o.time_used == 3 and o.mission_over and o.end_reason == "time_exhausted"
    with pytest.raises(RuntimeError):
        env.step(Action("long_scan"))


def test_insufficient_battery_is_refused():
    env = tiny_env(resources={"battery_capacity": 3, "time_budget": 50})
    o = env.step(Action("long_scan"))
    assert not o.ok and "not enough battery" in o.error and o.battery == pytest.approx(3.0)


def test_stranding_is_detected():
    env = tiny_env(resources={"battery_capacity": 3, "time_budget": 50, "reserve": 0})
    env.step(Action("move", "S"))
    env.step(Action("move", "S"))
    o = env.step(Action("move", "S"))
    assert o.mission_over and o.end_reason == "stranded"


def test_legal_actions_match_what_step_accepts():
    env = tiny_env()
    legal = {(a.type, a.direction) for a in env.legal_actions()}
    assert ("move", "N") not in legal and ("move", "E") in legal and ("recharge", None) not in legal
    for a in env.legal_actions():
        if a.type == "end_mission":
            continue
        e2 = tiny_env()
        assert e2.step(a).ok


def test_unknown_action_rejected():
    with pytest.raises(ValueError):
        tiny_env().step(Action("teleport"))


def test_public_info_has_no_hidden_truth():
    env = tiny_env()
    text = json.dumps(env.public_info())
    for word in ("true_reliability", "decoy", "\"eve\"", "clue_positions", "[5, 5]"):
        assert word not in text, word
    assert "items" not in text


def test_observations_reveal_eve_only_on_successful_local_scan():
    env = tiny_env()
    o = env.step(Action("long_scan"))
    text = json.dumps(o.to_dict())
    assert "eve_position" not in text and "true_reliability" not in text
    o = env.step(Action("local_scan"))              # EVE is far away
    assert o.scan["eve_detected"] is False and "eve_position" not in o.scan
    # walk next to EVE (5,5) and scan
    for d in "SSSSS":
        env.step(Action("move", d))
    for d in "EEEE":
        env.step(Action("move", d))
    o = env.step(Action("local_scan"))
    assert o.scan["eve_detected"] is True and o.scan["eve_position"] == [5, 5]


def test_clues_found_only_by_local_scan_in_range():
    env = tiny_env()
    assert env.step(Action("long_scan")).clues == []
    o = env.step(Action("local_scan"))           # (0,0) radius 1 does not reach (1,1)? distance 1.41 > 1
    assert o.clues == []
    env.step(Action("move", "E"))
    o = env.step(Action("local_scan"))           # (1,0) -> (1,1) is distance 1
    assert len(o.clues) == 1 and o.clues[0]["type"] == "track" and o.clues[0]["first_time"]
    o = env.step(Action("local_scan"))
    assert o.clues[0]["first_time"] is False
