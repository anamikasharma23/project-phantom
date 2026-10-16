"""Belief maths, Signal DNA, Ghost Decisions and the no-cheating guarantee."""
import ast
import copy
import math
from pathlib import Path

import numpy as np
import pytest

from phantom.agent import PhantomAgent
from phantom.belief import BeliefModel
from phantom.config import generate_mission
from phantom.geometry import bearing_deg
from phantom.mission import run_mission
from phantom.planner import local_scan_eig
from phantom.world import Environment

PKG = Path(__file__).resolve().parent.parent / "phantom"


# ---------------------------------------------------------------- no cheating
@pytest.mark.parametrize("module", ["agent.py", "belief.py", "planner.py", "signal_dna.py", "baselines.py"])
def test_agent_side_modules_never_import_the_environment(module):
    tree = ast.parse((PKG / module).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in ("world", "phantom.world", "mission", "phantom.mission"), module
        if isinstance(node, ast.Import):
            assert all("world" not in n.name for n in node.names), module


def test_first_decision_ignores_hidden_truth(sample_cfg):
    """Change ONLY hidden truth (EVE, clues, true reliability). The agent's
    starting belief and first decision must not change at all."""
    other = copy.deepcopy(sample_cfg)
    other["eve"]["position"] = [3, 1]
    other["eve"]["status"] = "dormant"
    other["clues"]["items"] = [{"type": "track", "position": [0, 0]}]
    other["sensors"]["sources"]["relay"]["true_reliability"] = 0.01
    a1 = PhantomAgent(Environment(sample_cfg).public_info(), seed=1)
    a2 = PhantomAgent(Environment(other).public_info(), seed=1)
    assert np.array_equal(a1.belief.P, a2.belief.P)
    assert a1.decide()[1]["candidates"] == a2.decide()[1]["candidates"]


# ---------------------------------------------------------------- belief maths
def _belief(sample_cfg):
    return BeliefModel(Environment(sample_cfg).public_info())


def test_belief_starts_normalised_and_zero_on_rock(sample_cfg):
    b = _belief(sample_cfg)
    assert b.P.sum() == pytest.approx(1.0)
    assert (b.P[:, ~b.passable] == 0).all()
    assert (b.P[:, b.passable] > 0).all()


def test_belief_stays_normalised_after_many_updates(sample_cfg):
    agent = PhantomAgent(Environment(sample_cfg).public_info(), seed=0)
    env = Environment(sample_cfg)
    for _ in range(30):
        action, _ = agent.decide()
        from phantom.world import Action
        obs = env.step(Action(action["type"], action.get("direction")))
        agent.observe(obs)
        assert agent.belief.P.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.isfinite(agent.belief.P).all() and (agent.belief.P >= 0).all()
        assert (agent.belief.P[:, agent.belief.passable] > 0).all()     # floor: nothing is ruled out forever
        if obs.mission_over:
            break


def test_empty_scan_lowers_probability_inside_radius(sample_cfg):
    b = _belief(sample_cfg)
    before = b.mass_within((10, 9), 2)
    b.add_scan(1, {"center": [10, 9], "radius": 2, "eve_detected": False}, {"detect_prob": 0.9})
    b.update({})
    after = b.mass_within((10, 9), 2)
    assert after < before * 0.2


def test_detection_collapses_belief(sample_cfg):
    b = _belief(sample_cfg)
    b.add_scan(1, {"center": [19, 4], "radius": 2, "eve_detected": True, "eve_position": [19, 3],
                   "status_reading": "active"}, {"detect_prob": 0.9, "status_accuracy": 0.85})
    b.update({})
    assert b.top_cells(1)[0]["cell"] == [19, 3] and b.top_cells(1)[0]["p"] > 0.99
    assert b.p_active() == pytest.approx(0.85, abs=0.01)


def test_clean_bearings_concentrate_belief_near_truth(sample_cfg):
    b = _belief(sample_cfg)
    truth = (19, 3)
    spec = {"kind": "relay", "noise_deg": 5}
    # origins on different sides of EVE: parallel bearings cannot fix the distance
    for origin in [(2, 3), (21, 15), (8, 9), (12, 15)]:
        r = {"source": "relay", "kind": "relay", "origin": list(origin), "bearing_deg": bearing_deg(origin, truth)}
        b.add_bearing(1, r, spec)
    b.update({"relay": 0.95})
    assert b.mass_within(truth, 2) > 0.8


def test_local_scan_eig_is_bounded_and_zero_where_belief_is_empty(sample_cfg):
    b = _belief(sample_cfg)
    spec = {"radius": 2, "detect_prob": 0.9}
    eig, pdet = local_scan_eig(b, b.P, (10, 9), spec)
    assert 0 < eig < b.entropy() and 0 < pdet < 1
    b.add_scan(1, {"center": [10, 9], "radius": 2, "eve_detected": False}, {"detect_prob": 0.9})
    b.add_scan(2, {"center": [10, 9], "radius": 2, "eve_detected": False}, {"detect_prob": 0.9})
    b.update({})
    eig2, _ = local_scan_eig(b, b.P, (10, 9), spec)
    assert eig2 < eig * 0.2                     # re-scanning a cleared area is worth little


# ---------------------------------------------------------------- Signal DNA
def test_dna_ranks_decoy_below_honest_sources(decoy_cfg):
    t = run_mission(decoy_cfg, "phantom")
    dna = t["events"][-1]["belief"]["dna"]
    assert dna["ghost_echo"]["reliability"] < dna["relay"]["reliability"]
    assert dna["ghost_echo"]["reliability"] < 0.3


def test_dna_never_calls_an_honest_source_conflicting():
    """Across seeded decoy missions, a source whose true reliability is >= 0.5
    must never be labelled 'conflicting' at any step."""
    for seed in range(8):
        t = run_mission(generate_mission(seed, "decoy"), "phantom")
        truth = t["evaluation"]["source_true_reliability"]
        for e in t["events"]:
            for s, v in e["belief"]["dna"].items():
                if truth[s] >= 0.5:
                    assert v["label"] != "conflicting", (seed, e["step"], s)


def test_dna_off_uses_fixed_trust(sample_cfg):
    t = run_mission(sample_cfg, "phantom_no_dna")
    rel = t["events"][-1]["belief"]["reliability_used"]
    assert set(rel.values()) == {0.8}


def test_dna_judgments_use_logged_numbers(decoy_cfg):
    t = run_mission(decoy_cfg, "phantom")
    found = 0
    for e in t["events"][1:]:
        for j in e["belief_update"]["dna_judgments"]:
            if "q_true" in j:
                assert j["agreement"] == pytest.approx(j["q_true"] / (j["q_true"] + j["q_noise"]), abs=2e-3)
                found += 1
    assert found > 0


# ---------------------------------------------------------------- Ghost Decisions
def test_chosen_action_is_best_feasible_unless_a_rule_forced_it(sample_cfg):
    t = run_mission(sample_cfg, "phantom")
    for e in t["events"][1:]:
        d = e["decision"]
        chosen = next(c for c in d["candidates"] if c["chosen"])
        assert chosen["feasible"]
        if d["forced_by"] is None:
            info = [c for c in d["candidates"] if c["type"] in ("local_scan", "long_scan", "goto_scan")
                    and c["feasible"]]
            assert chosen["score"] == max(c["score"] for c in info)
        # executed action really is the chosen candidate's first action
        assert e["observation"]["action"] == chosen["first_action"]


def test_scores_follow_the_documented_formula(sample_cfg):
    t = run_mission(sample_cfg, "phantom")
    checked = 0
    for e in t["events"][1:]:
        for c in e["decision"]["candidates"]:
            if c["type"] in ("local_scan", "long_scan", "goto_scan") and not c.get("commit_bonus"):
                effort = c["battery_cost"] + 0.5 * c["time_cost"]
                expect = c["eig_bits"] / effort * (1 - 0.5 * c["risk"])
                assert c["score"] == pytest.approx(expect, rel=2e-3, abs=6e-5)   # logged inputs are rounded
                checked += 1
    assert checked > 50


def test_vetoed_plans_really_would_break_the_reserve(sample_cfg):
    t = run_mission(sample_cfg, "phantom")
    reserve = t["public"]["reserve"]
    for e in t["events"][1:]:
        for c in e["decision"]["candidates"]:
            if c.get("veto", "") and "returning needs" in (c["veto"] or ""):
                assert c["battery_after"] - c["return_cost_after"] < reserve
