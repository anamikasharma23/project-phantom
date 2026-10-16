"""End to end: safety, repeatability, trace/replay consistency, report."""
import json

import numpy as np
import pytest

from phantom.config import generate_mission
from phantom.geometry import DIRECTIONS, ROCK
from phantom.mission import AGENTS, run_mission, write_outputs
from phantom.report import build_report


def test_sample_mission_runs_and_finds_eve(sample_cfg):
    t = run_mission(sample_cfg, "phantom")
    assert t["report"]["outcome"] == "EVE located"
    assert t["evaluation"]["metrics"]["success"]


@pytest.mark.parametrize("agent", AGENTS)
def test_same_config_and_seed_give_identical_traces(sample_cfg, agent):
    a = json.dumps(run_mission(sample_cfg, agent), sort_keys=True)
    b = json.dumps(run_mission(sample_cfg, agent), sort_keys=True)
    assert a == b


@pytest.mark.parametrize("scenario", ["clean", "tight_battery", "decoy"])
def test_never_stranded_across_seeded_missions(scenario):
    for seed in range(6):
        for agent in ("phantom", "sweep", "random"):
            t = run_mission(generate_mission(seed, scenario), agent)
            assert t["report"]["end_reason"] != "stranded", (scenario, seed, agent)
            for e in t["events"][1:]:
                assert e["battery"] >= -1e-9


def test_replay_reconstruction_is_consistent(decoy_cfg):
    """Everything the UI replays must be derivable step by step from the trace."""
    t = run_mission(decoy_cfg, "phantom")
    pub = t["public"]
    terrain = np.array(pub["terrain"])
    pos = tuple(pub["start"])
    battery = pub["start_battery"]
    for e in t["events"][1:]:
        o = e["observation"]
        a = o["action"]
        if a["type"] == "move" and o["ok"]:
            dx, dy = DIRECTIONS[a["direction"]]
            pos = (pos[0] + dx, pos[1] + dy)
            assert terrain[pos[1], pos[0]] != ROCK
        assert tuple(e["position"]) == pos == tuple(o["position"])
        battery += o["battery_delta"]
        assert e["battery"] == pytest.approx(battery, abs=1e-6)
        loc = np.array(e["belief"]["location"])
        assert loc.sum() == pytest.approx(1.0, abs=1e-3)
        top = e["belief"]["top_cells"][0]
        assert loc[top["cell"][1] * pub["width"] + top["cell"][0]] == pytest.approx(top["p"], abs=1e-4)
    # belief really changes during the mission (replay must not show only the final map)
    first = np.array(t["events"][0]["belief"]["location"])
    last = np.array(t["events"][-1]["belief"]["location"])
    assert np.abs(first - last).sum() > 0.5


def test_report_is_built_only_from_trace(decoy_cfg):
    t = run_mission(decoy_cfg, "phantom")
    stripped = {k: v for k, v in t.items() if k not in ("evaluation", "report")}
    assert build_report(stripped) == t["report"]
    r = t["report"]
    for key in ("estimated_location", "location_confidence", "inferred_status", "status_confidence",
                "evidence", "resources", "recommended_next_step", "remaining_uncertainty"):
        assert key in r


def test_outputs_written_and_jsonl_parses(sample_cfg, tmp_path):
    t = run_mission(sample_cfg, "phantom")
    out = write_outputs(t, tmp_path / "o", tmp_path / "ui")
    lines = (out / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    parsed = [json.loads(l) for l in lines]
    assert parsed[0]["type"] == "header" and parsed[-1]["type"] == "evaluation"
    assert len(parsed) == len(t["events"]) + 3
    assert (out / "report.md").read_text(encoding="utf-8").startswith("# Mission report")
    assert (tmp_path / "ui" / "mission.js").read_text(encoding="utf-8").startswith("window.PHANTOM_MISSION")


def test_evaluation_never_appears_inside_agent_events(decoy_cfg):
    t = run_mission(decoy_cfg, "phantom")
    events_text = json.dumps(t["events"])
    assert "true_reliability" not in events_text and "decoys" not in events_text
