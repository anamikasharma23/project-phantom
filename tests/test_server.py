"""Live console API: config validation, step-by-step runner, no truth leaks."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from phantom.config import ConfigError
from phantom.mission import MissionRunner, run_mission
from phantom.server import Handler, build_config


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    import phantom.server as srv
    old = srv.OUTPUT_DIR
    srv.OUTPUT_DIR = tmp_path_factory.mktemp("live_outputs")     # never write into the real outputs/
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    srv.OUTPUT_DIR = old


def call(base, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_step_runner_matches_run_mission(decoy_cfg):
    runner = MissionRunner(decoy_cfg, "phantom")
    while not runner.done:
        runner.step()
    assert json.dumps(runner.trace, sort_keys=True) == json.dumps(run_mission(decoy_cfg, "phantom"), sort_keys=True)


def test_runner_has_no_truth_until_finished(decoy_cfg):
    runner = MissionRunner(decoy_cfg, "phantom")
    for _ in range(5):
        runner.step()
    assert "evaluation" not in runner.trace and "true_reliability" not in json.dumps(runner.trace)


@pytest.mark.parametrize("req, msg", [
    ({"preset": "file:sample_mission", "battery_capacity": 5}, "between 20 and 400"),
    ({"preset": "file:sample_mission", "time_budget": 5000}, "between 20 and 1000"),
    ({"preset": "file:sample_mission", "seed": "abc"}, "must be a number"),
    ({"preset": "file:../secrets"}, "unknown preset"),
    ({"preset": "scenario:nope"}, "unknown preset"),
])
def test_build_config_rejects_bad_input(req, msg):
    with pytest.raises(ConfigError, match=msg):
        build_config(req)


def test_build_config_applies_overrides():
    cfg = build_config({"preset": "scenario:decoy", "seed": 77, "battery_capacity": 90, "time_budget": 150})
    assert cfg["mission"]["seed"] == 77 and cfg["resources"]["battery_capacity"] == 90
    assert cfg["resources"]["start_battery"] == 90 and cfg["resources"]["time_budget"] == 150


def test_api_session_step_finish(server, sample_cfg):
    code, info = call(server, "/api/info")
    assert code == 200 and info["live"] and any(p["id"] == "file:sample_mission" for p in info["presets"])
    code, s = call(server, "/api/session", {"preset": "file:sample_mission", "agent": "phantom"})
    assert code == 200 and len(s["trace"]["events"]) == 1
    assert "evaluation" not in s["trace"] and "true_reliability" not in json.dumps(s)
    code, st = call(server, "/api/step", {"session": s["session"], "count": 3})
    assert code == 200 and [e["step"] for e in st["events"]] == [1, 2, 3] and not st["done"]
    assert "true_reliability" not in json.dumps(st)
    code, fin = call(server, "/api/finish", {"session": s["session"]})
    assert code == 200 and fin["done"] and fin["report"]["outcome"] == "EVE located"
    offline = run_mission(sample_cfg, "phantom")
    assert 3 + len(fin["events"]) == len(offline["events"]) - 1


def test_api_errors_are_readable(server):
    assert call(server, "/api/step", {"session": "nope"})[0] == 404
    code, e = call(server, "/api/session", {"preset": "file:sample_mission", "battery_capacity": 1})
    assert code == 400 and "battery capacity" in e["error"]
