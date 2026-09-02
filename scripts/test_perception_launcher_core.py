"""perception_launcher_core 순수 로직 검증. ROS·ssh 불필요."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from object_registry import DEFAULT_REGISTRY, load_registry  # noqa: E402
from perception_launcher_core import (  # noqa: E402
    Command, RemoteState, build_status, parse_command, parse_remote_status, plan_actions,
)

REG = load_registry(DEFAULT_REGISTRY)


def test_parse_start_resolves_aliases_and_dedups():
    cmd = parse_command(json.dumps({"op": "start", "objects": ["cup_big_s080", "shaker_closed",
                                                                "cup_big_s100"], "viewer": True}), REG)
    assert cmd == Command(op="start", objects=("cup_big_s100", "shaker_closed"), viewer=True, camera=False)


def test_parse_rejects_bad_input():
    with pytest.raises(ValueError, match="op"):
        parse_command(json.dumps({"objects": ["shaker_closed"]}), REG)
    with pytest.raises(ValueError, match="unknown object"):
        parse_command(json.dumps({"op": "start", "objects": ["teapot"]}), REG)
    with pytest.raises(ValueError, match="objects"):
        parse_command(json.dumps({"op": "start", "objects": []}), REG)
    with pytest.raises(ValueError, match="JSON"):
        parse_command("not json", REG)


def test_parse_stop_and_viewer():
    assert parse_command('{"op":"stop","camera":true}', REG) == Command("stop", (), None, True)
    assert parse_command('{"op":"viewer","on":false}', REG) == Command("viewer", (), False, False)


def test_parse_remote_status():
    st = parse_remote_status('{"camera_up": true, "containers": {"fpp_shaker_closed": "Up 3 minutes"},'
                             ' "viewer_up": false}')
    assert st == RemoteState(camera_up=True, containers={"fpp_shaker_closed": "Up 3 minutes"},
                             viewer_up=False)
    with pytest.raises(ValueError, match="status"):
        parse_remote_status("garbage")


def test_plan_start_is_idempotent_and_prunes_extras():
    state = RemoteState(camera_up=True, containers={"fpp_shaker_closed": "Up 1 minute",
                                                    "fpp_old": "Up 9 minutes"}, viewer_up=False)
    cmd = Command("start", ("shaker_closed", "cup_big_s100"), viewer=True, camera=False)
    assert plan_actions(cmd, state) == [("fpp_down", "fpp_old"), ("fpp_up", "cup_big_s100"),
                                        ("viewer_up",)]


def test_plan_start_cold_brings_camera_first():
    state = RemoteState(camera_up=False, containers={}, viewer_up=False)
    cmd = Command("start", ("shaker_closed",), viewer=None, camera=False)
    assert plan_actions(cmd, state) == [("camera_up",), ("fpp_up", "shaker_closed")]


def test_plan_stop_tears_down_everything_camera_only_when_asked():
    state = RemoteState(camera_up=True, containers={"fpp_a": "Up", "fpp_b": "Exited (1)"}, viewer_up=True)
    assert plan_actions(Command("stop", (), None, False), state) == [
        ("viewer_down",), ("fpp_down", "fpp_a"), ("fpp_down", "fpp_b")]
    assert plan_actions(Command("stop", (), None, True), state)[-1] == ("camera_down",)


def test_plan_viewer_toggle():
    up = RemoteState(True, {}, True)
    assert plan_actions(Command("viewer", (), True, False), up) == []
    assert plan_actions(Command("viewer", (), False, False), up) == [("viewer_down",)]


def test_build_status_shape():
    st = RemoteState(True, {"fpp_shaker_closed": "Up 2 minutes"}, False)
    out = build_status(st, camera_hz=29.9, pose_ages={"shaker_closed": 0.05, "cup_big_s100": None},
                       busy=False, error=None)
    assert out["camera_hz"] == 29.9 and out["camera_up"] is True
    assert out["objects"]["shaker_closed"] == {"container": "Up 2 minutes", "pose_age_s": 0.05}
    assert out["objects"]["cup_big_s100"] == {"container": None, "pose_age_s": None}
    assert out["viewer"] is False and out["busy"] is False and out["error"] is None
    assert build_status(None, 0.0, {}, True, "ssh failed")["error"] == "ssh failed"
