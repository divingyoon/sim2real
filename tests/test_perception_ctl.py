"""perception_ctl 순수부(build_payload) — start 는 --viewer 없이 뷰어를 건드리지 않는다."""
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from object_registry import DEFAULT_REGISTRY, load_registry  # noqa: E402
from perception_ctl import build_payload  # noqa: E402
from perception_launcher_core import Command, RemoteState, parse_command, plan_actions  # noqa: E402

REG = load_registry(DEFAULT_REGISTRY)


def test_start_without_viewer_flag_omits_key_and_keeps_viewer_running():
    payload = build_payload(Namespace(op="start", objects=["cup_big_s080"], viewer=False), REG)
    assert payload == {"op": "start", "objects": ["cup_big_s100"]}
    import json
    cmd = parse_command(json.dumps(payload), REG)
    assert cmd.viewer is None
    state = RemoteState(camera_up=True, containers={"fpp_cup_big_s100": "Up 1 minute"}, viewer_up=True)
    assert plan_actions(cmd, state) == []


def test_start_with_viewer_flag_and_stop_payloads():
    assert build_payload(Namespace(op="start", objects=["shaker_closed"], viewer=True), REG) == {
        "op": "start", "objects": ["shaker_closed"], "viewer": True}
    assert build_payload(Namespace(op="stop", camera=True), REG) == {"op": "stop", "camera": True}
    assert build_payload(Namespace(op="viewer", on="off"), REG) == {"op": "viewer", "on": False}
    assert build_payload(Namespace(op="status"), REG) is None
