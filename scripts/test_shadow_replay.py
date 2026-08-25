"""재생기가 실기에 무엇을 내보내는가 — ROS 없이 검증한다.

노드 자체는 rclpy 를 필요로 하지만, **무엇을 보낼지 정하는 부분**은 순수 코드다:
프로필에서 계약을 읽고, 기록과 맞는지 확인하고, canonical→source 로 리맵한다.
그 세 곳이 이 태스크의 알려진 함정과 정확히 겹친다.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from jtc_bridge_core import JointRemap  # noqa: E402
from robot_profile import load_robot_profile  # noqa: E402
from shadow_replay import ABORT_TRACKING_ERR_RAD, build_plan, describe  # noqa: E402

PROFILE = "gripper_left"


@pytest.fixture(scope="module")
def profile():
    return load_robot_profile(PROFILE)


def _npz(tmp_path, *, joints=None, grip=("l_hj_gripper_1", "l_hj_gripper_2"), frames=20):
    joints = joints or [f"l_aj_{i}" for i in range(1, 8)]
    path = tmp_path / "sim.npz"
    np.savez_compressed(
        path,
        arm_target=np.zeros((frames, 1, len(joints)), dtype=np.float32),
        grip_cmd=np.tile(np.array([0.02, 0.02], dtype=np.float32), (frames, 1, 1))[:, :, : len(grip)],
        meta_joint_names=np.array(joints),
        meta_grip_names=np.array(list(grip)),
        meta_step_dt=np.array([1 / 60]),
    )
    return path


def test_the_plan_takes_the_jaw_the_profile_names(tmp_path, profile):
    """sim 은 두 조를 다 지령하지만 실기 URDF 는 mimic 이 살아 있어 한 조면 따라온다."""
    plan = build_plan(_npz(tmp_path), rate_scale=1.0, profile=profile)

    assert plan.gripper_name == "l_hj_gripper_1"
    assert plan.grip_target.shape == (20,)


def test_a_recording_whose_joint_order_differs_is_refused(tmp_path, profile):
    """순서가 다른데 그냥 흘려보내면 관절이 뒤바뀐 채로 로봇이 움직인다."""
    shuffled = [f"l_aj_{i}" for i in (2, 1, 3, 4, 5, 6, 7)]

    with pytest.raises(SystemExit, match="팔 관절 순서"):
        build_plan(_npz(tmp_path, joints=shuffled), rate_scale=1.0, profile=profile)


def test_a_recording_without_the_profiles_jaw_is_refused(tmp_path, profile):
    with pytest.raises(SystemExit, match="그리퍼"):
        build_plan(_npz(tmp_path, grip=("l_hj_other",)), rate_scale=1.0, profile=profile)


def test_the_plan_warns_when_it_demands_more_than_the_profile_allows(tmp_path, profile):
    """재생 전에 알아야 한다 — 실기 한계를 넘는 요구는 브리지가 조용히 깎는다."""
    path = tmp_path / "fast.npz"
    joints = [f"l_aj_{i}" for i in range(1, 8)]
    target = np.zeros((10, 1, 7), dtype=np.float32)
    target[:, 0, 0] = np.linspace(0.0, 3.0, 10)          # 60 Hz 로 18 rad/s
    np.savez_compressed(
        path, arm_target=target,
        grip_cmd=np.zeros((10, 1, 2), dtype=np.float32),
        meta_joint_names=np.array(joints),
        meta_grip_names=np.array(["l_hj_gripper_1", "l_hj_gripper_2"]),
        meta_step_dt=np.array([1 / 60]),
    )
    plan = build_plan(path, rate_scale=1.0, profile=profile)

    text = describe(plan, profile)

    assert "요구가 한계를 넘는다" in text
    assert "--rate-scale" in text


def test_slowing_the_replay_brings_the_demand_under_the_limit(tmp_path, profile):
    path = tmp_path / "fast.npz"
    joints = [f"l_aj_{i}" for i in range(1, 8)]
    target = np.zeros((10, 1, 7), dtype=np.float32)
    target[:, 0, 0] = np.linspace(0.0, 3.0, 10)
    np.savez_compressed(
        path, arm_target=target,
        grip_cmd=np.zeros((10, 1, 2), dtype=np.float32),
        meta_joint_names=np.array(joints),
        meta_grip_names=np.array(["l_hj_gripper_1", "l_hj_gripper_2"]),
        meta_step_dt=np.array([1 / 60]),
    )

    assert "요구가 한계를 넘는다" not in describe(
        build_plan(path, rate_scale=0.05, profile=profile), profile)


def test_the_arm_remap_is_identity_for_this_profile(profile):
    """좌팔은 sign 이 전부 +1 이고 순서도 같다 — 그래도 매핑을 통과시켜 확인한다."""
    remap = JointRemap(list(profile.arm_canonical), list(profile.arm_source),
                       profile.joint_limits)
    values = np.array([0.0, -0.3, 0.0, 0.9, -0.4, 0.0, -0.3])

    assert remap.apply(values) == pytest.approx(values)


def test_the_gripper_remap_clamps_to_the_real_stroke(profile):
    remap = JointRemap(["l_hj_gripper_1"], list(profile.ee_source), profile.joint_limits)

    assert remap.apply(np.array([0.10]))[0] == pytest.approx(0.044)
    assert remap.apply(np.array([-0.01]))[0] == pytest.approx(0.0)


def test_without_execute_the_replayer_publishes_nothing(tmp_path, profile):
    """robotctl 과 같은 규약 — `--execute` 가 없으면 wire 에 아무것도 올리지 않는다.

    rclpy import 자체가 `--execute` 뒤에 있으므로, ROS 없는 환경에서 dry-run 이 도는지가
    그 구조의 증거다.
    """
    result = subprocess.run(
        [sys.executable, str(_HERE / "shadow_replay.py"), "--sim", str(_npz(tmp_path)),
         "--robot", PROFILE, "--rate-scale", "0.25"],
        capture_output=True, text=True, timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert "발행" in result.stdout


def test_the_abort_threshold_is_tighter_than_the_droop_we_predict():
    """중단 문턱이 예측 처짐보다 낮으면 정상 동작에서 매번 멈춘다.

    §6-2 좌팔 예측: 펌웨어 게인에서 최악 관절 정착 오차 69.5 mrad. 중단은 그보다
    넉넉히 위(300 mrad)에 두되, 관절 한계를 향해 달려가는 실패는 잡아야 한다.
    """
    predicted_droop_rad = 0.0695

    assert ABORT_TRACKING_ERR_RAD > predicted_droop_rad * 3
    assert ABORT_TRACKING_ERR_RAD < 0.5
