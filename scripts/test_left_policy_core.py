"""좌 v2 정책 코어 테스트 — 한 tick 의 규약을 못 박는다.

정책·fabric 은 가짜를 주입해 **배선과 규약**만 검사한다. 실제 정책 품질은 여기서
다루지 않는다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from left_policy_core import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    LeftPolicyCore,
    LeftSensors,
    gripper_command,
    home_from_run,
)

SIM2REAL = Path(__file__).resolve().parents[1]
RUN_YAML = SIM2REAL / "logs/policy/left_v2E29/params/env.yaml"
URDF = Path("/home/user/rl_ws/urdf/generated/rl/openarm_tesollo_sensor_rl.urdf")

pytestmark = pytest.mark.skipif(
    not (RUN_YAML.exists() and URDF.exists()), reason="dump 또는 URDF 없음")

GOAL7 = np.array([0.35, 0.20, 0.35, 1.0, 0.0, 0.0, 0.0])


def _sensors(core, *, cup=None, grip=GRIPPER_OPEN):
    return LeftSensors(
        arm_q=core.home, arm_qd=np.zeros(7), grip_q=grip, grip_qd=0.0,
        cup_pos=np.array([0.35, 0.20, 0.30]) if cup is None else np.asarray(cup),
        cup_quat=np.array([1.0, 0.0, 0.0, 0.0]),
    )


def _core(action=None, fabric=None):
    act = np.zeros(7) if action is None else np.asarray(action, dtype=float)
    return LeftPolicyCore(policy=lambda obs: act, fabric=fabric,
                          run_env_yaml=RUN_YAML, goal7=GOAL7)


# ── 그리퍼 규약 ────────────────────────────────────────────────────────────
def test_negative_action_closes_when_gate_open():
    """IsaacLab BinaryJointAction: a<0 이 **닫기**. 부호를 뒤집으면 잡을 때 손이 열린다."""
    assert gripper_command(-0.5, gate_open=True) == GRIPPER_CLOSE


def test_positive_action_opens_when_gate_open():
    assert gripper_command(+0.5, gate_open=True) == GRIPPER_OPEN


def test_zero_action_opens():
    assert gripper_command(0.0, gate_open=True) == GRIPPER_OPEN


def test_closed_gate_forces_open_regardless_of_action():
    """접근 성공 전에는 그리퍼를 못 닫는다 — 그 제약 위에서 정책이 학습됐다."""
    assert gripper_command(-1.0, gate_open=False) == GRIPPER_OPEN


# ── 홈 ────────────────────────────────────────────────────────────────────
def test_home_comes_from_run_dump_not_source_constant():
    """소스 상수는 v1 트랙 홈(j4 +0.9336)이다. dump 는 v2(j4 +0.5665)."""
    home = home_from_run(RUN_YAML)
    assert home.shape == (7,)
    assert home[3] == pytest.approx(0.5665, abs=1e-4)
    assert home[6] == pytest.approx(-0.8304, abs=1e-4)


# ── tick 배선 ──────────────────────────────────────────────────────────────
def test_tick_produces_49d_obs_and_7d_action():
    core = _core()
    out = core.step(_sensors(core))
    assert out.obs.shape == (49,)
    assert out.action.shape == (7,)
    assert out.palm_target.shape == (6,)


def test_obs_gate_slot_matches_gate_state():
    """관측 36번째 칸이 게이트다 — 이 칸이 늘 0 이면 정책이 다른 상태를 본다."""
    core = _core()
    out = core.step(_sensors(core, cup=[1.5, 1.5, 1.5]))     # 컵이 멀다
    assert out.gate_open is False
    assert out.obs[35] == 0.0


def test_gate_opens_when_cup_sits_at_the_jaw_and_obs_follows():
    from left_grasp_gate import quat_to_matrix
    core = _core()
    poses = core.fk.poses(core.home, GRIPPER_OPEN, GRIPPER_OPEN)
    approach = quat_to_matrix(poses.base_quat)[:, 2]
    mid = 0.5 * (poses.finger_l_pos + poses.finger_r_pos) + approach * 0.0319
    cup = mid - np.array([0.0, 0.0, -0.5 * (0.08209 + 0.00709)])
    out = core.step(_sensors(core, cup=cup))
    assert out.gate_open is True
    assert out.obs[35] == 1.0


def test_last_action_is_fed_back_into_next_obs():
    """obs 28..34 는 직전 액션이다. 안 물리면 정책이 자기 이력을 못 본다."""
    act = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.4])
    core = _core(act)
    first = core.step(_sensors(core))
    assert np.allclose(first.obs[28:35], 0.0), "첫 스텝의 직전 액션은 0"
    second = core.step(_sensors(core))
    assert np.allclose(second.obs[28:35], act)


def test_reset_clears_last_action_and_gate():
    core = _core(np.array([0.1, 0.1, 0.1, 0, 0, 0, -1.0]))
    core.step(_sensors(core))
    core.reset()
    assert core.step_count == 0
    out = core.step(_sensors(core))
    assert np.allclose(out.obs[28:35], 0.0)


def test_palm_target_is_absolute_box_center_for_zero_action():
    core = _core()
    out = core.step(_sensors(core))
    assert out.palm_target[0] == pytest.approx(0.5 * (0.22 + 0.60))
    assert out.palm_target[1] == pytest.approx(0.5 * (0.10 + 0.43))
    assert out.palm_target[2] == pytest.approx(0.5 * (0.16 + 0.60))


def test_fabric_output_is_returned_as_arm_target():
    want = np.arange(7, dtype=float)
    core = _core(fabric=lambda palm: want)
    out = core.step(_sensors(core))
    assert np.allclose(out.arm_q_target, want)


def test_missing_fabric_yields_nan_target_not_silent_zero():
    """fabric 없이 돌면 목표는 NaN 이어야 한다 — 0 을 내면 팔이 차렷으로 튄다."""
    out = _core().step(_sensors(_core()))
    assert np.all(np.isnan(out.arm_q_target))


def test_rejects_wrong_action_dim():
    core = LeftPolicyCore(policy=lambda obs: np.zeros(5), fabric=None,
                          run_env_yaml=RUN_YAML, goal7=GOAL7)
    with pytest.raises(ValueError):
        core.step(_sensors(core))


def test_diag_carries_jaw_geometry():
    core = _core()
    out = core.step(_sensors(core))
    assert out.diag["lateral"] is not None and out.diag["along"] is not None
    assert out.diag["step"] == 1
