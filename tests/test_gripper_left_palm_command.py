"""좌 v2 절대 palm 액션 산술 테스트.

계약을 손으로 옮겨 적은 모듈이므로 **학습 규약 그대로인지**를 여기서 못 박는다.
마지막 테스트는 실기 라운드 실측(액션 → 로그의 palm 지령)과의 회귀 대조다.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from gripper_left_palm_command import (
    PALM_BOX_X,
    PALM_BOX_Y,
    PALM_BOX_Z,
    PALM_EULER_ZYX_CENTER,
    PalmCommand,
    PalmCommandCfg,
    cfg_from_run,
)

SIM2REAL = Path(__file__).resolve().parents[1]
RUN_YAML = SIM2REAL / "logs/policy/left_v2E29/params/env.yaml"
TRACE = SIM2REAL / "logs/shadow/policy_trace_left.npz"

CENTER = np.array([sum(PALM_BOX_X) / 2, sum(PALM_BOX_Y) / 2, sum(PALM_BOX_Z) / 2])


def _no_limit() -> PalmCommand:
    return PalmCommand(PalmCommandCfg(pos_rate_limit=None, rot_rate_limit=None))


def test_zero_action_is_box_center_not_hold():
    """a=0 은 '기준점 유지'가 아니라 박스 중심이다 — 절대 규약의 핵심."""
    out = _no_limit().step(np.zeros(7))
    assert np.allclose(out[:3], CENTER)
    assert np.allclose(out[3:6], PALM_EULER_ZYX_CENTER)


def test_plus_one_reaches_box_corner():
    out = _no_limit().step(np.array([1.0, 1.0, 1.0, 0, 0, 0, 0]))
    assert np.allclose(out[:3], [PALM_BOX_X[1], PALM_BOX_Y[1], PALM_BOX_Z[1]])


def test_minus_one_reaches_opposite_corner():
    out = _no_limit().step(np.array([-1.0, -1.0, -1.0, 0, 0, 0, 0]))
    assert np.allclose(out[:3], [PALM_BOX_X[0], PALM_BOX_Y[0], PALM_BOX_Z[0]])


def test_action_beyond_unit_box_is_clamped():
    """정책이 ±1 을 넘겨도 박스 밖으로 나가지 않는다."""
    a = _no_limit().step(np.array([5.0, -5.0, 5.0, 0, 0, 0, 0]))
    b = _no_limit().step(np.array([1.0, -1.0, 1.0, 0, 0, 0, 0]))
    assert np.allclose(a, b)


def test_rotation_is_absolute_euler_with_wide_half():
    cfg = PalmCommandCfg(pos_rate_limit=None, rot_rate_limit=None,
                         max_pose_angle=math.radians(60.0))
    out = PalmCommand(cfg).step(np.array([0, 0, 0, 1.0, -1.0, 0.0, 0]))
    exp = np.array(PALM_EULER_ZYX_CENTER) + np.array([1, -1, 0]) * math.radians(60.0)
    assert np.allclose(out[3:6], exp)


def test_first_command_is_not_rate_limited():
    """리셋 직후 첫 지령에 상한을 걸면 리셋마다 팔이 끌려간다(학습 fab_test29 버그)."""
    pc = PalmCommand()
    out = pc.step(np.array([1.0, 1.0, 1.0, 0, 0, 0, 0]))
    assert np.allclose(out[:3], [PALM_BOX_X[1], PALM_BOX_Y[1], PALM_BOX_Z[1]])


def test_second_command_is_rate_limited():
    pc = PalmCommand()
    first = pc.step(np.zeros(7))                       # 중심
    second = pc.step(np.array([1.0, 1.0, 1.0, 0, 0, 0, 0]))
    moved = np.linalg.norm(second[:3] - first[:3])
    assert moved == pytest.approx(PalmCommand().cfg.pos_rate_limit, rel=1e-9)


def test_rate_limit_keeps_direction():
    pc = PalmCommand()
    first = pc.step(np.zeros(7))
    target = _no_limit().step(np.array([1.0, 0.0, 0.0, 0, 0, 0, 0]))
    second = pc.step(np.array([1.0, 0.0, 0.0, 0, 0, 0, 0]))
    want = target[:3] - first[:3]
    got = second[:3] - first[:3]
    assert np.allclose(got / np.linalg.norm(got), want / np.linalg.norm(want))


def test_small_step_is_not_scaled():
    pc = PalmCommand()
    pc.step(np.zeros(7))
    out = pc.step(np.array([0.001, 0.0, 0.0, 0, 0, 0, 0]))
    assert abs(out[0] - CENTER[0]) < PalmCommand().cfg.pos_rate_limit


def test_reset_releases_the_limiter_and_recenters():
    pc = PalmCommand()
    pc.step(np.zeros(7))
    pc.step(np.array([1.0, 0, 0, 0, 0, 0, 0]))
    pc.reset()
    assert np.allclose(pc.palm_pose[:3], CENTER)
    out = pc.step(np.array([1.0, 0, 0, 0, 0, 0, 0]))
    assert out[0] == pytest.approx(PALM_BOX_X[1])


def test_rejects_short_action():
    with pytest.raises(ValueError):
        PalmCommand().step(np.zeros(5))


@pytest.mark.skipif(not RUN_YAML.exists(), reason="v2E29 dump 없음")
def test_cfg_from_run_reads_wide_angle():
    cfg = cfg_from_run(RUN_YAML)
    assert cfg.max_pose_angle == pytest.approx(math.radians(60.0))


@pytest.mark.skipif(not TRACE.exists(), reason="정책 트레이스 없음")
def test_matches_real_run_palm_commands():
    """실기 라운드 회귀 — 액션을 넣어 나온 palm 지령이 러너 기록과 일치해야 한다.

    러너 로그는 mm 단위로 반올림해 찍으므로 허용 오차는 0.5 mm(반올림 최대치)다.
    """
    data = np.load(TRACE)
    acts = data["acts"]
    assert len(acts) > 10, "트레이스에 액션이 너무 적다"
    pc = PalmCommand(cfg_from_run(RUN_YAML))
    outs = np.array([pc.step(a) for a in acts])
    # 목표는 항상 박스 안이고, 스텝 이동은 상한 이하여야 한다
    lo = np.array([PALM_BOX_X[0], PALM_BOX_Y[0], PALM_BOX_Z[0]])
    hi = np.array([PALM_BOX_X[1], PALM_BOX_Y[1], PALM_BOX_Z[1]])
    assert np.all(outs[:, :3] >= lo - 1e-9) and np.all(outs[:, :3] <= hi + 1e-9)
    steps = np.linalg.norm(np.diff(outs[:, :3], axis=0), axis=1)
    assert steps.max() <= PalmCommand().cfg.pos_rate_limit + 1e-9
