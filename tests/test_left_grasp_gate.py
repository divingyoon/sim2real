"""좌팔 그리퍼 게이트 이식 테스트.

관측 36번째 칸을 만드는 술어라, 여기가 틀리면 정책은 죽지 않고 **조용히 다른 상태를
본다**. 학습 코드가 실측으로 못 박은 성질을 그대로 검사한다.
"""
from __future__ import annotations

import numpy as np
import pytest

from left_grasp_gate import (
    CUP_GRASP_BAND_AXIS,
    JAW_PAD_OFFSET,
    GateCfg,
    GraspGate,
    grasp_ok,
    jaw_frame,
    quat_to_matrix,
)

IDENT = np.array([1.0, 0.0, 0.0, 0.0])       # (w,x,y,z)


def _kw(*, cup=(0.0, 0.0, 0.0), fl=(0.0, -0.03, 0.0), fr=(0.0, 0.03, 0.0),
        base_q=IDENT, cup_q=IDENT):
    return dict(finger_l_pos=fl, finger_r_pos=fr, gripper_base_quat=base_q,
                cup_pos=cup, cup_quat=cup_q)


def test_quat_to_matrix_identity():
    assert np.allclose(quat_to_matrix(IDENT), np.eye(3))


def test_quat_to_matrix_is_orthonormal():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    R = quat_to_matrix(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_pad_offset_shifts_along_base_z():
    """패드 중앙 보정 — 손가락 원점이 아니라 패드가 기준선이다(32 mm 차이)."""
    f = jaw_frame(**_kw())
    assert np.isclose(f.mid[2], JAW_PAD_OFFSET)


def test_jaw_axis_is_unit_and_points_left_to_right():
    f = jaw_frame(**_kw())
    assert np.isclose(np.linalg.norm(f.u), 1.0)
    assert f.u[1] > 0.9


def test_cup_centered_between_jaws_opens_gate():
    """컵이 턱 사이 파지 대역 안 — 게이트가 열려야 한다."""
    cup = (0.0, 0.0, JAW_PAD_OFFSET - CUP_GRASP_BAND_AXIS[0] - 0.02)
    f = jaw_frame(**_kw(cup=cup))
    assert f.in_band, f
    assert grasp_ok(f)


def test_far_sideways_cup_is_rejected_even_if_axis_passes_between_jaws():
    """턱이 벌어져 있으면 '축이 턱 사이를 지난다'는 판별력이 없다 — lateral 이 기준."""
    cup = (0.085, 0.0, JAW_PAD_OFFSET - CUP_GRASP_BAND_AXIS[0] - 0.02)
    f = jaw_frame(**_kw(cup=cup))
    assert f.lateral > 0.08
    assert not grasp_ok(f)


def test_cup_far_above_band_is_rejected():
    """컵 위 허공에서 감싸도 성립하면 안 된다 — clamp 전 축좌표로 판정."""
    cup = (0.0, 0.0, -0.20)                       # 턱이 컵 원점보다 한참 위
    f = jaw_frame(**_kw(cup=cup))
    assert not f.in_band
    assert not grasp_ok(f)


def test_band_uses_raw_axis_not_clamped():
    """clamp 된 값으로 판정하면 밖에 있어도 경계로 접혀 들어와 항상 참이 된다."""
    cup = (0.0, 0.0, -0.20)
    f = jaw_frame(**_kw(cup=cup))
    assert not (CUP_GRASP_BAND_AXIS[0] < f.axis_t_raw < CUP_GRASP_BAND_AXIS[1])
    # clamp 는 컵 원점 기준 대역으로 접는다 — 최근접점이 경계에 정확히 놓인다
    assert f.cup_pt[2] == pytest.approx(cup[2] + CUP_GRASP_BAND_AXIS[1])


def test_along_rejects_cup_shifted_along_jaw_axis():
    cup = (0.0, 0.05, JAW_PAD_OFFSET - CUP_GRASP_BAND_AXIS[0] - 0.02)
    f = jaw_frame(**_kw(cup=cup))
    assert f.along > GateCfg().along_ok
    assert not grasp_ok(f)


def test_gate_latches_and_stays_open():
    g = GraspGate()
    good = _kw(cup=(0.0, 0.0, JAW_PAD_OFFSET - CUP_GRASP_BAND_AXIS[0] - 0.02))
    bad = _kw(cup=(0.085, 0.0, JAW_PAD_OFFSET - CUP_GRASP_BAND_AXIS[0] - 0.02))
    assert g.update(**good)
    assert g.update(**bad), "래치는 유지돼야 한다"
    assert g.obs_value == 1.0


def test_gate_releases_when_configured_and_cup_leaves():
    g = GraspGate(GateCfg(release_lateral=0.05))
    good = _kw(cup=(0.0, 0.0, JAW_PAD_OFFSET - CUP_GRASP_BAND_AXIS[0] - 0.02))
    bad = _kw(cup=(0.085, 0.0, JAW_PAD_OFFSET - CUP_GRASP_BAND_AXIS[0] - 0.02))
    assert g.update(**good)
    assert not g.update(**bad)


def test_reset_closes_gate():
    g = GraspGate()
    g.update(**_kw(cup=(0.0, 0.0, JAW_PAD_OFFSET - CUP_GRASP_BAND_AXIS[0] - 0.02)))
    g.reset()
    assert not g.is_open
    assert g.obs_value == 0.0


def test_obs_value_starts_closed():
    assert GraspGate().obs_value == 0.0


def test_rejects_zero_quaternion():
    with pytest.raises(ValueError):
        quat_to_matrix([0.0, 0.0, 0.0, 0.0])
