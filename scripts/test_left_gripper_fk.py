"""좌 그리퍼 FK 테스트 — 교차검증이 핵심이다.

이 모듈은 `gripper_base` 까지를 `robot_control` 체인으로 풀고 그 뒤(고정·프리즈매틱)를
직접 합성한다. 그래서 **tcp 를 체인으로 직접 푼 값**과 일치해야 한다 — 서로 독립적인
두 경로가 같은 점을 가리키면 오프셋 합성이 맞다는 뜻이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from left_gripper_fk import DEFAULT_URDF, LeftGripperFK, quat_from_matrix

sys.path.insert(0, "/home/user/rl_ws/robot_control/src")

pytestmark = pytest.mark.skipif(not Path(DEFAULT_URDF).exists(), reason="URDF 없음")

HOME = np.array([-0.0136, -0.3255, -0.0010, 0.5665, -0.4655, 0.0088, -0.8304])


@pytest.fixture(scope="module")
def fk():
    return LeftGripperFK()


@pytest.fixture(scope="module")
def tcp_chain():
    from robot_control.kinematics import chain_from_urdf
    return chain_from_urdf(Path(DEFAULT_URDF).read_text(),
                           [f"l_aj_{i}" for i in range(1, 8)], "l_hl_gripper_tcp")


def test_tcp_matches_independent_chain(fk, tcp_chain):
    """★교차검증 — base+오프셋 경로와 tcp 직접 경로가 같은 점을 내야 한다."""
    for q in (HOME, np.zeros(7), HOME * 0.5):
        got = fk.poses(q, 0.0, 0.0).tcp_pos
        want = tcp_chain.pose(q)[:3, 3]
        assert np.allclose(got, want, atol=1e-9), (q, got, want)


def test_home_tcp_is_the_measured_value(fk):
    """오늘 중력 모델에서 쓴 홈 TCP 와 같아야 한다(회귀 고정)."""
    tcp = fk.poses(HOME, 0.0, 0.0).tcp_pos
    assert np.allclose(tcp, [0.2863, 0.3523, 0.2952], atol=5e-4)


def test_base_quat_is_unit(fk):
    q = fk.poses(HOME, 0.0, 0.0).base_quat
    assert np.isclose(np.linalg.norm(q), 1.0)


def test_fingers_are_symmetric_about_base(fk):
    """두 손가락은 base 의 ±y 로 같은 거리만큼 벌어진다."""
    p = fk.poses(HOME, 0.02, 0.02)
    mid = 0.5 * (p.finger_l_pos + p.finger_r_pos)
    # 중점은 손가락이 벌어져도 base 축 위에 남는다
    p0 = fk.poses(HOME, 0.0, 0.0)
    mid0 = 0.5 * (p0.finger_l_pos + p0.finger_r_pos)
    assert np.allclose(mid, mid0, atol=1e-12)


def test_opening_increases_finger_separation(fk):
    closed = fk.poses(HOME, 0.0, 0.0)
    opened = fk.poses(HOME, 0.044, 0.044)
    d0 = np.linalg.norm(closed.finger_l_pos - closed.finger_r_pos)
    d1 = np.linalg.norm(opened.finger_l_pos - opened.finger_r_pos)
    assert d1 - d0 == pytest.approx(2 * 0.044, abs=1e-9)


def test_closed_separation_is_urdf_offset(fk):
    p = fk.poses(HOME, 0.0, 0.0)
    d = np.linalg.norm(p.finger_l_pos - p.finger_r_pos)
    assert d == pytest.approx(2 * 0.006, abs=1e-9)


def test_tcp_is_ahead_of_base_along_approach_axis(fk):
    """tcp 는 base 의 +z(접근축) 로 80 mm 앞에 있다."""
    p = fk.poses(HOME, 0.0, 0.0)
    from left_grasp_gate import quat_to_matrix
    approach = quat_to_matrix(p.base_quat)[:, 2]
    assert np.dot(p.tcp_pos - p.base_pos, approach) == pytest.approx(0.08, abs=1e-9)


def test_rejects_wrong_arm_dim(fk):
    with pytest.raises(ValueError):
        fk.poses(np.zeros(6), 0.0, 0.0)


def test_quat_from_matrix_roundtrip():
    from left_grasp_gate import quat_to_matrix
    for ang in (0.1, 1.0, 2.5, 3.0):
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        assert np.allclose(quat_to_matrix(quat_from_matrix(R)), R, atol=1e-12)


def test_gate_opens_when_cup_sits_at_the_jaw(fk):
    """FK + 게이트 통합 — 턱 사이 파지 위치에 컵을 두면 게이트가 열린다."""
    from left_grasp_gate import GraspGate, quat_to_matrix
    p = fk.poses(HOME, 0.02, 0.02)
    approach = quat_to_matrix(p.base_quat)[:, 2]
    mid = 0.5 * (p.finger_l_pos + p.finger_r_pos) + approach * 0.0319
    # 컵 축을 world z 로 두고, 파지 대역 중앙이 턱 중점에 오도록 원점을 내린다
    band_mid = -0.5 * (0.08209 + 0.00709)
    cup = mid - np.array([0.0, 0.0, band_mid])
    gate = GraspGate()
    assert gate.update(finger_l_pos=p.finger_l_pos, finger_r_pos=p.finger_r_pos,
                       gripper_base_quat=p.base_quat, cup_pos=cup,
                       cup_quat=[1.0, 0.0, 0.0, 0.0])
