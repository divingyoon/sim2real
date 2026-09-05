"""URDF 에서 읽은 head 체인 FK 테스트."""

import math

import numpy as np
import pytest

from head_fk_chain import HEAD_URDF, head_chain, t_base_neck


def test_chain_matches_urdf():
    """체인이 URDF 그대로여야 한다 — 상수를 손으로 베끼면 드리프트한다."""
    chain = head_chain()
    assert [j.name for j in chain] == [
        "head_j_mount", "head_j_pan", "head_j_tilt"]
    assert chain[0].xyz == pytest.approx([0.0, 0.0, 0.750])
    assert chain[1].axis == pytest.approx([0.0, 0.0, -1.0])   # pan
    assert chain[2].axis == pytest.approx([0.0, 1.0, 0.0])    # tilt


def test_zero_pose_is_pure_translation():
    T = t_base_neck(0.0, 0.0)
    assert np.allclose(T[:3, :3], np.eye(3), atol=1e-9)
    # head_v1 (2026-09-05): 0.750 + 0.035603 + 0.030400 = 0.816003 · y = 0.000034 − 0.011580
    assert T[:3, 3] == pytest.approx([0.0225, -0.011546, 0.816003], abs=1e-9)


def test_pan_rotates_about_negative_z():
    """axis = (0,0,-1) 이므로 +각도는 z 음의 방향 회전이다."""
    T = t_base_neck(90.0, 0.0)
    # -90° about z: x축이 -y 로 간다
    assert T[:3, 0] == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)


def test_tilt_rotates_about_y():
    T = t_base_neck(0.0, 90.0)
    # +90° about y: x축이 -z 로 간다
    assert T[:3, 0] == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)


def test_tilt_offset_moves_with_pan():
    """tilt 원점이 pan 뒤에 있으므로 pan 이 그것을 끌고 돌아야 한다."""
    straight = t_base_neck(0.0, 0.0)[:3, 3]
    turned = t_base_neck(90.0, 0.0)[:3, 3]
    assert not np.allclose(straight[:2], turned[:2])
    assert straight[2] == pytest.approx(turned[2], abs=1e-9)   # 높이는 같다


def test_is_rigid_transform():
    for pan, tilt in ((0, 0), (12.5, -20.0), (-33.0, 7.5)):
        R = t_base_neck(pan, tilt)[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-9)


def test_urdf_path_exists():
    assert HEAD_URDF.is_file()


# ---------- 인코더 ↔ URDF 각도 규약 (2026-09-01 hand-eye 로 판정) ----------

def test_pan_sign_is_inverted():
    """URDF pan 축이 (0,0,-1) 이라 인코더 양의 방향과 반대다.

    판정 근거: 부호를 뒤집어야 hand-eye 가 보드를 테이블 높이(z=+0.23 m)에 놓는다.
    안 뒤집으면 z=+1.48 m — 카메라(z=0.82)보다 66 cm 위로, 물리적으로 불가능하다.
    """
    from head_fk_chain import urdf_from_encoder
    assert urdf_from_encoder(10.0, -20.0)[0] == pytest.approx(-10.0)


def test_tilt_sign_is_preserved():
    """tilt 를 올리면 보드가 영상에서 위로 간다(카메라가 아래를 본다).

    URDF tilt 축 (0,1,0) 도 양의 각도가 앞을 아래로 내리므로 같은 부호다.
    (dy/dtilt = -0.0106 m/deg 실측, pan≈0 샘플 5개)
    """
    from head_fk_chain import urdf_from_encoder
    assert urdf_from_encoder(10.0, -20.0)[1] == pytest.approx(-20.0)


def test_encoder_from_urdf_is_the_inverse():
    """★ROS joint_states 는 URDF 규약으로 흐른다 — 되돌리는 길이 있어야 한다."""
    from head_fk_chain import encoder_from_urdf, urdf_from_encoder
    for pan, tilt in ((0.0, -20.0), (15.0, -27.0), (-33.0, 7.5)):
        assert encoder_from_urdf(*urdf_from_encoder(pan, tilt)) == pytest.approx((pan, tilt))
