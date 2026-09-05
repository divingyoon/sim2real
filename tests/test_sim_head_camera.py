"""실기 카메라 → sim 카메라 이식 사양 테스트."""

import numpy as np
import pytest

from sim_head_camera import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    HeadCameraSpec,
    build_spec,
    quat_wxyz_from_matrix,
    render_isaac_snippet,
)

K = [606.604, 0.0, 320.020, 0.0, 605.652, 240.574, 0.0, 0.0, 1.0]


def test_quat_roundtrip_identity():
    assert quat_wxyz_from_matrix(np.eye(3)) == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_quat_is_unit():
    R = np.array([[-0.0184, -0.9997, -0.0181],
                  [-0.9998, +0.0185, -0.0052],
                  [+0.0056, +0.0180, -0.9998]])
    # 실측값은 완전한 직교가 아니므로 가장 가까운 회전으로 투영해 쓴다
    u, _, vt = np.linalg.svd(R)
    q = quat_wxyz_from_matrix(u @ vt)
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-9)


def test_quat_matches_known_rotation():
    """z축 +90°: (w,x,y,z) = (cos45, 0, 0, sin45)."""
    c = np.cos(np.pi / 4)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert quat_wxyz_from_matrix(R) == pytest.approx([c, 0.0, 0.0, c], abs=1e-9)


def _spec(**kw) -> HeadCameraSpec:
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    T[:3, 3] = [0.05, 0.05, 0.01]
    return build_spec(T, K, **kw)


def test_spec_carries_translation_unchanged():
    """★offset 은 head_camera 링크 기준이므로 T_neck_cam 병진 그대로다."""
    assert _spec().pos == pytest.approx([0.05, 0.05, 0.01])


def test_spec_defaults_to_measured_resolution():
    s = _spec()
    assert (s.width, s.height) == (DEFAULT_WIDTH, DEFAULT_HEIGHT) == (640, 480)


def test_spec_keeps_intrinsics_verbatim():
    """K 는 Isaac 의 from_intrinsic_matrix 가 변환한다 — 우리가 미리 손대지 않는다."""
    assert _spec().intrinsic_matrix == pytest.approx(K)


def test_rejects_non_rigid_transform():
    bad = np.eye(4)
    bad[:3, :3] *= 2.0
    with pytest.raises(ValueError, match="회전"):
        build_spec(bad, K)


def test_rejects_malformed_intrinsics():
    with pytest.raises(ValueError, match="K"):
        build_spec(np.eye(4), [1.0, 2.0, 3.0])


def test_snippet_mentions_ros_convention_and_link():
    """★convention='ros' 가 빠지면 카메라가 엉뚱한 데를 본다."""
    text = render_isaac_snippet(_spec())
    assert 'convention="ros"' in text
    assert "head_camera" in text
    assert "from_intrinsic_matrix" in text
