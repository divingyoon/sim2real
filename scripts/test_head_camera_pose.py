"""목 각도 → T_base_cam 계산 테스트 (ROS·하드웨어 무관)."""

import numpy as np
import pytest

from head_camera_pose import (
    DEFAULT_HEAD_EXTRINSICS,
    HOME_PAN_DEG,
    HOME_TILT_DEG,
    base_cam_pose,
    load_neck_to_camera,
)


def test_shipped_calibration_loads():
    T = load_neck_to_camera()
    assert T.shape == (4, 4)
    assert np.allclose(T[3], [0, 0, 0, 1])
    assert DEFAULT_HEAD_EXTRINSICS.is_file()


def test_home_pose_matches_deployed_static_value():
    """★기준 자세 계산이 global_camera_extrinsics.yaml 의 정적값과 같아야 한다.

    둘이 갈리면 목 인식 모드를 켜는 순간 컵 좌표가 조용히 점프한다.
    """
    import yaml
    pos, quat = base_cam_pose(HOME_PAN_DEG, HOME_TILT_DEG)
    cam = yaml.safe_load(
        (DEFAULT_HEAD_EXTRINSICS.parent / "global_camera_extrinsics.yaml")
        .read_text(encoding="utf-8"))["camera"]
    assert pos == pytest.approx(cam["position"], abs=1e-6)
    dot = abs(float(np.dot(quat, np.array(cam["orientation_wxyz"]))))
    assert dot == pytest.approx(1.0, abs=1e-6)      # 부호 모호성 흡수


def test_quaternion_is_unit():
    for pan, tilt in ((0, -20), (15, -30), (-25, -5)):
        assert np.linalg.norm(base_cam_pose(pan, tilt)[1]) == pytest.approx(1.0, abs=1e-9)


def test_pan_moves_camera_laterally():
    """pan 은 수직축 회전이라 높이는 그대로고 xy 만 돈다."""
    a, _ = base_cam_pose(0.0, -20.0)
    b, _ = base_cam_pose(15.0, -20.0)
    assert a[2] == pytest.approx(b[2], abs=1e-9)
    assert not np.allclose(a[:2], b[:2])


def test_tilt_changes_height_but_not_y():
    a, _ = base_cam_pose(0.0, -20.0)
    b, _ = base_cam_pose(0.0, -30.0)
    assert a[1] == pytest.approx(b[1], abs=1e-9)
    assert a[2] != pytest.approx(b[2], abs=1e-6)


def test_pan_sign_follows_encoder_convention():
    """★인코더 부호를 그대로 받는다 — URDF 반전은 내부에서 한다."""
    right, _ = base_cam_pose(+15.0, -20.0)
    left, _ = base_cam_pose(-15.0, -20.0)
    assert right[1] > left[1]        # +pan 이면 카메라가 +y 로 간다


def test_rejects_bad_matrix(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("neck_to_camera:\n  matrix: [[1,0],[0,1]]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="4x4"):
        load_neck_to_camera(p)
