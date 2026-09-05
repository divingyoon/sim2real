# scripts/test_calibrate_camera_extrinsics.py
from pathlib import Path
import numpy as np
import pytest

from calibrate_camera_extrinsics import (
    average_corners, detect_aruco_corners, quat_wxyz_to_matrix, rpy_to_matrix,
    update_camera_extrinsics_yaml,
)

SAMPLE = """\
# 주석 헤더 보존 확인
camera:
  frame: camera_color_optical_frame
  position: [0.0, 0.0, 0.0]
  orientation_wxyz: [1.0, 0.0, 0.0, 0.0]

# cad_to_body 주석 보존
cad_to_body:
  position: [0.0, 0.0, 0.0]
  orientation_wxyz: [0.707107, 0.707107, 0.0, 0.0]

base_frame: base_link
"""

SAMPLE_CAD_FIRST = """\
cad_to_body:
  position: [0.1, 0.2, 0.3]
  orientation_wxyz: [0.707107, 0.707107, 0.0, 0.0]

camera:
  frame: camera_color_optical_frame
  position: [0.0, 0.0, 0.0]
  orientation_wxyz: [1.0, 0.0, 0.0, 0.0]

base_frame: base_link
"""

SAMPLE_NO_CAMERA = """\
cad_to_body:
  position: [0.0, 0.0, 0.0]
  orientation_wxyz: [1.0, 0.0, 0.0, 0.0]

base_frame: base_link
"""


def test_update_preserves_other_blocks(tmp_path):
    p = tmp_path / "ext.yaml"
    p.write_text(SAMPLE)
    out = update_camera_extrinsics_yaml(str(p), [0.1, -0.2, 0.9],
                                        [0.5, 0.5, -0.5, 0.5])
    assert "0.707107" in out                         # cad_to_body 보존
    assert "# cad_to_body 주석 보존" in out           # 주석 보존
    assert "base_frame: base_link" in out
    # camera 블록만 갱신
    import yaml
    data = yaml.safe_load(out)
    assert np.allclose(data["camera"]["position"], [0.1, -0.2, 0.9])
    assert np.allclose(data["camera"]["orientation_wxyz"], [0.5, 0.5, -0.5, 0.5])
    assert np.allclose(data["cad_to_body"]["orientation_wxyz"],
                       [0.707107, 0.707107, 0.0, 0.0])


def test_update_only_touches_camera_block(tmp_path):
    p = tmp_path / "ext.yaml"
    p.write_text(SAMPLE)
    out = update_camera_extrinsics_yaml(str(p), [1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
    # cad_to_body.position은 그대로 [0,0,0]
    import yaml
    data = yaml.safe_load(out)
    assert np.allclose(data["cad_to_body"]["position"], [0.0, 0.0, 0.0])


def test_update_camera_after_cad_to_body_block(tmp_path):
    """cad_to_body: 가 camera: 보다 먼저 나오고 둘 다 동일 필드를 가질 때
    camera 블록만 갱신되고 cad_to_body는 그대로여야 한다."""
    p = tmp_path / "ext.yaml"
    p.write_text(SAMPLE_CAD_FIRST)
    out = update_camera_extrinsics_yaml(str(p), [9.0, 8.0, 7.0], [0.1, 0.2, 0.3, 0.4])
    import yaml
    data = yaml.safe_load(out)
    assert np.allclose(data["camera"]["position"], [9.0, 8.0, 7.0])
    assert np.allclose(data["camera"]["orientation_wxyz"], [0.1, 0.2, 0.3, 0.4])
    # cad_to_body (camera보다 먼저 나오는 블록)는 변경되지 않아야 함
    assert np.allclose(data["cad_to_body"]["position"], [0.1, 0.2, 0.3])
    assert np.allclose(data["cad_to_body"]["orientation_wxyz"],
                       [0.707107, 0.707107, 0.0, 0.0])


def test_update_raises_when_no_camera_block(tmp_path):
    p = tmp_path / "ext.yaml"
    p.write_text(SAMPLE_NO_CAMERA)
    with pytest.raises(ValueError):
        update_camera_extrinsics_yaml(str(p), [1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])


def test_average_corners_mean_of_two():
    c1 = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    c2 = np.array([[2.0, 2.0], [12.0, 2.0], [12.0, 12.0], [2.0, 12.0]])
    avg = average_corners([c1, c2])
    expected = np.array([[1.0, 1.0], [11.0, 1.0], [11.0, 11.0], [1.0, 11.0]])
    assert avg.shape == (4, 2)
    assert np.allclose(avg, expected)


def test_update_preserves_comments_with_brackets(tmp_path):
    """verify that inline comments containing brackets are preserved during update.

    regression test for: re.sub without count=1 was replacing all bracket groups,
    including those in trailing comments, e.g., '# calibrated [rig A]'.
    """
    sample_with_comment = """\
camera:
  frame: camera_color_optical_frame
  position: [0.0, 0.0, 0.0]  # calibrated [rig A]
  orientation_wxyz: [1.0, 0.0, 0.0, 0.0]  # quat from [baseline]
"""
    p = tmp_path / "ext.yaml"
    p.write_text(sample_with_comment)
    out = update_camera_extrinsics_yaml(str(p), [1.5, 2.5, 3.5], [0.7, 0.7, 0.0, 0.0])

    # verify the values are updated
    import yaml
    data = yaml.safe_load(out)
    assert np.allclose(data["camera"]["position"], [1.5, 2.5, 3.5])
    assert np.allclose(data["camera"]["orientation_wxyz"], [0.7, 0.7, 0.0, 0.0])

    # verify the comments are preserved
    assert "[rig A]" in out, "Comment '[rig A]' should be preserved in position line"
    assert "[baseline]" in out, "Comment '[baseline]' should be preserved in orientation_wxyz line"


def test_rpy_to_matrix_single_axis_matches_elementary_rotation():
    # roll only == Rx
    roll = 0.4
    R = rpy_to_matrix(roll, 0.0, 0.0)
    cr, sr = np.cos(roll), np.sin(roll)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    assert np.allclose(R, Rx)

    # pitch only == Ry
    pitch = -0.6
    R = rpy_to_matrix(0.0, pitch, 0.0)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    assert np.allclose(R, Ry)

    # yaw only == Rz
    yaw = 0.3
    R = rpy_to_matrix(0.0, 0.0, yaw)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(R, Rz)


def test_rpy_to_matrix_compound_matches_rz_ry_rx_composition():
    """regression test for the axis-angle (cv2.Rodrigues) bug: a compound rpy
    (roll, pitch, yaw all nonzero) must equal the ROS/URDF fixed-axis
    composition R = Rz(yaw) @ Ry(pitch) @ Rx(roll), built independently here."""
    roll, pitch, yaw = 0.1, -0.6, 0.3

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    expected = Rz @ Ry @ Rx

    R = rpy_to_matrix(roll, pitch, yaw)
    assert np.allclose(R, expected)

    # sanity: must actually differ from the buggy axis-angle interpretation of
    # the same 3-vector (single rotation about axis (roll,pitch,yaw))
    theta = np.linalg.norm([roll, pitch, yaw])
    axis = np.array([roll, pitch, yaw]) / theta
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    axis_angle_R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    assert not np.allclose(R, axis_angle_R)


def test_rpy_to_matrix_is_valid_rotation():
    R = rpy_to_matrix(0.1, -0.6, 0.3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_quat_wxyz_to_matrix_identity():
    R = quat_wxyz_to_matrix([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(R, np.eye(3))


def test_quat_wxyz_to_matrix_known_90deg_about_z():
    # 90 deg about z: w=cos(45deg), z=sin(45deg)
    half = np.pi / 4.0
    q = [np.cos(half), 0.0, 0.0, np.sin(half)]
    R = quat_wxyz_to_matrix(q)
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(R, expected, atol=1e-9)
    # valid rotation
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_quat_wxyz_to_matrix_roundtrip_orthonormal_for_nonunit_input():
    # not pre-normalized input should still yield a valid rotation matrix
    q = [2.0, 0.0, 2.0, 0.0]
    R = quat_wxyz_to_matrix(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def _make_synthetic_aruco_rgb(marker_id: int = 0, marker_px: int = 200,
                               border_px: int = 40):
    """DICT_6X6_250 마커(marker_id)를 생성해 흰 여백으로 패딩한 3채널 RGB 반환."""
    import cv2
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    try:                                              # OpenCV >= 4.7
        marker_img = cv2.aruco.generateImageMarker(adict, marker_id, marker_px)
    except AttributeError:                            # OpenCV <= 4.6
        marker_img = cv2.aruco.drawMarker(adict, marker_id, marker_px)

    canvas_size = marker_px + 2 * border_px
    canvas = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)
    canvas[border_px:border_px + marker_px, border_px:border_px + marker_px] = marker_img
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)


def test_detect_aruco_corners_finds_synthetic_marker():
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("cv2.aruco")
    rgb = _make_synthetic_aruco_rgb(marker_id=0)

    corners = detect_aruco_corners(rgb, "DICT_6X6_250", 0)

    assert corners is not None
    assert corners.shape == (4, 2)
    h, w = rgb.shape[:2]
    assert np.all(corners[:, 0] >= 0) and np.all(corners[:, 0] <= w)
    assert np.all(corners[:, 1] >= 0) and np.all(corners[:, 1] <= h)


def test_detect_aruco_corners_returns_none_for_wrong_marker_id():
    pytest.importorskip("cv2")
    pytest.importorskip("cv2.aruco")
    rgb = _make_synthetic_aruco_rgb(marker_id=0)

    corners = detect_aruco_corners(rgb, "DICT_6X6_250", 5)

    assert corners is None
