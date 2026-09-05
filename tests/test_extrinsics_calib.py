import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from extrinsics_calib import (
    qr_object_points, solve_qr_pose, compose_base_cam,
    mat_to_pos_quat_wxyz, reprojection_residual_px,
)

K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])


def _rt(rvec, t):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def _project(T_cam_qr, qr_size):
    obj = qr_object_points(qr_size)
    cam = (T_cam_qr[:3, :3] @ obj.T + T_cam_qr[:3, 3:4]).T
    uv = (K @ cam.T).T
    return uv[:, :2] / uv[:, 2:3]


def test_solve_recovers_known_pose():
    T_true = _rt([0.05, -0.1, 0.02], [0.03, -0.02, 0.7])
    corners = _project(T_true, 0.08)
    T_est = solve_qr_pose(corners, 0.08, K, None)
    assert np.allclose(T_est[:3, 3], T_true[:3, 3], atol=2e-3)
    assert np.allclose(T_est[:3, :3], T_true[:3, :3], atol=2e-3)


def test_reprojection_residual_small_for_true_pose():
    T_true = _rt([0.0, 0.0, 0.0], [0.0, 0.0, 0.6])
    corners = _project(T_true, 0.08)
    assert reprojection_residual_px(corners, T_true, 0.08, K, None) < 0.5


def test_compose_base_cam():
    # T_base_qr는 비-단위 변환(회전+평행이동)으로 고정해 합성 순서를 못박는다.
    T_base_qr = _rt([0.1, 0.2, -0.3], [0.5, -0.4, 1.2])
    T_cam_qr = _rt([0, 0, 0], [0.0, 0.0, 0.7])
    T_base_cam = compose_base_cam(T_base_qr, T_cam_qr)
    # T_base_cam ∘ T_cam_qr == T_base_qr
    assert np.allclose(T_base_cam @ T_cam_qr, T_base_qr, atol=1e-9)


def test_mat_to_pos_quat_wxyz_identity():
    pos, quat = mat_to_pos_quat_wxyz(np.eye(4))
    assert np.allclose(pos, [0, 0, 0])
    assert np.allclose(quat, [1, 0, 0, 0])       # wxyz
