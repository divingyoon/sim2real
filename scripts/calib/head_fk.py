#!/usr/bin/env python3
"""head(목) 2-DOF FK — CAD 기준 T_base_cam 예측 (ChArUco 캘리브 교차검증용).

체인 (openarm_tesollo_sensor_rl.urdf, base=body_link):
  body_link --[fixed 0,0,0.750]-->            head_base
            --[head_j_pan  rev, o(0.0225,0,0.0385), axis(0,0,-1)]--> head_mid
            --[head_j_tilt rev, o(0,-0.0145,0.0275), axis(0,1,0)]--> head_camera
            --[fixed (0.0147,0.0145,0.0365)]-->                     head_cam_view

head_cam_view 는 rpy=0(=body z-up 규약)이라 OpenCV optical(z-전방)과 다름.
optical 로 바꾸려면 T_VIEW_OPTICAL 을 곱한다.
⚠ head_cam_view 가 CAD상 optical 의도인지 미확정 → FK vs ChArUco 비교로 검증.
"""
import numpy as np

# --- URDF 상수 (m, rad-축) ---
MOUNT_XYZ = np.array([0.0, 0.0, 0.750])
PAN_XYZ = np.array([0.0225, 0.0, 0.0385]);  PAN_AXIS = np.array([0.0, 0.0, -1.0])
TILT_XYZ = np.array([0.0, -0.0145, 0.0275]); TILT_AXIS = np.array([0.0, 1.0, 0.0])
CAMVIEW_XYZ = np.array([0.0147, 0.0145, 0.0365])

# view(z-up 로봇규약) → optical(x-우/y-하/z-전방):
#   x_opt = -y_view, y_opt = -z_view, z_opt = +x_view
# 열 = optical 축을 view 프레임에서 표현.
T_VIEW_OPTICAL = np.array([[0.0, 0.0, 1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """축-각 회전 (Rodrigues, 3x3)."""
    a = axis / np.linalg.norm(axis)
    x, y, z = a
    c, s, C = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _tf(xyz: np.ndarray, R: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = xyz
    return T


def head_fk(pan: float, tilt: float) -> np.ndarray:
    """T_body_camview (base=body_link). pan=head_j_pan, tilt=head_j_tilt [rad]."""
    return (_tf(MOUNT_XYZ)
            @ _tf(PAN_XYZ, _rot(PAN_AXIS, pan))
            @ _tf(TILT_XYZ, _rot(TILT_AXIS, tilt))
            @ _tf(CAMVIEW_XYZ))


def head_fk_base_optical(pan: float, tilt: float) -> np.ndarray:
    """T_base_cam (optical 프레임). ChArUco T_base_cam 과 직접 비교용."""
    T = head_fk(pan, tilt).copy()
    T[:3, :3] = T[:3, :3] @ T_VIEW_OPTICAL
    return T
