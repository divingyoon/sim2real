#!/usr/bin/env python3
"""목 각도로 `T_base_cam` 을 **매번 계산**한다 — 목을 돌려도 컵 좌표가 맞는다.

`global_camera_extrinsics.yaml` 의 `camera:` 블록은 **한 목 자세의 정적 스냅샷**이다.
목이 15° 만 돌아도 카메라가 11 mm 이동하고 크게 회전하므로 그 값은 통째로 틀린다.
여기서는 hand-eye 로 얻은 `T_neck_cam` 과 URDF FK 로 자세마다 다시 만든다:

    T_base_cam(pan, tilt) = T_base_neck(pan, tilt) ∘ T_neck_cam

★입력은 **실기 인코더 각(deg)** 이다. URDF 부호 반전(pan)은 `head_fk_chain` 이 한다 —
호출하는 쪽이 그 규약을 알 필요가 없게 한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from head_fk_chain import t_base_neck_from_encoder

DEFAULT_HEAD_EXTRINSICS = (Path(__file__).resolve().parents[1]
                           / "config" / "head_extrinsics.yaml")
#: `config/head_home.yaml` 의 기준 자세. 정적값과의 정합 검증에 쓴다.
HOME_PAN_DEG, HOME_TILT_DEG = 0.0, -20.0


def load_neck_to_camera(path: Path | str = DEFAULT_HEAD_EXTRINSICS) -> np.ndarray:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    try:
        matrix = np.array(raw["neck_to_camera"]["matrix"], dtype=float)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: neck_to_camera.matrix 가 없다") from exc
    if matrix.shape != (4, 4):
        raise ValueError(f"{path}: neck_to_camera.matrix 는 4x4 여야 한다 — {matrix.shape}")
    return matrix


def quat_wxyz_from_matrix(R: np.ndarray) -> np.ndarray:
    """회전행렬 → (w,x,y,z). 실측 행렬은 완전 직교가 아니라 가장 가까운 회전으로 투영한다."""
    u, _, vt = np.linalg.svd(np.asarray(R, dtype=float))
    m = u @ vt
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s,
             (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
             (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
             0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
             (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q, dtype=float)
    return q / np.linalg.norm(q)


def base_cam_pose(pan_encoder_deg: float, tilt_encoder_deg: float,
                  neck_to_camera: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """실기 인코더 각(deg) → (base 기준 카메라 위치 3, 쿼터니언 wxyz 4)."""
    T_nc = load_neck_to_camera() if neck_to_camera is None else neck_to_camera
    T = t_base_neck_from_encoder(pan_encoder_deg, tilt_encoder_deg) @ T_nc
    return T[:3, 3].copy(), quat_wxyz_from_matrix(T[:3, :3])


if __name__ == "__main__":
    for pan, tilt in ((0.0, -20.0), (0.0, -30.0), (15.0, -20.0), (-15.0, -20.0)):
        p, q = base_cam_pose(pan, tilt)
        print(f"pan {pan:+6.1f} tilt {tilt:+6.1f} → pos {np.round(p, 6).tolist()} "
              f"quat {np.round(q, 6).tolist()}")
