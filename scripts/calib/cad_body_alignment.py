"""cup.obj(mesh) ↔ sim 컵 body 프레임 정합. 순수 numpy — ROS 불필요.

mesh 규약: Y축이 높이(측정: x/z=9cm, y=17.76cm), 원점 임의(바닥이 y=min).
sim body 규약: +z=위, 원점=**mesh 원점 그대로**(Y-up→Z-up 회전만 적용, 재중심화 없음).

근거(evidence): hdgp `pour_v5/pour_right_env_cfg.py:198`의 cup geometry 주석은
".usd 기준: bottom=-0.077m, rim=+0.100m"이며, 이는 `cup.obj`의 raw Y-AABB
(-0.0773 .. 0.1003)와 소수점 3자리까지 일치한다. 즉 sim body 프레임은
"바닥 중심"이 아니라 mesh 원점(y=0)을 그대로 body z=0으로 매핑한다.
따라서 T_cad_body = (mesh를 Y-up→Z-up 회전, Rx(+90°)) + (평행이동 없음, translation=0).
"""
from __future__ import annotations
import numpy as np


def mesh_aabb(obj_path: str) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([np.inf, np.inf, np.inf])
    hi = -lo.copy()
    with open(obj_path) as f:
        for ln in f:
            if ln.startswith("v "):
                xyz = np.array([float(v) for v in ln.split()[1:4]])
                lo = np.minimum(lo, xyz)
                hi = np.maximum(hi, xyz)
    return lo, hi


def cad_to_body_yup_to_zup(
    aabb_min: np.ndarray, aabb_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(pos_xyz, quat_wxyz): mesh(Y-up) → sim body(Z-up, 원점=mesh 원점).

    aabb_min/aabb_max는 API 안정성을 위해 시그니처에 유지하나 translation
    계산에는 더 이상 쓰이지 않는다(evidence: hdgp pour_v5/pour_right_env_cfg.py:198
    bottom=-0.077/rim=+0.100 == cup.obj raw Y-AABB → sim body 원점은
    바닥중심이 아니라 mesh 원점 그대로).
    """
    # Y-up→Z-up: x축 기준 +90° 회전 (Y→Z, Z→-Y)
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    quat = np.array([c, s, 0.0, 0.0])  # wxyz, Rx(+90°)
    # 평행이동 없음: sim body 원점 = mesh 원점 (재중심화하지 않음)
    pos = np.zeros(3)
    return pos, quat
