#!/usr/bin/env python3
"""회전 대칭 물체의 pose 정규화 — 대칭축 둘레 twist 를 버리고 swing 만 남긴다.

원통(shaker·cup)은 축 둘레 회전(yaw)이 FP++ 추적기의 자유 방향이라 프레임마다 흘러간다.
q = swing ⊗ twist(축 둘레, 로컬 프레임) 로 분해해 twist 를 제거하면 축의 방향(기울기)은
그대로이고 축 둘레 회전은 항상 0 이 된다. 위치는 건드리지 않는다.

쿼터니언은 전부 wxyz. numpy 만. test_pose_symmetry.py 대상.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a ⊗ b (wxyz)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_axis_direction(q: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """로컬 축 `axis` 가 q 로 회전된 방향 벡터."""
    v = np.r_[0.0, np.asarray(axis, float)]
    return quat_mul(quat_mul(q, v), quat_conj(q))[1:]


def remove_twist(q: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """q 에서 로컬 `axis` 둘레 twist 를 제거한 swing 쿼터니언(wxyz, 단위)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    proj = np.dot(q[1:], a) * a
    twist = np.r_[q[0], proj]
    norm = np.linalg.norm(twist)
    if norm < _EPS:
        return q                      # 축에 수직한 180° 회전 — twist 가 정의되지 않으니 그대로
    twist = twist / norm
    swing = quat_mul(q, quat_conj(twist))
    return swing / np.linalg.norm(swing)
