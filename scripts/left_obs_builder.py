#!/usr/bin/env python3
"""좌팔 `grasp_sensor_v2` actor obs(49D)를 실물 입력으로 조립하는 순수 로직 (numpy).

**왜 이게 필요한가.** 실기에는 학습 env 가 없다. 관측을 만드는 것은 **배포 코드**이고,
그 코드가 학습 env 와 한 칸이라도 어긋나면 정책은 죽는 게 아니라 **조용히 이상하게
돈다**. 그래서 계약을 손으로 옮겨 적지 않고 env 에서 뽑았다
(`logs/policy/left_v2B25/obs_layout.json`, `probe_obs_layout.py` 가 만든다).

obs 레이아웃 (env 실측, 오프셋 포함):

    0..8    joint_pos               9   팔7+그리퍼2, **기본자세 대비 상대**
    9..17   joint_vel               9   〃
    18..20  object_position         3   컵 위치, robot root 프레임 (실기: /cup_pose)
    21..27  target_object_position  7   목표 pose(pos3+quat4), root 프레임
    28..34  actions                 7   직전 액션
    35      gripper_gate            1   그리퍼 개방 게이트 (배포 노드가 들고 있어야 한다)
    36..38  tcp_pos                 3   TCP 위치, root 프레임을 **palm 박스로 정규화**
    39..44  palm_rot                6   그리퍼 base 회전행렬의 **앞 두 열**, root 프레임
    45..47  goal_minus_cup          3   목표 − 컵 (world)
    48      cup_upright             1   컵 z축의 world z 성분
    ------------------------------------------------------------------
    합계                           49

★**상수는 복제하지 않는다.** palm 박스는 `robot_profile.load_hdgp_module(p, "preset")`
  으로 hdgp 에서 읽는다 — 여기 적어 두면 sim 이 값을 바꿨을 때 조용히 어긋난다.

★**joint 항은 기본자세를 뺀 값이다**(`mdp.joint_pos_rel`). 절대 엔코더값을 그대로
  넣으면 49칸이 통째로 어긋난다.
"""

from __future__ import annotations

import numpy as np

#: (이름, 차원) — env 에서 뽑은 순서 그대로. 바꾸면 정책이 다른 것을 본다.
SEGMENTS: tuple[tuple[str, int], ...] = (
    ("joint_pos", 9),
    ("joint_vel", 9),
    ("object_position", 3),
    ("target_object_position", 7),
    ("actions", 7),
    ("gripper_gate", 1),
    ("tcp_pos", 3),
    ("palm_rot", 6),
    ("goal_minus_cup", 3),
    ("cup_upright", 1),
)

ACTOR_OBS_DIM = sum(d for _, d in SEGMENTS)
NUM_ARM_AND_GRIPPER_DOF = 9
NUM_ACTIONS = 7


def quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """wxyz 쿼터니언 → 3×3 회전행렬."""
    w, x, y, z = (float(v) for v in np.asarray(quat_wxyz, dtype=float))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot6d_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """회전행렬의 **앞 두 열**(6D 표현).

    ★열이지 행이 아니다 — 행을 쓰면 전치가 되어 정책이 다른 자세를 본다.
    euler 를 안 쓰는 이유는 학습 쪽 주석대로 ±π 경계에서 널뛰기 때문이다.
    """
    R = quat_to_matrix(quat_wxyz)
    return np.concatenate([R[:, 0], R[:, 1]])


def subtract_frame(
    root_pos: np.ndarray, root_quat: np.ndarray,
    pos: np.ndarray, quat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """world 자세를 root 프레임으로 (`isaaclab.utils.math.subtract_frame_transforms`)."""
    R = quat_to_matrix(root_quat)
    pos_b = R.T @ (np.asarray(pos, dtype=float) - np.asarray(root_pos, dtype=float))
    if quat is None:
        return pos_b, None
    R_b = R.T @ quat_to_matrix(quat)
    return pos_b, R_b


def normalize_tcp(pos_b: np.ndarray, palm_box) -> np.ndarray:
    """palm 박스 기준 정규화 — 중심이 0, 모서리가 ±1. **자르지 않는다**.

    잘라 버리면 "박스를 나갔다"는 사실이 obs 에서 사라진다.
    """
    lo = np.array([b[0] for b in palm_box], dtype=float)
    hi = np.array([b[1] for b in palm_box], dtype=float)
    return (np.asarray(pos_b, dtype=float) - (lo + hi) / 2.0) / ((hi - lo) / 2.0)


def cup_upright(cup_quat_wxyz: np.ndarray) -> float:
    """컵 z축의 world z 성분. 세워져 있으면 1, 눕혀지면 0."""
    return float(quat_to_matrix(cup_quat_wxyz)[2, 2])


def _check(name: str, arr: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(arr, dtype=float).reshape(-1)
    if a.size != n:
        raise ValueError(f"{name} 는 {n}개여야 하는데 {a.size}개다")
    return a


def assemble_actor_obs(
    *,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    joint_pos_default: np.ndarray,
    joint_vel_default: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    cup_pos: np.ndarray,
    cup_quat: np.ndarray,
    goal_pos: np.ndarray,
    goal_quat: np.ndarray,
    tcp_pos: np.ndarray,
    gripper_base_pos: np.ndarray,
    gripper_base_quat: np.ndarray,
    last_action: np.ndarray,
    gripper_gate: float,
    palm_box,
) -> np.ndarray:
    """49D actor obs. 입력은 전부 **world 프레임**(관절 제외)."""
    q = _check("joint_pos", joint_pos, NUM_ARM_AND_GRIPPER_DOF)
    qd = _check("joint_vel", joint_vel, NUM_ARM_AND_GRIPPER_DOF)
    q0 = _check("joint_pos_default", joint_pos_default, NUM_ARM_AND_GRIPPER_DOF)
    qd0 = _check("joint_vel_default", joint_vel_default, NUM_ARM_AND_GRIPPER_DOF)
    act = _check("last_action", last_action, NUM_ACTIONS)

    cup_b, _ = subtract_frame(root_pos, root_quat, cup_pos)
    tcp_b, _ = subtract_frame(root_pos, root_quat, tcp_pos)
    _, base_R = subtract_frame(root_pos, root_quat, gripper_base_pos, gripper_base_quat)

    obs = np.concatenate([
        q - q0,
        qd - qd0,
        cup_b,
        np.asarray(goal_pos, dtype=float).reshape(3),
        np.asarray(goal_quat, dtype=float).reshape(4),
        act,
        np.array([float(gripper_gate)]),
        normalize_tcp(tcp_b, palm_box),
        np.concatenate([base_R[:, 0], base_R[:, 1]]),
        np.asarray(goal_pos, dtype=float).reshape(3) - np.asarray(cup_pos, dtype=float).reshape(3),
        np.array([cup_upright(cup_quat)]),
    ])
    if obs.size != ACTOR_OBS_DIM:
        raise ValueError(f"조립 결과가 {obs.size}차원 — 계약은 {ACTOR_OBS_DIM}")
    return obs
