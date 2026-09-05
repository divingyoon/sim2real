#!/usr/bin/env python3
"""우팔 `grasp_s2r` actor obs(155D)를 실물 입력으로 조립하는 순수 로직 (numpy).

★기존 `grasp_obs_builder.py`(154D)는 **grasp_v1 용**이다. s2r 은 레이아웃이 다르다 —
  `palm_ax` 6 이 새로 있고, `object_onehot` 8 대신 `goal_rel` 3 이다. 그래서 고치지
  않고 따로 둔다(v1 배포 경로를 깨지 않기 위해서다).

레이아웃 (env 표본으로 검산, `logs/policy/right_e1/obs_layout.json`):

      0..6    arm_q            7   팔 엔코더 — **절대값**(좌팔과 달리 상대 아님)
      7..13   arm_qd           7   팔 엔코더
     14..33   hand_q          20   손 엔코더 ★**sim DOF 순**
     34..53   hand_qd         20   〃
     54..56   palm_pos         3   palm body 원점, env-local
     57..62   palm_ax          6   palm 회전행렬의 **앞 두 열**
     63..77   tips_rel_palm   15   (5×3) tip − palm
     78..80   palm_to_obj      3   컵 − palm
     81..95   obj_to_tips     15   (5×3) tip − 컵
     96..110  tip_force       15   (5×3) **팁 로컬** 3축 / contact_force_max
    111..130  joint_err       20   (시너지목표 − 실측) / joint_pos_err_max, ±1 클램프
    131..151  actions         21   직전 액션
    152..154  goal_rel         3   목표 − 컵
    ----------------------------------------------------------------- 합계 155

★★**손 20관절은 canonical 순이 아니라 Isaac DOF 순이다.**
  `index_1, middle_1, pinky_1, ring_1, thumb_1, index_2, …` — 마디가 바깥, 손가락이 안쪽.
  canonical(`thumb_1..4, index_1..4, …`)로 넣으면 손 40칸과 joint_err 20칸이 통째로
  스크램블된다. 09.01 표본 대조: DOF 순 오차 0.024(노이즈 수준) vs canonical 1.572.
  정책은 죽지 않고 **조용히 이상하게 돈다** — [[warm-bank-joint-order-scramble]] 과
  같은 계열의 사고다.

★`tip_force` 는 **팁 로컬 프레임**이다. 실기 F/T 가 센서 로컬 출력이라 변환 없이 받기
  위해 sim 이 일부러 맞춰 둔 것이다. world 를 넣으면 팔 자세마다 같은 접촉이 다른 값이 된다.

★`joint_err` 의 부호를 지우면 안 된다. 인벨롭이 잘 될수록 팁 F/T 가 0 을 읽어서, 이
  추종 오차가 **주 파지력 관측**이 된다.

★정규화 상수(`contact_force_max`, `joint_pos_err_max`)는 여기 적지 않는다 —
  `robot_profile.load_env_cfg_literals(cfg_path, class_name="GraspS2REnvCfg")` 로 읽는다.
"""

from __future__ import annotations

import numpy as np

#: (이름, 차원) — env 순서 그대로.
SEGMENTS: tuple[tuple[str, int], ...] = (
    ("arm_q", 7),
    ("arm_qd", 7),
    ("hand_q", 20),
    ("hand_qd", 20),
    ("palm_pos", 3),
    ("palm_ax", 6),
    ("tips_rel_palm", 15),
    ("palm_to_obj", 3),
    ("obj_to_tips", 15),
    ("tip_force", 15),
    ("joint_err", 20),
    ("actions", 21),
    ("goal_rel", 3),
)

ACTOR_OBS_DIM = sum(d for _, d in SEGMENTS)
NUM_ARM_DOF = 7
NUM_HAND_DOF = 20
NUM_FINGERTIPS = 5
NUM_ACTIONS = 21

#: sim 이 손 **관절**을 주는 순서 — 마디(1..4)가 바깥, 손가락이 안쪽.
_DOF_FINGER_ORDER = ("index", "middle", "pinky", "ring", "thumb")

#: 손끝 **body** 순서는 관절 순서와 **다르다** — 손가락 canonical 순이다.
#: 09.01 표본 역추적으로 확정. 알파벳순(body_names 순)으로 넣으면 22 cm 어긋난다.
_TIP_FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def hand_dof_order(side: str = "r") -> list[str]:
    """sim DOF 순 손 **관절** 이름 20개. **canonical 순이 아니다.**"""
    return [f"{side}_hj_{f}_{j}" for j in range(1, 5) for f in _DOF_FINGER_ORDER]


def tip_body_order(side: str = "r") -> list[str]:
    """obs 가 쓰는 손끝 **body** 순서 5개.

    ★관절 순서와 다르다. `body_names` 알파벳순으로 넣으면 `tips_rel_palm` 과
      `obj_to_tips` 가 **22 cm** 어긋난다(09.01 표본 대조로 확정).
    """
    return [f"{side}_hl_{f}_tip" for f in _TIP_FINGER_ORDER]


def reorder(values, from_names, to_names) -> np.ndarray:
    """`from_names` 순 값을 `to_names` 순으로 옮긴다. 못 옮기면 죽는다."""
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size != len(from_names):
        raise ValueError(f"값 {v.size}개 vs 이름 {len(from_names)}개 — 개수가 다르다")
    lookup = {n: i for i, n in enumerate(from_names)}
    missing = [n for n in to_names if n not in lookup]
    if missing:
        raise KeyError(f"옮길 값이 없다: {missing}")
    return v[[lookup[n] for n in to_names]]


def quat_to_matrix(quat_wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(quat_wxyz, dtype=float))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot6d_columns(quat_wxyz) -> np.ndarray:
    """회전행렬의 앞 두 **열**. 행을 쓰면 전치가 되어 다른 자세가 된다."""
    R = quat_to_matrix(quat_wxyz)
    return np.concatenate([R[:, 0], R[:, 1]])


def tip_force_local(force_world, tip_quat_wxyz, contact_force_max: float) -> np.ndarray:
    """손끝 힘을 **팁 로컬**로 회전하고 포화점으로 정규화. (T, 3)."""
    f = np.asarray(force_world, dtype=float).reshape(-1, 3)
    q = np.asarray(tip_quat_wxyz, dtype=float).reshape(-1, 4)
    if f.shape[0] != q.shape[0]:
        raise ValueError(f"힘 {f.shape[0]}개 vs 자세 {q.shape[0]}개 — 개수가 다르다")
    return np.stack([quat_to_matrix(q[i]).T @ f[i] for i in range(f.shape[0])]) / float(
        contact_force_max)


def normalized_joint_err(measured, target, joint_pos_err_max: float) -> np.ndarray:
    """(목표 − 실측) / 상한, **±1 클램프 · 부호 보존**."""
    err = np.asarray(target, dtype=float).reshape(-1) - np.asarray(measured, dtype=float).reshape(-1)
    return np.clip(err / float(joint_pos_err_max), -1.0, 1.0)


def _check(name: str, arr, n: int) -> np.ndarray:
    a = np.asarray(arr, dtype=float).reshape(-1)
    if a.size != n:
        raise ValueError(f"{name} 는 {n}개여야 하는데 {a.size}개다")
    return a


def _check_tips(name: str, arr) -> np.ndarray:
    a = np.asarray(arr, dtype=float).reshape(-1, 3)
    if a.shape[0] != NUM_FINGERTIPS:
        raise ValueError(f"{name} 는 손끝 {NUM_FINGERTIPS}개여야 하는데 {a.shape[0]}개다")
    return a


def assemble_actor_obs(
    *,
    arm_q, arm_qd,
    hand_q, hand_qd,
    joint_err_profile_order,
    palm_pos, palm_quat,
    tip_pos,
    cup_pos, goal_pos,
    tip_force_world, tip_quat,
    last_action,
    contact_force_max: float,
    joint_pos_err_max: float,
) -> np.ndarray:
    """155D actor obs.

    ★★**손 순서가 obs 안에서 두 종류다**(09.03 sim 대조로 확정):
      · `hand_q`/`hand_qd`(슬롯 14~53) — **Isaac DOF 순**(`find_joints` 반환 순)
      · `joint_err`(슬롯 111~130) — **프로필 순**(`hand_joint_names`, `_syn_ids`)
      env `_joint_pos_err()` 가 `_syn_ids` 로 재는데 `hand_q` 는 `_hand_ids_t` 로 잰다.
      그래서 `joint_err` 은 **호출자가 프로필 순으로 만들어** 넘긴다 — 여기서 DOF 순
      `hand_q` 로 계산하면 20칸이 통째로 스크램블된다. 정책은 죽지 않고 조용히
      이상하게 돈다. `normalized_joint_err()` 로 만들되 **프로필 순 쌍**을 줄 것.
    """
    aq = _check("arm_q", arm_q, NUM_ARM_DOF)
    aqd = _check("arm_qd", arm_qd, NUM_ARM_DOF)
    hq = _check("hand_q", hand_q, NUM_HAND_DOF)
    hqd = _check("hand_qd", hand_qd, NUM_HAND_DOF)
    jerr = _check("joint_err_profile_order", joint_err_profile_order, NUM_HAND_DOF)
    act = _check("last_action", last_action, NUM_ACTIONS)
    tips = _check_tips("tip_pos", tip_pos)
    palm = _check("palm_pos", palm_pos, 3)
    cup = _check("cup_pos", cup_pos, 3)
    goal = _check("goal_pos", goal_pos, 3)

    obs = np.concatenate([
        aq, aqd, hq, hqd,
        palm,
        rot6d_columns(palm_quat),
        (tips - palm).reshape(-1),
        cup - palm,
        (tips - cup).reshape(-1),
        tip_force_local(tip_force_world, tip_quat, contact_force_max).reshape(-1),
        np.clip(jerr, -1.0, 1.0),
        act,
        goal - cup,
    ])
    if obs.size != ACTOR_OBS_DIM:
        raise ValueError(f"조립 결과가 {obs.size}차원 — 계약은 {ACTOR_OBS_DIM}")
    return obs
