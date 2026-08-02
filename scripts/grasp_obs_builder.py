#!/usr/bin/env python3
"""grasp-v1 actor obs(114D)를 실물 입력으로 조립하는 순수 로직 (numpy, Isaac 불필요).

obs 레이아웃 (tesollo/right/grasp_v1, grasp_right_env.py `_get_observations`):

    arm_joint_pos            7    로봇 팔 엔코더
    arm_joint_vel            7    로봇 팔 엔코더
    finger_joint_pos        20    손 엔코더 (r_hj_*)
    finger_joint_vel        20    손 엔코더
    palm_center_pos          3    palm FK (world/base)
    fingertip_pos_rel_palm  15    (5×3)  tip - palm
    palm_to_cup              3    cup - palm
    cup_to_fingertip        15    (5×3)  tip - cup
    fingertip_contact_binary 5    FT 센서 이진 접촉
    last_actions            11    직전 action
    ---------------------------------------------  base 106
    object_onehot            8    MultiAsset 8종 (2026-07-26, base 뒤 append)
    ---------------------------------------------  total 114

palm_center_pos·fingertip_pos 는 Fabrics FK 결과(호출자가 계산)로 넘어온다 — pour_obs_builder
가 컵 pose를 입력으로 받는 것과 같은 패턴. 여기선 배열 조립·차원/부호 검증만 담당한다.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Dimensions (grasp_right_constants.py 와 일치)
# ---------------------------------------------------------------------------
NUM_ARM_DOF = 7
NUM_HAND_DOF = 20
NUM_FINGERTIPS = 5
NUM_ACTIONS = 11
BASE_OBS_DIM = 106
NUM_OBJECT_CLASSES = 8
ACTOR_OBS_DIM = BASE_OBS_DIM + NUM_OBJECT_CLASSES  # 114

# ---------------------------------------------------------------------------
# MultiAsset 물체군 순서 (grasp_right_env_cfg.py `_ACTIVE_OBJECT_SPECS`).
# onehot·bbox 조회 키 순서 = env_id % 8 결정론적 배정 순서.
# ---------------------------------------------------------------------------
OBJECT_NAMES: tuple[str, ...] = (
    "cup_big_s085",       # 0
    "cup_big_s100",       # 1  ★ 실물 컵(cup_big 표준 스케일)
    "cup_big_s115",       # 2
    "cup_big_s130",       # 3
    "shaker_body",        # 4
    "large_5_cyl",        # 5
    "large_8_cyl_h12",    # 6
    "large_12_cyl_h12",   # 7
)
REAL_CUP_INDEX = 1  # 실물 컵 기본 onehot 인덱스 (cup_big_s100)


def make_object_onehot(obj: int | str) -> np.ndarray:
    """물체 인덱스(0..7) 또는 이름 → onehot (8,) float64.

    라이브 배포 시 잡는 물체 하나로 고정된다. env 의 F.one_hot 과 동일한 표현.
    """
    if isinstance(obj, str):
        if obj not in OBJECT_NAMES:
            raise ValueError(f"unknown object name {obj!r}; expected one of {OBJECT_NAMES}")
        idx = OBJECT_NAMES.index(obj)
    else:
        idx = int(obj)
        if not (0 <= idx < NUM_OBJECT_CLASSES):
            raise ValueError(f"object index {idx} out of range [0, {NUM_OBJECT_CLASSES})")
    onehot = np.zeros(NUM_OBJECT_CLASSES, dtype=np.float64)
    onehot[idx] = 1.0
    return onehot


def _check(name: str, arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.shape != shape:
        raise ValueError(f"{name} expected shape {shape}, got {out.shape}")
    return out


def assemble_actor_obs(
    arm_joint_pos: np.ndarray,     # (7,)
    arm_joint_vel: np.ndarray,     # (7,)
    finger_joint_pos: np.ndarray,  # (20,)
    finger_joint_vel: np.ndarray,  # (20,)
    palm_center: np.ndarray,       # (3,)   Fabrics FK
    fingertip_pos: np.ndarray,     # (5, 3) Fabrics FK
    cup_pos: np.ndarray,           # (3,)   /cup_pose (base)
    binary_contact: np.ndarray,    # (5,)   FT 이진 접촉
    last_actions: np.ndarray,      # (11,)
    object_onehot: np.ndarray,     # (8,)
) -> np.ndarray:
    """실물 입력 → grasp_v1 actor obs (114,). 입력은 canonical 순서라고 가정."""
    arm_p = _check("arm_joint_pos", arm_joint_pos, (NUM_ARM_DOF,))
    arm_v = _check("arm_joint_vel", arm_joint_vel, (NUM_ARM_DOF,))
    fin_p = _check("finger_joint_pos", finger_joint_pos, (NUM_HAND_DOF,))
    fin_v = _check("finger_joint_vel", finger_joint_vel, (NUM_HAND_DOF,))
    palm = _check("palm_center", palm_center, (3,))
    tips = _check("fingertip_pos", fingertip_pos, (NUM_FINGERTIPS, 3))
    cup = _check("cup_pos", cup_pos, (3,))
    contact = _check("binary_contact", binary_contact, (NUM_FINGERTIPS,))
    actions = _check("last_actions", last_actions, (NUM_ACTIONS,))
    onehot = _check("object_onehot", object_onehot, (NUM_OBJECT_CLASSES,))

    fingertip_rel_palm = (tips - palm).reshape(-1)   # (15,) tip - palm
    palm_to_cup = cup - palm                          # (3,)  cup - palm
    cup_to_fingertip = (tips - cup).reshape(-1)       # (15,) tip - cup

    obs = np.concatenate([
        arm_p,                 # 7
        arm_v,                 # 7
        fin_p,                 # 20
        fin_v,                 # 20
        palm,                  # 3
        fingertip_rel_palm,    # 15
        palm_to_cup,           # 3
        cup_to_fingertip,      # 15
        contact,               # 5
        actions,               # 11
        onehot,                # 8
    ])
    if obs.shape[0] != ACTOR_OBS_DIM:
        raise RuntimeError(f"actor obs dim {obs.shape[0]} != {ACTOR_OBS_DIM}")
    return obs
