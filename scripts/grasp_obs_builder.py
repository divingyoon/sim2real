#!/usr/bin/env python3
"""grasp_v1 actor obs(154D)를 실물 입력으로 조립하는 순수 로직 (numpy, Isaac 불필요).

레이아웃 진실원천 = `grasp_{side}_env.py::_get_observations` 의 actor `torch.cat`.
아래 `OBS_SEGMENTS` 가 그 순서를 그대로 옮긴 **단일 진실원천**이며, 슬라이스는 여기서
파생한다(오프셋을 손으로 적지 않는다).

    arm_joint_pos            7    팔 엔코더
    arm_joint_vel            7    팔 엔코더
    finger_joint_pos        20    손 엔코더
    finger_joint_vel        20    손 엔코더
    palm_center_pos          3    palm FK
    fingertip_pos_rel_palm  15    (5×3) tip − palm
    palm_to_cup              3    cup − palm
    cup_to_fingertip        15    (5×3) tip − cup
    tip_force_local         15    (5×3) 손끝 3축 힘 / 10N, tip-local  ★08.16 신규
    joint_pos_err           20    (지령 − 실측) / 1.2rad, 부호 보존     ★08.16 신규
    last_actions            21    직전 action (정책 원출력)
    object_onehot            8    MultiAsset 8종
    ---------------------------------------------------------------- total 154

★`tip_force_local` 은 **tip-local 프레임 그대로** 넣는다. sim 이 일부러 tip-local 로 맞춰
  둔 이유가 "실물 F/T(센서 로컬 출력)를 변환 없이 그대로 받기 위해서"다. 배포에서 world
  변환을 넣으면 그 의도를 깨고, 변환이 어긋나면 조용히 잘못된 obs 가 된다.

★`joint_pos_err` 은 **인벨롭 그립의 주 힘 관측**이다. 인벨롭이 잘 될수록 접촉이 중간·
  원위마디로 가서 팁 F/T 는 0 을 읽는다(sim 실측: 40% 구간 팁 무접촉). 전 마디를 덮는
  신호는 이 오차뿐이므로 `abs()` 로 방향을 지우면 안 된다.

palm_center·fingertip_pos 는 Fabrics FK 결과(호출자 계산)로 넘어온다. 여기선 조립과
차원·유한성 검증만 한다.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 차원 (grasp_{side}_constants.py 와 대조 — test_grasp_contract_parity.py)
# ---------------------------------------------------------------------------
NUM_ARM_DOF = 7
NUM_HAND_DOF = 20
NUM_FINGERTIPS = 5
NUM_ACTIONS = 21
NUM_OBJECT_CLASSES = 8

# 정규화 상수 — hdgp 와 반드시 동일해야 한다
CONTACT_FORCE_MAX = 10.0    # grasp_{side}_constants.py CONTACT_FORCE_MAX
JOINT_POS_ERR_MAX = 1.2     # grasp_{side}_constants.py JOINT_POS_ERR_MAX

# ---------------------------------------------------------------------------
# 세그먼트 테이블 — 순서·차원의 단일 진실원천
# ---------------------------------------------------------------------------
OBS_SEGMENTS: tuple[tuple[str, int], ...] = (
    ("arm_joint_pos", NUM_ARM_DOF),
    ("arm_joint_vel", NUM_ARM_DOF),
    ("finger_joint_pos", NUM_HAND_DOF),
    ("finger_joint_vel", NUM_HAND_DOF),
    ("palm_center_pos", 3),
    ("fingertip_pos_rel_palm", NUM_FINGERTIPS * 3),
    ("palm_to_cup", 3),
    ("cup_to_fingertip", NUM_FINGERTIPS * 3),
    ("tip_force_local", NUM_FINGERTIPS * 3),
    ("joint_pos_err", NUM_HAND_DOF),
    ("last_actions", NUM_ACTIONS),
    ("object_onehot", NUM_OBJECT_CLASSES),
)


def _build_slices() -> dict[str, slice]:
    out: dict[str, slice] = {}
    off = 0
    for name, dim in OBS_SEGMENTS:
        out[name] = slice(off, off + dim)
        off += dim
    return out


OBS_SLICES: dict[str, slice] = _build_slices()
ACTOR_OBS_DIM: int = sum(dim for _, dim in OBS_SEGMENTS)   # 154

# ---------------------------------------------------------------------------
# MultiAsset 물체군 순서 (grasp_{side}_env_cfg.py `_ACTIVE_OBJECT_SPECS`).
# onehot 순서 = env_id % 8 결정론적 배정 순서.
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
    """물체 인덱스(0..7) 또는 이름 → onehot (8,) float64."""
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


def _check(name: str, arr, shape: tuple[int, ...]) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.shape != shape:
        raise ValueError(f"{name} expected shape {shape}, got {out.shape}")
    return out


def normalize_tip_force(tip_force_local, contact_force_max: float = CONTACT_FORCE_MAX):
    """(5,3) [N] tip-local → (15,) 정규화·clamp, tip-major.

    정규화를 빌더 안에서 하는 이유: 호출자가 잊는 경로를 없애기 위해서다.
    """
    f = _check("tip_force_local", tip_force_local, (NUM_FINGERTIPS, 3))
    return np.clip(f / float(contact_force_max), -1.0, 1.0).reshape(-1)


def compute_joint_pos_err(
    hand_cmd_prev, finger_joint_pos, joint_pos_err_max: float = JOINT_POS_ERR_MAX
):
    """(20,) (직전 전송 지령 − 실측)/max, clamp ±1. **부호 보존**.

    sim 대응: `(hand_joint_targets − hand_joint_pos) / JOINT_POS_ERR_MAX`.
    시간 정렬도 동일하다 — sim 은 스텝 t 의 목표 vs 물리 후 실측, 배포는 tick t−1 에
    보낸 지령 vs 현재 실측.
    """
    cmd = _check("hand_cmd_prev", hand_cmd_prev, (NUM_HAND_DOF,))
    act = _check("finger_joint_pos", finger_joint_pos, (NUM_HAND_DOF,))
    return np.clip((cmd - act) / float(joint_pos_err_max), -1.0, 1.0)


def assemble_actor_obs(
    *,
    arm_joint_pos,        # (7,)
    arm_joint_vel,        # (7,)
    finger_joint_pos,     # (20,)  실측
    finger_joint_vel,     # (20,)
    palm_center,          # (3,)   Fabrics FK
    fingertip_pos,        # (5,3)  Fabrics FK
    cup_pos,              # (3,)   /cup_pose (base)
    tip_force_local,      # (5,3)  [N] tip-local **원시값**(정규화 전)
    hand_cmd_prev,        # (20,)  직전에 실제로 보낸 손 지령
    last_actions,         # (21,)  정책 원출력(coupling 이전)
    object_onehot,        # (8,)
    contact_force_max: float = CONTACT_FORCE_MAX,
    joint_pos_err_max: float = JOINT_POS_ERR_MAX,
) -> np.ndarray:
    """실물 입력 → grasp_v1 actor obs (154,). 입력은 canonical 순서라고 가정.

    ★keyword-only 인 이유: 구 시그니처는 위치인자 10개였다. 계약이 114→154 로 바뀌는
      과정에서 인자 순서가 어긋나면 **조용히 통과**한다. 키워드 강제가 그 부류를 막는다.
    """
    arm_p = _check("arm_joint_pos", arm_joint_pos, (NUM_ARM_DOF,))
    arm_v = _check("arm_joint_vel", arm_joint_vel, (NUM_ARM_DOF,))
    fin_p = _check("finger_joint_pos", finger_joint_pos, (NUM_HAND_DOF,))
    fin_v = _check("finger_joint_vel", finger_joint_vel, (NUM_HAND_DOF,))
    palm = _check("palm_center", palm_center, (3,))
    tips = _check("fingertip_pos", fingertip_pos, (NUM_FINGERTIPS, 3))
    cup = _check("cup_pos", cup_pos, (3,))
    actions = _check("last_actions", last_actions, (NUM_ACTIONS,))
    onehot = _check("object_onehot", object_onehot, (NUM_OBJECT_CLASSES,))

    obs = np.concatenate([
        arm_p,
        arm_v,
        fin_p,
        fin_v,
        palm,
        (tips - palm).reshape(-1),                       # tip − palm
        cup - palm,                                      # cup − palm
        (tips - cup).reshape(-1),                        # tip − cup
        normalize_tip_force(tip_force_local, contact_force_max),
        compute_joint_pos_err(hand_cmd_prev, fin_p, joint_pos_err_max),
        actions,
        onehot,
    ])
    if obs.shape[0] != ACTOR_OBS_DIM:
        raise RuntimeError(f"actor obs dim {obs.shape[0]} != {ACTOR_OBS_DIM}")
    if not np.all(np.isfinite(obs)):
        bad = [n for n, sl in OBS_SLICES.items() if not np.all(np.isfinite(obs[sl]))]
        raise RuntimeError(f"actor obs 에 비유한값 — 세그먼트 {bad}")
    return obs
