#!/usr/bin/env python3
"""sim(hdgp) ↔ 배포(sim2real) **계약 parity** — 좌/우.

이 저장소가 반복해 겪은 실패는 전부 "sim 이 계약을 바꿨는데 배포가 몰랐다" 형태다
(obs 114 vs 154, action 11 vs 21, 래칫, palm delta, 자산 6.5cm). 그 격차를 사람이
발견하기 전에 **테스트가 먼저 죽도록** 고정한다.

전략:
  · `grasp_{side}_constants.py` / `_utils.py` 는 torch 만 필요 → **직접 import** 해 대조
  · `grasp_{side}_env_cfg.py` / `_env.py` 는 isaaclab(→pxr) 의존 → **AST 로 소스 파싱**
  · hdgp 가 없으면 skip (배포 전용 머신)
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from robot_profile import (  # noqa: E402
    HDGP_OPENARM_SRC,
    available_profiles,
    hdgp_task_dir,
    load_hdgp_module,
    load_profile_env_cfg,
    load_robot_profile,
)

import grasp_action_decoder as D  # noqa: E402
import grasp_obs_builder as O  # noqa: E402

SIDES = ["right", "left"]
PROFILES = {"right": "tesollo_bi_s__right", "left": "tesollo_bi_s__left"}
# ★모든 구성(grasp_v1 좌/우, grasp_sensor …)을 자동으로 덮는다.
ALL_PROFILES = available_profiles()

pytestmark = pytest.mark.skipif(not HDGP_OPENARM_SRC.exists(), reason="hdgp 소스 없음")


def _env_path_for(profile) -> Path:
    return hdgp_task_dir(profile) / f"grasp_{profile.acting_side}_env.py"


def _env_path(side: str) -> Path:
    """bi_s 좌우 쌍 전용(동형성 비교)."""
    return _env_path_for(load_robot_profile(PROFILES[side]))


def _consts_of(name: str):
    return load_hdgp_module(load_robot_profile(name), "constants")


def _consts(side: str):
    return _consts_of(PROFILES[side])


# --------------------------------------------------------------------------
# (a) 상수 parity — 배포가 복제한 값이 sim 과 같은가
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_PROFILES)
def test_dimension_constants(name):
    C = _consts_of(name)
    assert O.ACTOR_OBS_DIM == C.NUM_OBSERVATIONS
    assert D.NUM_ACTIONS == C.NUM_ACTIONS
    assert O.NUM_ACTIONS == C.NUM_ACTIONS
    assert O.NUM_ARM_DOF == C.NUM_ARM_DOF
    assert O.NUM_HAND_DOF == C.NUM_HAND_DOF
    assert D.NUM_FINGER_ACTION == C.NUM_FINGER_ACTION
    assert D.NUM_FINGER_CHANNELS == C.NUM_FINGER_CHANNELS
    assert D.NUM_PALM_ACTION == C.NUM_PALM_ACTION
    assert O.NUM_OBJECT_CLASSES == C.NUM_OBJECT_CLASSES


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_normalization_constants(name):
    """정규화 상수가 어긋나면 obs 스케일이 통째로 달라진다(조용한 실패)."""
    C = _consts_of(name)
    assert O.CONTACT_FORCE_MAX == C.CONTACT_FORCE_MAX
    assert O.JOINT_POS_ERR_MAX == C.JOINT_POS_ERR_MAX


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_segment_dims_sum_to_contract(name):
    C = _consts_of(name)
    assert sum(d for _, d in O.OBS_SEGMENTS) == C.NUM_OBSERVATIONS
    assert D.NUM_PALM_ACTION + D.NUM_FINGER_ACTION == C.NUM_ACTIONS


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_profile_contract_matches_constants(name):
    """구성 프로필이 선언한 계약도 같아야 한다(로드 시 검증되지만 명시적으로 고정)."""
    C = _consts_of(name)
    p = load_robot_profile(name)
    assert p.contract.obs_dim == C.NUM_OBSERVATIONS
    assert p.contract.action_dim == C.NUM_ACTIONS


# --------------------------------------------------------------------------
# (b) obs 세그먼트 순서 — sim `_get_observations` 의 torch.cat 과 대조
# --------------------------------------------------------------------------

def _sim_actor_segments(name: str) -> list[str]:
    """`actor_obs = torch.cat([...])` 의 식별자 시퀀스를 소스에서 뽑는다."""
    src = _env_path_for(load_robot_profile(name)).read_text()
    m = re.search(r"actor_obs\s*=\s*torch\.cat\(\s*\[(.*?)\]\s*,", src, re.S)
    if not m:
        pytest.skip(f"{name}: actor_obs torch.cat 블록을 찾지 못함")
    names = []
    for line in m.group(1).splitlines():
        line = line.split("#")[0].strip().rstrip(",").strip()
        if line and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", line):
            names.append(line)
    return names


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_obs_segment_order_matches_sim(name):
    """★sim 이 세그먼트를 재배열/추가하면 즉시 RED."""
    sim = _sim_actor_segments(name)
    local = [name for name, _ in O.OBS_SEGMENTS]
    assert sim == local, (
        f"obs 세그먼트 순서 불일치\n  sim   {sim}\n  배포  {local}\n"
        "  grasp_obs_builder.OBS_SEGMENTS 를 sim 순서로 맞춰라."
    )


# --------------------------------------------------------------------------
# (c) env_cfg 리터럴 parity — 배포는 AST 로 읽으므로 값 자체는 자동 정합.
#     여기서는 **계약이 유지되고 있는지**(액션 미러 금지 등)를 고정한다.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_PROFILES)
def test_cfg_gate_flags(name):
    cfg = load_profile_env_cfg(load_robot_profile(name))
    assert cfg["couple_four_fingers"] is True, "4지 공통닫힘이 꺼지면 3지 국소최적이 돌아온다"
    assert cfg["retighten_after_latch"] is False, (
        "sim 이 retighten 을 켰다면 배포 디코더도 같이 켜야 한다"
    )


def test_palm_delta_identical_between_sides():
    """★액션 의미는 미러되지 않는다 — 배포에서 부호를 뒤집으면 안 된다."""
    r = load_profile_env_cfg(load_robot_profile(PROFILES["right"]))
    l = load_profile_env_cfg(load_robot_profile(PROFILES["left"]))
    assert r["palm_delta_xyz"] == l["palm_delta_xyz"]
    assert r["palm_delta_rot_deg"] == l["palm_delta_rot_deg"]


def test_home_and_lift_are_mirrored():
    """반대로 홈·리프트 방향은 미러여야 한다."""
    r = load_profile_env_cfg(load_robot_profile(PROFILES["right"]))
    l = load_profile_env_cfg(load_robot_profile(PROFILES["left"]))
    assert r["reset_home_palm_pose"][1] == pytest.approx(-l["reset_home_palm_pose"][1])
    assert r["reset_home_palm_pose"][3] == pytest.approx(-l["reset_home_palm_pose"][3])
    assert r["lift_wait_joint7_delta"] == pytest.approx(-l["lift_wait_joint7_delta"])


# --------------------------------------------------------------------------
# (d) 좌우 소스 동형성 — "단일 구현으로 좌우 커버" 전제를 소스에서 보증
# --------------------------------------------------------------------------

def _normalize_side_tokens(text: str) -> str:
    text = re.sub(r"\bright\b", "SIDE", text)
    text = re.sub(r"\bleft\b", "SIDE", text)
    text = re.sub(r"\bRIGHT\b", "SIDE", text)
    text = re.sub(r"\bLEFT\b", "SIDE", text)
    text = re.sub(r"\br_(aj|hj|hl)_", "X_\\1_", text)
    text = re.sub(r"\bl_(aj|hj|hl)_", "X_\\1_", text)
    return text


def _finger_block(side: str) -> str:
    """손가락 폐쇄 적분 블록(cmd_ch → hand_target)만 잘라낸다."""
    src = _env_path(side).read_text()
    start = src.find("cmd_ch = 0.5 *")
    end = src.find("hand_joint_targets", start)
    if start < 0 or end < 0:
        pytest.skip(f"{side}: 손가락 블록을 찾지 못함")
    return src[start:end]


def test_left_right_finger_logic_identical():
    """좌우 손가락 로직이 side 토큰 말고는 완전히 같아야 한다.

    다르면 "단일 구현으로 좌우 커버" 전제가 깨진 것이므로, 배포도 갈라야 한다는 신호다.
    """
    r = _normalize_side_tokens(_finger_block("right"))
    l = _normalize_side_tokens(_finger_block("left"))
    r_code = [ln.split("#")[0].rstrip() for ln in r.splitlines() if ln.split("#")[0].strip()]
    l_code = [ln.split("#")[0].rstrip() for ln in l.splitlines() if ln.split("#")[0].strip()]
    assert r_code == l_code, "좌우 손가락 로직이 갈라졌다 — 배포 단일 구현 전제 재검토 필요"


# --------------------------------------------------------------------------
# (e) 수치 parity — hdgp utils 직접 호출 (구 test_grasp_decoder_parity 흡수)
# --------------------------------------------------------------------------

def _utils(name: str):
    p = load_robot_profile(name)
    path = hdgp_task_dir(p) / f"grasp_{p.acting_side}_utils.py"
    if not path.exists():
        pytest.skip(f"{name}: utils 없음")
    pytest.importorskip("torch")
    return load_hdgp_module(p, "utils")


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_scale_matches_sim(name):
    import torch

    U = _utils(name)
    rng = np.random.RandomState(0)
    a = rng.uniform(-1, 1, 6)
    lo = np.array([-0.15, -0.35, -0.15, -0.35, -0.35, -0.35])
    hi = -lo
    mine = D.scale_palm_delta(a, lo, hi)
    theirs = U.scale(
        torch.tensor(a).unsqueeze(0), torch.tensor(lo), torch.tensor(hi)
    )[0].numpy()
    assert np.allclose(mine, theirs, atol=1e-12)


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_joint7_lift_wait_matches_sim(name):
    import torch

    U = _utils(name)
    cfg = load_profile_env_cfg(load_robot_profile(name))
    arm = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    mine = D.joint7_lift_wait_target(
        arm, joint7_delta=cfg["lift_wait_joint7_delta"],
        joint7_min=cfg["warm_j7_min"], joint7_max=cfg["warm_j7_max"],
    )
    theirs = U.compute_joint7_lift_wait_target(
        torch.tensor(arm).unsqueeze(0),
        joint7_delta=cfg["lift_wait_joint7_delta"],
        joint7_min=cfg["warm_j7_min"], joint7_max=cfg["warm_j7_max"],
    )[0].numpy()
    assert np.allclose(mine, theirs, atol=1e-12)


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_lift_latch_matches_sim_sequence(name):
    import torch

    U = _utils(name)
    cfg = load_profile_env_cfg(load_robot_profile(name))
    latch = D.LiftLatch(min_contacts=cfg["lift_start_min_grip_fingers"],
                        hold_steps=cfg["grasp_ready_hold_steps"])
    hold = torch.zeros(1, dtype=torch.long)
    latched = torch.zeros(1, dtype=torch.bool)
    for n in [3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3]:
        mine = latch.update(n)
        hold, _ready, latched = U.compute_lift_readiness(
            num_contacts=torch.tensor([n]),
            is_grasp_phase=~latched,
            previous_hold_count=hold,
            previous_latched=latched,
            min_contacts=cfg["lift_start_min_grip_fingers"],
            hold_steps=cfg["grasp_ready_hold_steps"],
        )
        assert bool(mine) == bool(latched[0]), f"n={n}"


# --------------------------------------------------------------------------
# (f) 손가락 적분기 parity — sim 블록이 인라인이라 전사본과 비교
# --------------------------------------------------------------------------

def _sim_finger_reference(action15, tip, close_buf, rate, couple, hand_open, hand_full):
    """grasp_{side}_env.py:860-1049 전사(torch 없이 numpy 로 1:1).

    ★출처 라인 앵커를 유지할 것. sim 이 이 블록을 바꾸면 (d) 동형성 테스트와 함께
      여기도 갱신해야 한다.
    """
    fa = np.asarray(action15, dtype=np.float64).reshape(5, 3)
    if couple:                                        # :873-876  (clamp 이전)
        fa = np.concatenate([fa[0:1], np.tile(fa[1:5].mean(axis=0), (4, 1))], axis=0)
    cmd_ch = 0.5 * (np.clip(fa, -1.0, 1.0) + 1.0)     # :981
    g_zero = np.zeros(5)
    g34 = np.clip(np.asarray(tip, dtype=np.float64), 0.0, 1.0)   # tip-only(라이브 전제)
    gate20 = np.stack([g_zero, g_zero, g34, g34], axis=1).reshape(-1)   # :993
    cmd20 = np.stack(
        [cmd_ch[:, 0], cmd_ch[:, 1], cmd_ch[:, 2], cmd_ch[:, 2]], axis=1
    ).reshape(-1)                                     # :1008-1010
    delta = np.clip(cmd20 - close_buf, -rate, rate)   # :1020
    buf = np.clip(close_buf + delta * (1.0 - gate20), 0.0, 1.0)   # :1021-1022
    return buf, hand_open + buf * (hand_full - hand_open)


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_finger_integrator_matches_sim_reference(name):
    """랜덤 200스텝에서 배포 디코더 == sim 전사본."""
    cfg = load_profile_env_cfg(load_robot_profile(name))
    preset = load_hdgp_module(load_robot_profile(name), "preset")
    hand_open = np.asarray(preset.HAND_APPROACH_POSE, dtype=np.float64)
    hand_full = np.asarray(preset.HAND_FULL_GRIP_POSE, dtype=np.float64)
    rate = cfg["finger_close_speed"]

    ctrl = D.GraspFingerController(
        hand_open, hand_full, close_speed=rate,
        couple_four=cfg["couple_four_fingers"],
        retighten_after_latch=cfg["retighten_after_latch"],
    )
    buf = np.zeros(20)
    rng = np.random.RandomState(7)
    for step in range(200):
        act = rng.uniform(-1.2, 1.2, 15)
        tip = (rng.uniform(0, 1, 5) > 0.7).astype(np.float64)
        mine = ctrl.step(act, tip)
        buf, theirs = _sim_finger_reference(
            act, tip, buf, rate, cfg["couple_four_fingers"], hand_open, hand_full
        )
        assert np.allclose(ctrl.close_buf, buf, atol=1e-12), f"close_buf step {step}"
        assert np.allclose(mine, theirs, atol=1e-12), f"hand_target step {step}"
