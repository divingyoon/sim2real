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
    DELTA_ANCHOR,
    available_profiles,
    profiles_with_convention,
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
# ★이 모듈은 **델타+기준점 규약**(palm delta 6D + 홈/컵 기준점, obs 154D / action 21D)을
#   검증한다. `gripper/left/grasp_sensor` 는 절대 palm 규약이라 a=0 의 뜻부터 다르고
#   손가락 적분기도 lift 래치도 없다 — 그쪽에 이 단언을 적용하면 사실이 아닌 것을 요구하게
#   된다. 이름을 나열하지 않고 **프로필이 선언한 규약**으로 고르므로, 새 델타 구성은
#   자동으로 대상이 되고 규약을 빠뜨린 구성은 로드 단계에서 거부된다.
DELTA_ANCHOR_PROFILES = profiles_with_convention(DELTA_ANCHOR)
ALL_PROFILES = DELTA_ANCHOR_PROFILES

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
    """구성 프로필이 선언한 계약도 같아야 한다(로드 시 검증되지만 명시적으로 고정).

    ★source=checkpoint 프로필은 제외 — 계약의 진실원천이 hdgp_package 상수가 아니라
    체크포인트다(예: tesollo_sensor__right 는 08.31 부터 grasp_s2r 의 m1, obs 155 —
    package 상수 154 와 **의도적으로** 다르다). 그쪽은
    test_robot_profile 의 checkpoint 대조가 잡는다.
    """
    p = load_robot_profile(name)
    if p.contract.source == "checkpoint":
        pytest.skip(f"{name}: 계약 원천이 체크포인트 — constants 대조 비적용")
    C = _consts_of(name)
    assert p.contract.obs_dim == C.NUM_OBSERVATIONS
    assert p.contract.action_dim == C.NUM_ACTIONS


# --------------------------------------------------------------------------
# (b) obs 세그먼트 순서 — sim `_get_observations` 의 torch.cat 과 대조
# --------------------------------------------------------------------------

def _sim_actor_segments(name: str) -> list[str]:
    """sim `_get_observations` 의 obs concat 세그먼트 이름 시퀀스.

    ★AST 추출기(`obs_contract`)를 쓴다. 구 정규식은 단순 식별자가 아닌 항을 **조용히
    버려서** 세그먼트 수가 어긋났고, `actor_obs_parts` 우회나 `nan_to_num` 재대입을
    쓰는 태스크(hdgp 4개)에서는 아예 찾지 못했다.
    """
    from obs_contract import extract_obs_contract

    return extract_obs_contract(_env_path_for(load_robot_profile(name))).names


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


# ===========================================================================
# Fabrics 계층 parity — 모션 계층이 갈리면 그 위(obs·action)는 전부 무의미하다.
#
# 최종 아키텍처가 RL → palm 목표 → **Fabrics** → 관절이므로, sim 과 실기가 같은
# Fabrics 를 **같은 인자로** 만들어야 한다(가이드 §31). 자산 신원이 어긋나 palm 이
# 6.5cm 밀렸던 사고(08.18)가 정확히 이 층에서 났다.
# ===========================================================================

def _call_kwargs(src: str, func_name: str) -> list[dict]:
    """소스에서 `func_name(...)` 호출들의 **리터럴 키워드 인자**를 뽑는다."""
    tree = ast.parse(src)
    out: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
        if name != func_name:
            continue
        kw = {}
        for k in node.keywords:
            if k.arg is None:
                continue
            try:
                kw[k.arg] = ast.literal_eval(k.value)
            except (ValueError, SyntaxError):
                kw[k.arg] = "<non-literal>"
        out.append(kw)
    return out


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_fabric_class_matches_sim(name):
    """프로필이 선언한 fabric 클래스로 sim 이 실제로 만드는가."""
    prof = load_robot_profile(name)
    src = _env_path_for(prof).read_text()
    calls = _call_kwargs(src, prof.fabrics.class_name)
    assert calls, f"{name}: sim 이 {prof.fabrics.class_name} 를 생성하지 않는다"


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_fabric_robot_asset_matches_sim(name):
    """★자산 신원 — `robot_dir_name`/`robot_name` 이 프로필과 같아야 한다.

    배포가 이 인자를 생략해 기본값을 쓰던 시절 palm 이 6.5cm 어긋났다. sim 이 자산을
    바꾸면 여기서 먼저 죽는다.
    """
    prof = load_robot_profile(name)
    src = _env_path_for(prof).read_text()
    calls = _call_kwargs(src, prof.fabrics.class_name)
    for kw in calls:
        assert kw.get("robot_dir_name") == prof.fabrics.robot_dir, (
            f"{name}: sim robot_dir_name={kw.get('robot_dir_name')} != "
            f"프로필 {prof.fabrics.robot_dir}"
        )
        assert kw.get("robot_name") == prof.fabrics.robot_name


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_fabric_world_matches_sim(name):
    """충돌 월드가 다르면 같은 palm 목표에서 다른 관절해가 나온다."""
    prof = load_robot_profile(name)
    src = _env_path_for(prof).read_text()
    worlds = {kw.get("world_filename") for kw in _call_kwargs(src, "WorldMeshesModel")}
    assert worlds, f"{name}: sim 이 WorldMeshesModel 을 만들지 않는다"
    assert worlds == {prof.fabrics.world}, (
        f"{name}: sim world={worlds} != 프로필 {prof.fabrics.world}"
    )


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_fabric_flags_match_deployment(name):
    """`graph_capturable`/`use_hand_fabric` 은 배포와 같아야 한다.

    use_hand_fabric=True 로 바뀌면 손이 fabric 제어로 넘어가 action 의미가 통째로 달라진다.
    """
    prof = load_robot_profile(name)
    src = _env_path_for(prof).read_text()
    for kw in _call_kwargs(src, prof.fabrics.class_name):
        assert kw.get("graph_capturable") is False
        assert kw.get("use_hand_fabric") is False


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_reset_fabric_damping_matches_constant(name):
    """리셋 fabric 감쇠는 cfg 가 아니라 env 코드에 박힌 리터럴이라 놓치기 쉽다."""
    from robot_profile import RESET_FABRICS_DAMPING_GAIN

    prof = load_robot_profile(name)
    src = _env_path_for(prof).read_text()
    m = re.search(r"_reset_damping\s*=\s*([0-9.]+)\s*\*\s*torch\.ones", src)
    assert m, f"{name}: sim 에서 _reset_damping 리터럴을 찾지 못했다"
    assert float(m.group(1)) == pytest.approx(RESET_FABRICS_DAMPING_GAIN)


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_main_fabric_cspace_hand_is_grasp_pose(name):
    """메인 fabric 의 null-space 손 자세 = GRASP_POSE (FULL_GRIP 아님).

    다르면 같은 palm 목표에서 다른 팔 해가 나온다(여유자유도 해소가 달라짐).
    """
    prof = load_robot_profile(name)
    src = _env_path_for(prof).read_text()
    assert re.search(
        r"cspace_default\[:,\s*NUM_ARM_DOF:\]\s*=\s*self\.hand_grasp_pose", src
    ), f"{name}: sim 메인 fabric cspace 손이 hand_grasp_pose 가 아니다"


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_fabric_arm_attractor_is_home_at_reset(name):
    """리셋 시 팔 null-space attractor = pregrasp 팔 자세.

    고정 홈 구성에서는 pregrasp 팔 = q_home 이므로 배포의 `default_config[:7] = q_home`
    과 일치한다. sim 이 이걸 바꾸면(예: ARM_START 로 되돌림) 시작 흔들림이 달라진다.
    """
    prof = load_robot_profile(name)
    src = _env_path_for(prof).read_text()
    assert re.search(
        r"default_config\[env_ids,\s*:NUM_ARM_DOF\]\s*=\s*q_pregrasp\[:,\s*:NUM_ARM_DOF\]", src
    ), f"{name}: sim 리셋에서 팔 attractor 설정을 찾지 못했다"
