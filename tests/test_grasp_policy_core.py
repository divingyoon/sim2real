#!/usr/bin/env python3
"""grasp_policy_core 순수 로직 테스트 (fabrics/warp 불필요).

fabric 을 쓰는 부분(FK·IK·홈 rollout)은 IsaacLab 번들 python 이 필요해 여기서 다루지
않는다. 대신 **논리 오류가 숨기 쉬운 순수 부분**을 고정한다: 홈 유효성, palm 목표
clamp(도달성), 접촉 거리게이트.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from grasp_policy_core import (  # noqa: E402
    action_anchor_pose,
    require_anchor_established,
    gate_tip_contact,
    home_pose_to_radians,
    palm_target_from_delta,
    pregrasp_anchor_pose,
    validate_home_in_workspace,
)
from robot_profile import (  # noqa: E402
    HDGP_OPENARM_SRC,
    DELTA_ANCHOR,
    available_profiles,
    profiles_with_convention,
    load_action_anchor,
    load_hdgp_module,
    load_profile_env_cfg,
    load_robot_profile,
)

PROFILES = {"right": "tesollo_bi_s__right", "left": "tesollo_bi_s__left"}
SIDES = ["right", "left"]
# ★이 모듈은 **델타+기준점 규약**(palm delta 6D + 홈/컵 기준점, obs 154D / action 21D)을
#   검증한다. `gripper/left/grasp_sensor` 는 절대 palm 규약이라 a=0 의 뜻부터 다르고
#   손가락 적분기도 lift 래치도 없다 — 그쪽에 이 단언을 적용하면 사실이 아닌 것을 요구하게
#   된다. 이름을 나열하지 않고 **프로필이 선언한 규약**으로 고르므로, 새 델타 구성은
#   자동으로 대상이 되고 규약을 빠뜨린 구성은 로드 단계에서 거부된다.
DELTA_ANCHOR_PROFILES = profiles_with_convention(DELTA_ANCHOR)
ALL_PROFILES = DELTA_ANCHOR_PROFILES

pytestmark = pytest.mark.skipif(not HDGP_OPENARM_SRC.exists(), reason="hdgp 소스 없음")


def _ctx(side):
    return _ctx_by_name(PROFILES[side])


def _ctx_by_name(name):
    prof = load_robot_profile(name)
    cfg = load_profile_env_cfg(prof)
    preset = load_hdgp_module(prof, "preset")
    mins = np.asarray(preset.palm_pose_mins(cfg["max_pose_angle"]), dtype=float)
    maxs = np.asarray(preset.palm_pose_maxs(cfg["max_pose_angle"]), dtype=float)
    home = home_pose_to_radians(cfg["reset_home_palm_pose"])
    return prof, cfg, home, mins, maxs


# --------------------------------------------------------------------------
# 홈 자세
# --------------------------------------------------------------------------

def test_home_pose_degrees_to_radians():
    out = home_pose_to_radians((0.28, -0.38, 0.42, 90.0, 0.0, 90.0))
    assert np.allclose(out[:3], [0.28, -0.38, 0.42])
    assert out[3] == pytest.approx(math.pi / 2)
    assert out[5] == pytest.approx(math.pi / 2)


def test_home_pose_wrong_length_raises():
    with pytest.raises(ValueError):
        home_pose_to_radians((0.1, 0.2, 0.3))


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_configured_home_is_inside_workspace(name):
    """모든 구성의 cfg 홈이 palm workspace 안이어야 한다."""
    _, _, home, mins, maxs = _ctx_by_name(name)
    validate_home_in_workspace(home, mins, maxs)


@pytest.mark.parametrize("side", SIDES)
def test_home_outside_workspace_raises(side):
    """조용히 클램프하지 않고 예외 — 액션 기준점이 어긋나면 의미가 통째로 바뀐다."""
    _, _, home, mins, maxs = _ctx(side)
    bad = home.copy()
    bad[2] = maxs[2] + 0.5
    with pytest.raises(ValueError, match="workspace"):
        validate_home_in_workspace(bad, mins, maxs)


# --------------------------------------------------------------------------
# palm 목표 — ★도달성 회귀
# --------------------------------------------------------------------------

def test_zero_delta_keeps_home():
    _, _, home, mins, maxs = _ctx("right")
    assert np.allclose(palm_target_from_delta(home, np.zeros(6), mins, maxs), home)


def test_clamp_range_relaxed_to_include_home():
    home = np.array([0.28, -0.38, 0.42, 1.57, 0.0, 1.57])
    mins = home + 0.10
    maxs = home + 0.20
    assert np.allclose(palm_target_from_delta(home, np.zeros(6), mins, maxs), home)


def _far_cup_y(cfg, home_y: float) -> float:
    """스폰 박스에서 홈으로부터 **가장 먼** 컵 y 를 유도한다(하드코딩 금지)."""
    c = cfg.get("object_spawn_y_center")
    r = cfg.get("object_spawn_y_range")
    if c is None or r is None:
        pytest.skip("cfg 에 스폰 y 박스가 없다")
    return max(c - r, c + r, key=lambda v: abs(v - home_y))


# --------------------------------------------------------------------------
# 액션 기준점 (anchor) — sim 소스에서 유도한다. 손으로 선언하면 드리프트한다.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_PROFILES)
def test_action_anchor_is_derived_from_sim_source(name):
    """anchor 는 hdgp env 소스에서 읽는다 — 프로필에 적어두면 sim 변경과 어긋난다."""
    prof = load_robot_profile(name)
    assert load_action_anchor(prof) in ("home", "cup")


def test_grasp_v1_anchors_at_home_and_grasp_sensor_at_cup():
    """08.19 hdgp c99b37d 이후 두 트랙의 기준점이 갈렸다 — 그 사실을 고정한다."""
    assert load_action_anchor(load_robot_profile("tesollo_bi_s__right")) == "home"
    assert load_action_anchor(load_robot_profile("tesollo_bi_s__left")) == "home"
    assert load_action_anchor(load_robot_profile("tesollo_sensor__right")) == "cup"


def test_pregrasp_anchor_is_cup_plus_offset_clamped():
    cfg = {"pregrasp_offset_x": -0.06, "pregrasp_offset_y": -0.07, "pregrasp_offset_z": 0.0}
    mins = np.array([0.0, -1.0, 0.0, -9, -9, -9], dtype=float)
    maxs = np.array([1.0, 1.0, 1.0, 9, 9, 9], dtype=float)
    out = pregrasp_anchor_pose(np.array([0.30, -0.20, 0.30]), cfg, mins, maxs)
    assert out[:3] == pytest.approx([0.24, -0.27, 0.30])
    # 자세는 sim 리터럴 90/0/90 도 = π/2, 0, π/2
    assert out[3:] == pytest.approx([math.pi / 2, 0.0, math.pi / 2])


def test_pregrasp_anchor_respects_workspace_clamp():
    cfg = {"pregrasp_offset_x": 0.0, "pregrasp_offset_y": 0.0, "pregrasp_offset_z": 0.0}
    mins = np.array([0.30, -1.0, 0.0, -9, -9, -9], dtype=float)
    maxs = np.array([1.0, 1.0, 1.0, 9, 9, 9], dtype=float)
    out = pregrasp_anchor_pose(np.array([0.10, -0.20, 0.30]), cfg, mins, maxs)
    assert out[0] == pytest.approx(0.30)


def test_pregrasp_anchor_rejects_bad_cup_shape():
    cfg = {"pregrasp_offset_x": 0.0, "pregrasp_offset_y": 0.0, "pregrasp_offset_z": 0.0}
    mins = np.zeros(6); maxs = np.ones(6)
    with pytest.raises(ValueError, match="컵"):
        pregrasp_anchor_pose(np.array([0.1, 0.2]), cfg, mins, maxs)


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_action_anchor_pose_matches_declared_anchor(name):
    """`action_anchor_pose` 가 anchor 종류에 따라 홈/컵 기준을 고른다."""
    prof, cfg, home, mins, maxs = _ctx_by_name(name)
    anchor = load_action_anchor(prof)
    cup = np.array([0.30, -0.20 if prof.acting_side == "right" else 0.20, 0.30])
    out = action_anchor_pose(anchor, home, cup, cfg, mins, maxs)
    if anchor == "home":
        assert out == pytest.approx(home)
    else:
        assert out == pytest.approx(pregrasp_anchor_pose(cup, cfg, mins, maxs))


def test_action_anchor_pose_rejects_unknown_anchor():
    with pytest.raises(ValueError, match="anchor"):
        action_anchor_pose("elbow", np.zeros(6), np.zeros(3), {}, np.zeros(6), np.ones(6))


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_palm_delta_reaches_far_cup_under_its_own_anchor(name):
    """★도달성 — 기준점 종류에 맞는 조건으로 판정한다.

    home 기준: 홈에서 스폰 박스의 **가장 먼** 컵까지 palm delta 로 닿아야 한다
              (구 스칼라 0.15 로는 구조적 불가였다 — 필요 0.28 m).
    cup 기준 : 기준점이 이미 컵을 따라오므로 필요한 건 **pregrasp offset 을 덮는 것**이다
              (그래야 action 이 pregrasp 에서 컵까지 좁힐 수 있다).

    ★값 하나만 대조하면 오판한다: `grasp_sensor` 의 palm_delta_y 가 0.35→0.15 로 줄어든 것은
    회귀가 아니라 기준점이 홈→컵으로 옮겨간 결과다(hdgp c99b37d).
    """
    prof, cfg, home, mins, maxs = _ctx_by_name(name)
    anchor = load_action_anchor(prof)
    dxyz = np.asarray(cfg["palm_delta_xyz"], dtype=float)

    if anchor == "home":
        cup_y = _far_cup_y(cfg, home[1])
        need = abs(cup_y - home[1])
        assert dxyz[1] >= need, f"{name}: palm_delta_y={dxyz[1]} < 필요 {need:.3f}"
        delta = np.zeros(6)
        delta[1] = dxyz[1] if cup_y > home[1] else -dxyz[1]
        reached = palm_target_from_delta(home, delta, mins, maxs)[1]
        assert abs(reached - home[1]) >= need - 1e-9
    else:
        need = np.abs([cfg["pregrasp_offset_x"], cfg["pregrasp_offset_y"], cfg["pregrasp_offset_z"]])
        assert np.all(dxyz >= need), f"{name}: palm_delta {dxyz} < pregrasp offset {need}"


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_old_scalar_delta_cannot_reach(name):
    """**홈 기준** 구성에서 구 스칼라 0.15 로는 못 간다 — 회귀가 되살아나면 여기서 잡힌다.

    컵 기준 구성에는 해당하지 않는다(기준점이 컵을 따라오므로 0.15 로 충분하다).
    """
    prof, cfg, home, mins, maxs = _ctx_by_name(name)
    if load_action_anchor(prof) != "home":
        pytest.skip("컵 기준 구성 — 홈 기준 도달성 조건이 적용되지 않는다")
    cup_y = _far_cup_y(cfg, home[1])
    delta = np.zeros(6)
    delta[1] = 0.15 if cup_y > home[1] else -0.15
    reached = palm_target_from_delta(home, delta, mins, maxs)[1]
    assert abs(reached - home[1]) < abs(cup_y - home[1])


def test_action_semantics_not_mirrored_between_sides():
    """★palm delta 는 좌우 **동일**하다 — 배포에서 미러 부호를 넣으면 안 된다."""
    _, r_cfg, _, _, _ = _ctx("right")
    _, l_cfg, _, _, _ = _ctx("left")
    assert r_cfg["palm_delta_xyz"] == l_cfg["palm_delta_xyz"]
    assert r_cfg["palm_delta_rot_deg"] == l_cfg["palm_delta_rot_deg"]


# --------------------------------------------------------------------------
# 접촉 거리 게이트
# --------------------------------------------------------------------------

def _force(idx=0, mag=5.0):
    f = np.zeros((5, 3))
    f[idx] = [0.0, 0.0, mag]
    return f


def test_contact_gated_out_when_palm_far():
    f, b = gate_tip_contact(_force(), palm_cup_dist=0.16, gate_dist=0.10, threshold=0.1)
    assert np.allclose(b, 0.0)
    assert np.allclose(f, 0.0)


def test_contact_kept_when_palm_near():
    f, b = gate_tip_contact(_force(), palm_cup_dist=0.05, gate_dist=0.10, threshold=0.1)
    assert b[0] == 1.0 and b[1:].sum() == 0.0
    assert f[0, 2] == pytest.approx(5.0)


def test_binary_threshold_applied():
    f, b = gate_tip_contact(_force(mag=0.05), palm_cup_dist=0.05, gate_dist=0.10, threshold=0.1)
    assert np.allclose(b, 0.0)
    assert f[0, 2] == pytest.approx(0.05)


def test_force_obs_not_gated_when_disabled():
    f, b = gate_tip_contact(
        _force(), palm_cup_dist=0.16, gate_dist=0.10, threshold=0.1, gate_force_obs=False
    )
    assert np.allclose(b, 0.0)
    assert f[0, 2] == pytest.approx(5.0)


def test_gate_validates_shape():
    with pytest.raises(ValueError):
        gate_tip_contact(np.zeros((5, 2)), 0.05, 0.10, 0.1)


# --------------------------------------------------------------------------
# 기준점 확립 게이트 — 컵 기준 구성은 리셋 전에 명령을 내면 안 된다
# --------------------------------------------------------------------------

def test_home_anchor_never_requires_establishment():
    require_anchor_established("home", established=False)   # 예외 없음


def test_cup_anchor_raises_before_reset():
    with pytest.raises(RuntimeError, match="기준점"):
        require_anchor_established("cup", established=False)


def test_cup_anchor_ok_after_reset():
    require_anchor_established("cup", established=True)
