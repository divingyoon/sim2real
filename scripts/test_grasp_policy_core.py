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

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from grasp_policy_core import (  # noqa: E402
    gate_tip_contact,
    home_pose_to_radians,
    palm_target_from_delta,
    validate_home_in_workspace,
)
from robot_profile import (  # noqa: E402
    HDGP_OPENARM_SRC,
    load_hdgp_module,
    load_profile_env_cfg,
    load_robot_profile,
)

PROFILES = {"right": "tesollo_bi_s__right", "left": "tesollo_bi_s__left"}
SIDES = ["right", "left"]

pytestmark = pytest.mark.skipif(not HDGP_OPENARM_SRC.exists(), reason="hdgp 소스 없음")


def _ctx(side):
    prof = load_robot_profile(PROFILES[side])
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


@pytest.mark.parametrize("side", SIDES)
def test_configured_home_is_inside_workspace(side):
    """실제 cfg 홈이 palm workspace 안이어야 한다(양측)."""
    _, _, home, mins, maxs = _ctx(side)
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


@pytest.mark.parametrize("side,cup_y", [("right", -0.10), ("left", 0.10)])
def test_palm_delta_y_reaches_far_cup(side, cup_y):
    """★도달성: 홈 y(∓0.38)에서 가장 먼 컵 y(∓0.10)까지 닿아야 한다.

    스폰 박스 y = 우측 [-0.30,-0.10] / 좌측 [+0.10,+0.30] → 필요 |Δy| = 0.28.
    palm_delta_xyz[1]=0.35 이면 도달, 구 스칼라 0.15 면 **구조적 불가**.
    """
    _, cfg, home, mins, maxs = _ctx(side)
    need = abs(cup_y - home[1])
    assert need == pytest.approx(0.28, abs=1e-9)

    dy = cfg["palm_delta_xyz"][1]
    assert dy >= need, f"palm_delta_y={dy} < 필요 {need}"

    delta = np.zeros(6)
    delta[1] = dy if cup_y > home[1] else -dy
    reached = palm_target_from_delta(home, delta, mins, maxs)[1]
    assert abs(reached - home[1]) >= need - 1e-9


@pytest.mark.parametrize("side,cup_y", [("right", -0.10), ("left", 0.10)])
def test_old_scalar_delta_cannot_reach(side, cup_y):
    """구 스칼라 0.15 로는 못 간다 — 회귀가 되살아나면 여기서 잡힌다."""
    _, _, home, mins, maxs = _ctx(side)
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
