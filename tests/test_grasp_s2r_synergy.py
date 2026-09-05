#!/usr/bin/env python3
"""`grasp_s2r_synergy` 계약 테스트 — env `_synergy_targets` 와 1:1 인지 본다."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from grasp_s2r_synergy import (  # noqa: E402
    FINGERS,
    HAND_JOINT_NAMES,
    SynergyCfg,
    SynergyHand,
    cfg_from_run,
)

RUN = Path(__file__).resolve().parents[1] / "logs/policy/right_g1/params/env.yaml"
N = len(HAND_JOINT_NAMES)


def _open_hand(**kw):
    return SynergyHand(SynergyCfg(**kw))


def test_joint_order_is_profile_order_not_dof_order():
    """★프로필 순(`r_hj_{finger}_{1..4}`)이다 — obs 의 손 40칸(DOF 순)과 다르다."""
    assert HAND_JOINT_NAMES[:4] == ("r_hj_thumb_1", "r_hj_thumb_2",
                                    "r_hj_thumb_3", "r_hj_thumb_4")
    assert len(HAND_JOINT_NAMES) == 20


def test_channels_are_three_per_finger():
    """coupled3 — [외전, MCP, PIP·DIP 공통]. `_3` 과 `_4` 가 같은 채널이다."""
    h = _open_hand()
    assert h.n_ch == 3
    ch = dict(zip(HAND_JOINT_NAMES, h.ch))
    assert ch["r_hj_index_3"] == ch["r_hj_index_4"] == 2
    assert ch["r_hj_index_2"] == 1 and ch["r_hj_index_1"] == 0


def test_action_is_absolute_closure_not_velocity():
    """★액션은 절대 폐쇄도 목표다. 같은 액션을 계속 줘도 그 값에서 **멈춰야** 한다."""
    h = _open_hand(close_speed=0.5)
    tgt = h.step(np.zeros(15))              # a=0 → 폐쇄도 0.5
    for _ in range(19):
        tgt = h.step(np.zeros(15))          # 계속 줘도 0.5 에서 멈춘다
    assert h.close[h.movable].max() == pytest.approx(0.5, abs=1e-9)
    assert np.isfinite(tgt).all()


def test_close_speed_limits_rate_not_target():
    """`close_speed` 는 목표를 향한 변화율 상한이다."""
    h = _open_hand(close_speed=0.01)
    h.step(np.ones(15))                      # 목표 폐쇄도 1.0
    assert h.close[h.movable].max() == pytest.approx(0.01, abs=1e-12)
    h.step(np.ones(15))
    assert h.close[h.movable].max() == pytest.approx(0.02, abs=1e-12)


def test_close_gate_scales_closing_only():
    """★게이트는 닫는 방향만 막는다 — 푸는 방향은 항상 통과해야 한다."""
    h = _open_hand(close_speed=0.1)
    h.step(np.ones(15), close_gate=0.0)
    assert h.close.max() == 0.0              # 게이트 0 이면 못 닫는다
    h2 = _open_hand(close_speed=0.1)
    h2.step(np.ones(15))                     # 먼저 조금 닫고
    before = h2.close.copy()
    h2.step(-np.ones(15), close_gate=0.0)    # 게이트 0 이어도 풀려야 한다
    assert h2.close[h2.movable].max() < before[h2.movable].max()


def test_four_fingers_are_coupled_thumb_is_independent():
    """엄지만 독립. 4지는 채널별 평균이라 개별 지령이 사라진다."""
    h = _open_hand(close_speed=1.0, couple_four_fingers=True)
    a = np.zeros((5, 3))
    a[FINGERS.index("index"), 2] = 1.0       # 검지만 강하게
    a[FINGERS.index("thumb"), 2] = -1.0      # 엄지는 반대로
    h.step(a.reshape(-1))
    close = dict(zip(HAND_JOINT_NAMES, h.close))
    # 검지·중지·약지·소지의 `_3` 폐쇄도가 같아야 한다(평균으로 뭉개짐).
    vals = [close[f"r_hj_{f}_3"] for f in ("index", "middle", "ring", "pinky")]
    assert max(vals) - min(vals) < 1e-12
    # 엄지는 그 평균과 달라야 한다.
    assert abs(close["r_hj_thumb_3"] - vals[0]) > 1e-6


def test_residual_scale_one_restores_individual_commands():
    """잔차 1.0 이면 커플링이 항등이 된다(15채널과 동일)."""
    h = _open_hand(close_speed=1.0, residual_scale=1.0)
    a = np.zeros((5, 3))
    a[FINGERS.index("index"), 2] = 1.0
    h.step(a.reshape(-1))
    close = dict(zip(HAND_JOINT_NAMES, h.close))
    assert close["r_hj_index_3"] > close["r_hj_middle_3"]


def test_oppose_knob_rewrites_thumb_grip_pose():
    """★g1 은 `oppose_grip_delta_rad=-0.6` — 엄지 ch1 의 grip 이 open+δ 로 바뀐다."""
    h = _open_hand(oppose_grip_delta_rad=-0.6)
    i = HAND_JOINT_NAMES.index("r_hj_thumb_2")
    assert h.grip_pose[i] == pytest.approx(h.open_pose[i] - 0.6)
    # 그 결과 엄지 `_2` 가 **가동 관절이 된다**(기본표에서는 open==grip 이라 고정이었다).
    assert bool(h.movable[i]) is True


def test_immovable_joints_stay_put():
    """open == grip 인 관절은 명령해도 안 움직인다(전 `_1`)."""
    h = _open_hand(close_speed=1.0, oppose_grip_delta_rad=0.0)
    tgt = h.step(np.ones(15))
    for f in FINGERS:
        i = HAND_JOINT_NAMES.index(f"r_hj_{f}_1")
        assert tgt[i] == pytest.approx(h.open_pose[i])


def test_contact_freeze_joint_scope_locks_own_link_only():
    """★관절 단위 동결 — `_3` 은 중간마디 접촉, `_4` 는 원위 접촉으로만 잠긴다."""
    h = _open_hand(close_speed=0.1, freeze_scope="joint")
    mid = np.zeros(5, dtype=bool)
    dist = np.zeros(5, dtype=bool)
    mid[FINGERS.index("index")] = True       # 검지 중간마디만 닿음
    h.step(np.ones(15), contact_mid=mid, contact_dist=dist)
    close = dict(zip(HAND_JOINT_NAMES, h.close))
    assert close["r_hj_index_3"] == 0.0      # 얼었다
    assert close["r_hj_index_4"] > 0.0       # 원위는 계속 감긴다 — 이게 감쌈이다


def test_finger_scope_locks_whole_finger():
    h = _open_hand(close_speed=0.1, freeze_scope="finger")
    mid = np.zeros(5, dtype=bool)
    mid[FINGERS.index("index")] = True
    h.step(np.ones(15), contact_mid=mid, contact_dist=np.zeros(5, dtype=bool))
    close = dict(zip(HAND_JOINT_NAMES, h.close))
    assert close["r_hj_index_3"] == 0.0 and close["r_hj_index_4"] == 0.0
    assert close["r_hj_middle_3"] > 0.0      # 다른 손가락은 영향 없음


def test_blocked_needs_error_and_room_to_limit():
    """★실기 대체 신호 — 오차 조건만으로는 안 된다(과지령이라 허공에서도 성립)."""
    lim = np.tile(np.array([-1.571, 1.571]), (20, 1))
    h = SynergyHand(SynergyCfg(close_speed=0.5), soft_limits=lim)
    h.step(np.ones(15))                       # 목표를 크게 밀어둔다
    q_far = np.zeros(20)                      # 실측은 0 — 오차 크고 한계에서 멀다
    assert h.blocked(q_far)[h.movable].any()
    q_at_limit = np.full(20, 1.571)           # 한계에 붙음 → 막힌 게 아니라 한계다
    assert not h.blocked(q_at_limit).any()


def test_stall_freeze_stops_closing_but_allows_release():
    h = SynergyHand(SynergyCfg(close_speed=0.5),
                    soft_limits=np.tile(np.array([-1.571, 1.571]), (20, 1)))
    h.step(np.ones(15))
    stalled = np.zeros(20)                    # 실측이 안 따라옴 = 막힘
    before = h.close.copy()
    h.step(np.ones(15), hand_q=stalled)
    frozen = h.freeze_mid | h.freeze_dist
    assert np.allclose(h.close[frozen], before[frozen])   # 더 안 닫힌다
    h.step(-np.ones(15), hand_q=stalled)
    assert h.close[frozen].max() < before[frozen].max()   # 풀리는 건 된다


def test_soft_limits_absorb_overcommanded_grip():
    """grip 1.8 은 한계(1.571) 초과 과지령 — 클램프가 흡수해야 한다."""
    lim = np.tile(np.array([-1.571, 1.571]), (20, 1))
    h = SynergyHand(SynergyCfg(close_speed=1.0), soft_limits=lim)
    tgt = h.step(np.ones(15))
    assert tgt.max() <= 1.571 + 1e-12


def test_reset_clears_closure():
    h = _open_hand(close_speed=0.5)
    h.step(np.ones(15))
    h.reset()
    assert h.close.max() == 0.0


@pytest.mark.skipif(not RUN.exists(), reason="g1 런 dump 없음")
def test_cfg_from_run_reads_g1_contract():
    c = cfg_from_run(RUN)
    assert c.close_speed == pytest.approx(0.005)
    assert c.couple_four_fingers is True
    assert c.residual_scale == pytest.approx(0.0)
    assert c.hand_layout == "coupled3"
    assert c.oppose_grip_delta_rad == pytest.approx(-0.6)
    assert c.freeze_scope == "joint"
    assert c.blocked_err_thr_rad == pytest.approx(0.3)
    assert c.blocked_limit_eps_rad == pytest.approx(0.05)
