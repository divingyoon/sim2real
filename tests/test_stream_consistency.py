#!/usr/bin/env python3
"""폐루프 기록 스트림(v2 스키마)의 자기일관성 — Isaac 없이 numpy 로 잡는 회귀.

스트림은 `probe_grasp_then_carry.py --save-stream` 의 산출물이고, 통합 씬 폐루프
러너(`probe_bimanual_closedloop.py`)의 goal·스폰·재현 대조 기준이다. 여기서
깨지면 러너가 엉뚱한 기준으로 돈다 — 실행 전에 여기서 울어야 한다.

관측은 **스스로를 검증한다**: obs 안의 (palm, 컵상대, goal상대) 세 항과 기록된
컵·goal·관절이 맞물려야 하고, 래치 뒤에는 obs 의 컵이 palm 강체 추정으로
바뀌어야 한다(`obs_object_rigid_after_latch`). 그 규약이 뒤틀리면 러너의 우팔
관측 바인딩도 같은 방식으로 뒤틀린 것이다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RIGHT = REPO / "logs" / "shadow" / "pour_entry" / "stream_right_e1_v2.npz"
LEFT = REPO / "logs" / "shadow" / "pour_entry" / "stream_left_v2b25.npz"

# 우 155D obs 오프셋 (logs/policy/right_e1/obs_layout.json 과 동일 계약)
R_HAND_Q = slice(14, 34)
R_PALM = slice(54, 57)
R_PALM_ROT6 = slice(57, 63)
R_PALM_TO_OBJ = slice(78, 81)
R_GOAL_REL = slice(152, 155)


def _load(p: Path):
    if not p.exists():
        pytest.skip(f"스트림이 없다: {p.name}")
    return np.load(p, allow_pickle=True)


# ── 스키마 ─────────────────────────────────────────────────────────────────
def test_right_stream_has_the_closedloop_keys():
    z = _load(RIGHT)
    need = {"actions", "obs", "arm_target", "hand_target", "arm_meas", "hand_meas",
            "cup_pos3", "goal", "palm_targets", "syn_target", "close_gate",
            "latched", "meta_cup_spawn"}
    assert need <= set(z.files), f"빠진 키: {need - set(z.files)}"


def test_left_stream_has_the_closedloop_keys():
    z = _load(LEFT)
    need = {"actions", "obs", "arm_target", "hand_target", "arm_meas",
            "cup_pos3", "goal", "palm_targets", "meta_cup_spawn"}
    assert need <= set(z.files), f"빠진 키: {need - set(z.files)}"


def test_contract_dimensions():
    zr, zl = _load(RIGHT), _load(LEFT)
    assert zr["obs"].shape[1] == 155 and zr["actions"].shape[1] == 21
    assert zl["obs"].shape[1] == 49 and zl["actions"].shape[1] == 7
    assert zr["goal"].shape[1] == 3 and zl["goal"].shape[1] == 7


def test_actions_are_bounded_per_policy_regime():
    """우(tanh mu)는 유계. 좌(MLP mu)는 **raw 저장**이 맞다 — 텀이 내부 클램프하고
    obs `actions` 는 raw 를 본다. 러너가 사전 클램프하면 학습과 다른 행동이 된다."""
    assert np.abs(_load(RIGHT)["actions"]).max() <= 1.0 + 1e-5
    la = _load(LEFT)["actions"]
    assert np.isfinite(la).all() and np.abs(la).max() < 5.0


# ── 관측 ↔ 상태 자기일관성 (우) ────────────────────────────────────────────
def test_right_obs_hand_matches_measured_joints():
    """obs 손 관절 = 실측(+노이즈 σ0.01). ★두 함정을 동시에 다룬다:
    ①기록 배열은 프로필 순(finger-major), obs 는 sim DOF 순 — 재정렬 필수
    ②obs[k] 는 스텝 전, meas[k] 는 스텝 후 — 한 스텝 시차 정렬 필수."""
    from grasp_s2r_obs_builder import hand_dof_order
    z = _load(RIGHT)
    rec = [str(x) for x in z["meta_hand_names"]]
    perm = [rec.index(n) for n in hand_dof_order("r")]
    d = np.abs(z["obs"][1:, R_HAND_Q] - z["hand_meas"][:-1][:, perm])
    assert d.max() < 0.08, f"손 관절 obs↔실측 최대 {d.max():.3f} rad — 순서 스크램블 의심"


def test_right_obs_object_matches_recorded_cup_before_latch():
    """래치 전 obs 물체 = 실제 컵 (palm 상대). 래치 전 구간만 잰다."""
    z = _load(RIGHT)
    pre = z["latched"] < 0.5
    if pre.sum() < 5:
        pytest.skip("래치 전 구간이 짧다")
    pre1 = pre[:-1]
    est = z["obs"][1:][pre1, R_PALM] + z["obs"][1:][pre1, R_PALM_TO_OBJ]
    d = np.linalg.norm(est - z["cup_pos3"][:-1][pre1], axis=1)
    # 한 스텝 시차 정렬. E1 은 obs 물체 노이즈 σ15mm — 3σ 초과만 잡는다.
    assert d.max() < 0.09, f"래치 전 obs 컵 오차 최대 {d.max()*1000:.0f} mm"


def test_right_obs_object_switches_to_rigid_estimate_after_latch():
    """★래치 뒤 obs 컵 = palm 강체 추정 — 실기에서 손이 컵을 가릴 때의 그 규약.

    러너의 우팔 관측 바인딩이 지켜야 할 핵심 거동이라 스트림에서 잠근다.
    """
    z = _load(RIGHT)
    lat = z["latched"] > 0.5
    if lat.sum() < 10:
        pytest.skip("래치 구간이 짧다")
    i0 = int(np.argmax(lat))
    obs_obj = z["obs"][:, R_PALM] + z["obs"][:, R_PALM_TO_OBJ]

    def rot(row):
        c0, c1 = row[R_PALM_ROT6][:3], row[R_PALM_ROT6][3:]
        c2 = np.cross(c0, c1)
        return np.stack([c0, c1, c2], axis=1)

    off = rot(z["obs"][i0]).T @ (obs_obj[i0] - z["obs"][i0, R_PALM])
    later = range(i0, len(z["obs"]))
    err = [np.linalg.norm(z["obs"][t, R_PALM] + rot(z["obs"][t]) @ off - obs_obj[t])
           for t in later]
    # 스냅샷과 각 프레임 양쪽에 노이즈(σ15mm·coherent)가 얹힌다 — 굵은 프레임 오류만 잡는다.
    assert float(np.median(err)) < 0.05 and max(err) < 0.15, \
        f"래치 후 강체 모델 위반 중앙값 {np.median(err)*1000:.0f} / 최대 {max(err)*1000:.0f} mm"


def test_right_goal_rel_ties_goal_cup_and_obs_together():
    z = _load(RIGHT)
    pre = (z["latched"] < 0.5)[:-1]
    d = np.linalg.norm(
        z["obs"][1:][pre, R_GOAL_REL]
        - (z["goal"][:-1][pre] - z["cup_pos3"][:-1][pre]), axis=1)
    assert d.max() < 0.09, f"goal_rel 불일치 최대 {d.max()*1000:.0f} mm"


def test_right_latch_engages_and_stays():
    """opposition 래치는 단조다 — 걸렸다 풀리면 발췌 이식이 잘못된 것."""
    z = _load(RIGHT)
    lat = z["latched"] > 0.5
    assert lat.any(), "래치가 한 번도 안 걸렸다 — 파지 기록이 아니다"
    i0 = int(np.argmax(lat))
    assert lat[i0:].all(), "래치가 중간에 풀렸다 (단조 위반)"


# ── 좌 ─────────────────────────────────────────────────────────────────────
def test_left_obs_joints_match_measured_relative_to_home():
    """좌 obs joint 항은 홈 상대(joint_pos_rel). 실측 − obs = 홈(상수)여야 한다."""
    z = _load(LEFT)
    home = z["arm_meas"][:-1] - z["obs"][1:, :7]   # 한 스텝 시차 정렬
    spread = home.std(axis=0).max()
    assert spread < 0.01, f"홈 추정 산포 {spread:.4f} rad — rel 규약이 깨졌다"


def test_left_goal_is_constant_within_episode():
    """65프레임(1.3s) < 리샘플 5s — goal 이 도중에 바뀌면 러너의 고정 goal 이 틀린다."""
    z = _load(LEFT)
    assert np.abs(z["goal"] - z["goal"][0]).max() < 1e-6


def test_left_gate_is_recorded_in_obs():
    z = _load(LEFT)
    gate = z["obs"][:, 35]
    assert set(np.round(gate).tolist()) <= {0.0, 1.0}


def test_cup_spawn_matches_first_frame():
    for z in (_load(RIGHT), _load(LEFT)):
        d = np.linalg.norm(z["meta_cup_spawn"] - z["cup_pos3"][0])
        assert d < 0.02, f"스폰 메타 ↔ 첫 프레임 컵 {d*1000:.0f} mm"
