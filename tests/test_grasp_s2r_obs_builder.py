#!/usr/bin/env python3
"""`grasp_s2r_obs_builder` 테스트 — 우팔 155D 관측 조립. Isaac 불필요."""

from __future__ import annotations

import numpy as np
import pytest

from grasp_s2r_obs_builder import (
    ACTOR_OBS_DIM,
    NUM_FINGERTIPS,
    SEGMENTS,
    assemble_actor_obs,
    hand_dof_order,
    normalized_joint_err,
    reorder,
    rot6d_columns,
    tip_force_local,
)

IDENT = np.array([1.0, 0.0, 0.0, 0.0])
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CANON = [f"r_hj_{f}_{j}" for f in FINGERS for j in range(1, 5)]
#: sim 이 실제로 주는 순서 — 마디(1..4)가 바깥, 손가락이 안쪽이다.
DOF = [f"r_hj_{f}_{j}" for j in range(1, 5) for f in ("index", "middle", "pinky", "ring", "thumb")]


def _inputs(**over) -> dict:
    base = dict(
        arm_q=np.arange(7) * 0.1,
        arm_qd=np.arange(7) * 0.01,
        hand_q=np.arange(20) * 0.05,
        hand_qd=np.arange(20) * 0.005,
        joint_err_profile_order=np.linspace(-0.5, 0.5, 20),
        palm_pos=np.array([0.30, -0.33, 0.41]),
        palm_quat=IDENT,
        tip_pos=np.arange(15).reshape(5, 3) * 0.01,
        cup_pos=np.array([0.37, -0.14, 0.40]),
        goal_pos=np.array([0.36, -0.16, 0.50]),
        tip_force_world=np.zeros((5, 3)),
        tip_quat=np.tile(IDENT, (5, 1)),
        last_action=np.arange(21) * 0.01,
        contact_force_max=10.0,
        joint_pos_err_max=1.2,
    )
    base.update(over)
    return base


# ── 레이아웃 ───────────────────────────────────────────────────────────────
def test_segments_sum_to_the_contract_dimension():
    assert sum(d for _, d in SEGMENTS) == ACTOR_OBS_DIM == 155


def test_segment_order_matches_the_training_env():
    assert [n for n, _ in SEGMENTS] == [
        "arm_q", "arm_qd", "hand_q", "hand_qd", "palm_pos", "palm_ax",
        "tips_rel_palm", "palm_to_obj", "obj_to_tips", "tip_force",
        "joint_err", "actions", "goal_rel",
    ]


def test_assemble_returns_the_contract_length():
    assert assemble_actor_obs(**_inputs()).shape == (155,)


def test_assemble_is_finite():
    assert np.isfinite(assemble_actor_obs(**_inputs())).all()


# ── ★손 관절 순서 ──────────────────────────────────────────────────────────
def test_hand_dof_order_is_phalanx_major_not_finger_major():
    """★sim 은 `index_1, middle_1, pinky_1, ring_1, thumb_1, index_2, …` 로 준다.

    canonical(thumb_1..4, index_1..4, …) 로 넣으면 손 40칸과 joint_err 20칸이
    통째로 스크램블되어, 정책이 죽는 게 아니라 **조용히 이상하게 돈다**
    (09.01 표본 대조: DOF 순 오차 0.024 vs canonical 1.572).
    """
    assert hand_dof_order()[:5] == [
        "r_hj_index_1", "r_hj_middle_1", "r_hj_pinky_1", "r_hj_ring_1", "r_hj_thumb_1"]
    assert hand_dof_order() == DOF


def test_reorder_maps_canonical_values_into_dof_order():
    values = {n: float(i) for i, n in enumerate(CANON)}
    out = reorder([values[n] for n in CANON], CANON, DOF)

    assert out[0] == values["r_hj_index_1"]
    assert out[4] == values["r_hj_thumb_1"]


def test_reorder_refuses_a_name_it_cannot_place():
    with pytest.raises(KeyError, match="없다"):
        reorder([0.0], ["a"], ["b"])


def test_reorder_refuses_a_length_mismatch():
    with pytest.raises(ValueError, match="개수"):
        reorder([0.0, 1.0], ["a"], ["a"])


# ── 항별 ───────────────────────────────────────────────────────────────────
def test_arm_terms_are_absolute_not_relative():
    """좌팔과 달리 우팔은 기본자세를 빼지 않는다 — 빼면 7칸이 어긋난다."""
    obs = assemble_actor_obs(**_inputs())

    assert obs[:7] == pytest.approx(np.arange(7) * 0.1)


def test_palm_ax_is_the_first_two_columns():
    obs = assemble_actor_obs(**_inputs())

    assert obs[57:63] == pytest.approx([1, 0, 0, 0, 1, 0])


def test_tips_are_relative_to_the_palm():
    obs = assemble_actor_obs(**_inputs())
    tips = np.arange(15).reshape(5, 3) * 0.01

    assert obs[63:78] == pytest.approx((tips - np.array([0.30, -0.33, 0.41])).reshape(-1))


def test_palm_to_obj_and_obj_to_tips_use_the_cup():
    obs = assemble_actor_obs(**_inputs())

    assert obs[78:81] == pytest.approx([0.07, 0.19, -0.01], abs=1e-9)
    assert obs[81:84] == pytest.approx(np.array([0.0, 0.01, 0.02]) - np.array([0.37, -0.14, 0.40]))


def test_goal_rel_is_goal_minus_cup():
    obs = assemble_actor_obs(**_inputs())

    assert obs[152:155] == pytest.approx([-0.01, -0.02, 0.10], abs=1e-9)


# ── 정규화 ─────────────────────────────────────────────────────────────────
def test_joint_err_is_normalised_and_keeps_its_sign():
    """부호를 지우면 인벨롭 그립의 주 파지력 신호가 방향을 잃는다."""
    out = normalized_joint_err(np.zeros(3), np.array([0.6, -0.6, 0.0]), 1.2)

    assert out == pytest.approx([0.5, -0.5, 0.0])


def test_joint_err_is_clamped_to_one():
    out = normalized_joint_err(np.zeros(2), np.array([12.0, -12.0]), 1.2)

    assert out == pytest.approx([1.0, -1.0])


def test_tip_force_is_rotated_into_the_tip_frame():
    """실기 F/T 는 센서 로컬 출력이다 — world 를 그대로 넣으면 자세마다 값이 달라진다."""
    q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])   # z 90°
    out = tip_force_local(np.array([[1.0, 0.0, 0.0]]), np.array([q]), 10.0)

    assert out[0] == pytest.approx([0.0, -0.1, 0.0], abs=1e-9)


def test_tip_force_is_normalised_by_the_saturation_point():
    out = tip_force_local(np.array([[5.0, 0.0, 0.0]]), np.array([IDENT]), 10.0)

    assert out[0][0] == pytest.approx(0.5)


def test_rot6d_columns_of_identity():
    assert rot6d_columns(IDENT) == pytest.approx([1, 0, 0, 0, 1, 0])


# ── 입력 검증 ──────────────────────────────────────────────────────────────
def test_assemble_refuses_a_wrong_hand_count():
    with pytest.raises(ValueError, match="20"):
        assemble_actor_obs(**_inputs(hand_q=np.zeros(19)))


def test_assemble_refuses_a_wrong_action_count():
    with pytest.raises(ValueError, match="21"):
        assemble_actor_obs(**_inputs(last_action=np.zeros(20)))


def test_assemble_refuses_a_wrong_tip_count():
    with pytest.raises(ValueError, match=str(NUM_FINGERTIPS)):
        assemble_actor_obs(**_inputs(tip_pos=np.zeros((4, 3))))


def test_tip_body_order_is_finger_canonical_not_alphabetical():
    """★손끝 body 순서는 관절 순서와 다르다. 알파벳순이면 22 cm 어긋난다(표본 대조)."""
    from grasp_s2r_obs_builder import tip_body_order

    assert tip_body_order() == [
        "r_hl_thumb_tip", "r_hl_index_tip", "r_hl_middle_tip",
        "r_hl_ring_tip", "r_hl_pinky_tip"]


def test_tip_and_joint_orders_disagree_on_purpose():
    from grasp_s2r_obs_builder import tip_body_order

    assert [n.split("_")[2] for n in tip_body_order()] != \
           [n.split("_")[2] for n in hand_dof_order()[:5]]
