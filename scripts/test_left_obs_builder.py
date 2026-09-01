#!/usr/bin/env python3
"""`left_obs_builder` 테스트 — 좌팔 grasp_sensor_v2 의 49D 관측 조립. Isaac 불필요."""

from __future__ import annotations

import numpy as np
import pytest

from left_obs_builder import (
    ACTOR_OBS_DIM,
    SEGMENTS,
    assemble_actor_obs,
    cup_upright,
    normalize_tcp,
    quat_to_matrix,
    rot6d_from_quat,
    subtract_frame,
)

BOX = ((0.22, 0.60), (0.10, 0.43), (0.16, 0.60))
IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def _inputs(**over) -> dict:
    base = dict(
        joint_pos=np.arange(9) * 0.01,
        joint_vel=np.arange(9) * 0.001,
        joint_pos_default=np.zeros(9),
        joint_vel_default=np.zeros(9),
        root_pos=np.zeros(3),
        root_quat=IDENT,
        cup_pos=np.array([0.40, 0.25, 0.30]),
        cup_quat=IDENT,
        goal_pos=np.array([0.38, 0.24, 0.45]),
        goal_quat=IDENT,
        tcp_pos=np.array([0.41, 0.26, 0.38]),
        gripper_base_pos=np.array([0.41, 0.26, 0.40]),
        gripper_base_quat=IDENT,
        last_action=np.arange(7) * 0.1,
        gripper_gate=1.0,
        palm_box=BOX,
    )
    base.update(over)
    return base


# ── 레이아웃 ───────────────────────────────────────────────────────────────
def test_segments_sum_to_the_contract_dimension():
    assert sum(d for _, d in SEGMENTS) == ACTOR_OBS_DIM == 49


def test_segment_order_matches_the_training_env():
    """env 에서 뽑은 계약 그대로다 — 순서가 바뀌면 정책이 조용히 이상하게 돈다."""
    assert [n for n, _ in SEGMENTS] == [
        "joint_pos", "joint_vel", "object_position", "target_object_position",
        "actions", "gripper_gate", "tcp_pos", "palm_rot", "goal_minus_cup", "cup_upright",
    ]


def test_assemble_returns_the_contract_length():
    assert assemble_actor_obs(**_inputs()).shape == (49,)


def test_assemble_is_finite():
    assert np.isfinite(assemble_actor_obs(**_inputs())).all()


# ── 항별 ───────────────────────────────────────────────────────────────────
def test_joint_terms_are_relative_to_default():
    """`mdp.joint_pos_rel` 은 기본자세를 뺀 값이다 — 절대값을 넣으면 통째로 어긋난다."""
    obs = assemble_actor_obs(**_inputs(joint_pos_default=np.full(9, 0.02)))

    assert obs[:9] == pytest.approx(np.arange(9) * 0.01 - 0.02)


def test_object_position_is_in_the_root_frame():
    obs = assemble_actor_obs(**_inputs(root_pos=np.array([0.1, 0.0, 0.0])))

    assert obs[18:21] == pytest.approx([0.30, 0.25, 0.30])


def test_target_object_position_carries_pose_seven():
    obs = assemble_actor_obs(**_inputs())

    assert obs[21:24] == pytest.approx([0.38, 0.24, 0.45])
    assert obs[24:28] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_actions_and_gate_land_where_the_contract_says():
    obs = assemble_actor_obs(**_inputs(gripper_gate=0.0))

    assert obs[28:35] == pytest.approx(np.arange(7) * 0.1)
    assert obs[35] == 0.0


def test_goal_minus_cup_is_a_world_difference():
    obs = assemble_actor_obs(**_inputs())

    assert obs[45:48] == pytest.approx([-0.02, -0.01, 0.15])


def test_cup_upright_is_one_for_an_upright_cup():
    assert cup_upright(IDENT) == pytest.approx(1.0)


def test_cup_upright_is_zero_for_a_cup_on_its_side():
    lying = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])   # x 축 90°

    assert cup_upright(lying) == pytest.approx(0.0, abs=1e-9)


# ── 정규화 ─────────────────────────────────────────────────────────────────
def test_normalize_tcp_maps_the_box_centre_to_zero():
    centre = np.array([(0.22 + 0.60) / 2, (0.10 + 0.43) / 2, (0.16 + 0.60) / 2])

    assert normalize_tcp(centre, BOX) == pytest.approx([0, 0, 0])


def test_normalize_tcp_maps_the_box_corner_to_plus_one():
    assert normalize_tcp(np.array([0.60, 0.43, 0.60]), BOX) == pytest.approx([1, 1, 1])


def test_normalize_tcp_is_not_clamped():
    """박스 밖도 그대로 보고한다 — 잘라 버리면 '나갔다'는 사실이 사라진다."""
    assert normalize_tcp(np.array([0.79, 0.43, 0.60]), BOX)[0] == pytest.approx(2.0)


# ── 회전 ───────────────────────────────────────────────────────────────────
def test_rot6d_of_identity_is_the_first_two_columns():
    assert rot6d_from_quat(IDENT) == pytest.approx([1, 0, 0, 0, 1, 0])


def test_rot6d_takes_columns_not_rows():
    """행을 쓰면 전치가 되어 정책이 다른 자세를 본다."""
    q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])   # z 축 90°
    R = quat_to_matrix(q)

    assert rot6d_from_quat(q) == pytest.approx(np.concatenate([R[:, 0], R[:, 1]]))


def test_quat_to_matrix_is_orthonormal():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    R = quat_to_matrix(q)

    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-9)


def test_subtract_frame_undoes_a_root_rotation():
    q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])   # z 90°
    pos, _ = subtract_frame(np.zeros(3), q, np.array([1.0, 0.0, 0.0]), IDENT)

    assert pos == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)


# ── 입력 검증 ──────────────────────────────────────────────────────────────
def test_assemble_refuses_a_wrong_joint_count():
    with pytest.raises(ValueError, match="9"):
        assemble_actor_obs(**_inputs(joint_pos=np.zeros(7)))


def test_assemble_refuses_a_wrong_action_count():
    with pytest.raises(ValueError, match="7"):
        assemble_actor_obs(**_inputs(last_action=np.zeros(6)))
