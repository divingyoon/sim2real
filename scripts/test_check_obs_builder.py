#!/usr/bin/env python3
"""배포 obs 빌더 ↔ 학습 env 표본 대조. **Isaac 없이** 회귀를 잡는다.

표본은 `probe_obs_layout.py` 가 sim 에서 남긴 것이고 저장소에 들어 있다. 빌더가
드리프트하면 이 테스트가 먼저 운다 — 실기에서 알게 되면 늦다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from check_obs_builder import TOLERANCES, SegmentDiff, compare, describe, load_layout

REPO = Path(__file__).resolve().parents[1]
RIGHT_LAYOUT = REPO / "logs" / "policy" / "right_e1" / "obs_layout.json"


# ── 순수 로직 ──────────────────────────────────────────────────────────────
SEGS = (("a", 2), ("b", 3))


def test_compare_reports_one_diff_per_segment():
    diffs = compare(np.zeros(5), np.zeros(5), SEGS)

    assert [d.name for d in diffs] == ["a", "b"]
    assert [d.offset for d in diffs] == [0, 2]


def test_compare_finds_the_worst_element_in_a_segment():
    built = np.array([0.0, 0.0, 0.0, 0.9, 0.0])

    assert compare(built, np.zeros(5), SEGS)[1].max_abs == pytest.approx(0.9)


def test_a_segment_within_tolerance_is_ok():
    assert compare(np.full(5, 0.01), np.zeros(5), SEGS)[0].ok


def test_a_segment_beyond_tolerance_is_not_ok():
    assert not compare(np.full(5, 0.9), np.zeros(5), SEGS)[0].ok


def test_velocity_segments_get_a_looser_tolerance():
    """속도 관측은 DR 노이즈가 크다 — 위치와 같은 잣대를 대면 거짓 경보가 난다."""
    assert TOLERANCES["arm_qd"] > 0.05


def test_compare_refuses_a_length_mismatch():
    with pytest.raises(ValueError, match="길이"):
        compare(np.zeros(4), np.zeros(5), SEGS)


def test_compare_refuses_segments_that_do_not_add_up():
    with pytest.raises(ValueError, match="세그먼트 합"):
        compare(np.zeros(5), np.zeros(5), (("a", 2),))


def test_describe_names_the_segments_that_failed():
    text = describe(compare(np.array([0, 0, 9.0, 0, 0]), np.zeros(5), SEGS))

    assert "b" in text and "불일치" in text


def test_describe_says_so_when_everything_matches():
    assert "일치" in describe(compare(np.zeros(5), np.zeros(5), SEGS))


def test_load_layout_refuses_a_file_without_a_sample(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"obs_dim": 3}')

    with pytest.raises(KeyError, match="sample_obs"):
        load_layout(p)


# ── ★실제 계약 회귀 ────────────────────────────────────────────────────────
@pytest.mark.skipif(not RIGHT_LAYOUT.exists(), reason="표본이 없다")
def test_right_builder_matches_the_recorded_env_sample():
    """우팔 155D 빌더가 학습 env 표본과 일치하는가 — 드리프트를 여기서 잡는다."""
    from grasp_s2r_obs_builder import (
        SEGMENTS, assemble_actor_obs, hand_dof_order, tip_body_order)

    d = load_layout(RIGHT_LAYOUT)
    obs = np.array(d["sample_obs"])
    st = d["state"]
    names, bod = st["joint_names"], st["body_names"]
    q, qd = np.array(st["joint_pos"]), np.array(st["joint_vel"])
    qt = np.array(st["joint_pos_target"])
    bp = np.array(st["body_pos_env_local"]).reshape(-1, 3)
    bq = np.array(st["body_quat_wxyz"]).reshape(-1, 4)

    ia = [names.index(f"r_aj_{i}") for i in range(1, 8)]
    ih = [names.index(n) for n in hand_dof_order("r")]
    pi = bod.index("r_hl_palm")
    # 컵·목표는 표본 obs 에서 역산한다 — 표본이 스스로 담고 있는 값이다.
    palm = obs[54:57]
    cup, goal = palm + obs[78:81], palm + obs[78:81] + obs[152:155]

    built = assemble_actor_obs(
        arm_q=q[ia], arm_qd=qd[ia], hand_q=q[ih], hand_qd=qd[ih], hand_target=qt[ih],
        palm_pos=bp[pi], palm_quat=bq[pi],
        tip_pos=np.array([bp[bod.index(t)] for t in tip_body_order("r")]),
        cup_pos=cup, goal_pos=goal,
        tip_force_world=np.zeros((5, 3)), tip_quat=np.tile([1.0, 0, 0, 0], (5, 1)),
        last_action=obs[131:152], contact_force_max=10.0, joint_pos_err_max=1.2)

    diffs = compare(built, obs, SEGMENTS)
    assert all(d.ok for d in diffs), "\n" + describe(diffs)


@pytest.mark.skipif(not RIGHT_LAYOUT.exists(), reason="표본이 없다")
def test_canonical_hand_order_would_be_caught():
    """★순서를 canonical 로 바꾸면 이 하네스가 반드시 잡아야 한다.

    안 잡히면 하네스가 무의미하다 — 그 사고가 바로 09.01 에 있었던 것이다.
    """
    from grasp_s2r_obs_builder import SEGMENTS, assemble_actor_obs, tip_body_order

    d = load_layout(RIGHT_LAYOUT)
    obs = np.array(d["sample_obs"])
    st = d["state"]
    names, bod = st["joint_names"], st["body_names"]
    q, qd = np.array(st["joint_pos"]), np.array(st["joint_vel"])
    qt = np.array(st["joint_pos_target"])
    bp = np.array(st["body_pos_env_local"]).reshape(-1, 3)
    bq = np.array(st["body_quat_wxyz"]).reshape(-1, 4)

    canon = [f"r_hj_{f}_{j}" for f in ("thumb", "index", "middle", "ring", "pinky")
             for j in range(1, 5)]
    ih = [names.index(n) for n in canon]                      # ← 일부러 틀린 순서
    ia = [names.index(f"r_aj_{i}") for i in range(1, 8)]
    pi = bod.index("r_hl_palm")
    palm = obs[54:57]
    cup, goal = palm + obs[78:81], palm + obs[78:81] + obs[152:155]

    built = assemble_actor_obs(
        arm_q=q[ia], arm_qd=qd[ia], hand_q=q[ih], hand_qd=qd[ih], hand_target=qt[ih],
        palm_pos=bp[pi], palm_quat=bq[pi],
        tip_pos=np.array([bp[bod.index(t)] for t in tip_body_order("r")]),
        cup_pos=cup, goal_pos=goal,
        tip_force_world=np.zeros((5, 3)), tip_quat=np.tile([1.0, 0, 0, 0], (5, 1)),
        last_action=obs[131:152], contact_force_max=10.0, joint_pos_err_max=1.2)

    bad = [d.name for d in compare(built, obs, SEGMENTS) if not d.ok]
    assert "hand_q" in bad, "손 관절 순서를 틀렸는데 하네스가 못 잡았다"


LEFT_LAYOUT = REPO / "logs" / "policy" / "left_v2B25" / "obs_layout.json"
LEFT_PARAMS = REPO / "logs" / "policy" / "left_v2B25" / "params" / "env.yaml"


@pytest.mark.skipif(not LEFT_LAYOUT.exists(), reason="표본이 없다")
def test_left_builder_matches_the_recorded_env_sample():
    """좌팔 49D 빌더 ↔ env 표본. rot6d 인터리브 규약이 여기서 잠긴다."""
    import yaml
    from left_obs_builder import SEGMENTS, assemble_actor_obs, quat_to_matrix
    from robot_profile import load_hdgp_module, load_robot_profile

    d = load_layout(LEFT_LAYOUT)
    obs = np.array(d["sample_obs"])
    st = d["state"]
    names, bod = st["joint_names"], st["body_names"]
    q, qd = np.array(st["joint_pos"]), np.array(st["joint_vel"])
    bp = np.array(st["body_pos_env_local"]).reshape(-1, 3)
    bq = np.array(st["body_quat_wxyz"]).reshape(-1, 4)

    left9 = [f"l_aj_{i}" for i in range(1, 8)] + ["l_hj_gripper_1", "l_hj_gripper_2"]
    il = [names.index(n) for n in left9]
    jp = yaml.unsafe_load(LEFT_PARAMS.read_text())["scene"]["robot"]["init_state"]["joint_pos"]
    q0 = np.array([jp.get(n, 0.0) for n in left9])

    preset = load_hdgp_module(load_robot_profile("gripper_left"), "preset")
    box = (preset.PALM_BOX_X, preset.PALM_BOX_Y, preset.PALM_BOX_Z)
    gi = bod.index(preset.GRIPPER_BASE_BODY)
    # TCP = gripper_base + R·(0,0,TCP_OFFSET_IN_BASE_Z) — 표본 실측 80 mm 와 일치.
    tcp = bp[gi] + quat_to_matrix(bq[gi]) @ np.array([0.0, 0.0, preset.TCP_OFFSET_IN_BASE_Z])

    built = assemble_actor_obs(
        joint_pos=q[il], joint_vel=qd[il], joint_pos_default=q0,
        joint_vel_default=np.zeros(9),
        root_pos=np.zeros(3), root_quat=np.array([1.0, 0, 0, 0]),
        cup_pos=np.array(st["object_pos_env_local"]),
        cup_quat=np.array(st["object_quat_wxyz"]),
        goal_pos=obs[21:24], goal_quat=obs[24:28],       # 목표는 표본이 스스로 담는 값
        tcp_pos=tcp, gripper_base_pos=bp[gi], gripper_base_quat=bq[gi],
        last_action=obs[28:35], gripper_gate=float(obs[35]),
        palm_box=box)

    diffs = compare(built, obs, SEGMENTS)
    assert all(d.ok for d in diffs), "\n" + describe(diffs)


@pytest.mark.skipif(not LEFT_LAYOUT.exists(), reason="표본이 없다")
def test_left_column_stacked_rot6d_would_be_caught():
    """★rot6d 를 열 스택으로 바꾸면 하네스가 반드시 잡아야 한다 (표본 오차 1.88)."""
    from left_obs_builder import SEGMENTS, quat_to_matrix

    d = load_layout(LEFT_LAYOUT)
    obs = np.array(d["sample_obs"])
    st = d["state"]
    bod = st["body_names"]
    bq = np.array(st["body_quat_wxyz"]).reshape(-1, 4)
    R = quat_to_matrix(bq[bod.index("l_hl_gripper_base")])

    wrong = obs.copy()
    wrong[39:45] = np.concatenate([R[:, 0], R[:, 1]])    # ← 일부러 열 스택

    bad = [x.name for x in compare(wrong, obs, SEGMENTS) if not x.ok]
    assert "palm_rot" in bad, "rot6d 규약을 틀렸는데 하네스가 못 잡았다"
