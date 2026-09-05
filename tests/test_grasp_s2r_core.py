#!/usr/bin/env python3
"""`grasp_s2r_core` 계약 테스트 — FK/IK 는 가짜로 주입해 코어 로직만 본다."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from grasp_s2r_core import (  # noqa: E402
    DOF_TO_PROFILE,
    PROFILE_TO_DOF,
    GraspS2RCore,
    S2RSensors,
    _DOF_NAMES,
    _quat_from_matrix,
    _rot_euler_zyx,
)
from grasp_s2r_synergy import HAND_JOINT_NAMES  # noqa: E402

RUN = Path(__file__).resolve().parents[1] / "logs/policy/right_g1"
pytestmark = pytest.mark.skipif(not (RUN / "params/env.yaml").exists(),
                                reason="g1 런 dump 없음")

HOME_ARM = np.zeros(7)
HOME_HAND = np.zeros(20)


def _fake_fk(palm_pos=(0.28, -0.38, 0.42), spread=0.05):
    """손끝 5개를 palm 주위에 고정 배치하는 가짜 FK. 관절과 무관하게 일정하다."""
    def palm_pose(_q27):
        return np.array([*palm_pos, np.pi / 2, 0.0, np.pi / 2])

    def tips(_q27):
        p = np.asarray(palm_pos, dtype=float)
        return np.array([
            p + np.array([spread, 0.0, 0.0]),          # thumb — 대향
            p + np.array([-spread, 0.01, 0.0]),
            p + np.array([-spread, 0.0, 0.0]),
            p + np.array([-spread, -0.01, 0.0]),
            p + np.array([-spread, -0.02, 0.0]),
        ])
    return palm_pose, tips


def _core(policy=None, **kw):
    palm_pose, tips = _fake_fk()
    calls = {"fab": 0}

    def fab_step(palm6, n=0):
        calls["fab"] += 1
        return np.full(7, float(np.sum(palm6[:3])))

    c = GraspS2RCore(
        policy=policy or (lambda obs: np.zeros(21)),
        fabric_palm_pose=palm_pose, fabric_tips=tips, fabric_step=fab_step,
        run_dir=RUN, goal3=(0.43, 0.22, 0.42),
        soft_limits=np.tile(np.array([-1.571, 1.571]), (20, 1)), **kw)
    c._calls = calls
    return c


def _sensors(**kw):
    d = dict(arm_q=HOME_ARM, arm_qd=np.zeros(7), hand_q=HOME_HAND,
             hand_qd=np.zeros(20), object_pos=np.array([0.35, -0.30, 0.30]),
             tip_force_world=np.zeros((5, 3)),
             tip_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (5, 1)))
    d.update(kw)
    return S2RSensors(**d)


# ---------------------------------------------------------------------------
def test_fabric_order_permutation_is_applied_to_fk_input():
    """★★Fabrics 는 자기 관절 순서를 쓴다 — 안 바꾸면 손끝이 148 mm 어긋난다(09.03)."""
    seen = {}
    perm = np.roll(np.arange(20), 3)

    def palm_pose(q27):
        seen["q"] = np.asarray(q27, dtype=float).copy()
        return np.array([0.28, -0.38, 0.42, np.pi / 2, 0.0, np.pi / 2])

    _, tips = _fake_fk()
    c = GraspS2RCore(
        policy=lambda o: np.zeros(21), fabric_palm_pose=palm_pose,
        fabric_tips=tips, fabric_step=lambda p6, n=0: np.zeros(7),
        run_dir=RUN, goal3=np.zeros(3),
        soft_limits=np.tile(np.array([-1.571, 1.571]), (20, 1)),
        hand_dof_to_fabric=perm)
    hand = np.arange(20, dtype=float)
    c.reset(arm_q=HOME_ARM, hand_q=hand, object_pos=(0.35, -0.30, 0.30))
    assert np.array_equal(seen["q"][7:], hand[perm])


def test_hand_order_permutations_are_inverse():
    """★DOF 순 ↔ 프로필 순은 서로 역이어야 한다 — 섞이면 40칸이 통째로 어긋난다."""
    a = np.arange(20)
    assert np.array_equal(a[DOF_TO_PROFILE][PROFILE_TO_DOF], a)
    assert [_DOF_NAMES[i] for i in DOF_TO_PROFILE] == list(HAND_JOINT_NAMES)


def test_obs_is_155d():
    c = _core()
    c.reset(arm_q=HOME_ARM, hand_q=HOME_HAND, object_pos=(0.35, -0.30, 0.30))
    out = c.step(_sensors())
    assert out.obs.shape == (155,)


def test_first_tick_joint_err_is_zero():
    """리셋이 목표를 실측으로 두므로 첫 tick 의 `joint_err`(111~130) 은 0 이다."""
    c = _core()
    c.reset(arm_q=HOME_ARM, hand_q=HOME_HAND, object_pos=(0.35, -0.30, 0.30))
    out = c.step(_sensors())
    assert np.allclose(out.obs[111:131], 0.0, atol=1e-12)


def test_last_action_feeds_next_obs():
    """★obs 의 actions 21칸은 **직전 액션**이다. 첫 tick 은 0."""
    seq = [np.full(21, 0.5), np.full(21, -0.25)]
    it = iter(seq)
    c = _core(policy=lambda obs: next(it))
    c.reset(arm_q=HOME_ARM, hand_q=HOME_HAND, object_pos=(0.35, -0.30, 0.30))
    first = c.step(_sensors())
    assert np.allclose(first.obs[131:152], 0.0)
    second = c.step(_sensors())
    assert np.allclose(second.obs[131:152], 0.5)


def test_goal_rel_is_goal_minus_object():
    c = _core()
    obj = np.array([0.35, -0.30, 0.30])
    c.reset(arm_q=HOME_ARM, hand_q=HOME_HAND, object_pos=obj)
    out = c.step(_sensors(object_pos=obj))
    assert np.allclose(out.obs[152:155], np.array([0.43, 0.22, 0.42]) - obj)


def test_spawn_anchor_is_snapshotted_at_reset():
    """★물체가 움직여도 앵커는 리셋 시점 값이다(되먹임 차단)."""
    c = _core()
    c.reset(arm_q=HOME_ARM, hand_q=HOME_HAND, object_pos=(0.35, -0.30, 0.30))
    a0 = c.palm.anchor().copy()
    c.step(_sensors(object_pos=np.array([0.50, 0.10, 0.30])))
    assert np.allclose(c.palm.anchor(), a0)


def test_cage_is_calibrated_at_reset_and_rigid_to_palm():
    """케이지는 홈에서 한 번 재고 palm 에 강체로 붙는다."""
    c = _core()
    c.reset(arm_q=HOME_ARM, hand_q=HOME_HAND, object_pos=(0.35, -0.30, 0.30))
    assert c._r_cage == pytest.approx(0.05, abs=1e-3)
    assert c._cage_off is not None


def test_close_gate_opens_only_near_cage_center():
    c = _core()
    c.reset(arm_q=HOME_ARM, hand_q=HOME_HAND, object_pos=(0.28, -0.38, 0.42))
    R = _rot_euler_zyx([np.pi / 2, 0.0, np.pi / 2])
    palm = np.array([0.28, -0.38, 0.42])
    cage = palm + R @ c._cage_off
    assert c.close_gate(palm, R, cage) == pytest.approx(1.0)          # 정렬 → 열림
    far = cage + np.array([0.5, 0.0, 0.0])
    assert c.close_gate(palm, R, far) == pytest.approx(0.0)           # 멀면 닫힘


def test_hand_target_is_returned_in_dof_order():
    """★발행 순서와 맞춰 DOF 순으로 돌려줘야 한다."""
    c = _core(policy=lambda obs: np.concatenate([np.zeros(6), np.ones(15)]))
    c.reset(arm_q=HOME_ARM, hand_q=HOME_HAND, object_pos=(0.28, -0.38, 0.42))
    out = c.step(_sensors(object_pos=np.array([0.28, -0.38, 0.42])))
    assert out.hand_q_target.shape == (20,)
    # 프로필 순으로 되돌리면 시너지 내부 목표와 같아야 한다.
    assert np.allclose(out.hand_q_target[DOF_TO_PROFILE], c.hand.target)


def test_arm_target_comes_from_injected_fabric():
    c = _core()
    c.reset(arm_q=HOME_ARM, hand_q=HOME_HAND, object_pos=(0.35, -0.30, 0.30))
    out = c.step(_sensors())
    assert out.arm_q_target.shape == (7,)
    assert c._calls["fab"] == 1          # tick 당 정확히 한 번 적분


def test_quat_from_matrix_roundtrip():
    for e in ([0.0, 0.0, 0.0], [np.pi / 2, 0.0, np.pi / 2], [0.3, -0.7, 1.1]):
        R = _rot_euler_zyx(e)
        q = _quat_from_matrix(R)
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-9)
        from grasp_s2r_obs_builder import quat_to_matrix
        assert np.allclose(quat_to_matrix(q), R, atol=1e-9)


def test_tip_force_identity_when_tip_quat_is_unit():
    """실기 `tip_forces_xyz` 는 이미 팁 로컬 — 단위 쿼터니언이면 그대로 통과한다."""
    c = _core()
    c.reset(arm_q=HOME_ARM, hand_q=HOME_HAND, object_pos=(0.35, -0.30, 0.30))
    f = np.zeros((5, 3))
    f[0] = [1.0, 2.0, 3.0]
    out = c.step(_sensors(tip_force_world=f))
    assert np.allclose(out.obs[96:99], np.array([1.0, 2.0, 3.0]) / 10.0)
