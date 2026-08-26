#!/usr/bin/env python3
"""action_bag_core 테스트 — 손실이 조용히 지나가지 않는지가 요점."""
from __future__ import annotations

import numpy as np
import pytest

from action_bag_core import (
    BagPlan,
    build_group,
    build_plan,
    load_npz,
    max_safe_rate_scale,
    velocity_verdict,
)

ARM = ["l_aj_1", "l_aj_2"]
GRIP = ["l_hj_gripper_1"]
PROFILE = {
    "l_aj_1": {"source": "openarm_left_joint1", "sign": 1.0,
               "lower": -1.0, "upper": 1.0, "velocity": 2.0},
    "l_aj_2": {"source": "openarm_left_joint2", "sign": -1.0,
               "lower": -1.0, "upper": 1.0, "velocity": 2.0},
    "l_hj_gripper_1": {"source": "openarm_left_finger_joint1", "sign": 1.0,
                       "lower": 0.0, "upper": 0.044, "velocity": 0.2},
}


def _npz(arm: np.ndarray, grip: np.ndarray, dt: float = 0.02) -> dict:
    n = arm.shape[0]
    return {
        "arm_target": arm,
        "grip_cmd": grip,
        "action": np.zeros((n, 7), dtype=np.float32),
        "palm_cmd_pos": np.zeros((n, 3), dtype=np.float32),
        "palm_cmd_quat_wxyz": np.tile([1.0, 0, 0, 0], (n, 1)).astype(np.float32),
        "meta_joint_names": np.array(ARM),
        "meta_grip_names": np.array(["l_hj_gripper_1", "l_hj_gripper_2"]),
        "meta_step_dt": np.array([dt]),
    }


def _grip(n: int, value: float = 0.02, sibling: float | None = None) -> np.ndarray:
    other = value if sibling is None else sibling
    return np.stack([np.full(n, value), np.full(n, other)], axis=1).astype(np.float32)


def test_canonical_names_become_the_drivers_source_names():
    g = build_group(values=np.zeros((3, 2)), sim_canonical=ARM,
                    group_canonical=ARM, profile_joints=PROFILE)
    assert g.source_names == ("openarm_left_joint1", "openarm_left_joint2")


def test_the_profiles_sign_is_applied():
    vals = np.tile([0.5, 0.5], (3, 1))
    g = build_group(values=vals, sim_canonical=ARM, group_canonical=ARM,
                    profile_joints=PROFILE)
    assert g.positions[0, 0] == pytest.approx(0.5)
    assert g.positions[0, 1] == pytest.approx(-0.5)  # sign -1


def test_clamping_is_counted_rather_than_hidden():
    vals = np.tile([5.0, 0.0], (4, 1))  # l_aj_1 upper=1.0 → 4 스텝 전부 clamp
    g = build_group(values=vals, sim_canonical=ARM, group_canonical=ARM,
                    profile_joints=PROFILE)
    assert g.positions[:, 0].max() == pytest.approx(1.0)
    assert g.clamped == (4, 0)
    assert g.clamp_total == 4


def test_a_mimic_sibling_that_agrees_is_dropped():
    plan = build_plan(_npz(np.zeros((3, 2)), _grip(3)), profile_joints=PROFILE,
                      arm_group=ARM, grip_group=GRIP)
    assert plan.grip.source_names == ("openarm_left_finger_joint1",)
    assert plan.grip.dropped == ("l_hj_gripper_2",)


def test_a_mimic_sibling_that_disagrees_is_refused():
    npz = _npz(np.zeros((3, 2)), _grip(3, value=0.02, sibling=0.03))
    with pytest.raises(ValueError, match="mimic"):
        build_plan(npz, profile_joints=PROFILE, arm_group=ARM, grip_group=GRIP)


def test_a_missing_channel_is_named_not_guessed():
    npz = _npz(np.zeros((3, 2)), _grip(3))
    del npz["palm_cmd_pos"]
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "x.npz"
        np.savez(p, **npz)
        with pytest.raises(KeyError, match="palm_cmd_pos"):
            load_npz(p)


def test_non_finite_frames_are_refused_not_interpolated():
    arm = np.zeros((3, 2))
    arm[1, 0] = np.nan
    with pytest.raises(ValueError, match="유한"):
        build_plan(_npz(arm, _grip(3)), profile_joints=PROFILE,
                   arm_group=ARM, grip_group=GRIP)


def test_a_joint_the_controller_needs_but_the_recording_lacks_is_refused():
    with pytest.raises(KeyError, match="l_aj_3"):
        build_group(values=np.zeros((3, 2)), sim_canonical=ARM,
                    group_canonical=["l_aj_1", "l_aj_3"], profile_joints=PROFILE)


def test_rate_scale_stretches_time_without_moving_the_path():
    arm = np.linspace(0, 0.5, 10).repeat(2).reshape(10, 2)
    full = build_plan(_npz(arm, _grip(10)), profile_joints=PROFILE,
                      arm_group=ARM, grip_group=GRIP, rate_scale=1.0)
    half = build_plan(_npz(arm, _grip(10)), profile_joints=PROFILE,
                      arm_group=ARM, grip_group=GRIP, rate_scale=0.5)
    np.testing.assert_allclose(full.arm.positions, half.arm.positions)
    assert half.publish_dt == pytest.approx(2 * full.publish_dt)
    assert half.duration_sec == pytest.approx(2 * full.duration_sec)
    np.testing.assert_allclose(half.peak_speed(), full.peak_speed() / 2)


def test_rate_scale_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match="rate_scale"):
        build_plan(_npz(np.zeros((3, 2)), _grip(3)), profile_joints=PROFILE,
                   arm_group=ARM, grip_group=GRIP, rate_scale=1.5)


def test_timestamps_are_monotonic_and_start_at_zero():
    plan = build_plan(_npz(np.zeros((5, 2)), _grip(5)), profile_joints=PROFILE,
                      arm_group=ARM, grip_group=GRIP)
    assert plan.t_ns[0] == 0
    assert np.all(np.diff(plan.t_ns) > 0)


def test_a_demand_over_the_profile_limit_is_reported_as_over():
    arm = np.zeros((3, 2))
    arm[1, 0] = 0.5  # 0.5 rad / 0.02 s = 25 rad/s ≫ 2.0
    plan = build_plan(_npz(arm, _grip(3)), profile_joints=PROFILE,
                      arm_group=ARM, grip_group=GRIP)
    rows = velocity_verdict(plan, PROFILE)
    assert rows[0]["over"] is True
    assert rows[0]["peak"] == pytest.approx(25.0)


def test_the_suggested_rate_scale_actually_brings_the_demand_under_the_limit():
    arm = np.zeros((6, 2))
    arm[1::2, 0] = 0.1
    plan = build_plan(_npz(arm, _grip(6)), profile_joints=PROFILE,
                      arm_group=ARM, grip_group=GRIP)
    scale = max_safe_rate_scale(plan, PROFILE)
    slowed = build_plan(_npz(arm, _grip(6)), profile_joints=PROFILE,
                        arm_group=ARM, grip_group=GRIP, rate_scale=scale)
    assert not any(r["over"] for r in velocity_verdict(slowed, PROFILE))


def test_an_unknown_velocity_limit_says_unknown_rather_than_passing():
    profile = {k: dict(v) for k, v in PROFILE.items()}
    profile["l_aj_1"].pop("velocity")
    plan = build_plan(_npz(np.zeros((3, 2)), _grip(3)), profile_joints=profile,
                      arm_group=ARM, grip_group=GRIP)
    assert velocity_verdict(plan, profile)[0]["limit"] is None
    assert max_safe_rate_scale(plan, profile) is None


def test_a_multi_env_recording_is_refused_with_a_way_out():
    npz = _npz(np.zeros((3, 2)), _grip(3))
    npz["arm_target"] = np.zeros((3, 4, 2))
    with pytest.raises(ValueError, match="num_envs 1"):
        build_plan(npz, profile_joints=PROFILE, arm_group=ARM, grip_group=GRIP)
