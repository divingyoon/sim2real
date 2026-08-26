#!/usr/bin/env python3
"""npz → 백 → npz 왕복. 백이 드라이버 계약을 무손실로 나른다는 증명."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from action_bag import write_bag
from action_bag_core import build_plan, load_npz
from robot_profile import load_robot_profile

pytest.importorskip("rosbag2_py")
from action_bag_read import read_bag  # noqa: E402

RECORDING = Path(__file__).resolve().parents[1] / "logs/shadow/sim_fab_test16_gcON.npz"
pytestmark = pytest.mark.skipif(not RECORDING.exists(), reason="기록 npz 없음")


@pytest.fixture(scope="module")
def profile():
    return load_robot_profile("gripper_left")


@pytest.fixture(scope="module")
def baked(tmp_path_factory, profile):
    plan = build_plan(load_npz(RECORDING), profile_joints=profile.joint_limits,
                      arm_group=list(profile.arm_canonical),
                      grip_group=list(profile.ee_canonical), rate_scale=0.5)
    out = tmp_path_factory.mktemp("bag") / "rt.bag"
    write_bag(plan, profile, out)
    return plan, out


def test_the_bag_carries_the_arm_trajectory_unchanged(baked, profile):
    plan, out = baked
    back = read_bag(out, profile=profile)
    np.testing.assert_allclose(back["arm_target"][:, 0, :], plan.arm.positions,
                               atol=1e-6)


def test_source_names_survive_the_round_trip_as_canonical(baked, profile):
    _, out = baked
    back = read_bag(out, profile=profile)
    assert tuple(back["meta_joint_names"]) == tuple(profile.arm_canonical)
    assert tuple(back["meta_grip_names"]) == tuple(profile.ee_canonical)


def test_the_bag_knows_its_own_publish_rate(baked):
    plan, out = baked
    back = read_bag(out, profile=load_robot_profile("gripper_left"))
    assert float(back["meta_step_dt"][0]) == pytest.approx(plan.publish_dt)


def test_the_bag_records_the_pose_the_robot_must_start_from(baked, profile):
    plan, out = baked
    meta = read_bag(out, profile=profile)["meta_bag_meta"]
    first = meta["first_frame"]["arm"]
    assert list(first) == list(plan.arm.source_names)
    np.testing.assert_allclose(list(first.values()), plan.arm.positions[0], atol=1e-6)


def test_writing_over_an_existing_bag_is_refused(baked, profile):
    plan, out = baked
    with pytest.raises(FileExistsError):
        write_bag(plan, profile, out)
