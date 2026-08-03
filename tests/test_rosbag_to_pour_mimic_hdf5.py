from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np


SCRIPT = Path("/home/user/rl_ws/sim2real/scripts/rosbag_to_pour_mimic_hdf5.py")


def _load_converter():
    spec = importlib.util.spec_from_file_location("rosbag_to_pour_mimic_hdf5", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_follower_topics_are_selected_and_leader_topics_are_ignored() -> None:
    converter = _load_converter()

    topics = [
        "/openarm/right/joint_states",
        "/openarm/left/joint_states",
        "/dg5f_right/joint_states",
        "/tesollo/right/sensor",
        "/openarm/left/leader/gripper_state",
        "/dg5f_right/rj_dg_pospid/reference",
    ]

    assert converter.select_follower_topics(topics) == {
        "/openarm/right/joint_states",
        "/openarm/left/joint_states",
        "/dg5f_right/joint_states",
        "/tesollo/right/sensor",
    }


def test_joint_state_positions_are_ordered_by_isaaclab_joint_names() -> None:
    converter = _load_converter()

    hand_names = ["rj_dg_3_2", "rj_dg_1_1", "rj_dg_5_4", "rj_dg_2_1"]
    hand_pos = [32.0, 11.0, 54.0, 21.0]
    ordered = converter.order_joint_positions(
        hand_names,
        hand_pos,
        ["rj_dg_1_1", "rj_dg_2_1", "rj_dg_3_2", "rj_dg_5_4"],
    )
    np.testing.assert_allclose(ordered, [11.0, 21.0, 32.0, 54.0])

    arm_names = [f"right_follower_arm_joint_{i}" for i in range(7)] + ["right_follower_hand_joint_0"]
    ordered_arm = converter.order_joint_positions(
        arm_names,
        list(range(8)),
        converter.RIGHT_ARM_JOINTS,
        aliases=converter.RIGHT_ARM_ALIASES,
    )
    np.testing.assert_allclose(ordered_arm, np.arange(7))


def test_previous_sample_resampling_uses_nearest_prior_values() -> None:
    converter = _load_converter()
    series = converter.TopicSeries(
        timestamps_ns=np.array([10, 20, 40], dtype=np.int64),
        values=np.array([[1.0], [2.0], [4.0]], dtype=np.float64),
    )

    sampled = converter.sample_previous(series, np.array([20, 30, 40], dtype=np.int64))

    np.testing.assert_allclose(sampled[:, 0], [2.0, 2.0, 4.0])


def test_follower_samples_build_18d_actions_and_sensor_aux() -> None:
    converter = _load_converter()
    samples = converter.FollowerSamples(
        timestamps_ns=np.array([1, 2, 3], dtype=np.int64),
        right_arm=np.array([[0.0] * 7, [0.1] * 7, [0.2] * 7], dtype=np.float64),
        left_arm=np.array([[0.0] * 7, [0.01] * 7, [0.02] * 7], dtype=np.float64),
        right_hand=np.zeros((3, 20), dtype=np.float64),
        right_sensor=np.ones((3, 30), dtype=np.float64),
    )
    samples.right_hand[:, converter.CURL_JOINT_IDX] = converter.CURL_MAX

    def fk(joints: np.ndarray) -> np.ndarray:
        return np.array([joints[0], joints[1], joints[2], 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    actions = converter.build_actions_from_samples(samples, right_palm_fk=fk)

    assert actions.shape == (3, 18)
    np.testing.assert_allclose(actions[0, :6], 0.0)
    np.testing.assert_allclose(actions[:, 6:11], 1.0)
    np.testing.assert_allclose(actions[1, 11:18], 0.1)


def test_auxiliary_sensor_can_be_added_to_hdf5(tmp_path) -> None:
    converter = _load_converter()
    path = tmp_path / "demo.hdf5"

    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.create_dataset("actions", data=np.zeros((2, 18), dtype=np.float32))

    converter.write_auxiliary_observations(
        path,
        "demo_0",
        {"tesollo_right_sensor": np.ones((2, 30), dtype=np.float32)},
    )

    with h5py.File(path, "r") as handle:
        assert handle["data/demo_0/obs/aux/tesollo_right_sensor"].shape == (2, 30)
