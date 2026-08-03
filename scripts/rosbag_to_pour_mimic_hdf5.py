#!/usr/bin/env python3
"""Convert follower ROS2 bag data into a Pour-Mimic source HDF5.

This script intentionally ignores leader/reference topics. It samples follower
state topics at a fixed rate, converts them to the 18D Pour-Mimic action
contract, replays those actions in the IsaacLab task, and writes the resulting
source demo for ``annotate_demos.py``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import traceback
import importlib
from pathlib import Path
from typing import Callable, Iterable, NamedTuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ros2_demo_recorder import (  # noqa: E402
    ACTION_DIM,
    CURL_JOINT_IDX,
    CURL_MAX,
    CURL_MIN,
    FabricsRightPalmFK,
    HDF5DemoWriter,
    LEFT_ARM_DELTA_JOINT_SCALE,
    RIGHT_PALM_DELTA_ROT_SCALE,
    RIGHT_PALM_DELTA_XYZ_SCALE,
    EnvRightPalmPoseProvider,
    build_pour_mimic_action,
    normalize_curl_joints,
    pose7_xyzw_delta,
)


RIGHT_ARM_TOPIC = "/openarm/right/joint_states"
LEFT_ARM_TOPIC = "/openarm/left/joint_states"
RIGHT_HAND_TOPIC = "/dg5f_right/joint_states"
RIGHT_SENSOR_TOPIC = "/tesollo/right/sensor"
FOLLOWER_TOPICS = {RIGHT_ARM_TOPIC, LEFT_ARM_TOPIC, RIGHT_HAND_TOPIC, RIGHT_SENSOR_TOPIC}
REQUIRED_TOPICS = {RIGHT_ARM_TOPIC, LEFT_ARM_TOPIC, RIGHT_HAND_TOPIC}

RIGHT_ARM_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]
LEFT_ARM_JOINTS = [f"openarm_left_joint{i}" for i in range(1, 8)]
RIGHT_HAND_JOINTS = [f"rj_dg_{finger}_{joint}" for finger in range(1, 6) for joint in range(1, 5)]

RIGHT_ARM_ALIASES = {
    isaac_name: (isaac_name, f"right_follower_arm_joint_{idx}")
    for idx, isaac_name in enumerate(RIGHT_ARM_JOINTS)
}
LEFT_ARM_ALIASES = {
    isaac_name: (isaac_name, f"left_follower_arm_joint_{idx}")
    for idx, isaac_name in enumerate(LEFT_ARM_JOINTS)
}

PoseFk = Callable[[np.ndarray], np.ndarray]


class TopicSeries(NamedTuple):
    timestamps_ns: np.ndarray
    values: np.ndarray


class FollowerSamples(NamedTuple):
    timestamps_ns: np.ndarray
    right_arm: np.ndarray
    left_arm: np.ndarray
    right_hand: np.ndarray
    right_sensor: np.ndarray | None = None


def select_follower_topics(topic_names: Iterable[str]) -> set[str]:
    """Return only follower topics used by the offline conversion."""

    return {topic for topic in topic_names if topic in FOLLOWER_TOPICS}


def order_joint_positions(
    names: Iterable[str],
    positions: Iterable[float],
    target_names: Iterable[str],
    *,
    aliases: dict[str, Iterable[str]] | None = None,
) -> np.ndarray:
    """Order a JointState position vector by IsaacLab asset joint names."""

    name_to_pos = {name: float(pos) for name, pos in zip(names, positions)}
    ordered: list[float] = []
    missing: list[str] = []
    for target_name in target_names:
        candidates = tuple(aliases.get(target_name, (target_name,)) if aliases else (target_name,))
        for candidate in candidates:
            if candidate in name_to_pos:
                ordered.append(name_to_pos[candidate])
                break
        else:
            missing.append(target_name)
    if missing:
        raise KeyError(f"JointState missing required joints: {missing}")
    return np.asarray(ordered, dtype=np.float64)


def sample_previous(series: TopicSeries, target_timestamps_ns: np.ndarray) -> np.ndarray:
    """Sample the nearest prior value for each target timestamp."""

    indices = np.searchsorted(series.timestamps_ns, target_timestamps_ns, side="right") - 1
    if np.any(indices < 0):
        first_bad = int(target_timestamps_ns[np.where(indices < 0)[0][0]])
        raise ValueError(f"no previous sample available at timestamp {first_bad}")
    return series.values[indices]


def make_sample_grid(series_by_topic: dict[str, TopicSeries], hz: float) -> np.ndarray:
    """Build a fixed-rate timestamp grid shared by the required topics."""

    if hz <= 0.0:
        raise ValueError("--hz must be positive")
    missing = sorted(REQUIRED_TOPICS - set(series_by_topic))
    if missing:
        raise KeyError(f"bag is missing required follower topics: {missing}")

    start_ns = max(int(series_by_topic[topic].timestamps_ns[0]) for topic in REQUIRED_TOPICS)
    end_ns = min(int(series_by_topic[topic].timestamps_ns[-1]) for topic in REQUIRED_TOPICS)
    if end_ns <= start_ns:
        raise ValueError("required follower topics do not overlap in time")
    period_ns = int(round(1_000_000_000.0 / hz))
    return np.arange(start_ns, end_ns + 1, period_ns, dtype=np.int64)


def resample_follower_series(series_by_topic: dict[str, TopicSeries], hz: float) -> FollowerSamples:
    timestamps = make_sample_grid(series_by_topic, hz)
    sensor = None
    if RIGHT_SENSOR_TOPIC in series_by_topic:
        valid_sensor_timestamps = timestamps[timestamps >= series_by_topic[RIGHT_SENSOR_TOPIC].timestamps_ns[0]]
        if len(valid_sensor_timestamps) == len(timestamps):
            sensor = sample_previous(series_by_topic[RIGHT_SENSOR_TOPIC], timestamps)
    return FollowerSamples(
        timestamps_ns=timestamps,
        right_arm=sample_previous(series_by_topic[RIGHT_ARM_TOPIC], timestamps),
        left_arm=sample_previous(series_by_topic[LEFT_ARM_TOPIC], timestamps),
        right_hand=sample_previous(series_by_topic[RIGHT_HAND_TOPIC], timestamps),
        right_sensor=sensor,
    )


def build_actions_from_samples(samples: FollowerSamples, *, right_palm_fk: PoseFk) -> np.ndarray:
    """Convert follower joint samples into 18D Pour-Mimic actions."""

    actions = np.zeros((len(samples.timestamps_ns), ACTION_DIM), dtype=np.float32)
    current_right_arm = samples.right_arm[0].copy()
    current_left_arm = samples.left_arm[0].copy()
    current_palm_pose = right_palm_fk(current_right_arm)

    for idx in range(len(samples.timestamps_ns)):
        target_right_arm = samples.right_arm[idx]
        target_left_arm = samples.left_arm[idx]
        target_palm_pose = right_palm_fk(target_right_arm)
        palm_delta = pose7_xyzw_delta(current_palm_pose, target_palm_pose)

        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[:3] = (palm_delta[:3] / RIGHT_PALM_DELTA_XYZ_SCALE).astype(np.float32)
        action[3:6] = (palm_delta[3:6] / RIGHT_PALM_DELTA_ROT_SCALE).astype(np.float32)
        action[6:11] = normalize_curl_joints(samples.right_hand[idx]).astype(np.float32)
        action[11:18] = ((target_left_arm - current_left_arm) / LEFT_ARM_DELTA_JOINT_SCALE).astype(np.float32)
        actions[idx] = action

        current_right_arm = target_right_arm.copy()
        current_left_arm = target_left_arm.copy()
        current_palm_pose = target_palm_pose
    return actions


def read_rosbag_follower_series(bag_dir: str | Path) -> dict[str, TopicSeries]:
    """Read follower topics from a ROS2 sqlite3 bag via rosbag2_py."""

    try:
        from rclpy.serialization import deserialize_message
        from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
        from rosidl_runtime_py.utilities import get_message
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ROS2 Python modules are unavailable for this interpreter. "
            "When running under Isaac Sim Python, use the automatic /usr/bin/python3 extraction path."
        ) from exc

    bag_path = Path(bag_dir).expanduser().resolve()
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(bag_path), storage_id="sqlite3"), ConverterOptions("", ""))
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    selected = select_follower_topics(topic_types)
    missing = sorted(REQUIRED_TOPICS - selected)
    if missing:
        raise KeyError(f"bag is missing required follower topics: {missing}")

    raw: dict[str, list[tuple[int, np.ndarray]]] = {topic: [] for topic in selected}
    msg_types = {topic: get_message(topic_types[topic]) for topic in selected}

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        if topic not in selected:
            continue
        msg = deserialize_message(data, msg_types[topic])
        if topic == RIGHT_ARM_TOPIC:
            value = order_joint_positions(msg.name, msg.position, RIGHT_ARM_JOINTS, aliases=RIGHT_ARM_ALIASES)
        elif topic == LEFT_ARM_TOPIC:
            value = order_joint_positions(msg.name, msg.position, LEFT_ARM_JOINTS, aliases=LEFT_ARM_ALIASES)
        elif topic == RIGHT_HAND_TOPIC:
            value = order_joint_positions(msg.name, msg.position, RIGHT_HAND_JOINTS)
        elif topic == RIGHT_SENSOR_TOPIC:
            value = np.asarray(msg.data, dtype=np.float64)
        else:
            continue
        raw[topic].append((int(timestamp_ns), value))

    series_by_topic: dict[str, TopicSeries] = {}
    for topic, rows in raw.items():
        if not rows:
            continue
        timestamps = np.asarray([row[0] for row in rows], dtype=np.int64)
        values = np.stack([row[1] for row in rows], axis=0)
        order = np.argsort(timestamps)
        series_by_topic[topic] = TopicSeries(timestamps_ns=timestamps[order], values=values[order])
    return series_by_topic


def write_auxiliary_observations(
    output_file: str | Path,
    demo_name: str,
    observations: dict[str, np.ndarray],
) -> None:
    """Append auxiliary observations after the native IsaacLab writer flushes."""

    if not observations:
        return
    import h5py

    with h5py.File(output_file, "a") as handle:
        demo = handle[f"data/{demo_name}"]
        aux = demo.require_group("obs").require_group("aux")
        for name, values in observations.items():
            if name in aux:
                del aux[name]
            aux.create_dataset(name, data=np.asarray(values, dtype=np.float32), compression="gzip")


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _obs_to_numpy(obs) -> np.ndarray:
    return _to_numpy(obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs).reshape(-1).astype(np.float32)


def _get_subtask_signals(env) -> dict[str, bool]:
    if not hasattr(env, "get_subtask_term_signals"):
        return {}
    try:
        raw = env.get_subtask_term_signals()
    except Exception:
        return {}
    return {key: bool(value[0].item()) if hasattr(value, "__len__") else bool(value) for key, value in raw.items()}


def _register_sim_args(writer: HDF5DemoWriter, env) -> None:
    cfg = getattr(env, "cfg", None)
    sim_args: dict = {}
    if cfg is not None:
        sim_cfg = getattr(cfg, "sim", None)
        if sim_cfg is not None:
            sim_args["dt"] = float(getattr(sim_cfg, "dt", 0.0))
        sim_args["decimation"] = int(getattr(cfg, "decimation", 1))
        sim_args["render_interval"] = int(getattr(cfg, "decimation", 1))
        sim_args["num_envs"] = int(getattr(getattr(cfg, "scene", None), "num_envs", 1))
    writer.set_env_args(sim_args)


def replay_samples_to_hdf5(samples: FollowerSamples, args) -> str:
    """Replay sampled follower commands in IsaacLab and export demo_0."""

    print("[rosbag_to_pour_mimic_hdf5] launching IsaacLab AppLauncher", flush=True)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import gymnasium as gym

    openarm_src = REPO_ROOT / "hdgp" / "source" / "openarm"
    if str(openarm_src) not in sys.path:
        sys.path.insert(0, str(openarm_src))
    import openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.both.pour_v1_mimic  # noqa: F401

    env = None
    writer = None
    try:
        print("[rosbag_to_pour_mimic_hdf5] importing Pour-Mimic env cfg module", flush=True)
        cfg_module = importlib.import_module(
            "openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.both"
            ".pour_v1_mimic.pour_mimic_managed_env_cfg"
        )
        print("[rosbag_to_pour_mimic_hdf5] resolving PourMimicManagedMimicEnvCfg", flush=True)
        PourMimicManagedMimicEnvCfg = getattr(cfg_module, "PourMimicManagedMimicEnvCfg")

        print("[rosbag_to_pour_mimic_hdf5] instantiating Pour-Mimic env cfg", flush=True)
        env_cfg = PourMimicManagedMimicEnvCfg()
        print("[rosbag_to_pour_mimic_hdf5] env cfg instantiated", flush=True)
        env_cfg.sim.device = args.device
        env_cfg.scene.num_envs = 1
        env_cfg.env_name = args.task
        print(
            "[rosbag_to_pour_mimic_hdf5] creating env "
            f"{args.task} device={args.device} num_envs={env_cfg.scene.num_envs}",
            flush=True,
        )
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        print(f"[rosbag_to_pour_mimic_hdf5] created env {args.task}", flush=True)
        print("[rosbag_to_pour_mimic_hdf5] resetting env", flush=True)
        env.reset()
        print("[rosbag_to_pour_mimic_hdf5] env reset complete", flush=True)

        from ros2_demo_recorder import EpisodeBuffer

        writer = HDF5DemoWriter(args.output_file, env_name=args.task)
        _register_sim_args(writer, env)
        episode = EpisodeBuffer()
        scene = getattr(env, "scene", None)
        if scene is not None and hasattr(scene, "get_state"):
            episode.set_initial_state(scene.get_state(is_relative=True))

        right_palm_fk = FabricsRightPalmFK(device=args.device, repo_root=REPO_ROOT)
        current_palm_pose_provider = EnvRightPalmPoseProvider(env)
        current_right_arm = _read_env_arm_joints(env, RIGHT_ARM_JOINTS, fallback=samples.right_arm[0])
        current_left_arm = _read_env_arm_joints(env, LEFT_ARM_JOINTS, fallback=samples.left_arm[0])

        print(f"[rosbag_to_pour_mimic_hdf5] replaying {len(samples.timestamps_ns)} actions", flush=True)
        for idx in range(len(samples.timestamps_ns)):
            action = build_pour_mimic_action(
                current_right_arm=current_right_arm,
                target_right_arm=samples.right_arm[idx],
                target_right_hand=samples.right_hand[idx],
                current_left_arm=current_left_arm,
                target_left_arm=samples.left_arm[idx],
                right_palm_fk=right_palm_fk,
                current_right_palm_pose=current_palm_pose_provider(),
            )
            action_batch = _torch_action(action, env)
            obs, reward, terminated, truncated, _info = env.step(action_batch)
            done = bool(_to_numpy(terminated).any() or _to_numpy(truncated).any())
            episode.append(_obs_to_numpy(obs), action, float(_to_numpy(reward).reshape(-1)[0]), done, _get_subtask_signals(env))
            current_right_arm = samples.right_arm[idx].copy()
            current_left_arm = samples.left_arm[idx].copy()
            if (idx + 1) % 100 == 0:
                print(f"[rosbag_to_pour_mimic_hdf5] replayed {idx + 1}/{len(samples.timestamps_ns)}", flush=True)

        demo_name = writer.write_episode(episode, success=True)
        writer.close()
        writer = None

        aux = {"rosbag_timestamps_ns": samples.timestamps_ns.astype(np.float64)}
        if samples.right_sensor is not None:
            aux["tesollo_right_sensor"] = samples.right_sensor
        write_auxiliary_observations(args.output_file, demo_name, aux)
        return demo_name
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        simulation_app.close()


def _read_env_arm_joints(env, joint_names: list[str], *, fallback: np.ndarray) -> np.ndarray:
    try:
        robot = env.scene["robot"]
        names = robot.joint_names
        joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
        return np.asarray([joint_pos[names.index(name)] for name in joint_names], dtype=np.float64)
    except Exception:
        return np.asarray(fallback, dtype=np.float64).copy()


def _torch_action(action: np.ndarray, env):
    try:
        import torch

        return torch.as_tensor(action[None, :], dtype=torch.float32, device=getattr(env, "device", "cpu"))
    except Exception:
        return action[None, :]


def save_samples_npz(samples: FollowerSamples, path: str | Path) -> None:
    payload = {
        "timestamps_ns": samples.timestamps_ns,
        "right_arm": samples.right_arm,
        "left_arm": samples.left_arm,
        "right_hand": samples.right_hand,
    }
    if samples.right_sensor is not None:
        payload["right_sensor"] = samples.right_sensor
    np.savez_compressed(path, **payload)


def load_samples_npz(path: str | Path) -> FollowerSamples:
    with np.load(path) as data:
        sensor = data["right_sensor"] if "right_sensor" in data else None
        return FollowerSamples(
            timestamps_ns=data["timestamps_ns"],
            right_arm=data["right_arm"],
            left_arm=data["left_arm"],
            right_hand=data["right_hand"],
            right_sensor=sensor,
        )


def extract_samples_with_system_python(args: argparse.Namespace) -> FollowerSamples:
    tmp = tempfile.NamedTemporaryFile(prefix="pour_mimic_bag_samples_", suffix=".npz", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    cmd = [
        "/usr/bin/python3",
        str(Path(__file__).resolve()),
        "--bag_dir",
        str(args.bag_dir),
        "--hz",
        str(args.hz),
        "--extract_samples_npz",
        str(tmp_path),
    ]
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = (
        "/opt/ros/humble/local/lib/python3.10/dist-packages:"
        "/opt/ros/humble/lib/python3.10/site-packages"
    )
    try:
        subprocess.run(cmd, check=True, env=env)
        return load_samples_npz(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert follower ROS2 bag to Pour-Mimic source HDF5")
    parser.add_argument("--bag_dir", required=True)
    parser.add_argument("--output_file")
    parser.add_argument("--task", default="Pour-Mimic-V1-Mimic-v0")
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--samples_npz", help="Use pre-extracted follower samples instead of reading --bag_dir.")
    parser.add_argument("--extract_samples_npz", help="Read --bag_dir and write resampled follower arrays to this NPZ.")
    parser.add_argument(
        "--dry_actions_only",
        action="store_true",
        help="Read the bag and validate 18D action conversion without launching IsaacLab.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        _main_impl(args)
    except BaseException:
        print("[rosbag_to_pour_mimic_hdf5] fatal error:", flush=True)
        traceback.print_exc()
        raise


def _main_impl(args: argparse.Namespace) -> None:
    if not args.extract_samples_npz and not args.dry_actions_only and not args.output_file:
        raise ValueError("--output_file is required unless --extract_samples_npz or --dry_actions_only is used")
    if args.samples_npz:
        samples = load_samples_npz(args.samples_npz)
    else:
        try:
            series_by_topic = read_rosbag_follower_series(args.bag_dir)
            samples = resample_follower_series(series_by_topic, args.hz)
        except RuntimeError as exc:
            if args.extract_samples_npz or "ROS2 Python modules are unavailable" not in str(exc):
                raise
            print(f"[rosbag_to_pour_mimic_hdf5] {exc}")
            print("[rosbag_to_pour_mimic_hdf5] extracting bag samples with /usr/bin/python3")
            samples = extract_samples_with_system_python(args)

    print(
        f"[rosbag_to_pour_mimic_hdf5] sampled {len(samples.timestamps_ns)} steps "
        f"from {Path(args.bag_dir).resolve() if args.bag_dir else args.samples_npz}"
    )
    if args.extract_samples_npz:
        save_samples_npz(samples, args.extract_samples_npz)
        print(f"[rosbag_to_pour_mimic_hdf5] wrote sample cache {args.extract_samples_npz}")
        return
    if args.dry_actions_only:
        actions = build_actions_from_samples(samples, right_palm_fk=FabricsRightPalmFK(device=args.device, repo_root=REPO_ROOT))
        print(f"[rosbag_to_pour_mimic_hdf5] dry actions shape={actions.shape}")
        return
    demo_name = replay_samples_to_hdf5(samples, args)
    print(f"[rosbag_to_pour_mimic_hdf5] wrote {demo_name} to {args.output_file}")


if __name__ == "__main__":
    main()
