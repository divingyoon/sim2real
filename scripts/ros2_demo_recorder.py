#!/usr/bin/env python3
"""ROS2 teleop command recorder for Pour-Mimic-V1 demonstrations.

The module keeps the action-conversion logic importable without ROS2 so it can
be unit-tested in the hdgp workspace. The ROS2 node and IsaacLab environment are
created only from the CLI entry point.
"""

from __future__ import annotations

import argparse
import json
import math
import select
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np


RIGHT_ARM_TOPIC = "/isaacsim/right_arm_cmd"
RIGHT_HAND_TOPIC = "/isaacsim/right_hand_cmd"
LEFT_ARM_TOPIC = "/isaacsim/left_arm_cmd"
LEFT_GRIPPER_TOPIC = "/isaacsim/left_gripper_cmd"

CURL_JOINT_IDX = [1, 5, 9, 13, 18]
CURL_MIN = np.array([-math.pi, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
CURL_MAX = np.array([0.0, 2.007, 1.955, 1.902, math.pi / 2.0], dtype=np.float64)

CONTROL_HZ = 60.0
ACTION_DIM = 18
RIGHT_PALM_DELTA_XYZ_SCALE = 0.30
RIGHT_PALM_DELTA_ROT_SCALE = 0.30
LEFT_ARM_DELTA_JOINT_SCALE = 0.10

PoseFk = Callable[[np.ndarray], np.ndarray]
PoseProvider = Callable[[], np.ndarray]


def _as_vector(name: str, values: np.ndarray | list[float] | tuple[float, ...], size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (size,):
        raise ValueError(f"{name} expects {size} values, got {arr.size}")
    return arr


class TeleopCommandState:
    """Latest ROS2 teleop command snapshot."""

    def __init__(
        self,
        right_arm: np.ndarray | list[float] | tuple[float, ...] | None = None,
        right_hand: np.ndarray | list[float] | tuple[float, ...] | None = None,
        left_arm: np.ndarray | list[float] | tuple[float, ...] | None = None,
        left_gripper: float = 0.0,
    ) -> None:
        self.right_arm = _as_vector("right_arm", np.zeros(7) if right_arm is None else right_arm, 7)
        self.right_hand = _as_vector("right_hand", np.zeros(20) if right_hand is None else right_hand, 20)
        self.left_arm = _as_vector("left_arm", np.zeros(7) if left_arm is None else left_arm, 7)
        self.left_gripper = float(left_gripper)
        self.timestamp = time.time()

    def copy(self) -> "TeleopCommandState":
        state = TeleopCommandState(self.right_arm.copy(), self.right_hand.copy(), self.left_arm.copy(), self.left_gripper)
        state.timestamp = self.timestamp
        return state

    def update_right_arm(self, values: np.ndarray | list[float]) -> None:
        self.right_arm = _as_vector("right_arm", values, 7)
        self.timestamp = time.time()

    def update_right_hand(self, values: np.ndarray | list[float]) -> None:
        self.right_hand = _as_vector("right_hand", values, 20)
        self.timestamp = time.time()

    def update_left_arm(self, values: np.ndarray | list[float]) -> None:
        self.left_arm = _as_vector("left_arm", values, 7)
        self.timestamp = time.time()

    def update_left_gripper(self, value: float) -> None:
        self.left_gripper = float(value)
        self.timestamp = time.time()


def normalize_curl_joints(right_hand: np.ndarray | list[float]) -> np.ndarray:
    """Extract and normalize the five Pour-Mimic hand curl joints to [-1, 1]."""

    hand = _as_vector("right_hand", right_hand, 20)
    curl = hand[CURL_JOINT_IDX]
    scaled = 2.0 * (curl - CURL_MIN) / (CURL_MAX - CURL_MIN) - 1.0
    return np.clip(scaled, -1.0, 1.0)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    if q[3] < 0.0:
        q = -q
    angle = 2.0 * math.atan2(np.linalg.norm(q[:3]), max(min(q[3], 1.0), -1.0))
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return q[:3] / np.linalg.norm(q[:3]) * angle


def pose7_xyzw_delta(current_pose: np.ndarray | list[float], target_pose: np.ndarray | list[float]) -> np.ndarray:
    """Return [d_xyz, d_axis_angle] from current pose to target pose."""

    current = _as_vector("current_pose", current_pose, 7)
    target = _as_vector("target_pose", target_pose, 7)
    delta = np.zeros(6, dtype=np.float64)
    delta[:3] = target[:3] - current[:3]
    q_delta = _quat_multiply(_quat_normalize(target[3:7]), _quat_conjugate(_quat_normalize(current[3:7])))
    delta[3:6] = _quat_to_axis_angle(q_delta)
    return delta


def build_pour_mimic_action(
    *,
    current_right_arm: np.ndarray | list[float],
    target_right_arm: np.ndarray | list[float],
    target_right_hand: np.ndarray | list[float],
    current_left_arm: np.ndarray | list[float],
    target_left_arm: np.ndarray | list[float],
    right_palm_fk: PoseFk,
    current_right_palm_pose: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Convert ROS2 joint teleop commands to Pour-Mimic-V1's 18D action."""

    current_right_arm = _as_vector("current_right_arm", current_right_arm, 7)
    target_right_arm = _as_vector("target_right_arm", target_right_arm, 7)
    current_left_arm = _as_vector("current_left_arm", current_left_arm, 7)
    target_left_arm = _as_vector("target_left_arm", target_left_arm, 7)

    current_palm_pose = _as_vector(
        "current_palm_pose",
        right_palm_fk(current_right_arm) if current_right_palm_pose is None else current_right_palm_pose,
        7,
    )
    target_palm_pose = _as_vector("target_palm_pose", right_palm_fk(target_right_arm), 7)

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    palm_delta = pose7_xyzw_delta(current_palm_pose, target_palm_pose)
    action[:3] = (palm_delta[:3] / RIGHT_PALM_DELTA_XYZ_SCALE).astype(np.float32)
    action[3:6] = (palm_delta[3:6] / RIGHT_PALM_DELTA_ROT_SCALE).astype(np.float32)
    action[6:11] = normalize_curl_joints(target_right_hand).astype(np.float32)
    action[11:18] = ((target_left_arm - current_left_arm) / LEFT_ARM_DELTA_JOINT_SCALE).astype(np.float32)
    return action


class EpisodeBuffer:
    """In-memory episode buffer with robomimic-style keys."""

    def __init__(self) -> None:
        self.obs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        # subtask_term_signals per step: list of {signal_name: bool}
        self.subtask_signals: list[dict[str, bool]] = []
        # scene state captured immediately after env.reset (is_relative=True)
        self.initial_state: dict | None = None

    def set_initial_state(self, state: dict) -> None:
        """Store a deep-copy of the scene state dict (dict of dict of np.ndarray)."""
        self.initial_state = _numpy_state(state)

    def append(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        subtask_signals: dict[str, bool] | None = None,
    ) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(_as_vector("action", action, ACTION_DIM).astype(np.float32))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.subtask_signals.append(subtask_signals or {})

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.subtask_signals.clear()
        self.initial_state = None

    def __len__(self) -> int:
        return len(self.actions)


def _numpy_state(state: dict) -> dict:
    """Recursively convert a torch-tensor or ndarray state dict to float32 numpy."""
    out = {}
    for k, v in state.items():
        if isinstance(v, dict):
            out[k] = _numpy_state(v)
        else:
            try:
                arr = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
            except Exception:
                arr = np.asarray(v)
            out[k] = arr.astype(np.float32)
    return out


class HDF5DemoWriter:
    """Append episodes under /data/demo_XXXX in IsaacLab Mimic HDF5 format.

    HDF5 layout written:
      data.attrs["env_args"]   — JSON: {"env_name": task_name, "type": 2, "sim_args": {...}}
      data.attrs["total"]      — cumulative step count
      demo_XXXX/
        attrs: num_samples, success
        actions           (T, 18) float32
        obs/actor_obs     (T, 91) float32  — deploy-compatible policy obs
        obs/datagen_info/subtask_term_signals/{signal}  (T,) float32
        initial_state/    — nested scene state (annotate_demos.py reset_to target)
        rewards           (T,)
        dones             (T,)
    """

    def __init__(self, output_file: str | Path, *, env_name: str = "Pour-Mimic-V1-Mimic-v0") -> None:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("h5py is required to write demonstration datasets") from exc

        self._h5py = h5py
        self.output_file = Path(output_file).expanduser().resolve()
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._env_name = env_name
        self._next_index = 0
        # env_args written lazily on first episode flush
        self._env_args: dict | None = None
        self._native_handler = None
        self._native_episode_cls = None
        self._torch = None
        try:
            from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
            import torch

            handler = HDF5DatasetFileHandler()
            if self.output_file.exists():
                handler.open(str(self.output_file), mode="a")
                handler.set_env_name(env_name)
            else:
                handler.create(str(self.output_file), env_name)
            self._native_handler = handler
            self._native_episode_cls = EpisodeData
            self._torch = torch
        except Exception:
            self._native_handler = None

    def set_env_args(self, sim_args: dict) -> None:
        """Set the sim_args sub-dict written to data.attrs['env_args']."""
        self._env_args = {"env_name": self._env_name, "type": 2, "sim_args": sim_args}
        if self._native_handler is not None:
            self._native_handler.add_env_args({"sim_args": sim_args})

    def write_episode(self, episode: EpisodeBuffer, *, success: bool) -> str:
        if len(episode) == 0:
            raise ValueError("cannot write an empty episode")
        if self._native_handler is not None:
            return self._write_episode_native(episode, success=success)

        with self._h5py.File(self.output_file, "a") as handle:
            data = handle.require_group("data")

            # initialise file-level attrs if absent
            if "total" not in data.attrs:
                data.attrs["total"] = 0
            if self._env_args is not None and "env_args" not in data.attrs:
                data.attrs["env_args"] = json.dumps(self._env_args)

            # find next free demo slot
            while f"demo_{self._next_index:04d}" in data:
                self._next_index += 1
            name = f"demo_{self._next_index:04d}"
            group = data.create_group(name)

            T = len(episode)
            group.attrs["num_samples"] = T
            group.attrs["success"] = bool(success)

            # --- actions (T, 18) ---
            group.create_dataset("actions", data=np.stack(episode.actions, axis=0), compression="gzip")

            # --- policy obs (T, 91) ---
            group.create_dataset(
                "obs/actor_obs", data=np.stack(episode.obs, axis=0), compression="gzip"
            )

            # --- subtask term signals (T,) per signal ---
            if episode.subtask_signals:
                all_signal_names = {k for step in episode.subtask_signals for k in step}
                for sig in sorted(all_signal_names):
                    arr = np.array(
                        [float(step.get(sig, 0.0)) for step in episode.subtask_signals],
                        dtype=np.float32,
                    )
                    group.create_dataset(
                        f"obs/datagen_info/subtask_term_signals/{sig}",
                        data=arr,
                        compression="gzip",
                    )

            # --- initial_state (nested group) ---
            if episode.initial_state is not None:
                _write_nested_dict(group, "initial_state", episode.initial_state)

            # --- auxiliary ---
            group.create_dataset(
                "rewards", data=np.asarray(episode.rewards, dtype=np.float32), compression="gzip"
            )
            group.create_dataset(
                "dones", data=np.asarray(episode.dones, dtype=np.bool_), compression="gzip"
            )

            data.attrs["total"] = int(data.attrs["total"]) + T
            self._next_index += 1
            return name

    def close(self) -> None:
        if self._native_handler is not None:
            self._native_handler.close()

    def _write_episode_native(self, episode: EpisodeBuffer, *, success: bool) -> str:
        torch = self._torch
        episode_cls = self._native_episode_cls
        assert torch is not None and episode_cls is not None and self._native_handler is not None

        native_episode = episode_cls()
        native_episode.success = bool(success)
        native_episode.data = {
            "actions": torch.as_tensor(np.stack(episode.actions, axis=0), dtype=torch.float32),
            "rewards": torch.as_tensor(np.asarray(episode.rewards, dtype=np.float32)),
            "dones": torch.as_tensor(np.asarray(episode.dones, dtype=np.bool_)),
        }
        if episode.obs:
            native_episode.data["obs"] = {
                "actor_obs": torch.as_tensor(np.stack(episode.obs, axis=0), dtype=torch.float32)
            }
        if episode.initial_state is not None:
            native_episode.data["initial_state"] = _torch_state(episode.initial_state, torch)

        demo_id = self._native_handler.demo_count
        self._native_handler.write_episode(native_episode)
        self._native_handler.flush()
        return f"demo_{demo_id}"


def _write_nested_dict(parent_group, key: str, value) -> None:
    """Recursively write a dict-of-arrays as nested HDF5 groups/datasets."""
    if isinstance(value, dict):
        grp = parent_group.require_group(key)
        for sub_key, sub_val in value.items():
            _write_nested_dict(grp, sub_key, sub_val)
    else:
        arr = np.asarray(value, dtype=np.float32)
        parent_group.create_dataset(key, data=arr, compression="gzip")


def _torch_state(state: dict, torch_module) -> dict:
    """Recursively convert a numpy scene state dict to CPU torch tensors."""
    out = {}
    for k, v in state.items():
        if isinstance(v, dict):
            out[k] = _torch_state(v, torch_module)
        else:
            out[k] = torch_module.as_tensor(np.asarray(v), dtype=torch_module.float32)
    return out


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _identity_right_palm_fk(joints: np.ndarray) -> np.ndarray:
    """Fallback FK for dry import tests; real recording must pass env FK."""

    del joints
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


class FabricsRightPalmFK:
    """FABRICS FK provider for target right-palm pose from 7D right arm joints."""

    def __init__(self, *, device: str = "cpu", repo_root: Path | None = None) -> None:
        root = repo_root or Path(__file__).resolve().parents[2]
        fabrics_src = root / "hdgp" / "source" / "FABRICS" / "src"
        if fabrics_src.exists() and str(fabrics_src) not in sys.path:
            sys.path.insert(0, str(fabrics_src))

        import torch
        from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric
        from fabrics_sim.utils.utils import initialize_warp
        from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

        initialize_warp("pour_mimic_fk")
        self._torch = torch
        self._device = device
        world = WorldMeshesModel(batch_size=1, max_objects_per_env=0, device=device)
        self._fabric = OpenArmTeoslloPoseFabric(
            batch_size=1,
            device=device,
            timestep=1.0 / 300.0,
            graph_capturable=False,
            use_hand_fabric=False,
        )
        self._q = torch.zeros(1, 27, dtype=torch.float32, device=device)

    def __call__(self, joints: np.ndarray) -> np.ndarray:
        joints = _as_vector("right_arm_fk_joints", joints, 7)
        with self._torch.inference_mode():
            self._q[0, :7] = self._torch.as_tensor(joints, dtype=self._torch.float32, device=self._device)
            pose = self._fabric.get_palm_pose(self._q, "quaternion")[0]
        return pose.detach().cpu().numpy().astype(np.float64)


class EnvRightPalmPoseProvider:
    """Read current right palm pose from the live IsaacLab articulation."""

    def __init__(self, env, *, body_name: str = "rl_dg_palm") -> None:
        self.env = env
        self.body_name = body_name
        self._body_index: int | None = None

    def __call__(self) -> np.ndarray:
        robot = self.env.scene["robot"]
        if self._body_index is None:
            self._body_index = robot.data.body_names.index(self.body_name)
        pos = robot.data.body_pos_w[0, self._body_index, :]
        if hasattr(self.env.scene, "env_origins"):
            pos = pos - self.env.scene.env_origins[0]
        quat_wxyz = robot.data.body_quat_w[0, self._body_index, :]
        pose = self._torch_cat_pose(pos, quat_wxyz[[1, 2, 3, 0]])
        return pose.detach().cpu().numpy().astype(np.float64)

    @staticmethod
    def _torch_cat_pose(pos, quat_xyzw):
        import torch

        return torch.cat([pos, quat_xyzw], dim=-1)


class ROS2DemoRecorder:
    """ROS2 command sampler that steps a Pour-Mimic env and records HDF5 demos.

    Usage per episode:
      1. Call ``reset_episode()`` after env.reset() to capture initial_state.
      2. Call ``step_once()`` in a loop; it records obs/actions/subtask signals.
      3. Call ``save_success()`` or ``discard_episode()`` at the end.
    """

    def __init__(
        self,
        env,
        output_file: str | Path,
        *,
        task_name: str = "Pour-Mimic-V1-Mimic-v0",
        right_palm_fk: PoseFk | None = None,
        current_right_palm_pose: PoseProvider | None = None,
        control_hz: float = CONTROL_HZ,
    ) -> None:
        self.env = env
        self.command = TeleopCommandState()
        self.current_right_arm = np.zeros(7, dtype=np.float64)
        self.current_left_arm = np.zeros(7, dtype=np.float64)
        self.right_palm_fk = right_palm_fk or _identity_right_palm_fk
        self.current_right_palm_pose = current_right_palm_pose
        self.control_hz = float(control_hz)
        self.episode = EpisodeBuffer()
        self.writer = HDF5DemoWriter(output_file, env_name=task_name)
        self._sim_args_registered = False
        self._saved_count = 0

    def _register_sim_args(self) -> None:
        """Write sim_args to env_args once, after env is ready."""
        if self._sim_args_registered:
            return
        cfg = getattr(self.env, "cfg", None)
        sim_args: dict = {}
        if cfg is not None:
            sim_cfg = getattr(cfg, "sim", None)
            if sim_cfg is not None:
                sim_args["dt"] = float(getattr(sim_cfg, "dt", 0.0))
            sim_args["decimation"] = int(getattr(cfg, "decimation", 1))
            sim_args["render_interval"] = int(getattr(cfg, "decimation", 1))
            sim_args["num_envs"] = int(getattr(getattr(cfg, "scene", None), "num_envs", 1))
        self.writer.set_env_args(sim_args)
        self._sim_args_registered = True

    def reset_episode(self) -> None:
        """Capture scene initial_state and clear the buffer.

        Call this immediately after env.reset() so that initial_state is
        recorded before the first step.
        """
        self._register_sim_args()
        self.episode.clear()
        self._sync_current_joints_from_env()
        self.command.update_right_arm(self.current_right_arm)
        self.command.update_left_arm(self.current_left_arm)
        scene = getattr(self.env, "scene", None)
        if scene is not None and hasattr(scene, "get_state"):
            try:
                state = scene.get_state(is_relative=True)
                self.episode.set_initial_state(state)
            except Exception as exc:
                print(f"[ROS2DemoRecorder] Warning: get_state failed: {exc}")

    def make_action(self) -> np.ndarray:
        current_palm_pose = self.current_right_palm_pose() if self.current_right_palm_pose is not None else None
        return build_pour_mimic_action(
            current_right_arm=self.current_right_arm,
            target_right_arm=self.command.right_arm,
            target_right_hand=self.command.right_hand,
            current_left_arm=self.current_left_arm,
            target_left_arm=self.command.left_arm,
            right_palm_fk=self.right_palm_fk,
            current_right_palm_pose=current_palm_pose,
        )

    def _get_subtask_signals(self) -> dict[str, bool]:
        """Read current subtask term signals from the env (best-effort)."""
        if not hasattr(self.env, "get_subtask_term_signals"):
            return {}
        try:
            raw = self.env.get_subtask_term_signals()
            return {k: bool(v[0].item()) if hasattr(v, "__len__") else bool(v) for k, v in raw.items()}
        except Exception:
            return {}

    def step_once(self) -> tuple[np.ndarray, float, bool]:
        action = self.make_action()
        action_batch = action[None, :]
        try:
            import torch

            action_batch = torch.as_tensor(action_batch, dtype=torch.float32, device=getattr(self.env, "device", "cpu"))
        except Exception:
            pass
        obs, reward, terminated, truncated, _info = self.env.step(action_batch)
        done = bool(_to_numpy(terminated).any() or _to_numpy(truncated).any())

        obs_np = _to_numpy(
            obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs
        ).reshape(-1)
        reward_f = float(_to_numpy(reward).reshape(-1)[0])
        subtask_signals = self._get_subtask_signals()

        self.episode.append(obs_np, action, reward_f, done, subtask_signals=subtask_signals)
        self.current_right_arm = self.command.right_arm.copy()
        self.current_left_arm = self.command.left_arm.copy()
        return action, reward_f, done

    def save_success(self) -> str:
        name = self.writer.write_episode(self.episode, success=True)
        self._saved_count += 1
        self.episode.clear()
        return name

    def discard_episode(self) -> None:
        self.episode.clear()

    @property
    def saved_count(self) -> int:
        return self._saved_count

    def close(self) -> None:
        self.writer.close()

    def _sync_current_joints_from_env(self) -> None:
        try:
            robot = self.env.scene["robot"]
            names = robot.joint_names
            joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
            right_names = [f"openarm_right_joint{i}" for i in range(1, 8)]
            left_names = [f"openarm_left_joint{i}" for i in range(1, 8)]
            self.current_right_arm = np.asarray([joint_pos[names.index(name)] for name in right_names], dtype=np.float64)
            self.current_left_arm = np.asarray([joint_pos[names.index(name)] for name in left_names], dtype=np.float64)
        except Exception:
            pass


class _NonBlockingKeyboard:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled and sys.stdin.isatty()
        self._termios = None
        self._old_attrs = None

    def __enter__(self):
        if self.enabled:
            import termios
            import tty

            self._termios = termios
            self._old_attrs = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        if self.enabled and self._termios is not None and self._old_attrs is not None:
            self._termios.tcsetattr(sys.stdin, self._termios.TCSADRAIN, self._old_attrs)

    def poll(self) -> str | None:
        if not self.enabled:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if readable:
            return sys.stdin.read(1).lower()
        return None


def _parse_env_kwargs(items: list[str]) -> dict:
    if len(items) % 2 != 0:
        raise ValueError(f"unknown env args must be --key value pairs, got: {items}")
    out = {}
    for key, value in zip(items[0::2], items[1::2]):
        if not key.startswith("--"):
            raise ValueError(f"unknown env arg key must start with '--', got {key!r}")
        out[key[2:]] = value
    return out


def _make_ros2_subscriber(recorder: ROS2DemoRecorder):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64, Float64MultiArray

    class ROS2CommandSubscriber(Node):
        def __init__(self) -> None:
            super().__init__("pour_mimic_demo_recorder")
            self.create_subscription(Float64MultiArray, RIGHT_ARM_TOPIC, self._right_arm_cb, 10)
            self.create_subscription(Float64MultiArray, RIGHT_HAND_TOPIC, self._right_hand_cb, 10)
            self.create_subscription(Float64MultiArray, LEFT_ARM_TOPIC, self._left_arm_cb, 10)
            self.create_subscription(Float64, LEFT_GRIPPER_TOPIC, self._left_gripper_cb, 10)

        def _right_arm_cb(self, msg) -> None:
            recorder.command.update_right_arm(list(msg.data))

        def _right_hand_cb(self, msg) -> None:
            recorder.command.update_right_hand(list(msg.data))

        def _left_arm_cb(self, msg) -> None:
            recorder.command.update_left_arm(list(msg.data))

        def _left_gripper_cb(self, msg) -> None:
            recorder.command.update_left_gripper(float(msg.data))

    if not rclpy.ok():
        rclpy.init(args=None)
    return rclpy, ROS2CommandSubscriber()


def _reset_env_and_episode(env, recorder: ROS2DemoRecorder) -> None:
    env.reset()
    recorder.reset_episode()


def _run_recording_loop(args, env, recorder: ROS2DemoRecorder) -> None:
    rclpy = None
    node = None
    if not args.dry_commands:
        rclpy, node = _make_ros2_subscriber(recorder)

    _reset_env_and_episode(env, recorder)
    print(
        "[ros2_demo_recorder] Recording. "
        "Interactive keys: S save, R discard/reset, Q quit."
        if not args.headless
        else "[ros2_demo_recorder] Headless recording."
    )

    step_in_episode = 0
    period = 1.0 / float(args.control_hz)
    try:
        with _NonBlockingKeyboard(enabled=not args.headless) as keyboard:
            while recorder.saved_count < args.num_demos:
                t0 = time.monotonic()
                if node is not None:
                    rclpy.spin_once(node, timeout_sec=0.0)
                _action, _reward, done = recorder.step_once()
                step_in_episode += 1

                key = keyboard.poll()
                if key == "s":
                    name = recorder.save_success()
                    print(f"[ros2_demo_recorder] Saved {name} ({recorder.saved_count}/{args.num_demos})")
                    if recorder.saved_count >= args.num_demos:
                        break
                    _reset_env_and_episode(env, recorder)
                    step_in_episode = 0
                elif key == "r":
                    recorder.discard_episode()
                    _reset_env_and_episode(env, recorder)
                    step_in_episode = 0
                    print("[ros2_demo_recorder] Discarded episode and reset.")
                elif key in ("q", "\x03"):
                    break

                reached_horizon = step_in_episode >= args.max_steps or done
                if args.headless and reached_horizon:
                    if args.save_on_success:
                        name = recorder.save_success()
                        print(f"[ros2_demo_recorder] Saved {name} ({recorder.saved_count}/{args.num_demos})")
                    else:
                        recorder.discard_episode()
                        print("[ros2_demo_recorder] Discarded headless episode; pass --save_on_success to export.")
                    if recorder.saved_count >= args.num_demos:
                        break
                    _reset_env_and_episode(env, recorder)
                    step_in_episode = 0

                sleep_s = period - (time.monotonic() - t0)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()


def _main() -> None:
    parser = argparse.ArgumentParser(description="Record Pour-Mimic-V1 demos from /isaacsim ROS2 teleop topics")
    parser.add_argument("--task", default="Pour-Mimic-V1-Mimic-v0")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--num_demos", type=int, default=35)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--control_hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--save_on_success", action="store_true")
    parser.add_argument("--dry_commands", action="store_true", help="Run without ROS2 and hold reset command targets.")
    args, unknown = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[2]
    openarm_src = repo_root / "hdgp" / "source" / "openarm"
    sys.path.insert(0, str(openarm_src))

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.both.pour_v1_mimic  # noqa: F401

    env = None
    recorder = None
    try:
        env_kwargs = _parse_env_kwargs(unknown)
        env = gym.make(args.task, device=args.device, headless=args.headless, **env_kwargs)
        unwrapped = env.unwrapped
        recorder = ROS2DemoRecorder(
            unwrapped,
            args.output_file,
            task_name=args.task,
            right_palm_fk=FabricsRightPalmFK(device=args.device, repo_root=repo_root),
            current_right_palm_pose=EnvRightPalmPoseProvider(unwrapped),
            control_hz=args.control_hz,
        )
        print(f"[ros2_demo_recorder] task={args.task} output={args.output_file} num_demos={args.num_demos}")
        _run_recording_loop(args, unwrapped, recorder)
    finally:
        if recorder is not None:
            recorder.close()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    _main()
