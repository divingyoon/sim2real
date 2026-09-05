#!/usr/bin/env python3
"""IsaacLab Se3Device-compatible ROS2 teleop device.

The plain-Python path is intentionally dependency-light so unit tests can import
and exercise the device without a ROS2 installation.
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable

import numpy as np


DEFAULT_DELTA_TOPIC = "/isaacsim/eef_delta_cmd"
DEFAULT_GRIPPER_TOPIC = "/isaacsim/eef_gripper_cmd"
DEFAULT_RESET_TOPIC = "/isaacsim/eef_reset"


def _vec3(name: str, values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} expects 3 values, got {arr.size}")
    return arr


class Se3ROS2Device:
    """Minimal Se3Device adapter returning (delta_pos, delta_rot, gripper, reset)."""

    def __init__(
        self,
        *,
        use_ros: bool = True,
        delta_topic: str = DEFAULT_DELTA_TOPIC,
        gripper_topic: str = DEFAULT_GRIPPER_TOPIC,
        reset_topic: str = DEFAULT_RESET_TOPIC,
    ) -> None:
        self.delta_pos = np.zeros(3, dtype=np.float64)
        self.delta_rot = np.zeros(3, dtype=np.float64)
        self.gripper = 0.0
        self.reset = False
        self.timestamp = time.time()
        self._node = None

        if use_ros:
            try:
                import rclpy
                from rclpy.node import Node
                from std_msgs.msg import Bool, Float64, Float64MultiArray
            except ImportError as exc:
                raise RuntimeError("ROS2 rclpy and std_msgs are required when use_ros=True") from exc

            if not rclpy.ok():
                rclpy.init(args=None)

            class _DeviceNode(Node):
                pass

            self._rclpy = rclpy
            self._node = _DeviceNode("se3_ros2_device")
            self._node.create_subscription(Float64MultiArray, delta_topic, self._delta_cb, 10)
            self._node.create_subscription(Float64, gripper_topic, self._gripper_cb, 10)
            self._node.create_subscription(Bool, reset_topic, self._reset_cb, 10)

    def _delta_cb(self, msg) -> None:
        data = list(msg.data)
        if len(data) != 6:
            raise ValueError(f"eef delta command expects 6 values, got {len(data)}")
        self.update_command(delta_pos=data[:3], delta_rot=data[3:6])

    def _gripper_cb(self, msg) -> None:
        self.update_command(gripper=float(msg.data))

    def _reset_cb(self, msg) -> None:
        self.update_command(reset=bool(msg.data))

    def update_command(
        self,
        *,
        delta_pos: Iterable[float] | None = None,
        delta_rot: Iterable[float] | None = None,
        gripper: float | None = None,
        reset: bool | None = None,
    ) -> None:
        if delta_pos is not None:
            self.delta_pos = _vec3("delta_pos", delta_pos)
        if delta_rot is not None:
            self.delta_rot = _vec3("delta_rot", delta_rot)
        if gripper is not None:
            self.gripper = float(np.clip(float(gripper), -1.0, 1.0))
        if reset is not None:
            self.reset = bool(reset)
        self.timestamp = time.time()

    def advance(self):
        """Return the tuple expected by IsaacLab teleop SE3 agents."""

        if self._node is not None:
            self._rclpy.spin_once(self._node, timeout_sec=0.0)
        reset = self.reset
        self.reset = False
        return self.delta_pos.copy(), self.delta_rot.copy(), float(self.gripper), reset

    def reset_internal_state(self) -> None:
        self.delta_pos.fill(0.0)
        self.delta_rot.fill(0.0)
        self.gripper = 0.0
        self.reset = False

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None


def _main() -> None:
    parser = argparse.ArgumentParser(description="Read a ROS2 SE3 teleop command once")
    parser.add_argument("--no-ros", action="store_true", help="Use dependency-free dry mode")
    args = parser.parse_args()

    device = Se3ROS2Device(use_ros=not args.no_ros)
    print(device.advance())
    device.close()


if __name__ == "__main__":
    _main()
