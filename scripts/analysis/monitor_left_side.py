#!/usr/bin/env python3

"""Monitor current left-side command/state topics in one terminal."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory


LEFT_ARM_JOINTS = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
]


class LeftSideMonitor(Node):
    def __init__(self) -> None:
        super().__init__("left_side_monitor")
        self.last_left_arm_cmd: list[float] | None = None
        self.last_left_gripper_cmd: float | None = None
        self.last_left_arm_traj: list[float] | None = None
        self.create_subscription(Float64MultiArray, "/isaacsim/left_arm_cmd", self._left_arm_cb, 10)
        self.create_subscription(Float64, "/isaacsim/left_gripper_cmd", self._left_gripper_cb, 10)
        self.create_subscription(JointTrajectory, "/left_joint_trajectory_controller/joint_trajectory", self._left_arm_traj_cb, 10)
        self.create_subscription(JointState, "/isaacsim/joint_states", self._joint_state_cb, 10)

    def _left_arm_cb(self, msg: Float64MultiArray) -> None:
        self.last_left_arm_cmd = list(msg.data)
        print(f"[LEFT CMD] arm={self.last_left_arm_cmd}")

    def _left_gripper_cb(self, msg: Float64) -> None:
        self.last_left_gripper_cmd = float(msg.data)
        print(f"[LEFT CMD] gripper={self.last_left_gripper_cmd}")

    def _left_arm_traj_cb(self, msg: JointTrajectory) -> None:
        if msg.points:
            self.last_left_arm_traj = list(msg.points[0].positions)
            print(f"[LEFT TRAJ] positions={self.last_left_arm_traj}")
            if self.last_left_arm_cmd is not None:
                arm_error = [
                    round(cmd - cur, 4)
                    for cmd, cur in zip(self.last_left_arm_cmd, self.last_left_arm_traj)
                ]
                print(
                    f"[LEFT COMPARE] arm_cmd={self.last_left_arm_cmd} "
                    f"arm_target={self.last_left_arm_traj} arm_error={arm_error}"
                )

    def _joint_state_cb(self, msg: JointState) -> None:
        state_map = {
            name: msg.position[i]
            for i, name in enumerate(msg.name)
            if i < len(msg.position)
        }
        paired = [(name, state_map[name]) for name in LEFT_ARM_JOINTS if name in state_map]
        if "openarm_left_finger_joint1" in state_map:
            paired.append(("openarm_left_finger_joint1", state_map["openarm_left_finger_joint1"]))
        if paired:
            print(f"[LEFT STATE] {paired}")
        if self.last_left_arm_cmd is not None and len(paired) >= 7:
            arm_state = [state_map[name] for name in LEFT_ARM_JOINTS]
            arm_error = [
                round(cmd - cur, 4) for cmd, cur in zip(self.last_left_arm_cmd, arm_state)
            ]
            print(f"[LEFT COMPARE] arm_cmd={self.last_left_arm_cmd} arm_state={arm_state} arm_error={arm_error}")
        if self.last_left_gripper_cmd is not None and "openarm_left_finger_joint1" in state_map:
            finger_state = state_map["openarm_left_finger_joint1"]
            finger_error = round(self.last_left_gripper_cmd - finger_state, 4)
            print(
                f"[LEFT COMPARE] gripper_cmd={self.last_left_gripper_cmd} "
                f"gripper_state={finger_state} gripper_error={finger_error}"
            )


def main() -> None:
    rclpy.init()
    node = LeftSideMonitor()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
