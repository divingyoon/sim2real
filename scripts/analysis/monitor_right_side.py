#!/usr/bin/env python3

"""Monitor current right-side command/state topics in one terminal."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory


RIGHT_HAND_JOINTS = {
    "rj_dg_1_1", "rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4",
    "rj_dg_2_1", "rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4",
    "rj_dg_3_1", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4",
    "rj_dg_4_1", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4",
    "rj_dg_5_1", "rj_dg_5_2", "rj_dg_5_3", "rj_dg_5_4",
}

RIGHT_ARM_JOINTS = [
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
]


class RightSideMonitor(Node):
    def __init__(self) -> None:
        super().__init__("right_side_monitor")
        self.last_right_arm_cmd: list[float] | None = None
        self.last_right_hand_cmd: list[float] | None = None
        self.last_right_arm_traj: list[float] | None = None
        self.last_right_hand_traj: list[float] | None = None
        self.create_subscription(Float64MultiArray, "/isaacsim/right_arm_cmd", self._right_arm_cb, 10)
        self.create_subscription(Float64MultiArray, "/isaacsim/right_hand_cmd", self._right_hand_cb, 10)
        self.create_subscription(JointTrajectory, "/right_joint_trajectory_controller/joint_trajectory", self._right_arm_traj_cb, 10)
        self.create_subscription(JointTrajectory, "/dg5f_right/dg5f_right_controller/joint_trajectory", self._right_hand_traj_cb, 10)
        self.create_subscription(JointState, "/isaacsim/joint_states", self._joint_state_cb, 10)

    def _right_arm_cb(self, msg: Float64MultiArray) -> None:
        self.last_right_arm_cmd = list(msg.data)
        print(f"[RIGHT CMD] arm={self.last_right_arm_cmd}")

    def _right_hand_cb(self, msg: Float64MultiArray) -> None:
        self.last_right_hand_cmd = list(msg.data)
        print(f"[RIGHT CMD] hand={self.last_right_hand_cmd}")

    def _right_arm_traj_cb(self, msg: JointTrajectory) -> None:
        if msg.points:
            self.last_right_arm_traj = list(msg.points[0].positions)
            print(f"[RIGHT TRAJ] arm_positions={self.last_right_arm_traj}")
            if self.last_right_arm_cmd is not None:
                arm_error = [
                    round(cmd - cur, 4)
                    for cmd, cur in zip(self.last_right_arm_cmd, self.last_right_arm_traj)
                ]
                print(
                    f"[RIGHT COMPARE] arm_cmd={self.last_right_arm_cmd} "
                    f"arm_target={self.last_right_arm_traj} arm_error={arm_error}"
                )

    def _right_hand_traj_cb(self, msg: JointTrajectory) -> None:
        if msg.points:
            self.last_right_hand_traj = list(msg.points[0].positions)
            print(f"[RIGHT TRAJ] hand_positions={self.last_right_hand_traj}")
            if self.last_right_hand_cmd is not None:
                hand_error = [
                    round(cmd - cur, 4)
                    for cmd, cur in zip(self.last_right_hand_cmd, self.last_right_hand_traj)
                ]
                print(
                    f"[RIGHT COMPARE] hand_cmd={self.last_right_hand_cmd} "
                    f"hand_target={self.last_right_hand_traj} hand_error={hand_error}"
                )

    def _joint_state_cb(self, msg: JointState) -> None:
        state_map = {
            name: msg.position[i]
            for i, name in enumerate(msg.name)
            if i < len(msg.position)
        }
        paired = [(name, state_map[name]) for name in RIGHT_ARM_JOINTS if name in state_map]
        hand_joint_names = sorted(RIGHT_HAND_JOINTS)
        paired.extend((name, state_map[name]) for name in hand_joint_names if name in state_map)
        if paired:
            print(f"[RIGHT STATE] {paired}")
        if self.last_right_arm_cmd is not None and all(name in state_map for name in RIGHT_ARM_JOINTS):
            arm_state = [state_map[name] for name in RIGHT_ARM_JOINTS]
            arm_error = [
                round(cmd - cur, 4) for cmd, cur in zip(self.last_right_arm_cmd, arm_state)
            ]
            print(f"[RIGHT COMPARE] arm_cmd={self.last_right_arm_cmd} arm_state={arm_state} arm_error={arm_error}")
        if self.last_right_hand_cmd is not None:
            ordered_hand_names = [name for name in hand_joint_names if name in state_map]
            if len(ordered_hand_names) == len(self.last_right_hand_cmd):
                hand_state = [state_map[name] for name in ordered_hand_names]
                hand_error = [
                    round(cmd - cur, 4) for cmd, cur in zip(self.last_right_hand_cmd, hand_state)
                ]
                print(
                    f"[RIGHT COMPARE] hand_cmd={self.last_right_hand_cmd} "
                    f"hand_state={hand_state} hand_error={hand_error}"
                )


def main() -> None:
    rclpy.init()
    node = RightSideMonitor()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
