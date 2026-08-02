#!/usr/bin/env python3
"""브리지: /isaacsim/right_{arm,hand}_cmd (canonical) → robot_control JTC.

정책 노드(grasp_inference/pour_inference)가 발행하는 canonical 관절 명령을 robot_control
컨트롤러가 받는 source 관절 순서·부호·한계로 변환해 단일포인트 JointTrajectory 로 스트리밍한다.
robot_control 코드 무수정(Option B). robot PC(controllers 와 co-locate) 에서 실행 권장.

실행 (robot PC):
    python3 isaacsim_cmd_to_jtc.py \
        [--profile ~/rl_ws/robot_control/src/robot_control/profiles/openarm_tesollo.yaml] \
        [--arm-topic /right_joint_trajectory_controller/joint_trajectory] \
        [--hand-topic /dg5f_right/dg5f_right_controller/joint_trajectory] \
        [--control-dt 0.0167] [--horizon 2.0]

구독: /isaacsim/right_arm_cmd (Float64MultiArray, 7 canonical r_aj_1..7)
      /isaacsim/right_hand_cmd (Float64MultiArray, 20 canonical r_hj_* finger-major)
발행: <arm-topic>, <hand-topic> (trajectory_msgs/JointTrajectory, 단일포인트)

★ time_from_start = control_dt · horizon > 0 (0이면 JTC 무동작, [[jtc-none-interpolation-silent-stall]]).
  컨트롤러가 이 시각까지 현재→목표를 보간하므로 60Hz 스트림이 부드럽게 추종된다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from jtc_bridge_core import JointRemap, load_profile_joints, time_from_start_sec

ARM_CANON = [f"r_aj_{i}" for i in range(1, 8)]
ARM_SOURCE = [f"openarm_right_joint{i}" for i in range(1, 8)]
_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
HAND_CANON = [f"r_hj_{f}_{j}" for f in _FINGERS for j in range(1, 5)]
HAND_SOURCE = [f"rj_dg_{fi}_{j}" for fi in range(1, 6) for j in range(1, 5)]

DEFAULT_PROFILE = str(
    Path.home() / "rl_ws/robot_control/src/robot_control/profiles/openarm_tesollo.yaml"
)


def _duration(sec_float: float) -> Duration:
    sec = int(sec_float)
    nanosec = int(round((sec_float - sec) * 1e9))
    return Duration(sec=sec, nanosec=nanosec)


class IsaacsimCmdToJtc(Node):
    def __init__(
        self,
        profile_path: str,
        arm_topic: str,
        hand_topic: str,
        control_dt: float,
        horizon: float,
    ) -> None:
        super().__init__("isaacsim_cmd_to_jtc")
        prof = load_profile_joints(profile_path)
        self.arm_remap = JointRemap(ARM_CANON, ARM_SOURCE, prof)
        self.hand_remap = JointRemap(HAND_CANON, HAND_SOURCE, prof)
        self.tfs = time_from_start_sec(control_dt, horizon)

        self.arm_pub = self.create_publisher(JointTrajectory, arm_topic, 10)
        self.hand_pub = self.create_publisher(JointTrajectory, hand_topic, 10)
        self.create_subscription(Float64MultiArray, "/isaacsim/right_arm_cmd", self._arm_cb, 10)
        self.create_subscription(Float64MultiArray, "/isaacsim/right_hand_cmd", self._hand_cb, 10)

        self.get_logger().info(
            f"브리지 준비: arm→{arm_topic}, hand→{hand_topic}, tfs={self.tfs*1000:.1f}ms\n"
            f"  profile={profile_path}"
        )

    def _publish(self, pub, remap: JointRemap, values, n: int, label: str) -> None:
        if len(values) != n:
            self.get_logger().warn(f"{label} cmd 길이 {len(values)} != {n}, 무시")
            return
        positions = remap.apply(list(values))
        jt = JointTrajectory()
        jt.joint_names = list(remap.output_source)
        pt = JointTrajectoryPoint()
        pt.positions = positions.tolist()
        pt.time_from_start = _duration(self.tfs)
        jt.points = [pt]
        pub.publish(jt)

    def _arm_cb(self, msg: Float64MultiArray) -> None:
        self._publish(self.arm_pub, self.arm_remap, msg.data, 7, "arm")

    def _hand_cb(self, msg: Float64MultiArray) -> None:
        self._publish(self.hand_pub, self.hand_remap, msg.data, 20, "hand")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--arm-topic", default="/right_joint_trajectory_controller/joint_trajectory")
    parser.add_argument("--hand-topic", default="/dg5f_right/dg5f_right_controller/joint_trajectory")
    parser.add_argument("--control-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--horizon", type=float, default=2.0,
                        help="목표 도달 주기 수 (tfs = control_dt·horizon, >0)")
    args = parser.parse_args()

    rclpy.init()
    node = IsaacsimCmdToJtc(
        profile_path=args.profile,
        arm_topic=args.arm_topic,
        hand_topic=args.hand_topic,
        control_dt=args.control_dt,
        horizon=args.horizon,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
