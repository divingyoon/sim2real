#!/usr/bin/env python3
"""fake 손 상태 퍼블리셔 — 테솔로 손 분리 상태에서 풀 파이프라인 플러밍 검증용.

grasp_inference 는 /dg5f_right/joint_states(손 20관절)와 /dg5f_right/contact_forces(tip 5)
가 있어야 start 게이트를 통과하고 obs 를 조립한다. 손을 분리했을 때 이 노드가 **정적 손 상태**
(APPROACH 자세)와 **접촉 0**을 발행해, 지각→정책→브리지→팔 경로를 손 없이 흐르게 한다.

⚠️ 실제 손 구동은 없음(손 분리). 정책의 손 명령은 무시된다 — 팔 궤적만 검증하는 용도.

발행:
    /dg5f_right/joint_states     sensor_msgs/JointState  (source명 rj_dg_*, APPROACH 자세)
    /dg5f_right/contact_forces   std_msgs/Float64MultiArray (5×0.0)
"""

from __future__ import annotations

import argparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# source 관절명 (dg5f 드라이버 발행명), finger-major
HAND_SOURCE = [f"rj_dg_{fi}_{j}" for fi in range(1, 6) for j in range(1, 5)]

# grasp_v1 preset HAND_APPROACH_POSE (canonical finger-major, sign +1 → source 동일 순서)
HAND_APPROACH_POSE = [
    0.0, -1.57, -0.5, 0.0,   # thumb
    0.0, 0.0, 0.0, 0.0,      # index
    0.0, 0.0, 0.0, 0.0,      # middle
    0.0, 0.0, 0.0, 0.0,      # ring
    0.0, 0.0, 0.0, 0.0,      # pinky
]


class FakeHandState(Node):
    def __init__(self, rate_hz: float) -> None:
        super().__init__("fake_hand_state_pub")
        self.js_pub = self.create_publisher(JointState, "/dg5f_right/joint_states", 10)
        self.ct_pub = self.create_publisher(Float64MultiArray, "/dg5f_right/contact_forces", 10)
        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f"fake 손 상태 발행: /dg5f_right/joint_states (APPROACH), "
            f"/dg5f_right/contact_forces (0), {rate_hz:g}Hz\n"
            "  ⚠️ 손 분리 상태 플러밍 검증용 — 실제 손 구동 아님"
        )

    def _tick(self) -> None:
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(HAND_SOURCE)
        js.position = list(HAND_APPROACH_POSE)
        js.velocity = [0.0] * len(HAND_SOURCE)
        self.js_pub.publish(js)

        ct = Float64MultiArray()
        ct.data = [0.0] * 5
        self.ct_pub.publish(ct)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=30.0)
    args = parser.parse_args()
    rclpy.init()
    node = FakeHandState(args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
