#!/usr/bin/env python3
"""임시 fingertip 접촉(0) 퍼블리셔 — 실제 dg5f 손 드라이버와 병행용 stopgap.

실제 dg5f_right_driver 는 /dg5f_right/joint_states 와 컨트롤러는 제공하지만
/dg5f_right/contact_forces(정책 obs·start 게이트 필요) 는 발행하지 않는다.
Tesollo tip F/T → 5 tip 접촉 변환 노드가 준비되기 전까지, 이 노드가 접촉 0 을
발행해 게이트를 통과시키고 손 관절은 실제 드라이버 값을 쓰게 한다.

⚠️ 접촉은 0 고정 — 접촉-게이트 파지/lift 는 미완성. 손 관절 반응만 실값으로 검증하는 용도.
실접촉이 필요하면 F/T 변환 노드로 교체할 것. [[grasp-v2-contact-obs-sim2real]]

발행: <tip_force_xyz> 15D (5×3×0.0) + <tip_force_norm> 5D (5×0.0)
      — 구성 프로필의 토픽을 쓴다(좌/우 공통).
"""

from __future__ import annotations

import argparse

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from robot_profile import load_robot_profile

NUM_TIPS = 5


class FakeTipContact(Node):
    def __init__(self, profile, rate_hz: float) -> None:
        super().__init__("fake_tip_contact_pub")
        self.xyz_topic = profile.topics["tip_force_xyz"]
        self.norm_topic = profile.topics["tip_force_norm"]
        self.pub_xyz = self.create_publisher(Float64MultiArray, self.xyz_topic, 10)
        self.pub_norm = self.create_publisher(Float64MultiArray, self.norm_topic, 10)
        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f"임시 접촉(0) 발행 [{profile.name}]: {self.xyz_topic} (15×0.0) + "
            f"{self.norm_topic} ({NUM_TIPS}×0.0), {rate_hz:g}Hz\n"
            "  ⚠️ 실접촉 아님 — 실 dg5f joint_states 병행 stopgap(게이트 통과용)"
        )

    def _tick(self) -> None:
        xyz = Float64MultiArray()
        xyz.data = [0.0] * (NUM_TIPS * 3)
        self.pub_xyz.publish(xyz)
        norm = Float64MultiArray()
        norm.data = [0.0] * NUM_TIPS
        self.pub_norm.publish(norm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="tesollo_bi_s__right",
                        help="config/robots 의 구성 프로필 이름")
    parser.add_argument("--rate", type=float, default=30.0)
    args = parser.parse_args()
    profile = load_robot_profile(args.robot)
    rclpy.init()
    node = FakeTipContact(profile, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
