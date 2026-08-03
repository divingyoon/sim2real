#!/usr/bin/env python3
"""실물 Tesollo tip F/T → /dg5f_right/contact_forces 변환 노드.

dg5f 드라이버를 `fingertip_sensor:=true` 로 올리면 벤더 ForceTorqueSensorBroadcaster 가
`/fingertip_{1..5}_broadcaster/wrench`(WrenchStamped) 를 발행한다. 이 노드가 그 5개를
구독해 bias 제거 force norm 5D(`Float64MultiArray`) 로 `/dg5f_right/contact_forces` 에
발행한다 — fake_tip_contact_pub(항상 0) 의 실센서 대체.

  · bias: 기동 후 첫 --bias-samples 프레임 평균 (★손이 아무것도 안 닿은 상태에서 기동할 것)
  · /tip_contact/rebias 서비스: 무접촉 자세에서 bias 재캡처
  · 이진화는 grasp_inference 의 CONTACT_FORCE_THRESHOLD(0.1N) — 실물 노이즈 대비 튜닝 필요

tip 순서: fingertip_1..5 = thumb..pinky (= rj_dg_1..5 = canonical r_hj thumb..pinky).

실행(robot PC):
    ros2 launch dg5f_driver dg5f_right_driver.launch.py fingertip_sensor:=true
    python3 tip_contact_pub.py [--rate 60] [--bias-samples 30]
"""

from __future__ import annotations

import argparse

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from tip_contact_core import TipForceExtractor

NUM_TIPS = 5
DEFAULT_WRENCH_TOPIC_FMT = "/fingertip_{i}_broadcaster/wrench"


class TipContactPub(Node):
    def __init__(self, rate_hz: float, bias_samples: int, topic_fmt: str) -> None:
        super().__init__("tip_contact_pub")
        self.extractor = TipForceExtractor(num_tips=NUM_TIPS, bias_samples=bias_samples)
        self.pub = self.create_publisher(Float64MultiArray, "/dg5f_right/contact_forces", 10)

        self._rx_count = [0] * NUM_TIPS
        for tip in range(NUM_TIPS):
            topic = topic_fmt.format(i=tip + 1)
            self.create_subscription(
                WrenchStamped, topic,
                lambda msg, t=tip: self._wrench_cb(t, msg), 10,
            )
        self.create_service(Trigger, "/tip_contact/rebias", self._rebias_cb)
        self.create_timer(1.0 / rate_hz, self._tick)
        self._bias_announced = False
        self.get_logger().info(
            f"tip F/T→접촉 변환 기동: {topic_fmt.format(i='1..5')} → /dg5f_right/contact_forces "
            f"({rate_hz:g}Hz, bias {bias_samples}샘플)\n"
            "  ★bias 캡처 중 — 손끝이 아무것도 닿지 않은 상태여야 함"
        )

    def _wrench_cb(self, tip: int, msg: WrenchStamped) -> None:
        f = msg.wrench.force
        self.extractor.update(tip, [f.x, f.y, f.z])
        self._rx_count[tip] += 1

    def _rebias_cb(self, request, response):
        self.extractor.reset_bias()
        self._bias_announced = False
        response.success = True
        response.message = "bias 재캡처 시작 (무접촉 유지)"
        self.get_logger().info(response.message)
        return response

    def _tick(self) -> None:
        if not self._bias_announced and self.extractor.biased():
            self._bias_announced = True
            self.get_logger().info("bias 캡처 완료 — 접촉력 발행 유효")
        msg = Float64MultiArray()
        msg.data = self.extractor.forces().tolist()
        self.pub.publish(msg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=60.0)
    parser.add_argument("--bias-samples", type=int, default=30)
    parser.add_argument("--topic-fmt", default=DEFAULT_WRENCH_TOPIC_FMT,
                        help="tip wrench 토픽 패턴 ({i}=1..5)")
    args = parser.parse_args()

    rclpy.init()
    node = TipContactPub(args.rate, args.bias_samples, args.topic_fmt)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
