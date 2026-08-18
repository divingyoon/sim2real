#!/usr/bin/env python3
"""실물 Tesollo tip F/T → 접촉력 토픽 변환 노드 (구성 프로필 기반, 좌/우).

dg5f 드라이버를 `fingertip_sensor:=true` 로 올리면 벤더 ForceTorqueSensorBroadcaster 가
`fingertip_{1..5}_broadcaster/wrench`(WrenchStamped) 를 발행한다. 이 노드가 그 5개를
구독해 bias 를 제거하고 **두 토픽**으로 발행한다:

  · `<tip_force_xyz>`  15D (5×3, tip-major)  ← 정책 obs `tip_force_local` 주경로
  · `<tip_force_norm>`  5D (norm)            ← 기존 구독자·모니터링 하위호환

★왜 토픽을 나누는가: 같은 이름(`contact_forces`)의 차원만 5→15 로 바꾸면 구 발행자가
  살아 있을 때 혼재가 조용히 통과한다. 이름을 분리하면 "데이터가 아예 안 온다"로
  시끄럽게 실패한다. 이 저장소의 사고 이력은 전부 조용한 실패 계열이었다.

  · bias: 기동 후 첫 --bias-samples 프레임 평균 (★손이 아무것도 안 닿은 상태에서 기동할 것)
  · /tip_contact/rebias 서비스: 무접촉 자세에서 bias 재캡처
    ⚠️ bias 는 자세 의존적이다 — 손이 크게 움직였으면 무접촉이 확실한 시점에 재호출할 것
  · 이진화는 소비자 쪽에서 norm > CONTACT_FORCE_THRESHOLD(0.1N) 로 파생 — 실물 노이즈
    대비 튜닝 필요

tip 순서: fingertip_1..5 = thumb..pinky (= {r|l}j_dg_1..5 = canonical {r|l}_hj thumb..pinky).

실행:
    ros2 launch dg5f_driver dg5f_right_driver.launch.py fingertip_sensor:=true
    python3 tip_contact_pub.py --robot tesollo_bi_s__right [--rate 60] [--bias-samples 30]
"""

from __future__ import annotations

import argparse

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from std_srvs.srv import Trigger

from robot_profile import load_robot_profile
from tip_contact_core import TipForceExtractor

NUM_TIPS = 5


def _xyz_layout(msg: Float64MultiArray) -> None:
    """레이아웃을 명시해 잘못된 발행자를 식별 가능하게 한다 (tip-major 5×3)."""
    msg.layout.dim = [
        MultiArrayDimension(label="tip", size=NUM_TIPS, stride=NUM_TIPS * 3),
        MultiArrayDimension(label="axis", size=3, stride=3),
    ]


class TipContactPub(Node):
    def __init__(self, profile, rate_hz: float, bias_samples: int, topic_fmt: str) -> None:
        super().__init__("tip_contact_pub")
        self.profile = profile
        self.extractor = TipForceExtractor(
            num_tips=NUM_TIPS, bias_samples=bias_samples, sign=profile.tip_force_sign,
        )
        xyz_topic = profile.topics["tip_force_xyz"]
        norm_topic = profile.topics["tip_force_norm"]
        self.pub_xyz = self.create_publisher(Float64MultiArray, xyz_topic, 10)
        self.pub_norm = self.create_publisher(Float64MultiArray, norm_topic, 10)

        self._rx_count = [0] * NUM_TIPS
        for tip in range(NUM_TIPS):
            self.create_subscription(
                WrenchStamped, topic_fmt.format(i=tip + 1),
                lambda msg, t=tip: self._wrench_cb(t, msg), 10,
            )
        self.create_service(Trigger, "/tip_contact/rebias", self._rebias_cb)
        self.create_timer(1.0 / rate_hz, self._tick)
        self._bias_announced = False
        self.get_logger().info(
            f"tip F/T→접촉 변환 기동 [{profile.name}]\n"
            f"  구독 {topic_fmt.format(i='1..5')}\n"
            f"  발행 {xyz_topic} (15D, 정책 주경로) · {norm_topic} (5D norm, 하위호환)\n"
            f"  {rate_hz:g}Hz · bias {bias_samples}샘플 · force_sign {profile.tip_force_sign:+g}\n"
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

        xyz = Float64MultiArray()
        _xyz_layout(xyz)
        xyz.data = self.extractor.forces_xyz().reshape(-1).tolist()   # tip-major (15,)
        self.pub_xyz.publish(xyz)

        norm = Float64MultiArray()
        norm.data = self.extractor.forces().tolist()                  # (5,)
        self.pub_norm.publish(norm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="tesollo_bi_s__right",
                        help="config/robots 의 구성 프로필 이름")
    parser.add_argument("--rate", type=float, default=60.0)
    parser.add_argument("--bias-samples", type=int, default=30)
    parser.add_argument("--topic-fmt", default=None,
                        help="tip wrench 토픽 패턴 ({i}=1..5). 기본은 프로필 값")
    args = parser.parse_args()

    profile = load_robot_profile(args.robot)
    topic_fmt = args.topic_fmt or profile.topics["tip_wrench_fmt"]

    rclpy.init()
    node = TipContactPub(profile, args.rate, args.bias_samples, topic_fmt)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
