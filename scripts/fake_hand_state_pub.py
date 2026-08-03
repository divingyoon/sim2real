#!/usr/bin/env python3
"""fake 손 상태 퍼블리셔 — 테솔로 손 분리 상태에서 풀 파이프라인 플러밍 검증용.

grasp_inference 는 /dg5f_right/joint_states(손 20관절)와 /dg5f_right/contact_forces(tip 5)
가 있어야 start 게이트를 통과하고 obs 를 조립한다. 손을 분리했을 때 이 노드가 **정적 손 상태**
(APPROACH 자세)와 **접촉 0**을 발행해, 지각→정책→브리지→팔 경로를 손 없이 흐르게 한다.

⚠️ 실제 손 구동은 없음(손 분리). 정책의 손 명령은 무시된다 — 팔 궤적만 검증하는 용도.

--echo 모드(08.03, RUNNING 팔 후퇴 진단): 정적 APPROACH 대신 정책의 손 명령
`/isaacsim/right_hand_cmd`(canonical 20D)를 그대로 관절상태로 되돌려 발행한다.
sim 에서는 손이 명령을 즉시 추종하므로, echo 는 "sim 처럼 진화하는 손 obs"를 손 없이
재현한다 → 정적 손 obs 오염(LSTM 발산) 가설의 실기측 분리 실험. 접촉은 여전히 0
(lift 래치는 미발동 — 접근 거동만 판정).

발행:
    /dg5f_right/joint_states     sensor_msgs/JointState  (source명 rj_dg_*, APPROACH 또는 echo)
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
    def __init__(self, rate_hz: float, echo: bool = False) -> None:
        super().__init__("fake_hand_state_pub")
        self.echo = echo
        self.rate_hz = rate_hz
        self.js_pub = self.create_publisher(JointState, "/dg5f_right/joint_states", 10)
        self.ct_pub = self.create_publisher(Float64MultiArray, "/dg5f_right/contact_forces", 10)
        # echo 모드: 정책 손 명령(canonical 20D, sign +1 → source 동일 순서)을 상태로 반사
        self._last_cmd: list[float] = list(HAND_APPROACH_POSE)
        self._prev_pub: list[float] = list(HAND_APPROACH_POSE)
        if echo:
            self.create_subscription(
                Float64MultiArray, "/isaacsim/right_hand_cmd", self._cmd_cb, 10
            )
        self.create_timer(1.0 / rate_hz, self._tick)
        mode = "echo(/isaacsim/right_hand_cmd 반사)" if echo else "APPROACH 정적"
        self.get_logger().info(
            f"fake 손 상태 발행[{mode}]: /dg5f_right/joint_states, "
            f"/dg5f_right/contact_forces (0), {rate_hz:g}Hz\n"
            "  ⚠️ 손 분리 상태 플러밍 검증용 — 실제 손 구동 아님"
        )

    def _cmd_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= len(HAND_SOURCE):
            self._last_cmd = list(msg.data[: len(HAND_SOURCE)])

    def _tick(self) -> None:
        pos = self._last_cmd if self.echo else list(HAND_APPROACH_POSE)
        vel = [(p - q) * self.rate_hz for p, q in zip(pos, self._prev_pub)]
        self._prev_pub = list(pos)

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(HAND_SOURCE)
        js.position = list(pos)
        js.velocity = vel
        self.js_pub.publish(js)

        ct = Float64MultiArray()
        ct.data = [0.0] * 5
        self.ct_pub.publish(ct)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument(
        "--echo", action="store_true", default=False,
        help="정책 손 명령(/isaacsim/right_hand_cmd)을 관절상태로 반사 — 진화하는 손 obs 재현",
    )
    args = parser.parse_args()
    rclpy.init()
    node = FakeHandState(args.rate, echo=args.echo)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
