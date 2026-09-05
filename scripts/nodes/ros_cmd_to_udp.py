#!/usr/bin/env python3
"""ROS `/isaacsim/*_cmd` → UDP — Isaac 미러(sim follower)에게 지령을 흘려보낸다.

`udp_cmd_to_ros.py`(Isaac→ROS)의 **역방향**이다. Isaac(py3.11)에는 rclpy 가 없으므로
ROS 쪽 이 노드가 지령 토픽을 모아 UDP 로 내보내고, `probe_sim_follower.py` 가 받아
sim 로봇에 적용한다. 그러면 RViz 없이 Isaac GUI 가 "지령이 만드는 움직임"을 보여준다
— 같은 토픽이 실기 JTC 로도 가므로 sim 과 실기가 **같은 지령**을 받는다.

패킷 v1: `<Id35f>` = magic 0x5A2B10 · 발신시각 · 좌팔7 + 우팔7 + 좌그리퍼1 + 우손20.
  아직 안 온 채널은 NaN — follower 는 NaN 채널을 **건드리지 않는다**(현재 유지).
  0 으로 채우면 "차렷으로 가라"는 지령을 지어내는 것이 된다.

발행 주기 고정 50 Hz(수신 시점 전달 아님 — 토픽이 조용하면 마지막 값을 재전송해
follower 가 홀드를 유지한다).
"""

from __future__ import annotations

import argparse
import math
import socket
import struct
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray

MAGIC = 0x5A2B10
FMT = "<Id35f"
SEND_HZ = 50.0


class RosCmdToUdp(Node):
    def __init__(self, host: str, port: int) -> None:
        super().__init__("ros_cmd_to_udp")
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        nan = float("nan")
        self._left = [nan] * 7
        self._right = [nan] * 7
        self._grip = [nan]
        self._hand = [nan] * 20
        self.create_subscription(Float64MultiArray, "/isaacsim/left_arm_cmd",
                                 self._mk_arm(self._left), 10)
        self.create_subscription(Float64MultiArray, "/isaacsim/right_arm_cmd",
                                 self._mk_arm(self._right), 10)
        self.create_subscription(Float64, "/isaacsim/left_gripper_cmd",
                                 self._grip_cb, 10)
        self.create_subscription(Float64MultiArray, "/isaacsim/right_hand_cmd",
                                 self._mk_arm(self._hand), 10)
        self._n_sent = 0
        self.create_timer(1.0 / SEND_HZ, self._tick)
        self.get_logger().info(f"→ UDP {host}:{port} @ {SEND_HZ:.0f} Hz "
                               "(수신 전 채널은 NaN=현재유지)")

    def _mk_arm(self, buf: list) -> "callable":
        def cb(msg: Float64MultiArray) -> None:
            data = list(msg.data)
            if len(data) != len(buf):
                self.get_logger().warning(
                    f"차원 불일치: 기대 {len(buf)} 수신 {len(data)} — 무시")
                return
            buf[:] = [float(v) for v in data]
        return cb

    def _grip_cb(self, msg: Float64) -> None:
        self._grip[0] = float(msg.data)

    def _tick(self) -> None:
        payload = self._left + self._right + self._grip + self._hand
        self._sock.sendto(struct.pack(FMT, MAGIC, time.time(), *payload), self._addr)
        self._n_sent += 1
        if self._n_sent % (int(SEND_HZ) * 10) == 0:
            live = [n for n, b in (("L팔", self._left), ("R팔", self._right),
                                   ("L그립", self._grip), ("R손", self._hand))
                    if not math.isnan(b[0])]
            self.get_logger().info(f"{self._n_sent} 패킷 · 살아있는 채널 {live}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=47321)
    args = parser.parse_args()
    rclpy.init()
    node = RosCmdToUdp(args.host, args.port)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
