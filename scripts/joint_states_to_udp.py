#!/usr/bin/env python3
"""실기 JointState → 실측 35f UDP. 러너 --echo-port 가 받아 GUI=실기 미러가 된다.

이름 기반 추출이라 토픽 구성이 어떻든 동작한다: l_aj_1..7 / r_aj_1..7 /
l_hj_gripper_1 / rj_dg_1_1..5_4 만 골라 담고, 한 번도 안 온 관절은 NaN(러너가
해당 채널을 건드리지 않는다). 50 Hz 재전송(홀드 유지).
"""
from __future__ import annotations

import argparse
import socket
import struct
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

MAGIC = 0x5A2B12
FMT = "<Id35f"
# 35슬롯 canonical 배치: 좌팔7 · 우팔7 · 좌그립1 · 우손20(thumb-major).
# 실기 /joint_states 는 source 명(openarm_*_joint, rj_dg_*)을 쓰므로 두 이름
# 체계를 같은 슬롯으로 병합한다 (manifest source_to_canonical 근거, 09.02).
IDX: dict[str, int] = {}
for i in range(7):
    IDX[f"l_aj_{i + 1}"] = i
    IDX[f"openarm_left_joint{i + 1}"] = i
    IDX[f"r_aj_{i + 1}"] = 7 + i
    IDX[f"openarm_right_joint{i + 1}"] = 7 + i
IDX["l_hj_gripper_1"] = 14
IDX["openarm_left_finger_joint1"] = 14
_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
for f in range(5):
    for j in range(4):
        slot = 15 + f * 4 + j
        IDX[f"rj_dg_{f + 1}_{j + 1}"] = slot          # 실기 source (dg_1=thumb)
        IDX[f"r_hj_{_FINGERS[f]}_{j + 1}"] = slot     # canonical


class StatesToUdp(Node):
    def __init__(self, dest: tuple, topics: list[str]) -> None:
        super().__init__("joint_states_to_udp")
        self.dest = dest
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.buf = [float("nan")] * 35
        for t in topics:
            self.create_subscription(JointState, t, self._cb, 20)
        self.create_timer(1.0 / 50.0, self._tick)
        self.n = 0
        self.get_logger().info(f"{topics} → UDP {dest} @50Hz")

    def _cb(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            i = IDX.get(name)
            if i is not None:
                self.buf[i] = float(pos)

    def _tick(self) -> None:
        self.sock.sendto(struct.pack(FMT, MAGIC, time.time(), *self.buf), self.dest)
        self.n += 1
        if self.n % 500 == 0:
            import math
            alive = sum(0 if math.isnan(v) else 1 for v in self.buf)
            self.get_logger().info(f"{self.n} 패킷 · 유효 관절 {alive}/35")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest-ip", default="127.0.0.1")
    ap.add_argument("--dest-port", type=int, default=47332)
    ap.add_argument("--topics", default="/joint_states",
                    help="쉼표 구분 JointState 토픽들 (이름 기반 병합)")
    args = ap.parse_args()
    rclpy.init()
    rclpy.spin(StatesToUdp((args.dest_ip, args.dest_port),
                           [t.strip() for t in args.topics.split(",") if t.strip()]))


if __name__ == "__main__":
    main()
