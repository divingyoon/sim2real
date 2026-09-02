#!/usr/bin/env python3
"""러너 지령 35f UDP → /isaacsim/*_cmd 4토픽. Step 3 실기 사슬의 수신단.

  probe_bimanual_closedloop --stream ──▶ 여기 ──▶ /isaacsim/{left_arm,left_gripper,
  right_arm,right_hand}_cmd ──▶ isaacsim_cmd_to_jtc(좌·우 2인스턴스) ──▶ JTC

패킷 `<Id35f>` magic 0x5A2B11: 좌팔7 + 우팔7 + 좌그립1 + 우손20. NaN 채널은
발행하지 않는다(실기 홀드 = 마지막 목표 유지, 지어내지 않는다).
`--execute` 없으면 수신 통계만 찍는 DRY.
"""
from __future__ import annotations

import argparse
import math
import socket
import struct
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

MAGIC = 0x5A2B11
FMT = "<Id35f"
SIZE = struct.calcsize(FMT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=47331)
    ap.add_argument("--execute", action="store_true",
                    help="없으면 수신·통계만 (DRY — 발행 안 함)")
    args = ap.parse_args()
    rclpy.init()
    node = Node("udp35_to_ros_cmd")
    pubs = {
        "la": node.create_publisher(Float64MultiArray, "/isaacsim/left_arm_cmd", 10),
        "lg": node.create_publisher(Float64MultiArray, "/isaacsim/left_gripper_cmd", 10),
        "ra": node.create_publisher(Float64MultiArray, "/isaacsim/right_arm_cmd", 10),
        "rh": node.create_publisher(Float64MultiArray, "/isaacsim/right_hand_cmd", 10),
    }
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(0.5)
    node.get_logger().info(
        f"UDP :{args.port} → /isaacsim/*_cmd ({'실발행' if args.execute else 'DRY'})")
    n, live, last = 0, set(), time.monotonic()
    while rclpy.ok():
        try:
            data, _ = sock.recvfrom(SIZE)
        except socket.timeout:
            continue
        if len(data) != SIZE:
            continue
        v = struct.unpack(FMT, data)
        if v[0] != MAGIC:
            continue
        pay = v[2:]
        for key, seg in (("la", pay[0:7]), ("ra", pay[7:14]),
                         ("lg", pay[14:15]), ("rh", pay[15:35])):
            if math.isnan(seg[0]):
                continue
            live.add(key)
            if args.execute:
                m = Float64MultiArray()
                m.data = [float(x) for x in seg]
                pubs[key].publish(m)
        n += 1
        now = time.monotonic()
        if now - last >= 5.0:
            node.get_logger().info(f"수신 {n} · 활성 채널 {sorted(live)}")
            last = now


if __name__ == "__main__":
    main()
