#!/usr/bin/env python3
"""Isaac 라이브 스트림(UDP) → /isaacsim/left_arm_cmd·left_gripper_cmd 발행 어댑터.

왜 이 모양인가: Isaac 파이썬(3.11)엔 rclpy 가 없다. Isaac 은 ROS 를 모른 채 매 정책
스텝의 관절 목표를 UDP 로 쏘고(probe_v2_shadow_record.py --stream_udp), 이 노드가 받아
기존 명령 채널로 넘긴다. 그 뒤는 재생과 **같은 사슬**(isaacsim_cmd_to_jtc 브리지)이라
재생에서 검증된 rate-limit·time_from_start=0 규약을 그대로 탄다.

★그리퍼는 브리지가 Float64MultiArray 를 구독하므로(프로필 주석의 Float64 스칼라와
  다르다 — isaacsim_cmd_to_jtc.py:95 실측) [1] 배열로 보낸다.
★`--execute` 없으면 아무것도 발행하지 않는다(robotctl 규약).
★수신 CSV(t_recv·step·targets)를 남긴다 — shadow_report 의 정렬 키.
"""
from __future__ import annotations

import argparse
import csv
import socket
import struct
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

MAGIC = 0x5A2B01
FMT = "<Id8f"          # v1: magic · t_send · 실행지령 arm7+grip1
SIZE = struct.calcsize(FMT)
# v2: 실행지령(8) + action(7) + sim지령(7) + sim실측(7) — /shadow/* 발행용
MAGIC2 = 0x5A2B02
FMT2 = "<Id29f"
SIZE2 = struct.calcsize(FMT2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=47311)
    ap.add_argument("--arm-topic", default="/isaacsim/left_arm_cmd")
    ap.add_argument("--gripper-topic", default="/isaacsim/left_gripper_cmd")
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--execute", action="store_true",
                    help="없으면 수신·기록만 하고 발행하지 않는다")
    args = ap.parse_args()

    rclpy.init()
    node = Node("udp_cmd_to_ros")
    arm_pub = node.create_publisher(Float64MultiArray, args.arm_topic, 10)
    grip_pub = node.create_publisher(Float64MultiArray, args.gripper_topic, 10)
    # v2 3종 — bag 하나로 ACTION/SIM/REAL 정렬 데이터셋을 만들기 위한 발행
    act_pub = node.create_publisher(Float64MultiArray, "/shadow/action", 10)
    tgt_pub = node.create_publisher(Float64MultiArray, "/shadow/sim_target", 10)
    ms_pub = node.create_publisher(Float64MultiArray, "/shadow/sim_meas", 10)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    sock.settimeout(0.5)
    node.get_logger().info(
        f"UDP :{args.port} → {args.arm_topic} + {args.gripper_topic} "
        f"({'실발행' if args.execute else 'DRY — 발행 안 함'})")

    writer = None
    fh = None
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(args.log, "w", newline="", buffering=1)  # 라인버퍼 — 강제종료에도 행 손실 없음
        writer = csv.writer(fh)
        writer.writerow(["t_recv", "t_send", "step_idx",
                         *[f"arm_target_{i}" for i in range(7)], "grip_cmd"])

    n = 0
    last_report = time.monotonic()
    try:
        while rclpy.ok():
            try:
                data, _ = sock.recvfrom(SIZE2)
            except socket.timeout:
                continue
            if len(data) == SIZE2:
                vals = struct.unpack(FMT2, data)
                if vals[0] != MAGIC2:
                    continue
                t_send, payload = vals[1], vals[2:]
                arm, grip = payload[:7], payload[7]
                if args.execute:
                    for pub, seg in ((act_pub, payload[8:15]),
                                     (tgt_pub, payload[15:22]),
                                     (ms_pub, payload[22:29])):
                        mm = Float64MultiArray(); mm.data = list(seg); pub.publish(mm)
            elif len(data) == SIZE:
                vals = struct.unpack(FMT, data)
                if vals[0] != MAGIC:
                    continue
                t_send, payload = vals[1], vals[2:]
                arm, grip = payload[:7], payload[7]
            else:
                continue
            if args.execute:
                m = Float64MultiArray(); m.data = list(arm); arm_pub.publish(m)
                g = Float64MultiArray(); g.data = [float(grip)]; grip_pub.publish(g)
            if writer is not None:
                writer.writerow([f"{time.time():.6f}", f"{t_send:.6f}", n,
                                 *[f"{v:.6f}" for v in arm], f"{grip:.6f}"])
            n += 1
            now = time.monotonic()
            if now - last_report >= 5.0:
                node.get_logger().info(f"수신 {n} 프레임 (~{n and 1/((now-last_report)/ (n or 1)):.0f})")
                last_report = now
    except KeyboardInterrupt:
        pass
    finally:
        if fh:
            fh.close()
        node.get_logger().info(f"종료 — 총 {n} 프레임")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
