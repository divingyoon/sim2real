#!/usr/bin/env python3
"""FD++ base 프레임 pose(/cup_pose·/shaker_pose)를 UDP 로 중계한다.

수신자: probe_bimanual_closedloop --live-follow <port> (spawn-only 실시간 추종).
패킷: <Bfff> = side(0=cup 우 · 1=shaker 좌), x, y, z  [base_link, m]
"""
from __future__ import annotations

import argparse
import socket
import struct

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class PoseUdpTx(Node):
    def __init__(self, dest: tuple[str, int], cup_side: int = 0,
                 shaker: bool = True) -> None:
        super().__init__("pose_udp_tx")
        self.dest = dest
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.count = [0, 0]
        # ★side 는 sim 물체를 고른다(0=우 cup_big · 1=좌 대상). 실물이 하나뿐인데
        #   두 side 로 보내면 sim 물체 둘이 같은 자리에 겹쳐 솔버가 등속으로 밀어낸다
        #   (09.02: 좌 물체가 z 2.7 m 까지 날아감). 실물 개수만큼만 보낼 것.
        self.create_subscription(PoseStamped, "/cup_pose",
                                 lambda m: self._tx(cup_side, m), 10)
        if shaker:
            self.create_subscription(PoseStamped, "/shaker_pose",
                                     lambda m: self._tx(1, m), 10)
        self.create_timer(10.0, self._report)

    def _tx(self, side: int, msg: PoseStamped) -> None:
        p = msg.pose.position
        self.sock.sendto(struct.pack("<Bfff", side, p.x, p.y, p.z), self.dest)
        self.count[side] += 1

    def _report(self) -> None:
        print(f"[tx] cup {self.count[0]} · shaker {self.count[1]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cup-side", type=int, default=0, choices=(0, 1),
                    help="컵 검출을 어느 sim 물체에 물릴지 (0=우 cup_big · 1=좌 대상)")
    ap.add_argument("--no-shaker", action="store_true",
                    help="shaker 토픽 구독 안 함 — 실물에 shaker 가 없을 때")
    ap.add_argument("--dest-ip", default="100.103.21.126")
    ap.add_argument("--dest-port", type=int, default=46011)
    args = ap.parse_args()
    rclpy.init()
    node = PoseUdpTx((args.dest_ip, args.dest_port),
                     cup_side=args.cup_side, shaker=not args.no_shaker)
    print(f"[tx] → {args.dest_ip}:{args.dest_port}", flush=True)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
