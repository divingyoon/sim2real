#!/usr/bin/env python3
"""cup_pose 발행 상태 실시간 워처 — 1초마다 rate/신선도/위치 한 줄 출력.

FP++ 컵 체인은 ①기동 후 첫 검출까지 ~50s ②YOLO 검출 손실(가림/시야밖) 시 발행이
조용히 정지한다. hz 일회성 확인으론 정지를 못 보므로, 이 워처를 상시 터미널로 띄워
발행 여부를 한눈에 본다.

    STALL 이 뜨면: 컵이 카메라 시야에 있는지 / 팔이 가리는지 확인.
    (검출 상태 원본: docker logs -f fpp_cup | grep -v track_one)

실행 (vision-3090, §0 preamble 후):
    python3 cup_pose_watch.py [--topic /cup_pose] [--stale-sec 1.0]
"""

from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class CupPoseWatch(Node):
    def __init__(self, topic: str, stale_sec: float) -> None:
        super().__init__("cup_pose_watch")
        self.stale_sec = stale_sec
        self.count = 0
        self.window: list[float] = []       # 최근 수신 시각(rate 계산용)
        self.last_pos: tuple[float, float, float] | None = None
        self.last_rx: float | None = None
        self.create_subscription(PoseStamped, topic, self._cb, 10)
        self.create_timer(1.0, self._report)
        self.get_logger().info(f"watch 시작: {topic} (stale 판정 {stale_sec}s)")

    def _cb(self, msg: PoseStamped) -> None:
        now = time.monotonic()
        self.count += 1
        self.last_rx = now
        p = msg.pose.position
        self.last_pos = (p.x, p.y, p.z)
        self.window.append(now)
        cutoff = now - 3.0
        self.window = [t for t in self.window if t > cutoff]

    def _report(self) -> None:
        now = time.monotonic()
        if self.last_rx is None or self.last_pos is None:
            print("⏳ 수신 대기 (기동 후 첫 검출까지 ~50s 정상)", flush=True)
            return
        age = now - self.last_rx
        rate = len(self.window) / 3.0
        x, y, z = self.last_pos
        if age > self.stale_sec:
            print(f"🔴 STALL {age:5.1f}s 무발행  (마지막 pos=[{x:+.3f},{y:+.3f},{z:+.3f}], "
                  f"총 {self.count}) — 컵 가림/시야 확인", flush=True)
        else:
            print(f"🟢 {rate:4.1f}Hz  age {age:4.2f}s  pos=[{x:+.3f},{y:+.3f},{z:+.3f}]  "
                  f"총 {self.count}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/cup_pose")
    ap.add_argument("--stale-sec", type=float, default=1.0)
    args = ap.parse_args()

    rclpy.init()
    node = CupPoseWatch(args.topic, args.stale_sec)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass   # SIGTERM 등으로 이미 shutdown 된 경우 무해


if __name__ == "__main__":
    main()
