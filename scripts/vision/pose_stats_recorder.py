#!/usr/bin/env python3
"""FP++ pose 토픽을 N 초 받아 위치 통계·상하 반전·추적 상태를 요약한다 (정지 물체 기준).

용도: shaker CAD 교체 전/후 A/B (2026-09-03). 물체를 가만히 두고 돌리므로
위치 표준편차 = 인지 떨림, 물체 z축(카메라 프레임)의 부호 전환 = 상하 반전이다.

  python3 pose_stats_recorder.py --topic /perception_plus_plus/shaker/pose \
      --status /perception_plus_plus/shaker/tracking_status --seconds 30 --out /tmp/a.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


def quat_to_zaxis(w: float, x: float, y: float, z: float) -> np.ndarray:
    """회전 행렬 3열 = 물체 z축을 부모 프레임에서 본 방향."""
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


class Recorder(Node):
    def __init__(self, topic: str, status_topic: str | None) -> None:
        super().__init__("pose_stats_recorder")
        self.samples: list[tuple[float, float, float, float, float, float, float, float]] = []
        self.states: dict[int, int] = {}
        self.reasons: dict[str, int] = {}
        self.create_subscription(PoseStamped, topic, self._on_pose, qos_profile_sensor_data)
        if status_topic:
            try:
                from perception_plus_plus_msgs.msg import TrackingStatus
            except ImportError:
                self.get_logger().warning("perception_plus_plus_msgs 없음 — status 집계 생략")
            else:
                self.create_subscription(TrackingStatus, status_topic, self._on_status, 10)

    def _on_pose(self, msg: PoseStamped) -> None:
        p, q = msg.pose.position, msg.pose.orientation
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.samples.append((stamp, p.x, p.y, p.z, q.w, q.x, q.y, q.z))

    def _on_status(self, msg) -> None:
        self.states[int(msg.state)] = self.states.get(int(msg.state), 0) + 1
        if msg.failure_reason:
            self.reasons[msg.failure_reason] = self.reasons.get(msg.failure_reason, 0) + 1


def summarize(samples: list, states: dict, reasons: dict, seconds: float) -> dict:
    if not samples:
        return {"n": 0, "seconds": seconds, "states": states, "reasons": reasons}
    arr = np.asarray(samples, dtype=float)
    pos = arr[:, 1:4]
    zaxis = np.stack([quat_to_zaxis(*row) for row in arr[:, 4:8]])
    # 카메라 optical 프레임에서 "위"는 −y. 물체 z축이 −y 를 향하면 정립.
    upright = zaxis[:, 1] < 0
    flips = int(np.sum(upright[1:] != upright[:-1]))
    return {
        "n": int(len(arr)), "seconds": seconds, "hz": float(len(arr) / max(seconds, 1e-6)),
        "mean_xyz": pos.mean(0).round(4).tolist(),
        "std_xyz_mm": (pos.std(0) * 1000).round(1).tolist(),
        "p2p_xyz_mm": ((pos.max(0) - pos.min(0)) * 1000).round(1).tolist(),
        "upright_ratio": float(upright.mean()), "flips": flips,
        "zaxis_mean": zaxis.mean(0).round(3).tolist(),
        "states": states, "reasons": reasons,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", default="/perception_plus_plus/shaker/pose")
    ap.add_argument("--status", default=None)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="원시 샘플 npy 저장 경로")
    args = ap.parse_args()
    rclpy.init()
    node = Recorder(args.topic, args.status)
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    report = summarize(node.samples, node.states, node.reasons, args.seconds)
    node.destroy_node()
    rclpy.shutdown()
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)
    if args.dump and node.samples:
        np.save(args.dump, np.asarray(node.samples, dtype=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
