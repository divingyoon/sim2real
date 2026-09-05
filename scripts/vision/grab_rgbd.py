#!/usr/bin/env python3
"""정렬된 RGB + depth + K 를 npz 한 장으로 저장한다 (테이블 평면 적합 등 오프라인 검증용).

  python3 grab_rgbd.py --out /tmp/rgbd.npz
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class Grab(Node):
    def __init__(self) -> None:
        super().__init__("grab_rgbd")
        self.rgb = self.depth = self.k = None
        self.create_subscription(Image, "/camera/camera/color/image_raw",
                                 self._rgb, qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw",
                                 self._depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/camera/color/camera_info",
                                 self._info, qos_profile_sensor_data)

    def _rgb(self, msg: Image) -> None:
        self.rgb = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()

    def _depth(self, msg: Image) -> None:
        if msg.encoding != "16UC1":
            raise SystemExit(f"depth encoding {msg.encoding} 미지원")
        d = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width)
        self.depth = d.astype(np.float32) * 1e-3

    def _info(self, msg: CameraInfo) -> None:
        self.k = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)

    def ready(self) -> bool:
        return self.rgb is not None and self.depth is not None and self.k is not None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()
    rclpy.init()
    node = Grab()
    deadline = time.monotonic() + args.timeout
    while not node.ready() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    if not node.ready():
        raise SystemExit("rgb/depth/camera_info 를 다 못 받았다")
    np.savez_compressed(args.out, rgb=node.rgb, depth=node.depth, K=node.k)
    print(f"saved {args.out} rgb={node.rgb.shape} depth valid={float((node.depth > 0).mean()):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
