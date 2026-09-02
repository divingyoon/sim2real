#!/usr/bin/env python3
"""컬러 이미지 1장을 PNG 로 저장한다 (장면·마스크 육안 확인용).

  python3 grab_frame.py --out /tmp/frame.png [--topic /camera/camera/color/image_raw]
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class Grab(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("grab_frame")
        self.img = None
        self.create_subscription(Image, topic, self._cb, qos_profile_sensor_data)

    def _cb(self, msg: Image) -> None:
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        self.img = arr[:, :, ::-1].copy() if msg.encoding == "rgb8" else arr.copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/camera/camera/color/image_raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()
    rclpy.init()
    node = Grab(args.topic)
    deadline = time.monotonic() + args.timeout
    while node.img is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    if node.img is None:
        raise SystemExit(f"{args.topic} 에서 이미지를 못 받았다")
    cv2.imwrite(args.out, node.img)
    print(f"saved {args.out} {node.img.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
