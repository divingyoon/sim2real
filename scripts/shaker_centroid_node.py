#!/usr/bin/env python3
"""shaker 경량 위치 노드 — FoundationPose 없이 YOLO 마스크 중심+깊이로 중심 3D.

FP++ 가 무광 검은 원통에서 등록 복불복(z −3.5~−8.8cm 편향, 2026-09-02 실측)을
벗어나지 못해 도입. 자세는 항등(학습 관측은 중심 pos 만 사용). 등록/추적 상태가
없어 프레임마다 독립 — 복구 개념 자체가 필요 없다.

발행: /perception_plus_plus/shaker/pose (camera optical frame, **물체 중심**)
→ relay 는 cad_to_body 0 으로 통과시켜야 한다 (중복 +0.119 금지).
"""
from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

import message_filters

CLASS_ID = 41          # COCO cup — 검은 shaker 가 cup 으로 검출된다 (실측)
CONF = 0.35
RADIUS = 0.046         # shaker 반경 (m) — 표면점→축 보정
DEPTH_PCT = 30         # 마스크 깊이 하위 백분위 — 열린 윗면의 내부 깊이 오염 회피
RAY_CORR = 0.7 * RADIUS  # 전면 표면 중앙값 ≈ 축距 − 0.7r
MIN_BLUENESS = 15.0    # 파랑 우세(B−max(R,G)) 문턱 — 파란 shaker 전용(09.02 색 변경).
                       # 파랗지 않은 마스크(빨간 컵 등)는 발행하지 않는다
EVERY_N = 2


class ShakerCentroid(Node):
    def __init__(self) -> None:
        super().__init__("shaker_centroid")
        from ultralytics import YOLO
        self.model = YOLO("/workspace/perception_plus_plus/models/yolo/yolov8m-seg.pt")
        self.pub = self.create_publisher(
            PoseStamped, "/perception_plus_plus/shaker/pose", 10)
        self.n = 0
        rgb = message_filters.Subscriber(self, Image, "/camera/camera/color/image_raw")
        depth = message_filters.Subscriber(
            self, Image, "/camera/camera/aligned_depth_to_color/image_raw")
        info = message_filters.Subscriber(
            self, CameraInfo, "/camera/camera/color/camera_info")
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb, depth, info], 10, 0.04)
        self.sync.registerCallback(self._cb)

    def _cb(self, rgb_msg: Image, depth_msg: Image, info: CameraInfo) -> None:
        self.n += 1
        if self.n % EVERY_N:
            return
        rgb = np.frombuffer(rgb_msg.data, np.uint8).reshape(
            rgb_msg.height, rgb_msg.width, 3)
        result = self.model(rgb, conf=CONF, verbose=False)[0]
        if result.boxes is None or result.masks is None:
            return
        classes = result.boxes.cls.cpu().numpy().astype(int)
        masks = result.masks.data.cpu().numpy()
        best, best_score = None, -1e9
        import cv2
        for i, cid in enumerate(classes):
            if cid != CLASS_ID:
                continue
            m = masks[i].astype(np.uint8)
            if m.shape != rgb.shape[:2]:
                m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            mb = m.astype(bool)
            if not mb.any():
                continue
            m3 = rgb[mb].astype(np.float32)
            score = float((m3[:, 2] - np.maximum(m3[:, 0], m3[:, 1])).mean())
            if score > best_score and score > MIN_BLUENESS:
                best, best_score = mb, score
        if best is None:
            return
        # 경계 깊이 번짐 제거
        core = cv2.erode(best.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        if core.sum() < 50:
            core = best
        depth = np.frombuffer(depth_msg.data, np.uint16).reshape(
            depth_msg.height, depth_msg.width).astype(np.float32) / 1000.0
        dvals = depth[core]
        dvals = dvals[(dvals > 0.15) & (dvals < 2.0)]
        if dvals.size < 30:
            return
        d = float(np.percentile(dvals, DEPTH_PCT)) + RAY_CORR
        ys, xs = np.nonzero(best)
        u, v = float(xs.mean()), float(ys.mean())
        k = info.k
        ray = np.array([(u - k[2]) / k[0], (v - k[5]) / k[4], 1.0])
        ray /= np.linalg.norm(ray)
        p = ray * d
        msg = PoseStamped()
        msg.header = rgb_msg.header
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (
            float(p[0]), float(p[1]), float(p[2]))
        msg.pose.orientation.w = 1.0
        self.pub.publish(msg)


def main() -> None:
    rclpy.init()
    rclpy.spin(ShakerCentroid())


if __name__ == "__main__":
    main()
