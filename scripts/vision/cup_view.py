#!/usr/bin/env python3
"""RGB 라이브 뷰 + FP++ 컵 추적 오버레이 (vision-3090 GUI 터미널 전용).

카메라 RGB 위에 FP++ 컵 위치(카메라 프레임 pose 를 intrinsics 로 투영)를 그려
"영상에 실시간으로 추적이 보이는" 확인 창을 띄운다. cv_bridge 무의존(rgb8 수동 디코드).

    창 표시:  🟢 추적 중(원+십자선+base 좌표) / 🔴 pose 두절(마지막 위치 회색+STALL 초)
    종료:     창에서 q 또는 Ctrl-C

실행 (vision-3090 모니터 붙은 GUI 터미널, §0 preamble 후):
    python3 cup_view.py
"""

from __future__ import annotations

import time

import numpy as np
import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped

STALE_SEC = 1.0


class CupView(Node):
    def __init__(self) -> None:
        super().__init__("cup_view")
        self.K: np.ndarray | None = None
        self.cam_pose: tuple[float, float, float] | None = None   # 카메라 프레임 컵 위치
        self.base_pose: tuple[float, float, float] | None = None  # base 프레임 (표시용)
        self.last_pose_rx: float | None = None
        self.frame: np.ndarray | None = None
        self.create_subscription(Image, "/camera/camera/color/image_raw", self._img_cb, 2)
        self.create_subscription(CameraInfo, "/camera/camera/color/camera_info", self._info_cb, 2)
        self.create_subscription(
            PoseStamped, "/perception_plus_plus/cup/pose", self._cam_pose_cb, 10)
        self.create_subscription(PoseStamped, "/cup_pose", self._base_pose_cb, 10)

    def _info_cb(self, msg: CameraInfo) -> None:
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _cam_pose_cb(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self.cam_pose = (p.x, p.y, p.z)
        self.last_pose_rx = time.monotonic()

    def _base_pose_cb(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self.base_pose = (p.x, p.y, p.z)

    def _img_cb(self, msg: Image) -> None:
        if msg.encoding not in ("rgb8", "bgr8"):
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.frame = img[..., ::-1].copy() if msg.encoding == "rgb8" else img.copy()

    def draw(self) -> np.ndarray | None:
        if self.frame is None:
            return None
        img = self.frame
        now = time.monotonic()
        stale = self.last_pose_rx is None or now - self.last_pose_rx > STALE_SEC
        if self.cam_pose is not None and self.K is not None and self.cam_pose[2] > 0.05:
            x, y, z = self.cam_pose
            u = int(self.K[0, 0] * x / z + self.K[0, 2])
            v = int(self.K[1, 1] * y / z + self.K[1, 2])
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                color = (128, 128, 128) if stale else (0, 255, 0)
                cv2.circle(img, (u, v), 18, color, 2)
                cv2.drawMarker(img, (u, v), color, cv2.MARKER_CROSS, 30, 2)
        if stale:
            age = 0.0 if self.last_pose_rx is None else now - self.last_pose_rx
            txt = f"STALL {age:.0f}s — cup pose 두절 (가림/시야 확인)"
            cv2.putText(img, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        elif self.base_pose is not None:
            bx, by, bz = self.base_pose
            cv2.putText(img, f"cup(base) [{bx:+.3f} {by:+.3f} {bz:+.3f}]",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return img


def main() -> None:
    rclpy.init()
    node = CupView()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            img = node.draw()
            if img is not None:
                cv2.imshow("cup_view (q=quit)", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
