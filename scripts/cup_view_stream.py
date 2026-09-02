#!/usr/bin/env python3
"""RGB 라이브 + FP++ 2물체(cup·shaker) 오버레이를 MJPEG 로 내보낸다.

`cup_view.py`(imshow — 모니터 필요)와 `stream_head_view.py`(카메라 직접 열기 —
ROS 노드와 점유 충돌)의 하이브리드: **ROS 토픽을 구독**하므로 카메라 점유가 없고,
MJPEG 라 local5090 브라우저에서 본다 (head_view_up.sh 의 8080 터널 재사용).

    # vision-3090
    source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=126
    python3 cup_view_stream.py --port 8080
    # local5090 브라우저: http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

STALE_SEC = 1.5
#: 메시 AABB (m, 오브젝트 프레임) — 3D bounding box 투영용 (mesh 실측 09.02)
AABB = {
    "cup": ((-0.0463, -0.0773, -0.044), (0.0437, 0.1003, 0.046)),
    "shaker": ((-0.046, -0.046, 0.0), (0.046, 0.046, 0.238)),
}
#: (라벨, 카메라프레임 pose 토픽, base 프레임 relay 토픽, BGR 색)
OBJECTS = (
    ("cup", "/perception_plus_plus/cup/pose", "/cup_pose", (0, 0, 255)),
    ("shaker", "/perception_plus_plus/shaker/pose", "/shaker_pose", (255, 200, 0)),
)
_BOX_EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
              (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))


def _quat_to_rot(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def draw_box(bgr, K, pose, aabb, color) -> None:
    lo, hi = aabb
    corners = np.array([[lo[0] if not i & 1 else hi[0],
                         lo[1] if not i & 2 else hi[1],
                         lo[2] if not i & 4 else hi[2]] for i in range(8)])
    R = _quat_to_rot(pose.orientation)
    t = np.array([pose.position.x, pose.position.y, pose.position.z])
    cam = corners @ R.T + t
    if (cam[:, 2] < 0.05).any():
        return
    uv = (cam / cam[:, 2:3]) @ K.T
    pts = uv[:, :2].astype(int)
    for a, b in _BOX_EDGES:
        cv2.line(bgr, tuple(pts[a]), tuple(pts[b]), color, 2)


class View(Node):
    def __init__(self, compressed: bool = False) -> None:
        super().__init__("cup_view_stream")
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.K: np.ndarray | None = None
        self.cam: dict[str, tuple[PoseStamped, float]] = {}
        self.base: dict[str, tuple[PoseStamped, float]] = {}
        # ★비압축 RGB 는 1280×720×3 = 2.7 MB/frame · 28 Hz. 같은 호스트라도 기본
        #   FastDDS 에서 통째로 유실된다("A message was lost!!!") — 컨테이너는
        #   --ipc=host 공유메모리라 멀쩡하고 호스트 프로세스만 검은 화면이 됐다
        #   (09.02 실측). compressed(~100 KB)면 그 경로를 피한다.
        if compressed:
            self.create_subscription(
                CompressedImage, "/camera/camera/color/image_raw/compressed",
                self._img_compressed, 5)
        else:
            self.create_subscription(
                Image, "/camera/camera/color/image_raw", self._img, 5)
        self.create_subscription(
            CameraInfo, "/camera/camera/color/camera_info", self._info, 5)
        for name, cam_t, base_t, _ in OBJECTS:
            self.create_subscription(
                PoseStamped, cam_t,
                lambda m, n=name: self._pose(self.cam, n, m), 10)
            self.create_subscription(
                PoseStamped, base_t,
                lambda m, n=name: self._pose(self.base, n, m), 10)

    def _info(self, msg: CameraInfo) -> None:
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _pose(self, store: dict, name: str, msg: PoseStamped) -> None:
        store[name] = (msg, time.monotonic())

    def _img_compressed(self, msg: CompressedImage) -> None:
        bgr = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn("compressed 프레임 디코드 실패",
                                   throttle_duration_sec=5.0)
            return
        self._draw(bgr)

    def _img(self, msg: Image) -> None:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self._draw(cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy())

    def _draw(self, bgr: np.ndarray) -> None:
        now = time.monotonic()
        y0 = 26
        for name, _, _, color in OBJECTS:
            cam = self.cam.get(name)
            base = self.base.get(name)
            fresh = cam is not None and now - cam[1] < STALE_SEC
            if fresh and self.K is not None:
                p = cam[0].pose.position
                if p.z > 0.05:
                    uv = self.K @ np.array([p.x / p.z, p.y / p.z, 1.0])
                    u, v = int(uv[0]), int(uv[1])
                    cv2.drawMarker(bgr, (u, v), color, cv2.MARKER_CROSS, 20, 2)
                    draw_box(bgr, self.K, cam[0].pose, AABB[name], color)
            if base is not None and now - base[1] < STALE_SEC:
                q = base[0].pose.position
                txt = f"{name}: base ({q.x:+.3f}, {q.y:+.3f}, {q.z:+.3f})"
            elif fresh:
                txt = f"{name}: cam OK / base pending"
            else:
                txt = f"{name}: no detection"
            cv2.putText(bgr, txt, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        color if fresh else (128, 128, 128), 2)
            y0 += 26
        with self.lock:
            self.frame = bgr


def make_handler(view: View):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            return

        def do_GET(self):
            if self.path != "/stream":
                body = (b"<title>head view</title><body style=\"margin:0;background:#111\">"
                        b"<img src=/stream style=\"width:100vw\"></body>")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    with view.lock:
                        frame = None if view.frame is None else view.frame.copy()
                    if frame is not None:
                        ok, jpg = cv2.imencode(".jpg", frame,
                                               [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok:
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(jpg.tobytes())
                            self.wfile.write(b"\r\n")
                    time.sleep(0.08)
            except (BrokenPipeError, ConnectionResetError):
                pass
    return H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--show", action="store_true", help="로컬 모니터에도 창 표시 (DISPLAY 필요)")
    ap.add_argument("--compressed", action="store_true",
                    help="압축 이미지 토픽 구독 — 호스트에서 비압축이 DDS 유실될 때")
    args = ap.parse_args()
    rclpy.init()
    view = View(compressed=args.compressed)
    threading.Thread(target=rclpy.spin, args=(view,), daemon=True).start()
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(view))
    print(f"MJPEG http://{args.bind}:{args.port} — 2물체 오버레이", flush=True)
    if not args.show:
        server.serve_forever()
        return
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cv2.namedWindow("head view", cv2.WINDOW_NORMAL)
    while True:
        with view.lock:
            frame = None if view.frame is None else view.frame.copy()
        if frame is not None:
            cv2.imshow("head view", frame)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
