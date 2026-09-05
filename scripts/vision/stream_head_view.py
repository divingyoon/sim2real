#!/usr/bin/env python3
"""head RealSense 를 MJPEG 로 내보낸다 — 손으로 자세를 맞추는 동안 보기 위한 것.

**vision-3090 에서 실행한다** (RealSense 가 거기 붙어 있다). 모터는 local5090 이므로
화면과 손이 다른 기계에 있다. X11 포워딩은 vision-3090 sshd 가 막아 두었고, 어차피
`realsense-viewer` 는 OpenGL 앱이라 포워딩하면 느리다. MJPEG 이 더 낫다.

기본으로 **127.0.0.1 에만 바인딩**한다 — 네트워크에 노출하지 않는다.
local5090 에서 SSH 터널로 가져와 브라우저로 본다:

    # vision-3090
    ~/rl_ws/perception_plus_plus/.venv/bin/python stream_head_view.py

    # local5090
    ssh -N -L 8080:127.0.0.1:8080 vision-3090
    # 브라우저에서 http://127.0.0.1:8080

화면에 조준용 십자선과 **중앙 영역 깊이(m)** 를 겹쳐 그린다 — 카메라를 어디에
맞추고 있는지 숫자로 확인하면서 모터를 돌릴 수 있다.
"""

from __future__ import annotations

import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pyrealsense2 as rs

DEFAULT_PORT = 8080
DEFAULT_BIND = "127.0.0.1"
WIDTH, HEIGHT, FPS = 640, 480, 30
JPEG_QUALITY = 80
DEPTH_PATCH_PX = 20          # 중앙 깊이를 재는 정사각 반폭
CROSSHAIR_PX = 24
OVERLAY_COLOR = (0, 255, 0)
BOUNDARY = b"--frame"

_PAGE = b"""<!doctype html><meta charset=utf-8><title>head view</title>
<style>body{margin:0;background:#111;color:#eee;font:14px system-ui;text-align:center}
img{max-width:100%;height:auto;image-rendering:pixelated}</style>
<p>RealSense D435 &middot; head &mdash; \xec\x86\x90\xec\x9c\xbc\xeb\xa1\x9c \xeb\x8f\x8c\xeb\xa6\xac\xeb\xa9\xb0 \xeb\xb3\xb4\xec\x84\xb8\xec\x9a\x94</p>
<img src="/stream">"""


class Camera:
    """RealSense 파이프라인 하나를 여러 요청이 나눠 쓴다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        self._profile = self._pipeline.start(config)
        self._align = rs.align(rs.stream.color)
        self._scale = self._profile.get_device().first_depth_sensor().get_depth_scale()

    def close(self) -> None:
        with self._lock:
            self._pipeline.stop()

    def frame(self) -> bytes | None:
        """오버레이를 그린 JPEG 한 장. 프레임을 못 받으면 None."""
        with self._lock:
            frames = self._align.process(self._pipeline.wait_for_frames())
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color:
                return None
            image = np.asanyarray(color.get_data()).copy()
            depth_m = self._center_depth_m(depth) if depth else None

        _draw_overlay(image, depth_m)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if ok else None

    def _center_depth_m(self, depth) -> float | None:
        raw = np.asanyarray(depth.get_data())
        cy, cx = raw.shape[0] // 2, raw.shape[1] // 2
        patch = raw[cy - DEPTH_PATCH_PX:cy + DEPTH_PATCH_PX,
                    cx - DEPTH_PATCH_PX:cx + DEPTH_PATCH_PX]
        valid = patch[patch > 0]
        return float(np.median(valid)) * self._scale if valid.size else None


def _draw_overlay(image: np.ndarray, depth_m: float | None) -> None:
    """중앙 십자선과 깊이 숫자. 조준을 눈이 아니라 값으로 확인하게 한다."""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(image, (cx - CROSSHAIR_PX, cy), (cx + CROSSHAIR_PX, cy), OVERLAY_COLOR, 1)
    cv2.line(image, (cx, cy - CROSSHAIR_PX), (cx, cy + CROSSHAIR_PX), OVERLAY_COLOR, 1)
    cv2.rectangle(image, (cx - DEPTH_PATCH_PX, cy - DEPTH_PATCH_PX),
                  (cx + DEPTH_PATCH_PX, cy + DEPTH_PATCH_PX), OVERLAY_COLOR, 1)
    text = f"center {depth_m:.3f} m" if depth_m is not None else "center --- (no depth)"
    cv2.putText(image, text, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                OVERLAY_COLOR, 2, cv2.LINE_AA)


def make_handler(camera: Camera):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args) -> None:      # 접속마다 찍히는 소음을 끈다
            pass

        def do_GET(self) -> None:
            if self.path.rstrip("/") in ("", "/index.html"):
                self._send_page()
            elif self.path.startswith("/stream"):
                self._send_stream()
            else:
                self.send_error(404)

        def _send_page(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_PAGE)))
            self.end_headers()
            self.wfile.write(_PAGE)

        def _send_stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    jpeg = camera.frame()
                    if jpeg is None:
                        continue
                    self.wfile.write(BOUNDARY + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass                                 # 브라우저가 닫은 것 — 정상
    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--bind", default=DEFAULT_BIND,
                        help="기본 127.0.0.1 — 네트워크에 노출하지 않는다")
    args = parser.parse_args()

    try:
        camera = Camera()
    except RuntimeError as exc:
        print(f"❌ RealSense 를 열 수 없다: {exc}")
        print("   `rs-enumerate-devices` 로 연결을 확인할 것")
        return 1

    server = ThreadingHTTPServer((args.bind, args.port), make_handler(camera))
    print(f"http://{args.bind}:{args.port}  (Ctrl-C 로 종료)")
    print("local5090 에서:  ssh -N -L "
          f"{args.port}:127.0.0.1:{args.port} vision-3090")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        server.server_close()
        camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
