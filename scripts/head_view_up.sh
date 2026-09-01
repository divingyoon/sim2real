#!/usr/bin/env bash
# head RealSense 라이브 화면을 local5090 브라우저로 가져온다.
#
#   ./head_view_up.sh          기동 (배포 → 원격 시작 → 터널)
#   ./head_view_up.sh down     정리
#
# 카메라는 vision-3090, 모터는 local5090 이라 두 기계에 걸쳐 있다. vision-3090 의
# sshd 는 X11Forwarding 이 꺼져 있고, realsense-viewer 는 OpenGL 앱이라 포워딩해도
# 느리다. 그래서 MJPEG + SSH 터널을 쓴다 — sudo·설정변경·네트워크 노출이 없다.
set -euo pipefail

HOST=vision-3090
PORT=8080
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-up}" = "down" ]; then
  ssh "$HOST" '/tmp/head_stream_ctl.sh stop' || true
  # ★패턴이 이 스크립트의 명령줄에 없도록 포트만으로 찾는다 (pkill 자살 방지)
  ss -ltnp 2>/dev/null | grep ":$PORT" | grep -oP 'pid=\K[0-9]+' | xargs -r kill || true
  echo "정리 완료"; exit 0
fi

scp -q "$HERE/stream_head_view.py" "$HERE/head_stream_ctl.sh" "$HOST:/tmp/"
ssh "$HOST" "chmod +x /tmp/head_stream_ctl.sh && /tmp/head_stream_ctl.sh stop && /tmp/head_stream_ctl.sh start"
ss -ltn 2>/dev/null | grep -q ":$PORT" || \
  ssh -f -N -o ExitOnForwardFailure=yes -L "$PORT:127.0.0.1:$PORT" "$HOST"
echo
echo "  ▶ 브라우저에서  http://127.0.0.1:$PORT"
