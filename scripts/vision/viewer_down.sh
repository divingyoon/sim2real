#!/bin/bash
# 뷰어 종료. SIGTERM 후 2 초 안에 안 죽으면 SIGKILL 로 승급한다.
# ★호출하는 ssh 명령줄에 'cup_view_stream' 문자열이 있으면 pkill 이 그 셸을 죽인다 — 스크립트 경로로만 부를 것.
source "$(dirname "$0")/common.sh"
PATTERN="python3 cup_view_stream.py"
pkill -f "$PATTERN" || true
for _ in $(seq 1 10); do
  pgrep -f "$PATTERN" >/dev/null || { echo "viewer down"; exit 0; }
  sleep 0.2
done
pkill -9 -f "$PATTERN" || true
sleep 0.3
pgrep -f "$PATTERN" >/dev/null && { echo "viewer still alive after SIGKILL" >&2; exit 1; }
echo "viewer down (killed)"
