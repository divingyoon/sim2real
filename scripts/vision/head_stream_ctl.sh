#!/bin/bash
# head MJPEG 스트리머 제어. ★ssh 명령줄에 앱 이름이 들어가지 않게 하려고 파일로 뺐다
# (pkill -f 가 자기 자신을 죽이는 문제).
PY="$HOME/rl_ws/perception_plus_plus/.venv/bin/python"
APP="/tmp/stream_head_view.py"
PIDF=/tmp/head_stream.pid
LOG=/tmp/head_stream.log

case "${1:-status}" in
  stop)
    [ -f "$PIDF" ] && kill -9 "$(cat "$PIDF")" 2>/dev/null
    rm -f "$PIDF"; echo "stopped" ;;
  fg)
    timeout "${2:-10}" "$PY" -u "$APP" 2>&1 | head -20; echo "[fg 종료]" ;;
  start)
    setsid nohup "$PY" -u "$APP" > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDF"
    for i in $(seq 1 12); do
      sleep 1
      c=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/ 2>/dev/null)
      [ "$c" = "200" ] && { echo "✓ ${i}초만에 기동 · HTTP 200 · pid $(cat "$PIDF")"; exit 0; }
    done
    echo "❌ 12초 내 기동 실패"; echo "--- log ---"; cat "$LOG" ;;
  status)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      echo "실행 중 pid $(cat "$PIDF") · HTTP $(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/)"
    else echo "실행 안 됨"; fi
    echo "--- log ---"; tail -10 "$LOG" 2>/dev/null ;;
esac
