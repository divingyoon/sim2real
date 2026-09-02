#!/bin/bash
# 모니터 창(DISPLAY=:0) + MJPEG 8080. usage: viewer_up.sh <name>...
source "$(dirname "$0")/common.sh"
[ $# -ge 1 ] || { echo "need object names" >&2; exit 1; }
pkill -f cup_view_stream.py || true; sleep 0.5
DISPLAY=:0 setsid bash -c "source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=126;
  cd $SIM2REAL/scripts && exec python3 cup_view_stream.py --show --compressed --port 8080 --objects $*" \
  </dev/null >"$LOGDIR/viewer.log" 2>&1 &
sleep 2; pgrep -f cup_view_stream.py >/dev/null && echo "viewer up" || { cat "$LOGDIR/viewer.log" >&2; exit 1; }
