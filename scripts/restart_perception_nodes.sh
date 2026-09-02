#!/bin/bash
# 로컬 인지 노드(런처·pose) 재시작. ★pkill 패턴이 호출자 명령줄에 나오지 않도록 반드시 이 파일로 부른다.
#   usage: restart_perception_nodes.sh [launcher|pose|all]   (기본 all)
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=126
WHAT=${1:-all}
restart() {  # $1 = 스크립트 파일명, $2 = 로그
  pkill -f "python3 $1" || true
  sleep 0.5
  (cd "$HERE" && setsid python3 "$1" </dev/null >"$2" 2>&1 &)
  sleep 1.5
  pgrep -f "python3 $1" >/dev/null && echo "$1 up (log $2)" || { echo "$1 failed"; cat "$2"; exit 1; }
}
[ "$WHAT" = pose ] || restart perception_launcher_node.py /tmp/perception_launcher.log
[ "$WHAT" = launcher ] || restart object_pose_node.py /tmp/object_pose_node.log
