#!/bin/bash
# 옛 체인 정리: vision 쪽 relay/UDP tx/옛 컨테이너 이름(fpp_cup·fpp_shaker) — 로컬 노드가 대체했다.
source "$(dirname "$0")/common.sh"
pkill -f cup_pose_relay.py || true
pkill -f pose_udp_tx.py || true
docker rm -f fpp_cup fpp_shaker >/dev/null 2>&1 || true
echo "legacy down"
