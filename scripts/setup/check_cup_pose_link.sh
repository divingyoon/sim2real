#!/usr/bin/env bash
# vision ↔ 로봇 제어 PC 간 /cup_pose 연계 점검. 각 PC 에서 실행.
#   기대: ROS_DOMAIN_ID=126 동일 + (vision 기동 상태면) /cup_pose 가 보인다.
set -uo pipefail
EXPECT_DOMAIN="${1:-126}"
fail=0
if [[ "${ROS_DOMAIN_ID:-unset}" != "$EXPECT_DOMAIN" ]]; then
  echo "[FAIL] ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset} (기대 $EXPECT_DOMAIN) — export ROS_DOMAIN_ID=$EXPECT_DOMAIN"
  fail=1
else
  echo "[OK] ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
fi
echo "[info] RMW=${RMW_IMPLEMENTATION:-default} CYCLONEDDS_URI=${CYCLONEDDS_URI:-unset}"
if ros2 topic list 2>/dev/null | grep -qx /cup_pose; then
  echo "[OK] /cup_pose 가시 — 타입: $(ros2 topic type /cup_pose 2>/dev/null)"
else
  echo "[warn] /cup_pose 미가시 (vision 노드 미기동이면 정상)"
fi
exit $fail
