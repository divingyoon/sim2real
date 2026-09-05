#!/bin/bash
# Step 3 실기 라운드 bag 기록 — ACTION(지령)/REAL(실측) 동형 규약.
# 사용:  ./record_step3_bag.sh <라벨>     예) ./record_step3_bag.sh round1_left
# 종료:  Ctrl+C  (bag 은 sim2real/logs/bags/step3_<날짜>/<라벨>/ 에 남는다)
set -e
LABEL=${1:?라벨을 줘라 — 예: round1_left}
OUT=~/rl_ws/sim2real/logs/bags/step3_$(date +%m%d)/${LABEL}
mkdir -p "$(dirname "$OUT")"
source /opt/ros/humble/setup.bash
exec ros2 bag record -o "$OUT" \
  /isaacsim/left_arm_cmd /isaacsim/left_gripper_cmd \
  /isaacsim/right_arm_cmd /isaacsim/right_hand_cmd \
  /joint_states /dg5f_right/joint_states \
  /left_joint_trajectory_controller/controller_state \
  /right_joint_trajectory_controller/controller_state
