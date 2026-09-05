#!/usr/bin/env bash
# 우팔+손 preset 재생을 **한 bag** 에 담는다 — SIM(지령·실측) + REAL(실측) 같은 시계.
#
# 좌팔 run6(08.31) 프레임워크와 동형이다. 이 bag 하나가 real2sim 튜닝의 입력이 된다:
#   /shadow/sim_target  sim 이 원한 관절 목표      (npz arm_target)
#   /shadow/sim_meas    sim 물리가 실현한 값        (npz q_meas — sim 자체 처짐 포함)
#   /shadow/sim_hand    sim 손 지령 20D            (--with-hand 일 때)
#   /right_joint_trajectory_controller/joint_trajectory  우리가 실제로 보낸 세트포인트
#   /joint_states       실기 실측 (+effort)
#   /dg5f_right/joint_states  실기 손 실측
#   /dg5f_right/dynamic_joint_states  ★손 관절 **온도** (+pos/vel/effort)
#     08.31 신설: 하드웨어가 온도를 export 하는데 URDF 에 state_interface 선언이 없어
#     controller_manager 가 버리고 있었다. 20관절에 선언을 넣어 살렸다. 발열이 어디서
#     어떻게 오르는지가 손 튜닝의 핵심 정보다(엄지가 막힌 채 밀어붙여 뜨거워진 이력).
#     ⚠팔(openarm)은 하드웨어가 온도를 아예 안 낸다 — effort 로 간접 추정할 수밖에 없다.
#
# ★셋을 구분해서 남기는 이유: "sim 이 원한 것 / 우리가 보낸 것 / 실기가 간 것"을 뭉개면
#   추종 실패가 리미터 탓인지 팔 탓인지 못 가른다(08.31 좌팔에서 실측으로 배웠다).
#
# 사용:
#   ./record_right_shadow_bag.sh            # 기록만 (재생은 별 터미널에서)
#   ./record_right_shadow_bag.sh myrun      # bag 이름 지정
set -eo pipefail   # ★-u 금지: ROS setup.bash 가 unbound 변수를 참조한다

STAMP="${1:-$(date +%H%M%S)}"
OUT="$(cd "$(dirname "$0")/../.." && pwd)/logs/rosbags/right_preset_${STAMP}"

source /opt/ros/humble/setup.bash

echo "bag → ${OUT}"
echo "  Ctrl-C 로 종료. 재생이 끝난 뒤 몇 초 더 두고 끊을 것(꼬리 잘림 방지)."
exec ros2 bag record -o "${OUT}" \
  /joint_states \
  /dg5f_right/joint_states \
  /dg5f_right/dynamic_joint_states \
  /right_joint_trajectory_controller/joint_trajectory \
  /right_joint_trajectory_controller/controller_state \
  /dg5f_right/dg5f_right_controller/joint_trajectory \
  /isaacsim/right_arm_cmd \
  /isaacsim/right_hand_cmd \
  /shadow/sim_target \
  /shadow/sim_meas \
  /shadow/sim_hand
