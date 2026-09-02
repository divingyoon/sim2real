#!/bin/bash
# vision-3090 전용 공통 설정. 모든 vision/*.sh 가 source 한다.
# ★set -u 금지: /opt/ros/humble/setup.bash 가 미정의 변수를 참조해 즉사한다.
set -eo pipefail
export ROS_DOMAIN_ID=126
SIM2REAL=/home/usr/rl_ws/sim2real
PPP=/home/usr/rl_ws/perception_plus_plus
LOGDIR=/tmp/perception
mkdir -p "$LOGDIR"
source /opt/ros/humble/setup.bash
