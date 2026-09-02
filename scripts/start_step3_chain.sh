#!/bin/bash
# Step 3 실기 지령 사슬 — 한 터미널에서 실행, Ctrl+C 로 일괄 종료.
# (러너 stream 35f → /isaacsim/*_cmd → JTC ×2. 유휴 중엔 지령이 없어 로봇 정지 유지)
#
# ★09.02 두 가지 변경 — 근거는 계획서 "좌 preset j7 처짐".
#  1) joint_states_to_udp(실기→sim 미러) 를 사슬에 넣는다. 이게 없으면 러너의
#     echo_apply(유휴 미러)와 shadow_guard(괴리 0.5rad 감시)가 **둘 다 조용히
#     무동작** 이라 GUI 가 sim 자기 믿음만 보여준다(09.02 좌팔 처짐이 안 보인 이유).
#  2) 좌팔 --arm-offset 제거. 정적 처짐은 gravity_comp_node(좌팔) 로 지운다 —
#     선보상까지 같이 켜면 같은 몫을 두 번 지워 j7 이 반대로 4.2° 들린다.
#     (그 값은 08.31 **구 홈**(정책 홈과 j4 21°·j7 28.6° 다름)에서 잰 한 점 보상이라
#      정책 실행 중 다른 자세에서는 어차피 안 맞는다.)
#  ⚠ 좌팔 중력보상은 이 사슬에 없다 — 팔에 토크를 거는 일이라 사용자가 직접 켠다:
#      ~/rl_ws/robot_control/ros_ws/load_effort_controllers.sh left
#      python3 gravity_comp_node.py --group openarm_left_arm --scale 1.0 --execute
set +u
source /opt/ros/humble/setup.bash
cd ~/rl_ws/sim2real/scripts
trap 'kill 0' INT TERM
# 인자로 팔을 고른다: both(기본) | left | right. 편측 라운드에선 쓰지 않는 팔의
# 브리지를 아예 안 띄워 지령이 한 줄도 못 나가게 한다(09.02 우 j7 고장 후 규약).
SIDE=${1:-both}
# 팔 속도상한[rad/s]. 정책 지령이 1.13~1.69 rad/s 를 요구하므로 이 값이 낮을수록
# 실기가 느리고 안전하다 — 뒤처지는 몫은 러너의 --shadow-pace 가 sim 을 멈춰 메운다.
MAXVEL=${MAXVEL:-1.0}
case "$SIDE" in both|left|right) ;; *) echo "인자는 both|left|right"; exit 2 ;; esac
echo "[사슬] side=$SIDE · max-vel=$MAXVEL"

python3 udp35_to_ros_cmd.py --port 47331 --execute &
python3 joint_states_to_udp.py --dest-port 47332 \
    --topics /joint_states,/dg5f_right/joint_states &
[ "$SIDE" != right ] && python3 isaacsim_cmd_to_jtc.py --robot gripper_left --max-vel "$MAXVEL" &
[ "$SIDE" != left ] && python3 isaacsim_cmd_to_jtc.py --robot tesollo_bi_s__right --max-vel "$MAXVEL" &
wait
