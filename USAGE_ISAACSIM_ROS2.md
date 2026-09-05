# 사용설명서 — 실물 로봇 ↔ Isaac Sim (ROS 2)

OpenArm · Tesollo · RH56F1 각각을 **제어(bringup) → Isaac Sim 연결(bridge) → 동작 test**
순서로 실행하는 런북이다. 개념·토픽 표·설계 배경은 `ROBOT_ISAACSIM_CONNECTION.md`,
브리지 내부는 `isaacsim_bridge/README.md`를 본다. 이 문서는 "무슨 명령을 어떤 순서로
치는가"만 다룬다.

명령은 별도 터미널에서 계속 떠 있어야 하는 것(드라이버·브리지)과 일회성(test)이 섞여
있다. 각 블록 위에 `[터미널 N]`을 표시한다.

---

## 0. 공통 준비 (모든 터미널)

```bash
# 매 터미널에서 먼저 실행
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=126          # 모든 노드가 같은 값이어야 서로 보인다

# 워크스페이스 경로 (환경에 맞게 한 번만 정의)
export RL_WS=/home/user/rl_ws
export TELEOP_WS=$RL_WS/teleopration_openarm_tesollo   # OpenArm + Tesollo 드라이버
export INSPIRE_WS=$RL_WS/sim2real/vendor/inspire_ws     # RH56F1 드라이버
export BRIDGE_WS=$RL_WS/sim2real                        # isaacsim_bridge (colcon 빌드 위치)
```

- Isaac Sim 쪽은 Action Graph로 `/isaacsim/*` 토픽을 주고받는다
  (`isaacsim_bridge/ISAACSIM_ACTION_GRAPH.md`). 실물 없이 배선만 볼 때는 §5 dry-run.
- 상태 확인 상시 명령: `ros2 node list`, `ros2 topic list`, `ros2 topic hz <topic>`.

---

## 1. OpenArm (양팔 7-DOF)

### 1-1. 제어 (bringup)

```bash
# [터미널 1] 실물 팔 드라이버 (CAN/MIT)
cd $TELEOP_WS && source install/setup.bash
ros2 launch openarm_bringup openarm.bimanual.launch.py \
    robot_controller:=joint_trajectory_controller \
    right_can_interface:=can0 left_can_interface:=can1
```

확인: `ros2 topic echo /joint_states --once` 에 `openarm_*_joint1..7`이 보이면 OK.

### 1-2. Isaac Sim 연결 (bridge)

```bash
# [터미널 2] 브리지 (팔 명령을 컨트롤러로 전달)
cd $BRIDGE_WS && source install/setup.bash
ros2 launch isaacsim_bridge isaacsim_bridge.launch.py
```

`/isaacsim/{left,right}_arm_cmd`(Float64MultiArray[7], 라디안)를 받아
`*_joint_trajectory_controller`로 보낸다. 팔 상태는 `/isaacsim/joint_states`로 병합 발행.

### 1-3. test

```bash
# [터미널 3] 오른팔을 영자세로
cd $BRIDGE_WS && source install/setup.bash
python3 scripts/nodes/manual_command_pub.py right-arm 0 0 0 0 0 0 0
# 관절 하나만 살짝
python3 scripts/nodes/manual_command_pub.py right-arm 0 0 0 0 0 0 0.2

# 병합 상태 확인
ros2 topic echo /isaacsim/joint_states --once
```

`ros2 topic pub`을 직접 쓸 수도 있다:

```bash
ros2 topic pub --once /isaacsim/right_arm_cmd std_msgs/msg/Float64MultiArray \
  "{data: [0,0,0,0,0,0,0.2]}"
```

---

## 2. Tesollo DG-5F (오른손 20-DOF)

### 2-1. 제어 (bringup)

```bash
# [터미널 1] Tesollo 드라이버 + JTC
cd $TELEOP_WS && source install/setup.bash
ros2 launch dg5f_driver dg5f_right_driver.launch.py
```

확인: `ros2 topic echo /dg5f_right/joint_states --once` 에 `rj_dg_*` 20개.
드라이버 단독 스모크(브리지 우회):

```bash
python3 $TELEOP_WS/src/delto_m_ros2/dg5f_driver/script/dg5f_right_pid_test.py
```

### 2-2. Isaac Sim 연결 (bridge)

```bash
# [터미널 2] 브리지 (OpenArm과 동일 노드가 Tesollo 손도 담당)
cd $BRIDGE_WS && source install/setup.bash
ros2 launch isaacsim_bridge isaacsim_bridge.launch.py
```

`/isaacsim/right_hand_cmd`(Float64MultiArray[20])를 `dg5f_right_controller`로 전달.
관절 순서는 `rj_dg_1_1 … rj_dg_5_4`.

### 2-3. test

```bash
# [터미널 3] 20개 값(엄지→소지, 관절1→4). 모두 0 = 편 손
cd $BRIDGE_WS && source install/setup.bash
python3 scripts/nodes/manual_command_pub.py right-hand \
    0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

ros2 topic echo /isaacsim/joint_states --once
```

---

## 3. RH56F1 (양손 6-DOF/손)

### 3-1. 제어 (bringup)

```bash
# [터미널 1] RH56F1 드라이버 (485/CANFD). 단일 손 예:
cd $INSPIRE_WS && colcon build --symlink-install --base-paths src/ros2/src   # 최초 1회 (ROS2 패키지는 src/ros2/src 아래)
source install/setup.bash
ros2 launch inspire_control_ros2 inspire_control_single_device.launch.py \
    device_name:=hand_right
# 양손이면 multi-device (config의 device 목록만큼 노드 생성)
# ros2 launch inspire_control_ros2 inspire_control_multi_device.launch.py
```

포트·프로토콜·hand_id는 `src/ros2/src/driver/config/device_protocol_config.yaml`,
관절 이름/토픽은 `ros2_controller_config.yaml`에서 조정한다(기본 config는 RH5DG2_485이니
RH56F1은 protocol type과 joint_names 6항을 맞춘다).

확인:

```bash
ros2 topic echo /hand_right/angle_actual --once     # 레지스터 정수 int[6]
```

드라이버 직접 명령 스모크(브리지 우회, 레지스터 단위):

```bash
ros2 topic pub --once /hand_right/angle_set rh56f1_interfaces/msg/SetAngle1 \
  "{hand_id: 2, joint_values: [1200,1200,1200,1200,1200,1200]}"
```

### 3-2. Isaac Sim 연결 (bridge)

RH56F1은 손 메시지가 Tesollo와 달라 **별도 노드 + 팔 노드**를 함께 띄운다.

```bash
# [터미널 2] 팔 브리지 — Tesollo 경로는 끄고, 손 상태를 병합
cd $BRIDGE_WS && source install/setup.bash
ros2 run isaacsim_bridge bridge_node --ros-args \
    -p right_hand_enabled:=false \
    -p "extra_joint_state_topics:=[/rh56f1/joint_states]"
```

```bash
# [터미널 3] RH56F1 양손 브리지
cd $BRIDGE_WS && source install/setup.bash
ros2 launch isaacsim_bridge rh56f1_hand_bridge.launch.py
# 오른손만: ros2 launch isaacsim_bridge rh56f1_hand_bridge.launch.py hands:="[right]"
```

라디안↔레지스터 변환값은 `isaacsim_bridge/config/rh56f1_hand_calibration.yaml`.
`rh56f1_interfaces`가 빌드/소스되지 않으면 손 브리지는 경고만 내고 명령/상태 채널을 끈다.

### 3-3. test

명령은 **라디안 6개**, 순서 `[thumb_1, thumb_2, index_1, middle_1, ring_1, pinky_1]`
(정책 action 순서). `manual_command_pub.py right-hand`는 Tesollo 20-DOF 전용이라 여기선
`ros2 topic pub`을 쓴다.

```bash
# [터미널 4] 오른손 살짝 굽힘 (index~pinky 0.5rad)
ros2 topic pub --once /isaacsim/right_hand_cmd std_msgs/msg/Float64MultiArray \
  "{data: [0.0, 0.0, 0.5, 0.5, 0.5, 0.5]}"

# 왼손
ros2 topic pub --once /isaacsim/left_hand_cmd std_msgs/msg/Float64MultiArray \
  "{data: [0.0, 0.0, 0.5, 0.5, 0.5, 0.5]}"

# 라디안으로 되돌아온 손 상태 확인
ros2 topic echo /rh56f1/joint_states --once
```

> 실물 첫 연결 시: 어느 손가락이 어느 슬롯인지, 굽힘 방향(레지스터 증감)이 맞는지
> 하나씩 굽혀 확인하고 `rh56f1_hand_calibration.yaml`만 보정한다(코드 수정 불필요).

---

## 4. 통합 세팅 (팔 + 손 동시)

### 4-A. OpenArm + Tesollo

브리지 launch가 하드웨어 bringup까지 함께 띄운다.

```bash
# [터미널 1] 브리지 + 실기(OpenArm+Tesollo) 한 번에
cd $BRIDGE_WS && source install/setup.bash
ros2 launch isaacsim_bridge isaacsim_bridge.launch.py \
    with_hardware:=true \
    left_can_interface:=can1 right_can_interface:=can0 \
    dg5f_right_ip:=169.254.186.72 dg5f_right_port:=502
```

test는 §1-3(팔) + §2-3(손) 명령을 그대로 사용.

### 4-B. OpenArm + RH56F1

세 터미널: 팔 드라이버 / (팔 브리지 + RH56F1 손 브리지) / RH56F1 드라이버.

```bash
# [터미널 1] OpenArm 드라이버              → §1-1
# [터미널 2] RH56F1 드라이버(양손)          → §3-1 (multi-device)
# [터미널 3] 팔 브리지 (Tesollo off, 손 병합) → §3-2 터미널 2
# [터미널 4] RH56F1 손 브리지               → §3-2 터미널 3
```

test는 §1-3(팔) + §3-3(손).

---

## 5. dry-run (하드웨어 없이 배선만 확인)

```bash
# [터미널 1] fake hardware + RViz
cd $TELEOP_WS && source install/setup.bash
ros2 launch openarm_bringup openarm.bimanual.launch.py use_fake_hardware:=true

# [터미널 2] 브리지
cd $BRIDGE_WS && source install/setup.bash
ros2 run isaacsim_bridge bridge_node

# [터미널 3] 명령을 넣고 RViz에서 움직임 확인
python3 $BRIDGE_WS/scripts/nodes/manual_command_pub.py right-arm 0 0 0 0 0 0 0.3
```

Tesollo/RH56F1은 fake hardware가 없어 실물 시리얼/CAN이 필요하다. 정책 배포 전체
예시는 `SIM2REAL_INFERENCE.md`(OpenArm+Tesollo `5g_grasp_right_v7`).

---

## 6. 안전 · 트러블슈팅

```bash
# 비상정지: 모든 나가는 명령 차단 (해제는 data: false)
ros2 topic pub --once /isaacsim/emergency_stop std_msgs/msg/Bool "{data: true}"
```

| 증상 | 확인 |
|---|---|
| 노드가 서로 안 보임 | 모든 터미널 `ROS_DOMAIN_ID` 동일한가 |
| 명령 무시됨 | e-stop 활성 아닌가 / 값 개수가 관절 수와 맞는가(로그 warning) |
| RH56F1 손 안 움직임 | `rh56f1_interfaces` 소스됐나 / 드라이버 `angle_set` 구독 중인가 |
| 손가락 엉뚱하게 굽음 | `rh56f1_hand_calibration.yaml`의 `driver_index`·`reg_lo/hi` 보정 |
| 병합 상태에 손 없음 | 팔 브리지에 `extra_joint_state_topics:=[/rh56f1/joint_states]` 줬나 |

---

## 7. 비전 노드 — Isaac ROS FoundationPose → `/cup_pose` (pour s2r P1)

D435i 글로벌 카메라 영상에서 소스 컵 6-DOF pose를 추정해 `/cup_pose`
(PoseStamped, robot base 프레임)로 발행한다. grasp 단계 + pour 시작 시
grasp offset 캡처 1회에만 쓰이고, pour 루프는 FK라 이 노드 없이도 돈다.

> **Isaac Sim과 무관.** Isaac ROS는 실물용 ROS2 지각 패키지 모음이다
> (x86_64 + Ubuntu 22.04 + ROS2 Humble). Docker(Isaac ROS dev 컨테이너)로 실행.

### 7-1. 설치 (1회)

```bash
# [호스트] Docker + nvidia-container-toolkit 필요
# Isaac ROS 공통 dev 환경
mkdir -p ~/workspaces/isaac_ros-dev/src && cd ~/workspaces/isaac_ros-dev/src
git clone -b main https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone -b main https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation.git

# dev 컨테이너 진입 (이후 명령은 전부 컨테이너 안)
cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && ./scripts/run_dev.sh

# [컨테이너] FoundationPose + RealSense 패키지
sudo apt-get update && sudo apt-get install -y \
    ros-humble-isaac-ros-foundationpose \
    ros-humble-isaac-ros-examples \
    ros-humble-realsense2-camera

# 모델(TensorRT 변환용 onnx) 다운로드 — quickstart 자산 스크립트 사용
# https://nvidia-isaac-ros.github.io/robots/foundationpose 의 quickstart 참조
```

준비물(설치와 별개):
- **컵 CAD 메시** (textured .obj) — FoundationPose model-based 입력.
- CAD 원점/축이 sim body 프레임(원점=바닥 중심, +z=위)과 다르면
  `config/global_camera_extrinsics.yaml`의 `cad_to_body`로 보정.

### 7-2. 캘리브레이션 (1회)

```bash
# 글로벌 카메라 extrinsics(T_base ← camera_color_optical_frame)를 실측 후
# config/global_camera_extrinsics.yaml 의 camera.position/orientation_wxyz 교체.
# ⚠ 기본값은 PLACEHOLDER(identity) — 교체 전 실기 구동 금지.
```

### 7-3. 실행

```bash
# [컨테이너 터미널 1] D435i + FoundationPose (컵 메시 경로 지정)
ros2 launch isaac_ros_examples isaac_ros_examples.launch.py \
    launch_fragments:=realsense_mono_rect_depth,foundationpose \
    mesh_file_path:=/path/to/cup.obj \
    texture_path:=/path/to/cup_texture.png

# [호스트 터미널 2] 릴레이 (Detection3DArray → /cup_pose, base 프레임)
cd $RL_WS/sim2real/scripts
python3 cup_pose_relay.py --in-topic /poses --min-score 0.0
```

컨테이너↔호스트는 같은 `ROS_DOMAIN_ID`면 DDS로 바로 통신된다.

### 7-4. test

```bash
# pose 수신 확인 (컵을 카메라 앞에서 움직이며 값 추종 확인)
ros2 topic echo /cup_pose --once
ros2 topic hz /poses            # FoundationPose 추론 주기 확인

# 릴레이 순수 로직 회귀 (ROS 불필요)
python3 -m pytest scripts/test_cup_pose_relay.py -q
```

| 증상 | 확인 |
|---|---|
| `/poses` 안 나옴 | 메시/텍스처 경로, 모델 변환(TensorRT 엔진 생성 수 분 소요) |
| `/cup_pose` 값이 이상함 | extrinsics PLACEHOLDER 그대로 아닌가 / `cad_to_body` 정합 |
| 컨테이너↔호스트 안 보임 | 양쪽 `ROS_DOMAIN_ID`·RMW 구현 동일한가 |

---

## 관련 문서

| 문서 | 내용 |
|---|---|
| `INSTALL.md` | 새 PC 세팅 (step-by-step 설치, `scripts/setup_check.sh` 진단) |
| `ROBOT_ISAACSIM_CONNECTION.md` | 로봇별 연동 상세, 토픽 표, 설계 배경 |
| `isaacsim_bridge/README.md` | 브리지 파라미터·튜닝·Action Graph |
| `SIM2REAL_INFERENCE.md` | 정책 배포 전체 절차(OpenArm+Tesollo) |
| `hdgp/scripts/r2s_autotune/README.md` | 실물 응답으로 sim actuator 보정(반대 방향) |
