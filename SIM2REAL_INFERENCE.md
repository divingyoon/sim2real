> ⚠️ **DEPRECATED — 구 v7 계약(obs 106D / action 11D)**
>
> 현행 `grasp_v1` 계약은 **obs 154D / action 21D**, 진입점은
> `scripts/grasp_inference.py --robot <구성 프로필>` 이다.
> 계약 요약은 `docs/CONTRACT_grasp_v1_{right,left}.md`, 실행 절차는
> `docs/RUNBOOK_GRASP_V1_LIVE.md` 를 보라. 이 문서는 이력 보존용이다.

# sim2real Inference — 5g_grasp_right_v7

학습된 `5g_grasp_right_v7` 정책을 실물 OpenArm + Teosllo 로봇에서 실행하는 파이프라인.

---

## 구성 파일

| 파일 | 역할 |
|------|------|
| `scripts/sim2real_inference.py` | ROS2 추론 노드 (실물 하드웨어용) |
| `scripts/sim2real_dryrun.py` | ROS2 시각화 노드 (하드웨어 없이 RViz 확인용) |
| `scripts/policy_loader.py` | rl_games actor MLP 로더 (Isaac Sim 의존성 없음) |
| `scripts/fabrics_ros_interface.py` | ROS2 명령 퍼블리셔 (`Sim2RealCommandPublisher`) |

**외부 의존 경로:**

| 경로 | 용도 |
|------|------|
| `hdgp/source/FABRICS/src/` | Geometric Fabrics 라이브러리 |
| `hdgp/source/openarm/` | v7 preset / constants (관절명, 포즈, 워크스페이스 한계) |
| `hdgp/log/rl_games/pipeline/right/5g_grasp_right_v7/test4/` | 체크포인트 경로 |

---

## 체크포인트 경로

```
nn/5g_grasp_right-v7.pth      ← best model
params/agent.yaml              ← 네트워크 설정
```

```bash
AGENT=/home/user/rl_ws/hdgp/log/rl_games/pipeline/right/5g_grasp_right_v7/test4/params/agent.yaml
CKPT=/home/user/rl_ws/hdgp/log/rl_games/pipeline/right/5g_grasp_right_v7/test4/nn/5g_grasp_right-v7.pth
```

---

## ROS2 토픽 구조

두 하드웨어가 **독립 ros2_control 스택**으로 분리되어 있어 별도 토픽을 사용합니다.

```
[sim2real_inference / sim2real_dryrun]
  publish → /isaacsim/right_arm_cmd  (Float64MultiArray 7D)
  publish → /isaacsim/right_hand_cmd (Float64MultiArray 20D)
                    ↓
          [isaacsim_bridge]  ← 반드시 실행 필요
                    ↓
  ┌─ OpenArm arm (ros2_control, CAN) ──────────────────────────────┐
  │  /right_joint_trajectory_controller/joint_trajectory           │
  │  상태: /joint_states  (openarm_right_joint1~7)                │
  └────────────────────────────────────────────────────────────────┘
  ┌─ Teosllo hand (ros2_control, Ethernet, ns=dg5f_right) ─────────┐
  │  /dg5f_right/dg5f_right_controller/joint_trajectory            │
  │  상태: /dg5f_right/joint_states  (rj_dg_1_1 ~ rj_dg_5_4)     │
  │  command_interface: effort (내부 PD, p=1.5, d=0.0)             │
  └────────────────────────────────────────────────────────────────┘
```

> **참고 — Teosllo 제어 방식**: `dg5f_right_controller`는 position target을 받아
> 내부 PD(p=1.5)로 effort 변환 후 모터에 전달. 시뮬 설정(stiffness=30, damping=5)과
> 다르므로 실기 검증 시 게인 조정이 필요할 수 있음.

---

## 동작 흐름

```
[준비]
  /cup_pose 토픽 수신 (외부 perception → cup 위치, robot base frame)
  /joint_states, /dg5f_right/joint_states 수신 (arm 7D / hand 20D)

       ↓  ros2 service call /sim2real/start

[APPROACHING  ~4초]
  Fabrics IK rollout (60스텝) → pregrasp_arm_pos 계산
  pregrasp_arm_pos + HAND_APPROACH_POSE → 10Hz 반복 전송
  로봇이 물리적으로 pregrasp 위치로 이동 + settle

       ↓  settle_time 경과

[RUNNING  10초 / 600스텝 @ 60Hz]
  ┌─ Grasp phase  (스텝 0~479, 8s) ─────────────────────────────────────┐
  │  1. 실제 관절값 → fabric_q 동기화                                   │
  │  2. Fabrics FK → palm_center (3D), fingertip_pos (5×3D)            │
  │  3. 106D obs 구성                                                    │
  │  4. policy.get_action(obs) → 11D action                             │
  │  5. action[0:6] → Fabrics IK → arm_cmd (7D)                        │
  │  6. action[6:11] → per-finger lerp → hand_cmd (20D)                │
  │     (action=-1 → APPROACH_POSE, action=+1 → GRASP_POSE)           │
  │  7. /isaacsim/right_arm_cmd, /isaacsim/right_hand_cmd 발행         │
  └──────────────────────────────────────────────────────────────────────┘
  ┌─ Lift phase  (스텝 480~599, 2s) ────────────────────────────────────┐
  │  scripted: j4 += 0.31 rad 선형 보간 (약 10cm 수직 상승)            │
  │  hand: 파지 자세 고정 (마지막 frame 유지)                           │
  └──────────────────────────────────────────────────────────────────────┘

       ↓  600스텝 완료

[DONE → IDLE]
```

---

## 관측값 구성 (106D Actor Obs)

| 항목 | 차원 | 설명 |
|------|------|------|
| `arm_joint_pos` | 7 | openarm_right_joint1~7 [rad] |
| `arm_joint_vel` | 7 | [rad/s] |
| `finger_joint_pos` | 20 | rj_dg_{1~5}_{1~4} [rad] |
| `finger_joint_vel` | 20 | [rad/s] |
| `palm_center_pos` | 3 | world frame [m], Fabrics FK |
| `fingertip_pos_rel_palm` | 15 | 5 fingertips − palm_center [m] |
| `palm_to_cup_pos` | 3 | cup_pos − palm_center [m] |
| `cup_to_fingertip` | 15 | fingertip_pos − cup_pos [m] |
| `fingertip_contact_binary` | 5 | `/dg5f_right/contact_forces` > 0.1N → 1 (sensor 없으면 0) |
| `last_actions` | 11 | 직전 스텝 action |
| **합계** | **106** | |

---

## 액션 공간 (11D)

| 인덱스 | 항목 | 변환 |
|--------|------|------|
| `[0:6]` | 6D palm pose delta (x,y,z,ez,ey,ex) | pregrasp_palm_pose + delta → Fabrics IK → arm 7D |
| `[6:11]` | 5D per-finger lerp (thumb~pinky) | -1 → APPROACH_POSE, +1 → GRASP_POSE → position target 20D |

**Palm workspace (MAX_POSE_ANGLE=45°):**

| 축 | min | max |
|----|-----|-----|
| x [m] | 0.20 | 0.65 |
| y [m] | −0.55 | −0.02 |
| z [m] | 0.20 | 0.65 |
| ez [rad] | π/4 | 3π/4 |
| ey [rad] | −π/4 | π/4 |
| ex [rad] | π/4 | 3π/4 |

**Pregrasp 방향:** ez=90°, ey=0°, ex=90°
→ palm +X(손바닥 법선) = world +Y, palm +Z(손가락) = world +X

---

## 손 자세 기준값

```
HAND_APPROACH_POSE (per-finger lerp action=−1):
  thumb : [0.0, −1.57, −0.5,  0.0]   # opposition pre-curl
  index : [0.0,  0.0,   0.0,  0.0]
  middle: [0.0,  0.0,   0.0,  0.0]
  ring  : [0.0,  0.0,   0.0,  0.0]
  pinky : [0.0,  0.0,   0.0,  0.0]

HAND_GRASP_POSE (per-finger lerp action=+1):
  thumb : [0.0, −1.57,  1.5,  1.5]
  index : [0.0,  1.6,   1.5,  1.5]
  middle: [0.0,  1.6,   1.5,  1.5]
  ring  : [0.0,  1.6,   1.5,  1.5]
  pinky : [0.0,  0.0,   1.5,  1.5]
```

---

## ROS2 인터페이스

### 구독 토픽

| 토픽 | 메시지 타입 | 내용 |
|------|-------------|------|
| `/joint_states` | `sensor_msgs/JointState` | arm 7D pos/vel (joint_state_broadcaster) |
| `/dg5f_right/joint_states` | `sensor_msgs/JointState` | hand 20D pos/vel (dg5f_driver) |
| `/cup_pose` | `geometry_msgs/PoseStamped` | cup 위치 (robot base frame) |
| `/dg5f_right/contact_forces` | `std_msgs/Float64MultiArray` | FT force 5D [N] (fingertip_sensor:=true 시) |

### 발행 토픽

| 토픽 | 메시지 타입 | 내용 |
|------|-------------|------|
| `/isaacsim/right_arm_cmd` | `std_msgs/Float64MultiArray` | arm position target 7D [rad] |
| `/isaacsim/right_hand_cmd` | `std_msgs/Float64MultiArray` | hand position target 20D [rad] |

### 서비스

| 서비스 | 동작 |
|--------|------|
| `/sim2real/start` | IDLE → APPROACHING → RUNNING |
| `/sim2real/stop` | 즉시 중단 → IDLE |
| `/sim2real/reset` | 에피소드 카운터 초기화 → IDLE |

---

## 실행 방법

### A. RViz 시각화 (dry-run, 하드웨어 불필요)

```bash
# 터미널 1: fake hardware + RViz
ros2 launch openarm_control openarm_left_gripper_bimanual_real.launch.py use_fake_hardware:=true

# 터미널 2: isaacsim_bridge
ros2 launch isaacsim_bridge isaacsim_bridge.launch.py

# 터미널 3: dry-run 노드
python3 /home/user/rl_ws/sim2real/scripts/sim2real_dryrun.py \
    --agent  $AGENT \
    --ckpt   $CKPT \
    --cup_x 0.40 --cup_y -0.15 --cup_z 0.38

# 터미널 4: 시작
ros2 service call /sim2real/start std_srvs/srv/Trigger
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--cup_x/y/z` | 0.40 / -0.15 / 0.38 | cup 위치 [m] (고정값) |
| `--settle_time` | `3.0` | APPROACHING 보간 시간 [s] |

### B. 실물 로봇

```bash
# 터미널 1: 하드웨어 드라이버 (arm + hand 통합)
ros2 launch integrated_control openarm_left_gripper_right_dg5_real.launch.py

# 터미널 2: isaacsim_bridge
ros2 launch isaacsim_bridge isaacsim_bridge.launch.py

# 터미널 3: 추론 노드
python3 /home/user/rl_ws/sim2real/scripts/sim2real_inference.py \
    --agent  $AGENT \
    --ckpt   $CKPT \
    --settle_time 4.0

# 터미널 4: cup 위치 수신 확인 후 시작
ros2 service call /sim2real/start std_srvs/srv/Trigger
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--agent` | 필수 | agent.yaml 경로 |
| `--ckpt` | 필수 | .pth 체크포인트 경로 |
| `--device` | `cuda:0` | torch device |
| `--settle_time` | `4.0` | pregrasp 이동 후 안정화 대기 시간 [s] |

### cup_pose 수동 퍼블리시 (perception 없을 때)

```bash
ros2 topic pub /cup_pose geometry_msgs/msg/PoseStamped \
    "{header: {frame_id: 'base'}, pose: {position: {x: 0.40, y: -0.15, z: 0.38}}}" --once
```

---

## 검증 절차

### 단계 1 — policy_loader 단독 테스트

```bash
python3 /home/user/rl_ws/sim2real/scripts/policy_loader.py \
    --agent $AGENT --ckpt $CKPT
# 예상: action 11D 출력, 약 ±0.X 범위
```

### 단계 2 — Fabrics FK 오프라인 확인

```bash
python3 - <<'EOF'
import sys
sys.path.insert(0, "/home/user/rl_ws/hdgp/source/FABRICS/src")
from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric
from fabrics_sim.utils.utils import initialize_warp
import torch

initialize_warp("0")
fabric = OpenArmTeoslloPoseFabric(1, "cpu", 1/60, graph_capturable=False, use_hand_fabric=False)

q = torch.tensor([[0.5, 0.1, 0.4, 0.60, -0.2, 0.0, 0.0] + [0.0]*20])
palm = fabric.get_palm_pose(q, "euler_zyx")
tips = fabric.get_fingertip_positions(q)
print("palm_center:", palm[0, :3].tolist())
print("fingertips :", tips[0].tolist())
EOF
```

### 단계 3 — dry-run RViz 시각화

위 **실행 방법 A** 참조.

### 단계 4 — 실물 로봇

위 **실행 방법 B** 참조. `--settle_time`은 로봇 이동 속도에 따라 조정 (기본 4.0s).

---

## 주요 상수

| 상수 | 값 | 설명 |
|------|-----|------|
| `GRASP_PHASE_STEPS` | 480 | Grasp phase (8s @ 60Hz) |
| `LIFT_PHASE_STEPS` | 120 | Lift phase (2s @ 60Hz) |
| `EPISODE_STEPS` | 600 | 총 에피소드 (10s) |
| `PREGRASP_FABRICS_STEPS` | 60 | Pregrasp IK rollout 반복 수 |
| `CONTACT_FORCE_THRESHOLD` | 0.1 N | binary contact 판정 임계값 |
| `PREGRASP_OFFSET` | [0.0, −0.12, 0.05] | cup 대비 pregrasp 오프셋 [m] |
| `RIGHT_ARM_START_POSE` | [0.5, 0.1, 0.4, 0.60, −0.2, 0.0, 0.0] | Fabrics rollout 초기 자세 |

---

## Teosllo 게인 비고

| | 시뮬 (Isaac Lab) | 실기 현재 | 비고 |
|-|-----------------|-----------|------|
| stiffness / p | 30.0 | **1.5** | 추후 확인 후 조정 |
| damping / d | 5.0 | **0.0** | 추후 확인 후 조정 |

게인 변경 시: `robot_control/ros_ws/src/delto_m_ros2/dg5f_driver/config/dg5f_right_controller.yaml`의 `gains` 섹션 수정(드라이버는 robot_control 소유).

---

## 트러블슈팅

| 증상 | 원인 | 조치 |
|------|------|------|
| start 서비스 실패 "미수신 토픽" | `/cup_pose` 미퍼블리시 | perception 또는 수동 pub 확인 |
| pregrasp 위치 부정확 | `PREGRASP_OFFSET` 값 | `grasp_right_preset.py` 조정 |
| settle 후 로봇 떨림 | `settle_time` 부족 | `--settle_time` 증가 |
| Lift phase에서 물체 낙하 | j4 증가량 부족 | `prelift[3] += 0.31` 값 조정 |
| 손이 안 닫힘 | Teosllo PD 게인 낮음 | `dg5f_right_controller.yaml` p/d 조정 |
| `rl_dg_*_tip` 링크 오류 | URDF tip 링크 미추가 | `openarm_tesollo.urdf` line 1248~ 확인 |
| isaacsim_bridge 없이 명령 무시 | bridge 미실행 | `isaacsim_bridge.launch.py` 실행 확인 |
