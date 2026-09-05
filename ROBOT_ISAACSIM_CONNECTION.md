# 로봇별 실물 ↔ Isaac Sim 연동 가이드

세 로봇을 실제 하드웨어와 Isaac Sim 사이에서 연결하는 방법을 로봇별로 정리한다.
이 문서는 **실물 ↔ sim 실시간 연동**(sim2real 방향)을 다룬다. sim 안에서 정책을
학습·재생하는 것은 `hdgp` 쪽 `play.py` / task 등록이 담당하며 여기서 다루지 않는다.

## 데이터 흐름

```
정책 / Isaac Sim  →  /isaacsim/* 명령 토픽  →  isaacsim_bridge  →  ros2_control  →  실제 하드웨어
                                                     ↑
실제 하드웨어 상태  →  /joint_states, /dg5f_right/joint_states  →  병합  →  /isaacsim/joint_states
```

핵심 노드는 `isaacsim_bridge/isaacsim_bridge/bridge_node.py` 하나다. 이 노드가
`/isaacsim/*_cmd` 다섯 토픽을 구독해 각 하드웨어 컨트롤러로 전달한다.

| 구독 (입력) | 타입 | 전달 대상 (출력) |
|---|---|---|
| `/isaacsim/left_arm_cmd` | `Float64MultiArray[7]` | `/left_joint_trajectory_controller/joint_trajectory` |
| `/isaacsim/right_arm_cmd` | `Float64MultiArray[7]` | `/right_joint_trajectory_controller/joint_trajectory` |
| `/isaacsim/right_hand_cmd` | `Float64MultiArray[20]` | `/dg5f_right/dg5f_right_controller/joint_trajectory` |
| `/isaacsim/left_gripper_cmd` | `Float64` | `/left_gripper_controller/gripper_cmd` |
| `/isaacsim/emergency_stop` | `Bool` | (전 채널 정지) |

브리지는 위치 목표를 `trajectory_time_sec`(기본 0.2s) 짜리 `JointTrajectory`로 보간해
내보낸다. 이 보간은 실시간 배포에는 적절하지만, actuator 파라미터 식별에는 부적절하다
(그 경우는 `hdgp/scripts/r2s_autotune/README.md` 참조).

---

## 지원 현황 (2026-07-11 기준)

| 로봇 | 실물 드라이버 | 브리지 연동 | sim task |
|---|---|---|---|
| OpenArm (양팔 7-DOF) | 있음 (`openarm_hardware`, CAN/MIT) | 있음 (`bridge_node`) | 있음 |
| Tesollo DG-5F (오른손 20-DOF) | 있음 (`delto_m_ros2/dg5f_driver`) | 있음 (`bridge_node`) | 있음 |
| RH56F1 (양손) | 있음 (`sim2real/vendor/inspire_ws`, `inspire_control_ros2`, 485/CANFD) | 있음 (`rh56f1_hand_bridge`) | 있음 |

세 로봇 모두 Isaac Sim ↔ ROS2 연결이 가능하다. 손 인터페이스가 서로 달라 브리지가
둘로 나뉜다.

- `bridge_node` — OpenArm 팔(양) + Tesollo 오른손 + 왼쪽 gripper (`JointTrajectory`, 라디안).
- `rh56f1_hand_bridge` — RH56F1 양손 (`SetAngle1`, 레지스터 정수). §3 참조.

RH56F1 세팅에서는 `bridge_node`(팔) + `rh56f1_hand_bridge`(손)를 함께 띄운다. 이때
`bridge_node`는 `right_hand_enabled:=false`로 Tesollo 경로를 끄고, 손 상태는
`extra_joint_state_topics:=[/rh56f1/joint_states]`로 병합한다.

---

## 1. OpenArm (팔)

### 하드웨어

`openarm_hardware/src/v10_simple_hardware.cpp`가 관절을 MIT 모드 PD로 구동한다.

```cpp
arm_params.push_back({kp_[i], kd_[i], pos_commands_[i], vel_commands_[i], tau_commands_[i]});
openarm_->get_arm().mit_control_all(arm_params);   // τ = kp·(q*−q) + kd·(q̇*−q̇)
```

게인은 `openarm_description/config/arm/v10/control_gains.yaml`에 있다.

| | j1 | j2 | j3 | j4 | j5 | j6 | j7 |
|---|---|---|---|---|---|---|---|
| kp | 70 | 70 | 70 | 60 | 10 | 10 | 10 |
| kd | 2.75 | 2.5 | 2.0 | 2.0 | 0.7 | 0.6 | 0.5 |

상태: position / velocity / effort 모두 발행 (`/joint_states`).

### 브링업

```bash
cd /home/user/rl_ws/teleopration_openarm_tesollo
source /opt/ros/humble/setup.bash && source install/setup.bash

# 기본값: right_can_interface=can0, left_can_interface=can1
ros2 launch openarm_bringup openarm.bimanual.launch.py \
    robot_controller:=joint_trajectory_controller
```

`joint_trajectory_controller`가 브리지의 팔 출력 토픽과 맞물린다.
raw 위치를 직접 넣으려면 `robot_controller:=forward_position_controller`로 띄우고
`/right_forward_position_controller/commands`(`Float64MultiArray[7]`,
순서 `openarm_right_joint1..7`)로 발행한다.

### 관절 이름

`openarm_left_joint1..7`, `openarm_right_joint1..7`. `Float64MultiArray`에는 이름이
없으므로 순서를 지켜야 한다. 컨트롤러 yaml의 `joints` 목록이 기준이다.

---

## 2. Tesollo DG-5F (오른손)

### 하드웨어

`delto_m_ros2/dg5f_driver`가 20-DOF 손을 구동한다. 두 가지 명령 인터페이스가 있다.

- `/dg5f_right/dg5f_right_controller/joint_trajectory` — ros2_control JTC (브리지 출력)
- `/dg5f_right/rj_dg_pospid/reference` — 드라이버 내부 PID의 raw 레퍼런스
  (`control_msgs/MultiDOFCommand`, 100 Hz, `dof_names` 포함)

실시간 정책 배포는 위(JTC)를 쓴다. actuator 식별은 아래(raw)를 쓴다.

### 관절 순서 (20-DOF)

```
rj_dg_1_1 rj_dg_1_2 rj_dg_1_3 rj_dg_1_4   (엄지)
rj_dg_2_1 rj_dg_2_2 rj_dg_2_3 rj_dg_2_4   (검지)
rj_dg_3_1 rj_dg_3_2 rj_dg_3_3 rj_dg_3_4   (중지)
rj_dg_4_1 rj_dg_4_2 rj_dg_4_3 rj_dg_4_4   (약지)
rj_dg_5_1 rj_dg_5_2 rj_dg_5_3 rj_dg_5_4   (소지)
```

`/isaacsim/right_hand_cmd`(`Float64MultiArray[20]`)도 이 순서다.

### 브링업

목적에 따라 launch 파일이 나뉜다.

```bash
# 드라이버 + JTC (브리지의 right_hand 출력과 맞물림)
ros2 launch dg5f_driver dg5f_right_driver.launch.py

# 또는 raw PID 레퍼런스(rj_dg_pospid/reference)를 받는 컨트롤러
ros2 launch dg5f_driver dg5f_right_pid_controller.launch.py
```

상태는 `/dg5f_right/joint_states`, 힘센서는 `/tesollo/right/sensor`.

### 연결 검증 (스모크)

```bash
# 손끝을 조금 굽혔다 펴는 raw 명령 (드라이버 직접, 브리지 우회)
python3 teleopration_openarm_tesollo/src/delto_m_ros2/dg5f_driver/script/dg5f_right_pid_test.py
```

---

## 3. RH56F1 — ROS2 드라이버 + Isaac Sim 브리지

### 하드웨어 드라이버

`sim2real/vendor/inspire_ws`의 `inspire_control_ros2` 패키지가 RH56F1(및 RH5DG2)을
구동한다. 485/CANFD 프로토콜을 지원하고, 노드 하나가 device별 토픽·서비스를 연다.
`device_protocol_config.yaml`에서 포트(`/dev/ttyUSB0`)·`Hand_ID`·프로토콜 타입을,
`ros2_controller_config.yaml`에서 device별 토픽 이름·`joint_names`·update_rate(기본 50Hz)를
정한다. (기본 config는 `RH5DG2_485`로 설정되어 있으니 RH56F1은 protocol type과
`joint_names`(6항)를 맞춰야 한다.)

RH56F1 메시지 타입은 `rh56f1_interfaces`에 있다.

| 방향 | 토픽 (오른손 예) | 타입 | 내용 |
|---|---|---|---|
| 명령 | `/hand_right/angle_set` | `SetAngle1` | `int32[6] joint_values`, `hand_id` |
| 측정 | `/hand_right/angle_actual` | `GetAngleAct1` | `int32[6] joint_values` + `string[6] joint_names` |
| 측정 | `/hand_right/force_actual` | `GetForceAct1` | 손끝 힘 |
| 측정 | `/hand_right/touch_data` | `TouchData1` | 촉각 |

모드/속도/힘/에러클리어 등은 서비스(`/hand_right/set_mode` 등)로 건다.
`joint_values`는 **레지스터 정수 단위**(예: 900~1740)이지 라디안이 아니다.

### 브링업

```bash
cd /home/user/rl_ws/sim2real/vendor/inspire_ws
# (빌드: colcon build) 후
source install/setup.bash
ros2 launch inspire_control_ros2 inspire_control_single_device.launch.py device_name:=hand_right
```

### Isaac Sim 브리지 (`rh56f1_hand_bridge`)

손 인터페이스가 Tesollo(`JointTrajectory`, 라디안)와 완전히 달라, RH56F1은 기존
`bridge_node`가 아니라 전용 노드 `isaacsim_bridge/rh56f1_hand_bridge_node.py`가 담당한다.

```
/isaacsim/{right,left}_hand_cmd (Float64MultiArray[6], 라디안)
    → 레지스터 변환 → SetAngle1 → /hand_{side}/angle_set
/hand_{side}/angle_actual (레지스터 int[6])
    → 라디안 변환 → JointState(canonical r_hj_*/l_hj_*) → /rh56f1/joint_states
```

명령 6개는 sim canonical drive 관절 순서 `[thumb_1, thumb_2, index_1, middle_1,
ring_1, pinky_1]`(정책 action 순서)를 따른다. 라디안↔레지스터 선형 변환과 관절↔슬롯
순열은 `rh56f1_hand.py`(순수 로직)가 처리하고, 값은 `config/rh56f1_hand_calibration.yaml`로
노출된다. 손가락↔액추에이터 ID 순서와 레지스터 증가 방향은 하드웨어별로 다르므로
실물에서 이 yaml만 보정하면 된다(코드 수정 불필요).

```bash
# 팔(OpenArm) 브리지 — Tesollo 경로는 끄고 손 상태를 병합
ros2 run isaacsim_bridge bridge_node --ros-args \
    -p right_hand_enabled:=false \
    -p "extra_joint_state_topics:=[/rh56f1/joint_states]"

# RH56F1 양손 브리지
ros2 launch isaacsim_bridge rh56f1_hand_bridge.launch.py
```

`rh56f1_interfaces`(SetAngle1/GetAngleAct1)가 빌드돼 있지 않으면 손 브리지는 명령/상태
채널을 비활성화한 채 뜬다(경고 로그). `sim2real/vendor/inspire_ws`를 colcon 빌드하고
install을 source해야 실제로 드라이버와 주고받는다.

### real2sim 식별 경로

명령(`angle_set`)과 측정(`angle_actual`)이 모두 ROS 토픽이므로 Tesollo와 같은
rosbag 경로를 그대로 쓴다. `angle_set` + `angle_actual`을 녹화해 변환하면
`hdgp/scripts/r2s_autotune`의 replay/MSE 단계로 넘어간다. 단, 변환기에서 두 가지를
처리해야 한다.

1. **메시지 타입 확장** — `db3_to_identification_hdf5.py`의 `_to_named_sample`는 지금
   `MultiDOFCommand`/`JointState`/`Float64MultiArray`만 안다. `SetAngle1`(이름 없음,
   순서 필요)·`GetAngleAct1`(이름 포함)을 추가해야 한다.
2. **단위·이름 변환** — 레지스터 정수 → 라디안, 드라이버 관절 이름(`hand_right/thumb_mcp` 등)
   → sim canonical(`r_hj_*`). canonical 이름과 drive/mimic 그룹은 이미 정의돼 있다
   (`urdf/generated/rl/openarm_bi_rh56f1_rl_manifest.yaml`).

---

## 4. dry-run (하드웨어 없이 검증)

실물 없이 브리지 배선과 정책 출력을 확인하려면 fake hardware + RViz를 쓴다.
`SIM2REAL_INFERENCE.md`의 4-터미널 절차가 `5g_grasp_right_v7` 정책 기준의 완결된 예다.

```bash
# 터미널 1: fake hardware + RViz
ros2 launch openarm_bringup openarm.bimanual.launch.py use_fake_hardware:=true

# 터미널 2: isaacsim_bridge
ros2 run isaacsim_bridge bridge_node

# 터미널 3: dry-run 노드 (하드웨어 대신 RViz로 확인)
python3 sim2real/scripts/deprecated/sim2real_dryrun.py
```

로봇별 정책·체크포인트를 바꿔가며 이 골격을 재사용한다.

---

## 관련 문서

| 문서 | 내용 |
|---|---|
| `isaacsim_bridge/README.md` | 브리지 패키지 상세, 튜닝 리포트 |
| `isaacsim_bridge/ISAACSIM_ACTION_GRAPH.md` | Isaac Sim 쪽 Action Graph로 5토픽 발행 |
| `isaacsim_bridge/ISAACSIM_POLICY_WIRING.md` | Action Graph에 정책 출력 연결 |
| `SIM2REAL_INFERENCE.md` | OpenArm+Tesollo `5g_grasp_right_v7` 배포 전체 절차 |
| `hdgp/scripts/r2s_autotune/README.md` | 실물 응답으로 sim actuator 보정 (반대 방향) |
