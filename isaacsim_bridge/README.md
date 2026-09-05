# Isaac Sim Bridge

이 패키지는 Isaac Sim 쪽 제어 입력 토픽을 실제 OpenArm / Tesollo / RH56F1 제어기로 전달하는 ROS 2 브리지입니다.

현재 실제 제어 흐름은 아래와 같습니다.

`FABRICS (또는 다른 Python 제어루프) -> /isaacsim/* ROS 2 토픽 -> isaacsim_bridge -> ros2_control / inspire_control_ros2 -> 실제 하드웨어`

손 인터페이스가 서로 달라 노드가 둘로 나뉩니다.

- `bridge_node` — OpenArm 팔(양) + Tesollo 오른손 + 왼쪽 gripper (`JointTrajectory`, 라디안).
- `rh56f1_hand_bridge` — RH56F1 양손 (`SetAngle1`, 레지스터 정수). 순수 변환은 `rh56f1_hand.py`,
  캘리브레이션은 `config/rh56f1_hand_calibration.yaml`. 로봇별 연동 상세는
  `../ROBOT_ISAACSIM_CONNECTION.md` 참조.

RH56F1 세팅에서는 `bridge_node`를 `right_hand_enabled:=false`(Tesollo 경로 끔) +
`extra_joint_state_topics:=[/rh56f1/joint_states]`(손 상태 병합)로 띄우고, `rh56f1_hand_bridge`를
함께 실행합니다.

## 역할

- Isaac Sim 또는 외부 제어기(FABRICS)가 발행한 `/isaacsim/*` 명령을 실제 하드웨어 제어기로 전달
- OpenArm + Tesollo joint state를 병합해서 `/isaacsim/joint_states` 로 재발행
- 실기 상태와 Sim shadow 상태를 비교하는 오차 기록 지원
- 튜닝용 리포트 / 다음 drive config 생성 지원

## 관련 파일

### 이 패키지 내부

- `ISAACSIM_ACTION_GRAPH.md`
- `ISAACSIM_POLICY_WIRING.md`
- `scripts/create_action_graph.py`
- `scripts/create_sim_joint_state_publish_graph.py`
- `scripts/create_ros_joint_state_publish_graph.py`
- `scripts/tune_shadow_joint_drives.py`
- `scripts/apply_joint_drive_config.py`

### FABRICS 연동용 보조 스크립트

이 패키지 밖에 있지만 같이 쓰는 파일들입니다.

- `../scripts/fabrics_ros_interface.py`
- `../scripts/file_command_transport.py`
- `../scripts/nodes/file_command_relay.py`
- `../scripts/nodes/manual_command_pub.py`

## 브리지 입력 토픽

Isaac Sim 또는 FABRICS가 아래 토픽으로 명령을 넣습니다.

- `/isaacsim/left_arm_cmd`
  - 타입: `std_msgs/Float64MultiArray`
  - 길이: 7
- `/isaacsim/right_arm_cmd`
  - 타입: `std_msgs/Float64MultiArray`
  - 길이: 7
- `/isaacsim/left_gripper_cmd`
  - 타입: `std_msgs/Float64`
  - 길이: 스칼라 1개
- `/isaacsim/right_hand_cmd`
  - 타입: `std_msgs/Float64MultiArray`
  - 길이: 20
- `/isaacsim/emergency_stop`
  - 타입: `std_msgs/Bool`

## 브리지 출력 대상

- `/left_joint_trajectory_controller/joint_trajectory`
- `/right_joint_trajectory_controller/joint_trajectory`
- `/left_gripper_controller/gripper_cmd`
- `/dg5f_right/dg5f_right_controller/joint_trajectory`

## 상태 병합

브리지는 아래 토픽을 구독합니다.

- `/joint_states`
- `/dg5f_right/joint_states`

그리고 병합된 실기 상태를 아래로 재발행합니다.

- `/isaacsim/joint_states`

## 안전 기능

- 모든 출력 명령은 설정된 범위 안으로 clamp
- `/isaacsim/emergency_stop` 가 `true` 이면 모든 출력 명령 중단

## 설치 / 빌드

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
cd "${REPO_DIR}"
./scripts/build_vendor_pkgs.sh
source "${REPO_DIR}/install/setup.bash"
```

브리지만 다시 빌드할 때:

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
cd "${REPO_DIR}"
./scripts/build_vendor_pkgs.sh --bridge-only
source "${REPO_DIR}/install/setup.bash"
```

## FABRICS 연동 방법

### 1) FABRICS가 `rclpy` 를 직접 import 할 수 있는 경우

추가 패키지 빌드는 필요 없습니다. 위 기본 빌드만 하면 됩니다.

FABRICS Python 루프에서 `../scripts/fabrics_ros_interface.py` 를 import 해서 직접 `/isaacsim/*` 토픽으로 발행하면 됩니다.

사용 클래스:

- `Sim2RealCommandPublisher`

주요 메서드:

- `send_left_arm(values)`
- `send_left_gripper(value)`
- `send_right_arm(values)`
- `send_right_hand(values)`
- `send_left_full(arm, gripper)`
- `send_right_full(arm, hand)`

즉, FABRICS가 같은 ROS 2 Python 환경에 있으면 이 방식이 가장 단순합니다.

### 2) FABRICS가 별도 Python 환경이라 `rclpy` 를 import 할 수 없는 경우

이 경우 파일 기반 중계 방식을 씁니다.

구성:

1. FABRICS 프로세스
   - `../scripts/file_command_transport.py` 사용
   - 기본 파일: `/tmp/sim2real_cmd.json`
2. ROS 2 프로세스
   - `../scripts/nodes/file_command_relay.py` 실행
   - 파일 내용을 읽어 `/isaacsim/*` 토픽으로 재발행

중계기 실행:

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
python3 "${REPO_DIR}/scripts/nodes/file_command_relay.py"
```

기본 공유 파일:

- `/tmp/sim2real_cmd.json`

## 수동 명령 테스트

FABRICS 없이 토픽만 빠르게 테스트할 때:

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
python3 "${REPO_DIR}/scripts/nodes/manual_command_pub.py" left-arm 0 0 0 0 0 0 0
```

예시:

- 왼팔만:

```bash
python3 "${REPO_DIR}/scripts/nodes/manual_command_pub.py" left-arm 0 0 0 0 0 0 0
```

- 왼쪽 그리퍼:

```bash
python3 "${REPO_DIR}/scripts/nodes/manual_command_pub.py" left-gripper 0.015
```

- 오른팔만:

```bash
python3 "${REPO_DIR}/scripts/nodes/manual_command_pub.py" right-arm 0 0 0 0 0 0 0
```

## 브리지 실행

### 브리지만 실행

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 launch isaacsim_bridge isaacsim_bridge.launch.py
```

### 브리지 + 실기 하드웨어 같이 실행

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 launch isaacsim_bridge isaacsim_bridge.launch.py \
  with_hardware:=true \
  left_can_interface:=can1 \
  right_can_interface:=can0 \
  dg5f_right_ip:=169.254.186.72 \
  dg5f_right_port:=502
```

### 브리지 + 실기 + 오차 기록기 같이 실행

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 launch isaacsim_bridge isaacsim_bridge.launch.py \
  with_hardware:=true \
  with_recorder:=true \
  recorder_output_path:=/tmp/isaacsim_joint_error.csv
```

## Isaac Sim 설정

### 1) 명령 입력 Action Graph

Isaac Sim Script Editor에서 아래 파일을 열어 실행합니다.

- `scripts/create_action_graph.py`

이 그래프는 `/isaacsim/*` 입력 토픽으로 명령을 내보내는 예제 Action Graph를 만듭니다.

### 2) Sim shadow joint state 퍼블리시

Isaac Sim Script Editor에서 아래 파일을 열어 실행합니다.

- `scripts/create_sim_joint_state_publish_graph.py`

이 그래프는 Sim shadow 로봇의 joint state를 아래 토픽으로 발행합니다.

- `/isaacsim/sim_joint_states`

### 3) 실기 상태 퍼블리시 그래프

Isaac Sim Script Editor에서 아래 파일을 열어 실행합니다.

- `scripts/create_ros_joint_state_publish_graph.py`

이 그래프는 Isaac Sim articulation joint state를 ROS 2 쪽으로 발행합니다.

### 4) 기본 drive 강하게 설정

Isaac Sim Script Editor에서 아래 파일을 열어 실행합니다.

- `scripts/tune_shadow_joint_drives.py`

용도:

- shadow robot이 명령 자세를 더 강하게 유지하도록 기본 stiffness / damping 증가

### 5) 자동 튜닝 결과 적용

Isaac Sim Script Editor에서 아래 파일을 열어 실행합니다.

- `scripts/apply_joint_drive_config.py`

기본 입력 파일:

- `/tmp/isaacsim_next_joint_drive_config.json`

## 오차 기록 및 튜닝

### 1) 실기-시뮬레이터 오차 기록

실기 `/isaacsim/joint_states` 와 Sim shadow `/isaacsim/sim_joint_states` 를 비교해서 CSV를 기록합니다.

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 run isaacsim_bridge joint_error_recorder \
  --ros-args \
  -p real_joint_states_topic:=/isaacsim/joint_states \
  -p sim_joint_states_topic:=/isaacsim/sim_joint_states \
  -p output_path:=/tmp/isaacsim_joint_error.csv
```

출력:

- `/tmp/isaacsim_joint_error.csv`
- `/isaacsim/joint_error_summary`

### 2) 튜닝 리포트 생성

기록된 CSV를 기반으로 joint별 오차 통계와 튜닝 힌트를 생성합니다.

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 run isaacsim_bridge joint_tuning_report -- \
  --input /tmp/isaacsim_joint_error.csv \
  --output /tmp/isaacsim_joint_tuning_report.json
```

출력:

- `/tmp/isaacsim_joint_tuning_report.json`

### 3) 자동 1회 튜닝 사이클

오차 CSV를 읽어서:

- 리포트 JSON 생성
- 다음 stiffness / damping / offset 제안 JSON 생성

```bash
source /opt/ros/humble/setup.bash
REPO_DIR="/path/to/sim2real_control"
source "${REPO_DIR}/install/setup.bash"
ros2 run isaacsim_bridge joint_tuning_cycle -- \
  --input-csv /tmp/isaacsim_joint_error.csv \
  --output-report /tmp/isaacsim_joint_tuning_report.json \
  --output-drive-config /tmp/isaacsim_next_joint_drive_config.json
```

출력:

- `/tmp/isaacsim_joint_tuning_report.json`
- `/tmp/isaacsim_next_joint_drive_config.json`

### 4) 권장 튜닝 순서

1. Isaac Sim에서 `create_action_graph.py` 실행
2. Isaac Sim에서 `create_sim_joint_state_publish_graph.py` 실행
3. ROS 2에서 `isaacsim_bridge.launch.py` 실행
4. FABRICS 또는 수동 명령으로 `/isaacsim/*` 토픽에 명령 입력
5. `joint_error_recorder` 로 오차 기록
6. `joint_tuning_cycle` 로 다음 drive config 생성
7. Isaac Sim에서 `apply_joint_drive_config.py` 실행
8. 반복 측정

## 참고

- `ISAACSIM_ACTION_GRAPH.md`: Action Graph 구성 상세
- `ISAACSIM_POLICY_WIRING.md`: 정책/FABRICS 출력을 Action Graph에 연결하는 방식
