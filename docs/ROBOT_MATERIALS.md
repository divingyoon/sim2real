# 로봇용 자료 매니페스트 (연결 시 바로 사용)

로봇 제어 PC + 카메라(D435i)를 연결했을 때 grasp-v1 sim2real 이 "바로 사용"
가능하도록, 필요한 로봇용 자료(정책 체크포인트, 드라이버/vendor, launch)의
실제 경로와 perception ↔ sim2real ↔ hdgp/log 디렉토리 연계를 정리한다.
(hdgp 는 READ-ONLY 로 조사만 함. 이 문서 자체는 sim2real 저장소에만 기록.)

---

## 1. 정책 산출물 (hdgp/log → sim2real)

- **grasp-v1 체크포인트(.pth)**:
  `/home/user/rl_ws/hdgp/log/rl_games/open-tesol/right/grasp-v1/lstm_test1/nn/FINAL_frozen_ep1000_abd600.pth`
  → sim2real 실행 시 `--ckpt` 로 지정.
  - 같은 디렉토리에 중간 체크포인트도 존재: `last_..._ep_500_rew_7460.3354.pth`,
    `last_..._ep_1000_rew_11157.004.pth`, `open-tesol_r_grasp_v1-lstm.pth`.
    `FINAL_frozen_ep1000_abd600.pth` 가 가장 최신(2026-07-01 14:15)이며 이름상
    "frozen" 배포용 산출물이라 이것을 기본값으로 채택.
- **agent.yaml(params)**:
  `/home/user/rl_ws/hdgp/log/rl_games/open-tesol/right/grasp-v1/lstm_test1/params/agent.yaml`
  → `--agent` 로 지정. `network.rnn.name: lstm, units: 1024, layers: 1` /
  `network.mlp.units: [512, 512]` — `RLGamesLstmActorPolicy` 대상(MLP 전용
  `RLGamesActorPolicy` 아님).
- **obs/action 차원**: `hdgp/source/openarm/openarm/tesollo/right/grasp_v1/grasp_right_constants.py`
  기준 `NUM_OBSERVATIONS = 106`(actor, sim2real 가능), `NUM_ACTIONS = 11`
  (palm 6D + finger lerp 5D). `policy_loader.py` 호출 시 `obs_dim=106,
  action_dim=11` 로 맞춰야 한다(문서/코드에 있는 `obs_dim=55, action_dim=12`
  예시는 다른 태스크용 예시이므로 grasp-v1 에는 적용하지 말 것).
- **⚠️ 기존 문서 경로 불일치 발견**: 저장소 내 기존 `SIM2REAL_INFERENCE.md` 와
  `scripts/deprecated/sim2real_inference.py` 상단 docstring 은 체크포인트 경로를
  `hdgp/log/rl_games/pipeline/right/5g_grasp_right_v7/test4/{nn/5g_grasp_right-v7.pth,
  params/agent.yaml}` 로 기술하고 있으나, 이 경로는 **현재 hdgp 에 존재하지
  않는다**(`hdgp/log/rl_games/pipeline/` 디렉토리 자체가 없음 — 실측 확인
  완료, 2026-07-22 기준). 학습 산출물이 이후 `open-tesol/right/grasp-v1/
  lstm_test1/` 구조로 재정리된 것으로 보인다. `5g_grasp_right_v1`(≒v7,
  Fabrics 팔 학습 + per-finger lerp) 이라는 태스크 이름 자체는
  `hdgp/source/openarm/openarm/tesollo/right/grasp_v1/*.py` 모듈 docstring
  들과 일치하므로 **같은 정책 계열**이 맞다. 실행 전 반드시 위 실제 경로를
  사용하고, 옛 문서의 `AGENT=`/`CKPT=` 변수 값은 그대로 복붙하지 말 것.
  `hdgp/outputs/` 아래에는(2026-07-16~07-21 날짜 디렉토리들) grasp-v1 관련
  산출물이 없음을 확인함(모두 이후 distillation/grasp-v2 실험).
- **연계**: hdgp 학습 산출물(READ-ONLY). `sim2real_inference.py` /
  `sim2real_dryrun.py` 가 `policy_loader.py` 를 통해 로드. sim2real 로 복사
  하거나 절대경로로 그대로 참조(같은 PC 라면 절대경로 참조가 더 간단).

---

## 2. 로봇 드라이버/vendor (로봇 제어 PC, clone 대상)

**2026-07-27 갱신**: Tesollo 드라이버는 robot_control 로 일원화되어 이
저장소에서 제거됐다. OpenArm 드라이버는 아직 여기 남아 있다 — 아래 §
"OpenArm 이관 보류" 참고.

OpenArm vendor 소스는 별도의 "로봇제어 레포"가 아니라 **이 sim2real 저장소
안에 git 추적 파일로 커밋되어 있다**(`vendor/openarm`, 서브모듈 아님,
`git ls-files vendor | wc -l` → 887개 추적 파일; 예외로
`vendor/openarm/openarm_teleop/` 만 `.gitignore` 처리). 즉 로봇 제어 PC에는
별도 레포를 clone 할 필요 없이 **이 sim2real 저장소 자체를 배치하고
colcon build** 하면 된다(`scripts/build_vendor_pkgs.sh` 참고, `INSTALL.md`
Step 1~3,5 = control 역할).

- **OpenArm 팔 드라이버**: `vendor/openarm/{openarm_description, openarm_can,
  openarm_hardware, openarm_bringup}` (ros2_control, CAN 인터페이스).
  Launch: `sim2real/openarm_control/launch/openarm_left_gripper_bimanual_real.launch.py`
  (내부적으로 `openarm_description`/`robot_state_publisher`/
  `ros2_control_node` 기동). 양팔 통합 launch는
  `sim2real/integrated_control/launch/openarm_left_gripper_right_dg5_real.launch.py`.
  - 발행: `/joint_states` (arm 7D, `openarm_right_joint1~7`)
  - 구독(명령): `/right_joint_trajectory_controller/joint_trajectory`
- **Tesollo dg5f_right 드라이버**: `robot_control/ros_ws/src/delto_m_ros2/`
  (`dg_description`, `dg_msgs`, `delto_tcp_comm`, `delto_hardware`,
  `dg5f_driver`). robot_control이 canonical 스냅샷을 소유하며, 여기서는
  빌드된 install을 오버레이한다 — `ROBOT_CONTROL_INSTALL` 로 경로 지정 가능.
  Launch: `sim2real/tesollo_control/launch/dg5f_right_real.launch.py`
  (기본 `dg5f_right_ip=169.254.186.72`, `dg5f_right_port=502`, Ethernet/Modbus).
  - 발행: `/dg5f_right/joint_states` (hand 20D, `rj_dg_1_1~rj_dg_5_4`),
    `/dg5f_right/contact_forces` (FT force 5D, `Float64MultiArray`)
  - 구독(명령): `/dg5f_right/dg5f_right_controller/joint_trajectory`
    (position target → 내부 PD `p=1.5, d=0.0` 로 effort 변환 — 시뮬 게인과
    다르므로 실기에서 게인 재조정 필요할 수 있음, `SIM2REAL_INFERENCE.md` 참고)
- **isaacsim_bridge**: `sim2real/isaacsim_bridge/` (colcon 패키지,
  `package.xml` 보유). Launch: `isaacsim_bridge/launch/isaacsim_bridge.launch.py`.
  - 구독: `/isaacsim/right_arm_cmd` (Float64MultiArray 7D),
    `/isaacsim/right_hand_cmd` (Float64MultiArray 20D) — `sim2real_inference.py`
    / `sim2real_dryrun.py` 가 `Sim2RealCommandPublisher` 로 발행하는 것을
    받아 위 컨트롤러 토픽으로 중계.
- **빌드**: robot_control `ros_ws/build.sh` 를 먼저 돌린 뒤
  `scripts/build_vendor_pkgs.sh` 가 `vendor/openarm` 4개와 `isaacsim_bridge`
  를 빌드한다(`--bridge-only` 옵션은 isaacsim_bridge 만). 스크립트는
  robot_control install 을 오버레이로 source 하며, 없으면 무엇을 해야 하는지
  말하고 종료한다. `scripts/setup_check.sh control` 로 사전 점검.
- **위치**: 로봇제어 PC 에는 **sim2real 과 robot_control 을 나란히** 배치한다
  (기본 탐색 경로 `../robot_control/ros_ws/install`, `ROBOT_CONTROL_INSTALL`
  로 변경 가능). Tesollo 드라이버가 robot_control 로 옮겨간 만큼 sim2real 은
  더 이상 자기 완결적이지 않다.

---

## 3. 비전 (perception → sim2real)

- `/home/user/rl_ws/perception` 이 컵 6-DOF pose 를 생산한다. 두 모드가 계약
  동일(`/cup_pose`, `geometry_msgs/PoseStamped`, robot base frame):
  - **현행/레거시**: Isaac ROS FoundationPose(NITROS, pseudo-tracking) —
    arm3070 PC 에 배포·동작 중.
  - **베이스(예정)**: FoundationPose++ (FP + Cutie tracker + KalmanFilter6D,
    움직이는 컵 추적) — vision-3090 에 스택 설치 완료, 라이브 노드는 Phase 2.
- **relay**: `sim2real/scripts/cup_pose_relay.py` 가 perception 원출력
  (`vision_msgs/Detection3DArray`, camera optical frame, 컵 CAD 프레임)을
  `--extrinsics config/global_camera_extrinsics.yaml` 로 `T_base_cam ∘
  T_cam_cad ∘ T_cad_body` 합성해 `/cup_pose` 로 재발행.
  - `config/global_camera_extrinsics.yaml` 의 `camera.position/orientation_wxyz`
    는 현재 **PLACEHOLDER**(`[0,0,0]`/identity, 파일 내 주석
    "TODO(calibration): ... 이 상태로 실기 구동 금지") — 카메라 장착 후
    `tools/calibrate_extrinsics.py`(perception, ArUco)로 1회 캘리브레이션
    필요(범위 밖, 하드웨어 대기).
  - 소비자: `sim2real_inference.py`(grasp 단계에서 실시간 구독),
    `pour_inference.py`(pour 는 FK 라 1회 capture 후 relay 죽어도 계속 동작).
- **ROS_DOMAIN_ID=126** 를 perception PC / sim2real PC / 로봇제어 PC 모두
  동일하게 export 해야 DDS 로 서로 보인다(`INSTALL.md` Step 2,
  `scripts/setup_check.sh` 에서 확인).
- **점검 스크립트**: `sim2real/scripts/check_cup_pose_link.sh [domain_id]`
  (기본 126) — `ROS_DOMAIN_ID` 일치 여부 + `ros2 topic list` 에 `/cup_pose`
  가시 여부를 각 PC 에서 실행해 확인.

---

## 4. 디렉토리 유기적 연계 (데이터 흐름)

```
perception(/cup_pose, ROS_DOMAIN_ID=126) ──────────────┐
                                                        │
hdgp/log(grasp-v1 .pth + agent.yaml, READ-ONLY) ───────┼──▶ sim2real
                                                        │    (cup_pose_relay.py,
vendor/{openarm,tesollo} + openarm_control/             │     policy_loader.py,
tesollo_control/ + isaacsim_bridge (sim2real 내부) ─────┘     sim2real_inference.py,
                                                              sim2real_dryrun.py)
                                                                    │
                                                                    ▼
                                        로봇 드라이버 (OpenArm CAN / Tesollo dg5f_right Ethernet)
                                        via isaacsim_bridge → ros2_control 컨트롤러
```

- **sim2real = 런타임 글루 디렉토리.** 정책은 `hdgp/log`(READ-ONLY,
  실측상 `hdgp/log/rl_games/open-tesol/right/grasp-v1/lstm_test1/`), 비전은
  `perception`(`/cup_pose` 계약), **Tesollo 드라이버는 robot_control**
  (오버레이로 source), OpenArm 드라이버는 아직 `sim2real/vendor/openarm`.
  그 위에 얹힌 control 패키지(`openarm_control`, `tesollo_control`,
  `integrated_control`, `isaacsim_bridge`)는 sim2real 안에서 colcon build.
- 세 디렉토리(perception / hdgp / sim2real)는 서로 다른 PC 에 있을 수
  있으나 **ROS_DOMAIN_ID=126 공유 DDS** 하나로 연결된다(perception →
  `/cup_pose`, sim2real → 컨트롤러 명령 토픽). hdgp 는 네트워크 연결이
  아니라 **파일(체크포인트) 참조**로만 연결된다(런타임에 hdgp 프로세스가
  떠 있을 필요 없음).

---

## 5. 연결 시 체크리스트

- [ ] 정책 `.pth`(`hdgp/log/rl_games/open-tesol/right/grasp-v1/lstm_test1/nn/
      FINAL_frozen_ep1000_abd600.pth`) + `agent.yaml`(같은 트리 `params/`)을
      sim2real 실행 커맨드의 `--ckpt`/`--agent` 로 지정(경로 그대로 참조
      가능, 복사 불필요).
- [ ] 로봇 드라이버: 로봇제어 PC 에 sim2real 과 robot_control 을 나란히 배치 →
      robot_control `ros_ws/build.sh` 먼저 →
      `scripts/build_vendor_pkgs.sh` 로 `vendor/openarm`,
      `isaacsim_bridge` colcon build → `scripts/setup_check.sh control` 로
      CAN 인터페이스/vendor 빌드 확인 → `openarm_control` +
      `tesollo_control`(또는 `integrated_control` 통합) + `isaacsim_bridge`
      launch 기동.
- [ ] perception `/cup_pose` 기동(FoundationPose 현행 모드 or FP++) +
      모든 PC `ROS_DOMAIN_ID=126` 확인 (`scripts/check_cup_pose_link.sh`).
- [ ] extrinsics 캘리브(`T_base_cam`, 현재 `config/global_camera_extrinsics.yaml`
      은 PLACEHOLDER) — 카메라 장착 후 perception `tools/calibrate_extrinsics.py`
      (ArUco)로 1회 수행, 이전에는 실기 구동 금지.

---

## 범위 밖 (하드웨어 대기)

- `T_base_cam` extrinsics **캘리브레이션** — 카메라 장착 후(perception
  `tools/calibrate_extrinsics.py`, ArUco).
- FP++ **라이브 ROS 노드**(camera → `/cup_pose`) — vision-3090 D435i +
  py3.8 ↔ Humble rclpy 브리지(별도 결정 대기).
- 감독 하 **라이브 grasp 동작** 자체(grasp-v1 실기 실행) — 이 문서는 자료
  위치 매니페스트일 뿐, 실행 검증은 하드웨어 연결 후 별도 태스크.

## OpenArm 이관 보류 (2026-07-27)

Tesollo 5개 패키지는 robot_control 사본과 `diff -rq` 차이 0줄로 완전
동일했고(`dg_description` 만 mesh URI 표기가 달랐으나 이 저장소의 비-vendor
코드가 참조하지 않음), 제거 후 robot_control install 오버레이로 대체했다.

OpenArm 4개는 같이 옮기지 않았다. 두 사본이 각자 진화했고, 특히
**ros2_control 하드웨어 플러그인 클래스명이 다르다**:

| | 플러그인 |
|---|---|
| sim2real | `openarm_hardware/OpenArm_v10HW` |
| robot_control | `openarm_hardware/OpenArmHW` |

robot_control 로 넘어가려면 실물 bringup 이 쓰는 xacro 가 새 이름을 지목해야
한다. 그런데 `integrated_control/launch/openarm_left_gripper_right_dg5_real.launch.py`
가 참조하는 `urdf/openarm_left_gripper_bimanual_real.xacro` 는 현재
**존재하지 않는다**(`urdf/` 를 `*_rl` 구조로 재편할 때 갱신되지 않음). 이
경로를 복구하면서 플러그인 이름을 함께 맞추는 것이 이관의 선행 조건이다.

`openarm_can` 의 `set_temp_param` 시그니처도 `int`→`double` 로 바뀌었으나,
호출부는 vendor 내부 2곳뿐이라 패키지를 통째로 교체하면 함께 이동한다.
