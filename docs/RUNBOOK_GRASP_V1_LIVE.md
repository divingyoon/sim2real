> ## ⚠️ 2026-08-18 갱신 — 먼저 읽을 것
>
> **머신 구성이 바뀌었다.** 이 문서 본문은 구 2-PC 전제(vision-3090 정책 + 5070ti 로봇)로
> 쓰여 있다. 현행은:
>
> | 머신 | 역할 |
> |---|---|
> | **local 5090** | 로봇제어 + 정책추론 (CAN·손 이더넷·robot_control·브리지·grasp_inference) |
> | **vision-3090** | 비전 전용 (RealSense + FP++ → `/cup_pose`) |
>
> 아래 §A(로봇)·§B(정책)는 **같은 머신(5090)** 에서 돌린다. 컵 pose 만 3090 에서 온다.
>
> ### s2r 1차 대상 = `grasp_sensor` (2026-08-18 결정)
>
> 실기 구성이 **우 DG-5F + 좌 2지 그리퍼**(`openarm_tesollo_sensor_rl`)이므로, 라이브가
> 실제로 가능한 구성은 `grasp_sensor` 뿐이다. `grasp_v1`(bi_s, 양손 DG-5FS)은 좌 Tesollo
> 미장착이라 오프라인 검증까지만 가능하다.
>
> ```
> --robot tesollo_sensor__right     # ← 1차 대상 (실기 구성)
> --robot tesollo_bi_s__right       # 좌손 장착 후
> ```
>
> 계약(obs 154D / action 21D / 고정 홈 / 축별 palm delta)은 두 구성이 **동일**하고
> **자산만 다르다**(Fabrics `openarm_tesollo` vs `openarm_tesollo_bi_s`, palm 6.5cm 차이).
> 그래서 q_home 도 다르다 — 프로필을 섞어 쓰면 조용히 틀린다.
> 구성별 계약: `docs/CONTRACT_grasp_sensor_right.md`, `docs/CONTRACT_grasp_v1_{right,left}.md`
>
> **★기동 전 유휴(좌)팔을 rest 자세로 두어야 한다.** sim 은 유휴 팔을 rest 로 고정한 채
> 학습했고 그 팔은 물리 충돌체다. `/grasp/start` 가 0.15 rad 초과 시 거부한다.
> `grasp_sensor` 좌팔 rest = `[-0.0431, -0.6706, -0.0961, +0.7342, -0.3750, -0.5678, -0.6709]`
> (그리퍼 0.044 개방). robotctl 로 옮긴 뒤 start 할 것.
>
> **계약도 바뀌었다** — obs 114D→**154D**, action 11D→**21D**, 리셋은 컵 참값 pregrasp →
> **고정 홈**. 접촉 토픽은 5D norm → **15D 3축**(`<tip_force_xyz>`).
> 계약 전문: `docs/CONTRACT_grasp_v1_{right,left}.md`
> 측정·격차 기록: `docs/measure/ACTION_CONTROL_DEMAND.md`
>
> **모든 노드가 `--robot <구성 프로필>` 을 받는다**(`config/robots/*.yaml`).
> 좌/우와 손·그리퍼 조합이 여기서 결정된다. 기본값 `tesollo_bi_s__right`.
>
> **체크포인트 경로는 아직 확정 전이다** — 본문의 `lstm_test3` 경로는 **존재하지 않는다**.
> 재학습(`lstm_test2`) 완료 후 갱신할 것.

# grasp-v1 라이브 실행 런북 (2026-08-03 재정립)

tesollo right grasp-v1 라이브 정책의 머신별 기동 명령. 커밋 62909aa(zeros 손 obs 방어 +
tip_contact_pub) 기준. 토폴로지: **vision-3090**(정책+지각) ↔ **5070ti robot PC**(브리지+드라이버),
유선 DDS(도메인 126).

---

## 0. 공통 규칙 — 모든 ROS 터미널(두 머신 공통)

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=126
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/fastdds_wired.xml
```

- ⚠️ **한 프로세스라도 프로파일을 안 물면 DDS 데이터가 단방향 불통**(discovery만 되고 콜백 0).
  의심 시 검증: `tr '\0' '\n' < /proc/<PID>/environ | grep FASTRTPS`
- 노드/브리지 **재기동 전 옛 프로세스 pkill 필수**(중복 2대가 두 번 사고 남).
- 유선 정적 IP: vision-3090 `192.168.100.1` ↔ 5070ti `192.168.100.2`.

### 0-1. 배선/네트워크 사전검사 (★매 세션 첫 순서 — 랜 오배선 재발 방지)

08.03 사고: 두 NIC 의 IP 설정이 물리 배선과 교차되어 손(169.254.186.72) 패킷이 vision 랜선으로
나감 → 손 제어 전면 불통(코드 무죄). 확정 배선(5070ti):

| NIC | 물리 연결 | IP | NM 프로파일 |
|---|---|---|---|
| `enp0s31f6` (메인보드) | vision-3090 직결 | 192.168.100.2/24 | `vision-link` |
| `enxb0386cf2c43a` (USB 허브) | 손 DG-5F | 169.254.186.100/24 | `hand-link` |

```bash
bash ~/rl_ws/sim2real/scripts/net_preflight.sh   # PASS 전까지 스택 기동 금지
```

교차/IP 교차 검사 + 인터페이스 강제 ping(정방향·역방향)으로 배선 정합을 판정한다.
NM 프로파일 영구화(1회, sudo 필요 — 아래 실행 후엔 재부팅에도 유지):

```bash
sudo nmcli con delete "Wired connection 1" "Wired connection 2"   # 교차 잔재 제거
sudo nmcli con modify lab-link connection.id hand-link connection.autoconnect yes \
  connection.autoconnect-priority 100 ipv4.never-default yes ipv6.method disabled
sudo nmcli con add type ethernet ifname enp0s31f6 con-name vision-link ipv4.method manual \
  ipv4.addresses 192.168.100.2/24 ipv4.never-default yes ipv6.method disabled \
  connection.autoconnect yes connection.autoconnect-priority 100
sudo nmcli con up vision-link && sudo nmcli con up hand-link
```

- USB NIC 이름 `enxb0386cf2c43a`는 MAC 유래라 포트를 옮겨도 불변(어댑터 교체 시에만 갱신 필요).
- vision-3090 쪽 `192.168.100.1`도 동일하게 nmcli 영구화 필요(수동 `ip addr`는 재부팅 시 증발).
- 케이블 양단 물리 라벨링 권장: "HAND-USB" / "VISION-MOBO".

---

## A. 5070ti robot PC (`usr@100.106.38.98`)

워크스페이스 소싱: `source ~/rl_ws/robot_control/ros_ws/install/setup.bash` (§0 다음에).

### A1. 전원 인가 후 CAN

```bash
sudo openarm-can-configure-socketcan-4-arms -fd
# ★인터페이스 인자 없음 — can0~can3 내장(usage 문구는 구버전 오기). can0 넣으면 Unknown option.
#  없는 can2 에서 not found 로 멈추는 건 무해(can0/can1 은 이미 설정됨). 확인: ip -br link show can0
```

### A2. 오른팔 bringup

```bash
ros2 launch openarm_bringup openarm.bimanual.launch.py use_fake_hardware:=false
```

- j7 주의: 손(~1.4kg) 부착 시 고피치 홀딩 과부하(7Nm 한계) — bringup만으로 15s 내 red 가능.
  저토크 자세 유지 or `robotctl pose gravity`(중력FF, `--urdf` 손질량 포함).

### A3. 손(dg5f) — ★Phase 1 복구 절차 포함

0. **★모드 스위치/LED 확인**(delto_m_ros2/dg5f_driver/README "Before You Control" + images/manual.png):
   ros2 드라이버는 **Developer Mode 전용** — 스위치 **②(Developer)+④(EtherNET)** 여야 함.
   부팅 LED로 판별: 전체 LED **2회 깜빡=Developer(정상)** / 1회=Operator(잘못, Modbus 내장모드
   — 관절정보 수신 비활성이라 GET_DATA 가 전부 0으로 옴) / 빨간 LED 깜빡=스위치·통신 이상.
   파란 LED 켜짐=소켓 연결됨. **분리→재부착 작업 후 이 스위치부터 볼 것.**
1. **손 전원 재인가** (Modbus 세션/펌웨어 꼬임 리셋 — 관절 0.000 고정 증상의 처방)
2. 드라이버 기동(**F/T 브로드캐스터 포함**):
   ```bash
   ros2 launch dg5f_driver dg5f_right_driver.launch.py delto_ip:=169.254.186.72 fingertip_sensor:=true
   ```
3. 검증(순서대로, 실패 시 벤더 테스트 스크립트부터):
   ```bash
   ros2 topic echo /dg5f_right/joint_states --once        # 0.000 고정이면 실패
   ros2 topic echo /fingertip_1_broadcaster/wrench --once  # F/T 나오는지
   ```

### A4. tip 접촉 변환 노드 (★손끝 무접촉 상태에서 기동 — bias 캡처)

```bash
python3 ~/rl_ws/sim2real/scripts/nodes/tip_contact_pub.py --robot tesollo_sensor__right
# bias 재캡처(무접촉 자세에서): ros2 service call /tip_contact/rebias std_srvs/srv/Trigger
# 검증: ros2 topic echo /dg5f_right/tip_forces_xyz --once  → 15D(5×3), 무접촉 ~0,
#       컵 누르면 해당 tip 3축이 움직임. (5D norm 은 /dg5f_right/contact_forces 로 병행 발행)
```

- 임계 0.1N(CONTACT_FORCE_THRESHOLD)은 실물 노이즈 대비 튜닝 필요(무접촉 노이즈 < 0.1 확인).

### A5. 브리지

```bash
pkill -f isaacsim_cmd_to_jtc; sleep 1
python3 ~/rl_ws/sim2real/scripts/nodes/isaacsim_cmd_to_jtc.py --robot tesollo_sensor__right --max-vel 0.1
```

- max_vel 0.1은 후퇴 원인 아님(08.03 재현 실험으로 무죄 판정) — 그대로 시작, 추종지연 체감 시 상향.
- 손은 `--hand-max-vel`(기본 1.0 rad/s)로 **팔과 분리** — 공용 0.1을 손에 쓰면 APPROACH 도달
  15.7s 로 settle 게이트(수 s) 오탐 + 손가락 폐쇄가 기어감(08.03 실측 thumb_2=-0.408=0.1×4s).
- 정책 명령 **유휴 1s 후 발행 자동 중지**(CMD_TIMEOUT_SEC, b1755e8) — JTC 가 마지막 포인트를
  자체 홀딩하므로 에피소드 밖에선 robotctl 수동조작이 브리지 켠 채로 가능. 스트림 재개 시
  세트포인트는 실측 위치에서 재시작(robotctl 로 옮긴 자세와 점프 없음).
  (구판 브리지는 60Hz 무한 재발행이라 robotctl 이 매 tick 덮였음 — "EXECUTED 인데 무동작" 증상)

### A6. 관측(권장, 매 실행 병행)

```bash
python3 ~/rl_ws/sim2real/scripts/analysis/joint_monitor.py    # 로그 sim2real/logs/*.csv
```

---

## B. vision-3090 (`ssh vision-3090`)

### B1. 컵 pose — 기본은 실컵(perception_plus_plus)

(a) **실컵** (FP++ 라이브, 기본):
```bash
# 터미널1: 목 고정(캘리브 자세 pan -90 / tilt 280 재현 필수 — extrinsics 유효 조건)
python3 ~/rl_ws/sim2real/scripts/head_position_hold_node.py
# 터미널2: RealSense (카메라 점유 — 컨테이너보다 먼저)
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
# 터미널3: FP++ 컨테이너 + relay 일괄 (컨테이너 fpp_cup + /tmp/run_relay.sh detached)
~/rl_ws/perception_plus_plus/scripts/run_cup_pose_live.sh
# 터미널3-1: ★상태 워처 상시 유지 — 🟢 8Hz 정상 / 🔴 STALL=YOLO 검출 손실(가림·시야밖)
python3 ~/rl_ws/sim2real/scripts/vision/cup_pose_watch.py
# 터미널3-2: (GUI) RGB 라이브 + 추적 오버레이 — 영상 위 초록 십자선=추적 중
python3 ~/rl_ws/sim2real/scripts/vision/cup_view.py
```

- **첫 검출까지 ~50s 걸림**(실측 51s) — 그 전에 start 하면 "미수신 /cup_pose" 거부(정상).
- 추적을 잃으면 YOLO 재검출로 재-앵커해야 재개 — 컵이 가려지면(팔 포함) 발행이 조용히
  정지하므로 워처(🔴 STALL)로만 알 수 있다. 에피소드 중 정지 시 정책은 staleness 홀드.
- 컵 체인(컨테이너→relay→grasp_inference)은 **전부 vision-3090 내부** → SHM 전송으로
  fastdds_wired 프로파일 유무 섞여도 통신됨(크로스머신인 팔 명령 경로만 프로파일 일치 필수).
- 컵 배치: 오른팔 작업권 앞-우측(base 기준 x 0.35~0.45, y −0.20~0.00 권장) + 카메라 FOV 내.
  로그 검증: `/cup_pose` 값이 자로 잰 실측과 ~cm 수준 일치하는지 1회 확인.

(b) fake (플러밍 테스트 전용):
```bash
python3 ~/rl_ws/sim2real/scripts/fakes/fake_cup_pose_pub.py --x 0.40 --y -0.15 --z 0.38
```

### B2. (손 분리 테스트 시에만) fake 손

```bash
python3 ~/rl_ws/sim2real/scripts/fakes/fake_hand_state_pub.py          # APPROACH 정적
python3 ~/rl_ws/sim2real/scripts/fakes/fake_hand_state_pub.py --echo   # 손 명령 반사(진화 obs)
```

- 실손 세션에서는 금지(드라이버와 토픽 충돌). 실손+접촉만 없을 땐 fake_tip_contact_pub 대신
  **A4 tip_contact_pub 사용**(이제 실센서 있음).

### B3. 정책 노드 (★venv 필수 — Isaac py3.11엔 rclpy 없음)

```bash
source /opt/ros/humble/setup.bash
source ~/grasp_infer_venv/bin/activate
export ROS_DOMAIN_ID=126
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/fastdds_wired.xml
# ⚠️ 아래 lstm_test3 경로는 존재하지 않는다.
#    현행 대상 = log/rl_games/open-tesol/right/grasp-sensor/lstm_test1 (학습 중, 로컬 5090)
python3 ~/rl_ws/sim2real/scripts/nodes/grasp_inference.py \
  --robot tesollo_sensor__right \
  --agent <grasp-sensor run>/params/agent.yaml \
  --ckpt  <재학습 run>/nn/<최종>.pth \
  2>&1 | tee /tmp/grasp_infer.log
```

- `tee /tmp/grasp_infer.log` 필수 — 에피소드 사후분석의 유일한 기록(joint_monitor 는 measured 만).
- 옵션: `--episode-steps 1200`(2배 천천히, sim 기본 600) · `--log-dir`(CSV 위치) ·
  `--contact-threshold`(tip 접촉 임계[N], 기본 sim 0.1).
- **에피소드 사이클(b66d6b9)**: start → APPROACHING(pregrasp IK 이동) → RUNNING(grasp+lift,
  per-step CSV `logs/grasp_ep_*.csv`: action·관절·effort·tip힘·접촉·cup·palm·명령) →
  **PLACING**(명령 궤적 역재생 2배 감속 — 컵 제자리 반환·pregrasp 복귀) → IDLE → 재트리거.
- start 게이트: 전 센서 1s 신선도(START_FRESH_SEC) 요구. RUNNING 두절 5s 지속 시 자동 중단.
- tip 접촉은 palm-컵 0.10m 이내에서만 유효(CONTACT_GATE_DIST — 테이블 접촉發 거짓 lift 차단).
  rebias 는 **팔 부착·공중 자세**에서: `ros2 service call /tip_contact/rebias std_srvs/srv/Trigger`.

### B4. 에피소드 제어

```bash
ros2 service call /grasp/start std_srvs/srv/Trigger    # /grasp/stop, /grasp/reset 동일
```

**방어 동작(62909aa+67ade46, 정상):**
- start 시 손 자세가 APPROACH와 어긋나면 **경고만**(휴지 자세일 수 있음 — 아직 명령 전).
- **settle 종료 시 능동 판별**: APPROACH를 settle 동안 명령한 뒤에도 피드백 미추종이면
  (예: 물리 −1.57인데 0.000 보고 = Modbus 피드백 동결) **관절명 출력 + IDLE 복귀**.
  복구는 A3-1(손 전원 재인가). 의도된 경우 `--allow-hand-mismatch`.
- RUNNING 중 팔/손/컵 토픽 0.5s 두절 → `[RUNNING] 센서 두절` 로그 + **명령 홀드**.

---

## C. 트러블슈팅 요약

| 증상 | 원인/처방 |
|---|---|
| 팔이 컵에서 멀어지다 정지 | 손 obs zeros — start 게이트가 이제 차단. A3-1 손 전원 재인가 |
| discovery 되는데 데이터 불통 | FASTRTPS 프로파일 프로세스별 불일치 — §0 env 확인 |
| goal success인데 팔 무동작 | JTC interpolation none + 미래 tfs — 브리지 최신판(tfs=0) 확인 |
| 손 관절 전부 0.000 고정 | Modbus 세션 꼬임 — 손 전원 재인가 후 벤더 스크립트부터 |
| start 거부(미수신 토픽) | 해당 토픽 발행 노드/DDS 확인 (`ros2 topic hz <topic>`) |
| settle 후 손 미추종 → IDLE | 피드백 동결 확정(물리 자세≠보고값) — 손 전원 재인가 후 `/dg5f_right/joint_states` 실값 확인 |

## D. 오프라인 진단 도구 (로컬 pc5090)

```bash
# 폐루프 재현기(정책+Fabrics+mock팔): max_vel/손모드/지연 ablation
/home/user/rl_ws/IsaacLab/isaaclab.sh -p /home/user/rl_ws/sim2real/scripts/analysis/grasp_loop_sim.py \
  --agent <agent.yaml> --ckpt <ckpt.pth> --max-vel 0.1 --hand-mode static|zero|echo [--obs-delay 6]

# Isaac 죽은 손 probe
cd /home/user/rl_ws/hdgp && /home/user/rl_ws/IsaacLab/isaaclab.sh \
  -p scripts/reinforcement_learning/rl_games/play.py --task open-tesol_r_grasp_v1-play-lstm \
  --checkpoint <ckpt> --num_envs 32 --headless --dead_hand_probe
```
