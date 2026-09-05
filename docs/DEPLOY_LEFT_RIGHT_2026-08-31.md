# 실기 재생 절차 — left v2H_wide + right m1 (08.31 준비분)

> 실행 주체: 사용자(로봇 PC 5070ti). DDS 설정은 사용자 몫(기존 방침).
> 이 문서는 이 세션이 구운 산출물과 그 재생 순서만 적는다.

## 산출물

| 산출물 | 내용 | 상태 |
|---|---|---|
| `logs/shadow/bag_reset_left_both` + `bag_reset_right_both` | 차렷(전관절 0) → 두 정책 홈. 2605프레임·52.1 s. **두 백 동시 재생** | ✅ (08.27, 홈 일치 재검증 08.31) |
| `logs/shadow/bag_v2H_wide` | 좌 정책 30 s 롤아웃, rate 0.53 → 56.6 s | ✅ |
| 우 m1 정책 bag | grasp_s2r 성공 에피소드 | ⛔ 블로커 — m1 학습 소스(08.29 dirty) 소실로 성공 재현 불가. `logs/policy/right_m1/README.md` 참조 |
| `logs/shadow/pour_traj/` | pour 성공 에피소드 60 Hz 시계열 + 초기 상태 | ✅ (P3+ 는 별도 트랙) |

## 순서

1. **전 관절 0 확인** (실기 시작 규약 · USD 차렷 관통 항목: 우손이 몸에 닿는지 실물 확인 — 미해결)
2. 리셋 bag 동시 재생 → 두 팔 정책 홈 도달 (`robotctl` 검증: 좌 preset home / 우 grasp_s2r rest)
3. `lowlevel_check` TEST0(기동 자세)·TEST1(hold 드리프트) — 좌팔 우선
4. 정책 bag 재생: `shadow_replay.py --bag logs/shadow/bag_v2H_wide --execute`
   (직행 `ros2 bag play` 는 첫 프레임 자세 일치 확인 후에만)
5. 중단 조건: 관절 추종오차 > 0.3 rad · `l_aj_7` effort > 5 N·m · 명령 두절 1 s

## 주의

- 좌 bag 은 **rate 0.53** — 요구 peak 3.73 rad/s 가 실기 한계(2.0)를 넘어 시간을 늘렸다.
  경로는 그대로다(클램프 아님).
- 리셋 bag 끝 자세 == 두 정책 홈(소수점 일치 실측) — 리셋 bag 을 건너뛰고 정책 bag 을
  틀면 첫 프레임 도약이 난다.
- pour 연계(P2 판정): pour 초기 pose 는 두 정책 goal 분포 **밖**(좌 y −14.5 cm ·
  우 x −11 cm) → goal 지령 이송 불가, **브리지 램프**(1회 계산 후 재생) 필요.
  상세: `logs/shadow/pour_traj/README.md`

---

## 우팔 차렷↔rest 시퀀스 (08.31 실기 확정 · 손 무전원)

전제: 테솔로 손은 기계적으로만 부착, 전원 미연결(무전원·limp). 손 채널은 **발행하지
않는다**(`--arm-only`). 좌팔은 차렷이어야 한다(우팔 3구간 경로가 sim 에서 그 장면으로
검증됨 — 순서: 좌 차렷 먼저, 그 다음 우팔).

```bash
cd ~/rl_ws/sim2real && source /opt/ros/humble/setup.bash && . .venv/bin/activate

# ① 좌팔이 preset 에 있다면 먼저 차렷으로 (역재생, 52.1 s · peak 0.25 rad/s)
python3 scripts/nodes/shadow_replay.py --sim logs/shadow/reset_both/reset_left_reverse.npz \
    --robot gripper_left --rate-scale 1.0 --allow-idle-arm-mismatch --execute

# ② 우팔 차렷 → rest(j2 0.3 · j4 2.0)  (104 s @0.5 · peak 0.125 rad/s)
python3 scripts/nodes/shadow_replay.py --sim logs/shadow/reset_both/reset_right.npz \
    --robot tesollo_sensor__right --rate-scale 0.5 --arm-only \
    --allow-idle-arm-mismatch --execute

# ③ 우팔 rest → 차렷 복귀  (208 s @0.25 · ★임계 0.55 필수)
python3 scripts/nodes/shadow_replay.py --sim logs/shadow/reset_both/reset_right_reverse.npz \
    --robot tesollo_sensor__right --rate-scale 0.25 --arm-only \
    --allow-idle-arm-mismatch --abort-tracking-err 0.55 --execute
```

실측 (08.31):
- ②는 완주하지만 **rest 에서 정적 처짐이 크다**: j1 −0.125 / j4 −0.13 / **j7 −0.28 rad**
  (무전원 손 무게). 실측 j2 +0.279·j4 +1.871 — 손끝이 테이블에 아슬아슬하게 닿는다.
- ③을 기본 임계(0.30)로 돌리면 **시작 직후 j7 이 0.32 rad 뒤처져 중단**된다.
  충돌이 아니다 — j7 effort 2.8 N·m 뿐, kp 10 손목이 limp 손을 못 드는 것.
  `--abort-tracking-err 0.55` 로 열어야 복귀가 된다(effort 캡 5 N·m 은 유지되어
  실충돌은 여전히 잡는다).
- ③ 완주 통계: err max j7 0.318 rad · eff max j2 12.96 / j4 12.49 N·m(들어올리는 중력일)
  · 차렷 잔차 max 0.75°.
- 손 전원을 켜서 파지 자세로 오므리면 COM 이 손목에 가까워져 j7 처짐이 줄 것으로 예상
  — 무전원 상태 수치를 그대로 일반화하지 말 것.

---

## 08.31 저녁 갱신 — 배포 정책 교체 + 새 preset 세팅

### 배포 체크포인트 (사용자 지정)

| 팔 | 체크포인트 | 계약 | 홈(런 dump 근거) |
|---|---|---|---|
| 좌 | `logs/policy/left_v2E19/nn/v2E19_zfloor_ep1800.pth` | obs49/act7/MLP | **구 v1 홈** — J147 아님 |
| 우 | `logs/policy/right_b1/nn/b1_ep10800.pth` (★8종 전수) | obs155/act21/LSTM | (0.0380,0.4012,0.6015,0.9643,0.0294,0.7060,0.4213) |

둘 다 params/ 회수 완료(m1 재생불가 사태 방지 요건). m1_final 은 후보 잔류.

### preset 리셋 궤적 (팔별 개별 재생)

- 좌팔: **기존 `reset_both/reset_left.npz` 그대로** (구홈 확정) — 런북 ① 절차 동일
- 우팔: **`reset_both/reset_right_v2.npz` 신규** (차렷→b1 홈, 1080프레임·43.2s@0.5배)
  - sim 검증 ✅ 새 접촉 0건 (probe_reset_v2_homes.py run4, start_hold 100)
  - 경로: ①j2 0→0.9 ②j4 0→2.0(완전접기) ③접힌 채 손목자세(j3·j5·j6·j7) ④j2·j4 동시 보간으로 위에서 홈 안착
  - ⚠2차 시도의 교훈: **j2 0.9 에서 팔꿈치를 편 채 손목을 움직이면 손이 테이블을 관통**한다(팔 전개 기하)
  - npz grip_cmd = s2r init 손자세 20D(엄지 −1.57/−0.5) — 손 전원 있으면 함께 세울 수 있음, 무전원이면 `--arm-only`

```bash
# 우팔 preset (실기)
python3 scripts/nodes/shadow_replay.py --sim logs/shadow/reset_both/reset_right_v2.npz \
    --robot tesollo_sensor__right --rate-scale 0.5 --arm-only \
    --allow-idle-arm-mismatch --execute
```

### 테솔로 우손 연결 + 주먹 규약 (08.31 확정)

**연결 = 이더넷 Modbus TCP** (CAN 아님). 손이 안 보이면 IP 부터 본다:

```bash
ip -br addr show eno1                       # 주소가 없으면 링크는 살아도 통신 불가
sudo ip addr add 169.254.186.1/16 dev eno1  # 손은 링크로컬 169.254.186.72:502
ping -c2 169.254.186.72

cd ~/rl_ws && source /opt/ros/humble/setup.bash && source robot_control/ros_ws/install/setup.bash
ros2 launch dg5f_driver dg5f_right_driver.launch.py delto_ip:=169.254.186.72
# → "Connected to Delto Gripper (Model: DG5F-R (0x5F22), Motors: 20)" · FW v2.9
```
★`dg5f_driver` 는 `robot_control/ros_ws/install` 에만 빌드돼 있다(urdf 워크스페이스엔 없음).

**주먹 규약(사용자 확정)**: hand bringup 은 **주먹 자세**에서 한다 — 손가락이 바닥
베이스에 닿지 않는 자세. 그 자세를 스냅샷해 이후 모든 bringup 의 기준으로 삼는다.

```bash
python3 scripts/calib/capture_right_hand_pose.py          # 확인만
python3 scripts/calib/capture_right_hand_pose.py --save   # → config/right_hand_fist.yaml
```

**팔·손 순서**: ①전 구간 **주먹 유지**로 팔 이동 → ②홈 안착 후 **손 펴기**.
`reset_right_v2.npz` 가 이 순서로 만들어져 있다(R①~④ 팔, R⑤ 손펴기 104°).
시작 손자세는 **실물 스냅샷**으로 검증했다(`--start_hand_yaml`) — sim 합성 주먹이
아니라 실기가 실제로 지나갈 자세다. sim 판정 ✅ 새 접촉 0건 · 테이블 0.00 N.
⚠실물 `rj_dg_5_3` 가 소프트 한계를 +0.6 mrad 넘겨 서 있어 sim 에서 clamp 된다(무해).

```bash
# 손 포함 재생 (1474프레임 · 58.9 s @0.5)
python3 scripts/nodes/shadow_replay.py --sim logs/shadow/reset_both/reset_right_v2.npz \
    --robot tesollo_sensor__right --rate-scale 0.5 --with-hand \
    --allow-idle-arm-mismatch --abort-tracking-err 0.55 --execute
# 손 전원이 없을 때만 --arm-only (손 채널 미발행)
```

### RViz → Isaac 미러 (신규 사슬)

```
shadow_replay/정책 ─ROS /isaacsim/*_cmd─▶ 실기 JTC (기존)
                       └▶ ros_cmd_to_udp.py ─UDP:47321─▶ probe_sim_follower.py (Isaac GUI 상시)
```

```bash
# ① Isaac 미러 (venv 없는 셸, GUI 상시 유지)
cd ~/rl_ws/IsaacLab && ./isaaclab.sh -p ~/rl_ws/sim2real/scripts/probes/probe_sim_follower.py
# ② ROS→UDP 어댑터 (ROS 셸)
python3 ~/rl_ws/sim2real/scripts/nodes/ros_cmd_to_udp.py
# ③ 그 다음 shadow_replay --execute 를 돌리면 sim·실기가 같은 지령을 받는다
```
