# 다음 세션 준비 — 우팔 튜닝 이어가기 (08.31 마감 시점)

오늘 중력보상까지 끝났다(처짐 12.76° → 2.05°). 다음은 **①페이로드 실질량 반영 검증**과
**②kd 교정**, 그 다음 **③손가락 개별 튜닝**이다. 세 가지 모두 명령까지 준비해 두었다.

## 매 세션 공통 — 시작 절차

```bash
# 0) 모터 전원을 켠 **뒤** bringup (enable 은 활성화 시 1회만 나간다)
ros2 launch openarm_bringup openarm.bimanual.launch.py \
    use_fake_hardware:=false right_can_interface:=can0 left_can_interface:=can1

# 손을 쓸 때만 (이더넷 Modbus — IP 부터 확인)
ip -br addr show eno1                        # 주소 없으면:
sudo ip addr add 169.254.186.1/16 dev eno1
source ~/rl_ws/robot_control/ros_ws/install/setup.bash
ros2 launch dg5f_driver dg5f_right_driver.launch.py delto_ip:=169.254.186.72

# 1) 팔은 차렷에서 시작한다(전 관절 0). 확인:
ros2 topic echo --once /joint_states | head -30
```

★`openarm.bimanual.launch.py` 는 `openarm_bringup` 의 자체 controllers.yaml 을 쓴다 —
거기엔 effort 컨트롤러가 **이미 선언돼 있다**. `urdf/config/…` 를 수정할 필요 없다
(그 파일은 다른 launch 용이고, 08.31 에 같은 블록을 넣어 두었다).

## ★ 왜 preset 궤적을 보정하지 않는가 (사용자 질문, 08.31)

궤적 자체는 sim 에서 접촉 0 건으로 검증됐다. 실기에서 테이블과 겹친 것은 **실기가 그
궤적을 못 따라가서** 실제 경로가 아래로 처졌기 때문이다(j7 −12.8°, palm −50 mm).
보상이 켜지면 sim 궤적 = 실기 궤적이 되고 간섭이 사라진다.

⇒ **`reset_right_safe.npz`(j2 +0.40)는 임시방편이었고, 보상이 서면 버린다.**
   원래 정책 홈 궤적 `reset_right_v2.npz` 로 돌아간다 — b1 정책이 요구하는 자세다.

⇒ 단, 재생 **중에도** 보상이 필요하다. `robotctl pose gravity` 는 정지 유지용이므로
   `scripts/nodes/gravity_comp_node.py`(08.31 신설)를 별도 터미널에서 돌린다:

```bash
# 터미널 A — 연속 보상(실측 자세로 매 20 ms 재계산, Ctrl-C 시 0 송출)
python3 scripts/nodes/gravity_comp_node.py --payload 0.9130,-0.00450,-0.01723,0.22147 \
    --scale 1.0 --execute
# 터미널 B — 그 위에서 궤적 재생
python3 scripts/nodes/shadow_replay.py --sim logs/shadow/reset_both/reset_right_v2.npz ... --execute
```

## ① 페이로드 실질량 반영 — scale 1.0 검증 (30분)

**가설**: 08.31 의 최적 스케일 1.1 은 모델 오차가 아니라 **URDF 손 질량이 가벼워서**다.
사용자 실측 **1.763 kg** vs URDF 1.685 kg = **4.6% 과소**(1.0463배). 최적 1.1 과는 4.9%
차이가 남으므로 질량이 절반쯤 설명하고 나머지는 케이블·커넥터나 COM 오차일 것이다.

```bash
# 자세 만들기(재생 먼저 — 처진 상태에서 보상만 켜면 마찰로 안 올라온다)
cd ~/rl_ws/sim2real && source /opt/ros/humble/setup.bash && . .venv/bin/activate
python3 scripts/nodes/shadow_replay.py --sim logs/shadow/reset_both/reset_right_safe.npz \
    --robot tesollo_sensor__right --rate-scale 0.5 --arm-only \
    --allow-idle-arm-mismatch --abort-tracking-err 0.65 --execute

# effort 컨트롤러 로드
~/rl_ws/robot_control/ros_ws/load_effort_controllers.sh right

# ★페이로드 0.913 (= 0.835 + 1.763−1.685) 로 스윕
robotctl pose gravity --group openarm_right_arm \
    --urdf ~/rl_ws/urdf/generated/rl/openarm_tesollo_sensor_rl.urdf \
    --payload 0.9130,-0.00450,-0.01723,0.22147 \
    --sweep 0.9,1.0,1.1,1.2 --output logs/gravity_right_m1763.json --execute
```

**판정**: 최적이 1.0 근처로 내려오면 가설 확인. 그러면 앞으로 스케일은 1.0 고정으로 두고
질량만 관리하면 된다(스케일로 질량 오차를 덮는 것보다 정직하고 자세 의존성도 작다).

## ② kd 교정 — 남은 마찰 잡기

남은 1~2° 는 중력보상으로 못 잡는다(마찰). 07.29 autotune 실측이 **kd 가 2.6~3.6배
부족**하다고 말한다: wrist 0.6 → **2.154**, elbow 2.0 → 5.635, proximal 2.4167 → 6.376.

**kp 는 건드리지 말 것.** 계단 실측(74.7/75.1/69.5/60.9/10.8/14.5/10.5)이 벤더 스펙과
≤4% 일치하고, 07.29 에 범위를 넓혔더니 elbow 가 실측의 2.4배로 밀리며 오차가 오히려
나빠졌다(3.38e-2 vs 2.86e-2).

수정 대상 파일 (**bringup 재시작 필요**, colcon build·source 불필요):
```
~/rl_ws/robot_control/ros_ws/src/openarm_ros2/openarm_bringup/config/controllers/… 가 아니라
xacro 가 읽는 곳: sim2real/vendor/openarm/openarm_description/config/arm/v10/control_gains.yaml
  joint1: kp 70.0 kd 2.75      joint5: kp 10.0 kd 0.7
  joint2: kp 70.0 kd 2.5       joint6: kp 10.0 kd 0.6
  joint3: kp 70.0 kd 2.0       joint7: kp 10.0 kd 0.5
  joint4: kp 60.0 kd 2.0
```
⚠ `ros2 pkg prefix openarm_description` 가 가리키는 install 사본을 확인하고 고칠 것 —
저장소에 같은 파일이 **8곳**에 있다(08.31 확인). 실제로 로드되는 것은
`urdf/install/openarm_description/share/…` 였다.

**검증**: 게인 변경 후 같은 preset 궤적을 재생하고 `analyze_right_preset_bag.py` 로
전후를 비교한다. 볼 것은 C−B(추종오차)의 **RMSE 와 지연**이다. kd 는 처짐(정적)보다
추종 지연·진동에 듣는다.

## ③ 손가락 개별 튜닝

팔과 같은 방식이지만 손은 **자기 컨트롤러**(`/dg5f_right/…`)를 쓴다.

```bash
# 현재 자세 스냅샷(bringup 기준 주먹) — 매번 확인
python3 scripts/calib/capture_right_hand_pose.py

# 발열이 의심되면 즉시 이완(지령=실측으로 맞춰 버티는 토크를 없앤다)
python3 scripts/ops/relax_right_hand.py --execute
```

**★08.31 미해결 — 엄지 `rj_dg_1_4`**: 자세와 무관하게 effort **38~79**(나머지 19관절은
0~18). 지령 0 으로 가려다 +0.103 rad 에서 막혀 계속 밀어붙인다. 발열의 주범.

**진단 결과(사용자, 08.31)**: 손으로 움직여 보니 **뻑뻑하지 않고 다른 관절과 비슷**하다.
⇒ 기계적 간섭이 아니라 **드라이버/캘리브 문제**다. 유력한 후보는 **homing offset 이
틀어져 지령 0 이 도달 불가 위치를 가리키는 것**. 확인 방법: 그 관절만 여러 목표
(−0.2, −0.1, 0, +0.1, +0.2)로 지령해 실제 도달 각과 effort 를 재고, 다른 손가락의
같은 관절(`rj_dg_2_4` 등)과 대조한다. 도달 범위가 통째로 어긋나 있으면 offset 이다.

손가락 스윕 도구는 아직 없다. 팔의 `probe_right_joint_sweep.py` 를 손 토픽으로 옮기면
되지만, 위 엄지 문제를 먼저 해결해야 의미가 있다.

## 오늘 남긴 산출물

| 파일 | 내용 |
|---|---|
| `docs/GRAVITY_COMP_RIGHT_2026-08-31.md` | 중력보상 전 과정·실측·함정 |
| `logs/gravity_right_{sweep,j7,j5,j2,final}.json` | 스윕 원자료(`r2s identify` 가 읽는 형식) |
| `logs/rosbags/right_preset_{tune1,safe1}` | ACTION/SIM/REAL 3신호 bag |
| `logs/shadow/reset_both/reset_right_{v2,safe}{,_reverse}.npz` | preset 궤적 4종 |
| `config/right_hand_fist.yaml` | 실물 주먹 스냅샷(bringup 기준) |
| `scripts/calib/hand_payload.py` | 손 페이로드 계산 |
| `scripts/analysis/analyze_right_preset_bag.py` | bag 3신호 분석 |
| `scripts/ops/hold_right_arm.py` · `relax_right_hand.py` | 자세 유지 · 손 토크 이완 |

## 안전 규약 (08.31 확립)

- 실기를 움직이는 모든 명령은 **사용자 승인 후**
- 자세 실험은 **테이블을 치운 조건**에서만
- 측정은 **정착 후**에 — 과도기를 실측으로 오독한 전례 있음
- effort 컨트롤러는 쓰고 나면 **반드시 `--unload`**
- 세션 종료 전 팔을 **차렷으로 복귀**시키고 드라이버를 내린다

---

## ★★엄지 `rj_dg_1_4` 규명 완료 (08.31 밤) — 가동범위가 지령의 절반

오늘 종일 발열의 주범이던 관절의 정체를 스윕으로 확정했다.

**같은 위치 관절과의 대조** (전 관절 0 에서 출발, ±0.25 rad 스윕):

| 관절 | 지령 +0.250 | 실측 | 도달률 | effort |
|---|---|---|---|---|
| 검지 `2_4` | +0.250 | **+0.204** | **82%** | **8** |
| 엄지 `1_4` | +0.250 | +0.120 | **48%** | **62** |

엄지의 도달점을 모으면 **실제 가동범위는 −0.14 ~ +0.12 rad** 뿐이다:

| 지령 | 실측 |
|---|---|
| +0.250 | +0.120 |
| **0.000** | **−0.131** ← 지령 0 조차 도달 못한다 |
| −0.250 | −0.138 |

히스테리시스(마찰) **−13.85°** — 검지 대비 크다.

⇒ **막힌 것도 느린 것도 아니고, 애초에 갈 수 없는 곳을 지령받아 무한히 밀어붙이는
   상태였다.** 그래서 자세와 무관하게 effort 38~79 가 나왔고 가만히 있는데 뜨거워졌다.

**해야 할 일**
1. 엄지만 **작은 진폭으로 정밀 스윕**(±0.10, 0.05 rad/s)해 가동 끝을 확정
2. 확정한 범위를 프로필(`config/robots/tesollo_sensor__right.yaml` joint_limits)과
   sim 양쪽에 반영 — **정책이 그 밖을 지령하면 실기에서 계속 발열한다**
3. 캘리브(homing offset)로 고칠 수 있는지 벤더 문서/드라이버 확인. 기계적 한계라면
   범위 제한이 유일한 답이다
4. 나머지 18관절도 같은 방식으로 도달률 측정 — 엄지만의 문제인지 확인

## 손 스윕 도구 (`probe_hand_joint_sweep.py`) — 판정 규약

08.31 에 세 번 고쳤다. 그 과정 자체가 이 하드웨어의 성질을 말한다:

| 시도 | 판정 | 왜 틀렸나 |
|---|---|---|
| ① effort > 20 | 크기 | **기동 토크가 23** — 정상 동작을 끊었다 |
| ② effort > 40 이 1s 지속 | 지속시간 | **엄지는 지령의 46% 속도로 계속 가고 있었다** — 느린 것을 막힘으로 오판 |
| ③ effort > 40 을 1s 쓰는데 **0.01 rad/s 미만** | 토크+정지 | ✅ 정확. "토크를 쓰는데 안 움직인다"가 막힘의 정의 |

★기동 토크(23~47)와 막힘(62~79)은 **크기로 겹친다**. 둘을 가르는 것은 크기가 아니라
  "움직이고 있는가"다.
★충돌 판정 기준은 **그 스윕 시작 시점의 실측**이어야 한다. 지령값을 기준으로 삼으면
  지령을 못 지키는 관절(엄지)이 늘 "밀렸다"로 잡힌다(08.31 오탐).
★전 관절을 동시에 크게 움직이면 각자 뒤처져 effort 가 치솟는다. 스윕 준비는
  대상 관절 위주로 최소한만.

## 손 온도 계측 (08.31 신설)

하드웨어(`delto_hardware/src/system_interface.cpp:406`)가 온도를 export 하는데 **URDF
ros2_control 블록에 `state_interface name="temperature"` 선언이 없어** controller_manager
가 버리고 있었다. 20관절에 선언을 넣어 살렸다
(`urdf/vendor/delto_m_ros2/dg5f_driver/urdf/dg5f_right_ros2_control.xacro`, 드라이버 재시작 필요).

- `/dg5f_right/dynamic_joint_states` 에 `temperature` 로 나온다. bag 기록에 포함시켰다
- 실측: 쉬는 중에도 **39~44℃**, 토크 0 으로 40초 둬도 안 식음(하드웨어 특성 — 사용자 확인)
- ⚠팔(openarm)은 하드웨어가 온도를 아예 안 낸다. effort 로 간접 추정만 가능

---

## ★ sim 게인 검증 완료 (08.31 밤) — 실측 게인이 실기에 더 가깝다

`grasp_s2r` sim 에서 실기와 **같은 궤적**(reset_right_v2, 배속 0.5)을 재생해 비교했다.
정책은 돌리지 않고 관절 지령만 흘려보내는 순수 재생이다(`probe_s2r_gain_replay.py`).

| 관절 | KUKA (kp 300/100/50/25) | **실측 게인** (73.1/60.9/11.9) | 실기 (중력보상 ON) |
|---|---|---|---|
| j3 | 1.09° | **0.54°** | 0.54° ← 일치 |
| j4 | 0.87° | **0.52°** | 1.43° |
| **j7** | **4.07°** | **1.73°** | 1.00° ← 2.4배 개선 |
| **전체 RMSE** | **1.89°** | **1.31°** | **0.94°** |

**실기와의 관절별 차이 합 6.64° → 4.32° (35% 감소).** 손목 j7 이 가장 크게 개선됐다 —
KUKA 의 kp 25 가 이 관절에서 가장 비현실적이었다.

### ★★중력보상은 이미 학습에 반영되어 있었다

`grasp_s2r_env_cfg.py:118` — 로봇 `disable_gravity=True`(컵은 `False`, 608줄).
**우팔 정책은 중력 없는 sim 에서 학습됐다.** 08.31 에 실기에 중력보상을 켠 것이 정확히
그 조건을 재현한 것이고, RMSE 0.94° 가 나온 것이 우연이 아니다.
⇒ "중력보상을 학습에 알려줘야 하나"(사용자 질문)의 답: **이미 알려져 있다.**
   좌팔 `grip_l` 은 반대로 `disable_gravity=False`(중력 켜고 학습) — 트랙마다 다르다.

### sim kp/kd 와 실기 kp/kd 는 **같은 의미**다

`v10_simple_hardware.cpp:276` 이 모터에 `{kp, kd, pos, vel, tau_ff}` 를 보내 MIT 모드로
구동한다: `tau = kp(q_des−q) + kd(qd_des−qd) + tau_ff`. Isaac ImplicitActuator 와 같은
식이고 단위도 N·m/rad 로 같다. 07.29 autotune 이 이 전제로 돌아 실측과 4% 안에서 맞은
것이 그 검증이다. (다만 Isaac 에는 `armature`·`friction` 이 별도 항으로 있고, 실기에는
감속기 마찰이 kp 로 표현되지 않는 형태로 섞여 있다 — autotune 이 friction 을 함께 찾는 이유.)

### 게인 전환 스위치 (08.31 신설)

`HDGP_S2R_REAL_GAINS=1` → 실측 게인, 미설정 → KUKA(기본).
`hdgp/source/openarm/openarm/agnostic/tasks/grasp_s2r/robot_profiles.py:296` 근처.
**기본값을 바꾸지 않은 이유**: 배포 정책 `b1_ep10800` 이 KUKA 게인에서 학습됐다.
게인을 확정해 기본값으로 바꾸려면 재학습이 필요하다.

### 남은 미지수 = damping

sim 1.31° vs 실기 0.94° 로 여전히 sim 이 못 따라간다. 손목 j5·j6 이 실기보다 나쁘다
(1.79/2.10 vs 0.78/0.99). kp 는 계단 실측으로 확실하지만(벤더 스펙과 4% 일치)
**kd 는 준정적 데이터로 식별할 수 없다** — 여진 데이터가 필요하다.
(부수: sim dt 0.0167 vs 실기 지령 25 Hz 라 한 지령을 2스텝 유지하는 계단이 max 오차를
 키운다. 엄밀한 비교에는 sim dt 정렬도 필요하다.)

## 여진 데이터 수집 — ★테이블 충돌 대비 (사용자 우려)

**진폭은 생각보다 작다.** `amplitude_scale=0.65` 일 때 관절별 실제 진폭:

| 관절 | j1 | j2 | j3 | j4 | j5 | j6 | j7 |
|---|---|---|---|---|---|---|---|
| 진폭 | ±9.1° | ±6.5° | ±5.9° | ±4.5° | ±5.9° | ±2.9° | ±5.9° |

`amplitude = (upper − lower) × 0.05 × scale` 이라 관절 범위에 비례한다(cli.py:2123).

**★여진은 "팔이 지금 있는 자세" 주변에서 흔든다** — `neutral = _start_pose(..., adapter.read_state())`
(cli.py:2138, 주석: "Around where the arm is, not the middle of its range").
⇒ **먼저 안전한 자세로 옮긴 뒤 collect 를 돌리면 된다.** 자세는 우리가 정한다.

**권장 자세 = R② 경유점** `(0, 0.9, 0, 2.0, 0, 0, 0)` — 팔을 접어 올린 곳.
`reset_right_v2.npz` 궤적이 지나는 지점이고 sim 에서 몸통·테이블 무접촉으로 검증됐다.
거기서 ±9° 는 여유가 충분하다.

```bash
# ① 안전 자세로 이동 (reset_right_v2 를 R② 까지만 재생 — --frames 로 자른다)
python3 scripts/nodes/shadow_replay.py --sim logs/shadow/reset_both/reset_right_v2.npz \
    --robot tesollo_sensor__right --rate-scale 0.5 --arm-only --frames 640 \
    --allow-idle-arm-mismatch --execute
#   (R①어깨올림 180+30 + R②팔꿈치접기 400+30 = 640 프레임)

# ② 여진 수집 — 중력보상을 켜 둔 채로 할 것(sim 이 중력 없는 조건이므로)
python3 scripts/nodes/gravity_comp_node.py --payload 0.9130,-0.00450,-0.01723,0.22147 \
    --scale 1.0 --execute      # 터미널 A
robotctl r2s collect --group openarm_right_arm --amplitude-scale 0.65 \
    --output ~/r2s/right_track_0901.npz --execute    # 터미널 B

# ③ 정규화 → HDF5 → autotune
robotctl r2s normalize ...
python3 ~/rl_ws/hdgp/scripts/r2s_autotune/... (configs/tesollo_sensor_right_arm.yaml 의
    real_track.hdf5 경로를 새 파일로 바꾼 뒤 실행)
```

⚠**sim 으로 먼저 검증할 것**: 그 자세에서 ±9° 여진이 테이블·몸통에 안 닿는지.
`probe_reset_v2_homes.py` 와 같은 접촉 판정 방식을 쓰면 된다(기준선 뺀 새 접촉만 본다).
