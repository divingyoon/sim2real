# 우팔 중력보상 — 프레임워크와 실행 절차 (08.31)

## 왜 필요한가 (실측)

테솔로 손을 단 우팔이 preset 자세를 유지하지 못한다. 08.31 실측:

| 관절 | 지령 대비 처짐 | 정지 시 effort |
|---|---|---|
| j2 | −9.1° | 11.1 N·m |
| j7 | −12.8° | 2.2 N·m |
| j5 | −10.5° | 1.8 N·m |

palm 이 지령 대비 **50 mm 내려앉아** 손끝이 테이블을 긁는다. j2 를 +0.40 rad 올려도
palm 은 +20 mm 밖에 안 올랐다 — 올릴수록 모멘트암이 길어져 처짐이 같이 커지기 때문이다
(j2 처짐 −4.3° → −9.1°). **자세를 바꿔서는 못 이긴다.**

방향 성분 분해(올릴 때 vs 내릴 때, `real_right_tune1.csv` / `real_right_unreset.csv`):

| 관절 | 히스테리시스(마찰) | 방향무관(중력) |
|---|---|---|
| j3 | −4.27° | −5.09° |
| j6 | −4.80° | **+12.48°** |
| j7 | −4.08° | **−10.35°** |

중력분이 마찰의 2~3배다. 마찰은 kd 로, 중력은 **tau_ff 로** 다뤄야 한다.

## 무엇이 막고 있었나

`tau_ff` 경로는 **원래부터 살아 있었다**(하드웨어가 MIT 모드로
`tau = kp(q_des−q) + kd(qd_des−qd) + tau_ff` 를 보낸다). `robotctl pose gravity` 도
이미 구현·커밋되어 있었다. 막고 있던 것은 둘뿐:

**① 실기 controllers.yaml 에 effort 컨트롤러 선언이 없었다** → 08.31 추가함
(`urdf/config/openarm_left_gripper_bimanual_controllers.yaml`).
launch 는 일부러 spawn 하지 않는다 — 토크 경로를 켜는 것은 사람이 의도해서 해야 한다.

**② 중력 모델이 손가락 질량을 통째로 버린다.**
`kinematics._lumped()` 는 **가동 조인트 뒤의 링크를 제외**한다(설계상 의도: 그 위치가
조인트 값에 달렸는데 팔 체인은 그 값을 모른다). 테솔로 손은 20관절이 전부 가동이라:

| | 질량 | COM z (r_al_7 프레임) | 모멘트 |
|---|---|---|---|
| 손 전체 (진값) | 1.685 kg | 0.170 m | 0.286 kg·m |
| 모델이 세는 몫 | 0.850 kg | 0.119 m | 0.101 kg·m |
| **빠진 몫** | **0.835 kg** | **0.221 m** | **0.186 kg·m** |

**과소 2.83배** — 07.29 캘리브가 기록한 "j6 약 3.4배 과소"와 같은 현상이다.

## 프레임워크 (08.31 신설)

| 조각 | 위치 | 역할 |
|---|---|---|
| `hand_payload.py` | sim2real/scripts | 손 자세 → 빠진 질량·COM 계산. `--format arg` 로 CLI 인자 출력 |
| `with_payload()` | robot_control/kinematics.py | 체인 마지막 링크에 페이로드 합성(불변, 새 Chain 반환) |
| `--payload` | robot_control/cli.py `pose gravity` | `MASS,X,Y,Z` 를 받아 중력 체인에 얹음 |
| effort 컨트롤러 | urdf/config/…controllers.yaml | `right/left_forward_effort_controller` 선언 |
| `load_effort_controllers.sh` | robot_control/ros_ws | 켜고 끄기 (기존) |

테스트 14개 추가, robot_control 전체 **516 passed**.

## 실행 절차 (실기 — ★사용자 승인 후)

```bash
# 0) controllers.yaml 이 바뀌었으므로 bringup 재시작 필요
#    (모터 전원은 bringup 전에 켜 둘 것 — enable 은 활성화 시 1회만 나간다)

# 1) 손 자세에 맞는 페이로드를 뽑는다
cd ~/rl_ws/sim2real && . .venv/bin/activate
python3 scripts/hand_payload.py --pose config/right_hand_fist.yaml --format arg
#   → --payload 0.8350,-0.00450,-0.01723,0.22147   (자세마다 다르다, 주먹/폄 각각)

# 2) effort 컨트롤러를 켠다 (JTC 는 계속 position 을 잡고 있다)
~/rl_ws/robot_control/ros_ws/load_effort_controllers.sh right

# 3) 중력보상 발행 — 먼저 scale 0.5 로 시작해 처짐 감소를 확인
robotctl pose gravity --group openarm_right_arm \
    --urdf ~/rl_ws/urdf/generated/rl/openarm_tesollo_sensor_rl.urdf \
    --payload 0.8350,-0.00450,-0.01723,0.22147 \
    --scale 0.5 --execute

# 4) 스케일 스윕으로 최적점을 찾는다
robotctl pose gravity --group openarm_right_arm \
    --urdf ~/rl_ws/urdf/generated/rl/openarm_tesollo_sensor_rl.urdf \
    --payload 0.8350,-0.00450,-0.01723,0.22147 \
    --sweep 0.0,0.3,0.6,0.9,1.0 --output logs/gravity_right.json --execute

# 5) 끝나면 반드시 내린다
~/rl_ws/robot_control/ros_ws/load_effort_controllers.sh --unload right
```

★`--urdf` 는 **필수**다. 실기 description 에는 손이 아예 없다
(`openarm_left_gripper_bimanual_real.xacro:48-49` `hand="false" ee_type="none"`).
자산 URDF 를 넘기지 않으면 손 질량이 0 인 모델로 보상하게 된다.

★`--payload` 없이 `--urdf` 만 주면 손의 **절반(팔레트만)** 으로 보상한다 — 손목이
여전히 1/2.8 만 받는다. 둘은 함께 써야 한다.

## 실측 결과 (08.31 실기)

preset safe 홈(j2 0.8012)에서 스케일 스윕. **관절오차 = 실측 − JTC 지령.**

| scale | worst | mean | palm (판 위) |
|---|---|---|---|
| 0.0 (보상 없음) | **12.76°** | 7.17° | 118 mm ← 손끝이 테이블에 닿던 상태 |
| 0.3 | 9.79° | 5.59° | — |
| 0.6 | 5.92° | 3.55° | — |
| 0.9 | 2.31° | 1.32° | **250 mm** |
| **per-joint `1.1,1.1,1.1,1.0,1.1,0.9,1.1`** | **2.05°** | **1.03°** | — |

관절별 최적을 따로 찾으면 **j7·j5·j2 모두 1.1** 로 수렴했다(각각 독립 스윕).
j6 만 0.9 — 그 관절은 중력 방향이 반대다.

### ★ 스케일 1.1 의 정체 = URDF 손 질량 오차

**실제 테솔로 손은 1.8 kg**(사용자 실측)인데 URDF 는 **1.685 kg** — 6.4% 가볍다.
실측 최적 1.1 과 1.068 배는 2.9% 차이로, **스케일이 질량 오차를 덮고 있었다.**

⇒ 더 정직한 설정은 스케일이 아니라 **페이로드를 실제 질량에 맞추는 것**:

```
--payload 0.9500,-0.00450,-0.01723,0.22147   # 0.835 + (1.8 − 1.685)
```
이러면 scale 1.0 근처가 최적이 되어야 한다 — **다음 세션에서 검증할 것**.
(COM 은 URDF 분포를 그대로 쓴다. 어느 부분이 무거운지는 모르므로 균일 근사다.)

### 남은 오차의 정체 = 마찰

1~2° 는 중력보상으로 원리적으로 못 잡는다. 양방향 분해(올릴 때 vs 내릴 때)에서
히스테리시스가 4~5° 로 나왔고, 그 절반이 편측 오차로 남는 것과 일관된다.
→ **kd 교정**이 다음 단계다(07.29 autotune: wrist kd 0.6 → **2.154**).

### ★★순서 규약 — 재생 먼저, 보상은 유지용

처진 상태에서 중력보상만 켜면 **마찰 때문에 완전히 안 올라온다**. 실측으로 확인:
같은 스케일이어도 재생 직후(이미 자세가 선 상태)에는 palm 250 mm, 처진 상태에서
켰을 때는 과도기가 길었다. 실전 순서는:

1. `shadow_replay` 로 자세를 만든다 (JTC 가 능동적으로 이동)
2. 그 자세에서 중력보상을 켜 **유지**한다

⚠측정은 **정착 후**에 한다. `robotctl` 자신의 출력이 정착값이고, 도중에 `/joint_states`
를 읽으면 과도기를 실측으로 오독한다(08.31 에 9.57° 로 오독했다가 철회).

## 이후 순서

1. ✅ 중력보상 스케일 확정 (08.31) — per-joint 1.1, 처짐 12.76° → 2.05°
2. ⬜ **페이로드를 1.8 kg 기준으로 올려 scale 1.0 검증** (위 §스케일 1.1 의 정체)
3. ⬜ 남은 오차로 **kd 교정** (07.29 autotune 실측: wrist kd 0.6 → **2.154**, 2.6~3.6배 부족)
3. kp 는 실측값(74.7/75.1/69.5/60.9/10.8/14.5/10.5)이 벤더 스펙과 ≤4% 일치하므로
   **자유 탐색 금지** — 07.29 에 범위를 넓혔더니 elbow 가 실측의 2.4배로 밀리며 오차가
   오히려 나빠졌다. kp·Fc 는 실측 ±10% 고정, 미측정 damping 만 넓게 탐색할 것.
4. 손가락도 같은 방식으로 개별 스윕 → 튜닝값 추출

## 함정

- 자세 실험은 **테이블을 치운 조건**에서만. sim 통과가 실기 안전의 근거가 되지 못한다
  (처짐 때문에 실기 자세는 sim 과 다르다).
- 실기를 움직이는 모든 명령은 **사용자 승인 후**.
- 페이로드는 **손 자세마다 다시 계산**한다(주먹 0.257 vs 폄 0.286 kg·m, 11% 차이).
- effort 컨트롤러를 켠 채 방치하지 말 것 — 쓰고 나면 `--unload`.
