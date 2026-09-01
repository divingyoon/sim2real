# 우팔 여진(r2s collect) 실험 — 준비 완료, 실행 대기

**목적.** 08.31 에 중력보상으로 정적 처짐을 12.76° → 2.05° 로 잡았다. 남은 1~2° 는
마찰·감쇠이고 중력보상으로는 원리적으로 못 잡는다. 그 kd 를 재려면 팔을 **흔들어야**
한다 — 준정적 궤적(0.125 rad/s)에서는 damping 항이 신호에 나타나지 않는다.

**산출물.** `r2s collect` 3회 → `r2s fit` → kd 교정값 → sim 프로필 갱신 → 재학습.

---

## 여진이 실제로 무엇을 하는가

`robotctl r2s collect` 는 **팔이 지금 있는 자세 주변**을 흔든다
(`cli.py:_collect_track_run` — "Around where the arm is, not the middle of its range").
한 트랙은 6 초이고 네 위상으로 되어 있다(`identification.py:746-753`):

| 위상 | 길이 | 지령 |
|---|---|---|
| hold | 0.5 s | neutral |
| step | 0.5 s | neutral + amplitude |
| hold | 0.5 s | 〃 |
| ramp | 1.0 s | +amplitude → −amplitude |
| **multisine** | **3.0 s** | neutral + wave·amplitude |
| hold | 0.5 s | neutral |

진폭은 `(upper − lower) × 0.05 × amplitude_scale`:

| scale | r_aj_1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| 0.30 | 4.20° | 3.00° | 2.70° | 2.10° | 2.70° | 1.35° | 2.70° |
| 0.65 | 9.10° | 6.50° | 5.85° | 4.55° | 5.85° | 2.92° | 5.85° |
| 1.00 | 14.00° | 10.00° | 9.00° | 7.00° | 9.00° | 4.50° | 9.00° |

★**검사 공간은 1차원이다.** 네 위상 모두 관절마다 **같은 스칼라**를 곱한다
(`values = neutral + wave[:, None] * amplitude`). 방문 자세는 `q(α) = neutral + α·amp`
라는 선분이지 2^7 개 조합이 아니다. 위상을 잇는 `bridge` 만 상자 안을 지난다.

---

## ★ 자세 선정 — 왜 R3 인가

`scripts/probe_excite_clearance.py` 로 계산했다. 여진이 **최저점을 얼마나 더 내리는지**(Δz):

| 자세 | 홈대비 높이 | Δz@0.30 | Δz@0.65 | Δz@1.00 | 최저 링크 |
|---|---|---|---|---|---|
| R2 팔접음 | +209 mm | 51.8 | 120.6 | 190.9 | thumb:r_aj_4 |
| **R3 손목맞춤** | **+204 mm** | **11.1** | **22.7** | **74.9** | **thumb:r_aj_4** |
| preset 정책홈 | 0 mm | 68.7 | 141.7 | 201.3 | pinky:tip |
| safe j2+0.40 | +95 mm | 77.7 | 161.3 | 230.3 | pinky:tip |

**R3 = `[0.038, 0.9, 0.6015, 2.0, 0.0294, 0.706, 0.4213]`** 를 고른 이유 셋:

1. **여유가 크다** — 정책 홈보다 204 mm 높다.
2. **Δz 가 압도적으로 작다** — scale 0.65 에서 22.7 mm. 손목이 접혀 있어 여진이
   손끝을 아래로 밀지 않는다. 같은 높이의 R2 보다 5 배 낫다.
3. **★손 전원과 무관하다** — R3 의 최저점은 팔꿈치(`thumb:r_aj_4`)다. 손가락을 전부
   펴도(전원이 없어 늘어져도) 최저점이 바뀌지 않는다. 정책 홈은 `pinky:tip` 이라
   손이 펴지면 Δz 가 68.7 → 85.6 mm 로 커진다.
4. R3 는 preset 궤적의 **실제 경유점**이라 거기까지 가는 경로가 이미 검증돼 있다
   (`contact_table` 합 0).

⇒ **R3 에서 amplitude_scale 0.65**. 안전하면서 진폭이 크다(2.9~9.1°).
   더 큰 신호가 필요하면 1.00 까지 여유가 있다(Δz 74.9 mm).

### 검산

이 FK 는 정책 홈의 최저점을 `pinky:tip` 으로 짚는다 — 08.31 에 사용자가 관찰한
"마지막 preset 자세는 새끼 손가락이 테이블에 닿고 있어"와 같은 손가락이다.

⚠**절대 높이는 주장하지 않는다.** 이 저장소에는 FK 가 둘 있고 서로 17 cm 어긋난다:
`hdgp/scripts/tools/openarm_fk.py` 는 fabrics 용 URDF + 손으로 맞춘 캘리브 오프셋이고,
이 도구는 자산 원본 `openarm_tesollo_sensor_rl.urdf` 를 쓴다. 어느 쪽이 sim 월드의
진실인지는 별도 문제이므로 **차분만** 낸다 — 차분은 오프셋과 무관하다.

---

## 실행 절차 (★각 단계 사용자 승인)

### 0) 준비 상태

- OpenArm 전원 **ON**, `openarm.bimanual.launch.py` bringup
- 손: 전원 여부 무관(R3 는 손 상태에 둔감). 켠다면 주먹 유지.
- 테이블: 있어도 된다 — R3 는 204 mm 위다.

### 1) 트랙 설계만 확인 (아무것도 안 움직인다)

```bash
R=/home/user/rl_ws/robot_control
$R/.venv/bin/robotctl r2s collect --group openarm_right_arm \
    --amplitude-scale 0.65 --dry-run
```
게이트가 트랙 전체를 미리 검사한다. 여기서 거절되면 실기는 손도 안 댄 것이다.

### 2) R3 까지 이동 (차렷 → R3, 811 프레임 = 16 s)

```bash
cd /home/user/rl_ws/sim2real
# 터미널 A — 연속 중력보상 (Ctrl-C 시 0 송출)
python3 scripts/gravity_comp_node.py --payload 0.9130,-0.00450,-0.01723,0.22147 \
    --scale 1.0 --execute
# 터미널 B
python3 scripts/shadow_replay.py --sim logs/shadow/reset_both/reset_right_to_R3.npz \
    --robot tesollo_sensor__right --arm-only --rate-scale 0.5 \
    --abort-tracking-err 0.3 --execute
```

⚠순서 규약(08.31): **재생이 먼저, 보상은 유지용**. 처진 상태에서 보상만 켜면 마찰
때문에 완전히 안 올라온다. 보상 노드를 먼저 띄우고 그 위에서 재생하는 것이 맞다.

### 3) 여진 3회

```bash
$R/.venv/bin/robotctl r2s collect --group openarm_right_arm \
    --amplitude-scale 0.65 --repetitions 3 \
    --output /home/user/rl_ws/sim2real/logs/r2s/right_R3_s065.npz --execute
```
중력보상은 **켠 채로 둔다** — `gravity_comp_node` 는 `/joint_states` 를 읽어 매 20 ms
다시 계산하므로 여진 중에도 따라간다. JTC 는 position 을, 보상은 effort 를 잡으므로
둘은 겹치지 않는다(`tau = kp(q_des−q) + kd(qd_des−qd) + tau_ff`).

소요 ≈ 6 s × 3 + 복귀 2회 ≈ 30 s. 짧아서 발열 여유가 있다.

### 4) 차렷 복귀

```bash
python3 scripts/shadow_replay.py --sim logs/shadow/reset_both/reset_right_to_R3_reverse.npz \
    --robot tesollo_sensor__right --arm-only --rate-scale 0.5 \
    --abort-tracking-err 0.3 --execute
```
그 다음 터미널 A 를 Ctrl-C (0 토크를 송출하고 나간다).

### 5) 적합

```bash
$R/.venv/bin/robotctl r2s fit --track .../right_R3_s065_manifest.json --output ...
```

---

## ★ 실행 후 알게 된 것 (09.01) — 게이트는 오버슈트를 못 막는다

`r_aj_6` 실측 최대 **+0.7738 rad, 한계 +0.785 → 여유 0.66°** 로 끝났다. 사전 계산은
지령 기준 0.89° 였는데 손목이 지령의 **2.07배**로 튀어 그만큼 더 밀었다.

`authorize_trajectory` 는 **지령만** 검사한다. 실측 오버슈트는 게이트 밖이다.
다음 여진에서 scale 을 올리려면 j6 이 한계에서 더 떨어진 자세를 고르거나,
아래 어림으로 여유를 확인할 것:

    필요 여유 ≳ 진폭 × 오버슈트배율 = 2.92° × 2.07 ≈ 6.0°   (scale 0.65 기준)

R3 에서 j6 은 지령 0.706, 실측 0.7185(중력보상 처짐으로 +0.72° 위) 였고 한계가
0.785 이므로 실제 여유는 3.8° 뿐이었다. **결과적으로 통과했지만 설계상 여유가 아니었다.**

결과 분석은 `EXCITE_RESULT_RIGHT_2026-09-01.md` 참조.

## 중단 조건

하나라도 걸리면 즉시 정지(터미널 B Ctrl-C → 터미널 A Ctrl-C):

- 관절 추종오차 > 0.3 rad — `shadow_replay` 는 `--abort-tracking-err 0.3` 으로 스스로
  멈춘다. `r2s collect` 에는 그런 옵션이 없으므로 3) 단계는 눈으로 지켜봐야 한다.
- `r_aj_7` effort > 5 N·m (07.29 에 로터 과열 0xC 로 래치된 전례)
- 모터 온도 > 60 ℃
- 눈으로 보아 테이블·몸통에 다가감

## 실패 시 복구

- 보상 노드가 죽어도 JTC 가 position 을 계속 잡는다 — 자유낙하하지 않는다.
  다만 12.8° 처진다. R3 는 204 mm 높으므로 그래도 테이블에 안 닿는다. **이것이
  R3 를 고른 두 번째 이유다.**
- effort 가 남으면: `$R/ros_ws/load_effort_controllers.sh --unload right`

## 남은 미검증

- **자기충돌(몸통·반대팔)은 이 도구가 보지 않는다.** R3 는 팔을 접은 자세라
  여진 ±9° 가 몸통에 가까워질 수 있다. 눈으로 지켜볼 것. sim 확인을 원하면
  GPU 승인을 받아 `probe_sim_follower.py` 로 미러해 볼 수 있다.
- 링크 **원점**만 본다(메시 껍데기가 아니라). 손가락 살집만큼 낙관적이라 판정에
  30 mm 여유를 넣었다.

---

## 준비물 점검 (09.01 작성 시점)

| 항목 | 상태 |
|---|---|
| `scripts/probe_excite_clearance.py` | ✅ 신규. 여진 Δz 계산·자세 비교 |
| `logs/shadow/reset_both/reset_right_to_R3.npz` | ✅ 신규. 차렷 → R3, 811 프레임 |
| `logs/shadow/reset_both/reset_right_to_R3_reverse.npz` | ✅ 신규. R3 → 차렷 |
| `scripts/gravity_comp_node.py` | ✅ 08.31. 연속 보상 |
| `robotctl r2s collect` | ✅ 기존. `--dry-run` 으로 먼저 확인 |
| 자기충돌 검증 | ⬜ 미검증 — 눈으로 볼 것 |

`shadow_replay` 는 램프가 **항상** 켜져 있다(별도 플래그 없음): 실측 자세에서
0.1 rad/s 로 첫 프레임까지 들어간다. 그래서 차렷이 아닌 자세에서 시작해도 안전하다.
