# sim2real 인터페이스 등가성 — 살아있는 체크리스트

> **이 문서의 규칙.** 가이드(`~/Downloads/sim2real_rl_debugging_guide.md`) §22 체크리스트를
> 표로 옮긴 것이다. **빈 칸이 곧 "부족한 세팅"** 이다. 추측으로 채우지 않는다 —
> 근거 파일이 없는 값은 `?` 로 둔다. 각 단계는 자기 절을 실측값으로 채워야 끝난 것으로 본다.
>
> 판정: ✅ 일치 · ❌ 불일치(원인 확정) · ⬜ 미측정 · ⚠️ 측정했으나 해석 미확정
>
> 대상: `openarm_tesollo_sensor_rl` (우팔 7 + DG-5F 우손 20 + 좌 2지 그리퍼 + D435i 헤드)
> 진행 기록: Notion `DRL / Robot_Control / 26.08.20~ 진행사항`
> 계획: `~/.claude/plans/home-user-rl-ws-sim2real-docs-plan-s2r-hashed-quasar.md`

**하드웨어 상태 (2026-08-20)**: 테솔로 핸드와 오픈암이 **분리**됨, 둘 다 local 5090 연결.
손목 중력토크가 작아 팔 단독 측정이 안전하다. 조립 후 재측정이 필요한 항목은 🔩 로 표시.

---

## 0. 저수준 제어 (Low-Level Control) — 가이드 §22

| # | 항목 | 기대 (sim) | 실측 (real) | 판정 | 근거 | 날짜 |
|---|---|---|---|---|---|---|
| L1 | 현재 자세 hold 가능 (팔) | 오차 0 (무중력·kp400) | ? | ⬜ | T1 | |
| L2 | 현재 자세 hold 가능 (손) | 오차 0 | ? | ⬜ | T1 | |
| L3 | 팔 전 관절 개별 command | 7/7 | ? | ⬜ | T2 | |
| L4 | 손 전 관절 개별 command | 20/20 | ? | ⬜ | T2 | |
| L5 | joint direction 일치 (팔) | 부호 +1 전 관절 | ? | ⬜ | T2 | |
| L6 | joint direction 일치 (손) | 부호 +1 전 관절 | ? | ⬜ | T2 | |
| L7 | joint zero/offset 검증 | offset 0 | ? | ⬜ | T2 | |
| L8 | command unit (rad) | rad | rad (선언) | ⚠️ | `profiles/openarm_tesollo.yaml` | 08.20 |
| L9 | 기동 직후 팔 거동 | — | **전 관절 0(차렷)으로 2초 램프** | ❌ | `openarm_simple_hardware.cpp:223-236`,`:305-353` | 08.20 |
| L10 | 기동 직후 손 거동 | 자세 유지 | **완전 무토크(limp)** — effort만 존재, enable 단계 없음 | ❌ | `system_interface.cpp:457-468`,`:476-488` | 08.20 |
| L11 | 실기 모드 확인 | — | `use_fake_hardware` 기본 **true** → `:=false` 필수 | ❌ | `openarm.bimanual.launch.py:181-184` | 08.20 |

## 1. 제어 모드 (Control Mode)

| # | 항목 | sim | real | 판정 | 근거 |
|---|---|---|---|---|---|
| C1 | 팔 명령 인터페이스 | position target (`set_joint_position_target`) | position (JTC) → MIT `kp(q_des−q)+kd(qd_des−qd)+tau_ff` | ✅ 의미 일치 | `grasp_lift_env.py:196`; `openarm_simple_hardware.cpp:285-303` |
| C2 | 팔 `qd_des` | 0 (속도 타깃 미사용) | 0 (JTC가 position만 claim) | ✅ | 위 동일 |
| C3 | 팔 `tau_ff` | 해당 없음 | **0** (effort controller 미로드) | ❌ 중력보상 없음 | `ros_ws/load_effort_controllers.sh` |
| C4 | 손 명령 인터페이스 | position target | **effort** → JTC 내부 PID(p=1.5,i=0,d=0) → PWM duty | ❌ 의미 다름 | `dg5f_right_controller.yaml:64-68`,`:70-210` |
| C5 | 손 effort 단위 | N·m | **정규화 duty**, `\|effort\|=0.9946`에서 포화 | ⚠️ 벤더 바이너리 해석 — M4로 확증 | `libdelto_gripper_helper.so` `ConvertDuty` |
| C6 | 손 effort **상태값** | N·m | **모터 전류** (`efforts_ = current_`) | ❌ 단위 다름 | `system_interface.cpp:536-539` |
| C7 | 중력 | **로봇 `disable_gravity=True`** | 실재 | ❌ | `grasp_lift_env_cfg.py:42` |
| C8 | 워치독 | 해당 없음 | **하드웨어에 없음** (마지막 포인트를 full kp로 유지) | ⚠️ 안전 설계 필요 | 조사 08.20 |

## 2. 액션 (Action)

| # | 항목 | sim | real(배포) | 판정 | 근거 |
|---|---|---|---|---|---|
| A1 | raw action 차원 | ? (트랙별) | ? | ⬜ | T3 |
| A2 | action scale | ? | ? | ⬜ | T3 |
| A3 | offset / default joint pos | ? | ? | ⬜ | T3 |
| A4 | processed action 동일 | — | — | ⬜ | T3 |
| A5 | 브리지 rate-limit | 없음 | 0.1~0.5 rad/s (**근거 없는 값**) | ⚠️ | `isaacsim_cmd_to_jtc.py:173-176` |
| A6 | 정책 요구 관절속도 | ? | — | ⬜ | P5 |

## 3. 관측 (Observation)

| # | 항목 | sim | real | 판정 | 근거 |
|---|---|---|---|---|---|
| O1 | shape | ? | ? | ⬜ | P3 |
| O2 | 세그먼트 순서 | ? | ? | ⬜ | P3 |
| O3 | joint order | canonical `r_aj_*`/`r_hj_*` | 이름 기반 재정렬 | ✅ 설계상 | `robot_profile.py` |
| O4 | unit | rad, m | rad, m | ⬜ 확인 필요 | P3 |
| O5 | coordinate frame | ? | ? | ⬜ | P3 |
| O6 | quaternion convention | ? | ? | ⬜ | P3 |
| O7 | normalization | ? | ? | ⬜ | P3 |
| O8 | previous action | 유지 | 유지 | ⬜ | P3 |
| O9 | tip force 프레임·부호 | tip-local | **미검증** | ⬜ | P3 |
| O10 | `/cup_pose` 정확도 | 참값 | **미검증** (extrinsics는 pan −90/tilt 280 전용) | ⬜ | P6 |

## 4. 타이밍 (Timing)

| # | 항목 | sim | real | 판정 | 근거 |
|---|---|---|---|---|---|
| T-1 | 물리 주기 | 120 Hz | — | — | `grasp_lift_env_cfg.py` |
| T-2 | 정책 주기 | 60 Hz (decimation 2) | 60 Hz (목표) | ⬜ 실측 필요 | |
| T-3 | 팔 controller_manager | — | **750 Hz** | ✅ | `openarm_bimanual_controllers.yaml:18` |
| T-4 | 손 controller_manager | — | **100 Hz** | ✅ | `dg5f_right_controller.yaml:3` |
| T-5 | 팔 state 발행 | — | 50 Hz (JTC state) | ⚠️ | `:219-220` |
| T-6 | 손 state 발행 | — | 500 Hz | ✅ | `:37-38` |
| T-7 | end-to-end latency | 0 | ? | ⬜ | P4 |

## 5. 안전 (Safety)

| # | 항목 | 상태 | 근거 |
|---|---|---|---|
| S1 | emergency stop | ⬜ 절차 문서화 필요 | |
| S2 | velocity limit | ⚠️ 브리지 rate-limit만 (근거 없는 값) | A5 |
| S3 | acceleration limit | ⬜ 없음 | |
| S4 | joint limit | ✅ 프로필 주입 | `robot_profile.py` |
| S5 | invalid action 처리 | ✅ 차원 불일치 시 예외 | `grasp_obs_builder.py` |
| S6 | 명령 스트림 두절 | ⚠️ 하드웨어 홀딩(워치독 없음) | C8 |

---

## 6. 중력 처짐 예측 (URDF 계산, 2026-08-20)

펌웨어 게인 `kp = {70,70,70,60,10,10,10}` 기준, agnostic 홈 자세.
계산: `scratchpad/probe_droop_nohand.py` (`sim2real/scripts/arm_inertia.py` 재사용)

| 관절 | kp | τ 손장착 [N·m] | τ 손분리 | 처짐 장착 | 처짐 분리 | 실측 |
|---|---|---|---|---|---|---|
| r_aj_1 | 70 | −8.10 | −2.68 | −6.6° | −2.2° | ⬜ |
| r_aj_2 | 70 | −8.39 | −4.69 | −6.9° | −3.8° | ⬜ |
| r_aj_3 | 70 | −2.89 | −0.90 | −2.4° | −0.7° | ⬜ |
| r_aj_4 | 60 | −8.50 | −3.17 | −8.1° | −3.0° | ⬜ |
| r_aj_5 | 10 | −2.02 | −0.16 | −11.6° | −0.9° | ⬜ |
| r_aj_6 | 10 | +0.01 | +0.03 | +0.1° | +0.1° | ⬜ |
| r_aj_7 | 10 | −3.14 | −0.32 | **−18.0°** | −1.8° | ⬜ |

- 🔩 손 장착 후 재측정 필수. URDF 손 COM 이 손목에 붙어 있어 실제 손목 토크는 **2~3배 클 수 있다**
  (07.29 실측) → 장착 시 실측치는 −18° 보다 나쁠 수 있다.
- 참고 실측(robot_control, 다른 자세): 팔꿈치 **0.179 rad = 10.3°**,
  `pose ee` 1회 이동이 **58~85 mm** 빗나감.

---

## 6-1. F1/F2 독립 검증 (2026-08-20)

설계 검토가 제기한 두 블로커를 **URDF 로 직접 재계산해 확인**했다(Isaac 불필요).
계산: `scratchpad/probe_f1_torque_authority.py`, `probe_f2_home_settle.py`

**F1 — 한 액션 스텝의 홀딩 토크 ÷ 중력토크** (diff IK 가 실측 EE 에 재앵커하므로 위치오차가
액션 한 스텝으로 상한된다. `|Δq| = |J†δ|`, δ = 1 cm / 0.05 rad, dls λ=0.01)

| 관절 | τ_g | \|Δq\|max | @펌웨어 | @r2s | **@sim 400** |
|---|---|---|---|---|---|
| r_aj_1 | −8.10 | 0.0434 | **0.38** | **0.36** | 2.14 |
| r_aj_2 | −8.39 | 0.0364 | **0.30** | **0.29** | 1.73 |
| r_aj_3 | −2.89 | 0.0253 | **0.61** | **0.59** | 3.50 |
| r_aj_4 | −8.50 | 0.0680 | **0.48** | **0.54** | 3.20 |
| r_aj_5 | −2.02 | 0.0394 | **0.20** | **0.23** | 7.81 |
| r_aj_7 | −3.14 | 0.0913 | **0.29** | **0.35** | 11.62 |

→ **실측 게인에서 부하 관절 전부 1.0 미만 = 팔이 자기 무게를 못 든다.**
→ **sim 현재 400 에서는 전부 1.0 초과** = 지금은 문제 없고, 게인을 내리는 순간 무너진다.
   S2(중력 ON @400/80) 와 S3(게인 내리기)를 분리해야 하는 이유가 여기서 수치로 확인된다.

**F2 — 홈 자세 정착 위치** (정적 평형 `kp(q_cmd−q*) = τ_g(q*)`, 마찰 데드밴드 포함)

| 게인 | Δq 최대 [mrad] | palm 이동 | 테이블 아래 손끝 |
|---|---|---|---|
| sim 400/80 | −21 | **21 mm** | **0/5** ✅ |
| 펌웨어 70/60/10 | −266 (r_aj_7) | **142 mm** | **4/5** ❌ |
| r2s 식별 + 마찰 | −219 (r_aj_7) | **129 mm** | **4/5** ❌ |

명령 홈 palm = `[0.2800, −0.3801, 0.4178]` (의도 0.28/−0.38/0.42 — 무중력 IK 는 정확).
실측 게인에서 검지 팁 z 0.443 → **0.239** (테이블 상면 0.29) = **손가락이 테이블 속 50 mm**.
→ F2 확정. `init_joint_pos` 를 중력 사전보상 명령 자세로 재유도해야 한다(S4).

*(교차검증: 독립 설계검토가 낸 값 palm 128 mm / 검지팁 0.240 과 일치)*

---

## 7. 미해결 미지수 (측정 전까지 어떤 값도 근거로 쓰지 않는다)

| # | 미지수 | 후보값들 | 해소 방법 |
|---|---|---|---|
| U1 | 실기 손의 물리 강성 [N·m/rad] | JTC p=1.5(duty 단위) / r2s 36.15(**합성데이터**) / sim 5.0 | M4 + M5 |
| U2 | 손 최대 토크 τ_max [N·m] | `effort_limit_sim=1.5` (**사용자 단언, 측정 아님**) | M4 (F/T 계단) |
| U3 | 팔꿈치 kp | 66.979(**탐색 상한 고착**) / 60.9(계단 실측) / 60(펌웨어) | M3 |
| U4 | 팔 damping | 펌웨어 2.0 / 식별 5.635 | M2 → V2 |
| U5 | 손 duty 변환 상수 | 12.065/12 × 100 (**바이너리 해석**) | M4 |
| U6 | 전송 지연 | 0 (sim) | `delay_steps` 미적용 — P4 |

---

## 8. 발견: 액션 기준점이 sim 과 배포에서 다르다 (2026-08-20, `grasp_sensor`)

**계약 파리티 테스트가 잡아낸 실제 불일치.** `test_palm_delta_y_reaches_far_cup` 이 RED 가 되어
추적한 결과:

| | sim (`grasp_sensor`, hdgp `c99b37d` 이후) | 배포 (`grasp_policy_core`) |
|---|---|---|
| 물리 리셋 | 고정 홈 `(0.28, −0.38, 0.42)` | 고정 홈 ✅ 일치 |
| **액션 기준점** | **컵 정준 pregrasp** (`obj_pos + pregrasp_offset + noise`, rot 90/0/90, workspace clamp) | **홈** ❌ |
| `action = 0` 의 뜻 | "정렬된 pregrasp 로 접근하라" (Fabrics 가 홈→pregrasp 를 스스로 감) | "홈에 머물러라" |
| `palm_delta_xyz` | `(0.15, 0.15, 0.15)` — 기준점이 컵이라 0.15 로 충분 | 0.15 를 **홈** 기준으로 해석 → 먼 컵(Δy 0.28) **도달 불가** |

근거: `grasp_right_env.py:2005-2016`(`pregrasp_palm_pose_buf`, "액션 기준점"),
`:2051-2053`("delta action 기준점: action=0 → pregrasp 위치 유지"),
`grasp_policy_core.py:63-73`(`base = home_pose`).
sim 주석이 배포 요구사항을 명시하고 있다 — *"실기 미러: grasp_inference 가 인지된 컵 pose 로
동일 기준점을 계산한다"* — **그런데 미구현이다.**

- 이력: `(0.15, **0.35**, 0.15)` → `(0.15, **0.15**, 0.15)` 로 바꾼 커밋이
  `c99b37d "고정홈 fresh 단일 레시피 — 컵-정준 액션 기준점"`. 값만 보면 회귀처럼 보이지만
  **기준점이 함께 바뀌었으므로 회귀가 아니다.** 값 하나만 비교하면 오판한다.
- 영향: 배포로 `grasp_sensor` 정책을 돌리면 팔이 홈 근처에서만 움직인다 —
  **증상 B "제자리에서 움직임만" 과 정확히 같은 그림**이다.
- **해소됨 (2026-08-20)**: 배포가 기준점을 **sim 소스에서 유도**해 고른다.
  `robot_profile.load_action_anchor()` 가 `grasp_{side}_env.py` 의 고정 홈 분기에서
  `pregrasp_palm_pose` 를 `home_palm_pose` 로 덮어쓰는지 AST 로 판별한다 —
  덮어쓰면 `home`, 그대로면 `cup`. **프로필에 손으로 적지 않는다**(적으면 sim 변경과
  조용히 어긋난다). cup 구성은 리셋 때 컵 pose 가 없으면 예외이고, 기준점이 확립되기 전
  명령을 내면 `require_anchor_established()` 가 예외를 던진다 — 홈으로 조용히 되돌아가지
  않는다.
- ★교훈: **`palm_delta` 같은 스칼라를 sim 과 대조하는 것만으로는 부족하다.** 그 값이 무엇을
  기준으로 한 델타인지(기준점)까지 계약에 넣어야 한다. 프로필에 `action_anchor: home | cup`
  같은 필드를 추가하는 것이 P3 의 후보다.


---

## 9. Fabrics 단독 검증 (2026-08-20) — 하드웨어·Isaac Sim 앱 없이

`fabrics_sim` 이 **Isaac Sim 앱 없이 단독 실행**된다(warp 1.8.1 + Isaac 번들 python,
`PYTHONPATH=hdgp/source/FABRICS/src`). 그래서 배포 제어 경로 전체
(obs → 정책 → 디코드 → **액션 기준점** → Fabrics → 관절 명령)를 실기 없이 검증할 수 있다.
이것이 "실기 외 준비"의 핵심 지렛대다.

**zero-action 기하 검증** (`scratchpad/probe_anchor_fabrics.py`, 체크포인트 불필요):
컵 `(0.30, −0.20, 0.297)`, 정책을 zeros 스텁으로 대체, 완전 추종 가정(sim 과 동일).

| 구성 | anchor | 기준점 palm | step 0 | step 360 | 컵까지 | 기준점 잔차 |
|---|---|---|---|---|---|---|
| `tesollo_sensor__right` | **cup** | `[0.240, −0.270, 0.297]` | `[0.290, −0.394, 0.415]` | `[0.239, −0.270, 0.301]` | 0.093 m | **4.4 mm** |
| `tesollo_bi_s__right` | **home** | `[0.280, −0.380, 0.420]` | `[0.281, −0.382, 0.420]` | `[0.279, −0.361, 0.420]` | 0.204 m | 18.8 mm |

- cup 구성: action=0 에서 **홈 → 컵 정준 pregrasp 로 1초(60스텝) 만에 수렴**.
  컵까지 남은 거리 0.093 m = `|pregrasp_offset|` = √(0.06²+0.07²) = 0.0922 ✅
  sim 계약 *"action=0 이면 Fabrics 가 홈에서 정렬된 pregrasp 까지 스스로 접근한다"* 와 일치.
- home 구성: action=0 에서 **홈에 머문다** ✅ (잔차 18.8 mm = Fabrics null-space 정상상태)
- 두 거동이 **구성에서 자동으로 갈린다** — 배포 코드에 분기 리터럴이 없다.
