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

---

## 6-2. 좌팔 + 2지 그리퍼 (2026-08-25) — §6·§6-1 을 그대로 인용하면 안 된다

§6·§6-1 은 **우팔 + DG-5F 20관절 손** 기준이다. 다음 s2r 대상인
`open-grip_l_grasp_sensor_fab` 은 좌팔 + 2지 그리퍼이고 끝단 질량이 전혀 다르므로 같은
계산을 좌팔 체인으로 다시 했다. 재현: `python3 scripts/report_left_arm_gravity.py`
(URDF·preset·control_gains.yaml 만 읽는다. Isaac·하드웨어 불필요)

홈 `[-0.0136, -0.3757, -0.0010, +0.9336, -0.4655, +0.0003, -0.3306]`,
명령 홈 TCP(base) `[0.2717, 0.3073, 0.3442]`.

**F1 — 한 액션 스텝(palm 5 mm / 2.9°)의 홀딩토크 ÷ 중력토크**

| 관절 | τ_g [N·m] | \|Δq\| [rad] | @펌웨어 | @sim 400 | I [kg·m²] |
|---|---|---|---|---|---|
| l_aj_1 | −3.921 | 0.0049 | **0.09** | 0.50 | 0.3608 |
| l_aj_2 | −3.772 | 0.0023 | **0.04** | 0.25 | 0.3214 |
| l_aj_3 | −1.408 | 0.0009 | **0.05** | 0.26 | 0.0891 |
| l_aj_4 | +3.633 | 0.0218 | **0.36** | 2.41 | 0.1247 |
| l_aj_5 | +0.051 | 0.0138 | 2.69 | 107.49 | 0.0023 |
| l_aj_6 | +0.020 | 0.0524 | 25.79 | 1031.67 | 0.0096 |
| l_aj_7 | −0.742 | 0.0449 | **0.61** | 24.23 | 0.0101 |

★ **δ 가 다르면 F1 은 비교 불가다.** §6-1 의 우팔 표는 δ = 1 cm / 0.05 rad 로 냈고 여기는
정책이 실제로 낼 수 있는 상한(`PALM_CMD_RATE_LIMIT` 5 mm, `PALM_ROT_RATE_LIMIT` 0.05 rad)을
썼다. 그래서 좌팔 숫자가 더 작게 보이는 것이지 좌팔이 더 나쁜 것이 아니다 —
**비교해야 할 것은 F2 다.**

**F2 — 홈 자세 정적 정착**

| 게인 | Δq 최대 [mrad] | 최악 관절 | TCP 이동 |
|---|---|---|---|
| sim 400/80 | +9.5 | l_aj_1 | **8.8 mm** |
| 펌웨어 70/60/10 (좌우 공통) | +69.5 | l_aj_7 | **53.0 mm** |
| r2s 식별(★우팔 값) | +58.2 | l_aj_7 | 51.6 mm |

→ 좌팔 처짐은 우팔의 **절반 이하**다(53 mm vs 129~142 mm). 2지 그리퍼가 DG-5F 보다 훨씬
가볍다. 그래도 53 mm 는 파지 대역폭(그리퍼 개구 44 mm)보다 크다 — 무시할 크기가 아니다.

**대역폭 — 2차계 (홈 자세 대각 근사, 결합·코리올리 무시)**

| 관절 | 펌웨어 f_n | 펌웨어 ζ | sim f_n | sim ζ | 비 |
|---|---|---|---|---|---|
| l_aj_1 | 2.22 Hz | **0.27** | 5.30 | 3.33 | 2.39 |
| l_aj_2 | 2.35 | **0.26** | 5.62 | 3.53 | 2.39 |
| l_aj_3 | 4.46 | **0.40** | 10.66 | 6.70 | 2.39 |
| l_aj_4 | 3.49 | **0.37** | 9.01 | 5.66 | 2.58 |
| l_aj_5 | 10.55 | 2.32 | 66.73 | 41.93 | 6.32 |
| l_aj_6 | 5.14 | **0.97** | 32.48 | 20.41 | 6.32 |
| l_aj_7 | 5.02 | **0.79** | 31.75 | 19.95 | 6.32 |

대역폭비 2.39~6.32(§③ 우팔 3.86~5.77 과 같은 급 — 팔이 같으니 당연하다). 눈여겨볼 것은
**감쇠비**다: 펌웨어 게인에서 j1~j4 가 ζ 0.26~0.40 으로 **크게 부족감쇠**이고 sim 은 전
관절 과감쇠(ζ 3.3~41.9)다. 정책은 오버슛도 링잉도 겪어본 적이 없다.

**★ 좌팔에는 실측 캘리브레이션이 없다.** `right_arm_best_calibration.json` 안의
`openarm_left_arm` 항목은 400/80 인데 이건 측정치가 아니라 **sim 기본값이 그대로 남은 것**
이다(U3~U4 와 같은 부류의 함정). 위 "식별" 열은 우팔 값을 좌팔 관성에 얹은 참고치다.

**★ 이 태스크는 sim 에서도 중력을 켜고 학습했다** — `grasp_left_env_cfg.py:123,281`
`disable_gravity=False`, 계약 테스트가 강제한다(`test_lift_contract.py:559`). C7 의
"sim 은 중력을 끈다"는 `grasp_lift` 트랙 얘기이고 **이 트랙에는 해당하지 않는다.**
즉 좌 그리퍼 격차는 "중력 유무"가 아니라 **게인 차이 하나**로 좁혀진다. 더해서 액션 항이
관절공간 처짐 적분 보상을 넣는데(`grasp_left_fabric_action.py`, `GRAVITY_COMP_GAIN` 0.05,
상한 = effort/강성), 그 상한이 **sim 강성 400 기준**이라 실기 70 에서는 뜻이 달라진다 —
그림자 기록에서 `fabric_q` 와 처짐 보정분을 반드시 분리해 남길 것.

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

---

## 10. ★블로커 — 배포 파이썬 환경이 아예 없었다 (2026-08-20 발견 · 해소)

`grasp_inference.py` 를 실행할 수 있는 인터프리터가 이 머신에 **하나도 없었다.**

| 인터프리터 | rclpy | torch | warp | rl_games | fabrics_sim | GPU |
|---|---|---|---|---|---|---|
| ROS Humble `python3.10` | ✅ | 2.2.1+**cu121** | ❌ | ❌ | ❌ | ❌ **sm_90 까지** |
| Isaac 번들 `python3.11` | ❌ | 2.7.0+cu128 | ✅ | ✅ | ❌(경로) | ✅ |

- ROS 쪽 torch 는 `arch_list = [sm_50 … sm_90]` 인데 **RTX 5090 은 sm_120** →
  `RuntimeError: CUDA error: no kernel image is available for execution on the device`.
  즉 ROS 인터프리터로는 **정책도 Fabrics 도 GPU 에서 못 돌린다**.
- Isaac 쪽은 GPU 는 되지만 `rclpy` 가 없다(ROS Humble 의 rclpy 는 py3.10 빌드).
- 저장소에 `file_command_transport.py`("rclpy 를 import 할 수 없는 환경용")가 있는 걸 보면
  과거에 이 벽을 만나 우회를 시도한 흔적이 있다. 다만 그건 **명령 방향 단방향**이라
  60 Hz 센서 수신(관절·컵·tip)은 못 덮는다.

**해소**: `scripts/setup_deploy_env.sh` — ROS 의 python3.10 위에
`--system-site-packages` venv 를 만들어 rclpy 를 상속하고, torch(cu128)·warp·rl_games·
fabrics_sim 만 venv 안에서 덮어쓴다. 시스템은 건드리지 않는다.

| 항목 | 값 | 비고 |
|---|---|---|
| python | 3.10.12 | ROS Humble 과 동일 |
| torch | **2.7.1+cu128** | `sm_120` 포함, 5090에서 CUDA op 검증 |
| warp-lang | 1.8.1 | fabrics_sim 요구 `<1.8.2`, Isaac 과 동일 |
| rl-games | 1.6.1 | Isaac 학습 환경과 동일(체크포인트 정합) |
| numpy / scipy | 1.26.4 / 1.13.1 | fabrics_sim 요구 `numpy<2.0` — torch·rl_games 가 2.x 를 끌어오므로 **명시 핀 필수** |
| urdfpy | 0.0.22 + 패치 | 끌려오는 networkx 2.2 가 py3.10 에서 죽음 → `urdfpy_patch.sh` (venv 안에서만) |

**검증**: 이 환경에서 zero-action Fabrics 궤적이 Isaac python 결과와 **완전히 동일**하다
(§9 표와 같은 값). 테스트 스위트 417 passed / 23 skipped.
`grasp_inference.py --help` 도 import 를 전부 해결한다.

실행 전 **반드시** `source /opt/ros/humble/setup.bash` (rclpy 상속 전제).

---

## 11. 관측 계약 자동 추출 (2026-08-20) — "여러 환경 세팅"에 대한 답

hdgp 에는 관측을 정의하는 태스크가 **22개** 있다(grasp_v1/v2/v7_2/v10_3/v11/adapt/sensor ·
pour_v1/v3/v4/v5/sensor · rh56f1 · gripper · agnostic ×2). 어느 것을 배포 대상으로 삼든
obs 계약을 손으로 옮겨 적는 순간 드리프트가 시작된다 → **소스에서 자동 추출**한다.

`scripts/obs_contract.py` + `scripts/obs_contract_report.py`

```bash
python3 obs_contract_report.py                                # 전 태스크 요약
python3 obs_contract_report.py --task tesollo/right/grasp_sensor   # 상세(정의식·상수·노이즈)
python3 obs_contract_report.py --diff tesollo_sensor__right        # 배포 빌더와 대조
```

**결과: 22/22 태스크 추출 성공.** 형태가 3가지라 전부 다뤄야 했다 —
`torch.cat([...])` 리터럴 · `parts = [...]; cat(parts)` 우회(4개 태스크) ·
`obs = cat(...)` 뒤 `obs = nan_to_num(obs)` 재대입(4개 태스크).

### 자동으로 나오는 것 / 안 나오는 것 (정직하게)

| | 자동 | 근거 |
|---|---|---|
| 세그먼트 **순서·이름** | ✅ | 재배열·개명·추가를 즉시 잡는다 |
| 각 세그먼트 **정의식** | ✅ | 프레임 힌트(`_local`)·차분(`a − b`)이 보인다 |
| **정규화 상수**(ALL_CAPS) | ✅ | `CONTACT_FORCE_MAX`, `JOINT_POS_ERR_MAX` 등 |
| **DR 노이즈** 적용 여부 | ✅ | `randn_like` 등 검출 |
| 세그먼트 **차원** | ❌ | 대부분 상위 스코프·`self.*` 라 정적으로 못 센다 |
| **프레임·단위의 의미** | ❌ | world vs local, rad vs deg 는 사람 판단 |

차원과 의미는 **런타임 덤프 대조(P3 본편)로만** 확정된다. 이 도구는 그 앞단의 뼈대를 준다.

### 부수 발견 — sim 은 obs 에 DR 노이즈를 더한다
`grasp_sensor` 는 `arm_joint_pos`·`arm_joint_vel`·`finger_joint_pos`·`finger_joint_vel`·
`palm_center_pos` 5개 세그먼트에 `randn_like` 노이즈를 더한다(`pour_*` 는 4개).
**배포는 노이즈를 더하지 않는다** — 실센서가 곧 노이즈다. 이건 올바른 비대칭이고,
`--diff` 가 매번 그 목록을 찍어 잊지 않게 한다.

### 기존 파리티 테스트를 이 추출기로 교체
구 정규식 추출기는 단순 식별자가 아닌 항을 **조용히 버렸고**, `parts` 우회·`nan_to_num`
재대입 태스크에서는 아예 찾지 못해 `pytest.skip` 으로 넘어갔다(= 검사 안 됨).
AST 추출기로 바꾼 뒤 세그먼트 순서를 일부러 뒤바꿔 3개 프로필 전부 RED 되는 것을 확인했다.
