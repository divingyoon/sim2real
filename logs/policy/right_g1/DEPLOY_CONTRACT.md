# 우팔 s2r 배포 계약 — `g1_ep17000` (2026-09-03 감사)

> 이 문서의 값은 전부 **런 dump(`params/`) 또는 env 소스에서 직접 읽어** 확인한 것이다.
> 기억이나 다른 런에서 옮겨 적은 값은 없다. 09.03 좌팔에서 하루를 태운 결함 네 개가
> 전부 "상수를 손으로 옮기다 어긋난 것"이었다 — 그래서 재도출하지 말고 여기를 볼 것.

## 왜 g1 인가 (다른 우팔 체크포인트는 배포 불가)

| 런 | 팔 kp (j1~4 / j5 / j6 / j7) | 배포 |
|---|---|---|
| b1 · d3 · e1 · m1 | **300 / 100 / 50 / 25** | ✗ |
| **g1** (`g1_rot20_fresh`) | **70·70·70·60 / 10 / 10 / 10** | ✓ |
| 실기 `control_gains.yaml` v10 | 70·70·70·60 / 10 / 10 / 10 | — |

09.03 실측: 좌팔에 kp 300/45 를 올렸더니 **실기가 발진**했다(사용자 청취). sim 에는
CAN 지연이 없어 그 게인이 멀쩡히 돌지만 실기에서는 위상 여유를 다 먹는다. 따라서
kp 300 계열 체크포인트(d3 포함 — 메모리의 "09.01 확정"은 이 근거로 **폐기**)는
게인을 맞출 수 없어 재현 불가다.

g1 의 kd(7.05·4.18·7.80·6.53 / 2.24 / 0.58 / 0.24)는 벤더 kd 와 다르지만 **r2s 로
동정한 값**이다 — 실기의 마찰까지 포함해 sim 이 실기처럼 거동하도록 맞춘 것이라
오히려 더 충실하다(kp 는 벤더값 고정, kd 만 적합: `r2s_pipeline`).

## 체크포인트

```
logs/policy/right_g1/nn/g1_ep17000.pth
  task  open-sens_r_grasp_s2r-lstm   ·   full  g1_rot20_fresh   ·   epoch 17000
  obs 155 · act 21 · state 193 · **LSTM**(hidden 1024 → actor_mlp 입력 1179)
  clip_actions(params.env) = 1.0
```

★`clip_actions = 1.0` 이고 env 도 `self.actions = actions.clamp(-1, 1)` 로 **자른 값을
obs 에 넣는다** — 좌팔 fab79(=100, 자르지 않음)와 정반대다. 배포 로더의
`action_clip` 은 반드시 런의 `params.env.clip_actions` 에서 읽을 것.

## 주기

```
decimation 2 × sim dt 0.008333 = env.step_dt 0.016667 s  (정책 60 Hz)
episode_length_s 10.0
```

★좌팔은 0.02 다. fabric 적분 dt = `env.step_dt` 이므로 **런마다 다르다** —
`step_dt_from_run()` 으로 읽을 것(09.03 좌팔에서 1/60/dec 를 박아 2.4배 틀렸다).

## 관측 155D — env 소스와 순서 일치 확인

`grasp_s2r_env._get_observations` 의 actor `torch.cat` 순서 그대로:

| 슬롯 | 항 | 차원 | 비고 |
|---|---|---|---|
| 0..6 | arm_q | 7 | **절대값**(좌팔과 달리 상대 아님) |
| 7..13 | arm_qd | 7 | |
| 14..33 | hand_q | 20 | ★**Isaac DOF 순**(canonical 아님) |
| 34..53 | hand_qd | 20 | 〃 |
| 54..56 | palm_pos | 3 | env-local |
| 57..62 | palm_ax | 6 | `_palm_ee_R()` 의 **열 0·1** |
| 63..77 | tips_rel_palm | 15 | tip − palm (world 차, 프레임 무관) |
| 78..80 | palm_to_obj | 3 | env-local |
| 81..95 | obj_to_tips | 15 | env-local |
| 96..110 | tip_force | 15 | **팁 로컬** / `contact_force_max`, ±1 클램프 |
| 111..130 | joint_err | 20 | (시너지목표 − 실측) / `joint_pos_err_max`, ±1 |
| 131..151 | actions | 21 | **클램프된** 직전 액션 |
| 152..154 | goal_rel | 3 | 목표 − 컵 |

배포 모듈 `scripts/grasp_s2r_obs_builder.py` 의 `SEGMENTS` 와 **완전 일치**한다
(09.03 소스 대조). 손 20관절 DOF 순 규약도 그 모듈 docstring 에 있다.

## 액션 21D

**팔 `a[0:6]`** — 앵커 + 델타(절대 박스가 아니다):

```
delta        = 0.5*(a[0:6]+1)*(hi-lo) + lo          # 성분별
palm_target  = anchor + delta                        # 6D = pos3 + euler3
palm_target  = clamp(palm_target, box_lo, box_hi)
# 변화율 리미터 — 첫 지령은 걸지 않는다(_palm_cmd_primed)
#   위치: 벡터 노름 기준 스케일링, 상한 palm_cmd_rate_limit_m
#   회전: 벡터 노름 기준 스케일링, 상한 palm_cmd_rate_limit_rot_deg
```

| 상수 | g1 값 | 출처 |
|---|---|---|
| `palm_delta_xyz` | (0.1, 0.1, 0.1) | dump |
| `palm_delta_rot_deg` | **20.0** (= 이름의 rot20) | dump |
| `palm_anchor_mode` | **spawn** | dump |
| `palm_anchor_offset_xyz` | (−0.066, −0.022, 0.085) | dump |
| `palm_cmd_rate_limit_m` | 0.02 | dump |
| `palm_cmd_rate_limit_rot_deg` | 2.9 | dump |
| `palm_box_min` / `max` | (0.20, −0.55, 0.20) / (0.55, 0.22, 0.70) | **프로필 코드**(`robot_profiles.tesollo_right`) |
| 홈 palm | (0.28, −0.38, 0.42) / ez90·ey0·ex90 | 프로필 `reset_home_palm_pose` |
| 홈 관절 | r_aj_1..4 = 0.0380, 0.4012, 0.6015, 0.9643, … | 프로필 |

★`box_lo = min(palm_box_min, home_palm)`, `box_hi = max(palm_box_max, home_palm)` —
앵커가 항상 박스 안이어야 `a=0` 의 의미가 유지된다.
★앵커가 **spawn** 이므로 배포는 **에피소드 시작 시 물체 위치를 한 번 스냅샷**해
`anchor = spawn + offset` 으로 고정해야 한다. 실시간 물체를 쓰면 액션 원점이 따라
움직이는 되먹임이 된다(env 주석의 명시적 경고).

**손 `a[6:21]`** — 시너지(절대 폐쇄도 목표):

| 상수 | g1 값 |
|---|---|
| `hand_layout` | **coupled3** |
| `couple_four_fingers` | **true** (엄지만 독립, 나머지 채널별 평균) |
| `synergy_close_speed` | 0.005 (목표를 향한 **변화율 상한**, 속도지령 아님) |
| `close_gate_enabled` / `close_gate_ramp` | true / 0.5 |
| `grasp_z_deadband` | 0.03 |

★닫기 게이트: `cage = palm + R·cage_offset_palm`, `d = banded_dist(cage − obj)`,
`g = clamp((r_cage − d)/(ramp·r_cage), 0, 1)`, 래치 후에는 1 로 해제.
`r_cage` 와 `cage_offset_palm` 은 **홈 자세의 손끝 FK 에서 측정**해 고정한다
(엄지 tip vs 나머지 4 tip 평균의 중점·반거리) — 배포도 같은 방식으로 재현 가능.

## 정규화 상수

```
contact_force_max      10.0     # tip_force 나눗수
contact_force_threshold 1.0
joint_pos_err_max       1.2     # joint_err 나눗수
```

## 기타

- `enable_adr: false` — ADR 없음(좌팔 v2E29 는 level 4 였다)
- `obs_object_noise_coherent: true` — 물체 파생 obs 3항이 같은 추정값에서 나온다
- 재소환 있음이나 **`respawn_penalty: 2.0`** — 좌팔 v2 의 공짜 재소환과 다르다
- 손 액추에이터(sim) kp 5.0 / kd 2.0 · 실기 튜닝값은 p=4.5 (bringup 마다 재적용)

## 아직 없는 것 (배포 잔여 작업)

1. `grasp_s2r` 액션 디코더 — palm 앵커·델타·박스·리미터 + 시너지 + 닫기 게이트
2. `grasp_s2r` 정책 코어 — FK(palm·5 tip) → obs 155 → LSTM → 액션 → fabric
3. 우팔 라이브 노드 — 기존 `grasp_inference.py` 는 **grasp_v1(154D·비LSTM)** 계약이라 못 쓴다
4. 실기 배선 — 손끝 F/T 15D(`tip_force`), Tesollo 드라이버, 손 게인 재적용

좌팔에서 검증된 요소는 그대로 옮긴다: 런 기반 계약 자동 로딩 · `step_dt_from_run` ·
정착(홈에 실제로 앉히기) · 실측 기준 가드 · **가드 정지 시 자동 홈 복귀**.
