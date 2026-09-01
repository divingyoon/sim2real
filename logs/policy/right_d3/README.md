# 우팔 배포 정책 — d3_ep20000 (2026-09-01 사용자 지정)

| 항목 | 값 |
|---|---|
| 파일 | `nn/d3_ep20000.pth` (git 미추적 — `logs/policy/**/*.pth`) |
| md5 | `485d8abf` (`485d8abf63106f488ba4a7d7a2797859`) |
| epoch | 20000 |
| 계약 | **obs 155 / act 21 / state 193 / LSTM 1024** |
| 태스크 | `open-sens_r_grasp_s2r` |
| 출처 | `hdgp/log/rl_games/open-sens/right/grasp-s2r/_DEPLOY/d3_ep20000.pth` (server 학습분) |
| 런 dump | `params/{agent,env}.yaml` (`_DEPLOY/d3_params` 에서 회수) |
| 보조 | `d3_ep19600.pth` md5 `24077a03` (같은 `_DEPLOY`) |

## b1_ep10800 (08.31 배포본) 과 무엇이 다른가

`params/env.yaml` 대조 (31줄 차이 중 과제에 영향 있는 것):

| 항목 | b1 | **d3** | 뜻 |
|---|---|---|---|
| goal 오프셋 | spawn + (0, **0.05**, **0.08**) | spawn + (0, **0.0**, **0.12**) | 옆으로 안 가고 **수직으로만 12 cm** 든다 |
| `lift_height_ref` | 0.10 | 0.06 | |
| `success_require_lifted` | false | **true** | 성공에 리프트가 **필수** |
| `grasp_weight` | 12.0 | 4.0 | 파지 보상 축소 |
| `finger_closure_weight` | 1.0 | 3.0 | 손가락 오므림 보상 확대 |
| `enable_adr` | true | **false** | **ADR 없음** |
| `adr_goal_sample` | true | false | goal 랜덤화 없음 |
| `respawn_clearance_m` | 0.05 | 0.12 | |

즉 d3 는 **liftonly 계열**(서버 런 이름도 `s2r_d3_liftonly_fresh_v2`)이고 **도메인
랜덤화가 꺼져 있다.** 실기 강건성은 b1 보다 낮을 수 있다 — 실기 투입 시 관찰 대상.

## 홈

이 태스크는 `palm_anchor_mode: spawn` 이라 **홈이 설정값에 없다.** 리셋마다 palm
앵커에서 유도된다. b1·d3 둘 다 같은 모드이므로 메커니즘은 같지만, **수치가 같다는
보장은 없다** — 현재 우팔 preset 궤적(`reset_right_v2.npz`)은 b1 의 step0 실측
자세로 만든 것이다.

⬜ **미확인**: d3 의 step0 자세를 sim 에서 한 번 뽑아 b1 과 대조할 것 (GPU 필요).
차이가 크면 좌팔과 같은 문제 — preset 궤적을 다시 구워야 한다.

## 재생 시

- **라이브 노드가 없다.** `grasp_inference.py` 의 `grasp_obs_builder.py:80` 은
  `ACTOR_OBS_DIM = 154` 로 이 정책(155)과 맞지 않는다. 좌팔과 같은
  Isaac-in-the-loop 형태(우팔판 `probe_*_shadow_record` + `--stream_udp --stream_meas`)
  를 만드는 것이 계획된 경로다.
- 게인: 이 정책은 **KUKA 게인**에서 학습됐다. 09.01 r2s 정합으로 얻은 실측 게인
  (`HDGP_S2R_REAL_GAINS=1`)에서 학습된 체크포인트는 아직 없다.
- fabric 자산은 `openarm_tesollo_sensor_right` 를 쓸 것 — 레거시 `openarm_tesollo`
  는 팔 베이스가 +8 mm 어긋난다.
