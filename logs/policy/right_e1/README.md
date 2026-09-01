# 우팔 배포 정책 — E1 (2026-09-01 사용자 지정)

| 항목 | 값 |
|---|---|
| 파일 | `nn/e1_best.pth` (git 미추적) |
| md5 | `fc48c5cc` (`fc48c5cc483c867a7573dcb4fcd35391`) · 71,538,437 B |
| epoch | **14160** · rew 30062 — ★**학습 중 스냅샷**(목표 20000) |
| 계약 | obs 155 / act 21 / state 193 / LSTM 1024 |
| 태스크 | `open-sens_r_grasp_s2r-lstm` |
| 출처 | `server:~/rl_ws/hdgp/log/rl_games/open-sens/right/grasp-s2r/e1_perc/nn/open-sens_r_grasp_s2r-lstm.pth` |

★**학습이 이 파일을 계속 덮어쓴다.** 가져올 때는 서버에서 `cp` 로 스냅샷을 뜬 뒤
md5 를 대조해 받을 것 — 그냥 scp 하면 찢어진 파일을 받고
`PytorchStreamReader failed reading zip archive` 로 죽는다(09.01 실제 발생).

## d3 와 무엇이 다른가 (params dump 대조)

| 항목 | d3 | E1 |
|---|---|---|
| `replicate_physics` | true | **false** (다물체) |
| 자산 | `env.usd` | **`env_rigid.usd`** |
| `object_origin_offset_z` | 0.0773 | **0.10049** |
| `object_spawn_z` | 0.2823 | **0.30549** |
| `adr_goal_z_max` | 0.08 | 0.12 |
| `adr_goal_y_max` | 0.12 | 0.0 |
| obs 노이즈 / rigid-after-latch | 없음 | 추가 |
| `gain_dr_joints` | — | `arm` |

`goal_offset_xyz = (0, 0, 0.12)` — 컵을 스폰 자리에서 **수직 12 cm** 든다.
`enable_adr: false` · `use_real_gains: false`(KUKA 팔 게인) · 손 게인 kp5.0/kd2.0.

## sim 실측 (09.01 · 285 상태)

에피소드마다 **32/32 리프트** · 접촉 평균 4.38 · 5접촉 50.5% · 팁힘 4.3~6.8 N.
파지 품질은 d3 의 최대접촉 선별판(4.22 · 42.4%)보다 낫다 — 선별 없이.

★**d3 와 파지 종료 자세가 전혀 다르다** — `r_aj_6` 75.2° · `r_aj_4` 54.9° ·
palm 59.8 mm. 컵은 같은 자리로 가지만 팔이 여유자유도를 다르게 쓴다. 그래서
**pour 를 E1 뱅크로 다시 학습해야 한다** — 자세한 것은
`docs/HANDOFF_GRASP_TO_POUR_2026-09-01.md`.

뱅크: `hdgp/data/grasp_warm_s2r_e1_n256.hdf5` (검증용 285개).
본판은 E1 학습이 끝난 뒤 `--target_count 2048` 로 다시 모은다.
