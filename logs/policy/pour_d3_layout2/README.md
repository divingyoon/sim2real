# pour 정책 — d3_layout2 (2026-09-01 · 궤적 추출용)

| 항목 | 값 |
|---|---|
| 파일 | `nn/d3_layout2_ep1500.pth` (git 미추적) |
| md5 | `69288c3e` |
| 계약 | obs 55 / act 12 / LSTM 256 |
| 태스크 | `open-tesol_r_pour_sensor-lstm` |
| 출처 | `server:~/rl_ws/hdgp/log/rl_games/open-tesol/right/pour-sensor/d3_layout2/` |
| warm 뱅크 | `data/grasp_warm_s2r_d3_n2048_maxgrip.hdf5` (d3 파지 정책 산출물) |

## 왜 `best` 가 아니라 `last` 인가

`open-tesol_r_pour_sensor-lstm.pth`(rl_games 의 best)는 **ep173 · rew 29614** 다.
보상은 초기에 꺾였지만 **성공률은 단조 상승** 중이라 best 를 쓰면 훨씬 나쁜 정책을 쓴다.

| iter | 100 | 399 | 797 | 1196 | 1594 | 1992 |
|---|---|---|---|---|---|---|
| success | 0.124 | 0.412 | 0.535 | 0.592 | 0.629 | **0.654** |
| bead | 0.077 | 0.437 | 0.563 | 0.660 | 0.726 | **0.743** |

같은 시점 다른 런: `d3_layout1` 0.588 · `d3_maxgrip1-r2` 0.486. layout2 가 최고다.
(로컬 `b1_multicup1-r3` 의 0.162 는 **다른 뱅크·다른 라운드**다 — 혼동하지 말 것.)

## 이 정책으로 뽑은 궤적

`logs/shadow/pour_traj_d3/` — 자세한 판정은 `docs/POUR_REPLAY_D3_2026-09-01.md`.
