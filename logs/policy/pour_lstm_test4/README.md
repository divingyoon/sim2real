# pour lstm_test4 ep_1300 — 궤적 추출 전용 (라이브 배포 안 함)

- 원본: `vision-3090:.../open-tesol/both/pour-sensor/lstm_test4/` (런 시점 2026-07-22 23:36)
- 파일: `nn/last_open-tesol_b_pour_sensor-lstm_ep_1300_rew__33892.785_.pth`
  (md5 `a6d15868…74ad` 대조 완료. best 자동저장은 초기(23:47) 산출이라 쓰지 않음)
- 계약(실측): **obs 55 / action 15 / state 144 / LSTM(512)** — action 15 = palm6 + nullspace1 + hand5(lerp) + left_tcp3
- ★현재 hdgp 소스는 `NUM_ACTIONS=6` 으로 갈렸다 — **이 체크포인트는 현 소스로 재생 불가**.
  재생은 런 시점 커밋 **`9b43f40`** worktree = `/home/user/rl_ws/hdgp_pour23` 에서.
- ★warm 캐시: env.yaml 의 `data/grasp_warm_tesollo.hdf5` 는 08.17 DG-5FS 자산 교체로 **격리됨**.
  이 런은 구 자산(DG-5F, `openarm_tesollo_sensor_rl` — 아직 존재) 시절이라 아카이브 캐시가 정합:
  `/home/user/rl_ws/archive/hdf5_2026-08-17_pre_dg5fs/_quarantined_from_hdgp_data/grasp_warm_tesollo.hdf5`
  재생 시 `warm_state_paths` 를 이 경로로 **오버라이드**(격리 해제 금지).
- 성공 판정: `_bead_in_target_fraction ≥ fill_ratio` ∧ `_spill_ratio ≤ success_spill_max` (`pour_right_env.py:2705`)
- 용도: 성공 에피소드의 좌/우 palm_ee + 전 관절 시계열 추출 → grasp S2R 뒤 궤적 재생. P3 이후는 별도 트랙.
