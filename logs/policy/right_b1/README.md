# right b1_ep10800 — S2R 배포 정책 (사용자 지정, 08.31)

- 원본: 옆 세션 scratchpad `ckpt/b1/b1_ep10800.pth` · md5 `fa93a20d…` (제시값 일치)
- **런 특정**: arm4090 `log/rl_games/open-sens/right/grasp-s2r/s2r_b1_anylink/nn/last_…ep_10800_rew_13429.149.pth` (md5 동일)
- 계약(실측): **obs 155 / action 21(palm6+손15) / LSTM hidden 1024** · ep 10800 — m1 과 동일 계약, 프로필 드롭인
- 성능(사용자 보고): ★8종 전수 파지(인벨롭)+리프트 (512env·1,024ep 프로브)
- **params/ 확보됨**(agent+env.yaml, arm4090 회수) — m1 재생불가 사태의 재발 방지 요건 충족
  - `palm_anchor_mode: spawn` (m1 과 동일 — 재생 시 restore_run_cfg 필수)
  - `hand_layout: coupled3` (per_finger 아님 — 액션 21 유지)
  - 우팔 홈 dump = (0.0380, 0.4012, 0.6015, 0.9643, 0.0294, 0.7060, 0.4213) + 손 init(엄지 −1.57/−0.5)
    → 리셋 궤적 `reset_right_v2.npz` 의 목표와 일치
- m1_final 은 `right_m1/` 에 배포 후보로 잔류 (⛔재생 블로커 이력은 그 README 참조)
