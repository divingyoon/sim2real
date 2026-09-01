# left v2H_wide_best — S2R 배포 정책 (사용자 지정, 08.31)

- 원본: `hdgp/log/checkpoints_keep/v2H_wide_best.pth` (런: vision-3090 `grasp-sensor-v2/v2H_wide`)
- 계약(실측): **obs 49 / action 7 / MLP · RNN 없음** · ep740 · rew 98.98
  obs 49 = jp9+jv9+obj_noisy3+cmd7+act7+gate1+tcp3+rot6d6+goal_minus_cup3+cup_upright1
- 태스크 `open-grip_l_grasp_sensor_v2` — v1 preset 상속(홈·기하 불변 → **리셋 bag 재사용 가능**,
  끝값 일치 실측). 액션 = FabricPalmAction 절대 palm 6D + 그리퍼 게이트(불변).
- ★원 런 디렉토리에 학습이 계속 쓰는 중이었다(ep2400 rew 72↓) — keep 본이 정본.
- ★로컬 live hdgp == vision-3090 학습 소스 (v2 모듈 + 상속 grasp_sensor, rsync -rcn 일치, 08.31)
- 롤아웃: `logs/shadow/sim_v2H_wide.npz` (1500스텝·리셋0회·L1 mean 14.3mm·L2 mean 24.6mm)
- 백: `logs/shadow/bag_v2H_wide` — **rate_scale 0.53** (요구 peak 3.73 rad/s > 한계 2.0), 56.6 s
- goal command 박스: x[0.325,0.425] y[0.174,0.314] z[0.397,0.497] (pour 연계 판정에 사용)
