# left v2E19_zfloor_ep1800 — S2R 배포 정책 (사용자 지정, 08.31)

- 원본: `hdgp/log/checkpoints_keep/v2E19_zfloor_ep1800.pth` · md5 `af7f6625…`
- 런: arm5080 `log/rl_games/open-grip/left/grasp-sensor-v2/v2E19_zfloor` (seed 43) — **params/ 회수 완료**
- 계약(실측): **obs 49 / action 7 / MLP · RNN 없음** · ep 1800 — v2H_wide/v2B17 과 전 층 동일, 드롭인
- ★★홈 판정(런 dump 근거): init joint_pos = **구 v1 홈**
  (-0.0136, -0.3757, -0.0010, +0.9336, -0.4655, +0.0003, -0.3306)
  — live 트리 v2_preset 의 J147(라운드17 교체홈)이 **아니다**. dump 전문에 J147 값 0건.
  ⇒ 리셋 궤적은 기존 `logs/shadow/reset_both/reset_left.npz` 그대로 유효.
  "zfloor"(A18/E19) 노선이 긁힘을 홈 교체가 아니라 z-floor 로 다룬 것으로 보임.
- 교체 계보: v2H_wide → v2B17_ep800_S997(스테이징만, 미배포) → **v2E19**(현행)
