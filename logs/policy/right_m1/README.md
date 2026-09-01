# right m1_final — S2R 배포 정책 (사용자 지정, 08.31)

- 원본: 옆 세션 scratchpad `ckpt/m1/m1_final.pth` (**/tmp 휘발 위치라 즉시 확보**, md5 42ea4d… 일치)
- 계약(실측): **obs 155 / action 21 / state 193(critic 전용) / LSTM 1024** · ep20000 · frame 328M
- action 21 = palm 6D **홈 기준 델타** + 손 시너지 15 (`grasp_s2r_env.py:507,568` — 관절 6개 아님)
- 태스크 `open-sens_r_grasp_s2r-lstm` (agnostic grasp_s2r · profile tesollo_right)
- ★원 런 커밋은 로컬에 없음(로컬 런은 ep9800까지) — params 는 같은 계보 `s2r_m1_k1warm_multi` dump
- ★fabric = `openarm_tesollo_sensor_right` + 전용 params (레거시 openarm_tesollo 는 팔베이스 +8mm 오차)
- 홈 = grasp_s2r rest (0.0380,0.4012,…) — **리셋 bag(bag_reset_right_both) 끝값과 일치**.
  구 grasp_sensor preset 의 q_home 미러(j2=0.6706)와는 **다른 IK 분기** — 혼용 금지.
- goal = spawn+(0,0.05,0.08)±ADR — pour 연계 판정에 사용

## ⛔ 08.31 재생 블로커 — m1 학습 소스 소실

현 live 소스에서 4개 변형(anchor 복원 · ADR 0/1.0 · deterministic/stochastic) 전부 실패.
**서명이 일관됨**: 파지·리프트는 성공(54~71mm ≥ 임계 40) · goal 을 35~60mm 못 미침
(tolerance 25mm) — 정책이 "자기가 아는 goal 위치"로 정확히 나르는데 그 자리가 현
소스의 goal 과 4~6cm 어긋난 형태. cfg 복원(run_cfg_restore)으로 안 잡히는 **코드 경로
드리프트**(goal 산출 or obs 조립)로 판단.

- m1 은 로컬 HEAD 0293644(08.28) + **08.29 dirty** 에서 학습 — 그 dirty 는 커밋도
  스냅샷도 없이 현재 상태로 진화함. arm4090 은 로컬의 rsync 미러(mtime 초 단위 일치)라
  동결본 아님. vision-3090 은 다른 계보.
- 참고: m1 학습 자체의 stay(60스텝 홀드)도 reward 0.0001 — 학습 성공 지표는
  `success_now`(0.53)·species 0.61~0.79 였다.
- 진행 옵션: ① 트리 소유 세션이 m1 재생을 자기 상태에서 재현 ② 현 소스로 학습된
  체크포인트(z 계열)로 교체 지정 ③ m1 재학습. 결정은 사용자 몫.
