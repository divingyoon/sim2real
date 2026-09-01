# pour 궤적 추출 산출물 (08.31 · lstm_test4 ep1300 · worktree 9b43f40)

프로브: `hdgp_pour23/scripts/probes/probe_pour_traj_extract.py` · 성공 3/13 에피소드 중
peak 속도 최소 에피소드 선정.

## 파일

- `pour_traj_env3_s694_e1812.npz` — **1119스텝 · 60 Hz · 18.6 s** 성공 에피소드(fill 1.0).
  채널: `q_meas/qd_meas/q_target`(36관절: 팔14+손20+그리퍼2) · `palm_r`(r_hl_palm 7D)
  · `tcp_l`(l_hl_gripper_base 7D) · `cup_src/cup_recv`(7D) · `action`(15) · `fill/spill/done`.
  포즈는 env-local(pos3+quat wxyz). `meta_joint_names` 가 관절 순서의 진실원천.
- `pour_init.npz` — 에피소드 첫 프레임(파지 완료 warm 상태). 재생 전 도달해야 할 자세.

## ★재생 시 준수

- `q_target[0]` 은 실측값으로 **보정해 뒀다**(원본은 `q_target_raw_step0`) — 리셋
  텔레포트 프레임이라 그대로 재생하면 90.6 rad/s 도약. 보정 후 max 5.67 rad/s
  (좌팔 l_aj_4, 받는 컵 급이동 구간 step 120~135) · p99 2.49.
- **5.67 rad/s 는 실기 한계(2.0)의 2.8배** — bag 전역 감속 시 18.6 s → ~53 s 가 되고,
  붓기는 동역학 과제라 **느리게 재생하면 결과가 달라질 수 있다**(비드 낙하는 실시간).
  구간별 처리(급구간만 감속/재설계)는 P3 트랙 몫.
- 초기 상태 정의: **이미 두 컵을 파지한 상태**다(warm start). 파지 전 단계는
  grasp 정책들이 만든다.

## ★★P2 판정 — pour 초기 pose 는 두 grasp 정책의 goal 분포 **밖** (실측)

| | pour 초기(env-local) | grasp 정책 goal 범위 | 판정 |
|---|---|---|---|
| 받는 컵(좌) | (0.290, 0.029, 0.324) | v2 command x[0.325,0.425] y[0.174,0.314] z[0.397,0.497] | ❌ x −3.5 · **y −14.5** · z −7.3 cm |
| 붓는 컵(우) | (0.200, −0.097, 0.350) | m1 goal ≈ spawn(x≈0.36,y −0.16)+오프셋(0,0.05,0.08)±ADR(y0.12) | ❌ **x −11 cm** (y·z 는 안) |

→ **goal 지령만으로는 이송이 안 된다.** 권고: 리셋 궤적과 같은 방식의 **브리지 램프**
  (grasp goal 분포 안 도달점 → pour_init 관절자세, 파지 유지·충돌 검증·1회 계산 후 재생).
  좌표 전제: 두 env 모두 로봇 베이스 원점의 env-local — 동일 프레임으로 간주(육안 검증 권장).

파지 자세 불일치 리스크: 라이브 grasp 의 파지와 pour warm 파지가 다르면 같은 팔 궤적에도
컵 기울기가 달라진다 — 전환 전 파지 자세 대조 필요.
