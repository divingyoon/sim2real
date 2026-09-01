# real2sim 정합 프레임워크 — 인수인계

**무엇을 하는 도구인가.** 실기 로봇의 동특성을 측정해 **sim 액추에이터 파라미터를
맞춘다.** 목적은 정책이 sim 에서 실기와 같은 진동·지연·처짐을 겪게 하는 것이다.
09.01 우팔에 적용해 오버슈트 재현 오차를 **0.429 → 0.084** 로 줄였다.

## 한 줄 실행

```bash
cd /home/user/rl_ws/sim2real
python3 scripts/r2s_pipeline.py --stage all      # fit→apply→verify→report
python3 scripts/r2s_pipeline.py --stage fit      # 해석만(GPU 불필요, 초 단위)
python3 scripts/r2s_pipeline.py --stage report   # 이미 있는 결과로 판정만
```

## 파이프라인

```
collect(실기)  →  fit  →  apply  →  verify  →  report
   사용자        해석    코드수정   sim재생     판정
   ~30초         초      즉시      ~12분       즉시
```

| 단계 | 하는 일 | 산출 |
|---|---|---|
| **collect** | `robotctl r2s collect --repetitions 3` — 실기를 6초 흔든다 ×3 | `logs/r2s/*.npz` |
| **fit** | 관절별 2차계 `J q̈ + kd q̇ + kp q = kp q_des` 를 맞춰 (ωn, ζ) 를 얻고, **실기 kp** 로 sim kd 를 계산 | `pipeline_params.json` |
| **apply** | `robot_profiles.py` 의 `HDGP_S2R_REAL_GAINS=1` 분기를 갱신 | 코드 수정 |
| **verify** | 여진 재생(동특성) + 궤적 재생(정적) **둘 다** | `pipeline_{excite,preset}.npz` |
| **report** | 합격 판정 | 표준출력 |

## ★★반드시 지킬 것 (전부 값을 치르고 배웠다)

### 1. kp 는 실기 설정 파일이 진실이다 — 여진으로 바꾸지 마라

실기 kp/kd 는 `openarm_description/config/arm/v10/control_gains.yaml` 에 있고
하드웨어 인터페이스가 그대로 모터에 보낸다(`v10_simple_hardware.cpp:65-71,276`).
**여진에서 역산한 kp 를 넣었더니 정적 추종이 j2 RMSE 0.94° → 10.77° 로 무너졌다.**
여진은 한 자세에서 ±3~9° 만 흔드는데 실제 궤적은 ±50° 를 움직이고 관성이 자세에
따라 변하기 때문이다.

⇒ **kp 는 벤더값 고정, kd 만 `2ζ·√(kp·J_sim)` 으로 맞춘다.**
  (그렇게 계산한 j6 kd 0.580 이 벤더 설정 0.6 과 거의 같다 — 방법의 방증)

### 2. 동특성과 정적을 **둘 다** 봐라

위 실패는 여진 지표만 보면 못 잡는다. `verify` 가 두 재생을 다 도는 이유다.

### 3. `robotctl r2s fit` 의 kp 를 믿지 마라

그 모델에는 armature 가 없어 관성을 kp 로 흡수한다 — j6 을 **19배**(10 → 189.5)
부풀렸다. 이 파이프라인은 kp 를 밖에서 주입한다.

### 4. sim 의 `friction`(static)은 우리 조건에서 안 먹는다 — ★범위 정정

`1.019 → 0` 으로 바꿔도 재생 결과가 소수점까지 같았다(PhysX). 그래서 fit 을 마찰 없는
모델(`--no-friction`)로 돌려 그 효과를 ζ 에 흡수시켰고, `apply` 는 `friction=0.0` 을 쓴다.

**★2026-09-01 정정 — "sim 에 마찰을 넣을 수단이 없다"로 읽으면 틀린다.**
RL 트랙의 지적(`R2S_ISSUES_FROM_RL_2026-09-01.md` §1)을 받아 확인했다:

- `ActuatorBase` 에는 `friction`(static) · **`dynamic_friction`** · **`viscous_friction`**
  세 필드가 있다(`isaaclab/actuators/actuator_base.py:96-103`, Isaac Sim 5.1.0).
  이 파이프라인은 **static 하나만** 건드렸다 — 나머지 둘은 한 번도 설정한 적이 없다.
- static 이 안 먹은 것은 PhysX 의 결함이 아니라 **조건** 탓이다. 학습·probe 는 로봇
  중력을 끄고 돌아서 전달 하중이 작고, μs 를 키워도 저항 토크가 거의 안 생긴다.
  한 자세 소진폭 여진도 마찬가지다.

⇒ **쿨롱/점성 감쇠를 넣을 수단이 없는 게 아니라 쓰지 않았다.**

**단, 지금 그냥 켜면 이중 계산이다.** 현 `kd = 2ζ·√(kp·J_sim)` 의 ζ 에 마찰 몫이 이미
흡수돼 있다. 다음 fit 은 **마찰을 모델에 넣어** 돌리고 결과를 `kd`(순수 점성) ·
`dynamic_friction`(쿨롱) · `viscous_friction` 으로 **분리해서** 내야 한다. 그래야
소진폭·무접촉에서 식별한 등가 점성이 대진폭·접촉 영역으로 잘못 외삽되지 않는다.

**왜 중요한가**: 이 등가 치환이 깨지는 곳이 바로 학습이다. 실측 게인으로 돌린
`e2_s2r_full`(arm4090)은 파지·리프트는 0.96 인데 **이송만 실패**해 success 0.000 에서
멈췄다 — 접촉력 77.5 N(KUKA 갈래 10.8 N) · 컵 밀림 0.091 m 가 그대로 목표 오차가 됐다.

### 4-b. bag 을 읽을 때의 함정 둘 (RL 트랙 실측)

- **`effort` 는 토크가 아니라 전류다.** `delto_hardware` 가 `efforts_ = current_` 로
  채우고 `CURRENT_SCALE = 1.0` 이라 raw 카운트다. 토크로 읽으면 손가락에 172 N·m 가
  나온다. 손 토크 상한의 진실원천은 URDF 의 `<limit effort="7.5"/>` 다.
- **`/joint_states` 는 750 Hz 로 보이지만 실제 갱신은 380~535 Hz 다.** `right_preset_grav1`
  은 250,052 메시지 중 내용이 바뀐 것이 5.7% 뿐이고 t=52 s 이후 282 초는 비트 단위로
  동일한 latched 값이다. 전체 std 를 그냥 내면 **σ=0.0** 이 나온다. 그리고 한 팔이
  움직일 때 **반대 팔은 항상 latched** 라 좌우를 한 bag 에서 동시에 못 잰다.

  실측 노이즈(6 bag · 운동구간 하이패스 중앙값): 팔 position 양자화 3.815e-4 rad
  (=100/2¹⁸) · 정지 σ 2e-4 · **운동 σ 9e-4** / 팔 velocity 정지 4e-3 · **운동 4.5e-2**
  / 손 position 양자화 1.745e-3 rad. ★sim 기본 `obs_noise_qpos = 0.01` 은 실측의
  **10배**이고 `obs_noise_qvel = 0.05` 는 운동 구간과 거의 맞는다.

### 5. 튜닝 지표로 ptp(최대−최소)를 쓰지 마라

신호의 두 점만 보아 ζ 와 단순 대응하지 않는다 — kd 를 3배 키워도 12 % 밖에 안 변해
최적화가 서지 않는다. **lock-in 주파수 응답**을 쓴다(위상·지연 무관).

### 6. armature 는 넣지 않는다

sim 자산의 관성이 이미 대체로 옳다(sim 관성 + 실기 ωn 으로 역산한 kp 가 벤더값과
6~12 % 안). 한때 0.8~1.0 을 넣었던 것은 링크 관성을 point-mass(`Σm·d²`)로 어림해
**링크 자체 회전관성을 빼먹은** 탓이다(sim 실측은 그 어림의 4.5~5.6배).

### 7. Isaac probe 는 끝나도 GPU 를 안 놓는다

여러 번 돌리면 누적돼 OOM 이 난다(09.01 에 26 GB). 파이프라인이 매 단계 뒤
`_cleanup_gpu()` 로 정리한다. 수동 실행 시에도 같은 정리를 할 것.

### 8. 실기는 사용자 승인 후에만

`collect` 는 이 프레임워크에서 **유일하게 로봇이 움직이는 단계**다. 자동화하지 않았다.

## 09.01 우팔 결과 (기준선)

| 지표 | KUKA(기본값) | **정합 후** | 실기 |
|---|---|---|---|
| 오버슈트 재현 오차 (전체) | 0.429 | **0.084** | — |
| 〃 팔 j1-4 | 0.119 | **0.038** | — |
| 〃 손목 j5-7 | 0.843 | **0.147** | — |
| 정적 추종 RMSE | — | **1.33°** | 0.94° |

확정 파라미터:

```
kp  70 / 70 / 70 / 60 / 10 / 10 / 10          ← 실기 벤더값 그대로
kd  7.053 / 4.182 / 7.804 / 6.531 / 2.236 / 0.580 / 0.242
armature 0 · friction 0
```

측정된 실기 동특성:

```
ωn[Hz]  1.45 / 2.58 / 1.46 / 1.24 / 1.40 / 2.36 / 1.39
ζ       0.372 / 0.579 / 0.163 / 0.292 / 0.071 / 0.012 / 0.069
```

★손목(j5-7)만 부족감쇠이고 j6 은 2.1 Hz 에서 5.4배 공진한다. 그 관성의 ~90 %가
테솔로 손(1.763 kg)이다.

## ★손(Tesollo DG-5F) — 확정 파라미터

### 실기 (JTC PID)

| | p | i | d | 근거 |
|---|---|---|---|---|
| 벤더 기본 | 1.5 | 0 | 0 | `dg5f_driver/config/dg5f_right_controller.yaml`, 20관절 동일 |
| **★확정** | **4.5** | **0** | **0** | 아래 |

```bash
python3 scripts/apply_hand_gains.py --execute      # bringup 이후 매번
python3 scripts/apply_hand_gains.py --restore --execute   # 벤더 기본으로
```

⚠**bringup 을 다시 하면 벤더 기본(1.5)으로 돌아간다.** vendor yaml 은 고치지 않았다.

**확정 근거 (전부 다관절 완전 주먹, 09.01 실측)**

| p | 최대 σ | 도달률 | 정상오차 | 판정 |
|---|---|---|---|---|
| 1.5 (벤더) | 0.096° | 82 %\* | — | 느리다 |
| 3.0 | 0.100° | 91 %\* | — | |
| **4.5** | **0.095°** | **98~101 %** | **0.39°** | ★채택 |
| 6.0 | **0.160°** | — | — | ⚠진동 시작 |
| 12.0 | — | — | — | ⚠육안으로 확실한 진동 |

\*82/91 %는 3 s 급속 지령의 단일 관절 값. 실사용(4 s 주먹)에서 p=4.5 는 98~101 %.
기저 σ(손이 가만히 있을 때 엔코더 노이즈)는 **0.09°** 다 — 0.095 는 그 수준이다.

`d` 는 0 / 0.01 / 0.03 / 0.05 에서 σ 가 0.095~0.099° 로 **진동에 영향이 없고**,
단일 관절에서 d>0.02 는 오버슈트(+60.8° vs 목표 +57.3°)를 만든다 ⇒ **d=0 유지**.

### sim (`grasp_s2r/robot_profiles.py`)

**`kp 5.0 · kd 2.0` 을 그대로 둔다.** 같은 주먹 램프에서:

| | 정상상태 오차 |
|---|---|
| 실기 (p=4.5) | 평균 0.39° · 최대 1.50° |
| sim (kp 5.0) | 평균 0.05° · 최대 0.60° |

sim 이 8배 정확하지만 **차이 0.34° 는 손가락 끝에서 1 mm 이하**라 파지에 영향이 없다.
맞추려고 sim kp 를 낮추면 **파지력이 그만큼 준다**(kp 5.0 은 파지력 기준으로 정해진
값 — grasp_v1 kd 스윕). 실기 p 를 더 올리는 쪽은 **p=6 진동**이 막는다.
⇒ 실기를 진동 한계 안에서 최대(4.5)로 올리고 sim 은 유지하는 것이 최선이다.

### ★★게인은 반드시 **다관절 동시**로 검증할 것

관절 하나로 시험하면 σ 가 0 이라 진동을 놓친다 — 09.01 에 단일 관절로 p=12 를
"도달률 97 %" 라고 정해 20관절에 걸었다가 전 손가락이 진동했다. 진동은 **손가락 간
커플링**에서 나온다. `probe_hand_multi_gain.py` 가 주먹 자세로 동시에 움직여 잰다.

★튜닝 지표에 **진동(정착 후 위치 σ)** 을 넣을 것. 도달률만 최적화하면 "빠를수록 좋다"가
되어 **진동을 유발하는 방향으로 최적화**하게 된다.

### 게인이 세 층에 있다

| 층 | 위치 | 성격 |
|---|---|---|
| ① 모터 펌웨어 | 손 내부 (읽는 서비스 없음) | `SetJointGainP/D/I` 로 **쓸 수만** 있다. `dg_sdk_ros2_bridge` 필요 |
| ② ros2_control JTC PID | `dg5f_right_controller.yaml` | position 오차 → **effort**. 위 확정값이 여기다 |
| ③ sim | `robot_profiles.py` | `kp 5.0 · kd 2.0` |

팔은 `control_gains.yaml` 의 kp/kd 가 **MIT 모드로 모터에 직접** 간다. 층을 섞지 말 것.

### 정책이 구동하는 관절은 13개

```
thumb {_3,_4}   index {_2,_3,_4}   middle {_2,_3,_4}   ring {_2,_3,_4}   pinky {_3,_4}
```

`_1`(외전) 5개 + `thumb_2` + `pinky_2` 는 정책이 안 쓴다(`hand_finger_channels`).
튜닝·측정에서 빼면 손가락끼리 충돌하는 축이 사라진다.

### 관절 한계는 **관절별로** 읽을 것

| 손가락 | `_1` | `_2` | `_3`·`_4` |
|---|---|---|---|
| thumb | [-0.38,+0.89] | **[-3.14,+0.00]** | ±1.571 |
| index | [-0.42,+0.61] | **[+0.00,+2.01]** | ±1.571 |
| middle | [-0.61,+0.61] | [+0.00,+1.95] | ±1.571 |
| ring | [-0.61,+0.42] | [+0.00,+1.90] | ±1.571 |
| pinky | [-0.02,+1.05] | **[-0.42,+0.61]** | ±1.571 |

접미사로 뭉뚱그려 엄지 `_2`([-3.14,0])를 전 손가락에 적용하면, 실하한이 0 인 관절은
**아예 움직이지 않는다**. 온도는 `/dg5f_right/dynamic_joint_states` 의 `temperature`,
발열 상한 55℃ 로 잡을 것.

⚠**가동범위는 게인을 세운 뒤에 재야 한다.** p=1.5 로 재면 7관절이 전부 +54~56° 에서
멈추는데, 그것은 관절 한계가 아니라 제어기가 못 따라간 것이다(같은 관절을 5 s
정착시키면 +63.8° 까지 간다).

## 다른 팔/로봇에 쓰려면

`r2s_pipeline.py` 상단 **§설정 상수**만 바꾼다:

| 상수 | 뜻 |
|---|---|
| `ARM` | 관절 이름 |
| `EXCITE_RUNS` · `EXCITE_HOLDOUT` | 여진 기록(2 fit + 1 holdout) |
| `PRESET_NPZ` | 정적 검증용 궤적 |
| `VENDOR_GAINS` | ★실기 kp/kd 설정 파일 — **사본이 여럿이면 전부 같은지 확인** |
| `PROFILE_PY` | sim 액추에이터 정의 |
| `J_SIM_DEFAULT` | sim 관절 관성. `verify` 결과에서 역산해 갱신 |
| `WRIST_SCALE` | 손목 kd 배율(스윕 산출) |

## 도구

| 파일 | 역할 |
|---|---|
| `scripts/r2s_pipeline.py` | 오케스트레이터. 위 5단계 |
| `scripts/fit_excite_model.py` | 2차 모델 fit(±마찰). `--kp` 로 kp 주입 |
| `scripts/probe_excite_sim_replay.py` | 여진을 **물리 dt 격자**로 sim 재생. `--num-envs N --kd-scale lo,hi` 로 6분에 N개 조합 스윕 |
| `scripts/probe_s2r_gain_replay.py` | 궤적 재생(정적 추종) |
| `scripts/probe_excite_clearance.py` | 여진 전 안전 판정(자세·진폭) |
| `scripts/gravity_comp_node.py` | 실기 연속 중력보상(collect 중 필수) |
| **손** | |
| `scripts/apply_hand_gains.py` | 확정 게인(p=4.5·d=0) 적용 — bringup 이후 매번 |
| `scripts/probe_hand_multi_gain.py` | ★**다관절 동시** 게인 시험(진동 σ 측정 + 응답 기록) |
| `scripts/probe_hand_gain_sweep.py` | 단일 관절 게인 스윕 — 진동은 못 잡으니 보조로만 |
| `scripts/probe_hand_sim_replay.py` | 실기 손 응답을 sim 에서 재생해 대조 |
| `scripts/probe_hand_range.py` | 가동범위 측정(관절별 한계·온도 상한·휴식) |

★`probe_s2r_gain_replay.py` 로 **여진을** 재생하면 안 된다 — 지령당 `env.step_dt`
(16.7 ms)를 쓰는데 여진 지령은 10 ms 간격이라 시간축이 1.67배 늘어난다.

## 남은 과제

- **j6** — 정적에서 혼자 RMSE 2.78°·max 16.57° 로 벗어난다. `kp=10·kd=0.6·ωn=2.36 Hz`
  를 동시에 만족하는 해가 없다. 단일관절 여진이 있어야 커플링/구조유연성이 갈린다.
- **손목 3관절 커플링** — 관절별 독립 스윕은 한계(예상 0.486 → 실제 0.782).
  3차원 동시 최적화(라틴 하이퍼큐브 또는 좌표하강)가 필요하다.
- **`r2s identify`** 로 kp 독립 확인 — 여러 자세가 필요해 **테이블 없는 상태**여야 한다.
- **학습 반영** — 환경변수를 켜고 학습한다. ★09.01 현재 **배포 정책들은 이 게인에
  맞춰진 것이 아니다**(재학습 진행 중). 재생·평가 전에 그 런의 `params/` dump 로
  어느 게인에서 학습됐는지 확인할 것. **기본값(KUKA)은 바꾸지 않는다** —
  배포 정책 b1_ep10800 이 그 위에서 학습됐고, 기본값을 바꾸면 그 체크포인트의 재생이
  조용히 달라진다.

  ```bash
  cd /home/user/rl_ws/IsaacLab
  HDGP_S2R_REAL_GAINS=1 ./isaaclab.sh -p \
      /home/user/rl_ws/hdgp/scripts/reinforcement_learning/rl_games/train.py \
      --task open-sens_r_grasp_s2r-lstm --headless --num_envs 1024
  ```

  태스크 id 는 `open-{tag}_grasp_s2r{suffix}` 이고 suffix 는
  ``/`-play`/`-lstm`/`-play-lstm` 이다(`grasp_s2r/config/__init__.py:72-87`).
  우팔 tag 는 `sens_r`. **배포 정책이 LSTM 이면 `-lstm` 으로 학습해야 한다.**

  ★게인이 바뀌면 동특성이 바뀌므로 **FRESH 학습이 기본**이다 — LSTM 정책은
  체크포인트에서 재개하면 기존 행동을 답습한다([[fresh-vs-warmstart-lstm-rule]]).

  ★ADR `robot_joint_stiffness_and_damping` 의 중심을 실측값으로 옮기면 sim2real
  강건성이 오른다. 범위는 `probe_excite_sim_replay.py --kd-scale` 스윕이 준 감도로
  정한다(우팔 손목은 배율 0.7~2.0 에서 주파수응답이 0.666→0.498 로 완만하게 변했다).

## 상세 기록

`docs/EXCITE_RESULT_RIGHT_2026-09-01.md` — 측정·분석 전 과정과 **철회한 주장들**.
`docs/EXCITE_RIGHT_2026-09-01.md` — 실기 여진 절차·안전.
