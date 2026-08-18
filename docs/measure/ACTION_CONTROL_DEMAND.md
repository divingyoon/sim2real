# 정책 액션 ↔ 로봇 제어 정합 측정

**작성 2026-08-18 · 대상 `tesollo/right·left grasp_v1`**

## 왜 이 문서가 있는가

배포 시도에서 **"정책의 액션을 로봇 컨트롤러가 이해하지 못하고 움직임을 따라가지 못한다"**
는 증상이 반복됐다. 원인을 추정으로 남기지 않고 수치로 확정하기 위한 측정 기록이다.

격차는 세 층으로 나뉘고, **서로 독립적**이다 — 하나를 고쳐도 나머지가 남는다.

| 층 | 내용 | 상태 |
|---|---|---|
| ① 계약 | obs 114→154, action 11→21, 래칫, palm delta, 고정 홈 | ✅ 해소(코드 반영) |
| ② 자산 | 배포 Fabrics 가 구 URDF → palm **6.5cm** 어긋남 | ✅ 해소(프로필+검증) |
| ③ 제어 동특성 | sim 팔이 실기보다 훨씬 잘 추종 | ⬜ **측정 완료 · 대응 미정** |

---

## s2r 1차 대상은 `grasp_sensor` 다 (08.18 결정)

실기 구성이 **우 DG-5F + 좌 2지 그리퍼**(`openarm_tesollo_sensor_rl`)라, 라이브가 가능한
구성은 `grasp_sensor` 뿐이다. 아래 격차 분석은 **두 구성에 모두 적용**되지만 ②만 다르다:

| | `grasp_v1` (bi_s) | `grasp_sensor` |
|---|---|---|
| 계약 ① | obs 154D / action 21D / 고정 홈 / 축별 delta | **동일** |
| 자산 ② | Fabrics `openarm_tesollo_bi_s` | Fabrics `openarm_tesollo` |
| 배포 구 기본값과의 관계 | **6.5cm 어긋났음** | **원래 맞았음** |
| Fabrics↔물리 palm 차 | 8 mm | 2.1 mm |
| q_home | `[0.308, 0.579, 0.097, 0.581, 0.268, 0.528, 0.579]` | `[0.043, 0.671, 0.096, 0.734, 0.375, 0.568, 0.671]` |
| 실기 라이브 | 좌 Tesollo 미장착 → 불가 | **가능** |
| ③ 제어 동특성 | 아래 표 그대로 (같은 팔·같은 게인) | **동일** |

두 자산의 palm 은 6.5cm 다르므로 **프로필 혼용은 금지**다. 구성별 프로필이 자산·계약·
q_home 을 각각 확정하고, 매니페스트 대조와 홈 IK 검증이 혼용을 기동 시점에 막는다.

③ 대역폭 격차는 **팔이 같으므로 두 구성에 동일하게 적용된다** — 아래 측정은 그대로 유효하다.

---

## ① 계약 격차 (해소됨)

| 항목 | sim(학습) | 배포(구) |
|---|---|---|
| actor obs | **154D** | 114D |
| action | **21D** (palm 6 + 손가락 5×3) | 11D (palm 6 + 손가락 5) |
| 손가락 폐쇄 | 절대 목표 + 변화율 상한(**감소 가능**) | 래칫(단조 증가) |
| palm delta | 축별 **(0.15, 0.35, 0.15)** | 스칼라 0.15 |
| 리셋 | 고정 홈 (0.28, ∓0.38, 0.42) | 컵 **참값** pregrasp |

**palm delta 가 특히 치명적이었다.** 홈 y(∓0.38)에서 스폰 박스의 가장 먼 컵 y(∓0.10)까지
필요한 이동량은 **0.28 m** 인데, 스칼라 0.15 로는 **구조적으로 도달 불가**였다.
회귀 테스트로 고정: `test_grasp_policy_core.py::test_palm_delta_y_reaches_far_cup`.

## ② 자산 격차 (해소됨)

배포는 `OpenArmTeoslloPoseFabric` 을 `robot_dir_name` 없이 생성해 기본값 `openarm_tesollo`
를 썼고, sim 은 08.05 부터 `openarm_tesollo_bi_s` 를 쓴다.

| URDF | palm_link FK z (q=0) |
|---|---|
| `openarm_tesollo` (배포가 쓰던 것) | 0.12863 m |
| `openarm_tesollo_bi_s` (sim 학습) | 0.19350 m |
| **차이** | **0.0649 m** |

palm 은 obs 154D 중 **36차원**(`palm_to_cup`·`fingertip_pos_rel_palm`·`cup_to_fingertip`)의
기준이자 **Fabrics IK 의 목표**다. 관측과 지령이 동시에 틀린다.
08.03 의 "Fabrics palm vs sim FK 0.0mm 일치" 검증은 그 시점엔 옳았고, `bi_s` 자산이 신설되며
sim 만 이동한 것이다 — **자산 신원을 검증하는 곳이 없어서** 아무도 몰랐다.

방어선: 구성 프로필이 자산 매니페스트(`control_joint_order`)와 대조하고, 홈 IK 결과를 sim
preset 유도값과 0.05 rad 이내로 검증한다. 구 자산으로 되돌리면 테스트가 죽는다
(`test_robot_profile.py::test_fabrics_asset_matches_sim`).

---

## ③ 제어 동특성 격차 — ★측정 결과

sim 은 팔을 `set_joint_position_target` + **stiffness 400 / damping 80** 으로 굴린다.
실기는 JTC(position) → CAN MIT 펌웨어 PD 이고, 실측 식별 게인은 훨씬 약하다.
같은 관성에 대해 2차계 특성(고유진동수 f_n, 감쇠비 ζ)을 비교한다.

| 관절 | I[kg·m²] | 실기 kp/kd | 실기 f_n[Hz] | 실기 ζ | sim f_n[Hz] | sim ζ | 대역폭비 |
|---|---|---|---|---|---|---|---|
| `r_aj_1` | 0.4803 | 67.6 / 6.38 | 1.89 | 0.56 | 4.59 | 2.89 | **2.43배** |
| `r_aj_2` | 0.5350 | 67.6 / 6.38 | 1.79 | 0.53 | 4.35 | 2.73 | **2.43배** |
| `r_aj_3` | 0.1054 | 67.6 / 6.38 | 4.03 | 1.19 | 9.81 | 6.16 | **2.43배** |
| `r_aj_4` | 0.2012 | 67.0 / 5.64 | 2.90 | 0.77 | 7.10 | 4.46 | **2.44배** |
| `r_aj_5` | 0.0157 | 12.0 / 2.15 | 4.40 | 2.48 | 25.37 | 15.94 | **5.77배** |
| `r_aj_6` | 0.0201 | 12.0 / 2.15 | 3.90 | 2.19 | 22.48 | 14.12 | **5.77배** |
| `r_aj_7` | 0.0287 | 12.0 / 2.15 | 3.26 | 1.83 | 18.80 | 11.81 | **5.77배** |

대역폭비 평균 **3.86배**, 최대 **5.77배**. 실기 ζ 0.53~2.48 vs sim 2.73~15.94.

※ sim 액추에이터 = stiffness 400 / damping 80 (grasp_{side}_env_cfg.py ImplicitActuatorCfg)

**출처(전부 실측/자산 — 지어낸 값 없음)**
- 게인: `hdgp/log/logs/r2s_autotune/results/right_arm_best_calibration.json`
  (dataset `/home/user/r2s/right_track.hdf5`)
- 관성: `urdf/generated/rl/openarm_tesollo_bi_s_rl.urdf` 에서 계산 — 홈 자세, 대각 근사
  (`scripts/arm_inertia.py`). 관절 간 결합·코리올리·중력은 무시하므로 **추종 대역폭 비교
  용도**이지 절대 토크 예측용이 아니다.
- sim: `grasp_{side}_env_cfg.py` `ImplicitActuatorCfg`

**재현**: `python3 scripts/report_arm_bandwidth.py [--md]`

### 해석

1. **sim 팔이 실기보다 평균 3.9배(손목 5.8배) 빠르게 추종한다.**
   정책은 "지령하면 거의 즉시 그 자리에 가는 팔"에서 학습됐다.
2. **감쇠 성격이 반대다.** sim 은 전 관절 과감쇠(ζ 2.7~15.9)라 오버슛이 없다. 실기는
   어깨·팔꿈치가 **부족감쇠**(ζ 0.53~0.77)라 오버슛하고 링잉한다.
3. 즉 정책은 **droop 도 지연도 진동도 겪어본 적이 없다.** `robot_control` 문서가 이미
   경고한 그대로다 — *"이걸 거꾸로 하면 팔보다 훨씬 잘 추종하는 sim 이 나오고, droop 과
   싸우는 법을 영영 못 배우는 정책이 나온다."*
   (`robot_control/docs/superpowers/plans/2026-07-26-isaac-sim-parameter-transfer.md:262-268`)

### 아직 측정하지 않은 것 — 요구(demand) 쪽

위 표는 **능력(capability)** 이다. 판정하려면 정책이 실제로 요구하는 관절 속도가 필요하다.

```bash
# 요구 프로파일 (sim 처럼 즉시 추종 → 정책 순수 지령)
IsaacLab/isaaclab.sh -p scripts/grasp_loop_sim.py --robot tesollo_bi_s__right \
    --agent <params/agent.yaml> --ckpt <ckpt.pth> \
    --arm-model rate --max-vel 99 --demand-csv logs/measure/demand_right.csv

# 능력–요구 스윕 (실측 게인 PD)
for v in 0.1 0.3 0.5 1.0 2.0; do
  IsaacLab/isaaclab.sh -p scripts/grasp_loop_sim.py --robot tesollo_bi_s__right \
      --agent ... --ckpt ... --arm-model pd --max-vel $v
done
```

판정 기준은 하네스가 출력하는 **palm→cup 최소거리**다. `--max-vel` 을 낮추면 접근이
실패하는 것이 재현되어야 그 다음 실기 실험(Phase 3)의 해석이 가능하다.

> 학습(`lstm_test2`) 완료 후 최종 체크포인트로 돌릴 것. 중간 ckpt 로도 경향은 보인다.

### 대응 선택지 (측정 후 결정)

| 안 | 내용 | 비용/위험 |
|---|---|---|
| A | 브리지 `--max-vel` 상향 + 추종오차 감시 | 싸다. 단 실기 대역폭 자체는 못 올린다 |
| B | 실기 펌웨어 kp/kd 상향 | 하드웨어 위험(발진·과열). j7 은 effort 7N·m 한계 |
| C | 캘리브 주입 재학습 (`real2sim_actuator_cfg` 경로 존재) | 가장 사실적. 학습 시간 소요 |
| D | 정책 감속(`--episode-steps` 2배) | 즉시 가능. 근본 해결 아님 |

**결정은 요구 프로파일이 나온 뒤에 한다.** 요구 ≤ 능력이면 ③ 은 실무상 문제가 아니다.
