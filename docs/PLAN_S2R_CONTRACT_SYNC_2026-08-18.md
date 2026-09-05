# grasp_v1 sim2real 계약 동기화 계획서

**작성 2026-08-18 · 대상 저장소 `/home/user/rl_ws/sim2real` · 참조 sim `/home/user/rl_ws/hdgp`**

---

## 0. 배경과 목표

배포 스택은 **2026-08-03 시점의 계약(obs 114D / action 11D)** 에 고정되어 있고,
08.16~18 sim 개편이 하나도 반영되어 있지 않다. 지금 배포 노드를 실행하면
정책 로드 단계에서 shape mismatch 로 즉시 실패한다.

목표는 **배포 코드를 현재 sim 계약에 맞추고, 그 일치를 테스트로 고정하는 것**이다.
실기 하드웨어 작업(dg5f 손 전원, 캘리브)은 이 계획서 범위 밖이다.

### 격차 요약 (전부 코드에서 확인된 값)

| 항목 | 배포 현재 | sim 현재 | 근거 |
|---|---|---|---|
| actor obs | **114D** | **154D** | `grasp_obs_builder.py:34-37` / `grasp_right_constants.py:119-121` |
| 접촉 obs | binary 5D | **tip-local 3축 힘 15D** | `grasp_right_env.py:1192-1200` |
| 관절오차 obs | 없음 | **20D** | `grasp_right_env.py:1210-1213` |
| last_actions | 11 | **21** | `grasp_right_constants.py:107` |
| action | 11D (palm6+손가락5) | **21D** (palm6+손가락15) | 위와 동일 |
| 손가락 채널 | `np.repeat(cmd,4)` | `[ch0,ch1,ch2,ch2]` | `grasp_right_env.py:1032-1034` |
| 4지 공통닫힘 | **없음** | **있음**(채널별 평균) | `grasp_right_env.py:893-896` |
| 폐쇄 적분 | **래칫**(단조증가) | 절대목표 + 변화율 상한 | `grasp_right_env.py:1036-1039` |
| palm delta | 스칼라 ±0.15 | **축별 (0.15, 0.35, 0.15)** | `grasp_inference.py:129` / `grasp_right_env_cfg.py:301` |
| 리셋 | **컵 참값 pregrasp** | **고정 홈** | `grasp_inference.py:471` / `grasp_right_env_cfg.py:249-250` |

> ⚠ **가장 치명적인 둘**
> - `palm delta y = ±0.15` 로는 홈(y=−0.38)에서 컵(y=−0.10)까지 **0.28m 를 구조적으로 도달 불가**.
> - 배포는 정확히 sim 이 폐기한 방식(컵 **참값** pregrasp)을 쓴다. perception 오차가
>   초기 자세에 그대로 실린다.

### 이미 정상인 것 (건드리지 말 것)

- **extrinsics 는 PLACEHOLDER 가 아니다.** `config/global_camera_extrinsics.yaml` 에
  2026-08-02 ChArUco 실캘리브 값(재투영 0.274px)이 들어 있다.
  단 **목 자세 pan=−90 / tilt=280 에서만 유효** → 배포 시 `head_position_hold_node.py` 로 재현 필요.
- `/cup_pose` 배선(FoundationPose → `cup_pose_relay.py` → `grasp_inference.py:305`)은 정상.
- staleness 게이트, `/grasp/start|stop|reset` 서비스, `isaacsim_cmd_to_jtc.py` 브리지 — 런북과 코드 일치.
- `LiftLatch` / `scale_palm_delta` / `compute_joint7_lift_wait_target` — parity 테스트로 검증됨.

---

## 1. 작업 원칙 (에이전트가 반드시 지킬 것)

1. **sim 을 고치지 말 것.** 이 계획서는 `sim2real` 만 수정한다.
   `hdgp` 는 **읽기 전용 참조**다. 학습이 진행 중이라 sim 변경은 학습을 무효화한다.
2. **sim 값을 하드코딩으로 베끼지 말고, 가능하면 `hdgp` 상수를 import 하라.**
   `test_grasp_decoder_parity.py:1-20` 이 이미 그 방식을 쓴다(소스 없으면 skip).
   import 불가한 값만 상수로 복제하고, **출처를 `파일:라인` 주석으로 남겨라.**
3. **각 단계마다 테스트를 먼저 고치고(RED) 구현을 맞춰라(GREEN).**
   기존 테스트가 구 계약을 하드 assert 하므로 반드시 같이 갱신된다.
4. **차원 검증은 예외로 던져라.** 조용히 zeros 로 채우지 말 것 —
   과거 "손 obs zeros(20) 로 팔 후퇴" 사고와 동형이다.
5. 커밋은 단계별로 나눈다. 한 커밋에 obs·action·리셋을 섞지 말 것.

---

## 2. Step 1 — obs 빌더 154D 재작성

### 대상
- `scripts/grasp_obs_builder.py` (124줄)
- `scripts/test_grasp_obs_builder.py` (119줄) — `test_dim_is_114_and_base_106` 이 114 하드 assert

### sim 의 정답 레이아웃 (`grasp_right_env.py:1222-1235`)

```
arm_joint_pos            7      /joint_states  position (r_aj_1..7)
arm_joint_vel            7      /joint_states  velocity
finger_joint_pos        20      /dg5f_right/joint_states position (r_hj_*)
finger_joint_vel        20      /dg5f_right/joint_states velocity
palm_center_pos          3      FK
fingertip_pos_rel_palm  15      (tip - palm)
palm_to_cup              3      (cup - palm)
cup_to_fingertip        15      (tip - cup)
tip_force_local         15  ★신규
joint_pos_err           20  ★신규
last_actions            21  ★11 → 21
object_onehot            8
                       ---
                       154
```

> 소스의 인라인 주석 `# 8. last actions (11D)`, `last_actions, # 13`, `# 146D` 는
> **전부 stale 이다.** 값은 `grasp_right_constants.py:107,119-121` 이 정답
> (`NUM_ACTIONS=21`, `NUM_OBSERVATIONS=154`). 주석을 믿지 말 것.

### 2-A. `tip_force_local` 15D

sim 계산 (`grasp_right_env.py:1192-1200`):

```python
_tip_f_local = quat_apply(quat_conjugate(tip_quat_w), contact_force_xyz_raw)   # world → tip-local
tip_force_local = (_tip_f_local / CONTACT_FORCE_MAX).clamp(-1.0, 1.0)          # CONTACT_FORCE_MAX = 10.0
```

**프레임이 tip-local 인 것이 핵심이다.** sim 이 일부러 tip-local 로 맞춰 놓은 이유는
실물 F/T 가 센서 로컬 출력이라 **실기 값이 변환 없이 그대로 들어가게** 하기 위해서다
(`grasp_right_env.py:1186-1191` 주석). 배포에서 world 변환을 넣으면 안 된다.

배포 입력: `/dg5f_right/fingertip_{1..5}_broadcaster/wrench` 의 **force 3축만** (torque 미사용).

```python
tip_force_local = np.clip(force_xyz_5x3 / CONTACT_FORCE_MAX, -1.0, 1.0).reshape(-1)  # (15,)
```

### 2-B. `joint_pos_err` 20D

sim 계산 (`grasp_right_env.py:1210-1213`):

```python
joint_pos_err = ((hand_joint_targets - hand_joint_pos) / JOINT_POS_ERR_MAX).clamp(-1.0, 1.0)
# JOINT_POS_ERR_MAX = 1.2 (rad),  grasp_right_constants.py:159
```

**부호를 보존한다** — 어느 방향으로 막혔는지가 정보다. `abs()` 금지.

배포에서 계산 가능하다:
- `hand_joint_targets` = **우리가 방금 보낸 20D 지령** (`GraspFingerController.step` 의 반환값)
- `hand_joint_pos` = `/dg5f_right/joint_states` 실측

→ 빌더 시그니처에 `hand_joint_targets` 를 추가하고, 호출자(`grasp_inference.py`)가
직전 스텝에 전송한 지령을 넘기도록 배선한다. **지령을 보관하는 상태가 없으면 새로 만들어라.**

### 2-C. 상수

`hdgp` 에서 import 하되 실패 시 명시적으로 실패시켜라(조용한 fallback 금지):

```python
CONTACT_FORCE_MAX = 10.0   # grasp_right_constants.py:145
JOINT_POS_ERR_MAX = 1.2    # grasp_right_constants.py:159
NUM_ACTIONS       = 21     # grasp_right_constants.py:107
ACTOR_OBS_DIM     = 154    # grasp_right_constants.py:121
```

### 검증

```bash
cd /home/user/rl_ws/sim2real && python3 -m pytest scripts/test_grasp_obs_builder.py -q
```

- 차원 154 assert
- 각 세그먼트 슬라이스 오프셋이 위 표와 일치
- 접촉력 정규화: 입력 20N → 출력 clamp 1.0 확인
- 관절오차 부호 보존: target < actual 이면 음수

---

## 3. Step 2 — tip 힘 발행을 3축 벡터로

### 대상
- `scripts/tip_contact_core.py` (72줄) — 현재 `norm(f - bias)` 스칼라 (`:66-69`)
- `scripts/nodes/tip_contact_pub.py` (101줄) — `/dg5f_right/contact_forces` 5D `Float64MultiArray` (`:41`)

### 변경

bias 제거는 유지하고 **norm 을 취하지 말고 3축 벡터를 그대로 발행**한다.

```
현재: force[t] = norm(f - bias[t])              → (5,)
변경: force[t] = f - bias[t]                    → (5, 3)  발행 시 flatten (15,)
```

- bias 도 3축 벡터로 누적/차감한다.
- `Float64MultiArray` 로 15D 발행. **레이아웃은 tip-major** `[tip0_xyz, tip1_xyz, ...]` —
  sim 의 `.view(N, -1)` 이 `(N,5,3)` 을 이 순서로 편다.
- 토픽명 유지 여부는 판단에 맡기되, **차원이 5→15 로 바뀌므로 구독자 전부를 함께 고쳐야 한다.**
  기존 5D 를 쓰는 곳: `grep -rn "contact_forces" /home/user/rl_ws/sim2real` 로 전수 확인.

### 주의

- binary contact 이 다른 용도(리프트 게이트)로 여전히 필요하다.
  `norm(vector) > CONTACT_FORCE_THRESHOLD(0.1)` 로 파생시켜라 —
  별도 토픽을 새로 만들지 말고 구독자 쪽에서 계산.
- **bias 미완 구간엔 0 을 유지**하는 현 동작을 그대로 지켜라(무접촉 전제).

### 검증

`scripts/test_tip_contact_core.py` (있으면 갱신, 없으면 신설):
- bias 제거가 축별로 동작
- bias 미완 시 0
- norm 파생 binary 가 임계 0.1 에서 전환

---

## 4. Step 3 — action 디코더 21D 화

### 대상
- `scripts/grasp_action_decoder.py:120-180` (`GraspFingerController`)
- `scripts/test_grasp_action_decoder.py` (5D 전제)
- `scripts/nodes/grasp_inference.py:209` `action_dim=11`, `:290,:460` `np.zeros(11)`

### sim 의 정답 (`grasp_right_env.py:880-896, 1000-1039`)

```python
# 1) (15,) → (5,3)  [손가락, 채널]   채널 0=_1 외전 / 1=_2 MCP / 2=_3·_4 공통
finger_action = action[6:21].reshape(5, 3)

# 2) 4지 공통닫힘 — 검지~소지를 채널별 평균으로 묶는다. 엄지는 독립.
if couple_four_fingers:                      # cfg 기본 True
    common4 = finger_action[1:5, :].mean(axis=0)
    finger_action = np.concatenate([finger_action[0:1], np.tile(common4, (4, 1))])

# 3) [-1,1] → [0,1] 절대 폐쇄도
cmd_ch = 0.5 * (np.clip(finger_action, -1, 1) + 1.0)          # (5,3)

# 4) 채널 → 20관절 전개.  [_1,_2,_3,_4] ← [ch0, ch1, ch2, ch2]
cmd20 = np.stack([cmd_ch[:,0], cmd_ch[:,1], cmd_ch[:,2], cmd_ch[:,2]], axis=1).reshape(-1)

# 5) ★래칫 제거 — 절대 목표를 향해 변화율 상한으로 이동(감소 가능)
delta   = np.clip(cmd20 - close_buf, -rate, +rate)            # rate = finger_close_speed = 0.05
advance = delta * (1.0 - gate20)
close_buf = np.clip(close_buf + advance, 0.0, 1.0)
```

`gate20` 은 현행 유지: `_1/_2` 무게이트, `_3/_4` 는 `(distal | tip)` 동결.
**라이브 배포는 distal 감지가 불가하므로 `distal_contact=None` → tip-only 게이트**
(`grasp_action_decoder.py:155-161` 의 현 설계 그대로). 이 차이는 의도된 것이니 유지한다.

`retighten_after_latch` 는 sim cfg `:382` 에서 **False** 다 → 배포도 현행(래치 후 동결 유지)이 맞다.
**이 값이 나중에 True 로 바뀌면 배포도 같이 바꿔야 한다** — 주석으로 남겨라.

### 병행 수정

- `grasp_inference.py:209` `action_dim=11` → `21`
- `:290`, `:460` `np.zeros(11)` → `np.zeros(21)`
- `:619` `self.last_actions = action.copy()` 는 그대로 (차원만 따라감)

### 검증

```bash
python3 -m pytest scripts/test_grasp_action_decoder.py scripts/test_grasp_decoder_parity.py -q
```

- 4지 공통닫힘: 검지~소지 채널값이 서로 다른 입력을 넣어도 출력 20D 에서 4지가 동일
- 채널 전개: `_3` 과 `_4` 지령이 항상 같음
- **래칫 제거**: `cmd` 를 1.0 → 0.0 으로 낮추면 `close_buf` 가 **감소**하는지
  (구현 전 이 테스트는 반드시 실패해야 한다)
- 변화율 상한: 한 스텝 변화량 ≤ `finger_close_speed`

---

## 5. Step 4 — palm delta 축별 + 고정 홈 리셋

### 5-A. palm delta 축별

`grasp_inference.py:129-130, 242-244`

```python
# 현재
PALM_DELTA_XYZ = 0.15
self.delta_mins = np.array([-PALM_DELTA_XYZ] * 3 + [-_dr] * 3)

# 변경
PALM_DELTA_XYZ = (0.15, 0.35, 0.15)          # grasp_right_env_cfg.py:301
self.delta_mins = np.array([-PALM_DELTA_XYZ[0], -PALM_DELTA_XYZ[1], -PALM_DELTA_XYZ[2]] + [-_dr] * 3)
self.delta_maxs = -self.delta_mins
```

`scale_palm_delta` 자체는 `delta_mins/maxs` 를 주입받으므로 **수정 불필요**
(`grasp_action_decoder.py:40-51`). 호출자만 고치면 된다.

### 5-B. 리셋을 고정 홈으로

`grasp_inference.py:471-500` `_compute_pregrasp()`

```python
# 현재 (컵 참값 의존 — 폐기 대상)
pregrasp_pos = self.cup_pos + PREGRASP_OFFSET      # PREGRASP_OFFSET = [0, -0.12, 0.05]

# 변경 (컵 무관 고정 홈)
RESET_HOME_PALM_POSE = (0.28, -0.38, 0.42, 90.0, 0.0, 90.0)   # x,y,z, ez,ey,ex(deg)
                                                # grasp_right_env_cfg.py:250
pose6 = np.concatenate([RESET_HOME_PALM_POSE[:3],
                        np.radians(RESET_HOME_PALM_POSE[3:])])
```

- Fabrics IK rollout(`PREGRASP_FABRICS_STEPS`) 구조는 **그대로 재사용**한다.
  입력 palm 목표만 컵 기반 → 고정 홈으로 바꾸면 된다.
- **액션 기준점도 홈으로 고정**: `:636` `self.pregrasp_palm_pose + delta` 의
  `pregrasp_palm_pose` 가 홈이 되므로 자동으로 따라간다. 변수명을 `home_palm_pose` 로
  개명하면 의도가 분명해진다(선택).
- 홈은 IK 를 **1회만** 풀면 된다 → `_compute_pregrasp()` 를 에피소드마다 부르지 말고
  노드 초기화 시 1회 계산해 캐시하는 편이 낫다. 단, 이건 최적화이니 동작이 먼저다.
- 홈이 palm workspace 밖이면 **조용히 클램프하지 말고 예외**를 던져라
  (sim `grasp_right_env.py:639-644` 가 그렇게 한다).

> `self.cup_pos` 는 여전히 **obs 입력으로 필요**하다(`palm_to_cup`, `cup_to_fingertip`).
> 리셋에서만 쓰지 않는 것이지 구독을 끊으면 안 된다.

### 검증 (하드웨어 없이)

`scripts/analysis/grasp_loop_sim.py` 오프라인 폐루프 재현기로:
- 홈에서 시작해 액션 y 가 컵 방향으로 갈 때 palm 이 실제로 컵 y 까지 도달하는지
  (±0.35 가 있어야 도달, ±0.15 면 못 감 — 이게 회귀 테스트가 된다)
- `grasp_loop_sim.py:100` 의 `action_dim` 도 21 로 갱신 필요

---

## 6. Step 5 — obs parity 테스트 신설 ★가장 중요

현재 **sim ↔ sim2real 을 잇는 obs parity 테스트가 양쪽 어디에도 없다**
(`grep -rl "sim2real\|parity"` in hdgp grasp_v1 tests = 0건).
Step 1~4 를 다 해도 이게 없으면 다음번에 또 어긋난다.

### 신설: `scripts/test_grasp_obs_parity.py`

`test_grasp_decoder_parity.py` 의 패턴을 그대로 따른다
(hdgp 소스를 import 하고, 없으면 `pytest.skip`).

**검증 방법**: 동일한 가짜 상태를 만들어 두 경로에 넣고 결과를 비교한다.

- sim 쪽은 `_get_observations` 가 env 인스턴스를 요구해 직접 호출이 불가하다.
  → **레이아웃 계약만이라도 고정하라**: `grasp_right_env.py:1222-1235` 의 `torch.cat`
  세그먼트 순서·차원을 소스에서 파싱하거나, `hdgp` 상수(`NUM_OBSERVATIONS` 등)와
  빌더의 `ACTOR_OBS_DIM` 이 일치하는지 assert.
- 최소한 다음은 반드시 assert 한다:
  - `ACTOR_OBS_DIM == hdgp NUM_OBSERVATIONS` (154)
  - `NUM_ACTIONS == hdgp NUM_ACTIONS` (21)
  - `CONTACT_FORCE_MAX`, `JOINT_POS_ERR_MAX` 가 hdgp 상수와 동일
  - 세그먼트 오프셋 테이블이 sim 조립 순서와 동일

> 여유가 있으면 `hdgp` 쪽에 순수 함수 `assemble_actor_obs(...)` 를 **추출**해
> 양쪽이 같은 함수를 쓰게 하는 게 근본 해법이다. 다만 **학습 중에는 sim 을 건드리지 말 것** —
> 학습 종료 후 별도 작업으로 남겨라.

---

## 7. Step 6 — 경로·문서 정리

### 7-A. 체크포인트 경로 (현재 전부 깨짐)

| 참조 | 경로 | 상태 |
|---|---|---|
| `grasp_inference.py:5-6` docstring | `.../grasp-v1/lstm_test3/nn/...ep_20000_rew_9920.256.pth` | ❌ `lstm_test3` 폴더 없음 |
| `docs/RUNBOOK_GRASP_V1_LIVE.md` B3 `--ckpt` | 위와 동일 | ❌ |
| `docs/RUNBOOK_GRASP_V1_LIVE.md` B3 `--agent` | `.../lstm_test3/params/agent.yaml` | ❌ |
| 저장소 사본 | `checkpoints/grasp_v1_right/grasp_v1_right_ep20000.pth` + `agent.yaml` | ✅ 존재하나 **어떤 코드도 참조 안 함** |

`checkpoints/grasp_v1_right/agent.yaml` 헤더는 `"Actor 132D … action 26D"` 로
**114D 도 154D 도 아닌 또 다른 세대(v10-3)** 다. 그대로 쓰면 안 된다.

→ **학습 완료 후** `hdgp/log/rl_games/open-tesol/right/grasp-v1/lstm_test2/` 의
최종 ckpt + `params/agent.yaml` 을 `checkpoints/grasp_v1_right/` 로 복사하고,
docstring·런북 경로를 그쪽으로 통일하라. **지금은 학습 중이라 확정 불가** —
이 단계는 마지막에 한다.

### 7-B. 계약 매니페스트 신설

`checkpoints/grasp_v1_right/CONTRACT.md` 에 다음을 명시:

- obs 세그먼트 순서·차원 (§2 표)
- action 레이아웃 (palm 6 + 손가락 15, 채널 의미)
- `palm_delta_xyz = (0.15, 0.35, 0.15)`, `palm_delta_rot_deg = 20.0`
- `reset_home_palm_pose = (0.28, -0.38, 0.42, 90, 0, 90)`
- `CONTACT_FORCE_MAX=10.0`, `JOINT_POS_ERR_MAX=1.2`, `CONTACT_FORCE_THRESHOLD=0.1`
- `finger_close_speed=0.05`, `couple_four_fingers=True`, `retighten_after_latch=False`
- palm workspace 박스 (`grasp_right_preset.py:171-189`, `MAX_POSE_ANGLE=45.0`)
- 리프트 래치 조건 (`lift_wait_joint7_delta=0.31`, `warm_j7_min/max=0.20/1.50`)

### 7-C. stale 문서 정정

**extrinsics 는 실제 캘리브되어 있다**(2026-08-02 ChArUco, 재투영 0.274px).
아래 3곳이 아직 PLACEHOLDER 라고 적어 오해를 부른다:

- `/home/user/rl_ws/perception/SETUP_GUIDE.md:233`
- `/home/user/rl_ws/perception/CHANGELOG.md:34`, `:138`
- `/home/user/rl_ws/perception/docs/EXTRINSICS_CALIBRATION.md:8`

정정하면서 **"목 자세 pan=−90 / tilt=280 에서만 유효"** 라는 조건을 반드시 함께 적어라.

런북 B1(a) 의 `~/rl_ws/perception_plus_plus/scripts/run_cup_pose_live.sh` 는
**이 워크스페이스에 존재하지 않는다.** 실재 스크립트는
`/home/user/rl_ws/perception/launch/run_cup_pose_{standalone,tracking}.sh`.
(vision-3090 원격에만 있을 수 있으니, 원격 경로면 그렇게 명시하라.)

또한 `SIM2REAL_INFERENCE.md` / `sim2real_inference.py` / `sim2real_dryrun.py` 는
**구 v7(106D)** 인데 deprecated 표시가 없다 → 표시 추가.

---

## 8. 실행 순서와 커밋 단위

```
① Step 2 (tip 3축 발행)        → 커밋  feat(s2r): tip F/T 3축 벡터 발행
② Step 1 (obs 154D)            → 커밋  feat(s2r): actor obs 154D 계약 동기화
③ Step 3 (action 21D)          → 커밋  feat(s2r): action 21D + 래칫 제거 + 4지 공통닫힘
④ Step 4 (delta 축별 + 고정홈) → 커밋  feat(s2r): 축별 palm delta + 고정 홈 리셋
⑤ Step 5 (parity 테스트)       → 커밋  test(s2r): sim↔배포 obs 계약 parity
⑥ Step 6 (경로·문서)           → 커밋  docs(s2r): 계약 매니페스트 + stale 경로 정정
```

Step 2 를 먼저 하는 이유: obs 빌더가 3축 힘을 입력으로 받으므로 **발행 쪽이 먼저** 준비돼야 한다.

### 전체 검증

```bash
cd /home/user/rl_ws/sim2real && python3 -m pytest scripts/ -q
IsaacLab/isaaclab.sh -p scripts/analysis/grasp_loop_sim.py      # 오프라인 폐루프
```

---

## 9. 알려진 함정 (실측으로 확인된 것들)

- **주석을 믿지 말 것.** `grasp_right_env.py` 의 `# 8. last actions (11D)`,
  `last_actions, # 13`, `# 146D` 는 전부 stale 이다. 정답은 `grasp_right_constants.py`.
- **`openarm_fk.py` 는 구 URDF(openarm_tesollo) 전용**이라 현 로봇(bi_s_rl)에서
  palm y 가 14cm 틀리게 나온다. FK 가 필요하면
  `/home/user/rl_ws/urdf/generated/rl/openarm_tesollo_bi_s_rl.urdf` 를 직접 쓸 것.
- **`pkill -f <패턴>` 이 자기 자신을 잡는다.** 명령 문자열에 패턴이 들어 있으면
  셸까지 죽는다(ssh exit 255). `pkill -f "[r]l_games/..."` 처럼 브래킷을 쓰거나
  `kill -9 -$pid` 로 프로세스 그룹만 죽여라.
- **Hydra 오버라이드**: yaml 에 없는 키는 `+` 접두사가 필요하다
  (`+agent.params.config.full_experiment_name=...`). 없으면 즉시 죽는다.
- **비대화형 ssh 에 conda 가 없다.** `source ~/miniforge3/etc/profile.d/conda.sh &&
  conda activate proj-hdgp-py311` 필요. 그리고 그 과정이 `isaacsim/setup_conda_env.sh`
  를 타는데 거기서 `ZSH_VERSION` 을 참조하므로 `set -u` 와 충돌한다 → 활성화 구간만 `set +u`.
- **cfg import 는 Isaac 앱 없이는 불가**(`isaaclab` → `pxr`). 테스트에서 cfg 값이
  필요하면 `ast` 로 소스에서 읽거나 `pytest.importorskip` 을 써라.

---

## 10. 범위 밖 (이 계획서에서 하지 않는 것)

- **sim(`hdgp`) 수정** — 학습 진행 중. 종료 후 별도 작업.
- **손 PD 게인 정합** — sim `stiffness 5.0 / damping 2.0` vs 실기
  `dg5f_right_controller.yaml` `p=1.5, d=0.0`. 하드웨어 복구 후 실측 필요.
- **라이브 실행** — dg5f 손 전원 재인가가 선행되어야 한다.
- **체크포인트 확정** — 학습(`lstm_test2`) 완료 후.
