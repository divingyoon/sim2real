# policy_control — obs → policy → fabric → pd 4노드 제어 모듈 (2026-09-06 상태)

계획: `~/.claude/plans/shiny-foraging-hamming.md` (09.05 승인). 목표 = Isaac 없이 ROS2 제어만으로 학습 정책 실현, 노드 4단 구조.

## 1. 구성(모두 `sim2real/policy_control`, ament_python, `colcon build --symlink-install --packages-select policy_control` 통과)

```
sensors ─▶ obs_node ─/policy_control/obs─▶ policy_node ─/policy_control/action─▶ fabric_node ─/policy_control/joint_target─▶ pd_node ─▶ forward_{position,velocity,effort}_controller
           (episode 마스터, /policy_control/episode)                      (액션 디코더 + Fabrics)                    (MIT 3중 q*,q̇*,τ_ff · JTC 교대 · 중력 모드 · 가드)
```
- 계약 = `logs/policy/<run>/deploy_contract.json` (`tools/build_deploy_contract.py --run … --grasp-band v1`), 로봇/센서 배선 = `config/robots/*.yaml`, pd knob = `config/pd_{left,right}.yaml`.
- 노드 = `chain.py` 스테이지(순수) + 얇은 rclpy 껍질. 골든 오라클 = 기존 `LeftPolicyCore`/`GraspS2RCore`.
- launch: `policy_chain.launch.py`(obs/policy/fabric), `pd_controller.launch.py`(pd 단독, `execute:=false` 기본), `fake_plant.launch.py`(도메인 0 거부).
- 도구: `episode_ctl.py`(engage→goto_home→reset→start→run→stop→release, `--execute --approve`), `pd_selftest.py`, `replay_to_pd.py`, `pd_release.py`, `status_to_csv.py`, `chain_recorder.py`, `episode_judge.py`, `contract_doc.py`, `fake_plant_run.sh`.
- 미션: `config/mission_policy_control.yaml` (`scripts/ops/mission_run.py --mission … --plan`).

## 2. 검증 결과
| 항목 | 결과 |
|---|---|
| 단위·배선 테스트 | `pytest tests/policy_control -m "not gpu"` **340 passed** (계약 18·obs 49·policy·decoder/fabric 36·pd 112·chain 27·CM/backends 37·launch 10·노드 26 …) |
| 골든(1) obs | 좌 65스텝 기록 ≤1e-5(게이트 1스텝 시프트), LeftPolicyCore ≤1e-9; 우 e1 194스텝 GraspS2RCore ≤1e-9 |
| 골든(2) action | 좌 MLP ≤1e-5; 우 LSTM 194스텝 ≤1e-4, 리셋/중복 seq 규칙 |
| 골든(3) fabric parity | 좌 v2B25 300스텝 TCP max 0.089 mm; 우 g1 dec2+테이블 world 정상 0.05 mm·과도 ≤4.2 mm(run-to-run 비결정, 2단 문턱) |
| 골든(4) 체인 | 좌: obs 0.0·palm 0.0·팔 목표 0.0 rad vs 오라클; pd 법칙 1e-12, (kd/kp) 항 부재 회귀 |
| fabric 타이밍 | batch-1 스텝 p95 좌 5.9 ms(예산 7 = 0.35·dt) / 우 6.3 ms(4.17 예산 미달 → CUDA graph 전까지 xfail) |
| fake 플랜트 폐루프 | engage(load/configure/STRICT switch) → goto_home(0.1 rad/s + settle err 0.009) → reset/start → 정책 루프 → 가드(판 여유 abort·추종오차 HOLD) → stop → release(JTC 복귀) **전 경로 동작**. rate 모델 250스텝 완주(rc 0), 홉 지연 obs→pd p50 12 ms(워밍업 후) |
| 실 controller_manager(mock hardware) | `openarm_bringup use_fake_hardware:=true` 상대로 engage/goto_home/release 성공, 교대 후 JTC active·forward inactive 복원, 관절 스텝 ≤0.057°(램프)·최대 0.54°(ramp_tol 0.01 rad 종료) |

fake 플랜트의 **파지 성공은 미달**: MockArm(관절별 2차 PD+마찰+중력, 커플링·테이블 접촉 없음)이 sim 팔 동역학과 달라 2~4스텝 뒤부터 obs/액션이 갈린다(마찰 0·이상추종 모델 모두). 파이프라인 등가는 골든(1)~(4)가 증명하고, 실기 판정은 §4 단계표로.

## 3. 이번에 확정된 사실(메모리에도 기록)
- **좌 v2B25 fabric 홈 = J147**(`grasp_left_preset.LEFT_ARM_HOME_JOINT_POS`, j4 0.9336/j7 −0.3306) ≠ 로봇 리셋 홈(dump init_state = v2 `LEFT_ARM_HOME_LOW`, 0.5665/−0.8304). 계약 `fabric.home_q`(J147) 와 `pd.home_arm`(dump)로 분리. sim 도 시작 시 fabric 목표에 0.42 rad 뒤처지므로 `abort_tracking` 0.5.
- 좌 v2B25 파지 대역 = v1(판 위 10~85 mm) — 계약 생성 시 `--grasp-band v1` 필수.
- fabrics_sim: 프로세스당 1개, CUDA 필수, 첫 스텝 ~180 ms(노드 기동 시 워밍업).
- /policy_control/episode 구독 depth 는 10(reset 직후 start 가 오면 depth 1 은 reset 을 버린다).
- 우 g1 `_fab_to_env` z ≈ +5 mm(계약 기본 0) — 우 실기는 재학습 후.

## 4. 실기 1단계(좌 v2B25) 절차 — 단계마다 사용자 승인
`mission_run.py --mission config/mission_policy_control.yaml --plan` 의 순서: preflight → bringup(robot_control openarm_bringup) → pd_load(execute false) → pd_selftest(`tools/pd_selftest.py --execute`, engage 후) → goto_home → shadow_pd(`replay_to_pd.py --rate-scale 0.25`, vel_ff_cap 0→0.5→2.0) → observe_only → policy_reduced → policy_full. 실기용 knob 은 `config/pd_left.yaml`(execute false·vel_ff_cap 0·lead_sec 0.1·max_vel full 2.0). fake 용은 `pd_left_fake.yaml`.

## 5. 남은 일 / 알려진 한계
- 실기 전: `pd_selftest` 실기 검증, use_fake_hardware 에서 engaged 상태 SIGINT release 경로, `/objects/cup_big_s100/pose` 실기 z(datum 0.205) 정합, 카메라 사슬 +21 mm.
- 우팔: 재학습 후 새 계약(같은 스키마) + fabric CUDA graph(타이밍) + hand PID/RH56F1 백엔드 실검증 + obs 노드 FK 제공자(fabric FK 토픽 또는 CPU FK).
- pd: 실측 스테일 시 HOLD(현재는 송출 건너뜀), thermal act 후 자동 rest 없음(사람이 robotctl), 하드웨어 명령 신선도 카운터(패치).
- hdgp 재학습 전: robots.py/robot_profiles 옛 자산명, 새 fabric URDF 의 fabric_params, `export_deploy_contract.py`(train 종료 훅).

## 6. 양팔 DG-5F-M 지원(09.06 오후) — 사용자 결정 "양팔 dg5f-m 먼저 · assets/robot 구성이 기본 · 시험은 한 팔씩"

### 6.1 무엇이 바뀌었나
- **계약 v2**(`contract.py`, schema `policy_control/deploy_contract/v2`, v1 파일 로드 호환): `sides[left|right] = SideCfg(arm/hand 관절, ee_kind, palm_body `{p}_hl_palm`, tip_bodies, 홈, pd_groups, gravity, sim_gains, fabric, palm/hand 디코더, action_groups)`, `asset`, `control_only`. 최상위 fabric/action/pd 는 primary side 의 거울이라 기존 소비자는 무변경.
- **자산 레지스트리 + 제어 전용 계약**(`contract_assets.py`): 기본 자산 `openarm_dg5f-m_bi_rl`. `build_deploy_contract.py --asset …` → `logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json`(정책 없음, 홈 차렷 0 + 손 open pose(좌는 `_HAND_SIGN` 미러), 게인 = control_gains.yaml, 중력 model_tau_ff 양팔). `--run … --asset …` 은 런 계약을 새 자산에 재기반(fabric dir/params·soft limit·fk urdf 만) → `right_g1/deploy_contract.dg5f-m.json`.
- **fabric 자산**: `openarm_dg5f-m_bi_{left,right}` URDF 에 손가락 구 52개 패치, params `openarm_dg5f-m_{left,right}_pose_params.yaml`, meshes 심링크(`gen_fabric_urdfs.sync_hdgp` 자동). 좌 fabric 도 내부 관절명은 우측 → `fabric_core` 가 l/r 무시 인덱스 매핑(`joint_key`) + 실제 URDF 순서 검증.
- **robot yaml**: `dg5f_m_{right,left,bi}_{real,fake}.yaml` — `joint_profiles:` 병합(좌손은 `config/openarm_tesollo_left_hand.yaml`), 양팔 yaml 은 역할 접미사(`arm_left`…), `sources.select_side()`.
- **노드**: obs/fabric `side` 파라미터, obs `urdf_chain` FK(자산 URDF, CPU), fabric `mode=direct`(control_only: `/policy_control/palm_cmd`·`hand_cmd` 구독, `palm_pose` 발행), **episode_master**(제어 전용 계약의 에피소드 서비스/이벤트 — obs 노드가 없으니), pd 팔 그룹 N개(`sides` 파라미터, 이름 기반 joint_target 분배, 좌 dg5f ns `dg5f_left` PID 4.5 적용 경로), fake 플랜트 양팔(`fake_arm_bridge --sides`, CM 스텁 양팔, 손/팁 fake 네임스페이스별).
- **도구**: `palm_cmd.py`(상대/절대 palm 목표), `hand_cmd.py`(open→grip 보간·좌 미러·관절 덮어쓰기), `fake_direct_run.sh`(제어 전용 direct 폐루프 한 팔), `fake_plant_run.sh MODE=pd`(pd 전용 selftest), `chain_recorder` 손 목표·제어 전용 seq 정렬, `contract_doc` sides 표.
- **단계표**: `config/mission_dg5f_m_control.yaml`(12단계, 우·좌 독립: pd_load → selftest → goto_home(차렷) → fabric direct(palm ±2 cm·hand 0.3) → release).

### 6.2 검증 수치(fake, 도메인 96/97 — 실기 아님)
| 항목 | 좌 | 우 |
|---|---|---|
| 자산 URDF FK vs fabric FK(palm/tip) | ≤3.0e-7 m | ≤1.8e-9 m (fabrics_sim 대비 ≤5.2e-7) |
| direct 폐루프 palm +5 cm z 지령 → fabric palm 이동 | +5.03 cm | +5.01 cm |
| pd 추종 \|q*−q\| p50 / p95 / max | 4.9 / 18.2 / 69 mrad | 3.9 / 17.8 / 81 mrad |
| pd HOLD·가드 | 0 | 0 (run1 은 아래 6.3 ① 로 HOLD → 수정 후 0) |
| 손 open→0.3 close→open (index_2) | 0 → 0.570 → 0 | 0 → 0.570 → 0 |
| τ_ff(model) 홈/올린 자세 j4 | 0.37 / 3.63 N·m | 0.37 / 3.64 N·m |
| pd 전용 selftest(±0.1 rad, 관절 3·6) | 비율 0.97/1.02, drift 0 | 0.96/0.85, drift 0 |
| 우 g1 정책 fabric parity(새 자산) | — | median 0.04 mm · max 3.3–3.8 mm (구 자산과 같은 대역) |

런 디렉터리: `logs/policy_control/fake_direct_left_run4`, `fake_direct_right_run3`, `fake_dg5fm_{left_run4,right_run2,both_run2}`. 테스트 451(비GPU) + GPU 골든/parity 통과.

### 6.3 이번에 잡은 함정
1. **우 tesollo fabric 은 손 목표가 처음 바뀌는 tick 에 306 ms** 를 쓴다(지연 컴파일) → pd 워치독 0.25 s HOLD. 워밍업이 손 목표 변화 스텝(`WARM_UP_HAND_DELTA`)도 밟게 고침(재현 run2 최대 98 ms).
2. **palm_cmd/hand_cmd 는 구독자 매칭 전 발행이 버려진다**(RELIABLE 이어도). 도구가 `get_subscription_count()≥1` 을 기다린 뒤 발행(우 run2 에서 두 번째 hand_cmd 소실 → run3 정상).
3. 제어 전용 계약에는 **에피소드 마스터가 없었다**(obs 노드 몫) → `episode_master` 노드. launch 가 control_only 면 자동 포함.
4. `ros2 service call` CLI 가 가끔 응답을 못 받고 끝난다(요청은 처리됨) → 런 스크립트는 응답 없을 때만 1회 재시도.
5. 노드 스크립트를 launch `use_source` 로 직접 실행하면 상대 import 가 죽는다 → 모든 노드에 `__package__` 심(episode_master 도).

### 6.4 남은 일(양팔)
- 양팔 **동시** direct 모드는 topic 이 팔별로 나뉘지 않았다(`palm_cmd`/`palm_pose` 공유) — 한 팔씩 시험에는 무관, 동시 제어가 필요해지면 `<topic>_<side>` 로 분리.
- 실기: `mission_dg5f_m_control.yaml` 순서로 사용자 승인 하에(pd_load 는 무발행). 좌 DG-5F 드라이버 PID 4.5 적용은 engage 응답의 `hand gains` 사유로 확인(fake 는 파라미터 서버가 없어 mismatch 사유만 남는다).
- hdgp 재학습 전: `robots.py`/`robot_profiles` 자산명 교체 + `export_deploy_contract`(계약 v2 스키마, sides 포함).

## 7. PD 게인 = 벤더값만 (09.06 사용자 확정)

**규칙**: 학습·제어 전부 벤더값만. 팔 = OpenArm `control_gains.yaml`(kp 70/70/70/60/10/10/10 · kd 2.75/2.5/2.0/2.0/0.7/0.6/0.5), DG-5F 손 = 드라이버 PID(p 1.5 · d 0). 게인을 바꾸려면 벤더 파일을 고친다.

| 계층 | 이전 | 이후 |
|---|---|---|
| hdgp 학습 기본(`robots.py`) | 팔 400/80 · 손 5.0/0.165 | 벤더 · 벤더 |
| 우 g1 트랙(`grasp_s2r`) | kp 벤더 + **r2s kd**(`HDGP_S2R_REAL_GAINS` 스위치) / KUKA 300·45 기본 | 벤더 단일(스위치·KUKA 분기 삭제) |
| `grasp_sensor` · `grasp_ua` | KUKA 300/45 · 400/80 | 벤더 |
| 좌 v2B25 계열 preset | v1 KUKA 테이퍼 300/100/50/25, v2 만 벤더 | v1·v2 모두 벤더(동일 출처) |
| sim2real 계약 게이트 | kp 만 대조, kd 는 정보 | **kp·kd 둘 다** 대조 |
| 실기 손 PID | bringup 마다 4.5 재적용 | 벤더 1.5(드라이버 기본과 같음) |

- 단일 출처 `hdgp/…/agnostic/modules/vendor_gains.py` + 패키지 안 yaml 사본 2개(★학습 서버에는 `rl_ws/urdf` 가 없다). 강제 테스트 `tests/test_vendor_gains.py` 20개가 값·소비처·드리프트 3축을 잠근다. 폐지 트랙 13개는 `LEGACY_ALLOWED` 로 동결.
- `test_pc_pd_gains.py` 가 워크스페이스 `control_gains.yaml` 사본 6개 일치와 손 PID ↔ 벤더 파일 일치를 잠근다.
- 자산 USD 는 이미 벤더값이라 **재빌드 불필요**(build_usd.py 가 벤더 파일에서 이식: 팔 14 + 손 40 관절).

**명시 예외**(벤더 PD 가 존재하지 않음 — `NO_VENDOR_PD`): RH56F1 손(RS-485 위치 서보), 스톡 그리퍼 조(모터축 회전 게인 ↔ 직동 관절, 환산 없음), 머리(Dynamixel). 관절 friction 은 PD 게인이 아니라 규칙 밖이다.

**결과·리스크**
- `right_g1` 은 이제 계약 게이트를 통과하지 못한다(r2s kd, 그중 4개는 MIT 상한 5.0 밖). **의도된 실패** — 재학습 대상이며 계약·골든 픽스처용으로만 남는다. `left_v2B25` 와 자산 제어 전용 계약은 통과한다.
- ⚠벤더 손 p=1.5 는 4 s 주먹 램프에서 지령의 82 % 까지만 간다(4.5 는 98~101 %). sim 도 1.5 라 정합은 좋아지지만 **파지력·도달률 재확인이 필요**하다.
- ⚠게인이 바뀌면 동특성이 바뀐다 ⇒ 모든 재학습은 **FRESH**. 기존 체크포인트와 비호환.
