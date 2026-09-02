# Step 3 실기 라이브 라운드 — 절차서 (09.02)

아키텍처: **Isaac-in-the-loop, 소유권 상태기계.**
- 유휴(idle): 실기→sim 미러 (echo 35f, 러너가 로봇 상태를 실측으로 씀)
- 전환(transit)·정책(policy): sim 이 로봇 소유, **sim 실측**을 50Hz 스트림 →
  실기가 그림자 추종 (지령 스트림은 긁는다 — 08.31 좌팔 라이브 실증)
- 정책 중 |sim−real| 팔 괴리 0.5 rad 가 0.5초 지속 → 자동 중단
- abort: 명령 파일에 `abort` — 스트림 즉시 정지(실기는 현 자세 유지)

## 사슬

```
러너(--real --console --stream <robot>:47331 --echo-port 47332 --live-follow 46011)
  ├─ stream 35f(0x5A2B11) → udp35_to_ros_cmd.py --execute (:47331)
  │     → /isaacsim/{left_arm,left_gripper,right_arm,right_hand}_cmd
  │     → isaacsim_cmd_to_jtc.py --robot gripper_left        (max-vel 0.5)
  │     → isaacsim_cmd_to_jtc.py --robot tesollo_bi_s__right (max-vel 0.5 / hand 1.0)
  └─ echo 35f(0x5A2B12) ← joint_states_to_udp.py (실기 /joint_states, 50Hz)
비전: RealSense → fpp_cup + shaker_centroid → relay×2 → pose_udp_tx (:46011)
```

## 브링업 체크리스트 (순서 고정)

1. ⚠ **테솔로 손 전원 격리 실험** — 손가락 주변 공간 확보 후 전원만 인가,
   자체 zero 정렬 여부 관찰(09.02 사고 재현 확인). 이상 시 중지·보고.
2. 모터 전원 → openarm bringup (`HDGP_V2_VENDOR_GAINS=1` — 좌 v2B25 필수)
3. 테솔로 드라이버 + 손게인 p4.5/d0 재적용 (RAM — bringup 마다)
4. head: I게인 400 재적용 → head_home (pan 0 / tilt −20)
5. gravity_comp_node 상주 · net_preflight 통과
6. 브리지 3종 기동(위 사슬) — **udp35 는 먼저 DRY 로** 흐름 확인 후 --execute 재기동
7. 비전 사슬 기동 → 컵 2개 배치 → capture JSON 2개 갱신
8. 러너 부팅(--real) — sim=차렷 미러 확인 (status 로 echo 수신 확인)

## 라운드 (각 동작 사용자 승인)

명령 파일: `echo <cmd> > <console-path>` — reset|preset|left|right|attention|abort|status|quit

★**편측 preset**(09.02 추가): `preset left|right|both`, `attention left|right|both`.
쓰지 않는 팔은 차렷에 둔다 — 우 j7 은 preset 자세에서 **3.17 N·m 를 18분 연속**(한계 7)
물어 과열 고장났다. 차렷에서는 0.03 N·m 다. **preset 상태 5분 이상 방치 금지.**

- **라운드1**: `preset left`(승인) → 컵 스폰 확인 → `left`(승인) → 결과
  → `attention left`(승인) → 사용자 컵 재세팅 → `reset`
- **라운드2**: `preset right` → `right` → `attention right` → 컵 재세팅 → `reset`
- **라운드3**: `preset`(both) → `left` → `right` → (pouring initial state 는 별도 승인)

각 라운드 뒤 `[preset검증:좌팔/우팔]`(sim−홈)·`[preset검증3:…]`(**실기**−홈)을 함께 읽는다.
합격선 실기−홈 |max| ≤1.5°.

전환 bag: 우 reset_right_v2(홈 정합 0.0°) · 좌 reset_left + 브리지 램프(j4 21°·j7
28.6° 갭, 2.5s). attention 은 손 릴리즈(2s) → 팔 사전정렬 램프 → 역재생(좌 먼저).
전 구간 속도 ≤0.25 rad/s.

## 안전 규약

- 실기 동작은 전부 단계별 사용자 승인 · 이상 시 `abort` 또는 E-stop
- preset 은 시작자세가 차렷이 아니면 자체 거부(|Δ|>0.3 rad)
- live-follow 는 정책 시작 후 잠금(reset 이 해제) — 들린 컵 끌어내림 방지
- bag 기록: ACTION/SIM/REAL 동형 규약(기존 record 스크립트)
