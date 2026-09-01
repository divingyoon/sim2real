# both/pour_sensor 양팔 배포 스택 (RA-L G4)

RA-L 논문 정책(`tesollo/both/pour_sensor`, 양팔·15D)을 실기에 올리기 위한 배포 코드.
기존 스택은 `tesollo/right/pour_v1`(한팔·12D)용이었다.

## 핵심 발견 — 포팅 부담이 작다

2026-08-16에 두 env를 코드 레벨로 대조한 결과:

| 항목 | pour_v1 | pour_sensor | 결론 |
|---|---|---|---|
| actor obs | 55D | 55D | **레이아웃 동일** — pour_v1 빌더가 이미 좌팔 9+9 슬롯과 target 컵 pose 인자를 갖고 있었다(한팔에선 0/정적값) |
| 컵 지오메트리 | rim z=0.100, r=0.045, dyn 0.15/0.30 | 동일 | 그대로 재사용 |
| 디코더 상수 18종 | palm_delta 0.03/15°, ema 0.7, gate 0.06/0.25, beta 0.854/3.0/0.06, z_margin 0.03, inner_r 0.041, corridor(0.015,−0.02,0.12,20), latch 0.60 | **전부 동일** | `pour_action_decoder` 그대로 재사용 |
| 모드 플래그 | b_trajectory / palm pivot / z_lock / orient_release | 동일 | 동일 |
| action | 12D | **15D** ([12:15]=왼팔 TCP) | ← 차이 |
| receiver 컵 | 정적(실측 pose) | 왼팔이 파지 | ← 차이 |

**→ 오른팔 제어 경로 전체를 재사용**하고 차이분만 새로 썼다.
이 전제는 `test_pour_sensor_bimanual.py`의 drift-guard 18개가 hdgp cfg와 매 실행 대조한다.

**비전 불필요**: actor obs 55D는 전부 proprio + FK다(양팔 관절, 손 엔코더, FK 파생 기하,
직전 action). 시작 시 grasp offset을 1회 캘리브하면 이후 컵 pose는 FK로 나온다.

## 구성

| 파일 | 역할 |
|---|---|
| `scripts/pour_sensor_bimanual.py` | 차이분 어댑터 — 15D 분해, `LeftTcpController`(누적·클램프·z캡·frozen), receiver 컵 FK |
| `scripts/pour_sensor_inference.py` | 배포 노드 — `PourInferenceNode` 상속, 좌팔 obs/명령만 덮어씀 |
| `scripts/test_pour_sensor_bimanual.py` | 어댑터 46 + drift-guard. 기존 30개와 합쳐 **76 pass** |
| (재사용) `pour_obs_builder` / `pour_obs_geometry` / `pour_action_decoder` | 무수정 |

`pour_inference.py`는 `ACTION_DIM` 클래스 속성 1줄만 추가(기본값 불변 → pour_v1 동작 동일).

## 배포 범위 — M0(frozen receiver) 축소

기본값은 `--receiver frozen`: 왼팔이 rest 자세로 receiver를 든 채 고정되고 오른팔만
정책이 구동한다.

**근거**: 논문이 nominal C0에서 M4(learned) 95.1% ≈ M0(frozen) 94.0%이고, EXP-2에서
receiver를 freeze/scale 0.5/delay 6으로 바꿔도 성능이 떨어지지 않음을 보고했다. 즉
frozen 배포는 임의 축소가 아니라 **논문 자체 분석이 정당화하는 범위**이며, 본문에도
그렇게 기술한다.

`--receiver learned`도 있으나 왼팔 DiffIK 경로가 미검증이므로 기본값이 아니다.

## 안전 계약

- **왼팔 TCP z 하강 금지** (`LEFT_TCP_Z_DOWN_M=0.0`) — receiver 컵이 테이블을 뚫는 것을
  구조적으로 차단. sim에서 adversarial probe로 실측 검증된 항목이며 테스트가 잠근다.
- 왼팔 TCP는 rest 기준 ±8 cm 박스, 스텝당 1 cm로 클램프.
- 왼팔 rest 자세·관절 이름이 hdgp preset과 일치하는지 테스트가 대조.

## 남은 일 (실기 진입 전)

1. **parity 검증** — 동일 체크포인트로 sim rollout vs 브리지 rollout 궤적 오차 측정.
   이 게이트를 통과하기 전에는 실기에 올리지 않는다.
2. `--target-cup` 실측값 교체 (현 기본값은 sim 유래).
3. 왼팔 `/joint_states` 토픽에서 `l_aj_*`·`l_hj_gripper_*` 이름이 실제로 오는지 확인
   (다르면 노드 상수 수정 — 미수신 시 rest로 채우고 경고).
4. dry-run(비드 없이 자세) → 비드/물 20 trial × 컵 2~3종 → Fig.7.
