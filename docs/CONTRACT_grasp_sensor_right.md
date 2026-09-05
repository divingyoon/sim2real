# 배포 계약 — `tesollo_sensor__right`

> 이 파일은 `scripts/analysis/report_contract.py` 가 구성 프로필과 hdgp 소스에서 **생성**한다.
> 손으로 고치지 말 것 — 값이 바뀌면 다시 생성하라.

## 로봇 구성

| 항목 | 값 |
|---|---|
| 작동 팔 | `right` |
| 엔드이펙터 | `tesollo_dg5f` (20 DOF) |
| 자산 매니페스트 | `openarm_tesollo_sensor_rl_manifest.yaml` |
| Fabrics 자산 | `openarm_tesollo` / `OpenArmTeoslloPoseFabric` |
| Fabrics 월드 | `open_tesollo_boxes_no_table` |
| hdgp 패키지 | `openarm.tesollo.right.grasp_sensor` |

## 토픽

| 역할 | 토픽 |
|---|---|
| `arm_state` | `/joint_states` |
| `ee_state` | `/dg5f_right/joint_states` |
| `tip_force_xyz` | `/dg5f_right/tip_forces_xyz` |
| `tip_force_norm` | `/dg5f_right/contact_forces` |
| `arm_cmd` | `/isaacsim/right_arm_cmd` |
| `ee_cmd` | `/isaacsim/right_hand_cmd` |
| `arm_traj` | `/right_joint_trajectory_controller/joint_trajectory` |
| `ee_traj` | `/dg5f_right/dg5f_right_controller/joint_trajectory` |

## Observation (154D)

| # | 세그먼트 | 차원 | 슬라이스 |
|---|---|---|---|
| 0 | `arm_joint_pos` | 7 | `[0:7]` |
| 1 | `arm_joint_vel` | 7 | `[7:14]` |
| 2 | `finger_joint_pos` | 20 | `[14:34]` |
| 3 | `finger_joint_vel` | 20 | `[34:54]` |
| 4 | `palm_center_pos` | 3 | `[54:57]` |
| 5 | `fingertip_pos_rel_palm` | 15 | `[57:72]` |
| 6 | `palm_to_cup` | 3 | `[72:75]` |
| 7 | `cup_to_fingertip` | 15 | `[75:90]` |
| 8 | `tip_force_local` | 15 | `[90:105]` |
| 9 | `joint_pos_err` | 20 | `[105:125]` |
| 10 | `last_actions` | 21 | `[125:146]` |
| 11 | `object_onehot` | 8 | `[146:154]` |

- `tip_force_local` = **tip-local 프레임 그대로** / 10.0 N, clamp ±1
- `joint_pos_err` = (직전 전송 지령 − 실측) / 1.2 rad, **부호 보존**
- `last_actions` = 정책 원출력(4지 공통닫힘 **이전**)

## Action (21D)

- `[0:6]` palm delta (x, y, z, ez, ey, ex)
- `[6:21]` 손가락 5×3 채널 — ch0=`_1` 외전 / ch1=`_2` MCP / ch2=`_3`·`_4` 공통
- 처리 순서: 4지 공통닫힘(**clamp 이전**) → 절대 폐쇄도 → `[ch0,ch1,ch2,ch2]` 전개 → 변화율 상한 → lerp(APPROACH, FULL_GRIP) → 관절 한계 clamp

## 파라미터

| 키 | 값 | 비고 |
|---|---|---|
| `palm_delta_xyz` | (0.15, 0.35, 0.15) | ★좌우 **동일** — 액션은 미러되지 않는다 |
| `palm_delta_rot_deg` | 20.0 | 축별 ± |
| `reset_home_palm_pose` | (0.28, -0.38, 0.42, 90.0, 0.0, 90.0) | x,y,z,ez°,ey°,ex° |
| `max_pose_angle` | 45.0 | workspace 회전 여유 |
| `finger_close_speed` | 0.05 | 변화율 상한/step |
| `couple_four_fingers` | True | 3지 국소최적 차단 |
| `retighten_after_latch` | False | True 면 배포도 바꿀 것 |
| `lift_wait_joint7_delta` | 0.31 | 좌우 부호 반대 |
| `warm_j7_min/max` | 0.2 / 1.5 | |
| `lift_start_min_grip_fingers` | 3 | |
| `grasp_ready_hold_steps` | 8 | |
| `fabrics_dt` × `fabric_decimation` | 0.016667 × 2 | |
| `fabrics_damping_gain` | 20.0 | 메인 |
| `reset_fabrics_damping_gain` | 10.0 | 홈 IK 전용 |
| `CONTACT_FORCE_THRESHOLD` | 0.1 | 실물 노이즈 위로 튜닝 |
| `EPISODE_STEPS` | 600 | grasp 480 + lift 120 |
| `PREGRASP_FABRICS_STEPS` | 60 | 홈 IK rollout |

## 홈 자세 기준값

`q_home` (sim preset 유도) = `[0.0431, 0.6706, 0.0961, 0.7342, 0.375, 0.5678, 0.6709]`

배포의 홈 IK 결과가 이 값과 0.05 rad 이상 다르면 **RuntimeError**. Fabrics 자산이
sim 과 다른지 먼저 의심할 것 — 구 자산은 palm 이 6.5cm 짧다.

## 배포–sim 의 의도된 차이

- **tip-only 게이트**: sim 은 손가락 동결에 (tip|mid|distal) 을 쓰지만 middle/distal 은
  critic 전용(privileged)이라 실기서 감지 불가하다. 라이브는 tip 접촉만 쓴다.
- **접촉 거리 게이트**: 실물 F/T 는 테이블 접촉도 잡는다(sim 접촉센서는 컵-필터).
  palm–컵 거리 0.10 m 밖에서는 접촉을 0 으로 만든다.

