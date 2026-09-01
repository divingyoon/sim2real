# fab_test79 ep_3000 — S2R 배포 후보 (사용자 지정 최상, 08.28)

- 원본: `vision-3090:/home/usr/rl_ws/hdgp/log/rl_games/open-grip/left/grasp-sensor-fab/fab_test79/`
- 파일: `nn/last_open-grip_l_grasp_sensor_fab_ep_3000_rew_147.03294.pth`
- sha256 `5bead337d21c…edafe` (원본과 대조 완료 08.28 22:49)
- ★가져온 뒤에도 원본 디렉토리에 학습이 계속 쓰는 중이었다(ep_4000 rew 119.99 — ep_3000 보다 낮음).
  `open-grip_l_grasp_sensor_fab.pth`(best 자동저장)는 15:11 산출이라 ep_3000 이 **아니다** — 반드시 ep 파일명으로 지정.

## 체크포인트에서 직접 읽은 계약 (torch.load 실측)

- **obs 45 / action 7** · MLP 256→…→64 (separate=False) · **RNN 없음** · epoch 3000 · frame 73.7M
- action 7 = FabricPalmAction 6D(palm) + GatedBinaryJointPositionAction 1D(그리퍼)
- obs 순서 (env.yaml dump):

| # | 항 | 차원 |
|---|---|---|
| 1 | joint_pos_rel | 9 (팔7+그리퍼2) |
| 2 | joint_vel_rel | 9 |
| 3 | object_position_in_robot_root_frame | 3 |
| 4 | target_object_position (generated_commands) | 7 |
| 5 | last_action | 7 |
| 6 | gripper_gate_open | 1 |
| 7 | tcp_position_in_root | 3 |
| 8 | palm_rot6d_in_root | 6 |
|   | 합 | **45** |

- ★obs에 **가상 컵이 두 항**(object_position·gripper_gate) — sim 마스터 구성이 필수인 근거 그대로.
- ★지령 slew 상태 +21차원은 **이 런에 없다**(env.yaml 에 해당 항 부재).

## 재생 시 준수

- env 재현은 **이 디렉토리의 `params/env.yaml`** 기준 — 로컬 hdgp 소스는 옆 세션이 계속 수정 중이라 어긋날 수 있다.
- 리셋 자세는 이 체크포인트의 홈에 딸린다(policy-reset-pose-belongs-to-checkpoint).
