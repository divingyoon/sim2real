# 좌팔 배포 정책 — v2B25_tip30_ep2150 (2026-09-01 사용자 지정)

| 항목 | 값 |
|---|---|
| 파일 | `nn/v2B25_tip30_ep2150.pth` (git 미추적 — `logs/policy/**/*.pth`) |
| md5 | `27219429` (`272194299637fef6aa89c8e93161d1b6`) |
| epoch | 2150 · rew 15.894433 |
| 계약 | **obs 49 / act 7 / MLP** (RNN 없음 · hidden 64) |
| 태스크 | `open-grip_l_grasp_sensor_v2` |
| 출처 런 | `vision-3090:~/rl_ws/hdgp/log/rl_games/open-grip/left/grasp-sensor-v2/v2B25_tip30_fresh/` |
| 런 dump | `params/{agent,env}.yaml` (회수 완료) · `test_history.md` |

`v2B25` = 좌팔 v2 사다리 라운드 25 (tip floor 마진 30 mm). 라운드 22~25 의 소스는
`hdgp` 커밋 `585ea39` 에 고정했다 — 그 전까지는 dirty 트리 학습이었다.

## ★★홈이 현재 preset 궤적과 다르다

`params/env.yaml` 의 `scene.robot.init_state.joint_pos` (rad):

| 관절 | v2E19 (현 `reset_left.npz` 도착점) | **v2B25 (이 정책)** | 차이 |
|---|---|---|---|
| `l_aj_1` | −0.0136 | −0.0136 | 0.0000 |
| `l_aj_2` | −0.3757 | **−0.3255** | **+0.0502** (2.9°) |
| `l_aj_3` | −0.0010 | −0.0010 | 0.0000 |
| `l_aj_4` | +0.9336 | **+0.5665** | **−0.3671** (21.0°) |
| `l_aj_5` | −0.4655 | −0.4655 | 0.0000 |
| `l_aj_6` | +0.0003 | +0.0088 | +0.0085 |
| `l_aj_7` | −0.3306 | **−0.8304** | **−0.4998** (28.6°) |

**현재 좌팔 preset 궤적(`logs/shadow/reset_both/reset_left.npz`)은 v2E19 홈으로
간다.** 그 자리에서 이 정책을 돌리면 **학습한 적 없는 자세에서 시작**한다 — 팔꿈치가
21°, 손목이 28.6° 어긋난다. 실기에 올리기 전에 둘 중 하나가 필요하다:

1. v2B25 홈으로 가는 **새 preset 궤적**을 만든다 (기존과 같은 방식: 한 번 계산해
   굽고 이후엔 재생만), 또는
2. 좌팔 배포본을 v2E19 계열로 되돌린다.

[[policy-reset-pose-belongs-to-checkpoint]] 가 말한 그대로다 — 리셋 홈은 코드가
아니라 **체크포인트에 딸린 것**이고, 판정 근거는 live 코드가 아니라 **런 dump** 다.

## 재생 시

- obs 49 / act 7 은 `left_v2E19`·`left_v2H_wide` 와 **같다** → 라이브 사슬
  (`probe_v2_shadow_record.py` → `udp_cmd_to_ros.py` → `isaacsim_cmd_to_jtc.py`)
  은 드롭인으로 바뀐다. 바뀌는 것은 **홈뿐**이다.
- 그림자 목표는 sim **지령**이 아니라 sim **실측**(`--stream_meas`)이다.
- 선보상 오프셋은 `docs/SIM2REAL_CALIBRATION_LEFT_2026-08-31.md` §4 참조.
