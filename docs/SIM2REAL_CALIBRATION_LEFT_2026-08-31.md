# 좌팔 sim2real 보정 실측 — 다음 학습에 반영할 것 (2026-08-31)

> 대상: `open-grip_l_grasp_sensor_v2` (v2H_wide) 라이브 그림자 6회 실측.
> 결론 요약: **관절 추종은 문제가 아니다**(실팔 vs sim팔 평균 ≤1.1°). 긁힘의 진범은
> ①"지령 vs sim실측" 괴리(손목 5~6°)와 ②테이블 절대높이 캘리브(미실측) 둘이었다.

## 1. 실측 수치 (재학습 튜닝의 입력)

### 정적 처짐 (실측−지령, 정착 후 · 홈+frame150/250 평균)
| 관절 | 처짐 [mrad] | 비고 |
|---|---|---|
| l_aj_1 | +67 | 자세의존(44~80) — 마찰/백래시 성분 |
| l_aj_2 | +30 | |
| l_aj_3 | +10 | |
| l_aj_4 | −55 | 팔꿈치 중력 |
| l_aj_5 | −17 | |
| l_aj_6 | 0 | |
| l_aj_7 | +73 | ★과열 전력 관절 |

- **선보상 값**(지령에 더함, canonical): `-0.0666,-0.0298,-0.0101,+0.0549,+0.0174,-0.0001,-0.0732`
- 적용처: `isaacsim_cmd_to_jtc.py --arm-offset` (08.31 신설)
- 보상 후 정적 잔차: 전 관절 |≤22| mrad

### sim 자체 처짐 (지령 vs sim 실측 — 정책이 실제로 산 몸)
- sim 관절 추종오차 mean **0.132 rad**, 특히 **j5 +100 / j7 −87 mrad 상시**
- ★★핵심 교훈: **정책의 무접촉·파지 성공은 "지령"이 아니라 "지령−sim처짐" 자세의 사실**이다.
  실기가 지령을 충실히 따르면(보상 후) 오히려 sim 과 다른 몸이 되어 긁는다.
  → 그림자 스트림은 `--stream_meas`(sim 실측 송신)가 기본. run6 실증: 실팔 vs sim팔
  mean ≤20 mrad·RMSE ≤24·max ≤76 (지연 180 ms 보정) · 긁힘 거의 소멸(사용자 관찰).

### 동적
- 명령경로 지연: **+180 ms** (scale 0.25 기준, 교차상관)
- 과도 스파이크: 빠른 손목 구간 max 15°(l_aj_7) — scale 을 올리면 커질 항목
- 라이브 감속: `--stream_rate_scale 0.25` (sim peak 3.73 rad/s → 0.93, 실기 한계 2.0)

## 2. 다음 학습(v2 재학습)에 반영할 것

1. **테이블 절대높이**: 실기 베이스→테이블 상판 높이를 **실측**해 sim 테이블 z 와 대조 (⬜ 미실측 — 최우선 TODO). 어긋난 만큼 sim 테이블/컵 z 수정.
2. **z 마진**: 접근·파지 경로에 손끝-테이블 여유(권고 ≥2~3 cm)를 보상/커리큘럼으로 강제. 근거: sim 은 처짐 낀 몸으로 마진 0 근처를 지난다.
3. **sim 게인 재검토**: sim kp400 의 자체 처짐(j5/j7 5~6°)이 "정책이 사는 몸"을 지령에서 분리시킨다. 선택지: (a) sim 처짐을 실기와 정합(게인/중력보상 정렬) (b) 현행 유지 + 배포는 stream_meas 규약 고정. 현재는 (b)로 동작 확인됨.
4. **그리퍼 스트로크**: 실측 0.0488 m > 프로필 상한 0.040 (D1 불일치 재확인) — URDF/프로필 정합 필요.
5. **l_aj_7**: 처짐 최대·과도 최대·과열 전력. 재학습 시 손목 롤 속도/사용을 아끼는 보상 고려.

## 3. 최적화 프레임워크 (r2s 동형 · 사용자 확정 방향)

한 bag = ACTION/SIM/REAL 시간동기 데이터셋:
```
/shadow/action       정책 원출력 7D        ┐
/shadow/sim_target   sim 지령(관절목표)     │ UDP v2 패킷 → udp_cmd_to_ros
/shadow/sim_meas     sim 실측(처짐 포함)    ┘
/joint_states        실기 실측 + effort
/left_*_controller/joint_trajectory  실기로 나간 지령(선보상 포함)
```
- 첫 데이터셋: `logs/shadow/rosbags/run6_143945` (1500스텝 · 판정 완료)
- 소비처: 지연/처짐/마찰 피팅(본 문서 §1 재산출), `hdgp/scripts/r2s_autotune`(게인 피팅) 입력과 동형

## 4. 재현 CLI (라이브 그림자 1회전)

```bash
# 0) CAN (사용자, sudo): can0/can1 FD 1M/5M — ip link set ... fd on
# 1) bringup (사용자 터미널 · 모터 전원 ON 후에!):
ros2 launch openarm_bringup openarm.bimanual.launch.py use_fake_hardware:=false \
    right_can_interface:=can0 left_can_interface:=can1
# ★모터 전원을 활성화 뒤에 넣으면 enable 을 못 받아 무동작(실측) — 반드시 전원 먼저.

cd ~/rl_ws/sim2real && source /opt/ros/humble/setup.bash && . .venv/bin/activate
# 2) bag
ros2 bag record -o logs/shadow/rosbags/run_$(date +%H%M%S) /joint_states \
  /left_joint_trajectory_controller/joint_trajectory /left_gripper_controller/joint_trajectory \
  /isaacsim/left_arm_cmd /isaacsim/left_gripper_cmd /shadow/action /shadow/sim_target /shadow/sim_meas &
# 3) 브리지(선보상) + 어댑터
python3 scripts/nodes/isaacsim_cmd_to_jtc.py --robot gripper_left --max-vel 1.0 \
  --arm-offset " -0.0666,-0.0298,-0.0101,+0.0549,+0.0174,-0.0001,-0.0732" &
python3 scripts/nodes/udp_cmd_to_ros.py --port 47311 --log logs/shadow/live_adapter.csv --execute &
# 4) 리셋(차렷→홈, 최초 1회) 또는 홈 복귀
python3 scripts/nodes/shadow_replay.py --sim logs/shadow/reset_both/reset_left.npz \
  --robot gripper_left --rate-scale 1.0 --allow-idle-arm-mismatch --execute   # 차렷에서
python3 scripts/ops/reset_pose.py --robot gripper_left --sim logs/shadow/sim_v2H_wide.npz --execute  # 근처에서
# 5) Isaac (★venv 없는 셸에서)
cd ~/rl_ws/IsaacLab && ./isaaclab.sh -p ~/rl_ws/sim2real/scripts/probes/probe_v2_shadow_record.py \
  --checkpoint ~/rl_ws/sim2real/logs/policy/left_v2H_wide/nn/v2H_wide_best.pth \
  --steps 1500 --out ~/rl_ws/sim2real/logs/shadow/sim_live.npz \
  --stream_udp 47311 --stream_rate_scale 0.25 --stream_meas --hold_open --gui
# 6) 분석 — bag 하나로 실팔 vs sim팔 (docs 본문 §1 수치 재산출)
#    (분석 스니펫: scripts/analysis/analyze_live_run.py + run6 판정 코드)
```

## 5. 함정 기록 (재발 방지)

- 모터 전원을 HW 인터페이스 활성화 **뒤에** 켜면 엔코더만 살고 enable 이 안 됨 → 재launch
- RViz 가 GPU SM 80~95% 를 먹는다(pmon 실측) — Isaac 부팅 8분 행의 원인. 라이브 중엔 끌 것
- Isaac `--gui` 는 완료 후에도 GPU 를 물고 있음 → `--hold_open` 으로 의도적 유지하거나 종료 확인
- venv 활성 셸에서 `isaaclab.sh` 실행 금지(isaaclab 모듈 못 찾음)
- j1 상시 6~7 N·m 는 접촉이 아니라 **마운트 기울기의 중력 유지토크**(양 bag 98% 구간) — effort 로 접촉판정 불가
- bag record 는 조용히 죽을 수 있다 — 분석 전 **커버리지(시각 범위) 확인 필수** (run5 실측: 567 s 공백을 옛값 클램프로 오독할 뻔)
