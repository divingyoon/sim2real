# 실기 런북 — 왼팔 단독: preset 리셋 → 정책 그림자 재생 (08.31)

> 우팔 작동 X · 인식(컵) 연결 X · sim 움직임을 실팔이 따라가는지만 본다.
> 도구는 `shadow_replay.py` — 정책을 돌리지 않고 기록을 재생만 하며,
> 실측 자세에서 0.1 rad/s 램프로 진입 + 중단조건 내장 + 재생 CSV 자동 기록.

## 사전 조건

- [ ] 실기 **왼팔 전 관절 0**(차렷)에서 시작 (규약)
- [ ] `left_joint_trajectory_controller` + `left_gripper_controller` 활성, `use_fake_hardware:=false`
- [ ] DDS 는 사용자 설정(재생 실행 PC ↔ 로봇 PC 가 같은 ROS 그래프)
- [ ] 재생 실행 PC 에 이 저장소 + 아래 npz 2개
      (다른 PC 라면: `scp logs/shadow/reset_both/reset_left.npz logs/shadow/sim_v2H_wide.npz <PC>:...`)

## ① 리셋: 차렷 → preset 홈 (52.1 s · 최대 0.250 rad/s)

```bash
cd ~/rl_ws/sim2real && source /opt/ros/humble/setup.bash && . .venv/bin/activate
# dry-run 으로 계획 확인 (발행 0)
python3 scripts/shadow_replay.py --sim logs/shadow/reset_both/reset_left.npz --robot gripper_left --rate-scale 1.0
# 실제 실행
python3 scripts/shadow_replay.py --sim logs/shadow/reset_both/reset_left.npz --robot gripper_left \
    --rate-scale 1.0 --log logs/shadow/real_reset_left.csv --execute
```
- 끝 자세 = preset 홈 `(-0.0136, -0.3757, -0.0010, +0.9336, -0.4655, +0.0003, -0.3306)` · 그리퍼 0.044(개방)
- sim 검증치: 최대 0.25 rad/s · 몸통 스침 시작부 6.9 N 뿐 · 홈 잔차 2.18°(중력보상 off 정적 처짐)

## ② 정책 그림자 재생 (홈에서 시작해야 함 — ① 직후)

```bash
# 1차: rate 0.25 (요구 peak 0.93 rad/s) — 붙는지부터
python3 scripts/shadow_replay.py --sim logs/shadow/sim_v2H_wide.npz --robot gripper_left \
    --rate-scale 0.25 --max-vel 1.0 --log logs/shadow/real_v2H_x025.csv --execute
# 2차: rate 0.5  → --max-vel 2.0
# 3차: rate 0.53 (bag 등가 속도, 요구 peak 1.98) → --max-vel 2.0
```
- ★`--max-vel` 기본 0.5 는 rate 0.53 요구속도(1.98)보다 낮아 **세트포인트가 기록에 뒤처진다** — 2·3차는 반드시 2.0 으로.
- ★정책 npz 첫 프레임 == 홈(차이 0.36°) — ① 을 건너뛰고 차렷에서 바로 ② 를 돌리면
  램프가 **검증 안 된 관절 직선**으로 홈까지 가므로 금지.

## 중단 (shadow_replay 가 자동으로 걸지만, 눈으로도)

관절 추종오차 > 0.3 rad · `l_aj_7` effort > 5 N·m · `/joint_states` 두절 1 s · 이상음/접촉 → 즉시 Ctrl-C

## ③ 판정 (재생 후, 아무 PC)

```bash
python3 scripts/shadow_report.py --sim logs/shadow/sim_v2H_wide.npz --real logs/shadow/real_v2H_x025.csv
```
- L3(실팔 vs arm_target) + 지연·지터가 나온다. rate 스윕 비교: 0.25 에서 붙고 1.0 에서
  벌어지면 대역폭, 0.25 에서도 일정 오프셋이면 중력 처짐 — 고치는 노브가 다르다.

## 알려진 것 (실기와 대조할 기대치)

- sim 자체의 관절 추종오차 mean 0.132 rad (kp400 기준) — 실기(kp70)는 더 처질 것
- 홈 정적 처짐: 리셋 프로브 실측 2.18° (comp off)
- ★sim 관찰(08.31): 리셋 잔차(6.4°)에서 정책을 시작하면 파지 실패 — 실기에서도 처짐이
  크면 같은 양상이 예상되나, **그림자 재생은 개루프라 이 문제와 무관**(따라가기만 판정)

## ②′ 라이브 그림자 (재생 대신 — sim 폐루프 실시간 추종) ★사용자 요청 구성

sim 안에서 obs(가상 컵 포함)→정책→액션이 돌고, 실기는 그 관절 목표를 실시간으로 따른다.

```bash
# [A] 이 PC — ROS 어댑터 (venv + ROS)
python3 scripts/udp_cmd_to_ros.py --port 47311 --log logs/shadow/live_adapter.csv --execute
# [B] 로봇 쪽 — 브리지 (기존 재생 사슬과 동일, rate-limit 이 하드 캡)
python3 scripts/isaacsim_cmd_to_jtc.py --robot gripper_left --max-vel 1.0
# [C] 이 PC — Isaac (★venv 비활성 셸에서!)
cd ~/rl_ws/IsaacLab && ./isaaclab.sh -p ~/rl_ws/sim2real/scripts/probe_v2_shadow_record.py \
    --checkpoint ~/rl_ws/sim2real/logs/policy/left_v2H_wide/nn/v2H_wide_best.pth \
    --steps 1500 --out ~/rl_ws/sim2real/logs/shadow/sim_live_run1.npz \
    --stream_udp 47311 --stream_rate_scale 0.25 --gui
```

- **감속 2중**: `--stream_rate_scale 0.25` 가 시간을 4배로 늘려 요구속도 peak 3.73→0.93 rad/s
  (사용자 경고 반영 — sim 속도 그대로는 실기가 망가질 수 있음), 브리지 `--max-vel` 이
  하드 캡. 익숙해지면 scale 0.5 → --max-vel 2.0 순으로 올린다.
- 실측: 이 PC sim 계산이 ~78 ms/스텝이라 scale 0.25(80 ms)와 정확히 맞아 지터가 최소다.
  **scale 1.0 은 이 머신에선 애초에 실시간이 안 나온다** — 올릴 땐 headless + 지터 확인.
- 시작 전 반드시 ① 리셋으로 홈에 도달해 있을 것(정책 첫 지령 == 홈, 0.36°).
- 검증: sim npz(arm_target) + 어댑터 CSV + `ros2 bag record /joint_states` → 시간 정렬 L3.
- 순서: 어댑터·브리지 먼저 → Isaac 마지막(부팅 ~1분 동안 스트림 없음, 브리지는 첫
  명령 전 발행 안 함 + 1s 두절 시 발행 중지라 안전).
