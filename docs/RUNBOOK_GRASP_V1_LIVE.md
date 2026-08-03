# grasp-v1 라이브 실행 런북 (2026-08-03 재정립)

tesollo right grasp-v1 라이브 정책의 머신별 기동 명령. 커밋 62909aa(zeros 손 obs 방어 +
tip_contact_pub) 기준. 토폴로지: **vision-3090**(정책+지각) ↔ **5070ti robot PC**(브리지+드라이버),
유선 DDS(도메인 126).

---

## 0. 공통 규칙 — 모든 ROS 터미널(두 머신 공통)

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=126
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/fastdds_wired.xml
```

- ⚠️ **한 프로세스라도 프로파일을 안 물면 DDS 데이터가 단방향 불통**(discovery만 되고 콜백 0).
  의심 시 검증: `tr '\0' '\n' < /proc/<PID>/environ | grep FASTRTPS`
- 노드/브리지 **재기동 전 옛 프로세스 pkill 필수**(중복 2대가 두 번 사고 남).
- 유선 정적 IP: vision-3090 `192.168.100.1` ↔ 5070ti `192.168.100.2`
  (`ip addr`는 리부트 시 소실 — nmcli 영속화 권장).

---

## A. 5070ti robot PC (`usr@100.106.38.98`)

워크스페이스 소싱: `source ~/rl_ws/robot_control/ros_ws/install/setup.bash` (§0 다음에).

### A1. 전원 인가 후 CAN

```bash
sudo openarm-can-configure-socketcan-4-arms can0 -fd
```

### A2. 오른팔 bringup

```bash
ros2 launch openarm_bringup openarm.bimanual.launch.py use_fake_hardware:=false
```

- j7 주의: 손(~1.4kg) 부착 시 고피치 홀딩 과부하(7Nm 한계) — bringup만으로 15s 내 red 가능.
  저토크 자세 유지 or `robotctl pose gravity`(중력FF, `--urdf` 손질량 포함).

### A3. 손(dg5f) — ★Phase 1 복구 절차 포함

1. **손 전원 재인가** (Modbus 세션/펌웨어 꼬임 리셋 — 관절 0.000 고정 증상의 처방)
2. 드라이버 기동(**F/T 브로드캐스터 포함**):
   ```bash
   ros2 launch dg5f_driver dg5f_right_driver.launch.py delto_ip:=169.254.186.72 fingertip_sensor:=true
   ```
3. 검증(순서대로, 실패 시 벤더 테스트 스크립트부터):
   ```bash
   ros2 topic echo /dg5f_right/joint_states --once        # 0.000 고정이면 실패
   ros2 topic echo /fingertip_1_broadcaster/wrench --once  # F/T 나오는지
   ```

### A4. tip 접촉 변환 노드 (★손끝 무접촉 상태에서 기동 — bias 캡처)

```bash
python3 ~/rl_ws/sim2real/scripts/tip_contact_pub.py
# bias 재캡처(무접촉 자세에서): ros2 service call /tip_contact/rebias std_srvs/srv/Trigger
# 검증: ros2 topic echo /dg5f_right/contact_forces --once  → 무접촉 ~0, 컵 누르면 해당 tip 상승
```

- 임계 0.1N(CONTACT_FORCE_THRESHOLD)은 실물 노이즈 대비 튜닝 필요(무접촉 노이즈 < 0.1 확인).

### A5. 브리지

```bash
pkill -f isaacsim_cmd_to_jtc; sleep 1
python3 ~/rl_ws/sim2real/scripts/isaacsim_cmd_to_jtc.py --max-vel 0.1
```

- max_vel 0.1은 후퇴 원인 아님(08.03 재현 실험으로 무죄 판정) — 그대로 시작, 추종지연 체감 시 상향.

### A6. 관측(권장, 매 실행 병행)

```bash
python3 ~/rl_ws/sim2real/scripts/joint_monitor.py    # 로그 sim2real/logs/*.csv
```

---

## B. vision-3090 (`ssh vision-3090`)

### B1. 컵 pose — 택1

(a) **fake** (플러밍/Stage A):
```bash
python3 ~/rl_ws/sim2real/scripts/fake_cup_pose_pub.py --x 0.40 --y -0.15 --z 0.38
```

(b) **실컵** (FP++ 라이브):
```bash
# 목 고정(캘리브 자세 pan -90 / tilt 280 재현 필수)
python3 ~/rl_ws/sim2real/scripts/head_position_hold_node.py
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
~/rl_ws/perception_plus_plus/scripts/run_cup_pose_live.sh
python3 ~/rl_ws/sim2real/scripts/cup_pose_relay.py --in-type posestamped
# 검증: ros2 topic echo /cup_pose --once  (base 프레임, ~8Hz)
```

### B2. (손 분리 테스트 시에만) fake 손

```bash
python3 ~/rl_ws/sim2real/scripts/fake_hand_state_pub.py          # APPROACH 정적
python3 ~/rl_ws/sim2real/scripts/fake_hand_state_pub.py --echo   # 손 명령 반사(진화 obs)
```

- 실손 세션에서는 금지(드라이버와 토픽 충돌). 실손+접촉만 없을 땐 fake_tip_contact_pub 대신
  **A4 tip_contact_pub 사용**(이제 실센서 있음).

### B3. 정책 노드 (★venv 필수 — Isaac py3.11엔 rclpy 없음)

```bash
source /opt/ros/humble/setup.bash
source ~/grasp_infer_venv/bin/activate
export ROS_DOMAIN_ID=126
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/fastdds_wired.xml
python3 ~/rl_ws/sim2real/scripts/grasp_inference.py \
  --agent ~/rl_ws/hdgp/log/rl_games/open-tesol/right/grasp-v1/lstm_test3/params/agent.yaml \
  --ckpt  ~/rl_ws/hdgp/log/rl_games/open-tesol/right/grasp-v1/lstm_test3/nn/last_open-tesol_r_grasp_v1-lstm_ep_20000_rew_9920.256.pth
```

### B4. 에피소드 제어

```bash
ros2 service call /grasp/start std_srvs/srv/Trigger    # /grasp/stop, /grasp/reset 동일
```

**62909aa 방어 동작(정상):**
- start 시 손 자세가 APPROACH와 0.6rad 이상 어긋나면 **관절명을 찍고 거부**
  (= 죽은 드라이버 0.000 감지. 복구는 A3-1. 의도된 자세면 `--allow-hand-mismatch`).
- RUNNING 중 팔/손/컵 토픽 0.5s 두절 → `[RUNNING] 센서 두절` 로그 + **명령 홀드**.

---

## C. 트러블슈팅 요약

| 증상 | 원인/처방 |
|---|---|
| 팔이 컵에서 멀어지다 정지 | 손 obs zeros — start 게이트가 이제 차단. A3-1 손 전원 재인가 |
| discovery 되는데 데이터 불통 | FASTRTPS 프로파일 프로세스별 불일치 — §0 env 확인 |
| goal success인데 팔 무동작 | JTC interpolation none + 미래 tfs — 브리지 최신판(tfs=0) 확인 |
| 손 관절 전부 0.000 고정 | Modbus 세션 꼬임 — 손 전원 재인가 후 벤더 스크립트부터 |
| start 거부(미수신 토픽) | 해당 토픽 발행 노드/DDS 확인 (`ros2 topic hz <topic>`) |
| start 거부(손 자세 어긋남) | 정상 방어 — 손 복구 또는 `--allow-hand-mismatch` |

## D. 오프라인 진단 도구 (로컬 pc5090)

```bash
# 폐루프 재현기(정책+Fabrics+mock팔): max_vel/손모드/지연 ablation
/home/user/rl_ws/IsaacLab/isaaclab.sh -p /home/user/rl_ws/sim2real/scripts/grasp_loop_sim.py \
  --agent <agent.yaml> --ckpt <ckpt.pth> --max-vel 0.1 --hand-mode static|zero|echo [--obs-delay 6]

# Isaac 죽은 손 probe
cd /home/user/rl_ws/hdgp && /home/user/rl_ws/IsaacLab/isaaclab.sh \
  -p scripts/reinforcement_learning/rl_games/play.py --task open-tesol_r_grasp_v1-play-lstm \
  --checkpoint <ckpt> --num_envs 32 --headless --dead_hand_probe
```
