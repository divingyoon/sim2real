# 런북 — 좌 그리퍼 그림자 추종 (카메라·물체 없음)

**무엇을 재는가.** 두 가지뿐이다.
1. 실팔이 정책 액션이 만든 관절 목표를 따라가는가 (지연·처짐·오버슛)
2. Fabrics IK 가 제대로 도는가 (L1/L2 는 sim 에서 이미 나왔다 — 아래 §0)

**무엇을 재지 않는가.** 정책의 과제 성능. 물체가 없고, 관측은 sim 에서만 온다.

**구조.** Isaac 이 정책을 굴려 관절 목표를 파일로 남기고(이미 완료), 실기는 그 파일을
따라가기만 한다. 정책이 실기 관측으로 돌지 않으므로 발산할 여지가 없고, 이상하면 그냥
멈추면 된다. `Ctrl-C` 가 곧 중단이다.

이 문서는 **로봇 PC(5070ti)에서 사람이 실행**하는 절차다. 그 머신은 이 세션에서 접근할 수
없다(`tailnet policy does not permit`).

---

## 0. 이미 나온 것 (실기 없이)

| | 값 | 뜻 |
|---|---|---|
| L1 FK vs 지령 | mean 11.9 mm · 정상구간 **3.6 mm** | Fabrics 는 도달 가능한 목표를 정확히 실현한다 |
| L1 꼬리 | max 402 mm | PALM_BOX 먼 꼭짓점은 **도달 불가**(최대 200 mm) — 워크스페이스 사실 |
| L2 물리 vs FK | mean **3.46 mm** | sim PD(kp400)는 fabric 해를 잘 따라간다 |
| 요구 관절속도 | mean < 0.1 · **첨두 3.93 rad/s** | 프로필 한계 2.0 의 **2배** — 1배속 재생 불가 |
| 좌팔 예측 처짐 | **53 mm** @펌웨어 게인 | 우팔(129~142 mm)의 절반 이하지만 무시할 크기가 아니다 |
| 좌팔 예측 ζ | j1~j4 **0.26~0.40** | 크게 부족감쇠 — 오버슛·링잉을 예상하고 볼 것 |

근거·재현: `docs/measure/S2R_INTERFACE_EQUIVALENCE.md` §6-2, §6-3.

---

## 0-1. 기록을 다시 만들려면 (Isaac 필요)

FABRICS 를 학습 시점으로 고정해야 한다. 저장소 사본을 쓰면 `openarm.tasks` 가 그것을
`sys.path[0]` 에 꽂아 PYTHONPATH 를 이기고, 두 트리는 같은 목표에서 관절 해가 최대
0.32 rad 갈린다(08.25 실측). 그래서 프로브에 `--fabrics_src` 를 준다.

```bash
cd ~/rl_ws/hdgp
git worktree add --detach /tmp/hdgp_bc86ca5 bc86ca5          # 학습 시점 FABRICS
../IsaacLab/isaaclab.sh -p scripts/probes/probe_fab_shadow_record.py \
    --checkpoint log/rl_games/open-grip/left/grasp-sensor-fab/fab_test16/nn/open-grip_l_grasp_sensor_fab.pth \
    --fabrics_src /tmp/hdgp_bc86ca5/source/FABRICS/src \
    --steps 1200 --num_envs 1 --out logs/shadow/sim_fab_test16_gcON.npz
```

`--gravity_comp off` 로 한 벌 더 뜨면 처짐 보상 유무를 실기에서 비교할 수 있다.

## 1. 사전 점검 (하드웨어 켜기 전)

```bash
cd ~/rl_ws/sim2real
source /opt/ros/humble/setup.bash && . .venv/bin/activate
python3 -m pytest scripts/ -q                       # 계약·브리지·재생 코어
python3 -m pytest ~/rl_ws/robot_control/tests -q     # 프로필·안전 게이트
```

기록 파일이 어떤 코드로 나왔는지 확인한다 — 이 트랙은 소스가 자주 바뀐다:

```bash
python3 -c "
import numpy as np; d=np.load('logs/shadow/sim_fab_test16_gcON.npz')
print(*d['meta_task_sha256'], sep='\n'); print(d['meta_fabrics'][0])"
```

## 2. 재생 계획 확인 (하드웨어 불필요, 무발행)

```bash
python3 scripts/shadow_replay.py --sim logs/shadow/sim_fab_test16_gcON.npz \
    --robot gripper_left --rate-scale 0.25
```

`--execute` 가 없으므로 **아무것도 발행하지 않는다**(rclpy import 자체가 그 뒤에 있다).
출력의 "요구 최대 관절속도"가 프로필 한계 아래인지 본다. 1.0 은 거부되고 **0.5 가 상한**이다.

## 3. fake_hardware 통과 (배선·순서·차원)

```bash
# 터미널 A
ros2 launch openarm_bringup openarm.bimanual.launch.py use_fake_hardware:=true
# 터미널 B — 관절 대시보드
python3 scripts/joint_monitor.py --arm-topic /joint_states
# 터미널 C
python3 scripts/shadow_replay.py --sim logs/shadow/sim_fab_test16_gcON.npz \
    --robot gripper_left --rate-scale 0.25 --frames 200 \
    --log logs/shadow/fake_x025.csv --execute
```

⚠ **이 단계는 추종 검증이 아니다.** mock 하드웨어는 droop 이 없어 무엇이든 완벽히
추종하므로 서보 rate-limit 버그를 숨긴다(`robot_control/src/robot_control/safety.py:134`).
여기서 보는 것은 관절이 **올바른 순서로 올바른 방향으로** 움직이는가뿐이다.

## 4. 실기 정지 검증 (이동 최소)

```bash
# ⚠ use_fake_hardware:=false 필수 — 기본값이 true 다
ros2 launch openarm_bringup openarm.bimanual.launch.py use_fake_hardware:=false \
    left_can_interface:=can1 right_can_interface:=can0

robotctl pose ready --group openarm_left_arm --execute     # 2단계, 0.1 rad/s

python3 scripts/lowlevel_check.py --robot gripper_left --group arm --dry-run
python3 scripts/lowlevel_check.py --robot gripper_left --group arm --execute
```

- **TEST1 hold 드리프트** = 좌팔 실측 중력 처짐. §6-2 예측 **53 mm** 와 대조한다.
- **TEST2 단일관절 스텝** = 부호·배율·crosstalk, 그리고 **지연**(명령→측정 63 % 도달 시각).

⚠ `l_aj_7` 은 effort 한계 7 N·m 이고 07.27 에 로터 과열(0xC)로 래치된 적이 있다. 드라이버가
에러 니블을 버리므로 `/joint_states` 에는 절대 안 나온다 — 의심되면
`python3 ~/rl_ws/robot_control/tools/read_motor_error.py` 로 raw CAN 을 본다.

## 5. 실기 그림자 재생

느린 것부터. 각 회차 사이에 팔을 다시 `pose ready` 로 돌린다.

```bash
for R in 0.10 0.25 0.50; do
  python3 scripts/shadow_replay.py --sim logs/shadow/sim_fab_test16_gcON.npz \
      --robot gripper_left --rate-scale $R --max-vel 0.5 \
      --log logs/shadow/real_x${R}.csv --execute
done
```

**중단 조건은 코드에 박혀 있다** — 관절 추종오차 0.3 rad · j5~7 effort 5 N·m ·
상태 두절 1 s. 걸리면 즉시 멈추고 이유를 찍는다. 그 밖에도 이상하면 `Ctrl-C`.

그리퍼는 팔이 통과한 뒤 **별도 회차**로 낸다. (현 기록에서 그리퍼는 계속 열린 상태
0.044 m 라 개폐가 없다 — 개폐를 보려면 하드 게이트가 열린 롤아웃을 따로 기록해야 한다.)

## 6. 판정

```bash
for R in 0.10 0.25 0.50; do
  python3 scripts/shadow_report.py --sim logs/shadow/sim_fab_test16_gcON.npz \
      --real logs/shadow/real_x${R}.csv --out logs/shadow/report_x${R}.md
done
```

읽는 법:
- **지연** — 교차상관 최대점. 전송+제어 지체의 합이다. TEST2 계단값과 대조하면 둘이 갈린다.
- **정렬 후 RMSE** — 지연을 걷어낸 뒤 남는 오차 = **정적 성분(중력 처짐)**.
- **정렬로 사라진 몫** = **대역폭 성분**.
- `--rate-scale` 스윕이 능력–요구 곡선이다. 느린 쪽에서 붙고 빠른 쪽에서 벌어지면
  대역폭 문제, 느린 쪽에서도 오프셋이 남으면 처짐 문제. **고치는 노브가 다르다.**

결과는 `docs/measure/S2R_INTERFACE_EQUIVALENCE.md` 의 L1~L7 · A1~A6 · T-2 · T-7 칸에
실측으로 채운다. 대응 선택지(A 브리지 max-vel · B 펌웨어 게인 · C 캘리브 주입 재학습 ·
D 정책 감속)는 **이 수치가 나온 뒤에** 고른다.
