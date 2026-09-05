#!/usr/bin/env python3
"""좌 v2 정책 **발행 노드** — Isaac 없이 실기를 돈다.

    실기 /joint_states ─┐
    물체 pose (UDP)    ─┴─▶ core(obs 49D → 정책 → palm → fabric) ─▶ JTC ─▶ 실기

`left_inference_dryrun.py` 와 같은 코어를 쓰되 **발행한다**. 그래서 안전장치가 붙는다.

지키는 것 (드라이런과 shadow_replay 의 규약을 합친 것)
  · `--execute` 없으면 **아무것도 발행하지 않는다**.
  · 첫 지령으로 도약하지 않는다 — 실측에서 목표까지 `--max-vel` 로 서브스텝을 낸다.
  · 속도 제한은 **시간 스케일링**(서브스텝)이다. 위치 클램프는 신호를 잘라 잘린 몫이
    누적된다(09.02 실측: 0.65 rad 누적 → 그림자 가드 중단).
  · 매 스텝 **발행 전에** 검사하고, 하나라도 걸리면 즉시 멈춘다:
      관절한계 · TCP 테이블 여유 · 추종오차 · 센서 두절
  · 정지할 때 마지막 지령을 다시 내지 않는다 — JTC 가 마지막 점을 홀딩한다.
  · `time_from_start=0` — 컨트롤러가 interpolation_method="none" 이라 미래 시각
    포인트는 영영 적용되지 않고 로봇이 무경고로 안 움직인다.

실행:
    python3 left_inference_node.py --steps 60 --max-vel 0.3            # 발행 없음
    python3 left_inference_node.py --steps 60 --max-vel 0.3 --execute  # ★실기 동작
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np


SIM2REAL = Path(__file__).resolve().parents[2]
URDF = Path("/home/user/rl_ws/urdf/generated/rl/openarm_tesollo_sensor_rl.urdf")
# ★상면 높이는 **보수적으로 높은 쪽**을 쓴다. 실기 테이블 = 0.205(09.05 줄자·Fusion CAD
#   정정, sim env_v1 도 0.205). 가드는 +10 mm 여유를 둔 0.215. 카메라 사슬이 +21 mm 높게
#   보는 문제(base→head 고정변환 datum, 마운트 0.750 의심 — B4 실측 대기)와는 별개다.
TABLE = dict(cx=0.375, cy=-0.165, sx=0.550, sy=0.970, top=0.215)
# ★학습 물체 z — `env.yaml` init_state 의 object pos z. 리셋 무작위화는 x,y 만 ±0.02 이고
#   z 범위는 (0.0, 0.0) 이다. 즉 정책은 이 높이 **하나**만 경험했다.
#   메시 바닥→원점 92.09 mm 이므로 상면 0.200 위에 선 컵의 원점이 정확히 이 값이다.
TRAIN_CUP_Z = 0.29209

# ★★학습 플랜트의 **적분형 처짐 보상**(`grasp_left_fabric_action.apply_actions`).
#   sim 은 `droop += GAIN·(fabric_q − q)` 를 누적해 목표에 더한다 — 적분이라 정상상태
#   오차가 0 으로 수렴한다. 즉 **sim 팔은 fabric_q 에 실제로 도달한다.**
#   09.03: "sim 도 중력에 처지니 실기도 처져야 한다"고 보고 좌팔 중력보상을 껐는데,
#   sim 은 액션 항 안에서 이미 보상하고 있었다. 그래서 실기만 뒤처져 fabric 목표를
#   6~10° 밑돌았고, 정책이 학습 때 본 적 없는 상태를 받았다.
#   클램프는 `ARM_IK_MAX_TRACKING_ERROR`(= effort 한계/강성) — anti-windup 이다.
# ★그리퍼 과도 압착 제한. 학습은 닫기 지령이 **0.000 m** 이고 sim 은 effort 한계가
#   힘을 잡아주지만, 실기 JTC 는 위치 오차를 그대로 토크로 바꾼다 — 09.03 실측에서
#   턱이 0.026 m 에서 물체에 걸렸는데 지령은 0.000 이라 26 mm 어치를 계속 밀었다.
#   지령을 "현재 개구 − 여유" 로 깎으면, 걸리기 전에는 계속 닫히고 걸린 뒤에는
#   그 여유만큼만 밀어 힘이 상한을 갖는다. 개폐 판정(정책·게이트)은 건드리지 않는다.
CLOSE_OVERTRAVEL = 0.008     # m

DROOP_GAIN = 0.05
DROOP_LIMIT = np.array([0.1000, 0.1000, 0.0675, 0.0675, 0.0175, 0.0175, 0.0175])


def over_table(p) -> bool:
    return (abs(p[0] - TABLE["cx"]) <= TABLE["sx"] / 2
            and abs(p[1] - TABLE["cy"]) <= TABLE["sy"] / 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, default=SIM2REAL / "logs/policy/left_v2E29")
    ap.add_argument("--stream", type=Path,
                    default=SIM2REAL / "logs/shadow/pour_entry/stream_left_v2b25.npz")
    ap.add_argument("--steps", type=int, default=60, help="정책 스텝 수")
    ap.add_argument("--policy-hz", type=float, default=50.0,
                    help="정책 결정 주기 (학습은 50 Hz = decimation 2 × dt 0.01)")
    ap.add_argument("--fabric-hz", type=float, default=100.0,
                    help="fabric 적분·발행 주기 (학습은 100 Hz). 정책과 **독립**이다")
    ap.add_argument("--max-vel", type=float, default=1.5,
                    help="관절목표 속도 상한 rad/s — 넘으면 그 스텝을 버리지 않고 **정지**한다")
    ap.add_argument("--return-home", action="store_true", default=True,
                    help="가드 정지 시 홈으로 자동 복귀(기본 켬)")
    ap.add_argument("--no-return-home", dest="return_home", action="store_false",
                    help="자동 복귀를 끈다 — 정지 자세를 그대로 보고 싶을 때")
    ap.add_argument("--return-rate", type=float, default=0.12,
                    help="복귀 램프 관절속도 rad/s")
    ap.add_argument("--return-hz", type=float, default=50.0,
                    help="복귀 램프 발행 Hz")
    ap.add_argument("--settle-tol", type=float, default=0.01,
                    help="정착 판정 rad — 실기가 홈에 이만큼 붙으면 정책 시작")
    ap.add_argument("--settle-clamp", type=float, default=0.12,
                    help="정착 선보상 상한 rad (정책 루프의 anti-windup 과 별개)")
    ap.add_argument("--settle-timeout", type=float, default=8.0,
                    help="정착 제한시간 s (0 이면 정착 없이 바로 시작)")
    ap.add_argument("--goal", default=None,
                    help="목표 x,y,z (기본: 런 명령상자 중심)")
    ap.add_argument("--cup-udp", type=int, default=46011)
    ap.add_argument("--cup-side", type=int, default=1)
    ap.add_argument("--cup-live", action="store_true",
                    help="물체 pose 를 매 tick 갱신한다(기본: 시작 시점 1회 고정)")
    ap.add_argument("--close-overtravel", type=float, default=CLOSE_OVERTRAVEL,
                    help="닫기 지령이 현재 개구보다 이만큼만 앞서게 한다(m). "
                         "0 이면 제한 없음(학습 그대로 0.000 을 낸다)")
    ap.add_argument("--cup-z", type=float, default=None,
                    help="물체 z 를 이 값으로 덮어쓴다. 학습은 z 무작위화가 **0** 이라 "
                         f"{TRAIN_CUP_Z} 하나만 보았다 — 인식 z 가 그와 다르면 정책은 "
                         "본 적 없는 입력을 받는다. ⚠ 실제 컵이 정말 그 높이에 "
                         "있을 때만 옳다(아니면 관측만 맞고 조준이 어긋난다)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--min-clearance", type=float, default=0.03,
                    help="TCP 테이블 여유 하한 m — 밑돌면 정지")
    ap.add_argument("--abort-tracking", type=float, default=0.5,
                    help="목표−실측 상한 rad — 넘으면 정지")
    ap.add_argument("--execute", action="store_true", help="★없으면 발행하지 않는다")
    args = ap.parse_args()

    sys.path.insert(0, str(SIM2REAL / "scripts"))

    from left_inference_dryrun import (goal_center_from_run, joint_limits,
                                       make_fabric, step_dt_from_run)
    from left_obs_builder import segments_from_run
    from left_grasp_gate import quat_to_matrix
    from left_policy_core import LeftPolicyCore, LeftSensors
    from policy_loader import RLGamesActorPolicy, RLGamesLstmActorPolicy

    env_yaml = args.run / "params/env.yaml"
    agent_yaml = args.run / "params/agent.yaml"
    ckpt = next((args.run / "nn").glob("*.pth"))
    # ★goal 은 **그 런의 명령 상자** 안이어야 한다 — 트랙마다 상자가 다르다.
    if args.goal:
        goal3 = [float(v) for v in args.goal.split(",")]
    else:
        goal3 = goal_center_from_run(env_yaml)
    goal7 = np.array(goal3 + [1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    import torch
    import yaml as _yaml
    is_rnn = "rnn" in _yaml.safe_load(agent_yaml.read_text())["params"]["network"]
    Loader = RLGamesLstmActorPolicy if is_rnn else RLGamesActorPolicy
    # ★obs 차원은 런 레이아웃이 정한다(v2 49D · fab 45D). 손으로 박으면 조용히 어긋난다.
    obs_dim = sum(d for _, d in segments_from_run(env_yaml))
    # ★액션 clip 은 **런의 계약**이다 — 학습 obs 의 last_action 이 그 값으로 잘린다.
    _acfg = _yaml.safe_load(agent_yaml.read_text())["params"]
    act_clip = float(_acfg.get("env", {}).get("clip_actions", 1.0))
    pol = Loader(str(agent_yaml), str(ckpt), obs_dim=obs_dim, action_dim=7,
                 device=args.device, action_clip=act_clip)

    def policy(obs):
        with torch.no_grad():
            a = pol.get_action(torch.as_tensor(obs, dtype=torch.float32,
                                               device=args.device).unsqueeze(0))
        return a[0].detach().cpu().numpy()

    core = LeftPolicyCore(policy=policy, fabric=None, run_env_yaml=env_yaml,
                          run_agent_yaml=agent_yaml, goal7=goal7, urdf_path=URDF)
    print(f"[left] 계약: obs {obs_dim}D · 대역 판 위 "
          f"{(core.gate.cfg.band_axis[0] + 0.09209) * 1000:.0f}~"
          f"{(core.gate.cfg.band_axis[1] + 0.09209) * 1000:.0f} mm · palm 회전 ±"
          f"{np.degrees(core.palm.cfg.max_pose_angle):.0f}° · act_clip "
          f"{act_clip:g} · goal "
          f"{np.round(goal7[:3], 3).tolist()}", flush=True)
    # ★fabric dt 는 학습 env 의 step_dt = sim dt × env decimation. dump 가 진실원천이다.
    env_step_dt = step_dt_from_run(env_yaml)
    fabric_block = make_fabric(core.home, args.device, env_step_dt=env_step_dt)
    n_sub = fabric_block.decimation
    core.fabric = lambda palm6, n=0: fabric_block(palm6, n)[-1]
    print(f"[left] fabric dt {env_step_dt:.4f} s × {n_sub} 서브스텝 "
          f"= 정책 스텝당 {env_step_dt * n_sub:.4f} s (학습과 동일)", flush=True)
    lo, hi = joint_limits()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    from jtc_bridge_core import JointRemap
    from robot_profile import load_robot_profile

    prof = load_robot_profile("gripper_left")
    src = list(prof.arm_source)
    # canonical(l_aj_*) → source(openarm_left_joint*) 부호·clamp. shadow_replay 와 동일 규약.
    remap = JointRemap(list(prof.arm_canonical), src, prof.joint_limits)

    rclpy.init()
    node = Node("left_inference")
    box: dict = {}

    def on_js(m):
        idx = {n: i for i, n in enumerate(m.name)}
        if not all(s in idx for s in src):
            return
        box["q"] = np.array([m.position[idx[s]] for s in src])
        box["qd"] = (np.array([m.velocity[idx[s]] for s in src])
                     if len(m.velocity) >= len(m.name) else np.zeros(7))
        g = idx.get("openarm_left_finger_joint1")
        box["g"] = float(m.position[g]) if g is not None else 0.044
        box["t"] = time.time()

    node.create_subscription(JointState, "/joint_states", on_js, qos_profile_sensor_data)
    pub = node.create_publisher(JointTrajectory, prof.topics["arm_traj"], 10)
    # ★그리퍼도 발행해야 한다. `LeftTick` 문서가 "arm_q_target 과 gripper_cmd 를
    #   발행한다"고 못박고 있는데 팔만 붙어 있어서, 09.03 실기에서 게이트가 열려도
    #   그리퍼가 끝내 움직이지 않았다.
    #   좌 그리퍼 컨트롤러는 **단일 관절**을 몬다(`ros2 param get … joints` 실측):
    #   나머지 한 짝은 mimic 이 따라온다.
    grip_pub = node.create_publisher(JointTrajectory, prof.topics["ee_traj"], 10)
    GRIP_JOINT = "openarm_left_finger_joint1"

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.cup_udp))
    sock.setblocking(False)

    def drain_cup(latch: bool = False) -> None:
        """UDP 큐를 비운다. `latch` 면 값을 갱신하지 않고 버린다.

        ★물체 pose 는 **정책 시작 시점의 한 값으로 고정**한다(사용자 규약).
          학습은 참값을 매 스텝 받지만, 실기의 인식 지터가 그대로 들어오면 팜
          목표가 매 스텝 흔들려 컵을 치고 간다. 큐는 계속 비워야 버퍼가 안 쌓인다.
        """
        try:
            while True:
                pkt, _ = sock.recvfrom(64)
                if latch or len(pkt) != 13:
                    continue
                sd, x, y, z = struct.unpack("<Bfff", pkt)
                if sd == args.cup_side:
                    box["cup"] = np.array([x, y, z], dtype=np.float64)
        except BlockingIOError:
            pass

    t0 = time.time()
    while time.time() - t0 < 10 and not ("q" in box and "cup" in box):
        rclpy.spin_once(node, timeout_sec=0.2)
        drain_cup()
    for k, what in (("q", "/joint_states"), ("cup", f"물체 UDP :{args.cup_udp}")):
        if k not in box:
            print(f"[left] {what} 수신 없음 — 중단", flush=True)
            return 1

    print(f"[left] {'★발행' if args.execute else '무발행(연습)'} · "
          f"체크포인트 {ckpt.name} · 물체 {np.round(box['cup'], 3).tolist()}", flush=True)
    print(f"[left] 상한: 속도 {args.max_vel} rad/s · TCP 여유 ≥{args.min_clearance*1000:.0f} mm"
          f" · 추종오차 ≤{args.abort_tracking} rad", flush=True)

    # ★★발행은 **정책 스텝당 한 번**이다. env 의 `apply_actions` 는 fabric 블록의
    #   최종 `_fabric_q` 를 관절목표로 쓰고 중간 서브스텝을 내보내지 않는다 —
    #   서브스텝은 fabric 내부 적분일 뿐 발행 단위가 아니다.
    #   저Hz 액션 사이를 메우는 건 그 뒤의 500~700 Hz PD 이고, sim 에서도 같은 목표를
    #   env decimation 만큼 유지한다. 09.03 에 이걸 "fabric 이 멈춘다"로 오해해
    #   주기 구조를 바꿨다가 `set_features` 호출 위치까지 어긋났다.
    n_fab_per_policy = 1
    fab_dt = 1.0 / args.policy_hz
    max_step = args.max_vel * fab_dt
    core.reset()
    rows, stop = [], None
    scale = env_step_dt * n_sub * args.policy_hz
    print(f"[left] 주기: 정책 = 발행 {args.policy_hz:.1f} Hz "
          f"(fabric 서브스텝 {n_sub} 회는 내부 적분) · PD 는 드라이버(500~700 Hz)",
          flush=True)
    print(f"[left] 시간축 {1/scale:.2f}배 느리게 재생(학습 정책주기 "
          f"{1/(env_step_dt*n_sub):.1f} Hz)", flush=True)

    def publish_grip(v: float) -> None:
        if not args.execute:
            return
        msg = JointTrajectory()
        msg.joint_names = [GRIP_JOINT]
        pt = JointTrajectoryPoint()
        pt.positions = [float(v)]
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 0
        msg.points = [pt]
        grip_pub.publish(msg)

    def publish(q_can: np.ndarray) -> None:
        if not args.execute:
            return
        msg = JointTrajectory()
        msg.joint_names = list(remap.output_source)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in remap.apply(list(q_can))]
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 0
        msg.points = [pt]
        pub.publish(msg)

    # ── 주 루프는 **fabric 주기**로 돈다. 정책은 n_fab_per_policy 마다 목표만 갱신하고,
    #    그 사이 fabric 이 유지된 목표로 계속 적분해 관절목표 스트림을 만든다.
    palm_target = None
    gate_open = False
    prev_tgt = None
    grip_cmd = None
    droop = np.zeros(7)          # ★적분형 처짐 보상 상태 (학습과 동일)
    # ★★파지 뒤에는 컵을 **그리퍼에 부착**해 따라가게 한다.
    #   게이트 해제 조건은 `lateral ≥ release_lateral(60 mm)` 인데, 컵 포즈를 시작
    #   시점에 고정해 두면 컵을 들어올리는 순간 "턱이 컵을 떠났다"로 읽혀 그리퍼가
    #   강제 개방되고 컵을 놓는다(09.03 실기: 45스텝 주기로 잡았다 놓기를 반복).
    #   학습에서는 컵 포즈가 참값이라 파지 중 같이 움직인다 — 그 물리를 복원한다.
    #   접근 구간은 여전히 고정값이므로 인식 지터는 들어오지 않는다.
    attach_local = None
    cup_latched = np.array(box["cup"], dtype=np.float64)   # ★시작 시점 1회 고정
    if args.cup_z is not None:
        print(f"[left] 물체 z 덮어쓰기 {cup_latched[2]:.3f} → {args.cup_z:.3f} "
              f"(학습값 {TRAIN_CUP_Z})", flush=True)
        cup_latched[2] = args.cup_z
    print(f"[left] 물체 pose 고정 {np.round(cup_latched, 3).tolist()}"
          f"{' (실시간 추종)' if args.cup_live else ' — 이후 갱신 없음'}", flush=True)

    # ── ★정책 시작 전 정착 — 실기를 **실제로 홈에 앉힌다** ──────────────
    #   sim 은 리셋에서 팔을 홈으로 텔레포트하므로 step 0 의 TCP 가 홈 그대로다
    #   (프로브 실측: 판 위 125 mm). 실기는 중력 처짐(j7 3.6~3.8°)만큼 낮게 시작해
    #   **TCP 가 52 mm 아래**(73 mm)에서 출발한다. 하강 속도는 sim 10 mm/step vs
    #   실기 13 mm/step 로 비슷하므로, 낮게 출발한 만큼 그대로 더 내려가 판에 닿는다.
    #   처짐 적분은 0 에서 쌓이는 데 ~0.4 s(20 스텝) 걸려 그 사이에 이미 내려간다.
    #   → 홈 목표로 적분을 **미리 채워** 실측이 홈에 붙은 뒤 정책을 시작한다.
    #   ⚠보상은 위로만 밀므로 정착 자체는 안전 방향이다.
    if args.settle_timeout > 0:
        t_settle = time.time()
        bias = np.zeros(7)
        err_home = float(np.abs(core.home - box["q"]).max())
        while time.time() - t_settle < args.settle_timeout:
            for _ in range(2):
                rclpy.spin_once(node, timeout_sec=0.002)
            err_home = float(np.abs(core.home - box["q"]).max())
            if err_home < args.settle_tol:
                break
            # ★정착 상한은 정책 루프의 anti-windup(`DROOP_LIMIT`, j7 1.0°)과 **다르다**.
            #   그 상한은 실제 처짐(3.6°)보다 작아 정착이 포화한다 — 정착은 제어 루프가
            #   아니라 **자세 잡기**이므로 넉넉히 준다.
            bias = np.clip(bias + DROOP_GAIN * (core.home - box["q"]),
                           -args.settle_clamp, args.settle_clamp)
            publish(core.home + bias)
            time.sleep(fab_dt)
        poses_s = core.fk.poses(box["q"], box["g"], box["g"])
        print(f"[left] 정착: 실기−홈 {np.degrees(err_home):.2f}° · TCP 판 위 "
              f"{(poses_s.tcp_pos[2] - 0.200) * 1000:.0f} mm (sim 리셋 125 mm) · "
              f"선보상 j7 {np.degrees(bias[6]):+.2f}°", flush=True)
        # ★넘길 때 적분을 **0 으로 되돌린다**. sim 도 리셋 직후 보상이 0 이고 팔은
        #   홈에 있다 — 그 상태에서 스텝이 진행되며 처지기 시작한다. 정착 선보상을
        #   그대로 들고 가면 실기만 sim 보다 높게 떠서 또 다른 괴리가 된다.
        droop = np.zeros(7)
    n_ticks = args.steps * n_fab_per_policy
    for tick in range(n_ticks):
        loop_start = time.time()
        for _ in range(2):
            rclpy.spin_once(node, timeout_sec=0.001)
        drain_cup(latch=not args.cup_live)
        if args.cup_live:
            cup_latched = np.array(box["cup"], dtype=np.float64)
            if args.cup_z is not None:
                cup_latched[2] = args.cup_z
        if time.time() - box.get("t", 0) > 0.5:
            stop = "센서 두절"
            break

        if tick % n_fab_per_policy == 0:          # ── 정책 결정 ──
            poses_now = core.fk.poses(box["q"], box["g"], box["g"])
            jaw_mid = 0.5 * (poses_now.finger_l_pos + poses_now.finger_r_pos)
            base_R = quat_to_matrix(poses_now.base_quat)
            if attach_local is not None:
                cup_now = jaw_mid + base_R @ attach_local
            else:
                cup_now = cup_latched
            sensors = LeftSensors(arm_q=box["q"], arm_qd=box["qd"], grip_q=box["g"],
                                  grip_qd=0.0, cup_pos=cup_now,
                                  cup_quat=np.array([1.0, 0.0, 0.0, 0.0]))
            # ★obs·게이트는 코어가 만들되, fabric 블록은 우리가 직접 받아 서브스텝을
            #   하나씩 발행한다. 코어의 `core.fabric` 은 블록의 마지막 줄을 주므로
            #   (env 가 다음 스텝에 쓰는 값과 같다) obs 일관성은 유지된다.
            out = core.step(sensors)
            palm_target = out.palm_target
            gate_open = out.gate_open
            grip_cmd = out.gripper_cmd
            # 게이트가 처음 열린 순간의 컵−턱 상대위치를 **그리퍼 좌표계**로 굳힌다.
            # 그래야 이후 손목이 돌아도 따라간다. 그리퍼가 다시 벌어지면 놓는다.
            if out.gate_open and attach_local is None:
                attach_local = base_R.T @ (cup_now - jaw_mid)
            elif not out.gate_open and attach_local is not None:
                attach_local = None
            tgt = out.arm_q_target
            # ★처짐 적분은 **정책 스텝당 한 번**만 (env 와 동일). 순간 오차를 그대로
            #   쓰면 가속 구간의 속도 지연까지 보상해 팔이 과격해진다.
            droop = np.clip(droop + DROOP_GAIN * (tgt - box["q"]),
                            -DROOP_LIMIT, DROOP_LIMIT)

        # ── 발행 전 검사 ────────────────────────────────────────────────
        if np.any(tgt < lo) or np.any(tgt > hi):
            stop = f"관절한계 밖 {np.round(np.degrees(tgt), 1).tolist()}"
            break
        if prev_tgt is not None and float(np.abs(tgt - prev_tgt).max()) > max_step:
            stop = (f"관절목표 속도 {float(np.abs(tgt-prev_tgt).max())/fab_dt:.2f} rad/s "
                    f"> 상한 {args.max_vel}")
            break
        poses = core.fk.poses(tgt, box["g"], box["g"])
        if over_table(poses.tcp_pos):
            clear = poses.tcp_pos[2] - TABLE["top"]
            if clear < args.min_clearance:
                stop = f"TCP 여유 {clear*1000:.0f} mm < 하한"
                break
        err = float(np.abs(box["q"] - tgt).max())
        if err > args.abort_tracking:
            stop = f"추종오차 {err:.2f} rad"
            break

        publish(tgt + droop)
        if grip_cmd is not None and tick % n_fab_per_policy == 0:
            g_out = grip_cmd
            if grip_cmd < box["g"]:      # 닫는 중 — 과도 압착만 깎는다
                g_out = max(grip_cmd, box["g"] - args.close_overtravel)
            publish_grip(g_out)
        prev_tgt = tgt.copy()
        rows.append((time.time(), tgt.copy(), np.asarray(palm_target).copy(),
                     gate_open, err, tick % n_fab_per_policy == 0,
                     np.asarray(box["q"], dtype=float).copy()))
        if tick % (n_fab_per_policy * 5) == 0:
            print(f"  [{tick // n_fab_per_policy:3d}] gate {str(gate_open):5s} · "
                  f"grip {grip_cmd:.3f}→{box['g']:.3f} · "
                  f"추종오차 {np.degrees(err):5.1f}° · palm "
                  f"{np.round(palm_target[:3], 3).tolist()}", flush=True)
        time.sleep(max(0.0, fab_dt - (time.time() - loop_start)))

    print(f"\n[left] {'정지: ' + stop if stop else '완료'} · {len(rows)}스텝", flush=True)

    # ── ★가드에 걸리면 **즉시 홈으로 복귀**한다(사용자 규약 09.03) ────────
    #   가드는 발행을 멈출 뿐이라, 팔이 판 코앞(실측 1~6 mm)에 그대로 선다.
    #   사람이 손댈 때까지 그 자세로 두면 위험하고, 다음 시도의 시작 조건도 오염된다.
    #   복귀는 **관절공간 직선 램프**이고, 매 프레임 TCP 가 판 위 여유를 지키는지
    #   확인한 뒤에만 발행한다 — 못 지키면 멈추고 사람을 부른다.
    if stop and args.execute and args.return_home:
        print("[left] 가드 정지 → 홈 복귀 시퀀스", flush=True)
        for _ in range(3):
            rclpy.spin_once(node, timeout_sec=0.05)
        q0 = np.asarray(box["q"], dtype=float)
        n_ramp = max(int(np.ceil(np.abs(core.home - q0).max()
                                 / args.return_rate * args.return_hz)), 1)
        blocked = None
        for i in range(1, n_ramp + 1):
            a = i / n_ramp
            q = q0 * (1.0 - a) + core.home * a
            pz = core.fk.poses(q, box["g"], box["g"]).tcp_pos
            if over_table(pz) and pz[2] - TABLE["top"] < args.min_clearance:
                blocked = f"복귀 경로 TCP 여유 {(pz[2]-TABLE['top'])*1000:.0f} mm"
                break
            publish(q)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(1.0 / args.return_hz)
        time.sleep(1.5)
        for _ in range(6):
            rclpy.spin_once(node, timeout_sec=0.05)
        pz = core.fk.poses(box["q"], box["g"], box["g"]).tcp_pos
        if blocked:
            print(f"[left] ★복귀 중단({blocked}) — 사람이 확인할 것", flush=True)
        else:
            print(f"[left] 복귀 완료 · 실기−홈 "
                  f"{np.degrees(np.abs(core.home - box['q']).max()):.2f}° · TCP 판 위 "
                  f"{(pz[2] - 0.200) * 1000:.0f} mm", flush=True)
    if rows:
        tg = np.array([r[1] for r in rows])
        errs = np.array([r[4] for r in rows])
        print(f"  추종오차 최대 {np.degrees(errs).max():.1f}° · 평균 "
              f"{np.degrees(errs).mean():.1f}°", flush=True)
        print(f"  게이트 열린 tick {sum(1 for r in rows if r[3])}/{len(rows)}", flush=True)
        ts = np.array([r[0] for r in rows])
        vel = np.abs(np.diff(tg, axis=0)).max(axis=1) / np.maximum(np.diff(ts), 1e-6)
        pol = [i for i, r in enumerate(rows) if r[5]]
        print(f"  발행 주기 중앙 {np.median(np.diff(ts))*1000:.0f} ms · "
              f"관절목표 속도 최대 {vel.max():.2f} rad/s", flush=True)
        if len(pol) > 1:
            print(f"  정책 결정 {len(pol)}회 · 주기 중앙 "
                  f"{np.median(np.diff(ts[pol]))*1000:.0f} ms", flush=True)
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            meas = np.array([r[6] for r in rows])
            palm = np.array([r[2][:3] for r in rows])
            pol_f = np.array([[1.0 if r[5] else 0.0] for r in rows])
            np.savetxt(
                args.csv,
                np.hstack([np.array([[r[0]] for r in rows]), tg, meas, palm, pol_f]),
                delimiter=",",
                header="t," + ",".join(f"q{i}" for i in range(1, 8)) + ","
                + ",".join(f"m{i}" for i in range(1, 8)) + ",px,py,pz,is_policy")
            print(f"  csv {args.csv}", flush=True)
    rclpy.shutdown()
    return 1 if stop else 0


if __name__ == "__main__":
    raise SystemExit(main())
