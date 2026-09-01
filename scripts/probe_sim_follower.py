#!/usr/bin/env python3
"""Isaac **미러 로봇** — RViz 대신. UDP 로 받은 지령을 sim 로봇에 적용해 보여준다.

사슬:  shadow_replay/정책 ──ROS /isaacsim/*_cmd──▶ 실기 JTC
                              └─ ros_cmd_to_udp ──UDP──▶ **여기(sim)**

sim 과 실기가 **같은 지령**을 받으므로, GUI 에서 보이는 움직임 = 실기에 보낸 명령의
sim 실현이다. 관측/정책은 없다 — 이 프로세스는 순수 뷰어이며 물리로 지령을 실현할
뿐이다(kp 는 태스크 cfg 값이므로 sim 처짐도 그대로 보인다).

  · 패킷 v1 `<Id35f>`: magic 0x5A2B10 · t · 좌팔7 + 우팔7 + 좌그리퍼1 + 우손20
  · NaN 채널은 **건드리지 않는다**(마지막 목표 유지) — 없는 지령을 지어내지 않는다.
  · 시작 자세 = 차렷(전 관절 0, 그리퍼 개방) — 실기 시작 규약과 동일.
  · 좌그리퍼 1값은 두 조(l_hj_gripper_1,2)에 복사 — sim USD 가 mimic 을 잃어서다.

실행(★venv 없는 셸):
    cd ~/rl_ws/IsaacLab && ./isaaclab.sh -p ~/rl_ws/sim2real/scripts/probe_sim_follower.py
GUI 는 kill 할 때까지 유지된다(RViz 처럼 상시 켜 두는 용도).
"""

from __future__ import annotations

import argparse
import math
import socket
import struct
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="open-grip_l_grasp_sensor_fab-play")
parser.add_argument("--port", type=int, default=47321)
parser.add_argument("--hand_pose", type=Path,
                    default=Path("/home/user/rl_ws/sim2real/config/right_hand_fist.yaml"),
                    help="시작 손자세(실물 스냅샷). 실기가 bringup 주먹에 있으므로 sim 도 "
                         "거기서 출발해야 화면이 실기와 같다. 'none' 이면 전 관절 0.")
parser.add_argument("--hand_stiffness", type=float, default=200.0,
                    help="유휴 손 관절 강성. 태스크 기본값 20 은 주먹을 못 쥐고 늘어진다 "
                         "(08.31 실측 — 미러가 실기와 달라 보이는 원인).")
parser.add_argument("--hdgp_root", type=Path,
                    default=Path("/home/user/rl_ws/hdgp_t6x"),
                    help="태스크 소스 트리(읽기 전용 사용)")
# --headless 는 AppLauncher 가 추가한다(직접 넣으면 충돌). 기본은 GUI —
# 이 프로브의 존재 이유가 화면이다.
AppLauncher.add_app_launcher_args(parser)
args, _unknown = parser.parse_known_args()

sys.path.insert(0, str(args.hdgp_root / "source/openarm"))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

import fabrics_sim                                                # noqa: E402,F401
import gymnasium as gym                                           # noqa: E402,F401
import openarm.tasks                                              # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg                    # noqa: E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P   # noqa: E402

MAGIC = 0x5A2B10
FMT = "<Id35f"
PKT = struct.calcsize(FMT)

_FINGER_OF_INDEX = {1: "thumb", 2: "index", 3: "middle", 4: "ring", 5: "pinky"}


def _load_hand_pose(path, names):
    """실물 스냅샷(rj_dg_* 또는 r_hj_*) → sim 관절 순서 배열. 'none' 이면 0."""
    if path is None or str(path).lower() == "none" or not Path(path).is_file():
        return np.zeros(len(names), dtype=np.float32)
    raw = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        try:
            val = float(value)
        except ValueError:
            continue
        if key.startswith("rj_dg_"):
            f, j = key.split("_")[2:4]
            raw[f"r_hj_{_FINGER_OF_INDEX[int(f)]}_{j}"] = val
        elif key.startswith("r_hj_"):
            raw[key] = val
    missing = [n for n in names if n not in raw]
    if missing:
        print(f"[미러] ⚠ 스냅샷에 없는 손 관절 {missing} — 0 으로 둔다")
    return np.array([raw.get(n, 0.0) for n in names], dtype=np.float32)


LEFT_ARM = [f"l_aj_{i}" for i in range(1, 8)]
LEFT_GRIP = ["l_hj_gripper_1", "l_hj_gripper_2"]
RIGHT_ARM = [f"r_aj_{i}" for i in range(1, 8)]


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1e6
    if hasattr(env_cfg.terminations, "object_dropping"):
        env_cfg.terminations.object_dropping = None
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    env.reset()
    dec = int(env.cfg.decimation)
    jn = robot.joint_names
    l_arm = [jn.index(n) for n in LEFT_ARM]
    l_grip = [jn.index(n) for n in LEFT_GRIP]
    r_arm = [jn.index(n) for n in RIGHT_ARM]
    r_hand_names = [n for n in P.RIGHT_HAND_JOINT_NAMES if n in jn]
    r_hand = [jn.index(n) for n in r_hand_names]

    # 손을 붙잡을 수 있게 강성을 올린다. 태스크 기본 20 은 유휴 손을 대충 세우는 값이라
    # 주먹을 유지하지 못한다 — 미러가 실기와 달라 보이는 원인이었다(08.31).
    if r_hand and args.hand_stiffness > 0:
        k = torch.full((1, len(r_hand)), float(args.hand_stiffness), device=env.device)
        robot.write_joint_stiffness_to_sim(k, joint_ids=r_hand)
        robot.write_joint_damping_to_sim(k * 0.05, joint_ids=r_hand)
        print(f"[미러] 손 강성 {args.hand_stiffness:g} 로 상향 (기본값은 주먹을 못 쥔다)")

    # 시작 = 차렷 + 실물 주먹 (실기 규약)
    hand_start = _load_hand_pose(args.hand_pose, r_hand_names)
    full = robot.data.joint_pos[0].clone()
    for i in l_arm + r_arm:
        full[i] = 0.0
    for j, i in enumerate(r_hand):
        full[i] = float(hand_start[j])
    for i in l_grip:
        full[i] = P.GRIPPER_OPEN_POS
    robot.write_joint_state_to_sim(full.unsqueeze(0), torch.zeros_like(full).unsqueeze(0))
    robot.set_joint_position_target(full.unsqueeze(0))
    robot.write_data_to_sim()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.setblocking(False)
    print(f"[미러] UDP :{args.port} 대기 — 채널: 좌팔7+우팔7+좌그립1+우손20 "
          f"(NaN=유지) · GUI {'off' if getattr(args, 'headless', False) else 'on'}", flush=True)

    T = lambda v: torch.tensor(v, device=env.device, dtype=torch.float32)  # noqa: E731
    tgt = {"l": None, "r": None, "g": None,
           "h": T(hand_start.tolist()) if r_hand else None}
    n_pkt, last_report = 0, time.monotonic()
    render = not getattr(args, "headless", False)

    while simulation_app.is_running():
        # 소켓 비우기 — 마지막 패킷만 쓴다 (밀린 지령을 순서대로 재생하면 지연이 쌓인다)
        pkt = None
        while True:
            try:
                data, _ = sock.recvfrom(4096)
            except BlockingIOError:
                break
            if len(data) == PKT:
                pkt = data
        if pkt is not None:
            vals = struct.unpack(FMT, pkt)
            if vals[0] == MAGIC:
                body = vals[2:]
                l, r, g, h = body[0:7], body[7:14], body[14], body[15:35]
                if not math.isnan(l[0]):
                    tgt["l"] = T(l)
                if not math.isnan(r[0]):
                    tgt["r"] = T(r)
                if not math.isnan(g):
                    tgt["g"] = T([g] * len(l_grip))
                if not math.isnan(h[0]):
                    tgt["h"] = T(list(h)[: len(r_hand)])
                n_pkt += 1

        if tgt["l"] is not None:
            robot.set_joint_position_target(tgt["l"].unsqueeze(0), joint_ids=l_arm)
        if tgt["r"] is not None:
            robot.set_joint_position_target(tgt["r"].unsqueeze(0), joint_ids=r_arm)
        if tgt["g"] is not None:
            robot.set_joint_position_target(tgt["g"].unsqueeze(0), joint_ids=l_grip)
        if tgt["h"] is not None and r_hand:
            robot.set_joint_position_target(tgt["h"].unsqueeze(0), joint_ids=r_hand)
        robot.write_data_to_sim()
        for _ in range(dec):
            env.sim.step(render=render)
        env.scene.update(env.sim.get_physics_dt())

        now = time.monotonic()
        if now - last_report > 10.0:
            last_report = now
            q_l = robot.data.joint_pos[0, l_arm].cpu().numpy()
            q_r = robot.data.joint_pos[0, r_arm].cpu().numpy()
            q_h = (robot.data.joint_pos[0, r_hand].cpu().numpy() if r_hand
                   else np.zeros(0))
            err_h = (float(np.abs(q_h - hand_start).max()) if r_hand else 0.0)
            print(f"[미러] 패킷 {n_pkt} · L {np.round(q_l, 3)} · R {np.round(q_r, 3)}"
                  f" · 손 시작자세대비 {np.degrees(err_h):.1f}°", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
