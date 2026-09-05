#!/usr/bin/env python3
"""실기 손 응답을 sim 에서 같은 지령으로 재생해 **맞물리는지** 본다.

09.01 실기(p=4.5·d=0)에서 주먹 램프 4 s 를 주면 13관절이 도달률 98~101 %,
정상상태 오차 평균 0.39° 로 따라온다. sim(`kp 5.0 · kd 2.0`)이 같은 지령에 같은
응답을 내야 정책이 배운 손 동작이 실기에서 재현된다.

★손은 팔과 달리 **실기 게인을 바꿀 수 있으므로**(JTC PID 를 `ros2 param set`),
  둘 중 어느 쪽을 움직여도 된다. 이 도구는 **현재 sim 이 실기와 얼마나 다른지**를
  재서 그 결정의 근거를 준다.

    ./isaaclab.sh -p probe_hand_sim_replay.py --npz .../hand_resp_p4.5_d0.npz \\
        --out .../sim_hand_resp.npz --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="open-sens_r_grasp_s2r-play")
parser.add_argument("--npz", type=Path, required=True, help="probe_hand_multi_gain --record 산출물")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--profile_yaml", type=Path,
                    default=Path("/home/user/rl_ws/robot_control/src/robot_control/"
                                 "profiles/openarm_tesollo.yaml"))
parser.add_argument("--hdgp_root", type=Path, default=Path("/home/user/rl_ws/hdgp"))
parser.add_argument("--settle", type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args, _unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + _unknown

sys.path.insert(0, str(args.hdgp_root / "source/openarm"))
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402
import yaml                                                       # noqa: E402

import fabrics_sim                                                # noqa: E402,F401
import gymnasium as gym                                           # noqa: E402,F401
import openarm.tasks                                              # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg                    # noqa: E402


def main() -> int:
    data = np.load(args.npz, allow_pickle=False)
    src_names = [str(x) for x in data["joint_names"]]
    cmd = data["command"].astype(np.float64)
    meas = data["measured"].astype(np.float64)
    dt_real = float(data["dt"][0])
    print(f"[실기] {args.npz.name} · {len(cmd)} 프레임 @ {1/dt_real:.0f} Hz "
          f"· p={float(data['gain_p'][0])} d={float(data['gain_d'][0])}")

    # 드라이버 이름(rj_dg_*) → sim 이름(r_hj_*)
    body = yaml.safe_load(args.profile_yaml.read_text())
    to_sim = {j["source"]: (j["canonical"], j.get("sign", 1)) for j in body["joints"]}

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1e6
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    env.reset()
    physics_dt = float(env.sim.get_physics_dt())
    jn = robot.joint_names

    pairs = []                                    # (기록 열, sim 관절 index, sign)
    for k, src in enumerate(src_names):
        if src not in to_sim:
            continue
        canonical, sign = to_sim[src]
        if canonical in jn:
            pairs.append((k, jn.index(canonical), sign))
    print(f"[sim] physics_dt {physics_dt*1000:.2f} ms · 매칭 {len(pairs)}/{len(src_names)} 관절")
    ids = torch.tensor([i for _, i, _ in pairs], device=env.device)
    kp = robot.data.joint_stiffness[0, ids].cpu().numpy()
    kd = robot.data.joint_damping[0, ids].cpu().numpy()
    print(f"[sim 게인] kp {kp.min():.1f}~{kp.max():.1f} · kd {kd.min():.2f}~{kd.max():.2f}")

    # 실기 50 Hz → 물리 dt 격자로 리샘플
    t_real = np.arange(len(cmd)) * dt_real
    grid = np.arange(0.0, t_real[-1], physics_dt)
    target = np.stack([np.interp(grid, t_real, cmd[:, k]) * sign
                       for k, _, sign in pairs], axis=1)

    full = robot.data.joint_pos[0].clone()
    for col, (_, idx, _) in enumerate(pairs):
        full[idx] = float(target[0, col])
    robot.write_joint_state_to_sim(full.unsqueeze(0), torch.zeros_like(full).unsqueeze(0))
    robot.set_joint_position_target(full.unsqueeze(0))
    robot.write_data_to_sim()
    for _ in range(args.settle):
        env.sim.step(render=False)
    env.scene.update(physics_dt)

    plan = torch.tensor(target, device=env.device, dtype=torch.float32)
    trace = torch.zeros((len(grid), len(pairs)), device=env.device, dtype=torch.float32)
    for step in range(len(grid)):
        robot.set_joint_position_target(plan[step].unsqueeze(0), joint_ids=ids)
        robot.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(physics_dt)
        trace[step] = robot.data.joint_pos[0, ids]
        if step % 200 == 0:
            print(f"  {step}/{len(grid)}", flush=True)
    sim = trace.cpu().numpy().astype(np.float64)

    print(f"\n═══ 최종 도달률·정상오차 (실기 vs sim) ═══")
    print(f"{'관절':10s} {'지령':>8s} {'실기':>8s} {'sim':>8s} "
          f"{'실기오차':>8s} {'sim오차':>8s}")
    real_err, sim_err = [], []
    for col, (k, _, sign) in enumerate(pairs):
        goal = cmd[-1, k] * sign
        if abs(goal) < 0.05:
            continue
        r, s = meas[-1, k] * sign, sim[-1, col]
        re_, se = np.degrees(goal - r), np.degrees(goal - s)
        real_err.append(abs(re_))
        sim_err.append(abs(se))
        print(f"{src_names[k].replace('rj_dg_',''):10s} {np.degrees(goal):+7.1f}° "
              f"{np.degrees(r):+7.1f}° {np.degrees(s):+7.1f}° {re_:+7.2f}° {se:+7.2f}°")
    print(f"\n정상상태 오차  실기 평균 {np.mean(real_err):.2f}° / 최대 {np.max(real_err):.2f}°")
    print(f"               sim  평균 {np.mean(sim_err):.2f}° / 최대 {np.max(sim_err):.2f}°")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, command=target.astype(np.float32), q_sim=sim.astype(np.float32),
        real_measured=meas.astype(np.float32), joint_names=np.array(src_names),
        physics_dt=np.array([physics_dt]), meta_kp=kp.astype(np.float32),
        meta_kd=kd.astype(np.float32), source=np.array([str(args.npz)]))
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
