#!/usr/bin/env python3
"""grasp_s2r sim 에서 **실기와 같은 궤적**을 재생해 게인을 검증한다 (real2sim A단계).

08.31 실기에서 얻은 것: 중력보상을 켠 우팔이 preset 궤적을 추종오차 **RMSE 0.94°**
로 따라갔다. sim 이 그 거동을 재현해야 "sim 에서 검증한 것이 실기에서도 성립한다"고
말할 수 있다.

★이 자산은 **로봇 중력이 꺼져 있다**(`grasp_s2r_env_cfg.py:118`). 실기에 중력보상을
  켠 것이 정확히 그 조건을 재현한 것이므로, 남은 차이는 **게인뿐**이다.

  현재 sim (KUKA)  : kp 300/100/50/25 · kd 45/20/15/15
  실기 실측        : kp 73.1/60.9/11.9 · kd 6.376/5.635/2.154 (07.29 계단+autotune)

`HDGP_S2R_REAL_GAINS=1` 로 후자를 켜고 같은 궤적을 돌려, 실기 bag 과 어느 쪽이
가까운지 본다. **정책은 돌리지 않는다** — 관절 지령만 흘려보내는 순수 재생이다.

    # KUKA 게인(현행)
    ./isaaclab.sh -p probe_s2r_gain_replay.py --npz .../reset_right_v2.npz \\
        --out logs/shadow/sim_gain_kuka.npz
    # 실측 게인
    HDGP_S2R_REAL_GAINS=1 ./isaaclab.sh -p probe_s2r_gain_replay.py ... \\
        --out logs/shadow/sim_gain_real.npz

    # 비교 (실기 bag 과)
    python3 compare_sim_real_gains.py --sim-kuka ... --sim-real ... --bag ...
"""

from __future__ import annotations

import argparse
import os as _os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="open-sens_r_grasp_s2r-play")
parser.add_argument("--npz", type=Path, required=True,
                    help="재생할 궤적(arm_target 을 읽는다). reset_right_v2.npz 등")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--rate_scale", type=float, default=0.5,
                    help="실기 재생과 같은 배속. 기록의 step_dt 를 이 값으로 나눈다.")
parser.add_argument("--hdgp_root", type=Path, default=Path("/home/user/rl_ws/hdgp"))
parser.add_argument("--fabrics_src", type=Path, default=None)
parser.add_argument("--gui", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, _unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + _unknown

sys.path.insert(0, str(args.hdgp_root / "source/openarm"))
if args.fabrics_src is not None:
    sys.path.insert(0, str(args.fabrics_src.resolve()))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

import fabrics_sim                                                # noqa: E402,F401
import gymnasium as gym                                           # noqa: E402,F401
import openarm.tasks                                              # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg                    # noqa: E402

RIGHT_ARM = [f"r_aj_{i}" for i in range(1, 8)]


def main() -> int:
    real_gains = _os.environ.get("HDGP_S2R_REAL_GAINS") == "1"
    print(f"[게인] {'실측(73.1/60.9/11.9)' if real_gains else 'KUKA(300/100/50/25)'}"
          f"  — HDGP_S2R_REAL_GAINS={_os.environ.get('HDGP_S2R_REAL_GAINS', '미설정')}")

    data = np.load(args.npz, allow_pickle=False)
    names = [str(x) for x in data["meta_joint_names"]]
    if tuple(names) != tuple(RIGHT_ARM):
        raise SystemExit(f"기록의 관절이 우팔이 아니다: {names}")
    target = data["arm_target"][:, 0].astype(np.float32)
    step_dt = float(data["meta_step_dt"][0])
    hand_cmd = data["grip_cmd"][:, 0].astype(np.float32) if "grip_cmd" in data else None
    print(f"[궤적] {args.npz.name} · {target.shape[0]} 프레임 · dt {step_dt:.4f} "
          f"· 배속 {args.rate_scale}")

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1e6
    # grasp_s2r 는 direct env 라 manager 기반 terminations 가 없다. 에피소드가 끝나
    # 리셋되면 재생이 오염되므로 길이만 늘려 둔다(위의 episode_length_s).
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    env.reset()
    dec = int(env.cfg.decimation)
    jn = robot.joint_names
    arm = [jn.index(n) for n in RIGHT_ARM]
    hand_names = [n for n in jn if n.startswith("r_hj_")]
    hand = [jn.index(n) for n in hand_names]

    # 실제 적용된 게인을 **읽어서** 찍는다 — cfg 를 믿지 않는다(그룹 이름이 어긋나면
    # 조용히 기본값으로 떨어진다는 것이 이 저장소의 기존 교훈이다).
    kp = robot.data.joint_stiffness[0, arm].cpu().numpy()
    kd = robot.data.joint_damping[0, arm].cpu().numpy()
    print("[게인 실측] kp " + " ".join(f"{v:.1f}" for v in kp))
    print("[게인 실측] kd " + " ".join(f"{v:.2f}" for v in kd))

    # 시작 자세 = 궤적 첫 프레임 (실기도 램프로 거기서 시작한다)
    full = robot.data.joint_pos[0].clone()
    for k, i in enumerate(arm):
        full[i] = float(target[0, k])
    if hand_cmd is not None and hand:
        for k, i in enumerate(hand[: hand_cmd.shape[1]]):
            full[i] = float(hand_cmd[0, k])
    robot.write_joint_state_to_sim(full.unsqueeze(0), torch.zeros_like(full).unsqueeze(0))
    robot.set_joint_position_target(full.unsqueeze(0))
    robot.write_data_to_sim()
    for _ in range(40):
        env.sim.step(render=args.gui)
    env.scene.update(env.sim.get_physics_dt())

    # ★실기 재생은 step_dt/rate_scale 주기로 지령을 갱신했다. sim 도 같은 시간축으로
    #   재생해야 비교가 성립한다 — sim 스텝이 더 잘면 한 지령을 여러 번 적용한다.
    hold = max(1, int(round((step_dt / args.rate_scale) / env.step_dt)))
    print(f"[재생] 지령 1개당 sim {hold} 스텝 (sim dt {env.step_dt:.4f})")

    cmd_log, meas_log = [], []
    for f in range(target.shape[0]):
        q = torch.tensor(target[f], device=env.device, dtype=torch.float32)
        for _ in range(hold):
            robot.set_joint_position_target(q.unsqueeze(0), joint_ids=arm)
            if hand_cmd is not None and hand:
                h = torch.tensor(hand_cmd[f][: len(hand)], device=env.device,
                                 dtype=torch.float32)
                robot.set_joint_position_target(h.unsqueeze(0), joint_ids=hand[: h.shape[0]])
            robot.write_data_to_sim()
            for _ in range(dec):
                env.sim.step(render=args.gui)
            env.scene.update(env.sim.get_physics_dt())
        cmd_log.append(target[f].copy())
        meas_log.append(robot.data.joint_pos[0, arm].cpu().numpy().copy())
        if f % 300 == 0:
            err = np.degrees(np.abs(meas_log[-1] - cmd_log[-1])).max()
            print(f"  {f}/{target.shape[0]}  최대오차 {err:5.2f}°", flush=True)

    cmd = np.stack(cmd_log)
    meas = np.stack(meas_log)
    err = meas - cmd
    print("\n═══ sim 추종오차 (지령 대비) ═══")
    print(f"{'관절':10s} {'mean':>9s} {'RMSE':>9s} {'max':>9s}")
    for k, n in enumerate(RIGHT_ARM):
        e = err[:, k]
        print(f"{n:10s} {np.degrees(e.mean()):+8.2f}° "
              f"{np.degrees(np.sqrt((e**2).mean())):8.2f}° "
              f"{np.degrees(np.abs(e).max()):8.2f}°")
    print(f"{'전체':10s} {np.degrees(err.mean()):+8.2f}° "
          f"{np.degrees(np.sqrt((err**2).mean())):8.2f}° "
          f"{np.degrees(np.abs(err).max()):8.2f}°")
    print("\n[실기 참고] 중력보상 ON preset 재생: mean −0.16° · RMSE 0.94° · max 2.19°")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, arm_target=cmd[:, None, :].astype(np.float32),
        q_meas=meas.astype(np.float32),
        meta_joint_names=np.array(RIGHT_ARM),
        meta_step_dt=np.array([step_dt / args.rate_scale]),
        meta_kp=kp.astype(np.float32), meta_kd=kd.astype(np.float32),
        meta_real_gains=np.array([1 if real_gains else 0]),
        meta_source=np.array([str(args.npz)]))
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
