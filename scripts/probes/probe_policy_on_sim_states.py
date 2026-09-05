#!/usr/bin/env python3
"""sim 기록 상태를 **배포 파이프라인**에 먹여 정책 액션을 sim 액션과 대조한다.

## 왜 이게 결정적인가

실기 라운드에서 정책이 손을 안 닫는다(지시 폐쇄 0.32 vs sim 0.76 · `joint_err` 이
0.07 에 머물러 손가락이 한 번도 안 눌림). 원인 후보가 둘로 갈린다:

  ① **배포 파이프라인이 obs 를 잘못 만든다** → 정책이 다른 상태를 보고 다르게 군다
  ② **실기 상태가 sim 과 다르다**(컵 z·판 높이·접촉력) → 파이프라인은 맞다

이 프로브는 `hdgp/log/grasp_traj/g1/*.hdf5` 의 **sim 실측 상태**를 배포 코어에
그대로 주입한다. 그러면 ②가 제거되므로, 액션이 sim 과 같으면 ①도 아니고,
다르면 그 차이가 곧 파이프라인 결함이다.

★IsaacLab 이 필요 없다 — 기록만 읽는다.

## 두 모드

- `--teacher-force` (기본): 매 스텝 손 목표를 sim 의 `hand_q_cmd` 로 덮어써 obs 의
  `joint_err` 을 sim 과 같게 만든다. **관측 조립만** 시험한다.
- 끄면: 시너지가 자기 액션으로 굴러간다(배포와 동일). 누적 발산을 본다.

⚠`tip_force` 는 기록에 없다. 0 으로 넣는다 — sim 은 접촉 시 실제 힘을 본다.
  그래서 접촉 이후 구간의 차이는 이 항 때문일 수 있다(그 자체가 답이 된다).

실행:
    python3 probe_policy_on_sim_states.py \\
        --traj ../../hdgp/log/grasp_traj/g1/g1_y00.hdf5 --run logs/policy/right_g1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


SIM2REAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM2REAL / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traj", type=Path, required=True)
    ap.add_argument("--run", type=Path, default=SIM2REAL / "logs/policy/right_g1")
    ap.add_argument("--robot", default="tesollo_bi_s__right")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--object-dz", type=float, default=0.0,
                    help="물체 z 를 이만큼 옮겨 먹인다[m] — 실기는 sim 보다 +0.019")
    ap.add_argument("--object-dxy", default="0,0",
                    help="물체 x,y 편차[m]")
    ap.add_argument("--hand-scale", type=float, default=1.0,
                    help="손 관절 실측을 이 비율로 줄여 먹인다 — 실기 손이 sim 만큼 "
                         "안 닫히는 상황을 흉내낸다(1.0 = 그대로)")
    ap.add_argument("--free-run", action="store_true",
                    help="손 목표를 sim 으로 덮어쓰지 않고 시너지가 굴러가게 둔다")
    args = ap.parse_args()

    import h5py


    from grasp_s2r_core import GraspS2RCore, S2RSensors
    from grasp_s2r_fabric import make_right_fabric, permutation as fab_perm
    from grasp_s2r_obs_builder import hand_dof_order
    from policy_loader import RLGamesActorPolicy, RLGamesLstmActorPolicy
    from right_inference_node import _home_arm, _scalar
    from robot_profile import load_robot_profile

    prof = load_robot_profile(args.robot)
    env_yaml = args.run / "params/env.yaml"

    with h5py.File(args.traj, "r") as f:
        A = dict(f.attrs)
        ep = f[f"episodes/{sorted(f['episodes'].keys())[0]}"]
        sim = {k: ep[k][:].astype(float)
               for k in ("action", "arm_q", "arm_qd", "hand_q", "hand_qd",
                         "hand_q_cmd", "object_pose")}
    sim_hand_names = [str(x) for x in A["hand_joint_names"]]

    # ── 순열: sim 기록 순 → 배포 DOF 순 ─────────────────────────────────
    # ★배포의 `hand_q` 는 `find_joints` 반환 순(DOF 순)이다. 기록도 같은 순서로
    #   저장돼 있는지 **이름으로 확인**한다 — 다르면 20칸이 통째로 스크램블된다.
    dof_names = list(prof.ee_canonical)          # 프로필 순
    hand_prof_to_sim = np.array([sim_hand_names.index(n) for n in dof_names])
    # ★DOF 순은 빌더가 이름으로 준다 — 순열을 추정하지 않는다.
    _dof_order = list(hand_dof_order("r"))
    sim_to_dof = np.array([sim_hand_names.index(n) for n in _dof_order])

    def to_dof(arr_sim_order: np.ndarray) -> np.ndarray:
        """기록 순 (20,) → 배포 DOF 순 (20,)."""
        return arr_sim_order[sim_to_dof]

    # ── fabric + 코어 ──────────────────────────────────────────────────
    fab_dt = float(_scalar(env_yaml, "fabrics_dt", 0.016666666666666666))
    damping = float(_scalar(env_yaml, "fabrics_damping_gain", 10.0))
    home = np.zeros(27)
    home[:7] = np.array(_home_arm(env_yaml))
    fabric = make_right_fabric(home_q27=home, device=args.device,
                               dt=fab_dt, damping=damping)
    fab_names = fabric.joint_names()
    ee_src = [prof.joint_limits[n]["source"] for n in dof_names]
    fab_hand = [n for n in fab_names if n in set(ee_src)]
    dof_src = [prof.joint_limits[n]["source"] for n in _dof_order]
    dof_to_fab = fab_perm(dof_src, fab_hand)

    import torch
    import yaml as _yaml
    agent_yaml = args.run / "params/agent.yaml"
    acfg = _yaml.safe_load(agent_yaml.read_text())["params"]
    is_rnn = "rnn" in acfg["network"]
    Loader = RLGamesLstmActorPolicy if is_rnn else RLGamesActorPolicy
    pol = Loader(str(agent_yaml), str(next((args.run / "nn").glob("*.pth"))),
                 obs_dim=155, action_dim=21, device=args.device,
                 action_clip=float(acfg.get("env", {}).get("clip_actions", 1.0)))
    if is_rnn:
        pol.reset_states()

    def policy(obs):
        with torch.no_grad():
            a = pol.get_action(torch.as_tensor(obs, dtype=torch.float32,
                                               device=args.device).unsqueeze(0))
        return a[0].detach().cpu().numpy()

    _dxy = [float(v) for v in args.object_dxy.split(",")]
    _obj_off = np.array([_dxy[0], _dxy[1], args.object_dz])
    # ★목표도 물체에서 나온다(`goal = settled + offset`) — 물체를 옮기면 목표도 따라간다.
    goal = sim["object_pose"][0, :3] + _obj_off + np.array([0.0, 0.0, 0.12])
    core = GraspS2RCore(
        policy=policy, fabric_palm_pose=fabric.palm_pose, fabric_tips=fabric.tips,
        fabric_step=fabric.step, run_dir=args.run, goal3=goal,
        soft_limits=np.array([[prof.joint_limits[n]["lower"],
                               prof.joint_limits[n]["upper"]] for n in dof_names]),
        hand_dof_to_fabric=dof_to_fab)

    hq0 = to_dof(sim["hand_q"][0])
    core.reset(arm_q=sim["arm_q"][0], hand_q=hq0,
               object_pos=sim["object_pose"][0, :3] + _obj_off)

    n = min(args.steps, len(sim["action"]))
    rows = []
    for k in range(n):
        if not args.free_run:
            # ★teacher forcing — obs 의 `joint_err` 을 sim 과 같게 만든다.
            core.hand.target = sim["hand_q_cmd"][k][hand_prof_to_sim].copy()
        obj = sim["object_pose"][k, :3] + _obj_off
        out = core.step(S2RSensors(
            arm_q=sim["arm_q"][k], arm_qd=sim["arm_qd"][k],
            hand_q=to_dof(sim["hand_q"][k]) * args.hand_scale,
            hand_qd=to_dof(sim["hand_qd"][k]),
            object_pos=obj,
            tip_force_world=np.zeros((5, 3)),
            tip_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (5, 1))))
        rows.append((out.action.copy(), sim["action"][k].copy()))

    mine = np.array([r[0] for r in rows])
    theirs = np.array([r[1] for r in rows])
    dm = np.abs(mine - theirs)
    cm = 0.5 * (np.clip(mine[:, 6:], -1, 1) + 1).mean(axis=1)
    ct = 0.5 * (np.clip(theirs[:, 6:], -1, 1) + 1).mean(axis=1)

    mode = "free-run" if args.free_run else "teacher-forced"
    print(f"■ {args.traj.name} · {n} 스텝 · {mode} · 물체 오프셋 "
          f"{np.round(_obj_off*1000,0).tolist()} mm · 손배율 {args.hand_scale:g}")
    print(f"  액션 차이  팔(a0~5) 평균 {dm[:, :6].mean():.3f} 최대 {dm[:, :6].max():.3f}")
    print(f"            손(a6~20) 평균 {dm[:, 6:].mean():.3f} 최대 {dm[:, 6:].max():.3f}")
    print(f"  지시 폐쇄  배포 {cm.mean():.3f}  vs  sim {ct.mean():.3f}")
    print("\n  스텝  배포폐쇄  sim폐쇄  |Δ손액션|max  |Δ팔액션|max")
    for k in range(0, n, max(1, n // 12)):
        print(f"  {k:4d}   {cm[k]:.3f}    {ct[k]:.3f}      {dm[k, 6:].max():.3f}"
              f"         {dm[k, :6].max():.3f}")
    print("\n★해석: 손 액션이 맞으면 배포 obs 파이프라인은 무죄다 — 원인은 실기 상태.")
    print("       크게 다르면 그 스텝의 obs 항을 파고들 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
