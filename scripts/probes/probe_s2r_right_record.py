#!/usr/bin/env python3
"""우팔 grasp_s2r(m1, LSTM) 롤아웃 기록 — 성공 에피소드를 골라 재생용 npz 로 남긴다.

★소스는 **live hdgp** 다. m1(08.29, ep20000)의 원 런 커밋은 로컬에 없고, 그 뒤 커밋
  (a85b3c4·0293644)은 보상 항 변경뿐이며 새 플래그는 기본 꺼짐이다. obs/action 계약은
  env 생성 시 출력되는 155/21 로 검증되고, 어긋나면 player.restore 가 죽는다.
★성공 판정 = `_stay_run ≥ stay_hold_steps`(goal 도달 + 안정 + 파지 유지 60스텝).
★기록은 env.step **직전** 스냅샷(pour 추출과 같은 규약). done 뒤 버퍼는 리셋되어 있어
  성공 판정은 기록된 시계열로 후처리한다.

산출물: 선정 에피소드의 q_target/q_meas(36관절)·palm·컵·goal·action(21) 시계열 npz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="open-sens_r_grasp_s2r-play-lstm")
parser.add_argument("--checkpoint", type=Path, default=Path(
    "/home/user/rl_ws/sim2real/logs/policy/right_m1/nn/m1_final.pth"))
parser.add_argument("--out", type=Path, default=Path(
    "/home/user/rl_ws/sim2real/logs/shadow/sim_m1_right"))
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--success_episodes", type=int, default=3)
parser.add_argument("--max_steps", type=int, default=4000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--adr_level", type=float, default=None,
                    help="ADR 레벨 강제(0~1). m1 은 만렙 학습 — level 0 재생에서 goal 을 "
                         "3~7cm 못 미치는 계통 실패가 나와 실측용으로 추가")
parser.add_argument("--stochastic", action="store_true",
                    help="학습 롤아웃과 같은 확률적 샘플링(σ 포함). 결정론 μ 재생이 "
                         "goal 3~7cm 못 미칠 때의 대조 실험용")
parser.add_argument("--gui", action="store_true")

_HDGP = Path("/home/user/rl_ws/hdgp")
sys.path.insert(0, str(_HDGP / "source" / "openarm"))
sys.path.insert(0, str(_HDGP / "scripts" / "tools"))

from isaaclab.app import AppLauncher                              # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args.headless = not args.gui

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math                                                        # noqa: E402
import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402
import gymnasium as gym                                            # noqa: E402

import openarm.tasks                                               # noqa: E402,F401
import fabrics_sim                                                 # noqa: E402
import openarm                                                     # noqa: E402
from isaaclab.envs import DirectRLEnvCfg                           # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config           # noqa: E402
from rl_games.common import env_configurations, vecenv             # noqa: E402
from rl_games.torch_runner import Runner                           # noqa: E402


def _assert_source_tree() -> None:
    for mod in (openarm, fabrics_sim):
        p = Path(mod.__file__).resolve()
        print(f"[소스] {mod.__name__}: {p}")
        if _HDGP not in p.parents:
            raise RuntimeError(f"{mod.__name__} 이 live hdgp 밖에서 로드됨: {p}")


@hydra_task_config(args.task, "rl_games_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    _assert_source_tree()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    # ★★런 dump 를 cfg 에 되씌운다 — 이거 없이 돌리면 현 소스 기본값으로 돈다.
    #   실측: m1 은 `palm_anchor_mode: spawn` 학습인데 현 기본값은 "home" — a=0 의
    #   앵커가 통째로 달라져 전 에피소드 실패(stay 0, ~160스텝 조기종료)했다.
    from run_cfg_restore import restore_run_cfg_if_available
    agent_cfg = restore_run_cfg_if_available(
        env_cfg, agent_cfg, resume_path=str(args.checkpoint),
        workspace_root=str(_HDGP.parent))
    # ★복원 **뒤에** 다시 강제한다(t6x 교훈: dump 의 num_envs 1024 가 조용히 되살아난다).
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg["params"]["seed"] = args.seed
    agent_cfg["params"]["config"]["device"] = "cuda:0"
    agent_cfg["params"]["config"]["device_name"] = "cuda:0"
    # 기록은 학습이 아니다 — 난이도 자동조절을 끈다(m1 은 만렙 학습이 끝난 정책).
    if args.adr_level is None:
        if hasattr(env_cfg, "enable_adr"):
            env_cfg.enable_adr = False
            print("[설정] enable_adr=False (복원 뒤 강제)")
    else:
        env_cfg.enable_adr = True
        print(f"[설정] enable_adr=True + _adr_level={args.adr_level} 강제 예정")
    print(f"[설정] palm_anchor_mode={getattr(env_cfg, 'palm_anchor_mode', '?')} (m1 dump 기대: spawn)")

    env = gym.make(args.task, cfg=env_cfg)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_act = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    print(f"[설정] clip_obs={clip_obs} clip_actions={clip_act}")
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_act)
    vecenv.register("IsaacRlgWrapper",
                    lambda cfg_name, n_actors, **kw: RlGamesGpuEnv(cfg_name, n_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(args.checkpoint)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(str(args.checkpoint))
    agent.reset()

    base = env.unwrapped
    if args.adr_level is not None:
        base._adr_level = float(args.adr_level)
        # 승급 로직이 도로 낮추지 못하게 임계도 만족 불가로
        base.cfg.adr_success_threshold = 2.0
        print(f"[설정] _adr_level={base._adr_level}")
    robot = base.robot
    origins = base.scene.env_origins
    jn = list(robot.joint_names)
    palm_i = robot.body_names.index("r_hl_palm")
    stay_need = int(base.cfg.stay_hold_steps)
    # ★판정 = `success_now` 지속 스텝. m1 학습 실측(m1.json): reward/stay 끝값 0.0001 —
    #   60스텝 stay 홀드는 **학습에서도 거의 달성된 적이 없다**. 학습이 세던 성공은
    #   gate/success_now(끝값 0.53)·species/success 0.61~0.79. stay 로 거르면 전부
    #   실패로 오독한다(실측 0/40+).
    ok_need = 10
    step_dt = float(base.step_dt)
    print(f"[설정] 관절 {len(jn)} · step_dt {step_dt:.4f}s · 성공 = success_now {ok_need}스텝+")

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    def _body_pose(idx: int) -> np.ndarray:
        pos = (robot.data.body_pos_w[:, idx] - origins).cpu().numpy()
        quat = robot.data.body_quat_w[:, idx].cpu().numpy()
        return np.concatenate([pos, quat], axis=-1).astype(np.float32)

    rec: dict[str, list] = {k: [] for k in (
        "q_meas", "qd_meas", "q_target", "palm_r", "obj", "goal",
        "action", "stay_run", "success_now", "done")}

    n_ok = 0
    step = 0
    with torch.inference_mode():
        while simulation_app.is_running() and step < args.max_steps:
            rec["q_meas"].append(robot.data.joint_pos.cpu().numpy().astype(np.float32))
            rec["qd_meas"].append(robot.data.joint_vel.cpu().numpy().astype(np.float32))
            rec["q_target"].append(robot.data.joint_pos_target.cpu().numpy().astype(np.float32))
            rec["palm_r"].append(_body_pose(palm_i))
            rec["obj"].append((base.object.data.root_pos_w - origins).cpu().numpy().astype(np.float32))
            rec["goal"].append(base.goal_pos.cpu().numpy().astype(np.float32))
            rec["stay_run"].append(base._stay_run.cpu().numpy().astype(np.int32))
            rec["success_now"].append(base._success_now.cpu().numpy().astype(bool))

            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=not args.stochastic)
            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]
            rec["action"].append(actions.cpu().numpy().astype(np.float32))
            d = dones.cpu().numpy().astype(bool)
            rec["done"].append(d)
            if agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0
            step += 1

            if d.any():
                arr_stay = np.stack(rec["stay_run"])
                arr_done = np.stack(rec["done"])
                for i in np.nonzero(d)[0]:
                    ends = np.nonzero(arr_done[:, i])[0]
                    s0 = 0 if len(ends) < 2 else ends[-2] + 1
                    _sn = np.stack(rec["success_now"])[s0:ends[-1] + 1, i]
                    ok = bool(_sn.sum() >= ok_need)
                    n_ok += int(ok)
                    # 진단: 어디서 멈췄나 — 물체가 들리긴 했나(z 상승), goal 에 갔나,
                    # palm 이 물체에 닿긴 했나.
                    _obj = np.stack(rec["obj"])[s0:ends[-1]+1, i]
                    _goal = np.stack(rec["goal"])[s0:ends[-1]+1, i]
                    _palm = np.stack(rec["palm_r"])[s0:ends[-1]+1, i, :3]
                    _lift = float(_obj[:, 2].max() - _obj[0, 2])
                    _gd = float(np.linalg.norm(_goal - _obj, axis=1).min())
                    _pd = float(np.linalg.norm(_palm - _obj, axis=1).min())
                    print(f"[ep] env{i} steps {ends[-1]-s0+1} succ_now {_sn.sum()}스텝 stay_max "
                          f"{arr_stay[s0:ends[-1]+1, i].max()} → {'✅성공' if ok else '실패'}"
                          f"  (누적 {n_ok}/{args.success_episodes})"
                          f"  lift {_lift*1000:.0f}mm · obj→goal 최소 {_gd*1000:.0f}mm"
                          f" · palm→obj 최소 {_pd*1000:.0f}mm")
                if n_ok >= args.success_episodes:
                    break

    A = {k: np.stack(v) for k, v in rec.items()}
    T, N = A["done"].shape
    cands = []
    for i in range(N):
        ends = np.nonzero(A["done"][:, i])[0]
        s0 = 0
        for e in ends:
            seg = slice(s0, e + 1)
            if A["success_now"][seg, i].sum() >= ok_need:
                qd = np.abs(np.diff(A["q_target"][seg, i], axis=0)) / step_dt
                cands.append((float(np.percentile(qd.max(1), 99)), i, s0, e))
            s0 = e + 1
    if not cands:
        print("[결론] ❌ 성공 에피소드 없음 — 시드를 바꿔 재시도할 것.")
        return 1
    cands.sort()
    p99, i, s0, e = cands[0]
    seg = slice(s0, e + 1)
    args.out.mkdir(parents=True, exist_ok=True)
    A["q_target"][s0] = A["q_meas"][s0]     # 리셋 텔레포트 프레임 보정(pour 추출과 동일)
    meta = dict(meta_joint_names=np.array(jn), meta_step_dt=np.float32(step_dt),
                meta_checkpoint=str(args.checkpoint), meta_stay_need=np.int32(stay_need))
    traj = args.out / f"m1_traj_env{i}_s{s0}_e{e}.npz"
    np.savez_compressed(traj, **{k: A[k][seg, i] for k in A}, **meta)
    n_steps = e + 1 - s0
    qd = np.abs(np.diff(A["q_target"][seg, i], axis=0)) / step_dt
    print(f"\n[선정] env{i} steps {n_steps} ({n_steps*step_dt:.1f}s) · |Δq_target|/dt "
          f"p99 {p99:.3f} · max {qd.max():.3f} rad/s · 후보 {len(cands)}개")
    print(f"[저장] {traj}")
    print(f"[초기] goal {np.round(A['goal'][s0, i], 4)} · obj {np.round(A['obj'][s0, i], 4)}")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code or 0)
