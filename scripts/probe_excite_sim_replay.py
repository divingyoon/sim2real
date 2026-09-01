#!/usr/bin/env python3
"""실기 여진 궤적을 **같은 시간축으로** sim 에서 재생해 오버슈트가 재현되는지 본다.

09.01 실기(R3 자세, amplitude_scale 0.65)에서 손목 3관절이 지령보다 크게 움직였다:

    r_aj_5 ×1.47 · r_aj_6 ×2.07 · r_aj_7 ×1.58   (팔 4관절은 0.96~1.02)

원인은 테솔로 손 1.763 kg 이 손목 관성을 10~12배 올려 ζ 를 1 아래로 끌어내린 것이다.
sim 이 같은 배율을 내면 **sim 이 실기의 동특성을 담고 있다**는 뜻이고, 정책이 그
오버슈트를 겪으며 학습할 수 있다.

★왜 `probe_s2r_gain_replay.py` 를 쓰지 않는가. 그것은 지령 1개당 `env.step_dt`(16.7 ms)
를 소비한다. 여진 지령은 10 ms 간격이므로 시간축이 1.67 배 늘어나고, 주파수가 그만큼
낮아진다. 동특성은 주파수의 함수이므로 그 비교는 성립하지 않는다. 여기서는 **물리
dt(8.33 ms) 격자로 리샘플**해 매 물리 스텝마다 지령을 갱신한다.

★손을 주먹으로 세운다. 실기 여진 때 손이 주먹이었고, 손 자세가 곧 손목 관성이다.
  손을 편 채로 재생하면 재현하려는 바로 그 양이 달라진다.

    # KUKA 게인(현행 기본값)
    ./isaaclab.sh -p probe_excite_sim_replay.py \
        --npz ~/rl_ws/sim2real/logs/r2s/right_R3_s0650.npz --out .../sim_excite_kuka.npz
    # 09.01 실측 게인
    HDGP_S2R_REAL_GAINS=1 ./isaaclab.sh -p probe_excite_sim_replay.py ... --out .../sim_excite_real.npz
"""

from __future__ import annotations

import argparse
import os as _os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="open-sens_r_grasp_s2r-play")
parser.add_argument("--npz", type=Path, required=True, help="r2s collect 가 쓴 여진 기록")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--fist", type=Path,
                    default=Path("/home/user/rl_ws/sim2real/config/right_hand_fist.yaml"))
parser.add_argument("--profile_yaml", type=Path,
                    default=Path("/home/user/rl_ws/robot_control/src/robot_control/"
                                 "profiles/openarm_tesollo.yaml"))
parser.add_argument("--hdgp_root", type=Path, default=Path("/home/user/rl_ws/hdgp"))
parser.add_argument("--fabrics_src", type=Path, default=None)
parser.add_argument("--settle", type=int, default=240, help="시작 자세 정착 물리스텝 수")
parser.add_argument("--num-envs", type=int, default=1,
                    help="★env 마다 다른 kd 를 주어 한 번에 여러 조합을 시험한다")
parser.add_argument("--kd-scale", default=None,
                    help="lo,hi — 손목 3관절(j5-7) kd 에 곱할 배율을 num_envs 개로 등분")
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
import yaml                                                       # noqa: E402

import fabrics_sim                                                # noqa: E402,F401
import gymnasium as gym                                           # noqa: E402,F401
import openarm.tasks                                              # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg                    # noqa: E402

ARM = [f"r_aj_{i}" for i in range(1, 8)]


def _fist(profile_yaml: Path, fist_yaml: Path) -> dict[str, float]:
    """드라이버 이름으로 적힌 주먹 자세를 URDF/sim 이름으로 옮긴다."""
    body = yaml.safe_load(profile_yaml.read_text())
    by_source = {j["source"]: (j["canonical"], j.get("sign", 1)) for j in body["joints"]}
    raw = yaml.safe_load(fist_yaml.read_text())["joints"]
    return {by_source[s][0]: float(v) * by_source[s][1]
            for s, v in raw.items() if s in by_source}


def main() -> int:
    real = _os.environ.get("HDGP_S2R_REAL_GAINS") == "1"
    print(f"[게인] {'09.01 실측(관절별)' if real else 'KUKA(기본값)'}")

    data = np.load(args.npz, allow_pickle=False)
    names = [str(x) for x in data["joint_names"]]
    if tuple(names) != tuple(ARM):
        raise SystemExit(f"우팔 기록이 아니다: {names}")
    cmd = data["command"].astype(np.float64)
    t_cmd = (data["command_time_ns"] - data["command_time_ns"][0]) / 1e9
    meas = data["measured"].astype(np.float64)
    t_meas = (data["measured_time_ns"] - data["measured_time_ns"][0]) / 1e9
    print(f"[기록] {args.npz.name} · 지령 {len(cmd)} @ {len(cmd)/t_cmd[-1]:.0f} Hz "
          f"· 실측 {len(meas)} @ {len(meas)/t_meas[-1]:.0f} Hz")

    n_env = max(1, int(args.num_envs))
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=n_env)
    env_cfg.scene.num_envs = n_env
    env_cfg.episode_length_s = 1e6
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    env.reset()

    physics_dt = float(env.sim.get_physics_dt())
    jn = robot.joint_names
    arm = [jn.index(n) for n in ARM]
    fist = _fist(args.profile_yaml, args.fist)
    hand = [(jn.index(n), v) for n, v in fist.items() if n in jn]
    print(f"[sim] physics_dt {physics_dt*1000:.2f} ms · 손 관절 {len(hand)} 개를 주먹으로 고정")

    kp = robot.data.joint_stiffness[0, arm].cpu().numpy()
    kd = robot.data.joint_damping[0, arm].cpu().numpy()
    print("[게인 실측] kp " + " ".join(f"{v:7.1f}" for v in kp))
    print("[게인 실측] kd " + " ".join(f"{v:7.2f}" for v in kd))
    # ★cfg 에 적었다고 적용된 것이 아니다 — 그룹 이름이 어긋나면 조용히 기본값으로
    #   떨어진다는 것이 이 저장소의 기존 교훈이다. 실제 버퍼를 읽어 찍는다.
    try:
        arm_inertia = robot.data.joint_armature[0, arm].cpu().numpy()
        print("[armature ] " + " ".join(f"{v:7.3f}" for v in arm_inertia))
    except AttributeError:
        arm_inertia = np.zeros(7)
        print("[armature ] 이 IsaacLab 버전은 joint_armature 를 노출하지 않는다")

    # ★env 별로 kd 를 달리 준다 — `Articulation` 이 env_ids 를 받는다
    #   (`articulation.py:640`). 한 번의 6분짜리 재생으로 num_envs 개 조합을 본다.
    #   팔(j1-4)은 이미 실기와 0.051 로 맞으므로 건드리지 않고 손목만 훑는다.
    scales = np.ones(n_env)
    if args.kd_scale and n_env > 1:
        lo, hi = (float(x) for x in args.kd_scale.split(","))
        scales = np.linspace(lo, hi, n_env)
        base = robot.data.joint_damping[0, arm].clone()
        kd_mat = base.unsqueeze(0).repeat(n_env, 1)
        factor = torch.tensor(scales, device=env.device, dtype=kd_mat.dtype)
        kd_mat[:, 4:7] = kd_mat[:, 4:7] * factor.unsqueeze(1)
        robot.write_joint_damping_to_sim(kd_mat, joint_ids=arm)
        applied = robot.data.joint_damping[:, arm].cpu().numpy()
        print(f"[kd 스윕] 손목 배율 {scales[0]:.2f}~{scales[-1]:.2f} · {n_env} env")
        print("  env0 kd " + " ".join(f"{v:6.3f}" for v in applied[0]))
        print(f"  env{n_env-1} kd " + " ".join(f"{v:6.3f}" for v in applied[-1]))
        kd = applied[0]

    # ★물리 dt 격자로 리샘플. 지령은 계단이 아니라 선형 보간 — r2s 트랙 자체가
    #   ramp/multisine 이라 계단으로 넣으면 없는 고주파를 집어넣게 된다.
    grid = np.arange(0.0, float(t_cmd[-1]), physics_dt)
    target = np.stack([np.interp(grid, t_cmd, cmd[:, k]) for k in range(7)], axis=1)
    print(f"[재생] {len(grid)} 물리스텝 ({grid[-1]:.2f} s)")

    full = robot.data.joint_pos[0].clone()
    for k, i in enumerate(arm):
        full[i] = float(target[0, k])
    for i, v in hand:
        full[i] = float(v)
    zero = torch.zeros_like(full)
    robot.write_joint_state_to_sim(full.unsqueeze(0).expand(n_env, -1).contiguous(),
                                   zero.unsqueeze(0).expand(n_env, -1).contiguous())
    robot.set_joint_position_target(full.unsqueeze(0).expand(n_env, -1).contiguous())
    robot.write_data_to_sim()
    for _ in range(args.settle):
        env.sim.step(render=False)
    env.scene.update(physics_dt)

    hand_ids = torch.tensor([i for i, _ in hand], device=env.device)
    hand_q = torch.tensor([v for _, v in hand], device=env.device,
                          dtype=torch.float32).unsqueeze(0).expand(n_env, -1).contiguous()
    arm_ids = torch.tensor(arm, device=env.device)
    # ★손 지령은 한 번만. PhysX 는 마지막 타겟을 유지하므로 매 스텝 다시 쓸 이유가
    #   없고, 매 스텝 쓰면 손 쪽 접촉 해석이 재생 루프에 얹혀 진단이 어려워진다.
    if len(hand):
        robot.set_joint_position_target(hand_q, joint_ids=hand_ids)
    # ★기록은 GPU 에 모았다가 끝나고 한 번에 내린다. 매 스텝 .cpu() 는 동기화를
    #   강제해 느리고, 그 동기화 자체가 멈춤 지점을 흐린다.
    plan = torch.tensor(target, device=env.device, dtype=torch.float32)
    trace = torch.zeros((len(grid), n_env, 7), device=env.device, dtype=torch.float32)
    log = np.zeros_like(target)
    step = 0
    try:
        for step in range(len(grid)):
            robot.set_joint_position_target(
                plan[step].unsqueeze(0).expand(n_env, -1).contiguous(), joint_ids=arm_ids)
            robot.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(physics_dt)
            trace[step] = robot.data.joint_pos[:, arm_ids]
            if step % 50 == 0:
                print(f"  {step}/{len(grid)}", flush=True)
        print(f"  {len(grid)}/{len(grid)} 재생 끝", flush=True)
    finally:
        # ★어디서 멈추든 여기까지 받은 것은 남긴다. 6 분짜리 실행을 다시 하지
        #   않으려면 부분 기록이라도 디스크에 있어야 한다.
        traced = trace.cpu().numpy().astype(np.float64)
        log = traced[:, 0, :]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out.with_suffix(".partial.npz"),
            command=target.astype(np.float32), q_sim=log.astype(np.float32),
            joint_names=np.array(ARM), physics_dt=np.array([physics_dt]),
            steps_done=np.array([step + 1]), q_sim_all=traced.astype(np.float32),
            kd_scales=scales.astype(np.float32),
            meta_kp=kp.astype(np.float32), meta_kd=kd.astype(np.float32),
            meta_real_gains=np.array([1 if real else 0]))
        print(f"  부분 기록 → {args.out.with_suffix('.partial.npz')}", flush=True)

    print("\n═══ 오버슈트 배율 (실측폭 / 지령폭) ═══")
    print(f"{'관절':8s} {'sim':>8s} {'실기':>8s} {'차':>8s}")
    sim_ratio, real_ratio = [], []
    for k in range(7):
        span_cmd = float(np.ptp(target[:, k]))
        s = float(np.ptp(log[:, k])) / max(span_cmd, 1e-9)
        r = float(np.ptp(meas[:, k])) / max(float(np.ptp(cmd[:, k])), 1e-9)
        sim_ratio.append(s)
        real_ratio.append(r)
        print(f"{ARM[k]:8s} {s:8.2f} {r:8.2f} {s-r:+8.2f}")
    sim_ratio, real_ratio = np.array(sim_ratio), np.array(real_ratio)
    print(f"{'평균오차':8s} {'':8s} {'':8s} {np.abs(sim_ratio-real_ratio).mean():+8.2f}")

    # ★스윕이면 env 별로 실기와 대조해 최적 배율을 고른다. 팔은 전 env 동일하므로
    #   손목(j5-7)만 점수에 넣는다 — 팔까지 넣으면 상수가 순위를 흐린다.
    best = None
    if n_env > 1:
        # ★점수는 **주파수 응답 진폭비**로 낸다. ptp 비(최대−최소)는 신호의 두 점만
        #   보아 ζ 와 단순 대응하지 않는다 — 09.01 에 kd 를 3배 키워도 ptp 가 12 %
        #   밖에 안 변해 최적화가 서지 않았다. lock-in 은 위상·지연에 무관하고
        #   대역별로 본다.
        freqs = (0.7, 1.3, 2.1, 3.7)

        def lockin(t, x, f):
            x = x - x.mean()
            return 2.0 * abs((x * np.exp(-2j * np.pi * f * t)).mean())

        t_sim = np.arange(len(grid)) * physics_dt
        hi = t_sim[-1] - 0.5
        win = (t_sim >= hi - 3.0) & (t_sim <= hi)          # multisine 구간
        t_c = (data["command_time_ns"] - data["command_time_ns"][0]) / 1e9
        t_m = (data["measured_time_ns"] - data["command_time_ns"][0]) / 1e9
        hr = t_c[-1] - 0.5
        wc = (t_c >= hr - 3.0) & (t_c <= hr)
        wm = (t_m >= hr - 3.0) & (t_m <= hr)
        real_fr = np.array([[lockin(t_m[wm], meas[wm, k], f)
                             / max(lockin(t_c[wc], cmd[wc, k], f), 1e-12)
                             for f in freqs] for k in range(7)])

        print(f"\n═══ kd 스윕 {n_env} env — 손목 주파수응답 오차 ═══")
        print(f"{'배율':>6s} " + " ".join(f"{f:>5.1f}Hz" for f in freqs)
              + f" {'손목오차':>9s}   (j5/j6/j7 평균)")
        rows = []
        for e in range(n_env):
            sim_fr = np.array([[lockin(t_sim[win], traced[win, e, k], f)
                                / max(lockin(t_sim[win], target[win, k], f), 1e-12)
                                for f in freqs] for k in range(7)])
            err = float(np.abs(sim_fr[4:] - real_fr[4:]).mean())
            rows.append((scales[e], sim_fr, err))
            print(f"{scales[e]:6.2f} "
                  + " ".join(f"{sim_fr[4:, j].mean():7.2f}" for j in range(len(freqs)))
                  + f" {err:9.3f}")
        print("실기    " + " ".join(f"{real_fr[4:, j].mean():7.2f}" for j in range(len(freqs))))
        best = min(rows, key=lambda r: r[2])
        print(f"\n★최적 배율 {best[0]:.2f} — 손목 주파수응답 오차 {best[2]:.3f}")
        print("  적용할 손목 kd = " + " ".join(f"{v*best[0]:.3f}" for v in kd[4:]))
        if best[0] in (scales[0], scales[-1]):
            print(f"  ⚠최적이 스윕 **끝**에 있다 — 범위를 넓혀 다시 볼 것")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, command=target.astype(np.float32), q_sim=log.astype(np.float32),
        best_scale=np.array([best[0] if best else 1.0]),
        joint_names=np.array(ARM), physics_dt=np.array([physics_dt]),
        sim_ratio=sim_ratio.astype(np.float32), real_ratio=real_ratio.astype(np.float32),
        meta_kp=kp.astype(np.float32), meta_kd=kd.astype(np.float32),
        meta_armature=arm_inertia.astype(np.float32),
        meta_real_gains=np.array([1 if real else 0]),
        meta_source=np.array([str(args.npz)]))
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
