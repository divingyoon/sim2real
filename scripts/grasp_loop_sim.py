#!/usr/bin/env python3
"""grasp_v1 라이브 루프 오프라인 재현기 — ROS·실기 불필요.

정책 tick 로직은 `GraspPolicyCore` 를 **그대로** 쓴다(라이브 노드와 동일 코어). 여기서
바꾸는 것은 로봇 쪽뿐이다: 실기 대신 mock 팔·손으로 폐루프를 닫는다.

팔 모델 (`--arm-model`)
  rate : arm ← arm + clip(cmd − arm, ±max_vel/CONTROL_HZ)
         브리지 velocity limiter 의 1차 근사. max_vel 을 크게 주면 sim 처럼 즉시 추종.
  pd   : 실기 펌웨어 MIT 루프 재현 — tau = kp(cmd − q) − kd·qd, qdd = (tau − Fc·sgn(qd))/I
         · kp/kd/Fc = **실측 캘리브레이션** (r2s_autotune right_arm_best_calibration.json)
         · I        = **자산 URDF 에서 계산** (arm_inertia.effective_inertia)
         값을 지어내지 않는다 — 둘 다 출처가 있는 수치이고 실행 시 로그로 남긴다.

손 모델 (`--hand-mode`)
  static = APPROACH 동결(fake손) / zero = 전관절 0(드라이버 두절 재현) / echo = 지령 즉시 반영(sim)

용도
  ① 계약·자산 회귀 확인 (홈 IK 검증이 코어 안에서 자동으로 돈다)
  ② ★요구 사양 측정: 정책이 요구하는 관절 속도·가속도를 CSV 로 남긴다(--demand-csv).
     `--arm-model rate --max-vel 99` 가 "sim 처럼 즉시 추종" = 순수 요구 프로파일이다.
  ③ 능력–요구 스윕: --arm-model pd 로 실기 게인을 넣고 접근 성공 여부를 본다.

실행(IsaacLab 번들 python 필요 — warp):
    IsaacLab/isaaclab.sh -p grasp_loop_sim.py --robot tesollo_bi_s__right \\
        --agent .../agent.yaml --ckpt .../ckpt.pth --arm-model pd --max-vel 0.5
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in [
    _SCRIPT_DIR.parent.parent / "hdgp" / "source" / "FABRICS" / "src",
    _SCRIPT_DIR.parent.parent / "repo" / "FABRICS" / "src",
]:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break
sys.path.insert(0, str(_SCRIPT_DIR))

from arm_inertia import effective_inertia                       # noqa: E402
from arm_pd_model import CALIB_JSON, MockArm, load_arm_pd, second_order_characteristics  # noqa: E402
from grasp_policy_core import GraspPolicyCore, TickSensors       # noqa: E402
from grasp_obs_builder import make_object_onehot                 # noqa: E402
from policy_loader import RLGamesLstmActorPolicy                 # noqa: E402
from robot_profile import (                                      # noqa: E402
    WS_ROOT,
    load_hdgp_module,
    load_robot_profile,
)

CONTROL_HZ = 60.0
NUM_ARM_DOF = 7


def run(args) -> None:
    profile = load_robot_profile(args.robot)
    preset = load_hdgp_module(profile, "preset")
    hand_approach = np.asarray(preset.HAND_APPROACH_POSE, dtype=np.float64)

    policy = RLGamesLstmActorPolicy(
        agent_yaml_path=args.agent, checkpoint_path=args.ckpt,
        obs_dim=profile.contract.obs_dim, action_dim=profile.contract.action_dim,
        device=args.device,
    )
    core = GraspPolicyCore(
        profile=profile, policy=policy, device=args.device,
        object_onehot=make_object_onehot(args.object),
    )
    print(f"[loop_sim] 구성={profile.name} fabrics={profile.fabrics.robot_dir} "
          f"obs={profile.contract.obs_dim} act={profile.contract.action_dim}")
    print("[loop_sim] q_home=[" + ", ".join(f"{v:+.4f}" for v in core.q_home_arm) + "]")

    dt = 1.0 / CONTROL_HZ
    kp = kd = fc = inertia = None
    if args.arm_model == "pd":
        kp, kd, fc, src = load_arm_pd()
        urdf = WS_ROOT / "urdf/generated/rl/openarm_tesollo_bi_s_rl.urdf"
        inertia = effective_inertia(
            urdf, list(profile.arm_canonical),
            {n: float(v) for n, v in zip(profile.arm_canonical, core.q_home_arm)},
        )
        print(f"[loop_sim] PD 모델 — 게인 출처 {CALIB_JSON.name} (dataset={src})")
        print("           kp=[" + ", ".join(f"{v:.1f}" for v in kp) + "]")
        print("           kd=[" + ", ".join(f"{v:.2f}" for v in kd) + "]")
        print(f"[loop_sim] 관성 출처 {urdf.name} (홈 자세 기준, 대각 근사)")
        print("           I =[" + ", ".join(f"{v:.4f}" for v in inertia) + "]")
        wn, zeta = second_order_characteristics(kp, kd, inertia)
        print("           → ω_n[Hz]=[" + ", ".join(f"{v/6.283:.2f}" for v in wn) + "]")
        print("           → ζ      =[" + ", ".join(f"{v:.2f}" for v in zeta) + "]")

    arm = MockArm(core.q_home_arm, args.arm_model, args.max_vel, dt,
                  kp, kd, fc, inertia, substeps=args.pd_substeps)
    hand = (np.zeros(20) if args.hand_mode == "zero" else hand_approach.copy())
    cup = np.array([args.cup_x, args.cup_y, args.cup_z])
    delay = deque(maxlen=max(1, args.obs_delay + 1))

    rows, dists = [], []
    prev_cmd = core.q_home_arm.copy()
    for step in range(args.steps):
        delay.append((arm.q.copy(), arm.qd.copy()))
        obs_arm, obs_arm_vel = delay[0]
        out = core.step(
            TickSensors(
                arm_pos=obs_arm, arm_vel=obs_arm_vel,
                hand_pos=hand, hand_vel=np.zeros(20),
                cup_pos=cup, tip_force_local=np.zeros((5, 3)),   # 접촉 없음(접근 거동만 판정)
            ),
            step_count=step,
        )
        dists.append(out.palm_cup_dist)
        cmd_vel = (out.arm_cmd - prev_cmd) * CONTROL_HZ
        rows.append(dict(
            step=step, dist=out.palm_cup_dist, is_lift=int(out.is_lift),
            **{f"cmd_vel_{i}": float(v) for i, v in enumerate(cmd_vel)},
            **{f"track_err_{i}": float(e) for i, e in enumerate(out.arm_cmd - arm.q)},
        ))
        prev_cmd = out.arm_cmd.copy()

        arm.step(out.arm_cmd)
        if args.hand_mode == "echo":
            hand = out.hand_cmd.copy()
        if step % 60 == 0:
            print(f"[loop_sim] step={step:4d} dist={out.palm_cup_dist:.3f} "
                  f"palm={out.palm_center.round(3).tolist()}")

    _summary(args, dists, rows)


def _summary(args, dists, rows) -> None:
    d0, dmin, dend = dists[0], min(dists), dists[-1]
    print("\n[loop_sim] === 결과 ===")
    print(f"  구성 arm_model={args.arm_model} max_vel={args.max_vel} "
          f"hand={args.hand_mode} obs_delay={args.obs_delay}")
    print(f"  palm→cup  시작 {d0:.3f} → 최소 {dmin:.3f} → 종료 {dend:.3f} [m]")

    vel = np.array([[r[f"cmd_vel_{i}"] for i in range(NUM_ARM_DOF)] for r in rows])
    err = np.array([[r[f"track_err_{i}"] for i in range(NUM_ARM_DOF)] for r in rows])
    absv = np.abs(vel)
    print("\n  ★정책이 요구한 관절 속도 [rad/s] (지령 차분)")
    print("    joint     mean     p95      max")
    for i in range(NUM_ARM_DOF):
        print(f"    j{i+1}     {absv[:, i].mean():7.3f} {np.percentile(absv[:, i], 95):7.3f} "
              f"{absv[:, i].max():7.3f}")
    print(f"    전체 최대 요구 속도 = {absv.max():.3f} rad/s")
    print(f"  추종오차(지령−실측) 최대 = {np.abs(err).max():.4f} rad")

    if args.demand_csv:
        path = Path(args.demand_csv).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  요구 프로파일 CSV → {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="tesollo_bi_s__right")
    ap.add_argument("--agent", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--object", default="cup_big_s100")
    ap.add_argument("--arm-model", choices=["rate", "pd"], default="rate",
                    help="rate=속도제한만(99=즉시추종=순수 요구) / pd=실측 게인 2차 적분")
    ap.add_argument("--max-vel", type=float, default=0.5,
                    help="브리지 관절 속도제한 [rad/s]")
    ap.add_argument("--hand-mode", choices=["static", "zero", "echo"], default="echo",
                    help="static=APPROACH 동결 / zero=드라이버 두절 재현 / echo=지령 반영(sim)")
    ap.add_argument("--obs-delay", type=int, default=0,
                    help="팔 관측 지연 [제어틱=1/60s] (6≈100ms)")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--cup-x", type=float, default=0.30)
    ap.add_argument("--cup-y", type=float, default=-0.20)
    ap.add_argument("--cup-z", type=float, default=0.297)
    ap.add_argument("--pd-substeps", type=int, default=32,
                    help="PD 물리 적분 서브스텝/제어틱 (저관성 관절 안정·정확도)")
    ap.add_argument("--demand-csv", default=None,
                    help="요구 속도·추종오차 per-step CSV 경로")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
