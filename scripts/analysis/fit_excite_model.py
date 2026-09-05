#!/usr/bin/env python3
"""여진 응답에 **2차 모델을 fit** 해서 sim 파라미터(kp·kd·armature)를 계산한다.

왜 이게 필요한가(09.01). `robotctl r2s fit` 은 모델에 **armature(모터 반사관성)가
없다**. 그런데 우팔 손목은 느끼는 관성의 95~97 %가 모터 쪽이라, 그걸 빼놓은 모델은
kp 를 부풀려 맞춘다 — fit 이 낸 j6 kp 189.5 로 관성을 역산하면 1.09 kg·m² 가 되고,
그건 손 전체(1.763 kg)가 손목에서 79 cm 떨어져 있어야 나오는 값이다.

그리고 sim 을 직접 돌려 맞추는 것(sim-in-the-loop)은 1회 6분이라 최적화가 안 된다.
여진은 자유공간의 작은 진동이므로 **관절별 선형 2차계**로 충분히 서는데, 그 fit 은
초 단위다. 여기서 파라미터를 뽑고 sim 으로 한 번 검증하는 것이 맞는 순서다.

모델(관절 독립, 중력은 보상으로 상쇄됨):

    J q̈ + kd q̇ + kp q = kp q_des + kd q̇_des        (`--with-vel-ff`)
    J q̈ + kd q̇ + kp q = kp q_des                    (기본)

  두 번째가 기본인 이유: 컨트롤러가 `interpolation_method: "none"` 이라 스트림
  포인트에서 속도 지령이 서지 않는다. 어느 쪽이 맞는지는 잔차가 답한다 — 둘 다
  돌려 보고 작은 쪽을 쓴다.

  ωn² = kp/J · 2ζωn = kd/J 로 두면 fit 되는 것은 (ωn, ζ, 지연)뿐이고 kp 와 J 는
  따로 결정되지 않는다(전달함수가 둘의 비만 담는다). 그래서 **kp 는 밖에서 주고**
  (`--kp`), J = kp/ωn² · armature = J − J_link · kd = 2ζωn J 로 환산한다.

    python3 fit_excite_model.py                      # 3런 fit + holdout
    python3 fit_excite_model.py --with-vel-ff        # 속도 피드포워드 모델
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares

sys.path.insert(0, "/home/user/rl_ws/robot_control/src")

R2S = Path("/home/user/rl_ws/sim2real/logs/r2s")
URDF = Path("/home/user/rl_ws/urdf/generated/rl/openarm_tesollo_sensor_rl.urdf")
PROFILE = Path("/home/user/rl_ws/robot_control/src/robot_control/profiles/openarm_tesollo.yaml")
FIST = Path("/home/user/rl_ws/sim2real/config/right_hand_fist.yaml")
ARM = [f"r_aj_{i}" for i in range(1, 8)]
R3 = np.array([0.038, 0.9, 0.6015, 2.0, 0.0294, 0.706, 0.4213])
#: 적분 격자. 실기 지령 100 Hz 보다 촘촘해야 2차계가 제대로 선다.
GRID_HZ = 400.0


def _simulate(cmd: np.ndarray, dt: float, wn: float, zeta: float,
              vel_ff: bool, fc: float = 0.0) -> np.ndarray:
    """2차계 + Coulomb 마찰을 적분한다. cmd 는 이미 지연이 반영된 지령.

    ★마찰을 모델에 넣는 이유. 빼면 그 감쇠 효과가 ζ 로 흡수되고, sim 에는 kd 와
    friction 을 **둘 다** 넣게 되어 감쇠가 이중 계산된다 — 09.01 에 실제로 그렇게
    되어 sim 오버슈트가 실기보다 작게 나왔다(j6 1.51 vs 2.01).

    *fc* 는 관성으로 정규화된 마찰(Fc/J)이라 단위가 rad/s². 토크로 되돌릴 때는
    J = kp/ωn² 를 곱한다.
    """
    q = np.zeros(len(cmd))
    v = 0.0
    q[0] = cmd[0]
    x = cmd[0]
    k = wn * wn
    c = 2.0 * zeta * wn
    dcmd = np.gradient(cmd, dt) if vel_ff else np.zeros(len(cmd))
    for i in range(1, len(cmd)):
        acc = k * (cmd[i - 1] - x) + c * (dcmd[i - 1] - v)
        if fc > 0.0:
            # 정지 마찰까지 흉내내지는 않는다. 움직이는 동안의 저항만 본다 —
            # 여진은 계속 움직이므로 그 구간이 신호의 대부분이다.
            if abs(v) > 1e-6:
                acc -= fc * np.sign(v)
            elif abs(acc) < fc:
                acc = 0.0
        v += acc * dt
        x += v * dt
        q[i] = x
    return q


def _fit_joint(t_cmd, cmd, t_meas, meas, vel_ff, with_friction):
    """(ωn, ζ, 지연[, 마찰]) 을 맞춘다. 잔차는 measured 시각에서 잰다."""
    dt = 1.0 / GRID_HZ
    grid = np.arange(t_cmd[0], t_cmd[-1], dt)

    def residual(p):
        wn, zeta, delay = p[0], p[1], p[2]
        fc = abs(p[3]) if with_friction else 0.0
        shifted = np.interp(grid - delay, t_cmd, cmd, left=cmd[0], right=cmd[-1])
        sim = _simulate(shifted, dt, abs(wn), abs(zeta), vel_ff, fc)
        return np.interp(t_meas, grid, sim) - meas

    lo = [0.5, 0.01, 0.0] + ([0.0] if with_friction else [])
    hi = [200.0, 5.0, 0.15] + ([50.0] if with_friction else [])
    best = None
    for wn0 in (5.0, 10.0, 15.0, 25.0):
        for z0 in (0.1, 0.4, 0.9):
            seed = [wn0, z0, 0.02] + ([0.5] if with_friction else [])
            try:
                out = least_squares(residual, seed, bounds=(lo, hi),
                                    xtol=1e-10, ftol=1e-10, max_nfev=300)
            except Exception:                     # noqa: BLE001
                continue
            if best is None or out.cost < best.cost:
                best = out
    return best


def _link_inertia() -> np.ndarray:
    from robot_control.kinematics import chain_from_urdf, with_payload

    urdf = URDF.read_text()
    chain = with_payload(chain_from_urdf(urdf, ARM, "r_hl_palm_ee"),
                         0.9130, [-0.00450, -0.01723, 0.22147])
    frames = chain.frames(R3)
    com = [f[:3, :3] @ chain.links[i].com + f[:3, 3] for i, f in enumerate(frames)]
    out = np.zeros(7)
    for j, frame in enumerate(frames):
        axis = frame[:3, :3] @ chain.joints[j].axis
        axis = axis / np.linalg.norm(axis)
        origin = frame[:3, 3]
        total = 0.0
        for i in range(j, len(chain.links)):
            d = com[i] - origin
            perp = d - np.dot(d, axis) * axis
            total += chain.links[i].mass * float(np.dot(perp, perp))
        out[j] = total
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="right_R3_s0650,right_R3_s0651")
    parser.add_argument("--holdout", default="right_R3_s0652")
    parser.add_argument("--with-vel-ff", action="store_true")
    parser.add_argument("--no-friction", action="store_true",
                        help="마찰 항을 빼고 맞춘다(그러면 그 효과가 ζ 로 흡수된다)")
    parser.add_argument("--kp", default=None,
                        help="관절별 kp 7개(쉼표). 없으면 r2s fit 의 값을 쓴다.")
    parser.add_argument("--out", type=Path, default=R2S / "excite_model_fit.json")
    args = parser.parse_args()

    runs = [np.load(R2S / f"{n}.npz", allow_pickle=False)
            for n in args.runs.split(",")]
    hold = np.load(R2S / f"{args.holdout}.npz", allow_pickle=False)
    print(f"fit {len(runs)} 런 · holdout {args.holdout} · "
          f"모델 {'kp·q_des + kd·q̇_des' if args.with_vel_ff else 'kp·q_des (속도 ff 없음)'}")

    if args.kp:
        kp = np.array([float(x) for x in args.kp.split(",")])
    else:
        kp = np.array(json.loads((R2S / "right_R3_s065_fit.json").read_text())["stiffness"])

    j_link = _link_inertia()
    print(f"\n{'관절':8s} {'ωn[Hz]':>8s} {'ζ':>7s} {'지연[ms]':>8s} "
          f"{'RMSE[°]':>9s} {'holdout':>9s}")
    result = {}
    for k, name in enumerate(ARM):
        wn_list, z_list, d_list, fc_list, rmse = [], [], [], [], []
        for run in runs:
            t_cmd = (run["command_time_ns"] - run["command_time_ns"][0]) / 1e9
            t_meas = (run["measured_time_ns"] - run["command_time_ns"][0]) / 1e9
            out = _fit_joint(t_cmd, run["command"][:, k], t_meas,
                             run["measured"][:, k], args.with_vel_ff,
                             not args.no_friction)
            if out is None:
                continue
            wn_list.append(abs(out.x[0]))
            z_list.append(abs(out.x[1]))
            d_list.append(out.x[2])
            fc_list.append(abs(out.x[3]) if not args.no_friction else 0.0)
            rmse.append(np.degrees(np.sqrt(2 * out.cost / len(t_meas))))
        wn, zeta, delay = np.mean(wn_list), np.mean(z_list), np.mean(d_list)
        fc = float(np.mean(fc_list))

        t_cmd = (hold["command_time_ns"] - hold["command_time_ns"][0]) / 1e9
        t_meas = (hold["measured_time_ns"] - hold["command_time_ns"][0]) / 1e9
        dt = 1.0 / GRID_HZ
        grid = np.arange(t_cmd[0], t_cmd[-1], dt)
        shifted = np.interp(grid - delay, t_cmd, hold["command"][:, k],
                            left=hold["command"][0, k], right=hold["command"][-1, k])
        pred = np.interp(t_meas, grid,
                         _simulate(shifted, dt, wn, zeta, args.with_vel_ff, fc))
        ho = np.degrees(np.sqrt(np.mean((pred - hold["measured"][:, k]) ** 2)))

        inertia = kp[k] / wn ** 2
        result[name] = {
            "wn_hz": wn / (2 * np.pi), "zeta": zeta, "delay_s": delay,
            "rmse_deg": float(np.mean(rmse)), "holdout_rmse_deg": float(ho),
            "kp": float(kp[k]), "inertia": float(inertia),
            "armature": float(max(inertia - j_link[k], 0.0)),
            "kd": float(2 * zeta * wn * inertia), "j_link": float(j_link[k]),
            "friction": float(fc * inertia), "fc_norm": fc,
        }
        print(f"{name:8s} {wn/(2*np.pi):8.2f} {zeta:7.3f} {delay*1000:8.1f} "
              f"{np.mean(rmse):9.3f} {ho:9.3f}")

    print(f"\n{'관절':8s} {'kp(입력)':>9s} {'J=kp/ωn²':>9s} {'J_link':>8s} "
          f"{'armature':>9s} {'kd':>8s} {'friction':>9s}")
    for name in ARM:
        r = result[name]
        print(f"{name:8s} {r['kp']:9.1f} {r['inertia']:9.4f} {r['j_link']:8.4f} "
              f"{r['armature']:9.3f} {r['kd']:8.2f} {r['friction']:9.3f}")

    payload = {"model": "vel_ff" if args.with_vel_ff else "pos_only",
               "runs": args.runs.split(","), "holdout": args.holdout,
               "joints": result}
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
