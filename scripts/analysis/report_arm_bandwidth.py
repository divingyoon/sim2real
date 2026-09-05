#!/usr/bin/env python3
"""sim 팔 vs 실기 팔의 **추종 대역폭 격차**를 표로 출력한다 (정책·ROS·warp 불필요).

"정책 액션을 로봇 컨트롤러가 못 따라간다" 는 증상의 물리적 근거를 수치로 만든다.
sim 은 팔을 position target + stiffness 400 / damping 80 으로 굴리고, 실기는 JTC →
CAN MIT 펌웨어 PD(kp 70·60·10 / kd 2.75~0.5)다. 같은 관성에 대해 2차계 특성
(ω_n, ζ)을 비교하면 격차가 한 눈에 보인다.

    python3 report_arm_bandwidth.py [--robot tesollo_bi_s__right] [--md]

입력 출처(전부 실측/자산 — 지어낸 값 없음):
  · 게인   r2s_autotune right_arm_best_calibration.json
  · 관성   urdf/generated/rl/<asset>.urdf 에서 계산 (홈 자세, 대각 근사)
  · sim    grasp_{side}_env_cfg.py ImplicitActuatorCfg (400/80)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


from arm_inertia import effective_inertia          # noqa: E402
from arm_pd_model import (                         # noqa: E402
    CALIB_JSON,
    SIM_DAMPING,
    SIM_STIFFNESS,
    bandwidth_gap,
    load_arm_pd,
)
from robot_profile import (                        # noqa: E402
    WS_ROOT,
    expected_q_home_arm,
    load_robot_profile,
)

DEFAULT_URDF = WS_ROOT / "urdf/generated/rl/openarm_tesollo_bi_s_rl.urdf"
HZ = 2.0 * np.pi


def compute(robot: str, urdf: Path = DEFAULT_URDF):
    prof = load_robot_profile(robot)
    q_home = np.asarray(expected_q_home_arm(prof), dtype=float)
    joints = list(prof.arm_canonical)
    inertia = effective_inertia(urdf, joints, {n: float(v) for n, v in zip(joints, q_home)})
    kp, kd, fc, dataset = load_arm_pd()
    return dict(
        profile=prof, joints=joints, q_home=q_home, inertia=inertia,
        kp=kp, kd=kd, fc=fc, dataset=dataset, urdf=urdf, **bandwidth_gap(inertia, kp, kd),
    )


def _rows(d):
    for i, n in enumerate(d["joints"]):
        yield (
            n, d["inertia"][i], d["kp"][i], d["kd"][i],
            d["wn_real"][i] / HZ, d["zeta_real"][i],
            d["wn_sim"][i] / HZ, d["zeta_sim"][i], d["ratio"][i],
        )


def print_text(d) -> None:
    print(f"구성        {d['profile'].name}")
    print(f"홈 자세     {np.round(d['q_home'], 4).tolist()}")
    print(f"게인 출처   {CALIB_JSON.name} (dataset={d['dataset']})")
    print(f"관성 출처   {d['urdf'].name} — 홈 자세 기준, 대각 근사\n")
    print("관절    I[kg·m²]   실기 kp/kd     f_n[Hz]    ζ     |  sim 400/80  f_n[Hz]    ζ     | 대역폭비")
    print("-" * 100)
    for n, I, kp, kd, fr, zr, fs, zs, ratio in _rows(d):
        print(f"{n:6s} {I:9.4f}  {kp:5.1f}/{kd:5.2f}   {fr:6.2f}  {zr:6.2f}  |"
              f"              {fs:6.2f}  {zs:6.2f}  |  {ratio:5.2f}배")
    print(f"\n대역폭비  평균 {d['ratio'].mean():.2f}배 · 최대 {d['ratio'].max():.2f}배"
          "   (sim 이 그만큼 빠르게 추종한다)")
    print(f"감쇠비 ζ  실기 {d['zeta_real'].min():.2f}~{d['zeta_real'].max():.2f}"
          f"  vs  sim {d['zeta_sim'].min():.2f}~{d['zeta_sim'].max():.2f}")
    if d["zeta_real"].min() < 1.0:
        under = [d["joints"][i] for i in np.where(d["zeta_real"] < 1.0)[0]]
        print(f"          ★실기 부족감쇠(ζ<1, 오버슛·진동) 관절: {under}")


def print_markdown(d) -> None:
    print("| 관절 | I[kg·m²] | 실기 kp/kd | 실기 f_n[Hz] | 실기 ζ | sim f_n[Hz] | sim ζ | 대역폭비 |")
    print("|---|---|---|---|---|---|---|---|")
    for n, I, kp, kd, fr, zr, fs, zs, ratio in _rows(d):
        print(f"| `{n}` | {I:.4f} | {kp:.1f} / {kd:.2f} | {fr:.2f} | {zr:.2f} "
              f"| {fs:.2f} | {zs:.2f} | **{ratio:.2f}배** |")
    print(f"\n대역폭비 평균 **{d['ratio'].mean():.2f}배**, 최대 **{d['ratio'].max():.2f}배**. "
          f"실기 ζ {d['zeta_real'].min():.2f}~{d['zeta_real'].max():.2f} "
          f"vs sim {d['zeta_sim'].min():.2f}~{d['zeta_sim'].max():.2f}.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="tesollo_bi_s__right")
    ap.add_argument("--urdf", default=str(DEFAULT_URDF))
    ap.add_argument("--md", action="store_true", help="Markdown 표로 출력")
    args = ap.parse_args()
    d = compute(args.robot, Path(args.urdf))
    (print_markdown if args.md else print_text)(d)
    print(f"\n※ sim 액추에이터 = stiffness {SIM_STIFFNESS:.0f} / damping {SIM_DAMPING:.0f}"
          " (grasp_{side}_env_cfg.py ImplicitActuatorCfg)")


if __name__ == "__main__":
    main()
