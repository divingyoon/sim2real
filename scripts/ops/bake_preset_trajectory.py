#!/usr/bin/env python3
"""전환 bag 에 **정책 홈까지의 브리지**를 구워 넣어 완결 preset 궤적을 만든다.

**왜 굽는가.** 기록된 전환 bag(`reset_left_safe.npz`)은 차렷에서 출발하지만 끝이 정책
홈과 어긋난다(v2E29 기준 j4 21°·j7 28.6°). 러너는 그 간극을 실행 중에 램프로 이었는데,
그러면 preset 이 Isaac 러너에 묶인다. 궤적에 미리 구워 두면 **`shadow_replay.py` 만으로**
preset 을 낼 수 있다 — sim 도 정책도 필요 없는 순수 ROS2 노드다.

굽는 것: bag 프레임 뒤에 ①홈까지 등속 램프 ②정착(홈 유지) 를 붙인다. 속도는 bag 과
같은 상한(기본 0.25 rad/s)을 지킨다.

★검산 — 붙인 구간의 TCP 가 테이블 위로 얼마나 뜨는지 FK 로 확인하고 리포트한다.
  붙이는 구간은 기록된 적이 없는 새 경로라, 여기가 유일한 안전 근거다.

    python3 bake_preset_trajectory.py \\
        --bag  logs/shadow/reset_both/reset_left_safe.npz \\
        --run  logs/policy/left_v2E29/params/env.yaml \\
        --out  logs/shadow/reset_both/preset_left_v2e29.npz
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

DEFAULT_URDF = Path("/home/user/rl_ws/urdf/generated/rl/openarm_tesollo_sensor_rl.urdf")
ROBOT_CONTROL_SRC = "/home/user/rl_ws/robot_control/src"
TABLE_Z = 0.200


def home_from_run(env_yaml: Path) -> np.ndarray:
    """런 dump 에서 좌팔 홈 — ★홈의 진실원천은 소스 상수가 아니라 dump 다."""
    text = env_yaml.read_text()
    out = []
    for i in range(1, 8):
        m = re.search(rf"^\s*l_aj_{i}:\s*(-?[0-9.eE+]+)\s*$", text, re.M)
        if m is None:
            raise SystemExit(f"dump 에 l_aj_{i} 가 없다: {env_yaml}")
        out.append(float(m.group(1)))
    return np.array(out, dtype=np.float32)


def tcp_clearance(qs: np.ndarray) -> tuple:
    """붙인 구간의 TCP z 최저값과 그 프레임 — 테이블 여유의 근거."""
    sys.path.insert(0, ROBOT_CONTROL_SRC)
    from robot_control.kinematics import chain_from_urdf
    from robot_control.profile import load_builtin_profile

    prof = load_builtin_profile("openarm_tesollo")
    grp = prof.groups["openarm_left_arm"]
    chain = chain_from_urdf(DEFAULT_URDF.read_text(), list(grp.joints), grp.asset_tip_link)
    zs = np.array([chain.pose(q)[2, 3] for q in qs])
    k = int(np.argmin(zs))
    return float(zs[k]), k, zs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bag", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="런 dump params/env.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-vel", type=float, default=0.25, help="브리지 상한 rad/s")
    ap.add_argument("--settle", type=float, default=1.5, help="홈 유지 시간 s")
    ap.add_argument("--min-clearance", type=float, default=0.05,
                    help="붙인 구간이 지켜야 할 테이블 여유 m — 못 지키면 굽지 않는다")
    args = ap.parse_args()

    d = np.load(args.bag, allow_pickle=True)
    arm = d["arm_target"].reshape(len(d["arm_target"]), -1).astype(np.float32)
    grip = d["grip_cmd"].reshape(len(d["grip_cmd"]), -1).astype(np.float32)
    dt = float(d["meta_step_dt"][0])
    names = [str(n) for n in d["meta_joint_names"]]
    home = home_from_run(args.run)
    if list(names) != [f"l_aj_{i}" for i in range(1, 8)]:
        raise SystemExit(f"좌팔 bag 이 아니다: {names}")

    gap = home - arm[-1]
    n_ramp = max(int(np.ceil(float(np.abs(gap).max()) / (args.max_vel * dt))), 1)
    ramp = np.array([arm[-1] + gap * (k + 1) / n_ramp for k in range(n_ramp)],
                    dtype=np.float32)
    n_settle = max(int(round(args.settle / dt)), 1)
    settle = np.repeat(home[None, :], n_settle, axis=0).astype(np.float32)
    added = np.concatenate([ramp, settle])

    z_min, k_min, zs = tcp_clearance(added.astype(np.float64))
    clear = z_min - TABLE_Z
    print(f"[bake] bag {len(arm)}f + 램프 {n_ramp}f + 정착 {n_settle}f = {len(arm)+len(added)}f")
    print(f"[bake] 브리지 간극(deg) {np.round(np.degrees(gap), 1).tolist()} · "
          f"|max| {np.degrees(np.abs(gap).max()):.1f}°")
    print(f"[bake] 붙인 구간 속도 {float(np.abs(gap).max())/(n_ramp*dt):.3f} rad/s "
          f"(상한 {args.max_vel})")
    print(f"[bake] 붙인 구간 TCP 최저 z {z_min:.3f} m · 테이블 여유 {clear*1000:+.0f} mm "
          f"(프레임 {k_min}/{len(added)})")
    if clear < args.min_clearance:
        raise SystemExit(
            f"[bake] 여유 {clear*1000:.0f} mm < 하한 {args.min_clearance*1000:.0f} mm — "
            "굽지 않는다. 브리지가 테이블을 스친다")

    out_arm = np.concatenate([arm, added])[:, None, :]
    out_grip = np.concatenate([grip, np.repeat(grip[-1][None, :], len(added), axis=0)])[:, None, :]
    payload = {
        "arm_target": out_arm, "grip_cmd": out_grip,
        "meta_joint_names": d["meta_joint_names"], "meta_grip_names": d["meta_grip_names"],
        "meta_step_dt": d["meta_step_dt"], "meta_start_q": d["meta_start_q"],
        "meta_home_q": home,
        "meta_world": d["meta_world"] if "meta_world" in d.files else np.array(["-"]),
        "meta_gravity_comp": (d["meta_gravity_comp"] if "meta_gravity_comp" in d.files
                              else np.array(["-"])),
        "meta_baked_from": np.array([str(args.bag)]),
        "meta_baked_run": np.array([str(args.run)]),
        "meta_baked_clearance_m": np.array([clear]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)
    print(f"[bake] 저장 {args.out}")
    print(f"[bake] 끝 프레임 − 홈(deg) "
          f"{np.round(np.degrees(out_arm[-1, 0] - home), 3).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
