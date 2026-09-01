#!/usr/bin/env python3
"""실기 정적 처짐(droop) 실측 — 자세별 (measured − commanded) 관절 오차표.

방법: 브리지( isaacsim_cmd_to_jtc )가 떠 있는 상태에서 /isaacsim/left_arm_cmd 로
고정 목표를 계속 쏘고, 정착 후 /joint_states 와의 차를 잰다. 목표는 ①preset 홈
②기록 npz 의 지정 프레임들 — **이미 실기로 지나간 궤적 위의 자세만** 쓴다(미검증
자세로 보내지 않는다).

출력: 관절별 mrad 표 + 평균 오프셋(브리지 선보상 주입값 후보).
★해석 주의: 이 값은 '지령 - 실측'이 아니라 실측−지령. 선보상은 **부호 반대로** 더한다.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

ARM_SRC = [f"openarm_left_joint{i}" for i in range(1, 8)]
CANON = [f"l_aj_{i}" for i in range(1, 8)]
HOME = [-0.0136, -0.3757, -0.0010, 0.9336, -0.4655, 0.0003, -0.3306]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", type=Path,
                    default=Path("logs/shadow/sim_v2H_wide.npz"))
    ap.add_argument("--frames", type=int, nargs="*", default=[150, 250],
                    help="npz 에서 목표로 쓸 프레임 인덱스(실기로 이미 지나간 구간만)")
    ap.add_argument("--hold", type=float, default=6.0, help="자세당 유지[s]")
    ap.add_argument("--settle", type=float, default=2.0, help="측정 전 정착[s]")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    d = np.load(args.sim, allow_pickle=True)
    targets = [("home", np.array(HOME))]
    for f in args.frames:
        targets.append((f"frame{f}", d["arm_target"][f, 0].astype(float)))

    rclpy.init()
    node = Node("probe_droop_real")
    pub = node.create_publisher(Float64MultiArray, "/isaacsim/left_arm_cmd", 10)
    meas: dict[str, float] = {}

    def _cb(msg: JointState) -> None:
        for i, n in enumerate(msg.name):
            meas[n] = msg.position[i]
    node.create_subscription(JointState, "/joint_states", _cb, 20)

    if not args.execute:
        for name, q in targets:
            print(f"{name}: {np.round(q, 4)}")
        print("DRY RUN — --execute 로 실제 이동/측정")
        return

    rows = []
    for name, q in targets:
        t0 = time.monotonic()
        samples = []
        while time.monotonic() - t0 < args.hold:
            m = Float64MultiArray(); m.data = list(q); pub.publish(m)
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() - t0 > args.hold - args.settle and all(s in meas for s in ARM_SRC):
                samples.append([meas[s] for s in ARM_SRC])
        if not samples:
            print(f"[{name}] ❌ /joint_states 미수신"); continue
        m_avg = np.mean(samples, axis=0)
        err = (m_avg - q) * 1000.0
        rows.append((name, err))
        print(f"[{name}] 실측−지령 [mrad]: "
              + "  ".join(f"{c}:{e:+.1f}" for c, e in zip(CANON, err)))

    if rows:
        avg = np.mean([e for _, e in rows], axis=0)
        print("\n자세 평균 [mrad]:  " + "  ".join(f"{c}:{e:+.1f}" for c, e in zip(CANON, avg)))
        comp = -avg / 1000.0
        print("선보상 후보(지령에 더할 값, rad): "
              + ",".join(f"{v:+.4f}" for v in comp))
    node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
