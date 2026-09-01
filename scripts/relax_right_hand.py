#!/usr/bin/env python3
"""테솔로 우손의 **버티는 토크를 없앤다** — 실측 자세를 그대로 지령해 오차를 0으로.

JTC 는 지령과 실측의 차이를 없애려 계속 전류를 쓴다. 도달 불가능한 지령(기계적으로
막힌 자세)을 주면 영원히 못 좁히고 **가만히 있는데 발열**한다. 08.31 실측: 엄지 원위
`rj_dg_1_4` 가 지령 0 · 실측 +0.103 rad 에서 effort **38**(나머지 19관절은 0~8).

이 스크립트는 현재 실측을 그대로 지령해 그 싸움을 끝낸다. 자세는 그 자리에 남는다.

    python3 relax_right_hand.py            # 확인만
    python3 relax_right_hand.py --execute  # 실측을 지령으로

`--joint rj_dg_1_4` 로 특정 관절만 완화할 수도 있다(나머지는 마지막 지령 유지).
"""

from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

STATE = "/dg5f_right/joint_states"
TRAJ = "/dg5f_right/dg5f_right_controller/joint_trajectory"
WAIT = 5.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--joint", action="append", default=None,
                        help="완화할 관절만 지정(반복 가능). 없으면 전 관절.")
    parser.add_argument("--repeat", type=int, default=10,
                        help="발행 횟수 — JTC 가 한 점을 놓칠 때를 대비")
    args = parser.parse_args()

    rclpy.init()
    node = Node("relax_right_hand")
    got: dict[str, float] = {}
    eff: dict[str, float] = {}

    def cb(msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            got[name] = float(msg.position[i])
            if msg.effort and i < len(msg.effort):
                eff[name] = float(msg.effort[i])

    node.create_subscription(JointState, STATE, cb, 10)
    deadline = time.monotonic() + WAIT
    while not got and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not got:
        print(f"❌ {STATE} 를 못 받았다 — 손 드라이버 확인")
        return 1

    names = sorted(got)
    hot = sorted(eff.items(), key=lambda kv: -abs(kv[1]))[:5]
    print("현재 |effort| 상위: " + ", ".join(f"{k} {v:+.0f}" for k, v in hot))
    print(f"완화 대상: {args.joint if args.joint else '전 관절'}")
    if not args.execute:
        print("DRY RUN — 실제로 보내려면 --execute")
        return 0

    pub = node.create_publisher(JointTrajectory, TRAJ, 10)
    msg = JointTrajectory()
    msg.joint_names = names
    pt = JointTrajectoryPoint()
    pt.positions = [got[n] for n in names]
    # ★0 이어야 한다 — 미래 시각 포인트는 interpolation_method:none 에서 영영 안 쓰인다.
    pt.time_from_start.sec = 0
    pt.time_from_start.nanosec = 0
    msg.points = [pt]
    for _ in range(args.repeat):
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)
    time.sleep(1.0)
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.05)
    hot2 = sorted(eff.items(), key=lambda kv: -abs(kv[1]))[:5]
    print("완화 후 |effort| 상위: " + ", ".join(f"{k} {v:+.0f}" for k, v in hot2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
