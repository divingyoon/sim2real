#!/usr/bin/env python3
"""우팔을 지정 자세로 **램프해서 보내고 유지**한다 — 처짐 선보상 적용용.

preset 재생이 끝난 뒤 실기는 지령을 유지하지만 중력 처짐만큼 아래에 선다. 손이
테이블에 닿을 만큼 처지면 지령에 **처짐의 반대**를 더해야 실측이 목표에 온다.

    # 홈 + 선보상(처짐 반대 부호)으로 3초에 걸쳐 올린 뒤 유지
    python3 hold_right_arm.py --offset 0.052,0.075,0.030,0.068,0.093,0.002,0.134

    # 절대 목표를 직접 주기 (--offset 과 배타)
    python3 hold_right_arm.py --target 0.038,0.401,0.602,0.964,0.029,0.706,0.421

★현재 실측에서 목표까지 `--speed`(기본 0.1 rad/s)로 램프한다. 도약시키지 않는다.
★`--execute` 없으면 계획만 출력한다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


from jtc_bridge_core import JointRemap        # noqa: E402
from robot_profile import load_robot_profile  # noqa: E402

HOME = [0.0380, 0.4012, 0.6015, 0.9643, 0.0294, 0.7060, 0.4213]
PUBLISH_DT = 0.02


def _parse7(text: str) -> np.ndarray:
    vals = [float(v) for v in text.replace(" ", "").split(",")]
    if len(vals) != 7:
        raise SystemExit(f"7개 값이 필요하다 — 받은 {len(vals)}개")
    return np.array(vals, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--offset", type=str, help="홈에 더할 선보상 7값(rad, canonical 순서)")
    src.add_argument("--target", type=str, help="절대 목표 7값(rad)")
    parser.add_argument("--robot", default="tesollo_sensor__right")
    parser.add_argument("--speed", type=float, default=0.1, help="램프 속도[rad/s]")
    parser.add_argument("--hold", type=float, default=0.0,
                        help="도착 후 유지 시간[s]. 0 이면 무한(Ctrl-C 로 종료)")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    profile = load_robot_profile(args.robot)
    home = np.array(HOME, dtype=float)
    if args.target:
        goal = _parse7(args.target)
    elif args.offset:
        goal = home + _parse7(args.offset)
    else:
        goal = home
    print(f"목표(canonical): {np.round(goal, 4).tolist()}")
    if not args.execute:
        print("DRY RUN — 실제로 보내려면 --execute")
        return 0

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    canon = list(profile.arm_canonical)
    remap = JointRemap(canon, list(profile.arm_source), profile.joint_limits)

    class Hold(Node):
        def __init__(self) -> None:
            super().__init__("hold_right_arm")
            self.pub = self.create_publisher(
                Float64MultiArray, profile.topics["arm_cmd"], 10)
            self.traj = self.create_publisher(
                JointTrajectory, profile.topics["arm_traj"], 10)
            self.create_subscription(JointState, profile.topics["arm_state"],
                                     self._cb, qos_profile_sensor_data)
            self.measured = np.zeros(7)
            self.have = False
            self.setpoint: np.ndarray | None = None
            self.t_done: float | None = None
            self.create_timer(PUBLISH_DT, self._tick)

        def _cb(self, msg: JointState) -> None:
            idx = {n: i for i, n in enumerate(msg.name)}
            for k, src in enumerate(profile.arm_source):
                i = idx.get(src)
                if i is not None:
                    self.measured[k] = msg.position[i] * profile.joint_limits[canon[k]]["sign"]
            self.have = True

        def _tick(self) -> None:
            if not self.have:
                return
            if self.setpoint is None:
                self.setpoint = self.measured.copy()
                span = float(np.max(np.abs(goal - self.setpoint)))
                self.get_logger().info(
                    f"실측에서 목표까지 {np.degrees(span):.2f}° · "
                    f"{span/max(args.speed,1e-9):.1f} s 램프")
            step = args.speed * PUBLISH_DT
            delta = goal - self.setpoint
            self.setpoint = self.setpoint + np.clip(delta, -step, step)
            if np.max(np.abs(goal - self.setpoint)) < 1e-4 and self.t_done is None:
                self.t_done = time.monotonic()
                err = self.measured - goal
                self.get_logger().info(
                    "도착 · 관절별 실측−목표(°): "
                    + " ".join(f"{np.degrees(v):+.2f}" for v in err))
            self.pub.publish(Float64MultiArray(data=[float(v) for v in self.setpoint]))
            msg = JointTrajectory()
            msg.joint_names = list(profile.arm_source)
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in remap.apply(self.setpoint)]
            pt.time_from_start.sec = 0
            pt.time_from_start.nanosec = 0
            msg.points = [pt]
            self.traj.publish(msg)

    rclpy.init()
    node = Hold()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if args.hold > 0 and node.t_done and time.monotonic() - node.t_done > args.hold:
                break
    except KeyboardInterrupt:
        pass
    err = node.measured - goal
    print("최종 실측−목표(°): " + " ".join(f"{np.degrees(v):+.2f}" for v in err))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
