#!/usr/bin/env python3
"""손 **여러 관절을 동시에** 움직이며 진동을 본다 — 단일 관절 시험이 놓치는 것.

09.01 실패. 관절 하나로 p 를 훑어 "p=12 가 도달률 97 %" 라고 정하고 20관절에 걸었더니
**전 손가락이 진동**했다. 단일 관절에서는 σ 가 0 이었다 — 진동은 손가락 간 커플링에서
나온다. 그래서 게인을 확정하기 전에 **실제 사용 자세로 동시에** 움직여야 한다.

동작: 전 관절 0 → 주먹 자세로 램프 → 유지하며 위치 σ·effort·온도 측정 → 0 복귀.
게인은 끝나거나 죽어도 finally 에서 되돌린다.

    python3 probe_hand_multi_gain.py --p 3,4.5,6 --execute
"""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import time
from pathlib import Path

import yaml

STATE = "/dg5f_right/joint_states"
TEMP = "/dg5f_right/dynamic_joint_states"
TRAJ = "/dg5f_right/dg5f_right_controller/joint_trajectory"
CTRL = "/dg5f_right/dg5f_right_controller"
FIST = Path("/home/user/rl_ws/sim2real/config/right_hand_fist.yaml")
PUBLISH_DT = 0.02
#: 기저 σ 는 0.09°(엔코더 노이즈). 그 위로 잡는다.
VIB_SIGMA_DEG = 0.20
MAX_TEMP = 55.0
MAX_EFFORT = 500.0


def _set_all(node, client, p: float, d: float) -> int:
    """★40개 파라미터를 **한 번의 서비스 호출**로 설정한다.

    `ros2 param set` 은 호출마다 CLI 프로세스를 띄워 40번이면 수십 초가 걸리고,
    09.01 에 25 s 타임아웃으로 스윕이 죽었다(게인이 중간 상태로 남을 뻔했다).
    """
    import rclpy
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters

    req = SetParameters.Request()
    for f in range(1, 6):
        for j in range(1, 5):
            for key, value in (("p", p), ("d", d)):
                req.parameters.append(Parameter(
                    name=f"gains.rj_dg_{f}_{j}.{key}",
                    value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                         double_value=float(value))))
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=20.0)
    result = future.result()
    if result is None:
        return 0
    return sum(1 for r in result.results if r.successful)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", default="3,4.5,6")
    parser.add_argument("--d", default="0")
    parser.add_argument("--ramp", type=float, default=4.0)
    parser.add_argument("--hold", type=float, default=4.0)
    parser.add_argument("--rest", type=float, default=3.0)
    parser.add_argument("--fraction", type=float, default=0.6,
                        help="주먹 자세의 몇 할까지 갈지 — 1.0 은 완전 주먹")
    parser.add_argument("--record", type=Path, default=None,
                        help="지령·실측 시계열을 npz 로 남긴다 — sim 정합의 입력")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    p_list = [float(x) for x in args.p.split(",")]
    d_list = [float(x) for x in args.d.split(",")]
    combos = [(p, d) for p in p_list for d in d_list]
    fist = yaml.safe_load(FIST.read_text())["joints"]

    print(f"동시 시험 {len(combos)} 조합 · 주먹의 {args.fraction:.0%} 까지 "
          f"· 램프 {args.ramp}s · 유지 {args.hold}s")
    print(f"진동 판정 σ > {VIB_SIGMA_DEG}° (기저 0.09°)")
    if not args.execute:
        print("\nDRY RUN — 실제로 하려면 --execute")
        return 0

    import rclpy
    from control_msgs.msg import DynamicJointState
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    rclpy.init()
    node = Node("hand_multi_gain")
    from rcl_interfaces.srv import SetParameters

    param_client = node.create_client(
        SetParameters, "/dg5f_right/dg5f_right_controller/set_parameters")
    if not param_client.wait_for_service(timeout_sec=10.0):
        print("❌ set_parameters 서비스가 없다")
        return 1
    pos: dict[str, float] = {}
    eff: dict[str, float] = {}
    tmp: dict[str, float] = {}

    def jcb(msg: JointState) -> None:
        for i, n in enumerate(msg.name):
            pos[n] = float(msg.position[i])
            if msg.effort and i < len(msg.effort):
                eff[n] = float(msg.effort[i])

    def tcb(msg) -> None:
        for n, iv in zip(msg.joint_names, msg.interface_values):
            for iname, value in zip(iv.interface_names, iv.values):
                if iname == "temperature":
                    tmp[n] = float(value)

    node.create_subscription(JointState, STATE, jcb, 50)
    node.create_subscription(DynamicJointState, TEMP, tcb, 10)
    deadline = time.monotonic() + 5.0
    while len(pos) < 20 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if len(pos) < 20:
        print(f"❌ 상태 {len(pos)}/20 — 드라이버 확인")
        return 1
    names = sorted(pos)
    goal = {n: fist.get(n, 0.0) * args.fraction for n in names}
    pub = node.create_publisher(JointTrajectory, TRAJ, 10)

    def send(target: dict[str, float]) -> None:
        msg = JointTrajectory()
        msg.joint_names = names
        pt = JointTrajectoryPoint()
        pt.positions = [target[n] for n in names]
        msg.points = [pt]
        pub.publish(msg)

    def ramp(frm: dict[str, float], to: dict[str, float], seconds: float,
             rec: tuple[list, list] | None = None) -> None:
        steps = max(1, int(seconds / PUBLISH_DT))
        for i in range(steps):
            a = (i + 1) / steps
            target = {n: frm[n] * (1 - a) + to[n] * a for n in names}
            send(target)
            rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
            if rec is not None:
                rec[0].append([target[n] for n in names])
                rec[1].append([pos[n] for n in names])

    zero = {n: 0.0 for n in names}
    rows = []
    try:
        print(f"\n{'p':>6s} {'d':>6s} {'최대σ':>8s} {'관절':>8s} {'평균σ':>8s} "
              f"{'최대eff':>8s} {'온도':>6s}")
        for p, d in combos:
            got = _set_all(node, param_client, p, d)
            if got != 40:
                print(f"❌ 게인 설정 {got}/40")
                break
            rec_cmd: list[list[float]] = []
            rec_meas: list[list[float]] = []
            ramp(dict(pos), zero, 2.0)
            ramp(dict(pos), goal, args.ramp,
                 (rec_cmd, rec_meas) if args.record else None)

            track: dict[str, list[float]] = {n: [] for n in names}
            worst_eff = 0.0
            for _ in range(int(args.hold / PUBLISH_DT)):
                send(goal)
                rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
                for n in names:
                    track[n].append(pos[n])
                worst_eff = max(worst_eff, max(abs(v) for v in eff.values()) if eff else 0.0)
                if args.record:
                    rec_cmd.append([goal[n] for n in names])
                    rec_meas.append([pos[n] for n in names])

            sig = {}
            for n in names:
                tail = track[n][int(len(track[n]) * 0.4):] or track[n]
                sig[n] = math.degrees(statistics.pstdev(tail)) if len(tail) > 1 else 0.0
            worst_j = max(sig, key=sig.get)
            temp = max(tmp.values()) if tmp else float("nan")
            rows.append((p, d, sig[worst_j], worst_j, statistics.mean(sig.values()),
                         worst_eff, temp))
            flag = "⚠진동" if sig[worst_j] > VIB_SIGMA_DEG else ""
            print(f"{p:6.2f} {d:6.3f} {sig[worst_j]:7.3f}° "
                  f"{worst_j.replace('rj_dg_',''):>8s} {statistics.mean(sig.values()):7.3f}° "
                  f"{worst_eff:8.0f} {temp:5.1f}℃ {flag}")

            if args.record:
                import numpy as np

                out = args.record.with_name(
                    f"{args.record.stem}_p{p:g}_d{d:g}{args.record.suffix}")
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out, command=np.array(rec_cmd, dtype=np.float32),
                    measured=np.array(rec_meas, dtype=np.float32),
                    joint_names=np.array(names), dt=np.array([PUBLISH_DT]),
                    gain_p=np.array([p]), gain_d=np.array([d]),
                    fraction=np.array([args.fraction]))
                print(f"       기록 → {out.name} ({len(rec_cmd)} 프레임)")

            if sig[worst_j] > VIB_SIGMA_DEG:
                print("  ⛔ 진동 — 여기서 멈춘다")
                break
            if worst_eff > MAX_EFFORT or (temp == temp and temp > MAX_TEMP):
                print(f"  ⛔ effort {worst_eff:.0f} / {temp:.1f}℃ — 멈춘다")
                break
            ramp(dict(pos), zero, 2.0)
            for _ in range(int(args.rest / PUBLISH_DT)):
                send(zero)
                rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
    finally:
        ramp(dict(pos), zero, 2.5)
        send({n: pos[n] for n in names})
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
        _set_all(node, param_client, 1.5, 0.0)
        print("\n게인 복원 p=1.5 d=0.0 · 손 0 자세")
        rclpy.shutdown()

    ok = [r for r in rows if r[2] <= VIB_SIGMA_DEG]
    if ok:
        best = max(ok, key=lambda r: r[0])
        print(f"\n★진동 없이 쓸 수 있는 최대 게인: p={best[0]} d={best[1]} "
              f"(최대 σ {best[2]:.3f}° · eff {best[5]:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
