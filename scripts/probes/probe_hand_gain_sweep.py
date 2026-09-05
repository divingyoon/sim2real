#!/usr/bin/env python3
"""손 JTC PID 게인을 바꿔가며 **추종 성능**을 잰다.

왜 필요한가(09.01 실측). 손 관절에 `0.15 rad/s` 로 `1.2 rad` 지령을 보내고 0.8 s
정착시켰더니 7관절이 **전부 +54~56°** 에서 멈췄다. 우연이 아니라 **제어기가 지령을
못 따라간 것**이다 — 같은 관절을 5 s 정착시키니 +63.8° 까지 갔다(81 % → 92 %).

즉 지금까지 "가동범위"로 잰 값은 관절 한계가 아니라 **제어기 성능**이다. 그래서
게인을 먼저 세우고, 그 뒤에 범위를 재야 한다.

기본 게인은 20관절 전부 `p=1.5 · i=0 · **d=0**`
(`delto_m_ros2/dg5f_driver/config/dg5f_right_controller.yaml`). JTC 는 이 PID 로
position 오차에서 **effort** 를 만든다(팔은 하드웨어가 MIT 모드로 kp/kd 를 쓴다 —
구조가 다르다). 런타임에 `ros2 param set` 으로 바꿀 수 있다.

    python3 probe_hand_gain_sweep.py --joint 4_3 --p 1.5,3,6,12 --execute
    python3 probe_hand_gain_sweep.py --joint 4_3 --p 6 --d 0,0.05,0.1 --execute

각 게인마다: 0 → target 램프 → 정착 → **도달률·정착시간·최대 effort·온도**.
★게인을 원래대로 되돌리고 끝난다. 중간에 죽어도 finally 에서 복원한다.
"""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import time

STATE = "/dg5f_right/joint_states"
TEMP = "/dg5f_right/dynamic_joint_states"
TRAJ = "/dg5f_right/dg5f_right_controller/joint_trajectory"
CTRL = "/dg5f_right/dg5f_right_controller"
PUBLISH_DT = 0.02
#: 이 온도[℃]를 넘으면 스윕을 멈춘다.
MAX_TEMP = 55.0
#: 이 effort 를 넘으면 그 게인은 위험하다고 보고 멈춘다.
MAX_EFFORT = 900.0
#: ★정착 후 위치 표준편차가 이 값을 넘으면 **진동**으로 본다. 게인을 더 올리지 않는다.
#  09.01 실측: 손이 가만히 있을 때의 기저 σ 가 0.09° 다(엔코더 노이즈). 그 위로 잡는다.
VIB_SIGMA_DEG = 0.20
#: 채택 조건 — 이 도달률 미만이면 후보에서 뺀다.
REACH_MIN = 0.90


def _set_param(name: str, value: float) -> bool:
    out = subprocess.run(["ros2", "param", "set", CTRL, name, str(value)],
                         capture_output=True, text=True, timeout=25)
    return "Set parameter successful" in out.stdout


def _get_param(name: str) -> float | None:
    out = subprocess.run(["ros2", "param", "get", CTRL, name],
                         capture_output=True, text=True, timeout=25)
    for token in out.stdout.split():
        try:
            return float(token)
        except ValueError:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joint", default="4_3", help="rj_dg_ 접두사 없이. 예 4_3")
    parser.add_argument("--p", default="1.5,3,6,12", help="시험할 p 게인, 쉼표 구분")
    parser.add_argument("--d", default=None, help="시험할 d 게인. 주면 p 는 첫 값 고정")
    parser.add_argument("--target", type=float, default=1.0, help="지령 각도[rad]")
    parser.add_argument("--ramp", type=float, default=3.0, help="지령 램프 시간[s]")
    parser.add_argument("--settle", type=float, default=3.0, help="정착 관찰[s]")
    parser.add_argument("--rest", type=float, default=3.0, help="게인 사이 휴식[s]")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    joint = args.joint if args.joint.startswith("rj_dg_") else f"rj_dg_{args.joint}"
    p_list = [float(x) for x in args.p.split(",")]
    d_list = [float(x) for x in args.d.split(",")] if args.d else [None]
    combos = ([(p_list[0], d) for d in d_list] if args.d
              else [(p, None) for p in p_list])

    print(f"관절 {joint} · 지령 0 → {args.target:+.2f} rad ({math.degrees(args.target):+.0f}°) "
          f"· 램프 {args.ramp}s · 정착 {args.settle}s")
    print(f"시험 조합 {len(combos)}: " + ", ".join(
        (f"p={p} d={d}" if d is not None else f"p={p}") for p, d in combos))
    if not args.execute:
        print("\nDRY RUN — 실제로 하려면 --execute")
        return 0

    import rclpy
    from control_msgs.msg import DynamicJointState
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    rclpy.init()
    node = Node("hand_gain_sweep")
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

    node.create_subscription(JointState, STATE, jcb, 10)
    node.create_subscription(DynamicJointState, TEMP, tcb, 10)
    deadline = time.monotonic() + 5.0
    while not pos and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    if joint not in pos:
        print(f"❌ {joint} 상태를 못 받았다")
        return 1
    names = sorted(pos)
    pub = node.create_publisher(JointTrajectory, TRAJ, 10)

    def send(goal: dict[str, float]) -> None:
        msg = JointTrajectory()
        msg.joint_names = names
        pt = JointTrajectoryPoint()
        pt.positions = [goal[n] for n in names]
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 0
        msg.points = [pt]
        pub.publish(msg)

    def ramp_to(value: float, seconds: float) -> tuple[float, float, float]:
        """지령을 선형으로 올리며 최대 effort 를 본다. (도달, 최대effort, 소요) 반환."""
        start = pos[joint]
        steps = max(1, int(seconds / PUBLISH_DT))
        worst = 0.0
        for i in range(steps):
            rest = {n: pos[n] for n in names}
            rest[joint] = start + (value - start) * ((i + 1) / steps)
            send(rest)
            rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
            worst = max(worst, abs(eff.get(joint, 0.0)))
        return pos[joint], worst, seconds

    original_p = _get_param(f"gains.{joint}.p")
    original_d = _get_param(f"gains.{joint}.d")
    print(f"원래 게인 p={original_p} d={original_d} — 끝나면 되돌린다\n")
    rows = []
    try:
        print(f"{'p':>6s} {'d':>6s} {'도달률':>6s} {'정착후':>9s} "
              f"{'σ':>7s} {'p-p':>7s} {'eff':>7s} {'온도':>6s}")
        for p, d in combos:
            if not _set_param(f"gains.{joint}.p", p):
                print(f"❌ p={p} 설정 실패")
                break
            if d is not None and not _set_param(f"gains.{joint}.d", d):
                print(f"❌ d={d} 설정 실패")
                break
            # 0 으로 되돌리고 시작 — 매 조합이 같은 출발점을 갖도록
            ramp_to(0.0, 2.0)
            for _ in range(int(1.0 / PUBLISH_DT)):
                send({n: pos[n] if n != joint else 0.0 for n in names})
                rclpy.spin_once(node, timeout_sec=PUBLISH_DT)

            reached, worst, _ = ramp_to(args.target, args.ramp)
            # ★지령을 **고정**한 채 정착시키며 진동을 잰다. 09.01 에 이 항을 빼고
            #   도달률만 보고 p 를 8배 올렸다가 전 손가락이 진동했다 — 도달률이
            #   높다는 것은 빠르다는 뜻이고, 빠르면 진동한다. 도달률만 최적화하면
            #   **진동을 유발하는 방향으로 최적화**하게 된다.
            track = []
            for _ in range(int(args.settle / PUBLISH_DT)):
                send({n: pos[n] if n != joint else args.target for n in names})
                rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
                worst = max(worst, abs(eff.get(joint, 0.0)))
                track.append(pos[joint])
            settled = pos[joint]
            # 정착 후반부(뒤 60%)만 본다 — 앞쪽은 아직 접근 중이다.
            tail = track[int(len(track) * 0.4):] or track
            sigma = math.degrees(statistics.pstdev(tail)) if len(tail) > 1 else 0.0
            peak = math.degrees(max(tail) - min(tail)) if tail else 0.0
            temp = tmp.get(joint, float("nan"))
            rows.append((p, d, reached, settled, worst, temp, sigma, peak))
            flag = "⚠진동" if sigma > VIB_SIGMA_DEG else ""
            print(f"{p:6.2f} {(d if d is not None else original_d):6.3f} "
                  f"{reached/args.target:6.0%} {math.degrees(settled):+8.1f}° "
                  f"{sigma:7.3f}° {peak:7.3f}° {worst:7.0f} {temp:5.1f}℃ {flag}")
            if sigma > VIB_SIGMA_DEG:
                print(f"  ⛔ 진동 σ {sigma:.3f}° > {VIB_SIGMA_DEG}° — 이 게인 이상은 보지 않는다")
                break

            if worst > MAX_EFFORT:
                print(f"  ⛔ effort {worst:.0f} > {MAX_EFFORT} — 중단")
                break
            if temp == temp and temp > MAX_TEMP:
                print(f"  ⛔ {temp:.1f}℃ > {MAX_TEMP} — 중단")
                break
            for _ in range(int(args.rest / PUBLISH_DT)):
                send({n: pos[n] for n in names})
                rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
    finally:
        # ★게인을 반드시 되돌린다. 남겨두면 다음 사람이 다른 로봇을 만지게 된다.
        ramp_to(0.0, 2.0)
        send({n: pos[n] for n in names})
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
        if original_p is not None:
            _set_param(f"gains.{joint}.p", original_p)
        if original_d is not None:
            _set_param(f"gains.{joint}.d", original_d)
        print(f"\n게인 복원 p={original_p} d={original_d}")
        rclpy.shutdown()

    if rows:
        # ★판정은 "도달률 ≥ REACH_MIN 이면서 진동 σ 최소". 도달률 최대가 아니다.
        good = [r for r in rows if r[2] / args.target >= REACH_MIN
                and r[6] <= VIB_SIGMA_DEG]
        print(f"\n{'':6s} 판정 기준: 도달률 ≥{REACH_MIN:.0%} 이면서 σ ≤{VIB_SIGMA_DEG}°")
        if good:
            best = min(good, key=lambda r: r[6])
            print(f"★채택 p={best[0]} d={best[1] if best[1] is not None else original_d}"
                  f" → 도달률 {best[2]/args.target:.0%} · σ {best[6]:.3f}° · "
                  f"정착 {math.degrees(best[3]):+.1f}°")
        else:
            print("★조건을 만족하는 조합이 없다 — 범위를 바꿔 다시 볼 것")
            for r in rows:
                print(f"   p={r[0]} d={r[1]} 도달률 {r[2]/args.target:.0%} σ {r[6]:.3f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
