#!/usr/bin/env python3
"""테솔로 우손 **20관절 실제 가동범위 전수 측정** — sim·정책·제어기가 공유할 치수.

왜 필요한가(08.31 실측). URDF 한계와 **실제로 갈 수 있는 곳이 다르다**:

  검지 `rj_dg_2_4`: 지령 +0.250 → 실측 +0.204 (82%), effort 8   ← 정상
  엄지 `rj_dg_1_4`: 지령 +0.250 → 실측 +0.120 (48%), effort 62  ← 지령 0 조차 도달 불가

도달 못하는 지령을 받으면 컨트롤러는 **영원히 밀어붙인다** — 가만히 있는데 발열한다.
정책이 URDF 한계를 믿고 액션을 내면 실기에서 그 상태가 상시화된다. 그래서 실측
가동범위를 재서 **sim·프로필·정책 액션 범위 셋에 같은 값**을 넣어야 한다.

측정 방법: 각 관절을 한쪽 끝으로 천천히 밀고, **토크를 쓰는데 안 움직이면**(stall)
거기가 실제 끝이다. 반대쪽도 같게. 나머지 관절은 0(손 폄)으로 붙잡는다.

    python3 probe_hand_range.py --joints 1_4,2_4 --execute        # 일부만
    python3 probe_hand_range.py --all --out config/right_hand_range.yaml --execute

출력 yaml 은 관절별 `measured: [하한, 상한]` 과 URDF 대비 도달률을 담는다.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

STATE = "/dg5f_right/joint_states"
TEMP = "/dg5f_right/dynamic_joint_states"
TRAJ = "/dg5f_right/dg5f_right_controller/joint_trajectory"
PUBLISH_DT = 0.02
WAIT_SEC = 5.0
#: ★관절 한계는 **프로필에서 관절별로** 읽는다. 접미사로 뭉뚱그리면 안 된다 —
#  09.01 에 그렇게 했다가 12관절 측정을 버렸다. 엄지 `_2` 는 [-3.142, 0] 인데
#  검지·중지·약지 `_2` 는 [0, +2.0] 이고 소지 `_2` 는 [-0.419, +0.611] 이다.
#  탐색을 [-3.142, 0] 으로 하면 실하한이 0 인 관절은 **아예 움직이지 않는다**.
PROFILE_YAML = Path("/home/user/rl_ws/robot_control/src/robot_control/profiles"
                    "/openarm_tesollo.yaml")


def _limits() -> dict[str, tuple[float, float]]:
    """드라이버 이름(rj_dg_f_j) → (하한, 상한). 프로필이 진실이다."""
    import yaml

    body = yaml.safe_load(PROFILE_YAML.read_text())
    out = {}
    for joint in body["joints"]:
        if joint["canonical"].startswith("r_hj_") and "gripper" not in joint["canonical"]:
            out[joint["source"]] = (float(joint["lower"]), float(joint["upper"]))
    if len(out) != 20:
        raise SystemExit(f"손 관절이 20개가 아니다: {len(out)}")
    return out
FINGERS = {1: "thumb", 2: "index", 3: "middle", 4: "ring", 5: "pinky"}
ALL_JOINTS = [f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joints", default="1_4", help="측정할 관절, 쉼표 구분")
    parser.add_argument("--all", action="store_true", help="20관절 전부")
    parser.add_argument("--max-span", type=float, default=1.2,
                        help="한 방향 최대 탐색 거리[rad]. URDF 한계와 함께 더 좁은 쪽을 쓴다.")
    parser.add_argument("--speed", type=float, default=0.25, help="탐색 속도[rad/s]")
    parser.add_argument("--return-speed", type=float, default=0.35, help="0 복귀 속도[rad/s]")
    parser.add_argument("--stall-effort", type=float, default=40.0)
    parser.add_argument("--stall-sec", type=float, default=0.8,
                        help="이만큼[s] 토크를 쓰는데 안 움직이면 그 지점이 가동 끝")
    parser.add_argument("--stall-speed", type=float, default=0.01,
                        help="이 속도[rad/s] 미만이면 멈춘 것으로 본다")
    parser.add_argument("--settle", type=float, default=0.8)
    parser.add_argument("--max-temp", type=float, default=55.0,
                        help="이 온도[℃]를 넘은 관절이 있으면 측정을 멈춘다")
    parser.add_argument("--rest-sec", type=float, default=3.0,
                        help="관절 하나를 끝낼 때마다 쉬는 시간[s] — 발열 누적을 끊는다")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    targets = ALL_JOINTS if args.all else [
        s if s.strip().startswith("rj_dg_") else f"rj_dg_{s.strip()}"
        for s in args.joints.split(",")]

    import rclpy
    from control_msgs.msg import DynamicJointState
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    rclpy.init()
    node = Node("hand_range")
    pos: dict[str, float] = {}
    eff: dict[str, float] = {}
    tmp: dict[str, float] = {}

    def cb(msg: JointState) -> None:
        for i, n in enumerate(msg.name):
            pos[n] = float(msg.position[i])
            if msg.effort and i < len(msg.effort):
                eff[n] = float(msg.effort[i])

    def tcb(msg) -> None:
        for n, iv in zip(msg.joint_names, msg.interface_values):
            for iname, value in zip(iv.interface_names, iv.values):
                if iname == "temperature":
                    tmp[n] = float(value)

    node.create_subscription(JointState, STATE, cb, 10)
    node.create_subscription(DynamicJointState, TEMP, tcb, 10)
    deadline = time.monotonic() + WAIT_SEC
    while not pos and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not pos:
        print(f"❌ {STATE} 를 못 받았다 — 손 드라이버 확인")
        return 1

    names = sorted(pos)
    unknown = [t for t in targets if t not in names]
    if unknown:
        print(f"❌ 없는 관절: {unknown}")
        return 1

    est = len(targets) * (2 * args.max_span / args.speed + 4 * args.settle)
    print(f"측정 대상 {len(targets)} 관절 · 방향당 최대 {args.max_span} rad @ {args.speed} rad/s")
    print(f"가동 끝 판정: effort > {args.stall_effort:.0f} 을 {args.stall_sec}s 쓰는데 "
          f"{args.stall_speed} rad/s 미만")
    print(f"예상 소요 최대 {est/60:.1f} 분 (일찍 멈추면 그만큼 짧다)")
    if not args.execute:
        print("\nDRY RUN — 실제로 측정하려면 --execute")
        return 0

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

    def go(joint: str, start: float, end: float, speed: float, zero_rest: bool):
        """joint 를 start→end 로 밀며 stall 을 감지한다. (도달 실측, 사유) 반환."""
        steps = max(1, int(math.ceil(abs(end - start) / max(speed * PUBLISH_DT, 1e-9))))
        rest = {n: 0.0 for n in names} if zero_rest else {n: pos[n] for n in names}
        hot_t: float | None = None
        hot_p = 0.0
        for i in range(steps):
            goal = dict(rest)
            goal[joint] = start + (end - start) * ((i + 1) / steps)
            send(goal)
            rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
            e = abs(eff.get(joint, 0.0))
            p = pos.get(joint, 0.0)
            now = time.monotonic()
            if e <= args.stall_effort:
                hot_t = None
                continue
            if hot_t is None:
                hot_t, hot_p = now, p
                continue
            held = now - hot_t
            if held < args.stall_sec:
                continue
            moved = abs(p - hot_p) / max(held, 1e-9)
            if moved >= args.stall_speed:
                hot_t, hot_p = now, p
                continue
            return p, f"stall (effort {e:.0f}, {held:.1f}s 동안 {math.degrees(abs(p-hot_p)):.2f}°)"
        # 끝까지 갔다 — 정착시켜 실측을 확정한다
        for _ in range(int(args.settle / PUBLISH_DT)):
            goal = dict(rest)
            goal[joint] = end
            send(goal)
            rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
        return pos.get(joint, 0.0), "reached"

    print("\n[준비] 전 관절 0 으로")
    span0 = max(abs(v) for v in pos.values())
    steps0 = max(1, int(math.ceil(span0 / max(args.return_speed * PUBLISH_DT, 1e-9))))
    start0 = dict(pos)
    for i in range(steps0):
        f = (i + 1) / steps0
        send({n: start0[n] * (1.0 - f) for n in names})
        rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
    for _ in range(int(1.0 / PUBLISH_DT)):
        send({n: 0.0 for n in names})
        rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
    print(f"  완료 · 0 대비 최대 편차 "
          f"{math.degrees(max(abs(v) for v in pos.values())):.1f}°")

    limits = _limits()
    results: dict[str, dict] = {}
    for t in targets:
        lo_u, hi_u = limits[t]
        hi_goal = min(hi_u, args.max_span)
        lo_goal = max(lo_u, -args.max_span)
        print(f"\n[{t}] URDF [{lo_u:+.3f}, {hi_u:+.3f}] · 탐색 [{lo_goal:+.3f}, {hi_goal:+.3f}]",
              flush=True)

        hi_meas, hi_why = go(t, 0.0, hi_goal, args.speed, True)
        print(f"  상한 {hi_meas:+.4f} ({math.degrees(hi_meas):+.1f}°) — {hi_why}", flush=True)
        go(t, hi_meas, 0.0, args.return_speed, True)
        lo_meas, lo_why = go(t, 0.0, lo_goal, args.speed, True)
        print(f"  하한 {lo_meas:+.4f} ({math.degrees(lo_meas):+.1f}°) — {lo_why}", flush=True)
        go(t, lo_meas, 0.0, args.return_speed, True)

        # ★관절 하나마다 쉰다. 09.01 에 연속 측정으로 발열이 쌓여 드라이버를 껐다.
        if args.rest_sec > 0:
            send({n: pos[n] for n in names})       # 지령=실측 → 토크 이완
            deadline_rest = time.monotonic() + args.rest_sec
            while time.monotonic() < deadline_rest:
                rclpy.spin_once(node, timeout_sec=PUBLISH_DT)

        span_m = hi_meas - lo_meas
        span_u = min(hi_u, args.max_span) - max(lo_u, -args.max_span)
        results[t] = {"urdf": [round(lo_u, 4), round(hi_u, 4)],
                      "measured": [round(lo_meas, 4), round(hi_meas, 4)],
                      "reach_ratio": round(span_m / span_u, 3) if span_u > 0 else 0.0,
                      "hi_why": hi_why.split(" ")[0], "lo_why": lo_why.split(" ")[0],
                      "temp_c": round(tmp.get(t, float("nan")), 1)}
        print(f"  가동폭 {math.degrees(span_m):.1f}° / 탐색폭 {math.degrees(span_u):.1f}° "
              f"= {results[t]['reach_ratio']:.0%} · {results[t]['temp_c']}℃", flush=True)

        # ★온도 상한. 넘으면 지령을 실측에 맞춰 토크를 풀고 멈춘다.
        hot = {k: v for k, v in tmp.items() if v > args.max_temp}
        if hot:
            send({n: pos[n] for n in names})
            for _ in range(10):
                rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
            print(f"\n⛔ {args.max_temp:.0f}℃ 초과 — 측정 중단: "
                  + ", ".join(f"{k} {v:.1f}℃" for k, v in sorted(hot.items())))
            print("   식힌 뒤 남은 관절만 --joints 로 이어서 잴 것")
            break

    # 지령을 실측에 맞춰 토크를 푼다
    send({n: pos[n] for n in names})
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.02)

    print("\n═══ 요약 ═══")
    print(f"{'관절':12s} {'하한':>9s} {'상한':>9s} {'가동폭':>9s} {'도달률':>7s}")
    for t, r in results.items():
        lo, hi = r["measured"]
        print(f"{t:12s} {math.degrees(lo):+8.1f}° {math.degrees(hi):+8.1f}° "
              f"{math.degrees(hi-lo):8.1f}° {r['reach_ratio']:6.0%}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# 테솔로 우손 실측 가동범위 (probe_hand_range.py)",
                 f"# 측정: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                 "# URDF 한계가 아니라 **실제로 갈 수 있는 곳**이다. 도달 못하는 지령을 주면",
                 "# 컨트롤러가 영원히 밀어붙여 발열한다 — sim·프로필·정책 액션에 같은 값을 쓸 것.",
                 "joints:"]
        for t, r in results.items():
            lines += [f"  {t}:",
                      f"    urdf: [{r['urdf'][0]}, {r['urdf'][1]}]",
                      f"    measured: [{r['measured'][0]}, {r['measured'][1]}]",
                      f"    reach_ratio: {r['reach_ratio']}",
                      f"    hi_why: {r['hi_why']}",
                      f"    lo_why: {r['lo_why']}",
                      f"    temp_c: {r['temp_c']}"]
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
