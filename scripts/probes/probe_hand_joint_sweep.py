#!/usr/bin/env python3
"""테솔로 우손 **관절별 개별 스윕** — 도달 범위·히스테리시스·막히는 지점을 잰다.

08.31 문제: 엄지 원위 `rj_dg_1_4` 가 자세와 무관하게 effort 38~79(나머지 19관절은
0~18)로 혼자 튄다. 지령 0 으로 가려다 +0.103 rad 에서 막혀 계속 밀어붙인다 —
가만히 있는데 발열한다. 손으로 만져 보니 **뻑뻑하지 않다**(사용자 확인) → 기계적
간섭이 아니라 **드라이버/캘리브** 문제이고, 유력한 후보는 homing offset 어긋남이다.

이 프로브는 한 관절씩 홀로 천천히 왕복시키며 (지령·실측·effort)를 남긴다. 읽는 법:

  · **도달 한계**: 지령을 올려도 실측이 안 따라오는 지점. 그 관절의 실제 가동 끝이다.
  · **오프셋**: 같은 손가락의 다른 관절과 도달 범위를 대조한다. 통째로 밀려 있으면
    homing offset 이다(예: 지령 0 인데 실측이 늘 +0.1).
  · **히스테리시스**: 올릴 때와 내릴 때 실측 차이 = 마찰.
  · **막힘**: 실측이 멈췄는데 effort 만 오르는 구간. 여기가 발열의 정체다.

★한 관절만 움직이고 나머지는 **붙잡는다**. 손가락은 서로 부딪히므로 전 관절을
  동시에 흔들면 무엇이 무엇을 막았는지 못 가른다.
★`--from-zero`(기본 켜짐): 스윕 전에 **전 관절 0(손 폄)** 으로 먼저 간다. 주먹에서
  재면 손가락이 서로 닿아 있어 "이 관절이 막혔다"와 "옆 손가락이 막았다"를 못 가른다.
  편 상태는 서로 떨어져 있어 관절 고유 특성만 남는다(사용자 지시, 08.31).
★온도를 함께 남긴다 — 발열이 어디서 오르는지가 이 트랙의 핵심 정보다.
★진폭은 작게 시작한다(기본 ±0.25 rad). 손가락은 서로 닿기 쉽다.

    python3 probe_hand_joint_sweep.py --joints 1_4,2_4 --amp 0.25 --execute
    python3 probe_hand_joint_sweep.py --joints 1_4 --amp 0.4 --out logs/hand_j14.csv --execute
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

STATE = "/dg5f_right/joint_states"
TEMP = "/dg5f_right/dynamic_joint_states"
TRAJ = "/dg5f_right/dg5f_right_controller/joint_trajectory"
PUBLISH_DT = 0.02
WAIT_SEC = 5.0
#: 08.31 실측 한계(sim2real 프로필). 스윕이 여기를 넘지 않게 자른다.
LIMITS = {"1": (-0.383972, 0.890118), "2": (-3.141593, 0.0),
          "3": (-1.570796, 1.570796), "4": (-1.570796, 1.570796)}


def _joint_name(spec: str) -> str:
    """`1_4` → `rj_dg_1_4`. 이미 전체 이름이면 그대로."""
    spec = spec.strip()
    return spec if spec.startswith("rj_dg_") else f"rj_dg_{spec}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joints", default="1_4",
                        help="스윕할 관절, 쉼표 구분. `1_4`(엄지 원위) 또는 `rj_dg_1_4`")
    parser.add_argument("--amp", type=float, default=0.25, help="편도 진폭[rad]")
    parser.add_argument("--speed", type=float, default=0.15, help="스윕 속도[rad/s]")
    parser.add_argument("--settle", type=float, default=1.0, help="반환점 정착[s]")
    parser.add_argument("--abort-effort", type=float, default=40.0,
                        help="|effort| 임계. ★단독으로는 중단시키지 않는다 — 이 손은 기동 "
                             "토크가 커서(실측 23~47) 막힘(79)과 크기로는 못 가른다. "
                             "--abort-effort-sec 만큼 **지속**될 때만 막힘으로 본다.")
    parser.add_argument("--abort-effort-sec", type=float, default=1.0,
                        help="★임계 초과가 이만큼[s] 지속되고 **그 동안 실측이 멈춰 있어야** "
                             "중단한다. 08.31 에 두 번 배웠다: 크기로 자르면 정상 기동을 "
                             "끊고, 지속시간만 보면 **느린 관절**을 막힘으로 오판한다 "
                             "(엄지는 지령의 46% 속도로 계속 가고 있었다). 막힘의 정의는 "
                             "'토크를 쓰는데 안 움직인다'다.")
    parser.add_argument("--stall-speed", type=float, default=0.01,
                        help="이 속도[rad/s] 미만이면 '멈춤'으로 본다. 지령 속도의 10% 수준.")
    parser.add_argument("--abort-track", type=float, default=0.45,
                        help="실측이 지령을 이만큼[rad] 뒤처지면 중단. ★엄지는 지령의 46% "
                             "속도라 정상 동작에서도 크게 뒤처진다(08.31) — 좁게 잡으면 "
                             "느린 관절을 끊는다. 실제 막힘은 stall 조건이 잡는다.")
    parser.add_argument("--abort-others", type=float, default=0.10,
                        help="★스윕하지 않는 관절이 시작 자세에서 이만큼[rad] 밀리면 중단 — "
                             "손가락끼리 부딪혔다는 신호다. 지령은 고정인데 실측이 움직였다면 "
                             "누가 밀었다는 것 말고는 설명이 없다.")
    parser.add_argument("--from-zero", action=argparse.BooleanOptionalAction, default=True,
                        help="스윕 전 전 관절 0(손 폄)으로 간다. 손가락이 서로 떨어져 "
                             "관절 고유 특성만 남는다. --no-from-zero 면 현재 자세 유지.")
    parser.add_argument("--open-speed", type=float, default=0.20,
                        help="전 관절 0 으로 갈 때 속도[rad/s]")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    targets = [_joint_name(s) for s in args.joints.split(",")]

    import rclpy
    from rclpy.node import Node
    from control_msgs.msg import DynamicJointState
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    rclpy.init()
    node = Node("hand_joint_sweep")
    state: dict[str, float] = {}
    effort: dict[str, float] = {}

    def cb(msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            state[name] = float(msg.position[i])
            if msg.effort and i < len(msg.effort):
                effort[name] = float(msg.effort[i])

    temp: dict[str, float] = {}

    def temp_cb(msg) -> None:
        for name, iv in zip(msg.joint_names, msg.interface_values):
            for iname, value in zip(iv.interface_names, iv.values):
                if iname == "temperature":
                    temp[name] = float(value)

    node.create_subscription(JointState, STATE, cb, 10)
    node.create_subscription(DynamicJointState, TEMP, temp_cb, 10)
    deadline = time.monotonic() + WAIT_SEC
    while not state and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not state:
        print(f"❌ {STATE} 를 못 받았다 — 손 드라이버 확인")
        return 1

    names = sorted(state)
    missing = [t for t in targets if t not in state]
    if missing:
        print(f"❌ 없는 관절: {missing}\n   가능한 이름: {names}")
        return 1

    base = {n: state[n] for n in names}
    # --from-zero 면 스윕 기준점은 0 이다. 계획 출력도 그 기준으로 보여야 한다.
    plan_base = {n: 0.0 for n in names} if args.from_zero else dict(base)
    print(f"스윕 기준 {'전 관절 0(손 폄)' if args.from_zero else '현재 자세'}"
          f" · {targets} · ±{args.amp} rad @ {args.speed} rad/s")
    for t in targets:
        lo, hi = LIMITS[t.split("_")[-1]]
        span = (max(lo, plan_base[t] - args.amp), min(hi, plan_base[t] + args.amp))
        print(f"  {t}: 현재 {base[t]:+.3f} → 기준 {plan_base[t]:+.3f} "
              f"→ 스윕 [{span[0]:+.3f}, {span[1]:+.3f}]  한계[{lo:+.2f},{hi:+.2f}]")
    if args.from_zero:
        span = max(abs(v) for v in base.values())
        print(f"  ※ 먼저 전 관절 0 으로 이동한다 (최대 {math.degrees(span):.0f}°, "
              f"{span / max(args.open_speed, 1e-9):.1f} s)")
    print(f"안전 중단: effort > {args.abort_effort:.0f} 을 {args.abort_effort_sec}s 쓰는데 "
          f"{args.stall_speed} rad/s 미만으로 **안 움직임** · "
          f"추종 {args.abort_track} rad · 다른 관절 밀림 {args.abort_others} rad")
    if not args.execute:
        print("DRY RUN — 실제로 보내려면 --execute")
        return 0

    pub = node.create_publisher(JointTrajectory, TRAJ, 10)
    rows: list[dict] = []

    def send(goal: dict[str, float]) -> None:
        msg = JointTrajectory()
        msg.joint_names = names
        pt = JointTrajectoryPoint()
        pt.positions = [goal[n] for n in names]
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 0
        msg.points = [pt]
        pub.publish(msg)

    def record(joint: str, cmd: float, phase: str) -> None:
        rows.append({"t": time.monotonic(), "joint": joint, "phase": phase,
                     "cmd": cmd, "meas": state.get(joint, float("nan")),
                     "effort": effort.get(joint, 0.0),
                     "temp": temp.get(joint, float("nan")),
                     **{f"meas_{n}": state.get(n, float("nan")) for n in names},
                     **{f"temp_{n}": temp.get(n, float("nan")) for n in names}})

    aborted: list[str] = []

    def safe_stop(reason: str) -> None:
        """지령을 **실측에 맞춰** 보내 버티는 토크를 없앤다. 그 자리에 선다."""
        aborted.append(reason)
        print(f"\n  ❌ 중단: {reason}", flush=True)
        for _ in range(10):
            send({n: state.get(n, base[n]) for n in names})
            rclpy.spin_once(node, timeout_sec=0.02)

    hold_ref: dict[str, float] = dict(base)   # 스윕 시작 시점의 **실측** (충돌 판정 기준)
    hot_since: dict[str, tuple[float, float]] = {}   # name -> (시작시각, 그때의 실측)

    def _effort_stuck(name: str) -> str | None:
        """**토크를 쓰는데 안 움직이면** 막힘. 느리게라도 가고 있으면 통과시킨다."""
        eff = abs(effort.get(name, 0.0))
        now = time.monotonic()
        pos = state.get(name, 0.0)
        if eff <= args.abort_effort:
            hot_since.pop(name, None)
            return None
        since, pos0 = hot_since.setdefault(name, (now, pos))
        held = now - since
        if held < args.abort_effort_sec:
            return None
        speed = abs(pos - pos0) / max(held, 1e-9)
        if speed >= args.stall_speed:
            # 느릴 뿐 움직이고 있다 — 창을 다시 열어 계속 감시한다.
            hot_since[name] = (now, pos)
            return None
        return (f"{name} effort {eff:.0f} 을 {held:.1f}s 쓰는데 "
                f"{math.degrees(abs(pos - pos0)):.2f}° 밖에 안 움직였다 — 막혔다")

    def check(joint: str, cmd: float) -> bool:
        """계속 가도 되는가. 셋 중 하나라도 걸리면 즉시 멈춘다."""
        for name in names:
            reason = _effort_stuck(name)
            if reason:
                safe_stop(reason)
                return False
        lag = abs(state.get(joint, cmd) - cmd)
        if lag > args.abort_track:
            safe_stop(f"{joint} 실측이 지령을 {math.degrees(lag):.1f}° 뒤처진다 — 막혔다")
            return False
        for other in names:
            if other == joint:
                continue
            # ★기준은 **이 스윕이 시작될 때의 실측**이다. 지령값을 기준으로 삼으면
            #   지령을 못 지키는 관절(엄지는 지령의 절반만 도달한다)이 늘 "밀렸다"로
            #   잡힌다 — 08.31 에 그렇게 오탐이 났다. 물어야 할 것은 "지금 이 관절을
            #   움직이는 동안 저 관절이 **실제로 움직였는가**"다.
            drift = abs(state.get(other, hold_ref[other]) - hold_ref[other])
            if drift > args.abort_others:
                safe_stop(f"{other} 가 {joint} 스윕 중 {math.degrees(drift):.1f}° 움직였다 "
                          f"— 부딪힌 것으로 보인다")
                return False
        return True

    def ramp(joint: str, start: float, end: float, phase: str) -> None:
        n = max(1, int(math.ceil(abs(end - start) / max(args.speed * PUBLISH_DT, 1e-9))))
        goal = dict(base)
        for i in range(n):
            if aborted:
                return
            goal[joint] = start + (end - start) * ((i + 1) / n)
            send(goal)
            rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
            record(joint, goal[joint], phase)
            if not check(joint, goal[joint]):
                return

    def hold(joint: str, value: float, phase: str) -> None:
        goal = dict(base)
        goal[joint] = value
        for _ in range(max(1, int(args.settle / PUBLISH_DT))):
            if aborted:
                return
            send(goal)
            rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
            record(joint, value, phase)
            if not check(joint, value):
                return

    if args.from_zero:
        span = max(abs(v) for v in base.values())
        steps = max(1, int(math.ceil(span / max(args.open_speed * PUBLISH_DT, 1e-9))))
        print(f"\n[준비] 전 관절 0 으로 {steps * PUBLISH_DT:.1f} s 램프 "
              f"(최대 {math.degrees(span):.0f}° 이동)", flush=True)
        start = dict(base)
        for i in range(steps):
            f = (i + 1) / steps
            goal = {n: start[n] * (1.0 - f) for n in names}
            send(goal)
            rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
            record("(open)", 0.0, "open")
            stuck = next((r for r in (_effort_stuck(n) for n in names) if r), None)
            if stuck:
                safe_stop(f"펴는 중 — {stuck}")
                break
        for _ in range(int(1.5 / PUBLISH_DT)):
            if aborted:
                break
            send({n: 0.0 for n in names})
            rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
            record("(open)", 0.0, "open_hold")
        if not aborted:
            base = {n: 0.0 for n in names}
            err = {n: state.get(n, 0.0) for n in names}
            worst = max(err, key=lambda n: abs(err[n]))
            print(f"  펴짐 완료 · 0 대비 최대 편차 {worst} {math.degrees(err[worst]):+.1f}°"
                  f" · 온도 {min(temp.values(), default=0):.0f}~{max(temp.values(), default=0):.0f}℃",
                  flush=True)

    for t in targets:
        if aborted:
            print(f"[{t}] 건너뜀 — 앞에서 중단됐다")
            continue
        hold_ref.clear()
        hold_ref.update({n: state.get(n, base[n]) for n in names})
        lo, hi = LIMITS[t.split("_")[-1]]
        up = min(hi, base[t] + args.amp)
        dn = max(lo, base[t] - args.amp)
        print(f"\n[{t}] +{up - base[t]:.3f} → {dn - base[t]:.3f} → 복귀", flush=True)
        ramp(t, base[t], up, "up"); hold(t, up, "up_hold")
        ramp(t, up, dn, "down"); hold(t, dn, "down_hold")
        ramp(t, dn, base[t], "back"); hold(t, base[t], "back_hold")

        seg = [r for r in rows if r["joint"] == t]
        for phase, label in (("up_hold", "상단"), ("down_hold", "하단"), ("back_hold", "복귀")):
            tail = [r for r in seg if r["phase"] == phase][-10:]
            if not tail:
                continue
            err = np.mean([r["meas"] - r["cmd"] for r in tail])
            eff = np.mean([abs(r["effort"]) for r in tail])
            tc = np.nanmean([r["temp"] for r in tail])
            print(f"  {label}: 지령 {tail[-1]['cmd']:+.3f} 실측 {tail[-1]['meas']:+.3f} "
                  f"오차 {math.degrees(err):+6.2f}° · |effort| {eff:.1f} · {tc:.1f}℃")
        ups = [r for r in seg if r["phase"] == "up_hold"][-10:]
        dns = [r for r in seg if r["phase"] == "down_hold"][-10:]
        if ups and dns:
            hyst = np.mean([r["meas"] - r["cmd"] for r in ups]) - \
                   np.mean([r["meas"] - r["cmd"] for r in dns])
            print(f"  히스테리시스(마찰) {math.degrees(hyst):+.2f}°")

    # 시작 자세로 되돌리고 지령=실측으로 맞춰 버티는 토크를 없앤다
    send({n: state[n] for n in names})
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.02)
    print("\n종료 — 지령을 실측에 맞춰 토크를 풀었다"
          + (f"\n★중단 사유: {aborted[0]}" if aborted else ""))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"→ {args.out}  ({len(rows)} 행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
