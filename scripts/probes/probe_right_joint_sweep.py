#!/usr/bin/env python3
"""우팔 관절별 **양방향 준정적 스윕** — kp·마찰을 분리 식별한다.

08.31 preset 재생에서 "올릴 때와 내릴 때 격차가 다르다"(사용자 관찰)를 확인했다.
방향 차이는 **마찰**, 방향 무관 성분은 **중력/kp 부족**이다. 둘은 고치는 노브가 다르다:

  히스테리시스 h  →  Fc ≈ kp · h/2      (마찰 토크)
  방향 무관 e     →  τ_grav ≈ kp · e    (중력 모멘트, tau_ff=0 이라 전부 오차로 버틴다)

한 관절씩 홀로 천천히 올렸다 내리며 (지령·실측·effort)를 기록한다. 다른 관절은
시작 자세에 고정한다 — 커플링을 섞으면 관절별 분리가 안 된다.

★안전: 시작 자세는 **팔을 든 자세**여야 한다. 차렷 근처에서 j2/j4 를 스윕하면 하한을
  뚫거나 몸통에 닿는다(07.29 실측: 차렷 근방 스윕 거부됨).
★속도는 준정적(기본 0.05 rad/s). 빠르면 damping 이 섞여 kp 를 못 가른다.

    python3 probe_right_joint_sweep.py --joints 5,6,7 --amp 0.3 --execute
    python3 probe_right_joint_sweep.py --joints 2,3,4 --amp 0.25 --execute --out sweep.csv
"""

from __future__ import annotations

import argparse
import csv
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

#: 스윕 시작 자세 — R② 경유점(팔을 접어 올린 곳). sim 에서 몸통·테이블 무접촉 검증됨.
SAFE_POSE = [0.0, 0.9, 0.0, 2.0, 0.0, 0.0, 0.0]
PUBLISH_DT = 0.02
#: 벤더 control_gains.yaml (v10). 식별값과 대조하기 위한 참조일 뿐 여기서 쓰진 않는다.
VENDOR_KP = [70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joints", default="5,6,7", help="스윕할 관절 번호(1..7), 쉼표 구분")
    parser.add_argument("--amp", type=float, default=0.30, help="편도 진폭[rad]")
    parser.add_argument("--speed", type=float, default=0.05, help="스윕 속도[rad/s] — 준정적")
    parser.add_argument("--approach-speed", type=float, default=0.15)
    parser.add_argument("--settle", type=float, default=1.5, help="반환점 정착[s]")
    parser.add_argument("--robot", default="tesollo_sensor__right")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    joints = [int(v) for v in args.joints.split(",")]
    if any(j < 1 or j > 7 for j in joints):
        raise SystemExit("--joints 는 1..7")
    profile = load_robot_profile(args.robot)
    canon = list(profile.arm_canonical)
    base = np.array(SAFE_POSE, dtype=float)

    print(f"시작 자세(안전) {base.tolist()}")
    print(f"스윕 관절 {joints} · 진폭 ±{args.amp} rad · {args.speed} rad/s (준정적)")
    for j in joints:
        lim = profile.joint_limits[canon[j - 1]]
        lo, hi = lim.get("lower"), lim.get("upper")
        c = base[j - 1]
        if lo is not None and (c - args.amp < lo or c + args.amp > hi):
            raise SystemExit(
                f"r_aj_{j}: 스윕 범위 [{c-args.amp:+.3f},{c+args.amp:+.3f}] 가 "
                f"한계 [{lo:+.3f},{hi:+.3f}] 밖 — --amp 를 줄이거나 SAFE_POSE 를 바꿀 것")
    est = sum(4 * args.amp / args.speed + 2 * args.settle for _ in joints)
    print(f"예상 소요 {est:.0f} s (+ 접근)")
    if not args.execute:
        print("DRY RUN — 실제로 보내려면 --execute")
        return 0

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    remap = JointRemap(canon, list(profile.arm_source), profile.joint_limits)
    rclpy.init()
    node = Node("right_joint_sweep")
    state = {"q": np.zeros(7), "eff": np.zeros(7), "have": False}

    def cb(msg: JointState) -> None:
        idx = {n: i for i, n in enumerate(msg.name)}
        for k, src in enumerate(profile.arm_source):
            i = idx.get(src)
            if i is None:
                continue
            sign = profile.joint_limits[canon[k]]["sign"]
            state["q"][k] = msg.position[i] * sign
            if msg.effort and i < len(msg.effort):
                state["eff"][k] = msg.effort[i]
        state["have"] = True

    node.create_subscription(JointState, profile.topics["arm_state"], cb,
                             qos_profile_sensor_data)
    pub = node.create_publisher(Float64MultiArray, profile.topics["arm_cmd"], 10)
    traj = node.create_publisher(JointTrajectory, profile.topics["arm_traj"], 10)

    def send(q: np.ndarray) -> None:
        pub.publish(Float64MultiArray(data=[float(v) for v in q]))
        msg = JointTrajectory()
        msg.joint_names = list(profile.arm_source)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in remap.apply(q)]
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 0
        msg.points = [pt]
        traj.publish(msg)

    deadline = time.monotonic() + 5.0
    while not state["have"] and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not state["have"]:
        raise SystemExit(f"{profile.topics['arm_state']} 를 못 받았다")

    rows: list[dict] = []

    def hold(q: np.ndarray, seconds: float, tag: str, jidx: int) -> None:
        n = max(1, int(seconds / PUBLISH_DT))
        for _ in range(n):
            send(q)
            rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
            rows.append(_row(q, tag, jidx))

    def _row(q: np.ndarray, tag: str, jidx: int) -> dict:
        return {"t": time.monotonic(), "phase": tag, "joint": jidx,
                **{f"cmd_{n}": float(q[k]) for k, n in enumerate(canon)},
                **{f"meas_{n}": float(state["q"][k]) for k, n in enumerate(canon)},
                **{f"eff_{n}": float(state["eff"][k]) for k, n in enumerate(canon)}}

    def ramp(q0: np.ndarray, q1: np.ndarray, speed: float, tag: str, jidx: int) -> None:
        span = float(np.max(np.abs(q1 - q0)))
        n = max(1, int(np.ceil(span / max(speed * PUBLISH_DT, 1e-9))))
        for i in range(n):
            q = q0 + (q1 - q0) * ((i + 1) / n)
            send(q)
            rclpy.spin_once(node, timeout_sec=PUBLISH_DT)
            rows.append(_row(q, tag, jidx))

    print("\n[접근] 현재 자세 → 안전 자세")
    ramp(state["q"].copy(), base, args.approach_speed, "approach", 0)
    hold(base, 2.0, "settle", 0)
    err = state["q"] - base
    print("  안전자세 도착 · 오차(°): " + " ".join(f"{np.degrees(v):+.2f}" for v in err))

    for j in joints:
        k = j - 1
        up_end = base.copy(); up_end[k] = base[k] + args.amp
        dn_end = base.copy(); dn_end[k] = base[k] - args.amp
        print(f"\n[r_aj_{j}] +{args.amp} → −{args.amp} → 복귀", flush=True)
        ramp(base.copy(), up_end, args.speed, f"j{j}_up", j)
        hold(up_end, args.settle, f"j{j}_up_hold", j)
        ramp(up_end, dn_end, args.speed, f"j{j}_down", j)
        hold(dn_end, args.settle, f"j{j}_down_hold", j)
        ramp(dn_end, base.copy(), args.speed, f"j{j}_back", j)
        hold(base, args.settle, f"j{j}_back_hold", j)
        # 즉석 요약 — 반환점 정착 오차가 곧 그 자세의 (중력+마찰) 합
        up = [r for r in rows if r["phase"] == f"j{j}_up_hold"][-10:]
        dn = [r for r in rows if r["phase"] == f"j{j}_down_hold"][-10:]
        eu = np.mean([r[f"meas_{canon[k]}"] - r[f"cmd_{canon[k]}"] for r in up])
        ed = np.mean([r[f"meas_{canon[k]}"] - r[f"cmd_{canon[k]}"] for r in dn])
        print(f"  상단 정착오차 {np.degrees(eu):+.2f}° · 하단 {np.degrees(ed):+.2f}° "
              f"· 차이 {np.degrees(eu-ed):+.2f}°", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n→ {args.out}  ({len(rows)} 행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
