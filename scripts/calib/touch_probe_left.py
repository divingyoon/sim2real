#!/usr/bin/env python3
"""좌팔로 테이블/보드를 **짚어** 그 점의 로봇 base 좌표를 잰다.

왜 필요한가 (09.03). 카메라 외부 파라미터가 z 로 **+25 mm** 틀어져 있음이 charuco 보드
평면 측정으로 확인됐다(보드 평면 0.2301 vs 실기 테이블 0.205 — 09.05 줄자·CAD 확정).
그 편향이 셰이커(+10 mm)·컵(+22 mm, 원점 오프셋 0.0773 기준) 인식을 전부 높게 만들었고, 정책이 물체를 실제보다 높게 보고
접근해 **손을 끝내 닫지 않았다**. 고치려면 `T_base_board` 를 알아야 하는데, 그건
**로봇이 직접 짚어** 얻는 것이 가장 확실하다(자로 재면 base 원점을 눈으로 찾아야 한다).

절차 (한 점당):
    ① 목표 (x, y) 위 안전 높이로 fabric 이동 — 매 프레임 가드
    ② z 를 조금씩 낮추며 하강. **추종오차가 문턱을 넘으면 = 접촉**
    ③ 그 순간의 **실측** 관절로 FK 한 TCP 를 기록하고 즉시 상승

★접촉 판정은 추종오차다. 좌 그리퍼에는 F/T 가 없다. 손목 kp 10 이라 가벼운 접촉도
  오차로 보인다 — 다만 그래서 **하강 속도를 아주 낮게** 둬야 오차가 접촉 때문인지
  가속 때문인지 갈린다.
★그리퍼는 **닫고** 짚는다. ⚠TCP(gripper_base +0.080) 는 손끝이 **아니다** — 닫힌 손끝
  끝단은 접근축으로 +0.0954(15.4 mm 더 앞, URDF finger 메시). 기록은 TCP 로 하되 표면
  z = TCP z − 0.0154·cos(기울기) 로 환산할 것(09.03 짚기 0.245 → 0.229 정정).
  벌린 채 짚으면 어느 턱이 닿았는지 알 수 없다.
⚠좌팔 중력보상이 켜져 있어야 한다. 없으면 처짐만큼 낮게 앉아 접촉 판정이 빨라진다.

실행 (★사용자 승인 후):
    python3 touch_probe_left.py --targets 0.30,0.20 0.35,0.25 0.30,0.28 \\
        --z-start 0.32 --z-floor 0.18 --execute --out /tmp/touch.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


SIM2REAL = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--targets", nargs="+", required=True,
                    help="짚을 지점들 'x,y' (base 기준 m)")
    ap.add_argument("--z-start", type=float, default=0.32,
                    help="하강 시작 높이 — 보드보다 충분히 위")
    ap.add_argument("--z-floor", type=float, default=0.18,
                    help="★절대 하한. 여기까지 내려가도 접촉이 없으면 실패로 본다")
    ap.add_argument("--descend-step", type=float, default=0.002,
                    help="한 tick 당 하강량 m — 작을수록 접촉 판정이 깨끗하다")
    ap.add_argument("--touch-err", type=float, default=math.radians(2.5),
                    help="접촉 판정 추종오차 rad")
    ap.add_argument("--approach-ticks", type=int, default=250,
                    help="목표 위로 이동할 때 fabric 을 굴릴 tick 수")
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--run", type=Path,
                    default=SIM2REAL / "logs/policy/left_fab79",
                    help="좌팔 홈·fabric dt 를 읽을 런 dump")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--execute", action="store_true", help="★없으면 발행하지 않는다")
    args = ap.parse_args()

    sys.path.insert(0, str(SIM2REAL / "scripts"))

    from jtc_bridge_core import JointRemap
    from left_gripper_fk import LeftGripperFK
    from left_inference_dryrun import make_fabric, step_dt_from_run
    from left_policy_core import home_from_run
    from robot_profile import load_robot_profile

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    env_yaml = args.run / "params/env.yaml"
    home = home_from_run(env_yaml)
    fk = LeftGripperFK()
    prof = load_robot_profile("gripper_left")
    remap = JointRemap(list(prof.arm_canonical), list(prof.arm_source),
                       prof.joint_limits)
    lo = np.array([prof.joint_limits[n]["lower"] for n in prof.arm_canonical])
    hi = np.array([prof.joint_limits[n]["upper"] for n in prof.arm_canonical])

    rclpy.init()
    node = Node("touch_probe_left")
    box: dict = {}

    def on_js(m):
        idx = {n: i for i, n in enumerate(m.name)}
        if not all(s in idx for s in prof.arm_source):
            return
        box["q"] = np.array([m.position[idx[s]] for s in prof.arm_source])
        g = idx.get("openarm_left_finger_joint1")
        box["g"] = float(m.position[g]) if g is not None else 0.0
        box["t"] = time.time()

    node.create_subscription(JointState, "/joint_states", on_js,
                             qos_profile_sensor_data)
    pub = node.create_publisher(JointTrajectory, prof.topics["arm_traj"], 10)
    gpub = node.create_publisher(JointTrajectory, prof.topics["ee_traj"], 10)

    t0 = time.time()
    while time.time() - t0 < 10 and "q" not in box:
        rclpy.spin_once(node, timeout_sec=0.2)
    if "q" not in box:
        print("[touch] /joint_states 수신 없음 — 중단", flush=True)
        return 1

    def publish(q):
        if not args.execute:
            return
        msg = JointTrajectory()
        msg.joint_names = list(remap.output_source)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in remap.apply(list(q))]
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 0
        msg.points = [pt]
        pub.publish(msg)

    def close_gripper(v: float):
        if not args.execute:
            return
        msg = JointTrajectory()
        msg.joint_names = ["openarm_left_finger_joint1"]
        pt = JointTrajectoryPoint()
        pt.positions = [float(v)]
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 0
        msg.points = [pt]
        gpub.publish(msg)

    # ★탐침 끝을 명확히 — 턱을 닫으면 손끝이 TCP 에서 만난다.
    close_gripper(0.0)
    time.sleep(1.5)

    step_dt = step_dt_from_run(env_yaml)
    fabric = make_fabric(home, args.device, env_step_dt=step_dt)
    dt = 1.0 / args.hz
    # 홈 palm 자세를 기준으로 회전은 고정하고 위치만 옮긴다.
    palm_home = fabric(np.zeros(6), 1)          # 워밍업(내부 상태는 홈)
    del palm_home
    euler_home = np.array([0.317093862, -1.4835298641951802, 3.094591725])

    results = []
    for spec in args.targets:
        x, y = (float(v) for v in spec.split(","))
        print(f"\n[touch] 목표 ({x:.3f}, {y:.3f}) — 안전높이 {args.z_start:.3f} 로 이동",
              flush=True)

        # ── ① 접근 ────────────────────────────────────────────────────
        ok = False
        for k in range(args.approach_ticks):
            rclpy.spin_once(node, timeout_sec=0.001)
            tgt = fabric(np.concatenate([[x, y, args.z_start], euler_home]), 1)
            if np.any(tgt < lo) or np.any(tgt > hi):
                print(f"[touch] ★관절한계 — 중단 {np.round(np.degrees(tgt),1).tolist()}",
                      flush=True)
                break
            p = fk.poses(tgt, box["g"], box["g"]).tcp_pos
            if p[2] < args.z_floor:
                print(f"[touch] ★접근 중 TCP 가 하한 아래 {p[2]:.3f} — 중단", flush=True)
                break
            publish(tgt)
            time.sleep(dt)
            if k % 25 == 0:
                err = float(np.abs(box["q"] - tgt).max())
                print(f"   [{k:3d}] TCP {np.round(p,3).tolist()} · 추종오차 "
                      f"{math.degrees(err):.1f}°", flush=True)
            if np.linalg.norm(p[:2] - np.array([x, y])) < 0.01 and abs(
                    p[2] - args.z_start) < 0.01:
                ok = True
                break
        if not ok:
            print("[touch] 접근 수렴 실패 — 이 지점 건너뜀", flush=True)
            continue

        # ── ② 하강 (접촉까지) ─────────────────────────────────────────
        z = args.z_start
        touched = None
        while z > args.z_floor:
            z -= args.descend_step
            for _ in range(2):
                rclpy.spin_once(node, timeout_sec=0.001)
            tgt = fabric(np.concatenate([[x, y, z], euler_home]), 1)
            if np.any(tgt < lo) or np.any(tgt > hi):
                print("[touch] ★하강 중 관절한계 — 중단", flush=True)
                break
            publish(tgt)
            time.sleep(dt)
            err = float(np.abs(box["q"] - tgt).max())
            if err > args.touch_err:
                p_meas = fk.poses(box["q"], box["g"], box["g"]).tcp_pos
                touched = p_meas.copy()
                print(f"[touch] ★접촉 — 추종오차 {math.degrees(err):.2f}° · "
                      f"실측 TCP {np.round(p_meas,4).tolist()}", flush=True)
                break
        if touched is None:
            print(f"[touch] 하한 {args.z_floor} 까지 접촉 없음 — 실패", flush=True)

        # ── ③ 상승 ────────────────────────────────────────────────────
        for _ in range(60):
            rclpy.spin_once(node, timeout_sec=0.001)
            tgt = fabric(np.concatenate([[x, y, args.z_start], euler_home]), 1)
            publish(tgt)
            time.sleep(dt)
        if touched is not None:
            results.append({"target_xy": [x, y], "tcp_base": touched.tolist()})

    print("\n[touch] 결과", flush=True)
    for r in results:
        print(f"  ({r['target_xy'][0]:.3f}, {r['target_xy'][1]:.3f}) → "
              f"{np.round(r['tcp_base'], 4).tolist()}", flush=True)
    if len(results) >= 3:
        z = np.array([r["tcp_base"][2] for r in results])
        print(f"  평면 z: 평균 {z.mean():.4f} · 표준편차 {z.std()*1000:.1f} mm", flush=True)
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
        print(f"  → {args.out}", flush=True)

    rclpy.shutdown()
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
