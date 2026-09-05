#!/usr/bin/env python3
"""인식 토픽에서 물체 위치를 한 번 읽어 `x,y,z` 한 줄로 출력한다(라운드 스크립트용).

★z 편향 보정을 여기서 한다. 09.03 실측: sim 에서 컵이 실제로 앉는 원점 높이는
  **0.266** 인데(소환 0.3105 에서 물리가 44 mm 떨어뜨린다) FP++ 는 **0.301** 을 준다.
  정책은 학습 내내 0.266 을 봤으므로, 보정 없이 주면 컵을 30 mm 높게 보고 그 높이로
  접근해 **손을 끝내 닫지 않는다**(실기 250스텝 실측: 게이트 1.00 인데 폐쇄도 0).
  ⚠인식 쪽을 고치면 이 보정을 반드시 0 으로 되돌릴 것 — 이중 보정이 된다.

★수신을 못 하면 **죽는다**. 조용히 기본값을 뱉으면 로봇이 엉뚱한 데를 잡는다.
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--topic", default="/objects/cup_big_s100/pose")
    ap.add_argument("--z-bias", type=float, default=0.0,
                    help="z 에 더할 보정(m). FP++ 가 높게 보면 음수")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--samples", type=int, default=20,
                    help="이만큼 모아 중앙값을 쓴다 — 한 샘플은 지터에 취약하다")
    args = ap.parse_args()

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    rclpy.init()
    node = Node("read_object_pose")
    got: list[tuple[float, float, float]] = []

    def cb(m):
        p = m.pose.position
        got.append((p.x, p.y, p.z))

    node.create_subscription(PoseStamped, args.topic, cb, qos_profile_sensor_data)
    t0 = time.time()
    while time.time() - t0 < args.timeout and len(got) < args.samples:
        rclpy.spin_once(node, timeout_sec=0.05)
    rclpy.shutdown()

    if not got:
        print(f"[read_object_pose] {args.topic} 수신 없음 — 중단", file=sys.stderr)
        return 1

    import statistics as st
    x = st.median(v[0] for v in got)
    y = st.median(v[1] for v in got)
    z = st.median(v[2] for v in got) + args.z_bias
    print(f"{x:.4f},{y:.4f},{z:.4f}")
    print(f"[read_object_pose] 표본 {len(got)}개 · z 보정 {args.z_bias:+.3f}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
