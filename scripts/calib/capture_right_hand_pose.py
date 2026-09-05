#!/usr/bin/env python3
"""실물 테솔로 우손의 **현재 자세를 스냅샷**해 yaml 로 남긴다 — hand bringup 기준 주먹.

사용자 규약(08.31): hand bringup 은 손을 **주먹 자세**로 만든 상태에서 한다(손가락이
바닥 베이스와 충돌하지 않는 자세). 그 자세를 여기로 기록해 두면, 이후 모든 bringup
직후 상태 검증과 리셋 궤적의 시작 손자세가 이 파일을 기준으로 삼는다.

    source /opt/ros/humble/setup.bash
    python3 scripts/calib/capture_right_hand_pose.py                 # 확인만 (기록 안 함)
    python3 scripts/calib/capture_right_hand_pose.py --save          # config/right_hand_fist.yaml

드라이버 토픽: /dg5f_right/joint_states (tesollo_sensor__right 프로필 ee_state).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

TOPIC = "/dg5f_right/joint_states"
OUT = Path(__file__).resolve().parents[2] / "config" / "right_hand_fist.yaml"
WAIT_SEC = 5.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true",
                        help="이것 없으면 출력만 하고 기록하지 않는다")
    parser.add_argument("--topic", default=TOPIC)
    args = parser.parse_args()

    rclpy.init()
    node = Node("capture_right_hand_pose")
    got: dict[str, float] = {}

    def cb(msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            got[name] = float(pos)

    node.create_subscription(JointState, args.topic, cb, 10)
    deadline = time.monotonic() + WAIT_SEC
    while not got and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not got:
        print(f"❌ {args.topic} 를 {WAIT_SEC:.0f}초 안에 못 받았다 — 손 드라이버 bringup 확인",
              file=sys.stderr)
        return 1

    names = sorted(got)
    print(f"수신 {len(names)} 관절 ({args.topic}):")
    for n in names:
        print(f"  {n}: {got[n]:+.4f}")

    if not args.save:
        print("\n(확인만 — 기록하려면 --save)")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 실물 테솔로 우손 hand bringup 기준 주먹 자세 (capture_right_hand_pose.py)",
        f"# 기록: {stamp} · 토픽: {args.topic}",
        "# 규약: 모든 hand bringup 직후 손은 이 자세여야 한다(바닥 베이스 무충돌).",
        "joints:",
    ]
    lines += [f"  {n}: {got[n]:+.4f}" for n in names]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
