#!/usr/bin/env python3
"""한 터미널 상태 모니터 — 손가락 온도·팔 effort·팁 F/T.

  python3 hand_status_monitor.py            # 1초 주기 화면 갱신

데이터 소스:
  · 손 온도/effort: /dg5f_right/dynamic_joint_states (temperature 인터페이스)
  · 팔: /dynamic_joint_states (openarm 은 온도 미노출 — effort 로 부하만 표시)
  · 팁 F/T: fingertip_*_broadcaster 의 wrench 토픽 (드라이버를
    fingertip_sensor:=true 로 켰을 때만 존재 — 없으면 OFF 표시)
"""
from __future__ import annotations

import math

import rclpy
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TEMP_WARN = 55.0     # ℃ — 표시 강조 문턱
TEMP_HOT = 65.0


class Monitor(Node):
    def __init__(self) -> None:
        super().__init__("hand_status_monitor")
        self.hand: dict[str, dict[str, float]] = {}
        self.arm: dict[str, dict[str, float]] = {}
        self.tips: dict[int, tuple[float, float, float]] = {}
        self.create_subscription(
            DynamicJointState, "/dg5f_right/dynamic_joint_states",
            lambda m: self._djs(m, self.hand), 10)
        self.create_subscription(
            DynamicJointState, "/dynamic_joint_states",
            lambda m: self._djs(m, self.arm), 10)
        for i in range(1, 6):
            for topic in (f"/dg5f_right/fingertip_{i}_broadcaster/wrench",
                          f"/fingertip_{i}_broadcaster/wrench"):
                self.create_subscription(
                    WrenchStamped, topic,
                    lambda m, k=i: self._tip(k, m), 10)
        self.create_timer(1.0, self._draw)

    def _djs(self, msg: DynamicJointState, store: dict) -> None:
        for name, iv in zip(msg.joint_names, msg.interface_values):
            store[name] = dict(zip(iv.interface_names, iv.values))

    def _tip(self, idx: int, msg: WrenchStamped) -> None:
        f = msg.wrench.force
        self.tips[idx] = (f.x, f.y, f.z)

    def _draw(self) -> None:
        out = ["\x1b[2J\x1b[H=== 손 (dg5f_right) — 관절 온도 ℃ / effort ==="]
        for fi, fn in enumerate(FINGERS, start=1):
            row = []
            for j in range(1, 5):
                d = self.hand.get(f"rj_dg_{fi}_{j}", {})
                t = d.get("temperature", float("nan"))
                mark = "🔥" if t >= TEMP_HOT else ("⚠" if t >= TEMP_WARN else " ")
                row.append(f"{j}:{t:5.1f}{mark}({d.get('effort', float('nan')):+5.2f})")
            out.append(f"  {fn:6s} " + "  ".join(row))
        out.append("\n=== 팔 (openarm) — 온도 미노출 · effort 만 ===")
        for side in ("left", "right"):
            row = []
            for j in range(1, 8):
                d = self.arm.get(f"openarm_{side}_joint{j}", {})
                row.append(f"{j}:{d.get('effort', float('nan')):+6.2f}")
            out.append(f"  {side:5s} " + " ".join(row))
        out.append("\n=== 팁 F/T (N) ===")
        if not self.tips:
            out.append("  OFF — 드라이버를 fingertip_sensor:=true 로 켜야 발행됨")
        else:
            for i in sorted(self.tips):
                x, y, z = self.tips[i]
                mag = math.sqrt(x * x + y * y + z * z)
                out.append(f"  {FINGERS[i - 1]:6s} |F| {mag:6.2f}  "
                           f"({x:+6.2f}, {y:+6.2f}, {z:+6.2f})")
        print("\n".join(out), flush=True)


def main() -> None:
    rclpy.init()
    node = Monitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
