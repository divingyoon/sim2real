#!/usr/bin/env python3
"""joint_monitor: 팔(left/right) + tesollo 손 관절값을 실시간 표시 + CSV 기록.

sim2real 라이브 실행(bringup·브리지·정책) 중 **모든 관절의 pos/vel/effort**를 한 화면에
보여주고 동시에 타임스탬프 CSV로 남긴다. effort(토크)를 함께 봐 J7 같은 과부하 지점을
즉시 포착하고, 로그로 "어디서 문제됐는지" 사후 분석한다.

실행 (robot PC, bringup 과 같은 ROS_DOMAIN_ID):
    python3 joint_monitor.py \
        [--arm-topic /joint_states] \
        [--hand-topic /dg5f_right/joint_states] \
        [--rate 5] [--effort-warn 5.0] [--log-dir ~/rl_ws/sim2real/logs]

구독: JointState 2개(팔 broadcaster + 손 broadcaster) — 이름으로 그룹 분류.
출력: 터미널 대시보드 + <log-dir>/joint_monitor_<타임스탬프>.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from joint_monitor_core import (
    JointSample,
    csv_header,
    csv_row,
    format_dashboard,
)


class JointMonitor(Node):
    def __init__(
        self,
        arm_topic: str,
        hand_topic: str,
        rate_hz: float,
        effort_warn: float,
        log_path: Path,
    ) -> None:
        super().__init__("joint_monitor")
        self.records: dict[str, JointSample] = {}
        self.effort_warn = effort_warn
        self._t0 = time.monotonic()

        self._order: list[str] | None = None
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self._log_path, "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._header_written = False

        self.create_subscription(JointState, arm_topic, self._cb, 20)
        self.create_subscription(JointState, hand_topic, self._cb, 20)
        self.create_timer(1.0 / rate_hz, self._tick)

        self.get_logger().info(
            f"joint_monitor: arm={arm_topic}, hand={hand_topic}, {rate_hz:g}Hz\n"
            f"  로그 → {self._log_path}"
        )

    def _cb(self, msg: JointState) -> None:
        n = len(msg.name)
        vel = list(msg.velocity) if len(msg.velocity) == n else [0.0] * n
        eff = list(msg.effort) if len(msg.effort) == n else [float("nan")] * n
        for i, name in enumerate(msg.name):
            self.records[name] = JointSample(pos=msg.position[i], vel=vel[i], eff=eff[i])

    def _tick(self) -> None:
        if not self.records:
            return
        elapsed = time.monotonic() - self._t0

        if self._order is None:
            self._order = sorted(self.records)
        if not self._header_written:
            self._csv.writerow(csv_header(self._order))
            self._header_written = True

        # 터미널: 화면 지우고 대시보드
        print("\033[2J\033[H", end="")
        print(format_dashboard(self.records, elapsed, self.effort_warn), flush=True)

        # 로그: 한 행 append
        self._csv.writerow(csv_row(elapsed, self.records, self._order))
        self._csv_file.flush()

    def close(self) -> None:
        try:
            self._csv_file.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-topic", default="/joint_states")
    parser.add_argument("--hand-topic", default="/dg5f_right/joint_states")
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--effort-warn", type=float, default=5.0,
                        help="|effort| 이 값 초과 시 대시보드에 '*' 경보 [Nm]")
    parser.add_argument("--log-dir", default=str(Path.home() / "rl_ws/sim2real/logs"))
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_dir) / f"joint_monitor_{stamp}.csv"

    rclpy.init()
    node = JointMonitor(
        arm_topic=args.arm_topic,
        hand_topic=args.hand_topic,
        rate_hz=args.rate,
        effort_warn=args.effort_warn,
        log_path=log_path,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()
        print(f"\n로그 저장됨: {log_path}")


if __name__ == "__main__":
    main()
