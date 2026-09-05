#!/usr/bin/env python3
"""합성 /cup_pose 발행 (카메라 무관 obs 검증용).

정지: --x 0.40 --y -0.15 --z 0.38
원운동(움직이는 컵 모사): --orbit 0.05,0.2  (반경 5cm, 0.2Hz, 중심=xyz)
"""
from __future__ import annotations
import argparse, math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=0.40)
    ap.add_argument("--y", type=float, default=-0.15)
    ap.add_argument("--z", type=float, default=0.38)
    ap.add_argument("--frame", default="base_link")
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--orbit", default="", help="r,hz (예: 0.05,0.2). 빈값=정지")
    args = ap.parse_args()
    r = hz = 0.0
    if args.orbit:
        r, hz = (float(v) for v in args.orbit.split(","))

    rclpy.init()
    node = Node("fake_cup_pose_pub")
    pub = node.create_publisher(PoseStamped, "/cup_pose", 10)
    t0 = node.get_clock().now()

    def tick():
        t = (node.get_clock().now() - t0).nanoseconds * 1e-9
        dx = r * math.cos(2 * math.pi * hz * t)
        dy = r * math.sin(2 * math.pi * hz * t)
        m = PoseStamped()
        m.header.stamp = node.get_clock().now().to_msg()
        m.header.frame_id = args.frame
        m.pose.position.x = args.x + dx
        m.pose.position.y = args.y + dy
        m.pose.position.z = args.z
        m.pose.orientation.w = 1.0
        pub.publish(m)

    node.create_timer(1.0 / args.rate, tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
