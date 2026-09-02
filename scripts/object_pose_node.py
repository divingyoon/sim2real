#!/usr/bin/env python3
"""노드 3 — 카메라 프레임 물체 pose → base_link `/objects/<name>/pose`.

vision-3090 의 FP++ 가 내는 `/perception_plus_plus/<name>/pose`(camera optical) 를
DDS 로 직접 구독하고, cup_pose_relay 의 변환(T_base_cam ∘ T_cam_cad ∘ T_cad_body)으로
base_link 로 바꿔 발행한다. camera 블록은 레지스트리가 가리키는 공유 extrinsics,
cad_to_body 는 물체 항목. 소비자: ROS 정책 노드(다음 스펙).

  python3 object_pose_node.py [--objects shaker_closed cup_big_s100] [--head-joint-topic /head/joint_states]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cup_pose_relay import (  # noqa: E402
    cad_pose_to_base_body, extrinsics_at_head, head_state_is_usable,
)
from object_registry import extrinsics_for, input_topic, load_registry, output_topic  # noqa: E402
from pose_symmetry import remove_twist  # noqa: E402


class PoseConverter:
    """순수부: 물체별 Extrinsics 를 미리 조립해 두고 변환만 한다."""

    def __init__(self, registry, names: list[str]) -> None:
        self.names: list[str] = []
        for raw in names:
            canon = registry.resolve(raw)
            if canon not in self.names:
                self.names.append(canon)
        self._ext = {n: extrinsics_for(registry.get(n), registry.camera_extrinsics) for n in self.names}
        self._axis = {n: registry.get(n).symmetry_axis for n in self.names}
        self.base_frame = next(iter(self._ext.values())).base_frame if self._ext else "base_link"

    def convert(self, name: str, pos_cam: np.ndarray, quat_cam: np.ndarray,
                head: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        ext = self._ext[name]
        if head is not None:
            ext = extrinsics_at_head(ext, *head)
        quat_cam = np.asarray(quat_cam, float)
        if self._axis[name] is not None:
            # 대칭축 둘레 twist 제거 — 축 방향은 보존, 축 둘레 회전(추적기 자유 방향)은 0 으로
            quat_cam = remove_twist(quat_cam, np.asarray(self._axis[name]))
        return cad_pose_to_base_body(ext, np.asarray(pos_cam, float), quat_cam)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", nargs="*", default=None, help="기본: 레지스트리 전체")
    ap.add_argument("--head-joint-topic", default=None,
                    help="주면 T_base_cam 을 목 각도로 매번 계산(정적 camera 블록은 pan0/tilt-20 전용)")
    ap.add_argument("--head-max-age", type=float, default=1.0)
    args = ap.parse_args()
    registry = load_registry()
    conv = PoseConverter(registry, args.objects or registry.names())

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node

    class ObjectPoseNode(Node):
        def __init__(self) -> None:
            super().__init__("object_pose_node")
            self._pubs = {n: self.create_publisher(PoseStamped, output_topic(n), 10) for n in conv.names}
            for n in conv.names:
                self.create_subscription(PoseStamped, input_topic(n), lambda m, n=n: self._on_pose(n, m), 10)
            self._head: tuple[float, float] | None = None
            self._head_stamp: float | None = None
            if args.head_joint_topic:
                from sensor_msgs.msg import JointState
                self.create_subscription(JointState, args.head_joint_topic, self._on_head, 10)
            self._count = {n: 0 for n in conv.names}
            self.create_timer(10.0, self._report)
            self.get_logger().info(f"objects {conv.names} → {[output_topic(n) for n in conv.names]}")

        def _on_head(self, msg) -> None:
            names = list(msg.name)
            try:
                pan = float(msg.position[names.index("head_j_pan")])
                tilt = float(msg.position[names.index("head_j_tilt")])
            except (ValueError, IndexError):
                return
            self._head = (np.degrees(pan), np.degrees(tilt))
            self._head_stamp = self.get_clock().now().nanoseconds * 1e-9

        def _on_pose(self, name: str, msg: PoseStamped) -> None:
            head = None
            if args.head_joint_topic:
                now = self.get_clock().now().nanoseconds * 1e-9
                if not head_state_is_usable(self._head_stamp, now, args.head_max_age):
                    self.get_logger().warning("목 각도가 없거나 오래됐다 — 발행 보류", throttle_duration_sec=5.0)
                    return
                head = self._head
            p, q = msg.pose.position, msg.pose.orientation
            pos, quat = conv.convert(name, np.array([p.x, p.y, p.z]), np.array([q.w, q.x, q.y, q.z]), head)
            out = PoseStamped()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = conv.base_frame
            out.pose.position.x, out.pose.position.y, out.pose.position.z = map(float, pos)
            (out.pose.orientation.w, out.pose.orientation.x,
             out.pose.orientation.y, out.pose.orientation.z) = map(float, quat)
            self._pubs[name].publish(out)
            self._count[name] += 1

        def _report(self) -> None:
            self.get_logger().info(" · ".join(f"{n} {c}" for n, c in self._count.items()))

    rclpy.init()
    node = ObjectPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
