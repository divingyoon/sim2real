#!/usr/bin/env python3
"""Isaac ROS FoundationPose 출력 → /cup_pose 릴레이 (P1 비전 노드).

Isaac ROS FoundationPose 노드는 카메라 optical 프레임 기준 컵 CAD 프레임
pose를 vision_msgs/Detection3DArray 로 발행한다. 이 노드가 글로벌 카메라
extrinsics(T_base_cam)와 CAD↔body 정합(T_cad_body)을 합성해 소비자 규약
(geometry_msgs/PoseStamped, robot base 프레임)으로 /cup_pose 를 발행한다.

    T_base_body = T_base_cam ∘ T_cam_cad ∘ T_cad_body

소비자: sim2real_inference.py (grasp 단계), pour_inference.py (capture 1회).
pour 루프는 FK라 이 노드가 죽어도 pouring은 계속된다.

사용:
    python3 cup_pose_relay.py \
        --extrinsics ../config/global_camera_extrinsics.yaml \
        --in-topic /poses --out-topic /cup_pose --min-score 0.0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from pour_obs_builder import compose_pose  # noqa: E402

DEFAULT_EXTRINSICS = _SCRIPT_DIR.parent / "config" / "global_camera_extrinsics.yaml"
QUAT_NORM_TOL = 1e-3


# ---------------------------------------------------------------------------
# 순수 로직 (ROS 불필요 — test_cup_pose_relay.py 대상)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Extrinsics:
    """base←camera 변환 + CAD→body 보정 (쿼터니언 wxyz)."""

    cam_pos: np.ndarray
    cam_quat: np.ndarray
    cad_to_body_pos: np.ndarray
    cad_to_body_quat: np.ndarray
    base_frame: str


def _validated_quat(raw, name: str) -> np.ndarray:
    q = np.asarray(raw, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"{name}: expected 4 values (wxyz), got shape {q.shape}")
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.0) > QUAT_NORM_TOL:
        raise ValueError(f"{name}: quaternion not normalized (|q|={norm:.4f})")
    return q / norm


def _validated_pos(raw, name: str) -> np.ndarray:
    p = np.asarray(raw, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError(f"{name}: expected 3 values, got shape {p.shape}")
    return p


def load_extrinsics(path: str | Path) -> Extrinsics:
    """yaml 로드 + 스키마 검증. 잘못된 값이면 즉시 ValueError."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: yaml root must be a mapping")
    for key in ("camera", "cad_to_body", "base_frame"):
        if key not in cfg:
            raise ValueError(f"{path}: missing required key '{key}'")
    cam = cfg["camera"]
    cad = cfg["cad_to_body"]
    return Extrinsics(
        cam_pos=_validated_pos(cam["position"], "camera.position"),
        cam_quat=_validated_quat(cam["orientation_wxyz"], "camera.orientation_wxyz"),
        cad_to_body_pos=_validated_pos(cad["position"], "cad_to_body.position"),
        cad_to_body_quat=_validated_quat(
            cad["orientation_wxyz"], "cad_to_body.orientation_wxyz"
        ),
        base_frame=str(cfg["base_frame"]),
    )


def extrinsics_at_head(ext: Extrinsics, pan_encoder_deg: float,
                       tilt_encoder_deg: float) -> Extrinsics:
    """목 각도에 맞는 `T_base_cam` 으로 **새 Extrinsics 를 만든다**(원본 불변).

    yaml 의 `camera:` 블록은 한 목 자세의 정적 스냅샷이라 목이 돌면 통째로 틀린다
    (pan 15° 면 카메라가 11 mm 이동 + 큰 회전 → 0.6 m 컵에서 수 cm). 여기서는
    hand-eye `T_neck_cam` 과 URDF FK 로 자세마다 다시 만든다.

    ★`cad_to_body` 는 건드리지 않는다 — 그건 목과 무관한 CAD↔body 정합이다.
    """
    from head_camera_pose import base_cam_pose

    pos, quat = base_cam_pose(pan_encoder_deg, tilt_encoder_deg)
    return replace(ext, cam_pos=pos, cam_quat=quat)


def head_state_is_usable(last_stamp_s: float | None, now_s: float,
                         max_age_s: float) -> bool:
    """목 각도가 아직 쓸 만한가. **틀린 컵 좌표는 없느니만 못하다.**"""
    if last_stamp_s is None:
        return False
    return (float(now_s) - float(last_stamp_s)) <= float(max_age_s)


def cad_pose_to_base_body(
    ext: Extrinsics,
    cad_pos_cam: np.ndarray,
    cad_quat_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """카메라 프레임 CAD pose → base 프레임 body pose (pos, quat wxyz)."""
    base_cad_pos, base_cad_quat = compose_pose(
        ext.cam_pos, ext.cam_quat, cad_pos_cam, cad_quat_cam
    )
    return compose_pose(
        base_cad_pos, base_cad_quat, ext.cad_to_body_pos, ext.cad_to_body_quat
    )


def select_best_detection(
    candidates: list[tuple[float, np.ndarray, np.ndarray]],
    min_score: float,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """(score, pos, quat_wxyz) 후보 중 min_score 이상 최고점. 없으면 None."""
    passing = [c for c in candidates if c[0] >= min_score]
    if not passing:
        return None
    return max(passing, key=lambda c: c[0])


def posestamped_to_candidate(
    px: float, py: float, pz: float,
    qw: float, qx: float, qy: float, qz: float,
    score: float = 1.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """camera-frame PoseStamped 필드 → relay 후보 (score, pos, quat_wxyz).

    FP++ 라이브 노드가 낼 camera optical 프레임 컵 pose 입력 경로.
    Detection3DArray 경로와 동일한 (score, pos, quat) 포맷을 반환한다.
    """
    return (
        float(score),
        np.array([px, py, pz], dtype=np.float64),
        np.array([qw, qx, qy, qz], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# ROS 노드
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FoundationPose → /cup_pose 릴레이")
    parser.add_argument("--extrinsics", default=str(DEFAULT_EXTRINSICS))
    parser.add_argument("--in-topic", default="/poses",
                        help="Isaac ROS FoundationPose Detection3DArray 출력 토픽")
    parser.add_argument("--in-type", choices=["detection3d", "posestamped"],
                        default="detection3d",
                        help="입력 메시지 타입: Isaac ROS FP=detection3d, FP++=posestamped")
    parser.add_argument("--out-topic", default="/cup_pose")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--head-joint-topic", default=None,
                        help="목 관절 상태 토픽(sensor_msgs/JointState, URDF 라디안). "
                             "주면 T_base_cam 을 목 각도로 매번 계산한다 — "
                             "yaml 의 정적 camera 블록은 한 자세에서만 맞는다")
    parser.add_argument("--head-max-age", type=float, default=1.0,
                        help="목 각도가 이보다 오래되면 발행하지 않는다(초)")
    args = parser.parse_args()

    ext = load_extrinsics(args.extrinsics)

    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped

    class CupPoseRelay(Node):
        def __init__(self) -> None:
            super().__init__("cup_pose_relay")
            self._pub = self.create_publisher(PoseStamped, args.out_topic, 10)
            if args.in_type == "posestamped":
                self.create_subscription(
                    PoseStamped, args.in_topic, self._posestamped_cb, 10)
            else:
                from vision_msgs.msg import Detection3DArray
                self.create_subscription(
                    Detection3DArray, args.in_topic, self._detections_cb, 10)
            self._published = 0
            self._head_urdf: tuple[float, float] | None = None
            self._head_stamp_s: float | None = None
            self._head_warned = 0
            if args.head_joint_topic:
                from sensor_msgs.msg import JointState
                self.create_subscription(
                    JointState, args.head_joint_topic, self._head_cb, 10)
                self.get_logger().info(
                    f"목 각도 인식 모드: {args.head_joint_topic} "
                    f"(최대 {args.head_max_age}s) — T_base_cam 을 매번 계산한다")
            self.get_logger().info(
                f"relay: {args.in_topic} → {args.out_topic} "
                f"(in_type={args.in_type}, extrinsics={args.extrinsics}, "
                f"min_score={args.min_score})"
            )

        def _head_cb(self, msg) -> None:
            """URDF 라디안으로 온다 — 여기서 인코더 각으로 되돌린다."""
            names = list(msg.name)
            try:
                pan = float(msg.position[names.index("head_j_pan")])
                tilt = float(msg.position[names.index("head_j_tilt")])
            except (ValueError, IndexError):
                return
            self._head_urdf = (np.degrees(pan), np.degrees(tilt))
            self._head_stamp_s = self.get_clock().now().nanoseconds * 1e-9

        def _current_extrinsics(self) -> Extrinsics | None:
            """목 인식 모드면 살아 있는 각도로 만든 것, 아니면 정적값.

            ★각도가 오래됐으면 **None 을 돌려 발행을 막는다.** 틀린 컵 좌표는
            없느니만 못하다 — 정책이 그걸 믿고 손을 뻗는다.
            """
            if not args.head_joint_topic:
                return ext
            now = self.get_clock().now().nanoseconds * 1e-9
            if not head_state_is_usable(self._head_stamp_s, now, args.head_max_age):
                self._head_warned += 1
                if self._head_warned % 30 == 1:
                    self.get_logger().warning(
                        f"목 각도가 없거나 오래됐다({args.head_joint_topic}) — "
                        f"cup_pose 를 발행하지 않는다 [{self._head_warned}회]")
                return None
            from head_fk_chain import encoder_from_urdf
            return extrinsics_at_head(ext, *encoder_from_urdf(*self._head_urdf))

        def _detections_cb(self, msg) -> None:
            candidates = []
            for det in msg.detections:
                for res in det.results:
                    p = res.pose.pose.position
                    q = res.pose.pose.orientation  # ROS xyzw
                    candidates.append((
                        float(res.hypothesis.score),
                        np.array([p.x, p.y, p.z]),
                        np.array([q.w, q.x, q.y, q.z]),
                    ))
            best = select_best_detection(candidates, args.min_score)
            if best is None:
                return
            _, cad_pos, cad_quat = best
            live = self._current_extrinsics()
            if live is None:
                return
            pos, quat = cad_pose_to_base_body(live, cad_pos, cad_quat)
            self._publish(msg.header.stamp, pos, quat)

        def _posestamped_cb(self, msg) -> None:
            p, q = msg.pose.position, msg.pose.orientation  # ROS xyzw
            cand = posestamped_to_candidate(p.x, p.y, p.z, q.w, q.x, q.y, q.z)
            best = select_best_detection([cand], args.min_score)
            if best is None:
                return
            _, cad_pos, cad_quat = best
            live = self._current_extrinsics()
            if live is None:
                return
            pos, quat = cad_pose_to_base_body(live, cad_pos, cad_quat)
            self._publish(msg.header.stamp, pos, quat)

        def _publish(self, stamp, pos, quat) -> None:
            out = PoseStamped()
            out.header.stamp = stamp
            out.header.frame_id = ext.base_frame
            out.pose.position.x, out.pose.position.y, out.pose.position.z = pos
            out.pose.orientation.w = quat[0]
            out.pose.orientation.x = quat[1]
            out.pose.orientation.y = quat[2]
            out.pose.orientation.z = quat[3]
            self._pub.publish(out)
            self._published += 1
            if self._published % 30 == 1:
                self.get_logger().info(
                    f"cup_pose #{self._published}: pos={np.round(pos, 3).tolist()}"
                )

    rclpy.init()
    node = CupPoseRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
