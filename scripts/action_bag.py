#!/usr/bin/env python3
"""sim 롤아웃 기록(npz) → **드라이버가 그대로 먹는 rosbag2**.

만드는 것은 "정책이 무슨 액션을 냈나"의 기록이 아니라 **로봇을 움직이는 명령 자체**다.
그래서 백에 들어가는 팔·그리퍼 토픽은 컨트롤러가 구독하는 바로 그 토픽이고, 관절명은
source 이름(`openarm_left_joint1..`)이며, 단일포인트 · `time_from_start=0` ·
`header.stamp=0` 규약을 그대로 지킨다 [[jtc-none-interpolation-silent-stall]].
`ros2 bag play <bag>` 만으로 팔이 움직인다.

해석용 채널도 같이 담는다(`/shadow/*`). 이건 드라이버가 안 보지만, 나중에 "왜 이렇게
움직였나"를 백 하나로 되짚으려면 있어야 한다 — 정책 원출력·palm 지령·스텝 인덱스.
`/shadow/meta` 에는 체크포인트·태스크 해시·fabric 자산·첫 프레임 자세가 JSON 으로 한 번
실린다. 백이 스스로를 설명해야 한 달 뒤에도 이게 무슨 기록인지 안다.

⚠ `ros2 bag play` 는 **접근 램프를 건너뛴다**. 첫 프레임이 지금 로봇 자세와 다르면 그건
이동이 아니라 도약이다. `/shadow/meta` 의 `first_frame` 으로 먼저 확인하거나,
`shadow_replay.py` 로 재생하라(램프·중단조건 포함).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from action_bag_core import (  # noqa: E402
    BagPlan,
    build_plan,
    load_npz,
    max_safe_rate_scale,
    velocity_verdict,
)
from robot_profile import load_robot_profile  # noqa: E402

TOPIC_STEP = "/shadow/step"
TOPIC_ACTION = "/shadow/action"
TOPIC_PALM = "/shadow/palm_cmd"
TOPIC_META = "/shadow/meta"


def _plan_from_args(args) -> tuple[BagPlan, object]:
    profile = load_robot_profile(args.robot)
    npz = load_npz(args.npz)
    plan = build_plan(
        npz,
        profile_joints=profile.joint_limits,
        arm_group=list(profile.arm_canonical),
        grip_group=list(profile.ee_canonical),
        rate_scale=args.rate_scale,
        env_index=args.env_index,
    )
    return plan, profile


def describe(plan: BagPlan, profile) -> str:
    """사람이 읽는 사전 판정. 재생 전에 이걸 먼저 보게 한다."""
    rows = velocity_verdict(plan, profile.joint_limits)
    safe = max_safe_rate_scale(plan, profile.joint_limits)
    out = [
        f"프레임 {plan.n_frames} · 발행 {plan.publish_dt*1000:.1f} ms "
        f"({1/plan.publish_dt:.1f} Hz) · 길이 {plan.duration_sec:.1f} s "
        f"· rate_scale {plan.rate_scale:.2f}",
        f"env  {plan.meta.get('env_index')}" if plan.meta.get("env_index") is not None else "env  단일",
        f"팔   {list(plan.arm.source_names)}",
        f"그리퍼 {list(plan.grip.source_names)}"
        + (f"  (mimic 로 버림: {list(plan.grip.dropped)})" if plan.grip.dropped else ""),
        "",
        f"{'관절':>22s} {'요구peak':>9s} {'한계':>7s}",
    ]
    for r in rows:
        limit = "  미상" if r["limit"] is None else f"{r['limit']:7.2f}"
        mark = " ⚠초과" if r["over"] else ""
        out.append(f"{r['source']:>22s} {r['peak']:9.3f} {limit}{mark}")

    clamp = plan.arm.clamp_total + plan.grip.clamp_total
    out.append("")
    if clamp:
        det = ", ".join(
            f"{n}×{c}" for n, c in zip(plan.arm.source_names, plan.arm.clamped) if c
        )
        out.append(f"⚠ 프로필 한계로 잘린 지령 {clamp} 회 ({det}) — 백은 실기가 받는 값이다.")
    else:
        out.append("✅ 모든 지령이 프로필 한계 안 — 잘린 곳 없음.")

    if any(r["over"] for r in rows):
        hint = "미상(프로필에 속도 한계 없음)" if safe is None else f"{safe:.2f}"
        out.append(f"⚠ 요구속도가 한계를 넘는다 → --rate-scale {hint} 로 다시 만들어라.")
        out.append("  시간을 늘리는 것이라 **경로는 그대로**다(rate-limit 클램프와 다름).")
    else:
        out.append("✅ 요구속도가 프로필 한계 안 — 이 속도로 재생 가능.")
    return "\n".join(out)


def _meta_json(plan: BagPlan, profile) -> str:
    return json.dumps(
        {
            "robot_profile": profile.name,
            "arm_topic": profile.topics["arm_traj"],
            "gripper_topic": profile.topics["ee_traj"],
            "arm_joint_names": list(plan.arm.source_names),
            "gripper_joint_names": list(plan.grip.source_names),
            "canonical_to_source": {
                c: s for c, s in zip(
                    plan.arm.canonical_names + plan.grip.canonical_names,
                    plan.arm.source_names + plan.grip.source_names,
                )
            },
            "dropped_mimic": list(plan.grip.dropped),
            "publish_dt": plan.publish_dt,
            "source_step_dt": plan.source_dt,
            "rate_scale": plan.rate_scale,
            "n_frames": plan.n_frames,
            # ★재생 전에 로봇을 여기로 보내야 한다. 아니면 첫 메시지가 도약이다.
            "first_frame": {
                "arm": dict(zip(plan.arm.source_names, plan.arm.positions[0].tolist())),
                "gripper": dict(zip(plan.grip.source_names,
                                    plan.grip.positions[0].tolist())),
            },
            "clamped": dict(zip(plan.arm.source_names, plan.arm.clamped)),
            "time_from_start": 0.0,
            "interpolation_assumption": "none",
            "recording_meta": plan.meta,
        },
        ensure_ascii=False,
        indent=2,
    )


def write_bag(plan: BagPlan, profile, out: Path, *, storage: str = "sqlite3") -> Path:
    """rosbag2 로 굽는다. ROS 의존은 이 함수 안에만 있다."""
    import rosbag2_py
    from builtin_interfaces.msg import Duration
    from geometry_msgs.msg import PoseStamped
    from rclpy.serialization import serialize_message
    from std_msgs.msg import Float64MultiArray, Int32, String
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    if out.exists():
        raise FileExistsError(f"{out} 이 이미 있다 — 덮어쓰지 않는다. 다른 이름을 주거나 지워라.")

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(out), storage_id=storage),
        rosbag2_py.ConverterOptions("", ""),
    )
    arm_topic = profile.topics["arm_traj"]
    grip_topic = profile.topics["ee_traj"]
    for name, mtype in (
        (arm_topic, "trajectory_msgs/msg/JointTrajectory"),
        (grip_topic, "trajectory_msgs/msg/JointTrajectory"),
        (TOPIC_STEP, "std_msgs/msg/Int32"),
        (TOPIC_ACTION, "std_msgs/msg/Float64MultiArray"),
        (TOPIC_PALM, "geometry_msgs/msg/PoseStamped"),
        (TOPIC_META, "std_msgs/msg/String"),
    ):
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=name, type=mtype, serialization_format="cdr"))

    meta = String()
    meta.data = _meta_json(plan, profile)
    writer.write(TOPIC_META, serialize_message(meta), int(plan.t_ns[0]))

    def _traj(names, positions) -> JointTrajectory:
        jt = JointTrajectory()
        jt.joint_names = list(names)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in positions]
        # none 보간 → 즉시 적용. 미래 시각이면 무경고로 아무 일도 안 일어난다.
        pt.time_from_start = Duration(sec=0, nanosec=0)
        jt.points = [pt]
        return jt

    for i in range(plan.n_frames):
        t = int(plan.t_ns[i])
        writer.write(arm_topic,
                     serialize_message(_traj(plan.arm.source_names,
                                             plan.arm.positions[i])), t)
        writer.write(grip_topic,
                     serialize_message(_traj(plan.grip.source_names,
                                             plan.grip.positions[i])), t)
        step = Int32()
        step.data = i
        writer.write(TOPIC_STEP, serialize_message(step), t)

        act = Float64MultiArray()
        act.data = [float(v) for v in plan.action[i]]
        writer.write(TOPIC_ACTION, serialize_message(act), t)

        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.header.stamp.sec = t // 1_000_000_000
        pose.header.stamp.nanosec = t % 1_000_000_000
        pose.pose.position.x = float(plan.palm_pos[i][0])
        pose.pose.position.y = float(plan.palm_pos[i][1])
        pose.pose.position.z = float(plan.palm_pos[i][2])
        qw, qx, qy, qz = (float(v) for v in plan.palm_quat_wxyz[i])
        pose.pose.orientation.w, pose.pose.orientation.x = qw, qx
        pose.pose.orientation.y, pose.pose.orientation.z = qy, qz
        writer.write(TOPIC_PALM, serialize_message(pose), t)

    del writer  # 닫아서 metadata.yaml 을 쓰게 한다
    (out / "shadow_meta.json").write_text(meta.data, encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("npz", type=Path, help="probe_fab_shadow_record.py 산출물")
    ap.add_argument("--robot", default="gripper_left", help="config/robots 구성 프로필")
    ap.add_argument("--out", type=Path, help="백 디렉터리. 없으면 판정만 출력")
    ap.add_argument("--env-index", type=int, default=None,
                    help="여러 env 를 기록했을 때 **어느 env** 를 재생할지. 평균을 내지 않는다 "
                         "— 평균 궤적은 어느 env 도 지나간 적 없는 경로다.")
    ap.add_argument("--rate-scale", type=float, default=1.0,
                    help="(0,1] 시간을 늘려 요구속도를 낮춘다. 경로는 안 변한다")
    ap.add_argument("--force", action="store_true",
                    help="요구속도가 한계를 넘어도 그대로 굽는다(재생 전 별도 감속 전제)")
    args = ap.parse_args()

    plan, profile = _plan_from_args(args)
    print(describe(plan, profile))

    if args.out is None:
        print("\n(--out 없음 → 굽지 않았다)")
        return 0

    over = any(r["over"] for r in velocity_verdict(plan, profile.joint_limits))
    if over and not args.force:
        print("\n❌ 굽지 않았다 — 요구속도가 실기 한계를 넘는다. "
              "--rate-scale 로 낮추거나, 알고도 굽겠다면 --force.")
        return 1

    path = write_bag(plan, profile, args.out)
    print(f"\n✅ {path}")
    print(f"   재생(안전): python3 scripts/shadow_replay.py --bag {path} --execute")
    print(f"   재생(직행): ros2 bag play {path}    ← 첫 프레임 자세 확인 필수")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
