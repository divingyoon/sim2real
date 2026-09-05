#!/usr/bin/env python3
"""정책 실행 **전에** sim 과 실기를 같은 리셋 자세에 세운다.

세 가지를 한 자리에서 한다:

  1. **어느 홈인가** — 기록이 아는 홈과 현재 preset 의 홈을 대조한다. 갈리면 기록 쪽이
     맞다(정책은 그 홈에서 출발하는 것을 배웠고 fabric cspace rest 도 그 홈이다).
     조용히 하나를 고르지 않고 **차이를 관절별로 찍는다**.
  2. **지금 어디 있나** — `/joint_states` 를 읽어 파킹 여부를 판정한다. 좌팔뿐 아니라
     **유휴 우팔**도 본다. fabric world 가 그 팔을 고정 위치의 장애물로 세워 두므로,
     실기 우팔이 다른 곳에 있으면 장면이 학습 때와 다르다.
  3. **거기로 보낸다** — `PARK_SPEED_RAD_PER_SEC`(0.1) 램프. 사람이 반응할 수 있는 속도.

⚠ 리셋 자세는 백의 첫 프레임이 **아니다**. 첫 프레임은 액션 한 스텝이 들어간 뒤의 값이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


from jtc_bridge_core import JointRemap  # noqa: E402
from reset_pose_core import (  # noqa: E402
    PARKED_TOL_RAD,
    HomePose,
    describe_disagreements,
    disagreements,
    home_from_preset,
    home_from_recording,
    not_parked,
)
from robot_profile import WS_ROOT, idle_arm_rest_pose, load_robot_profile  # noqa: E402
from shadow_replay_core import PARK_SPEED_RAD_PER_SEC, approach_ramp  # noqa: E402

PRESET = (WS_ROOT / "hdgp/source/openarm/openarm/gripper/left/grasp_sensor"
          / "grasp_left_preset.py")
PRESET_HOME_NAME = "LEFT_ARM_HOME_JOINT_POS"
PRESET_GRIP_OPEN_NAME = "GRIPPER_OPEN_POS"
PUBLISH_DT = 0.02


def resolve_home(args, profile) -> tuple[HomePose, HomePose | None]:
    """(쓸 홈, 대조용 preset 홈). preset 을 못 읽으면 대조는 None — 거짓말하지 않는다."""
    recorded: HomePose | None = None
    if args.sim is not None:
        npz = dict(np.load(args.sim, allow_pickle=True))
        recorded = home_from_recording(npz)

    preset: HomePose | None = None
    if PRESET.is_file():
        try:
            preset = home_from_preset(PRESET, PRESET_HOME_NAME)
        except KeyError as exc:
            print(f"⚠ preset 홈을 못 읽었다({exc}) — 대조 없이 진행한다.")

    if args.from_preset:
        if preset is None:
            raise SystemExit(f"--from-preset 인데 {PRESET} 에서 홈을 못 읽었다")
        return preset, recorded
    if recorded is None:
        if preset is None:
            raise SystemExit("홈의 출처가 없다 — --sim 으로 기록을 주거나 preset 을 고쳐라")
        print("⚠ 기록이 없어 preset 홈을 쓴다. 이 홈이 그 체크포인트의 홈인지는 "
              "확인되지 않았다.")
        return preset, None
    return recorded, preset


def report_home(chosen: HomePose, other: HomePose | None) -> bool:
    """홈 출처 보고. 두 출처가 갈리면 False (호출자가 막을 수 있게)."""
    print(f"홈 출처: {chosen.source}"
          + ("  ⚠유도값(직접 적힌 값이 아니다)" if chosen.derived else ""))
    if other is None:
        return True
    rows = disagreements(chosen, other)
    if not rows:
        print(f"✅ {other.source} 과 일치 — 어느 쪽을 써도 같다.")
        return True
    print(f"\n★ {chosen.source} 과 {other.source} 이 **다르다** — 홈이 바뀌었다.")
    print(describe_disagreements(rows, a_label="쓸 홈", b_label="대조"))
    print("\n  정책은 학습 당시 홈에서 출발하는 것을 배웠고, fabric 의 cspace rest 도 그"
          "\n  홈이다. 다른 홈에 파킹한 뒤 이 기록을 재생하면 첫 프레임이 도약이 된다.")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="gripper_left", help="config/robots 의 구성")
    ap.add_argument("--sim", type=Path, default=None,
                    help="홈의 출처가 될 기록 npz (권장). 없으면 preset 홈을 쓴다")
    ap.add_argument("--from-preset", action="store_true",
                    help="기록이 있어도 현재 preset 홈을 쓴다(새 체크포인트용)")
    ap.add_argument("--check", action="store_true",
                    help="/joint_states 를 읽어 파킹 여부만 판정한다")
    ap.add_argument("--execute", action="store_true", help="홈으로 이동한다")
    ap.add_argument("--allow-home-mismatch", action="store_true",
                    help="두 출처의 홈이 달라도 진행한다")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="램프 뒤 목표를 붙잡고 기다리는 시간[s]. 램프가 끝나는 순간 팔은 "
                         "아직 따라오는 중이라 여기서 판정하면 늘 '미파킹'이 된다. "
                         "정착 뒤에도 남는 오차는 지연이 아니라 **정적 처짐**이다.")
    ap.add_argument("--tol", type=float, default=PARKED_TOL_RAD,
                    help=f"파킹 판정 허용오차[rad] (기본 {PARKED_TOL_RAD})")
    args = ap.parse_args()

    profile = load_robot_profile(args.robot)
    home, other = resolve_home(args, profile)
    agreed = report_home(home, other)

    arm_names = list(profile.arm_canonical)
    target = home.as_array(arm_names)
    grip_open = None
    try:
        import ast as _ast
        for node in _ast.parse(PRESET.read_text()).body:
            if (isinstance(node, _ast.Assign)
                    and isinstance(node.targets[0], _ast.Name)
                    and node.targets[0].id == PRESET_GRIP_OPEN_NAME):
                grip_open = float(_ast.literal_eval(node.value))
    except Exception:
        pass
    if grip_open is None:
        grip_open = float(profile.joint_limits[profile.ee_canonical[0]]["upper"])
        print(f"⚠ preset 에서 그리퍼 개방값을 못 읽어 프로필 상한 {grip_open} 을 쓴다.")

    print("\n좌팔 홈 (canonical → source)")
    for i, can in enumerate(arm_names):
        print(f"  {can:8s} → {profile.joint_limits[can]['source']:22s} {target[i]:+8.4f}")
    print(f"  {profile.ee_canonical[0]:8s} → "
          f"{profile.joint_limits[profile.ee_canonical[0]]['source']:22s} {grip_open:+8.4f}")

    idle = idle_arm_rest_pose(profile)
    print(f"\n유휴 팔 rest ({len(idle)} 관절) — fabric world 가 장애물로 세워 두는 자세")

    if not agreed and not args.allow_home_mismatch and (args.check or args.execute):
        print("\n❌ 홈이 갈렸다 — 진행하지 않는다. 기록에 맞추려면 --sim 을 그 기록으로, "
              "새 코드에 맞추려면 --from-preset, 알고도 진행하려면 --allow-home-mismatch.")
        return 1

    if not (args.check or args.execute):
        print("\n(--check / --execute 없음 → 자세만 출력했다)")
        return 0

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    arm_remap = JointRemap(arm_names, list(profile.arm_source), profile.joint_limits)
    grip_remap = JointRemap([profile.ee_canonical[0]], list(profile.ee_source),
                            profile.joint_limits)

    class ResetPose(Node):
        def __init__(self) -> None:
            super().__init__("reset_pose")
            self.measured: dict[str, float] = {}
            self.create_subscription(JointState, profile.topics["arm_state"],
                                     self._on_state, qos_profile_sensor_data)
            self.arm_traj = self.create_publisher(
                JointTrajectory, profile.topics["arm_traj"], 10)
            self.grip_traj = self.create_publisher(
                JointTrajectory, profile.topics["ee_traj"], 10)

        def _on_state(self, msg: JointState) -> None:
            self.measured.update(dict(zip(msg.name, msg.position)))

        def arm_measured(self) -> np.ndarray | None:
            srcs = [profile.joint_limits[c]["source"] for c in arm_names]
            if not all(s in self.measured for s in srcs):
                return None
            return np.array([self.measured[s] for s in srcs], dtype=np.float64)

        def send(self, pub, remap, values: np.ndarray) -> None:
            jt = JointTrajectory()
            jt.joint_names = list(remap.output_source)
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in remap.apply(values)]
            pt.time_from_start.sec, pt.time_from_start.nanosec = 0, 0
            jt.points = [pt]
            pub.publish(jt)

    rclpy.init()
    node = ResetPose()
    deadline = node.get_clock().now().nanoseconds + int(5e9)
    measured = None
    while node.get_clock().now().nanoseconds < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        measured = node.arm_measured()
        if measured is not None:
            break
    if measured is None:
        print(f"\n❌ {profile.topics['arm_state']} 에서 좌팔 관절을 5 초 안에 못 받았다.")
        node.destroy_node(); rclpy.shutdown()
        return 1

    offenders = not_parked(measured, target, arm_names, tol=args.tol)
    print("\n현재 자세 대조 (목표 vs 실측)")
    print(describe_disagreements(offenders, a_label="홈", b_label="실측"))

    if args.check:
        ok = not offenders
        print(f"\n{'✅ 파킹됨 — 정책을 시작해도 된다.' if ok else '❌ 파킹 안 됨.'}")
        node.destroy_node(); rclpy.shutdown()
        return 0 if ok else 1

    ramp = approach_ramp(measured, target, speed=PARK_SPEED_RAD_PER_SEC, dt=PUBLISH_DT)
    settle_steps = max(0, int(round(args.settle / PUBLISH_DT)))
    print(f"\n램프 {len(ramp)} 스텝 · {len(ramp)*PUBLISH_DT:.1f} s "
          f"· {PARK_SPEED_RAD_PER_SEC} rad/s  (+ 정착 {args.settle:.1f} s)")
    rate = node.create_rate(1.0 / PUBLISH_DT)
    import threading
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    try:
        for row in ramp:
            node.send(node.arm_traj, arm_remap, row)
            node.send(node.grip_traj, grip_remap, np.array([grip_open]))
            rate.sleep()
        # 목표를 계속 붙잡는다. 발행을 멈추면 세트포인트가 사라지는 것이 아니라
        # 우리가 팔이 도착하는 것을 **안 보고 판정**하게 된다.
        for _ in range(settle_steps):
            node.send(node.arm_traj, arm_remap, target)
            node.send(node.grip_traj, grip_remap, np.array([grip_open]))
            rate.sleep()
    finally:
        final = node.arm_measured()
        if final is not None:
            left = not_parked(final, target, arm_names, tol=args.tol)
            print("\n도착 후")
            print(describe_disagreements(left, a_label="홈", b_label="실측"))
            if not left:
                print("\n✅ 파킹 완료 — 정책을 시작해도 된다.")
            else:
                worst = max(abs(r.delta) for r in left)
                print(f"\n⚠ 정착 {args.settle:.1f} s 뒤에도 최대 {worst*1000:.1f} mrad "
                      "남았다. 이건 지연이 아니라 **정적 처짐**이다 — 중력이 이기고 있거나 "
                      "게인이 낮다. 재생을 시작하면 이 오차가 그대로 깔린 채 간다.")
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
