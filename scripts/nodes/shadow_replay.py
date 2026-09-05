#!/usr/bin/env python3
"""sim 이 기록한 관절 목표를 실기로 다시 흘려보낸다 (ROS2 노드).

**이 노드는 정책을 돌리지 않는다.** 기록을 재생할 뿐이다. 그래서 실기가 이상하면 그냥
멈추면 되고, 관측이 어긋나 정책이 발산할 여지가 없다. 카메라도 물체도 없는 첫 실기
투입에 맞는 형태다.

계획 수립·검증은 `shadow_replay_core.py` 가 한다(로봇 없이 테스트된다). 여기는 발행과
기록만 담당한다.

지키는 것
  · `--execute` 없으면 **아무것도 발행하지 않는다**. robotctl 과 같은 규약이다.
  · 첫 프레임으로 도약하지 않는다 — 실측 자세에서 0.1 rad/s 로 램프해 들어간다.
  · 고정 주기 타이머로 발행한다. 수신 시점에 전진시키면 명령 rate 가 낮을 때 실효속도가
    줄어 목표에 영영 못 간다(08.03 정체의 근본원인).
  · 중단 조건을 넘으면 **즉시 멈춘다**(추종오차·effort·상태 두절).
  · 매 발행마다 `t_send`·`step_idx` 를 csv 로 남긴다 — `shadow_report.py` 의 정렬 키다.

시각 규약: `time_from_start=0`. 컨트롤러가 `interpolation_method="none"` 이라 미래 시각
포인트는 스트림에서 영영 적용되지 않고 로봇이 **무경고로 안 움직인다**.
[[jtc-none-interpolation-silent-stall]]

실행 (로봇 PC):
    python3 shadow_replay.py --sim logs/shadow/sim_fab_test16_gcON.npz \\
        --robot gripper_left --rate-scale 0.25 --log logs/shadow/real_x025.csv
    # 위는 dry-run. 실제로 내보내려면 --execute 를 붙인다.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


from jtc_bridge_core import JointRemap, velocity_limited_target  # noqa: E402
from robot_profile import load_robot_profile                     # noqa: E402
from shadow_replay_core import ReplayPlan                        # noqa: E402

#: 중단 조건. 넘으면 즉시 멈춘다 — 재생은 언제든 다시 하면 되고 팔은 하나뿐이다.
ABORT_TRACKING_ERR_RAD = 0.30
# ★브리지 램프 상한. 이 이상 떨어져 있으면 "홈 근처 오차 메우기"가 아니라
#   "어디서든 데려오기"가 되고, 직선 경로가 테이블을 지날 수 있다(09.03 충돌).
MAX_BRIDGE_GAP_RAD = 0.15
#: 유휴(반대편) 팔이 rest 에서 이만큼 벗어나면 **시작하지 않는다**.
#  fabric world 가 유휴 우팔을 (0.25, −0.20, 0.55) 반경 0.15 의 구로 세워 두기 때문이다
#  (`open_gripper_left_boxes_no_table.yaml`). 실기 우팔이 다른 곳에 있으면 fabric 은
#  **없는 장애물**을 피하고 **있는 팔**은 피하지 않는다. 값은 `grasp_inference` 와 같다 —
#  중력 처짐(예측 수십 mrad)은 통과시키고 자세가 다른 경우만 잡는 폭이다.
IDLE_ARM_MISMATCH_RAD = 0.15
ABORT_EFFORT_NM = {"l_aj_5": 5.0, "l_aj_6": 5.0, "l_aj_7": 5.0,
                   "r_aj_5": 5.0, "r_aj_6": 5.0, "r_aj_7": 5.0}
ABORT_STATE_STALE_SEC = 1.0


def tracking_offenders(measured, setpoint, names, tolerance):
    """**보낸 세트포인트** 대비 `tolerance` 를 넘은 관절 [(이름, 오차)].

    기록 target 이 아니라 세트포인트와 대는 이유: 우리가 `--max-vel` 로 일부러 붙잡은
    몫까지 팔 탓으로 세면, 리미터를 낮게 잡았다는 이유로 멀쩡한 팔에서 중단이 걸린다.
    "우리가 붙잡은 몫"은 `describe` 가 발행 전에, 리포트가 사후에 따로 보여 준다.
    """
    measured = np.asarray(measured, dtype=float).reshape(-1)
    setpoint = np.asarray(setpoint, dtype=float).reshape(-1)
    error = np.abs(measured - setpoint)
    return [(names[i], float(error[i])) for i in np.flatnonzero(error > tolerance)]


def idle_arm_offenders(measured, rest, names, tolerance):
    """rest 에서 `tolerance` 를 넘은 유휴 팔 관절 목록 [(이름, 실측, 기대)].

    거부는 반드시 **범인을 지목**해야 한다. 이름 없는 거부는 "자세가 이상한가 프로필이
    틀렸나"를 구분하지 못하게 만든다(robot_control 의 SafetyError 가 같은 이유로
    offender 를 명명한다).
    """
    measured = np.asarray(measured, dtype=float).reshape(-1)
    rest = np.asarray(rest, dtype=float).reshape(-1)
    return [
        (name, float(measured[i]), float(rest[i]))
        for i, name in enumerate(names)
        if abs(measured[i] - rest[i]) > tolerance
    ]


def build_plan(source: Path | dict, rate_scale: float, profile,
               arm_only: bool = False, multi_dof_hand: bool = False) -> ReplayPlan:
    """npz 경로 또는 이미 읽힌 기록 dict(백 되읽기 결과)를 받는다.

    백은 드라이버 계약(source 관절)으로 적혀 있지만 `action_bag_read.read_bag` 이
    canonical 로 되돌려 주므로, 여기서는 출처를 구분할 필요가 없다.
    """
    data = source if isinstance(source, dict) else np.load(source, allow_pickle=False)
    joint_names = [str(x) for x in data["meta_joint_names"]]
    if tuple(joint_names) != tuple(profile.arm_canonical):
        raise SystemExit(
            f"기록의 팔 관절 순서가 프로필과 다르다\n  기록 {joint_names}\n"
            f"  프로필 {list(profile.arm_canonical)}"
        )
    if multi_dof_hand:
        # ★다지 손(우 DG-5F 20관절): 1-jaw 가정을 쓰지 않고 **전 채널**을 그대로 싣는다.
        grip_names = [str(x) for x in data["meta_grip_names"]]
        if tuple(grip_names) != tuple(profile.ee_canonical):
            raise SystemExit(
                f"손 관절 순서가 프로필과 다르다\n  기록 {grip_names}\n"
                f"  프로필 {list(profile.ee_canonical)}")
        plan = ReplayPlan(
            arm_target=data["arm_target"][:, 0],
            grip_target=np.zeros(data["arm_target"].shape[0]),
            step_dt=float(data["meta_step_dt"][0]),
            rate_scale=rate_scale,
            joint_names=joint_names,
            gripper_name="",
        )
        plan.hand_target = data["grip_cmd"][:, 0].astype(float)
        return plan
    if arm_only:
        # ★손·그리퍼를 아예 발행하지 않는 재생. 손 전원이 없거나(테솔로 무전원)
        #   손 채널 차원이 이 재생기의 1-jaw 가정과 다를 때 쓴다.
        return ReplayPlan(
            arm_target=data["arm_target"][:, 0],
            grip_target=np.zeros(data["arm_target"].shape[0]),
            step_dt=float(data["meta_step_dt"][0]),
            rate_scale=rate_scale,
            joint_names=joint_names,
            gripper_name="",
        )
    grip_names = [str(x) for x in data["meta_grip_names"]]
    if profile.ee_canonical[0] not in grip_names:
        raise SystemExit(
            f"프로필 그리퍼 {profile.ee_canonical[0]} 이 기록 {grip_names} 에 없다"
        )
    # sim 은 USD 가 mimic 을 잃어 두 조를 다 지령하지만 실기는 한 조면 따라온다.
    jaw = grip_names.index(profile.ee_canonical[0])
    return ReplayPlan(
        arm_target=data["arm_target"][:, 0],
        grip_target=data["grip_cmd"][:, 0, jaw],
        step_dt=float(data["meta_step_dt"][0]),
        rate_scale=rate_scale,
        joint_names=joint_names,
        gripper_name=profile.ee_canonical[0],
    )


def describe(plan: ReplayPlan, profile, max_vel: float | None = None) -> str:
    """재생 **전에** 알아야 할 것: 얼마나 걸리고, 실기 한계 대비 무엇을 요구하는가."""
    lines = [
        f"프레임 {plan.n_frames}  ·  발행주기 {plan.publish_dt*1000:.2f} ms "
        f"({1/plan.publish_dt:.1f} Hz)  ·  총 {plan.schedule[-1]:.1f} s",
    ]
    limits = [profile.joint_limits[j].get("velocity") for j in plan.joint_names]
    if any(v is None for v in limits):
        unknown = [j for j, v in zip(plan.joint_names, limits) if v is None]
        lines.append(f"요구 최대 관절속도 {plan.peak_joint_speed:.3f} rad/s  ·  "
                     f"⚠ 한계를 모른다(프로필에 velocity 없음): {unknown}")
        return "\n".join(lines)
    limit = min(limits)
    lines.append(
        f"요구 최대 관절속도 {plan.peak_joint_speed:.3f} rad/s  vs  프로필 한계 {limit:.3f}")
    if plan.peak_joint_speed > limit:
        lines.append(f"  ⚠ 요구가 한계를 넘는다 — --rate-scale 을 "
                     f"{limit/plan.peak_joint_speed:.2f} 이하로 낮출 것")
    if max_vel is not None and plan.peak_joint_speed > max_vel:
        lines.append(
            f"  ⚠ 세트포인트 상한(--max-vel {max_vel:g})이 요구 "
            f"{plan.peak_joint_speed:.3f} 보다 낮다 — 세트포인트가 기록보다 **뒤처진다**.\n"
            f"    팔 탓이 아니라 우리가 붙잡는 것이다. 의도한 것이면 그대로 두고,"
            f" 아니면 --max-vel 을 {plan.peak_joint_speed:.2f} 이상으로."
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sim", type=Path, help="probe_fab_shadow_record 의 npz")
    src.add_argument("--bag", type=Path,
                     help="action_bag.py 가 구운 rosbag2. 백은 이미 감속(rate_scale)이 "
                          "반영돼 있으므로 --rate-scale 을 또 곱하지 않는다.")
    parser.add_argument("--robot", default="gripper_left", help="config/robots 의 구성")
    parser.add_argument("--rate-scale", type=float, default=0.25,
                        help="(0,1]. 낮출수록 느리게 재생한다. 처음에는 0.25 로 시작할 것.")
    parser.add_argument("--max-vel", type=float, default=0.5,
                        help="세트포인트 전진 상한[rad/s]. 재생 요구와 별개인 안전 캡.")
    parser.add_argument("--frames", type=int, default=0, help=">0 이면 앞에서 그만큼만")
    parser.add_argument("--log", type=Path, default=None, help="발행·측정 csv")
    parser.add_argument("--soft-start", type=float, default=0.6,
                        help="재생 속도를 0 에서 이 시간[s]에 걸쳐 올린다 — 정지에서 "
                             "전속으로 뛰는 급출발을 없앤다. 0 이면 끔")
    parser.add_argument("--allow-long-bridge", action="store_true",
                        help="기록 시작점에서 멀어도 직선 램프로 이어붙인다 — "
                             "★경로가 비었음을 눈으로 확인했을 때만")
    parser.add_argument("--allow-idle-arm-mismatch", action="store_true",
                        help="유휴 팔이 rest 밖이어도 시작한다. fabric world 가 그 팔을 "
                             "고정 위치의 구로 세워 두므로, 켜기 전에 실제 우팔이 어디 "
                             "있는지 눈으로 확인할 것.")
    parser.add_argument("--abort-tracking-err", type=float,
                        default=ABORT_TRACKING_ERR_RAD,
                        help="세트포인트 대비 추종오차 중단 임계[rad]. 무전원 손이 달린 "
                             "팔은 정적 처짐이 0.3 rad 에 이르므로 올려야 복귀가 가능하다. "
                             "effort 캡은 그대로 남아 실충돌은 잡는다.")
    parser.add_argument("--publish-shadow", action="store_true",
                        help="재생 중 sim 쪽 진실(기록의 지령·실측)을 /shadow/* 로 발행한다. "
                             "rosbag 하나에 SIM(지령·실측) + REAL(joint_states) 이 같은 "
                             "시계로 담겨 real2sim 튜닝 입력이 된다. 좌팔 라이브(run6)와 동형.")
    parser.add_argument("--with-hand", action="store_true",
                        help="다지 손(우 DG-5F 20관절)을 팔과 **같은 타임라인**으로 함께 발행한다. "
                             "손 전원·드라이버가 살아 있을 때만. 없으면 --arm-only 를 쓸 것.")
    parser.add_argument("--arm-only", action="store_true",
                        help="팔 7관절만 발행한다 — 그리퍼/손 채널은 계획·발행 모두 생략. "
                             "손 전원이 없는 우팔 검증용.")
    parser.add_argument("--execute", action="store_true",
                        help="이것 없으면 계획만 출력하고 아무것도 발행하지 않는다")
    args = parser.parse_args()

    profile = load_robot_profile(args.robot)
    if args.bag is not None:
        from action_bag_read import read_bag
        # 백의 dt 는 이미 구울 때의 rate_scale 이 들어간 값이다. 여기서 또 곱하면
        # 두 번 감속돼 "왜 이렇게 느리지"가 된다. 백은 1.0 으로 재생한다.
        source, rate_scale, label = read_bag(args.bag, profile=profile), 1.0, args.bag.name
        if args.rate_scale != parser.get_default("rate_scale"):
            print(f"⚠ --bag 은 감속이 이미 반영돼 있다 — --rate-scale "
                  f"{args.rate_scale} 무시하고 1.0 으로 재생한다.")
    else:
        source, rate_scale, label = args.sim, args.rate_scale, args.sim.name
    if args.with_hand and args.arm_only:
        raise SystemExit("--with-hand 와 --arm-only 는 함께 쓸 수 없다")
    plan = build_plan(source, rate_scale, profile, arm_only=args.arm_only,
                      multi_dof_hand=args.with_hand)
    # ★sim 실측(q_meas)은 npz 에만 있고 ReplayPlan 에는 없다 — 발행하려면 여기서 읽는다.
    sim_meas = None
    if args.publish_shadow:
        if args.bag is not None:
            print("⚠ --bag 은 sim 실측을 담지 않는다 — /shadow/sim_meas 는 발행하지 않는다")
        else:
            _npz = np.load(args.sim, allow_pickle=False)
            if "q_meas" in _npz:
                sim_meas = _npz["q_meas"].astype(float)
            else:
                print("⚠ 기록에 q_meas 가 없다 — /shadow/sim_meas 생략")
    source_hand = getattr(plan, "hand_target", None)
    if args.frames > 0:
        plan = ReplayPlan(
            arm_target=plan.arm_target[: args.frames],
            grip_target=plan.grip_target[: args.frames],
            step_dt=plan.step_dt, rate_scale=plan.rate_scale,
            joint_names=plan.joint_names, gripper_name=plan.gripper_name,
        )
        if args.with_hand:
            plan.hand_target = source_hand[: args.frames]

    print(f"구성 {profile.name}  ·  기록 {label}")
    print(describe(plan, profile, max_vel=args.max_vel))
    if not args.execute:
        print("\nDRY RUN: 아무것도 발행하지 않았다. 실제로 내보내려면 --execute 를 붙일 것.")
        return 0

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64, Float64MultiArray
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    class ShadowReplay(Node):
        def __init__(self) -> None:
            super().__init__("shadow_replay")
            self.plan = plan
            self.profile = profile
            self.allow_long_bridge = bool(args.allow_long_bridge)
            self.arm_remap = JointRemap(
                list(plan.joint_names), list(profile.arm_source), profile.joint_limits)
            self.hand_remap = JointRemap(
                list(profile.ee_canonical), list(profile.ee_source),
                profile.joint_limits) if args.with_hand else None
            self.grip_remap = None if (args.arm_only or args.with_hand) else JointRemap(
                [plan.gripper_name], list(profile.ee_source), profile.joint_limits)
            self.arm_pub = self.create_publisher(
                Float64MultiArray, profile.topics["arm_cmd"], 10)
            self.grip_pub = None if (args.arm_only or args.with_hand) else \
                self.create_publisher(Float64, profile.topics["ee_cmd"], 10)
            self.hand_pub = self.create_publisher(
                Float64MultiArray, profile.topics["ee_cmd"], 10) if args.with_hand else None
            self.arm_traj = self.create_publisher(
                JointTrajectory, profile.topics["arm_traj"], 10)
            self.grip_traj = None if args.arm_only else self.create_publisher(
                JointTrajectory, profile.topics["ee_traj"], 10)
            self.hand_setpoint: np.ndarray | None = None
            self.shadow_target = self.create_publisher(
                Float64MultiArray, "/shadow/sim_target", 10) if args.publish_shadow else None
            self.shadow_meas = self.create_publisher(
                Float64MultiArray, "/shadow/sim_meas", 10) if (
                    args.publish_shadow and sim_meas is not None) else None
            self.shadow_hand = self.create_publisher(
                Float64MultiArray, "/shadow/sim_hand", 10) if (
                    args.publish_shadow and args.with_hand) else None
            self.create_subscription(JointState, profile.topics["arm_state"],
                                     self._state_cb, qos_profile_sensor_data)

            self.measured = np.zeros(len(plan.joint_names))
            self.idle_measured = np.zeros(len(profile.idle_arm_canonical))
            self.effort = np.zeros(len(plan.joint_names))
            self.last_state = 0.0
            self._src_index: dict[str, int] = {}
            self.setpoint: np.ndarray | None = None
            self.ramp: np.ndarray | None = None
            self.cursor = 0
            self.rows: list[dict] = []
            self.aborted: str | None = None
            self._warned_behind = False
            self.timer = None

        def _state_cb(self, msg: JointState) -> None:
            if not self._src_index:
                self._src_index = {n: i for i, n in enumerate(msg.name)}
            for k, src in enumerate(self.profile.arm_source):
                i = self._src_index.get(src)
                if i is None:
                    continue
                sign = self.profile.joint_limits[self.plan.joint_names[k]]["sign"]
                self.measured[k] = msg.position[i] * sign
                if msg.effort and i < len(msg.effort):
                    self.effort[k] = msg.effort[i]
            for k, src in enumerate(self.profile.idle_arm_source):
                i = self._src_index.get(src)
                if i is None:
                    continue
                sign = self.profile.joint_limits[self.profile.idle_arm_canonical[k]]["sign"]
                self.idle_measured[k] = msg.position[i] * sign
            self.last_state = time.monotonic()

        def _check_bridge_gap(self) -> None:
            """★현재 자세가 bag 시작점에서 멀면 **거부한다**.

            09.03 사고: 진단하려고 `--frames 200` 으로 앞부분만 돌려 팔을 궤적
            **중간**에 세워둔 채 전체 bag 을 다시 돌렸다. 브리지 램프가 482프레임
            (9.6 s)짜리 **관절공간 직선**으로 이어붙었고, 그 직선은 테이블을 모른다 —
            손이 테이블과 충돌했다.

            램프는 "홈 근처의 오차를 메우는" 용도지 "어디서든 데려오는" 용도가 아니다.
            멀리 있으면 기록된 경로(역순 bag)로 먼저 돌아와야 한다.
            """
            gap = np.abs(self.measured - self.plan.arm_target[0])
            worst = int(gap.argmax())
            if float(gap.max()) <= MAX_BRIDGE_GAP_RAD:
                return
            if self.allow_long_bridge:
                self.get_logger().warning(
                    f"브리지 {math.degrees(float(gap.max())):.1f}° — 직선 램프는 "
                    "테이블을 모른다. 경로가 비어 있음을 직접 확인했어야 한다.")
                return
            raise SystemExit(
                f"현재 자세가 기록 시작점에서 멀다: "
                f"{self.plan.joint_names[worst]} {math.degrees(float(gap.max())):.1f}° "
                f"(상한 {math.degrees(MAX_BRIDGE_GAP_RAD):.1f}°).\n"
                "  브리지 램프는 관절공간 **직선**이라 테이블을 모른다 — 09.03 에 이걸로\n"
                "  손이 테이블에 부딪혔다. 먼저 역순 bag 으로 기록된 경로를 따라\n"
                "  돌아온 뒤 다시 실행할 것. 경로가 비었음을 눈으로 확인했다면\n"
                "  --allow-long-bridge 로 넘길 수 있다.")

        def start(self) -> None:
            deadline = time.monotonic() + 5.0
            while self.last_state == 0.0 and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
            if self.last_state == 0.0:
                raise SystemExit(
                    f"{self.profile.topics['arm_state']} 를 못 받았다 — bringup 확인"
                )
            self._check_idle_arm()
            self._check_bridge_gap()
            self._play_pos = 0.0
            self._play_ticks = 0
            self._soft_n = max(0, int(args.soft_start / self.plan.publish_dt))
            self.ramp = self.plan.ramp_from(self.measured.copy())
            self.setpoint = self.measured.copy()
            self.get_logger().info(
                f"실측에서 첫 프레임까지 램프 {len(self.ramp)} 프레임 "
                f"({len(self.ramp)*self.plan.publish_dt:.1f} s)")
            self.timer = self.create_timer(self.plan.publish_dt, self._tick)

        def _check_idle_arm(self) -> None:
            """유휴 팔이 sim 이 가정한 자리에 있는가.

            fabric world 는 그 팔을 고정 위치의 구로 세워 둔다. 다른 곳에 있으면 계획된
            궤적이 그 장면에서 안전하다는 근거가 사라진다 — 없는 장애물을 피하고 있는
            팔은 피하지 않는다.
            """
            from robot_profile import idle_arm_rest_pose

            rest = idle_arm_rest_pose(self.profile)
            offenders = idle_arm_offenders(
                self.idle_measured, rest,
                list(self.profile.idle_arm_canonical), IDLE_ARM_MISMATCH_RAD)
            if not offenders:
                self.get_logger().info("유휴 팔 rest 확인 — sim 이 가정한 장면과 일치")
                return
            detail = ", ".join(
                f"{n} {m:+.3f} (기대 {e:+.3f})" for n, m, e in offenders)
            if args.allow_idle_arm_mismatch:
                self.get_logger().warning(f"유휴 팔이 rest 밖인데 진행한다: {detail}")
                return
            raise SystemExit(
                f"유휴 팔이 rest 에서 벗어나 있다: {detail}\n"
                f"  fabric world 가 그 팔을 고정 위치의 구로 세워 두므로 계획된 궤적이\n"
                f"  이 장면에서 안전하다는 근거가 없다. `robotctl pose rest --group\n"
                f"  {'openarm_right_arm' if self.profile.acting_side == 'left' else 'openarm_left_arm'}"
                f" --execute` 로 정리하거나, 눈으로 확인한 뒤\n"
                f"  --allow-idle-arm-mismatch 로 진행할 것."
            )

        def _stop(self, why: str) -> None:
            self.aborted = why
            if self.timer is not None:
                self.timer.cancel()

        def _finish(self, why: str) -> None:
            """정상 종료. 중단과 같은 경로로 찍으면 완주가 실패처럼 보인다."""
            self.get_logger().info(f"재생 {why}")
            self._stop(why)

        def _abort(self, why: str) -> None:
            self.get_logger().error(f"중단: {why}")
            self._stop(why)

        def _tick(self) -> None:
            now = time.monotonic()
            if now - self.last_state > ABORT_STATE_STALE_SEC:
                return self._abort(f"상태 두절 {now - self.last_state:.2f} s")

            hand_all = getattr(self.plan, "hand_target", None)
            in_ramp = self.cursor < len(self.ramp)
            if in_ramp:
                target = self.ramp[self.cursor]
                grip = float(self.plan.grip_target[0])
                # ★램프 구간에서도 손은 **기록 첫 프레임**을 유지한다. 실기 손은 이미
                #   그 자세(bringup 주먹)에 있다는 것이 이 재생의 전제다.
                hand = hand_all[0] if hand_all is not None else None
                step_idx = -1
            else:
                # ★소프트 스타트 — 기록은 첫 프레임부터 전속(0.5 rad/s)으로 뛴다.
                #   정지 상태에서 그걸 그대로 주면 정지마찰+관성 때문에 세트포인트가
                #   바로 벌어져 추종 가드에 걸린다(09.03 반복 실측: j2/j7 0.30 rad).
                #   재생 **속도**를 0 에서 서서히 올려 그 급출발만 없앤다.
                self._play_ticks += 1
                if self._soft_n > 0:
                    self._play_pos += min(1.0, self._play_ticks / self._soft_n)
                else:
                    self._play_pos += 1.0
                frame = int(self._play_pos)
                if frame >= self.plan.n_frames:
                    return self._finish("완료")
                target = self.plan.arm_target[frame]
                grip = float(self.plan.grip_target[frame])
                hand = hand_all[frame] if hand_all is not None else None
                step_idx = frame

            # ★기록 target 이 아니라 **직전에 보낸 세트포인트** 대비로 판정한다.
            offenders = tracking_offenders(
                self.measured, self.setpoint, self.plan.joint_names,
                args.abort_tracking_err)
            if offenders:
                name, error = max(offenders, key=lambda item: item[1])
                return self._abort(f"{name} 이 보낸 세트포인트를 {error:.3f} rad 뒤처진다")
            behind = float(np.max(np.abs(self.setpoint - target)))
            if behind > ABORT_TRACKING_ERR_RAD and not self._warned_behind:
                self._warned_behind = True
                self.get_logger().warning(
                    f"세트포인트가 기록보다 {behind:.3f} rad 뒤처진다 — --max-vel "
                    f"{args.max_vel:g} 가 붙잡고 있다(팔 탓이 아니다).")
            for k, name in enumerate(self.plan.joint_names):
                cap = ABORT_EFFORT_NM.get(name)
                if cap is not None and abs(self.effort[k]) > cap:
                    return self._abort(f"{name} effort {self.effort[k]:.2f} N·m")

            self.setpoint = velocity_limited_target(
                target, self.setpoint, args.max_vel, self.plan.publish_dt)
            if hand is not None:
                # 손도 세트포인트 전진으로 속도 상한을 건다 — 팔과 같은 규약.
                base = self.hand_setpoint if self.hand_setpoint is not None else hand
                self.hand_setpoint = velocity_limited_target(
                    hand, base, args.max_vel, self.plan.publish_dt)
            self._publish(self.setpoint, grip, self.hand_setpoint)
            if self.shadow_target is not None:
                # ★기록의 **원본 프레임**을 낸다(우리가 리미터로 붙잡은 세트포인트가 아니라).
                #   그래야 bag 에서 "sim 이 원한 것 / 우리가 보낸 것 / 실기가 간 것" 셋이
                #   구분된다 — 셋을 뭉개면 추종 실패의 원인을 못 가른다.
                f = max(0, step_idx)
                self.shadow_target.publish(Float64MultiArray(
                    data=[float(v) for v in self.plan.arm_target[f]]))
                if self.shadow_meas is not None:
                    self.shadow_meas.publish(Float64MultiArray(
                        data=[float(v) for v in sim_meas[f]]))
                if self.shadow_hand is not None and hand_all is not None:
                    self.shadow_hand.publish(Float64MultiArray(
                        data=[float(v) for v in hand_all[f]]))
            self.rows.append({
                "t_send": now, "step_idx": step_idx,
                # ★리포트가 배속을 스스로 알아야 한다. 없으면 기록의 step_dt 를 쓰게 되고
                #   지연·지터가 rate_scale 배만큼 틀린 값으로 나온다(08.25 에 4배 틀렸다).
                "publish_dt": self.plan.publish_dt,
                **{f"cmd_{n}": float(self.setpoint[k])
                   for k, n in enumerate(self.plan.joint_names)},
                **{f"meas_{n}": float(self.measured[k])
                   for k, n in enumerate(self.plan.joint_names)},
                **{f"eff_{n}": float(self.effort[k])
                   for k, n in enumerate(self.plan.joint_names)},
                "cmd_gripper": grip,
            })
            self.cursor += 1

        def _publish(self, arm: np.ndarray, grip: float,
                     hand: np.ndarray | None = None) -> None:
            self.arm_pub.publish(Float64MultiArray(data=[float(v) for v in arm]))
            self.arm_traj.publish(self._traj(
                list(self.profile.arm_source), self.arm_remap.apply(arm)))
            if args.arm_only:
                return
            if hand is not None:
                self.hand_pub.publish(
                    Float64MultiArray(data=[float(v) for v in hand]))
                self.grip_traj.publish(self._traj(
                    list(self.profile.ee_source), self.hand_remap.apply(hand)))
                return
            self.grip_pub.publish(Float64(data=grip))
            self.grip_traj.publish(self._traj(
                list(self.profile.ee_source), self.grip_remap.apply(np.array([grip]))))

        @staticmethod
        def _traj(names: list[str], positions: np.ndarray) -> "JointTrajectory":
            msg = JointTrajectory()
            msg.joint_names = names
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in positions]
            # ★0 이어야 한다. 미래 시각이면 포인트가 영영 적용되지 않는다.
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = 0
            msg.points = [point]
            return msg

    rclpy.init()
    node = ShadowReplay()
    try:
        node.start()
        while node.aborted is None:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.aborted = "사용자 중단"
    finally:
        if args.log and node.rows:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            with args.log.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(node.rows[0]))
                writer.writeheader()
                writer.writerows(node.rows)
            print(f"-> {args.log}  ({len(node.rows)} 행)")
        node.destroy_node()
        # Ctrl-C 는 rclpy 가 컨텍스트를 이미 내린 뒤에 여기로 온다 — 두 번 내리면
        # RCLError 가 나고, 그게 종료 코드를 오염시켜 "실패"처럼 보인다.
        if rclpy.ok():
            rclpy.shutdown()
    print(f"종료: {node.aborted}")
    return 0 if node.aborted == "완료" else 1


if __name__ == "__main__":
    raise SystemExit(main())
