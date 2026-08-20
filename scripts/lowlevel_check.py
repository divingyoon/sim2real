#!/usr/bin/env python3
"""저수준 제어 검증 노드 — 정책 없이 실기 제어 인터페이스만 확인한다.

가이드(`sim2real_rl_debugging_guide.md`) Stage 0~2. **RL 을 완전히 끈 상태**에서:

    TEST0  기동 상태 기록      — 명령 전 팔/손이 실제로 어디에 있는가
    TEST1  hold                — q_cmd = q_measured 를 유지 → 관절별 드리프트(중력 처짐)
    TEST2  단일 관절 스텝      — 관절 하나만 ±Δ → 그 관절만/방향/크기 → sign·offset 표

실행 (robot PC):
    python3 lowlevel_check.py --robot tesollo_sensor__right --group arm --dry-run
    python3 lowlevel_check.py --robot tesollo_sensor__right --group arm --execute

`--execute` 없으면 **아무것도 발행하지 않는다**(계획만 출력). 실기 스택 기동 순서·주의는
`docs/RUNBOOK_GRASP_V1_LIVE.md`, 결과 기록처는 `docs/measure/S2R_INTERFACE_EQUIVALENCE.md`.

★ 팔 JTC 는 `interpolation_method="none"` 이므로 time_from_start=0 + 세트포인트 rate-limit 을
  쓴다(브리지와 동일 설계). 미래 시각 포인트를 주면 영영 적용되지 않는다.
  [[jtc-none-interpolation-silent-stall]]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from jtc_bridge_core import JointRemap, velocity_limited_target
from lowlevel_check_core import (
    DEFAULT_AMPLITUDES,
    build_step_plan,
    clamp_command,
    evaluate_hold,
    evaluate_step,
    summarize_sign_table,
)
from robot_profile import load_robot_profile

CONTROL_DT = 1.0 / 60.0
#: 세트포인트 전진 속도. 0.1 rad/s = 0.1 rad 스텝에 1초 — 비상정지 도달 가능한 속도.
DEFAULT_MAX_VEL = 0.1
#: 상태 미수신 허용 시간. 넘으면 진행하지 않는다(조용한 오작동 방지).
STATE_TIMEOUT_SEC = 5.0
#: 실측이 세트포인트에서 이만큼 벗어나면 중단. 처짐(예측 최대 0.27 rad)보다 크게 잡되
#: 폭주는 잡히도록.
ABORT_DEVIATION_RAD = 0.45


class LowLevelCheck(Node):
    def __init__(self, profile, group: str, args) -> None:
        super().__init__("lowlevel_check")
        self.profile = profile
        self.group = group
        self.args = args

        if group == "arm":
            self.canonical = list(profile.arm_canonical)
            self.source = list(profile.arm_source)
            traj_topic = profile.topics["arm_traj"]
            state_topics = [profile.topics["arm_state"]]
        elif group == "ee":
            self.canonical = list(profile.ee_canonical)
            self.source = list(profile.ee_source)
            traj_topic = profile.topics["ee_traj"]
            state_topics = [profile.topics["ee_state"]]
        else:
            raise ValueError(f"알 수 없는 group: {group}")

        if args.joints:
            keep = set(args.joints)
            unknown = keep - set(self.canonical)
            if unknown:
                raise ValueError(f"프로필에 없는 관절: {sorted(unknown)}")
            self.tested = [j for j in self.canonical if j in keep]
        else:
            self.tested = list(self.canonical)

        self.remap = JointRemap(self.canonical, self.source, profile.joint_limits)
        self.limits = {j: profile.joint_limits[j] for j in self.canonical}
        self.actual: dict[str, float] = {}
        self._state_seen = 0.0

        self.pub = self.create_publisher(JointTrajectory, traj_topic, 10)
        for t in state_topics:
            self.create_subscription(JointState, t, self._state_cb, 20)

        self.get_logger().info(
            f"저수준 검증 [{profile.name} / {group}]\n"
            f"  발행: {traj_topic}   구독: {', '.join(state_topics)}\n"
            f"  관절 {len(self.canonical)}개 중 검사 {len(self.tested)}개\n"
            f"  세트포인트 속도 {args.max_vel} rad/s · 스텝 상한 {args.max_step} rad\n"
            f"  {'DRY-RUN (발행 없음)' if not args.execute else '★ 실제 발행 ★'}"
        )

    # ------------------------------------------------------------------ 상태
    def _state_cb(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.actual[name] = float(msg.position[i])
        self._state_seen = time.monotonic()

    def measured(self) -> np.ndarray:
        """canonical 순서 실측 위치. 하나라도 없으면 예외 — zeros 로 채우지 않는다."""
        missing = [s for s in self.source if s not in self.actual]
        if missing:
            raise RuntimeError(f"상태 미수신 관절: {missing[:5]}{'…' if len(missing) > 5 else ''}")
        by_source = {s: self.actual[s] for s in self.source}
        return np.array(
            [by_source[src] / (self.profile.joint_limits[c]["sign"] or 1.0)
             for c, src in zip(self.canonical, self.source)],
            dtype=np.float64,
        )

    def wait_for_state(self) -> None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < STATE_TIMEOUT_SEC:
            rclpy.spin_once(self, timeout_sec=0.05)
            try:
                self.measured()
                return
            except RuntimeError:
                continue
        raise RuntimeError(
            f"{STATE_TIMEOUT_SEC}s 안에 전 관절 상태를 못 받았다 — 컨트롤러/DDS 확인"
        )

    # ------------------------------------------------------------------ 발행
    def _publish(self, canonical_cmd: np.ndarray) -> None:
        if not self.args.execute:
            return
        positions = self.remap.apply(list(canonical_cmd))
        jt = JointTrajectory()
        jt.joint_names = list(self.remap.output_source)
        pt = JointTrajectoryPoint()
        pt.positions = positions.tolist()
        pt.time_from_start = Duration(sec=0, nanosec=0)
        jt.points = [pt]
        self.pub.publish(jt)

    def run_segment(self, target: np.ndarray, duration_s: float, setpoint: np.ndarray):
        """target 으로 rate-limit 전진하며 duration_s 동안 유지. (samples, 마지막 세트포인트)."""
        samples: list[np.ndarray] = []
        n = max(1, int(duration_s / CONTROL_DT))
        sp = np.array(setpoint, dtype=np.float64)
        for _ in range(n):
            sp = velocity_limited_target(target, sp, self.args.max_vel, CONTROL_DT)
            self._publish(sp)
            rclpy.spin_once(self, timeout_sec=CONTROL_DT)
            q = self.measured()
            samples.append(q)
            dev = float(np.max(np.abs(q - sp)))
            if self.args.execute and dev > ABORT_DEVIATION_RAD:
                raise RuntimeError(
                    f"중단: 실측이 세트포인트에서 {dev:.3f} rad 이탈 (>{ABORT_DEVIATION_RAD})"
                )
        return samples, sp


def _write_csv(path: Path, canonical, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["phase", "joint", "amplitude", "t_sec"] + list(canonical))
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--robot", default="tesollo_sensor__right", help="config/robots 프로필")
    p.add_argument("--group", default="arm", choices=["arm", "ee"])
    p.add_argument("--joints", nargs="*", default=None, help="검사할 canonical 관절(기본 전부)")
    p.add_argument("--amplitudes", type=float, nargs="*", default=list(DEFAULT_AMPLITUDES))
    p.add_argument("--hold-sec", type=float, default=10.0)
    p.add_argument("--dwell-sec", type=float, default=2.0)
    p.add_argument("--max-vel", type=float, default=DEFAULT_MAX_VEL)
    p.add_argument("--max-step", type=float, default=0.12, help="기준 자세 대비 명령 상한[rad]")
    p.add_argument("--execute", action="store_true", help="없으면 아무것도 발행하지 않는다")
    p.add_argument("--log-dir", default="~/rl_ws/sim2real/logs/measure")
    args = p.parse_args()

    profile = load_robot_profile(args.robot)
    rclpy.init()
    node = LowLevelCheck(profile, args.group, args)
    plan = build_step_plan(
        node.tested, amplitudes=tuple(args.amplitudes),
        dwell_s=args.dwell_sec, hold_s=args.hold_sec,
    )

    if not args.execute:
        print(f"\n계획 {len(plan)}구간 (hold 1 + 스텝 {len(plan)-1}) — 발행하지 않음")
        for s in plan[:5]:
            print(f"  {s.phase:5} {s.joint or '-':16} {s.amplitude:+.3f} rad  {s.duration_s}s")
        print(f"  … 총 {len(plan)}구간, 예상 소요 {sum(s.duration_s for s in plan)/60:.1f}분")
        node.destroy_node(); rclpy.shutdown(); return

    rows, verdicts = [], []
    t0 = time.monotonic()
    try:
        node.wait_for_state()
        base = node.measured()
        print(f"\nTEST0 기준 자세: {np.round(base, 4)}")

        setpoint = base.copy()
        for spec in plan:
            if spec.phase == "hold":
                target = base
            else:
                target = base.copy()
                target[node.canonical.index(spec.joint)] += spec.amplitude
                target = clamp_command(base, target, node.canonical, node.limits, args.max_step)
            samples, setpoint = node.run_segment(target, spec.duration_s, setpoint)
            for q in samples:
                rows.append([spec.phase, spec.joint or "", spec.amplitude,
                             round(time.monotonic() - t0, 4)] + [round(float(v), 6) for v in q])
            if spec.phase == "hold":
                drift = evaluate_hold(node.canonical, samples)
                print("\nTEST1 hold 드리프트 [rad] (부호 = 처짐 방향):")
                for j, d in drift.items():
                    print(f"  {j:18} {d:+.4f}   ({np.degrees(d):+6.2f}°)")
            else:
                v = evaluate_step(node.canonical, base, samples[-1], spec)
                verdicts.append(v)
                mark = "OK " if v.ok else "FAIL"
                print(f"TEST2 {mark} {v.joint:18} 명령{v.commanded:+.3f} 실측{v.measured:+.4f} "
                      f"비율{v.ratio:5.2f}  {v.reason}")
            # 다음 스텝을 위해 기준 자세로 복귀
            if spec.phase == "step":
                _, setpoint = node.run_segment(base, 1.0, setpoint)

        table = summarize_sign_table(verdicts, all_joints=node.canonical)
        print("\n=== sign 표 (가이드 §6) ===")
        for j, row in table.items():
            sign = "?" if row["sign"] is None else f"{row['sign']:+.0f}"
            ok = "-" if row["ok"] is None else ("OK" if row["ok"] else "FAIL")
            print(f"  {j:18} sign={sign}  {ok}  ({row['steps']}스텝)"
                  + (f"  {row['reasons'][0]}" if row["reasons"] else ""))
    finally:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = Path(args.log_dir).expanduser() / f"lowlevel_{args.group}_{stamp}.csv"
        if rows:
            _write_csv(out, node.canonical, rows)
            print(f"\nCSV: {out}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
