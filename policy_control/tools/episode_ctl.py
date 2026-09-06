#!/usr/bin/env python3
"""에피소드 CLI — 서비스를 안전한 순서로 부른다 (플랜 §4.1 서비스 계약).

    pd engage → pd goto_home → episode reset → start → (N 스텝 대기 또는 Ctrl-C) → stop → pd release

    python3 policy_control/tools/episode_ctl.py --steps 250                       # 계획만 출력
    python3 policy_control/tools/episode_ctl.py --steps 250 --execute \\
        --approve pd_engage --approve pd_goto_home --approve ep_start              # 실행

규약(scripts/ops/mission_run.py 와 동일)
  ① ``--execute`` 없이는 아무 서비스도 부르지 않는다 — 계획표만 출력.
  ② 로봇을 움직이는 단계(pd_engage · pd_goto_home · ep_start)는 ``--approve <id>`` 가 전부 있어야 시작한다.
  ③ 단계마다 Trigger 응답 JSON({"ok","reasons"})을 출력하고, /policy_control/status/pd 가 기대 phase 를
     보고할 때까지 기다린다(타임아웃이면 중단). 실패·Ctrl-C 는 stop → release 로 내려온다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass

NS = "/policy_control"
PD_STATUS = f"{NS}/status/pd"
OBS_STATUS = f"{NS}/status/obs"
FABRIC_STATUS = f"{NS}/status/fabric"
DEFAULT_PHASE_TIMEOUT = 30.0
DEFAULT_SERVICE_TIMEOUT = 5.0


@dataclass(frozen=True)
class Stage:
    id: str
    title: str
    service: str | None            # None = 대기 단계
    expect_pd: tuple[str, ...]     # 단계 후 pd phase 허용값 (빈 튜플 = 검사 없음)
    touches_real: bool = False


STAGES: tuple[Stage, ...] = (
    Stage("pd_engage", "pd engage (JTC → forward 3종, 토크 블렌드)", f"{NS}/pd/engage",
          ("RAMPING", "TRACKING"), touches_real=True),
    Stage("pd_goto_home", "pd goto_home (계약 홈으로 0.1 rad/s 램프 + settle)", f"{NS}/pd/goto_home",
          ("TRACKING",), touches_real=True),
    Stage("ep_reset", "episode reset (obs 가 seq 0 준비, 앵커 스냅샷)", f"{NS}/episode/reset", ("TRACKING",)),
    Stage("ep_start", "episode start (정책 루프 시작)", f"{NS}/episode/start", ("TRACKING",), touches_real=True),
    Stage("run", "N 스텝 대기 또는 Ctrl-C (HOLD 면 조기 종료)", None, ()),
    Stage("ep_stop", "episode stop", f"{NS}/episode/stop", ("TRACKING", "HOLD")),
    Stage("pd_release", "pd release (역블렌드 → 0 송출 → JTC 복귀)", f"{NS}/pd/release", ("IDLE",)),
)
SAFE_TAIL = ("ep_stop", "pd_release")


def stage_by_id(stage_id: str) -> Stage:
    for s in STAGES:
        if s.id == stage_id:
            return s
    raise KeyError(f"unknown stage {stage_id!r}; known: {[s.id for s in STAGES]}")


def missing_approvals(approvals: frozenset[str]) -> list[str]:
    return [s.id for s in STAGES if s.touches_real and s.id not in approvals]


def parse_trigger(success: bool, message: str) -> tuple[bool, list[str]]:
    """Trigger 응답 → (ok, reasons). message 는 JSON {"ok","reasons"} 여야 한다(아니면 실패로 본다)."""
    try:
        body = json.loads(message) if message else {}
    except json.JSONDecodeError:
        return False, [f"non-JSON trigger message: {message!r}"]
    if not isinstance(body, dict):
        return False, [f"trigger message is not an object: {message!r}"]
    reasons = [str(r) for r in body.get("reasons", [])]
    ok = bool(success) and bool(body.get("ok", False))
    return ok, reasons


def phase_ok(status: dict | None, stage: Stage) -> bool:
    if not stage.expect_pd:
        return True
    return status is not None and status.get("phase") in stage.expect_pd


def print_plan(steps: int, approvals: frozenset[str], execute: bool) -> None:
    print(f"episode_ctl 계획 · steps {steps} · execute {execute}")
    for s in STAGES:
        real = "실기" if s.touches_real else "    "
        appr = ("승인" if s.id in approvals else "미승인") if s.touches_real else "  "
        svc = s.service or "(wait)"
        print(f"  {s.id:13} {real} {appr:4} {svc:32} → pd {list(s.expect_pd) or '-'}  {s.title}")
    if not execute:
        print("\nDRY RUN — 아무 서비스도 부르지 않았다. 실행하려면 --execute 와 --approve <id> (실기 단계 전부)")


# ---------------------------------------------------------------- ROS runner
class Runner:
    """서비스 호출 + status 대기. rclpy 는 --execute 일 때만 import 된다."""

    def __init__(self, service_timeout: float, phase_timeout: float) -> None:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        rclpy.init()
        self.rclpy = rclpy
        self.node = Node("episode_ctl")
        self.service_timeout = service_timeout
        self.phase_timeout = phase_timeout
        self.pd_status: dict | None = None
        self.obs_status: dict | None = None
        self.node.create_subscription(String, PD_STATUS, self._on_pd, 10)
        self.node.create_subscription(String, OBS_STATUS, self._on_obs, 10)
        self.fabric_status: dict | None = None
        self.node.create_subscription(String, FABRIC_STATUS, self._on_fabric, 10)

    def _on_pd(self, msg) -> None:
        self.pd_status = _loads(msg.data)

    def _on_obs(self, msg) -> None:
        self.obs_status = _loads(msg.data)

    def _on_fabric(self, msg) -> None:
        self.fabric_status = _loads(msg.data)

    def wait_fabric_armed(self) -> bool:
        """reset 뒤 fabric 노드가 실제로 arm 됐는지(status.armed) 보고 start 한다 — 이벤트 유실 방어(run15)."""
        deadline = time.monotonic() + self.phase_timeout
        while time.monotonic() < deadline:
            self.spin(0.1)
            if self.fabric_status is not None and self.fabric_status.get("armed") is True:
                return True
        print("    ✗ fabric not armed after reset (status.armed)")
        return False

    def spin(self, seconds: float) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=seconds)

    def call(self, service: str) -> tuple[bool, list[str]]:
        from std_srvs.srv import Trigger

        client = self.node.create_client(Trigger, service)
        if not client.wait_for_service(timeout_sec=self.service_timeout):
            return False, [f"service {service} unavailable"]
        future = client.call_async(Trigger.Request())
        self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=self.service_timeout)
        if not future.done() or future.result() is None:
            return False, [f"service {service} timeout"]
        resp = future.result()
        print(f"    ← {service}: success={resp.success} message={resp.message}")
        return parse_trigger(resp.success, resp.message)

    def wait_phase(self, stage: Stage) -> bool:
        deadline = time.monotonic() + self.phase_timeout
        while time.monotonic() < deadline:
            self.spin(0.1)
            if phase_ok(self.pd_status, stage):
                return True
        got = None if self.pd_status is None else self.pd_status.get("phase")
        print(f"    ✗ pd phase {got!r} not in {list(stage.expect_pd)} within {self.phase_timeout:.0f}s")
        return False

    def wait_steps(self, steps: int) -> bool:
        """obs seq ≥ steps 까지 대기. pd 가 HOLD 로 가면 False."""
        deadline = time.monotonic() + self.phase_timeout + steps * 1.0
        while time.monotonic() < deadline:
            self.spin(0.1)
            if self.pd_status is not None and self.pd_status.get("phase") == "HOLD":
                print(f"    ✗ pd HOLD: {self.pd_status.get('reasons')}")
                return False
            seq = None if self.obs_status is None else self.obs_status.get("seq")
            if seq is not None and int(seq) >= steps:
                print(f"    ✓ obs seq {seq} ≥ {steps}")
                return True
        print("    ✗ step wait timeout")
        return False

    def close(self) -> None:
        self.node.destroy_node()
        self.rclpy.shutdown()


def _loads(text: str) -> dict | None:
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return None
    return body if isinstance(body, dict) else None


def run_stage(runner: Runner, stage: Stage, steps: int) -> bool:
    print(f"\n▶ {stage.id} — {stage.title}")
    if stage.service is None:
        return runner.wait_steps(steps)
    ok, reasons = runner.call(stage.service)
    if not ok:
        print(f"    ✗ refused: {reasons}")
        return False
    if reasons:
        print(f"    note: {reasons}")
    if stage.id == "ep_reset" and not runner.wait_fabric_armed():
        return False
    return runner.wait_phase(stage)


def safe_tail(runner: Runner, done: set[str], steps: int) -> None:
    """실패·Ctrl-C 후에도 stop → release 는 반드시 시도한다."""
    for sid in SAFE_TAIL:
        if sid not in done:
            run_stage(runner, stage_by_id(sid), steps)


def execute(steps: int, service_timeout: float, phase_timeout: float) -> int:
    runner = Runner(service_timeout, phase_timeout)
    done: set[str] = set()
    rc = 0
    try:
        for stage in STAGES:
            if not run_stage(runner, stage, steps):
                rc = 2
                break
            done.add(stage.id)
    except KeyboardInterrupt:
        print("\nCtrl-C → stop/release")
        rc = 130
    finally:
        if "pd_engage" in done:
            safe_tail(runner, done, steps)
        runner.close()
    return rc


def _parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=250, help="run 단계에서 기다릴 obs seq")
    ap.add_argument("--execute", action="store_true", help="★실제로 서비스를 부른다")
    ap.add_argument("--approve", action="append", default=[], help="실기 단계 승인 (반복)")
    ap.add_argument("--service-timeout", type=float, default=DEFAULT_SERVICE_TIMEOUT)
    ap.add_argument("--phase-timeout", type=float, default=DEFAULT_PHASE_TIMEOUT)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.steps <= 0:
        print("--steps 는 양수여야 한다", file=sys.stderr)
        return 2
    approvals = frozenset(args.approve)
    unknown = [a for a in approvals if a not in {s.id for s in STAGES}]
    if unknown:
        print(f"unknown --approve {unknown}", file=sys.stderr)
        return 2
    print_plan(args.steps, approvals, args.execute)
    if not args.execute:
        return 0
    missing = missing_approvals(approvals)
    if missing:
        print(f"\n✗ 실기 단계 승인이 없다 — --approve {' --approve '.join(missing)}", file=sys.stderr)
        return 3
    return execute(args.steps, args.service_timeout, args.phase_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
