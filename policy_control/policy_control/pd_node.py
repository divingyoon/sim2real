"""pd_node — rclpy 껍질: 팔 단위(`pd_arm.ArmUnit`)를 `sides` 순서(우 → 좌)로 들고 IO 를 배선한다 (플랜 §4.4).

    /policy_control/joint_target (JointState, canonical 이름) ─▶ 이름으로 팔별 분배 ─┐
    /joint_states · /dg5f_<side>/joint_states (robot yaml arm/ee) ─▶ 팔별 SourceSet   ├─▶ ArmUnit.tick @ pd_hz (팔마다)
    /policy_control/episode (reset → new_episode, stop/abort → 현재 세트포인트 유지)  │
    /policy_control/estop (Bool, 래치 — 모든 팔)                                     ▼
    backends.write(cmd) [execute 일 때만 발행] · /policy_control/pd/applied (모든 팔 이어 붙임) · /policy_control/status/pd
    서비스 std_srvs/Trigger: /policy_control/pd/{engage, goto_home, release} — 선택한 팔을 순서대로(우 먼저)

ROS 파라미터 `sides` = 쉼표 목록(기본 '' = robot yaml 과 계약 양쪽에 있는 팔 전부). 한 팔의 HOLD 는 그 팔만 세운다;
estop/release 는 모든 팔에 건다. status 의 `phase` 는 팔들의 합성(HOLD > RELEASING > RAMPING > TRACKING > IDLE),
팔별 상세는 `arms.<side>`.

★execute = ROS 파라미터 `execute` AND pd_*.yaml `execute` — 둘 다 참일 때만 컨트롤러 토픽 발행·
  controller_manager 호출이 생긴다(백엔드·ControllerSwitch 가 각자 잠근다). 아니면 법칙·status·applied 만 돈다.
타이머·구독은 한 콜백 그룹(직렬), 서비스는 다른 그룹 — 서비스가 phase 를 기다리는 동안 타이머가 돈다
(MultiThreadedExecutor 하나). 공유 상태는 `_lock` 아래에서만 바꾼다. SIGINT → main 이 release 경로를 밟는다.
"""
from __future__ import annotations

import json
import re
import signal
import threading
import time
from pathlib import Path

import numpy as np
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

if __package__ in (None, ""):          # `python policy_control/policy_control/<node>.py` (launch use_source 모드)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"

from . import codec  # noqa: E402
from .chain import ChainError, StageStatus  # noqa: E402
from .codec import CodecError  # noqa: E402
from .contract import load_contract  # noqa: E402
from .pd_arm import ArmUnit, PdArmError, TickResult, select_sides  # noqa: E402
from .pd_backends import BackendError  # noqa: E402
from .pd_law import load_pd_config  # noqa: E402
from .pd_state import Phase  # noqa: E402
from .sources import RobotCfgError, load_profile, load_robot_cfg  # noqa: E402

NS = "/policy_control"
NODE_NAME = "pd_node"
STAGE_NODE = "pd"
PHASE_PRECEDENCE = (Phase.HOLD, Phase.RELEASING, Phase.RAMPING, Phase.TRACKING, Phase.IDLE)
_MOVING = (Phase.RAMPING, Phase.TRACKING)
_HANDLED = (CodecError, ChainError, BackendError, RobotCfgError, PdArmError, ValueError, KeyError)
_NUM_RE = re.compile(r"[-+]?\d+(\.\d+)?")


class PdNodeError(RuntimeError):
    """노드 배선/설정 오류(기동 거부) 또는 tick 입력 오류(status 사유)."""


# ------------------------------------------------------------------ helpers
def _qos_chain():
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    return QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)


def _qos_latched():
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)  # 구독 깊이 10: reset 직후 start 가 오면 depth 1 은 reset 을 버린다(run15)


def _qos_sensor():
    from rclpy.qos import qos_profile_sensor_data

    return qos_profile_sensor_data


def compact_reasons(joined: str | None) -> list[str]:
    """HOLD 사유는 tick 마다 숫자만 다른 문장이 누적된다('watchdog: target stale 0.252 s', 0.262 s …) —
    숫자를 뺀 본문이 같으면 첫 것만 남긴다(status 크기 상한)."""
    if not joined:
        return []
    seen, out = set(), []
    for r in joined.split("; "):
        key = _NUM_RE.sub("#", r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def trigger_reply(resp, ok: bool, reasons) -> object:
    resp.success = bool(ok)
    resp.message = json.dumps({"ok": bool(ok), "reasons": [str(r) for r in reasons]}, ensure_ascii=False)
    return resp


def aggregate_phase(phases) -> Phase:
    """팔들의 phase 합성: HOLD > RELEASING > RAMPING > TRACKING > IDLE (하나라도 서면 HOLD 로 보인다)."""
    phases = list(phases)
    for p in PHASE_PRECEDENCE:
        if p in phases:
            return p
    return Phase.IDLE


def parse_target_frame(frame_id: str) -> tuple[str, int]:
    ep, _, seq = str(frame_id).rpartition(":")
    if not ep or not seq.isdigit():
        raise CodecError(f"joint_target frame_id {frame_id!r} 는 '<episode>:<seq>' 가 아니다")
    return ep, int(seq)


# ================================================================== node
class PdNode(Node):
    """subscribe → codec.decode → ArmUnit.tick(타이머, 팔마다) → backends.write / codec.encode → publish."""

    def __init__(self, *, context=None, parameter_overrides=None) -> None:
        super().__init__(NODE_NAME, context=context, parameter_overrides=parameter_overrides)
        self._declare_params()
        self.cfg = load_pd_config(self._path("pd_config"))
        self.contract = load_contract(self._path("contract"))
        self.robot_cfg = load_robot_cfg(self._path("robot"))
        self.profile = load_profile(self.robot_cfg.joint_profiles)
        self.execute = bool(self.get_parameter("execute").value) and bool(self.cfg.execute)
        self.stage_name = str(self.get_parameter("stage").value)
        self.sides = select_sides(str(self.get_parameter("sides").value), self.contract, self.robot_cfg)
        self.units: dict[str, ArmUnit] = {
            s: ArmUnit(self, side_cfg=self.contract.side(s), robot_cfg=self.robot_cfg, cfg=self.cfg,
                       contract=self.contract, profile=self.profile, stage_name=self.stage_name, execute=self.execute)
            for s in self.sides}
        self.dt = 1.0 / self.cfg.pd_hz
        self._lock = threading.Lock()
        self._estop = False
        self._last_error: str | None = None
        self._last_status: dict = {}
        self._wire()
        self.get_logger().info(
            f"pd_node up · sides {self.sides} · execute {self.execute} (param {bool(self.get_parameter('execute').value)}"
            f" ∧ yaml {self.cfg.execute}) · stage {self.stage_name} · gains ok "
            f"{ {s: u.gains_ok for s, u in self.units.items()} } · gravity {self.cfg.gravity.mode}")

    # ---------------------------------------------------------------- setup
    def _declare_params(self) -> None:
        self.declare_parameter("contract", "")
        self.declare_parameter("robot", "")
        self.declare_parameter("pd_config", "")
        self.declare_parameter("execute", False)
        self.declare_parameter("stage", "reduced")
        self.declare_parameter("sides", "")
        self.declare_parameter("goto_home_timeout_sec", 30.0)

    def _path(self, name: str) -> Path:
        p = Path(str(self.get_parameter(name).value)).expanduser()
        if not p.is_file():
            raise PdNodeError(f"parameter {name}: 파일이 없다 — {p}")
        return p

    def _wire(self) -> None:
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, String
        from std_srvs.srv import Trigger

        self.cb_main = MutuallyExclusiveCallbackGroup()
        self.cb_srv = MutuallyExclusiveCallbackGroup()
        self._pub_applied = self.create_publisher(JointState, f"{NS}/pd/applied", _qos_chain())
        self._pub_status = self.create_publisher(String, f"{NS}/status/{STAGE_NODE}", _qos_chain())
        main = self.cb_main
        self.create_subscription(JointState, f"{NS}/joint_target", self._on_target, _qos_chain(), callback_group=main)
        self.create_subscription(String, f"{NS}/episode", self._on_episode, _qos_latched(), callback_group=main)
        self.create_subscription(Bool, f"{NS}/estop", self._on_estop, _qos_latched(), callback_group=main)
        topics: dict[str, list] = {}
        for unit in self.units.values():
            for topic, roles in unit.joint_topics().items():
                topics.setdefault(topic, []).append((unit, roles))
        for topic, targets in topics.items():
            self.create_subscription(JointState, topic, self._joint_cb(targets), _qos_sensor(), callback_group=main)
        services = (("engage", self._srv_engage), ("goto_home", self._srv_goto_home), ("release", self._srv_release))
        for name, fn in services:
            self.create_service(Trigger, f"{NS}/pd/{name}", fn, callback_group=self.cb_srv)
        self.create_timer(self.dt, self._on_timer, callback_group=main)

    # ---------------------------------------------------------------- subscriptions
    def _joint_cb(self, targets: list):
        def cb(msg) -> None:
            try:
                sample = codec.decode_joint_state(msg)
                now = time.monotonic()
                with self._lock:
                    for unit, roles in targets:
                        unit.on_joint_state(roles, sample, now)
            except _HANDLED as exc:
                self._note_error(f"joint_state({[r for _, r in targets]}): {exc}")
        return cb

    def _on_target(self, msg) -> None:
        try:
            sample = codec.decode_joint_state(msg)
            _, seq = parse_target_frame(msg.header.frame_id)
            now = time.monotonic()
            with self._lock:
                taken = [u.side for u in self.units.values() if u.take_target(sample, seq, now)]
        except _HANDLED as exc:
            self._note_error(f"joint_target: {exc}")
            return
        if not taken:
            self._note_error(f"joint_target: no selected side's arm joints among {list(msg.name)[:8]}…")

    def _on_episode(self, msg) -> None:
        try:
            ev = json.loads(msg.data)
            event, episode = str(ev["event"]), int(ev["episode"])
        except (TypeError, ValueError, KeyError) as exc:
            self._note_error(f"episode JSON: {exc}")
            return
        with self._lock:
            for unit in self.units.values():
                unit.on_episode(event, episode)

    def _on_estop(self, msg) -> None:
        with self._lock:
            self._estop = bool(msg.data)
        if self._estop:
            self.get_logger().error("estop latched")

    def _note_error(self, text: str) -> None:
        if text != self._last_error:
            self.get_logger().warning(text)
        self._last_error = text

    # ---------------------------------------------------------------- timer
    def _on_timer(self) -> None:
        t0 = time.perf_counter()
        with self._lock:
            try:
                self._tick(time.monotonic(), t0)
            except _HANDLED as exc:
                self._note_error(f"tick: {exc}")
                self._publish_status([(u, TickResult(None, None, u.idle_status())) for u in self.units.values()],
                                     t0, reasons=(f"tick: {exc}",))

    def _tick(self, now: float, t0: float) -> None:
        results = [(u, u.tick(now, self._estop)) for u in self.units.values()]
        for unit, r in results:
            if r.error and unit.engaged:
                self._note_error(f"no fresh joint state ({r.error}) — command frozen")
        self._publish_applied(results)
        self._publish_status(results, t0)

    def _publish_applied(self, results: list) -> None:
        names, q, qd, tau = [], [], [], []
        for unit, r in results:
            if r.cmd is None:
                continue
            names += unit.arm_joints
            q.append(r.cmd.q), qd.append(r.cmd.qd), tau.append(r.cmd.tau)
            if r.hand_written is not None:
                names += unit.hand_joints
                q.append(r.hand_written), qd.append(np.zeros(len(r.hand_written))), tau.append(np.zeros(len(r.hand_written)))
        if not names:
            return
        self._pub_applied.publish(codec.encode_joint_state(names, np.concatenate(q), velocity=np.concatenate(qd),
                                                           effort=np.concatenate(tau), stamp=time.time()))

    def _tag(self, unit: ArmUnit, text: str) -> str:
        return f"{unit.side}: {text}" if len(self.units) > 1 else text

    def _side_status(self, unit: ArmUnit, r: TickResult) -> dict:
        body = r.status.as_dict()
        body["reasons"] = compact_reasons("; ".join(r.status.reasons)) + ([r.error] if r.error else [])
        body["ok"] = bool(r.status.ok) and r.error is None
        body.update(unit.extras())
        return body

    def _publish_status(self, results: list, t0: float, reasons: tuple = ()) -> None:
        arms = {u.side: self._side_status(u, r) for u, r in results}
        first = next(iter(arms.values()))
        phase = aggregate_phase(u.phase for u, _ in results)
        seqs = [a["seq"] for a in arms.values() if a["seq"] >= 0]
        body = {
            "node": STAGE_NODE, "phase": phase.value, "episode": first["episode"], "seq": max(seqs) if seqs else first["seq"],
            "ok": all(a["ok"] for a in arms.values()) and not reasons,
            "reasons": [self._tag(u, x) for u, _ in results for x in arms[u.side]["reasons"]] + [r for r in reasons if r],
            "stage_ms": max(a["proc_ms"] for a in arms.values()), "proc_ms": (time.perf_counter() - t0) * 1e3,
            "execute": self.execute, "stage_cfg": self.stage_name, "sides": list(arms), "estop": self._estop,
            "gains": first["gains"], "thermal": {k: v for a in arms.values() for k, v in a["thermal"].items()},
            "blend": first["blend"], "hold": first["hold"], "target": first["target"], "arms": arms,
            "last_error": self._last_error, "t_pub_ns": time.time_ns()}
        self._last_status = body
        self._pub_status.publish(codec.encode_status(body))

    # ---------------------------------------------------------------- services
    def _srv_engage(self, req, resp):
        """선택한 팔을 순서대로(우 먼저): 거부 사유 → 손 PID → 컨트롤러 교대 → seed → RAMPING."""
        notes: list[str] = []
        for unit in self.units.values():
            states = unit.list_controllers()                              # 블로킹 호출은 락 밖에서
            with self._lock:
                refusals = unit.engage_refusals(states, self._estop, time.monotonic())
            if refusals:
                return trigger_reply(resp, False, notes + [self._tag(unit, r) for r in refusals])
            for step in (unit.apply_hand_gains, unit.switch_engage):
                ok, why = step()
                notes += [self._tag(unit, w) for w in why]
                if not ok:
                    return trigger_reply(resp, False, notes)
            ref = unit.read_reference(self)
            with self._lock:
                q_seed, note = unit.seed_from(ref, time.monotonic())
                phase = unit.engage_stage(q_seed, time.monotonic())
            notes += [note, self._tag(unit, f"phase {phase}")]
        return trigger_reply(resp, True, notes)

    def _srv_goto_home(self, req, resp):
        """계약 홈으로 0.1 rad/s 램프 + settle — 팔을 순서대로(우 먼저, 양팔 리셋 규약)."""
        notes: list[str] = []
        timeout = float(self.get_parameter("goto_home_timeout_sec").value)
        for unit in self.units.values():
            with self._lock:
                phase = unit.phase
                if phase not in _MOVING:
                    why = (f"phase {phase.value} is not RAMPING/TRACKING (engage first)" if phase is Phase.IDLE
                           else f"phase {phase.value} — release first")
                    return trigger_reply(resp, False, notes + [self._tag(unit, why)])
                unit.start_home()
            settled, why = self._wait(unit.home_settled, timeout, self._tag(unit, "settle"), hold_units=(unit,))
            with self._lock:
                phase, err = unit.phase.value, (None if unit.hold is None else unit.hold.err)
            notes += [*why, self._tag(unit, f"phase {phase}"), self._tag(unit, f"home err {err}")]
            if not (settled and phase == "TRACKING"):
                return trigger_reply(resp, False, notes)
        return trigger_reply(resp, True, notes)

    def _srv_release(self, req, resp):
        with self._lock:
            if not any(u.engaged for u in self.units.values()):
                return trigger_reply(resp, True, ["already IDLE — nothing to release"])
        ok, reasons = self.release_path()
        return trigger_reply(resp, ok, reasons)

    def _wait(self, done, timeout: float, what: str, *, hold_units: tuple = ()) -> tuple[bool, list[str]]:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self._lock:
                if done():
                    return True, []
                for u in hold_units:
                    if u.phase is Phase.HOLD:
                        return False, [f"{what}: HOLD ({'; '.join(compact_reasons(u.stage.state.fsm.hold_reason))})"]
            time.sleep(0.02)
        return False, [f"{what}: timeout {timeout:.1f}s"]

    def release_path(self) -> tuple[bool, list[str]]:
        """모든 engaged 팔: 역블렌드 → RELEASING 0 송출 ×N → IDLE → zero_release → switch → JTC. 서비스와 SIGINT 가 같이 쓴다."""
        now = time.monotonic()
        with self._lock:
            active = [u for u in self.units.values() if u.engaged or u.phase is Phase.RELEASING]
            if not active:
                return True, ["already IDLE"]
            for u in active:
                u.start_release(now)
        budget = self.cfg.blend_sec + self.cfg.release_zero_ticks * self.dt + 2.0
        idle, why = self._wait(lambda: all(u.phase is Phase.IDLE for u in active), budget, "release")
        with self._lock:
            for u in active:
                u.zero_release()
        ok_all, reasons = idle, list(why)
        for u in active:
            ok, r = u.switch_release()
            ok_all = ok_all and ok
            reasons += [self._tag(u, x) for x in r]
        return ok_all, reasons

    def shutdown_release(self) -> None:
        """SIGINT/SIGTERM: executor 가 멈춘 뒤 타이머 없이 tick 을 직접 돌려 release 경로를 끝낸다."""
        with self._lock:
            engaged = [u.side for u in self.units.values() if u.engaged]
        if not engaged:
            return
        self.get_logger().warning(f"shutdown with {engaged} engaged — releasing")
        with self._lock:
            for u in self.units.values():
                if u.engaged:
                    u.start_release(time.monotonic())
        deadline = time.monotonic() + self.cfg.blend_sec + self.cfg.release_zero_ticks * self.dt + 2.0
        while time.monotonic() < deadline:
            with self._lock:
                if all(u.phase is Phase.IDLE for u in self.units.values()):
                    break
                self._tick(time.monotonic(), time.perf_counter())
            time.sleep(self.dt)
        ok, reasons = self.release_path()
        self.get_logger().info(f"release {'ok' if ok else 'FAILED'}: {'; '.join(reasons)}")

    def close(self) -> None:
        for u in self.units.values():
            u.close()


def main(argv=None) -> int:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(args=argv, signal_handler_options=SignalHandlerOptions.NO)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    node = PdNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        while rclpy.ok() and not stop.is_set():
            executor.spin_once(timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.shutdown_release()
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
