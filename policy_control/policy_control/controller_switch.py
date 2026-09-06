"""controller_manager 클라이언트 래퍼 — JTC ↔ forward 3종 교대 (플랜 §4.4 engage/release).

규약
- 모든 호출은 ``(ok: bool, reasons: list[str])`` 를 돌려주고 **절대 예외를 던지지 않는다**
  (서비스 부재·타임아웃·STRICT 거부는 전부 reasons 로).
- ``execute=False`` 면 controller_manager 서비스를 **한 건도 부르지 않는다**(list 포함). 결과는
  ``ok=True`` + ``"dry_run: …"`` 사유 — pd 노드의 execute:=false 기본값에서 상태기계가 그대로
  돌 수 있게 하되, 하드웨어 쪽 부작용은 0 이어야 한다(테스트가 잠근다).
- 서비스는 **별도 helper 노드 + 전용 executor** 로 부른다. 호출자 노드의 콜백(서비스/타이머)
  안에서 불러도 자기 executor 를 기다리지 않아 교착이 없다.
- ``SetHardwareComponentState`` 는 여기 없다 — 부르면 return_to_zero/disable_all 을 유발한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

CM_NS = "/controller_manager"
ARM_JOINT_COUNT = 7
DEFAULT_TIMEOUT_SEC = 3.0


def source_arm_joints(side: str) -> list[str]:
    """robot_control bringup 의 팔 source 관절명(컨트롤러 joints 순)."""
    return [f"openarm_{side}_joint{i}" for i in range(1, ARM_JOINT_COUNT + 1)]


def jtc_state_topic(side: str) -> str:
    return f"/{side}_joint_trajectory_controller/controller_state"


class ServiceCaller:
    """helper 노드 위의 블로킹 서비스 호출. 결과는 ``(response | None, reason | None)``."""

    def __init__(self, node, suffix: str, timeout_sec: float) -> None:
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node

        if timeout_sec <= 0.0:
            raise ValueError("timeout_sec 는 양수여야 한다")
        self.timeout_sec = float(timeout_sec)
        self._node = Node(f"{node.get_name()}_{suffix}", context=node.context)
        self._exec = SingleThreadedExecutor(context=node.context)
        self._exec.add_node(self._node)
        self._clients: dict[str, Any] = {}
        self.logger = node.get_logger()

    @property
    def node(self):
        return self._node

    def call(self, srv_type, service: str, request) -> tuple[Any | None, str | None]:
        client = self._clients.get(service)
        if client is None:
            client = self._node.create_client(srv_type, service)
            self._clients[service] = client
        if not client.wait_for_service(timeout_sec=self.timeout_sec):
            return None, f"service {service} unavailable ({self.timeout_sec:.1f}s)"
        future = client.call_async(request)
        self._exec.spin_until_future_complete(future, timeout_sec=self.timeout_sec)
        if not future.done():
            return None, f"service {service} timeout ({self.timeout_sec:.1f}s)"
        exc = future.exception()
        if exc is not None:
            return None, f"service {service} failed: {exc}"
        return future.result(), None

    def spin_for(self, seconds: float, until=None) -> None:
        """helper 노드를 잠깐 돌린다(구독 대기용). ``until()`` 이 참이 되면 일찍 끝난다."""
        import time

        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            if until is not None and until():
                return
            self._exec.spin_once(timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))

    def close(self) -> None:
        self._exec.remove_node(self._node)
        self._exec.shutdown()
        self._node.destroy_node()


@dataclass(frozen=True)
class SwitchPlan:
    activate: tuple[str, ...]
    deactivate: tuple[str, ...]


class ControllerSwitch:
    """JTC ↔ forward(position/velocity/effort) 교대 클라이언트."""

    def __init__(self, node, side: str, forward_names: Sequence[str], jtc_name: str,
                 timeout_sec: float = DEFAULT_TIMEOUT_SEC, *, execute: bool = True,
                 cm_ns: str = CM_NS) -> None:
        if len(forward_names) != 3:
            raise ValueError(f"forward_names 는 position/velocity/effort 3개여야 한다: {list(forward_names)}")
        self.side = str(side)
        self.forward = tuple(str(n) for n in forward_names)
        self.jtc = str(jtc_name)
        self.execute = bool(execute)
        self.cm_ns = cm_ns.rstrip("/")
        self._caller = ServiceCaller(node, "cm_client", timeout_sec)
        self.logger = node.get_logger()

    def close(self) -> None:
        self._caller.close()

    # ------------------------------------------------------------ queries
    def list(self) -> dict[str, str]:
        """{controller: state}. 실패·dry-run 이면 {} (사유는 로그)."""
        if not self.execute:
            return {}
        from controller_manager_msgs.srv import ListControllers

        resp, reason = self._caller.call(ListControllers, f"{self.cm_ns}/list_controllers",
                                         ListControllers.Request())
        if resp is None:
            self.logger.warning(f"[controller_switch] list_controllers: {reason}")
            return {}
        return {c.name: c.state for c in resp.controller}

    # ------------------------------------------------------------ mutations
    def ensure_loaded_inactive(self, names: Sequence[str], *, allow_active: bool = False
                               ) -> tuple[bool, list[str]]:
        """미로드 컨트롤러를 load+configure 해 inactive 로 둔다. 이미 active 면 거부(allow_active 로만 통과)."""
        names = [str(n) for n in names]
        if not self.execute:
            return True, [f"dry_run: would load+configure {names}"]
        states = self.list()
        if not states:
            return False, ["list_controllers returned nothing — controller_manager unavailable?"]
        active = [n for n in names if states.get(n) == "active"]
        if active and not allow_active:
            return False, [f"forward controller already active: {active} (another node owns it — refuse)"]
        reasons: list[str] = []
        for name in names:
            state = states.get(name)
            if state is None:
                if not self._load(name, reasons):
                    return False, reasons
                state = "unconfigured"
            if state == "unconfigured" and not self._configure(name, reasons):
                return False, reasons
        return True, reasons

    def engage(self) -> tuple[bool, list[str]]:
        """switch(activate=forward 3, deactivate=[JTC], STRICT, activate_asap)."""
        return self._switch(SwitchPlan(self.forward, (self.jtc,)), label="engage")

    def release(self) -> tuple[bool, list[str]]:
        """역교대: STRICT 로 시도, 거부되면 BEST_EFFORT 로 한 번 더(팔을 컨트롤러 없이 두지 않는다)."""
        plan = SwitchPlan((self.jtc,), self.forward)
        ok, reasons = self._switch(plan, label="release")
        if ok or not self.execute:
            return ok, reasons
        ok2, reasons2 = self._switch(plan, label="release(best_effort)", best_effort=True)
        return ok2, reasons + reasons2

    # ------------------------------------------------------------ primitives
    def _load(self, name: str, reasons: list[str]) -> bool:
        from controller_manager_msgs.srv import LoadController

        req = LoadController.Request()
        req.name = name
        resp, reason = self._caller.call(LoadController, f"{self.cm_ns}/load_controller", req)
        if resp is None or not resp.ok:
            reasons.append(f"load_controller {name} failed: {reason or 'ok=False'}")
            return False
        reasons.append(f"loaded {name}")
        return True

    def _configure(self, name: str, reasons: list[str]) -> bool:
        from controller_manager_msgs.srv import ConfigureController

        req = ConfigureController.Request()
        req.name = name
        resp, reason = self._caller.call(ConfigureController, f"{self.cm_ns}/configure_controller", req)
        if resp is None or not resp.ok:
            reasons.append(f"configure_controller {name} failed: {reason or 'ok=False'}")
            return False
        reasons.append(f"configured {name}")
        return True

    def _switch(self, plan: SwitchPlan, *, label: str, best_effort: bool = False) -> tuple[bool, list[str]]:
        if not self.execute:
            return True, [f"dry_run: would switch activate={list(plan.activate)} deactivate={list(plan.deactivate)}"]
        from controller_manager_msgs.srv import SwitchController

        req = SwitchController.Request()
        req.activate_controllers = list(plan.activate)
        req.deactivate_controllers = list(plan.deactivate)
        req.strictness = req.BEST_EFFORT if best_effort else req.STRICT
        req.activate_asap = True
        resp, reason = self._caller.call(SwitchController, f"{self.cm_ns}/switch_controller", req)
        if resp is None:
            return False, [f"{label} switch_controller: {reason}"]
        if not resp.ok:
            mode = "BEST_EFFORT" if best_effort else "STRICT"
            return False, [f"{label} switch_controller refused ({mode}): activate={list(plan.activate)} "
                           f"deactivate={list(plan.deactivate)}"]
        return True, [f"{label} switch ok"]


# ---------------------------------------------------------------- JTC reference
def _pick_positions(msg, n: int) -> np.ndarray | None:
    """reference → desired → actual(실측) 순으로 길이 n 인 첫 벡터."""
    for pt in (msg.reference, msg.desired, msg.actual):
        if len(pt.positions) == n:
            return np.asarray(pt.positions, dtype=np.float64)
    return None


def _reorder(names: Sequence[str], values: np.ndarray, wanted: Sequence[str]) -> np.ndarray | None:
    idx = {str(n): i for i, n in enumerate(names)}
    if any(w not in idx for w in wanted):
        return None
    return np.array([values[idx[w]] for w in wanted], dtype=np.float64)


def read_jtc_reference(node, side: str, max_age_sec: float, wait_sec: float | None = None
                       ) -> np.ndarray | None:
    """JTC controller_state 의 세트포인트(SOURCE 순 openarm_<side>_joint1..7) 또는 None.

    ``max_age_sec`` 보다 오래된 메시지·관절명 불일치·수신 없음 → None (호출자가 실측으로 대체).
    ``wait_sec`` (기본 max_age_sec) 동안 helper 노드를 돌려 첫 신선한 메시지를 기다린다.
    """
    from control_msgs.msg import JointTrajectoryControllerState
    from rclpy.time import Time

    wanted = source_arm_joints(side)
    caller = ServiceCaller(node, "jtc_ref", timeout_sec=max(0.05, float(max_age_sec)))
    fresh: dict[str, np.ndarray | None] = {"q": None}

    def on_state(msg) -> None:
        stamp = Time.from_msg(msg.header.stamp, clock_type=caller.node.get_clock().clock_type)
        age = (caller.node.get_clock().now() - stamp).nanoseconds * 1e-9
        if stamp.nanoseconds > 0 and age > max_age_sec:
            return
        vals = _pick_positions(msg, len(msg.joint_names))
        if vals is None:
            return
        fresh["q"] = _reorder(msg.joint_names, vals, wanted)

    sub = caller.node.create_subscription(JointTrajectoryControllerState, jtc_state_topic(side), on_state, 10)
    try:
        caller.spin_for(max_age_sec if wait_sec is None else wait_sec, until=lambda: fresh["q"] is not None)
    finally:
        caller.node.destroy_subscription(sub)
        caller.close()
    return fresh["q"]
