"""controller_switch — controller_manager 클라이언트 래퍼 (ros 마커, 격리 도메인).

서버는 scripts/fakes/fake_arm_bridge.ControllerManagerStub (STRICT 의미론, 실기 bringup 초기
상태 = JTC active · forward 3종 미로드) 를 인프로세스로 띄운다.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytestmark = pytest.mark.ros

SIDE = "left"
JTC = f"{SIDE}_joint_trajectory_controller"
FWD = [f"{SIDE}_forward_{k}_controller" for k in ("position", "velocity", "effort")]
SRC = [f"openarm_{SIDE}_joint{i}" for i in range(1, 8)]


class _Spinner:
    """백그라운드 스레드에서 노드들을 돌린다(서비스 서버·프로브용)."""

    def __init__(self, context, *nodes):
        from rclpy.executors import SingleThreadedExecutor

        self.exec = SingleThreadedExecutor(context=context)
        for n in nodes:
            self.exec.add_node(n)
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        while not self._stop.is_set():
            self.exec.spin_once(timeout_sec=0.02)

    def close(self):
        self._stop.set()
        self._th.join(timeout=2.0)
        self.exec.shutdown()


@pytest.fixture
def cm(ros):
    """fake controller_manager 서버 + 클라이언트 노드."""
    from rclpy.node import Node

    from fake_arm_bridge import ControllerManagerStub

    server = Node("fake_cm", context=ros)
    stub = ControllerManagerStub(server, SIDE)
    client = Node("pd_test", context=ros)
    spin = _Spinner(ros, server)
    try:
        yield stub, client
    finally:
        spin.close()
        client.destroy_node()
        server.destroy_node()


def _switch(client, execute=True, timeout=2.0):
    from policy_control.controller_switch import ControllerSwitch

    return ControllerSwitch(client, SIDE, FWD, JTC, timeout_sec=timeout, execute=execute)


def test_list_reports_initial_bringup_state(cm):
    stub, client = cm
    sw = _switch(client)
    states = sw.list()
    assert states[JTC] == "active"
    assert not any(n in states for n in FWD)
    sw.close()


def test_ensure_engage_release_sequence(cm):
    stub, client = cm
    sw = _switch(client)
    ok, reasons = sw.ensure_loaded_inactive(FWD)
    assert ok, reasons
    assert all(stub.known[n] == "inactive" for n in FWD)

    ok, reasons = sw.engage()
    assert ok, reasons
    assert all(stub.known[n] == "active" for n in FWD)
    assert stub.known[JTC] == "inactive"

    ok, reasons = sw.release()
    assert ok, reasons
    assert all(stub.known[n] == "inactive" for n in FWD)
    assert stub.known[JTC] == "active"
    sw.close()


def test_ensure_is_idempotent(cm):
    stub, client = cm
    sw = _switch(client)
    assert sw.ensure_loaded_inactive(FWD)[0]
    ok, reasons = sw.ensure_loaded_inactive(FWD)      # 이미 inactive → 아무 것도 안 하고 ok
    assert ok, reasons
    sw.close()


def test_engage_without_load_is_refused_not_raised(cm):
    stub, client = cm
    sw = _switch(client)
    ok, reasons = sw.engage()
    assert ok is False
    assert reasons and any("switch" in r for r in reasons)
    assert stub.known[JTC] == "active"                  # STRICT 거부 = 상태 불변
    sw.close()


def test_ensure_refuses_already_active_forward(cm):
    stub, client = cm
    sw = _switch(client)
    assert sw.ensure_loaded_inactive(FWD)[0]
    assert sw.engage()[0]
    other = _switch(client)                             # 두 번째 pd (예: gravity_comp 가 effort 를 잡은 상황)
    ok, reasons = other.ensure_loaded_inactive(FWD)
    assert ok is False
    assert any("active" in r for r in reasons)
    ok, reasons = other.ensure_loaded_inactive(FWD, allow_active=True)
    assert ok, reasons
    other.close()
    sw.close()


def test_dry_run_calls_no_service(cm):
    stub, client = cm
    calls = {"n": 0}
    for name in ("_list", "_load", "_configure", "_switch"):
        orig = getattr(stub, name)

        def wrapped(req, res, _orig=orig):
            calls["n"] += 1
            return _orig(req, res)

        setattr(stub, name, wrapped)
    sw = _switch(client, execute=False)
    assert sw.list() == {}
    ok, reasons = sw.ensure_loaded_inactive(FWD)
    assert ok and any("dry_run" in r for r in reasons)
    ok, reasons = sw.engage()
    assert ok and any("dry_run" in r for r in reasons)
    ok, reasons = sw.release()
    assert ok and any("dry_run" in r for r in reasons)
    time.sleep(0.1)
    assert calls["n"] == 0
    assert stub.known[JTC] == "active" and not any(n in stub.known for n in FWD)
    sw.close()


def test_missing_server_returns_reasons(ros):
    from rclpy.node import Node

    client = Node("pd_test_noserver", context=ros)
    sw = _switch(client, timeout=0.2)
    ok, reasons = sw.engage()
    assert ok is False and reasons and any("unavailable" in r for r in reasons)
    ok, reasons = sw.release()
    assert ok is False and reasons
    assert sw.list() == {}
    sw.close()
    client.destroy_node()


# ------------------------------------------------------------------ read_jtc_reference
def _state_msg(node, names, reference=None, desired=None, actual=None, age_sec=0.0):
    from control_msgs.msg import JointTrajectoryControllerState
    from rclpy.duration import Duration

    msg = JointTrajectoryControllerState()
    stamp = node.get_clock().now() - Duration(seconds=age_sec)
    msg.header.stamp = stamp.to_msg()
    msg.joint_names = list(names)
    if reference is not None:
        msg.reference.positions = [float(v) for v in reference]
    if desired is not None:
        msg.desired.positions = [float(v) for v in desired]
    if actual is not None:
        msg.actual.positions = [float(v) for v in actual]
    return msg


@pytest.fixture
def jtc_pub(ros):
    from control_msgs.msg import JointTrajectoryControllerState
    from rclpy.node import Node

    pub_node = Node("fake_jtc", context=ros)
    pub = pub_node.create_publisher(JointTrajectoryControllerState, f"/{JTC}/controller_state", 10)
    reader = Node("pd_test_reader", context=ros)
    stop = threading.Event()
    holder = {"msg": None}

    def loop():
        while not stop.is_set():
            if holder["msg"] is not None:
                pub.publish(holder["msg"])
            time.sleep(0.02)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    try:
        yield pub_node, holder, reader
    finally:
        stop.set()
        th.join(timeout=1.0)
        reader.destroy_node()
        pub_node.destroy_node()


def test_read_jtc_reference_reorders_to_source_and_prefers_reference(jtc_pub):
    from policy_control.controller_switch import read_jtc_reference

    pub_node, holder, reader = jtc_pub
    shuffled = SRC[::-1]
    ref = np.arange(7, dtype=float) * 0.1
    holder["msg"] = _state_msg(pub_node, shuffled, reference=ref, desired=ref + 1.0, actual=ref + 2.0)
    out = read_jtc_reference(reader, SIDE, max_age_sec=0.5, wait_sec=2.0)
    assert out is not None
    np.testing.assert_allclose(out, ref[::-1])            # source 순 openarm_left_joint1..7


def test_read_jtc_reference_falls_back_to_desired_then_actual(jtc_pub):
    from policy_control.controller_switch import read_jtc_reference

    pub_node, holder, reader = jtc_pub
    ref = np.ones(7)
    holder["msg"] = _state_msg(pub_node, SRC, desired=ref * 2, actual=ref * 3)
    np.testing.assert_allclose(read_jtc_reference(reader, SIDE, 0.5, wait_sec=2.0), ref * 2)
    holder["msg"] = _state_msg(pub_node, SRC, actual=ref * 3)
    np.testing.assert_allclose(read_jtc_reference(reader, SIDE, 0.5, wait_sec=2.0), ref * 3)


def test_read_jtc_reference_rejects_stale_or_missing(jtc_pub):
    from policy_control.controller_switch import read_jtc_reference

    pub_node, holder, reader = jtc_pub
    holder["msg"] = _state_msg(pub_node, SRC, reference=np.zeros(7), age_sec=5.0)
    assert read_jtc_reference(reader, SIDE, max_age_sec=0.5, wait_sec=0.5) is None
    holder["msg"] = _state_msg(pub_node, SRC[:6] + ["bogus"], reference=np.zeros(7))
    assert read_jtc_reference(reader, SIDE, max_age_sec=0.5, wait_sec=0.5) is None
    holder["msg"] = None
    assert read_jtc_reference(reader, SIDE, max_age_sec=0.5, wait_sec=0.3) is None


# ------------------------------------------------------------------ 09.06 양팔 controller_manager 스텁 — 팔은 서로 독립
@pytest.fixture
def bi_cm(ros):
    from rclpy.node import Node

    from fake_cm_stub import ControllerManagerStub

    server = Node("fake_cm_bi", context=ros)
    stub = ControllerManagerStub(server, ("right", "left"))
    client = Node("pd_test_bi", context=ros)
    spin = _Spinner(ros, server)
    try:
        yield stub, client
    finally:
        spin.close()
        client.destroy_node()
        server.destroy_node()


def _fwd(side):
    return [f"{side}_forward_{k}_controller" for k in ("position", "velocity", "effort")]


def test_bimanual_stub_starts_with_both_jtcs_active_and_no_forward(bi_cm):
    from policy_control.controller_switch import ControllerSwitch

    stub, client = bi_cm
    assert stub.sides == ("right", "left") and stub.side == "right" and stub.jtc == "right_joint_trajectory_controller"
    sw = ControllerSwitch(client, "left", _fwd("left"), "left_joint_trajectory_controller", timeout_sec=2.0)
    states = sw.list()
    assert states["left_joint_trajectory_controller"] == "active" and states["right_joint_trajectory_controller"] == "active"
    assert not any(n in states for n in _fwd("left") + _fwd("right"))
    sw.close()


def test_bimanual_switches_are_independent(bi_cm):
    from policy_control.controller_switch import ControllerSwitch

    stub, client = bi_cm
    right = ControllerSwitch(client, "right", _fwd("right"), "right_joint_trajectory_controller", timeout_sec=2.0)
    left = ControllerSwitch(client, "left", _fwd("left"), "left_joint_trajectory_controller", timeout_sec=2.0)
    assert right.ensure_loaded_inactive(right.forward)[0] and right.engage()[0]
    assert all(stub.forward_active("right", k) for k in ("position", "velocity", "effort"))
    assert stub.known["left_joint_trajectory_controller"] == "active" and not any(n in stub.known for n in _fwd("left"))
    assert left.ensure_loaded_inactive(left.forward)[0] and left.engage()[0]
    assert stub.known["left_joint_trajectory_controller"] == "inactive" and stub.known["right_joint_trajectory_controller"] == "inactive"
    assert right.release()[0]                                    # 우팔만 돌려놓는다
    assert stub.known["right_joint_trajectory_controller"] == "active" and stub.forward_active("left", "position")
    assert left.release()[0]
    assert stub.known["left_joint_trajectory_controller"] == "active" and not stub.forward_active("left", "position")
    right.close()
    left.close()
