"""pd_backends — 출력 백엔드 3종 + 손 PID 클라이언트 (ros 마커, 격리 도메인).

프로브 구독자가 source 순서·부호·단일점·overtravel 을 확인하고, execute=False 에서는
발행·서비스 호출이 0건임을 잠근다.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytestmark = pytest.mark.ros

SIDE = "left"
CAN = [f"l_aj_{i}" for i in range(1, 8)]
SRC = [f"openarm_{SIDE}_joint{i}" for i in range(1, 8)]
FWD_TOPICS = {k: f"/{SIDE}_forward_{k}_controller/commands" for k in ("position", "velocity", "effort")}
GRIP_TOPIC = "/left_gripper_controller/joint_trajectory"
GRIP_JOINT = "openarm_left_finger_joint1"
HAND_TOPIC = "/dg5f_right/dg5f_right_controller/joint_trajectory"
HAND_CTRL = "/dg5f_right/dg5f_right_controller"
HAND_SRC = [f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HAND_CAN = [f"r_hj_{f}_{j}" for f in FINGERS for j in range(1, 5)]


def _profile_path():
    from policy_control import _paths

    return _paths.ROBOT_CONTROL_SRC / "robot_control" / "profiles" / "openarm_tesollo.yaml"


def _synthetic_profile(signs):
    """canonical → source, 지정 부호, 넓은 한계 (부호·순서 검증용)."""
    return {c: {"source": s, "sign": float(g), "lower": -10.0, "upper": 10.0, "unit": "rad",
                "velocity": 2.0, "effort": 40.0} for c, s, g in zip(CAN, SRC, signs)}


class _Probe:
    """토픽별 수신 메시지를 모으는 구독 노드(백그라운드 spin)."""

    def __init__(self, context, name, subs):
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node

        self.node = Node(name, context=context)
        self.got = {topic: [] for topic, _ in subs}
        for topic, mtype in subs:
            self.node.create_subscription(mtype, topic, self._cb(topic), 10)
        self.exec = SingleThreadedExecutor(context=context)
        self.exec.add_node(self.node)
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _cb(self, topic):
        def cb(msg):
            self.got[topic].append(msg)
        return cb

    def _run(self):
        while not self._stop.is_set():
            self.exec.spin_once(timeout_sec=0.02)

    def wait(self, topic, n, timeout=2.0):
        t0 = time.time()
        while len(self.got[topic]) < n and time.time() - t0 < timeout:
            time.sleep(0.01)
        return list(self.got[topic])

    def close(self):
        self._stop.set()
        self._th.join(timeout=2.0)
        self.exec.shutdown()
        self.node.destroy_node()


def _settle(node, probe_node, topic, timeout=2.0):
    """발행자-구독자 발견 대기(테스트 도메인 discovery)."""
    t0 = time.time()
    while node.count_subscribers(topic) < 1 and time.time() - t0 < timeout:
        time.sleep(0.02)


@pytest.fixture
def arm_probe(ros):
    from std_msgs.msg import Float64MultiArray

    p = _Probe(ros, "probe_arm", [(t, Float64MultiArray) for t in FWD_TOPICS.values()])
    try:
        yield p
    finally:
        p.close()


@pytest.fixture
def pub_node(ros):
    from rclpy.node import Node

    n = Node("pd_backend_test", context=ros)
    try:
        yield n
    finally:
        n.destroy_node()


def _cmd(q, qd, tau):
    from policy_control.pd_law import PdCommand

    return PdCommand(q=np.asarray(q, float), qd=np.asarray(qd, float), tau=np.asarray(tau, float),
                     limited=(), effort_fault=False)


# ------------------------------------------------------------------ arm forward
def test_arm_forward_publishes_source_order_and_sign(ros, arm_probe, pub_node):
    from jtc_bridge_core import JointRemap
    from policy_control.pd_backends import ArmForwardBackend

    signs = [1, -1, 1, -1, 1, 1, -1]
    scrambled = CAN[::-1]                                   # 입력은 canonical 역순
    remap = JointRemap(scrambled, SRC, _synthetic_profile(signs))
    be = ArmForwardBackend(pub_node, SIDE, remap, execute=True)
    for t in FWD_TOPICS.values():
        _settle(pub_node, arm_probe.node, t)
    q_can = np.arange(7, dtype=float)[::-1] * 0.1          # canonical 역순 값: l_aj_1 → 0.0, l_aj_7 → 0.6
    qd_can = np.full(7, 0.5)
    tau_can = np.full(7, 2.0)
    written = be.write(_cmd(q_can, qd_can, tau_can))

    exp_q = np.arange(7, dtype=float) * 0.1 * np.array(signs)
    exp_qd = 0.5 * np.array(signs)
    exp_tau = 2.0 * np.array(signs)
    np.testing.assert_allclose(written.q, exp_q)
    np.testing.assert_allclose(written.qd, exp_qd)
    np.testing.assert_allclose(written.tau, exp_tau)
    pos = arm_probe.wait(FWD_TOPICS["position"], 1)
    vel = arm_probe.wait(FWD_TOPICS["velocity"], 1)
    eff = arm_probe.wait(FWD_TOPICS["effort"], 1)
    np.testing.assert_allclose(pos[-1].data, exp_q)
    np.testing.assert_allclose(vel[-1].data, exp_qd)
    np.testing.assert_allclose(eff[-1].data, exp_tau)
    assert be.publish_count == 3


def test_arm_forward_position_clamped_by_profile_limits(ros, pub_node):
    from jtc_bridge_core import JointRemap, load_profile_joints
    from policy_control.pd_backends import ArmForwardBackend

    prof = load_profile_joints(_profile_path())
    remap = JointRemap(CAN, SRC, prof)
    be = ArmForwardBackend(pub_node, SIDE, remap, execute=True)
    written = be.write(_cmd(np.full(7, 9.0), np.zeros(7), np.zeros(7)))
    upper = np.array([prof[c]["upper"] for c in CAN])
    np.testing.assert_allclose(written.q, upper)


def test_arm_forward_zero_release_only_velocity_and_effort(ros, arm_probe, pub_node):
    from jtc_bridge_core import JointRemap
    from policy_control.pd_backends import ArmForwardBackend

    remap = JointRemap(CAN, SRC, _synthetic_profile([1] * 7))
    be = ArmForwardBackend(pub_node, SIDE, remap, execute=True)
    for t in FWD_TOPICS.values():
        _settle(pub_node, arm_probe.node, t)
    be.zero_release()
    vel = arm_probe.wait(FWD_TOPICS["velocity"], 1)
    eff = arm_probe.wait(FWD_TOPICS["effort"], 1)
    assert list(vel[-1].data) == [0.0] * 7
    assert list(eff[-1].data) == [0.0] * 7
    time.sleep(0.1)
    assert arm_probe.got[FWD_TOPICS["position"]] == []
    assert be.publish_count == 2


def test_arm_forward_rejects_bad_length(ros, pub_node):
    from jtc_bridge_core import JointRemap
    from policy_control.pd_backends import ArmForwardBackend, BackendError

    be = ArmForwardBackend(pub_node, SIDE, JointRemap(CAN, SRC, _synthetic_profile([1] * 7)), execute=True)
    with pytest.raises(BackendError):
        be.write(_cmd(np.zeros(6), np.zeros(7), np.zeros(7)))


def test_arm_forward_dry_run_publishes_nothing(ros, arm_probe, pub_node):
    from jtc_bridge_core import JointRemap
    from policy_control.pd_backends import ArmForwardBackend

    be = ArmForwardBackend(pub_node, SIDE, JointRemap(CAN, SRC, _synthetic_profile([1] * 7)), execute=False)
    written = be.write(_cmd(np.zeros(7), np.ones(7), np.ones(7)))
    be.zero_release()
    assert written.q.shape == (7,)                        # 법칙 결과는 돌려준다(status/applied 용)
    time.sleep(0.3)
    assert all(arm_probe.got[t] == [] for t in FWD_TOPICS.values())
    assert be.publish_count == 0
    assert all(pub_node.count_publishers(t) == 0 for t in FWD_TOPICS.values())


# ------------------------------------------------------------------ gripper
@pytest.fixture
def grip_probe(ros):
    from trajectory_msgs.msg import JointTrajectory

    p = _Probe(ros, "probe_grip", [(GRIP_TOPIC, JointTrajectory)])
    try:
        yield p
    finally:
        p.close()


def _grip(pub_node, **kw):
    from policy_control.pd_backends import GripperJtcBackend

    kw.setdefault("execute", True)
    kw.setdefault("max_vel", 0.2)
    return GripperJtcBackend(pub_node, GRIP_TOPIC, GRIP_JOINT, close_overtravel_m=0.008,
                             lower=0.0, upper=0.044, **kw)


def test_gripper_single_point_tfs_zero(ros, grip_probe, pub_node):
    from policy_control.pd_backends import GripperCmd

    be = _grip(pub_node)
    _settle(pub_node, grip_probe.node, GRIP_TOPIC)
    w = be.write(GripperCmd(q_star=0.040, q_meas=0.030, dt=0.01))
    msg = grip_probe.wait(GRIP_TOPIC, 1)[-1]
    assert msg.joint_names == [GRIP_JOINT]
    assert len(msg.points) == 1
    pt = msg.points[0]
    assert (pt.time_from_start.sec, pt.time_from_start.nanosec) == (0, 0)
    assert list(pt.velocities) == [] and list(pt.effort) == []
    assert pt.positions[0] == pytest.approx(w.q_cmd)


def test_gripper_overtravel_guard_when_closing(ros, pub_node):
    from policy_control.pd_backends import GripperCmd

    be = _grip(pub_node, max_vel=99.0)
    w = be.write(GripperCmd(q_star=0.0, q_meas=0.030, dt=0.01))
    assert w.q_cmd == pytest.approx(0.030 - 0.008)
    assert w.overtravel_guarded is True
    w = be.write(GripperCmd(q_star=0.025, q_meas=0.030, dt=0.01))   # 여유 안: 그대로
    assert w.q_cmd == pytest.approx(0.025) and w.overtravel_guarded is False
    w = be.write(GripperCmd(q_star=0.044, q_meas=0.030, dt=0.01))   # 열기: 가드 없음
    assert w.q_cmd == pytest.approx(0.044) and w.overtravel_guarded is False


def test_gripper_velocity_limit_from_previous_command(ros, pub_node):
    from policy_control.pd_backends import GripperCmd

    be = _grip(pub_node)                                   # max_vel 0.2 m/s · dt 0.01 → 0.002/틱
    w = be.write(GripperCmd(q_star=0.044, q_meas=0.0, dt=0.01))
    assert w.q_cmd == pytest.approx(0.002) and w.limited
    w = be.write(GripperCmd(q_star=0.044, q_meas=0.0, dt=0.01))
    assert w.q_cmd == pytest.approx(0.004)


def test_gripper_clamps_to_limits_and_rejects_bad_input(ros, pub_node):
    from policy_control.pd_backends import BackendError, GripperCmd

    be = _grip(pub_node, max_vel=99.0)
    assert be.write(GripperCmd(q_star=0.5, q_meas=0.044, dt=0.01)).q_cmd == pytest.approx(0.044)
    assert be.write(GripperCmd(q_star=-0.5, q_meas=0.0, dt=0.01)).q_cmd == pytest.approx(0.0)
    with pytest.raises(BackendError):
        be.write(GripperCmd(q_star=float("nan"), q_meas=0.0, dt=0.01))
    with pytest.raises(BackendError):
        be.write(GripperCmd(q_star=0.0, q_meas=0.0, dt=0.0))


def test_gripper_dry_run_publishes_nothing(ros, grip_probe, pub_node):
    from policy_control.pd_backends import GripperCmd

    be = _grip(pub_node, execute=False)
    be.write(GripperCmd(q_star=0.02, q_meas=0.02, dt=0.01))
    be.zero_release()
    time.sleep(0.3)
    assert grip_probe.got[GRIP_TOPIC] == [] and be.publish_count == 0
    assert pub_node.count_publishers(GRIP_TOPIC) == 0


# ------------------------------------------------------------------ dg5f hand
@pytest.fixture
def hand_probe(ros):
    from trajectory_msgs.msg import JointTrajectory

    p = _Probe(ros, "probe_hand", [(HAND_TOPIC, JointTrajectory)])
    try:
        yield p
    finally:
        p.close()


def _hand_remap(order, real_profile=False):
    from jtc_bridge_core import JointRemap, load_profile_joints

    if real_profile:
        return JointRemap(list(order), HAND_SRC, load_profile_joints(_profile_path()))
    wide = {c: {"source": s, "sign": 1.0, "lower": -10.0, "upper": 10.0, "unit": "rad",
                "velocity": 2.0, "effort": 7.5} for c, s in zip(HAND_CAN, HAND_SRC)}
    return JointRemap(list(order), HAND_SRC, wide)


def test_dg5f_single_point_positions_only_finger_major(ros, hand_probe, pub_node):
    from policy_control.pd_backends import Dg5fJtcBackend, HandCmd

    scrambled = HAND_CAN[::-1]
    be = Dg5fJtcBackend(pub_node, HAND_TOPIC, _hand_remap(scrambled), max_vel=99.0, execute=True)
    _settle(pub_node, hand_probe.node, HAND_TOPIC)
    q_can = np.linspace(0.0, 0.19, 20)[::-1]               # canonical 역순 → 값 r_hj_thumb_1=0.0 … pinky_4=0.19
    w = be.write(HandCmd(q_star=q_can, qd_star=np.ones(20), dt=1 / 60))
    msg = hand_probe.wait(HAND_TOPIC, 1)[-1]
    assert msg.joint_names == HAND_SRC
    assert len(msg.points) == 1
    pt = msg.points[0]
    assert (pt.time_from_start.sec, pt.time_from_start.nanosec) == (0, 0)
    assert list(pt.velocities) == []                       # q̇* 는 버린다
    np.testing.assert_allclose(pt.positions, np.linspace(0.0, 0.19, 20), atol=1e-12)
    np.testing.assert_allclose(w.q_cmd, np.linspace(0.0, 0.19, 20), atol=1e-12)


def test_dg5f_real_profile_clamps_thumb_2_to_zero(ros, pub_node):
    from policy_control.pd_backends import Dg5fJtcBackend, HandCmd

    be = Dg5fJtcBackend(pub_node, HAND_TOPIC, _hand_remap(HAND_CAN, real_profile=True), max_vel=99.0, execute=True)
    w = be.write(HandCmd(q_star=np.full(20, 0.01), qd_star=None, dt=1 / 60))
    assert w.names[1] == "rj_dg_1_2" and w.q_cmd[1] == pytest.approx(0.0)   # r_hj_thumb_2 upper 0.0
    assert w.q_cmd[0] == pytest.approx(0.01)


def test_dg5f_velocity_limit_and_bad_length(ros, pub_node):
    from policy_control.pd_backends import BackendError, Dg5fJtcBackend, HandCmd

    be = Dg5fJtcBackend(pub_node, HAND_TOPIC, _hand_remap(HAND_CAN), max_vel=1.0, execute=True)
    w = be.write(HandCmd(q_star=np.full(20, 0.5), qd_star=None, dt=0.01))
    np.testing.assert_allclose(w.q_cmd, 0.5)                # 첫 지령(실측 없음) = 무제한 시드 — 0 에서 램프하지 않는다
    assert not w.limited
    w = be.write(HandCmd(q_star=np.full(20, 0.6), qd_star=None, dt=0.01))
    np.testing.assert_allclose(w.q_cmd, 0.51)               # 직전 지령 기준 max_vel·dt
    assert w.limited
    be.zero_release()
    w = be.write(HandCmd(q_star=np.full(20, 0.5), qd_star=None, dt=0.01, q_meas=np.full(20, 0.2)))
    np.testing.assert_allclose(w.q_cmd, 0.21)               # 실측 시드 → 실측에서 램프
    with pytest.raises(BackendError):
        be.write(HandCmd(q_star=np.zeros(19), qd_star=None, dt=0.01))


def test_dg5f_dry_run_publishes_nothing(ros, hand_probe, pub_node):
    from policy_control.pd_backends import Dg5fJtcBackend, HandCmd

    be = Dg5fJtcBackend(pub_node, HAND_TOPIC, _hand_remap(HAND_CAN), max_vel=1.0, execute=False)
    be.write(HandCmd(q_star=np.zeros(20), qd_star=None, dt=0.01))
    be.zero_release()
    time.sleep(0.3)
    assert hand_probe.got[HAND_TOPIC] == [] and be.publish_count == 0
    assert pub_node.count_publishers(HAND_TOPIC) == 0


# ------------------------------------------------------------------ hand PID gains
@pytest.fixture
def hand_ctrl(ros):
    """/dg5f_right/dg5f_right_controller 흉내 — gains.<joint>.{p,d} 파라미터를 가진 노드."""
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    node = Node("dg5f_right_controller", namespace="/dg5f_right", context=ros)
    for j in HAND_SRC:
        node.declare_parameter(f"gains.{j}.p", 1.5)
        node.declare_parameter(f"gains.{j}.d", 0.0)
    ex = SingleThreadedExecutor(context=ros)
    ex.add_node(node)
    stop = threading.Event()

    def run():
        while not stop.is_set():
            ex.spin_once(timeout_sec=0.02)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    try:
        yield node
    finally:
        stop.set()
        th.join(timeout=2.0)
        ex.shutdown()
        node.destroy_node()


def _gains(node, key):
    return [node.get_parameter(f"gains.{j}.{key}").value for j in HAND_SRC]


def test_hand_gains_check_and_apply(ros, hand_ctrl, pub_node):
    from policy_control.pd_backends import HandGainsClient

    cl = HandGainsClient(pub_node, HAND_CTRL, HAND_SRC, timeout_sec=3.0, execute=True)
    ok, reasons = cl.check(4.5, 0.0)
    assert ok is False and any("mismatch" in r for r in reasons)
    ok, reasons = cl.check_and_apply(4.5, 0.0)
    assert ok, reasons
    assert _gains(hand_ctrl, "p") == [4.5] * 20 and _gains(hand_ctrl, "d") == [0.0] * 20
    ok, reasons = cl.check_and_apply(4.5, 0.0)             # 이미 일치 → set 없이 ok
    assert ok and not any("applied" in r for r in reasons)
    cl.close()


def test_hand_gains_dry_run_never_sets(ros, hand_ctrl, pub_node):
    from policy_control.pd_backends import HandGainsClient

    cl = HandGainsClient(pub_node, HAND_CTRL, HAND_SRC, timeout_sec=3.0, execute=False)
    ok, reasons = cl.check_and_apply(4.5, 0.0)
    assert ok is False and any("dry_run" in r for r in reasons)
    assert _gains(hand_ctrl, "p") == [1.5] * 20
    cl.close()


def test_hand_gains_missing_controller_returns_reasons(ros, pub_node):
    from policy_control.pd_backends import HandGainsClient

    cl = HandGainsClient(pub_node, "/nowhere/ctrl", HAND_SRC, timeout_sec=0.2, execute=True)
    ok, reasons = cl.check_and_apply(4.5, 0.0)
    assert ok is False and any("unavailable" in r for r in reasons)
    cl.close()


# ------------------------------------------------------------------ 09.06 손 namespace → 파라미터 노드 이름
def test_hand_controller_name_from_topic_and_namespace():
    from policy_control.pd_backends import hand_controller_name

    assert hand_controller_name("/dg5f_left/dg5f_left_controller/joint_trajectory", "dg5f_left") == "/dg5f_left/dg5f_left_controller"
    assert hand_controller_name("/dg5f_right/dg5f_right_controller/joint_trajectory", "/dg5f_right/") == "/dg5f_right/dg5f_right_controller"
    assert hand_controller_name("/dg5f_right/dg5f_right_controller/joint_trajectory", None) == "/dg5f_right/dg5f_right_controller"
    with pytest.raises(ValueError):
        hand_controller_name("/dg5f_right/dg5f_right_controller/joint_trajectory", "dg5f_left")   # 다른 손의 namespace
    with pytest.raises(ValueError):
        hand_controller_name("/joint_trajectory", None)
