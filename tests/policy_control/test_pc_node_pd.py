"""pd_node — rclpy 껍질 배선 테스트 (ros 마커, 격리 도메인, fake controller_manager + fake 플랜트).

좌 계약(left_v2B25) + left_gripper_fake.yaml + config/pd_left.yaml. execute=false 는 서비스·토픽 0건을,
execute=true 는 fake CM 스텁(STRICT)·forward 토픽 프로브·watchdog/estop/release 경로를 잠근다.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.ros

SIM2REAL = Path(__file__).resolve().parents[2]
LEFT_JSON = SIM2REAL / "logs/policy/left_v2B25/deploy_contract.json"
LEFT_ROBOT = SIM2REAL / "policy_control/config/robots/left_gripper_fake.yaml"
PD_YAML = SIM2REAL / "policy_control/config/pd_left.yaml"
needs_left = pytest.mark.skipif(not LEFT_JSON.exists(), reason="left_v2B25 contract 없음")

NS = "/policy_control"
SIDE = "left"
JTC = f"{SIDE}_joint_trajectory_controller"
FWD = {k: f"{SIDE}_forward_{k}_controller" for k in ("position", "velocity", "effort")}
ARM_SRC = [f"openarm_{SIDE}_joint{i}" for i in range(1, 8)]
ARM_CAN = [f"l_aj_{i}" for i in range(1, 8)]
GRIP_SRC = "openarm_left_finger_joint1"
PLANT_HZ = 100.0


class _Spinner:
    def __init__(self, context, *nodes, threads: int = 1):
        from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor

        self.exec = (MultiThreadedExecutor(num_threads=threads, context=context) if threads > 1
                     else SingleThreadedExecutor(context=context))
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
        self._th.join(timeout=3.0)
        self.exec.shutdown()


class Plant:
    """fake 플랜트: /joint_states 를 PLANT_HZ 로 내고 forward position 지령을 그대로 따른다."""

    def __init__(self, context, q0: np.ndarray):
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
        from std_msgs.msg import String

        self.node = Node("fake_plant", context=context)
        self.q = np.asarray(q0, dtype=float).copy()
        self.grip = 0.044
        self.running = True
        self.fwd = {k: [] for k in FWD}
        self.applied = []
        self.status = []
        self.pub = self.node.create_publisher(JointState, "/joint_states", 10)
        for k, name in FWD.items():
            self.node.create_subscription(Float64MultiArray, f"/{name}/commands", self._cb(k), 10)
        self.node.create_subscription(JointState, f"{NS}/pd/applied", lambda m: self.applied.append(m), 10)
        self.node.create_subscription(String, f"{NS}/status/pd", lambda m: self.status.append(m), 10)
        self.node.create_timer(1.0 / PLANT_HZ, self._tick)

    def _cb(self, kind):
        def cb(msg):
            self.fwd[kind].append(np.asarray(msg.data, dtype=float))
            if kind == "position" and len(msg.data) == 7:
                self.q = np.asarray(msg.data, dtype=float)      # 완벽 추종(부호 +1)
        return cb

    def _tick(self):
        if not self.running:
            return
        from policy_control import codec

        msg = codec.encode_joint_state(ARM_SRC + [GRIP_SRC], np.concatenate([self.q, [self.grip]]),
                                       velocity=np.zeros(8), effort=np.zeros(8), stamp=time.time())
        self.pub.publish(msg)

    def last_status(self) -> dict | None:
        return json.loads(self.status[-1].data) if self.status else None

    def wait_phase(self, phases, timeout: float = 3.0) -> dict | None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            st = self.last_status()
            if st is not None and st["phase"] in phases:
                return st
            time.sleep(0.01)
        return self.last_status()

    def close(self):
        self.node.destroy_node()


class Caller:
    """Trigger 호출 + joint_target 발행 + estop/episode 발행 (자기 executor)."""

    def __init__(self, context):
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, String

        from policy_control.controller_switch import ServiceCaller

        self.node = Node("pd_caller", context=context)
        self.sc = ServiceCaller(self.node, "srv", timeout_sec=10.0)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_target = self.node.create_publisher(JointState, f"{NS}/joint_target", 10)
        self.pub_estop = self.node.create_publisher(Bool, f"{NS}/estop", latched)
        self.pub_episode = self.node.create_publisher(String, f"{NS}/episode", latched)

    def trigger(self, name: str):
        from std_srvs.srv import Trigger

        resp, reason = self.sc.call(Trigger, f"{NS}/pd/{name}", Trigger.Request())
        assert resp is not None, reason
        body = json.loads(resp.message)
        assert body["ok"] is resp.success
        return resp.success, body["reasons"]

    def target(self, q7, seq: int, episode: str = "1", grip: float = 0.044):
        from policy_control import codec

        self.pub_target.publish(codec.encode_joint_target(ARM_CAN + ["l_hj_gripper_1"], list(q7) + [grip],
                                                          np.zeros(8), episode, seq, stamp=time.time()))

    def estop(self, value: bool):
        from std_msgs.msg import Bool

        self.pub_estop.publish(Bool(data=value))

    def episode(self, event: str, episode: int):
        from std_msgs.msg import String

        self.pub_episode.publish(String(data=json.dumps({"episode": episode, "event": event, "object_anchor": None,
                                                          "home_q": {}, "reasons": [], "t_ns": time.time_ns()})))

    def close(self):
        self.sc.close()
        self.node.destroy_node()


@pytest.fixture(scope="module")
def left():
    from policy_control import contract as C

    return C.load_contract(LEFT_JSON)


@pytest.fixture
def pd_yaml_execute(tmp_path) -> Path:
    raw = yaml.safe_load(PD_YAML.read_text())
    raw["execute"] = True
    p = tmp_path / "pd_left_execute.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def _make_node(context, execute: bool, pd_yaml: Path):
    from rclpy.parameter import Parameter

    from policy_control.pd_node import PdNode

    overrides = [Parameter("contract", value=str(LEFT_JSON)), Parameter("robot", value=str(LEFT_ROBOT)),
                 Parameter("pd_config", value=str(pd_yaml)), Parameter("execute", value=bool(execute))]
    return PdNode(context=context, parameter_overrides=overrides)


def _wait_status(plant: Plant, timeout: float = 3.0) -> dict:
    t0 = time.monotonic()
    while not plant.status and time.monotonic() - t0 < timeout:
        time.sleep(0.01)
    assert plant.status, "pd status 미수신"
    return plant.last_status()


@pytest.fixture
def cm(ros):
    from rclpy.node import Node

    from fake_arm_bridge import ControllerManagerStub

    server = Node("fake_cm", context=ros)
    stub = ControllerManagerStub(server, SIDE)
    calls = {"n": 0}
    for name in ("_list", "_load", "_configure", "_switch"):
        orig = getattr(stub, name)

        def wrapped(req, res, _orig=orig):
            calls["n"] += 1
            return _orig(req, res)

        setattr(stub, name, wrapped)
    spin = _Spinner(ros, server)
    try:
        yield stub, calls
    finally:
        spin.close()
        server.destroy_node()


@pytest.fixture
def rig_dry(ros, left):
    """execute=false 노드 + 플랜트(홈 근처) + 호출자."""
    node = _make_node(ros, False, PD_YAML)
    plant = Plant(ros, np.asarray(left.pd.home_arm) + 0.02)
    caller = Caller(ros)
    spin = _Spinner(ros, node, plant.node, threads=3)
    time.sleep(0.5)
    try:
        yield node, plant, caller
    finally:
        spin.close()
        caller.close()
        plant.close()
        node.destroy_node()


@pytest.fixture
def rig(ros, left, cm, pd_yaml_execute):
    """execute=true 노드 + fake CM + 플랜트(홈 근처) + 호출자."""
    node = _make_node(ros, True, pd_yaml_execute)
    plant = Plant(ros, np.asarray(left.pd.home_arm) + 0.02)
    caller = Caller(ros)
    spin = _Spinner(ros, node, plant.node, threads=3)
    time.sleep(0.5)
    try:
        yield node, plant, caller, cm[0]
    finally:
        spin.close()
        caller.close()
        plant.close()
        node.destroy_node()


# ------------------------------------------------------------------ dry run
@needs_left
def test_dry_run_engage_refused_and_nothing_published(rig_dry, cm):
    node, plant, caller = rig_dry
    stub, calls = cm
    st = _wait_status(plant)
    assert st["node"] == "pd" and st["phase"] == "IDLE" and st["execute"] is False
    assert st["gains"]["ok"] is True
    ok, reasons = caller.trigger("engage")
    assert ok is False and any("execute" in r for r in reasons)
    for name in FWD.values():
        assert plant.node.count_publishers(f"/{name}/commands") == 0
    assert plant.node.count_publishers("/left_gripper_controller/joint_trajectory") == 0
    time.sleep(0.2)
    assert calls["n"] == 0 and stub.known[JTC] == "active"
    assert plant.last_status()["phase"] == "IDLE"


@needs_left
def test_engage_refused_when_joint_state_stale(ros, left, cm, pd_yaml_execute):
    node = _make_node(ros, True, pd_yaml_execute)
    plant = Plant(ros, np.asarray(left.pd.home_arm))
    plant.running = False                               # /joint_states 없음
    caller = Caller(ros)
    spin = _Spinner(ros, node, plant.node, threads=3)
    try:
        time.sleep(0.4)
        ok, reasons = caller.trigger("engage")
        assert ok is False and any("stale" in r for r in reasons)
        assert cm[0].known[JTC] == "active"
    finally:
        spin.close()
        caller.close()
        plant.close()
        node.destroy_node()


# ------------------------------------------------------------------ execute
@needs_left
def test_engage_stream_watchdog_release(rig, left):
    node, plant, caller, stub = rig
    ok, reasons = caller.trigger("engage")
    assert ok, reasons
    st = plant.wait_phase(("RAMPING", "TRACKING"))
    assert st["phase"] in ("RAMPING", "TRACKING") and st["execute"] is True
    assert all(stub.known[n] == "active" for n in FWD.values()) and stub.known[JTC] == "inactive"
    # forward 3 토픽이 source 순 7값으로 나온다; applied 는 canonical 이름
    t0 = time.monotonic()
    while len(plant.fwd["position"]) < 5 and time.monotonic() - t0 < 2.0:
        time.sleep(0.01)
    assert len(plant.fwd["position"]) >= 5 and len(plant.fwd["velocity"]) >= 5 and len(plant.fwd["effort"]) >= 5
    assert plant.fwd["position"][-1].shape == (7,)
    assert tuple(plant.applied[-1].name)[:7] == tuple(ARM_CAN)
    np.testing.assert_allclose(plant.fwd["position"][-1], plant.applied[-1].position[:7], atol=1e-9)
    # joint_target 스트림(50 Hz) → TRACKING, 세트포인트가 목표로 움직인다
    q_goal = np.asarray(left.pd.home_arm) + 0.03
    for seq in range(30):
        caller.target(q_goal, seq)
        time.sleep(0.02)
    st = plant.wait_phase(("TRACKING",))
    assert st["phase"] == "TRACKING" and st["seq"] == 29 and st["ok"] is True
    assert np.abs(plant.fwd["position"][-1] - q_goal).max() < 0.02
    # 스트림 두절 → watchdog HOLD(세트포인트 동결·q̇ 0)
    st = plant.wait_phase(("HOLD",), timeout=2.0)
    assert st["phase"] == "HOLD" and any("watchdog" in r for r in st["reasons"]) and st["ok"] is False
    frozen = plant.fwd["position"][-1].copy()
    time.sleep(0.1)
    np.testing.assert_allclose(plant.fwd["position"][-1], frozen)
    np.testing.assert_allclose(plant.fwd["velocity"][-1], np.zeros(7))
    # release → 0 송출 → JTC 복귀 → IDLE
    n_vel = len(plant.fwd["velocity"])
    ok, reasons = caller.trigger("release")
    assert ok, reasons
    st = plant.wait_phase(("IDLE",))
    assert st["phase"] == "IDLE"
    assert stub.known[JTC] == "active" and all(stub.known[n] == "inactive" for n in FWD.values())
    tail = plant.fwd["velocity"][n_vel:]
    assert len(tail) >= 5 and all(np.all(v == 0.0) for v in tail[-5:])
    assert all(np.all(v == 0.0) for v in plant.fwd["effort"][-5:])


@needs_left
def test_estop_latch_holds_and_release_recovers(rig):
    node, plant, caller, stub = rig
    assert caller.trigger("engage")[0]
    plant.wait_phase(("RAMPING", "TRACKING"))
    caller.estop(True)
    st = plant.wait_phase(("HOLD",))
    assert st["phase"] == "HOLD" and any("estop" in r for r in st["reasons"]) and st["estop"] is True
    ok, reasons = caller.trigger("engage")
    assert ok is False and any("IDLE" in r for r in reasons)
    assert caller.trigger("release")[0]
    assert plant.wait_phase(("IDLE",))["phase"] == "IDLE"
    ok, reasons = caller.trigger("engage")             # estop 래치가 남아 있으면 engage 거부
    assert ok is False and any("estop" in r for r in reasons)


@needs_left
def test_goto_home_ramps_settles_and_tracks(rig, left):
    node, plant, caller, stub = rig
    assert caller.trigger("engage")[0]
    plant.wait_phase(("RAMPING", "TRACKING"))
    ok, reasons = caller.trigger("goto_home")
    assert ok, reasons
    st = plant.last_status()
    assert st["phase"] == "TRACKING"
    home = np.asarray(left.pd.home_arm)
    assert np.abs(plant.q - home).max() < 0.011
    # 에피소드 reset 이벤트 → droop 0 · 홈 유지, 스트림 없이도 HOLD 로 떨어지지 않는다
    caller.episode("reset", 1)
    time.sleep(0.4)
    st = plant.last_status()
    assert st["phase"] == "TRACKING" and st["episode"] == 1
    assert caller.trigger("release")[0]


@needs_left
def test_goto_home_refused_when_idle(rig_dry):
    node, plant, caller = rig_dry
    ok, reasons = caller.trigger("goto_home")
    assert ok is False and any("IDLE" in r for r in reasons)
    ok, reasons = caller.trigger("release")            # IDLE 에서 release 는 무해(ok)
    assert ok is True


# ================================================================== 09.06 양팔 DG-5F-M — asset 계약 + dg5f_m_bi_fake.yaml + pd_dg5f_m*.yaml
ASSET_JSON = SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json"
BI_ROBOT = SIM2REAL / "policy_control/config/robots/dg5f_m_bi_fake.yaml"
PD_BI = SIM2REAL / "policy_control/config/pd_dg5f_m.yaml"
PD_BI_FAKE = SIM2REAL / "policy_control/config/pd_dg5f_m_fake.yaml"
needs_asset = pytest.mark.skipif(not ASSET_JSON.exists(), reason="asset contract 없음")
BI_SIDES = ("right", "left")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _arm_src(side):
    return [f"openarm_{side}_joint{i}" for i in range(1, 8)]


def _arm_can(side):
    return [f"{side[0]}_aj_{i}" for i in range(1, 8)]


def _hand_src(side):
    return [f"{side[0]}j_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]


def _fwd_of(side):
    return {k: f"{side}_forward_{k}_controller" for k in ("position", "velocity", "effort")}


def _jtc_of(side):
    return f"{side}_joint_trajectory_controller"


class BiPlant:
    """양팔 fake 플랜트: /joint_states(양팔) + /dg5f_<side>/joint_states(손 20, 정적) 를 내고 forward position 을 그대로 따른다."""

    def __init__(self, context, q0: dict):
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray, String

        self.node = Node("fake_plant_bi", context=context)
        self.q = {s: np.asarray(q0[s], dtype=float).copy() for s in BI_SIDES}
        self.fwd = {s: {k: [] for k in _fwd_of(s)} for s in BI_SIDES}
        self.applied, self.status = [], []
        self.pub = self.node.create_publisher(JointState, "/joint_states", 10)
        self.hand_pubs = {s: self.node.create_publisher(JointState, f"/dg5f_{s}/joint_states", 10) for s in BI_SIDES}
        for s in BI_SIDES:
            for k, name in _fwd_of(s).items():
                self.node.create_subscription(Float64MultiArray, f"/{name}/commands", self._cb(s, k), 10)
        self.node.create_subscription(JointState, f"{NS}/pd/applied", lambda m: self.applied.append(m), 10)
        self.node.create_subscription(String, f"{NS}/status/pd", lambda m: self.status.append(m), 10)
        self.node.create_timer(1.0 / PLANT_HZ, self._tick)

    def _cb(self, side, kind):
        def cb(msg):
            self.fwd[side][kind].append(np.asarray(msg.data, dtype=float))
            if kind == "position" and len(msg.data) == 7:
                self.q[side] = np.asarray(msg.data, dtype=float)      # 완벽 추종(부호 +1)
        return cb

    def _tick(self):
        from policy_control import codec

        names = _arm_src("right") + _arm_src("left")
        q = np.concatenate([self.q["right"], self.q["left"]])
        self.pub.publish(codec.encode_joint_state(names, q, velocity=np.zeros(14), effort=np.zeros(14), stamp=time.time()))
        for s in BI_SIDES:
            self.hand_pubs[s].publish(codec.encode_joint_state(_hand_src(s), np.zeros(20), velocity=np.zeros(20),
                                                               stamp=time.time()))

    def last_status(self) -> dict | None:
        return json.loads(self.status[-1].data) if self.status else None

    def wait(self, pred, timeout: float = 3.0) -> dict | None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            st = self.last_status()
            if st is not None and pred(st):
                return st
            time.sleep(0.01)
        return self.last_status()

    def wait_phase(self, phases, timeout: float = 3.0) -> dict | None:
        return self.wait(lambda st: st["phase"] in phases, timeout)

    def close(self):
        self.node.destroy_node()


class BiCaller(Caller):
    def target_named(self, names, q, seq: int, episode: str = "1"):
        from policy_control import codec

        self.pub_target.publish(codec.encode_joint_target(list(names), list(q), np.zeros(len(names)), episode, seq,
                                                          stamp=time.time()))


@pytest.fixture
def bi_cm(ros):
    from rclpy.node import Node

    from fake_cm_stub import ControllerManagerStub

    server = Node("fake_cm_bi", context=ros)
    stub = ControllerManagerStub(server, BI_SIDES)
    calls = {"n": 0}
    for name in ("_list", "_load", "_configure", "_switch"):
        orig = getattr(stub, name)

        def wrapped(req, res, _orig=orig):
            calls["n"] += 1
            return _orig(req, res)

        setattr(stub, name, wrapped)
    spin = _Spinner(ros, server)
    try:
        yield stub, calls
    finally:
        spin.close()
        server.destroy_node()


@pytest.fixture
def bi_hand_ctrls(ros):
    """/dg5f_<side>/dg5f_<side>_controller 흉내 — gains.<joint>.{p,d} 파라미터(드라이버 기본 1.5/0)."""
    from rclpy.node import Node

    nodes = {}
    for s in BI_SIDES:
        n = Node(f"dg5f_{s}_controller", namespace=f"/dg5f_{s}", context=ros)
        for j in _hand_src(s):
            n.declare_parameter(f"gains.{j}.p", 1.5)
            n.declare_parameter(f"gains.{j}.d", 0.0)
        nodes[s] = n
    spin = _Spinner(ros, *nodes.values())
    try:
        yield nodes
    finally:
        spin.close()
        for n in nodes.values():
            n.destroy_node()


def _hand_p(node, side):
    return [node.get_parameter(f"gains.{j}.p").value for j in _hand_src(side)]


def _make_bi_node(context, execute: bool, pd_yaml: Path, sides: str = ""):
    from rclpy.parameter import Parameter

    from policy_control.pd_node import PdNode

    overrides = [Parameter("contract", value=str(ASSET_JSON)), Parameter("robot", value=str(BI_ROBOT)),
                 Parameter("pd_config", value=str(pd_yaml)), Parameter("execute", value=bool(execute)),
                 Parameter("sides", value=sides)]
    return PdNode(context=context, parameter_overrides=overrides)


def _bi_rig(ros, execute: bool, pd_yaml: Path, sides: str = ""):
    node = _make_bi_node(ros, execute, pd_yaml, sides)
    plant = BiPlant(ros, {s: np.full(7, 0.02) for s in BI_SIDES})          # 홈(0) 근처
    caller = BiCaller(ros)
    spin = _Spinner(ros, node, plant.node, threads=3)
    time.sleep(0.6)
    return node, plant, caller, spin


def _close_rig(node, plant, caller, spin):
    spin.close()
    caller.close()
    plant.close()
    node.close()
    node.destroy_node()


@needs_asset
def test_bi_dry_run_publishes_nothing_on_either_side(ros, bi_cm, bi_hand_ctrls):
    stub, calls = bi_cm
    node, plant, caller, spin = _bi_rig(ros, False, PD_BI)
    try:
        st = _wait_status(plant)
        assert st["sides"] == list(BI_SIDES) and st["phase"] == "IDLE" and st["execute"] is False
        assert set(st["arms"]) == set(BI_SIDES) and all(st["arms"][s]["gains"]["ok"] for s in BI_SIDES)
        ok, reasons = caller.trigger("engage")
        assert ok is False and any("execute" in r for r in reasons)
        for s in BI_SIDES:
            for name in _fwd_of(s).values():
                assert plant.node.count_publishers(f"/{name}/commands") == 0
            assert plant.node.count_publishers(f"/dg5f_{s}/dg5f_{s}_controller/joint_trajectory") == 0
            assert stub.known[_jtc_of(s)] == "active" and _hand_p(bi_hand_ctrls[s], s) == [1.5] * 20
        time.sleep(0.2)
        assert calls["n"] == 0 and plant.last_status()["phase"] == "IDLE"
    finally:
        _close_rig(node, plant, caller, spin)


@needs_asset
def test_bi_engage_right_first_goto_home_one_arm_hold_release(ros, bi_cm, bi_hand_ctrls):
    stub, calls = bi_cm
    node, plant, caller, spin = _bi_rig(ros, True, PD_BI_FAKE)
    try:
        ok, reasons = caller.trigger("engage")
        assert ok, reasons
        first = {s: next(i for i, r in enumerate(reasons) if r.startswith(f"{s}: phase")) for s in BI_SIDES}
        assert first["right"] < first["left"]                                   # 우팔 먼저
        for s in BI_SIDES:
            assert all(stub.known[n] == "active" for n in _fwd_of(s).values()) and stub.known[_jtc_of(s)] == "inactive"
            assert _hand_p(bi_hand_ctrls[s], s) == [1.5] * 20                  # 손 PID = 벤더값(09.06, 드라이버 기본과 같다)
        st = plant.wait(lambda st: all(a["phase"] in ("RAMPING", "TRACKING") for a in st["arms"].values()))
        assert st["phase"] in ("RAMPING", "TRACKING") and st["execute"] is True
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0 and not (plant.applied and set(_arm_can("right") + _arm_can("left"))
                                                      <= set(plant.applied[-1].name)):
            time.sleep(0.01)
        assert set(_arm_can("right") + _arm_can("left")) <= set(plant.applied[-1].name)   # applied 는 양팔을 싣는다
        ok, reasons = caller.trigger("goto_home")
        assert ok, reasons
        st = plant.last_status()
        assert all(st["arms"][s]["phase"] == "TRACKING" for s in BI_SIDES)
        assert all(np.abs(plant.q[s]).max() < 0.011 for s in BI_SIDES)          # 홈 = 0(차렷)
        # 양팔 목표 스트림 → 둘 다 TRACKING; 그 뒤 좌팔만 → 우팔 watchdog HOLD, 좌팔은 계속 TRACKING
        q_r, q_l = np.full(7, 0.03), np.full(7, 0.03)
        for seq in range(30):
            caller.target_named(_arm_can("right") + _arm_can("left"), np.concatenate([q_r, q_l]), seq)
            time.sleep(0.02)
        st = plant.wait(lambda st: st["phase"] == "TRACKING" and st["seq"] == 29)
        assert st["phase"] == "TRACKING" and st["seq"] == 29 and st["ok"] is True
        for seq in range(30, 70):
            caller.target_named(_arm_can("left"), q_l + 0.01, seq)
            time.sleep(0.02)
        st = plant.last_status()
        assert st["phase"] == "HOLD" and st["ok"] is False
        assert st["arms"]["right"]["phase"] == "HOLD" and st["arms"]["left"]["phase"] == "TRACKING"
        assert any(r.startswith("right:") and "watchdog" in r for r in st["reasons"])
        np.testing.assert_allclose(plant.q["right"], q_r, atol=0.02)               # 우팔 동결
        np.testing.assert_allclose(plant.q["left"], q_l + 0.01, atol=0.02)         # 좌팔은 새 목표를 따랐다
        np.testing.assert_allclose(plant.fwd["right"]["velocity"][-1], np.zeros(7))
        ok, reasons = caller.trigger("release")
        assert ok, reasons
        st = plant.wait_phase(("IDLE",))
        assert st["phase"] == "IDLE" and all(a["phase"] == "IDLE" for a in st["arms"].values())
        for s in BI_SIDES:
            assert stub.known[_jtc_of(s)] == "active" and all(stub.known[n] == "inactive" for n in _fwd_of(s).values())
    finally:
        _close_rig(node, plant, caller, spin)


@needs_asset
def test_bi_estop_holds_both_arms(ros, bi_cm, bi_hand_ctrls):
    stub, calls = bi_cm
    node, plant, caller, spin = _bi_rig(ros, True, PD_BI_FAKE)
    try:
        assert caller.trigger("engage")[0]
        plant.wait(lambda st: all(a["phase"] in ("RAMPING", "TRACKING") for a in st["arms"].values()))
        caller.estop(True)
        st = plant.wait(lambda st: all(a["phase"] == "HOLD" for a in st["arms"].values()))
        assert st["phase"] == "HOLD" and st["estop"] is True and all(a["phase"] == "HOLD" for a in st["arms"].values())
        assert caller.trigger("release")[0] and plant.wait_phase(("IDLE",))["phase"] == "IDLE"
        ok, reasons = caller.trigger("engage")
        assert ok is False and any("estop" in r for r in reasons)
    finally:
        _close_rig(node, plant, caller, spin)


@needs_asset
def test_bi_sides_param_selects_one_arm(ros, bi_cm, bi_hand_ctrls):
    stub, calls = bi_cm
    node, plant, caller, spin = _bi_rig(ros, True, PD_BI_FAKE, sides="left")
    try:
        st = _wait_status(plant)
        assert st["sides"] == ["left"] and set(st["arms"]) == {"left"}
        ok, reasons = caller.trigger("engage")
        assert ok, reasons
        assert stub.known[_jtc_of("left")] == "inactive" and stub.known[_jtc_of("right")] == "active"
        assert not any(n in stub.known for n in _fwd_of("right").values())      # 우팔은 손대지 않는다
        assert _hand_p(bi_hand_ctrls["right"], "right") == [1.5] * 20 and _hand_p(bi_hand_ctrls["left"], "left") == [1.5] * 20
        assert caller.trigger("release")[0] and plant.wait_phase(("IDLE",))["phase"] == "IDLE"
    finally:
        _close_rig(node, plant, caller, spin)
