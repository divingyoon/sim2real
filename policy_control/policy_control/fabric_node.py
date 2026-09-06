"""fabric_node — `chain.FabricStage` 의 얇은 rclpy 껍질 (플랜 §4.1/§4.5), **한 팔**(`side`) 담당.

policy 모드(계약에 정책이 있을 때 — 기존 경로):
    /policy_control/action ──▶ codec.decode_action ─┐   (전역 액션 벡터; 이 팔의 action_groups 조각은 디코더가 자른다)
    /policy_control/obs    ──▶ codec.decode_obs     ├─▶ FabricStage.tick ─▶ /policy_control/joint_target
    센서: robot yaml → select_side(side) 의 arm·ee·object     ┘                     /policy_control/palm_target
direct 모드(control_only 계약 — 정책 없음):
    /policy_control/palm_cmd (PoseStamped, base_link, 절대 palm 목표)  ─┐
    /policy_control/hand_cmd (JointState canonical, 선택 — 없으면 유지)  ├─▶ 계약 rate 타이머 ─▶ FabricStage.tick ─▶ joint_target
    센서(위와 같음)                                                      ┘
공통:
    /policy_control/episode (reset → FabricStage.reset + arm · start → direct 적분 시작 · stop/abort → 발행 중단)
    /policy_control/palm_pose (PoseStamped, latched) = fabric 현재 palm FK — tick 마다 + 심장박동마다.
        tools/palm_cmd.py 가 상대 이동의 기준으로 **한 번** 읽는다.
    /policy_control/status/fabric (side · mode 포함)

제어 논리는 전부 chain/decoder_core/fabric_core 에 있고 여기엔 배선만 있다. 어떤 메시지가 틀려도
죽지 않는다 — status ok:false + reasons 로 보고하고 다음 메시지를 기다린다. 판 여유 위반(FabricOut.abort)
이면 obs 의 episode/abort 서비스를 비동기로 부르고 다음 에피소드 이벤트까지 목표 발행을 멈춘다.
fabric 은 프로세스당 하나(fabrics_sim 제약) — 양팔은 fabric_node 두 개(side:=left / side:=right).
테스트는 `FabricNode(fabric=<가짜 backend FabricCore>)`.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rclpy.node import Node

if __package__ in (None, ""):          # `python policy_control/policy_control/<node>.py` (launch use_source 모드)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"

from . import _paths, codec  # noqa: E402
from .chain import ChainError, FabricIn, FabricOut, FabricStage, StageStatus, TableGuard  # noqa: E402
from .codec import CodecError  # noqa: E402
from .contract import ContractError, DeployContract, load_contract  # noqa: E402
from .decoder_core import DecoderError, euler_zyx_from_quat, make_decoder, side_soft_limits  # noqa: E402
from .fabric_core import FabricCore, FabricError  # noqa: E402
from .fk_numpy import FKError, make_fk  # noqa: E402
from .obs_core import split_segments  # noqa: E402
from .sources import RobotCfgError, SourceSet, load_robot_cfg, select_side  # noqa: E402

from grasp_s2r_core import _quat_from_matrix, _rot_euler_zyx  # noqa: E402  (scripts/)

NS = "/policy_control"
NODE_NAME = "fabric_node"
STAGE_NODE = "fabric"
TOPIC_OBS = f"{NS}/obs"
TOPIC_ACTION = f"{NS}/action"
TOPIC_EPISODE = f"{NS}/episode"
WARM_UP_HAND_DELTA = 0.1          # rad — 워밍업에서 손 목표를 홈에서 살짝 벗어나게 하는 양
TOPIC_PALM_CMD = f"{NS}/palm_cmd"
TOPIC_HAND_CMD = f"{NS}/hand_cmd"
TOPIC_JOINT_TARGET = f"{NS}/joint_target"
TOPIC_PALM_TARGET = f"{NS}/palm_target"
TOPIC_PALM_POSE = f"{NS}/palm_pose"
TOPIC_STATUS = f"{NS}/status/{STAGE_NODE}"
SRV_ABORT = f"{NS}/episode/abort"
MODE_POLICY = "policy"
MODE_DIRECT = "direct"
OBS_KEEP = 16                       # seq 짝맞춤용으로 들고 있는 최근 obs 수
HEARTBEAT_SEC = 1.0                 # 이벤트 없는 동안의 status/palm_pose 주기
SOURCE_ROLES = ("arm", "ee", "object")
OBJECT_SEGMENT_BUILDER = "object_pos_root"
DEFAULT_FRAME = "base_link"


class FabricNodeError(RuntimeError):
    """노드 배선/입력 오류 — status ok:false 사유가 된다."""


_HANDLED = (FabricNodeError, CodecError, ChainError, DecoderError, FabricError, FKError, ContractError,
            RobotCfgError, ValueError, KeyError)


@dataclass(frozen=True)
class ObsSlot:
    """한 obs 메시지에서 fabric 이 쓰는 값(seq 짝맞춤)."""

    seq: int
    gate_open: bool | None
    object_pos: np.ndarray | None


def _qos_chain():
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    return QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)


def _qos_latched(depth: int = 10):
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    # episode 구독 깊이 10: reset 직후 start 가 오면 depth 1 은 reset 을 버린다(run15). palm_pose 는 최신 1건.
    return QoSProfile(depth=depth, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def _qos_sensor():
    from rclpy.qos import qos_profile_sensor_data

    return qos_profile_sensor_data


def palm6_to_quat(palm6: np.ndarray) -> np.ndarray:
    """디코더 palm 목표(pos3 + euler_zyx3) → 쿼터니언 wxyz (grasp_s2r_core 규약)."""
    return _quat_from_matrix(_rot_euler_zyx(np.asarray(palm6, dtype=float).reshape(6)[3:]))


def palm6_from_pose(pos, quat_wxyz) -> np.ndarray:
    """PoseStamped(위치 + wxyz) → fabric palm 목표(pos3 + euler_zyx3)."""
    return np.concatenate([np.asarray(pos, dtype=float).reshape(3), euler_zyx_from_quat(quat_wxyz)])


def obs_slot(obs: np.ndarray, seq: int, contract: DeployContract) -> ObsSlot:
    """obs 벡터에서 gate 슬롯('gripper_gate')과 root 물체 위치(object_pos_root 빌더)를 뽑는다."""
    segs = split_segments(obs, contract)
    gate = segs.get("gripper_gate")
    obj_seg = next((s.name for s in contract.obs.segments if s.builder == OBJECT_SEGMENT_BUILDER), None)
    return ObsSlot(seq=int(seq), gate_open=None if gate is None else bool(float(gate[0]) > 0.5),
                   object_pos=None if obj_seg is None else np.asarray(segs[obj_seg], dtype=float).reshape(3))


def home_from_event(event: dict, contract: DeployContract, side: str | None = None) -> np.ndarray:
    """fabric 리셋 q = **계약 side fabric.home_q**(액션의 default_config). 이벤트의 home_q 는 로봇 리셋 홈(pd 몫)이라
    여기서는 쓰지 않는다 — 좌 v2 는 둘이 다르다(fabric J147 vs 로봇 LEFT_ARM_HOME_LOW, j4 21°·j7 28.6°)."""
    del event
    return np.array(contract.side(side or contract.primary_side).fabric.home_q, dtype=float)


class FabricNode(Node):
    """subscribe → codec.decode → FabricStage.tick → codec.encode → publish. 콜백은 작게, 상태는 스테이지에."""

    def __init__(self, *, context=None, fabric: FabricCore | None = None, fk=None,
                 parameter_overrides=None) -> None:
        super().__init__(NODE_NAME, context=context, parameter_overrides=parameter_overrides)
        self.declare_parameter("contract", "")
        self.declare_parameter("robot", "")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("side", "")
        contract_path = Path(str(self.get_parameter("contract").value)).expanduser()
        robot_path = Path(str(self.get_parameter("robot").value)).expanduser()
        device = str(self.get_parameter("device").value)
        if not contract_path.is_file() or not robot_path.is_file():
            raise FabricNodeError(f"contract/robot 파일이 없다: {contract_path} / {robot_path}")
        self.contract = load_contract(contract_path)
        self.side = str(self.get_parameter("side").value) or self.contract.primary_side
        self.side_cfg = self.contract.side(self.side)
        self.mode = MODE_DIRECT if self.contract.control_only else MODE_POLICY
        self.robot_cfg = select_side(load_robot_cfg(robot_path), self.side)
        self.fabric = fabric if fabric is not None else FabricCore(self.contract, device, side=self.side)
        if self.fabric.side != self.side:
            raise FabricNodeError(f"fabric 은 {self.fabric.side} 팔인데 side 파라미터는 {self.side}")
        self._warm_up()
        fk = fk if fk is not None else self._make_fk()
        decoder = make_decoder(self.contract, side=self.side, hand_soft_limits=side_soft_limits(self.contract, self.side))
        self.stage = FabricStage(self.contract, decoder, self.fabric, TableGuard.from_robot_cfg(self.robot_cfg.table),
                                 fk=fk)
        self.sources = SourceSet(self.robot_cfg)
        self.segments = [(s.name, s.dim) for s in self.contract.obs.segments]
        self.frame = self.robot_cfg.sources["object"].frame or DEFAULT_FRAME
        self._obs: dict[int, ObsSlot] = {}
        self._armed = False
        self._running = False           # direct 모드: start 이벤트 뒤에만 적분한다
        self._episode = 0
        self._abort_sent = False
        self._last_error: str | None = None
        self._t_status = 0.0
        self._palm_cmd: np.ndarray | None = None
        self._hand_cmd: np.ndarray | None = None
        self._seq = 0
        self._wire()
        self.get_logger().info(f"fabric_node up · side {self.side} · mode {self.mode} · contract {contract_path.name} · "
                               f"robot {self.robot_cfg.robot} · device {self.fabric.device} · "
                               f"fk {self.contract.obs.fk.get('kind')}")

    def _make_fk(self):
        """판 여유 가드가 obs FK(TCP)를 쓰는 좌 그리퍼만 제공자를 만든다 — 그 외는 fabric 손끝 FK 로 잰다."""
        if self.contract.obs.fk.get("kind") != "left_gripper":
            return None
        return make_fk(self.contract, _paths.RL_WS, side=self.side)

    # ---------------------------------------------------------------- wiring
    def _wire(self) -> None:
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray, String
        from std_srvs.srv import Trigger

        self._pub_target = self.create_publisher(JointState, TOPIC_JOINT_TARGET, _qos_chain())
        self._pub_palm = self.create_publisher(PoseStamped, TOPIC_PALM_TARGET, _qos_chain())
        self._pub_pose = self.create_publisher(PoseStamped, TOPIC_PALM_POSE, _qos_latched(depth=1))
        self._pub_status = self.create_publisher(String, TOPIC_STATUS, _qos_chain())
        self._abort_client = self.create_client(Trigger, SRV_ABORT)
        if self.mode == MODE_POLICY:
            # obs 구독을 action 보다 먼저 만든다 — 같은 wait-set 사이클에 둘 다 도착하면 생성 순으로 처리된다
            self.create_subscription(Float64MultiArray, TOPIC_OBS, self._on_obs, _qos_chain())
            self.create_subscription(Float64MultiArray, TOPIC_ACTION, self._on_action, _qos_chain())
        else:
            self.create_subscription(PoseStamped, TOPIC_PALM_CMD, self._on_palm_cmd, _qos_chain())
            self.create_subscription(JointState, TOPIC_HAND_CMD, self._on_hand_cmd, _qos_chain())
            self.create_timer(float(self.contract.rate.step_dt), self._on_direct_tick)
        self.create_subscription(String, TOPIC_EPISODE, self._on_episode, _qos_latched())
        by_topic: dict[tuple[str, str], list[str]] = {}
        for role in SOURCE_ROLES:
            s = self.robot_cfg.sources[role]
            by_topic.setdefault((s.topic, s.type), []).append(role)
        for (topic, typ), roles in by_topic.items():
            if typ == "joint_state":
                self.create_subscription(JointState, topic, self._joint_cb(roles), _qos_sensor())
            else:
                self.create_subscription(PoseStamped, topic, self._pose_cb(roles), _qos_sensor())
        self.create_timer(HEARTBEAT_SEC, self._heartbeat)

    def _heartbeat(self) -> None:
        """이벤트가 없어도 status/palm_pose 가 살아 있게(기동 확인·소스 상태·현재 palm). 최근 status 가 있으면 건너뛴다."""
        if time.monotonic() - self._t_status < HEARTBEAT_SEC:
            return
        state = self.sources.snapshot(time.monotonic())
        reasons = tuple(f"{what} {list(v)}" for what, v in (("missing", state.missing), ("stale", state.stale)) if v)
        try:
            self._publish_pose(self.fabric.palm_pose(self.fabric.q), time.time())
        except _HANDLED as exc:
            reasons = reasons + (f"palm_pose: {exc}",)
        self._publish_status(self._shell_status(seq=-1, ok=not reasons, reasons=reasons), time.perf_counter())

    def _joint_cb(self, roles: list[str]):
        def cb(msg) -> None:
            try:
                sample = codec.decode_joint_state(msg)
                for role in roles:
                    self.sources.update_from_joint_state(role, sample, time.monotonic())
            except _HANDLED as exc:
                self._note_error(f"joint_state({roles}): {exc}")
        return cb

    def _pose_cb(self, roles: list[str]):
        def cb(msg) -> None:
            try:
                sample = codec.decode_pose(msg)
                for role in roles:
                    self.sources.update_from_pose(role, sample, time.monotonic())
            except _HANDLED as exc:
                self._note_error(f"pose({roles}): {exc}")
        return cb

    def _note_error(self, text: str) -> None:
        if text != self._last_error:
            self.get_logger().warning(text)
        self._last_error = text

    # ---------------------------------------------------------------- episode
    def _on_episode(self, msg) -> None:
        t0 = time.perf_counter()
        try:
            ev = json.loads(msg.data)
            event, episode = str(ev["event"]), int(ev["episode"])
        except (TypeError, ValueError, KeyError) as exc:
            self._status_error(f"episode JSON: {exc}", t0)
            return
        if event == "reset":
            self._reset(ev, episode, t0)
            return
        if event == "start":
            self._running = self._armed and episode == self._episode
        elif event in ("stop", "abort"):
            self._armed = self._running = False
        self._episode = episode
        self._publish_status(self._shell_status(seq=-1, ok=True, reasons=(f"episode {event}",)), t0)

    def _warm_up(self, steps: int = 5) -> None:
        """첫 fabric 스텝(CUDA/warp 초기화 ~180 ms)을 에피소드 밖으로 뺀다 — 시작 직후 큐 지연 방지.

        손 목표가 **홈과 다른** 스텝도 밟는다: 우 tesollo fabric 은 손 목표가 처음 바뀌는 tick 에서 300 ms
        (fake_direct_right_run1, seq 877) 를 더 쓰며 그 한 번이 pd 워치독(0.25 s) HOLD 를 만든다."""
        f = self.fabric
        home = np.array(f.cfg.home_q, dtype=float)
        try:
            f.reset(home)
            hand = None if f.n_hand == 0 else np.array(f.q[f.n_arm:], dtype=float)
            palm = np.array(f.palm_pose(f.q), dtype=float)
            for _ in range(steps):
                f.step(palm, hand_target=hand)
            if hand is not None:
                for _ in range(steps):
                    f.step(palm, hand_target=hand + WARM_UP_HAND_DELTA)
            f.reset(home)
        except Exception as exc:  # noqa: BLE001 — 워밍업 실패는 치명적이지 않다, 기록만
            self.get_logger().warning(f"fabric warm-up skipped: {exc}")

    def _reset(self, ev: dict, episode: int, t0: float) -> None:
        try:
            state = self.sources.snapshot(time.monotonic())
            self._check_state(state, need_object=False)
            anchor = ev.get("object_anchor")
            rs = self.stage.reset(state.arm_q, state.ee_names, state.ee_q, object_anchor=anchor,
                                  home_q=home_from_event(ev, self.contract, self.side), episode=episode)
        except _HANDLED as exc:
            self._armed = self._running = False
            self._status_error(f"episode reset {episode} failed: {exc}", t0)
            return
        self._episode, self._armed, self._abort_sent, self._running = episode, True, False, False
        self._obs.clear()
        self._palm_cmd, self._hand_cmd, self._seq = rs.palm6_home.copy(), None, 0   # direct: 홈 palm 유지부터
        self._publish_pose(rs.palm6_home, time.time())
        self._publish_status(self._shell_status(seq=-1, ok=True, reasons=()), t0)

    # ---------------------------------------------------------------- policy mode: obs / action
    def _on_obs(self, msg) -> None:
        try:
            obs, seq = codec.decode_obs(msg, self.segments)
            self._obs[int(seq)] = obs_slot(obs, seq, self.contract)
        except _HANDLED as exc:
            self._status_error(f"obs: {exc}", time.perf_counter())
            return
        for old in sorted(self._obs)[:-OBS_KEEP]:
            del self._obs[old]

    def _on_action(self, msg) -> None:
        t0 = time.perf_counter()
        seq = -1
        try:
            action, seq = codec.decode_action(msg, self.contract.policy.action_dim)
            self._tick(action, int(seq), t0)
        except _HANDLED as exc:
            self._status_error(f"action: {exc}", t0, seq=int(seq))

    def _tick(self, action: np.ndarray, seq: int, t0: float) -> None:
        if not self._armed:
            raise FabricNodeError(f"no running episode (armed=False, episode {self._episode})")
        slot = self._obs.get(seq)
        if slot is None:
            raise FabricNodeError(f"obs seq {seq} not received (have {sorted(self._obs)[-3:]})")
        state = self.sources.snapshot(time.monotonic())
        need_object = slot.object_pos is None and self.stage.decoder.hand is not None
        self._check_state(state, need_object=need_object)
        object_pos = slot.object_pos if slot.object_pos is not None else state.object_pos
        out = self.stage.tick(FabricIn(action=action, seq=seq, gate_open=slot.gate_open, object_pos=object_pos,
                                       arm_q_meas=state.arm_q, ee_names=state.ee_names, ee_q=state.ee_q))
        self._emit(out, seq, t0)

    # ---------------------------------------------------------------- direct mode: palm_cmd / hand_cmd / timer
    def _on_palm_cmd(self, msg) -> None:
        try:
            pose = codec.decode_pose(msg)
            if pose.frame and pose.frame != self.frame:
                raise FabricNodeError(f"palm_cmd frame {pose.frame!r} ≠ {self.frame!r}")
            self._palm_cmd = palm6_from_pose(pose.pos, pose.quat)
        except _HANDLED as exc:
            self._status_error(f"palm_cmd: {exc}", time.perf_counter())

    def _on_hand_cmd(self, msg) -> None:
        try:
            sample = codec.decode_joint_state(msg)
            pos, _ = codec.select_joints(sample, self.stage.hand_joints)
            self._hand_cmd = np.asarray(pos, dtype=float)
        except _HANDLED as exc:
            self._status_error(f"hand_cmd: {exc}", time.perf_counter())

    def _on_direct_tick(self) -> None:
        """계약 rate 마다: 최신 palm_cmd(없으면 리셋 시 홈 palm)·hand_cmd(없으면 유지)로 한 스텝 적분한다."""
        if not (self._armed and self._running):
            return
        t0 = time.perf_counter()
        seq, self._seq = self._seq, self._seq + 1
        try:
            state = self.sources.snapshot(time.monotonic())
            self._check_state(state, need_object=False)
            out = self.stage.tick(FabricIn(action=self._palm_cmd, seq=seq, gate_open=None, object_pos=None,
                                           arm_q_meas=state.arm_q, ee_names=state.ee_names, ee_q=state.ee_q,
                                           hand_cmd=self._hand_cmd))
            self._emit(out, seq, t0)
        except _HANDLED as exc:
            self._status_error(f"direct tick: {exc}", t0, seq=seq)

    # ---------------------------------------------------------------- outputs
    def _emit(self, out: FabricOut, seq: int, t0: float) -> None:
        if out.abort:
            self._abort(out.reasons)
            self._publish_status(out.status, t0)
            return
        stamp = time.time()
        self._pub_target.publish(codec.encode_joint_target(out.joint_names, out.q, out.qd, str(self._episode),
                                                           seq, stamp=stamp))
        self._pub_palm.publish(codec.encode_pose(out.palm6[:3], palm6_to_quat(out.palm6), self.frame, stamp=stamp))
        if out.palm6_now is not None:
            self._publish_pose(out.palm6_now, stamp)
        self._publish_status(out.status, t0)

    def _publish_pose(self, palm6: np.ndarray, stamp: float) -> None:
        self._pub_pose.publish(codec.encode_pose(palm6[:3], palm6_to_quat(palm6), self.frame, stamp=stamp))

    def _check_state(self, state, *, need_object: bool) -> None:
        roles = ["arm", "ee"] + (["object"] if need_object else [])
        missing = [r for r in roles if r in state.missing]
        stale = [r for r in roles if r in state.stale]
        if missing or stale:
            raise FabricNodeError(f"sources missing {missing} stale {stale}")

    # ---------------------------------------------------------------- abort
    def _abort(self, reasons: tuple) -> None:
        self._armed = self._running = False
        if self._abort_sent:
            return
        self._abort_sent = True
        from std_srvs.srv import Trigger

        self.get_logger().error(f"fabric abort: {'; '.join(reasons)} → {SRV_ABORT}")
        if not self._abort_client.service_is_ready():
            self.get_logger().error("episode/abort service not ready — targets stopped, obs not notified")
            return
        self._abort_client.call_async(Trigger.Request())

    # ---------------------------------------------------------------- status
    def _shell_status(self, *, seq: int, ok: bool, reasons: tuple) -> StageStatus:
        phase = "running" if self._armed else ("aborted" if self._abort_sent else "idle")
        return StageStatus(node=STAGE_NODE, phase=phase, episode=self._episode, seq=int(seq), ok=ok,
                           reasons=tuple(reasons), proc_ms=0.0)

    def _status_error(self, text: str, t0: float, seq: int = -1) -> None:
        self._note_error(text)
        self._publish_status(self._shell_status(seq=seq, ok=False, reasons=(text,)), t0)

    def _publish_status(self, status: StageStatus, t0: float) -> None:
        body = status.as_dict()
        body.update({"stage_ms": body["proc_ms"], "proc_ms": (time.perf_counter() - t0) * 1e3,
                     "armed": self._armed, "running": self._running, "side": self.side, "mode": self.mode,
                     "t_pub_ns": time.time_ns()})
        self._t_status = time.monotonic()
        self._pub_status.publish(codec.encode_status(body))


def main(argv=None) -> int:
    import rclpy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

    rclpy.init(args=argv)
    node = None
    try:
        node = FabricNode()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
