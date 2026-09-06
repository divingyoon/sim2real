"""pd 노드 출력 백엔드 3종 + DG-5F PID 게인 클라이언트 (플랜 §4.4).

- ``ArmForwardBackend``  OpenArm 좌/우 — forward_{position,velocity,effort}_controller/commands,
  SOURCE 순·프로필 부호(위치는 프로필 한계 clamp, 속도·토크는 부호만).
- ``GripperJtcBackend``  좌 스톡 그리퍼 — 단일점 JointTrajectory(tfs 0, 위치만), 과압착 가드
  ``q_cmd = max(q*, q_meas − overtravel)`` (닫을 때만), 직전 지령 기준 속도 제한.
- ``Dg5fJtcBackend``     DG-5F — 단일점 JointTrajectory(tfs 0, 위치만, q̇* 버림), source finger-major.
- ``HandGainsClient``    /dg5f_right/dg5f_right_controller 의 gains.<joint>.{p,d} 를 GetParameters
  한 번으로 읽고 불일치면 SetParameters **한 번**(scripts/ops/apply_hand_gains.py 규약).

★하드 안전 규칙: ``execute=False`` 면 publisher 를 만들지도, 발행하지도, 서비스를 부르지도 않는다.
  법칙 결과(source 순 값)는 그대로 돌려줘 status/applied 발행과 드라이런 검증에 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import _paths  # noqa: F401  (scripts/ on sys.path)
from .controller_switch import ServiceCaller
from jtc_bridge_core import JointRemap, velocity_limited_target  # noqa: E402

FORWARD_KINDS = ("position", "velocity", "effort")
PARAM_KEYS = ("p", "d")
GAIN_TOL = 1e-9
QOS_DEPTH = 10


class BackendError(ValueError):
    """잘못된 지령(길이·NaN·dt) — 발행 전에 죽는다."""


def forward_topic(side: str, kind: str) -> str:
    return f"/{side}_forward_{kind}_controller/commands"


def _vec(values, n: int, what: str) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.shape[0] != n:
        raise BackendError(f"{what}: 길이 {v.shape[0]} != {n}")
    if not np.all(np.isfinite(v)):
        raise BackendError(f"{what}: 비유한값")
    return v


def _check_dt(dt: float) -> float:
    if not (dt > 0.0) or not np.isfinite(dt):
        raise BackendError(f"dt 는 양수여야 한다: {dt}")
    return float(dt)


class _GuardedPublisher:
    """execute=False 면 publisher 자체를 만들지 않는다 → count_publishers == 0."""

    def __init__(self, node, msg_type, topic: str, execute: bool) -> None:
        self.topic = topic
        self.execute = bool(execute)
        self._pub = node.create_publisher(msg_type, topic, QOS_DEPTH) if self.execute else None
        self.count = 0

    def publish(self, msg) -> None:
        if self._pub is None:
            return
        self._pub.publish(msg)
        self.count += 1


def _single_point_trajectory(joint_names: Sequence[str], positions: np.ndarray):
    """time_from_start=0 단일점 — interpolation_method none 컨트롤러 규약 (jtc_bridge_core 참조)."""
    from builtin_interfaces.msg import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    pt = JointTrajectoryPoint()
    pt.positions = [float(v) for v in positions]
    pt.time_from_start = Duration(sec=0, nanosec=0)
    msg = JointTrajectory()
    msg.joint_names = [str(n) for n in joint_names]
    msg.points = [pt]
    return msg


# ---------------------------------------------------------------- arm forward
@dataclass(frozen=True)
class ArmWritten:
    names: tuple[str, ...]      # source 순
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray


class ArmForwardBackend:
    def __init__(self, node, side: str, remap: JointRemap, *, execute: bool) -> None:
        from std_msgs.msg import Float64MultiArray

        self.side = str(side)
        self.remap = remap
        self.names = tuple(remap.output_source)
        self.execute = bool(execute)
        self._pubs = {k: _GuardedPublisher(node, Float64MultiArray, forward_topic(side, k), self.execute)
                      for k in FORWARD_KINDS}

    @property
    def publish_count(self) -> int:
        return sum(p.count for p in self._pubs.values())

    def _signed(self, values, what: str) -> np.ndarray:
        v = _vec(values, self.remap.input_len, what)
        return v[self.remap.input_idx] * self.remap.sign

    def write(self, cmd) -> ArmWritten:
        """cmd: pd_law.PdCommand(q, qd, tau) canonical 순 → 3 토픽 source 순."""
        q = self.remap.apply(_vec(cmd.q, self.remap.input_len, "q"))
        qd = self._signed(cmd.qd, "qd")
        tau = self._signed(cmd.tau, "tau")
        self._publish("position", q)
        self._publish("velocity", qd)
        self._publish("effort", tau)
        return ArmWritten(names=self.names, q=q, qd=qd, tau=tau)

    def zero_release(self) -> None:
        """velocity/effort 0 ×N — position 은 손대지 않는다(JTC 가 현재 위치를 이어받는다)."""
        zeros = np.zeros(len(self.names))
        self._publish("velocity", zeros)
        self._publish("effort", zeros)

    def _publish(self, kind: str, values: np.ndarray) -> None:
        from std_msgs.msg import Float64MultiArray

        self._pubs[kind].publish(Float64MultiArray(data=[float(v) for v in values]))


# ---------------------------------------------------------------- gripper (single-point JTC)
@dataclass(frozen=True)
class GripperCmd:
    q_star: float               # 목표 개구 [m]
    q_meas: float               # 실측 개구 [m]
    dt: float


@dataclass(frozen=True)
class GripperWritten:
    q_cmd: float
    limited: bool
    overtravel_guarded: bool


class GripperJtcBackend:
    def __init__(self, node, topic: str, joint: str, close_overtravel_m: float, max_vel: float, *,
                 lower: float, upper: float, execute: bool) -> None:
        from trajectory_msgs.msg import JointTrajectory

        if close_overtravel_m < 0.0 or max_vel <= 0.0 or not lower < upper:
            raise ValueError("close_overtravel_m ≥ 0, max_vel > 0, lower < upper 여야 한다")
        self.joint = str(joint)
        self.overtravel = float(close_overtravel_m)
        self.max_vel = float(max_vel)
        self.lower, self.upper = float(lower), float(upper)
        self.execute = bool(execute)
        self._pub = _GuardedPublisher(node, JointTrajectory, topic, self.execute)
        self._prev: float | None = None

    @property
    def publish_count(self) -> int:
        return self._pub.count

    def write(self, cmd: GripperCmd) -> GripperWritten:
        q_star = float(_vec([cmd.q_star], 1, "q_star")[0])
        q_meas = float(_vec([cmd.q_meas], 1, "q_meas")[0])
        dt = _check_dt(cmd.dt)
        q_t = float(np.clip(q_star, self.lower, self.upper))
        guarded = False
        if q_t < q_meas:                                    # 닫는 중 → 과압착 가드
            floor = q_meas - self.overtravel
            guarded = q_t < floor
            q_t = max(q_t, floor)
        prev = q_meas if self._prev is None else self._prev
        q_cmd = float(velocity_limited_target(np.array([q_t]), np.array([prev]), self.max_vel, dt)[0])
        q_cmd = float(np.clip(q_cmd, self.lower, self.upper))
        self._prev = q_cmd
        self._pub.publish(_single_point_trajectory([self.joint], np.array([q_cmd])))
        return GripperWritten(q_cmd=q_cmd, limited=abs(q_cmd - q_t) > 1e-12, overtravel_guarded=guarded)

    def zero_release(self) -> None:
        """JTC 는 마지막 점을 홀딩한다 — 보낼 0 이 없다(no-op). 직전 지령만 잊는다."""
        self._prev = None


# ---------------------------------------------------------------- DG-5F (single-point JTC)
@dataclass(frozen=True)
class HandCmd:
    q_star: np.ndarray          # canonical(remap 입력) 순 20
    qd_star: np.ndarray | None  # 버린다 — 서명 호환용
    dt: float
    q_meas: np.ndarray | None = None   # canonical 순 실측 — 첫 지령의 속도 제한 기준(없으면 첫 지령 무제한)


@dataclass(frozen=True)
class HandWritten:
    names: tuple[str, ...]      # source finger-major
    q_cmd: np.ndarray
    limited: bool


class Dg5fJtcBackend:
    def __init__(self, node, topic: str, remap: JointRemap, max_vel: float, *, execute: bool) -> None:
        from trajectory_msgs.msg import JointTrajectory

        if max_vel <= 0.0:
            raise ValueError("max_vel > 0 이어야 한다")
        self.remap = remap
        self.names = tuple(remap.output_source)
        self.max_vel = float(max_vel)
        self.execute = bool(execute)
        self._pub = _GuardedPublisher(node, JointTrajectory, topic, self.execute)
        self._prev: np.ndarray | None = None

    @property
    def publish_count(self) -> int:
        return self._pub.count

    def write(self, cmd: HandCmd) -> HandWritten:
        dt = _check_dt(cmd.dt)
        q_t = self.remap.apply(_vec(cmd.q_star, self.remap.input_len, "hand q_star"))
        prev = self._prev if self._prev is not None else self._seed(cmd.q_meas, q_t)
        q_cmd = velocity_limited_target(q_t, prev, self.max_vel, dt)
        q_cmd = np.clip(q_cmd, self.remap.lower, self.remap.upper)
        self._prev = q_cmd
        self._pub.publish(_single_point_trajectory(self.names, q_cmd))
        return HandWritten(names=self.names, q_cmd=q_cmd, limited=bool(np.any(np.abs(q_cmd - q_t) > 1e-12)))

    def _seed(self, q_meas, q_t: np.ndarray) -> np.ndarray:
        """첫 지령의 기준: 실측(있으면) — 0 벡터에서 램프하면 손이 0 쪽으로 튄다."""
        if q_meas is None:
            return q_t
        return self.remap.apply(_vec(q_meas, self.remap.input_len, "hand q_meas"))

    def zero_release(self) -> None:
        self._prev = None


# ---------------------------------------------------------------- hand PID gains
def hand_controller_name(topic: str, namespace: str | None = None) -> str:
    """JTC 토픽 → 파라미터 서버(컨트롤러 노드) 이름: '/dg5f_left/dg5f_left_controller/joint_trajectory' →
    '/dg5f_left/dg5f_left_controller'. ``namespace`` (robot yaml group) 가 있으면 토픽이 그 안에 있는지 확인한다 —
    손마다 controller_manager 가 자기 namespace 에서 돌므로 어긋나면 엉뚱한 손의 PID 를 바꾼다."""
    controller = str(topic).rstrip("/").rsplit("/", 1)[0]
    if namespace:
        ns = "/" + str(namespace).strip("/")
        if not controller.startswith(ns + "/"):
            raise ValueError(f"hand topic {topic!r} is not inside namespace {ns!r}")
    if not controller.strip("/"):
        raise ValueError(f"hand topic {topic!r} has no controller segment")
    return controller


class HandGainsClient:
    """gains.<joint>.p/.d 대조·적용. ``check``/``check_and_apply`` → (ok, reasons), 예외 없음."""

    def __init__(self, node, controller: str, joints: Sequence[str], timeout_sec: float, *,
                 execute: bool) -> None:
        self.controller = controller.rstrip("/")
        self.joints = tuple(str(j) for j in joints)
        self.execute = bool(execute)
        self._caller = ServiceCaller(node, "hand_gains", timeout_sec)
        self.names = [f"gains.{j}.{k}" for j in self.joints for k in PARAM_KEYS]

    def close(self) -> None:
        self._caller.close()

    def _read(self) -> tuple[dict[str, float] | None, str | None]:
        from rcl_interfaces.msg import ParameterType
        from rcl_interfaces.srv import GetParameters

        req = GetParameters.Request()
        req.names = list(self.names)
        resp, reason = self._caller.call(GetParameters, f"{self.controller}/get_parameters", req)
        if resp is None:
            return None, reason
        if len(resp.values) != len(self.names):
            return None, f"get_parameters returned {len(resp.values)} values for {len(self.names)} names"
        out: dict[str, float] = {}
        for name, val in zip(self.names, resp.values):
            if val.type != ParameterType.PARAMETER_DOUBLE:
                return None, f"parameter {name} is not a double (type {val.type})"
            out[name] = float(val.double_value)
        return out, None

    def _mismatches(self, current: dict[str, float], p: float, d: float) -> list[str]:
        want = {"p": float(p), "d": float(d)}
        return [f"{n}={current[n]:g} (want {want[n.rsplit('.', 1)[1]]:g})" for n in self.names
                if abs(current[n] - want[n.rsplit(".", 1)[1]]) > GAIN_TOL]

    def check(self, p: float, d: float) -> tuple[bool, list[str]]:
        current, reason = self._read()
        if current is None:
            return False, [f"hand gains read failed: {reason}"]
        bad = self._mismatches(current, p, d)
        if bad:
            return False, [f"hand gains mismatch ({len(bad)}/{len(self.names)}): {', '.join(bad[:4])}"]
        return True, []

    def check_and_apply(self, p: float, d: float) -> tuple[bool, list[str]]:
        ok, reasons = self.check(p, d)
        if ok or not reasons[0].startswith("hand gains mismatch"):
            return ok, reasons
        if not self.execute:
            return False, reasons + [f"dry_run: would set p={p:g} d={d:g} on {self.controller}"]
        applied, why = self._apply(p, d)
        if not applied:
            return False, reasons + [why]
        ok2, reasons2 = self.check(p, d)
        return ok2, reasons + [f"applied p={p:g} d={d:g} on {self.controller}"] + reasons2

    def _apply(self, p: float, d: float) -> tuple[bool, str]:
        from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
        from rcl_interfaces.srv import SetParameters

        req = SetParameters.Request()
        want = {"p": float(p), "d": float(d)}
        for name in self.names:
            value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=want[name.rsplit(".", 1)[1]])
            req.parameters.append(Parameter(name=name, value=value))
        resp, reason = self._caller.call(SetParameters, f"{self.controller}/set_parameters", req)
        if resp is None:
            return False, f"set_parameters failed: {reason}"
        n_ok = sum(1 for r in resp.results if r.successful)
        if n_ok != len(req.parameters):
            return False, f"set_parameters accepted {n_ok}/{len(req.parameters)}"
        return True, ""
