"""codec — ROS 2 메시지 ↔ numpy / frozen dataclass 순수 변환.

이 패키지에서 **메시지 타입을 아는 유일한 모듈**이다. 메시지 클래스는 함수 안에서 지연
import 하므로 ROS 없이도 모듈은 import 된다(코어·테스트가 dataclass 만 쓸 수 있게).

토픽 계약(플랜 §4.1):
  · obs / action  — std_msgs/Float64MultiArray.
    ★비표준 사용: `layout.dim[i].label` = 세그먼트 이름, `dim[i].size` = 그 폭,
      `layout.data_offset` = **seq**. 표준 의미(다차원 stride·데이터 오프셋)로는 쓰지
      않는다. 수신측은 라벨·크기가 계약과 다르면 **거부**한다(조용히 슬라이스하지 않는다).
  · joint_target  — sensor_msgs/JointState. name=canonical, position=q*, velocity=q̇*,
    effort=0, `header.frame_id = "<episode>:<seq>"`.
  · pose          — geometry_msgs/PoseStamped. 쿼터니언은 내부에서 **wxyz** 로 다룬다.
  · status        — std_msgs/String(JSON).
관절은 언제나 **이름으로** 옮긴다. 결손은 에러이며 0 으로 채우지 않는다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import _paths  # noqa: F401  (scripts/ 를 sys.path 에)
from grasp_s2r_obs_builder import reorder

Segments = Sequence[tuple[str, int]]


class CodecError(ValueError):
    """메시지가 계약(라벨·크기·이름·형식)과 맞지 않는다."""


# ------------------------------------------------------------------ 순수 표본
@dataclass(frozen=True)
class JointSample:
    """JointState 한 장 — 이름 순서는 메시지 그대로. 재정렬은 `select_joints` 로."""

    names: tuple
    position: np.ndarray
    velocity: np.ndarray | None
    effort: np.ndarray | None
    stamp: float


@dataclass(frozen=True)
class ArraySample:
    data: np.ndarray
    labels: tuple
    sizes: tuple
    seq: int


@dataclass(frozen=True)
class PoseSample:
    pos: np.ndarray          # (3,)
    quat: np.ndarray         # (4,) wxyz
    frame: str
    stamp: float


@dataclass(frozen=True)
class JointTarget:
    names: tuple
    position: np.ndarray
    velocity: np.ndarray
    episode: str
    seq: int
    stamp: float


# ------------------------------------------------------------------ helpers
def _vec(values, n: int | None, what: str) -> np.ndarray:
    a = np.array(values, dtype=np.float64).reshape(-1)
    if n is not None and a.size != n:
        raise CodecError(f"{what}: {a.size}개 — 계약은 {n}개")
    if not np.all(np.isfinite(a)):
        raise CodecError(f"{what}: NaN/inf 가 있다")
    return a


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def sec_to_stamp(t: float | None):
    from builtin_interfaces.msg import Time

    if t is None:
        return Time()
    sec = int(t)
    return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)))


def _multiarray(data: np.ndarray, labels: Sequence[str], sizes: Sequence[int], seq: int):
    from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout

    if len(labels) != len(sizes):
        raise CodecError("labels 와 sizes 길이가 다르다")
    dims = [MultiArrayDimension(label=str(lb), size=int(sz), stride=0) for lb, sz in zip(labels, sizes)]
    layout = MultiArrayLayout(dim=dims, data_offset=int(seq))
    return Float64MultiArray(layout=layout, data=[float(v) for v in data])


def decode_float_array(msg) -> ArraySample:
    """Float64MultiArray → ArraySample (라벨·크기·seq 그대로)."""
    data = _vec(msg.data, None, "Float64MultiArray.data")
    labels = tuple(str(d.label) for d in msg.layout.dim)
    sizes = tuple(int(d.size) for d in msg.layout.dim)
    return ArraySample(data=data, labels=labels, sizes=sizes, seq=int(msg.layout.data_offset))


def encode_float_array(data, labels: Sequence[str], sizes: Sequence[int], seq: int):
    return _multiarray(_vec(data, None, "float_array"), labels, sizes, seq)


# ------------------------------------------------------------------ obs / action
def encode_obs(obs, segments: Segments, seq: int):
    """obs 벡터 → Float64MultiArray (dim.label=세그먼트, data_offset=seq)."""
    n = sum(int(d) for _, d in segments)
    a = _vec(obs, n, "obs")
    return _multiarray(a, [nm for nm, _ in segments], [d for _, d in segments], seq)


def decode_obs(msg, segments: Segments) -> tuple[np.ndarray, int]:
    """라벨·크기가 계약과 한 칸이라도 다르면 CodecError."""
    s = decode_float_array(msg)
    want = tuple((str(nm), int(d)) for nm, d in segments)
    got = tuple(zip(s.labels, s.sizes))
    if got != want:
        raise CodecError(f"obs 레이아웃 불일치: 수신 {got} vs 계약 {want}")
    n = sum(d for _, d in want)
    if s.data.size != n:
        raise CodecError(f"obs 길이 {s.data.size} vs 계약 {n}")
    return s.data.copy(), s.seq


def encode_action(action, seq: int):
    a = _vec(action, None, "action")
    return _multiarray(a, ["action"], [a.size], seq)


def decode_action(msg, dim: int) -> tuple[np.ndarray, int]:
    s = decode_float_array(msg)
    if s.labels != ("action",) or s.sizes != (int(dim),) or s.data.size != int(dim):
        raise CodecError(f"action 레이아웃 불일치: {s.labels}/{s.sizes}/{s.data.size} vs action[{dim}]")
    return s.data.copy(), s.seq


# ------------------------------------------------------------------ joint state / target
def encode_joint_state(names: Sequence[str], position, velocity=None, effort=None, stamp: float | None = None):
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header

    n = len(names)
    msg = JointState(header=Header(stamp=sec_to_stamp(stamp), frame_id=""))
    msg.name = [str(x) for x in names]
    msg.position = [float(v) for v in _vec(position, n, "position")]
    msg.velocity = [] if velocity is None else [float(v) for v in _vec(velocity, n, "velocity")]
    msg.effort = [] if effort is None else [float(v) for v in _vec(effort, n, "effort")]
    return msg


def decode_joint_state(msg) -> JointSample:
    names = tuple(str(x) for x in msg.name)
    n = len(names)
    if len(set(names)) != n:
        raise CodecError("JointState 에 중복 이름이 있다")
    pos = _vec(msg.position, n, "JointState.position")
    vel = _vec(msg.velocity, n, "JointState.velocity") if len(msg.velocity) else None
    eff = _vec(msg.effort, n, "JointState.effort") if len(msg.effort) else None
    return JointSample(names=names, position=pos, velocity=vel, effort=eff, stamp=stamp_to_sec(msg.header.stamp))


def select_joints(sample: JointSample, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray | None]:
    """이름으로 (position, velocity) 를 뽑아 `names` 순으로 돌려준다. 결손 → CodecError."""
    try:
        pos = reorder(sample.position, sample.names, list(names))
        vel = None if sample.velocity is None else reorder(sample.velocity, sample.names, list(names))
    except KeyError as exc:
        raise CodecError(f"JointState 에 없는 관절: {exc}") from exc
    return pos, vel


def encode_joint_target(names: Sequence[str], q, qd, episode: str, seq: int, stamp: float | None = None):
    """fabric → pd 관절 목표. effort 는 0, frame_id 에 에피소드·seq."""
    msg = encode_joint_state(names, q, velocity=qd, effort=np.zeros(len(names)), stamp=stamp)
    msg.header.frame_id = f"{episode}:{int(seq)}"
    return msg


def decode_joint_target(msg, names: Sequence[str]) -> JointTarget:
    s = decode_joint_state(msg)
    if s.velocity is None:
        raise CodecError("joint_target 에 velocity 가 없다")
    ep, _, seq = str(msg.header.frame_id).rpartition(":")
    if not ep or not seq.isdigit():
        raise CodecError(f"joint_target frame_id {msg.header.frame_id!r} 는 '<episode>:<seq>' 가 아니다")
    pos, vel = select_joints(s, names)
    return JointTarget(names=tuple(names), position=pos, velocity=vel, episode=ep, seq=int(seq), stamp=s.stamp)


# ------------------------------------------------------------------ pose
def encode_pose(pos, quat_wxyz, frame: str, stamp: float | None = None):
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Header

    p = _vec(pos, 3, "pose.position")
    q = _vec(quat_wxyz, 4, "pose.quat")
    msg = PoseStamped(header=Header(stamp=sec_to_stamp(stamp), frame_id=str(frame)))
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in p)
    msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z = (float(v) for v in q)
    return msg


def decode_pose(msg) -> PoseSample:
    p, o = msg.pose.position, msg.pose.orientation
    pos = _vec([p.x, p.y, p.z], 3, "pose.position")
    quat = _vec([o.w, o.x, o.y, o.z], 4, "pose.quat")
    if abs(float(np.linalg.norm(quat)) - 1.0) > 1e-3:
        raise CodecError(f"pose 쿼터니언 노름 {np.linalg.norm(quat):.4f} ≠ 1")
    return PoseSample(pos=pos, quat=quat, frame=str(msg.header.frame_id), stamp=stamp_to_sec(msg.header.stamp))


# ------------------------------------------------------------------ status JSON
def encode_status(status: dict):
    from std_msgs.msg import String

    return String(data=json.dumps(status, ensure_ascii=False))


def decode_status(msg) -> dict:
    try:
        out = json.loads(msg.data)
    except (TypeError, json.JSONDecodeError) as exc:
        raise CodecError(f"status JSON 파싱 실패: {exc}") from exc
    if not isinstance(out, dict):
        raise CodecError("status JSON 은 객체여야 한다")
    return out
