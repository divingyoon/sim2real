"""obs_segments — 빌더 레지스트리 `name → fn(inputs, segment) -> ndarray`.

계약(`obs.segments[i].builder`)의 이름 하나가 함수 하나다. 본체는 전부 기존 순수 함수를
**호출**한다(`left_obs_builder`, `grasp_s2r_obs_builder`, `left_grasp_gate`) — 복제하지 않는다.

관절 순열은 **이름으로** 만든다: 상태의 `ee_names`(yaml/드라이버 순) → 계약
`obs.joint_orders[...]`(hand_obs = Isaac DOF 순, hand_profile = finger-major, ee = 그리퍼 둘).
슬라이스로 옮기면 손 40칸이 통째로 스크램블되고 정책은 조용히 이상하게 돈다(148 mm 사고).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import _paths  # noqa: F401
from .contract import DeployContract, ObsSegment
from .fk_numpy import FKPose
from .sources import RobotState
from grasp_s2r_obs_builder import normalized_joint_err, reorder, rot6d_columns, tip_force_local
from left_obs_builder import cup_upright, normalize_tcp, rot6d_from_quat


class ObsError(ValueError):
    """관측을 만들 수 없다(차원·이름·NaN·모르는 빌더)."""


class ObsBuildError(ObsError):
    """빌더 하나가 실패했다."""


@dataclass(frozen=True)
class ObsInputs:
    """한 tick 의 빌더 입력 — 에피소드 상태를 obs_core 가 채워 넘긴다."""

    contract: DeployContract
    state: RobotState
    fk: FKPose
    object_pos: np.ndarray          # 물체 모드 적용 후
    object_quat: np.ndarray
    goal: np.ndarray                # 좌 7(pos+quat) / 우 3
    last_action: np.ndarray
    gate: float | None              # 좌 그리퍼 게이트(0/1), 우 None
    decoder_target: np.ndarray | None   # hand_joints 순 직전 손 목표
    #: 이 팔(계약 SideCfg)의 palm 바디와 손 관절 순서. 비우면 primary side(계약 최상위) 것으로 본다.
    palm_body: str = ""
    hand_joints: tuple = ()


def _hand_joints(inp: ObsInputs) -> list[str]:
    if inp.hand_joints:
        return list(inp.hand_joints)
    if inp.contract.action.hand is None:
        raise ObsBuildError("계약에 손 관절이 없다 (action.hand / side.hand)")
    return list(inp.contract.action.hand.joints)


Builder = Callable[[ObsInputs, ObsSegment], np.ndarray]
BUILDERS: dict[str, Builder] = {}


def register(name: str) -> Callable[[Builder], Builder]:
    def deco(fn: Builder) -> Builder:
        BUILDERS[name] = fn
        return fn
    return deco


def builder(name: str) -> Builder:
    fn = BUILDERS.get(name)
    if fn is None:
        raise ObsBuildError(f"모르는 obs 빌더 {name!r} (등록: {sorted(BUILDERS)})")
    return fn


# ------------------------------------------------------------------ 이름 순열
def _order(inp: ObsInputs, key: str) -> list[str]:
    order = inp.contract.obs.joint_orders.get(key)
    if not order:
        raise ObsBuildError(f"계약 obs.joint_orders[{key!r}] 가 없다")
    return list(order)


def _ee_in(inp: ObsInputs, names, what: str) -> np.ndarray:
    st = inp.state
    try:
        return reorder(st.ee_q if what == "q" else st.ee_qd, list(st.ee_names), list(names))
    except (KeyError, ValueError) as exc:
        raise ObsBuildError(f"ee {what}: {exc}") from exc


def _arm(inp: ObsInputs, what: str) -> np.ndarray:
    a = inp.state.arm_q if what == "q" else inp.state.arm_qd
    n = len(_order(inp, "arm"))
    if a is None or a.size != n:
        raise ObsBuildError(f"arm {what}: {None if a is None else a.size} vs {n}")
    return np.asarray(a, dtype=np.float64)


def _tips_in_contract_order(inp: ObsInputs) -> np.ndarray:
    want = _order(inp, "tips")
    have = list(inp.fk.tip_names)
    try:
        idx = [have.index(n) for n in want]
    except ValueError as exc:
        raise ObsBuildError(f"FK tip 이름 {have} 에 없는 {exc}") from exc
    return inp.fk.tips[idx]


# ------------------------------------------------------------------ 좌(gripper_left) 빌더
def _joint_rel(inp: ObsInputs, seg: ObsSegment, what: str) -> np.ndarray:
    q = np.concatenate([_arm(inp, what), _ee_in(inp, _order(inp, "ee"), what)])
    default = np.asarray(seg.params["default"], dtype=np.float64).reshape(-1)
    if default.size != q.size:
        raise ObsBuildError(f"{seg.name}: default {default.size} vs 관절 {q.size}")
    return q - default


@register("joint_pos_rel")
def joint_pos_rel(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    return _joint_rel(inp, seg, "q")


@register("joint_vel_rel")
def joint_vel_rel(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    return _joint_rel(inp, seg, "qd")


@register("object_pos_root")
def object_pos_root(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    return np.asarray(inp.object_pos, dtype=np.float64).reshape(3).copy()   # root = base_link 원점


@register("goal_pose")
def goal_pose(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    return np.asarray(inp.goal, dtype=np.float64).reshape(-1).copy()


@register("last_action")
def last_action(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    return np.asarray(inp.last_action, dtype=np.float64).reshape(-1).copy()


@register("gripper_gate")
def gripper_gate(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    if inp.gate is None:
        raise ObsBuildError("gripper_gate: 게이트 값이 없다 (obs_core 가 GraspGate 를 갱신해야 한다)")
    return np.array([float(inp.gate)])


@register("tcp_pos_normalized")
def tcp_pos_normalized(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    tcp = inp.fk.extra.get("tcp")
    if tcp is None:
        raise ObsBuildError("tcp_pos_normalized: FK 가 tcp 를 주지 않는다")
    return normalize_tcp(tcp, seg.params["palm_box"])


@register("rot6d_rows")
def rot6d_rows(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    _check_body(inp, seg)
    return rot6d_from_quat(inp.fk.palm_quat)          # 행우선 인터리브 (좌 규약)


@register("goal_minus_object")
def goal_minus_object(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    return np.asarray(inp.goal, dtype=np.float64).reshape(-1)[:3] - np.asarray(inp.object_pos).reshape(3)


@register("object_upright")
def object_upright(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    return np.array([cup_upright(inp.object_quat)])


# ------------------------------------------------------------------ 우(grasp_s2r) 빌더
def _joint_abs(inp: ObsInputs, seg: ObsSegment, what: str) -> np.ndarray:
    key = seg.params.get("order")
    if key == "arm":
        return _arm(inp, what).copy()
    return _ee_in(inp, _order(inp, key), what)


@register("joint_pos_abs")
def joint_pos_abs(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    return _joint_abs(inp, seg, "q")


@register("joint_vel_abs")
def joint_vel_abs(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    return _joint_abs(inp, seg, "qd")


def _check_body(inp: ObsInputs, seg: ObsSegment) -> None:
    """FK 가 이 팔의 palm 바디를 주는지 확인한다.

    기준은 계약 `SideCfg.palm_body`(자산 manifest 의 링크). 세그먼트의 `params.body` 는 **런 dump 의
    이름**이라, 자산에 재기반한 계약(`contract.asset` 있음)에서는 옛 자산의 이름(`palm`)이 남아 있어도
    된다 — 그때는 side 의 바디만 본다. 자산 바인딩이 없는 v1 계약은 둘이 같아야 한다."""
    want = inp.palm_body or inp.fk.palm_body
    if inp.fk.palm_body != want:
        raise ObsBuildError(f"{seg.name}: side palm_body {want!r} ≠ FK palm_body {inp.fk.palm_body!r}")
    body = seg.params.get("body")
    if body and inp.contract.asset is None and body != want:
        raise ObsBuildError(f"{seg.name}: 계약 body {body!r} ≠ FK palm_body {want!r}")


@register("body_pos")
def body_pos(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    _check_body(inp, seg)
    return inp.fk.palm_pos.copy()


@register("rot6d_columns")
def rot6d_cols(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    _check_body(inp, seg)
    return rot6d_columns(inp.fk.palm_quat)            # 열 스택 (우 규약)


@register("tips_rel_palm")
def tips_rel_palm(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    return (_tips_in_contract_order(inp) - inp.fk.palm_pos).reshape(-1)


@register("palm_to_object")
def palm_to_object(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    return np.asarray(inp.object_pos).reshape(3) - inp.fk.palm_pos


@register("object_to_tips")
def object_to_tips(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    return (_tips_in_contract_order(inp) - np.asarray(inp.object_pos).reshape(3)).reshape(-1)


@register("tip_force_local")
def tip_force_local_seg(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    """실기 `tip_forces_xyz` 는 이미 팁 로컬 → 단위 쿼터니언으로 항등 변환(S2RSensors 규약)."""
    st = inp.state
    if st.tip_force is None:
        raise ObsBuildError("tip_force_local: 손끝 힘이 없다")
    want = _order(inp, "tips")
    have = list(st.tip_names)
    try:
        f = st.tip_force[[have.index(n) for n in want]]
    except ValueError as exc:
        raise ObsBuildError(f"tip_force 이름 {have} 에 없는 {exc}") from exc
    ident = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(want), 1))
    return tip_force_local(f, ident, float(seg.params["contact_force_max"])).reshape(-1)


@register("joint_err_norm")
def joint_err_norm(inp: ObsInputs, seg: ObsSegment) -> np.ndarray:
    """(직전 손 목표 − 실측)/상한, ±1 클램프. 둘 다 `params.order` 순으로 맞춘 뒤 뺀다."""
    order = _order(inp, seg.params["order"])
    hand_joints = _hand_joints(inp)
    if inp.decoder_target is None:
        raise ObsBuildError("joint_err_norm: 손 목표가 없다 (obs_core 가 open pose 로 채워야 한다)")
    tgt = np.asarray(inp.decoder_target, dtype=np.float64).reshape(-1)
    if tgt.size != len(hand_joints):
        raise ObsBuildError(f"joint_err_norm: 목표 {tgt.size}개 vs 손 관절 {len(hand_joints)}개")
    measured = _ee_in(inp, order, "q")
    target = reorder(tgt, hand_joints, order)
    return normalized_joint_err(measured, target, float(seg.params["joint_pos_err_max"]))
