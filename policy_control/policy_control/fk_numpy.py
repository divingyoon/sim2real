"""fk_numpy — obs 조립이 요구하는 바디 자세를 주는 FK 제공자 (덕타입 인터페이스 하나).

    fk.palm_body   : str                          — palm(좌: 그리퍼 base) 바디 이름
    fk.hand_joints : tuple[str, ...]              — `palm_pose` 의 hand_q 가 따르는 canonical 순서
    fk.tip_names   : tuple[str, ...]              — `tips` 행 이름
    fk.palm_pose(arm_q(7,), hand_q(len(hand_joints),)) -> FKPose(palm_pos(3,), palm_quat wxyz(4,),
                                                                tips(N,3), tip_names, extra{"tcp"|"palm_rot"})

세 제공자(계약 `obs.fk.kind`):
  · `left_gripper` — `scripts/left_gripper_fk.LeftGripperFK` 를 감싼다(URDF 기반, Isaac 무의존).
  · `fabric`       — Fabrics FK 는 CUDA 가 필요하고 fabric 노드가 이미 인스턴스를 들고 있으므로
                     **callable 두 개**를 받는 어댑터만 둔다 — `palm_pose_fn(q) -> palm6(pos3+euler_zyx3)`,
                     `tips_fn(q) -> (N,3)`. fabrics_sim 을 여기서 만들지 않는다.
  · `urdf_chain`   — 자산 URDF(`hdgp/assets/robot/<asset>/<asset>.urdf`, canonical 이름)를 numpy 로
                     직접 푼다(`UrdfChainFK`). 계약 `SideCfg`(arm_joints·hand_joints·palm_body·tip_bodies)
                     가 어느 팔인지 정한다. 고정 관절은 접고, 경로 위의 가동 관절이 arm/hand 목록에
                     없으면 **거부**한다(0 으로 조용히 채우지 않는다). mimic 관절은 원본 값으로 푼다.
쿼터니언 규약은 `grasp_s2r_core` 와 같다(회전행렬 → wxyz).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import _paths  # noqa: F401
from .contract import ContractError, DeployContract, SideCfg
from grasp_s2r_core import _quat_from_matrix, _rot_euler_zyx
from left_gripper_fk import BASE_LINK, LeftGripperFK
from robot_control.kinematics import _axis as _urdf_axis
from robot_control.kinematics import _origin as _urdf_origin
from robot_control.kinematics import _rotation as _rodrigues

FK_KINDS = ("left_gripper", "fabric", "urdf_chain")
_MOVABLE = ("revolute", "continuous", "prismatic")


class FKError(ValueError):
    """FK 제공자를 만들 수 없거나 입력 차원이 틀렸다."""


@dataclass(frozen=True)
class FKPose:
    palm_body: str
    palm_pos: np.ndarray
    palm_quat: np.ndarray        # wxyz
    tips: np.ndarray             # (N, 3)
    tip_names: tuple
    extra: dict = field(default_factory=dict)


def _vec(a, n: int, what: str) -> np.ndarray:
    v = np.asarray(a, dtype=np.float64).reshape(-1)
    if v.size != n:
        raise FKError(f"{what}: {v.size}개 — {n}개여야 한다")
    return v


# ------------------------------------------------------------------ left gripper
class LeftFK:
    """좌 스톡 그리퍼: palm = `l_hl_gripper_base`, tips = (left_finger, right_finger), extra tcp."""

    palm_body = BASE_LINK
    hand_joints = ("l_hj_gripper_1", "l_hj_gripper_2")
    tip_names = ("l_hl_gripper_left_finger", "l_hl_gripper_right_finger")

    def __init__(self, urdf_path: Path) -> None:
        p = Path(urdf_path)
        if not p.exists():
            raise FKError(f"URDF 가 없다: {p}")
        self._fk = LeftGripperFK(p)

    def palm_pose(self, arm_q, hand_q) -> FKPose:
        q = _vec(arm_q, 7, "arm_q")
        g = _vec(hand_q, 2, "hand_q(gripper_1, gripper_2)")
        p = self._fk.poses(q, float(g[0]), float(g[1]))
        return FKPose(palm_body=self.palm_body, palm_pos=p.base_pos.copy(), palm_quat=p.base_quat.copy(),
                      tips=np.stack([p.finger_l_pos, p.finger_r_pos]), tip_names=self.tip_names,
                      extra={"tcp": p.tcp_pos.copy()})


# ------------------------------------------------------------------ fabric adapter
class FabricFK:
    """Fabrics FK 어댑터 — callable 주입. hand_q 는 `hand_joints`(fabric 관절 순) 로 받는다."""

    def __init__(self, palm_pose_fn: Callable, tips_fn: Callable, tip_names, palm_body: str = "palm",
                 hand_joints=()) -> None:
        self._palm = palm_pose_fn
        self._tips = tips_fn
        self.tip_names = tuple(tip_names)
        self.palm_body = str(palm_body)
        self.hand_joints = tuple(hand_joints)

    def palm_pose(self, arm_q, hand_q) -> FKPose:
        q = np.concatenate([_vec(arm_q, 7, "arm_q"), np.asarray(hand_q, dtype=np.float64).reshape(-1)])
        if self.hand_joints and q.size != 7 + len(self.hand_joints):
            raise FKError(f"hand_q {q.size - 7}개 — fabric 손 관절은 {len(self.hand_joints)}개")
        palm6 = _vec(self._palm(q), 6, "palm_pose_fn 결과")
        tips = np.asarray(self._tips(q), dtype=np.float64).reshape(-1, 3)
        if tips.shape[0] != len(self.tip_names):
            raise FKError(f"tips_fn 결과 {tips.shape[0]}행 — tip_names 는 {len(self.tip_names)}개")
        R = _rot_euler_zyx(palm6[3:])
        return FKPose(palm_body=self.palm_body, palm_pos=palm6[:3].copy(), palm_quat=_quat_from_matrix(R),
                      tips=tips.copy(), tip_names=self.tip_names, extra={"palm_rot": R})


# ------------------------------------------------------------------ generic URDF tree
@dataclass(frozen=True)
class _Joint:
    """URDF 관절 하나: 부모 프레임 → 자식 프레임 = origin(T0) · motion(axis, q)."""

    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray            # (4, 4) 고정 오프셋
    axis: np.ndarray              # (3,) 가동 관절의 축(고정이면 무시)
    mimic: tuple | None = None    # (원본 관절, multiplier, offset)

    def motion(self, q: float) -> np.ndarray:
        T = np.eye(4)
        if self.type in ("revolute", "continuous"):
            T[:3, :3] = _rodrigues(self.axis, q)
        elif self.type == "prismatic":
            T[:3, 3] = self.axis * q
        return T


def _parse_joint(el: ET.Element) -> _Joint:
    typ = el.get("type", "")
    if typ not in (*_MOVABLE, "fixed"):
        raise FKError(f"joint {el.get('name')!r}: type {typ!r} 는 다루지 않는다 (floating/planar)")
    xyz, R = _urdf_origin(el)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, xyz
    mimic_el = el.find("mimic")
    mimic = None
    if mimic_el is not None:
        mimic = (str(mimic_el.get("joint")), float(mimic_el.get("multiplier", 1.0)),
                 float(mimic_el.get("offset", 0.0)))
    parent, child = el.find("parent"), el.find("child")
    if parent is None or child is None:
        raise FKError(f"joint {el.get('name')!r}: parent/child 가 없다")
    return _Joint(name=str(el.get("name")), type=typ, parent=str(parent.get("link")), child=str(child.get("link")),
                  origin=T, axis=_urdf_axis(el) if typ in _MOVABLE else np.zeros(3), mimic=mimic)


class UrdfTree:
    """URDF 의 링크 트리. `path(link)` 는 루트→링크 관절 열, `transforms(q)` 는 요청 링크의 루트 프레임 자세."""

    def __init__(self, urdf_path: Path) -> None:
        p = Path(urdf_path)
        if not p.exists():
            raise FKError(f"URDF 가 없다: {p}")
        root = ET.parse(p).getroot()
        # 직계 <joint> 만 — <ros2_control> 블록의 동명 <joint> 가 체인을 가리지 않게(kinematics.py 와 같은 이유)
        self.joints = {j.name: j for j in (_parse_joint(el) for el in root.findall("joint"))}
        self.links = {str(el.get("name")) for el in root.findall("link")}
        self._parent_joint = {j.child: j for j in self.joints.values()}
        self.path = p

    def path_to(self, link: str) -> tuple:
        """루트에서 `link` 까지의 관절(고정 포함) 순열."""
        if link not in self.links:
            raise FKError(f"{self.path.name}: 링크 {link!r} 가 없다")
        chain, cur = [], link
        while cur in self._parent_joint:
            j = self._parent_joint[cur]
            chain.append(j)
            cur = j.parent
        return tuple(reversed(chain))

    def movable_on(self, link: str) -> list:
        return [j.name for j in self.path_to(link) if j.type in _MOVABLE]

    def transforms(self, links: Sequence[str], q: dict) -> dict:
        """{링크: 4×4 루트 프레임 자세}. 가동 관절 값은 `q`(이름→값), mimic 은 원본에서 유도."""
        cache: dict = {}
        out = {}
        for link in links:
            T = np.eye(4)
            for j in self.path_to(link):
                if j.child in cache:
                    T = cache[j.child]
                    continue
                T = T @ j.origin @ j.motion(self._value(j, q))
                cache[j.child] = T
            out[link] = T
        return out

    @staticmethod
    def _value(j: _Joint, q: dict) -> float:
        if j.type == "fixed":
            return 0.0
        if j.name in q:
            return float(q[j.name])
        if j.mimic is not None and j.mimic[0] in q:
            src, mult, off = j.mimic
            return mult * float(q[src]) + off
        raise FKError(f"가동 관절 {j.name!r} 의 값이 없다")


# ------------------------------------------------------------------ urdf_chain provider
class UrdfChainFK:
    """자산 URDF 로 한 팔의 palm + 손끝 자세를 푼다 (numpy, Isaac/CUDA 무의존).

    Args:
        urdf_path: 루트 프레임 = 로봇 base(`body_root`, 마운트 플레이트 상면).
        arm_joints: palm 경로 위의 팔 관절 7개(이 순서로 arm_q 를 받는다).
        hand_joints: 손 관절(이 순서로 hand_q 를 받는다). 손끝 경로 위의 가동 관절은 전부 여기 있어야 한다.
        palm_body / tip_bodies: 자세를 낼 링크 이름들.
    """

    def __init__(self, urdf_path: Path, arm_joints: Sequence[str], hand_joints: Sequence[str],
                 palm_body: str, tip_bodies: Sequence[str]) -> None:
        self.tree = UrdfTree(urdf_path)
        self.arm_joints = tuple(str(j) for j in arm_joints)
        self.hand_joints = tuple(str(j) for j in hand_joints)
        self.palm_body = str(palm_body)
        self.tip_names = tuple(str(b) for b in tip_bodies)
        self._bodies = (self.palm_body, *self.tip_names)
        self._check_coverage()

    @classmethod
    def from_side(cls, urdf_path: Path, side: SideCfg) -> "UrdfChainFK":
        return cls(urdf_path, side.arm_joints, side.hand_joints, side.palm_body, side.tip_bodies)

    def _check_coverage(self) -> None:
        known = set(self.arm_joints) | set(self.hand_joints)
        mimic_ok = {n for n, j in self.tree.joints.items() if j.mimic is not None and j.mimic[0] in known}
        unknown = [j for j in self.arm_joints + self.hand_joints if j not in self.tree.joints]
        if unknown:
            raise FKError(f"{self.tree.path.name}: 관절 {unknown} 이 URDF 에 없다")
        palm_path = self.tree.movable_on(self.palm_body)
        if [j for j in palm_path if j in self.arm_joints] != list(self.arm_joints):
            raise FKError(f"palm {self.palm_body!r} 경로의 팔 관절 {palm_path} ≠ arm_joints {list(self.arm_joints)}")
        for body in self._bodies:
            stray = [j for j in self.tree.movable_on(body) if j not in known and j not in mimic_ok]
            if stray:
                raise FKError(f"{body!r} 경로의 가동 관절 {stray} 이 arm/hand 관절에 없다 — 0 으로 채우지 않는다")

    def _q(self, arm_q, hand_q) -> dict:
        a = _vec(arm_q, len(self.arm_joints), "arm_q")
        h = _vec(hand_q, len(self.hand_joints), "hand_q")
        return {**dict(zip(self.arm_joints, a)), **dict(zip(self.hand_joints, h))}

    def body_poses(self, arm_q, hand_q) -> dict:
        """{링크: (pos(3,), R(3,3))} for palm + tips."""
        T = self.tree.transforms(self._bodies, self._q(arm_q, hand_q))
        return {b: (T[b][:3, 3].copy(), T[b][:3, :3].copy()) for b in self._bodies}

    def palm_pose(self, arm_q, hand_q) -> FKPose:
        poses = self.body_poses(arm_q, hand_q)
        pos, R = poses[self.palm_body]
        tips = np.stack([poses[t][0] for t in self.tip_names]) if self.tip_names else np.zeros((0, 3))
        return FKPose(palm_body=self.palm_body, palm_pos=pos, palm_quat=_quat_from_matrix(R), tips=tips,
                      tip_names=self.tip_names, extra={"palm_rot": R})


# ------------------------------------------------------------------ factory
def _side_cfg(contract: DeployContract, side: str | None) -> SideCfg:
    try:
        return contract.side(side or contract.primary_side)
    except ContractError as exc:
        raise FKError(str(exc)) from exc


def make_fk(contract: DeployContract, rl_ws: Path, palm_pose_fn: Callable | None = None,
            tips_fn: Callable | None = None, side: str | None = None, kind: str | None = None):
    """계약 `obs.fk.kind`(또는 `kind` 인자)로 제공자를 고른다. `side` 기본 = 계약 primary_side."""
    kind = kind or contract.obs.fk.get("kind")
    if kind == "left_gripper":
        return LeftFK(Path(rl_ws) / contract.obs.fk["urdf"])
    if kind == "fabric":
        return _fabric_fk(contract, palm_pose_fn, tips_fn, side)
    if kind == "urdf_chain":
        urdf = contract.obs.fk.get("urdf")
        if not urdf:
            raise FKError("fk.kind=urdf_chain 은 obs.fk.urdf(자산 URDF 경로)가 필요하다")
        return UrdfChainFK.from_side(Path(rl_ws) / urdf, _side_cfg(contract, side))
    raise FKError(f"모르는 fk.kind {kind!r} (허용 {FK_KINDS})")


def _fabric_fk(contract: DeployContract, palm_pose_fn, tips_fn, side: str | None) -> FabricFK:
    if palm_pose_fn is None or tips_fn is None:
        raise FKError("fk.kind=fabric 은 palm_pose_fn/tips_fn 두 callable 이 필요하다 (fabric 노드가 준다)")
    s = _side_cfg(contract, side)
    tips = list(s.tip_bodies) or contract.obs.joint_orders.get("tips")
    if not tips:
        raise FKError("계약에 손끝 바디가 없다 (side.tip_bodies / obs.joint_orders.tips)")
    fabric = s.fabric if s.fabric is not None else contract.fabric
    return FabricFK(palm_pose_fn, tips_fn, tip_names=tips, palm_body=s.palm_body,
                    hand_joints=fabric.joint_order[len(s.arm_joints):])
