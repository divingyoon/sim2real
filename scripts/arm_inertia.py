#!/usr/bin/env python3
"""URDF 에서 팔 관절의 **유효 관성**을 계산한다 (numpy only).

왜 필요한가: 실기 팔의 추종 능력을 오프라인에서 재현하려면 PD 모델
`tau = kp(q_des − q) + kd(qd_des − qd)` 를 적분해야 하고, 그러려면 관절별 관성이 있어야
한다. 값을 지어내면 실험 결론이 통째로 무의미해지므로 **자산 URDF 에서 계산**한다.

방법(표준 근사): 관절 j 의 유효 관성 =
    Σ_{distal link L} [ aᵀ I_L^world a  +  m_L · d⊥²  ]
  a  = 관절 축(월드), d⊥ = 링크 COM 에서 관절 축까지의 수직거리.
회전 관절의 순간 관성이며, 자세에 따라 변한다 → 계산 자세를 함께 보고한다.

한계(정직하게): 이건 대각 근사다. 관절 간 결합항(off-diagonal)과 코리올리·중력은
무시한다. 접근 국면의 저속 운동에서는 대각항이 지배적이라 추종 대역폭 비교 용도로는
충분하지만, 절대 토크 예측용으로 쓰면 안 된다.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_rot(axis: np.ndarray, theta: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K


def parse_urdf(path: str | Path) -> dict:
    """URDF → {joints, links} 최소 표현."""
    root = ET.parse(str(path)).getroot()
    joints, links = {}, {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0").split())]) if o is not None else np.zeros(3)
        rpy = [float(v) for v in (o.get("rpy", "0 0 0").split())] if o is not None else [0.0, 0.0, 0.0]
        ax = j.find("axis")
        joints[j.get("name")] = dict(
            type=j.get("type"),
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
            xyz=xyz, R=_rpy(*rpy),
            axis=np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None
            else np.array([0.0, 0.0, 1.0]),
        )
    for l in root.findall("link"):
        inert = l.find("inertial")
        if inert is None:
            links[l.get("name")] = None
            continue
        o = inert.find("origin")
        c = np.array([float(v) for v in (o.get("xyz", "0 0 0").split())]) if o is not None else np.zeros(3)
        rpy = [float(v) for v in (o.get("rpy", "0 0 0").split())] if o is not None else [0.0, 0.0, 0.0]
        i = inert.find("inertia")
        I = np.array([
            [float(i.get("ixx")), float(i.get("ixy", 0)), float(i.get("ixz", 0))],
            [float(i.get("ixy", 0)), float(i.get("iyy")), float(i.get("iyz", 0))],
            [float(i.get("ixz", 0)), float(i.get("iyz", 0)), float(i.get("izz"))],
        ])
        links[l.get("name")] = dict(
            mass=float(inert.find("mass").get("value")), com=c, R_com=_rpy(*rpy), I=I
        )
    return dict(joints=joints, links=links)


def _link_transforms(model: dict, q: dict[str, float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """모든 링크의 (R, p) — base 기준. 부모→자식 순회."""
    joints, links = model["joints"], model["links"]
    children: dict[str, list[str]] = {}
    for name, j in joints.items():
        children.setdefault(j["parent"], []).append(name)
    all_children = {j["child"] for j in joints.values()}
    roots = [l for l in links if l not in all_children]

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    stack = [(r, np.eye(3), np.zeros(3)) for r in roots]
    while stack:
        link, R, p = stack.pop()
        out[link] = (R, p)
        for jn in children.get(link, []):
            j = joints[jn]
            Rj = R @ j["R"]
            pj = p + R @ j["xyz"]
            if j["type"] in ("revolute", "continuous"):
                Rj = Rj @ _axis_rot(j["axis"], float(q.get(jn, 0.0)))
            stack.append((j["child"], Rj, pj))
    return out


def _subtree_links(model: dict, joint_name: str) -> list[str]:
    joints = model["joints"]
    children: dict[str, list[str]] = {}
    for name, j in joints.items():
        children.setdefault(j["parent"], []).append(name)
    out, stack = [], [joints[joint_name]["child"]]
    while stack:
        link = stack.pop()
        out.append(link)
        for jn in children.get(link, []):
            stack.append(joints[jn]["child"])
    return out


def effective_inertia(
    urdf_path: str | Path, joint_names: list[str], q: dict[str, float] | None = None
) -> np.ndarray:
    """관절별 유효 관성 [kg·m²] (지정 자세에서의 순간값)."""
    model = parse_urdf(urdf_path)
    q = q or {}
    missing = [j for j in joint_names if j not in model["joints"]]
    if missing:
        raise KeyError(f"URDF 에 없는 관절: {missing}")

    tf = _link_transforms(model, q)
    joints, links = model["joints"], model["links"]
    out = np.zeros(len(joint_names))
    for k, jn in enumerate(joint_names):
        j = joints[jn]
        R_par, p_par = tf[j["parent"]]
        R_j = R_par @ j["R"]
        p_j = p_par + R_par @ j["xyz"]
        axis_w = R_j @ (j["axis"] / np.linalg.norm(j["axis"]))
        total = 0.0
        for ln in _subtree_links(model, jn):
            info = links.get(ln)
            if info is None:
                continue
            R_l, p_l = tf[ln]
            com_w = p_l + R_l @ info["com"]
            R_I = R_l @ info["R_com"]
            I_w = R_I @ info["I"] @ R_I.T
            r = com_w - p_j
            d_perp = r - np.dot(r, axis_w) * axis_w
            total += float(axis_w @ I_w @ axis_w) + info["mass"] * float(d_perp @ d_perp)
        out[k] = total
    return out
