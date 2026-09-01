#!/usr/bin/env python3
"""head(목) 체인의 FK — `base ← head_camera`(tilt 링크) 변환.

★**URDF 를 그대로 읽는다.** 상수를 손으로 베끼면 자산이 바뀔 때 조용히 어긋난다
(저장소에 FK 가 둘 있고 17 cm 어긋난 전례가 있다).

체인 (openarm_tesollo_sensor_rl.urdf):

    body_root ─(fixed)─ body_link ─(fixed 0,0,0.750)─ head_base
              ─(revolute pan,  axis 0 0 -1)─ head_mid
              ─(revolute tilt, axis 0 1 0)─ head_camera

**`head_camera` 까지만 계산한다.** 그 뒤의 고정 변환(카메라 마운트 + optical 프레임
규약)은 hand-eye 가 `T_neck_cam` 으로 통째로 풀어내므로 여기서 알 필요가 없다 —
알아내려다 규약을 잘못 짚는 것보다 낫다.

각도 단위는 **도(deg)** 이고, 실기 인코더 각(`tick_to_deg`)과 같은 부호로 넣는다.
그 둘이 정말 같은 영점을 쓰는지는 hand-eye 잔차가 말해 준다.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

HEAD_URDF = (Path(__file__).resolve().parents[2]
             / "urdf" / "generated" / "rl" / "openarm_tesollo_sensor_rl.urdf")
#: base 로 삼는 링크. 이 위에서 팔 체인도 시작한다.
BASE_LINK = "body_link"
#: tilt 가 움직이는 링크. 카메라는 여기에 고정돼 있다.
NECK_LINK = "head_camera"


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    xyz: list[float]
    rpy: list[float]
    axis: list[float]


def _floats(text: str | None, default: tuple[float, float, float]) -> list[float]:
    return [float(v) for v in text.split()] if text else list(default)


@lru_cache(maxsize=1)
def head_chain() -> tuple[Joint, ...]:
    """BASE_LINK → NECK_LINK 로 가는 관절들을 URDF 순서대로."""
    root = ET.parse(HEAD_URDF).getroot()
    by_child: dict[str, ET.Element] = {}
    for j in root.findall("joint"):
        by_child[j.find("child").get("link")] = j

    reverse: list[Joint] = []
    link = NECK_LINK
    while link != BASE_LINK:
        if link not in by_child:
            raise RuntimeError(f"URDF 체인이 끊겼다: {link} 의 부모가 없다")
        j = by_child[link]
        origin = j.find("origin")
        axis = j.find("axis")
        reverse.append(Joint(
            name=j.get("name"), kind=j.get("type"),
            xyz=_floats(origin.get("xyz") if origin is not None else None, (0, 0, 0)),
            rpy=_floats(origin.get("rpy") if origin is not None else None, (0, 0, 0)),
            axis=_floats(axis.get("xyz") if axis is not None else None, (0, 0, 1)),
        ))
        link = j.find("parent").get("link")
    return tuple(reversed(reverse))


def _rpy_matrix(rpy: list[float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle_matrix(axis: list[float], angle_rad: float) -> np.ndarray:
    a = np.array(axis, dtype=float)
    norm = np.linalg.norm(a)
    if norm < 1e-12:
        return np.eye(3)
    a = a / norm
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * (K @ K)


def _homogeneous(R: np.ndarray, t: list[float] | np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


#: 인코더 각 → URDF 관절 각. **pan 만 부호가 반대다.**
#:
#: URDF 의 pan 축은 (0,0,-1) 이라 양의 관절각이 인코더의 양의 방향과 반대로 돈다.
#: 2026-09-01 hand-eye 로 판정했다 — 뒤집어야 보드가 테이블 높이(z=+0.23 m)에 놓이고,
#: 안 뒤집으면 z=+1.48 m 로 카메라(z=0.82)보다 66 cm 위에 나온다.
#: tilt 는 실측으로 같은 부호임을 확인했다(dy/dtilt = -0.0106 m/deg, 카메라가 아래를 봄).
PAN_ENCODER_TO_URDF = -1.0
TILT_ENCODER_TO_URDF = +1.0


def urdf_from_encoder(pan_deg: float, tilt_deg: float) -> tuple[float, float]:
    """실기 인코더 각(deg) → URDF 관절 각(deg)."""
    return (PAN_ENCODER_TO_URDF * float(pan_deg),
            TILT_ENCODER_TO_URDF * float(tilt_deg))


def encoder_from_urdf(pan_deg: float, tilt_deg: float) -> tuple[float, float]:
    """URDF 관절 각(deg) → 실기 인코더 각(deg). `urdf_from_encoder` 의 역이다.

    ROS `joint_states` 는 URDF 관절 이름을 쓰므로 **URDF 규약**으로 흐른다. 반면
    캘리브·설정(`head_home.yaml`)은 사람이 읽는 **인코더 각**이다. 그 사이를 건널 때
    이 함수를 쓴다 — 부호를 손으로 뒤집으면 언젠가 반대로 넣는다.
    """
    return (pan_deg / PAN_ENCODER_TO_URDF, tilt_deg / TILT_ENCODER_TO_URDF)


def t_base_neck_from_encoder(pan_deg: float, tilt_deg: float) -> np.ndarray:
    """실기 인코더 각으로 바로 FK. 부호 변환을 잊지 않게 하는 창구다."""
    return t_base_neck(*urdf_from_encoder(pan_deg, tilt_deg))


def t_base_neck(pan_deg: float, tilt_deg: float) -> np.ndarray:
    """`base ← head_camera` 4x4. 관절 각도는 도(deg)."""
    angles = {"head_j_pan": math.radians(pan_deg),
              "head_j_tilt": math.radians(tilt_deg)}
    T = np.eye(4)
    for joint in head_chain():
        T = T @ _homogeneous(_rpy_matrix(joint.rpy), joint.xyz)
        if joint.kind in ("revolute", "continuous"):
            angle = angles.get(joint.name)
            if angle is None:
                raise RuntimeError(f"각도를 모르는 관절: {joint.name}")
            T = T @ _homogeneous(_axis_angle_matrix(joint.axis, angle), (0, 0, 0))
    return T


if __name__ == "__main__":
    print(f"URDF {HEAD_URDF}")
    for j in head_chain():
        print(f"  {j.name:14} {j.kind:9} xyz={j.xyz} rpy={j.rpy} axis={j.axis}")
    for pan, tilt in ((0.0, 0.0), (0.0, -20.0)):
        T = t_base_neck(pan, tilt)
        print(f"\npan={pan:+.1f} tilt={tilt:+.1f} → 위치 "
              f"[{T[0,3]:+.4f}, {T[1,3]:+.4f}, {T[2,3]:+.4f}] m")
