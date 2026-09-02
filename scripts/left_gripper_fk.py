#!/usr/bin/env python3
"""좌 그리퍼 FK — 배포용 순수 numpy (Isaac·torch 무의존).

관측 조립(`left_obs_builder`)과 게이트(`left_grasp_gate`)가 **바디 자세**를 요구하는데
sim 은 물리엔진에서 읽고 실기에는 그런 게 없다. 여기서 URDF 로 만든다.

    팔 7관절 ──(회전)──▶ l_al_7 ──(fixed)──▶ gripper_base ─┬─(fixed)──▶ tcp
                                                          ├─(prismatic)─▶ right_finger
                                                          └─(prismatic)─▶ left_finger

`robot_control.kinematics.Chain` 은 **회전관절만** 다루므로 팔 구간에만 쓰고, 그리퍼
구간(고정 2개 + 프리즈매틱 2개)은 여기서 직접 합성한다. 축이 전부 정렬돼 있고 rpy 가
0 이라 오프셋 합성으로 끝난다(URDF 실측):

    gripper_mount  fixed      xyz (0, 0, 0.1001)
    gripper_tcp    fixed      xyz (0, 0, 0.08)
    gripper_1      prismatic  xyz (0, -0.006, 0.015)  axis (0, -1, 0)   0..0.044
    gripper_2      prismatic  xyz (0, +0.006, 0.015)  axis (0, +1, 0)   0..0.044

★ 손가락은 base 의 y 로만 미끄러진다 — 그래서 게이트의 접근축(base z)은 손가락 자세와
  같다(학습 코드 `_jaw_frame` 의 전제 그대로).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_URDF = Path("/home/user/rl_ws/urdf/generated/rl/openarm_tesollo_sensor_rl.urdf")
ARM_JOINTS = tuple(f"l_aj_{i}" for i in range(1, 8))
BASE_LINK = "l_hl_gripper_base"
GRIPPER_JOINTS = ("l_hj_gripper_mount", "l_hj_gripper_tcp",
                  "l_hj_gripper_1", "l_hj_gripper_2")


@dataclass(frozen=True)
class GripperPoses:
    """world(=robot base) 프레임 바디 자세. 회전은 그리퍼 base 자세를 공유한다."""

    base_pos: np.ndarray
    base_quat: np.ndarray          # (w, x, y, z)
    tcp_pos: np.ndarray
    finger_l_pos: np.ndarray       # l_hj_gripper_2 쪽 (+y)
    finger_r_pos: np.ndarray       # l_hj_gripper_1 쪽 (−y)


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """3×3 회전행렬 → (w, x, y, z). 수치적으로 안정한 분기식."""
    m = np.asarray(R, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                         (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                         (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                         0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                     (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def _read_gripper_geometry(urdf_path: Path) -> dict:
    """URDF 에서 그리퍼 구간 오프셋·축을 읽는다 — 상수를 손으로 옮기지 않는다."""
    root = ET.parse(urdf_path).getroot()
    out: dict = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in GRIPPER_JOINTS:
            continue
        origin = joint.find("origin")
        if origin is None:
            raise SystemExit(f"[left_gripper_fk] {name} 에 <origin> 이 없다")
        xyz = np.array([float(v) for v in (origin.get("xyz") or "0 0 0").split()])
        rpy = np.array([float(v) for v in (origin.get("rpy") or "0 0 0").split()])
        if not np.allclose(rpy, 0.0):
            raise SystemExit(
                f"[left_gripper_fk] {name} 의 rpy 가 0 이 아니다({rpy}) — 이 모듈은 "
                "축 정렬을 전제한다. 회전 오프셋을 합성하도록 고쳐라")
        axis_el = joint.find("axis")
        axis = None
        if axis_el is not None:
            axis_txt = axis_el.get("xyz")
            if axis_txt is None:
                raise SystemExit(f"[left_gripper_fk] {name} 의 <axis> 에 xyz 가 없다")
            axis = np.array([float(v) for v in axis_txt.split()])
        out[name] = {"xyz": xyz, "axis": axis, "type": joint.get("type")}
    missing = [j for j in GRIPPER_JOINTS if j not in out]
    if missing:
        raise SystemExit(f"[left_gripper_fk] URDF 에 없는 관절: {missing}")
    return out


class LeftGripperFK:
    """팔 7관절 + 그리퍼 2관절 → 바디 자세.

    팔 구간은 `robot_control.kinematics` 로 푼다(오늘 중력 모델을 검증한 그 체인이다).
    """

    def __init__(self, urdf_path: Path | str = DEFAULT_URDF,
                 robot_control_src: str | None = None) -> None:
        import sys
        sys.path.insert(0, robot_control_src or "/home/user/rl_ws/robot_control/src")
        from robot_control.kinematics import chain_from_urdf

        self.urdf_path = Path(urdf_path)
        urdf = self.urdf_path.read_text()
        self._chain_base = chain_from_urdf(urdf, list(ARM_JOINTS), BASE_LINK)
        self._geo = _read_gripper_geometry(self.urdf_path)

    def poses(self, arm_q, gripper_q1: float, gripper_q2: float) -> GripperPoses:
        """arm_q(7) + 그리퍼 두 관절(m) → 바디 자세.

        그리퍼 값이 한 개뿐이면(실기 mimic) 같은 값을 두 번 넘기면 된다.
        """
        arm_q = np.asarray(arm_q, dtype=np.float64).reshape(-1)
        if arm_q.size != 7:
            raise ValueError(f"팔 관절은 7개 — 받은 {arm_q.size}")
        T = self._chain_base.pose(arm_q)
        R, p = T[:3, :3], T[:3, 3]
        base_quat = quat_from_matrix(R)

        def local(offset: np.ndarray) -> np.ndarray:
            return p + R @ offset

        g = self._geo
        tcp = local(g["l_hj_gripper_tcp"]["xyz"])
        f_r = local(g["l_hj_gripper_1"]["xyz"]
                    + g["l_hj_gripper_1"]["axis"] * float(gripper_q1))
        f_l = local(g["l_hj_gripper_2"]["xyz"]
                    + g["l_hj_gripper_2"]["axis"] * float(gripper_q2))
        return GripperPoses(base_pos=p, base_quat=base_quat, tcp_pos=tcp,
                            finger_l_pos=f_l, finger_r_pos=f_r)
