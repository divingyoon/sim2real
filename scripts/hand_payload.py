#!/usr/bin/env python3
"""손 전체를 **손목 끝단에 매달린 하나의 페이로드**로 환산한다 (질량 + COM).

왜 필요한가. `robot_control.kinematics._lumped()` 는 가동 조인트 **뒤**의 링크를
중력 모델에서 제외한다 — 그 위치가 조인트 값에 달렸는데 팔 체인은 그 값을 모르기
때문이다(설계상 의도, 버그 아님). 그런데 테솔로 손은 20관절이 전부 가동이라
**손가락 0.835 kg 이 통째로 빠진다**. 남는 것은 adapter+base+palm 0.85 kg 뿐이고,
손목이 실제로 느끼는 모멘트의 **1/3.5** 만 보상하게 된다.
(08.31 실측 대조: 07.29 캘리브가 기록한 "j6 약 3.4배 과소"와 3.46배가 일치.)

이 도구는 **손 관절각을 알고 있는 쪽**(우리)이 그 값을 계산해 넘겨주기 위한 것이다.
손 자세가 정해지면 손가락 위치도 정해지므로 질량·COM 은 확정된다.

    # 실기 주먹 스냅샷 자세의 페이로드
    python3 hand_payload.py --pose config/right_hand_fist.yaml

    # 정책 시작 자세(손 폄)
    python3 hand_payload.py --pose open

    # robotctl 에 그대로 넘길 형식
    python3 hand_payload.py --pose open --format arg
        → --payload 1.685,0.0021,-0.0009,0.1214

출력 COM 은 **팔 끝단 링크(`--tip`, 기본 r_al_7)의 프레임** 기준이다 — 중력 체인이
페이로드를 얹는 자리가 거기이기 때문이다.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
DEFAULT_URDF = Path("/home/user/rl_ws/urdf/generated/rl/openarm_tesollo_sensor_rl.urdf")
#: 손가락 관절 순서 규약(sim canonical) — 드라이버 이름 rj_dg_<f>_<j> 와 1:1.
_FINGER_OF_INDEX = {1: "thumb", 2: "index", 3: "middle", 4: "ring", 5: "pinky"}
#: grasp_s2r init 손자세(엄지 대향). 나머지 관절 0.
_OPEN_POSE = {"r_hj_thumb_2": -1.57, "r_hj_thumb_3": -0.5}


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def _origin(element: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    node = element.find("origin")
    xyz = np.zeros(3)
    rpy = np.zeros(3)
    if node is not None:
        if node.get("xyz"):
            xyz = np.array([float(v) for v in node.get("xyz").split()])
        if node.get("rpy"):
            rpy = np.array([float(v) for v in node.get("rpy").split()])
    return xyz, _rpy_matrix(rpy)


def load_pose(spec: str | Path) -> dict[str, float]:
    """`open` / `fist`(=zero) / yaml 경로 → sim canonical 관절각 dict."""
    if str(spec) == "open":
        return dict(_OPEN_POSE)
    if str(spec) == "zero":
        return {}
    path = Path(spec)
    if not path.is_file():
        raise SystemExit(f"손 자세를 못 읽었다: {spec} (open|zero|<yaml 경로>)")
    out: dict[str, float] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#")[0].strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        try:
            val = float(value)
        except ValueError:
            continue
        if key.startswith("rj_dg_"):
            # 드라이버 이름 → sim canonical. 부호는 동일(sim2real 프로필 실측).
            finger, joint = key.split("_")[2:4]
            out[f"r_hj_{_FINGER_OF_INDEX[int(finger)]}_{joint}"] = val
        elif key.startswith("r_hj_"):
            out[key] = val
    if not out:
        raise SystemExit(f"{path} 에서 손 관절을 하나도 못 읽었다")
    return out


def payload_from_urdf(urdf_path: Path, pose: dict[str, float], tip: str,
                      prefix: str, *, fixed_only: bool = False,
                      include_tip: bool = False) -> tuple[float, np.ndarray, int]:
    """(질량, tip 프레임 기준 COM, 링크 수).

    `fixed_only=True` 면 **고정 조인트만** 따라간다 — 이것이 robot_control 의
    중력 체인(`_lumped`)이 실제로 세는 몫이다. 둘의 차이가 과소 보상분이다.
    `include_tip=True` 면 tip 링크 자신의 질량도 포함한다(체인 값과 직접 대조용).
    """
    root = ET.parse(urdf_path).getroot()
    links = {el.get("name"): el for el in root.findall("link")}
    children: dict[str, list[ET.Element]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        if parent is not None:
            children.setdefault(parent.get("link"), []).append(joint)

    total_mass = 0.0
    moment = np.zeros(3)
    count = 0
    # tip 자신은 팔 링크이므로 제외하고, 그 아래로 내려간 것만 센다.
    stack: list[tuple[str, np.ndarray]] = [(tip, np.eye(4))]
    while stack:
        name, transform = stack.pop()
        if (include_tip and name == tip) or (name != tip and name.startswith(prefix)):
            element = links.get(name)
            inertial = element.find("inertial") if element is not None else None
            if inertial is not None and inertial.find("mass") is not None:
                mass = float(inertial.find("mass").get("value"))
                com_local, _ = _origin(inertial)
                com = transform[:3, :3] @ com_local + transform[:3, 3]
                total_mass += mass
                moment += mass * com
                count += 1
        for joint in children.get(name, ()):
            movable = joint.get("type") not in ("fixed", None)
            if fixed_only and movable:
                continue
            xyz, rot = _origin(joint)
            step = _homogeneous(rot, xyz)
            if movable:
                axis_node = joint.find("axis")
                axis = (np.array([float(v) for v in axis_node.get("xyz").split()])
                        if axis_node is not None else np.array([1.0, 0.0, 0.0]))
                angle = float(pose.get(joint.get("name"), 0.0))
                step = step @ _homogeneous(_axis_angle(axis, angle), np.zeros(3))
            stack.append((joint.find("child").get("link"), transform @ step))
    if total_mass <= 0.0:
        raise SystemExit(f"{tip} 아래에서 질량을 못 찾았다 — --prefix/--tip 확인")
    return total_mass, moment / total_mass, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--pose", default="open",
                        help="open | zero | 손 자세 yaml 경로(rj_dg_* 또는 r_hj_*)")
    parser.add_argument("--tip", default="r_al_7", help="페이로드를 얹을 팔 끝단 링크")
    parser.add_argument("--prefix", default="r_hl_", help="페이로드로 셀 링크 접두사")
    parser.add_argument("--format", choices=["human", "arg", "json"], default="human")
    args = parser.parse_args()

    pose = load_pose(args.pose)
    mass, com, count = payload_from_urdf(args.urdf, pose, args.tip, args.prefix)

    if args.format in ("arg", "json"):
        seen_m, seen_com, _ = payload_from_urdf(
            args.urdf, pose, args.tip, args.prefix, fixed_only=True)
        miss_m, miss_com = _difference(mass, com, seen_m, seen_com)
    if args.format == "arg":
        print(f"--payload {miss_m:.4f},{miss_com[0]:.5f},"
              f"{miss_com[1]:.5f},{miss_com[2]:.5f}")
        return 0
    if args.format == "json":
        import json
        print(json.dumps({"total_mass": mass, "total_com": com.tolist(),
                          "payload_mass": miss_m, "payload_com": miss_com.tolist(),
                          "links": count, "tip": args.tip, "pose": args.pose}))
        return 0

    seen_m, seen_com, seen_n = payload_from_urdf(
        args.urdf, pose, args.tip, args.prefix, fixed_only=True)
    miss_m, miss_com = _difference(mass, com, seen_m, seen_com)

    print(f"URDF  {args.urdf}")
    print(f"자세  {args.pose}  (지정된 관절 {len(pose)}개, 나머지 0)")
    print(f"프레임 {args.tip}\n")
    rows = [("손 전체 (진값)", mass, com, count),
            ("중력체인이 세는 몫 (fixed만)", seen_m, seen_com, seen_n),
            ("★빠진 몫 = --payload", miss_m, miss_com, count - seen_n)]
    print(f"{'':30s} {'질량[kg]':>9s} {'COM z[m]':>9s} {'모멘트[kg·m]':>12s} {'링크':>5s}")
    for label, m, c, n in rows:
        print(f"{label:30s} {m:9.4f} {c[2]:9.5f} "
              f"{m*float(np.linalg.norm(c)):12.5f} {n:5d}")
    ratio = (mass * float(np.linalg.norm(com))) / max(
        seen_m * float(np.linalg.norm(seen_com)), 1e-12)
    print(f"\n과소 배율 {ratio:.2f}배 — 손목 중력보상이 필요량의 1/{ratio:.1f} 만 나온다.")
    print(f"넘길 인자:  --payload {miss_m:.4f},{miss_com[0]:.5f},"
          f"{miss_com[1]:.5f},{miss_com[2]:.5f}")
    return 0


def _difference(total_m: float, total_com: np.ndarray,
                seen_m: float, seen_com: np.ndarray) -> tuple[float, np.ndarray]:
    """전체에서 이미 센 몫을 뺀 나머지 (질량, COM). 모멘트 보존으로 뺀다."""
    rest_m = total_m - seen_m
    if rest_m <= 1e-9:
        return 0.0, np.zeros(3)
    return rest_m, (total_m * total_com - seen_m * seen_com) / rest_m


if __name__ == "__main__":
    raise SystemExit(main())
