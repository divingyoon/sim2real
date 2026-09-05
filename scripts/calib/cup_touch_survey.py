#!/usr/bin/env python3
"""좌 그리퍼로 컵 옆면을 여러 점 짚어 **컵 축(x, y)과 반경**을 재고 인식값과 대조한다.

## 왜 필요한가 (09.03)

우팔 s2r 이 컵을 못 잡는다. 팔은 시킨 자리로 정확히 간다(추종오차 1.6°, sim 1.34°)
`joint_err` 은 0.048 에 머문다(sim 접촉 0.248) — **손가락이 한 번도 안 막힌다**.
그리고 정책을 안 쓰는 **궤적 재생**(sim 성공 관절 지령 그대로)도 똑같이 빗나갔다.
정책·obs·시너지를 다 우회했는데 빗나갔으면 남는 건 **기하** 하나다.

★그런데 카메라의 **평면 안 x·y** 는 한 번도 검증한 적이 없다. 보드 평면 측량
(`board_plane_survey.py`)이 잡아준 것은 **회전과 z** 뿐이고, 그 스크립트가
"평면 안에서의 x·y 이동과 yaw 는 로봇이 아는 점을 짚어야 한다"고 적어 두었다.
이 스크립트가 그 미검증 축을 잰다.

## 방법

컵을 세워 두고 **옆면 3점 이상**을 그리퍼 TCP 로 짚는다. TCP 는 두 턱 사이의 점이라
**닫고** 짚으면 탐침 끝이 명확하다. 짚은 점들은 전부 컵 원통면 위에 있으므로
원을 적합하면 **축 중심과 반경**이 나온다.

  ✓ 컵 축의 x·y 와 반경 — 인식값과 직접 비교
  ✗ 컵의 z(높이)는 이 방법으로 안 나온다(원통은 z 로 대칭) — 그건 이미 쟀다

★★점은 **골고루 벌려서** 잡을 것. 한쪽에 몰리면 원 적합이 불안정하다(잔차는
  작은데 중심이 크게 틀리는 전형적 함정). 90° 이상 벌어진 3점이면 충분하다.

사용:
    # 그리퍼를 컵 옆면에 대고
    python3 cup_touch_survey.py add --tag p1
    # 컵 반대편으로 옮겨 대고
    python3 cup_touch_survey.py add --tag p2
    python3 cup_touch_survey.py add --tag p3
    python3 cup_touch_survey.py solve
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


SIM2REAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM2REAL / "scripts"))
DEFAULT_STORE = SIM2REAL / "logs" / "cup_touch"


def _read_tcp(timeout: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """현재 좌팔 실측으로 FK 한 TCP 위치와 관절값."""
    import rclpy

    from left_gripper_fk import LeftGripperFK
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from robot_profile import load_robot_profile
    from sensor_msgs.msg import JointState

    prof = load_robot_profile("gripper_left")
    rclpy.init()
    node = Node("cup_touch_survey")
    box: dict = {}

    def on_js(m):
        idx = {n: i for i, n in enumerate(m.name)}
        if not all(s in idx for s in prof.arm_source):
            return
        box["q"] = np.array([m.position[idx[s]] for s in prof.arm_source])
        g = idx.get("openarm_left_finger_joint1")
        box["g"] = float(m.position[g]) if g is not None else 0.0

    node.create_subscription(JointState, "/joint_states", on_js,
                             qos_profile_sensor_data)
    t0 = time.time()
    while time.time() - t0 < timeout and "q" not in box:
        rclpy.spin_once(node, timeout_sec=0.2)
    rclpy.shutdown()
    if "q" not in box:
        raise SystemExit("[touch] /joint_states 수신 없음 — 좌팔 bringup 확인")
    fk = LeftGripperFK()
    return fk.poses(box["q"], box["g"], box["g"]).tcp_pos, box["q"]


def _read_perception(topic: str, timeout: float = 8.0, n: int = 20):
    """인식 컵 위치의 중앙값. 못 받으면 None(대조는 건너뛴다)."""
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    rclpy.init()
    node = Node("cup_touch_perc")
    got: list = []
    node.create_subscription(
        PoseStamped, topic,
        lambda m: got.append((m.pose.position.x, m.pose.position.y,
                              m.pose.position.z)),
        qos_profile_sensor_data)
    t0 = time.time()
    while time.time() - t0 < timeout and len(got) < n:
        rclpy.spin_once(node, timeout_sec=0.05)
    rclpy.shutdown()
    if not got:
        return None
    return np.median(np.array(got), axis=0)


def fit_circle(pts_xy: np.ndarray) -> tuple[np.ndarray, float, float]:
    """xy 점들에 원을 적합해 (중심, 반경, 잔차 RMS)를 낸다.

    대수적 적합(Kåsa): |p|² = 2·c·p + (r² − |c|²) 를 최소제곱으로 푼다.
    """
    x, y = pts_xy[:, 0], pts_xy[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    r = float(np.sqrt(sol[2] + c @ c))
    resid = np.linalg.norm(pts_xy - c, axis=1) - r
    return c, r, float(np.sqrt((resid**2).mean()))


def add(store: Path, tag: str) -> int:
    store.mkdir(parents=True, exist_ok=True)
    tcp, q = _read_tcp()
    (store / f"{tag}.json").write_text(json.dumps(
        {"tcp": tcp.tolist(), "q": q.tolist()}))
    print(f"[touch] {tag}: TCP {np.round(tcp, 4).tolist()}")
    return 0


def solve(store: Path, topic: str, gripper_radius: float) -> int:
    files = sorted(store.glob("*.json"))
    if len(files) < 3:
        print(f"[touch] 점이 {len(files)}개뿐이다 — 컵 둘레로 **골고루** 3점 이상",
              file=sys.stderr)
        return 1
    pts = np.array([json.loads(f.read_text())["tcp"] for f in files])
    c, r, rms = fit_circle(pts[:, :2])

    # ★TCP 가 컵 표면에 닿았으므로 적합 반경 = 컵 반경 + 탐침 반경.
    cup_r = r - gripper_radius
    print(f"\n[touch] 점 {len(files)}개")
    for f, p in zip(files, pts):
        d = np.linalg.norm(p[:2] - c) - r
        print(f"  {f.stem:8s} TCP {np.round(p, 4).tolist()} · 원에서 {d*1000:+6.2f} mm")
    ang = np.degrees(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    spread = float(np.ptp(np.sort(ang)))
    print(f"\n  적합 중심 (x, y) = ({c[0]:.4f}, {c[1]:.4f})")
    print(f"  적합 반경 {r*1000:.1f} mm → 컵 반경 {cup_r*1000:.1f} mm "
          f"(탐침 {gripper_radius*1000:.0f} mm 뺌)")
    print(f"  잔차 RMS {rms*1000:.2f} mm · 각도 분포 {spread:.0f}°"
          f"{'   ★점이 한쪽에 몰렸다 — 중심이 못 미더움' if spread < 90 else ''}")

    perc = _read_perception(topic)
    if perc is None:
        print(f"\n  ⚠{topic} 수신 없음 — 인식 대조 건너뜀")
        return 0
    d = (perc[:2] - c) * 1000
    print(f"\n  인식 컵 (x, y) = ({perc[0]:.4f}, {perc[1]:.4f})  z {perc[2]:.4f}")
    print(f"  ★★인식 − 실측 = ({d[0]:+.1f}, {d[1]:+.1f}) mm · "
          f"거리 {np.hypot(*d):.1f} mm")
    if np.hypot(*d) > 20.0:
        print("     이만큼 어긋나 있으면 손이 컵을 스치고 지나간다 — "
              "우팔이 못 잡는 이유로 충분하다.")
    else:
        print("     인식 x·y 는 맞다 — 원인을 다른 데서 찾아야 한다.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="지금 그리퍼가 닿아 있는 점을 기록")
    a.add_argument("--tag", required=True)
    a.add_argument("--store", type=Path, default=DEFAULT_STORE)
    s = sub.add_parser("solve", help="모은 점으로 컵 축을 적합하고 인식과 대조")
    s.add_argument("--store", type=Path, default=DEFAULT_STORE)
    s.add_argument("--topic", default="/objects/cup_big_s100/pose")
    s.add_argument("--gripper-radius", type=float, default=0.0,
                   help="탐침(닫은 턱) 끝의 유효 반경[m]. 점 접촉이면 0")
    r = sub.add_parser("reset", help="모은 점을 비운다")
    r.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = ap.parse_args()

    if args.cmd == "add":
        return add(args.store, args.tag)
    if args.cmd == "reset":
        for f in args.store.glob("*.json"):
            f.unlink()
        print("[touch] 비웠다")
        return 0
    return solve(args.store, args.topic, args.gripper_radius)


if __name__ == "__main__":
    raise SystemExit(main())
