#!/usr/bin/env python3
"""perception 이 준 컵 pose 를 붙잡아 파일로 남기고, sim 에 넣어도 되는지 판정한다.

왜 파일인가. 라이브 경로는 **vision-3090 ↔ 로봇 PC 유선 링크**로 설계돼 있고(런북
§네트워크), Isaac 이 도는 5090 은 그 경로에 있던 적이 없다. WiFi 로 DDS 를 붙여 보려
했으나 유니캐스트 피어를 지정해도 discovery 가 안 붙었다(08.25 실측: 발행 221건 수신 0,
ping 은 정상). 게다가 **파지 한 에피소드 동안 컵은 정지해 있다** — 흘려보낼 이유가 없다.
그래서 인지 결과를 한 번 붙잡아 파일로 옮기고, 그 파일이 곧 sim 의 컵 자리다.
카메라가 붙으면 이 노드를 vision-3090 에서 돌리고 파일만 가져오면 된다.

★가장 중요한 일은 **분포 밖을 말해 주는 것**이다. 정책은 좁은 스폰 상자에서만 학습됐다
  (preset 이 정의한다). 인지가 그 밖의 컵을 주면 정책은 분포 밖에서 도는 것이고, 그
  결과를 "정책이 못한다"로 읽으면 틀린 결론이 된다. 조용히 넣지 않는다.

붙잡기 (ROS 가 /cup_pose 를 보는 곳에서):
    python3 scripts/calib/cup_pose_capture.py --out logs/shadow/cup_pose.json
판정만 (ROS 불필요):
    python3 scripts/calib/cup_pose_capture.py --check logs/shadow/cup_pose.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path

HDGP_PRESET = (Path.home() / "rl_ws/hdgp/source/openarm/openarm/gripper/left"
               / "grasp_sensor/grasp_left_preset.py")

#: 컵 높이 허용 오차[m]. 테이블 위에 놓인 컵의 원점 높이는 preset 이 정한다 —
#  크게 벗어나면 인지가 다른 물체를 봤거나 extrinsics 가 틀린 것이고, 둘 다 그대로
#  넣으면 컵이 공중이나 테이블 속에 소환된다.
Z_TOLERANCE_M = 0.03


@dataclass(frozen=True)
class CupPose:
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    frame: str
    stamp: float
    #: 무엇이 만든 pose 인가 — 인지인지 합성인지. 없으면 사후 해석이 불가능하다.
    source: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass(frozen=True)
class SpawnBox:
    """정책이 **학습한** 컵 위치 범위. 값의 출처는 hdgp preset 하나뿐이다."""

    x: tuple[float, float]
    y: tuple[float, float]
    z: float


@dataclass(frozen=True)
class Verdict:
    inside: bool
    offenders: list[tuple[str, float, tuple[float, float]]]
    pose: CupPose
    box: SpawnBox

    def describe(self) -> str:
        head = (f"컵 {tuple(round(v, 4) for v in self.pose.position)} "
                f"[{self.pose.frame}] ← {self.pose.source}")
        if self.inside:
            return f"{head}\n  ✅ 학습 분포 안 — 그대로 소환해도 된다."
        detail = "\n".join(
            f"    {axis}: {value:.3f}  (학습 범위 {lo:.3f}~{hi:.3f})"
            for axis, value, (lo, hi) in self.offenders)
        return (f"{head}\n  ⚠ **학습 분포 밖**이다 — 소환은 되지만 정책은 겪어 본 적 없는\n"
                f"    자리에서 돈다. 실패하면 정책 탓이 아니라 분포 탓일 수 있다.\n"
                f"{detail}")


def _preset():
    spec = importlib.util.spec_from_file_location("_grasp_left_preset", HDGP_PRESET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: 배포본 기준 스폰 상자 (09.02 갱신). `spawn_box_from_preset` 는 좌 v1 기준이라
#: 낡았다 — 배포 정책의 학습 분포는 아래가 진실이다.
#:   left  = v2B25 ADR 최대 상자 (v2_preset.ADR_SPAWN_BOX_MAX, 절대 x_lo,x_hi,y_lo,y_hi)
#:           z = shaker 정착 실측 0.292 (env_rigid 테이블)
#:   right = g1/E1 spawn center (0.362,-0.16) ± adr_spawn_range_max 0.05
#:           (grasp_s2r robot_profiles + g1 dump) · z = cup_big 정착 실측 0.282
DEPLOY_SPAWN_BOXES = {
    "left": SpawnBox(x=(0.330, 0.390), y=(0.122, 0.295), z=0.292),
    "right": SpawnBox(x=(0.362 - 0.05, 0.362 + 0.05),
                      y=(-0.16 - 0.05, -0.16 + 0.05), z=0.282),
}


def spawn_box_for_side(side: str) -> SpawnBox:
    """배포 정책(좌 v2B25 · 우 g1)의 학습 스폰 상자."""
    if side not in DEPLOY_SPAWN_BOXES:
        raise ValueError(f"side 는 left/right 여야 한다: {side!r}")
    return DEPLOY_SPAWN_BOXES[side]


def spawn_box_from_preset(preset=None) -> SpawnBox:
    """학습 스폰 상자를 preset 에서 유도한다 — 숫자를 여기 적지 않는다."""
    p = preset if preset is not None else _preset()
    return SpawnBox(
        x=(p.CUP_SPAWN_X_CENTER - p.CUP_SPAWN_X_RANGE,
           p.CUP_SPAWN_X_CENTER + p.CUP_SPAWN_X_RANGE),
        y=(p.CUP_SPAWN_Y_CENTER - p.CUP_SPAWN_Y_RANGE,
           p.CUP_SPAWN_Y_CENTER + p.CUP_SPAWN_Y_RANGE),
        z=float(p.CUP_SPAWN_Z),
    )


def verdict(pose: CupPose, box: SpawnBox) -> Verdict:
    offenders: list[tuple[str, float, tuple[float, float]]] = []
    for axis, value, bounds in (("x", pose.position[0], box.x),
                                ("y", pose.position[1], box.y)):
        if not bounds[0] <= value <= bounds[1]:
            offenders.append((axis, value, bounds))
    z_bounds = (box.z - Z_TOLERANCE_M, box.z + Z_TOLERANCE_M)
    if not z_bounds[0] <= pose.position[2] <= z_bounds[1]:
        offenders.append(("z", pose.position[2], z_bounds))
    return Verdict(inside=not offenders, offenders=offenders, pose=pose, box=box)


def load_capture(path: Path, *, expect_frame: str) -> CupPose:
    raw = json.loads(Path(path).read_text())
    if "source" not in raw:
        raise ValueError(
            f"{path}: source 가 없다 — 인지가 만든 것인지 합성인지 사후에 알 수 없다"
        )
    if raw.get("frame") != expect_frame:
        raise ValueError(
            f"{path}: 프레임이 {raw.get('frame')!r} 인데 {expect_frame!r} 이어야 한다.\n"
            f"  `/cup_pose` 계약은 base 프레임이다(cup_pose_relay 가 extrinsics 를 이미 "
            f"적용한다). 카메라 프레임 pose 를 그대로 넣으면 컵이 엉뚱한 곳에 소환된다."
        )
    return CupPose(
        position=tuple(float(v) for v in raw["position"]),
        orientation_wxyz=tuple(float(v) for v in raw["orientation_wxyz"]),
        frame=raw["frame"], stamp=float(raw["stamp"]), source=str(raw["source"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", type=Path, help="붙잡지 않고 파일만 판정한다(ROS 불필요)")
    parser.add_argument("--out", type=Path, help="붙잡아 저장할 경로")
    parser.add_argument("--topic", default="/cup_pose")
    parser.add_argument("--frame", default="base_link")
    parser.add_argument("--side", choices=("left", "right"), default=None,
                        help="배포 스폰 상자 선택 (좌 v2B25 · 우 g1). 생략 시 구 v1 preset")
    parser.add_argument("--samples", type=int, default=30,
                        help="이만큼 받아 **중앙값**을 쓴다 — 한 프레임의 튐을 그대로 "
                             "소환하지 않기 위해서다")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    box = (spawn_box_for_side(args.side) if args.side
           else spawn_box_from_preset())

    if args.check:
        pose = load_capture(args.check, expect_frame=args.frame)
        result = verdict(pose, box)
        print(result.describe())
        return 0 if result.inside else 1

    if not args.out:
        raise SystemExit("--out 또는 --check 중 하나가 필요하다")

    import time

    import numpy as np
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    samples: list[tuple[float, ...]] = []
    frames: set[str] = set()

    class Capture(Node):
        def __init__(self) -> None:
            super().__init__("cup_pose_capture")
            self.create_subscription(PoseStamped, args.topic, self._cb,
                                     qos_profile_sensor_data)

        def _cb(self, msg: PoseStamped) -> None:
            frames.add(msg.header.frame_id)
            p, q = msg.pose.position, msg.pose.orientation
            samples.append((p.x, p.y, p.z, q.w, q.x, q.y, q.z))

    rclpy.init()
    node = Capture()
    deadline = time.monotonic() + args.timeout
    while len(samples) < args.samples and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    if not samples:
        raise SystemExit(
            f"{args.topic} 에서 아무것도 못 받았다({args.timeout:.0f} s).\n"
            f"  인지 체인이 도는지 확인할 것: vision-3090 `scripts/run_cup_pose_live.sh`\n"
            f"  (카메라 연결 + 목 pan −90 / tilt 280 고정이 extrinsics 유효 조건이다)"
        )
    if len(frames) != 1:
        raise SystemExit(f"프레임이 섞였다: {sorted(frames)}")

    median = np.median(np.array(samples), axis=0)
    pose = CupPose(
        position=tuple(float(v) for v in median[:3]),
        orientation_wxyz=tuple(float(v) for v in median[3:]),
        frame=frames.pop(), stamp=time.time(), source=f"{args.topic} ({len(samples)} 샘플 중앙값)",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(pose.to_json())
    print(f"-> {args.out}")
    result = verdict(pose, box)
    print(result.describe())
    return 0 if result.inside else 1


if __name__ == "__main__":
    raise SystemExit(main())
