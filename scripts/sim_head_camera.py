#!/usr/bin/env python3
"""**실기 카메라를 sim 에 그대로 소환**하기 위한 사양을 만든다.

sim 자산(URDF `head_cam_view`)의 카메라 위치는 실제와 **59.5 mm 어긋난다**. 그걸
고치려면 자산을 재빌드해야 하므로 **건드리지 않는다.** 대신 hand-eye 로 실측한
`T_neck_cam` 을 써서 `head_camera` 링크에 카메라를 **새로** 붙인다.

이렇게 하면 나중에 distillation 을 할 때 현실 카메라와 가상 카메라의 시야가 맞는다 —
학생망이 sim 에서 본 것과 실기에서 볼 것이 같아진다.

이식하는 것 셋:

  1. **extrinsics** `T_neck_cam` (hand-eye 실측) → `OffsetCfg(pos, rot, convention="ros")`
  2. **intrinsics** 실측 K → `PinholeCameraCfg.from_intrinsic_matrix(...)`
  3. **프레임 규약** — `T_neck_cam` 의 목적지는 카메라 optical 프레임이고, 그것이 곧
     Isaac 의 `convention="ros"`(+z 전방 · +x 우 · +y 하)다. 그래서 회전을 그대로 넘긴다.

★`prim_path` 는 **`head_camera`**(tilt 링크)여야 한다. `head_cam_view` 에 붙이면
그 링크가 이미 어긋난 자리에 있어 이식의 의미가 없다.

    python sim_head_camera.py                    # Isaac 설정 조각 출력
    python sim_head_camera.py --json out.json    # 기계가 읽을 형태로
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

DEFAULT_EXTRINSICS = Path(__file__).resolve().parents[1] / "config" / "head_extrinsics.yaml"
#: 2026-09-01 실측 (25프레임 전부 동일). RealSense D435 컬러, 정렬 후.
DEFAULT_INTRINSICS = [606.604, 0.0, 320.020,
                      0.0, 605.652, 240.574,
                      0.0, 0.0, 1.0]
DEFAULT_WIDTH, DEFAULT_HEIGHT = 640, 480
DEFAULT_CLIPPING = (0.01, 10.0)
#: 카메라를 붙일 링크. tilt 가 움직이는 링크라 목을 돌려도 따라간다.
NECK_LINK = "head_camera"
ORTHOGONALITY_TOL = 1e-3


@dataclass(frozen=True)
class HeadCameraSpec:
    link: str
    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    width: int
    height: int
    intrinsic_matrix: tuple[float, ...]
    clipping_range: tuple[float, float]


def quat_wxyz_from_matrix(R: np.ndarray) -> list[float]:
    """회전행렬 → (w, x, y, z). Shepperd 방식으로 수치적으로 안전하게."""
    m = np.asarray(R, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return [0.25 * s, (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                0.25 * s, (m[1, 2] + m[2, 1]) / s]
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s, 0.25 * s]


def build_spec(
    t_neck_cam: np.ndarray,
    intrinsic_matrix=DEFAULT_INTRINSICS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    clipping_range: tuple[float, float] = DEFAULT_CLIPPING,
) -> HeadCameraSpec:
    """`T_neck_cam` 과 K 를 Isaac 이 먹을 형태로 옮긴다. K 는 손대지 않는다."""
    T = np.asarray(t_neck_cam, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T_neck_cam 은 4x4 여야 한다: {T.shape}")
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=ORTHOGONALITY_TOL):
        raise ValueError("회전 성분이 직교하지 않는다 — T_neck_cam 을 확인할 것")
    K = list(map(float, intrinsic_matrix))
    if len(K) != 9:
        raise ValueError(f"K 는 행우선 9개여야 한다: {len(K)}개")

    # 실측 행렬은 완전 직교가 아니므로 가장 가까운 회전으로 투영한다
    u, _, vt = np.linalg.svd(R)
    return HeadCameraSpec(
        link=NECK_LINK,
        pos=tuple(float(v) for v in T[:3, 3]),
        quat_wxyz=tuple(quat_wxyz_from_matrix(u @ vt)),
        width=int(width), height=int(height),
        intrinsic_matrix=tuple(K), clipping_range=tuple(clipping_range),
    )


def load_spec(path: Path = DEFAULT_EXTRINSICS, **kwargs) -> HeadCameraSpec:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return build_spec(np.array(data["neck_to_camera"]["matrix"], dtype=float), **kwargs)


def render_isaac_snippet(spec: HeadCameraSpec) -> str:
    """Isaac Lab 태스크 cfg 에 그대로 붙여 넣을 수 있는 조각."""
    K = ", ".join(f"{v:.6f}" for v in spec.intrinsic_matrix)
    pos = ", ".join(f"{v:+.9f}" for v in spec.pos)
    quat = ", ".join(f"{v:+.9f}" for v in spec.quat_wxyz)
    return f'''# ── head 카메라 (실기 hand-eye 실측을 그대로 이식) ──────────────────────
# ★자산의 head_cam_view 는 실제와 59.5 mm 어긋난다 — 쓰지 말 것.
#   여기서는 tilt 링크 `{spec.link}` 에 실측 오프셋으로 직접 붙인다.
# ★convention="ros" 필수 — 빠지면 카메라가 엉뚱한 곳을 본다.
head_camera_cfg: CameraCfg = CameraCfg(
    prim_path="{{ENV_REGEX_NS}}/Robot/{spec.link}/head_cam_real",
    update_period=0.0,
    height={spec.height},
    width={spec.width},
    data_types=["rgb", "distance_to_image_plane"],
    offset=CameraCfg.OffsetCfg(
        pos=({pos}),
        rot=({quat}),
        convention="ros",
    ),
    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=[{K}],
        width={spec.width},
        height={spec.height},
        clipping_range={spec.clipping_range},
    ),
)'''


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extrinsics", default=str(DEFAULT_EXTRINSICS))
    parser.add_argument("--json", default=None, help="사양을 JSON 으로도 저장")
    args = parser.parse_args()

    try:
        spec = load_spec(Path(args.extrinsics))
    except (OSError, KeyError, ValueError, yaml.YAMLError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    fx, fy = spec.intrinsic_matrix[0], spec.intrinsic_matrix[4]
    print(f"# 출처 {args.extrinsics}")
    print(f"# 해상도 {spec.width}x{spec.height} · fx {fx:.2f} fy {fy:.2f} · "
          f"FOV {math.degrees(2 * math.atan(spec.width / 2 / fx)):.2f}° x "
          f"{math.degrees(2 * math.atan(spec.height / 2 / fy)):.2f}°\n")
    print(render_isaac_snippet(spec))

    if args.json:
        Path(args.json).write_text(json.dumps(spec.__dict__, indent=1), encoding="utf-8")
        print(f"\n# 저장: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
