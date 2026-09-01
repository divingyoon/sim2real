#!/usr/bin/env python3
"""hand-eye 를 풀어 `T_neck_cam` 과 `T_base_board` 를 **동시에** 얻는다.

`cv2.calibrateRobotWorldHandEye` 는 robot-world/hand-eye 문제를 푼다:

    T_base_board ∘ T_board_cam(측정) = T_base_neck(FK) ∘ T_neck_cam

미지수가 둘(`T_base_board`, `T_neck_cam`)이고 자세마다 방정식이 하나씩 생기므로,
회전이 충분히 다양한 자세 3개 이상이면 풀린다. **보드를 어디에 뒀는지 몰라도 된다** —
그것도 결과로 나온다. 테이블 세팅이 예전과 달라도 상관없다.

기존 `recompute_from_cad.py` 는 보드가 CAD 위치(0.420,-0.050,0.200)에 있다는 전제라
지금은 쓸 수 없다. 그리고 그 결과는 **한 목 자세에서만** 유효했다. 여기서 얻는
`T_neck_cam` 은 목을 돌려도 유효하다.

    python solve_head_handeye.py ../logs/handeye/run1/samples.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from head_fk_chain import t_base_neck_from_encoder as t_base_neck

MIN_SAMPLES = 3
METHODS = ("LI", "SHAH")
#: hand-eye 의 병진 정확도는 회전 크기에 비례한다. 이보다 좁으면 잔차가 작아도
#: 병진이 병렬해라 믿을 수 없다 — 2026-09-01 에 잔차 5mm 인데 보드가 z=1.48m 로 나왔다.
MIN_ROTATION_SPREAD_DEG = 30.0
#: 보드는 테이블 위에 평평히 놓여 있다. 법선이 base +z 에서 이만큼 벗어나면 의심한다.
MAX_BOARD_TILT_DEG = 25.0


def _inv(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def load_samples(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(data) < MIN_SAMPLES:
        raise ValueError(f"자세가 {len(data)}개뿐이다 — 최소 {MIN_SAMPLES}개 필요")
    return data


def solve(samples: list[dict], method_name: str) -> tuple[np.ndarray, np.ndarray]:
    """(T_base_board, T_neck_cam)."""
    import cv2

    method = getattr(cv2, f"CALIB_ROBOT_WORLD_HAND_EYE_{method_name}")
    # cv2 규약: base←gripper 의 **역**(gripper←base)을 R_world2cam 쪽에 맞춰 넣는다.
    r_world2cam, t_world2cam, r_base2gripper, t_base2gripper = [], [], [], []
    for s in samples:
        T_cam_board = np.array(s["t_cam_board"], dtype=float)
        T_base_neck = t_base_neck(s["pan_meas"], s["tilt_meas"])
        T_neck_base = _inv(T_base_neck)
        r_world2cam.append(T_cam_board[:3, :3])
        t_world2cam.append(T_cam_board[:3, 3])
        r_base2gripper.append(T_neck_base[:3, :3])
        t_base2gripper.append(T_neck_base[:3, 3])

    r_bw, t_bw, r_gc, t_gc = cv2.calibrateRobotWorldHandEye(
        r_world2cam, t_world2cam, r_base2gripper, t_base2gripper, method=method)

    # ★OpenCV 의 "a2b" 는 **b ← a** 를 뜻한다. 반환값은
    #   base2world = T_board_base · gripper2cam = T_cam_neck
    # 이므로 우리가 원하는 방향으로 뒤집어야 한다. 거꾸로 쓰면 잔차가 미터 단위로 뛴다.
    T_board_base = np.eye(4); T_board_base[:3, :3] = r_bw; T_board_base[:3, 3] = t_bw.ravel()
    T_cam_neck = np.eye(4); T_cam_neck[:3, :3] = r_gc; T_cam_neck[:3, 3] = t_gc.ravel()
    return _inv(T_board_base), _inv(T_cam_neck)


def residuals(samples: list[dict], T_base_board: np.ndarray,
              T_neck_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """자세마다 좌변·우변의 어긋남 (위치 mm, 회전 deg)."""
    pos, rot = [], []
    for s in samples:
        left = T_base_board @ _inv(np.array(s["t_cam_board"], dtype=float))
        right = t_base_neck(s["pan_meas"], s["tilt_meas"]) @ T_neck_cam
        pos.append(np.linalg.norm(left[:3, 3] - right[:3, 3]) * 1000.0)
        dR = left[:3, :3].T @ right[:3, :3]
        rot.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    return np.array(pos), np.array(rot)


def rotation_spread_deg(samples: list[dict]) -> float:
    """자세들 사이 최대 회전각. hand-eye 조건수를 좌우한다."""
    rots = [t_base_neck(s["pan_meas"], s["tilt_meas"])[:3, :3] for s in samples]
    worst = 0.0
    for i, a in enumerate(rots):
        for b in rots[i + 1:]:
            d = a.T @ b
            worst = max(worst, np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))))
    return worst


def board_tilt_deg(T_base_board: np.ndarray) -> float:
    """보드 법선(object z)과 base +z 사이 각. 테이블 위 평면이면 작아야 한다."""
    normal = T_base_board[:3, 2]
    return float(np.degrees(np.arccos(np.clip(abs(normal[2]) / np.linalg.norm(normal), -1, 1))))


def _show(label: str, T: np.ndarray) -> None:
    print(f"  {label}")
    for row in T:
        print("    [" + "  ".join(f"{v:+9.5f}" for v in row) + "]")
    print(f"    위치 [{T[0,3]:+.4f}, {T[1,3]:+.4f}, {T[2,3]:+.4f}] m")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", help="capture_head_handeye.py 의 samples.json")
    parser.add_argument("--out", default=None, help="결과 JSON 경로")
    args = parser.parse_args()

    try:
        samples = load_samples(Path(args.samples))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    pans = [s["pan_meas"] for s in samples]
    tilts = [s["tilt_meas"] for s in samples]
    print(f"자세 {len(samples)}개 · pan {min(pans):+.1f}~{max(pans):+.1f}° · "
          f"tilt {min(tilts):+.1f}~{max(tilts):+.1f}°")
    print(f"검출 재투영 {min(s['reproj_px'] for s in samples):.3f}~"
          f"{max(s['reproj_px'] for s in samples):.3f} px\n")

    best = None
    for name in METHODS:
        try:
            T_base_board, T_neck_cam = solve(samples, name)
        except Exception as exc:                       # 방법마다 수렴 실패가 다르다
            print(f"{name}: 실패 — {exc}")
            continue
        pos, rot = residuals(samples, T_base_board, T_neck_cam)
        print(f"{name}: 잔차 위치 {pos.mean():.2f}±{pos.std():.2f} mm (최대 {pos.max():.2f}) · "
              f"회전 {rot.mean():.3f}±{rot.std():.3f}° (최대 {rot.max():.3f})")
        if best is None or pos.mean() < best[0]:
            best = (pos.mean(), name, T_base_board, T_neck_cam, pos, rot)

    if best is None:
        print("❌ 어떤 방법도 풀지 못했다", file=sys.stderr)
        return 1

    _, name, T_base_board, T_neck_cam, pos, rot = best
    print(f"\n★ 채택: {name}\n")

    spread = rotation_spread_deg(samples)
    tilt = board_tilt_deg(T_base_board)
    print("진단 — 잔차만 보면 속는다:")
    print(f"  회전 폭 {spread:.1f}°" +
          ("" if spread >= MIN_ROTATION_SPREAD_DEG
           else f"  ⚠ {MIN_ROTATION_SPREAD_DEG:.0f}° 미만 — 병진이 병렬해다"))
    print(f"  보드 기울기 {tilt:.1f}° (base +z 대비)" +
          ("" if tilt <= MAX_BOARD_TILT_DEG else "  ⚠ 테이블 위 평면이라기엔 크다"))
    print(f"  보드 높이 z={T_base_board[2, 3]:+.3f} m"
          f"  ← 테이블 높이와 맞는지 눈으로 확인할 것\n")
    _show("T_neck_cam (head_camera ← 카메라 optical)", T_neck_cam)
    print()
    _show("T_base_board (base ← 보드) — 보드가 테이블 어디 있는지", T_base_board)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "method": name,
            "samples": len(samples),
            "T_neck_cam": T_neck_cam.tolist(),
            "T_base_board": T_base_board.tolist(),
            "residual_pos_mm": {"mean": float(pos.mean()), "max": float(pos.max())},
            "residual_rot_deg": {"mean": float(rot.mean()), "max": float(rot.max())},
        }, indent=1), encoding="utf-8")
        print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
