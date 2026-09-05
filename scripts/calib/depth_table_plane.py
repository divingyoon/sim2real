#!/usr/bin/env python3
"""depth 영상으로 **테이블 상면의 base 기준 높이**를 잰다 (ChArUco 코너와 독립인 모달리티).

왜. 카메라 사슬(T_base_cam = T_base_neck(pan,tilt)·T_neck_cam)이 테이블을 0.226~0.230 으로
보는데 줄자·env_v1 CAD 는 0.205 다. 그 21 mm 가 hand-eye(코너 재투영)의 산물인지, 사슬
자체의 편향인지는 **코너를 안 쓰는** depth 평면으로 가려진다. depth 도 0.226 이면 편향은
사슬(마운트 높이 B4)에 있고, depth 가 0.205 면 charuco 쪽(보드 두께·인쇄 배율)이다.

방법. hand-eye 캡처(logs/handeye/run1/frame_XX.npz: rgb·depth[m]·K, samples.json: 실측
pan/tilt 인코더각)를 그대로 쓴다. 픽셀을 depth 로 역투영 → T_base_cam 으로 base 로 옮기고
테이블 영역(base x∈[0.05,0.70], |y|<0.45, z∈[0.05,0.40]) 점에 RANSAC 평면을 맞춘다.
프레임마다 평면의 z(보드 자리에서)와 법선 기울기를 내고, 25 프레임의 중앙값을 결론으로 쓴다.

    python3 depth_table_plane.py [logs/handeye/run1] [--extrinsics config/head_extrinsics.yaml]

depth 가 컬러에 정렬돼 있어야 한다(K 는 컬러 내부행렬). 정렬 검사로 각 프레임의 charuco
보드 원점(t_cam_board)을 K 로 투영한 픽셀의 depth 와 보드 원점 z(카메라 z) 를 비교해 찍는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from head_fk_chain import t_base_neck_from_encoder  # noqa: E402

SIM2REAL = Path(__file__).resolve().parents[2]
TABLE_X = (0.05, 0.70)
TABLE_Y = (-0.45, 0.45)
TABLE_Z = (0.05, 0.40)
RANSAC_ITERS = 300
RANSAC_INLIER_M = 0.004
TARGET_Z = 0.205


def load_t_neck_cam(path: Path) -> np.ndarray:
    return np.array(yaml.safe_load(path.read_text())["neck_to_camera"]["matrix"], dtype=float)


def backproject(depth: np.ndarray, K: np.ndarray, stride: int = 4) -> np.ndarray:
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    ok = np.isfinite(z) & (z > 0.15) & (z < 2.5)
    u, v, z = u[ok].astype(float), v[ok].astype(float), z[ok].astype(float)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=1)


def ransac_plane(points: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, np.ndarray]:
    """(normal, d, inlier_mask) with n·p + d = 0, normal pointing +z."""
    best, best_mask = None, None
    for _ in range(RANSAC_ITERS):
        idx = rng.choice(len(points), 3, replace=False)
        p0, p1, p2 = points[idx]
        n = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        d = -n @ p0
        mask = np.abs(points @ n + d) < RANSAC_INLIER_M
        if best_mask is None or mask.sum() > best_mask.sum():
            best, best_mask = (n, d), mask
    inl = points[best_mask]
    c = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - c)
    n = vt[2]
    if n[2] < 0:
        n = -n
    return n, float(-n @ c), best_mask


def analyse_frame(npz: Path, sample: dict, t_neck_cam: np.ndarray, rng, scale_by_charuco: bool = False) -> dict:
    data = np.load(npz)
    depth, K = data["depth"].astype(float), data["K"].astype(float)
    if scale_by_charuco:
        # RealSense depth carries a distance-proportional bias; rescale so the depth at the
        # charuco origin pixel equals the charuco-pose camera z (one point per frame).
        t_cb = np.array(sample["t_cam_board"], dtype=float)
        o = t_cb[:3, 3]
        u, v = int(round(K[0, 0] * o[0] / o[2] + K[0, 2])), int(round(K[1, 1] * o[1] / o[2] + K[1, 2]))
        win = depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
        win = win[np.isfinite(win) & (win > 0)]
        if win.size:
            depth = depth * (o[2] / float(np.median(win)))
    t_base_cam = t_base_neck_from_encoder(sample["pan_meas"], sample["tilt_meas"]) @ t_neck_cam
    pts_cam = backproject(depth, K)
    pts = pts_cam @ t_base_cam[:3, :3].T + t_base_cam[:3, 3]
    box = ((pts[:, 0] > TABLE_X[0]) & (pts[:, 0] < TABLE_X[1]) & (pts[:, 1] > TABLE_Y[0]) & (pts[:, 1] < TABLE_Y[1])
           & (pts[:, 2] > TABLE_Z[0]) & (pts[:, 2] < TABLE_Z[1]))
    cand = pts[box]
    if len(cand) < 200:
        return {"frame": npz.name, "error": f"table candidates {len(cand)}"}
    n, d, mask = ransac_plane(cand, rng)
    inl = cand[mask]
    # plane height where the charuco board sat (T_base_board xy) and at the inlier centroid
    bx, by = 0.1707, -0.0841
    z_at_board = -(n[0] * bx + n[1] * by + d) / n[2]
    z_centroid = -(n[0] * inl[:, 0].mean() + n[1] * inl[:, 1].mean() + d) / n[2]
    tilt_deg = float(np.degrees(np.arccos(min(1.0, n[2]))))
    # alignment check: depth at the projected charuco origin vs its camera-z from the charuco pose
    t_cb = np.array(sample["t_cam_board"], dtype=float)
    o = t_cb[:3, 3]
    u, v = int(round(K[0, 0] * o[0] / o[2] + K[0, 2])), int(round(K[1, 1] * o[1] / o[2] + K[1, 2]))
    win = depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
    win = win[np.isfinite(win) & (win > 0)]
    depth_at_board = float(np.median(win)) if win.size else float("nan")
    return {"frame": npz.name, "pan": sample["pan_meas"], "tilt": sample["tilt_meas"],
            "inliers": int(mask.sum()), "candidates": int(len(cand)),
            "z_at_board": z_at_board, "z_centroid": z_centroid, "plane_tilt_deg": tilt_deg,
            "normal": n.round(4).tolist(),
            "board_cam_z": float(o[2]), "depth_at_board_px": depth_at_board,
            "align_mm": (depth_at_board - o[2]) * 1000.0}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default=str(SIM2REAL / "logs" / "handeye" / "run1"))
    ap.add_argument("--extrinsics", default=str(SIM2REAL / "config" / "head_extrinsics.yaml"))
    ap.add_argument("--json", default=None, help="결과를 JSON 으로 저장")
    ap.add_argument("--scale-by-charuco", action="store_true",
                    help="프레임마다 charuco 원점 픽셀의 depth 를 charuco z 에 맞춰 depth 배율을 보정")
    args = ap.parse_args(argv)
    run = Path(args.run)
    samples = json.loads((run / "samples.json").read_text())
    t_neck_cam = load_t_neck_cam(Path(args.extrinsics))
    rng = np.random.default_rng(0)
    rows = []
    for i, sample in enumerate(samples):
        npz = run / f"frame_{i:02d}.npz"
        if not npz.is_file():
            continue
        rows.append(analyse_frame(npz, sample, t_neck_cam, rng, args.scale_by_charuco))
    good = [r for r in rows if "error" not in r]
    print(f"{'frame':9s} {'pan':>6s} {'tilt':>6s} {'inl':>6s} {'z@board':>8s} {'z@cent':>8s} {'tilt°':>6s} {'align':>7s}")
    for r in rows:
        if "error" in r:
            print(f"{r['frame']:9s} {r['error']}")
            continue
        print(f"{r['frame']:9s} {r['pan']:6.1f} {r['tilt']:6.1f} {r['inliers']:6d} {r['z_at_board']:8.4f} "
              f"{r['z_centroid']:8.4f} {r['plane_tilt_deg']:6.2f} {r['align_mm']:+6.1f}mm")
    if good:
        zb = np.array([r["z_at_board"] for r in good])
        zc = np.array([r["z_centroid"] for r in good])
        al = np.array([r["align_mm"] for r in good if np.isfinite(r["align_mm"])])
        print(f"\n{len(good)} frames: table z@board median {np.median(zb):.4f} (mean {zb.mean():.4f}, std {zb.std():.4f}); "
              f"z@centroid median {np.median(zc):.4f}; plane tilt median {np.median([r['plane_tilt_deg'] for r in good]):.2f}°; "
              f"depth/charuco alignment {np.median(al):+.1f} mm (std {al.std():.1f})")
        print(f"target {TARGET_Z:.3f} -> bias {np.median(zb) - TARGET_Z:+.4f} m  |  charuco T_base_board z 0.2264 -> {np.median(zb) - 0.2264:+.4f} m")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
