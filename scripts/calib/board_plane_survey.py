#!/usr/bin/env python3
"""보드를 테이블 여러 곳에 두고 찍어 **테이블 평면**을 카메라 좌표로 정밀 추정한다.

★★왜 여러 곳인가 (09.03). 한 지점에서 한 번 재면 그 근처만 맞는다. 카메라 자세가
  기울어져 있으면 오차가 **거리에 비례**해 커지므로, 한 점으로는 기울기와 높이를
  가를 수 없다. 테이블 위 여러 곳의 보드는 전부 **같은 평면**에 있어야 하니, 그
  제약이 기울기를 드러낸다.

  이 방법이 주는 것 / 못 주는 것을 분명히 해 둔다:
    ✓ 카메라 기준 **테이블 평면**(법선 2 + 거리 1) — 기울기 오차가 여기서 잡힌다
    ✗ 평면 **안에서의** x·y 이동과 yaw — 이건 로봇이 아는 점을 짚어야 한다

사용:
    # 보드를 한 곳에 두고
    python3 board_plane_survey.py add --tag pos1
    # 옮기고 다시
    python3 board_plane_survey.py add --tag pos2
    ...
    python3 board_plane_survey.py solve
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIM2REAL = HERE.parent
DEFAULT_STORE = SIM2REAL / "logs" / "board_survey"


def capture(store: Path, tag: str) -> int:
    """RGBD 한 장 → charuco 검출 → JSON 저장."""
    store.mkdir(parents=True, exist_ok=True)
    npz = store / f"{tag}.npz"
    out = store / f"{tag}.json"
    r = subprocess.run([sys.executable, str(HERE / "grab_rgbd.py"), "--out", str(npz)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not npz.exists():
        print(f"[survey] 캡처 실패: {r.stderr.strip()[:200]}", file=sys.stderr)
        return 1
    r = subprocess.run([sys.executable, str(HERE / "charuco_calib.py"), str(npz)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[survey] 보드 검출 실패: {r.stderr.strip()[:200]}", file=sys.stderr)
        return 1
    d = json.loads(r.stdout)
    out.write_text(json.dumps(d))
    T = np.array(d["T_cam_board"])
    print(f"[survey] {tag}: 코너 {d['charuco_corners']}개 · 재투영 {d['reproj_px']:.3f} px"
          f" · 보드중심(cam) {np.round(T[:3, 3], 3).tolist()}")
    return 0


def _corners_cam(d) -> np.ndarray:
    T = np.array(d["T_cam_board"])
    objp = np.array(d["board_objp"])
    return (T @ np.hstack([objp, np.ones((len(objp), 1))]).T).T[:, :3]


def solve(store: Path, extrinsics: Path) -> int:
    files = sorted(store.glob("*.json"))
    if len(files) < 2:
        print(f"[survey] 캡처가 {len(files)}개뿐이다 — 서로 떨어진 자리에서 "
              "최소 3~4번 모을 것", file=sys.stderr)
        return 1

    pts, per = [], {}
    for f in files:
        c = _corners_cam(json.loads(f.read_text()))
        per[f.stem] = c
        pts.append(c)
    allp = np.vstack(pts)

    # ── 카메라 좌표에서 테이블 평면 적합 (전 캡처 통합) ──────────────────
    ctr = allp.mean(axis=0)
    _, _, vt = np.linalg.svd(allp - ctr)
    nrm = vt[2] / np.linalg.norm(vt[2])
    resid = (allp - ctr) @ nrm
    print(f"\n[survey] 캡처 {len(files)}개 · 코너 {len(allp)}개")
    print(f"  통합 평면 잔차: RMS {resid.std()*1000:.2f} mm · 최대 "
          f"{np.abs(resid).max()*1000:.2f} mm")
    print("  ★잔차가 크면 (a) 보드가 테이블에 안 붙었거나 (b) 검출이 나쁘다")

    # 캡처별 평면과 통합 평면의 어긋남 — 기울기 오차가 여기서 보인다
    print("\n  캡처별 (자기 평면 법선이 통합 법선과 이루는 각 · 통합평면까지 평균거리)")
    for tag, c in per.items():
        cc = c.mean(axis=0)
        _, _, v = np.linalg.svd(c - cc)
        n2 = v[2] / np.linalg.norm(v[2])
        ang = np.degrees(np.arccos(abs(float(n2 @ nrm))))
        off = float(((c - ctr) @ nrm).mean())
        print(f"    {tag:12s} {ang:5.2f}° · {off*1000:+6.2f} mm")

    # ── 현재 외부 파라미터로 base 로 옮겨 본다 ──────────────────────────
    import yaml
    e = yaml.safe_load(extrinsics.read_text())["camera"]
    p = np.array(e["position"], dtype=float)
    w, x, y, z = (float(v) for v in e["orientation_wxyz"])
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                  [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    base = (R @ allp.T).T + p
    n_base = R @ nrm
    if n_base[2] < 0:
        n_base = -n_base
    tilt = np.degrees(np.arccos(np.clip(n_base[2], -1, 1)))
    print("\n[survey] 현재 외부 파라미터로 base 변환")
    print(f"  테이블 평면 법선 {np.round(n_base, 4).tolist()} → 수직에서 {tilt:.2f}°")
    print("  ★테이블은 수평이므로 이 각이 곧 **카메라 자세 오차**다(0 이어야 한다)")
    print(f"  z 평균 {base[:,2].mean():.4f} · 범위 {base[:,2].min():.4f}~"
          f"{base[:,2].max():.4f}")
    print(f"  x {base[:,0].min():.3f}~{base[:,0].max():.3f} · "
          f"y {base[:,1].min():.3f}~{base[:,1].max():.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="현재 보드 위치를 한 번 캡처")
    a.add_argument("--tag", required=True)
    a.add_argument("--store", type=Path, default=DEFAULT_STORE)
    s = sub.add_parser("solve", help="모은 캡처로 평면 추정")
    s.add_argument("--store", type=Path, default=DEFAULT_STORE)
    s.add_argument("--extrinsics", type=Path,
                   default=SIM2REAL / "config" / "global_camera_extrinsics.yaml")
    args = ap.parse_args()
    if args.cmd == "add":
        return capture(args.store, args.tag)
    return solve(args.store, args.extrinsics)


if __name__ == "__main__":
    raise SystemExit(main())
