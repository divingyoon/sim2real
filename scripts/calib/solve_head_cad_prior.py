#!/usr/bin/env python3
"""head_v1 CAD 를 사전 모델로 한 hand-eye 재계산 (2026-09-05).

기존 hand-eye(`refine_head_handeye.py`)는 T_neck_cam 6 자유도를 통째로 풀었다. 그래서
tilt 인코더 영점 오프셋과 카메라 장착 자세가 한 덩어리로 들어갔고(CAD 대비 tilt 축
기준 91°), 마운트 높이 오차는 보드 자세로 흡수돼 **관측 불가**였다.

head_v1 은 카메라 광학 프레임(color_frame)까지 CAD 로 정해져 있으므로 여기서는
카메라 장착을 **CAD 로 고정**하고 미지수를 줄인다:
    pan_off, tilt_off   인코더 영점 → CAD 관절각 (deg)   [URDF 각 = 부호변환(인코더) + off]
    dz                  마운트 높이 보정 (m)             [보드 z 를 0.205 로 묶을 때 관측됨]
    board x, y, yaw     보드가 테이블 어디에 있나        [z=0.205 · 수평(level) 고정]
변형:
    --free-board-z      dz 대신 보드 z 를 풀어 "CAD 사슬이 보는 테이블 높이"를 낸다
    --cam-rot           카메라 장착 회전 잔차(rx,ry,rz)도 푼다 (CAD 장착 자세 검증)

    python3 solve_head_cad_prior.py [logs/handeye/run1]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import head_fk_chain as fk  # noqa: E402

SIM2REAL = Path(__file__).resolve().parents[2]
SX, SY, SQUARE, MARKER = 7, 5, 0.030, 0.022
TABLE_Z = 0.205
# head_v1 color_frame in the head_camera (tilt) link frame (vendor/head_v1/head_data.json)
CAD_CAM_XYZ = np.array([0.0115, 0.000096, 0.036327])
CAD_CAM_R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # optical x,y,z in head frame
# charuco board frame on the table (found by freeing roll/pitch on 2026-09-05):
# board x -> +y_base, board y -> -x_base, board z -> +z_base (= rot_z(90°)); yaw on top
BOARD_FLIP = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def detect(npz: Path):
    data = np.load(npz)
    gray = cv2.cvtColor(data["rgb"], cv2.COLOR_RGB2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard_create(SX, SY, SQUARE, MARKER, dictionary)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None or len(ids) < 4:
        return None
    n, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if cc is None or n < 4:
        return None
    objp = np.asarray(board.chessboardCorners)[ci.ravel()]
    return cc.reshape(-1, 2).astype(float), objp.astype(float), data["K"].astype(float)


def rot_x(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def rot_y(a): c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rot_z(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def t_cam(cam_rot):
    T = np.eye(4)
    T[:3, :3] = CAD_CAM_R @ rot_x(cam_rot[0]) @ rot_y(cam_rot[1]) @ rot_z(cam_rot[2])
    T[:3, 3] = CAD_CAM_XYZ
    return T


def t_base_cam(pan_enc, tilt_enc, p):
    pan_u, tilt_u = fk.urdf_from_encoder(pan_enc, tilt_enc)
    T = fk.t_base_neck(pan_u + p["pan_off"], tilt_u + p["tilt_off"]).copy()
    # optional head-mount misalignment: the whole head rotated about base x/y at the mount point
    if p.get("mx", 0.0) or p.get("my", 0.0):
        M = np.eye(4); M[:3, :3] = rot_x(p["mx"]) @ rot_y(p["my"])
        mount = np.array([0.0, 0.0, 0.750])
        T[:3, 3] -= mount; T = M @ T; T[:3, 3] += mount
    T[2, 3] += p["dz"]
    Tc = t_cam(p["cam_rot"])
    Tc[:3, 3] += np.array([p.get("tx", 0.0), p.get("ty", 0.0), p.get("tz", 0.0)])
    return T @ Tc


def t_base_board(p):
    T = np.eye(4)
    T[:3, :3] = rot_z(p["yaw"]) @ rot_x(p["roll"]) @ rot_y(p["pitch"]) @ BOARD_FLIP
    T[:3, 3] = [p["bx"], p["by"], p["bz"]]
    return T


def unpack(x, spec, fixed):
    p = dict(fixed)
    for name, value in zip(spec, x):
        p[name] = value
    p["cam_rot"] = np.array([p.get("rx", 0.0), p.get("ry", 0.0), p.get("rz", 0.0)])
    return p


def residuals(x, spec, fixed, obs):
    p = unpack(x, spec, fixed)
    Tb = t_base_board(p)
    out = []
    for o in obs:
        Tc = t_base_cam(o["pan"], o["tilt"], p)
        pts_b = o["objp"] @ Tb[:3, :3].T + Tb[:3, 3]
        Tcb = np.linalg.inv(Tc)
        pc = pts_b @ Tcb[:3, :3].T + Tcb[:3, 3]
        u = o["K"][0, 0] * pc[:, 0] / pc[:, 2] + o["K"][0, 2]
        v = o["K"][1, 1] * pc[:, 1] / pc[:, 2] + o["K"][1, 2]
        out.append(np.stack([u, v], 1) - o["imgp"])
    return np.concatenate(out).ravel()


def solve(obs, spec, fixed, x0):
    r = least_squares(residuals, x0, args=(spec, fixed, obs), method="lm", max_nfev=20000)
    rms = math.sqrt(np.mean(r.fun ** 2) * 2)  # per-corner (2D) RMS in px
    return unpack(r.x, spec, fixed), rms, r


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default=str(SIM2REAL / "logs" / "handeye" / "run1"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    run = Path(args.run)
    samples = json.loads((run / "samples.json").read_text())
    obs = []
    for i, s in enumerate(samples):
        d = detect(run / f"frame_{i:02d}.npz")
        if d is None:
            continue
        imgp, objp, K = d
        obs.append({"pan": s["pan_meas"], "tilt": s["tilt_meas"], "imgp": imgp, "objp": objp, "K": K})
    print(f"{len(obs)} frames, {sum(len(o['imgp']) for o in obs)} corners")

    base_fixed = {"dz": 0.0, "bz": TABLE_Z, "roll": 0.0, "pitch": 0.0, "pan_off": 0.0, "tilt_off": 0.0,
                  "bx": 0.17, "by": -0.08, "yaw": 0.0}
    variants = {
        # A. CAD camera, table constraint: encoder offsets + mount dz + board x,y,yaw
        "A_cad_table": (["pan_off", "tilt_off", "dz", "bx", "by", "yaw"], {}),
        # B. CAD camera, dz fixed 0, board z free: what table height does the CAD chain imply?
        "B_cad_freez": (["pan_off", "tilt_off", "bx", "by", "bz", "yaw"], {}),
        # C. B + board roll/pitch free (is the board level under the CAD chain?)
        "C_cad_freez_tilt": (["pan_off", "tilt_off", "bx", "by", "bz", "yaw", "roll", "pitch"], {}),
        # D. A + camera mounting rotation residual (does CAD mounting hold?)
        "D_cad_table_camrot": (["pan_off", "tilt_off", "dz", "bx", "by", "yaw", "rx", "ry", "rz"], {}),
        # E. CAD camera + mounting rotation residual, board z free (mount dz fixed 0)
        "E_cad_freez_camrot": (["pan_off", "tilt_off", "bx", "by", "bz", "yaw", "rx", "ry", "rz"], {}),
        # F. CAD camera + head MOUNT rotation (mx,my), board z free
        "F_cad_freez_mountrot": (["pan_off", "tilt_off", "bx", "by", "bz", "yaw", "mx", "my"], {}),
        # G. CAD camera orientation + camera translation residual (tx,ty,tz), board z free
        "G_cad_freez_camtrans": (["pan_off", "tilt_off", "bx", "by", "bz", "yaw", "tx", "ty", "tz"], {}),
        # H. only a camera yaw residual (ry) on top of CAD, board z free
        "H_cad_freez_ry": (["pan_off", "tilt_off", "bx", "by", "bz", "yaw", "ry"], {}),
    }
    results = {}
    for name, (spec, extra) in variants.items():
        fixed = dict(base_fixed); fixed.update(extra)
        x0 = []
        for k in spec:
            x0.append({"tilt_off": 90.0, "bz": 0.22}.get(k, fixed.get(k, 0.0)))
        p, rms, r = solve(obs, spec, fixed, np.array(x0, float))
        home = t_base_cam(0.0, -20.0, p)
        summary = {k: (float(p[k]) if k in ("pan_off", "tilt_off") else float(p[k])) for k in spec}
        summary["rms_px"] = rms
        summary["cam_home_pan0_tilt-20"] = home[:3, 3].round(4).tolist()
        summary["cam_home_forward"] = home[:3, 2].round(3).tolist()
        summary["board_z_used"] = float(p["bz"])
        results[name] = summary
        print(f"\n== {name}: RMS {rms:.3f} px")
        for k in spec:
            unit = "deg" if k in ("pan_off", "tilt_off", "yaw", "roll", "pitch", "rx", "ry", "rz", "mx", "my") else "m"
            val = math.degrees(p[k]) if k in ("yaw", "roll", "pitch", "rx", "ry", "rz", "mx", "my") else p[k]
            print(f"   {k:8s} = {val:+.4f} {unit}")
        print(f"   camera @ pan0/tilt-20: pos {summary['cam_home_pan0_tilt-20']} forward {summary['cam_home_forward']}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
