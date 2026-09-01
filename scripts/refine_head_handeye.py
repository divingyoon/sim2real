#!/usr/bin/env python3
"""hand-eye 결과를 **재투영 오차 최소화**로 정밀화한다.

`solve_head_handeye.py`(SHAH)는 자세마다 방정식 하나만 쓴다. 목이 2자유도뿐이라
회전 폭이 좁으면(2026-09-01: 24.6°) 병진이 약하게 구속돼, 잔차가 mm 단위여도 값이
통째로 틀릴 수 있다.

여기서는 **검출된 코너를 전부** 쓴다 — 25자세 × 최대 24코너 = 600개 점 제약이다.
★**인코더 영점 오프셋은 미지수로 넣지 않는다 — 관측 불가능하다.**
넣어 보면 야코비안에 **정확히 0인 특이값이 2개** 생긴다(조건수 1.8e12). 이유는 기하다:

  · pan 은 체인의 **첫** 관절이라 오프셋이 `T_base_neck` 에 **왼쪽 곱**으로 붙는다
    → `T_base_board` 를 같은 만큼 돌리면 모든 영상이 **완전히 동일**해진다
  · tilt 는 **둘째** 관절이라 **오른쪽 곱**으로 붙는다 → `T_neck_cam` 에 흡수된다

그래서 오프셋을 풀면 RMS 는 0.0000 px 로 완벽히 맞으면서 값은 틀린다. 게다가 tilt
쪽 축퇴가 정작 필요한 `T_neck_cam` 을 오염시킨다. 다행히 오프셋을 0 으로 고정해도
`T_neck_cam` 은 **정확히 복원된다**(왼쪽 곱은 오른쪽 인자에 흡수될 수 없다) — 대신
`T_base_board` 만 그만큼 어긋난다. 우리가 쓰는 건 `T_neck_cam` 이므로 문제없다.

미지수 12개: T_neck_cam(6) · T_base_board(6)

    python refine_head_handeye.py ../logs/handeye/run1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from head_fk_chain import t_base_neck, urdf_from_encoder

SX, SY, SQUARE, MARKER = 7, 5, 0.030, 0.022
DICT = cv2.aruco.DICT_6X6_250


def se3(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=float))[0]
    T[:3, 3] = np.asarray(tvec, dtype=float)
    return T


def se3_params(T: np.ndarray) -> np.ndarray:
    return np.concatenate([cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]])


def invert(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def detect_frame(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """NPZ 한 장에서 (코너 2D Nx2, 보드 3D Nx3, K 3x3). 검출 실패면 None."""
    data = np.load(npz_path)
    gray = cv2.cvtColor(data["rgb"], cv2.COLOR_RGB2GRAY)
    board = cv2.aruco.CharucoBoard((SX, SY), SQUARE, MARKER,
                                   cv2.aruco.getPredefinedDictionary(DICT))
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) < 4:
        return None
    objp, imgp = board.matchImagePoints(corners, ids)
    if objp is None or len(objp) < 4:
        return None
    return imgp.reshape(-1, 2), objp.reshape(-1, 3), data["K"].astype(float)


def residual_fn(observations: list[dict]):
    """파라미터 12개 → 모든 코너의 재투영 잔차(px)."""

    def residuals(params: np.ndarray) -> np.ndarray:
        T_neck_cam = se3(params[0:3], params[3:6])
        T_base_board = se3(params[6:9], params[9:12])

        out: list[np.ndarray] = []
        for obs in observations:
            pan, tilt = urdf_from_encoder(obs["pan"], obs["tilt"])
            T_base_cam = t_base_neck(pan, tilt) @ T_neck_cam
            T_cam_board = invert(T_base_cam) @ T_base_board
            projected, _ = cv2.projectPoints(
                obs["objp"], cv2.Rodrigues(T_cam_board[:3, :3])[0],
                T_cam_board[:3, 3], obs["K"], None)
            out.append((projected.reshape(-1, 2) - obs["imgp"]).ravel())
        return np.concatenate(out)

    return residuals


def load_observations(run_dir: Path) -> list[dict]:
    samples = json.loads((run_dir / "samples.json").read_text(encoding="utf-8"))
    observations: list[dict] = []
    for index, sample in enumerate(samples):
        npz = run_dir / f"frame_{index:02d}.npz"
        if not npz.is_file():
            continue
        found = detect_frame(npz)
        if found is None:
            continue
        imgp, objp, K = found
        observations.append({"pan": sample["pan_meas"], "tilt": sample["tilt_meas"],
                             "imgp": imgp, "objp": objp, "K": K})
    return observations


def rms(residuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean(residuals.reshape(-1, 2) ** 2) * 2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--init", default=None, help="solve_head_handeye 의 result.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    try:
        observations = load_observations(run_dir)
    except (OSError, ValueError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2
    if len(observations) < 3:
        print(f"❌ 사용 가능한 프레임이 {len(observations)}개뿐이다", file=sys.stderr)
        return 1

    points = sum(len(o["imgp"]) for o in observations)
    print(f"프레임 {len(observations)}개 · 코너 {points}개 → 잔차 {points * 2}개 · 미지수 12개")

    init_path = Path(args.init) if args.init else run_dir / "result.json"
    init = json.loads(init_path.read_text(encoding="utf-8"))
    x0 = np.concatenate([se3_params(np.array(init["T_neck_cam"])),
                         se3_params(np.array(init["T_base_board"]))])

    fn = residual_fn(observations)
    print(f"초기 RMS 재투영 {rms(fn(x0)):.3f} px")

    result = least_squares(fn, x0, method="lm", max_nfev=20000)
    T_neck_cam = se3(result.x[0:3], result.x[3:6])
    T_base_board = se3(result.x[6:9], result.x[9:12])

    print(f"정밀화 RMS 재투영 **{rms(result.fun):.3f} px** ({result.status}, "
          f"평가 {result.nfev}회)")
    singular = np.linalg.svd(result.jac, compute_uv=False)
    print(f"야코비안 조건수 {singular[0] / singular[-1]:.2e}"
          + ("" if singular[0] / singular[-1] < 1e8 else "  ⚠ 병렬해 — 자세를 더 다양하게"))
    print(f"\nT_neck_cam   위치 [{T_neck_cam[0,3]:+.4f}, {T_neck_cam[1,3]:+.4f}, "
          f"{T_neck_cam[2,3]:+.4f}] m")
    print(f"T_base_board 위치 [{T_base_board[0,3]:+.4f}, {T_base_board[1,3]:+.4f}, "
          f"{T_base_board[2,3]:+.4f}] m")

    out = Path(args.out) if args.out else run_dir / "refined.json"
    out.write_text(json.dumps({
        "rms_reproj_px": rms(result.fun),
        "frames": len(observations), "corners": points,
        "T_neck_cam": T_neck_cam.tolist(), "T_base_board": T_base_board.tolist(),
        "jacobian_condition": float(singular[0] / singular[-1]),
    }, indent=1), encoding="utf-8")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
