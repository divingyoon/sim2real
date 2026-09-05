#!/usr/bin/env python3
"""보드를 CAD 위치에 고정한 뒤 T_base_cam 재계산 (순수 수학, cv2 무관).

입력: charuco_calib.py 가 뱉은 JSON (T_cam_board 4x4 포함, optical 프레임).
방식: T_base_board(CAD 고정값) ∘ inv(T_cam_board) → global_camera_extrinsics.yaml
      camera 블록만 갱신. 카메라 위치 가정/yaw sweep 불필요.

CAD 배치 (사용자 확정, robot_base 기준):
  - ChArUco 7x5, square 0.030 → 보드 전체 0.210(긴변) x 0.150(짧은변) m
  - 정중앙 = (0.420, -0.050, 0.200) m
  - yaw=0 정렬, 긴 변(0.210, 7칸)이 base +y, 인쇄면이 위(수평 z=0.200 평면)
  - 4 외곽 코너(=BOARD_CORNERS_BASE)로 교차검증
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from extrinsics_calib import compose_base_cam, mat_to_pos_quat_wxyz
from calibrate_camera_extrinsics import update_camera_extrinsics_yaml

YAML = str(_HERE.parent / "config" / "global_camera_extrinsics.yaml")

# --- 보드 기하 (7x5, square 0.030) ---
BOARD_W = 7 * 0.030   # 긴 변 0.210 (보드 object x축, squaresX 방향)
BOARD_H = 5 * 0.030   # 짧은 변 0.150 (보드 object y축, squaresY 방향)
HALF = np.array([BOARD_W / 2, BOARD_H / 2, 0.0])   # 중앙 오프셋 (0.105, 0.075, 0)
CENTER_BASE = np.array([0.420, -0.050, 0.200])     # CAD 정중앙

# 긴 변(board x) → base +y, 인쇄면 위(board z → base -z), 우수직교:
#   R = Rz(90) @ Rx(180) = [[0,1,0],[1,0,0],[0,0,-1]]
# ⚠ 실물 검출 전엔 두 가지 모호성이 남음 → head_fk.py 교차검증으로 확정:
#   ① yaw 90 vs 270 (긴변 +y vs -y, board x축 부호)
#   ② board z 방향 (Rx180 vs Rx0) — 인쇄면이 실제로 위인지. z 뒤집히면 카메라 z/pitch 부호가 어긋남.
# CAM_XY_MAX/CAM_Z_RANGE 박스체크는 coarse — 정밀 확정은 FK 예측과 비교.
R_BASE_BOARD = np.array([[0.0, 1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0]])

# 사용자 CAD 실측 외곽 코너 (robot_base, m) — 교차검증용
BOARD_CORNERS_BASE = np.array([
    [0.345,  0.055, 0.200],
    [0.495,  0.055, 0.200],
    [0.345, -0.155, 0.200],
    [0.495, -0.155, 0.200],
])

# sanity: 카메라가 base 위쪽 근방이어야 정상 (head 카메라 ~z 0.83, board z 0.20)
CAM_XY_MAX = 0.35
CAM_Z_RANGE = (0.3, 1.1)


def t_base_board() -> np.ndarray:
    """OpenCV 보드 object 프레임(원점=좌하단 코너) → base 4x4."""
    origin = CENTER_BASE - R_BASE_BOARD @ HALF   # 중앙 → 코너 원점
    T = np.eye(4)
    T[:3, :3] = R_BASE_BOARD
    T[:3, 3] = origin
    return T


def board_outer_corners_base() -> np.ndarray:
    """R·center 로 계산한 4 외곽 코너(base). CAD 실측과 교차검증용."""
    signs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    return np.array([CENTER_BASE + R_BASE_BOARD @ np.array([sx * HALF[0], sy * HALF[1], 0.0])
                     for sx, sy in signs])


def _assert_corners_match() -> None:
    computed = board_outer_corners_base()
    for g in BOARD_CORNERS_BASE:
        if not any(np.allclose(g, c, atol=1e-6) for c in computed):
            raise AssertionError(f"보드 코너 불일치: CAD {g} 가 계산 코너와 안 맞음 → R/center 확인")


def compose_t_base_cam(T_cam_board: np.ndarray) -> np.ndarray:
    """T_base_cam = T_base_board ∘ inv(T_cam_board)."""
    return compose_base_cam(t_base_board(), np.asarray(T_cam_board, dtype=float))


def main() -> None:
    ap = argparse.ArgumentParser(description="ChArUco JSON + CAD 보드위치 → T_base_cam yaml 갱신")
    ap.add_argument("json_path", help="charuco_calib.py 출력 JSON (T_cam_board 포함)")
    ap.add_argument("--write", action="store_true", help="yaml 기록(없으면 dry-run)")
    a = ap.parse_args()

    _assert_corners_match()   # 기동 시 기하 sanity
    try:
        data = json.loads(Path(a.json_path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"JSON 읽기 실패 {a.json_path!r}: {e}")
    if "T_cam_board" not in data:
        sys.exit(f"JSON에 T_cam_board 없음 (charuco 검출 실패?): keys={list(data)}")

    T_base_cam = compose_t_base_cam(np.array(data["T_cam_board"], dtype=float))
    pos, quat = mat_to_pos_quat_wxyz(T_base_cam)

    print("보드원점(base):", t_base_board()[:3, 3].round(4).tolist())
    print("T_base_cam pos:", [round(v, 4) for v in pos])
    print("T_base_cam quat_wxyz:", [round(v, 4) for v in quat])
    print("reproj(px):", data.get("reproj_px"), "| corners:", data.get("charuco_corners"))
    if not (abs(pos[0]) < CAM_XY_MAX and abs(pos[1]) < CAM_XY_MAX
            and CAM_Z_RANGE[0] < pos[2] < CAM_Z_RANGE[1]):
        print("  ⚠ 카메라 위치가 예상 범위 밖 → R x축 부호(yaw 90 vs 270) 뒤집힘 의심")

    if a.write:
        updated = update_camera_extrinsics_yaml(YAML, pos, quat)
        with open(YAML, "w") as f:
            f.write(updated)
        print("→ yaml 갱신 완료:", YAML)
    else:
        print("(--write 없이 dry-run. 값 확인 후 --write)")


if __name__ == "__main__":
    main()
