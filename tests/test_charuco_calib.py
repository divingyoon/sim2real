import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from charuco_calib import build_board, detect_charuco, SX, SY, SQUARE

# 구 cv2(신 ChArUco API 없음)에선 검출 테스트만 스킵. import·수집은 정상.
pytestmark = pytest.mark.skipif(
    not hasattr(cv2.aruco, "CharucoDetector"),
    reason="cv2 신 ChArUco API 없음(≥4.7 필요, vision-3090 .venv 5.0.0)")


def test_board_geometry():
    # 7x5 보드 → 내부 체스보드 코너 (7-1)*(5-1)=24, 전체 0.210 x 0.150
    board = build_board()
    corners = board.getChessboardCorners()
    assert corners.shape == (24, 3)
    span = corners.max(0) - corners.min(0)
    assert np.allclose(span[:2], [(SX - 2) * SQUARE, (SY - 2) * SQUARE], atol=1e-9)


def test_detect_recovers_synthetic_pose():
    # 정면 렌더 → detect_charuco 가 T_cam_board 를 낮은 재투영으로 복원
    board = build_board()
    img = board.generateImage((700, 500))
    K = np.array([[900.0, 0, 350.0], [0, 900.0, 250.0], [0, 0, 1]])
    res = detect_charuco(img, K)
    assert res is not None
    T, objp, _, reproj = res
    assert objp.shape[0] >= 6
    assert reproj < 1.0
    assert T[2, 3] > 0            # 보드가 카메라 앞(z>0)
