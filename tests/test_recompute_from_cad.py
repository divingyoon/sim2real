import numpy as np
import pytest

from recompute_from_cad import (
    t_base_board, board_outer_corners_base, compose_t_base_cam,
    CENTER_BASE, BOARD_CORNERS_BASE,
)


def test_board_origin_matches_given_corner():
    # OpenCV 보드 원점(좌하단 코너)은 주어진 물리 코너 (0.345,-0.155,0.200)
    T = t_base_board()
    assert np.allclose(T[:3, 3], [0.345, -0.155, 0.200], atol=1e-9)


def test_board_z_is_200mm():
    assert CENTER_BASE[2] == pytest.approx(0.200)
    assert np.allclose(t_base_board()[:3, 3][2], 0.200)


def test_outer_corners_match_cad_measurements():
    # R·center 로 계산한 4 외곽 코너가 사용자 CAD 실측 코너와 (집합적으로) 일치
    computed = board_outer_corners_base()
    given = BOARD_CORNERS_BASE
    for g in given:
        assert any(np.allclose(g, c, atol=1e-9) for c in computed), f"{g} 불일치"
    assert len(computed) == 4


def test_rotation_is_valid_and_expected():
    R = t_base_board()[:3, :3]
    assert np.isclose(np.linalg.det(R), 1.0)          # 우수직교
    assert np.allclose(R, [[0, 1, 0], [1, 0, 0], [0, 0, -1]])  # Rz(90)@Rx(180)


def test_roundtrip_recovers_camera_pose():
    # 임의 카메라 GT → T_cam_board 합성 → compose 가 카메라 위치를 복원
    cv2 = pytest.importorskip("cv2")
    Rcam, _ = cv2.Rodrigues(np.array([2.7, 0.0, 0.0]))   # 아래로 tilt 유사
    T_base_cam_gt = np.eye(4)
    T_base_cam_gt[:3, :3] = Rcam
    T_base_cam_gt[:3, 3] = [0.05, -0.02, 0.83]
    T_cam_board = np.linalg.inv(T_base_cam_gt) @ t_base_board()
    T_est = compose_t_base_cam(T_cam_board)
    assert np.allclose(T_est[:3, 3], T_base_cam_gt[:3, 3], atol=1e-9)
    assert np.allclose(T_est, T_base_cam_gt, atol=1e-9)
