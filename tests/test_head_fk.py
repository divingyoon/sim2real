import numpy as np

from head_fk import head_fk, head_fk_base_optical, T_VIEW_OPTICAL


def test_zero_angles_translation():
    # pan=tilt=0: 병진 = 각 origin z 합, x=0.0225+0.0147, y=-0.0145+0.0145
    T = head_fk(0.0, 0.0)
    assert np.allclose(T[:3, 3], [0.0372, 0.0, 0.8525], atol=1e-9)
    assert np.allclose(T[:3, :3], np.eye(3), atol=1e-12)


def test_valid_rotations():
    for pan, tilt in [(0.3, -0.2), (-0.5, 0.5), (0.0, 0.523)]:
        R = head_fk(pan, tilt)[:3, :3]
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_optical_forward_looks_down_when_tilted():
    # optical +z(전방)가 tilt>0에서 아래(base -z)를 향해야 함
    T = head_fk_base_optical(0.0, 0.30)
    fwd = T[:3, :3] @ np.array([0, 0, 1.0])   # optical forward in base
    assert fwd[2] < -0.2, f"tilt+에서 아래를 봐야: fwd={fwd}"
    # x-전방 성분은 양(앞쪽 테이블)
    assert fwd[0] > 0.0


def test_optical_convention_matrix_valid():
    assert np.isclose(np.linalg.det(T_VIEW_OPTICAL), 1.0)
    # 광축(optical z)이 view x(전방)와 정렬
    assert np.allclose(T_VIEW_OPTICAL @ np.array([0, 0, 1.0]), [1, 0, 0], atol=1e-12)
