"""정밀화의 수학을 합성 데이터로 검증한다 — 참값을 되찾아야 한다."""

import cv2
import numpy as np
from scipy.optimize import least_squares

from head_fk_chain import t_base_neck, urdf_from_encoder
from refine_head_handeye import invert, residual_fn, rms, se3, se3_params

K = np.array([[606.6, 0.0, 320.0], [0.0, 605.7, 240.6], [0.0, 0.0, 1.0]])
TRUE_NECK_CAM = se3([0.02, -0.01, 1.57], [0.05, 0.05, 0.01])
TRUE_BASE_BOARD = se3([0.0, 0.0, 1.57], [0.17, -0.08, 0.226])
POSES = [(p, t) for p in (-10.0, -5.0, 0.0, 5.0, 10.0)
         for t in (-27.0, -20.0, -13.0)]


def test_se3_roundtrip():
    T = se3([0.1, -0.2, 0.3], [1.0, 2.0, 3.0])
    assert np.allclose(se3(se3_params(T)[:3], se3_params(T)[3:]), T, atol=1e-12)


def test_invert_is_inverse():
    T = se3([0.3, 0.1, -0.2], [0.4, -0.5, 0.6])
    assert np.allclose(T @ invert(T), np.eye(4), atol=1e-12)


def _synthetic(noise_px=0.0, seed=0):
    rng = np.random.default_rng(seed)
    objp = np.array([[x * 0.03, y * 0.03, 0.0]
                     for x in range(1, 7) for y in range(1, 5)], dtype=float)
    observations = []
    for pan, tilt in POSES:
        T_base_cam = t_base_neck(*urdf_from_encoder(pan, tilt)) @ TRUE_NECK_CAM
        T_cam_board = invert(T_base_cam) @ TRUE_BASE_BOARD
        imgp, _ = cv2.projectPoints(objp, cv2.Rodrigues(T_cam_board[:3, :3])[0],
                                    T_cam_board[:3, 3], K, None)
        imgp = imgp.reshape(-1, 2)
        if noise_px:
            imgp = imgp + rng.normal(0.0, noise_px, imgp.shape)
        observations.append({"pan": pan, "tilt": tilt, "imgp": imgp, "objp": objp, "K": K})
    return observations


def test_perfect_data_has_zero_residual_at_truth():
    fn = residual_fn(_synthetic())
    x = np.concatenate([se3_params(TRUE_NECK_CAM), se3_params(TRUE_BASE_BOARD)])
    assert rms(fn(x)) < 1e-6


def test_recovers_truth_from_perturbed_start():
    """참값에서 흔들어 놓고 최적화하면 되돌아와야 한다."""
    fn = residual_fn(_synthetic())
    x0 = np.concatenate([
        se3_params(TRUE_NECK_CAM) + np.array([0.02, -0.02, 0.02, 0.01, -0.01, 0.01]),
        se3_params(TRUE_BASE_BOARD) + np.array([0.02, 0.02, -0.02, 0.02, -0.02, 0.02])])
    out = least_squares(fn, x0, method="lm", max_nfev=20000)
    assert rms(out.fun) < 1e-3
    assert np.allclose(se3(out.x[0:3], out.x[3:6])[:3, 3], TRUE_NECK_CAM[:3, 3], atol=1e-3)


def test_tilt_offset_is_absorbed_into_neck_cam():
    """★tilt 영점 오프셋은 관측 불가능하다 — T_neck_cam 에 그대로 흡수된다.

    tilt 는 체인의 둘째 관절이라 오프셋이 T_base_neck 에 **오른쪽 곱**으로 붙는다.
    그래서 오프셋을 미지수로 넣으면 야코비안이 특이해진다(0 특이값 2개, 조건수 1.8e12).
    """
    obs = _synthetic()
    for o in obs:                       # 인코더가 tilt 를 +2° 높게 읽는 상황
        o["tilt"] = o["tilt"] + 2.0
    fn = residual_fn(obs)
    x0 = np.concatenate([se3_params(TRUE_NECK_CAM), se3_params(TRUE_BASE_BOARD)])
    out = least_squares(fn, x0, method="lm", max_nfev=20000)
    assert rms(out.fun) < 1e-3          # 완벽히 맞는다 = 오프셋이 흡수됐다는 증거


def test_pan_offset_leaves_neck_cam_intact():
    """★pan 오프셋은 **왼쪽 곱**이라 보드 자세로 흡수되고 T_neck_cam 은 살아남는다.

    그래서 오프셋을 0 으로 고정해도 우리가 쓰는 T_neck_cam 은 정확하다.
    """
    obs = _synthetic()
    for o in obs:
        o["pan"] = o["pan"] + 2.0
    fn = residual_fn(obs)
    x0 = np.concatenate([se3_params(TRUE_NECK_CAM), se3_params(TRUE_BASE_BOARD)])
    out = least_squares(fn, x0, method="lm", max_nfev=20000)
    assert rms(out.fun) < 1e-3
    assert np.allclose(se3(out.x[0:3], out.x[3:6])[:3, 3], TRUE_NECK_CAM[:3, 3], atol=2e-3)


def test_parameter_vector_stays_twelve():
    """오프셋을 다시 넣지 말 것 — 관측 불가능하고 T_neck_cam 을 오염시킨다."""
    fn = residual_fn(_synthetic())
    x = np.concatenate([se3_params(TRUE_NECK_CAM), se3_params(TRUE_BASE_BOARD)])
    assert len(x) == 12 and rms(fn(x)) < 1e-6


def test_noise_gives_comparable_rms():
    fn = residual_fn(_synthetic(noise_px=0.2, seed=7))
    x0 = np.concatenate([se3_params(TRUE_NECK_CAM), se3_params(TRUE_BASE_BOARD)])
    out = least_squares(fn, x0, method="lm", max_nfev=20000)
    assert rms(out.fun) < 0.5
