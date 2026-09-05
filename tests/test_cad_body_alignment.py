import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cad_body_alignment import mesh_aabb, cad_to_body_yup_to_zup  # noqa: E402
from pour_obs_geometry import quat_apply  # noqa: E402

CUP_OBJ = str(Path(__file__).resolve().parents[2] / "perception/assets/Cup/cup.obj")

def test_mesh_aabb_matches_measured_cup():
    mn, mx = mesh_aabb(CUP_OBJ)
    assert np.allclose(mn, [-0.0463, -0.0773, -0.0440], atol=1e-3)
    assert np.allclose(mx, [ 0.0437,  0.1003,  0.0460], atol=1e-3)

def test_yup_to_zup_rotation_maps_mesh_Y_to_body_Z():
    _, quat = cad_to_body_yup_to_zup(
        np.array([-0.0463, -0.0773, -0.0440]),
        np.array([ 0.0437,  0.1003,  0.0460]))
    # mesh +Y(높이축)가 body +Z 로 가야 한다
    assert np.allclose(quat_apply(quat, [0, 1, 0]), [0, 0, 1], atol=1e-9)

def test_yup_to_zup_preserves_mesh_origin_matches_sim_body_convention():
    aabb_min = np.array([-0.0463, -0.0773, -0.0440])
    aabb_max = np.array([ 0.0437,  0.1003,  0.0460])
    pos, quat = cad_to_body_yup_to_zup(aabb_min, aabb_max)
    # (a) 평행이동 없음: sim body 원점 = mesh 원점
    assert np.allclose(pos, [0.0, 0.0, 0.0], atol=1e-12)
    # (b) mesh 바닥/림 점이 body z=-0.0773/+0.1003 로 매핑
    #     (evidence: hdgp pour_v1/pour_sensor bottom=-0.077/rim=+0.100
    #      == cup.obj raw Y-AABB)
    xc = (aabb_min[0] + aabb_max[0]) / 2
    zc = (aabb_min[2] + aabb_max[2]) / 2
    bottom_mesh = np.array([xc, aabb_min[1], zc])
    rim_mesh = np.array([xc, aabb_max[1], zc])
    bottom_body = np.array(quat_apply(quat, bottom_mesh)) + pos
    rim_body = np.array(quat_apply(quat, rim_mesh)) + pos
    assert np.isclose(bottom_body[2], -0.0773, atol=1e-3)
    assert np.isclose(rim_body[2], 0.1003, atol=1e-3)
