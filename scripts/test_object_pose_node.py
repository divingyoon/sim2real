"""object_pose_node 순수부 — 레지스트리 → Extrinsics → base_link 변환."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from object_pose_node import PoseConverter  # noqa: E402
from object_registry import DEFAULT_REGISTRY, load_registry  # noqa: E402

REG = load_registry(DEFAULT_REGISTRY)


def test_shaker_camera_pose_maps_to_measured_base_pose():
    """09.03 실측: 카메라 프레임 (0.0363,-0.1111,0.5877) → base (0.3627,-0.0032,0.3224)."""
    conv = PoseConverter(REG, ["shaker_closed"])
    pos, quat = conv.convert("shaker_closed", np.array([0.0363, -0.1111, 0.5877]),
                             np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(pos, [0.3627, -0.0032, 0.3224], atol=0.01)
    assert np.isclose(np.linalg.norm(quat), 1.0)


def test_cup_applies_yup_to_zup_but_same_camera():
    conv = PoseConverter(REG, ["shaker_closed", "cup_big_s100"])
    p_s, q_s = conv.convert("shaker_closed", np.zeros(3), np.array([1.0, 0, 0, 0]))
    p_c, q_c = conv.convert("cup_big_s100", np.zeros(3), np.array([1.0, 0, 0, 0]))
    assert np.allclose(p_s, p_c)          # cad_to_body 는 위치 0 이라 위치 동일
    assert not np.allclose(q_c, q_s)      # cup 은 Y-up→Z-up 회전이 붙는다


def test_unknown_name_rejected_and_names_resolved():
    conv = PoseConverter(REG, ["cup_big_s080"])
    assert conv.names == ["cup_big_s100"]
    try:
        PoseConverter(REG, ["teapot"])
    except ValueError as err:
        assert "unknown object" in str(err)
    else:
        raise AssertionError("expected ValueError")
