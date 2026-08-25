"""`PalmCommandBuilder` 가 hdgp 의 `FabricPalmAction` 과 같은 것을 계산하는지.

이 파일은 두 종류를 섞어 담는다. 산술 자체의 성질(경계, 변화율 상한, 리셋)과,
**원본 소스에 대한 drift-guard**. 후자가 없으면 sim 쪽이 순서를 하나 바꾸는 순간
이쪽은 조용히 옛 규약을 계속 계산한다 — 이 저장소가 pour 계열에서 같은 이유로
`test_constants_match_pour_v1_env_cfg` 를 두고 있다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gripper_left_palm_command import PalmCommandBuilder  # noqa: E402

HDGP = Path.home() / "rl_ws/hdgp"
TASK = HDGP / "source/openarm/openarm/gripper/left/grasp_sensor"
PRESET = TASK / "grasp_left_preset.py"
ACTION = TASK / "grasp_left_fabric_action.py"


def _load_preset():
    if not PRESET.is_file():
        pytest.skip(f"hdgp preset not found: {PRESET}")
    spec = importlib.util.spec_from_file_location("_grasp_left_preset", PRESET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def preset():
    return _load_preset()


@pytest.fixture
def builder(preset):
    return PalmCommandBuilder.from_preset(preset)


def test_zero_action_is_the_centre_of_the_box(builder, preset):
    """절대 규약. a=0 이 "지금 자리를 유지"가 아니라 박스 중심이다."""
    pos, _ = builder.step(np.zeros(6))

    expected = 0.5 * np.array(
        [
            preset.PALM_BOX_X[0] + preset.PALM_BOX_X[1],
            preset.PALM_BOX_Y[0] + preset.PALM_BOX_Y[1],
            preset.PALM_BOX_Z[0] + preset.PALM_BOX_Z[1],
        ]
    )
    assert pos == pytest.approx(expected)


def test_the_extremes_of_the_action_are_the_corners_of_the_box(builder, preset):
    low, _ = builder.step(-np.ones(6))
    builder.reset()
    high, _ = builder.step(np.ones(6))

    assert low == pytest.approx([preset.PALM_BOX_X[0], preset.PALM_BOX_Y[0], preset.PALM_BOX_Z[0]])
    assert high == pytest.approx([preset.PALM_BOX_X[1], preset.PALM_BOX_Y[1], preset.PALM_BOX_Z[1]])


def test_an_action_past_one_is_clamped_not_extrapolated(builder, preset):
    pos, _ = builder.step(np.array([9.0, 9.0, 9.0, 0.0, 0.0, 0.0]))

    assert pos == pytest.approx([preset.PALM_BOX_X[1], preset.PALM_BOX_Y[1], preset.PALM_BOX_Z[1]])


def test_the_first_step_after_a_reset_is_not_rate_limited(builder, preset):
    """리셋 직후 클램프하면 이전 에피소드의 지령이 시작을 끌고 온다."""
    far, _ = builder.step(np.ones(6))

    assert far == pytest.approx([preset.PALM_BOX_X[1], preset.PALM_BOX_Y[1], preset.PALM_BOX_Z[1]])
    assert builder.last_step_norm == 0.0


def test_a_later_step_moves_no_further_than_the_rate_limit(builder, preset):
    builder.step(-np.ones(6))
    before = builder._prev_pos.copy()

    after, _ = builder.step(np.ones(6))

    moved = np.linalg.norm(after - before)
    assert moved == pytest.approx(preset.PALM_CMD_RATE_LIMIT, rel=1e-9)
    assert builder.last_step_norm == pytest.approx(moved)


def test_the_rotation_never_exceeds_the_declared_maximum(builder, preset):
    builder.rate_limit_enabled = False
    for action in (np.array([0, 0, 0, 1.0, 1.0, 1.0]), np.array([0, 0, 0, -1.0, 0.7, 0.2])):
        builder.reset()
        _, quat = builder.step(action)

        relative = _relative_angle(quat, preset.PALM_REF_QUAT_WXYZ)
        assert relative <= preset.PALM_ROT_MAX_RAD + 1e-9


def _relative_angle(quat_wxyz, ref_wxyz) -> float:
    dot = abs(float(np.dot(np.asarray(quat_wxyz), np.asarray(ref_wxyz))))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def test_zero_rotation_action_leaves_the_reference_pose_untouched(builder, preset):
    _, quat = builder.step(np.zeros(6))

    assert quat == pytest.approx(np.array(preset.PALM_REF_QUAT_WXYZ), abs=1e-12)


def test_the_features_vector_hands_the_quaternion_over_as_xyzw(builder):
    """규약을 틀리면 fabric 은 오류 없이 **다른 자세로** 간다."""
    pos, quat = builder.step(np.zeros(6))

    features = PalmCommandBuilder.as_features(pos, quat)

    assert features[:3] == pytest.approx(pos)
    assert features[3:6] == pytest.approx(quat[1:4])
    assert features[6] == pytest.approx(quat[0])


# --------------------------------------------------------------------------
# drift-guard — hdgp 원본에 대한 대조
# --------------------------------------------------------------------------


def _action_source() -> str:
    if not ACTION.is_file():
        pytest.skip(f"hdgp action term not found: {ACTION}")
    return ACTION.read_text()


def test_every_constant_comes_from_the_preset_not_from_here():
    """숫자를 옮겨 적는 순간 드리프트가 시작된다 — 그래서 하나도 두지 않는다."""
    source = Path(__file__).with_name("gripper_left_palm_command.py").read_text()
    body = source.split("class PalmCommandBuilder", 1)[1]

    for literal in ("0.005", "0.05", "0.22", "0.10", "0.43", "0.60", "30.0"):
        assert literal not in body, (
            f"{literal!r} 이 박혀 있다 — preset 에서 받아야 한다"
        )


def test_the_action_term_still_maps_the_box_the_way_we_do():
    source = _action_source()

    assert "self._box_center + actions[:, :3].clamp(-1.0, 1.0) * self._box_half" in source, (
        "박스 매핑이 바뀌었다 — PalmCommandBuilder.step 의 위치 산술을 맞출 것"
    )


def test_the_action_term_still_reorders_the_quaternion_to_xyzw():
    source = _action_source()

    assert "self._palm_target_xyz_q[:, 3:6] = q_target[:, 1:4]" in source
    assert "self._palm_target_xyz_q[:, 6] = q_target[:, 0]" in source


def test_the_action_term_still_composes_the_delta_onto_the_reference_in_world_frame():
    """`quat_mul(q_delta, ref)` 이지 `quat_mul(ref, q_delta)` 가 아니다."""
    source = _action_source()

    assert "quat_mul(q_delta, self._ref_quat_wxyz)" in source


def test_the_action_term_still_exempts_the_first_step_from_the_rate_limit():
    source = _action_source()

    assert "_fresh = ~self._cmd_primed | (not P.PALM_CMD_RATE_LIMIT_ENABLED)" in source


def test_the_delta_is_composed_in_the_world_frame_not_the_body_frame(builder, preset):
    """`q_delta ⊗ ref`, 그 반대가 아니다.

    두 순서 모두 기준 자세에서 같은 **크기**만큼 떨어진 자세를 낸다. 그래서 상대각을
    재는 검사도, a=0 검사도 둘 다 통과한다 — 순서를 뒤집어 보고서야 알았다. 회전축이
    어느 프레임에 있는지가 다르므로 결과 자세는 다르고, fabric 은 그 다른 자세로
    말없이 간다. 두 후보를 다 계산해 하나만 맞는지 본다.
    """
    from gripper_left_palm_command import _quat_mul_wxyz, quat_from_angle_axis

    builder.rate_limit_enabled = False
    action = np.array([0.0, 0.0, 0.0, 0.6, -0.3, 0.2])

    _, quat = builder.step(action)

    rotvec = action[3:6] * preset.PALM_ROT_MAX_RAD
    angle = float(np.linalg.norm(rotvec))
    delta = quat_from_angle_axis(angle, rotvec / angle)
    reference = np.array(preset.PALM_REF_QUAT_WXYZ)
    world = _quat_mul_wxyz(delta, reference)
    body = _quat_mul_wxyz(reference, delta)

    assert not np.allclose(world, body, atol=1e-6), "이 액션은 두 순서를 못 가른다"
    assert quat == pytest.approx(world, abs=1e-12)
