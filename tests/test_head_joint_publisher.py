"""목 관절 상태 발행의 순수 부분 테스트 (ROS·하드웨어 무관)."""

import math

import pytest

from head_joint_publisher import JOINT_NAMES, joint_state_fields


def test_publishes_urdf_convention_radians():
    """★`joint_states` 는 URDF 관절 이름을 쓰므로 URDF 규약으로 나가야 한다.

    인코더 각을 그대로 실으면 pan 이 반대로 흐른다.
    """
    names, positions = joint_state_fields(pan_encoder_deg=10.0, tilt_encoder_deg=-20.0)
    assert names == JOINT_NAMES == ["head_j_pan", "head_j_tilt"]
    assert positions[0] == pytest.approx(math.radians(-10.0))   # pan 부호 반전
    assert positions[1] == pytest.approx(math.radians(-20.0))   # tilt 그대로


def test_home_pose_maps_to_expected_radians():
    _, positions = joint_state_fields(0.0, -20.0)
    assert positions == pytest.approx([0.0, math.radians(-20.0)])


def test_roundtrip_through_relay_conversion():
    """★발행 → 릴레이 역변환이 원래 인코더 각으로 돌아와야 한다."""
    from head_fk_chain import encoder_from_urdf
    for pan, tilt in ((0.0, -20.0), (12.5, -27.0), (-8.0, -13.0)):
        _, pos = joint_state_fields(pan, tilt)
        back = encoder_from_urdf(math.degrees(pos[0]), math.degrees(pos[1]))
        assert back == pytest.approx((pan, tilt))
