"""M2 — codec: ROS 메시지 ↔ numpy/dataclass 순수 변환.

메시지 타입을 아는 유일한 모듈이다. 라벨/크기가 계약과 다르면 거부하고, 관절은 이름으로
재정렬하며(결손 → 에러, 0 채움 금지), seq 는 layout.data_offset 으로 실어 나른다.
"""
from __future__ import annotations

import numpy as np
import pytest

from policy_control import codec

pytestmark = pytest.mark.unit
ros_msgs = pytest.importorskip("std_msgs.msg")

SEGS = (("a", 2), ("b", 3))


def test_module_imports_without_ros_message_classes_at_top_level():
    import importlib
    src = importlib.util.find_spec("policy_control.codec").origin
    text = open(src).read().split("def ", 1)[0]          # 모듈 머리 부분만
    assert "from std_msgs" not in text and "from sensor_msgs" not in text


# ------------------------------------------------------------------ obs / action
def test_obs_roundtrip_keeps_values_labels_and_seq():
    obs = np.arange(5, dtype=float)
    msg = codec.encode_obs(obs, SEGS, seq=17)
    assert [d.label for d in msg.layout.dim] == ["a", "b"]
    assert [d.size for d in msg.layout.dim] == [2, 3]
    assert msg.layout.data_offset == 17
    got, seq = codec.decode_obs(msg, SEGS)
    assert seq == 17
    np.testing.assert_array_equal(got, obs)
    assert got is not obs


def test_decode_obs_refuses_label_or_size_mismatch():
    msg = codec.encode_obs(np.zeros(5), SEGS, seq=0)
    with pytest.raises(codec.CodecError):
        codec.decode_obs(msg, (("a", 2), ("c", 3)))
    with pytest.raises(codec.CodecError):
        codec.decode_obs(msg, (("a", 3), ("b", 2)))
    msg.data = list(msg.data[:-1])
    with pytest.raises(codec.CodecError):
        codec.decode_obs(msg, SEGS)


def test_encode_obs_refuses_wrong_length_or_nan():
    with pytest.raises(codec.CodecError):
        codec.encode_obs(np.zeros(4), SEGS, seq=0)
    with pytest.raises(codec.CodecError):
        codec.encode_obs(np.array([0, 1, np.nan, 3, 4.0]), SEGS, seq=0)


def test_action_roundtrip_and_seq():
    a = np.array([0.1, -0.2, 0.3])
    msg = codec.encode_action(a, seq=5)
    got, seq = codec.decode_action(msg, 3)
    assert seq == 5
    np.testing.assert_array_equal(got, a)
    with pytest.raises(codec.CodecError):
        codec.decode_action(msg, 4)


# ------------------------------------------------------------------ joint state
def test_joint_state_roundtrip_and_select_by_name():
    msg = codec.encode_joint_state(["j2", "j1"], [2.0, 1.0], velocity=[0.2, 0.1], stamp=1.5)
    s = codec.decode_joint_state(msg)
    assert s.names == ("j2", "j1")
    assert s.stamp == pytest.approx(1.5)
    pos, vel = codec.select_joints(s, ["j1", "j2"])
    np.testing.assert_array_equal(pos, [1.0, 2.0])
    np.testing.assert_array_equal(vel, [0.1, 0.2])


def test_select_joints_missing_joint_is_an_error_never_zero_fill():
    s = codec.decode_joint_state(codec.encode_joint_state(["j1"], [1.0]))
    with pytest.raises(codec.CodecError, match="j9"):
        codec.select_joints(s, ["j1", "j9"])


def test_select_joints_without_velocity_returns_none_velocity():
    s = codec.decode_joint_state(codec.encode_joint_state(["j1"], [1.0]))
    pos, vel = codec.select_joints(s, ["j1"])
    assert vel is None


def test_joint_target_roundtrip_reorders_by_name_and_carries_episode_seq():
    msg = codec.encode_joint_target(["b", "a"], q=[2.0, 1.0], qd=[0.2, 0.1], episode="ep3", seq=42)
    assert msg.header.frame_id == "ep3:42"
    assert list(msg.effort) == [0.0, 0.0]
    t = codec.decode_joint_target(msg, ["a", "b"])
    assert (t.episode, t.seq) == ("ep3", 42)
    np.testing.assert_array_equal(t.position, [1.0, 2.0])
    np.testing.assert_array_equal(t.velocity, [0.1, 0.2])
    with pytest.raises(codec.CodecError):
        codec.decode_joint_target(msg, ["a", "zz"])


def test_decode_joint_target_rejects_bad_frame_id():
    msg = codec.encode_joint_target(["a"], q=[1.0], qd=[0.0], episode="ep", seq=1)
    msg.header.frame_id = "garbage"
    with pytest.raises(codec.CodecError):
        codec.decode_joint_target(msg, ["a"])


# ------------------------------------------------------------------ pose / float array / status
def test_pose_roundtrip_wxyz():
    pos = np.array([0.1, 0.2, 0.3])
    quat = np.array([0.5, 0.5, 0.5, 0.5])
    msg = codec.encode_pose(pos, quat, frame="base_link", stamp=2.0)
    assert msg.pose.orientation.w == pytest.approx(0.5)
    p = codec.decode_pose(msg)
    np.testing.assert_allclose(p.pos, pos)
    np.testing.assert_allclose(p.quat, quat)
    assert p.frame == "base_link" and p.stamp == pytest.approx(2.0)


def test_float_array_roundtrip_with_layout():
    data = np.arange(15, dtype=float)
    msg = codec.encode_float_array(data, labels=("tip", "axis"), sizes=(5, 3), seq=3)
    s = codec.decode_float_array(msg)
    assert s.labels == ("tip", "axis") and s.sizes == (5, 3) and s.seq == 3
    np.testing.assert_array_equal(s.data, data)


def test_status_json_roundtrip():
    st = {"node": "obs", "ok": True, "reasons": ["x"], "seq": 3}
    msg = codec.encode_status(st)
    assert codec.decode_status(msg) == st
    msg.data = "{not json"
    with pytest.raises(codec.CodecError):
        codec.decode_status(msg)
