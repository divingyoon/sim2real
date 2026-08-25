"""perception 이 준 컵 pose 를 sim 에 넣기 전에 무엇을 확인해야 하는가.

가장 중요한 것은 **학습 분포 밖인지 말해 주는 것**이다. 정책은 좁은 스폰 상자
(x 0.36~0.42 · y 0.17~0.21 · z 0.307)에서만 학습됐다. 인지가 그 밖의 컵을 주면 정책은
분포 밖에서 도는 것이고, 그 결과를 "정책이 못한다"로 읽으면 틀린 결론이 된다.
조용히 넣지 않고 **말해 준다**.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cup_pose_capture import (  # noqa: E402
    CupPose,
    SpawnBox,
    load_capture,
    spawn_box_from_preset,
    verdict,
)


#: 테스트 본문에서 쓰는 "정상 높이" — preset 에서 온다.
SPAWN_Z = spawn_box_from_preset().z


@pytest.fixture(scope="module")
def box():
    return spawn_box_from_preset()


def test_the_box_comes_from_the_preset_not_from_here(box):
    """숫자를 여기 적으면 sim 이 스폰을 옮겨도 이쪽은 옛 상자를 계속 본다.

    ★기대값도 리터럴로 쓰지 않고 preset 에서 다시 계산한다. 처음엔 SPAWN_Z 를 적었다가
      실패했는데, 그 숫자는 preset **주석**에서 베낀 것이었고 주석이 15 mm 낡아 있었다
      (실제 TABLE_SURFACE_Z 0.200 + 0.09209 = 0.29209). 리터럴을 금지하는 테스트가
      리터럴 때문에 틀린 셈이다.
    """
    import importlib.util

    from cup_pose_capture import HDGP_PRESET

    spec = importlib.util.spec_from_file_location("_preset_for_test", HDGP_PRESET)
    preset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preset)

    assert box.x == pytest.approx((preset.CUP_SPAWN_X_CENTER - preset.CUP_SPAWN_X_RANGE,
                                   preset.CUP_SPAWN_X_CENTER + preset.CUP_SPAWN_X_RANGE))
    assert box.y == pytest.approx((preset.CUP_SPAWN_Y_CENTER - preset.CUP_SPAWN_Y_RANGE,
                                   preset.CUP_SPAWN_Y_CENTER + preset.CUP_SPAWN_Y_RANGE))
    assert box.z == pytest.approx(preset.CUP_SPAWN_Z)

    source = Path(__file__).with_name("cup_pose_capture.py").read_text()
    body = source.split("def spawn_box_from_preset", 1)[1].split("\ndef ", 1)[0]
    for literal in ("0.36", "0.42", "0.17", "0.21", "0.39", "0.19", "0.292", "0.307"):
        assert literal not in body, f"{literal!r} 이 박혀 있다 — preset 에서 받아야 한다"


def test_a_pose_inside_the_box_is_accepted(box):
    pose = CupPose(position=(0.39, 0.19, SPAWN_Z), orientation_wxyz=(1, 0, 0, 0),
                   frame="base_link", stamp=0.0, source="test")

    result = verdict(pose, box)

    assert result.inside is True
    assert result.offenders == []


@pytest.mark.parametrize("position,axis", [
    ((0.30, 0.19, SPAWN_Z), "x"),
    ((0.50, 0.19, SPAWN_Z), "x"),
    ((0.39, 0.05, SPAWN_Z), "y"),
    ((0.39, 0.40, SPAWN_Z), "y"),
])
def test_a_pose_outside_the_box_names_the_axis(position, axis, box):
    pose = CupPose(position=position, orientation_wxyz=(1, 0, 0, 0),
                   frame="base_link", stamp=0.0, source="test")

    result = verdict(pose, box)

    assert result.inside is False
    assert [name for name, _, _ in result.offenders] == [axis]


def test_a_height_far_from_the_table_is_reported(box):
    """컵은 테이블 위에 있어야 한다. z 가 크게 다르면 인지가 다른 물체를 봤거나
    extrinsics 가 틀린 것이고, 둘 다 그대로 넣으면 안 된다."""
    pose = CupPose(position=(0.39, 0.19, 0.55), orientation_wxyz=(1, 0, 0, 0),
                   frame="base_link", stamp=0.0, source="test")

    result = verdict(pose, box)

    assert result.inside is False
    assert any(name == "z" for name, _, _ in result.offenders)


def test_a_pose_in_the_wrong_frame_is_refused(box, tmp_path):
    """`/cup_pose` 계약은 base 프레임이다. 카메라 프레임 pose 를 그대로 넣으면
    컵이 로봇 뒤 어딘가에 소환된다."""
    path = tmp_path / "c.json"
    path.write_text(json.dumps({
        "position": [0.39, 0.19, 0.307], "orientation_wxyz": [1, 0, 0, 0],
        "frame": "camera_color_optical_frame", "stamp": 1.0, "source": "test"}))

    with pytest.raises(ValueError, match="프레임"):
        load_capture(path, expect_frame="base_link")


def test_a_capture_round_trips(tmp_path):
    pose = CupPose(position=(0.4, 0.2, 0.31), orientation_wxyz=(1, 0, 0, 0),
                   frame="base_link", stamp=12.5, source="fake_cup_pose_pub")
    path = tmp_path / "c.json"
    path.write_text(pose.to_json())

    loaded = load_capture(path, expect_frame="base_link")

    assert loaded == pose


def test_a_capture_without_a_source_is_refused(tmp_path):
    """무엇이 만든 pose 인지 모르면 사후에 해석할 수 없다 — 인지인지 합성인지."""
    path = tmp_path / "c.json"
    path.write_text(json.dumps({
        "position": [0.39, 0.19, 0.307], "orientation_wxyz": [1, 0, 0, 0],
        "frame": "base_link", "stamp": 1.0}))

    with pytest.raises(ValueError, match="source"):
        load_capture(path, expect_frame="base_link")


def test_the_verdict_reads_as_a_sentence(box):
    pose = CupPose(position=(0.50, 0.19, SPAWN_Z), orientation_wxyz=(1, 0, 0, 0),
                   frame="base_link", stamp=0.0, source="perception_plus_plus")

    text = verdict(pose, box).describe()

    assert "학습 분포 밖" in text
    assert "x" in text and "0.500" in text
