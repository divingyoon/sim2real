#!/usr/bin/env python3
"""reset_pose_core 테스트 — 출처가 갈릴 때 조용히 하나를 고르지 않는지가 요점."""
from __future__ import annotations

import numpy as np
import pytest

from reset_pose_core import (
    HOME_AGREEMENT_TOL_RAD,
    HomePose,
    describe_disagreements,
    disagreements,
    home_from_preset,
    home_from_recording,
    not_parked,
)

NAMES = ["l_aj_1", "l_aj_2"]


def test_a_recorded_home_is_used_verbatim_when_the_recording_states_it():
    npz = {"meta_joint_names": np.array(NAMES),
           "meta_home_q": np.array([0.1, 0.2]),
           "arm_meas": np.zeros((3, 1, 2))}
    home = home_from_recording(npz)
    assert home.joints == {"l_aj_1": 0.1, "l_aj_2": 0.2}
    assert not home.derived


def test_an_older_recording_derives_the_home_and_says_it_derived_it():
    npz = {"meta_joint_names": np.array(NAMES),
           "arm_meas": np.array([[[0.3, 0.4]], [[0.9, 0.9]]])}
    home = home_from_recording(npz)
    assert home.joints == {"l_aj_1": 0.3, "l_aj_2": 0.4}
    assert home.derived is True  # 유도된 값이라는 사실이 따라다녀야 한다


def test_a_recording_with_neither_refuses_rather_than_inventing_a_home():
    with pytest.raises(KeyError, match="지어내지 않는다"):
        home_from_recording({"meta_joint_names": np.array(NAMES)})


def test_a_home_q_of_the_wrong_length_is_refused():
    npz = {"meta_joint_names": np.array(NAMES), "meta_home_q": np.array([0.1])}
    with pytest.raises(ValueError, match="길이"):
        home_from_recording(npz)


def test_the_preset_home_is_read_without_importing_the_simulator(tmp_path):
    src = tmp_path / "p.py"
    src.write_text("import isaaclab_does_not_exist\nH = {'l_aj_1': 0.9, 'l_aj_2': -0.4}\n")
    home = home_from_preset(src, "H")
    assert home.joints == {"l_aj_1": 0.9, "l_aj_2": -0.4}


def test_a_missing_preset_constant_is_named():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        src = pathlib.Path(d) / "p.py"
        src.write_text("X = 1\n")
        with pytest.raises(KeyError, match="NOPE"):
            home_from_preset(src, "NOPE")


def test_two_homes_that_agree_report_nothing():
    a = HomePose({"l_aj_1": 0.5}, source="a")
    b = HomePose({"l_aj_1": 0.5 + HOME_AGREEMENT_TOL_RAD / 2}, source="b")
    assert disagreements(a, b) == []


def test_a_changed_home_is_reported_per_joint():
    a = HomePose({"l_aj_1": -0.009, "l_aj_2": -0.374}, source="기록")
    b = HomePose({"l_aj_1": 0.900, "l_aj_2": -0.376}, source="preset")
    rows = disagreements(a, b)
    assert [r.joint for r in rows] == ["l_aj_1"]
    assert rows[0].delta == pytest.approx(0.909, abs=1e-3)


def test_only_shared_joints_are_compared():
    a = HomePose({"l_aj_1": 0.0}, source="a")
    b = HomePose({"l_aj_1": 0.0, "head_j_pan": 1.0}, source="b")
    assert disagreements(a, b) == []


def test_a_robot_sitting_at_the_target_is_parked():
    assert not_parked(np.array([0.1, 0.2]), np.array([0.1, 0.2]), NAMES) == []


def test_a_robot_away_from_the_target_names_the_offending_joints():
    rows = not_parked(np.array([0.1, 0.9]), np.array([0.1, 0.2]), NAMES)
    assert [r.joint for r in rows] == ["l_aj_2"]


def test_mismatched_shapes_are_refused_rather_than_broadcast():
    with pytest.raises(ValueError, match="모양"):
        not_parked(np.array([0.1]), np.array([0.1, 0.2]), NAMES)


def test_the_description_of_no_disagreement_says_so():
    assert "일치" in describe_disagreements([], a_label="a", b_label="b")


def test_the_description_names_every_offending_joint():
    a = HomePose({"l_aj_1": 0.0, "l_aj_2": 0.0}, source="a")
    b = HomePose({"l_aj_1": 1.0, "l_aj_2": 1.0}, source="b")
    text = describe_disagreements(disagreements(a, b), a_label="기록", b_label="preset")
    assert "l_aj_1" in text and "l_aj_2" in text and "1000.0" in text
