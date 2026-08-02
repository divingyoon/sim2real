import math

from joint_monitor_core import (
    JointSample,
    classify_joint,
    csv_header,
    csv_row,
    format_dashboard,
    group_records,
)


def test_classify_source_names():
    assert classify_joint("openarm_right_joint4") == "right_arm"
    assert classify_joint("openarm_left_joint1") == "left_arm"
    assert classify_joint("rj_dg_1_2") == "tesollo_right"
    assert classify_joint("openarm_left_finger_joint1") == "gripper"


def test_classify_canonical_names():
    assert classify_joint("r_aj_7") == "right_arm"
    assert classify_joint("l_aj_3") == "left_arm"
    assert classify_joint("r_hj_thumb_2") == "tesollo_right"
    assert classify_joint("l_hj_gripper_1") == "gripper"


def test_classify_unknown_is_other():
    assert classify_joint("some_random_joint") == "other"


def test_group_records_sorted_within_group():
    recs = {
        "openarm_right_joint2": JointSample(0.2, 0, 0),
        "openarm_right_joint1": JointSample(0.1, 0, 0),
        "rj_dg_1_1": JointSample(0.0, 0, 0),
    }
    grouped = group_records(recs)
    right = [n for n, _ in grouped["right_arm"]]
    assert right == ["openarm_right_joint1", "openarm_right_joint2"]
    assert len(grouped["tesollo_right"]) == 1


def test_dashboard_flags_overtorque():
    recs = {
        "openarm_right_joint7": JointSample(0.8, 0.0, 9.5),   # 과부하
        "openarm_right_joint1": JointSample(0.0, 0.0, 0.2),
    }
    out = format_dashboard(recs, elapsed_sec=1.23, effort_warn=5.0)
    assert "RIGHT ARM" in out
    # j7 줄에 경보 '*', j1 줄엔 없음
    j7_line = [ln for ln in out.splitlines() if "joint7" in ln][0]
    j1_line = [ln for ln in out.splitlines() if "joint1" in ln][0]
    assert j7_line.rstrip().endswith("*")
    assert not j1_line.rstrip().endswith("*")


def test_csv_header_and_row_order():
    order = ["r_aj_1", "r_aj_7"]
    assert csv_header(order) == [
        "t_sec", "r_aj_1.pos", "r_aj_1.vel", "r_aj_1.eff",
        "r_aj_7.pos", "r_aj_7.vel", "r_aj_7.eff",
    ]
    recs = {"r_aj_1": JointSample(0.1, 0.2, 0.3)}   # r_aj_7 없음 → NaN
    row = csv_row(2.5, recs, order)
    assert row[0] == 2.5
    assert row[1:4] == [0.1, 0.2, 0.3]
    assert all(math.isnan(v) for v in row[4:7])
