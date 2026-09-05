import pytest

from head_position_hold_node import (
    MotorCalibration,
    TICK_MAX,
    deg_to_tick,
    load_calibration_yaml,
    motor_deg_from_center_offset,
    parse_angle_targets,
    parse_ids,
    tick_to_deg,
    unwrap_calibrated_range,
    unwrap_target_for_current,
)


def test_signed_absolute_degrees_map_to_encoder_ticks():
    assert deg_to_tick(-180.0) == 0
    assert deg_to_tick(-90.0) == 1024
    assert deg_to_tick(0.0) == 2048
    assert deg_to_tick(90.0) == 3071
    assert deg_to_tick(179.9) == 4094


def test_unwrapped_degrees_wrap_to_same_single_turn_tick():
    assert deg_to_tick(215.0) > TICK_MAX
    assert deg_to_tick(215.0) - deg_to_tick(-145.0) == TICK_MAX


def test_tick_to_degrees_uses_absolute_encoder_frame():
    assert tick_to_deg(0) == pytest.approx(-180.0)
    assert tick_to_deg(2048) == pytest.approx(0.04, abs=0.05)
    assert tick_to_deg(TICK_MAX) == pytest.approx(180.0)


def test_parse_ids_requires_unique_motor_ids():
    assert parse_ids("1,2") == (1, 2)
    with pytest.raises(ValueError):
        parse_ids("1")
    with pytest.raises(ValueError):
        parse_ids("1,1")


def test_parse_angle_targets_pairs_ids_with_absolute_degrees():
    assert parse_angle_targets((1, 2), "-45,215") == {1: -45.0, 2: 215.0}
    with pytest.raises(ValueError):
        parse_angle_targets((1, 2), "45")


def test_calibration_offsets_are_signed_absolute_targets():
    pan = MotorCalibration("pan", 1, -45.0, 45.0, False)
    tilt = MotorCalibration("tilt", 2, -75.0, 145.0, False)

    assert motor_deg_from_center_offset(pan, 0.0) == pytest.approx(0.0)
    assert motor_deg_from_center_offset(pan, -30.0) == pytest.approx(-30.0)
    assert motor_deg_from_center_offset(pan, 30.0) == pytest.approx(30.0)
    assert motor_deg_from_center_offset(tilt, 145.0) == pytest.approx(145.0)


def test_calibration_offset_ignores_inverted_for_absolute_targets():
    calibration = MotorCalibration("tilt", 2, -75.0, 145.0, True)

    assert motor_deg_from_center_offset(calibration, 30.0) == pytest.approx(30.0)
    assert motor_deg_from_center_offset(calibration, -30.0) == pytest.approx(-30.0)


def test_center_offset_rejects_commands_outside_calibrated_range():
    calibration = MotorCalibration("pan", 1, -45.0, 45.0, False)

    with pytest.raises(ValueError):
        motor_deg_from_center_offset(calibration, 46.0)
    with pytest.raises(ValueError):
        motor_deg_from_center_offset(calibration, -46.0)


def test_calibration_range_can_cross_encoder_seam_when_unwrapped():
    calibration = MotorCalibration("tilt", 2, 145.0, 215.0, False)

    assert motor_deg_from_center_offset(calibration, 145.0) == pytest.approx(145.0)
    assert motor_deg_from_center_offset(calibration, 215.0) == pytest.approx(215.0)
    assert deg_to_tick(215.0) > TICK_MAX
    assert deg_to_tick(215.0) - deg_to_tick(-145.0) == TICK_MAX


def test_min_greater_than_max_means_directed_wrap_across_encoder_seam():
    calibration = MotorCalibration("tilt", 2, 90.0, -90.0, False)

    assert unwrap_calibrated_range(90.0, -90.0) == pytest.approx((90.0, 270.0))
    assert motor_deg_from_center_offset(calibration, 90.0) == pytest.approx(90.0)
    assert motor_deg_from_center_offset(calibration, 180.0) == pytest.approx(180.0)
    assert motor_deg_from_center_offset(calibration, -90.0) == pytest.approx(270.0)
    assert deg_to_tick(270.0) > deg_to_tick(180.0)


def test_target_uses_current_range_copy_instead_of_nearest_angle():
    assert unwrap_target_for_current(
        current_deg=-90.0,
        target_deg=90.0,
        min_deg=90.0,
        max_deg=-90.0,
    ) == pytest.approx(-270.0)
    assert unwrap_target_for_current(
        current_deg=120.0,
        target_deg=-100.0,
        min_deg=-120.0,
        max_deg=120.0,
    ) == pytest.approx(-100.0)


def test_load_calibration_yaml_reads_saved_ui_format(tmp_path):
    path = tmp_path / "head_dynamixel_calibration.yaml"
    path.write_text(
        "port: /dev/ttyUSB0\n"
        "baud: 1000000\n"
        "motors:\n"
        "  pan:\n"
        "    id: 1\n"
        "    min_deg: 45.0\n"
        "    max_deg: 135.0\n"
        "    inverted: false\n"
        "  tilt:\n"
        "    id: 2\n"
        "    min_deg: -75.0\n"
        "    max_deg: 145.0\n"
        "    inverted: true\n"
    )

    config = load_calibration_yaml(path)

    assert config.port == "/dev/ttyUSB0"
    assert config.baud == 1_000_000
    assert config.motors == {
        "pan": MotorCalibration("pan", 1, 45.0, 135.0, False),
        "tilt": MotorCalibration("tilt", 2, -75.0, 145.0, True),
    }
