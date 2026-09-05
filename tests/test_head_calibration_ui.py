from pathlib import Path

import pytest

from head_calibration_ui import (
    MotorCalibration,
    deg_to_tick,
    equivalent_signed_deg,
    format_deg_with_equivalent,
    motor_deg_from_ui_percent,
    parse_motor_ids,
    save_calibration_yaml,
    tick_to_deg,
    unwrap_calibrated_range,
    unwrap_target_for_current,
    unwrap_target_into_range,
)


def test_signed_absolute_encoder_degrees_map_to_ticks():
    assert deg_to_tick(-180.0) == 0
    assert deg_to_tick(-90.0) == 1024
    assert deg_to_tick(0.0) == 2048
    assert deg_to_tick(179.9) == 4094
    assert deg_to_tick(215.0) > 4095
    assert deg_to_tick(215.0) - deg_to_tick(-145.0) == 4095
    assert tick_to_deg(0) == pytest.approx(-180.0)
    assert tick_to_deg(4095) == pytest.approx(180.0)


def test_unwrapped_display_shows_signed_equivalent():
    assert equivalent_signed_deg(325.0) == pytest.approx(-35.0)
    assert format_deg_with_equivalent(325.0) == "325.0 (=-35.0)"
    assert format_deg_with_equivalent(-75.0) == "-75.0"


def test_limited_axis_maps_ui_percent_to_motor_degrees():
    calibration = MotorCalibration(
        name="tilt",
        dxl_id=2,
        min_deg=40.0,
        max_deg=140.0,
        inverted=False,
    )

    assert motor_deg_from_ui_percent(calibration, 0.0) == pytest.approx(40.0)
    assert motor_deg_from_ui_percent(calibration, 50.0) == pytest.approx(90.0)
    assert motor_deg_from_ui_percent(calibration, 100.0) == pytest.approx(140.0)


def test_inverted_axis_reverses_ui_percent_mapping():
    calibration = MotorCalibration(
        name="tilt",
        dxl_id=2,
        min_deg=40.0,
        max_deg=140.0,
        inverted=True,
    )

    assert motor_deg_from_ui_percent(calibration, 0.0) == pytest.approx(140.0)
    assert motor_deg_from_ui_percent(calibration, 100.0) == pytest.approx(40.0)


def test_ui_percent_can_cross_encoder_seam_with_min_greater_than_max():
    calibration = MotorCalibration(
        name="tilt",
        dxl_id=2,
        min_deg=90.0,
        max_deg=-90.0,
        inverted=False,
    )

    assert unwrap_calibrated_range(90.0, -90.0) == pytest.approx((90.0, 270.0))
    assert motor_deg_from_ui_percent(calibration, 0.0) == pytest.approx(90.0)
    assert motor_deg_from_ui_percent(calibration, 50.0) == pytest.approx(180.0)
    assert motor_deg_from_ui_percent(calibration, 100.0) == pytest.approx(270.0)
    assert unwrap_target_into_range(-90.0, 90.0, -90.0) == pytest.approx(270.0)
    assert unwrap_target_for_current(-90.0, 90.0, 90.0, -90.0) == pytest.approx(
        -270.0
    )


def test_parse_motor_ids_requires_two_unique_ids():
    assert parse_motor_ids("1,2") == (1, 2)
    with pytest.raises(ValueError):
        parse_motor_ids("1")
    with pytest.raises(ValueError):
        parse_motor_ids("1,1")


def test_save_calibration_yaml_writes_stable_file(tmp_path: Path):
    output = tmp_path / "head_dynamixel_calibration.yaml"
    save_calibration_yaml(
        output,
        port="/dev/ttyUSB0",
        baud=1_000_000,
        motors=[
            MotorCalibration("pan", 1, -90.0, 90.0, False),
            MotorCalibration("tilt", 2, -75.0, 145.0, True),
        ],
    )

    assert output.read_text() == (
        "port: /dev/ttyUSB0\n"
        "baud: 1000000\n"
        "motors:\n"
        "  pan:\n"
        "    id: 1\n"
        "    min_deg: -90.0\n"
        "    max_deg: 90.0\n"
        "    inverted: false\n"
        "  tilt:\n"
        "    id: 2\n"
        "    min_deg: -75.0\n"
        "    max_deg: 145.0\n"
        "    inverted: true\n"
    )
