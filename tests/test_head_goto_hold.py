import math

import pytest

from head_goto_hold import (
    rad_to_tick, tick_to_rad, clamp_tick, to_rad, validate_within_limits, parse_ids,
    RES_RAD_PER_TICK, OFFSET_RAD, TICK_MAX, PAN_LIMIT_RAD, TILT_LIMIT_RAD,
    ADDR_OPERATING_MODE, ADDR_TORQUE_ENABLE,
    ADDR_GOAL_POSITION, ADDR_PRESENT_POSITION, OP_MODE_POSITION,
)


def test_center_tick_is_zero_rad():
    assert rad_to_tick(0.0) == 2048
    assert tick_to_rad(2048) == pytest.approx(0.0, abs=RES_RAD_PER_TICK)


def test_full_range_endpoints():
    assert rad_to_tick(OFFSET_RAD) == 0                 # -π → tick 0
    assert rad_to_tick(-OFFSET_RAD) == TICK_MAX         # +π → clamp 4095


def test_roundtrip_within_resolution():
    for deg in (-90, -30, 0, 15, 30, 89):
        r = math.radians(deg)
        assert tick_to_rad(rad_to_tick(r)) == pytest.approx(r, abs=RES_RAD_PER_TICK)


def test_clamp_bounds():
    assert clamp_tick(-100) == 0
    assert clamp_tick(999999) == TICK_MAX
    assert clamp_tick(2048) == 2048


def test_tilt_30deg_maps_above_center():
    # tilt=+30° → 중앙(2048)보다 큰 tick (부호 규약 sanity)
    assert rad_to_tick(math.radians(30)) > 2048


def test_deg_is_default_unit():
    assert to_rad(30.0, use_deg=True) == pytest.approx(math.radians(30))
    assert to_rad(0.5236, use_deg=False) == pytest.approx(0.5236)


def test_calibration_pose_within_limits():
    # 캘리브 기본 자세 pan=0°, tilt=30° 는 한계 안
    validate_within_limits(0.0, math.radians(30))


def test_reject_raw_30_interpreted_as_radians():
    # ★CRITICAL 회귀 가드: 30을 라디안으로 해석(=1718°)하면 반드시 거부
    with pytest.raises(ValueError):
        validate_within_limits(0.0, 30.0)
    with pytest.raises(ValueError):
        validate_within_limits(30.0, 0.0)


def test_limits_are_90deg():
    assert PAN_LIMIT_RAD == pytest.approx(math.radians(90), abs=1e-4)
    assert TILT_LIMIT_RAD == pytest.approx(math.radians(90), abs=1e-4)


def test_parse_ids():
    assert parse_ids("1,2") == (1, 2)
    for bad in ("1", "1,1", "1,2,3", "a,b"):
        with pytest.raises(ValueError):
            parse_ids(bad)


def test_control_table_addresses():
    # XC330 컨트롤 테이블 (repo/dynamixel_hardware_interface xc330_m288.model 근거)
    assert (ADDR_OPERATING_MODE, ADDR_TORQUE_ENABLE) == (11, 64)
    assert (ADDR_GOAL_POSITION, ADDR_PRESENT_POSITION) == (116, 132)
    assert OP_MODE_POSITION == 3
