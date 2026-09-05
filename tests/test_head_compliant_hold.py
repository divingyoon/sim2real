"""head_compliant_hold 의 하드웨어 무관 로직 테스트."""

import pytest

from head_compliant_hold import (
    ACTION_FOLLOW,
    ACTION_HOLD,
    ACTION_LATCH,
    ADDR_CURRENT_LIMIT,
    ADDR_GOAL_CURRENT,
    ADDR_PRESENT_CURRENT,
    ADDR_PRESENT_TEMPERATURE,
    ADDR_PRESENT_VELOCITY,
    CURRENT_UNIT_MA,
    GOAL_CURRENT_HARD_CAP_MA,
    OP_MODE_CURRENT_POSITION,
    FollowConfig,
    FollowState,
    clamp_follow_step,
    deg_per_s_to_velocity_raw,
    follow_decision,
    ma_to_raw,
    raw_to_ma,
    sags_under_gravity,
    travel_exceeded,
    validate_goal_current,
)


# ---------- 제어 테이블 상수 (XC330-M288 model 파일 근거) ----------

def test_control_table_addresses():
    assert ADDR_CURRENT_LIMIT == 38
    assert ADDR_GOAL_CURRENT == 102
    assert ADDR_PRESENT_CURRENT == 126
    assert ADDR_PRESENT_VELOCITY == 128
    assert ADDR_PRESENT_TEMPERATURE == 146
    assert OP_MODE_CURRENT_POSITION == 5


# ---------- 전류 변환·검증 ----------

def test_current_roundtrip():
    for ma in (1.0, 20.0, 150.0):
        assert raw_to_ma(ma_to_raw(ma)) == pytest.approx(ma, abs=CURRENT_UNIT_MA)


def test_goal_current_must_be_positive():
    with pytest.raises(ValueError):
        validate_goal_current(0.0)
    with pytest.raises(ValueError):
        validate_goal_current(-5.0)


def test_goal_current_rejects_above_hard_cap():
    """상한을 clamp 하지 않고 거부한다 — 조용히 낮추면 사용자가 모른다."""
    with pytest.raises(ValueError):
        validate_goal_current(GOAL_CURRENT_HARD_CAP_MA + 1.0)


def test_goal_current_accepts_within_cap():
    assert validate_goal_current(GOAL_CURRENT_HARD_CAP_MA) == GOAL_CURRENT_HARD_CAP_MA
    assert validate_goal_current(20.0) == 20.0


# ---------- 속도 변환 ----------

def test_velocity_unit_matches_model_file():
    """0.0239691227 rad/s per raw = 1.3733 deg/s per raw."""
    assert deg_per_s_to_velocity_raw(1.3733) == pytest.approx(1.0, abs=0.01)


# ---------- 추종 상태기계 ----------

CFG = FollowConfig(
    deadband_tick=10,
    vel_threshold_raw=3,
    still_cycles_needed=5,
    max_step_tick=50,
)


def test_hold_when_still_and_undisplaced():
    state = FollowState(goal_tick=2048, still_cycles=0, action=ACTION_HOLD)
    out = follow_decision(state, present_tick=2050, velocity_raw=0, config=CFG)
    assert out.action == ACTION_HOLD
    assert out.goal_tick == 2048          # 목표는 그대로
    assert out.still_cycles == 0


def test_follow_when_hand_is_moving_it():
    state = FollowState(goal_tick=2048, still_cycles=0, action=ACTION_HOLD)
    out = follow_decision(state, present_tick=2080, velocity_raw=20, config=CFG)
    assert out.action == ACTION_FOLLOW
    assert out.goal_tick == 2080          # 목표가 따라간다
    assert out.still_cycles == 0


def test_displaced_but_stopped_counts_toward_latch():
    state = FollowState(goal_tick=2048, still_cycles=0, action=ACTION_FOLLOW)
    out = follow_decision(state, present_tick=2080, velocity_raw=0, config=CFG)
    assert out.action == ACTION_HOLD
    assert out.goal_tick == 2048          # 아직 확정 전 — 목표 유지
    assert out.still_cycles == 1


def test_latch_after_enough_still_cycles():
    state = FollowState(goal_tick=2048, still_cycles=4, action=ACTION_HOLD)
    out = follow_decision(state, present_tick=2080, velocity_raw=0, config=CFG)
    assert out.action == ACTION_LATCH
    assert out.goal_tick == 2080          # 손 뗀 자리로 확정
    assert out.still_cycles == 0


def test_inside_deadband_resets_still_counter():
    state = FollowState(goal_tick=2048, still_cycles=3, action=ACTION_HOLD)
    out = follow_decision(state, present_tick=2052, velocity_raw=0, config=CFG)
    assert out.action == ACTION_HOLD
    assert out.still_cycles == 0


def test_decision_never_mutates_input_state():
    state = FollowState(goal_tick=2048, still_cycles=0, action=ACTION_HOLD)
    follow_decision(state, present_tick=2080, velocity_raw=20, config=CFG)
    assert state.goal_tick == 2048 and state.still_cycles == 0


# ---------- 스텝 제한 ----------

def test_large_jump_is_rate_limited():
    """통신 글리치나 급격한 당김이 목표를 순간이동시키지 못한다."""
    assert clamp_follow_step(goal_tick=2048, present_tick=3000, max_step_tick=50) == 2098


def test_large_negative_jump_is_rate_limited():
    assert clamp_follow_step(goal_tick=2048, present_tick=1000, max_step_tick=50) == 1998


def test_small_move_passes_through():
    assert clamp_follow_step(goal_tick=2048, present_tick=2070, max_step_tick=50) == 2070


# ---------- 안전 판정 ----------

def test_travel_limit():
    assert not travel_exceeded(present_tick=2200, origin_tick=2048, max_travel_tick=200)
    assert travel_exceeded(present_tick=2300, origin_tick=2048, max_travel_tick=200)
    assert travel_exceeded(present_tick=1800, origin_tick=2048, max_travel_tick=200)


def test_gravity_sag_detection():
    assert not sags_under_gravity(displacement_tick=5, limit_tick=20)
    assert sags_under_gravity(displacement_tick=-40, limit_tick=20)


# ---------- 부호 변환 (Present Current/Velocity 는 signed) ----------

def test_signed_conversion_2byte():
    from head_compliant_hold import to_signed
    assert to_signed(0, 16) == 0
    assert to_signed(100, 16) == 100
    assert to_signed(0xFFFF, 16) == -1
    assert to_signed(0xFF9C, 16) == -100


def test_signed_conversion_4byte():
    from head_compliant_hold import to_signed
    assert to_signed(0xFFFFFFFF, 32) == -1
    assert to_signed(1000, 32) == 1000


# ---------- 포트 자동탐지 ----------

def test_autodetect_single_port():
    from head_compliant_hold import autodetect_port
    assert autodetect_port(["/dev/ttyUSB0"]) == "/dev/ttyUSB0"


def test_autodetect_refuses_when_ambiguous():
    from head_compliant_hold import autodetect_port
    with pytest.raises(RuntimeError, match="여러"):
        autodetect_port(["/dev/ttyUSB0", "/dev/ttyACM0"])


def test_autodetect_refuses_when_none():
    from head_compliant_hold import autodetect_port
    with pytest.raises(RuntimeError, match="없"):
        autodetect_port([])


# ---------- 주기 보정 ----------

def test_sleep_subtracts_work_time():
    """읽기에 쓴 시간을 빼야 설정한 Hz 가 실제 Hz 가 된다."""
    from head_compliant_hold import remaining_sleep
    assert remaining_sleep(period=0.02, elapsed=0.005) == pytest.approx(0.015)


def test_sleep_never_negative_when_overrun():
    from head_compliant_hold import remaining_sleep
    assert remaining_sleep(period=0.02, elapsed=0.05) == 0.0


# ---------- id 파싱 (재사용한 head_position_hold_node.parse_ids 는 2개를 강제한다) ----------

def test_parse_ids_accepts_single_motor():
    """실기 head 버스에 모터가 하나만 있는 경우가 있다."""
    from head_compliant_hold import parse_motor_ids
    assert parse_motor_ids("1") == (1,)


def test_parse_ids_accepts_many():
    from head_compliant_hold import parse_motor_ids
    assert parse_motor_ids("1,2,3") == (1, 2, 3)
    assert parse_motor_ids(" 1 , 2 ") == (1, 2)


def test_parse_ids_rejects_empty_and_duplicates():
    from head_compliant_hold import parse_motor_ids
    with pytest.raises(ValueError):
        parse_motor_ids("")
    with pytest.raises(ValueError, match="중복"):
        parse_motor_ids("1,1")


def test_parse_ids_rejects_out_of_range():
    from head_compliant_hold import parse_motor_ids
    with pytest.raises(ValueError):
        parse_motor_ids("253")
    with pytest.raises(ValueError):
        parse_motor_ids("-1")


# ---------- 토크 환산 (mA 는 사람이 판단할 수 없는 숫자다) ----------

def test_current_to_torque_uses_stall_spec():
    """XC330-M288-T: 0.93 N·m @ 1.8 A → 0.000517 N·m/mA."""
    from head_compliant_hold import ma_to_torque_nm
    assert ma_to_torque_nm(1800.0) == pytest.approx(0.93, abs=0.01)
    assert ma_to_torque_nm(150.0) == pytest.approx(0.0775, abs=0.001)


def test_torque_to_hand_force_at_lever():
    """손으로 이길 수 있는지는 N·m 이 아니라 지렛대 끝 N 으로 판단한다."""
    from head_compliant_hold import torque_to_force_n
    assert torque_to_force_n(0.0775, lever_m=0.05) == pytest.approx(1.55, abs=0.01)


def test_hard_cap_stays_hand_overcomeable():
    """상한에서도 5 cm 지렛대 기준 손으로 이길 수 있어야 한다 (< 5 N)."""
    from head_compliant_hold import (GOAL_CURRENT_HARD_CAP_MA, ma_to_torque_nm,
                                     torque_to_force_n)
    assert torque_to_force_n(ma_to_torque_nm(GOAL_CURRENT_HARD_CAP_MA), 0.05) < 5.0


# ---------- 최소 유지 전류 선택 ----------

def test_picks_lowest_current_that_holds():
    from head_compliant_hold import select_min_holding
    results = [(10.0, 45), (20.0, 30), (40.0, 5), (80.0, 2)]   # (mA, 처짐 tick)
    assert select_min_holding(results, limit_tick=20) == 40.0


def test_returns_none_when_nothing_holds():
    from head_compliant_hold import select_min_holding
    assert select_min_holding([(10.0, 90), (20.0, 70)], limit_tick=20) is None


def test_ignores_a_lucky_low_value_below_a_failure():
    """낮은 값이 우연히 통과해도, 그 위에 실패가 있으면 신뢰할 수 없다."""
    from head_compliant_hold import select_min_holding
    results = [(10.0, 2), (20.0, 60), (40.0, 3)]
    assert select_min_holding(results, limit_tick=20) == 40.0


def test_position_gain_addresses():
    from head_compliant_hold import (ADDR_POSITION_D_GAIN, ADDR_POSITION_I_GAIN,
                                     ADDR_POSITION_P_GAIN)
    assert (ADDR_POSITION_D_GAIN, ADDR_POSITION_I_GAIN, ADDR_POSITION_P_GAIN) == (80, 82, 84)
