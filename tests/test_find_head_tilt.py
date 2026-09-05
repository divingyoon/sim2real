import pytest

from find_head_tilt import (
    SweepRow,
    generate_tilt_candidates,
    midpoint,
    select_best,
)


def test_midpoint_pan_and_tilt():
    # head_v1 기본 자세: pan -180..0 -> -90, tilt 135..325 -> 230
    assert midpoint(-180.0, 0.0) == pytest.approx(-90.0)
    assert midpoint(135.0, 325.0) == pytest.approx(230.0)


def test_midpoint_across_seam():
    # end < start 이면 방향성 arc 로 unwrap (350..10 => 350..370, 중앙 360)
    assert midpoint(350.0, 10.0) == pytest.approx(360.0)


def test_candidates_inclusive_endpoints():
    cands = generate_tilt_candidates(135.0, 325.0, 10.0)
    assert cands[0] == pytest.approx(135.0)
    assert cands[-1] == pytest.approx(325.0)
    assert len(cands) == 20  # 135..325 step 10 => 20점


def test_candidates_non_divisible_appends_end():
    cands = generate_tilt_candidates(0.0, 25.0, 10.0)
    assert cands == [0.0, 10.0, 20.0, 25.0]


def test_candidates_reversed_start_end():
    assert generate_tilt_candidates(325.0, 135.0, 10.0)[0] == pytest.approx(135.0)


def test_candidates_never_exceed_max_for_odd_steps():
    # tilt 135..325 에서 12/13/15 스텝이 예전엔 327/330 을 만들었음(범위 밖)
    for step in (12.0, 13.0, 15.0):
        cands = generate_tilt_candidates(135.0, 325.0, step)
        assert cands[-1] == pytest.approx(325.0)
        assert all(c <= 325.0 + 1e-9 for c in cands)


def test_candidates_step_must_be_positive():
    for bad in (0.0, -5.0):
        with pytest.raises(ValueError):
            generate_tilt_candidates(0.0, 10.0, bad)


def test_select_best_picks_max_corners():
    rows = [SweepRow(200.0, 12, 0.4), SweepRow(210.0, 20, 0.9), SweepRow(220.0, 6, 0.1)]
    best = select_best(rows, min_corners=8)
    assert best.tilt_deg == pytest.approx(210.0)


def test_select_best_tiebreak_lowest_reproj():
    rows = [SweepRow(200.0, 15, 0.8), SweepRow(210.0, 15, 0.3)]
    best = select_best(rows, min_corners=8)
    assert best.tilt_deg == pytest.approx(210.0)


def test_select_best_none_when_below_threshold_or_failed():
    rows = [SweepRow(200.0, 4, 0.2), SweepRow(210.0, 30, None)]
    assert select_best(rows, min_corners=8) is None
