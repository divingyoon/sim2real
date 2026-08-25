"""그림자 판정 도구 — 지연 검출과 정렬이 실제로 맞는가.

가장 중요한 것은 `lag_by_cross_correlation` 이다. 여기가 틀리면 "느려서 뒤처진 것"과
"덜 가서 뒤처진 것"을 못 가르고, 그 둘은 고치는 노브가 다르다. 그래서 **알려진 지연을
합성 신호에 넣어 되찾는지** 확인한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shadow_report import (  # noqa: E402
    lag_by_cross_correlation,
    quat_angle,
    stats,
)


@pytest.mark.parametrize("true_lag", [0, 1, 5, 12])
def test_a_known_delay_is_recovered(true_lag):
    t = np.linspace(0, 8 * np.pi, 600)
    command = np.sin(t) + 0.3 * np.sin(2.7 * t)
    measured = np.roll(command, true_lag)

    assert lag_by_cross_correlation(command, measured, max_lag=30) == true_lag


def test_a_delay_is_still_found_under_static_offset_and_gain():
    """중력 처짐은 오프셋으로, 게인 부족은 진폭 축소로 온다 — 지연 추정이 흔들리면 안 된다."""
    t = np.linspace(0, 8 * np.pi, 600)
    command = np.sin(t)
    measured = 0.6 * np.roll(command, 7) - 0.25

    assert lag_by_cross_correlation(command, measured, max_lag=30) == 7


def test_a_delay_is_still_found_under_noise():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 8 * np.pi, 900)
    command = np.sin(t) + 0.4 * np.sin(3.1 * t)
    measured = np.roll(command, 4) + rng.normal(0, 0.03, command.size)

    assert lag_by_cross_correlation(command, measured, max_lag=30) == 4


def test_a_motionless_signal_reports_zero_rather_than_a_spurious_lag():
    """움직이지 않는 신호에서 상관은 무의미하다 — 지어낸 지연을 내놓지 않는다."""
    command = np.full(400, 0.31)
    measured = np.full(400, 0.29)

    assert lag_by_cross_correlation(command, measured, max_lag=30) == 0


def test_a_delay_beyond_the_search_window_is_not_invented():
    t = np.linspace(0, 8 * np.pi, 600)
    command = np.sin(t)
    measured = np.roll(command, 25)

    found = lag_by_cross_correlation(command, measured, max_lag=10)

    assert 0 <= found <= 10


def test_identical_quaternions_are_zero_degrees_apart():
    q = np.array([[0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]])

    assert quat_angle(q, q) == pytest.approx(np.zeros(2), abs=1e-6)


def test_a_negated_quaternion_is_the_same_rotation():
    """q 와 −q 는 같은 자세다 — 부호 때문에 180° 를 보고하면 안 된다."""
    q = np.array([[0.0, 0.0, 0.3826834, 0.9238795]])

    assert quat_angle(q, -q) == pytest.approx(np.zeros(1), abs=1e-6)


def test_a_ninety_degree_rotation_reads_as_ninety_degrees():
    a = np.array([[0.0, 0.0, 0.0, 1.0]])
    b = np.array([[0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]])

    assert quat_angle(a, b) == pytest.approx(np.array([90.0]), abs=1e-4)


def test_stats_report_the_tail_not_only_the_mean():
    values = np.concatenate([np.zeros(99), [100.0]])

    result = stats(values)

    assert result["mean"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(100.0)
    assert result["p95"] < result["max"]
