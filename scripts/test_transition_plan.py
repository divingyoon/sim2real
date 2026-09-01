#!/usr/bin/env python3
"""`transition_plan` 순수 로직 테스트. sim·로봇 불필요."""

from __future__ import annotations

import numpy as np
import pytest

from transition_plan import (
    contact_set,
    describe_transition,
    new_contacts,
    ramp,
    steps_for,
)


# ── 램프 ───────────────────────────────────────────────────────────────────
def test_steps_for_is_set_by_the_joint_that_moves_farthest():
    assert steps_for([0.0, 0.0], [1.0, 0.2], max_vel=0.5, dt=0.1) == 20


def test_steps_for_is_at_least_one_even_when_nothing_moves():
    assert steps_for([0.3], [0.3], max_vel=0.5, dt=0.1) == 1


def test_ramp_starts_at_the_start_and_ends_at_the_goal():
    path = ramp([0.0, 1.0], [1.0, 1.0], max_vel=0.5, dt=0.1)

    assert path[0] == pytest.approx([0.0, 1.0])
    assert path[-1] == pytest.approx([1.0, 1.0])


def test_ramp_never_exceeds_the_velocity_limit():
    path = ramp([0.0, 0.0], [2.0, -1.0], max_vel=0.5, dt=0.1)

    assert (np.abs(np.diff(path, axis=0)) / 0.1).max() <= 0.5 + 1e-9


def test_ramp_moves_every_joint_monotonically():
    """직선 보간이다 — 중간에 되돌아가면 궤적이 잘못 만들어진 것이다."""
    path = ramp([0.0, 1.0], [1.0, -1.0], max_vel=0.5, dt=0.1)

    assert (np.diff(path[:, 0]) >= -1e-12).all()
    assert (np.diff(path[:, 1]) <= 1e-12).all()


def test_ramp_of_a_zero_move_is_a_single_frame():
    assert ramp([0.5], [0.5], max_vel=0.5, dt=0.1).shape == (1, 1)


def test_ramp_refuses_a_shape_mismatch():
    with pytest.raises(ValueError, match="길이"):
        ramp([0.0, 0.0], [1.0], max_vel=0.5, dt=0.1)


def test_ramp_refuses_a_nonpositive_velocity():
    with pytest.raises(ValueError, match="속도"):
        ramp([0.0], [1.0], max_vel=0.0, dt=0.1)


# ── 접촉 판정 ──────────────────────────────────────────────────────────────
NAMES = ["base", "l_link", "r_link", "table_pad"]


def test_contact_set_keeps_only_bodies_above_the_threshold():
    forces = np.array([0.0, 5.0, 0.05, 2.0])

    assert contact_set(NAMES, forces, threshold=1.0) == frozenset({"l_link", "table_pad"})


def test_contact_set_of_a_quiet_robot_is_empty():
    assert contact_set(NAMES, np.zeros(4), threshold=1.0) == frozenset()


def test_contact_set_refuses_a_length_mismatch():
    with pytest.raises(ValueError, match="개수"):
        contact_set(NAMES, np.zeros(3), threshold=1.0)


def test_new_contacts_ignores_what_was_already_touching():
    """시작 자세에서 이미 닿아 있던 것(파지한 컵 등)은 새 충돌이 아니다."""
    baseline = frozenset({"r_link"})

    assert new_contacts(baseline, frozenset({"r_link", "l_link"})) == frozenset({"l_link"})


def test_new_contacts_is_empty_when_nothing_is_added():
    assert new_contacts(frozenset({"a"}), frozenset({"a"})) == frozenset()


def test_new_contacts_does_not_report_a_contact_that_disappeared():
    assert new_contacts(frozenset({"a", "b"}), frozenset({"a"})) == frozenset()


# ── 보고 ───────────────────────────────────────────────────────────────────
def test_describe_transition_passes_when_nothing_new_touches():
    text = describe_transition("우팔", ramp([0.0], [1.0], max_vel=0.5, dt=0.1),
                               dt=0.1, worst={}, min_z={"link": 0.31}, table_z=0.2)

    assert "통과" in text


def test_describe_transition_names_the_bodies_that_newly_touch():
    text = describe_transition("우팔", ramp([0.0], [1.0], max_vel=0.5, dt=0.1),
                               dt=0.1, worst={"r_hl_palm": (12, 34.5)},
                               min_z={"r_hl_palm": 0.25}, table_z=0.2)

    assert "r_hl_palm" in text
    assert "34.5" in text
    assert "통과" not in text


def test_describe_transition_flags_a_link_below_the_table():
    text = describe_transition("좌팔", ramp([0.0], [0.5], max_vel=0.5, dt=0.1),
                               dt=0.1, worst={}, min_z={"l_link": 0.18}, table_z=0.2)

    assert "작업면" in text
    assert "l_link" in text


def test_describe_transition_reports_how_long_the_move_takes():
    path = ramp([0.0], [1.0], max_vel=0.5, dt=0.1)

    text = describe_transition("우팔", path, dt=0.1, worst={}, min_z={}, table_z=0.2)

    assert "2.0 s" in text
