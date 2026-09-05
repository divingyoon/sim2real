"""그림자 재생기의 순수 코어 — 무엇을 언제 보낼지.

ROS 도 로봇도 없이 검증한다. 여기서 지켜야 할 것은 세 가지다:
  · 첫 프레임으로 **순간이동시키지 않는 것** — 실측 자세에서 램프로 들어간다
  · 결손 프레임을 **보간하지 않고 거부하는 것** — 없는 명령을 지어내면 로봇이 그걸 실행한다
  · `--rate-scale` 이 시각만 늘리고 **경로는 건드리지 않는 것**
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from shadow_replay_core import (  # noqa: E402
    ReplayPlan,
    approach_ramp,
    frame_schedule,
)

PARK_SPEED = 0.1     # rad/s — robotctl 이 쓰는 것과 같은 페이싱


def test_the_ramp_starts_where_the_arm_actually_is():
    start = np.array([0.0, 0.1, 0.0, 0.5, 0.0, 0.0, 0.0])
    first = np.array([0.2, 0.1, 0.0, 0.5, 0.0, 0.0, 0.0])

    ramp = approach_ramp(start, first, speed=PARK_SPEED, dt=1 / 60)

    assert ramp[0] == pytest.approx(start)
    assert ramp[-1] == pytest.approx(first)


def test_the_ramp_never_exceeds_the_park_speed():
    start = np.zeros(7)
    first = np.array([1.5, -1.0, 0.3, 0.9, 0.0, 0.0, -0.4])
    dt = 1 / 60

    ramp = approach_ramp(start, first, speed=PARK_SPEED, dt=dt)

    step = np.abs(np.diff(ramp, axis=0)).max()
    assert step <= PARK_SPEED * dt + 1e-12


def test_a_ramp_to_where_we_already_are_is_a_single_frame():
    q = np.array([0.1] * 7)

    ramp = approach_ramp(q, q.copy(), speed=PARK_SPEED, dt=1 / 60)

    assert ramp.shape == (1, 7)


def test_rate_scale_stretches_time_without_touching_the_path():
    dt = 1 / 60
    schedule_fast = frame_schedule(n_frames=100, step_dt=dt, rate_scale=1.0)
    schedule_slow = frame_schedule(n_frames=100, step_dt=dt, rate_scale=0.25)

    assert len(schedule_fast) == len(schedule_slow) == 100
    assert schedule_slow[-1] == pytest.approx(schedule_fast[-1] * 4.0)
    assert np.allclose(np.diff(schedule_slow), dt / 0.25)


def test_a_rate_scale_above_one_is_refused():
    """sim 보다 빨리 재생할 이유가 없고, 그러면 안전 한계 밖으로 나간다."""
    with pytest.raises(ValueError, match="rate_scale"):
        frame_schedule(n_frames=10, step_dt=1 / 60, rate_scale=1.5)


def _plan(**kw):
    """arm_target 만 주면 그리퍼 프레임 수를 맞춰 준다 — 불일치를 보는 테스트는 명시한다."""
    arm = kw.get("arm_target", np.zeros((5, 7)))
    defaults = dict(
        arm_target=arm,
        grip_target=np.zeros(len(arm)),
        step_dt=1 / 60,
        rate_scale=1.0,
        joint_names=[f"l_aj_{i}" for i in range(1, 8)],
        gripper_name="l_hj_gripper_1",
    )
    defaults.update(kw)
    return ReplayPlan(**defaults)


def test_a_non_finite_frame_is_refused_rather_than_replayed():
    target = np.zeros((5, 7))
    target[2, 3] = np.nan

    with pytest.raises(ValueError, match="유한하지 않"):
        _plan(arm_target=target)


def test_a_frame_count_mismatch_between_arm_and_gripper_is_refused():
    with pytest.raises(ValueError, match="프레임 수"):
        _plan(arm_target=np.zeros((5, 7)), grip_target=np.zeros(4))


def test_a_joint_count_that_does_not_match_the_names_is_refused():
    with pytest.raises(ValueError, match="관절 수"):
        _plan(arm_target=np.zeros((5, 6)))


def test_the_plan_reports_what_it_will_demand_of_the_arm():
    """재생 전에 요구 속도를 알아야 한다 — 실기 한계와 대볼 유일한 수치다."""
    dt = 1 / 60
    target = np.zeros((3, 7))
    target[1, 0] = 0.01
    target[2, 0] = 0.02

    plan = _plan(arm_target=target, step_dt=dt, rate_scale=1.0)

    assert plan.peak_joint_speed == pytest.approx(0.01 / dt)


def test_slowing_the_replay_lowers_what_it_demands():
    dt = 1 / 60
    target = np.zeros((3, 7))
    target[1, 0], target[2, 0] = 0.01, 0.02

    fast = _plan(arm_target=target, step_dt=dt, rate_scale=1.0)
    slow = _plan(arm_target=target, step_dt=dt, rate_scale=0.25)

    assert slow.peak_joint_speed == pytest.approx(fast.peak_joint_speed * 0.25)
