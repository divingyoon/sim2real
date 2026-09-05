#!/usr/bin/env python3
"""`pour_traj_to_replay` 순수 로직 테스트. 로봇·sim 불필요."""

from __future__ import annotations

import numpy as np
import pytest

from pour_traj_to_replay import (
    REPLAY_SOURCES,
    column_index,
    describe_feasibility,
    peak_velocity,
    required_rate_scale,
    to_replay,
)

ARM = [f"r_aj_{i}" for i in range(1, 8)]
HAND = [f"r_hj_{f}_{j}" for f in ("thumb", "index", "middle", "ring", "pinky") for j in range(1, 5)]
SIM = ["l_aj_1", "l_aj_2"] + ARM + [f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]

#: sim 이름 → canonical. 실제 매핑은 프로필이 준다.
ALIAS = {f"rj_dg_{f}_{j}": HAND[(f - 1) * 4 + (j - 1)] for f in range(1, 6) for j in range(1, 5)}


def _traj(n: int = 5) -> dict:
    t = np.arange(n, dtype=np.float32)[:, None]
    return {
        "q_target": (t * 0.01 * np.ones((1, len(SIM)))).astype(np.float32),
        "q_meas": (t * 0.008 * np.ones((1, len(SIM)))).astype(np.float32),
        "meta_joint_names": np.array(SIM),
        "meta_step_dt": np.float32(1 / 60),
    }


# ── 열 고르기 ──────────────────────────────────────────────────────────────
def test_column_index_finds_the_wanted_joints_in_order():
    assert column_index(SIM, ["r_aj_3", "r_aj_1"], what="팔") == [4, 2]


def test_column_index_accepts_an_alias_map_for_renamed_joints():
    """sim 은 rj_dg_*, 드라이버 계약은 r_hj_* 다. 이름이 갈리는 곳이 여기다."""
    idx = column_index(SIM, HAND[:2], what="손", alias=ALIAS)

    assert idx == [SIM.index("rj_dg_1_1"), SIM.index("rj_dg_1_2")]


def test_column_index_names_every_missing_joint_at_once():
    with pytest.raises(KeyError, match="없다"):
        column_index(SIM, ["r_aj_1", "환상_1", "환상_2"], what="팔")


def test_column_index_refuses_a_duplicate_request():
    """같은 열을 두 번 실으면 조용히 어긋난 궤적이 된다."""
    with pytest.raises(ValueError, match="중복"):
        column_index(SIM, ["r_aj_1", "r_aj_1"], what="팔")


# ── 변환 ───────────────────────────────────────────────────────────────────
def test_to_replay_shapes_match_what_shadow_replay_reads():
    out = to_replay(_traj(), ARM, HAND, alias=ALIAS, source="target")

    assert out["arm_target"].shape == (5, 1, 7)
    assert out["grip_cmd"].shape == (5, 1, 20)
    assert list(out["meta_joint_names"]) == ARM
    assert list(out["meta_grip_names"]) == HAND


def test_to_replay_keeps_step_dt_as_a_one_element_array():
    """`shadow_replay` 가 `meta_step_dt[0]` 으로 읽는다 — 스칼라로 넣으면 죽는다."""
    out = to_replay(_traj(), ARM, HAND, alias=ALIAS, source="target")

    assert out["meta_step_dt"].shape == (1,)
    assert float(out["meta_step_dt"][0]) == pytest.approx(1 / 60)


def test_to_replay_can_take_the_commanded_or_the_measured_trajectory():
    cmd = to_replay(_traj(), ARM, HAND, alias=ALIAS, source="target")
    meas = to_replay(_traj(), ARM, HAND, alias=ALIAS, source="meas")

    assert not np.allclose(cmd["arm_target"], meas["arm_target"])
    assert meas["arm_target"][1, 0, 0] == pytest.approx(0.008)


def test_to_replay_refuses_an_unknown_source():
    with pytest.raises(ValueError, match="모르는"):
        to_replay(_traj(), ARM, HAND, alias=ALIAS, source="점괘")


def test_to_replay_carries_the_measured_state_for_the_shadow_topic():
    out = to_replay(_traj(), ARM, HAND, alias=ALIAS, source="target")

    assert out["q_meas"].shape == (5, 7)


def test_to_replay_records_where_it_came_from():
    """재생 결과를 나중에 대조하려면 출처가 파일 안에 있어야 한다."""
    out = to_replay(_traj(), ARM, HAND, alias=ALIAS, source="meas",
                    provenance={"meta_checkpoint": "d3.pth"})

    assert str(out["meta_replay_source"]) == "meas"
    assert str(out["meta_checkpoint"]) == "d3.pth"


# ── 실기 가능성 ────────────────────────────────────────────────────────────
def test_peak_velocity_is_per_joint_and_uses_the_step_dt():
    target = np.array([[0.0, 0.0], [0.1, 0.02], [0.2, 0.04]])

    peak = peak_velocity(target, dt=0.1)

    assert peak == pytest.approx([1.0, 0.2])


def test_peak_velocity_of_a_single_frame_is_zero():
    assert peak_velocity(np.zeros((1, 3)), dt=0.1) == pytest.approx([0, 0, 0])


def test_required_rate_scale_is_one_when_already_within_limits():
    assert required_rate_scale([1.0, 0.5], [2.0, 2.0]) == pytest.approx(1.0)


def test_required_rate_scale_slows_down_to_the_worst_joint():
    """한 관절이라도 넘으면 전체를 늦춰야 한다 — 궤적 모양을 지키려면."""
    assert required_rate_scale([4.0, 1.0], [2.0, 2.0]) == pytest.approx(0.5)


def test_required_rate_scale_ignores_joints_without_a_limit():
    assert required_rate_scale([9.0, 1.0], [None, 2.0]) == pytest.approx(1.0)


def test_describe_feasibility_names_the_joints_that_exceed():
    text = describe_feasibility(ARM, [1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0], [2.0] * 7, dt=1 / 60)

    assert "r_aj_2" in text
    assert "2.50" in text  # 5.0 / 2.0


def test_describe_feasibility_says_so_when_nothing_exceeds():
    text = describe_feasibility(ARM, [0.5] * 7, [2.0] * 7, dt=1 / 60)

    assert "한계 안" in text


def test_the_declared_sources_are_the_two_we_support():
    assert set(REPLAY_SOURCES) == {"target", "meas"}
