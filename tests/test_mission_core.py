#!/usr/bin/env python3
"""`mission_core` 의 순수 로직 테스트. 로봇·ROS·GPU·파일시스템 전부 불필요."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mission_core import (
    STATUS_ABORTED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    Evidence,
    Mission,
    Stage,
    advance,
    begin,
    gate,
    initial_state,
    load_mission,
    next_stage_id,
    plan,
    stage_by_id,
)


# ── 시험용 미션 ────────────────────────────────────────────────────────────
def _mission() -> Mission:
    return Mission(
        name="test",
        stages=(
            Stage(id="preflight", title="점검"),
            Stage(
                id="preset_right",
                title="우팔 preset",
                needs=("preflight",),
                artifacts=("reset_right_npz",),
                touches_real=True,
            ),
            Stage(
                id="preset_left",
                title="좌팔 preset",
                needs=("preset_right",),
                needs_why="좌팔 fabric 이 우팔을 rest 자리에 있다고 가정한다",
                artifacts=("reset_left_npz",),
                touches_real=True,
            ),
            Stage(id="perceive", title="컵 인식", needs=("preset_left",)),
            Stage(
                id="grasp_left",
                title="좌팔 파지",
                needs=("perceive",),
                checkpoints=("left",),
                touches_real=True,
            ),
            Stage(
                id="pour",
                title="물붓기",
                needs=("grasp_left",),
                blocked="배포 가능한 pour 정책이 없다",
                evidence="docs/pour_sensor_deployment.md",
            ),
        ),
        loop_to="perceive",
        artifacts={
            "reset_right_npz": "logs/shadow/reset_both/reset_right_v2.npz",
            "reset_left_npz": "logs/shadow/reset_both/reset_left.npz",
        },
        checkpoints={
            "left": {
                "path": "logs/policy/left_v2B25/nn/v2B25_tip30_ep2150.pth",
                "md5": "27219429",
                "params": "logs/policy/left_v2B25/params",
            }
        },
    )


def _everything_present(mission: Mission, **over) -> Evidence:
    base = dict(
        present_artifacts=frozenset(mission.artifacts),
        checkpoint_digests={"left": "27219429"},
        present_params=frozenset(mission.checkpoints),
        approvals=frozenset(s.id for s in mission.stages),
    )
    base.update(over)
    return Evidence(**base)


def _at(mission: Mission, stage: str, done: tuple[str, ...]):
    return replace(initial_state(mission), stage=stage, completed=done)


# ── 미션 구조 ──────────────────────────────────────────────────────────────
def test_initial_state_starts_pending_at_the_first_stage():
    state = initial_state(_mission())

    assert state.stage == "preflight"
    assert state.status == STATUS_PENDING
    assert state.completed == ()
    assert state.cycle == 0


def test_stage_by_id_raises_for_an_unknown_stage():
    with pytest.raises(KeyError, match="없는 단계"):
        stage_by_id(_mission(), "환상의단계")


def test_next_stage_id_walks_the_declared_order():
    mission = _mission()

    assert next_stage_id(mission, "preflight") == "preset_right"
    assert next_stage_id(mission, "preset_right") == "preset_left"


def test_next_stage_id_loops_back_after_the_last_stage():
    assert next_stage_id(_mission(), "pour") == "perceive"


def test_next_stage_id_returns_none_when_the_mission_does_not_loop():
    mission = Mission(name="once", stages=(Stage(id="only", title="하나"),))

    assert next_stage_id(mission, "only") is None


# ── 가드: 선행 단계 ────────────────────────────────────────────────────────
def test_gate_passes_when_everything_is_in_place():
    mission = _mission()

    result = gate(mission, "preflight", initial_state(mission), _everything_present(mission))

    assert result.ok
    assert result.reasons == ()


def test_gate_refuses_a_stage_whose_predecessor_is_not_done():
    mission = _mission()

    result = gate(
        mission, "preset_left", initial_state(mission), _everything_present(mission)
    )

    assert not result.ok
    assert any("preset_right" in r for r in result.reasons)


def test_gate_explains_why_the_order_matters_not_just_that_it_was_violated():
    """'선행 미완료' 만 말하면 사용자는 순서를 건너뛸 근거를 못 판단한다."""
    mission = _mission()

    result = gate(
        mission, "preset_left", initial_state(mission), _everything_present(mission)
    )

    assert any("fabric" in r for r in result.reasons)


# ── 가드: 산출물 ───────────────────────────────────────────────────────────
def test_gate_refuses_when_an_artifact_is_missing_and_names_its_path():
    mission = _mission()

    result = gate(
        mission,
        "preset_right",
        _at(mission, "preset_right", ("preflight",)),
        _everything_present(mission, present_artifacts=frozenset()),
    )

    assert not result.ok
    assert any("reset_right_v2.npz" in r for r in result.reasons)


def test_gate_reports_every_problem_at_once_not_just_the_first():
    """하나 고치고 다시 돌려서 또 막히는 것은 시간 낭비다."""
    mission = _mission()

    result = gate(
        mission,
        "preset_right",
        initial_state(mission),
        _everything_present(mission, present_artifacts=frozenset(), approvals=frozenset()),
    )

    assert len(result.reasons) >= 3  # 선행 + 산출물 + 승인


# ── 가드: 체크포인트 ───────────────────────────────────────────────────────
_BEFORE_GRASP = ("preflight", "preset_right", "preset_left", "perceive")


def _grasp_gate(mission: Mission, **over):
    return gate(
        mission,
        "grasp_left",
        _at(mission, "grasp_left", _BEFORE_GRASP),
        _everything_present(mission, **over),
    )


def test_gate_refuses_when_the_checkpoint_digest_differs():
    """정책을 바꿔 끼웠는데 조용히 옛것으로 도는 사고를 막는다."""
    result = _grasp_gate(_mission(), checkpoint_digests={"left": "deadbeef"})

    assert not result.ok
    assert any("27219429" in r and "deadbeef" in r for r in result.reasons)


def test_gate_refuses_when_the_checkpoint_file_is_absent():
    result = _grasp_gate(_mission(), checkpoint_digests={})

    assert not result.ok
    assert any("v2B25_tip30_ep2150.pth" in r for r in result.reasons)


def test_gate_refuses_when_the_run_params_dump_is_absent():
    """dump 가 없으면 홈을 알 수 없고, 홈이 틀리면 preset 이 엉뚱한 곳에 도착한다."""
    result = _grasp_gate(_mission(), present_params=frozenset())

    assert not result.ok
    assert any("params" in r for r in result.reasons)


def test_gate_accepts_a_digest_given_in_full_when_the_mission_lists_a_prefix():
    result = _grasp_gate(
        _mission(), checkpoint_digests={"left": "272194299637fef6aa89c8e93161d1b6"}
    )

    assert result.ok


# ── 가드: 승인 · BLOCKED ───────────────────────────────────────────────────
def test_gate_refuses_a_real_robot_stage_without_approval():
    mission = _mission()

    result = gate(
        mission,
        "preset_right",
        _at(mission, "preset_right", ("preflight",)),
        _everything_present(mission, approvals=frozenset()),
    )

    assert not result.ok
    assert any("승인" in r for r in result.reasons)


def test_gate_allows_a_sim_only_stage_without_approval():
    mission = _mission()

    result = gate(
        mission,
        "perceive",
        _at(mission, "perceive", ("preflight", "preset_right", "preset_left")),
        _everything_present(mission, approvals=frozenset()),
    )

    assert result.ok


def test_gate_refuses_a_blocked_stage_and_says_why_with_its_evidence():
    mission = _mission()

    result = gate(
        mission,
        "pour",
        _at(mission, "pour", _BEFORE_GRASP + ("grasp_left",)),
        _everything_present(mission),
    )

    assert not result.ok
    assert any("pour 정책이 없다" in r for r in result.reasons)
    assert any("pour_sensor_deployment.md" in r for r in result.reasons)


# ── 전이 ───────────────────────────────────────────────────────────────────
def test_begin_marks_the_stage_running():
    assert begin(initial_state(_mission())).status == STATUS_RUNNING


def test_advance_on_done_moves_to_the_next_stage_and_records_completion():
    mission = _mission()

    state = advance(mission, begin(initial_state(mission)), STATUS_DONE)

    assert state.stage == "preset_right"
    assert state.status == STATUS_PENDING
    assert state.completed == ("preflight",)


def _at_last(mission: Mission):
    return replace(
        initial_state(mission),
        stage="pour",
        status=STATUS_RUNNING,
        completed=_BEFORE_GRASP + ("grasp_left",),
    )


def test_advance_on_the_last_stage_loops_and_counts_a_cycle():
    mission = _mission()

    looped = advance(mission, _at_last(mission), STATUS_DONE)

    assert looped.stage == "perceive"
    assert looped.cycle == 1


def test_a_new_cycle_forgets_the_stages_that_must_run_again():
    """반복 지점 뒤의 단계는 다시 해야 한다 — completed 에 남아 있으면 가드가 눈이 먼다."""
    mission = _mission()

    looped = advance(mission, _at_last(mission), STATUS_DONE)

    assert "perceive" not in looped.completed
    assert "grasp_left" not in looped.completed
    assert "preflight" in looped.completed  # 반복 지점 앞은 유지


def test_advance_on_failure_stays_put_and_records_nothing():
    mission = _mission()

    state = advance(mission, begin(initial_state(mission)), STATUS_FAILED, note="타임아웃")

    assert state.stage == "preflight"
    assert state.status == STATUS_FAILED
    assert state.completed == ()
    assert state.note == "타임아웃"


def test_advance_on_abort_stays_put():
    mission = _mission()

    state = advance(mission, begin(initial_state(mission)), STATUS_ABORTED)

    assert state.status == STATUS_ABORTED
    assert state.stage == "preflight"


def test_advance_rejects_an_outcome_it_does_not_know():
    mission = _mission()

    with pytest.raises(ValueError, match="모르는 결과"):
        advance(mission, begin(initial_state(mission)), "그럭저럭")


def test_advance_returns_a_new_object_and_leaves_the_original_alone():
    mission = _mission()
    before = begin(initial_state(mission))

    after = advance(mission, before, STATUS_DONE)

    assert before.stage == "preflight" and before.completed == ()
    assert after is not before


# ── 전체 계획 ──────────────────────────────────────────────────────────────
def test_plan_covers_every_stage_in_order():
    mission = _mission()

    rows = plan(mission, initial_state(mission), _everything_present(mission))

    assert [r.stage.id for r in rows] == [s.id for s in mission.stages]


def test_plan_marks_blocked_stages_so_they_stand_out():
    mission = _mission()

    blocked = [
        r
        for r in plan(mission, initial_state(mission), _everything_present(mission))
        if r.stage.blocked
    ]

    assert [r.stage.id for r in blocked] == ["pour"]
    assert not blocked[0].result.ok


def test_plan_judges_each_stage_as_if_its_predecessors_had_run():
    """지금 못 한다고 다 BLOCKED 로 칠하면 진짜 막힌 것이 묻힌다."""
    mission = _mission()

    by_id = {
        r.stage.id: r
        for r in plan(mission, initial_state(mission), _everything_present(mission))
    }

    assert by_id["preset_left"].result.ok
    assert by_id["grasp_left"].result.ok


# ── yaml 적재 (경계에서 검증) ──────────────────────────────────────────────
def _raw() -> dict:
    return {
        "name": "pour",
        "loop_to": "perceive",
        "artifacts": {"a": "path/a.npz"},
        "checkpoints": {"left": {"path": "p.pth", "md5": "abc", "params": "d"}},
        "stages": [
            {"id": "perceive", "title": "인식"},
            {"id": "go", "title": "간다", "needs": ["perceive"], "artifacts": ["a"]},
        ],
    }


def test_load_mission_builds_the_declared_structure():
    mission = load_mission(_raw())

    assert mission.name == "pour"
    assert [s.id for s in mission.stages] == ["perceive", "go"]
    assert mission.stages[1].needs == ("perceive",)


def test_load_mission_rejects_a_stage_needing_something_that_does_not_exist():
    raw = _raw()
    raw["stages"][1]["needs"] = ["환상"]

    with pytest.raises(ValueError, match="환상"):
        load_mission(raw)


def test_load_mission_rejects_a_stage_needing_a_later_stage():
    """뒤 단계를 선행으로 두면 영영 못 간다. 부팅에서 죽는 편이 낫다."""
    raw = _raw()
    raw["stages"][0]["needs"] = ["go"]

    with pytest.raises(ValueError, match="뒤"):
        load_mission(raw)


def test_load_mission_rejects_an_undeclared_artifact_key():
    raw = _raw()
    raw["stages"][1]["artifacts"] = ["없는키"]

    with pytest.raises(ValueError, match="없는키"):
        load_mission(raw)


def test_load_mission_rejects_an_undeclared_checkpoint_key():
    raw = _raw()
    raw["stages"][1]["checkpoints"] = ["없는정책"]

    with pytest.raises(ValueError, match="없는정책"):
        load_mission(raw)


def test_load_mission_rejects_a_loop_target_that_is_not_a_stage():
    raw = _raw()
    raw["loop_to"] = "어디"

    with pytest.raises(ValueError, match="어디"):
        load_mission(raw)


def test_load_mission_rejects_duplicate_stage_ids():
    raw = _raw()
    raw["stages"].append({"id": "go", "title": "또 간다"})

    with pytest.raises(ValueError, match="중복"):
        load_mission(raw)


def test_load_mission_rejects_a_checkpoint_entry_missing_its_digest():
    raw = _raw()
    del raw["checkpoints"]["left"]["md5"]

    with pytest.raises(ValueError, match="md5"):
        load_mission(raw)
