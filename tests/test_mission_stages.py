#!/usr/bin/env python3
"""`mission_stages` 테스트 — 명령을 **조립만** 하고 실행하지 않는다는 것을 잠근다."""

from __future__ import annotations

from pathlib import Path

import pytest

from mission_core import Mission, Stage
from mission_stages import CommandSpec, commands_for, load_runbook, resolve

REPO = Path("/repo")


def _mission() -> Mission:
    return Mission(
        name="test",
        stages=(
            Stage(id="preset_right", title="우팔", artifacts=("reset_right_npz",)),
            Stage(id="grasp_left", title="좌팔", checkpoints=("left",)),
            Stage(id="pour", title="붓기", blocked="정책이 없다"),
        ),
        artifacts={"reset_right_npz": "logs/shadow/reset_both/reset_right_v2.npz"},
        checkpoints={
            "left": {
                "path": "logs/policy/left_v2B25/nn/v2B25_tip30_ep2150.pth",
                "md5": "27219429",
                "params": "logs/policy/left_v2B25/params",
            }
        },
    )


def _raw_runbook() -> dict:
    return {
        "preset_right": [
            {
                "note": "우팔 preset 재생",
                "argv": ["python3", "{repo}/scripts/nodes/shadow_replay.py", "--npz", "{artifact:reset_right_npz}"],
                "execute_args": ["--execute"],
            }
        ],
        "grasp_left": [
            {
                "note": "좌팔 정책",
                "argv": ["python3", "run.py", "--ckpt", "{checkpoint:left}", "--cfg", "{params:left}"],
            }
        ],
    }


# ── 치환 ───────────────────────────────────────────────────────────────────
def test_resolve_expands_the_repo_root():
    assert resolve("{repo}/scripts/a.py", _mission(), repo=REPO) == "/repo/scripts/a.py"


def test_resolve_expands_an_artifact_to_an_absolute_path():
    out = resolve("{artifact:reset_right_npz}", _mission(), repo=REPO)

    assert out == "/repo/logs/shadow/reset_both/reset_right_v2.npz"


def test_resolve_expands_a_checkpoint_and_its_params_dump():
    mission = _mission()

    assert resolve("{checkpoint:left}", mission, repo=REPO).endswith("v2B25_tip30_ep2150.pth")
    assert resolve("{params:left}", mission, repo=REPO).endswith("left_v2B25/params")


def test_resolve_leaves_plain_text_alone():
    assert resolve("--robot", _mission(), repo=REPO) == "--robot"


def test_resolve_refuses_an_unknown_artifact_and_names_it():
    with pytest.raises(KeyError, match="없는열쇠"):
        resolve("{artifact:없는열쇠}", _mission(), repo=REPO)


def test_resolve_refuses_an_unknown_placeholder_kind():
    with pytest.raises(ValueError, match="날씨"):
        resolve("{날씨:서울}", _mission(), repo=REPO)


# ── 명령 조립 ──────────────────────────────────────────────────────────────
def test_commands_for_resolves_every_placeholder_in_argv():
    book = load_runbook(_raw_runbook(), _mission())

    argv = commands_for(book, _mission(), "preset_right", repo=REPO, execute=False)[0].argv

    assert argv[1] == "/repo/scripts/nodes/shadow_replay.py"
    assert argv[3].endswith("reset_right_v2.npz")


def test_dry_run_never_carries_the_execute_flag():
    """--execute 없이는 아무것도 발행하지 않는다 — 저장소 전역 규약이다."""
    book = load_runbook(_raw_runbook(), _mission())

    argv = commands_for(book, _mission(), "preset_right", repo=REPO, execute=False)[0].argv

    assert "--execute" not in argv


def test_execute_appends_the_declared_flag():
    book = load_runbook(_raw_runbook(), _mission())

    argv = commands_for(book, _mission(), "preset_right", repo=REPO, execute=True)[0].argv

    assert argv[-1] == "--execute"


def test_a_stage_with_no_commands_yields_nothing():
    """BLOCKED 단계에는 실행할 것이 없다. 빈 목록이 정상이다."""
    book = load_runbook(_raw_runbook(), _mission())

    assert commands_for(book, _mission(), "pour", repo=REPO, execute=False) == ()


def test_the_note_survives_so_the_plan_can_explain_itself():
    book = load_runbook(_raw_runbook(), _mission())

    cmd = commands_for(book, _mission(), "preset_right", repo=REPO, execute=False)[0]

    assert cmd.note == "우팔 preset 재생"


# ── 적재 검증 ──────────────────────────────────────────────────────────────
def test_load_runbook_rejects_a_stage_that_the_mission_does_not_have():
    raw = _raw_runbook()
    raw["환상의단계"] = [{"argv": ["true"]}]

    with pytest.raises(ValueError, match="환상의단계"):
        load_runbook(raw, _mission())


def test_load_runbook_rejects_an_empty_argv():
    raw = _raw_runbook()
    raw["pour"] = [{"argv": []}]

    with pytest.raises(ValueError, match="argv"):
        load_runbook(raw, _mission())


def test_load_runbook_keeps_the_declared_order_of_commands():
    raw = {"preset_right": [{"argv": ["first"]}, {"argv": ["second"]}]}

    book = load_runbook(raw, _mission())

    assert [c.argv[0] for c in book.commands["preset_right"]] == ["first", "second"]


def test_command_spec_is_immutable():
    spec = CommandSpec(argv=("a",))

    with pytest.raises(Exception):
        spec.argv = ("b",)  # type: ignore[misc]


def test_a_background_command_is_marked_so_the_runner_keeps_it_alive():
    """중력보상 노드처럼 단계가 끝날 때까지 살아 있어야 하는 명령이 있다."""
    raw = {"preset_right": [{"argv": ["node"], "background": True}, {"argv": ["replay"]}]}

    book = load_runbook(raw, _mission())
    cmds = commands_for(book, _mission(), "preset_right", repo=REPO, execute=False)

    assert (cmds[0].background, cmds[1].background) == (True, False)


def test_background_defaults_to_false_so_commands_run_in_order():
    book = load_runbook(_raw_runbook(), _mission())

    cmd = commands_for(book, _mission(), "preset_right", repo=REPO, execute=False)[0]

    assert cmd.background is False


def test_a_manual_command_is_marked_so_the_runner_refuses_to_run_it():
    """sudo·다른 PC·venv 없는 셸에서 쳐야 하는 명령을 여기서 돌리면 엉뚱한 데서 돈다."""
    raw = {"preset_right": [{"argv": ["ros2", "launch"], "manual": True}]}

    book = load_runbook(raw, _mission())
    cmd = commands_for(book, _mission(), "preset_right", repo=REPO, execute=False)[0]

    assert cmd.manual is True


def test_commands_are_run_by_the_runner_unless_marked_manual():
    book = load_runbook(_raw_runbook(), _mission())

    cmd = commands_for(book, _mission(), "preset_right", repo=REPO, execute=False)[0]

    assert cmd.manual is False
