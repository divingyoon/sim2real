#!/usr/bin/env python3
"""`mission_run` 테스트 — 조사·기록·CLI 규약. 로봇·ROS 불필요."""

from __future__ import annotations

import hashlib

import pytest

import mission_run
from mission_core import STATUS_DONE, Evidence, Mission, MissionState, Stage

PLAIN = b"hello"
PLAIN_MD5 = hashlib.md5(PLAIN).hexdigest()  # noqa: S324


def _mission() -> Mission:
    return Mission(
        name="test",
        stages=(
            Stage(id="check", title="점검"),
            Stage(id="move", title="이동", needs=("check",), touches_real=True),
        ),
        artifacts={"traj": "logs/t.npz"},
        checkpoints={"left": {"path": "nn/p.pth", "md5": PLAIN_MD5[:8], "params": "params"}},
    )


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "nn").mkdir()
    (tmp_path / "params").mkdir()
    (tmp_path / "logs" / "t.npz").write_bytes(b"x")
    (tmp_path / "nn" / "p.pth").write_bytes(PLAIN)
    return tmp_path


# ── 바깥 세계 조사 ─────────────────────────────────────────────────────────
def test_gather_evidence_sees_the_files_that_exist(repo):
    ev = mission_run.gather_evidence(_mission(), repo=repo, approvals=frozenset())

    assert ev.present_artifacts == frozenset({"traj"})
    assert ev.checkpoint_digests["left"] == PLAIN_MD5
    assert ev.present_params == frozenset({"left"})


def test_gather_evidence_omits_a_checkpoint_that_is_not_there(repo):
    (repo / "nn" / "p.pth").unlink()

    ev = mission_run.gather_evidence(_mission(), repo=repo, approvals=frozenset())

    assert "left" not in ev.checkpoint_digests


def test_gather_evidence_omits_an_artifact_that_is_not_there(repo):
    (repo / "logs" / "t.npz").unlink()

    ev = mission_run.gather_evidence(_mission(), repo=repo, approvals=frozenset())

    assert ev.present_artifacts == frozenset()


def test_gather_evidence_passes_the_approvals_through(repo):
    ev = mission_run.gather_evidence(_mission(), repo=repo, approvals=frozenset({"move"}))

    assert ev.approvals == frozenset({"move"})


# ── 계획 뷰의 승인 규칙 ────────────────────────────────────────────────────
def test_the_plan_view_assumes_approval_so_real_blockers_stand_out():
    """승인 없음을 13줄 반복하면 진짜 막힌 것이 묻힌다."""
    mission = _mission()

    ev = mission_run.plan_evidence(mission, Evidence())

    assert ev.approvals == frozenset({"check", "move"})


def test_the_plan_view_does_not_touch_anything_else():
    ev = Evidence(present_artifacts=frozenset({"traj"}), checkpoint_digests={"left": "a"})

    out = mission_run.plan_evidence(_mission(), ev)

    assert out.present_artifacts == ev.present_artifacts
    assert out.checkpoint_digests == ev.checkpoint_digests


# ── 기록 ───────────────────────────────────────────────────────────────────
def test_state_survives_a_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(mission_run, "MISSION_LOG_DIR", tmp_path)
    state = MissionState(stage="move", status=STATUS_DONE, completed=("check",), cycle=2)

    mission_run.save_state("r1", state)

    assert mission_run.load_state("r1") == state


def test_every_transition_is_appended_so_the_history_is_not_lost(tmp_path, monkeypatch):
    monkeypatch.setattr(mission_run, "MISSION_LOG_DIR", tmp_path)

    mission_run.save_state("r1", MissionState(stage="check"))
    mission_run.save_state("r1", MissionState(stage="move"))

    assert len((tmp_path / "r1" / "state.jsonl").read_text().strip().split("\n")) == 2


def test_resuming_a_run_that_was_never_recorded_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(mission_run, "MISSION_LOG_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="이어갈 기록이 없다"):
        mission_run.load_state("없는기록")


# ── CLI 규약 ───────────────────────────────────────────────────────────────
def test_the_real_mission_file_loads_and_declares_what_is_blocked():
    """배포된 미션 정의가 실제로 적재되는지 — 오타 하나면 실기 날에 알게 된다."""
    mission, runbook = mission_run._load(mission_run.DEFAULT_MISSION)

    blocked = [s.id for s in mission.stages if s.blocked]
    assert blocked, "막힌 단계가 하나도 없다면 선언을 빠뜨린 것이다"
    assert all(s.evidence for s in mission.stages if s.blocked), "막힘에는 근거가 있어야 한다"
    assert set(runbook.commands) <= {s.id for s in mission.stages}


def test_every_blocked_stage_names_a_file_that_exists():
    """근거가 없는 경로를 가리키면 사용자가 확인할 수 없다."""
    mission, _ = mission_run._load(mission_run.DEFAULT_MISSION)

    missing = [
        s.evidence
        for s in mission.stages
        if s.blocked and not (mission_run.REPO / s.evidence).exists()
    ]

    assert missing == []


def test_plan_run_publishes_nothing_and_exits_clean(capsys):
    code = mission_run.main(["--plan"])

    assert code == 0
    assert "막힘" in capsys.readouterr().out


def test_a_real_stage_is_refused_without_approval(capsys):
    """'모든 실기 진행은 사용자 허락과 함께' 를 코드로 옮긴 것이다."""
    code = mission_run.main(["--stage", "preset_head", "--execute"])

    assert code == 1
    assert "승인이 없다" in capsys.readouterr().out


def test_a_dry_run_of_a_ready_stage_says_it_published_nothing(capsys):
    code = mission_run.main(["--stage", "preflight"])

    out = capsys.readouterr().out
    assert code == 0
    assert "아무것도 발행하지 않았다" in out
    assert "--execute" not in out.split("드라이런")[0]


def test_a_blocked_stage_still_previews_its_commands_in_a_dry_run(capsys):
    """무엇을 하려던 것인지는 보여야 한다 — 발행이 없으니 위험도 없다."""
    code = mission_run.main(["--stage", "preset_left"])

    out = capsys.readouterr().out
    assert code == 1                       # 가드는 막았다고 말한다
    assert "미리보기" in out
    assert "nodes/shadow_replay.py" in out       # 그래도 무엇을 하려던 것인지는 보인다
    assert "--execute" not in out.split("드라이런")[0]


def test_the_preview_of_a_blocked_stage_never_carries_execute(capsys):
    mission_run.main(["--stage", "preset_right"])

    argv_lines = [l for l in capsys.readouterr().out.split("\n") if "nodes/shadow_replay.py" in l]
    assert argv_lines and all("--execute" not in l for l in argv_lines)
