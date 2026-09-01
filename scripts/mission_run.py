#!/usr/bin/env python3
"""미션 러너 — 순서대로 부르고, 막히면 왜 막혔는지 말한다.

    python3 scripts/mission_run.py --plan                    # 전체 판정 (발행 없음)
    python3 scripts/mission_run.py --stage preset_right      # 그 단계의 명령만 출력
    python3 scripts/mission_run.py --stage preset_right --execute --approve preset_right
    python3 scripts/mission_run.py --resume 20260901_161200 --plan
    python3 scripts/mission_run.py --abort  20260901_161200

세 가지 규약을 코드로 잠근다.

  ① **`--execute` 없이는 아무것도 발행하지 않는다.** 이 저장소 전역 규약이다.
  ② **실기 단계는 `--approve <id>` 없이 돌지 않는다.** "모든 실기 진행은 사용자 허락
     과 함께"를 코드로 옮긴 것이다. `--execute` 여도 승인은 따로 받는다.
  ③ **`manual` 명령은 러너가 절대 실행하지 않는다.** sudo·다른 PC·venv 없는 셸에서
     사람이 쳐야 하는 것들이다. 여기서 대신 돌리면 조용히 엉뚱한 데서 돈다.

자식은 전부 **새 세션**으로 띄운다. 그래야 `--abort` 하나로 프로세스 그룹째 정지할
수 있다 — 지금은 여러 노드가 동시에 돌 때 일괄 정지 수단이 없다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import yaml

from mission_core import (
    STATUS_ABORTED,
    STATUS_DONE,
    STATUS_FAILED,
    Evidence,
    Mission,
    MissionState,
    advance,
    begin,
    gate,
    initial_state,
    load_mission,
    plan,
    stage_by_id,
)
from mission_stages import Command, Runbook, commands_for, load_runbook

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MISSION = REPO / "config" / "mission_pour.yaml"
MISSION_LOG_DIR = REPO / "logs" / "mission"

#: 자식이 SIGTERM 을 무시할 때 SIGKILL 까지 기다리는 시간 [s].
KILL_GRACE_S = 5.0


# ── 바깥 세계 조사 ─────────────────────────────────────────────────────────
def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 — 무결성 대조용이지 보안용이 아니다
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gather_evidence(mission: Mission, *, repo: Path, approvals: frozenset[str]) -> Evidence:
    """파일이 있는지, md5 가 무엇인지 실제로 본다. 판정은 `mission_core` 가 한다."""
    present = frozenset(k for k, rel in mission.artifacts.items() if (repo / rel).exists())
    digests, params = {}, set()
    for key, spec in mission.checkpoints.items():
        path = repo / spec["path"]
        if path.is_file():
            digests[key] = _md5(path)
        if (repo / spec["params"]).is_dir():
            params.add(key)
    return Evidence(
        present_artifacts=present,
        checkpoint_digests=digests,
        present_params=frozenset(params),
        approvals=approvals,
    )


# ── 상태 보관 ──────────────────────────────────────────────────────────────
def _run_dir(run_id: str) -> Path:
    return MISSION_LOG_DIR / run_id


def save_state(run_id: str, state: MissionState) -> None:
    d = _run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    row = {
        "stage": state.stage,
        "status": state.status,
        "completed": list(state.completed),
        "cycle": state.cycle,
        "note": state.note,
    }
    (d / "state.json").write_text(json.dumps(row, ensure_ascii=False, indent=2))
    with (d / "state.jsonl").open("a") as fh:
        fh.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), **row}, ensure_ascii=False) + "\n")


def load_state(run_id: str) -> MissionState:
    path = _run_dir(run_id) / "state.json"
    if not path.is_file():
        raise FileNotFoundError(f"이어갈 기록이 없다: {path}")
    row = json.loads(path.read_text())
    return MissionState(
        stage=row["stage"],
        status=row["status"],
        completed=tuple(row.get("completed", ())),
        cycle=int(row.get("cycle", 0)),
        note=row.get("note", ""),
    )


# ── 출력 ───────────────────────────────────────────────────────────────────
def plan_evidence(mission: Mission, ev: Evidence) -> Evidence:
    """전체 계획을 볼 때는 **승인이 다 떨어졌다고 친다**.

    이 표의 목적은 "무엇을 더 만들어야 하는가"이지 "지금 승인이 있는가"가 아니다.
    승인 없음을 13줄에 걸쳐 반복하면 진짜 막힌 것이 묻힌다. 승인은 실행 시점에
    `gate` 가 원래 evidence 로 다시 본다 — 여기서 눈감아 준다고 실행이 되지는 않는다.
    """
    return replace(ev, approvals=frozenset(s.id for s in mission.stages))


def print_plan(mission: Mission, runbook: Runbook, state: MissionState, ev: Evidence) -> None:
    rows = plan(mission, state, plan_evidence(mission, ev))
    ready = sum(1 for r in rows if r.result.ok)
    blocked = [r for r in rows if r.stage.blocked]
    print(f"\n미션: {mission.name}   ({len(rows)}단계 · 통과 {ready} · 막힘 {len(blocked)})")
    print(f"현재: {state.stage} [{state.status}] · 사이클 {state.cycle}\n")
    for row in rows:
        mark = "✓" if row.result.ok else ("■" if row.stage.blocked else "·")
        real = " 실기" if row.stage.touches_real else "    "
        done = "완료" if row.stage.id in state.completed else "  "
        print(f"  {mark} {row.stage.id:14}{real} {done}  {row.stage.title}")
        for reason in row.result.reasons:
            print(f"        └ {reason}")
        for cmd in commands_for(runbook, mission, row.stage.id, repo=REPO, execute=False):
            kind = "수동" if cmd.manual else ("배경" if cmd.background else "실행")
            print(f"          [{kind}] {cmd.note or ' '.join(cmd.argv[:3])}")
    print("\n  실기 단계는 실행할 때 --approve <id> 를 따로 받는다 (이 표에서는 묻지 않는다).")
    if blocked:
        print(f"\n막힌 {len(blocked)}단계가 다음에 만들 것이다 — 위의 이유와 근거 파일을 볼 것.")


def print_commands(mission: Mission, runbook: Runbook, stage_id: str, execute: bool) -> None:
    cmds = commands_for(runbook, mission, stage_id, repo=REPO, execute=execute)
    if not cmds:
        print("  (명령 없음 — 산출물 확인만 하는 단계다)")
        return
    for i, cmd in enumerate(cmds, 1):
        kind = "★수동" if cmd.manual else ("배경" if cmd.background else "실행")
        print(f"  {i}. [{kind}] {cmd.note}")
        print(f"     {' '.join(cmd.argv)}")


# ── 실행 ───────────────────────────────────────────────────────────────────
def _confirm(question: str, *, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _spawn(cmd: Command) -> subprocess.Popen:
    """새 세션으로 띄운다 — 그래야 그룹째 정지할 수 있다."""
    return subprocess.Popen(cmd.argv, cwd=str(REPO), start_new_session=True)


def _stop(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        if proc.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            continue
    for proc in procs:
        try:
            proc.wait(timeout=KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            with_pid = os.getpgid(proc.pid)
            os.killpg(with_pid, signal.SIGKILL)


def _record_pids(run_id: str, procs: list[subprocess.Popen]) -> None:
    if not procs:
        return
    (_run_dir(run_id) / "pids").write_text(
        "\n".join(str(p.pid) for p in procs if p.poll() is None) + "\n"
    )


def run_stage(
    mission: Mission, runbook: Runbook, stage_id: str, *, run_id: str, assume_yes: bool
) -> str:
    """이 단계의 명령을 순서대로 실행한다. 결과(DONE/FAILED/ABORTED)를 낸다."""
    background: list[subprocess.Popen] = []
    try:
        for cmd in commands_for(runbook, mission, stage_id, repo=REPO, execute=True):
            if cmd.manual:
                print(f"\n  ★수동 — 다른 셸에서 직접 실행할 것: {cmd.note}")
                print(f"    {' '.join(cmd.argv)}")
                if not _confirm("  실행했고 정상인가?", assume_yes=assume_yes):
                    return STATUS_ABORTED
                continue
            print(f"\n  ▶ {cmd.note or ' '.join(cmd.argv[:3])}")
            if not _confirm("  이 명령을 실행할까?", assume_yes=assume_yes):
                return STATUS_ABORTED
            proc = _spawn(cmd)
            if cmd.background:
                background.append(proc)
                _record_pids(run_id, background)
                continue
            if proc.wait() != 0:
                return STATUS_FAILED
        return STATUS_DONE
    except KeyboardInterrupt:
        print("\n  중단 — 띄워 둔 자식을 정지한다")
        return STATUS_ABORTED
    finally:
        _stop(background)


def abort_run(run_id: str) -> int:
    """다른 터미널에서 띄워 둔 자식들을 그룹째 정지한다."""
    path = _run_dir(run_id) / "pids"
    if not path.is_file():
        print(f"띄워 둔 자식 기록이 없다: {path}")
        return 0
    stopped = 0
    for line in path.read_text().split():
        try:
            os.killpg(os.getpgid(int(line)), signal.SIGTERM)
            stopped += 1
        except (ProcessLookupError, PermissionError, ValueError):
            continue
    path.unlink()
    print(f"{stopped}개 프로세스 그룹에 SIGTERM 을 보냈다")
    return stopped


# ── CLI ────────────────────────────────────────────────────────────────────
def _load(path: Path) -> tuple[Mission, Runbook]:
    raw = yaml.safe_load(path.read_text())
    mission = load_mission(raw)
    return mission, load_runbook(raw.get("run", {}), mission)


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mission", type=Path, default=DEFAULT_MISSION)
    p.add_argument("--plan", action="store_true", help="전 단계 판정을 출력하고 끝낸다")
    p.add_argument("--stage", help="이 단계만 본다 / 실행한다")
    p.add_argument("--execute", action="store_true", help="★실제로 실행한다")
    p.add_argument("--approve", action="append", default=[], help="이 단계의 실기 동작을 승인한다")
    p.add_argument("--yes", action="store_true", help="명령마다 묻지 않는다 (자동화용)")
    p.add_argument("--run", help="기록 이름 (기본: 지금 시각)")
    p.add_argument("--resume", help="이 기록을 이어간다")
    p.add_argument("--abort", help="이 기록이 띄워 둔 자식을 정지한다")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.abort:
        abort_run(args.abort)
        return 0

    mission, runbook = _load(args.mission)
    state = load_state(args.resume) if args.resume else initial_state(mission)
    run_id = args.resume or args.run or datetime.now().strftime("%Y%m%d_%H%M%S")
    evidence = gather_evidence(mission, repo=REPO, approvals=frozenset(args.approve))

    if args.plan or not args.stage:
        print_plan(mission, runbook, state, evidence)
        return 0

    stage = stage_by_id(mission, args.stage)
    result = gate(mission, stage.id, state, evidence)
    print(f"\n단계 {stage.id} — {stage.title}")
    for reason in result.reasons:
        print(f"  ✗ {reason}")
    if not result.ok and args.execute:
        print("\n  → 실행하지 않는다.")
        return 1

    if not args.execute:
        # 가드가 막고 있어도 **무엇을 하려던 것인지는 보여준다.** 아무것도 발행하지
        # 않으므로 위험이 없고, 순서상 아직 못 가는 단계를 미리 검토할 수 있어야 한다.
        if not result.ok:
            print("\n  가드가 막고 있다 — 아래는 미리보기다 (실행되지 않는다):")
        print_commands(mission, runbook, stage.id, execute=False)
        print("\n  (드라이런 — --execute 를 주면 실행한다. 아무것도 발행하지 않았다.)")
        return 0 if result.ok else 1

    state = begin(state)
    save_state(run_id, state)
    outcome = run_stage(mission, runbook, stage.id, run_id=run_id, assume_yes=args.yes)
    state = advance(mission, state, outcome)
    save_state(run_id, state)
    print(f"\n  결과: {outcome} → 다음 {state.stage}  (기록 logs/mission/{run_id})")
    return 0 if outcome == STATUS_DONE else 1


if __name__ == "__main__":
    sys.exit(main())
