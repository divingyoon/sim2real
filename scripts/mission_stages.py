#!/usr/bin/env python3
"""단계 → **기존 스크립트 명령**. 조립만 하고 실행하지 않는다 (ROS·로봇 무의존).

오케스트레이터는 새 제어 코드를 쓰지 않는다. `shadow_replay.py` 의 램프·중단조건,
`isaacsim_cmd_to_jtc.py` 의 `time_from_start=0` 규약 같은 것은 사고로 얻은 것이고,
그것을 다시 구현하면 그 교훈을 잃는다. 여기서 하는 일은 **어떤 스크립트를 어떤
인자로 부르는지**를 yaml 한 곳에 모으는 것뿐이다.

판정(`mission_core`)과 분리한 이유: 무엇이 필요한가와 어떻게 실행하는가는 서로 다른
속도로 바뀐다. 스크립트 인자가 바뀌어도 가드는 그대로여야 한다.

★`--execute` 는 **선언된 단계에서, 명시적으로 켤 때만** 붙는다. 이 저장소의 모든
이동 스크립트는 그 플래그가 없으면 아무것도 발행하지 않으며, 이 규약은 테스트가
잠근다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from mission_core import Mission

#: `{종류:열쇠}` 또는 `{repo}`. 종류를 `[a-z_]` 로 좁히면 **오타가 조용히 통과한다**
#: — 못 알아본 자리표시자가 그대로 인자에 실려 나간다. 넓게 잡고 모르는 종류는 죽인다.
_PLACEHOLDER = re.compile(r"\{([^\s{}:]+)(?::([^}]*))?\}")


@dataclass(frozen=True)
class CommandSpec:
    """yaml 에 적힌 그대로의 명령. 아직 치환 전이다."""

    argv: tuple[str, ...]
    note: str = ""
    #: `--execute` 일 때만 덧붙는 인자.
    execute_args: tuple[str, ...] = ()
    #: 참이면 **띄워 두고 다음 명령으로 넘어간다**. 중력보상 노드나 라이브 사슬의
    #: 브리지처럼 단계가 끝날 때까지 살아 있어야 하는 것들이다.
    background: bool = False
    #: 참이면 **러너가 절대 실행하지 않는다** — 화면에 띄우고 사람이 친 뒤 확인만
    #: 받는다. sudo 가 필요하거나(CAN), 다른 PC 에서 돌거나(vision-3090), venv 없는
    #: 셸이어야 하는(Isaac) 명령들이다. 여기서 대신 돌리면 조용히 엉뚱한 데서 돈다.
    manual: bool = False


@dataclass(frozen=True)
class Command:
    """치환이 끝나 바로 실행할 수 있는 명령."""

    argv: tuple[str, ...]
    note: str = ""
    background: bool = False
    manual: bool = False


@dataclass(frozen=True)
class Runbook:
    commands: Mapping[str, tuple[CommandSpec, ...]] = field(default_factory=dict)


def _lookup(kind: str, key: str, mission: Mission, repo: Path) -> str:
    if kind == "artifact":
        if key not in mission.artifacts:
            raise KeyError(f"산출물 키 '{key}' 가 미션에 선언되지 않았다")
        return str(repo / mission.artifacts[key])
    if kind in ("checkpoint", "params"):
        if key not in mission.checkpoints:
            raise KeyError(f"체크포인트 키 '{key}' 가 미션에 선언되지 않았다")
        field_name = "path" if kind == "checkpoint" else "params"
        return str(repo / mission.checkpoints[key][field_name])
    raise ValueError(f"모르는 자리표시자 종류: {kind}")


def resolve(text: str, mission: Mission, *, repo: Path) -> str:
    """`{repo}` · `{artifact:키}` · `{checkpoint:키}` · `{params:키}` 를 치환한다."""

    def sub(m: re.Match) -> str:
        kind, key = m.group(1), m.group(2)
        if kind == "repo" and key is None:
            return str(repo)
        if key is None:
            raise ValueError(f"모르는 자리표시자 종류: {kind}")
        return _lookup(kind, key, mission, repo)

    return _PLACEHOLDER.sub(sub, text)


def commands_for(
    runbook: Runbook, mission: Mission, stage_id: str, *, repo: Path, execute: bool
) -> tuple[Command, ...]:
    """이 단계에서 부를 명령들. 실행하지는 않는다."""
    out = []
    for spec in runbook.commands.get(stage_id, ()):
        argv = [resolve(a, mission, repo=repo) for a in spec.argv]
        if execute:
            argv += [resolve(a, mission, repo=repo) for a in spec.execute_args]
        out.append(Command(argv=tuple(argv), note=spec.note, background=spec.background, manual=spec.manual))
    return tuple(out)


def _spec_from_raw(raw: Mapping, stage_id: str) -> CommandSpec:
    argv = tuple(str(a) for a in raw.get("argv", ()) or ())
    if not argv:
        raise ValueError(f"단계 '{stage_id}' 의 명령에 argv 가 비어 있다")
    return CommandSpec(
        argv=argv,
        note=str(raw.get("note", "")),
        execute_args=tuple(str(a) for a in raw.get("execute_args", ()) or ()),
        background=bool(raw.get("background", False)),
        manual=bool(raw.get("manual", False)),
    )


def load_runbook(raw: Mapping, mission: Mission) -> Runbook:
    """단계별 명령 정의를 검증하며 적재한다. 미션에 없는 단계는 거부한다."""
    known = {s.id for s in mission.stages}
    commands: dict[str, tuple[CommandSpec, ...]] = {}
    for stage_id, specs in (raw or {}).items():
        if stage_id not in known:
            raise ValueError(f"명령이 붙은 '{stage_id}' 가 미션의 단계가 아니다")
        commands[stage_id] = tuple(_spec_from_raw(s, stage_id) for s in specs or ())
    return Runbook(commands=commands)
