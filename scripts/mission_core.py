#!/usr/bin/env python3
"""미션의 **순서를 아는 주체** — 순수 코어 (ROS·로봇·파일시스템 무의존).

이 저장소에는 실기 한 사이클을 도는 데 필요한 부품이 거의 다 있다. 없는 것은
**순서**다. 지금은 사람이 런북 셋을 번갈아 보며 터미널을 친다. 더 나쁜 것은
**무엇이 막혀 있는지가 문서 여기저기 흩어져 있다**는 것이다 — pour 초기 자세가
grasp goal 분포 밖이라는 판정은 로그 디렉토리의 README 안에만 있고, 우팔 obs 계약
불일치는 테스트 주석에만 있다.

그래서 이 모듈의 값어치는 "자동으로 돌린다"가 아니라 **막힌 것을 그 자리에서
말한다**에 있다. 아직 만들지 않은 단계는 `Stage.blocked` 로 선언하고, 가드가 그
이유와 근거 파일을 함께 낸다. 숨기지 않는다.

세 가지를 지킨다.

  ① **바깥 세계는 호출자가 조사한다.** 파일이 있는지, md5 가 무엇인지, 승인이
     떨어졌는지는 `Evidence` 로 받는다. 여기서는 판정만 한다 — 그래서 로봇 없이
     테스트된다.
  ② **문제를 한꺼번에 말한다.** 하나 고치고 다시 돌려서 또 막히는 것은 시간 낭비다.
  ③ **상태는 불변이다.** 전이는 새 객체를 낸다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_DONE = "DONE"
STATUS_FAILED = "FAILED"
STATUS_ABORTED = "ABORTED"

#: `advance` 가 받아들이는 결과. 이 밖의 값은 오타이므로 죽인다.
_OUTCOMES = (STATUS_DONE, STATUS_FAILED, STATUS_ABORTED)

#: 체크포인트 항목에 반드시 있어야 하는 키. 하나라도 빠지면 부팅에서 죽는다 —
#: params dump 가 없으면 홈을 알 수 없고, 홈이 틀리면 preset 이 엉뚱한 곳에 도착한다.
_CHECKPOINT_KEYS = ("path", "md5", "params")


@dataclass(frozen=True)
class Stage:
    """미션의 한 단계. **어떻게 실행하는가가 아니라 무엇이 필요한가**를 적는다.

    실행 방법은 `mission_stages` 가 안다. 여기 있는 것은 판정에 쓰이는 것뿐이다.
    """

    id: str
    title: str
    needs: tuple[str, ...] = ()
    #: 순서가 왜 강제되는지 한 문장. "선행 미완료"만 말하면 사용자는 건너뛸 근거를
    #: 판단할 수 없다.
    needs_why: str = ""
    artifacts: tuple[str, ...] = ()
    checkpoints: tuple[str, ...] = ()
    touches_real: bool = False
    #: 비어 있지 않으면 **아직 만들지 않은 단계**다. 그 문장이 막힌 이유다.
    blocked: str = ""
    #: 그 판정의 근거 파일. 사용자가 직접 확인할 수 있어야 한다.
    evidence: str = ""


@dataclass(frozen=True)
class Mission:
    name: str
    stages: tuple[Stage, ...]
    #: 마지막 단계 뒤 돌아갈 단계. 비어 있으면 반복하지 않는다.
    loop_to: str = ""
    artifacts: Mapping[str, str] = field(default_factory=dict)
    checkpoints: Mapping[str, Mapping[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class MissionState:
    stage: str
    status: str = STATUS_PENDING
    completed: tuple[str, ...] = ()
    cycle: int = 0
    note: str = ""


@dataclass(frozen=True)
class Evidence:
    """가드가 판정에 쓰는 **바깥 세계의 사실**. 조사는 호출자가 한다."""

    present_artifacts: frozenset[str] = frozenset()
    #: 체크포인트 키 → 실제 md5. 파일이 없으면 키가 아예 빠진다.
    checkpoint_digests: Mapping[str, str] = field(default_factory=dict)
    #: params dump 가 있는 체크포인트 키.
    present_params: frozenset[str] = frozenset()
    #: 사용자가 승인한 단계 id.
    approvals: frozenset[str] = frozenset()


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class StagePlan:
    stage: Stage
    result: GateResult


# ── 미션 구조 ──────────────────────────────────────────────────────────────
def stage_by_id(mission: Mission, stage_id: str) -> Stage:
    for stage in mission.stages:
        if stage.id == stage_id:
            return stage
    raise KeyError(f"없는 단계: {stage_id}")


def _index_of(mission: Mission, stage_id: str) -> int:
    for i, stage in enumerate(mission.stages):
        if stage.id == stage_id:
            return i
    raise KeyError(f"없는 단계: {stage_id}")


def next_stage_id(mission: Mission, stage_id: str) -> str | None:
    """다음 단계. 마지막이면 `loop_to`, 반복하지 않는 미션이면 None."""
    i = _index_of(mission, stage_id)
    if i + 1 < len(mission.stages):
        return mission.stages[i + 1].id
    return mission.loop_to or None


def initial_state(mission: Mission) -> MissionState:
    if not mission.stages:
        raise ValueError("단계가 하나도 없는 미션은 시작할 수 없다")
    return MissionState(stage=mission.stages[0].id)


# ── 가드 ───────────────────────────────────────────────────────────────────
def _blocked_reasons(stage: Stage) -> list[str]:
    if not stage.blocked:
        return []
    reasons = [f"아직 막혀 있다 — {stage.blocked}"]
    if stage.evidence:
        reasons.append(f"근거: {stage.evidence}")
    return reasons


def _order_reasons(stage: Stage, state: MissionState) -> list[str]:
    missing = [need for need in stage.needs if need not in state.completed]
    if not missing:
        return []
    reasons = [f"선행 단계 '{need}' 가 아직 끝나지 않았다" for need in missing]
    if stage.needs_why:
        reasons.append(f"이 순서가 필요한 이유: {stage.needs_why}")
    return reasons


def _artifact_reasons(mission: Mission, stage: Stage, evidence: Evidence) -> list[str]:
    return [
        f"산출물이 없다: {mission.artifacts.get(key, key)}  (키 {key})"
        for key in stage.artifacts
        if key not in evidence.present_artifacts
    ]


def _one_checkpoint_reasons(spec: Mapping[str, str], key: str, evidence: Evidence) -> list[str]:
    path, expected = spec.get("path", key), str(spec.get("md5", ""))
    actual = evidence.checkpoint_digests.get(key)
    reasons: list[str] = []
    if actual is None:
        reasons.append(f"체크포인트가 없다: {path}  (키 {key})")
    elif not actual.lower().startswith(expected.lower()):
        reasons.append(
            f"체크포인트 md5 가 다르다 — 미션은 {expected}, 실제는 {actual}  ({path})"
        )
    if key not in evidence.present_params:
        reasons.append(
            f"런 params dump 가 없다: {spec.get('params', '?')}  "
            "— 없으면 홈을 알 수 없고, 홈이 틀리면 preset 이 엉뚱한 곳에 도착한다"
        )
    return reasons


def _checkpoint_reasons(mission: Mission, stage: Stage, evidence: Evidence) -> list[str]:
    reasons: list[str] = []
    for key in stage.checkpoints:
        reasons += _one_checkpoint_reasons(mission.checkpoints.get(key, {}), key, evidence)
    return reasons


def _approval_reasons(stage: Stage, evidence: Evidence) -> list[str]:
    if not stage.touches_real or stage.id in evidence.approvals:
        return []
    return [f"실기를 움직이는 단계다 — 승인이 없다  (--approve {stage.id})"]


def gate(
    mission: Mission, stage_id: str, state: MissionState, evidence: Evidence
) -> GateResult:
    """이 단계를 지금 실행해도 되는가. **막힌 것을 전부** 모아서 낸다."""
    stage = stage_by_id(mission, stage_id)
    reasons = (
        _blocked_reasons(stage)
        + _order_reasons(stage, state)
        + _artifact_reasons(mission, stage, evidence)
        + _checkpoint_reasons(mission, stage, evidence)
        + _approval_reasons(stage, evidence)
    )
    return GateResult(ok=not reasons, reasons=tuple(reasons))


def plan(mission: Mission, state: MissionState, evidence: Evidence) -> tuple[StagePlan, ...]:
    """전 단계를 훑는다.

    각 단계는 **앞 단계가 다 끝났다고 치고** 판정한다. 지금 못 한다는 이유로 전부
    빨갛게 칠하면 진짜 막힌 것이 묻힌다 — 이 표를 보는 목적은 "무엇을 더 만들어야
    하는가"를 아는 것이지 "지금 어디까지 왔는가"가 아니다.
    """
    rows = []
    for i, stage in enumerate(mission.stages):
        as_if = replace(state, completed=tuple(s.id for s in mission.stages[:i]))
        rows.append(StagePlan(stage=stage, result=gate(mission, stage.id, as_if, evidence)))
    return tuple(rows)


# ── 전이 ───────────────────────────────────────────────────────────────────
def begin(state: MissionState) -> MissionState:
    return replace(state, status=STATUS_RUNNING, note="")


def _completed_after_loop(mission: Mission, completed: tuple[str, ...]) -> tuple[str, ...]:
    """반복 지점 뒤의 단계를 잊는다.

    다시 해야 하는 단계가 `completed` 에 남아 있으면 다음 사이클의 가드가 눈이 먼다.
    """
    if not mission.loop_to:
        return completed
    keep = {s.id for s in mission.stages[: _index_of(mission, mission.loop_to)]}
    return tuple(sid for sid in completed if sid in keep)


def advance(
    mission: Mission, state: MissionState, outcome: str, *, note: str = ""
) -> MissionState:
    """단계 하나가 끝났다. 성공이면 다음으로, 아니면 그 자리에 멈춘다."""
    if outcome not in _OUTCOMES:
        raise ValueError(f"모르는 결과: {outcome} (가능한 값 {_OUTCOMES})")
    if outcome != STATUS_DONE:
        return replace(state, status=outcome, note=note)

    completed = state.completed + (state.stage,)
    nxt = next_stage_id(mission, state.stage)
    if nxt is None:
        return replace(state, status=STATUS_DONE, completed=completed, note=note)

    looped = _index_of(mission, nxt) <= _index_of(mission, state.stage)
    return MissionState(
        stage=nxt,
        status=STATUS_PENDING,
        completed=_completed_after_loop(mission, completed) if looped else completed,
        cycle=state.cycle + (1 if looped else 0),
        note=note,
    )


# ── yaml 적재 (경계에서 검증한다) ──────────────────────────────────────────
def _stage_from_raw(raw: Mapping) -> Stage:
    if not raw.get("id"):
        raise ValueError(f"id 가 없는 단계가 있다: {raw}")
    return Stage(
        id=str(raw["id"]),
        title=str(raw.get("title", raw["id"])),
        needs=tuple(raw.get("needs", ()) or ()),
        needs_why=str(raw.get("needs_why", "")),
        artifacts=tuple(raw.get("artifacts", ()) or ()),
        checkpoints=tuple(raw.get("checkpoints", ()) or ()),
        touches_real=bool(raw.get("touches_real", False)),
        blocked=str(raw.get("blocked", "")),
        evidence=str(raw.get("evidence", "")),
    )


def _validate_checkpoints(checkpoints: Mapping[str, Mapping[str, str]]) -> None:
    for key, spec in checkpoints.items():
        missing = [k for k in _CHECKPOINT_KEYS if not spec.get(k)]
        if missing:
            raise ValueError(f"체크포인트 '{key}' 에 {missing} 가 없다")


def _validate_stage_refs(mission: Mission) -> None:
    ids = [s.id for s in mission.stages]
    for i, stage in enumerate(mission.stages):
        for need in stage.needs:
            if need not in ids:
                raise ValueError(f"단계 '{stage.id}' 의 선행 '{need}' 가 미션에 없다")
            if ids.index(need) >= i:
                raise ValueError(
                    f"단계 '{stage.id}' 가 뒤에 오는 '{need}' 를 선행으로 두었다 — 영영 못 간다"
                )
        for key in stage.artifacts:
            if key not in mission.artifacts:
                raise ValueError(f"단계 '{stage.id}' 의 산출물 키 '{key}' 가 선언되지 않았다")
        for key in stage.checkpoints:
            if key not in mission.checkpoints:
                raise ValueError(f"단계 '{stage.id}' 의 체크포인트 키 '{key}' 가 선언되지 않았다")


def load_mission(raw: Mapping) -> Mission:
    """미션 정의(dict)를 검증하며 적재한다. 파일 읽기는 호출자 몫이다."""
    stages = tuple(_stage_from_raw(s) for s in raw.get("stages", ()))
    if not stages:
        raise ValueError("stages 가 비어 있다")
    ids = [s.id for s in stages]
    dupes = sorted({sid for sid in ids if ids.count(sid) > 1})
    if dupes:
        raise ValueError(f"단계 id 가 중복이다: {dupes}")

    checkpoints = dict(raw.get("checkpoints", {}) or {})
    _validate_checkpoints(checkpoints)

    loop_to = str(raw.get("loop_to", ""))
    if loop_to and loop_to not in ids:
        raise ValueError(f"loop_to '{loop_to}' 가 미션의 단계가 아니다")

    mission = Mission(
        name=str(raw.get("name", "mission")),
        stages=stages,
        loop_to=loop_to,
        artifacts=dict(raw.get("artifacts", {}) or {}),
        checkpoints=checkpoints,
    )
    _validate_stage_refs(mission)
    return mission
