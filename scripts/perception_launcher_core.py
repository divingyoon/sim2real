#!/usr/bin/env python3
"""perception_launcher_node 의 순수 로직 — 명령 파싱·액션 계획·상태 집계.

ROS·ssh 없이 import 된다. test_perception_launcher_core.py 대상.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from object_registry import CONTAINER_PREFIX, container_name  # noqa: E402

_OPS = ("start", "stop", "viewer")


@dataclass(frozen=True)
class Command:
    op: str
    objects: tuple[str, ...]
    viewer: bool | None
    camera: bool


@dataclass(frozen=True)
class RemoteState:
    camera_up: bool
    containers: dict[str, str]
    viewer_up: bool


def parse_command(text: str, registry) -> Command:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as err:
        raise ValueError(f"command is not JSON: {err}") from err
    if not isinstance(raw, dict) or raw.get("op") not in _OPS:
        raise ValueError(f"command 'op' must be one of {_OPS}: {text!r}")
    op = raw["op"]
    objects: tuple[str, ...] = ()
    if op == "start":
        names = raw.get("objects")
        if not isinstance(names, list) or not names:
            raise ValueError("start needs a non-empty 'objects' list")
        seen: list[str] = []
        for name in names:
            canon = registry.resolve(str(name))
            if canon not in seen:
                seen.append(canon)
        objects = tuple(seen)
    viewer = raw.get("viewer") if op == "start" else raw.get("on")
    if viewer is not None and not isinstance(viewer, bool):
        raise ValueError("'viewer'/'on' must be a boolean")
    if op == "viewer" and viewer is None:
        raise ValueError("viewer needs 'on': true|false")
    return Command(op=op, objects=objects, viewer=viewer, camera=bool(raw.get("camera", False)))


def parse_remote_status(text: str) -> RemoteState:
    try:
        raw = json.loads(text.strip().splitlines()[-1]) if text.strip() else None
    except json.JSONDecodeError:
        raw = None
    if not isinstance(raw, dict) or "containers" not in raw:
        raise ValueError(f"remote status is not the expected JSON: {text[:200]!r}")
    return RemoteState(camera_up=bool(raw.get("camera_up")),
                       containers={str(k): str(v) for k, v in raw["containers"].items()},
                       viewer_up=bool(raw.get("viewer_up")))


def plan_actions(cmd: Command, state: RemoteState) -> list[tuple[str, ...]]:
    actions: list[tuple[str, ...]] = []
    if cmd.op == "start":
        if not state.camera_up:
            actions.append(("camera_up",))
        wanted = {container_name(n) for n in cmd.objects}
        for cname in sorted(state.containers):
            if cname.startswith(CONTAINER_PREFIX) and cname not in wanted:
                actions.append(("fpp_down", cname))
        for name in cmd.objects:
            if not state.containers.get(container_name(name), "").startswith("Up"):
                actions.append(("fpp_up", name))
        if cmd.viewer is True and not state.viewer_up:
            actions.append(("viewer_up",))
        if cmd.viewer is False and state.viewer_up:
            actions.append(("viewer_down",))
        return actions
    if cmd.op == "stop":
        if state.viewer_up:
            actions.append(("viewer_down",))
        for cname in sorted(state.containers):
            if cname.startswith(CONTAINER_PREFIX):
                actions.append(("fpp_down", cname))
        if cmd.camera and state.camera_up:
            actions.append(("camera_down",))
        return actions
    if cmd.viewer and not state.viewer_up:
        actions.append(("viewer_up",))
    if cmd.viewer is False and state.viewer_up:
        actions.append(("viewer_down",))
    return actions


def build_status(state: RemoteState | None, camera_hz: float, pose_ages: dict[str, float | None],
                 busy: bool, error: str | None) -> dict:
    objects = {}
    for name, age in pose_ages.items():
        cont = state.containers.get(container_name(name)) if state else None
        objects[name] = {"container": cont, "pose_age_s": age}
    return {
        "camera_up": bool(state.camera_up) if state else None,
        "camera_hz": round(float(camera_hz), 2),
        "objects": objects,
        "viewer": bool(state.viewer_up) if state else None,
        "busy": bool(busy),
        "error": error,
    }
