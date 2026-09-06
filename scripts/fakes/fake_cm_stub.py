#!/usr/bin/env python3
"""controller_manager 스텁 — robot_control bringup(양팔 하나의 controller_manager)을 흉내내 pd 노드의
engage/release 순서를 검증한다.

시작 상태 = 실기 bringup 과 동일: 팔마다 JTC active, forward 3종은 **로드되지 않음**. pd 노드는
load → configure → switch(STRICT) 로 forward 를 잡고, release 때 JTC 로 되돌린다.
STRICT 규칙: 이름을 모르거나 이미 그 상태면 실패(ok=False). 팔은 서로 독립이다(한 팔의 교대가 다른 팔을 안 건드린다).

`ControllerManagerStub(node, "left")` (한 팔) 또는 `ControllerManagerStub(node, ("right", "left"))` (양팔).
"""
from __future__ import annotations

from typing import Sequence

FORWARD_KINDS = ("position", "velocity", "effort")
STATES = ("unconfigured", "inactive", "active")


def jtc_of(side: str) -> str:
    return f"{side}_joint_trajectory_controller"


def forward_of(side: str, kind: str) -> str:
    return f"{side}_forward_{kind}_controller"


def forward_topic(side: str, kind: str) -> str:
    return f"/{forward_of(side, kind)}/commands"


class ControllerManagerStub:
    """list/load/configure/switch 서비스 4종. 상태는 `known` (controller → state)."""

    STATES = STATES

    def __init__(self, node, sides: str | Sequence[str]) -> None:
        from controller_manager_msgs.srv import (ConfigureController, ListControllers,
                                                 LoadController, SwitchController)
        self.sides = (sides,) if isinstance(sides, str) else tuple(sides)
        if not self.sides:
            raise ValueError("ControllerManagerStub needs at least one side")
        self.side = self.sides[0]                       # 한 팔 호환(name/topic/jtc)
        self.jtc = jtc_of(self.side)
        self.known = {**{jtc_of(s): "active" for s in self.sides}, "joint_state_broadcaster": "active"}
        self.loadable = {forward_of(s, k) for s in self.sides for k in FORWARD_KINDS}
        self.log = node.get_logger()
        node.create_service(ListControllers, "/controller_manager/list_controllers", self._list)
        node.create_service(LoadController, "/controller_manager/load_controller", self._load)
        node.create_service(ConfigureController, "/controller_manager/configure_controller", self._configure)
        node.create_service(SwitchController, "/controller_manager/switch_controller", self._switch)

    # ---------------------------------------------------------------- names
    def jtc_of(self, side: str) -> str:
        return jtc_of(side)

    def name_of(self, side: str, kind: str) -> str:
        return forward_of(side, kind)

    def topic_of(self, side: str, kind: str) -> str:
        return forward_topic(side, kind)

    def name(self, kind: str) -> str:
        return forward_of(self.side, kind)

    def topic(self, kind: str) -> str:
        return forward_topic(self.side, kind)

    def is_active(self, name: str) -> bool:
        return self.known.get(name) == "active"

    def forward_active(self, side: str, kind: str) -> bool:
        return self.is_active(forward_of(side, kind))

    # ---------------------------------------------------------------- services
    def _list(self, req, res):
        from controller_manager_msgs.msg import ControllerState
        for name, state in self.known.items():
            cs = ControllerState()
            cs.name, cs.state = name, state
            cs.type = ("forward_command_controller/ForwardCommandController" if "forward" in name
                       else "joint_trajectory_controller/JointTrajectoryController")
            res.controller.append(cs)
        return res

    def _load(self, req, res):
        res.ok = req.name in self.loadable and req.name not in self.known
        if res.ok:
            self.known[req.name] = "unconfigured"
        self.log.info(f"[cm] load {req.name} → {res.ok}")
        return res

    def _configure(self, req, res):
        res.ok = self.known.get(req.name) == "unconfigured"
        if res.ok:
            self.known[req.name] = "inactive"
        self.log.info(f"[cm] configure {req.name} → {res.ok}")
        return res

    def _switch(self, req, res):
        act = list(req.activate_controllers) + list(req.start_controllers)
        deact = list(req.deactivate_controllers) + list(req.stop_controllers)
        bad = [n for n in act if self.known.get(n) != "inactive"] + \
              [n for n in deact if self.known.get(n) != "active"]
        if bad and req.strictness == req.STRICT:
            res.ok = False
            self.log.error(f"[cm] switch STRICT 거부: {bad}")
            return res
        for n in deact:
            if self.known.get(n) == "active":
                self.known[n] = "inactive"
        for n in act:
            if self.known.get(n) == "inactive":
                self.known[n] = "active"
        res.ok = True
        self.log.info(f"[cm] switch ok · active={[n for n, s in self.known.items() if s == 'active']}")
        return res
