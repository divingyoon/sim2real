#!/usr/bin/env python3
"""정책의 **리셋 자세**를 어디서 가져올지 정하는 순수 코어 (ROS 무의존).

정책을 실행하기 전에 sim 과 실기가 같은 자세에서 출발해야 한다. 그런데 "같은 자세"가
무엇인지에 답이 두 개 있고 **둘이 다를 수 있다**:

  ① 기록이 아는 자세 — 그 체크포인트가 실제로 출발했던 곳.
  ② 현재 preset 의 홈 — 지금 코드가 출발시키는 곳.

둘이 갈리면 ①이 맞다. 정책은 학습 당시 홈에서 출발하는 gradient flow 를 배웠고,
fabric 의 cspace rest(`default_config_override`)도 그 홈이다. 현재 홈으로 파킹한 뒤
옛 기록을 재생하면 첫 프레임이 **도약**이 된다.

그래서 기본값은 ①이고, ②는 항상 **대조해서 다르면 말한다**. 조용히 하나를 고르지 않는다.

⚠ 리셋 자세는 백의 첫 프레임과 다르다. 첫 프레임은 이미 액션 한 스텝이 들어간 뒤의
fabric 출력이다. 파킹 목표는 **홈**이지 첫 프레임이 아니다.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: 두 출처의 홈이 이보다 벌어지면 "다른 자세"로 본다 [rad].
#: 관절 하나가 이만큼 틀어지면 손끝은 이미 센티미터 단위로 다른 곳에 있다.
HOME_AGREEMENT_TOL_RAD = 0.01

#: 실측이 목표에서 이보다 벌어져 있으면 "파킹 안 됨"으로 본다 [rad].
PARKED_TOL_RAD = 0.02


@dataclass(frozen=True)
class HomePose:
    """관절 홈 + **어디서 왔는지**. 출처를 잃으면 대조가 불가능해진다."""

    joints: dict[str, float]
    source: str
    derived: bool = False  # True = 직접 적힌 값이 아니라 실측에서 유도했다

    def as_array(self, order: list[str]) -> np.ndarray:
        missing = [n for n in order if n not in self.joints]
        if missing:
            raise KeyError(f"{self.source}: 관절 {missing} 이 없다")
        return np.array([self.joints[n] for n in order], dtype=np.float64)


@dataclass(frozen=True)
class Disagreement:
    joint: str
    a: float
    b: float

    @property
    def delta(self) -> float:
        return self.b - self.a


def home_from_recording(npz: dict, *, joint_names: list[str] | None = None) -> HomePose:
    """기록에서 홈을 꺼낸다. `meta_home_q` 가 있으면 그것, 없으면 step0 실측에서 유도."""
    names = joint_names or [str(x) for x in npz["meta_joint_names"]]
    if "meta_home_q" in npz:
        q = np.asarray(npz["meta_home_q"]).reshape(-1)
        if q.size != len(names):
            raise ValueError(
                f"meta_home_q 길이 {q.size} 가 관절 수 {len(names)} 와 다르다"
            )
        return HomePose({n: float(v) for n, v in zip(names, q)},
                        source="기록 meta_home_q")
    if "arm_meas" not in npz:
        raise KeyError(
            "기록에 meta_home_q 도 arm_meas 도 없다 — 홈을 지어내지 않는다. "
            "meta_home_q 를 남기는 최신 probe 로 다시 기록하라."
        )
    q = np.asarray(npz["arm_meas"])
    if q.ndim == 3:
        q = q[:, 0, :]
    return HomePose({n: float(v) for n, v in zip(names, q[0])},
                    source="기록 step0 실측(유도)", derived=True)


def home_from_preset(preset_path: str | Path, name: str) -> HomePose:
    """preset 소스에서 dict 상수를 AST 로 읽는다 (isaaclab import 없이)."""
    path = Path(preset_path)
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == name:
                value = ast.literal_eval(node.value)
                if not isinstance(value, dict):
                    raise TypeError(f"{name} 이 dict 가 아니다: {type(value)}")
                return HomePose({k: float(v) for k, v in value.items()},
                                source=f"preset {name}")
    raise KeyError(f"{path.name} 에 {name} 이 없다")


def disagreements(a: HomePose, b: HomePose, *,
                  tol: float = HOME_AGREEMENT_TOL_RAD) -> list[Disagreement]:
    """두 홈의 불일치. 한쪽에만 있는 관절은 비교 대상이 아니다(있는 것끼리만)."""
    shared = [n for n in a.joints if n in b.joints]
    return [
        Disagreement(n, a.joints[n], b.joints[n])
        for n in shared
        if abs(b.joints[n] - a.joints[n]) > tol
    ]


def not_parked(measured: np.ndarray, target: np.ndarray, names: list[str], *,
               tol: float = PARKED_TOL_RAD) -> list[Disagreement]:
    """실측이 목표에서 벗어난 관절. 빈 리스트여야 정책을 시작할 수 있다."""
    if measured.shape != target.shape:
        raise ValueError(f"실측 {measured.shape} 와 목표 {target.shape} 모양이 다르다")
    return [
        Disagreement(n, float(target[i]), float(measured[i]))
        for i, n in enumerate(names)
        if abs(measured[i] - target[i]) > tol
    ]


def describe_disagreements(rows: list[Disagreement], *,
                           a_label: str, b_label: str) -> str:
    if not rows:
        return "  (일치)"
    width = max(len(r.joint) for r in rows)
    head = f"  {'관절':<{width}s} {a_label:>12s} {b_label:>12s} {'차이(mrad)':>12s}"
    body = [
        f"  {r.joint:<{width}s} {r.a:12.4f} {r.b:12.4f} {r.delta*1000:12.1f}"
        for r in rows
    ]
    return "\n".join([head, *body])
