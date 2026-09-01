#!/usr/bin/env python3
"""hdgp 태스크의 **관측 계약을 소스에서 자동 추출**한다 (ast 만 사용, ROS·torch 무의존).

왜 필요한가: hdgp 에는 관측을 정의하는 태스크가 22개 있고(grasp_v1/v2/v7_2/v10_3/v11/
adapt/sensor · pour_v1/v3/v4/v5/sensor · rh56f1 · gripper · agnostic …), 어느 것을 배포
대상으로 삼든 그 태스크의 obs 계약을 손으로 옮겨 적는 순간 드리프트가 시작된다. 실측
결과 **22/22 태스크가 `torch.cat([...])` 리터럴 리스트**를 쓰므로 구조는 기계적으로 뽑힌다.

무엇이 자동으로 나오고, 무엇이 안 나오는가 (정직하게):
  ✅ 세그먼트 **순서·이름**            — 재배열/개명/추가를 즉시 잡는다
  ✅ 각 세그먼트의 **정의식**          — 프레임 힌트(`_local`), 차분(`a - b`)이 보인다
  ✅ **정규화 상수** (ALL_CAPS)        — 값이 바뀌면 대조로 잡힌다
  ✅ **DR 노이즈** 적용 여부           — 배포는 노이즈를 더하면 안 된다(실센서가 노이즈다)
  ❌ 세그먼트 **차원**                 — 대부분 상위 스코프/`self.*` 라 정적으로 못 센다
  ❌ **프레임·단위의 의미**            — world vs local, rad vs deg 는 사람이 판단해야 한다
차원과 의미는 런타임 덤프 대조(P3)로만 확정된다. 이 도구는 그 앞단의 **뼈대**를 준다.

구 정규식 추출기는 단순 식별자가 아닌 항을 **조용히 버려서** 세그먼트 수가 어긋났다.
여기서는 표현식도 세그먼트로 보존한다.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

#: obs 를 담는 변수 이름 후보. 앞에서부터 먼저 발견된 것을 쓴다.
OBS_VAR_CANDIDATES = ("actor_obs", "policy_obs", "obs")
#: DR 노이즈로 간주하는 호출.
NOISE_CALLS = ("randn_like", "randn", "rand_like", "rand", "uniform_", "normal_")


class ObsContractError(RuntimeError):
    """계약을 추출할 수 없을 때. 추측해서 반쯤 맞는 계약을 내놓지 않는다."""


@dataclass(frozen=True)
class ObsSegment:
    index: int
    name: str                    # 단순 식별자면 그 이름, 아니면 표현식 원문
    expr: str                    # 정의식(추적되면), 아니면 자기 자신
    is_named: bool               # 단순 식별자였는가
    dr_noise: bool               # sim 에서 노이즈가 더해지는가
    constants: tuple[str, ...]   # 정의식에 등장하는 ALL_CAPS 상수


@dataclass(frozen=True)
class ObsContract:
    source: str
    var: str
    segments: tuple[ObsSegment, ...]

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.segments]


def _find_obs_function(tree: ast.AST) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_get_observations":
            return node
    raise ObsContractError("`_get_observations` 를 찾지 못했다")


def _local_assignments(fn: ast.FunctionDef) -> dict[str, list[ast.AST]]:
    """함수 안 `name = <expr>` 의 **모든** 정의를 순서대로 모은다.

    마지막 정의만 담으면 안 된다 — hdgp 4개 태스크는 `actor_obs = torch.cat(...)` 뒤에
    `actor_obs = torch.nan_to_num(actor_obs, ...)` 로 덮어써서 cat 이 가려진다.
    """
    out: dict[str, list[ast.AST]] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.setdefault(t.id, []).append(node.value)
    return out


def _appended_elements(fn: ast.FunctionDef, var: str) -> list[ast.AST]:
    """`var.append(x)` 로 쌓아 만든 리스트의 원소들(등장 순서)."""
    out = []
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == var
            and node.args
        ):
            out.append(node.args[0])
    return out


def _resolve_elements(fn: ast.FunctionDef, arg: ast.AST,
                      assigns: dict[str, list[ast.AST]]) -> list[ast.AST] | None:
    """cat 의 첫 인자 → 세그먼트 원소 목록. 리스트 리터럴/변수/append 누적을 모두 처리."""
    if isinstance(arg, (ast.List, ast.Tuple)):
        return list(arg.elts)
    if isinstance(arg, ast.Name):
        for value in assigns.get(arg.id, []):
            if isinstance(value, (ast.List, ast.Tuple)) and value.elts:
                return list(value.elts)
        appended = _appended_elements(fn, arg.id)
        if appended:
            return appended
    return None


def _returned_policy_vars(fn: ast.FunctionDef) -> list[str]:
    """`return {"policy": ...}` 안에서 참조되는 지역 변수 이름들 (등장 순)."""
    out: list[str] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
            continue
        for key, val in zip(node.value.keys, node.value.values):
            if not (isinstance(key, ast.Constant) and key.value == "policy"):
                continue
            for sub in ast.walk(val):
                if isinstance(sub, ast.Name) and sub.id not in out:
                    out.append(sub.id)
    return out


def _find_obs_elements(
    fn: ast.FunctionDef, assigns: dict[str, list[ast.AST]]
) -> tuple[str, list[ast.AST]]:
    """obs 변수를 만드는 `torch.cat(...)` 의 세그먼트 원소 목록을 찾는다."""
    for var in OBS_VAR_CANDIDATES:
        for value in assigns.get(var, []):
            if not (isinstance(value, ast.Call) and getattr(value.func, "attr", None) == "cat"):
                continue
            if not value.args:
                continue
            elts = _resolve_elements(fn, value.args[0], assigns)
            if elts:
                return var, elts
    # 후보 이름에 없으면 **반환문의 `"policy"` 키**를 따라간다. obs 변수 이름은
    # 태스크마다 다르고(grasp_s2r 은 `_noisy`), 후보 목록을 손으로 늘리면 또 새는다.
    for name in _returned_policy_vars(fn):
        for value in assigns.get(name, []):
            if not (isinstance(value, ast.Call) and getattr(value.func, "attr", None) == "cat"):
                continue
            if not value.args:
                continue
            elts = _resolve_elements(fn, value.args[0], assigns)
            if elts:
                return name, elts
    raise ObsContractError(
        "obs 를 만드는 `torch.cat(...)` 을 찾지 못했다 "
        f"(찾은 변수 후보: {', '.join(OBS_VAR_CANDIDATES)}, "
        f"반환 policy: {', '.join(_returned_policy_vars(fn)) or '없음'})"
    )


def _has_noise(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            fname = getattr(n.func, "attr", None) or getattr(n.func, "id", None)
            if fname in NOISE_CALLS:
                return True
    return False


def _constants(node: ast.AST) -> tuple[str, ...]:
    """표현식에 등장하는 ALL_CAPS 식별자(정규화 상수 후보)."""
    found = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id.isupper() and len(n.id) > 2:
            found.append(n.id)
        elif isinstance(n, ast.Attribute) and n.attr.isupper() and len(n.attr) > 2:
            found.append(n.attr)
    return tuple(dict.fromkeys(found))


def extract_obs_contract(env_py: str | Path) -> ObsContract:
    """`grasp_*_env.py` / `pour_*_env.py` → 관측 계약."""
    path = Path(env_py)
    if not path.exists():
        raise FileNotFoundError(f"env 소스 없음: {path}")
    tree = ast.parse(path.read_text())
    fn = _find_obs_function(tree)
    assigns = _local_assignments(fn)
    var, elts = _find_obs_elements(fn, assigns)

    segments = []
    for i, elt in enumerate(elts):
        raw = ast.unparse(elt)
        is_named = isinstance(elt, ast.Name)
        defining = next(iter(assigns.get(elt.id, [])), None) if is_named else None
        expr_node = defining if defining is not None else elt
        segments.append(
            ObsSegment(
                index=i,
                name=raw,
                expr=ast.unparse(expr_node),
                is_named=is_named,
                dr_noise=_has_noise(expr_node),
                constants=_constants(expr_node),
            )
        )
    return ObsContract(source=str(path), var=var, segments=tuple(segments))


def diff_segments(sim_names, deploy_names) -> list[str]:
    """sim ↔ 배포 세그먼트 이름 목록 차이를 사람이 읽는 문장으로."""
    sim, dep = list(sim_names), list(deploy_names)
    if sim == dep:
        return []
    out: list[str] = []
    for n in sim:
        if n not in dep:
            out.append(f"배포에 없음: {n}")
    for n in dep:
        if n not in sim:
            out.append(f"sim 에 없음: {n}")
    if not out and sim != dep:
        out.append(f"순서 불일치\n    sim  {sim}\n    배포 {dep}")
    return out
