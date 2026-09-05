# FP++ 인지 서브시스템(로컬 ROS2 노드 3종) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** pc5090 에서 `perception_ctl.py start shaker_closed cup_big_s100 --viewer` 한 줄로 vision-3090 의 카메라·물체별 FP++ 컨테이너·모니터 창을 띄우고, 물체 pose 를 `/objects/<name>/pose`(base_link) 로 발행한다.

**Architecture:** 로컬 런처 노드가 tailscale ssh 로 vision-3090 의 repo 스크립트(`scripts/vision/*.sh`)를 실행한다. 물체 정의는 `config/objects.yaml` 레지스트리 하나가 진실원천이며 FP++ yaml 을 거기서 생성한다. 카메라 프레임 pose 는 두 PC 사이 DDS(LAN, domain 126)로 로컬에 도착하고, 로컬 `object_pose_node` 가 기존 `cup_pose_relay` 의 순수 변환 함수로 base_link 로 바꿔 발행한다.

**Tech Stack:** Python 3.10, ROS2 humble(rclpy, std_msgs/String JSON, geometry_msgs/PoseStamped), numpy, pyyaml, pytest, bash, docker(vision-3090), tailscale ssh.

**Spec:** `sim2real/docs/superpowers/specs/2026-09-03-fpp-perception-nodes-design.md`

## Global Constraints

- 모든 파일은 `rl_ws/sim2real/` 아래. 스크립트는 `scripts/`(기존 flat 관례), 설정은 `config/`, vision-3090 전용 셸은 `scripts/vision/`.
- 순수 로직은 ROS 없이 import 가능해야 한다(테스트가 `python3 -m pytest scripts/test_*.py` 로 돈다). ROS import 는 `main()`/노드 클래스 안에서만.
- 커스텀 msg 를 만들지 않는다. 명령/상태는 `std_msgs/String` JSON.
- ROS_DOMAIN_ID=126. vision-3090 호스트 이름은 ssh config 의 `vision-3090`(계정 `usr`, 경로 `/home/usr/rl_ws/...`).
- 토픽 규약: 입력 `/perception_plus_plus/<name>/pose`, 상태 `/perception_plus_plus/<name>/tracking_status`, 출력 `/objects/<name>/pose`, 명령 `/perception/cmd`, 상태 `/perception/status`.
- 컨테이너 이름 `fpp_<name>`, 이미지 `perception-plus-plus:humble-cup`, yaml 주입 경로 `/opt/params/<name>.yaml`.
- 실패는 status 의 `error` 문자열 + 로거 error 로 드러낸다. 조용한 재시도 금지.
- 커밋 메시지: `<type>: <설명>` (feat/fix/test/docs/chore), 본문 끝에
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` 와 `Claude-Session: https://claude.ai/code/session_012DQ2PQgeUuk1KhqLvPFLPE`.

## File Structure

| 파일 | 책임 |
|---|---|
| `config/objects.yaml` | 물체 레지스트리(진실원천): FP++ 파라미터·cad_to_body·sim 원점·AABB·alias |
| `scripts/object_registry.py` | 레지스트리 로드/검증/alias 해석/FP++ yaml 렌더/Extrinsics 조립 (순수) |
| `scripts/test_object_registry.py` | 위 순수 로직 pytest |
| `scripts/perception_launcher_core.py` | 명령 파싱, 원하는 상태→액션 계획, 원격 상태 파싱, status 집계 (순수) |
| `scripts/test_perception_launcher_core.py` | 위 순수 로직 pytest |
| `scripts/nodes/perception_launcher_node.py` | 노드 1: `/perception/cmd` → ssh 실행, `/perception/status` 발행 |
| `scripts/ops/perception_ctl.py` | 노드 2 의 사용자 면: CLI → `/perception/cmd` 발행, status 표 출력 |
| `scripts/nodes/object_pose_node.py` | 노드 3: 카메라 프레임 pose → base_link `/objects/<name>/pose` |
| `scripts/test_object_pose_node.py` | 노드 3 순수부(레지스트리→Extrinsics→변환) pytest |
| `scripts/vision/camera_up.sh` `camera_down.sh` | RealSense ROS 노드 기동/종료 |
| `scripts/vision/fpp_up.sh` `fpp_down.sh` | 물체별 FP++ 컨테이너 기동/종료 |
| `scripts/vision/viewer_up.sh` `viewer_down.sh` | 모니터 창(DISPLAY=:0)+MJPEG 뷰어 |
| `scripts/vision/status.sh` | 카메라·컨테이너·뷰어 상태 JSON 한 줄 |
| `scripts/vision/legacy_down.sh` | 옛 relay/UDP tx 프로세스·옛 컨테이너 정리 |
| `scripts/vision/cup_view_stream.py` (수정) | 물체 목록·AABB 를 레지스트리에서 받도록 변경 |

---

### Task 1: 물체 레지스트리 (`config/objects.yaml` + `object_registry.py`)

**Files:**
- Create: `config/objects.yaml`
- Create: `scripts/object_registry.py`
- Test: `scripts/test_object_registry.py`

**Interfaces:**
- Consumes: `cup_pose_relay.load_extrinsics(path) -> Extrinsics`, `cup_pose_relay.Extrinsics`(dataclass, frozen), `cup_pose_relay._validated_pos/_validated_quat`.
- Produces:
  - `ObjectSpec`(frozen dataclass): `name: str`, `real: str`, `fpp: dict`, `cad_to_body_pos: np.ndarray`, `cad_to_body_quat: np.ndarray`, `sim_usd: str`, `origin_above_bottom_m: float`, `aabb: tuple[tuple[float,float,float], tuple[float,float,float]]`
  - `Registry`(frozen dataclass): `objects: dict[str, ObjectSpec]`, `aliases: dict[str, str]`, `camera_extrinsics: Path`; 메서드 `resolve(name) -> str`, `get(name) -> ObjectSpec`, `names() -> list[str]`
  - `load_registry(path: Path = DEFAULT_REGISTRY) -> Registry`
  - `input_topic(name) -> str`, `status_topic(name) -> str`, `output_topic(name) -> str`, `container_name(name) -> str`
  - `render_fpp_yaml(spec: ObjectSpec) -> str`
  - `extrinsics_for(spec: ObjectSpec, camera_yaml: Path) -> Extrinsics`

- [ ] **Step 1: 레지스트리 yaml 작성**

`config/objects.yaml`:

```yaml
# 물체 레지스트리 — FP++ 인지 서브시스템의 진실원천 (2026-09-03).
# 이름 = sim 물체 이름(cup_pose_stub.json 의 "물체"). FP++ 노드 yaml 은 여기서 생성한다.
# ★FP++ 메시는 sim 자산 메시와 같은 물체여야 한다 (shaker 09.03 교훈: 조립본 CAD 는 z 파묻힘).
camera_extrinsics: config/global_camera_extrinsics.yaml   # camera: 블록만 공유, cad_to_body 는 아래 항목이 이긴다
objects:
  shaker_closed:
    real: "파란 열린 shaker (무광, 뚜껑 없음)"
    fpp:
      mesh_path: assets/meshes/shaker_sim.ply     # hdgp/scripts/tools/export_shaker_sim_mesh.py 산출
      mesh_scale_to_meters: 1.0
      cup_class_id: 41
      detection_pick: blue
      yolo_confidence: 0.35
    cad_to_body:
      position: [0.0, 0.0, 0.0]
      orientation_wxyz: [1.0, 0.0, 0.0, 0.0]
    sim:
      usd: hdgp/assets/cup/shaker_closed_rl.usd
      origin_above_bottom_m: 0.0921
    aabb: [[-0.044, -0.044, -0.0921], [0.044, 0.044, 0.0829]]    # 물체 프레임, 뷰어 3D 박스용
  cup_big_s100:
    real: "빨간 컵"
    fpp:
      mesh_path: assets/meshes/cup.obj
      mesh_scale_to_meters: 1.0
      cup_class_id: 41
      detection_pick: red
      yolo_confidence: 0.15
    cad_to_body:
      position: [0.0, 0.0, 0.0]
      orientation_wxyz: [0.707107, 0.707107, 0.0, 0.0]   # cup.obj Y-up → sim Z-up
    sim:
      usd: hdgp/assets/cup/cup_big_rl.usd
      origin_above_bottom_m: 0.0773
    aabb: [[-0.0463, -0.0773, -0.044], [0.0437, 0.1003, 0.046]]   # CAD(Y-up) 프레임 AABB
aliases:
  cup_big_s080: cup_big_s100     # sim 스케일 변형 — 실물은 하나
  cup_big_s120: cup_big_s100
```

- [ ] **Step 2: 실패하는 테스트 작성**

`scripts/test_object_registry.py`:

```python
"""object_registry 순수 로직 검증. numpy+yaml 만, ROS 불필요."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from object_registry import (  # noqa: E402
    DEFAULT_REGISTRY, container_name, extrinsics_for, input_topic, load_registry,
    output_topic, render_fpp_yaml, status_topic,
)


def test_default_registry_loads_two_real_objects():
    reg = load_registry(DEFAULT_REGISTRY)
    assert set(reg.names()) == {"shaker_closed", "cup_big_s100"}
    assert reg.get("shaker_closed").origin_above_bottom_m == pytest.approx(0.0921)


def test_alias_resolves_to_canonical_and_unknown_raises():
    reg = load_registry(DEFAULT_REGISTRY)
    assert reg.resolve("cup_big_s080") == "cup_big_s100"
    assert reg.get("cup_big_s120").name == "cup_big_s100"
    with pytest.raises(ValueError, match="unknown object"):
        reg.resolve("teapot")


def test_topic_and_container_names_follow_convention():
    assert input_topic("shaker_closed") == "/perception_plus_plus/shaker_closed/pose"
    assert status_topic("shaker_closed") == "/perception_plus_plus/shaker_closed/tracking_status"
    assert output_topic("shaker_closed") == "/objects/shaker_closed/pose"
    assert container_name("shaker_closed") == "fpp_shaker_closed"


def test_render_fpp_yaml_matches_node_parameter_schema():
    reg = load_registry(DEFAULT_REGISTRY)
    doc = yaml.safe_load(render_fpp_yaml(reg.get("shaker_closed")))
    params = doc["cup_tracking"]["ros__parameters"]
    assert params["mesh_path"] == "assets/meshes/shaker_sim.ply"
    assert params["mesh_scale_to_meters"] == 1.0
    assert params["detection_pick"] == "blue"
    assert params["pose_topic"] == "/perception_plus_plus/shaker_closed/pose"
    assert params["status_topic"] == "/perception_plus_plus/shaker_closed/tracking_status"
    assert params["child_frame_id"] == "shaker_closed"
    assert params["rgb_topic"] == "/camera/camera/color/image_raw"


def test_extrinsics_for_uses_shared_camera_but_object_cad_to_body():
    reg = load_registry(DEFAULT_REGISTRY)
    cam_yaml = DEFAULT_REGISTRY.parent / "global_camera_extrinsics.yaml"
    ext_s = extrinsics_for(reg.get("shaker_closed"), cam_yaml)
    ext_c = extrinsics_for(reg.get("cup_big_s100"), cam_yaml)
    assert np.allclose(ext_s.cam_pos, ext_c.cam_pos)
    assert np.allclose(ext_s.cad_to_body_quat, [1, 0, 0, 0])
    assert np.allclose(ext_c.cad_to_body_quat, [0.707107, 0.707107, 0, 0], atol=1e-5)
    assert ext_s.base_frame == "base_link"


def test_invalid_registry_fails_fast(tmp_path):
    bad = tmp_path / "objects.yaml"
    bad.write_text(
        "camera_extrinsics: config/global_camera_extrinsics.yaml\n"
        "objects:\n  x:\n    real: r\n    fpp: {mesh_path: a, mesh_scale_to_meters: 1.0,"
        " cup_class_id: 41, detection_pick: red, yolo_confidence: 0.3}\n"
        "    cad_to_body: {position: [0,0,0], orientation_wxyz: [2,0,0,0]}\n"
        "    sim: {usd: u, origin_above_bottom_m: 0.1}\n"
        "    aabb: [[0,0,0],[1,1,1]]\n")
    with pytest.raises(ValueError, match="not normalized"):
        load_registry(bad)
    cyc = tmp_path / "cyc.yaml"
    cyc.write_text("camera_extrinsics: c\nobjects: {}\naliases: {a: b, b: a}\n")
    with pytest.raises(ValueError, match="alias"):
        load_registry(cyc)
```

- [ ] **Step 3: 실패 확인**

Run: `cd /home/user/rl_ws/sim2real && python3 -m pytest scripts/test_object_registry.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'object_registry'`

- [ ] **Step 4: 구현**

`scripts/object_registry.py`:

```python
#!/usr/bin/env python3
"""물체 레지스트리(config/objects.yaml) — FP++ 인지 서브시스템의 진실원천.

이름 = sim 물체 이름. 한 항목이 FP++ 노드 파라미터(메시·색 pick), CAD→sim body 정합,
sim 원점 오프셋, 뷰어용 AABB 를 함께 가진다. FP++ 노드 yaml 은 `render_fpp_yaml` 로
여기서 생성한다 — 수기 yaml 두 벌을 따로 관리하다 갈라지는 일을 막기 위해서다
(09.03 shaker: 조립본 CAD 가 sim 자산과 다른 물체라 z 파묻힘·떨림).

ROS 불필요. test_object_registry.py 대상.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from cup_pose_relay import Extrinsics, _validated_pos, _validated_quat, load_extrinsics  # noqa: E402

DEFAULT_REGISTRY = _SCRIPT_DIR.parent / "config" / "objects.yaml"
INPUT_NS = "/perception_plus_plus"
OUTPUT_NS = "/objects"
CONTAINER_PREFIX = "fpp_"
_FPP_KEYS = ("mesh_path", "mesh_scale_to_meters", "cup_class_id",
             "detection_pick", "yolo_confidence")
_PICKS = ("confidence", "dark", "bright", "red", "blue")


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    real: str
    fpp: dict
    cad_to_body_pos: np.ndarray
    cad_to_body_quat: np.ndarray
    sim_usd: str
    origin_above_bottom_m: float
    aabb: tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class Registry:
    objects: dict[str, ObjectSpec]
    aliases: dict[str, str]
    camera_extrinsics: Path

    def names(self) -> list[str]:
        return list(self.objects)

    def resolve(self, name: str) -> str:
        if name in self.objects:
            return name
        if name in self.aliases:
            return self.aliases[name]
        raise ValueError(f"unknown object '{name}' (known: {sorted(self.objects)}, "
                         f"aliases: {sorted(self.aliases)})")

    def get(self, name: str) -> ObjectSpec:
        return self.objects[self.resolve(name)]


def input_topic(name: str) -> str:
    return f"{INPUT_NS}/{name}/pose"


def status_topic(name: str) -> str:
    return f"{INPUT_NS}/{name}/tracking_status"


def output_topic(name: str) -> str:
    return f"{OUTPUT_NS}/{name}/pose"


def container_name(name: str) -> str:
    return f"{CONTAINER_PREFIX}{name}"


def _parse_object(name: str, raw: dict) -> ObjectSpec:
    for key in ("real", "fpp", "cad_to_body", "sim", "aabb"):
        if key not in raw:
            raise ValueError(f"objects.{name}: missing key '{key}'")
    fpp = dict(raw["fpp"])
    for key in _FPP_KEYS:
        if key not in fpp:
            raise ValueError(f"objects.{name}.fpp: missing key '{key}'")
    if fpp["detection_pick"] not in _PICKS:
        raise ValueError(f"objects.{name}.fpp.detection_pick must be one of {_PICKS}")
    cad = raw["cad_to_body"]
    aabb = raw["aabb"]
    if len(aabb) != 2 or any(len(c) != 3 for c in aabb):
        raise ValueError(f"objects.{name}.aabb must be [[x,y,z],[x,y,z]]")
    return ObjectSpec(
        name=name, real=str(raw["real"]), fpp=fpp,
        cad_to_body_pos=_validated_pos(cad["position"], f"objects.{name}.cad_to_body.position"),
        cad_to_body_quat=_validated_quat(cad["orientation_wxyz"],
                                         f"objects.{name}.cad_to_body.orientation_wxyz"),
        sim_usd=str(raw["sim"]["usd"]),
        origin_above_bottom_m=float(raw["sim"]["origin_above_bottom_m"]),
        aabb=(tuple(float(v) for v in aabb[0]), tuple(float(v) for v in aabb[1])),
    )


def _validate_aliases(aliases: dict, objects: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for alias, target in (aliases or {}).items():
        if alias in objects:
            raise ValueError(f"alias '{alias}' collides with an object name")
        if target not in objects:
            raise ValueError(f"alias '{alias}' → '{target}' is not a registered object")
        out[str(alias)] = str(target)
    return out


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> Registry:
    path = Path(path)
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict) or "objects" not in cfg or "camera_extrinsics" not in cfg:
        raise ValueError(f"{path}: need 'camera_extrinsics' and 'objects' keys")
    objects = {str(n): _parse_object(str(n), raw) for n, raw in (cfg["objects"] or {}).items()}
    aliases = _validate_aliases(cfg.get("aliases"), objects)
    cam = Path(cfg["camera_extrinsics"])
    if not cam.is_absolute():
        cam = path.parent.parent / cam
    return Registry(objects=objects, aliases=aliases, camera_extrinsics=cam)


def render_fpp_yaml(spec: ObjectSpec) -> str:
    """FP++ ROS 노드(cup_tracking)의 parameters_file 내용. 최상위 키는 노드 이름."""
    params = {
        "rgb_topic": "/camera/camera/color/image_raw",
        "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
        "camera_info_topic": "/camera/camera/color/camera_info",
        "pose_topic": input_topic(spec.name),
        "status_topic": status_topic(spec.name),
        "child_frame_id": spec.name,
        "mesh_path": str(spec.fpp["mesh_path"]),
        "mesh_scale_to_meters": float(spec.fpp["mesh_scale_to_meters"]),
        "yolo_weights": "models/yolo/yolov8m-seg.pt",
        "cup_class_id": int(spec.fpp["cup_class_id"]),
        "detection_pick": str(spec.fpp["detection_pick"]),
        "yolo_confidence": float(spec.fpp["yolo_confidence"]),
        "tracking_config": "config/cup_tracking.yaml",
        "sync_slop_seconds": 0.04,
        "sync_queue_size": 10,
    }
    header = (f"# 생성됨 — sim2real/config/objects.yaml 의 '{spec.name}' 항목. 손으로 고치지 말 것.\n"
              f"# 실물: {spec.real}\n")
    return header + yaml.safe_dump({"cup_tracking": {"ros__parameters": params}},
                                   sort_keys=False, allow_unicode=True)


def extrinsics_for(spec: ObjectSpec, camera_yaml: str | Path) -> Extrinsics:
    """공유 camera 블록 + 이 물체의 cad_to_body."""
    base = load_extrinsics(camera_yaml)
    return replace(base, cad_to_body_pos=spec.cad_to_body_pos,
                   cad_to_body_quat=spec.cad_to_body_quat)
```

- [ ] **Step 5: 통과 확인**

Run: `cd /home/user/rl_ws/sim2real && python3 -m pytest scripts/test_object_registry.py -q`
Expected: 6 passed

- [ ] **Step 6: 커밋**

```bash
cd /home/user/rl_ws/sim2real
git add config/objects.yaml scripts/object_registry.py scripts/test_object_registry.py
git commit -m "feat(perception): 물체 레지스트리 objects.yaml + FP++ yaml 렌더"
```

---

### Task 2: 런처 순수 로직 (`perception_launcher_core.py`)

**Files:**
- Create: `scripts/perception_launcher_core.py`
- Test: `scripts/test_perception_launcher_core.py`

**Interfaces:**
- Consumes: `Registry.resolve(name)` (Task 1).
- Produces:
  - `Command`(frozen): `op: str`("start"|"stop"|"viewer"), `objects: tuple[str, ...]`(canonical), `viewer: bool | None`, `camera: bool`
  - `parse_command(text: str, registry) -> Command` — 잘못되면 `ValueError`
  - `RemoteState`(frozen): `camera_up: bool`, `containers: dict[str, str]`(컨테이너 이름→docker Status 문자열), `viewer_up: bool`
  - `parse_remote_status(text: str) -> RemoteState` — `status.sh` 의 JSON 한 줄
  - `plan_actions(cmd: Command, state: RemoteState) -> list[tuple[str, ...]]` — 액션은 `("camera_up",)`, `("fpp_up", name)`, `("fpp_down", container)`, `("viewer_up",)`, `("viewer_down",)`, `("camera_down",)` 순서 보장
  - `build_status(state: RemoteState | None, camera_hz: float, pose_ages: dict[str, float | None], busy: bool, error: str | None) -> dict`

- [ ] **Step 1: 실패하는 테스트 작성**

`scripts/test_perception_launcher_core.py`:

```python
"""perception_launcher_core 순수 로직 검증. ROS·ssh 불필요."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from object_registry import DEFAULT_REGISTRY, load_registry  # noqa: E402
from perception_launcher_core import (  # noqa: E402
    Command, RemoteState, build_status, parse_command, parse_remote_status, plan_actions,
)

REG = load_registry(DEFAULT_REGISTRY)


def test_parse_start_resolves_aliases_and_dedups():
    cmd = parse_command(json.dumps({"op": "start", "objects": ["cup_big_s080", "shaker_closed",
                                                                "cup_big_s100"], "viewer": True}), REG)
    assert cmd == Command(op="start", objects=("cup_big_s100", "shaker_closed"), viewer=True, camera=False)


def test_parse_rejects_bad_input():
    with pytest.raises(ValueError, match="op"):
        parse_command(json.dumps({"objects": ["shaker_closed"]}), REG)
    with pytest.raises(ValueError, match="unknown object"):
        parse_command(json.dumps({"op": "start", "objects": ["teapot"]}), REG)
    with pytest.raises(ValueError, match="objects"):
        parse_command(json.dumps({"op": "start", "objects": []}), REG)
    with pytest.raises(ValueError, match="JSON"):
        parse_command("not json", REG)


def test_parse_stop_and_viewer():
    assert parse_command('{"op":"stop","camera":true}', REG) == Command("stop", (), None, True)
    assert parse_command('{"op":"viewer","on":false}', REG) == Command("viewer", (), False, False)


def test_parse_remote_status():
    st = parse_remote_status('{"camera_up": true, "containers": {"fpp_shaker_closed": "Up 3 minutes"},'
                             ' "viewer_up": false}')
    assert st == RemoteState(camera_up=True, containers={"fpp_shaker_closed": "Up 3 minutes"},
                             viewer_up=False)
    with pytest.raises(ValueError, match="status"):
        parse_remote_status("garbage")


def test_plan_start_is_idempotent_and_prunes_extras():
    state = RemoteState(camera_up=True, containers={"fpp_shaker_closed": "Up 1 minute",
                                                    "fpp_old": "Up 9 minutes"}, viewer_up=False)
    cmd = Command("start", ("shaker_closed", "cup_big_s100"), viewer=True, camera=False)
    assert plan_actions(cmd, state) == [("fpp_down", "fpp_old"), ("fpp_up", "cup_big_s100"),
                                        ("viewer_up",)]


def test_plan_start_cold_brings_camera_first():
    state = RemoteState(camera_up=False, containers={}, viewer_up=False)
    cmd = Command("start", ("shaker_closed",), viewer=None, camera=False)
    assert plan_actions(cmd, state) == [("camera_up",), ("fpp_up", "shaker_closed")]


def test_plan_stop_tears_down_everything_camera_only_when_asked():
    state = RemoteState(camera_up=True, containers={"fpp_a": "Up", "fpp_b": "Exited (1)"}, viewer_up=True)
    assert plan_actions(Command("stop", (), None, False), state) == [
        ("viewer_down",), ("fpp_down", "fpp_a"), ("fpp_down", "fpp_b")]
    assert plan_actions(Command("stop", (), None, True), state)[-1] == ("camera_down",)


def test_plan_viewer_toggle():
    up = RemoteState(True, {}, True)
    assert plan_actions(Command("viewer", (), True, False), up) == []
    assert plan_actions(Command("viewer", (), False, False), up) == [("viewer_down",)]


def test_build_status_shape():
    st = RemoteState(True, {"fpp_shaker_closed": "Up 2 minutes"}, False)
    out = build_status(st, camera_hz=29.9, pose_ages={"shaker_closed": 0.05, "cup_big_s100": None},
                       busy=False, error=None)
    assert out["camera_hz"] == 29.9 and out["camera_up"] is True
    assert out["objects"]["shaker_closed"] == {"container": "Up 2 minutes", "pose_age_s": 0.05}
    assert out["objects"]["cup_big_s100"] == {"container": None, "pose_age_s": None}
    assert out["viewer"] is False and out["busy"] is False and out["error"] is None
    assert build_status(None, 0.0, {}, True, "ssh failed")["error"] == "ssh failed"
```

- [ ] **Step 2: 실패 확인**

Run: `cd /home/user/rl_ws/sim2real && python3 -m pytest scripts/test_perception_launcher_core.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception_launcher_core'`

- [ ] **Step 3: 구현**

`scripts/perception_launcher_core.py`:

```python
#!/usr/bin/env python3
"""perception_launcher_node 의 순수 로직 — 명령 파싱·액션 계획·상태 집계.

ROS·ssh 없이 import 된다. test_perception_launcher_core.py 대상.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from object_registry import CONTAINER_PREFIX, container_name

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
```

- [ ] **Step 4: 통과 확인**

Run: `cd /home/user/rl_ws/sim2real && python3 -m pytest scripts/test_perception_launcher_core.py scripts/test_object_registry.py -q`
Expected: 15 passed

- [ ] **Step 5: 커밋**

```bash
cd /home/user/rl_ws/sim2real
git add scripts/perception_launcher_core.py scripts/test_perception_launcher_core.py
git commit -m "feat(perception): 런처 순수 로직 — 명령 파싱·액션 계획·상태 집계"
```

---

### Task 3: vision-3090 쪽 스크립트 (`scripts/vision/*.sh`) + 뷰어 레지스트리화

**Files:**
- Create: `scripts/vision/common.sh`, `camera_up.sh`, `camera_down.sh`, `fpp_up.sh`, `fpp_down.sh`, `viewer_up.sh`, `viewer_down.sh`, `status.sh`, `legacy_down.sh`
- Modify: `scripts/vision/cup_view_stream.py` (OBJECTS/AABB 상수 → `--objects` 인자 + 레지스트리)

**Interfaces:**
- Consumes: `render_fpp_yaml` 산출 yaml 이 `/opt/params/<name>.yaml` 경로로 마운트됨(런처가 `/home/usr/rl_ws/sim2real/log/fpp_params/<name>.yaml` 에 써 두고 그 경로를 `fpp_up.sh` 에 준다).
- Produces(런처가 호출하는 계약, 전부 `bash <script> ...`, exit 0 = 성공):
  - `camera_up.sh` / `camera_down.sh`
  - `fpp_up.sh <name> <yaml_host_path>` / `fpp_down.sh <container|all>`
  - `viewer_up.sh <name>...` / `viewer_down.sh`
  - `status.sh` → stdout 마지막 줄 JSON `{"camera_up":bool,"containers":{...},"viewer_up":bool}`
  - `legacy_down.sh` → 옛 relay/tx/컨테이너 정리

- [ ] **Step 1: 공통 셸 + 스크립트 작성**

`scripts/vision/common.sh`:

```bash
#!/bin/bash
# vision-3090 전용 공통 설정. 모든 vision/*.sh 가 source 한다.
# ★set -u 금지: /opt/ros/humble/setup.bash 가 미정의 변수를 참조해 즉사한다.
set -eo pipefail
export ROS_DOMAIN_ID=126
SIM2REAL=/home/usr/rl_ws/sim2real
PPP=/home/usr/rl_ws/perception_plus_plus
LOGDIR=/tmp/perception
mkdir -p "$LOGDIR"
source /opt/ros/humble/setup.bash
```

`scripts/vision/camera_up.sh`:

```bash
#!/bin/bash
# RealSense ROS 노드 기동(멱등). align_depth 필수 — FP++ 가 정렬 깊이를 구독한다.
source "$(dirname "$0")/common.sh"
if pgrep -f realsense2_camera_node >/dev/null; then echo "camera already up"; exit 0; fi
setsid bash -c 'source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=126;
  exec ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true' \
  </dev/null >"$LOGDIR/realsense.log" 2>&1 &
for _ in $(seq 1 30); do
  if timeout 2 ros2 topic echo --once /camera/camera/color/camera_info >/dev/null 2>&1; then
    echo "camera up"; exit 0; fi
done
echo "camera did not publish within 60s (see $LOGDIR/realsense.log)" >&2; exit 1
```

`scripts/vision/camera_down.sh`:

```bash
#!/bin/bash
source "$(dirname "$0")/common.sh"
pkill -f realsense2_camera_node || true
pkill -f "ros2 launch realsense2_camera" || true
echo "camera down"
```

`scripts/vision/fpp_up.sh`:

```bash
#!/bin/bash
# 물체 하나의 FP++ 컨테이너 기동. usage: fpp_up.sh <name> <yaml_host_path>
# 이미지는 baked 라 패치 파일은 파일 단위 바인드 마운트(09.02 규약). 이름 fpp_<name>.
source "$(dirname "$0")/common.sh"
NAME=${1:?name}; YAML=${2:?yaml path}
[ -f "$YAML" ] || { echo "yaml missing: $YAML" >&2; exit 1; }
docker rm -f "fpp_$NAME" >/dev/null 2>&1 || true
docker run -d --name "fpp_$NAME" --network host --ipc=host --gpus all -e ROS_DOMAIN_ID=126 \
  -v $PPP/perception_plus_plus_core/detection/yolo.py:/workspace/perception_plus_plus/perception_plus_plus_core/detection/yolo.py:ro \
  -v $PPP/perception_plus_plus_core/fp_adapter/foundationpose_plus_plus.py:/workspace/perception_plus_plus/perception_plus_plus_core/fp_adapter/foundationpose_plus_plus.py:ro \
  -v $PPP/ros_ws/src/perception_plus_plus_ros/perception_plus_plus_ros/node.py:/opt/perception_plus_plus/lib/python3.10/site-packages/perception_plus_plus_ros/node.py:ro \
  -v $PPP/assets/meshes:/workspace/perception_plus_plus/assets/meshes:ro \
  -v "$YAML":/opt/params/"$NAME".yaml:ro \
  perception-plus-plus:humble-cup bash -lc "
    source /opt/ros/humble/setup.bash
    source /opt/perception_plus_plus/setup.bash
    cd /workspace/perception_plus_plus
    exec ros2 launch perception_plus_plus_ros cup_tracking.launch.py parameters_file:=/opt/params/$NAME.yaml" \
  >/dev/null
echo "fpp_$NAME up"
```

`scripts/vision/fpp_down.sh`:

```bash
#!/bin/bash
# usage: fpp_down.sh <container|all>
source "$(dirname "$0")/common.sh"
TARGET=${1:?container name or all}
if [ "$TARGET" = all ]; then
  docker ps -a --filter name='^fpp_' --format '{{.Names}}' | xargs -r docker rm -f >/dev/null
else
  docker rm -f "$TARGET" >/dev/null 2>&1 || true
fi
echo "down $TARGET"
```

`scripts/vision/viewer_up.sh`:

```bash
#!/bin/bash
# 모니터 창(DISPLAY=:0) + MJPEG 8080. usage: viewer_up.sh <name>...
source "$(dirname "$0")/common.sh"
[ $# -ge 1 ] || { echo "need object names" >&2; exit 1; }
pkill -f cup_view_stream.py || true; sleep 0.5
DISPLAY=:0 setsid bash -c "source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=126;
  cd $SIM2REAL/scripts && exec python3 cup_view_stream.py --show --compressed --port 8080 --objects $*" \
  </dev/null >"$LOGDIR/viewer.log" 2>&1 &
sleep 2; pgrep -f cup_view_stream.py >/dev/null && echo "viewer up" || { cat "$LOGDIR/viewer.log" >&2; exit 1; }
```

`scripts/vision/viewer_down.sh`:

```bash
#!/bin/bash
source "$(dirname "$0")/common.sh"
pkill -f cup_view_stream.py || true
echo "viewer down"
```

`scripts/vision/status.sh`:

```bash
#!/bin/bash
# 마지막 줄이 JSON. 카메라 hz 는 로컬 런처가 DDS 로 직접 잰다(여기선 프로세스 유무만).
source "$(dirname "$0")/common.sh"
python3 - <<'EOF'
import json, subprocess
def up(pat):
    return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0
out = subprocess.run(["docker", "ps", "-a", "--filter", "name=^fpp_", "--format", "{{.Names}}\t{{.Status}}"],
                     capture_output=True, text=True).stdout
containers = dict(line.split("\t", 1) for line in out.splitlines() if "\t" in line)
print(json.dumps({"camera_up": up("realsense2_camera_node"), "containers": containers,
                  "viewer_up": up("cup_view_stream.py")}))
EOF
```

`scripts/vision/legacy_down.sh`:

```bash
#!/bin/bash
# 옛 체인 정리: vision 쪽 relay/UDP tx/옛 컨테이너 이름(fpp_cup·fpp_shaker) — 로컬 노드가 대체했다.
source "$(dirname "$0")/common.sh"
pkill -f cup_pose_relay.py || true
pkill -f pose_udp_tx.py || true
docker rm -f fpp_cup fpp_shaker >/dev/null 2>&1 || true
echo "legacy down"
```

- [ ] **Step 2: 뷰어를 레지스트리 기반으로 수정**

`scripts/vision/cup_view_stream.py` 에서 상수 `AABB`, `OBJECTS` 를 지우고, `View.__init__(self, objects, compressed)` 가 `(name, cam_topic, base_topic, color, aabb)` 목록을 받도록 바꾼다. `main()` 에 `--objects` 인자를 추가:

```python
from object_registry import input_topic, load_registry, output_topic  # 파일 상단 import 에 추가

_COLORS = ((0, 0, 255), (255, 200, 0), (0, 255, 0), (255, 0, 255))


def objects_from_registry(names: list[str]) -> list[tuple]:
    reg = load_registry()
    out = []
    for i, raw in enumerate(names):
        spec = reg.get(raw)
        out.append((spec.name, input_topic(spec.name), output_topic(spec.name),
                    _COLORS[i % len(_COLORS)], spec.aabb))
    return out
```

`View.__init__` 의 구독 루프는 `for name, cam_t, base_t, _, _ in self.objects:`, `_draw` 의 루프는 `for name, _, _, color, aabb in self.objects:` 로 바꾸고 `draw_box(..., aabb, color)` 를 쓴다. `main()`:

```python
    ap.add_argument("--objects", nargs="+", required=True, help="레지스트리 물체 이름")
    ...
    view = View(objects_from_registry(args.objects), compressed=args.compressed)
```

- [ ] **Step 3: 셸 문법·뷰어 import 확인 (로컬)**

Run: `cd /home/user/rl_ws/sim2real && bash -n scripts/vision/*.sh && python3 -c "import ast,sys; ast.parse(open('scripts/vision/cup_view_stream.py').read()); print('ok')"`
Expected: `ok`, 오류 없음

- [ ] **Step 4: 커밋**

```bash
cd /home/user/rl_ws/sim2real
chmod +x scripts/vision/*.sh
git add scripts/vision scripts/vision/cup_view_stream.py
git commit -m "feat(perception): vision-3090 쪽 기동 스크립트 repo 화 + 뷰어 레지스트리 기반"
```

---

### Task 4: 런처 노드 (`perception_launcher_node.py`) + CLI (`perception_ctl.py`)

**Files:**
- Create: `scripts/nodes/perception_launcher_node.py`
- Create: `scripts/ops/perception_ctl.py`

**Interfaces:**
- Consumes: Task 1 (`load_registry`, `render_fpp_yaml`, `output_topic`, `container_name`), Task 2 (`parse_command`, `parse_remote_status`, `plan_actions`, `build_status`), Task 3 스크립트 계약.
- Produces: `/perception/cmd` 구독, `/perception/status` 발행. `RemoteExec.run(script, *args) -> str`(stdout, 실패 시 `RuntimeError`), `RemoteExec.put(text, remote_path)`.

- [ ] **Step 1: 런처 노드 작성**

`scripts/nodes/perception_launcher_node.py`:

```python
#!/usr/bin/env python3
"""노드 1 — 로컬에서 vision-3090 의 인지 체인을 켜고 끈다.

  /perception/cmd   (std_msgs/String JSON)  ← perception_ctl.py
    {"op":"start","objects":["shaker_closed","cup_big_s100"],"viewer":true}
    {"op":"stop","camera":false} · {"op":"viewer","on":false}
  /perception/status (std_msgs/String JSON, 1 Hz)

원격 실행은 tailscale ssh(무비밀번호) 로 repo 의 scripts/vision/*.sh 를 부른다.
FP++ yaml 은 레지스트리에서 생성해 vision 의 log/fpp_params/<name>.yaml 에 써 넣는다.
카메라 hz 는 DDS 로 직접 잰다(두 PC 가 같은 LAN, domain 126).
실패는 status.error 로 드러낸다 — 조용한 재시도 없음.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from object_registry import container_name, load_registry, output_topic, render_fpp_yaml  # noqa: E402
from perception_launcher_core import (  # noqa: E402
    build_status, parse_command, parse_remote_status, plan_actions,
)

REMOTE_SIM2REAL = "/home/usr/rl_ws/sim2real"
REMOTE_PARAMS = f"{REMOTE_SIM2REAL}/log/fpp_params"
_SCRIPT_FOR = {"camera_up": "camera_up.sh", "camera_down": "camera_down.sh",
               "fpp_up": "fpp_up.sh", "fpp_down": "fpp_down.sh",
               "viewer_up": "viewer_up.sh", "viewer_down": "viewer_down.sh"}


class RemoteExec:
    def __init__(self, host: str, timeout_s: float = 120.0) -> None:
        self.host, self.timeout_s = host, timeout_s

    def _ssh(self, command: str, stdin: str | None = None) -> str:
        proc = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", self.host, command],
                              input=stdin, capture_output=True, text=True, timeout=self.timeout_s)
        if proc.returncode != 0:
            raise RuntimeError(f"ssh {self.host} '{command[:60]}…' rc={proc.returncode}: "
                               f"{(proc.stderr or proc.stdout).strip()[-300:]}")
        return proc.stdout

    def run(self, script: str, *args: str) -> str:
        quoted = " ".join(f"'{a}'" for a in args)
        return self._ssh(f"bash {REMOTE_SIM2REAL}/scripts/vision/{script} {quoted}")

    def put(self, text: str, remote_path: str) -> None:
        self._ssh(f"mkdir -p $(dirname '{remote_path}') && cat > '{remote_path}'", stdin=text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="vision-3090")
    ap.add_argument("--poll", type=float, default=5.0, help="원격 상태 폴링 주기(s)")
    args = ap.parse_args()
    registry = load_registry()
    remote = RemoteExec(args.host)

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo
    from std_msgs.msg import String

    class Launcher(Node):
        def __init__(self) -> None:
            super().__init__("perception_launcher")
            self._lock = threading.Lock()
            self._busy = False
            self._error: str | None = None
            self._state = None
            self._last_pose: dict[str, float] = {}
            self._cam_stamps: list[float] = []
            self._pub = self.create_publisher(String, "/perception/status", 10)
            self.create_subscription(String, "/perception/cmd", self._on_cmd, 10)
            self.create_subscription(CameraInfo, "/camera/camera/color/camera_info", self._on_cam, 5)
            for name in registry.names():
                self.create_subscription(PoseStamped, output_topic(name),
                                         lambda _m, n=name: self._last_pose.__setitem__(n, time.monotonic()), 10)
            self.create_timer(1.0, self._publish_status)
            self.create_timer(args.poll, self._poll_remote)
            self._poll_remote()

        def _on_cam(self, _msg) -> None:
            now = time.monotonic()
            self._cam_stamps = [t for t in self._cam_stamps if now - t < 2.0] + [now]

        def _poll_remote(self) -> None:
            if self._busy:
                return
            try:
                self._state = parse_remote_status(remote.run("status.sh"))
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as err:
                self._error = f"status: {err}"
                self.get_logger().error(self._error)

        def _publish_status(self) -> None:
            now = time.monotonic()
            ages = {n: (round(now - self._last_pose[n], 3) if n in self._last_pose else None)
                    for n in registry.names()}
            payload = build_status(self._state, len(self._cam_stamps) / 2.0, ages, self._busy, self._error)
            self._pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

        def _on_cmd(self, msg: String) -> None:
            try:
                cmd = parse_command(msg.data, registry)
            except ValueError as err:
                self._error = f"cmd: {err}"
                self.get_logger().error(self._error)
                return
            with self._lock:
                if self._busy:
                    self._error = "busy: previous command still running"
                    self.get_logger().warning(self._error)
                    return
                self._busy = True
            threading.Thread(target=self._execute, args=(cmd,), daemon=True).start()

        def _execute(self, cmd) -> None:
            try:
                self._error = None
                state = parse_remote_status(remote.run("status.sh"))
                actions = plan_actions(cmd, state)
                self.get_logger().info(f"{cmd.op}: {actions or '변경 없음'}")
                for action in actions:
                    self._do(action, cmd, state)
                self._state = parse_remote_status(remote.run("status.sh"))
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as err:
                self._error = f"{cmd.op}: {err}"
                self.get_logger().error(self._error)
            finally:
                self._busy = False

        def _do(self, action, cmd, state) -> None:
            kind = action[0]
            if kind == "fpp_up":
                name = action[1]
                path = f"{REMOTE_PARAMS}/{name}.yaml"
                remote.put(render_fpp_yaml(registry.get(name)), path)
                out = remote.run("fpp_up.sh", name, path)
            elif kind == "viewer_up":
                # viewer 단독 명령엔 물체 목록이 없다 — 떠 있는 컨테이너에서 이름을 되찾는다.
                names = list(cmd.objects) or [c[len("fpp_"):] for c, st in state.containers.items()
                                              if c.startswith("fpp_") and st.startswith("Up")]
                if not names:
                    raise RuntimeError("viewer_up: 물체가 없다 — start 로 먼저 컨테이너를 띄울 것")
                out = remote.run("viewer_up.sh", *names)
            elif kind == "fpp_down":
                out = remote.run("fpp_down.sh", action[1])
            else:
                out = remote.run(_SCRIPT_FOR[kind])
            self.get_logger().info(f"  {action} → {out.strip().splitlines()[-1] if out.strip() else 'ok'}")

    rclpy.init()
    node = Launcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: CLI 작성**

`scripts/ops/perception_ctl.py`:

```python
#!/usr/bin/env python3
"""노드 2 의 사용자 면 — 물체를 골라 인지 체인을 켜고 끈다.

  perception_ctl.py start shaker_closed cup_big_s100 [--viewer]
  perception_ctl.py stop [--camera]
  perception_ctl.py viewer on|off
  perception_ctl.py status
  perception_ctl.py list

이름은 레지스트리(config/objects.yaml)로 검증·alias 해석 후 /perception/cmd 에 발행한다.
perception_launcher_node.py 가 떠 있어야 한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from object_registry import load_registry  # noqa: E402


def build_payload(args, registry) -> dict | None:
    if args.op == "start":
        return {"op": "start", "objects": [registry.resolve(n) for n in args.objects],
                "viewer": bool(args.viewer)}
    if args.op == "stop":
        return {"op": "stop", "camera": bool(args.camera)}
    if args.op == "viewer":
        return {"op": "viewer", "on": args.on == "on"}
    return None


def print_status(payload: dict) -> None:
    print(f"camera: {'up' if payload['camera_up'] else 'down'} ({payload['camera_hz']} Hz)"
          f" · viewer: {payload['viewer']} · busy: {payload['busy']}")
    for name, info in payload["objects"].items():
        age = info["pose_age_s"]
        print(f"  {name:16s} container={info['container'] or '-':22s} "
              f"pose={'-' if age is None else f'{age:.2f}s ago'}")
    if payload.get("error"):
        print(f"  ERROR: {payload['error']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="op", required=True)
    s = sub.add_parser("start"); s.add_argument("objects", nargs="+"); s.add_argument("--viewer", action="store_true")
    st = sub.add_parser("stop"); st.add_argument("--camera", action="store_true", help="카메라까지 내린다")
    v = sub.add_parser("viewer"); v.add_argument("on", choices=("on", "off"))
    sub.add_parser("status"); sub.add_parser("list")
    args = ap.parse_args()
    registry = load_registry()
    if args.op == "list":
        for name in registry.names():
            print(f"{name:16s} {registry.get(name).real}")
        for alias, target in registry.aliases.items():
            print(f"{alias:16s} → {target}")
        return 0
    payload = build_payload(args, registry)

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    rclpy.init()
    node = Node("perception_ctl")
    if payload is not None:
        pub = node.create_publisher(String, "/perception/cmd", 10)
        deadline = time.monotonic() + 3.0
        while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if pub.get_subscription_count() == 0:
            print("perception_launcher_node 가 안 떠 있다 (/perception/cmd 구독자 0)", file=sys.stderr)
            return 1
        pub.publish(String(data=json.dumps(payload)))
        print(f"sent {payload}")
    box: list[str] = []
    node.create_subscription(String, "/perception/status", lambda m: box.append(m.data), 10)
    deadline = time.monotonic() + 3.0
    while not box and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if box:
        print_status(json.loads(box[-1]))
    else:
        print("/perception/status 가 안 온다", file=sys.stderr)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: 문법 확인**

Run: `cd /home/user/rl_ws/sim2real && python3 -m py_compile scripts/nodes/perception_launcher_node.py scripts/ops/perception_ctl.py && echo ok`
Expected: `ok`

- [ ] **Step 4: 커밋**

```bash
cd /home/user/rl_ws/sim2real
git add scripts/nodes/perception_launcher_node.py scripts/ops/perception_ctl.py
git commit -m "feat(perception): 런처 노드(ssh 실행·status) + perception_ctl CLI"
```

---

### Task 5: pose 노드 (`object_pose_node.py`)

**Files:**
- Create: `scripts/nodes/object_pose_node.py`
- Test: `scripts/test_object_pose_node.py`

**Interfaces:**
- Consumes: Task 1 `load_registry`, `extrinsics_for`, `input_topic`, `output_topic`; `cup_pose_relay.cad_pose_to_base_body(ext, pos, quat)`, `extrinsics_at_head`, `head_state_is_usable`.
- Produces: `PoseConverter`(순수): `__init__(registry, names: list[str])`, `convert(name, pos_cam, quat_cam, head=None) -> (pos, quat)`; 노드가 `/objects/<name>/pose` 발행.

- [ ] **Step 1: 실패하는 테스트 작성**

`scripts/test_object_pose_node.py`:

```python
"""object_pose_node 순수부 — 레지스트리 → Extrinsics → base_link 변환."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from object_pose_node import PoseConverter  # noqa: E402
from object_registry import DEFAULT_REGISTRY, load_registry  # noqa: E402

REG = load_registry(DEFAULT_REGISTRY)


def test_shaker_camera_pose_maps_to_measured_base_pose():
    """09.03 실측: 카메라 프레임 (0.0363,-0.1111,0.5877) → base (0.3627,-0.0032,0.3224)."""
    conv = PoseConverter(REG, ["shaker_closed"])
    pos, quat = conv.convert("shaker_closed", np.array([0.0363, -0.1111, 0.5877]),
                             np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(pos, [0.3627, -0.0032, 0.3224], atol=0.01)
    assert np.isclose(np.linalg.norm(quat), 1.0)


def test_cup_applies_yup_to_zup_but_same_camera():
    conv = PoseConverter(REG, ["shaker_closed", "cup_big_s100"])
    p_s, _ = conv.convert("shaker_closed", np.zeros(3), np.array([1.0, 0, 0, 0]))
    p_c, q_c = conv.convert("cup_big_s100", np.zeros(3), np.array([1.0, 0, 0, 0]))
    assert np.allclose(p_s, p_c)          # cad_to_body 는 위치 0 이라 위치 동일
    assert not np.allclose(q_c, conv.convert("shaker_closed", np.zeros(3), np.array([1.0, 0, 0, 0]))[1])


def test_unknown_name_rejected_and_names_resolved():
    conv = PoseConverter(REG, ["cup_big_s080"])
    assert conv.names == ["cup_big_s100"]
    try:
        PoseConverter(REG, ["teapot"])
    except ValueError as err:
        assert "unknown object" in str(err)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: 실패 확인**

Run: `cd /home/user/rl_ws/sim2real && python3 -m pytest scripts/test_object_pose_node.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'object_pose_node'`

- [ ] **Step 3: 구현**

`scripts/nodes/object_pose_node.py`:

```python
#!/usr/bin/env python3
"""노드 3 — 카메라 프레임 물체 pose → base_link `/objects/<name>/pose`.

vision-3090 의 FP++ 가 내는 `/perception_plus_plus/<name>/pose`(camera optical) 를
DDS 로 직접 구독하고, cup_pose_relay 의 변환(T_base_cam ∘ T_cam_cad ∘ T_cad_body)으로
base_link 로 바꿔 발행한다. camera 블록은 레지스트리가 가리키는 공유 extrinsics,
cad_to_body 는 물체 항목. 소비자: ROS 정책 노드(다음 스펙).

  python3 object_pose_node.py [--objects shaker_closed cup_big_s100] [--head-joint-topic /head/joint_states]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cup_pose_relay import (  # noqa: E402
    cad_pose_to_base_body, extrinsics_at_head, head_state_is_usable,
)
from object_registry import extrinsics_for, input_topic, load_registry, output_topic  # noqa: E402


class PoseConverter:
    """순수부: 물체별 Extrinsics 를 미리 조립해 두고 변환만 한다."""

    def __init__(self, registry, names: list[str]) -> None:
        self.names = []
        for raw in names:
            canon = registry.resolve(raw)
            if canon not in self.names:
                self.names.append(canon)
        self._ext = {n: extrinsics_for(registry.get(n), registry.camera_extrinsics) for n in self.names}
        self.base_frame = next(iter(self._ext.values())).base_frame if self._ext else "base_link"

    def convert(self, name: str, pos_cam: np.ndarray, quat_cam: np.ndarray,
                head: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        ext = self._ext[name]
        if head is not None:
            ext = extrinsics_at_head(ext, *head)
        return cad_pose_to_base_body(ext, np.asarray(pos_cam, float), np.asarray(quat_cam, float))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", nargs="*", default=None, help="기본: 레지스트리 전체")
    ap.add_argument("--head-joint-topic", default=None,
                    help="주면 T_base_cam 을 목 각도로 매번 계산(정적 camera 블록은 pan0/tilt-20 전용)")
    ap.add_argument("--head-max-age", type=float, default=1.0)
    args = ap.parse_args()
    registry = load_registry()
    conv = PoseConverter(registry, args.objects or registry.names())

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node

    class ObjectPoseNode(Node):
        def __init__(self) -> None:
            super().__init__("object_pose_node")
            self._pubs = {n: self.create_publisher(PoseStamped, output_topic(n), 10) for n in conv.names}
            for n in conv.names:
                self.create_subscription(PoseStamped, input_topic(n), lambda m, n=n: self._on_pose(n, m), 10)
            self._head: tuple[float, float] | None = None
            self._head_stamp: float | None = None
            if args.head_joint_topic:
                from sensor_msgs.msg import JointState
                self.create_subscription(JointState, args.head_joint_topic, self._on_head, 10)
            self._count = {n: 0 for n in conv.names}
            self.create_timer(10.0, self._report)
            self.get_logger().info(f"objects {conv.names} → {[output_topic(n) for n in conv.names]}")

        def _on_head(self, msg) -> None:
            names = list(msg.name)
            try:
                pan = float(msg.position[names.index("head_j_pan")])
                tilt = float(msg.position[names.index("head_j_tilt")])
            except (ValueError, IndexError):
                return
            self._head = (np.degrees(pan), np.degrees(tilt))
            self._head_stamp = self.get_clock().now().nanoseconds * 1e-9

        def _on_pose(self, name: str, msg: PoseStamped) -> None:
            head = None
            if args.head_joint_topic:
                now = self.get_clock().now().nanoseconds * 1e-9
                if not head_state_is_usable(self._head_stamp, now, args.head_max_age):
                    self.get_logger().warning("목 각도가 없거나 오래됐다 — 발행 보류", throttle_duration_sec=5.0)
                    return
                head = self._head
            p, q = msg.pose.position, msg.pose.orientation
            pos, quat = conv.convert(name, np.array([p.x, p.y, p.z]), np.array([q.w, q.x, q.y, q.z]), head)
            out = PoseStamped()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = conv.base_frame
            out.pose.position.x, out.pose.position.y, out.pose.position.z = map(float, pos)
            out.pose.orientation.w, out.pose.orientation.x, out.pose.orientation.y, out.pose.orientation.z = map(float, quat)
            self._pubs[name].publish(out)
            self._count[name] += 1

        def _report(self) -> None:
            self.get_logger().info(" · ".join(f"{n} {c}" for n, c in self._count.items()))

    rclpy.init()
    node = ObjectPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 통과 확인**

Run: `cd /home/user/rl_ws/sim2real && python3 -m pytest scripts/test_object_pose_node.py scripts/test_object_registry.py scripts/test_perception_launcher_core.py scripts/test_cup_pose_relay.py -q`
Expected: 모두 passed

- [ ] **Step 5: 커밋**

```bash
cd /home/user/rl_ws/sim2real
git add scripts/nodes/object_pose_node.py scripts/test_object_pose_node.py
git commit -m "feat(perception): object_pose_node — 카메라 프레임 → /objects/<name>/pose"
```

---

### Task 6: vision-3090 배포 + 통합 검증 (카메라 필요)

**Files:** 없음(운영). 결과는 이 계획서 하단 "검증 기록" 에 적는다.

- [ ] **Step 1: 로컬 커밋을 vision-3090 으로 동기화**

vision 의 작업본은 `config/global_camera_extrinsics.yaml` 을 미커밋 수정으로 들고 있고(값은 로컬 커밋 daa6735 와 동일), 옛 HEAD(ccf6a44) 다.

```bash
ssh vision-3090 'cd ~/rl_ws/sim2real && git stash push -m pre-perception config/global_camera_extrinsics.yaml && git pull --ff-only && grep -c 0.0674838252 config/global_camera_extrinsics.yaml && git log --oneline -1'
```
Expected: 마지막 두 줄 `1`(새 extrinsics 값 존재) 과 최신 커밋 해시. (`git pull` 의 원격이 로컬 pc5090 이 아니면 먼저 로컬에서 `git push` 후 실행)

- [ ] **Step 2: 옛 체인 정리 + 런처·pose 노드 기동 (로컬, 백그라운드 2개)**

```bash
ssh vision-3090 'bash ~/rl_ws/sim2real/scripts/vision/legacy_down.sh'
cd /home/user/rl_ws/sim2real/scripts
source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=126
setsid python3 perception_launcher_node.py </dev/null >/tmp/perception_launcher.log 2>&1 &
setsid python3 object_pose_node.py </dev/null >/tmp/object_pose_node.log 2>&1 &
```

- [ ] **Step 3: start → status → pose 확인**

```bash
python3 perception_ctl.py start shaker_closed --viewer
# 60 초 뒤
python3 perception_ctl.py status
timeout 5 ros2 topic echo --once /objects/shaker_closed/pose
```
Expected: status 에 `fpp_shaker_closed container=Up …`, `pose=0.0xs ago`, vision-3090 모니터에 "head view" 창. pose z ≈ 0.322(09.03 기준선 ±0.01), frame_id `base_link`.

- [ ] **Step 4: 2물체 + stop**

```bash
python3 perception_ctl.py start shaker_closed cup_big_s100
# 60 초 뒤 (컨테이너 둘, 빨간 컵이 화면에 있어야 cup pose 가 온다)
python3 perception_ctl.py status
python3 perception_ctl.py stop
ssh vision-3090 'docker ps --filter name=^fpp_ --format "{{.Names}}"'
```
Expected: stop 후 컨테이너 목록 비어 있음, 카메라는 유지(`status` 에 camera up).

- [ ] **Step 5: 검증 기록 + 커밋**

이 파일 하단에 "## 검증 기록 (날짜)" 를 추가해 위 결과(z 값·창·컨테이너 수)를 적고 커밋:

```bash
cd /home/user/rl_ws/sim2real
git add docs/superpowers/plans/2026-09-03-fpp-perception-nodes.md
git commit -m "docs: FP++ 인지 노드 통합 검증 기록"
```

## 검증 기록 (2026-09-03)

- 배포: vision-3090 ← git bundle(feat/grasp-v1-live-policy, 7de58e9). 두 PC origin 이 달라 GitHub 미경유.
  vision 의 미커밋 extrinsics 는 로컬 커밋과 동일해 stash 후 pull, 값(0.0674838252) 보존 확인.
- `perception_ctl.py start shaker_closed --viewer` → 약 50 s 뒤 `fpp_shaker_closed Up`, `/objects/shaker_closed/pose`
  frame_id base_link, z **0.326**(09.03 기준선 0.322, shaker 재배치 포함). vision 모니터에 "head view" 창(xwininfo 확인).
- `start shaker_closed cup_big_s100` → 컨테이너 2, 두 pose 모두 수신(0.02/0.07 s). `stop` → 컨테이너 0, 카메라 유지.
- 발견·수정 2건: ① CLI 가 `--viewer` 없으면 `viewer:false` 를 보내 뷰어를 내렸다 → 키 생략(7de58e9).
  ② 뷰어가 SIGTERM 에 안 죽었다(rclpy 가 SIGTERM 을 가로채고 waitKey 루프는 rclpy.ok 를 안 봄) →
  루프 조건 + viewer_down.sh SIGKILL 승급. 재검증: `viewer off` 가 "viewer down"(정상 종료), 플래그 없는 start "변경 없음".
- 테스트: `python3 -m pytest scripts/test_object_registry.py scripts/test_perception_launcher_core.py
  scripts/test_object_pose_node.py scripts/test_perception_ctl.py scripts/test_cup_pose_relay.py` → 34 passed.
- 운영 메모: pkill 자기매칭 — vision 스크립트는 반드시 경로로만 호출(ssh 명령줄에 패턴 문자열 금지).
  status 의 pose_age 는 컨테이너가 내려간 뒤에도 "마지막 수신 후 경과"로 계속 커진다(의도).
