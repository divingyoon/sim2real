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


def _validate_aliases(aliases: dict | None, objects: dict) -> dict[str, str]:
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
