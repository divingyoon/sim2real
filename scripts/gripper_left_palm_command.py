#!/usr/bin/env python3
"""좌팔 v2 **절대 palm 액션 산술** — 배포용 순수 numpy (ROS·torch 무의존).

**왜 별도 모듈인가.** 좌팔 `grasp_sensor_v2` 는 다른 트랙과 액션 규약이 다르다.
`grasp_v1`/`grasp_sensor` 는 "델타 + 기준점(anchor)" 이지만 좌 v2 는 **절대 palm 6D**
다 — `a=0` 은 "기준점 유지"가 아니라 `PALM_BOX` 의 **중심**이다. 그래서
`action_anchor` 개념 자체가 적용되지 않는다(`config/robots/gripper_left.yaml` 주석).

학습 env 의 `GripperLeftPoseFabricAction.process_actions` 를 그대로 옮긴 것이고,
배포 노드가 이걸 써서 액션을 fabric attractor 의 palm 목표로 바꾼다:

    action(7) ─┬─ [0:3] 위치 → 박스 절대 좌표 + 변화율 상한
               ├─ [3:6] 회전 → euler_zyx 절대 + 변화율 상한
               └─ [6]   그리퍼 (여기서 다루지 않음)

**v2E29 런에서 꺼져 있는 것**(dump 확인, 2026-09-02): `appr_ey_max` · `fine_latch_cup_z`
· `fine_cmd_rate_limit` 이 전부 null 이다. 그래서 FINE 래치와 접근각 상한은 옮기지
않았다 — 켜진 런을 배포할 때 여기에 추가할 것.

★리미터는 **리셋 후 첫 지령에는 걸지 않는다**(`primed`). 홈에서 첫 목표까지의 거리는
 "변화"가 아니라 초기화이고, 여기에 상한을 걸면 리셋마다 팔이 몇 스텝 끌려간다
 (학습 코드 fab_test29 주석의 그 버그).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ── 학습 preset 상수 (grasp_left_preset.py) ────────────────────────────────
#   좌 v2 는 이 박스를 v1 에서 그대로 상속한다(v2_preset 에 재정의 없음).
PALM_BOX_X = (0.22, 0.60)
PALM_BOX_Y = (0.10, 0.43)
PALM_BOX_Z = (0.16, 0.60)
PALM_EULER_ZYX_CENTER = (0.317093862, -1.4835298641951802, 3.094591725)  # (ez, ey, ex)
PALM_MAX_POSE_ANGLE = math.radians(20.0)        # v1 기본
PALM_MAX_POSE_ANGLE_WIDE = math.radians(60.0)   # v2 `v2_rot_wide` — E29 가 쓰는 값
PALM_CMD_RATE_LIMIT = 0.02   # m/step   (1.0 m/s @60Hz)
PALM_ROT_RATE_LIMIT = 0.05   # rad/step (2.5 rad/s @60Hz)


@dataclass(frozen=True)
class PalmCommandCfg:
    """액션 산술 파라미터 — 런 dump 에서 읽어 채우는 것이 원칙이다."""

    box_lo: tuple = (PALM_BOX_X[0], PALM_BOX_Y[0], PALM_BOX_Z[0])
    box_hi: tuple = (PALM_BOX_X[1], PALM_BOX_Y[1], PALM_BOX_Z[1])
    euler_center: tuple = PALM_EULER_ZYX_CENTER
    max_pose_angle: float = PALM_MAX_POSE_ANGLE_WIDE
    pos_rate_limit: float | None = PALM_CMD_RATE_LIMIT
    rot_rate_limit: float | None = PALM_ROT_RATE_LIMIT


class PalmCommand:
    """액션 → palm 목표(pos3 + euler_zyx3). 리셋 사이에 상태(직전 지령)를 들고 있다."""

    def __init__(self, cfg: PalmCommandCfg | None = None) -> None:
        self.cfg = cfg or PalmCommandCfg()
        lo = np.asarray(self.cfg.box_lo, dtype=np.float64)
        hi = np.asarray(self.cfg.box_hi, dtype=np.float64)
        if np.any(hi <= lo):
            raise ValueError(f"PALM_BOX 가 뒤집혔다: lo={lo} hi={hi}")
        self._lo, self._hi = lo, hi
        self._center = 0.5 * (lo + hi)
        self._half = 0.5 * (hi - lo)
        self._euler_center = np.asarray(self.cfg.euler_center, dtype=np.float64)
        self._euler_half = np.full(3, float(self.cfg.max_pose_angle))
        self.reset()

    # ── 상태 ────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """에피소드 시작 — 목표를 박스 중심/회전 중심으로 두고 리미터를 푼다."""
        self._prev_pos = self._center.copy()
        self._prev_euler = self._euler_center.copy()
        self._primed = False

    @property
    def palm_pose(self) -> np.ndarray:
        """직전에 낸 목표 (pos3 + euler_zyx3)."""
        return np.concatenate([self._prev_pos, self._prev_euler])

    # ── 산술 ────────────────────────────────────────────────────────────
    def step(self, action) -> np.ndarray:
        """액션(≥6D)을 palm 목표 6D 로. 7번째(그리퍼)는 호출자가 따로 쓴다."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size < 6:
            raise ValueError(f"액션은 6D 이상이어야 한다 — 받은 {a.size}")

        pos = self._center + np.clip(a[0:3], -1.0, 1.0) * self._half
        pos = np.minimum(np.maximum(pos, self._lo), self._hi)
        euler = self._euler_center + np.clip(a[3:6], -1.0, 1.0) * self._euler_half

        if self._primed:
            pos = _rate_limit(self._prev_pos, pos, self.cfg.pos_rate_limit)
            euler = _rate_limit(self._prev_euler, euler, self.cfg.rot_rate_limit)

        self._prev_pos, self._prev_euler = pos, euler
        self._primed = True          # ★첫 지령 **뒤에** 켠다
        return np.concatenate([pos, euler])


def _rate_limit(prev: np.ndarray, target: np.ndarray, limit: float | None) -> np.ndarray:
    """한 스텝 이동량을 `limit` 로 묶는다 — 방향은 유지, 도달 범위는 그대로."""
    if limit is None:
        return target
    step = target - prev
    dist = float(np.linalg.norm(step))
    if dist <= limit or dist < 1e-12:
        return target
    return prev + step * (limit / dist)


def cfg_from_run(env_yaml_path) -> PalmCommandCfg:
    """런 dump 에서 액션 산술 파라미터를 읽는다 — 상수를 손으로 옮기지 않는다.

    dump 는 `!!python/tuple` 태그를 담고 있어 `yaml.safe_load` 가 못 읽으므로 필요한
    스칼라만 정규식으로 뽑는다(임의 객체 역직렬화 회피).
    """
    import re
    from pathlib import Path

    text = Path(env_yaml_path).read_text()

    def _scalar(key: str):
        m = re.search(rf"^\s*{key}:\s*(-?[0-9.eE+]+|null)\s*$", text, re.M)
        return None if (m is None or m.group(1) == "null") else float(m.group(1))

    angle = _scalar("palm_max_pose_angle")
    wide = re.search(r"^v2_rot_wide:\s*true\s*$", text, re.M) is not None
    if angle is None:
        angle = PALM_MAX_POSE_ANGLE_WIDE if wide else PALM_MAX_POSE_ANGLE
    for key in ("appr_ey_max", "fine_latch_cup_z", "fine_cmd_rate_limit"):
        if _scalar(key) is not None:
            raise SystemExit(
                f"[palm_command] 이 런은 `{key}` 가 켜져 있다 — 이 모듈은 그 분기를 "
                "옮기지 않았다. 학습 코드(grasp_left_fabric_action)를 보고 추가하라")
    return PalmCommandCfg(max_pose_angle=float(angle))
