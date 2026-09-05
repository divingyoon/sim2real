#!/usr/bin/env python3
"""우팔 `grasp_s2r` 팔 액션(6D) → palm 목표 6D 로 바꾸는 순수 로직 (numpy).

`grasp_s2r_env._pre_physics_step` 의 팔 구간을 1:1 이식한다. Isaac/torch/ROS 무의존.

★★좌팔(`gripper_left_palm_command`)과 **규약이 다르다.** 좌팔은 절대 박스 매핑
  (`a=0` → 박스 중심)이지만, 우팔은 **앵커 + 델타**다(`a=0` → 앵커). 두 모듈을 섞으면
  액션 원점이 통째로 어긋나므로 따로 둔다.

한 tick 의 순서 (env 와 동일):

    delta   = 0.5·(clip(a,-1,1)+1)·(hi−lo) + lo        # 성분별, hi/lo = ±delta 박스
    target  = anchor + delta                            # 6D = pos3 + euler_zyx3
    target  = clip(target, box_lo, box_hi)              # 박스 포화(축별 기록)
    target  = rate_limit(prev, target)                  # ★벡터 노름 기준 스케일링
                                                        #   위치·회전 따로, 첫 지령 제외

★앵커(`palm_anchor_mode`)
  - `home`  : 프로필 홈 palm 자세 그대로
  - `spawn` : **에피소드 시작 시 물체 위치 스냅샷** + `palm_anchor_offset_xyz`
              (회전 성분은 홈 그대로 — 위치만 재중심한다)
  ⚠`spawn` 은 반드시 **스냅샷**이어야 한다. 실시간 물체를 쓰면 컵이 밀릴 때 액션
    원점이 따라가는 되먹임이 되어, 정책이 스크립트처럼 동작한다(env 주석의 경고).

★변화율 리미터는 **성분별 클램프가 아니라 벡터 노름 스케일링**이다. 성분별로 자르면
  방향이 바뀐다 — env 는 `scale = min(1, lim/‖Δ‖)` 로 방향을 보존한다.

★첫 지령은 리미터를 걸지 않는다(`primed`). 리셋 직후의 첫 값은 "변화"가 아니라
  초기화라, 걸면 이전 에피소드 지령에서 끌려온다.

계약값은 런 dump 와 프로필에서 온다 — `cfg_from_run()` 을 쓰고 숫자를 손으로 옮기지
말 것. 09.03 좌팔에서 상수를 손으로 옮기다 네 곳이 어긋났다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ── 프로필 상수 (`grasp_s2r/robot_profiles.py::tesollo_right`) ───────────────
#: palm 목표 박스(env-local 절대, m). `palm_box_verified=True` — P-2 도달성 통과.
PALM_BOX_MIN = (0.20, -0.55, 0.20)
PALM_BOX_MAX = (0.55, 0.22, 0.70)
#: 회전 박스 = 중심 ± 반폭 (euler_zyx: ez, ey, ex)
PALM_ROT_CENTER_DEG = (90.0, 0.0, 90.0)
PALM_ROT_HALF_DEG = 45.0
#: 홈 palm 자세 — `reset_home_palm_pose` (0.28, −0.38, 0.42 / ez90·ey0·ex90)
HOME_PALM = (0.28, -0.38, 0.42,
             math.radians(90.0), math.radians(0.0), math.radians(90.0))


@dataclass(frozen=True)
class PalmCmdCfg:
    """런에서 읽는 값 — 기본값은 g1(`g1_rot20_fresh`) 실측이다."""

    delta_xyz: tuple = (0.1, 0.1, 0.1)
    delta_rot_deg: float = 20.0
    anchor_mode: str = "spawn"
    anchor_offset_xyz: tuple = (-0.066, -0.022, 0.085)
    rate_limit_m: float = 0.02
    rate_limit_rot_deg: float = 2.9
    palm_box_min: tuple = PALM_BOX_MIN
    palm_box_max: tuple = PALM_BOX_MAX
    rot_center_deg: tuple = PALM_ROT_CENTER_DEG
    rot_half_deg: float = PALM_ROT_HALF_DEG
    home_palm: tuple = HOME_PALM


@dataclass
class PalmCmdState:
    """tick 간 유지되는 상태. 에피소드마다 `reset()` 해야 한다."""

    prev_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    prev_rot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    primed: bool = False
    box_sat: np.ndarray = field(default_factory=lambda: np.zeros(3))
    step_raw: float = 0.0


def _rate_limit(prev: np.ndarray, target: np.ndarray, limit: float) -> np.ndarray:
    """벡터 노름 기준 스케일링 — 방향을 보존한다(성분별 클램프가 아니다)."""
    if limit <= 0.0:
        return target
    step = target - prev
    norm = float(np.linalg.norm(step))
    scale = min(1.0, limit / max(norm, 1e-9))
    return prev + step * scale


class PalmCommand:
    """우팔 palm 지령기. 한 로봇(단일 env) 기준."""

    def __init__(self, cfg: PalmCmdCfg | None = None) -> None:
        self.cfg = cfg or PalmCmdCfg()
        c = self.cfg
        d = np.asarray(c.delta_xyz, dtype=float)
        r = math.radians(float(c.delta_rot_deg))
        self._delta_lo = np.concatenate([-d, np.full(3, -r)])
        self._delta_hi = np.concatenate([d, np.full(3, r)])

        rc = np.radians(np.asarray(c.rot_center_deg, dtype=float))
        rh = math.radians(float(c.rot_half_deg))
        palm_lo = np.concatenate([np.asarray(c.palm_box_min, dtype=float), rc - rh])
        palm_hi = np.concatenate([np.asarray(c.palm_box_max, dtype=float), rc + rh])
        home = np.asarray(c.home_palm, dtype=float)
        # ★앵커(=홈)가 항상 박스 안이어야 `a=0` 의 의미가 유지된다.
        self._box_lo = np.minimum(palm_lo, home)
        self._box_hi = np.maximum(palm_hi, home)
        self._home = home

        if c.anchor_mode not in ("home", "spawn"):
            raise SystemExit(
                f"[palm] palm_anchor_mode={c.anchor_mode!r} 는 'home' 또는 'spawn' 이어야 한다")
        self._anchor_off = np.asarray(c.anchor_offset_xyz, dtype=float)
        self._spawn: np.ndarray | None = None
        self.state = PalmCmdState()
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, object_spawn_pos=None) -> None:
        """에피소드 시작. `spawn` 앵커면 물체 위치를 **여기서 한 번** 스냅샷한다."""
        self.state = PalmCmdState()
        if object_spawn_pos is not None:
            self._spawn = np.asarray(object_spawn_pos, dtype=float).reshape(3)
        anchor = self.anchor()
        self.state.prev_pos = anchor[:3].copy()
        self.state.prev_rot = anchor[3:].copy()

    def anchor(self) -> np.ndarray:
        """액션 원점 6D. 회전은 홈 그대로, 위치만 재중심한다."""
        if self.cfg.anchor_mode == "home" or self._spawn is None:
            return self._home.copy()
        pos = self._spawn + self._anchor_off
        return np.concatenate([pos, self._home[3:]])

    # ------------------------------------------------------------------
    def step(self, action6) -> np.ndarray:
        """액션 6D → palm 목표 6D (pos3 + euler_zyx3)."""
        a = np.clip(np.asarray(action6, dtype=float).reshape(6), -1.0, 1.0)
        delta = 0.5 * (a + 1.0) * (self._delta_hi - self._delta_lo) + self._delta_lo
        raw = self.anchor() + delta
        target = np.clip(raw, self._box_lo, self._box_hi)
        # 축별 박스 포화 — 값이 바뀌었으면 그 축의 도달영역이 부족한 것이다(진단).
        self.state.box_sat = (target[:3] != raw[:3]).astype(float)

        pos, rot = target[:3], target[3:]
        # 클램프 **전** 원 이동량 — 상한이 물리는 비율의 유일한 근거다.
        self.state.step_raw = (float(np.linalg.norm(pos - self.state.prev_pos))
                               if self.state.primed else 0.0)
        if self.state.primed:
            pos = _rate_limit(self.state.prev_pos, pos, self.cfg.rate_limit_m)
            rot = _rate_limit(self.state.prev_rot, rot,
                              math.radians(self.cfg.rate_limit_rot_deg))
        self.state.prev_pos, self.state.prev_rot = pos.copy(), rot.copy()
        self.state.primed = True
        return np.concatenate([pos, rot])


# ---------------------------------------------------------------------------
def cfg_from_run(env_yaml_path) -> PalmCmdCfg:
    """런 dump 에서 팔 액션 파라미터를 읽는다 — 숫자를 손으로 옮기지 않는다.

    dump 는 `!!python/tuple` 태그를 담고 있어 `yaml.safe_load` 가 못 읽으므로 필요한
    스칼라·튜플만 정규식으로 뽑는다(임의 객체 역직렬화 회피).
    """
    import re
    from pathlib import Path

    text = Path(env_yaml_path).read_text()

    def scalar(key, default=None):
        m = re.search(rf"^\s*{key}:\s*(-?[0-9.eE+]+|null|true|false|[a-z_]+)\s*$",
                      text, re.M)
        if m is None or m.group(1) == "null":
            return default
        v = m.group(1)
        if v in ("true", "false"):
            return v == "true"
        try:
            return float(v)
        except ValueError:
            return v

    def triple(key, default):
        # ★`strip("- ")` 은 **음수 부호까지 지운다**. 값을 정규식으로 직접 잡는다.
        m = re.search(rf"^\s*{key}:.*?\n((?:\s*- -?[\d.eE+-]+\n){{3}})", text, re.M)
        if m is None:
            return default
        vals = re.findall(r"-?[\d.eE+-]+", m.group(1).replace("- ", " "))
        if len(vals) != 3:
            raise SystemExit(f"[palm] {key} 를 3값으로 못 읽었다: {m.group(1)!r}")
        return tuple(float(v) for v in vals)

    mode = scalar("palm_anchor_mode", "spawn")
    if mode not in ("home", "spawn"):
        raise SystemExit(f"[palm] 런의 palm_anchor_mode={mode!r} 를 해석할 수 없다")
    return PalmCmdCfg(
        delta_xyz=triple("palm_delta_xyz", (0.1, 0.1, 0.1)),
        delta_rot_deg=float(scalar("palm_delta_rot_deg", 20.0)),
        anchor_mode=str(mode),
        anchor_offset_xyz=triple("palm_anchor_offset_xyz", (-0.066, -0.022, 0.085)),
        rate_limit_m=float(scalar("palm_cmd_rate_limit_m", 0.02)),
        rate_limit_rot_deg=float(scalar("palm_cmd_rate_limit_rot_deg", 2.9)),
    )
