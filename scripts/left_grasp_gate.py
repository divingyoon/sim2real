#!/usr/bin/env python3
"""좌팔 v2 **그리퍼 게이트** — 배포용 순수 numpy (ROS·torch·Isaac 무의존).

정책이 그리퍼를 닫을 수 있는지는 env 의 `grasp_ok` 술어가 정한다. 관측 36번째 칸
(`gripper_gate`)이 바로 이 값이고, 액션 항은 게이트가 열리기 전까지 그리퍼를 강제
개방한다. **배포에서 이 술어를 다시 만들지 않으면 관측 한 칸이 늘 0 이 되어 정책이
조용히 다른 상태를 본다.**

학습 코드 `grasp_left_rewards.grasp_ok` / `jaw_lateral` / `_jaw_frame` 을 그대로 옮겼다.
sim 은 바디 자세를 물리엔진에서 읽지만 실기는 **FK + `/cup_pose`** 로 같은 값을 만든다.

기하 (전부 world 프레임):

    패드 중앙   = 손가락 원점 + (그리퍼 base z축) · pad_offset
                  ★손가락 강체 원점은 base z=+15 mm 인데 성공 파지의 컵 축은 +46.9 mm 다.
                    원점 그대로 쓰면 판정이 32 mm 어긋난다.
    u           = 두 패드를 잇는 단위벡터 (턱 축)
    mid         = 두 패드의 중점
    cup_pt      = 컵 축 위 최근접점 — **파지 대역으로 clamp**
                  ★clamp 가 없으면 컵 축이 무한 직선이라 컵 위 허공에서도 성립한다.
    along       = |(cup_pt − mid) · u|          턱 축 방향 거리
    lateral     = |(cup_pt − mid) 의 u 수직성분|  턱 직선에서 컵 축까지
    in_band     = clamp **전** 축좌표가 파지 대역 안인가
                  ★clamp 된 값을 쓰면 밖에 있어도 경계로 접혀 들어와 항상 참이 된다.

    게이트 = (lateral < lat_ok) & (along < along_ok) & in_band   … 래치(단조)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── 학습 preset 상수 ───────────────────────────────────────────────────────
GRASP_DEPTH_IN_BASE_Z = 0.0469   # 성공 파지 시 컵 축의 base z (실측 중앙값)
JAW_FINGER_BODY_Z = 0.015        # 손가락 강체 원점의 base z
JAW_PAD_OFFSET = GRASP_DEPTH_IN_BASE_Z - JAW_FINGER_BODY_Z      # 0.0319 m

# ★★파지 대역은 **v2_preset 이 기본 preset 을 덮어쓴다.** 기본(v1)은 판 위 10~85 mm,
#   v2E29 는 판 위 80~150 mm 다(체크포인트 이름의 "band80" 이 그 뜻이다).
#   09.03 실기: v1 값을 쓰는 바람에 정책이 판 위 115 mm 를 조준하는데 게이트는
#   10~85 mm 를 요구해 **한 번도 열리지 않았다** — 그리퍼가 영원히 벌어진 채였다.
#   v2_env_cfg.py 주석이 이 실패 모드를 이미 경고하고 있었다:
#     "보상만 바꾸고 그리퍼 게이트를 두면 보상은 받는데 그리퍼가 안 열린다".
#   ⚠ 다른 체크포인트를 배포하면 그 런의 트랙(v1/v2)을 확인해 이 값을 맞출 것.
CUP_BOTTOM_TO_ORIGIN = 0.09209   # 컵 바닥 → 원점 (shaker_closed_rl)

#: 트랙별 파지 높이 대역(판 위 m). ★dump 에는 `grasp_band` 가 직렬화되지 않고 액션
#: 클래스 경로도 두 트랙이 같아서, **`agent.yaml` 의 태스크 이름만이 신뢰할 수 있는
#: 신호**다. 모르는 트랙은 추측하지 말고 죽인다.
GRASP_HEIGHT_BAND_BY_TASK = {
    "open-grip_l_grasp_sensor_v2": (0.080, 0.150),   # v2_preset 이 덮어쓴 값
    "open-grip_l_grasp_sensor_fab": (0.010, 0.085),  # cfg 기본 None → v1 대역
    "open-grip_l_grasp_sensor": (0.010, 0.085),
}
GRASP_HEIGHT_BAND = GRASP_HEIGHT_BAND_BY_TASK["open-grip_l_grasp_sensor_v2"]
CUP_GRASP_BAND_AXIS = (GRASP_HEIGHT_BAND[0] - CUP_BOTTOM_TO_ORIGIN,
                       GRASP_HEIGHT_BAND[1] - CUP_BOTTOM_TO_ORIGIN)


def band_axis_from_run(agent_yaml_path) -> tuple[float, float]:
    """런의 `agent.yaml` 태스크 이름으로 파지 대역(컵 원점 기준 축좌표)을 고른다."""
    import re
    from pathlib import Path as _P

    text = _P(agent_yaml_path).read_text()
    m = re.search(r"^\s*name:\s*(open-grip_l_[a-z0-9_]+)\s*$", text, re.M)
    if m is None:
        raise SystemExit(f"[gate] {agent_yaml_path} 에서 태스크 이름을 못 읽었다")
    task = m.group(1)
    if task not in GRASP_HEIGHT_BAND_BY_TASK:
        raise SystemExit(
            f"[gate] 모르는 트랙 `{task}` — 파지 대역을 추측할 수 없다. "
            "그 트랙의 preset(GRASP_HEIGHT_BAND)을 확인해 등록하라")
    lo, hi = GRASP_HEIGHT_BAND_BY_TASK[task]
    return (lo - CUP_BOTTOM_TO_ORIGIN, hi - CUP_BOTTOM_TO_ORIGIN)
LATERAL_OK = 0.03                # dump: lateral_ok
ALONG_OK = 0.03                  # dump: along_ok


@dataclass(frozen=True)
class GateCfg:
    pad_offset: float = JAW_PAD_OFFSET
    lateral_ok: float = LATERAL_OK
    along_ok: float = ALONG_OK
    band_axis: tuple = CUP_GRASP_BAND_AXIS
    #: 래치 해제 — 컵이 턱에서 이만큼 벗어나면 게이트를 닫는다. None 이면 해제 없음.
    release_lateral: float | None = None


@dataclass(frozen=True)
class JawFrame:
    """턱 기준 프레임 — 진단에 그대로 쓸 수 있게 중간값을 전부 돌려준다."""

    mid: np.ndarray
    u: np.ndarray
    cup_pt: np.ndarray
    along: float
    lateral: float
    axis_t_raw: float
    in_band: bool


def quat_to_matrix(q_wxyz) -> np.ndarray:
    """(w,x,y,z) → 3×3 회전행렬."""
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        raise ValueError("영 쿼터니언")
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z),     s * (x * z + w * y)],
        [s * (x * y + w * z),     1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y),     s * (y * z + w * x),     1 - s * (x * x + y * y)],
    ])


def jaw_frame(
    *,
    finger_l_pos,
    finger_r_pos,
    gripper_base_quat,
    cup_pos,
    cup_quat,
    cfg: GateCfg | None = None,
) -> JawFrame:
    """턱↔컵 기하. 손가락은 base 의 y 로만 미끄러지므로 접근축은 base 자세에서 얻는다."""
    cfg = cfg or GateCfg()
    approach = quat_to_matrix(gripper_base_quat)[:, 2]        # base z축 = 접근축
    p_l = np.asarray(finger_l_pos, dtype=np.float64).reshape(3) + approach * cfg.pad_offset
    p_r = np.asarray(finger_r_pos, dtype=np.float64).reshape(3) + approach * cfg.pad_offset
    mid = 0.5 * (p_l + p_r)
    jaw = p_r - p_l
    u = jaw / max(float(np.linalg.norm(jaw)), 1e-6)

    cup_pos = np.asarray(cup_pos, dtype=np.float64).reshape(3)
    cup_z = quat_to_matrix(cup_quat)[:, 2]
    axis_t_raw = float(np.dot(mid - cup_pos, cup_z))
    axis_t = float(np.clip(axis_t_raw, cfg.band_axis[0], cfg.band_axis[1]))
    cup_pt = cup_pos + cup_z * axis_t

    d = cup_pt - mid
    along = abs(float(np.dot(d, u)))
    lateral = float(np.linalg.norm(d - u * np.dot(d, u)))
    in_band = bool(cfg.band_axis[0] < axis_t_raw < cfg.band_axis[1])
    return JawFrame(mid=mid, u=u, cup_pt=cup_pt, along=along,
                    lateral=lateral, axis_t_raw=axis_t_raw, in_band=in_band)


def grasp_ok(frame: JawFrame, cfg: GateCfg | None = None) -> bool:
    """접근 성공 — 컵이 실제로 턱 사이 파지 위치에 있는가.

    ★기준은 `lateral` 이다. "축이 턱 사이를 지난다"만 보면 턱이 벌어졌을 때 컵에서
      8.5 cm 떨어져도 성립한다(학습 fab_test11 이 그 상태로 4000 epoch 을 돌았다).
    """
    cfg = cfg or GateCfg()
    return bool(frame.lateral < cfg.lateral_ok
                and frame.along < cfg.along_ok
                and frame.in_band)


class GraspGate:
    """게이트 래치 — 한 번 성립하면 유지, 컵이 턱에서 완전히 벗어날 때만 해제."""

    def __init__(self, cfg: GateCfg | None = None) -> None:
        self.cfg = cfg or GateCfg()
        self._open = False
        self.last: JawFrame | None = None

    def reset(self) -> None:
        self._open = False
        self.last = None

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def obs_value(self) -> float:
        """관측 36번째 칸에 넣는 값."""
        return 1.0 if self._open else 0.0

    def update(self, **jaw_kwargs) -> bool:
        frame = jaw_frame(cfg=self.cfg, **jaw_kwargs)
        self.last = frame
        if not self._open:
            self._open = grasp_ok(frame, self.cfg)
        elif self.cfg.release_lateral is not None:
            if frame.lateral > self.cfg.release_lateral:
                self._open = False
        return self._open
