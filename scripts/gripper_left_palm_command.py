"""`FabricPalmAction` 의 액션→palm 지령 산술만 떼어낸 것. Isaac 도 torch 도 필요 없다.

왜 떼어내는가: sim 쪽 원본(`grasp_left_fabric_action.py:process_actions`)은
`ActionTerm` 이라 `isaaclab` 없이는 import 되지 않는다. 그런데 이 산술이 곧
"정책이 무엇을 지시했는가"이고, 실기 그림자·오프라인 프로브 양쪽이 그것을 알아야 한다.

**여기 상수는 하나도 없다.** 전부 hdgp `grasp_left_preset` 에서 받는다 — 값을 옮겨 적으면
그 순간부터 조용히 어긋난다(이 저장소가 pour 계열에서 이미 겪은 방식). 순서까지 원본과
같게 유지한다: 박스 매핑 → 위치 변화율 상한 → 회전 변화율 상한 → 노름 클램프 → 쿼터니언
합성 → **xyzw 재배열**. 마지막 재배열은 `set_features` 의 규약이고, wxyz 로 넘기면
말없이 다른 자세로 간다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton 곱, 둘 다 wxyz."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_from_angle_axis(angle: float, axis: np.ndarray) -> np.ndarray:
    """wxyz. angle≈0 이면 축이 무의미하지만 결과는 identity 라 안전하다."""
    half = 0.5 * angle
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


@dataclass
class PalmCommandBuilder:
    """액션 6D → fabric `set_features` 가 받는 palm 목표 7D(xyz + **xyzw**)."""

    box_low: np.ndarray
    box_high: np.ndarray
    ref_quat_wxyz: np.ndarray
    rot_max_rad: float
    pos_rate_limit: float
    rot_rate_limit: float
    rate_limit_enabled: bool

    @classmethod
    def from_preset(cls, preset) -> "PalmCommandBuilder":
        low = np.array([preset.PALM_BOX_X[0], preset.PALM_BOX_Y[0], preset.PALM_BOX_Z[0]])
        high = np.array([preset.PALM_BOX_X[1], preset.PALM_BOX_Y[1], preset.PALM_BOX_Z[1]])
        return cls(
            box_low=low,
            box_high=high,
            ref_quat_wxyz=np.array(preset.PALM_REF_QUAT_WXYZ),
            rot_max_rad=float(preset.PALM_ROT_MAX_RAD),
            pos_rate_limit=float(preset.PALM_CMD_RATE_LIMIT),
            rot_rate_limit=float(preset.PALM_ROT_RATE_LIMIT),
            rate_limit_enabled=bool(preset.PALM_CMD_RATE_LIMIT_ENABLED),
        )

    def __post_init__(self) -> None:
        self._center = 0.5 * (self.box_low + self.box_high)
        self._half = 0.5 * (self.box_high - self.box_low)
        self.reset()

    def reset(self) -> None:
        """에피소드 경계. 첫 스텝은 변화율 상한을 적용하지 않는다.

        직전 에피소드의 지령에서 끌어오면 시작이 오염된다 — 원본이 `_cmd_primed`
        로 같은 일을 하고, 그 주석은 이 태스크에서 리셋 오염에 세 번 당했다고 적는다.
        """
        self._prev_pos = np.zeros(3)
        self._prev_rot = np.zeros(3)
        self._primed = False
        self.last_step_norm = 0.0

    def step(self, action6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(palm_pos[3], palm_quat_wxyz[4]) 를 돌려준다.

        `set_features` 로 넘길 때는 `as_features` 를 쓸 것 — 거기서 xyzw 가 된다.
        """
        action6 = np.asarray(action6, dtype=float).reshape(6)
        fresh = (not self._primed) or (not self.rate_limit_enabled)

        pos = self._center + np.clip(action6[:3], -1.0, 1.0) * self._half
        if not fresh:
            delta = pos - self._prev_pos
            norm = np.linalg.norm(delta)
            pos = self._prev_pos + delta * min(self.pos_rate_limit / max(norm, 1e-9), 1.0)
        self.last_step_norm = 0.0 if fresh else float(np.linalg.norm(pos - self._prev_pos))
        self._prev_pos = pos

        rotvec = action6[3:6] * self.rot_max_rad
        if not fresh:
            delta = rotvec - self._prev_rot
            norm = np.linalg.norm(delta)
            rotvec = self._prev_rot + delta * min(self.rot_rate_limit / max(norm, 1e-9), 1.0)
        self._prev_rot = rotvec
        self._primed = True

        angle = float(np.linalg.norm(rotvec))
        if angle > self.rot_max_rad:
            rotvec = rotvec * (self.rot_max_rad / max(angle, 1e-9))
            angle = self.rot_max_rad
        axis = rotvec / max(angle, 1e-9)
        quat = _quat_mul_wxyz(quat_from_angle_axis(angle, axis), self.ref_quat_wxyz)
        return pos, quat

    @staticmethod
    def as_features(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        """fabric 이 받는 7D. 쿼터니언은 **xyzw** 로 뒤집힌다."""
        return np.concatenate([pos, quat_wxyz[1:4], quat_wxyz[:1]])
