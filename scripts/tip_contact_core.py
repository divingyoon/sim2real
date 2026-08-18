#!/usr/bin/env python3
"""Tesollo tip F/T → 접촉력 추출 순수 로직 (numpy only, ROS 무의존).

실물 dg5f 드라이버(`fingertip_sensor:=true`)는 손끝 F/T 를
`fingertip_{1..5}_broadcaster/wrench`(WrenchStamped) 로 발행한다. 여기서:

  1. 시작 시(무접촉 전제) tip 별 bias **벡터** = 첫 bias_samples 평균
  2. f_i = sign · R_i · (raw_i − bias_i)      ← 3축 벡터를 그대로 유지
  3. 이진 접촉은 ‖f_i‖ > threshold 로 **파생**한다(별도 토픽 만들지 않음)

★3축을 유지하는 이유: sim 학습 obs 가 `tip_force_local` 15D(5×3) 를 쓴다
  (grasp_right_env.py). sim 은 이 값을 **tip-local 프레임**으로 계산해 두었는데,
  실물 F/T 가 센서 로컬 출력이라 변환 없이 그대로 들어가게 하려는 의도다.
  따라서 배포에서 world 변환을 넣으면 안 된다.

⚠️ 프레임·부호 미검증: 실물 broadcaster 의 출력 프레임이 URDF tip 링크와 같은지,
   힘 부호 규약(센서에 가해진 힘 vs 반작용)이 sim 과 같은지 확인된 바 없다.
   `rotation`/`sign` 을 구성 프로필로 노출해 두었으니, 손끝을 알려진 방향으로 눌러
   3축 부호를 표로 기록하는 캘리브를 마친 뒤 채울 것. 그 전에는 identity/+1.

⚠️ bias 는 자세 의존적이다. 기동 시 1회 캡처한 bias 는 손이 움직이면 중력 성분이
   tip-local 에서 회전해 어긋난다 → 무접촉이 확실한 시점에 재캡처하는 것이 옳다.

tip 순서: 0..4 = thumb..pinky (= rj_dg_1..5 = fingertip_{1..5}_broadcaster).
"""

from __future__ import annotations

import numpy as np

NUM_TIPS_DEFAULT = 5
BIAS_SAMPLES_DEFAULT = 30


class TipForceExtractor:
    """tip 별 bias 캡처 후 bias-제거 force **벡터**를 유지한다."""

    def __init__(
        self,
        num_tips: int = NUM_TIPS_DEFAULT,
        bias_samples: int = BIAS_SAMPLES_DEFAULT,
        rotation: np.ndarray | None = None,
        sign: float = 1.0,
    ) -> None:
        if num_tips < 1 or bias_samples < 1:
            raise ValueError("num_tips/bias_samples must be >= 1")
        self.num_tips = int(num_tips)
        self.bias_samples = int(bias_samples)
        self.sign = float(sign)
        if rotation is None:
            self.rotation = np.tile(np.eye(3), (self.num_tips, 1, 1))
        else:
            R = np.asarray(rotation, dtype=np.float64)
            if R.shape != (self.num_tips, 3, 3):
                raise ValueError(f"rotation expected shape ({self.num_tips},3,3), got {R.shape}")
            self.rotation = R
        self.reset_bias()

    def reset_bias(self) -> None:
        """bias 재캡처 시작 (손 무접촉 상태에서 호출할 것)."""
        self._bias_acc = np.zeros((self.num_tips, 3), dtype=np.float64)
        self._bias_n = np.zeros(self.num_tips, dtype=np.int64)
        self._bias = np.zeros((self.num_tips, 3), dtype=np.float64)
        self._force_xyz = np.zeros((self.num_tips, 3), dtype=np.float64)

    def biased(self) -> bool:
        """모든 tip 의 bias 캡처 완료 여부."""
        return bool(np.all(self._bias_n >= self.bias_samples))

    def update(self, tip: int, force_xyz) -> None:
        """tip 의 새 wrench force 샘플 반영 (비유한 샘플은 무시)."""
        t = int(tip)
        if not (0 <= t < self.num_tips):
            raise ValueError(f"tip index {t} out of range [0, {self.num_tips})")
        f = np.asarray(force_xyz, dtype=np.float64).reshape(-1)
        if f.shape != (3,):
            raise ValueError(f"force_xyz expected shape (3,), got {f.shape}")
        if not np.all(np.isfinite(f)):
            return   # 센서 글리치 — 직전값 유지

        if self._bias_n[t] < self.bias_samples:
            self._bias_acc[t] += f
            self._bias_n[t] += 1
            if self._bias_n[t] == self.bias_samples:
                self._bias[t] = self._bias_acc[t] / self.bias_samples
            return   # bias 완료 전엔 0 유지 (무접촉 전제)

        # bias 차감 → 프레임 정렬 → 부호. 순서를 바꾸면 bias 가 회전되어 틀린다.
        self._force_xyz[t] = self.sign * (self.rotation[t] @ (f - self._bias[t]))

    def forces_xyz(self) -> np.ndarray:
        """(num_tips, 3) bias-제거 force 벡터 [N] 사본. bias 미완 tip 은 0.

        정책 obs(`tip_force_local` 15D)의 주경로다.
        """
        return self._force_xyz.copy()

    def forces(self) -> np.ndarray:
        """(num_tips,) force norm [N] — 하위호환·모니터링용 파생값."""
        return np.linalg.norm(self._force_xyz, axis=1)

    def binary(self, threshold: float) -> np.ndarray:
        """(num_tips,) 이진 접촉 = ‖f‖ > threshold. 별도 토픽을 만들지 않고 여기서 파생."""
        return (self.forces() > float(threshold)).astype(np.float64)
