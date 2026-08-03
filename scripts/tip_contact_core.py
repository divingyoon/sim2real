#!/usr/bin/env python3
"""Tesollo tip F/T → 접촉력 norm 추출 순수 로직 (numpy only, ROS 무의존).

실물 dg5f 드라이버(`fingertip_sensor:=true`)는 손끝 F/T 를
`fingertip_{1..5}_broadcaster/wrench`(WrenchStamped) 로 발행한다. 정책 obs·게이트가
쓰는 `/dg5f_right/contact_forces` 는 **tip 당 스칼라 힘[N] 5개** — 여기서 변환한다:

  1. 시작 시(무접촉 전제) tip 별 bias 벡터 = 첫 bias_samples 평균 (센서 오프셋/자세 성분 제거)
  2. force_i = ‖f_i − bias_i‖  (force 3축만; torque 미사용 — sim 학습 정합,
     [[grasp-v2-contact-obs-sim2real]] 3정합 중 grasp_v1 은 이진 접촉이라 norm 으로 충분)
  3. 이진화는 하지 않는다 — grasp_inference 가 CONTACT_FORCE_THRESHOLD(0.1N) 로 판정.
     ★실물 임계는 하드웨어에서 튜닝 필요(무접촉 노이즈 < 임계 < 실접촉 확인).

tip 순서: 0..4 = thumb..pinky (= rj_dg_1..5 = fingertip_{1..5}_broadcaster).
"""

from __future__ import annotations

import numpy as np

NUM_TIPS_DEFAULT = 5
BIAS_SAMPLES_DEFAULT = 30


class TipForceExtractor:
    """tip 별 bias 캡처 후 bias-제거 force norm 을 유지한다."""

    def __init__(
        self,
        num_tips: int = NUM_TIPS_DEFAULT,
        bias_samples: int = BIAS_SAMPLES_DEFAULT,
    ) -> None:
        if num_tips < 1 or bias_samples < 1:
            raise ValueError("num_tips/bias_samples must be >= 1")
        self.num_tips = int(num_tips)
        self.bias_samples = int(bias_samples)
        self.reset_bias()

    def reset_bias(self) -> None:
        """bias 재캡처 시작 (손 무접촉 상태에서 호출할 것)."""
        self._bias_acc = np.zeros((self.num_tips, 3), dtype=np.float64)
        self._bias_n = np.zeros(self.num_tips, dtype=np.int64)
        self._bias = np.zeros((self.num_tips, 3), dtype=np.float64)
        self._force = np.zeros(self.num_tips, dtype=np.float64)

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
            return   # bias 완료 전엔 force 0 유지 (무접촉 전제)

        self._force[t] = float(np.linalg.norm(f - self._bias[t]))

    def forces(self) -> np.ndarray:
        """(num_tips,) bias-제거 force norm [N] 사본. bias 미완 tip 은 0."""
        return self._force.copy()
