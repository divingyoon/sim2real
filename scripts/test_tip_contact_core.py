"""tip_contact_core 유닛 테스트 — F/T wrench → bias 제거 → tip force norm.

실행: pytest scripts/test_tip_contact_core.py -q  (ROS 불필요, numpy only)
"""

from __future__ import annotations

import pytest

from tip_contact_core import TipForceExtractor


def _fill_bias(ex: TipForceExtractor, tip: int, bias, n: int) -> None:
    for _ in range(n):
        ex.update(tip, bias)


class TestBiasCapture:
    def test_forces_zero_until_bias_complete(self):
        ex = TipForceExtractor(num_tips=5, bias_samples=10)
        _fill_bias(ex, 0, [1.0, 0.0, 0.0], 9)   # 아직 bias 미완
        assert ex.forces()[0] == 0.0
        assert not ex.biased()

    def test_bias_subtracted_norm(self):
        ex = TipForceExtractor(num_tips=5, bias_samples=5)
        for t in range(5):
            _fill_bias(ex, t, [0.5, -0.2, 9.8], 5)   # 정지 바이어스(중력 성분 등)
        assert ex.biased()
        # bias + 순수 접촉력 [0,0,-1.5] → norm 1.5
        ex.update(2, [0.5, -0.2, 9.8 - 1.5])
        f = ex.forces()
        assert f[2] == pytest.approx(1.5, abs=1e-9)
        # 나머지 tip 은 마지막 샘플=bias → 0
        assert f[0] == pytest.approx(0.0, abs=1e-9)

    def test_reset_bias_recaptures(self):
        ex = TipForceExtractor(num_tips=5, bias_samples=3)
        for t in range(5):
            _fill_bias(ex, t, [1.0, 0.0, 0.0], 3)
        ex.update(0, [3.0, 0.0, 0.0])
        assert ex.forces()[0] == pytest.approx(2.0)
        ex.reset_bias()
        assert not ex.biased()
        assert ex.forces()[0] == 0.0
        for t in range(5):
            _fill_bias(ex, t, [3.0, 0.0, 0.0], 3)   # 새 바이어스
        ex.update(0, [3.0, 0.0, 4.0])
        assert ex.forces()[0] == pytest.approx(4.0)


class TestValidation:
    def test_invalid_tip_index_raises(self):
        ex = TipForceExtractor(num_tips=5, bias_samples=3)
        with pytest.raises(ValueError):
            ex.update(5, [0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            ex.update(-1, [0.0, 0.0, 0.0])

    def test_wrong_shape_raises(self):
        ex = TipForceExtractor(num_tips=5, bias_samples=3)
        with pytest.raises(ValueError):
            ex.update(0, [0.0, 0.0])

    def test_non_finite_sample_ignored(self):
        ex = TipForceExtractor(num_tips=5, bias_samples=2)
        for t in range(5):
            _fill_bias(ex, t, [1.0, 0.0, 0.0], 2)
        ex.update(0, [1.0, 0.0, 2.0])
        f_before = ex.forces()[0]
        ex.update(0, [float("nan"), 0.0, 0.0])   # 글리치 → 직전값 유지
        assert ex.forces()[0] == pytest.approx(f_before)
