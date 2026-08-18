"""tip_contact_core 유닛 테스트 — F/T wrench → bias 제거 → tip force 3축 벡터.

실행: pytest scripts/test_tip_contact_core.py -q  (ROS 불필요, numpy only)
"""

from __future__ import annotations

import numpy as np
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


# --------------------------------------------------------------------------
# 3축 벡터화 (08.18) — sim obs `tip_force_local` 15D 정합
# --------------------------------------------------------------------------

class TestForcesXyz:
    """norm 이 아니라 **3축 벡터**가 주경로다. norm 은 파생으로만 남는다."""

    @staticmethod
    def _biased(**kw):
        ex = TipForceExtractor(num_tips=5, bias_samples=2, **kw)
        for t in range(5):
            for _ in range(2):
                ex.update(t, [0.0, 0.0, 0.0])
        return ex

    def test_shape_and_zero_before_bias(self):
        ex = TipForceExtractor(num_tips=5, bias_samples=3)
        ex.update(0, [9.0, 9.0, 9.0])            # bias 캡처 구간
        assert ex.forces_xyz().shape == (5, 3)
        assert np.allclose(ex.forces_xyz(), 0.0)  # bias 미완 → 0 유지(무접촉 전제)
        assert not ex.biased()

    def test_axis_wise_bias_subtraction(self):
        """bias 는 축별 벡터로 차감된다 — norm 으로 뭉개지 않는다."""
        ex = TipForceExtractor(num_tips=5, bias_samples=2)
        for _ in range(2):
            ex.update(1, [1.0, -2.0, 3.0])       # tip1 bias
        for t in (0, 2, 3, 4):
            for _ in range(2):
                ex.update(t, [0.0, 0.0, 0.0])
        ex.update(1, [1.5, -2.0, 3.0])           # x 로만 +0.5
        assert ex.forces_xyz()[1] == pytest.approx([0.5, 0.0, 0.0])

    def test_sign_preserved_per_axis(self):
        """부호가 보존되어야 방향 정보가 산다(norm 이면 사라진다)."""
        ex = self._biased()
        ex.update(2, [0.0, 0.0, -1.5])
        assert ex.forces_xyz()[2] == pytest.approx([0.0, 0.0, -1.5])
        assert ex.forces()[2] == pytest.approx(1.5)   # norm 은 양수

    def test_forces_is_norm_of_xyz(self):
        ex = self._biased()
        ex.update(3, [3.0, 4.0, 0.0])
        assert ex.forces()[3] == pytest.approx(5.0)
        assert np.allclose(ex.forces(), np.linalg.norm(ex.forces_xyz(), axis=1))

    def test_binary_threshold(self):
        ex = self._biased()
        ex.update(0, [0.05, 0.0, 0.0])           # 임계 미만
        ex.update(1, [0.2, 0.0, 0.0])            # 임계 초과
        b = ex.binary(0.1)
        assert b[0] == 0.0 and b[1] == 1.0
        assert b.shape == (5,)

    def test_sign_flip_applied(self):
        ex = self._biased(sign=-1.0)
        ex.update(0, [1.0, 2.0, 3.0])
        assert ex.forces_xyz()[0] == pytest.approx([-1.0, -2.0, -3.0])

    def test_rotation_applied_after_bias(self):
        """회전은 bias 차감 **뒤** 적용 — 순서를 바꾸면 bias 가 회전돼 틀린다."""
        Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # +90° about z
        rot = np.tile(Rz, (5, 1, 1))
        ex = TipForceExtractor(num_tips=5, bias_samples=2, rotation=rot)
        for _ in range(2):
            ex.update(0, [10.0, 0.0, 0.0])       # 큰 bias
        for t in range(1, 5):
            for _ in range(2):
                ex.update(t, [0.0, 0.0, 0.0])
        ex.update(0, [11.0, 0.0, 0.0])           # bias 대비 x 로 +1
        # 올바른 순서(차감→회전): R([11,0,0]-[10,0,0]) = R[1,0,0] = [0,1,0]
        # 잘못된 순서(회전→차감): R[11,0,0]-[10,0,0] = [0,11,0]-[10,0,0] = [-10,11,0]
        # 두 값이 명확히 갈리므로 이 assert 가 순서를 고정한다.
        assert ex.forces_xyz()[0] == pytest.approx([0.0, 1.0, 0.0])

    def test_rotation_shape_validated(self):
        with pytest.raises(ValueError, match="rotation"):
            TipForceExtractor(num_tips=5, rotation=np.eye(3))

    def test_reset_bias_clears_vectors(self):
        ex = self._biased()
        ex.update(0, [1.0, 1.0, 1.0])
        assert not np.allclose(ex.forces_xyz(), 0.0)
        ex.reset_bias()
        assert np.allclose(ex.forces_xyz(), 0.0)
        assert not ex.biased()
