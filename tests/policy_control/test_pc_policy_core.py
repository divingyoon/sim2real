"""M3 — policy core: seq 프로토콜 · clip 의미 · 입력 검사 (체크포인트 없이, fake backend).

규칙(계획 §4.5):
  * seq 0 → hidden 리셋(zero_rnn_on_done) + 카운터 리셋 후 forward.
  * seq 중복/역행 → **forward 없이** 직전 액션 반환(hidden 불변).
  * 첫 obs 는 반드시 seq 0 (에피소드 시작 신호는 obs 의 seq 0 하나).
  * clip 은 계약값: None → 무클립, 100.0 → ±1 로 자르지 않는다(policy_loader 79-86 사고).
  * obs 차원 불일치 / NaN·inf → 예외(조용한 기본값 금지).
"""
from __future__ import annotations

import numpy as np
import pytest

from policy_control import policy_core as P

pytestmark = pytest.mark.unit

OBS_DIM = 4
ACT_DIM = 3


class FakeBackend:
    """forward 횟수를 hidden 처럼 누적해 '재계산'과 '캐시 반환'을 구분한다."""

    def __init__(self) -> None:
        self.hidden = 0
        self.resets = 0
        self.calls: list[np.ndarray] = []

    def forward(self, obs: np.ndarray) -> np.ndarray:
        self.calls.append(obs)
        self.hidden += 1
        return np.full(ACT_DIM, float(self.hidden) * float(obs[0]), dtype=np.float32)

    def reset(self) -> None:
        self.resets += 1
        self.hidden = 0


def _obs(v: float) -> np.ndarray:
    return np.full(OBS_DIM, v, dtype=np.float32)


# ---------------------------------------------------------------- apply_seq_rule (pure)
def test_seq_zero_resets_backend_and_forwards():
    b = FakeBackend()
    b.hidden = 7
    state, action = P.apply_seq_rule(P.SeqState(), 0, _obs(1.0), b.forward, b.reset)
    assert b.resets == 1
    assert state.seq == 0
    np.testing.assert_allclose(action, [1.0, 1.0, 1.0])       # hidden 0→1 after reset


def test_first_obs_must_carry_seq_zero():
    b = FakeBackend()
    with pytest.raises(P.SeqError):
        P.apply_seq_rule(P.SeqState(), 3, _obs(1.0), b.forward, b.reset)
    assert b.calls == [] and b.resets == 0


def test_negative_seq_rejected():
    b = FakeBackend()
    with pytest.raises(P.SeqError):
        P.apply_seq_rule(P.SeqState(), -1, _obs(1.0), b.forward, b.reset)


def test_advancing_seq_forwards_each_step():
    b = FakeBackend()
    s, a0 = P.apply_seq_rule(P.SeqState(), 0, _obs(1.0), b.forward, b.reset)
    s, a1 = P.apply_seq_rule(s, 1, _obs(1.0), b.forward, b.reset)
    s, a2 = P.apply_seq_rule(s, 2, _obs(1.0), b.forward, b.reset)
    assert s.seq == 2 and len(b.calls) == 3 and b.resets == 1
    np.testing.assert_allclose([a0[0], a1[0], a2[0]], [1.0, 2.0, 3.0])


def test_duplicate_seq_returns_previous_action_without_forward():
    b = FakeBackend()
    s, _ = P.apply_seq_rule(P.SeqState(), 0, _obs(1.0), b.forward, b.reset)
    s, a1 = P.apply_seq_rule(s, 1, _obs(1.0), b.forward, b.reset)
    n_calls = len(b.calls)
    s2, a_dup = P.apply_seq_rule(s, 1, _obs(5.0), b.forward, b.reset)   # obs differs, seq same
    assert len(b.calls) == n_calls and b.hidden == 2
    np.testing.assert_array_equal(a_dup, a1)
    assert s2 == s


def test_backward_seq_returns_previous_action_without_forward():
    b = FakeBackend()
    s, _ = P.apply_seq_rule(P.SeqState(), 0, _obs(1.0), b.forward, b.reset)
    s, _ = P.apply_seq_rule(s, 1, _obs(1.0), b.forward, b.reset)
    s, a2 = P.apply_seq_rule(s, 2, _obs(1.0), b.forward, b.reset)
    n_calls = len(b.calls)
    s2, a_back = P.apply_seq_rule(s, 1, _obs(9.0), b.forward, b.reset)
    assert len(b.calls) == n_calls and s2.seq == 2 and b.resets == 1
    np.testing.assert_array_equal(a_back, a2)


def test_seq_zero_after_progress_resets_again():
    b = FakeBackend()
    s, _ = P.apply_seq_rule(P.SeqState(), 0, _obs(1.0), b.forward, b.reset)
    s, _ = P.apply_seq_rule(s, 1, _obs(1.0), b.forward, b.reset)
    s, a = P.apply_seq_rule(s, 0, _obs(1.0), b.forward, b.reset)
    assert b.resets == 2 and s.seq == 0
    np.testing.assert_allclose(a, [1.0, 1.0, 1.0])


def test_gap_in_seq_advances():
    b = FakeBackend()
    s, _ = P.apply_seq_rule(P.SeqState(), 0, _obs(1.0), b.forward, b.reset)
    s, a = P.apply_seq_rule(s, 5, _obs(1.0), b.forward, b.reset)
    assert s.seq == 5 and len(b.calls) == 2
    np.testing.assert_allclose(a, [2.0, 2.0, 2.0])


def test_apply_seq_rule_does_not_mutate_input_state():
    b = FakeBackend()
    s0 = P.SeqState()
    s1, _ = P.apply_seq_rule(s0, 0, _obs(1.0), b.forward, b.reset)
    assert s0.seq is None and s0.action is None
    assert s1 is not s0


def test_backend_action_dim_is_checked():
    def bad_forward(obs: np.ndarray) -> np.ndarray:
        return np.zeros(ACT_DIM + 1, dtype=np.float32)

    with pytest.raises(P.PolicyIOError):
        P.apply_seq_rule(P.SeqState(), 0, _obs(1.0), bad_forward, lambda: None, action_dim=ACT_DIM)


# ---------------------------------------------------------------- clip_action (pure)
def test_clip_none_passes_raw_mu_through():
    mu = np.array([-2.4045, 1.6, 0.3], dtype=np.float32)
    out = P.clip_action(mu, None)
    np.testing.assert_array_equal(out, mu)
    assert out is not mu


def test_clip_100_does_not_clamp_to_unit():
    mu = np.array([-2.4045, 1.6, 0.3], dtype=np.float32)
    np.testing.assert_array_equal(P.clip_action(mu, 100.0), mu)
    np.testing.assert_array_equal(P.clip_action(np.array([150.0, -150.0]), 100.0), [100.0, -100.0])


def test_clip_1_clamps():
    mu = np.array([-2.4045, 1.6, 0.3], dtype=np.float32)
    np.testing.assert_allclose(P.clip_action(mu, 1.0), [-1.0, 1.0, 0.3], rtol=0, atol=1e-7)


def test_clip_rejects_nonpositive():
    with pytest.raises(P.PolicyIOError):
        P.clip_action(np.zeros(2, dtype=np.float32), 0.0)


# ---------------------------------------------------------------- check_obs (pure)
def test_check_obs_returns_float32_copy():
    obs = np.arange(OBS_DIM, dtype=np.float64)
    out = P.check_obs(obs, OBS_DIM)
    assert out.dtype == np.float32 and out.shape == (OBS_DIM,)
    out[0] = 99.0
    assert obs[0] == 0.0


@pytest.mark.parametrize("bad", [np.zeros(OBS_DIM - 1), np.zeros(OBS_DIM + 1), np.zeros((1, OBS_DIM)), np.zeros(())])
def test_check_obs_rejects_wrong_shape(bad):
    with pytest.raises(P.PolicyIOError):
        P.check_obs(bad, OBS_DIM)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_check_obs_rejects_nonfinite(value):
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[2] = value
    with pytest.raises(P.PolicyIOError):
        P.check_obs(obs, OBS_DIM)


# ---------------------------------------------------------------- PolicyCore with a fake backend
def _core(clip: float | None = None) -> tuple[P.PolicyCore, FakeBackend]:
    b = FakeBackend()
    return P.PolicyCore.with_backend(b, obs_dim=OBS_DIM, action_dim=ACT_DIM, action_clip=clip), b


def test_core_dims_and_act_pipeline():
    core, b = _core(clip=1.0)
    assert core.obs_dim == OBS_DIM and core.action_dim == ACT_DIM
    a = core.act(_obs(3.0), 0)
    assert a.shape == (ACT_DIM,) and a.dtype == np.float32
    np.testing.assert_allclose(a, [1.0, 1.0, 1.0])          # raw 3.0 clipped to 1.0
    assert b.resets == 1 and core.state.seq == 0


def test_core_act_rejects_bad_obs_before_touching_backend():
    core, b = _core()
    with pytest.raises(P.PolicyIOError):
        core.act(np.zeros(OBS_DIM + 1, dtype=np.float32), 0)
    nan = _obs(1.0)
    nan[1] = np.nan
    with pytest.raises(P.PolicyIOError):
        core.act(nan, 0)
    assert b.calls == [] and b.resets == 0


def test_core_duplicate_seq_returns_copy_of_previous():
    core, b = _core()
    core.act(_obs(1.0), 0)
    a1 = core.act(_obs(2.0), 1)
    a_dup = core.act(_obs(7.0), 1)                            # seq 0 은 언제나 리셋 — 중복은 seq 1 로 본다
    np.testing.assert_array_equal(a_dup, a1)
    assert len(b.calls) == 2 and b.resets == 1
    a_dup[0] = -1.0                                           # caller mutation must not leak
    np.testing.assert_array_equal(core.act(_obs(7.0), 1), a1)


def test_core_reset_clears_state_and_backend():
    core, b = _core()
    core.act(_obs(1.0), 0)
    core.act(_obs(1.0), 1)
    core.reset()
    assert core.state.seq is None and core.state.action is None and b.resets == 2
    with pytest.raises(P.SeqError):
        core.act(_obs(1.0), 2)


def test_core_clip_none_keeps_raw_mu():
    core, _ = _core(clip=None)
    np.testing.assert_allclose(core.act(_obs(3.0), 0), [3.0, 3.0, 3.0])
