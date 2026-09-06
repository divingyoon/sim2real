"""M3 골든(2) — 실제 체크포인트로 기록된 obs 를 재생해 기록된 액션과 대조.

LEFT (v2B25, MLP, clip 100): `stream_left_v2b25.npz` 65 스텝. 기록 actions 는 정책
  **원출력 mu**(|mu|max 1.608 > 1 — clip 100 은 아무것도 자르지 않는다). ≤1e-5.

RIGHT (e1, LSTM 1024, clip 1.0): `stream_right_e1_v2.npz` 194 스텝, hidden 연속.
  기록 actions 는 **±1 클립 후** 값(원소의 88 % 가 정확히 ±1). 두 오라클로 잠근다:
  * fp64 독립 재생(rl_games 모델을 직접 double 로 돌린 것) ≤1e-5 — hidden 연속성.
  * 기록 대비 ≤2e-3 — 기록은 Isaac play 의 배치 cuDNN(TF32, 상대 2^-11≈4.9e-4) 잡음을
    품고 있어 fp32/fp64 재생 어느 쪽도 1.6e-3 아래로 못 내려간다(fp64 vs 기록 1.619e-3,
    fp32 vs fp64 1.0e-6, 09.05 실측). 포화(±1) 원소는 정확히 일치해야 한다.
  음성: 중간 리셋 → 발산, seq 중복/역행 → hidden 불변(이후 스트림이 계속 맞는다).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from policy_control import contract as C
from policy_control import contract_build as B
from policy_control import policy_core as P

pytestmark = pytest.mark.golden

SIM2REAL = Path(__file__).resolve().parents[2]
LEFT_RUN = SIM2REAL / "logs/policy/left_v2B25"
RIGHT_E1_RUN = SIM2REAL / "logs/policy/right_e1"
FX = SIM2REAL / "tests/fixtures/policy_control"

LEFT_TOL = 1e-5
RIGHT_TOL_FP64 = 1e-5
RIGHT_TOL_RECORD = 2e-3
RESET_STEP = 100

needs_left = pytest.mark.skipif(not (LEFT_RUN / "nn").exists(), reason="left_v2B25 run dir 없음")
needs_right = pytest.mark.skipif(not (RIGHT_E1_RUN / "nn").exists(), reason="right_e1 run dir 없음")


def _cuda() -> bool:
    torch = pytest.importorskip("torch")
    return bool(torch.cuda.is_available())


def _devices() -> list[str]:
    return ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]


def _replay(core: P.PolicyCore, obs: np.ndarray, seq_of=lambda t: t) -> tuple[np.ndarray, list[float]]:
    outs, lat = [], []
    for t, o in enumerate(obs):
        t0 = time.perf_counter()
        outs.append(core.act(o, seq_of(t)))
        lat.append(time.perf_counter() - t0)
    return np.stack(outs), lat


def _report(tag: str, lat: list[float]) -> None:
    ms = np.asarray(lat) * 1e3
    print(f"[latency] {tag}: p50={np.percentile(ms, 50):.3f} ms p95={np.percentile(ms, 95):.3f} ms (n={len(ms)})")


# ---------------------------------------------------------------- LEFT
@pytest.fixture(scope="module")
def left_stream() -> dict:
    d = np.load(FX / "stream_left_v2b25.npz", allow_pickle=True)
    return {"obs": d["obs"], "actions": d["actions"]}


@needs_left
@pytest.mark.parametrize("device", _devices())
def test_left_mlp_matches_recorded_actions(left_stream, device):
    if device.startswith("cuda") and not _cuda():
        pytest.skip("CUDA 없음")
    contract = C.load_contract(LEFT_RUN / "deploy_contract.json")
    assert contract.policy.rnn is None and contract.policy.action_clip == pytest.approx(100.0)
    core = P.PolicyCore(contract, LEFT_RUN, device=device)
    assert (core.obs_dim, core.action_dim) == (49, 7)

    out, lat = _replay(core, left_stream["obs"])
    _report(f"left MLP {device}", lat)
    err = np.abs(out - left_stream["actions"]).max()
    assert err <= LEFT_TOL, f"max abs {err:.3e}"
    # 기록은 pre-clip(원출력): ±1 을 넘는 값이 그대로 들어 있고 우리도 자르지 않았다.
    assert np.abs(left_stream["actions"]).max() > 1.0
    assert np.abs(out).max() > 1.0


# ---------------------------------------------------------------- RIGHT
@pytest.fixture(scope="module")
def right_stream() -> dict:
    d = np.load(FX / "stream_right_e1_v2.npz", allow_pickle=True)
    return {"obs": d["obs"], "actions": d["actions"]}


@pytest.fixture(scope="module")
def right_core():
    if not _cuda():
        pytest.skip("CUDA 없음 — LSTM 골든은 GPU 에서만")
    contract = B.build_contract(RIGHT_E1_RUN)
    assert contract.run.checkpoint == "nn/e1_best.pth"
    assert contract.policy.rnn is not None and contract.policy.rnn["units"] == 1024
    assert contract.policy.action_clip == pytest.approx(1.0)
    return P.PolicyCore(contract, RIGHT_E1_RUN, device="cuda:0")


def _fp64_reference(run: Path, contract: C.DeployContract, obs: np.ndarray) -> np.ndarray:
    """rl_games 모델을 double 로 직접 돌린 독립 오라클(policy_core 를 거치지 않는다)."""
    import torch
    from policy_loader import RLGamesLstmActorPolicy

    pol = RLGamesLstmActorPolicy(str(run / "params/agent.yaml"), str(run / contract.run.checkpoint),
                                 obs_dim=contract.policy.obs_dim, action_dim=contract.policy.action_dim,
                                 device="cuda:0", action_clip=None)
    model = pol.model.double()
    states = [s.double().to("cuda:0") for s in model.get_default_rnn_state()]
    outs = []
    with torch.inference_mode():
        for o in obs:
            res = model({"is_train": False, "obs": torch.as_tensor(o, device="cuda:0").double().unsqueeze(0),
                         "rnn_states": states, "seq_length": 1, "rnn_masks": None})
            states = res["rnn_states"]
            outs.append(res["mus"].clamp(-1.0, 1.0)[0].cpu().numpy())
    return np.stack(outs)


@needs_right
@pytest.mark.gpu
def test_right_lstm_hidden_carried_matches_record(right_core, right_stream):
    out, lat = _replay(right_core, right_stream["obs"])
    _report("right LSTM cuda:0", lat)
    rec = right_stream["actions"]
    err = np.abs(out - rec)
    assert err.max() <= RIGHT_TOL_RECORD, f"max abs {err.max():.3e} at step {int(err.max(1).argmax())}"
    saturated = np.abs(rec) >= 1.0
    assert saturated.mean() > 0.5                       # 기록은 post-clip(±1 포화가 다수)
    assert err[saturated].max() == 0.0                  # 포화 원소는 클립 후 정확히 일치
    assert np.abs(out).max() <= 1.0


@needs_right
@pytest.mark.gpu
def test_right_lstm_matches_fp64_reference(right_core, right_stream):
    out, _ = _replay(right_core, right_stream["obs"])
    ref = _fp64_reference(RIGHT_E1_RUN, right_core.contract, right_stream["obs"])
    err = np.abs(out - ref).max()
    assert err <= RIGHT_TOL_FP64, f"fp32 core vs fp64 reference: {err:.3e}"


@needs_right
@pytest.mark.gpu
def test_right_reset_mid_stream_diverges(right_core, right_stream):
    obs, rec = right_stream["obs"], right_stream["actions"]
    out, _ = _replay(right_core, obs, seq_of=lambda t: t - RESET_STEP if t >= RESET_STEP else t)
    before = np.abs(out[:RESET_STEP] - rec[:RESET_STEP]).max()
    after = np.abs(out[RESET_STEP:] - rec[RESET_STEP:]).max()
    assert before <= RIGHT_TOL_RECORD
    assert after > 0.1, f"hidden 리셋이 액션을 바꾸지 않았다 (after={after:.3e})"


@needs_right
@pytest.mark.gpu
def test_right_duplicate_seq_keeps_hidden(right_core, right_stream):
    obs, rec = right_stream["obs"], right_stream["actions"]
    outs = []
    for t, o in enumerate(obs):
        a = right_core.act(o, t)
        if t == RESET_STEP:
            dup = right_core.act(obs[t + 1], t)           # 다른 obs, 같은 seq → 무시
            np.testing.assert_array_equal(dup, a)
        outs.append(a)
    err = np.abs(np.stack(outs) - rec).max()
    assert err <= RIGHT_TOL_RECORD, f"duplicate seq advanced the hidden state: {err:.3e}"


@needs_right
@pytest.mark.gpu
def test_right_backward_seq_keeps_hidden(right_core, right_stream):
    obs, rec = right_stream["obs"], right_stream["actions"]
    outs = []
    for t, o in enumerate(obs):
        a = right_core.act(o, t)
        if t == RESET_STEP:
            back = right_core.act(obs[t - 5], t - 5)      # 역행 seq → 무시
            np.testing.assert_array_equal(back, a)
        outs.append(a)
    err = np.abs(np.stack(outs) - rec).max()
    assert err <= RIGHT_TOL_RECORD, f"backward seq advanced the hidden state: {err:.3e}"
