"""policy core — 계약이 고른 rl_games actor 를 "벡터 in → 벡터 out" 으로 감싼다.

태스크 불변: 이 모듈은 obs 의 의미도 액션의 의미도 모른다. 아는 것은 계약의
``policy`` 절(obs_dim / action_dim / rnn / action_clip)과 체크포인트 신원뿐이다.

seq 프로토콜(계획 §4.1·§4.5) — 리셋 신호는 obs 의 ``seq == 0`` 하나:
  * ``seq == 0``  → hidden 리셋(zero_rnn_on_done) + 카운터 리셋 후 forward.
  * ``seq > last`` → forward (gap 은 허용; LSTM 런의 gap 상한은 에피소드 쪽 책임).
  * ``seq <= last`` (중복·역행) → **forward 없이** 직전 액션을 돌려준다. hidden 불변.
  * 리셋을 본 적 없는 ``seq > 0`` → ``SeqError`` (에피소드 중간 합류 금지).

action_clip 은 계약값을 이 모듈이 단 한 곳에서 적용한다(로더에는 ``action_clip=None``).
``None`` 은 무클립, 좌 v2B25 의 100.0 은 ±1 로 자르지 않는다 — 학습 obs 의 last_action
은 원출력 mu 이므로 ±1 로 자르면 다음 스텝 obs 가 틀린다(policy_loader.py 79-86).

torch 텐서는 ``RlGamesBackend`` 안에만 산다. 순수 부분(``check_obs`` · ``clip_action``
· ``apply_seq_rule``)은 체크포인트 없이 fake backend 로 시험한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from . import _paths  # noqa: F401  (scripts/ on sys.path for policy_loader)
from .contract import ContractError, DeployContract, verify_checkpoint

Forward = Callable[[np.ndarray], np.ndarray]
Reset = Callable[[], None]


class SeqError(ValueError):
    """seq 프로토콜 위반: 음수 seq, 또는 seq 0 을 보기 전의 seq > 0."""


class PolicyIOError(ValueError):
    """obs/action 벡터가 계약과 맞지 않는다 (차원·비유한값·잘못된 clip)."""


@dataclass(frozen=True)
class SeqState:
    """마지막으로 **수락된** seq 와 그때의 원출력(클립 전) 액션. 시작 = (None, None)."""

    seq: int | None = None
    action: np.ndarray | None = None


class PolicyBackend(Protocol):
    """obs(obs_dim,) float32 → 원출력 mu(action_dim,); reset 은 hidden 을 0 으로."""

    def forward(self, obs: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


# ------------------------------------------------------------------ pure helpers
def check_obs(obs: np.ndarray, obs_dim: int) -> np.ndarray:
    """계약 차원의 1-D 유한 벡터인지 확인하고 float32 **사본**을 돌려준다."""
    arr = np.asarray(obs)
    if arr.shape != (obs_dim,):
        raise PolicyIOError(f"obs shape {arr.shape} != ({obs_dim},)")
    out = arr.astype(np.float32, copy=True)
    if not np.isfinite(out).all():
        bad = np.flatnonzero(~np.isfinite(out)).tolist()
        raise PolicyIOError(f"obs has non-finite values at indices {bad}")
    return out


def clip_action(action: np.ndarray, clip: float | None) -> np.ndarray:
    """계약 ``action_clip`` 적용. None → 무클립(사본 반환). clip 은 양수여야 한다."""
    arr = np.asarray(action, dtype=np.float32)
    if clip is None:
        return arr.copy()
    if not clip > 0.0:
        raise PolicyIOError(f"action_clip must be positive or None, got {clip!r}")
    return np.clip(arr, -float(clip), float(clip))


def apply_seq_rule(state: SeqState, seq: int, obs: np.ndarray, forward: Forward, reset: Reset,
                   action_dim: int | None = None) -> tuple[SeqState, np.ndarray]:
    """seq 규칙 한 스텝. 새 ``SeqState`` 와 원출력 액션(사본)을 돌려준다 — 입력은 손대지 않는다."""
    if isinstance(seq, bool) or not isinstance(seq, (int, np.integer)):
        raise SeqError(f"seq must be an integer, got {type(seq).__name__}")
    seq = int(seq)
    if seq < 0:
        raise SeqError(f"seq must be >= 0, got {seq}")
    if seq == 0:
        reset()
        return _advance(seq, obs, forward, action_dim)
    if state.seq is None or state.action is None:
        raise SeqError(f"episode not started: first obs must carry seq 0 (got seq {seq})")
    if seq <= state.seq:
        return state, state.action.copy()
    return _advance(seq, obs, forward, action_dim)


def _advance(seq: int, obs: np.ndarray, forward: Forward, action_dim: int | None) -> tuple[SeqState, np.ndarray]:
    raw = np.array(forward(obs), dtype=np.float32)
    want = (action_dim,) if action_dim is not None else None
    if raw.ndim != 1 or (want is not None and raw.shape != want):
        raise PolicyIOError(f"policy returned shape {raw.shape}, expected {want or '(action_dim,)'}")
    if not np.isfinite(raw).all():
        raise PolicyIOError(f"policy returned non-finite action at seq {seq}")
    raw.setflags(write=False)
    return SeqState(seq=seq, action=raw), raw.copy()


# ------------------------------------------------------------------ rl_games backend
def _sim2real_root(run_dir: Path, contract_run_dir: str) -> Path:
    """``verify_checkpoint`` 가 원하는 루트: root / contract.run.dir == run_dir 이어야 한다."""
    run_dir = Path(run_dir).resolve()
    rel = Path(contract_run_dir)
    if rel.is_absolute():
        if rel.resolve() != run_dir:
            raise ContractError(f"run_dir {run_dir} != contract run.dir {rel}")
        return run_dir
    n = len(rel.parts)
    if n == 0 or run_dir.parts[-n:] != rel.parts:
        raise ContractError(f"run_dir {run_dir} does not end with contract run.dir {contract_run_dir!r}")
    return Path(*run_dir.parts[:-n])


class RlGamesBackend:
    """policy_loader 의 MLP/LSTM actor 를 계약대로 골라 적재한다. 클립은 하지 않는다."""

    def __init__(self, contract: DeployContract, run_dir: Path, device: str) -> None:
        import torch
        from policy_loader import RLGamesActorPolicy, RLGamesLstmActorPolicy

        run_dir = Path(run_dir)
        agent_yaml = run_dir / "params" / "agent.yaml"
        if not agent_yaml.exists():
            raise ContractError(f"agent.yaml missing: {agent_yaml}")
        ckpt = verify_checkpoint(contract, _sim2real_root(run_dir, contract.run.dir))
        rnn = contract.policy.rnn
        if rnn is None:
            loader = RLGamesActorPolicy
        elif rnn.get("type") == "lstm":
            loader = RLGamesLstmActorPolicy
        else:
            raise ContractError(f"unsupported policy.rnn {rnn!r} (only None or type 'lstm')")
        self._torch = torch
        self._device = device
        self._is_rnn = rnn is not None
        self._policy = loader(str(agent_yaml), str(ckpt), obs_dim=contract.policy.obs_dim,
                              action_dim=contract.policy.action_dim, device=device, action_clip=None)
        if bool(self._policy.model.is_rnn()) != self._is_rnn:
            raise ContractError("contract policy.rnn disagrees with the checkpoint network (rnn vs mlp)")

    def forward(self, obs: np.ndarray) -> np.ndarray:
        t = self._torch.as_tensor(obs, dtype=self._torch.float32, device=self._device).unsqueeze(0)
        mu = self._policy.get_action(t)
        return mu[0].detach().cpu().numpy()

    def reset(self) -> None:
        if self._is_rnn:
            self._policy.reset_states()


# ------------------------------------------------------------------ core
class PolicyCore:
    """계약 + 런 디렉토리 → ``act(obs, seq) -> action``. 상태는 ``SeqState`` 하나."""

    def __init__(self, contract: DeployContract, run_dir: Path, device: str = "cuda:0") -> None:
        backend = RlGamesBackend(contract, Path(run_dir), device)
        self._setup(backend, contract.policy.obs_dim, contract.policy.action_dim,
                    contract.policy.action_clip, contract)

    @classmethod
    def with_backend(cls, backend: PolicyBackend, *, obs_dim: int, action_dim: int,
                     action_clip: float | None, contract: DeployContract | None = None) -> "PolicyCore":
        """체크포인트 없이 임의 backend(fake 포함)로 코어를 만든다 — 테스트·오프라인 체인용."""
        core = cls.__new__(cls)
        core._setup(backend, obs_dim, action_dim, action_clip, contract)
        return core

    def _setup(self, backend: PolicyBackend, obs_dim: int, action_dim: int,
               action_clip: float | None, contract: DeployContract | None) -> None:
        if obs_dim <= 0 or action_dim <= 0:
            raise PolicyIOError(f"obs_dim/action_dim must be positive, got {obs_dim}/{action_dim}")
        if action_clip is not None and not action_clip > 0.0:
            raise PolicyIOError(f"action_clip must be positive or None, got {action_clip!r}")
        self._backend = backend
        self._obs_dim = int(obs_dim)
        self._action_dim = int(action_dim)
        self._clip = action_clip
        self._contract = contract
        self._state = SeqState()

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_clip(self) -> float | None:
        return self._clip

    @property
    def contract(self) -> DeployContract | None:
        return self._contract

    @property
    def state(self) -> SeqState:
        return self._state

    def reset(self) -> None:
        """hidden 과 seq 카운터를 지운다. 다음 obs 는 seq 0 이어야 한다."""
        self._backend.reset()
        self._state = SeqState()

    def act(self, obs: np.ndarray, seq: int) -> np.ndarray:
        """obs(obs_dim,) + seq → 클립된 액션(action_dim,) float32 새 배열."""
        obs32 = check_obs(obs, self._obs_dim)
        self._state, raw = apply_seq_rule(self._state, seq, obs32, self._backend.forward,
                                          self._backend.reset, action_dim=self._action_dim)
        return clip_action(raw, self._clip)
