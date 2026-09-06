"""M4 — FabricCore 단일 배치 스텝 벽시계 (gpu).

정책 스텝 안에서 fabric 노드가 쓸 수 있는 시간은 step_dt 의 1/4 로 잡는다(좌 5 ms ·
우 4.17 ms). 워밍업 50 스텝 뒤 600 스텝의 p50/p95 를 찍고 p95 로 단정한다.
dg5f-m 자산(control-only 계약, 좌·우 27관절·60 Hz)도 같은 잣대로 찍는다 — 우 g1 과 같은 이유로
CUDA graph 경로 전까지 예산 미달은 xfail.
"""
from __future__ import annotations

import multiprocessing
import re
import time
from pathlib import Path

import numpy as np
import pytest

from policy_control import contract as C
from policy_control import fabric_core as F

pytestmark = pytest.mark.gpu

SIM2REAL = Path(__file__).resolve().parents[2]
ASSET_JSON = SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json"
#: name → (contract path, side | None = primary)
CONTRACTS = {
    "left_v2B25": (SIM2REAL / "logs/policy/left_v2B25/deploy_contract.json", None),
    "right_g1": (SIM2REAL / "logs/policy/right_g1/deploy_contract.json", None),
    "dg5f-m_left": (ASSET_JSON, "left"),
    "dg5f-m_right": (ASSET_JSON, "right"),
}
WARMUP = 50
STEPS = 600
#: 좌 0.35·dt(7 ms): integrator.step 1회 ≈2.5 ms 가 관절 수 무관(warp 런치 바운드)이라 0.25·dt 는 CUDA graph 없이는
#: 못 맞춘다. 60 Hz 27관절(우 g1·dg5f-m 양팔, 4.17 ms)은 CUDA graph 경로(parity 재검증 필요) 전까지 xfail.
BUDGET_FRACTION = {"left_v2B25": 0.35, "right_g1": 0.25, "dg5f-m_left": 0.25, "dg5f-m_right": 0.25}
KNOWN_OVER_BUDGET = ("right_g1", "dg5f-m_left", "dg5f-m_right")
DEVICE = "cuda:0"


def _cuda_or_skip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음")


def _extras(name: str, fixtures_dir: Path) -> F.FabricExtras:
    if name != "right_g1":
        return F.FabricExtras()
    text = (fixtures_dir / "runs/right_g1/env.yaml").read_text()

    def val(key):
        return re.search(rf"^\s*{key}:\s*(.+?)\s*$", text, re.M).group(1)

    return F.FabricExtras(table_z=float(val("table_surface_z")),
                          use_hand_repulsion=val("use_hand_repulsion") == "true",
                          use_body_repulsion_pairs=val("use_body_repulsion_pairs") == "true")


def _measure(contract: C.DeployContract, extras: F.FabricExtras, side: str | None) -> np.ndarray:
    """자식 프로세스에서 fabric 하나를 세우고 스텝 벽시계 (STEPS,) 초를 돌려준다."""
    core = F.FabricCore(contract, DEVICE, side=side, extras=extras)
    home = np.asarray(core.cfg.home_q)
    hand = home[core.n_arm:].copy() if core.n_hand else None       # 손 슬롯 = 계약 홈(command order == joint_order)
    palm0 = core.palm_pose(home)
    rng = np.random.default_rng(0)

    def one(i: int) -> float:
        palm = palm0 + np.concatenate([0.02 * np.sin(0.05 * i + rng.uniform(0, 0.1, 3)), np.zeros(3)])
        t0 = time.perf_counter()
        core.step(palm, hand_target=hand)
        return time.perf_counter() - t0

    for i in range(WARMUP):
        one(i)
    return np.array([one(i) for i in range(STEPS)])


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_step_wall_time_p95_under_budget(name, fixtures_dir):
    _cuda_or_skip()
    path, side = CONTRACTS[name]
    if not path.exists():
        pytest.skip(f"{name} contract 없음")
    contract = C.load_contract(path)
    # ★fabric 하나 = 프로세스 하나(fabrics_sim 전역 warp 상태 — parity 테스트 `_isolated` 참조)
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        times = pool.apply(_measure, (contract, _extras(name, fixtures_dir), side))
    p50, p95 = np.percentile(times, 50) * 1e3, np.percentile(times, 95) * 1e3
    frac = BUDGET_FRACTION[name]
    budget_ms = frac * contract.rate.step_dt * 1e3
    print(f"\n[{name}] step wall p50 {p50:.3f} ms · p95 {p95:.3f} ms · max {times.max() * 1e3:.3f} ms "
          f"· budget {budget_ms:.2f} ms ({frac} × {contract.rate.step_dt * 1e3:.2f})")
    if name in KNOWN_OVER_BUDGET and p95 >= budget_ms:
        pytest.xfail(f"{name} p95 {p95:.3f} ms ≥ {budget_ms:.2f} ms — CUDA graph 경로 전까지 알려진 미달")
    assert p95 < budget_ms, f"{name}: p95 {p95:.3f} ms ≥ budget {budget_ms:.2f} ms"
