"""골든(3) — 배포 FabricCore 가 Isaac 학습 기록과 **같은 관절 해**를 내는가 (gpu).

좌: `left_v2B25_end.npz` env 0 의 palm 지령을 같은 초기상태에서 같은 순서로 적분해
    `fabric_q` 와 TCP 공간에서 대조한다(`probes/probe_fabric_deploy_parity.py` 이식, 1.0 mm).
우: `g1_y00.hdf5` 의 `palm_cmd` 를 g1 홈에서 재생하고 손 슬롯은 `hand_q_cmd` 로 동기화해
    `arm_q_cmd` 와 대조한다. decimation {2,1} × 테이블 world {on,off} 네 조합을 전부 찍고
    **계약 조합(2, on)** 만 1.0 mm 로 단정한다 — 문턱을 늦추지 않는다.
"""
from __future__ import annotations

import dataclasses
import math
import multiprocessing
import re
from pathlib import Path

import numpy as np
import pytest

from policy_control import contract as C
from policy_control import fabric_core as F

pytestmark = [pytest.mark.gpu, pytest.mark.golden]

SIM2REAL = Path(__file__).resolve().parents[2]
LEFT_JSON = SIM2REAL / "logs/policy/left_v2B25/deploy_contract.json"
RIGHT_JSON = SIM2REAL / "logs/policy/right_g1/deploy_contract.json"
TOL_MM = 1.0
#: 우 g1: 정상상태 ≤0.1 mm(중앙값) + 과도구간(가속·손 과지령 60~135 스텝) ≤5 mm — run-to-run 비결정
#: (float32·warp)으로 max 가 2.6~4.2 mm 로 흔들려 1.0 mm 단일 문턱은 의미 불일치와 수치 민감도를 못 가른다.
RIGHT_STEADY_MEDIAN_MM = 0.1
RIGHT_TRANSIENT_MAX_MM = 5.0
LEFT_STEPS = 300
DEVICE = "cuda:0"


def _cuda_or_skip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음 — fabrics_sim 은 cuda 전용")


def _env_yaml_scalar(path: Path, key: str) -> str:
    m = re.search(rf"^\s*{key}:\s*(.+?)\s*$", path.read_text(), re.M)
    if m is None:
        raise KeyError(f"{path} 에 {key} 가 없다")
    return m.group(1)


def _euler_zyx_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """wxyz → (ez, ey, ex) with R = Rz(ez)·Ry(ey)·Rx(ex) (fabric `euler_zyx` 규약)."""
    w, x, y, z = (float(v) for v in q)
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + z * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    return np.array([math.atan2(r10, r00), -math.asin(max(-1.0, min(1.0, r20))),
                     math.atan2(r21, r22)])


def _tcp_mm(core: F.FabricCore, q_a: np.ndarray, q_b: np.ndarray) -> float:
    return float(np.linalg.norm(core.palm_pose(q_a)[:3] - core.palm_pose(q_b)[:3]) * 1000.0)


def _isolated(fn, *args):
    """★fabric 하나 = 프로세스 하나. fabrics_sim 은 프로세스 전역 warp 상태를 써서 한 프로세스에
    fabric 을 둘 이상 세우면(7관절 뒤 27관절, 또는 해제 뒤 재생성) illegal memory access 로
    죽는다(09.05 실측, 비결정적). 자식(spawn)에서 돌리고 numpy 결과만 받는다."""
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        return pool.apply(fn, args)


# ---------------------------------------------------------------- right
def _right_variants(contract: C.DeployContract):
    for dec in (2, 1):
        for table in (True, False):
            fab = dataclasses.replace(contract.fabric, decimation=dec,
                                      world={**contract.fabric.world, "table_obstacle": table})
            yield (dec, table), dataclasses.replace(contract, fabric=fab)


def _replay_right(contract, extras, ep, hand_perm, n):
    core = F.FabricCore(contract, DEVICE, extras=extras)
    palm = ep["palm_cmd"][:n]
    hand_cmd = ep["hand_q_cmd"][:n]
    arm_cmd = ep["arm_q_cmd"][:n]
    tcp = np.zeros(n)
    for t in range(n):
        out = core.step(palm[t], hand_target=hand_cmd[t][hand_perm])
        q_ref = np.concatenate([arm_cmd[t], out.q_full[7:]])
        tcp[t] = _tcp_mm(core, out.q_full, q_ref)
    return tcp


@pytest.mark.skipif(not RIGHT_JSON.exists(), reason="right_g1 contract 없음")
def test_right_g1_fabric_parity_four_combinations(fixtures_dir):
    _cuda_or_skip()
    h5py = pytest.importorskip("h5py")
    from grasp_s2r_fabric import permutation

    contract = C.load_contract(RIGHT_JSON)
    env_yaml = fixtures_dir / "runs/right_g1/params/env.yaml"
    if not env_yaml.exists():
        env_yaml = fixtures_dir / "runs/right_g1/env.yaml"
    extras = F.FabricExtras(
        table_z=float(_env_yaml_scalar(env_yaml, "table_surface_z")),
        use_hand_repulsion=_env_yaml_scalar(env_yaml, "use_hand_repulsion") == "true",
        use_body_repulsion_pairs=_env_yaml_scalar(env_yaml, "use_body_repulsion_pairs") == "true",
    )
    with h5py.File(fixtures_dir / "g1_y00.hdf5", "r") as f:
        assert int(f.attrs["decimation"]) == contract.fabric.decimation
        assert float(f.attrs["dt"]) == pytest.approx(contract.fabric.dt)
        dof_names = [str(s) for s in f.attrs["hand_joint_names"]]
        g = f["episodes/ep_000"]
        ep = {k: np.asarray(g[k], dtype=np.float64) for k in ("palm_cmd", "hand_q_cmd", "arm_q_cmd")}
        np.testing.assert_allclose(np.asarray(g["arm_q"][0]), contract.fabric.home_q[:7], atol=1e-6)
    # hand_q_cmd 는 DOF 순 → contract.action.hand.joints(프로필 순) — 이름으로
    hand_perm = permutation(dof_names, contract.action.hand.joints)
    n = len(ep["palm_cmd"])

    results = {}
    for key, variant in _right_variants(contract):
        tcp = _isolated(_replay_right, variant, extras, ep, hand_perm, n)
        results[key] = tcp
        print(f"\n[right g1] decimation {key[0]} · table {'on ' if key[1] else 'off'} · "
              f"{n} steps · TCP mean {tcp.mean():.3f} p95 {np.percentile(tcp, 95):.3f} "
              f"max {tcp.max():.3f} mm (worst {int(tcp.argmax())})")
    want = (contract.fabric.decimation, bool(contract.fabric.world["table_obstacle"]))
    assert want == (2, True)
    worst = results[want].max()
    best = min(results, key=lambda k: results[k].max())
    assert best == want, f"계약 조합 {want} 보다 {best} 가 낫다: " + ", ".join(
        f"{k}: {v.max():.3f}" for k, v in results.items())
    median = float(np.median(results[want]))
    assert median <= RIGHT_STEADY_MEDIAN_MM, f"계약 조합 {want} 정상상태 중앙값 {median:.3f} mm > {RIGHT_STEADY_MEDIAN_MM}"
    assert worst <= RIGHT_TRANSIENT_MAX_MM, (
        f"계약 조합 {want} 가 {worst:.3f} mm > {RIGHT_TRANSIENT_MAX_MM} — 전 조합: "
        + ", ".join(f"{k}: {v.max():.3f}" for k, v in results.items()))


# ---------------------------------------------------------------- left
@pytest.mark.skipif(not LEFT_JSON.exists(), reason="left_v2B25 contract 없음")
def test_left_v2b25_fabric_parity(fixtures_dir):
    _cuda_or_skip()
    sim = np.load(fixtures_dir / "left_v2B25_end.npz", allow_pickle=False)
    contract = C.load_contract(LEFT_JSON)
    # 기록 메타 = 기록 당시 fabric 파라미터. 계약과 어긋나면 대조 자체가 무의미하다.
    assert float(sim["meta_fabric_dt"][0]) == pytest.approx(contract.fabric.dt)
    assert int(sim["meta_fabric_decimation"][0]) == contract.fabric.decimation
    assert float(sim["meta_fabric_damping"][0]) == pytest.approx(contract.fabric.damping)
    assert str(sim["meta_fabric_robot_dir"][0]) == contract.fabric.robot_dir
    assert str(sim["meta_fabric_world"][0]) == contract.fabric.world["filename"]
    home = np.asarray(sim["meta_home_q"], dtype=np.float64)   # ★기록의 홈(cspace rest)
    n = min(LEFT_STEPS, len(sim["palm_cmd_pos"]))
    tcp = _isolated(_replay_left, contract, home, sim["palm_cmd_pos"][:n, 0],
                    sim["palm_cmd_quat_wxyz"][:n, 0], sim["fabric_q"][:n, 0])
    print(f"\n[left v2B25] {n} steps · TCP mean {tcp.mean():.3f} p95 "
          f"{np.percentile(tcp, 95):.3f} max {tcp.max():.3f} mm (worst step {int(tcp.argmax())})")
    assert tcp.max() <= TOL_MM, f"좌 fabric 해가 갈린다: max {tcp.max():.3f} mm > {TOL_MM}"


def _replay_left(contract, home, pos, quat, isaac_q):
    core = F.FabricCore(contract, DEVICE, home_q=home)
    tcp = np.zeros(len(pos))
    for t in range(len(pos)):
        palm6 = np.concatenate([pos[t], _euler_zyx_from_quat_wxyz(quat[t])])
        out = core.step(palm6)
        tcp[t] = _tcp_mm(core, out.q_arm, isaac_q[t])
    return tcp


# ---------------------------------------------------------------- dg5f-m asset (09.05 line-up)
ASSET_JSON = SIM2REAL / "logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json"
DG5FM_JSON = SIM2REAL / "logs/policy/right_g1/deploy_contract.dg5f-m.json"
FABRIC_URDF_ARM = [f"openarm_right_joint{i}" for i in range(1, 8)]
CONTROL_STEPS = 30
CONTROL_TARGET_DX = 0.05
CONTROL_MIN_PROGRESS_M = 0.03
MIRROR_ATOL_M = 1e-4
#: 우 g1 을 dg5f-m 자산(openarm_dg5f-m_bi_right + dg5f-m params) 위에서 재생 — 09.06 실측(598 스텝):
#:   dg5f-m: TCP 중앙값 0.044 · p95 0.74 · max 3.30 mm · |Δq| max 0.010 rad
#:   옛 자산(같은 세션): 중앙값 0.050 · p95 1.34 · max 3.54 mm · |Δq| max 0.026 rad
#: 자산이 바뀌어도 g1 기록을 옛 자산만큼(정상상태 0.05 mm 급, 과도 3~4 mm 급 = run-to-run 비결정 폭) 재현하므로
#: 옛 자산과 **같은 2단 문턱**(RIGHT_STEADY_MEDIAN_MM · RIGHT_TRANSIENT_MAX_MM)을 그대로 쓴다.


def _g1_episode(fixtures_dir: Path, contract: C.DeployContract):
    h5py = pytest.importorskip("h5py")
    from grasp_s2r_fabric import permutation

    with h5py.File(fixtures_dir / "g1_y00.hdf5", "r") as f:
        assert int(f.attrs["decimation"]) == contract.fabric.decimation
        assert float(f.attrs["dt"]) == pytest.approx(contract.fabric.dt)
        dof_names = [str(s) for s in f.attrs["hand_joint_names"]]
        g = f["episodes/ep_000"]
        ep = {k: np.asarray(g[k], dtype=np.float64) for k in ("palm_cmd", "hand_q_cmd", "arm_q_cmd")}
        np.testing.assert_allclose(np.asarray(g["arm_q"][0]), contract.fabric.home_q[:7], atol=1e-6)
    return ep, permutation(dof_names, contract.side("right").hand.joints), len(ep["palm_cmd"])


def _replay_right_dq(contract, ep, hand_perm, n):
    """TCP 오차 [mm] 와 팔 관절 오차 [rad] — 계약 side 'right' 로 짓는다(dg5f-m 계약은 sides 가 진실)."""
    core = F.FabricCore(contract, DEVICE, side="right")
    tcp, dq = np.zeros(n), np.zeros(n)
    for t in range(n):
        out = core.step(ep["palm_cmd"][t], hand_target=ep["hand_q_cmd"][t][hand_perm])
        q_ref = np.concatenate([ep["arm_q_cmd"][t], out.q_full[7:]])
        tcp[t] = _tcp_mm(core, out.q_full, q_ref)
        dq[t] = float(np.abs(out.q_full[:7] - ep["arm_q_cmd"][t]).max())
    return tcp, dq, list(core.backend.fabric.get_joint_names())


@pytest.mark.skipif(not DG5FM_JSON.exists(), reason="right_g1 dg5f-m contract 없음")
def test_right_g1_parity_on_dg5fm_asset(fixtures_dir):
    _cuda_or_skip()
    contract = C.load_contract(DG5FM_JSON)
    s = contract.side("right")
    assert contract.asset.name == "openarm_dg5f-m_bi_rl" and s.fabric.robot_dir == "openarm_dg5f-m_bi_right"
    assert s.fabric.table_z == pytest.approx(0.2) and s.fabric.use_body_repulsion_pairs   # 계약이 extras 를 든다
    ep, hand_perm, n = _g1_episode(fixtures_dir, contract)
    tcp, dq, names = _isolated(_replay_right_dq, contract, ep, hand_perm, n)
    median, worst = float(np.median(tcp)), float(tcp.max())
    print(f"\n[right g1 on dg5f-m] {n} steps · TCP mean {tcp.mean():.3f} median {median:.3f} p95 "
          f"{np.percentile(tcp, 95):.3f} max {worst:.3f} mm (worst {int(tcp.argmax())}) · |Δq| max {dq.max():.4f} "
          f"median {np.median(dq):.5f} rad")
    assert names[:7] == FABRIC_URDF_ARM
    assert median <= RIGHT_STEADY_MEDIAN_MM, f"정상상태 중앙값 {median:.3f} mm > {RIGHT_STEADY_MEDIAN_MM}"
    assert worst <= RIGHT_TRANSIENT_MAX_MM, f"과도 최대 {worst:.3f} mm > {RIGHT_TRANSIENT_MAX_MM}"


def _control_only_side(side: str) -> dict:
    """자산 계약(control-only)에서 한 팔 fabric 을 세우고 홈 palm 에서 +5 cm x 로 30 스텝 적분한다."""
    contract = C.load_contract(ASSET_JSON)
    core = F.FabricCore(contract, DEVICE, side=side)
    home = np.asarray(core.cfg.home_q)
    hand = home[core.n_arm:].copy()
    palm0 = core.palm_pose(home)
    target = palm0.copy()
    target[0] += CONTROL_TARGET_DX
    out = None
    for _ in range(CONTROL_STEPS):
        out = core.step(target, hand_target=hand)
    return {"palm0": palm0, "palm": core.palm_pose(out.q_full), "target": target, "tips": core.tips(out.q_full),
            "names": list(core.backend.fabric.get_joint_names()), "joint_names": core.joint_names}


@pytest.mark.skipif(not ASSET_JSON.exists(), reason="asset_openarm_dg5f-m_bi_rl contract 없음")
def test_dg5fm_control_only_sides_reach_target_and_mirror_at_home():
    _cuda_or_skip()
    r = {side: _isolated(_control_only_side, side) for side in ("left", "right")}
    for side, d in r.items():
        moved = float(d["palm"][0] - d["palm0"][0])
        err0 = float(np.linalg.norm(d["target"][:3] - d["palm0"][:3]))
        err = float(np.linalg.norm(d["target"][:3] - d["palm"][:3]))
        print(f"\n[{side} control-only] palm0 {np.round(d['palm0'], 4)} → +{CONTROL_STEPS} steps "
              f"{np.round(d['palm'][:3], 4)} · moved x {moved * 1e3:.1f} mm · err {err0 * 1e3:.1f} → {err * 1e3:.1f} mm")
        assert d["names"][:7] == FABRIC_URDF_ARM and all(j.startswith(side[0] + "_") for j in d["joint_names"])
        assert d["tips"].shape == (5, 3)
        assert moved >= CONTROL_MIN_PROGRESS_M, f"{side}: {moved * 1e3:.1f} mm < {CONTROL_MIN_PROGRESS_M * 1e3:.0f} mm"
        assert err < err0
    left, right = r["left"]["palm0"], r["right"]["palm0"]
    np.testing.assert_allclose(left[[0, 2]], right[[0, 2]], atol=MIRROR_ATOL_M)     # x·z 같고
    np.testing.assert_allclose(left[1], -right[1], atol=MIRROR_ATOL_M)              # y 는 거울
    assert abs(left[1]) > 0.1                                                        # 팔이 몸통 옆에 있다(0 이 아닌 거울)
    np.testing.assert_allclose(left[3], -right[3], atol=1e-4)                        # ez 부호 반대
