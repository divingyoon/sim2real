"""fk_numpy.UrdfChainFK — 자산 URDF(canonical 이름)로 푼 palm/손끝 자세가 같은 팔의 fabric URDF 와 같다.

두 URDF 는 같은 기구를 다른 이름으로 적은 것이다(자산: `r_aj_1`·`r_hl_palm`·`r_hl_thumb_tip`,
fabric: `openarm_right_joint1`·`palm_link`·`rl_dg_1_tip` — 좌팔 fabric URDF 도 우측 이름을 쓴다).
canonical→fabric 이름 사상으로 같은 체인을 두 파일에서 풀어 ≤1e-6 m 로 대조한다. fabric URDF 가
팔 뿌리 z 를 0.68999977, 손 마운트 y 를 ∓3e-7 로 적어 좌팔에서 3e-7 m 가 남는다(실측, 문턱 안).

gpu: 같은 자산 대 fabrics_sim 의 자체 FK(`get_palm_pose`/`get_fingertip_positions`, float32 warp).
"""
from __future__ import annotations

import multiprocessing
from pathlib import Path

import numpy as np
import pytest

from policy_control import contract as C
from policy_control import contract_assets as A
from policy_control import fk_numpy as K

pytestmark = pytest.mark.unit

SIM2REAL = Path(__file__).resolve().parents[2]
RL_WS = SIM2REAL.parent
ASSET = A.ASSETS["openarm_dg5f-m_bi_rl"]
FABRIC_URDF = RL_WS / "hdgp/source/FABRICS/src/fabrics_sim/models/robots/urdf"
RIGHT_RUN = SIM2REAL / "logs/policy/right_g1"
DG5FM_CONTRACT = RIGHT_RUN / "deploy_contract.dg5f-m.json"
SIDES = ("right", "left")
#: 자산 URDF ↔ fabric URDF: 두 파일이 같은 기구를 적었다는 문턱(m). 실측 우 1.8e-9 / 좌 3.0e-7.
URDF_PARITY_M = 1e-6
#: fabrics_sim FK(float32 warp) ↔ numpy float64. 실측(09.06, q=0·g1 홈·난수 4): 우 pos 1.3e-7 m / 회전 3.7e-7,
#: 좌 pos 4.5e-7 m / 회전 4.5e-7 — float32 반올림 수준. 문턱은 그 10배.
FABRIC_FK_POS_M = 5e-6
FABRIC_FK_ROT = 5e-6
N_RANDOM = 4
DEVICE = "cuda:0"

needs_asset = pytest.mark.skipif(not ASSET.manifest.exists(), reason="dg5f-m 자산 없음")
needs_g1 = pytest.mark.skipif(not (RIGHT_RUN / "nn").exists(), reason="right_g1 run dir 없음")

#: canonical → fabric URDF 이름 (두 fabric URDF 모두 우측 이름; 손가락 thumb=1 … pinky=5)
FAB_ARM = [f"openarm_right_joint{i}" for i in range(1, 8)]
FAB_HAND = [f"rj_dg_{f}_{i}" for f in range(1, 6) for i in range(1, 5)]
FAB_TIPS = [f"rl_dg_{f}_tip" for f in range(1, 6)]
FAB_PALM = "palm_link"


def _fabric_urdf(side: str) -> Path:
    return FABRIC_URDF / f"openarm_dg5f-m_bi_{side}" / f"openarm_dg5f-m_bi_{side}.urdf"


@pytest.fixture(scope="module")
def ctl() -> C.DeployContract:
    """양팔 제어 전용 계약 — 홈은 g1(우) + 미러(좌)."""
    if (RIGHT_RUN / "nn").exists():
        return A.build_asset_contract(home="run:logs/policy/right_g1")
    return A.build_asset_contract()


def _configs(side_cfg: C.SideCfg, seed: int = 0) -> list:
    """(이름, arm_q, hand_q): q=0, g1 홈(+open 손), 시드 난수 N_RANDOM 개."""
    rng = np.random.default_rng(seed)
    home_hand = np.array([side_cfg.home_hand[j] for j in side_cfg.hand_joints])
    out = [("zero", np.zeros(7), np.zeros(20)), ("home", np.array(side_cfg.home_arm), home_hand)]
    for k in range(N_RANDOM):
        out.append((f"rand{k}", rng.uniform(-1.2, 1.2, 7), rng.uniform(-0.4, 1.2, 20)))
    return out


def _asset_fk(side_cfg: C.SideCfg) -> K.UrdfChainFK:
    return K.UrdfChainFK.from_side(ASSET.urdf, side_cfg)


def _fabric_fk(side: str) -> K.UrdfChainFK:
    return K.UrdfChainFK(_fabric_urdf(side), FAB_ARM, FAB_HAND, FAB_PALM, FAB_TIPS)


# ------------------------------------------------------------------ CPU: asset URDF ↔ fabric URDF
@needs_asset
@pytest.mark.parametrize("side", SIDES)
def test_asset_urdf_matches_fabric_urdf_palm_and_tips(ctl, side):
    s = ctl.side(side)
    fk_a, fk_f = _asset_fk(s), _fabric_fk(side)
    assert fk_a.palm_body == f"{side[0]}_hl_palm" and fk_a.tip_names == tuple(s.tip_bodies)
    worst = {"pos": 0.0, "rot": 0.0, "tips": 0.0}
    for name, aq, hq in _configs(s):
        a, f = fk_a.palm_pose(aq, hq), fk_f.palm_pose(aq, hq)
        worst["pos"] = max(worst["pos"], float(np.abs(a.palm_pos - f.palm_pos).max()))
        worst["rot"] = max(worst["rot"], float(np.abs(a.extra["palm_rot"] - f.extra["palm_rot"]).max()))
        worst["tips"] = max(worst["tips"], float(np.abs(a.tips - f.tips).max()))
        assert a.tips.shape == (5, 3)
    print(f"[fk_urdf {side}] asset↔fabric URDF max |Δ| pos {worst['pos']:.2e} m rot {worst['rot']:.2e} "
          f"tips {worst['tips']:.2e} m")
    assert worst["pos"] <= URDF_PARITY_M and worst["tips"] <= URDF_PARITY_M, worst
    assert worst["rot"] <= URDF_PARITY_M, worst


@needs_asset
@pytest.mark.parametrize("side", SIDES)
def test_canonical_palm_is_the_alias_frame_and_the_kinematics_chain(ctl, side):
    """`{p}_hl_palm` ≡ `{p}_hl_palm_alias`(항등 고정 관절) 이고, 팔 구간은 robot_control 의 검증된 체인과 같다."""
    from robot_control.kinematics import chain_from_urdf

    s = ctl.side(side)
    fk = _asset_fk(s)
    chain = chain_from_urdf(ASSET.urdf.read_text(), list(s.arm_joints), s.palm_body)
    for _, aq, hq in _configs(s, seed=1):
        T = fk.tree.transforms([s.palm_body, f"{s.palm_body}_alias"], fk._q(aq, hq))
        np.testing.assert_allclose(T[s.palm_body], T[f"{s.palm_body}_alias"], atol=0.0, rtol=0.0)
        pose = fk.palm_pose(aq, hq)
        Tc = chain.pose(aq)
        np.testing.assert_allclose(pose.palm_pos, Tc[:3, 3], atol=1e-12)
        np.testing.assert_allclose(pose.extra["palm_rot"], Tc[:3, :3], atol=1e-12)
        R = pose.extra["palm_rot"]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


@needs_asset
def test_left_and_right_are_mirror_images_at_zero(ctl):
    """양팔 자산: q=0 에서 좌우 palm 은 y 만 부호가 다르다(자산 생성기의 미러 검증).
    자산 URDF 의 `r_hj_mount` 는 y 3e-7, `l_hj_mount` 는 0 이라 3e-7 m 가 남는다(생성기 반올림, 문턱 안)."""
    right = _asset_fk(ctl.side("right")).palm_pose(np.zeros(7), np.zeros(20))
    left = _asset_fk(ctl.side("left")).palm_pose(np.zeros(7), np.zeros(20))
    np.testing.assert_allclose(left.palm_pos, right.palm_pos * np.array([1.0, -1.0, 1.0]), atol=URDF_PARITY_M)
    np.testing.assert_allclose(left.tips[:, [0, 2]], right.tips[:, [0, 2]], atol=URDF_PARITY_M)
    np.testing.assert_allclose(left.tips[:, 1], -right.tips[:, 1], atol=URDF_PARITY_M)


# ------------------------------------------------------------------ CPU: validation
@needs_asset
def test_urdf_chain_refuses_uncovered_movable_joint_and_bad_names(ctl):
    s = ctl.side("right")
    with pytest.raises(K.FKError, match="0 으로 채우지 않는다"):
        K.UrdfChainFK(ASSET.urdf, s.arm_joints, s.hand_joints[1:], s.palm_body, s.tip_bodies)   # thumb_1 빠짐
    with pytest.raises(K.FKError, match="URDF 에 없다"):
        K.UrdfChainFK(ASSET.urdf, s.arm_joints, [*s.hand_joints[:-1], "r_hj_nope"], s.palm_body, s.tip_bodies)
    with pytest.raises(K.FKError, match="링크"):
        K.UrdfChainFK(ASSET.urdf, s.arm_joints, s.hand_joints, "r_hl_nope", s.tip_bodies)
    with pytest.raises(K.FKError, match="arm_joints"):
        K.UrdfChainFK(ASSET.urdf, list(reversed(s.arm_joints)), s.hand_joints, s.palm_body, s.tip_bodies)
    fk = _asset_fk(s)
    with pytest.raises(K.FKError):
        fk.palm_pose(np.zeros(6), np.zeros(20))
    with pytest.raises(K.FKError):
        fk.palm_pose(np.zeros(7), np.zeros(19))
    with pytest.raises(K.FKError, match="없다"):
        K.UrdfChainFK(ASSET.urdf / "missing.urdf", s.arm_joints, s.hand_joints, s.palm_body, s.tip_bodies)


def test_urdf_tree_handles_prismatic_and_mimic(tmp_path):
    """작은 합성 URDF: 회전 + 프리즈매틱 + mimic(원본 값으로 유도) + 고정 관절."""
    urdf = """<robot name="t">
      <link name="base"/><link name="a"/><link name="b"/><link name="c"/><link name="tip"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="a"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/></joint>
      <joint name="j2" type="prismatic"><parent link="a"/><child link="b"/>
        <origin xyz="0.2 0 0" rpy="0 0 0"/><axis xyz="1 0 0"/></joint>
      <joint name="j3" type="revolute"><parent link="b"/><child link="c"/>
        <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/><mimic joint="j1" multiplier="-1" offset="0"/></joint>
      <joint name="jf" type="fixed"><parent link="c"/><child link="tip"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint>
    </robot>"""
    p = tmp_path / "t.urdf"
    p.write_text(urdf)
    fk = K.UrdfChainFK(p, ["j1"], ["j2"], "a", ["tip"])
    pose = fk.palm_pose([np.pi / 2], [0.1])
    np.testing.assert_allclose(pose.palm_pos, [0.0, 0.0, 0.1], atol=1e-12)
    # j1=90° 회전 → x 축이 y 로; j2 0.1 → (0.3, 0, 0) 회전 → (0, 0.3, 0); j3 = −j1 (mimic) → 손끝 오프셋 z 0.05 는
    # y 축 회전 −90° 로 −x(로컬) → 월드 −y … 로컬 x 는 월드 y 이므로 (0, 0.3−0.05, 0.1)
    np.testing.assert_allclose(pose.tips[0], [0.0, 0.25, 0.1], atol=1e-12)
    assert fk.tree.movable_on("tip") == ["j1", "j2", "j3"]
    with pytest.raises(K.FKError, match="floating"):
        p.write_text(urdf.replace('type="prismatic"', 'type="floating"'))
        K.UrdfChainFK(p, ["j1"], ["j2"], "a", ["tip"])


# ------------------------------------------------------------------ CPU: factory
@needs_asset
def test_make_fk_urdf_chain_picks_the_side(ctl):
    for side in SIDES:
        fk = K.make_fk(ctl, RL_WS, side=side)
        assert isinstance(fk, K.UrdfChainFK)
        assert fk.palm_body == ctl.side(side).palm_body and fk.hand_joints == tuple(ctl.side(side).hand_joints)
    assert K.make_fk(ctl, RL_WS).palm_body == ctl.side(ctl.primary_side).palm_body
    with pytest.raises(K.FKError, match="no side"):
        K.make_fk(ctl, RL_WS, side="middle")
    with pytest.raises(K.FKError, match="모르는 fk.kind"):
        K.make_fk(ctl, RL_WS, kind="nope")


@needs_asset
@needs_g1
def test_make_fk_on_policy_contract_uses_side_palm_body():
    """dg5f-m 에 재기반한 g1 계약: fabric 어댑터도 urdf_chain 도 palm 바디는 side(`r_hl_palm`)를 쓴다."""
    c = C.load_contract(DG5FM_CONTRACT)
    fk = K.make_fk(c, RL_WS, kind="urdf_chain")
    assert isinstance(fk, K.UrdfChainFK) and fk.palm_body == "r_hl_palm"
    fab = K.make_fk(c, RL_WS, palm_pose_fn=lambda q: np.zeros(6), tips_fn=lambda q: np.zeros((5, 3)))
    assert isinstance(fab, K.FabricFK) and fab.palm_body == "r_hl_palm"
    assert fab.hand_joints == tuple(c.side("right").fabric.joint_order[7:])
    with pytest.raises(K.FKError, match="callable"):
        K.make_fk(c, RL_WS)


# ------------------------------------------------------------------ GPU: fabrics_sim FK
def _fabric_fk_worker(side: str, robot_dir: str, params: str, class_name: str, qs: np.ndarray) -> dict:
    """★fabric 하나 = 프로세스 하나(fabrics_sim 전역 warp 상태). numpy 결과만 돌려준다."""
    import torch

    import policy_control._paths  # noqa: F401  (FABRICS src on sys.path)
    from fabrics_sim.fabrics import openarm_tesollo_pose_fabric as M
    from fabrics_sim.utils.utils import initialize_warp

    initialize_warp(DEVICE[-1])
    fab = getattr(M, class_name)(1, DEVICE, 1.0 / 60.0, graph_capturable=False, use_hand_fabric=False,
                                 tip_per_finger=False, hand_mode="pca", robot_dir_name=robot_dir,
                                 robot_name=robot_dir, fabric_params_filename=params)
    palms, tips = [], []
    with torch.inference_mode():
        for q in qs:
            qt = torch.tensor(np.asarray(q, dtype=np.float32), device=DEVICE).unsqueeze(0)
            palms.append(fab.get_palm_pose(qt, "euler_zyx")[0].cpu().numpy().astype(np.float64))
            tips.append(fab.get_fingertip_positions(qt)[0].cpu().numpy().astype(np.float64))
    return {"palm6": np.stack(palms), "tips": np.stack(tips), "num_joints": int(fab.num_joints)}


def _isolated(fn, *args):
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        return pool.apply(fn, args)


@pytest.mark.gpu
@needs_asset
@pytest.mark.parametrize("side", SIDES)
def test_asset_urdf_fk_matches_fabrics_sim_fk(ctl, side):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA 없음 — fabrics_sim 은 cuda 전용")
    from grasp_s2r_core import _rot_euler_zyx

    s = ctl.side(side)
    f = s.fabric
    cfgs = _configs(s, seed=2)
    qs = np.stack([np.concatenate([aq, hq]) for _, aq, hq in cfgs])      # fabric 순 = 계약 joint_order(팔 + 손 프로필 순)
    got = _isolated(_fabric_fk_worker, side, f.robot_dir, f.params, f.class_name, qs)
    assert got["num_joints"] == len(f.joint_order) == 27
    fk = _asset_fk(s)
    worst = {"pos": 0.0, "rot": 0.0, "tips": 0.0}
    for k, (_, aq, hq) in enumerate(cfgs):
        mine = fk.palm_pose(aq, hq)
        worst["pos"] = max(worst["pos"], float(np.abs(mine.palm_pos - got["palm6"][k, :3]).max()))
        worst["rot"] = max(worst["rot"], float(np.abs(mine.extra["palm_rot"] - _rot_euler_zyx(got["palm6"][k, 3:])).max()))
        worst["tips"] = max(worst["tips"], float(np.abs(mine.tips - got["tips"][k]).max()))
    print(f"[fk_urdf gpu {side}] asset URDF numpy ↔ fabrics_sim max |Δ| pos {worst['pos']:.2e} m "
          f"rot {worst['rot']:.2e} tips {worst['tips']:.2e} m")
    assert worst["pos"] <= FABRIC_FK_POS_M and worst["tips"] <= FABRIC_FK_POS_M, worst
    assert worst["rot"] <= FABRIC_FK_ROT, worst
