"""M1 — deploy contract: 로더/검증 + 런 dump 로부터의 생성(드리프트 가드).

원칙: 계약 JSON 의 모든 숫자는 런 dump(env.yaml/agent.yaml/nn) 또는 학습 소스 상수에서
읽어 온 값이어야 한다. 손으로 옮겨 적은 값은 0개 — 이 테스트가 기존 리더(`cfg_from_run`
계열)의 결과와 계약 값을 대조해 그 원칙을 잠근다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from policy_control import contract as C
from policy_control import contract_build as B

SIM2REAL = Path(__file__).resolve().parents[2]
LEFT_RUN = SIM2REAL / "logs/policy/left_v2B25"
RIGHT_RUN = SIM2REAL / "logs/policy/right_g1"
D3_RUN = SIM2REAL / "logs/policy/right_d3"
GAINS = SIM2REAL.parent / "urdf/vendor/openarm_description/config/arm/v10/control_gains.yaml"

needs_left = pytest.mark.skipif(not (LEFT_RUN / "nn").exists(), reason="left_v2B25 run dir 없음")
needs_right = pytest.mark.skipif(not (RIGHT_RUN / "nn").exists(), reason="right_g1 run dir 없음")
needs_d3 = pytest.mark.skipif(not (D3_RUN / "nn").exists(), reason="right_d3 run dir 없음")


@pytest.fixture(scope="module")
def left():
    return B.build_contract(LEFT_RUN, grasp_band="v1")


@pytest.fixture(scope="module")
def right():
    return B.build_contract(RIGHT_RUN)


# ---------------------------------------------------------------- family detection
@needs_left
def test_detects_left_gripper_family():
    assert B.detect_family(LEFT_RUN / "params/env.yaml") == "gripper_left"


@needs_right
def test_detects_grasp_s2r_family():
    assert B.detect_family(RIGHT_RUN / "params/env.yaml") == "grasp_s2r"


# ---------------------------------------------------------------- left v2B25
@needs_left
def test_left_rate_and_policy(left):
    from left_inference_dryrun import step_dt_from_run

    assert left.rate.step_dt == pytest.approx(step_dt_from_run(LEFT_RUN / "params/env.yaml"))
    assert left.rate.policy_hz == pytest.approx(50.0)
    assert left.rate.episode_steps == 250
    assert left.policy.obs_dim == 49 and left.policy.action_dim == 7
    assert left.policy.rnn is None
    assert left.policy.action_clip == pytest.approx(100.0)
    assert left.policy.normalize_input is True


@needs_left
def test_left_obs_segments_match_run(left):
    from left_obs_builder import segments_from_run

    want = segments_from_run(LEFT_RUN / "params/env.yaml")
    got = [(s.name, s.dim) for s in left.obs.segments]
    assert got == list(want)
    assert sum(d for _, d in got) == left.policy.obs_dim
    by = {s.name: s for s in left.obs.segments}
    assert by["joint_pos"].builder == "joint_pos_rel"
    assert by["palm_rot"].builder == "rot6d_rows"          # 좌 = 행우선 인터리브
    assert by["tcp_pos"].params["palm_box"] == [[0.22, 0.60], [0.10, 0.43], [0.16, 0.60]]
    # 기본자세 = 런 dump 의 홈 + 그리퍼 개방
    assert by["joint_pos"].params["default"][:7] == pytest.approx(left.pd.home_arm)
    assert by["joint_pos"].params["default"][7:] == pytest.approx([0.044, 0.044])


@needs_left
def test_left_home_and_goal_from_run(left):
    from left_inference_dryrun import goal_center_from_run
    from left_policy_core import home_from_run

    assert left.pd.home_arm == pytest.approx(home_from_run(LEFT_RUN / "params/env.yaml").tolist())
    goal = left.obs.segment("target_object_position").params["goal"]
    assert goal[:3] == pytest.approx(goal_center_from_run(LEFT_RUN / "params/env.yaml"))
    assert goal[3:] == [1.0, 0.0, 0.0, 0.0]


@needs_left
def test_left_action_matches_palm_reader(left):
    from gripper_left_palm_command import cfg_from_run

    cfg = cfg_from_run(LEFT_RUN / "params/env.yaml")
    p = left.action.palm
    assert p.convention == "absolute_palm"
    assert p.max_pose_angle == pytest.approx(cfg.max_pose_angle)
    assert p.box_lo == pytest.approx(list(cfg.box_lo)) and p.box_hi == pytest.approx(list(cfg.box_hi))
    assert p.euler_center == pytest.approx(list(cfg.euler_center))
    assert p.pos_rate_limit == pytest.approx(cfg.pos_rate_limit)
    assert p.rot_rate_limit == pytest.approx(cfg.rot_rate_limit)
    h = left.action.hand
    assert h.decoder == "binary_gripper"
    assert h.params["open"] == 0.044 and h.params["close"] == 0.0
    assert [g.name for g in left.action.groups] == ["palm", "gripper"]
    assert left.action.groups[1].slice == [6, 7]


@needs_left
def test_left_gate_params_from_run(left):
    from left_grasp_gate import CUP_BOTTOM_TO_ORIGIN
    from openarm.gripper.left.grasp_sensor import grasp_left_preset as P

    g = left.obs.segment("gripper_gate").params
    # ★v2B25 는 v1 대역(판 위 10~85 mm)으로 학습됐다 — 골든 스트림이 62 mm 에서 게이트가 열림을 증명.
    #   태스크 이름 규칙(band_axis_from_run → v2 대역 80~150)은 이 런에 틀리다.
    want = [P.GRASP_HEIGHT_BAND[0] - CUP_BOTTOM_TO_ORIGIN, P.GRASP_HEIGHT_BAND[1] - CUP_BOTTOM_TO_ORIGIN]
    assert g["band_axis"] == pytest.approx(want)
    assert "v1" in g["band_source"]
    with pytest.raises(SystemExit, match="grasp-band"):
        B.build_contract(LEFT_RUN)                      # 대역을 명시하지 않으면 거부
    assert g["release_lateral"] == 0.06 and g["pad_offset"] == 0.0319
    assert g["lateral_ok"] == 0.03 and g["along_ok"] == 0.03


@needs_left
def test_left_fabric_and_gravity(left):
    f = left.fabric
    assert f.class_name == "OpenArmGripperLeftPoseFabric"
    assert f.robot_dir == "openarm_tesollo_sensor_left_gripper"
    assert f.world == {"filename": "open_gripper_left_boxes_no_table"}
    assert f.dt == pytest.approx(0.02) and f.decimation == 2 and f.damping == 10.0
    assert f.vel_ff_scale == 1.0
    assert f.joint_order == [f"l_aj_{i}" for i in range(1, 8)]
    # ★fabric 홈(J147, 액션의 default_config) ≠ 로봇 리셋 홈(dump init_state): j4 0.9336 vs 0.5665
    from openarm.gripper.left.grasp_sensor import grasp_left_preset as P
    assert f.home_q == pytest.approx([P.LEFT_ARM_HOME_JOINT_POS[j] for j in f.joint_order])
    assert f.home_q[3] == pytest.approx(0.9336) and left.pd.home_arm[3] == pytest.approx(0.5665)
    assert "fabric default_config" in (f.home_source or "")
    assert left.pd.gravity.mode == "integral_droop"
    assert left.pd.gravity.gain == 0.05
    assert left.pd.gravity.limit == pytest.approx([0.1, 0.1, 0.0675, 0.0675, 0.0175, 0.0175, 0.0175])
    assert left.pd.sim_gains.kp == [70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0]
    assert left.pd.sim_gains.kd == [2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5]


# ---------------------------------------------------------------- right g1
@needs_right
def test_right_rate_policy_rnn(right):
    assert right.rate.policy_hz == pytest.approx(60.0)
    assert right.rate.episode_steps == 600
    assert right.policy.obs_dim == 155 and right.policy.action_dim == 21
    assert right.policy.rnn == {"type": "lstm", "units": 1024, "layers": 1, "before_mlp": True,
                                "layer_norm": True, "concat_input": True, "concat_output": True}
    assert right.policy.action_clip == 1.0 and right.policy.obs_clip == 5.0


@needs_right
def test_right_obs_segments_and_orders(right):
    from grasp_s2r_obs_builder import SEGMENTS, hand_dof_order, tip_body_order

    assert [(s.name, s.dim) for s in right.obs.segments] == list(SEGMENTS)
    assert right.obs.joint_orders["hand_obs"] == hand_dof_order("r")
    assert right.obs.joint_orders["tips"] == tip_body_order("r")
    assert right.obs.joint_orders["hand_profile"][:4] == ["r_hj_thumb_1", "r_hj_thumb_2", "r_hj_thumb_3", "r_hj_thumb_4"]
    assert right.obs.segment("palm_ax").builder == "rot6d_columns"   # 우 = 열 스택
    assert right.obs.segment("tip_force").params["contact_force_max"] == 10.0
    assert right.obs.segment("joint_err").params["joint_pos_err_max"] == 1.2
    assert right.obs.segment("goal_rel").params["goal_offset"] == [0.0, 0.0, 0.12]


@needs_right
def test_right_action_matches_readers(right):
    from grasp_s2r_palm_command import cfg_from_run as palm_cfg
    from grasp_s2r_synergy import cfg_from_run as syn_cfg

    pc = palm_cfg(RIGHT_RUN / "params/env.yaml")
    p = right.action.palm
    assert p.convention == "delta_anchor"
    assert p.delta_xyz == pytest.approx(list(pc.delta_xyz)) and p.delta_rot_deg == pc.delta_rot_deg
    assert p.anchor == {"mode": "spawn", "offset_xyz": list(pc.anchor_offset_xyz), "fab_to_env": [0.0, 0.0, 0.0]}
    assert p.pos_rate_limit == pc.rate_limit_m and p.rot_rate_limit_deg == pc.rate_limit_rot_deg
    sc = syn_cfg(RIGHT_RUN / "params/env.yaml")
    h = right.action.hand
    assert h.decoder == "synergy"
    assert h.params["close_speed"] == sc.close_speed and h.params["hand_layout"] == sc.hand_layout
    assert h.params["hold_mode"] == "contact"
    assert h.params["close_gate"] == {"enabled": True, "ramp": 0.5, "z_deadband": 0.03}
    assert len(h.params["open_pose"]) == 20 and len(h.params["grip_pose"]) == 20


@needs_right
def test_right_fabric_and_gravity(right):
    f = right.fabric
    assert f.class_name == "OpenArmTeoslloPoseFabric"
    assert f.robot_dir == "openarm_tesollo_sensor_right"
    assert f.params == "openarm_tesollo_sensor_pose_params.yaml"
    assert f.dt == pytest.approx(1 / 60) and f.decimation == 2 and f.damping == 10.0
    assert f.world == {"table_obstacle": True, "margin_xy": 0.1, "thickness": 0.05}
    assert f.hand_sync == "syn_target"
    assert f.table_z == 0.2 and f.use_hand_repulsion is False and f.use_body_repulsion_pairs is True
    lim = right.action.hand.params["soft_limits"]
    assert len(lim) == 20 and all(lo < hi for lo, hi in lim)
    assert right.pd.gravity.mode == "model_tau_ff"
    assert right.pd.sim_gains.kp == [70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0]
    assert right.pd.sim_gains.kd == pytest.approx([7.053, 4.182, 7.804, 6.531, 2.236, 0.58, 0.242])


# ---------------------------------------------------------------- roundtrip / validation / gains
@needs_left
def test_json_roundtrip_and_md5(left, tmp_path):
    path = tmp_path / "deploy_contract.json"
    C.save_contract(left, path)
    back = C.load_contract(path)
    assert back == left
    assert back.run.checkpoint_md5 == "272194299637fef6aa89c8e93161d1b6"
    C.verify_checkpoint(back, SIM2REAL)      # 파일과 md5 대조 — 어긋나면 ContractError


@needs_left
def test_dim_mismatch_is_rejected(left, tmp_path):
    raw = json.loads(json.dumps(C.to_dict(left)))
    raw["policy"]["obs_dim"] = 48
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(C.ContractError, match="obs"):
        C.load_contract(path)


def test_unknown_obs_term_is_refused(tmp_path):
    env = (SIM2REAL / "tests/fixtures/policy_control/runs/left_v2B25/env.yaml").read_text()
    env = env.replace("    cup_upright:\n", "    mystery_term:\n", 1)
    run = tmp_path / "run"
    (run / "params").mkdir(parents=True)
    (run / "params/env.yaml").write_text(env)
    (run / "params/agent.yaml").write_text(
        (SIM2REAL / "tests/fixtures/policy_control/runs/left_v2B25/agent.yaml").read_text())
    with pytest.raises(SystemExit, match="mystery_term"):
        B.build_contract(run, checkpoint=None, grasp_band="v1")


@needs_left
def test_gains_match_left(left):
    assert C.compare_gains(left, GAINS).ok


@needs_right
def test_right_g1_kd_is_refused_under_the_vendor_only_rule(right):
    """★2026-09-06 정책 전환: kp 뿐 아니라 **kd 도** 벤더값이어야 한다.

    g1 은 그 전에 학습돼 r2s 적합 kd(7.053/4.182/…)를 달고 있다. 09.06 이후 규칙에서
    이 체크포인트는 배포 불가다 — 재학습 대상이므로 의도된 실패다(d3 와 같은 부류).
    """
    rep = C.compare_gains(right, GAINS)
    assert not rep.ok
    assert all("kp" not in r for r in rep.reasons), "kp 는 이미 벤더값이었다"
    assert "trained kd 7.053 != driver kd 2.75" in rep.reasons[0]
    assert "impossible" in rep.kd_note              # 7.053 > MIT_KD_MAX 5.0
    with pytest.raises(C.GainMismatch):
        C.require_gains(right, GAINS)


@needs_d3
def test_gains_mismatch_d3_is_intended_failure():
    d3 = B.build_contract(D3_RUN)
    rep = C.compare_gains(d3, GAINS)
    assert not rep.ok and "r_aj_1" in rep.reasons[0]
    with pytest.raises(C.GainMismatch):
        C.require_gains(d3, GAINS)
