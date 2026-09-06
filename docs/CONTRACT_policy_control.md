# policy_control 배포 계약(생성물 — 원본은 deploy_contract.json)

obs → policy → fabric → pd 4노드가 읽는 계약을 사람이 읽을 수 있게 펼친 것. 수정은 `tools/build_deploy_contract.py` 로 계약을 다시 만들고 이 문서를 재생성한다.

## open-grip_l_grasp_sensor_v2 — `logs/policy/left_v2B25`

- checkpoint `nn/v2B25_tip30_ep2150.pth` md5 `272194299637fef6aa89c8e93161d1b6` · experiment `v2B25_tip30_fresh`
- rate: policy 50 Hz (step_dt 0.02000 s) · episode 250 steps
- policy: obs 49 / action 7 · rnn none · mlp [256, 128, 64] · action_clip 100.0 · obs_clip 100.0

### obs segments
| # | name | dim [offset] | builder | params |
| --- | --- | --- | --- | --- |
| 0 | `joint_pos` | 9 [0:9] | joint_pos_rel |  |
| 1 | `joint_vel` | 9 [9:18] | joint_vel_rel |  |
| 2 | `object_position` | 3 [18:21] | object_pos_root |  |
| 3 | `target_object_position` | 7 [21:28] | goal_pose | goal=[0.375, 0.244, 0.447, 1.0, 0.0, 0.0, 0.0] |
| 4 | `actions` | 7 [28:35] | last_action |  |
| 5 | `gripper_gate` | 1 [35:36] | gripper_gate | band_axis=[-0.08209000000000001, -0.007089999999999999], band_source=grasp_left_preset.GRASP_HEIGHT_BAND (v1), pad_offse |
| 6 | `tcp_pos` | 3 [36:39] | tcp_pos_normalized | palm_box=[[0.22, 0.6], [0.1, 0.43], [0.16, 0.6]] |
| 7 | `palm_rot` | 6 [39:45] | rot6d_rows | body=l_hl_gripper_base |
| 8 | `goal_minus_cup` | 3 [45:48] | goal_minus_object |  |
| 9 | `cup_upright` | 1 [48:49] | object_upright |  |

joint orders: arm=7, ee=2
fk: {'kind': 'left_gripper', 'urdf': 'urdf/generated/rl/openarm_tesollo_sensor_rl.urdf'}

### action
| group | slice |
| --- | --- |
| palm | [0, 6] |
| gripper | [6, 7] |

- palm `absolute_palm` box [0.22, 0.1, 0.16]–[0.6, 0.43, 0.6] · pos rate 0.02 · euler_center [0.317093862, -1.4835298641951802, 3.094591725] · max_pose_angle 1.0471975511965976
- hand `binary_gripper` joints 1 · open=0.044, close=0.0, close_when=a<0, force_open_when_gate_closed=True

### fabric
- OpenArmGripperLeftPoseFabric · openarm_tesollo_sensor_left_gripper · openarm_gripper_left_pose_params.yaml · world {'filename': 'open_gripper_left_boxes_no_table'}
- dt 0.02000 × decimation 2 · damping 10.0 · vel_ff 1.0 · hand_sync None · table_z None · body_repulsion_pairs False

### pd
- groups ['left_arm', 'left_gripper'] · gravity `integral_droop` gain 0.05 limit [0.1, 0.1, 0.0675, 0.0675, 0.0175, 0.0175, 0.0175] · sim gravity disabled False
- trained gains kp [70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0] / kd [2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5]
- home arm [-0.0136, -0.3255, -0.001, 0.5665, -0.4655, 0.0088, -0.8304]

### sides (v2)
- asset run asset (training-time fabric URDF) · primary `left` · control_only False

| side | ee | hand joints | palm body | tips | pd groups | fabric | action groups | gravity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| left | gripper | 2 | `l_hl_gripper_base` | 0 | ['left_arm', 'left_gripper'] | OpenArmGripperLeftPoseFabric / openarm_tesollo_sensor_left_gripper | ['palm', 'gripper'] | integral_droop |

## open-sens_r_grasp_s2r-lstm — `logs/policy/right_g1`

- checkpoint `nn/g1_ep17000.pth` md5 `ccbfddc94f6b54414968232447351a18` · experiment `g1_rot20_fresh`
- rate: policy 60 Hz (step_dt 0.01667 s) · episode 600 steps
- policy: obs 155 / action 21 · rnn {'type': 'lstm', 'units': 1024, 'layers': 1, 'before_mlp': True, 'layer_norm': True, 'concat_input': True, 'concat_output': True} · mlp [512, 512] · action_clip 1.0 · obs_clip 5.0

### obs segments
| # | name | dim [offset] | builder | params |
| --- | --- | --- | --- | --- |
| 0 | `arm_q` | 7 [0:7] | joint_pos_abs | order=arm |
| 1 | `arm_qd` | 7 [7:14] | joint_vel_abs | order=arm |
| 2 | `hand_q` | 20 [14:34] | joint_pos_abs | order=hand_obs |
| 3 | `hand_qd` | 20 [34:54] | joint_vel_abs | order=hand_obs |
| 4 | `palm_pos` | 3 [54:57] | body_pos | body=palm |
| 5 | `palm_ax` | 6 [57:63] | rot6d_columns | body=palm |
| 6 | `tips_rel_palm` | 15 [63:78] | tips_rel_palm |  |
| 7 | `palm_to_obj` | 3 [78:81] | palm_to_object |  |
| 8 | `obj_to_tips` | 15 [81:96] | object_to_tips |  |
| 9 | `tip_force` | 15 [96:111] | tip_force_local | contact_force_max=10.0 |
| 10 | `joint_err` | 20 [111:131] | joint_err_norm | joint_pos_err_max=1.2, order=hand_profile, target_source=decoder_target |
| 11 | `actions` | 21 [131:152] | last_action |  |
| 12 | `goal_rel` | 3 [152:155] | goal_minus_object | goal_offset=[0.0, 0.0, 0.12] |

joint orders: arm=7, hand_obs=20, hand_profile=20, tips=5
fk: {'kind': 'fabric'}

### action
| group | slice |
| --- | --- |
| palm | [0, 6] |
| hand | [6, 21] |

- palm `delta_anchor` box [0.2, -0.55, 0.2]–[0.55, 0.22, 0.7] · pos rate 0.02 · delta [0.1, 0.1, 0.1]/20.0° · anchor {'mode': 'spawn', 'offset_xyz': [-0.066, -0.022, 0.085], 'fab_to_env': [0.0, 0.0, 0.0]}
- hand `synergy` joints 20 · close_speed=0.005, couple_four_fingers=True, residual_scale=0.0, hand_layout=coupled3, oppose_grip_delta_rad=-0.6, weak_finger=, weak_finger_curl_scale=1.0, freeze_scope=joint, release_deadband=0.0, blocked_err_thr_rad=0.3, blocked_limit_eps_rad=0.05, hold_mode=contact, contact_freeze=True, close_ga

### fabric
- OpenArmTeoslloPoseFabric · openarm_tesollo_sensor_right · openarm_tesollo_sensor_pose_params.yaml · world {'table_obstacle': True, 'margin_xy': 0.1, 'thickness': 0.05}
- dt 0.01667 × decimation 2 · damping 10.0 · vel_ff 1.0 · hand_sync syn_target · table_z 0.2 · body_repulsion_pairs True

### pd
- groups ['right_arm', 'right_hand'] · gravity `model_tau_ff` · sim gravity disabled True
- trained gains kp [70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0] / kd [7.053, 4.182, 7.804, 6.531, 2.236, 0.58, 0.242]
- home arm [0.038, 0.4012, 0.6015, 0.9643, 0.0294, 0.706, 0.4213]

### sides (v2)
- asset run asset (training-time fabric URDF) · primary `right` · control_only False

| side | ee | hand joints | palm body | tips | pd groups | fabric | action groups | gravity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| right | dg5f | 20 | `palm` | 5 | ['right_arm', 'right_hand'] | OpenArmTeoslloPoseFabric / openarm_tesollo_sensor_right | ['palm', 'hand'] | model_tau_ff |

## open-sens_r_grasp_s2r-lstm — `logs/policy/right_g1`

- checkpoint `nn/g1_ep17000.pth` md5 `ccbfddc94f6b54414968232447351a18` · experiment `g1_rot20_fresh`
- rate: policy 60 Hz (step_dt 0.01667 s) · episode 600 steps
- policy: obs 155 / action 21 · rnn {'type': 'lstm', 'units': 1024, 'layers': 1, 'before_mlp': True, 'layer_norm': True, 'concat_input': True, 'concat_output': True} · mlp [512, 512] · action_clip 1.0 · obs_clip 5.0

### obs segments
| # | name | dim [offset] | builder | params |
| --- | --- | --- | --- | --- |
| 0 | `arm_q` | 7 [0:7] | joint_pos_abs | order=arm |
| 1 | `arm_qd` | 7 [7:14] | joint_vel_abs | order=arm |
| 2 | `hand_q` | 20 [14:34] | joint_pos_abs | order=hand_obs |
| 3 | `hand_qd` | 20 [34:54] | joint_vel_abs | order=hand_obs |
| 4 | `palm_pos` | 3 [54:57] | body_pos | body=palm |
| 5 | `palm_ax` | 6 [57:63] | rot6d_columns | body=palm |
| 6 | `tips_rel_palm` | 15 [63:78] | tips_rel_palm |  |
| 7 | `palm_to_obj` | 3 [78:81] | palm_to_object |  |
| 8 | `obj_to_tips` | 15 [81:96] | object_to_tips |  |
| 9 | `tip_force` | 15 [96:111] | tip_force_local | contact_force_max=10.0 |
| 10 | `joint_err` | 20 [111:131] | joint_err_norm | joint_pos_err_max=1.2, order=hand_profile, target_source=decoder_target |
| 11 | `actions` | 21 [131:152] | last_action |  |
| 12 | `goal_rel` | 3 [152:155] | goal_minus_object | goal_offset=[0.0, 0.0, 0.12] |

joint orders: arm=7, hand_obs=20, hand_profile=20, tips=5
fk: {'kind': 'fabric', 'urdf': 'hdgp/assets/robot/openarm_dg5f-m_bi_rl/openarm_dg5f-m_bi_rl.urdf'}

### action
| group | slice |
| --- | --- |
| palm | [0, 6] |
| hand | [6, 21] |

- palm `delta_anchor` box [0.2, -0.55, 0.2]–[0.55, 0.22, 0.7] · pos rate 0.02 · delta [0.1, 0.1, 0.1]/20.0° · anchor {'mode': 'spawn', 'offset_xyz': [-0.066, -0.022, 0.085], 'fab_to_env': [0.0, 0.0, 0.0]}
- hand `synergy` joints 20 · close_speed=0.005, couple_four_fingers=True, residual_scale=0.0, hand_layout=coupled3, oppose_grip_delta_rad=-0.6, weak_finger=, weak_finger_curl_scale=1.0, freeze_scope=joint, release_deadband=0.0, blocked_err_thr_rad=0.3, blocked_limit_eps_rad=0.05, hold_mode=contact, contact_freeze=True, close_ga

### fabric
- OpenArmTeoslloPoseFabric · openarm_dg5f-m_bi_right · openarm_dg5f-m_right_pose_params.yaml · world {'table_obstacle': True, 'margin_xy': 0.1, 'thickness': 0.05}
- dt 0.01667 × decimation 2 · damping 10.0 · vel_ff 1.0 · hand_sync syn_target · table_z 0.2 · body_repulsion_pairs True

### pd
- groups ['right_arm', 'right_hand'] · gravity `model_tau_ff` · sim gravity disabled True
- trained gains kp [70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0] / kd [7.053, 4.182, 7.804, 6.531, 2.236, 0.58, 0.242]
- home arm [0.038, 0.4012, 0.6015, 0.9643, 0.0294, 0.706, 0.4213]

### sides (v2)
- asset `openarm_dg5f-m_bi_rl` (dg5f) urdf `hdgp/assets/robot/openarm_dg5f-m_bi_rl/openarm_dg5f-m_bi_rl.urdf` · primary `right` · control_only False

| side | ee | hand joints | palm body | tips | pd groups | fabric | action groups | gravity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| right | dg5f | 20 | `r_hl_palm` | 5 | ['right_arm', 'right_hand'] | OpenArmTeoslloPoseFabric / openarm_dg5f-m_bi_right | ['palm', 'hand'] | model_tau_ff |

## asset:openarm_dg5f-m_bi_rl — ``

- checkpoint `` md5 `` · experiment `control_only`
- rate: policy 60 Hz (step_dt 0.01667 s) · episode 0 steps
- policy: obs 0 / action 0 · rnn none · mlp [] · action_clip None · obs_clip None

### obs segments
| # | name | dim [offset] | builder | params |
| --- | --- | --- | --- | --- |

joint orders: arm=7, hand_profile=20, tips=5
fk: {'kind': 'urdf_chain', 'urdf': 'hdgp/assets/robot/openarm_dg5f-m_bi_rl/openarm_dg5f-m_bi_rl.urdf'}

### action
| group | slice |
| --- | --- |

- control-only contract: no policy, no action decoders (fabric takes palm_cmd / hand_cmd)

### fabric
- OpenArmTeoslloPoseFabric · openarm_dg5f-m_bi_right · openarm_dg5f-m_right_pose_params.yaml · world {'filename': 'open_tesollo_boxes_no_table'}
- dt 0.01667 × decimation 2 · damping 10.0 · vel_ff 1.0 · hand_sync syn_target · table_z None · body_repulsion_pairs True

### pd
- groups ['right_arm', 'right_hand'] · gravity `model_tau_ff` · sim gravity disabled False
- trained gains kp [70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0] / kd [2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5]
- home arm [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

### sides (v2)
- asset `openarm_dg5f-m_bi_rl` (dg5f) urdf `hdgp/assets/robot/openarm_dg5f-m_bi_rl/openarm_dg5f-m_bi_rl.urdf` · primary `right` · control_only True

| side | ee | hand joints | palm body | tips | pd groups | fabric | action groups | gravity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| left | dg5f | 20 | `l_hl_palm` | 5 | ['left_arm', 'left_hand'] | OpenArmTeoslloLeftPoseFabric / openarm_dg5f-m_bi_left | [] | model_tau_ff |
| right | dg5f | 20 | `r_hl_palm` | 5 | ['right_arm', 'right_hand'] | OpenArmTeoslloPoseFabric / openarm_dg5f-m_bi_right | [] | model_tau_ff |
