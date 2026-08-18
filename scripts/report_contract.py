#!/usr/bin/env python3
"""구성 프로필 → 배포 계약 매니페스트(Markdown) 생성.

값을 손으로 적으면 sim 이 바뀔 때 문서만 낡는다. 프로필·hdgp 소스에서 **읽어서** 만든다.

    python3 report_contract.py --robot tesollo_bi_s__right > docs/CONTRACT_grasp_v1_right.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import grasp_action_decoder as D  # noqa: E402
import grasp_obs_builder as O  # noqa: E402
from robot_profile import (  # noqa: E402
    expected_q_home_arm,
    load_hdgp_module,
    load_profile_env_cfg,
    load_robot_profile,
)


def render(robot: str) -> str:
    p = load_robot_profile(robot)
    cfg = load_profile_env_cfg(p)
    C = load_hdgp_module(p, "constants")
    L = []
    a = L.append
    a(f"# 배포 계약 — `{p.name}`\n")
    a("> 이 파일은 `scripts/report_contract.py` 가 구성 프로필과 hdgp 소스에서 **생성**한다.")
    a("> 손으로 고치지 말 것 — 값이 바뀌면 다시 생성하라.\n")

    a("## 로봇 구성\n")
    a("| 항목 | 값 |")
    a("|---|---|")
    a(f"| 작동 팔 | `{p.acting_side}` |")
    a(f"| 엔드이펙터 | `{p.ee_type}` ({p.ee_dof} DOF) |")
    a(f"| 자산 매니페스트 | `{p.manifest_path.name}` |")
    a(f"| Fabrics 자산 | `{p.fabrics.robot_dir}` / `{p.fabrics.class_name}` |")
    a(f"| Fabrics 월드 | `{p.fabrics.world}` |")
    a(f"| hdgp 패키지 | `{p.contract.hdgp_package}` |\n")

    a("## 토픽\n")
    a("| 역할 | 토픽 |")
    a("|---|---|")
    for k in ("arm_state", "ee_state", "tip_force_xyz", "tip_force_norm",
              "arm_cmd", "ee_cmd", "arm_traj", "ee_traj"):
        a(f"| `{k}` | `{p.topics[k]}` |")
    a("")

    a(f"## Observation ({O.ACTOR_OBS_DIM}D)\n")
    a("| # | 세그먼트 | 차원 | 슬라이스 |")
    a("|---|---|---|---|")
    for i, (name, dim) in enumerate(O.OBS_SEGMENTS):
        s = O.OBS_SLICES[name]
        a(f"| {i} | `{name}` | {dim} | `[{s.start}:{s.stop}]` |")
    a("")
    a(f"- `tip_force_local` = **tip-local 프레임 그대로** / {O.CONTACT_FORCE_MAX} N, clamp ±1")
    a(f"- `joint_pos_err` = (직전 전송 지령 − 실측) / {O.JOINT_POS_ERR_MAX} rad, **부호 보존**")
    a("- `last_actions` = 정책 원출력(4지 공통닫힘 **이전**)\n")

    a(f"## Action ({D.NUM_ACTIONS}D)\n")
    a(f"- `[{D.PALM_SLICE.start}:{D.PALM_SLICE.stop}]` palm delta (x, y, z, ez, ey, ex)")
    a(f"- `[{D.FINGER_SLICE.start}:{D.FINGER_SLICE.stop}]` 손가락 5×3 채널 "
      "— ch0=`_1` 외전 / ch1=`_2` MCP / ch2=`_3`·`_4` 공통")
    a("- 처리 순서: 4지 공통닫힘(**clamp 이전**) → 절대 폐쇄도 → `[ch0,ch1,ch2,ch2]` 전개 "
      "→ 변화율 상한 → lerp(APPROACH, FULL_GRIP) → 관절 한계 clamp\n")

    a("## 파라미터\n")
    a("| 키 | 값 | 비고 |")
    a("|---|---|---|")
    a(f"| `palm_delta_xyz` | {cfg['palm_delta_xyz']} | ★좌우 **동일** — 액션은 미러되지 않는다 |")
    a(f"| `palm_delta_rot_deg` | {cfg['palm_delta_rot_deg']} | 축별 ± |")
    a(f"| `reset_home_palm_pose` | {cfg['reset_home_palm_pose']} | x,y,z,ez°,ey°,ex° |")
    a(f"| `max_pose_angle` | {cfg['max_pose_angle']} | workspace 회전 여유 |")
    a(f"| `finger_close_speed` | {cfg['finger_close_speed']} | 변화율 상한/step |")
    a(f"| `couple_four_fingers` | {cfg['couple_four_fingers']} | 3지 국소최적 차단 |")
    a(f"| `retighten_after_latch` | {cfg['retighten_after_latch']} | True 면 배포도 바꿀 것 |")
    a(f"| `lift_wait_joint7_delta` | {cfg['lift_wait_joint7_delta']} | 좌우 부호 반대 |")
    a(f"| `warm_j7_min/max` | {cfg['warm_j7_min']} / {cfg['warm_j7_max']} | |")
    a(f"| `lift_start_min_grip_fingers` | {cfg['lift_start_min_grip_fingers']} | |")
    a(f"| `grasp_ready_hold_steps` | {cfg['grasp_ready_hold_steps']} | |")
    a(f"| `fabrics_dt` × `fabric_decimation` | {cfg['fabrics_dt']:.6f} × {cfg['fabric_decimation']} | |")
    a(f"| `fabrics_damping_gain` | {cfg['fabrics_damping_gain']} | 메인 |")
    a(f"| `reset_fabrics_damping_gain` | {cfg['reset_fabrics_damping_gain']} | 홈 IK 전용 |")
    a(f"| `CONTACT_FORCE_THRESHOLD` | {C.CONTACT_FORCE_THRESHOLD} | 실물 노이즈 위로 튜닝 |")
    a(f"| `EPISODE_STEPS` | {C.EPISODE_STEPS} | grasp {C.GRASP_PHASE_STEPS} + lift {C.LIFT_WAIT_PHASE_STEPS} |")
    a(f"| `PREGRASP_FABRICS_STEPS` | {C.PREGRASP_FABRICS_STEPS} | 홈 IK rollout |\n")

    a("## 홈 자세 기준값\n")
    q = expected_q_home_arm(p)
    a(f"`q_home` (sim preset 유도) = `{[round(v, 4) for v in q]}`\n")
    a("배포의 홈 IK 결과가 이 값과 0.05 rad 이상 다르면 **RuntimeError**. Fabrics 자산이")
    a("sim 과 다른지 먼저 의심할 것 — 구 자산은 palm 이 6.5cm 짧다.\n")

    a("## 배포–sim 의 의도된 차이\n")
    a("- **tip-only 게이트**: sim 은 손가락 동결에 (tip|mid|distal) 을 쓰지만 middle/distal 은")
    a("  critic 전용(privileged)이라 실기서 감지 불가하다. 라이브는 tip 접촉만 쓴다.")
    a("- **접촉 거리 게이트**: 실물 F/T 는 테이블 접촉도 잡는다(sim 접촉센서는 컵-필터).")
    a("  palm–컵 거리 0.10 m 밖에서는 접촉을 0 으로 만든다.\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="tesollo_bi_s__right")
    print(render(ap.parse_args().robot))


if __name__ == "__main__":
    main()
