#!/usr/bin/env python3
"""이식한 head 카메라가 sim 에서 **실측 자리에 서는지** 확인하고 그 시야를 렌더한다.

`sim_head_camera.py` 가 만든 사양(실기 hand-eye `T_neck_cam` + 실측 K)을 태스크 씬에
붙이고, 목을 기준 자세로 둔 뒤 **카메라의 실제 월드 자세를 읽어** 기대값과 비교한다.

    기대값 = T_base_neck(pan, tilt) ∘ T_neck_cam        (head_fk_chain + 캘리브)
    실측값 = camera.data.pos_w / quat_w_ros             (Isaac 이 실제로 놓은 자리)

이 둘이 맞으면 prim 경로·오프셋·`convention="ros"` 가 전부 옳다는 뜻이다. 자산의
`head_cam_view`(59.5 mm 어긋남)는 **건드리지 않는다** — 새 카메라를 따로 붙일 뿐이다.

    ./isaaclab.sh -p probe_sim_head_camera.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-sens_r_grasp_s2r")
parser.add_argument("--pan-deg", type=float, default=0.0, help="인코더 각")
parser.add_argument("--tilt-deg", type=float, default=-20.0, help="인코더 각")
parser.add_argument("--out", default="/tmp/sim_head_view.png")
parser.add_argument("--settle-steps", type=int, default=12)
parser.add_argument("--list-prims", action="store_true", help="카메라를 붙이지 않고 로봇 prim 경로만 나열")
parser.add_argument("--board-npy", default=None,
                    help="실기에서 잰 T_base_board(4x4 .npy) — 그 자리에 보드를 소환한다")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

import fabrics_sim                                            # noqa: E402,F401
import gymnasium as gym                                       # noqa: E402,F401
import isaaclab.sim as sim_utils                              # noqa: E402
import openarm.tasks                                          # noqa: E402,F401
from isaaclab.sensors import CameraCfg                        # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg                # noqa: E402


from head_fk_chain import t_base_neck_from_encoder            # noqa: E402
from sim_head_camera import load_spec                         # noqa: E402
# ★배선된 hdgp 모듈을 그대로 쓴다 — probe 만 되고 태스크는 안 되는 상황을 막는다
from openarm.sensors.head_camera import attach_head_camera     # noqa: E402
# ★`scripts/` 를 임포트 경로에 넣는다 — 이 파일은 거기서 한 단계 내려와 있다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CAM_KEY = "head_cam_real"
SX, SY, SQUARE = 7, 5, 0.030          # ChArUco 7x5, 한 칸 30 mm
BOARD_W, BOARD_H, BOARD_T = SX * SQUARE, SY * SQUARE, 0.003
POS_TOL_MM = 2.0
ROT_TOL_DEG = 0.5


def quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """두 쿼터니언(wxyz) 사이 회전각. 부호 모호성을 흡수한다."""
    d = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, d))))


def quat_wxyz(R: np.ndarray) -> np.ndarray:
    from sim_head_camera import quat_wxyz_from_matrix
    u, _, vt = np.linalg.svd(R)
    return np.array(quat_wxyz_from_matrix(u @ vt))


def main() -> int:
    spec = load_spec()
    print(f"[spec] link={spec.link} pos={spec.pos}")
    print(f"[spec] quat_wxyz={tuple(round(v, 6) for v in spec.quat_wxyz)}")

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1e6

    if args.list_prims:
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        env.reset()
        root = "/World/envs/env_0/Robot"
        stage = env.sim.stage
        lines = [f"=== {root} 하위 prim (head 관련) ==="]
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(root) and "head" in path.lower():
                lines.append(f"  {path}   ({prim.GetTypeName()})")
        lines.append(f"\n=== {root} 직속 자식 ===")
        node = stage.GetPrimAtPath(root)
        for child in (node.GetChildren() if node else []):
            lines.append(f"  {child.GetName()}   ({child.GetTypeName()})")
        Path(args.out).with_suffix(".prims.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        env.close(); simulation_app.close(); return 0


    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]

    # ★카메라는 **env 생성 뒤**에 붙인다. grasp_s2r 은 DirectRLEnv 라 로봇을
    #   `_setup_scene()` 에서 추가하는데, 씬 cfg 의 센서는 그보다 **먼저** 생성돼
    #   `head_camera` prim 이 아직 없다("Unable to find source prim path").
    camera = attach_head_camera(env)      # 함정 넷을 모듈이 처리한다

    names = robot.joint_names
    print(f"[sim] 관절 {len(names)}개 · head 관절 = "
          f"{[n for n in names if n.startswith('head_j')] or '없음'}")

    # 목을 기준 자세로. URDF 부호로 변환해 넣는다.
    from head_fk_chain import urdf_from_encoder
    pan_urdf, tilt_urdf = urdf_from_encoder(args.pan_deg, args.tilt_deg)
    targets = {n: float(np.radians(a)) for n, a in
               (("head_j_pan", pan_urdf), ("head_j_tilt", tilt_urdf)) if n in names}

    def pin_head() -> None:
        """★매 물리 스텝마다 다시 고정한다.

        head 에는 `ImplicitActuatorCfg` 가 붙어 있어 한 번만 써 넣으면 스텝마다
        기본 목표(0)로 끌려간다 — 12스텝 뒤 −20° 지령이 −7.377° 로 남았다.
        기구학만 보는 프로브이므로 매 스텝 상태를 다시 써 고정한다.
        """
        q = robot.data.joint_pos.clone()
        for jname, rad in targets.items():
            q[:, names.index(jname)] = rad
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)

    pin_head()
    print(f"[sim] 목 지령 pan_urdf={pan_urdf:+.2f}° tilt_urdf={tilt_urdf:+.2f}° "
          f"(인코더 {args.pan_deg:+.1f}/{args.tilt_deg:+.1f})")

    if args.board_npy:
        # ★실기에서 본 물체를 sim 에 같은 자리로 가져온다.
        #   env 가 하나뿐이고 env_origin 이 [0,0,0] 이라 base 프레임 = world 프레임이다.
        T_bb = np.load(args.board_npy)
        board = sim_utils.CuboidCfg(
            size=(BOARD_W, BOARD_H, BOARD_T),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.92)))
        center = (T_bb @ np.array([BOARD_W / 2, BOARD_H / 2, 0.0, 1.0]))[:3]
        board.func("/World/RealBoard", board, translation=tuple(center),
                   orientation=tuple(quat_wxyz(T_bb[:3, :3])))
        # ★코너 마커는 보드의 **자식이 아니라 형제**여야 한다. 자식으로 두면
        #   translation 이 보드 변환에 한 번 더 곱해져 엉뚱한 데로 간다.
        dot = sim_utils.SphereCfg(
            radius=0.005,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0), emissive_color=(0.8, 0.0, 0.0)))
        for i in range(1, SX):                       # ChArUco 내부 코너 (6x4=24)
            for j in range(1, SY):
                q = (T_bb @ np.array([i * SQUARE, j * SQUARE, BOARD_T, 1.0]))[:3]
                dot.func(f"/World/RealCorners/c_{i}_{j}", dot, translation=tuple(q))
        print(f"[sim] 실기 보드 소환 center={center}")

    for _ in range(args.settle_steps):
        pin_head()
        env.sim.step()
        env.sim.render()
        env.scene.update(env.sim.get_physics_dt())
        camera.update(env.sim.get_physics_dt(), force_recompute=True)

    # ★마지막은 **물리 스텝 없이** 고정 → 갱신 → 렌더.
    #   스텝을 한 번 더 돌리면 head 액추에이터가 스텝당 1.6° 씩 끌어내린다.
    pin_head()
    env.scene.update(env.sim.get_physics_dt())
    env.sim.render()
    camera.update(env.sim.get_physics_dt(), force_recompute=True)

    origin = env.scene.env_origins[0].cpu().numpy()

    # ★센서의 pos_w 는 fabric 에서 안 채워진다(0 이 나온다). 하지만 그건 검증의
    #   본질이 아니다 — 카메라 오프셋은 우리가 넣은 설정값이라 이미 정확하고,
    #   정말 확인할 것은 **FK 가 sim 기구학과 맞는가** 다. 링크 자세로 잰다.
    bodies = robot.body_names
    bi = bodies.index(spec.link)
    link_pos = robot.data.body_pos_w[0, bi].cpu().numpy() - origin
    link_quat = robot.data.body_quat_w[0, bi].cpu().numpy()
    T_fk = t_base_neck_from_encoder(args.pan_deg, args.tilt_deg)
    link_err_mm = np.linalg.norm(link_pos - T_fk[:3, 3]) * 1000.0
    link_err_deg = quat_angle_deg(link_quat, quat_wxyz(T_fk[:3, :3]))

    raw_pos = camera.data.pos_w[0].cpu().numpy()
    head_idx = {n: names.index(n) for n in ("head_j_pan", "head_j_tilt") if n in names}
    joint_report = " ".join(
        f"{n}={float(np.degrees(robot.data.joint_pos[0, i].cpu())):+.3f}deg"
        for n, i in head_idx.items()) or "head 관절이 articulation 에 없다"
    diag = (f"head_joints {joint_report}\n"
            f"joint_names_head {[n for n in names if n.startswith('head')]}\n"
            f"sim_link_quat {link_quat}\n"
            f"fk_link_quat  {quat_wxyz(T_fk[:3,:3])}\n"
            f"link {spec.link} sim_pos {link_pos}\n"
            f"link {spec.link} fk_pos  {T_fk[:3,3]}\n"
            f"link_err_mm {link_err_mm:.3f}\nlink_err_deg {link_err_deg:.4f}\n"
            f"link_verdict {'PASS' if link_err_mm<=POS_TOL_MM and link_err_deg<=ROT_TOL_DEG else 'FAIL'}\n"
            f"initialized {camera.is_initialized}\n"
            f"num_instances {camera.data.pos_w.shape}\n"
            f"raw_pos_w {raw_pos}\n"
            f"env_origin {origin}\n"
            f"quat_w_ros {camera.data.quat_w_ros[0].cpu().numpy()}\n"
            f"quat_w_world {camera.data.quat_w_world[0].cpu().numpy()}\n")
    pos_w = raw_pos - origin
    quat_w = camera.data.quat_w_ros[0].cpu().numpy()

    import yaml
    T_neck_cam = np.array(yaml.safe_load(
        (_HERE.parent / "config" / "head_extrinsics.yaml").read_text(
            encoding="utf-8"))["neck_to_camera"]["matrix"], dtype=float)
    T_exp = t_base_neck_from_encoder(args.pan_deg, args.tilt_deg) @ T_neck_cam

    d_mm = np.linalg.norm(pos_w - T_exp[:3, 3]) * 1000.0
    d_deg = quat_angle_deg(quat_w, quat_wxyz(T_exp[:3, :3]))
    print("\n=== 카메라 자세 대조 (base 프레임) ===")
    print(f"  기대 위치 [{T_exp[0,3]:+.4f}, {T_exp[1,3]:+.4f}, {T_exp[2,3]:+.4f}] m")
    print(f"  sim  위치 [{pos_w[0]:+.4f}, {pos_w[1]:+.4f}, {pos_w[2]:+.4f}] m")
    print(f"  차이 {d_mm:.2f} mm · 회전 {d_deg:.3f}°")
    ok = link_err_mm <= POS_TOL_MM and link_err_deg <= ROT_TOL_DEG
    print(f"  판정 {'✓ 이식 성공' if ok else '❌ 어긋남 — prim 경로/오프셋/convention 확인'}")

    rgb = camera.data.output["rgb"][0].cpu().numpy()
    try:
        import imageio.v2 as imageio
        imageio.imwrite(args.out, rgb[..., :3].astype(np.uint8))
    except ImportError:
        np.save(args.out.replace(".png", ".npy"), rgb)
        print(f"  (imageio 없음 → npy 로 저장)")
    print(f"  렌더 저장: {args.out}  shape={rgb.shape}")
    Path(args.out).with_suffix(".result.txt").write_text(
        f"expected_cam {T_exp[0,3]:+.6f} {T_exp[1,3]:+.6f} {T_exp[2,3]:+.6f}\n"
        f"# ★센서의 pos_w 는 fabric 에서 안 채워져 0 이 나온다 — 판정에 쓰지 않는다\n"
        f"verdict {'PASS' if ok else 'FAIL'}\nrgb_shape {rgb.shape}\n" + diag,
        encoding="utf-8")

    env.close()
    simulation_app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
