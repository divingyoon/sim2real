#!/usr/bin/env python3
"""컨트롤러 자리에 서는 루프백 리그 — 실기 없이 **ROS 경로 전체**를 돌린다.

무엇을 대신하나. 로봇 PC 에서는 `openarm_bringup` 이 JointTrajectory / forward 명령을 받아 컨트롤러로
넘기고 `joint_state_broadcaster` 가 `/joint_states` 를 낸다. 이 노드는 그 두 역할만 흉내 낸다 —
지령을 받아 팔 모델(MockArm, 팔마다 하나)로 적분하고 양팔을 한 `/joint_states` 에 싣는다.

**왜 필요한가.** 새로 쓴 것은 재생기·브리지·pd 노드이고, 벤더 스택은 이미 검증된 코드다.
그런데 새 코드의 ROS 부분(발행·구독·리맵·타이머·게이트·csv)은 로봇 없이는 한 번도
돌지 않는다. 여기서 돌려 두면 실기에서 남는 미지수는 **로봇 자체**뿐이다.

두 모드:
  레거시 프로필  --robot gripper_left|tesollo_sensor__right (scripts/robot_profile, 구 자산 URDF) — 한 팔 + 유휴 팔 정적
  계약          --contract <deploy_contract.json> --robot-yaml <policy_control/config/robots/*.yaml> --sides right,left
                — 계약 sides 의 팔마다 MockArm, controller_manager 스텁 하나가 양팔 JTC+forward 3종을 안다(STRICT),
                  중력은 --pd-config 의 모델(pd 노드와 같은 식, 팔별 tip/payload) 또는 --gravity(자산 URDF 체인),
                  유효관성은 계약 홈(또는 --inertia-q) 자세의 자산 URDF. 선택하지 않은 팔은 홈에 정적으로 실린다.

⚠ **이것은 추종 검증이 아니다.** `--model rate` 는 속도제한만 있는 완벽 추종이라
`use_fake_hardware:=true` 와 같은 한계를 갖는다 — droop 이 없어 서보 버그를 숨긴다.
`--model pd` 는 실측 게인 PD 로 지연·오버슛을 **예측**하지만 그것도 모델이지 로봇이 아니다.
숫자를 실측으로 인용하지 말 것. 마찰(Fc)은 **우팔** r2s 캘리브 값이다(좌팔 식별 캘리브는 없다).

실행:
    source /opt/ros/humble/setup.bash && . .venv/bin/activate
    python3 scripts/fakes/fake_arm_bridge.py --robot gripper_left --model pd --forward --gravity
    python3 scripts/fakes/fake_arm_bridge.py --contract logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json \
        --robot-yaml policy_control/config/robots/dg5f_m_bi_fake.yaml --sides right,left \
        --pd-config policy_control/config/pd_dg5f_m_fake.yaml --rate-hz 100
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SIM2REAL = _HERE.parents[1]
for _p in (_SIM2REAL / "scripts", _HERE, _SIM2REAL / "policy_control"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from arm_pd_model import load_arm_pd  # noqa: E402
from fake_arm_side import (NUM_ARM, SideArm, SideSpec, driver_gains, gravity_from_pd_config,  # noqa: E402
                           gravity_from_urdf, inertia_at, side_spec_from_contract, side_spec_from_profile,
                           static_rows)
from fake_cm_stub import ControllerManagerStub, jtc_of  # noqa: E402,F401  (re-export: tests import it from here)

#: 실측 캘리브레이션. `right_arm_best_calibration.json` 은 **우팔** 값이다 —
#  좌팔에는 식별 캘리브가 없다(그 파일의 openarm_left_arm 400/80 은 sim 기본값이 남은 것).
#  같은 팔 하드웨어라 예측용으로 쓰되, 실측이라고 부르지 않는다.
CALIBRATION = Path.home() / "rl_ws/hdgp/log/logs/r2s_autotune/results/right_arm_best_calibration.json"
ASSET_URDF = Path.home() / "rl_ws/urdf/generated/rl/openarm_tesollo_sensor_rl.urdf"          # 레거시 프로필 모드
PROFILE_YAML = Path.home() / "rl_ws/robot_control/src/robot_control/profiles/openarm_tesollo.yaml"
DEFAULT_GAINS = Path.home() / "rl_ws/urdf/vendor/openarm_description/config/arm/v10/control_gains.yaml"
RL_WS = _SIM2REAL.parent
STATE_TOPIC = "/joint_states"
SIDE_ORDER = ("right", "left")


@dataclass(frozen=True)
class GripSpec:
    """그리퍼 행: JTC 토픽의 첫 점 위치 하나를 source 관절들에 싣는다."""
    topic: str
    source: tuple
    sign: tuple


@dataclass(frozen=True)
class PlantSpec:
    arms: tuple                 # SideArm (명령을 받는 팔)
    statics: tuple              # (SideSpec, q) — 정적으로 싣는 팔
    grip: GripSpec | None
    forward: bool               # forward 3종 + controller_manager 스텁
    with_effort: bool           # /joint_states effort 에 모델 토크(pd 모델)


def _legacy_tip(side: str) -> str:
    """★robot_control.profile.load_builtin_profile 은 자산 manifest(09.05 에 교체됨)를 요구해 죽는다.
    체인 tip 이름만 필요하므로 프로필 yaml 의 groups 를 직접 읽는다."""
    import yaml

    group = yaml.safe_load(PROFILE_YAML.read_text())["groups"][f"openarm_{side}_arm"]
    return group.get("asset_tip_link") or group["tip_link"]


def _inertia_q(text: str | None) -> np.ndarray | None:
    if text is None:
        return None
    q = np.array([float(v) for v in text.split(",")])
    if q.shape != (NUM_ARM,):
        raise SystemExit(f"--inertia-q 는 {NUM_ARM}값")
    return q


def _friction(args) -> np.ndarray:
    _, _, fc, _ = load_arm_pd(CALIBRATION)
    return np.asarray(fc, dtype=float) * float(args.friction_scale)


def _arm(spec: SideSpec, args, kp, kd, fc, inertia, gravity) -> SideArm:
    # ★substeps 를 줄이면 마찰 데드밴드가 부풀어 저관성 관절이 조용히 얼어붙는다(`MockArm` 이 거부한다). 기본값을 쓴다.
    return SideArm(spec, model=args.model, max_vel=args.max_vel, dt=1.0 / args.rate_hz, kp=kp, kd=kd, fc=fc,
                   inertia=inertia, gravity=gravity)


def build_legacy(args) -> PlantSpec:
    """--robot 프로필 모드: 한 팔 MockArm + 유휴 팔(hdgp preset rest) 정적 + EE 그리퍼 행."""
    from robot_profile import idle_arm_rest_pose, load_robot_profile

    profile = load_robot_profile(args.robot)
    side = profile.acting_side
    q_inertia = _inertia_q(args.inertia_q)
    spec = side_spec_from_profile(profile, side, profile.arm_canonical, np.zeros(NUM_ARM) if q_inertia is None else q_inertia)
    kp, kd, fc, dataset = load_arm_pd(CALIBRATION)
    if args.forward:
        # ★pd 노드와 **같은** kp/kd: 모터 MIT 루프가 쓰는 control_gains.yaml. r2s 캘리브의
        #   stiffness/damping 은 JTC 시대 등가 게인이라 여기서는 쓰지 않는다.
        kp, kd = driver_gains(args.gains)
    fc = np.asarray(fc, dtype=float) * float(args.friction_scale)
    gravity = gravity_from_urdf(ASSET_URDF, spec, _legacy_tip(side)) if args.gravity else None
    arm = _arm(spec, args, kp, kd, fc, inertia_at(ASSET_URDF, spec), gravity)
    idle_side = "left" if side == "right" else "right"
    idle_spec = side_spec_from_profile(profile, idle_side, profile.idle_arm_canonical, np.zeros(NUM_ARM))
    idle_q = np.asarray(idle_arm_rest_pose(profile)) + args.idle_arm_offset
    lim = profile.joint_limits
    grip = GripSpec(topic=profile.topics["ee_traj"], source=tuple(lim[c]["source"] for c in profile.ee_canonical),
                    sign=tuple(float(lim[c]["sign"]) for c in profile.ee_canonical))
    print(f"[fake_arm_bridge] 레거시 프로필 {profile.name} · 게인 출처 "
          f"{'control_gains.yaml' if args.forward else CALIBRATION.name + ' (dataset=' + str(dataset) + ')'} — ★우팔 식별값")
    return PlantSpec(arms=(arm,), statics=((idle_spec, idle_q),), grip=grip, forward=bool(args.forward),
                     with_effort=args.model == "pd")


def _contract_grip(robot_cfg, profile: dict, side: str) -> GripSpec | None:
    for name, g in robot_cfg.groups.items():
        if g.get("backend") == "jtc_single_point" and name.startswith(side):
            src = str(g["joint"])
            sign = next((float(v["sign"]) for v in profile.values() if v["source"] == src), 1.0)
            return GripSpec(topic=str(g["topic"]), source=(src,), sign=(sign,))
    return None


def build_contract(args) -> PlantSpec:
    """--contract 모드: 계약 sides 의 팔마다 MockArm(자산 URDF 관성·pd yaml 중력), 나머지 팔은 홈에 정적."""
    from policy_control.contract import load_contract
    from policy_control.sources import load_profile, load_robot_cfg

    contract = load_contract(Path(args.contract))
    if contract.asset is None:
        raise SystemExit("[fake_arm_bridge] 계약 모드는 asset 이 있는 계약(v2)이어야 한다")
    robot_cfg = load_robot_cfg(Path(args.robot_yaml))
    profile = load_profile(robot_cfg.joint_profiles)
    wanted = [s.strip() for s in (args.sides or "").split(",") if s.strip()] or list(contract.side_names)
    bad = [s for s in wanted if s not in contract.sides]
    if bad:
        raise SystemExit(f"[fake_arm_bridge] --sides {bad} 는 계약에 없다 (있는 팔 {contract.side_names})")
    sides = [s for s in SIDE_ORDER if s in wanted]
    urdf = RL_WS / contract.asset.urdf
    kp, kd = driver_gains(args.gains)
    fc = _friction(args)
    q_inertia = _inertia_q(args.inertia_q)
    arms, grips = [], []
    for side in sides:
        spec = side_spec_from_contract(contract, profile, side)
        if args.pd_config:
            gravity = gravity_from_pd_config(Path(args.pd_config), contract, side)
        else:
            gravity = gravity_from_urdf(urdf, spec, contract.side(side).palm_body) if args.gravity else None
        arms.append(_arm(spec, args, kp, kd, fc, inertia_at(urdf, spec, q_inertia), gravity))
        grips.append(_contract_grip(robot_cfg, profile, side))
    statics = tuple((side_spec_from_contract(contract, profile, s), np.asarray(contract.side(s).home_arm, dtype=float))
                    for s in contract.side_names if s not in sides)
    grip = next((g for g in grips if g is not None), None)
    print(f"[fake_arm_bridge] 계약 {contract.run.task} · 팔 {sides} · 정적 {[s.side for s, _ in statics]} · "
          f"중력 {'pd yaml ' + Path(args.pd_config).name if args.pd_config else ('urdf' if args.gravity else 'off')} · "
          f"관성 자세 {'--inertia-q' if q_inertia is not None else '계약 홈'} · 게인 control_gains.yaml")
    return PlantSpec(arms=tuple(arms), statics=statics, grip=grip, forward=True, with_effort=args.model == "pd")


# ================================================================== node
def make_node(spec: PlantSpec, rate_hz: float):
    import rclpy  # noqa: F401
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
    from trajectory_msgs.msg import JointTrajectory

    class FakeArmBridge(Node):
        def __init__(self) -> None:
            super().__init__("fake_arm_bridge")
            self.spec = spec
            self.grip = 0.0
            self.pub = self.create_publisher(JointState, STATE_TOPIC, 10)
            self.cm = ControllerManagerStub(self, tuple(a.spec.side for a in spec.arms)) if spec.forward else None
            for arm in spec.arms:
                self.create_subscription(JointTrajectory, arm.spec.jtc_topic, self._jtc_cb(arm), 10)
                if self.cm is not None:
                    for kind in ("position", "velocity", "effort"):
                        self.create_subscription(Float64MultiArray, self.cm.topic_of(arm.spec.side, kind),
                                                 self._fwd_cb(arm, kind), 10)
            if spec.grip is not None:
                self.create_subscription(JointTrajectory, spec.grip.topic, self._grip_cb, 10)
            self.create_timer(1.0 / rate_hz, self._tick)
            self.get_logger().info(
                f"팔 {[a.spec.side for a in spec.arms]} · {rate_hz:.0f} Hz · forward {spec.forward} · "
                f"{STATE_TOPIC} 발행 (정적 {[s.side for s, _ in spec.statics]})")

        @property
        def received(self) -> int:
            return sum(a.received for a in self.spec.arms)

        def _jtc_cb(self, arm: SideArm):
            def cb(msg: JointTrajectory) -> None:
                if not msg.points:
                    return
                if self.cm is not None and not self.cm.is_active(jtc_of(arm.spec.side)):
                    return                               # JTC 비활성 — forward 가 잡고 있다
                arm.set_jtc(list(msg.joint_names), list(msg.points[0].positions))
            return cb

        def _fwd_cb(self, arm: SideArm, kind: str):
            def cb(msg: Float64MultiArray) -> None:
                v = arm.forward_vector(msg.data)
                if v is None:
                    self.get_logger().warning(f"{arm.spec.side} forward {kind} 길이 {len(msg.data)} != {NUM_ARM} — 무시")
                    return
                if self.cm.forward_active(arm.spec.side, kind):   # 활성 컨트롤러의 토픽만 받는다(실기와 같다)
                    arm.set_forward(kind, v)
            return cb

        def _grip_cb(self, msg: JointTrajectory) -> None:
            if msg.points and msg.points[0].positions:
                self.grip = float(msg.points[0].positions[0])

        def _tick(self) -> None:
            names, pos, vel, eff = [], [], [], []
            for arm in self.spec.arms:
                arm.step(mit=self.spec.forward)
                n, p, v, e = arm.rows(self.spec.with_effort)
                names += n; pos += p; vel += v; eff += e
            for sspec, q in self.spec.statics:
                n, p, v, e = static_rows(sspec, q)
                names += n; pos += p; vel += v; eff += e
            if self.spec.grip is not None:
                for src, sign in zip(self.spec.grip.source, self.spec.grip.sign):
                    names.append(src); pos.append(self.grip * sign); vel.append(0.0); eff.append(0.0)
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name, msg.position, msg.velocity, msg.effort = names, pos, vel, eff
            self.pub.publish(msg)

    return FakeArmBridge()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default=None, help="레거시 프로필 이름(config/robots, scripts/robot_profile)")
    parser.add_argument("--contract", default=None, help="계약 모드: deploy_contract.json (asset 포함 v2)")
    parser.add_argument("--robot-yaml", default=None, help="계약 모드: policy_control/config/robots/*.yaml")
    parser.add_argument("--sides", default="", help="계약 모드: 명령을 받는 팔(쉼표). 기본 = 계약의 팔 전부")
    parser.add_argument("--pd-config", default=None,
                        help="계약 모드: pd yaml — 그 gravity 블록으로 g(q) 를 만든다(pd 노드 τ_ff 와 같은 식)")
    parser.add_argument("--model", choices=["rate", "pd"], default="pd")
    parser.add_argument("--max-vel", type=float, default=20.0,
                        help="MockArm 내부 세트포인트 leash[rad/s]. ★기본을 크게 둔 이유: 브리지 rate-limit 은 "
                             "**재생기/pd 가 이미** 걸었다. 여기서 또 걸면 leash 가 실제 위치를 참조해 종단속도가 "
                             "(kp·max_vel·dt − fc)/kd 에 갇힌다 — 팔이 아니라 모델이 만든 정체다.")
    parser.add_argument("--rate-hz", type=float, default=50.0,
                        help="적분·발행 주기. ★높일수록 세트포인트 선행량이 작아져 마찰 데드밴드에 걸린다 — MockArm 이 거부하면 낮출 것.")
    parser.add_argument("--idle-arm-offset", type=float, default=0.0, help="레거시: 유휴 팔을 rest 에서 이만큼 어긋나게(게이트 시험용)")
    parser.add_argument("--forward", action="store_true", default=False,
                        help="pd 노드 경로: /<side>_forward_{position,velocity,effort}_controller/commands 를 받아 MIT 3중으로 "
                             "적분하고 controller_manager 서비스(list/load/configure/switch)를 흉내낸다. 계약 모드는 항상 켜진다.")
    parser.add_argument("--gravity", action="store_true", default=False,
                        help="자산 URDF 체인 g(q) 를 모델에 넣는다(페이로드 없음). 계약 모드에서 --pd-config 가 있으면 그쪽이 우선")
    parser.add_argument("--inertia-q", default=None,
                        help="유효관성을 계산할 관절자세 7값 CSV(canonical 순). 레거시 기본 0(차렷), 계약 모드 기본 = 계약 홈")
    parser.add_argument("--friction-scale", type=float, default=1.0, help="r2s 캘리브 쿨롱 마찰 Fc 배율. 0 = sim 처럼 마찰 없음")
    parser.add_argument("--gains", type=Path, default=DEFAULT_GAINS, help="MIT kp/kd 진실원천(pd 노드와 같은 파일)")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if bool(args.contract) == bool(args.robot):
        raise SystemExit("--robot(레거시) 또는 --contract + --robot-yaml(계약 모드) 중 하나")
    if args.contract and not args.robot_yaml:
        raise SystemExit("--contract 에는 --robot-yaml 이 필요하다")
    spec = build_contract(args) if args.contract else build_legacy(args)

    import rclpy

    rclpy.init()
    node = make_node(spec, args.rate_hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"수신한 지령 {node.received} 건")
        node.destroy_node()
        # Ctrl-C 는 rclpy 가 컨텍스트를 이미 내린 뒤에 여기로 온다 — 두 번 내리면 RCLError 가 나고 종료 코드를 오염시킨다.
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
