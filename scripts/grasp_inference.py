#!/usr/bin/env python3
"""grasp_v1 라이브 sim2real inference 노드 (구성 프로필 기반, 좌/우).

로직은 `grasp_policy_core.GraspPolicyCore` 에 있고 이 노드는 **ROS 배선과 안전 게이트**만
담당한다(오프라인 재현기 grasp_loop_sim 과 같은 코어를 써서 드리프트를 막는다).

실행:
    python3 grasp_inference.py --robot tesollo_bi_s__right \
        --agent /path/to/params/agent.yaml --ckpt /path/to/...pth \
        [--device cuda:0] [--settle_time 4.0] [--object cup_big_s100]

    ros2 service call /grasp/start std_srvs/srv/Trigger

구독 (토픽은 구성 프로필에서 해석 — 좌/우 자동):
    <arm_state>       arm 7D   (source명 → canonical 매핑)
    <ee_state>        hand 20D
    /cup_pose         PoseStamped — 컵 위치 (robot base)
    <tip_force_xyz>   Float64MultiArray 15D — 손끝 3축 힘 (tip-major, tip-local, [N])

발행: <arm_cmd> 7D rad, <ee_cmd> 20D rad
      → 브리지(isaacsim_cmd_to_jtc)가 robot_control JTC 로 변환

계약 (sim `grasp_{side}_env.py` 재현, 08.16 개편 반영):
    - obs 154D  : arm14 + finger40 + palm3 + tip_rel15 + palm_to_cup3 + cup_to_tip15
                  + tip_force_local15 + joint_pos_err20 + last_actions21 + onehot8
    - action 21D: palm 6 + 손가락 15(5×3 채널, 4지 공통닫힘, 절대 폐쇄도 + 변화율 상한)
    - 리셋      : **고정 홈**(컵 참값 pregrasp 폐기 — 실기 컵 위치는 perception 결과라
                  참값 텔레포트가 불가능하고, 홈→컵 접근은 정책이 학습한 몫)
    - lift 진입 : 접촉 래치(≥3손가락 8스텝 hold) → joint7-only lift-wait
    - 정책 60Hz, fabric_decimation=2

★ tip-only 제어: sim 의 손가락 게이트·lift 래치는 (tip|mid|distal) 을 쓰지만 middle/distal
  은 **critic 전용(privileged)** 이라 실기서 감지 불가하다. 라이브는 tip 접촉만 쓴다 —
  의도된 차이다.

★ 안전 게이트(전부 실사고 대응, 유지할 것):
  · START_FRESH_SEC : 묵은 컵 pose 로 start 되어 재개 순간 발진한 사고
  · STALL_ABORT_SEC : 무기한 홀드 금지
  · CONTACT_GATE_DIST: 실물 F/T 가 테이블 접촉도 잡아 생긴 거짓 lift 래치
  · 손 피드백 추종 검사: 드라이버 두절 시 obs zeros → LSTM 발산(팔 후퇴)
"""

from __future__ import annotations

import argparse
import sys
import time
from enum import Enum, auto
from pathlib import Path

import numpy as np

# ── Fabrics 경로 ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in [
    _SCRIPT_DIR.parent.parent / "hdgp" / "source" / "FABRICS" / "src",
    _SCRIPT_DIR.parent.parent / "repo" / "FABRICS" / "src",
]:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

# ── Task 경로 (grasp_v1 preset/constants; Isaac 무의존) ───────────────────────
_OPENARM_SRC = _SCRIPT_DIR.parent.parent / "hdgp" / "source" / "openarm"
sys.path.insert(0, str(_OPENARM_SRC))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

sys.path.insert(0, str(_SCRIPT_DIR))
from fabrics_ros_interface import create_publisher
from policy_loader import RLGamesLstmActorPolicy
from episode_recorder import EpisodeCsvRecorder
from grasp_obs_builder import REAL_CUP_INDEX, make_object_onehot
from grasp_policy_core import GraspPolicyCore, TickSensors
from robot_profile import idle_arm_rest_pose, load_hdgp_module, load_robot_profile

# 계약 상수·자세는 **구성 프로필**을 통해 hdgp 에서 해석한다(값 복제 금지).
# 복제하면 sim 이 바꿨을 때 조용히 어긋나고, 좌우 미러 자세가 특히 그렇다.

# ---------------------------------------------------------------------------
# 노드 운영 상수 (구성과 무관 — sim 계약이 아니라 배포 정책)
# ---------------------------------------------------------------------------
CONTROL_HZ = 60.0               # 정책 루프(sim decimation 2 @120Hz = 60Hz 와 동일)
APPROACH_CMD_HZ = 10.0          # APPROACHING 명령 발행
HAND_START_MISMATCH_RAD = 0.6   # settle 후 |hand − APPROACH| 허용 한계(죽은손 thumb_2 오차 1.57)
SENSOR_STALE_SEC = 0.5          # RUNNING 중 팔/손/컵 두절 판정
START_FRESH_SEC = 1.0           # start 허용: 모든 센서가 이 시간 내 수신됐을 것
STALL_ABORT_SEC = 5.0           # 두절 홀드가 이 시간 지속되면 에피소드 자동 중단
REPLAY_DECIMATION = 2           # PLACING 역재생 감속(2 = 30Hz)
CONTACT_GATE_DIST = 0.10        # palm-컵 거리[m] 이내에서만 tip 접촉 유효(테이블 접촉 배제)
IDLE_ARM_MISMATCH_RAD = 0.15    # 유휴(반대편) 팔이 rest 에서 벗어난 허용 한계


class State(Enum):
    IDLE = auto()
    APPROACHING = auto()
    RUNNING = auto()
    PLACING = auto()    # 에피소드 완료 후 명령 궤적 역재생(컵 제자리 반환) → IDLE
    DONE = auto()



class GraspInferenceNode(Node):

    def __init__(
        self,
        agent_yaml: str,
        checkpoint_path: str,
        robot: str = "tesollo_bi_s__right",
        device: str = "cuda:0",
        settle_time: float = 4.0,
        object_name: str | int | None = None,
        allow_hand_mismatch: bool = False,
        allow_idle_arm_mismatch: bool = False,
        contact_threshold: float | None = None,
        episode_steps: int | None = None,
        log_dir: str = "~/rl_ws/sim2real/logs",
    ) -> None:
        super().__init__("grasp_inference")
        self.device = device
        self.settle_time = settle_time
        self.allow_hand_mismatch = allow_hand_mismatch
        self.allow_idle_arm_mismatch = allow_idle_arm_mismatch
        self.recorder = EpisodeCsvRecorder(log_dir)
        self._traj: list[tuple[np.ndarray, np.ndarray]] = []   # (arm_cmd, hand_cmd) — 역재생용
        self._replay_idx = 0
        self._replay_tick = 0

        # ── 구성 프로필 ──────────────────────────────────────────────────────
        # 자산·토픽·관절·계약을 여기서 해석한다. 매니페스트 대조 검증이 함께 돈다 —
        # 배포가 sim 과 다른 로봇 자산을 쓰던 사고(palm 6.5cm 어긋남)의 방어선.
        self.profile = load_robot_profile(robot)
        self.get_logger().info(
            f"구성 [{self.profile.name}] side={self.profile.acting_side} "
            f"ee={self.profile.ee_type}/{self.profile.ee_dof} "
            f"fabrics={self.profile.fabrics.robot_dir}"
        )
        preset = load_hdgp_module(self.profile, "preset")
        consts = load_hdgp_module(self.profile, "constants")
        self.hand_approach = np.asarray(preset.HAND_APPROACH_POSE, dtype=np.float64)
        self.hand_joint_names = list(self.profile.ee_canonical)
        self.episode_steps = int(consts.EPISODE_STEPS if episode_steps is None else episode_steps)
        # sim 상수(0.1N)는 노이즈 없는 sim 접촉센서 기준 — 실물 F/T 노이즈/진동 위로 튜닝.
        self.contact_threshold = float(
            consts.CONTACT_FORCE_THRESHOLD if contact_threshold is None else contact_threshold
        )

        # /joint_states 는 source명으로 발행되므로 source→(canonical index, sign) 매핑.
        _lim = self.profile.joint_limits
        self._arm_src = {
            _lim[c]["source"]: (i, _lim[c]["sign"])
            for i, c in enumerate(self.profile.arm_canonical)
        }
        self._hand_src = {
            _lim[c]["source"]: (i, _lim[c]["sign"])
            for i, c in enumerate(self.profile.ee_canonical)
        }
        # 유휴(반대편) 팔 — 명령하지 않고 **자세만 확인**한다. sim 은 유휴 팔을 rest 로
        # 고정한 채 학습하고 그 팔은 물리 충돌체다. 실기가 다른 곳에 있으면 학습된 궤적이
        # 안전하지 않다(특히 이 로봇은 양팔이 같은 작업공간을 공유한다).
        self._idle_src = {
            _lim[c]["source"]: (i, _lim[c]["sign"])
            for i, c in enumerate(self.profile.idle_arm_canonical)
        }
        self.idle_arm_rest = np.asarray(idle_arm_rest_pose(self.profile), dtype=np.float64)
        self.idle_arm_pos = self.idle_arm_rest.copy()   # 미수신 시 "정상" 가정 금지용 초기값
        self._idle_ready = False

        # ── Policy ───────────────────────────────────────────────────────────
        self.get_logger().info("Policy 로드 중...")
        self.policy = RLGamesLstmActorPolicy(
            agent_yaml_path=agent_yaml,
            checkpoint_path=checkpoint_path,
            obs_dim=self.profile.contract.obs_dim,        # 154
            action_dim=self.profile.contract.action_dim,  # 21
            device=device,
        )

        # 잡는 물체 onehot (라이브 고정). 기본 cup_big_s100(index 1).
        _obj = REAL_CUP_INDEX if object_name is None else object_name
        self.object_onehot = make_object_onehot(_obj)
        self.get_logger().info(f"물체 onehot 고정: {_obj} → {self.object_onehot.tolist()}")

        # ── 정책 tick 코어 (Fabrics·obs·디코더 전부 여기서) ────────────────────
        self.get_logger().info("Fabrics·정책 코어 초기화 중...")
        self.core = GraspPolicyCore(
            profile=self.profile,
            policy=self.policy,
            device=device,
            contact_threshold=self.contact_threshold,
            contact_gate_dist=CONTACT_GATE_DIST,
            object_onehot=self.object_onehot,
        )
        self.get_logger().info(
            "고정 홈 q_home=["
            + ", ".join(f"{v:+.4f}" for v in self.core.q_home_arm.tolist())
            + "]  (컵 참값 pregrasp 폐기 — 접근은 정책이 학습)"
        )

        # ── 센서 버퍼 ────────────────────────────────────────────────────────
        # ★hand_pos 초기값 = APPROACH (zeros 금지): 손 상태 미수신 시 zeros(20) obs 가
        #   LSTM 을 발산시켜 팔이 후퇴함 (08.03 grasp_loop_sim hand=zero 재현으로 실증).
        self.arm_pos = np.zeros(7)
        self.arm_vel = np.zeros(7)
        self.arm_eff = np.zeros(7)
        self.hand_pos = self.hand_approach.copy()
        self.hand_vel = np.zeros(20)
        self.hand_eff = np.zeros(20)
        self.cup_pos = np.zeros(3)
        # ★08.16 계약: 접촉 obs 는 이진 5D 가 아니라 **3축 힘 15D**(tip-local, 원시 [N]).
        self.tip_force_local = np.zeros((5, 3))

        self._arm_ready = False
        self._hand_ready = False
        self._cup_ready = False
        # ★tip 을 정식 감시 대상으로 넣는다: 접촉 힘은 이제 obs 15차원이라, 발행이 없으면
        #   조용히 "무접촉" 이 되어 정책이 영원히 lift 하지 못한다(무접촉과 두절이 구분 안 됨).
        self._tip_ready = False
        self._last_rx = {"arm": 0.0, "hand": 0.0, "cup": 0.0, "tip": 0.0}

        # ── 에피소드 상태 ────────────────────────────────────────────────────
        self.state = State.IDLE
        self.step_count = 0
        self._stall_since = None
        self._approach_start_time = 0.0

        # ── ROS2 (토픽은 전부 구성 프로필에서) ─────────────────────────────────
        t = self.profile.topics
        self.create_subscription(JointState, t["arm_state"], self._arm_cb, 10)
        self.create_subscription(JointState, t["ee_state"], self._hand_cb, 10)
        self.create_subscription(PoseStamped, "/cup_pose", self._cup_cb, 10)
        self.create_subscription(Float64MultiArray, t["tip_force_xyz"], self._tip_cb, 10)

        self.cmd_pub = create_publisher()
        self.arm_cmd_topic = t["arm_cmd"]
        self.hand_cmd_topic = t["ee_cmd"]

        self.create_service(Trigger, "/grasp/start", self._start_cb)
        self.create_service(Trigger, "/grasp/stop", self._stop_cb)
        self.create_service(Trigger, "/grasp/reset", self._reset_cb)

        self.create_timer(1.0 / APPROACH_CMD_HZ, self._approach_loop)
        self.create_timer(1.0 / CONTROL_HZ, self._policy_loop)

        self.get_logger().info("준비 완료. '/grasp/start' 서비스 호출 시 에피소드 시작.")

    # ------------------------------------------------------------------
    # 센서 Callbacks
    # ------------------------------------------------------------------
    def _arm_cb(self, msg: JointState) -> None:
        got = False
        for i, name in enumerate(msg.name):
            idle = self._idle_src.get(name)
            if idle is not None:
                self.idle_arm_pos[idle[0]] = idle[1] * msg.position[i]
                self._idle_ready = True
            m = self._arm_src.get(name)
            if m is not None:
                idx, sign = m
                self.arm_pos[idx] = sign * msg.position[i]
                if msg.velocity:
                    self.arm_vel[idx] = sign * msg.velocity[i]
                if msg.effort:
                    self.arm_eff[idx] = sign * msg.effort[i]
                got = True
        if got:
            self._arm_ready = True
            self._last_rx["arm"] = time.monotonic()

    def _hand_cb(self, msg: JointState) -> None:
        got = False
        for i, name in enumerate(msg.name):
            m = self._hand_src.get(name)
            if m is not None:
                idx, sign = m
                self.hand_pos[idx] = sign * msg.position[i]
                if msg.velocity:
                    self.hand_vel[idx] = sign * msg.velocity[i]
                if msg.effort:
                    self.hand_eff[idx] = sign * msg.effort[i]
                got = True
        if got:
            self._hand_ready = True
            self._last_rx["hand"] = time.monotonic()

    def _cup_cb(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self.cup_pos[:] = [p.x, p.y, p.z]
        self._cup_ready = True
        self._last_rx["cup"] = time.monotonic()

    def _tip_cb(self, msg: Float64MultiArray) -> None:
        """<tip_force_xyz> 15D (tip-major 5×3, tip-local 원시 [N]) 수신.

        ★차원 검증은 조용히 넘기지 않는다. 구 5D `contact_forces` 를 이 토픽에 잘못
          연결하면 길이 5 로 들어오는데, 이를 zeros 로 메우면 "접촉 전무" 로 보여
          정책이 영원히 lift 하지 못한다(과거 손 obs zeros 사고와 동형).
        """
        n = len(msg.data)
        if n != 15:
            self.get_logger().error(
                f"tip 힘 토픽 길이 {n} != 15 — 구 5D contact_forces 를 연결했는지 확인 "
                f"({self.profile.topics['tip_force_xyz']} 는 15D tip-major)",
                throttle_duration_sec=5.0,
            )
            return
        self.tip_force_local = np.asarray(list(msg.data), dtype=np.float64).reshape(5, 3)
        self._tip_ready = True
        self._last_rx["tip"] = time.monotonic()

    # ------------------------------------------------------------------
    # 서비스 Callbacks
    # ------------------------------------------------------------------
    def _start_cb(self, request, response):
        if self.state not in (State.IDLE, State.DONE):
            response.success = False
            response.message = f"ERROR: 현재 상태={self.state.name}, IDLE/DONE 에서만 start"
            return response
        if not (self._arm_ready and self._hand_ready and self._cup_ready and self._tip_ready):
            missing = []
            if not self._arm_ready:  missing.append(self.profile.topics["arm_state"])
            if not self._hand_ready: missing.append(self.profile.topics["ee_state"])
            if not self._cup_ready:  missing.append("/cup_pose")
            if not self._tip_ready:  missing.append(self.profile.topics["tip_force_xyz"])
            response.success = False
            response.message = f"ERROR: 미수신 토픽: {missing}"
            self.get_logger().error(response.message)
            return response

        # ★신선도 게이트: "한 번이라도 수신"만 보면 묵은 컵 pose 로 start 가 통과된다
        #   (08.03 실사고: 6분 전 pose 로 시작 → RUNNING 41s 두절 홀드 → 컵 재개 순간
        #    예고 없이 팔이 움직임). 모든 센서가 지금 흐르고 있어야만 start 허용.
        _now = time.monotonic()
        _stale = [k for k, t in self._last_rx.items() if _now - t > START_FRESH_SEC]
        if _stale:
            response.success = False
            response.message = (
                f"ERROR: 토픽 신선도 불량 {_stale} (>{START_FRESH_SEC}s 무수신) — "
                "발행 재개 확인 후 start (cup_pose_watch 🟢 확인)"
            )
            self.get_logger().error(response.message)
            return response

        # 손 자세 참고 로그: start 시점 손은 아직 APPROACH 명령 전이라 휴지 자세(0 근방)일 수
        # 있음 — 여기선 경고만. 판별 게이트는 APPROACHING settle 종료 시(_hand_mismatch 참조,
        # APPROACH 를 settle 동안 명령한 뒤 추종 여부로 죽은 드라이버를 확정).
        _bad = self._hand_mismatch()
        if _bad:
            self.get_logger().warning(
                f"start 시점 손 자세가 APPROACH 와 어긋남({len(_bad)}관절) — "
                "APPROACHING 에서 명령 추종을 검사합니다"
            )

        # ★유휴 팔 자세 확인: sim 은 유휴 팔을 rest 로 고정한 채 학습했고 그 팔은 물리
        #   충돌체다. 실기가 다른 곳에 있으면 학습된 궤적이 안전하지 않다.
        if not self._idle_ready:
            self.get_logger().warning(
                f"유휴 팔({self.profile.idle_arm_canonical[0][:1]}_aj_*) 상태 미수신 — "
                "자세를 확인할 수 없다. 브링업이 양팔을 올렸는지 점검하라."
            )
        _idle_bad = self._idle_arm_mismatch()
        if _idle_bad and not self.allow_idle_arm_mismatch:
            detail = ", ".join(f"{n}={v:+.3f}(기대 {e:+.3f})" for n, v, e in _idle_bad[:7])
            response.success = False
            response.message = (
                f"ERROR: 유휴 팔이 rest 에서 벗어남({len(_idle_bad)}관절): {detail}\n"
                "  sim 은 이 팔을 rest 에 고정한 채 학습했다 — 다른 자세면 파지 팔 궤적이"
                " 학습 때 없던 장애물을 만난다.\n"
                f"  robotctl 등으로 rest 로 옮긴 뒤 재시도 (의도된 경우 --allow-idle-arm-mismatch)"
            )
            self.get_logger().error(response.message)
            return response

        self._reset_episode_state()
        _csv = self.recorder.start()
        self.get_logger().info(f"에피소드 CSV 기록 시작: {_csv}")
        self.state = State.APPROACHING
        self._approach_start_time = time.monotonic()
        response.success = True
        response.message = (
            f"APPROACHING 시작 (settle={self.settle_time}s). "
            f"home_arm={[f'{v:.3f}' for v in self.core.q_home_arm.tolist()]}"
        )
        self.get_logger().info(response.message)
        return response

    def _stop_cb(self, request, response):
        self.recorder.close()
        self._traj.clear()
        self.state = State.IDLE
        response.success = True
        response.message = "중단 → IDLE"
        return response

    def _reset_cb(self, request, response):
        self.recorder.close()
        self.state = State.IDLE
        self._reset_episode_state()
        response.success = True
        response.message = "리셋 → IDLE"
        return response

    def _hand_mismatch(self) -> list[tuple[str, float, float]]:
        """|hand_pos − APPROACH| > HAND_START_MISMATCH_RAD 인 (관절명, 실측, 기대) 목록.

        죽은 드라이버(Modbus 두절)는 물리 자세와 무관하게 0.000 을 발행 — APPROACH 명령 후에도
        thumb_2(기대 −1.57)가 0 으로 남아 걸린다. 피드백이 얼면 obs 가 zeros 로 조립되어
        LSTM 발산(팔 후퇴, 08.03 실증)이라 RUNNING 진입 금지가 안전.
        """
        err = np.abs(self.hand_pos - self.hand_approach)
        return [
            (self.hand_joint_names[i], float(self.hand_pos[i]), float(self.hand_approach[i]))
            for i in np.where(err > HAND_START_MISMATCH_RAD)[0]
        ]

    def _idle_arm_mismatch(self) -> list[tuple[str, float, float]]:
        """유휴 팔이 rest 에서 벗어난 (관절명, 실측, 기대) 목록.

        sim 학습 장면은 유휴 팔이 rest 에 있는 상태다. 실기가 다르면 파지 팔의 궤적이
        학습 때 없던 물체(반대편 팔)를 만난다 — 조용히 넘어가면 충돌로 나타난다.
        """
        if not self._idle_ready:
            return []          # 미수신은 별도 게이트(_idle_ready)에서 다룬다
        err = np.abs(self.idle_arm_pos - self.idle_arm_rest)
        return [
            (self.profile.idle_arm_canonical[i], float(self.idle_arm_pos[i]),
             float(self.idle_arm_rest[i]))
            for i in np.where(err > IDLE_ARM_MISMATCH_RAD)[0]
        ]

    def _reset_episode_state(self) -> None:
        """에피소드 상태 초기화 — 코어가 fabric·디코더·래치·LSTM 을 함께 되돌린다.

        코어 기준 자세는 **고정 홈**이다(컵 참값 pregrasp 폐기). 실기에서 컵 위치는
        perception 결과라 참값으로 팔을 텔레포트할 수 없고, 접근은 정책이 학습한다.
        """
        self.step_count = 0
        self._traj.clear()
        self._stall_since = None
        self.core.reset_episode(self.core.q_home_arm, self.hand_pos)

    # ------------------------------------------------------------------
    # APPROACHING (10Hz)
    # ------------------------------------------------------------------
    def _approach_loop(self) -> None:
        if self.state != State.APPROACHING:
            return
        # 컵과 무관한 **고정 홈** 으로 이동한다. 컵 위치는 perception 결과라 참값
        # pregrasp 로 텔레포트할 수 없고, 홈→컵 접근은 정책이 학습한 몫이다.
        self.cmd_pub.send_side_full(
            self.profile.acting_side, self.core.q_home_arm.tolist(), self.hand_approach.tolist()
        )
        if time.monotonic() - self._approach_start_time >= self.settle_time:
            # ★죽은 손 판별 게이트: settle 동안 APPROACH 를 명령했는데도 손 피드백이
            #   안 따라오면(예: 물리 -1.57 인데 0.000 보고 = Modbus 피드백 동결) RUNNING 금지.
            _bad = self._hand_mismatch()
            if _bad and not self.allow_hand_mismatch:
                detail = ", ".join(f"{n}={v:.3f}(기대 {e:.3f})" for n, v, e in _bad[:6])
                self.get_logger().error(
                    f"APPROACH 명령 {self.settle_time:.0f}s 후에도 손 피드백 미추종"
                    f"({len(_bad)}관절): {detail} — 드라이버 피드백 두절 의심. "
                    "손 전원 재인가/드라이버 재기동 후 재시도 (의도된 경우 --allow-hand-mismatch) → IDLE"
                )
                self.recorder.close()
                self.state = State.IDLE
                return
            # RUNNING 진입 직전 코어 상태를 **실측**에서 다시 시작한다(첫 tick 점프 방지).
            # 이후로는 fabric_q 를 실측 재동기화하지 않는다 — 느린 실팔 위치로 명령이
            # 붕괴해 전진 불가(08.03 RUNNING 동결 근본원인).
            self.core.reset_episode(self.arm_pos, self.hand_pos)
            self.state = State.RUNNING
            self.get_logger().info("settle 완료 → RUNNING")

    # ------------------------------------------------------------------
    # RUNNING (60Hz)
    # ------------------------------------------------------------------
    def _policy_loop(self) -> None:
        if self.state == State.PLACING:
            self._placing_tick()
            return
        if self.state != State.RUNNING:
            return

        # 0. 센서 staleness 감시: 팔/손/컵 중 하나라도 두절이면 명령 홀드(전송 중단).
        #    죽은 값으로 obs 를 조립하면 LSTM 발산(팔 후퇴) — 마지막 명령 유지가 안전.
        _now = time.monotonic()
        _stale = [k for k, t in self._last_rx.items() if _now - t > SENSOR_STALE_SEC]
        if _stale:
            # ★무기한 홀드 금지: 두절이 길어진 뒤 센서가 재개되면 그 순간 예고 없이 로봇이
            #   움직인다(08.03 실사고 41s 홀드→재개 순간 발진). 지속 두절은 에피소드 중단.
            if self._stall_since is None:
                self._stall_since = _now
            elif _now - self._stall_since > STALL_ABORT_SEC:
                self.get_logger().error(
                    f"[RUNNING] 센서 두절 {_stale} {STALL_ABORT_SEC:.0f}s 지속 — 에피소드 중단 → IDLE"
                )
                self.recorder.close()
                self._stall_since = None
                self.state = State.IDLE
                return
            self.get_logger().error(
                f"[RUNNING] 센서 두절 {_stale} (>{SENSOR_STALE_SEC}s) — 명령 홀드",
                throttle_duration_sec=1.0,
            )
            return
        self._stall_since = None

        # ── 정책 tick: FK → obs 154D → action 21D → 손 20D/팔 7D ────────────
        # 로직은 전부 GraspPolicyCore 에 있다(grasp_loop_sim 과 공유해 드리프트 차단).
        out = self.core.step(
            TickSensors(
                arm_pos=self.arm_pos, arm_vel=self.arm_vel,
                hand_pos=self.hand_pos, hand_vel=self.hand_vel,
                cup_pos=self.cup_pos, tip_force_local=self.tip_force_local,
            ),
            step_count=self.step_count,
        )

        # [debug] palm 이 컵으로 접근하는지 (거리 줄면 정상)
        self.get_logger().info(
            f"[RUNNING] palm={out.palm_center.round(3).tolist()} "
            f"cup={self.cup_pos.round(3).tolist()} dist={out.palm_cup_dist:.3f}m"
            + (f"  lift(j7 {out.action[0]:+.2f})" if out.is_lift else ""),
            throttle_duration_sec=0.5,
        )
        if out.just_entering_lift:
            self.get_logger().info(
                f"[Lift latch] step={self.step_count}, "
                f"j7: {self.core.lift_arm_start[6]:.3f} → {self.core.prelift_target[6]:.3f}"
            )

        # 명령 전송 (+역재생용 궤적 기록)
        self.cmd_pub.send_side_full(
            self.profile.acting_side, out.arm_cmd.tolist(), out.hand_cmd.tolist()
        )
        self._traj.append((out.arm_cmd.copy(), out.hand_cmd.copy()))

        # CSV per-step 기록 (action·관절·모터·센서·obj)
        self.recorder.record(
            step=self.step_count, is_lift=out.is_lift, action=out.action,
            arm_pos=self.arm_pos, arm_vel=self.arm_vel, arm_eff=self.arm_eff,
            hand_pos=self.hand_pos, hand_eff=self.hand_eff,
            tip_force=np.linalg.norm(out.tip_force_gated, axis=1),
            contact=out.tip_contact,
            cup=self.cup_pos, palm=out.palm_center, dist=out.palm_cup_dist,
            arm_cmd=out.arm_cmd, hand_cmd=out.hand_cmd,
        )

        # 9. 스텝 → 완료 시 역재생(PLACING)으로 컵 반환
        self.step_count += 1
        if self.step_count >= self.episode_steps:
            self.recorder.close()
            self._replay_idx = len(self._traj) - 1
            self._replay_tick = 0
            self.state = State.PLACING
            self.get_logger().info(
                f"에피소드 완료 ({self.episode_steps}스텝 / "
                f"{self.episode_steps / CONTROL_HZ:.1f}s) → PLACING 역재생 "
                f"({len(self._traj)}스텝 × {REPLAY_DECIMATION}배 감속, CSV={self.recorder.path})"
            )

    def _placing_tick(self) -> None:
        """에피소드 명령 궤적 역재생 — 컵을 잡은 역순으로 되돌려 제자리에 놓고 IDLE 대기.

        개루프 재생(60Hz 타이머를 REPLAY_DECIMATION 으로 감속). 역순이므로 lift 하강 →
        손가락 재개방 → pregrasp 복귀가 자연히 이뤄진다. /grasp/stop 으로 즉시 중단 가능.
        """
        self._replay_tick += 1
        if self._replay_tick % REPLAY_DECIMATION != 0:
            return
        if self._replay_idx < 0:
            self._traj.clear()
            self.state = State.IDLE
            self.get_logger().info("PLACING 완료 — 컵 반환·pregrasp 복귀 → IDLE (재트리거 대기)")
            return
        arm_cmd, hand_cmd = self._traj[self._replay_idx]
        self.cmd_pub.send_side_full(
            self.profile.acting_side, arm_cmd.tolist(), hand_cmd.tolist()
        )
        self._replay_idx -= 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--robot", default="tesollo_bi_s__right",
                        help="config/robots 의 구성 프로필 이름 (자산·토픽·관절·계약)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--settle_time", type=float, default=4.0)
    parser.add_argument("--object", default="cup_big_s100",
                        help="잡는 물체 onehot id 또는 인덱스 (기본 cup_big_s100)")
    parser.add_argument("--episode-steps", type=int, default=None,
                        help="에피소드 길이[스텝]. 기본은 sim 계약(600). 1200=2배 천천히")
    parser.add_argument("--log-dir", default="~/rl_ws/sim2real/logs",
                        help="에피소드 CSV 저장 디렉토리")
    parser.add_argument("--contact-threshold", type=float, default=None,
                        help="tip 접촉 판정 임계[N]. 기본은 sim 계약(0.1) — 실물 노이즈 위로 튜닝")
    parser.add_argument("--allow-hand-mismatch", action="store_true", default=False,
                        help="settle 후 손 피드백 추종 게이트 우회")
    parser.add_argument("--allow-idle-arm-mismatch", action="store_true", default=False,
                        help="유휴(반대편) 팔 rest 자세 확인 게이트 우회")
    args = parser.parse_args()

    obj: str | int
    try:
        obj = int(args.object)
    except ValueError:
        obj = args.object

    rclpy.init()
    node = GraspInferenceNode(
        agent_yaml=args.agent,
        checkpoint_path=args.ckpt,
        robot=args.robot,
        device=args.device,
        settle_time=args.settle_time,
        object_name=obj,
        allow_hand_mismatch=args.allow_hand_mismatch,
        allow_idle_arm_mismatch=args.allow_idle_arm_mismatch,
        contact_threshold=args.contact_threshold,
        episode_steps=args.episode_steps,
        log_dir=args.log_dir,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
