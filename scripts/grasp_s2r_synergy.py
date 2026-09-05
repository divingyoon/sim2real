#!/usr/bin/env python3
"""우팔 `grasp_s2r` 손 액션(15D) → 손 20관절 목표로 바꾸는 순수 로직 (numpy).

`grasp_s2r_control._synergy_targets` 를 이식한다. Isaac/torch/ROS 무의존.

한 tick 의 순서 (env 와 동일):

    a       = clip(a_hand, -1, 1).reshape(5, 3)          # 손가락 × 채널(coupled3)
    a       = couple(a)                                  # 엄지만 독립, 나머지는 채널 평균
    cmd     = 0.5·(a + 1)                                # **절대 폐쇄도** [0,1]
    cmd_j   = cmd[fi, ch]                                # 관절 20개로 전개
    delta   = clip(cmd_j − close, −rate, +rate)          # rate = synergy_close_speed
    delta   = where(delta > 0, delta·gate, delta)        # ★닫는 방향만 게이트
    delta   = where(freeze & delta > 0, 0, delta)        # ★닫는 방향만 동결
    close   = clip(close + delta, 0, 1)
    target  = lerp(open_pose, grip_pose, close)  →  clip(soft limits)

★★액션은 **속도가 아니라 절대 폐쇄도 목표**다. `close_speed` 는 그 목표를 향한 변화율
  상한이다. 속도로 해석하면 탐색 노이즈 평균만으로 완전 폐쇄되고 되돌릴 수 없다.

★푸는 방향은 게이트도 동결도 **막지 않는다**. 막으면 잘못 오므린 상태에서 빠져나올
  길이 사라진다.

★폐쇄도는 **관절별 독립 진행도**다. 동결이 관절마다 따로 걸리기 때문이다.

──────────────────────────────────────────────────────────────────────────────
⚠**실기 편차 하나 — 동결 신호**

  g1 은 `synergy_hold_mode=contact` · `freeze_scope=joint` 로, 각 관절의 **자기 링크**
  (`_3`=중간마디 / `_4`=원위마디) 접촉력으로 동결한다. 그런데 실기 Tesollo 는
  **손끝 렌치만** 발행한다(`fingertip_{i}_broadcaster/wrench`, `tip_forces_xyz`) —
  마디별 접촉이 없다. 우팔 5개 런이 전부 같은 방식이라 다른 체크포인트로 피할 수 없다.

  동결을 그냥 빼면 목표가 물체를 지나 계속 전진해 `joint_err`(obs 111~130)가 ±1 로
  포화한다. 그 20칸은 빌더 문서가 "주 파지력 관측"이라 부르는 값이라 무시할 수 없다.

  ⇒ 실기에서는 env 가 이미 갖고 있는 **`blocked` 판정**(관절 실속)을 대체 신호로 쓴다:
        막힘 = |목표 − 실측| > blocked_err_thr_rad   AND   실측이 자기 한계에서 떨어짐
     둘을 함께 봐야 "한계에 부딪힌 것"과 "물체에 막힌 것"이 갈린다 — grip 자세가
     soft limit 을 넘겨 과지령이라(1.8 vs 1.571) 오차 조건만으로는 허공에서 주먹을
     쥐어도 성립한다. 이 신호는 목표·실측·한계만 쓰므로 실기에서 전부 얻을 수 있다.
  이건 **알려진 편차**다. sim 대조로 영향을 재기 전까지는 그렇게 다뤄야 한다.
──────────────────────────────────────────────────────────────────────────────

계약값은 런 dump 와 프로필에서 온다 — `cfg_from_run()` 을 쓰고 손으로 옮기지 말 것.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── 프로필 상수 (`grasp_s2r/robot_profiles.py::TESOLLO_RIGHT`) ───────────────
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
#: 관절 순서 — `r_hj_{finger}_{1..4}`. ★articulation(Isaac DOF) 순서와 **다르다**.
#:   obs 의 손 40칸은 DOF 순이고, 여기 시너지는 이 프로필 순이다. 섞으면 조용히 어긋난다.
HAND_JOINT_NAMES = tuple(f"r_hj_{f}_{j}" for f in FINGERS for j in range(1, 5))
#: 폐쇄도 0 (접근 자세)
HAND_OPEN_POSE = (
    0.0, -1.57, -0.5, 0.0,      # thumb — _2 는 opposition 고정, _3 은 pre-curl
    0.0, 0.0, 0.0, 0.0,         # index
    0.0, 0.0, 0.0, 0.0,         # middle
    0.0, 0.0, 0.0, 0.0,         # ring
    0.0, 0.0, 0.0, 0.0,         # pinky
)
#: 폐쇄도 1 (완전 파지). ★1.8 은 관절한계(±1.571) 초과 **과지령**이고 soft limit 이
#:   흡수한다 — 목표를 한계에 정확히 두면 PD 가 한계 직전에서 힘을 못 낸다.
HAND_GRIP_POSE = (
    0.0, -1.57, 1.8, 1.8,       # thumb
    0.0, 1.9, 1.8, 1.8,         # index
    0.0, 1.9, 1.8, 1.8,         # middle
    0.0, 1.9, 1.8, 1.8,         # ring
    0.0, 0.0, 1.8, 1.8,         # pinky — _2 는 실측 가동폭 0
)
#: 관절 접미사 → 폐쇄 채널. [외전, MCP, PIP·DIP 공통] — 손가락당 3채널.
HAND_CHANNEL_OF_JOINT = {"1": 0, "2": 1, "3": 2, "4": 2}
#: 접촉 시 동결할 접미사. 이것이 감쌈 생성 메커니즘이다(풀면 핀치가 된다).
HAND_FREEZE_SUFFIXES = ("3", "4")
#: 대향 그룹(엄지) — 4지 커플링에서 홀로 독립이다.
CONTACT_GROUP_A = ("thumb",)


@dataclass(frozen=True)
class SynergyCfg:
    """런에서 읽는 값 — 기본값은 g1(`g1_rot20_fresh`) 실측이다."""

    close_speed: float = 0.005
    couple_four_fingers: bool = True
    residual_scale: float = 0.0          # 0 = 순수 채널평균(구 coupled 항등)
    hand_layout: str = "coupled3"
    oppose_grip_delta_rad: float = -0.6
    weak_finger: str = ""
    weak_finger_curl_scale: float = 1.0
    freeze_scope: str = "joint"          # 'joint' | 'finger'
    release_deadband: float = 0.0
    blocked_err_thr_rad: float = 0.3
    blocked_limit_eps_rad: float = 0.05


def _pose_tables(cfg: SynergyCfg):
    """open/grip 자세표에 런의 노브를 적용한다(`_apply_pose_knobs` 이식)."""
    open_pose = np.asarray(HAND_OPEN_POSE, dtype=float)
    grip_pose = np.asarray(HAND_GRIP_POSE, dtype=float)
    ch_of = {nm: HAND_CHANNEL_OF_JOINT[nm.rsplit("_", 1)[1]] for nm in HAND_JOINT_NAMES}

    d = float(cfg.oppose_grip_delta_rad)
    if d != 0.0:
        idx = [i for i, nm in enumerate(HAND_JOINT_NAMES)
               if ch_of[nm] == 1 and any(f"_{f}_" in nm for f in CONTACT_GROUP_A)]
        if not idx:
            raise SystemExit(
                "[synergy] 대향 손가락의 ch1 관절을 못 찾았다 — "
                "oppose_grip_delta_rad 가 조용히 무효가 된다")
        for i in idx:
            grip_pose[i] = open_pose[i] + d

    wf, ws = str(cfg.weak_finger), float(cfg.weak_finger_curl_scale)
    if wf and ws != 1.0:
        idx = [i for i, nm in enumerate(HAND_JOINT_NAMES)
               if f"_{wf}_" in nm and ch_of[nm] == 2]
        if not idx:
            raise SystemExit(f"[synergy] 손가락 '{wf}' 의 ch2 관절을 못 찾았다")
        for i in idx:
            grip_pose[i] = open_pose[i] + ws * (grip_pose[i] - open_pose[i])
    return open_pose, grip_pose


class SynergyHand:
    """손 액션 15D → 관절 목표 20D. 한 로봇(단일 env) 기준.

    `soft_limits` 는 (20, 2) — 자산의 soft joint position limit. 주지 않으면 클램프를
    하지 않는다(★그러면 과지령 1.8 이 그대로 나가므로 실기에서는 반드시 줄 것).
    """

    def __init__(self, cfg: SynergyCfg | None = None, soft_limits=None) -> None:
        self.cfg = cfg or SynergyCfg()
        if self.cfg.hand_layout != "coupled3":
            raise SystemExit(
                f"[synergy] hand_layout={self.cfg.hand_layout!r} 는 이 모듈이 옮기지 "
                "않았다(per_finger 는 별도 슬롯 매핑이 필요하다)")
        self.open_pose, self.grip_pose = _pose_tables(self.cfg)

        sfx = [nm.rsplit("_", 1)[1] for nm in HAND_JOINT_NAMES]
        self.ch = np.array([HAND_CHANNEL_OF_JOINT[s] for s in sfx], dtype=int)
        self.fi = np.array([FINGERS.index(nm.split("_")[2]) for nm in HAND_JOINT_NAMES],
                           dtype=int)
        self.n_ch = len(set(self.ch.tolist()))
        # ★가동 관절 — open == grip 이면 명령해도 안 움직인다(전 `_1`·pinky_2·thumb_2).
        self.movable = np.abs(self.grip_pose - self.open_pose) > 1e-4
        self.flex = np.array([s in ("2", "3", "4") for s in sfx])
        self.freeze_mid = np.array([s in HAND_FREEZE_SUFFIXES and s == "3" for s in sfx])
        self.freeze_dist = np.array([s in HAND_FREEZE_SUFFIXES and s != "3" for s in sfx])
        self._group_a = [FINGERS.index(f) for f in CONTACT_GROUP_A]

        self.soft_lo = self.soft_hi = None
        if soft_limits is not None:
            lim = np.asarray(soft_limits, dtype=float).reshape(len(HAND_JOINT_NAMES), 2)
            self.soft_lo, self.soft_hi = lim[:, 0], lim[:, 1]

        self.close = np.zeros(len(HAND_JOINT_NAMES))
        self.target = self.open_pose.copy()

    # ------------------------------------------------------------------
    def reset(self, hand_q=None) -> None:
        """에피소드 시작. 폐쇄도 0, 목표는 실측(주면) 또는 open 자세."""
        self.close = np.zeros(len(HAND_JOINT_NAMES))
        self.target = (np.asarray(hand_q, dtype=float).copy()
                       if hand_q is not None else self.open_pose.copy())

    # ------------------------------------------------------------------
    def _couple(self, a: np.ndarray) -> np.ndarray:
        """(5, n_ch) 액션에 4지 커플링을 적용한다."""
        if not self.cfg.couple_four_fingers:
            return a
        mask = np.ones(len(FINGERS), dtype=bool)
        mask[self._group_a] = False
        common = a[mask, :].mean(axis=0, keepdims=True)
        rs = float(self.cfg.residual_scale)
        blend = common if rs == 0.0 else common + rs * (a - common)
        out = a.copy()
        out[mask, :] = np.broadcast_to(blend, (int(mask.sum()), a.shape[1])) \
            if blend.shape[0] == 1 else blend[mask, :]
        return out

    def blocked(self, hand_q) -> np.ndarray:
        """관절이 **외부에 막혀** 있는가 (20,) bool — 실기 동결 대체 신호.

        판별은 두 조건의 AND 다: 목표를 못 따라가고(오차), 자기 한계에서는 떨어져 있다.
        오차 조건만 보면 grip 과지령(1.8 rad) 때문에 허공에서 주먹을 쥐어도 성립한다.
        """
        q = np.asarray(hand_q, dtype=float).reshape(len(HAND_JOINT_NAMES))
        stuck = np.abs(self.target - q) > float(self.cfg.blocked_err_thr_rad)
        if self.soft_lo is None or self.soft_hi is None:
            # ★한계를 모르면 "한계에서 떨어져 있다"를 판정할 수 없다 — 오차 조건만
            #   남으면 허공에서 주먹을 쥐어도 막힘이 된다. 그래서 한계는 필수다.
            free = np.ones_like(stuck)
        else:
            eps = float(self.cfg.blocked_limit_eps_rad)
            free = (q > self.soft_lo + eps) & (q < self.soft_hi - eps)
        return stuck & free & self.movable

    # ------------------------------------------------------------------
    def step(self, action_hand, *, close_gate: float = 1.0, hand_q=None,
             contact_mid=None, contact_dist=None) -> np.ndarray:
        """손 액션 15D → 관절 목표 20D.

        동결 신호는 하나만 쓴다:
          - `contact_mid`/`contact_dist`(각 5,) 를 주면 **sim 과 같은 접촉 동결**
          - 안 주면 `hand_q` 로 **실속 판정**(실기 대체 신호). 둘 다 없으면 동결 없음.
        """
        c = self.cfg
        a = np.clip(np.asarray(action_hand, dtype=float).reshape(len(FINGERS), self.n_ch),
                    -1.0, 1.0)
        a = self._couple(a)
        cmd = 0.5 * (a + 1.0)                       # 절대 폐쇄도 [0, 1]
        cmd_j = cmd[self.fi, self.ch]

        rate = float(c.close_speed)
        delta = np.clip(cmd_j - self.close, -rate, rate)
        # ★닫는 방향만 게이트로 스케일 — 푸는 방향은 항상 허용해야 빠져나올 수 있다.
        delta = np.where(delta > 0.0, delta * float(close_gate), delta)

        hold = self._freeze_mask(hand_q, contact_mid, contact_dist)
        if hold is not None:
            delta = np.where(hold & (delta > 0.0), 0.0, delta)
            rdb = float(c.release_deadband)
            if rdb > 0.0:
                delta = np.where(hold & (delta < 0.0) & (delta > -rdb), 0.0, delta)

        self.close = np.clip(self.close + delta, 0.0, 1.0)
        tgt = self.open_pose + (self.grip_pose - self.open_pose) * self.close
        if self.soft_lo is not None:
            tgt = np.clip(tgt, self.soft_lo, self.soft_hi)
        self.target = tgt
        return tgt

    def _freeze_mask(self, hand_q, contact_mid, contact_dist):
        """동결 마스크 (20,) bool 또는 None."""
        if contact_mid is not None and contact_dist is not None:
            mid = np.asarray(contact_mid, dtype=bool).reshape(len(FINGERS))
            dist = np.asarray(contact_dist, dtype=bool).reshape(len(FINGERS))
            h_mid, h_dist = mid[self.fi], dist[self.fi]
            if self.cfg.freeze_scope == "finger":
                return (h_mid | h_dist) & self.flex
            return (h_mid & self.freeze_mid) | (h_dist & self.freeze_dist)
        if hand_q is not None:
            # 실기 대체 — 실속한 관절만 얼린다. 동결 접미사(_3/_4)로 범위를 맞춘다.
            return self.blocked(hand_q) & (self.freeze_mid | self.freeze_dist)
        return None


# ---------------------------------------------------------------------------
def cfg_from_run(env_yaml_path) -> SynergyCfg:
    """런 dump 에서 시너지 파라미터를 읽는다."""
    import re
    from pathlib import Path

    text = Path(env_yaml_path).read_text()

    def val(key, default):
        m = re.search(rf"^\s*{key}:\s*(.+?)\s*$", text, re.M)
        if m is None:
            return default
        v = m.group(1).strip().strip("'\"")
        if v in ("true", "false"):
            return v == "true"
        if v == "null":
            return default
        try:
            return float(v)
        except ValueError:
            return v

    mode = str(val("synergy_hold_mode", "contact"))
    if mode not in ("contact", "blocked"):
        raise SystemExit(f"[synergy] 모르는 synergy_hold_mode={mode!r}")
    return SynergyCfg(
        close_speed=float(val("synergy_close_speed", 0.005)),
        couple_four_fingers=bool(val("couple_four_fingers", True)),
        residual_scale=float(val("finger_residual_scale", 0.0)),
        hand_layout=str(val("hand_layout", "coupled3")),
        oppose_grip_delta_rad=float(val("oppose_grip_delta_rad", 0.0)),
        weak_finger=str(val("weak_finger", "")),
        weak_finger_curl_scale=float(val("weak_finger_curl_scale", 1.0)),
        freeze_scope=str(val("synergy_freeze_scope", "joint")),
        release_deadband=float(val("synergy_release_deadband", 0.0)),
        blocked_err_thr_rad=float(val("blocked_err_thr_rad", 0.3)),
        blocked_limit_eps_rad=float(val("blocked_limit_eps_rad", 0.05)),
    )
