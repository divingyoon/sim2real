#!/usr/bin/env python3
"""real2sim 파이프라인 — 실기 여진 기록에서 sim 액추에이터 파라미터까지, 한 명령으로.

09.01 우팔에서 손으로 하루 걸려 얻은 절차를 그대로 자동화했다. 다른 팔·다른 로봇에
쓰려면 §설정 상수만 바꾸면 된다.

## 무엇을 하는가

    collect(실기)  →  fit  →  apply  →  verify  →  report
       사용자        해석    코드수정    sim재생     판정

  · **fit**    여진 응답에 관절별 2차계 `J q̈ + kd q̇ + kp q + Fc·sign(q̇) = kp q_des`
               를 맞춘다. 나오는 것은 (ωn, ζ, 지연, Fc/J) — kp 와 J 는 **비만** 담기므로
               따로 결정되지 않는다.
  · **apply**  ★kp 는 **실기 하드웨어 값을 그대로** 쓴다(설정 파일에서 읽는다).
               kd 만 `2ζ·√(kp·J_sim)` 으로 맞춘다. J_sim 은 sim 자산의 실제 관성이다.
  · **verify** 두 가지를 **둘 다** 본다 — 동특성(여진 재생)과 정적 추종(궤적 재생).
  · **report** 합격 기준으로 판정한다.

## 왜 이 순서인가 (09.01 에 값을 치르고 배운 것)

1. **kp 를 여진으로 바꾸지 마라.** 여진은 한 자세에서 ±3~9° 만 흔드는데 실제 궤적은
   ±50° 를 움직이고 관성이 자세에 따라 변한다. 여진에서 역산한 kp 를 넣었더니 정적
   추종이 j2 RMSE 0.94° → **10.77°** 로 무너졌다. kp 는 실기 설정 파일이 진실이다.
2. **동특성만 보지 마라.** 위 실패는 여진 지표만 봤으면 못 잡는다.
3. **`r2s fit`(robotctl)의 kp 를 믿지 마라.** 그 모델에는 armature 가 없어 관성을
   kp 로 흡수한다 — j6 을 19배 부풀렸다. 여기서는 kp 를 밖에서 주입한다.
4. **sim 의 `friction` 은 작동하지 않는다**(PhysX). 감쇠는 kd 로만 들어간다.
5. **튜닝 지표로 ptp(최대−최소)를 쓰지 마라.** kd 를 3배 키워도 12 % 밖에 안 변한다.
   lock-in 주파수 응답을 쓴다.
6. **Isaac probe 는 끝나도 GPU 를 안 놓는다.** 이 스크립트가 매 단계 뒤 정리한다.

    python3 r2s_pipeline.py --stage fit          # 해석만(GPU 불필요, 초 단위)
    python3 r2s_pipeline.py --stage apply        # robot_profiles.py 수정
    python3 r2s_pipeline.py --stage verify       # sim 2회(GPU, ~12분)
    python3 r2s_pipeline.py --stage all          # fit→apply→verify→report
    python3 r2s_pipeline.py --stage report       # 이미 있는 결과로 판정만
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R2S = ROOT / "logs" / "r2s"

# ── 설정 (다른 팔/로봇에 쓰려면 여기만 바꾼다) ────────────────────────────
ARM = [f"r_aj_{i}" for i in range(1, 8)]
#: 실기 여진 기록. `robotctl r2s collect --repetitions 3` 산출물.
EXCITE_RUNS = ["right_R3_s0650", "right_R3_s0651"]
EXCITE_HOLDOUT = "right_R3_s0652"
#: 정적 검증에 쓸 궤적(sim 이 기록한 관절 목표). 실기에서도 재생한 것이어야 한다.
PRESET_NPZ = ROOT / "logs/shadow/reset_both/reset_right_v2.npz"
#: ★실기 하드웨어 게인의 **출처**. 하드웨어 인터페이스가 이 값을 모터로 보낸다
#  (`v10_simple_hardware.cpp:65-71,276`). 워크스페이스 사본이 여럿이면 전부 같은지
#  확인할 것 — 다르면 어느 것이 로드되는지부터 가려야 한다.
VENDOR_GAINS = Path("/home/user/rl_ws/sim2real/vendor/openarm/openarm_description"
                    "/config/arm/v10/control_gains.yaml")
#: sim 액추에이터 정의. `HDGP_S2R_REAL_GAINS=1` 분기를 이 스크립트가 고친다.
PROFILE_PY = Path("/home/user/rl_ws/hdgp/source/openarm/openarm/agnostic/tasks"
                  "/grasp_s2r/robot_profiles.py")
ISAACLAB = Path("/home/user/rl_ws/IsaacLab/isaaclab.sh")
#: sim 자산의 관절별 유효 관성. `--stage verify` 가 sim 응답에서 역산해 갱신한다.
#  초기값은 09.01 우팔 실측.
J_SIM_DEFAULT = [0.9135, 0.1417, 1.1442, 1.5106, 0.1415, 0.0397, 0.2315]
#: 손목 kd 에 곱하는 배율. `probe_excite_sim_replay.py --kd-scale` 스윕이 준 값.
WRIST_SCALE = {4: 5.0, 5: 4.0, 6: 0.5}
#: 합격 기준 — 실기 정적 추종 RMSE 와 그 허용 배수.
REAL_STATIC_RMSE_DEG = 0.94
STATIC_TOLERANCE = 1.5
#: 여진 주파수(멀티사인). `robot_control/identification.py:677` 과 같아야 한다.
FREQS = (0.7, 1.3, 2.1, 3.7)


def _log(msg: str) -> None:
    print(f"[r2s] {msg}", flush=True)


def _vendor_kp_kd() -> tuple[list[float], list[float]]:
    """실기 하드웨어 게인을 설정 파일에서 읽는다 — 추정하지 않는다."""
    body = yaml.safe_load(VENDOR_GAINS.read_text())
    kp = [float(body[f"joint{i}"]["kp"]) for i in range(1, len(ARM) + 1)]
    kd = [float(body[f"joint{i}"]["kd"]) for i in range(1, len(ARM) + 1)]
    return kp, kd


def _cleanup_gpu() -> None:
    """Isaac probe 는 끝나도 앱을 안 닫는다 — 누적되면 OOM 이 난다."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", "probe_excite_sim_replay|probe_s2r_gain_replay"],
            capture_output=True, text=True, timeout=20)
        pids = [p for p in out.stdout.split() if p.isdigit()]
        for pid in pids:
            subprocess.run(["kill", "-9", pid], capture_output=True, timeout=10)
        if pids:
            _log(f"GPU 정리: {len(pids)} 개 프로세스 종료")
            time.sleep(3)
    except Exception as exc:                      # noqa: BLE001 — 정리는 실패해도 계속
        _log(f"GPU 정리 건너뜀 ({exc})")


def stage_fit(args) -> dict:
    """여진 응답 → (ωn, ζ) → sim kd. kp 는 실기 값을 그대로 쓴다."""
    kp, vendor_kd = _vendor_kp_kd()
    _log(f"실기 kp {kp}")
    _log(f"실기 kd {vendor_kd} (참고 — sim kd 는 아래에서 다시 계산한다)")

    cmd = [sys.executable, str(HERE / "fit_excite_model.py"),
           "--runs", ",".join(EXCITE_RUNS), "--holdout", EXCITE_HOLDOUT,
           "--kp", ",".join(f"{v:g}" for v in kp),
           "--no-friction",                       # sim friction 이 안 먹으므로 등가 점성
           "--out", str(R2S / "pipeline_fit.json")]
    _log("2차 모델 fit (마찰은 ζ 에 흡수 — sim friction 이 작동하지 않으므로)")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"fit 실패:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    print(proc.stdout)

    fit = json.loads((R2S / "pipeline_fit.json").read_text())["joints"]
    j_sim = args.j_sim or J_SIM_DEFAULT
    out = {}
    for k, name in enumerate(ARM):
        zeta = fit[name]["zeta"]
        # ★kd 는 **sim 의 관성**으로 계산한다. 실기 관성이 아니다 — sim 에서 그 ζ 가
        #   나와야 하기 때문이다.
        kd = 2.0 * zeta * float(np.sqrt(kp[k] * j_sim[k])) * WRIST_SCALE.get(k, 1.0)
        out[name] = {"kp": kp[k], "kd": round(kd, 3), "armature": 0.0, "friction": 0.0,
                     "zeta": zeta, "wn_hz": fit[name]["wn_hz"], "j_sim": j_sim[k],
                     "vendor_kd": vendor_kd[k]}
    (R2S / "pipeline_params.json").write_text(json.dumps(out, indent=2))
    _log(f"→ {R2S / 'pipeline_params.json'}")
    print(f"\n{'관절':8s} {'kp':>7s} {'kd':>8s} {'벤더kd':>8s} {'ωn[Hz]':>7s} {'ζ':>7s}")
    for name in ARM:
        r = out[name]
        print(f"{name:8s} {r['kp']:7.1f} {r['kd']:8.3f} {r['vendor_kd']:8.2f} "
              f"{r['wn_hz']:7.2f} {r['zeta']:7.3f}")
    return out


def stage_apply(args) -> None:
    """robot_profiles.py 의 실측 게인 분기를 갱신한다."""
    params = json.loads((R2S / "pipeline_params.json").read_text())
    text = PROFILE_PY.read_text()
    for k, name in enumerate(ARM, start=1):
        r = params[f"r_aj_{k}"]
        body = (f'"right_arm_j{k}": dict(joint_names_expr=["r_aj_{k}"], '
                f'stiffness={r["kp"]:.1f},\n'
                f'                                     damping={r["kd"]:.3f}, '
                f'friction=0.0,\n'
                f'                                     effort_limit_sim=300.0),')
        pat = re.compile(
            r'"right_arm_j%d": dict\(joint_names_expr=\["r_aj_%d"\], stiffness=[\d.]+,\n'
            r'\s+damping=[\d.]+, friction=[\d.]+,(?: armature=[\d.]+,)?\n'
            r'\s+effort_limit_sim=300\.0\),' % (k, k))
        if not pat.search(text):
            raise SystemExit(f"r_aj_{k} 블록을 못 찾았다 — {PROFILE_PY} 형식이 바뀌었나?")
        text = pat.sub(body, text, count=1)
    if args.execute:
        PROFILE_PY.write_text(text)
        _log(f"갱신 → {PROFILE_PY}")
    else:
        _log("DRY RUN — 실제로 쓰려면 --execute")


def _run_sim(script: str, npz: Path, out: Path, extra: list[str]) -> None:
    cmd = [str(ISAACLAB), "-p", str(HERE / script), "--npz", str(npz),
           "--out", str(out), "--headless"] + extra
    _log(f"sim 실행: {script} ({npz.name})")
    env = {"HDGP_S2R_REAL_GAINS": "1", "PATH": "/usr/bin:/bin:/usr/local/bin",
           "HOME": str(Path.home())}
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(ISAACLAB.parent), env=env, timeout=1800)
    tail = "\n".join(proc.stdout.splitlines()[-25:])
    print(tail)
    if not out.is_file():
        raise SystemExit(f"결과가 없다: {out}\n{proc.stderr[-1500:]}")
    _cleanup_gpu()


def stage_verify(args) -> None:
    """동특성과 정적 추종을 **둘 다** 본다. 한쪽만 보면 다른 쪽이 조용히 무너진다."""
    _cleanup_gpu()
    _run_sim("probe_excite_sim_replay.py", R2S / f"{EXCITE_RUNS[0]}.npz",
             R2S / "pipeline_excite.npz", [])
    _run_sim("probe_s2r_gain_replay.py", PRESET_NPZ,
             R2S / "pipeline_preset.npz", ["--rate_scale", "0.5"])


def stage_report(args) -> int:
    """합격 판정. 동특성은 주파수 응답, 정적은 추종 RMSE."""
    ok = True
    print("\n" + "=" * 64)
    print("real2sim 정합 리포트")
    print("=" * 64)

    preset = R2S / "pipeline_preset.npz"
    if preset.is_file():
        d = np.load(preset)
        err = d["q_meas"] - d["arm_target"][:, 0]
        rmse = float(np.degrees(np.sqrt((err ** 2).mean())))
        limit = REAL_STATIC_RMSE_DEG * STATIC_TOLERANCE
        mark = "✅" if rmse <= limit else "❌"
        ok &= rmse <= limit
        print(f"\n[정적 추종] sim RMSE {rmse:.2f}° · 실기 {REAL_STATIC_RMSE_DEG}° "
              f"· 기준 ≤{limit:.2f}°  {mark}")
        for k, name in enumerate(ARM):
            e = err[:, k]
            print(f"  {name:8s} RMSE {np.degrees(np.sqrt((e**2).mean())):6.2f}° "
                  f"max {np.degrees(np.abs(e).max()):6.2f}°")
    else:
        print("\n[정적 추종] 결과 없음 — --stage verify 를 먼저")
        ok = False

    exc = R2S / "pipeline_excite.npz"
    if exc.is_file():
        d = np.load(exc)
        sim, real = d["sim_ratio"], d["real_ratio"]
        e = np.abs(sim - real)
        print(f"\n[동특성] 오버슈트 재현 평균절대오차 "
              f"전체 {e.mean():.3f} · 팔 {e[:4].mean():.3f} · 손목 {e[4:].mean():.3f}")
        print(f"  {'관절':8s} {'sim':>7s} {'실기':>7s} {'차':>7s}")
        for k, name in enumerate(ARM):
            print(f"  {name:8s} {sim[k]:7.2f} {real[k]:7.2f} {sim[k]-real[k]:+7.2f}")
        print("  ※참고값(09.01 우팔): KUKA 0.429 · 정합 후 팔 0.051")
    else:
        print("\n[동특성] 결과 없음")
        ok = False

    print("\n" + ("✅ 합격" if ok else "❌ 기준 미달 — 위 항목 확인"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True,
                        choices=["fit", "apply", "verify", "report", "all"])
    parser.add_argument("--execute", action="store_true",
                        help="apply 가 실제로 파일을 고친다")
    parser.add_argument("--j-sim", type=lambda s: [float(x) for x in s.split(",")],
                        default=None, help="sim 관절 관성 7개. 없으면 내장 실측값")
    args = parser.parse_args()

    if args.stage in ("fit", "all"):
        stage_fit(args)
    if args.stage in ("apply", "all"):
        if args.stage == "all":
            args.execute = True
        stage_apply(args)
    if args.stage in ("verify", "all"):
        stage_verify(args)
    if args.stage in ("report", "all"):
        return stage_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
