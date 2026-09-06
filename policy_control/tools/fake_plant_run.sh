#!/usr/bin/env bash
# M8 — 무하드웨어 폐루프 한 판.
#   MODE=chain(기본): fake 플랜트 + 체인 3노드 + pd(execute) + episode_ctl + judge + status csv (좌 v2B25 정책).
#   MODE=pd         : 제어 전용(asset 계약) — fake 플랜트(계약 모드) + pd(execute) 만: engage → goto_home →
#                     pd_selftest(hold + 관절 2개 ±0.1 rad 램프 0.1 rad/s) → release. 팔 SIDE=left|right|both.
#   ROS_DOMAIN_ID 는 반드시 실기 도메인(0/unset)이 아니어야 한다(launch 가 거부한다).
#   usage: ROS_DOMAIN_ID=99 policy_control/tools/fake_plant_run.sh [steps] [logdir]
#          MODE=pd SIDE=left ROS_DOMAIN_ID=97 policy_control/tools/fake_plant_run.sh 0 logs/policy_control/fake_dg5fm_left_run1
#   env(MODE=pd): SIDE, ROBOT(기본 dg5f_m_<SIDE>_fake / bi), CONTRACT(기본 asset_openarm_dg5f-m_bi_rl), PD_CONFIG(기본 dg5f_m_fake),
#                 STEP_JOINTS(기본 <p>_aj_3,<p>_aj_6 — 홈 0 에서 j4 는 하한 0 이라 −스텝이 한계 가드에 걸린다), AMPS(기본 0.1), HOLD_S(기본 3), DWELL_S(기본 2)
set -o pipefail
cd "$(dirname "$0")/../.."
source /opt/ros/humble/setup.bash && . .venv/bin/activate
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-99}"
MODE="${MODE:-chain}"
STEPS="${1:-250}"; LOG="${2:-logs/policy_control/fake_$(date +%m%d_%H%M%S)}"; mkdir -p "$LOG"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -- -"$p" 2>/dev/null; done; sleep 1; for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -9 -- -"$p" 2>/dev/null; done; }
trap cleanup EXIT
launch_bg() { setsid "$@" & PIDS+=($!); }     # 프로세스 그룹으로 띄워 launch 의 자식까지 한 번에 정리한다
wait_status() {   # $1 = 기대 status 토픽 수
  for i in $(seq 1 60); do
    n=$(timeout 3 ros2 topic list 2>/dev/null | grep -c "policy_control/status/" || true)
    [ "$n" -ge "$1" ] && break; sleep 1
  done
  echo "[run] status topics: $n"
}
trigger() {   # $1 = 서비스 이름 → 응답 출력, 실패면 rc 1
  out=$(timeout 120 ros2 service call /policy_control/pd/$1 std_srvs/srv/Trigger "{}" 2>&1 | tr -d '\n'); echo "[run] $1: $out"
  echo "$out" | grep -q "success=True" ; return $?
}

if [ "$MODE" = "pd" ]; then
  SIDE="${SIDE:-left}"; CONTRACT="${CONTRACT:-logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json}"
  PD_CONFIG="${PD_CONFIG:-dg5f_m_fake}"
  if [ "$SIDE" = "both" ]; then ROBOT="${ROBOT:-dg5f_m_bi_fake}"; SIDES="right left"; else ROBOT="${ROBOT:-dg5f_m_${SIDE}_fake}"; SIDES="$SIDE"; fi
  echo "[run] MODE=pd · domain $ROS_DOMAIN_ID · side $SIDE · robot $ROBOT · pd_config $PD_CONFIG · log $LOG"
  launch_bg ros2 launch policy_control/launch/fake_plant.launch.py side:=$SIDE robot:=$ROBOT contract:=$CONTRACT \
      plant_model:=${PLANT_MODEL:-pd} plant_friction:=${PLANT_FRICTION:-1.0} > "$LOG/fake_plant.log" 2>&1
  sleep 4
  launch_bg ros2 launch policy_control/launch/pd_controller.launch.py contract:=$CONTRACT robot:=$ROBOT pd_config:=$PD_CONFIG \
      sides:=$SIDE execute:=true stage:=${PD_STAGE:-full} fake:=true use_source:=true > "$LOG/pd.log" 2>&1
  wait_status 1
  SEC=$(( 40 + 30 * $(echo $SIDES | wc -w) ))
  launch_bg python policy_control/tools/status_to_csv.py --seconds $SEC --out "$LOG/status.csv" --jsonl "$LOG/status.jsonl" > "$LOG/status_summary.txt" 2>&1
  for s in $SIDES; do
    launch_bg python policy_control/tools/chain_recorder.py --contract $CONTRACT --side $s --seconds $SEC --out "$LOG/chain_$s.npz" > "$LOG/recorder_$s.log" 2>&1
  done
  sleep 2
  RC=0
  trigger engage || RC=1
  [ $RC -eq 0 ] && { trigger goto_home || RC=1; }
  if [ $RC -eq 0 ]; then
    for s in $SIDES; do
      p=${s:0:1}; J="${STEP_JOINTS:-${p}_aj_3,${p}_aj_6}"
      python policy_control/tools/pd_selftest.py --contract $CONTRACT --robot policy_control/config/robots/$ROBOT.yaml \
          --side $s --joints $J --amplitudes ${AMPS:-0.1} --hold-s ${HOLD_S:-3} --dwell-s ${DWELL_S:-2} --execute \
          --out "$LOG/selftest_$s.json" 2>&1 | tee "$LOG/selftest_$s.log" || RC=1
    done
  fi
  trigger release || RC=1
  sleep 2
  python policy_control/tools/status_to_csv.py --help > /dev/null   # (no-op) keep venv warm
  for p in "${PIDS[@]:2}"; do wait "$p" 2>/dev/null; done            # status/recorder 가 끝날 때까지
  python - "$LOG" $SIDES <<'PY' | tee "$LOG/summary.txt"
import json, sys, numpy as np
from pathlib import Path
log = Path(sys.argv[1]); sides = sys.argv[2:]
rows = [json.loads(l) for l in (log / "status.jsonl").read_text().splitlines() if l.strip()]
pd = [r for r in rows if r.get("topic") == "pd"]
holds = [r for r in pd if r.get("phase") == "HOLD"]
phases = [r["phase"] for r in pd]
print(f"pd status rows {len(pd)} · phases {sorted(set(phases))} · HOLD rows {len(holds)} (guards 0 ⇔ 0) · not-ok rows {sum(1 for r in pd if not r.get('ok'))}")
for s in sides:
    d = np.load(log / f"chain_{s}.npz")
    js, t = d["js_q"], d["t_js"]
    step = np.abs(np.diff(js, axis=0)).max() if len(js) > 1 else float('nan')
    tau = d["app_tau"]; q_app = d["app_q"]
    # τ_ff at home = applied effort while applied q is within 0.02 rad of home (zeros) and after engage blend
    near = np.all(np.abs(q_app) < 0.02, axis=1) if len(q_app) else np.zeros(0, bool)
    tau_home = tau[near] if near.any() else tau
    print(f"[{s}] joint_states samples {len(js)} · max |Δq| per sample {step:.4f} rad · applied rows {len(tau)} · "
          f"τ_ff@home (median over {int(near.sum())} rows) {np.round(np.median(tau_home, axis=0), 3).tolist() if len(tau_home) else '-'} N·m · "
          f"final |q| max {np.abs(js[-1]).max():.4f}" if len(js) else f"[{s}] no joint_states")
    st = json.loads((log / f"selftest_{s}.json").read_text()) if (log / f"selftest_{s}.json").exists() else None
    if st:
        print(f"[{s}] selftest ok {st['ok']} · hold drift worst {max(abs(v) for v in st['hold'].values()):.4f} rad · steps "
              + ", ".join(f"{v['joint']} {v['commanded']:+.2f}→{v['measured']:+.4f} (ratio {v['ratio']:.2f}{'' if v['ok'] else ' ✗'})" for v in st['steps']))
PY
  echo "[run] rc=$RC"; cat "$LOG/status_summary.txt"
  exit $RC
fi

# ------------------------------------------------------------------ MODE=chain (좌 v2B25 정책 폐루프)
CONTRACT=logs/policy/left_v2B25/deploy_contract.json
ROBOT=policy_control/config/robots/left_gripper_fake.yaml
echo "[run] domain $ROS_DOMAIN_ID · steps $STEPS · log $LOG"
launch_bg ros2 launch policy_control/launch/fake_plant.launch.py side:=left contract:=$PWD/$CONTRACT plant_model:=${PLANT_MODEL:-pd} plant_friction:=${PLANT_FRICTION:-1.0} > "$LOG/fake_plant.log" 2>&1
sleep 4
launch_bg ros2 launch policy_control/launch/pd_controller.launch.py contract:=$CONTRACT robot:=$ROBOT pd_config:=policy_control/config/pd_left_fake.yaml execute:=true stage:=${PD_STAGE:-full} fake:=true use_source:=true > "$LOG/pd.log" 2>&1
launch_bg ros2 launch policy_control/launch/policy_chain.launch.py contract:=$CONTRACT robot:=$ROBOT device:=cuda:0 fake:=true use_source:=true > "$LOG/chain.log" 2>&1
echo "[run] waiting for status topics…"
wait_status 4
launch_bg python policy_control/tools/status_to_csv.py --seconds $(( STEPS / 50 + 25 )) --out "$LOG/status.csv" --policy-dt 0.02 --jsonl "$LOG/status.jsonl" > "$LOG/status_summary.txt" 2>&1
launch_bg python policy_control/tools/episode_judge.py --contract $CONTRACT --seconds $(( STEPS / 50 + 20 )) --out "$LOG/verdict.json" > "$LOG/judge.log" 2>&1
launch_bg python policy_control/tools/chain_recorder.py --contract $CONTRACT --seconds $(( STEPS / 50 + 20 )) --out "$LOG/chain.npz" > "$LOG/recorder.log" 2>&1
sleep 2
python policy_control/tools/episode_ctl.py --steps "$STEPS" --execute --approve pd_engage --approve pd_goto_home --approve ep_start --service-timeout 90 --phase-timeout 90 2>&1 | tee "$LOG/episode_ctl.log"
RC=${PIPESTATUS[0]}
wait "${PIDS[3]}" 2>/dev/null; wait "${PIDS[4]}" 2>/dev/null
echo "[run] episode_ctl rc=$RC"; cat "$LOG/status_summary.txt"; cat "$LOG/verdict.json" 2>/dev/null
exit $RC
