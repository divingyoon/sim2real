#!/usr/bin/env bash
# 제어 전용(asset 계약) fake 폐루프 — fabric direct 모드까지 한 팔씩:
#   fake 플랜트 + pd(execute) + [episode_master + fabric_node(direct)] → engage → goto_home → episode reset/start
#   → palm_cmd(+5 cm z) → hand_cmd(0.3 → 0) → episode stop → release. 요약은 summary.txt.
#   usage: SIDE=left ROS_DOMAIN_ID=96 policy_control/tools/fake_direct_run.sh [logdir]
#   env: SIDE(left|right), ROBOT(기본 dg5f_m_<SIDE>_fake), CONTRACT(기본 asset_openarm_dg5f-m_bi_rl), PD_CONFIG(기본 dg5f_m_fake),
#        DZ(기본 0.05 m palm z 델타), CLOSE(기본 0.3), PLANT_MODEL(pd|rate)
set -o pipefail
cd "$(dirname "$0")/../.."
source /opt/ros/humble/setup.bash && . .venv/bin/activate
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-96}"
[ "$ROS_DOMAIN_ID" = "0" ] && { echo "✗ ROS_DOMAIN_ID 0 은 실기 도메인이다"; exit 3; }
SIDE="${SIDE:-left}"; ROBOT="${ROBOT:-dg5f_m_${SIDE}_fake}"; PD_CONFIG="${PD_CONFIG:-dg5f_m_fake}"
CONTRACT="${CONTRACT:-logs/policy/asset_openarm_dg5f-m_bi_rl/deploy_contract.json}"
LOG="${1:-logs/policy_control/fake_direct_${SIDE}_$(date +%m%d_%H%M%S)}"; mkdir -p "$LOG"
DZ="${DZ:-0.05}"; CLOSE="${CLOSE:-0.3}"; SEC=110
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -- -"$p" 2>/dev/null; done; sleep 1; for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -9 -- -"$p" 2>/dev/null; done; }
trap cleanup EXIT
launch_bg() { setsid "$@" & PIDS+=($!); }
wait_status() { for i in $(seq 1 90); do n=$(timeout 3 ros2 topic list 2>/dev/null | grep -c "policy_control/status/" || true); [ "$n" -ge "$1" ] && break; sleep 1; done; echo "[run] status topics: $n"; }
call() {   # ros2 CLI 가 가끔 응답을 못 받고 끝난다(요청은 처리됨) → 응답이 없을 때만 한 번 더 시도
  for attempt in 1 2; do
    out=$(timeout 120 ros2 service call /policy_control/$1 std_srvs/srv/Trigger "{}" 2>&1 | tr -d '\n'); echo "[run] $1: ${out:0:220}"
    echo "$out" | grep -q "response:" && break
    echo "[run] $1: no response captured (attempt $attempt)"
  done
  echo "$out" | grep -q "success=True"
}
echo "[run] direct · domain $ROS_DOMAIN_ID · side $SIDE · robot $ROBOT · pd_config $PD_CONFIG · log $LOG"
launch_bg ros2 launch policy_control/launch/fake_plant.launch.py side:=$SIDE robot:=$ROBOT contract:=$CONTRACT plant_model:=${PLANT_MODEL:-pd} > "$LOG/fake_plant.log" 2>&1
sleep 3
launch_bg ros2 launch policy_control/launch/pd_controller.launch.py contract:=$CONTRACT robot:=$ROBOT pd_config:=$PD_CONFIG sides:=$SIDE execute:=true stage:=${PD_STAGE:-full} fake:=true use_source:=true > "$LOG/pd.log" 2>&1
launch_bg ros2 launch policy_control/launch/policy_chain.launch.py contract:=$CONTRACT robot:=$ROBOT side:=$SIDE device:=cuda:0 fake:=true use_source:=true > "$LOG/chain.log" 2>&1
wait_status 3
launch_bg python policy_control/tools/status_to_csv.py --seconds $SEC --out "$LOG/status.csv" --jsonl "$LOG/status.jsonl" > "$LOG/status_summary.txt" 2>&1
launch_bg python policy_control/tools/chain_recorder.py --contract $CONTRACT --side $SIDE --seconds $SEC --out "$LOG/chain_$SIDE.npz" > "$LOG/recorder_$SIDE.log" 2>&1
sleep 2; RC=0
call pd/engage || RC=1
[ $RC -eq 0 ] && { call pd/goto_home || RC=1; }
if [ $RC -eq 0 ]; then
  call episode/reset || RC=1; sleep 2
  call episode/start || RC=1; sleep 2
  python policy_control/tools/palm_cmd.py --rel 0 0 0 --dry-run 2>&1 | tee "$LOG/palm_before.txt"
  python policy_control/tools/palm_cmd.py --rel 0 0 $DZ --hold 1.0 2>&1 | tee "$LOG/palm_cmd.txt" || RC=1
  sleep 8
  python policy_control/tools/palm_cmd.py --rel 0 0 0 --dry-run 2>&1 | tee "$LOG/palm_after.txt"
  python policy_control/tools/hand_cmd.py --contract $CONTRACT --side $SIDE --close $CLOSE --hold 1.0 > "$LOG/hand_close.txt" 2>&1 || RC=1
  sleep 5
  python policy_control/tools/hand_cmd.py --contract $CONTRACT --side $SIDE --close 0 --hold 1.0 > "$LOG/hand_open.txt" 2>&1 || RC=1
  sleep 4
  call episode/stop || RC=1
fi
call pd/release || RC=1
sleep 2
for p in "${PIDS[@]:3}"; do wait "$p" 2>/dev/null; done
python - "$LOG" "$SIDE" "$DZ" <<'PY' | tee "$LOG/summary.txt"
import json, re, sys, numpy as np
from pathlib import Path
log, side, dz = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
rows = [json.loads(l) for l in (log / "status.jsonl").read_text().splitlines() if l.strip()]
pd = [r for r in rows if r.get("topic") == "pd"]; fab = [r for r in rows if r.get("topic") == "fabric"]
holds = [r for r in pd if r.get("phase") == "HOLD"]
print(f"pd rows {len(pd)} · phases {sorted({r['phase'] for r in pd})} · HOLD rows {len(holds)} · not-ok {sum(1 for r in pd if not r.get('ok'))}"
      + (f" · first HOLD reasons {holds[0].get('reasons')}" if holds else ""))
print(f"fabric rows {len(fab)} · running rows {sum(1 for r in fab if r.get('running'))} · not-ok {sum(1 for r in fab if not r.get('ok'))} · modes {sorted({str(r.get('mode')) for r in fab})} · sides {sorted({str(r.get('side')) for r in fab})}")
def palm(txt):
    m = re.search(r"current\s+(.*)", (log / txt).read_text()); return m.group(1).strip() if m else "-"
print(f"palm before {palm('palm_before.txt')}\npalm after  {palm('palm_after.txt')}  (commanded dz {dz:+.3f} m)")
d = np.load(log / f"chain_{side}.npz"); keys = list(d.files)
js = d["js_q"] if "js_q" in keys else None
if js is not None and len(js):
    print(f"joint_states {len(js)} samples · max|Δq|/sample {np.abs(np.diff(js, axis=0)).max():.4f} rad · final q {np.round(js[-1], 3).tolist()}")
tq = d["target_q"]
if len(tq):
    print(f"joint_target rows {len(tq)} · arm target min {np.round(np.nanmin(tq, 0), 3).tolist()} max {np.round(np.nanmax(tq, 0), 3).tolist()}")
hq = d["hand_target_q"] if "hand_target_q" in keys else np.zeros((0, 0))
if hq.size:
    names = list(d["hand_target_names"]); j2 = [i for i, n in enumerate(names) if n.endswith("index_2")]
    print(f"hand target rows {len(hq)} · index_2 min {np.nanmin(hq[:, j2]):.3f} max {np.nanmax(hq[:, j2]):.3f} (close 0.3 → open+0.3·(grip−open)) · final {np.round(hq[-1][:4], 3).tolist()}…")
if "app_q" in keys and len(d["app_q"]) and len(js):
    idx = np.searchsorted(d["t_js"], d["t_app"]).clip(1, len(js) - 1); err = np.abs(d["app_q"] - js[idx])
    print(f"tracking |q*-q| p50 {np.percentile(err, 50):.4f} p95 {np.percentile(err, 95):.4f} max {err.max():.4f} rad · τ_ff median {np.round(np.median(d['app_tau'], 0), 3).tolist()}")
PY
echo "[run] rc=$RC"
