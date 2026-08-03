#!/usr/bin/env bash
# net_preflight.sh — 5070ti 유선 배선/IP 정합 사전검사 (랜 오배선 재발 방지, 2026-08-03)
#
# 배경: 두 NIC 의 IP 설정이 물리 배선과 교차되어 손(169.254.186.72)으로 갈 패킷이
#       카메라/vision 랜선으로 나가던 사고. 스택 기동 전 이 스크립트로 배선 정합을 확인한다.
#
# 정합 기준(2026-08-03 확정 배선):
#   - 메인보드 NIC enp0s31f6        = vision-3090 직결   (192.168.100.2 ↔ peer .1)
#   - USB NIC     enxb0386cf2c43a   = 손(DG-5F) USB 허브 (169.254.186.100 ↔ peer .72)
#
# 사용: bash net_preflight.sh          # 0=전부 정상, 1=실패 항목 있음
set -u

VISION_IF=enp0s31f6
VISION_IP=192.168.100.2
VISION_PEER=192.168.100.1
HAND_IF=enxb0386cf2c43a
HAND_IP=169.254.186.100
HAND_PEER=169.254.186.72

fail=0
ok()   { printf '  ✓ %s\n' "$1"; }
bad()  { printf '  ✗ %s\n' "$1"; fail=1; }
warn() { printf '  ⚠ %s\n' "$1"; }

echo "[1/4] 인터페이스 존재/링크"
for ifc in "$VISION_IF" "$HAND_IF"; do
    if ! ip link show "$ifc" >/dev/null 2>&1; then
        bad "$ifc 인터페이스 없음 (USB 어댑터 분리? 이름 변경?)"
        continue
    fi
    state=$(ip -br link show "$ifc" | awk '{print $2}')
    if [ "$state" = "UP" ]; then ok "$ifc 링크 UP"; else bad "$ifc 링크 $state (케이블/허브/상대 전원 확인)"; fi
done

echo "[2/4] IP 배치 (NIC↔서브넷 교차 검사)"
vision_addrs=$(ip -br addr show "$VISION_IF" 2>/dev/null)
hand_addrs=$(ip -br addr show "$HAND_IF" 2>/dev/null)
case "$vision_addrs" in *"$VISION_IP"*) ok "$VISION_IF = $VISION_IP" ;; *) bad "$VISION_IF 에 $VISION_IP 없음 (nmcli con up vision-link)" ;; esac
case "$hand_addrs"   in *"$HAND_IP"*)   ok "$HAND_IF = $HAND_IP"     ;; *) bad "$HAND_IF 에 $HAND_IP 없음 (nmcli con up hand-link)"   ;; esac
case "$vision_addrs" in *169.254.186.*) bad "★교차 감지: 손 서브넷이 $VISION_IF(메인보드)에 있음 — 사고 원인 재발" ;; esac
case "$hand_addrs"   in *192.168.100.*) bad "★교차 감지: vision 서브넷이 $HAND_IF(USB)에 있음" ;; esac

echo "[3/4] 상대 도달성 (인터페이스 강제 지정 ping)"
if ping -c1 -W1 -I "$HAND_IF" "$HAND_PEER" >/dev/null 2>&1; then
    ok "손 $HAND_PEER ← $HAND_IF 응답"
else
    bad "손 $HAND_PEER 무응답 via $HAND_IF (손 전원/허브/모드스위치 확인)"
fi
if ping -c1 -W1 -I "$VISION_IF" "$VISION_PEER" >/dev/null 2>&1; then
    ok "vision-3090 $VISION_PEER ← $VISION_IF 응답"
else
    bad "vision-3090 $VISION_PEER 무응답 via $VISION_IF (vision 쪽 IP/케이블 확인)"
fi

echo "[4/4] 역방향 오배선 검사 (잘못된 NIC 로 상대가 보이면 배선 교차)"
if ping -c1 -W1 -I "$VISION_IF" "$HAND_PEER" >/dev/null 2>&1; then
    bad "★손($HAND_PEER)이 $VISION_IF(메인보드)에서 응답 — 케이블이 교차되어 있음"
else
    ok "손은 $VISION_IF 에서 안 보임 (정상)"
fi
if ping -c1 -W1 -I "$HAND_IF" "$VISION_PEER" >/dev/null 2>&1; then
    bad "★vision($VISION_PEER)이 $HAND_IF(USB)에서 응답 — 케이블이 교차되어 있음"
else
    ok "vision 은 $HAND_IF 에서 안 보임 (정상)"
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "PASS — 배선/IP 정합 정상. 스택 기동 진행."
else
    echo "FAIL — 위 ✗ 항목 해결 전 스택 기동 금지 (런북 §0-1 참고)."
fi
exit "$fail"
