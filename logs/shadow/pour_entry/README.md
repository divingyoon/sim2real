# pour_entry — 양팔 파지 리허설 산출물

## 유효 (2026-09-02 폐루프)

| 파일 | 내용 |
|---|---|
| `bimanual_closedloop.gif` | ★**정본** — 정책 2개가 통합 pour 씬을 직접 폐루프 제어. preset → 좌 v2B25 가 shaker 파지·리프트(105스텝) → 유지 → (물리 dt 100→120Hz 전환) → 우 E1 이 cup_big_s100 파지·리프트(166스텝) → 양팔 유지. 텔레포트·컵 고정 없음, 파지는 접촉이 만든 것. 러너: `hdgp/scripts/probes/probe_bimanual_closedloop.py` |
| `stream_left_v2b25.npz` / `stream_right_e1_v2.npz` | 각 정책의 자기 env 폐루프 기록 (v2 스키마: actions·obs·지령·실측·gate/latch). 러너의 goal·스폰·재현 대조 기준. 회귀: `sim2real/scripts/test_stream_consistency.py` |
| `state_left_end_v2.json` / `state_right_end_v2.json` | 파지 종료 상태 (실측+지령 — 지령이 파지력) |

## 폐기 (미러 시대 — 물리 우회, 교훈 보존용)

`bimanual_sequence.gif` · `left/right_grasp_carry.gif` · `stream_right_e1.npz` ·
`state_*_end.json`(v2 아님) — **미러 방식**(매 프레임 관절 텔레포트 + 컵 root 고정)의
산출물. 09.02 사용자 관찰로 반증됨: 손가락 관통·반대팔 360° 회전·컵 순간 부착이
전부 그 우회의 서명이었다. 경위는 `probe_bimanual_mirror.py` 독스트링 참조.
