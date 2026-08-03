# 진행 원장 — cup_pose 라이브 노드 + extrinsics 캘리브

계획: sim2real/docs/superpowers/plans/2026-07-27-cup-pose-live-node-extrinsics.md
- Task 1 (SP1 anchor 기하 헬퍼, vision-3090): complete (commit dcf492d, review clean; minor: xyxy 미주석·empty-box edge, 계획유래)
- Task 2 (SP1 anchored ROS 노드, vision-3090): complete (commit f3ac163, review clean; minor: except BaseException·global fd suppress·no _callback test)
- 최종 통합 리뷰(opus): Ready to merge. Important 1건(t_base_qr rpy Euler) 수정 완료 bd96abc. Minor는 fast-follow.
- Task 3 (SP2 QR 캘리브 수학, pc5090): complete (commits b577c66..126bbde, review clean)
- Task 4 (SP2 캘리브 CLI, pc5090): complete (commits 126bbde..6d854ed, review clean)

실행 순서: 로컬 SP2(3,4) 먼저 → 원격 SP1(1,2, scp+ssh).

## 후속(라이브 발견)
- 마커=QR 아님, ArUco DICT_6X6 id 0. 검출부 QR→ArUco 교체 f2c40b4 (review clean). 실프레임 라이브 검증=인식+pose 0.062px 성공. pose수학 무변경.
- 남은 하드웨어: (a) ArUco 실측 크기 (b) T_base_qr 실측 → 실제 T_base_cam 기록. (c) FP++ 노드→relay→/cup_pose 라이브.
