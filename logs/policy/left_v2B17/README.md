# left v2B17 — S2R 배포 정책 교체분 (사용자 지정, 08.31)

- 원본: `hdgp/log/checkpoints_keep/v2B17_*.pth` (B 변형 재학습 트랙, 옆 세션 산출)
- 보존 3종 (md5 사용자 제시값과 대조 완료):
  | 파일 | md5 | 비고 |
  |---|---|---|
  | v2B17_ep800_S997.pth  | 6dbbe7c8… | succ 99.7 — **배포 기본값** |
  | v2B17_ep1200_S989.pth | 04d36220… | succ 98.9 |
  | v2B17_ep2000_last.pth | da0c6a33… | last |
- 계약(체크포인트 실측): **obs 49 / action 7 / MLP · RNN 없음** — v2H_wide 와 전 층 shape 일치 → 배포 스택 드롭인.
- ⚠ **런 cfg dump(params/) 미확보** — 로컬·vision-3090 어디에도 런 디렉토리가 없다
  (keep 본만 존재). m1 교훈([[checkpoint-replay-needs-run-cfg]]): env cfg 가 현
  소스 기본값과 다르면(예: 테이블 z 재학습 변경분) 재생 정합이 조용히 깨진다.
  → 옆 세션(학습 주체)에서 params/{agent,env}.yaml dump 를 받아 여기 채울 것.
- 교체 전 정책: left_v2H_wide (프로필 주석 참조)
