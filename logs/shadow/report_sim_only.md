# 그림자 판정 — sim_fab_test16_gcON.npz

스텝 1200  ·  step_dt 0.0200 s  ·  중력보상 on
fabrics_sim /tmp/claude-1000/-home-user-rl-ws/f383c1ff-711d-4fa2-a90c-19a29d9f3540/scratchpad/hdgp_bc86ca5/source/FABRICS/src/fabrics_sim/__init__.py

## L1 — Fabrics attractor 가 지령을 실현하나 (FK(fabric_q) vs palm 지령)
  위치 mean   11.89  p95   49.31  max  401.97  mm
  자세 mean    3.86  p95   21.12  max   80.74  deg

## L2 — sim 물리가 fabric 해를 따라가나 (물리 TCP vs FK)
  위치 mean    3.46  p95   18.78  max  116.55  mm

## sim 관절 추종오차 (kp 400 — 실기 대비 기준선)
  관절          mean[mrad]      p95      max
  l_aj_1           29.81    67.58   205.38
  l_aj_2           31.79    44.50   217.06
  l_aj_3           21.67    79.13   224.01
  l_aj_4           18.81    57.11   158.58
  l_aj_5           10.03    47.99   165.90
  l_aj_6            9.62    18.54    90.07
  l_aj_7           15.24    40.58   164.16

## 정책이 요구하는 것 (실기 능력과 대볼 값)
  관절          mean[rad/s]      p95      max
  l_aj_1            0.093    0.632    3.203
  l_aj_2            0.057    0.209    3.930
  l_aj_3            0.085    0.735    3.370
  l_aj_4            0.057    0.361    1.942
  l_aj_5            0.066    0.159    2.898
  l_aj_6            0.051    0.318    1.716
  l_aj_7            0.088    0.512    2.428
  palm 지령 이동량  mean 0.00  p95 0.00  max 0.00 mm/step

## 중력 처짐 보상분 (★sim 강성 400 기준으로 상한이 잡혀 있다)
  관절          mean[mrad]      max
  l_aj_1           27.05    95.62
  l_aj_2           30.88    80.46
  l_aj_3           18.21    67.50
  l_aj_4           16.68    67.50
  l_aj_5            4.80    17.50
  l_aj_6            7.50    17.50
  l_aj_7            9.74    17.50

실기 csv 없음 — L3·지연·지터는 재생 뒤에 채운다.