#!/usr/bin/env python3
"""리셋 npz 를 **UDP 로 직접** Isaac 미러에 재생 — ROS 무경유 = 실기 절대 안 움직인다.

`shadow_replay --execute` 는 JTC 로도 발행하므로 "sim 에서만 보고 싶다"에 못 쓴다.
이 피더는 `probe_sim_follower.py` 의 패킷(v1 `<Id35f>`)만 만든다.

  python3 npz_udp_feed.py --sim logs/shadow/reset_both/reset_right_v2.npz --side right
  # --rate-scale 0.5 · --loop 로 반복 재생
"""

from __future__ import annotations

import argparse

import socket
import struct
import time
from pathlib import Path

import numpy as np

MAGIC = 0x5A2B10
FMT = "<Id35f"
NAN = float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=47321)
    parser.add_argument("--rate-scale", type=float, default=0.5)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    data = np.load(args.sim, allow_pickle=False)
    arm = data["arm_target"][:, 0]
    grip = data["grip_cmd"][:, 0]
    dt = float(data["meta_step_dt"][0]) / args.rate_scale
    n = arm.shape[0]
    print(f"{args.sim.name}: {n}프레임 · 발행 {1/dt:.1f} Hz · 총 {n*dt:.1f} s"
          f" · side={args.side} (반대편 채널은 NaN=유지)")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        t0 = time.monotonic()
        for i in range(n):
            l7, r7, g1, h20 = [NAN] * 7, [NAN] * 7, NAN, [NAN] * 20
            if args.side == "left":
                l7 = [float(v) for v in arm[i]]
                g1 = float(grip[i][0])
            else:
                r7 = [float(v) for v in arm[i]]
                h20 = ([float(v) for v in grip[i]] + [NAN] * 20)[:20]
            payload = l7 + r7 + [g1] + h20
            sock.sendto(struct.pack(FMT, MAGIC, time.time(), *payload), (args.host, args.port))
            # 벽시계 페이싱 — 밀리면 건너뛰지 않고 따라붙는다(지령 프레임은 전부 보낸다)
            rest = t0 + (i + 1) * dt - time.monotonic()
            if rest > 0:
                time.sleep(rest)
            if i % 200 == 0:
                print(f"  {i}/{n}", flush=True)
        print("재생 완료" + (" — 반복" if args.loop else ""))
        if not args.loop:
            return 0
        time.sleep(2.0)


if __name__ == "__main__":
    raise SystemExit(main())
