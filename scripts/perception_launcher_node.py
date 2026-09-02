#!/usr/bin/env python3
"""노드 1 — 로컬에서 vision-3090 의 인지 체인을 켜고 끈다.

  /perception/cmd   (std_msgs/String JSON)  ← perception_ctl.py
    {"op":"start","objects":["shaker_closed","cup_big_s100"],"viewer":true}
    {"op":"stop","camera":false} · {"op":"viewer","on":false}
  /perception/status (std_msgs/String JSON, 1 Hz)

원격 실행은 tailscale ssh(무비밀번호) 로 repo 의 scripts/vision/*.sh 를 부른다.
FP++ yaml 은 레지스트리에서 생성해 vision 의 log/fpp_params/<name>.yaml 에 써 넣는다.
카메라 hz 는 DDS 로 직접 잰다(두 PC 가 같은 LAN, domain 126).
실패는 status.error 로 드러낸다 — 조용한 재시도 없음.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from object_registry import load_registry, output_topic, render_fpp_yaml  # noqa: E402
from perception_launcher_core import (  # noqa: E402
    build_status, parse_command, parse_remote_status, plan_actions,
)

REMOTE_SIM2REAL = "/home/usr/rl_ws/sim2real"
REMOTE_PARAMS = f"{REMOTE_SIM2REAL}/log/fpp_params"
_SCRIPT_FOR = {"camera_up": "camera_up.sh", "camera_down": "camera_down.sh",
               "fpp_up": "fpp_up.sh", "fpp_down": "fpp_down.sh",
               "viewer_up": "viewer_up.sh", "viewer_down": "viewer_down.sh"}


class RemoteExec:
    def __init__(self, host: str, timeout_s: float = 120.0) -> None:
        self.host, self.timeout_s = host, timeout_s

    def _ssh(self, command: str, stdin: str | None = None) -> str:
        proc = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", self.host, command],
                              input=stdin, capture_output=True, text=True, timeout=self.timeout_s)
        if proc.returncode != 0:
            raise RuntimeError(f"ssh {self.host} '{command[:60]}…' rc={proc.returncode}: "
                               f"{(proc.stderr or proc.stdout).strip()[-300:]}")
        return proc.stdout

    def run(self, script: str, *args: str) -> str:
        quoted = " ".join(f"'{a}'" for a in args)
        return self._ssh(f"bash {REMOTE_SIM2REAL}/scripts/vision/{script} {quoted}")

    def put(self, text: str, remote_path: str) -> None:
        self._ssh(f"mkdir -p $(dirname '{remote_path}') && cat > '{remote_path}'", stdin=text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="vision-3090")
    ap.add_argument("--poll", type=float, default=5.0, help="원격 상태 폴링 주기(s)")
    args = ap.parse_args()
    registry = load_registry()
    remote = RemoteExec(args.host)

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo
    from std_msgs.msg import String

    class Launcher(Node):
        def __init__(self) -> None:
            super().__init__("perception_launcher")
            self._lock = threading.Lock()
            self._busy = False
            self._error: str | None = None
            self._state = None
            self._last_pose: dict[str, float] = {}
            self._cam_stamps: list[float] = []
            self._pub = self.create_publisher(String, "/perception/status", 10)
            self.create_subscription(String, "/perception/cmd", self._on_cmd, 10)
            self.create_subscription(CameraInfo, "/camera/camera/color/camera_info", self._on_cam, 5)
            for name in registry.names():
                self.create_subscription(PoseStamped, output_topic(name),
                                         lambda _m, n=name: self._last_pose.__setitem__(n, time.monotonic()), 10)
            self.create_timer(1.0, self._publish_status)
            self.create_timer(args.poll, self._poll_remote)
            self._poll_remote()

        def _on_cam(self, _msg) -> None:
            now = time.monotonic()
            self._cam_stamps = [t for t in self._cam_stamps if now - t < 2.0] + [now]

        def _poll_remote(self) -> None:
            if self._busy:
                return
            try:
                self._state = parse_remote_status(remote.run("status.sh"))
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as err:
                self._error = f"status: {err}"
                self.get_logger().error(self._error)

        def _publish_status(self) -> None:
            now = time.monotonic()
            ages = {n: (round(now - self._last_pose[n], 3) if n in self._last_pose else None)
                    for n in registry.names()}
            payload = build_status(self._state, len(self._cam_stamps) / 2.0, ages, self._busy, self._error)
            self._pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

        def _on_cmd(self, msg: String) -> None:
            try:
                cmd = parse_command(msg.data, registry)
            except ValueError as err:
                self._error = f"cmd: {err}"
                self.get_logger().error(self._error)
                return
            with self._lock:
                if self._busy:
                    self._error = "busy: previous command still running"
                    self.get_logger().warning(self._error)
                    return
                self._busy = True
            threading.Thread(target=self._execute, args=(cmd,), daemon=True).start()

        def _execute(self, cmd) -> None:
            try:
                self._error = None
                state = parse_remote_status(remote.run("status.sh"))
                actions = plan_actions(cmd, state)
                self.get_logger().info(f"{cmd.op}: {actions or '변경 없음'}")
                for action in actions:
                    self._do(action, cmd, state)
                self._state = parse_remote_status(remote.run("status.sh"))
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as err:
                self._error = f"{cmd.op}: {err}"
                self.get_logger().error(self._error)
            finally:
                self._busy = False

        def _do(self, action, cmd, state) -> None:
            kind = action[0]
            if kind == "fpp_up":
                name = action[1]
                path = f"{REMOTE_PARAMS}/{name}.yaml"
                remote.put(render_fpp_yaml(registry.get(name)), path)
                out = remote.run("fpp_up.sh", name, path)
            elif kind == "viewer_up":
                # viewer 단독 명령엔 물체 목록이 없다 — 떠 있는 컨테이너에서 이름을 되찾는다.
                names = list(cmd.objects) or [c[len("fpp_"):] for c, st in state.containers.items()
                                              if c.startswith("fpp_") and st.startswith("Up")]
                if not names:
                    raise RuntimeError("viewer_up: 물체가 없다 — start 로 먼저 컨테이너를 띄울 것")
                out = remote.run("viewer_up.sh", *names)
            elif kind == "fpp_down":
                out = remote.run("fpp_down.sh", action[1])
            else:
                out = remote.run(_SCRIPT_FOR[kind])
            self.get_logger().info(f"  {action} → {out.strip().splitlines()[-1] if out.strip() else 'ok'}")

    rclpy.init()
    node = Launcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
