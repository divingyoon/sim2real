#!/usr/bin/env python3
"""Tkinter calibration UI for two absolute-encoder DYNAMIXEL motors."""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox


TICK_MAX = 4095
PROTOCOL = 2.0

ADDR_OPERATING_MODE = 11
ADDR_HOMING_OFFSET = 20
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

OP_MODE_EXTENDED_POSITION = 4
TORQUE_OFF = 0
TORQUE_ON = 1

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "head_dynamixel_calibration.yaml"
DEFAULT_PORT_CANDIDATES = ("/dev/ttyACM0", "/dev/ttyUSB0")


@dataclass
class MotorCalibration:
    name: str
    dxl_id: int
    min_deg: float
    max_deg: float
    inverted: bool = False


def deg_to_tick(deg: float) -> int:
    return round((float(deg) + 180.0) / 360.0 * TICK_MAX)


def tick_to_deg(tick: int) -> float:
    return int(tick) / TICK_MAX * 360.0 - 180.0


def equivalent_signed_deg(deg: float) -> float:
    wrapped = ((float(deg) + 180.0) % 360.0) - 180.0
    if wrapped == -180.0 and float(deg) > 0:
        return 180.0
    return wrapped


def format_deg_with_equivalent(deg: float) -> str:
    equivalent = equivalent_signed_deg(deg)
    if abs(equivalent - float(deg)) < 0.05:
        return f"{deg:.1f}"
    return f"{deg:.1f} (={equivalent:.1f})"


def parse_motor_ids(text: str) -> tuple[int, int]:
    try:
        ids = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"ids must be comma-separated integers: {text!r}") from exc
    if len(ids) != 2:
        raise ValueError(f"ids must contain exactly two motor ids: {text!r}")
    if ids[0] == ids[1]:
        raise ValueError(f"motor ids must be unique: {ids}")
    for dxl_id in ids:
        if not 0 <= dxl_id <= 252:
            raise ValueError(f"motor id must be in 0..252: {dxl_id}")
    return ids


def motor_deg_from_ui_percent(calibration: MotorCalibration, percent: float) -> float:
    if not 0.0 <= percent <= 100.0:
        raise ValueError(f"ui percent must be in 0..100: {percent}")
    start, end = unwrap_calibrated_range(calibration.min_deg, calibration.max_deg)
    span = end - start
    fraction = percent / 100.0
    if calibration.inverted:
        fraction = 1.0 - fraction
    return start + span * fraction


def unwrap_calibrated_range(min_deg: float, max_deg: float) -> tuple[float, float]:
    start = float(min_deg)
    end = float(max_deg)
    if end < start:
        end += 360.0
    if end - start > 360.0:
        raise ValueError(
            f"calibrated range must fit within one directed turn: "
            f"{min_deg:.1f}..{max_deg:.1f}"
        )
    return start, end


def unwrap_target_into_range(target_deg: float, min_deg: float, max_deg: float) -> float:
    start, end = unwrap_calibrated_range(min_deg, max_deg)
    target = float(target_deg)
    while target < start:
        target += 360.0
    while target > end:
        target -= 360.0
    if not start <= target <= end:
        raise ValueError(
            f"target {target_deg:+.1f}deg is outside calibrated range "
            f"{min_deg:.1f}..{max_deg:.1f}"
        )
    return target


def unwrap_range_copy_for_current(
    current_deg: float, min_deg: float, max_deg: float
) -> tuple[float, float]:
    start, end = unwrap_calibrated_range(min_deg, max_deg)
    current = float(current_deg)
    for turn in range(-20, 21):
        copy_start = start + 360.0 * turn
        copy_end = end + 360.0 * turn
        if copy_start <= current <= copy_end:
            return copy_start, copy_end
    turn = round((current - start) / 360.0)
    return start + 360.0 * turn, end + 360.0 * turn


def unwrap_target_for_current(
    current_deg: float, target_deg: float, min_deg: float, max_deg: float
) -> float:
    copy_start, copy_end = unwrap_range_copy_for_current(
        current_deg, min_deg, max_deg
    )
    target = float(target_deg)
    while target < copy_start:
        target += 360.0
    while target > copy_end:
        target -= 360.0
    if not copy_start <= target <= copy_end:
        raise ValueError(
            f"target {target_deg:+.1f}deg is outside current directed range "
            f"{copy_start:.1f}..{copy_end:.1f}"
        )
    return target


def load_calibration_yaml(path: Path) -> dict[str, MotorCalibration]:
    if not path.exists():
        return {}
    text = path.read_text()
    try:
        import yaml
    except ModuleNotFoundError:
        data = _load_saved_calibration_without_yaml(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict) or not isinstance(data.get("motors"), dict):
        return {}

    motors: dict[str, MotorCalibration] = {}
    for name, raw in data["motors"].items():
        if not isinstance(raw, dict):
            continue
        motors[str(name)] = MotorCalibration(
            name=str(name),
            dxl_id=int(raw["id"]),
            min_deg=float(raw["min_deg"]),
            max_deg=float(raw["max_deg"]),
            inverted=bool(raw.get("inverted", False)),
        )
    return motors


def _load_saved_calibration_without_yaml(text: str) -> dict[str, object]:
    """Read the simple YAML shape written by save_calibration_yaml."""
    motors: dict[str, dict[str, object]] = {}
    current: dict[str, object] | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("  ") and line.endswith(":"):
            name = line.strip()[:-1]
            current = {}
            motors[name] = current
            continue
        if current is not None and line.startswith("    ") and ":" in line:
            key, value = line.strip().split(":", 1)
            value = value.strip()
            if value.lower() in ("true", "false"):
                current[key] = value.lower() == "true"
            elif key == "id":
                current[key] = int(value)
            else:
                current[key] = float(value)
    return {"motors": motors}


def save_calibration_yaml(
    path: Path,
    *,
    port: str,
    baud: int,
    motors: list[MotorCalibration],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"port: {port}",
        f"baud: {int(baud)}",
        "motors:",
    ]
    for motor in motors:
        lines.extend(
            [
                f"  {motor.name}:",
                f"    id: {int(motor.dxl_id)}",
                f"    min_deg: {float(motor.min_deg):.1f}",
                f"    max_deg: {float(motor.max_deg):.1f}",
                f"    inverted: {'true' if motor.inverted else 'false'}",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


class DynamixelUiBus:
    def __init__(self, port_name: str, baud: int):
        from dynamixel_sdk import PacketHandler, PortHandler

        self.port_name = port_name
        self.port = PortHandler(port_name)
        self.packet = PacketHandler(PROTOCOL)
        if not self.port.openPort():
            raise RuntimeError(f"failed to open port: {port_name}")
        try:
            if not self.port.setBaudRate(baud):
                raise RuntimeError(f"failed to set baud: {baud}")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self.port.closePort()

    def _check(self, comm_result: int, packet_error: int, context: str) -> None:
        from dynamixel_sdk import COMM_SUCCESS

        if comm_result != COMM_SUCCESS:
            raise RuntimeError(f"{context}: {self.packet.getTxRxResult(comm_result)}")
        if packet_error:
            raise RuntimeError(f"{context}: {self.packet.getRxPacketError(packet_error)}")

    def ping(self, dxl_id: int) -> None:
        _model, comm, err = self.packet.ping(self.port, dxl_id)
        self._check(comm, err, f"ping id={dxl_id}")

    def read_present_tick(self, dxl_id: int) -> int:
        value, comm, err = self.packet.read4ByteTxRx(
            self.port, dxl_id, ADDR_PRESENT_POSITION
        )
        self._check(comm, err, f"present position id={dxl_id}")
        return int(value)

    def write1(self, dxl_id: int, address: int, value: int, label: str) -> None:
        comm, err = self.packet.write1ByteTxRx(self.port, dxl_id, address, value)
        self._check(comm, err, f"{label} id={dxl_id}")

    def write4(self, dxl_id: int, address: int, value: int, label: str) -> None:
        comm, err = self.packet.write4ByteTxRx(self.port, dxl_id, address, value)
        self._check(comm, err, f"{label} id={dxl_id}")

    def configure_position_mode(self, dxl_id: int) -> None:
        self.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF, "torque off")
        self.write1(
            dxl_id,
            ADDR_OPERATING_MODE,
            OP_MODE_EXTENDED_POSITION,
            "operating mode",
        )
        self.write4(dxl_id, ADDR_PROFILE_ACCELERATION, 20, "profile acceleration")
        self.write4(dxl_id, ADDR_PROFILE_VELOCITY, 50, "profile velocity")
        self.write4(
            dxl_id,
            ADDR_GOAL_POSITION,
            self.read_present_tick(dxl_id),
            "seed goal position",
        )
        self.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON, "torque on")

    def move_absolute_deg(self, dxl_id: int, deg: float) -> None:
        self.write4(dxl_id, ADDR_GOAL_POSITION, deg_to_tick(deg), "goal position")

    def torque_off(self, dxl_id: int) -> None:
        self.write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF, "torque off")


class MotorFrame:
    def __init__(
        self,
        parent: tk.Widget,
        *,
        name: str,
        dxl_id: int,
        default_min: float,
        default_max: float,
    ):
        self.name = name
        self.dxl_id = dxl_id
        self.frame = tk.LabelFrame(parent, text=f"{name} motor id={dxl_id}", padx=8, pady=8)
        self.frame.pack(fill="x", padx=8, pady=6)

        self.present_var = tk.StringVar(value="present: not connected")
        self.min_var = tk.DoubleVar(value=default_min)
        self.max_var = tk.DoubleVar(value=default_max)
        self.target_var = tk.DoubleVar(value=default_min)
        self.inverted_var = tk.BooleanVar(value=False)

        tk.Label(self.frame, textvariable=self.present_var, anchor="w").pack(fill="x")
        self.slider = tk.Scale(
            self.frame,
            from_=default_min,
            to=default_max,
            orient="horizontal",
            resolution=0.1,
            variable=self.target_var,
            label="signed absolute encoder target deg",
        )
        self.slider.pack(fill="x")

        row = tk.Frame(self.frame)
        row.pack(fill="x")
        tk.Label(row, text="min deg").pack(side="left")
        tk.Entry(row, textvariable=self.min_var, width=8).pack(side="left", padx=4)
        tk.Label(row, text="max deg").pack(side="left")
        tk.Entry(row, textvariable=self.max_var, width=8).pack(side="left", padx=4)
        tk.Checkbutton(row, text="invert direction", variable=self.inverted_var).pack(
            side="left", padx=8
        )

    def calibration(self) -> MotorCalibration:
        return MotorCalibration(
            name=self.name,
            dxl_id=self.dxl_id,
            min_deg=float(self.min_var.get()),
            max_deg=float(self.max_var.get()),
            inverted=bool(self.inverted_var.get()),
        )

    def target_motor_deg(self) -> float:
        target_deg = float(self.target_var.get())
        calibration = self.calibration()
        try:
            return unwrap_target_into_range(
                target_deg,
                calibration.min_deg,
                calibration.max_deg,
            )
        except ValueError as exc:
            raise ValueError(
                f"{self.name} target {target_deg:+.1f}deg is outside "
                f"{calibration.min_deg:.1f}..{calibration.max_deg:.1f}"
            ) from exc

    def target_motor_deg_for_current(self, current_tick: int) -> float:
        target_deg = float(self.target_var.get())
        calibration = self.calibration()
        try:
            return unwrap_target_for_current(
                tick_to_deg(current_tick),
                target_deg,
                calibration.min_deg,
                calibration.max_deg,
            )
        except ValueError as exc:
            raise ValueError(
                f"{self.name} target {target_deg:+.1f}deg is outside "
                f"{calibration.min_deg:.1f}..{calibration.max_deg:.1f}"
            ) from exc

    def set_present(self, tick: int) -> None:
        deg = tick_to_deg(tick)
        self.present_var.set(
            f"present: tick={tick} deg={format_deg_with_equivalent(deg)}"
        )

    def set_current_as_min(self, tick: int) -> None:
        self.set_min_deg(round(tick_to_deg(tick), 1))

    def set_current_as_max(self, tick: int) -> None:
        self.set_max_deg(round(tick_to_deg(tick), 1))

    def set_min_deg(self, value: float) -> None:
        self.min_var.set(value)
        start, end = unwrap_calibrated_range(value, float(self.max_var.get()))
        self.slider.configure(from_=start, to=end)
        if float(self.target_var.get()) < value:
            self.target_var.set(start)

    def set_max_deg(self, value: float) -> None:
        self.max_var.set(value)
        start, end = unwrap_calibrated_range(float(self.min_var.get()), value)
        self.slider.configure(from_=start, to=end)
        if float(self.target_var.get()) > end:
            self.target_var.set(end)

    def load_calibration(self, calibration: MotorCalibration) -> None:
        self.min_var.set(calibration.min_deg)
        self.max_var.set(calibration.max_deg)
        start, end = unwrap_calibrated_range(calibration.min_deg, calibration.max_deg)
        self.slider.configure(from_=start, to=end)
        self.slider.configure(
            label=(
                "signed absolute encoder target deg "
                f"({format_deg_with_equivalent(start)} -> "
                f"{format_deg_with_equivalent(end)})"
            )
        )
        self.inverted_var.set(calibration.inverted)
        self.target_var.set(start)


class CalibrationApp:
    def __init__(self, args: argparse.Namespace):
        ids = parse_motor_ids(args.ids)
        args.port = resolve_port(args.port)
        self.args = args
        self.bus: DynamixelUiBus | None = None

        self.root = tk.Tk()
        self.root.title("DYNAMIXEL Head Calibration")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        top = tk.Frame(self.root, padx=8, pady=8)
        top.pack(fill="x")
        self.status_var = tk.StringVar(value="disconnected")
        tk.Label(top, text=f"port={args.port} baud={args.baud}").pack(anchor="w")
        tk.Label(top, textvariable=self.status_var).pack(anchor="w")

        saved = load_calibration_yaml(Path(args.output))
        self.pan = MotorFrame(
            self.root,
            name="pan",
            dxl_id=ids[0],
            default_min=-180.0,
            default_max=180.0,
        )
        self.tilt = MotorFrame(
            self.root,
            name="tilt",
            dxl_id=ids[1],
            default_min=-180.0,
            default_max=180.0,
        )
        self.frames = [self.pan, self.tilt]
        for motor_frame in self.frames:
            if motor_frame.name in saved:
                motor_frame.load_calibration(saved[motor_frame.name])

        buttons = tk.Frame(self.root, padx=8, pady=8)
        buttons.pack(fill="x")
        tk.Button(buttons, text="Connect", command=self.connect).pack(side="left")
        tk.Button(buttons, text="Refresh Present", command=self.refresh_present).pack(
            side="left", padx=4
        )
        tk.Button(buttons, text="Move Both", command=self.move_both).pack(side="left", padx=4)
        tk.Button(buttons, text="Torque Off", command=self.torque_off).pack(side="left", padx=4)
        tk.Button(buttons, text="Save YAML", command=self.save).pack(side="left", padx=4)

        per_motor = tk.Frame(self.root, padx=8, pady=4)
        per_motor.pack(fill="x")
        for motor_frame in self.frames:
            tk.Button(
                per_motor,
                text=f"{motor_frame.name}: Set Min = Current",
                command=lambda mf=motor_frame: self.set_current_as(mf, "min"),
            ).pack(side="left", padx=4)
            tk.Button(
                per_motor,
                text=f"{motor_frame.name}: Set Max = Current",
                command=lambda mf=motor_frame: self.set_current_as(mf, "max"),
            ).pack(side="left", padx=4)

    def run(self) -> None:
        self.root.mainloop()

    def connect(self) -> None:
        try:
            if self.bus is not None:
                self.bus.close()
            self.bus = DynamixelUiBus(self.args.port, self.args.baud)
            for motor_frame in self.frames:
                self.bus.ping(motor_frame.dxl_id)
                self.bus.configure_position_mode(motor_frame.dxl_id)
            self.status_var.set("connected; position mode configured; torque on")
            self.refresh_present()
        except Exception as exc:
            self.status_var.set(f"connect failed: {exc}")
            messagebox.showerror("Connect failed", str(exc))

    def require_bus(self) -> DynamixelUiBus:
        if self.bus is None:
            raise RuntimeError("not connected")
        return self.bus

    def refresh_present(self) -> None:
        try:
            bus = self.require_bus()
            for motor_frame in self.frames:
                motor_frame.set_present(bus.read_present_tick(motor_frame.dxl_id))
        except Exception as exc:
            self.status_var.set(f"refresh failed: {exc}")

    def move_both(self) -> None:
        try:
            bus = self.require_bus()
            for motor_frame in self.frames:
                current_tick = bus.read_present_tick(motor_frame.dxl_id)
                target_deg = motor_frame.target_motor_deg_for_current(current_tick)
                bus.move_absolute_deg(motor_frame.dxl_id, target_deg)
            self.status_var.set("move command sent; torque remains on")
            self.root.after(250, self.refresh_present)
        except Exception as exc:
            self.status_var.set(f"move failed: {exc}")
            messagebox.showerror("Move failed", str(exc))

    def set_current_as(self, motor_frame: MotorFrame, bound: str) -> None:
        try:
            tick = self.require_bus().read_present_tick(motor_frame.dxl_id)
            if bound == "min":
                motor_frame.set_current_as_min(tick)
            else:
                motor_frame.set_current_as_max(tick)
            self.status_var.set(f"{motor_frame.name} {bound} set from current position")
        except Exception as exc:
            self.status_var.set(f"set {bound} failed: {exc}")

    def torque_off(self) -> None:
        try:
            bus = self.require_bus()
            for motor_frame in self.frames:
                bus.torque_off(motor_frame.dxl_id)
            self.status_var.set("torque off")
        except Exception as exc:
            self.status_var.set(f"torque off failed: {exc}")

    def save(self) -> None:
        try:
            save_calibration_yaml(
                Path(self.args.output),
                port=self.args.port,
                baud=self.args.baud,
                motors=[motor_frame.calibration() for motor_frame in self.frames],
            )
            self.status_var.set(f"saved: {self.args.output}")
        except Exception as exc:
            self.status_var.set(f"save failed: {exc}")
            messagebox.showerror("Save failed", str(exc))

    def close(self) -> None:
        if self.bus is not None:
            self.bus.close()
            self.bus = None
        self.root.destroy()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DYNAMIXEL head calibration UI")
    parser.add_argument(
        "--port",
        default="auto",
        help="serial port path, or 'auto' to use /dev/serial/by-id, ttyACM0, ttyUSB0",
    )
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--ids", default="1,2", help="pan_id,tilt_id")
    parser.add_argument("--output", default=str(DEFAULT_CONFIG_PATH))
    return parser


def resolve_port(port: str) -> str:
    if port and port != "auto":
        path = Path(port)
        if not path.exists():
            raise ValueError(
                f"port does not exist: {port}. "
                "Run `ls -l /dev/serial/by-id/ /dev/ttyACM0 /dev/ttyUSB0` "
                "and pass the path that exists."
            )
        return port

    by_id_dir = Path("/dev/serial/by-id")
    if by_id_dir.exists():
        for path in sorted(by_id_dir.iterdir()):
            if path.exists():
                return str(path)
    for candidate in DEFAULT_PORT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise ValueError(
        "no DYNAMIXEL serial port found. Checked /dev/serial/by-id, "
        "/dev/ttyACM0, and /dev/ttyUSB0. Replug U2D2/OpenRB and check power/USB."
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        parse_motor_ids(args.ids)
        args.port = resolve_port(args.port)
    except ValueError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    app = CalibrationApp(args)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
