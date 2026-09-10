r"""Run the existing PAL lane keeper and log the driven Open Road path.

The vehicle-control behavior is unchanged from
``qcar2_autonomous_pal_existing_car.py``. A button-control window replaces
the controller's keyboard listener so the keyboard remains available for other
programs. A background logger connects to
QLabs, reads the selected QCar 2 actor's world transform, and updates
``open_road_reference.json`` when the controller exits normally or with an
exception.

The measured road points use this format::

    [x_m, y_m, elevation_z_m]

The original JSON is copied to a timestamped ``.backup-*.json`` file before it
is replaced. The logger does not spawn, move, control, or delete any actor.

Examples
--------
Use actor 0 and a JSON file beside this script::

    python qcar2_autonomous_pal_existing_car_with_location_logger.py

Choose another actor or reference file::

    python qcar2_autonomous_pal_existing_car_with_location_logger.py ^
        --actor 1 --reference C:\path\to\open_road_reference.json

Button controls
---------------
- Toggle Manual / Autonomous Mode
- Request / Pause Autonomy
- Emergency Stop
- Open / Close Color Picker
- Press-and-hold Forward, Reverse, Left, and Right in manual mode
- Stop, Save, and Exit
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2

import qcar2_autonomous_pal_manual_lkas as controller_module
from yellow_lane_detector import YellowLaneTracker


DEFAULT_HOST = "localhost"
DEFAULT_ACTOR_NUMBER = 0
DEFAULT_SAMPLE_RATE_HZ = 20.0
DEFAULT_MINIMUM_DISTANCE_M = 1.0
DEFAULT_REFERENCE_FILE = Path(__file__).with_name("open_road_reference.json")


class OpenRoadLocationLogger:
    """Sample one existing QCar 2 actor and update an Open Road reference."""

    def __init__(
        self,
        reference_path: Path,
        host: str = DEFAULT_HOST,
        actor_number: int = DEFAULT_ACTOR_NUMBER,
        sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
        minimum_distance_m: float = DEFAULT_MINIMUM_DISTANCE_M,
    ) -> None:
        self.reference_path = reference_path.resolve()
        self.host = host
        self.actor_number = actor_number
        self.sample_period = 1.0 / sample_rate_hz
        self.minimum_distance_m = minimum_distance_m

        self.points: list[list[float]] = []
        self.started_at = datetime.now(timezone.utc)
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: Exception | None = None
        self._runtime_error: Exception | None = None

    def start(self, timeout_s: float = 5.0) -> None:
        """Start sampling and verify that the requested actor exists."""
        self._thread = threading.Thread(
            target=self._worker,
            name="open-road-location-logger",
            daemon=True,
        )
        self._thread.start()

        if not self._ready_event.wait(timeout_s):
            self.stop()
            raise TimeoutError("Timed out while connecting the logger to QLabs.")
        if self._startup_error is not None:
            self.stop()
            raise RuntimeError(str(self._startup_error)) from self._startup_error

    def stop(self) -> None:
        """Stop the worker and close its QLabs connection."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def save(self) -> Path:
        """Atomically replace the road points with measured 3-D samples."""
        if len(self.points) < 2:
            detail = f" Logger error: {self._runtime_error}" if self._runtime_error else ""
            raise RuntimeError(
                f"Only {len(self.points)} valid location sample(s) were recorded; "
                f"the reference file was not changed.{detail}"
            )

        with self.reference_path.open("r", encoding="utf-8") as file:
            document = json.load(file)

        if "road_reference" not in document:
            raise KeyError("The JSON file does not contain 'road_reference'.")

        recorded_at = datetime.now(timezone.utc)
        points = [[round(value, 3) for value in point] for point in self.points]

        document["version"] = max(int(document.get("version", 1)), 2)
        document["units"] = "meters"
        document["bounds"] = {
            "min_x": min(point[0] for point in points),
            "max_x": max(point[0] for point in points),
            "min_y": min(point[1] for point in points),
            "max_y": max(point[1] for point in points),
            "min_z": min(point[2] for point in points),
            "max_z": max(point[2] for point in points),
        }

        road_reference = document["road_reference"]
        road_reference["kind"] = "qlabs_driven_centerline_3d"
        road_reference["point_format"] = ["x", "y", "elevation_z"]
        road_reference["points"] = points
        road_reference["measurement"] = {
            "actor_number": self.actor_number,
            "qlabs_host": self.host,
            "started_at_utc": self.started_at.isoformat(),
            "finished_at_utc": recorded_at.isoformat(),
            "sample_period_s": self.sample_period,
            "minimum_horizontal_spacing_m": self.minimum_distance_m,
            "sample_count": len(points),
        }

        source = document.setdefault("source", {})
        source["location_data"] = (
            "Measured from the QLabs QCar 2 world transform while the PAL lane "
            "keeper drove the Open Road. Z is the QLabs world elevation."
        )

        timestamp = recorded_at.strftime("%Y%m%d-%H%M%S")
        backup_path = self.reference_path.with_name(
            f"{self.reference_path.stem}.backup-{timestamp}{self.reference_path.suffix}"
        )
        shutil.copy2(self.reference_path, backup_path)

        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{self.reference_path.stem}-",
                suffix=".tmp",
                dir=self.reference_path.parent,
                delete=False,
            ) as temporary_file:
                temporary_name = temporary_file.name
                json.dump(document, temporary_file, indent=2, ensure_ascii=False)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(temporary_name, self.reference_path)
        except Exception:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink(missing_ok=True)
                except OSError:
                    pass
            raise

        return backup_path

    def _worker(self) -> None:
        qlabs = QuanserInteractiveLabs()
        ready_was_reported = False

        try:
            connection_result = qlabs.open(self.host)
            if connection_result is False:
                raise ConnectionError(f"QLabs rejected the connection to {self.host}.")

            qcar = QLabsQCar2(qlabs)
            qcar.actorNumber = self.actor_number
            if not qcar.ping():
                raise RuntimeError(
                    f"QCar 2 actor {self.actor_number} does not exist in QLabs."
                )

            self._ready_event.set()
            ready_was_reported = True

            next_sample_time = time.monotonic()
            while not self._stop_event.is_set():
                status, location, _rotation, _scale = qcar.get_world_transform()
                if status:
                    point = [float(location[0]), float(location[1]), float(location[2])]
                    if self._should_keep(point):
                        self.points.append(point)

                next_sample_time += self.sample_period
                delay = next_sample_time - time.monotonic()
                if delay > 0:
                    self._stop_event.wait(delay)
                else:
                    next_sample_time = time.monotonic()

        except Exception as exc:
            if ready_was_reported:
                self._runtime_error = exc
            else:
                self._startup_error = exc
                self._ready_event.set()
        finally:
            try:
                qlabs.close()
            except Exception:
                pass

    def _should_keep(self, point: list[float]) -> bool:
        if not all(math.isfinite(value) for value in point):
            return False
        if not self.points:
            return True

        previous = self.points[-1]
        horizontal_distance = math.hypot(
            point[0] - previous[0],
            point[1] - previous[1],
        )
        return horizontal_distance >= self.minimum_distance_m


class _ButtonListener:
    """Keyboard-listener substitute expected by the controller."""

    def stop(self) -> None:
        pass

    def join(self, timeout: float | None = None) -> None:
        pass


class ButtonCommandBridge:
    """Expose the controller's KeyState safely to GUI button callbacks."""

    def __init__(self) -> None:
        self.keys = None
        self.ready_event = threading.Event()

    def attach(self, keys):
        self.keys = keys
        self.ready_event.set()
        return _ButtonListener()

    def _with_keys(self, action) -> bool:
        keys = self.keys
        if keys is None:
            return False
        action(keys)
        return True

    def toggle_mode(self) -> bool:
        return self._with_keys(lambda keys: keys.request_drive_mode_toggle())

    def toggle_autonomy(self) -> bool:
        return self._with_keys(lambda keys: keys.request_lkas_toggle())

    def emergency_stop(self) -> bool:
        def action(keys):
            keys.clear_drive_keys()
            keys.request_stop()

        return self._with_keys(action)

    def toggle_color_picker(self) -> bool:
        return self._with_keys(lambda keys: keys.request_color_toggle())

    def press_drive(self, name: str) -> bool:
        return self._with_keys(lambda keys: keys.press(name))

    def release_drive(self, name: str) -> bool:
        return self._with_keys(lambda keys: keys.release(name))

    def clear_drive(self) -> bool:
        return self._with_keys(lambda keys: keys.clear_drive_keys())

    def quit(self) -> bool:
        def action(keys):
            keys.clear_drive_keys()
            keys.request_stop()
            keys.quit_requested = True

        return self._with_keys(action)


class LoggerControlWindow:
    """Tk button panel controlling the background PAL lane controller."""

    def __init__(
        self,
        bridge: ButtonCommandBridge,
        logger: OpenRoadLocationLogger,
        controller_thread: threading.Thread,
        controller_result: dict,
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("QCar2 Logger Button Control")
        self.root.resizable(False, False)
        self.bridge = bridge
        self.logger = logger
        self.controller_thread = controller_thread
        self.controller_result = controller_result
        self.stop_requested = False

        frame = ttk.Frame(self.root, padding=14)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(
            frame,
            text="QCar2 Lane Controller and Location Logger",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, columnspan=3, pady=(0, 10))

        self.status_var = tk.StringVar(value="Starting lane controller...")
        ttk.Label(
            frame, textvariable=self.status_var, justify="left", width=58
        ).grid(row=1, column=0, columnspan=3, pady=(0, 10))

        ttk.Button(
            frame,
            text="Toggle Manual / Autonomous Mode",
            command=self._toggle_mode,
        ).grid(row=2, column=0, columnspan=3, sticky="ew", pady=3)
        ttk.Button(
            frame,
            text="Request / Pause Autonomy",
            command=self._toggle_autonomy,
        ).grid(row=3, column=0, columnspan=3, sticky="ew", pady=3)
        ttk.Button(
            frame,
            text="Open / Close Color Picker",
            command=self._toggle_color,
        ).grid(row=4, column=0, columnspan=3, sticky="ew", pady=3)

        ttk.Label(
            frame,
            text="Manual drive (press and hold)",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=5, column=0, columnspan=3, pady=(12, 4))

        self._drive_button(frame, "Forward", "w", 6, 1)
        self._drive_button(frame, "Left", "a", 7, 0)
        self._drive_button(frame, "Reverse", "s", 7, 1)
        self._drive_button(frame, "Right", "d", 7, 2)

        emergency = tk.Button(
            frame,
            text="EMERGENCY STOP",
            command=self._emergency_stop,
            bg="#c62828",
            fg="white",
            activebackground="#8e0000",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            width=20,
        )
        emergency.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(14, 4))

        ttk.Button(
            frame,
            text="Stop, Save, and Exit",
            command=self.request_exit,
        ).grid(row=9, column=0, columnspan=3, sticky="ew", pady=3)

        for column in range(3):
            frame.columnconfigure(column, weight=1)

        # Never leave a manual command held if the control window loses focus.
        self.root.bind("<FocusOut>", lambda _event: self.bridge.clear_drive())
        self.root.protocol("WM_DELETE_WINDOW", self.request_exit)
        self.root.after(100, self._poll)

    def _drive_button(self, parent, label: str, key: str, row: int, column: int):
        button = self.tk.Button(parent, text=label, width=14)
        button.grid(row=row, column=column, padx=3, pady=3, sticky="ew")
        button.bind("<ButtonPress-1>", lambda _event: self.bridge.press_drive(key))
        button.bind("<ButtonRelease-1>", lambda _event: self.bridge.release_drive(key))
        button.bind("<Leave>", lambda _event: self.bridge.release_drive(key))

    def _set_command_status(self, message: str, accepted: bool) -> None:
        self.status_var.set(
            message if accepted else "Controller is still starting; try again."
        )

    def _toggle_mode(self) -> None:
        self._set_command_status(
            "Drive-mode toggle sent.", self.bridge.toggle_mode()
        )

    def _toggle_autonomy(self) -> None:
        self._set_command_status(
            "Autonomy request/pause command sent.", self.bridge.toggle_autonomy()
        )

    def _toggle_color(self) -> None:
        self._set_command_status(
            "Color-picker toggle sent.", self.bridge.toggle_color_picker()
        )

    def _emergency_stop(self) -> None:
        self._set_command_status(
            "Emergency stop sent; autonomy cancelled.",
            self.bridge.emergency_stop(),
        )

    def request_exit(self) -> None:
        if self.stop_requested:
            return
        self.stop_requested = True
        self.bridge.quit()
        self.status_var.set("Stopping controller and preparing to save...")

    def _poll(self) -> None:
        if self.stop_requested and self.controller_thread.is_alive():
            # Also handles Exit being pressed before KeyState finished starting.
            self.bridge.quit()
        if self.stop_requested and not self.controller_thread.is_alive():
            self.root.quit()
            return
        if not self.controller_thread.is_alive():
            self.stop_requested = True
            self.status_var.set("Controller stopped; preparing to save...")
            self.root.after(100, self._poll)
            return
        if not self.stop_requested:
            controller_state = (
                "ready" if self.bridge.ready_event.is_set() else "starting"
            )
            self.status_var.set(
                f"Controller: {controller_state} | "
                f"Recorded points: {len(self.logger.points):,}\n"
                "The computer keyboard is not captured by this program."
            )
        self.root.after(100, self._poll)

    def run(self) -> None:
        self.root.mainloop()
        self.root.destroy()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PAL lane keeping and log the QCar's 3-D QLabs path."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="QLabs host name or IP")
    parser.add_argument(
        "--actor", type=int, default=DEFAULT_ACTOR_NUMBER, help="existing QCar 2 actor number"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_FILE,
        help="open_road_reference.json path",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help="QLabs location samples requested per second",
    )
    parser.add_argument(
        "--minimum-distance",
        type=float,
        default=DEFAULT_MINIMUM_DISTANCE_M,
        help="minimum horizontal distance in meters between saved points",
    )
    args = parser.parse_args()

    if args.actor < 0:
        parser.error("--actor must be zero or greater")
    if args.sample_rate <= 0:
        parser.error("--sample-rate must be greater than zero")
    if args.minimum_distance <= 0:
        parser.error("--minimum-distance must be greater than zero")
    if not args.reference.is_file():
        parser.error(f"reference file not found: {args.reference}")

    return args


def run() -> int:
    args = parse_arguments()
    logger = OpenRoadLocationLogger(
        reference_path=args.reference,
        host=args.host,
        actor_number=args.actor,
        sample_rate_hz=args.sample_rate,
        minimum_distance_m=args.minimum_distance,
    )

    print(f"Connecting location logger to QCar 2 actor {args.actor}...")
    try:
        logger.start()
    except Exception as exc:
        print(f"Location logger could not start: {exc}", file=sys.stderr)
        return 1

    print(
        f"Logging to {args.reference.resolve()} at {args.sample_rate:g} Hz "
        f"with {args.minimum_distance:g} m minimum point spacing."
    )

    bridge = ButtonCommandBridge()
    controller_result = {"exit_code": 0, "error": None}
    original_listener_factory = controller_module.start_keyboard_listener
    original_wait_key = controller_module.cv2.waitKey

    # Keep OpenCV window events responsive but discard all keyboard commands.
    def wait_key_without_commands(delay_ms: int) -> int:
        original_wait_key(delay_ms)
        return -1

    controller_module.start_keyboard_listener = bridge.attach
    controller_module.cv2.waitKey = wait_key_without_commands

    def controller_worker() -> None:
        try:
            result = controller_module.main(
                setup_vehicle=False,
                tracker_factory=YellowLaneTracker,
            )
            if isinstance(result, int):
                controller_result["exit_code"] = result
        except SystemExit as exc:
            if isinstance(exc.code, int):
                controller_result["exit_code"] = exc.code
        except BaseException as exc:
            controller_result["error"] = exc

    controller_thread = threading.Thread(
        target=controller_worker,
        name="pal-lane-controller",
        daemon=True,
    )
    controller_thread.start()

    try:
        control_window = LoggerControlWindow(
            bridge=bridge,
            logger=logger,
            controller_thread=controller_thread,
            controller_result=controller_result,
        )
        control_window.run()
    except KeyboardInterrupt:
        print("Button controller interrupted by user.")
    except BaseException as exc:
        controller_result["error"] = exc
    finally:
        bridge.quit()
        controller_thread.join(timeout=5.0)
        controller_module.start_keyboard_listener = original_listener_factory
        controller_module.cv2.waitKey = original_wait_key
        logger.stop()

    try:
        backup_path = logger.save()
    except Exception as exc:
        print(f"Location data was not saved: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {len(logger.points)} measured [x, y, z] road points.")
    print(f"Updated: {args.reference.resolve()}")
    print(f"Backup:  {backup_path}")

    controller_error = controller_result["error"]
    if controller_error is not None:
        print(
            f"Lane controller stopped with an error: {controller_error}",
            file=sys.stderr,
        )
        return 1
    return int(controller_result["exit_code"])


if __name__ == "__main__":
    sys.exit(run())
