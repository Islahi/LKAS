"""Keyboard-controlled version of the autonomous QCar2 lane test.

This launcher reuses the autonomous perception and control implementation from
qcar2_autonomous_lane_g920_noise_120.py, but does not require a Logitech G920.

Controls:
    A / D       hold to add left/right steering disturbance (+/-6 degrees)
    L           start or pause autonomy
    SPACE       emergency stop and pause
    U           unstuck after collision; reacquire lane before restarting
    Q / ESC     stop and exit

Open QLabs and load Open Road. This program spawns QCar2 actor 0 itself.
"""

import threading
import time
import tkinter as tk

import cv2
import numpy as np

import qcar2_autonomous_lane_g920_noise_120 as core


KEYBOARD_DISTURBANCE_DEG = 6.0


class KeyboardState:
    def __init__(self):
        self._lock = threading.Lock()
        self._held = set()
        self._toggle_requested = False
        self._stop_requested = False
        self._unstuck_requested = False
        self.quit_requested = False

    def press(self, name):
        with self._lock:
            self._held.add(name)

    def release(self, name):
        with self._lock:
            self._held.discard(name)

    def is_down(self, name):
        with self._lock:
            return name in self._held

    def request_toggle(self):
        with self._lock:
            self._toggle_requested = True

    def consume_toggle(self):
        with self._lock:
            value = self._toggle_requested
            self._toggle_requested = False
            return value

    def request_stop(self):
        with self._lock:
            self._stop_requested = True

    def consume_stop(self):
        with self._lock:
            value = self._stop_requested
            self._stop_requested = False
            return value

    def request_unstuck(self):
        with self._lock:
            self._unstuck_requested = True

    def consume_unstuck(self):
        with self._lock:
            value = self._unstuck_requested
            self._unstuck_requested = False
            return value


def start_keyboard_listener(state):
    from pynput import keyboard

    def normalize(key):
        try:
            return key.char.lower() if key.char else None
        except AttributeError:
            if key == keyboard.Key.space:
                return "space"
            if key == keyboard.Key.esc:
                return "esc"
            return None

    def on_press(key):
        name = normalize(key)
        if name is None:
            return
        if name in {"a", "d"}:
            state.press(name)
        elif name == "l":
            if not state.is_down("l"):
                state.press("l")
                state.request_toggle()
        elif name == "space":
            if not state.is_down("space"):
                state.press("space")
                state.request_stop()
        elif name == "u":
            if not state.is_down("u"):
                state.press("u")
                state.request_unstuck()
        elif name in {"q", "esc"}:
            state.quit_requested = True
            return False

    def on_release(key):
        name = normalize(key)
        if name in {"a", "d", "l", "space", "u"}:
            state.release(name)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    return listener


def keyboard_disturbance(keys):
    if keys.is_down("a") and not keys.is_down("d"):
        return -KEYBOARD_DISTURBANCE_DEG
    if keys.is_down("d") and not keys.is_down("a"):
        return KEYBOARD_DISTURBANCE_DEG
    return 0.0


class TunableLanePID:
    """Lane controller with PID values supplied live by the control panel."""

    INTEGRAL_LIMIT = 0.40
    DERIVATIVE_FILTER = 0.82

    def __init__(self):
        self.integral = 0.0
        self.previous_error = None
        self.filtered_derivative = 0.0
        self.previous_output_deg = 0.0

    def reset(self):
        self.integral = 0.0
        self.previous_error = None
        self.filtered_derivative = 0.0
        self.previous_output_deg = 0.0

    def update(self, lateral_error, heading_error, dt, parameters):
        dt = float(np.clip(dt, 0.01, 0.15))
        combined_error = lateral_error + heading_error
        self.integral = float(np.clip(
            self.integral + combined_error * dt,
            -self.INTEGRAL_LIMIT,
            self.INTEGRAL_LIMIT,
        ))

        raw_derivative = 0.0
        if self.previous_error is not None:
            raw_derivative = (combined_error - self.previous_error) / dt
        self.filtered_derivative = (
            self.DERIVATIVE_FILTER * self.filtered_derivative
            + (1.0 - self.DERIVATIVE_FILTER) * raw_derivative
        )

        target_deg = -(
            parameters["lateral_kp"] * lateral_error
            + parameters["heading_kp"] * heading_error
            + parameters["ki"] * self.integral
            + parameters["kd"] * self.filtered_derivative
        )
        target_deg = float(np.clip(
            target_deg,
            -core.MAX_AUTO_STEERING_DEG,
            core.MAX_AUTO_STEERING_DEG,
        ))

        max_step = parameters["steering_rate"] * dt
        output_deg = float(np.clip(
            target_deg,
            self.previous_output_deg - max_step,
            self.previous_output_deg + max_step,
        ))
        self.previous_error = combined_error
        self.previous_output_deg = output_deg
        return output_deg


class PIDControlPanel:
    """Non-blocking Tk control panel serviced from the vehicle loop."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("QCar2 Autonomous PID Control")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.request_quit)

        self._requests = {
            "start": False,
            "pause": False,
            "stop": False,
            "unstuck": False,
            "reset_setup": False,
            "reset_pid": False,
            "quit": False,
        }

        self.status_var = tk.StringVar(value="Starting QLabs setup...")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Arial", 11, "bold"),
            fg="#FFFFFF",
            bg="#202020",
            anchor="w",
            padx=8,
            pady=8,
        ).grid(row=0, column=0, columnspan=2, sticky="ew")

        self.lateral_kp = self._add_scale(
            1, "Lateral Kp (deg)", 0.0, 50.0, core.LATERAL_GAIN_DEG, 0.5
        )
        self.heading_kp = self._add_scale(
            2, "Heading Kp (deg)", 0.0, 60.0, core.HEADING_GAIN_DEG, 0.5
        )
        self.ki = self._add_scale(3, "Integral Ki", 0.0, 10.0, 0.0, 0.1)
        self.kd = self._add_scale(
            4, "Derivative Kd", 0.0, 10.0, core.STEERING_D_GAIN_DEG, 0.1
        )
        self.steering_rate = self._add_scale(
            5,
            "Steering rate (deg/s)",
            20.0,
            200.0,
            core.MAX_AUTO_STEERING_RATE_DEG_S,
            5.0,
        )

        button_frame = tk.Frame(self.root, padx=6, pady=6)
        button_frame.grid(row=6, column=0, columnspan=2)
        buttons = [
            ("START", "start", "#16803A"),
            ("PAUSE", "pause", "#9A6A00"),
            ("STOP", "stop", "#B02020"),
            ("UNSTUCK", "unstuck", "#5555A8"),
            ("RESET & SETUP", "reset_setup", "#444444"),
            ("RESET PID", "reset_pid", "#444444"),
            ("QUIT", "quit", "#222222"),
        ]
        for index, (label, request, color) in enumerate(buttons):
            tk.Button(
                button_frame,
                text=label,
                width=14,
                command=lambda name=request: self._request(name),
                fg="#FFFFFF",
                bg=color,
                activeforeground="#FFFFFF",
                activebackground=color,
                font=("Arial", 9, "bold"),
            ).grid(row=index // 3, column=index % 3, padx=3, pady=3)

    def _add_scale(self, row, label, minimum, maximum, initial, resolution):
        tk.Label(self.root, text=label, anchor="w", padx=8).grid(
            row=row, column=0, sticky="w"
        )
        variable = tk.DoubleVar(value=initial)
        tk.Scale(
            self.root,
            variable=variable,
            from_=minimum,
            to=maximum,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            length=300,
            showvalue=True,
        ).grid(row=row, column=1, padx=6, pady=2)
        return variable

    def _request(self, name):
        self._requests[name] = True

    def request_quit(self):
        self._request("quit")

    def consume(self, name):
        value = self._requests[name]
        self._requests[name] = False
        return value

    def parameters(self):
        return {
            "lateral_kp": float(self.lateral_kp.get()),
            "heading_kp": float(self.heading_kp.get()),
            "ki": float(self.ki.get()),
            "kd": float(self.kd.get()),
            "steering_rate": float(self.steering_rate.get()),
        }

    def reset_pid_values(self):
        self.lateral_kp.set(core.LATERAL_GAIN_DEG)
        self.heading_kp.set(core.HEADING_GAIN_DEG)
        self.ki.set(0.0)
        self.kd.set(core.STEERING_D_GAIN_DEG)
        self.steering_rate.set(core.MAX_AUTO_STEERING_RATE_DEG_S)

    def set_status(self, text):
        self.status_var.set(text)

    def update(self):
        self.root.update_idletasks()
        self.root.update()

    def destroy(self):
        try:
            self.root.destroy()
        except Exception:
            pass


def main():
    qlabs = None
    qcar = None
    listener = None
    panel = None

    try:
        print("=" * 76)
        print("QCar2 AUTONOMOUS INITIAL-LANE TRACKER - KEYBOARD")
        print(
            f"Adaptive acceleration to {core.MAX_STRAIGHT_SPEED_KMH:.0f} km/h "
            "+ keyboard +/-6 degree disturbances"
        )
        print("=" * 76)
        print("Open QLabs and load Open Road; this program will spawn the car.\n")

        panel = PIDControlPanel()
        panel.update()

        qlabs, qcar = core.setup_qlabs_car()
        tracker = core.InitialLaneTracker()
        steering_controller = TunableLanePID()
        keys = KeyboardState()
        listener = start_keyboard_listener(keys)

        autonomy_enabled = False
        command_speed_kmh = 0.0
        alignment_complete = False
        alignment_frames = 0
        collision_latched = False
        last_location = None
        last_rotation = None
        previous_time = time.monotonic()
        fps_filtered = 0.0

        core.command_vehicle(qcar, 0.0, 0.0, brake=True)
        panel.set_status("Acquiring initial lane...")
        print("Acquiring the initial lane. Press L after LOCKED appears.")
        print("A/D noise | L start/pause | SPACE stop | U unstuck | Q/ESC quit\n")

        while not keys.quit_requested:
            loop_start = time.monotonic()
            dt = float(np.clip(loop_start - previous_time, 0.001, 0.20))
            previous_time = loop_start
            disturbance_deg = keyboard_disturbance(keys)

            try:
                panel.update()
            except tk.TclError:
                keys.quit_requested = True
                break

            if panel.consume("quit"):
                keys.quit_requested = True
                break

            if panel.consume("reset_pid"):
                panel.reset_pid_values()
                steering_controller.reset()
                print("\nPID values restored to defaults.")

            if panel.consume("reset_setup"):
                print("\nReset & Setup requested; stopping and respawning QCar2...")
                autonomy_enabled = False
                command_speed_kmh = 0.0
                try:
                    core.command_vehicle(qcar, 0.0, 0.0, brake=True)
                except Exception:
                    pass
                try:
                    qlabs.close()
                except Exception:
                    pass
                panel.set_status("Resetting QLabs and respawning QCar2...")
                panel.update()
                qlabs, qcar = core.setup_qlabs_car()
                tracker = core.InitialLaneTracker()
                steering_controller = TunableLanePID()
                alignment_complete = False
                alignment_frames = 0
                collision_latched = False
                last_location = None
                last_rotation = None
                previous_time = time.monotonic()
                core.command_vehicle(qcar, 0.0, 0.0, brake=True)
                print("Reset complete; acquiring a fresh initial lane.")

            gui_unstuck = panel.consume("unstuck")

            if keys.consume_unstuck() or gui_unstuck:
                autonomy_enabled = False
                command_speed_kmh = 0.0
                alignment_complete = False
                alignment_frames = 0
                steering_controller.reset()
                if core.unstuck_vehicle(qcar, last_location, last_rotation):
                    collision_latched = False
                    tracker._clear_lock()
                    print(
                        "\nUNSTUCK complete. Waiting for a fresh lane lock; "
                        "press L afterward."
                    )
                    time.sleep(0.5)
                else:
                    print("\nUNSTUCK unavailable: vehicle pose is not ready.")

            image_ok, frame_bgr = qcar.get_image(core.CAMERA)
            if not image_ok or frame_bgr is None:
                autonomy_enabled = False
                command_speed_kmh = core.rate_limit_speed(
                    command_speed_kmh, 0.0, dt
                )
                core.command_vehicle(qcar, command_speed_kmh, 0.0, brake=False)
                print("Camera image unavailable; autonomy paused.", end="\r")
                time.sleep(0.02)
                continue

            result = tracker.process(frame_bgr)
            height, width = frame_bgr.shape[:2]
            geometry = core.lane_geometry(result, tracker, width, height)

            keyboard_toggle = keys.consume_toggle()
            gui_start = panel.consume("start")
            gui_pause = panel.consume("pause")

            if gui_pause:
                if autonomy_enabled:
                    print("\nAutonomy paused from control panel.")
                autonomy_enabled = False
                alignment_complete = False
                alignment_frames = 0

            start_requested = gui_start or (
                keyboard_toggle and not autonomy_enabled
            )
            pause_requested = keyboard_toggle and autonomy_enabled

            if pause_requested:
                autonomy_enabled = False
                alignment_complete = False
                alignment_frames = 0
                print("\nAutonomy paused.")
            elif start_requested:
                if autonomy_enabled:
                    pass
                elif collision_latched:
                    print("\nCollision is latched. Press U to unstuck first.")
                elif tracker.lane_locked and geometry is not None:
                    autonomy_enabled = True
                    command_speed_kmh = 0.0
                    alignment_complete = False
                    alignment_frames = 0
                    steering_controller.reset()
                    print("\nAutonomy active; low-speed alignment started.")
                else:
                    print("\nCannot start: waiting for initial-lane lock.")

            if (
                keys.consume_stop()
                or keys.is_down("space")
                or panel.consume("stop")
            ):
                if autonomy_enabled:
                    print("\nEmergency stop; autonomy paused.")
                autonomy_enabled = False
                command_speed_kmh = 0.0
                alignment_complete = False
                alignment_frames = 0
                steering_controller.reset()

            auto_steering_deg = 0.0
            lateral_error = 0.0
            heading_error = 0.0
            if geometry is not None:
                lateral_error, heading_error, _, _, _, _, _ = geometry
                auto_steering_deg = steering_controller.update(
                    lateral_error,
                    heading_error,
                    dt,
                    panel.parameters(),
                )
            else:
                steering_controller.reset()

            if autonomy_enabled and geometry is not None:
                curve_limit_kmh = core.curve_speed_limit_kmh(
                    auto_steering_deg, heading_error
                )
                confidence_limit_kmh = core.confidence_speed_limit_kmh(
                    result, tracker
                )
                tracking_limit_kmh = core.tracking_error_speed_limit_kmh(
                    lateral_error, heading_error
                )

                if not alignment_complete:
                    geometry_mode = geometry[-1]
                    well_aligned = (
                        geometry_mode == "TWO BOUNDARIES"
                        and abs(lateral_error) <= core.ALIGNMENT_LATERAL_TOLERANCE
                        and abs(heading_error) <= core.ALIGNMENT_HEADING_TOLERANCE
                    )
                    alignment_frames = alignment_frames + 1 if well_aligned else 0
                    if alignment_frames >= core.ALIGNMENT_STABLE_FRAMES:
                        alignment_complete = True
                        print("\nAlignment stable; normal adaptive acceleration enabled.")

                if alignment_complete:
                    drive_phase = "CRUISE"
                    target_speed_kmh = min(
                        core.MAX_STRAIGHT_SPEED_KMH,
                        curve_limit_kmh,
                        confidence_limit_kmh,
                        tracking_limit_kmh,
                    )
                else:
                    drive_phase = (
                        f"ALIGNING {alignment_frames}/{core.ALIGNMENT_STABLE_FRAMES}"
                    )
                    target_speed_kmh = min(
                        core.ALIGNMENT_CRAWL_SPEED_KMH,
                        curve_limit_kmh,
                        confidence_limit_kmh,
                        tracking_limit_kmh,
                    )

                command_speed_kmh = core.rate_limit_speed(
                    command_speed_kmh, target_speed_kmh, dt
                )
                final_steering_deg = float(np.clip(
                    auto_steering_deg + disturbance_deg,
                    -core.MAX_TOTAL_STEERING_DEG,
                    core.MAX_TOTAL_STEERING_DEG,
                ))
            else:
                curve_limit_kmh = 0.0
                confidence_limit_kmh = 0.0
                tracking_limit_kmh = 0.0
                target_speed_kmh = 0.0
                drive_phase = "PAUSED"
                command_speed_kmh = core.rate_limit_speed(
                    command_speed_kmh, 0.0, dt
                )
                final_steering_deg = 0.0

            status, location, rotation, front_hit, rear_hit = core.command_vehicle(
                qcar,
                command_speed_kmh,
                final_steering_deg,
                brake=not autonomy_enabled and command_speed_kmh <= 0.01,
            )
            if not status:
                raise RuntimeError("QLabs rejected the vehicle command.")
            last_location = location
            last_rotation = rotation

            if front_hit or rear_hit:
                if not collision_latched:
                    print(
                        "\nCollision detected; autonomy paused. "
                        "Press U to unstuck."
                    )
                collision_latched = True
                autonomy_enabled = False
                command_speed_kmh = 0.0
                target_speed_kmh = 0.0
                final_steering_deg = 0.0
                alignment_complete = False
                alignment_frames = 0
                steering_controller.reset()
                core.command_vehicle(qcar, 0.0, 0.0, brake=True)

            panel_state = (
                "COLLISION PAUSED"
                if collision_latched
                else (drive_phase if autonomy_enabled else "PAUSED")
            )
            lane_state = (
                "LOCKED"
                if tracker.lane_locked
                else f"ACQUIRING {tracker.acquire_frames}/{core.LOCK_ACQUIRE_FRAMES}"
            )
            panel.set_status(
                f"{panel_state} | lane {lane_state} | "
                f"speed {command_speed_kmh:.1f} km/h"
            )

            display = core.draw_overlay(frame_bgr, result)
            display = core.draw_autonomy_overlay(
                display,
                tracker,
                geometry,
                autonomy_enabled,
                curve_limit_kmh,
                confidence_limit_kmh,
                tracking_limit_kmh,
                target_speed_kmh,
                command_speed_kmh,
                drive_phase,
                collision_latched,
                auto_steering_deg,
                disturbance_deg,
                final_steering_deg,
                "Keyboard: A/D noise | L start/pause | SPACE stop | U unstuck | Q/ESC quit",
            )

            fps = 1.0 / dt
            fps_filtered = fps if fps_filtered == 0.0 else (
                0.9 * fps_filtered + 0.1 * fps
            )
            cv2.putText(
                display, f"control FPS: {fps_filtered:.1f}", (15, 336),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1
            )
            cv2.imshow("QCar2 - Autonomous Lane Camera Probe (Keyboard)", display)

            debug = cv2.hconcat([
                cv2.cvtColor(result.edges, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(result.roi_edges, cv2.COLOR_GRAY2BGR),
            ])
            cv2.putText(
                debug, "All Canny edges", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )
            cv2.putText(
                debug, "ROI edges used for lane detection",
                (debug.shape[1] // 2 + 10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )
            cv2.imshow("QCar2 - Autonomous Lane Edge Probe (Keyboard)", debug)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                keys.quit_requested = True

            remaining_s = (1.0 / core.CONTROL_RATE_HZ) - (
                time.monotonic() - loop_start
            )
            if remaining_s > 0.0:
                time.sleep(remaining_s)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1
    finally:
        if qcar is not None:
            try:
                core.command_vehicle(qcar, 0.0, 0.0, brake=True)
            except Exception:
                pass
        cv2.destroyAllWindows()
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
        if panel is not None:
            panel.destroy()
        if qlabs is not None:
            try:
                qlabs.close()
            except Exception:
                pass
        print("Keyboard autonomous controller closed; QCar2 command set to zero.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
