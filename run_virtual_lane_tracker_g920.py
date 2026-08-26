"""QCar2 virtual lane tracker controlled with a Logitech G920.

This is a separate launcher; the original keyboard launcher is unchanged.

Controls:
    Steering wheel     Manual steering; overrides and cancels LKAS
    Accelerator pedal  Manual throttle
    Brake pedal        Immediate zero throttle and LKAS disengagement
    R1 / right button  Reverse while held (use with accelerator)
    A button            Toggle LKAS steering
    B button            Emergency stop and disengage LKAS
    Q / ESC             Quit (OpenCV fallback)

LKAS controls steering only. Accelerator, brake, and reverse remain under
driver control in every mode.
"""

import sys
import time

import cv2
import numpy as np

import settings as cfg
from lane_detector import LaneTracker, draw_overlay
from run_virtual_lane_tracker import setup_qlabs
from steering_controller import SteeringCommand, SteeringPID


# G920 input tuning. Change the sign if wheel steering is reversed.
G920_STEERING_SIGN = -1.0
G920_STEERING_DEADZONE = 0.05
G920_OVERRIDE_THRESHOLD = 0.10
G920_PEDAL_DEADZONE = 0.05

REVERSE_BUTTON_NAMES = ("buttonRight", "buttonR1", "buttonRB")
LKAS_BUTTON_NAMES = ("buttonA",)
STOP_BUTTON_NAMES = ("buttonB",)


def _button_value(g920, names):
    """Read a G920 button across common PAL attribute names."""
    for name in names:
        if hasattr(g920, name):
            return bool(getattr(g920, name)), name
    return False, None


def _apply_deadzone(value, deadzone):
    """Remove axis noise and rescale the remaining travel to 0..1."""
    value = float(np.clip(value, 0.0, 1.0))
    if value <= deadzone:
        return 0.0
    return (value - deadzone) / (1.0 - deadzone)


def read_g920_commands(g920):
    """Return throttle, steering, brake, reverse, LKAS and stop button state."""
    g920.read()

    accelerator = _apply_deadzone(g920.throttle, G920_PEDAL_DEADZONE)
    brake = _apply_deadzone(g920.brake, G920_PEDAL_DEADZONE)
    reverse, reverse_name = _button_value(g920, REVERSE_BUTTON_NAMES)
    lkas_button, lkas_name = _button_value(g920, LKAS_BUTTON_NAMES)
    stop_button, stop_name = _button_value(g920, STOP_BUTTON_NAMES)

    throttle_limit = abs(
        cfg.MANUAL_REVERSE_THROTTLE if reverse else cfg.MANUAL_THROTTLE
    )
    throttle = (-1.0 if reverse else 1.0) * accelerator * throttle_limit

    # The QCar interface has no separate brake command. Removing drive power is
    # the safest mapping; the simulated vehicle then decelerates naturally.
    if brake > 0.0:
        throttle = 0.0

    wheel = float(np.clip(G920_STEERING_SIGN * g920.steering, -1.0, 1.0))
    if abs(wheel) <= G920_STEERING_DEADZONE:
        wheel = 0.0
    else:
        wheel = np.sign(wheel) * (
            (abs(wheel) - G920_STEERING_DEADZONE)
            / (1.0 - G920_STEERING_DEADZONE)
        )
    steering = float(wheel * cfg.MANUAL_STEERING)

    throttle = float(np.clip(throttle, -cfg.MAX_ABS_THROTTLE, cfg.MAX_ABS_THROTTLE))
    steering = float(np.clip(steering, -cfg.MAX_ABS_STEERING, cfg.MAX_ABS_STEERING))

    names = (reverse_name, lkas_name, stop_name)
    return throttle, steering, brake, reverse, lkas_button, stop_button, names


def main():
    print("=" * 72)
    print("QCar2 VIRTUAL LANE KEEPING ASSIST - LOGITECH G920")
    print("G920 pedals + manual wheel steering + driver-enabled PID assistance")
    print("=" * 72)
    print("Open QLabs and load Open Road before running this program.\n")

    qlabs = None
    camera = None
    car = None
    g920 = None

    try:
        qlabs = setup_qlabs()

        # PAL is imported after the QLabs real-time model has started.
        from pal.products.qcar import QCar, QCarRealSense
        from pal.utilities.steering import LogitechG920

        camera = QCarRealSense(
            mode="RGB",
            frameWidthRGB=cfg.IMAGE_WIDTH,
            frameHeightRGB=cfg.IMAGE_HEIGHT,
            frameRateRGB=cfg.CAMERA_FPS,
            readMode=0,
        )
        car = QCar(readMode=0, pwmLimit=cfg.MAX_ABS_THROTTLE, steeringBias=0)
        if not car.card.is_valid():
            raise RuntimeError(
                "PAL QCar2 connection is invalid. If prompted for virtual QCar type, choose 2."
            )

        print("Connecting Logitech G920...")
        g920 = LogitechG920()
        tracker = LaneTracker()
        controller = SteeringPID()

        lkas_requested = False
        lkas_engaged = False
        confident_frames = 0
        last_command = SteeringCommand(0.0, None)
        previous_lkas_button = False
        button_names_reported = False

        leds = np.zeros(8, dtype=int)
        prev_time = time.perf_counter()
        fps_filtered = 0.0

        print("\nReady.")
        print("  Wheel       = manual steering / LKAS override")
        print("  Accelerator = forward or reverse throttle")
        print("  Brake       = zero throttle and disengage LKAS")
        print("  R1          = reverse while held")
        print("  A           = toggle LKAS")
        print("  B           = emergency stop")
        print("  Q/ESC       = quit\n")

        running = True
        while running:
            throttle, manual_steering, brake, reverse, lkas_button, stop_button, names = (
                read_g920_commands(g920)
            )

            if not button_names_reported:
                reverse_name, lkas_name, stop_name = names
                if reverse_name is None:
                    raise RuntimeError("G920 R1/right button was not found by PAL.")
                if lkas_name is None or stop_name is None:
                    available = [name for name in dir(g920) if "button" in name.lower()]
                    raise RuntimeError(
                        "G920 A/B buttons were not found. Available button attributes: "
                        f"{available}"
                    )
                print(
                    f"G920 buttons: reverse={reverse_name}, "
                    f"LKAS={lkas_name}, stop={stop_name}"
                )
                button_names_reported = True

            lkas_pressed = lkas_button and not previous_lkas_button
            previous_lkas_button = lkas_button

            new_frame = camera.read_RGB()
            if not new_frame:
                if lkas_requested or lkas_engaged:
                    print("LKAS disengaged: camera frame unavailable")
                lkas_requested = False
                lkas_engaged = False
                confident_frames = 0
                controller.reset()
                car.read_write_std(throttle, manual_steering, leds)
                cv2.waitKey(1)
                continue

            frame_rgb = camera.imageBufferRGB.copy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            result = tracker.process(frame_bgr)
            display = draw_overlay(frame_bgr, result)

            now = time.perf_counter()
            dt = max(now - prev_time, 1e-6)
            prev_time = now

            if lkas_pressed:
                lkas_requested = not lkas_requested
                lkas_engaged = False
                confident_frames = 0
                controller.reset()
                print(
                    "LKAS requested; waiting for confident lanes"
                    if lkas_requested else "LKAS off"
                )

            if stop_button or brake > 0.0:
                if lkas_requested or lkas_engaged:
                    reason = "brake pedal" if brake > 0.0 else "emergency stop"
                    print(f"LKAS disengaged: {reason}")
                lkas_requested = False
                lkas_engaged = False
                confident_frames = 0
                controller.reset()
                throttle = 0.0

            # Significant wheel movement is an immediate driver override.
            if abs(manual_steering) >= G920_OVERRIDE_THRESHOLD * cfg.MANUAL_STEERING:
                if lkas_requested or lkas_engaged:
                    print("LKAS disengaged: steering-wheel override")
                lkas_requested = False
                lkas_engaged = False
                confident_frames = 0
                controller.reset()

            confidence_ok = (
                result.error_px is not None
                and result.left_detected
                and result.right_detected
                and result.confidence >= cfg.LKAS_MIN_CONFIDENCE
            )

            if lkas_requested:
                if confidence_ok:
                    confident_frames += 1
                    if confident_frames >= cfg.LKAS_ENGAGE_FRAMES:
                        lkas_engaged = True
                else:
                    if lkas_engaged:
                        print(f"LKAS disengaged: lane confidence {result.confidence:.2f}")
                    lkas_requested = False
                    lkas_engaged = False
                    confident_frames = 0
                    controller.reset()

            if lkas_engaged and result.error_px is not None:
                last_command = controller.update(result.error_px, display.shape[1], dt)
                steering = last_command.steering
            else:
                last_command = SteeringCommand(manual_steering, None)
                steering = manual_steering

            steering = float(np.clip(steering, -cfg.MAX_ABS_STEERING, cfg.MAX_ABS_STEERING))
            car.read_write_std(throttle, steering, leds)

            fps = 1.0 / dt
            fps_filtered = fps if fps_filtered == 0 else 0.9 * fps_filtered + 0.1 * fps
            mode = "ENGAGED" if lkas_engaged else ("ARMING" if lkas_requested else "OFF")
            mode_color = (
                (0, 255, 0) if lkas_engaged
                else ((0, 255, 255) if lkas_requested else (0, 165, 255))
            )
            gear = "R" if reverse else "D"

            cv2.putText(display, f"FPS: {fps_filtered:.1f}", (15, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            cv2.putText(display,
                        f"LKAS: {mode} ({confident_frames}/{cfg.LKAS_ENGAGE_FRAMES})",
                        (15, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.60, mode_color, 2)
            cv2.putText(display,
                        f"gear={gear} throttle={throttle:+.2f} brake={brake:.2f} steering={steering:+.2f}",
                        (15, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
            cv2.putText(display,
                        "G920: wheel/pedals | R1 reverse | A LKAS | B stop | Q/ESC quit",
                        (15, display.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (255, 255, 255), 1)

            cv2.imshow("QCar2 - G920 Virtual Lane Tracker", display)
            if cfg.SHOW_EDGE_WINDOW:
                debug = cv2.hconcat([
                    cv2.cvtColor(result.edges, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(result.roi_edges, cv2.COLOR_GRAY2BGR),
                ])
                cv2.putText(debug, "Canny edges", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
                cv2.putText(debug, "ROI used by Hough", (debug.shape[1] // 2 + 10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("QCar2 - G920 Lane Tracker Debug", debug)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                running = False

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:
        print("\nERROR:", exc)
        return 1
    finally:
        if car is not None:
            try:
                car.read_write_std(0.0, 0.0, np.zeros(8, dtype=int))
            except Exception:
                pass
            try:
                car.terminate()
            except Exception:
                pass
        if camera is not None:
            try:
                camera.terminate()
            except Exception:
                pass
        if g920 is not None:
            try:
                g920.terminate()
            except Exception:
                pass
        cv2.destroyAllWindows()
        if qlabs is not None:
            try:
                qlabs.close()
            except Exception:
                pass
        print("G920 lane tracker closed. QCar2 command set to zero.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
