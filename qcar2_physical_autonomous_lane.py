"""Autonomous yellow-lane controller for a physical QCar2.

Run this file directly on the QCar2 Jetson. It uses the onboard QCar2 HIL card
and front Intel RealSense camera through Quanser PAL. It never connects to
QLabs and never creates or configures a virtual vehicle.

The car is commanded to zero at startup and remains stopped until L is pressed
and both lane boundaries have been detected confidently for several frames.
For the first test, raise the drive wheels off the floor and verify steering
direction and SPACE emergency stop before putting the car on the track.

Controls:
    M          toggle AUTONOMOUS / MANUAL drive mode
    W / S      manual-mode forward/reverse throttle
    L          request/pause autonomous driving
    A / D      manual steering, or autonomous steering disturbance
    P          open/close yellow HSV and camera-center calibration window
    SPACE      immediate stop; cancels autonomy
    Q / ESC    stop and exit
"""

import sys
import time

import cv2
import numpy as np

import settings as cfg
from lane_detector import draw_overlay
from run_virtual_lane_tracker import KeyState, start_keyboard_listener
from steering_controller import SteeringCommand, SteeringPID
from yellow_lane_detector import YellowLaneTracker


# ---------------------------------------------------------------------------
# Physical hardware limits
# ---------------------------------------------------------------------------
# QCar throttle is a PWM command, not km/h. These conservative values are
# deliberately lower than the virtual program's ceiling.
PHYSICAL_PWM_LIMIT = 0.080
AUTONOMOUS_MAX_THROTTLE = 0.080
MANUAL_FORWARD_THROTTLE = 0.080
MANUAL_REVERSE_THROTTLE = -0.050

# PAL task-based I/O settings for the onboard QCar2 card and RealSense.
QCAR_IO_FREQUENCY_HZ = 500
QCAR_READ_MODE = 1
REALSENSE_READ_MODE = 1

# Mechanical steering calibration, in radians. Keep zero until measured.
PHYSICAL_STEERING_BIAS = 0.0

# Longitudinal command slew rates in PWM units per second.
THROTTLE_ACCEL_RATE_PER_S = 0.040
THROTTLE_DECEL_RATE_PER_S = 0.160

# Lane-lock and driver-disturbance settings from the accepted virtual logic.
REENGAGE_CONFIDENT_FRAMES = cfg.LKAS_ENGAGE_FRAMES
KEYBOARD_NOISE_STEERING = 0.070
LKAS_WITH_NOISE_MAX_ABS_STEERING = 0.280

# One fresh boundary: continue PID briefly using the tracker's recent estimate.
DEGRADED_TURN_GRACE_S = 1.0
DEGRADED_TURN_MAX_THROTTLE = 0.055

# No fresh boundary: hold the last PID steering command, straight or turning.
BLIND_COMMAND_HOLD_S = 2.0
BLIND_HOLD_MAX_THROTTLE = 0.040

WINDOW_TITLE = "Physical QCar2 - Autonomous Yellow-Lane Controller"
PROBE_TITLE = "Physical QCar2 - Lane Detection Probe"


def approach(value, target, rate_per_s, dt):
    max_step = max(0.0, rate_per_s * dt)
    return float(np.clip(target, value - max_step, value + max_step))


def automatic_throttle_target(result, steering):
    """Reduce physical PWM for steering demand and imperfect confidence."""
    steering_ratio = float(np.clip(
        abs(steering) / max(cfg.LKAS_MAX_ABS_STEERING, 1e-6), 0.0, 1.0
    ))
    steering_factor = float(np.interp(
        steering_ratio,
        np.array([0.0, 0.20, 0.45, 0.70, 1.0]),
        np.array([1.0, 0.95, 0.78, 0.55, 0.35]),
    ))
    confidence_factor = float(np.interp(
        result.confidence,
        np.array([cfg.LKAS_MIN_CONFIDENCE, 0.75, 0.90, 1.0]),
        np.array([0.45, 0.65, 0.85, 1.0]),
    ))
    return AUTONOMOUS_MAX_THROTTLE * min(
        steering_factor, confidence_factor
    )


def keyboard_steering_noise(keys):
    # PAL/QCar2 steering is positive left and negative right.
    if keys.is_down("a") and not keys.is_down("d"):
        return KEYBOARD_NOISE_STEERING
    if keys.is_down("d") and not keys.is_down("a"):
        return -KEYBOARD_NOISE_STEERING
    return 0.0


def physical_manual_commands(keys):
    throttle = 0.0
    steering = 0.0

    if keys.is_down("w") and not keys.is_down("s"):
        throttle = MANUAL_FORWARD_THROTTLE
    elif keys.is_down("s") and not keys.is_down("w"):
        throttle = MANUAL_REVERSE_THROTTLE

    if keys.is_down("a") and not keys.is_down("d"):
        steering = cfg.MANUAL_STEERING
    elif keys.is_down("d") and not keys.is_down("a"):
        steering = -cfg.MANUAL_STEERING

    return (
        float(np.clip(throttle, -PHYSICAL_PWM_LIMIT, PHYSICAL_PWM_LIMIT)),
        float(np.clip(
            steering, -cfg.MAX_ABS_STEERING, cfg.MAX_ABS_STEERING
        )),
    )


def _as_scalar(value):
    values = np.asarray(value).reshape(-1)
    return float(values[0]) if values.size else float("nan")


def _open_physical_interfaces():
    """Open only onboard QCar2 hardware and reject virtual PAL targets."""
    from pal.products import qcar as pal_qcar

    if not bool(getattr(pal_qcar, "IS_PHYSICAL_QCAR", False)):
        raise RuntimeError(
            "Physical-mode safety check failed: Quanser PAL does not identify "
            "this computer as the physical QCar. Run this file directly on "
            "the QCar2 Jetson, not on the Windows/QLabs computer."
        )

    car = pal_qcar.QCar(
        readMode=QCAR_READ_MODE,
        frequency=QCAR_IO_FREQUENCY_HZ,
        pwmLimit=PHYSICAL_PWM_LIMIT,
        steeringBias=PHYSICAL_STEERING_BIAS,
    )
    if not car.card.is_valid():
        try:
            car.terminate()
        except Exception:
            pass
        raise RuntimeError("PAL could not open the onboard QCar2 hardware card.")
    if not bool(getattr(car, "hardware", False)):
        car.terminate()
        raise RuntimeError("PAL opened a virtual vehicle instead of hardware.")
    if getattr(car, "carType", None) != 2:
        car.terminate()
        raise RuntimeError("The connected vehicle is not configured as QCar2.")

    leds = np.zeros(8, dtype=int)
    car.read_write_std(0.0, 0.0, leds)

    try:
        camera = pal_qcar.QCarRealSense(
            mode="RGB",
            frameWidthRGB=cfg.IMAGE_WIDTH,
            frameHeightRGB=cfg.IMAGE_HEIGHT,
            frameRateRGB=cfg.CAMERA_FPS,
            readMode=REALSENSE_READ_MODE,
        )
    except Exception:
        car.read_write_std(0.0, 0.0, leds)
        car.terminate()
        raise

    return car, camera


def main():
    car = None
    camera = None
    listener = None
    tracker = None

    try:
        print("=" * 76)
        print("PHYSICAL QCar2 AUTONOMOUS YELLOW-LANE CONTROLLER")
        print("Onboard RealSense + nearest-boundary detector + PID steering")
        print("=" * 76)
        print("No QLabs connection or virtual setup will be performed.\n")

        car, camera = _open_physical_interfaces()
        keys = KeyState()
        listener = start_keyboard_listener(keys)
        tracker = YellowLaneTracker()
        controller = SteeringPID()

        autonomy_requested = False
        autonomy_engaged = False
        manual_mode = False
        confident_frames = 0
        throttle = 0.0
        last_command = SteeringCommand(0.0, None)
        degraded_start_time = None
        degraded_turn_active = False
        degraded_elapsed_s = 0.0
        blind_start_time = None
        blind_hold_active = False
        blind_elapsed_s = 0.0

        leds = np.zeros(8, dtype=int)
        previous_time = time.perf_counter()
        fps_filtered = 0.0

        print("Hardware connected. The motor command is zero.")
        print("  M       = toggle AUTONOMOUS / MANUAL drive mode")
        print("  L       = request/pause autonomy")
        print("  W / S   = manual-mode throttle")
        print("  A / D   = manual steering or autonomous steering noise")
        print("  P       = yellow HSV and camera-center calibration")
        print("  SPACE   = immediate stop")
        print("  Q / ESC = stop and quit\n")

        while not keys.quit_requested:
            loop_start = time.perf_counter()
            dt = float(np.clip(loop_start - previous_time, 0.001, 0.20))
            previous_time = loop_start

            if keys.consume_drive_mode_toggle():
                manual_mode = not manual_mode
                autonomy_requested = False
                autonomy_engaged = False
                confident_frames = 0
                throttle = 0.0
                degraded_start_time = None
                degraded_turn_active = False
                blind_start_time = None
                blind_hold_active = False
                controller.reset()
                keys.clear_drive_keys()
                car.read_write_std(0.0, 0.0, leds)
                print(
                    "MANUAL mode: hold W/S and A/D to drive."
                    if manual_mode
                    else "AUTONOMOUS mode: press L after stable detection."
                )

            new_frame = camera.read_RGB()
            if not new_frame:
                if manual_mode:
                    throttle, steering = physical_manual_commands(keys)
                    car.read_write_std(throttle, steering, leds)
                else:
                    autonomy_engaged = False
                    confident_frames = 0
                    degraded_start_time = None
                    degraded_turn_active = False
                    blind_start_time = None
                    blind_hold_active = False
                    controller.reset()
                    throttle = approach(
                        throttle, 0.0, THROTTLE_DECEL_RATE_PER_S, dt
                    )
                    car.read_write_std(throttle, 0.0, leds)
                cv2.waitKey(1)
                continue

            frame_rgb = camera.imageBufferRGB.copy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            result = tracker.process(frame_bgr)
            display = draw_overlay(frame_bgr, result)

            if keys.consume_color_toggle():
                tracker.toggle_color_editor()

            if keys.consume_lkas_toggle():
                if manual_mode:
                    print("L ignored in MANUAL mode; press M for AUTONOMOUS.")
                else:
                    autonomy_requested = not autonomy_requested
                    autonomy_engaged = False
                    confident_frames = 0
                    degraded_start_time = None
                    degraded_turn_active = False
                    blind_start_time = None
                    blind_hold_active = False
                    controller.reset()
                    print(
                        "Autonomy requested; waiting for stable lanes."
                        if autonomy_requested else "Autonomy paused."
                    )

            if keys.consume_stop():
                autonomy_requested = False
                autonomy_engaged = False
                confident_frames = 0
                throttle = 0.0
                degraded_start_time = None
                degraded_turn_active = False
                blind_start_time = None
                blind_hold_active = False
                controller.reset()
                car.read_write_std(0.0, 0.0, leds)
                print("Emergency stop; autonomy cancelled.")

            steering_noise = (
                0.0 if manual_mode else keyboard_steering_noise(keys)
            )

            confidence_ok = (
                result.error_px is not None
                and result.left_detected
                and result.right_detected
                and result.confidence >= cfg.LKAS_MIN_CONFIDENCE
            )
            degraded_tracking_ok = (
                result.error_px is not None
                and result.left is not None
                and result.right is not None
                and (result.left_detected or result.right_detected)
            )
            degraded_turn_active = False
            degraded_elapsed_s = 0.0
            blind_hold_active = False
            blind_elapsed_s = 0.0

            if autonomy_requested and not manual_mode:
                if confidence_ok:
                    degraded_start_time = None
                    blind_start_time = None
                    confident_frames = min(
                        confident_frames + 1, REENGAGE_CONFIDENT_FRAMES
                    )
                    if confident_frames >= REENGAGE_CONFIDENT_FRAMES:
                        if not autonomy_engaged:
                            print("Autonomy engaged.")
                        autonomy_engaged = True
                elif autonomy_engaged and degraded_tracking_ok:
                    blind_start_time = None
                    if degraded_start_time is None:
                        degraded_start_time = time.perf_counter()
                        print(
                            "One boundary fresh; continuing PID slowly from "
                            "the recent lane estimate."
                        )
                    degraded_elapsed_s = (
                        time.perf_counter() - degraded_start_time
                    )
                    if degraded_elapsed_s <= DEGRADED_TURN_GRACE_S:
                        degraded_turn_active = True
                    else:
                        print("One-line grace expired; slowing to reacquire.")
                        autonomy_engaged = False
                        confident_frames = 0
                        degraded_start_time = None
                        controller.reset()
                elif autonomy_engaged:
                    degraded_start_time = None
                    if blind_start_time is None:
                        blind_start_time = time.perf_counter()
                        print(
                            "Both boundaries lost; holding the last steering "
                            "command slowly."
                        )
                    blind_elapsed_s = time.perf_counter() - blind_start_time
                    if blind_elapsed_s <= BLIND_COMMAND_HOLD_S:
                        blind_hold_active = True
                    else:
                        print("Blind hold expired; slowing to reacquire.")
                        autonomy_engaged = False
                        confident_frames = 0
                        blind_start_time = None
                        controller.reset()
                else:
                    autonomy_engaged = False
                    confident_frames = 0
                    degraded_start_time = None
                    blind_start_time = None
                    controller.reset()

            if manual_mode:
                throttle, steering = physical_manual_commands(keys)
                last_command = SteeringCommand(steering, None)
            elif autonomy_engaged and blind_hold_active:
                # Freeze the last valid PID output: zero remains straight and a
                # nonzero command continues the same turn through the blind gap.
                steering = float(np.clip(
                    last_command.steering + steering_noise,
                    -LKAS_WITH_NOISE_MAX_ABS_STEERING,
                    LKAS_WITH_NOISE_MAX_ABS_STEERING,
                ))
                # Never accelerate without a fresh visual reference.
                target_throttle = min(throttle, BLIND_HOLD_MAX_THROTTLE)
                throttle = approach(
                    throttle,
                    target_throttle,
                    THROTTLE_DECEL_RATE_PER_S,
                    dt,
                )
            elif autonomy_engaged and result.error_px is not None:
                last_command = controller.update(
                    result.error_px, display.shape[1], dt
                )
                steering = float(np.clip(
                    last_command.steering + steering_noise,
                    -LKAS_WITH_NOISE_MAX_ABS_STEERING,
                    LKAS_WITH_NOISE_MAX_ABS_STEERING,
                ))
                target_throttle = automatic_throttle_target(result, steering)
                if degraded_turn_active:
                    target_throttle = min(
                        target_throttle, DEGRADED_TURN_MAX_THROTTLE
                    )
                throttle_rate = (
                    THROTTLE_ACCEL_RATE_PER_S
                    if target_throttle > throttle
                    else THROTTLE_DECEL_RATE_PER_S
                )
                throttle = approach(
                    throttle, target_throttle, throttle_rate, dt
                )
            else:
                steering = 0.0
                last_command = SteeringCommand(steering, None)
                throttle = approach(
                    throttle, 0.0, THROTTLE_DECEL_RATE_PER_S, dt
                )

            minimum_throttle = -PHYSICAL_PWM_LIMIT if manual_mode else 0.0
            throttle = float(np.clip(
                throttle, minimum_throttle, PHYSICAL_PWM_LIMIT
            ))
            steering = float(np.clip(
                steering, -cfg.MAX_ABS_STEERING, cfg.MAX_ABS_STEERING
            ))
            car.read_write_std(throttle, steering, leds)

            fps = 1.0 / dt
            fps_filtered = fps if fps_filtered == 0.0 else (
                0.9 * fps_filtered + 0.1 * fps
            )
            if manual_mode:
                mode = "MANUAL"
                mode_color = (255, 0, 255)
            elif blind_hold_active:
                mode = (
                    f"BLIND-HOLD {blind_elapsed_s:.1f}/"
                    f"{BLIND_COMMAND_HOLD_S:.1f}s"
                )
                mode_color = (0, 0, 255)
            elif degraded_turn_active:
                mode = (
                    f"TURN-GRACE {degraded_elapsed_s:.1f}/"
                    f"{DEGRADED_TURN_GRACE_S:.1f}s"
                )
                mode_color = (0, 255, 255)
            elif autonomy_engaged:
                mode = "ENGAGED"
                mode_color = (0, 255, 0)
            elif autonomy_requested:
                mode = "REACQUIRING"
                mode_color = (0, 255, 255)
            else:
                mode = "PAUSED"
                mode_color = (0, 165, 255)

            battery_v = _as_scalar(car.batteryVoltage)
            speed_mps = _as_scalar(car.motorTach)
            cv2.putText(
                display, f"control FPS: {fps_filtered:.1f}", (15, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1,
            )
            cv2.putText(
                display,
                f"AUTONOMY: {mode} ({confident_frames}/{REENGAGE_CONFIDENT_FRAMES})",
                (15, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.58, mode_color, 2,
            )
            cv2.putText(
                display,
                (
                    f"MANUAL PWM throttle={throttle:+.3f} steering={steering:+.3f}"
                    if manual_mode
                    else f"PWM throttle={throttle:.3f}/{AUTONOMOUS_MAX_THROTTLE:.3f} "
                    f"auto={last_command.steering:+.3f} noise={steering_noise:+.3f} "
                    f"final={steering:+.3f}"
                ),
                (15, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1,
            )
            cv2.putText(
                display,
                f"physical speed={speed_mps:.2f} m/s battery={battery_v:.1f} V",
                (15, 224), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1,
            )
            cv2.putText(
                display,
                "M mode | W/S throttle | A/D steer/noise | L autonomy | P color",
                (15, display.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (255, 255, 255), 1,
            )
            cv2.imshow(WINDOW_TITLE, display)

            if cfg.SHOW_EDGE_WINDOW:
                debug = cv2.hconcat([
                    cv2.cvtColor(result.edges, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(result.roi_edges, cv2.COLOR_GRAY2BGR),
                ])
                cv2.putText(
                    debug, "All detector edges", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                cv2.putText(
                    debug, "ROI edges used by Hough",
                    (debug.shape[1] // 2 + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                cv2.imshow(PROBE_TITLE, debug)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                keys.quit_requested = True
            elif key == ord(" "):
                keys.request_stop()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:
        print(f"\nERROR: {exc}")
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
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
        if tracker is not None:
            tracker.close_color_editor()
        cv2.destroyAllWindows()
        print("Physical controller closed; QCar2 command set to zero.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
