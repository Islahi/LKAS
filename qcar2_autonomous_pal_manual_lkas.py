"""Low-speed autonomous QCar2 using the proven manual-LKAS control path.

This version deliberately uses the same PAL real-time model, continuous
QCarRealSense camera, nearest-boundary LaneTracker, and pixel-error PID as the
manual keyboard launcher. It adds conservative automatic throttle.

Unlike direct QLabs velocity control, speed is a motor PWM command and is not
expressed in km/h. The starting maximum matches the manual program's low
forward-throttle value.

Controls:
    L          request/pause autonomous driving
    A / D      add left/right steering disturbance; LKAS remains active
    SPACE      immediate stop; cancels autonomy
    Q / ESC    stop and exit

Open QLabs and load Open Road. This program spawns QCar2 actor 0 and starts the
QCAR2 real-time model itself.
"""

import sys
import time

import cv2
import numpy as np

import settings as cfg
from lane_detector import LaneTracker, draw_overlay
from run_virtual_lane_tracker import (
    KeyState,
    manual_commands,
    setup_qlabs,
    start_keyboard_listener,
)
from steering_controller import SteeringCommand, SteeringPID


# Low-speed automatic longitudinal control in QCar motor PWM units.
AUTONOMOUS_MAX_THROTTLE = cfg.MANUAL_THROTTLE
THROTTLE_ACCEL_RATE_PER_S = 0.040
THROTTLE_DECEL_RATE_PER_S = 0.160

# Maintain a request through brief perception loss. The car slows toward zero,
# then automatically re-engages after this many fresh confident frames.
REENGAGE_CONFIDENT_FRAMES = cfg.LKAS_ENGAGE_FRAMES

# Keyboard disturbance in PAL/QCar steering units. Positive is left. The final
# LKAS-plus-noise command is separately bounded below the manual steering limit.
KEYBOARD_NOISE_STEERING = 0.070
LKAS_WITH_NOISE_MAX_ABS_STEERING = 0.280

def approach(value, target, rate_per_s, dt):
    max_step = max(0.0, rate_per_s * dt)
    return float(np.clip(target, value - max_step, value + max_step))


def automatic_throttle_target(result, steering):
    """Reduce PWM for steering demand and imperfect lane confidence."""
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
    if keys.is_down("a") and not keys.is_down("d"):
        return KEYBOARD_NOISE_STEERING
    if keys.is_down("d") and not keys.is_down("a"):
        return -KEYBOARD_NOISE_STEERING
    return 0.0


def main(setup_vehicle=True, tracker_factory=LaneTracker):
    qlabs = None
    camera = None
    car = None
    listener = None
    tracker = None

    try:
        print("=" * 76)
        print("QCar2 AUTONOMOUS PAL LANE KEEPING")
        print("Manual-LKAS perception/PID + conservative automatic throttle")
        print("=" * 76)
        if setup_vehicle:
            print("Open QLabs and load Open Road before running.\n")
            qlabs = setup_qlabs()
        else:
            print(
                "Using an existing QCar2 actor and real-time model; "
                "no QLabs scene changes will be made.\n"
            )

        # PAL connects only after QLabs has spawned QCar2 and started its RT model.
        from pal.products.qcar import QCar, QCarRealSense

        camera = QCarRealSense(
            mode="RGB",
            frameWidthRGB=cfg.IMAGE_WIDTH,
            frameHeightRGB=cfg.IMAGE_HEIGHT,
            frameRateRGB=cfg.CAMERA_FPS,
            readMode=0,
        )
        car = QCar(
            readMode=0,
            pwmLimit=cfg.MAX_ABS_THROTTLE,
            steeringBias=0,
        )
        if not car.card.is_valid():
            raise RuntimeError(
                "PAL QCar2 connection is invalid. If prompted for virtual QCar type, choose 2."
            )

        keys = KeyState()
        listener = start_keyboard_listener(keys)
        tracker = tracker_factory()
        controller = SteeringPID()

        autonomy_requested = False
        autonomy_engaged = False
        manual_mode = False
        confident_frames = 0
        throttle = 0.0
        last_command = SteeringCommand(0.0, None)

        leds = np.zeros(8, dtype=int)
        previous_time = time.perf_counter()
        fps_filtered = 0.0

        print("Ready. Lane tracking runs while stopped.")
        print("  M       = toggle AUTONOMOUS / MANUAL drive mode")
        print("  L       = request/pause autonomy")
        print("  W / S   = manual-mode throttle")
        print("  A / D   = manual steering or autonomous steering noise")
        print("  SPACE   = immediate stop")
        print("  Q / ESC = quit\n")
        if hasattr(tracker, "toggle_color_editor"):
            print("  P       = open/close lane color picker\n")

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
                controller.reset()
                keys.clear_drive_keys()
                car.read_write_std(0.0, 0.0, leds)
                print(
                    "MANUAL mode: W/S throttle and A/D steering."
                    if manual_mode
                    else "AUTONOMOUS mode: press L after stable lane detection."
                )

            new_frame = camera.read_RGB()
            if not new_frame:
                if manual_mode:
                    throttle, steering = manual_commands(keys)
                    car.read_write_std(throttle, steering, leds)
                else:
                    autonomy_engaged = False
                    confident_frames = 0
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
                toggle_editor = getattr(tracker, "toggle_color_editor", None)
                if toggle_editor is not None:
                    toggle_editor()

            if keys.consume_lkas_toggle():
                if manual_mode:
                    print("L ignored in MANUAL mode; press M for AUTONOMOUS mode.")
                else:
                    autonomy_requested = not autonomy_requested
                    autonomy_engaged = False
                    confident_frames = 0
                    controller.reset()
                    print(
                        "Autonomy requested; waiting for stable lanes."
                        if autonomy_requested else "Autonomy paused."
                    )

            emergency_stop = keys.consume_stop()
            if emergency_stop:
                autonomy_requested = False
                autonomy_engaged = False
                confident_frames = 0
                throttle = 0.0
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

            if autonomy_requested and not manual_mode:
                if confidence_ok:
                    confident_frames = min(
                        confident_frames + 1, REENGAGE_CONFIDENT_FRAMES
                    )
                    if confident_frames >= REENGAGE_CONFIDENT_FRAMES:
                        if not autonomy_engaged:
                            print("Autonomy engaged.")
                        autonomy_engaged = True
                else:
                    if autonomy_engaged:
                        print("Lane confidence lost; slowing and waiting to reacquire.")
                    autonomy_engaged = False
                    confident_frames = 0
                    controller.reset()

            if manual_mode:
                throttle, steering = manual_commands(keys)
                last_command = SteeringCommand(steering, None)
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
                throttle_rate = (
                    THROTTLE_ACCEL_RATE_PER_S
                    if target_throttle > throttle
                    else THROTTLE_DECEL_RATE_PER_S
                )
                throttle = approach(
                    throttle,
                    target_throttle,
                    throttle_rate,
                    dt,
                )
            else:
                steering = 0.0
                last_command = SteeringCommand(steering, None)
                throttle = approach(
                    throttle, 0.0, THROTTLE_DECEL_RATE_PER_S, dt
                )

            minimum_throttle = -cfg.MAX_ABS_THROTTLE if manual_mode else 0.0
            throttle = float(np.clip(
                throttle, minimum_throttle, cfg.MAX_ABS_THROTTLE
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
            elif autonomy_engaged:
                mode = "ENGAGED"
                mode_color = (0, 255, 0)
            elif autonomy_requested:
                mode = "REACQUIRING"
                mode_color = (0, 255, 255)
            else:
                mode = "PAUSED"
                mode_color = (0, 165, 255)

            cv2.putText(
                display, f"control FPS: {fps_filtered:.1f}", (15, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1
            )
            cv2.putText(
                display,
                f"AUTONOMY: {mode} ({confident_frames}/{REENGAGE_CONFIDENT_FRAMES})",
                (15, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.58, mode_color, 2
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
                (255, 255, 255), 1
            )
            cv2.putText(
                display,
                "M mode | W/S throttle | A/D steer/noise | L autonomy | P color"
                if hasattr(tracker, "toggle_color_editor")
                else "M mode | W/S throttle | A/D steer/noise | L autonomy",
                (15, display.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (255, 255, 255), 1
            )
            cv2.imshow("QCar2 - Autonomous PAL Manual-LKAS Logic", display)

            if cfg.SHOW_EDGE_WINDOW:
                debug = cv2.hconcat([
                    cv2.cvtColor(result.edges, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(result.roi_edges, cv2.COLOR_GRAY2BGR),
                ])
                cv2.putText(
                    debug, "All Canny edges", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                cv2.putText(
                    debug, "ROI edges used by Hough",
                    (debug.shape[1] // 2 + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                cv2.imshow("QCar2 - Autonomous PAL Lane Probe", debug)

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
        if tracker is not None:
            close_editor = getattr(tracker, "close_color_editor", None)
            if close_editor is not None:
                close_editor()
        cv2.destroyAllWindows()
        if qlabs is not None:
            try:
                qlabs.close()
            except Exception:
                pass
        print("PAL autonomous controller closed; QCar2 command set to zero.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
