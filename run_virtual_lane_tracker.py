"""Virtual QCar2 lane tracking + manual keyboard driving.

IMPORTANT:
    This program does NOT provide autonomous steering.
    W/A/S/D are the only source of vehicle motion.

Normal use:
    1. Open QLabs.
    2. Load the Open Road workspace.
    3. Run: python run_virtual_lane_tracker.py
    4. If PAL asks which virtual QCar you use, enter 2.

Keys (held):
    W        forward
    S        reverse
    A        steer left
    D        steer right
    SPACE    immediate manual stop
    Q / ESC  quit
"""

import sys
import time
import threading

import cv2
import numpy as np

import settings as cfg
from lane_detector import LaneTracker, draw_overlay


class KeyState:
    """Track held keyboard keys using pynput so driving is independent of OpenCV focus."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pressed = set()
        self.quit_requested = False

    def press(self, name: str):
        with self._lock:
            self._pressed.add(name)

    def release(self, name: str):
        with self._lock:
            self._pressed.discard(name)

    def is_down(self, name: str) -> bool:
        with self._lock:
            return name in self._pressed

    def clear_drive_keys(self):
        with self._lock:
            self._pressed.difference_update({"w", "a", "s", "d"})


def start_keyboard_listener(state: KeyState):
    try:
        from pynput import keyboard
    except ImportError as exc:
        raise RuntimeError(
            "Missing 'pynput'. Install it with: python -m pip install pynput"
        ) from exc

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
        if name in {"w", "a", "s", "d"}:
            state.press(name)
        elif name == "space":
            state.clear_drive_keys()
        elif name in {"q", "esc"}:
            state.quit_requested = True
            state.clear_drive_keys()
            return False

    def on_release(key):
        name = normalize(key)
        if name in {"w", "a", "s", "d"}:
            state.release(name)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    return listener


def setup_qlabs():
    """Spawn only QCar2 and start the QCar2 real-time model."""
    from qvl.qlabs import QuanserInteractiveLabs
    from qvl.qcar2 import QLabsQCar2
    from qvl.real_time import QLabsRealTime
    import pal.resources.rtmodels as rtmodels

    print("[1/3] Connecting to QLabs...")
    qlabs = QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        raise RuntimeError(
            "Could not connect to QLabs. Open QLabs and load Open Road first."
        )
    print("      QLabs connected.")

    print("[2/3] Spawning a clean QCar2 scene...")
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()
    time.sleep(0.4)

    qcar_actor = QLabsQCar2(qlabs)
    status = qcar_actor.spawn_id(
        actorNumber=cfg.QCAR_ACTOR_NUMBER,
        location=cfg.QCAR_START_POSITION,
        rotation=cfg.QCAR_START_ORIENTATION,
        scale=cfg.QCAR_SCALE,
        waitForConfirmation=True,
    )
    if status != 0:
        raise RuntimeError(f"QCar2 spawn failed (status={status}).")

    # Show the same forward RGB sensor family used for lane tracking.
    try:
        qcar_actor.possess(qcar_actor.CAMERA_RGB)
    except Exception:
        # Older QVL versions may behave differently; trailing possession is not
        # required for the tracker itself, so do not fail setup for this.
        pass

    print("      QCar2 actor 0 spawned.")

    print("[3/3] Starting QCar2 real-time model...")
    QLabsRealTime().start_real_time_model(rtmodels.QCAR2)
    time.sleep(cfg.RT_START_WAIT_S)
    print("      Real-time model started.")
    return qlabs


def manual_commands(keys: KeyState):
    throttle = 0.0
    steering = 0.0

    if keys.is_down("w") and not keys.is_down("s"):
        throttle = cfg.MANUAL_THROTTLE
    elif keys.is_down("s") and not keys.is_down("w"):
        throttle = cfg.MANUAL_REVERSE_THROTTLE

    if keys.is_down("a") and not keys.is_down("d"):
        steering = cfg.MANUAL_STEERING
    elif keys.is_down("d") and not keys.is_down("a"):
        steering = -cfg.MANUAL_STEERING

    throttle = float(np.clip(throttle, -cfg.MAX_ABS_THROTTLE, cfg.MAX_ABS_THROTTLE))
    steering = float(np.clip(steering, -cfg.MAX_ABS_STEERING, cfg.MAX_ABS_STEERING))
    return throttle, steering


def main():
    print("=" * 72)
    print("QCar2 VIRTUAL LANE TRACKER")
    print("Tracking only - NO autonomous steering")
    print("=" * 72)
    print("Open QLabs and load Open Road before running this program.\n")

    qlabs = None
    camera = None
    car = None
    listener = None

    try:
        qlabs = setup_qlabs()

        # Deliberately import PAL only after the QLabs RT model is started.
        # On the first virtual use, PAL may ask QCar1/QCar2. Choose 2.
        print("\nConnecting PAL to virtual QCar2...")
        from pal.products.qcar import QCar, QCarRealSense

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
                "PAL QCar2 connection is not valid. If prompted for virtual QCar type, choose 2."
            )

        keys = KeyState()
        listener = start_keyboard_listener(keys)
        tracker = LaneTracker()

        print("\nReady.")
        print("  Hold W = forward")
        print("  Hold S = reverse")
        print("  Hold A = steer left")
        print("  Hold D = steer right")
        print("  SPACE  = stop")
        print("  Q/ESC  = quit")
        print("\nThe green lane lines and cyan lane-center marker are VISUAL ONLY.\n")

        leds = np.zeros(8, dtype=int)
        prev_time = time.perf_counter()
        fps_filtered = 0.0

        while not keys.quit_requested:
            # -------------------------------------------------------------
            # 1) Read RGB camera
            # -------------------------------------------------------------
            new_frame = camera.read_RGB()
            if not new_frame:
                # Still send manual commands even if one camera frame is missed.
                throttle, steering = manual_commands(keys)
                car.read_write_std(throttle, steering, leds)
                cv2.waitKey(1)
                continue

            # PAL's imageBufferRGB is RGB; OpenCV drawing/display conventions are BGR.
            frame_rgb = camera.imageBufferRGB.copy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # -------------------------------------------------------------
            # 2) Lane tracking - perception only
            # -------------------------------------------------------------
            result = tracker.process(frame_bgr)
            display = draw_overlay(frame_bgr, result)

            # -------------------------------------------------------------
            # 3) Manual keyboard control - completely independent of tracker
            # -------------------------------------------------------------
            throttle, steering = manual_commands(keys)
            car.read_write_std(throttle, steering, leds)

            # -------------------------------------------------------------
            # 4) On-screen diagnostics
            # -------------------------------------------------------------
            now = time.perf_counter()
            dt = max(now - prev_time, 1e-6)
            prev_time = now
            fps = 1.0 / dt
            fps_filtered = fps if fps_filtered == 0 else 0.9 * fps_filtered + 0.1 * fps

            cv2.putText(display, f"FPS: {fps_filtered:.1f}", (15, 116),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            cv2.putText(display,
                        f"MANUAL throttle={throttle:+.2f} steering={steering:+.2f}",
                        (15, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(display, "WASD manual | SPACE stop | Q/ESC quit", (15, display.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow("QCar2 - Virtual Lane Tracker", display)
            if cfg.SHOW_EDGE_WINDOW:
                debug = cv2.hconcat([
                    cv2.cvtColor(result.edges, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(result.roi_edges, cv2.COLOR_GRAY2BGR),
                ])
                cv2.putText(debug, "Canny edges", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
                cv2.putText(debug, "ROI used by Hough", (debug.shape[1] // 2 + 10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("QCar2 - Lane Tracker Debug", debug)

            # OpenCV key handling is only a fallback for quit/stop.
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                keys.quit_requested = True
            elif k == ord(" "):
                keys.clear_drive_keys()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:
        print("\nERROR:", exc)
        print("\nIf this is a PAL connection error, confirm:")
        print("  1) QLabs is open")
        print("  2) Open Road is loaded")
        print("  3) when PAL asks QCar1/QCar2, enter 2")
        return 1
    finally:
        # Always stop the QCar before closing interfaces.
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

        cv2.destroyAllWindows()

        if qlabs is not None:
            try:
                qlabs.close()
            except Exception:
                pass

        print("Lane tracker closed. QCar2 command set to zero.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
