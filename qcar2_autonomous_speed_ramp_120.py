"""Autonomous QLabs QCar2 speed staircase to 120 km/h.

The commanded speed starts at 0 km/h, increases by 10 km/h every five
seconds, and remains at 120 km/h. Steering stays centered.

This program drives an already-spawned QCar2 actor directly through QLabs.
Do not start the QCAR2 real-time model at the same time.

Controls:
    ESC     stop and exit
    Ctrl+C  stop and exit
"""

import threading
import time

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2


QCAR_ACTOR_NUMBER = 0
CONTROL_RATE_HZ = 30.0

SPEED_STEP_KMH = 10.0
STEP_INTERVAL_S = 5.0
TARGET_SPEED_KMH = 120.0

POSSESS_FRONT_CAMERA = True

stop_event = threading.Event()


def speed_for_elapsed_time(elapsed_s):
    """Return the requested 10 km/h staircase speed for elapsed time."""
    completed_intervals = int(max(0.0, elapsed_s) // STEP_INTERVAL_S)
    return min(completed_intervals * SPEED_STEP_KMH, TARGET_SPEED_KMH)


def escape_listener():
    """Monitor ESC without requiring the terminal to have keyboard focus."""
    try:
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.esc:
                stop_event.set()
                return False

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
    except Exception:
        # Ctrl+C remains available if pynput cannot start a listener.
        return


def command_vehicle(qcar, speed_kmh, brake=False):
    """Send a centered-steering speed command and request actor state."""
    return qcar.set_velocity_and_request_state(
        forward=float(speed_kmh) / 3.6,
        turn=0.0,
        headlights=False,
        leftTurnSignal=False,
        rightTurnSignal=False,
        brakeSignal=bool(brake),
        reverseSignal=False,
    )


def main():
    qlabs = None
    qcar = None
    listener = None

    try:
        print("QCar2 autonomous speed staircase")
        print("0 -> 120 km/h in 10 km/h steps every 5 seconds")
        print("Steering command: 0 rad (straight)")
        print("Press ESC or Ctrl+C to stop.\n")

        qlabs = QuanserInteractiveLabs()
        print("Connecting to QLabs...")
        if not qlabs.open("localhost"):
            raise RuntimeError(
                "Unable to connect to QLabs. Open QLabs and load Open Road first."
            )

        qcar = QLabsQCar2(qlabs)
        qcar.actorNumber = QCAR_ACTOR_NUMBER
        if not qcar.ping():
            raise RuntimeError(
                f"QCar2 actor {QCAR_ACTOR_NUMBER} was not found. "
                "Run the Open Road setup that spawns actor 0 first."
            )

        if POSSESS_FRONT_CAMERA:
            qcar.possess(qcar.CAMERA_CSI_FRONT)

        # Establish an explicit safe initial command before starting the timer.
        status, _, _, _, _ = command_vehicle(qcar, 0.0, brake=True)
        if not status:
            raise RuntimeError("QLabs rejected the initial stop command.")

        listener = threading.Thread(target=escape_listener, daemon=True)
        listener.start()

        start_time = time.monotonic()
        previous_speed = None
        period_s = 1.0 / CONTROL_RATE_HZ

        while not stop_event.is_set():
            loop_start = time.monotonic()
            elapsed_s = loop_start - start_time
            requested_speed_kmh = speed_for_elapsed_time(elapsed_s)

            if requested_speed_kmh != previous_speed:
                if requested_speed_kmh < TARGET_SPEED_KMH:
                    print(
                        f"t={elapsed_s:5.1f} s | command={requested_speed_kmh:5.1f} km/h"
                    )
                else:
                    print(
                        f"t={elapsed_s:5.1f} s | command={TARGET_SPEED_KMH:5.1f} km/h "
                        "| target reached; holding constant"
                    )
                previous_speed = requested_speed_kmh

            status, _, _, front_hit, rear_hit = command_vehicle(
                qcar, requested_speed_kmh
            )
            if not status:
                raise RuntimeError("QLabs rejected a velocity command.")

            if front_hit or rear_hit:
                print("Collision detected; stopping.")
                stop_event.set()
                break

            remaining_s = period_s - (time.monotonic() - loop_start)
            if remaining_s > 0.0:
                time.sleep(remaining_s)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1
    finally:
        stop_event.set()
        if qcar is not None:
            try:
                command_vehicle(qcar, 0.0, brake=True)
            except Exception:
                pass
        if listener is not None:
            listener.join(timeout=0.5)
        if qlabs is not None:
            try:
                qlabs.close()
            except Exception:
                pass
        print("QCar2 stopped safely.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
