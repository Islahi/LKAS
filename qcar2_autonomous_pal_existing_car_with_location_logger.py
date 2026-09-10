r"""Run the existing PAL lane keeper and log the driven Open Road path.

The vehicle-control behavior is unchanged from
``qcar2_autonomous_pal_existing_car.py``. A background logger connects to
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

from qcar2_autonomous_pal_manual_lkas import main
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

    controller_exit_code = 0
    controller_error: BaseException | None = None
    try:
        result = main(setup_vehicle=False, tracker_factory=YellowLaneTracker)
        if isinstance(result, int):
            controller_exit_code = result
    except KeyboardInterrupt:
        print("Lane controller interrupted by user.")
    except SystemExit as exc:
        if isinstance(exc.code, int):
            controller_exit_code = exc.code
    except BaseException as exc:
        controller_error = exc
    finally:
        logger.stop()

    try:
        backup_path = logger.save()
    except Exception as exc:
        print(f"Location data was not saved: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {len(logger.points)} measured [x, y, z] road points.")
    print(f"Updated: {args.reference.resolve()}")
    print(f"Backup:  {backup_path}")

    if controller_error is not None:
        print(f"Lane controller stopped with an error: {controller_error}", file=sys.stderr)
        return 1
    return controller_exit_code


if __name__ == "__main__":
    sys.exit(run())
