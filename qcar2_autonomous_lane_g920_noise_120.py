"""Autonomous QLabs QCar2 lane keeping with a G920 disturbance input.

This is a standalone direct-QLabs controller. Existing LKAS, G920, and speed
ramp programs are not imported or modified.

Operation:
    * Acquire and lock the two boundaries of the initial lane.
    * Press A on the G920 to start/pause autonomous driving.
    * Accelerate smoothly toward the configured validation speed on safe straights.
    * Reduce speed for curves, weak detection, or lost boundaries, then recover.
    * Add G920 wheel input as a bounded +/-6 degree steering disturbance.
    * Correct back toward the locked initial lane after the wheel is released.

Controls:
    G920 wheel      additive steering disturbance (+/-6 degrees)
    G920 A          start/pause autonomous driving
    G920 B          emergency stop and pause
    G920 X          unstuck after collision; reacquire lane before restart
    Brake pedal     emergency stop and pause
    ESC / Ctrl+C    stop and exit

QLabs setup:
    Open QLabs and load the Open Road workspace. This program stops conflicting
    real-time models, clears previously spawned actors, and spawns QCar2 actor 0.
"""

import math
import threading
import time

import cv2
import numpy as np

import settings as cfg
from lane_detector import LaneTracker, TrackingResult, draw_overlay
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2


# -----------------------------------------------------------------------------
# QLabs and adaptive speed control
# -----------------------------------------------------------------------------
QCAR_ACTOR_NUMBER = 0
CONTROL_RATE_HZ = 25.0
CAMERA = QLabsQCar2.CAMERA_RGB

MAX_STRAIGHT_SPEED_KMH = 60.0
# Raise the validation cap gradually only after full-course tests are reliable.
# 6 km/h/s reaches 60 km/h in about 10 seconds on a clear straight.
MAX_ACCEL_KMH_PER_S = 1.0
# Normal curve/confidence braking. Emergency stop commands zero immediately.
MAX_DECEL_KMH_PER_S = 20.0

# Start with a controlled crawl and require stable centering before cruising.
ALIGNMENT_CRAWL_SPEED_KMH = 7.0
ALIGNMENT_LATERAL_TOLERANCE = 0.06
ALIGNMENT_HEADING_TOLERANCE = 0.045
ALIGNMENT_STABLE_FRAMES = 20

# Steering-angle limits used by QLabsQCar2.set_velocity_and_request_state().
MAX_AUTO_STEERING_DEG = 12.0
MAX_TOTAL_STEERING_DEG = 15.0
MAX_AUTO_STEERING_RATE_DEG_S = 90.0

# Lane controller gains. Lateral and heading errors are normalized to roughly
# -1..1 before these gains are applied.
LATERAL_GAIN_DEG = 22.0
HEADING_GAIN_DEG = 30.0
STEERING_D_GAIN_DEG = 1.8
DERIVATIVE_FILTER = 0.82

# Look-ahead rows used to estimate lane heading/curvature.
NEAR_Y_FRACTION = 0.88
FAR_Y_FRACTION = 0.52

# Initial-lane association gates. After lock, boundaries are matched to their
# previous positions rather than selected again around the camera center.
LOCK_BOTTOM_GATE_PX = 170.0
LOCK_FAR_GATE_PX = 120.0
LOCK_ACQUIRE_FRAMES = 6
LOCK_RESET_AFTER_LOST_FRAMES = 150
LOCK_REFERENCE_SMOOTHING = 0.55
AUTONOMOUS_LINE_SMOOTHING = 0.35

# A remembered lane width allows short recovery with exactly one fresh boundary.
SINGLE_BOUNDARY_MAX_SPEED_KMH = 25.0
MAX_SINGLE_BOUNDARY_RECOVERY_FRAMES = 75

# -----------------------------------------------------------------------------
# Logitech G920
# -----------------------------------------------------------------------------
G920_STEERING_SIGN = -1.0
G920_STEERING_DEADZONE = 0.05
G920_PEDAL_DEADZONE = 0.05
G920_MAX_DISTURBANCE_DEG = 6.0

START_BUTTON_NAMES = ("buttonA",)
STOP_BUTTON_NAMES = ("buttonB",)
UNSTUCK_BUTTON_NAMES = ("buttonX",)
UNSTUCK_LIFT_M = 1.0


stop_event = threading.Event()


def button_value(g920, names):
    for name in names:
        if hasattr(g920, name):
            return bool(getattr(g920, name)), name
    return False, None


def normalized_positive_axis(value, deadzone):
    value = float(np.clip(value, 0.0, 1.0))
    if value <= deadzone:
        return 0.0
    return (value - deadzone) / (1.0 - deadzone)


def normalized_signed_axis(value, deadzone):
    value = float(np.clip(value, -1.0, 1.0))
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    scaled = (magnitude - deadzone) / (1.0 - deadzone)
    return float(np.sign(value) * scaled)


def escape_listener():
    try:
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.esc:
                stop_event.set()
                return False

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
    except Exception:
        # Ctrl+C and the OpenCV ESC handler remain available.
        return


def setup_qlabs_car():
    """Connect to Open Road and spawn a clean direct-control QCar2 actor."""
    from qvl.real_time import QLabsRealTime

    print("[1/3] Connecting to QLabs...")
    qlabs = QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        raise RuntimeError(
            "Could not connect to QLabs. Open QLabs and load Open Road first."
        )
    print("      QLabs connected.")

    print("[2/3] Clearing old actors and conflicting real-time models...")
    QLabsRealTime().terminate_all_real_time_models()
    qlabs.destroy_all_spawned_actors()
    time.sleep(0.5)

    print("[3/3] Spawning QCar2 actor 0...")
    qcar = QLabsQCar2(qlabs)
    status = qcar.spawn_id(
        actorNumber=QCAR_ACTOR_NUMBER,
        location=cfg.QCAR_START_POSITION,
        rotation=cfg.QCAR_START_ORIENTATION,
        scale=cfg.QCAR_SCALE,
        waitForConfirmation=True,
    )
    if status != 0:
        raise RuntimeError(f"QCar2 spawn failed (status={status}).")
    if not qcar.ping():
        raise RuntimeError(f"Spawned QCar2 actor {QCAR_ACTOR_NUMBER} did not respond.")
    if not qcar.possess(CAMERA):
        raise RuntimeError("Could not possess the QCar2 front RGB camera.")
    print("      QCar2 spawned and front RGB camera possessed.")
    return qlabs, qcar


class InitialLaneTracker(LaneTracker):
    """LaneTracker that preserves the initially acquired boundary identities."""

    def __init__(self):
        super().__init__()
        self.lane_locked = False
        self.acquire_frames = 0
        self.pending_left = None
        self.pending_right = None
        self.reference_left = None
        self.reference_right = None
        self.reference_width_near = None
        self.reference_width_far = None
        self.single_boundary_frames = 0

    @staticmethod
    def _smooth(previous, current):
        """Faster smoothing used only by this autonomous controller."""
        if previous is None:
            return current
        return type(current)(
            a=(
                AUTONOMOUS_LINE_SMOOTHING * previous.a
                + (1.0 - AUTONOMOUS_LINE_SMOOTHING) * current.a
            ),
            b=(
                AUTONOMOUS_LINE_SMOOTHING * previous.b
                + (1.0 - AUTONOMOUS_LINE_SMOOTHING) * current.b
            ),
            weight=current.weight,
        )

    @staticmethod
    def _blend_line(previous, current, smoothing):
        if previous is None:
            return current
        return type(current)(
            a=smoothing * previous.a + (1.0 - smoothing) * current.a,
            b=smoothing * previous.b + (1.0 - smoothing) * current.b,
            weight=current.weight,
        )

    def _clear_lock(self):
        self.lane_locked = False
        self.acquire_frames = 0
        self.pending_left = None
        self.pending_right = None
        self.reference_left = None
        self.reference_right = None
        self.reference_width_near = None
        self.reference_width_far = None
        self.single_boundary_frames = 0
        self.prev_left = None
        self.prev_right = None
        self.left_lost_frames = 0
        self.right_lost_frames = 0

    @staticmethod
    def _association_cost(candidate, previous, y_bottom, y_far):
        bottom_error = abs(candidate.x_at(y_bottom) - previous.x_at(y_bottom))
        far_error = abs(candidate.x_at(y_far) - previous.x_at(y_far))
        if bottom_error > LOCK_BOTTOM_GATE_PX or far_error > LOCK_FAR_GATE_PX:
            return None
        return bottom_error + 0.8 * far_error

    def _associate_locked_pair(self, clusters, y_bottom, y_far):
        if self.reference_left is None or self.reference_right is None:
            return None, None

        best_pair = (None, None)
        best_cost = float("inf")
        for left_index, left_candidate in enumerate(clusters):
            left_cost = self._association_cost(
                left_candidate, self.reference_left, y_bottom, y_far
            )
            if left_cost is None:
                continue
            for right_index, right_candidate in enumerate(clusters):
                if right_index == left_index:
                    continue
                right_cost = self._association_cost(
                    right_candidate, self.reference_right, y_bottom, y_far
                )
                if right_cost is None:
                    continue
                if left_candidate.x_at(y_far) >= right_candidate.x_at(y_far):
                    continue
                cost = left_cost + right_cost
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (left_candidate, right_candidate)
        if best_pair != (None, None):
            return best_pair

        # A dashed or briefly occluded marking may yield only one match. Keep
        # the matching boundary fresh and let the other use its short fallback.
        best_left = (float("inf"), None, None)
        best_right = (float("inf"), None, None)
        for index, candidate in enumerate(clusters):
            left_cost = self._association_cost(
                candidate, self.reference_left, y_bottom, y_far
            )
            if left_cost is not None and left_cost < best_left[0]:
                best_left = (left_cost, index, candidate)
            right_cost = self._association_cost(
                candidate, self.reference_right, y_bottom, y_far
            )
            if right_cost is not None and right_cost < best_right[0]:
                best_right = (right_cost, index, candidate)

        if best_left[1] is not None and best_left[1] == best_right[1]:
            if best_left[0] <= best_right[0]:
                best_right = (float("inf"), None, None)
            else:
                best_left = (float("inf"), None, None)
        return best_left[2], best_right[2]

    @staticmethod
    def _initial_pair(clusters, vehicle_center_x, error_y, image_width):
        left_clusters = [
            line for line in clusters
            if line.x_at(error_y) < vehicle_center_x - cfg.CENTER_EXCLUSION_PX
        ]
        right_clusters = [
            line for line in clusters
            if line.x_at(error_y) > vehicle_center_x + cfg.CENTER_EXCLUSION_PX
        ]
        detected_left = max(
            left_clusters, key=lambda line: line.x_at(error_y), default=None
        )
        detected_right = min(
            right_clusters, key=lambda line: line.x_at(error_y), default=None
        )
        if detected_left is None or detected_right is None:
            return None, None

        width = detected_right.x_at(error_y) - detected_left.x_at(error_y)
        min_width = cfg.MIN_LANE_WIDTH_FRACTION * image_width
        max_width = cfg.MAX_LANE_WIDTH_FRACTION * image_width
        if not (min_width <= width <= max_width):
            return None, None
        return detected_left, detected_right

    def process(self, frame_bgr):
        edges, roi_edges, polygon = self.preprocess(frame_bgr)
        height, width = frame_bgr.shape[:2]
        vehicle_center_x = width / 2.0
        y_bottom = float(height - 1)
        error_y = float(NEAR_Y_FRACTION * (height - 1))
        far_y = float(FAR_Y_FRACTION * (height - 1))

        raw = cv2.HoughLinesP(
            roi_edges,
            cfg.HOUGH_RHO,
            np.deg2rad(cfg.HOUGH_THETA_DEG),
            cfg.HOUGH_THRESHOLD,
            np.array([]),
            minLineLength=cfg.HOUGH_MIN_LINE_LENGTH,
            maxLineGap=cfg.HOUGH_MAX_LINE_GAP,
        )

        candidates = []
        if raw is not None:
            for wrapped in raw:
                fitted = self._fit_segment(wrapped.reshape(4))
                if fitted is None:
                    continue
                bottom_x = fitted.x_at(y_bottom)
                if -0.4 * width <= bottom_x <= 1.4 * width:
                    candidates.append(fitted)

        clusters = self._cluster_candidates(candidates, y_bottom)
        if self.lane_locked:
            detected_left, detected_right = self._associate_locked_pair(
                clusters, y_bottom, far_y
            )
        else:
            detected_left, detected_right = self._initial_pair(
                clusters, vehicle_center_x, error_y, width
            )
            if detected_left is not None and detected_right is not None:
                self.pending_left = self._blend_line(
                    self.pending_left, detected_left, cfg.SMOOTHING
                )
                self.pending_right = self._blend_line(
                    self.pending_right, detected_right, cfg.SMOOTHING
                )
                self.acquire_frames += 1
                if self.acquire_frames >= LOCK_ACQUIRE_FRAMES:
                    self.lane_locked = True
                    self.reference_left = self.pending_left
                    self.reference_right = self.pending_right
            else:
                self.acquire_frames = 0
                self.pending_left = None
                self.pending_right = None

        left, self.prev_left, self.left_lost_frames = self._update_with_fallback(
            detected_left, self.prev_left, self.left_lost_frames
        )
        right, self.prev_right, self.right_lost_frames = self._update_with_fallback(
            detected_right, self.prev_right, self.right_lost_frames
        )

        # The display fallback deliberately expires after a few frames, but the
        # locked-lane reference must survive so fresh segments can be associated
        # when a dashed marking or temporary occlusion reappears.
        if self.lane_locked:
            if detected_left is not None:
                self.reference_left = self._blend_line(
                    self.reference_left,
                    detected_left,
                    LOCK_REFERENCE_SMOOTHING,
                )
            if detected_right is not None:
                self.reference_right = self._blend_line(
                    self.reference_right,
                    detected_right,
                    LOCK_REFERENCE_SMOOTHING,
                )

            if detected_left is not None and detected_right is not None:
                measured_width_near = (
                    detected_right.x_at(error_y) - detected_left.x_at(error_y)
                )
                measured_width_far = (
                    detected_right.x_at(far_y) - detected_left.x_at(far_y)
                )
                width_smoothing = LOCK_REFERENCE_SMOOTHING
                if self.reference_width_near is None:
                    self.reference_width_near = measured_width_near
                    self.reference_width_far = measured_width_far
                else:
                    self.reference_width_near = (
                        width_smoothing * self.reference_width_near
                        + (1.0 - width_smoothing) * measured_width_near
                    )
                    self.reference_width_far = (
                        width_smoothing * self.reference_width_far
                        + (1.0 - width_smoothing) * measured_width_far
                    )
                self.single_boundary_frames = 0
            elif detected_left is not None or detected_right is not None:
                self.single_boundary_frames += 1
            else:
                self.single_boundary_frames = 0

            # A long loss is more likely to be a false initial lock or a major
            # scene change. Speed confidence is already zero during this period,
            # so the vehicle decelerates before fresh acquisition is allowed.
            if (
                self.left_lost_frames >= LOCK_RESET_AFTER_LOST_FRAMES
                and self.right_lost_frames >= LOCK_RESET_AFTER_LOST_FRAMES
            ):
                self._clear_lock()
                left = None
                right = None
                detected_left = None
                detected_right = None

        lane_center_x = None
        error_px = None
        if left is not None and right is not None:
            left_x = left.x_at(error_y)
            right_x = right.x_at(error_y)
            if left_x < right_x:
                lane_center_x = (left_x + right_x) / 2.0
                error_px = lane_center_x - vehicle_center_x

        left_detected = detected_left is not None
        right_detected = detected_right is not None
        lane_width_px = None
        confidence = 0.0
        if left_detected and right_detected:
            lane_width_px = (
                detected_right.x_at(error_y) - detected_left.x_at(error_y)
            )
            min_width = cfg.MIN_LANE_WIDTH_FRACTION * width
            max_width = cfg.MAX_LANE_WIDTH_FRACTION * width
            if min_width <= lane_width_px <= max_width:
                weaker_weight = min(detected_left.weight, detected_right.weight)
                confidence = float(np.clip(
                    weaker_weight / cfg.CONFIDENCE_FULL_WEIGHT, 0.0, 1.0
                ))

        return TrackingResult(
            left=left,
            right=right,
            lane_center_x=lane_center_x,
            vehicle_center_x=vehicle_center_x,
            error_px=error_px,
            left_lost_frames=self.left_lost_frames,
            right_lost_frames=self.right_lost_frames,
            left_detected=left_detected,
            right_detected=right_detected,
            lane_width_px=lane_width_px,
            confidence=confidence,
            raw_segments=raw,
            edges=edges,
            roi_edges=roi_edges,
            roi_polygon=polygon,
        )


def lane_geometry(result, tracker, width, height):
    """Return lane geometry from two boundaries or a short one-line recovery."""
    near_y = float(NEAR_Y_FRACTION * (height - 1))
    far_y = float(FAR_Y_FRACTION * (height - 1))

    if result.left_detected and result.right_detected:
        near_x = (
            result.left.x_at(near_y) + result.right.x_at(near_y)
        ) / 2.0
        far_x = (
            result.left.x_at(far_y) + result.right.x_at(far_y)
        ) / 2.0
        mode = "TWO BOUNDARIES"
    elif (
        result.left_detected
        and result.left is not None
        and tracker.reference_width_near is not None
        and tracker.single_boundary_frames <= MAX_SINGLE_BOUNDARY_RECOVERY_FRAMES
    ):
        near_x = result.left.x_at(near_y) + tracker.reference_width_near / 2.0
        far_x = result.left.x_at(far_y) + tracker.reference_width_far / 2.0
        mode = "LEFT-ONLY RECOVERY"
    elif (
        result.right_detected
        and result.right is not None
        and tracker.reference_width_near is not None
        and tracker.single_boundary_frames <= MAX_SINGLE_BOUNDARY_RECOVERY_FRAMES
    ):
        near_x = result.right.x_at(near_y) - tracker.reference_width_near / 2.0
        far_x = result.right.x_at(far_y) - tracker.reference_width_far / 2.0
        mode = "RIGHT-ONLY RECOVERY"
    else:
        return None

    lateral = float(np.clip(
        (near_x - result.vehicle_center_x) / max(width / 2.0, 1.0), -1.0, 1.0
    ))
    lookahead_height = max(near_y - far_y, 1.0)
    heading = float(np.clip(
        (far_x - near_x) / lookahead_height, -1.0, 1.0
    ))
    return lateral, heading, near_x, near_y, far_x, far_y, mode


class LaneSteeringController:
    def __init__(self):
        self.previous_combined_error = None
        self.filtered_derivative = 0.0
        self.previous_output_deg = 0.0

    def reset(self):
        self.previous_combined_error = None
        self.filtered_derivative = 0.0
        self.previous_output_deg = 0.0

    def update(self, lateral_error, heading_error, dt):
        dt = float(np.clip(dt, 0.01, 0.15))
        combined_error = lateral_error + heading_error
        raw_derivative = 0.0
        if self.previous_combined_error is not None:
            raw_derivative = (
                combined_error - self.previous_combined_error
            ) / dt
        self.filtered_derivative = (
            DERIVATIVE_FILTER * self.filtered_derivative
            + (1.0 - DERIVATIVE_FILTER) * raw_derivative
        )

        # Positive image errors point right; direct QLabs positive turn is right.
        target_deg = -(
            LATERAL_GAIN_DEG * lateral_error
            + HEADING_GAIN_DEG * heading_error
            + STEERING_D_GAIN_DEG * self.filtered_derivative
        )
        target_deg = float(np.clip(
            target_deg, -MAX_AUTO_STEERING_DEG, MAX_AUTO_STEERING_DEG
        ))

        max_step = MAX_AUTO_STEERING_RATE_DEG_S * dt
        output_deg = float(np.clip(
            target_deg,
            self.previous_output_deg - max_step,
            self.previous_output_deg + max_step,
        ))
        self.previous_combined_error = combined_error
        self.previous_output_deg = output_deg
        return output_deg


def curve_speed_limit_kmh(auto_steering_deg, heading_error):
    """Map steering demand and visible lane heading to a conservative speed."""
    severity_deg = max(
        abs(auto_steering_deg), abs(heading_error) * HEADING_GAIN_DEG
    )
    severity_points = np.array([0.0, 2.5, 5.0, 8.0, 13.0, 20.0, 24.0])
    speed_points = np.array([60.0, 56.0, 48.0, 38.0, 28.0, 18.0, 12.0])
    return float(np.interp(severity_deg, severity_points, speed_points))


def confidence_speed_limit_kmh(result, tracker):
    if result.left_detected and result.right_detected:
        confidence_points = np.array([0.0, 0.45, 0.60, 0.75, 1.0])
        speed_points = np.array([0.0, 15.0, 30.0, 48.0, 60.0])
        return float(np.interp(result.confidence, confidence_points, speed_points))
    if (
        result.left_detected != result.right_detected
        and tracker.reference_width_near is not None
        and tracker.single_boundary_frames <= MAX_SINGLE_BOUNDARY_RECOVERY_FRAMES
    ):
        return SINGLE_BOUNDARY_MAX_SPEED_KMH
    else:
        return 0.0


def tracking_error_speed_limit_kmh(lateral_error, heading_error):
    """Slow early from raw geometry, before rate-limited steering grows."""
    lateral_limit = float(np.interp(
        abs(lateral_error),
        np.array([0.0, 0.04, 0.08, 0.14, 0.24, 0.40]),
        np.array([60.0, 58.0, 50.0, 38.0, 24.0, 10.0]),
    ))
    heading_limit = float(np.interp(
        abs(heading_error),
        np.array([0.0, 0.025, 0.05, 0.09, 0.15, 0.25]),
        np.array([60.0, 56.0, 46.0, 34.0, 20.0, 10.0]),
    ))
    return min(lateral_limit, heading_limit)


def rate_limit_speed(current_kmh, target_kmh, dt):
    rate = MAX_ACCEL_KMH_PER_S if target_kmh > current_kmh else MAX_DECEL_KMH_PER_S
    max_change = rate * float(np.clip(dt, 0.0, 0.20))
    return float(np.clip(
        target_kmh, current_kmh - max_change, current_kmh + max_change
    ))


def command_vehicle(qcar, speed_kmh, steering_deg, brake=False):
    return qcar.set_velocity_and_request_state(
        forward=float(speed_kmh) / 3.6,
        turn=math.radians(float(steering_deg)),
        headlights=False,
        leftTurnSignal=False,
        rightTurnSignal=False,
        brakeSignal=bool(brake),
        reverseSignal=False,
    )


def unstuck_vehicle(qcar, location, rotation):
    """Stop, level, and lift the car while preserving position and yaw."""
    if location is None or rotation is None:
        return False
    command_vehicle(qcar, 0.0, 0.0, brake=True)
    reset_location = [
        float(location[0]),
        float(location[1]),
        float(location[2]) + UNSTUCK_LIFT_M,
    ]
    reset_rotation = [0.0, 0.0, float(rotation[2])]
    result = qcar.set_transform_and_request_state(
        location=reset_location,
        rotation=reset_rotation,
        enableDynamics=True,
        headlights=False,
        leftTurnSignal=False,
        rightTurnSignal=False,
        brakeSignal=True,
        reverseSignal=False,
        waitForConfirmation=True,
    )
    return bool(result[0])


def draw_autonomy_overlay(
    display,
    tracker,
    geometry,
    enabled,
    curve_limit_kmh,
    confidence_limit_kmh,
    tracking_limit_kmh,
    target_speed_kmh,
    command_kmh,
    drive_phase,
    collision_latched,
    auto_steering_deg,
    disturbance_deg,
    final_steering_deg,
    control_hint,
):
    height, width = display.shape[:2]
    if geometry is not None:
        _, heading, near_x, near_y, far_x, far_y, geometry_mode = geometry
        near_point = (int(near_x), int(near_y))
        far_point = (int(far_x), int(far_y))
        probe_color = (
            (0, 255, 255)
            if geometry_mode == "TWO BOUNDARIES"
            else (0, 165, 255)
        )
        cv2.line(display, near_point, far_point, probe_color, 4)
        cv2.circle(display, far_point, 8, probe_color, -1)
        cv2.putText(
            display,
            f"look-ahead heading={heading:+.3f} | {geometry_mode}",
            (15, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.50, probe_color, 1
        )

    state = "COLLISION PAUSED" if collision_latched else (
        drive_phase if enabled else "PAUSED"
    )
    state_color = (0, 0, 255) if collision_latched else (
        (0, 255, 0) if enabled else (0, 165, 255)
    )
    lock_text = (
        "LOCKED"
        if tracker.lane_locked
        else f"ACQUIRING {tracker.acquire_frames}/{LOCK_ACQUIRE_FRAMES}"
    )
    cv2.putText(
        display, f"AUTONOMY: {state} | initial lane: {lock_text}", (15, 245),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2
    )
    cv2.putText(
        display,
        f"speed limits: max={MAX_STRAIGHT_SPEED_KMH:4.0f} curve={curve_limit_kmh:4.0f} "
        f"confidence={confidence_limit_kmh:4.0f} tracking={tracking_limit_kmh:4.0f}",
        (15, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1
    )
    cv2.putText(
        display,
        f"speed km/h: target={target_speed_kmh:5.1f} command={command_kmh:5.1f}",
        (15, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1
    )
    cv2.putText(
        display,
        f"steering deg: auto={auto_steering_deg:+5.1f} "
        f"G920 noise={disturbance_deg:+4.1f} final={final_steering_deg:+5.1f}",
        (15, 314), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1
    )
    cv2.putText(
        display,
        control_hint,
        (15, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (255, 255, 255), 1
    )
    return display


def main():
    qlabs = None
    qcar = None
    g920 = None
    listener_thread = None

    try:
        from pal.utilities.steering import LogitechG920

        print("=" * 76)
        print("QCar2 AUTONOMOUS INITIAL-LANE TRACKER")
        print(
            f"Adaptive acceleration to {MAX_STRAIGHT_SPEED_KMH:.0f} km/h "
            "+ G920 +/-6 degree disturbances"
        )
        print("=" * 76)
        print("Open QLabs and load Open Road; this program will spawn the car.\n")

        qlabs, qcar = setup_qlabs_car()

        g920 = LogitechG920()
        tracker = InitialLaneTracker()
        steering_controller = LaneSteeringController()

        listener_thread = threading.Thread(target=escape_listener, daemon=True)
        listener_thread.start()

        autonomy_enabled = False
        previous_start_button = False
        previous_stop_button = False
        previous_unstuck_button = False
        buttons_reported = False
        command_speed_kmh = 0.0
        alignment_complete = False
        alignment_frames = 0
        collision_latched = False
        last_location = None
        last_rotation = None
        previous_time = time.monotonic()
        fps_filtered = 0.0

        command_vehicle(qcar, 0.0, 0.0, brake=True)
        print("Acquiring the initial lane. Press G920 A to start after LOCKED appears.")

        while not stop_event.is_set():
            loop_start = time.monotonic()
            dt = float(np.clip(loop_start - previous_time, 0.001, 0.20))
            previous_time = loop_start

            g920.read()
            start_button, start_name = button_value(g920, START_BUTTON_NAMES)
            stop_button, stop_name = button_value(g920, STOP_BUTTON_NAMES)
            unstuck_button, unstuck_name = button_value(g920, UNSTUCK_BUTTON_NAMES)
            brake = normalized_positive_axis(g920.brake, G920_PEDAL_DEADZONE)
            wheel = normalized_signed_axis(
                G920_STEERING_SIGN * g920.steering, G920_STEERING_DEADZONE
            )
            disturbance_deg = wheel * G920_MAX_DISTURBANCE_DEG

            if not buttons_reported:
                if start_name is None or stop_name is None or unstuck_name is None:
                    available = [
                        name for name in dir(g920) if "button" in name.lower()
                    ]
                    raise RuntimeError(
                        "G920 A/B/X buttons were not found. Available attributes: "
                        f"{available}"
                    )
                print(
                    f"G920 controls detected: start={start_name}, "
                    f"stop={stop_name}, unstuck={unstuck_name}"
                )
                buttons_reported = True

            start_pressed = start_button and not previous_start_button
            stop_pressed = stop_button and not previous_stop_button
            unstuck_pressed = (
                unstuck_button and not previous_unstuck_button
            )
            previous_start_button = start_button
            previous_stop_button = stop_button
            previous_unstuck_button = unstuck_button

            if unstuck_pressed:
                autonomy_enabled = False
                command_speed_kmh = 0.0
                alignment_complete = False
                alignment_frames = 0
                steering_controller.reset()
                if unstuck_vehicle(qcar, last_location, last_rotation):
                    collision_latched = False
                    tracker._clear_lock()
                    print(
                        "\nUNSTUCK complete. Waiting for a fresh lane lock; "
                        "press A afterward."
                    )
                    time.sleep(0.5)
                else:
                    print("\nUNSTUCK unavailable: vehicle pose is not ready.")

            image_ok, frame_bgr = qcar.get_image(CAMERA)
            if not image_ok or frame_bgr is None:
                autonomy_enabled = False
                command_speed_kmh = rate_limit_speed(command_speed_kmh, 0.0, dt)
                command_vehicle(qcar, command_speed_kmh, 0.0, brake=False)
                print("Camera image unavailable; autonomy paused.", end="\r")
                time.sleep(0.02)
                continue

            result = tracker.process(frame_bgr)
            height, width = frame_bgr.shape[:2]
            geometry = lane_geometry(result, tracker, width, height)

            if start_pressed:
                if autonomy_enabled:
                    autonomy_enabled = False
                    alignment_complete = False
                    alignment_frames = 0
                    print("\nAutonomy paused.")
                elif collision_latched:
                    print("\nCollision is latched. Press G920 X to unstuck first.")
                elif tracker.lane_locked and geometry is not None:
                    autonomy_enabled = True
                    command_speed_kmh = 0.0
                    alignment_complete = False
                    alignment_frames = 0
                    steering_controller.reset()
                    print("\nAutonomy active; low-speed alignment started.")
                else:
                    print("\nCannot start: waiting for initial-lane lock.")

            if stop_pressed or stop_button or brake > 0.0:
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
                    lateral_error, heading_error, dt
                )
            else:
                steering_controller.reset()

            if autonomy_enabled and geometry is not None:
                curve_limit_kmh = curve_speed_limit_kmh(
                    auto_steering_deg, heading_error
                )
                confidence_limit_kmh = confidence_speed_limit_kmh(result, tracker)
                tracking_limit_kmh = tracking_error_speed_limit_kmh(
                    lateral_error, heading_error
                )

                if not alignment_complete:
                    geometry_mode = geometry[-1]
                    well_aligned = (
                        geometry_mode == "TWO BOUNDARIES"
                        and abs(lateral_error) <= ALIGNMENT_LATERAL_TOLERANCE
                        and abs(heading_error) <= ALIGNMENT_HEADING_TOLERANCE
                    )
                    alignment_frames = alignment_frames + 1 if well_aligned else 0
                    if alignment_frames >= ALIGNMENT_STABLE_FRAMES:
                        alignment_complete = True
                        print("\nAlignment stable; normal adaptive acceleration enabled.")

                if alignment_complete:
                    drive_phase = "CRUISE"
                    target_speed_kmh = min(
                        MAX_STRAIGHT_SPEED_KMH,
                        curve_limit_kmh,
                        confidence_limit_kmh,
                        tracking_limit_kmh,
                    )
                else:
                    drive_phase = (
                        f"ALIGNING {alignment_frames}/{ALIGNMENT_STABLE_FRAMES}"
                    )
                    target_speed_kmh = min(
                        ALIGNMENT_CRAWL_SPEED_KMH,
                        curve_limit_kmh,
                        confidence_limit_kmh,
                        tracking_limit_kmh,
                    )
                command_speed_kmh = rate_limit_speed(
                    command_speed_kmh, target_speed_kmh, dt
                )
                final_steering_deg = float(np.clip(
                    auto_steering_deg + disturbance_deg,
                    -MAX_TOTAL_STEERING_DEG,
                    MAX_TOTAL_STEERING_DEG,
                ))
            else:
                curve_limit_kmh = 0.0
                confidence_limit_kmh = 0.0
                tracking_limit_kmh = 0.0
                target_speed_kmh = 0.0
                drive_phase = "PAUSED"
                command_speed_kmh = rate_limit_speed(
                    command_speed_kmh, 0.0, dt
                )
                final_steering_deg = 0.0

            status, location, rotation, front_hit, rear_hit = command_vehicle(
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
                        "Press G920 X to unstuck."
                    )
                collision_latched = True
                autonomy_enabled = False
                command_speed_kmh = 0.0
                target_speed_kmh = 0.0
                final_steering_deg = 0.0
                alignment_complete = False
                alignment_frames = 0
                steering_controller.reset()
                command_vehicle(qcar, 0.0, 0.0, brake=True)

            display = draw_overlay(frame_bgr, result)
            display = draw_autonomy_overlay(
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
                "G920: wheel noise | A start/pause | B/brake stop | X unstuck | ESC quit",
            )
            fps = 1.0 / dt
            fps_filtered = fps if fps_filtered == 0.0 else (
                0.9 * fps_filtered + 0.1 * fps
            )
            cv2.putText(
                display, f"control FPS: {fps_filtered:.1f}", (15, 336),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1
            )
            cv2.imshow("QCar2 - Autonomous Lane Camera Probe", display)

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
            cv2.imshow("QCar2 - Autonomous Lane Edge Probe", debug)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                stop_event.set()

            remaining_s = (1.0 / CONTROL_RATE_HZ) - (
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
        stop_event.set()
        if qcar is not None:
            try:
                command_vehicle(qcar, 0.0, 0.0, brake=True)
            except Exception:
                pass
        if g920 is not None:
            try:
                g920.terminate()
            except Exception:
                pass
        cv2.destroyAllWindows()
        if listener_thread is not None:
            listener_thread.join(timeout=0.5)
        if qlabs is not None:
            try:
                qlabs.close()
            except Exception:
                pass
        print("Autonomous controller closed; QCar2 command set to zero.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
