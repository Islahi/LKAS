"""Conservative PID steering controller for the QCar2 lane tracker."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

import settings as cfg


@dataclass
class SteeringCommand:
    steering: float
    normalized_error: Optional[float]
    p_term: float = 0.0
    i_term: float = 0.0
    d_term: float = 0.0


class SteeringPID:
    """PID with integral limiting, derivative filtering, and output slew limiting."""

    def __init__(self):
        self.integral = 0.0
        self.previous_error: Optional[float] = None
        self.filtered_derivative = 0.0
        self.previous_output = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None
        self.filtered_derivative = 0.0
        self.previous_output = 0.0

    def update(self, error_px: float, image_width: int, dt: float) -> SteeringCommand:
        dt = float(np.clip(dt, cfg.PID_MIN_DT_S, cfg.PID_MAX_DT_S))
        half_width = max(image_width / 2.0, 1.0)
        error = float(np.clip(error_px / half_width, -1.0, 1.0))

        self.integral = float(np.clip(
            self.integral + error * dt,
            -cfg.PID_INTEGRAL_LIMIT,
            cfg.PID_INTEGRAL_LIMIT,
        ))

        raw_derivative = 0.0
        if self.previous_error is not None:
            raw_derivative = (error - self.previous_error) / dt
        alpha = float(np.clip(cfg.PID_DERIVATIVE_FILTER, 0.0, 1.0))
        self.filtered_derivative = (
            alpha * self.filtered_derivative + (1.0 - alpha) * raw_derivative
        )

        p_term = cfg.PID_KP * error
        i_term = cfg.PID_KI * self.integral
        d_term = cfg.PID_KD * self.filtered_derivative

        # Positive QCar steering is left. A positive image error means the lane
        # center is to the right, so the steering sign must be inverted.
        target = float(np.clip(
            -(p_term + i_term + d_term),
            -cfg.LKAS_MAX_ABS_STEERING,
            cfg.LKAS_MAX_ABS_STEERING,
        ))

        max_step = cfg.LKAS_MAX_STEERING_RATE * dt
        output = float(np.clip(
            target,
            self.previous_output - max_step,
            self.previous_output + max_step,
        ))

        self.previous_error = error
        self.previous_output = output
        return SteeringCommand(output, error, p_term, i_term, d_term)
