"""Simple Canny + Hough lane tracker for QCar2.

Design basis:
- Grayscale / optional CLAHE
- Gaussian blur
- Canny edges
- Configurable trapezoid ROI
- Probabilistic Hough transform
- Segment filtering and clustering
- Select nearest left/right boundaries around the camera center
- Temporal smoothing and short lost-line hold

This module only estimates lane geometry. It never commands the vehicle.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

import settings as cfg


@dataclass
class FittedLine:
    """Lane boundary represented as x = a*y + b in image coordinates."""

    a: float
    b: float
    weight: float = 1.0

    def x_at(self, y: float) -> float:
        return self.a * y + self.b


@dataclass
class TrackingResult:
    left: Optional[FittedLine]
    right: Optional[FittedLine]
    lane_center_x: Optional[float]
    vehicle_center_x: float
    error_px: Optional[float]
    left_lost_frames: int
    right_lost_frames: int
    left_detected: bool
    right_detected: bool
    lane_width_px: Optional[float]
    confidence: float
    raw_segments: Optional[np.ndarray]
    edges: np.ndarray
    roi_edges: np.ndarray
    roi_polygon: np.ndarray

    @property
    def both_found(self) -> bool:
        return self.left is not None and self.right is not None


class LaneTracker:
    def __init__(self):
        self.prev_left: Optional[FittedLine] = None
        self.prev_right: Optional[FittedLine] = None
        self.left_lost_frames = 0
        self.right_lost_frames = 0

        self._clahe = cv2.createCLAHE(
            clipLimit=cfg.CLAHE_CLIP_LIMIT,
            tileGridSize=cfg.CLAHE_GRID_SIZE,
        )

    @staticmethod
    def _roi_polygon(width: int, height: int) -> np.ndarray:
        points = [
            (int(xf * width), int(yf * height))
            for xf, yf in cfg.ROI_POINTS
        ]
        return np.array([points], dtype=np.int32)

    def _vehicle_center_x(self, width: int) -> float:
        """Image X used as the vehicle reference; subclasses may calibrate it."""
        return width / 2.0

    def _max_lost_frames(self) -> int:
        return cfg.MAX_LOST_FRAMES

    def preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if cfg.USE_CLAHE:
            gray = self._clahe.apply(gray)

        blurred = cv2.GaussianBlur(gray, cfg.GAUSSIAN_KERNEL, 0)
        edges = cv2.Canny(blurred, cfg.CANNY_LOW, cfg.CANNY_HIGH)

        h, w = edges.shape[:2]
        polygon = self._roi_polygon(w, h)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygon, 255)
        roi_edges = cv2.bitwise_and(edges, mask)
        return edges, roi_edges, polygon

    @staticmethod
    def _fit_segment(segment: Sequence[int]) -> Optional[FittedLine]:
        x1, y1, x2, y2 = [float(v) for v in segment]
        dy = y2 - y1
        dx = x2 - x1

        if abs(dy) < 2.0:
            return None

        a = dx / dy  # x = a*y + b
        abs_a = abs(a)
        if not (cfg.MIN_ABS_DX_OVER_DY <= abs_a <= cfg.MAX_ABS_DX_OVER_DY):
            return None

        b = x1 - a * y1
        length = float(np.hypot(dx, dy))
        return FittedLine(a=a, b=b, weight=length)

    @staticmethod
    def _cluster_candidates(candidates: List[FittedLine], y_bottom: float) -> List[FittedLine]:
        """Cluster lines by their predicted X coordinate at the bottom of the image."""
        if not candidates:
            return []

        candidates = sorted(candidates, key=lambda line: line.x_at(y_bottom))
        groups: List[List[FittedLine]] = []

        for candidate in candidates:
            x = candidate.x_at(y_bottom)
            if not groups:
                groups.append([candidate])
                continue

            last_group = groups[-1]
            last_x = np.average(
                [ln.x_at(y_bottom) for ln in last_group],
                weights=[ln.weight for ln in last_group],
            )
            if abs(x - last_x) <= cfg.CLUSTER_BOTTOM_X_PX:
                last_group.append(candidate)
            else:
                groups.append([candidate])

        clustered: List[FittedLine] = []
        for group in groups:
            weights = np.array([ln.weight for ln in group], dtype=float)
            total_weight = float(weights.sum())
            if total_weight < cfg.MIN_CLUSTER_WEIGHT:
                continue

            a = float(np.average([ln.a for ln in group], weights=weights))
            b = float(np.average([ln.b for ln in group], weights=weights))
            clustered.append(FittedLine(a=a, b=b, weight=total_weight))

        return clustered

    @staticmethod
    def _smooth(previous: Optional[FittedLine], current: FittedLine) -> FittedLine:
        if previous is None:
            return current
        s = float(cfg.SMOOTHING)
        return FittedLine(
            a=s * previous.a + (1.0 - s) * current.a,
            b=s * previous.b + (1.0 - s) * current.b,
            weight=current.weight,
        )

    def _update_with_fallback(
        self,
        detected: Optional[FittedLine],
        previous: Optional[FittedLine],
        lost_frames: int,
    ) -> Tuple[Optional[FittedLine], Optional[FittedLine], int]:
        if detected is not None:
            smoothed = self._smooth(previous, detected)
            return smoothed, smoothed, 0

        lost_frames += 1
        if previous is not None and lost_frames <= self._max_lost_frames():
            # Briefly hold the previous visual estimate for display. Confidence
            # remains zero unless both boundaries were freshly detected, so the
            # controller cannot steer from these held lines.
            return previous, previous, lost_frames

        return None, None, lost_frames

    def process(self, frame_bgr: np.ndarray) -> TrackingResult:
        edges, roi_edges, polygon = self.preprocess(frame_bgr)
        h, w = frame_bgr.shape[:2]
        vehicle_center_x = self._vehicle_center_x(w)
        y_bottom = float(h - 1)

        raw = cv2.HoughLinesP(
            roi_edges,
            cfg.HOUGH_RHO,
            np.deg2rad(cfg.HOUGH_THETA_DEG),
            cfg.HOUGH_THRESHOLD,
            np.array([]),
            minLineLength=cfg.HOUGH_MIN_LINE_LENGTH,
            maxLineGap=cfg.HOUGH_MAX_LINE_GAP,
        )

        candidates: List[FittedLine] = []
        if raw is not None:
            for wrapped in raw:
                fitted = self._fit_segment(wrapped.reshape(4))
                if fitted is not None:
                    # Discard absurd predictions far outside the frame.
                    xb = fitted.x_at(y_bottom)
                    if -0.4 * w <= xb <= 1.4 * w:
                        candidates.append(fitted)

        clusters = self._cluster_candidates(candidates, y_bottom)

        left_clusters = [
            ln for ln in clusters
            if ln.x_at(y_bottom) < vehicle_center_x - cfg.CENTER_EXCLUSION_PX
        ]
        right_clusters = [
            ln for ln in clusters
            if ln.x_at(y_bottom) > vehicle_center_x + cfg.CENTER_EXCLUSION_PX
        ]

        detected_left = max(
            left_clusters,
            key=lambda ln: ln.x_at(y_bottom),
            default=None,
        )
        detected_right = min(
            right_clusters,
            key=lambda ln: ln.x_at(y_bottom),
            default=None,
        )

        left, self.prev_left, self.left_lost_frames = self._update_with_fallback(
            detected_left, self.prev_left, self.left_lost_frames
        )
        right, self.prev_right, self.right_lost_frames = self._update_with_fallback(
            detected_right, self.prev_right, self.right_lost_frames
        )

        error_y = float(np.clip(cfg.ERROR_Y_FRACTION, 0.0, 1.0) * (h - 1))
        lane_center_x = None
        error_px = None
        if left is not None and right is not None:
            lx = left.x_at(error_y)
            rx = right.x_at(error_y)
            if lx < rx:
                lane_center_x = (lx + rx) / 2.0
                error_px = lane_center_x - vehicle_center_x

        left_detected = detected_left is not None
        right_detected = detected_right is not None
        lane_width_px = None
        confidence = 0.0
        if left_detected and right_detected and detected_left is not None and detected_right is not None:
            lane_width_px = detected_right.x_at(error_y) - detected_left.x_at(error_y)
            min_width = cfg.MIN_LANE_WIDTH_FRACTION * w
            max_width = cfg.MAX_LANE_WIDTH_FRACTION * w
            if min_width <= lane_width_px <= max_width:
                weight_score = min(
                    detected_left.weight,
                    detected_right.weight,
                ) / cfg.CONFIDENCE_FULL_WEIGHT
                confidence = float(np.clip(weight_score, 0.0, 1.0))

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


def _line_endpoints(line: FittedLine, height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    y1 = height - 1
    y2 = int(height * 0.56)
    x1 = int(line.x_at(y1))
    x2 = int(line.x_at(y2))
    return (x1, y1), (x2, y2)


def draw_overlay(frame_bgr: np.ndarray, result: TrackingResult) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    if cfg.SHOW_ROI_POLYGON:
        cv2.polylines(out, result.roi_polygon, True, (255, 180, 0), 2)

    if cfg.SHOW_RAW_HOUGH_SEGMENTS and result.raw_segments is not None:
        for wrapped in result.raw_segments:
            x1, y1, x2, y2 = wrapped.reshape(4)
            cv2.line(out, (x1, y1), (x2, y2), (80, 80, 80), 1)

    # Vehicle center
    vehicle_x = int(result.vehicle_center_x)
    cv2.line(out, (vehicle_x, h), (vehicle_x, int(h * 0.65)), (255, 0, 255), 2)

    # Tracked lane boundaries
    if result.left is not None:
        p1, p2 = _line_endpoints(result.left, h)
        cv2.line(out, p1, p2, (0, 255, 0), cfg.LINE_THICKNESS)
        cv2.putText(out, "LEFT", p2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if result.right is not None:
        p1, p2 = _line_endpoints(result.right, h)
        cv2.line(out, p1, p2, (0, 255, 0), cfg.LINE_THICKNESS)
        cv2.putText(out, "RIGHT", p2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Center tracking marker at the configured error evaluation row.
    error_y = int(np.clip(cfg.ERROR_Y_FRACTION, 0.0, 1.0) * (h - 1))
    if result.lane_center_x is not None:
        lane_x = int(result.lane_center_x)
        cv2.circle(out, (lane_x, error_y), 9, (255, 255, 0), -1)
        cv2.line(out, (vehicle_x, error_y), (lane_x, error_y), (255, 255, 0), 3)

    status = (
        "TRACKING"
        if result.left_detected and result.right_detected and result.error_px is not None
        else "DEGRADED"
    )
    status_color = (0, 255, 0) if status == "TRACKING" else (0, 165, 255)

    cv2.putText(out, f"Lane tracker: {status}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

    if result.error_px is not None:
        side = "RIGHT" if result.error_px > 0 else "LEFT"
        cv2.putText(out, f"Lane-center error: {result.error_px:+.1f} px ({side})", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    else:
        cv2.putText(out, "Lane-center error: unavailable", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

    cv2.putText(out,
                f"lost L/R: {result.left_lost_frames}/{result.right_lost_frames}",
                (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    cv2.putText(out, f"confidence: {result.confidence:.2f}", (15, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

    return out
