"""HSV yellow-guide-line detector with an interactive color picker/editor."""

import cv2
import numpy as np

import settings as cfg
from lane_detector import LaneTracker


COLOR_EDITOR_WINDOW = "Lane Color Picker - Yellow HSV"


class YellowLaneTracker(LaneTracker):
    """LaneTracker whose Canny input is restricted by a live HSV color mask."""

    def __init__(self):
        super().__init__()
        # Broad yellow starting range. Use P and click the actual guide line to
        # adapt it to the map's lighting and material.
        self.lower_hsv = np.array([15, 80, 80], dtype=np.uint8)
        self.upper_hsv = np.array([40, 255, 255], dtype=np.uint8)
        self.editor_visible = False
        self.last_hsv = None
        self.last_frame_shape = None
        self.last_mask = None

    @staticmethod
    def _noop(_value):
        pass

    def open_color_editor(self):
        if self.editor_visible:
            return
        # AUTOSIZE keeps mouse coordinates aligned with camera pixels, which is
        # important when clicking the left-hand preview to sample a color.
        cv2.namedWindow(COLOR_EDITOR_WINDOW, cv2.WINDOW_AUTOSIZE)
        controls = (
            ("H low", 0, 179, int(self.lower_hsv[0])),
            ("H high", 0, 179, int(self.upper_hsv[0])),
            ("S low", 0, 255, int(self.lower_hsv[1])),
            ("S high", 0, 255, int(self.upper_hsv[1])),
            ("V low", 0, 255, int(self.lower_hsv[2])),
            ("V high", 0, 255, int(self.upper_hsv[2])),
        )
        for name, _minimum, maximum, initial in controls:
            cv2.createTrackbar(
                name, COLOR_EDITOR_WINDOW, initial, maximum, self._noop
            )
        cv2.setMouseCallback(COLOR_EDITOR_WINDOW, self._pick_from_preview)
        self.editor_visible = True

    def close_color_editor(self):
        if not self.editor_visible:
            return
        try:
            cv2.destroyWindow(COLOR_EDITOR_WINDOW)
        except cv2.error:
            pass
        self.editor_visible = False

    def toggle_color_editor(self):
        if self.editor_visible:
            self.close_color_editor()
        else:
            self.open_color_editor()

    def _read_editor_values(self):
        if not self.editor_visible:
            return
        try:
            h_low = cv2.getTrackbarPos("H low", COLOR_EDITOR_WINDOW)
            h_high = cv2.getTrackbarPos("H high", COLOR_EDITOR_WINDOW)
            s_low = cv2.getTrackbarPos("S low", COLOR_EDITOR_WINDOW)
            s_high = cv2.getTrackbarPos("S high", COLOR_EDITOR_WINDOW)
            v_low = cv2.getTrackbarPos("V low", COLOR_EDITOR_WINDOW)
            v_high = cv2.getTrackbarPos("V high", COLOR_EDITOR_WINDOW)
        except cv2.error:
            self.editor_visible = False
            return

        self.lower_hsv = np.array(
            [min(h_low, h_high), min(s_low, s_high), min(v_low, v_high)],
            dtype=np.uint8,
        )
        self.upper_hsv = np.array(
            [max(h_low, h_high), max(s_low, s_high), max(v_low, v_high)],
            dtype=np.uint8,
        )

    def _pick_from_preview(self, event, x, y, _flags, _parameter):
        if event != cv2.EVENT_LBUTTONDOWN or self.last_hsv is None:
            return
        height, width = self.last_hsv.shape[:2]
        # The left half of the editor is the unscaled camera frame.
        if not (0 <= x < width and 0 <= y < height):
            return

        h_value, s_value, v_value = [
            int(value) for value in self.last_hsv[y, x]
        ]
        bounds = {
            "H low": max(0, h_value - 8),
            "H high": min(179, h_value + 8),
            "S low": max(0, s_value - 70),
            "S high": min(255, s_value + 70),
            "V low": max(0, v_value - 70),
            "V high": min(255, v_value + 70),
        }
        try:
            for name, value in bounds.items():
                cv2.setTrackbarPos(name, COLOR_EDITOR_WINDOW, value)
        except cv2.error:
            self.editor_visible = False

    def _update_editor_preview(self, frame_bgr, hsv, mask):
        if not self.editor_visible:
            return
        try:
            if cv2.getWindowProperty(
                COLOR_EDITOR_WINDOW, cv2.WND_PROP_VISIBLE
            ) < 1:
                self.editor_visible = False
                return
        except cv2.error:
            self.editor_visible = False
            return

        camera_preview = frame_bgr.copy()
        mask_preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_preview[:, :, 0] = 0
        mask_preview[:, :, 2] = 0

        midpoint_hsv = np.uint8([[[
            (int(self.lower_hsv[0]) + int(self.upper_hsv[0])) // 2,
            (int(self.lower_hsv[1]) + int(self.upper_hsv[1])) // 2,
            (int(self.lower_hsv[2]) + int(self.upper_hsv[2])) // 2,
        ]]])
        midpoint_bgr = cv2.cvtColor(midpoint_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        swatch_color = tuple(int(value) for value in midpoint_bgr)
        cv2.rectangle(camera_preview, (470, 8), (630, 52), swatch_color, -1)
        cv2.rectangle(camera_preview, (470, 8), (630, 52), (255, 255, 255), 1)
        cv2.putText(
            camera_preview,
            "Click yellow guide line to sample",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2,
        )
        cv2.putText(
            mask_preview,
            "Green = pixels accepted as guide-line color",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1,
        )
        cv2.putText(
            mask_preview,
            f"HSV {self.lower_hsv.tolist()} to {self.upper_hsv.tolist()}",
            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
        )
        cv2.imshow(
            COLOR_EDITOR_WINDOW,
            cv2.hconcat([camera_preview, mask_preview]),
        )
        self.last_hsv = hsv.copy()
        self.last_frame_shape = frame_bgr.shape
        self.last_mask = mask.copy()

    def preprocess(self, frame_bgr):
        self._read_editor_values()
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # Join small paint gaps and remove isolated colored texture.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        color_mask = cv2.morphologyEx(
            color_mask, cv2.MORPH_CLOSE, kernel, iterations=2
        )
        color_mask = cv2.morphologyEx(
            color_mask, cv2.MORPH_OPEN, kernel, iterations=1
        )
        blurred = cv2.GaussianBlur(color_mask, cfg.GAUSSIAN_KERNEL, 0)
        edges = cv2.Canny(blurred, cfg.CANNY_LOW, cfg.CANNY_HIGH)

        height, width = edges.shape[:2]
        polygon = self._roi_polygon(width, height)
        roi_mask = np.zeros_like(edges)
        cv2.fillPoly(roi_mask, polygon, 255)
        roi_edges = cv2.bitwise_and(edges, roi_mask)

        self.last_hsv = hsv.copy()
        self.last_frame_shape = frame_bgr.shape
        self.last_mask = color_mask.copy()
        self._update_editor_preview(frame_bgr, hsv, color_mask)
        return edges, roi_edges, polygon
