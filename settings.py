"""Easy-to-edit settings for QCar2 lane tracking and LKAS control."""

# -----------------------------------------------------------------------------
# QLABS / QCAR2 SETUP
# -----------------------------------------------------------------------------
QCAR_ACTOR_NUMBER = 0
QCAR_START_POSITION = [0.788, 5, 1.5]
QCAR_START_ORIENTATION = [0.0, 0.0, 0.0]
QCAR_SCALE = [1.0, 1.0, 1.0]
RT_START_WAIT_S = 2.0

# -----------------------------------------------------------------------------
# CAMERA
# QCarRealSense virtual RGB is 640 x 480 in current PAL.
# -----------------------------------------------------------------------------
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_FPS = 30

# -----------------------------------------------------------------------------
# IMAGE PROCESSING (based on the simple Canny + Hough approach)
# -----------------------------------------------------------------------------
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
GAUSSIAN_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150

# ROI trapezoid as fractions of image width/height.
# Order: bottom-left, top-left, top-right, bottom-right.
# These are starting values for Open Road; tune while looking at the ROI window.
ROI_POINTS = [
    (0.02, 1.00),
    (0.30, 0.52),
    (0.70, 0.52),
    (0.98, 1.00),
]

# Hough transform
HOUGH_RHO = 2
HOUGH_THETA_DEG = 1
HOUGH_THRESHOLD = 35
HOUGH_MIN_LINE_LENGTH = 25
HOUGH_MAX_LINE_GAP = 80

# Reject nearly horizontal / nearly vertical-noisy segments using |dx/dy|.
# In image coordinates the lane boundaries are better represented as x(y).
MIN_ABS_DX_OVER_DY = 0.05
MAX_ABS_DX_OVER_DY = 2.5

# Candidate clustering: Hough segments that predict similar X at the bottom
# are treated as the same physical lane marking.
CLUSTER_BOTTOM_X_PX = 65
MIN_CLUSTER_WEIGHT = 35.0

# A detected boundary must be at least this far from the camera center.
CENTER_EXCLUSION_PX = 20

# Temporal smoothing. Higher = trust the previous line more.
SMOOTHING = 0.50
MAX_LOST_FRAMES = 5

# Where to evaluate lane-center error for visualization.
# 1.0 = very bottom of image; 0.75 = 75% down the image.
ERROR_Y_FRACTION = 0.88

# Confidence requires two freshly detected boundaries with plausible separation.
MIN_LANE_WIDTH_FRACTION = 0.28
MAX_LANE_WIDTH_FRACTION = 0.78
CONFIDENCE_FULL_WEIGHT = 140.0

# -----------------------------------------------------------------------------
# MANUAL KEYBOARD DRIVING
# -----------------------------------------------------------------------------
MANUAL_THROTTLE = 0.20
MANUAL_REVERSE_THROTTLE = -0.08
MANUAL_STEERING = 0.32

# QCar write safety clamps.
MAX_ABS_THROTTLE = 0.50
MAX_ABS_STEERING = 0.50

# -----------------------------------------------------------------------------
# LKAS PID STEERING
# L toggles LKAS. Throttle remains manual. A/D immediately disengage LKAS.
# -----------------------------------------------------------------------------
PID_KP = 0.50
PID_KI = 0.015
PID_KD = 0.040
PID_INTEGRAL_LIMIT = 0.35
PID_DERIVATIVE_FILTER = 0.80
PID_MIN_DT_S = 0.005
PID_MAX_DT_S = 0.10

# Conservative steering magnitude and slew-rate limits.
LKAS_MAX_ABS_STEERING = 0.22
LKAS_MAX_STEERING_RATE = 0.70

# Engagement requires stable confidence; disengagement is immediate.
LKAS_MIN_CONFIDENCE = 0.60
LKAS_ENGAGE_FRAMES = 8

# -----------------------------------------------------------------------------
# DISPLAY
# -----------------------------------------------------------------------------
SHOW_RAW_HOUGH_SEGMENTS = True
SHOW_EDGE_WINDOW = False
SHOW_ROI_POLYGON = True
LINE_THICKNESS = 6
