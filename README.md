# QCar2 Virtual Lane Tracker v1 — Perception Baseline

This is a deliberately small **perception-first** prototype for Quanser QLabs/QCar2.

It is based on the simple structure of the `imdiora/Lane-keeping-assistance-` project:

**grayscale/contrast → Gaussian blur → Canny → ROI → HoughLinesP → line filtering/smoothing**.

This version extends that idea for a multi-lane QLabs road by clustering Hough segments and selecting the **nearest lane boundary to the left and right of the camera center**.

## Baseline status

This commit is the stable **perception-only baseline** for the QCar2 LKAS project. It detects and tracks the current lane while the driver controls steering and throttle manually. Automatic steering control, controller tuning, and LKAS disengagement logic are intentionally reserved for later versions.

## Important: this is NOT autonomous control

The lane tracker never changes the steering or throttle.

You drive manually with the keyboard while the program only displays:

- detected left boundary (green)
- detected right boundary (green)
- vehicle/camera center (magenta)
- estimated lane center (cyan marker)
- lateral error in pixels
- Canny/ROI debug view
- lost-line counters

This lets us prove the lane tracking works before connecting it to an LKAS controller.

---

## Files

```text
qcar2_virtual_lane_tracker_v1/
├── .gitignore                    <-- excludes Python cache files
├── run_virtual_lane_tracker.py   <-- run this
├── lane_detector.py              <-- perception algorithm
├── settings.py                   <-- tune values here
├── requirements.txt
└── README.md
```

For normal testing, you only need to run `run_virtual_lane_tracker.py` and edit `settings.py` if the ROI/detection needs tuning.

---

# Quick start

## 1. Install the small extra dependencies

Your Quanser Python environment should already contain `pal` and `qvl`.

From this folder:

```bash
python -m pip install -r requirements.txt
```

## 2. Open QLabs

Open **Quanser Interactive Labs** and load the **Open Road** workspace.

Do not run another QCar setup script. This program spawns the QCar2 and starts its real-time model itself.

## 3. Run the tracker

```bash
python run_virtual_lane_tracker.py
```

On the first virtual PAL use you may see:

```text
Would you like to use virtual QCar1 or QCar2? (enter 1 or 2)
```

Enter:

```text
2
```

## 4. Drive manually

The keyboard listener works independently from which OpenCV window has focus.

```text
W       forward
S       reverse
A       steer left
D       steer right
SPACE   stop
Q/ESC   quit
```

The tracker does **not** modify these commands.

---

# What you should see

Two windows open.

### `QCar2 - Virtual Lane Tracker`

- gray thin lines: raw Hough segments
- green thick lines: selected left/right lane boundaries
- magenta line: camera/vehicle center
- cyan marker: estimated center of the current lane
- cyan horizontal line: lane-center error

The desired result while the car is centered is approximately:

```text
      LEFT                       RIGHT
        \                         /
         \                       /
          \          |          /
           \         |         /
            \        |        /
                    vehicle

Lane-center error ~= 0 px
```

### `QCar2 - Lane Tracker Debug`

Left half = complete Canny edge image.

Right half = only the trapezoidal ROI sent to the Hough transform.

This is the first window to inspect if the program detects the wrong markings.

---

# First things to tune

Everything is near the top of `settings.py`.

## ROI

Start here if the tracker sees the barrier, horizon, or other lanes too aggressively:

```python
ROI_POINTS = [
    (0.05, 1.00),
    (0.38, 0.56),
    (0.62, 0.56),
    (0.95, 1.00),
]
```

Coordinates are fractions of the image size, so they remain understandable:

```text
(0,0) -------------------- (1,0)
  |                           |
  |        camera image       |
  |                           |
(0,1) -------------------- (1,1)
```

## Canny

```python
CANNY_LOW = 50
CANNY_HIGH = 150
```

Raise them if too many weak edges appear. Lower them if lane markings disappear.

## Hough

```python
HOUGH_THRESHOLD = 35
HOUGH_MIN_LINE_LENGTH = 25
HOUGH_MAX_LINE_GAP = 80
```

`HOUGH_MAX_LINE_GAP` is intentionally fairly large so dashed lines can still form a stable lane-boundary estimate.

## Temporal smoothing

```python
SMOOTHING = 0.65
MAX_LOST_FRAMES = 5
```

The GitHub baseline retains a previous right-line estimate when current detection disappears. This prototype applies the same basic idea to both boundaries, but discards the estimate after more than `MAX_LOST_FRAMES` missed frames.

---

# Why this version does not yet use Bird's Eye View / PID

We are deliberately validating one layer at a time:

```text
Phase 1 (this package)
Camera → Canny → ROI → Hough → current-lane boundaries

Phase 2
Improve color filtering / curves / perspective

Phase 3
Calculate calibrated lateral + heading error

Phase 4
Connect steering controller

Phase 5
Solid/dashed classification and lane-changing state machine
```

A controller will not be added until Phase 1 reliably follows the correct two markings while you manually drive through straight sections and bends.

---

# Troubleshooting

## `The remote peer refused the connection`

This program starts the QCar2 real-time model before importing PAL. If you still receive the message:

1. close the program;
2. keep QLabs open;
3. reload Open Road;
4. run `python run_virtual_lane_tracker.py` again;
5. enter `2` if PAL asks QCar1/QCar2.

Send the complete terminal output if it still fails.

## No green lane lines

Look at `QCar2 - Lane Tracker Debug`.

If the white lane markings are not visible as strong edges in the right half, tune the ROI/Canny settings first.

## Tracker selects the next lane instead of my lane

Reduce `CLUSTER_BOTTOM_X_PX` or narrow the ROI. The algorithm selects the cluster whose predicted position at the bottom of the image is nearest to the vehicle center on each side.

## Green lines jump around

Increase:

```python
SMOOTHING = 0.75
```

or increase Hough `MIN_LINE_LENGTH`.

---

# Technical basis

The reference GitHub project uses Canny edge detection, an ROI, `cv2.HoughLinesP`, slope-based lane selection, and—on its right-side implementation—a previous-line estimate with temporal smoothing. This prototype keeps that simple debugging philosophy but combines left/right tracking in one program and adds clustering so a multi-lane QLabs highway does not average every visible marking into one line.

Quanser PAL's `QCarRealSense` uses the virtual RGB camera server at port 18965 and fixes virtual RGB frames to 640×480. The program uses that front RGB stream, then converts the PAL RGB image buffer to BGR for OpenCV processing/display.

---

## Current scope

Success for v1 means:

> While you manually drive the virtual QCar2, the two green lines remain attached to the actual left/right markings of the lane occupied by the QCar and the displayed lane-center error changes sensibly as you move left or right.

Nothing more is required yet.

The next development stage is to use the measured lane-center error as input to a conservative steering controller with confidence checks and safe disengagement.
