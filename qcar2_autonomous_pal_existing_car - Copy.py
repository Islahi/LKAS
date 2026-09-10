"""Attach the PAL autonomous lane controller to an existing virtual QCar2.

This launcher does not connect to QLabs for scene management. It does not:
    * destroy actors;
    * spawn or reposition QCar2;
    * start or terminate real-time models;
    * assume a particular QLabs map.

Before running it, use the target map's own setup procedure to:
    1. spawn QCar2 actor 0;
    2. place it in the desired lane;
    3. start the QCAR2 real-time model.

The launcher then attaches through PAL/QCarRealSense and uses the same
low-speed automatic throttle, nearest-boundary detection, pixel-error PID, and
A/D additive steering-noise behavior as qcar2_autonomous_pal_manual_lkas.py.

Controls:
    M          toggle AUTONOMOUS / MANUAL drive mode
    W / S      manual-mode forward/reverse throttle
    L          request/pause autonomous driving
    A / D      manual steering, or autonomous steering disturbance
    P          open/close yellow lane color picker/editor
    SPACE      immediate stop; cancels autonomy
    Q / ESC    stop and exit
"""

import sys

from qcar2_autonomous_pal_manual_lkas import main
from yellow_lane_detector import YellowLaneTracker


if __name__ == "__main__":
    sys.exit(main(setup_vehicle=False, tracker_factory=YellowLaneTracker))
