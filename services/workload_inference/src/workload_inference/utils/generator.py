"""Simple fake gaze data generator for UI testing.

Runs a background thread that emits smooth gaze points at a fixed frequency
and calls a user-provided callback with a list containing one `GazeData`.
The generated screen coordinates are normalized in [0, 1].
"""
from __future__ import annotations

import math
import threading
import time
import logging
from typing import Callable, Optional

import numpy as np

from workload_inference.data.data_structures import GazeData, DroneData

logger = logging.getLogger("FakeGazeGenerator")


class FakeGazeGenerator:
    """Generate smooth fake gaze points and call `callback` at a fixed rate.

    Args:
        callback: callable receiving `list[GazeData]` on each update.
        frequency: updates per second (default 30.0).
        noise: standard deviation of random noise applied each step (normalized coords).
        speed: typical velocity magnitude (units per second, normalized domain).
        pupil_mean: average pupil diameter (float).
    """

    def __init__(
        self,
        callback: Callable[[list[GazeData]], None],
        frequency: float = 30.0,
        noise: float = 0.01,
        speed: float = 0.4,
        pupil_mean: float = 3.5,
    ) -> None:
        self.callback = callback
        self.frequency = float(frequency)
        self._period = 1.0 / max(0.1, self.frequency)
        self.noise = float(noise)
        self.speed = float(speed)
        self.pupil_mean = float(pupil_mean)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # State: positions and velocities for left/right eyes (normalized 0..1)
        # initialize near center
        self._pos = np.array([0.5, 0.5], dtype=float)
        self._vel = np.zeros(2, dtype=float)

    def start(self) -> None:
        """Start the generator thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        logger.info("FakeGazeGenerator started at %.1f Hz", self.frequency)

    def stop(self, join_timeout: float = 1.0) -> None:
        """Stop the generator thread and optionally wait for it to finish."""
        self._stop_event.set()
        th = None
        with self._lock:
            th = self._thread
            self._thread = None
        if th is not None and th is not threading.current_thread():
            th.join(timeout=join_timeout)
            if th.is_alive():
                logger.warning("FakeGazeGenerator thread did not stop within timeout")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        next_time = time.perf_counter()
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            # advance motion using a simple damped random-walk (Ornstein-Uhlenbeck like)
            dt = max(1e-6, self._period)
            # velocity relax toward small values and is driven by noise
            self._vel = 0.9 * self._vel + (np.random.randn(2) * self.noise * self.speed)
            # integrate position
            self._pos = self._pos + self._vel * dt
            # bounce / clamp inside [0,1]
            for i in (0, 1):
                if self._pos[i] < 0.0:
                    self._pos[i] = -self._pos[i]
                    self._vel[i] = -self._vel[i]
                elif self._pos[i] > 1.0:
                    self._pos[i] = 2.0 - self._pos[i]
                    self._vel[i] = -self._vel[i]

            # make a small offset between left and right eye to look realistic
            left_x, left_y = float(self._pos[0] + 0.002), float(self._pos[1] + 0.001)
            right_x, right_y = float(self._pos[0] - 0.002), float(self._pos[1] - 0.001)
            # clamp
            left_x = min(max(left_x, 0.0), 1.0)
            left_y = min(max(left_y, 0.0), 1.0)
            right_x = min(max(right_x, 0.0), 1.0)
            right_y = min(max(right_y, 0.0), 1.0)

            ts = np.int64(time.time() * 1_000)

            gaze = GazeData(
                timestamp=ts,
                left_gaze_point_x=np.float32(0.0),
                left_gaze_point_y=np.float32(0.0),
                left_gaze_point_z=np.float32(0.0),
                right_gaze_point_x=np.float32(0.0),
                right_gaze_point_y=np.float32(0.0),
                right_gaze_point_z=np.float32(0.0),
                left_point_screen_x=np.float32(left_x),
                left_point_screen_y=np.float32(left_y),
                right_point_screen_x=np.float32(right_x),
                right_point_screen_y=np.float32(right_y),
                left_validity=np.int8(0),
                right_validity=np.int8(0),
                left_pupil_diameter=np.float32(self.pupil_mean + np.random.randn() * 0.1),
                right_pupil_diameter=np.float32(self.pupil_mean + np.random.randn() * 0.1),
            )

            # callback with a single-element list for compatibility
            try:
                self.callback([gaze])
            except Exception:
                logger.exception("Error in gaze callback")

            # sleep until next tick while accounting for work time
            next_time += self._period
            sleep_for = next_time - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # if we're late, yield briefly to avoid tight loop
                time.sleep(0.001)


class FakeDroneGenerator:
    """Generate fake drone data for testing.
    
    Generates data from a drone looping in a circle while oscillating up and down
    """

    def __init__(
        self,
        callback: Callable[[list[DroneData]], None],
        frequency: float = 30.0,
        radius: float = 1.0,
        height: float = 1.0,
    ) -> None:
        self.callback = callback
        self.frequency = float(frequency)
        self._period = 1.0 / max(0.1, self.frequency)
        self.radius = float(radius)
        self.height = float(height)
        self._cb_cnt = 0

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the generator thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        logger.info("FakeDroneGenerator started at %.1f Hz", self.frequency)

    def stop(self, join_timeout: float = 1.0) -> None:
        """Stop the generator thread and optionally wait for it to finish."""
        self._stop_event.set()
        print(f"Stopping FakeDroneGenerator after {self._cb_cnt} callbacks")
        th = None
        with self._lock:
            th = self._thread
            self._thread = None
        if th is not None and th is not threading.current_thread():
            th.join(timeout=join_timeout)
            if th.is_alive():
                logger.warning("FakeDroneGenerator thread did not stop within timeout")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        idx = 0
        start_time = time.perf_counter()
        next_time = start_time
        while not self._stop_event.is_set():
            # use monotonic elapsed time for t to avoid idx-related drift
            t = time.perf_counter() - start_time
            x = np.cos(t) * self.radius
            y = np.sin(t) * self.radius
            z = np.sin(t * 0.5) * (self.height / 2) + (self.height / 2)

            drone_data = DroneData(
                timestamp=np.int64(time.time() * 1_000),
                id = np.int8(0),
                position_x=np.float32(x),
                position_y=np.float32(y),
                position_z=np.float32(z),
                velocity_x=np.float32(-np.sin(t) * self.radius),
                velocity_y=np.float32(np.cos(t) * self.radius),
                velocity_z=np.float32(np.cos(t * 0.5) * (self.height / 2) * 0.5),
                orientation_x=np.float32(0.0),
                orientation_y=np.float32(0.0),
                orientation_z=np.float32(math.radians(t) % (2 * math.pi)),
                angular_velocity_x=np.float32(0.0),
                angular_velocity_y=np.float32(0.0),
                angular_velocity_z=np.float32(math.radians(10.0)),
                acceleration_x=np.float32(-np.cos(t) * self.radius),
                acceleration_y=np.float32(-np.sin(t) * self.radius),
                acceleration_z=np.float32(0.0),
            )

            try:
                self.callback([drone_data])
                self._cb_cnt += 1
            except Exception:
                logger.exception("Error in drone callback")

            idx += 1

            # throttle to target frequency, accounting for processing time
            next_time += self._period
            sleep_for = next_time - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # if we're running behind, don't sleep long — yield briefly
                time.sleep(0.001)