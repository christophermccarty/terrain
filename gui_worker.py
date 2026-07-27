"""Threaded simulation worker independent of Tkinter.

The worker owns mutable simulation state and publishes atomic completed frames.
Keeping it outside ``main.py`` makes its synchronization contract testable
without constructing a GUI.
"""
from __future__ import annotations

import logging
import time
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Callable

from planet_params import EARTH, PlanetParams
from simulate import TimeScaleMode, simulate_step
from time_policy import substeps_for_mode

LOG = logging.getLogger("planetsim")


class SimulationWorker(Thread):
    """Background physics loop with synchronized state ownership."""

    def __init__(
        self,
        initial_state,
        days_per_step=1.0,
        wind_block_size=8,
        diagnostics=None,
        time_scale_mode: TimeScaleMode = TimeScaleMode.DAILY,
        planet_params: PlanetParams = EARTH,
        *,
        step_function: Callable = simulate_step,
    ):
        super().__init__(daemon=True)
        self.state = initial_state
        self.days_per_step = days_per_step
        self.wind_block_size = wind_block_size
        self.diagnostics = diagnostics
        self.time_scale_mode = time_scale_mode
        self.planet_params = planet_params
        self._step_function = step_function
        self.running = Event()
        self.paused = Event()
        self.paused.set()
        self.state_lock = Lock()
        self.frame_queue = Queue(maxsize=1)
        self.error_queue = Queue(maxsize=1)

    def run(self):
        """Run physics cycles until stopped."""
        self.running.set()
        while self.running.is_set():
            if self.paused.is_set():
                time.sleep(0.05)
                continue

            try:
                with self.state_lock:
                    if self.paused.is_set():
                        continue
                    mode = self.time_scale_mode
                    substeps = substeps_for_mode(mode, self.planet_params)

                    new_state = self.state
                    temp_components: dict = {}
                    for step_days, do_wind in substeps:
                        new_state, temp_components = self._step_function(
                            new_state,
                            days=step_days,
                            wind_block_size=self.wind_block_size,
                            update_wind=do_wind,
                            debug_log=False,
                            track_components=self.diagnostics is not None,
                            time_scale=mode,
                            planet_params=self.planet_params,
                        )
                        if self.diagnostics is not None:
                            self.diagnostics.record_step(
                                new_state,
                                new_state.day_of_year,
                                days_elapsed=step_days,
                                component_contributions=temp_components,
                            )

                    self.state = new_state
                    self._publish_latest((new_state, temp_components))
            except Exception as exc:
                LOG.exception("Simulation thread error")
                try:
                    self.error_queue.put_nowait(str(exc))
                except Full:
                    pass
                self.paused.set()

    def _publish_latest(self, frame) -> None:
        """Publish one frame, replacing stale queued output under backpressure."""
        try:
            self.frame_queue.put_nowait(frame)
        except Full:
            try:
                self.frame_queue.get_nowait()
            except Empty:
                pass
            try:
                self.frame_queue.put_nowait(frame)
            except Full:
                pass

    def pause(self):
        self.paused.set()

    def pause_and_get_state(self):
        """Pause and wait for any in-flight cycle before returning state."""
        self.paused.set()
        with self.state_lock:
            state = self.state
        try:
            while True:
                self.frame_queue.get_nowait()
        except Empty:
            pass
        return state

    def snapshot_state(self):
        """Return synchronized state without changing the prior pause state."""
        was_paused = self.paused.is_set()
        state = self.pause_and_get_state()
        if not was_paused and self.running.is_set():
            self.resume()
        return state

    def resume(self):
        self.paused.clear()

    def stop(self):
        self.running.clear()

    def update_days_per_step(self, days):
        """Update legacy speed setting retained for compatibility."""
        self.days_per_step = days

    def update_time_scale(self, mode: TimeScaleMode):
        self.time_scale_mode = mode

    def update_wind_block_size(self, block_size):
        self.wind_block_size = block_size

    def update_planet_params(self, planet_params: PlanetParams):
        self.planet_params = planet_params
