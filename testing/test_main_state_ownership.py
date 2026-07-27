"""Headless regression tests for GUI simulation-state ownership."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

import main
from terrain import clear_elevation_cache, get_elevation_cache


def test_pause_and_get_state_waits_for_inflight_cycle(monkeypatch):
    entered_step = threading.Event()
    release_step = threading.Event()

    def blocking_step(state, **_kwargs):
        entered_step.set()
        assert release_step.wait(timeout=5.0)
        return state + 1, {}

    monkeypatch.setattr(main, "simulate_step", blocking_step)
    worker = main.SimulationThread(initial_state=0)
    worker.start()
    worker.resume()
    assert entered_step.wait(timeout=2.0)

    result = []
    snapshot_thread = threading.Thread(
        target=lambda: result.append(worker.pause_and_get_state())
    )
    snapshot_thread.start()
    time.sleep(0.05)
    assert snapshot_thread.is_alive(), "snapshot did not wait for active physics"

    release_step.set()
    snapshot_thread.join(timeout=2.0)
    try:
        assert result == [1]
        assert worker.paused.is_set()
        time.sleep(0.05)
        assert worker.state == 1
    finally:
        worker.stop()
        worker.join(timeout=2.0)


def test_snapshot_state_restores_running_state(monkeypatch):
    stepped = threading.Event()

    def one_step(state, **_kwargs):
        stepped.set()
        return state + 1, {}

    monkeypatch.setattr(main, "simulate_step", one_step)
    worker = main.SimulationThread(initial_state=0)
    worker.start()
    worker.resume()
    assert stepped.wait(timeout=2.0)

    try:
        snapshot = worker.snapshot_state()
        assert snapshot >= 1
        assert not worker.paused.is_set()
    finally:
        worker.pause()
        worker.stop()
        worker.join(timeout=2.0)


def test_worker_queues_atomic_latest_frame(monkeypatch):
    stepped = threading.Event()

    def one_step(state, **_kwargs):
        next_state = state + 1
        if next_state >= 3:
            stepped.set()
        return next_state, {"cycle": next_state}

    monkeypatch.setattr(main, "simulate_step", one_step)
    worker = main.SimulationThread(initial_state=0)
    worker.start()
    worker.resume()
    assert stepped.wait(timeout=2.0)
    worker.pause()
    with worker.state_lock:
        pass

    try:
        queued_state, components = worker.frame_queue.get_nowait()
        assert components["cycle"] == queued_state
    finally:
        worker.stop()
        worker.join(timeout=2.0)


def test_worker_reports_physics_error_to_ui_queue(monkeypatch):
    failed = threading.Event()

    def broken_step(_state, **_kwargs):
        failed.set()
        raise RuntimeError("synthetic physics failure")

    monkeypatch.setattr(main, "simulate_step", broken_step)
    worker = main.SimulationThread(initial_state=0)
    worker.start()
    worker.resume()
    assert failed.wait(timeout=2.0)

    try:
        deadline = time.monotonic() + 2.0
        while worker.error_queue.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert worker.error_queue.get_nowait() == "synthetic physics failure"
        assert worker.paused.is_set()
    finally:
        worker.stop()
        worker.join(timeout=2.0)


def test_cache_saved_elevation_restores_exact_array():
    elevation = np.arange(24, dtype=np.float32).reshape(4, 6) / 24.0
    state = type("SavedState", (), {"elevation": elevation})()

    try:
        main._cache_saved_elevation(state, Path("saves/test.pkl"))
        cached, key = get_elevation_cache()
        np.testing.assert_array_equal(cached, elevation)
        assert key == ("loaded", "saved-state:saves/test.pkl")
    finally:
        clear_elevation_cache()
