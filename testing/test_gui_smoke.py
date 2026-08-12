"""Minimal Tk startup coverage for the interactive application."""
from __future__ import annotations

import tkinter as tk

import pytest

import main


def test_gui_assembles_and_enters_event_loop(monkeypatch):
    """Build the real UI, process its first event, then close without user input."""
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display is unavailable: {exc}")

    root.withdraw()
    root.after(1, root.destroy)
    monkeypatch.setattr(main, "load_settings", lambda: {})

    main.main(root_factory=lambda: root)
