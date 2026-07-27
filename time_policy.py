"""Central time-scale integration policy.

The GUI and headless runner must use exactly the same substep schedule.  Slow
climate modes are defined relative to the simulated planet's orbital period so
one displayed month/year cannot silently precess through the seasons.
"""
from __future__ import annotations

from typing import Protocol


class _ModeLike(Protocol):
    value: str


class _PlanetLike(Protocol):
    orbital_period_days: float


_MONTHS_PER_ORBIT = 12
_MONTHLY_SUBSTEPS = 5
_ANNUAL_SUBSTEPS = 52


def _mode_name(mode: _ModeLike | str) -> str:
    value = getattr(mode, "value", mode)
    return str(value).lower()


def substeps_for_mode(
    mode: _ModeLike | str,
    planet_params: _PlanetLike,
) -> tuple[tuple[float, bool], ...]:
    """Return ``(step_days, update_wind)`` entries for one UI/headless cycle.

    DAILY and WEEKLY retain literal Earth-day durations.  MONTHLY means one
    twelfth of *this planet's* orbit and ANNUAL means one complete orbit.
    Equal fractional substeps preserve the existing call counts and therefore
    the intended performance/fidelity profile.
    """
    name = _mode_name(mode)
    period = float(planet_params.orbital_period_days)
    if period <= 0.0:
        raise ValueError("orbital_period_days must be positive")

    if name == "daily":
        return ((1.0, True),)
    if name == "weekly":
        return ((1.0, True),) * 7
    if name == "monthly":
        step_days = period / (_MONTHS_PER_ORBIT * _MONTHLY_SUBSTEPS)
        return ((step_days, False),) * _MONTHLY_SUBSTEPS
    if name == "annual":
        step_days = period / _ANNUAL_SUBSTEPS
        return ((step_days, False),) * _ANNUAL_SUBSTEPS
    raise ValueError(f"Unsupported time-scale mode: {mode!r}")


def cycle_days(mode: _ModeLike | str, planet_params: _PlanetLike) -> float:
    """Return the exact simulated days advanced by one mode cycle."""
    return float(sum(step_days for step_days, _ in substeps_for_mode(mode, planet_params)))
