"""Long-period orbital-element evolution for scenario experiments."""
from __future__ import annotations

from dataclasses import replace
import math

from planet_params import PlanetParams


OBLIQUITY_PERIOD_YEARS = 41_000.0
ECCENTRICITY_PERIOD_YEARS = 100_000.0
PRECESSION_PERIOD_YEARS = 23_000.0


def orbital_params_at_time(
    base: PlanetParams,
    total_days: float,
) -> PlanetParams:
    """Return effective orbital parameters at monotonic simulation time.

    The input remains the immutable scenario baseline; this function never
    feeds its output back as the next baseline, avoiding cumulative phase drift.
    """
    if not base.enable_milankovitch_cycles:
        return base
    if base.orbital_period_days <= 0.0:
        raise ValueError("orbital_period_days must be positive")
    if base.milankovitch_time_scale < 0.0:
        raise ValueError("milankovitch_time_scale must be non-negative")

    simulated_years = float(total_days) / float(base.orbital_period_days)
    cycle_years = simulated_years * float(base.milankovitch_time_scale)
    tau = 2.0 * math.pi

    obliquity = base.obliquity_deg + base.milankovitch_obliquity_amplitude_deg * math.sin(
        tau * cycle_years / OBLIQUITY_PERIOD_YEARS
    )
    eccentricity = base.eccentricity + base.milankovitch_eccentricity_amplitude * math.sin(
        tau * cycle_years / ECCENTRICITY_PERIOD_YEARS
    )
    perihelion_day = (
        base.perihelion_day
        + base.orbital_period_days * cycle_years / PRECESSION_PERIOD_YEARS
    ) % base.orbital_period_days

    return replace(
        base,
        obliquity_deg=float(max(0.0, min(90.0, obliquity))),
        eccentricity=float(max(0.0, min(0.95, eccentricity))),
        perihelion_day=float(perihelion_day),
    )
