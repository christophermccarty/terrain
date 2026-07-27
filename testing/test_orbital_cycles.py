from __future__ import annotations

import dataclasses

import pytest

from orbital_cycles import (
    ECCENTRICITY_PERIOD_YEARS,
    OBLIQUITY_PERIOD_YEARS,
    PRECESSION_PERIOD_YEARS,
    orbital_params_at_time,
)
from planet_params import EARTH


def _days_for_cycle_years(pp, cycle_years):
    return pp.orbital_period_days * cycle_years / pp.milankovitch_time_scale


def test_disabled_orbital_cycles_are_exact_noop():
    assert orbital_params_at_time(EARTH, 1e9) is EARTH


def test_obliquity_and_eccentricity_hit_configured_amplitudes():
    pp = dataclasses.replace(
        EARTH,
        enable_milankovitch_cycles=True,
        milankovitch_time_scale=1000.0,
    )
    obliquity_peak = orbital_params_at_time(
        pp, _days_for_cycle_years(pp, OBLIQUITY_PERIOD_YEARS / 4.0)
    )
    eccentricity_peak = orbital_params_at_time(
        pp, _days_for_cycle_years(pp, ECCENTRICITY_PERIOD_YEARS / 4.0)
    )

    assert obliquity_peak.obliquity_deg == pytest.approx(
        pp.obliquity_deg + pp.milankovitch_obliquity_amplitude_deg
    )
    assert eccentricity_peak.eccentricity == pytest.approx(
        pp.eccentricity + pp.milankovitch_eccentricity_amplitude
    )
    assert pp.obliquity_deg == EARTH.obliquity_deg


def test_precession_advances_perihelion_through_full_orbit():
    pp = dataclasses.replace(
        EARTH,
        enable_milankovitch_cycles=True,
        milankovitch_time_scale=1000.0,
    )
    half_cycle = orbital_params_at_time(
        pp, _days_for_cycle_years(pp, PRECESSION_PERIOD_YEARS / 2.0)
    )
    expected = (pp.perihelion_day + 0.5 * pp.orbital_period_days) % pp.orbital_period_days
    assert half_cycle.perihelion_day == pytest.approx(expected)


def test_negative_cycle_time_scale_rejected():
    pp = dataclasses.replace(
        EARTH,
        enable_milankovitch_cycles=True,
        milankovitch_time_scale=-1.0,
    )
    with pytest.raises(ValueError, match="non-negative"):
        orbital_params_at_time(pp, 1.0)
