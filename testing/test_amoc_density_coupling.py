"""Tests for the AMOC temperature-density term (FEATURES.md item 5).

Mirrors the existing salinity-AMOC coupling (Feature 3, see test_salinity.py's
sibling mechanism in simulate.py), but for temperature: a warmer North
Atlantic (50-75N) is less dense and should weaken amoc_factor, a colder one
should strengthen it. Ships default off (`temperature_amoc_scale=0.0`) --
these tests exercise the opt-in path directly.

Latitude convention matches simulate.py's internal `_lat_ice` exactly: signed
latitude (row 0 = north pole, +90 -> -90), NOT `abs(lat)` -- the North
Atlantic sinking region the AMOC terms target is Northern-Hemisphere-only, and
using signed latitude is what keeps the 50-75 degree band out of the Southern
Ocean.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from masks import get_masks
from planet_params import EARTH
from simulate import create_initial_state, simulate_step
from testing.conftest import make_mixed_elev

H, W = 32, 64


def _signed_lat_deg(H: int) -> np.ndarray:
    """Reproduce simulate.py's `_lat_ice` exactly (signed, north-positive)."""
    return (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * 180.0


def _na_and_arctic_rows(H: int) -> tuple[np.ndarray, np.ndarray]:
    lat = _signed_lat_deg(H)
    return (lat >= 50.0) & (lat <= 75.0), lat >= 75.0


def test_temperature_amoc_scale_default_is_inert():
    """0.0 (the default) must reproduce the pre-existing behavior exactly."""
    elev = make_mixed_elev(H, W)
    state = create_initial_state(elev, day_of_year=80.0)
    baseline, _ = simulate_step(state, days=1.0, planet_params=EARTH)

    pp_perturbed_ref = dataclasses.replace(EARTH, temperature_amoc_reference_k=250.0)
    changed, _ = simulate_step(state, days=1.0, planet_params=pp_perturbed_ref)

    # scale stays 0.0 -> the term's `if` guard never fires, regardless of
    # how the (otherwise inert) reference constant is perturbed.
    assert np.allclose(baseline.temperature, changed.temperature, atol=1e-6)


def test_zero_anomaly_is_a_near_no_op_at_any_scale():
    """Reference pinned to the run's own North Atlantic mean -> anomaly ~0 ->
    the term should barely move the state, even at a large scale."""
    elev = make_mixed_elev(H, W)
    pp0 = dataclasses.replace(EARTH, temperature_amoc_scale=0.0)
    state = create_initial_state(elev, day_of_year=80.0, planet_params=pp0)
    state, _ = simulate_step(state, days=1.0, planet_params=pp0)

    na_rows, _ = _na_and_arctic_rows(H)
    sea_mask, _ = get_masks(state.elevation)
    na_sea = sea_mask & na_rows[:, None]
    assert np.any(na_sea), "fixture must have some North Atlantic ocean cells"
    na_mean_k = float(np.mean(state.temperature[na_sea]))

    pp_pinned = dataclasses.replace(
        EARTH, temperature_amoc_scale=8.0, temperature_amoc_reference_k=na_mean_k,
    )
    baseline_next, _ = simulate_step(state, days=1.0, planet_params=pp0)
    pinned_next, _ = simulate_step(state, days=1.0, planet_params=pp_pinned)
    assert np.allclose(baseline_next.temperature, pinned_next.temperature, atol=1e-3), (
        "anomaly=0 (reference pinned to the actual N.Atlantic mean) should be a "
        "near-no-op even at a large scale"
    )


def test_warm_north_atlantic_weakens_amoc_and_cools_the_arctic():
    """Directional physics check, isolated from the direct-diffusion confound
    of injecting an artificial temperature: inject the SAME warm North
    Atlantic into two runs, one with temperature_amoc_scale=0 (no density
    feedback) and one with it turned on. The turned-on run should show a
    weaker AMOC bonus and therefore a cooler NH Arctic (lat >= 75N) than the
    scale=0 run, since the warm-water diffusion confound is identical in both.
    """
    elev = make_mixed_elev(H, W)
    pp_base = dataclasses.replace(EARTH, salinity_amoc_scale=0.0)  # isolate this term
    state = create_initial_state(elev, day_of_year=80.0, planet_params=pp_base)
    state, _ = simulate_step(state, days=1.0, planet_params=pp_base)

    na_rows, arctic_rows = _na_and_arctic_rows(H)
    sea_mask, _ = get_masks(state.elevation)
    na_sea = sea_mask & na_rows[:, None]
    assert np.any(na_sea), "fixture must have some North Atlantic ocean cells"
    assert np.any(arctic_rows), "fixture must have some Arctic rows"

    warm_temp = state.temperature.copy()
    warm_temp[na_sea] = 295.0  # unambiguously warm, well above the 277.15K reference
    state_warm = state._replace(temperature=warm_temp)

    pp_off = dataclasses.replace(pp_base, temperature_amoc_scale=0.0)
    pp_on = dataclasses.replace(pp_base, temperature_amoc_scale=5.0)

    step_off, _ = simulate_step(state_warm, days=1.0, planet_params=pp_off)
    step_on, _ = simulate_step(state_warm, days=1.0, planet_params=pp_on)

    arctic_off = float(np.mean(step_off.temperature[arctic_rows]))
    arctic_on = float(np.mean(step_on.temperature[arctic_rows]))
    assert arctic_on < arctic_off, (
        f"turning on the density term with a warm N.Atlantic should weaken AMOC "
        f"and cool the Arctic relative to the no-feedback run: "
        f"scale=0 Arctic mean={arctic_off:.4f}K, scale=5 Arctic mean={arctic_on:.4f}K"
    )
