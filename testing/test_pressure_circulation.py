from __future__ import annotations

import dataclasses
import numpy as np

from atmosphere import flux_divergence_spherical
from planet_params import EARTH
from pressure_circulation import (
    balanced_thermal_wind_u,
    close_upper_mass_flux,
    smooth_spherical_scalar,
    spherical_divergence,
)
from simulate import create_initial_state, simulate_step


def test_horizontal_mass_closure_reduces_column_divergence_residual():
    h, w = 24, 48
    lon = np.linspace(0.0, 2.0 * np.pi, w, endpoint=False)
    upper_u = np.broadcast_to(8.0 * np.sin(lon)[None, :], (h, w)).copy()
    zeros = np.zeros((h, w))
    closure = close_upper_mass_flux(zeros, zeros, zeros, zeros, upper_u, zeros, radius_m=6.371e6)
    assert closure.residual_after_s < 0.5 * closure.residual_before_s
    assert np.max(np.hypot(closure.upper_u_correction, closure.upper_v_correction)) <= 12.001


def test_horizontal_mass_closure_removes_divergence_free_equatorial_throughflow():
    h, w = 24, 48
    lat = np.radians(90.0 - (np.arange(h) + 0.5) * 180.0 / h)
    # v*cos(latitude) is constant and therefore invisible to divergence.
    upper_v = np.broadcast_to(6.0 / np.cos(lat)[:, None], (h, w))
    zeros = np.zeros((h, w))
    closure = close_upper_mass_flux(
        zeros, zeros, zeros, zeros, zeros, upper_v,
        radius_m=6.371e6, throughflow_max_speed_m_s=80.0,
    )
    assert abs(closure.equatorial_throughflow_after_m_s) < 1e-6
    corrected = upper_v + closure.upper_v_correction
    equatorial = (h // 2 - 1, h // 2)
    assert abs(float(np.mean(corrected[equatorial]))) < 1e-5


def test_divergence_matches_northward_meridional_sign_convention():
    h, w = 24, 48
    v = np.full((h, w), 5.0)
    divergence = spherical_divergence(np.zeros_like(v), v, radius_m=6.371e6)
    # Uniform northward flow converges in the Northern Hemisphere and diverges
    # in the Southern Hemisphere because meridians converge poleward.
    assert float(np.mean(divergence[: h // 3])) < 0.0
    assert float(np.mean(divergence[-h // 3 :])) > 0.0


def test_divergence_is_identical_to_production_unit_mass_operator():
    rng = np.random.default_rng(41)
    h, w = 24, 48
    u = rng.normal(size=(h, w))
    v = rng.normal(size=(h, w))
    latitude = np.radians(90.0 - (np.arange(h) + 0.5) * 180.0 / h)
    expected = flux_divergence_spherical(
        np.ones((h, w)), u, v, latitude, radius_m=6.371e6,
    )
    np.testing.assert_allclose(
        spherical_divergence(u, v, radius_m=6.371e6), expected,
    )


def test_divergence_filter_preserves_area_mean_and_damps_grid_scale_noise():
    h, w = 24, 48
    checkerboard = (-1.0) ** (np.add.outer(np.arange(h), np.arange(w)))
    filtered = smooth_spherical_scalar(checkerboard, strength=0.30, passes=2)
    weights = np.cos(np.radians(90.0 - (np.arange(h) + 0.5) * 180.0 / h))[:, None]
    np.testing.assert_allclose(np.sum(filtered * weights), np.sum(checkerboard * weights), atol=1e-12)
    assert float(np.sqrt(np.mean(filtered**2))) < float(np.sqrt(np.mean(checkerboard**2)))


def test_balanced_thermal_wind_puts_westerly_shear_outside_hadley_cell():
    h, w = 36, 72
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    temperature = 300.0 - 0.45 * np.abs(latitude)[:, None]
    target = balanced_thermal_wind_u(
        np.zeros((h, w)), np.broadcast_to(temperature, (h, w)),
        radius_m=6.371e6, sidereal_day_hours=24.0,
        surface_pressure_pa=101325.0, upper_pressure_pa=30_000.0,
        hadley_edge_deg=24.0,
    )
    zonal = np.mean(target, axis=1)
    assert float(np.max(zonal[np.abs(latitude) >= 30.0])) > 5.0
    assert float(np.max(np.abs(zonal[np.abs(latitude) <= 5.0]))) < 0.5


def test_pressure_column_gate_plumbs_closure_into_persisted_upper_wind():
    shape = (12, 24)
    common = dict(
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
        enable_three_level_pressure_column=True,
    )
    disabled = dataclasses.replace(EARTH, **common)
    enabled = dataclasses.replace(
        EARTH, **common, enable_three_level_horizontal_mass_flux_closure=True,
    )
    state = create_initial_state(np.zeros(shape, dtype=np.float32), planet_params=disabled)
    longitude_wave = np.broadcast_to(
        8.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, shape[1], endpoint=False))[None, :],
        shape,
    ).astype(np.float32)
    state = state._replace(
        wind_u=np.zeros(shape, dtype=np.float32),
        wind_v=np.zeros(shape, dtype=np.float32),
        wind_u_aloft=longitude_wave,
        wind_v_aloft=np.zeros(shape, dtype=np.float32),
        midlevel_wind_u=np.zeros(shape, dtype=np.float32),
        midlevel_wind_v=np.zeros(shape, dtype=np.float32),
    )
    control, _ = simulate_step(state, days=1.0, planet_params=disabled)
    closed, _ = simulate_step(state, days=1.0, planet_params=enabled)
    assert closed.wind_u_aloft is not None
    assert closed.omega_lower_mid_pa_s is not None
    assert closed.omega_mid_upper_pa_s is not None
    assert np.all(np.isfinite(closed.wind_u_aloft))
    assert np.all(np.isfinite(closed.omega_lower_mid_pa_s))
    assert np.all(np.isfinite(closed.omega_mid_upper_pa_s))
    assert float(np.mean(np.abs(closed.wind_u_aloft - control.wind_u_aloft))) > 1e-3
