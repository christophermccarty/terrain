from __future__ import annotations

import dataclasses
import numpy as np

from atmosphere import flux_divergence_spherical
from planet_params import EARTH
from pressure_circulation import (
    balanced_thermal_wind_u,
    close_upper_mass_flux,
    diabatic_interface_mass_flux,
    diabatic_interface_mass_flux_from_heating,
    evolve_large_scale_heating_reservoir,
    shared_pressure_coordinate_circulation,
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


def test_large_scale_heating_reservoir_is_zonal_balanced_and_radiatively_adjusted():
    h, w = 24, 48
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    condensation = np.broadcast_to(
        (1.0 + 8.0 * np.exp(-(latitude / 18.0) ** 2))[:, None], (h, w)
    )
    temperature = np.full((h, w), 280.0)
    step = evolve_large_scale_heating_reservoir(
        None, condensation, temperature, temperature - 20.0, temperature - 45.0,
        dt_seconds=86400.0, surface_pressure_pa=101325.0,
    )
    weights = np.cos(np.radians(latitude))[:, None]
    assert np.all(step.radiative_adjustment_time_s > 86400.0)
    assert abs(float(np.sum(step.heating_w_m2 * weights))) < 1e-4
    assert float(np.max(step.heating_w_m2)) > 0.0
    interface = diabatic_interface_mass_flux_from_heating(
        step.heating_w_m2, temperature, temperature - 20.0, temperature - 45.0,
        dt_seconds=86400.0, surface_pressure_pa=101325.0,
        lower_mid_pressure_depth_pa=35000.0, mid_upper_pressure_depth_pa=25000.0,
    )
    assert interface.lower_mid_vertical_courant_max < 1.0


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


def test_diabatic_interface_mass_flux_is_zonal_mean_mass_closed_and_cfl_reported():
    h, w = 24, 48
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lower = np.broadcast_to(300.0 - 0.20 * np.abs(latitude)[:, None], (h, w))
    middle = lower - 22.0
    upper = middle - 24.0
    precipitation = np.broadcast_to(
        (2.0 + 3.0 * np.exp(-(latitude / 18.0) ** 2))[:, None], (h, w)
    )
    result = diabatic_interface_mass_flux(
        precipitation, lower, middle, upper,
        dt_seconds=86400.0, surface_pressure_pa=101325.0,
        lower_mid_pressure_depth_pa=35000.0, mid_upper_pressure_depth_pa=30000.0,
    )
    weighted_divergence = (
        0.40 * result.lower_divergence_s
        + 0.35 * result.midlevel_divergence_s
        + 0.25 * result.upperlevel_divergence_s
    )
    np.testing.assert_allclose(weighted_divergence, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.omega_lower_mid_pa_s,
        0.5 * 35000.0 * (result.lower_divergence_s - result.midlevel_divergence_s),
        rtol=2e-6, atol=2e-8,
    )
    assert float(np.min(result.omega_lower_mid_pa_s)) < 0.0
    assert float(np.max(result.omega_lower_mid_pa_s)) > 0.0
    assert float(np.min(result.omega_mid_upper_pa_s)) < 0.0
    assert float(np.max(result.omega_mid_upper_pa_s)) > 0.0
    assert result.lower_mid_vertical_courant_max < 0.25
    assert result.mid_upper_vertical_courant_max < 0.25


def test_diabatic_interface_mass_flux_uses_no_hidden_stability_floor():
    shape = (8, 16)
    lower = np.full(shape, 280.0)
    # A convectively unstable lower-mid interface must be left for the
    # convective closure, not turned into a large arbitrary pressure velocity.
    # At the midlevel's lower pressure this must be much colder than the
    # lower layer to make potential temperature decrease upward. A raw-T
    # inversion alone is not convective instability.
    middle = np.full(shape, 210.0)
    upper = np.full(shape, 250.0)
    precipitation = np.broadcast_to(
        (1.0 + 5.0 * np.exp(-((np.arange(shape[0]) - 0.5 * shape[0]) / 1.5) ** 2))[:, None], shape
    )
    result = diabatic_interface_mass_flux(
        precipitation, lower, middle, upper,
        dt_seconds=86400.0, surface_pressure_pa=101325.0,
        lower_mid_pressure_depth_pa=35000.0, mid_upper_pressure_depth_pa=30000.0,
    )
    np.testing.assert_array_equal(result.omega_lower_mid_pa_s, np.zeros(shape, dtype=np.float32))


def test_diabatic_interface_mass_flux_uses_potential_not_raw_temperature_stability():
    """A raw-T-neutral but pressure-stratified column must not be singular."""
    shape = (8, 16)
    heating = np.broadcast_to(
        np.linspace(-80.0, 80.0, shape[0])[:, None], shape
    )
    result = diabatic_interface_mass_flux_from_heating(
        heating,
        np.full(shape, 280.0),
        np.full(shape, 280.0),
        np.full(shape, 280.0),
        dt_seconds=86400.0, surface_pressure_pa=101325.0,
        lower_mid_pressure_depth_pa=35000.0, mid_upper_pressure_depth_pa=30000.0,
    )
    assert np.all(np.isfinite(result.omega_lower_mid_pa_s))
    assert np.all(np.isfinite(result.omega_mid_upper_pa_s))
    assert float(np.max(np.abs(result.omega_lower_mid_pa_s))) < 0.1
    assert float(np.max(np.abs(result.omega_mid_upper_pa_s))) < 0.1
    assert np.any(result.omega_mid_upper_pa_s != 0.0)


def test_shared_pressure_coordinate_circulation_matches_its_omega_divergence():
    h, w = 24, 48
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lower = np.broadcast_to(300.0 - 0.20 * np.abs(latitude)[:, None], (h, w))
    middle = lower - 22.0
    upper = middle - 24.0
    precipitation = np.broadcast_to(
        (2.0 + 3.0 * np.exp(-(latitude / 18.0) ** 2))[:, None], (h, w)
    )
    raw_u = np.broadcast_to(15.0 * np.sin(np.radians(latitude))[:, None], (h, w))
    circulation = shared_pressure_coordinate_circulation(
        precipitation, lower, middle, upper, raw_u, 0.5 * raw_u, 1.5 * raw_u,
        dt_seconds=86400.0, radius_m=6.371e6, surface_pressure_pa=101325.0,
        lower_mid_pressure_depth_pa=35000.0, mid_upper_pressure_depth_pa=30000.0,
    )
    for u, v, target in (
        (circulation.lower_u, circulation.lower_v, circulation.interface_mass_flux.lower_divergence_s),
        (circulation.midlevel_u, circulation.midlevel_v, circulation.interface_mass_flux.midlevel_divergence_s),
        (circulation.upperlevel_u, circulation.upperlevel_v, circulation.interface_mass_flux.upperlevel_divergence_s),
    ):
        np.testing.assert_allclose(
            spherical_divergence(u, v, radius_m=6.371e6), target, atol=2e-12,
        )
    assert float(np.sqrt(np.mean(circulation.upperlevel_v**2))) < 3.0
    weighted = (
        0.40 * circulation.interface_mass_flux.lower_divergence_s
        + 0.35 * circulation.interface_mass_flux.midlevel_divergence_s
        + 0.25 * circulation.interface_mass_flux.upperlevel_divergence_s
    )
    np.testing.assert_allclose(weighted, 0.0, atol=1e-12)


def test_diabatic_interface_runtime_substeps_instead_of_capping_courant():
    shape = (12, 24)
    planet = dataclasses.replace(
        EARTH,
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
        enable_three_level_pressure_column=True,
        enable_closed_three_level_thermodynamics=True,
        enable_diabatic_interface_mass_flux=True,
    )
    state = create_initial_state(np.zeros(shape, dtype=np.float32), planet_params=planet)
    latitude = 90.0 - (np.arange(shape[0]) + 0.5) * 180.0 / shape[0]
    state = state._replace(precipitation=np.broadcast_to(
        (1.0 + 50.0 * np.exp(-(latitude / 16.0) ** 2))[:, None], shape
    ).astype(np.float32))
    debug: dict = {}
    evolved, _ = simulate_step(
        state, days=1.0, planet_params=planet, precipitation_debug=debug
    )

    assert evolved.omega_lower_mid_pa_s is not None
    assert debug["diabatic_interface_vertical_substeps"] > 1
    cfl = max(
        float(debug["diabatic_interface_lower_mid_courant_max"]),
        float(debug["diabatic_interface_mid_upper_courant_max"]),
    )
    assert cfl / int(debug["diabatic_interface_vertical_substeps"]) <= 0.25 + 1e-12


def test_shared_pressure_coordinate_runtime_routes_winds_with_the_interface_source():
    shape = (12, 24)
    planet = dataclasses.replace(
        EARTH,
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
        enable_three_level_pressure_column=True,
        enable_closed_three_level_thermodynamics=True,
        enable_diabatic_interface_mass_flux=True,
        enable_shared_pressure_coordinate_circulation=True,
    )
    latitude = 90.0 - (np.arange(shape[0]) + 0.5) * 180.0 / shape[0]
    state = create_initial_state(np.zeros(shape, dtype=np.float32), planet_params=planet)
    state = state._replace(precipitation=np.broadcast_to(
        (1.0 + 8.0 * np.exp(-(latitude / 20.0) ** 2))[:, None], shape
    ).astype(np.float32))
    debug: dict = {}
    evolved, _ = simulate_step(
        state, days=1.0, planet_params=planet, precipitation_debug=debug
    )

    assert debug["shared_pressure_circulation"] is True
    assert evolved.midlevel_wind_v is not None
    assert evolved.upperlevel_wind_v is not None
    assert evolved.omega_lower_mid_pa_s is not None
    assert float(np.sqrt(np.mean(evolved.upperlevel_wind_v**2))) < 5.0
    assert float(np.sqrt(np.mean(evolved.omega_lower_mid_pa_s**2))) < 0.1


def test_prognostic_overturning_heat_reservoir_persists_a_derived_heating_state():
    shape = (12, 24)
    planet = dataclasses.replace(
        EARTH,
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
        enable_three_level_pressure_column=True,
        enable_closed_three_level_thermodynamics=True,
        enable_diabatic_interface_mass_flux=True,
        enable_shared_pressure_coordinate_circulation=True,
        enable_pressure_coordinate_moisture_closure=True,
        enable_prognostic_overturning_heat_reservoir=True,
    )
    initial = create_initial_state(np.zeros(shape, dtype=np.float32), planet_params=planet)
    debug: dict = {}
    evolved, _ = simulate_step(
        initial, days=1.0, planet_params=planet, precipitation_debug=debug
    )
    assert evolved.pressure_overturning_heating_w_m2 is not None
    assert evolved.pressure_overturning_heating_w_m2.shape == shape
    assert "prognostic_overturning_adjustment_time_days" in debug
    assert float(np.mean(debug["prognostic_overturning_adjustment_time_days"])) > 1.0


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
    # Section 17 (PRIOR_ART_IMPLEMENTATION_PLAN.md): the mass-flux closure's
    # correction now applies to the three-level path's own independent
    # upperlevel_wind_u/v state, not the shared, always-on jet-stream kernel
    # -- so the gate must change the former and leave the latter
    # bit-identical.
    assert closed.upperlevel_wind_u is not None
    assert float(np.mean(np.abs(closed.upperlevel_wind_u - control.upperlevel_wind_u))) > 1e-3
    np.testing.assert_array_equal(closed.wind_u_aloft, control.wind_u_aloft)
