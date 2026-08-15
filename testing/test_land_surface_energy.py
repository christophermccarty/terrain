from __future__ import annotations

import dataclasses

import numpy as np

from planet_params import EARTH
from simulate import create_initial_state, simulate_step
from atmospheric_heat_transport import (
    apply_sensible_heat_to_atmospheric_column,
    close_global_heat_convergence,
    external_energy_budget_heat_convergence,
    summarize_heat_convergence_samples,
    temperature_transport_to_heat_convergence,
)


def _land_state():
    elevation = np.zeros((16, 32), dtype=np.float32)
    elevation[4:12, 8:24] = 0.5
    return elevation, create_initial_state(elevation, planet_params=EARTH)


def test_land_surface_energy_default_is_the_calibrated_path():
    _, state = _land_state()
    default, _ = simulate_step(state, days=1.0, planet_params=EARTH)
    explicit_default, _ = simulate_step(
        state,
        days=1.0,
        planet_params=dataclasses.replace(
            EARTH, enable_land_surface_energy=True, land_surface_energy_strength=0.001
        ),
    )
    np.testing.assert_array_equal(default.temperature, explicit_default.temperature)


def test_land_surface_energy_changes_land_state_at_experimental_strength():
    elevation, state = _land_state()
    baseline, _ = simulate_step(state, days=1.0, planet_params=EARTH)
    enabled, _ = simulate_step(
        state,
        days=1.0,
        planet_params=dataclasses.replace(EARTH, land_surface_energy_strength=0.05),
    )
    land = elevation > 0.0
    assert not np.array_equal(enabled.temperature[land], baseline.temperature[land])


def test_force_restore_land_is_a_gated_replacement_with_persistent_deep_soil():
    elevation, state = _land_state()
    enabled, _ = simulate_step(
        state,
        days=1.0,
        planet_params=dataclasses.replace(EARTH, enable_force_restore_land=True),
    )
    assert enabled.land_deep_temperature is not None
    land = elevation > 0.0
    assert np.all(np.isfinite(enabled.land_deep_temperature[land]))


def test_temperature_transport_converts_column_increment_to_energy_flux():
    flux = temperature_transport_to_heat_convergence(
        np.array([[1.0, -1.0]], dtype=np.float32),
        surface_pressure_pa=98_066.5,
        cp_j_kg_k=1_000.0,
        gravity_m_s2=9.80665,
        dt_seconds=10_000.0,
    )
    np.testing.assert_allclose(flux, [[1_000.0, -1_000.0]])


def test_sensible_heat_exchange_adds_flux_energy_only_over_land():
    air = np.full((1, 2), 280.0, dtype=np.float32)
    updated, gain = apply_sensible_heat_to_atmospheric_column(
        air,
        np.array([[100.0, 100.0]], dtype=np.float32),
        np.array([[True, False]]),
        surface_pressure_pa=100_000.0,
        cp_j_kg_k=1_000.0,
        gravity_m_s2=10.0,
        dt_seconds=100_000.0,
    )
    np.testing.assert_allclose(gain, [[100.0, 0.0]])
    np.testing.assert_allclose(updated, [[281.0, 280.0]])


def test_external_energy_budget_uses_downward_positive_boundary_fluxes():
    convergence = external_energy_budget_heat_convergence(
        np.array([100.0, 100.0]),
        np.array([120.0, 80.0]),
        atmospheric_storage_w_m2=np.array([5.0, -5.0]),
    )
    np.testing.assert_array_equal(convergence, [25.0, -25.0])


def test_closed_horizontal_heat_convergence_has_zero_area_integral():
    latitude = np.deg2rad(np.array([60.0, 0.0, -60.0]))
    field = np.array([[10.0, 10.0], [4.0, 8.0], [-2.0, 6.0]], dtype=np.float32)
    closed = close_global_heat_convergence(field, latitude)
    weights = np.cos(latitude)[:, None]
    assert abs(float(np.sum(closed * weights))) < 1e-5
    np.testing.assert_allclose(closed - field, np.full_like(field, closed[0, 0] - field[0, 0]))


def test_heat_convergence_summary_reports_closure_and_plausibility_range():
    latitude = np.deg2rad(np.array([45.0, -45.0]))
    samples = [
        close_global_heat_convergence(
            np.array([[10.0, -4.0], [2.0, 8.0]], dtype=np.float32), latitude
        ),
        close_global_heat_convergence(
            np.array([[5.0, -3.0], [-1.0, 7.0]], dtype=np.float32), latitude
        ),
    ]
    summary = summarize_heat_convergence_samples(samples, latitude)
    assert summary["sample_count"] == 2
    assert summary["max_abs_global_area_mean_w_m2"] < 1e-6
    assert summary["p05_w_m2"] < summary["p95_w_m2"]
    assert summary["max_abs_w_m2"] >= summary["mean_rms_w_m2"]


def test_temperature_to_flux_is_timestep_invariant_for_the_same_tendency():
    one_day = temperature_transport_to_heat_convergence(
        np.array([[0.2]], dtype=np.float32),
        surface_pressure_pa=100_000.0,
        cp_j_kg_k=1_000.0,
        gravity_m_s2=10.0,
        dt_seconds=86_400.0,
    )
    half_day = temperature_transport_to_heat_convergence(
        np.array([[0.1]], dtype=np.float32),
        surface_pressure_pa=100_000.0,
        cp_j_kg_k=1_000.0,
        gravity_m_s2=10.0,
        dt_seconds=43_200.0,
    )
    np.testing.assert_allclose(one_day, half_day)


def test_force_restore_heat_convergence_gate_is_default_off_and_effective():
    elevation, state = _land_state()
    force_restore = dataclasses.replace(EARTH, enable_force_restore_land=True)
    baseline, _ = simulate_step(state, days=1.0, planet_params=force_restore)
    explicit_off, _ = simulate_step(
        state,
        days=1.0,
        planet_params=dataclasses.replace(
            force_restore, enable_force_restore_atmospheric_heat_convergence=False
        ),
    )
    enabled, diagnostics = simulate_step(
        state,
        days=1.0,
        planet_params=dataclasses.replace(
            force_restore, enable_force_restore_atmospheric_heat_convergence=True
        ),
        track_components=True,
    )
    np.testing.assert_array_equal(baseline.temperature, explicit_off.temperature)
    land = elevation > 0.0
    assert not np.array_equal(enabled.temperature[land], baseline.temperature[land])
    assert "atmospheric_heat_convergence_w_m2" in diagnostics


def test_conservative_land_air_exchange_is_default_off_and_closes_sensible_flux():
    elevation, state = _land_state()
    force_restore = dataclasses.replace(
        EARTH,
        enable_force_restore_land=True,
        enable_force_restore_atmospheric_heat_convergence=True,
    )
    baseline, _ = simulate_step(state, days=1.0, planet_params=force_restore)
    explicit_off, _ = simulate_step(
        state,
        days=1.0,
        planet_params=dataclasses.replace(
            force_restore, enable_force_restore_conservative_land_air_exchange=False
        ),
    )
    enabled, diagnostics = simulate_step(
        state,
        days=1.0,
        planet_params=dataclasses.replace(
            force_restore, enable_force_restore_conservative_land_air_exchange=True
        ),
        track_components=True,
    )
    np.testing.assert_array_equal(baseline.temperature, explicit_off.temperature)
    np.testing.assert_array_equal(baseline.air_temperature, explicit_off.air_temperature)
    land = elevation > 0.0
    assert not np.array_equal(enabled.air_temperature[land], baseline.air_temperature[land])
    closure = diagnostics["land_air_sensible_exchange_closure_w_m2"]
    assert float(np.max(np.abs(closure))) < 1e-6
    gain = diagnostics["land_air_sensible_atmospheric_gain_w_m2"]
    assert np.all(np.isfinite(gain[land]))


def test_resolved_heat_convergence_is_bounded_under_half_step_partition():
    elevation, state = _land_state()
    params = dataclasses.replace(
        EARTH,
        enable_force_restore_land=True,
        enable_force_restore_atmospheric_heat_convergence=True,
    )
    _, full = simulate_step(
        state, days=1.0, planet_params=params, track_components=True, update_wind=False
    )
    half_state, first_half = simulate_step(
        state, days=0.5, planet_params=params, track_components=True, update_wind=False
    )
    _, second_half = simulate_step(
        half_state, days=0.5, planet_params=params, track_components=True, update_wind=False
    )
    full_flux = full["atmospheric_heat_convergence_w_m2"]
    split_flux = 0.5 * (
        first_half["atmospheric_heat_convergence_w_m2"]
        + second_half["atmospheric_heat_convergence_w_m2"]
    )
    correlation = float(np.corrcoef(full_flux.ravel(), split_flux.ravel())[0, 1])
    relative_rms = float(
        np.sqrt(np.mean((full_flux - split_flux) ** 2))
        / np.sqrt(np.mean(full_flux**2))
    )
    assert correlation > 0.95
    assert relative_rms < 0.25
    for diagnostics in (full, first_half, second_half):
        assert abs(
            diagnostics["atmospheric_heat_convergence_applied_grid_area_mean_w_m2"]
        ) < 1e-4
