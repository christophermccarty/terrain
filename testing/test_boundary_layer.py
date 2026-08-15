from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from boundary_layer import (
    close_free_air_transport_energy,
    mixed_layer_pressure_thickness,
    overlying_layer_pressure_thickness,
    step_boundary_layer_energy,
    step_boundary_layer_interface_energy,
    transport_boundary_layer_energy,
)
from planet_params import EARTH
from simulate import create_initial_state, simulate_step


def _step(boundary, free, sensible, *, wind=0.005, dt=86_400.0):
    boundary = np.asarray(boundary, dtype=np.float64)
    return step_boundary_layer_energy(
        boundary,
        np.broadcast_to(np.asarray(free, dtype=np.float64), boundary.shape),
        np.broadcast_to(np.asarray(sensible, dtype=np.float64), boundary.shape),
        np.ones(boundary.shape, dtype=bool),
        surface_pressure_pa=100_000.0,
        cp_j_kg_k=1_000.0,
        gravity_m_s2=10.0,
        gas_constant_j_kg_k=287.0,
        reference_temperature_k=290.0,
        mixed_layer_depth_m=1_000.0,
        entrainment_velocity_m_s=wind,
        dt_seconds=dt,
    )


def test_free_air_transport_projection_conserves_variable_capacity_energy():
    initial = np.array([[280.0, 285.0], [295.0, 300.0]])
    raw = initial + np.array([[2.0, -1.0], [3.0, 0.5]])
    capacity = np.array([[8.0, 10.0], [7.0, 10.0]]) * 1.0e6
    latitude = np.deg2rad(np.array([45.0, -45.0]))
    corrected = close_free_air_transport_energy(
        initial, raw, capacity, latitude
    )
    weights = np.cos(latitude)[:, None]
    energy_change = capacity * (corrected - initial)
    assert abs(float(np.sum(weights * energy_change))) < 1e-5
    equilibrium = close_free_air_transport_energy(
        initial, initial, capacity, latitude
    )
    np.testing.assert_array_equal(equilibrium, initial)


def test_interface_column_uses_hydrostatic_mass_and_conserves_energy():
    boundary = np.array([[285.0, 295.0]])
    interface = np.array([[288.0, 292.0]])
    free = np.array([[290.0, 290.0]])
    sensible = np.array([[40.0, -15.0]])
    result = step_boundary_layer_interface_energy(
        boundary,
        interface,
        free,
        sensible,
        np.ones_like(boundary, dtype=bool),
        surface_pressure_pa=100_000.0,
        cp_j_kg_k=1_000.0,
        gravity_m_s2=10.0,
        gas_constant_j_kg_k=287.0,
        reference_temperature_k=290.0,
        mixed_layer_depth_m=1_000.0,
        entrainment_velocity_m_s=0.005,
        dt_seconds=21_600.0,
        wind_speed_m_s=np.full_like(boundary, 4.0),
    )
    boundary_capacity = result.boundary_pressure_thickness_pa / 10.0 * 1_000.0
    interface_capacity = result.interface_pressure_thickness_pa / 10.0 * 1_000.0
    free_capacity = (
        100_000.0
        - result.boundary_pressure_thickness_pa
        - result.interface_pressure_thickness_pa
    ) / 10.0 * 1_000.0
    before = (
        boundary_capacity * boundary
        + interface_capacity * interface
        + free_capacity * free
    )
    after = (
        boundary_capacity * result.boundary_temperature
        + interface_capacity * result.interface_temperature
        + free_capacity * result.free_temperature
    )
    np.testing.assert_allclose(after - before, sensible * 21_600.0, atol=1e-5)
    expected_interface_dp = overlying_layer_pressure_thickness(
        surface_pressure_pa=100_000.0,
        gravity_m_s2=10.0,
        gas_constant_j_kg_k=287.0,
        reference_temperature_k=290.0,
        layer_base_m=1_000.0,
        layer_depth_m=1_000.0,
    )
    assert result.interface_pressure_thickness_pa == pytest.approx(
        expected_interface_dp
    )


def test_interface_tke_entrainment_responds_to_wind_and_buoyancy_production():
    common = dict(
        surface_pressure_pa=100_000.0,
        cp_j_kg_k=1_000.0,
        gravity_m_s2=10.0,
        gas_constant_j_kg_k=287.0,
        reference_temperature_k=290.0,
        mixed_layer_depth_m=1_000.0,
        entrainment_velocity_m_s=0.005,
        dt_seconds=3_600.0,
    )
    calm = step_boundary_layer_interface_energy(
        np.full((1, 1), 285.0),
        np.full((1, 1), 290.0),
        np.full((1, 1), 290.0),
        np.zeros((1, 1)),
        np.ones((1, 1), dtype=bool),
        wind_speed_m_s=np.full((1, 1), 1.0),
        **common,
    )
    windy = step_boundary_layer_interface_energy(
        np.full((1, 1), 285.0),
        np.full((1, 1), 290.0),
        np.full((1, 1), 290.0),
        np.zeros((1, 1)),
        np.ones((1, 1), dtype=bool),
        wind_speed_m_s=np.full((1, 1), 8.0),
        **common,
    )
    heated = step_boundary_layer_interface_energy(
        np.full((1, 1), 285.0),
        np.full((1, 1), 290.0),
        np.full((1, 1), 290.0),
        np.full((1, 1), 100.0),
        np.ones((1, 1), dtype=bool),
        wind_speed_m_s=np.full((1, 1), 1.0),
        **common,
    )
    assert windy.effective_entrainment_velocity_m_s[0, 0] > (
        calm.effective_entrainment_velocity_m_s[0, 0]
    )
    assert heated.convective_entrainment_velocity_m_s[0, 0] > 0.0
    assert heated.effective_entrainment_velocity_m_s[0, 0] > (
        calm.effective_entrainment_velocity_m_s[0, 0]
    )
    assert all(np.all(np.isfinite(value)) for value in heated)


def test_pressure_thickness_is_hydrostatic_and_defines_capacity():
    thickness = mixed_layer_pressure_thickness(
        surface_pressure_pa=100_000.0, gravity_m_s2=10.0,
        gas_constant_j_kg_k=287.0, reference_temperature_k=290.0,
        mixed_layer_depth_m=1_000.0,
    )
    assert thickness == pytest.approx(11_326.0, rel=2e-3)
    assert 0.0 < thickness < 100_000.0


@pytest.mark.parametrize(
    "boundary,free,sensible,wind",
    [
        ([310.0], [300.0], [250.0], 0.005),
        ([275.0], [280.0], [15.0], 0.005),
        ([290.0], [290.0], [0.0], 0.005),
        ([330.0], [250.0], [-100.0], 1e-7),
    ],
)
def test_boundary_layer_cases_are_finite_and_conservative(boundary, free, sensible, wind):
    result = _step(boundary, free, sensible, wind=wind)
    cb = result.pressure_thickness_pa / 10.0 * 1_000.0
    cf = (100_000.0 - result.pressure_thickness_pa) / 10.0 * 1_000.0
    energy_before = cb * np.asarray(boundary) + cf * np.asarray(free)
    energy_after = cb * result.boundary_temperature + cf * result.free_temperature
    np.testing.assert_allclose(
        energy_after - energy_before, np.asarray(sensible) * 86_400.0,
        rtol=2e-5, atol=64.0,
    )
    assert np.all(np.isfinite(result.boundary_temperature))
    assert np.all(np.isfinite(result.free_temperature))
    np.testing.assert_allclose(
        result.exchange_gain_w_m2,
        -cf * (result.free_temperature - np.asarray(free)) / 86_400.0,
        rtol=2e-5, atol=2e-4,
    )


def test_equal_temperature_equilibrium_does_not_drift():
    result = _step([287.0, 287.0], [287.0, 287.0], [0.0, 0.0])
    np.testing.assert_array_equal(result.boundary_temperature, [287.0, 287.0])
    np.testing.assert_array_equal(result.free_temperature, [287.0, 287.0])


def test_exchange_is_stable_and_nearly_split_step_invariant():
    full = _step([315.0], [270.0], [0.0], dt=86_400.0)
    half1 = _step([315.0], [270.0], [0.0], dt=43_200.0)
    half2 = _step(half1.boundary_temperature, half1.free_temperature, [0.0], dt=43_200.0)
    np.testing.assert_allclose(full.boundary_temperature, half2.boundary_temperature, atol=2e-5)
    np.testing.assert_allclose(full.free_temperature, half2.free_temperature, atol=2e-5)


def test_bulk_richardson_gate_suppresses_stable_but_not_unstable_exchange():
    common = dict(
        surface_pressure_pa=100_000.0, cp_j_kg_k=1_000.0,
        gravity_m_s2=10.0, gas_constant_j_kg_k=287.0,
        reference_temperature_k=290.0, mixed_layer_depth_m=1_000.0,
        entrainment_velocity_m_s=0.005, dt_seconds=86_400.0,
        wind_speed_m_s=np.array([5.0]), stability_dependent_exchange=True,
    )
    stable = step_boundary_layer_energy(
        np.array([270.0]), np.array([290.0]), np.array([0.0]), np.array([True]),
        **common,
    )
    unstable = step_boundary_layer_energy(
        np.array([300.0]), np.array([290.0]), np.array([0.0]), np.array([True]),
        **common,
    )
    assert stable.bulk_richardson_number[0] > 0.25
    assert 0.0 < stable.effective_entrainment_velocity_m_s[0] < 0.005
    assert unstable.bulk_richardson_number[0] < 0.0
    assert unstable.effective_entrainment_velocity_m_s[0] == pytest.approx(0.005)
    assert 270.0 < stable.boundary_temperature[0] < 290.0


def _cell_area(height, width, radius=6.4e6):
    edges = np.linspace(np.pi / 2.0, -np.pi / 2.0, height + 1)
    rows = radius**2 * (2.0 * np.pi / width) * (
        np.sin(edges[:-1]) - np.sin(edges[1:])
    )
    return np.broadcast_to(rows[:, None], (height, width))


def test_flux_form_transport_preserves_uniform_equilibrium():
    shape = (8, 16)
    result = transport_boundary_layer_energy(
        np.full(shape, 285.0), np.full(shape, 285.0),
        np.full(shape, 18.0), np.linspace(-8.0, 8.0, shape[0])[:, None] * np.ones(shape),
        pressure_thickness_pa=10_000.0, surface_pressure_pa=100_000.0,
        cp_j_kg_k=1_000.0, gravity_m_s2=10.0, radius_m=6.4e6,
        dt_seconds=86_400.0,
    )
    np.testing.assert_allclose(result.boundary_temperature, 285.0, atol=1e-12)
    np.testing.assert_allclose(result.free_temperature, 285.0, atol=1e-12)


def test_flux_form_transport_conserves_total_atmospheric_energy():
    shape = (8, 16)
    boundary = np.full(shape, 280.0)
    boundary[:, 3:6] = 310.0
    free = np.full(shape, 270.0)
    u = np.linspace(-20.0, 20.0, shape[1])[None, :] * np.ones(shape)
    v = np.linspace(10.0, -10.0, shape[0])[:, None] * np.ones(shape)
    result = transport_boundary_layer_energy(
        boundary, free, u, v,
        pressure_thickness_pa=10_000.0, surface_pressure_pa=100_000.0,
        cp_j_kg_k=1_000.0, gravity_m_s2=10.0, radius_m=6.4e6,
        dt_seconds=3.0 * 86_400.0,
    )
    area = _cell_area(*shape)
    cb = 10_000.0 / 10.0 * 1_000.0
    cf = 90_000.0 / 10.0 * 1_000.0
    before = np.sum(area * (cb * boundary + cf * free))
    after = np.sum(area * (cb * result.boundary_temperature + cf * result.free_temperature))
    assert after == pytest.approx(before, rel=2e-15)
    area_mean = float(np.sum(area * result.horizontal_convergence_w_m2) / np.sum(area))
    assert abs(area_mean) < 1e-10
    assert np.all(np.isfinite(result.boundary_temperature))
    assert np.all(np.isfinite(result.free_temperature))
    assert result.substeps > 1


def test_land_only_transport_conserves_energy_and_leaves_no_ocean_reservoir():
    shape = (6, 12)
    land = np.zeros(shape, dtype=bool)
    land[1:5, 3:9] = True
    boundary = np.full(shape, 300.0)
    boundary[land] = 285.0
    free = np.full(shape, 275.0)
    result = transport_boundary_layer_energy(
        boundary, free, np.full(shape, 12.0), np.full(shape, -4.0),
        pressure_thickness_pa=10_000.0, surface_pressure_pa=100_000.0,
        cp_j_kg_k=1_000.0, gravity_m_s2=10.0, radius_m=6.4e6,
        dt_seconds=86_400.0, active_mask=land,
    )
    area = _cell_area(*shape)
    cb = 10_000.0 / 10.0 * 1_000.0
    cf_land = 90_000.0 / 10.0 * 1_000.0
    cf_ocean = 100_000.0 / 10.0 * 1_000.0
    before = np.sum(area * np.where(
        land, cb * boundary + cf_land * free, cf_ocean * free
    ))
    after = np.sum(area * np.where(
        land,
        cb * result.boundary_temperature + cf_land * result.free_temperature,
        cf_ocean * result.free_temperature,
    ))
    assert after == pytest.approx(before, rel=2e-15)
    np.testing.assert_allclose(
        result.boundary_temperature[~land], result.free_temperature[~land], atol=1e-12
    )
    total_convergence = (
        result.horizontal_convergence_w_m2
        + result.free_horizontal_convergence_w_m2
    )
    assert abs(float(np.sum(area * total_convergence) / np.sum(area))) < 1e-10


def test_boundary_layer_gate_is_default_off_identity_and_persistent_when_enabled():
    elevation = np.zeros((8, 16), dtype=np.float32)
    elevation[2:6, 4:12] = 0.5
    state = create_initial_state(elevation, planet_params=EARTH)
    baseline, _ = simulate_step(state, days=1.0, planet_params=EARTH)
    explicit_off, _ = simulate_step(
        state, days=1.0,
        planet_params=dataclasses.replace(EARTH, enable_force_restore_boundary_layer=False),
    )
    np.testing.assert_array_equal(baseline.temperature, explicit_off.temperature)
    np.testing.assert_array_equal(baseline.air_temperature, explicit_off.air_temperature)
    assert explicit_off.boundary_layer_temperature is None
    orphan_capacity_gate, _ = simulate_step(
        state,
        days=1.0,
        planet_params=dataclasses.replace(
            EARTH, enable_boundary_layer_capacity_aware_airsea_exchange=True
        ),
    )
    np.testing.assert_array_equal(baseline.temperature, orphan_capacity_gate.temperature)
    np.testing.assert_array_equal(
        baseline.air_temperature, orphan_capacity_gate.air_temperature
    )

    params = dataclasses.replace(
        EARTH, enable_force_restore_land=True,
        enable_force_restore_boundary_layer=True,
        enable_force_restore_atmospheric_heat_convergence=True,
    )
    enabled, diagnostics = simulate_step(
        state, days=1.0, planet_params=params, track_components=True
    )
    assert enabled.boundary_layer_temperature is not None
    assert enabled.boundary_layer_temperature.shape == elevation.shape
    assert np.all(np.isfinite(enabled.boundary_layer_temperature))
    assert diagnostics["boundary_layer_horizontal_transport"] == "omitted"
    assert diagnostics["resolved_heat_convergence_destination"] == "free_atmosphere"
    np.testing.assert_allclose(
        diagnostics["boundary_layer_exchange_gain_w_m2"],
        -diagnostics["free_air_exchange_gain_w_m2"],
    )

    disabled_again, _ = simulate_step(
        enabled, days=1.0, planet_params=EARTH, block_size=2
    )
    np.testing.assert_array_equal(
        disabled_again.boundary_layer_temperature,
        enabled.boundary_layer_temperature,
    )


def test_capacity_aware_airsea_exchange_closes_both_ocean_energy_operators():
    elevation = np.zeros((8, 16), dtype=np.float32)
    elevation[2:6, 4:12] = 0.5
    state = create_initial_state(elevation, planet_params=EARTH)
    params = dataclasses.replace(
        EARTH,
        enable_force_restore_land=True,
        enable_force_restore_boundary_layer=True,
        enable_boundary_layer_capacity_aware_airsea_exchange=True,
        enable_boundary_layer_near_surface_cloud_temperature=True,
    )
    _, diagnostics = simulate_step(
        state, days=1.0, planet_params=params, track_components=True
    )
    assert np.max(np.abs(
        diagnostics["ocean_air_relaxation_physical_energy_residual_w_m2"]
    )) < 2e-5
    assert np.max(np.abs(
        diagnostics["airsea_physical_energy_residual_w_m2"]
    )) < 2e-5
    assert diagnostics["cloud_temperature_source"] == "boundary_layer_over_land"


def test_interface_reservoir_is_initialized_and_persisted_only_when_enabled():
    elevation = np.zeros((8, 16), dtype=np.float32)
    elevation[2:6, 4:12] = 0.5
    state = create_initial_state(elevation, planet_params=EARTH)
    params = dataclasses.replace(
        EARTH,
        enable_force_restore_land=True,
        enable_force_restore_boundary_layer=True,
        enable_boundary_layer_interface_reservoir=True,
    )
    advanced, diagnostics = simulate_step(
        state, days=1.0, planet_params=params, track_components=True
    )
    assert advanced.boundary_layer_interface_temperature is not None
    assert advanced.boundary_layer_interface_temperature.shape == elevation.shape
    assert np.all(np.isfinite(advanced.boundary_layer_interface_temperature))
    closure = (
        diagnostics["boundary_layer_exchange_gain_w_m2"]
        + diagnostics["boundary_layer_interface_exchange_gain_w_m2"]
        + diagnostics["free_air_exchange_gain_w_m2"]
    )
    assert np.max(np.abs(closure)) < 2e-5
