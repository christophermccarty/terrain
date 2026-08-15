from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from boundary_layer import mixed_layer_pressure_thickness, step_boundary_layer_energy
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
