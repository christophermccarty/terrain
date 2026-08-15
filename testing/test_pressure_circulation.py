from __future__ import annotations

import dataclasses
import numpy as np
import pytest

from atmosphere import flux_divergence_spherical
from condensate import column_water_forcing_from_budget
from planet_params import EARTH
from pressure_circulation import (
    balanced_thermal_wind_u,
    column_mse_storage_tendency_w_m2,
    close_upper_mass_flux,
    converge_joint_pressure_column_coupling,
    diagnose_hydrostatic_sigma_continuity,
    diabatic_interface_mass_flux,
    diabatic_interface_mass_flux_from_heating,
    diagnose_joint_pressure_column_coupling_residual,
    evolve_large_scale_heating_reservoir,
    evolve_joint_mse_momentum_pressure_column,
    evolve_joint_mse_momentum_pressure_column_runtime,
    evolve_prognostic_pressure_layer_mass,
    evolve_prognostic_pressure_coordinate_transport,
    evolve_hydrostatic_sigma_pressure_coordinate_transport,
    evolve_hydrostatic_sigma_phase_reservoir_transport,
    evolve_hydrostatic_sigma_mass_momentum,
    evolve_variable_mass_pressure_momentum,
    evolve_variable_mass_pressure_coordinate_transport,
    evolve_simultaneous_joint_pressure_column_runtime,
    evolve_three_level_zonal_momentum,
    mse_constrained_pressure_coordinate_circulation,
    momentum_constrained_three_branch_mse_pressure_coordinate_circulation,
    water_constrained_three_branch_mse_pressure_coordinate_circulation,
    three_branch_mse_constrained_pressure_coordinate_circulation,
    shared_pressure_coordinate_circulation,
    smooth_spherical_scalar,
    spherical_divergence,
)
from simulate import create_initial_state, simulate_step


def _prognostic_transport_inputs(shape: tuple[int, int] = (8, 16)) -> dict[str, object]:
    h, w = shape
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lower_q = np.broadcast_to(0.012 + 0.001 * np.cos(np.radians(latitude))[:, None], shape).copy()
    return dict(
        lower_humidity=lower_q,
        midlevel_humidity=0.30 * lower_q,
        upperlevel_humidity=0.08 * lower_q,
        lower_temperature_k=np.broadcast_to(300.0 - 0.15 * np.abs(latitude)[:, None], shape).copy(),
        midlevel_temperature_k=np.broadcast_to(274.0 - 0.10 * np.abs(latitude)[:, None], shape).copy(),
        upperlevel_temperature_k=np.broadcast_to(244.0 - 0.06 * np.abs(latitude)[:, None], shape).copy(),
        lower_u_m_s=np.full(shape, 8.0), lower_v_m_s=np.full(shape, -1.0),
        midlevel_u_m_s=np.full(shape, 12.0), midlevel_v_m_s=np.full(shape, 1.5),
        upperlevel_u_m_s=np.full(shape, 20.0), upperlevel_v_m_s=np.full(shape, 2.0),
        lower_surface_vapour_source_kg_m2_s=np.zeros(shape),
        dt_seconds=3600.0, radius_m=6.371e6, sidereal_day_hours=24.0,
        surface_pressure_pa=101325.0, dx_m=100_000.0, dy_m=100_000.0,
        cell_area_m2=1.0e10, x_face_length_m=100_000.0, y_face_length_m=100_000.0,
    )


def test_prognostic_pressure_coordinate_transport_conserves_water_and_mse():
    inputs = _prognostic_transport_inputs()
    layer_mass = np.array((0.40, 0.35, 0.25)) * 101325.0 / 9.80665
    area = float(inputs["cell_area_m2"])
    before_water = sum(
        layer_mass[index] * np.sum(inputs[name]) * area
        for index, name in enumerate(("lower_humidity", "midlevel_humidity", "upperlevel_humidity"))
    )
    result = evolve_prognostic_pressure_coordinate_transport(**inputs)
    after_water = sum(
        layer_mass[index] * np.sum(value) * area
        for index, value in enumerate((result.lower_humidity, result.midlevel_humidity, result.upperlevel_humidity))
    )
    np.testing.assert_allclose(after_water, before_water, rtol=3e-6)
    assert abs(result.water_relative_residual) < 1e-12
    assert abs(result.moist_static_energy_relative_residual) < 1e-12
    assert np.all(np.isfinite(result.upperlevel_temperature))


def test_prognostic_pressure_coordinate_transport_accounts_for_surface_water_and_latent_energy():
    inputs = _prognostic_transport_inputs()
    source = np.full((8, 16), 2.0e-5)
    inputs.update(
        lower_surface_vapour_source_kg_m2_s=source,
        lower_u_m_s=np.zeros((8, 16)), lower_v_m_s=np.zeros((8, 16)),
        midlevel_u_m_s=np.zeros((8, 16)), midlevel_v_m_s=np.zeros((8, 16)),
        upperlevel_u_m_s=np.zeros((8, 16)), upperlevel_v_m_s=np.zeros((8, 16)),
        sidereal_day_hours=1.0e30,
    )
    result = evolve_prognostic_pressure_coordinate_transport(**inputs)
    lower_mass = 0.40 * 101325.0 / 9.80665
    expected_q_increment = source * float(inputs["dt_seconds"]) / lower_mass
    np.testing.assert_allclose(
        result.lower_humidity, inputs["lower_humidity"] + expected_q_increment, rtol=2e-6,
    )
    # Latent energy enters with the supplied vapour, so a uniform source does
    # not manufacture a temperature tendency in the absence of transport.
    np.testing.assert_allclose(result.lower_temperature, inputs["lower_temperature_k"], atol=3e-5)


def test_prognostic_pressure_coordinate_transport_substeps_its_horizontal_cfl():
    inputs = _prognostic_transport_inputs()
    inputs.update(
        lower_u_m_s=np.full((8, 16), 100.0), midlevel_u_m_s=np.full((8, 16), 100.0),
        upperlevel_u_m_s=np.full((8, 16), 100.0), dt_seconds=86400.0,
    )
    result = evolve_prognostic_pressure_coordinate_transport(**inputs)
    assert result.horizontal_substeps > 1
    assert result.horizontal_courant_max <= 0.5 + 1e-12
    assert np.all(result.lower_humidity >= 0.0)


def _prognostic_layer_mass_inputs(shape: tuple[int, int] = (8, 16)) -> dict[str, object]:
    zeros = np.zeros(shape)
    return dict(
        lower_pressure_depth_pa=np.full(shape, 40_000.0),
        midlevel_pressure_depth_pa=np.full(shape, 35_000.0),
        upperlevel_pressure_depth_pa=np.full(shape, 25_000.0),
        lower_u_m_s=zeros.copy(), lower_v_m_s=zeros.copy(),
        midlevel_u_m_s=zeros.copy(), midlevel_v_m_s=zeros.copy(),
        upperlevel_u_m_s=zeros.copy(), upperlevel_v_m_s=zeros.copy(),
        lower_mid_interface_mass_flux_kg_m2_s=zeros.copy(),
        mid_upper_interface_mass_flux_kg_m2_s=zeros.copy(),
        dt_seconds=3600.0, gravity_m_s2=9.80665,
        dx_m=100_000.0, dy_m=100_000.0, cell_area_m2=1.0e10,
        x_face_length_m=100_000.0, y_face_length_m=100_000.0,
    )


def test_prognostic_pressure_layer_mass_preserves_a_resting_state():
    inputs = _prognostic_layer_mass_inputs()
    result = evolve_prognostic_pressure_layer_mass(**inputs)
    for input_name, result_value in (
        ("lower_pressure_depth_pa", result.lower_pressure_depth_pa),
        ("midlevel_pressure_depth_pa", result.midlevel_pressure_depth_pa),
        ("upperlevel_pressure_depth_pa", result.upperlevel_pressure_depth_pa),
    ):
        np.testing.assert_allclose(result_value, inputs[input_name], atol=1e-4)
    assert abs(result.relative_mass_residual) < 1e-12
    assert result.vertical_substeps == 1


def test_prognostic_pressure_layer_mass_moves_interface_flux_conservatively():
    inputs = _prognostic_layer_mass_inputs()
    lower_mid_flux = np.full((8, 16), 1.0e-3)
    mid_upper_flux = np.full((8, 16), 2.0e-3)
    inputs.update(
        lower_mid_interface_mass_flux_kg_m2_s=lower_mid_flux,
        mid_upper_interface_mass_flux_kg_m2_s=mid_upper_flux,
    )
    result = evolve_prognostic_pressure_layer_mass(**inputs)
    g = float(inputs["gravity_m_s2"])
    dt = float(inputs["dt_seconds"])
    np.testing.assert_allclose(
        result.lower_pressure_depth_pa,
        inputs["lower_pressure_depth_pa"] - g * lower_mid_flux * dt, rtol=2e-6,
    )
    np.testing.assert_allclose(
        result.midlevel_pressure_depth_pa,
        inputs["midlevel_pressure_depth_pa"] + g * (lower_mid_flux - mid_upper_flux) * dt, rtol=2e-6,
    )
    np.testing.assert_allclose(
        result.upperlevel_pressure_depth_pa,
        inputs["upperlevel_pressure_depth_pa"] + g * mid_upper_flux * dt, rtol=2e-6,
    )
    pressure_total = (
        result.lower_pressure_depth_pa + result.midlevel_pressure_depth_pa + result.upperlevel_pressure_depth_pa
    )
    np.testing.assert_allclose(pressure_total, 100_000.0, rtol=2e-6)
    assert abs(result.relative_mass_residual) < 1e-12


def test_prognostic_pressure_layer_mass_substeps_combined_donor_courant_and_evolves_flux_state():
    inputs = _prognostic_layer_mass_inputs()
    inputs.update(
        lower_mid_interface_mass_flux_kg_m2_s=np.full((8, 16), -0.40),
        mid_upper_interface_mass_flux_kg_m2_s=np.full((8, 16), 0.40),
        lower_mid_interface_mass_flux_tendency_kg_m2_s2=np.full((8, 16), 1.0e-5),
        mid_upper_interface_mass_flux_tendency_kg_m2_s2=np.full((8, 16), -1.0e-5),
    )
    result = evolve_prognostic_pressure_layer_mass(**inputs)
    assert result.vertical_substeps > 1
    assert result.vertical_courant_max <= 0.25 + 1e-12
    np.testing.assert_allclose(
        result.lower_mid_interface_mass_flux_kg_m2_s,
        -0.40 + float(inputs["dt_seconds"]) * 1.0e-5, rtol=2e-6,
    )
    np.testing.assert_allclose(
        result.mid_upper_interface_mass_flux_kg_m2_s,
        0.40 - float(inputs["dt_seconds"]) * 1.0e-5, rtol=2e-6,
    )


def _variable_mass_transport_inputs(shape: tuple[int, int] = (8, 16)) -> dict[str, object]:
    mass = _prognostic_layer_mass_inputs(shape)
    transport = _prognostic_transport_inputs(shape)
    return dict(
        lower_pressure_depth_pa=mass["lower_pressure_depth_pa"],
        midlevel_pressure_depth_pa=mass["midlevel_pressure_depth_pa"],
        upperlevel_pressure_depth_pa=mass["upperlevel_pressure_depth_pa"],
        lower_humidity=transport["lower_humidity"], midlevel_humidity=transport["midlevel_humidity"],
        upperlevel_humidity=transport["upperlevel_humidity"],
        lower_temperature_k=transport["lower_temperature_k"], midlevel_temperature_k=transport["midlevel_temperature_k"],
        upperlevel_temperature_k=transport["upperlevel_temperature_k"],
        lower_u_m_s=transport["lower_u_m_s"], lower_v_m_s=transport["lower_v_m_s"],
        midlevel_u_m_s=transport["midlevel_u_m_s"], midlevel_v_m_s=transport["midlevel_v_m_s"],
        upperlevel_u_m_s=transport["upperlevel_u_m_s"], upperlevel_v_m_s=transport["upperlevel_v_m_s"],
        lower_mid_interface_mass_flux_kg_m2_s=mass["lower_mid_interface_mass_flux_kg_m2_s"],
        mid_upper_interface_mass_flux_kg_m2_s=mass["mid_upper_interface_mass_flux_kg_m2_s"],
        lower_surface_vapour_source_kg_m2_s=np.zeros(shape),
        dt_seconds=3600.0, gravity_m_s2=9.80665,
        dx_m=100_000.0, dy_m=100_000.0, cell_area_m2=1.0e10,
        x_face_length_m=100_000.0, y_face_length_m=100_000.0,
    )


def test_variable_mass_pressure_transport_closes_water_and_mse_across_two_interfaces():
    inputs = _variable_mass_transport_inputs()
    inputs.update(
        lower_mid_interface_mass_flux_kg_m2_s=np.full((8, 16), 1.0e-3),
        mid_upper_interface_mass_flux_kg_m2_s=np.full((8, 16), -1.5e-3),
    )
    result = evolve_variable_mass_pressure_coordinate_transport(**inputs)
    area = float(inputs["cell_area_m2"])
    g = float(inputs["gravity_m_s2"])
    before_water = sum(
        np.sum(inputs[p_name] / g * inputs[q_name]) * area
        for p_name, q_name in (
            ("lower_pressure_depth_pa", "lower_humidity"),
            ("midlevel_pressure_depth_pa", "midlevel_humidity"),
            ("upperlevel_pressure_depth_pa", "upperlevel_humidity"),
        )
    )
    after_water = sum(
        np.sum(pressure / g * humidity) * area
        for pressure, humidity in (
            (result.lower_pressure_depth_pa, result.lower_humidity),
            (result.midlevel_pressure_depth_pa, result.midlevel_humidity),
            (result.upperlevel_pressure_depth_pa, result.upperlevel_humidity),
        )
    )
    np.testing.assert_allclose(after_water, before_water, rtol=4e-6)
    assert abs(result.water_relative_residual) < 1e-12
    assert abs(result.moist_static_energy_relative_residual) < 1e-12
    assert np.all(result.lower_humidity >= 0.0)
    assert np.all(np.isfinite(result.upperlevel_temperature))


def test_variable_mass_pressure_transport_accounts_for_surface_water_and_latent_energy():
    inputs = _variable_mass_transport_inputs()
    shape = (8, 16)
    inputs.update(
        lower_surface_vapour_source_kg_m2_s=np.full(shape, 2.0e-5),
        lower_u_m_s=np.zeros(shape), lower_v_m_s=np.zeros(shape),
        midlevel_u_m_s=np.zeros(shape), midlevel_v_m_s=np.zeros(shape),
        upperlevel_u_m_s=np.zeros(shape), upperlevel_v_m_s=np.zeros(shape),
    )
    result = evolve_variable_mass_pressure_coordinate_transport(**inputs)
    increment = (
        inputs["lower_surface_vapour_source_kg_m2_s"] * float(inputs["dt_seconds"])
        / (inputs["lower_pressure_depth_pa"] / float(inputs["gravity_m_s2"]))
    )
    np.testing.assert_allclose(result.lower_humidity, inputs["lower_humidity"] + increment, rtol=2e-6)
    np.testing.assert_allclose(result.lower_temperature, inputs["lower_temperature_k"], atol=3e-5)


def test_hydrostatic_sigma_continuity_derives_unique_interface_fluxes_from_mass_transport():
    inputs = _prognostic_layer_mass_inputs()
    longitude = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    inputs.update(
        lower_u_m_s=np.broadcast_to(12.0 * np.sin(longitude)[None, :], (8, 16)).copy(),
        midlevel_u_m_s=np.zeros((8, 16)), upperlevel_u_m_s=np.zeros((8, 16)),
    )
    result = diagnose_hydrostatic_sigma_continuity(
        inputs["lower_pressure_depth_pa"], inputs["midlevel_pressure_depth_pa"], inputs["upperlevel_pressure_depth_pa"],
        inputs["lower_u_m_s"], inputs["lower_v_m_s"], inputs["midlevel_u_m_s"], inputs["midlevel_v_m_s"],
        inputs["upperlevel_u_m_s"], inputs["upperlevel_v_m_s"],
        dt_seconds=inputs["dt_seconds"], gravity_m_s2=inputs["gravity_m_s2"],
        dx_m=inputs["dx_m"], dy_m=inputs["dy_m"], cell_area_m2=inputs["cell_area_m2"],
        x_face_length_m=inputs["x_face_length_m"], y_face_length_m=inputs["y_face_length_m"],
    )
    total = result.surface_pressure_tendency_pa_s / float(inputs["gravity_m_s2"])
    np.testing.assert_allclose(result.lower_pressure_mass_tendency_kg_m2_s, 0.40 * total, rtol=2e-6)
    np.testing.assert_allclose(result.midlevel_pressure_mass_tendency_kg_m2_s, 0.35 * total, rtol=2e-6)
    np.testing.assert_allclose(result.upperlevel_pressure_mass_tendency_kg_m2_s, 0.25 * total, rtol=2e-6)
    # If only the lower layer has horizontal convergence H, the two upward
    # interface fluxes must be 0.60 H and 0.25 H to retain sigma fractions.
    np.testing.assert_allclose(
        result.lower_mid_interface_mass_flux_kg_m2_s,
        0.60 * total, rtol=2e-5, atol=3e-8,
    )
    np.testing.assert_allclose(
        result.mid_upper_interface_mass_flux_kg_m2_s,
        0.25 * total, rtol=2e-5, atol=3e-8,
    )
    assert result.relative_continuity_residual < 1e-6


def test_hydrostatic_sigma_transport_keeps_pressure_partition_and_closes_tracers():
    inputs = _variable_mass_transport_inputs()
    longitude = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    inputs.update(
        lower_u_m_s=np.broadcast_to(12.0 * np.sin(longitude)[None, :], (8, 16)).copy(),
        midlevel_u_m_s=np.zeros((8, 16)), upperlevel_u_m_s=np.zeros((8, 16)),
    )
    result = evolve_hydrostatic_sigma_pressure_coordinate_transport(
        inputs["lower_pressure_depth_pa"], inputs["midlevel_pressure_depth_pa"], inputs["upperlevel_pressure_depth_pa"],
        inputs["lower_humidity"], inputs["midlevel_humidity"], inputs["upperlevel_humidity"],
        inputs["lower_temperature_k"], inputs["midlevel_temperature_k"], inputs["upperlevel_temperature_k"],
        inputs["lower_u_m_s"], inputs["lower_v_m_s"], inputs["midlevel_u_m_s"], inputs["midlevel_v_m_s"],
        inputs["upperlevel_u_m_s"], inputs["upperlevel_v_m_s"],
        lower_surface_vapour_source_kg_m2_s=inputs["lower_surface_vapour_source_kg_m2_s"],
        dt_seconds=inputs["dt_seconds"], gravity_m_s2=inputs["gravity_m_s2"],
        dx_m=inputs["dx_m"], dy_m=inputs["dy_m"], cell_area_m2=inputs["cell_area_m2"],
        x_face_length_m=inputs["x_face_length_m"], y_face_length_m=inputs["y_face_length_m"],
    )
    state = result.transport
    total_pressure = state.lower_pressure_depth_pa + state.midlevel_pressure_depth_pa + state.upperlevel_pressure_depth_pa
    np.testing.assert_allclose(state.lower_pressure_depth_pa, 0.40 * total_pressure, rtol=3e-6, atol=3e-3)
    np.testing.assert_allclose(state.midlevel_pressure_depth_pa, 0.35 * total_pressure, rtol=3e-6, atol=3e-3)
    np.testing.assert_allclose(state.upperlevel_pressure_depth_pa, 0.25 * total_pressure, rtol=3e-6, atol=3e-3)
    assert result.continuity.relative_continuity_residual < 1e-6
    assert abs(state.water_relative_residual) < 1e-12
    assert abs(state.moist_static_energy_relative_residual) < 1e-12


def test_hydrostatic_sigma_mass_momentum_moves_the_carrier_together():
    inputs = _prognostic_layer_mass_inputs()
    shape = (8, 16)
    longitude = np.linspace(0.0, 2.0 * np.pi, shape[1], endpoint=False)
    lower_u = np.broadcast_to(18.0 * np.sin(longitude)[None, :], shape).copy()
    result = evolve_hydrostatic_sigma_mass_momentum(
        inputs["lower_pressure_depth_pa"], inputs["midlevel_pressure_depth_pa"], inputs["upperlevel_pressure_depth_pa"],
        np.full(shape, 280.0), np.full(shape, 270.0), np.full(shape, 250.0),
        lower_u, np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape),
        dt_seconds=30.0 * 86400.0, radius_m=6.371e6, sidereal_day_hours=1e30,
        gravity_m_s2=inputs["gravity_m_s2"], dx_m=inputs["dx_m"], dy_m=inputs["dy_m"],
        cell_area_m2=inputs["cell_area_m2"], x_face_length_m=inputs["x_face_length_m"],
        y_face_length_m=inputs["y_face_length_m"],
        lower_humidity=np.full(shape, 0.014), midlevel_humidity=np.full(shape, 0.006), upperlevel_humidity=np.full(shape, 0.002),
        lower_surface_vapour_source_kg_m2_s=np.full(shape, 2.0e-5),
        lower_cloud_condensate_kg_m2=np.full(shape, 0.2), midlevel_cloud_condensate_kg_m2=np.full(shape, 0.1), upperlevel_cloud_condensate_kg_m2=np.full(shape, 0.05),
        lower_hydrometeors_kg_m2=np.full(shape, 0.03), midlevel_hydrometeors_kg_m2=np.full(shape, 0.02), upperlevel_hydrometeors_kg_m2=np.full(shape, 0.01),
    )
    total_pressure = result.lower_pressure_depth_pa + result.midlevel_pressure_depth_pa + result.upperlevel_pressure_depth_pa
    np.testing.assert_allclose(result.lower_pressure_depth_pa, 0.40 * total_pressure, rtol=3e-6, atol=3e-3)
    np.testing.assert_allclose(result.midlevel_pressure_depth_pa, 0.35 * total_pressure, rtol=3e-6, atol=3e-3)
    np.testing.assert_allclose(result.upperlevel_pressure_depth_pa, 0.25 * total_pressure, rtol=3e-6, atol=3e-3)
    for value in (
        result.lower_pressure_depth_pa, result.midlevel_pressure_depth_pa, result.upperlevel_pressure_depth_pa,
        result.lower_u, result.lower_v, result.midlevel_u, result.midlevel_v, result.upperlevel_u, result.upperlevel_v,
        result.lower_humidity, result.midlevel_humidity, result.upperlevel_humidity,
        result.lower_temperature, result.midlevel_temperature, result.upperlevel_temperature,
        result.lower_cloud_condensate_kg_m2, result.midlevel_cloud_condensate_kg_m2, result.upperlevel_cloud_condensate_kg_m2,
        result.lower_hydrometeors_kg_m2, result.midlevel_hydrometeors_kg_m2, result.upperlevel_hydrometeors_kg_m2,
    ):
        assert np.all(np.isfinite(value))
    assert np.all(result.lower_pressure_depth_pa > 0.0)
    assert result.substeps > 1
    assert result.horizontal_courant_max <= 0.9 + 1e-12
    assert result.vertical_courant_max <= 0.25 + 1e-12
    assert abs(result.relative_mass_residual) < 1e-12
    assert result.horizontal_momentum_relative_residual < 1e-10
    assert abs(result.water_relative_residual) < 1e-12
    assert abs(result.moist_static_energy_relative_residual) < 1e-12
    heat = result.horizontal_mse_convergence_w_m2
    assert heat is not None
    area = np.broadcast_to(np.asarray(inputs["cell_area_m2"], dtype=np.float64), shape)
    assert abs(float(np.sum(heat * area)) / float(np.sum(area))) < 1e-4


def test_hydrostatic_sigma_mass_momentum_rejects_nonhydrostatic_carrier_speed():
    inputs = _prognostic_layer_mass_inputs()
    shape = (8, 16)
    with pytest.raises(RuntimeError, match="hydrostatic gravity-wave speed"):
        evolve_hydrostatic_sigma_mass_momentum(
            inputs["lower_pressure_depth_pa"], inputs["midlevel_pressure_depth_pa"], inputs["upperlevel_pressure_depth_pa"],
            np.full(shape, 280.0), np.full(shape, 270.0), np.full(shape, 250.0),
            np.full(shape, 400.0), np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape),
            dt_seconds=3600.0, radius_m=6.371e6, sidereal_day_hours=24.0,
            gravity_m_s2=inputs["gravity_m_s2"], dx_m=inputs["dx_m"], dy_m=inputs["dy_m"],
            cell_area_m2=inputs["cell_area_m2"], x_face_length_m=inputs["x_face_length_m"],
            y_face_length_m=inputs["y_face_length_m"],
        )


def test_hydrostatic_sigma_phase_transition_closes_water_with_layer_reservoirs():
    inputs = _variable_mass_transport_inputs()
    shape = (8, 16)
    inputs.update(
        lower_humidity=np.full(shape, 0.030), midlevel_humidity=np.full(shape, 0.020), upperlevel_humidity=np.full(shape, 0.012),
        lower_u_m_s=np.zeros(shape), lower_v_m_s=np.zeros(shape), midlevel_u_m_s=np.zeros(shape), midlevel_v_m_s=np.zeros(shape), upperlevel_u_m_s=np.zeros(shape), upperlevel_v_m_s=np.zeros(shape),
    )
    zeros = np.zeros(shape)
    result = evolve_hydrostatic_sigma_phase_reservoir_transport(
        inputs["lower_pressure_depth_pa"], inputs["midlevel_pressure_depth_pa"], inputs["upperlevel_pressure_depth_pa"],
        inputs["lower_humidity"], inputs["midlevel_humidity"], inputs["upperlevel_humidity"],
        inputs["lower_temperature_k"], inputs["midlevel_temperature_k"], inputs["upperlevel_temperature_k"],
        inputs["lower_u_m_s"], inputs["lower_v_m_s"], inputs["midlevel_u_m_s"], inputs["midlevel_v_m_s"], inputs["upperlevel_u_m_s"], inputs["upperlevel_v_m_s"],
        zeros, zeros, zeros, zeros, zeros, zeros,
        lower_surface_vapour_source_kg_m2_s=zeros, dt_seconds=inputs["dt_seconds"], gravity_m_s2=inputs["gravity_m_s2"],
        radius_m=6.371e6, sidereal_day_hours=24.0,
        critical_relative_humidity=0.8, autoconversion_timescale_days=0.5, fallout_timescale_days=1.0, cloud_retention_kg_m2=1.0,
        dx_m=inputs["dx_m"], dy_m=inputs["dy_m"], cell_area_m2=inputs["cell_area_m2"], x_face_length_m=inputs["x_face_length_m"], y_face_length_m=inputs["y_face_length_m"],
    )
    state = result.transport.transport
    g = float(inputs["gravity_m_s2"])
    before = sum(np.sum(inputs[p] / g * inputs[q]) for p, q in (("lower_pressure_depth_pa", "lower_humidity"), ("midlevel_pressure_depth_pa", "midlevel_humidity"), ("upperlevel_pressure_depth_pa", "upperlevel_humidity")))
    after = sum(pressure / g * humidity for pressure, humidity in ((state.lower_pressure_depth_pa, state.lower_humidity), (state.midlevel_pressure_depth_pa, state.midlevel_humidity), (state.upperlevel_pressure_depth_pa, state.upperlevel_humidity)))
    after = np.sum(after + result.lower_cloud_condensate_kg_m2 + result.midlevel_cloud_condensate_kg_m2 + result.upperlevel_cloud_condensate_kg_m2 + result.lower_hydrometeors_kg_m2 + result.midlevel_hydrometeors_kg_m2 + result.upperlevel_hydrometeors_kg_m2 + result.fallout_kg_m2)
    np.testing.assert_allclose(after, before, rtol=4e-6)
    assert np.any(result.fallout_kg_m2 > 0.0)
    assert np.all(np.isfinite(state.upperlevel_temperature))
    assert result.momentum.vertical_courant_max <= 0.25 + 1e-12


def test_variable_mass_pressure_momentum_conserves_interface_momentum():
    inputs = _prognostic_layer_mass_inputs()
    shape = (8, 16)
    lower_u, mid_u, upper_u = np.full(shape, 2.0), np.full(shape, 8.0), np.full(shape, -4.0)
    lower_v, mid_v, upper_v = np.full(shape, 1.0), np.full(shape, -2.0), np.full(shape, 5.0)
    step = evolve_variable_mass_pressure_momentum(
        inputs["lower_pressure_depth_pa"], inputs["midlevel_pressure_depth_pa"], inputs["upperlevel_pressure_depth_pa"],
        lower_u, lower_v, mid_u, mid_v, upper_u, upper_v,
        np.full(shape, 280.0), np.full(shape, 280.0), np.full(shape, 280.0),
        np.full(shape, -0.03), np.full(shape, 0.02),
        dt_seconds=3600.0, radius_m=6.371e6, sidereal_day_hours=1e30,
    )
    g = 9.80665
    before_u = 40_000 / g * lower_u + 35_000 / g * mid_u + 25_000 / g * upper_u
    before_v = 40_000 / g * lower_v + 35_000 / g * mid_v + 25_000 / g * upper_v
    # Interface transfer changes layer masses; use the known flux update to
    # weight the post-transition velocity inventory.
    lm, mu, dt = -0.03, 0.02, 3600.0
    m0, m1, m2 = 40_000 / g - lm * dt, 35_000 / g + lm * dt - mu * dt, 25_000 / g + mu * dt
    after_u = m0 * step.lower_u + m1 * step.midlevel_u + m2 * step.upperlevel_u
    after_v = m0 * step.lower_v + m1 * step.midlevel_v + m2 * step.upperlevel_v
    np.testing.assert_allclose(after_u, before_u, rtol=3e-6)
    np.testing.assert_allclose(after_v, before_v, rtol=3e-6)
    assert step.vertical_courant_max <= 0.25 + 1e-12
    assert step.horizontal_momentum_relative_residual < 1e-6


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


def test_mse_constrained_circulation_closes_mass_and_exports_balanced_heating():
    h, w = 24, 48
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    heating = np.broadcast_to(
        100.0 * np.exp(-(latitude[:, None] / 18.0) ** 2), (h, w)
    )
    lower_t = np.full((h, w), 300.0)
    middle_t = np.full((h, w), 272.0)
    upper_t = np.full((h, w), 240.0)
    result = mse_constrained_pressure_coordinate_circulation(
        heating, lower_t, middle_t, upper_t,
        np.full((h, w), 0.014), np.full((h, w), 0.004), np.full((h, w), 0.001),
        np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w)),
        dt_seconds=86400.0, radius_m=6.371e6, surface_pressure_pa=101325.0,
        lower_mid_pressure_depth_pa=35000.0, mid_upper_pressure_depth_pa=30000.0,
    )

    weights = np.cos(np.radians(latitude))[:, None]
    assert abs(float(np.sum(result.diabatic_forcing_w_m2 * weights))) < 5e-4
    assert abs(result.energy_closure_residual_w) < 1e4
    assert np.all(np.abs(result.lower_upper_mse_contrast_j_kg) > 0.0)
    assert float(np.mean(result.meridional_mse_transport_w_m[: h // 2])) > 0.0
    assert float(np.mean(result.meridional_mse_transport_w_m[h // 2 :])) < 0.0
    circulation = result.circulation
    weighted_divergence = (
        0.40 * circulation.interface_mass_flux.lower_divergence_s
        + 0.35 * circulation.interface_mass_flux.midlevel_divergence_s
        + 0.25 * circulation.interface_mass_flux.upperlevel_divergence_s
    )
    np.testing.assert_allclose(weighted_divergence, 0.0, atol=2e-12)
    assert np.all(np.isfinite(circulation.interface_mass_flux.omega_lower_mid_pa_s))
    assert np.all(np.isfinite(circulation.interface_mass_flux.omega_mid_upper_pa_s))


def test_mse_constrained_circulation_has_zero_response_to_uniform_forcing():
    shape = (12, 24)
    result = mse_constrained_pressure_coordinate_circulation(
        np.full(shape, 50.0), np.full(shape, 300.0), np.full(shape, 270.0), np.full(shape, 240.0),
        np.full(shape, 0.012), np.full(shape, 0.004), np.full(shape, 0.001),
        np.zeros(shape), np.zeros(shape), np.zeros(shape),
        dt_seconds=86400.0, radius_m=6.371e6, surface_pressure_pa=101325.0,
        lower_mid_pressure_depth_pa=35000.0, mid_upper_pressure_depth_pa=30000.0,
    )

    np.testing.assert_allclose(result.diabatic_forcing_w_m2, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.meridional_mse_transport_w_m, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.circulation.lower_v, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.circulation.upperlevel_v, 0.0, atol=1e-12)


def test_mse_constrained_circulation_rejects_vanishing_branch_mse_contrast():
    shape = (8, 16)
    upper_temperature = (1004.0 * 300.0 - 9.80665 * 8000.0) / 1004.0
    with np.testing.assert_raises_regex(ValueError, "MSE contrast"):
        mse_constrained_pressure_coordinate_circulation(
            np.ones(shape), np.full(shape, 300.0), np.full(shape, 260.0), np.full(shape, upper_temperature),
            np.zeros(shape), np.zeros(shape), np.zeros(shape),
            np.zeros(shape), np.zeros(shape), np.zeros(shape),
            dt_seconds=86400.0, radius_m=6.371e6, surface_pressure_pa=101325.0,
            lower_mid_pressure_depth_pa=35000.0, mid_upper_pressure_depth_pa=30000.0,
        )


def test_three_branch_mse_circulation_closes_layer_energy_with_both_interfaces():
    h, w = 24, 48
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    heating = np.broadcast_to(
        90.0 * np.exp(-(latitude[:, None] / 18.0) ** 2), (h, w)
    )
    result = three_branch_mse_constrained_pressure_coordinate_circulation(
        heating, np.full((h, w), 300.0), np.full((h, w), 272.0), np.full((h, w), 240.0),
        np.full((h, w), 0.014), np.full((h, w), 0.004), np.full((h, w), 0.001),
        np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w)),
        dt_seconds=86400.0, radius_m=6.371e6, surface_pressure_pa=101325.0,
        lower_mid_pressure_depth_pa=35000.0, mid_upper_pressure_depth_pa=30000.0,
    )

    circulation = result.circulation
    weighted_divergence = (
        0.40 * circulation.interface_mass_flux.lower_divergence_s
        + 0.35 * circulation.interface_mass_flux.midlevel_divergence_s
        + 0.25 * circulation.interface_mass_flux.upperlevel_divergence_s
    )
    np.testing.assert_allclose(weighted_divergence, 0.0, atol=2e-12)
    np.testing.assert_allclose(
            result.lower_diabatic_deposition_w_m2
            + result.midlevel_diabatic_deposition_w_m2
            + result.upperlevel_diabatic_deposition_w_m2,
        result.diabatic_forcing_w_m2, atol=5e-5,
    )
    assert np.any(np.abs(circulation.midlevel_v) > 0.0)
    assert np.any(np.abs(result.midlevel_diabatic_deposition_w_m2) > 0.0)
    assert np.all(result.mse_variance_j2_kg2 > 0.0)


def test_three_branch_mse_circulation_rejects_uniform_mse_profile():
    shape = (8, 16)
    with np.testing.assert_raises_regex(ValueError, "MSE variance"):
        three_branch_mse_constrained_pressure_coordinate_circulation(
            np.ones(shape), np.full(shape, 280.0), np.full(shape, 280.0), np.full(shape, 280.0),
            np.zeros(shape), np.zeros(shape), np.zeros(shape),
            np.zeros(shape), np.zeros(shape), np.zeros(shape),
            dt_seconds=86400.0, radius_m=6.371e6, surface_pressure_pa=101325.0,
            lower_mid_pressure_depth_pa=35000.0, mid_upper_pressure_depth_pa=30000.0,
            layer_heights_m=(0.0, 0.0, 0.0),
        )


def test_three_branch_momentum_constraint_closes_mass_mse_and_zonal_momentum():
    h, w = 24, 48
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    heating = np.broadcast_to(
        75.0 * np.exp(-(latitude[:, None] / 18.0) ** 2), (h, w)
    )
    lower_u = np.full((h, w), -6.0)
    middle_u = np.full((h, w), 2.0)
    upper_u = np.full((h, w), 14.0)
    result = momentum_constrained_three_branch_mse_pressure_coordinate_circulation(
        heating, np.full((h, w), 300.0), np.full((h, w), 272.0), np.full((h, w), 240.0),
        np.full((h, w), 0.014), np.full((h, w), 0.004), np.full((h, w), 0.001),
        lower_u, middle_u, upper_u,
        dt_seconds=86400.0, radius_m=6.371e6, surface_pressure_pa=101325.0,
        lower_mid_pressure_depth_pa=35000.0, mid_upper_pressure_depth_pa=30000.0,
    )

    circulation = result.circulation
    mass_flux = 0.40 * circulation.lower_v + 0.35 * circulation.midlevel_v + 0.25 * circulation.upperlevel_v
    zonal_momentum_flux = (
        0.40 * lower_u * circulation.lower_v
        + 0.35 * middle_u * circulation.midlevel_v
        + 0.25 * upper_u * circulation.upperlevel_v
    )
    np.testing.assert_allclose(mass_flux, 0.0, atol=1e-7)
    np.testing.assert_allclose(zonal_momentum_flux, 0.0, atol=2e-6)
    discrete_mass_divergence = (
        0.40 * spherical_divergence(circulation.lower_u, circulation.lower_v, 6.371e6)
        + 0.35 * spherical_divergence(circulation.midlevel_u, circulation.midlevel_v, 6.371e6)
        + 0.25 * spherical_divergence(circulation.upperlevel_u, circulation.upperlevel_v, 6.371e6)
    )
    np.testing.assert_allclose(discrete_mass_divergence, 0.0, atol=5e-13)
    np.testing.assert_allclose(
        result.lower_diabatic_deposition_w_m2
        + result.midlevel_diabatic_deposition_w_m2
        + result.upperlevel_diabatic_deposition_w_m2,
        result.diabatic_forcing_w_m2, atol=5e-5,
    )


def test_three_branch_momentum_constraint_rejects_barotropic_winds():
    shape = (8, 16)
    with np.testing.assert_raises_regex(ValueError, "zonal-wind variance"):
        momentum_constrained_three_branch_mse_pressure_coordinate_circulation(
            np.broadcast_to(np.arange(shape[0])[:, None], shape),
            np.full(shape, 300.0), np.full(shape, 272.0), np.full(shape, 240.0),
            np.full(shape, 0.014), np.full(shape, 0.004), np.full(shape, 0.001),
            np.full(shape, 5.0), np.full(shape, 5.0), np.full(shape, 5.0),
            dt_seconds=86400.0, radius_m=6.371e6, surface_pressure_pa=101325.0,
            lower_mid_pressure_depth_pa=35000.0, mid_upper_pressure_depth_pa=30000.0,
        )


def test_three_branch_water_constraint_closes_mass_with_independent_vapour_budget():
    h, w = 16, 32
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    heating = np.broadcast_to(40.0 * np.exp(-(latitude[:, None] / 18.0) ** 2), (h, w))
    water_forcing = np.broadcast_to(
        2.0e-5 * np.exp(-(latitude[:, None] / 18.0) ** 2), (h, w)
    )
    result = water_constrained_three_branch_mse_pressure_coordinate_circulation(
        heating, water_forcing,
        np.full((h, w), 300.0), np.full((h, w), 272.0), np.full((h, w), 240.0),
        np.full((h, w), 0.014), np.full((h, w), 0.004), np.full((h, w), 0.001),
        np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w)),
        dt_seconds=86400.0, radius_m=6.371e6, surface_pressure_pa=101325.0,
        lower_mid_pressure_depth_pa=35000.0, mid_upper_pressure_depth_pa=30000.0,
    )
    circulation = result.circulation
    mass_divergence = (
        0.40 * spherical_divergence(circulation.lower_u, circulation.lower_v, 6.371e6)
        + 0.35 * spherical_divergence(circulation.midlevel_u, circulation.midlevel_v, 6.371e6)
        + 0.25 * spherical_divergence(circulation.upperlevel_u, circulation.upperlevel_v, 6.371e6)
    )
    np.testing.assert_allclose(mass_divergence, 0.0, atol=5e-13)
    assert np.any(np.abs(circulation.midlevel_v) > 0.0)
    assert np.all(np.isfinite(circulation.interface_mass_flux.omega_mid_upper_pa_s))


def test_three_branch_water_constraint_partitions_nonzero_mse_with_zero_water_flux():
    """A zero vapour transport must not make the finite-volume MSE face singular."""
    h, w = 16, 32
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    heating = np.broadcast_to(40.0 * np.exp(-(latitude[:, None] / 18.0) ** 2), (h, w))
    result = water_constrained_three_branch_mse_pressure_coordinate_circulation(
        heating, np.zeros((h, w)),
        np.full((h, w), 300.0), np.full((h, w), 272.0), np.full((h, w), 240.0),
        np.full((h, w), 0.014), np.full((h, w), 0.004), np.full((h, w), 0.001),
        np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w)),
        dt_seconds=86400.0, radius_m=6.371e6, surface_pressure_pa=101325.0,
        lower_mid_pressure_depth_pa=35000.0, mid_upper_pressure_depth_pa=30000.0,
    )
    np.testing.assert_allclose(
        result.lower_diabatic_deposition_w_m2
        + result.midlevel_diabatic_deposition_w_m2
        + result.upperlevel_diabatic_deposition_w_m2,
        result.diabatic_forcing_w_m2, atol=5e-5,
    )
    assert np.all(np.isfinite(result.circulation.midlevel_v))


def test_three_branch_water_constraint_rejects_rank_deficient_active_transport():
    """No arbitrary branch may be selected when MSE and water are dependent."""
    h, w = 12, 24
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    heating = np.broadcast_to(40.0 * np.exp(-(latitude[:, None] / 18.0) ** 2), (h, w))
    water_forcing = np.broadcast_to(
        2.0e-5 * np.exp(-(latitude[:, None] / 18.0) ** 2), (h, w)
    )
    with np.testing.assert_raises_regex(ValueError, "linearly dependent"):
        water_constrained_three_branch_mse_pressure_coordinate_circulation(
            heating, water_forcing,
            np.full((h, w), 280.0), np.full((h, w), 280.0), np.full((h, w), 280.0),
            np.full((h, w), 0.014), np.full((h, w), 0.004), np.full((h, w), 0.001),
            np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w)),
            dt_seconds=86400.0, radius_m=6.371e6, surface_pressure_pa=101325.0,
            lower_mid_pressure_depth_pa=35000.0, mid_upper_pressure_depth_pa=30000.0,
            layer_heights_m=(0.0, 0.0, 0.0),
        )


def test_three_level_mse_storage_tendency_is_zero_for_an_unchanged_column():
    h, w = 6, 12
    q0 = np.full((h, w), 0.012)
    q1 = np.full((h, w), 0.004)
    q2 = np.full((h, w), 0.001)
    t0 = np.full((h, w), 300.0)
    t1 = np.full((h, w), 272.0)
    t2 = np.full((h, w), 240.0)
    storage = column_mse_storage_tendency_w_m2(
        q0, q1, q2, t0, t1, t2, q0, q1, q2, t0, t1, t2,
        dt_seconds=86400.0, surface_pressure_pa=101325.0,
    )
    np.testing.assert_allclose(storage, 0.0, atol=1e-12)


def test_three_level_zonal_momentum_is_exactly_split_stable_for_fixed_pgf():
    h, w = 12, 24
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    temperature = np.broadcast_to(280.0 - 0.25 * latitude[:, None], (h, w))
    initial = dict(
        lower_u_m_s=np.full((h, w), 4.0), lower_v_m_s=np.full((h, w), -1.0),
        midlevel_u_m_s=np.full((h, w), 7.0), midlevel_v_m_s=np.full((h, w), 2.0),
        upperlevel_u_m_s=np.full((h, w), 12.0), upperlevel_v_m_s=np.full((h, w), -3.0),
        lower_temperature_k=temperature, midlevel_temperature_k=temperature - 24.0,
        upperlevel_temperature_k=temperature - 50.0,
        omega_lower_mid_pa_s=np.zeros((h, w)), omega_mid_upper_pa_s=np.zeros((h, w)),
        radius_m=6.371e6, sidereal_day_hours=24.0, surface_pressure_pa=101325.0,
    )
    full = evolve_three_level_zonal_momentum(**initial, dt_seconds=3600.0)
    first = evolve_three_level_zonal_momentum(**initial, dt_seconds=1800.0)
    second = evolve_three_level_zonal_momentum(
        first.lower_u, first.lower_v, first.midlevel_u, first.midlevel_v,
        first.upperlevel_u, first.upperlevel_v, temperature, temperature - 24.0,
        temperature - 50.0, np.zeros((h, w)), np.zeros((h, w)),
        dt_seconds=1800.0, radius_m=6.371e6, sidereal_day_hours=24.0,
        surface_pressure_pa=101325.0,
    )
    for name in ("lower_u", "lower_v", "midlevel_u", "midlevel_v", "upperlevel_u", "upperlevel_v"):
        np.testing.assert_allclose(getattr(second, name), getattr(full, name), atol=3e-5)


def test_three_level_zonal_momentum_conserves_momentum_under_vertical_exchange():
    shape = (8, 16)
    lower_u = np.full(shape, 2.0)
    mid_u = np.full(shape, 8.0)
    upper_u = np.full(shape, -4.0)
    lower_v = np.full(shape, 1.0)
    mid_v = np.full(shape, -2.0)
    upper_v = np.full(shape, 5.0)
    omega_lm = np.full(shape, -0.03)
    omega_mu = np.full(shape, 0.02)
    step = evolve_three_level_zonal_momentum(
        lower_u, lower_v, mid_u, mid_v, upper_u, upper_v,
        np.full(shape, 280.0), np.full(shape, 280.0), np.full(shape, 280.0),
        omega_lm, omega_mu, dt_seconds=3600.0, radius_m=6.371e6,
        sidereal_day_hours=1e30, surface_pressure_pa=101325.0,
    )
    for before, after in (
        (0.40 * lower_u + 0.35 * mid_u + 0.25 * upper_u,
         0.40 * step.lower_u + 0.35 * step.midlevel_u + 0.25 * step.upperlevel_u),
        (0.40 * lower_v + 0.35 * mid_v + 0.25 * upper_v,
         0.40 * step.lower_v + 0.35 * step.midlevel_v + 0.25 * step.upperlevel_v),
    ):
        np.testing.assert_allclose(after, before, atol=2e-6)
    assert step.vertical_courant_max > 0.0


def test_joint_pressure_column_uses_one_cfl_policy_and_closes_vertical_budgets():
    h, w = 8, 16
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    heating = np.broadcast_to(
        80.0 * np.exp(-(latitude[:, None] / 18.0) ** 2), (h, w)
    )
    result = evolve_joint_mse_momentum_pressure_column(
        heating,
        np.full((h, w), 0.014), np.full((h, w), 0.004), np.full((h, w), 0.001),
        np.full((h, w), 300.0), np.full((h, w), 272.0), np.full((h, w), 240.0),
        np.full((h, w), -6.0), np.full((h, w), 2.0),
        np.full((h, w), 2.0), np.full((h, w), -1.0),
        np.full((h, w), 14.0), np.full((h, w), 3.0),
        dt_seconds=86400.0, radius_m=6.371e6, sidereal_day_hours=24.0,
        surface_pressure_pa=101325.0, lower_mid_pressure_depth_pa=35000.0,
        mid_upper_pressure_depth_pa=30000.0,
    )
    assert result.substeps >= 1
    assert result.vertical_courant_max <= 0.25 + 1e-10
    assert abs(result.water_residual_kg_m2) < 1e-7
    assert abs(result.moist_static_energy_residual_j_m2) < 1e-3
    assert np.all(np.isfinite(result.lower_u))
    assert np.all(np.isfinite(result.upperlevel_temperature))


def test_water_constrained_joint_pressure_column_uses_the_same_cfl_policy():
    """The independent water constraint must not reopen a split timestep path."""
    h, w = 8, 16
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    heating = np.broadcast_to(
        40.0 * np.exp(-(latitude[:, None] / 18.0) ** 2), (h, w)
    )
    water_forcing = np.broadcast_to(
        2.0e-5 * np.exp(-(latitude[:, None] / 18.0) ** 2), (h, w)
    )
    result = evolve_joint_mse_momentum_pressure_column(
        heating,
        np.full((h, w), 0.014), np.full((h, w), 0.004), np.full((h, w), 0.001),
        np.full((h, w), 300.0), np.full((h, w), 272.0), np.full((h, w), 240.0),
        np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w)),
        np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w)),
        dt_seconds=86400.0, radius_m=6.371e6, sidereal_day_hours=24.0,
        surface_pressure_pa=101325.0, lower_mid_pressure_depth_pa=35000.0,
        mid_upper_pressure_depth_pa=30000.0,
        column_water_forcing_kg_m2_s=water_forcing,
    )
    assert result.substeps >= 1
    assert result.vertical_courant_max <= 0.25 + 1e-10
    assert abs(result.water_residual_kg_m2) < 1e-7
    assert abs(result.moist_static_energy_residual_j_m2) < 1e-3
    assert np.all(np.isfinite(result.midlevel_v))


def test_joint_pressure_column_is_identity_for_uniform_zero_state():
    shape = (6, 12)
    result = evolve_joint_mse_momentum_pressure_column(
        np.zeros(shape),
        np.full(shape, 0.010), np.full(shape, 0.004), np.full(shape, 0.001),
        np.full(shape, 280.0), np.full(shape, 280.0), np.full(shape, 280.0),
        np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape),
        dt_seconds=3600.0, radius_m=6.371e6, sidereal_day_hours=24.0,
        surface_pressure_pa=101325.0, lower_mid_pressure_depth_pa=35000.0,
        mid_upper_pressure_depth_pa=30000.0,
        layer_heights_m=(0.0, 0.0, 0.0),
    )
    assert result.substeps == 1
    np.testing.assert_allclose(result.lower_u, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.midlevel_v, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.upperlevel_temperature, 280.0, atol=1e-5)


def test_joint_runtime_adapter_owns_horizontal_mse_and_phase_conversion():
    h, w = 8, 16
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    heating = np.broadcast_to(80.0 * np.exp(-(latitude[:, None] / 18.0) ** 2), (h, w))
    humidity = np.broadcast_to(0.012 + 0.002 * np.cos(np.radians(latitude))[:, None], (h, w))
    result = evolve_joint_mse_momentum_pressure_column_runtime(
        heating,
        humidity, 0.30 * humidity, 0.08 * humidity,
        humidity, 0.30 * humidity, 0.08 * humidity,
        np.full((h, w), 300.0), np.full((h, w), 272.0), np.full((h, w), 240.0),
        np.full((h, w), -6.0), np.full((h, w), 2.0),
        np.full((h, w), 2.0), np.full((h, w), -1.0),
        np.full((h, w), 14.0), np.full((h, w), 3.0),
        lower_vapour_source_kg_kg_day=np.zeros((h, w)),
        dt_seconds=86400.0, radius_m=6.371e6, sidereal_day_hours=24.0,
        surface_pressure_pa=101325.0, lower_mid_pressure_depth_pa=35000.0,
        mid_upper_pressure_depth_pa=30000.0, dx_m=100_000.0, dy_m=100_000.0,
        cell_area_m2=1.0e10, x_face_length_m=100_000.0, y_face_length_m=100_000.0,
        critical_relative_humidity=0.8,
    )
    assert result.joint.substeps >= 1
    assert result.joint.vertical_courant_max <= 0.25 + 1e-10
    assert abs(result.joint.water_residual_kg_m2) < 1e-7
    assert abs(result.joint.moist_static_energy_residual_j_m2) < 1e-3
    assert abs(result.horizontal_mse_relative_energy_residual) < 1e-12
    assert np.all(result.lower_condensed_specific_humidity >= 0.0)
    assert np.all(np.isfinite(result.joint.upperlevel_temperature))


def test_joint_pressure_column_coupling_residual_uses_current_reservoir_fallout():
    """The nonlinear residual owns phase, reservoir, water, and heating units."""
    h, w = 8, 16
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    heating = np.broadcast_to(80.0 * np.exp(-(latitude[:, None] / 18.0) ** 2), (h, w))
    humidity = np.broadcast_to(0.012 + 0.002 * np.cos(np.radians(latitude))[:, None], (h, w))
    surface_source = np.full((h, w), 1.0)
    result = diagnose_joint_pressure_column_coupling_residual(
        heating, np.zeros((h, w)), np.zeros((h, w)), surface_source,
        np.full((h, w), 0.5), np.full((h, w), 0.25),
        humidity, 0.30 * humidity, 0.08 * humidity,
        humidity, 0.30 * humidity, 0.08 * humidity,
        np.full((h, w), 300.0), np.full((h, w), 272.0), np.full((h, w), 240.0),
        np.full((h, w), -6.0), np.full((h, w), 2.0),
        np.full((h, w), 2.0), np.full((h, w), -1.0),
        np.full((h, w), 14.0), np.full((h, w), 3.0),
        lower_vapour_source_kg_kg_day=np.zeros((h, w)),
        dt_seconds=86400.0, radius_m=6.371e6, sidereal_day_hours=24.0,
        surface_pressure_pa=101325.0, lower_mid_pressure_depth_pa=35000.0,
        mid_upper_pressure_depth_pa=30000.0, dx_m=100_000.0, dy_m=100_000.0,
        cell_area_m2=1.0e10, x_face_length_m=100_000.0, y_face_length_m=100_000.0,
        critical_relative_humidity=0.8, cloud_retention_kg_m2=1.0,
        autoconversion_timescale_days=0.25, fallout_timescale_days=1.0,
    )
    layer_mass = np.array((0.40, 0.35, 0.25)) * 101325.0 / 9.80665
    water_before = (
        layer_mass[0] * humidity + layer_mass[1] * 0.30 * humidity
        + layer_mass[2] * 0.08 * humidity + 0.5 + 0.25
    )
    water_after = (
        layer_mass[0] * result.runtime.joint.lower_humidity
        + layer_mass[1] * result.runtime.joint.midlevel_humidity
        + layer_mass[2] * result.runtime.joint.upperlevel_humidity
        + result.cloud_condensate_kg_m2 + result.precipitating_hydrometeors_kg_m2
    )
    np.testing.assert_allclose(
        result.diagnosed_water_forcing_kg_m2_s,
        column_water_forcing_from_budget(
            surface_source, result.fallout_kg_m2, water_before, water_after,
            dt_seconds=86400.0,
        ),
    )
    assert np.all(result.cloud_condensate_kg_m2 >= 0.0)
    assert np.all(result.precipitating_hydrometeors_kg_m2 >= 0.0)
    assert np.all(np.isfinite(result.heating_residual_w_m2))
    assert np.all(np.isfinite(result.water_forcing_residual_kg_m2_s))


def test_joint_pressure_column_coupling_residual_excludes_uniform_boundary_fluxes():
    """Closed meridional circulation cannot constrain a global-mean source."""
    h, w = 6, 12
    humidity = np.full((h, w), 0.012)
    common = dict(
        previous_heating_w_m2=np.zeros((h, w)), surface_source_kg_m2=np.ones((h, w)),
        cloud_condensate_kg_m2=np.zeros((h, w)),
        precipitating_hydrometeors_kg_m2=np.zeros((h, w)),
        lower_humidity_before_horizontal=humidity,
        midlevel_humidity_before_horizontal=0.3 * humidity,
        upperlevel_humidity_before_horizontal=0.08 * humidity,
        lower_humidity_after_horizontal=humidity,
        midlevel_humidity_after_horizontal=0.3 * humidity,
        upperlevel_humidity_after_horizontal=0.08 * humidity,
        lower_temperature_k=np.full((h, w), 300.0),
        midlevel_temperature_k=np.full((h, w), 272.0),
        upperlevel_temperature_k=np.full((h, w), 240.0),
        lower_u_m_s=np.zeros((h, w)), lower_v_m_s=np.zeros((h, w)),
        midlevel_u_m_s=np.zeros((h, w)), midlevel_v_m_s=np.zeros((h, w)),
        upperlevel_u_m_s=np.zeros((h, w)), upperlevel_v_m_s=np.zeros((h, w)),
        lower_vapour_source_kg_kg_day=np.zeros((h, w)), dt_seconds=86400.0,
        radius_m=6.371e6, sidereal_day_hours=24.0, surface_pressure_pa=101325.0,
        lower_mid_pressure_depth_pa=35000.0, mid_upper_pressure_depth_pa=30000.0,
        dx_m=100_000.0, dy_m=100_000.0, cell_area_m2=1.0e10,
        x_face_length_m=100_000.0, y_face_length_m=100_000.0,
        critical_relative_humidity=0.8, cloud_retention_kg_m2=1.0,
        autoconversion_timescale_days=0.25, fallout_timescale_days=1.0,
    )
    baseline = diagnose_joint_pressure_column_coupling_residual(
        np.zeros((h, w)), np.zeros((h, w)), **common,
    )
    shifted = diagnose_joint_pressure_column_coupling_residual(
        np.full((h, w), 17.0), np.full((h, w), 2.0e-5), **common,
    )
    np.testing.assert_allclose(
        shifted.heating_residual_w_m2, baseline.heating_residual_w_m2, atol=1e-6,
    )
    np.testing.assert_allclose(
        shifted.water_forcing_residual_kg_m2_s,
        baseline.water_forcing_residual_kg_m2_s, atol=1e-12,
    )


def test_joint_pressure_column_coupling_converges_the_exact_candidate_map():
    h, w = 8, 16
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    initial_heating = np.broadcast_to(
        80.0 * np.exp(-(latitude[:, None] / 18.0) ** 2), (h, w)
    )
    humidity = np.broadcast_to(
        0.012 + 0.002 * np.cos(np.radians(latitude))[:, None], (h, w)
    )
    zeros = np.zeros((h, w))

    def evaluate(candidate_heating: np.ndarray, candidate_water: np.ndarray):
        return diagnose_joint_pressure_column_coupling_residual(
            candidate_heating, candidate_water, zeros, np.full((h, w), 1.0),
            np.full((h, w), 0.5), np.full((h, w), 0.25),
            humidity, 0.30 * humidity, 0.08 * humidity,
            humidity, 0.30 * humidity, 0.08 * humidity,
            np.full((h, w), 300.0), np.full((h, w), 272.0), np.full((h, w), 240.0),
            np.full((h, w), -6.0), np.full((h, w), 2.0),
            np.full((h, w), 2.0), np.full((h, w), -1.0),
            np.full((h, w), 14.0), np.full((h, w), 3.0),
            lower_vapour_source_kg_kg_day=zeros,
            dt_seconds=86400.0, radius_m=6.371e6, sidereal_day_hours=24.0,
            surface_pressure_pa=101325.0, lower_mid_pressure_depth_pa=35000.0,
            mid_upper_pressure_depth_pa=30000.0, dx_m=100_000.0, dy_m=100_000.0,
            cell_area_m2=1.0e10, x_face_length_m=100_000.0, y_face_length_m=100_000.0,
            critical_relative_humidity=0.8, cloud_retention_kg_m2=1.0,
            autoconversion_timescale_days=0.25, fallout_timescale_days=1.0,
        )

    result = converge_joint_pressure_column_coupling(evaluate, initial_heating, zeros)
    zero_seed = converge_joint_pressure_column_coupling(evaluate, zeros, zeros)
    assert result.iterations < 8 * h
    assert float(np.max(np.abs(result.residual.heating_residual_w_m2))) < 1e-4
    assert float(np.max(np.abs(result.residual.water_forcing_residual_kg_m2_s))) < 1e-9
    np.testing.assert_allclose(result.heating_w_m2, zero_seed.heating_w_m2, atol=1e-4)
    np.testing.assert_allclose(
        result.water_forcing_kg_m2_s, zero_seed.water_forcing_kg_m2_s, atol=1e-10,
    )


def test_simultaneous_runtime_adapter_returns_converged_reservoirs_and_heating():
    h, w = 8, 16
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    humidity = np.broadcast_to(
        0.012 + 0.002 * np.cos(np.radians(latitude))[:, None], (h, w)
    )
    result = evolve_simultaneous_joint_pressure_column_runtime(
        np.broadcast_to(80.0 * np.exp(-(latitude[:, None] / 18.0) ** 2), (h, w)),
        np.full((h, w), 1.0), np.full((h, w), 0.5), np.full((h, w), 0.25),
        humidity, 0.30 * humidity, 0.08 * humidity,
        humidity, 0.30 * humidity, 0.08 * humidity,
        np.full((h, w), 300.0), np.full((h, w), 272.0), np.full((h, w), 240.0),
        np.full((h, w), -6.0), np.full((h, w), 2.0),
        np.full((h, w), 2.0), np.full((h, w), -1.0),
        np.full((h, w), 14.0), np.full((h, w), 3.0),
        lower_vapour_source_kg_kg_day=np.zeros((h, w)),
        dt_seconds=86400.0, radius_m=6.371e6, sidereal_day_hours=24.0,
        surface_pressure_pa=101325.0, lower_mid_pressure_depth_pa=35000.0,
        mid_upper_pressure_depth_pa=30000.0, dx_m=100_000.0, dy_m=100_000.0,
        cell_area_m2=1.0e10, x_face_length_m=100_000.0, y_face_length_m=100_000.0,
        critical_relative_humidity=0.8, cloud_retention_kg_m2=1.0,
        autoconversion_timescale_days=0.25, fallout_timescale_days=1.0,
        reservoir_transport_u_m_s=np.full((h, w), 2.0),
        reservoir_transport_v_m_s=np.zeros((h, w)), cloud_transport_scale=1.0,
        transport_hydrometeors=True,
    )
    assert result.iterations < 8 * h
    assert np.all(result.cloud_condensate_kg_m2 >= 0.0)
    assert np.all(result.precipitating_hydrometeors_kg_m2 >= 0.0)
    assert np.all(np.isfinite(result.heating_w_m2))
    assert np.all(np.isfinite(result.water_forcing_kg_m2_s))


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


def test_joint_pressure_column_runtime_replaces_the_legacy_monthly_split():
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
        enable_pressure_coordinate_mse_transport=True,
        enable_mse_constrained_pressure_circulation=True,
        enable_three_branch_mse_pressure_circulation=True,
        enable_momentum_constrained_three_branch_mse_circulation=True,
        enable_prognostic_pressure_coordinate_momentum=True,
    )
    state = create_initial_state(np.zeros(shape, dtype=np.float32), planet_params=planet)
    latitude = 90.0 - (np.arange(shape[0]) + 0.5) * 180.0 / shape[0]
    heating = np.broadcast_to(80.0 * np.exp(-(latitude[:, None] / 18.0) ** 2), shape)
    state = state._replace(pressure_overturning_heating_w_m2=heating.astype(np.float32))
    debug: dict = {}
    evolved, _ = simulate_step(
        state, days=1.0, planet_params=planet, precipitation_debug=debug
    )

    assert debug["joint_pressure_column_runtime"] is True
    assert "prognostic_pressure_coordinate_momentum" not in debug
    assert debug["joint_pressure_column_vertical_courant_max"] <= 0.25 + 1e-10
    assert evolved.upperlevel_wind_u is not None
    assert evolved.midlevel_wind_v is not None
    assert evolved.omega_lower_mid_pa_s is not None
    assert np.all(np.isfinite(evolved.upperlevel_wind_u))
    assert np.all(np.isfinite(evolved.omega_lower_mid_pa_s))


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


def test_hydrostatic_sigma_gate_off_preserves_persisted_experimental_state():
    shape = (8, 16)
    state = create_initial_state(np.zeros(shape, dtype=np.float32), planet_params=EARTH)
    fields = {
        "lower_pressure_depth_pa": np.full(shape, 40_000.0, dtype=np.float32),
        "midlevel_pressure_depth_pa": np.full(shape, 35_000.0, dtype=np.float32),
        "upperlevel_pressure_depth_pa": np.full(shape, 25_000.0, dtype=np.float32),
        "lower_pressure_cloud_condensate": np.full(shape, 0.2, dtype=np.float32),
        "midlevel_pressure_cloud_condensate": np.full(shape, 0.3, dtype=np.float32),
        "upperlevel_pressure_cloud_condensate": np.full(shape, 0.1, dtype=np.float32),
        "lower_pressure_hydrometeors": np.full(shape, 0.05, dtype=np.float32),
        "midlevel_pressure_hydrometeors": np.full(shape, 0.07, dtype=np.float32),
        "upperlevel_pressure_hydrometeors": np.full(shape, 0.03, dtype=np.float32),
        "pressure_coordinate_heat_convergence_w_m2": np.full(shape, 7.0, dtype=np.float32),
    }
    evolved, _ = simulate_step(state._replace(**fields), days=1.0, planet_params=EARTH)
    for name, expected in fields.items():
        np.testing.assert_array_equal(getattr(evolved, name), expected)


def test_hydrostatic_sigma_runtime_owns_and_persists_the_full_layer_state():
    shape = (8, 16)
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
        enable_pressure_coordinate_mse_transport=True,
        enable_mse_constrained_pressure_circulation=True,
        enable_three_branch_mse_pressure_circulation=True,
        enable_momentum_constrained_three_branch_mse_circulation=True,
        enable_prognostic_pressure_coordinate_momentum=True,
        enable_hydrostatic_sigma_pressure_coordinate_transport=True,
    )
    initial = create_initial_state(np.zeros(shape, dtype=np.float32), planet_params=EARTH)
    debug: dict = {}
    evolved, _ = simulate_step(
        initial, days=1.0, planet_params=planet, precipitation_debug=debug
    )

    assert debug["hydrostatic_sigma_runtime"] is True
    for name in (
        "lower_pressure_depth_pa", "midlevel_pressure_depth_pa", "upperlevel_pressure_depth_pa",
        "lower_pressure_cloud_condensate", "midlevel_pressure_cloud_condensate", "upperlevel_pressure_cloud_condensate",
        "lower_pressure_hydrometeors", "midlevel_pressure_hydrometeors", "upperlevel_pressure_hydrometeors",
        "midlevel_temperature", "upperlevel_temperature", "upperlevel_wind_u",
        "pressure_coordinate_heat_convergence_w_m2",
    ):
        value = getattr(evolved, name)
        assert value is not None
        assert value.shape == shape
        assert np.all(np.isfinite(value))
    # The atomic transition replaces the legacy heating-reservoir owner; it
    # must not manufacture that otherwise-absent state as a side effect.
    assert evolved.pressure_overturning_heating_w_m2 is None
