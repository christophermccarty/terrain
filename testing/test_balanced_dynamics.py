from __future__ import annotations

import numpy as np

from balanced_dynamics import (
    balanced_pressure_wind,
    diabatic_overturning_speed,
    moist_static_energy_overturning_speed,
    pressure_level_geopotential,
    thermally_direct_overturning,
)
from planet_params import EARTH
from simulate import create_initial_state, simulate_step
import dataclasses


def test_hydrostatic_geopotential_uses_zonal_mean_temperature_when_requested():
    elevation = np.zeros((24, 48))
    temperature = np.full((24, 48), 280.0)
    temperature[:, 3] += 10.0
    phi = pressure_level_geopotential(
        elevation, temperature, gravity_m_s2=9.81, gas_constant_dry=287.05,
        surface_pressure_pa=101325.0, level_pressure_pa=30000.0,
    )
    assert np.allclose(phi, phi[:, :1])


def test_balanced_pressure_wind_creates_westerlies_from_equator_warm_profile():
    h, w = 36, 72
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    temperature = 300.0 - 0.45 * np.abs(latitude)[:, None]
    phi = pressure_level_geopotential(
        np.zeros((h, w)), np.broadcast_to(temperature, (h, w)),
        gravity_m_s2=9.81, gas_constant_dry=287.05,
        surface_pressure_pa=101325.0, level_pressure_pa=30000.0,
    )
    wind = balanced_pressure_wind(
        phi, radius_m=6.371e6, sidereal_day_hours=24.0,
        hadley_edge_deg=24.0,
    )
    zonal_u = np.mean(wind.u, axis=1)
    assert float(np.max(zonal_u[np.abs(latitude) >= 30.0])) > 5.0
    assert float(np.max(np.abs(zonal_u[np.abs(latitude) <= 5.0]))) < 0.5


def test_ageostrophic_component_is_bounded_against_geostrophic_speed():
    h, w = 24, 48
    longitude = np.linspace(0.0, 2.0 * np.pi, w, endpoint=False)
    phi = np.broadcast_to(500.0 * np.sin(longitude)[None, :], (h, w))
    wind = balanced_pressure_wind(
        phi, radius_m=6.371e6, sidereal_day_hours=24.0,
        hadley_edge_deg=20.0, ageostrophic_timescale_hours=24.0,
    )
    assert np.max(np.hypot(wind.u_ageostrophic, wind.v_ageostrophic)) <= (
        0.5 * np.max(np.hypot(wind.u_geostrophic, wind.v_geostrophic)) + 1.0
    )


def test_thermally_direct_overturning_is_mass_conserving_and_centres_on_heat_maximum():
    h, w = 36, 72
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    temperature = 290.0 - 0.2 * np.abs(latitude[:, None] - 8.0)
    overturning = thermally_direct_overturning(
        np.broadcast_to(temperature, (h, w)),
        hadley_edge_deg=24.0, lower_branch_speed_m_s=0.5,
    )
    mass_flux = (
        0.40 * overturning.lower_v
        + 0.35 * overturning.middle_v
        + 0.25 * overturning.upper_v
    )
    assert abs(overturning.thermal_equator_deg - 8.0) < 2.0
    assert np.max(np.abs(mass_flux)) < 1e-6
    north_of_itcz = np.argmin(np.abs(latitude - 15.0))
    assert float(np.mean(overturning.lower_v[north_of_itcz])) < 0.0


def test_diabatic_overturning_speed_is_zero_without_heating_and_bounded_when_heated():
    lower = np.full((24, 48), 290.0)
    reference_middle = lower - 6.5e-3 * 1500.0
    common = dict(
        radius_m=6.371e6, hadley_edge_deg=24.0, layer_pressure_depth_pa=40000.0,
        midlevel_height_m=1500.0, relaxation_days=4.0, max_speed_m_s=2.0,
    )
    assert diabatic_overturning_speed(lower, reference_middle, **common) == 0.0
    heated = diabatic_overturning_speed(lower, reference_middle + 4.0, **common)
    assert 0.0 < heated <= 2.0


def test_balanced_pressure_gate_changes_middle_and_upper_wind_states():
    shape = (12, 24)
    common = dict(
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
        enable_three_level_pressure_column=True,
    )
    control_pp = dataclasses.replace(EARTH, **common)
    balanced_pp = dataclasses.replace(
        EARTH, **common,
        enable_native_balanced_pressure_dynamics=True,
        native_balanced_pressure_relaxation=0.5,
    )
    state = create_initial_state(np.zeros(shape, dtype=np.float32), planet_params=control_pp)
    control, _ = simulate_step(state, days=1.0, planet_params=control_pp)
    balanced, _ = simulate_step(state, days=1.0, planet_params=balanced_pp)
    assert balanced.midlevel_wind_u is not None
    assert balanced.wind_u_aloft is not None
    assert float(np.mean(np.abs(balanced.wind_u - control.wind_u))) > 1e-3
    assert float(np.mean(np.abs(balanced.midlevel_wind_u - control.midlevel_wind_u))) > 1e-3
    # Section 17 (PRIOR_ART_IMPLEMENTATION_PLAN.md): the balanced-pressure
    # blend's "upper" target now applies to the three-level path's own
    # independent upperlevel_wind_u/v state, not the shared, always-on
    # jet-stream kernel (wind_u_aloft/wind_v_aloft) -- so the gate must
    # change the former and leave the latter bit-identical.
    assert balanced.upperlevel_wind_u is not None
    assert float(np.mean(np.abs(balanced.upperlevel_wind_u - control.upperlevel_wind_u))) > 1e-3
    np.testing.assert_array_equal(balanced.wind_u_aloft, control.wind_u_aloft)


def test_moist_static_energy_overturning_speed_is_zero_without_heating():
    lower = np.full((24, 48), 290.0)
    reference_middle = lower - 6.5e-3 * 1500.0
    common = dict(
        radius_m=6.371e6, hadley_edge_deg=24.0, layer_pressure_depth_pa=40000.0,
        midlevel_height_m=1500.0, latent_relaxation_days=4.0,
        radiative_relaxation_days=10.0, max_speed_m_s=2.0,
    )
    result = moist_static_energy_overturning_speed(
        lower, reference_middle, radiative_equilibrium_temperature_k=lower, **common,
    )
    assert result.speed_m_s == 0.0
    assert result.latent_heating_k_s == 0.0
    assert result.radiative_heating_k_s == 0.0
    assert result.total_heating_k_s == 0.0


def test_moist_static_energy_overturning_speed_is_bounded_with_heavy_precipitation():
    lower = np.full((24, 48), 290.0)
    reference_middle = lower - 6.5e-3 * 1500.0
    common = dict(
        radius_m=6.371e6, hadley_edge_deg=24.0, layer_pressure_depth_pa=40000.0,
        midlevel_height_m=1500.0, latent_relaxation_days=4.0,
        radiative_relaxation_days=10.0, max_speed_m_s=2.0,
    )
    result = moist_static_energy_overturning_speed(
        lower, reference_middle, radiative_equilibrium_temperature_k=lower,
        precipitation_mm_day=np.full((24, 48), 500.0), **common,
    )
    assert result.speed_m_s == 2.0


def test_moist_static_energy_overturning_heating_terms_are_additive():
    lower = np.full((24, 48), 290.0)
    middle = lower - 6.5e-3 * 1500.0 + 3.0
    radiative_eq = lower + 5.0
    result = moist_static_energy_overturning_speed(
        lower, middle, radiative_equilibrium_temperature_k=radiative_eq,
        radius_m=6.371e6, hadley_edge_deg=24.0, layer_pressure_depth_pa=40000.0,
        midlevel_height_m=1500.0, latent_relaxation_days=4.0,
        radiative_relaxation_days=10.0, max_speed_m_s=2.0,
    )
    assert result.latent_heating_k_s > 0.0
    assert result.radiative_heating_k_s > 0.0
    assert abs(
        result.total_heating_k_s - (result.latent_heating_k_s + result.radiative_heating_k_s)
    ) < 1e-15


def test_moist_static_energy_overturning_falls_back_to_midlevel_anomaly_without_precipitation():
    lower = np.full((24, 48), 290.0)
    middle = lower - 6.5e-3 * 1500.0 + 3.0
    result = moist_static_energy_overturning_speed(
        lower, middle, radiative_equilibrium_temperature_k=lower,
        radius_m=6.371e6, hadley_edge_deg=24.0, layer_pressure_depth_pa=40000.0,
        midlevel_height_m=1500.0, latent_relaxation_days=4.0,
        radiative_relaxation_days=10.0, max_speed_m_s=2.0,
    )
    assert result.radiative_heating_k_s == 0.0
    assert result.latent_heating_k_s > 0.0
    assert result.speed_m_s > 0.0


def test_moist_static_energy_overturning_speed_feeds_mass_conserving_three_level_structure():
    h, w = 36, 72
    latitude = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lower = np.broadcast_to(290.0 - 0.2 * np.abs(latitude[:, None] - 8.0), (h, w)).copy()
    middle = lower - 6.5e-3 * 1500.0 + 3.0
    result = moist_static_energy_overturning_speed(
        lower, middle, radiative_equilibrium_temperature_k=lower,
        radius_m=6.371e6, hadley_edge_deg=24.0, layer_pressure_depth_pa=40000.0,
        midlevel_height_m=1500.0, latent_relaxation_days=4.0,
        radiative_relaxation_days=10.0, max_speed_m_s=2.0,
    )
    overturning = thermally_direct_overturning(
        lower, hadley_edge_deg=24.0, lower_branch_speed_m_s=result.speed_m_s,
    )
    mass_flux = (
        0.40 * overturning.lower_v
        + 0.35 * overturning.middle_v
        + 0.25 * overturning.upper_v
    )
    assert result.speed_m_s > 0.0
    assert np.max(np.abs(mass_flux)) < 1e-6


def test_moist_static_energy_overturning_gate_is_default_off():
    assert EARTH.enable_native_balanced_moist_static_energy_overturning is False


def test_two_level_thermally_direct_overturning_is_default_off_and_wired():
    assert EARTH.enable_two_level_thermally_direct_overturning is False
    state = create_initial_state(np.zeros((12, 24), dtype=np.float32), planet_params=EARTH)
    control, _ = simulate_step(state, days=1.0, planet_params=EARTH)
    active = dataclasses.replace(
        EARTH, enable_two_level_thermally_direct_overturning=True,
        two_level_thermally_direct_overturning_speed_m_s=0.5,
    )
    changed, _ = simulate_step(state, days=1.0, planet_params=active)
    assert float(np.mean(np.abs(changed.wind_v - control.wind_v))) > 0.0
    assert float(np.mean(np.abs(changed.wind_v_aloft - control.wind_v_aloft))) > 0.0
    lower_anomaly = changed.wind_v - control.wind_v
    upper_anomaly = changed.wind_v_aloft - control.wind_v_aloft
    # The extracted primitive uses the same 0.40 lower / 0.60 return-layer
    # mass split as its three-level source. Its *increment* must therefore
    # add no meridional column mass flux.
    assert np.max(np.abs(0.40 * lower_anomaly + 0.60 * upper_anomaly)) < 1e-6


def test_moist_static_energy_overturning_gate_changes_native_balanced_winds():
    shape = (12, 24)
    common = dict(
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
        enable_three_level_pressure_column=True,
    )
    control_pp = dataclasses.replace(EARTH, **common)
    mse_pp = dataclasses.replace(
        EARTH, **common,
        enable_native_balanced_moist_static_energy_overturning=True,
        native_balanced_mse_overturning_max_speed_m_s=2.0,
    )
    state = create_initial_state(np.zeros(shape, dtype=np.float32), planet_params=control_pp)
    # The midlevel/upper reservoirs are None on the very first step, so the
    # overturning gate has nothing to act on until they exist -- warm the
    # state up once under the shared control config before branching.
    warm, _ = simulate_step(state, days=1.0, planet_params=control_pp)
    control, _ = simulate_step(warm, days=1.0, planet_params=control_pp)
    mse_result, _ = simulate_step(warm, days=1.0, planet_params=mse_pp)
    assert float(np.mean(np.abs(mse_result.wind_v - control.wind_v))) > 0.0


def test_mse_toa_radiative_target_gate_is_default_off():
    assert EARTH.native_balanced_mse_use_toa_radiative_target is False


def test_mse_toa_radiative_target_changes_the_diagnosed_overturning():
    shape = (12, 24)
    common = dict(
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
        enable_three_level_pressure_column=True,
        enable_native_balanced_moist_static_energy_overturning=True,
        # Uncapped: at the default 2.0 m/s cap both targets saturate to the
        # same bound on this toy grid, masking the difference under test.
        native_balanced_mse_overturning_max_speed_m_s=1000.0,
    )
    ocean_target_pp = dataclasses.replace(EARTH, **common)
    toa_target_pp = dataclasses.replace(
        EARTH, **common, native_balanced_mse_use_toa_radiative_target=True,
    )
    state = create_initial_state(np.zeros(shape, dtype=np.float32), planet_params=ocean_target_pp)
    # Warm the state once under a shared config so the midlevel/upper
    # reservoirs exist before the two radiative targets diverge.
    warm, _ = simulate_step(state, days=1.0, planet_params=ocean_target_pp)
    ocean_result, _ = simulate_step(warm, days=1.0, planet_params=ocean_target_pp)
    toa_result, _ = simulate_step(warm, days=1.0, planet_params=toa_target_pp)
    assert float(np.mean(np.abs(toa_result.wind_v - ocean_result.wind_v))) > 0.0
