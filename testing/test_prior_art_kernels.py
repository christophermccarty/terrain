from __future__ import annotations

import numpy as np

from column_water import evolve_column_water
from condensate import stability_aware_condensation
from atmosphere import generate_precipitation
from land_surface import force_restore_penman_monteith
from orographic_linear import smith_barstad_precipitation_anomaly
from planet_params import EARTH
from scripts.sweep_force_restore_land import candidate_overrides
from scripts.check_linear_orographic_theory import _local_tile_slices
from regional_validation import OROGRAPHIC_PAIRS
from simulate import create_initial_state, simulate_step
import dataclasses


def test_smith_barstad_is_zero_over_flat_terrain():
    out = smith_barstad_precipitation_anomaly(
        np.zeros((16, 32)), dx_m=10_000.0, dy_m=10_000.0,
        wind_u_m_s=10.0, wind_v_m_s=0.0, latitude_deg=45.0,
    )
    np.testing.assert_array_equal(out, 0.0)


def test_smith_barstad_reverses_the_ridge_footprint_with_wind():
    terrain = np.zeros((64, 64))
    terrain[:, 31:34] = 1500.0
    east = smith_barstad_precipitation_anomaly(
        terrain, dx_m=10_000.0, dy_m=10_000.0,
        wind_u_m_s=10.0, wind_v_m_s=0.0, latitude_deg=45.0,
    )
    west = smith_barstad_precipitation_anomaly(
        terrain, dx_m=10_000.0, dy_m=10_000.0,
        wind_u_m_s=-10.0, wind_v_m_s=0.0, latitude_deg=45.0,
    )
    assert float(np.mean(east[:, :31])) > float(np.mean(east[:, 34:]))
    assert float(np.mean(west[:, 34:])) > float(np.mean(west[:, :31]))


def test_force_restore_partitions_more_wet_energy_into_latent_heat():
    shape = (4, 8)
    common = dict(
        surface_temperature_k=np.full(shape, 303.0),
        air_temperature_k=np.full(shape, 295.0),
        deep_temperature_k=np.full(shape, 296.0),
        wind_speed_m_s=np.full(shape, 4.0),
        net_radiation_w_m2=np.full(shape, 350.0),
        land_mask=np.ones(shape, dtype=bool),
        dt_days=1.0,
        surface_heat_capacity_j_m2_k=1_500_000.0,
        deep_heat_capacity_j_m2_k=12_000_000.0,
        restore_days=30.0,
        surface_resistance_min_s_m=70.0,
        surface_resistance_dry_s_m=2_000.0,
    )
    wet = force_restore_penman_monteith(soil_moisture=np.ones(shape), **common)
    dry = force_restore_penman_monteith(soil_moisture=np.zeros(shape), **common)
    assert float(np.mean(wet.latent_heat_w_m2)) > float(np.mean(dry.latent_heat_w_m2))
    assert float(np.mean(wet.temperature)) < float(np.mean(dry.temperature))


def test_force_restore_initializes_an_unknown_deep_reservoir_at_surface_equilibrium():
    shape = (2, 4)
    surface = np.full(shape, 295.0)
    step = force_restore_penman_monteith(
        surface, np.full(shape, 280.0), None, np.zeros(shape), np.ones(shape),
        np.zeros(shape), np.ones(shape, dtype=bool), dt_days=1.0,
        surface_heat_capacity_j_m2_k=1_500_000.0,
        deep_heat_capacity_j_m2_k=12_000_000.0,
        restore_days=30.0,
        surface_resistance_min_s_m=70.0,
        surface_resistance_dry_s_m=2_000.0,
    )
    np.testing.assert_allclose(step.deep_temperature, surface)


def test_force_restore_soil_moisture_deep_none_matches_shallow_fallback():
    """`soil_moisture_deep=None` must reproduce passing the shallow bucket twice.

    This is the exact-backward-compatibility contract documented on
    `force_restore_penman_monteith`: callers that predate the deep-bucket wiring
    (or run before the deep bucket has spun up) get identical output.
    """
    shape = (4, 8)
    common = dict(
        surface_temperature_k=np.full(shape, 303.0),
        air_temperature_k=np.full(shape, 295.0),
        deep_temperature_k=np.full(shape, 296.0),
        soil_moisture=np.full(shape, 0.3),
        wind_speed_m_s=np.full(shape, 4.0),
        net_radiation_w_m2=np.full(shape, 350.0),
        land_mask=np.ones(shape, dtype=bool),
        dt_days=1.0,
        surface_heat_capacity_j_m2_k=1_500_000.0,
        deep_heat_capacity_j_m2_k=12_000_000.0,
        restore_days=30.0,
        surface_resistance_min_s_m=70.0,
        surface_resistance_dry_s_m=2_000.0,
    )
    implicit = force_restore_penman_monteith(**common)
    explicit = force_restore_penman_monteith(soil_moisture_deep=np.full(shape, 0.3), **common)
    np.testing.assert_array_equal(implicit.temperature, explicit.temperature)
    np.testing.assert_array_equal(implicit.deep_temperature, explicit.deep_temperature)


def test_force_restore_wetter_deep_soil_damps_the_deep_reservoir_step():
    """Wetter deep soil should carry more thermal inertia (a smaller step).

    Holds the shallow bucket, radiation, and restore flux conditions fixed and
    only varies `soil_moisture_deep`; the moisture-scaled deep heat capacity
    should make the wetter case's deep-temperature step smaller in magnitude.
    """
    shape = (3, 5)
    common = dict(
        surface_temperature_k=np.full(shape, 305.0),
        air_temperature_k=np.full(shape, 295.0),
        deep_temperature_k=np.full(shape, 290.0),
        soil_moisture=np.full(shape, 0.4),
        wind_speed_m_s=np.full(shape, 3.0),
        net_radiation_w_m2=np.full(shape, 300.0),
        land_mask=np.ones(shape, dtype=bool),
        dt_days=5.0,
        surface_heat_capacity_j_m2_k=1_500_000.0,
        deep_heat_capacity_j_m2_k=12_000_000.0,
        restore_days=30.0,
        surface_resistance_min_s_m=70.0,
        surface_resistance_dry_s_m=2_000.0,
    )
    dry_deep = force_restore_penman_monteith(soil_moisture_deep=np.full(shape, 0.05), **common)
    wet_deep = force_restore_penman_monteith(soil_moisture_deep=np.full(shape, 0.95), **common)
    dry_step = np.abs(dry_deep.deep_temperature - 290.0)
    wet_step = np.abs(wet_deep.deep_temperature - 290.0)
    assert float(np.mean(wet_step)) < float(np.mean(dry_step))


def test_force_restore_root_zone_resistance_blends_shallow_and_deep():
    """A dry shallow layer over wet deep soil should evapotranspire more than
    dry-over-dry, because the resistance term is root-zone (blended), not
    shallow-only.
    """
    shape = (2, 4)
    common = dict(
        surface_temperature_k=np.full(shape, 305.0),
        air_temperature_k=np.full(shape, 295.0),
        deep_temperature_k=np.full(shape, 296.0),
        soil_moisture=np.full(shape, 0.05),
        wind_speed_m_s=np.full(shape, 4.0),
        net_radiation_w_m2=np.full(shape, 350.0),
        land_mask=np.ones(shape, dtype=bool),
        dt_days=1.0,
        surface_heat_capacity_j_m2_k=1_500_000.0,
        deep_heat_capacity_j_m2_k=12_000_000.0,
        restore_days=30.0,
        surface_resistance_min_s_m=70.0,
        surface_resistance_dry_s_m=2_000.0,
    )
    dry_shallow_dry_deep = force_restore_penman_monteith(
        soil_moisture_deep=np.full(shape, 0.05), **common
    )
    dry_shallow_wet_deep = force_restore_penman_monteith(
        soil_moisture_deep=np.full(shape, 0.95), **common
    )
    assert float(np.mean(dry_shallow_wet_deep.latent_heat_w_m2)) > float(
        np.mean(dry_shallow_dry_deep.latent_heat_w_m2)
    )


def test_column_water_transport_conserves_mass_without_sources_or_sinks():
    water = np.zeros((8, 16), dtype=np.float32)
    water[:, 4:8] = 20.0
    step = evolve_column_water(
        water, np.zeros_like(water), np.zeros_like(water),
        np.full_like(water, 5.0), np.zeros_like(water),
        dx_m=100_000.0, dy_m=100_000.0, dt_days=0.05,
    )
    np.testing.assert_allclose(np.sum(step.water_mm), np.sum(water), rtol=0.0, atol=1e-5)
    assert abs(step.residual_mm) < 1e-5


def test_column_water_transport_conserves_area_weighted_mass_on_variable_grid():
    water = np.zeros((6, 12), dtype=np.float32)
    water[2:4, 3:7] = 18.0
    area = np.linspace(0.4, 1.0, water.shape[0])[:, None] * np.ones(water.shape[1])
    x_faces = np.ones_like(water)
    y_faces = np.ones((water.shape[0] + 1, water.shape[1]))
    y_faces[[0, -1]] = 0.0
    step = evolve_column_water(
        water, np.zeros_like(water), np.zeros_like(water),
        np.full_like(water, 0.03), np.full_like(water, -0.01),
        dx_m=np.ones_like(water), dy_m=1.0, dt_days=0.001,
        cell_area_m2=area, x_face_length_m=x_faces, y_face_length_m=y_faces,
    )
    np.testing.assert_allclose(
        np.sum(step.water_mm * area), np.sum(water * area), rtol=0.0, atol=3e-5
    )
    assert abs(step.residual_mm) < 3e-5


def test_force_restore_sweep_enables_only_the_replacement_branch():
    candidates = list(candidate_overrides((15.0,), (6_000_000.0,), (1_000.0, 2_000.0)))
    assert len(candidates) == 2
    assert all(candidate["enable_force_restore_land"] for candidate in candidates)


def test_orographic_local_tiles_enclose_both_validation_flanks():
    for pair in OROGRAPHIC_PAIRS:
        rows, cols = _local_tile_slices((256, 512), pair, buffer_degrees=6.0)
        assert rows.stop > rows.start
        assert cols.stop > cols.start


def test_terrain_pgf_multiplier_one_preserves_wind_evolution():
    elevation = np.zeros((12, 24), dtype=np.float32)
    elevation[3:9, 8:16] = 0.6
    state = create_initial_state(elevation, planet_params=EARTH)
    baseline, _ = simulate_step(state, days=1.0, planet_params=EARTH)
    explicit, _ = simulate_step(
        state, days=1.0,
        planet_params=dataclasses.replace(EARTH, wind_terrain_pgf_scale=1.0),
    )
    np.testing.assert_array_equal(baseline.wind_u, explicit.wind_u)
    np.testing.assert_array_equal(baseline.wind_v, explicit.wind_v)


def test_precipitation_exposes_closed_allocator_column_water_diagnostic():
    shape = (8, 16)
    debug: dict = {}
    generate_precipitation(
        *shape, np.zeros(shape, dtype=np.float32),
        humidity=np.full(shape, 0.01, dtype=np.float32), dt_days=1.0,
        debug_fields=debug,
    )
    assert "column_water_allocator_residual_mm" in debug
    assert "column_water_rainout_removal_mm" in debug
    np.testing.assert_allclose(debug["column_water_allocator_residual_mm"], 0.0, atol=2e-6)


def test_prognostic_column_water_bypasses_imposed_row_target_rescales():
    shape = (8, 16)
    debug: dict = {}
    generate_precipitation(
        *shape, np.zeros(shape, dtype=np.float32),
        humidity=np.full(shape, 0.01, dtype=np.float32), dt_days=1.0,
        target_mean_mm_day=5.0,
        planet_params=dataclasses.replace(EARTH, enable_prognostic_column_water=True),
        debug_fields=debug,
    )
    assert debug["column_water_mode"] == "raw_prognostic"
    np.testing.assert_array_equal(debug["zonal_rescale_factor"], 1.0)


def test_column_water_mode_transports_condensate_with_the_same_budget_kernel():
    shape = (8, 16)
    debug: dict = {}
    generate_precipitation(
        *shape, np.zeros(shape, dtype=np.float32),
        humidity=np.full(shape, 0.01, dtype=np.float32), dt_days=1.0,
        planet_params=dataclasses.replace(
            EARTH,
            enable_prognostic_column_water=True,
            enable_prognostic_condensate=True,
        ),
        debug_fields=debug,
    )
    assert "condensate_transport_relative_residual" in debug
    assert abs(float(debug["condensate_transport_relative_residual"])) < 1e-10
    assert "column_water_total_budget_relative_residual" in debug
    assert abs(float(debug["column_water_total_budget_relative_residual"])) < 1e-5


def test_bulk_condensate_gate_removes_empirical_vapor_rainout():
    shape = (8, 16)
    debug: dict = {}
    generate_precipitation(
        *shape, np.zeros(shape, dtype=np.float32),
        humidity=np.full(shape, 0.01, dtype=np.float32), dt_days=1.0,
        planet_params=dataclasses.replace(
            EARTH,
            enable_prognostic_column_water=True,
            enable_prognostic_condensate=True,
            column_water_use_bulk_condensate_rainfall=True,
        ),
        debug_fields=debug,
    )
    assert debug["column_water_precipitation_closure"] == "bulk_condensate"
    np.testing.assert_array_equal(debug["rainout_raw_dq"], 0.0)


def test_stability_aware_condensation_requires_moist_buoyant_ascending_air():
    shape = (2, 2)
    temperature = np.full(shape, 300.0)
    saturation = np.full(shape, 0.022)
    vapor = np.array([[0.021, 0.008], [0.021, 0.021]])
    ascent = np.array([[1.0, 1.0], [0.0, 1.0]])
    vapor_next, condensate, rainout, cape, activation = stability_aware_condensation(
        vapor, saturation, temperature, ascent, None,
        surface_pressure_hpa=1013.25, dt_days=0.1,
        condensation_timescale_days=0.1, fallout_timescale_days=1.0,
    )
    assert cape[0, 0] > 0.0
    assert activation[0, 0] > 0.0
    assert vapor_next[0, 0] < vapor[0, 0]
    np.testing.assert_allclose(vapor_next[0, 1], vapor[0, 1])
    np.testing.assert_allclose(vapor_next[1, 0], vapor[1, 0])
    np.testing.assert_allclose(vapor_next + condensate + rainout, vapor, atol=1e-9)


def test_stability_aware_condensation_is_exposed_only_when_enabled():
    shape = (8, 16)
    debug: dict = {}
    generate_precipitation(
        *shape, np.zeros(shape, dtype=np.float32),
        humidity=np.full(shape, 0.015, dtype=np.float32), dt_days=1.0,
        planet_params=dataclasses.replace(
            EARTH,
            enable_prognostic_condensate=True,
            enable_stability_aware_condensation=True,
        ),
        debug_fields=debug,
    )
    assert debug["condensate_closure"] == "stability_aware"
    assert "stability_cape_proxy_j_kg" in debug


def test_two_layer_adjustment_persists_midlevel_temperature_only_when_gated():
    elevation = np.zeros((12, 24), dtype=np.float32)
    planet = dataclasses.replace(
        EARTH,
        enable_prognostic_condensate=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
    )
    state = create_initial_state(elevation, planet_params=planet)
    evolved, _ = simulate_step(state, days=1.0, planet_params=planet)
    assert evolved.midlevel_temperature is not None
    assert evolved.midlevel_temperature.shape == elevation.shape
    assert np.all(np.isfinite(evolved.midlevel_temperature))
    assert evolved.midlevel_humidity is not None
    assert evolved.midlevel_humidity.shape == elevation.shape
    assert np.all(evolved.midlevel_humidity >= 0.0)


def test_two_layer_upper_humidity_transport_closes_the_column_water_budget():
    """The upper partition is transported/exchanged, never added to total water."""
    shape = (8, 16)
    initial_humidity = np.full(shape, 0.018, dtype=np.float32)
    initial_upper_humidity = np.zeros(shape, dtype=np.float32)
    initial_upper_humidity[:, 3:7] = 0.004
    debug: dict = {}
    planet = dataclasses.replace(
        EARTH,
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
    )
    result = generate_precipitation(
        *shape,
        np.zeros(shape, dtype=np.float32),
        humidity=initial_humidity,
        midlevel_humidity=initial_upper_humidity,
        wind_u=np.full(shape, 2.0, dtype=np.float32),
        wind_v=np.zeros(shape, dtype=np.float32),
        wind_u_aloft=np.full(shape, -4.0, dtype=np.float32),
        wind_v_aloft=np.zeros(shape, dtype=np.float32),
        dt_days=0.05,
        planet_params=planet,
        debug_fields=debug,
        return_condensate=True,
        return_midlevel_temperature=True,
        return_midlevel_humidity=True,
    )
    _, humidity_next, _, _, _, _, upper_humidity_next = result
    assert np.any(upper_humidity_next > 0.0)
    assert not np.array_equal(upper_humidity_next, initial_upper_humidity)
    np.testing.assert_allclose(
        humidity_next - upper_humidity_next,
        np.clip(humidity_next - upper_humidity_next, 0.0, None),
    )
    assert debug["condensate_transport_layer"] == "midlevel"
    assert abs(float(debug["midlevel_humidity_transport_relative_residual"])) < 1e-10
    assert abs(float(debug["column_water_total_budget_relative_residual"])) < 1e-5


def test_active_upper_layer_condenses_supersaturation_without_losing_water():
    shape = (6, 12)
    initial_upper_humidity = np.full(shape, 0.020, dtype=np.float32)
    debug: dict = {}
    planet = dataclasses.replace(
        EARTH,
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
    )
    result = generate_precipitation(
        *shape,
        np.zeros(shape, dtype=np.float32),
        temperature=np.full(shape, 300.0, dtype=np.float32),
        humidity=np.full(shape, 0.030, dtype=np.float32),
        midlevel_humidity=initial_upper_humidity,
        midlevel_temperature=np.full(shape, 275.0, dtype=np.float32),
        wind_u=np.zeros(shape, dtype=np.float32),
        wind_v=np.zeros(shape, dtype=np.float32),
        dt_days=0.1,
        planet_params=planet,
        debug_fields=debug,
        return_condensate=True,
        return_midlevel_temperature=True,
        return_midlevel_humidity=True,
    )
    precipitation, _, _, _, _, midlevel_temperature, upper_humidity = result
    assert np.all(debug["midlevel_condensed_q"] > 0.0)
    assert np.all(upper_humidity < initial_upper_humidity)
    assert np.all(midlevel_temperature > 275.0)
    assert float(np.mean(precipitation)) > 0.0
    assert abs(float(debug["column_water_total_budget_relative_residual"])) < 1e-5


def test_pressure_coordinate_vertical_velocity_drives_both_layer_exchanges():
    shape = (8, 16)
    longitude = np.arange(shape[1], dtype=np.float32)[None, :]
    lower_u = np.broadcast_to(5.0 * np.sin(2.0 * np.pi * longitude / shape[1]), shape).copy()
    upper_u = -lower_u
    debug: dict = {}
    planet = dataclasses.replace(
        EARTH,
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
    )
    generate_precipitation(
        *shape,
        np.zeros(shape, dtype=np.float32),
        temperature=np.full(shape, 300.0, dtype=np.float32),
        humidity=np.full(shape, 0.016, dtype=np.float32),
        midlevel_humidity=np.full(shape, 0.004, dtype=np.float32),
        wind_u=lower_u,
        wind_v=np.zeros(shape, dtype=np.float32),
        wind_u_aloft=upper_u,
        wind_v_aloft=np.zeros(shape, dtype=np.float32),
        dt_days=0.1,
        planet_params=planet,
        debug_fields=debug,
    )
    assert float(np.min(debug["midlevel_omega_pa_s"])) < 0.0
    assert float(np.max(debug["midlevel_omega_pa_s"])) > 0.0
    assert float(np.max(debug["two_layer_entrained_q"])) > 0.0
    assert float(np.max(debug["two_layer_detrained_q"])) > 0.0
    assert abs(float(debug["column_water_total_budget_relative_residual"])) < 1e-5


def test_midlevel_temperature_anomaly_feeds_back_to_resolved_air_when_gated():
    elevation = np.zeros((12, 24), dtype=np.float32)
    planet = dataclasses.replace(
        EARTH,
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
    )
    base = create_initial_state(elevation, planet_params=planet)
    reference_midlevel = base.air_temperature - 6.5e-3 * planet.stability_condensation_reference_height_m
    warm_midlevel = base._replace(midlevel_temperature=reference_midlevel + 6.0)
    base_next, _ = simulate_step(base, days=1.0, planet_params=planet)
    warm_next, _ = simulate_step(warm_midlevel, days=1.0, planet_params=planet)
    assert float(np.mean(warm_next.air_temperature - base_next.air_temperature)) > 0.0


def test_midlevel_condensate_feeds_back_into_radiative_cloud_cover_when_gated():
    elevation = np.zeros((12, 24), dtype=np.float32)
    planet = dataclasses.replace(
        EARTH,
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
    )
    base = create_initial_state(elevation, planet_params=planet)
    cloudy = base._replace(
        atmospheric_condensate=np.full(elevation.shape, 0.004, dtype=np.float32)
    )
    base_next, _ = simulate_step(base, days=1.0, planet_params=planet)
    cloudy_next, _ = simulate_step(cloudy, days=1.0, planet_params=planet)
    assert float(np.mean(cloudy_next.cloud_cover - base_next.cloud_cover)) > 0.0


def test_cloud_precipitating_partition_caps_only_radiative_condensate():
    elevation = np.zeros((12, 24), dtype=np.float32)
    common = dict(
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
    )
    unpartitioned = dataclasses.replace(EARTH, **common)
    partitioned = dataclasses.replace(
        EARTH,
        **common,
        enable_cloud_precipitating_condensate_partition=True,
        cloud_optical_condensate_cap_q=0.0005,
    )
    base = create_initial_state(elevation, planet_params=unpartitioned)._replace(
        atmospheric_condensate=np.full(elevation.shape, 0.004, dtype=np.float32)
    )
    unpartitioned_next, _ = simulate_step(base, days=1.0, planet_params=unpartitioned)
    partitioned_next, _ = simulate_step(base, days=1.0, planet_params=partitioned)
    assert float(np.mean(partitioned_next.cloud_cover)) < float(
        np.mean(unpartitioned_next.cloud_cover)
    )
    # The mass path is unchanged directly; the small first-step difference is
    # the expected temperature feedback from changed cloud radiation.
    assert float(np.max(np.abs(
        partitioned_next.atmospheric_condensate
        - unpartitioned_next.atmospheric_condensate
    ))) < 1e-4


def test_three_level_pressure_column_persists_upper_temperature_and_humidity():
    elevation = np.zeros((12, 24), dtype=np.float32)
    planet = dataclasses.replace(
        EARTH,
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
        enable_three_level_pressure_column=True,
    )
    state = create_initial_state(elevation, planet_params=planet)
    evolved, _ = simulate_step(state, days=1.0, planet_params=planet)
    assert evolved.upperlevel_temperature is not None
    assert evolved.upperlevel_humidity is not None
    assert evolved.midlevel_wind_u is not None
    assert evolved.midlevel_wind_v is not None
    assert evolved.upperlevel_temperature.shape == elevation.shape
    assert evolved.upperlevel_humidity.shape == elevation.shape
    assert evolved.midlevel_wind_u.shape == elevation.shape
    assert evolved.midlevel_wind_v.shape == elevation.shape
    assert np.all(evolved.upperlevel_humidity >= 0.0)
    assert np.all(evolved.humidity >= evolved.midlevel_humidity + evolved.upperlevel_humidity)


def test_separate_hydrometeor_state_persists_when_gated():
    elevation = np.zeros((12, 24), dtype=np.float32)
    planet = dataclasses.replace(
        EARTH,
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
        enable_three_level_pressure_column=True,
        enable_separate_precipitating_hydrometeors=True,
    )
    state = create_initial_state(elevation, planet_params=planet)._replace(
        humidity=np.full(elevation.shape, 0.030, dtype=np.float32)
    )
    evolved, _ = simulate_step(state, days=1.0, planet_params=planet)
    assert evolved.precipitating_hydrometeors is not None
    assert evolved.precipitating_hydrometeors.shape == elevation.shape
    assert np.all(evolved.precipitating_hydrometeors >= 0.0)


def test_middle_wind_state_remains_absent_without_three_level_gate():
    elevation = np.zeros((12, 24), dtype=np.float32)
    state = create_initial_state(elevation, planet_params=EARTH)
    evolved, _ = simulate_step(state, days=1.0, planet_params=EARTH)
    assert evolved.midlevel_wind_u is None
    assert evolved.midlevel_wind_v is None


def test_three_level_explicit_middle_wind_controls_both_pressure_interfaces():
    shape = (12, 24)
    planet = dataclasses.replace(
        EARTH,
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
        enable_three_level_pressure_column=True,
    )
    zeros = np.zeros(shape, dtype=np.float32)
    longitude_wave = np.sin(
        2.0 * np.pi * np.arange(shape[1], dtype=np.float32) / shape[1]
    )[None, :]
    mid_u = np.broadcast_to(20.0 * longitude_wave, shape).copy()
    common = dict(
        temperature=np.full(shape, 280.0, dtype=np.float32),
        humidity=np.full(shape, 0.012, dtype=np.float32),
        midlevel_humidity=np.full(shape, 0.004, dtype=np.float32),
        upperlevel_humidity=np.full(shape, 0.001, dtype=np.float32),
        midlevel_temperature=np.full(shape, 270.0, dtype=np.float32),
        upperlevel_temperature=np.full(shape, 250.0, dtype=np.float32),
        wind_u=zeros,
        wind_v=zeros,
        wind_u_aloft=zeros,
        wind_v_aloft=zeros,
        dt_days=0.25,
        planet_params=planet,
    )
    diagnostic_debug: dict = {}
    generate_precipitation(
        *shape, zeros, debug_fields=diagnostic_debug, **common
    )
    explicit_debug: dict = {}
    generate_precipitation(
        *shape, zeros,
        wind_u_midlevel=mid_u,
        wind_v_midlevel=zeros,
        debug_fields=explicit_debug,
        **common,
    )
    assert float(np.mean(np.abs(explicit_debug["midlevel_omega_pa_s"]))) > 0.0
    assert float(np.mean(np.abs(explicit_debug["upperlevel_omega_pa_s"]))) > 0.0
    assert not np.allclose(
        explicit_debug["midlevel_omega_pa_s"],
        diagnostic_debug["midlevel_omega_pa_s"],
    )
    assert not np.allclose(
        explicit_debug["upperlevel_omega_pa_s"],
        diagnostic_debug["upperlevel_omega_pa_s"],
    )


def test_three_level_diabatic_ascent_raises_tropical_condensation_driver():
    shape = (24, 32)
    common = dict(
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
        enable_three_level_pressure_column=True,
    )
    baseline = dataclasses.replace(EARTH, **common)
    forced = dataclasses.replace(
        EARTH, **common, three_level_diabatic_ascent_scale=0.5
    )
    inputs = dict(
        temperature=np.full(shape, 300.0, dtype=np.float32),
        humidity=np.full(shape, 0.018, dtype=np.float32),
        wind_u=np.zeros(shape, dtype=np.float32),
        wind_v=np.zeros(shape, dtype=np.float32),
        dt_days=1.0,
        day_of_year=80.0,
    )
    base_debug: dict = {}
    forced_debug: dict = {}
    generate_precipitation(*shape, np.zeros(shape, dtype=np.float32), planet_params=baseline, debug_fields=base_debug, **inputs)
    generate_precipitation(*shape, np.zeros(shape, dtype=np.float32), planet_params=forced, debug_fields=forced_debug, **inputs)
    equator = slice(shape[0] // 2 - 1, shape[0] // 2 + 1)
    assert float(np.mean(forced_debug["ascent"][equator])) > float(
        np.mean(base_debug["ascent"][equator])
    )


def test_energy_limited_evaporation_caps_raw_column_surface_source():
    shape = (12, 24)
    unconstrained_debug: dict = {}
    capped_debug: dict = {}
    common = dict(
        temperature=np.full(shape, 310.0, dtype=np.float32),
        humidity=np.zeros(shape, dtype=np.float32),
        wind_u=np.full(shape, 12.0, dtype=np.float32),
        wind_v=np.zeros(shape, dtype=np.float32),
        dt_days=1.0,
    )
    raw_planet = dataclasses.replace(EARTH, enable_prognostic_column_water=True)
    capped_planet = dataclasses.replace(
        raw_planet, enable_energy_limited_evaporation=True
    )
    generate_precipitation(
        *shape, np.zeros(shape, dtype=np.float32),
        planet_params=raw_planet, debug_fields=unconstrained_debug, **common,
    )
    generate_precipitation(
        *shape, np.zeros(shape, dtype=np.float32),
        planet_params=capped_planet, debug_fields=capped_debug, **common,
    )
    assert "energy_limited_evaporation_cap_mm_day" in capped_debug
    assert float(np.mean(capped_debug["ocean_evap"])) < float(
        np.mean(unconstrained_debug["ocean_evap"])
    )
    assert np.all(capped_debug["energy_limited_evaporation_fraction"] <= 1.0)

    longwave_debug: dict = {}
    longwave_planet = dataclasses.replace(
        capped_planet, evaporation_downwelling_longwave_w_m2=25.0
    )
    generate_precipitation(
        *shape, np.zeros(shape, dtype=np.float32),
        planet_params=longwave_planet, debug_fields=longwave_debug, **common,
    )
    assert float(np.mean(longwave_debug["energy_limited_evaporation_cap_mm_day"])) > float(
        np.mean(capped_debug["energy_limited_evaporation_cap_mm_day"])
    )

    dry_sky_debug: dict = {}
    humid_cloudy_debug: dict = {}
    diagnostic_planet = dataclasses.replace(
        capped_planet,
        enable_humidity_dependent_downwelling_longwave=True,
        evaporation_longwave_reference_emissivity=0.80,
    )
    generate_precipitation(
        *shape,
        np.full(shape, 0.001, dtype=np.float32),
        cloud_fraction=np.zeros(shape, dtype=np.float32),
        planet_params=diagnostic_planet,
        debug_fields=dry_sky_debug,
        **{key: value for key, value in common.items() if key != "humidity"},
    )
    generate_precipitation(
        *shape,
        np.full(shape, 0.015, dtype=np.float32),
        cloud_fraction=np.full(shape, 0.8, dtype=np.float32),
        planet_params=diagnostic_planet,
        debug_fields=humid_cloudy_debug,
        **{key: value for key, value in common.items() if key != "humidity"},
    )
    assert float(np.mean(humid_cloudy_debug["energy_limited_downwelling_longwave_w_m2"])) > float(
        np.mean(dry_sky_debug["energy_limited_downwelling_longwave_w_m2"])
    )


def test_three_level_upper_reservoir_condenses_and_closes_water_budget():
    shape = (6, 12)
    debug: dict = {}
    planet = dataclasses.replace(
        EARTH,
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
        enable_three_level_pressure_column=True,
    )
    result = generate_precipitation(
        *shape,
        np.zeros(shape, dtype=np.float32),
        temperature=np.full(shape, 300.0, dtype=np.float32),
        humidity=np.full(shape, 0.030, dtype=np.float32),
        midlevel_humidity=np.full(shape, 0.006, dtype=np.float32),
        upperlevel_humidity=np.full(shape, 0.015, dtype=np.float32),
        midlevel_temperature=np.full(shape, 275.0, dtype=np.float32),
        upperlevel_temperature=np.full(shape, 255.0, dtype=np.float32),
        wind_u=np.zeros(shape, dtype=np.float32),
        wind_v=np.zeros(shape, dtype=np.float32),
        dt_days=0.1,
        planet_params=planet,
        debug_fields=debug,
        return_condensate=True,
        return_midlevel_temperature=True,
        return_midlevel_humidity=True,
        return_upperlevel_state=True,
    )
    _, _, _, _, _, _, _, upper_temperature, upper_humidity = result
    assert np.all(debug["upperlevel_condensed_q"] > 0.0)
    assert np.all(upper_humidity < 0.015)
    assert np.all(upper_temperature > 255.0)
    np.testing.assert_allclose(
        debug["precipitation_final_mm_day"],
        debug["vapor_precipitation_mm_day"]
        + debug["lowerlevel_condensate_precipitation_mm_day"]
        + debug["midlevel_condensate_precipitation_mm_day"]
        + debug["upperlevel_condensate_precipitation_mm_day"],
        rtol=1e-6,
        atol=1e-6,
    )
    assert abs(float(debug["column_water_total_budget_relative_residual"])) < 1e-5


def test_three_level_upper_temperature_anomaly_feeds_back_to_resolved_air():
    elevation = np.zeros((12, 24), dtype=np.float32)
    planet = dataclasses.replace(
        EARTH,
        enable_prognostic_column_water=True,
        enable_prognostic_condensate=True,
        column_water_use_bulk_condensate_rainfall=True,
        enable_stability_aware_condensation=True,
        enable_two_layer_convective_adjustment=True,
        enable_three_level_pressure_column=True,
    )
    initial = create_initial_state(elevation, planet_params=planet)
    mid_reference = initial.air_temperature - 6.5e-3 * planet.stability_condensation_reference_height_m
    upper_reference = initial.air_temperature - 6.5e-3 * planet.three_level_upper_height_m
    base = initial._replace(midlevel_temperature=mid_reference)
    warm_upper = base._replace(upperlevel_temperature=upper_reference + 6.0)
    base_next, _ = simulate_step(base, days=1.0, planet_params=planet)
    warm_next, _ = simulate_step(warm_upper, days=1.0, planet_params=planet)
    assert float(np.mean(warm_next.air_temperature - base_next.air_temperature)) > 0.0
