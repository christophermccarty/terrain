from __future__ import annotations

import numpy as np

from pressure_column import (
    evolve_closed_three_level_thermodynamic_column,
    evolve_three_level_column,
    transport_closed_three_level_mse,
)


def test_three_level_pressure_column_conserves_total_vapor_for_opposing_interfaces():
    shape = (4, 8)
    total = np.full(shape, 0.018, dtype=np.float32)
    # Lower->mid ascent on the left and mid->upper ascent on the right;
    # opposite signs elsewhere exercise both upward and downward donor paths.
    lower_div = np.tile(np.array([-1.0e-6, -1.0e-6, 1.0e-6, 1.0e-6, -1.0e-6, -1.0e-6, 1.0e-6, 1.0e-6]), (4, 1))
    mid_div = -lower_div
    upper_div = lower_div
    step = evolve_three_level_column(
        total,
        np.full(shape, 300.0),
        lower_div,
        mid_div,
        upper_div,
        midlevel_humidity=np.full(shape, 0.004),
        upperlevel_humidity=np.full(shape, 0.003),
        dt_days=0.25,
    )
    np.testing.assert_allclose(
        step.lower_humidity + step.midlevel_humidity + step.upperlevel_humidity,
        total,
        atol=3e-9,
    )
    assert abs(step.relative_water_residual) < 1e-12
    assert float(np.min(step.omega_lower_mid_pa_s)) < 0.0
    assert float(np.max(step.omega_lower_mid_pa_s)) > 0.0
    assert float(np.min(step.omega_mid_upper_pa_s)) < 0.0
    assert float(np.max(step.omega_mid_upper_pa_s)) > 0.0


def test_three_level_pressure_column_relaxes_both_temperatures_to_lapse_profile():
    shape = (2, 3)
    step = evolve_three_level_column(
        np.full(shape, 0.012),
        np.full(shape, 300.0),
        np.zeros(shape),
        np.zeros(shape),
        np.zeros(shape),
        midlevel_temperature_k=np.full(shape, 260.0),
        upperlevel_temperature_k=np.full(shape, 240.0),
        dt_days=10.0,
        thermal_relaxation_days=10.0,
    )
    assert float(np.mean(step.midlevel_temperature)) > 260.0
    assert float(np.mean(step.upperlevel_temperature)) > 240.0
    assert float(np.mean(step.upperlevel_temperature)) < float(np.mean(step.midlevel_temperature))


def test_mass_closure_uses_upper_divergence_to_close_column_residual():
    shape = (2, 4)
    step = evolve_three_level_column(
        np.full(shape, 0.012),
        np.full(shape, 300.0),
        np.full(shape, 1.0e-6),
        np.full(shape, 2.0e-6),
        np.zeros(shape),
        dt_days=0.1,
        enforce_column_mass_closure=True,
    )
    # Upper divergence is -(0.4*1 + 0.35*2)/0.25 = -4.4 micro-s^-1,
    # therefore omega(mid, upper)=0.5*30000*(2 - -4.4) micro-Pa/s.
    np.testing.assert_allclose(step.omega_mid_upper_pa_s, 0.096, atol=1e-7)


def _closed_column_inputs(shape=(2, 3)):
    return dict(
        lower_humidity=np.full(shape, 0.014),
        midlevel_humidity=np.full(shape, 0.007),
        upperlevel_humidity=np.full(shape, 0.002),
        lower_temperature_k=np.full(shape, 299.0),
        midlevel_temperature_k=np.full(shape, 273.0),
        upperlevel_temperature_k=np.full(shape, 243.0),
        omega_lower_mid_pa_s=np.array([[-0.020, 0.015, -0.010], [0.012, -0.018, 0.009]]),
        omega_mid_upper_pa_s=np.array([[0.012, -0.016, 0.008], [-0.010, 0.014, -0.006]]),
        dt_seconds=3600.0,
    )


def test_closed_thermodynamic_column_conserves_mass_weighted_water_and_energy():
    step = evolve_closed_three_level_thermodynamic_column(**_closed_column_inputs())

    assert abs(step.water_residual_kg_m2) < 1e-8
    assert abs(step.moist_static_energy_residual_j_m2) < 1e-4
    assert np.all(step.lower_humidity >= 0.0)
    assert np.all(step.midlevel_humidity >= 0.0)
    assert np.all(step.upperlevel_humidity >= 0.0)
    assert np.all(step.lower_mid_mass_flux_kg_m2_s > 0.0)
    assert np.all(step.mid_upper_mass_flux_kg_m2_s > 0.0)


def test_closed_thermodynamic_column_releases_latent_heat_without_creating_energy():
    shape = (2, 3)
    condensed = np.full(shape, 0.001)
    common = dict(
        lower_humidity=np.full(shape, 0.010),
        midlevel_humidity=np.full(shape, 0.005),
        upperlevel_humidity=np.full(shape, 0.002),
        lower_temperature_k=np.full(shape, 290.0),
        midlevel_temperature_k=np.full(shape, 270.0),
        upperlevel_temperature_k=np.full(shape, 245.0),
        omega_lower_mid_pa_s=np.zeros(shape),
        omega_mid_upper_pa_s=np.zeros(shape),
        dt_seconds=3600.0,
    )
    step = evolve_closed_three_level_thermodynamic_column(
        **common,
        condensed_specific_humidity=(condensed, np.zeros(shape), np.zeros(shape)),
    )

    np.testing.assert_allclose(
        step.lower_temperature - common["lower_temperature_k"],
        2.5e6 / 1004.0 * condensed,
        rtol=1e-5,
    )
    np.testing.assert_allclose(step.lower_humidity, common["lower_humidity"] - condensed)
    assert abs(step.water_residual_kg_m2) < 1e-8
    assert abs(step.moist_static_energy_residual_j_m2) < 1e-4


def test_closed_thermodynamic_column_radiation_is_an_explicit_energy_source_and_timestep_stable():
    shape = (2, 3)
    common = _closed_column_inputs(shape)
    common["omega_lower_mid_pa_s"] = np.zeros(shape)
    common["omega_mid_upper_pa_s"] = np.zeros(shape)
    radiation = (
        np.full(shape, 10.0),
        np.full(shape, -4.0),
        np.full(shape, 2.0),
    )
    full = evolve_closed_three_level_thermodynamic_column(
        **common, radiative_flux_w_m2=radiation
    )
    half = dict(common, dt_seconds=1800.0)
    first_half = evolve_closed_three_level_thermodynamic_column(
        **half, radiative_flux_w_m2=radiation
    )
    second_half = evolve_closed_three_level_thermodynamic_column(
        lower_humidity=first_half.lower_humidity,
        midlevel_humidity=first_half.midlevel_humidity,
        upperlevel_humidity=first_half.upperlevel_humidity,
        lower_temperature_k=first_half.lower_temperature,
        midlevel_temperature_k=first_half.midlevel_temperature,
        upperlevel_temperature_k=first_half.upperlevel_temperature,
        omega_lower_mid_pa_s=np.zeros(shape),
        omega_mid_upper_pa_s=np.zeros(shape),
        dt_seconds=1800.0,
        radiative_flux_w_m2=radiation,
    )

    assert abs(full.moist_static_energy_residual_j_m2) < 1e-4
    assert full.radiative_energy_input_j_m2 == 8.0 * 3600.0 * np.prod(shape)
    np.testing.assert_allclose(second_half.lower_temperature, full.lower_temperature, atol=3e-5)
    np.testing.assert_allclose(second_half.midlevel_temperature, full.midlevel_temperature, atol=3e-5)
    np.testing.assert_allclose(second_half.upperlevel_temperature, full.upperlevel_temperature, atol=3e-5)


def test_closed_column_mse_transport_follows_vapour_and_closes_energy():
    shape = (4, 8)
    lower_before = np.full(shape, 0.010)
    lower_before[2, 2] = 0.020
    winds = np.full(shape, 150.0)
    zeros = np.zeros(shape)
    from column_water import evolve_column_water

    lower_mass = 101325.0 / 9.80665 * 0.40
    transported_lower = evolve_column_water(
        lower_before * lower_mass, zeros, zeros, winds, zeros,
        dx_m=100_000.0, dy_m=100_000.0, dt_days=0.1,
    ).water_mm / lower_mass
    step = transport_closed_three_level_mse(
        lower_before, np.full(shape, 0.004), np.full(shape, 0.001),
        transported_lower, np.full(shape, 0.004), np.full(shape, 0.001),
        np.full(shape, 290.0), np.full(shape, 270.0), np.full(shape, 245.0),
        winds, zeros, zeros, zeros, zeros, zeros,
        lower_vapour_source_kg_kg_day=zeros, dt_days=0.1,
        dx_m=100_000.0, dy_m=100_000.0, cell_area_m2=1.0e10,
        x_face_length_m=100_000.0, y_face_length_m=np.full((5, 8), 100_000.0),
    )

    assert abs(step.energy_residual_j) < 1e-2
    assert abs(step.relative_energy_residual) < 1e-12
    # The vapour anomaly carries latent MSE eastward; recovering temperature
    # from transported MSE leaves a uniform-temperature tracer unchanged.
    np.testing.assert_allclose(step.lower_temperature, 290.0, atol=3e-5)


def test_closed_thermodynamic_column_rejects_unclosed_or_invalid_sources():
    common = _closed_column_inputs()
    with np.testing.assert_raises_regex(ValueError, "summing to one"):
        evolve_closed_three_level_thermodynamic_column(
            **common, layer_mass_fractions=(0.4, 0.3, 0.2)
        )
    with np.testing.assert_raises_regex(ValueError, "cannot exceed"):
        evolve_closed_three_level_thermodynamic_column(
            **common,
            condensed_specific_humidity=(
                np.full((2, 3), 0.020), np.zeros((2, 3)), np.zeros((2, 3)),
            ),
        )
