from __future__ import annotations

import numpy as np

from pressure_column import evolve_three_level_column


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
