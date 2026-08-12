from __future__ import annotations

import numpy as np

from two_layer_overturning_diagnostics import diagnose_two_layer_overturning_state


def test_normal_two_layer_diagnostic_reports_missing_thermodynamic_level():
    shape = (24, 48)
    report = diagnose_two_layer_overturning_state(
        np.full(shape, 290.0), np.full(shape, 288.0), np.full(shape, 4.0),
        np.full(shape, 1.5), np.full(shape, -1.0), hadley_edge_deg=24.0,
    )
    assert report["upper_temperature_available"] is False
    assert report["diagnosed_mse_strength_available"] is False
    assert report["tropical_latent_heating_w_m2"] > 100.0
    assert report["two_layer_mass_flux_available"] is True
    assert abs(report["two_layer_mass_flux_residual_mean_m_s"]) < 1e-12


def test_two_layer_diagnostic_reports_upper_temperature_when_present():
    shape = (24, 48)
    report = diagnose_two_layer_overturning_state(
        np.full(shape, 290.0), np.full(shape, 288.0), np.zeros(shape),
        None, None, hadley_edge_deg=24.0, upper_temperature_k=np.full(shape, 260.0),
    )
    assert report["upper_temperature_available"] is True
    assert np.isclose(report["tropical_air_minus_upper_k"], 28.0)
    assert report["two_layer_mass_flux_available"] is False
