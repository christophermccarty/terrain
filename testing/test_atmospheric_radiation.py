from __future__ import annotations

import numpy as np
import pytest

from atmospheric_radiation import (
    STEFAN_BOLTZMANN,
    effective_radiating_temperature,
    grey_surface_atmosphere_radiation,
    pressure_defined_temperature_profile,
    resolved_midlevel_emission_temperature,
)


@pytest.mark.parametrize(
    "surface, atmosphere, shortwave, emissivity",
    [
        (300.0, 260.0, 240.0, 0.8),
        (240.0, 220.0, 80.0, 0.3),
        (330.0, 280.0, 400.0, 1.0),
        (200.0, 190.0, 0.0, 0.0),
    ],
)
def test_grey_radiation_is_finite_and_conserves_column_energy(
    surface, atmosphere, shortwave, emissivity
):
    shape = (2, 3)
    result = grey_surface_atmosphere_radiation(
        np.full(shape, surface),
        np.full(shape, atmosphere),
        np.full(shape, shortwave),
        np.full(shape, emissivity),
    )
    np.testing.assert_allclose(
        result.surface_gain_w_m2 + result.atmospheric_gain_w_m2,
        result.toa_net_radiation_w_m2,
        atol=1e-12,
    )
    assert all(np.all(np.isfinite(field)) for field in result)


def test_effective_radiating_temperature_inverts_stefan_boltzmann():
    temperature = np.array([200.0, 255.0, 310.0])
    outgoing = STEFAN_BOLTZMANN * temperature**4
    np.testing.assert_allclose(
        effective_radiating_temperature(outgoing), temperature, rtol=1e-14
    )


def test_pressure_defined_profile_conserves_dry_potential_temperature():
    lower = np.array([[296.0, 240.0], [330.0, 200.0]])
    profile = pressure_defined_temperature_profile(
        lower, 101_325.0, 35_000.0, 30_000.0
    )
    exponent = 287.05 / 1004.0
    np.testing.assert_allclose(profile.midlevel_pressure_pa, 66_325.0)
    np.testing.assert_allclose(profile.upperlevel_pressure_pa, 36_325.0)
    np.testing.assert_allclose(
        profile.midlevel_temperature_k
        / (profile.midlevel_pressure_pa / 101_325.0) ** exponent,
        lower,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        profile.upperlevel_temperature_k
        / (profile.upperlevel_pressure_pa / 101_325.0) ** exponent,
        lower,
        rtol=1e-14,
    )
    assert 261.0 < profile.midlevel_temperature_k[0, 0] < 263.0
    assert np.all(profile.upperlevel_temperature_k < profile.midlevel_temperature_k)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"surface_pressure_pa": 0.0},
        {"lower_mid_pressure_depth_pa": 0.0},
        {"lower_mid_pressure_depth_pa": 80_000.0, "mid_upper_pressure_depth_pa": 30_000.0},
        {"gas_constant_dry_air_j_kg_k": 1004.0},
    ],
)
def test_pressure_defined_profile_rejects_invalid_columns(kwargs):
    inputs = {
        "lower_temperature_k": np.full((2, 2), 296.0),
        "surface_pressure_pa": 101_325.0,
        "lower_mid_pressure_depth_pa": 35_000.0,
        "mid_upper_pressure_depth_pa": 30_000.0,
    }
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        pressure_defined_temperature_profile(**inputs)


def test_resolved_emission_temperature_has_no_free_air_fallback():
    with pytest.raises(ValueError, match="resolved midlevel"):
        resolved_midlevel_emission_temperature(None, expected_shape=(2, 3))


def test_resolved_emission_temperature_preserves_the_explicit_state():
    midlevel = np.full((2, 3), 262.0)
    emission = resolved_midlevel_emission_temperature(
        midlevel, expected_shape=midlevel.shape
    )
    np.testing.assert_array_equal(emission, midlevel)
    with pytest.raises(ValueError, match="unexpected shape"):
        resolved_midlevel_emission_temperature(midlevel, expected_shape=(3, 2))
