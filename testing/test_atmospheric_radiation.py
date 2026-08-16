from __future__ import annotations

import numpy as np
import pytest

from atmospheric_radiation import (
    STEFAN_BOLTZMANN,
    atmospheric_emissivity_for_target_olr,
    effective_radiating_temperature,
    grey_radiative_convective_equilibrium_temperatures,
    grey_surface_atmosphere_radiation,
    pressure_defined_temperature_profile,
    pressure_split_emissivities_from_optical_depth,
    resolved_midlevel_emission_temperature,
    two_layer_grey_radiation,
    two_layer_optical_depth_for_target_olr,
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


def test_diagnosed_emissivity_reproduces_target_olr_exactly():
    surface = np.array([290.0, 300.0, 320.0])
    atmosphere = np.array([250.0, 260.0, 280.0])
    expected_emissivity = np.array([0.0, 0.45, 1.0])
    surface_emission = STEFAN_BOLTZMANN * surface**4
    atmospheric_emission = STEFAN_BOLTZMANN * atmosphere**4
    target = (
        (1.0 - expected_emissivity) * surface_emission
        + expected_emissivity * atmospheric_emission
    )
    diagnosed = atmospheric_emissivity_for_target_olr(
        surface, atmosphere, target
    )
    np.testing.assert_allclose(diagnosed, expected_emissivity, atol=2e-15)
    result = grey_surface_atmosphere_radiation(
        surface, atmosphere, np.full(3, 240.0), diagnosed
    )
    np.testing.assert_allclose(result.outgoing_longwave_w_m2, target, atol=1e-12)


def test_diagnosed_emissivity_handles_isothermal_column_without_division():
    temperature = np.full((2, 3), 260.0)
    target = STEFAN_BOLTZMANN * temperature**4
    emissivity = atmospheric_emissivity_for_target_olr(
        temperature, temperature, target
    )
    np.testing.assert_array_equal(emissivity, 0.0)
    assert np.all(np.isfinite(emissivity))


def test_diagnosed_emissivity_rejects_unrepresentable_target_olr():
    surface = np.full((2, 2), 300.0)
    atmosphere = np.full((2, 2), 260.0)
    with pytest.raises(ValueError, match="outside"):
        atmospheric_emissivity_for_target_olr(
            surface, atmosphere, np.full((2, 2), 100.0)
        )


@pytest.mark.parametrize(
    "surface,middle,upper,shortwave,middle_e,upper_e",
    [
        (300.0, 265.0, 225.0, 240.0, 0.7, 0.5),
        (220.0, 205.0, 180.0, 0.0, 0.0, 0.0),
        (350.0, 300.0, 240.0, 500.0, 1.0, 1.0),
    ],
)
def test_two_layer_grey_budget_is_finite_and_conservative(
    surface, middle, upper, shortwave, middle_e, upper_e
):
    shape = (2, 3)
    result = two_layer_grey_radiation(
        np.full(shape, surface),
        np.full(shape, middle),
        np.full(shape, upper),
        np.full(shape, shortwave),
        np.full(shape, middle_e),
        np.full(shape, upper_e),
    )
    np.testing.assert_allclose(
        result.surface_gain_w_m2
        + result.midlevel_gain_w_m2
        + result.upperlevel_gain_w_m2,
        result.toa_net_radiation_w_m2,
        atol=2e-12,
    )
    assert all(np.all(np.isfinite(field)) for field in result)
    np.testing.assert_array_equal(
        result.lower_downward_emission_w_m2,
        result.lower_upward_emission_w_m2,
    )
    np.testing.assert_array_equal(
        result.upper_downward_emission_w_m2,
        result.upper_upward_emission_w_m2,
    )


def test_pressure_split_optical_depth_reproduces_target_olr():
    surface = np.array([[300.0, 310.0], [280.0, 330.0]])
    middle = 0.90 * surface
    upper = 0.76 * surface
    target_temperature = 0.86 * surface
    target = STEFAN_BOLTZMANN * target_temperature**4
    closure = two_layer_optical_depth_for_target_olr(
        surface, middle, upper, target, 35_000.0, 30_000.0
    )
    result = two_layer_grey_radiation(
        surface,
        middle,
        upper,
        np.full(surface.shape, 240.0),
        closure.midlevel_emissivity,
        closure.upperlevel_emissivity,
    )
    np.testing.assert_allclose(result.outgoing_longwave_w_m2, target, rtol=2e-14)
    assert np.all(closure.total_optical_depth > 0.0)
    assert np.all((closure.midlevel_emissivity > 0.0) & (closure.midlevel_emissivity < 1.0))
    assert np.all((closure.upperlevel_emissivity > 0.0) & (closure.upperlevel_emissivity < 1.0))


def test_pressure_split_optical_depth_accepts_surface_inversion():
    shape = (2, 2)
    surface = np.full(shape, 300.0)
    middle = np.full(shape, 265.0)
    upper = np.full(shape, 225.0)
    inverted_surface = np.full(shape, 250.0)
    target = STEFAN_BOLTZMANN * np.full(shape, 240.0) ** 4
    closure = two_layer_optical_depth_for_target_olr(
        inverted_surface, middle, upper, target, 35_000.0, 30_000.0
    )
    result = two_layer_grey_radiation(
        inverted_surface, middle, upper, np.zeros(shape),
        closure.midlevel_emissivity, closure.upperlevel_emissivity,
    )
    np.testing.assert_allclose(result.outgoing_longwave_w_m2, target, rtol=2e-14)


def test_pressure_split_optical_depth_rejects_upper_inversion_and_out_of_range_target():
    shape = (2, 2)
    surface = np.full(shape, 300.0)
    middle = np.full(shape, 265.0)
    upper = np.full(shape, 225.0)
    with pytest.raises(ValueError, match="upper level"):
        two_layer_optical_depth_for_target_olr(
            surface, middle, np.full(shape, 270.0), np.full(shape, 240.0),
            35_000.0, 30_000.0,
        )
    with pytest.raises(ValueError, match="outside"):
        two_layer_optical_depth_for_target_olr(
            surface, middle, upper, np.full(shape, 50.0), 35_000.0, 30_000.0
        )


def test_pressure_split_optical_depth_reports_explicit_opaque_limit():
    shape = (2, 2)
    surface = np.full(shape, 300.0)
    middle = np.full(shape, 265.0)
    upper = np.full(shape, 225.0)
    target = STEFAN_BOLTZMANN * np.full(shape, 210.0) ** 4
    closure = two_layer_optical_depth_for_target_olr(
        surface, middle, upper, target, 35_000.0, 30_000.0,
        allow_opaque_limit=True,
    )
    np.testing.assert_array_equal(closure.opaque_limited, True)
    np.testing.assert_array_equal(closure.midlevel_emissivity, 1.0)
    np.testing.assert_array_equal(closure.upperlevel_emissivity, 1.0)
    expected_residual = STEFAN_BOLTZMANN * (225.0**4 - 210.0**4)
    np.testing.assert_allclose(closure.target_olr_residual_w_m2, expected_residual)
    assert np.all(np.isfinite(closure.total_optical_depth))


def test_pressure_split_emissivity_preserves_total_transmission():
    optical_depth = np.array([0.0, 0.5, 3.0, 64.0])
    split = pressure_split_emissivities_from_optical_depth(
        optical_depth, 35_000.0, 30_000.0
    )
    combined_transmission = (
        (1.0 - split.midlevel_emissivity)
        * (1.0 - split.upperlevel_emissivity)
    )
    np.testing.assert_allclose(
        combined_transmission, np.exp(-optical_depth), atol=1e-27
    )


def _layer_center_potential_temperatures(lower, middle, upper):
    """Potential temperatures at the omega diagnostic's layer centres."""
    edges = np.array((1.0, 0.60, 0.25, 0.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    kappa = 287.05 / 1004.0
    return (
        lower * (1.0 / centers[0]) ** kappa,
        middle * (1.0 / centers[1]) ** kappa,
        upper * (1.0 / centers[2]) ** kappa,
    )


def test_grey_equilibrium_has_zero_layer_gains_where_adiabatically_admissible():
    shape = (2, 2)
    surface = np.full(shape, 300.0)
    lower = np.full(shape, 290.0)
    optical_depth = np.full(shape, 2.0)
    profile = grey_radiative_convective_equilibrium_temperatures(
        surface, lower, optical_depth, 35_000.0, 30_000.0
    )
    assert not np.any(profile.adiabatic_limited_midlevel)
    assert not np.any(profile.adiabatic_limited_upperlevel)
    assert 264.0 < profile.midlevel_temperature_k[0, 0] < 267.0
    assert 233.0 < profile.upperlevel_temperature_k[0, 0] < 236.0
    split = pressure_split_emissivities_from_optical_depth(
        optical_depth, 35_000.0, 30_000.0
    )
    result = two_layer_grey_radiation(
        surface,
        profile.midlevel_temperature_k,
        profile.upperlevel_temperature_k,
        np.full(shape, 240.0),
        split.midlevel_emissivity,
        split.upperlevel_emissivity,
    )
    np.testing.assert_allclose(result.midlevel_gain_w_m2, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.upperlevel_gain_w_m2, 0.0, atol=1e-10)


def test_grey_equilibrium_adiabatic_limiter_binds_for_warm_lower_cold_surface():
    shape = (2, 2)
    surface = np.full(shape, 250.0)
    lower = np.full(shape, 290.0)
    optical_depth = np.full(shape, 2.0)
    profile = grey_radiative_convective_equilibrium_temperatures(
        surface, lower, optical_depth, 35_000.0, 30_000.0
    )
    assert np.all(profile.adiabatic_limited_midlevel)
    assert not np.any(profile.adiabatic_limited_upperlevel)
    kappa = 287.05 / 1004.0
    expected_floor = 290.0 * (0.425 / 0.8) ** kappa
    np.testing.assert_allclose(
        profile.midlevel_temperature_k, expected_floor, rtol=1e-14
    )
    # The clamped state is warmer than the radiative equilibrium, so a bounded
    # residual cooling gain remains -- but the profile is statically stable.
    split = pressure_split_emissivities_from_optical_depth(
        optical_depth, 35_000.0, 30_000.0
    )
    result = two_layer_grey_radiation(
        surface,
        profile.midlevel_temperature_k,
        profile.upperlevel_temperature_k,
        np.zeros(shape),
        split.midlevel_emissivity,
        split.upperlevel_emissivity,
    )
    assert np.all(result.midlevel_gain_w_m2 < 0.0)
    theta = _layer_center_potential_temperatures(
        lower, profile.midlevel_temperature_k, profile.upperlevel_temperature_k
    )
    assert np.all(theta[1] >= theta[0] - 1e-9)
    assert np.all(theta[2] >= theta[1] - 1e-9)


def test_grey_equilibrium_is_statically_stable_for_extreme_contrasts():
    rng = np.random.default_rng(0)
    shape = (8, 16)
    surface = rng.uniform(220.0, 310.0, shape)
    lower = rng.uniform(230.0, 320.0, shape)
    optical_depth = rng.uniform(0.0, 8.0, shape)
    profile = grey_radiative_convective_equilibrium_temperatures(
        surface, lower, optical_depth, 35_000.0, 30_000.0
    )
    theta = _layer_center_potential_temperatures(
        lower, profile.midlevel_temperature_k, profile.upperlevel_temperature_k
    )
    assert np.all(theta[1] >= theta[0] - 1e-9)
    assert np.all(theta[2] >= theta[1] - 1e-9)
    # Unclamped cells must sit exactly at the zero-gain equilibrium.
    split = pressure_split_emissivities_from_optical_depth(
        optical_depth, 35_000.0, 30_000.0
    )
    result = two_layer_grey_radiation(
        surface,
        profile.midlevel_temperature_k,
        profile.upperlevel_temperature_k,
        np.zeros(shape),
        split.midlevel_emissivity,
        split.upperlevel_emissivity,
    )
    unclamped = ~profile.adiabatic_limited_midlevel
    np.testing.assert_allclose(
        result.midlevel_gain_w_m2[unclamped], 0.0, atol=1e-8
    )
    unclamped_upper = ~profile.adiabatic_limited_upperlevel
    np.testing.assert_allclose(
        result.upperlevel_gain_w_m2[unclamped_upper], 0.0, atol=1e-8
    )


@pytest.mark.parametrize("optical_depth", [0.0, 64.0])
def test_grey_equilibrium_is_finite_at_optical_depth_edges(optical_depth):
    shape = (2, 3)
    profile = grey_radiative_convective_equilibrium_temperatures(
        np.full(shape, 300.0),
        np.full(shape, 290.0),
        np.full(shape, optical_depth),
        35_000.0,
        30_000.0,
    )
    assert np.all(np.isfinite(profile.midlevel_temperature_k))
    assert np.all(np.isfinite(profile.upperlevel_temperature_k))
    assert np.all(profile.midlevel_temperature_k > 0.0)
    assert np.all(profile.upperlevel_temperature_k > 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_optical_depth": np.full((2, 2), -1.0)},
        {"surface_temperature_k": np.full((2, 2), 0.0)},
        {"lower_temperature_k": np.full((2, 2), -5.0)},
        {"gas_constant_dry_air_j_kg_k": 1004.0},
        {"layer_mass_fractions": (0.5, 0.3, 0.1)},
        {"total_optical_depth": np.full((3, 3), 1.0)},
    ],
)
def test_grey_equilibrium_rejects_invalid_inputs(kwargs):
    inputs = {
        "surface_temperature_k": np.full((2, 2), 300.0),
        "lower_temperature_k": np.full((2, 2), 290.0),
        "total_optical_depth": np.full((2, 2), 2.0),
        "lower_mid_pressure_depth_pa": 35_000.0,
        "mid_upper_pressure_depth_pa": 30_000.0,
    }
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        grey_radiative_convective_equilibrium_temperatures(**inputs)
