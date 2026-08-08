from __future__ import annotations

import numpy as np

from circulation_diagnostics import (
    circulation_scorecard,
    hadley_edges_deg,
    jet_latitudes_deg,
    latitude_centres_deg,
    meridional_transport_diagnostics,
)


def test_jet_latitudes_find_each_hemisphere_peak():
    h, w = 32, 64
    lat = latitude_centres_deg(h)
    u = np.maximum(0.0, 30.0 - np.abs(np.abs(lat[:, None]) - 45.0))
    jets = jet_latitudes_deg(np.broadcast_to(u, (h, w)))
    assert abs(jets["nh_deg"] - 45.0) < 6.0
    assert abs(jets["sh_deg"] + 45.0) < 6.0


def test_hadley_edges_interpolate_flow_reversal():
    h, w = 32, 64
    lat = latitude_centres_deg(h)
    v = np.sign(lat[:, None]) * (25.0 - np.abs(lat[:, None]))
    edges = hadley_edges_deg(np.broadcast_to(v, (h, w)))
    assert abs(edges["nh_deg"] - 25.0) < 6.0
    assert abs(edges["sh_deg"] - 25.0) < 6.0


def test_scorecard_reports_layer_and_omega_values():
    zeros = np.zeros((16, 32), dtype=np.float32)
    report = circulation_scorecard(zeros, zeros, omega_lower_mid_pa_s=zeros, omega_mid_upper_pa_s=zeros)
    assert report["omega_lower_mid_pa_s"] == {"mean": 0.0, "rms": 0.0}


def test_meridional_transport_reports_cross_equatorial_energy_flux():
    h, w = 24, 48
    fields = np.ones((h, w))
    transport = meridional_transport_diagnostics(
        280.0 * fields,
        0.01 * fields,
        2.0 * fields,
        radius_m=6.371e6,
        surface_pressure_pa=101325.0,
        gravity_m_s2=9.81,
        cp_dry_j_kg_k=1004.0,
    )
    assert transport["cross_equatorial_total_energy_transport_pw"] > 0.0


def test_scorecard_includes_horizontal_divergence_and_transport_when_supplied():
    h, w = 24, 48
    zero = np.zeros((h, w))
    scorecard = circulation_scorecard(
        zero, zero,
        temperature_k=np.full((h, w), 280.0),
        humidity=np.full((h, w), 0.01),
        radius_m=6.371e6,
        surface_pressure_pa=101325.0,
        gravity_m_s2=9.81,
        cp_dry_j_kg_k=1004.0,
    )
    assert scorecard["horizontal_divergence_s"]["rms"] == 0.0
    assert "meridional_transport" in scorecard


def test_scorecard_uses_lower_only_transport_when_layer_thermodynamics_are_unavailable():
    h, w = 24, 48
    zero = np.zeros((h, w))
    scorecard = circulation_scorecard(
        zero, np.full((h, w), 2.0), upper_u=zero, upper_v=zero,
        temperature_k=np.full((h, w), 280.0), humidity=np.full((h, w), 0.01),
        radius_m=6.371e6, surface_pressure_pa=101325.0,
        gravity_m_s2=9.81, cp_dry_j_kg_k=1004.0,
        layer_mass_fractions=(0.4, 0.35, 0.25),
    )
    assert scorecard["meridional_transport"]["cross_equatorial_total_energy_transport_pw"] > 0.0


def test_layer_weighted_transport_cancels_opposing_meridional_levels():
    h, w = 24, 48
    fields = np.ones((h, w))
    transport = meridional_transport_diagnostics(
        280.0 * fields, 0.01 * fields, 2.0 * fields,
        radius_m=6.371e6, surface_pressure_pa=101325.0,
        gravity_m_s2=9.81, cp_dry_j_kg_k=1004.0,
        midlevel_temperature_k=280.0 * fields,
        midlevel_humidity=0.01 * fields,
        midlevel_meridional_wind_m_s=-2.0 * fields,
        upperlevel_temperature_k=280.0 * fields,
        upperlevel_humidity=0.01 * fields,
        upperlevel_meridional_wind_m_s=-2.0 * fields,
        layer_mass_fractions=(0.5, 0.25, 0.25),
    )
    assert abs(transport["cross_equatorial_total_energy_transport_pw"]) < 1e-9


def test_geopotential_term_adds_to_dry_static_energy_transport():
    h, w = 24, 48
    fields = np.ones((h, w))
    without_gz = meridional_transport_diagnostics(
        280.0 * fields, 0.01 * fields, 2.0 * fields,
        radius_m=6.371e6, surface_pressure_pa=101325.0,
        gravity_m_s2=9.81, cp_dry_j_kg_k=1004.0,
    )
    with_gz = meridional_transport_diagnostics(
        280.0 * fields, 0.01 * fields, 2.0 * fields,
        radius_m=6.371e6, surface_pressure_pa=101325.0,
        gravity_m_s2=9.81, cp_dry_j_kg_k=1004.0,
        lower_geopotential_m2_s2=50_000.0 * fields,
    )
    assert with_gz["cross_equatorial_dry_static_energy_transport_pw"] > (
        without_gz["cross_equatorial_dry_static_energy_transport_pw"] + 1e-9
    )
    assert with_gz["cross_equatorial_total_energy_transport_pw"] > (
        without_gz["cross_equatorial_total_energy_transport_pw"] + 1e-9
    )


def test_geopotential_term_requires_matching_shape():
    h, w = 24, 48
    fields = np.ones((h, w))
    try:
        meridional_transport_diagnostics(
            280.0 * fields, 0.01 * fields, 2.0 * fields,
            radius_m=6.371e6, surface_pressure_pa=101325.0,
            gravity_m_s2=9.81, cp_dry_j_kg_k=1004.0,
            lower_geopotential_m2_s2=np.ones((h, w + 1)),
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_geopotential_term_per_layer_is_zero_when_only_lower_supplied():
    h, w = 24, 48
    fields = np.ones((h, w))
    transport = meridional_transport_diagnostics(
        280.0 * fields, 0.01 * fields, 2.0 * fields,
        radius_m=6.371e6, surface_pressure_pa=101325.0,
        gravity_m_s2=9.81, cp_dry_j_kg_k=1004.0,
        midlevel_temperature_k=280.0 * fields,
        midlevel_humidity=0.01 * fields,
        midlevel_meridional_wind_m_s=-2.0 * fields,
        upperlevel_temperature_k=280.0 * fields,
        upperlevel_humidity=0.01 * fields,
        upperlevel_meridional_wind_m_s=-2.0 * fields,
        layer_mass_fractions=(0.5, 0.25, 0.25),
        lower_geopotential_m2_s2=50_000.0 * fields,
    )
    # Mid/upper geopotential defaults to zero, but the lower-layer term still
    # contributes -- opposing-branch cancellation is broken by an asymmetric gz.
    assert abs(transport["cross_equatorial_total_energy_transport_pw"]) > 1e-9
