from __future__ import annotations

import numpy as np
import pytest

from atmosphere import generate_precipitation
from condensate import (
    column_water_forcing_from_boundary_fluxes,
    column_water_forcing_from_budget,
    evolve_bulk_condensate,
    evolve_pressure_condensate_reservoirs,
    transport_pressure_condensate_reservoirs,
    separate_cloud_and_hydrometeor_reservoirs,
    simplified_betts_miller_condensation,
    stability_aware_condensation,
)
from planet_params import PlanetParams


def test_bulk_condensate_conserves_column_water():
    vapor = np.array([[0.012, 0.004]])
    condensate = np.array([[0.001, 0.002]])
    q_next, qc_next, rainout = evolve_bulk_condensate(
        vapor, np.array([[0.012, 0.010]]), np.array([[2.0, 0.5]]), condensate,
        dt_days=1.0, condensation_timescale_days=0.5, fallout_timescale_days=1.0,
    )
    assert np.allclose(q_next + qc_next + rainout, vapor + condensate)
    assert np.all(q_next >= 0.0)
    assert np.all(qc_next >= 0.0)
    assert np.all(rainout >= 0.0)


def test_bulk_condensate_does_not_condense_static_or_dry_air():
    vapor = np.full((2, 2), 0.004)
    q_next, qc_next, rainout = evolve_bulk_condensate(
        vapor, np.full((2, 2), 0.010), np.zeros((2, 2)), None,
        dt_days=1.0, condensation_timescale_days=0.5, fallout_timescale_days=1.0,
    )
    assert np.array_equal(q_next, vapor)
    assert np.array_equal(qc_next, np.zeros_like(vapor))
    assert np.array_equal(rainout, np.zeros_like(vapor))


def test_simplified_betts_miller_relaxes_only_ascending_excess_and_closes_water():
    vapor = np.array([[0.016, 0.016]], dtype=np.float64)
    qsat = np.full_like(vapor, 0.020)
    ascent = np.array([[2.0, 0.0]], dtype=np.float64)
    q_next, cloud_next, rainout, condensed = simplified_betts_miller_condensation(
        vapor,
        qsat,
        ascent,
        np.full_like(vapor, 0.001),
        dt_days=0.25,
        relaxation_hours=2.0,
        target_relative_humidity=0.70,
        fallout_timescale_days=1.0,
    )
    assert condensed[0, 0] > 0.0
    assert condensed[0, 1] == 0.0
    np.testing.assert_allclose(
        q_next + cloud_next + rainout,
        vapor + 0.001,
        atol=1e-12,
    )


def test_stability_closure_routes_supersaturation_through_condensate():
    vapor = np.full((2, 2), 0.020)
    qsat = np.full((2, 2), 0.010)
    q_next, condensate_next, rainout, _, _ = stability_aware_condensation(
        vapor,
        qsat,
        np.full((2, 2), 280.0),
        np.zeros((2, 2)),
        None,
        surface_pressure_hpa=1013.25,
        dt_days=0.25,
        condensation_timescale_days=0.5,
        fallout_timescale_days=1.0,
    )
    # No resolved ascent is required to correct a genuinely supersaturated
    # transported parcel, but its water stays in the cloud reservoir until
    # fallout rather than being emitted as an instantaneous vapor-rain sink.
    np.testing.assert_allclose(q_next, qsat)
    assert np.all(condensate_next > 0.0)
    assert np.all(rainout > 0.0)
    assert np.all(rainout < vapor - qsat)
    np.testing.assert_allclose(q_next + condensate_next + rainout, vapor)


def test_separate_cloud_hydrometeor_reservoirs_conserve_condensate():
    cloud, hydrometeors, fallout = separate_cloud_and_hydrometeor_reservoirs(
        np.full((2, 2), 0.001),
        np.full((2, 2), 0.002),
        np.full((2, 2), 0.004),
        dt_days=0.25,
        autoconversion_timescale_days=0.25,
        fallout_timescale_days=0.5,
        cloud_retention_q=0.001,
    )
    np.testing.assert_allclose(cloud + hydrometeors + fallout, 0.007)
    assert np.all(cloud >= 0.001)
    assert np.all(hydrometeors >= 0.0)


def test_pressure_reservoir_transition_conserves_cloud_hydrometeor_mass():
    cloud_in = np.full((2, 2), 1.0)
    hydrometeors_in = np.full((2, 2), 2.0)
    condensed = np.full((2, 2), 4.0)
    result = evolve_pressure_condensate_reservoirs(
        cloud_in, hydrometeors_in, condensed,
        dt_days=0.25, autoconversion_timescale_days=0.25,
        fallout_timescale_days=0.5, cloud_retention_kg_m2=1.0,
    )
    np.testing.assert_allclose(
        result.cloud_condensate_kg_m2
        + result.precipitating_hydrometeors_kg_m2
        + result.fallout_kg_m2,
        cloud_in + hydrometeors_in + condensed,
        atol=1e-6,
    )
    assert np.all(result.autoconverted_kg_m2 > 0.0)


def test_pressure_reservoir_transport_keeps_each_reservoir_mass():
    cloud = np.array([[1.0, 2.0, 3.0, 2.0], [2.0, 3.0, 1.0, 2.0]])
    hydrometeors = 0.5 * cloud
    result = transport_pressure_condensate_reservoirs(
        cloud, hydrometeors, np.full_like(cloud, 2.0), np.zeros_like(cloud),
        dt_days=0.25, fallout_timescale_days=1.0, cloud_transport_scale=1.0,
        transport_hydrometeors=True, dx_m=100_000.0, dy_m=100_000.0,
        cell_area_m2=1.0e10, x_face_length_m=100_000.0, y_face_length_m=100_000.0,
    )
    np.testing.assert_allclose(np.sum(result.cloud_condensate_kg_m2), np.sum(cloud))
    np.testing.assert_allclose(
        np.sum(result.precipitating_hydrometeors_kg_m2), np.sum(hydrometeors),
    )


def test_column_water_forcing_uses_only_external_boundary_mass_fluxes():
    forcing = column_water_forcing_from_boundary_fluxes(
        np.array([[4.0, 1.0]]), np.array([[1.0, 3.0]]), dt_seconds=2.0,
    )
    np.testing.assert_allclose(forcing, np.array([[1.5, -1.0]]))


def test_column_water_forcing_from_budget_includes_storage_tendency():
    forcing = column_water_forcing_from_budget(
        np.array([[5.0, 4.0]]), np.array([[1.0, 1.0]]),
        np.array([[10.0, 10.0]]), np.array([[12.0, 7.0]]), dt_seconds=2.0,
    )
    # (source - fallout - (after - before)) / dt
    np.testing.assert_allclose(forcing, np.array([[1.0, 3.0]]))


@pytest.mark.parametrize("name,value", [("dt_days", 0.0), ("condensation_timescale_days", 0.0)])
def test_bulk_condensate_rejects_nonphysical_timescales(name, value):
    kwargs = dict(dt_days=1.0, condensation_timescale_days=1.0, fallout_timescale_days=1.0)
    kwargs[name] = value
    with pytest.raises(ValueError):
        evolve_bulk_condensate(
            np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)), None, **kwargs
        )


def test_precipitation_threads_an_enabled_condensate_reservoir():
    shape = (4, 8)
    result = generate_precipitation(
        *shape,
        np.zeros(shape),
        temperature=np.full(shape, 295.0),
        wind_u=np.full(shape, 4.0),
        wind_v=np.full(shape, 1.0),
        humidity=np.full(shape, 0.014),
        condensate=np.full(shape, 0.001),
        dt_days=1.0,
        planet_params=PlanetParams(enable_prognostic_condensate=True),
        return_condensate=True,
    )
    precipitation, vapor, _, _, condensate = result
    assert precipitation.shape == shape
    assert vapor.shape == shape
    assert condensate.shape == shape
    assert np.all(np.isfinite(condensate))
    assert np.all(condensate >= 0.0)
