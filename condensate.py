"""Mass-conserving bulk vapor/condensate/rain microphysics.

The climate model's humidity field is a column specific-humidity proxy.  This
module adds the minimal prognostic complement needed for a physical rainfall
closure: vapor condenses in saturated, ascending air; condensate persists; and
fallout transfers it to precipitation.  All quantities are expressed as the
same column mixing-ratio proxy, so each call conserves ``vapor + condensate +
rainout`` to floating-point precision.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class PressureCondensateReservoirStep(NamedTuple):
    """Pressure-column cloud and hydrometeor reservoirs after one transition.

    All fields are column masses in kg m-2 (numerically mm).  Keeping this
    transition separate from the vapour phase change lets a coupled circulation
    solve use the same reservoir mass source/sink as the runtime adapter.
    """

    cloud_condensate_kg_m2: np.ndarray
    precipitating_hydrometeors_kg_m2: np.ndarray
    fallout_kg_m2: np.ndarray
    autoconverted_kg_m2: np.ndarray


class PressureCondensateTransportStep(NamedTuple):
    """Cloud and hydrometeor reservoirs after their horizontal transport leg."""

    cloud_condensate_kg_m2: np.ndarray
    precipitating_hydrometeors_kg_m2: np.ndarray
    cloud_relative_residual: float
    hydrometeor_relative_residual: float | None


def evolve_bulk_condensate(
    vapor: np.ndarray,
    saturation_vapor: np.ndarray,
    ascent: np.ndarray,
    condensate: np.ndarray | None,
    *,
    dt_days: float,
    condensation_timescale_days: float,
    fallout_timescale_days: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(vapor_next, condensate_next, rainout_mixing_ratio)``.

    Condensation activates smoothly above 80% relative humidity and strengthens
    with resolved ascent.  The activation represents cooling along an ascent,
    not spontaneous condensation of unsaturated static air.  Condensate then
    falls out exponentially, which is stable for every caller timestep.
    """
    if dt_days <= 0.0:
        raise ValueError("dt_days must be positive")
    if condensation_timescale_days <= 0.0 or fallout_timescale_days <= 0.0:
        raise ValueError("condensation and fallout timescales must be positive")
    q = np.clip(np.asarray(vapor, dtype=np.float64), 0.0, None)
    qsat = np.maximum(np.asarray(saturation_vapor, dtype=np.float64), 1e-12)
    updraft = np.clip(np.asarray(ascent, dtype=np.float64), 0.0, None)
    if q.shape != qsat.shape or q.shape != updraft.shape:
        raise ValueError("vapor, saturation_vapor, and ascent must share a shape")
    qc = np.zeros_like(q) if condensate is None else np.clip(
        np.asarray(condensate, dtype=np.float64), 0.0, None
    )
    if qc.shape != q.shape:
        raise ValueError("condensate shape does not match vapor")

    relative_humidity = q / qsat
    humidity_activation = np.clip((relative_humidity - 0.8) / 0.2, 0.0, 1.0)
    ascent_activation = updraft / (1.0 + updraft)
    condensation_fraction = 1.0 - np.exp(
        -dt_days * humidity_activation * ascent_activation / condensation_timescale_days
    )
    condensed = q * condensation_fraction
    q_after_condensation = q - condensed
    qc_available = qc + condensed

    fallout_fraction = 1.0 - np.exp(
        -dt_days * (0.25 + 0.75 * ascent_activation) / fallout_timescale_days
    )
    rainout = qc_available * fallout_fraction
    qc_next = qc_available - rainout
    return q_after_condensation, qc_next, rainout


def stability_aware_condensation(
    vapor: np.ndarray,
    saturation_vapor: np.ndarray,
    temperature_k: np.ndarray,
    resolved_ascent: np.ndarray,
    condensate: np.ndarray | None,
    *,
    environment_temperature_k: np.ndarray | None = None,
    surface_pressure_hpa: float,
    dt_days: float,
    condensation_timescale_days: float,
    fallout_timescale_days: float,
    critical_relative_humidity: float = 0.70,
    reference_height_m: float = 3500.0,
    environmental_lapse_rate_k_per_km: float = 6.5,
    moist_lapse_rate_k_per_km: float = 6.0,
    cape_scale_j_kg: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Conservative large-scale condensation gated by a slab CAPE proxy.

    This is deliberately a one-layer approximation, not a claim to resolve a
    full sounding.  A surface parcel is lifted dry-adiabatically to its lifting
    condensation level and moist-adiabatically to ``reference_height_m``.  Its
    buoyancy relative to an environmental lapse rate supplies a CAPE-like
    stability gate; resolved horizontal convergence supplies the independent
    ascent gate.  Moisture above ``critical_relative_humidity * qsat`` then
    relaxes to the reservoir over ``condensation_timescale_days``.

    All transfers remain in the caller's column mixing-ratio units, so vapor +
    condensate + rainout is conserved to floating-point precision.
    """
    if (
        dt_days <= 0.0
        or condensation_timescale_days <= 0.0
        or fallout_timescale_days <= 0.0
        or surface_pressure_hpa <= 0.0
        or reference_height_m <= 0.0
        or cape_scale_j_kg <= 0.0
    ):
        raise ValueError("physical timescales, pressure, height, and CAPE scale must be positive")
    if not 0.0 < critical_relative_humidity < 1.0:
        raise ValueError("critical_relative_humidity must lie strictly between zero and one")

    q = np.clip(np.asarray(vapor, dtype=np.float64), 0.0, None)
    qsat = np.maximum(np.asarray(saturation_vapor, dtype=np.float64), 1e-12)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    ascent = np.clip(np.asarray(resolved_ascent, dtype=np.float64), 0.0, None)
    if not (q.shape == qsat.shape == temperature.shape == ascent.shape):
        raise ValueError("vapor, saturation, temperature, and ascent must share a shape")
    qc = np.zeros_like(q) if condensate is None else np.clip(
        np.asarray(condensate, dtype=np.float64), 0.0, None
    )
    if qc.shape != q.shape:
        raise ValueError("condensate shape does not match vapor")

    temperature_c = np.clip(temperature - 273.15, -80.0, 70.0)
    vapor_pressure_hpa = np.clip(
        q * float(surface_pressure_hpa) / 0.622, 1e-5, 0.99 * float(surface_pressure_hpa)
    )
    log_ratio = np.log(vapor_pressure_hpa / 6.112)
    dewpoint_c = 243.5 * log_ratio / (17.67 - log_ratio)
    lcl_m = np.clip(125.0 * (temperature_c - dewpoint_c), 0.0, reference_height_m)
    dry_lapse = 9.8e-3
    parcel_temperature = temperature - (
        dry_lapse * lcl_m
        + (moist_lapse_rate_k_per_km * 1e-3) * (reference_height_m - lcl_m)
    )
    if environment_temperature_k is None:
        environment_temperature = temperature - (
            environmental_lapse_rate_k_per_km * 1e-3 * reference_height_m
        )
    else:
        environment_temperature = np.asarray(environment_temperature_k, dtype=np.float64)
        if environment_temperature.shape != q.shape:
            raise ValueError("environment_temperature_k shape does not match vapor")
    buoyancy_temperature = parcel_temperature - environment_temperature
    cape_proxy = 9.81 * reference_height_m * np.clip(
        buoyancy_temperature / np.maximum(environment_temperature, 150.0), 0.0, None
    )
    stability_activation = 1.0 - np.exp(-cape_proxy / cape_scale_j_kg)
    ascent_activation = ascent / (1.0 + ascent)
    relative_humidity = q / qsat
    moisture_excess = np.clip(q - critical_relative_humidity * qsat, 0.0, None)
    condensation_fraction = 1.0 - np.exp(-dt_days / condensation_timescale_days)
    # Numerical transport and vertical exchange can place vapor in a colder
    # column faster than the resolved ascent/CAPE trigger can respond.  That
    # true supersaturation is still a phase change, not instantaneous surface
    # rain: transfer it to the prognostic condensate reservoir so it receives
    # the same fallout time scale, transport, cloud signature, and latent heat
    # treatment as every other condensate source.  The caller's former final
    # ``q - qsat`` rainout safeguard then becomes inactive except for roundoff.
    # ``maximum`` avoids double-counting the part of the activated transfer
    # which already removes enough vapor to restore saturation.
    activated_condensed = (
        moisture_excess * condensation_fraction * stability_activation * ascent_activation
    )
    supersaturation_condensed = np.maximum(q - qsat, 0.0)
    condensed = np.maximum(activated_condensed, supersaturation_condensed)
    condensed = np.minimum(condensed, q)
    q_after_condensation = q - condensed
    qc_available = qc + condensed

    fallout_fraction = 1.0 - np.exp(
        -dt_days * (0.25 + 0.75 * ascent_activation) / fallout_timescale_days
    )
    rainout = qc_available * fallout_fraction
    qc_next = qc_available - rainout
    return (
        q_after_condensation,
        qc_next,
        rainout,
        cape_proxy.astype(np.float32),
        stability_activation.astype(np.float32),
    )


def simplified_betts_miller_condensation(
    vapor: np.ndarray,
    saturation_vapor: np.ndarray,
    resolved_ascent: np.ndarray,
    condensate: np.ndarray | None,
    *,
    dt_days: float,
    relaxation_hours: float = 2.0,
    target_relative_humidity: float = 0.70,
    fallout_timescale_days: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Relax convectively ascending vapor toward a reference RH conservatively.

    This is the humidity half of a simplified Betts--Miller adjustment: an
    ascending column above the prescribed reference humidity transfers its
    excess vapor to cloud condensate over a fixed convective timescale. The
    subsequent fallout is the same prognostic reservoir closure used by the
    large-scale scheme, so vapor + cloud water + rainout is conserved.
    """
    if dt_days <= 0.0 or relaxation_hours <= 0.0 or fallout_timescale_days <= 0.0:
        raise ValueError("dt_days and relaxation/fallout timescales must be positive")
    if not 0.0 < target_relative_humidity < 1.0:
        raise ValueError("target_relative_humidity must lie strictly between zero and one")
    q = np.clip(np.asarray(vapor, dtype=np.float64), 0.0, None)
    qsat = np.maximum(np.asarray(saturation_vapor, dtype=np.float64), 1e-12)
    ascent = np.clip(np.asarray(resolved_ascent, dtype=np.float64), 0.0, None)
    cloud = np.zeros_like(q) if condensate is None else np.clip(
        np.asarray(condensate, dtype=np.float64), 0.0, None
    )
    if not (q.shape == qsat.shape == ascent.shape == cloud.shape):
        raise ValueError("vapor, saturation, ascent, and condensate must share a shape")
    adjustment = 1.0 - np.exp(-dt_days * 24.0 / relaxation_hours)
    ascent_activation = ascent / (1.0 + ascent)
    excess = np.maximum(q - target_relative_humidity * qsat, 0.0)
    condensed = np.minimum(q, excess * adjustment * ascent_activation)
    q_next = q - condensed
    cloud = cloud + condensed
    fallout = cloud * (1.0 - np.exp(-dt_days / fallout_timescale_days))
    return q_next, cloud - fallout, fallout, condensed.astype(np.float32)


def separate_cloud_and_hydrometeor_reservoirs(
    cloud_condensate: np.ndarray,
    precipitating_condensate: np.ndarray | None,
    newly_condensed: np.ndarray,
    *,
    dt_days: float,
    autoconversion_timescale_days: float,
    fallout_timescale_days: float,
    cloud_retention_q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert cloud water to hydrometeors, then sediment the latter.

    This deliberately keeps cloud optical mass and precipitating mass as
    distinct reservoirs.  Only condensate above the retained cloud-water
    amount autoconverts; fallout never directly depletes cloud water.
    """
    if dt_days <= 0.0 or autoconversion_timescale_days <= 0.0 or fallout_timescale_days <= 0.0:
        raise ValueError("time scales and dt_days must be positive")
    cloud = np.clip(np.asarray(cloud_condensate, dtype=np.float64), 0.0, None)
    new = np.clip(np.asarray(newly_condensed, dtype=np.float64), 0.0, None)
    rain = np.zeros_like(cloud) if precipitating_condensate is None else np.clip(
        np.asarray(precipitating_condensate, dtype=np.float64), 0.0, None
    )
    if not (cloud.shape == rain.shape == new.shape):
        raise ValueError("cloud, hydrometeor, and new-condensate arrays must share a shape")
    cloud = cloud + new
    excess = np.maximum(cloud - float(max(cloud_retention_q, 0.0)), 0.0)
    converted = excess * (1.0 - np.exp(-dt_days / autoconversion_timescale_days))
    cloud = cloud - converted
    rain = rain + converted
    fallout = rain * (1.0 - np.exp(-dt_days / fallout_timescale_days))
    return cloud, rain - fallout, fallout


def evolve_pressure_condensate_reservoirs(
    cloud_condensate_kg_m2: np.ndarray,
    precipitating_hydrometeors_kg_m2: np.ndarray,
    newly_condensed_kg_m2: np.ndarray,
    *,
    dt_days: float,
    autoconversion_timescale_days: float,
    fallout_timescale_days: float,
    cloud_retention_kg_m2: float,
) -> PressureCondensateReservoirStep:
    """Advance pressure-column cloud and falling-water reservoirs exactly once.

    This is the pressure-coordinate analogue of the older generic reservoir
    helper, but reports autoconversion explicitly and keeps the units in
    physical column mass.  It deliberately has no phase or circulation
    diagnosis: its inputs are the contemporaneous vapour-to-cloud conversion,
    so its only external water sink is the returned fallout.
    """
    if dt_days <= 0.0 or autoconversion_timescale_days <= 0.0 or fallout_timescale_days <= 0.0:
        raise ValueError("time scales and dt_days must be positive")
    if cloud_retention_kg_m2 < 0.0:
        raise ValueError("cloud_retention_kg_m2 must be non-negative")
    cloud, hydrometeors, new = (
        np.asarray(value, dtype=np.float64)
        for value in (
            cloud_condensate_kg_m2,
            precipitating_hydrometeors_kg_m2,
            newly_condensed_kg_m2,
        )
    )
    if cloud.shape != hydrometeors.shape or cloud.shape != new.shape:
        raise ValueError("pressure condensate reservoirs must share a shape")
    if not all(np.all(np.isfinite(value)) and np.all(value >= 0.0) for value in (cloud, hydrometeors, new)):
        raise ValueError("pressure condensate reservoirs must be finite and non-negative")
    cloud_total = cloud + new
    cloud_excess = np.maximum(cloud_total - float(cloud_retention_kg_m2), 0.0)
    autoconverted = cloud_excess * (
        1.0 - np.exp(-float(dt_days) / float(autoconversion_timescale_days))
    )
    cloud_next = cloud_total - autoconverted
    hydrometeor_total = hydrometeors + autoconverted
    fallout = hydrometeor_total * (
        1.0 - np.exp(-float(dt_days) / float(fallout_timescale_days))
    )
    return PressureCondensateReservoirStep(
        cloud_next.astype(np.float32), (hydrometeor_total - fallout).astype(np.float32),
        fallout.astype(np.float32), autoconverted.astype(np.float32),
    )


def transport_pressure_condensate_reservoirs(
    cloud_condensate_kg_m2: np.ndarray,
    precipitating_hydrometeors_kg_m2: np.ndarray,
    wind_u_m_s: np.ndarray,
    wind_v_m_s: np.ndarray,
    *,
    dt_days: float,
    fallout_timescale_days: float,
    cloud_transport_scale: float,
    transport_hydrometeors: bool,
    dx_m: np.ndarray | float,
    dy_m: float,
    cell_area_m2: np.ndarray | float,
    x_face_length_m: np.ndarray | float,
    y_face_length_m: np.ndarray | float,
) -> PressureCondensateTransportStep:
    """Transport pressure condensate reservoirs without changing their mass."""
    if dt_days <= 0.0 or fallout_timescale_days <= 0.0:
        raise ValueError("dt_days and fallout_timescale_days must be positive")
    cloud, hydrometeors, u, v = (
        np.asarray(value, dtype=np.float64)
        for value in (cloud_condensate_kg_m2, precipitating_hydrometeors_kg_m2, wind_u_m_s, wind_v_m_s)
    )
    if not (cloud.shape == hydrometeors.shape == u.shape == v.shape):
        raise ValueError("pressure condensate transport fields must share a shape")
    if not all(np.all(np.isfinite(value)) and np.all(value >= 0.0) for value in (cloud, hydrometeors)):
        raise ValueError("pressure condensate reservoirs must be finite and non-negative")
    from column_water import evolve_column_water

    scale = float(np.clip(cloud_transport_scale, 0.0, 1.0))
    if scale > 0.0:
        cloud_step = evolve_column_water(
            cloud, np.zeros_like(cloud), np.zeros_like(cloud), u, v,
            dx_m=dx_m, dy_m=dy_m, dt_days=dt_days, cell_area_m2=cell_area_m2,
            x_face_length_m=x_face_length_m, y_face_length_m=y_face_length_m,
        )
        cloud_next = (1.0 - scale) * cloud + scale * cloud_step.water_mm
        cloud_residual = float(cloud_step.relative_residual)
    else:
        cloud_next = cloud
        cloud_residual = 0.0
    hydro_residual: float | None = None
    hydro_next = hydrometeors
    if transport_hydrometeors and np.any(hydrometeors):
        hydro_step = evolve_column_water(
            hydrometeors, np.zeros_like(hydrometeors), np.zeros_like(hydrometeors), u, v,
            dx_m=dx_m, dy_m=dy_m, dt_days=min(dt_days, fallout_timescale_days),
            cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
            y_face_length_m=y_face_length_m,
        )
        hydro_next = hydro_step.water_mm
        hydro_residual = float(hydro_step.relative_residual)
    return PressureCondensateTransportStep(
        cloud_next.astype(np.float32), hydro_next.astype(np.float32),
        cloud_residual, hydro_residual,
    )


def column_water_forcing_from_boundary_fluxes(
    surface_source_kg_m2: np.ndarray,
    fallout_kg_m2: np.ndarray,
    *,
    dt_seconds: float,
) -> np.ndarray:
    """Return the total-atmospheric-water forcing for the circulation solve.

    Vapour-to-cloud conversion is internal to the atmospheric column and must
    not appear here.  The only external terms are surface supply and fallout;
    expressing both as step-integrated physical masses makes the returned
    kg m-2 s-1 forcing unambiguous at the pressure-circulation boundary.
    """
    if dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be positive")
    source = np.asarray(surface_source_kg_m2, dtype=np.float64)
    fallout = np.asarray(fallout_kg_m2, dtype=np.float64)
    if source.shape != fallout.shape:
        raise ValueError("surface source and fallout must share a shape")
    if not all(np.all(np.isfinite(value)) and np.all(value >= 0.0) for value in (source, fallout)):
        raise ValueError("surface source and fallout must be finite and non-negative")
    return ((source - fallout) / float(dt_seconds)).astype(np.float32)


def column_water_forcing_from_budget(
    surface_source_kg_m2: np.ndarray,
    fallout_kg_m2: np.ndarray,
    water_before_kg_m2: np.ndarray,
    water_after_kg_m2: np.ndarray,
    *,
    dt_seconds: float,
) -> np.ndarray:
    """Return horizontal water convergence implied by a transient column budget.

    ``surface source - fallout`` alone is valid only for a steady atmospheric
    water store.  A prognostic pressure column instead obeys

    ``horizontal convergence = source - fallout - d(storage)/dt``.

    The storage includes vapour and both condensate reservoirs.  This is the
    term a simultaneous circulation solve must constrain; phase conversion is
    internal because it cancels between vapour and cloud storage.
    """
    if dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be positive")
    source, fallout, before, after = (
        np.asarray(value, dtype=np.float64)
        for value in (surface_source_kg_m2, fallout_kg_m2, water_before_kg_m2, water_after_kg_m2)
    )
    if not (source.shape == fallout.shape == before.shape == after.shape):
        raise ValueError("column-water budget fields must share a shape")
    if not all(np.all(np.isfinite(value)) and np.all(value >= 0.0) for value in (source, fallout, before, after)):
        raise ValueError("column-water budget fields must be finite and non-negative")
    return ((source - fallout - (after - before)) / float(dt_seconds)).astype(np.float32)
