"""Regional, read-only decomposition of the legacy precipitation pathway.

The supported atmosphere does not yet evolve a fully conservative horizontal
moisture flux. These diagnostics therefore distinguish *available observables*
from an unclaimed physical closure: ``lower_wind_convergence_proxy`` is
the negative lower-wind divergence used by the rainfall scheme, not a resolved
column moisture-flux convergence.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from atmosphere import flux_divergence_spherical
from regional_validation import (
    ClimateRegion,
    EARTH_PRECIP_REGIONS,
    latitude_longitude_centers,
    region_mask,
)


_METRIC_NAMES = (
    "precipitation_final_mm_day",
    "precipitation_raw_mm_day",
    "post_raw_precip_adjustment_mm_day",
    "land_evaporation_mm_day",
    "ocean_evaporation_mm_day",
    "lower_wind_convergence_proxy",
    "moisture_flux_convergence_driver",
    "humidity_before_rainout_q",
    "ascent_driver",
    "convection_driver",
    "orographic_uplift_driver",
    "rain_shadow_suppression",
    "row_target_achieved_fraction",
    "physical_moisture_flux_convergence_q_s",
    "lower_zonal_wind_m_s",
    "lower_meridional_wind_m_s",
    "lower_wind_speed_m_s",
    "storm_track_window",
    "subsidence_suppression",
    "source_ocean_humidity_q",
    "source_ocean_surface_temperature_k",
    "source_ocean_air_surface_inversion_k",
    "source_ocean_cloud_fraction",
    "source_ocean_evaporation_mm_day",
    "land_minus_source_ocean_surface_temperature_k",
    "source_to_region_lower_wind_m_s",
    "upwind_sst_anomaly_k",
)


# These ocean boxes are deliberately controls, not precipitation targets. They
# describe the nearest broad marine source relevant to the Phase-1 pathways and
# make the regional diagnostic answer a falsifiable question (for example,
# whether eastward lower wind actually carries western-Pacific humidity toward
# East China). The boxes are wide enough to retain ocean cells on the compact
# 64x128 grid; results remain ``None`` instead of silently falling back to land.
REGIONAL_MOISTURE_SOURCE_REGIONS: dict[str, ClimateRegion] = {
    "Atacama": ClimateRegion(
        "SE Pacific source", -20.0, -28.0, -84.0, -72.0, "ocean_control", 0.0, 0.0
    ),
    "East China": ClimateRegion(
        "W Pacific source", 30.0, 23.0, 122.0, 138.0, "ocean_control", 0.0, 0.0
    ),
    "S Japan": ClimateRegion(
        "Japan Pacific source", 34.0, 28.0, 136.0, 150.0, "ocean_control", 0.0, 0.0
    ),
    "Central Europe": ClimateRegion(
        "NE Atlantic source", 53.0, 47.0, -15.0, -2.0, "ocean_control", 0.0, 0.0
    ),
}

SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")


def regional_moisture_budget_snapshot(
    elevation: np.ndarray,
    debug_fields: Mapping[str, Any],
    *,
    pathway_fields: Mapping[str, Any] | None = None,
    radius_m: float = 6.371e6,
) -> dict[str, dict[str, float | None]]:
    """Summarize one non-mutating precipitation diagnostic by named region.

    ``debug_fields`` must come from ``atmosphere.generate_precipitation`` for
    the same grid. Every rate is in mm/day except the explicitly named
    dimensionless drivers and the divergence proxy. The raw-to-final delta
    records the net effect of the calibrated allocator and later rain-export
    constraints; it deliberately does not call that delta new physical supply.
    """
    elevation = np.asarray(elevation)
    shape = elevation.shape
    if elevation.ndim != 2:
        raise ValueError("regional moisture budget requires a two-dimensional grid")
    land = elevation > 0.0
    ocean = ~land

    def field(name: str, *, required: bool = False) -> np.ndarray | None:
        value = debug_fields.get(name)
        if value is None:
            if required:
                raise ValueError(f"precipitation diagnostic lacks required field {name!r}")
            return None
        values = np.asarray(value, dtype=np.float64)
        if values.shape == (shape[0],):
            values = np.broadcast_to(values[:, None], shape)
        if values.shape != shape:
            raise ValueError(
                f"precipitation diagnostic field {name!r} shape {values.shape} "
                f"does not match elevation shape {shape}"
            )
        return values

    final_precip = field("precipitation_final_mm_day", required=True)
    raw_precip = field("precipitation_raw_mm_day", required=True)
    divergence = field("div")
    rates: dict[str, np.ndarray | None] = {
        "precipitation_final_mm_day": final_precip,
        "precipitation_raw_mm_day": raw_precip,
        "post_raw_precip_adjustment_mm_day": final_precip - raw_precip,
        "land_evaporation_mm_day": field("land_evaporation_mm_day"),
        "ocean_evaporation_mm_day": field("ocean_evaporation_mm_day"),
        "lower_wind_convergence_proxy": (
            None if divergence is None else -divergence
        ),
        "moisture_flux_convergence_driver": field("conv"),
        "humidity_before_rainout_q": field("humidity_before_rainout"),
        "ascent_driver": field("ascent_driver"),
        "convection_driver": field("conv_driver"),
        "orographic_uplift_driver": field("orog"),
        "rain_shadow_suppression": field("rain_shadow_suppression"),
        "row_target_achieved_fraction": field("precip_target_achieved_fraction"),
    }
    pathway_fields = {} if pathway_fields is None else pathway_fields

    def pathway_field(name: str) -> np.ndarray | None:
        value = pathway_fields.get(name)
        if value is None:
            return None
        values = np.asarray(value, dtype=np.float64)
        if values.shape == (shape[0],):
            values = np.broadcast_to(values[:, None], shape)
        if values.shape != shape:
            raise ValueError(
                f"regional pathway field {name!r} shape {values.shape} "
                f"does not match elevation shape {shape}"
            )
        return values

    lower_u = pathway_field("lower_wind_u_m_s")
    lower_v = pathway_field("lower_wind_v_m_s")
    surface_temperature = pathway_field("surface_temperature_k")
    air_temperature = pathway_field("air_temperature_k")
    cloud_fraction = pathway_field("cloud_fraction")
    humidity = rates["humidity_before_rainout_q"]
    physical_flux_convergence = None
    if humidity is not None and lower_u is not None and lower_v is not None:
        lat_deg, _ = latitude_longitude_centers(*shape)
        physical_flux_convergence = -flux_divergence_spherical(
            humidity,
            lower_u,
            lower_v,
            np.radians(lat_deg),
            radius_m=float(radius_m),
        )

    def mean(values: np.ndarray | None, mask: np.ndarray) -> float | None:
        return None if values is None or not np.any(mask) else float(np.mean(values[mask]))

    result: dict[str, dict[str, float | None]] = {}
    for region in EARTH_PRECIP_REGIONS:
        mask = region_mask(shape, region, cell_mask=land)
        metrics = {name: mean(values, mask) for name, values in rates.items()}
        metrics.update(
            {
                "physical_moisture_flux_convergence_q_s": mean(
                    physical_flux_convergence, mask
                ),
                "lower_zonal_wind_m_s": mean(lower_u, mask),
                "lower_meridional_wind_m_s": mean(lower_v, mask),
                "lower_wind_speed_m_s": (
                    None
                    if lower_u is None or lower_v is None
                    else mean(np.hypot(lower_u, lower_v), mask)
                ),
                "storm_track_window": mean(field("storm_window"), mask),
                "subsidence_suppression": mean(field("subsidence_suppression"), mask),
                "source_ocean_humidity_q": None,
                "source_ocean_surface_temperature_k": None,
                "source_ocean_air_surface_inversion_k": None,
                "source_ocean_cloud_fraction": None,
                "source_ocean_evaporation_mm_day": None,
                "land_minus_source_ocean_surface_temperature_k": None,
                "source_to_region_lower_wind_m_s": None,
                "upwind_sst_anomaly_k": mean(field("upwind_sst_anomaly"), mask),
            }
        )
        source = REGIONAL_MOISTURE_SOURCE_REGIONS.get(region.name)
        if source is not None:
            source_mask = region_mask(shape, source, cell_mask=ocean)
            source_temperature = mean(surface_temperature, source_mask)
            source_air_temperature = mean(air_temperature, source_mask)
            regional_surface_temperature = mean(surface_temperature, mask)
            metrics.update(
                {
                    "source_ocean_humidity_q": mean(humidity, source_mask),
                    "source_ocean_surface_temperature_k": source_temperature,
                    "source_ocean_air_surface_inversion_k": (
                        None
                        if source_air_temperature is None or source_temperature is None
                        else source_air_temperature - source_temperature
                    ),
                    "source_ocean_cloud_fraction": mean(cloud_fraction, source_mask),
                    "source_ocean_evaporation_mm_day": mean(
                        rates["ocean_evaporation_mm_day"], source_mask
                    ),
                    "land_minus_source_ocean_surface_temperature_k": (
                        None
                        if regional_surface_temperature is None or source_temperature is None
                        else regional_surface_temperature - source_temperature
                    ),
                }
            )
            if lower_u is not None and lower_v is not None and np.any(mask):
                # Local tangent-plane projection. Positive means the regional
                # mean lower wind points from the named ocean control toward
                # the land box; it is a directional pathway diagnostic, not a
                # parcel trajectory or a resolved transport budget.
                dlon = ((0.5 * (region.lon_w + region.lon_e) - 0.5 * (source.lon_w + source.lon_e) + 180.0) % 360.0) - 180.0
                dlat = 0.5 * (region.lat_n + region.lat_s) - 0.5 * (source.lat_n + source.lat_s)
                east = dlon * np.cos(np.radians(0.5 * (region.lat_n + region.lat_s)))
                norm = float(np.hypot(east, dlat))
                if norm > 0.0:
                    metrics["source_to_region_lower_wind_m_s"] = float(
                        (east * mean(lower_u, mask) + dlat * mean(lower_v, mask)) / norm
                    )
        result[region.name] = metrics
    return result


def season_for_day(
    day_of_year: float,
    *,
    orbital_period_days: float,
    vernal_equinox_day: float,
) -> str:
    """Return an astronomical three-month season label for a model day."""
    if orbital_period_days <= 0.0:
        raise ValueError("orbital_period_days must be positive")
    phase = (float(day_of_year) - float(vernal_equinox_day)) % float(orbital_period_days)
    quarter = int(4.0 * phase / float(orbital_period_days))
    return ("MAM", "JJA", "SON", "DJF")[min(quarter, 3)]


def seasonal_regional_moisture_budget(
    snapshots: Iterable[tuple[str, float, Mapping[str, Mapping[str, float | None]]]],
) -> dict[str, Any]:
    """Duration-weight the pathway snapshots by astronomical season.

    Empty seasons remain explicit with zero sampled days and ``None`` metrics;
    a partial validation run therefore cannot masquerade as a full annual
    seasonal diagnosis.
    """
    grouped: dict[str, list[tuple[float, Mapping[str, Mapping[str, float | None]]]]] = {
        season: [] for season in SEASON_ORDER
    }
    for season, duration, snapshot in snapshots:
        if season not in grouped:
            raise ValueError(f"unknown season {season!r}")
        grouped[season].append((float(duration), snapshot))
    return {
        "season_order": list(SEASON_ORDER),
        "seasons": {
            season: {
                "sampled_days": float(sum(duration for duration, _ in entries)),
                "regions": time_average_regional_moisture_budget(entries),
            }
            for season, entries in grouped.items()
        },
    }


def time_average_regional_moisture_budget(
    snapshots: Iterable[tuple[float, Mapping[str, Mapping[str, float | None]]]],
) -> dict[str, dict[str, float | None]]:
    """Duration-weight a sequence of regional precipitation snapshots."""
    totals: dict[str, dict[str, float]] = {}
    weights: dict[str, dict[str, float]] = {}
    for duration_days, snapshot in snapshots:
        duration = float(duration_days)
        if duration <= 0.0:
            raise ValueError("regional moisture budget duration must be positive")
        for region_name, metrics in snapshot.items():
            for name, value in metrics.items():
                if value is None or not np.isfinite(value):
                    continue
                totals.setdefault(region_name, {})[name] = (
                    totals.setdefault(region_name, {}).get(name, 0.0) + duration * float(value)
                )
                weights.setdefault(region_name, {})[name] = (
                    weights.setdefault(region_name, {}).get(name, 0.0) + duration
                )
    return {
        region.name: {
            name: (
                totals[region.name][name] / weights[region.name][name]
                if name in weights.get(region.name, {})
                else None
            )
            for name in _METRIC_NAMES
        }
        for region in EARTH_PRECIP_REGIONS
    }
