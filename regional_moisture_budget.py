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

from regional_validation import EARTH_PRECIP_REGIONS, region_mask


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
)


def regional_moisture_budget_snapshot(
    elevation: np.ndarray,
    debug_fields: Mapping[str, Any],
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
    result: dict[str, dict[str, float | None]] = {}
    for region in EARTH_PRECIP_REGIONS:
        mask = region_mask(shape, region, cell_mask=land)
        result[region.name] = {
            name: (
                None if values is None or not np.any(mask)
                else float(np.mean(values[mask], dtype=np.float64))
            )
            for name, values in rates.items()
        }
    return result


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
