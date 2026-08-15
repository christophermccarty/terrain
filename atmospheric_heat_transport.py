"""Energy-unit diagnostics for atmospheric horizontal heat transport."""
from __future__ import annotations

import numpy as np


def temperature_transport_to_heat_convergence(
    temperature_increment_k: np.ndarray,
    *,
    surface_pressure_pa: float,
    cp_j_kg_k: float,
    gravity_m_s2: float,
    dt_seconds: float,
) -> np.ndarray:
    """Convert a resolved column-temperature increment to W m-2.

    Positive output means horizontal atmospheric transport heated the column.
    The supported atmosphere is a single prognostic temperature layer, so its
    represented dry-air column heat capacity is ``(p_s / g) * cp``.
    """
    if dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be positive")
    if surface_pressure_pa <= 0.0 or cp_j_kg_k <= 0.0 or gravity_m_s2 <= 0.0:
        raise ValueError("pressure, heat capacity, and gravity must be positive")
    increment = np.asarray(temperature_increment_k)
    scale = (float(surface_pressure_pa) / float(gravity_m_s2)) * float(cp_j_kg_k)
    return (increment * (scale / float(dt_seconds))).astype(np.float32, copy=False)


def apply_sensible_heat_to_atmospheric_column(
    air_temperature_k: np.ndarray,
    upward_sensible_heat_w_m2: np.ndarray,
    land_mask: np.ndarray,
    *,
    surface_pressure_pa: float,
    cp_j_kg_k: float,
    gravity_m_s2: float,
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Add land's upward sensible flux to the represented air column.

    The force-restore surface budget already subtracts this same positive-upward
    flux.  Applying it here with the single-layer atmospheric column heat
    capacity makes that boundary-transfer term equal and opposite.
    """
    if dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be positive")
    if surface_pressure_pa <= 0.0 or cp_j_kg_k <= 0.0 or gravity_m_s2 <= 0.0:
        raise ValueError("pressure, heat capacity, and gravity must be positive")
    air = np.asarray(air_temperature_k, dtype=np.float64)
    sensible = np.asarray(upward_sensible_heat_w_m2, dtype=np.float64)
    land = np.asarray(land_mask, dtype=bool)
    if air.shape != sensible.shape or air.shape != land.shape:
        raise ValueError("air temperature, sensible heat, and land mask must share a shape")
    atmospheric_gain = np.where(land, sensible, 0.0)
    column_capacity = (
        float(surface_pressure_pa) / float(gravity_m_s2) * float(cp_j_kg_k)
    )
    increment = atmospheric_gain * float(dt_seconds) / column_capacity
    return (
        (air + increment).astype(np.float32, copy=False),
        atmospheric_gain.astype(np.float32, copy=False),
    )


def external_energy_budget_heat_convergence(
    net_top_radiation_w_m2: np.ndarray,
    surface_downward_heat_flux_w_m2: np.ndarray,
    *,
    atmospheric_storage_w_m2: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Diagnose atmospheric horizontal heat convergence from column fluxes.

    Both boundary fluxes use ExoPlaSim's downward-positive convention.  The
    atmospheric budget is ``storage = top - surface + convergence``, hence
    ``convergence = storage - top + surface``.  Monthly climatologies commonly
    neglect storage only after verifying the annual-mean residual is small.
    """
    top = np.asarray(net_top_radiation_w_m2)
    surface = np.asarray(surface_downward_heat_flux_w_m2)
    storage = np.asarray(atmospheric_storage_w_m2)
    return (storage - top + surface).astype(np.float32, copy=False)


def close_global_heat_convergence(
    heat_convergence_w_m2: np.ndarray,
    latitude_radians: np.ndarray,
) -> np.ndarray:
    """Remove the area-weighted global mean from a horizontal convergence.

    Horizontal transport redistributes energy and therefore integrates to zero
    over a closed sphere.  This projection removes only the impossible global
    source/sink left by the supported finite-difference transport operators.
    """
    field = np.asarray(heat_convergence_w_m2, dtype=np.float64)
    latitude = np.asarray(latitude_radians, dtype=np.float64)
    if field.ndim != 2 or latitude.shape != (field.shape[0],):
        raise ValueError("field must be 2-D and latitude must match its rows")
    weights = np.cos(latitude)[:, None]
    area_mean = np.sum(field * weights) / (field.shape[1] * np.sum(weights))
    return (field - area_mean).astype(np.float32, copy=False)


def summarize_heat_convergence_samples(
    samples: list[np.ndarray],
    latitude_radians: np.ndarray,
    *,
    applied_grid_area_means_w_m2: list[float] | None = None,
) -> dict[str, float | int]:
    """Summarize a sequence of resolved heat-convergence fields."""
    if not samples:
        return {}
    latitude = np.asarray(latitude_radians, dtype=np.float64)
    shape = np.asarray(samples[0]).shape
    if len(shape) != 2 or latitude.shape != (shape[0],):
        raise ValueError("samples must be 2-D and latitude must match their rows")
    weights = np.cos(latitude)[:, None]
    denominator = shape[1] * np.sum(weights)
    area_means: list[float] = []
    rms_values: list[float] = []
    flattened: list[np.ndarray] = []
    for sample in samples:
        field = np.asarray(sample, dtype=np.float64)
        if field.shape != shape or not np.all(np.isfinite(field)):
            raise ValueError("heat-convergence samples must be finite with consistent shapes")
        area_means.append(float(np.sum(field * weights) / denominator))
        rms_values.append(float(np.sqrt(np.sum(field * field * weights) / denominator)))
        flattened.append(field.reshape(-1))
    values = np.concatenate(flattened)
    closure_means = area_means
    if applied_grid_area_means_w_m2 is not None:
        if len(applied_grid_area_means_w_m2) != len(samples):
            raise ValueError("applied-grid area means must match sample count")
        closure_means = [float(value) for value in applied_grid_area_means_w_m2]
    return {
        "sample_count": len(samples),
        "max_abs_global_area_mean_w_m2": float(np.max(np.abs(closure_means))),
        "display_grid_max_abs_area_mean_w_m2": float(np.max(np.abs(area_means))),
        "mean_rms_w_m2": float(np.mean(rms_values)),
        "p01_w_m2": float(np.percentile(values, 1.0)),
        "p05_w_m2": float(np.percentile(values, 5.0)),
        "median_w_m2": float(np.percentile(values, 50.0)),
        "p95_w_m2": float(np.percentile(values, 95.0)),
        "p99_w_m2": float(np.percentile(values, 99.0)),
        "max_abs_w_m2": float(np.max(np.abs(values))),
    }
