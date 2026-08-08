"""Versioned monthly-climatology references and area-weighted map skill.

The simulator's Köppen map is derived from twelve monthly temperature and
precipitation fields.  This module keeps the corresponding validation input
small, explicit, and independent of a particular reanalysis provider.  A
reference is a safe ``.npz`` file (pickle is never enabled) containing:

``temperature_k``
    Float array shaped ``(12, H, W)``.
``precipitation_mm_day``
    Float array shaped ``(12, H, W)``.
``land_fraction`` (optional)
    Float array shaped ``(H, W)``.  It lets a coarse reference exclude mixed
    coastal and open-ocean cells without borrowing the simulation's mask.
``metadata_json`` (required)
    A JSON object recording source, period, licence, and preprocessing.

Input rows are north-to-south and columns span [-180, 180), matching the
simulation grid and the bundled Köppen reference.  Regridding uses exact
equirectangular-cell overlap weighted by spherical area.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MONTHLY_CLIMATOLOGY_SCHEMA_VERSION = 1
_MONTHS = 12
_PRECIP_LOG_FLOOR_MM_DAY = 0.05


def _validate_grid_shape(shape: tuple[int, ...], *, name: str) -> tuple[int, int]:
    if len(shape) != 3 or shape[0] != _MONTHS:
        raise ValueError(f"{name} must have shape (12, H, W), got {shape}")
    _, height, width = shape
    if height < 1 or width != 2 * height:
        raise ValueError(f"{name} must use a 2:1 equirectangular grid, got {shape}")
    return height, width


def _as_finite_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class MonthlyClimatology:
    """A validated twelve-month climate normal on a regular global grid."""

    temperature_k: np.ndarray
    precipitation_mm_day: np.ndarray
    metadata: Mapping[str, Any]
    land_fraction: np.ndarray | None = None

    def __post_init__(self) -> None:
        temperature = _as_finite_array(self.temperature_k, name="temperature_k")
        precipitation = _as_finite_array(
            self.precipitation_mm_day, name="precipitation_mm_day"
        )
        shape = _validate_grid_shape(temperature.shape, name="temperature_k")
        if precipitation.shape != temperature.shape:
            raise ValueError(
                "precipitation_mm_day shape does not match temperature_k "
                f"({precipitation.shape} vs {temperature.shape})"
            )
        if np.any(precipitation < 0.0):
            raise ValueError("precipitation_mm_day must be non-negative")
        metadata = dict(self.metadata)
        version = int(metadata.get("schema_version", MONTHLY_CLIMATOLOGY_SCHEMA_VERSION))
        if version != MONTHLY_CLIMATOLOGY_SCHEMA_VERSION:
            raise ValueError(
                f"monthly climatology schema v{version} is unsupported; expected "
                f"v{MONTHLY_CLIMATOLOGY_SCHEMA_VERSION}"
            )
        if not str(metadata.get("source", "")).strip():
            raise ValueError("monthly climatology metadata must name its source")
        if not str(metadata.get("period", "")).strip():
            raise ValueError("monthly climatology metadata must name its period")

        land_fraction = self.land_fraction
        if land_fraction is not None:
            land_fraction = _as_finite_array(land_fraction, name="land_fraction")
            if land_fraction.shape != shape:
                raise ValueError(
                    f"land_fraction must have shape {shape}, got {land_fraction.shape}"
                )
            if np.any((land_fraction < 0.0) | (land_fraction > 1.0)):
                raise ValueError("land_fraction must lie in [0, 1]")

        temperature.setflags(write=False)
        precipitation.setflags(write=False)
        if land_fraction is not None:
            land_fraction.setflags(write=False)
        metadata["schema_version"] = MONTHLY_CLIMATOLOGY_SCHEMA_VERSION
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "precipitation_mm_day", precipitation)
        object.__setattr__(self, "land_fraction", land_fraction)
        object.__setattr__(self, "metadata", metadata)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.temperature_k.shape[1:])


def load_monthly_climatology(path: str | Path) -> MonthlyClimatology:
    """Load a validated reference without allowing pickle execution."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"temperature_k", "precipitation_mm_day", "metadata_json"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"monthly climatology is missing required arrays: {', '.join(missing)}")
        raw_metadata = archive["metadata_json"]
        if raw_metadata.size != 1:
            raise ValueError("metadata_json must contain exactly one JSON value")
        try:
            metadata = json.loads(str(raw_metadata.reshape(-1)[0]))
        except json.JSONDecodeError as exc:
            raise ValueError("metadata_json is not valid JSON") from exc
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must decode to an object")
        land_fraction = archive["land_fraction"] if "land_fraction" in archive.files else None
        return MonthlyClimatology(
            temperature_k=archive["temperature_k"],
            precipitation_mm_day=archive["precipitation_mm_day"],
            land_fraction=land_fraction,
            metadata=metadata,
        )


def save_monthly_climatology(reference: MonthlyClimatology, path: str | Path) -> Path:
    """Write the portable reference format used by the validation CLI."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "temperature_k": reference.temperature_k.astype(np.float32),
        "precipitation_mm_day": reference.precipitation_mm_day.astype(np.float32),
        "metadata_json": np.asarray(json.dumps(dict(reference.metadata), sort_keys=True)),
    }
    if reference.land_fraction is not None:
        arrays["land_fraction"] = reference.land_fraction.astype(np.float32)
    np.savez_compressed(path, **arrays)
    return path


def _overlap_weights(source_edges: np.ndarray, target_edges: np.ndarray, *, sine: bool) -> np.ndarray:
    """Fraction of each target cell covered by each source cell."""
    source_lo = np.minimum(source_edges[:-1], source_edges[1:])
    source_hi = np.maximum(source_edges[:-1], source_edges[1:])
    target_lo = np.minimum(target_edges[:-1], target_edges[1:])
    target_hi = np.maximum(target_edges[:-1], target_edges[1:])
    overlap_lo = np.maximum(target_lo[:, None], source_lo[None, :])
    overlap_hi = np.minimum(target_hi[:, None], source_hi[None, :])
    intersects = overlap_hi > overlap_lo
    overlap = np.maximum(overlap_hi - overlap_lo, 0.0)
    if sine:
        overlap = intersects * np.abs(
            np.sin(np.radians(overlap_hi)) - np.sin(np.radians(overlap_lo))
        )
        target_extent = np.abs(
            np.sin(np.radians(target_hi)) - np.sin(np.radians(target_lo))
        )
    else:
        target_extent = target_hi - target_lo
    return overlap / target_extent[:, None]


def regrid_monthly_climatology(
    reference: MonthlyClimatology, height: int, width: int
) -> MonthlyClimatology:
    """Area-conservatively regrid a reference to any regular 2:1 grid."""
    if height < 1 or width != 2 * height:
        raise ValueError("target grid must be a 2:1 equirectangular grid")
    if reference.shape == (height, width):
        return reference
    source_h, source_w = reference.shape
    src_lat = np.linspace(90.0, -90.0, source_h + 1)
    dst_lat = np.linspace(90.0, -90.0, height + 1)
    src_lon = np.linspace(-180.0, 180.0, source_w + 1)
    dst_lon = np.linspace(-180.0, 180.0, width + 1)
    lat_weights = _overlap_weights(src_lat, dst_lat, sine=True)
    lon_weights = _overlap_weights(src_lon, dst_lon, sine=False)

    def regrid(field: np.ndarray) -> np.ndarray:
        return np.einsum("ai,mij,bj->mab", lat_weights, field, lon_weights, optimize=True)

    land = None
    if reference.land_fraction is not None:
        land = np.einsum(
            "ai,ij,bj->ab", lat_weights, reference.land_fraction, lon_weights, optimize=True
        )
        # Exact overlap weights sum to one analytically, but floating-point
        # roundoff can leave a coastal fraction infinitesimally outside the
        # validated [0, 1] interval.
        land = np.clip(land, 0.0, 1.0)
    metadata = dict(reference.metadata)
    metadata["regridded_from"] = {"height": source_h, "width": source_w}
    return MonthlyClimatology(
        temperature_k=regrid(reference.temperature_k),
        precipitation_mm_day=regrid(reference.precipitation_mm_day),
        land_fraction=land,
        metadata=metadata,
    )


def _weights_for(reference: MonthlyClimatology, model_land_mask: np.ndarray | None, minimum_land_fraction: float) -> np.ndarray:
    height, width = reference.shape
    lat = (0.5 - (np.arange(height, dtype=np.float64) + 0.5) / height) * np.pi
    weights = np.broadcast_to(np.cos(lat)[:, None], (height, width)).copy()
    if reference.land_fraction is not None:
        weights *= np.asarray(reference.land_fraction >= minimum_land_fraction, dtype=np.float64)
    if model_land_mask is not None:
        mask = np.asarray(model_land_mask, dtype=bool)
        if mask.shape != (height, width):
            raise ValueError("model_land_mask shape does not match reference")
        weights *= mask
    return weights


def _weighted_summary(model: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    weights_3d = np.broadcast_to(weights[None, :, :], model.shape)
    valid = np.isfinite(model) & np.isfinite(reference) & (weights_3d > 0.0)
    if not np.any(valid):
        return {"bias": float("nan"), "rmse": float("nan"), "correlation": float("nan")}
    w = weights_3d[valid]
    delta = model[valid] - reference[valid]
    bias = float(np.sum(w * delta) / np.sum(w))
    rmse = float(np.sqrt(np.sum(w * delta**2) / np.sum(w)))
    x = model[valid]
    y = reference[valid]
    x_mean = float(np.sum(w * x) / np.sum(w))
    y_mean = float(np.sum(w * y) / np.sum(w))
    x_dev = x - x_mean
    y_dev = y - y_mean
    denom = float(np.sqrt(np.sum(w * x_dev**2) * np.sum(w * y_dev**2)))
    correlation = float(np.sum(w * x_dev * y_dev) / denom) if denom > 0.0 else float("nan")
    return {"bias": bias, "rmse": rmse, "correlation": correlation}


def score_monthly_climatology(
    monthly_temperature_k: np.ndarray,
    monthly_precipitation_mm_day: np.ndarray,
    reference: MonthlyClimatology,
    *,
    model_land_mask: np.ndarray | None = None,
    minimum_land_fraction: float = 0.5,
) -> dict[str, Any]:
    """Score a model's monthly climate against a compatible reference grid."""
    model_temperature = _as_finite_array(monthly_temperature_k, name="monthly_temperature_k")
    model_precipitation = _as_finite_array(
        monthly_precipitation_mm_day, name="monthly_precipitation_mm_day"
    )
    _validate_grid_shape(model_temperature.shape, name="monthly_temperature_k")
    if model_temperature.shape != reference.temperature_k.shape:
        raise ValueError(
            f"model grid {model_temperature.shape[1:]} does not match reference {reference.shape}"
        )
    if model_precipitation.shape != model_temperature.shape:
        raise ValueError("monthly precipitation shape does not match monthly temperature")
    if np.any(model_precipitation < 0.0):
        raise ValueError("monthly precipitation must be non-negative")
    if not 0.0 <= minimum_land_fraction <= 1.0:
        raise ValueError("minimum_land_fraction must lie in [0, 1]")

    weights = _weights_for(reference, model_land_mask, minimum_land_fraction)
    temperature = _weighted_summary(model_temperature - 273.15, reference.temperature_k - 273.15, weights)
    precipitation = _weighted_summary(
        model_precipitation, reference.precipitation_mm_day, weights
    )
    log_precipitation = _weighted_summary(
        np.log(np.maximum(model_precipitation, _PRECIP_LOG_FLOOR_MM_DAY)),
        np.log(np.maximum(reference.precipitation_mm_day, _PRECIP_LOG_FLOOR_MM_DAY)),
        weights,
    )
    model_annual = np.mean(model_precipitation, axis=0)
    reference_annual = np.mean(reference.precipitation_mm_day, axis=0)
    annual_weights = weights[None, :, :]
    annual_log = _weighted_summary(
        np.log(np.maximum(model_annual[None, :, :], _PRECIP_LOG_FLOOR_MM_DAY)),
        np.log(np.maximum(reference_annual[None, :, :], _PRECIP_LOG_FLOOR_MM_DAY)),
        annual_weights[0],
    )
    return {
        "source": str(reference.metadata["source"]),
        "period": str(reference.metadata["period"]),
        "grid": {"height": reference.shape[0], "width": reference.shape[1]},
        "scored_area_fraction": float(np.mean(weights > 0.0)),
        "temperature_c": {
            "monthly_bias": temperature["bias"],
            "monthly_rmse": temperature["rmse"],
            "monthly_correlation": temperature["correlation"],
        },
        "precipitation_mm_day": {
            "monthly_bias": precipitation["bias"],
            "monthly_rmse": precipitation["rmse"],
            "monthly_correlation": precipitation["correlation"],
            "monthly_log_rmse": log_precipitation["rmse"],
            "monthly_log_correlation": log_precipitation["correlation"],
            "annual_log_rmse": annual_log["rmse"],
            "annual_log_correlation": annual_log["correlation"],
        },
    }
