"""Portable exchange contracts for offline external dynamical-core references.

PlanetSim remains the interactive model.  This module makes an external GCM a
reproducible *reference* experiment: terrain/planet forcing is exported to a
safe NPZ request, and postprocessed monthly fields are normalized into the
same climatology contract used for CRU scoring.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import numpy as np

from masks import get_masks
from monthly_climatology import (
    MonthlyClimatology,
    regrid_monthly_climatology,
    save_monthly_climatology,
    score_monthly_climatology,
)
from planet_params import PlanetParams


EXTERNAL_DYCORE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExoPlaSimRequest:
    """Fully explicit offline ExoPlaSim forcing request."""

    resolution: str = "T21"
    height: int = 32
    width: int = 64
    layers: int = 10
    spinup_years: int = 20
    evaluation_years: int = 30

    def validate(self) -> None:
        if self.height < 8 or self.width != 2 * self.height:
            raise ValueError("external dycore grid must be a 2:1 global grid")
        if self.layers < 3 or self.spinup_years < 0 or self.evaluation_years < 1:
            raise ValueError("layers must be >= 3, spinup non-negative, and evaluation >= 1 year")


def _regrid_2d(field: np.ndarray, height: int, width: int) -> np.ndarray:
    """Area-conservatively regrid a scalar field through the shared contract."""
    source = np.asarray(field, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2 * source.shape[0]:
        raise ValueError("field must be a 2:1 global grid")
    repeated = np.broadcast_to(source, (12, *source.shape)).copy()
    climatology = MonthlyClimatology(
        temperature_k=repeated,
        precipitation_mm_day=np.maximum(repeated, 0.0),
        metadata={"schema_version": 1, "source": "PlanetSim exchange", "period": "static"},
    )
    return regrid_monthly_climatology(climatology, height, width).temperature_k[0]


def export_exoplasim_request(
    elevation: np.ndarray,
    planet: PlanetParams,
    path: str | Path,
    *,
    request: ExoPlaSimRequest = ExoPlaSimRequest(),
) -> Path:
    """Export terrain and planetary forcing without importing ExoPlaSim.

    ``topography_m`` and fractional ``land_fraction`` are intentionally kept
    in the request rather than writing ExoPlaSim's SRA files here.  The Linux
    runner writes those format-specific boundary files using ExoPlaSim's own
    supported writer, keeping the binary/text format out of PlanetSim.
    """
    request.validate()
    elevation = np.asarray(elevation, dtype=np.float64)
    if elevation.ndim != 2 or elevation.shape[1] != 2 * elevation.shape[0]:
        raise ValueError("elevation must be a 2:1 global grid")
    _, land = get_masks(elevation.astype(np.float32, copy=False))
    topography_m = _regrid_2d(
        np.clip(elevation, 0.0, 1.0) * float(planet.max_elevation_km) * 1000.0,
        request.height,
        request.width,
    )
    land_fraction = _regrid_2d(land.astype(np.float64), request.height, request.width)
    metadata: dict[str, Any] = {
        "schema_version": EXTERNAL_DYCORE_SCHEMA_VERSION,
        "engine": "ExoPlaSim",
        "request": asdict(request),
        "coordinate_convention": "north-to-south rows; longitude [-180, 180)",
        "planet": {
            "surface_gravity_m_s2": float(planet.surface_gravity),
            "radius_m": float(planet.radius_m),
            "surface_pressure_pa": float(planet.surface_pressure_pa),
            "solar_constant_w_m2": float(planet.solar_constant),
            "obliquity_deg": float(planet.obliquity_deg),
            "orbital_period_days": float(planet.orbital_period_days),
            "sidereal_day_hours": float(planet.sidereal_day_hours),
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        topography_m=topography_m.astype(np.float32),
        land_fraction=np.clip(land_fraction, 0.0, 1.0).astype(np.float32),
        latitude_deg=(90.0 - (np.arange(request.height) + 0.5) * 180.0 / request.height).astype(np.float32),
        longitude_deg=(-180.0 + (np.arange(request.width) + 0.5) * 360.0 / request.width).astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return path


def load_exoplasim_request(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and validate a portable external-dycore forcing request."""
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"topography_m", "land_fraction", "latitude_deg", "longitude_deg", "metadata_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"external dycore request missing: {', '.join(sorted(missing))}")
        metadata = json.loads(str(archive["metadata_json"].reshape(-1)[0]))
        if metadata.get("schema_version") != EXTERNAL_DYCORE_SCHEMA_VERSION:
            raise ValueError("unsupported external dycore request schema")
        arrays = {name: np.asarray(archive[name]) for name in required if name != "metadata_json"}
    h, w = arrays["topography_m"].shape
    if w != 2 * h or arrays["land_fraction"].shape != (h, w):
        raise ValueError("external dycore terrain fields must share a 2:1 grid")
    return arrays, metadata


def _monthly(field: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim == 2:
        return np.broadcast_to(values, (12, *values.shape)).copy()
    if values.ndim != 3:
        raise ValueError(f"{name} must have shape (time, H, W) or (H, W)")
    if values.shape[0] == 12:
        return values
    if values.shape[0] < 12:
        raise ValueError(f"{name} needs at least 12 time samples")
    bins = np.array_split(values, 12, axis=0)
    return np.stack([np.mean(part, axis=0) for part in bins], axis=0)


def average_exoplasim_archives(
    archive_paths: Iterable[str | Path], *, temperature_key: str, precipitation_key: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Average each archive's monthly normal across a multi-year evaluation.

    Snapshot archives may contain more than twelve output times.  Reducing
    each year to twelve months *before* averaging prevents a concatenated
    24-sample snapshot sequence from mixing neighbouring calendar months.
    """
    paths = [Path(path) for path in archive_paths]
    if not paths:
        raise ValueError("at least one ExoPlaSim archive is required")
    temperature_sum: np.ndarray | None = None
    precipitation_sum: np.ndarray | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            missing = [key for key in (temperature_key, precipitation_key) if key not in archive.files]
            if missing:
                raise ValueError(f"{path.name} lacks requested field(s): {', '.join(missing)}")
            temperature = _monthly(archive[temperature_key], name=temperature_key)
            precipitation = _monthly(archive[precipitation_key], name=precipitation_key)
        if temperature_sum is None:
            temperature_sum = np.zeros_like(temperature)
            precipitation_sum = np.zeros_like(precipitation)
        if temperature.shape != temperature_sum.shape or precipitation.shape != precipitation_sum.shape:
            raise ValueError("ExoPlaSim archive fields have inconsistent shapes")
        temperature_sum += temperature
        precipitation_sum += precipitation
    assert temperature_sum is not None and precipitation_sum is not None
    return temperature_sum / len(paths), precipitation_sum / len(paths), len(paths)


def _resolve_exoplasim_archive(raw_path: str | Path) -> Path:
    """Find ExoPlaSim's regular archive after ``Model.finalize``.

    ``finalize`` takes an *output directory*, despite a tempting filename-like
    argument.  It places the regular model archive beside snapshot and
    metadata archives there.  Accepting the directory keeps the public bridge
    faithful to that API while still permitting a direct archive path for
    fixtures and other postprocessors.
    """
    path = Path(raw_path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise ValueError(f"ExoPlaSim output does not exist: {path}")
    candidates = sorted(
        candidate
        for candidate in path.glob("*.npz")
        if not candidate.name.endswith("_metadata.npz") and "_snapshot" not in candidate.stem
    )
    if len(candidates) != 1:
        names = ", ".join(candidate.name for candidate in candidates) or "none"
        raise ValueError(f"could not uniquely identify regular ExoPlaSim archive in {path}: {names}")
    return candidates[0]


def canonicalize_exoplasim_output(
    raw_path: str | Path,
    request_path: str | Path,
    output_path: str | Path,
    *,
    temperature_key: str = "ts",
    precipitation_key: str = "pr",
    precipitation_units: Literal["mm_day", "kg_m2_s", "m_s"] = "kg_m2_s",
    runner_provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Normalize ExoPlaSim postprocessor output to PlanetSim's monthly NPZ contract."""
    arrays, request = load_exoplasim_request(request_path)
    archive_path = _resolve_exoplasim_archive(raw_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        if temperature_key not in archive.files or precipitation_key not in archive.files:
            raise ValueError(
                f"missing requested ExoPlaSim fields; available: {', '.join(sorted(archive.files))}"
            )
        temperature = _monthly(archive[temperature_key], name=temperature_key)
        precipitation = _monthly(archive[precipitation_key], name=precipitation_key)
    if temperature.shape != precipitation.shape or temperature.shape[1:] != arrays["topography_m"].shape:
        raise ValueError("ExoPlaSim outputs do not match the request grid")
    if precipitation_units == "kg_m2_s":
        precipitation = precipitation * 86400.0
    elif precipitation_units == "m_s":
        precipitation = precipitation * 86400.0 * 1000.0
    elif precipitation_units != "mm_day":
        raise ValueError("unknown precipitation units")
    reference = MonthlyClimatology(
        temperature_k=temperature,
        precipitation_mm_day=np.maximum(precipitation, 0.0),
        land_fraction=np.asarray(arrays["land_fraction"], dtype=np.float64),
        metadata={
            "schema_version": 1,
            "source": "ExoPlaSim external-dycore reference",
            "period": f"{request['request']['evaluation_years']}-year model climatology",
            "engine_request": request,
            "temperature_key": temperature_key,
            "precipitation_key": precipitation_key,
            "precipitation_units_input": precipitation_units,
            "raw_archive": archive_path.name,
            "external_runner": dict(runner_provenance or {}),
        },
    )
    return save_monthly_climatology(reference, output_path)


def score_external_dycore_against_cru(
    model: MonthlyClimatology,
    cru_reference: MonthlyClimatology,
    *,
    minimum_land_fraction: float = 0.5,
) -> dict[str, Any]:
    """Score an external monthly climate using PlanetSim's CRU/Köppen contract.

    The external model is not reclassified with an additional terrain lapse
    adjustment: its surface temperature already includes the terrain supplied
    to its dynamical core.  This avoids silently double-counting altitude when
    comparing it with the native solver's map score.
    """
    reference = regrid_monthly_climatology(cru_reference, *model.shape)
    land_mask = None
    if model.land_fraction is not None:
        land_mask = np.asarray(model.land_fraction >= minimum_land_fraction, dtype=bool)
    monthly_score = score_monthly_climatology(
        model.temperature_k,
        model.precipitation_mm_day,
        reference,
        model_land_mask=land_mask,
        minimum_land_fraction=minimum_land_fraction,
    )
    koppen_score: dict[str, Any] = {}
    if land_mask is not None and np.any(land_mask):
        from climate_averages import classify_koppen
        from koppen_reference import score_koppen_map

        codes = classify_koppen(
            model.temperature_k,
            model.precipitation_mm_day,
            land_mask,
            elevation=None,
            orbital_period_days=float(
                model.metadata.get("engine_request", {})
                .get("planet", {})
                .get("orbital_period_days", 365.2422)
            ),
        )
        report = score_koppen_map(codes, land_mask=land_mask)
        if "group" in report:
            koppen_score = {
                "group_accuracy": report["group"]["accuracy"],
                "group_kappa": report["group"]["kappa"],
                "group_share_mae_pp": report["group"]["share_mae_pp"],
                "class_accuracy": report["class"]["accuracy"],
                "class_kappa": report["class"]["kappa"],
                "scored_cells": report["scored_cells"],
                "group_accuracy_by_zone": report["group_accuracy_by_zone"],
                "region_group_accuracy": {
                    name: (None if value is None else value["group_accuracy"])
                    for name, value in report["per_region"].items()
                },
            }
    return {
        "monthly_climatology": monthly_score,
        "koppen_map_skill": koppen_score,
        "model_grid": {"height": model.shape[0], "width": model.shape[1]},
        "minimum_land_fraction": minimum_land_fraction,
    }


def score_native_against_external_dycore(
    native: MonthlyClimatology,
    external: MonthlyClimatology,
    *,
    native_land_mask: np.ndarray | None = None,
    minimum_land_fraction: float = 0.5,
) -> dict[str, Any]:
    """Compare native monthly climate directly with the ExoPlaSim reference.

    This remains a falsification metric, not an online coupling: ExoPlaSim's
    temperatures are never injected into PlanetSim.  The external climate is
    regridded to the native grid so its precipitation placement can be compared
    alongside the common CRU scores.
    """
    reference = regrid_monthly_climatology(external, *native.shape)
    land_mask = native_land_mask
    if land_mask is None and native.land_fraction is not None:
        land_mask = np.asarray(native.land_fraction >= minimum_land_fraction, dtype=bool)
    if land_mask is not None and np.asarray(land_mask).shape != native.shape:
        raise ValueError("native land mask must match the native monthly grid")
    return {
        "monthly_climatology": score_monthly_climatology(
            native.temperature_k,
            native.precipitation_mm_day,
            reference,
            model_land_mask=land_mask,
            minimum_land_fraction=minimum_land_fraction,
        ),
        "native_grid": {"height": native.shape[0], "width": native.shape[1]},
        "external_grid": {"height": external.shape[0], "width": external.shape[1]},
        "minimum_land_fraction": minimum_land_fraction,
    }
