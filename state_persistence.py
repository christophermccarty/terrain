"""Versioned persistence for :class:`simulation_state.PlanetState`."""
from __future__ import annotations

import dataclasses
from datetime import datetime
import json
from pathlib import Path
import pickle

import numpy as np

from planet_params import PlanetParams
from simulation_state import PlanetState

STATE_SCHEMA_VERSION = 1


def _save_state_npz(state: PlanetState, filepath: Path) -> None:
    """Write a non-executable, versioned NumPy/JSON state archive."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, float | int | bool | str] = {}
    none_fields: list[str] = []
    planet_params_data: dict | None = None

    for field in PlanetState._fields:
        value = getattr(state, field)
        if isinstance(value, np.ndarray):
            arrays[field] = value
        elif value is None:
            none_fields.append(field)
        elif field == "planet_params":
            planet_params_data = dataclasses.asdict(value)
        elif isinstance(value, (float, int, bool, str)):
            scalars[field] = value
        else:
            raise TypeError(f"Cannot safely serialize PlanetState.{field}: {type(value)!r}")

    metadata = {
        "format": "planetsim-npz",
        "schema_version": STATE_SCHEMA_VERSION,
        "scalars": scalars,
        "none_fields": none_fields,
        "planet_params": planet_params_data,
    }
    arrays["__metadata__"] = np.asarray(json.dumps(metadata, separators=(",", ":")))

    temp_path = filepath.with_name(filepath.name + ".tmp.npz")
    try:
        np.savez_compressed(temp_path, **arrays)
        temp_path.replace(filepath)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    file_size_mb = filepath.stat().st_size / 1e6
    print(f"State saved safely to {filepath} ({file_size_mb:.1f} MB)")


def _load_state_npz(filepath: Path) -> PlanetState:
    """Load and validate a non-executable NumPy/JSON state archive."""
    with np.load(filepath, allow_pickle=False) as archive:
        if "__metadata__" not in archive.files:
            raise ValueError("Safe state archive is missing __metadata__")
        raw_metadata = archive["__metadata__"]
        if raw_metadata.ndim != 0 or raw_metadata.dtype.kind not in ("U", "S"):
            raise ValueError("Safe state metadata must be a scalar JSON string")
        metadata = json.loads(str(raw_metadata.item()))
        if metadata.get("format") != "planetsim-npz":
            raise ValueError("Not a PlanetSim safe state archive")
        version = int(metadata.get("schema_version", -1))
        if version > STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported save schema v{version} "
                f"(this build supports up to v{STATE_SCHEMA_VERSION})"
            )
        if version < 1:
            raise ValueError(f"Unsupported save schema v{version}")

        known_fields = set(PlanetState._fields)
        scalars = metadata.get("scalars", {})
        none_fields = set(metadata.get("none_fields", []))
        archive_fields = set(archive.files) - {"__metadata__"}
        unknown = (set(scalars) | none_fields | archive_fields) - known_fields
        if unknown:
            raise ValueError(f"Safe state contains unknown fields: {sorted(unknown)}")

        values = dict(PlanetState._field_defaults)
        values.update(scalars)
        values.update({field: None for field in none_fields})
        for field in archive_fields:
            values[field] = np.array(archive[field], copy=True)

    pp_data = metadata.get("planet_params")
    if pp_data is not None:
        pp_data = dict(pp_data)
        for field in dataclasses.fields(PlanetParams):
            if isinstance(field.default, tuple) and isinstance(pp_data.get(field.name), list):
                pp_data[field.name] = tuple(pp_data[field.name])
        values["planet_params"] = PlanetParams(**pp_data)
    else:
        values["planet_params"] = None
    if "day_of_year" not in values or "elevation" not in values:
        raise ValueError("Safe state is missing day_of_year or elevation")

    elevation = np.asarray(values["elevation"])
    if elevation.ndim != 2:
        raise ValueError(f"elevation must be 2D, got {elevation.shape}")
    H, W = elevation.shape
    for field, value in values.items():
        if not isinstance(value, np.ndarray) or field == "elevation":
            continue
        if field in ("monthly_temp", "monthly_precip"):
            expected = (12, H, W)
        elif field == "monthly_sample_count":
            expected = (12,)
        else:
            expected = (H, W)
        if value.shape != expected:
            raise ValueError(
                f"PlanetState.{field} has shape {value.shape}; expected {expected}"
            )

    state = PlanetState(**values)
    print(f"State loaded safely from {filepath} (day {state.total_days:.1f})")
    return state


def save_state(state: PlanetState, filepath: str | Path) -> None:
    """Save a state as safe NPZ or a legacy-compatible pickle envelope."""
    filepath = Path(filepath)
    if filepath.suffix.lower() == ".npz":
        _save_state_npz(state, filepath)
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)

    envelope = {
        "schema_version": STATE_SCHEMA_VERSION,
        "planet_params": state.planet_params,
        "state": state,
    }
    with open(filepath, "wb") as file:
        pickle.dump(envelope, file, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = filepath.stat().st_size / 1e6
    print(f"State saved to {filepath} ({file_size_mb:.1f} MB)")


def load_state(filepath: str | Path) -> PlanetState:
    """Load safe NPZ, versioned pickle, or a legacy raw pickle."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"State file not found: {filepath}")
    if filepath.suffix.lower() == ".npz":
        return _load_state_npz(filepath)

    with open(filepath, "rb") as file:
        obj = pickle.load(file)

    if isinstance(obj, dict) and "schema_version" in obj:
        version = int(obj["schema_version"])
        if version > STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported save schema v{version} "
                f"(this build supports up to v{STATE_SCHEMA_VERSION})"
            )
        state = obj["state"]
        if state.planet_params is None and obj.get("planet_params") is not None:
            state = state._replace(planet_params=obj["planet_params"])
    else:
        state = obj

    print(f"State loaded from {filepath} (day {state.total_days:.1f})")
    return state


def auto_save(
    state: PlanetState,
    save_dir: str | Path = "saves",
    every_n_days: float = 365,
) -> None:
    """Save state when its integer day reaches the requested interval."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    day_num = int(state.total_days)
    if day_num % int(every_n_days) == 0 and day_num > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"state_day{day_num:06d}_{timestamp}.pkl"
        save_state(state, save_dir / filename)
