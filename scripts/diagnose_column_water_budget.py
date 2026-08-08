"""Inspect the legacy rainfall path in common column-water units."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atmosphere import generate_precipitation  # noqa: E402
from planet_params import EARTH  # noqa: E402
from real_terrain_validation import RealTerrainValidationConfig, run_real_terrain_validation  # noqa: E402
from regional_validation import EARTH_PRECIP_REGIONS, region_mask  # noqa: E402


def _mean(field: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(field[mask])) if np.any(mask) else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--spinup-years", type=float, default=1.0)
    parser.add_argument(
        "--raw-conserved-column", action="store_true",
        help="Inspect the experimental conservative-transport/no-row-target path.",
    )
    args = parser.parse_args()
    pp = dataclasses.replace(
        EARTH, enable_prognostic_column_water=args.raw_conserved_column
    )
    state, _ = run_real_terrain_validation(
        RealTerrainValidationConfig(
            height=args.height, width=args.width,
            spinup_years=args.spinup_years, evaluation_years=1.0,
        ),
        planet_params=pp,
    )
    debug: dict = {}
    generate_precipitation(
        args.height, args.width, state.elevation,
        temperature=state.temperature, wind_u=state.wind_u, wind_v=state.wind_v,
        humidity=state.humidity, soil_moisture=state.soil_moisture,
        soil_moisture_deep=state.soil_moisture_deep, cloud_fraction=state.cloud_cover,
        day_of_year=state.day_of_year, dt_days=1.0,
        surface_pressure_hpa=pp.surface_pressure_pa / 100.0,
        planet_params=pp, debug_fields=debug,
    )
    fields = (
        "column_water_before_precip_mm", "column_water_rainout_removal_mm",
        "column_water_after_precip_mm", "column_water_evaporation_source_mm",
        "column_water_allocator_residual_mm",
    )
    land = state.elevation > 0.0
    print(f"Column-water diagnostic ({debug['column_water_mode']})")
    print("global land means [mm per call]")
    for field in fields:
        print(f"  {field:<42} {_mean(debug[field], land):10.5f}")
    print("regional land means [mm per call]")
    for region in EARTH_PRECIP_REGIONS:
        mask = region_mask(state.elevation.shape, region, cell_mask=land)
        removal = _mean(debug["column_water_rainout_removal_mm"], mask)
        source = _mean(debug["column_water_evaporation_source_mm"], mask)
        residual = _mean(debug["column_water_allocator_residual_mm"], mask)
        print(f"  {region.name:<20} removal={removal:8.4f} source={source:8.4f} residual={residual: .2e}")
    if args.raw_conserved_column:
        print(
            "  transport residual [mm m2] "
            f"{float(debug['column_water_transport_residual_mm_m2']): .3e}; relative "
            f"{float(debug['column_water_transport_relative_residual']): .3e}"
        )
        print(
            "  total budget relative residual "
            f"{float(debug['column_water_total_budget_relative_residual']): .3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
