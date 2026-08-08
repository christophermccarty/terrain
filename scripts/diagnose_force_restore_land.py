"""Report regional energy terms for the gated force-restore land experiment."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planet_params import EARTH  # noqa: E402
from real_terrain_validation import load_bundled_earth_dem  # noqa: E402
from regional_validation import EARTH_PRECIP_REGIONS, region_mask  # noqa: E402
from simulate import create_initial_state, simulate_step  # noqa: E402
from simulation_state import TimeScaleMode  # noqa: E402


def _mean(field: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(np.asarray(field)[mask])) if np.any(mask) else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--spinup-years", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.spinup_years < 0:
        raise SystemExit("--spinup-years must be non-negative")
    pp = dataclasses.replace(EARTH, enable_force_restore_land=True)
    elevation = load_bundled_earth_dem(args.height, args.width)
    state = create_initial_state(elevation, planet_params=pp)
    dt = float(pp.orbital_period_days) / 12.0
    for _ in range(args.spinup_years * 12):
        state, _ = simulate_step(state, days=dt, planet_params=pp, time_scale=TimeScaleMode.MONTHLY)

    series: dict[str, dict[str, list[float]]] = {
        region.name: {key: [] for key in (
            "surface_temp_c", "air_temp_c", "deep_temp_c", "net_radiation_w_m2",
            "latent_heat_w_m2", "sensible_heat_w_m2",
        )}
        for region in EARTH_PRECIP_REGIONS
    }
    for _ in range(12):
        state, components = simulate_step(
            state, days=dt, planet_params=pp, time_scale=TimeScaleMode.MONTHLY,
            track_components=True,
        )
        for region in EARTH_PRECIP_REGIONS:
            mask = region_mask(state.temperature.shape, region, cell_mask=elevation > 0.0)
            entries = series[region.name]
            entries["surface_temp_c"].append(_mean(state.temperature - 273.15, mask))
            entries["air_temp_c"].append(_mean(state.air_temperature - 273.15, mask))
            entries["deep_temp_c"].append(_mean(state.land_deep_temperature - 273.15, mask))
            entries["net_radiation_w_m2"].append(_mean(components["net_radiation"], mask))
            entries["latent_heat_w_m2"].append(_mean(components["land_latent_heat_w_m2"], mask))
            entries["sensible_heat_w_m2"].append(_mean(components["land_sensible_heat_w_m2"], mask))
    report = {
        "config": {"height": args.height, "width": args.width, "spinup_years": args.spinup_years},
        "regions": {
            name: {
                key: {
                    "annual_mean": float(np.nanmean(values)),
                    "seasonal_range": float(np.nanmax(values) - np.nanmin(values)),
                }
                for key, values in terms.items()
            }
            for name, terms in series.items()
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
