"""Seasonally sample local winds and stability for offline Smith--Barstad tests."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orographic_linear import smith_barstad_precipitation_anomaly  # noqa: E402
from planet_params import EARTH  # noqa: E402
from real_terrain_validation import load_bundled_earth_dem  # noqa: E402
from regional_validation import OROGRAPHIC_PAIRS, region_mask  # noqa: E402
from scripts.check_linear_orographic_theory import _local_tile_slices  # noqa: E402
from simulate import create_initial_state, simulate_step  # noqa: E402
from simulation_state import TimeScaleMode  # noqa: E402


def _values(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated values")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--spinup-years", type=int, default=1)
    parser.add_argument("--moist-stabilities", type=_values, default=(0.004, 0.006, 0.008))
    parser.add_argument("--tile-buffer-degrees", type=float, default=6.0)
    parser.add_argument("--background-mm-day", type=float, default=1.0)
    parser.add_argument("--terrain-pgf-scale", type=float, default=1.0,
                        help="gated multiplier for resolved terrain PGF in the sampled wind run")
    args = parser.parse_args()
    if args.terrain_pgf_scale < 0.0:
        raise SystemExit("--terrain-pgf-scale must be non-negative")
    pp = dataclasses.replace(EARTH, wind_terrain_pgf_scale=args.terrain_pgf_scale)
    elevation = load_bundled_earth_dem(args.height, args.width)
    elevation_m = elevation * float(EARTH.max_elevation_km) * 1000.0
    state = create_initial_state(elevation, planet_params=pp)
    dt = float(pp.orbital_period_days) / 12.0
    for _ in range(args.spinup_years * 12):
        state, _ = simulate_step(state, days=dt, planet_params=pp, time_scale=TimeScaleMode.MONTHLY)
    samples = []
    for _ in range(12):
        state, _ = simulate_step(state, days=dt, planet_params=pp, time_scale=TimeScaleMode.MONTHLY)
        samples.append((state.wind_u.copy(), state.wind_v.copy()))

    dy = np.pi * float(EARTH.radius_m) / args.height
    print("Seasonal local-wind Smith-Barstad screen (offline; not precipitation coupling)")
    print("range                 mean wind  persist  stability       annual W/L    Earth target")
    for pair in OROGRAPHIC_PAIRS:
        rows, cols = _local_tile_slices(elevation.shape, pair, buffer_degrees=args.tile_buffer_degrees)
        terrain = elevation_m[rows, cols]
        windward = region_mask(elevation.shape, pair.windward)[rows, cols]
        leeward = region_mask(elevation.shape, pair.leeward)[rows, cols]
        pair_cells = windward | leeward
        lat = 90.0 - 180.0 * (0.5 * (rows.start + rows.stop)) / args.height
        dx = 2.0 * np.pi * float(EARTH.radius_m) * max(np.cos(np.deg2rad(lat)), 0.08) / args.width
        seasonal_wind = [
            (
                float(np.mean(wind_u[rows, cols][pair_cells])),
                float(np.mean(wind_v[rows, cols][pair_cells])),
            )
            for wind_u, wind_v in samples
        ]
        mean_u = float(np.mean([u for u, _ in seasonal_wind]))
        mean_v = float(np.mean([v for _, v in seasonal_wind]))
        mean_speed = float(np.mean([np.hypot(u, v) for u, v in seasonal_wind]))
        persistence = float(np.hypot(mean_u, mean_v) / max(mean_speed, 1e-12))
        for stability in args.moist_stabilities:
            annual = np.zeros_like(terrain, dtype=np.float64)
            for u, v in seasonal_wind:
                annual += np.maximum(
                    smith_barstad_precipitation_anomaly(
                        terrain, dx_m=dx, dy_m=dy, wind_u_m_s=u, wind_v_m_s=v,
                        latitude_deg=lat, moist_stability_s=stability,
                    ),
                    0.0,
                ) + args.background_mm_day
            ratio = float(np.mean(annual[windward]) / (np.mean(annual[leeward]) + 1e-12))
            print(
                f"{pair.name:<20} ({mean_u:4.1f},{mean_v:4.1f}) {persistence:5.2f}"
                f"    {stability:0.3f} s-1     {ratio:6.2f}       {pair.ratio_min:.0f}-{pair.ratio_max:.0f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
