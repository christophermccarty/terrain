"""Offline Smith--Barstad terrain test against PlanetSim's named mountain pairs.

This does not alter model precipitation.  It establishes whether the published
linear-theory mechanism produces the required windward/leeward footprint on
the bundled DEM before any moisture-budget integration is attempted.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orographic_linear import smith_barstad_precipitation_anomaly  # noqa: E402
from planet_params import EARTH  # noqa: E402
from real_terrain_validation import (  # noqa: E402
    RealTerrainValidationConfig,
    load_bundled_earth_dem,
    run_real_terrain_validation,
)
from regional_validation import OROGRAPHIC_PAIRS, region_mask  # noqa: E402


def _latitude(shape: tuple[int, int], lat_s: float, lat_n: float) -> float:
    return 0.5 * (lat_s + lat_n)


def _local_tile_slices(
    shape: tuple[int, int], pair, *, buffer_degrees: float
) -> tuple[slice, slice]:
    """Return a padded non-wrapping local tile enclosing an orographic pair."""
    H, W = shape
    lat_n = min(90.0, max(pair.windward.lat_n, pair.leeward.lat_n) + buffer_degrees)
    lat_s = max(-90.0, min(pair.windward.lat_s, pair.leeward.lat_s) - buffer_degrees)
    lon_w = min(pair.windward.lon_w, pair.leeward.lon_w) - buffer_degrees
    lon_e = max(pair.windward.lon_e, pair.leeward.lon_e) + buffer_degrees
    if lon_w < -180.0 or lon_e > 180.0:
        raise ValueError(f"{pair.name} tile crosses the dateline; add wrapped-tile support")
    r0 = max(0, int(np.floor((90.0 - lat_n) / 180.0 * H)))
    r1 = min(H, int(np.ceil((90.0 - lat_s) / 180.0 * H)))
    c0 = max(0, int(np.floor((lon_w + 180.0) / 360.0 * W)))
    c1 = min(W, int(np.ceil((lon_e + 180.0) / 360.0 * W)))
    if r1 - r0 < 4 or c1 - c0 < 4:
        raise ValueError(f"{pair.name} tile is unresolved at {H}x{W}")
    return slice(r0, r1), slice(c0, c1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--wind-speed", type=float, default=10.0)
    parser.add_argument("--background-mm-day", type=float, default=1.0)
    parser.add_argument("--conversion-time-s", type=float, default=1000.0)
    parser.add_argument("--fallout-time-s", type=float, default=1000.0)
    parser.add_argument("--local-model-wind", action="store_true",
                        help="run a 1+1 year model state and use each pair's local mean wind")
    parser.add_argument("--spinup-years", type=float, default=1.0,
                        help="used only with --local-model-wind")
    parser.add_argument("--tile-buffer-degrees", type=float, default=6.0,
                        help="terrain padding around each pair when using local wind")
    args = parser.parse_args()
    if args.height < 32 or args.width < 64:
        raise SystemExit("use at least 32x64 so a terrain range is represented")

    elevation = load_bundled_earth_dem(args.height, args.width)
    elevation_m = elevation * float(EARTH.max_elevation_km) * 1000.0
    dy = np.pi * float(EARTH.radius_m) / args.height
    state = None
    if args.local_model_wind:
        state, _ = run_real_terrain_validation(
            RealTerrainValidationConfig(
                height=args.height, width=args.width,
                spinup_years=args.spinup_years, evaluation_years=1.0,
            ),
            planet_params=EARTH,
        )
    print("Smith-Barstad offline terrain diagnostic (not coupled to rainfall)")
    print("range                 wind (u,v) m/s    W/L ratio     Earth target")
    for pair in OROGRAPHIC_PAIRS:
        lat = _latitude(elevation.shape, pair.windward.lat_s, pair.windward.lat_n)
        terrain = elevation_m
        windward = region_mask(elevation.shape, pair.windward)
        leeward = region_mask(elevation.shape, pair.leeward)
        if state is None:
            # Fixed westerly isolates transfer-function behavior.  The local
            # mode below is the integration-relevant experiment.
            u, v = float(args.wind_speed), 0.0
        else:
            rows, cols = _local_tile_slices(
                elevation.shape, pair, buffer_degrees=args.tile_buffer_degrees
            )
            terrain = elevation_m[rows, cols]
            pair_cells = (windward | leeward)[rows, cols]
            u = float(np.mean(state.wind_u[rows, cols][pair_cells]))
            v = float(np.mean(state.wind_v[rows, cols][pair_cells]))
            lat = 90.0 - 180.0 * (0.5 * (rows.start + rows.stop)) / args.height
            windward = windward[rows, cols]
            leeward = leeward[rows, cols]
        dx = 2.0 * np.pi * float(EARTH.radius_m) * max(np.cos(np.deg2rad(lat)), 0.08) / args.width
        anomaly = smith_barstad_precipitation_anomaly(
            terrain, dx_m=dx, dy_m=dy, wind_u_m_s=u, wind_v_m_s=v,
            latitude_deg=lat, conversion_time_s=args.conversion_time_s,
            fallout_time_s=args.fallout_time_s,
        )
        rate = np.maximum(anomaly, 0.0) + float(args.background_mm_day)
        ratio = float("nan") if not np.any(windward) or not np.any(leeward) else float(
            np.mean(rate[windward]) / (np.mean(rate[leeward]) + 1e-12)
        )
        print(f"{pair.name:<20} ({u:5.1f},{v:5.1f})       {ratio:6.2f}       {pair.ratio_min:.0f}-{pair.ratio_max:.0f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
