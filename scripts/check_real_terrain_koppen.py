"""check_real_terrain_koppen.py — Real-terrain Koppen/precip/temperature checkpoint.

Continues a real-terrain save (default `saves/earth.pkl`, 512x1024) forward a
fixed window and reports the diagnostics used to track the desert-over-
extension / continental-interior-dryness gap and the mid-latitude land
winter cold-bias gap (see known-physics-gaps.md items on both):

- Arid / humid-temperate-continental / polar / tropical land-fraction
  breakdown (Koppen classification)
- Named region precipitation (desert boxes: Sahara/Kalahari/Atacama;
  continental-interior boxes: Canadian Prairies/US Midwest/Central Europe)
- 45-50 N/S coldest-month land temperature

Meant to be run manually at each calibration step (too slow for CI), not a
one-off throwaway -- accepts arbitrary PlanetParams overrides so it doubles
as the measurement step for parameter sweeps.

IMPORTANT: `koppen_type` and `climate_precip_avg` are 10-year exponential
moving averages (climate_averages.update_climate_averages, window_days=3650).
A short continuation (e.g. 1yr) barely nudges them if the save already has a
long history -- after `days` of new physics the EMA is still
exp(-days/3650)-weighted toward whatever it was before. This script reports
the *instantaneous* precipitation field averaged over the run's second half
as the fast, physics-reflecting signal; the EMA-based numbers are printed
too, for context, but expect them to lag by design unless `--days` is in the
multi-decade range (see known-physics-gaps.md).

Usage
-----
    python scripts/check_real_terrain_koppen.py [--save PATH] [--days N]
        [--time-scale MONTHLY] [--param name=value ...]

Example
-------
    python scripts/check_real_terrain_koppen.py --days 365 --time-scale MONTHLY
    python scripts/check_real_terrain_koppen.py --param moisture_advection_scale=0.5
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (name, lat_n, lat_s, lon_w, lon_e) -- lon in [-180, 180], lat in [-90, 90]
DESERT_BOXES = [
    ("Sahara", 30.0, 15.0, -10.0, 30.0),
    ("Kalahari", -20.0, -28.0, 15.0, 25.0),
    ("Atacama", -20.0, -28.0, -71.0, -68.0),
]
CONTINENTAL_BOXES = [
    ("Canadian Prairies", 55.0, 50.0, -110.0, -100.0),
    ("US Midwest", 45.0, 38.0, -100.0, -90.0),
    ("Central Europe", 53.0, 47.0, 5.0, 20.0),
]


def _box_slice(H: int, W: int, lat_n: float, lat_s: float, lon_w: float, lon_e: float):
    row0 = int(round((90.0 - lat_n) / 180.0 * H))
    row1 = int(round((90.0 - lat_s) / 180.0 * H))
    col0 = int(round((lon_w + 180.0) / 360.0 * W))
    col1 = int(round((lon_e + 180.0) / 360.0 * W))
    return slice(max(0, row0), min(H, row1)), slice(max(0, col0), min(W, col1))


def _box_land_mask(land_mask, box):
    _, lat_n, lat_s, lon_w, lon_e = box
    H, W = land_mask.shape
    rows, cols = _box_slice(H, W, lat_n, lat_s, lon_w, lon_e)
    return rows, cols, land_mask[rows, cols]


def _box_instantaneous_precip_mm_day(state, land_mask, box) -> float | None:
    rows, cols, land = _box_land_mask(land_mask, box)
    if land.sum() == 0 or state.precipitation is None:
        return None
    return float(np.mean(state.precipitation[rows, cols][land]))


def _box_ema_precip_mm_yr(state, land_mask, box) -> float | None:
    rows, cols, land = _box_land_mask(land_mask, box)
    if land.sum() == 0 or state.climate_precip_avg is None:
        return None
    p = state.climate_precip_avg[rows, cols][land]
    return float(np.mean(p)) * 365.25


def _land_coldest_month_c(state, land_mask, lat_n: float, lat_s: float) -> float | None:
    H, W = land_mask.shape
    row0 = int(round((90.0 - lat_n) / 180.0 * H))
    row1 = int(round((90.0 - lat_s) / 180.0 * H))
    rows = slice(max(0, row0), min(H, row1))
    land = land_mask[rows, :]
    if land.sum() == 0 or state.monthly_temp is None:
        return None
    coldest = state.monthly_temp.min(axis=0)[rows, :][land] - 273.15
    return float(np.mean(coldest))


def _koppen_breakdown(state, land_mask) -> dict[str, float]:
    from climate_averages import KOPPEN_NAMES

    k = state.koppen_type
    if k is None:
        return {}
    vals, counts = np.unique(k[land_mask], return_counts=True)
    total = land_mask.sum()
    arid = sum(c for v, c in zip(vals, counts) if KOPPEN_NAMES.get(int(v), "").startswith(("BW", "BS")))
    humid = sum(c for v, c in zip(vals, counts) if KOPPEN_NAMES.get(int(v), "")[:2] in ("Cf", "Cs", "Cw", "Df", "Dw"))
    polar = sum(c for v, c in zip(vals, counts) if KOPPEN_NAMES.get(int(v), "").startswith(("ET", "EF")))
    tropical = sum(c for v, c in zip(vals, counts) if KOPPEN_NAMES.get(int(v), "").startswith(("Af", "Am", "Aw")))
    return {
        "arid_pct": 100.0 * arid / total,
        "humid_pct": 100.0 * humid / total,
        "polar_pct": 100.0 * polar / total,
        "tropical_pct": 100.0 * tropical / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-terrain Koppen/precip/temperature checkpoint")
    parser.add_argument("--save", type=str, default=str(ROOT / "saves" / "earth.pkl"))
    parser.add_argument("--days", type=float, default=365.0)
    parser.add_argument("--time-scale", type=str, default="MONTHLY",
                         choices=["DAILY", "WEEKLY", "MONTHLY", "ANNUAL"])
    parser.add_argument("--param", action="append", default=[],
                         help="PlanetParams override, name=value (repeatable)")
    args = parser.parse_args()

    from simulate import load_state, simulate_step, TimeScaleMode
    from masks import get_masks
    from planet_params import EARTH

    overrides: dict[str, float] = {}
    for spec in args.param:
        name, _, value = spec.partition("=")
        overrides[name] = float(value)
    pp = dataclasses.replace(EARTH, **overrides) if overrides else EARTH

    time_scale = TimeScaleMode[args.time_scale]
    step_days = {"DAILY": 1.0, "WEEKLY": 7.0, "MONTHLY": 30.44, "ANNUAL": 91.31}[args.time_scale]
    n_steps = max(1, int(round(args.days / step_days)))

    state = load_state(args.save)
    sea_mask, land_mask = get_masks(state.elevation)

    print(f"Continuing {args.save} forward {args.days:.0f} days ({args.time_scale}, {n_steps} steps)")
    if overrides:
        print(f"PlanetParams overrides: {overrides}")
    all_boxes = DESERT_BOXES + CONTINENTAL_BOXES
    instantaneous_samples: dict[str, list[float]] = {box[0]: [] for box in all_boxes}
    half = n_steps // 2
    for i in range(n_steps):
        state, _ = simulate_step(state, days=step_days, block_size=4, wind_block_size=4,
                                  time_scale=time_scale, planet_params=pp)
        if i >= half:
            for box in all_boxes:
                p = _box_instantaneous_precip_mm_day(state, land_mask, box)
                if p is not None:
                    instantaneous_samples[box[0]].append(p)

    print()
    print("=== Koppen land-fraction breakdown (10yr EMA -- lags a short run, see module docstring) ===")
    for k, v in _koppen_breakdown(state, land_mask).items():
        print(f"  {k}: {v:.1f}%")

    print()
    print("=== Desert box precip (want low, <200 mm/yr) ===")
    for box in DESERT_BOXES:
        name = box[0]
        inst = instantaneous_samples[name]
        inst_str = f"{np.mean(inst) * 365.25:.0f} mm/yr (instantaneous, 2nd half)" if inst else "no land in box"
        ema = _box_ema_precip_mm_yr(state, land_mask, box)
        ema_str = f"{ema:.0f} mm/yr (10yr EMA)" if ema is not None else "n/a"
        print(f"  {name}: {inst_str}  |  {ema_str}")

    print()
    print("=== Continental-interior box precip (want high, ~350-450 mm/yr) ===")
    for box in CONTINENTAL_BOXES:
        name = box[0]
        inst = instantaneous_samples[name]
        inst_str = f"{np.mean(inst) * 365.25:.0f} mm/yr (instantaneous, 2nd half)" if inst else "no land in box"
        ema = _box_ema_precip_mm_yr(state, land_mask, box)
        ema_str = f"{ema:.0f} mm/yr (10yr EMA)" if ema is not None else "n/a"
        print(f"  {name}: {inst_str}  |  {ema_str}")

    print()
    print("=== Mid-latitude coldest-month land T (want -5 to -20 C) ===")
    for label, lat_n, lat_s in [("45-50N", 50.0, 45.0), ("45-50S", -45.0, -50.0)]:
        t = _land_coldest_month_c(state, land_mask, lat_n, lat_s)
        print(f"  {label}: {t:.1f} C" if t is not None else f"  {label}: no land in band")


if __name__ == "__main__":
    main()
