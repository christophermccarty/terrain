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
# Open-ocean control boxes for --wind-diagnostics: baseline p_anom/convergence
# magnitude away from any land effect, for comparison against the desert/
# continental boxes above.
OCEAN_BOXES = [
    ("Mid-Pacific", 10.0, -10.0, -160.0, -140.0),
    ("Mid-Atlantic", 10.0, -10.0, -40.0, -20.0),
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


def _box_mean(field: np.ndarray, mask: np.ndarray, box) -> float | None:
    """Mean of `field` over `box`, restricted to cells where `mask` is True."""
    _, lat_n, lat_s, lon_w, lon_e = box
    H, W = mask.shape
    rows, cols = _box_slice(H, W, lat_n, lat_s, lon_w, lon_e)
    sel = mask[rows, cols]
    if sel.sum() == 0:
        return None
    return float(np.mean(field[rows, cols][sel]))


def _wind_diagnostic_fields(state, pp, time_scale_mode: str = "MONTHLY") -> tuple[dict, dict]:
    """Harvest the real per-term pressure-anomaly and convergence arrays for
    this state by calling the real wind/generate_precipitation formulas a
    second time with debug_fields (their actual return values are discarded)
    -- avoids duplicating those functions' formulas here, so this diagnostic
    can't silently drift out of sync with the real physics.

    Uses `generate_wind_field` (block_size=1, matching simulate.py's own
    MONTHLY/ANNUAL-mode call) for DAILY/WEEKLY-unrelated modes, since that is
    the function actually driving precip whenever `update_wind=False`
    (known-physics-gaps.md item 3b) -- NOT `evolve_wind`, which only runs
    prognostically in DAILY/WEEKLY mode. Falls back to `evolve_wind` for
    DAILY/WEEKLY so the decomposition always matches the real per-step path."""
    from atmosphere import evolve_wind, generate_wind_field, generate_precipitation

    wind_debug: dict = {}
    if state.temperature is not None:
        if time_scale_mode in ("MONTHLY", "ANNUAL"):
            H, W = state.elevation.shape
            generate_wind_field(
                H, W, day_of_year=int(state.day_of_year), block_size=1,
                temperature=state.temperature, elevation=state.elevation,
                time_days=state.total_days, planet_params=pp,
                jet_index_nh=state.jet_index_nh, jet_index_sh=state.jet_index_sh,
                jet_block_nh=(state.jet_block_lon_nh, state.jet_block_days_left_nh, state.jet_block_total_days_nh),
                jet_block_sh=(state.jet_block_lon_sh, state.jet_block_days_left_sh, state.jet_block_total_days_sh),
                debug_fields=wind_debug,
            )
        else:
            evolve_wind(
                state.wind_u, state.wind_v, state.temperature, None, state.elevation,
                dt_days=1.0, time_days=state.total_days, planet_params=pp,
                ice_cover=state.ice_cover,
                jet_index_nh=state.jet_index_nh, jet_index_sh=state.jet_index_sh,
                jet_block_nh=(state.jet_block_lon_nh, state.jet_block_days_left_nh, state.jet_block_total_days_nh),
                jet_block_sh=(state.jet_block_lon_sh, state.jet_block_days_left_sh, state.jet_block_total_days_sh),
                debug_fields=wind_debug,
            )
    precip_debug: dict = {}
    if state.temperature is not None and state.wind_u is not None:
        H, W = state.elevation.shape
        generate_precipitation(
            H, W, state.elevation,
            temperature=state.temperature, wind_u=state.wind_u, wind_v=state.wind_v,
            humidity=state.humidity, soil_moisture=state.soil_moisture,
            soil_moisture_deep=state.soil_moisture_deep,
            cloud_fraction=state.cloud_cover, day_of_year=int(state.day_of_year),
            dt_days=1.0, planet_params=pp, debug_fields=precip_debug,
        )
    return wind_debug, precip_debug


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
    parser.add_argument("--wind-diagnostics", action="store_true",
                         help="Also report p_anom term decomposition and div/ascent/conv "
                              "convergence-driver magnitudes per region (desert/continental/ocean), "
                              "averaged over the run's second half. See known-physics-gaps.md item "
                              "3b -- measures whether the wind model's own convergence signal "
                              "favors continental interior before attempting a structural fix.")
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
    # Mirrors main.py's SimulationThread.run() dispatch: MONTHLY/ANNUAL pass
    # update_wind=False, which routes wind through generate_wind_field's
    # cached diagnostic path instead of the prognostic evolve_wind -- this
    # script previously never set this flag at all (always defaulted to
    # True), meaning every past --time-scale MONTHLY/ANNUAL measurement with
    # this script was silently using DAILY-fidelity prognostic wind under a
    # MONTHLY-sized time step, NOT what a real GUI user at MONTHLY/ANNUAL
    # speed actually experiences. Fixed here to match real dispatch so this
    # diagnostic (and any precip numbers derived from it) reflects the real
    # per-mode code path. See known-physics-gaps.md item 3b.
    update_wind = args.time_scale in ("DAILY", "WEEKLY")

    state = load_state(args.save)
    sea_mask, land_mask = get_masks(state.elevation)

    print(f"Continuing {args.save} forward {args.days:.0f} days ({args.time_scale}, {n_steps} steps)")
    if overrides:
        print(f"PlanetParams overrides: {overrides}")
    all_boxes = DESERT_BOXES + CONTINENTAL_BOXES
    wind_boxes = DESERT_BOXES + CONTINENTAL_BOXES + OCEAN_BOXES
    instantaneous_samples: dict[str, list[float]] = {box[0]: [] for box in all_boxes}
    # field name -> box name -> list of per-step box-mean samples
    wind_samples: dict[str, dict[str, list[float]]] = {
        f: {box[0]: [] for box in wind_boxes}
        for f in ("p_thermal", "p_terrain", "zonal_blend_eff", "p_anom_synoptic", "div", "ascent", "conv",
                   "q", "precip_potential_prerescale", "remove_frac_prerescale", "rh_release", "convective", "orog",
                   "temp_norm", "qsat", "base_q", "soil", "land_evap", "ocean_evap",
                   "soil_deep", "soil_evap_factor")
    }
    half = n_steps // 2
    global_rescale_samples: list[float] = []
    for i in range(n_steps):
        state, _ = simulate_step(state, days=step_days, block_size=4, wind_block_size=4,
                                  time_scale=time_scale, planet_params=pp, update_wind=update_wind)
        if i >= half:
            for box in all_boxes:
                p = _box_instantaneous_precip_mm_day(state, land_mask, box)
                if p is not None:
                    instantaneous_samples[box[0]].append(p)
            if args.wind_diagnostics:
                wind_debug, precip_debug = _wind_diagnostic_fields(state, pp, args.time_scale)
                fields = {**wind_debug, **precip_debug}
                if "global_rescale_factor" in precip_debug:
                    global_rescale_samples.append(precip_debug["global_rescale_factor"])
                for fname, arr in fields.items():
                    if fname not in wind_samples or arr is None:
                        continue
                    for box in wind_boxes:
                        mask = sea_mask if box in OCEAN_BOXES else land_mask
                        v = _box_mean(arr, mask, box)
                        if v is not None:
                            wind_samples[fname][box[0]].append(v)

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

    if args.wind_diagnostics:
        print()
        print("=== Wind-diagnostics: p_anom decomposition (Pa) + convergence drivers ===")
        print("    (annual-mean of per-step box means, 2nd half of run; see known-physics-gaps.md 3b)")
        if global_rescale_samples:
            print(f"    global_rescale_factor (planet-wide, clipped [0.2,3.0]): {np.mean(global_rescale_samples):.3f}")
        header = f"  {'region':<20}" + "".join(f"{f:>16}" for f in wind_samples)
        print(header)
        for box in wind_boxes:
            name = box[0]
            row = f"  {name:<20}"
            for fname in wind_samples:
                vals = wind_samples[fname][name]
                row += f"{np.mean(vals):>16.4f}" if vals else f"{'n/a':>16}"
            print(row)


if __name__ == "__main__":
    main()
