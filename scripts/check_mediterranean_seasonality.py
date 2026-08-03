"""Winter-wet / summer-dry precipitation seasonality, the Koppen Cs gate's input.

Built for ACCURACY_AUDIT.md's missing-Mediterranean finding (H10-DONE): the
model emits **zero** Csa/Csb cells, and the root cause is not a classifier
threshold. `climate_averages.classify_koppen` reaches Cs only when

    P_summer_driest < P_winter_wettest / 3          (ratio < 0.333)

and the model's planet-wide *minimum* of that ratio is ~0.61 -- land
precipitation peaks in local summer essentially everywhere, so the
Mediterranean signal is not weak, it is inverted. The corresponding Cw gate
(`P_winter_driest < P_summer_wettest / 10`, i.e. ratio < 0.100) is a nearer
miss at ~0.18.

This script reports both ratios over the six real-world Mediterranean regions
and, for context, four monsoon regions, so a mechanism aimed at seasonal
dry-belt migration can be judged on the quantity the classifier actually reads
rather than on annual-mean precipitation, which is blind to it.

Usage
-----
    python scripts/check_mediterranean_seasonality.py --state saves/test.npz
    python scripts/check_mediterranean_seasonality.py --height 128 --width 256 \
        --spinup-years 2 --set drybelt_seasonal_response=0.5
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from climate_averages import KOPPEN_NAMES  # noqa: E402
from masks import get_masks  # noqa: E402
from planet_params import EARTH  # noqa: E402
from regional_validation import ClimateRegion, region_mask  # noqa: E402


def _box(name: str, lat_n: float, lat_s: float, lon_w: float, lon_e: float) -> ClimateRegion:
    return ClimateRegion(name, lat_n, lat_s, lon_w, lon_e, "seasonality", 0.0, 1.0e9)


# Real-world Koppen Cs cores (Csa/Csb), from the designated reference map.
MEDITERRANEAN_BOXES: tuple[ClimateRegion, ...] = (
    _box("Iberia/W Med", 41.0, 37.0, -8.0, 0.0),
    _box("Greece/W Turkey", 40.0, 36.0, 21.0, 30.0),
    _box("California", 40.0, 34.0, -123.0, -119.0),
    _box("Central Chile", -30.0, -36.0, -72.0, -70.0),
    _box("Cape Town", -32.0, -35.0, 18.0, 22.0),
    _box("SW Australia", -30.0, -35.0, 115.0, 119.0),
)

# Real-world Cwa cores, whose gate is the opposite phase (winter-dry monsoon).
MONSOON_BOXES: tuple[ClimateRegion, ...] = (
    _box("N India", 27.0, 22.0, 75.0, 85.0),
    _box("S China", 28.0, 23.0, 105.0, 115.0),
    _box("SE Brazil", -18.0, -23.0, -50.0, -43.0),
    _box("S Africa plateau", -23.0, -28.0, 26.0, 31.0),
)


def _seasonal_ratios(state, planet_params) -> tuple[np.ndarray, np.ndarray]:
    """(summer_driest / winter_wettest, winter_driest / summer_wettest) per cell.

    Mirrors `climate_averages.classify_koppen`'s own month masks exactly, so a
    number here is the number that gate sees.
    """
    monthly_precip = np.asarray(state.monthly_precip, dtype=np.float64)
    if monthly_precip.ndim != 3 or monthly_precip.shape[0] != 12:
        raise SystemExit("state has no (12,H,W) monthly_precip -- run it long enough to fill one")
    H = monthly_precip.shape[1]
    lat = (0.5 - (np.arange(H, dtype=np.float64) + 0.5) / H) * 180.0
    is_southern = np.broadcast_to((lat < 0.0)[:, None], monthly_precip.shape[1:])
    summer_nh = np.array([False, False, False, True, True, True,
                          True, True, True, False, False, False])
    is_summer = summer_nh[:, None, None] ^ is_southern[None, :, :]

    days_per_month = float(planet_params.orbital_period_days) / 12.0
    monthly_mm = monthly_precip * days_per_month
    summer_driest = np.where(is_summer, monthly_mm, np.inf).min(axis=0)
    winter_driest = np.where(is_summer, np.inf, monthly_mm).min(axis=0)
    summer_wettest = np.where(is_summer, monthly_mm, -np.inf).max(axis=0)
    winter_wettest = np.where(is_summer, -np.inf, monthly_mm).max(axis=0)
    cs_ratio = summer_driest / np.maximum(winter_wettest, 1e-6)
    cw_ratio = winter_driest / np.maximum(summer_wettest, 1e-6)
    return cs_ratio, cw_ratio


def _koppen_shares(state, land_mask: np.ndarray) -> dict[str, float]:
    if state.koppen_type is None:
        return {}
    H = land_mask.shape[0]
    lat = (0.5 - (np.arange(H, dtype=np.float64) + 0.5) / H) * 180.0
    weights = np.broadcast_to(np.cos(np.radians(lat))[:, None], land_mask.shape) * land_mask
    total = float(weights.sum())
    if total <= 0.0:
        return {}
    codes = np.asarray(state.koppen_type)
    shares: dict[str, float] = {}
    for code, name in sorted(KOPPEN_NAMES.items()):
        short = name.split(" - ")[0]
        shares[short] = 100.0 * float(weights[(codes == code) & land_mask].sum()) / total
    return shares


def _report(state, planet_params, land_mask, *, label: str) -> None:
    cs_ratio, cw_ratio = _seasonal_ratios(state, planet_params)
    print()
    print("=" * 78)
    print(f"  {label}")
    print("=" * 78)
    print("  Cs gate needs summer_driest/winter_wettest < 0.333 (Earth Med: ~0.2-0.4)")
    print(f"  {'region':<20s}{'Cs ratio':>12s}{'Cw ratio':>12s}{'cells':>8s}")
    print("  " + "-" * 52)
    for box in MEDITERRANEAN_BOXES:
        mask = region_mask(cs_ratio.shape, box, cell_mask=land_mask)
        if not mask.any():
            print(f"  {box.name:<20s}{'UNRESOLVED':>12s}")
            continue
        print(f"  {box.name:<20s}{np.median(cs_ratio[mask]):12.3f}"
              f"{np.median(cw_ratio[mask]):12.3f}{int(mask.sum()):8d}")
    print()
    print("  Cw gate needs winter_driest/summer_wettest < 0.100")
    print(f"  {'region':<20s}{'Cs ratio':>12s}{'Cw ratio':>12s}{'cells':>8s}")
    print("  " + "-" * 52)
    for box in MONSOON_BOXES:
        mask = region_mask(cw_ratio.shape, box, cell_mask=land_mask)
        if not mask.any():
            print(f"  {box.name:<20s}{'UNRESOLVED':>12s}")
            continue
        print(f"  {box.name:<20s}{np.median(cs_ratio[mask]):12.3f}"
              f"{np.median(cw_ratio[mask]):12.3f}{int(mask.sum()):8d}")

    land = np.asarray(land_mask, dtype=bool)
    print()
    print("  planet-wide over land:  Cs ratio  min {:.3f}  p1 {:.3f}  median {:.3f}"
          .format(float(cs_ratio[land].min()), float(np.percentile(cs_ratio[land], 1)),
                  float(np.median(cs_ratio[land]))))
    print("                          Cw ratio  min {:.3f}  p1 {:.3f}  median {:.3f}"
          .format(float(cw_ratio[land].min()), float(np.percentile(cw_ratio[land], 1)),
                  float(np.median(cw_ratio[land]))))
    print("  land cells passing the Cs gate (<0.333): {:.2f}%   Cw gate (<0.100): {:.2f}%"
          .format(100.0 * float(np.mean(cs_ratio[land] < 0.333)),
                  100.0 * float(np.mean(cw_ratio[land] < 0.100))))

    shares = _koppen_shares(state, land)
    if shares:
        missing = {k: shares.get(k, 0.0) for k in ("Csa", "Csb", "Cwa", "Cfc", "Dwd")}
        print("  area-weighted land share of the classes H10 found missing: "
              + "  ".join(f"{k} {v:.2f}%" for k, v in missing.items()))


def _coerce(name: str, text: str):
    field = {f.name: f for f in dataclasses.fields(EARTH)}.get(name)
    if field is not None and field.type in ("bool", bool):
        return text.strip().lower() in ("1", "true", "yes", "on")
    return float(text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--state", type=str, default=None)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--spinup-years", type=float, default=2.0)
    parser.add_argument("--evaluation-years", type=float, default=1.0)
    parser.add_argument("--set", type=str, action="append", default=[])
    args = parser.parse_args()

    planet_params = EARTH
    for override in args.set:
        name, _, value = override.partition("=")
        planet_params = dataclasses.replace(
            planet_params, **{name.strip(): _coerce(name.strip(), value)}
        )

    if args.state:
        from simulate import load_state

        state = load_state(args.state)
        label = f"{args.state} ({state.elevation.shape[0]}x{state.elevation.shape[1]})"
    else:
        from real_terrain_validation import (
            RealTerrainValidationConfig,
            run_real_terrain_validation,
        )

        config = RealTerrainValidationConfig(
            height=args.height,
            width=args.width,
            spinup_years=args.spinup_years,
            evaluation_years=args.evaluation_years,
        )
        state, _ = run_real_terrain_validation(config, planet_params=planet_params)
        label = f"fresh {args.height}x{args.width}, {args.spinup_years}yr spinup"
    if args.set:
        label += "  [" + ", ".join(args.set) + "]"

    _, land_mask = get_masks(state.elevation, use_cache=False)
    _report(state, planet_params, land_mask, label=label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
