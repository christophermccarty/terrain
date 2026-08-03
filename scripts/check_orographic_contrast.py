"""Windward/leeward precipitation contrast, measured at every pipeline stage.

Built for ACCURACY_AUDIT.md A5, and specifically to satisfy its process note 11:
none of the nine tracked named boxes is mountainous, so an orographic change
scores as "no effect" against the standard instrument by construction. These
purpose-built box pairs (``regional_validation.OROGRAPHIC_PAIRS``) resolve it.

The value of this script over a plain precipitation readout is that it reports
the windward:leeward ratio *at each stage of the precipitation pipeline*, so a
change that raises the orographic signal but has it absorbed downstream is
visibly distinguishable from one that has no effect at all. A5's three
documented absorption stages are all directly observable here:

    orog                       -> the raw uplift signal, after clip
    precip_potential           -> after the six-term weighted sum + smoothing
    remove_frac                -> after the 0.85 rain-out ceiling
    P (final)                  -> after the moisture-budget row rescale

Usage
-----
    # single-step ablation on an existing state (fast, no stepping)
    python scripts/check_orographic_contrast.py --state saves/earth.pkl

    # same, sweeping a parameter to see which stage responds
    python scripts/check_orographic_contrast.py --state saves/earth.pkl \
        --sweep orographic_uplift_clip=2,4,8,20

    # fresh spinup at a resolution that resolves the ranges (slower, full loop)
    python scripts/check_orographic_contrast.py --height 256 --width 512 \
        --spinup-years 1.0
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

from atmosphere import generate_precipitation  # noqa: E402
from masks import get_masks  # noqa: E402
from planet_params import EARTH  # noqa: E402
from regional_validation import (  # noqa: E402
    OROGRAPHIC_PAIRS,
    orographic_contrast,
    region_mask,
)

STAGES = (
    ("orog", "orog"),
    ("precip_potential_prerescale", "potential"),
    ("remove_frac_prerescale", "remove_frac"),
    ("P", "final P"),
)


def _single_step_fields(state, planet_params) -> dict:
    """Run one precipitation call and return its debug fields plus final P."""
    debug: dict = {}
    height, width = state.elevation.shape
    precipitation = generate_precipitation(
        height,
        width,
        state.elevation,
        temperature=state.temperature,
        wind_u=state.wind_u,
        wind_v=state.wind_v,
        humidity=state.humidity,
        soil_moisture=state.soil_moisture,
        soil_moisture_deep=state.soil_moisture_deep,
        cloud_fraction=state.cloud_cover,
        day_of_year=state.day_of_year,
        dt_days=1.0,
        surface_pressure_hpa=planet_params.surface_pressure_pa / 100.0,
        planet_params=planet_params,
        debug_fields=debug,
    )
    # generate_precipitation returns
    # (precip_mm_day, humidity, soil_moisture, soil_moisture_deep).
    debug["P"] = np.asarray(precipitation[0])
    return debug


def _report(fields: dict, land_mask: np.ndarray, planet_params, *, label: str) -> dict:
    print()
    print("=" * 84)
    print(f"  {label}")
    print("=" * 84)

    orog = np.asarray(fields["orog"])
    remove_frac = np.asarray(fields["remove_frac_prerescale"])
    clip_value = float(planet_params.orographic_uplift_clip)
    land = np.asarray(land_mask, dtype=bool)
    steep = orog[land] >= np.percentile(orog[land], 95.0)
    print(
        "  saturation:  orog at clip {:.4g}: {:5.2f}% of land, {:5.2f}% of the "
        "steepest 5%".format(
            clip_value,
            100.0 * float(np.mean(orog[land] >= clip_value - 1e-6)),
            100.0 * float(np.mean(orog[land][steep] >= clip_value - 1e-6)),
        )
    )
    print(
        "               remove_frac at its 0.85 ceiling: {:5.2f}% of land".format(
            100.0 * float(np.mean(remove_frac[land] >= 0.85 - 1e-6))
        )
    )

    header = "  {:<15s}".format("range") + "".join(f"{name:>13s}" for _, name in STAGES)
    print()
    print(header + "      Earth")
    print("  " + "-" * (len(header) + 8))
    ratios: dict[str, float] = {}
    for pair in OROGRAPHIC_PAIRS:
        cells_w = int(np.count_nonzero(region_mask(orog.shape, pair.windward, cell_mask=land)))
        cells_l = int(np.count_nonzero(region_mask(orog.shape, pair.leeward, cell_mask=land)))
        row = f"  {pair.name:<15s}"
        for key, _ in STAGES:
            if key not in fields:
                row += f"{'--':>13s}"
                continue
            contrast = orographic_contrast(np.asarray(fields[key]), pair, land_mask=land)
            if contrast is None:
                row += f"{'--':>13s}"
                continue
            row += f"{contrast['ratio']:13.2f}"
            if key == "P":
                ratios[pair.name] = contrast["ratio"]
        row += f"   {pair.ratio_min:.0f}-{pair.ratio_max:.0f}x"
        if cells_w == 0 or cells_l == 0:
            row += "  (UNRESOLVED)"
        print(row)

    finite = [value for value in ratios.values() if np.isfinite(value)]
    if finite:
        deficits = [
            max(0.0, pair.ratio_min - ratios[pair.name]) / pair.ratio_min
            for pair in OROGRAPHIC_PAIRS
            if pair.name in ratios and np.isfinite(ratios[pair.name])
        ]
        print()
        print(
            "  mean final W/L ratio {:.2f}   mean shortfall vs Earth's floor "
            "{:.1%}".format(float(np.mean(finite)), float(np.mean(deficits)))
        )
    return ratios


def _coerce(name: str, text: str):
    """Match the declared type of the PlanetParams field (bool fields exist)."""
    field = {f.name: f for f in dataclasses.fields(EARTH)}.get(name)
    if field is not None and field.type in ("bool", bool):
        return text.strip().lower() in ("1", "true", "yes", "on")
    return float(text)


def _parse_sweep(text: str) -> tuple[str, list]:
    name, _, values = text.partition("=")
    if not values:
        raise SystemExit(f"--sweep needs NAME=v1,v2,...  (got {text!r})")
    name = name.strip()
    return name, [_coerce(name, value) for value in values.split(",")]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--state", type=str, default=None,
                        help="existing save to run a single-step ablation against")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--spinup-years", type=float, default=0.0,
                        help="spin up a fresh state instead of loading one")
    parser.add_argument("--sweep", type=str, default=None,
                        help="PlanetParams field to sweep, e.g. orographic_uplift_clip=2,4,8")
    parser.add_argument("--set", type=str, action="append", default=[],
                        help="override a PlanetParams field, e.g. --set orog_weight=0.5")
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
    else:
        from real_terrain_validation import (
            RealTerrainValidationConfig,
            run_real_terrain_validation,
        )

        config = RealTerrainValidationConfig(
            height=args.height,
            width=args.width,
            spinup_years=max(args.spinup_years, 0.0),
            evaluation_years=1.0,
        )
        state, _ = run_real_terrain_validation(config, planet_params=planet_params)

    _, land_mask = get_masks(state.elevation, use_cache=False)
    height, width = state.elevation.shape
    if height < 256:
        print(
            f"WARNING: {height}x{width} does not resolve these ranges "
            "(a cell spans the whole crest); use >= 256x512.",
            file=sys.stderr,
        )

    if args.sweep:
        name, values = _parse_sweep(args.sweep)
        for value in values:
            swept = dataclasses.replace(planet_params, **{name: value})
            fields = _single_step_fields(state, swept)
            _report(fields, land_mask, swept, label=f"{name} = {value}")
    else:
        fields = _single_step_fields(state, planet_params)
        _report(fields, land_mask, planet_params,
                label=f"baseline ({height}x{width})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
