"""Score the simulated Köppen map against the gridded reference (audit H10).

Unlike the named-box and zonal-band checks, this reports a per-cell verdict, so
a regional pattern error that a spatial average hides shows up directly.

Usage
-----
    # tracked deterministic benchmark (64x128, 1yr spinup + 1yr eval, ~15s)
    python scripts/check_koppen_map_skill.py

    # an existing save at its own resolution (instant -- no stepping)
    python scripts/check_koppen_map_skill.py --state saves/earth.pkl

    # higher-resolution fresh run, full per-class tables, and an error map PNG
    python scripts/check_koppen_map_skill.py --height 128 --width 256 \
        --detail --error-map koppen_errors.png

Note on saves: ``state.koppen_type`` is a long-window climatological average, so
a save carries the classification its *history* produced. After a physics change,
prefer the fresh-spinup benchmark -- see ACCURACY_AUDIT.md A4, where reading a
lagging EMA off a continuation run produced two collateral effects that a fresh
run showed did not exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from koppen_reference import (  # noqa: E402
    GROUP_DESCRIPTIONS,
    earth_group_shares,
    load_reference_grid,
    score_koppen_map,
    short_class_name,
)
from masks import get_masks  # noqa: E402


def _print_headline(report: dict) -> None:
    group, klass = report["group"], report["class"]
    print("=" * 78)
    print(
        f"GRIDDED KÖPPEN MAP SKILL  "
        f"({report['grid']['height']}x{report['grid']['width']}, "
        f"{report['scored_cells']} cells scored)"
    )
    print("=" * 78)
    print(
        "  group (A/B/C/D/E)   accuracy {:.3f}   kappa {:.3f}   share MAE {:.2f} pp".format(
            group["accuracy"], group["kappa"], group["share_mae_pp"]
        )
    )
    print(
        "  class (Af/BWh/...)  accuracy {:.3f}   kappa {:.3f}".format(
            klass["accuracy"], klass["kappa"]
        )
    )
    mismatch = report["coastline_mismatch"]
    print(
        "  coastline mismatch: {:.1f}% model-only land, {:.1f}% reference-only "
        "(excluded from scoring)".format(
            mismatch["model_land_not_reference_pct"],
            mismatch["reference_land_not_model_pct"],
        )
    )
    print()
    print("  group    model%   ref%    prec  recall     f1")
    for name, values in group["per_class"].items():
        print(
            "    {} {:8.1f} {:6.1f}   {:5.3f}  {:5.3f}  {:5.3f}   {}".format(
                name,
                values["model_share_pct"],
                values["reference_share_pct"],
                values["precision"],
                values["recall"],
                values["f1"],
                GROUP_DESCRIPTIONS[name],
            )
        )


def _print_confusion(report: dict) -> None:
    confusion = np.array(report["group"]["confusion"]["matrix"], dtype=float)
    labels = report["group"]["confusion"]["labels"]
    total = confusion.sum()
    print()
    print("  confusion, % of scored land area (rows = reference, cols = model):")
    print("            " + "".join(f"{name:>8s}" for name in labels))
    for index, name in enumerate(labels):
        cells = "".join(f"{100.0 * value / total:8.2f}" for value in confusion[index])
        print(f"    ref {name}   {cells}")


def _print_regions(report: dict) -> None:
    print()
    print("  named regions (the boxes the rest of the suite tracks):")
    print("    region                 group%   class%   model -> reference")
    for name, values in report["per_region"].items():
        if values is None:
            print(f"    {name:20s}       --       --   (no scored cells at this resolution)")
            continue
        flag = "  <-- group mismatch" if values["group_accuracy"] < 0.5 else ""
        print(
            "    {:20s} {:7.1f}  {:7.1f}   {:4s} -> {}{}".format(
                name,
                100.0 * values["group_accuracy"],
                100.0 * values["class_accuracy"],
                values["model_dominant"],
                values["reference_dominant"],
                flag,
            )
        )


def _print_zones(report: dict) -> None:
    print()
    print("  group accuracy by latitude band (worst bands are where to look):")
    entries = sorted(
        report["group_accuracy_by_zone"].items(),
        key=lambda item: int(item[0].split(":")[0]),
    )
    for name, value in entries:
        low, high = name.split(":")
        bar = "#" * int(round(value * 40))
        print(f"    {low:>4s}..{high:<4s} {100.0 * value:5.1f}%  {bar}")


def _print_class_detail(report: dict) -> None:
    print()
    print("  per-class detail (model vocabulary; reference folded onto it):")
    print("    class   model%    ref%    prec  recall      f1")
    for name, values in report["class"]["per_class"].items():
        if values["model_share_pct"] == 0.0 and values["reference_share_pct"] == 0.0:
            continue
        print(
            "    {:6s} {:7.2f} {:7.2f}   {:5.3f}  {:5.3f}  {:6.3f}".format(
                name,
                values["model_share_pct"],
                values["reference_share_pct"],
                values["precision"],
                values["recall"],
                values["f1"],
            )
        )


def _write_error_map(path: Path, model_codes, land_mask, height, width) -> None:
    from PIL import Image

    from koppen_reference import koppen_group

    reference = load_reference_grid(height, width)
    scored = (reference.codes != 0) & np.asarray(land_mask, dtype=bool) & (model_codes != 0)
    model_groups = koppen_group(np.asarray(model_codes, dtype=np.int64))
    reference_groups = koppen_group(reference.codes.astype(np.int64))

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[...] = (16, 24, 40)                                   # ocean / unscored
    agree = scored & (model_groups == reference_groups)
    rgb[agree] = (40, 70, 40)                                 # correct group
    for index, colour in enumerate(
        [(60, 170, 60), (220, 60, 60), (230, 200, 60), (70, 120, 230), (200, 200, 210)],
        start=1,
    ):
        wrong = scored & (model_groups != reference_groups) & (model_groups == index)
        rgb[wrong] = colour                                   # coloured by what the MODEL said
    Image.fromarray(rgb).save(path)
    print()
    print(f"  error map written to {path}")
    print("    dark green = correct group; coloured = wrong, tinted by the model's")
    print("    claim (A green, B red, C yellow, D blue, E grey)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", type=str, default=None,
                        help="score an existing save instead of running the benchmark")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--min-land-fraction", type=float, default=0.0,
                        help="reference land fraction a cell needs to be scored (default 0.0)")
    parser.add_argument("--detail", action="store_true", help="also print the per-class table")
    parser.add_argument("--error-map", type=str, default=None, help="write a PNG error map")
    parser.add_argument("--json", type=str, default=None, help="write the full report as JSON")
    args = parser.parse_args()

    if args.state:
        from simulate import load_state

        state = load_state(args.state)
    else:
        from real_terrain_validation import (
            RealTerrainValidationConfig,
            run_real_terrain_validation,
        )

        config = RealTerrainValidationConfig(height=args.height, width=args.width)
        state, _ = run_real_terrain_validation(config)

    if state.koppen_type is None:
        print("state has no koppen_type field; nothing to score", file=sys.stderr)
        return 1

    _, land_mask = get_masks(state.elevation, use_cache=False)
    report = score_koppen_map(
        state.koppen_type,
        land_mask=land_mask,
        min_reference_land_fraction=args.min_land_fraction,
    )
    if "group" not in report:
        print(report.get("error", "scoring produced no result"), file=sys.stderr)
        return 1

    _print_headline(report)
    _print_confusion(report)
    _print_regions(report)
    _print_zones(report)
    if args.detail:
        _print_class_detail(report)

    height, width = state.koppen_type.shape
    print()
    print("  reference map's own Earth land shares (area-weighted, this resolution):")
    shares = earth_group_shares(height, width)
    print("    " + "   ".join(f"{name} {value:.1f}%" for name, value in shares.items()))

    if args.error_map:
        _write_error_map(Path(args.error_map), state.koppen_type, land_mask, height, width)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  full report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
