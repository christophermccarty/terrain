"""Render the active real-terrain baseline as a compact Markdown status page."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_terrain_validation import DEFAULT_BASELINE_PATH, load_validation_report  # noqa: E402
from regional_validation import EARTH_PRECIP_REGIONS  # noqa: E402


def _format_precipitation_target(minimum: float, maximum: float) -> str:
    """Format a bounded annual precipitation target for a status table."""
    if minimum <= 0.0:
        return f"< {maximum:.0f}"
    return f"{minimum:.0f}–{maximum:.0f}"


def render_current_baseline(report: dict) -> str:
    """Return the intentionally small, current-facing subset of a full report."""
    config = report["config"]
    metrics = report["metrics"]
    global_metrics = metrics["global"]
    skill = metrics.get("koppen_map_skill") or {}
    thresholds = metrics.get("koppen_temperature_thresholds") or {}
    closure = (metrics.get("precip_rescale") or {}).get("closure") or {}
    regional_precip = metrics.get("regional_precip_mm_year") or {}
    regional_error = metrics.get("regional_target_error_fraction") or {}
    lines = [
        "# Current Earth baseline",
        "",
        "Generated from the tracked deterministic real-terrain validation report. "
        "Historical plans and audits provide investigation context; this page is the "
        "current regression contract.",
        "",
        "## Configuration",
        "",
        f"- Grid: {config['height']}×{config['width']}",
        f"- Time scale: {config['time_scale']}",
        f"- Spin-up/evaluation: {config['spinup_years']} / {config['evaluation_years']} years",
        "",
        "## Headline skill",
        "",
        f"- Köppen group accuracy: {float(skill.get('group_accuracy', float('nan'))):.3f}",
        f"- Köppen class accuracy: {float(skill.get('class_accuracy', float('nan'))):.3f}",
        f"- Coldest-month threshold accuracy: {float((thresholds.get('coldest_month') or {}).get('accuracy', float('nan'))):.3f}",
        f"- Warmest-month threshold accuracy: {float((thresholds.get('warmest_month') or {}).get('accuracy', float('nan'))):.3f}",
        "",
        "## Climate state",
        "",
        f"- Global temperature: {float(global_metrics['temperature_k']) - 273.15:.2f} °C",
        f"- Global precipitation: {float(global_metrics['precip_mm_day']):.3f} mm/day",
        f"- Cloud fraction: {float(global_metrics['cloud_fraction']):.3f}",
    ]
    if regional_precip or regional_error:
        lines.extend(
            [
                "",
                "## Regional precipitation targets",
                "",
                "Annual regional means from the validation fixture. These are broad "
                "climatological guardrails, not station-level targets.",
                "",
                "| Region | Model (mm/year) | Target (mm/year) | Status |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for region in EARTH_PRECIP_REGIONS:
            value = regional_precip.get(region.name)
            error = regional_error.get(region.name)
            if value is None or error is None:
                rendered_value = "—"
                status = "not scored"
            else:
                rendered_value = f"{float(value):.0f}"
                status = "within target" if abs(float(error)) <= 1e-12 else "outside target"
            target = _format_precipitation_target(
                region.precip_min_mm_year, region.precip_max_mm_year
            )
            lines.append(f"| {region.name} | {rendered_value} | {target} | {status} |")
    if closure:
        lines.extend(
            [
                "",
                "## Precipitation closure diagnostic",
                "",
                f"- Raw-to-final precipitation ratio: {float(closure.get('raw_to_final_ratio', float('nan'))):.3f}",
                f"- Target-side adjustment at the sampled state: {float(closure.get('target_adjustment_mm_day', float('nan'))):+.3f} mm/day",
            ]
        )
    lines.extend(
        [
            "",
            "Run `scripts/run_real_terrain_validation.py --compare` to reproduce this "
            "baseline, and regenerate this page after an intentional baseline update.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a compact current-baseline Markdown page.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "CURRENT_BASELINE.md")
    args = parser.parse_args()
    text = render_current_baseline(load_validation_report(args.baseline))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
