"""Section 18 follow-up: is cross-equatorial transport noise, or a real parameter effect?

Section 18 (docs/PRIOR_ART_IMPLEMENTATION_PLAN.md) found that
`three_level_upper_wind_pgf_fraction`/`three_level_upper_wind_damping`
produce 5-10 PW of row-to-row transport swing in a finer joint sweep,
comparable to the spread between rows -- looking like noise rather than a
real parameter effect. But the model has no free global RNG seed to
directly test a noise floor against: `_storm_pressure_anomaly` and friends
are deterministic, stateless functions of `time_days` and hashed
(hemisphere, slot, generation) identity (see simulate.py's own comments
there), not driven by any reseedable global RNG.

This script instead perturbs the evaluation window's *phase* relative to
those deterministic-but-oscillating storm/Rossby/seasonal systems, without
touching any physics parameter at all: it holds
`three_level_upper_wind_pgf_fraction=0.1`/`three_level_upper_wind_damping=
0.08` (Section 17's candidate point) fixed and sweeps only
`RealTerrainValidationConfig.spinup_years`, which shifts which day the
one-year evaluation window starts on without changing what is being
evaluated. If transport swings by a similar magnitude to Section 18's
parameter sweep purely from this phase shift, that is direct evidence the
metric is phase/chaos-sensitive at this compact protocol regardless of what
is varied -- meaning it cannot be used to pick between nearby parameter
values at this resolution/duration at all, independent of any question
about the upper-wind parameters specifically.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planet_params import EARTH  # noqa: E402
from real_terrain_validation import (  # noqa: E402
    RealTerrainValidationConfig,
    run_real_terrain_validation,
)

SHARED = dict(
    enable_prognostic_column_water=True,
    enable_prognostic_condensate=True,
    column_water_use_bulk_condensate_rainfall=True,
    enable_stability_aware_condensation=True,
    enable_two_layer_convective_adjustment=True,
    enable_three_level_pressure_column=True,
    enable_three_level_horizontal_mass_flux_closure=True,
    enable_energy_limited_evaporation=True,
    evaporation_downwelling_longwave_w_m2=25.0,
    enable_native_balanced_moist_static_energy_overturning=True,
    native_balanced_mse_radiative_relaxation_days=10.0,
    three_level_upper_wind_pgf_fraction=0.1,
    three_level_upper_wind_damping=0.08,
)

TARGETS = {"ocean": False, "toa": True}
# Phase shifts only -- evaluation_years stays 1.0 throughout, only which day
# the evaluation window starts on changes.
SPINUP_YEARS_VALUES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)


def _extract(metrics: dict) -> dict:
    circulation = metrics.get("circulation") or {}
    transport = circulation.get("meridional_transport") or {}
    koppen = metrics.get("koppen_map_skill") or {}
    return {
        "koppen_group_accuracy": koppen.get("group_accuracy"),
        "cross_equatorial_total_energy_transport_pw": transport.get(
            "cross_equatorial_total_energy_transport_pw"
        ),
    }


def main() -> int:
    results: dict[str, dict] = {}
    for target_name, use_toa in TARGETS.items():
        pp = dataclasses.replace(
            EARTH, **SHARED, native_balanced_mse_use_toa_radiative_target=use_toa,
        )
        for spinup_years in SPINUP_YEARS_VALUES:
            key = f"{target_name}_spinup{spinup_years}"
            config = RealTerrainValidationConfig(
                height=32, width=64, spinup_years=spinup_years, evaluation_years=1.0,
            )
            _, report = run_real_terrain_validation(config, planet_params=pp)
            results[key] = _extract(report["metrics"])
            print(f"--- {key} ---")
            print(json.dumps(results[key], indent=2, default=str))

    output_path = ROOT / "scripts" / "upper_wind_transport_noise_floor_result.json"
    output_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
