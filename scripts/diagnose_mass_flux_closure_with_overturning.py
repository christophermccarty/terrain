"""Does `close_upper_mass_flux`'s own residual closure survive overturning heating?

Section 13 showed, on a zero-heating configuration, that `residual_after_s`
closes cleanly and monotonically with `three_level_horizontal_mass_flux_max_speed_m_s`
(6% at 12 m/s, 86% at 1000 m/s), via a `simulate.close_upper_mass_flux`
recording wrapper -- no production code changed. Section 14 then found that
restoring overturning heating (both ocean/TOA radiative targets) decouples
the cap from cross-equatorial transport and composite reference error, but
did not re-check whether the closure's *own* local diagnostic still responds
to the cap once heating contributes to the pre-closure divergence field.

This script reuses that same non-invasive recording technique on one of
Section 14's heated configurations (TOA target, 10-day relaxation) at three
cap values (20/60/120 m/s) to answer that directly.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import simulate  # noqa: E402
from planet_params import EARTH  # noqa: E402
from pressure_circulation import close_upper_mass_flux as _real_close_upper_mass_flux  # noqa: E402
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
    native_balanced_mse_use_toa_radiative_target=True,
)

CAP_VALUES_M_S = (20.0, 60.0, 120.0)

_calls: list[dict] = []


def _recording_close_upper_mass_flux(*args, **kwargs):
    result = _real_close_upper_mass_flux(*args, **kwargs)
    _calls.append(
        {
            "residual_before_s": result.residual_before_s,
            "residual_after_s": result.residual_after_s,
            "equatorial_throughflow_after_m_s": result.equatorial_throughflow_after_m_s,
        }
    )
    return result


def _run_one(cap: float) -> dict:
    _calls.clear()
    simulate.close_upper_mass_flux = _recording_close_upper_mass_flux
    config = RealTerrainValidationConfig(
        height=32, width=64, spinup_years=1.0, evaluation_years=1.0,
    )
    pp = dataclasses.replace(
        EARTH, **SHARED, three_level_horizontal_mass_flux_max_speed_m_s=cap,
    )
    _, report = run_real_terrain_validation(config, planet_params=pp)
    metrics = report["metrics"]
    transport = ((metrics.get("circulation") or {}).get("meridional_transport") or {})

    before_rms = np.array([c["residual_before_s"] for c in _calls])
    after_rms = np.array([c["residual_after_s"] for c in _calls])
    after_thru = np.array([c["equatorial_throughflow_after_m_s"] for c in _calls])

    return {
        "cap_m_s": cap,
        "num_calls": len(_calls),
        "residual_before_s_mean": float(np.mean(before_rms)),
        "residual_after_s_mean": float(np.mean(after_rms)),
        "residual_closure_fraction": float(1.0 - np.mean(after_rms) / np.mean(before_rms)),
        "equatorial_throughflow_after_m_s_mean_abs": float(np.mean(np.abs(after_thru))),
        "cross_equatorial_total_energy_transport_pw": transport.get(
            "cross_equatorial_total_energy_transport_pw"
        ),
    }


def main() -> int:
    results = [_run_one(v) for v in CAP_VALUES_M_S]
    for row in results:
        print(json.dumps(row, indent=2))
    output_path = ROOT / "scripts" / "mass_flux_closure_with_overturning_diagnostic.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
