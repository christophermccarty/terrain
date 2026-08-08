"""Instrument `close_upper_mass_flux`'s own before/after diagnostics during a run.

Section 12 of docs/PRIOR_ART_IMPLEMENTATION_PLAN.md found that the horizontal
mass-flux closure alone (independent of any overturning heating) still leaves
about -20 PW of unphysical cross-equatorial transport, down from an
unconstrained +1184 PW. `close_upper_mass_flux` (pressure_circulation.py)
already computes `equatorial_throughflow_before_m_s`/`_after_m_s` and a
divergence-residual RMS before/after its own correction, but `simulate.py`
discards both -- it only consumes the upper-wind correction fields.

This script does not modify production code. It monkeypatches
`simulate.close_upper_mass_flux` with a recording wrapper around the same
function, runs the exact zero-heating configuration from Section 12 at
several values of `three_level_horizontal_mass_flux_max_speed_m_s` (the cap
applied to the eddy/Fourier-inverted upper-wind correction, separate from the
`throughflow_max_speed_m_s` cap on the zonal-mean null-mode term), and
reports both the divergence-residual closure and the resulting
cross-equatorial transport at each cap value. If raising the cap closes the
residual and shrinks the transport, the cap -- not the Fourier inversion
itself -- is the fixable culprit.
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
)

MAX_SPEED_VALUES_M_S = (12.0, 40.0, 120.0, 1000.0)

_calls: list[dict] = []


def _recording_close_upper_mass_flux(*args, **kwargs):
    result = _real_close_upper_mass_flux(*args, **kwargs)
    _calls.append(
        {
            "residual_before_s": result.residual_before_s,
            "residual_after_s": result.residual_after_s,
            "equatorial_throughflow_before_m_s": result.equatorial_throughflow_before_m_s,
            "equatorial_throughflow_after_m_s": result.equatorial_throughflow_after_m_s,
        }
    )
    return result


def _run_one(max_speed_m_s: float) -> dict:
    _calls.clear()
    simulate.close_upper_mass_flux = _recording_close_upper_mass_flux
    config = RealTerrainValidationConfig(
        height=32, width=64, spinup_years=1.0, evaluation_years=1.0,
    )
    pp = dataclasses.replace(
        EARTH, **SHARED, three_level_horizontal_mass_flux_max_speed_m_s=max_speed_m_s,
    )
    _, report = run_real_terrain_validation(config, planet_params=pp)
    metrics = report["metrics"]
    transport = ((metrics.get("circulation") or {}).get("meridional_transport") or {})

    if not _calls:
        raise RuntimeError("close_upper_mass_flux was never invoked -- gate wiring problem")

    before_rms = np.array([c["residual_before_s"] for c in _calls])
    after_rms = np.array([c["residual_after_s"] for c in _calls])
    after_thru = np.array([c["equatorial_throughflow_after_m_s"] for c in _calls])

    return {
        "max_speed_m_s": max_speed_m_s,
        "num_calls": len(_calls),
        "residual_before_s_mean": float(np.mean(before_rms)),
        "residual_after_s_mean": float(np.mean(after_rms)),
        "residual_closure_fraction": float(
            1.0 - np.mean(after_rms) / np.mean(before_rms)
        ),
        "equatorial_throughflow_after_m_s_mean_abs": float(np.mean(np.abs(after_thru))),
        "cross_equatorial_total_energy_transport_pw": transport.get(
            "cross_equatorial_total_energy_transport_pw"
        ),
        "global_precip_mm_day": metrics["global"]["precip_mm_day"],
        "koppen_group_accuracy": (metrics.get("koppen_map_skill") or {}).get("group_accuracy"),
        "koppen_class_accuracy": (metrics.get("koppen_map_skill") or {}).get("class_accuracy"),
    }


def main() -> int:
    results = [_run_one(v) for v in MAX_SPEED_VALUES_M_S]
    for row in results:
        print(json.dumps(row, indent=2))

    output_path = ROOT / "scripts" / "mass_flux_closure_throughflow_diagnostic.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
