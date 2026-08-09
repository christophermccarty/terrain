"""ERA5/CRU-inspired climatology validation gates.

Two tiers:

1. Zonal-band gates (below) run on cheap synthetic terrain and anchor gross
   latitude-band behaviour against a handful of published zonal means. They
   predate the real reference below and stay useful as a fast general sanity
   check independent of the bundled Earth DEM's exact coastline geometry.
2. Real CRU TS v4.10 map-correlation gates (bottom of file) score the model
   against `testing/reference_data/cru_ts_v4.10_1991_2020.npz` -- a real
   0.5deg monthly land climatology, area-conservatively regridded via
   `monthly_climatology.score_monthly_climatology` -- on the actual bundled
   Earth DEM. This was previously only exercised by ad-hoc scripts
   (`scripts/run_real_terrain_validation.py --monthly-climatology ...`); this
   file is what turns that into an enforced regression gate. See
   docs/MONTHLY_CLIMATOLOGY_REFERENCE.md for the reference's provenance and
   licence, and for why the .npz itself is a local, reproducible, gitignored
   artifact rather than a committed fixture -- the tests below skip (not
   fail) when it is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_terrain_validation import EARTH_ZONAL_REFERENCE

# Shared with the real-terrain platform so manual, synthetic, and real-DEM
# validation cannot silently use different climatology anchors.
ERA5_CRU_REFERENCE = {
    name: {"t_c": values["t_c"], "p_mm_yr": values["p_mm_yr"]}
    for name, values in EARTH_ZONAL_REFERENCE.items()
}

CRU_REFERENCE_PATH = ROOT / "testing" / "reference_data" / "cru_ts_v4.10_1991_2020.npz"


def _zonal_band_means(state, *, lat_edges):
    from diagnostics import area_weighted_global_mean

    T = state.air_temperature if state.air_temperature is not None else state.temperature
    P = state.precipitation
    H = T.shape[0]
    lat_deg = (0.5 - (np.arange(H, dtype=np.float64) + 0.5) / H) * 180.0
    out = {}
    for name, (lo, hi) in lat_edges.items():
        mask_rows = (lat_deg >= lo) & (lat_deg < hi)
        if not np.any(mask_rows):
            continue
        band_T = T[mask_rows]
        band_P = P[mask_rows] if P is not None else None
        out[name] = {
            "t_c": float(np.mean(band_T)) - 273.15,
            "p_mm_yr": float(np.mean(band_P)) * 365.25 if band_P is not None else float("nan"),
        }
    return out


@pytest.fixture(scope="module")
def spun_state():
    from simulate import simulate_step, create_initial_state
    from testing.conftest import make_mixed_elev

    state = create_initial_state(make_mixed_elev(32, 64), day_of_year=80.0)
    for _ in range(365):
        state, _ = simulate_step(state, days=1.0, block_size=4, enable_carbon_cycle=False)
    return state


@pytest.mark.slow
def test_zonal_temperature_bias_within_20k_of_reanalysis(spun_state):
    lat_edges = {
        "0-10N": (0.0, 10.0),
        "10-20N": (10.0, 20.0),
        "40-50N": (40.0, 50.0),
        "0-10S": (-10.0, 0.0),
    }
    bands = _zonal_band_means(spun_state, lat_edges=lat_edges)
    for name, ref in ERA5_CRU_REFERENCE.items():
        if name not in bands:
            continue
        bias = bands[name]["t_c"] - ref["t_c"]
        assert abs(bias) < 20.0, f"{name} T bias {bias:+.1f}°C vs ERA5/CRU {ref['t_c']}°C"


@pytest.mark.slow
def test_zonal_precipitation_within_factor_of_three(spun_state):
    """Tropical bands after 1-year spinup; mid-latitude precip needs longer runs."""
    lat_edges = {
        "0-10N": (0.0, 10.0),
        "0-10S": (-10.0, 0.0),
    }
    bands = _zonal_band_means(spun_state, lat_edges=lat_edges)
    for name, ref in ERA5_CRU_REFERENCE.items():
        if name not in bands or not np.isfinite(bands[name]["p_mm_yr"]):
            continue
        ratio = bands[name]["p_mm_yr"] / max(ref["p_mm_yr"], 1.0)
        assert 0.25 < ratio < 4.0, (
            f"{name} precip {bands[name]['p_mm_yr']:.0f} mm/yr vs "
            f"ERA5/CRU {ref['p_mm_yr']} mm/yr (ratio={ratio:.2f})"
        )


@pytest.mark.slow
def test_midlatitude_precip_reanalysis_anchor(spun_state):
    lat_edges = {"40-50N": (40.0, 50.0), "40-50S": (-50.0, -40.0)}
    bands = _zonal_band_means(spun_state, lat_edges=lat_edges)
    for name, ref in ERA5_CRU_REFERENCE.items():
        if name not in bands:
            continue
        ratio = bands[name]["p_mm_yr"] / max(ref["p_mm_yr"], 1.0)
        assert 0.25 < ratio < 4.0


# ---------------------------------------------------------------------------
# Real CRU TS v4.10 map-correlation gates (real bundled Earth DEM)
# ---------------------------------------------------------------------------
#
# These score the model against an actual 0.5deg gridded monthly land
# climatology -- not a handful of hand-typed zonal boxes -- via the same
# `monthly_climatology.score_monthly_climatology` path used throughout
# docs/PRIOR_ART_IMPLEMENTATION_PLAN.md and docs/MONTHLY_CLIMATOLOGY_REFERENCE.md.
# Bounds below carry headroom over the measured current baseline (recorded in
# docs/MONTHLY_CLIMATOLOGY_REFERENCE.md and re-verified fresh when this gate
# was added: temperature +2.98C bias / 6.28C RMSE / 0.930 correlation;
# precipitation -0.109 mm/day bias / 1.406 log-RMSE / 0.463 log-correlation;
# 25.8% scored land-area fraction), the same convention as this project's
# other regression gates (e.g. test_co2_budget_near_steady_state) -- these
# assert the model has not regressed against a real, versioned reference,
# not that it has reached any particular accuracy target.

_cru_reference_missing = not CRU_REFERENCE_PATH.exists()
_cru_skip_reason = (
    f"CRU TS reference not built locally: {CRU_REFERENCE_PATH} is missing. "
    "Run `python scripts/build_cru_ts_reference.py` to fetch and build it "
    "(see docs/MONTHLY_CLIMATOLOGY_REFERENCE.md); it is a reproducible, "
    "gitignored local artifact, not a committed fixture."
)


@pytest.fixture(scope="module")
def cru_scored_report():
    """Run the real-DEM validation harness scored against CRU TS v4.10.

    64x128, one-year spin-up / one-year evaluation -- the same compact
    configuration docs/MONTHLY_CLIMATOLOGY_REFERENCE.md uses for its tracked
    baseline, on the real bundled Earth DEM (not synthetic terrain, since
    map-correlation against a real gridded reference needs real geography to
    be meaningful).
    """
    from real_terrain_validation import RealTerrainValidationConfig, run_real_terrain_validation

    config = RealTerrainValidationConfig(
        height=64, width=128, spinup_years=1.0, evaluation_years=1.0,
    )
    _, report = run_real_terrain_validation(
        config, monthly_climatology_path=CRU_REFERENCE_PATH,
    )
    return report["metrics"]["monthly_climatology"]


@pytest.mark.slow
@pytest.mark.skipif(_cru_reference_missing, reason=_cru_skip_reason)
def test_cru_temperature_map_correlation_and_error(cru_scored_report):
    temperature = cru_scored_report["temperature_c"]
    assert abs(temperature["monthly_bias"]) < 4.5, (
        f"CRU land monthly temperature bias {temperature['monthly_bias']:+.2f}C "
        "exceeds regression bound (measured baseline +2.98C)"
    )
    assert temperature["monthly_rmse"] < 7.5, (
        f"CRU land monthly temperature RMSE {temperature['monthly_rmse']:.2f}C "
        "exceeds regression bound (measured baseline 6.28C)"
    )
    assert temperature["monthly_correlation"] > 0.88, (
        f"CRU land monthly temperature correlation {temperature['monthly_correlation']:.3f} "
        "below regression bound (measured baseline 0.930)"
    )


@pytest.mark.slow
@pytest.mark.skipif(_cru_reference_missing, reason=_cru_skip_reason)
def test_cru_precipitation_map_correlation_and_error(cru_scored_report):
    precipitation = cru_scored_report["precipitation_mm_day"]
    assert abs(precipitation["monthly_bias"]) < 0.5, (
        f"CRU land monthly precipitation bias {precipitation['monthly_bias']:+.3f} mm/day "
        "exceeds regression bound (measured baseline -0.109 mm/day)"
    )
    assert precipitation["monthly_log_rmse"] < 1.6, (
        f"CRU land monthly log-precipitation RMSE {precipitation['monthly_log_rmse']:.3f} "
        "exceeds regression bound (measured baseline 1.406)"
    )
    assert precipitation["monthly_log_correlation"] > 0.35, (
        f"CRU land monthly log-precipitation correlation "
        f"{precipitation['monthly_log_correlation']:.3f} "
        "below regression bound (measured baseline 0.463)"
    )


@pytest.mark.slow
@pytest.mark.skipif(_cru_reference_missing, reason=_cru_skip_reason)
def test_cru_scored_area_fraction_is_meaningful(cru_scored_report):
    """Guards the regrid/mask pipeline itself, independent of physics.

    A near-zero scored fraction would mean the area-weighting or
    minimum-land-fraction mask silently stopped scoring anything -- every
    other CRU gate would then pass vacuously. Bound is well under the
    measured 25.8% so it only trips on an actual pipeline break, not normal
    physics drift.
    """
    assert cru_scored_report["scored_area_fraction"] > 0.15, (
        f"CRU scored land-area fraction {cru_scored_report['scored_area_fraction']:.3f} "
        "is implausibly low (measured baseline 0.258) -- check the regrid/mask pipeline"
    )


# ---------------------------------------------------------------------------
# NCEP/NCAR Reanalysis 1 wind map-correlation gate (real bundled Earth DEM)
# ---------------------------------------------------------------------------
#
# CRU TS publishes no wind variable at all (verified by listing its actual
# server directory: only cld/dtr/frs/pet/pre/tmn/tmp/tmx/vap/wet exist), so
# wind uses a separate provider -- NCEP/NCAR Reanalysis 1, a global
# land+ocean model reanalysis, anonymously downloadable, same 1991-2020
# period. Unlike temperature/precipitation, wind is scored as a single
# evaluation-period time-mean against the reference's own annual mean, not
# true month-by-month (see monthly_climatology.score_monthly_climatology's
# docstring) -- a documented current limitation, not an oversight.

NCEP_WIND_REFERENCE_PATH = ROOT / "testing" / "reference_data" / "ncep_ncar_wind_1991_2020.npz"
_ncep_wind_reference_missing = not NCEP_WIND_REFERENCE_PATH.exists()
_ncep_wind_skip_reason = (
    f"NCEP/NCAR wind reference not built locally: {NCEP_WIND_REFERENCE_PATH} is missing. "
    "Run `python scripts/build_ncep_wind_reference.py` to fetch and build it "
    "(see docs/MONTHLY_CLIMATOLOGY_REFERENCE.md); it is a reproducible, gitignored "
    "local artifact, not a committed fixture."
)


@pytest.fixture(scope="module")
def ncep_wind_scored_report():
    """Run the real-DEM validation harness scored against NCEP/NCAR Reanalysis 1 wind.

    Same 64x128, one-year spin-up / one-year evaluation configuration as
    cru_scored_report, on the real bundled Earth DEM, scored against wind
    only (no --monthly-climatology) so this gate is independent of whether
    the CRU reference happens to be present too.
    """
    from real_terrain_validation import RealTerrainValidationConfig, run_real_terrain_validation

    config = RealTerrainValidationConfig(
        height=64, width=128, spinup_years=1.0, evaluation_years=1.0,
    )
    _, report = run_real_terrain_validation(
        config, wind_climatology_path=NCEP_WIND_REFERENCE_PATH,
    )
    return report["metrics"]["monthly_climatology"]


@pytest.mark.slow
@pytest.mark.skipif(_ncep_wind_reference_missing, reason=_ncep_wind_skip_reason)
def test_ncep_wind_speed_map_correlation_and_error(ncep_wind_scored_report):
    """Regression gate, not an accuracy target.

    The model's single-layer, largely-diagnostic wind field is far cruder
    than a full reanalysis, so bounds here are deliberately loose -- this
    exists to catch a real pipeline break (e.g. a broken regrid or a sign
    error), not to enforce realism the model was never designed to reach.
    """
    wind = ncep_wind_scored_report["wind_speed_ms"]
    assert wind["source"].startswith("NCEP/NCAR"), f"unexpected wind source: {wind['source']}"
    assert abs(wind["annual_bias"]) < 6.0, (
        f"NCEP wind speed bias {wind['annual_bias']:+.2f} m/s exceeds regression bound "
        "(measured baseline -1.10 m/s)"
    )
    assert wind["annual_rmse"] < 6.0, (
        f"NCEP wind speed RMSE {wind['annual_rmse']:.2f} m/s exceeds regression bound "
        "(measured baseline 2.73 m/s)"
    )
    assert wind["annual_correlation"] > -0.2, (
        f"NCEP wind speed correlation {wind['annual_correlation']:.3f} exceeds regression "
        "bound (measured baseline 0.151) -- a strongly negative correlation would suggest "
        "a real pipeline defect (e.g. a longitude misalignment), not just weak realism"
    )
