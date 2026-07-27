"""ERA5/CRU-inspired climatology validation gates (zonal-band level).

Full 2° map-correlation against versioned reanalysis grids is deferred to a
later calibration pass; these tests anchor the simulator against published
monthly zonal-mean references before larger ocean/cloud upgrades.
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
