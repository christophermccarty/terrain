"""test_eddy_heat_flux.py — Feature 7: meridional eddy heat flux parameterisation.

Tests that the eddy flux reduces the meridional temperature gradient and that
the feedback flag disables it correctly.

Note: eddy Laplacian diffusion does not guarantee mid-latitude *warming* — it
equalises gradients, cooling the warm side and warming the cool side.  The
physically observable signal is a reduction in the equator-to-pole temperature
spread (std of the zonal-mean T profile).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_elev(H: int = 32, W: int = 64, land_frac: float = 0.35) -> np.ndarray:
    lon = np.linspace(0.0, 2.0 * np.pi, W, endpoint=False)
    lat = np.linspace(np.pi / 2.0, -np.pi / 2.0, H)
    lon_g, lat_g = np.meshgrid(lon, lat)
    signal = (0.5 * np.sin(2.0 * lon_g + 0.5) * np.cos(lat_g)
              + 0.3 * np.sin(5.0 * lon_g + 1.2) * np.cos(2.0 * lat_g - 0.3))
    thr = np.percentile(signal, (1.0 - land_frac) * 100.0)
    return np.where(signal > thr, (signal - thr) * 0.6, 0.0).astype(np.float32)


def _run_steps(n_days: int, eddy_coeff: float,
               feedback_flags: dict | None = None,
               H: int = 32, W: int = 64):
    """Run n_days of MONTHLY-equivalent steps and return final state."""
    from simulate import create_initial_state, simulate_step
    from planet_params import PlanetParams
    elev = _make_elev(H, W)
    pp = PlanetParams(eddy_heat_flux_coeff=eddy_coeff)
    state = create_initial_state(elev, day_of_year=80.0, planet_params=pp)
    kwargs: dict = dict(block_size=4, planet_params=pp)
    if feedback_flags:
        kwargs["feedback_flags"] = feedback_flags
    n_steps = max(1, n_days // 30)
    for _ in range(n_steps):
        state, _ = simulate_step(state, days=30.0, **kwargs)
    return state


def _midlat_mean(state) -> float:
    """Zonal mean T over 30–65°N."""
    T = state.temperature
    assert T is not None
    H = T.shape[0]
    lats = (0.5 - (np.arange(H) + 0.5) / H) * 180.0
    rows = (lats >= 30) & (lats <= 65)
    return float(np.mean(T[rows]))


def _zonal_mean_std(state) -> float:
    """Std-dev of the zonal-mean T profile — proxy for meridional gradient magnitude."""
    T = state.temperature
    assert T is not None
    return float(np.std(np.mean(T, axis=1)))


def test_eddy_flux_reduces_gradient():
    """Strong eddy flux should reduce the meridional temperature gradient (std of zonal T).

    Uses eddy_coeff=0.05 (8× default) to produce a detectable signal over 2 years.
    The correct physical observable is gradient *reduction*, not mid-lat warming,
    because Laplacian diffusion moves heat from high-T to low-T regions.

    ocean_transport/ice_albedo are disabled to isolate the eddy term: the
    differential being measured is small (~0.2 K std) and has twice now been
    flipped negative by *unrelated* changes re-phasing ocean-transport noise
    (first the eddy sub-stepping fix's interaction with ocean noise — see
    module history — then the 2026-07-03 western-boundary/thermal-diffusion
    fixes). With those feedbacks off, the probe measures the eddy diffusion
    itself: verified +0.19 K std reduction vs a spurious −0.16 with them on.
    """
    _iso = {"ocean_transport": False, "ice_albedo": False}
    state_no   = _run_steps(720, eddy_coeff=0.0,  H=32, W=64, feedback_flags=_iso)
    state_with = _run_steps(720, eddy_coeff=0.05, H=32, W=64, feedback_flags=_iso)

    std_no   = _zonal_mean_std(state_no)
    std_with = _zonal_mean_std(state_with)
    delta    = std_no - std_with  # positive = gradient reduction

    assert 0 < delta < 20.0, (
        f"Eddy flux gradient change = {delta:.2f} K std (expected 0–20 K); "
        f"no-eddy std={std_no:.2f} K, with-eddy std={std_with:.2f} K"
    )


def test_eddy_flag_disables_flux():
    """feedback_flags={'eddy_heat_flux': False} should behave like eddy_coeff=0."""
    T_flag_off  = _midlat_mean(_run_steps(365, eddy_coeff=0.006,
                                          feedback_flags={'eddy_heat_flux': False}))
    T_coeff_off = _midlat_mean(_run_steps(365, eddy_coeff=0.0))
    assert abs(T_flag_off - T_coeff_off) < 1.0, (
        f"Flag-off ({T_flag_off:.2f} K) diverges from coeff=0 ({T_coeff_off:.2f} K)"
    )


def test_eddy_flux_no_nan():
    """Eddy flux must not introduce NaN or Inf in the temperature field."""
    state = _run_steps(365, eddy_coeff=0.006)
    T = state.temperature
    assert T is not None
    assert np.all(np.isfinite(T)), "NaN or Inf in temperature with eddy flux enabled"
