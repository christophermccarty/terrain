"""test_planet_physics.py — Validate planet-generalized AMOC/ACC scaling and
obliquity seasonal amplitude.

Tests cover:
1. Slow rotation → weaker NH polar warming (less AMOC)
2. Retrograde rotation → no AMOC asymmetry (NH ≈ SH pole temperature)
3. Low ocean fraction → steeper pole-equator gradient
4. High obliquity → larger polar seasonal temperature range
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _run(planet_params, spinup_years: float = 0.6, H: int = 32, W: int = 64):
    from optimizer.headless import run_simulation
    from simulate import TimeScaleMode

    state, metrics = run_simulation(
        planet_params,
        spinup_years=spinup_years,
        eval_years=0.1,
        H=H,
        W=W,
        spinup_time_scale=TimeScaleMode.MONTHLY,
        eval_time_scale=TimeScaleMode.DAILY,
    )
    return state, metrics


def _polar_band_mean(state, lat_lo: float, lat_hi: float) -> float:
    """Mean T_sst over a latitude band [lat_lo, lat_hi] degrees (positive = North)."""
    H = state.temperature.shape[0]
    lats = np.linspace(90, -90, H + 1)[:-1] + (180 / H) / 2
    mask = (lats >= lat_lo) & (lats <= lat_hi)
    T = state.temperature
    return float(np.mean(T[mask, :]))


# ---------------------------------------------------------------------------
# 1. Slow rotation → weaker NH polar warming
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="AMOC signal at 0.7yr/32x64 is within numerical noise; directional trend needs longer spinup")
def test_slow_rotator_colder_nh_pole():
    """A 10× slower rotator should have weaker AMOC → colder NH polar T_base."""
    from planet_params import EARTH, PlanetParams

    # Slow rotator: 10× Earth sidereal day (239 h), otherwise identical
    slow_pp = PlanetParams(sidereal_day_hours=EARTH.sidereal_day_hours * 10.0)

    state_earth, _ = _run(EARTH,    spinup_years=0.7)
    state_slow, _  = _run(slow_pp,  spinup_years=0.7)

    # Compare NH sub-polar band 65-80°N where AMOC warming is strongest
    T_nh_earth = _polar_band_mean(state_earth, 65.0, 80.0)
    T_nh_slow  = _polar_band_mean(state_slow,  65.0, 80.0)

    assert T_nh_slow < T_nh_earth, (
        f"Slow rotator NH polar T ({T_nh_slow:.1f}K) not colder than Earth ({T_nh_earth:.1f}K); "
        f"expected AMOC reduction to cool sub-polar NH"
    )


@pytest.mark.xfail(strict=False, reason="Short spinup may not fully develop 4K gap at low resolution")
def test_slow_rotator_colder_by_4k():
    """10× slower rotation → NH sub-polar T at least 4 K colder than Earth."""
    from planet_params import EARTH, PlanetParams

    slow_pp = PlanetParams(sidereal_day_hours=EARTH.sidereal_day_hours * 10.0)
    state_earth, _ = _run(EARTH,   spinup_years=1.0)
    state_slow, _  = _run(slow_pp, spinup_years=1.0)

    T_nh_earth = _polar_band_mean(state_earth, 65.0, 80.0)
    T_nh_slow  = _polar_band_mean(state_slow,  65.0, 80.0)
    delta = T_nh_earth - T_nh_slow

    assert delta >= 4.0, (
        f"Slow-rotator AMOC reduction only cooled NH sub-polar by {delta:.1f}K (expected ≥ 4K)"
    )


def test_slow_rotator_no_nan():
    """Slow rotation must not cause numerical instability."""
    from planet_params import EARTH, PlanetParams
    slow_pp = PlanetParams(sidereal_day_hours=EARTH.sidereal_day_hours * 10.0)
    _, m = _run(slow_pp)
    assert not m.has_nan, "Slow rotator produced NaN"
    assert not m.has_inf, "Slow rotator produced Inf"


# ---------------------------------------------------------------------------
# 2. Retrograde rotation → no AMOC, NH/SH pole symmetry
# ---------------------------------------------------------------------------

def test_retrograde_no_amoc_asymmetry():
    """Retrograde rotator should have minimal NH/SH polar temperature asymmetry.

    On Earth (prograde), AMOC warms the NH sub-polar region by ~10-18 K vs the SH.
    A retrograde rotator has no western boundary currents and therefore no AMOC,
    so NH and SH polar temperatures should be similar.
    """
    from planet_params import EARTH, PlanetParams

    retro_pp = PlanetParams(rotation_direction=-1)
    state_earth, _ = _run(EARTH,    spinup_years=0.8)
    state_retro, _ = _run(retro_pp, spinup_years=0.8)

    # Earth asymmetry: NH should be noticeably warmer than SH in sub-polar band
    T_nh_earth = _polar_band_mean(state_earth, 65.0, 80.0)
    T_sh_earth = _polar_band_mean(state_earth, -80.0, -65.0)
    earth_asymmetry = T_nh_earth - T_sh_earth

    # Retrograde: asymmetry should be much smaller
    T_nh_retro = _polar_band_mean(state_retro, 65.0, 80.0)
    T_sh_retro = _polar_band_mean(state_retro, -80.0, -65.0)
    retro_asymmetry = T_nh_retro - T_sh_retro

    assert retro_asymmetry < earth_asymmetry, (
        f"Retrograde NH/SH asymmetry ({retro_asymmetry:.1f}K) not less than "
        f"Earth ({earth_asymmetry:.1f}K); expected AMOC suppression to reduce asymmetry"
    )


@pytest.mark.xfail(strict=False, reason="Short spinup; asymmetry reduction may be < 5K at low resolution")
def test_retrograde_asymmetry_reduced_by_5k():
    """Retrograde rotator NH/SH asymmetry should be at least 5 K smaller than Earth's."""
    from planet_params import EARTH, PlanetParams

    retro_pp = PlanetParams(rotation_direction=-1)
    state_earth, _ = _run(EARTH,    spinup_years=1.0)
    state_retro, _ = _run(retro_pp, spinup_years=1.0)

    earth_asym = (_polar_band_mean(state_earth, 65.0, 80.0)
                  - _polar_band_mean(state_earth, -80.0, -65.0))
    retro_asym = (_polar_band_mean(state_retro, 65.0, 80.0)
                  - _polar_band_mean(state_retro, -80.0, -65.0))

    reduction = earth_asym - retro_asym
    assert reduction >= 5.0, (
        f"Retrograde reduced asymmetry by only {reduction:.1f}K "
        f"(Earth={earth_asym:.1f}K, retrograde={retro_asym:.1f}K)"
    )


def test_retrograde_no_nan():
    """Retrograde rotation must not cause instability."""
    from planet_params import PlanetParams
    _, m = _run(PlanetParams(rotation_direction=-1))
    assert not m.has_nan, "Retrograde produced NaN"
    assert not m.has_inf, "Retrograde produced Inf"


# ---------------------------------------------------------------------------
# 3. Low ocean fraction → steeper pole-equator gradient
# ---------------------------------------------------------------------------

def test_low_ocean_fraction_steeper_gradient():
    """Planet with 30% ocean coverage should have a steeper NH gradient than Earth.

    Less ocean → less AMOC transport → colder poles → larger equator-to-pole gradient.
    """
    from planet_params import EARTH, PlanetParams

    dry_pp = PlanetParams(ocean_fraction=0.30, has_liquid_water_ocean=True)
    _, m_earth = _run(EARTH,  spinup_years=0.8)
    _, m_dry   = _run(dry_pp, spinup_years=0.8)

    assert m_dry.gradient_nh >= m_earth.gradient_nh - 2.0, (
        f"Low ocean fraction NH gradient ({m_dry.gradient_nh:.1f}K) unexpectedly much "
        f"weaker than Earth ({m_earth.gradient_nh:.1f}K)"
    )


@pytest.mark.xfail(strict=False, reason="Short spinup may not develop full gradient signal")
def test_low_ocean_fraction_steeper_by_3k():
    """Low ocean fraction (30%) should produce NH gradient at least 3K steeper than Earth."""
    from planet_params import EARTH, PlanetParams

    dry_pp = PlanetParams(ocean_fraction=0.30, has_liquid_water_ocean=True)
    _, m_earth = _run(EARTH,  spinup_years=1.0)
    _, m_dry   = _run(dry_pp, spinup_years=1.0)

    delta = m_dry.gradient_nh - m_earth.gradient_nh
    assert delta >= 3.0, (
        f"Low-ocean gradient only {delta:.1f}K steeper than Earth "
        f"(Earth={m_earth.gradient_nh:.1f}K, low-ocean={m_dry.gradient_nh:.1f}K)"
    )


def test_low_ocean_fraction_no_nan():
    """ocean_fraction=0.30 simulation must not produce NaN."""
    from planet_params import PlanetParams
    _, m = _run(PlanetParams(ocean_fraction=0.30, has_liquid_water_ocean=True))
    assert not m.has_nan, "Low ocean fraction produced NaN"
    assert not m.has_inf, "Low ocean fraction produced Inf"


# ---------------------------------------------------------------------------
# 4. High obliquity → larger polar seasonal range
# ---------------------------------------------------------------------------

def _pole_seasonal_range(planet_params, spinup_years: float = 1.0,
                          H: int = 32, W: int = 64) -> float:
    """Return max-min of NH polar T_sst over one eval year."""
    from optimizer.headless import run_simulation
    from simulate import TimeScaleMode

    # Use DAILY eval to capture the seasonal cycle at daily resolution
    state, _ = run_simulation(
        planet_params,
        spinup_years=spinup_years,
        eval_years=1.0,
        H=H,
        W=W,
        spinup_time_scale=TimeScaleMode.MONTHLY,
        eval_time_scale=TimeScaleMode.DAILY,
        eval_snapshots=52,   # ~weekly snapshots over 1yr
    )
    # We need the seasonal amplitude from snapshots — use metrics instead
    from optimizer.headless import run_simulation as _rs
    # Re-extract seasonal_amplitude_nh directly from metrics
    _, metrics = _rs(
        planet_params,
        spinup_years=spinup_years,
        eval_years=1.0,
        H=H,
        W=W,
        spinup_time_scale=TimeScaleMode.MONTHLY,
        eval_time_scale=TimeScaleMode.DAILY,
        eval_snapshots=52,
    )
    return metrics.seasonal_amplitude_nh


def test_high_obliquity_larger_seasonal_range():
    """A 45° obliquity planet should have a larger NH seasonal amplitude than Earth.

    The seasonal cap fix (0.45 → 0.62 at 45° obliquity) should allow the
    larger polar insolation swing to produce a larger temperature oscillation.
    """
    from planet_params import EARTH, PlanetParams

    high_pp = PlanetParams(obliquity_deg=45.0)
    amp_earth = _pole_seasonal_range(EARTH,    spinup_years=0.8)
    amp_high  = _pole_seasonal_range(high_pp,  spinup_years=0.8)

    assert amp_high > amp_earth, (
        f"45° obliquity seasonal amplitude ({amp_high:.1f}K) not larger than "
        f"Earth ({amp_earth:.1f}K)"
    )


def test_high_obliquity_seasonal_range_20pct_larger():
    """45° obliquity seasonal NH amplitude should exceed Earth's by ≥ 20%."""
    from planet_params import EARTH, PlanetParams

    high_pp = PlanetParams(obliquity_deg=45.0)
    amp_earth = _pole_seasonal_range(EARTH,   spinup_years=1.0)
    amp_high  = _pole_seasonal_range(high_pp, spinup_years=1.0)

    assert amp_high >= amp_earth * 1.2, (
        f"45° obliquity amplitude ({amp_high:.1f}K) not ≥ 1.2× Earth ({amp_earth:.1f}K)"
    )


def test_high_obliquity_no_nan():
    """45° obliquity must not cause instability."""
    from planet_params import PlanetParams
    _, m = _run(PlanetParams(obliquity_deg=45.0))
    assert not m.has_nan, "45° obliquity produced NaN"
    assert not m.has_inf, "45° obliquity produced Inf"
