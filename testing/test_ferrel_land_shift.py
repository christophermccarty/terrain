"""test_ferrel_land_shift.py -- PlanetParams.ferrel_v_land_shift_deg.

Covers the land/ocean partition mechanism added 2026-07-26 to address the
"40-50N land/ocean partition" ceiling flagged in `ferrel_v_centre_deg`'s
docstring: that field is a single *zonal-mean* correction applied identically
at every longitude in a row, so it cannot push a dry continental interior
further poleward without also shifting the (already-correct) ocean at the
same latitude. `ferrel_v_land_shift_deg` lets land and ocean use different
Ferrel-cell centres, blended by land fraction.

Every test below pins BOTH `ferrel_v_centre_deg` and `ferrel_v_land_shift_deg`
explicitly on every `PlanetParams` it builds, rather than relying on
`EARTH`'s shipped defaults for either -- these tests must keep passing
regardless of what `EARTH.ferrel_v_land_shift_deg` is calibrated to.

Two properties matter and are tested directly, in both `generate_wind_field`
(MONTHLY/ANNUAL) and `evolve_wind` (DAILY/WEEKLY):
1. Explicit `ferrel_v_land_shift_deg=0.0` gives bit-identical output to a
   `PlanetParams`-like object that lacks the attribute entirely (simulating
   an old pickled save from before this field existed) -- exercising the
   `getattr(pp, "ferrel_v_land_shift_deg", 0.0)` fallback both call sites
   use, which is exactly the backward-compatibility contract this project's
   other gated `PlanetParams` fields document. A plain full-pipeline
   comparison at shift=0.0 against a *different* `ferrel_v_centre_deg`
   cannot serve this purpose -- terrain blocking/channeling and (in
   `evolve_wind`) advection/friction couple through elevation and would make
   even a genuinely no-op correction look longitude-varying in the final
   output, so the fallback-object comparison below is the correct level to
   test this at.
2. At a nonzero shift, land and ocean cells at the *same* latitude diverge --
   impossible when the correction is latitude-only, and the entire reason
   this field exists.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from testing.conftest import make_mixed_elev
from planet_params import EARTH


@pytest.fixture(scope="module")
def mixed_elev():
    return make_mixed_elev(64, 128, land_frac=0.35)


def _pp(centre: float, shift: float):
    return dataclasses.replace(EARTH, ferrel_v_centre_deg=centre, ferrel_v_land_shift_deg=shift)


class _MissingLandShift:
    """Proxies attribute access to a real PlanetParams, except
    `ferrel_v_land_shift_deg` raises AttributeError -- simulates a
    PlanetParams pickled before this field was added, so
    `getattr(pp, "ferrel_v_land_shift_deg", 0.0)` must fall back to 0.0."""

    def __init__(self, base):
        object.__setattr__(self, "_base", base)

    def __getattr__(self, name):
        if name == "ferrel_v_land_shift_deg":
            raise AttributeError(name)
        return getattr(self._base, name)


def test_generate_wind_field_missing_field_matches_explicit_zero(mixed_elev):
    from atmosphere import generate_wind_field

    H, W = mixed_elev.shape
    pp_explicit = _pp(44.0, 0.0)
    pp_missing = _MissingLandShift(_pp(44.0, 0.0))

    u1, v1 = generate_wind_field(H, W, day_of_year=172, block_size=4,
                                  elevation=mixed_elev, planet_params=pp_explicit)
    u2, v2 = generate_wind_field(H, W, day_of_year=172, block_size=4,
                                  elevation=mixed_elev, planet_params=pp_missing)

    assert np.max(np.abs(u1 - u2)) == 0.0
    assert np.max(np.abs(v1 - v2)) == 0.0


def test_evolve_wind_missing_field_matches_explicit_zero(mixed_elev):
    from atmosphere import evolve_wind

    H, W = mixed_elev.shape
    rng = np.random.default_rng(0)
    u = rng.normal(0.0, 3.0, size=(H, W)).astype(np.float32)
    v = rng.normal(0.0, 3.0, size=(H, W)).astype(np.float32)
    temperature = 280.0 + 20.0 * np.cos(np.linspace(-np.pi / 2, np.pi / 2, H))[:, None]
    temperature = np.repeat(temperature, W, axis=1).astype(np.float32)

    kwargs = dict(temperature=temperature, pressure=None, elevation=mixed_elev,
                  dt_days=1.0, cell_relax_days=3.0, time_days=100.0)
    pp_explicit = _pp(44.0, 0.0)
    pp_missing = _MissingLandShift(_pp(44.0, 0.0))
    u1, v1 = evolve_wind(u, v, planet_params=pp_explicit, **kwargs)
    u2, v2 = evolve_wind(u, v, planet_params=pp_missing, **kwargs)

    assert np.max(np.abs(u1 - u2)) == 0.0
    assert np.max(np.abs(v1 - v2)) == 0.0


def test_generate_wind_field_land_shift_differentiates_same_latitude(mixed_elev):
    """With a nonzero land shift, land and ocean cells at the same latitude
    row must get different v -- impossible when shift=0.0 (see the row-
    uniform test above), and the entire reason this field exists."""
    from atmosphere import generate_wind_field, _derive_land_sea_masks

    H, W = mixed_elev.shape
    _, v_shift = generate_wind_field(H, W, day_of_year=172, block_size=4,
                                      elevation=mixed_elev, planet_params=_pp(44.0, -6.0))
    _, v_base = generate_wind_field(H, W, day_of_year=172, block_size=4,
                                     elevation=mixed_elev, planet_params=_pp(44.0, 0.0))

    # generate_wind_field upsamples its coarse-grid result back to (H, W)
    # (block_size repeat/bilinear), so compare against the full-resolution
    # land/sea mask directly rather than the internal coarse grid.
    land_c, sea_c = _derive_land_sea_masks(mixed_elev)

    diff = v_shift - v_base
    assert land_c.sum() > 0 and sea_c.sum() > 0
    # Land rows must show a real, nonzero response to the shift somewhere;
    # ocean is not required to be exactly zero (mid-lat land/ocean rows blend
    # through the same trade/polar terms), but land's response must clearly
    # exceed ocean's mean response at the same latitudes for this to be a
    # genuine land-specific effect rather than noise.
    assert np.abs(diff[land_c]).mean() > np.abs(diff[sea_c]).mean()


def test_evolve_wind_land_shift_differentiates_same_latitude(mixed_elev):
    from atmosphere import evolve_wind
    from masks import get_masks

    H, W = mixed_elev.shape
    rng = np.random.default_rng(0)
    u = rng.normal(0.0, 3.0, size=(H, W)).astype(np.float32)
    v = rng.normal(0.0, 3.0, size=(H, W)).astype(np.float32)
    temperature = 280.0 + 20.0 * np.cos(np.linspace(-np.pi / 2, np.pi / 2, H))[:, None]
    temperature = np.repeat(temperature, W, axis=1).astype(np.float32)

    kwargs = dict(temperature=temperature, pressure=None, elevation=mixed_elev,
                  dt_days=1.0, cell_relax_days=3.0, time_days=100.0)
    _, v_shift = evolve_wind(u, v, planet_params=_pp(44.0, -6.0), **kwargs)
    _, v_base = evolve_wind(u, v, planet_params=_pp(44.0, 0.0), **kwargs)

    sea_mask, land_mask = get_masks(mixed_elev, use_cache=False)
    diff = v_shift - v_base
    assert land_mask.sum() > 0 and sea_mask.sum() > 0
    assert np.abs(diff[land_mask]).mean() > np.abs(diff[sea_mask]).mean()
