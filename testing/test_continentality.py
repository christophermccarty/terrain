"""test_continentality.py -- masks.get_continentality() correctness.

get_continentality builds a [0,1] "distance from coast" proxy via screened
diffusion (see its docstring for why plain diffusion doesn't work: no decay
term means the field has no non-trivial steady state and just slowly
saturates the whole domain to the ocean value regardless of distance --
found and fixed while calibrating this function against real terrain).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from masks import get_continentality, get_masks, invalidate


def _synthetic_elevation(H: int = 64, W: int = 128) -> np.ndarray:
    """All-ocean border with a wide land block in the middle (a "continent"
    with a real interior, not just a thin coastal strip)."""
    elev = np.zeros((H, W), dtype=np.float32)
    elev[16:48, 32:96] = 0.5
    return elev


def test_zero_over_ocean():
    elev = _synthetic_elevation()
    c = get_continentality(elev, use_cache=False)
    sea_mask, _ = get_masks(elev, use_cache=False)
    assert np.all(c[sea_mask] == 0.0)


def test_increases_with_distance_from_coast():
    elev = _synthetic_elevation()
    c = get_continentality(elev, use_cache=False)
    # Row 32 crosses the land block from column 32 (coast) to column 64 (center).
    coast = c[32, 33]
    mid = c[32, 48]
    center = c[32, 63]
    assert 0.0 <= coast < mid < center, (coast, mid, center)


def test_bounded_zero_to_one():
    elev = _synthetic_elevation()
    c = get_continentality(elev, use_cache=False)
    assert c.min() >= 0.0
    assert c.max() <= 1.0


def test_small_landmass_stays_low():
    """A landmass narrow relative to the diffusion length scale shouldn't
    develop a strong "interior" -- confirms the screened-diffusion decay
    term is actually doing something, not just returning ~1 everywhere land."""
    H, W = 64, 128
    elev = np.zeros((H, W), dtype=np.float32)
    elev[30:34, 60:68] = 0.5  # a small island, ~8x4 cells
    c = get_continentality(elev, use_cache=False)
    land_mask = get_masks(elev, use_cache=False)[1]
    assert c[land_mask].max() < 0.3


def test_caching_returns_same_object_for_same_array():
    elev = _synthetic_elevation()
    invalidate(elev)
    c1 = get_continentality(elev, use_cache=True)
    c2 = get_continentality(elev, use_cache=True)
    assert c1 is c2
    invalidate(elev)


def test_cache_invalidated_on_content_change():
    # Same id() (in-place mutation), different content -- the fingerprint
    # guard must detect this rather than silently returning a stale result
    # (same convention as get_masks's own cache).
    elev = _synthetic_elevation()
    invalidate(elev)
    c1 = get_continentality(elev, use_cache=True)
    elev[:, :] = 0.0  # now all-ocean, in place
    c2 = get_continentality(elev, use_cache=True)
    assert not np.array_equal(c1, c2)
    assert np.all(c2 == 0.0)
    invalidate(elev)


def test_land_fraction_zero_gives_all_zero_field():
    elev = np.zeros((32, 64), dtype=np.float32)  # all ocean
    c = get_continentality(elev, use_cache=False)
    assert np.all(c == 0.0)
