"""masks.py — Canonical land/sea mask computation for PlanetSim.

All modules should call ``get_masks(elevation)`` instead of computing their
own inline splits.  The result is optionally cached by array ``id``.

Mask convention
---------------
- ``sea_mask``  : True where ocean / sea-ice can exist.
- ``land_mask`` : True where land processes apply (``~sea_mask``).

Sea-level detection heuristic
------------------------------
- Loaded DEMs encode ocean as exactly ``0.0``.  If >5% of cells are exactly
  zero we treat those as ocean (``elev == 0.0``).
- Procedural terrain uses a smooth signal with no exact zeros, so we fall back
  to a median split (``elev <= median``), matching historical simulate.py
  behaviour.
"""
from __future__ import annotations

import numpy as np

# Module-level cache: maps id(elevation_array) -> (sea_mask, land_mask)
_MASK_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}
# Lightweight content fingerprint to guard against Python id() reuse after GC.
# Stores (first_elem, last_elem) of the elevation array at cache-write time.
_MASK_CACHE_FP: dict[int, tuple[float, float]] = {}


def get_masks(
    elevation: np.ndarray,
    *,
    assume_loaded_if_zeros_frac: float = 0.05,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sea_mask, land_mask)`` for *elevation*.

    Results are optionally cached by ``id(elevation)``.  Only use the cache
    for *persistent* elevation arrays (e.g. ``state.elevation``) whose Python
    object identity is stable across calls.  Pass ``use_cache=False`` for
    transient arrays (coarse downsamples, test fixtures) to avoid stale hits
    caused by Python id-reuse after garbage collection.

    Parameters
    ----------
    elevation:
        2-D float32 normalised elevation array (same shape as the simulation
        grid).
    assume_loaded_if_zeros_frac:
        Fraction of exact-zero cells above which the array is treated as a
        loaded DEM.
    use_cache:
        Whether to read/write the module-level id-based cache.

    Returns
    -------
    sea_mask, land_mask : np.ndarray of dtype bool, same shape as *elevation*.
    """
    if use_cache:
        key = id(elevation)
        if key in _MASK_CACHE:
            cached = _MASK_CACHE[key]
            elev_r = np.asarray(elevation, dtype=np.float32).ravel()
            n = elev_r.size
            fp = (float(elev_r[0]), float(elev_r[-1]), float(elev_r.sum())) if n >= 2 else (0.0, 0.0, 0.0)
            if cached[0].shape == elevation.shape and _MASK_CACHE_FP.get(key) == fp:
                return cached
            # id() reused or content changed — invalidate
            del _MASK_CACHE[key]
            _MASK_CACHE_FP.pop(key, None)

    elev = np.asarray(elevation, dtype=np.float32)
    zeros_frac = float(np.mean(elev == 0.0)) if elev.size else 0.0
    if zeros_frac > assume_loaded_if_zeros_frac:
        sea_mask = elev == 0.0
    else:
        sea_mask = elev <= float(np.median(elev))

    land_mask = ~sea_mask
    # Store as read-only to catch accidental mutation
    sea_mask.flags.writeable = False
    land_mask.flags.writeable = False
    result: tuple[np.ndarray, np.ndarray] = (sea_mask, land_mask)
    if use_cache:
        elev_r = elev.ravel()
        n = elev_r.size
        fp = (float(elev_r[0]), float(elev_r[-1]), float(elev_r.sum())) if n >= 2 else (0.0, 0.0, 0.0)
        _MASK_CACHE_FP[id(elevation)] = fp
        _MASK_CACHE[id(elevation)] = result
    return result


def invalidate(elevation: np.ndarray) -> None:
    """Remove *elevation* from the mask cache (call after terrain mutation)."""
    key = id(elevation)
    _MASK_CACHE.pop(key, None)
    _MASK_CACHE_FP.pop(key, None)
