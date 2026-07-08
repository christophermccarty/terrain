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
    """Remove *elevation* from the mask and continentality caches (call after terrain mutation)."""
    key = id(elevation)
    _MASK_CACHE.pop(key, None)
    _MASK_CACHE_FP.pop(key, None)
    _CONTINENTALITY_CACHE.pop(key, None)
    _CONTINENTALITY_CACHE_FP.pop(key, None)


# Module-level cache for get_continentality, same id+fingerprint convention as
# the mask cache above.
_CONTINENTALITY_CACHE: dict[int, np.ndarray] = {}
_CONTINENTALITY_CACHE_FP: dict[int, tuple[float, float, float]] = {}


def get_continentality(
    elevation: np.ndarray,
    *,
    coarsen: int = 4,
    iterations: int = 600,
    diffusion_coeff: float = 0.24,
    decay: float = 0.002,
    use_cache: bool = True,
) -> np.ndarray:
    """Return a ``[0, 1]`` "distance from coast" proxy for *elevation*.

    ``0`` at/near the coast, growing toward ``1`` deep in continental
    interiors (never quite reaching it -- see below); always ``0`` over
    ocean. Built with a *screened* diffusion process -- iteratively diffusing
    the sea mask inland (same periodic-longitude/clamped-latitude stencil as
    the rest of the codebase's smoothing passes, e.g. ``atmosphere._laplacian``)
    with a small decay term pulling every cell back toward 0 each pass. The
    decay term is what makes this a genuine distance-*like* field with a
    stable steady state and a natural length scale
    ``L ~ sqrt(diffusion_coeff / decay)`` grid cells: without it, plain
    diffusion with an ocean boundary pinned at 1 has no non-trivial steady
    state at all (it slowly saturates the entire domain to 1 given enough
    iterations, regardless of distance from the coast) -- confirmed by
    direct measurement while calibrating this function, not assumed. Not an
    exact Euclidean distance transform -- precision doesn't matter for a
    smooth amplification factor, and this keeps ``masks.py`` free of a new
    ``scipy`` dependency (not used anywhere else in the project) and avoids a
    circular import with ``atmosphere.py`` (which itself imports
    ``get_masks`` from this module).

    Diffusion runs on a coarsened grid (``coarsen``, nearest-neighbor
    downsample) purely for speed -- this is a smooth, low-frequency field, so
    the coarse-grid blockiness is irrelevant after diffusion smooths it out
    and it's read by physics that does further smoothing downstream anyway.
    Defaults calibrated on the real 512x1024 Earth terrain (`saves/earth.pkl`)
    to give a meaningfully differentiated spread across named regions: e.g.
    Siberia (deep interior) ~0.5, Canadian Prairies/US Midwest ~0.33-0.35,
    Central Europe/coastal regions ~0.03-0.05.

    Results are cached by ``id(elevation)`` with the same content fingerprint
    guard as :func:`get_masks`; pass ``use_cache=False`` for transient arrays.
    """
    if use_cache:
        key = id(elevation)
        if key in _CONTINENTALITY_CACHE:
            cached = _CONTINENTALITY_CACHE[key]
            elev_r = np.asarray(elevation, dtype=np.float32).ravel()
            n = elev_r.size
            fp = (float(elev_r[0]), float(elev_r[-1]), float(elev_r.sum())) if n >= 2 else (0.0, 0.0, 0.0)
            if cached.shape == elevation.shape and _CONTINENTALITY_CACHE_FP.get(key) == fp:
                return cached
            del _CONTINENTALITY_CACHE[key]
            _CONTINENTALITY_CACHE_FP.pop(key, None)

    elev = np.asarray(elevation, dtype=np.float32)
    H, W = elev.shape
    sea_mask, land_mask = get_masks(elev, use_cache=use_cache)

    cs = max(1, int(coarsen))
    Hc = max(1, H // cs)
    Wc = max(1, W // cs)
    sea_c = sea_mask[: Hc * cs : cs, : Wc * cs : cs].astype(np.float32)

    field = sea_c.copy()
    for _ in range(max(0, int(iterations))):
        n_ = np.concatenate([field[:1, :], field[:-1, :]], axis=0)      # clamped at poles
        s_ = np.concatenate([field[1:, :], field[-1:, :]], axis=0)
        e_ = np.concatenate([field[:, 1:], field[:, :1]], axis=1)       # periodic in longitude
        w_ = np.concatenate([field[:, -1:], field[:, :-1]], axis=1)
        lap = n_ + s_ + e_ + w_ - 4.0 * field
        field = field + float(diffusion_coeff) * lap - float(decay) * field
        field = np.clip(field, 0.0, 1.0)
        field = np.where(sea_c > 0.5, 1.0, field)  # re-pin the ocean source each pass

    field_full = np.repeat(np.repeat(field, cs, axis=0), cs, axis=1)
    field_full = field_full[:H, :W]
    if field_full.shape != (H, W):
        pad_h = H - field_full.shape[0]
        pad_w = W - field_full.shape[1]
        field_full = np.pad(field_full, ((0, pad_h), (0, pad_w)), mode="edge")

    continentality = (land_mask.astype(np.float32) * (1.0 - field_full)).astype(np.float32)
    continentality.flags.writeable = False

    if use_cache:
        elev_r = elev.ravel()
        n = elev_r.size
        fp = (float(elev_r[0]), float(elev_r[-1]), float(elev_r.sum())) if n >= 2 else (0.0, 0.0, 0.0)
        _CONTINENTALITY_CACHE_FP[id(elevation)] = fp
        _CONTINENTALITY_CACHE[id(elevation)] = continentality
    return continentality
