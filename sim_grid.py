"""Grid coarsening utilities and static-elevation cache."""
from __future__ import annotations

import numpy as np

_ELEV_COARSEN_CACHE: dict[tuple[int, int, int, int], np.ndarray] = {}
_ELEV_COARSEN_CACHE_FP: dict[
    tuple[int, int, int, int], tuple[float, float, float]
] = {}


def _pad_edge_inplace(buf: np.ndarray, H: int, W: int) -> None:
    """Fill padded margins by replicating the final valid row and column."""
    Hp, Wp = buf.shape[-2], buf.shape[-1]
    if Hp > H:
        buf[..., H:, :W] = buf[..., H - 1 : H, :W]
    if Wp > W:
        buf[..., :, W:] = buf[..., :, W - 1 : W]


def _coarsen(arr: np.ndarray, Hc: int, Wc: int, bs: int) -> np.ndarray:
    """Downsample a 2-D field by block averaging with edge padding."""
    source = arr if arr.dtype == np.float32 else arr.astype(np.float32)
    H, W = source.shape
    Hp, Wp = Hc * bs, Wc * bs
    if Hp == H and Wp == W:
        buffer = source
    else:
        buffer = np.empty((Hp, Wp), dtype=np.float32)
        buffer[:H, :W] = source
        _pad_edge_inplace(buffer, H, W)
    return (
        buffer.reshape(Hc, bs, Wc, bs)
        .mean(axis=(1, 3))
        .astype(np.float32, copy=False)
    )


def _coarsen_many(
    fields: dict[str, np.ndarray],
    Hc: int,
    Wc: int,
    bs: int,
) -> dict[str, np.ndarray]:
    """Coarsen same-shaped fields in one stacked reduction."""
    if not fields:
        return {}
    keys = list(fields)
    values = list(fields.values())
    count = len(values)
    H, W = values[0].shape
    Hp, Wp = Hc * bs, Wc * bs
    if Hp == H and Wp == W:
        stack = np.stack(
            [field if field.dtype == np.float32 else field.astype(np.float32)
             for field in values],
            axis=0,
        )
    else:
        stack = np.empty((count, Hp, Wp), dtype=np.float32)
        for index, field in enumerate(values):
            stack[index, :H, :W] = (
                field if field.dtype == np.float32 else field.astype(np.float32)
            )
        _pad_edge_inplace(stack, H, W)
    output = (
        stack.reshape(count, Hc, bs, Wc, bs)
        .mean(axis=(2, 4))
        .astype(np.float32, copy=False)
    )
    return {key: output[index] for index, key in enumerate(keys)}


def _coarsen_elevation_cached(
    elevation: np.ndarray,
    Hc: int,
    Wc: int,
    bs: int,
) -> np.ndarray:
    """Coarsen stable terrain once per identity, shape, and block size."""
    key = (id(elevation), Hc, Wc, bs)
    flattened = np.asarray(elevation, dtype=np.float32).ravel()
    size = flattened.size
    fingerprint = (
        (float(flattened[0]), float(flattened[-1]), float(flattened.sum()))
        if size >= 2
        else (0.0, 0.0, 0.0)
    )
    cached = _ELEV_COARSEN_CACHE.get(key)
    if (
        cached is not None
        and cached.shape == (Hc, Wc)
        and _ELEV_COARSEN_CACHE_FP.get(key) == fingerprint
    ):
        return cached
    result = _coarsen(elevation, Hc, Wc, bs)
    result.flags.writeable = False
    _ELEV_COARSEN_CACHE[key] = result
    _ELEV_COARSEN_CACHE_FP[key] = fingerprint
    return result


def clear_grid_caches() -> None:
    """Clear all coarsened static-terrain entries."""
    _ELEV_COARSEN_CACHE.clear()
    _ELEV_COARSEN_CACHE_FP.clear()
