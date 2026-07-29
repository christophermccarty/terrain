"""Experimental land-surface runoff and river routing.

The climate core stores water as an equivalent depth per grid cell. Routing is
performed in area-weighted volume space so moving one millimetre from a wide
equatorial cell into a narrow high-latitude cell conserves water globally.
"""
from __future__ import annotations

import numpy as np

from masks import get_masks


def _receiver_indices(elevation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return steepest-downhill D8 receiver and whether a lower neighbor exists."""
    elev = np.asarray(elevation, dtype=np.float64)
    H, W = elev.shape
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]

    neighbor_elevations: list[np.ndarray] = []
    neighbor_rows: list[np.ndarray] = []
    neighbor_cols: list[np.ndarray] = []
    for dy, dx in (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ):
        rr = np.clip(rows + dy, 0, H - 1)
        cc = (cols + dx) % W
        neighbor_elevations.append(elev[rr, cc])
        neighbor_rows.append(np.broadcast_to(rr, (H, W)))
        neighbor_cols.append(np.broadcast_to(cc, (H, W)))

    stack = np.stack(neighbor_elevations, axis=0)
    direction = np.argmin(stack, axis=0)
    receiver_row = np.take_along_axis(
        np.stack(neighbor_rows, axis=0), direction[None, ...], axis=0
    )[0]
    receiver_col = np.take_along_axis(
        np.stack(neighbor_cols, axis=0), direction[None, ...], axis=0
    )[0]
    receiver_elevation = np.take_along_axis(
        stack, direction[None, ...], axis=0
    )[0]
    drains = receiver_elevation < (elev - 1e-9)
    return receiver_row, receiver_col, drains


def route_surface_water(
    elevation: np.ndarray,
    runoff_mm_day: np.ndarray,
    previous_storage_mm: np.ndarray | None,
    *,
    dt_days: float,
    routing_passes: int = 8,
    routing_fraction: float = 0.55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Route runoff downhill and return storage, throughflow, ocean outflow
    (by draining land cell), and ocean river input (by receiving ocean cell).

    This is a deliberately compact D8 routing model, not a channel hydraulics
    solver. Depressions retain water as lakes; cells with a downhill path pass a
    fraction of their available volume per routing pass. Longitude is periodic
    and latitude edges are clamped.

    `ocean_outflow_mm_day` is indexed at the *land* cell the water drains from
    (matches this function's original, still-used-for-display convention).
    `ocean_river_input_mm_day` is indexed at the *ocean* cell(s) that actually
    receive that flow (the D8 receiver), for feeding a river-mouth freshwater
    flux into ocean salinity -- distinct because a coastal land cell can be
    hydrologically empty (mask-wise 0) while still being the correct place to
    report outflow *from*, and one ocean cell can receive flow from several
    different draining land neighbors.
    """
    elev = np.asarray(elevation, dtype=np.float64)
    runoff = np.asarray(runoff_mm_day, dtype=np.float64)
    if runoff.shape != elev.shape:
        raise ValueError("runoff_mm_day must match elevation shape")
    if dt_days < 0.0:
        raise ValueError("dt_days must be non-negative")
    if routing_passes < 0:
        raise ValueError("routing_passes must be non-negative")
    if not 0.0 <= routing_fraction <= 1.0:
        raise ValueError("routing_fraction must be between 0 and 1")

    sea_mask, land_mask = get_masks(elev.astype(np.float32), use_cache=False)
    if previous_storage_mm is None:
        previous = np.zeros_like(elev)
    else:
        previous = np.asarray(previous_storage_mm, dtype=np.float64)
        if previous.shape != elev.shape:
            raise ValueError("previous_storage_mm must match elevation shape")
    if dt_days == 0.0:
        return (
            np.where(land_mask, np.maximum(previous, 0.0), 0.0).astype(np.float32),
            np.zeros_like(elev, dtype=np.float32),
            np.zeros_like(elev, dtype=np.float32),
            np.zeros_like(elev, dtype=np.float32),
        )

    H, W = elev.shape
    lat = (0.5 - (np.arange(H, dtype=np.float64) + 0.5) / H) * np.pi
    area_weight = np.broadcast_to(np.cos(lat)[:, None], (H, W))

    storage_volume = np.where(
        land_mask,
        (np.maximum(previous, 0.0) + np.maximum(runoff, 0.0) * dt_days)
        * area_weight,
        0.0,
    )
    throughflow_volume = np.zeros_like(storage_volume)
    ocean_outflow_volume = np.zeros_like(storage_volume)
    ocean_river_input_volume = np.zeros_like(storage_volume)
    receiver_row, receiver_col, drains = _receiver_indices(elev)
    source_can_drain = land_mask & drains

    flat_size = H * W
    receiver_flat = receiver_row * W + receiver_col
    receiver_is_land = land_mask[receiver_row, receiver_col]

    for _ in range(routing_passes):
        moved = np.where(
            source_can_drain,
            storage_volume * routing_fraction,
            0.0,
        )
        storage_volume -= moved
        throughflow_volume += moved

        # np.bincount is an exact match for np.add.at here (both accumulate
        # duplicate-index contributions rather than overwriting), but is far
        # faster at this array size -- np.add.at's per-element safety checks
        # made it the dominant cost of this function (measured: enabling
        # hydrology cost 2.7x wall time on a 512x1024 real-terrain run before
        # this change, almost entirely from the two add.at calls below).
        to_land = source_can_drain & receiver_is_land
        if np.any(to_land):
            storage_volume.reshape(flat_size)[:] += np.bincount(
                receiver_flat[to_land], weights=moved[to_land], minlength=flat_size
            )
        to_ocean = source_can_drain & ~receiver_is_land
        ocean_outflow_volume[to_ocean] += moved[to_ocean]
        if np.any(to_ocean):
            ocean_river_input_volume.reshape(flat_size)[:] += np.bincount(
                receiver_flat[to_ocean], weights=moved[to_ocean], minlength=flat_size
            )

    safe_area = np.maximum(area_weight, 1e-9)
    dt_safe = max(float(dt_days), 1e-9)
    storage_mm = np.where(land_mask, storage_volume / safe_area, 0.0)
    throughflow_mm_day = np.where(
        land_mask, throughflow_volume / safe_area / dt_safe, 0.0
    )
    ocean_river_input_mm_day = np.where(
        sea_mask, ocean_river_input_volume / safe_area / dt_safe, 0.0
    )
    ocean_outflow_mm_day = np.where(
        land_mask, ocean_outflow_volume / safe_area / dt_safe, 0.0
    )
    return (
        storage_mm.astype(np.float32),
        throughflow_mm_day.astype(np.float32),
        ocean_outflow_mm_day.astype(np.float32),
        ocean_river_input_mm_day.astype(np.float32),
    )
