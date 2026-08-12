"""Simple 3-cell-per-hemisphere wind model for equirectangular maps.

Hadley, Ferrel, and Polar cells are approximated by prescribing zonal (u)
and meridional (v) surface winds by latitude, with Coriolis turning that
creates easterlies and westerlies in the appropriate bands.

This is intentionally lightweight for interactive use; it returns a dense
vector field and a pre-rendered arrow RGB overlay for display.
"""

from __future__ import annotations

import math
from functools import lru_cache
import numpy as np
from temperature import temperature_kelvin_for_lat
from planet_params import PlanetParams, EARTH
from masks import get_masks, get_continentality
from condensate import (
    evolve_bulk_condensate,
    separate_cloud_and_hydrometeor_reservoirs,
    simplified_betts_miller_condensation,
    stability_aware_condensation,
)
from column_water import evolve_column_water
from pressure_column import (
    evolve_closed_three_level_thermodynamic_column,
    evolve_three_level_column,
)
from pressure_circulation import diabatic_interface_mass_flux
from pressure_circulation import smooth_spherical_scalar

# Numba JIT compilation for performance
try:
    from numba import jit, prange  # pyright: ignore[reportMissingImports]
    NUMBA_AVAILABLE = True
except ImportError:
    # Fallback: create dummy decorators if Numba not installed
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    NUMBA_AVAILABLE = False

# scipy is only used for semi-Lagrangian interpolation in evolve_wind's hot
# loop; import once at module level rather than on every call (it was being
# re-imported 8x per evolve_wind invocation via the per-substep code path).
try:
    from scipy.ndimage import map_coordinates as _scipy_map_coordinates  # pyright: ignore[reportMissingImports]
except ImportError:
    _scipy_map_coordinates = None


def _latitudes_h(height: int) -> np.ndarray:
    # Row-centered latitudes θ ∈ [π/2, -π/2] (north to south)
    return (0.5 - (np.arange(int(height), dtype=np.float32) + 0.5) / float(height)) * np.pi


# ---------------------------------------------------------------------------
# Cached static grids for evolve_wind. These depend only on (H, W) and a few
# planet constants, but were being rebuilt from scratch on every call — once
# per simulated day at production resolution. Keyed by grid shape + the
# planet parameters that enter the arrays.
# ---------------------------------------------------------------------------
_WIND_GRID_CACHE: dict = {"key": None, "grids": None}
_MGRID_CACHE: dict = {"key": None, "yx": None}


def _wind_static_grids(H: int, W: int, pp: PlanetParams):
    """Return (lat_2d, dx, dy, f, eq_window, lon_1d) for evolve_wind, cached."""
    key = (H, W, round(float(pp.radius_m), 1), round(float(pp.omega), 12),
           float(pp.rotation_direction))
    if _WIND_GRID_CACHE["key"] == key:
        return _WIND_GRID_CACHE["grids"]
    lat = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * np.pi
    lat_2d = np.repeat(lat[:, None], W, axis=1)
    # Floor cos(lat) so zonal grid spacing dx doesn't collapse to ~0 at the
    # pole: on a lat-lon grid, any zonal PGF term (dp_dx / dx) diverges as
    # cos(lat) -> 0, regardless of how physically small the actual zonal
    # pressure difference is. Real models handle this with a polar filter /
    # reduced grid; here we simply pin dx to its value at 80 deg poleward of
    # that, a standard "polar cap" simplification that only ever shrinks the
    # zonal PGF response in the cap (never inflates it elsewhere).
    cos_lat_floor = float(np.cos(np.deg2rad(65.0)))
    dx = pp.radius_m * (2 * np.pi / W) * np.maximum(np.cos(lat_2d), cos_lat_floor)
    dy = pp.radius_m * (np.pi / H)
    f = pp.coriolis_parameter(lat_2d)
    abs_lat_deg_2d = np.abs(np.rad2deg(lat_2d)).astype(np.float32, copy=False)
    eq_window = np.exp(-((abs_lat_deg_2d / 12.0) ** 2)).astype(np.float32, copy=False)
    lon_1d = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)
    grids = (lat_2d, dx, dy, f, eq_window, lon_1d)
    _WIND_GRID_CACHE.update({"key": key, "grids": grids})
    return grids


@lru_cache(maxsize=16)
def _column_water_spherical_geometry(H: int, W: int, radius_m: float):
    """Exact finite-volume geometry for the gated column-water transport."""
    dlat = np.pi / float(H)
    dlon = 2.0 * np.pi / float(W)
    edges = np.pi / 2.0 - np.arange(H + 1, dtype=np.float64) * dlat
    area_row = radius_m ** 2 * dlon * (np.sin(edges[:-1]) - np.sin(edges[1:]))
    area = np.broadcast_to(area_row[:, None], (H, W)).copy()
    # East/west faces run north-south.  North/south faces follow a latitude
    # circle and therefore shrink to zero at the poles.
    x_face_length = np.full((H, W), radius_m * dlat, dtype=np.float64)
    y_face_length = np.broadcast_to(
        (radius_m * np.maximum(np.cos(edges), 0.0) * dlon)[:, None],
        (H + 1, W),
    ).copy()
    return area, x_face_length, y_face_length


# ---------------------------------------------------------------------------
# Tunable atmospheric constants
# These control the large-scale circulation structure and are candidates for
# optimizer sweeps or per-planet customisation.
# ---------------------------------------------------------------------------

RHO_AIR: float = 1.225
"""Legacy Earth sea-level density [kg/m³]; retained for external imports."""

# 3-cell circulation centres and widths [degrees latitude]
HADLEY_CELL_CENTER_DEG: float = 14.0   # Trade-wind peak latitude
MID_LAT_JET_CENTER_DEG: float = 48.0   # Westerly jet core latitude
POLAR_CELL_CENTER_DEG:  float = 74.0   # Polar cell centre latitude

HADLEY_CELL_WIDTH_DEG:  float = 8.5
MID_LAT_JET_WIDTH_DEG:  float = 13.0
POLAR_CELL_WIDTH_DEG:   float = 9.0

# Zonal (u) and meridional (v) wind targets for the 3-cell relaxation [m/s]
U_TARGET_TRADE:  float = -6.0    # Trade easterlies (negative = easterly)
U_TARGET_MIDLAT: float = 11.5   # Westerly jet
U_TARGET_POLAR:  float = -1.5   # Polar easterlies
V_TARGET_TRADE:  float = -6.4   # Equatorward Hadley return flow
V_TARGET_MIDLAT: float = 10.0   # Poleward Ferrel flow
V_TARGET_POLAR:  float = -1.0   # Equatorward polar flow

# Rossby/synoptic wave modes: (zonal wavenumber, period_days, phase, amplitude_hPa)
ROSSBY_MODES: list[tuple[float, float, float, float]] = [
    (3.0, 20.0,  0.3, 0.60),   # wavenumber-3, 20-day period
    (5.0, 30.0,  1.1, 0.45),   # wavenumber-5, 30-day period
    (7.0, 45.0, -0.7, 0.30),   # wavenumber-7, 45-day period
]

# Precipitation latitude windows [degrees]
ITCZ_HALF_WIDTH_DEG:   float = 10.0   # ITCZ Gaussian half-width (σ) — narrowed 14→10° to reduce ITCZ over-precipitation
STORM_TRACK_CENTER_DEG: float = 48.0  # Mid-latitude storm track centre
DRYBELT_CENTER_DEG:     float = 28.0  # Subtropical dry belt centre

# Earth annual-mean zonal-mean precipitation shape, by |latitude| [mm/day at each
# 10-degree breakpoint], derived from diagnostics.EARTH_LATITUDE_BANDS_NH (mm/yr / 365.25).
# Used only as a *shape* (renormalized to a mean of 1.0 in _zonal_precip_target_profile,
# then multiplied by the caller's target_mean_mm_day) so it preserves the existing global
# calibration point exactly while giving the rescale a realistic latitudinal distribution.
#
# Why this exists (2026-07, real-world-vs-sim audit, see
# itcz-global-rescale-coupling-2026-07 memory): generate_precipitation's final rescale used
# to be a single flat scalar (target_mean_mm_day / mean(P)) applied uniformly to every cell.
# Because that scalar is solved to hit the *global* mean, reducing the ITCZ's own raw share
# (e.g. by trimming its lat_shape/post_shape weighting) just made the solver raise the scalar
# further to compensate, re-inflating the ITCZ almost as much as the trim removed -- a
# self-defeating feedback loop, confirmed by direct measurement (trimming the ITCZ weight to
# zero only cut tropical precip 13%, not the ~50% needed). A zonal (latitude-band) rescale
# breaks that coupling: the ITCZ and mid-latitudes now each get solved toward their own
# realistic target instead of sharing one knob.
_PRECIP_TARGET_LAT_BREAKS_DEG = np.array(
    [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0], dtype=np.float64)
_PRECIP_TARGET_MM_DAY = np.array(
    [5.48, 4.93, 2.74, 1.64, 2.47, 2.19, 1.37, 0.82, 0.55, 0.27], dtype=np.float64)

# The raw table above is only sampled every 10 degrees; plain linear
# interpolation draws a dead-straight ramp between each pair, and the 10-20
# deg segment (ITCZ edge -> subtropical dry belt, 4.93 -> 2.74 mm/day) is by
# far the steepest -- roughly 4x the slope of its neighbors. Combined with
# `moisture_budget_precip_rescale` enforcing this target tightly on every row
# regardless of the underlying terrain, that single steep linear segment
# produced a razor-sharp, perfectly latitude-aligned rainforest/desert
# boundary in the Koppen map (reported 2026-07-28, ~4yr into a MONTHLY
# real-terrain run) -- visible identically across every continent since the
# target has no longitude dependence at all. Real Earth's ITCZ-to-
# subtropical-high transition is smeared over roughly 20-25 degrees (monsoon
# systems, ocean-basin contrasts, and the ITCZ's own seasonal migration
# average out to a much gentler zonal-mean curve than this coarse 10-degree
# table implies). Pre-smoothing the profile with a Gaussian (sigma matched to
# that real-world transition width) spreads the same total precip drop over a
# wider latitude band instead of concentrating it in one 10-degree segment,
# without changing the overall equator-to-pole shape or the calibration this
# table encodes -- `_zonal_precip_target_profile` still renormalizes to a
# row-mean of 1.0 on the actual simulation grid, so the global calibration
# point this table anchors is untouched.
_PRECIP_TARGET_FINE_RES_DEG = 0.1
_PRECIP_TARGET_SMOOTH_SIGMA_DEG = 4.0


def _build_smoothed_precip_target_profile() -> tuple[np.ndarray, np.ndarray]:
    """Precompute a Gaussian-smoothed version of `_PRECIP_TARGET_MM_DAY` over a fine
    |latitude| grid -- see the comment above `_PRECIP_TARGET_FINE_RES_DEG` for why."""
    fine_lat = np.arange(0.0, 90.0 + _PRECIP_TARGET_FINE_RES_DEG, _PRECIP_TARGET_FINE_RES_DEG)
    fine_shape = np.interp(fine_lat, _PRECIP_TARGET_LAT_BREAKS_DEG, _PRECIP_TARGET_MM_DAY)

    # Mirror across the equator (x=0) so the kernel has real data on both
    # sides of that boundary instead of falling off toward zero there.
    mirrored_shape = np.concatenate([fine_shape[:0:-1], fine_shape])

    sigma_samples = _PRECIP_TARGET_SMOOTH_SIGMA_DEG / _PRECIP_TARGET_FINE_RES_DEG
    radius_samples = int(math.ceil(4.0 * sigma_samples))
    kernel_x = np.arange(-radius_samples, radius_samples + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (kernel_x / sigma_samples) ** 2)
    kernel /= kernel.sum()

    # Reflect-pad both ends (equator side already mirrored above; the polar
    # end is reflected about 90 deg) so the convolution doesn't lose
    # energy/flatten toward zero near either boundary.
    pad_lo = mirrored_shape[radius_samples:0:-1]
    pad_hi = mirrored_shape[-2:-radius_samples - 2:-1]
    padded = np.concatenate([pad_lo, mirrored_shape, pad_hi])
    smoothed_mirrored = np.convolve(padded, kernel, mode="valid")

    # Pull the 0..90 deg half back out (mirrored_shape[len(fine_lat)-1:] is
    # the 0..90 deg portion by construction above).
    half_start = len(fine_lat) - 1
    smoothed_shape = smoothed_mirrored[half_start:half_start + len(fine_lat)]
    return fine_lat, smoothed_shape.astype(np.float64)


_PRECIP_TARGET_FINE_LAT_DEG, _PRECIP_TARGET_FINE_SHAPE = _build_smoothed_precip_target_profile()


def _zonal_precip_target_profile(lat_deg: np.ndarray) -> np.ndarray:
    """Earth-like zonal precip *shape* by |lat|, normalized to an unweighted row-mean
    of 1.0 (matching the plain `np.mean(P)` convention the old scalar rescale used) so
    `profile * target_mean_mm_day` reproduces the exact same global calibration point.

    Sampled from a pre-smoothed fine-resolution curve (see
    `_build_smoothed_precip_target_profile`) rather than linearly interpolating the
    raw 10-degree breakpoint table directly -- the raw table's steepest segment
    (10-20 deg) was sharp enough to draw a razor-straight biome boundary on the
    actual simulation grid; see that function's docstring."""
    abs_lat = np.abs(np.asarray(lat_deg, dtype=np.float64))
    shape = np.interp(abs_lat, _PRECIP_TARGET_FINE_LAT_DEG, _PRECIP_TARGET_FINE_SHAPE)
    shape = shape / (float(np.mean(shape)) + 1e-12)
    return shape.astype(np.float32)


@lru_cache(maxsize=32)
def _itcz_window_annual_mean(
    H: int,
    itcz_seasonal_response: float,
    obliquity_deg: float,
    half_width_deg: float,
    n_samples: int = 48,
) -> tuple:
    """Time-average of `itcz_window` (see `generate_precipitation`) over one full
    seasonal cycle, per row -- the reference the seasonal-target-deficit fix
    (`PlanetParams.itcz_seasonal_target_response`) compares each month's actual
    `itcz_window` against.

    Independent of `orbital_period_days`/`vernal_equinox_day`: `equinox_phase`
    advances linearly with `day_of_year`, so a uniform grid over phase [0, 2pi)
    is exactly equivalent (by substitution) to a uniform grid over one full
    orbital period regardless of its length in days or phase offset -- both
    cancel out of the average. Cached (grid resolution + two scalar params
    only, no simulation state) since this would otherwise redo a 48-point
    sweep every precipitation step."""
    lat_deg = (0.5 - (np.arange(H, dtype=np.float64) + 0.5) / H) * 180.0
    obliquity_rad = math.radians(obliquity_deg)
    acc = np.zeros(H, dtype=np.float64)
    for i in range(n_samples):
        phase = 2.0 * math.pi * i / n_samples
        decl_rad = math.asin(math.sin(obliquity_rad) * math.sin(phase))
        center = itcz_seasonal_response * math.degrees(decl_rad)
        acc += np.exp(-(((lat_deg - center) / half_width_deg) ** 2))
    acc /= n_samples
    return tuple(acc.tolist())


def _moisture_budget_precip_rescale(
    dq: np.ndarray,
    q: np.ndarray,
    target_row_mm_day: np.ndarray,
    *,
    dt_days: float,
    column_mm_per_q: float,
    allocation_affinity: np.ndarray | None = None,
    target_cell_weight: np.ndarray | None = None,
    max_total_removal_fraction: float | np.ndarray = 0.85,
    max_added_removal_fraction: float | np.ndarray = 0.15,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Move row precipitation toward targets without multiplying every cell.

    Excess precipitation is scaled down conservatively. Deficits are filled
    preferentially where condensation is already active, subject to both the
    remaining atmospheric moisture and a per-step rainout cap. The target is
    therefore aspirational: dry/moisture-limited rows may remain below it
    instead of receiving an arbitrarily large multiplier.

    `target_cell_weight`, if given, reshapes `target_row_mm_day` (a single
    scalar per row) into a per-cell target via `target_row * weight`, where
    `weight` must average to 1.0 across each row (row *totals* -- and
    therefore the zonal calibration every latitude-band test checks -- are
    unaffected by construction). Without it (`None`, the default), every
    cell in a row shares the exact same target, reproducing this function's
    original row-uniform behavior byte-for-byte -- existing callers are
    unaffected. See its caller in `generate_precipitation` for why: a
    row-uniform target has zero longitude/terrain dependence, so the
    latitude at which a row's classification threshold gets crossed is
    identical for every column regardless of real terrain -- see
    razor-sharp-biome-line-precip-target-smoothing-2026-07-28 memory.

    `max_total_removal_fraction`/`max_added_removal_fraction` accept either a
    scalar (the original behavior, applied identically to every row -- exact
    byte-for-byte match with prior callers, including this module's own unit
    tests) or a per-row `np.ndarray` of shape `(H,)`, letting a caller vary
    the removal cap by latitude/regime (see `PlanetParams.
    moisture_budget_tropical_cap_boost`).
    """
    dq64 = np.asarray(dq, dtype=np.float64)
    q64 = np.maximum(np.asarray(q, dtype=np.float64), 0.0)
    targets = np.asarray(target_row_mm_day, dtype=np.float64)
    if dq64.shape != q64.shape:
        raise ValueError("dq and q must have identical shapes")
    if targets.shape != (dq64.shape[0],):
        raise ValueError("target_row_mm_day must have one value per latitude row")
    if allocation_affinity is None:
        allocation = np.ones_like(dq64)
    else:
        allocation = np.clip(np.asarray(allocation_affinity, dtype=np.float64), 0.0, None)
        if allocation.shape != dq64.shape:
            raise ValueError("allocation_affinity must match dq shape")
    if target_cell_weight is None:
        cell_weight = None
    else:
        cell_weight = np.asarray(target_cell_weight, dtype=np.float64)
        if cell_weight.shape != dq64.shape:
            raise ValueError("target_cell_weight must match dq shape")
    if dt_days <= 0.0 or column_mm_per_q <= 0.0:
        raise ValueError("dt_days and column_mm_per_q must be positive")

    H, W = dq64.shape
    total_cap_row = np.broadcast_to(
        np.asarray(max_total_removal_fraction, dtype=np.float64), (H,)
    )
    added_cap_row = np.broadcast_to(
        np.asarray(max_added_removal_fraction, dtype=np.float64), (H,)
    )
    result = np.clip(dq64, 0.0, total_cap_row[:, None] * q64)
    raw_row_mean = np.mean(result, axis=1)
    target_row_dq = targets * float(dt_days) / float(column_mm_per_q)
    capacity_limited = np.zeros(H, dtype=bool)
    unmet_row_mm_day = np.zeros(H, dtype=np.float64)

    for row in range(H):
        target_total = max(float(target_row_dq[row]), 0.0) * W
        target_cell_row = (
            None if cell_weight is None else target_row_dq[row] * cell_weight[row]
        )
        if target_cell_row is not None:
            # Always squeeze cells raining above their own (terrain-shaped)
            # share back down to it, regardless of whether the row in
            # aggregate lands in the trim or fill branch below. Gating this
            # solely on the row's aggregate trim/fill status (the previous
            # behavior) left individual over-share cells -- e.g. subsiding
            # desert land sitting in an otherwise moisture-limited row --
            # completely untouched whenever their row's *aggregate* fell in
            # the fill branch, which became common once
            # `_zonal_precip_target_profile`'s smoothing raised subtropical
            # row targets; that was a real desert-precipitation regression,
            # not just a reshaping no-op. The reclaimed amount isn't
            # discarded -- it lowers `current_total` below, so the fill step
            # further down has more room to redistribute it toward cells
            # still below their own share.
            excess_over_own = np.maximum(result[row] - target_cell_row, 0.0)
            if np.any(excess_over_own > 0.0):
                result[row] -= excess_over_own
        current_total = float(np.sum(result[row]))
        if current_total > target_total and current_total > 0.0:
            result[row] *= target_total / current_total
            continue
        remaining = target_total - current_total
        if remaining <= 1e-15:
            continue

        total_cap = total_cap_row[row] * q64[row]
        per_step_cap = result[row] + added_cap_row[row] * q64[row]
        capacity = np.maximum(np.minimum(total_cap, per_step_cap) - result[row], 0.0)
        if target_cell_row is not None:
            # Hard per-cell fill ceiling at each cell's own (terrain-shaped)
            # share -- without this, `affinity`'s super-linear existing-
            # condensation term can still out-vote `shortfall`'s (soft,
            # priority-only) deprioritization and push an already-wetter
            # desert-land cell's fill past its own share, even though it
            # never gets more than its capacity-limited allotment overall.
            # Since `target_cell_row` sums to exactly `target_total` across
            # the row (cell_weight's row-mean=1.0 construction), capping
            # every cell here can only ever leave moisture unplaced (reported
            # via `capacity_limited`/`unmet_row_mm_day`), never invent extra
            # target -- consistent with this function's "aspirational target"
            # contract.
            capacity = np.minimum(capacity, np.maximum(target_cell_row - result[row], 0.0))
        available = float(np.sum(capacity))
        if available <= 1e-15:
            capacity_limited[row] = True
            unmet_row_mm_day[row] = remaining * column_mm_per_q / (dt_days * W)
            continue

        requested = min(remaining, available)
        # Super-linear affinity concentrates correction in existing rain/cloud
        # systems. The small row-relative floor keeps perfectly uniform or
        # initially dry analytic fixtures well-defined without flattening real
        # longitudinal contrasts.
        affinity_floor = max(float(np.mean(result[row])) * 0.05, 1e-12)
        affinity = (result[row] + affinity_floor) ** 2 * allocation[row]
        if target_cell_row is not None:
            # Additionally prioritize cells that are furthest below their own
            # (terrain-shaped) share of the target, not just cells that
            # already happen to be raining -- ties the fill order directly to
            # the per-cell target shape instead of only to existing
            # condensation, so a capacity-limited row's shortfall lands
            # preferentially on cells whose *own* target is low (e.g.
            # subsiding/arid terrain) rather than spreading evenly across
            # every column regardless of local terrain.
            shortfall_floor = max(float(np.mean(target_cell_row)) * 0.05, 1e-12)
            shortfall = np.maximum(target_cell_row - result[row], 0.0) + shortfall_floor
            affinity = affinity * shortfall
        amount_left = requested
        active_capacity = capacity.copy()
        for _ in range(6):
            weights = affinity * active_capacity
            weight_sum = float(np.sum(weights))
            if amount_left <= 1e-15 or weight_sum <= 1e-30:
                break
            proposed = amount_left * weights / weight_sum
            accepted = np.minimum(proposed, active_capacity)
            result[row] += accepted
            active_capacity -= accepted
            amount_left -= float(np.sum(accepted))

        unmet = remaining - (requested - amount_left)
        if unmet > 1e-12:
            capacity_limited[row] = True
            unmet_row_mm_day[row] = unmet * column_mm_per_q / (dt_days * W)

    new_row_mean = np.mean(result, axis=1)
    effective_scale = new_row_mean / (raw_row_mean + 1e-12)
    achieved_fraction = np.divide(
        new_row_mean,
        target_row_dq,
        out=np.ones_like(new_row_mean),
        where=target_row_dq > 1e-12,
    )
    return result.astype(np.float32), {
        "effective_scale": effective_scale.astype(np.float32),
        "target_achieved_fraction": achieved_fraction.astype(np.float32),
        "capacity_limited": capacity_limited,
        "unmet_row_mm_day": unmet_row_mm_day.astype(np.float32),
    }

# Discrete moving storm/wave systems. Unlike ROSSBY_MODES (a standing sinusoid
# that only ever translates -- same wavenumber and amplitude forever, which is
# what makes it look mechanically repetitive no matter how long you watch it),
# these are individual pressure cells with a birth/track/death lifecycle,
# deterministically generated from `time_days` alone (see _storm_pressure_anomaly).
#
# Two populations, matching this model's own wind climatology:
# - Mid-latitude storms (35-55 deg): eastward-translating cyclones, matching the
#   westerly jet (U_TARGET_MIDLAT is positive/eastward).
# - Trade-wind/subtropical waves (12-32 deg): westward-translating disturbances
#   (real-world analogue: easterly waves), matching the trade easterlies
#   (U_TARGET_TRADE is negative/westward) -- this band is where the "faint
#   ripple" from Rossby waves alone is most visually dominant/static, since it
#   previously had no birth/death transient mechanism at all (v1 storm scope
#   was mid-latitude-only). Weaker amplitude and shorter lifecycle than the
#   mid-latitude storms, matching real easterly waves' smaller/faster character.
N_STORM_SLOTS: int = 4                # concurrent storm slots per hemisphere
STORM_LIFECYCLE_DAYS: float = 9.0     # spin-up + mature + decay, per storm
STORM_LAT_CENTER_DEG: float = 45.0    # genesis latitude (matches Rossby storm_w window)
STORM_LAT_JITTER_DEG: float = 10.0
STORM_LON_DRIFT_DEG_PER_DAY: tuple[float, float] = (5.0, 11.0)   # eastward translation range
STORM_LAT_DRIFT_DEG_PER_DAY: tuple[float, float] = (0.15, 0.55)  # poleward drift range
STORM_RADIUS_KM: tuple[float, float] = (900.0, 1600.0)

N_TRADE_WAVE_SLOTS: int = 5                 # concurrent wave slots per hemisphere
TRADE_WAVE_LIFECYCLE_DAYS: float = 5.0      # faster life cycle than mid-lat storms
TRADE_WAVE_LAT_CENTER_DEG: float = 22.0     # genesis latitude (trade-wind/subtropical belt)
TRADE_WAVE_LAT_JITTER_DEG: float = 10.0
TRADE_WAVE_LON_DRIFT_DEG_PER_DAY: tuple[float, float] = (-13.0, -6.0)  # westward (easterly flow)
TRADE_WAVE_LAT_DRIFT_DEG_PER_DAY: tuple[float, float] = (-0.20, 0.20)  # weak/mixed drift
TRADE_WAVE_RADIUS_KM: tuple[float, float] = (500.0, 1000.0)


def _storm_pressure_anomaly(
    lat_2d: np.ndarray,
    lon_1d: np.ndarray,
    time_days: float,
    amp_pa: float,
    n_slots: int = N_STORM_SLOTS,
    lifecycle_days: float = STORM_LIFECYCLE_DAYS,
    lat_center_deg: float = STORM_LAT_CENTER_DEG,
    lat_jitter_deg: float = STORM_LAT_JITTER_DEG,
    lon_drift_range: tuple[float, float] = STORM_LON_DRIFT_DEG_PER_DAY,
    lat_drift_range: tuple[float, float] = STORM_LAT_DRIFT_DEG_PER_DAY,
    radius_km_range: tuple[float, float] = STORM_RADIUS_KM,
    population_id: int = 0,
    lat_shift_nh_deg: float = 0.0,
    lat_shift_sh_deg: float = 0.0,
    planet_radius_km: float = 6371.0,
) -> np.ndarray:
    """Deterministic, stateless pressure anomaly from a population of moving storm/wave systems.

    A pure function of `time_days`: identical `time_days` always yields
    identical output (no global RNG state touched, no persistent storm
    identity stored anywhere), matching the same reproducibility contract as
    ROSSBY_MODES. Each of `n_slots` slots per hemisphere cycles through births
    spaced `lifecycle_days` apart; each instance's genesis position/track/
    strength is drawn from a fresh RNG seeded purely from its (population,
    hemisphere, slot, generation) identity, so a given storm/wave's entire
    life history is fully determined by when it was born. `population_id`
    only needs to differ between calls that could otherwise collide on the
    same (hemisphere, slot, generation) key (e.g. mid-latitude storms vs.
    trade-wind waves called with the same slot count).

    `lat_shift_{nh,sh}_deg` bias the genesis latitude per hemisphere (used to
    make storm genesis track a meandering jet, atmosphere._update_jet_index);
    both default to 0.0, matching the original behaviour.

    Returns a (H, W) float32 Pa anomaly to be added to the caller's `p_anom`.
    """
    H, W = lat_2d.shape
    out = np.zeros((H, W), dtype=np.float32)
    if amp_pa == 0.0:
        return out
    t = float(time_days)
    for hemi in (1.0, -1.0):
        lat_shift = lat_shift_nh_deg if hemi > 0 else lat_shift_sh_deg
        for slot_i in range(n_slots):
            slot_offset = slot_i * lifecycle_days / n_slots
            gen_i = math.floor((t - slot_offset) / lifecycle_days)
            t_local = (t - slot_offset) - gen_i * lifecycle_days
            frac = t_local / lifecycle_days
            envelope = math.sin(math.pi * frac) ** 0.7 if 0.0 < frac < 1.0 else 0.0
            if envelope <= 1e-4:
                continue
            key = population_id * 10_000_000 + int(hemi > 0) * 1_000_003 + slot_i * 97 + gen_i * 131
            rng = np.random.default_rng(abs(int(key)))
            birth_lon_deg = rng.uniform(-180.0, 180.0)
            birth_lat_deg = hemi * (lat_center_deg + lat_shift + rng.uniform(-lat_jitter_deg, lat_jitter_deg))
            dlon_dt = rng.uniform(*lon_drift_range)
            dlat_dt = hemi * rng.uniform(*lat_drift_range)
            radius_km = rng.uniform(*radius_km_range)
            peak_pa = amp_pa * rng.uniform(0.7, 1.3)

            lon_now_deg = birth_lon_deg + dlon_dt * t_local
            lat_now_deg = birth_lat_deg + dlat_dt * t_local
            lon_now = math.radians(((lon_now_deg + 180.0) % 360.0) - 180.0)
            lat_now = math.radians(float(np.clip(lat_now_deg, -85.0, 85.0)))

            dlat = lat_2d - lat_now
            dlon = (lon_1d - lon_now + np.pi) % (2 * np.pi) - np.pi
            dx_km = dlon * np.cos(lat_2d) * planet_radius_km
            dy_km = dlat * planet_radius_km
            d2 = dx_km * dx_km + dy_km * dy_km
            out -= (envelope * peak_pa) * np.exp(-d2 / (radius_km * radius_km)).astype(np.float32, copy=False)
    return out


# ---------------------------------------------------------------------------
# Jet stream dynamics: persistent meander index + blocking events
#
# Unlike ROSSBY_MODES and _storm_pressure_anomaly above (both pure, stateless
# functions of time_days), a meandering/blocking jet genuinely needs memory:
# a blocking ridge holds a fixed longitude for weeks regardless of what the
# pressure field would otherwise do. These two "_update_*" functions are the
# only pieces of real prognostic state in the jet-stream feature (persisted
# in PlanetState); everything downstream (evolve_wind's use of the resulting
# index/block values) stays a pure function of its inputs, and the noise/
# trigger draws are seeded from time_days -- not a stored RNG -- so a given
# (state, total_days) pair always produces the same next state.
# ---------------------------------------------------------------------------

def _update_jet_index(
    index_prev: float,
    gradient_k: float,
    dt_days: float,
    total_days: float,
    hemisphere_seed: int,
    tau_days: float = 10.0,
    noise_amp: float = 0.35,
    gradient_ref_k: float = 40.0,
) -> float:
    """AR1 update of the persistent jet meander/waviness index.

    Mean-reverts toward a target derived from the actual simulated
    pole-equator temperature gradient: gradients weaker than `gradient_ref_k`
    push the target positive (wavier, more-easily-blocked jet); stronger
    gradients push it negative (fast, zonal jet). This is a simplified
    stand-in for the Arctic-amplification-weakens-the-jet hypothesis, tied to
    physics the model already simulates (ice cover / polar cooling change the
    gradient) rather than an independent decorative signal.

    Stochastic forcing is a deterministic hashed draw seeded from
    `total_days` (not a stored RNG), matching the reproducibility contract of
    ROSSBY_MODES / _storm_pressure_anomaly: identical (total_days, inputs)
    always yields identical output.
    """
    ref = max(float(gradient_ref_k), 1e-6)
    target = float(np.clip((ref - float(gradient_k)) / ref, -1.0, 1.0))
    tau = max(float(tau_days), 1e-6)
    dt = float(dt_days)
    a = 1.0 - math.exp(-dt / tau)

    seed_key = int(round(float(total_days) * 1000.0)) * 100 + int(hemisphere_seed)
    rng = np.random.default_rng(abs(seed_key))
    noise = float(rng.normal(0.0, float(noise_amp) * math.sqrt(max(dt, 1e-6))))

    index_new = float(index_prev) + a * (target - float(index_prev)) + noise
    return float(np.clip(index_new, -2.0, 2.0))


def _update_jet_blocking(
    block_lon_prev: float,
    days_left_prev: float,
    total_duration_prev: float,
    jet_index: float,
    dt_days: float,
    total_days: float,
    hemisphere_seed: int,
    trigger_rate_per_day: float = 0.015,
    duration_range_days: tuple[float, float] = (10.0, 40.0),
) -> tuple[float, float, float]:
    """Two-state blocking-ridge machine.

    Active: holds a fixed longitude for the remainder of a drawn duration
    (a real block is quasi-stationary once established -- it doesn't drift
    like a storm). Inactive: rolls a deterministic hashed trigger each step,
    scaled up when the jet index is already elevated (wavier flow is more
    prone to amplifying into a cutoff/blocking pattern).

    Returns (block_lon_deg, days_left, total_duration) -- block_lon_deg is
    -1.0 and the other two are 0.0 when inactive. `total_duration` is carried
    alongside `days_left` (rather than only decrementing a countdown) purely
    so the caller can compute a smooth ramp-up/ramp-down envelope without
    needing to know how long ago the block started from `days_left` alone.
    """
    dt = float(dt_days)
    if days_left_prev > 0.0:
        days_left_new = days_left_prev - dt
        if days_left_new > 0.0:
            return float(block_lon_prev), float(days_left_new), float(total_duration_prev)
        return -1.0, 0.0, 0.0

    seed_key = int(round(float(total_days) * 1000.0)) * 100 + int(hemisphere_seed) + 50
    rng = np.random.default_rng(abs(seed_key))
    # Sigmoid centered at index=0.5: near-baseline flow rarely blocks, wavy flow often does.
    waviness = 1.0 / (1.0 + math.exp(-3.0 * (float(jet_index) - 0.5)))
    p_trigger = float(trigger_rate_per_day) * waviness * dt
    if rng.random() < p_trigger:
        lon = float(rng.uniform(-180.0, 180.0))
        duration = float(rng.uniform(*duration_range_days))
        return lon, duration, duration
    return -1.0, 0.0, 0.0


def _blocking_ridge_pressure_anomaly(
    lat_2d: np.ndarray,
    lon_1d: np.ndarray,
    lat_center_deg: float,
    lon_center_deg: float,
    days_left: float,
    total_duration_days: float,
    amp_pa: float,
    radius_km: float,
    ramp_days: float = 2.0,
    planet_radius_km: float = 6371.0,
) -> np.ndarray:
    """Stationary high-pressure blob for an active blocking ridge.

    Unlike _storm_pressure_anomaly (a moving low with a sin(pi*frac)
    lifecycle envelope), a block is a persistent, quasi-stationary high:
    fixed lat/lon for its whole lifetime, with a short ramp-up/ramp-down
    (both ends, `ramp_days` each) instead of a full spin-up/decay lifecycle,
    since a real block's onset/decay is slower/smoother than a storm's --
    and a hard on/off step risked exactly the kind of discontinuous-forcing
    runaway already noted (and avoided) elsewhere in this module's PGF
    terms.
    """
    H, W = lat_2d.shape
    if amp_pa == 0.0 or days_left <= 0.0:
        return np.zeros((H, W), dtype=np.float32)

    elapsed = float(total_duration_days) - float(days_left)
    ramp = max(float(ramp_days), 1e-6)
    envelope = min(elapsed / ramp, float(days_left) / ramp, 1.0)
    envelope = float(np.clip(envelope, 0.0, 1.0))
    if envelope <= 1e-4:
        return np.zeros((H, W), dtype=np.float32)

    lat_c = math.radians(float(lat_center_deg))
    lon_c = math.radians(((float(lon_center_deg) + 180.0) % 360.0) - 180.0)
    dlat = lat_2d - lat_c
    dlon = (lon_1d - lon_c + np.pi) % (2 * np.pi) - np.pi
    dx_km = dlon * np.cos(lat_2d) * planet_radius_km
    dy_km = dlat * planet_radius_km
    d2 = dx_km * dx_km + dy_km * dy_km
    r = float(radius_km)
    return (envelope * float(amp_pa) * np.exp(-d2 / (r * r))).astype(np.float32, copy=False)


def _synoptic_wave_pressure_anomaly(
    lat_2d: np.ndarray,
    lon_1d: np.ndarray,
    time_days: float,
    jet_index_nh: float,
    jet_index_sh: float,
    jet_block_nh: tuple[float, float, float],
    jet_block_sh: tuple[float, float, float],
    pp: PlanetParams,
) -> np.ndarray:
    """Deterministic synoptic-scale pressure perturbation [Pa]: Rossby waves +
    discrete storm/trade-wave systems + blocking ridges, all keyed off
    `time_days` and modulated by the persistent jet meander/blocking state.

    Extracted out of evolve_wind's inline pressure-anomaly construction so
    generate_wind_field's *diagnostic* wind (used whenever full prognostic
    evolution is skipped -- MONTHLY/ANNUAL time-scale mode, and the
    wind_relax blending target in faster modes) carries the same storm/jet
    signal as the prognostic surface layer, instead of falling back to an
    entirely separate, simpler wave model with different constants and no
    storms/blocking at all. That mismatch was what made switching speed
    modes visibly disrupt the monthly precip statistics Köppen classification
    reads (see jet-stream-vs-real-world memory).
    """
    H, W = lat_2d.shape
    t = float(time_days)
    abs_deg_1d = np.rad2deg(np.abs(lat_2d[:, 0])).astype(np.float32, copy=False)
    sign_lat_1d = np.sign(lat_2d[:, 0]).astype(np.float32, copy=False)
    storm_w_base = np.exp(-((abs_deg_1d - 45.0) / 18.0) ** 2).astype(np.float32, copy=False)
    wave_scale_nh = 1.0 + float(pp.jet_wave_amp_scale_per_index) * float(jet_index_nh)
    wave_scale_sh = 1.0 + float(pp.jet_wave_amp_scale_per_index) * float(jet_index_sh)
    storm_w = np.where(sign_lat_1d >= 0.0, storm_w_base * wave_scale_nh, storm_w_base * wave_scale_sh).astype(np.float32, copy=False)

    wave = np.zeros((H, W), dtype=np.float32)
    for k, per, ph, amp_hpa in ROSSBY_MODES:
        wave += (amp_hpa * 100.0) * np.cos(k * lon_1d[None, :] + (2.0 * np.pi * t / per) + ph).astype(np.float32, copy=False)
    p_anom = storm_w[:, None] * wave

    _pr_km = float(pp.radius_m) / 1000.0
    storm_amp = float(pp.storm_pressure_amp_pa)
    if storm_amp != 0.0:
        p_anom = p_anom + _storm_pressure_anomaly(
            lat_2d, lon_1d[None, :], t, storm_amp, population_id=0,
            lat_shift_nh_deg=float(pp.jet_lat_shift_per_index) * float(jet_index_nh),
            lat_shift_sh_deg=float(pp.jet_lat_shift_per_index) * float(jet_index_sh),
            planet_radius_km=_pr_km,
        )

    trade_wave_amp = float(pp.trade_wave_pressure_amp_pa)
    if trade_wave_amp != 0.0:
        p_anom = p_anom + _storm_pressure_anomaly(
            lat_2d, lon_1d[None, :], t, trade_wave_amp,
            n_slots=N_TRADE_WAVE_SLOTS,
            lifecycle_days=TRADE_WAVE_LIFECYCLE_DAYS,
            lat_center_deg=TRADE_WAVE_LAT_CENTER_DEG,
            lat_jitter_deg=TRADE_WAVE_LAT_JITTER_DEG,
            lon_drift_range=TRADE_WAVE_LON_DRIFT_DEG_PER_DAY,
            lat_drift_range=TRADE_WAVE_LAT_DRIFT_DEG_PER_DAY,
            radius_km_range=TRADE_WAVE_RADIUS_KM,
            population_id=1,
            planet_radius_km=_pr_km,
        )

    block_amp = float(pp.jet_block_pressure_amp_pa)
    block_radius = float(pp.jet_block_radius_km)
    if block_amp != 0.0:
        jet_lat_nh = MID_LAT_JET_CENTER_DEG + float(pp.jet_lat_shift_per_index) * float(jet_index_nh)
        jet_lat_sh = -(MID_LAT_JET_CENTER_DEG + float(pp.jet_lat_shift_per_index) * float(jet_index_sh))
        block_lon_nh, block_days_left_nh, block_total_nh = jet_block_nh
        block_lon_sh, block_days_left_sh, block_total_sh = jet_block_sh
        p_anom = p_anom + _blocking_ridge_pressure_anomaly(
            lat_2d, lon_1d[None, :], jet_lat_nh, block_lon_nh,
            block_days_left_nh, block_total_nh, block_amp, block_radius,
            planet_radius_km=_pr_km,
        )
        p_anom = p_anom + _blocking_ridge_pressure_anomaly(
            lat_2d, lon_1d[None, :], jet_lat_sh, block_lon_sh,
            block_days_left_sh, block_total_sh, block_amp, block_radius,
            planet_radius_km=_pr_km,
        )
    return p_anom


def _coarse_shape(H: int, W: int, block_size: int) -> tuple[int, int]:
    bs = max(1, int(block_size))
    Hc = max(1, (H + bs - 1) // bs)
    Wc = max(1, (W + bs - 1) // bs)
    return Hc, Wc


def _upsample_repeat(field: np.ndarray, H: int, W: int, block_size: int) -> np.ndarray:
    bs = max(1, int(block_size))
    up = np.repeat(np.repeat(field, bs, axis=0), bs, axis=1)
    return up[:H, :W]


def _upsample_bilinear(field: np.ndarray, H: int, W: int, block_size: int) -> np.ndarray:
    """Upsample using bilinear interpolation to eliminate blocky artifacts."""
    bs = max(1, int(block_size))
    Hc, Wc = field.shape
    if bs == 1:
        return field[:H, :W]

    y0, y1, wy, x0, x1, wx = _bilinear_plan(int(H), int(W), int(Hc), int(Wc))
    f0 = field[y0, :].astype(np.float32, copy=False)
    f1 = field[y1, :].astype(np.float32, copy=False)
    top = f0[:, x0] * (1.0 - wx)[None, :] + f0[:, x1] * wx[None, :]
    bot = f1[:, x0] * (1.0 - wx)[None, :] + f1[:, x1] * wx[None, :]
    return (top * (1.0 - wy)[:, None] + bot * wy[:, None]).astype(np.float32, copy=False)


@lru_cache(maxsize=64)
def _bilinear_plan(H: int, W: int, Hc: int, Wc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 1D sampling plan; avoids per-call 2D meshgrid/indices allocations.
    y = np.linspace(0, Hc - 1, int(H), dtype=np.float32)
    x = np.linspace(0, Wc - 1, int(W), dtype=np.float32)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    wy = (y - y0).astype(np.float32, copy=False)
    wx = (x - x0).astype(np.float32, copy=False)
    y0 = np.clip(y0, 0, Hc - 1)
    x0 = np.clip(x0, 0, Wc - 1)
    y1 = np.clip(y0 + 1, 0, Hc - 1)
    x1 = np.clip(x0 + 1, 0, Wc - 1)
    return y0, y1, wy, x0, x1, wx


def _upsample_bilinear_many(fields: dict[str, np.ndarray], H: int, W: int, block_size: int) -> dict[str, np.ndarray]:
    """Upsample multiple (Hc,Wc) fields sharing the same sampling plan."""
    if not fields:
        return {}
    bs = max(1, int(block_size))
    first = next(iter(fields.values()))
    Hc, Wc = first.shape
    if bs == 1:
        return {k: v[:H, :W] for k, v in fields.items()}

    keys = list(fields.keys())
    stack = np.stack([fields[k].astype(np.float32, copy=False) for k in keys], axis=0)
    y0, y1, wy, x0, x1, wx = _bilinear_plan(int(H), int(W), int(Hc), int(Wc))

    if NUMBA_AVAILABLE:
        out = _upsample_bilinear_numba_kernel(
            stack,
            y0.astype(np.int32), y1.astype(np.int32), wy.astype(np.float32),
            x0.astype(np.int32), x1.astype(np.int32), wx.astype(np.float32),
        )
    else:
        f0 = stack[:, y0, :]
        f1 = stack[:, y1, :]
        top = f0[:, :, x0] * (1.0 - wx)[None, None, :] + f0[:, :, x1] * wx[None, None, :]
        bot = f1[:, :, x0] * (1.0 - wx)[None, None, :] + f1[:, :, x1] * wx[None, None, :]
        out = (top * (1.0 - wy)[None, :, None] + bot * wy[None, :, None]).astype(np.float32, copy=False)
    return {k: out[i] for i, k in enumerate(keys)}


def _majority_filter(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(np.int8)
    for _ in range(max(1, iterations)):
        pad = np.pad(out, 1, mode="edge")
        neigh = (
            pad[0:-2, 0:-2] + pad[0:-2, 1:-1] + pad[0:-2, 2:]
            + pad[1:-1, 0:-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:]
            + pad[2:, 0:-2] + pad[2:, 1:-1] + pad[2:, 2:]
        )
        out = (neigh >= 5).astype(np.int8)
    return out.astype(bool)


def _derive_land_sea_masks(elevation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper around the canonical mask utility."""
    sea, land = get_masks(np.asarray(elevation, dtype=np.float32), use_cache=False)
    return land, sea


def _laplacian(field: np.ndarray) -> np.ndarray:
    # Periodic in longitude (axis=1), clamped at poles (axis=0).
    n = np.concatenate([field[:1, :], field[:-1, :]], axis=0)   # north (edge-clamped)
    s = np.concatenate([field[1:, :], field[-1:, :]], axis=0)   # south (edge-clamped)
    e = np.concatenate([field[:, 1:], field[:, :1]], axis=1)    # east (periodic)
    w = np.concatenate([field[:, -1:], field[:, :-1]], axis=1)  # west (periodic)
    return n + s + e + w - 4.0 * field


def _ddx_periodic(field: np.ndarray) -> np.ndarray:
    """Central difference in x with periodic wrap (axis=1). Returns derivative per grid index."""
    return 0.5 * (
        np.concatenate([field[:, 1:], field[:, :1]], axis=1)
        - np.concatenate([field[:, -1:], field[:, :-1]], axis=1)
    )


def _zonal_gaussian_smooth(field: np.ndarray, sigma_deg: float) -> np.ndarray:
    """Smooth `field` along longitude (axis=1) with a periodic Gaussian kernel.

    FFT-based circular convolution (numpy-only, no scipy dependency) --
    the longitude axis genuinely wraps, so a periodic kernel is exact here
    rather than an edge-effect approximation. `sigma_deg` is in degrees of
    longitude; converted to grid cells internally so it's resolution-
    independent. A `sigma_deg` of ~0 (below one grid cell) returns `field`
    unchanged rather than dividing by a near-zero kernel width.
    """
    W = field.shape[1]
    sigma_cells = sigma_deg / 360.0 * float(W)
    if sigma_cells <= 0.5:
        return field
    idx = np.arange(W)
    dist = np.minimum(idx, W - idx).astype(np.float64)
    kernel = np.exp(-0.5 * (dist / sigma_cells) ** 2)
    kernel /= kernel.sum()
    field_hat = np.fft.rfft(field.astype(np.float64), axis=1)
    kernel_hat = np.fft.rfft(kernel)
    smoothed = np.fft.irfft(field_hat * kernel_hat[None, :], n=W, axis=1)
    return smoothed.astype(np.float32, copy=False)


def _advect_scalar(
    field: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    u_scale: np.ndarray,
    v_scale: np.ndarray,
) -> np.ndarray:
    """Short-range donor-cell blend (NumPy fallback for `_advect_humidity_numba`).

    Empirically load-bearing for `generate_precipitation`'s continental-
    interior/desert balance (see known-physics-gaps.md's moisture-transport
    investigation): a full-distance semi-Lagrangian replacement dried
    continental land out further, apparently because this short-range blend
    keeps evaporated moisture from moving away faster than the RH/
    convergence-based precip trigger can capture it. Kept as the base term,
    blended with a longer-range semi-Lagrangian contribution
    (`_advect_scalar_semi_lagrangian`) via `moisture_advection_scale`.
    """
    east = np.roll(field, -1, axis=1)
    west = np.roll(field, 1, axis=1)
    # Meridional neighbors are edge-clamped, NOT wrapped: np.roll on axis 0
    # would connect the north pole row to the south pole row. Row index
    # increases southward, so the southern neighbor of row i is row i+1.
    # Upwind donor matches _advect_humidity_numba: northward wind (v>=0)
    # brings air from the south (row i+1).
    row_south = np.concatenate([field[1:, :], field[-1:, :]], axis=0)  # field[i+1]
    row_north = np.concatenate([field[:1, :], field[:-1, :]], axis=0)  # field[i-1]
    adv_x = field + u_scale * (np.where(u >= 0, west, east) - field)
    adv_xy = adv_x + v_scale * (np.where(v >= 0, row_south, row_north) - adv_x)
    return adv_xy


# ============================================================================
# Numba-accelerated compute kernels for wind evolution
# These provide 10-50x speedup for the most expensive operations
# ============================================================================

@jit(nopython=True, parallel=True, cache=True)
def _friction_kernel_numba(u: np.ndarray, v: np.ndarray, elevation: np.ndarray,
                           drag_base: float, drag_elev_scale: float,
                           eq_damping: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply quadratic friction (drag * v * |v|) with elevation enhancement.

    Returns du, dv scaled by dt (when dt=1.0, numerically equals acceleration).
    Caller multiplies result by dt_sub to get total velocity change.

    BUG FIX: Uses quadratic friction to match fallback code.
    """
    H, W = u.shape
    du = np.zeros_like(u)
    dv = np.zeros_like(v)

    for i in prange(H):
        for j in range(W):
            # Elevation-enhanced drag
            drag = drag_base + drag_elev_scale * elevation[i, j]
            # Equatorial boost
            drag_total = drag + eq_damping[i, j] * 2.0e-6
            # Quadratic friction: -drag * |V| * (u, v) * dt.
            # |V| is the full wind speed — using |u| and |v| separately would
            # decouple the components (e.g. u=3, v=4 should both feel |V|=5).
            speed = (u[i, j] * u[i, j] + v[i, j] * v[i, j]) ** 0.5
            du[i, j] = -drag_total * u[i, j] * speed * dt
            dv[i, j] = -drag_total * v[i, j] * speed * dt

    return du, dv


# ============================================================================
# Numba-accelerated compute kernels for precipitation
# These accelerate humidity advection, moisture convergence, and precipitation
# ============================================================================

@jit(nopython=True, parallel=True, cache=True)
def _advect_humidity_numba(q: np.ndarray, u: np.ndarray, v: np.ndarray,
                           u_scale: np.ndarray, v_scale: np.ndarray) -> np.ndarray:
    """Short-range donor-cell blend (Numba fast path for `_advect_scalar`).

    See `_advect_scalar`'s docstring: kept as the base term in
    `generate_precipitation`'s moisture advection, blended with a longer-
    range semi-Lagrangian contribution via `moisture_advection_scale`.
    """
    H, W = q.shape
    q_out = np.zeros_like(q)

    for i in prange(H):
        for j in range(W):
            # Zonal advection (periodic boundary)
            j_east = (j + 1) % W
            j_west = (j - 1 + W) % W

            if u[i, j] >= 0:
                q_x = q[i, j_west]
            else:
                q_x = q[i, j_east]

            q_adv_x = q[i, j] + u_scale[i, j] * (q_x - q[i, j])

            # Meridional advection (edge boundary)
            if i == 0:
                q_out[i, j] = q_adv_x  # North pole edge
            elif i == H - 1:
                q_out[i, j] = q_adv_x  # South pole edge
            else:
                if v[i, j] >= 0:
                    q_y = q[i + 1, j]  # Southward
                else:
                    q_y = q[i - 1, j]  # Northward

                q_out[i, j] = q_adv_x + v_scale[i, j] * (q_y - q_adv_x)

    return q_out


@jit(nopython=True, parallel=True, cache=True)
def _laplacian_numba(field: np.ndarray) -> np.ndarray:
    """Compute Laplacian (5-point stencil) with periodic x, edge y boundaries.

    Returns Laplacian of field (∇²f).
    """
    H, W = field.shape
    lap = np.zeros_like(field)

    for i in prange(1, H - 1):  # Skip poles
        for j in range(W):
            j_east = (j + 1) % W
            j_west = (j - 1 + W) % W

            # 5-point stencil
            c = field[i, j]
            n = field[i - 1, j]
            s = field[i + 1, j]
            e = field[i, j_east]
            w = field[i, j_west]

            lap[i, j] = n + s + e + w - 4.0 * c

    # Handle poles separately (copy from neighbors)
    for j in range(W):
        lap[0, j] = lap[1, j]
        lap[H - 1, j] = lap[H - 2, j]

    return lap


def compute_convective_precipitation(
    temperature: np.ndarray,
    humidity: np.ndarray,
    dt_days: float = 1.0,
    trigger_temp_c: float = 20.0,
    trigger_rh: float = 0.8,
    max_rate_mm_day: float = 10.0,
    surface_pressure_hpa: float = 1013.25,
) -> np.ndarray:
    """Enhanced convective precipitation with CAPE-like triggering (Phase 2).

    Simulates tropical thunderstorms and deep convection that occur when:
    1. Surface is warm (T > 20°C) - provides buoyancy
    2. Humidity is high (RH > 80%) - provides fuel

    This addresses the underprediction of ITCZ rainfall in the original model.

    Args:
        temperature: (H,W) Surface temperature [K]
        humidity: (H,W) Specific humidity [kg/kg]
        dt_days: Time step size [days]
        trigger_temp_c: Minimum temperature for convection [°C]
        trigger_rh: Minimum relative humidity for convection [0-1]
        max_rate_mm_day: Maximum convective precipitation rate [mm/day]

    Returns:
        (H,W) Convective precipitation contribution [mm/day]
    """
    # Convert to Celsius
    T_celsius = temperature - 273.15

    # Saturation humidity (Clausius-Clapeyron)
    T_c_clipped = np.clip(T_celsius, -60.0, 60.0)
    es = 6.112 * np.exp(17.67 * T_c_clipped / (T_c_clipped + 243.5))  # hPa
    qsat = np.clip(0.622 * es / surface_pressure_hpa, 1e-6, 0.035)  # kg/kg

    # Relative humidity
    rh = np.clip(humidity / (qsat + 1e-9), 0.0, 1.5)

    # Convective instability triggers
    # Warm trigger: 0 at trigger_temp_c, 1 at (trigger_temp_c + 10°C)
    warm_trigger = np.maximum(0.0, (T_celsius - trigger_temp_c) / 10.0)
    warm_trigger = np.clip(warm_trigger, 0.0, 1.0)

    # Moisture trigger: 0 at trigger_rh, 1 at 100% RH
    moist_trigger = np.maximum(0.0, (rh - trigger_rh) / (1.0 - trigger_rh))
    moist_trigger = np.clip(moist_trigger, 0.0, 1.0)

    # Convective precipitation rate (mm/day)
    # Both triggers must be satisfied (multiplicative)
    P_conv = max_rate_mm_day * warm_trigger * moist_trigger

    return P_conv.astype(np.float32)


@jit(nopython=True, parallel=True, cache=True)
def _upsample_bilinear_numba_kernel(
    stack: np.ndarray,
    y0: np.ndarray, y1: np.ndarray, wy: np.ndarray,
    x0: np.ndarray, x1: np.ndarray, wx: np.ndarray,
) -> np.ndarray:
    """Parallel bilinear interpolation: (N, Hc, Wc) → (N, H, W).

    Avoids the large intermediate arrays created by NumPy fancy indexing and
    parallelises over output rows so all CPU cores contribute.
    """
    N = stack.shape[0]
    H = len(wy)
    W = len(wx)
    out = np.zeros((N, H, W), dtype=np.float32)
    for i in prange(H):
        iy0 = y0[i]
        iy1 = y1[i]
        wi = wy[i]
        wi1 = 1.0 - wi
        for j in range(W):
            ix0 = x0[j]
            ix1 = x1[j]
            wj = wx[j]
            wj1 = 1.0 - wj
            for n in range(N):
                f00 = stack[n, iy0, ix0]
                f01 = stack[n, iy0, ix1]
                f10 = stack[n, iy1, ix0]
                f11 = stack[n, iy1, ix1]
                out[n, i, j] = (f00 * wj1 + f01 * wj) * wi1 + (f10 * wj1 + f11 * wj) * wi
    return out


@jit(nopython=True, parallel=True, cache=True)
def _moisture_convergence_numba(q: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute moisture flux convergence: -∇·(q·V).

    Returns convergence field (positive = moisture converging).
    """
    H, W = q.shape
    conv = np.zeros_like(q)

    for i in prange(1, H - 1):
        for j in range(W):
            j_east = (j + 1) % W
            j_west = (j - 1 + W) % W

            # Moisture flux
            flux_x_here = q[i, j] * u[i, j]
            flux_y_here = q[i, j] * v[i, j]

            # Central differences. Row index increases SOUTHWARD while v is
            # positive NORTHWARD, so the physical northward flux derivative is
            # the NEGATIVE of the along-index derivative: ∂F/∂y_north = -∂F/∂i.
            d_flux_x = 0.5 * (q[i, j_east] * u[i, j_east] - q[i, j_west] * u[i, j_west])
            d_flux_y = -0.5 * (q[i + 1, j] * v[i + 1, j] - q[i - 1, j] * v[i - 1, j])

            # Convergence (negative divergence)
            conv[i, j] = -(d_flux_x + d_flux_y)

            # Clip to positive (only interested in convergence)
            if conv[i, j] < 0.0:
                conv[i, j] = 0.0

    return conv


def flux_divergence_spherical(
    q: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    lat_rad: np.ndarray,
    *,
    radius_m: float = 6.371e6,
    cos_floor: float = 1e-3,
) -> np.ndarray:
    """Signed spherical divergence of the moisture flux ``q * V`` [kg/kg per second].

    Implements the true spherical form

        div(F) = 1/(a cos(phi)) * [ dFx/dlambda + d(Fy cos(phi))/dphi ]

    which `_moisture_convergence_numba` omits entirely: that kernel takes raw
    index differences, so the zonal term is under-weighted by 1/cos(phi)
    (negligible in the tropics, x2 at 60 deg, x3.9 at 75 deg) and the meridional
    term is missing the cos(phi) flux weighting that accounts for converging
    meridians. It also skips both pole rows; this function includes them via
    one-sided differences.

    Sign conventions, matching the rest of this module: `u` is eastward-positive,
    `v` is northward-positive, and the row index increases SOUTHWARD, so
    dphi/di = -pi/H.

    Returns *signed* divergence (positive = diverging). Callers wanting the
    convergence driver should take ``clip(-result, 0, None)``, mirroring
    `_moisture_convergence_numba`'s built-in clip.

    Verified against closed-form cases in
    `testing/test_spherical_metric.py` (solid-body rotation, a zonally-varying
    zonal flow whose divergence scales as 1/cos(phi), and a uniform meridional
    flow whose divergence is -V*tan(phi)/a).
    """
    H, W = q.shape
    cos_phi = np.maximum(np.cos(np.asarray(lat_rad, dtype=np.float64)), cos_floor)
    if cos_phi.ndim == 1:
        cos_phi = cos_phi[:, None]

    Fx = np.asarray(q, dtype=np.float64) * np.asarray(u, dtype=np.float64)
    Fy = np.asarray(q, dtype=np.float64) * np.asarray(v, dtype=np.float64)

    # Zonal term: periodic central difference in longitude.
    dlam = 2.0 * np.pi / W
    dFx_dlam = (np.roll(Fx, -1, axis=1) - np.roll(Fx, 1, axis=1)) / (2.0 * dlam)

    # Meridional term: d(Fy cos phi)/dphi. Central in the interior, one-sided at
    # the poles so rows 0 and H-1 are real values rather than identically zero.
    G = Fy * cos_phi
    dG_di = np.empty_like(G)
    dG_di[1:-1] = 0.5 * (G[2:] - G[:-2])
    dG_di[0] = G[1] - G[0]
    dG_di[-1] = G[-1] - G[-2]
    dphi_di = -np.pi / H
    dG_dphi = dG_di / dphi_di

    return (dFx_dlam + dG_dphi) / (float(radius_m) * cos_phi)


def _normalize_positive_driver(field: np.ndarray) -> np.ndarray:
    """Normalize a non-negative diagnostic without imposing an SI-scale floor.

    Spherical divergence is expressed per second and naturally has magnitudes
    near 1e-8. Adding the legacy dimensionless ``1e-6`` epsilon to its mean
    suppresses the signal by orders of magnitude. Exact-zero fields remain
    zero; every nonzero field is normalized by its own mean, making the
    diagnostic invariant to the units used by the derivative operator.
    """
    values = np.maximum(np.asarray(field, dtype=np.float64), 0.0)
    mean_value = float(np.mean(values, dtype=np.float64))
    if not np.isfinite(mean_value) or mean_value <= np.finfo(np.float64).tiny:
        return np.zeros_like(values, dtype=np.float32)
    return (values / mean_value).astype(np.float32)


def _streamfunction_from_vorticity(omega: np.ndarray) -> np.ndarray:
    H, W = omega.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(H)
    kx = 2.0 * np.pi * np.fft.rfftfreq(W)
    K2 = ky[:, None] ** 2 + kx[None, :] ** 2
    omega_hat = np.fft.rfft2(omega)
    psi_hat = np.zeros_like(omega_hat)
    mask = K2 > 1e-9
    psi_hat[mask] = -omega_hat[mask] / K2[mask]
    psi_hat[0, 0] = 0.0
    psi = np.fft.irfft2(psi_hat, s=omega.shape)
    return psi.astype(np.float32)


def _semi_lagrangian_departure(
    u: np.ndarray,
    v: np.ndarray,
    dt_seconds: float,
    dx_meters: np.ndarray,
    dy_meters: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-trajectory departure points shared by all semi-Lagrangian advection.

    Given a velocity field and an elapsed time, returns the (y, x) grid
    coordinates each cell's air parcel arrived *from* -- the same lookup
    whether the field being advected is the wind itself (`u`/`v`) or a
    passive scalar carried by that wind (e.g. humidity). Extracted out of
    `_advect_wind_semi_lagrangian` so `_advect_scalar_semi_lagrangian` doesn't
    duplicate this math.

    Args:
        u: (H,W) Eastward wind [m/s]
        v: (H,W) Northward wind [m/s]
        dt_seconds: Time step [seconds]
        dx_meters: (H,W) Grid spacing in x-direction [meters]
        dy_meters: Grid spacing in y-direction [meters]

    Returns:
        (y_departure, x_departure) grid-index coordinates for map_coordinates.
    """
    H, W = u.shape

    # Current grid coordinates (physical indices) — static per shape, cached
    if _MGRID_CACHE["key"] != (H, W):
        _MGRID_CACHE.update({"key": (H, W), "yx": np.mgrid[0:H, 0:W]})
    y_grid, x_grid = _MGRID_CACHE["yx"]

    # Backward trajectory: where did the air parcel come from?
    # dx_cells = (u * dt) / dx_meters  (convert m/s to grid cells)
    # Handle varying dx (smaller near poles)
    dx_cells = (u * dt_seconds) / (dx_meters + 1e-3)
    dy_cells = (v * dt_seconds) / dy_meters

    # Departure points (where air came from)
    x_departure = x_grid - dx_cells
    y_departure = y_grid - dy_cells

    # Periodic boundary in longitude (wraps around)
    x_departure = np.mod(x_departure, W)

    # Wall boundary in latitude (clamp at poles)
    y_departure = np.clip(y_departure, 0, H - 1)

    return y_departure, x_departure


def _advect_wind_semi_lagrangian(
    u: np.ndarray,
    v: np.ndarray,
    dt_seconds: float,
    dx_meters: np.ndarray,
    dy_meters: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Semi-Lagrangian wind advection (unconditionally stable, Phase 4).

    Instead of forward Euler (CFL-limited):
        u(x, t+dt) = u(x,t) + du/dt

    Use backward trajectory:
        u(x, t+dt) = u(x - V·dt, t)

    This removes the CFL constraint, allowing arbitrary timesteps without instability.

    Args:
        u: (H,W) Eastward wind [m/s]
        v: (H,W) Northward wind [m/s]
        dt_seconds: Time step [seconds]
        dx_meters: (H,W) Grid spacing in x-direction [meters]
        dy_meters: Grid spacing in y-direction [meters]

    Returns:
        (u_new, v_new) advected wind fields
    """
    map_coordinates = _scipy_map_coordinates
    if map_coordinates is None:
        from scipy.ndimage import map_coordinates  # last-resort fallback

    y_departure, x_departure = _semi_lagrangian_departure(u, v, dt_seconds, dx_meters, dy_meters)

    # Interpolate u, v at departure points (bilinear interpolation)
    u_new = map_coordinates(u, [y_departure, x_departure], order=1, mode='wrap')
    v_new = map_coordinates(v, [y_departure, x_departure], order=1, mode='wrap')

    return u_new.astype(np.float32), v_new.astype(np.float32)


def _advect_scalar_semi_lagrangian(
    field: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dt_seconds: float,
    dx_meters: np.ndarray,
    dy_meters: float,
) -> np.ndarray:
    """Semi-Lagrangian advection of a passive scalar (e.g. humidity) by wind (u, v).

    Same backward-trajectory, unconditionally-stable scheme as
    `_advect_wind_semi_lagrangian`, but samples an arbitrary scalar field at
    the departure point instead of u/v themselves. This is what makes
    moisture transport distance scale correctly with wind speed and dt_days
    (a genuine Courant number) instead of the old fixed ~3-cell donor-cell
    blend used previously in `generate_precipitation` (see
    known-physics-gaps.md / ROADMAP.md's "CFL-linked humidity advection"
    item) -- at typical mid-latitude wind speeds a full day's transport spans
    many tens of grid cells, and a fixed small-blend loop simply couldn't
    represent that without an infeasible number of substeps at MONTHLY/ANNUAL
    dt (the same CFL problem this pattern already solved for wind).

    Args:
        field: (H,W) scalar field to advect (e.g. specific humidity)
        u: (H,W) Eastward wind [m/s]
        v: (H,W) Northward wind [m/s]
        dt_seconds: Time step [seconds]
        dx_meters: (H,W) Grid spacing in x-direction [meters]
        dy_meters: Grid spacing in y-direction [meters]

    Returns:
        Advected field, same shape/dtype (float32) as input.
    """
    map_coordinates = _scipy_map_coordinates
    if map_coordinates is None:
        from scipy.ndimage import map_coordinates  # last-resort fallback

    y_departure, x_departure = _semi_lagrangian_departure(u, v, dt_seconds, dx_meters, dy_meters)
    field_new = map_coordinates(field, [y_departure, x_departure], order=1, mode='wrap')
    return field_new.astype(np.float32)


def _smear_along_wind(
    field: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dx_meters: np.ndarray,
    dy_meters: float,
    upwind_km: float,
    downwind_km: float,
    *,
    max_samples: int = 24,
) -> np.ndarray:
    """Give a pointwise terrain signal a directional footprint along the wind.

    Written for the orographic-uplift term, whose defect is *shape*, not
    magnitude: on a real DEM the upslope product is a 1-2 cell spike sitting on
    the crest, whereas real orographic precipitation covers a broad windward
    flank. Air begins ascending well upstream of a barrier (upstream blocking
    and deceleration), and the condensate it produces is carried some distance
    downstream before it reaches the ground. A box mean over 4-6 cells -- the
    only measure comparable to Earth's published windward/leeward ratios --
    therefore dilutes a model signal that is correct exactly where it exists and
    absent everywhere else in the box. See ACCURACY_AUDIT.md A5-OROG, which
    names this as the next lever after establishing that all four pointwise
    ceilings in the pipeline are exhausted.

    ``upwind_km`` is the e-folding length of the **upstream** footprint: a cell
    samples points *downwind* of itself, which is what places a crest's uplift
    onto the flank ahead of it. ``downwind_km`` is the spillover in the other
    direction (a cell samples points *upwind* of itself), representing
    hydrometeor drift past the crest. Both are physical distances, not cell
    counts, so the mechanism is resolution-invariant -- the fixed 20-cell
    monsoon mask that meant 7 deg at 1024 columns and 56 deg at 128 is the
    failure mode being avoided here.

    The result is a weighted average whose weights sum to 1 (the cell itself
    carries weight 1, each sample ``exp(-distance / e-folding length)``), so a
    spatially uniform field is returned unchanged and total signal is broadly
    conserved while its distribution broadens.
    """
    if upwind_km <= 0.0 and downwind_km <= 0.0:
        return field

    map_coordinates = _scipy_map_coordinates
    if map_coordinates is None:
        from scipy.ndimage import map_coordinates  # last-resort fallback

    H, W = field.shape
    if _MGRID_CACHE["key"] != (H, W):
        _MGRID_CACHE.update({"key": (H, W), "yx": np.mgrid[0:H, 0:W]})
    y_grid, x_grid = _MGRID_CACHE["yx"]

    # Wind *direction* only -- the footprint length is a prescribed physical
    # scale, so a stronger wind must not also stretch it (that would make the
    # mechanism a second, uncalibrated function of wind speed on top of the
    # upslope product `gx*u + gy*v`, which already carries the speed dependence).
    speed = np.sqrt(u * u + v * v)
    u_hat = u / (speed + 1e-6)
    v_hat = v / (speed + 1e-6)

    # Sampling cadence: fine enough not to step over intervening cells, capped
    # so a long footprint on a fine grid stays affordable. `dy_meters` is the
    # meridional spacing and is the resolution-invariant cell scale here (`dx`
    # collapses toward the poles, where these ranges do not sit).
    cell_km = float(dy_meters) / 1000.0
    accumulated = field.astype(np.float64, copy=True)
    total_weight = 1.0  # the cell's own contribution

    for length_km, sign in ((float(upwind_km), 1.0), (float(downwind_km), -1.0)):
        if length_km <= 0.0:
            continue
        reach_km = 3.0 * length_km  # ~95% of an exponential's mass
        n_steps = int(np.clip(np.ceil(reach_km / max(cell_km, 1e-6)), 1, max_samples))
        step_km = reach_km / n_steps
        for step in range(1, n_steps + 1):
            distance_m = 1000.0 * step_km * step
            x_sample = x_grid + sign * u_hat * distance_m / (dx_meters + 1e-3)
            y_sample = y_grid - sign * v_hat * distance_m / dy_meters
            x_sample = np.mod(x_sample, W)
            y_sample = np.clip(y_sample, 0, H - 1)
            weight = float(np.exp(-step_km * step / length_km))
            accumulated += weight * map_coordinates(
                field, [y_sample, x_sample], order=1, mode="wrap"
            )
            total_weight += weight

    return (accumulated / total_weight).astype(field.dtype, copy=False)


# Scale [K] at which the SST -> land coupling's response saturates.  Both halves
# of that mechanism use `tanh(anomaly / this)`, so their strength knobs mean
# "largest fractional change" rather than "change per kelvin, unbounded".  Fixed
# rather than a PlanetParams field: it describes how far an ocean temperature
# anomaly can plausibly shift the boundary layer above it, which is not a
# per-planet tuning freedom -- the strength knobs already span that.
_SST_COUPLING_REFERENCE_K = 2.0


def _upwind_sst_anomaly(
    temperature: np.ndarray,
    sea_f: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dx_meters: np.ndarray,
    dy_meters: float,
    reach_km: float,
) -> np.ndarray:
    """Ocean-fraction-weighted SST anomaly of the water *upwind* of each cell [K].

    Audit D3/D5.  D5 built real eastern-boundary upwelling (Benguela -0.54 K,
    Humboldt -0.25 K vs gyres-off) and then measured that it changes adjacent
    land precipitation by nothing at all, at any gyre strength -- settling that
    per-cell SST anomalies do not reach land climate in this model's atmosphere.
    This is the missing link, built as its own field so the coupling can be
    tested independently of whatever produces the anomaly.

    The anomaly is taken against each row's **own ocean mean**, so it isolates
    basin/current structure from the meridional temperature gradient: a
    cold-current cell is cold *for its latitude*, which is the physically
    meaningful quantity and the only one a land cell at the same latitude can
    respond to differentially.

    ``reach_km`` is an e-folding fetch, not a cell count -- the
    resolution-invariance trap this file has already had to fix twice (the
    monsoon inland mask's 20 cells, `_maritime_proximity`'s pass cap).
    ``_smear_along_wind``'s second length samples *upwind* of each cell, which
    is the direction air actually arrives from; its weights sum to 1 including
    the cell's own (zero over land) contribution, so the returned field is the
    mean upwind-ocean anomaly scaled by how oceanic that fetch is. It therefore
    decays to zero inland on its own, with no separate coastal mask -- unlike
    `coastal_upwelling_fog_strength`'s hand-built two-cell decay, which is the
    diagnostic proxy this replaces.
    """
    # float64 for the row reduction: this is a small difference between large
    # absolute temperatures, so accumulating a full row in float32 costs real
    # precision in exactly the quantity being extracted. Same reason
    # `_land_pp_sum` below reduces in float64.
    per_row = sea_f.sum(axis=1, keepdims=True, dtype=np.float64)
    row_mean = np.divide(
        (temperature * sea_f).sum(axis=1, keepdims=True, dtype=np.float64),
        np.maximum(per_row, 1.0),
    )
    anomaly = ((temperature - row_mean) * sea_f).astype(np.float32, copy=False)
    return _smear_along_wind(
        anomaly, u, v, dx_meters, dy_meters, 0.0, float(reach_km)
    )


@jit(nopython=True, parallel=True, cache=True)
def _advect_scalar_cfl_step_numba(q: np.ndarray, u: np.ndarray, v: np.ndarray,
                                  u_cfl: np.ndarray, v_cfl: np.ndarray) -> np.ndarray:
    """One CFL-safe upwind advection step of `dq/dt = -u*dq/dx - v*dq/dy`.

    Same donor-cell structure as `_advect_humidity_numba` (upwind donor
    selected by each cell's own wind sign, sequential x-then-y blend), but
    `u_cfl`/`v_cfl` are real Courant numbers (`|wind|*dt_sub/spacing`, clipped
    to a stable range by the caller) instead of that function's fixed
    empirical blend fractions -- this is what lets the *total* displacement
    over many calls scale correctly with `wind_speed * dt_seconds`.

    Deliberately the *advection* (material-derivative) equation, not a flux-
    divergence/conservation-law form: specific humidity here is a mixing
    ratio (moisture per unit air mass) with no companion air-density
    continuity equation in this model, so a `-div(q*V)` formulation would add
    an unphysical `q*div(V)` reaction term that compounds exponentially over
    a multi-day integration under perfectly ordinary wind divergence (found
    directly while building this: a smooth, realistic ~9e-5/s divergence
    blew a bounded field up over a 5-day integration). Advection form has no
    such term -- it can only reshuffle/smooth the field, never blow it up.
    """
    H, W = q.shape
    q_out = np.empty_like(q)

    for i in prange(H):
        for j in range(W):
            j_east = (j + 1) % W
            j_west = (j - 1 + W) % W

            if u[i, j] >= 0:
                q_x = q[i, j_west]
            else:
                q_x = q[i, j_east]
            q_adv_x = q[i, j] + u_cfl[i, j] * (q_x - q[i, j])

            if i == 0 or i == H - 1:
                q_out[i, j] = q_adv_x
            else:
                if v[i, j] >= 0:
                    q_y = q[i + 1, j]  # Southward: donor is the south neighbor
                else:
                    q_y = q[i - 1, j]  # Northward: donor is the north neighbor
                q_out[i, j] = q_adv_x + v_cfl[i, j] * (q_y - q_adv_x)

    return q_out


def _advect_scalar_cfl_step(q: np.ndarray, u: np.ndarray, v: np.ndarray,
                           u_cfl: np.ndarray, v_cfl: np.ndarray) -> np.ndarray:
    """NumPy fallback for `_advect_scalar_cfl_step_numba` -- see its docstring."""
    east = np.roll(q, -1, axis=1)
    west = np.roll(q, 1, axis=1)
    row_south = np.concatenate([q[1:, :], q[-1:, :]], axis=0)
    row_north = np.concatenate([q[:1, :], q[:-1, :]], axis=0)

    adv_x = q + u_cfl * (np.where(u >= 0, west, east) - q)
    adv_xy = adv_x + v_cfl * (np.where(v >= 0, row_south, row_north) - adv_x)
    adv_xy[0, :] = adv_x[0, :]
    adv_xy[-1, :] = adv_x[-1, :]
    return adv_xy.astype(np.float32)


def _advect_scalar_flux_eulerian(
    field: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dt_seconds: float,
    dx_meters: np.ndarray,
    dy_meters: float,
    max_courant: float = 0.5,
) -> np.ndarray:
    """CFL-safe Eulerian upwind advection of a passive scalar (e.g. specific
    humidity) by wind (u, v), integrated over the full `dt_seconds` in many
    small substeps instead of `_advect_scalar_semi_lagrangian`'s single
    backward-trajectory jump.

    Fixes the mechanism that function used to implement: a single point
    sample covering the *entire* `dt_seconds` at once, which at real
    MONTHLY-mode substep dt (~7.6 days) and typical mid-latitude wind speed is
    a ~5000km jump -- nearly a quarter of the latitude circle. Measured
    directly (2026-07 moisture-advection-jump-dilution investigation, see
    project memory) to dilute even the *ocean* source cells (coastal RH
    100%->66% at `moisture_advection_scale` 0->0.7), not just fail to reach
    the continental interior, because a single huge jump samples an
    effectively uncorrelated point in the domain -- closer to diffusive
    homogenization toward the domain mean than believable local transport.

    This instead takes `n_sub` sequential upwind steps (see
    `_advect_scalar_cfl_step_numba`), each respecting `max_courant`, so no
    individual step moves moisture more than a bounded fraction of one grid
    cell -- the total displacement across all substeps still equals
    `wind_speed * dt_seconds` (a real Courant distance), it just accumulates
    via many small, well-behaved steps rather than one huge one.

    Cost scales with `max(|u|/dx, |v|/dy) * dt_seconds / max_courant` substeps
    -- at real MONTHLY-mode substep dt and this grid's resolution, that's on
    the order of several hundred to ~1500 substeps (each a cheap O(H*W) pass).
    Only paid when the caller actually enables the term this feeds
    (`PlanetParams.moisture_advection_scale`, default 0.0/never called in
    default runs).

    Args:
        field: (H,W) scalar field to advect (e.g. specific humidity)
        u: (H,W) Eastward wind [m/s]
        v: (H,W) Northward wind [m/s]
        dt_seconds: Time step [seconds]
        dx_meters: (H,W) Grid spacing in x-direction [meters]
        dy_meters: Grid spacing in y-direction [meters]
        max_courant: Per-substep, per-direction CFL bound (fraction of one
            grid cell moved per substep, each direction independently). 0.5
            keeps the upwind scheme comfortably stable and monotone (no new
            extrema) -- matches `_advect_humidity_numba`'s sequential x-then-y
            blend structure, where each direction's own convex-combination
            step just needs its own factor in [0, 1].

    Returns:
        Advected field, same shape/dtype (float32) as input.
    """
    dx_f = dx_meters.astype(np.float32, copy=False)
    courant_u = np.abs(u) / dx_f
    courant_v = np.abs(v) / float(dy_meters)
    max_c = float(max(courant_u.max(), courant_v.max()))
    if max_c <= 0.0:
        return field.astype(np.float32, copy=True)

    n_sub = max(1, int(np.ceil(max_c * float(dt_seconds) / max_courant)))
    dt_sub = float(dt_seconds) / n_sub
    u_cfl = np.clip(courant_u * dt_sub, 0.0, max_courant).astype(np.float32)
    v_cfl = np.clip(courant_v * dt_sub, 0.0, max_courant).astype(np.float32)

    q = field.astype(np.float32, copy=True)
    step = _advect_scalar_cfl_step_numba if NUMBA_AVAILABLE else _advect_scalar_cfl_step
    for _ in range(n_sub):
        q = step(q, u, v, u_cfl, v_cfl)
    return q.astype(np.float32, copy=False)


def evolve_wind(
    u: np.ndarray,
    v: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray | None,
    elevation: np.ndarray,
    dt_days: float = 1.0,
    damping: float = 0.25,
    pgf_temp_scale: float = 450.0,
    pgf_terrain_scale: float = 900.0,
    drag_base: float = 2.0e-7,
    drag_elev_scale: float = 6.0e-7,
    vmax_clip: float = 150.0,
    baroclinic_jet_amp: float = 0.0,
    baroclinic_mix: float = 0.0,
    cell_relax_days: float = 0.0,
    time_days: float | None = None,
    planet_params: PlanetParams | None = None,
    ice_cover: np.ndarray | None = None,
    ice_pressure_scale: float = 40.0,
    jet_index_nh: float = 0.0,
    jet_index_sh: float = 0.0,
    jet_block_nh: tuple[float, float, float] = (-1.0, 0.0, 0.0),
    jet_block_sh: tuple[float, float, float] = (-1.0, 0.0, 0.0),
    u_aloft: np.ndarray | None = None,
    v_aloft: np.ndarray | None = None,
    debug_fields: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve wind field using simplified primitive momentum equations.

    Equations:
    du/dt = - (u*du/dx + v*du/dy) + f*v - (1/rho)*dp/dx + F_x
    dv/dt = - (u*dv/dx + v*dv/dy) - f*u - (1/rho)*dp/dy + F_y

    Physics included:
    - Advection (self-transport)
    - Coriolis force (rotation)
    - Pressure Gradient Force (thermal + dynamic)
    - Friction (surface drag)
    - Jet stream dynamics (thermal wind balance)

    `jet_index_{nh,sh}` are the persistent per-hemisphere meander/waviness
    indices (atmosphere._update_jet_index): they shift the mid-lat jet's
    relaxation-target latitude/speed and the Rossby-wave amplitude below.
    `jet_block_{nh,sh}` are `(lon_deg, days_left, total_duration_days)`
    tuples describing an active blocking ridge (atmosphere._update_jet_blocking
    / _blocking_ridge_pressure_anomaly); `lon_deg == -1.0` means inactive.

    `u_aloft`/`v_aloft`, if provided, are the current upper-level (1.5-layer)
    prognostic wind field (see `evolve_wind_aloft`). When present, the
    baroclinic mixing term relaxes the surface wind toward this field on a
    per-cell, direction-sensitive basis instead of the old magnitude-only
    `|dT/dy|` proxy.

    `debug_fields`, if provided (an empty dict to populate), receives the
    per-term pressure-anomaly decomposition (`p_anom_thermal`, `p_anom_terrain`,
    `p_anom_ice`, `p_anom_synoptic` -- any not applicable this call are left
    out) *before* they're summed into `p_anom`. Diagnostic-only; no effect on
    the returned wind when omitted (default `None`). Used by
    `scripts/check_real_terrain_koppen.py --wind-diagnostics` to measure the
    real magnitude of each pressure-forcing term without duplicating this
    function's formulas in a separate script.
    """
    H, W = u.shape
    dt_total = dt_days * 86400.0  # seconds
    pp = planet_params or EARTH

    # Static per-(shape, planet) grids — cached (see _wind_static_grids).
    # Equatorial damping window: in a single-layer model, PGF can over-accelerate
    # winds where f≈0; boost drag within ~±12° to recover calmer doldrums.
    lat_2d, dx, dy, f, eq_window, _lon_1d_cached = _wind_static_grids(H, W, pp)

    # With the rotation-matrix Coriolis (unconditionally stable), 24 sub-steps
    # are no longer needed for stability.  8 sub-steps keep PGF and friction
    # accurate while halving the per-day wind computation cost.
    n_steps = 8
    dt_sub = dt_total / n_steps
    
    # Gradients dx, dy
    # (dx,dy already computed above)
    
    rho = pp.reference_air_density

    u_curr, v_curr = u.copy(), v.copy()
    
    # Pre-calculate PGF (constant over the day)
    if pressure is None:
        # P ~ P0 * exp(-z/H) * (T0/T)^g/R... simplified:
        # NOTE: Keep this in Pa (not hPa). Scale is intentionally "synoptic-ish":
        # order ~5-10 hPa swings across large temperature gradients.
        p_thermal = -float(pgf_temp_scale) * ((temperature - 273.15) / 30.0)  # Pa anomaly
        p_anom = p_thermal
        if debug_fields is not None:
            debug_fields["p_anom_thermal"] = p_thermal
        if elevation is not None:
             # Terrain effect: flow around obstacles, high pressure wedge
             p_terrain = float(pgf_terrain_scale) * elevation
             p_anom = p_anom + p_terrain
             if debug_fields is not None:
                 debug_fields["p_anom_terrain"] = p_terrain
        if ice_cover is not None and float(ice_pressure_scale) != 0.0:
            # Sea ice → wind/pressure feedback. Physically: ice-covered surfaces
            # radiatively cool efficiently, reinforcing a shallow cold-air dome
            # (katabatic outflow, polar-high intensification) beyond what the
            # smoothed T→pressure relationship alone captures.
            #
            # CAUTION: a nearly identical *flat, land-based* pressure contrast
            # was tried and reverted just above (see the NOTE below) because it
            # caused a runaway ice-albedo feedback loop (SH pole → 201 K).
            # ice_pressure_scale defaults to 40 Pa at full ice cover — well
            # below both the terrain term's typical range and the reverted
            # 150 Pa land-sea contrast — specifically to avoid reproducing
            # that failure mode. Dynamically coupled to the ice model (grows/
            # shrinks with `ice_cover`) rather than a static continent-scale
            # bonus, which should make it self-limiting rather than persistent.
            p_ice = float(ice_pressure_scale) * np.clip(ice_cover, 0.0, 1.0)
            p_anom = p_anom + p_ice
            if debug_fields is not None:
                debug_fields["p_anom_ice"] = p_ice
    else:
        # Copy: the wave/storm/blocking terms below accumulate into p_anom, and
        # aliasing the caller's array would silently mutate their buffer.
        p_anom = np.array(pressure, dtype=np.float32, copy=True)

    # Synoptic-scale planetary waves + storms/trade-waves/blocking ridges: see
    # _synoptic_wave_pressure_anomaly (shared with generate_wind_field's
    # diagnostic wind so both paths carry the same storm/jet signal).
    if time_days is not None:
        p_synoptic = _synoptic_wave_pressure_anomaly(
            lat_2d, _lon_1d_cached, time_days,
            jet_index_nh, jet_index_sh, jet_block_nh, jet_block_sh, pp,
        )
        p_anom = p_anom + p_synoptic
        if debug_fields is not None:
            debug_fields["p_anom_synoptic"] = p_synoptic

    # NOTE: A constant land-sea pressure contrast was tried here but removed.
    # Antarctica (~all land) got a permanent +150 Pa high, driving persistent cold-air
    # outflow and triggering runaway SH sea-ice cooling (SH pole → 201 K).
    # The land-sea contrast is already partially encoded in the temperature field T via
    # land/ocean differential heating in simulate.py's _evolve_temperature.

    # Smooth (2026-07-28, tropical-speckle-fix): `generate_wind_field`'s
    # diagnostic path already does `pressure + 0.2*laplacian(pressure)` before
    # differentiating it, but this prognostic path differentiated `p_anom`
    # (which includes raw/unsmoothed `p_terrain = pgf_terrain_scale *
    # elevation`) directly via central differences with no diffusion term.
    # That was dormant while MONTHLY/ANNUAL used the diagnostic path, but
    # became a real, sustained (not just transient) source of grid-scale
    # precip roughness -- salt-and-pepper desert-classified speckle inside the
    # tropical rainforest belt (Amazon/Congo/Indonesia) -- once
    # `wind_prognostic_substep_days` defaulting to 1.0 routed MONTHLY/ANNUAL
    # through this prognostic solver instead.
    p_anom = p_anom + 0.2 * _laplacian(p_anom)

    dp_dx = _ddx_periodic(p_anom) / (dx + 1e-3)
    # Axis 0 is north→south (index increases southward), so physical northward gradient is negated.
    dp_dy = -np.gradient(p_anom, axis=0) / dy

    pgf_u = -1.0/rho * dp_dx
    pgf_v = -1.0/rho * dp_dy

    # --- Baroclinic / eddy-driven mid-lat westerly tendency ---
    # Real per-cell vertical momentum mixing toward the upper-level (1.5-layer)
    # wind field (evolve_wind_aloft), replacing the old magnitude-only
    # `baroclinic_jet_amp * jet_window * |dT/dy|` proxy. Direction-sensitive
    # (carries the actual sign of the aloft-surface wind difference) where the
    # old hack wasn't -- `|dT/dy|` discarded sign entirely, so it could never
    # distinguish a cold-to-warm gradient from a warm-to-cold one.
    b_amp = float(baroclinic_jet_amp)
    b_mix = float(baroclinic_mix)
    mix_active = b_amp != 0.0 and b_mix > 0.0 and u_aloft is not None and v_aloft is not None
    if mix_active:
        abs_deg_1d = np.rad2deg(np.abs(lat_2d[:, 0])).astype(np.float32, copy=False)  # (H,)
        jet_window_mix = np.exp(-((abs_deg_1d - 45.0) / 12.0) ** 2).astype(np.float32, copy=False)[:, None]  # (H,1)
        u_aloft_arr = u_aloft.astype(np.float32, copy=False)
        v_aloft_arr = v_aloft.astype(np.float32, copy=False)

    # --- 3-cell surface tendency (Hadley/Ferrel/Polar) ---
    # A single-layer model won't spontaneously generate the full overturning circulation.
    # This optional, weak relaxation nudges zonal-mean (u,v) toward an Earth-like 3-cell
    # surface signature: trades (easterly), mid-lat westerlies, polar easterlies; plus
    # equatorward/poleward v bands by hemisphere.
    tau_cell = float(cell_relax_days)
    if tau_cell > 0.0:
        abs_deg_1d = np.rad2deg(np.abs(lat_2d[:, 0])).astype(np.float32, copy=False)  # (H,)
        sign_lat = np.sign(lat_2d[:, 0]).astype(np.float32, copy=False)  # +N, -S
        # Broaden the windows + reduce amplitudes to avoid razor-thin zonal bands.
        w_trade = np.exp(-((abs_deg_1d - HADLEY_CELL_CENTER_DEG) / HADLEY_CELL_WIDTH_DEG) ** 2).astype(np.float32, copy=False)
        # Mid-lat jet window is split per hemisphere so the persistent meander index
        # (atmosphere._update_jet_index) can shift each hemisphere's jet core
        # latitude/speed independently -- this is what makes the relaxation target
        # itself meander over time instead of sitting at a fixed 48 degrees forever.
        jet_center_nh = MID_LAT_JET_CENTER_DEG + float(pp.jet_lat_shift_per_index) * float(jet_index_nh)
        jet_center_sh = MID_LAT_JET_CENTER_DEG + float(pp.jet_lat_shift_per_index) * float(jet_index_sh)
        w_mid_nh = np.where(
            sign_lat >= 0.0,
            np.exp(-((abs_deg_1d - jet_center_nh) / MID_LAT_JET_WIDTH_DEG) ** 2),
            0.0,
        ).astype(np.float32, copy=False)
        w_mid_sh = np.where(
            sign_lat < 0.0,
            np.exp(-((abs_deg_1d - jet_center_sh) / MID_LAT_JET_WIDTH_DEG) ** 2),
            0.0,
        ).astype(np.float32, copy=False)
        w_mid = w_mid_nh + w_mid_sh
        w_polar = np.exp(-((abs_deg_1d - POLAR_CELL_CENTER_DEG) / POLAR_CELL_WIDTH_DEG) ** 2).astype(np.float32, copy=False)
        speed_nh = 1.0 + float(pp.jet_speed_scale_per_index) * float(jet_index_nh)
        speed_sh = 1.0 + float(pp.jet_speed_scale_per_index) * float(jet_index_sh)
        u_mid = U_TARGET_MIDLAT * (speed_nh * w_mid_nh + speed_sh * w_mid_sh)
        # Optimized circulation targets for realistic Earth-like winds with sub-stepping
        # Trade winds (easterlies), stronger mid-lat westerlies, weaker polar easterlies.
        u_target = (U_TARGET_TRADE * w_trade + u_mid + U_TARGET_POLAR * w_polar).astype(np.float32, copy=False)
        # v_target: Hadley (equatorward), Ferrel (poleward), Polar (equatorward), by hemisphere.
        # Strengthen Ferrel return flow while reducing polar leakage into the 30-60° band.
        # The meridional (Ferrel) lobe uses its OWN centre, decoupled from the
        # westerly jet's, exactly as in generate_wind_field -- see
        # PlanetParams.ferrel_v_centre_deg. Both paths must use the same centre or
        # DAILY/WEEKLY (this prognostic solver) and MONTHLY/ANNUAL (the diagnostic
        # one) would place the subtropical dry belt at different latitudes, which
        # is the speed-inconsistency bug class that speed-switch-biome-consistency
        # worked to close. The meander shift (jet_lat_shift_per_index) is applied
        # to both centres identically so the cells still migrate together.
        _v_centre = float(getattr(pp, "ferrel_v_centre_deg", MID_LAT_JET_CENTER_DEG))
        if _v_centre == MID_LAT_JET_CENTER_DEG:
            w_mid_v_nh, w_mid_v_sh = w_mid_nh, w_mid_sh
        else:
            _vc_nh = _v_centre + float(pp.jet_lat_shift_per_index) * float(jet_index_nh)
            _vc_sh = _v_centre + float(pp.jet_lat_shift_per_index) * float(jet_index_sh)
            w_mid_v_nh = np.where(
                sign_lat >= 0.0,
                np.exp(-((abs_deg_1d - _vc_nh) / MID_LAT_JET_WIDTH_DEG) ** 2),
                0.0,
            ).astype(np.float32, copy=False)
            w_mid_v_sh = np.where(
                sign_lat < 0.0,
                np.exp(-((abs_deg_1d - _vc_sh) / MID_LAT_JET_WIDTH_DEG) ** 2),
                0.0,
            ).astype(np.float32, copy=False)
        # Combined window for the v-relaxation *strength* below (a_v_row), so the
        # strength weighting peaks at the same latitude as v_target itself rather
        # than at the jet's centre -- see the a_v_row comment for the mismatch
        # this closes. Bit-identical to w_mid when ferrel_v_centre_deg == 48.
        w_mid_v = w_mid_v_nh + w_mid_v_sh
        v_mid = V_TARGET_MIDLAT * (speed_nh * w_mid_v_nh + speed_sh * w_mid_v_sh)
        v_target = (V_TARGET_TRADE * w_trade + v_mid + V_TARGET_POLAR * w_polar).astype(np.float32, copy=False) * sign_lat
        # Remove the equator sign ambiguity (sign(0)=0) so the equator stays calm.
        v_target = np.where(np.abs(lat_2d[:, 0]) < np.deg2rad(2.0), 0.0, v_target).astype(np.float32, copy=False)
        # PlanetParams.ferrel_v_land_shift_deg: blend v_target toward a further
        # -shifted, land-only centre, mirroring generate_wind_field's identical
        # mechanism (see that function and the field's own docstring for why:
        # this zonal-mean nudge applies the same delta at every longitude in a
        # row, so a shared centre can't fix under-wet continental interiors
        # without also over-wetting the already-good ocean at the same
        # latitude). 0.0 (default) skips this entirely -- exact no-op,
        # bit-identical to the shared-centre v_target above. The a_v_row
        # relaxation *strength* below deliberately still uses the shared-centre
        # w_mid_v, not a land-blended version -- the same reviewability
        # trade-off already accepted for the jet-vs-centre strength mismatch
        # this file's a_v_row comment documents.
        _v_land_shift = float(getattr(pp, "ferrel_v_land_shift_deg", 0.0))
        if _v_land_shift == 0.0:
            v_target_2d = v_target[:, None]
        else:
            _, land_mask_full = get_masks(elevation)
            land_f_full = land_mask_full.astype(np.float32)
            _land_shift_taper = np.clip((55.0 - abs_deg_1d) / 10.0, 0.0, 1.0)
            _land_shift_taper = (
                _land_shift_taper * _land_shift_taper * (3.0 - 2.0 * _land_shift_taper)
            ).astype(np.float32)
            land_shift_f = land_f_full * _land_shift_taper[:, None]
            _vc_land_nh = _v_centre + _v_land_shift + float(pp.jet_lat_shift_per_index) * float(jet_index_nh)
            _vc_land_sh = _v_centre + _v_land_shift + float(pp.jet_lat_shift_per_index) * float(jet_index_sh)
            w_mid_v_land_nh = np.where(
                sign_lat >= 0.0,
                np.exp(-((abs_deg_1d - _vc_land_nh) / MID_LAT_JET_WIDTH_DEG) ** 2),
                0.0,
            ).astype(np.float32, copy=False)
            w_mid_v_land_sh = np.where(
                sign_lat < 0.0,
                np.exp(-((abs_deg_1d - _vc_land_sh) / MID_LAT_JET_WIDTH_DEG) ** 2),
                0.0,
            ).astype(np.float32, copy=False)
            v_mid_land = V_TARGET_MIDLAT * (speed_nh * w_mid_v_land_nh + speed_sh * w_mid_v_land_sh)
            v_target_land = (V_TARGET_TRADE * w_trade + v_mid_land + V_TARGET_POLAR * w_polar).astype(np.float32, copy=False) * sign_lat
            v_target_land = np.where(np.abs(lat_2d[:, 0]) < np.deg2rad(2.0), 0.0, v_target_land).astype(np.float32, copy=False)
            v_target_2d = v_target[:, None] * (1.0 - land_shift_f) + v_target_land[:, None] * land_shift_f
        k_cell = 1.0 / (tau_cell * 86400.0)
    
    for _ in range(n_steps):
        # 1. Semi-Lagrangian advection (unconditionally stable)
        u_adv, v_adv = _advect_wind_semi_lagrangian(u_curr, v_curr, dt_sub, dx, dy)

        # 2. Coriolis — exact rotation matrix (operator splitting).
        #    R(θ) = [cos θ,  sin θ; -sin θ, cos θ]   with θ = f · dt_sub
        #    This is the exact solution to du/dt = f·v, dv/dt = -f·u and is
        #    unconditionally stable for any dt, unlike the first-order Euler
        #    tendency (du = f·v·dt) that required 24 sub-steps.
        theta = f * dt_sub           # (H, W), radians of rotation per sub-step
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        u_rot = cos_t * u_adv + sin_t * v_adv
        v_rot = -sin_t * u_adv + cos_t * v_adv

        # 3. PGF + friction evaluated at the Coriolis-rotated state
        if NUMBA_AVAILABLE and elevation is not None:
            du_fric, dv_fric = _friction_kernel_numba(
                u_rot, v_rot, elevation.astype(np.float32),
                float(drag_base), float(drag_elev_scale),
                eq_window.astype(np.float32), 1.0
            )
            du = (pgf_u + du_fric) * dt_sub
            dv = (pgf_v + dv_fric) * dt_sub
        else:
            drag = float(drag_base)
            if elevation is not None:
                drag += float(drag_elev_scale) * elevation
            drag = drag + (2.0e-6 * eq_window)
            speed_rot = np.hypot(u_rot, v_rot)
            friction_u = -drag * u_rot * speed_rot
            friction_v = -drag * v_rot * speed_rot
            du = (pgf_u + friction_u) * dt_sub
            dv = (pgf_v + friction_v) * dt_sub

        # 4. Baroclinic jet mixing: per-cell relaxation toward the upper-level
        # wind (vertical momentum coupling), not a zonal-mean-only nudge.
        if mix_active:
            k = b_amp / (b_mix * 86400.0)
            du = du + (u_aloft_arr - u_rot) * jet_window_mix * k * dt_sub
            dv = dv + (v_aloft_arr - v_rot) * jet_window_mix * k * dt_sub

        u_curr = u_rot + du * damping
        v_curr = v_rot + dv * damping

        # Relax zonal-mean toward 3-cell surface targets (apply directly so it isn't
        # weakened by the global `damping` factor above).
        if tau_cell > 0.0:
            a = float(np.clip(dt_sub * k_cell, 0.0, 1.0))
            u_zm = np.mean(u_curr, axis=1, keepdims=True)  # (H,1)
            v_zm = np.mean(v_curr, axis=1, keepdims=True)  # (H,1)
            # Pull u toward the target across all latitudes, with stronger forcing where needed
            u_t = np.clip(u_target, -15.0, 15.0).astype(np.float32, copy=False)  # Allow stronger targets
            # Trade relaxation re-enabled (2×): without this, trades rely only on PGF which is
            # weak in the tropics (small T gradient), leaving trades at ~0.7 m/s vs target -5 m/s.
            # Mid-lat (5×) and polar (10×) remain unchanged.
            a_u_row = np.clip(a * (1.0 + 2.5 * w_trade[:, None] + 9.0 * w_mid[:, None] + 2.5 * w_polar[:, None]), 0.0, 1.0).astype(np.float32, copy=False)
            u_curr = u_curr + (u_t[:, None] - u_zm) * a_u_row
            # Relax v with differentiated mid-lat / polar strength.
            # Mid-lat (6×): enough freedom for longitudinal variability in mid-lat eddies.
            # Trade (2×): matches u-relaxation strength so PGF can't hold v poleward in tropics.
            # Polar (25×): strong constraint prevents unrestricted poleward/equatorward
            # surges that caused extreme SH cooling when v-relaxation was too loose (8×).
            # With tau_cell=3d, a≈0.042: trade → 12.5%, mid-lat → 29%, polar → 65% per sub-step.
            # Uses w_mid_v (the v-target's own, decoupled centre), not w_mid (the jet's) --
            # closes the strength/direction mismatch left open when ferrel_v_centre_deg was
            # decoupled from the jet centre (see PlanetParams.ferrel_v_centre_deg / the
            # "known residual" note in midwest-ferrel-and-spherical-metric-2026-07-25 memory).
            # Bit-identical to the old behaviour whenever ferrel_v_centre_deg == 48.
            a_v_row = np.clip(a * (1.0 + 5.0 * w_trade[:, None] + 12.0 * w_mid_v[:, None] + 3.0 * w_polar[:, None]), 0.0, 0.75).astype(np.float32, copy=False)
            v_curr = v_curr + (v_target_2d - v_zm) * a_v_row
        
        # Soft clamp to prevent explosion
        total_v = np.hypot(u_curr, v_curr)
        vmax = float(vmax_clip)
        mask_high = total_v > vmax
        scale = vmax / (total_v + 1e-6)
        u_curr[mask_high] *= scale[mask_high]
        v_curr[mask_high] *= scale[mask_high]
    
    return u_curr.astype(np.float32), v_curr.astype(np.float32)


def evolve_wind_aloft(
    u2: np.ndarray,
    v2: np.ndarray,
    temperature: np.ndarray,
    dt_days: float = 1.0,
    pgf_temp_scale: float = 450.0,
    upper_pgf_amp: float = 2.5,
    damping_rate: float = 0.05,
    vmax_clip: float = 150.0,
    planet_params: PlanetParams | None = None,
    hadley_edge_deg: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve the upper-level (1.5-layer) prognostic wind field.

    This is the real, independent momentum budget behind the "1.5-layer
    atmosphere": the upper level has its own advected, Coriolis-rotated,
    pressure-driven wind (u2, v2) rather than being a per-step diagnostic
    slaved to the surface -- what actually lets vertical shear and
    baroclinic-style jet behavior emerge from the temperature field, instead
    of the old magnitude-only `baroclinic_jet_amp * jet_window * |dT/dy|`
    proxy this replaces (see evolve_wind's baroclinic-mixing term, which now
    relaxes the surface wind toward this layer's output instead).

    Deliberately simpler than `evolve_wind`: no terrain blocking/channeling,
    no ice-pressure feedback, no Rossby/storm/blocking pressure terms --
    those are surface/boundary-layer phenomena, kept there for this pass
    (see ROADMAP.md Theme 1's deferred-retirement note). Friction is a weak
    Rayleigh damping rather than the surface's quadratic/terrain-enhanced
    drag, since the upper troposphere is nearly frictionless compared to the
    boundary layer. The coupling back to the surface (in evolve_wind) is
    one-way (surface relaxes toward this layer) rather than a fully
    momentum-conserving exchange -- a deliberate simplification for this
    first pass, consistent with how real boundary-layer schemes already
    treat the near-surface layer asymmetrically relative to the free
    troposphere.

    `hadley_edge_deg` (see PlanetParams.wind_upper_hadley_edge_deg) sets the
    width of an extra equatorial-suppression window specific to this layer,
    representing the real Hadley cell's direct-circulation footprint (not
    otherwise modeled here) that keeps the free thermal-wind jet from
    forming too close to the equator -- see that field's docstring for the
    diagnostic that motivated it.
    """
    H, W = u2.shape
    dt_total = dt_days * 86400.0  # seconds
    pp = planet_params or EARTH

    lat_2d, dx, dy, f, eq_window, _lon_1d_cached = _wind_static_grids(H, W, pp)

    # Weaker friction than the surface layer means a longer stability
    # timescale; 4 sub-steps (half the surface's 8) keeps this affordable.
    n_steps = 4
    dt_sub = dt_total / n_steps
    rho = pp.reference_air_density

    u_curr, v_curr = u2.copy(), v2.copy()

    # Pressure anomaly: OPPOSITE sign convention from the surface's thermal
    # PGF term (`-pgf_temp_scale * (T-273.15)/30`), not just an amplified
    # copy of it -- confirmed empirically (see the calibration notes in
    # PLAN.md) that copying the surface's sign produces easterlies, not
    # westerlies, at mid-latitudes once actually integrated through the
    # Coriolis/PGF dynamics without the surface's 3-cell relaxation crutch.
    # This matches the real hypsometric relationship: a warm column is
    # thicker, so upper-level geopotential/pressure is relatively HIGHER over
    # warm regions and LOWER over cold ones -- inverted from the naive
    # surface "cold air = surface high" pattern. With this sign, a
    # poleward-decreasing temperature genuinely produces a poleward-directed
    # pressure gradient whose geostrophic response is westerly in both
    # hemispheres (f flips sign together with the gradient's effective
    # direction), which is what makes real thermal-wind-driven jets emerge
    # here instead of needing a relaxation target.
    p_anom = float(pgf_temp_scale) * float(upper_pgf_amp) * ((temperature - 273.15) / 30.0)

    dp_dx = _ddx_periodic(p_anom) / (dx + 1e-3)
    dp_dy = -np.gradient(p_anom, axis=0) / dy
    pgf_u = -1.0 / rho * dp_dx
    pgf_v = -1.0 / rho * dp_dy

    # Thermal-wind balance -- the whole basis of this layer's PGF term -- breaks
    # down near the equator where f -> 0: Coriolis rotation barely turns the
    # flow each sub-step, so the persistent PGF forcing just keeps accelerating
    # u/v under this layer's otherwise-uniform weak friction instead of
    # settling into a balanced jet. That produced a spurious "false jet" in the
    # deep tropics (empirically the strongest zonal-mean response in the whole
    # profile) that swamped the much weaker, but physically real, mid-latitude
    # thermal-wind signal. The surface layer already corrects for this with an
    # equatorial drag bump (evolve_wind's `2.0e-6 * eq_window` quadratic-drag
    # term); mirror it here as extra linear Rayleigh damping, scaled to this
    # layer's own weak-friction units, so unbalanced tropical acceleration
    # doesn't dominate over genuine mid-latitude jet dynamics.
    # Separately, the polar rows suffer the opposite failure mode: dx = R *
    # (2*pi/W) * cos(lat) collapses toward the pole, so the same dp_dx/dx PGF
    # term diverges there even after the _wind_static_grids dx floor above
    # softens it -- any residual per-column zonal temperature asymmetry near
    # the pole (from the land/ice distribution) still gets amplified into a
    # locally huge, noisy pgf_u/pgf_v that this layer's uniform weak friction
    # can't relax away before the vmax_clip below silently caps it every step.
    # Add the same style of extra damping there, windowed by distance from
    # the pole rather than the equator.
    abs_lat_deg_local = np.abs(np.rad2deg(lat_2d)).astype(np.float32, copy=False)
    polar_window = np.exp(-(((90.0 - abs_lat_deg_local) / 10.0) ** 2)).astype(np.float32, copy=False)

    # Hadley-cell footprint suppression -- wider than the surface layer's
    # `eq_window` (sigma=12 deg, tuned for surface Ekman/frictional damping
    # in the deep tropics). Real subtropical jets sit at the Hadley cell's
    # poleward edge (~25-30 deg), not at the local dT/dy peak: within the
    # cell's footprint, direct meridional overturning (not modeled by this
    # layer's pure thermal-wind balance) dominates over geostrophic dynamics,
    # so free thermal-wind response should stay suppressed out to roughly the
    # cell edge rather than just the deep tropics. Without this wider window,
    # diagnostics (see jet-stream-vs-real-world memory) showed the emergent
    # jet peaking at ~18 deg in both hemispheres regardless of where the
    # actual simulated dT/dy peaked (pole-ish in the NH, a very Earth-like
    # ~46 deg in the SH) -- i.e. the peak tracked the edge of the (too
    # narrow) equatorial damping bump, not the real gradient shape.
    _hadley_edge_deg = float(hadley_edge_deg)
    eq_window_aloft = np.exp(-((abs_lat_deg_local / _hadley_edge_deg) ** 2)).astype(np.float32, copy=False)

    k_damp = float(damping_rate) / 86400.0  # 1/day -> 1/s
    k_damp = k_damp + (1.5 / 86400.0) * eq_window_aloft + (3.0 / 86400.0) * polar_window

    for _ in range(n_steps):
        # 1. Semi-Lagrangian advection (unconditionally stable) -- same scheme
        # as the surface layer.
        u_adv, v_adv = _advect_wind_semi_lagrangian(u_curr, v_curr, dt_sub, dx, dy)

        # 2. Coriolis -- exact rotation matrix, same as the surface layer.
        theta = f * dt_sub
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        u_rot = cos_t * u_adv + sin_t * v_adv
        v_rot = -sin_t * u_adv + cos_t * v_adv

        # 3. PGF + weak Rayleigh friction (no terrain/quadratic drag aloft).
        friction_u = -k_damp * u_rot
        friction_v = -k_damp * v_rot
        du = (pgf_u + friction_u) * dt_sub
        dv = (pgf_v + friction_v) * dt_sub

        u_curr = u_rot + du
        v_curr = v_rot + dv

        # Soft clamp to prevent explosion (same pattern as evolve_wind).
        total_v = np.hypot(u_curr, v_curr)
        vmax = float(vmax_clip)
        mask_high = total_v > vmax
        scale = vmax / (total_v + 1e-6)
        u_curr[mask_high] *= scale[mask_high]
        v_curr[mask_high] *= scale[mask_high]

    return u_curr.astype(np.float32), v_curr.astype(np.float32)


def generate_wind_field(
    height: int,
    width: int,
    *,
    day_of_year: float = 80.0,
    block_size: int = 3,
    upsample: str = "repeat",
    temperature: np.ndarray | None = None,
    elevation: np.ndarray | None = None,
    terrain_influence: float = 1.0,
    weather_amp: float = 1.0,
    zonal_pressure: float = 0.85,
    terrain_pressure_amp: float = 1.0,
    terrain_flow_amp: float = 1.0,
    time_days: float | None = None,
    planet_params: PlanetParams | None = None,
    debug_log: bool = False,
    jet_index_nh: float = 0.0,
    jet_index_sh: float = 0.0,
    jet_block_nh: tuple[float, float, float] = (-1.0, 0.0, 0.0),
    jet_block_sh: tuple[float, float, float] = (-1.0, 0.0, 0.0),
    debug_fields: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (u, v) near-surface winds derived from pressure gradients.

    Build surface pressure from temperature (with land-sea contrast and seasonal
    variation), add terrain and weather-system perturbations, then derive winds
    from pressure gradients via geostrophic balance. A streamfunction solver
    ensures divergence-free flow while preserving realistic meridional components.

    `jet_index_{nh,sh}`/`jet_block_{nh,sh}` (see atmosphere._update_jet_index /
    _update_jet_blocking) let the `weather_amp` perturbation share the exact
    same storm/Rossby-wave/blocking-ridge pressure signal as the prognostic
    evolve_wind() path (_synoptic_wave_pressure_anomaly), instead of this
    diagnostic wind falling back to its own separate, simpler wave model.
    This is what makes switching time-scale modes (which swap between this
    diagnostic wind and the full prognostic evolve_wind) not visibly disrupt
    the monthly precip statistics Köppen classification reads.

    IMPORTANT: this is the wind actually used for MONTHLY/ANNUAL time-scale
    modes (`update_wind=False` in `simulate_step`), which is this project's
    established calibration methodology -- not just a preview/fallback path.
    When `temperature` is provided (the real simulated field, always true for
    in-sim calls), `zonal_pressure` (default 0.85) blends 85% of the pressure
    signal from the *zonal-mean* temperature at that latitude, diluting any
    actual per-cell land/ocean contrast -- `pgf_continentality_amp`
    (PlanetParams) locally reduces that blend fraction over continental
    interior (`masks.get_continentality`), letting the real per-cell thermal
    signal matter more precisely where the zonal-mean dilution was
    suppressing the thermal-low/monsoon convergence signal
    (known-physics-gaps.md item 3b).

    `debug_fields`, if provided (an empty dict to populate), receives
    `p_thermal`, `p_terrain`, and `zonal_blend_eff` (the continentality-
    adjusted zonal blend fraction actually used). Diagnostic-only; no effect
    on the returned wind when omitted (default `None`).
    """

    pp = planet_params or EARTH
    H = int(height)
    W = int(width)
    Hc, Wc = _coarse_shape(H, W, block_size)
    lat = _latitudes_h(Hc)
    lon = np.linspace(-np.pi, np.pi, Wc, endpoint=False)
    abs_deg = np.rad2deg(np.abs(lat))

    # Prepare terrain and land/sea masks
    elev_c: np.ndarray
    land_mask: np.ndarray
    sea_mask: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    
    if elevation is not None:
        elev_pad = np.pad(
            elevation.astype(np.float32),
            ((0, Hc * block_size - H), (0, Wc * block_size - W)),
            mode="edge",
        )
        elev_c = elev_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
        land_mask, sea_mask = _derive_land_sea_masks(elev_c)
        # np.gradient returns (d/d_row, d/d_col): row axis is meridional
        # (index increases SOUTHWARD), col axis is zonal. gx must be the
        # eastward slope and gy the physical NORTHWARD slope, so unpack in
        # the right order and negate the row derivative.
        _g_row, _g_col = np.gradient(elev_c)
        gx = _g_col
        gy = -_g_row
        # Full-resolution continentality (masks.get_continentality, id-cached
        # per elevation array), coarsened the same way elev_c is above.
        _cont_full = get_continentality(elevation.astype(np.float32, copy=False))
        _cont_pad = np.pad(
            _cont_full,
            ((0, Hc * block_size - H), (0, Wc * block_size - W)),
            mode="edge",
        )
        continentality_c = _cont_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
    else:
        elev_c = np.zeros((Hc, Wc), dtype=np.float32)
        land_mask = np.zeros((Hc, Wc), dtype=bool)
        sea_mask = np.ones((Hc, Wc), dtype=bool)
        gx = np.zeros((Hc, Wc), dtype=np.float32)
        gy = np.zeros((Hc, Wc), dtype=np.float32)
        continentality_c = np.zeros((Hc, Wc), dtype=np.float32)

    # Temperature field:
    # - If provided, use it (it should come from the simulation state).
    # - Otherwise fall back to a lightweight climatology.
    if temperature is not None:
        temp = temperature.astype(np.float32)
        if temp.shape == (H, W):
            temp_pad = np.pad(
                temp,
                ((0, Hc * block_size - H), (0, Wc * block_size - W)),
                mode="edge",
            )
            T = temp_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
        elif temp.shape == (Hc, Wc):
            T = temp
        else:
            raise ValueError(f"temperature must be shape {(H, W)} or {(Hc, Wc)}; got {temp.shape}")
        # Mild smoothing only (keep gradients that actually drive winds).
        T = np.clip(T + 0.05 * _laplacian(T), 200.0, 330.0)
        season_phase = 2.0 * np.pi * (day_of_year - 80) / pp.orbital_period_days
        land_f = land_mask.astype(np.float32)
    else:
        # Build 2D temperature field with land-sea contrast and longitudinal variation
        T_lat = temperature_kelvin_for_lat(lat, day_of_year=day_of_year, planet_params=pp).astype(np.float32, copy=False)
        T = np.repeat(T_lat[:, None], Wc, axis=1).astype(np.float32, copy=False)
        
        # Land-sea temperature contrast (land warmer in summer, cooler in winter)
        land_f = land_mask.astype(np.float32)
        season_phase = 2.0 * np.pi * (day_of_year - 80) / pp.orbital_period_days
        seasonal_contrast = 8.0 * np.sin(season_phase) * np.cos(lat[:, None])  # NH summer = positive
        T += seasonal_contrast * land_f
        
        # Coastal gradients (sharp temperature transitions)
        coastal_grad = _laplacian(land_f)
        T += 3.5 * coastal_grad * (T / 280.0)
        
        # Add longitudinal temperature waves (continentality, monsoon drivers)
        T_wave = 6.0 * np.sin(lon[None, :] * 2.5 + season_phase) * land_f
        T_wave += 3.5 * np.sin(lon[None, :] * 4.0 - 0.8) * land_f
        T_wave *= np.exp(-((abs_deg[:, None] - 35.0) / 25.0) ** 2)  # Peak at mid-latitudes
        T += T_wave
        
        # Smooth temperature to avoid numerical instabilities
        T = np.clip(T + 0.15 * _laplacian(T), 200.0, 320.0)
    
    # Convert temperature to surface pressure (ideal gas law approximation)
    # For sim-driven winds we keep it mostly zonal to avoid stationary continent-scale blobs.
    # Locally reduced over continental interior (pgf_continentality_amp) so the real
    # per-cell land/ocean thermal contrast isn't diluted away exactly where it should
    # matter most -- see this function's docstring / known-physics-gaps.md item 3b.
    T_ref = 273.15
    zp = float(np.clip(zonal_pressure, 0.0, 1.0))
    zonal_blend_eff = zp * (1.0 - np.clip(float(pp.pgf_continentality_amp) * continentality_c, 0.0, 1.0))
    if temperature is not None and zp > 0.0:
        T_zonal = np.mean(T, axis=1, keepdims=True)
        T_used = zonal_blend_eff * T_zonal + (1.0 - zonal_blend_eff) * T
    else:
        T_used = T
    # Warmer = lower pressure (thermal low), colder = higher pressure (thermal high)
    p_thermal = (pp.surface_pressure_pa / 100.0) * (T_ref / (T_used + 1e-6)) ** 2.2  # hPa
    if debug_fields is not None:
        debug_fields["p_thermal"] = p_thermal
        debug_fields["zonal_blend_eff"] = zonal_blend_eff

    # Add terrain pressure anomalies (mountains create blocking highs)
    tp = float(np.clip(terrain_pressure_amp, 0.0, 1.0))
    p_terrain = 25.0 * tp * terrain_influence * np.clip(elev_c, 0.0, 1.0)
    if debug_fields is not None:
        debug_fields["p_terrain"] = p_terrain
    
    # Optional: inject weak synoptic-scale perturbations.
    wamp = float(np.clip(weather_amp, 0.0, 1.0))
    if wamp > 0.0:
        # If simulation time is provided, use traveling-wave perturbations (moving eddies)
        # instead of stationary, day-seeded blobs.
        t_days = float(time_days) if time_days is not None else float(day_of_year)
        if temperature is not None:
            # Same deterministic Rossby-wave + storm/trade-wave + blocking-ridge
            # signal as the prognostic evolve_wind() path (shared helper), so
            # this diagnostic wind doesn't fall back to a separate, simpler
            # wave model with different constants and no storms/blocking --
            # see _synoptic_wave_pressure_anomaly's docstring.
            lat_2d_c = np.repeat(lat[:, None], Wc, axis=1)
            synoptic_pa = _synoptic_wave_pressure_anomaly(
                lat_2d_c, lon, t_days,
                jet_index_nh, jet_index_sh, jet_block_nh, jet_block_sh, pp,
            )
            p_thermal += wamp * (synoptic_pa / 100.0)  # Pa -> hPa
        else:
            # Fallback (static view): small stationary systems
            rng = np.random.default_rng(int(day_of_year) + 9001)
            n_systems = 6
            for _ in range(n_systems):
                sys_lon = rng.uniform(-np.pi, np.pi)
                sys_lat = rng.uniform(-0.6 * np.pi, 0.6 * np.pi)
                sys_strength = rng.uniform(-12.0, 12.0) * wamp
                sys_scale = rng.uniform(0.18, 0.40)
                
                dist_lon = np.abs(lon[None, :] - sys_lon)
                dist_lon = np.minimum(dist_lon, 2.0 * np.pi - dist_lon)
                dist_lat = np.abs(lat[:, None] - sys_lat)
                dist = np.sqrt(dist_lon ** 2 + dist_lat ** 2)
                
                p_system = sys_strength * np.exp(-(dist / sys_scale) ** 2)
                p_thermal += p_system
    
    # Total pressure field
    pressure = p_thermal + p_terrain
    pressure = pressure + 0.2 * _laplacian(pressure)  # Smooth
    
    # Derive geostrophic winds from pressure gradients
    # In geostrophic balance: u ∝ -∂p/∂y, v ∝ ∂p/∂x (Northern Hemisphere)
    # Scale by Coriolis parameter f = 2Ω sin(φ)
    f_coriolis = pp.coriolis_parameter(lat).astype(np.float32, copy=False)
    # Avoid division by zero at equator: enforce minimum magnitude while preserving sign
    f_min = max(3e-5, 0.1 * abs(float(pp.omega)))
    mask_pos = f_coriolis >= 0
    f_coriolis = np.where(mask_pos, np.maximum(f_coriolis, f_min), np.minimum(f_coriolis, -f_min))
    
    # Axis 0 is north→south (index increases southward), so physical northward gradient is negated.
    # Use proper metric terms (meters), consistent with `evolve_wind`.
    R_earth = float(pp.radius_m)
    lat_2d = np.repeat(lat[:, None], Wc, axis=1)
    dx = R_earth * (2 * np.pi / Wc) * np.cos(lat_2d)
    dy = R_earth * (np.pi / Hc)
    p_pa = (pressure * 100.0).astype(np.float32, copy=False)  # hPa -> Pa
    dp_dy = -np.gradient(p_pa, axis=0) / dy
    dp_dx = _ddx_periodic(p_pa) / (dx + 1e-3)
    
    # Geostrophic wind (m/s)
    rho = pp.reference_air_density
    u_geo = -(1.0 / (rho * f_coriolis[:, None])) * dp_dy
    v_geo = (1.0 / (rho * f_coriolis[:, None])) * dp_dx
    
    # Add tropical wind model (trade winds/Walker circulation)
    # Geostrophic approximation breaks down near equator; use direct tropical circulation
    abs_lat = np.abs(lat)
    tropical_mask = abs_lat < np.deg2rad(25.0)  # ±25° tropical zone
    
    # Tropical winds (Hadley-cell-like):
    # - Doldrums near the equator (weak surface winds)
    # - Trades peak off-equator (~10-20°) and converge toward ITCZ.
    lat0 = np.deg2rad(25.0)
    absn = np.clip(abs_lat / lat0, 0.0, 1.0)
    trade_profile = np.sin(np.pi * absn)  # 0 at equator and 25°, peak near 12.5°
    u_tropical = -(4.0 * trade_profile[:, None]) * (1.0 + 0.12 * np.sin(lon[None, :] * 1.4))  # easterlies
    v_tropical = -1.5 * np.tanh(lat[:, None] / np.deg2rad(9.0)) * (1.0 - absn[:, None])  # toward equator
    
    # Blend: tropical model in tropics, geostrophic elsewhere
    # Tropical zones use primarily tropical model with small geostrophic component
    geo_scale_mid = 0.16
    geo_scale_trop = 0.03
    u_geo = np.where(tropical_mask[:, None], 
                     u_tropical + geo_scale_trop * u_geo,  # Small geostrophic contribution
                     geo_scale_mid * u_geo)  # Mid/high latitudes use geostrophic
    v_geo = np.where(tropical_mask[:, None],
                     v_tropical + geo_scale_trop * v_geo,
                     geo_scale_mid * v_geo)
    
    # Latitude-dependent clipping for realistic wind speeds
    tropical_limit_u = 12.0  # Weaker tropical winds
    midlat_limit_u = 22.0    # Stronger mid-latitude storm tracks
    tropical_limit_v = 8.0
    midlat_limit_v = 15.0
    
    # Scale limits linearly from equator (0) to 60° (1); equator keeps tropical caps
    lat_factor = np.clip(np.abs(lat) / np.deg2rad(60.0), 0.0, 1.0)
    u_limit = tropical_limit_u + (midlat_limit_u - tropical_limit_u) * lat_factor
    v_limit = tropical_limit_v + (midlat_limit_v - tropical_limit_v) * lat_factor
    
    u_geo = np.clip(u_geo, -u_limit[:, None], u_limit[:, None])
    v_geo = np.clip(v_geo, -v_limit[:, None], v_limit[:, None])
    
    # Add ageostrophic component (cross-isobar flow toward low pressure)
    ageo_frac = 0.03
    u_ageo = -ageo_frac * dp_dx / (np.abs(dp_dx) + 1e-3) * np.abs(u_geo) * 0.10
    v_ageo = -ageo_frac * dp_dy / (np.abs(dp_dy) + 1e-3) * np.abs(v_geo) * 0.10
    
    uc = u_geo + u_ageo
    vc = v_geo + v_ageo

    # --- Tiny 2-layer jet correction (thermal-wind inspired) ---
    # Real jets strengthen where meridional temperature gradients are strong (mid-lats).
    # We approximate an upper-level westerly anomaly from |dT/dy| and mix a fraction down.
    # Use the same metric as the pressure-gradient step above (meters per latitude row).
    dT_dy = np.gradient(T, axis=0) / dy
    jet_window = np.exp(-((abs_deg[:, None] - 48.0) / 14.0) ** 2).astype(np.float32, copy=False)
    thermal_wind_coeff = 2.8e6  # tuned: stronger mid-lat jet support without polar amplification
    u_aloft = thermal_wind_coeff * jet_window * np.abs(dT_dy)
    surface_mix = 0.24
    uc = uc + surface_mix * u_aloft
    trop_amp = 0.16 * np.exp(-((abs_deg - 15.0) / 12.0) ** 2)
    mid_amp = 0.72 * np.exp(-((abs_deg - 48.0) / 14.0) ** 2)
    polar_amp = 0.04 * np.exp(-((abs_deg - 76.0) / 12.0) ** 2)
    lat_amp = 0.10 + trop_amp + mid_amp + polar_amp
    global_amp = 0.65
    uc = uc * lat_amp[:, None] * global_amp
    vc = vc * lat_amp[:, None] * global_amp
    
    # Convert to vorticity and solve via streamfunction to ensure divergence-free
    dvc_dx = _ddx_periodic(vc)
    duc_dy = -np.gradient(uc, axis=0)
    omega = dvc_dx - duc_dy
    
    # Solve for streamfunction
    psi = _streamfunction_from_vorticity(omega)
    u_stream = -np.gradient(psi, axis=0) * (Hc / (np.pi))
    v_stream = -_ddx_periodic(psi) * (Wc / (2.0 * np.pi))
    
    # Blend: mostly from pressure gradients, streamfunction ensures consistency
    uc = 0.75 * uc + 0.25 * u_stream
    vc = 0.75 * vc + 0.25 * v_stream

    # The pressure-gradient solve captures synoptic structure but still under-produces
    # the near-surface 3-cell climatology, especially Ferrel flow. Apply a weak
    # zonal-mean correction so the diagnostic wind remains a useful relaxation target.
    sign_lat = np.sign(lat).astype(np.float32, copy=False)
    w_trade = np.exp(-((abs_deg - 14.0) / 9.0) ** 2).astype(np.float32, copy=False)
    w_mid = np.exp(-((abs_deg - 48.0) / 13.0) ** 2).astype(np.float32, copy=False)
    w_polar = np.exp(-((abs_deg - 74.0) / 10.0) ** 2).astype(np.float32, copy=False)
    u_surface = (-3.5 * w_trade + 8.5 * w_mid - 1.5 * w_polar).astype(np.float32, copy=False)
    # The meridional cell structure gets its OWN mid-latitude centre, decoupled
    # from u_surface's. Both used `w_mid` (48 deg) until 2026-07-25, which meant
    # the latitude where the zonal-mean flow switches from diverging to
    # converging could not be moved without also moving the surface jet.
    #
    # Why this matters: that crossing latitude is what decides whether a mid-
    # latitude continent sits in the subtropical dry belt. Measured on real
    # terrain, the model's zonal-mean divergence crosses zero at ~48N vs Earth's
    # ~40N, which puts the whole 38-45N band -- every continent at that latitude,
    # the US Midwest box among them -- on the diverging side. Its divergence is
    # 85% zonal-mean, so no local perturbation can overcome it (which is why the
    # two attempts in us-midwest-wind-convergence-investigation-2026-07 failed).
    # The analytic crossing responds ~1:1 to this centre: 48->46.4N, 44->42.6N,
    # 42->40.7N, 40->38.8N. See PLAN_PHYSICS_FIXES.md.
    _v_centre = float(getattr(pp, "ferrel_v_centre_deg", 48.0))
    w_mid_v = (w_mid if _v_centre == 48.0
               else np.exp(-((abs_deg - _v_centre) / 13.0) ** 2).astype(np.float32, copy=False))
    # PlanetParams.ferrel_v_land_shift_deg: land cells get their own, further
    # -shifted centre; ocean cells keep `_v_centre` unshifted. Blended by
    # `land_f` (computed above from the same elevation-derived mask the rest
    # of this function uses), not a hard mask, so the transition across a
    # coastline is smooth rather than a step discontinuity in the v-target.
    # 0.0 (default) skips this branch entirely -- exact no-op, bit-identical
    # to the single-centre behaviour above.
    _v_land_shift = float(getattr(pp, "ferrel_v_land_shift_deg", 0.0))
    if _v_land_shift == 0.0:
        v_surface = (-3.5 * w_trade + 5.0 * w_mid_v - 1.2 * w_polar).astype(np.float32, copy=False) * sign_lat
        v_surface = np.where(abs_deg < 2.0, 0.0, v_surface).astype(np.float32, copy=False)
        v_surface_2d = v_surface[:, None]
    else:
        _v_centre_land = _v_centre + _v_land_shift
        w_mid_v_land = np.exp(-((abs_deg - _v_centre_land) / 13.0) ** 2).astype(np.float32, copy=False)
        _land_shift_taper = np.clip((55.0 - abs_deg) / 10.0, 0.0, 1.0)
        _land_shift_taper = (
            _land_shift_taper * _land_shift_taper * (3.0 - 2.0 * _land_shift_taper)
        ).astype(np.float32)
        land_shift_f = land_f * _land_shift_taper[:, None]
        w_mid_v_2d = w_mid_v[:, None] * (1.0 - land_shift_f) + w_mid_v_land[:, None] * land_shift_f
        v_surface_2d = (-3.5 * w_trade[:, None] + 5.0 * w_mid_v_2d - 1.2 * w_polar[:, None]).astype(np.float32, copy=False) * sign_lat[:, None]
        v_surface_2d = np.where(abs_deg[:, None] < 2.0, 0.0, v_surface_2d).astype(np.float32, copy=False)
    uc_zm = np.mean(uc, axis=1, keepdims=True)
    vc_zm = np.mean(vc, axis=1, keepdims=True)
    u_nudge = (0.18 + 0.18 * w_mid[:, None] + 0.06 * w_trade[:, None]).astype(np.float32, copy=False)
    v_nudge = (0.16 + 0.12 * w_trade[:, None] + 0.18 * w_mid[:, None]).astype(np.float32, copy=False)
    uc = uc + (u_surface[:, None] - uc_zm) * u_nudge
    vc = vc + (v_surface_2d - vc_zm) * v_nudge
    
    # Apply terrain effects: blocking, channeling, deflection
    if terrain_influence > 0:
        elev_norm = np.clip(elev_c, 0.0, 1.0)
        tf = float(np.clip(terrain_flow_amp, 0.0, 1.0))
        
        # Mountain blocking
        block_factor = np.clip(1.0 - tf * terrain_influence * 0.5 * elev_norm, 0.6, 1.0)
        
        # Terrain channeling (flow follows valleys) - reduced from 0.25 to 0.15
        slope_mag = np.hypot(gx, gy)
        channel_factor = 1.0 + tf * terrain_influence * 0.08 * slope_mag
        
        # Deflection around obstacles
        deflect_u = -tf * terrain_influence * 0.12 * gx * slope_mag
        deflect_v = -tf * terrain_influence * 0.12 * gy * slope_mag
        
        uc = (uc * block_factor + deflect_u) * channel_factor
        vc = (vc * block_factor + deflect_v) * channel_factor
        
        # Land-sea friction contrast (keep subtle; strong contrast creates stationary speed blobs)
        friction_factor = np.where(sea_mask, 1.03, 0.97)
        uc *= friction_factor
        vc *= friction_factor
    
    # Final smoothing
    uc = uc + 0.10 * _laplacian(uc)
    vc = vc + 0.10 * _laplacian(vc)
    
    uc = np.clip(uc, -u_limit[:, None], u_limit[:, None]).astype(np.float32, copy=False)
    vc = np.clip(vc, -v_limit[:, None], v_limit[:, None]).astype(np.float32, copy=False)
    
    # Debug logging for wind diagnostics
    if debug_log:
        from terrain import LOG
        wind_mag = np.sqrt(uc*uc + vc*vc)
        LOG.info(f"[Wind Debug Day {day_of_year}]")
        LOG.info(f"  Pressure: min={float(np.min(pressure)):.1f}, mean={float(np.mean(pressure)):.1f}, max={float(np.max(pressure)):.1f} hPa")
        LOG.info(f"  Pressure gradients: dp_dx mean={float(np.mean(np.abs(dp_dx))):.4f}, dp_dy mean={float(np.mean(np.abs(dp_dy))):.4f}")
        LOG.info(f"  f_coriolis: min={float(np.min(np.abs(f_coriolis))):.2e}, max={float(np.max(np.abs(f_coriolis))):.2e}")
        LOG.info(f"  u_final: min={float(np.min(uc)):.1f}, mean={float(np.mean(uc)):.1f}, max={float(np.max(uc)):.1f} m/s")
        LOG.info(f"  v_final: min={float(np.min(vc)):.1f}, mean={float(np.mean(vc)):.1f}, max={float(np.max(vc)):.1f} m/s")
        LOG.info(f"  Wind magnitude: min={float(np.min(wind_mag)):.1f}, mean={float(np.mean(wind_mag)):.1f}, max={float(np.max(wind_mag)):.1f} m/s")
        LOG.info(f"  Wind percentiles: p10={float(np.percentile(wind_mag, 10)):.1f}, p50={float(np.percentile(wind_mag, 50)):.1f}, p90={float(np.percentile(wind_mag, 90)):.1f} m/s")
        
        # Latitude band breakdown
        eq_band = np.abs(lat) < np.deg2rad(10)
        trop_band = (np.abs(lat) >= np.deg2rad(10)) & (np.abs(lat) < np.deg2rad(30))
        mid_band = (np.abs(lat) >= np.deg2rad(30)) & (np.abs(lat) < np.deg2rad(60))
        
        LOG.info(f"  By latitude - Equatorial (0-10°): mean={float(np.mean(wind_mag[eq_band[:, None].repeat(Wc, 1)])):.1f} m/s")
        LOG.info(f"  By latitude - Tropical (10-30°): mean={float(np.mean(wind_mag[trop_band[:, None].repeat(Wc, 1)])):.1f} m/s")
        LOG.info(f"  By latitude - Mid-lat (30-60°): mean={float(np.mean(wind_mag[mid_band[:, None].repeat(Wc, 1)])):.1f} m/s")
        
        # Clipping statistics
        u_clipped = np.sum((uc == 30.0) | (uc == -30.0))
        v_clipped = np.sum((vc == 20.0) | (vc == -20.0))
        total_cells = uc.size
        LOG.info(f"  Clipping: u_clipped={u_clipped}/{total_cells} ({100.0*u_clipped/total_cells:.1f}%), v_clipped={v_clipped}/{total_cells} ({100.0*v_clipped/total_cells:.1f}%)")

    up = _upsample_bilinear if str(upsample).lower() == "bilinear" else _upsample_repeat
    return up(uc, H, W, block_size), up(vc, H, W, block_size)


def _cos_window(x_deg: np.ndarray, a: float, b: float) -> np.ndarray:
    # Smooth 0→1→0 window over [a,b] degrees using raised cosine; outside=0.
    x = np.asarray(x_deg, dtype=np.float32)
    y = np.zeros_like(x)
    m = (x >= a) & (x <= b)
    t = (x[m] - a) / max(b - a, 1e-6)
    y[m] = 0.5 * (1.0 - np.cos(np.pi * t))
    return y


def render_wind_arrows(height: int, width: int, u: np.ndarray, v: np.ndarray, *, step: int | None = None, target_arrows: int | None = 250, scale: float = 0.8) -> np.ndarray:
    """Rasterize sparse white arrows onto black background; returns (H,W,3) float.

    - `step` controls arrow density; `scale` scales arrow length relative to step.
    - Arrows point in wind direction; length varies slightly with wind speed for readability.
    """
    H = int(height); W = int(width)
    img = np.zeros((H, W, 3), dtype=np.float32)
    mag = np.sqrt(u * u + v * v) + 1e-9
    umax = np.percentile(mag, 95.0) + 1e-6
    if step is None:
        # Choose step so ~target_arrows triangles are drawn on equirectangular grid
        n_target = max(50, int(target_arrows or 250))
        step_f = np.sqrt((H * W) / float(n_target))
        sx = sy = max(6, int(step_f))
    else:
        sx = sy = max(4, int(step))
    step_len = float(min(sx, sy))
    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    for y in range(sy // 2, H, sy):
        for x in range(sx // 2, W, sx):
            uu = float(u[y, x]); vv = float(v[y, x])
            m = np.sqrt(uu * uu + vv * vv)
            if m < 1e-3:
                continue
            # Direction vector (normalized). Note: screen y grows downward, so flip v.
            dx = uu / m
            dy = -vv / m  # screen y grows down

            # Mostly-constant arrow size (matches "quiver"-like look); small speed modulation.
            t = min(1.0, m / umax)
            arrow_len = scale * step_len * (0.42 + 0.22 * t)
            arrow_len = max(3.0, min(arrow_len, step_len * 0.85))

            # Shaft from tail -> just before head.
            head_len = max(2.0, arrow_len * 0.45)
            shaft_len = max(1.0, arrow_len - head_len)

            tail_x = x - dx * (shaft_len * 0.55)
            tail_y = y - dy * (shaft_len * 0.55)
            head_base_x = x + dx * (shaft_len * 0.45)
            head_base_y = y + dy * (shaft_len * 0.45)
            tip_x = head_base_x + dx * head_len
            tip_y = head_base_y + dy * head_len

            _draw_line(img, tail_x, tail_y, head_base_x, head_base_y, white)

            # Arrow head: narrow filled triangle.
            perp_x = -dy
            perp_y = dx
            head_w = max(1.0, head_len * 0.55)
            p1_x = head_base_x + perp_x * head_w
            p1_y = head_base_y + perp_y * head_w
            p2_x = head_base_x - perp_x * head_w
            p2_y = head_base_y - perp_y * head_w
            _draw_triangle(img, tip_x, tip_y, p1_x, p1_y, p2_x, p2_y, white)
    return img


def _smooth_circular(arr: np.ndarray, window: int) -> np.ndarray:
    """Circular (wraparound) moving average -- longitude is periodic, so a plain
    moving average would falsely dim/shift the line near the +/-180 deg seam."""
    w = int(window)
    if w <= 1:
        return arr
    pad = w // 2
    padded = np.concatenate([arr[-pad:], arr, arr[:pad]])
    kernel = np.ones(2 * pad + 1, dtype=np.float32) / (2 * pad + 1)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def render_jet_stream_overlay(
    height: int,
    width: int,
    u_aloft: np.ndarray,
    v_aloft: np.ndarray,
    *,
    lat_band_deg: tuple[float, float] = (20.0, 60.0),
    speed_low: float = 8.0,
    speed_high: float = 30.0,
    alpha_range: tuple[float, float] = (0.35, 0.95),
    smooth_columns: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace the jet-stream core (peak upper-level wind band per column) as a colored ribbon.

    `lat_band_deg` is deliberately narrower than "all mid+high latitudes": this model's
    upper-level wind (evolve_wind_aloft) develops a second, even faster wind maximum near
    the poles (a polar-vortex-like feature, not the tropospheric jet), separated from the
    actual mid-latitude jet by a real minimum around 55-65 deg. A wider band would have the
    traced line snap to that poleward feature instead of the jet whenever it's locally
    stronger. The default (20, 60) brackets the jet's climatological core (MID_LAT_JET_CENTER_DEG
    = 48 deg) plus its full meander range (+/- jet_lat_shift_per_index * 2) while staying clear
    of that valley.

    For each hemisphere and column, the traced latitude is a speed-weighted (speed**3, to
    favor the true core over the band's tails) centroid row rather than a hard argmax -- a
    single noisy column's argmax can jump between comparable local peaks, which drew a
    jagged, right-angled line before this was added. `smooth_columns` (must stay odd-ish;
    halved and re-doubled internally) applies a circular longitude moving average on top,
    since the physical meander wavelength (Rossby wavenumbers 3-7, see ROSSBY_MODES) spans
    tens of degrees -- far wider than single-column sensor noise -- so smoothing at this
    scale cleans up noise without erasing the actual meander/blocking shape.
    Color/opacity fade continuously from pale yellow (near `speed_low`) to red
    (>= `speed_high`) rather than a hard on/off cutoff, so locally weak stretches of the jet
    stay faintly visible instead of leaving a misleading gap.

    Returns (overlay_rgb (H,W,3) float, alpha (H,W) float) for standard
    "(1-alpha)*base + alpha*overlay" compositing.
    """
    H, W = int(height), int(width)
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    alpha = np.zeros((H, W), dtype=np.float32)
    speed = np.hypot(u_aloft, v_aloft).astype(np.float32)
    lat_deg = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * 180.0

    color_slow = np.array([0.95, 0.85, 0.25], dtype=np.float32)  # pale yellow
    color_fast = np.array([0.95, 0.10, 0.05], dtype=np.float32)  # red
    lo_speed, hi_speed = float(speed_low), float(speed_high)
    a_lo, a_hi = alpha_range
    lat_lo, lat_hi = lat_band_deg

    def _trace(row_mask: np.ndarray) -> None:
        rows = np.where(row_mask)[0]
        if rows.size == 0:
            return
        band = speed[rows, :]                       # (Rb, W)
        peak_speed = _smooth_circular(band.max(axis=0), smooth_columns)
        weight = band.astype(np.float64) ** 3
        centroid_row = (rows[:, None] * weight).sum(axis=0) / np.maximum(weight.sum(axis=0), 1e-9)
        centroid_row = _smooth_circular(centroid_row.astype(np.float32), smooth_columns)
        prev_x = prev_y = None
        for x in range(W):
            y = int(round(float(centroid_row[x])))
            y = max(0, min(H - 1, y))
            t = float(np.clip((float(peak_speed[x]) - lo_speed) / max(hi_speed - lo_speed, 1e-6), 0.0, 1.0))
            col = color_slow + (color_fast - color_slow) * t
            a = a_lo + (a_hi - a_lo) * t
            if prev_x is not None:
                _draw_line(overlay, prev_x, prev_y, x, y, col)
            overlay[y, x, :] = np.maximum(overlay[y, x, :], col)
            alpha[y, x] = max(alpha[y, x], a)
            prev_x, prev_y = x, y

    _trace((lat_deg >= lat_lo) & (lat_deg <= lat_hi))
    _trace((lat_deg <= -lat_lo) & (lat_deg >= -lat_hi))

    # Connecting-line pixels between per-column peaks got a color from _draw_line's
    # max-blend but not necessarily an alpha entry; backfill those to the floor alpha.
    touched = overlay.any(axis=-1) & (alpha <= 0.0)
    alpha[touched] = a_lo
    return overlay, alpha


def wind_speed_to_rgb(
    speed: np.ndarray,
    *,
    vmax: float | None = None,
    gamma: float = 0.75,
) -> np.ndarray:
    """Map wind speed (m/s) -> RGB float image (H,W,3).

    If `vmax` is None, choose a robust per-frame scale from the data
    (99.5th percentile). This avoids fixed ceilings and lets the colormap
    adapt to whatever range the simulation produces.
    """
    s = speed.astype(np.float32)
    if vmax is None:
        # Robust scale: ignore rare extremes so the map doesn't saturate.
        vm = float(np.nanpercentile(s, 99.5))
    else:
        vm = float(vmax)
    vm = max(1e-6, vm)
    t = np.clip(s / vm, 0.0, 1.0)
    t = t ** float(gamma)

    # Slow → fast: blue → green → yellow → red.
    cstops = np.array(
        [
            [0.06, 0.25, 0.92],  # blue (slow)
            [0.10, 0.78, 0.55],  # green
            [0.95, 0.88, 0.20],  # yellow
            [0.92, 0.18, 0.10],  # red (fast)
        ],
        dtype=np.float32,
    )
    bp = np.array([0.0, 0.50, 0.80, 1.0], dtype=np.float32)
    i = np.clip(np.searchsorted(bp, t, side="right") - 1, 0, len(bp) - 2)
    c0 = cstops[i]
    c1 = cstops[i + 1]
    tt = (t - bp[i]) / (bp[i + 1] - bp[i] + 1e-6)
    return c0 + (c1 - c0) * tt[..., None]


def _draw_line(img: np.ndarray, x0: float, y0: float, x1: float, y1: float, col: np.ndarray) -> None:
    H, W, _ = img.shape
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    for i in range(n):
        t = i / max(n - 1, 1)
        xi = int(round(x0 + (x1 - x0) * t))
        yi = int(round(y0 + (y1 - y0) * t))
        if 0 <= xi < W and 0 <= yi < H:
            img[yi, xi, :] = np.maximum(img[yi, xi, :], col)


def _draw_triangle(img: np.ndarray, x0: float, y0: float, x1: float, y1: float, x2: float, y2: float, col: np.ndarray) -> None:
    """Draw filled triangle using scanline algorithm."""
    H, W, _ = img.shape
    # Sort vertices by y
    pts = [(x0, y0), (x1, y1), (x2, y2)]
    pts.sort(key=lambda p: p[1])
    x0, y0 = pts[0]; x1, y1 = pts[1]; x2, y2 = pts[2]
    
    # Bounding box
    min_x = int(max(0, min(x0, x1, x2)))
    max_x = int(min(W - 1, max(x0, x1, x2)))
    min_y = int(max(0, min(y0, y1, y2)))
    max_y = int(min(H - 1, max(y0, y1, y2)))
    
    if min_y >= max_y or min_x >= max_x:
        return
    
    # Scanline fill
    for y in range(min_y, max_y + 1):
        # Find intersections with triangle edges
        intersections = []
        # Edge 0-1
        if y0 != y1:
            t = (y - y0) / (y1 - y0)
            if 0 <= t <= 1:
                intersections.append(x0 + t * (x1 - x0))
        # Edge 0-2
        if y0 != y2:
            t = (y - y0) / (y2 - y0)
            if 0 <= t <= 1:
                intersections.append(x0 + t * (x2 - x0))
        # Edge 1-2
        if y1 != y2:
            t = (y - y1) / (y2 - y1)
            if 0 <= t <= 1:
                intersections.append(x1 + t * (x2 - x1))
        
        if len(intersections) >= 2:
            x_start = int(round(min(intersections)))
            x_end = int(round(max(intersections)))
            for x in range(max(min_x, x_start), min(max_x + 1, x_end + 1)):
                if 0 <= x < W and 0 <= y < H:
                    img[y, x, :] = np.maximum(img[y, x, :], col)


def _speed_color(t: float) -> np.ndarray:
    # Blue→Red gradient by speed fraction t∈[0,1]
    t = float(np.clip(t, 0.0, 1.0))
    c0 = np.array([0.05, 0.40, 1.00], dtype=np.float32)
    c1 = np.array([1.00, 0.20, 0.05], dtype=np.float32)
    return (1.0 - t) * c0 + t * c1


def generate_precipitation(
    height: int,
    width: int,
    elevation: np.ndarray,
    *,
    temperature: np.ndarray | None = None,
    wind_u: np.ndarray | None = None,
    wind_v: np.ndarray | None = None,
    wind_u_aloft: np.ndarray | None = None,
    wind_v_aloft: np.ndarray | None = None,
    wind_u_midlevel: np.ndarray | None = None,
    wind_v_midlevel: np.ndarray | None = None,
    humidity: np.ndarray | None = None,
    soil_moisture: np.ndarray | None = None,
    soil_moisture_deep: np.ndarray | None = None,
    cloud_fraction: np.ndarray | None = None,
    condensate: np.ndarray | None = None,
    precipitating_hydrometeors: np.ndarray | None = None,
    midlevel_temperature: np.ndarray | None = None,
    midlevel_humidity: np.ndarray | None = None,
    upperlevel_temperature: np.ndarray | None = None,
    upperlevel_humidity: np.ndarray | None = None,
    column_lower_temperature: np.ndarray | None = None,
    previous_precipitation_mm_day: np.ndarray | None = None,
    day_of_year: float = 80.0,
    dt_days: float = 1.0,
    evap_coeff: float = 1.0,
    uplift_coeff: float = 1.0,
    rain_efficiency: float = 0.7,
    target_mean_mm_day: float = 2.7,
    max_precip_mm_day: float = 120.0,
    surface_pressure_hpa: float = 1013.25,
    planet_params: PlanetParams | None = None,
    debug_fields: dict | None = None,
    return_condensate: bool = False,
    return_midlevel_temperature: bool = False,
    return_midlevel_humidity: bool = False,
    return_upperlevel_state: bool = False,
    return_precipitating_hydrometeors: bool = False,
    _static_cache: dict | None = None,
) -> tuple:
    """Return precipitation, vapor, and soil reservoirs (plus condensate on request).

    `debug_fields`, if provided (an empty dict to populate), receives the
    `div`/`ascent`/`conv` wind-convergence driver arrays used internally.
    Diagnostic-only; no effect on the returned precipitation when omitted
    (default `None`). Used by `scripts/check_real_terrain_koppen.py
    --wind-diagnostics` to measure the real convergence signal without
    duplicating this function's formulas in a separate script.

    The model keeps a prognostic surface humidity field and a 2-layer soil-moisture
    bucket (fast surface layer `soil_moisture` + slow deep/root-zone reservoir
    `soil_moisture_deep`, see PlanetParams' `soil_field_capacity`/
    `soil_percolation_rate`/`soil_deep_drain_rate`/`soil_deep_evap_weight`) while
    blending three precipitation triggers: moisture convergence,
    orographic lift, and convective instability. Everything runs at the native
    grid resolution so it can operate in both snapshot and time-stepping modes.

    `cloud_fraction`, if provided, is this step's already-diagnosed cloud cover
    (from `_evolve_temperature`). It adds a stratiform term so widespread cloud
    sheets (frontal/persistent cover) produce rain even without a deep-convective
    trigger, closing the gap where clouds and precipitation were diagnosed from
    shared RH/ascent fields but never actually informed each other.

    `_static_cache` (internal, opt-in): when the caller drives several
    consecutive substeps with the *same* `elevation`/`temperature`/`wind`/
    `surface_pressure`/`dt` and only lets `humidity`/`soil` evolve (see
    `simulate._generate_precipitation_substepped`), pass a fresh empty dict so
    the loop-invariant fields — land/sea masks, `qsat`, wind divergence and its
    subsidence-suppression, the advection Courant scalings, the ascent proxy,
    and the percentile-normalised orographic uplift / rain-shadow blocks — are
    computed on the first substep and reused thereafter. Every cached quantity
    is a pure function of those invariant inputs, so the result is identical to
    recomputing it; the arrays are stored read-only. Leave it `None` (the
    default, and every other call site) to recompute everything as before.
    """

    pp = planet_params or EARTH

    H = int(height)
    W = int(width)
    elev = elevation.astype(np.float32, copy=False)
    lat_deg = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * 180.0
    abs_lat_deg = np.abs(lat_deg)
    # Seasonal ITCZ migration (2026-07-30, real-terrain-vs-reference-map audit): the belt's
    # *center*, not just its width, follows the sub-solar latitude in reality (thermal-inertia-
    # damped, not the full solar declination swing) -- this is what carves a wet season (ITCZ
    # overhead) and a dry season (ITCZ displaced to the other hemisphere) out of savanna
    # latitudes. `itcz_seasonal_response` is the damping fraction; 0.0 recovers the exact prior
    # static-equator behavior. See its PlanetParams docstring for the root-cause measurement.
    _itcz_center_deg = float(pp.itcz_seasonal_response) * math.degrees(pp.solar_declination(day_of_year))
    itcz_window = np.exp(-(((lat_deg - _itcz_center_deg) / ITCZ_HALF_WIDTH_DEG) ** 2)).astype(np.float32, copy=False)

    _sc = _static_cache
    if _sc is not None and "drybelt_window" in _sc:
        # Held fixed for every substep of one outer call, exactly like
        # temperature and wind (see `simulate._generate_precipitation_substepped`).
        # These windows are day-independent unless a seasonal response is enabled,
        # so caching them is a no-op at the defaults; with one enabled it is what
        # keeps `subsidence_suppression` -- which is cached below and reads
        # `drybelt_regime_window` -- consistent with the uncached terms that read
        # the same window further down.
        storm_window = _sc["storm_window"]
        drybelt_window = _sc["drybelt_window"]
        drybelt_regime_window = _sc["drybelt_regime_window"]
    else:
        # Seasonal migration of the subtropical high and the storm track. Unlike
        # the ITCZ, whose centre is a single signed latitude, these two belts are
        # symmetric about the equator, so their migration is hemisphere-
        # ANTIsymmetric: in NH summer the NH belt moves poleward while the SH
        # belt (in its winter) moves equatorward. Shifting `abs_lat_deg` by
        # `+/-shift` per hemisphere expresses that in one line and applies it to
        # the Gaussian and the wide regime window built from it alike.
        # See PlanetParams.drybelt_seasonal_response for why this matters
        # (Mediterranean climate is exactly this migration).
        _declination_deg = math.degrees(pp.solar_declination(day_of_year))
        _hemisphere = np.sign(lat_deg).astype(np.float32)
        _drybelt_shift = (
            float(pp.drybelt_seasonal_response) * _declination_deg * _hemisphere
        )
        # The Hadley cell's descending branch does not translate rigidly: its
        # poleward edge advances strongly into the summer hemisphere while its
        # equatorward edge is held near the ITCZ, so the belt *widens* in summer
        # rather than sliding off the subtropical deserts. A rigid translation
        # measurably uncovers them -- at response 0.3 the Sahara goes 129 -> 223
        # mm/yr, straight through its <200 target, because the whole belt (and
        # with it `subsidence_suppression`) leaves 15-22N for half the year.
        #
        # Expressed as one coordinate warp so the Gaussian and the wide regime
        # window below both inherit it: the shift is scaled by a weight that is
        # 0 equatorward of the belt centre and 1 poleward of it.
        _equatorward_fraction = float(pp.drybelt_seasonal_equatorward_fraction)
        # A latitude-dependent shift can fold the coordinate onto itself if it
        # changes faster than latitude does, which would give the belt a
        # spurious second peak. The warp's slope is
        # `|shift| * (1 - f) * max(d/dlat of smoothstep) = |shift|*(1-f)*1.5/W`,
        # so widening the transition with the shift keeps it below 1 for any
        # response in [0, 1]. At the shipped 0.25 the shift is ~5.9 deg and this
        # leaves the 16 deg transition untouched; the widening only engages
        # past ~0.45, where a fixed 16 deg would have started folding.
        _warp_width_deg = max(16.0, 1.6 * float(np.max(np.abs(_drybelt_shift))))
        _poleward_weight = np.clip(
            (abs_lat_deg - DRYBELT_CENTER_DEG) / _warp_width_deg + 0.5, 0.0, 1.0
        )
        _poleward_weight = (
            _poleward_weight * _poleward_weight * (3.0 - 2.0 * _poleward_weight)
        ).astype(np.float32)
        _drybelt_lat_deg = abs_lat_deg - _drybelt_shift * (
            _equatorward_fraction
            + (1.0 - _equatorward_fraction) * _poleward_weight
        )
        _storm_lat_deg = abs_lat_deg - (
            float(pp.storm_track_seasonal_response) * _declination_deg * _hemisphere
        )
        storm_window = np.exp(-((_storm_lat_deg - STORM_TRACK_CENTER_DEG) / 15.0) ** 2).astype(np.float32, copy=False)
        drybelt_window = np.exp(-((_drybelt_lat_deg - DRYBELT_CENTER_DEG) / 8.0) ** 2).astype(np.float32, copy=False)
        # The Gaussian identifies the dry-belt centre but falls to only 7% by
        # 15 degrees, leaving the southern Sahara/Sahel edge effectively outside
        # the regime.  Use a smooth 16-34 degree core (with 10-16 and 34-38 degree
        # shoulders) for regime decisions; keep `drybelt_window` itself for the
        # older calibrated latitude-shape terms below.
        _belt_rise = np.clip((_drybelt_lat_deg - 10.0) / 6.0, 0.0, 1.0)
        _belt_rise = _belt_rise * _belt_rise * (3.0 - 2.0 * _belt_rise)
        _belt_fall = np.clip((38.0 - _drybelt_lat_deg) / 4.0, 0.0, 1.0)
        _belt_fall = _belt_fall * _belt_fall * (3.0 - 2.0 * _belt_fall)
        drybelt_regime_window = np.maximum(
            drybelt_window, (_belt_rise * _belt_fall).astype(np.float32)
        )
        if _sc is not None:
            _sc["storm_window"] = storm_window
            _sc["drybelt_window"] = drybelt_window
            _sc["drybelt_regime_window"] = drybelt_regime_window

    if _sc is not None and "land_mask" in _sc:
        land_mask, sea_mask = _sc["land_mask"], _sc["sea_mask"]
        land_f, sea_f = _sc["land_f"], _sc["sea_f"]
        monsoon_margin_factor = _sc["monsoon_margin_factor"]
    else:
        land_mask, sea_mask = _derive_land_sea_masks(elev)
        land_f = land_mask.astype(np.float32)
        sea_f = sea_mask.astype(np.float32)
        # East-coast/monsoon-margin proxy (2026-07-30, se-us-east-asia-drybelt
        # fix): real subtropical highs sit over the EASTERN ocean basins
        # (Azores/Bermuda high off NW Africa, NE Pacific high off California),
        # so their clockwise (NH; mirrored SH) circulation drives dry subsiding
        # air down the WESTERN continental margins (Sahara, Mexico/SW US,
        # Kalahari/Namib, Atacama, Australian interior -- everything
        # `coastal_upwelling_fog_strength` above already targets) while pumping
        # warm moist air up the EASTERN continental margins from the western
        # boundary currents (Gulf Stream -> SE US, Kuroshio -> East China/
        # Japan). `drybelt_window` below has no way to tell these apart --
        # it's pure |latitude|, so it was suppressing SE US/East China/S Japan
        # (measured: subsidence_suppression as low as 0.14-0.28, at or below
        # actual Sahara's 0.24 on `saves/test.npz`) as hard as real deserts at
        # the same latitude, misclassifying all three as BSh/BSk against the
        # reference Koppen map (real classification: Cfa). This mask mirrors
        # `_west_coast_land` below (ocean-adjacency test) but decays much
        # further inland -- monsoon moisture penetrates whole river basins
        # (Yangtze, Mississippi/Gulf watershed), not just a coastal fog band --
        # and is applied as a REDUCTION of `drybelt_window`'s penalty (see
        # `monsoon_east_margin_exemption`), not real monsoon-circulation
        # physics, the same diagnostic-gate pattern the fog mechanism uses.
        _ocean_east = np.roll(sea_mask, -1, axis=1)
        _coast_core_east = (land_mask & _ocean_east).astype(np.float32)
        # Keep the inland reach geographic rather than grid-cell based.  The
        # original 20 cells represented about 7 degrees at the 1024-column
        # calibration grid, but silently expanded to 56 degrees in the tracked
        # 128-column validation fixture (reaching across southern Africa into
        # the Kalahari).  Scale both reach and per-cell decay so changing
        # resolution preserves a calibrated ~10-degree physical longitude
        # footprint (long enough to cover the East China box without crossing
        # southern Africa into the Kalahari).
        _reference_width = 1024.0
        _monsoon_decay_cells = max(1, int(round(30.0 * W / _reference_width)))
        _monsoon_decay_rate = np.float32(0.88 ** (_reference_width / W))
        monsoon_margin_factor = np.zeros_like(_coast_core_east)
        _shifted = _coast_core_east
        _weight = np.float32(1.0)
        for _ in range(_monsoon_decay_cells + 1):
            monsoon_margin_factor += _weight * _shifted
            _shifted = np.roll(_shifted, -1, axis=1)
            _weight *= _monsoon_decay_rate
        monsoon_margin_factor = np.clip(monsoon_margin_factor, 0.0, 1.0) * land_f
        if _sc is not None:
            for _k, _v in (("land_mask", land_mask), ("sea_mask", sea_mask),
                           ("land_f", land_f), ("sea_f", sea_f),
                           ("monsoon_margin_factor", monsoon_margin_factor)):
                _v.flags.writeable = False
                _sc[_k] = _v

    if temperature is None:
        lat = np.deg2rad(lat_deg).astype(np.float32, copy=False)
        T_lat = temperature_kelvin_for_lat(lat, day_of_year=day_of_year)
        temperature = np.repeat(T_lat[:, None], W, axis=1).astype(np.float32, copy=False)
    else:
        temperature = temperature.astype(np.float32, copy=False)

    u: np.ndarray
    v: np.ndarray
    if wind_u is None or wind_v is None:
        u, v = generate_wind_field(
            H,
            W,
            day_of_year=day_of_year,
            block_size=1,
            elevation=elev,
            debug_log=False,
        )
    else:
        u = wind_u.astype(np.float32, copy=False)
        v = wind_v.astype(np.float32, copy=False)

    wind_speed = np.sqrt(u * u + v * v) + 1e-6
    temp_norm = np.clip((temperature - 255.0) / 45.0, 0.0, 1.0)
    # Hoisted from the moisture-advection block below (cached on (H, W, pp), so
    # this is free): the SST coupling gate inside the subsidence block needs the
    # same metric grids, and that block runs first.
    _lat_2d_grid, dx_grid, dy_grid, _f_grid, _eq_window_grid, _lon_1d_grid = _wind_static_grids(H, W, pp)

    # Upwind SST anomaly (audit D3), built once and consumed twice: by the
    # suppression gate in the subsidence block below, and by the target-share
    # blend in the moisture budget. Those sit on opposite sides of the row
    # rescale, which is the distinction process note 9 exists for -- see
    # `sst_land_target_weight`.
    _sst_strength = float(getattr(pp, "sst_land_coupling_strength", 0.0))
    _sst_target_w = float(getattr(pp, "sst_land_target_weight", 0.0))
    _sst_anom_field = None
    if (
        _sst_strength != 0.0
        or _sst_target_w != 0.0
        or debug_fields is not None
    ) and np.any(sea_mask) and np.any(land_mask):
        if _sc is not None and "sst_anom" in _sc:
            _sst_anom_field = _sc["sst_anom"]
        else:
            _sst_anom_field = _upwind_sst_anomaly(
                temperature,
                sea_f,
                u,
                v,
                dx_grid,
                dy_grid,
                float(pp.sst_land_coupling_km),
            )
            if _sc is not None:
                _sst_anom_field.flags.writeable = False
                _sc["sst_anom"] = _sst_anom_field

    if _sc is not None and "qsat" in _sc:
        qsat = _sc["qsat"]
    else:
        Tc = np.clip(temperature - 273.15, -60.0, 60.0)
        es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))
        qsat = np.clip(0.622 * es / surface_pressure_hpa, 0.0, 0.035).astype(np.float32, copy=False)
        if _sc is not None:
            qsat.flags.writeable = False
            _sc["qsat"] = qsat

    # Subsidence/dry-belt suppression, computed early (needs only u/v/lat, no q)
    # so it can also gate land evapotranspiration below, not just precip_potential
    # further down (which reuses this same array unchanged). Moved up because
    # desert land_evap was riding up with temperature/qsat alone (hot desert air
    # has high evaporative demand) with nothing holding it down: the soil-moisture
    # bucket doesn't reliably differentiate desert from continental-interior land
    # (both settle near the same low floor -- see known-physics-gaps.md item 3
    # UPDATE 2), so absolute humidity q came out HIGHER over deserts than over
    # continental interior despite precip_potential ranking them correctly.
    # Final rainfall is dq = remove_frac * q, an absolute quantity, so the
    # inflated desert q dominated over the correctly-ranked precip_potential
    # (measured directly: Sahara q ~3.5x Canadian Prairies' despite Prairies'
    # precip_potential being ~2.4x Sahara's -- see UPDATE 4).
    #
    # Meridional term sign: row index increases southward while v is northward,
    # so physical divergence is ∂u/∂x − ∂v/∂i. With the previous (+) sign the
    # ITCZ's meridional convergence registered as DIVERGENCE (and the horse
    # latitudes' divergence as convergence), inverting the ascent/subsidence
    # drivers wherever the meridional wind dominated.
    if _sc is not None and "div" in _sc:
        div = _sc["div"]
        subsidence_suppression = _sc["subsidence_suppression"]
    else:
        div = _ddx_periodic(u) - np.gradient(v, axis=0)
        _div_for_subsidence = div
        _regime_gate_strength = float(getattr(pp, "subsidence_divergence_regime_gate", 0.0))
        if _regime_gate_strength > 0.0:
            # Split divergence into the latitude-row background and the local
            # departure from it.  B1's displaced circulation contaminates the
            # Midwest almost entirely through the former (85% of its signal),
            # whereas local departures still carry useful terrain/synoptic
            # information.  Attenuate only that background away from the true
            # subtropical belt; retaining the local term avoids the previous
            # latitude-only gate's broad desert-wetting side effect.
            _div_zonal_mean = np.mean(div, axis=1, dtype=np.float64).astype(np.float32)
            _div_local = div - _div_zonal_mean[:, None]
            _zonal_keep = (
                (1.0 - _regime_gate_strength)
                + _regime_gate_strength * drybelt_regime_window
            )
            _div_for_subsidence = _div_local + _div_zonal_mean[:, None] * _zonal_keep[:, None]
        _div_pos_early = np.clip(_div_for_subsidence, 0.0, None)
        # Cap each cell's contribution to the normalizer's reference mean (not
        # the numerator -- genuinely-more-subsiding cells still rank higher).
        # `wind_prognostic_substep_days` defaulting to 1.0 (2026-07-28, see
        # razor-sharp-biome-line-precip-target-smoothing memory) routes MONTHLY/
        # ANNUAL wind through the real prognostic evolve_wind solver instead of
        # generate_wind_field's smoothed diagnostic snapshot. The prognostic
        # divergence field has a much heavier tail (p90/p99 measured 0.222/0.243
        # -> 0.281/0.429) even though its bulk barely moves, which inflates this
        # array's global mean ~37% and dilutes subsidence_suppression EVERYWHERE
        # -- including over real deserts whose own local divergence never
        # changed (2026-07-29, desert-wetting-regression memory: Sahara/Kalahari/
        # Atacama all measurably wetter after the substep-days flip, isolated via
        # scripts/run_real_terrain_validation.py --param sweeps to this mechanism
        # specifically, not the target-smoothing or coastal-fog work from the
        # same day). 0.02 was chosen empirically against that validation script
        # (real desert-local div, 0.16-0.28, still saturates distinctly above
        # it); lower caps close the desert gap further but monotonically cost
        # already-under-target continental interior too (US Midwest 715->648
        # mm/yr across the sweep), so this is a real trade-off, not a free lever
        # -- deliberately not pushed past 0.02.
        _div_pos_norm_ref = np.minimum(_div_pos_early, 0.02)
        _subsidence_norm_early = np.clip(_div_pos_early / (np.mean(_div_pos_norm_ref) + 1e-6), 0.0, 2.5)
        subsidence_suppression = np.clip(
            1.0 - 0.34 * _subsidence_norm_early - 1.00 * drybelt_regime_window[:, None],
            0.02,
            1.0,
        ).astype(np.float32, copy=False)
        # `ascent`/`conv`/`orog` (computed further down, reusing `div` cached
        # here) each get a `+0.15*laplacian(...)` smoothing pass; this raw-div-
        # derived driver was the one that skipped it (2026-07-28,
        # tropical-speckle-fix memory) -- dormant while MONTHLY/ANNUAL used the
        # smoothed diagnostic wind, but a real, sustained source of grid-scale
        # speckle in `land_evap`/`precip_potential` once the prognostic solver
        # (unsmoothed `p_anom`, see `evolve_wind`) took over by default.
        _lap_early = _laplacian_numba if NUMBA_AVAILABLE else _laplacian
        subsidence_suppression = np.clip(
            subsidence_suppression + 0.15 * _lap_early(subsidence_suppression), 0.02, 1.0
        ).astype(np.float32, copy=False)
        # ITCZ row-to-row/column-to-column shape consistency (2026-07-29, see
        # PlanetParams.itcz_zonal_smooth_deg docstring for the full mechanism
        # and root-cause measurement). Longitude-only Gaussian smoothing of
        # the shared `subsidence_suppression` signal, applied before the
        # coastal-fog gate (deliberately narrow, would be washed out by a
        # wide smooth) but after the local Laplacian pass above (a much
        # smaller-scale smooth that doesn't address the Rossby-wave-period
        # noise this targets). 0.0 default is an exact no-op.
        _itcz_smooth_deg = float(getattr(pp, "itcz_zonal_smooth_deg", 0.0))
        if _itcz_smooth_deg > 0.0:
            subsidence_suppression = np.clip(
                _zonal_gaussian_smooth(subsidence_suppression, _itcz_smooth_deg), 0.02, 1.0
            ).astype(np.float32, copy=False)
        # Coastal-fog/cold-current desert suppression (2026-07-27,
        # coastal-fog-desert memory; reconstructed from that memory's writeup
        # after an accidental `git checkout` discarded the uncommitted
        # original -- functionally faithful, not guaranteed byte-identical).
        # Real Atacama/Namib-class deserts are this dry despite sitting on a
        # coast because eastern-boundary-current upwelling (Humboldt,
        # Benguela, California, Canary) cools coastal SST enough to trap
        # moisture under a marine fog inversion instead of releasing it as
        # rain -- a mechanism this model has no ocean-current/upwelling
        # physics for at all (`ocean.calculate_ocean_heat_transport` models
        # western-boundary-current *warming* but has no eastern-boundary
        # *cooling* counterpart). Implemented as a diagnostic gate on
        # `subsidence_suppression` (and therefore both `land_evap` and
        # `precip_potential`, which both read it) rather than real SST-driven
        # physics -- building genuine upwelling coupling was judged too large
        # a lift for the coastal-fog session alone.
        #
        # `_west_coast_land`: land cells with ocean immediately to their west
        # (`np.roll(sea_mask, 1, axis=1)` -- the real eastern-boundary-current
        # coastlines above are all west coasts), decayed over the two cells
        # inland (east) the same "1-2 cells downstream" pattern
        # `ocean.calculate_ocean_heat_transport`'s western-boundary-current
        # enhancement already uses. An immediate-coastline-only mask was
        # tried first and measured to touch only 13% of the Atacama named
        # box's land cells (Atacama moved <3 mm/yr across the full strength
        # range); decaying 2 cells inland raised coverage to ~40% of cells at
        # meaningful weight (mean 0.28).
        _ocean_west = np.roll(sea_mask, 1, axis=1)
        _coast_core = (land_mask & _ocean_west).astype(np.float32)
        _west_coast_land = np.clip(
            _coast_core
            + 0.6 * np.roll(_coast_core, 1, axis=1)  # 1 cell inland
            + 0.3 * np.roll(_coast_core, 2, axis=1),  # 2 cells inland
            0.0,
            1.0,
        )
        _fog_gate = np.clip(
            1.0 - float(pp.coastal_upwelling_fog_strength) * _west_coast_land * drybelt_window[:, None],
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        subsidence_suppression = (subsidence_suppression * _fog_gate).astype(np.float32, copy=False)
        # SST -> adjacent-land climate coupling (2026-08-05, audit D3).
        #
        # The gate above is a *geometry* proxy: west-facing coast x subtropical
        # latitude, with no ocean temperature in it at all. D5 then built the
        # real physics it stands in for -- a wind-stress-curl gyre solve that
        # produces genuine eastern-boundary cooling (Benguela -0.54 K, Canary
        # -0.36 K, Humboldt -0.25 K) -- and measured that Atacama precipitation
        # does not move by a single mm/yr across gyre strengths 0.0 to 3.0.
        # D3's conclusion was that the missing piece is not the ocean cooling
        # but the coupling, and that any future attempt should build *this*
        # first and verify it transmits, or it will reproduce that null result.
        #
        # This is that coupling. Cold water upwind stabilises the boundary layer
        # it feeds (a marine inversion caps convection, which is why Atacama and
        # the Namib are hyper-arid *on a coast*); warm water upwind destabilises
        # it and loads it with moisture (Gulf Stream, Kuroshio). Both signs come
        # from one term, applied to `subsidence_suppression` -- deliberately the
        # same array the fog gate uses, because that is the one pathway in this
        # model already *demonstrated* to reach land precipitation (the fog gate
        # moved Atacama 123 -> 102 mm/yr). Per-cell ocean evaporation already
        # responds to SST through `qsat`, and the moisture budget's row rescale
        # erases it again; process note 9.
        #
        # Deliberately composed with the fog gate rather than replacing it: that
        # knob is calibrated and carries A1's Atacama result, so retiring it is a
        # separate decision from establishing whether this transmits.
        #
        # **This half is measured to be inert on Earth, and both reasons are
        # worth knowing before reaching for it again.** (1) In the deserts it
        # was aimed at, `subsidence_suppression` is already 0.02-0.04, i.e. on
        # its own floor after A5's regime architecture -- a multiplicative
        # drying gate has no headroom left there at all, which also means the
        # fog gate's documented "Atacama 123 -> 102 mm/yr" would not reproduce
        # today. (2) Where there *is* headroom the moisture budget's row rescale
        # absorbs the change: process note 9. The target-side companion
        # (`sst_land_target_weight`) is the half that survives.
        if _sst_strength != 0.0 and _sst_anom_field is not None:
            # Land only. An ocean cell's own suppression is not what this term
            # is about, and letting it feed back on the water that produced the
            # anomaly would be a self-reinforcing loop with no physical
            # counterpart.
            _sst_gate = np.clip(
                1.0
                + _sst_strength
                * np.tanh(_sst_anom_field / _SST_COUPLING_REFERENCE_K)
                * land_f,
                0.25,
                2.0,
            ).astype(np.float32, copy=False)
            subsidence_suppression = np.clip(
                subsidence_suppression * _sst_gate, 0.02, 1.0
            ).astype(np.float32, copy=False)
            if debug_fields is not None:
                debug_fields["sst_land_gate"] = _sst_gate
        if debug_fields is not None and _sst_anom_field is not None:
            debug_fields["upwind_sst_anomaly"] = _sst_anom_field
        # East-coast/monsoon-margin exemption (2026-07-30, se-us-east-asia-
        # drybelt fix; see `monsoon_margin_factor` above for the mechanism).
        # Applied post-hoc, AFTER `itcz_zonal_smooth_deg`'s wide (32-deg
        # radius) periodic longitude smoothing rather than folded into the
        # per-cell formula upstream of it -- measured directly: baking the
        # exemption into the pre-smoothing formula let the zonal smooth
        # average it away against the much larger swath of unexempted ocean
        # and non-coastal land sharing the same latitude circle (SE US
        # subsidence_suppression moved only 0.197->0.219 at full strength=1.0
        # that way, on `saves/test.npz`). Adding it back afterward, mirroring
        # where `_fog_gate` sits for the identical reason, recovers up to the
        # full `0.45*drybelt_window` penalty this cell would otherwise carry,
        # scaled by how much of a real east-facing monsoon margin the cell
        # actually is (`monsoon_margin_factor`) and by
        # `monsoon_east_margin_exemption` itself (0.0 = exact no-op).
        _monsoon_recover = (
            1.00 * drybelt_regime_window[:, None]
            * monsoon_margin_factor
            * float(pp.monsoon_east_margin_exemption)
        )
        subsidence_suppression = np.clip(
            subsidence_suppression + _monsoon_recover, 0.02, 1.0
        ).astype(np.float32, copy=False)
        if _sc is not None:
            div.flags.writeable = False
            subsidence_suppression.flags.writeable = False
            _sc["div"] = div
            _sc["subsidence_suppression"] = subsidence_suppression
    if debug_fields is not None:
        debug_fields["subsidence_suppression"] = subsidence_suppression
        # The latitude windows themselves, so a seasonal-migration change is
        # inspectable directly rather than only through what it does to
        # `subsidence_suppression` (which also carries the divergence field and
        # is smoothed, so a window defect is not cleanly readable there).
        debug_fields["drybelt_window"] = drybelt_window
        debug_fields["drybelt_regime_window"] = drybelt_regime_window
        debug_fields["storm_window"] = storm_window
        debug_fields["itcz_window"] = itcz_window

    if humidity is None:
        base_q = np.where(sea_mask, 0.013, 0.009).astype(np.float32, copy=False)
    else:
        base_q = humidity.astype(np.float32, copy=False)
    convective_closure_active = (
        bool(pp.enable_stability_aware_condensation)
        or bool(pp.enable_simplified_betts_miller_convection)
    )
    two_layer_active = (
        bool(pp.enable_prognostic_column_water)
        and convective_closure_active
        and bool(pp.enable_two_layer_convective_adjustment)
    )
    three_level_active = two_layer_active and bool(pp.enable_three_level_pressure_column)
    closed_three_level_active = (
        three_level_active and bool(pp.enable_closed_three_level_thermodynamics)
    )
    closed_lower_temperature_next: np.ndarray | None = None
    closed_column_water_residual = 0.0
    closed_column_energy_residual = 0.0
    diabatic_interface_mass_flux_step = None
    if two_layer_active:
        # The legacy experiment stores one total-humidity proxy and partitions
        # it afresh.  The closed path instead persists a lower-layer mixing
        # ratio plus two independent layer values.  On its first call, migrate
        # the legacy total into the documented 0.40/0.35/0.25 pressure masses.
        _closed_first_step = closed_three_level_active and (
            midlevel_humidity is None or upperlevel_humidity is None
        )
        upper_q_in = (
            base_q * (0.35 if _closed_first_step else float(pp.two_layer_upper_humidity_fraction))
            if midlevel_humidity is None
            else np.asarray(midlevel_humidity, dtype=np.float32)
        )
        if upper_q_in.shape != base_q.shape:
            raise ValueError("midlevel_humidity must match precipitation grid")
        upper_q_in = np.clip(upper_q_in, 0.0, base_q)
        upperlevel_q_in = (
            base_q * (0.25 if _closed_first_step else float(pp.three_level_upper_humidity_fraction))
            if three_level_active and upperlevel_humidity is None
            else np.zeros_like(base_q)
            if not three_level_active
            else np.asarray(upperlevel_humidity, dtype=np.float32)
        )
        if upperlevel_q_in.shape != base_q.shape:
            raise ValueError("upperlevel_humidity must match precipitation grid")
        upperlevel_q_in = np.clip(upperlevel_q_in, 0.0, base_q - upper_q_in)
        lower_base_q = (
            base_q * 0.40 if _closed_first_step
            else base_q if closed_three_level_active
            else base_q - upper_q_in - upperlevel_q_in
        )
    else:
        upper_q_in = np.zeros_like(base_q)
        upperlevel_q_in = np.zeros_like(base_q)
        lower_base_q = base_q
    midlevel_humidity_next = midlevel_humidity
    upperlevel_humidity_next = upperlevel_humidity

    if soil_moisture is None:
        soil = np.where(land_mask, 0.55, 0.0).astype(np.float32, copy=False)
    else:
        soil = soil_moisture.astype(np.float32)

    if soil_moisture_deep is None:
        soil_deep = np.where(land_mask, 0.3, 0.0).astype(np.float32, copy=False)
    else:
        soil_deep = soil_moisture_deep.astype(np.float32)

    dt = max(float(dt_days), 1.0)
    # At large dt (monthly/annual mode) evaporation would saturate the entire humidity
    # field in one step, erasing the spatial gradients that determine climate zones.
    # Cap evaporation at 1.5-day equivalent so dry and wet cells stay differentiated
    # across substeps even when dt=6 (monthly) or dt=7 (annual).
    dt_evap = min(dt, 1.5)

    # Evaporation and evapotranspiration sources
    wind_norm = np.clip(wind_speed / 15.0, 0.0, 1.5)
    ocean_evap = evap_coeff * sea_f * (0.45 + 0.55 * wind_norm) * np.clip(qsat - lower_base_q, 0.0, None)
    # Soil factor draws from whichever layer has more effective moisture: the fast
    # surface layer directly, or the slow deep/root-zone reservoir at reduced
    # efficiency (root uptake) via soil_deep_evap_weight. Lets deep moisture
    # "rescue" evaporation when the surface is dry without needing to dominate when
    # the surface is already adequately moist -- see PlanetParams.soil_deep_evap_weight.
    soil_evap_factor = np.maximum(soil, float(pp.soil_deep_evap_weight) * soil_deep)
    land_evap = (
        evap_coeff
        * land_f
        * (0.20 + 0.65 * temp_norm)
        * (0.35 + 0.65 * soil_evap_factor)
        * np.clip(qsat - lower_base_q, 0.0, None)
        * subsidence_suppression
    )
    if (
        bool(pp.enable_prognostic_column_water)
        and bool(pp.enable_energy_limited_evaporation)
    ):
        # A conserved moisture column needs an independent energy constraint at
        # its lower boundary.  The humidity-deficit bulk flux above is useful
        # for *partitioning* ocean/land evaporation, but on its own permits an
        # arbitrarily large latent flux whenever circulation exports vapor.  A
        # daily-mean top-of-atmosphere insolation calculation supplies the
        # available surface shortwave; atmospheric transmission, albedo, and
        # the latent-heat share then give an upper bound in the same mm/day
        # column units as the moisture source.
        _lat_rad = np.radians(lat_deg.astype(np.float64))
        _declination = float(pp.solar_declination(day_of_year))
        _hour_angle = np.arccos(np.clip(
            -np.tan(_lat_rad) * math.tan(_declination), -1.0, 1.0
        ))
        _daily_toa_w_m2 = (
            float(pp.solar_constant) / math.pi
            * (
                _hour_angle * np.sin(_lat_rad) * math.sin(_declination)
                + np.cos(_lat_rad) * math.cos(_declination) * np.sin(_hour_angle)
            )
        )[:, None]
        _surface_albedo = 0.06 * sea_f + 0.20 * land_f
        _cloud_for_evap = (
            np.zeros_like(lower_base_q)
            if cloud_fraction is None
            else np.clip(np.asarray(cloud_fraction, dtype=np.float32), 0.0, 1.0)
        )
        _longwave_increment_w_m2 = np.full_like(
            lower_base_q,
            float(max(pp.evaporation_downwelling_longwave_w_m2, 0.0)),
            dtype=np.float32,
        )
        if bool(pp.enable_humidity_dependent_downwelling_longwave):
            # Brutsaert's clear-sky emissivity relation uses vapour pressure in
            # hPa.  The shortwave-only cap already represents a dry reference
            # surface-energy budget, so add only the emissivity excess above
            # that reference rather than double-counting the full atmospheric
            # longwave flux.  Cloud fills part of the remaining emissivity
            # deficit, approaching a blackbody sky as cloud fraction grows.
            _vapour_pressure_hpa = (
                np.clip(lower_base_q, 0.0, None)
                * float(pp.surface_pressure_pa)
                / (0.622 + 0.378 * np.clip(lower_base_q, 0.0, None))
                / 100.0
            )
            _clear_emissivity = np.clip(
                1.24 * np.power(
                    _vapour_pressure_hpa / np.maximum(temperature, 180.0),
                    1.0 / 7.0,
                ),
                float(np.clip(pp.evaporation_longwave_clear_sky_emissivity_floor, 0.0, 1.0)),
                1.0,
            )
            _sky_emissivity = np.clip(
                _clear_emissivity
                + (1.0 - _clear_emissivity)
                * _cloud_for_evap
                * float(np.clip(pp.evaporation_longwave_cloud_emissivity_weight, 0.0, 1.0)),
                0.0,
                1.0,
            )
            _longwave_increment_w_m2 = _longwave_increment_w_m2 + np.maximum(
                _sky_emissivity
                - float(np.clip(pp.evaporation_longwave_reference_emissivity, 0.0, 1.0)),
                0.0,
            ) * (5.670374419e-8 * np.maximum(temperature, 180.0) ** 4)

        _surface_energy_w_m2 = (
            _daily_toa_w_m2
            * float(np.clip(pp.evaporation_surface_shortwave_transmissivity, 0.0, 1.0))
            * (1.0 - _surface_albedo)
            * (1.0 - 0.50 * _cloud_for_evap)
            + _longwave_increment_w_m2
        )
        _latent_cap_mm_day = np.maximum(
            _surface_energy_w_m2
            * float(np.clip(pp.evaporation_latent_energy_fraction, 0.0, 1.0))
            * 86400.0 / 2.5e6,
            0.0,
        )
        _unconstrained_evap_q_day = ocean_evap + land_evap
        _energy_cap_q_day = _latent_cap_mm_day / 2000.0
        _energy_fraction = np.minimum(
            1.0,
            _energy_cap_q_day / (_unconstrained_evap_q_day + 1e-12),
        )
        ocean_evap = ocean_evap * _energy_fraction
        land_evap = land_evap * _energy_fraction
    else:
        _latent_cap_mm_day = None
        _energy_fraction = None
    sources = (ocean_evap + land_evap) * dt_evap
    q = np.clip(lower_base_q + sources, 0.0, qsat)
    if debug_fields is not None:
        debug_fields["temp_norm"] = temp_norm
        debug_fields["qsat"] = qsat
        debug_fields["base_q"] = base_q
        debug_fields["soil"] = soil
        debug_fields["soil_deep"] = soil_deep
        debug_fields["soil_evap_factor"] = soil_evap_factor
        debug_fields["land_evap"] = land_evap
        debug_fields["ocean_evap"] = ocean_evap
        # Explicit rates make the regional moisture-budget report independent
        # of the caller's precipitation timestep. The legacy q/day source
        # fields above are retained for existing low-level diagnostics.
        debug_fields["land_evaporation_mm_day"] = (
            land_evap * 2000.0
        ).astype(np.float32, copy=False)
        debug_fields["ocean_evaporation_mm_day"] = (
            ocean_evap * 2000.0
        ).astype(np.float32, copy=False)
        if _latent_cap_mm_day is not None:
            debug_fields["energy_limited_evaporation_cap_mm_day"] = (
                _latent_cap_mm_day.astype(np.float32)
            )
            debug_fields["energy_limited_evaporation_fraction"] = (
                _energy_fraction.astype(np.float32)
            )
            debug_fields["energy_limited_downwelling_longwave_w_m2"] = (
                _longwave_increment_w_m2.astype(np.float32)
            )

    # Moisture advection: hybrid of the old short-range donor-cell blend
    # (kept as the base term) plus an additional longer-range transport
    # contribution, blended in by `moisture_advection_scale`.
    #
    # A full replacement with pure transport (moving moisture the physically-
    # correct wind_speed*dt distance) was tried first and reverted: measured
    # on a 60yr MONTHLY real-terrain-shaped spinup, it monotonically *dried
    # out* mid-latitude continental-interior land as transport distance
    # increased (soil moisture 0.335 -> 0.126, precip 106 -> 56 mm/yr from
    # scale 0 -> 1), the opposite of the intended fix. Best diagnosis: the
    # short-range blend was accidentally load-bearing -- it keeps evaporated
    # moisture from moving away faster than the RH/convergence-based precip
    # trigger (rh_release, convective) can capture it near its source; genuine
    # long-range transport instead exports land moisture onward before it
    # rains out locally, since land evaporation is soil-bucket-limited (finite
    # source) unlike the ocean's effectively unlimited supply. Kept as an
    # additive nudge (not a replacement) so the load-bearing short-range
    # dynamics stay intact; `moisture_advection_scale` controls how much of
    # the longer-range contribution blends in (0 = old behavior exactly).
    #
    # The long-range term itself was originally a single semi-Lagrangian
    # backward-trajectory sample covering the whole substep's distance at
    # once (`_advect_scalar_semi_lagrangian`) -- found (2026-07 moisture-
    # advection-jump-dilution investigation, see project memory) to dilute
    # even the *ocean* source cells at real MONTHLY-mode substep dt (~7.6
    # days: a ~5000km single jump, coastal RH 100%->66% at scale 0->0.7)
    # before land ever enters the picture, because one huge jump samples an
    # effectively uncorrelated point rather than believable local transport.
    # Replaced with `_advect_scalar_flux_eulerian`: a CFL-safe Eulerian upwind
    # advection scheme that integrates many small substeps instead of one huge
    # jump (see its docstring). Verified directly against real terrain
    # (saves/earth.pkl, same transect the diagnosing session used): ocean-cell
    # RH held at 92%/93.6%/94.2% across scale 0/0.3/0.7, vs. the old scheme's
    # 92%/78.6%/59.3% collapse. This fixes the transport mechanism's own
    # correctness; it does not by itself resolve whether transport helps the
    # continental-interior/desert gap above -- that remains a separate, open
    # question (the RH-trigger-favors-local-moisture mechanism described in
    # the paragraph above still applies).
    # (`dx_grid`/`dy_grid` are computed once, further up, where the SST coupling
    # gate also needs them; `_wind_static_grids` is cached on (H, W, pp) anyway.)
    if _sc is not None and "u_scale" in _sc:
        u_scale, v_scale = _sc["u_scale"], _sc["v_scale"]
    else:
        if pp.humidity_advection_cfl:
            # Real Courant number (matches the long-range term's own CFL check
            # below) instead of the fixed |u|/20, |v|/12 divisors.
            courant_u = np.abs(u) * dt * 86400.0 / dx_grid
            courant_v = np.abs(v) * dt * 86400.0 / dy_grid
            u_scale = np.clip(courant_u, 0.0, 1.0) * (0.32 + 0.16 * storm_window[:, None])
            v_scale = np.clip(courant_v, 0.0, 1.0) * (
                0.34 + 0.06 * drybelt_window[:, None] + 0.16 * storm_window[:, None]
            )
        else:
            u_scale = np.clip(np.abs(u) / 20.0, 0.0, 1.0) * (0.32 + 0.16 * storm_window[:, None])
            v_scale = np.clip(np.abs(v) / 12.0, 0.0, 1.0) * (
                0.34 + 0.06 * drybelt_window[:, None] + 0.16 * storm_window[:, None]
            )
        if _sc is not None:
            u_scale = u_scale.astype(np.float32, copy=False)
            v_scale = v_scale.astype(np.float32, copy=False)
            u_scale.flags.writeable = False
            v_scale.flags.writeable = False
            _sc["u_scale"] = u_scale
            _sc["v_scale"] = v_scale
    q_short = q
    if pp.enable_prognostic_column_water:
        # The experimental path transports actual column depth, not the
        # legacy donor-cell blending proxy.  The source is the amount that
        # survived the same local saturation limiter used by the established
        # evaporation code, so no new water is invented at this boundary.
        # Saturation limiting can make ``q`` lower than its carried-in value
        # over a cold cell.  That is not negative evaporation: the explicit
        # supersaturation rainout below owns that sink.  Passing it here as a
        # source would double-remove water and, after the transport kernel's
        # positivity guard, silently violate the column budget.
        effective_evaporation_q_day = np.maximum(q - lower_base_q, 0.0) / dt
        area_m2, x_face_length_m, y_face_length_m = _column_water_spherical_geometry(
            H, W, float(pp.radius_m)
        )
        column_step = evolve_column_water(
            lower_base_q * 2000.0,
            effective_evaporation_q_day * 2000.0,
            np.zeros_like(q),
            u,
            v,
            dx_m=dx_grid,
            dy_m=float(dy_grid),
            dt_days=dt,
            cell_area_m2=area_m2,
            x_face_length_m=x_face_length_m,
            y_face_length_m=y_face_length_m,
        )
        q_short = column_step.water_mm / 2000.0
        if two_layer_active:
            upper_u = u if wind_u_aloft is None else np.asarray(wind_u_aloft)
            upper_v = v if wind_v_aloft is None else np.asarray(wind_v_aloft)
            if upper_u.shape != q.shape or upper_v.shape != q.shape:
                raise ValueError("upper-level wind fields must match precipitation grid")
            mid_u = (
                np.asarray(wind_u_midlevel)
                if three_level_active and wind_u_midlevel is not None
                else 0.5 * (u + upper_u)
                if three_level_active
                else upper_u
            )
            mid_v = (
                np.asarray(wind_v_midlevel)
                if three_level_active and wind_v_midlevel is not None
                else 0.5 * (v + upper_v)
                if three_level_active
                else upper_v
            )
            if mid_u.shape != q.shape or mid_v.shape != q.shape:
                raise ValueError("midlevel wind fields must match precipitation grid")
            upper_step = evolve_column_water(
                upper_q_in * 2000.0,
                np.zeros_like(q),
                np.zeros_like(q),
                mid_u,
                mid_v,
                dx_m=dx_grid,
                dy_m=float(dy_grid),
                dt_days=dt,
                cell_area_m2=area_m2,
                x_face_length_m=x_face_length_m,
                y_face_length_m=y_face_length_m,
            )
            midlevel_humidity_next = upper_step.water_mm / 2000.0
            if three_level_active:
                upperlevel_step = evolve_column_water(
                    upperlevel_q_in * 2000.0,
                    np.zeros_like(q),
                    np.zeros_like(q),
                    upper_u,
                    upper_v,
                    dx_m=dx_grid,
                    dy_m=float(dy_grid),
                    dt_days=dt,
                    cell_area_m2=area_m2,
                    x_face_length_m=x_face_length_m,
                    y_face_length_m=y_face_length_m,
                )
                upperlevel_humidity_next = upperlevel_step.water_mm / 2000.0
        if debug_fields is not None:
            debug_fields["column_water_transport_tendency_mm_day"] = (
                column_step.transport_tendency_mm_day
            )
            debug_fields["column_water_transport_residual_mm_m2"] = column_step.residual_mm
            debug_fields["column_water_transport_relative_residual"] = (
                column_step.relative_residual
            )
            if two_layer_active:
                debug_fields["midlevel_humidity_transport_relative_residual"] = (
                    upper_step.relative_residual
                )
                if three_level_active:
                    debug_fields["upperlevel_humidity_transport_relative_residual"] = (
                        upperlevel_step.relative_residual
                    )
    elif NUMBA_AVAILABLE:
        for _ in range(3):
            # _advect_humidity_numba always allocates a fresh output (np.zeros_like
            # internally), so q_short is a unique array from here on -- safe to mutate
            # in place below instead of allocating a new array for the add and another
            # for the clip.
            q_short = _advect_humidity_numba(q_short.astype(np.float32), u, v, u_scale, v_scale)
            lap_q = _laplacian_numba(q_short)
            # Adaptive diffusion: stronger in regions with sharp gradients
            q_grad_strength = np.abs(lap_q) / (np.mean(np.abs(lap_q)) + 1e-9)
            diffusion_coeff = (0.11 + 0.03 * storm_window[:, None]) * (
                1.0 + 0.3 * np.clip(q_grad_strength, 0.0, 2.0)
            )
            np.add(q_short, diffusion_coeff * lap_q, out=q_short)
            np.clip(q_short, 0.0, qsat, out=q_short)
    else:
        for _ in range(3):
            # _advect_scalar always returns a fresh array -- same in-place reasoning
            # as the Numba branch above.
            q_short = _advect_scalar(q_short, u, v, u_scale, v_scale)
            lap_q = _laplacian(q_short)
            np.add(q_short, (0.11 + 0.03 * storm_window[:, None]) * lap_q, out=q_short)
            np.clip(q_short, 0.0, qsat, out=q_short)

    _blend = float(pp.moisture_advection_scale)
    if _blend > 0.0 and not pp.enable_prognostic_column_water:
        q_long = _advect_scalar_flux_eulerian(q, u, v, dt * 86400.0, dx_grid, dy_grid)
        # Gate the long-range contribution by the same subsidence_suppression
        # that already gates land_evap, so imported moisture is damped in
        # descending/dry-belt (desert) cells rather than reaching the
        # convective term's RH>=0.8 trigger ungated. Without this, real-
        # terrain sweeps show the fixed transport wets deserts (Kalahari
        # 167->350, Atacama 57->137 mm/yr at scale 0->1) proportionally more
        # than it helps continental interior -- see
        # moisture-advection-scale-real-terrain-sweep-2026-07 project memory.
        # Only a partial fix (re-measured after adding this gate: Kalahari
        # still 167->306, Atacama 57->122 at scale 0->1 -- reduced overshoot
        # but not eliminated); subsidence_suppression is already in
        # [0.08, 1.0] (1.0 = no suppression), so this only ever damps the
        # blend, never amplifies it.
        #
        # Second gate (moisture-advection-scale-real-terrain-sweep-2026-07
        # follow-up): subsidence_suppression alone doesn't distinguish "this
        # cell is in the drybelt" from "this cell is merely descending" --
        # additionally damp by drybelt_window directly (deserts sit at its
        # peak; continental-interior boxes used in the sweep are near-zero
        # here), since the desert overshoot is disproportionately larger than
        # continental gains, not just proportionally present everywhere.
        _effective_blend = _blend * subsidence_suppression * (1.0 - 0.5 * drybelt_window[:, None])
        q = np.clip((1.0 - _effective_blend) * q_short + _effective_blend * q_long, 0.0, qsat)
    else:
        q = q_short

    # Moisture-flux convergence driver
    if bool(getattr(pp, "spherical_metric_precip", False)):
        # Metric-correct path (opt-in, see PlanetParams.spherical_metric_precip).
        # Same clip/normalise/smooth pipeline as the two legacy branches below --
        # only the divergence operator differs, so an A/B isolates the metric.
        _div_q = flux_divergence_spherical(
            q, u, v, np.radians(lat_deg.astype(np.float64)), radius_m=float(pp.radius_m)
        )
        conv = _normalize_positive_driver(-_div_q)
        _lap_c = _laplacian_numba if NUMBA_AVAILABLE else _laplacian
        conv = np.clip(conv + 0.15 * _lap_c(conv), 0.0, 3.0)
    elif NUMBA_AVAILABLE:
        # Fast path: Numba-accelerated convergence
        conv = _moisture_convergence_numba(q.astype(np.float32), u, v)
        conv = conv / (np.mean(conv) + 1e-6)
        lap_conv = _laplacian_numba(conv)
        conv = np.clip(conv + 0.15 * lap_conv, 0.0, 3.0)
    else:
        # Fallback: original NumPy implementation.
        # Row index increases southward, v is northward: ∂F/∂y_north = -∂F/∂i.
        flux_x = q * u
        flux_y = q * v
        conv = np.clip(-(_ddx_periodic(flux_x) - np.gradient(flux_y, axis=0)), 0.0, None)
        conv = conv / (np.mean(conv) + 1e-6)
        conv = np.clip(conv + 0.15 * _laplacian(conv), 0.0, 3.0)

    # Large-scale ascent proxy from wind convergence. `div` and
    # `subsidence_suppression` (used to gate both this and land_evap above) were
    # already computed early, right after wind/qsat, since neither needs q --
    # see the sign-convention note there.
    _lap = _laplacian_numba if NUMBA_AVAILABLE else _laplacian
    if _sc is not None and "ascent" in _sc:
        ascent = _sc["ascent"]
    else:
        ascent = np.clip(-div, 0.0, None)
        ascent = ascent / (np.mean(ascent) + 1e-6)
        if three_level_active and float(pp.three_level_diabatic_ascent_scale) > 0.0:
            # Tropical radiative/latent heating supplies a resolved diabatic
            # mass-flux contribution in addition to horizontal convergence.
            # This is confined to the migrating ITCZ and remains opt-in.
            ascent = ascent + (
                float(pp.three_level_diabatic_ascent_scale) * itcz_window[:, None]
            )
        ascent = np.clip(ascent + 0.15 * _lap(ascent.astype(np.float32)), 0.0, 3.0)
        if _sc is not None:
            ascent = ascent.astype(np.float32, copy=False)
            ascent.flags.writeable = False
            _sc["ascent"] = ascent

    if two_layer_active:
        # Pressure-coordinate vertical velocity from mass continuity.  With
        # omega positive downward, d(omega)/dp = -div(V); a convergent lower
        # layer beneath a divergent upper layer therefore gives omega < 0
        # (upward motion) at the interface.  The half-layer difference is a
        # compact centred estimate that uses both resolved wind levels instead
        # of treating surface convergence and upper descent as unrelated
        # empirical triggers.
        upper_u_for_omega = u if wind_u_aloft is None else np.asarray(wind_u_aloft)
        upper_v_for_omega = v if wind_v_aloft is None else np.asarray(wind_v_aloft)
        if upper_u_for_omega.shape != q.shape or upper_v_for_omega.shape != q.shape:
            raise ValueError("upper-level wind fields must match precipitation grid")
        _unit_mass = np.ones_like(q, dtype=np.float32)
        _lat_rad = np.radians(lat_deg.astype(np.float64))
        lower_divergence_si = flux_divergence_spherical(
            _unit_mass, u, v, _lat_rad, radius_m=float(pp.radius_m)
        )
        upper_divergence_si = flux_divergence_spherical(
            _unit_mass, upper_u_for_omega, upper_v_for_omega, _lat_rad,
            radius_m=float(pp.radius_m),
        )
        omega_scale = float(pp.two_layer_vertical_velocity_scale_pa_s)
        if three_level_active:
            # The interpolated midlevel wind supplies the third divergence
            # profile. It is deliberately explicit here (rather than hidden in
            # a scalar ascent proxy) so the native three-level path has two
            # independently diagnosed pressure interfaces.
            mid_u_for_omega = (
                np.asarray(wind_u_midlevel)
                if wind_u_midlevel is not None
                else 0.5 * (u + upper_u_for_omega)
            )
            mid_v_for_omega = (
                np.asarray(wind_v_midlevel)
                if wind_v_midlevel is not None
                else 0.5 * (v + upper_v_for_omega)
            )
            mid_divergence_si = flux_divergence_spherical(
                _unit_mass, mid_u_for_omega, mid_v_for_omega, _lat_rad,
                radius_m=float(pp.radius_m),
            )
            _divergence_filter_passes = int(pp.three_level_divergence_filter_passes)
            _divergence_filter_strength = float(pp.three_level_divergence_filter_strength)
            if _divergence_filter_passes > 0 and _divergence_filter_strength > 0.0:
                lower_divergence_si = smooth_spherical_scalar(
                    lower_divergence_si,
                    strength=_divergence_filter_strength,
                    passes=_divergence_filter_passes,
                )
                mid_divergence_si = smooth_spherical_scalar(
                    mid_divergence_si,
                    strength=_divergence_filter_strength,
                    passes=_divergence_filter_passes,
                )
                upper_divergence_si = smooth_spherical_scalar(
                    upper_divergence_si,
                    strength=_divergence_filter_strength,
                    passes=_divergence_filter_passes,
                )
            if closed_three_level_active:
                # The new closure owns its pressure-mass continuity rather
                # than accepting the older experiment's optional algebraic
                # switch.  This makes its layer masses and omegas one contract.
                lower_temperature_for_column = (
                    np.asarray(temperature, dtype=np.float64)
                    if column_lower_temperature is None
                    else np.asarray(column_lower_temperature, dtype=np.float64)
                )
                if lower_temperature_for_column.shape != q.shape:
                    raise ValueError("column_lower_temperature must match precipitation grid")
                mid_temperature_for_column = (
                    lower_temperature_for_column - 6.5e-3 * float(pp.stability_condensation_reference_height_m)
                    if midlevel_temperature is None
                    else np.asarray(midlevel_temperature, dtype=np.float64)
                )
                upper_temperature_for_column = (
                    lower_temperature_for_column - 6.5e-3 * float(pp.three_level_upper_height_m)
                    if upperlevel_temperature is None
                    else np.asarray(upperlevel_temperature, dtype=np.float64)
                )
                if bool(pp.enable_diabatic_interface_mass_flux):
                    diabatic_interface_mass_flux_step = diabatic_interface_mass_flux(
                        previous_precipitation_mm_day,
                        lower_temperature_for_column, mid_temperature_for_column,
                        upper_temperature_for_column,
                        dt_seconds=dt * 86400.0,
                        surface_pressure_pa=float(surface_pressure_hpa) * 100.0,
                        lower_mid_pressure_depth_pa=float(pp.two_layer_pressure_depth_pa),
                        mid_upper_pressure_depth_pa=float(pp.three_level_mid_upper_pressure_depth_pa),
                        gravity_m_s2=float(pp.surface_gravity),
                        cp_dry_j_kg_k=float(pp.cp_dry),
                    )
                    omega_midlevel_pa_s = diabatic_interface_mass_flux_step.omega_lower_mid_pa_s
                    omega_upperlevel_pa_s = diabatic_interface_mass_flux_step.omega_mid_upper_pa_s
                else:
                    upper_divergence_si = -(
                        0.40 * lower_divergence_si + 0.35 * mid_divergence_si
                    ) / 0.25
                    omega_midlevel_pa_s = 0.5 * float(pp.two_layer_pressure_depth_pa) * (
                        lower_divergence_si - mid_divergence_si
                    )
                    omega_upperlevel_pa_s = 0.5 * float(pp.three_level_mid_upper_pressure_depth_pa) * (
                        mid_divergence_si - upper_divergence_si
                    )
                _vertical_substeps = 1
                if diabatic_interface_mass_flux_step is not None:
                    # The flux is preserved exactly; only its finite-volume
                    # integration is divided until no interface transports
                    # more than one quarter layer in one inner update.
                    _vertical_substeps = max(1, int(np.ceil(max(
                        diabatic_interface_mass_flux_step.lower_mid_vertical_courant_max,
                        diabatic_interface_mass_flux_step.mid_upper_vertical_courant_max,
                    ) / 0.25)))
                _vertical_dt_seconds = dt * 86400.0 / _vertical_substeps
                for _ in range(_vertical_substeps):
                    closed_step = evolve_closed_three_level_thermodynamic_column(
                        q, midlevel_humidity_next, upperlevel_humidity_next,
                        lower_temperature_for_column, mid_temperature_for_column,
                        upper_temperature_for_column, omega_midlevel_pa_s,
                        omega_upperlevel_pa_s, dt_seconds=_vertical_dt_seconds,
                        surface_pressure_pa=float(surface_pressure_hpa) * 100.0,
                        layer_heights_m=(0.0, float(pp.stability_condensation_reference_height_m), float(pp.three_level_upper_height_m)),
                    )
                    q = closed_step.lower_humidity
                    midlevel_humidity_next = closed_step.midlevel_humidity
                    upperlevel_humidity_next = closed_step.upperlevel_humidity
                    lower_temperature_for_column = closed_step.lower_temperature
                    mid_temperature_for_column = closed_step.midlevel_temperature
                    upper_temperature_for_column = closed_step.upperlevel_temperature
                    closed_column_water_residual += closed_step.water_residual_kg_m2
                    closed_column_energy_residual += closed_step.moist_static_energy_residual_j_m2
                closed_lower_temperature_next = lower_temperature_for_column
                three_level_mid_temperature_next = mid_temperature_for_column
                upperlevel_temperature_next = upper_temperature_for_column
            else:
                three_level_step = evolve_three_level_column(
                    q + midlevel_humidity_next + upperlevel_humidity_next,
                    temperature,
                    lower_divergence_si,
                    mid_divergence_si,
                    upper_divergence_si,
                    midlevel_humidity=midlevel_humidity_next,
                    upperlevel_humidity=upperlevel_humidity_next,
                    midlevel_temperature_k=midlevel_temperature,
                    upperlevel_temperature_k=upperlevel_temperature,
                    dt_days=dt,
                    lower_mid_pressure_depth_pa=float(pp.two_layer_pressure_depth_pa),
                    mid_upper_pressure_depth_pa=float(pp.three_level_mid_upper_pressure_depth_pa),
                    vertical_velocity_scale_pa_s=omega_scale,
                    exchange_days=float(pp.two_layer_entrainment_days),
                    midlevel_fraction=float(pp.two_layer_upper_humidity_fraction),
                    upperlevel_fraction=float(pp.three_level_upper_humidity_fraction),
                    midlevel_height_m=float(pp.stability_condensation_reference_height_m),
                    upperlevel_height_m=float(pp.three_level_upper_height_m),
                    thermal_relaxation_days=float(pp.two_layer_midlevel_relaxation_days),
                    use_flux_form_exchange=bool(pp.enable_three_level_flux_form_exchange),
                    enforce_column_mass_closure=bool(pp.enforce_three_level_mass_closure),
                )
                q = three_level_step.lower_humidity
                midlevel_humidity_next = three_level_step.midlevel_humidity
                upperlevel_humidity_next = three_level_step.upperlevel_humidity
                three_level_mid_temperature_next = three_level_step.midlevel_temperature
                upperlevel_temperature_next = three_level_step.upperlevel_temperature
                omega_midlevel_pa_s = three_level_step.omega_lower_mid_pa_s
                omega_upperlevel_pa_s = three_level_step.omega_mid_upper_pa_s
            upward_motion = np.clip(-omega_midlevel_pa_s / omega_scale, 0.0, 6.0)
            downward_motion = np.clip(omega_midlevel_pa_s / omega_scale, 0.0, 6.0)
            upperlevel_upward_motion = np.clip(
                -omega_upperlevel_pa_s / omega_scale, 0.0, 6.0
            )
            entrained_q = np.zeros_like(q)
            detrained_q = np.zeros_like(q)
        else:
            omega_midlevel_pa_s = 0.5 * float(pp.two_layer_pressure_depth_pa) * (
                lower_divergence_si - upper_divergence_si
            )
            upward_motion = np.clip(-omega_midlevel_pa_s / omega_scale, 0.0, 6.0)
            downward_motion = np.clip(omega_midlevel_pa_s / omega_scale, 0.0, 6.0)
            # Resolved upward mass flux lofts lower-layer vapor into the
            # independently transported midlevel partition. This is an internal
            # transfer, so it cannot create or destroy conserved column water.
            entrainment_fraction = 1.0 - np.exp(
                -dt * (upward_motion / (1.0 + upward_motion))
                / float(pp.two_layer_entrainment_days)
            )
            entrained_q = q * entrainment_fraction
            q = q - entrained_q
            midlevel_humidity_next = midlevel_humidity_next + entrained_q
            # The compensating descending branch returns upper vapor to the lower
            # reservoir. Without it, every ascent event becomes a one-way water
            # trap aloft; this is the minimal conservative overturning circulation
            # for the two-layer column rather than a storage-only split.
            detrainment_fraction = 1.0 - np.exp(
                -dt * (downward_motion / (1.0 + downward_motion))
                / float(pp.two_layer_entrainment_days)
            )
            detrained_q = midlevel_humidity_next * detrainment_fraction
            midlevel_humidity_next = midlevel_humidity_next - detrained_q
            q = q + detrained_q

    if debug_fields is not None:
        debug_fields["div"] = div
        debug_fields["ascent"] = ascent
        debug_fields["conv"] = conv
        if two_layer_active:
            debug_fields["two_layer_entrained_q"] = entrained_q
            debug_fields["two_layer_detrained_q"] = detrained_q
            debug_fields["midlevel_omega_pa_s"] = omega_midlevel_pa_s.astype(np.float32)
            debug_fields["midlevel_upward_motion"] = upward_motion.astype(np.float32)
            if three_level_active:
                debug_fields["upperlevel_omega_pa_s"] = omega_upperlevel_pa_s.astype(np.float32)
                debug_fields["upperlevel_upward_motion"] = upperlevel_upward_motion.astype(np.float32)
                if diabatic_interface_mass_flux_step is not None:
                    debug_fields["diabatic_interface_latent_heating_w_m2"] = (
                        diabatic_interface_mass_flux_step.latent_heating_w_m2
                    )
                    debug_fields["diabatic_interface_lower_mid_courant_max"] = float(
                        diabatic_interface_mass_flux_step.lower_mid_vertical_courant_max
                    )
                    debug_fields["diabatic_interface_mid_upper_courant_max"] = float(
                        diabatic_interface_mass_flux_step.mid_upper_vertical_courant_max
                    )
                    debug_fields["diabatic_interface_vertical_substeps"] = int(
                        _vertical_substeps
                    )

    # Orographic uplift signal.
    # gy must be the physical NORTHWARD slope (row index increases southward),
    # so gy = -∂elev/∂i; otherwise northward wind blowing up a north-facing
    # slope registered as downslope (and vice versa), inverting the meridional
    # half of both the uplift and rain-shadow terms.
    if _sc is not None and "orog" in _sc:
        orog = _sc["orog"]
        rain_shadow_suppression = _sc["rain_shadow_suppression"]
    else:
        gx = _ddx_periodic(elev)
        gy = -np.gradient(elev, axis=0)
        slope = np.hypot(gx, gy)
        orog = np.clip(gx * u + gy * v, 0.0, None) + 0.25 * slope
        orog = land_f * orog
        # Upwind footprint (A5-OROG's named next lever): the four pointwise
        # ceilings downstream are all exhausted, and the residual defect is that
        # this product is a 1-2 cell spike on the crest while real orographic
        # precipitation covers a broad windward flank. Smear it along the wind
        # *before* normalizing, so the percentile below describes the broadened
        # field and the clip truncates it consistently. Re-masked to land
        # afterwards: the smear samples across the coast, and an ocean cell has
        # no orographic uplift to receive.
        if (
            float(pp.orographic_upwind_footprint_km) > 0.0
            or float(pp.orographic_spillover_km) > 0.0
        ):
            orog = land_f * _smear_along_wind(
                orog,
                u,
                v,
                dx_grid,
                dy_grid,
                float(pp.orographic_upwind_footprint_km),
                float(pp.orographic_spillover_km),
            )
        # Normalizer choice decides whether the clip below carries any spatial
        # information at all. `np.percentile(orog, 90.0)` spans the WHOLE grid,
        # but `orog` was just zeroed over ocean — on Earth that is ~66% of cells
        # sitting at exactly 0.0, so the nominal "90th percentile" is really only
        # land's ~70th. The normalizer comes out ~3.8x too small, every land value
        # is inflated by that factor, and the ceiling below then truncates 20% of
        # land and 100% of the steepest 5% — flattening windward and leeward
        # flanks onto the same clipped value and destroying the very contrast the
        # term exists to create. Measured on saves/earth.pkl: along the Cascades
        # and S Andes transects `orog` reads a saturated 2.000 on both flanks.
        # See ACCURACY_AUDIT.md A5. Land-only restores the intended meaning of
        # "90th percentile of the orographic signal" and drops clip truncation to
        # ~6% of land.
        if bool(pp.orographic_normalizer_land_only):
            # Restrict to land cells that carry signal, not merely to land. On
            # Earth 99.7% of land is nonzero so this is a 0.4% difference from a
            # plain land percentile, but it is what keeps the normalizer
            # meaningful on degenerate terrain: for a continent with no relief,
            # the only nonzero cells are the land/ocean step at the coast, a
            # plain land percentile collapses toward zero, and every coastal cell
            # then divides by ~1e-6 and pins at the clip. That failure mode is
            # not hypothetical for non-Earth presets.
            land_cells = orog[(land_f > 0.5) & (orog > 0.0)]
            normalizer = (
                float(np.percentile(land_cells, 90.0)) if land_cells.size else 0.0
            )
        else:
            normalizer = float(np.percentile(orog, 90.0))
        orog = orog / (normalizer + 1e-6)
        orog = np.clip(
            orog + 0.15 * _lap(orog.astype(np.float32)),
            0.0,
            float(pp.orographic_uplift_clip),
        )

        # Rain-shadow drying: the mirror image of orographic uplift. Descending air
        # on the lee side of a range compresses and warms, lowering RH — this is why
        # the Atacama (lee of the Andes), Patagonia, the Great Basin, and the Gobi
        # (lee of the Himalaya/Tibetan Plateau) are deserts in reality. Previously
        # `orog` only ever added rain (windward term clipped to >=0) with no leeward
        # counterpart, so these regions had no mechanism to dry out relative to the
        # surrounding potential field (observed: Atacama ~780 mm/yr vs Earth's ~0).
        downslope = np.clip(-(gx * u + gy * v), 0.0, None)
        downslope = land_f * downslope
        downslope = downslope / (np.percentile(downslope, 90.0) + 1e-6)
        downslope = np.clip(downslope, 0.0, 2.0)
        rain_shadow_suppression = np.clip(1.0 - 0.40 * downslope, 0.35, 1.0).astype(np.float32, copy=False)
        if _sc is not None:
            orog = orog.astype(np.float32, copy=False)
            orog.flags.writeable = False
            rain_shadow_suppression.flags.writeable = False
            _sc["orog"] = orog
            _sc["rain_shadow_suppression"] = rain_shadow_suppression

    condensate_next = condensate
    hydrometeors_next = precipitating_hydrometeors
    midlevel_temperature_next = (
        three_level_mid_temperature_next if three_level_active else midlevel_temperature
    )
    upperlevel_temperature_next = (
        upperlevel_temperature_next if three_level_active else upperlevel_temperature
    )

    def _apply_closed_phase_conversion(
        lower_condensed_q: np.ndarray,
        mid_condensed_q: np.ndarray,
        upper_condensed_q: np.ndarray,
    ) -> None:
        """Apply an already-diagnosed phase change without a second heat term."""
        nonlocal q, midlevel_humidity_next, upperlevel_humidity_next
        nonlocal closed_lower_temperature_next, midlevel_temperature_next
        nonlocal upperlevel_temperature_next, closed_column_water_residual
        nonlocal closed_column_energy_residual
        if not closed_three_level_active:
            return
        assert closed_lower_temperature_next is not None
        assert midlevel_temperature_next is not None
        assert upperlevel_temperature_next is not None
        zero_omega = np.zeros_like(q, dtype=np.float64)
        phase_step = evolve_closed_three_level_thermodynamic_column(
            q, midlevel_humidity_next, upperlevel_humidity_next,
            closed_lower_temperature_next, midlevel_temperature_next,
            upperlevel_temperature_next, zero_omega, zero_omega,
            dt_seconds=dt * 86400.0,
            surface_pressure_pa=float(surface_pressure_hpa) * 100.0,
            layer_heights_m=(0.0, float(pp.stability_condensation_reference_height_m), float(pp.three_level_upper_height_m)),
            condensed_specific_humidity=(
                lower_condensed_q, mid_condensed_q, upper_condensed_q,
            ),
        )
        q = phase_step.lower_humidity
        midlevel_humidity_next = phase_step.midlevel_humidity
        upperlevel_humidity_next = phase_step.upperlevel_humidity
        closed_lower_temperature_next = phase_step.lower_temperature
        midlevel_temperature_next = phase_step.midlevel_temperature
        upperlevel_temperature_next = phase_step.upperlevel_temperature
        closed_column_water_residual += phase_step.water_residual_kg_m2
        closed_column_energy_residual += phase_step.moist_static_energy_residual_j_m2

    condensate_precipitation = np.zeros_like(q, dtype=np.float32)
    lowerlevel_condensate_precipitation = np.zeros_like(q, dtype=np.float32)
    midlevel_condensate_precipitation = np.zeros_like(q, dtype=np.float32)
    upperlevel_condensate_precipitation = np.zeros_like(q, dtype=np.float32)
    if bool(pp.enable_prognostic_condensate):
        condensate_in = (
            np.zeros_like(q) if condensate is None else np.asarray(condensate, dtype=np.float64)
        )
        hydrometeors_for_microphysics = precipitating_hydrometeors
        if (
            bool(pp.enable_separate_precipitating_hydrometeors)
            and bool(pp.enable_hydrometeor_transport)
            and bool(pp.enable_prognostic_column_water)
            and precipitating_hydrometeors is not None
        ):
            # Hydrometeors are not stationary cloud water: while they fall,
            # the resolved cloud-layer flow displaces them horizontally.  Use
            # the conservative column-water transport kernel for no longer
            # than their sedimentation lifetime; transporting for an entire
            # coarse climate step would imply that rain remains aloft after it
            # has already reached the surface.
            _hydrometeor_u = (
                np.asarray(wind_u_midlevel)
                if three_level_active and wind_u_midlevel is not None
                else np.asarray(wind_u_aloft)
                if wind_u_aloft is not None
                else u
            )
            _hydrometeor_v = (
                np.asarray(wind_v_midlevel)
                if three_level_active and wind_v_midlevel is not None
                else np.asarray(wind_v_aloft)
                if wind_v_aloft is not None
                else v
            )
            _hydrometeor_transport = evolve_column_water(
                np.asarray(precipitating_hydrometeors, dtype=np.float64) * 2000.0,
                np.zeros_like(q),
                np.zeros_like(q),
                _hydrometeor_u,
                _hydrometeor_v,
                dx_m=dx_grid,
                dy_m=float(dy_grid),
                dt_days=min(dt, float(pp.condensate_fallout_timescale_days)),
                cell_area_m2=area_m2,
                x_face_length_m=x_face_length_m,
                y_face_length_m=y_face_length_m,
            )
            hydrometeors_for_microphysics = _hydrometeor_transport.water_mm / 2000.0
            if debug_fields is not None:
                debug_fields["hydrometeor_transport_relative_residual"] = (
                    _hydrometeor_transport.relative_residual
                )
        transport_scale = float(np.clip(pp.condensate_transport_scale, 0.0, 1.0))
        if transport_scale > 0.0:
            if pp.enable_prognostic_column_water:
                # The vapor leg is already in finite-volume column units in
                # this mode.  Carry suspended condensate through the identical
                # geometry, so the vapor+condensate budget does not mix a
                # conservative and a non-conservative transport operator.
                # In the active vertical closure, suspended condensate belongs
                # to the midlevel cloud layer and therefore follows the
                # resolved upper wind, not the near-surface circulation used
                # by legacy condensate.  The fallback keeps every older gate
                # bit-identical.
                condensate_u = (
                    u if not two_layer_active or wind_u_aloft is None
                    else np.asarray(wind_u_aloft)
                )
                condensate_v = (
                    v if not two_layer_active or wind_v_aloft is None
                    else np.asarray(wind_v_aloft)
                )
                condensate_transport = evolve_column_water(
                    condensate_in * 2000.0,
                    np.zeros_like(condensate_in),
                    np.zeros_like(condensate_in),
                    condensate_u,
                    condensate_v,
                    dx_m=dx_grid,
                    dy_m=float(dy_grid),
                    dt_days=dt,
                    cell_area_m2=area_m2,
                    x_face_length_m=x_face_length_m,
                    y_face_length_m=y_face_length_m,
                )
                condensate_advected = condensate_transport.water_mm / 2000.0
                if debug_fields is not None:
                    debug_fields["condensate_transport_relative_residual"] = (
                        condensate_transport.relative_residual
                    )
                    debug_fields["condensate_transport_layer"] = (
                        "midlevel" if two_layer_active else "surface"
                    )
            else:
                condensate_advected = _advect_scalar_flux_eulerian(
                    condensate_in,
                    u,
                    v,
                    dt * 86400.0,
                    dx_grid,
                    dy_grid,
                )
            condensate_in = (1.0 - transport_scale) * condensate_in + transport_scale * condensate_advected
        if convective_closure_active:
            use_two_layer_adjustment = bool(pp.enable_two_layer_convective_adjustment)
            upper_condensed_q = np.zeros_like(q)
            upper_rainout_q = np.zeros_like(q)
            upperlevel_condensed_q = np.zeros_like(q)
            upperlevel_rainout_q = np.zeros_like(q)
            upper_qsat = None
            reference_midlevel_temperature = temperature - (
                6.5e-3 * float(pp.stability_condensation_reference_height_m)
            )
            if use_two_layer_adjustment:
                midlevel_in = (
                    reference_midlevel_temperature
                    if midlevel_temperature_next is None
                    else np.asarray(midlevel_temperature_next, dtype=np.float64)
                )
                if midlevel_in.shape != q.shape:
                    raise ValueError("midlevel_temperature must match precipitation grid")
            else:
                midlevel_in = None
            _midlevel_target_rh = float(
                pp.betts_miller_midlevel_target_relative_humidity
                if pp.enable_simplified_betts_miller_convection
                else pp.stability_condensation_critical_rh
            )
            _upperlevel_target_rh = float(
                pp.betts_miller_upper_target_relative_humidity
                if pp.enable_simplified_betts_miller_convection
                else pp.stability_condensation_critical_rh
            )
            if pp.enable_simplified_betts_miller_convection and not all(
                0.0 < value < 1.0
                for value in (
                    float(pp.betts_miller_target_relative_humidity),
                    _midlevel_target_rh,
                    _upperlevel_target_rh,
                )
            ):
                raise ValueError("Betts--Miller vertical target humidities must lie strictly between zero and one")
            q_before_condensation = q
            if bool(pp.enable_simplified_betts_miller_convection):
                q, condensate_next, condensate_rainout, _bm_condensed_q = (
                    simplified_betts_miller_condensation(
                        q,
                        qsat,
                        ascent,
                        condensate_in,
                        dt_days=dt,
                        relaxation_hours=float(pp.betts_miller_relaxation_hours),
                        target_relative_humidity=float(pp.betts_miller_target_relative_humidity),
                        fallout_timescale_days=float(pp.condensate_fallout_timescale_days),
                    )
                )
                cape_proxy = np.zeros_like(q, dtype=np.float32)
                stability_activation = (ascent / (1.0 + ascent)).astype(np.float32)
            else:
                (
                    q,
                    condensate_next,
                    condensate_rainout,
                    cape_proxy,
                    stability_activation,
                ) = stability_aware_condensation(
                    q,
                    qsat,
                    temperature,
                    ascent,
                    condensate_in,
                    environment_temperature_k=midlevel_in,
                    surface_pressure_hpa=surface_pressure_hpa,
                    dt_days=dt,
                    condensation_timescale_days=float(pp.condensate_condensation_timescale_days),
                    fallout_timescale_days=float(pp.condensate_fallout_timescale_days),
                    critical_relative_humidity=float(pp.stability_condensation_critical_rh),
                    reference_height_m=float(pp.stability_condensation_reference_height_m),
                    cape_scale_j_kg=float(pp.stability_condensation_cape_scale_j_kg),
                )
            if use_two_layer_adjustment and not closed_three_level_active:
                relax_fraction = 1.0 - np.exp(
                    -dt / float(pp.two_layer_midlevel_relaxation_days)
                )
                # Latent heating is deposited aloft, opposing continued deep
                # convection until radiation/dynamics restore the background
                # lapse profile.  Water remains in the single conserved column;
                # this is only the companion thermal tendency.
                latent_heating_k = 0.40 * (2.5e6 / 1004.0) * (
                    q_before_condensation - q
                )
                midlevel_temperature_next = np.clip(
                    midlevel_in
                    + relax_fraction * (reference_midlevel_temperature - midlevel_in)
                    + latent_heating_k,
                    150.0,
                    350.0,
                ).astype(np.float32)
            if closed_three_level_active:
                lower_condensed_q = np.maximum(q_before_condensation - q, 0.0)
                # The condensation routine has already removed vapour. Restore
                # its input briefly so the closed operator makes the identical
                # removal while converting Lv*dq into layer temperature.
                q = q_before_condensation
                _apply_closed_phase_conversion(
                    lower_condensed_q,
                    np.zeros_like(q),
                    np.zeros_like(q),
                )
            if two_layer_active:
                # The upper partition is an active layer, not merely a storage
                # bin.  Diagnose saturation at its reference pressure and let
                # resolved ascent condense water above the same RH threshold as
                # the lower stability closure.  Any true supersaturation is
                # removed regardless of ascent so a numerical humidity clip is
                # never allowed to act as an untracked sink.
                upper_pressure_hpa = float(surface_pressure_hpa) * math.exp(
                    -float(pp.stability_condensation_reference_height_m) / 8000.0
                )
                upper_temperature = np.asarray(midlevel_temperature_next, dtype=np.float64)
                upper_tc = np.clip(upper_temperature - 273.15, -60.0, 60.0)
                upper_es = 6.112 * np.exp(17.67 * upper_tc / (upper_tc + 243.5))
                upper_qsat = np.clip(
                    0.622 * upper_es / upper_pressure_hpa, 0.0, 0.035
                )
                upper_threshold = _midlevel_target_rh * upper_qsat
                upper_excess = np.maximum(midlevel_humidity_next - upper_threshold, 0.0)
                ascent_activation = upward_motion / (1.0 + upward_motion)
                condensation_fraction = 1.0 - np.exp(
                    -dt / float(pp.condensate_condensation_timescale_days)
                )
                upper_condensed_q = np.maximum(
                    upper_excess * ascent_activation * condensation_fraction,
                    np.maximum(midlevel_humidity_next - upper_qsat, 0.0),
                )
                upper_condensed_q = np.minimum(upper_condensed_q, midlevel_humidity_next)
                upper_rainout_q = upper_condensed_q * (
                    1.0 - np.exp(-dt / float(pp.condensate_fallout_timescale_days))
                )
                if not closed_three_level_active:
                    midlevel_humidity_next = midlevel_humidity_next - upper_condensed_q
                condensate_next = np.asarray(condensate_next, dtype=np.float64) + (
                    upper_condensed_q - upper_rainout_q
                )
                # Latent heat from both layers remains in the persistent upper
                # temperature reservoir, which is exchanged with resolved air
                # by simulate_step on the next temperature advance.
                if closed_three_level_active:
                    _apply_closed_phase_conversion(
                        np.zeros_like(q), upper_condensed_q, np.zeros_like(q)
                    )
                else:
                    midlevel_temperature_next = np.clip(
                        np.asarray(midlevel_temperature_next, dtype=np.float64)
                        + 0.40 * (2.5e6 / 1004.0) * upper_condensed_q,
                        150.0,
                        350.0,
                    ).astype(np.float32)
            if three_level_active:
                # The third reservoir has its own, colder pressure level and
                # interface ascent.  It shares the suspended condensate pool
                # only after a conservative vapor-to-condensate transfer, so
                # the existing fallout and cloud-radiation plumbing can remain
                # a single auditable column-water budget.
                upperlevel_pressure_hpa = float(surface_pressure_hpa) * math.exp(
                    -float(pp.three_level_upper_height_m) / 8000.0
                )
                upperlevel_temperature = np.asarray(upperlevel_temperature_next, dtype=np.float64)
                upperlevel_tc = np.clip(upperlevel_temperature - 273.15, -60.0, 60.0)
                upperlevel_es = 6.112 * np.exp(
                    17.67 * upperlevel_tc / (upperlevel_tc + 243.5)
                )
                upperlevel_qsat = np.clip(
                    0.622 * upperlevel_es / upperlevel_pressure_hpa, 0.0, 0.035
                )
                upperlevel_threshold = _upperlevel_target_rh * upperlevel_qsat
                upperlevel_excess = np.maximum(
                    upperlevel_humidity_next - upperlevel_threshold, 0.0
                )
                upperlevel_activation = (
                    upperlevel_upward_motion / (1.0 + upperlevel_upward_motion)
                )
                upperlevel_condensed_q = np.maximum(
                    upperlevel_excess * upperlevel_activation * condensation_fraction,
                    np.maximum(upperlevel_humidity_next - upperlevel_qsat, 0.0),
                )
                upperlevel_condensed_q = np.minimum(
                    upperlevel_condensed_q, upperlevel_humidity_next
                )
                upperlevel_rainout_q = upperlevel_condensed_q * (
                    1.0 - np.exp(-dt / float(pp.condensate_fallout_timescale_days))
                )
                if not closed_three_level_active:
                    upperlevel_humidity_next = (
                        upperlevel_humidity_next - upperlevel_condensed_q
                    )
                condensate_next = np.asarray(condensate_next, dtype=np.float64) + (
                    upperlevel_condensed_q - upperlevel_rainout_q
                )
                if closed_three_level_active:
                    _apply_closed_phase_conversion(
                        np.zeros_like(q), np.zeros_like(q), upperlevel_condensed_q
                    )
                else:
                    upperlevel_temperature_next = np.clip(
                        upperlevel_temperature
                        + 0.40 * (2.5e6 / 1004.0) * upperlevel_condensed_q,
                        150.0,
                        350.0,
                    ).astype(np.float32)
        else:
            q, condensate_next, condensate_rainout = evolve_bulk_condensate(
                q,
                qsat,
                ascent,
                condensate_in,
                dt_days=dt,
                condensation_timescale_days=float(pp.condensate_condensation_timescale_days),
                fallout_timescale_days=float(pp.condensate_fallout_timescale_days),
            )
        if bool(pp.enable_separate_precipitating_hydrometeors):
            # Reconstruct the cloud mass before the legacy immediate-fallout
            # step, then advance explicit cloud and hydrometeor reservoirs.
            # This gate keeps older paths byte-for-byte intact while the new
            # state contract is threaded through every caller.
            _cloud_available = np.asarray(condensate_next, dtype=np.float64) + condensate_rainout
            if convective_closure_active and two_layer_active:
                _cloud_available = _cloud_available + upper_rainout_q
            if convective_closure_active and three_level_active:
                _cloud_available = _cloud_available + upperlevel_rainout_q
            condensate_next, hydrometeors_next, condensate_rainout = (
                separate_cloud_and_hydrometeor_reservoirs(
                    np.zeros_like(_cloud_available),
                    hydrometeors_for_microphysics,
                    _cloud_available,
                    dt_days=dt,
                    autoconversion_timescale_days=float(pp.condensate_autoconversion_timescale_days),
                    fallout_timescale_days=float(pp.condensate_fallout_timescale_days),
                    cloud_retention_q=float(pp.cloud_optical_condensate_cap_q),
                )
            )
            upper_rainout_q = np.zeros_like(condensate_rainout)
            upperlevel_rainout_q = np.zeros_like(condensate_rainout)
        condensate_precipitation = (
            condensate_rainout * (2000.0 / dt)
        ).astype(np.float32, copy=False)
        lowerlevel_condensate_precipitation = condensate_precipitation.copy()
        if convective_closure_active and two_layer_active:
            midlevel_condensate_precipitation = (
                upper_rainout_q * (2000.0 / dt)
            ).astype(np.float32, copy=False)
            condensate_precipitation = (
                condensate_precipitation + midlevel_condensate_precipitation
            )
        if convective_closure_active and three_level_active:
            upperlevel_condensate_precipitation = (
                upperlevel_rainout_q * (2000.0 / dt)
            ).astype(np.float32, copy=False)
            condensate_precipitation = (
                condensate_precipitation + upperlevel_condensate_precipitation
            )
        if debug_fields is not None:
            debug_fields["condensate_rainout_dq"] = condensate_rainout
            debug_fields["condensate_precipitation_mm_day"] = condensate_precipitation
            debug_fields["lowerlevel_condensate_precipitation_mm_day"] = (
                lowerlevel_condensate_precipitation
            )
            debug_fields["midlevel_condensate_precipitation_mm_day"] = (
                midlevel_condensate_precipitation
            )
            debug_fields["upperlevel_condensate_precipitation_mm_day"] = (
                upperlevel_condensate_precipitation
            )
            debug_fields["condensate_closure"] = (
                "betts_miller" if pp.enable_simplified_betts_miller_convection
                else "stability_aware" if pp.enable_stability_aware_condensation
                else "rh_ascent"
            )
            if convective_closure_active:
                debug_fields["stability_cape_proxy_j_kg"] = cape_proxy
                debug_fields["stability_condensation_activation"] = stability_activation
                if bool(pp.enable_two_layer_convective_adjustment):
                    debug_fields["midlevel_temperature_k"] = midlevel_temperature_next
                if two_layer_active:
                    debug_fields["midlevel_qsat"] = upper_qsat.astype(np.float32)
                    debug_fields["midlevel_condensed_q"] = upper_condensed_q.astype(np.float32)
                    debug_fields["midlevel_rainout_q"] = upper_rainout_q.astype(np.float32)
                if three_level_active:
                    debug_fields["upperlevel_qsat"] = upperlevel_qsat.astype(np.float32)
                    debug_fields["upperlevel_condensed_q"] = upperlevel_condensed_q.astype(np.float32)
                    debug_fields["upperlevel_rainout_q"] = upperlevel_rainout_q.astype(np.float32)

    # Phase 2: Enhanced convective precipitation with CAPE-like triggering
    # This significantly improves tropical rainfall (ITCZ) realism
    rh = q / (qsat + 1e-6)
    P_convective = compute_convective_precipitation(
        temperature, q, dt_days=dt,
        trigger_temp_c=20.0,  # Tropical threshold
        trigger_rh=0.8,        # High humidity requirement
        max_rate_mm_day=10.0,  # Realistic tropical thunderstorm rate
        surface_pressure_hpa=surface_pressure_hpa,
    )
    # Normalize convective contribution to blend with other terms
    conv_norm = np.clip(conv, 0.0, 1.5) / 1.5
    if _sc is not None and "ascent_norm" in _sc:
        ascent_norm = _sc["ascent_norm"]
    else:
        # Depends only on the already-cached `ascent`, not on q -- substep-invariant.
        ascent_norm = np.clip(ascent, 0.0, 1.5) / 1.5
        if _sc is not None:
            ascent_norm = ascent_norm.astype(np.float32, copy=False)
            ascent_norm.flags.writeable = False
            _sc["ascent_norm"] = ascent_norm
    convective = np.clip(P_convective / 10.0, 0.0, 2.0)  # Scale to [0, 2] for blending
    convective = np.clip(convective + 0.10 * conv, 0.0, 2.0)
    convective = convective * (0.05 + 0.40 * itcz_window[:, None]) * (
        0.18 + 0.82 * conv_norm
    ) * (
        0.22 + 0.78 * ascent_norm
    )

    # Blend drivers into precipitation potential, then apply subsidence drying.
    # Subsidence_suppression reduces precip in divergent (descending) zones,
    # creating the subtropical dry belt that the convergence-only scheme lacks.
    # ITCZ-window weights retuned down (2026-07-03, divergence-sign fix): these
    # boosts were calibrated when the meridional half of the convergence/ascent
    # signal was inverted (the ITCZ's real convergence registered as divergence),
    # so they were compensating for a missing physical signal. With the sign
    # fixed, conv/ascent now genuinely peak at the ITCZ and the old prescribed
    # boosts double-counted it (tropical band hit 9.6 mm/day vs the 8.0 gate).
    rh_release = rh * (0.10 + 0.22 * itcz_window[:, None] + 0.06 * storm_window[:, None])
    conv_driver = conv * (0.12 + 0.22 * itcz_window[:, None] + 0.08 * storm_window[:, None])
    if _sc is not None and "ascent_driver" in _sc:
        ascent_driver = _sc["ascent_driver"]
    else:
        # Depends only on the already-cached `ascent` and the static lat windows.
        ascent_driver = ascent * (0.20 + 0.20 * itcz_window[:, None] + 0.08 * storm_window[:, None])
        if _sc is not None:
            ascent_driver = ascent_driver.astype(np.float32, copy=False)
            ascent_driver.flags.writeable = False
            _sc["ascent_driver"] = ascent_driver
    # Stratiform term: existing cloud cover (frontal/persistent sheets) rains even
    # without a fresh convective trigger. target_mean_mm_day rescaling below keeps
    # the global mean calibrated, so this mainly reshapes *where* rain falls to track
    # cloud cover rather than changing the overall total. Weight retuned 0.10 -> 0.06
    # (2026-07): 0.10 pushed SH subtropical mean precip to 2.83 mm/day, just over
    # test_subtropical_precip_quantity's 2.8 cap (bisected precisely — 0.09 still
    # fails at 2.81, 0.08 is the first passing value; 0.06 leaves headroom rather
    # than sitting right at that boundary again).
    if _sc is not None and "stratiform" in _sc:
        stratiform = _sc["stratiform"]
    else:
        # Depends only on cloud_fraction, which the substep loop passes through
        # unchanged (only humidity/soil evolve between substeps) -- invariant.
        if cloud_fraction is not None:
            stratiform = np.clip(cloud_fraction.astype(np.float32), 0.0, 1.0)
        else:
            stratiform = np.zeros((H, W), dtype=np.float32)
        if _sc is not None:
            stratiform.flags.writeable = False
            _sc["stratiform"] = stratiform
    # A5 stage 4 (found 2026-08-02, not in the audit's original three): this
    # ceiling binds on windward cells and is why raising `orog` past a point
    # stops helping -- `precip_potential`'s windward/leeward ratio saturates at
    # ~2.9 whether the incoming `orog` ratio is 3.0 or 11.7, because the windward
    # cell is pinned here while the leeward one is not. 3.0 is the historical
    # hardcoded value, so the default is an exact no-op.
    _potential_ceiling = float(pp.precip_potential_ceiling)

    # `precip_orographic_weight` is A5's stage 2: `orog`'s share of this sum
    # against the five terms that are not orographically organized. The other
    # five are rescaled to hold the total weight at its historical 1.10, so this
    # knob moves orography's *relative* contribution rather than the overall
    # magnitude of `precip_potential` (which the moisture budget would absorb
    # anyway). 0.20 reproduces the historical weights exactly.
    _orog_weight = float(pp.precip_orographic_weight)
    _other_scale = (1.10 - _orog_weight) / 0.90
    precip_potential = uplift_coeff * (
        0.18 * _other_scale * rh_release +
        0.24 * _other_scale * conv_driver +
        _orog_weight * orog +
        0.20 * _other_scale * convective +
        0.22 * _other_scale * ascent_driver +
        0.06 * _other_scale * stratiform
    ) * subsidence_suppression * rain_shadow_suppression
    lat_shape = np.clip(0.78 + 0.20 * itcz_window[:, None] + 0.02 * storm_window[:, None], 0.60, 1.40)
    precip_potential = precip_potential * lat_shape

    if NUMBA_AVAILABLE:
        # Fast path: Numba-accelerated smoothing. precip_potential was assigned via
        # `* lat_shape` just above (a fresh array), and `.astype()` below copies before
        # the Numba call reads it, so it's always safe to mutate in place: each
        # iteration fully computes lap_p from the old values before writing.
        for _ in range(3):
            lap_p = _laplacian_numba(precip_potential.astype(np.float32))
            np.add(precip_potential, 0.18 * lap_p, out=precip_potential)
            np.clip(precip_potential, 0.0, _potential_ceiling, out=precip_potential)
    else:
        # Fallback: original NumPy implementation. Same in-place reasoning as above --
        # _laplacian reads and fully materializes its output before we mutate.
        for _ in range(3):
            lap_p = _laplacian(precip_potential)
            np.add(precip_potential, 0.18 * lap_p, out=precip_potential)
            np.clip(precip_potential, 0.0, _potential_ceiling, out=precip_potential)
    post_shape = np.clip(0.92 + 0.20 * itcz_window[:, None] - 0.10 * storm_window[:, None], 0.82, 1.12)
    # Convert the dimensionless precipitation drivers into a realistic daily
    # rain-out timescale by regime.  The previous uniform conversion left raw
    # production about 5.5x short globally, forcing the moisture-budget fill
    # to invent most rainfall.  Do not amplify the subtropical subsidence
    # regime: its scarce production is physical and is what distinguishes
    # deserts.  Smooth shoulders avoid introducing latitude seams, while the
    # established east-margin mask lets monsoon coasts retain the wet-regime
    # conversion at the same latitude as genuine deserts.
    _subtropical_plateau = drybelt_regime_window
    _monsoon_wet_affinity = np.clip(
        float(pp.monsoon_east_margin_exemption) * monsoon_margin_factor,
        0.0,
        1.0,
    )
    _raw_conversion_affinity = np.clip(
        1.0 - _subtropical_plateau[:, None]
        * (1.0 - _monsoon_wet_affinity),
        0.0,
        1.0,
    )
    # Gated 2026-08-02 (ACCURACY_AUDIT.md process note 6). The 4.5 default
    # reproduces the hardcoded value this shipped with, so this is a no-op;
    # `precip_raw_conversion_gain=0.0` is the in-place ablation that a git
    # bisect had to stand in for the first time this mechanism misbehaved.
    _raw_conversion_gain = 1.0 + float(pp.precip_raw_conversion_gain) * _raw_conversion_affinity
    precip_potential = np.clip(
        _raw_conversion_gain * precip_potential * post_shape, 0.0, _potential_ceiling
    )

    # Convert potential to precipitation (mm/day) with moisture conservation
    # Cap removal fraction: at dt=6 the uncapped value clips to 1.0 (total moisture
    # stripping), leaving humidity_next≈0 everywhere and erasing spatial gradients.
    # Limiting to dt=2.0 and 0.85 ensures cells retain ~15% of moisture, so
    # the next substep starts with a spatially differentiated humidity field.
    # The ceiling is `PlanetParams.precip_rain_out_ceiling` (0.85 default = the
    # long-standing hardcoded value). It is the second of A5's absorption stages:
    # a windward flank that "wants" to rain harder cannot, because it is already
    # stripping the maximum permitted fraction of its column. Measured on
    # saves/earth.pkl, the entire S Andes windward slope sits pinned at 0.85.
    remove_frac = np.clip(
        rain_efficiency * precip_potential * min(dt, 2.0),
        0.0,
        float(pp.precip_rain_out_ceiling),
    )
    dq = np.clip(remove_frac * q, 0.0, q)
    use_bulk_condensate_rainfall = (
        bool(pp.enable_prognostic_column_water)
        and bool(pp.enable_prognostic_condensate)
        and bool(pp.column_water_use_bulk_condensate_rainfall)
    )
    if use_bulk_condensate_rainfall:
        # In the fully experimental closure vapor can reach the surface only
        # after entering the persistent condensate reservoir.  Retain the
        # saturation adjustment below as a non-negotiable numerical/physical
        # safeguard, but do not stack empirical potential rain on top of it.
        dq = np.zeros_like(q)
    column_mm_per_q = 2000.0  # ~20 mm PW for q=0.01
    P = dq * (column_mm_per_q / dt)
    if debug_fields is not None:
        debug_fields["q"] = q
        # Closure accounting: these are the physical rainout terms before the
        # target-side moisture-budget allocator changes the result. Validation
        # uses them to distinguish a better precipitation mechanism from a
        # better post-hoc fit to the same zonal target.
        debug_fields["humidity_before_rainout"] = q
        debug_fields["rainout_raw_dq"] = dq
        debug_fields["precipitation_raw_mm_day"] = P
        debug_fields["precip_potential_prerescale"] = precip_potential
        debug_fields["remove_frac_prerescale"] = remove_frac
        debug_fields["column_water_precipitation_closure"] = (
            "bulk_condensate" if use_bulk_condensate_rainfall
            else "empirical_vapor_rainout"
        )
        debug_fields["rh_release"] = rh_release
        debug_fields["conv_driver"] = conv_driver
        debug_fields["ascent_driver"] = ascent_driver
        debug_fields["stratiform"] = stratiform
        debug_fields["convective"] = convective
        debug_fields["orog"] = orog
        debug_fields["rain_shadow_suppression"] = rain_shadow_suppression
    if target_mean_mm_day > 0.0:
        # Zonal (per-latitude-row) rescale, replacing the old single flat scalar
        # (2026-07, itcz-global-rescale-coupling fix). A flat scalar solved to hit
        # the *global* mean structurally coupled every latitude to the same knob:
        # trimming the ITCZ's raw share just made the solver raise the scalar to
        # compensate, re-inflating the ITCZ almost as much as the trim removed
        # (measured: zeroing the ITCZ lat_shape/post_shape weight only cut tropical
        # precip 13%, far short of the ~50% needed -- see that memory for the full
        # decomposition). A per-row target (Earth's real zonal precip shape,
        # renormalized to preserve the exact same global calibration point --
        # see `_zonal_precip_target_profile`) lets the ITCZ and mid-latitudes each
        # solve toward their own realistic target instead of fighting over one.
        #
        # An EARLIER, DIFFERENT fix attempt (2026-07, aridity-drift-30yr
        # investigation) tried raising the flat scalar's ceiling and rescaling
        # `precip_potential` pre-clip instead of the already-clipped `dq` -- that
        # was REVERTED, it made desert-vs-continental ranking worse (see
        # known-physics-gaps.md item 3 for the full write-up). This zonal-rescale
        # fix is orthogonal to that one: it changes the *shape* of the target
        # across latitude, not how hard any single cell is allowed to wring out
        # its own moisture, so the earlier failure mode (collapsing continental
        # interior q under a higher removal ceiling) doesn't apply here.
        # Blended, not a full per-row correction: a pure target/raw ratio per row
        # can *erase* real within-row physical signals that only vary by latitude
        # (e.g. a windward-vs-leeward orographic test on a meridional elevation
        # ramp, where the whole row is spatially uniform) -- since both the
        # "correct" and "wrong" physics would independently get pulled to the same
        # row target regardless of which produced more rain. Blending with the old
        # flat global scalar keeps a real, tested floor under the correction:
        # blend=0.0 reproduces the exact old behavior; 1.0 is the full zonal-target
        # correction.
        #
        # Row-heterogeneity gating (2026-07 follow-up session): a single fixed
        # blend forced a bad trade-off -- strong enough to meaningfully fix real
        # terrain's ITCZ meant erasing the orographic test's signal (that test's
        # row is perfectly spatially uniform across longitude, since its elevation
        # ramp only varies by latitude); weak enough to preserve that test meant
        # barely denting the real ITCZ bug (a flat blend=0.15 only got real-terrain
        # tropical precip to ~8.2 mm/day, vs Earth's ~5.5). Gate the blend by the
        # row's own coefficient of variation (std/mean of the *raw* precip field
        # across longitude) so a spatially uniform row (the orographic test)
        # automatically falls back toward the safe flat scalar, while a
        # heterogeneous row (real terrain, always) gets more correction.
        # Measured trade-off calibrating ZONAL_BLEND_MAX/CV_REF: real terrain's
        # row-CV (mean ~0.79, median ~0.75) IS meaningfully higher than
        # `mixed_initial_state` -- the synthetic fixture
        # test_cloud_cover_plausible_range uses (mean ~0.51, median ~0.43) -- but
        # not by enough margin to fully decouple them: pushing ZONAL_BLEND_MAX/
        # CV_REF up to exploit that gap (tried up to 0.8/1.0) still regressed
        # global mean cloud cover below that test's 0.18 floor on the 5-day-old
        # synthetic spinup, before the orographic-test-style scenario is even
        # reached. 0.3/0.5 is the largest setting that passes all four previously
        # fragile tests robustly (checked over repeated runs) -- real-terrain
        # tropical precip lands at ~7.9 mm/day, a real but modest improvement over
        # the flat-blend version. See itcz-global-rescale-coupling-2026-07 memory
        # for the full tuning history and why a much stronger fix remains blocked.
        ZONAL_BLEND_MAX = 0.3
        CV_REF = 0.5  # row-CV at which the blend reaches full strength (saturates at 1.0 above this)
        target_profile = _zonal_precip_target_profile(lat_deg)  # (H,), mean(unweighted)=1.0
        target_row_mm_day = target_profile * np.float32(target_mean_mm_day)  # (H,)
        if pp.itcz_seasonal_target_response > 0.0 and pp.itcz_seasonal_response > 0.0:
            # Seasonal deficit-vs-target fix (2026-07-31, direct follow-up to
            # itcz_seasonal_response): that fix gave `itcz_window` a real wet/dry
            # seasonal cycle, but `target_row_mm_day` above is still a pure |lat|
            # shape with zero day_of_year dependence -- every month this rescale
            # was pulling a savanna-latitude row back toward the SAME annual-mean
            # target regardless of season, i.e. refilling exactly the dip the
            # other fix introduced (this was flagged, not fixed, in that fix's
            # own docstring). Fix: let the target itself track the same seasonal
            # signal, so a real dry-season month reads as "near its own target"
            # instead of "in deficit" and doesn't get force-filled.
            #
            # Mean-preserving by construction: the modulation is
            # `1 + k*(itcz_window(day) - itcz_window_annual_mean)`, and
            # `itcz_window_annual_mean` is the actual time-average of
            # `itcz_window` over a full seasonal cycle (see
            # `_itcz_window_annual_mean`), so the modulation's own time-average
            # is exactly 1.0 -- the row's calibrated annual-mean target (and
            # therefore every existing latitude-band/desert/continental
            # calibration built on it) is unaffected in the long run; only the
            # within-year distribution changes. Self-limiting at high latitude
            # (both `itcz_window(day)` and its mean shrink together there) and
            # self-gating at `itcz_seasonal_response=0.0` (itcz_window is then
            # time-invariant, so it always equals its own mean and this is an
            # exact no-op even if `itcz_seasonal_target_response` is nonzero).
            _itcz_window_mean = np.asarray(
                _itcz_window_annual_mean(
                    H,
                    float(pp.itcz_seasonal_response),
                    float(pp.obliquity_deg),
                    float(ITCZ_HALF_WIDTH_DEG),
                ),
                dtype=np.float64,
            )
            # Per-row cap on the modulation strength so the dry-season trough can
            # never remove more than (1 - min_fraction) of a row's annual-mean
            # target (2026-08-05). The additive form below is unbounded below:
            # its trough is `1 - k*(mean - window_min)`, which goes NEGATIVE for
            # any row with `itcz_window_annual_mean > 1/k` -- at k=2.0 that is
            # every row inside ~9 deg of the equator, i.e. exactly the rainforest
            # band. Those rows were surviving only on the `clip(0.05)` floor
            # below, which is a 95% dry-season shutoff and drove deep-tropical
            # driest-month precipitation to <10 mm (Earth: 60-150 mm), erasing Af
            # entirely. Capping `k` at `(1 - f)/window_mean` instead bounds the
            # trough at `f + (1-f)*window_min/window_mean >= f` per row, and
            # binds ONLY where the old form was already clipping: rows with
            # `window_mean <= (1-f)/k` (|lat| >~ 10 deg at the defaults, i.e. the
            # savanna belt this knob exists to serve) keep the full `k`.
            # See PlanetParams.itcz_seasonal_target_min_fraction.
            _k_seasonal = float(pp.itcz_seasonal_target_response)
            _min_frac = float(pp.itcz_seasonal_target_min_fraction)
            if _min_frac > 0.0:
                _k_eff = np.minimum(
                    _k_seasonal,
                    (1.0 - _min_frac) / np.maximum(_itcz_window_mean, 1e-6),
                )
            else:
                _k_eff = _k_seasonal
            _seasonal_target_mod = 1.0 + _k_eff * (
                itcz_window.astype(np.float64) - _itcz_window_mean
            )
            _seasonal_target_mod = np.clip(
                _seasonal_target_mod, max(_min_frac, 0.05), None
            )
            target_row_mm_day = (
                target_row_mm_day.astype(np.float64) * _seasonal_target_mod
            ).astype(np.float32)
            if debug_fields is not None:
                debug_fields["itcz_seasonal_target_modulation"] = _seasonal_target_mod.astype(np.float32)
        # The prognostic condensate path is already physical rainfall. Let it
        # satisfy the same row target before allocating any additional vapor
        # rain, rather than adding a second independent rainfall total after
        # the allocator. The unadjusted target is retained for diagnostics.
        target_row_total_mm_day = target_row_mm_day.copy()
        if bool(pp.enable_prognostic_condensate):
            condensate_row_mm_day = np.mean(
                condensate_precipitation, axis=1, dtype=np.float64
            ).astype(np.float32)
            target_row_mm_day = np.maximum(
                target_row_mm_day - condensate_row_mm_day, 0.0
            ).astype(np.float32)
            if debug_fields is not None:
                debug_fields["condensate_target_share_mm_day"] = condensate_row_mm_day
        dq_before = dq.copy()
        # Terrain-shaped per-cell target weight (mean=1.0 per row by
        # construction, so it reshapes each row's target -- and thus the
        # moisture-budget fill order below -- without touching the row's
        # *total*, which is what the latitude-band calibration tests check).
        # Reused for the post-hoc desert/continental redistribution further
        # down; computed once here so the moisture-budget path (which needs
        # it earlier, to shape the target itself rather than only redistribute
        # an already-fixed row total) and the legacy path share one formula.
        # See DESERT_REDISTRIBUTION_STRENGTH's comment below for the full
        # rationale and history of this specific weight.
        DESERT_REDISTRIBUTION_STRENGTH = 0.9
        _desert_factor = np.clip(
            1.0 - DESERT_REDISTRIBUTION_STRENGTH * (1.0 - subsidence_suppression) * land_f,
            0.05, 1.0,
        ).astype(np.float32, copy=False)
        # The 38-45 degree continental band sits under B1's displaced
        # zonal-mean subsidence maximum.  Once raw production is no longer
        # globally scarce, give land in that band its own share of the row
        # target instead of letting moisture-rich ocean cells consume it.
        # This is mean-preserving after `_row_norm` below and therefore does
        # not over-wet the already-calibrated 40-50 degree zonal total.
        _midlat_rise = np.clip((abs_lat_deg - 36.0) / 2.0, 0.0, 1.0)
        _midlat_rise = _midlat_rise * _midlat_rise * (3.0 - 2.0 * _midlat_rise)
        _midlat_fall = np.clip((47.0 - abs_lat_deg) / 2.0, 0.0, 1.0)
        _midlat_fall = _midlat_fall * _midlat_fall * (3.0 - 2.0 * _midlat_fall)
        _midlat_land_recovery = (_midlat_rise * _midlat_fall)[:, None] * land_f
        _desert_factor = _desert_factor * (1.0 + _midlat_land_recovery)
        # Raw-precip-shape blend (2026-07-31 follow-up to the itcz-seasonal-
        # target-deficit fix): `_desert_factor` above only differentiates via
        # `subsidence_suppression`, which is ~uniformly near 1.0 across the
        # whole ITCZ zone (no subsidence signal to speak of that close to the
        # equator) -- so inside the tropics `cell_weight` was previously a
        # near-exact no-op, even though `global_rescale_factor` averages ~5x
        # on real terrain (2026-07-31 baseline measurement), meaning the
        # aspirational-fill mechanism supplies most of a typical row's final
        # rain and therefore does most of the work of shaping *where* it
        # lands. `precip_potential` (this cell's pre-rescale raw signal) has
        # already been independently verified to correctly rank wet vs dry
        # areas (known-physics-gaps.md item 3b UPDATE 3: continental interior
        # consistently 2.5-3x desert every year, pre-rescale) -- and unlike
        # `subsidence_suppression` it varies meaningfully *within* the ITCZ
        # too (real per-cell wind-convergence differences, e.g. a Congo-basin
        # convergence hotspot vs. a savanna-margin cell at the same latitude).
        # Blending its own row-relative shape into the target gives the
        # aspirational fill a genuine per-cell signal everywhere, not only in
        # the subtropics. Mean-preserving by construction (renormalized below
        # exactly like `_desert_factor` alone was), so row totals -- and the
        # orographic/cloud-cover tests that a purely per-row target would
        # break -- are unaffected; `precip_raw_shape_weight=0.0` is an exact
        # no-op (recovers the original `_desert_factor`-only cell_weight
        # byte-for-byte).
        # Normalized against the row's LAND-only mean (weighted by `land_f`),
        # not the whole row: precip_potential runs much higher over open
        # ocean than land almost everywhere (abundant moisture, no dry-belt/
        # subsidence gating), so a whole-row mean would make every land
        # cell's raw_shape read as "below average" purely from sharing a row
        # with ocean -- diluting/inverting the intended land-internal
        # contrast (measured directly: an earlier whole-row-mean version of
        # this blend collapsed Canadian Prairies/US Midwest/Central Europe
        # ~65-75% and made Kalahari/Atacama *wetter*, because land's
        # relative weight-share was being bid away by ocean's typically
        # higher raw signal in the same row, not reshaped within land).
        _land_pp_sum = np.sum(precip_potential * land_f, axis=1, dtype=np.float64)
        _land_f_sum = np.sum(land_f, axis=1, dtype=np.float64)
        _land_pp_row_mean = (_land_pp_sum / (_land_f_sum + 1e-6)).astype(np.float32)
        _raw_shape = np.clip(
            precip_potential / (_land_pp_row_mean[:, None] + 1e-6), 0.2, 3.0
        ).astype(np.float32, copy=False)
        # Shared by the orographic gate below and the ITCZ-gated blend beneath
        # it. Built only when at least one of them is active, so the historical
        # both-weights-zero configuration does no extra work.
        _orog_shape_w = float(pp.precip_orographic_shape_weight)
        _raw_shape_w = float(pp.precip_raw_shape_weight)
        _land_shape_w = float(pp.precip_land_shape_weight)
        if _orog_shape_w > 0.0 or _raw_shape_w > 0.0 or _land_shape_w > 0.0:
            _raw_shape_applied = np.where(land_f > 0.5, _raw_shape, 1.0).astype(
                np.float32, copy=False
            )
        def _apply_shape_blend(factor, blend):
            # Shared by all three gatings below, which differ only in `blend`.
            #
            # `blend` is clipped to [0, 1] because the expression is a linear
            # interpolation from `factor` toward `factor * _raw_shape_applied`
            # and is only meaningful inside that range. Above 1.0 it extrapolates
            # past the endpoint, and since `_raw_shape` is floored at 0.2 -- which
            # it actually reaches, on 13.9% of land on the tracked benchmark --
            # that drives `_desert_factor` negative (blend 1.7 x shape 0.2 gives
            # -0.36): measured at 20.0% of land for a weight of 1.5. A negative
            # `_desert_factor` becomes a negative `cell_weight`, i.e. a negative
            # share of the row's precipitation target.
            #
            # This fails *silently* without the clip -- downstream clips keep
            # final precipitation non-negative, so the only symptom is a
            # corrupted spatial weighting. No shipped default reaches the
            # extrapolating range, but these weights exist to be swept as
            # ablation handles (A5-LEAD swept one to 5.0), so the guard earns its
            # keep. See testing/test_land_shape_blend.py.
            blend = np.clip(blend, 0.0, 1.0).astype(np.float32)
            return factor * (1.0 - blend + blend * _raw_shape_applied)

        # A5 stage 3, added 2026-08-02. The moisture-budget fill is a
        # *deficit-filling* mechanism: it tops each cell up toward a target, so
        # wherever it supplies most of the rain it actively erases whatever
        # spatial contrast raw production created. That is why an orographic gain
        # upstream does not survive to final precipitation -- measured on
        # saves/earth.pkl, the S Andes pair reaches a `precip_potential` ratio of
        # 1.69 and comes out at 0.88 in final P, i.e. inverted.
        #
        # The fix is to make the *target* orographically aware, which is process
        # note 9's rule ("check which side of the rescale a mechanism sits on")
        # applied to orography. The identical raw-shape blend already exists
        # directly below for the tropics, but it is gated by `itcz_window` and so
        # cannot reach a mid-latitude mountain range at all. This gate is the
        # orographic signal itself, so the blend acts only where orography is
        # doing something and leaves the validated desert/continental mid-latitude
        # weighting untouched everywhere else.
        if _orog_shape_w > 0.0:
            _orog_gate = np.clip(
                orog / max(float(pp.orographic_uplift_clip), 1e-6), 0.0, 1.0
            )
            _desert_factor = _apply_shape_blend(
                _desert_factor, _orog_shape_w * _orog_gate * (land_f > 0.5)
            )
        if _land_shape_w > 0.0:
            # Land-wide, ungated by latitude or terrain. Ships inert at 0.0 and is
            # expected to stay that way: this is the mechanism audit A5-OROG's
            # deferred lead claimed to be "effectively" measuring, and testing it
            # honestly (2026-08-02, audit A5-LEAD) refuted the lead. Enabled, it
            # degrades every bounded H10 metric monotonically and at 1.0
            # reproduces the 2026-07-31 rejection of the ungated
            # `precip_raw_shape_weight` almost exactly -- arid share 27.5->33.2%,
            # US Midwest 900->399 mm/yr.
            #
            # The lead's actual gain came from the buggy normalizer making the
            # *orographic* gate terrain-weighted-but-broad, not from breadth as
            # such; a uniform land-wide blend does not reproduce it. Kept as a
            # tested mechanism (process note 2) so the next session to reach for
            # this finds the audit's sweep table instead of rebuilding it.
            _desert_factor = _apply_shape_blend(
                _desert_factor, _land_shape_w * (land_f > 0.5)
            )
        if _raw_shape_w > 0.0:
            # Gated by `itcz_window` (real-terrain measurement, 2026-07-31):
            # an ungated version -- applying this blend at every latitude --
            # nearly doubled `arid_pct` (19.5%->28.4% at w=1.0) and, worse,
            # *decreased* Aw (8.8%->6.4%) instead of increasing it: cells
            # below their row's land-mean raw signal were being pushed
            # straight past Koppen's aridity threshold into BW/BS instead of
            # landing in the Aw "real dry season, not full desert" band. Two
            # of three continental-interior boxes also lost significant
            # precip (US Midwest 610->400, Canadian Prairies 462->377
            # mm/yr) -- collateral damage to latitudes where
            # `subsidence_suppression`-based `_desert_factor` was already
            # doing its job correctly (this project's own hard-won
            # desert-vs-continental ranking work, see
            # moisture-budget-desert-ceiling-fix-2026-07-28 and
            # ferrel-land-ocean-split-2026-07-26). This mechanism was
            # motivated specifically by `subsidence_suppression` being a
            # near-no-op *inside the ITCZ*, not by any deficiency at
            # mid-latitudes -- gating by `itcz_window` confines it to where
            # the gap it targets actually exists, leaving the
            # already-validated mid-latitude mechanism untouched.
            _raw_shape_w_row = (_raw_shape_w * itcz_window).astype(np.float32)[:, None]
            # Ocean cells keep a neutral factor of 1.0 in the blend -- their
            # share of the row's target is untouched, exactly like
            # `_desert_factor` alone -- so only land is reshaped, and only
            # relative to other land in the same row. (`_raw_shape_applied` is
            # now built just above, shared with the orographic gate.)
            _desert_factor = _apply_shape_blend(_desert_factor, _raw_shape_w_row)
        # SST -> land precipitation, target side (audit D3, 2026-08-05).
        #
        # The suppression gate up in the subsidence block is the intuitive place
        # for this and is measurably inert: it sits *upstream* of the rescale, so
        # the deficit fill tops the cell back up toward a target that never heard
        # about the SST anomaly. Process note 9's rule, third instance -- the
        # target is the lever, not raw production.
        #
        # Here the anomaly moves the cell's *share of its row's target*: land
        # downwind of water that is warm for its latitude claims more of the row,
        # land downwind of cold water claims less. Renormalized by `_row_norm`
        # immediately below like every other term here, so it is exactly
        # row-mean-preserving -- it redistributes within a latitude circle and
        # cannot change a zonal total the model is already calibrated against.
        #
        # Land-only and multiplicative on `_desert_factor` rather than routed
        # through `_apply_shape_blend`: those three blends all interpolate toward
        # `_raw_shape`, a *production* signal, whereas this is an independent
        # physical driver with its own sign. Floored well above zero so a cold
        # anomaly can suppress a cell's share but never zero or invert it.
        #
        # **Saturating, not linear in kelvin, and that is load-bearing.** A
        # linear response bounded only by a clip lets a single outlier cell take
        # the whole bound: the model's Kuroshio anomaly reaches +1.9 K where the
        # land-cell median is 0.35 K, so at weight 0.9 the S Japan box claimed
        # 2.7x its row's target share and came out at 2809 mm/yr against a
        # 1100-2200 target at 256x512 -- a real regression traceable entirely to
        # the tail. `tanh` keeps the response very nearly linear over the
        # +-1 K where most coastal land sits while bounding the tail, so the
        # weight means "largest fractional change in target share" rather than
        # "change per kelvin, unbounded until the clip". The 2.0 K reference is
        # a fixed constant rather than a knob: it is a property of how far an
        # SST anomaly can plausibly shift a boundary layer, not something to
        # tune per planet, and the strength knob already spans that freedom.
        #
        # **Cold side only, and that is a physics decision rather than a
        # tuning one.** The suppressing half is a mechanism this model has
        # nothing else for: cold water caps the boundary layer above it with a
        # marine inversion, which is why Atacama and the Namib are hyper-arid
        # *on a coastline*. The warming half is already represented twice --
        # ocean evaporation responds to SST through `qsat`, and
        # `monsoon_east_margin_exemption` was calibrated at 3.0 specifically for
        # SE US / East China / S Japan, i.e. the exact warm-boundary-current
        # margins a symmetric form would boost again. Measured, that
        # double-count is not subtle: the symmetric version drives S Japan to
        # 2633 mm/yr against its 1100-2200 target at 256x512, while the
        # cold-only form leaves every warm margin untouched and keeps the
        # desert gains. `np.minimum(..., 0)` rather than a second knob, because
        # a warm-side strength that must stay at zero is not a freedom.
        if _sst_target_w != 0.0 and _sst_anom_field is not None:
            _sst_target_factor = np.clip(
                1.0
                + _sst_target_w
                * np.minimum(
                    np.tanh(_sst_anom_field / _SST_COUPLING_REFERENCE_K), 0.0
                )
                * land_f,
                0.2,
                3.0,
            ).astype(np.float32, copy=False)
            _desert_factor = _desert_factor * _sst_target_factor
            if debug_fields is not None:
                debug_fields["sst_target_factor"] = _sst_target_factor
        _row_norm = np.mean(_desert_factor, axis=1, dtype=np.float64).astype(np.float32)
        cell_weight = _desert_factor / (_row_norm[:, None] + 1e-6)
        # The column-water migration deliberately starts by removing the two
        # imposed row-target corrections.  ``dq`` is already capped by the
        # available local vapour above; leaving it unscaled therefore gives a
        # directly auditable precipitation sink for the existing prognostic
        # humidity reservoir.  This remains opt-in until the climate score
        # establishes that its source/sink closure is an improvement.
        use_row_target_rescale = (
            bool(pp.moisture_budget_precip_rescale)
            and not bool(pp.enable_prognostic_column_water)
        )
        if use_row_target_rescale:
            # Regime-varying removal caps (2026-08-01, A5 follow-up): raises
            # both the total- and per-step-added removal-fraction caps inside
            # the ITCZ only, via `moisture_budget_tropical_cap_boost` (0.0 =
            # exact no-op, both rows equal the prior flat 0.85/0.15
            # everywhere). Motivated by a direct measurement showing the
            # moisture-budget rescale is capacity-limited (pinned at these
            # caps, still short of target) at essentially every latitude
            # today, including the tropics (~0.85-0.94 achieved fraction)
            # despite sitting on the planet's most abundant moisture supply --
            # see PlanetParams.moisture_budget_tropical_cap_boost docstring
            # for the full measurement and why this is scoped differently
            # from the previously-reverted flat/global cap raise (which
            # touched every latitude, including fragile mid-latitude/dry-belt
            # rows, and made desert/continental ranking worse).
            _cap_boost = float(pp.moisture_budget_tropical_cap_boost)
            total_cap_row = np.clip(
                0.85 + _cap_boost * 0.10 * itcz_window, 0.0, 0.95
            ).astype(np.float64)
            added_cap_row = np.clip(
                0.15 + _cap_boost * 0.15 * itcz_window, 0.0, 0.30
            ).astype(np.float64)
            budget_dq_cap = np.minimum(
                total_cap_row[:, None] * q, dq_before + added_cap_row[:, None] * q
            )
            drybelt_land_protection = np.clip(
                1.0 - 0.995 * drybelt_window[:, None] * land_f * (
                    1.0 - float(pp.monsoon_east_margin_exemption) * monsoon_margin_factor
                ),
                0.005,
                1.0,
            )
            dynamic_affinity = np.where(
                land_f > 0.5,
                np.clip(subsidence_suppression, 0.005, 1.0),
                1.0,
            )
            dq, budget_diag = _moisture_budget_precip_rescale(
                dq,
                q,
                target_row_mm_day,
                dt_days=dt,
                column_mm_per_q=column_mm_per_q,
                allocation_affinity=drybelt_land_protection * dynamic_affinity,
                target_cell_weight=cell_weight,
                max_total_removal_fraction=total_cap_row,
                max_added_removal_fraction=added_cap_row,
            )
            scale_row = budget_diag["effective_scale"]
        elif not pp.enable_prognostic_column_water:
            # dtype=float64 accumulation: headless vs. threaded/blocked call paths can
            # feed this reduction the same float32 values in a different summation
            # order (chunk boundaries), which float32 accumulation is sensitive enough
            # to for the two paths to disagree at the ~1e-4 level once rescaled back
            # across the whole row (see test_headless_matches_threaded_call_pattern).
            # Accumulating in float64 makes the reduction itself effectively
            # order-invariant at float32 precision.
            P_zonal_mean64 = np.mean(P, axis=1, dtype=np.float64)  # (H,)
            P_zonal_mean = P_zonal_mean64.astype(np.float32)
            P_row_std64 = np.std(P, axis=1, dtype=np.float64)  # (H,)
            row_cv = (P_row_std64 / (np.abs(P_zonal_mean64) + 1e-6)).astype(np.float32)
            zonal_blend_row = ZONAL_BLEND_MAX * np.clip(row_cv / CV_REF, 0.0, 1.0)
            flat_scale = float(np.clip(target_mean_mm_day / (float(np.mean(P, dtype=np.float64)) + 1e-6), 0.2, 3.0))
            raw_zonal_scale = target_row_mm_day / (P_zonal_mean + 1e-6)
            scale_row = np.clip(
                flat_scale + zonal_blend_row * (raw_zonal_scale - flat_scale), 0.15, 5.0
            ).astype(np.float32)
            dq = np.clip(dq * scale_row[:, None], 0.0, q)
        else:
            # Diagnostic raw-conservation path: neither the bounded allocator
            # nor the legacy multiplicative row rescale may alter the local
            # condensation sink.
            scale_row = np.ones(q.shape[0], dtype=np.float32)
        P = dq * (column_mm_per_q / dt)
        if debug_fields is not None:
            debug_fields["zonal_rescale_factor"] = scale_row
            # Backward-compat scalar (unweighted mean, matching the old flat-scalar
            # convention): callers that only look at a single "how hard did the
            # rescale have to work overall" number still get one.
            debug_fields["global_rescale_factor"] = float(np.mean(scale_row))
            debug_fields["precip_rescale_dq_added"] = np.maximum(dq - dq_before, 0.0)
            debug_fields["column_water_mode"] = (
                "raw_prognostic" if pp.enable_prognostic_column_water
                else "row_target_budget" if use_row_target_rescale
                else "legacy_row_rescale"
            )
            if use_row_target_rescale:
                debug_fields["precip_target_achieved_fraction"] = budget_diag[
                    "target_achieved_fraction"
                ]
                debug_fields["precip_rescale_capacity_limited"] = budget_diag[
                    "capacity_limited"
                ]
                debug_fields["precip_rescale_unmet_mm_day"] = budget_diag[
                    "unmet_row_mm_day"
                ]
        # Desert-vs-continental redistribution (2026-07 follow-up session).
        # EARLIER ATTEMPT, REVERTED: a blanket post-rescale "desert suppression"
        # multiplier (1 - k*drybelt_window*land_f) was tried to counter the zonal
        # rescale inflating dry-belt LAND cells (real deserts) more than the old
        # flat scalar did. It was too blunt: drybelt_window*land_f applies
        # uniformly to ALL land at a dry-belt latitude, not just actually-arid
        # land -- on a 64x128 synthetic fixture it crushed the 20-30N band to
        # ~30% of its target (298 vs 1000 mm/yr) because most of that fixture's
        # land at that latitude sits near the drybelt center, and even a strong
        # coefficient (k=0.97) only partially fixed real-terrain Sahara
        # (877->513 mm/yr, still >2x the <200 target) while regressing
        # test_latitude_band_precip_bias_reasonable. Also, critically, it wasn't
        # mean-preserving -- it just multiplied dq down, silently lowering each
        # row's *total* rainfall rather than only reshaping where it falls, which
        # is what let it fight the zonal-rescale calibration above.
        #
        # THIS ATTEMPT: redistribute, don't suppress. Uses the existing
        # `subsidence_suppression` field (wind-divergence-derived, already
        # differentiates genuinely subsiding/arid land from convergent/wet land
        # at the *same* latitude -- see the desert-evapotranspiration-fix-2026-07
        # memory for how that signal was validated) to build a per-cell weight,
        # renormalized so each row's *mean* weight is exactly 1.0 -- a
        # redistribution of each row's already-calibrated target, not a change to
        # its magnitude. This has two structural advantages over the reverted
        # attempt: (1) it cannot fight the zonal-rescale calibration or
        # `test_latitude_band_precip_bias_reasonable`, since every row's total is
        # unchanged by construction; (2) it's a no-op wherever a row has no
        # subsidence/land heterogeneity to redistribute -- e.g. the orographic
        # test's uniform wind field gives uniform `subsidence_suppression` within
        # every row, so `cell_weight` normalizes to exactly 1.0 there and this
        # step touches nothing, unlike the row-level zonal correction above (which
        # needed the heterogeneity gating specifically because it changes each
        # row's *magnitude*, not just its internal shape).
        #
        # `cell_weight` itself is computed once, earlier (before the
        # moisture-budget-rescale branch above), because the budget path now
        # reuses it as `target_cell_weight` to shape the *target* each cell
        # chases (razor-sharp-biome-line-precip-target-smoothing-2026-07-28
        # memory) rather than only redistributing an already-fixed row total
        # after the fact. Applying this post-hoc multiply on top of that would
        # double-apply the same signal and re-introduce the asymmetric-clipping
        # problem that motivated moving it upstream (`budget_dq_cap` bounds how
        # much a cell's rain can *increase* relative to its pre-rescale value,
        # but not how much this step can *decrease* it -- stacking both passes
        # would bias the net result toward drying, not just reshaping). Only
        # the legacy (non-budget) path still needs it applied here, since that
        # path has no other terrain-aware mechanism at all.
        if not use_row_target_rescale and not pp.enable_prognostic_column_water:
            dq = np.clip(dq * cell_weight, 0.0, q)
        P = dq * (column_mm_per_q / dt)
        if debug_fields is not None:
            debug_fields["desert_redistribution_weight"] = cell_weight
    rain_export_factor = np.clip(
        0.94 - 0.14 * itcz_window[:, None] + 0.08 * storm_window[:, None],
        0.70,
        1.06,
    ).astype(np.float32, copy=False)
    dq = np.clip(dq * rain_export_factor, 0.0, q)
    if use_row_target_rescale and target_mean_mm_day > 0.0:
        dq = np.minimum(dq, budget_dq_cap)
    P = dq * (column_mm_per_q / dt)
    if max_precip_mm_day > 0.0:
        cap = np.minimum(1.0, max_precip_mm_day / (P + 1e-9))
        dq = dq * cap
        P = P * cap
    if pp.enable_prognostic_column_water:
        # Conservative transport can carry humid air into a colder cell.  Rain
        # out any resulting supersaturation explicitly instead of letting the
        # final humidity clip silently discard that column water.
        dq = np.clip(np.maximum(dq, q - qsat), 0.0, q)
        P = dq * (column_mm_per_q / dt)
    vapor_precipitation = P.copy()
    P = P + condensate_precipitation
    if (
        debug_fields is not None
        and use_row_target_rescale
        and target_mean_mm_day > 0.0
    ):
        final_row_mean = np.mean(P, axis=1, dtype=np.float64)
        final_achieved = final_row_mean / (
            np.asarray(target_row_total_mm_day, dtype=np.float64) + 1e-12
        )
        final_scale = np.mean(dq, axis=1, dtype=np.float64) / (
            np.mean(dq_before, axis=1, dtype=np.float64) + 1e-12
        )
        debug_fields["zonal_rescale_factor"] = final_scale.astype(np.float32)
        debug_fields["global_rescale_factor"] = float(np.mean(final_scale))
        debug_fields["precip_target_achieved_fraction"] = final_achieved.astype(np.float32)
        debug_fields["precip_rescale_capacity_limited"] = final_achieved < 0.999
        debug_fields["precip_rescale_unmet_mm_day"] = np.maximum(
            np.asarray(target_row_total_mm_day, dtype=np.float64) - final_row_mean,
            0.0,
        ).astype(np.float32)

    if debug_fields is not None:
        debug_fields["precipitation_final_mm_day"] = P
        debug_fields["vapor_precipitation_mm_day"] = vapor_precipitation

    # Update humidity and soil moisture reservoirs
    lower_humidity_next = np.clip(q - dq, 0.0, qsat)
    if two_layer_active:
        midlevel_humidity_next = np.clip(midlevel_humidity_next, 0.0, 0.035)
        if three_level_active:
            upperlevel_humidity_next = np.clip(upperlevel_humidity_next, 0.0, 0.035)
            humidity_next = (
                lower_humidity_next
                if closed_three_level_active
                else lower_humidity_next + midlevel_humidity_next + upperlevel_humidity_next
            )
        else:
            humidity_next = lower_humidity_next + midlevel_humidity_next
    else:
        humidity_next = lower_humidity_next
    if debug_fields is not None:
        # Transitional column-water accounting.  ``q`` is still the legacy
        # mixing-ratio proxy, but these fields express every storage/removal in
        # the same mm-column units used by the future prognostic closure.  The
        # equality below is a local identity for allocator rain; any mismatch
        # is explicitly reported instead of being hidden by row rescaling.
        # Outside the gated two-layer experiment ``q`` remains the complete
        # legacy humidity reservoir, and there is no upper partition to add.
        # Keeping that branch explicit preserves the long-standing diagnostic
        # identity for every default configuration.
        vapor_before_precip = (
            q + midlevel_humidity_next + upperlevel_humidity_next
            if three_level_active
            else q + midlevel_humidity_next
            if two_layer_active
            else q
        )
        before_precip_mm = vapor_before_precip * column_mm_per_q
        allocator_removal_mm = dq * column_mm_per_q
        after_precip_mm = humidity_next * column_mm_per_q
        debug_fields["column_water_before_precip_mm"] = before_precip_mm.astype(np.float32)
        debug_fields["column_water_after_precip_mm"] = after_precip_mm.astype(np.float32)
        debug_fields["column_water_rainout_removal_mm"] = allocator_removal_mm.astype(np.float32)
        # Compatibility name retained while diagnostics migrate; in raw mode
        # this is a physical rainout sink, not an allocator adjustment.
        debug_fields["column_water_allocator_removal_mm"] = allocator_removal_mm.astype(np.float32)
        debug_fields["column_water_allocator_residual_mm"] = (
            after_precip_mm - (before_precip_mm - allocator_removal_mm)
        ).astype(np.float32)
        debug_fields["column_water_evaporation_source_mm"] = (
            (ocean_evap + land_evap) * dt_evap * column_mm_per_q
        ).astype(np.float32)
        if closed_three_level_active:
            debug_fields["closed_column_lower_temperature_k"] = (
                closed_lower_temperature_next.astype(np.float32)
            )
            debug_fields["closed_column_water_residual_kg_m2"] = float(
                closed_column_water_residual
            )
            debug_fields["closed_column_mse_residual_j_m2"] = float(
                closed_column_energy_residual
            )
            debug_fields["closed_column_radiative_source"] = "resolved_host_temperature_step"
        if pp.enable_prognostic_column_water:
            # Whole-call closure for the experimental system.  Transport may
            # redistribute individual cells, so this is intentionally
            # area-weighted globally rather than a false per-cell assertion.
            condensate_start = (
                np.zeros_like(base_q)
                if condensate is None
                else np.asarray(condensate, dtype=np.float64)
            )
            condensate_end = (
                np.zeros_like(base_q)
                if condensate_next is None
                else np.asarray(condensate_next, dtype=np.float64)
            )
            hydrometeor_start = (
                np.zeros_like(base_q)
                if precipitating_hydrometeors is None
                else np.asarray(precipitating_hydrometeors, dtype=np.float64)
            )
            hydrometeor_end = (
                np.zeros_like(base_q)
                if hydrometeors_next is None
                else np.asarray(hydrometeors_next, dtype=np.float64)
            )
            # ``q`` has since been transported and rained out; reconstruct the
            # actual source from the pre-transport saturation-limited value.
            source_q = np.maximum(
                np.clip(lower_base_q + sources, 0.0, qsat) - lower_base_q,
                0.0,
            )
            effective_source_mm = source_q * column_mm_per_q
            water_before_mm = (base_q + condensate_start + hydrometeor_start) * column_mm_per_q
            water_after_mm = (humidity_next + condensate_end + hydrometeor_end) * column_mm_per_q
            rainout_mm = P * dt
            expected_after_mm = water_before_mm + effective_source_mm - rainout_mm
            closure_residual = float(np.sum((water_after_mm - expected_after_mm) * area_m2))
            closure_scale = max(
                float(np.sum(np.abs(water_before_mm) * area_m2)), 1.0
            )
            debug_fields["column_water_total_budget_residual_mm_m2"] = closure_residual
            debug_fields["column_water_total_budget_relative_residual"] = (
                closure_residual / closure_scale
            )

    # Both terms must scale with the *same* elapsed-time basis. The precip
    # replenishment previously didn't scale with dt at all (a no-op bug at dt=1
    # DAILY mode, but under-replenishing ~30x at dt~30 MONTHLY mode). The drain
    # term scaled with the full dt, but land_evap itself only actually reached the
    # humidity reservoir up to dt_evap (<=1.5d, capped above to avoid saturating q
    # in one step) -- so draining soil by the *uncapped* dt double-charged it for
    # evaporation that never actually left the soil into the air. Together these
    # drove continental-interior soil moisture to its 0.05 floor within a few
    # decades of MONTHLY-mode spinup, which then throttles land_evap itself
    # (0.35+0.65*soil factor) in a self-reinforcing desiccation spiral that
    # collapsed precip to ~12 mm/yr (Earth: 350-450 mm/yr for e.g. the Canadian
    # Prairies).
    #
    # FOLLOW-UP FIX (2026-07): soil was saturating to its 1.0 ceiling almost
    # everywhere on land except near the poles (measured 0.96-1.00 in every
    # non-polar latitude band, real downsampled Earth terrain and synthetic
    # terrain alike) -- the desiccation-spiral fix above swapped a floor-collapse
    # bug for an equally-uninformative ceiling-saturation one, and the soil
    # bucket had lost essentially all spatial discriminating power between wet
    # and dry regions. The gain/drain balance here is a genuinely bistable
    # system (via land_evap's 0.35+0.65*soil feedback): sweeping the gain
    # coefficient from 0.0006 down to 0.00015 found a sharp bifurcation between
    # 0.00025 (soil stays pinned ~0.96-0.99, no desert improvement) and 0.00015
    # (soil properly de-saturates and differentiates, drybelt land precip drops
    # ~40%, e.g. 350->214 mm/yr on the synthetic 60yr fixture), with no stable
    # middle ground. The de-saturated regime pushes SH mid-lat *ocean* precip to
    # ~4.0-4.07 mm/day via the shared target_mean_mm_day rescale -- reproducing
    # the same regression a prior evap_suppression attempt hit (see
    # test_climate_drift.py's module docstring), just via a different mechanism.
    # Accepted this time (2026-07 decision) as a worthwhile trade: the desert/
    # continental-interior realism gain is substantial and the SH mid-lat ocean
    # band's cap was widened accordingly (test_earth_benchmark.py,
    # test_midlat_precip_quantity) rather than leaving the ceiling-saturation
    # bug in place.
    soil += (P * land_f) * 0.00015 * dt - (land_evap * dt_evap) * 0.4

    # 2-LAYER BUCKET (2026-07): the floor/ceiling above is a safety net, not a
    # differentiator -- item FOLLOW-UP FIX above found the single-layer gain/drain
    # balance is genuinely bistable (soil either saturates near 1.0 everywhere or
    # collapses to the 0.05 floor, with no stable middle ground under one global
    # gain constant), which was silently collapsing non-desert continental interior
    # onto the same low branch used to fix desert over-wetting. A slow deep/
    # root-zone reservoir breaks that: its large capacity and slow time constant
    # can't snap between two states in one spinup, so each region's real long-term
    # precip/evap balance can settle at its own differentiated equilibrium instead.
    #
    # The deep layer is fed *directly* by precipitation (like the surface layer,
    # just at a much smaller rate), not via surface-moisture overflow/percolation --
    # an earlier version gated it on the surface exceeding a field-capacity
    # threshold, which measured out as a chicken-and-egg trap: the surface layer is
    # *already* pinned at its 0.05 floor (the exact bug being fixed), so it can
    # never climb back above a threshold on its own, and the deep layer just decays
    # to zero with no input. Feeding it directly from P sidesteps that: its
    # equilibrium reflects each region's real long-run precip rate, independent of
    # whichever branch the bistable surface layer happens to be on.
    soil_deep += (P * land_f) * float(pp.soil_deep_gain_rate) * dt

    # Deep-layer drain: slow baseflow/groundwater loss -- a genuine sink (water
    # leaves the system, unlike the surface layer's evaporation which stays in the
    # atmosphere-humidity budget). Gives the reservoir real multi-year memory
    # instead of another instant-equilibrium bucket.
    soil_deep -= float(pp.soil_deep_drain_rate) * soil_deep * dt

    soil = np.where(land_mask, np.clip(soil, 0.05, 1.0), 0.0)
    soil_deep = np.where(land_mask, np.clip(soil_deep, 0.0, 1.0), 0.0)

    result = (
        P.astype(np.float32),
        humidity_next.astype(np.float32),
        soil.astype(np.float32),
        soil_deep.astype(np.float32),
    )
    if return_condensate:
        if condensate_next is None:
            condensate_next = np.zeros_like(q, dtype=np.float32)
        if return_midlevel_temperature:
            if midlevel_temperature_next is None:
                midlevel_temperature_next = np.zeros_like(q, dtype=np.float32)
            if return_midlevel_humidity:
                if midlevel_humidity_next is None:
                    midlevel_humidity_next = np.zeros_like(q, dtype=np.float32)
                if return_upperlevel_state:
                    if upperlevel_temperature_next is None:
                        upperlevel_temperature_next = np.zeros_like(q, dtype=np.float32)
                    if upperlevel_humidity_next is None:
                        upperlevel_humidity_next = np.zeros_like(q, dtype=np.float32)
                    if return_precipitating_hydrometeors:
                        if hydrometeors_next is None:
                            hydrometeors_next = np.zeros_like(q, dtype=np.float32)
                        return (
                            *result,
                            np.asarray(condensate_next, dtype=np.float32),
                            np.asarray(midlevel_temperature_next, dtype=np.float32),
                            np.asarray(midlevel_humidity_next, dtype=np.float32),
                            np.asarray(upperlevel_temperature_next, dtype=np.float32),
                            np.asarray(upperlevel_humidity_next, dtype=np.float32),
                            np.asarray(hydrometeors_next, dtype=np.float32),
                        )
                    return (
                        *result,
                        np.asarray(condensate_next, dtype=np.float32),
                        np.asarray(midlevel_temperature_next, dtype=np.float32),
                        np.asarray(midlevel_humidity_next, dtype=np.float32),
                        np.asarray(upperlevel_temperature_next, dtype=np.float32),
                        np.asarray(upperlevel_humidity_next, dtype=np.float32),
                    )
                return (
                    *result,
                    np.asarray(condensate_next, dtype=np.float32),
                    np.asarray(midlevel_temperature_next, dtype=np.float32),
                    np.asarray(midlevel_humidity_next, dtype=np.float32),
                )
            return (
                *result,
                np.asarray(condensate_next, dtype=np.float32),
                np.asarray(midlevel_temperature_next, dtype=np.float32),
            )
        return (*result, np.asarray(condensate_next, dtype=np.float32))
    return result
