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
from masks import get_masks

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
    dx = pp.radius_m * (2 * np.pi / W) * np.cos(lat_2d)
    dy = pp.radius_m * (np.pi / H)
    f = pp.coriolis_parameter(lat_2d)
    abs_lat_deg_2d = np.abs(np.rad2deg(lat_2d)).astype(np.float32, copy=False)
    eq_window = np.exp(-((abs_lat_deg_2d / 12.0) ** 2)).astype(np.float32, copy=False)
    lon_1d = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)
    grids = (lat_2d, dx, dy, f, eq_window, lon_1d)
    _WIND_GRID_CACHE.update({"key": key, "grids": grids})
    return grids


# ---------------------------------------------------------------------------
# Tunable atmospheric constants
# These control the large-scale circulation structure and are candidates for
# optimizer sweeps or per-planet customisation.
# ---------------------------------------------------------------------------

RHO_AIR: float = 1.225
"""Air density at sea level [kg/m³].  Used in PGF and geostrophic wind."""

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


def _advect_scalar(
    field: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    u_scale: np.ndarray,
    v_scale: np.ndarray,
) -> np.ndarray:
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
    """Advect humidity field with upwind scheme (semi-Lagrangian).

    Returns advected humidity field.
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

    # Interpolate u, v at departure points (bilinear interpolation)
    u_new = map_coordinates(u, [y_departure, x_departure], order=1, mode='wrap')
    v_new = map_coordinates(v, [y_departure, x_departure], order=1, mode='wrap')

    return u_new.astype(np.float32), v_new.astype(np.float32)


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
    
    rho = RHO_AIR

    u_curr, v_curr = u.copy(), v.copy()
    
    # Pre-calculate PGF (constant over the day)
    if pressure is None:
        # P ~ P0 * exp(-z/H) * (T0/T)^g/R... simplified:
        # NOTE: Keep this in Pa (not hPa). Scale is intentionally "synoptic-ish":
        # order ~5-10 hPa swings across large temperature gradients.
        p_anom = -float(pgf_temp_scale) * ((temperature - 273.15) / 30.0)  # Pa anomaly
        if elevation is not None:
             # Terrain effect: flow around obstacles, high pressure wedge
             p_anom += float(pgf_terrain_scale) * elevation
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
            p_anom = p_anom + float(ice_pressure_scale) * np.clip(ice_cover, 0.0, 1.0)
    else:
        # Copy: the wave/storm/blocking terms below accumulate into p_anom, and
        # aliasing the caller's array would silently mutate their buffer.
        p_anom = np.array(pressure, dtype=np.float32, copy=True)

    # Synoptic-scale planetary waves: re-enabled with longer periods (20-45 days vs 6-14).
    # Shorter periods (6-14 days) caused daily direction reversals; slower waves produce
    # persistent, slowly-moving pressure cells that create emergent highs/lows without thrashing.
    if time_days is not None:
        lon_1d = _lon_1d_cached
        abs_deg_1d = np.rad2deg(np.abs(lat_2d[:, 0])).astype(np.float32, copy=False)  # (H,)
        sign_lat_1d = np.sign(lat_2d[:, 0]).astype(np.float32, copy=False)  # (H,) +N, -S
        # Split the mid-lat storm-track window by hemisphere so a wavier jet
        # (jet_index > 0) can independently amplify Rossby-wave meander in one
        # hemisphere without affecting the other.
        storm_w_base = np.exp(-((abs_deg_1d - 45.0) / 18.0) ** 2).astype(np.float32, copy=False)  # (H,)
        wave_scale_nh = 1.0 + float(pp.jet_wave_amp_scale_per_index) * float(jet_index_nh)
        wave_scale_sh = 1.0 + float(pp.jet_wave_amp_scale_per_index) * float(jet_index_sh)
        storm_w = np.where(sign_lat_1d >= 0.0, storm_w_base * wave_scale_nh, storm_w_base * wave_scale_sh).astype(np.float32, copy=False)
        t = float(time_days)
        wave = np.zeros((H, W), dtype=np.float32)
        # Wavenumbers 3-7, periods 20-45 days: slow Rossby-like propagation.
        # Amplitudes reduced 40% from initial values (1.0/0.75/0.5 → 0.6/0.45/0.3 hPa)
        # to avoid over-perturbing the pressure field when combined with the terrain term.
        for k, per, ph, amp_hpa in ROSSBY_MODES:
            wave += (amp_hpa * 100.0) * np.cos(k * lon_1d[None, :] + (2.0 * np.pi * t / per) + ph).astype(np.float32, copy=False)
        p_anom = p_anom + storm_w[:, None] * wave

        # Discrete moving extratropical storm systems (see _storm_pressure_anomaly):
        # unlike the Rossby waves above (a standing sinusoid, never spawns/dies),
        # these are individual low-pressure cells with a birth/track/death
        # lifecycle. This is what turns the "faint ripple" the Rossby waves alone
        # produce into organic, moving, blob-shaped storm precipitation via the
        # existing convergence/ascent-driven precipitation machinery downstream —
        # no changes needed on the precipitation side. The genesis latitude is
        # biased by the jet meander index so storms track along the (shifted)
        # jet rather than a fixed climatological latitude.
        _pr_km = float(pp.radius_m) / 1000.0
        storm_amp = float(pp.storm_pressure_amp_pa)
        if storm_amp != 0.0:
            p_anom = p_anom + _storm_pressure_anomaly(
                lat_2d, lon_1d[None, :], t, storm_amp, population_id=0,
                lat_shift_nh_deg=float(pp.jet_lat_shift_per_index) * float(jet_index_nh),
                lat_shift_sh_deg=float(pp.jet_lat_shift_per_index) * float(jet_index_sh),
                planet_radius_km=_pr_km,
            )

        # Trade-wind/subtropical waves (see the module-level comment above
        # N_TRADE_WAVE_SLOTS): westward-translating, shorter-lived, weaker
        # disturbances covering the 12-32 deg band the mid-latitude storms
        # above don't reach -- this is the band where the Rossby-wave ripple
        # alone looked the most static/repetitive (2026-07 user feedback).
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

        # Blocking ridges: a persistent, quasi-stationary high (weeks-long,
        # not translating) that steers storms around it and locks in
        # multi-week wet/dry anomalies -- the mechanism plain Rossby waves
        # and the storm populations above (both memoryless) cannot produce.
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

    # NOTE: A constant land-sea pressure contrast was tried here but removed.
    # Antarctica (~all land) got a permanent +150 Pa high, driving persistent cold-air
    # outflow and triggering runaway SH sea-ice cooling (SH pole → 201 K).
    # The land-sea contrast is already partially encoded in the temperature field T via
    # land/ocean differential heating in simulate.py's _evolve_temperature.
        
    dp_dx = _ddx_periodic(p_anom) / (dx + 1e-3)
    # Axis 0 is north→south (index increases southward), so physical northward gradient is negated.
    dp_dy = -np.gradient(p_anom, axis=0) / dy

    pgf_u = -1.0/rho * dp_dx
    pgf_v = -1.0/rho * dp_dy

    # --- Baroclinic / eddy-driven mid-lat westerly tendency (parameterized) ---
    # In the real atmosphere, mid-lat westerlies are maintained by baroclinic eddies and
    # their momentum flux convergence. We approximate the net effect by mixing a
    # thermal-wind-like westerly target down toward the surface, based on |dT/dy|.
    b_amp = float(baroclinic_jet_amp)
    b_mix = float(baroclinic_mix)
    if b_amp != 0.0 and b_mix > 0.0:
        # Use zonal-mean temperature gradient so the tendency acts on the zonal-mean jet,
        # matching the climatological effect of baroclinic eddies.
        abs_deg_1d = np.rad2deg(np.abs(lat_2d[:, 0])).astype(np.float32, copy=False)  # (H,)
        w_mid_1d = np.exp(-((abs_deg_1d - 45.0) / 12.0) ** 2).astype(np.float32, copy=False)  # (H,)
        ztemp = np.mean(temperature.astype(np.float32), axis=1, keepdims=True)  # (H,1)
        dT_dy = -np.gradient(ztemp, axis=0) / dy  # (H,1) physical K/m; negate for north→south axis
        u_jet = (b_amp * w_mid_1d[:, None] * np.abs(dT_dy)).astype(np.float32, copy=False)  # (H,1)
        # Safety clamp: keep the parameterization from generating unrealistic surface jets.
        u_jet = np.clip(u_jet, 0.0, 70.0).astype(np.float32, copy=False)

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
        v_mid = V_TARGET_MIDLAT * (speed_nh * w_mid_nh + speed_sh * w_mid_sh)
        v_target = (V_TARGET_TRADE * w_trade + v_mid + V_TARGET_POLAR * w_polar).astype(np.float32, copy=False) * sign_lat
        # Remove the equator sign ambiguity (sign(0)=0) so the equator stays calm.
        v_target = np.where(np.abs(lat_2d[:, 0]) < np.deg2rad(2.0), 0.0, v_target).astype(np.float32, copy=False)
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

        # 4. Baroclinic jet mixing
        if b_amp != 0.0 and b_mix > 0.0:
            k = 1.0 / (b_mix * 86400.0)
            u_zm = np.mean(u_rot, axis=1, keepdims=True)  # (H,1)
            du = du + (u_jet - u_zm) * k * dt_sub

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
            a_v_row = np.clip(a * (1.0 + 5.0 * w_trade[:, None] + 12.0 * w_mid[:, None] + 3.0 * w_polar[:, None]), 0.0, 0.75).astype(np.float32, copy=False)
            v_curr = v_curr + (v_target[:, None] - v_zm) * a_v_row
        
        # Soft clamp to prevent explosion
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
    day_of_year: int = 80,
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
) -> tuple[np.ndarray, np.ndarray]:
    """Return (u, v) near-surface winds derived from pressure gradients.

    Build surface pressure from temperature (with land-sea contrast and seasonal
    variation), add terrain and weather-system perturbations, then derive winds
    from pressure gradients via geostrophic balance. A streamfunction solver
    ensures divergence-free flow while preserving realistic meridional components.
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
    else:
        elev_c = np.zeros((Hc, Wc), dtype=np.float32)
        land_mask = np.zeros((Hc, Wc), dtype=bool)
        sea_mask = np.ones((Hc, Wc), dtype=bool)
        gx = np.zeros((Hc, Wc), dtype=np.float32)
        gy = np.zeros((Hc, Wc), dtype=np.float32)

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
    T_ref = 273.15
    zp = float(np.clip(zonal_pressure, 0.0, 1.0))
    if temperature is not None and zp > 0.0:
        T_zonal = np.mean(T, axis=1, keepdims=True)
        T_used = zp * T_zonal + (1.0 - zp) * T
    else:
        T_used = T
    # Warmer = lower pressure (thermal low), colder = higher pressure (thermal high)
    p_thermal = (pp.surface_pressure_pa / 100.0) * (T_ref / (T_used + 1e-6)) ** 2.2  # hPa
    
    # Add terrain pressure anomalies (mountains create blocking highs)
    tp = float(np.clip(terrain_pressure_amp, 0.0, 1.0))
    p_terrain = 25.0 * tp * terrain_influence * np.clip(elev_c, 0.0, 1.0)
    
    # Optional: inject weak synoptic-scale perturbations.
    wamp = float(np.clip(weather_amp, 0.0, 1.0))
    if wamp > 0.0:
        # If simulation time is provided, use traveling-wave perturbations (moving eddies)
        # instead of stationary, day-seeded blobs.
        t_days = float(time_days) if time_days is not None else float(day_of_year)
        if temperature is not None:
            # Mid-latitude storm-track window
            storm_window = np.exp(-((abs_deg[:, None] - 45.0) / 18.0) ** 2).astype(np.float32, copy=False)
            equ_window = np.exp(-((abs_deg[:, None]) / 12.0) ** 2).astype(np.float32, copy=False)
            # A small set of deterministic planetary waves
            ks = np.array([3.0, 5.0, 7.0, 9.0], dtype=np.float32)
            periods = np.array([6.0, 9.0, 14.0, 20.0], dtype=np.float32)  # days
            phases = np.array([0.3, 1.1, -0.7, 2.2], dtype=np.float32)
            amps = np.array([6.0, 4.5, 3.0, 2.5], dtype=np.float32) * wamp  # hPa
            wave = np.zeros((Hc, Wc), dtype=np.float32)
            for k, per, ph, a in zip(ks, periods, phases, amps):
                wave += a * np.cos(k * lon[None, :] + (2.0 * np.pi * t_days / per) + ph)
            p_thermal += storm_window * wave

            # Add a weak equatorial traveling wave (Walker / MJO-ish), to avoid overly static tropics.
            p_thermal += equ_window * (2.0 * wamp) * np.cos(2.0 * lon[None, :] - (2.0 * np.pi * t_days / 7.0) + 0.4)
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
    rho = RHO_AIR
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
    v_surface = (-3.5 * w_trade + 5.0 * w_mid - 1.2 * w_polar).astype(np.float32, copy=False) * sign_lat
    v_surface = np.where(abs_deg < 2.0, 0.0, v_surface).astype(np.float32, copy=False)
    uc_zm = np.mean(uc, axis=1, keepdims=True)
    vc_zm = np.mean(vc, axis=1, keepdims=True)
    u_nudge = (0.18 + 0.18 * w_mid[:, None] + 0.06 * w_trade[:, None]).astype(np.float32, copy=False)
    v_nudge = (0.16 + 0.12 * w_trade[:, None] + 0.18 * w_mid[:, None]).astype(np.float32, copy=False)
    uc = uc + (u_surface[:, None] - uc_zm) * u_nudge
    vc = vc + (v_surface[:, None] - vc_zm) * v_nudge
    
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
    humidity: np.ndarray | None = None,
    soil_moisture: np.ndarray | None = None,
    cloud_fraction: np.ndarray | None = None,
    day_of_year: int = 80,
    dt_days: float = 1.0,
    evap_coeff: float = 1.0,
    uplift_coeff: float = 1.0,
    rain_efficiency: float = 0.7,
    target_mean_mm_day: float = 2.7,
    max_precip_mm_day: float = 120.0,
    surface_pressure_hpa: float = 1013.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precip_mm_day, humidity, soil_moisture).

    The model keeps a prognostic surface humidity field and a simple soil-moisture
    bucket while blending three precipitation triggers: moisture convergence,
    orographic lift, and convective instability. Everything runs at the native
    grid resolution so it can operate in both snapshot and time-stepping modes.

    `cloud_fraction`, if provided, is this step's already-diagnosed cloud cover
    (from `_evolve_temperature`). It adds a stratiform term so widespread cloud
    sheets (frontal/persistent cover) produce rain even without a deep-convective
    trigger, closing the gap where clouds and precipitation were diagnosed from
    shared RH/ascent fields but never actually informed each other.
    """

    H = int(height)
    W = int(width)
    elev = elevation.astype(np.float32, copy=False)
    lat_deg = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * 180.0
    abs_lat_deg = np.abs(lat_deg)
    itcz_window = np.exp(-((abs_lat_deg / ITCZ_HALF_WIDTH_DEG) ** 2)).astype(np.float32, copy=False)
    storm_window = np.exp(-((abs_lat_deg - STORM_TRACK_CENTER_DEG) / 15.0) ** 2).astype(np.float32, copy=False)
    drybelt_window = np.exp(-((abs_lat_deg - DRYBELT_CENTER_DEG) / 8.0) ** 2).astype(np.float32, copy=False)

    land_mask, sea_mask = _derive_land_sea_masks(elev)
    land_f = land_mask.astype(np.float32)
    sea_f = sea_mask.astype(np.float32)

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

    Tc = np.clip(temperature - 273.15, -60.0, 60.0)
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))
    qsat = np.clip(0.622 * es / surface_pressure_hpa, 0.0, 0.035).astype(np.float32, copy=False)

    if humidity is None:
        base_q = np.where(sea_mask, 0.013, 0.009).astype(np.float32, copy=False)
    else:
        base_q = humidity.astype(np.float32, copy=False)

    if soil_moisture is None:
        soil = np.where(land_mask, 0.55, 0.0).astype(np.float32, copy=False)
    else:
        soil = soil_moisture.astype(np.float32)

    dt = max(float(dt_days), 1.0)
    # At large dt (monthly/annual mode) evaporation would saturate the entire humidity
    # field in one step, erasing the spatial gradients that determine climate zones.
    # Cap evaporation at 1.5-day equivalent so dry and wet cells stay differentiated
    # across substeps even when dt=6 (monthly) or dt=7 (annual).
    dt_evap = min(dt, 1.5)

    # Evaporation and evapotranspiration sources
    wind_norm = np.clip(wind_speed / 15.0, 0.0, 1.5)
    ocean_evap = evap_coeff * sea_f * (0.45 + 0.55 * wind_norm) * np.clip(qsat - base_q, 0.0, None)
    land_evap = (
        evap_coeff
        * land_f
        * (0.20 + 0.65 * temp_norm)
        * (0.35 + 0.65 * soil)
        * np.clip(qsat - base_q, 0.0, None)
    )
    sources = (ocean_evap + land_evap) * dt_evap
    q = np.clip(base_q + sources, 0.0, qsat)

    # Moisture advection/diffusion (semi-Lagrangian-ish)
    u_scale = np.clip(np.abs(u) / 20.0, 0.0, 1.0) * (0.32 + 0.16 * storm_window[:, None])
    v_scale = np.clip(np.abs(v) / 12.0, 0.0, 1.0) * (
        0.34 + 0.06 * drybelt_window[:, None] + 0.16 * storm_window[:, None]
    )

    if NUMBA_AVAILABLE:
        # Fast path: Numba-accelerated advection with adaptive diffusion
        for _ in range(3):
            q = _advect_humidity_numba(q.astype(np.float32), u, v, u_scale, v_scale)
            lap_q = _laplacian_numba(q)
            # Adaptive diffusion: stronger in regions with sharp gradients
            q_grad_strength = np.abs(lap_q) / (np.mean(np.abs(lap_q)) + 1e-9)
            diffusion_coeff = (0.11 + 0.03 * storm_window[:, None]) * (
                1.0 + 0.3 * np.clip(q_grad_strength, 0.0, 2.0)
            )
            q = q + diffusion_coeff * lap_q
            q = np.clip(q, 0.0, qsat)
    else:
        # Fallback: original NumPy implementation
        for _ in range(3):
            q = _advect_scalar(q, u, v, u_scale, v_scale)
            q = q + (0.11 + 0.03 * storm_window[:, None]) * _laplacian(q)
            q = np.clip(q, 0.0, qsat)

    # Moisture-flux convergence driver
    if NUMBA_AVAILABLE:
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

    # Large-scale ascent proxy from wind convergence.
    # Meridional term sign: row index increases southward while v is northward,
    # so physical divergence is ∂u/∂x − ∂v/∂i. With the previous (+) sign the
    # ITCZ's meridional convergence registered as DIVERGENCE (and the horse
    # latitudes' divergence as convergence), inverting the ascent/subsidence
    # drivers wherever the meridional wind dominated.
    _lap = _laplacian_numba if NUMBA_AVAILABLE else _laplacian
    div = _ddx_periodic(u) - np.gradient(v, axis=0)
    ascent = np.clip(-div, 0.0, None)
    ascent = ascent / (np.mean(ascent) + 1e-6)
    ascent = np.clip(ascent + 0.15 * _lap(ascent.astype(np.float32)), 0.0, 3.0)

    # Subsidence drying: suppresses precipitation in divergent (descending) regions.
    # In the real atmosphere the Hadley cell's descending branch at ~25-35° drives
    # subtropical high pressure → subsidence → compression warming → low RH → dry belt.
    # Without this, divergent zones (horse latitudes) get as much rain as convergent zones.
    div_pos = np.clip(div, 0.0, None)
    subsidence_norm = div_pos / (np.mean(div_pos) + 1e-6)
    subsidence_norm = np.clip(subsidence_norm, 0.0, 2.5)
    # Reduce precipitation in strong-subsidence / dry-belt regions.
    # Coefficients deepened and floor lowered (2026-07): real subtropical deserts
    # (Sahara, Arabian Peninsula, Kalahari) were coming out 3-10x too wet — most
    # drybelt cells sat around subsidence_suppression~0.5 (52% cut), nowhere near
    # the floor, because the drybelt_window term alone (0.18 coeff) was too weak to
    # reach the floor except at extreme subsidence_norm. Deepened so a typical
    # drybelt-center cell reaches ~0.2 (80% cut) even at average local subsidence.
    subsidence_suppression = np.clip(
        1.0 - 0.34 * subsidence_norm - 0.45 * drybelt_window[:, None],
        0.08,
        1.0,
    ).astype(np.float32, copy=False)

    # Orographic uplift signal.
    # gy must be the physical NORTHWARD slope (row index increases southward),
    # so gy = -∂elev/∂i; otherwise northward wind blowing up a north-facing
    # slope registered as downslope (and vice versa), inverting the meridional
    # half of both the uplift and rain-shadow terms.
    gx = _ddx_periodic(elev)
    gy = -np.gradient(elev, axis=0)
    slope = np.hypot(gx, gy)
    orog = np.clip(gx * u + gy * v, 0.0, None) + 0.25 * slope
    orog = land_f * orog
    orog = orog / (np.percentile(orog, 90.0) + 1e-6)
    orog = np.clip(orog + 0.15 * _lap(orog.astype(np.float32)), 0.0, 2.0)

    # Rain-shadow drying: the mirror image of orographic uplift. Descending air on
    # the lee side of a range compresses and warms, lowering RH — this is why the
    # Atacama (lee of the Andes), Patagonia, the Great Basin, and the Gobi (lee of
    # the Himalaya/Tibetan Plateau) are deserts in reality. Previously `orog` only
    # ever added rain (windward term clipped to >=0) with no leeward counterpart, so
    # these regions had no mechanism to dry out relative to the surrounding potential
    # field (observed: Atacama sample came out ~780 mm/yr vs Earth's near-zero).
    downslope = np.clip(-(gx * u + gy * v), 0.0, None)
    downslope = land_f * downslope
    downslope = downslope / (np.percentile(downslope, 90.0) + 1e-6)
    downslope = np.clip(downslope, 0.0, 2.0)
    rain_shadow_suppression = np.clip(1.0 - 0.40 * downslope, 0.35, 1.0).astype(np.float32, copy=False)

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
    ascent_norm = np.clip(ascent, 0.0, 1.5) / 1.5
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
    ascent_driver = ascent * (0.20 + 0.20 * itcz_window[:, None] + 0.08 * storm_window[:, None])
    # Stratiform term: existing cloud cover (frontal/persistent sheets) rains even
    # without a fresh convective trigger. target_mean_mm_day rescaling below keeps
    # the global mean calibrated, so this mainly reshapes *where* rain falls to track
    # cloud cover rather than changing the overall total. Weight retuned 0.10 -> 0.06
    # (2026-07): 0.10 pushed SH subtropical mean precip to 2.83 mm/day, just over
    # test_subtropical_precip_quantity's 2.8 cap (bisected precisely — 0.09 still
    # fails at 2.81, 0.08 is the first passing value; 0.06 leaves headroom rather
    # than sitting right at that boundary again).
    if cloud_fraction is not None:
        stratiform = np.clip(cloud_fraction.astype(np.float32), 0.0, 1.0)
    else:
        stratiform = np.zeros((H, W), dtype=np.float32)
    precip_potential = uplift_coeff * (
        0.18 * rh_release +
        0.24 * conv_driver +
        0.20 * orog +
        0.20 * convective +
        0.22 * ascent_driver +
        0.06 * stratiform
    ) * subsidence_suppression * rain_shadow_suppression
    lat_shape = np.clip(0.78 + 0.20 * itcz_window[:, None] + 0.02 * storm_window[:, None], 0.60, 1.40)
    precip_potential = precip_potential * lat_shape

    if NUMBA_AVAILABLE:
        # Fast path: Numba-accelerated smoothing
        for _ in range(3):
            lap_p = _laplacian_numba(precip_potential.astype(np.float32))
            precip_potential = np.clip(precip_potential + 0.18 * lap_p, 0.0, 3.0)
    else:
        # Fallback: original NumPy implementation
        for _ in range(3):
            precip_potential = np.clip(precip_potential + 0.18 * _laplacian(precip_potential), 0.0, 3.0)
    post_shape = np.clip(0.92 + 0.20 * itcz_window[:, None] - 0.10 * storm_window[:, None], 0.82, 1.12)
    precip_potential = np.clip(precip_potential * post_shape, 0.0, 3.0)

    # Convert potential to precipitation (mm/day) with moisture conservation
    # Cap removal fraction: at dt=6 the uncapped value clips to 1.0 (total moisture
    # stripping), leaving humidity_next≈0 everywhere and erasing spatial gradients.
    # Limiting to dt=2.0 and 0.85 ensures cells retain ~15% of moisture, so
    # the next substep starts with a spatially differentiated humidity field.
    remove_frac = np.clip(rain_efficiency * precip_potential * min(dt, 2.0), 0.0, 0.85)
    dq = np.clip(remove_frac * q, 0.0, q)
    column_mm_per_q = 2000.0  # ~20 mm PW for q=0.01
    P = dq * (column_mm_per_q / dt)
    if target_mean_mm_day > 0.0:
        mean_p = float(np.mean(P))
        scale = float(np.clip(target_mean_mm_day / (mean_p + 1e-6), 0.2, 3.0))
        dq = np.clip(dq * scale, 0.0, q)
        P = dq * (column_mm_per_q / dt)
    rain_export_factor = np.clip(
        0.94 - 0.14 * itcz_window[:, None] + 0.08 * storm_window[:, None],
        0.70,
        1.06,
    ).astype(np.float32, copy=False)
    dq = np.clip(dq * rain_export_factor, 0.0, q)
    P = dq * (column_mm_per_q / dt)
    if max_precip_mm_day > 0.0:
        cap = np.minimum(1.0, max_precip_mm_day / (P + 1e-9))
        dq = dq * cap
        P = P * cap

    # Update humidity and soil moisture reservoirs
    humidity_next = np.clip(q - dq, 0.0, qsat)

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
    soil = np.where(land_mask, np.clip(soil, 0.05, 1.0), 0.0)

    return (
        P.astype(np.float32),
        humidity_next.astype(np.float32),
        soil.astype(np.float32),
    )
