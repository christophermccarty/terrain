"""Headless simulation runner — no GUI, no threads, no disk I/O.

Wraps simulate_step in a plain loop and extracts ClimateMetrics from
snapshots taken during an evaluation period.

Usage
-----
from optimizer.headless import run_simulation
from planet_params import EARTH

state, metrics = run_simulation(
    EARTH,
    spinup_years=2.0,
    eval_years=1.0,
    H=60, W=120,
)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planet_params import PlanetParams, EARTH
from simulate import PlanetState, simulate_step, create_initial_state, TimeScaleMode
from optimizer.scoring import ClimateMetrics


# ---------------------------------------------------------------------------
# Sub-step dispatch table (mirrors main.py SimulationThread.run)
# ---------------------------------------------------------------------------

_SUBSTEPS: dict[TimeScaleMode, list[tuple[float, bool]]] = {
    TimeScaleMode.DAILY:   [(1.0, True)],
    TimeScaleMode.WEEKLY:  [(1.0, True)] * 7,
    TimeScaleMode.MONTHLY: [(6.0, False)] * 5,
    TimeScaleMode.ANNUAL:  [(7.0, False)] * 52,
}

_DAYS_PER_CYCLE: dict[TimeScaleMode, float] = {
    TimeScaleMode.DAILY:   1.0,
    TimeScaleMode.WEEKLY:  7.0,
    TimeScaleMode.MONTHLY: 30.0,
    TimeScaleMode.ANNUAL:  364.0,
}


# ---------------------------------------------------------------------------
# Default synthetic elevation
# ---------------------------------------------------------------------------

def _make_default_elevation(H: int, W: int) -> np.ndarray:
    """Synthetic Earth-like elevation with ~30% land fraction (seed=42)."""
    rng = np.random.default_rng(42)
    lat = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
    lon = np.linspace(0, 2 * np.pi, W, endpoint=False)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    land = (
        0.40 * np.cos(1.0 * LON) * np.cos(0.8 * LAT)
        + 0.25 * np.cos(2.1 * LON + 0.5) * np.cos(1.5 * LAT)
        + 0.15 * np.sin(3.0 * LON + 1.2) * np.cos(2.0 * LAT)
        + 0.10 * rng.standard_normal((H, W))
    )
    land = (land - np.percentile(land, 70)) / (np.max(land) - np.min(land) + 1e-9)
    return np.clip(land, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _advance_one_cycle(
    state: PlanetState,
    mode: TimeScaleMode,
    *,
    planet_params: PlanetParams,
    **kwargs,
) -> PlanetState:
    """Advance state by one cycle (1 day / 1 week / 1 month / 1 year)."""
    for step_days, do_wind in _SUBSTEPS[mode]:
        state, _ = simulate_step(
            state,
            days=step_days,
            update_wind=do_wind,
            planet_params=planet_params,
            track_components=False,
            **kwargs,
        )
    return state


def _lat_rows(H: int) -> np.ndarray:
    return (0.5 - (np.arange(H) + 0.5) / H) * 180.0


def _zonal_mean_band(T: np.ndarray, lat_deg_1d: np.ndarray, lo: float, hi: float) -> float:
    mask = (lat_deg_1d >= lo) & (lat_deg_1d <= hi)
    if not np.any(mask):
        return 0.0
    lat_rad = np.deg2rad(lat_deg_1d[mask])
    weights = np.cos(lat_rad)
    weights /= weights.sum() + 1e-12
    return float(np.sum(np.mean(T[mask, :], axis=1) * weights))


def _extract_snapshot(
    state: PlanetState,
    H: int,
) -> dict[str, float]:
    """Extract scalar diagnostics from a state snapshot."""
    T = state.air_temperature if state.air_temperature is not None else state.temperature
    if T is None:
        return {}

    lat_deg = _lat_rows(H)
    lat_rad = np.deg2rad(lat_deg)
    weights = np.cos(lat_rad)
    weights /= weights.sum() + 1e-12

    T_zonal = np.mean(T, axis=1)
    global_mean_t = float(np.sum(T_zonal * weights))

    eq_idx = H // 2
    T_equator = float(T_zonal[eq_idx])
    gradient_nh = T_equator - float(T_zonal[0])
    gradient_sh = T_equator - float(T_zonal[-1])

    # Mid-latitude NH temperature (40-60°N) for seasonal tracking
    midlat_t_nh = _zonal_mean_band(T, lat_deg, 40.0, 60.0)

    # Ice fractions
    ice_frac_nh = ice_frac_sh = 0.0
    if state.ice_cover is not None:
        I_zonal = np.mean(state.ice_cover, axis=1)
        ICE_THRESH = 0.1
        nh_mask = lat_deg > 0
        sh_mask = lat_deg < 0
        ice_frac_nh = float(np.mean(I_zonal[nh_mask] > ICE_THRESH)) if np.any(nh_mask) else 0.0
        ice_frac_sh = float(np.mean(I_zonal[sh_mask] > ICE_THRESH)) if np.any(sh_mask) else 0.0

    # Precipitation
    mean_precip = 0.0
    if state.precipitation is not None:
        mean_precip = float(np.sum(np.mean(state.precipitation, axis=1) * weights))

    # Wind
    wind_trade_mean = wind_midlat_mean = wind_itcz_conv = 0.0
    if state.wind_u is not None and state.wind_v is not None:
        speed = np.sqrt(state.wind_u ** 2 + state.wind_v ** 2)
        abs_lat = np.abs(lat_deg)
        zspeed = np.mean(speed, axis=1)
        zv = np.mean(state.wind_v, axis=1)

        def _band(lo: float, hi: float) -> float:
            m = (abs_lat >= lo) & (abs_lat < hi)
            if not np.any(m):
                return 0.0
            w = weights[m] / (weights[m].sum() + 1e-12)
            return float(np.sum(zspeed[m] * w))

        wind_trade_mean = _band(5.0, 20.0)
        wind_midlat_mean = _band(30.0, 60.0)

        # ITCZ convergence: -d(v)/d(lat) near equator (positive = converging)
        dv_dlat = np.gradient(zv, lat_deg)
        eq_i = int(np.argmin(np.abs(lat_deg)))
        lo_i, hi_i = max(0, eq_i - 2), min(len(dv_dlat), eq_i + 3)
        wind_itcz_conv = float(-np.mean(dv_dlat[lo_i:hi_i]))

    return {
        "global_mean_t": global_mean_t,
        "gradient_nh": gradient_nh,
        "gradient_sh": gradient_sh,
        "ice_frac_nh": ice_frac_nh,
        "ice_frac_sh": ice_frac_sh,
        "mean_precip": mean_precip,
        "wind_trade_mean": wind_trade_mean,
        "wind_midlat_mean": wind_midlat_mean,
        "wind_itcz_conv": wind_itcz_conv,
        "midlat_t_nh": midlat_t_nh,   # for seasonal amplitude only
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_simulation(
    planet_params: PlanetParams = EARTH,
    spinup_years: float = 2.0,
    eval_years: float = 1.0,
    H: int = 60,
    W: int = 120,
    elevation: np.ndarray | None = None,
    spinup_time_scale: TimeScaleMode = TimeScaleMode.MONTHLY,
    eval_time_scale: TimeScaleMode = TimeScaleMode.DAILY,
    eval_snapshots: int = 12,
    **physics_kwargs,
) -> tuple[PlanetState, ClimateMetrics]:
    """Run a headless simulation and return the final state + climate metrics.

    The spinup advances the model from a cold start toward thermal equilibrium.
    The eval period collects metrics from ``eval_snapshots`` evenly spaced
    snapshots (default: monthly — 12 snapshots over 1 year).

    Parameters
    ----------
    planet_params:
        Physical constants for the simulated planet.
    spinup_years:
        How long to run before collecting metrics (default: 2 years).
    eval_years:
        Evaluation window over which metrics are averaged (default: 1 year).
    H, W:
        Grid dimensions (rows × cols). Default 60×120 ≈ 3°×3° resolution.
    elevation:
        Optional (H, W) float32 array in [0, 1]. Generates a synthetic
        Earth-like terrain if not provided.
    spinup_time_scale:
        Time-scale mode for the spinup phase (default: MONTHLY for speed).
    eval_time_scale:
        Time-scale mode for the evaluation phase (default: DAILY for accuracy).
    eval_snapshots:
        Number of snapshots to average over during evaluation (default: 12).
    **physics_kwargs:
        Extra keyword arguments forwarded to ``simulate_step`` (e.g.
        ``thermal_diffusion``, ``ice_albedo_strength``).

    Returns
    -------
    state:
        Final PlanetState after the evaluation period.
    metrics:
        Aggregated ClimateMetrics (averaged over evaluation snapshots).
    """
    if elevation is None:
        elevation = _make_default_elevation(H, W)

    state = create_initial_state(elevation, day_of_year=80.0)

    # --- Spinup ---
    cycle_days_spinup = _DAYS_PER_CYCLE[spinup_time_scale]
    n_spinup = max(1, round(spinup_years * planet_params.orbital_period_days / cycle_days_spinup))
    for _ in range(n_spinup):
        state = _advance_one_cycle(
            state, spinup_time_scale, planet_params=planet_params, **physics_kwargs
        )

    # --- Evaluation ---
    cycle_days_eval = _DAYS_PER_CYCLE[eval_time_scale]
    total_eval_days = eval_years * planet_params.orbital_period_days
    n_eval = max(1, round(total_eval_days / cycle_days_eval))
    sample_interval = max(1, n_eval // eval_snapshots)

    snapshots: list[dict[str, float]] = []
    for i in range(n_eval):
        state = _advance_one_cycle(
            state, eval_time_scale, planet_params=planet_params, **physics_kwargs
        )
        if (i % sample_interval == 0) or (i == n_eval - 1):
            snap = _extract_snapshot(state, H)
            if snap:
                snapshots.append(snap)

    # --- Aggregate metrics ---
    if not snapshots:
        return state, ClimateMetrics(has_nan=True)

    def _mean_snap(key: str) -> float:
        vals = [s[key] for s in snapshots if key in s]
        return float(np.mean(vals)) if vals else 0.0

    # Check for NaN/Inf in final state
    T_check = state.air_temperature if state.air_temperature is not None else state.temperature
    has_nan = bool(T_check is not None and np.any(np.isnan(T_check)))
    has_inf = bool(T_check is not None and np.any(np.isinf(T_check)))

    midlat_temps = [s["midlat_t_nh"] for s in snapshots if "midlat_t_nh" in s]
    seasonal_amplitude_nh = (
        float(max(midlat_temps) - min(midlat_temps)) if len(midlat_temps) >= 2 else 0.0
    )

    metrics = ClimateMetrics(
        global_mean_t=_mean_snap("global_mean_t"),
        gradient_nh=_mean_snap("gradient_nh"),
        gradient_sh=_mean_snap("gradient_sh"),
        ice_frac_nh=_mean_snap("ice_frac_nh"),
        ice_frac_sh=_mean_snap("ice_frac_sh"),
        mean_precip=_mean_snap("mean_precip"),
        wind_trade_mean=_mean_snap("wind_trade_mean"),
        wind_midlat_mean=_mean_snap("wind_midlat_mean"),
        wind_itcz_conv=_mean_snap("wind_itcz_conv"),
        seasonal_amplitude_nh=seasonal_amplitude_nh,
        has_nan=has_nan,
        has_inf=has_inf,
    )
    return state, metrics
