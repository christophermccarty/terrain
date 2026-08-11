"""JAX-native climate screening model for GPU-batched optimizer sweeps.

Phase 2-3 of the GPU-sweep-screening plan built a v1 of this model around
``thermal_diffusion`` / ``ice_albedo_strength`` / ``wind_damping`` (the
illustrative example in ``optimizer/sweep.py``'s docstring) using purely
diagnostic wind closures. Phase 4 validation against the real CPU model
found two things: those three parameters barely move the CPU model's score
at all at default resolution (score span 0.35 out of 100 across the whole
swept range -- it's buffered by ocean transport / humidity / the real
radiative-relaxation term, none of which v1 modeled), and independent of
that, v1's rankings were anti-correlated with the CPU model's (Spearman
-0.87) on the metrics it did model. See project memory
``gpu-sweep-screening-phase4-anticorrelated-2026-08-10`` for the full
writeup.

This v2 recalibrates around the parameters an actual production sweep
varies -- ``optimizer/configs/sweep_wind.json`` sweeps ``wind_damping``,
``wind_baroclinic_jet_amp``, ``wind_baroclinic_mix``, ``wind_pgf_temp_scale``,
and ``wind_cell_relax_days``, holding ``thermal_diffusion``/
``ice_albedo_strength`` fixed. Getting these five parameters to have real,
sign-correct leverage means replacing v1's instantaneous diagnostic wind
with a genuinely prognostic 1D-latitude port of the actual mechanism in
``atmosphere.py``'s ``evolve_wind``/``evolve_wind_aloft`` -- a relaxation
timescale or a mixing-strength parameter is meaningless against a field
that gets recomputed from scratch every step with no memory of its own.

What carries over from v1 (see that version's history for why, still true):
every default-weight ``ClimateMetrics`` field is a function of latitude
alone (a zonal mean), so this model still carries no longitude dimension --
state is a 1D latitude profile per batch element.

What's new in v2 -- a much closer, still-1D-collapsed port of the real
mechanism, not an invention
---------------------------------------------------------------------------
- **Surface wind is now prognostic** (``u``, ``v`` carried across steps,
  not recomputed from ``T`` every step), integrated via the same
  operator-split recipe as ``evolve_wind``: an exact Coriolis rotation
  matrix, then PGF + quadratic drag, then baroclinic-jet mixing, scaled by
  ``wind_damping`` (matching production's exact order -- damping scales the
  PGF+friction+baroclinic tendency, NOT the cell-relaxation step below),
  then direct (undamped) relaxation toward the idealized Hadley/Ferrel/
  Polar 3-cell target profile on a ``wind_cell_relax_days`` timescale. The
  target-profile shape (center latitudes, widths, target speeds) and the
  per-band relaxation-strength multipliers are copied verbatim from
  ``atmosphere.py``'s ``HADLEY_CELL_*``/``MID_LAT_JET_*``/``POLAR_CELL_*``/
  ``U_TARGET_*``/``V_TARGET_*`` constants and its ``a_u_row``/``a_v_row``
  formulas -- this part is a literal, not approximate, port: production's
  own cell-relaxation term already acts on the zonal mean
  (``np.mean(u_curr, axis=1)``), which is exactly what a 1D field already
  is.
- **A second prognostic "aloft" wind layer** (``u2``, ``v2``), a 1D port of
  ``evolve_wind_aloft``: same Coriolis rotation, a thermal PGF term with
  the OPPOSITE sign from the surface's (production's own comment explains
  why -- warm columns are geopotentially higher aloft, not lower, so this
  sign is what makes a real thermal-wind jet emerge instead of needing the
  cell-relaxation crutch), and weak Rayleigh damping instead of the
  surface's quadratic drag.
- **Baroclinic mixing** relaxes the surface wind toward this aloft layer on
  a ``wind_baroclinic_jet_amp / (wind_baroclinic_mix * 86400)`` rate,
  windowed around 45 deg latitude -- verbatim from ``evolve_wind``'s
  ``k = b_amp / (b_mix * 86400)`` / ``jet_window_mix`` terms.
- **PGF from temperature**: ``p_thermal = -pgf_temp_scale * (T-273.15)/30``
  for the surface layer, the aloft layer's sign-flipped equivalent scaled
  by a fixed ``upper_pgf_amp`` -- both verbatim from production.

What's dropped, and why it's a legitimate 1D-collapse consequence rather
than a new gap
---------------------------------------------------------------------------
- No zonal (``dp/dx``) pressure-gradient term: it needs longitude, which
  this model doesn't carry (same reasoning as v1 -- see the original
  module history). The physically essential jet-forming mechanism is the
  meridional (``dp/dy``) gradient converted to zonal flow by Coriolis
  rotation over many sub-steps (geostrophic adjustment / thermal-wind
  balance) -- that part needs no longitude and is fully modeled here.
- No terrain, ice-pressure, or synoptic-wave pressure terms, and no
  land-only Ferrel-centre shift (``ferrel_v_land_shift_deg``): all are
  land/longitude-dependent effects with nothing to attach to in a
  longitude-free, terrain-free model. The base (non-land) Ferrel centre
  (``ferrel_v_centre_deg``, EARTH default 44 deg, not 48) IS carried, since
  it applies regardless of land.
- No self-advection term (``u du/dx`` etc.) -- secondary relative to
  PGF/Coriolis/relaxation in production's own account, and this model
  already accepts approximation in exchange for staying cheap and batched.

Wind-driven temperature advection (added after the first v2 validation
pass): a 1D port of ``simulate.py``'s ``_advect_temperature_y_numba`` --
CFL-limited upwind advection of ``T`` by the prognostic ``v``, poles
excluded, verbatim from that kernel's own bounds/clip constants. The first
v2 validation (see project memory
``gpu-sweep-screening-phase4-anticorrelated-2026-08-10``) found the wind
mechanism itself matched production almost exactly (``wind_trade_mean``
Spearman 0.995) but the *overall* score barely correlated (0.203), because
roughly half of the CPU model's score variance for this parameter family
came from a small wind-driven temperature response that a T with zero wind
coupling structurally cannot reproduce.

Adding advection alone (above) raised overall-score correlation to 0.447
but left ``global_mean_t`` itself anti-correlated with the CPU model
(-0.37) -- a magnitude-only fix (damping the advection term) made this
WORSE (0.227), ruling out "too sensitive" as the explanation. The actual
cause: checking the CPU model's own internal relationship showed
wind_trade_mean vs. global_mean_t at Spearman -0.85 (more wind -> COOLER),
the opposite sign from pure heat-redistribution's expected warming effect
-- because production has a second, dominant wind-temperature coupling
this model didn't: wind-speed-driven evaporative cooling (a bulk
aerodynamic formula, ``simulate.py:4284-4300``). **Wind-speed-driven
evaporative cooling** (a 1D port of that formula, using a fixed assumed
humidity deficit in place of this model's absent humidity field -- see
``_EVAP_*`` constants) is added for exactly this reason.

Precipitation and the overturning/relaxation placeholder constants inherit
unchanged from v1 -- see ``_RELAX_RATE_PER_DAY`` / ``_MOISTURE_K`` /
``_CC_SLOPE_PER_K`` below, still new closures, still Phase-4-tunable, not
ports.

Orbital constants and the greenhouse/albedo constants are still held fixed
at ``PlanetParams.EARTH``'s values. Swept parameters for v2 are exactly
``optimizer/configs/sweep_wind.json``'s five: ``wind_damping``,
``wind_baroclinic_jet_amp``, ``wind_baroclinic_mix``, ``wind_pgf_temp_scale``,
``wind_cell_relax_days``. ``thermal_diffusion``/``ice_albedo_strength`` are
now fixed at that config's own ``fixed_params`` (0.04, 0.30) rather than
swept.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planet_params import EARTH  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


# ---------------------------------------------------------------------------
# Fixed grid + fixed physical constants (v2: not swept)
# ---------------------------------------------------------------------------

H = 128  # latitude rows only -- see module docstring for why no longitude dim
DT_DAYS = 1.0
N_SPINUP_DAYS = 730     # 2 years, matches optimizer/headless.py's spinup_years default
N_EVAL_DAYS = 365       # 1 year, matches optimizer/headless.py's eval_years default
START_DAY_OF_YEAR = 80.0  # matches headless.py's create_initial_state default

N_SURFACE_SUBSTEPS = 4   # atmosphere.py's evolve_wind uses 8; halved for a cheaper screening step
N_ALOFT_SUBSTEPS = 2     # evolve_wind_aloft uses 4; same halving

STEFAN_BOLTZMANN = 5.670374419e-8
GRAY_ATMOSPHERE_FACTOR = 0.5
EPSILON_EQUATOR = EARTH.epsilon_equator
EPSILON_POLE = EARTH.epsilon_pole
T_MIN_FLOOR_K = 215.0

ALBEDO_BASE_MIN = 0.25
ALBEDO_ICE_MAX = 0.80

ICE_FREEZE_TEMP_K = 269.9
ICE_MELT_TEMP_K = 271.4
ICE_FREEZE_RATE_PER_DAY = 0.045
ICE_MELT_RATE_PER_DAY = 0.19

RADIUS_M = EARTH.radius_m
GAS_CONSTANT_DRY = EARTH.gas_constant_dry  # unused directly in v2's wind (kept for reference)

# v2: fixed at optimizer/configs/sweep_wind.json's fixed_params, matching
# the real production sweep this model is now validated against.
THERMAL_DIFFUSION = 0.04
ICE_ALBEDO_STRENGTH = 0.30

# Wind constants fixed (not in sweep_wind.json's param_space), copied from
# atmosphere.py's evolve_wind/evolve_wind_aloft defaults.
DRAG_BASE = 2.0e-7
EQ_DRAG_BUMP = 2.0e-6
UPPER_PGF_AMP = 2.5
ALOFT_DAMPING_RATE_PER_DAY = 0.05
ALOFT_HADLEY_EDGE_DEG = 12.0
VMAX_CLIP = 50.0  # matches simulate_step's wind_vmax_clip default

# 3-cell target-profile constants, verbatim from atmosphere.py.
HADLEY_CELL_CENTER_DEG, HADLEY_CELL_WIDTH_DEG = 14.0, 8.5
MID_LAT_JET_CENTER_DEG, MID_LAT_JET_WIDTH_DEG = 48.0, 13.0
POLAR_CELL_CENTER_DEG, POLAR_CELL_WIDTH_DEG = 74.0, 9.0
U_TARGET_TRADE, U_TARGET_MIDLAT, U_TARGET_POLAR = -6.0, 11.5, -1.5
V_TARGET_TRADE, V_TARGET_MIDLAT, V_TARGET_POLAR = -6.4, 10.0, -1.0
FERREL_V_CENTRE_DEG = EARTH.ferrel_v_centre_deg  # 44.0 -- separate from the u-jet's 48 deg centre

# New v2 closure constants -- Phase 4 tuning targets, not physical constants.
_RELAX_RATE_PER_DAY = 0.08
_MOISTURE_K = 6.0          # mm/day per unit convergence*saturation-scale
_CC_SLOPE_PER_K = 0.07     # Clausius-Clapeyron-like fractional increase per Kelvin

# Scales the wind-advection CFL blend (see _step's T_advected). TRIED at 0.3
# (hypothesis: this model's T is over-sensitive to wind vs. production's
# ocean/land-buffered T_air) -- made overall-score Spearman WORSE (0.447 ->
# 0.227), not better, while barely moving global_mean_t's own correlation
# (-0.367 -> -0.332). That ruled out "just scale the magnitude down" and
# pointed at a SIGN mismatch instead -- which turned out to have a real
# cause, not a causal-ordering artifact (see _EVAP_* below): the real CPU
# model's own internal correlation between wind_trade_mean and
# global_mean_t is -0.85 (more wind -> COOLER, not warmer), because
# production's wind-speed-driven evaporative cooling (simulate.py's bulk
# aerodynamic formula, ~simulate.py:4284-4300) dominates over pure
# heat-redistribution. This model previously had that redistribution
# effect (advection) but not the cooling effect, which is exactly backwards
# for matching sign. Left at 1.0 (no damping) -- once the missing cooling
# term is added, this scale is no longer the lever to touch.
_ADVECTION_DAMPING_SCALE = 1.0

# Wind-speed-driven evaporative cooling: a 1D port of simulate.py's bulk
# aerodynamic evaporation formula (E = C_D * wind_speed * humidity_deficit,
# clipped, scaled by 2.5, subtracted from T -- verbatim structure and
# constants from simulate.py:4284-4300). Production computes the deficit
# from a real humidity field; this model has none (out of scope per Phase
# 1), so ``_ASSUMED_RH_DEFICIT_FRAC`` stands in for "how far below
# saturation the air typically sits" as a fixed fraction of qsat(T) --
# a new closure, Phase-4-tunable, not itself a production value. C_D is a
# single land/sea-blended drag coefficient (production distinguishes
# 1.5e-3 sea / 0.5e-3 land; this model has no land/sea distinction --
# see module docstring) roughly midway between the two.
_EVAP_DRAG_COEFF = 1.0e-3
_ASSUMED_RH_DEFICIT_FRAC = 0.5
_EVAP_COOLING_SCALE = 2.5   # verbatim from simulate.py's `E * 2.5`
_EVAP_E_MAX = 20.0          # verbatim from simulate.py's `np.clip(E, 0.0, 20.0)`
SURFACE_PRESSURE_HPA = EARTH.surface_pressure_pa / 100.0

# The only swept parameters v2 understands -- matches optimizer/configs/
# sweep_wind.json's param_space exactly. optimizer/sweep.py's GPU backend
# validates a ParamSpace against this before calling run_batch.
SUPPORTED_PARAMS = frozenset({
    "wind_damping", "wind_baroclinic_jet_amp", "wind_baroclinic_mix",
    "wind_pgf_temp_scale", "wind_cell_relax_days",
})

# Fallback values for any SUPPORTED_PARAMS name a caller doesn't sweep --
# matches optimizer/configs/earth_params.json's fixed_params (and
# simulate_step's own defaults) so an un-swept parameter behaves like
# "leave it at its normal value" rather than an arbitrary placeholder.
# optimizer/sweep.py::gpu_random_search uses this to fill in a partial
# ParamSpace, e.g. sweeping just wind_damping while leaving the other four
# at their defaults -- CPU-backend random_search allows exactly this
# (unswept simulate_step kwargs just keep their own defaults), so
# gpu_random_search needs the same behavior rather than requiring every
# SUPPORTED_PARAMS name to be present.
DEFAULT_PARAMS = {
    "wind_damping": 0.50,
    "wind_baroclinic_jet_amp": 1.0,
    "wind_baroclinic_mix": 2.0,
    "wind_pgf_temp_scale": 450.0,
    "wind_cell_relax_days": 3.0,
}
assert set(DEFAULT_PARAMS) == SUPPORTED_PARAMS

# optimizer.scoring.ClimateMetrics fields this model actually populates --
# everything else (cru_*, ncep_wind_*, circulation_score) is left at its
# dataclass default, which is exactly correct here: those all carry
# ReferenceClimate weight 0.0 by default (see module docstring / Phase 1).
METRICS_FIELDS = (
    "global_mean_t", "gradient_nh", "gradient_sh", "ice_frac_nh", "ice_frac_sh",
    "mean_precip", "wind_trade_mean", "wind_midlat_mean", "wind_itcz_conv",
    "seasonal_amplitude_nh",
)


# ---------------------------------------------------------------------------
# Static (non-batched) grid arrays, built once on the host
# ---------------------------------------------------------------------------

def _gauss_window(abs_lat_deg: np.ndarray, center_deg: float, width_deg: float) -> np.ndarray:
    return np.exp(-(((abs_lat_deg - center_deg) / width_deg) ** 2))


def _build_grid():
    i = np.arange(H)
    lat_deg = 90.0 - (i + 0.5) * 180.0 / H          # row 0 = north pole, matches headless._lat_rows
    lat_rad = np.deg2rad(lat_deg)
    dy_m = RADIUS_M * np.pi / H
    abs_lat_deg = np.abs(lat_deg)
    sign_lat = np.sign(lat_deg)

    weights = np.cos(lat_rad)
    weights = weights / weights.sum()

    albedo_base = ALBEDO_BASE_MIN + 0.04 * np.clip(1.0 - abs_lat_deg / 40.0, 0.0, None) ** 1.5
    epsilon_lat = EPSILON_POLE + (EPSILON_EQUATOR - EPSILON_POLE) * np.cos(np.deg2rad(abs_lat_deg))

    nh_mask = (lat_deg > 0).astype(np.float64)
    sh_mask = (lat_deg < 0).astype(np.float64)
    trade_mask = ((abs_lat_deg >= 5.0) & (abs_lat_deg < 20.0)).astype(np.float64)
    midlat_mask = ((abs_lat_deg >= 30.0) & (abs_lat_deg < 60.0)).astype(np.float64)
    nh_band_mask = ((lat_deg >= 40.0) & (lat_deg <= 60.0)).astype(np.float64)

    f = EARTH.coriolis_parameter(lat_rad)
    eq_window = _gauss_window(abs_lat_deg, 0.0, 12.0)
    jet_window_mix = _gauss_window(abs_lat_deg, 45.0, 12.0)

    w_trade = _gauss_window(abs_lat_deg, HADLEY_CELL_CENTER_DEG, HADLEY_CELL_WIDTH_DEG)
    w_mid = _gauss_window(abs_lat_deg, MID_LAT_JET_CENTER_DEG, MID_LAT_JET_WIDTH_DEG)
    w_polar = _gauss_window(abs_lat_deg, POLAR_CELL_CENTER_DEG, POLAR_CELL_WIDTH_DEG)
    w_mid_v = _gauss_window(abs_lat_deg, FERREL_V_CENTRE_DEG, MID_LAT_JET_WIDTH_DEG)

    u_target = np.clip(U_TARGET_TRADE * w_trade + U_TARGET_MIDLAT * w_mid + U_TARGET_POLAR * w_polar, -15.0, 15.0)
    v_target = (V_TARGET_TRADE * w_trade + V_TARGET_MIDLAT * w_mid_v + V_TARGET_POLAR * w_polar) * sign_lat
    v_target = np.where(abs_lat_deg < 2.0, 0.0, v_target)

    a_row_trade_u, a_row_mid_u, a_row_polar_u = 2.5, 9.0, 2.5
    a_row_trade_v, a_row_mid_v_, a_row_polar_v = 5.0, 12.0, 3.0

    polar_window = np.exp(-(((90.0 - abs_lat_deg) / 10.0) ** 2))  # aloft-layer polar damping bump

    return dict(
        lat_deg=lat_deg, lat_rad=lat_rad, dy_m=dy_m, weights=weights,
        albedo_base=albedo_base, epsilon_lat=epsilon_lat,
        nh_mask=nh_mask, sh_mask=sh_mask,
        trade_mask=trade_mask, midlat_mask=midlat_mask, nh_band_mask=nh_band_mask,
        f=f, eq_window=eq_window, jet_window_mix=jet_window_mix,
        w_trade=w_trade, w_mid=w_mid, w_polar=w_polar, w_mid_v=w_mid_v,
        u_target=u_target, v_target=v_target,
        a_row_trade_u=a_row_trade_u, a_row_mid_u=a_row_mid_u, a_row_polar_u=a_row_polar_u,
        a_row_trade_v=a_row_trade_v, a_row_mid_v=a_row_mid_v_, a_row_polar_v=a_row_polar_v,
        polar_window=polar_window,
    )


_GRID = _build_grid()


def _build_insolation_table(n_days: int) -> np.ndarray:
    """Daily-mean TOA insolation for each simulated day, reusing the real
    production function (PlanetParams.daily_mean_insolation) on the host.

    Shared across the whole batch -- orbital constants aren't swept, so
    this table is identical for every config and only needs building once.
    """
    lat_rad = _GRID["lat_rad"]
    table = np.zeros((n_days, H), dtype=np.float32)
    for step in range(n_days):
        day = (START_DAY_OF_YEAR + step) % EARTH.orbital_period_days
        table[step] = EARTH.daily_mean_insolation(lat_rad, day)
    return table


# ---------------------------------------------------------------------------
# Physics: single-day step (pure function of state, Q_today, params)
# ---------------------------------------------------------------------------

RHO = EARTH.reference_air_density


def _rotate(u, v, theta):
    cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)
    return cos_t * u + sin_t * v, -sin_t * u + cos_t * v


def _evolve_aloft(u2, v2, T, dt_days, grid):
    """1D port of atmosphere.evolve_wind_aloft -- see module docstring."""
    dt_sub = (dt_days * 86400.0) / N_ALOFT_SUBSTEPS
    f = grid["f"]
    p_anom = UPPER_PGF_AMP * ((T - 273.15) / 30.0)  # sign-flipped vs. surface -- see docstring
    dp_dy = -jnp.gradient(p_anom, grid["dy_m"])
    pgf_v = -(1.0 / RHO) * dp_dy
    k_damp = (
        ALOFT_DAMPING_RATE_PER_DAY / 86400.0
        + (1.5 / 86400.0) * grid["eq_window"]
        + (3.0 / 86400.0) * grid["polar_window"]
    )
    for _ in range(N_ALOFT_SUBSTEPS):
        theta = f * dt_sub
        u_rot, v_rot = _rotate(u2, v2, theta)
        du = (0.0 - k_damp * u_rot) * dt_sub
        dv = (pgf_v - k_damp * v_rot) * dt_sub
        u2, v2 = u_rot + du, v_rot + dv
        speed = jnp.hypot(u2, v2)
        scale = jnp.minimum(1.0, VMAX_CLIP / jnp.maximum(speed, 1e-6))
        u2, v2 = u2 * scale, v2 * scale
    return u2, v2


def _evolve_surface(u, v, u2, v2, T, dt_days, params, grid):
    """1D port of atmosphere.evolve_wind -- see module docstring."""
    dt_sub = (dt_days * 86400.0) / N_SURFACE_SUBSTEPS
    f = grid["f"]
    wind_damping = params["wind_damping"]
    b_amp = params["wind_baroclinic_jet_amp"]
    b_mix = params["wind_baroclinic_mix"]
    pgf_temp_scale = params["wind_pgf_temp_scale"]
    tau_cell = params["wind_cell_relax_days"]

    p_thermal = -pgf_temp_scale * ((T - 273.15) / 30.0)
    dp_dy = -jnp.gradient(p_thermal, grid["dy_m"])
    pgf_v = -(1.0 / RHO) * dp_dy

    k_mix = b_amp / (b_mix * 86400.0)
    k_cell = 1.0 / (tau_cell * 86400.0)

    for _ in range(N_SURFACE_SUBSTEPS):
        theta = f * dt_sub
        u_rot, v_rot = _rotate(u, v, theta)

        drag = DRAG_BASE + EQ_DRAG_BUMP * grid["eq_window"]
        speed_rot = jnp.hypot(u_rot, v_rot)
        friction_u = -drag * u_rot * speed_rot
        friction_v = -drag * v_rot * speed_rot
        du = friction_u * dt_sub          # pgf_u == 0 -- no zonal PGF without longitude
        dv = (pgf_v + friction_v) * dt_sub

        du = du + (u2 - u_rot) * grid["jet_window_mix"] * k_mix * dt_sub
        dv = dv + (v2 - v_rot) * grid["jet_window_mix"] * k_mix * dt_sub

        u_next = u_rot + du * wind_damping
        v_next = v_rot + dv * wind_damping

        a = jnp.clip(dt_sub * k_cell, 0.0, 1.0)
        a_u_row = jnp.clip(
            a * (1.0 + grid["a_row_trade_u"] * grid["w_trade"]
                 + grid["a_row_mid_u"] * grid["w_mid"]
                 + grid["a_row_polar_u"] * grid["w_polar"]),
            0.0, 1.0,
        )
        a_v_row = jnp.clip(
            a * (1.0 + grid["a_row_trade_v"] * grid["w_trade"]
                 + grid["a_row_mid_v"] * grid["w_mid_v"]
                 + grid["a_row_polar_v"] * grid["w_polar"]),
            0.0, 0.75,
        )
        u_next = u_next + (grid["u_target"] - u_next) * a_u_row
        v_next = v_next + (grid["v_target"] - v_next) * a_v_row

        speed = jnp.hypot(u_next, v_next)
        scale = jnp.minimum(1.0, VMAX_CLIP / jnp.maximum(speed, 1e-6))
        u, v = u_next * scale, v_next * scale

    return u, v


def _step(carry, x, params, grid):
    T, ice, u, v, u2, v2 = carry
    Q_today = x

    # --- Wind-driven meridional temperature advection: 1D port of
    # simulate.py's _advect_temperature_y_numba (CFL-limited upwind, poles
    # excluded -- matches that kernel's `prange(1, H-1)` bounds exactly).
    # Uses the wind carried in from the previous step (old wind advects this
    # step's air; the radiatively-updated T below then drives this step's
    # own wind update) -- a defensible causal ordering, not a claim of
    # matching simulate_step's exact same-step interleaving. This is what
    # closes the gap the first v2 validation found: without it, T has zero
    # sensitivity to wind, but roughly half of the CPU model's score
    # variance for this parameter family came from a small wind-driven T
    # response (see gpu-sweep-screening-phase4-anticorrelated memory).
    dt_sec = DT_DAYS * 86400.0
    v_cfl = _ADVECTION_DAMPING_SCALE * jnp.clip(jnp.abs(v) * dt_sec / grid["dy_m"], 0.0, 0.5)
    T_pad0 = jnp.pad(T, (1, 1), mode="edge")
    T_south, T_north = T_pad0[2:], T_pad0[:-2]  # T[i+1], T[i-1]
    T_y = jnp.where(v >= 0.0, T_south, T_north)
    adv_diff = jnp.clip(T_y - T, -12.0, 12.0)
    T_advected = T + v_cfl * adv_diff
    row_idx = jnp.arange(H)
    T_advected = jnp.where((row_idx == 0) | (row_idx == H - 1), T, T_advected)

    # --- Radiation ---
    albedo = grid["albedo_base"] + ICE_ALBEDO_STRENGTH * ice * (ALBEDO_ICE_MAX - grid["albedo_base"])
    F_net = jnp.maximum((1.0 - albedo) * Q_today, 1.0)
    gh_denom = jnp.maximum(1.0 - GRAY_ATMOSPHERE_FACTOR * grid["epsilon_lat"], 1e-6)
    T_eq = jnp.power(jnp.clip(F_net, 1e-9, None) / (STEFAN_BOLTZMANN * gh_denom), 0.25)
    T_eq = jnp.maximum(T_eq, T_MIN_FLOOR_K)

    # --- Diffusion (1D latitude Laplacian, edge-clamped) ---
    T_pad = jnp.pad(T_advected, (1, 1), mode="edge")
    lap = T_pad[:-2] + T_pad[2:] - 2.0 * T_advected
    T_diff = T_advected + THERMAL_DIFFUSION * 1.2 * jnp.clip(lap, -30.0, 30.0) * DT_DAYS

    # --- Relaxation toward radiative equilibrium (placeholder, Phase-4 tunable) ---
    T_relaxed = T_diff + _RELAX_RATE_PER_DAY * (T_eq - T_diff) * DT_DAYS

    # --- Wind-speed-driven evaporative cooling: 1D port of simulate.py's
    # bulk aerodynamic formula (see _EVAP_* constants above for what's a
    # verbatim production constant vs. a new closure standing in for the
    # missing humidity field). Uses the wind carried in from the previous
    # step, same sequencing choice as the advection term above.
    T_c = jnp.clip(T_relaxed - 273.15, -60.0, 60.0)
    es = 6.112 * jnp.exp(17.67 * T_c / (T_c + 243.5))
    qsat = jnp.clip(0.622 * es / SURFACE_PRESSURE_HPA, 1e-6, 0.035)
    deficit = qsat * _ASSUMED_RH_DEFICIT_FRAC
    wind_speed_old = jnp.hypot(u, v)
    E = _EVAP_DRAG_COEFF * wind_speed_old * deficit * 1000.0
    E = jnp.clip(E, 0.0, _EVAP_E_MAX)
    evap_cooling = E * _EVAP_COOLING_SCALE * DT_DAYS
    T_new = T_relaxed - evap_cooling

    # --- Ice cover: rate-limited freeze/melt hysteresis ---
    freezing = T_new < ICE_FREEZE_TEMP_K
    melting = T_new > ICE_MELT_TEMP_K
    ice_gain = jnp.where(freezing, ICE_FREEZE_RATE_PER_DAY * (1.0 - ice), 0.0) * DT_DAYS
    ice_loss = jnp.where(melting, ICE_MELT_RATE_PER_DAY * ice, 0.0) * DT_DAYS
    ice_new = jnp.clip(ice + ice_gain - ice_loss, 0.0, 1.0)

    # --- Wind: aloft layer first (held fixed across the surface's substeps
    # this day, matching production's once-per-day aloft update), then surface ---
    u2_new, v2_new = _evolve_aloft(u2, v2, T_new, DT_DAYS, grid)
    u_new, v_new = _evolve_surface(u, v, u2_new, v2_new, T_new, DT_DAYS, params, grid)

    wind_speed = jnp.hypot(u_new, v_new)

    # --- Precipitation: new closure, NOT a port -- see module docstring ---
    dv_dlat = jnp.gradient(v_new, grid["lat_deg"])
    convergence = jnp.maximum(-dv_dlat, 0.0)
    sat_scale = jnp.exp(_CC_SLOPE_PER_K * (T_new - 273.15))
    precip = _MOISTURE_K * convergence * sat_scale

    new_carry = (T_new, ice_new, u_new, v_new, u2_new, v2_new)
    snapshot = dict(T=T_new, ice=ice_new, wind_u=u_new, wind_v=v_new,
                     wind_speed=wind_speed, precip=precip)
    return new_carry, snapshot


# ---------------------------------------------------------------------------
# Per-config run: scan over the full spinup+eval horizon, score the eval tail
# ---------------------------------------------------------------------------

def _extract_metrics(snapshots, grid):
    """Reduce the eval-window snapshot stack to the ClimateMetrics fields
    optimizer/scoring.py's ReferenceClimate weights by default (weight > 0)."""
    weights = grid["weights"]
    T = snapshots["T"]              # (n_eval, H)
    ice = snapshots["ice"]
    speed = snapshots["wind_speed"]
    v = snapshots["wind_v"]
    precip = snapshots["precip"]

    global_mean_t = jnp.mean(jnp.sum(T * weights[None, :], axis=1))

    eq_idx = H // 2
    gradient_nh = jnp.mean(T[:, eq_idx] - T[:, 0])
    gradient_sh = jnp.mean(T[:, eq_idx] - T[:, -1])

    ice_frac_nh = jnp.mean(jnp.sum((ice > 0.1) * grid["nh_mask"][None, :], axis=1)
                            / jnp.maximum(jnp.sum(grid["nh_mask"]), 1.0))
    ice_frac_sh = jnp.mean(jnp.sum((ice > 0.1) * grid["sh_mask"][None, :], axis=1)
                            / jnp.maximum(jnp.sum(grid["sh_mask"]), 1.0))

    mean_precip = jnp.mean(jnp.sum(precip * weights[None, :], axis=1))

    def _band_mean(field_2d, mask):
        w = mask * weights
        return jnp.mean(jnp.sum(field_2d * w[None, :], axis=1) / jnp.maximum(jnp.sum(w), 1e-12))

    wind_trade_mean = _band_mean(speed, grid["trade_mask"])
    wind_midlat_mean = _band_mean(speed, grid["midlat_mask"])

    dv_dlat = jnp.gradient(v, grid["lat_deg"], axis=1)
    lo_i, hi_i = max(0, eq_idx - 2), min(H, eq_idx + 3)
    wind_itcz_conv = jnp.mean(-jnp.mean(dv_dlat[:, lo_i:hi_i], axis=1))

    midlat_t_nh_series = jnp.sum(T * grid["nh_band_mask"][None, :], axis=1) / jnp.maximum(
        jnp.sum(grid["nh_band_mask"]), 1.0
    )
    seasonal_amplitude_nh = jnp.max(midlat_t_nh_series) - jnp.min(midlat_t_nh_series)

    return dict(
        global_mean_t=global_mean_t, gradient_nh=gradient_nh, gradient_sh=gradient_sh,
        ice_frac_nh=ice_frac_nh, ice_frac_sh=ice_frac_sh, mean_precip=mean_precip,
        wind_trade_mean=wind_trade_mean, wind_midlat_mean=wind_midlat_mean,
        wind_itcz_conv=wind_itcz_conv, seasonal_amplitude_nh=seasonal_amplitude_nh,
    )


def _run_one(params, insolation_table, grid):
    T0 = jnp.full((H,), 280.0, dtype=jnp.float32)
    ice0 = jnp.where(jnp.abs(grid["lat_deg"]) > 60.0, 0.3, 0.0).astype(jnp.float32)
    zeros = jnp.zeros((H,), dtype=jnp.float32)
    carry0 = (T0, ice0, zeros, zeros, zeros, zeros)

    def body(carry, Q_today):
        return _step(carry, Q_today, params, grid)

    _, all_snapshots = jax.lax.scan(body, carry0, insolation_table)
    eval_snapshots = jax.tree_util.tree_map(lambda a: a[N_SPINUP_DAYS:], all_snapshots)
    return _extract_metrics(eval_snapshots, grid)


def run_batch(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Score a batch of configs. ``params`` values must all share shape (N,).

    Returns a dict of shape-(N,) arrays, one per ClimateMetrics field that
    optimizer/scoring.py's ReferenceClimate weights by default.
    """
    insolation_table = jnp.asarray(_build_insolation_table(N_SPINUP_DAYS + N_EVAL_DAYS))
    grid = {k: jnp.asarray(v) for k, v in _GRID.items()}
    params_j = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in params.items()}

    run_fn = jax.jit(jax.vmap(lambda p: _run_one(p, insolation_table, grid)))
    return jax.tree_util.tree_map(np.asarray, run_fn(params_j))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"jax devices: {jax.devices()}")
    print(f"Grid: H={H} (no longitude dim), {N_SPINUP_DAYS}+{N_EVAL_DAYS} days")

    n_configs = 8
    rng = np.random.default_rng(0)
    # Ranges match optimizer/configs/sweep_wind.json's param_space, EXCEPT
    # wind_baroclinic_jet_amp: that config's [200000, 3000000] is stale
    # (pre-1.5-layer-atmosphere semantics -- see simulate.py:1019-1026's
    # comment and this module's docstring); using a range around the
    # CURRENT default of 1.0 instead.
    params = dict(
        wind_damping=rng.uniform(0.25, 0.75, size=n_configs).astype(np.float32),
        wind_baroclinic_jet_amp=rng.uniform(0.3, 3.0, size=n_configs).astype(np.float32),
        wind_baroclinic_mix=rng.uniform(1.0, 4.0, size=n_configs).astype(np.float32),
        wind_pgf_temp_scale=rng.uniform(200.0, 800.0, size=n_configs).astype(np.float32),
        wind_cell_relax_days=rng.uniform(1.5, 6.0, size=n_configs).astype(np.float32),
    )

    t0 = time.perf_counter()
    metrics = run_batch(params)
    elapsed = time.perf_counter() - t0
    print(f"\nBatched run of {n_configs} configs: {elapsed:.3f}s ({elapsed / n_configs * 1000:.1f} ms/config)\n")

    print(f"{'config':>6} {'wdamp':>7} {'b_jamp':>10} {'b_mix':>6} {'pgf_ts':>7} {'relax_d':>8} | "
          f"{'gT':>7} {'grad_nh':>7} {'ice_nh':>7} {'precip':>7} {'w_trade':>7} {'w_mid':>7} {'itcz':>7} {'seas_nh':>7}")
    for i in range(n_configs):
        print(
            f"{i:>6} {params['wind_damping'][i]:>7.3f} {params['wind_baroclinic_jet_amp'][i]:>10.0f} "
            f"{params['wind_baroclinic_mix'][i]:>6.2f} {params['wind_pgf_temp_scale'][i]:>7.1f} "
            f"{params['wind_cell_relax_days'][i]:>8.2f} | "
            f"{metrics['global_mean_t'][i]:>7.1f} {metrics['gradient_nh'][i]:>7.1f} "
            f"{metrics['ice_frac_nh'][i]:>7.2f} {metrics['mean_precip'][i]:>7.2f} "
            f"{metrics['wind_trade_mean'][i]:>7.2f} {metrics['wind_midlat_mean'][i]:>7.2f} "
            f"{metrics['wind_itcz_conv'][i]:>7.3f} {metrics['seasonal_amplitude_nh'][i]:>7.1f}"
        )
    print("\nSanity targets (optimizer/scoring.py EARTH_REFERENCE ranges):")
    print("  global_mean_t in [286, 290] K, gradient_nh in [40, 65] K, ice_frac_nh in [0.02, 0.10],")
    print("  mean_precip in [2.2, 3.2] mm/day, wind_trade_mean in [4, 9] m/s, seasonal_amplitude_nh in [28, 55] K")
