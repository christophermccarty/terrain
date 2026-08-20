"""Time simulation for planet conditions.

Advances atmospheric systems (temperature, wind, precipitation) forward in time
with configurable time scales. Default unit is one day.
"""

from __future__ import annotations

import logging
import warnings
import numpy as np
from simulation_state import PlanetState, TimeScaleMode
from simulation_runner import initialize_state, run_multiple_steps
from sim_grid import (
    _coarsen,
    _coarsen_elevation_cached,
    _coarsen_many,
    _pad_edge_inplace,
    clear_grid_caches,
)
from state_persistence import (
    STATE_SCHEMA_VERSION,
    _load_state_npz,
    _save_state_npz,
    auto_save,
    load_state,
    save_state,
)
from sesam_coupling import sesam_column_closure_step
from sesam_wind_coupling import sesam_wind_and_eke_step
from sesam_radiation_coupling import sesam_radiation_step

LOG = logging.getLogger("planetsim")


from atmosphere import (
    generate_wind_field, generate_precipitation,
    evolve_wind, evolve_wind_aloft, _upsample_bilinear_many,
    _update_jet_index, _update_jet_blocking, flux_divergence_spherical,
    _normalize_positive_driver,
)
from temperature import (
    temperature_kelvin_for_lat,
    elevation_to_alt_km,
    STEFAN_BOLTZMANN,
    equilibrium_temperature_k,
)
from atmospheric_radiation import (
    TwoLayerGreyOpticalClosure,
    grey_radiative_convective_equilibrium_temperatures,
    pressure_defined_temperature_profile,
    pressure_split_emissivities_from_optical_depth,
    resolved_midlevel_emission_temperature,
    two_layer_grey_radiation,
    two_layer_optical_depth_for_target_olr,
)
from ocean import calculate_ocean_heat_transport, update_sea_ice, compute_ekman_transport, compute_gyre_currents
from carbon_cycle import (
    carbon_cycle_step, co2_temperature_response, CO2_PREINDUSTRIAL,
    co2_radiative_forcing, vegetation_albedo,
)
from climate_averages import (
    update_climate_averages, compute_stable_biomes,
    update_monthly_statistics, classify_koppen, koppen_to_legacy_biome,
)
from masks import get_masks
from planet_params import PlanetParams, EARTH
from pressure_circulation import balanced_thermal_wind_u, close_upper_mass_flux
from pressure_circulation import (
    shared_pressure_coordinate_circulation,
    evolve_large_scale_heating_reservoir,
    mse_constrained_pressure_coordinate_circulation,
    three_branch_mse_constrained_pressure_coordinate_circulation,
    momentum_constrained_three_branch_mse_pressure_coordinate_circulation,
    evolve_three_level_zonal_momentum,
)
from balanced_dynamics import (
    balanced_pressure_wind,
    diabatic_overturning_speed,
    moist_static_energy_overturning_speed,
    pressure_level_geopotential,
    thermally_direct_overturning,
)

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

# Cache for diagnostic/relaxation wind to avoid recomputing every step.
_RELAX_CACHE = {"key": None, "u": None, "v": None}


# A single generate_precipitation() call can only "rain out" the moisture
# reservoir once, regardless of how many days `dt_days` spans. At DAILY dt=1
# that's fine (365 independent calls/year, each free to evaporate-then-rain).
# At MONTHLY dt~30 a single call applies one snapshot of wind/humidity as if
# constant for the whole month and rains it out once, so precipitation doesn't
# scale up the way a real month's worth of independent weather events would --
# while the soil-moisture drain term (evaporation * dt) scales linearly with
# dt with no such ceiling. That mismatch was driving continental-interior soil
# moisture to its floor within a few decades of MONTHLY-mode spinup (observed:
# Canadian-Prairies-latitude precip collapsing to ~12 mm/yr vs Earth's
# 350-450 mm/yr), even though the underlying replenish/drain calibration is
# sound at dt=1. Sub-stepping at one-day cadence lets humidity evaporate and
# rain out repeatedly, closing the gap without changing per-call physics.
#
# One-day cadence is now the calibrated default.  A 32x64, 0.5-orbit
# DAILY-vs-MONTHLY convergence check reduced mean-precipitation error from
# 1.554 to 0.148 mm/day. PlanetParams.precip_substep_days can still override
# this threshold (for example 8.0 reproduces the former coarse cadence).
_PRECIP_SUBSTEP_DAYS = 1.0

# SESAM stage P6b (docs/SESAM_GAP_ANALYSIS.md Sec7): the diabatic bridge in
# sesam_coupling.py is a 1-day relaxation *rate* (K/day); calling it once per
# multi-day outer step (as MONTHLY mode's `days=30` does) applies that rate
# for the whole span in one Euler step, overshooting massively -- the same
# outer-step-vs-per-call-physics-timescale mismatch `_PRECIP_SUBSTEP_DAYS`
# above already exists to prevent for precipitation. Mirrors that idiom.
_SESAM_COLUMN_CLOSURE_SUBSTEP_DAYS = 1.0

# Reference temperature for `PlanetParams.land_transport_deficit_k`'s gate. The
# freezing point rather than a fitted constant: poleward eddy heat flux into a
# continent is a winter phenomenon, and 273.15 K is the physically meaningful
# boundary rather than one more tunable. The gate's *width* is the parameter.
_LAND_TRANSPORT_DEFICIT_REF_K = 273.15


def _generate_precipitation_substepped(H, W, elev, *, temperature, wind_u, wind_v,
                                        wind_u_aloft, wind_v_aloft,
                                        wind_u_midlevel, wind_v_midlevel,
                                        humidity, soil_moisture, soil_moisture_deep,
                                        condensate, precipitating_hydrometeors,
                                        midlevel_temperature,
                                        midlevel_humidity,
                                        upperlevel_temperature,
                                        upperlevel_humidity,
                                        midlevel_radiative_flux_w_m2=None,
                                        upperlevel_radiative_flux_w_m2=None,
                                        column_lower_temperature=None,
                                        previous_precipitation_mm_day=None,
                                        previous_large_scale_heating_w_m2=None,
                                        lower_pressure_depth_pa=None,
                                        midlevel_pressure_depth_pa=None,
                                        upperlevel_pressure_depth_pa=None,
                                        lower_pressure_cloud_condensate=None,
                                        midlevel_pressure_cloud_condensate=None,
                                        upperlevel_pressure_cloud_condensate=None,
                                        lower_pressure_hydrometeors=None,
                                        midlevel_pressure_hydrometeors=None,
                                        upperlevel_pressure_hydrometeors=None,
                                        cloud_fraction,
                                        day_of_year, dt_days,
                                        surface_pressure_hpa=1013.25,
                                        planet_params=None,
                                        debug_fields=None):
    dt_days = float(dt_days)
    # The deepest pressure-column experiment is itself a complete adaptive
    # state transition.  Calling it once per legacy daily microphysics chunk
    # would silently impose that old prescribed cadence on its joint
    # momentum/MSE/CFL solve.  At this nested gate, use the host climate step
    # as the one runtime call; its interface CFL policy supplies every needed
    # physical substep internally.
    _joint_pressure_column_runtime_active = planet_params is not None and all((
        bool(planet_params.enable_prognostic_column_water),
        bool(planet_params.enable_stability_aware_condensation),
        bool(planet_params.enable_two_layer_convective_adjustment),
        bool(planet_params.enable_three_level_pressure_column),
        bool(planet_params.enable_closed_three_level_thermodynamics),
        bool(planet_params.enable_diabatic_interface_mass_flux),
        bool(planet_params.enable_shared_pressure_coordinate_circulation),
        bool(planet_params.enable_pressure_coordinate_moisture_closure),
        bool(planet_params.enable_prognostic_overturning_heat_reservoir),
        bool(planet_params.enable_pressure_coordinate_mse_transport),
        bool(planet_params.enable_mse_constrained_pressure_circulation),
        bool(planet_params.enable_three_branch_mse_pressure_circulation),
        bool(planet_params.enable_momentum_constrained_three_branch_mse_circulation),
        bool(planet_params.enable_prognostic_pressure_coordinate_momentum),
    ))
    _pp_substep = float(planet_params.precip_substep_days) if planet_params is not None else 0.0
    substep_days = _pp_substep if _pp_substep > 0.0 else _PRECIP_SUBSTEP_DAYS
    if dt_days <= substep_days or _joint_pressure_column_runtime_active:
        return generate_precipitation(
            H, W, elev, temperature=temperature, wind_u=wind_u, wind_v=wind_v,
            wind_u_aloft=wind_u_aloft, wind_v_aloft=wind_v_aloft,
            wind_u_midlevel=wind_u_midlevel, wind_v_midlevel=wind_v_midlevel,
            humidity=humidity, soil_moisture=soil_moisture,
            soil_moisture_deep=soil_moisture_deep,
            condensate=condensate,
            precipitating_hydrometeors=precipitating_hydrometeors,
            midlevel_temperature=midlevel_temperature,
            midlevel_humidity=midlevel_humidity,
            upperlevel_temperature=upperlevel_temperature,
            upperlevel_humidity=upperlevel_humidity,
            midlevel_radiative_flux_w_m2=midlevel_radiative_flux_w_m2,
            upperlevel_radiative_flux_w_m2=upperlevel_radiative_flux_w_m2,
            column_lower_temperature=column_lower_temperature,
            previous_precipitation_mm_day=previous_precipitation_mm_day,
            previous_large_scale_heating_w_m2=previous_large_scale_heating_w_m2,
            lower_pressure_depth_pa=lower_pressure_depth_pa,
            midlevel_pressure_depth_pa=midlevel_pressure_depth_pa,
            upperlevel_pressure_depth_pa=upperlevel_pressure_depth_pa,
            lower_pressure_cloud_condensate=lower_pressure_cloud_condensate,
            midlevel_pressure_cloud_condensate=midlevel_pressure_cloud_condensate,
            upperlevel_pressure_cloud_condensate=upperlevel_pressure_cloud_condensate,
            lower_pressure_hydrometeors=lower_pressure_hydrometeors,
            midlevel_pressure_hydrometeors=midlevel_pressure_hydrometeors,
            upperlevel_pressure_hydrometeors=upperlevel_pressure_hydrometeors,
            cloud_fraction=cloud_fraction, day_of_year=day_of_year, dt_days=dt_days,
            surface_pressure_hpa=surface_pressure_hpa, planet_params=planet_params,
            return_condensate=True, return_midlevel_temperature=True,
            return_midlevel_humidity=True,
            return_upperlevel_state=True,
            return_precipitating_hydrometeors=True,
            debug_fields=debug_fields,
        )
    n_sub = max(1, int(round(dt_days / substep_days)))
    sub_dt = dt_days / n_sub
    hum, soil, soil_deep, cond, hydro, mid, mid_q, upper_t, upper_q = (
        humidity, soil_moisture, soil_moisture_deep, condensate,
        precipitating_hydrometeors, midlevel_temperature, midlevel_humidity,
        upperlevel_temperature, upperlevel_humidity,
    )
    P_accum = None
    period_days = float(
        planet_params.orbital_period_days if planet_params is not None else 365.2422
    )
    sub_day = float(day_of_year) - dt_days
    # Every substep sees identical elevation/temperature/wind/pressure/dt and only
    # evolves humidity+soil, so generate_precipitation's wind/terrain-derived fields
    # (masks, qsat, divergence, ascent, orographic uplift, ...) are the same each
    # pass. Compute them once and reuse via this cache (see its docstring); the
    # result is identical to recomputing them every substep.
    _static_cache: dict = {}
    _debug_accumulators: dict[str, np.ndarray | float] = {}
    for _ in range(n_sub):
        sub_day = (sub_day + sub_dt) % period_days
        _sub_debug: dict | None = {} if debug_fields is not None else None
        P_i, hum, soil, soil_deep, cond, mid, mid_q, upper_t, upper_q, hydro = generate_precipitation(
            H, W, elev, temperature=temperature, wind_u=wind_u, wind_v=wind_v,
            wind_u_aloft=wind_u_aloft, wind_v_aloft=wind_v_aloft,
            wind_u_midlevel=wind_u_midlevel, wind_v_midlevel=wind_v_midlevel,
            humidity=hum, soil_moisture=soil, soil_moisture_deep=soil_deep,
            condensate=cond,
            precipitating_hydrometeors=hydro,
            midlevel_temperature=mid,
            midlevel_humidity=mid_q,
            upperlevel_temperature=upper_t,
            upperlevel_humidity=upper_q,
            midlevel_radiative_flux_w_m2=midlevel_radiative_flux_w_m2,
            upperlevel_radiative_flux_w_m2=upperlevel_radiative_flux_w_m2,
            column_lower_temperature=column_lower_temperature,
            previous_precipitation_mm_day=previous_precipitation_mm_day,
            previous_large_scale_heating_w_m2=previous_large_scale_heating_w_m2,
            cloud_fraction=cloud_fraction,
            day_of_year=sub_day, dt_days=sub_dt,
            surface_pressure_hpa=surface_pressure_hpa, planet_params=planet_params,
            _static_cache=_static_cache, return_condensate=True,
            return_midlevel_temperature=True,
            return_midlevel_humidity=True,
            return_upperlevel_state=True,
            return_precipitating_hydrometeors=True,
            debug_fields=_sub_debug,
        )
        if _sub_debug is not None:
            # A precipitation debug rate is meaningful only as a time average
            # across the daily microphysics calls that make up a long outer
            # timestep.  Keep this deliberately narrow rather than exposing
            # one arbitrary substep's state under a misleading monthly label.
            for _name, _value in _sub_debug.items():
                if _name.endswith("_precipitation_mm_day"):
                    _weighted = np.asarray(_value) * (sub_dt / dt_days)
                    _debug_accumulators[_name] = (
                        _weighted
                        if _name not in _debug_accumulators
                        else _debug_accumulators[_name] + _weighted
                    )
                elif _name in {"pressure_moisture_cloud_created_mm"}:
                    # This is a water *amount* formed during the individual
                    # microphysics call, not a rate.  The prognostic
                    # overturning reservoir needs the full outer-step latent
                    # release, so sum rather than time-average it.
                    _debug_accumulators[_name] = (
                        np.asarray(_value)
                        if _name not in _debug_accumulators
                        else _debug_accumulators[_name] + np.asarray(_value)
                    )
                elif _name in {
                    "midlevel_omega_pa_s", "upperlevel_omega_pa_s",
                    "closed_column_lower_temperature_k",
                    "joint_pressure_column_lower_wind_u",
                    "joint_pressure_column_lower_wind_v",
                    "joint_pressure_column_midlevel_wind_u",
                    "joint_pressure_column_midlevel_wind_v",
                    "joint_pressure_column_upperlevel_wind_u",
                    "joint_pressure_column_upperlevel_wind_v",
                }:
                    # These are wind-derived instantaneous diagnostics.  They
                    # are unchanged through the moisture-only substeps, so one
                    # copy is the meaningful state diagnostic for this step.
                    _debug_accumulators[_name] = np.asarray(_value)
            if _sub_debug.get("joint_pressure_column_runtime"):
                # This gate evolves winds and the lower thermodynamic layer
                # inside the one-call pressure-column transition.  Carry that
                # state into the next monthly precipitation substep instead of
                # restarting its momentum evolution from the outer-step wind.
                wind_u = _sub_debug["joint_pressure_column_lower_wind_u"]
                wind_v = _sub_debug["joint_pressure_column_lower_wind_v"]
                wind_u_midlevel = _sub_debug["joint_pressure_column_midlevel_wind_u"]
                wind_v_midlevel = _sub_debug["joint_pressure_column_midlevel_wind_v"]
                wind_u_aloft = _sub_debug["joint_pressure_column_upperlevel_wind_u"]
                wind_v_aloft = _sub_debug["joint_pressure_column_upperlevel_wind_v"]
                column_lower_temperature = _sub_debug["closed_column_lower_temperature_k"]
                # Cached wind/temperature-derived precipitation drivers no
                # longer apply after the joint state transition.
                _static_cache = {}
        P_accum = P_i.astype(np.float32) if P_accum is None else P_accum + P_i
    if debug_fields is not None:
        debug_fields.update(_debug_accumulators)
    return (P_accum / n_sub).astype(np.float32, copy=False), hum, soil, soil_deep, cond, mid, mid_q, upper_t, upper_q, hydro


def _evolve_wind_substepped(
    u, v, u2, v2, *,
    temperature, elevation, ice_cover,
    dt_days_total: float, substep_days: float, time_days_end: float,
    damping, pgf_temp_scale, pgf_terrain_scale, drag_base, drag_elev_scale,
    vmax_clip, baroclinic_jet_amp, baroclinic_mix, cell_relax_days,
    planet_params, jet_index_nh, jet_index_sh, jet_block_nh, jet_block_sh,
    upper_pgf_amp, upper_damping, upper_hadley_edge_deg,
):
    """Advance surface+aloft wind by `dt_days_total`, in equal inner chunks of
    ~`substep_days` (PlanetParams.wind_prognostic_substep_days).

    Mirrors `_generate_precipitation_substepped`'s outer-loop-splitting idiom
    above, applied to wind instead of precip: re-runs `evolve_wind`/
    `evolve_wind_aloft`'s own already-tuned per-call physics `n_sub` times
    rather than asking one call to integrate a multi-day step it wasn't
    tuned for (both functions' internal sub-step count is fixed regardless
    of `dt_days`, so a single big-`dt_days` call means each of *their*
    internal sub-steps is proportionally larger than the ~1-day steps this
    physics was calibrated at).

    `substep_days <= 0` (the default, `wind_prognostic_substep_days=0.0`) or
    `>= dt_days_total` collapses to `n_sub=1` -- a single call at
    `dt_days=dt_days_total`, bit-identical to the un-substepped call sites
    this replaces. This is what keeps DAILY/WEEKLY (which already call this
    once per 1-day step) and the MONTHLY/ANNUAL default (gate off) exactly
    unchanged.
    """
    dt_days_total = float(dt_days_total)
    if substep_days <= 0.0 or substep_days >= dt_days_total:
        n_sub = 1
    else:
        n_sub = max(1, int(round(dt_days_total / substep_days)))
    sub_dt = dt_days_total / n_sub
    t = float(time_days_end) - dt_days_total
    for _ in range(n_sub):
        t += sub_dt
        u2, v2 = evolve_wind_aloft(
            u2, v2,
            temperature=temperature,
            dt_days=sub_dt,
            pgf_temp_scale=pgf_temp_scale,
            upper_pgf_amp=upper_pgf_amp,
            damping_rate=upper_damping,
            vmax_clip=vmax_clip,
            planet_params=planet_params,
            hadley_edge_deg=upper_hadley_edge_deg,
        )
        u, v = evolve_wind(
            u, v,
            temperature=temperature,
            pressure=None,
            elevation=elevation,
            dt_days=sub_dt,
            damping=damping,
            pgf_temp_scale=pgf_temp_scale,
            pgf_terrain_scale=pgf_terrain_scale,
            drag_base=drag_base,
            drag_elev_scale=drag_elev_scale,
            vmax_clip=vmax_clip,
            baroclinic_jet_amp=baroclinic_jet_amp,
            baroclinic_mix=baroclinic_mix,
            cell_relax_days=cell_relax_days,
            time_days=t,
            planet_params=planet_params,
            ice_cover=ice_cover,
            jet_index_nh=jet_index_nh,
            jet_index_sh=jet_index_sh,
            jet_block_nh=jet_block_nh,
            jet_block_sh=jet_block_sh,
            u_aloft=u2,
            v_aloft=v2,
        )
    return u, v, u2, v2


def _evolve_middle_wind_substepped(
    u, v, *, temperature, dt_days_total: float, substep_days: float,
    pgf_temp_scale: float, upper_pgf_amp: float, damping_rate: float,
    vmax_clip: float, planet_params, hadley_edge_deg: float,
):
    """Advance the native three-level column's independent middle wind.

    The middle level uses the same free-tropospheric momentum kernel as the
    existing upper level, but its weaker thermal forcing and stronger damping
    place it between the boundary layer and upper-tropospheric jet.  Splitting
    long prognostic wind steps here keeps its integration cadence consistent
    with the other two resolved wind levels.
    """
    dt_days_total = float(dt_days_total)
    if substep_days <= 0.0 or substep_days >= dt_days_total:
        n_sub = 1
    else:
        n_sub = max(1, int(round(dt_days_total / substep_days)))
    sub_dt = dt_days_total / n_sub
    for _ in range(n_sub):
        u, v = evolve_wind_aloft(
            u, v,
            temperature=temperature,
            dt_days=sub_dt,
            pgf_temp_scale=pgf_temp_scale,
            upper_pgf_amp=upper_pgf_amp,
            damping_rate=damping_rate,
            vmax_clip=vmax_clip,
            planet_params=planet_params,
            hadley_edge_deg=hadley_edge_deg,
        )
    return u, v


def _evolve_upper_wind_substepped(
    u, v, *, temperature, dt_days_total: float, substep_days: float,
    pgf_temp_scale: float, upper_pgf_amp: float, damping_rate: float,
    vmax_clip: float, planet_params, hadley_edge_deg: float,
):
    """Advance the native three-level column's own independent upper wind.

    PRIOR_ART_IMPLEMENTATION_PLAN.md Section 16 found that the three-level
    experimental path's excess cross-equatorial transport traces back to the
    fact that it was building its upper-level circulation directly on top of
    ``state.wind_u_aloft``/``wind_v_aloft`` -- the same always-on "1.5-layer
    atmosphere" jet-stream kernel that ``wind_upper_pgf_amp``/
    ``wind_upper_damping`` are extensively calibrated against for real
    jet-latitude/speed skill, via the exact same ``evolve_wind_aloft``
    function and constants. Any attempt to tame that field's meridional
    magnitude for the experimental path therefore risked regressing the
    already-validated default jet stream.

    This wrapper gives the three-level path its own, genuinely independent
    upper-level wind state instead: same free-tropospheric momentum kernel as
    both the shared upper level and the existing independent middle level
    (mirrors ``_evolve_middle_wind_substepped`` exactly), but with its own
    ``three_level_upper_wind_pgf_fraction`` (a fraction of
    ``wind_upper_pgf_amp``) and ``three_level_upper_wind_damping`` (starts
    higher than the shared level's ``wind_upper_damping``, per the middle
    level's own precedent and Section 16's magnitude finding). The shared
    ``wind_u_aloft``/``wind_v_aloft`` field and its calibrated jet-stream
    behavior are never read or written by this function.
    """
    dt_days_total = float(dt_days_total)
    if substep_days <= 0.0 or substep_days >= dt_days_total:
        n_sub = 1
    else:
        n_sub = max(1, int(round(dt_days_total / substep_days)))
    sub_dt = dt_days_total / n_sub
    for _ in range(n_sub):
        u, v = evolve_wind_aloft(
            u, v,
            temperature=temperature,
            dt_days=sub_dt,
            pgf_temp_scale=pgf_temp_scale,
            upper_pgf_amp=upper_pgf_amp,
            damping_rate=damping_rate,
            vmax_clip=vmax_clip,
            planet_params=planet_params,
            hadley_edge_deg=hadley_edge_deg,
        )
    return u, v


def _soft_min_cap(x: np.ndarray, cap: np.ndarray, width: float) -> np.ndarray:
    """Smooth, strictly-monotonic replacement for ``np.minimum(x, cap)``.

    ``width <= 0`` returns the exact hard minimum (bit-identical), so this is a
    drop-in no-op at the default. For ``width > 0`` returns
    ``cap - width * log(1 + exp((cap - x) / width))``, evaluated via
    ``np.logaddexp`` so the large-overshoot branch cannot overflow. The result
    tracks ``x`` well below ``cap``, asymptotes to ``cap`` well above it, and is
    strictly increasing in ``x`` everywhere in between -- which is the property
    the hard clamp destroys, and the reason the clamp flattened seven
    consecutive months onto one value. See `land_cap_softness_k`.
    """
    if width <= 0.0:
        return np.minimum(x, cap)
    z = (cap - x) / width
    return (cap - width * np.logaddexp(0.0, z)).astype(np.float32, copy=False)


def _ocean_seasonal_fraction(lat_deg: np.ndarray, pp) -> np.ndarray:
    """Fraction of the radiative seasonal swing that reaches ocean SST.

    Multiplies ``(T_lat_ocean_lagged - T_lat_annual_mean)`` in the ocean base
    temperature calculation, i.e. how much of a lagged radiative-equilibrium
    excursion the ocean's thermal inertia lets through.

    Two modes, selected by ``pp.derive_ocean_seasonal_lag``:

    - **Legacy (default, False)**: a hand-tuned per-latitude polynomial,
      unchanged since the model's early days. Equator ~24%, mid-lat ~10%,
      polar ~5% at Earth obliquity.
    - **Derived (True)**: computed from an explicit latitude-dependent mixed
      layer depth (``pp.mixed_layer_depth_tropical_m``/``_polar_m`` -- the
      same field `_evolve_temperature`'s T_sst relaxation step uses) via the
      standard slab-ocean thermal-relaxation response
      ΔT/ΔT_rad = 1/sqrt(1 + (2πτ/P)²), τ = ρ·cp·h/λ. Ships behind a flag
      (default off) because it has not yet been checked against the
      real-terrain regression baseline -- see FEATURES.md item 6.

    Both modes keep the same high-obliquity polar boost term: a real,
    distinct effect (extreme axial tilt drives polar insolation swings large
    enough to punch through even a deep mixed layer's damping), not curve-
    fitting specific to the legacy formula.
    """
    obliq_ratio = float(pp.obliquity_deg) / 23.44
    obliq_factor = np.clip(obliq_ratio, 0.6, 2.0) ** 0.5
    polar_lat_boost = np.sin(np.deg2rad(lat_deg)) ** 2
    high_obliq_boost = max(obliq_ratio - 1.0, 0.0)
    seasonal_cap = float(min(0.45 * obliq_factor, 0.85))

    if not pp.derive_ocean_seasonal_lag:
        frac = (
            (0.05 + 0.20 * np.cos(np.deg2rad(lat_deg))) * obliq_factor
            + 0.60 * high_obliq_boost * polar_lat_boost
        )
        return np.clip(frac, 0.03, seasonal_cap).astype(np.float32, copy=False)

    mld_trop = float(pp.mixed_layer_depth_tropical_m)
    mld_polar = float(pp.mixed_layer_depth_polar_m)
    mld = mld_trop + (mld_polar - mld_trop) * (np.abs(lat_deg) / 90.0) ** 1.5
    _WATER_HEAT_CAPACITY_J_M3_K = 4.186e6  # rho * cp for seawater
    lam = max(float(pp.ocean_thermal_relaxation_coefficient), 1e-6)
    tau_seconds = _WATER_HEAT_CAPACITY_J_M3_K * mld / lam
    tau_over_period = tau_seconds / (float(pp.orbital_period_days) * 86400.0)
    frac = 1.0 / np.sqrt(1.0 + (2.0 * np.pi * tau_over_period) ** 2)
    frac = frac + 0.60 * high_obliq_boost * polar_lat_boost
    return np.clip(frac, 0.03, seasonal_cap).astype(np.float32, copy=False)


def _ocean_mixed_layer_depth(
    lat_2d: np.ndarray, day_of_year: float, pp
) -> np.ndarray:
    """Return the seasonally varying slab-ocean mixed-layer depth [m]."""
    abs_lat = np.abs(np.rad2deg(lat_2d))
    mld_trop = float(pp.mixed_layer_depth_tropical_m)
    mld_polar = float(pp.mixed_layer_depth_polar_m)
    mld = mld_trop + (mld_polar - mld_trop) * (abs_lat / 90.0) ** 1.5
    solstice = (172.0 / 365.2422) * float(pp.orbital_period_days)
    gamma = (
        2.0 * np.pi * (float(day_of_year) - solstice)
        / float(pp.orbital_period_days)
    )
    nh_summer = float(0.5 * (1.0 + np.cos(gamma)))
    sh_summer = float(0.5 * (1.0 - np.cos(gamma)))
    hemi_summer = np.where(lat_2d >= 0, nh_summer, sh_summer)
    polar_ramp = np.clip((abs_lat - 55.0) / 30.0, 0.0, 1.0)
    return (mld * (1.0 - 0.50 * polar_ramp * hemi_summer)).astype(
        np.float32, copy=False
    )


_MARITIME_CACHE: dict = {"key": None, "field": None}


def _maritime_proximity(
    land_mask: np.ndarray,
    e_folding_km: float,
    planet_radius_km: float,
    upwind_ratio: float = 1.0,
) -> np.ndarray:
    """``exp(-distance_to_nearest_ocean / e_folding_km)`` over land, 0 on ocean.

    ``upwind_ratio`` > 1 makes the field **anisotropic**: ocean lying to a cell's
    west reaches it with an e-folding length ``upwind_ratio`` times longer than
    ocean in any other direction.  At midlatitudes the flow is westerly in both
    hemispheres, so the ocean that actually moderates a winter continent is the
    one upwind of it, not the nearest one in any direction -- New York (Dfa) has
    open water 100 km to its east and 4000 km of continent to its west, while
    Lisbon (Csb) at the same latitude has the ocean upwind.  An isotropic field
    calls those two cells similar; this one does not.  ``1.0`` restores the
    isotropic field exactly.  See ``land_transport_upwind_ratio``.

    Computed by max-decay dilation outward from the sea mask rather
    than a true distance transform, so this stays pure numpy -- the project has
    no scipy dependency in its simulation path and this is not a good reason to
    add one. The two differ only where the shortest path to open water is
    strongly diagonal, which is well inside this mechanism's own calibration
    uncertainty.

    Resolution-invariance is handled the way `atmosphere.py`'s monsoon inland
    mask handles it (and for the same reason -- a fixed cell reach silently
    became a different physical distance on every grid): the per-step decay is
    derived from each axis's own physical spacing in km, and the longitudinal
    spacing carries its cos(latitude) convergence, so the field is a function of
    real distance rather than of cell count.
    """
    H, W = land_mask.shape
    upwind_ratio = max(float(upwind_ratio), 1e-6)
    key = (H, W, float(e_folding_km), float(planet_radius_km), upwind_ratio,
           int(np.count_nonzero(land_mask)), hash(land_mask.tobytes()))
    if _MARITIME_CACHE["key"] == key:
        return _MARITIME_CACHE["field"]

    lat_spacing_km = np.pi * planet_radius_km / H
    lat = (0.5 - (np.arange(H, dtype=np.float64) + 0.5) / H) * np.pi
    lon_spacing_km = np.maximum(
        2.0 * np.pi * planet_radius_km * np.cos(lat) / W, 1e-3
    )
    base_km = max(e_folding_km, 1e-6)
    upwind_km = base_km * upwind_ratio
    decay_lat = np.float32(np.exp(-lat_spacing_km / base_km))
    decay_lon = np.exp(-lon_spacing_km / base_km).astype(np.float32)[:, None]
    # `np.roll(field, +1, axis=1)` gives each cell the value of its *western*
    # neighbour, so that is the term carrying upwind (westerly) ocean influence.
    decay_lon_upwind = np.exp(
        -lon_spacing_km / upwind_km
    ).astype(np.float32)[:, None]

    sea = (~np.asarray(land_mask, dtype=bool)).astype(np.float32)

    # The field is the better of two separately-computed reaches, because they are
    # different physical claims and have very different numerics:
    #
    #  1. **Isotropic**, `e_folding_km` in every direction -- "is there water near
    #     this cell". Short-ranged, so the iterated max-decay dilation below
    #     converges in a few dozen passes. It genuinely needs the 2-D iteration:
    #     the shortest path to open water can turn corners around a coastline.
    #  2. **Upwind**, `e_folding_km * upwind_ratio` due west -- "how far is this
    #     cell from the ocean the westerlies are bringing air from". That is a
    #     path *along a latitude circle* by definition, so it needs no corner
    #     turns and is exactly computable as a 1-D max-plus prefix scan.
    #
    # Keeping them separate is what makes the continental-scale reach affordable.
    # Run as one coupled 2-D dilation it does not converge in any usable number of
    # passes -- 580 single-shift passes at 512x1024, silently truncated by the
    # loop's own cap, i.e. a field that changed with resolution for a purely
    # numerical reason.  It is also the more faithful reading of the mechanism:
    # a long *upwind* reach should not be able to propagate meridionally by
    # turning corners, which is exactly what the coupled version let it do.
    # Three e-foldings along the *finest* axis is where the field has decayed to
    # ~5%; beyond that the iteration cannot change any value materially.  This
    # budget is set from `base_km` alone -- the upwind reach is handled below and
    # no longer inflates it, which is what the 512 cap used to silently truncate.
    finest = min(lat_spacing_km, float(lon_spacing_km.max()))
    steps = int(np.clip(np.ceil(3.0 * base_km / max(finest, 1e-6)), 1, 512))
    field = sea
    for _ in range(steps):
        updated = np.maximum(field, np.roll(field, 1, axis=0) * decay_lat)
        updated = np.maximum(updated, np.roll(updated, -1, axis=0) * decay_lat)
        updated = np.maximum(updated, np.roll(updated, 1, axis=1) * decay_lon)
        updated = np.maximum(updated, np.roll(updated, -1, axis=1) * decay_lon)
        if np.array_equal(updated, field):
            break
        field = updated

    if upwind_ratio > 1.0:
        # Exact 1-D scan by doubling: shift west by 1, then 2, then 4 ...,
        # squaring the decay each time, so `max_k sea[i-k] * decay**k` is reached
        # in O(log W) passes.  `np.roll(f, +1, axis=1)` gives each cell its
        # *western* neighbour's value, which is the upwind direction.
        upwind = sea
        shift, step_decay = 1, decay_lon_upwind
        while shift < W:
            upwind = np.maximum(
                upwind, np.roll(upwind, shift, axis=1) * step_decay
            )
            step_decay = step_decay * step_decay
            shift *= 2
        field = np.maximum(field, upwind)
    field = (field * np.asarray(land_mask, dtype=np.float32)).astype(np.float32)
    field.flags.writeable = False
    _MARITIME_CACHE["key"] = key
    _MARITIME_CACHE["field"] = field
    return field


_MARITIME_COARSE_CACHE: dict = {"key": None, "field": None}


def _maritime_proximity_coarse(
    elevation: np.ndarray,
    Hc: int,
    Wc: int,
    block_size: int,
    e_folding_km: float,
    planet_radius_km: float,
    upwind_ratio: float = 1.0,
) -> np.ndarray:
    """``_maritime_proximity`` evaluated on the *fine* grid, then block-averaged.

    The land heat-transport bonus lives on the coarse temperature grid, but the
    coarse **land mask** is not a coarsened coastline -- ``_coarsen`` block-means
    the elevation and ``get_masks`` then calls any block containing a sliver of
    land "land".  On the bundled Earth DEM that inflates land from 34.1% of area
    to 48.3% at ``block_size=4`` (Earth is 29%), pushing every coastline outward
    by up to a full coarse cell and flooding enclosed seas.  A maritime-proximity
    field built on that mask has most of its coastal contrast erased before any
    physics sees it: measured against the Köppen reference's own 35-45N C/D land,
    the coarse-mask field separates the two populations by 0.37 sd against 0.84
    for the same field computed at 128x256.

    Computing at native resolution and averaging down keeps the coastlines and
    only gives up the resolution, which recovers all of it -- 0.37 -> 0.83 sd
    isotropic, and 1.02 with ``upwind_ratio=32``, against an uncoarsened ceiling
    of 1.00.  Averaging rather than subsampling is what
    makes it a coarse-cell *mean* proximity, so a cell straddling a coast lands
    between its coastal and interior values instead of taking either one.

    The average is taken over each block's **land** cells only.  A plain block
    mean would divide a coastal sliver's high proximity by the whole block and
    report the most maritime cells on the grid as the most continental ones --
    the exact inversion this field exists to avoid.
    """
    elevation = np.asarray(elevation)
    H, W = elevation.shape
    if H != Hc * block_size or W != Wc * block_size:
        # Non-integer coarsening: fall back to the coarse mask rather than
        # guessing a reshape.  `_coarsen` is the only supported path here.
        _, land_c = get_masks(_coarsen_elevation_cached(elevation, Hc, Wc, block_size),
                             use_cache=False)
        return _maritime_proximity(
            land_c, e_folding_km, planet_radius_km, upwind_ratio
        )

    # Cheap O(1) content fingerprint rather than a full sum: this runs every
    # step, and `masks.py` already measured a full `.sum()` at ~160 us per call
    # on the 512x1024 grid.  Same strided-subset scheme as `masks._elev_fingerprint`
    # and `sim_grid._coarsen_elevation_cached`, plus id() to catch in-place edits
    # of a persistent array.
    flat = elevation.ravel()
    stride = max(1, flat.size // 512)
    key = (id(elevation), H, W, Hc, Wc, int(block_size), float(e_folding_km),
           float(planet_radius_km), float(upwind_ratio),
           float(flat[0]), float(flat[-1]), float(flat[::stride].sum()))
    if _MARITIME_COARSE_CACHE["key"] == key:
        return _MARITIME_COARSE_CACHE["field"]

    _, land_fine = get_masks(elevation)
    fine = _maritime_proximity(
        land_fine, e_folding_km, planet_radius_km, upwind_ratio
    )
    blocks = (Hc, block_size, Wc, block_size)
    total = fine.reshape(blocks).sum(axis=(1, 3), dtype=np.float64)
    land_count = land_fine.reshape(blocks).sum(axis=(1, 3), dtype=np.float64)
    field = np.divide(total, land_count, out=np.zeros_like(total),
                      where=land_count > 0)
    field = np.ascontiguousarray(field, dtype=np.float32)
    field.flags.writeable = False
    _MARITIME_COARSE_CACHE["key"] = key
    _MARITIME_COARSE_CACHE["field"] = field
    return field


def _evap_cooling_fraction(
    season: np.ndarray,
    soil: np.ndarray,
    coeff: float,
    strength: float,
) -> np.ndarray:
    """Fraction of a land cell's above-threshold excess removed by evapotranspiration.

    Shared by both of `simulate_step`'s forcing paths (the outer seasonal target
    and `_temperature_bases_for_day`'s inner resample) so the invariant below is
    maintained in exactly one place.

    **The `[0, 1]` clip is a correctness guard, not a tuning choice.**  The
    product ``season * coeff * strength * soil`` exceeds 1 once
    ``coeff * strength > 1 / soil`` -- about ``evap_cooling_strength`` 1.18 at
    the shipped 0.85 coefficient and a mid-range soil moisture -- past which the
    term removes *more* than the whole excess and the map from pre- to
    post-cooling temperature turns around, so a hotter cell comes out colder than
    a cooler one.  A contraction toward a reference must be monotone in its input
    and must not overshoot the reference; unclipped it is neither.  Inert at the
    shipped defaults, whose product peaks at 0.85.

    Audit process note 19: the bound and the invariant next to it have to be
    checked together, and the check has to call this function rather than
    re-derive the arithmetic.
    """
    return np.clip(
        np.asarray(season, dtype=np.float32)
        * (float(coeff) * float(strength))
        * np.clip(np.asarray(soil, dtype=np.float32), 0.0, 1.0),
        0.0,
        1.0,
    )


def _maritime_transport_factor(
    maritime: np.ndarray,
    land_mask: np.ndarray,
    lat: np.ndarray,
    decay: float,
    winter_weight_1d: np.ndarray,
) -> np.ndarray:
    """Row-mean-preserving continentality multiplier for the land transport bonus.

    Returns a factor to multiply the summed heat-transport trapezoids by.  It is
    ``1 + decay * winter * anomaly / spread`` where ``anomaly`` is each land
    cell's maritime proximity relative to **its own row's land mean**, so the
    calibrated zonal-mean winter level is untouched and only the contrast within
    a row changes.  Ocean cells and land-free rows get exactly 1.0.

    Two details that a planted-violation check showed are load-bearing:

    - ``spread`` is the land-area-weighted standard deviation of the anomaly,
      taken **globally**.  Dividing by it puts ``decay`` in units of "fraction of
      the bonus per standard deviation of continentality", so it keeps its
      meaning when ``land_transport_maritime_km`` or
      ``land_transport_upwind_ratio`` change the field's spread -- without it the
      strength and shape knobs are badly degenerate (a longer upwind reach
      flattens the field, and the same physical effect then needs decay 12 where
      a short reach needed 2).  Global rather than per-row because a row whose
      land is uniformly maritime genuinely has no contrast, and per-row
      normalization would manufacture one out of rounding.
    - ``winter_weight_1d`` gates it to the winter half.  Maritime moderation is a
      winter phenomenon; applied year-round the same contrast warms maritime
      *summers*, which is the wrong sign -- the real effect of maritime exposure
      on a summer maximum is to lower it.  Measured, the ungated form costs
      0.26pp of warmest-month threshold accuracy at 128x256 and 0.45pp at
      256x512, diffusely across reference groups C/D/E, while the gate removes
      that cost and leaves every coldest-month gain intact.
    """
    land_f = np.asarray(land_mask, dtype=np.float32)
    per_row = land_f.sum(axis=1, keepdims=True)
    row_mean = np.divide(
        (maritime * land_f).sum(axis=1, keepdims=True),
        np.maximum(per_row, 1.0),
    )
    anomaly = (maritime - row_mean) * land_f
    # Area weighting so the spread is a property of the planet's geography and
    # not of the grid's polar cell crowding.
    weight = land_f * np.cos(lat)[:, None].astype(np.float32)
    weight_sum = float(weight.sum())
    spread = (
        float(np.sqrt((weight * anomaly * anomaly).sum() / weight_sum))
        if weight_sum > 0.0
        else 0.0
    )
    # A planet with no continentality contrast at all (an aquaplanet, or one
    # land cell per row) gets no mechanism rather than a division blow-up.
    if spread <= 1e-6:
        return np.ones_like(anomaly, dtype=np.float32)
    deviation = (
        float(decay)
        * np.asarray(winter_weight_1d, dtype=np.float32)[:, None]
        * anomaly
        / spread
    )
    # A cell can lose its whole bonus but not reverse it into a penalty, so the
    # deviation is bounded to [-1, 1].  A bare clip would silently break the
    # mean preservation above -- at the shipped strength the anomaly reaches
    # ~3 standard deviations, so clipping alone leaves rows up to 22% of a bonus
    # heavier than they started, which is a zonal-level change wearing a
    # contrast mechanism's clothes.  Re-centring after each clip restores it;
    # six passes converges to 5e-5 of a bonus on Earth's geography (one pass
    # leaves 0.05, three leave 0.003) and the final clip guarantees the bound
    # even if a pathological planet has not converged.  At the shipped strength
    # 16.4% of land cells sit at the bound, so this is a genuinely saturating
    # redistribution, not a small correction.
    land_bool = land_f > 0.0
    row_land = np.maximum(per_row, 1.0)
    for _ in range(6):
        deviation = np.clip(deviation, -1.0, 1.0) * land_f
        offset = deviation.sum(axis=1, keepdims=True) / row_land
        deviation = (deviation - offset) * land_f
    deviation = np.clip(deviation, -1.0, 1.0)
    return np.where(land_bool, 1.0 + deviation, 1.0).astype(np.float32, copy=False)


def _evolve_temperature_substepped(
    T_prev, T_base, elevation, Hc, Wc, block_size, H, W,
    day_of_year, days: float, *,
    substep_days: float = 0.0,
    total_days: float | None = None,
    T_air_prev=None,
    prev_cloud_cover=None,
    T_deep_ocean=None,
    prev_cloud_water=None,
    temperature_bases_for_day=None,
    **kwargs,
):
    """Advance temperature by `days`, in equal inner chunks of ~`substep_days`
    (PlanetParams.temperature_substep_days).

    Mirrors `_evolve_wind_substepped`/`_generate_precipitation_substepped`'s
    outer-loop-splitting idiom, applied to `_evolve_temperature`: re-runs its
    own already-tuned per-call physics `n_sub` times, advancing `day_of_year`
    and `total_days` by each inner `sub_dt` and threading the prognostic
    state (`T_sst`/`T_air`/cloud/deep-ocean/cloud-water) forward between
    calls, instead of asking one call to integrate a multi-day seasonal span
    from a single `day_of_year` snapshot.

    `substep_days <= 0` (the default, `temperature_substep_days=0.0`) or
    `>= days` collapses to `n_sub=1` -- a single call at `days=days`,
    bit-identical to the un-substepped call this replaces.
    """
    days = float(days)
    if substep_days <= 0.0 or substep_days >= days:
        return _evolve_temperature(
            T_prev, T_base, elevation, Hc, Wc, block_size, H, W,
            day_of_year=day_of_year, days=days,
            T_air_prev=T_air_prev, total_days=total_days,
            prev_cloud_cover=prev_cloud_cover, T_deep_ocean=T_deep_ocean,
            prev_cloud_water=prev_cloud_water,
            **kwargs,
        )
    n_sub = max(1, int(round(days / substep_days)))
    sub_dt = days / n_sub
    T_sst, T_air = T_prev, T_air_prev
    cloud_cover, T_deep, cloud_water = prev_cloud_cover, T_deep_ocean, prev_cloud_water
    land_deep = kwargs.get("land_deep_temperature")
    boundary_layer = kwargs.get("boundary_layer_temperature")
    boundary_interface = kwargs.get("boundary_layer_interface_temperature")
    radiative_middle = kwargs.get("radiative_midlevel_temperature")
    radiative_upper = kwargs.get("radiative_upperlevel_temperature")
    radiative_optical_depth = kwargs.get("radiative_optical_depth")
    day = float(day_of_year) - days
    t = (float(total_days) - days) if total_days is not None else None
    cloud_c = snow_c = components = None
    accumulated_grey_middle = None
    accumulated_grey_upper = None
    for _ in range(n_sub):
        day += sub_dt
        if t is not None:
            t += sub_dt
        T_base_i, T_base_land_i = (
            temperature_bases_for_day(day)
            if temperature_bases_for_day is not None
            else (T_base, kwargs.get("T_base_land"))
        )
        kwargs_i = dict(kwargs)
        kwargs_i["T_base_land"] = T_base_land_i
        kwargs_i["land_deep_temperature"] = land_deep
        kwargs_i["boundary_layer_temperature"] = boundary_layer
        kwargs_i["boundary_layer_interface_temperature"] = boundary_interface
        kwargs_i["radiative_midlevel_temperature"] = radiative_middle
        kwargs_i["radiative_upperlevel_temperature"] = radiative_upper
        kwargs_i["radiative_optical_depth"] = radiative_optical_depth
        T_sst, T_air, cloud_c, snow_c, components, T_deep, cloud_water = _evolve_temperature(
            T_sst, T_base_i, elevation, Hc, Wc, block_size, H, W,
            day_of_year=day, days=sub_dt,
            T_air_prev=T_air, total_days=t,
            prev_cloud_cover=cloud_cover, T_deep_ocean=T_deep,
            prev_cloud_water=cloud_water,
            **kwargs_i,
        )
        cloud_cover = cloud_c
        land_deep = components.get("_land_deep_temperature", land_deep)
        boundary_layer = components.get("_boundary_layer_temperature", boundary_layer)
        boundary_interface = components.get(
            "_boundary_layer_interface_temperature", boundary_interface
        )
        radiative_middle = components.get(
            "_radiative_midlevel_temperature", radiative_middle
        )
        radiative_upper = components.get(
            "_radiative_upperlevel_temperature", radiative_upper
        )
        radiative_optical_depth = components.get(
            "_radiative_optical_depth", radiative_optical_depth
        )
        if "_grey_midlevel_gain_w_m2" in components:
            contribution = components["_grey_midlevel_gain_w_m2"] * sub_dt
            accumulated_grey_middle = (
                contribution
                if accumulated_grey_middle is None
                else accumulated_grey_middle + contribution
            )
        if "_grey_upperlevel_gain_w_m2" in components:
            contribution = components["_grey_upperlevel_gain_w_m2"] * sub_dt
            accumulated_grey_upper = (
                contribution
                if accumulated_grey_upper is None
                else accumulated_grey_upper + contribution
            )
    if accumulated_grey_middle is not None:
        components["_grey_midlevel_gain_w_m2"] = accumulated_grey_middle / days
    if accumulated_grey_upper is not None:
        components["_grey_upperlevel_gain_w_m2"] = accumulated_grey_upper / days
    return T_sst, T_air, cloud_c, snow_c, components, T_deep, cloud_water


# Cache for ocean heat transport adjustment.
# Ocean dynamics are slow (decorrelation time ~30 days), so we recompute
# only once per ocean_update_interval_days and reuse the cached ΔT array.
_OCEAN_ADJ_CACHE: dict = {"adj": None, "last_update_day": -9999.0}

# Cache for the slow-changing half of the carbon cycle: wildfire, permafrost
# thaw, wetland CH4 emission, and the biome classification that feeds
# vegetation NPP. These are all genuinely slow processes (fire risk, thaw, and
# biome shifts don't meaningfully change day to day) that were nonetheless
# being recomputed as full-resolution array passes every single step —
# ~22% of profiled per-step cost at production resolution (512x1024). Applied
# in every TimeScaleMode, including DAILY, unlike the Phase 1 "DAILY = full
# per-day physics" convention used elsewhere (see PLAN.md) — this is the one
# deliberate exception, because these four processes don't actually have
# meaningful per-day dynamics to resolve even in DAILY mode.
# Mirrors _OCEAN_ADJ_CACHE's pattern: recompute+apply a lump update every
# CARBON_SLOW_UPDATE_INTERVAL_DAYS (with dt_days = the accumulated interval,
# not the per-step days), hold state constant in between. Ocean CO2 exchange
# and vegetation NPP/growth are NOT included here — they stay fully per-step
# via carbon_cycle_step (fast-responding, and cheap relative to the four
# processes above).
CARBON_SLOW_UPDATE_INTERVAL_DAYS = 4.0
_CARBON_SLOW_CACHE: dict = {"key": None, "last_update_day": -9999.0, "biome": None}


# ============================================================================
# Numba-accelerated compute kernels for temperature evolution
# These provide 5-20x speedup for advection and diffusion operations
# ============================================================================

@jit(nopython=True, parallel=True, cache=True)
def _advect_temperature_x_numba(T: np.ndarray, u: np.ndarray,
                                u_cfl: np.ndarray) -> np.ndarray:
    """Advect temperature in x-direction (periodic boundaries).

    u_cfl is the pre-computed CFL number |u|*dt/dx, clipped to [0, 0.5].
    Sign of u selects upwind direction.
    Returns updated temperature field.
    """
    H, W = T.shape
    T_out = T.copy()

    for i in prange(H):
        for j in range(W):
            # Periodic wrap in x
            j_east = (j + 1) % W
            j_west = (j - 1 + W) % W

            # Upwind advection: u>0 means westward flow in grid coords
            if u[i, j] >= 0:
                T_x = T[i, j_west]
            else:
                T_x = T[i, j_east]

            # Temperature difference with manual clipping (Numba compatible)
            T_diff = T_x - T[i, j]
            if T_diff > 12.0:
                T_diff = 12.0
            elif T_diff < -12.0:
                T_diff = -12.0

            # Apply CFL-correct advection
            T_out[i, j] = T[i, j] + u_cfl[i, j] * T_diff

    return T_out


@jit(nopython=True, parallel=True, cache=True)
def _advect_temperature_y_numba(T: np.ndarray, v: np.ndarray,
                                v_cfl: np.ndarray) -> np.ndarray:
    """Advect temperature in y-direction (edge boundaries).

    v_cfl is the pre-computed CFL number |v|*dt/dy, clipped to [0, 0.5].
    Sign of v selects upwind direction.
    Returns updated temperature field.
    """
    H, W = T.shape
    T_out = T.copy()

    for i in prange(1, H-1):  # Skip poles (edges)
        for j in range(W):
            # Upwind advection
            if v[i, j] >= 0:
                T_y = T[i + 1, j]  # Southward (positive v)
            else:
                T_y = T[i - 1, j]  # Northward (negative v)

            # Temperature difference with manual clipping (Numba compatible)
            T_diff = T_y - T[i, j]
            if T_diff > 12.0:
                T_diff = 12.0
            elif T_diff < -12.0:
                T_diff = -12.0

            # Apply CFL-correct advection
            T_out[i, j] = T[i, j] + v_cfl[i, j] * T_diff

    return T_out


@jit(nopython=True, parallel=True, cache=True)
def _apply_diffusion_numba(T: np.ndarray, thermal_diff: float, days: float,
                          iterations: int = 3) -> np.ndarray:
    """Apply Laplacian diffusion to temperature field.

    Returns updated temperature field after specified iterations.
    """
    H, W = T.shape
    T_curr = T.copy()

    for _ in range(iterations):
        T_new = T_curr.copy()

        for i in prange(1, H-1):  # Skip edges
            for j in range(W):
                # Periodic in x
                j_east = (j + 1) % W
                j_west = (j - 1 + W) % W

                # Laplacian (5-point stencil)
                c = T_curr[i, j]
                n = T_curr[i - 1, j]
                s = T_curr[i + 1, j]
                e = T_curr[i, j_east]
                w = T_curr[i, j_west]

                T_lap = n + s + e + w - 4.0 * c

                # Clamp laplacian to prevent extreme smoothing (manual clip for Numba)
                if T_lap > 30.0:
                    T_lap = 30.0
                elif T_lap < -30.0:
                    T_lap = -30.0

                # Apply diffusion
                T_new[i, j] = c + thermal_diff * 1.2 * T_lap * days

        T_curr = T_new

    return T_curr


def _land_ice_flow_step(thickness: np.ndarray, land_mask: np.ndarray, k: float, dt: float) -> np.ndarray:
    """One explicit substep of mass-conservative, thickness-weighted diffusion.

    Flux-form (not `D * laplacian(H)`) so each interior face flux appears
    with opposite sign in the neighbor's own update and cancels exactly --
    the scheme conserves total thickness except at grid boundaries. Diffusivity
    `D = k * thickness` is evaluated per-cell and averaged onto each face, so
    thick ice spreads faster than thin ice (see
    `PlanetParams.ice_flow_diffusivity`). Periodic in longitude, clamped
    (mirrored) in latitude -- the same stencil `masks.get_continentality`
    uses. `land_mask` pins ice at ocean cells to zero every substep, so flux
    that would cross a coastline is discarded from the land reservoir (a
    simplified calving proxy) rather than transported onto/into the ocean.
    """
    D = k * thickness

    e_ = np.roll(thickness, -1, axis=1)
    d_e = 0.5 * (D + np.roll(D, -1, axis=1))
    flux_e = d_e * (e_ - thickness)

    w_ = np.roll(thickness, 1, axis=1)
    d_w = 0.5 * (D + np.roll(D, 1, axis=1))
    flux_w = d_w * (w_ - thickness)

    n_ = np.concatenate([thickness[:1, :], thickness[:-1, :]], axis=0)
    d_n = 0.5 * (D + np.concatenate([D[:1, :], D[:-1, :]], axis=0))
    flux_n = d_n * (n_ - thickness)

    s_ = np.concatenate([thickness[1:, :], thickness[-1:, :]], axis=0)
    d_s = 0.5 * (D + np.concatenate([D[1:, :], D[-1:, :]], axis=0))
    flux_s = d_s * (s_ - thickness)

    updated = thickness + (flux_e + flux_w + flux_n + flux_s) * dt
    return np.where(land_mask, np.clip(updated, 0.0, None), 0.0).astype(np.float32, copy=False)


def simulate_step(
    state: PlanetState,
    days: float = 1.0,
    *,
    block_size: int = 3,
    wind_block_size: int | None = None,
    precip_block_size: int | None = None,
    update_wind: bool = True,
    # Small relaxation toward a diagnostic wind (includes tropical trades + mid-lat storm-track structure).
    # This helps recover easterly trades, westerly mid-lats, and calmer doldrums in a single-layer model.
    wind_relax: float = 0.0,
    wind_target_weather_amp: float = 0.35,
    wind_target_zonal_pressure: float = 0.85,
    wind_target_terrain_pressure_amp: float = 0.25,
    wind_target_terrain_flow_amp: float = 0.25,
    wind_pgf_temp_scale: float = 450.0,
    wind_pgf_terrain_scale: float = 900.0,
    wind_drag_base: float = 2.0e-7,
    wind_drag_elev_scale: float = 6.0e-7,
    wind_damping: float = 0.50,  # PGF scaling: 0.25 halved PGF causing 3-7× weak winds; 0.5 is better
    wind_vmax_clip: float = 50.0,  # Phase 4 fix: Realistic maximum wind speed (strong jet stream)
    # Baroclinic eddy / vertical momentum coupling to the upper-level (1.5-layer)
    # wind (evolve_wind_aloft). Dimensionless coupling-strength multiplier
    # (1.0 = nominal full-strength coupling on the wind_baroclinic_mix-day
    # relaxation timescale) -- NOTE: prior to the 1.5-layer atmosphere upgrade
    # this parameter scaled a magnitude-only `|dT/dy|` proxy directly (hence
    # the old ~1e6 default); its meaning changed with the real upper layer,
    # so the default was recalibrated rather than reused.
    wind_baroclinic_jet_amp: float = 1.0,
    wind_baroclinic_mix: float = 2.0,
    # None uses PlanetParams.wind_cell_relax_days (historical default: 3.0).
    # An explicit value remains supported for compatibility with existing callers.
    wind_cell_relax_days: float | None = None,
    ocean_transport_coeff: float | None = None,  # None → pp.ocean_transport_coeff
    # Deprecated compatibility arguments. They are not active model controls
    # and must not appear in new configs; non-default legacy values warn below.
    ocean_exchange_floor: float = 0.65,
    ocean_exchange_span: float = 0.35,
    # Ocean-atmosphere restoring rate [K/day]. Default matches the value that
    # was previously hardcoded inside ocean.py (0.03) — the old default here
    # (0.08) was silently ignored, so 0.03 preserves actual behavior now that
    # the parameter is wired through.
    ocean_exchange_coeff: float = 0.03,
    ocean_exchange_inertia: float = 0.35,
    epsilon_equator: float | None = None,
    epsilon_pole: float | None = None,
    polar_cooling_scale: float | None = None,  # None → pp.polar_cooling_scale
    ice_freeze_temp: float = 269.9,  # Require colder water before new sea ice forms
    ice_melt_temp: float = 271.4,    # Preserve some hysteresis without locking in subpolar ice
    ice_freeze_rate: float = 0.045,
    ice_melt_rate: float = 0.19,
    ice_albedo_strength: float | None = None,  # None → pp.ice_albedo_strength
    thermal_diffusion: float | None = None,    # None → pp.thermal_diffusivity
    latent_cooling_coeff: float = 0.015,
    enable_carbon_cycle: bool = True,
    apply_greenhouse_forcing: bool = True,
    co2_climate_feedback: float | None = None,  # None → pp.co2_climate_feedback
    debug_log: bool = False,
    track_components: bool = False,
    precipitation_debug: dict | None = None,
    planet_params: PlanetParams | None = None,
    time_scale: TimeScaleMode = TimeScaleMode.DAILY,
    feedback_flags: dict[str, bool] | None = None,
) -> tuple[PlanetState, dict]:
    deprecated_values = {
        "ocean_exchange_floor": (ocean_exchange_floor, 0.65),
        "ocean_exchange_span": (ocean_exchange_span, 0.35),
        "latent_cooling_coeff": (latent_cooling_coeff, 0.015),
    }
    supplied_deprecated = [
        name for name, (value, default) in deprecated_values.items()
        if value != default
    ]
    if supplied_deprecated:
        warnings.warn(
            "Deprecated no-op simulate_step argument(s): "
            + ", ".join(supplied_deprecated)
            + ". They are accepted only for legacy configuration compatibility "
            "and have no effect; remove them from new callers.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Supported flags (all default True when absent):
    #   'ice_albedo'        — sea ice effect on surface albedo + latent heat
    #   'snow_albedo'       — snow pack effect on surface albedo
    #   'amoc_acc'          — dynamic AMOC/ACC circulation weakening with ice
    #   'co2_greenhouse'    — CO2 temperature offset applied to T_base
    #   'vegetation_albedo' — biome/Köppen-based land albedo
    #   'ocean_transport'   — ocean heat transport ΔT calculation
    """Advance planet state forward by `days`.

    Updates temperature, wind, and precipitation based on new day_of_year.
    Interactions:
    - Temperature depends on insolation (day_of_year), wind advection, land-sea effects
    - Wind depends on temperature gradients
    - Precipitation depends on wind (advection/convergence), temperature (evaporation),
      and elevation (orographic uplift)
    
    Temperature now includes:
    - Longitudinal variation (coastal effects: land-sea contrast)
    - Meridional heat transport (winds carry heat poleward/equatorward)
    - Diurnal variation (approximate day/night cycle via longitude)

    Args:
        state: Current planet state
        days: Time step in days (default 1.0)
        block_size: Coarse resolution for simulation (larger = faster, less accurate)
        wind_block_size: Coarse resolution used for wind evolution. If None, uses `block_size`.
        precip_block_size: Precipitation resolution divisor (1 or 2). None uses
            half resolution only for grids with at least 256 latitude rows.
        update_wind: Whether to recompute wind field

    Returns:
        New state with updated day_of_year and atmospheric fields
    """
    base_pp = planet_params or state.planet_params or EARTH
    if base_pp.enable_milankovitch_cycles:
        from orbital_cycles import orbital_params_at_time

        pp = orbital_params_at_time(base_pp, state.total_days)
    else:
        pp = base_pp
    if (
        pp.enable_coupled_two_layer_grey_radiation
        and not pp.enable_pressure_defined_radiative_temperature_profile
    ):
        raise ValueError(
            "coupled grey radiation requires the pressure-defined temperature profile"
        )
    if (
        pp.enable_coupled_two_layer_grey_radiation
        and state.grey_optical_depth is None
    ):
        raise ValueError(
            "coupled grey radiation requires a diagnostic optical-depth spinup"
        )
    if pp.enable_coupled_two_layer_grey_radiation and not all((
        pp.enable_prognostic_column_water,
        pp.enable_stability_aware_condensation,
        pp.enable_two_layer_convective_adjustment,
        pp.enable_three_level_pressure_column,
        pp.enable_closed_three_level_thermodynamics,
        pp.enable_diabatic_interface_mass_flux,
    )):
        raise ValueError(
            "coupled grey radiation requires closed three-level thermodynamics"
        )
    if pp.enable_coupled_two_layer_grey_radiation and float(days) > 1.0:
        raise ValueError("coupled grey radiation requires daily-or-finer timesteps")
    _wind_cell_relax_days_arg = (
        float(pp.wind_cell_relax_days)
        if wind_cell_relax_days is None
        else float(wind_cell_relax_days)
    )
    if precip_block_size not in (None, 1, 2):
        raise ValueError("precip_block_size must be None, 1, or 2")
    # Retain pressure-vertical velocity when its three-level source exists,
    # without enabling the wider (and more expensive) precipitation debug path
    # for ordinary one/two-level simulations.
    _precipitation_diagnostics = precipitation_debug
    if _precipitation_diagnostics is None and bool(pp.enable_three_level_pressure_column):
        _precipitation_diagnostics = {}

    # Resolve parameters that default to None → pp.<field> (allows per-planet tuning
    # while still permitting explicit overrides from the optimizer or tests).
    if ocean_transport_coeff is None:
        ocean_transport_coeff = pp.ocean_transport_coeff
    if polar_cooling_scale is None:
        polar_cooling_scale = pp.polar_cooling_scale
    if ice_albedo_strength is None:
        ice_albedo_strength = pp.ice_albedo_strength
    if thermal_diffusion is None:
        thermal_diffusion = pp.thermal_diffusivity
    if co2_climate_feedback is None:
        co2_climate_feedback = pp.co2_climate_feedback

    # Detect whether elevation is a loaded real-world DEM (vs procedural noise).
    # Loaded DEMs have a large fraction of exactly-zero ocean cells; procedural
    # terrain uses continuous noise values that rarely land on exactly 0.0.
    _zeros_frac = float(np.sum(state.elevation == 0.0)) / max(1, state.elevation.size)
    _is_loaded_dem = bool(_zeros_frac > 0.05)

    # Dynamic AMOC / ACC feedback — scale the parameterised ocean heat bonus by
    # the actual polar sea-ice state.
    #
    # Physics basis: AMOC strength is driven by the density contrast between warm
    # salty Atlantic water and cold dense deep water sinking in the Nordic Seas.
    # When extensive sea ice covers 60-75°N it freshens the surface layer via
    # meltwater and suppresses thermohaline sinking, weakening AMOC on multi-year
    # timescales (Dansgaard-Oeschger stadials).  Conversely, a nearly ice-free
    # sub-polar gyre allows full AMOC and strong poleward heat delivery.
    #
    # Reference values (Northern Hemisphere, 60-75°N band):
    #   <5 % ice  → amoc_factor = 1.00  (full strength)
    #   35 % ice  → amoc_factor = 0.30  (reduced to 30 % of nominal)
    #
    # ACC (Antarctic Circumpolar Current) is less variable, so a more conservative
    # range is used (minimum factor 0.50, reference threshold 15 % ice at 60-75°S).
    if state.ice_cover is not None:
        _H_ice = state.ice_cover.shape[0]
        _lat_ice = (0.5 - (np.arange(_H_ice, dtype=np.float32) + 0.5) / _H_ice) * 180.0
        _nh_rows = (_lat_ice >= 60.0) & (_lat_ice <= 75.0)
        _sh_rows = (_lat_ice >= -75.0) & (_lat_ice <= -60.0)
        _nh_ice_frac = float(np.mean(state.ice_cover[_nh_rows])) if np.any(_nh_rows) else 0.0
        _sh_ice_frac = float(np.mean(state.ice_cover[_sh_rows])) if np.any(_sh_rows) else 0.0
        amoc_factor = float(np.clip(1.0 - (_nh_ice_frac - 0.05) / 0.30, 0.30, 1.0))
        acc_factor  = float(np.clip(1.0 - (_sh_ice_frac - 0.15) / 0.30, 0.50, 1.0))
    else:
        amoc_factor = 1.0
        acc_factor  = 1.0
        _lat_ice = np.array([], dtype=np.float32)

    # Feature 3: salinity modulates AMOC strength.
    # Fresher N.Atlantic surface water (lower density) reduces thermohaline sinking.
    if state.salinity is not None and pp.has_liquid_water_ocean and pp.salinity_amoc_scale > 0.0:
        _na_rows_sal = (_lat_ice >= 50.0) & (_lat_ice <= 75.0)
        if np.any(_na_rows_sal):
            _na_sal = float(np.mean(state.salinity[_na_rows_sal]))
            sal_anomaly = _na_sal - pp.salinity_reference_psu
            sal_amoc = float(np.clip(1.0 + 0.15 * sal_anomaly * pp.salinity_amoc_scale, 0.15, 1.5))
            amoc_factor = float(np.clip(amoc_factor * sal_amoc, 0.15, 1.0))

    # Feature 3b: temperature (density) modulates AMOC strength -- the other
    # half of the density-driven sinking, alongside the salinity term above.
    # Warmer N.Atlantic surface water is less dense and further suppresses
    # thermohaline sinking (compounds with freshening); colder water is denser
    # and strengthens sinking. Same North Atlantic sinking region (50-75N),
    # same phenomenological-gain convention as the salinity term -- not a full
    # seawater equation of state. Ocean-only mean (unlike the salinity term,
    # `state.temperature` carries real land values, so an unmasked band mean
    # would be dominated by cold Canadian/Siberian winter land temperatures
    # rather than the intended SST signal). See FEATURES.md item 5 /
    # PlanetParams.temperature_amoc_scale for why this defaults to 0.0.
    if state.temperature is not None and pp.has_liquid_water_ocean and pp.temperature_amoc_scale > 0.0:
        _na_rows_temp = (_lat_ice >= 50.0) & (_lat_ice <= 75.0)
        if np.any(_na_rows_temp):
            _sea_mask_na, _ = get_masks(state.elevation)
            _na_sea = _sea_mask_na[_na_rows_temp]
            if np.any(_na_sea):
                _na_temp = float(np.mean(state.temperature[_na_rows_temp][_na_sea]))
                temp_anomaly = _na_temp - pp.temperature_amoc_reference_k
                temp_amoc = float(np.clip(1.0 - 0.05 * temp_anomaly * pp.temperature_amoc_scale, 0.15, 1.5))
                amoc_factor = float(np.clip(amoc_factor * temp_amoc, 0.15, 1.0))

    # Apply feedback flags — freeze individual feedback loops at neutral state for testing.
    # Planet-level disables (has_liquid_water_ocean=False) are merged in as flag overrides.
    _fb = dict(feedback_flags) if feedback_flags else {}
    if not pp.has_liquid_water_ocean:
        _fb.setdefault('ocean_transport', False)
        _fb.setdefault('ice_albedo', False)
    if not _fb.get('amoc_acc', True):
        amoc_factor = 1.0
        acc_factor = 1.0

    eps_eq = float(pp.epsilon_equator if epsilon_equator is None else epsilon_equator)
    eps_pole = float(pp.epsilon_pole if epsilon_pole is None else epsilon_pole)
    new_day = (state.day_of_year + days) % pp.orbital_period_days
    new_total_days = float(state.total_days) + float(days)
    H, W = state.elevation.shape
    Hc, Wc = (max(1, (H + block_size - 1) // block_size),
              max(1, (W + block_size - 1) // block_size))
    wind_bs = max(1, int(block_size if wind_block_size is None else wind_block_size))
    Hcw, Wcw = (max(1, (H + wind_bs - 1) // wind_bs),
                max(1, (W + wind_bs - 1) // wind_bs))

    # ------------------------------------------------------
    # Climate Averaging and Köppen Classification
    # ------------------------------------------------------
    # Update 10-year rolling climate averages for general smoothing
    # Update monthly statistics for Köppen seasonality detection
    # Reclassify Köppen climate zones every 30 days

    # The experimental mixed layer is the land near-surface/CRU temperature;
    # air_temperature remains the transported free-atmosphere reservoir.  Feed
    # the mixed layer into the existing climate accumulators without changing
    # supported/default or ocean behavior.
    _climate_state = state
    if (
        bool(pp.enable_force_restore_land)
        and bool(pp.enable_force_restore_boundary_layer)
        and state.boundary_layer_temperature is not None
    ):
        _, _climate_land = get_masks(state.elevation)
        _near_surface_temperature = np.where(
            _climate_land,
            state.boundary_layer_temperature,
            state.temperature,
        ).astype(np.float32, copy=False)
        _climate_state = state._replace(temperature=_near_surface_temperature)

    # Update climate averages (exponential moving average)
    temp_avg, precip_avg, sample_days = update_climate_averages(
        _climate_state, days, orbital_period_days=pp.orbital_period_days,
        window_years=10.0,
    )

    # Update monthly statistics for Köppen classification
    monthly_temp, monthly_precip, monthly_sample_count = update_monthly_statistics(
        _climate_state, days, window_years=1.0,
        orbital_period_days=pp.orbital_period_days,
    )

    # Initialize biome/Köppen variables
    biome_new = state.biome_type
    koppen_new = state.koppen_type
    biome_last_update = state.biome_last_update_day

    # Reclassify Köppen climate zones every 30 days
    BIOME_UPDATE_INTERVAL = 30.0  # 30 days (was 3 years)
    days_since_biome_update = new_total_days - state.biome_last_update_day

    # Compute Köppen classification if monthly data is available
    if monthly_temp is not None and monthly_precip is not None:
        if days_since_biome_update >= BIOME_UPDATE_INTERVAL or state.koppen_type is None:
            # Time to update Köppen classification
            _, land_mask_for_biomes = get_masks(state.elevation)
            # Coarse (block-averaged) elevation, upsampled back to full resolution -
            # this is the elevation baseline the temperature physics already used for
            # its own orographic lapse-rate cooling (_evolve_temperature). Passing it
            # lets classify_koppen apply only the *additional* fine-grained delta
            # (peaks colder / valleys warmer than their block average) instead of
            # double-applying the full lapse rate on an already-cooled input.
            elev_c_for_biomes = _coarsen_elevation_cached(state.elevation, Hc, Wc, block_size)
            elev_baseline_for_biomes = (
                _upsample_bilinear_many({"elev": elev_c_for_biomes}, H, W, block_size)["elev"]
                if block_size > 1 else elev_c_for_biomes
            )
            koppen_new = classify_koppen(
                monthly_temp, monthly_precip, land_mask_for_biomes,
                elevation=state.elevation,
                elevation_baseline=elev_baseline_for_biomes,
                orbital_period_days=pp.orbital_period_days,
                max_elevation_km=pp.max_elevation_km,
                lapse_rate_k_per_km=pp.lapse_rate_k_per_km,
            )
            # Convert Köppen to legacy biome for backward compatibility
            biome_new = koppen_to_legacy_biome(koppen_new)
            biome_last_update = new_total_days
            if debug_log:
                LOG.info(f"[Köppen Update] Day {new_total_days:.0f} - Climate zones reclassified from monthly data")
        else:
            # Keep existing classification
            koppen_new = state.koppen_type
            biome_new = state.biome_type
            biome_last_update = state.biome_last_update_day
    else:
        # Monthly data not yet initialized - will be computed on next step
        koppen_new = state.koppen_type
        biome_new = state.biome_type
        biome_last_update = state.biome_last_update_day

    # ------------------------------------------------------
    # Antarctic Ice Sheet Initial Seeding (loaded real-earth DEM only)
    # ------------------------------------------------------
    # The Antarctic ice sheet is ~26.5 million km³ of land ice accumulated over
    # millions of years — a boundary condition that must be imposed explicitly for
    # the real Earth, not spun up from scratch in a few simulated years.
    #
    # Rule: land tiles south of -60° are seeded as EF (ice cap) ONCE on the very
    # first step when a loaded DEM is detected.  After that, the normal 30-day
    # Köppen reclassification takes over — if a tile warms enough (T_warmest ≥ 0°C)
    # it will naturally transition away from EF.
    #
    # All OTHER land tiles (non-Antarctic or procedural terrain) must earn EF
    # classification through the natural growth mechanism tracked by `ice_sheet_age`
    # (see below).
    _is_first_init = (state.ice_sheet_age is None)
    if _is_loaded_dem and _is_first_init:
        lat_1d_seed = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * 180.0
        _, land_mask_seed = get_masks(state.elevation)
        # Cells south of -60° that are land
        antarctic_ice_seed = (lat_1d_seed[:, None] < -60.0) & land_mask_seed  # (H, W)
        if koppen_new is None:
            # First step with no monthly data — initialize all land/ocean to ET/ocean
            koppen_new = np.where(land_mask_seed, 18, 0).astype(np.int32)  # 18=ET, 0=ocean
        else:
            koppen_new = koppen_new.copy()
        koppen_new[antarctic_ice_seed] = 19  # KOPPEN_EF (ice cap)
        biome_new = koppen_to_legacy_biome(koppen_new)
        biome_last_update = new_total_days

    # ------------------------------------------------------
    # Ice Sheet Age Tracking
    # ------------------------------------------------------
    # Ice sheets form over centuries/millennia from sustained snow accumulation in
    # areas where ablation < accumulation. In this model the proxy for "mature ice
    # sheet" is: a land cell must be continuously classified as EF (T_warmest < 0°C
    # from monthly averages) for ICE_SHEET_THRESHOLD_DAYS before it receives the
    # high ice-sheet albedo (0.80).  Until that threshold is reached the cell is
    # physically treated as tundra (albedo 0.25) even though Köppen labels it EF.
    #
    # Seeded Antarctic cells start with age = threshold (already mature at t=0).
    # All other cells start at age = 0 and must grow naturally.
    ICE_SHEET_THRESHOLD_DAYS = 3.0 * pp.orbital_period_days  # 3 years of sustained EF conditions

    if _is_first_init:
        ice_sheet_age_new = np.zeros((H, W), dtype=np.float32)
        if _is_loaded_dem:
            # Seeded Antarctic cells are already at threshold — full albedo from step 1
            lat_1d_age = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * 180.0
            _, land_mask_age = get_masks(state.elevation)
            antarctic_mask_age = (lat_1d_age[:, None] < -60.0) & land_mask_age
            ice_sheet_age_new[antarctic_mask_age] = ICE_SHEET_THRESHOLD_DAYS
    else:
        ice_sheet_age_new = state.ice_sheet_age.copy().astype(np.float32, copy=False)

    # Update age each step: EF-classified land cells accumulate toward threshold;
    # non-EF land cells lose age at half the gain rate (hysteresis — ice sheets
    # are slow to melt even after conditions warm slightly).
    if koppen_new is not None:
        _, _land_mask_upd = get_masks(state.elevation)
        _ef_land = (koppen_new == 19) & _land_mask_upd
        ice_sheet_age_new = np.where(
            _ef_land,
            np.minimum(ice_sheet_age_new + float(days), ICE_SHEET_THRESHOLD_DAYS),
            np.maximum(ice_sheet_age_new - float(days) * 0.5, 0.0),
        ).astype(np.float32, copy=False)

    # ------------------------------------------------------
    # CO2/CH4 greenhouse forcing is independent of reservoir evolution.
    # `enable_carbon_cycle=False` means "hold concentrations fixed"; historically
    # it also removed radiative forcing, making fixed-CO2 experiments inert.
    # ------------------------------------------------------
    # CRITICAL FIX: CO2 forcing must be applied to BASE TEMPERATURE before temperature evolution,
    # not added to final temperature afterward (which would cause runaway warming).
    # The forcing represents the equilibrium temperature offset that the simulation should relax toward.
    co2_temp_offset = 0.0
    _co2_ref = pp.co2_baseline_ppm if pp.co2_baseline_ppm > 1.0 else CO2_PREINDUSTRIAL
    if apply_greenhouse_forcing and _fb.get('co2_greenhouse', True):
        co2_forcing = co2_radiative_forcing(state.co2_atmosphere, _co2_ref)
        co2_temp_offset = co2_temperature_response(co2_forcing, co2_climate_feedback)
        # Feature 4: CH4 radiative forcing added to equilibrium temperature offset
        if pp.ch4_baseline_ppb > 0.0:
            from carbon_cycle import ch4_radiative_forcing as _ch4_rf
            _ch4_forcing = _ch4_rf(state.ch4_atmosphere, pp.ch4_baseline_ppb)
            co2_temp_offset += co2_climate_feedback * _ch4_forcing
        if debug_log:
            LOG.info(f"CO2={state.co2_atmosphere:.1f} ppm, forcing={co2_forcing:.2f} W/m², T_offset={co2_temp_offset:.2f}K")

    # Get base insolation temperature (latitude-dependent) + CO2 offset
    # Ocean: seasonal lag of ~50 days (1.5 months) due to high heat capacity
    # Land: immediate response to current insolation
    lat = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi

    # Calculate temperature for current day (land response)
    T_lat_land = temperature_kelvin_for_lat(
        lat,
        day_of_year=new_day,
        polar_cooling_scale=polar_cooling_scale,
        planet_params=pp,
    )
    # Annual-mean radiative baseline, hoisted here from the ocean block below
    # (which still consumes it) so the land seasonal damping can share it. It
    # depends only on `lat`, `polar_cooling_scale` and `pp`, so the move is
    # side-effect free and saves nothing/costs nothing when damping is off.
    T_lat_annual_mean = temperature_kelvin_for_lat(
        lat,
        day_of_year=pp.vernal_equinox_day,
        polar_cooling_scale=polar_cooling_scale,
        planet_params=pp,
    )
    # Land seasonal amplitude damping (2026-08-04, audit C1b).
    #
    # `temperature_kelvin_for_lat` returns *instantaneous local radiative
    # equilibrium*, which has no surface heat capacity and no dynamical damping
    # in it at all. Measured at 41.4 deg, its annual half-range is ~81 K against
    # Earth land's ~28 K -- the single largest error anywhere in this stack.
    # Everything downstream is a patch for one half of it: the three transport
    # trapezoids exist to lift a -33 C radiative winter, and `_land_cap_1d`
    # exists to cut a +41 C radiative summer. Because the trapezoids are added
    # every month while the cap only removes in summer, the pair does not cancel
    # -- it leaves the forcing's annual mean 21 K too warm and writes a flat top
    # across seven months (the square wave C1b tracks).
    #
    # The ocean branch below has damped its own seasonal swing since the model's
    # early days (`ocean_seasonal_frac`, same mean-preserving form). Land simply
    # never got the equivalent, so it ran with an implicit thermal inertia of
    # zero. This is that term: a contraction of the seasonal anomaly toward the
    # cell's own annual mean, which is **exactly mean-preserving** -- it cannot
    # move the annual-mean calibration the trapezoids and `_land_cap_1d` were
    # tuned against, only the swing about it.
    #
    # Land's damping is legitimately much weaker than ocean's (soil's thermal
    # inertia is weeks, not years) but it is not 1.0: the anomaly is also damped
    # by the atmospheric heat transport that responds to it, which is why an
    # energy-balance land surface swings far less than its own radiative
    # equilibrium. `land_seasonal_amplitude = 1.0` is an exact no-op and
    # reproduces the historical behaviour bit-for-bit.
    #
    # Applied below, once the maritime field exists, because the strength is
    # per-cell rather than global -- see `land_seasonal_amplitude_maritime`.
    _land_seasonal_amp = float(pp.land_seasonal_amplitude)
    T_base_land = np.repeat(T_lat_land[:, None], Wc, axis=1).astype(np.float32, copy=False) + co2_temp_offset
    _T_annual_mean_2d = (
        np.repeat(T_lat_annual_mean[:, None], Wc, axis=1).astype(np.float32, copy=False)
        + co2_temp_offset
    )
    # Land summer temperature cap: temperature_kelvin_for_lat gives radiative equilibrium,
    # but real land surfaces never reach those peaks because latent heat (ET), soil heat
    # capacity, and turbulent exchange all limit summer temperatures.
    # At 55-65°N the formula returns ~305 K (32°C) in summer — unrealistically hot —
    # pulling the simulation 12-13°C above Earth via the land_blend.
    # Cap: 301 K (~28°C) held through 0-45°, tapering linearly to 286 K (13°C) at 60°+.
    # This matches observed mid-latitude summer maxima for mean daily temperatures.
    #
    # Taper start moved from 0deg to 45deg (2026-07, wind/precip-model revisit): the old
    # linear-from-the-equator taper capped land at only ~290.6K (17.5C) at ~41.5N in
    # summer -- measured directly on real terrain to be *below* the ocean's own summer
    # temperature at the same latitude (~296-298K) year-round, including at the summer
    # peak. Real land is warmer than adjacent ocean in summer (lower heat capacity, faster
    # response) -- that's the thermal-low mechanism that drives monsoon-style onshore
    # moisture inflow into continental interiors. With the cap capping land *colder* than
    # the ocean even at peak summer, that sign was backwards, and wind divergence over
    # continental-interior boxes measured persistently positive (divergent, not
    # convergent) with no seasonal signal at all -- see moisture-transport-investigation
    # memory. Holding 301K through 45deg (same ceiling already accepted at the equator,
    # just extended) restores the correct summer land > ocean sign at mid-latitudes while
    # leaving the original 55-65N fix (this taper's whole reason for existing) untouched:
    # the endpoint is still 286K at 60deg+.
    _abs_lat_deg_land = np.abs(np.rad2deg(lat))  # (Hc,)
    _land_cap_1d = 301.0 - 15.0 * np.clip((_abs_lat_deg_land - 45.0) / 15.0, 0.0, 1.0)
    # Atmospheric meridional heat transport warms high-latitude land.
    # The Ferrel cell and synoptic eddies carry ~60% of the ocean transport value
    # poleward even over the Antarctic continent.  Without this, Antarctic winter
    # equilibrium falls to ~185 K; the 200 K floor then dominates the annual mean,
    # giving ~203 K (vs Earth 224 K at the South Pole).  Adding 0.60 × 40 K = 24 K
    # at the pole lifts the winter minimum to ~210 K and the annual mean to ~220-226 K.
    # Symmetric for both poles (no AMOC; that term is ocean-only and applied separately).
    # Atmospheric meridional heat transport to high-latitude LAND.
    # Increased coefficient from 0.42 → 0.65: synoptic eddies, the Ferrel cell, and
    # frontal systems deliver substantially more heat to high-latitude continental
    # interiors than the previous value implied.  At 65°N the Ferrel/eddy transport
    # raises winter land T_base by ~18K (vs ~11K before), which is the primary driver
    # of the NH zonal-mean cold bias: when winter T_base_land(65-85°N) was too cold,
    # land cells dragged the zonal mean ~10°C below Earth reference.
    # The formula only ramps above 42°, so latitudes ≤50°N change by <1K. ✓
    _atm_land_transport_1d = (
        0.65 * 34.0 * np.clip((_abs_lat_deg_land - 42.0) / 28.0, 0.0, 1.0) ** 1.5
    )
    # Mid-latitude storm-track heat transport (22-50°): winter cyclones and frontal
    # systems deliver substantial poleward heat well before the 42° ramp above kicks
    # in. Without this, zonal-mean land coldest-month temperatures at 30-55° come out
    # 15-30K too cold (observed: -37 to -40°C at 45-55°N vs Earth's -5 to -15°C),
    # which spuriously classifies most of Canada/Siberia/Central Asia as Dwd (extreme
    # continental, requires coldest month < -38°C) instead of Dfb. Modeled as a
    # trapezoid (ramps 22°→42°, flat to 42°, decays 42°→50° handing off to the ramp
    # above) rather than a Gaussian so it doesn't reopen the 60°+ balance already
    # tuned by the ramp above. Cuts off by 50° (not 60°) so it doesn't dampen the
    # sea-ice/CO2 sensitivity signal at ice-forming latitudes (test_2x_co2_less_ice).
    _midlat_rise = np.clip((_abs_lat_deg_land - 22.0) / 20.0, 0.0, 1.0)
    _midlat_fall = np.clip((50.0 - _abs_lat_deg_land) / 8.0, 0.0, 1.0)
    _midlat_storm_bonus_1d = 27.0 * _midlat_rise * _midlat_fall

    # 45-55°N handoff-gap fix (2026-07-30, real-terrain audit): the two terms above
    # were each independently tuned so *neither* one, in isolation, disturbs the
    # other's already-validated latitude range -- `_midlat_storm_bonus_1d` decays
    # to exactly zero at 50° (by design, to protect test_2x_co2_less_ice), and
    # `_atm_land_transport_1d` only starts ramping at 42° and stays under 1K through
    # 50° (by design, per its own comment above). But summing the two reveals a real
    # trough, not a smooth handoff: their combined total falls from a 27K plateau at
    # 42° to just 3.4K at 50°, before `_atm_land_transport_1d` alone slowly climbs
    # back through the 50s. That trough sits almost exactly on 45-55°N -- verified
    # directly on real terrain (`saves/test.npz`, 512x1024): Berlin, Moscow,
    # Winnipeg, Novosibirsk, and Kiev (all 50-55°N) all showed coldest-month means of
    # -37 to -39°C (real Earth: 0 to -18°C depending on maritime/continental
    # exposure), spuriously classifying most of Europe/Russia/the Canadian Prairies
    # as Dwd (extreme continental, requires coldest month < -38°C -- in reality a
    # climate confined to the remotest Siberian cold pole) instead of Cfb/Dfb. This
    # is the exact failure mode the comment above already describes as the
    # motivation for `_midlat_storm_bonus_1d` -- the trough the two trapezoids leave
    # between themselves reproduces it almost unchanged.
    #
    # Fix: a third, narrow trapezoid that is exactly zero outside 44-66° (so it
    # cannot touch the already-validated 22-42° plateau, and decays back to zero
    # well before the 65-90° ice-forming latitudes `_midlat_storm_bonus_1d`'s own
    # cutoff was chosen to protect), peaking at 20K right in the 50-52° trough
    # floor. Combined total across 38-70° is a smooth 18-27K band instead of
    # dropping to single digits -- verified by direct computation before touching
    # any test, not assumed from the shape alone. See test_2x_co2_less_ice /
    # test_ecs_sensitivity.py for the ice-sensitivity guard this must not break.
    _handoff_rise = np.clip((_abs_lat_deg_land - 44.0) / 6.0, 0.0, 1.0)
    _handoff_fall = np.clip((66.0 - _abs_lat_deg_land) / 16.0, 0.0, 1.0)
    _handoff_bonus_1d = 20.0 * _handoff_rise * _handoff_fall

    # Local seasonal signal: +1 at the local summer solstice, -1 at the local
    # winter solstice, 0 at the equinoxes. Computed here (rather than at its
    # original site further down, just above the evapotranspiration cooling)
    # because the transport seasonality below needs it too -- it depends only
    # on `pp`, `new_day` and `lat`, so hoisting it is side-effect free.
    # Coarse land mask, hoisted here (from the evapotranspiration block below)
    # so the continentality weighting on the transport bonus can use it too.
    # Depends only on `state.elevation` and the coarse grid, so the move is
    # side-effect free and the cached coarsening is shared, not repeated.
    if state.elevation is not None:
        _elev_c_early = _coarsen_elevation_cached(state.elevation, Hc, Wc, block_size)
        _, _land_mask_early = get_masks(_elev_c_early)
    else:
        _land_mask_early = np.zeros((Hc, Wc), dtype=bool)

    _delta_season = pp.solar_declination(new_day)
    _summer_signal_1d = np.sign(lat) * (_delta_season / max(pp.obliquity_rad, 1e-6))
    _summer_factor_1d = np.clip(_summer_signal_1d, 0.0, 1.0).astype(np.float32)  # 0 in winter, 1 at local summer peak

    # Seasonal modulation of the three land heat-transport trapezoids above.
    # All three are sized for a winter deficit but applied year-round; real eddy
    # heat flux scales with the meridional temperature gradient, which is far
    # larger in winter. Adding the full winter-sized bonus in summer pushes the
    # summer target tens of K above anything physical, which forces _land_cap_1d
    # below to hard-clamp seven consecutive months to an identical value -- that
    # clamp is the square wave. Cutting the summer half lets the target fall back
    # under the cap so the underlying sinusoid survives.
    # `land_transport_seasonality = 0.0` reproduces the previous behaviour
    # exactly. See the field's docstring in planet_params.py for the stage-by-
    # stage measurement.
    _land_transport_season_1d = (
        1.0 - float(pp.land_transport_seasonality) * _summer_signal_1d
    ).astype(np.float32)

    # `land_transport_gain` is the companion knob to `land_seasonal_amplitude`
    # above and is meaningless without it. All three trapezoids were sized
    # against an *undamped* radiative winter -- 27 K at 41 deg to lift a -33 C
    # January, 22 K at 65 deg to keep Antarctic winter off the 200 K floor. Damp
    # the swing and that deficit shrinks by construction, so the calibrated peaks
    # become an overshoot applied year-round, i.e. the +21 K annual-mean error
    # `_land_cap_1d` currently hides. This scales all three together, preserving
    # the latitude shape C1's handoff work calibrated; 1.0 is an exact no-op.
    _transport_total_1d = (
        (_atm_land_transport_1d + _midlat_storm_bonus_1d + _handoff_bonus_1d)
        * _land_transport_season_1d
        * float(pp.land_transport_gain)
    )[:, None].astype(np.float32, copy=False)
    # Hoisted from the evapotranspiration cooling further down so the
    # evapotranspirative amplitude damping below can share it. Depends only on
    # `state.soil_moisture` and the coarse grid, so moving it is side-effect free.
    if state.soil_moisture is not None:
        _soil_2d = _coarsen_many({"soil": state.soil_moisture}, Hc, Wc, block_size)["soil"]
    else:
        _soil_2d = np.full((Hc, Wc), 0.55, dtype=np.float32)

    # Continentality (2026-08-04, audit C1b). All three trapezoids above are pure
    # functions of |latitude|, so every land cell in a row receives the identical
    # winter heat-transport bonus -- the model has no maritime moderation
    # gradient at all. Measured against the Koppen reference's own threshold
    # bounds (`koppen_temperature_thresholds`), that shows up as an error with
    # *both* signs inside one latitude zone: at 40-50N the model puts 22% of
    # reference land too warm and 15% too cold, and the split is by
    # continentality -- 94.8% of 35-45N reference-D (continental) land has a
    # coldest month above the -3 C its class requires, while 99.5% of 45-55N
    # reference-C (maritime) land is below it. No latitude-only term can fix a
    # defect whose sign flips within a row.
    #
    # Deliberately **mean-preserving across each row's land**: the zonal-mean
    # level here was calibrated by C1's handoff work and the three trapezoids'
    # own tuning, and this mechanism exists to add contrast, not to relitigate
    # that level. `land_transport_maritime_decay = 0.0` is an exact no-op.
    #
    # The anomaly is divided by its own land-area-weighted standard deviation,
    # so `decay` is in units of "fraction of the bonus per standard deviation of
    # continentality" and does not silently rescale when the *shape* knobs
    # (`land_transport_maritime_km`, `land_transport_upwind_ratio`) change the
    # field's spread. Without it the two are badly degenerate: a longer upwind
    # reach flattens the field toward 1 everywhere, and the same physical effect
    # then needs decay 12 where a short reach needed 2. Global rather than
    # per-row: a row whose land is uniformly maritime genuinely has no
    # continentality contrast, and per-row normalization would manufacture one.
    #
    # 2026-08-04 (this session): re-enabled after the two fixes below. The
    # earlier negative result -- "reference-D land at 35-45N is neither more
    # continental nor higher than reference-C land, 0.312 vs 0.310" -- was a
    # population artifact, not a fact about Earth. It was measured by
    # re-deriving the Köppen reference *at the 32x64 coarse grid* (35 C cells
    # and 15 D cells for the whole band), while the metric it was aimed at
    # scores the fine grid. On the population actually scored, the two separate
    # by 0.84 sd. See `_maritime_proximity_coarse` and
    # `land_transport_upwind_ratio` for the two fixes that recovered it.
    _maritime_decay = float(pp.land_transport_maritime_decay)
    _amp_maritime = float(pp.land_seasonal_amplitude_maritime)
    _maritime = None
    if (
        (_maritime_decay > 0.0 or _amp_maritime != 0.0)
        and np.any(_land_mask_early)
        and state.elevation is not None
    ):
        _maritime = _maritime_proximity_coarse(
            state.elevation,
            Hc,
            Wc,
            block_size,
            float(pp.land_transport_maritime_km),
            float(pp.radius_m) / 1000.0,
            float(pp.land_transport_upwind_ratio),
        )
    if _maritime_decay > 0.0 and _maritime is not None:
        # Winter weighting: `_summer_signal_1d` is +1 at the local summer
        # solstice and -1 at the local winter solstice, so this is 1 through
        # winter and 0 through the whole summer half, smoothly and per
        # hemisphere.  See `_maritime_transport_factor` for why.
        _maritime_factor = _maritime_transport_factor(
            _maritime,
            _land_mask_early,
            lat,
            _maritime_decay,
            np.clip(-_summer_signal_1d, 0.0, 1.0).astype(np.float32),
        )
        _transport_total_1d = (
            _transport_total_1d * _maritime_factor
        ).astype(np.float32, copy=False)

    # Per-cell land seasonal amplitude (2026-08-05, audit C1b).
    #
    # `land_seasonal_amplitude` above sets the zonal level of the damping; this
    # sets its contrast within a row, and the contrast is the part that carries
    # real physical content. A maritime climate *is* a damped seasonal cycle --
    # that is the definition, not a consequence -- so the same continentality
    # field the winter transport bonus already uses is the correct modulator.
    #
    # This is also the mechanism `_maritime_transport_factor`'s winter gate
    # exists to work around. That factor is an additive *bonus*, so applying it
    # year-round warms maritime summers, which is the wrong sign; it had to be
    # restricted to winter and therefore cannot touch the model's
    # maritime-summer error at all (measured at 128x256: 81.5% of -50:-40 and
    # 48.7% of -40:-30 reference-C land -- Chile, New Zealand, Tasmania -- has a
    # warmest month above the 22 C its class requires). An *amplitude* damping
    # has the correct sign in both seasons by construction: it warms maritime
    # winters and cools maritime summers, from one term, with the cell's annual
    # mean untouched. That is why it needs no seasonal gate.
    #
    # Sign: high maritime proximity must *lower* the amplitude, so the shared
    # factor is called with a negated decay. Everything else about it is wanted
    # as-is -- the row-mean preservation (so `land_seasonal_amplitude` keeps
    # meaning the row's mean damping), the global spread normalization (so this
    # knob does not silently rescale when `land_transport_maritime_km` or
    # `land_transport_upwind_ratio` change the field), and the [-1, 1] bound
    # (amplitude stays in [0, 2x] -- a fully maritime cell can lose its whole
    # seasonal cycle but never invert it).
    if _amp_maritime != 0.0 and _maritime is not None:
        _amp_field = _land_seasonal_amp * _maritime_transport_factor(
            _maritime,
            _land_mask_early,
            lat,
            -_amp_maritime,
            np.ones_like(lat, dtype=np.float32),
        )
    elif _land_seasonal_amp != 1.0:
        _amp_field = np.full((Hc, Wc), _land_seasonal_amp, dtype=np.float32)
    else:
        _amp_field = None

    # Evapotranspirative damping of the land seasonal amplitude (2026-08-05,
    # audit C1b-EVAP). The *other* evapotranspiration term in this function --
    # the contraction toward `evap_cooling_threshold_k` below -- is a level
    # operation wearing a shape mechanism's clothes, and that is measurably why
    # strengthening it costs accuracy: it subtracts from a cell's annual mean,
    # which in the wet tropics (hot, high soil moisture, almost no seasonal
    # cycle to flatten) pushes rainforest land under Koppen's 18 C A-boundary.
    # Measured at 128x256, `evap_cooling_strength` 1.0 -> 2.0 loses 2.05pp of
    # group accuracy in 0:10 and 1.08pp in -20:-10 while *gaining* 2.92pp on the
    # US Midwest -- the mechanism is right about the mid-latitudes and wrong
    # about the tropics, from one term.
    #
    # Damping the seasonal *amplitude* by the same soil-moisture field is the
    # shape-only form of the identical physics: latent heat flux buffers a moist
    # surface's swing, and a dry one's is unbuffered. Being exactly
    # mean-preserving in time it cannot move any cell's annual mean, so it is
    # inert by construction exactly where the threshold form does damage --
    # a cell with no seasonal cycle has no amplitude to damp. Process note 20:
    # the level was the wrong quantity to modulate; the amplitude is the right
    # one.
    #
    # Shares `land_seasonal_amplitude_maritime`'s row-mean preservation and
    # global spread normalization (so `land_seasonal_amplitude` keeps meaning
    # the row's mean damping and this knob is in units of "fraction of amplitude
    # per standard deviation of soil moisture"), and its [-1, 1] bound. Soil
    # moisture is a *different* discriminator from continentality, not a proxy
    # for it: the Sahel is dry without being continental and the Pacific
    # Northwest is wet without being maritime-dominated in summer.
    _amp_soil = float(pp.evap_cooling_amplitude)
    if _amp_soil != 0.0 and np.any(_land_mask_early):
        _amp_soil_field = _maritime_transport_factor(
            _soil_2d,
            _land_mask_early,
            lat,
            -_amp_soil,
            np.ones_like(lat, dtype=np.float32),
        )
        _amp_field = (
            _amp_soil_field
            if _amp_field is None
            else (_amp_field * _amp_soil_field).astype(np.float32, copy=False)
        )
    if _amp_field is not None:
        # Land only: ocean cells of `T_base_land` are never consumed (the land
        # blend in `_evolve_temperature` is zero there), but leaving them
        # untouched keeps the array meaningful for anything that inspects it.
        _amp_field = np.where(_land_mask_early, _amp_field, 1.0).astype(
            np.float32, copy=False
        )
        T_base_land = (
            _T_annual_mean_2d + _amp_field * (T_base_land - _T_annual_mean_2d)
        ).astype(np.float32, copy=False)
    # Deficit gating (2026-08-03, audit C1b). The three trapezoids above are
    # sized for a winter deficit but added every month, which is the *mean*
    # error `_land_cap_1d` below then hides: measured offline at 41.4 deg, they
    # push the annual-mean target to 32 C against Earth's ~10 C, and no
    # amplitude-side knob can touch that (a relaxation preserves the mean
    # exactly, and `land_transport_seasonality` also does, since its seasonal
    # signal averages to zero over a year). That is why C1b's four shipped
    # knobs all traded shape for level and lost H10 accuracy.
    #
    # Real eddy heat flux scales with the meridional temperature gradient *and
    # damps it*, so the physically-shaped form is self-limiting: full strength
    # into a cold winter continent, nothing into a warm summer one, with no
    # prescribed seasonal schedule at all. Gating on the cell's own pre-bonus
    # temperature gives exactly that, and per-cell rather than per-row, so cold
    # continental interiors draw more than mild maritime cells at the same
    # latitude -- which is also the right sign for the continentality gradient.
    # `land_transport_deficit_k = 0.0` disables it and reproduces the flat
    # trapezoids bit-for-bit.
    _deficit_k = float(pp.land_transport_deficit_k)
    if _deficit_k > 0.0:
        # `land_transport_deficit_gain` is the affordance the gate creates and
        # is meaningless without it: once summer transport is identically zero,
        # the winter magnitude can be raised without re-opening the summer
        # overshoot that forced the clamp in the first place. The three
        # trapezoids were each capped by exactly that constraint, so their
        # calibrated peaks are lower bounds, not tuned optima.
        _transport_total_1d = (
            _transport_total_1d
            * float(pp.land_transport_deficit_gain)
            * np.clip(
                (_LAND_TRANSPORT_DEFICIT_REF_K - T_base_land) / _deficit_k, 0.0, 1.0
            ).astype(np.float32, copy=False)
        )
    T_base_land = T_base_land + _transport_total_1d

    # Evapotranspiration/convective cooling (2026-07, root-cause fix for the summer
    # overheating _land_cap_1d above only ever patched post-hoc): real land loses
    # substantial absorbed energy evaporating soil/plant moisture rather than raising
    # sensible temperature, especially at high summer insolation -- and critically,
    # this depends on how much moisture is actually *available*, not just latitude.
    # Measured directly: temperature_kelvin_for_lat's own radiative-equilibrium calc
    # reaches ~305-314K (33-41C) essentially UNIFORMLY across 20-70N at NH summer
    # solstice (no evapotranspiration physics at all below that function's own
    # is_polar>55deg threshold, and even above it the mechanism is tuned very weak
    # via polar_cooling_scale) -- a flat, moisture-blind profile that _land_cap_1d
    # was reshaping into a realistic latitude-dependent ceiling via a hard, seasonally
    # -discontinuous, moisture-blind clamp. This adds the missing physical mechanism
    # instead: cooling scales with (a) how far the pre-cooling temperature exceeds a
    # reference (more excess energy = more evaporative demand), (b) local soil
    # moisture (deserts get little cooling and stay realistically hot -- e.g. Sahara
    # summer means are genuinely very high -- while moist continental interior/boreal
    # land gets strong cooling), and (c) the local hemisphere's own seasonal cycle via
    # solar declination (near-zero in winter, peaking at the local summer solstice) so
    # the transition is smooth in time rather than an instant on/off clamp.
    # NOTE (measured 2026-08-02): the intent below was that _land_cap_1d becomes a
    # safety-net backstop rather than the primary mechanism. That did not happen.
    # This term lowers the 41N summer target from ~67C to ~44C -- still far above
    # the 27.9C ceiling -- so the cap continues to do all the work, binding on
    # 55.7% of (month, row) pairs at 25-50 deg. Do not rely on the old claim that
    # the cap rarely binds; see ACCURACY_AUDIT.md C1b.
    # `_soil_2d` is computed further up (hoisted so the evapotranspirative
    # amplitude damping can share it) -- it depends only on
    # `state.soil_moisture` and the coarse grid, so hoisting is side-effect free.
    # `_land_mask_early` is computed further up (hoisted so the continentality
    # weighting on the transport bonus can share it) -- it depends only on
    # `state.elevation` and the coarse grid, so hoisting is side-effect free.
    # `_summer_factor_1d` is computed further up (hoisted so the winter transport
    # boost can share the same seasonal signal) -- the two are exactly out of phase.
    # Both constants are `PlanetParams` fields as of 2026-08-05 (audit
    # C1b-EVAP). The threshold in particular is not a neutral tuning constant:
    # it is the term's *reach*, and at 290 K it excludes every zone whose
    # warmest month is actually wrong -- see its docstring.
    _EVAP_COOL_THRESHOLD_K = float(pp.evap_cooling_threshold_k)
    _EVAP_COOL_COEFF_MAX = float(pp.evap_cooling_coeff)
    # Saturating seasonal gate: a contraction toward the threshold is only
    # shape-preserving if its *strength* is constant through the warm season.
    # Letting the strength track `_summer_factor_1d` all the way to the solstice
    # (evap_cooling_season_width = 1.0) makes the cooling peak where the cycle
    # peaks, squaring off the top -- see the field's docstring in planet_params.py.
    _evap_season_1d = np.clip(
        _summer_factor_1d / max(float(pp.evap_cooling_season_width), 1e-6), 0.0, 1.0
    ).astype(np.float32)
    _evap_excess_2d = np.maximum(T_base_land - _EVAP_COOL_THRESHOLD_K, 0.0)
    _evap_frac_2d = _evap_cooling_fraction(
        _evap_season_1d[:, None],
        _soil_2d,
        _EVAP_COOL_COEFF_MAX,
        float(pp.evap_cooling_strength),
    )
    _evap_cooling_2d = (
        _evap_frac_2d * _evap_excess_2d
    ) * _land_mask_early.astype(np.float32)
    T_base_land = (T_base_land - _evap_cooling_2d).astype(np.float32, copy=False)

    # Re-apply summer cap: atmospheric transport can only raise winter/polar-night
    # temperatures; it must not push summer land above observed peak means.
    # NOT the "rarely-binding safety net" this comment used to claim: measured, it
    # binds on 55.7% of (month, row) pairs at 25-50 deg and clamps seven
    # consecutive months at 41N to one identical value. `land_cap_softness_k > 0`
    # swaps the hard minimum for a strictly-monotonic soft-min so the clamp stops
    # writing a flat top into the annual cycle.
    T_base_land = _soft_min_cap(
        T_base_land, _land_cap_1d[:, None].astype(np.float32, copy=False),
        float(pp.land_cap_softness_k),
    )

    # Calculate temperature for lagged day (ocean response with 1.5 month delay)
    lag_days = pp.ocean_lag_days * (float(pp.orbital_period_days) / 365.2422)  # scale thermal lag with year length
    lagged_day = (new_day - lag_days) % float(pp.orbital_period_days)
    T_lat_ocean_lagged = temperature_kelvin_for_lat(
        lat,
        day_of_year=lagged_day,
        polar_cooling_scale=polar_cooling_scale,
        planet_params=pp,
    )

    # CRITICAL: Two corrections to ocean base temperature:
    #
    # 1) SEASONAL AMPLITUDE DAMPING: The ocean's thermal time constant is ~1-3 YEARS,
    #    far longer than a season. SST barely oscillates around the annual mean.
    #    The 50-day lag shifts phase but doesn't damp amplitude. Without damping,
    #    winter T_base at 55-60N drops to 210-230K causing unrealistic ice.
    #
    # 2) MERIDIONAL HEAT TRANSPORT WARMING: temperature_kelvin_for_lat computes LOCAL
    #    radiative equilibrium, which ignores the ~2 PW of poleward ocean heat transport.
    #    Real SST at 55-70N is 12-42K warmer than radiative equilibrium due to Gulf
    #    Stream, Kuroshio, and thermohaline circulation. This offset is standard in
    #    energy balance climate models (Budyko 1969, Sellers 1969).

    # (`T_lat_annual_mean` is computed once, up with the land branch, which also
    # consumes it for `land_seasonal_amplitude`.)

    # Meridional heat transport warming: concentrated at high latitudes
    # 0K below 40°, ramping to 40K at 70°+ (matches observed SST - radiative eq deficit)
    # Profile: steep ramp starting at 40° prevents over-warming subtropics
    # The explicit ocean transport function handles finer redistribution (western
    # boundary currents, seasonal variation, east-west asymmetry)
    #
    # AMOC asymmetry (physically motivated): The Atlantic Meridional Overturning
    # Circulation transports ~1.2 PW northward into the Arctic with no SH equivalent.
    # This is why Earth's Arctic Ocean is ~10-15°C warmer than the Southern Ocean at
    # the same latitude.  Without this term both poles get identical base temperatures,
    # the NH Arctic falls into an ice-albedo runaway while the SH warms excessively.
    # Fix: NH latitudes > 50° receive a bonus +18 K (AMOC-driven warming), ramping
    # over 50-75°N where AMOC heat delivery is strongest.
    # Start at 50° (not 60°): the T_base_land summer cap already prevents mid-latitude
    # overheating, so we no longer need to restrict the AMOC ramp to avoid it.
    # Starting at 60° (Round 2) gave 0 K at 60°N and only 11 K at 70°N, which was
    # insufficient to keep Arctic SSTs above the ice melt threshold (NH edge regressed
    # from 70°N to 54°N).  50° start gives 7 K at 60°N and 14 K at 70°N — same as
    # the successful Round 1 ramp but with a slightly lower peak (18 vs 20 K).
    # SH transport raised from 50% to 65% of base: the Southern Ocean needs more warmth
    # to avoid its own cold runaway, while still remaining cooler than AMOC-warmed NH.
    lat_deg_1d = np.abs(np.rad2deg(lat))
    _ocean_scale = float(pp.has_liquid_water_ocean)
    # Scale AMOC/ACC with planet rotation rate and ocean fraction.
    # AMOC strength ∝ ω^0.4 (Coriolis drives western boundary currents; weaker on slow rotators).
    # AMOC suppressed entirely for retrograde rotators (Coriolis deflects opposite → no WBC).
    # ACC is primarily wind-driven so scales only with ocean_fraction, not rotation.
    _EARTH_OMEGA = 7.2921e-5  # rad/s  (2π / 23.9345 h)
    _rotation_scale = float(np.clip((pp.omega / _EARTH_OMEGA) ** 0.4, 0.05, 2.0))
    _ocean_frac_scale = float(pp.ocean_fraction / 0.71)
    _amoc_scale = _ocean_scale * _rotation_scale * _ocean_frac_scale * float(pp.rotation_direction > 0)
    _acc_scale  = _ocean_scale * _ocean_frac_scale
    _transport_base = _acc_scale * 34.0 * np.clip((lat_deg_1d - 42.0) / 28.0, 0.0, 1.0) ** 1.5
    # AMOC bonus: steep ramp from 65-75°N (3K at 65°N → 18K at 75°N+).
    # Scaled by dynamic feedback factor (amoc_factor: 0.30–1.00) and planet rotation/ocean params.
    # Geographic taper: bonus tapers to zero above pp.amoc_cutoff_lat to prevent NH pole over-warming.
    _amoc_taper = np.clip((pp.amoc_cutoff_lat - lat_deg_1d) / 10.0, 0.0, 1.0)
    _amoc_bonus = _amoc_scale * amoc_factor * _amoc_taper * np.where(
        lat > 0,
        pp.amoc_bonus_near * np.clip((lat_deg_1d - 42.0) / 23.0, 0.0, 1.0)
        + pp.amoc_bonus_far * np.clip((lat_deg_1d - 65.0) / 10.0, 0.0, 1.0),
        0.0,
    )  # NH only; tapers to 0 above amoc_cutoff_lat
    # ACC (Antarctic Circumpolar Current) bonus scaled by acc_factor (0.50–1.00).
    # Extensive Antarctic sea ice partially blocks CDW upwelling and reduces
    # the net poleward heat delivery by the ACC.
    _acc_bonus = _acc_scale * acc_factor * np.where(
        lat < 0,
        pp.acc_bonus_near * np.clip((lat_deg_1d - 55.0) / 10.0, 0.0, 1.0)
        + pp.acc_bonus_far * np.clip((lat_deg_1d - 65.0) / 10.0, 0.0, 1.0),
        0.0,
    )  # SH only; at 75-85°S total = acc_bonus_near+acc_bonus_far at full strength
    _sh_factor = np.where(lat > 0, 1.0, 0.58)   # SH gets weaker baseline transport than NH (no AMOC)
    # _transport_base's ramp only depends on |lat|, so it stays flat at its 34K max
    # all the way to the exact pole — unlike amoc_bonus, which already tapers to
    # zero above amoc_cutoff_lat. That left the NH pole cell ~30K warmer than
    # intended (the dominant cause of the too-small NH equator-pole gradient) and
    # made amoc_bonus_near/far tuning ineffective, since the metric samples the
    # exact pole row that amoc_bonus never reaches. Taper the NH share of the
    # generic transport too, so basin-average heat delivery also falls off near
    # the pole. Uses its own narrower (5°) taper rather than _amoc_taper's 10°:
    # widening it to overlap 60-70°N measurably fought the eddy-heat-flux
    # Laplacian smoothing (which acts over 20-70°) and *increased* zonal-mean
    # variance instead of reducing it once eddies were enabled (tested — 20°
    # width made the interaction worse, not better). Keeping the ramp entirely
    # above 70°N (75-85°N here) avoids overlapping the eddy band at all. SH
    # (ACC) side is left untouched — out of scope for the NH gradient fix.
    _nh_transport_taper = np.clip((pp.amoc_cutoff_lat - lat_deg_1d) / 5.0, 0.0, 1.0)
    _nh_transport_taper = np.where(lat > 0, _nh_transport_taper, 1.0)
    transport_warming = _transport_base * _sh_factor * _nh_transport_taper + _amoc_bonus + _acc_bonus

    # Seasonal fraction: what fraction of the radiative swing the ocean actually
    # feels. See `_ocean_seasonal_fraction`'s docstring for the legacy-vs-derived
    # (pp.derive_ocean_seasonal_lag) formulas.
    ocean_seasonal_frac = _ocean_seasonal_fraction(lat_deg_1d, pp)

    # Final ocean base: annual mean + transport warming + small seasonal oscillation
    T_lat_ocean = (T_lat_annual_mean + transport_warming
                   + ocean_seasonal_frac * (T_lat_ocean_lagged - T_lat_annual_mean))

    T_base_ocean = np.repeat(T_lat_ocean[:, None], Wc, axis=1).astype(np.float32, copy=False) + co2_temp_offset

    def _temperature_bases_for_day(day: float) -> tuple[np.ndarray, np.ndarray]:
        """Rebuild seasonal targets at an inner integration date.

        The static transport calibration is shared with the outer calculation;
        only radiative/seasonal terms are resampled.  This prevents temperature
        substeps from repeatedly relaxing toward a stale end-of-chunk land
        target.
        """
        T_land_lat = temperature_kelvin_for_lat(
            lat,
            day_of_year=day,
            polar_cooling_scale=polar_cooling_scale,
            planet_params=pp,
        )
        summer_signal = np.sign(lat) * (
            pp.solar_declination(day) / max(pp.obliquity_rad, 1e-6)
        )
        summer_factor = np.clip(
            np.clip(summer_signal, 0.0, 1.0)
            / max(float(pp.evap_cooling_season_width), 1e-6),
            0.0, 1.0,
        ).astype(np.float32)
        # Same seasonal transport modulation as the outer path, resampled at this
        # inner date. (NOTE: this path has always summed only two of the outer
        # path's three transport trapezoids -- `_handoff_bonus_1d` is absent
        # here. That is a pre-existing inconsistency, inert at the default
        # `temperature_substep_days=0.0`; left as found rather than changed
        # blind, since this path cannot be validated while it is disabled.)
        land_base_radiative = (
            np.repeat(T_land_lat[:, None], Wc, axis=1).astype(np.float32, copy=False)
            + co2_temp_offset
        )
        if _amp_field is not None:
            land_base_radiative = (
                _T_annual_mean_2d
                + _amp_field * (land_base_radiative - _T_annual_mean_2d)
            ).astype(np.float32, copy=False)
        land_base = (
            land_base_radiative
            + (
                (_atm_land_transport_1d + _midlat_storm_bonus_1d)
                * (1.0 - float(pp.land_transport_seasonality) * summer_signal)
                * float(pp.land_transport_gain)
            )[:, None]
        )
        evap_excess = np.maximum(land_base - _EVAP_COOL_THRESHOLD_K, 0.0)
        evap_cooling = (
            _evap_cooling_fraction(
                summer_factor[:, None],
                _soil_2d,
                _EVAP_COOL_COEFF_MAX,
                float(pp.evap_cooling_strength),
            )
            * evap_excess
            * _land_mask_early.astype(np.float32)
        )
        land_base = _soft_min_cap(
            land_base - evap_cooling,
            _land_cap_1d[:, None],
            float(pp.land_cap_softness_k),
        ).astype(np.float32, copy=False)

        ocean_day = (
            float(day) - lag_days
        ) % float(pp.orbital_period_days)
        ocean_lagged = temperature_kelvin_for_lat(
            lat,
            day_of_year=ocean_day,
            polar_cooling_scale=polar_cooling_scale,
            planet_params=pp,
        )
        ocean_lat = (
            T_lat_annual_mean
            + transport_warming
            + ocean_seasonal_frac * (ocean_lagged - T_lat_annual_mean)
        )
        ocean_base = (
            np.repeat(ocean_lat[:, None], Wc, axis=1).astype(np.float32, copy=False)
            + co2_temp_offset
        )
        return ocean_base, land_base

    def _compute_T_base_ocean_full() -> np.ndarray:
        """Full-resolution fallback base temperature.

        Only needed when wind evolves at full resolution (wind_bs <= 1) before
        `state.temperature` is initialized — computed lazily because the
        production path (wind_bs > 1 with initialized temperature) never uses
        it, and this block costs three full-resolution
        temperature_kelvin_for_lat calls plus the transport math per step.
        """
        lat_full = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * np.pi
        T_lat_ocean_full_lagged = temperature_kelvin_for_lat(
            lat_full,
            day_of_year=lagged_day,
            polar_cooling_scale=polar_cooling_scale,
            planet_params=pp,
        )
        T_lat_annual_mean_full = temperature_kelvin_for_lat(
            lat_full,
            day_of_year=pp.vernal_equinox_day,
            polar_cooling_scale=polar_cooling_scale,
            planet_params=pp,
        )
        lat_deg_full = np.abs(np.rad2deg(lat_full))
        _transport_base_full = _acc_scale * 34.0 * np.clip((lat_deg_full - 42.0) / 28.0, 0.0, 1.0) ** 1.5
        _amoc_taper_full = np.clip((pp.amoc_cutoff_lat - lat_deg_full) / 10.0, 0.0, 1.0)
        _amoc_bonus_full = _amoc_scale * amoc_factor * _amoc_taper_full * np.where(
            lat_full > 0,
            pp.amoc_bonus_near * np.clip((lat_deg_full - 42.0) / 23.0, 0.0, 1.0)
            + pp.amoc_bonus_far * np.clip((lat_deg_full - 65.0) / 10.0, 0.0, 1.0),
            0.0,
        )
        _acc_bonus_full = _acc_scale * acc_factor * np.where(
            lat_full < 0,
            pp.acc_bonus_near * np.clip((lat_deg_full - 55.0) / 10.0, 0.0, 1.0)
            + pp.acc_bonus_far * np.clip((lat_deg_full - 65.0) / 10.0, 0.0, 1.0),
            0.0,
        )
        _sh_factor_full = np.where(lat_full > 0, 1.0, 0.58)
        _nh_transport_taper_full = np.clip((pp.amoc_cutoff_lat - lat_deg_full) / 5.0, 0.0, 1.0)
        _nh_transport_taper_full = np.where(lat_full > 0, _nh_transport_taper_full, 1.0)
        transport_warming_full = (
            _transport_base_full * _sh_factor_full * _nh_transport_taper_full
            + _amoc_bonus_full + _acc_bonus_full
        )
        ocean_seasonal_frac_full = _ocean_seasonal_fraction(lat_deg_full, pp)
        T_lat_ocean_full = (T_lat_annual_mean_full + transport_warming_full
                            + ocean_seasonal_frac_full * (T_lat_ocean_full_lagged - T_lat_annual_mean_full))
        return np.repeat(T_lat_ocean_full[:, None], W, axis=1).astype(np.float32, copy=False) + co2_temp_offset

    def _compute_T_toa_equilibrium_full() -> np.ndarray:
        """Full-resolution top-of-atmosphere radiative-equilibrium temperature.

        Unlike ``_compute_T_base_ocean_full``, this carries no AMOC/ACC bonus,
        hemisphere asymmetry, or ocean seasonal lag -- it is
        ``temperature_kelvin_for_lat`` evaluated directly at the current
        (unlagged) day, plus the same CO2 forcing offset applied everywhere
        else.  It exists to isolate genuine radiative-heating skill from the
        transport-borrowed asymmetry that ``_compute_T_base_ocean_full``
        bakes in, per PRIOR_ART_IMPLEMENTATION_PLAN.md Section 10.
        """
        lat_full = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * np.pi
        T_lat_toa_full = temperature_kelvin_for_lat(
            lat_full,
            day_of_year=new_day,
            polar_cooling_scale=polar_cooling_scale,
            planet_params=pp,
        )
        return np.repeat(T_lat_toa_full[:, None], W, axis=1).astype(np.float32, copy=False) + co2_temp_offset


    # Blend based on land fraction (will be calculated in _evolve_temperature)
    # Use ocean-lagged temperature as base; _evolve_temperature will handle land/ocean mixing
    T_base = T_base_ocean  # Start with ocean (lagged), land will be corrected in evolution
    
    # Compute coarse elevation grid for wind/temperature evolution
    elev_c = _coarsen_elevation_cached(state.elevation, Hc, Wc, block_size) if state.elevation is not None else None

    # Update temperature with wind advection and land-sea effects.
    # Batched (see `_coarsen_many`): T_prev/T_air/ice/ice_thickness all share the
    # same (Hc, Wc, block_size) coarsening. Fallbacks (T_base.copy(), etc.) are
    # applied after the batch, same as the original per-field logic — a fallback
    # never depends on a value the batch itself needed to produce, except
    # T_air_coarse's fallback to T_prev_coarse, which is resolved first below.
    _group_a_in: dict[str, np.ndarray] = {}
    if state.temperature is not None:
        _group_a_in["T_prev"] = state.temperature
    # Downsample T_air; initialize from T_sst if not yet present (first step or old save)
    _T_air_src = state.air_temperature if state.air_temperature is not None else state.temperature
    if _T_air_src is not None:
        _group_a_in["T_air"] = _T_air_src
    if state.ice_cover is not None:
        _group_a_in["ice"] = state.ice_cover
    # Feature 6: sea ice thickness — initialize to 1 m where ice exists on first step
    _ice_thick_src = state.ice_thickness
    if _ice_thick_src is None and state.ice_cover is not None:
        _ice_thick_src = np.where(state.ice_cover > 0, 1.0, 0.0).astype(np.float32, copy=False)
    if _ice_thick_src is not None:
        _group_a_in["ice_thick"] = _ice_thick_src

    _group_a_out = _coarsen_many(_group_a_in, Hc, Wc, block_size)
    T_prev_coarse = _group_a_out["T_prev"] if "T_prev" in _group_a_out else T_base.copy()
    T_air_coarse = _group_a_out["T_air"] if "T_air" in _group_a_out else T_prev_coarse.copy()
    ice_prev_coarse = _group_a_out.get("ice")
    ice_thick_prev_coarse = _group_a_out.get("ice_thick")

    # Gated two-layer thermodynamic feedback.  Latent heating accumulated in
    # the persistent midlevel reservoir is not allowed to remain diagnostically
    # isolated: a mass-weighted vertical exchange transfers its anomaly into
    # the resolved air field before this step's radiation/advection update.
    # The companion adjustment to ``midlevel_temperature_for_precip`` removes
    # the same layer-energy anomaly, so this is an exchange rather than an
    # untracked temperature source.
    _two_layer_thermo_active = (
        bool(pp.enable_prognostic_column_water)
        and bool(pp.enable_stability_aware_condensation)
        and bool(pp.enable_two_layer_convective_adjustment)
    )
    midlevel_temperature_for_precip = state.midlevel_temperature
    upperlevel_temperature_for_precip = state.upperlevel_temperature
    if _two_layer_thermo_active and state.midlevel_temperature is not None:
        _lower_temperature_before_vertical_exchange = T_air_coarse
        _midlevel_coarse = _coarsen(
            state.midlevel_temperature, Hc, Wc, block_size
        )
        _reference_midlevel = _lower_temperature_before_vertical_exchange - (
            6.5e-3 * float(pp.stability_condensation_reference_height_m)
        )
        _exchange_fraction = 1.0 - np.exp(
            -float(days) / float(pp.two_layer_vertical_mixing_days)
        )
        _midlevel_exchange_coarse = _exchange_fraction * (
            _midlevel_coarse - _reference_midlevel
        )
        _midlevel_mass_fraction = float(
            np.clip(pp.two_layer_upper_mass_fraction, 0.05, 0.95)
        )
        _upperlevel_mass_fraction = (
            float(np.clip(pp.three_level_upper_humidity_fraction, 0.01, 0.50))
            if bool(pp.enable_three_level_pressure_column)
            else 0.0
        )
        _lower_mass_fraction = max(
            1.0 - _midlevel_mass_fraction - _upperlevel_mass_fraction, 0.05
        )
        _air_temperature_exchange = (
            _midlevel_mass_fraction / _lower_mass_fraction
        ) * _midlevel_exchange_coarse
        _upperlevel_exchange_coarse = None
        if (
            bool(pp.enable_three_level_pressure_column)
            and state.upperlevel_temperature is not None
        ):
            _upperlevel_coarse = _coarsen(
                state.upperlevel_temperature, Hc, Wc, block_size
            )
            _reference_upperlevel = _lower_temperature_before_vertical_exchange - (
                6.5e-3 * float(pp.three_level_upper_height_m)
            )
            _upperlevel_exchange_coarse = _exchange_fraction * (
                _upperlevel_coarse - _reference_upperlevel
            )
            _air_temperature_exchange = _air_temperature_exchange + (
                _upperlevel_mass_fraction / _lower_mass_fraction
            ) * _upperlevel_exchange_coarse
        T_air_coarse = np.clip(
            _lower_temperature_before_vertical_exchange + _air_temperature_exchange,
            150.0,
            350.0,
        ).astype(np.float32, copy=False)
        if block_size > 1:
            _midlevel_exchange_full = _upsample_bilinear_many(
                {"exchange": _midlevel_exchange_coarse}, H, W, block_size
            )["exchange"]
        else:
            _midlevel_exchange_full = _midlevel_exchange_coarse
        midlevel_temperature_for_precip = np.clip(
            state.midlevel_temperature - _midlevel_exchange_full,
            150.0,
            350.0,
        ).astype(np.float32, copy=False)
        if _upperlevel_exchange_coarse is not None:
            if block_size > 1:
                _upperlevel_exchange_full = _upsample_bilinear_many(
                    {"exchange": _upperlevel_exchange_coarse}, H, W, block_size
                )["exchange"]
            else:
                _upperlevel_exchange_full = _upperlevel_exchange_coarse
            upperlevel_temperature_for_precip = np.clip(
                state.upperlevel_temperature - _upperlevel_exchange_full,
                150.0,
                350.0,
            ).astype(np.float32, copy=False)
    
    # ------------------------------------------------------
    # Jet stream dynamics: persistent meander index + blocking events
    # (see atmosphere._update_jet_index / _update_jet_blocking). Computed once
    # per step from the actual simulated temperature field -- weaker
    # pole-equator gradient nudges the index toward "wavy" -- then fed into
    # whichever evolve_wind() call below executes.
    # ------------------------------------------------------
    lat_c_deg_1d = np.rad2deg((0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi)
    T_zm = np.mean(T_prev_coarse, axis=1).astype(np.float64, copy=False)

    def _pole_eq_gradient(hemi_sign: float) -> float:
        trop_mask = (lat_c_deg_1d * hemi_sign >= 0.0) & (np.abs(lat_c_deg_1d) <= 30.0)
        polar_mask = (lat_c_deg_1d * hemi_sign >= 0.0) & (np.abs(lat_c_deg_1d) >= 60.0)
        if not np.any(trop_mask) or not np.any(polar_mask):
            return float(pp.jet_gradient_ref_k)
        return float(np.mean(T_zm[trop_mask]) - np.mean(T_zm[polar_mask]))

    jet_index_nh_new = _update_jet_index(
        state.jet_index_nh, _pole_eq_gradient(1.0), days, new_total_days, hemisphere_seed=1,
        tau_days=float(pp.jet_meander_tau_days), noise_amp=float(pp.jet_meander_noise_amp),
        gradient_ref_k=float(pp.jet_gradient_ref_k),
    )
    jet_index_sh_new = _update_jet_index(
        state.jet_index_sh, _pole_eq_gradient(-1.0), days, new_total_days, hemisphere_seed=2,
        tau_days=float(pp.jet_meander_tau_days), noise_amp=float(pp.jet_meander_noise_amp),
        gradient_ref_k=float(pp.jet_gradient_ref_k),
    )
    jet_block_lon_nh_new, jet_block_days_left_nh_new, jet_block_total_nh_new = _update_jet_blocking(
        state.jet_block_lon_nh, state.jet_block_days_left_nh, state.jet_block_total_days_nh,
        jet_index_nh_new, days, new_total_days, hemisphere_seed=1,
        trigger_rate_per_day=float(pp.jet_block_trigger_rate_per_day),
        duration_range_days=pp.jet_block_duration_range_days,
    )
    jet_block_lon_sh_new, jet_block_days_left_sh_new, jet_block_total_sh_new = _update_jet_blocking(
        state.jet_block_lon_sh, state.jet_block_days_left_sh, state.jet_block_total_days_sh,
        jet_index_sh_new, days, new_total_days, hemisphere_seed=2,
        trigger_rate_per_day=float(pp.jet_block_trigger_rate_per_day),
        duration_range_days=pp.jet_block_duration_range_days,
    )
    _jet_block_nh = (jet_block_lon_nh_new, jet_block_days_left_nh_new, jet_block_total_nh_new)
    _jet_block_sh = (jet_block_lon_sh_new, jet_block_days_left_sh_new, jet_block_total_sh_new)

    # ------------------------------------------------------
    # NEW: Prognostic Wind Evolution (Physics Items 16-33)
    # ------------------------------------------------------
    # If wind is None, initialize it near-rest (small noise) so circulation spins up
    # from pressure gradients (Hadley-like overturning) rather than a synthetic target.
    if state.wind_u is None or state.wind_v is None:
        rng = np.random.default_rng(12345)
        u_full = rng.normal(0.0, 0.15, size=(H, W)).astype(np.float32, copy=False)
        v_full = rng.normal(0.0, 0.15, size=(H, W)).astype(np.float32, copy=False)
    else:
        u_full, v_full = state.wind_u, state.wind_v

    # 1.5-layer atmosphere: upper-level prognostic wind (Feature 8). Lazy-init
    # as a copy of the surface wind (a physically-reasonable warm start) if
    # this is the first step or an old save predates this field.
    if state.wind_u_aloft is None or state.wind_v_aloft is None:
        u2_full, v2_full = u_full.copy(), v_full.copy()
    else:
        u2_full, v2_full = state.wind_u_aloft, state.wind_v_aloft

    # Evolve wind at `wind_block_size` resolution (can differ from temperature/precip `block_size`)
    # Then upsample to full resolution for precipitation
    # Cached diagnostic wind for relaxation (once per day/shape/params).
    def _diag_wind_cached(h: int, w: int, temp_field: np.ndarray, elev_field: np.ndarray):
        key = (
            h,
            w,
            # `new_day` wraps every orbital period (0..~365), so keying on it
            # alone made this cache reuse year-1's storm/Rossby-wave snapshot
            # for every later year's same calendar day at MONTHLY/ANNUAL speed
            # -- freezing the diagnostic wind's weather into a single repeating
            # year instead of continuing to evolve with `new_total_days`. Key
            # on the monotonic day count instead; still only varies once per
            # simulated day, which was the actual perf intent.
            int(new_total_days),
            float(wind_target_weather_amp),
            float(wind_target_zonal_pressure),
            float(wind_target_terrain_pressure_amp),
            float(wind_target_terrain_flow_amp),
            round(float(jet_index_nh_new), 3),
            round(float(jet_index_sh_new), 3),
            tuple(round(float(x), 2) for x in _jet_block_nh),
            tuple(round(float(x), 2) for x in _jet_block_sh),
            round(float(pp.solar_constant), 4),
            round(float(pp.obliquity_deg), 4),
            round(float(pp.sidereal_day_hours), 4),
            round(float(pp.radius_m), 1),
            round(float(pp.pgf_continentality_amp), 4),
            round(float(getattr(pp, "ferrel_v_centre_deg", 48.0)), 4),
            round(float(getattr(pp, "ferrel_v_land_shift_deg", 0.0)), 4),
        )
        cache = _RELAX_CACHE
        if cache["key"] == key and cache["u"] is not None and cache["v"] is not None:
            return cache["u"], cache["v"]
        u_diag, v_diag = generate_wind_field(
            h,
            w,
            day_of_year=new_day,
            block_size=1,
            temperature=temp_field,
            elevation=elev_field,
            weather_amp=float(wind_target_weather_amp),
            zonal_pressure=float(wind_target_zonal_pressure),
            terrain_pressure_amp=float(wind_target_terrain_pressure_amp),
            terrain_flow_amp=float(wind_target_terrain_flow_amp),
            time_days=new_total_days,
            planet_params=pp,
            jet_index_nh=jet_index_nh_new,
            jet_index_sh=jet_index_sh_new,
            jet_block_nh=_jet_block_nh,
            jet_block_sh=_jet_block_sh,
        )
        cache.update({"key": key, "u": u_diag, "v": v_diag})
        return u_diag, v_diag

    # Honor `update_wind`: MONTHLY/ANNUAL substeps pass update_wind=False by
    # design (PLAN.md Open Question 1: "cached relaxation target, chosen for
    # speed"). This flag was silently ignored — wind (including storm systems
    # and jet dynamics) evolved prognostically every step in every mode,
    # costing the exact work those modes were designed to skip. When
    # update_wind=False, wind now follows the cached *diagnostic* wind
    # (generate_wind_field's seasonal climatology, refreshed once per
    # simulated day via _RELAX_CACHE) instead of either full prognostic
    # evolution (old behavior, expensive) or a permanently frozen field
    # (which would lose the seasonal wind cycle entirely on long
    # MONTHLY/ANNUAL runs). Wind still evolves prognostically when no wind
    # exists yet (first step).
    _do_evolve_wind = bool(update_wind) or state.wind_u is None or state.wind_v is None
    # The legacy one-level wind path has a prescribed 3-cell zonal-mean
    # meridional target.  A native pressure-coordinate balanced core must not
    # simultaneously be forced toward those 6--10 m/s empirical cells: its
    # overturning has to arise from the resolved pressure/temperature state.
    # Keep the historical target untouched outside this explicitly gated path.
    _native_balanced_momentum_active = bool(
        pp.enable_native_balanced_pressure_dynamics
        and pp.enable_three_level_pressure_column
        and _two_layer_thermo_active
    )
    _wind_cell_relax_days = (
        0.0 if _native_balanced_momentum_active else _wind_cell_relax_days_arg
    )

    # Opt-in (PlanetParams.wind_prognostic_substep_days, default 0.0/off): when
    # the caller requested the diagnostic MONTHLY/ANNUAL wind (update_wind=
    # False), run the real prognostic evolve_wind/evolve_wind_aloft path
    # instead, internally sub-stepped in ~wind_prognostic_substep_days chunks
    # via _evolve_wind_substepped -- trading MONTHLY/ANNUAL's speed for
    # DAILY-consistent wind (see PlanetParams.wind_prognostic_substep_days
    # docstring for why/cost). Default 0.0 makes this an exact no-op: DAILY/
    # WEEKLY (update_wind=True) are untouched either way, and MONTHLY/ANNUAL
    # keep today's diagnostic path and speed unless a user opts in.
    _wind_substep_days = float(pp.wind_prognostic_substep_days)
    _prognostic_gate_active = (not _do_evolve_wind) and _wind_substep_days > 0.0
    if _prognostic_gate_active:
        _do_evolve_wind = True

    if not _do_evolve_wind:
        # Diagnostic/climatological wind on the wind grid, upsampled to full res.
        if wind_bs > 1:
            _elev_c_w = _coarsen_elevation_cached(state.elevation, Hcw, Wcw, wind_bs)
            if state.temperature is not None:
                _T_w = _coarsen(state.temperature, Hcw, Wcw, wind_bs)
            else:
                _T_w = None
            u_diag, v_diag = _diag_wind_cached(Hcw, Wcw, _T_w, _elev_c_w)
            uv = _upsample_bilinear_many({"u": u_diag, "v": v_diag}, H, W, wind_bs)
            u_full, v_full = uv["u"], uv["v"]
        else:
            u_full, v_full = _diag_wind_cached(H, W, state.temperature, state.elevation)
    elif wind_bs > 1:
        # Downsample wind/temperature/elevation/ice for evolution on the wind grid (batched).
        _group_b_in: dict[str, np.ndarray] = {
            "u": u_full, "v": v_full, "u2": u2_full, "v2": v2_full,
        }
        if state.temperature is not None:
            _group_b_in["T"] = state.temperature
        if state.ice_cover is not None:
            _group_b_in["ice"] = state.ice_cover
        _group_b_out = _coarsen_many(_group_b_in, Hcw, Wcw, wind_bs)
        u_coarse_evol = _group_b_out["u"]
        v_coarse_evol = _group_b_out["v"]
        u2_coarse_evol = _group_b_out["u2"]
        v2_coarse_evol = _group_b_out["v2"]
        ice_c_w = _group_b_out.get("ice")
        elev_c_w = _coarsen_elevation_cached(state.elevation, Hcw, Wcw, wind_bs)

        if state.temperature is not None:
            T_for_wind = _group_b_out["T"]
        else:
            # When temperature is not yet initialized, use the same lagged-ocean base but on the wind grid.
            lat_w = (0.5 - (np.arange(Hcw, dtype=np.float32) + 0.5) / Hcw) * np.pi
            T_lat_ocean_w = temperature_kelvin_for_lat(
                lat_w,
                day_of_year=lagged_day,
                polar_cooling_scale=polar_cooling_scale,
                planet_params=pp,
            )
            T_for_wind = np.repeat(T_lat_ocean_w[:, None], Wcw, axis=1).astype(np.float32, copy=False)

        # 1.5-layer atmosphere: evolve the upper-level wind first, at the same
        # wind-grid resolution, so evolve_wind's baroclinic mixing term below
        # can relax the surface toward this step's freshly-updated aloft wind.
        # Internally sub-stepped in ~wind_prognostic_substep_days chunks when
        # that gate is active (n_sub=1, i.e. one call at dt_days=days,
        # otherwise -- see _evolve_wind_substepped).
        u_coarse_evol, v_coarse_evol, u2_coarse_evol, v2_coarse_evol = _evolve_wind_substepped(
            u_coarse_evol, v_coarse_evol, u2_coarse_evol, v2_coarse_evol,
            temperature=T_for_wind,
            elevation=elev_c_w,
            ice_cover=ice_c_w,
            dt_days_total=days,
            substep_days=_wind_substep_days if _prognostic_gate_active else 0.0,
            time_days_end=float(new_total_days),
            damping=float(wind_damping),
            pgf_temp_scale=float(wind_pgf_temp_scale),
            pgf_terrain_scale=float(wind_pgf_terrain_scale) * float(pp.wind_terrain_pgf_scale),
            drag_base=float(wind_drag_base),
            drag_elev_scale=float(wind_drag_elev_scale),
            vmax_clip=float(wind_vmax_clip),
            baroclinic_jet_amp=float(wind_baroclinic_jet_amp),
            baroclinic_mix=float(wind_baroclinic_mix),
            cell_relax_days=_wind_cell_relax_days,
            planet_params=pp,
            jet_index_nh=jet_index_nh_new,
            jet_index_sh=jet_index_sh_new,
            jet_block_nh=_jet_block_nh,
            jet_block_sh=_jet_block_sh,
            upper_pgf_amp=float(pp.wind_upper_pgf_amp),
            upper_damping=float(pp.wind_upper_damping),
            upper_hadley_edge_deg=float(pp.wind_upper_hadley_edge_deg),
        )

        # Keep winds energized + seasonally varying by weakly relaxing toward a diagnostic wind
        # (generate_wind_field injects synoptic-scale "weather systems" seeded by day_of_year).
        if wind_relax > 0.0:
            u_diag, v_diag = _diag_wind_cached(Hcw, Wcw, T_for_wind, elev_c_w)
            a = float(np.clip(wind_relax, 0.0, 1.0))
            u_coarse_evol = (1.0 - a) * u_coarse_evol + a * u_diag
            v_coarse_evol = (1.0 - a) * v_coarse_evol + a * v_diag

        # Upsample back to full resolution using bilinear interpolation
        uv = _upsample_bilinear_many(
            {"u": u_coarse_evol, "v": v_coarse_evol, "u2": u2_coarse_evol, "v2": v2_coarse_evol},
            H, W, wind_bs,
        )
        u_full, v_full = uv["u"], uv["v"]
        u2_full, v2_full = uv["u2"], uv["v2"]
    else:
        # Full resolution evolution
        # If wind evolves at higher resolution than the temperature solver, drive it with the
        # coarse temperature field upsampled to full resolution. This avoids injecting
        # grid-scale temperature noise into the wind solver (which can blow up speeds),
        # while still allowing the wind numerics to run on the fine grid.
        T_wind_full = state.temperature if state.temperature is not None else _compute_T_base_ocean_full()
        elev_wind_full = state.elevation
        if wind_bs < block_size and block_size > 1:
            to_up = {}
            to_up["T"] = T_prev_coarse if state.temperature is not None else T_base
            if elev_c is not None:
                to_up["elev"] = elev_c
            up = _upsample_bilinear_many(to_up, H, W, block_size)
            T_wind_full = up["T"]
            if "elev" in up:
                elev_wind_full = up["elev"]

        # 1.5-layer atmosphere: evolve the upper-level wind first (same
        # full-resolution grid), so evolve_wind's baroclinic mixing term below
        # can relax the surface toward this step's freshly-updated aloft wind.
        # Internally sub-stepped in ~wind_prognostic_substep_days chunks when
        # that gate is active (see _evolve_wind_substepped).
        u_full, v_full, u2_full, v2_full = _evolve_wind_substepped(
            u_full, v_full, u2_full, v2_full,
            temperature=T_wind_full,
            elevation=elev_wind_full,
            ice_cover=state.ice_cover,
            dt_days_total=days,
            substep_days=_wind_substep_days if _prognostic_gate_active else 0.0,
            time_days_end=float(new_total_days),
            damping=float(wind_damping),
            pgf_temp_scale=float(wind_pgf_temp_scale),
            pgf_terrain_scale=float(wind_pgf_terrain_scale) * float(pp.wind_terrain_pgf_scale),
            drag_base=float(wind_drag_base),
            drag_elev_scale=float(wind_drag_elev_scale),
            vmax_clip=float(wind_vmax_clip),
            baroclinic_jet_amp=float(wind_baroclinic_jet_amp),
            baroclinic_mix=float(wind_baroclinic_mix),
            cell_relax_days=_wind_cell_relax_days,
            planet_params=pp,
            jet_index_nh=jet_index_nh_new,
            jet_index_sh=jet_index_sh_new,
            jet_block_nh=_jet_block_nh,
            jet_block_sh=_jet_block_sh,
            upper_pgf_amp=float(pp.wind_upper_pgf_amp),
            upper_damping=float(pp.wind_upper_damping),
            upper_hadley_edge_deg=float(pp.wind_upper_hadley_edge_deg),
        )
        if wind_relax > 0.0:
            T_for_wind = T_wind_full
            u_diag, v_diag = _diag_wind_cached(H, W, T_for_wind, elev_wind_full)
            a = float(np.clip(wind_relax, 0.0, 1.0))
            u_full = (1.0 - a) * u_full + a * u_diag
            v_full = (1.0 - a) * v_full + a * v_diag

    # Optional two-level extraction of the existing thermally-direct,
    # mass-conserving overturning primitive.  The same lower/return structure
    # used by the three-level experiment applies directly to the normal surface
    # and upper-wind state, avoiding a second circulation formulation.
    if bool(pp.enable_two_level_thermally_direct_overturning):
        _two_level_speed = max(float(pp.two_level_thermally_direct_overturning_speed_m_s), 0.0)
        if _two_level_speed > 0.0:
            _two_level_temperature = (
                state.air_temperature
                if state.air_temperature is not None and state.air_temperature.shape == u_full.shape
                else (state.temperature if state.temperature is not None else _compute_T_base_ocean_full())
            )
            _two_level_overturning = thermally_direct_overturning(
                _two_level_temperature,
                hadley_edge_deg=float(pp.wind_upper_hadley_edge_deg),
                lower_branch_speed_m_s=_two_level_speed,
            )
            v_full = (v_full + _two_level_overturning.lower_v).astype(np.float32, copy=False)
            v2_full = (v2_full + _two_level_overturning.upper_v).astype(np.float32, copy=False)

    # Native three-level pressure-coordinate momentum.  The middle pressure
    # level has an independent prognostic circulation, not a diagnostic mean
    # of the surface and upper winds.  Its divergence closes both vapor
    # exchange interfaces in generate_precipitation.
    _three_level_midwind_active = (
        _two_layer_thermo_active
        and bool(pp.enable_three_level_pressure_column)
    )
    midlevel_wind_u_full = state.midlevel_wind_u
    midlevel_wind_v_full = state.midlevel_wind_v
    # Section 17: the three-level path's own independent upper wind, genuinely
    # decoupled from the shared, always-on jet-stream kernel (u2_full/v2_full,
    # persisted as state.wind_u_aloft/wind_v_aloft). See
    # PRIOR_ART_IMPLEMENTATION_PLAN.md Section 16/17 and
    # `_evolve_upper_wind_substepped`'s docstring.
    upperlevel_wind_u_full = state.upperlevel_wind_u
    upperlevel_wind_v_full = state.upperlevel_wind_v
    if _three_level_midwind_active:
        if (
            upperlevel_wind_u_full is None
            or upperlevel_wind_v_full is None
            or upperlevel_wind_u_full.shape != u2_full.shape
            or upperlevel_wind_v_full.shape != v2_full.shape
        ):
            # Warm-start as a copy of the shared kernel's current value --
            # mirrors u2_full/v2_full's own lazy-init ("copy of the surface
            # wind" as a physically-reasonable warm start) and reproduces this
            # step's pre-Section-17 upper-wind value exactly as the starting
            # point before the two states evolve independently from here on.
            upperlevel_wind_u_full = u2_full.copy()
            upperlevel_wind_v_full = v2_full.copy()

        # Own prognostic momentum step -- the same free-tropospheric kernel as
        # the shared upper level and the independent middle level, but with
        # this level's own PGF fraction/damping (three_level_upper_wind_*),
        # entirely independent of the shared kernel's own evolution above.
        _upper_wind_temperature = (
            upperlevel_temperature_for_precip
            if upperlevel_temperature_for_precip is not None
            and upperlevel_temperature_for_precip.shape == upperlevel_wind_u_full.shape
            else state.air_temperature
            if state.air_temperature is not None
            and state.air_temperature.shape == upperlevel_wind_u_full.shape
            else _compute_T_base_ocean_full()
        )
        upperlevel_wind_u_full, upperlevel_wind_v_full = _evolve_upper_wind_substepped(
            upperlevel_wind_u_full,
            upperlevel_wind_v_full,
            temperature=_upper_wind_temperature,
            dt_days_total=days,
            substep_days=_wind_substep_days if _prognostic_gate_active else 0.0,
            pgf_temp_scale=float(wind_pgf_temp_scale),
            upper_pgf_amp=(
                float(pp.wind_upper_pgf_amp)
                * float(pp.three_level_upper_wind_pgf_fraction)
            ),
            damping_rate=float(pp.three_level_upper_wind_damping),
            vmax_clip=float(wind_vmax_clip),
            planet_params=pp,
            hadley_edge_deg=float(pp.wind_upper_hadley_edge_deg),
        )

        _balanced_pressure_relaxation = (
            float(np.clip(pp.native_balanced_pressure_relaxation, 0.0, 1.0))
            if bool(pp.enable_native_balanced_pressure_dynamics)
            else 0.0
        )
        _balanced_elevation_m = 1000.0 * elevation_to_alt_km(
            state.elevation, max_elevation_km=float(pp.max_elevation_km),
        )
        _balanced_ageo_hours = float(
            pp.native_balanced_ageostrophic_timescale_hours
        )
        def _balanced_target(level_temperature, level_pressure_pa):
            _phi = pressure_level_geopotential(
                _balanced_elevation_m,
                level_temperature,
                gravity_m_s2=float(pp.surface_gravity),
                gas_constant_dry=float(pp.gas_constant_dry),
                surface_pressure_pa=float(pp.surface_pressure_pa),
                level_pressure_pa=float(level_pressure_pa),
            )
            return balanced_pressure_wind(
                _phi,
                radius_m=float(pp.radius_m),
                sidereal_day_hours=float(pp.sidereal_day_hours),
                hadley_edge_deg=float(pp.wind_upper_hadley_edge_deg),
                ageostrophic_timescale_hours=_balanced_ageo_hours,
            )

        if _balanced_pressure_relaxation > 0.0:
            _lower_balanced_temperature = (
                state.air_temperature
                if state.air_temperature is not None
                and state.air_temperature.shape == u_full.shape
                else state.temperature
                if state.temperature is not None
                and state.temperature.shape == u_full.shape
                else _compute_T_base_ocean_full()
            )
            _lower_balanced = _balanced_target(
                _lower_balanced_temperature,
                pp.native_balanced_surface_pressure_pa,
            )
            u_full = (
                (1.0 - _balanced_pressure_relaxation) * u_full
                + _balanced_pressure_relaxation * _lower_balanced.u
            ).astype(np.float32, copy=False)
            v_full = (
                (1.0 - _balanced_pressure_relaxation) * v_full
                + _balanced_pressure_relaxation * _lower_balanced.v
            ).astype(np.float32, copy=False)
            # Section 17: this three-level-only blend now applies to the
            # independent upper-level state, never to the shared jet-stream
            # kernel (u2_full/v2_full) -- see PRIOR_ART_IMPLEMENTATION_PLAN.md
            # Section 16/17.
            _upper_balanced_temperature = (
                upperlevel_temperature_for_precip
                if upperlevel_temperature_for_precip is not None
                and upperlevel_temperature_for_precip.shape == upperlevel_wind_u_full.shape
                else state.air_temperature
                if state.air_temperature is not None
                and state.air_temperature.shape == upperlevel_wind_u_full.shape
                else _compute_T_base_ocean_full()
            )
            _upper_balanced = _balanced_target(
                _upper_balanced_temperature,
                pp.three_level_thermal_wind_upper_pressure_pa,
            )
            upperlevel_wind_u_full = (
                (1.0 - _balanced_pressure_relaxation) * upperlevel_wind_u_full
                + _balanced_pressure_relaxation * _upper_balanced.u
            ).astype(np.float32, copy=False)
            upperlevel_wind_v_full = (
                (1.0 - _balanced_pressure_relaxation) * upperlevel_wind_v_full
                + _balanced_pressure_relaxation * _upper_balanced.v
            ).astype(np.float32, copy=False)
        _thermal_wind_relaxation = float(np.clip(
            pp.three_level_balanced_thermal_wind_relaxation, 0.0, 1.0,
        ))
        if _thermal_wind_relaxation > 0.0:
            _thermal_wind_temperature = (
                state.air_temperature
                if state.air_temperature is not None
                and state.air_temperature.shape == u_full.shape
                else state.temperature
                if state.temperature is not None
                and state.temperature.shape == u_full.shape
                else _compute_T_base_ocean_full()
            )
            _thermal_wind_target = balanced_thermal_wind_u(
                u_full,
                _thermal_wind_temperature,
                radius_m=float(pp.radius_m),
                sidereal_day_hours=float(pp.sidereal_day_hours),
                surface_pressure_pa=float(pp.surface_pressure_pa),
                upper_pressure_pa=float(pp.three_level_thermal_wind_upper_pressure_pa),
                hadley_edge_deg=float(pp.wind_upper_hadley_edge_deg),
                gas_constant_dry=float(pp.gas_constant_dry),
            )
            # Section 17: redirected to the independent upper-level state (see
            # the balanced-pressure blend above for the same rationale).
            upperlevel_wind_u_full = (
                (1.0 - _thermal_wind_relaxation) * upperlevel_wind_u_full
                + _thermal_wind_relaxation * _thermal_wind_target
            ).astype(np.float32, copy=False)
        # Section 17: the midlevel wind's reference/relaxation target now
        # averages the surface with the three-level path's own independent
        # upper wind, not the shared jet-stream kernel -- consistent with the
        # goal of the middle level no longer being pinned to a shared field
        # that the three-level path does not otherwise touch.
        _mid_wind_reference_u = 0.5 * (u_full + upperlevel_wind_u_full)
        _mid_wind_reference_v = 0.5 * (v_full + upperlevel_wind_v_full)
        if (
            midlevel_wind_u_full is None
            or midlevel_wind_v_full is None
            or midlevel_wind_u_full.shape != u_full.shape
            or midlevel_wind_v_full.shape != v_full.shape
        ):
            midlevel_wind_u_full = _mid_wind_reference_u
            midlevel_wind_v_full = _mid_wind_reference_v

        _mid_wind_temperature = midlevel_temperature_for_precip
        if (
            _mid_wind_temperature is None
            or _mid_wind_temperature.shape != u_full.shape
        ):
            _mid_wind_temperature = (
                state.air_temperature
                if state.air_temperature is not None
                and state.air_temperature.shape == u_full.shape
                else state.temperature
                if state.temperature is not None
                and state.temperature.shape == u_full.shape
                else _compute_T_base_ocean_full()
            )
        midlevel_wind_u_full, midlevel_wind_v_full = _evolve_middle_wind_substepped(
            midlevel_wind_u_full,
            midlevel_wind_v_full,
            temperature=_mid_wind_temperature,
            dt_days_total=days,
            substep_days=_wind_substep_days if _prognostic_gate_active else 0.0,
            pgf_temp_scale=float(wind_pgf_temp_scale),
            upper_pgf_amp=(
                float(pp.wind_upper_pgf_amp)
                * float(pp.three_level_mid_wind_pgf_fraction)
            ),
            damping_rate=float(pp.three_level_mid_wind_damping),
            vmax_clip=float(wind_vmax_clip),
            planet_params=pp,
            hadley_edge_deg=float(pp.wind_upper_hadley_edge_deg),
        )
        _mid_wind_relaxation = float(np.clip(
            pp.three_level_mid_wind_relaxation, 0.0, 1.0
        ))
        midlevel_wind_u_full = (
            (1.0 - _mid_wind_relaxation) * midlevel_wind_u_full
            + _mid_wind_relaxation * _mid_wind_reference_u
        ).astype(np.float32, copy=False)
        midlevel_wind_v_full = (
            (1.0 - _mid_wind_relaxation) * midlevel_wind_v_full
            + _mid_wind_relaxation * _mid_wind_reference_v
        ).astype(np.float32, copy=False)

        if _balanced_pressure_relaxation > 0.0:
            _mid_balanced_temperature = (
                midlevel_temperature_for_precip
                if midlevel_temperature_for_precip is not None
                and midlevel_temperature_for_precip.shape == midlevel_wind_u_full.shape
                else _compute_T_base_ocean_full()
            )
            _mid_balanced = _balanced_target(
                _mid_balanced_temperature,
                pp.native_balanced_mid_pressure_pa,
            )
            midlevel_wind_u_full = (
                (1.0 - _balanced_pressure_relaxation) * midlevel_wind_u_full
                + _balanced_pressure_relaxation * _mid_balanced.u
            ).astype(np.float32, copy=False)
            midlevel_wind_v_full = (
                (1.0 - _balanced_pressure_relaxation) * midlevel_wind_v_full
                + _balanced_pressure_relaxation * _mid_balanced.v
            ).astype(np.float32, copy=False)

        _overturning_speed = float(max(pp.native_balanced_overturning_speed_m_s, 0.0))
        if (
            bool(pp.enable_native_balanced_moist_static_energy_overturning)
            and midlevel_temperature_for_precip is not None
            and midlevel_temperature_for_precip.shape == u_full.shape
        ):
            _lower_for_mse = (
                state.air_temperature
                if state.air_temperature is not None
                and state.air_temperature.shape == u_full.shape
                else _compute_T_base_ocean_full()
            )
            _precip_for_mse = (
                state.precipitation
                if state.precipitation is not None
                and state.precipitation.shape == u_full.shape
                else None
            )
            _mse_radiative_target = (
                _compute_T_toa_equilibrium_full()
                if bool(pp.native_balanced_mse_use_toa_radiative_target)
                else _compute_T_base_ocean_full()
            )
            _mse_overturning = moist_static_energy_overturning_speed(
                _lower_for_mse,
                midlevel_temperature_for_precip,
                radiative_equilibrium_temperature_k=_mse_radiative_target,
                precipitation_mm_day=_precip_for_mse,
                radius_m=float(pp.radius_m),
                hadley_edge_deg=float(pp.wind_upper_hadley_edge_deg),
                layer_pressure_depth_pa=float(pp.two_layer_pressure_depth_pa),
                midlevel_height_m=float(pp.stability_condensation_reference_height_m),
                latent_relaxation_days=float(pp.two_layer_midlevel_relaxation_days),
                radiative_relaxation_days=float(pp.native_balanced_mse_radiative_relaxation_days),
                max_speed_m_s=float(pp.native_balanced_mse_overturning_max_speed_m_s),
                surface_pressure_pa=float(pp.surface_pressure_pa),
                gravity_m_s2=float(pp.surface_gravity),
                cp_dry_j_kg_k=float(pp.cp_dry),
            )
            _overturning_speed = _mse_overturning.speed_m_s
        elif (
            bool(pp.enable_native_balanced_diabatic_overturning)
            and midlevel_temperature_for_precip is not None
            and midlevel_temperature_for_precip.shape == u_full.shape
        ):
            _lower_for_diabatic = (
                state.air_temperature
                if state.air_temperature is not None
                and state.air_temperature.shape == u_full.shape
                else _compute_T_base_ocean_full()
            )
            _overturning_speed = diabatic_overturning_speed(
                _lower_for_diabatic,
                midlevel_temperature_for_precip,
                radius_m=float(pp.radius_m),
                hadley_edge_deg=float(pp.wind_upper_hadley_edge_deg),
                layer_pressure_depth_pa=float(pp.two_layer_pressure_depth_pa),
                midlevel_height_m=float(pp.stability_condensation_reference_height_m),
                relaxation_days=float(pp.two_layer_midlevel_relaxation_days),
                max_speed_m_s=float(pp.native_balanced_diabatic_overturning_max_speed_m_s),
            )
        if _overturning_speed > 0.0:
            _overturning_temperature = (
                state.air_temperature
                if state.air_temperature is not None
                and state.air_temperature.shape == u_full.shape
                else _compute_T_base_ocean_full()
            )
            _overturning = thermally_direct_overturning(
                _overturning_temperature,
                hadley_edge_deg=float(pp.wind_upper_hadley_edge_deg),
                lower_branch_speed_m_s=_overturning_speed,
            )
            v_full = (v_full + _overturning.lower_v).astype(np.float32, copy=False)
            midlevel_wind_v_full = (
                midlevel_wind_v_full + _overturning.middle_v
            ).astype(np.float32, copy=False)
            # Section 17: the overturning's upper branch feeds the independent
            # upper-level state, not the shared jet-stream kernel.
            upperlevel_wind_v_full = (
                upperlevel_wind_v_full + _overturning.upper_v
            ).astype(np.float32, copy=False)

        if bool(pp.enable_three_level_horizontal_mass_flux_closure):
            # Section 17: the mass-flux closure's "upper" level is now the
            # independent three-level upper wind, not the shared kernel --
            # both the divergence residual it measures and the correction it
            # applies are scoped to the decoupled state.
            _mass_closure = close_upper_mass_flux(
                u_full,
                v_full,
                midlevel_wind_u_full,
                midlevel_wind_v_full,
                upperlevel_wind_u_full,
                upperlevel_wind_v_full,
                radius_m=float(pp.radius_m),
                strength=float(np.clip(
                    pp.three_level_horizontal_mass_flux_strength, 0.0, 1.0,
                )),
                  max_speed_m_s=float(
                      pp.three_level_horizontal_mass_flux_max_speed_m_s,
                  ),
                  throughflow_max_speed_m_s=float(
                      pp.three_level_horizontal_mass_flux_throughflow_max_speed_m_s,
                  ),
              )
            upperlevel_wind_u_full = (upperlevel_wind_u_full + _mass_closure.upper_u_correction).astype(
                np.float32, copy=False,
            )
            upperlevel_wind_v_full = (upperlevel_wind_v_full + _mass_closure.upper_v_correction).astype(
                np.float32, copy=False,
            )

    # Winds to couple into temperature evolution operate on the temperature grid (Hc,Wc).
    if block_size > 1:
        _uv_c = _coarsen_many({"u": u_full, "v": v_full}, Hc, Wc, block_size)
        u_coarse = _uv_c["u"]
        v_coarse = _uv_c["v"]
    else:
        u_coarse = u_full
        v_coarse = v_full

    # Apply temperature evolution with advection and radiation.
    # Batched: these four fields share the same (Hc, Wc, block_size) coarsening and
    # are all independently optional, so one stacked pad+reshape+mean replaces four
    # separate `_coarsen` calls (see `_coarsen_many`).
    _group_c_in: dict[str, np.ndarray] = {}
    if state.humidity is not None:
        _group_c_in["humidity"] = state.humidity
    if state.soil_moisture is not None:
        _group_c_in["soil_moisture"] = state.soil_moisture
    if state.soil_moisture_deep is not None:
        _group_c_in["soil_moisture_deep"] = state.soil_moisture_deep
    if state.snow_depth is not None:
        _group_c_in["snow_depth"] = state.snow_depth
    if state.precipitation is not None:
        _group_c_in["precipitation"] = state.precipitation
    if state.vegetation_biomass is not None:
        _group_c_in["biomass"] = state.vegetation_biomass
    _group_c_out = _coarsen_many(_group_c_in, Hc, Wc, block_size)
    humidity_coarse = _group_c_out.get("humidity")
    soil_moisture_coarse = _group_c_out.get("soil_moisture")
    soil_moisture_deep_coarse = _group_c_out.get("soil_moisture_deep")
    snow_depth_coarse = _group_c_out.get("snow_depth")
    precipitation_coarse = _group_c_out.get("precipitation")
    biomass_coarse = _group_c_out.get("biomass")

    # Downsample biomes / Köppen to coarse resolution (center-of-block sample, not average).
    _mid = block_size // 2
    if biome_new is not None:
        _bs = block_size
        _bh, _bw = Hc * _bs - H, Wc * _bs - W
        _bp = biome_new.astype(np.int32)
        if _bh > 0 or _bw > 0:
            _bp = np.pad(_bp, ((0, _bh), (0, _bw)), mode="edge")
        biome_coarse: np.ndarray | None = _bp.reshape(Hc, _bs, Wc, _bs)[:, _mid, :, _mid]
    else:
        biome_coarse = None

    if koppen_new is not None:
        _kp = koppen_new.astype(np.int32)
        _bh, _bw = Hc * block_size - H, Wc * block_size - W
        if _bh > 0 or _bw > 0:
            _kp = np.pad(_kp, ((0, _bh), (0, _bw)), mode="edge")
        koppen_coarse: np.ndarray | None = _kp.reshape(Hc, block_size, Wc, block_size)[:, _mid, :, _mid]
    else:
        koppen_coarse = None

    # Albedo-effective Köppen: cells classified as EF (ice cap, code 19) but not yet
    # mature (ice_sheet_age < threshold) are physically treated as ET (tundra, code 18)
    # for albedo purposes.  This prevents newly-cold cells from jumping straight to the
    # 0.80 ice-sheet albedo before they have accumulated enough ice to warrant it.
    # The displayed/stored koppen_type is unchanged — only the albedo computation differs.
    if koppen_coarse is not None:
        _isa_coarse = _coarsen(ice_sheet_age_new, Hc, Wc, block_size)
        _immature_ef = (koppen_coarse == 19) & (_isa_coarse < ICE_SHEET_THRESHOLD_DAYS)
        koppen_phys_coarse = koppen_coarse.copy()
        koppen_phys_coarse[_immature_ef] = 18  # treat as ET (tundra) albedo
    else:
        koppen_phys_coarse = None

    # Coarsen new fields for _evolve_temperature (Features 1, 5, 6) — batched (see above).
    _group_c2_in: dict[str, np.ndarray] = {}
    if state.cloud_cover is not None:
        _group_c2_in["cloud_cover"] = state.cloud_cover
    if state.T_deep_ocean is not None:
        _group_c2_in["T_deep"] = state.T_deep_ocean
    if state.cloud_water is not None:
        _group_c2_in["cloud_water"] = state.cloud_water
    if _two_layer_thermo_active and state.atmospheric_condensate is not None:
        if bool(pp.enable_pressure_coordinate_moisture_closure):
            # The pressure-mass moisture closure persists cloud water in
            # kg m-2 (numerically mm), whereas the older radiative adapter
            # consumes a midlevel mixing ratio.  Convert at this boundary;
            # passing the mass value through as q would make even a few mm of
            # cloud water optically opaque everywhere.
            _midlevel_mass_kg_m2 = (
                0.35 * float(pp.surface_pressure_pa) / float(pp.surface_gravity)
            )
            _group_c2_in["midlevel_condensate"] = (
                np.asarray(state.atmospheric_condensate, dtype=np.float32)
                / _midlevel_mass_kg_m2
            )
        else:
            _group_c2_in["midlevel_condensate"] = state.atmospheric_condensate
    if state.land_deep_temperature is not None:
        _group_c2_in["land_deep"] = state.land_deep_temperature
    if (
        pp.enable_pressure_defined_radiative_temperature_profile
        and state.midlevel_temperature is not None
    ):
        _group_c2_in["radiative_midlevel"] = state.midlevel_temperature
    if (
        pp.enable_pressure_defined_radiative_temperature_profile
        and state.upperlevel_temperature is not None
    ):
        _group_c2_in["radiative_upperlevel"] = state.upperlevel_temperature
    if (
        pp.enable_pressure_defined_radiative_temperature_profile
        and state.grey_optical_depth is not None
    ):
        _group_c2_in["radiative_optical_depth"] = state.grey_optical_depth
    if (
        bool(pp.enable_force_restore_land)
        and bool(pp.enable_force_restore_boundary_layer)
        and state.boundary_layer_temperature is not None
    ):
        _group_c2_in["boundary_layer"] = state.boundary_layer_temperature
    if (
        bool(pp.enable_force_restore_land)
        and bool(pp.enable_force_restore_boundary_layer)
        and state.boundary_layer_interface_temperature is not None
    ):
        _group_c2_in["boundary_interface"] = (
            state.boundary_layer_interface_temperature
        )
    _group_c2_out = _coarsen_many(_group_c2_in, Hc, Wc, block_size)
    cloud_cover_coarse = _group_c2_out.get("cloud_cover")
    T_deep_coarse = _group_c2_out.get("T_deep")
    cloud_water_coarse = _group_c2_out.get("cloud_water")
    midlevel_condensate_coarse = _group_c2_out.get("midlevel_condensate")
    land_deep_coarse = _group_c2_out.get("land_deep")
    boundary_layer_coarse = _group_c2_out.get("boundary_layer")
    boundary_interface_coarse = _group_c2_out.get("boundary_interface")
    radiative_midlevel_coarse = _group_c2_out.get("radiative_midlevel")
    radiative_upperlevel_coarse = _group_c2_out.get("radiative_upperlevel")
    radiative_optical_depth_coarse = _group_c2_out.get("radiative_optical_depth")
    # ice_thick_prev_coarse already computed above

    # The shallow mixed layer responds on sub-day timescales.  Couple it to
    # force-restore at no more than six-hour intervals; the analytic internal
    # exchange is unconditionally stable, but the sensible flux and radiative
    # forcing are state dependent and must be resampled within a daily host
    # step.  This affects only the experimental mixed-layer branch.
    _temperature_substep_days = float(pp.temperature_substep_days)
    if bool(pp.enable_force_restore_land) and bool(pp.enable_force_restore_boundary_layer):
        if _temperature_substep_days <= 0.0 or _temperature_substep_days > 0.25:
            _temperature_substep_days = 0.25

    # Always track components for diagnostics (minimal overhead)
    T_sst_coarse, T_air_coarse_new, cloud_c, snow_c, temp_components, T_deep_coarse_new, cloud_water_coarse_new = _evolve_temperature_substepped(
        T_prev_coarse, T_base, state.elevation, Hc, Wc, block_size, H, W,
        day_of_year=new_day, days=days,
        substep_days=_temperature_substep_days,
        T_air_prev=T_air_coarse,
        wind_u=u_coarse, wind_v=v_coarse,
        T_base_land=T_base_land,
        ice_cover=ice_prev_coarse,
        thermal_diffusion=thermal_diffusion,
        ocean_transport_coeff=ocean_transport_coeff,
        ocean_exchange_coeff=ocean_exchange_coeff,
        ocean_exchange_inertia=ocean_exchange_inertia,
        epsilon_equator=eps_eq,
        epsilon_pole=eps_pole,
        ice_albedo_strength=ice_albedo_strength,
        humidity=humidity_coarse,
        soil_moisture=soil_moisture_coarse,
        soil_moisture_deep=soil_moisture_deep_coarse,
        track_components=track_components,
        precipitation=precipitation_coarse,
        vegetation_biomass=biomass_coarse,
        biome=biome_coarse,
        koppen_type=koppen_phys_coarse,  # albedo-effective: EF→ET for immature ice sheets
        planet_params=pp,
        elev_c=elev_c,
        snow_depth=snow_depth_coarse,
        # Pass the resolved flags (_fb), not the raw caller dict: _fb includes
        # planet-level auto-disables (e.g. ocean_transport/ice_albedo off for
        # worlds without a liquid ocean) that were previously dropped here.
        feedback_flags=_fb,
        total_days=new_total_days,
        prev_cloud_cover=cloud_cover_coarse,  # Feature 1: cloud persistence
        T_deep_ocean=T_deep_coarse,            # Feature 5: deep ocean layer
        ice_thickness=ice_thick_prev_coarse,   # Feature 6: thickness-dependent albedo
        prev_cloud_water=cloud_water_coarse,   # Feature: prognostic cloud water
        midlevel_condensate=midlevel_condensate_coarse,
        land_deep_temperature=land_deep_coarse,
        boundary_layer_temperature=boundary_layer_coarse,
        boundary_layer_interface_temperature=boundary_interface_coarse,
        radiative_midlevel_temperature=radiative_midlevel_coarse,
        radiative_upperlevel_temperature=radiative_upperlevel_coarse,
        radiative_optical_depth=radiative_optical_depth_coarse,
        temperature_bases_for_day=_temperature_bases_for_day,
    )
    T_coarse = T_sst_coarse  # alias: T_coarse continues to mean T_sst going forward
    
    # Upsample components to full resolution if needed
    if block_size > 1 and temp_components:
        temp_components_full = {}
        to_up = {k: v for k, v in temp_components.items() if isinstance(v, np.ndarray) and v.shape == (Hc, Wc)}
        if to_up:
            up = _upsample_bilinear_many(to_up, H, W, block_size)
            temp_components_full.update(up)
        for name, field in temp_components.items():
            if name not in temp_components_full:
                # Scalar or already full resolution
                temp_components_full[name] = field
        temp_components = temp_components_full

    land_deep_full = temp_components.pop("_land_deep_temperature", None)
    boundary_layer_full = temp_components.pop("_boundary_layer_temperature", None)
    boundary_interface_full = temp_components.pop(
        "_boundary_layer_interface_temperature", None
    )
    radiative_midlevel_full = temp_components.pop(
        "_radiative_midlevel_temperature", None
    )
    radiative_upperlevel_full = temp_components.pop(
        "_radiative_upperlevel_temperature", None
    )
    radiative_optical_depth_full = temp_components.pop(
        "_radiative_optical_depth", None
    )
    grey_midlevel_gain_full = temp_components.pop("_grey_midlevel_gain_w_m2", None)
    grey_upperlevel_gain_full = temp_components.pop("_grey_upperlevel_gain_w_m2", None)
    
    if block_size > 1:
        _up_fields: dict[str, np.ndarray] = {"T": T_coarse, "cloud": cloud_c, "T_air": T_air_coarse_new}
        if T_deep_coarse_new is not None:
            _up_fields["T_deep"] = T_deep_coarse_new
        if cloud_water_coarse_new is not None:
            _up_fields["cloud_water"] = cloud_water_coarse_new
        up = _upsample_bilinear_many(_up_fields, H, W, block_size)
        T_full, cloud_full, T_air_full = up["T"], up["cloud"], up["T_air"]
        T_deep_full: np.ndarray | None = up.get("T_deep")
        cloud_water_full: np.ndarray | None = up.get("cloud_water")
    else:
        T_full = T_coarse
        cloud_full = cloud_c
        T_air_full = T_air_coarse_new
        T_deep_full = T_deep_coarse_new
        cloud_water_full = cloud_water_coarse_new

    if land_deep_full is None:
        land_deep_full = state.land_deep_temperature
    if boundary_layer_full is None:
        boundary_layer_full = state.boundary_layer_temperature
    if boundary_interface_full is None:
        boundary_interface_full = state.boundary_layer_interface_temperature

    # Feature 5: initialize deep ocean on first step (SST - 15K, clamped to 271-285K).
    # Use the same value for land and ocean so coarsening never produces unphysical averages.
    # The physics exchange is gated by sea_mask inside _evolve_temperature, so land values
    # never feed back into T_sst.
    if T_deep_full is None and pp.has_liquid_water_ocean and T_full is not None:
        T_deep_full = np.clip(T_full - 15.0, 271.0, 285.0).astype(np.float32, copy=False)

    _shared_pressure_coordinate_active = (
        _three_level_midwind_active
        and bool(pp.enable_closed_three_level_thermodynamics)
        and bool(pp.enable_diabatic_interface_mass_flux)
        and bool(pp.enable_shared_pressure_coordinate_circulation)
        and float(days) > 0.0
    )
    _prognostic_overturning_heat_active = (
        _shared_pressure_coordinate_active
        and bool(pp.enable_pressure_coordinate_moisture_closure)
        and bool(pp.enable_prognostic_overturning_heat_reservoir)
    )
    _mse_constrained_pressure_circulation_active = (
        _prognostic_overturning_heat_active
        and bool(pp.enable_pressure_coordinate_mse_transport)
        and bool(pp.enable_mse_constrained_pressure_circulation)
    )
    _three_branch_mse_pressure_circulation_active = (
        _mse_constrained_pressure_circulation_active
        and bool(pp.enable_three_branch_mse_pressure_circulation)
    )
    _momentum_constrained_three_branch_mse_active = (
        _three_branch_mse_pressure_circulation_active
        and bool(pp.enable_momentum_constrained_three_branch_mse_circulation)
    )
    _prognostic_pressure_momentum_active = (
        _momentum_constrained_three_branch_mse_active
        and bool(pp.enable_prognostic_pressure_coordinate_momentum)
    )
    _joint_pressure_column_runtime_active = _prognostic_pressure_momentum_active
    _hydrostatic_sigma_runtime_active = (
        _joint_pressure_column_runtime_active
        and bool(pp.enable_hydrostatic_sigma_pressure_coordinate_transport)
    )
    if _shared_pressure_coordinate_active and not _joint_pressure_column_runtime_active:
        # One large-scale circulation now supplies the divergent mass carrier,
        # the pressure interfaces, and the meridional energy transport.  The
        # longitude-mean raw u field is retained only as a non-divergent jet
        # component; raw v is deliberately not carried forward because its
        # divergence was the source of the invalid 15--30 Pa/s interface flux.
        _shared_mid_temperature = (
            midlevel_temperature_for_precip
            if midlevel_temperature_for_precip is not None
            and midlevel_temperature_for_precip.shape == T_air_full.shape
            else T_air_full - 6.5e-3 * float(pp.stability_condensation_reference_height_m)
        )
        _shared_upper_temperature = (
            upperlevel_temperature_for_precip
            if upperlevel_temperature_for_precip is not None
            and upperlevel_temperature_for_precip.shape == T_air_full.shape
            else T_air_full - 6.5e-3 * float(pp.three_level_upper_height_m)
        )
        _shared_latent_heating = (
            state.pressure_moisture_condensation_mm_day
            if bool(pp.enable_pressure_coordinate_moisture_closure)
            and state.pressure_moisture_condensation_mm_day is not None
            else state.precipitation
        )
        _shared_heating = (
            np.zeros_like(T_air_full, dtype=np.float32)
            if _prognostic_overturning_heat_active
            and state.pressure_overturning_heating_w_m2 is None
            else state.pressure_overturning_heating_w_m2
            if _prognostic_overturning_heat_active
            else None
        )
        if _prognostic_pressure_momentum_active:
            _previous_omega_lower = (
                np.asarray(state.omega_lower_mid_pa_s, dtype=np.float64)
                if state.omega_lower_mid_pa_s is not None
                and state.omega_lower_mid_pa_s.shape == T_air_full.shape
                else np.zeros_like(T_air_full, dtype=np.float64)
            )
            _previous_omega_upper = (
                np.asarray(state.omega_mid_upper_pa_s, dtype=np.float64)
                if state.omega_mid_upper_pa_s is not None
                and state.omega_mid_upper_pa_s.shape == T_air_full.shape
                else np.zeros_like(T_air_full, dtype=np.float64)
            )
            _momentum_step = evolve_three_level_zonal_momentum(
                u_full, v_full, midlevel_wind_u_full, midlevel_wind_v_full,
                upperlevel_wind_u_full, upperlevel_wind_v_full,
                T_air_full, _shared_mid_temperature, _shared_upper_temperature,
                _previous_omega_lower, _previous_omega_upper,
                dt_seconds=float(days) * 86400.0, radius_m=float(pp.radius_m),
                sidereal_day_hours=float(pp.sidereal_day_hours),
                surface_pressure_pa=float(pp.surface_pressure_pa),
                gravity_m_s2=float(pp.surface_gravity),
                gas_constant_dry_j_kg_k=float(pp.gas_constant_dry),
            )
            u_full, v_full = _momentum_step.lower_u, _momentum_step.lower_v
            midlevel_wind_u_full, midlevel_wind_v_full = (
                _momentum_step.midlevel_u, _momentum_step.midlevel_v
            )
            upperlevel_wind_u_full, upperlevel_wind_v_full = (
                _momentum_step.upperlevel_u, _momentum_step.upperlevel_v
            )
        if _mse_constrained_pressure_circulation_active:
            _lower_q = (
                np.asarray(state.humidity, dtype=np.float64)
                if state.humidity is not None and state.humidity.shape == T_air_full.shape
                else np.full_like(T_air_full, 0.010, dtype=np.float64)
            )
            _mid_q = (
                np.asarray(state.midlevel_humidity, dtype=np.float64)
                if state.midlevel_humidity is not None and state.midlevel_humidity.shape == T_air_full.shape
                else 0.25 * _lower_q
            )
            _upper_q = (
                np.asarray(state.upperlevel_humidity, dtype=np.float64)
                if state.upperlevel_humidity is not None and state.upperlevel_humidity.shape == T_air_full.shape
                else 0.15 * _lower_q
            )
            _mse_circulation_kwargs = dict(
                dt_seconds=float(days) * 86400.0, radius_m=float(pp.radius_m),
                surface_pressure_pa=float(pp.surface_pressure_pa),
                lower_mid_pressure_depth_pa=float(pp.two_layer_pressure_depth_pa),
                mid_upper_pressure_depth_pa=float(pp.three_level_mid_upper_pressure_depth_pa),
                gravity_m_s2=float(pp.surface_gravity), cp_dry_j_kg_k=float(pp.cp_dry),
                layer_heights_m=(
                    0.0, float(pp.stability_condensation_reference_height_m),
                    float(pp.three_level_upper_height_m),
                ),
            )
            _mse_circulation_function = (
                momentum_constrained_three_branch_mse_pressure_coordinate_circulation
                if _momentum_constrained_three_branch_mse_active
                else three_branch_mse_constrained_pressure_coordinate_circulation
                if _three_branch_mse_pressure_circulation_active
                else mse_constrained_pressure_coordinate_circulation
            )
            _mse_circulation = _mse_circulation_function(
                _shared_heating, T_air_full, _shared_mid_temperature,
                _shared_upper_temperature, _lower_q, _mid_q, _upper_q,
                u_full, midlevel_wind_u_full, upperlevel_wind_u_full,
                **_mse_circulation_kwargs,
            )
            _shared_circulation = _mse_circulation.circulation
        else:
            _shared_circulation = shared_pressure_coordinate_circulation(
                _shared_latent_heating,
                T_air_full, _shared_mid_temperature, _shared_upper_temperature,
                u_full, midlevel_wind_u_full, upperlevel_wind_u_full,
                dt_seconds=float(days) * 86400.0,
                radius_m=float(pp.radius_m),
                surface_pressure_pa=float(pp.surface_pressure_pa),
                lower_mid_pressure_depth_pa=float(pp.two_layer_pressure_depth_pa),
                mid_upper_pressure_depth_pa=float(pp.three_level_mid_upper_pressure_depth_pa),
                gravity_m_s2=float(pp.surface_gravity),
                cp_dry_j_kg_k=float(pp.cp_dry), large_scale_heating_w_m2=_shared_heating,
            )
        u_full, v_full = _shared_circulation.lower_u, _shared_circulation.lower_v
        midlevel_wind_u_full, midlevel_wind_v_full = (
            _shared_circulation.midlevel_u, _shared_circulation.midlevel_v
        )
        upperlevel_wind_u_full, upperlevel_wind_v_full = (
            _shared_circulation.upperlevel_u, _shared_circulation.upperlevel_v
        )
        if _precipitation_diagnostics is not None:
            _shared_interface = _shared_circulation.interface_mass_flux
            _precipitation_diagnostics["shared_pressure_circulation"] = True
            _precipitation_diagnostics["shared_pressure_lower_mid_courant_max"] = float(
                _shared_interface.lower_mid_vertical_courant_max
            )
            _precipitation_diagnostics["shared_pressure_mid_upper_courant_max"] = float(
                _shared_interface.mid_upper_vertical_courant_max
            )
            if _mse_constrained_pressure_circulation_active:
                _precipitation_diagnostics["mse_constrained_pressure_circulation"] = True
                _precipitation_diagnostics["mse_constrained_energy_closure_residual_w"] = float(
                    _mse_circulation.energy_closure_residual_w
                )
                _precipitation_diagnostics["mse_constrained_peak_transport_w_m"] = float(
                    np.max(np.abs(_mse_circulation.meridional_mse_transport_w_m))
                )
                if _three_branch_mse_pressure_circulation_active:
                    _precipitation_diagnostics["three_branch_mse_pressure_circulation"] = True
                    _precipitation_diagnostics["three_branch_mse_midlevel_deposition_w_m2"] = (
                        _mse_circulation.midlevel_diabatic_deposition_w_m2
                    )
                if _momentum_constrained_three_branch_mse_active:
                    _precipitation_diagnostics["momentum_constrained_three_branch_mse_circulation"] = True
                if _prognostic_pressure_momentum_active:
                    _precipitation_diagnostics["prognostic_pressure_coordinate_momentum"] = True
                    _precipitation_diagnostics["pressure_momentum_vertical_courant_max"] = float(
                        _momentum_step.vertical_courant_max
                    )

    # Section 17: the "upper" wind fed into precipitation's pressure-column
    # physics (upper-reservoir vapor transport, the mid-upper interface omega)
    # is the three-level path's own independent state when that path is
    # active, so its column physics stops building on the shared jet-stream
    # kernel too -- not only the tendency terms in the block above. When the
    # three-level gate is off (including the plain two-layer-only path, which
    # has no independent upper state), this is exactly u2_full/v2_full,
    # unchanged from before this change.
    _precip_wind_u_aloft = upperlevel_wind_u_full if _three_level_midwind_active else u2_full
    _precip_wind_v_aloft = upperlevel_wind_v_full if _three_level_midwind_active else v2_full

    if state.elevation is not None:
        # Run precipitation at half resolution (block_size=2) for large grids where
        # the half-resolution cell size (~0.7°) still resolves the subtropical dry belt
        # adequately. For small grids (H < 256) the subtropical band spans too few rows
        # at half resolution, so full-resolution precipitation is used instead.
        _pbs = 1 if _hydrostatic_sigma_runtime_active else (
            (2 if H >= 256 else 1) if precip_block_size is None else precip_block_size
        )
        _Hcp = max(1, H // _pbs)
        _Wcp = max(1, W // _pbs)
        if _pbs > 1 and H >= 4 and W >= 4:
            _elev_p = _coarsen_elevation_cached(state.elevation, _Hcp, _Wcp, _pbs)
            # Batched (see `_coarsen_many`): T/u/v are unconditional, humidity/soil/cloud
            # are independently optional, all sharing the same (_Hcp, _Wcp, _pbs) grid.
            _group_p_in: dict[str, np.ndarray] = {
                "T": T_full, "u": u_full, "v": v_full,
                "u2": _precip_wind_u_aloft, "v2": _precip_wind_v_aloft,
                "T_air": T_air_full,
            }
            if _three_level_midwind_active:
                _group_p_in["u_mid"] = midlevel_wind_u_full
                _group_p_in["v_mid"] = midlevel_wind_v_full
            if state.humidity is not None:
                _group_p_in["hum"] = state.humidity
            if state.soil_moisture is not None:
                _group_p_in["soil"] = state.soil_moisture
            if state.soil_moisture_deep is not None:
                _group_p_in["soil_deep"] = state.soil_moisture_deep
            if cloud_full is not None:
                _group_p_in["cloud"] = cloud_full
            if state.atmospheric_condensate is not None:
                _group_p_in["condensate"] = state.atmospheric_condensate
            if state.precipitating_hydrometeors is not None:
                _group_p_in["hydrometeors"] = state.precipitating_hydrometeors
            if state.precipitation is not None:
                _group_p_in["previous_precipitation"] = (
                    state.pressure_moisture_condensation_mm_day
                    if bool(pp.enable_pressure_coordinate_moisture_closure)
                    and state.pressure_moisture_condensation_mm_day is not None
                    else state.precipitation
                )
            if _prognostic_overturning_heat_active:
                _group_p_in["previous_large_scale_heating"] = (
                    np.zeros_like(T_full, dtype=np.float32)
                    if state.pressure_overturning_heating_w_m2 is None
                    else state.pressure_overturning_heating_w_m2
                )
            if midlevel_temperature_for_precip is not None:
                _group_p_in["midlevel_temperature"] = midlevel_temperature_for_precip
            if state.midlevel_humidity is not None:
                _group_p_in["midlevel_humidity"] = state.midlevel_humidity
            if upperlevel_temperature_for_precip is not None:
                _group_p_in["upperlevel_temperature"] = upperlevel_temperature_for_precip
            if grey_midlevel_gain_full is not None:
                _group_p_in["grey_midlevel_gain"] = grey_midlevel_gain_full
            if grey_upperlevel_gain_full is not None:
                _group_p_in["grey_upperlevel_gain"] = grey_upperlevel_gain_full
            if state.upperlevel_humidity is not None:
                _group_p_in["upperlevel_humidity"] = state.upperlevel_humidity
            _group_p_out = _coarsen_many(_group_p_in, _Hcp, _Wcp, _pbs)
            _T_p = _group_p_out["T"]
            _u_p = _group_p_out["u"]
            _v_p = _group_p_out["v"]
            _u2_p = _group_p_out["u2"]
            _v2_p = _group_p_out["v2"]
            _T_air_p = _group_p_out["T_air"]
            _umid_p = _group_p_out.get("u_mid")
            _vmid_p = _group_p_out.get("v_mid")
            _hum_p = _group_p_out.get("hum")
            _soil_p = _group_p_out.get("soil")
            _soil_deep_p = _group_p_out.get("soil_deep")
            _cloud_p = _group_p_out.get("cloud")
            _condensate_p = _group_p_out.get("condensate")
            _hydrometeors_p = _group_p_out.get("hydrometeors")
            _previous_precipitation_p = _group_p_out.get("previous_precipitation")
            _previous_large_scale_heating_p = _group_p_out.get("previous_large_scale_heating")
            _midlevel_temperature_p = _group_p_out.get("midlevel_temperature")
            _midlevel_humidity_p = _group_p_out.get("midlevel_humidity")
            _upperlevel_temperature_p = _group_p_out.get("upperlevel_temperature")
            _upperlevel_humidity_p = _group_p_out.get("upperlevel_humidity")
            _grey_midlevel_gain_p = _group_p_out.get("grey_midlevel_gain")
            _grey_upperlevel_gain_p = _group_p_out.get("grey_upperlevel_gain")
            P_p, hum_p_next, soil_p_next, soil_deep_p_next, condensate_p_next, midlevel_temperature_p_next, midlevel_humidity_p_next, upperlevel_temperature_p_next, upperlevel_humidity_p_next, hydrometeors_p_next = _generate_precipitation_substepped(
                _Hcp, _Wcp, _elev_p,
                temperature=_T_p, wind_u=_u_p, wind_v=_v_p,
                wind_u_aloft=_u2_p, wind_v_aloft=_v2_p,
                wind_u_midlevel=_umid_p, wind_v_midlevel=_vmid_p,
                humidity=_hum_p, soil_moisture=_soil_p, soil_moisture_deep=_soil_deep_p,
                condensate=_condensate_p,
                precipitating_hydrometeors=_hydrometeors_p,
                midlevel_temperature=_midlevel_temperature_p,
                midlevel_humidity=_midlevel_humidity_p,
                upperlevel_temperature=_upperlevel_temperature_p,
                upperlevel_humidity=_upperlevel_humidity_p,
                midlevel_radiative_flux_w_m2=_grey_midlevel_gain_p,
                upperlevel_radiative_flux_w_m2=_grey_upperlevel_gain_p,
                column_lower_temperature=_T_air_p,
                previous_precipitation_mm_day=_previous_precipitation_p,
                previous_large_scale_heating_w_m2=_previous_large_scale_heating_p,
                cloud_fraction=_cloud_p,
                day_of_year=new_day, dt_days=float(days),
                surface_pressure_hpa=pp.surface_pressure_pa / 100.0,
                planet_params=pp,
                debug_fields=_precipitation_diagnostics,
            )
            _up = _upsample_bilinear_many(
                {"P": P_p, "q": hum_p_next, "soil": soil_p_next, "soil_deep": soil_deep_p_next, "condensate": condensate_p_next, "hydrometeors": hydrometeors_p_next, "midlevel_temperature": midlevel_temperature_p_next, "midlevel_humidity": midlevel_humidity_p_next, "upperlevel_temperature": upperlevel_temperature_p_next, "upperlevel_humidity": upperlevel_humidity_p_next}, H, W, _pbs
            )
            P_full: np.ndarray | None = _up["P"]
            humidity_next: np.ndarray | None = _up["q"]
            soil_next: np.ndarray | None = _up["soil"]
            soil_deep_next: np.ndarray | None = _up["soil_deep"]
            condensate_next: np.ndarray | None = _up["condensate"]
            hydrometeors_next: np.ndarray | None = _up["hydrometeors"]
            midlevel_temperature_next: np.ndarray | None = _up["midlevel_temperature"]
            midlevel_humidity_next: np.ndarray | None = _up["midlevel_humidity"]
            upperlevel_temperature_next: np.ndarray | None = _up["upperlevel_temperature"]
            upperlevel_humidity_next: np.ndarray | None = _up["upperlevel_humidity"]
        else:
            P_full, humidity_next, soil_next, soil_deep_next, condensate_next, midlevel_temperature_next, midlevel_humidity_next, upperlevel_temperature_next, upperlevel_humidity_next, hydrometeors_next = _generate_precipitation_substepped(
                H, W, state.elevation,
                temperature=T_full, wind_u=u_full, wind_v=v_full,
                wind_u_aloft=_precip_wind_u_aloft, wind_v_aloft=_precip_wind_v_aloft,
                wind_u_midlevel=(
                    midlevel_wind_u_full if _three_level_midwind_active else None
                ),
                wind_v_midlevel=(
                    midlevel_wind_v_full if _three_level_midwind_active else None
                ),
                humidity=state.humidity, soil_moisture=state.soil_moisture,
                soil_moisture_deep=state.soil_moisture_deep,
                condensate=state.atmospheric_condensate,
                precipitating_hydrometeors=state.precipitating_hydrometeors,
                lower_pressure_depth_pa=state.lower_pressure_depth_pa,
                midlevel_pressure_depth_pa=state.midlevel_pressure_depth_pa,
                upperlevel_pressure_depth_pa=state.upperlevel_pressure_depth_pa,
                lower_pressure_cloud_condensate=state.lower_pressure_cloud_condensate,
                midlevel_pressure_cloud_condensate=state.midlevel_pressure_cloud_condensate,
                upperlevel_pressure_cloud_condensate=state.upperlevel_pressure_cloud_condensate,
                lower_pressure_hydrometeors=state.lower_pressure_hydrometeors,
                midlevel_pressure_hydrometeors=state.midlevel_pressure_hydrometeors,
                upperlevel_pressure_hydrometeors=state.upperlevel_pressure_hydrometeors,
                midlevel_temperature=midlevel_temperature_for_precip,
                midlevel_humidity=state.midlevel_humidity,
                upperlevel_temperature=upperlevel_temperature_for_precip,
                upperlevel_humidity=state.upperlevel_humidity,
                midlevel_radiative_flux_w_m2=grey_midlevel_gain_full,
                upperlevel_radiative_flux_w_m2=grey_upperlevel_gain_full,
                column_lower_temperature=T_air_full,
                previous_precipitation_mm_day=(
                    state.pressure_moisture_condensation_mm_day
                    if bool(pp.enable_pressure_coordinate_moisture_closure)
                    and state.pressure_moisture_condensation_mm_day is not None
                    else state.precipitation
                ),
                previous_large_scale_heating_w_m2=(
                    np.zeros_like(T_air_full, dtype=np.float32)
                    if _prognostic_overturning_heat_active
                    and state.pressure_overturning_heating_w_m2 is None
                    else state.pressure_overturning_heating_w_m2
                    if _prognostic_overturning_heat_active
                    else None
                ),
                cloud_fraction=cloud_full,
                day_of_year=new_day, dt_days=float(days),
                surface_pressure_hpa=pp.surface_pressure_pa / 100.0,
                planet_params=pp,
                debug_fields=_precipitation_diagnostics,
            )
    else:
        P_full = None
        humidity_next = None
        soil_next = None
        soil_deep_next = None
        condensate_next = None
        midlevel_temperature_next = None
        midlevel_humidity_next = None
        upperlevel_temperature_next = None
        upperlevel_humidity_next = None
        hydrometeors_next = None

    omega_lower_mid_full = (
        None if _precipitation_diagnostics is None
        else _precipitation_diagnostics.get("midlevel_omega_pa_s")
    )
    omega_mid_upper_full = (
        None if _precipitation_diagnostics is None
        else _precipitation_diagnostics.get("upperlevel_omega_pa_s")
    )
    _sigma_state_next = {
        "lower_pressure_depth_pa": state.lower_pressure_depth_pa,
        "midlevel_pressure_depth_pa": state.midlevel_pressure_depth_pa,
        "upperlevel_pressure_depth_pa": state.upperlevel_pressure_depth_pa,
        "lower_pressure_cloud_condensate": state.lower_pressure_cloud_condensate,
        "midlevel_pressure_cloud_condensate": state.midlevel_pressure_cloud_condensate,
        "upperlevel_pressure_cloud_condensate": state.upperlevel_pressure_cloud_condensate,
        "lower_pressure_hydrometeors": state.lower_pressure_hydrometeors,
        "midlevel_pressure_hydrometeors": state.midlevel_pressure_hydrometeors,
        "upperlevel_pressure_hydrometeors": state.upperlevel_pressure_hydrometeors,
    }
    closed_lower_temperature_full = (
        None if _precipitation_diagnostics is None
        else _precipitation_diagnostics.get("closed_column_lower_temperature_k")
    )
    if _hydrostatic_sigma_runtime_active:
        if _precipitation_diagnostics is None or not _precipitation_diagnostics.get("hydrostatic_sigma_runtime"):
            raise RuntimeError("hydrostatic sigma gate did not receive its atomic precipitation transition")
        _sigma_state_next.update({
            name: _precipitation_diagnostics[key]
            for name, key in {
                "lower_pressure_depth_pa": "hydrostatic_sigma_lower_pressure_depth_pa",
                "midlevel_pressure_depth_pa": "hydrostatic_sigma_midlevel_pressure_depth_pa",
                "upperlevel_pressure_depth_pa": "hydrostatic_sigma_upperlevel_pressure_depth_pa",
                "lower_pressure_cloud_condensate": "hydrostatic_sigma_lower_cloud",
                "midlevel_pressure_cloud_condensate": "hydrostatic_sigma_midlevel_cloud",
                "upperlevel_pressure_cloud_condensate": "hydrostatic_sigma_upperlevel_cloud",
                "lower_pressure_hydrometeors": "hydrostatic_sigma_lower_hydrometeors",
                "midlevel_pressure_hydrometeors": "hydrostatic_sigma_midlevel_hydrometeors",
                "upperlevel_pressure_hydrometeors": "hydrostatic_sigma_upperlevel_hydrometeors",
            }.items()
        })
        closed_lower_temperature_full = _precipitation_diagnostics["hydrostatic_sigma_lower_temperature"]
        midlevel_temperature_next = _precipitation_diagnostics["hydrostatic_sigma_midlevel_temperature"]
        upperlevel_temperature_next = _precipitation_diagnostics["hydrostatic_sigma_upperlevel_temperature"]
    _joint_runtime_winds = {
        name: None if _precipitation_diagnostics is None else _precipitation_diagnostics.get(key)
        for name, key in {
            "lower_u": "joint_pressure_column_lower_wind_u",
            "lower_v": "joint_pressure_column_lower_wind_v",
            "mid_u": "joint_pressure_column_midlevel_wind_u",
            "mid_v": "joint_pressure_column_midlevel_wind_v",
            "upper_u": "joint_pressure_column_upperlevel_wind_u",
            "upper_v": "joint_pressure_column_upperlevel_wind_v",
        }.items()
    }
    if _hydrostatic_sigma_runtime_active:
        _joint_runtime_winds = {
            name: _precipitation_diagnostics[key]
            for name, key in {
                "lower_u": "hydrostatic_sigma_lower_wind_u",
                "lower_v": "hydrostatic_sigma_lower_wind_v",
                "mid_u": "hydrostatic_sigma_midlevel_wind_u",
                "mid_v": "hydrostatic_sigma_midlevel_wind_v",
                "upper_u": "hydrostatic_sigma_upperlevel_wind_u",
                "upper_v": "hydrostatic_sigma_upperlevel_wind_v",
            }.items()
        }
    if state.elevation is not None and _pbs > 1:
        _omega_fields = {
            name: np.asarray(value)
            for name, value in {
                "lower": omega_lower_mid_full,
                "upper": omega_mid_upper_full,
                "closed_lower_temperature": closed_lower_temperature_full,
            }.items()
            if value is not None and np.asarray(value).shape == (_Hcp, _Wcp)
        }
        if _omega_fields:
            _omega_full = _upsample_bilinear_many(_omega_fields, H, W, _pbs)
            omega_lower_mid_full = _omega_full.get("lower", omega_lower_mid_full)
            omega_mid_upper_full = _omega_full.get("upper", omega_mid_upper_full)
            closed_lower_temperature_full = _omega_full.get(
                "closed_lower_temperature", closed_lower_temperature_full
            )
        _joint_coarse_winds = {
            name: np.asarray(value)
            for name, value in _joint_runtime_winds.items()
            if value is not None and np.asarray(value).shape == (_Hcp, _Wcp)
        }
        if _joint_coarse_winds:
            _joint_runtime_winds.update(
                _upsample_bilinear_many(_joint_coarse_winds, H, W, _pbs)
            )
    if not _three_level_midwind_active:
        omega_lower_mid_full = None
        omega_mid_upper_full = None
        closed_lower_temperature_full = None
    if (
        bool(pp.enable_closed_three_level_thermodynamics)
        and closed_lower_temperature_full is not None
        and np.asarray(closed_lower_temperature_full).shape == T_air_full.shape
    ):
        # `_evolve_temperature` is the resolved radiative/surface split step;
        # the pressure-column operator applies its conservative vertical and
        # phase adjustment afterwards. No radiative tendency is applied twice.
        T_air_full = np.asarray(closed_lower_temperature_full, dtype=np.float32)
    if _joint_pressure_column_runtime_active and all(
        value is not None and np.asarray(value).shape == T_air_full.shape
        for value in _joint_runtime_winds.values()
    ):
        u_full, v_full = _joint_runtime_winds["lower_u"], _joint_runtime_winds["lower_v"]
        midlevel_wind_u_full, midlevel_wind_v_full = (
            _joint_runtime_winds["mid_u"], _joint_runtime_winds["mid_v"]
        )
        upperlevel_wind_u_full, upperlevel_wind_v_full = (
            _joint_runtime_winds["upper_u"], _joint_runtime_winds["upper_v"]
        )

    # Preserve the historical state representation exactly while the new
    # closure is gated off; a zero array is not equivalent to an absent
    # reservoir for persistence/golden-state comparisons.
    if not pp.enable_prognostic_condensate:
        condensate_next = state.atmospheric_condensate
        hydrometeors_next = state.precipitating_hydrometeors
    if not pp.enable_two_layer_convective_adjustment:
        midlevel_temperature_next = state.midlevel_temperature
        midlevel_humidity_next = state.midlevel_humidity
    if not pp.enable_three_level_pressure_column:
        upperlevel_temperature_next = state.upperlevel_temperature
        upperlevel_humidity_next = state.upperlevel_humidity
        midlevel_wind_u_full = state.midlevel_wind_u
        midlevel_wind_v_full = state.midlevel_wind_v
        upperlevel_wind_u_full = state.upperlevel_wind_u
        upperlevel_wind_v_full = state.upperlevel_wind_v

    # This pressure-coordinate temperature state is independent of the
    # experimental moisture/convection column. A prognostic thermodynamic
    # layer remains its owner when active; otherwise the layer follows a dry
    # adiabat from resolved free air. Radiation can therefore consume a real
    # mid/upper profile without a fitted height lapse or near-surface proxy.
    if pp.enable_pressure_defined_radiative_temperature_profile:
        if pp.enable_coupled_two_layer_grey_radiation:
            if midlevel_temperature_next is None or upperlevel_temperature_next is None:
                raise RuntimeError("closed grey coupling did not return layer state")
        else:
            _radiative_profile = pressure_defined_temperature_profile(
                T_air_full,
                float(pp.surface_pressure_pa),
                float(pp.two_layer_pressure_depth_pa),
                float(pp.three_level_mid_upper_pressure_depth_pa),
                gas_constant_dry_air_j_kg_k=float(pp.gas_constant_dry),
                cp_dry_air_j_kg_k=float(pp.cp_dry),
            )
            if midlevel_temperature_next is None or not pp.enable_two_layer_convective_adjustment:
                midlevel_temperature_next = _radiative_profile.midlevel_temperature_k.astype(
                    np.float32, copy=False
                )
            if upperlevel_temperature_next is None or not pp.enable_three_level_pressure_column:
                upperlevel_temperature_next = _radiative_profile.upperlevel_temperature_k.astype(
                    np.float32, copy=False
                )

    pressure_overturning_heating_next = state.pressure_overturning_heating_w_m2
    pressure_coordinate_heat_convergence_next = state.pressure_coordinate_heat_convergence_w_m2
    if _hydrostatic_sigma_runtime_active:
        pressure_coordinate_heat_convergence_next = _precipitation_diagnostics.get(
            "hydrostatic_sigma_heat_convergence_w_m2"
        )
        if pressure_coordinate_heat_convergence_next is None:
            raise RuntimeError("hydrostatic sigma gate did not emit its MSE heat-convergence diagnostic")
    if _prognostic_overturning_heat_active and not _hydrostatic_sigma_runtime_active:
        _simultaneous_heating = (
            None if _precipitation_diagnostics is None
            else _precipitation_diagnostics.get("simultaneous_pressure_column_heating_w_m2")
        )
        if _simultaneous_heating is not None:
            _simultaneous_heating = np.asarray(_simultaneous_heating, dtype=np.float32)
            if _simultaneous_heating.shape != T_air_full.shape:
                if _simultaneous_heating.shape == (_Hcp, _Wcp) and _pbs > 1:
                    _simultaneous_heating = _upsample_bilinear_many(
                        {"heating": _simultaneous_heating}, H, W, _pbs
                    )["heating"]
                else:
                    raise ValueError("simultaneous pressure-column heating has an unexpected grid")
            # The nested simultaneous adapter already solved this heating
            # state with its current phase and reservoir transition.  Running
            # the historical next-step reservoir update here would make it a
            # second, lagged owner of the same latent heating.
            pressure_overturning_heating_next = _simultaneous_heating
            if _precipitation_diagnostics is not None:
                _precipitation_diagnostics["prognostic_overturning_heating_w_m2"] = (
                    pressure_overturning_heating_next
                )
        else:
            _condensation_mm = (
                None if _precipitation_diagnostics is None
                else _precipitation_diagnostics.get("pressure_moisture_cloud_created_mm")
            )
            if _condensation_mm is None:
                raise RuntimeError("prognostic overturning heat requires pressure-mass condensation diagnostics")
            _condensation_mm = np.asarray(_condensation_mm, dtype=np.float32)
            if _condensation_mm.shape != T_air_full.shape:
                if _condensation_mm.shape == (_Hcp, _Wcp) and _pbs > 1:
                    _condensation_mm = _upsample_bilinear_many(
                        {"condensation": _condensation_mm}, H, W, _pbs
                    )["condensation"]
                else:
                    raise ValueError("pressure-mass condensation diagnostic has an unexpected grid")
            if midlevel_temperature_next is None or upperlevel_temperature_next is None:
                raise RuntimeError("prognostic overturning heat requires all thermodynamic layers")
            _heating_step = evolve_large_scale_heating_reservoir(
                state.pressure_overturning_heating_w_m2,
                _condensation_mm / float(days), T_air_full,
                midlevel_temperature_next, upperlevel_temperature_next,
                dt_seconds=float(days) * 86400.0,
                surface_pressure_pa=float(pp.surface_pressure_pa),
                gravity_m_s2=float(pp.surface_gravity), cp_dry_j_kg_k=float(pp.cp_dry),
            )
            pressure_overturning_heating_next = _heating_step.heating_w_m2
            if _precipitation_diagnostics is not None:
                _precipitation_diagnostics["prognostic_overturning_heating_w_m2"] = (
                    pressure_overturning_heating_next
                )
                _precipitation_diagnostics["prognostic_overturning_adjustment_time_days"] = (
                    _heating_step.radiative_adjustment_time_s / 86400.0
                )

    if pp.enable_surface_hydrology and P_full is not None and soil_next is not None:
        from hydrology import route_surface_water

        _, _hydro_land = get_masks(state.elevation)
        _threshold = float(np.clip(pp.runoff_soil_threshold, 0.0, 0.999))
        # Trigger on soil_deep, not the fast surface bucket: the surface layer
        # sits chronically pinned near its 0.05 floor across nearly all real
        # terrain (see the high-latitude soil-desiccation fix), so a
        # surface-only threshold never crosses 0.75 -- measured directly on a
        # 10yr real-terrain continuation, this left runoff/river_discharge/
        # surface_water_mm bit-exactly zero everywhere. soil_deep (post
        # soil_deep_gain_rate fix) has the real spatial spread instead
        # (p50~0.12, p75~0.33, p90~0.78 on that same continuation), so it's
        # the field that can actually distinguish "wet enough for rivers"
        # from "not." Falls back to soil_next if soil_deep_next is
        # unavailable (e.g. soil_deep_gain_rate=0 and dry-planet/no-ocean
        # configs that never populate it) so this doesn't silently disable
        # runoff entirely in that configuration.
        _runoff_wetness = soil_deep_next if soil_deep_next is not None else soil_next
        _saturation_excess = np.clip(
            (_runoff_wetness - _threshold) / max(1.0 - _threshold, 1e-6),
            0.0,
            1.0,
        )
        _runoff_mm_day = (
            np.maximum(P_full, 0.0)
            * _saturation_excess
            * float(np.clip(pp.runoff_fraction, 0.0, 1.0))
            * _hydro_land.astype(np.float32)
        )
        (
            surface_water_mm_new,
            river_discharge_new,
            runoff_to_ocean_new,
            _ocean_river_input_mm_day,
        ) = route_surface_water(
            state.elevation,
            _runoff_mm_day,
            state.surface_water_mm,
            dt_days=float(days),
            routing_passes=max(0, int(pp.river_routing_passes)),
            routing_fraction=float(np.clip(pp.river_routing_fraction, 0.0, 1.0)),
        )
        # Open-water evaporation from standing surface water -- without this,
        # closed/flat-terrain basins have no sink at all and grow unboundedly
        # (see PlanetParams.lake_evap_mm_day docstring for the measured
        # runaway this fixes). Applied post-routing so it acts on the
        # storage actually left behind this step, not double-counted against
        # what already flowed onward.
        if T_full is not None:
            _lake_evap_factor = np.clip((T_full - 273.15) / 20.0, 0.1, 2.0).astype(np.float32, copy=False)
            _lake_evap_mm = float(pp.lake_evap_mm_day) * _lake_evap_factor * float(days)
            surface_water_mm_new = np.maximum(surface_water_mm_new - _lake_evap_mm, 0.0).astype(np.float32, copy=False)
        # Hard safety backstop -- see PlanetParams.surface_water_cap_mm docstring
        # for why evaporation alone cannot bound this for continent-scale basins.
        surface_water_mm_new = np.minimum(
            surface_water_mm_new, float(pp.surface_water_cap_mm)
        ).astype(np.float32, copy=False)
    else:
        surface_water_mm_new = state.surface_water_mm
        river_discharge_new = state.river_discharge_mm_day
        runoff_to_ocean_new = state.runoff_to_ocean_mm_day
        # Not persisted state -- recomputed fresh each step from route_surface_water
        # above, consumed immediately by evolve_salinity below. None when hydrology
        # is disabled (evolve_salinity treats that as no river freshwater input).
        _ocean_river_input_mm_day = None

    # NOTE: latent cooling from precipitation is already applied inside
    # _evolve_temperature (via evaporation) and generate_precipitation.
    # Applying it again here was a double-count and has been removed.
    if T_full is not None:
        T_full = np.clip(T_full, 150.0, 330.0)
    if T_full is not None and pp.has_liquid_water_ocean:
        ice_full, delta_ice, ice_thick_full = update_sea_ice(
            T_full, state.elevation, state.ice_cover, days,
            _ice_thick_src,  # prev_thickness (already initialized above)
            freeze_temp=ice_freeze_temp,
            melt_temp=ice_melt_temp,
            freeze_rate=ice_freeze_rate,
            melt_rate=ice_melt_rate,
        )
        # Ice-ocean latent heat feedback: freezing releases heat, melting absorbs heat
        # L_f=334 kJ/kg, rho_ice=917 kg/m³, ~1m effective thickness, ~100m mixed layer
        # gives ~3K per unit ice fraction change
        if _fb.get('ice_albedo', True):
            latent_scale = 3.0  # K per unit ice fraction change
            is_ocean_full, _ = get_masks(state.elevation)
            T_full = T_full + delta_ice * latent_scale * is_ocean_full.astype(np.float32)
    else:
        ice_full = state.ice_cover  # preserve existing (e.g. dry planet polar CO2 ice)
        delta_ice = np.zeros((H, W), dtype=np.float32)
        ice_thick_full = _ice_thick_src  # no thickness evolution on dry planets

    # Feature 3: salinity evolution (after sea-ice so delta_ice is available)
    from ocean import evolve_salinity  # imported here to avoid circular-import risk
    if pp.has_liquid_water_ocean:
        _sal_prev = state.salinity
        if _sal_prev is None:
            # First-step initialization: uniform ocean salinity
            _sea_sal, _ = get_masks(state.elevation)
            _sal_prev = np.where(_sea_sal, pp.salinity_reference_psu, 0.0).astype(np.float32, copy=False)
        salinity_new = evolve_salinity(
            _sal_prev, T_full, state.elevation,
            P_full, delta_ice, dt_days=float(days), pp=pp,
            river_input_mm_day=_ocean_river_input_mm_day,
        )
    else:
        salinity_new: np.ndarray | None = state.salinity

    # Snow depth evolution (degree-day accumulation / melt model)
    # Only over land; ocean has sea ice instead of snow pack.
    # Physics:
    #   Accumulation  — fraction of precipitation that falls as snow (T-dependent)
    #   Melt          — degree-day factor: 3 mm SWE per °C per day above freezing
    #   Sublimation   — 0.1 % of current pack per day (slow but persistent)
    #   Cap at 10 m SWE (realistic maximum for land ice / deep snow pack)
    _T_air_for_snow = T_air_full if T_air_full is not None else T_full
    _snow_prev = state.snow_depth if state.snow_depth is not None else np.zeros((H, W), dtype=np.float32)
    if P_full is not None and _T_air_for_snow is not None:
        _, _land_snow = get_masks(state.elevation)
        _T_air_c = _T_air_for_snow - 273.15  # °C
        # Snow fraction: 1 at ≤−3°C, 0 at ≥+2°C (linear ramp through the mixed-phase zone)
        _snow_frac = np.clip((-_T_air_c + 2.0) / 5.0, 0.0, 1.0).astype(np.float32, copy=False)
        # Snowfall in m SWE/day (P_full is mm/day liquid-water equiv; 1 mm = 0.001 m SWE)
        _snowfall = P_full * _snow_frac * 1e-3
        # Melt: 3 mm SWE per °C per day (standard degree-day factor for temperate/polar snow)
        _melt = np.clip(_T_air_c, 0.0, None).astype(np.float32, copy=False) * 3.0e-3
        # Sublimation: 0.1% of pack per day
        _sublim = _snow_prev * 0.001
        _snow_new = _snow_prev + (_snowfall - _melt - _sublim) * float(days)
        snow_depth_new = np.where(_land_snow, np.clip(_snow_new, 0.0, 10.0), 0.0).astype(np.float32, copy=False)

        # ------------------------------------------------------
        # Land ice mass balance, thickness, and flow (Phase 5 canvas item)
        # ------------------------------------------------------
        # Nested inside the snow block (not a duplicated top-level `if`) so
        # `_land_snow`/`_snow_new`/`_T_air_c` are guaranteed to already be
        # defined. See PlanetParams.enable_land_ice_dynamics for the full
        # design rationale (why thickness is water-equivalent, why flow
        # ignores terrain slope, what is deliberately NOT coupled yet).
        _land_ice_prev = (
            state.land_ice_thickness if state.land_ice_thickness is not None
            else np.zeros((H, W), dtype=np.float32)
        )
        if pp.enable_land_ice_dynamics:
            # Mass balance: gain the overflow the snow-depth cap above would
            # otherwise silently discard; lose mass to a degree-day
            # ablation term with its own (typically higher) factor.
            _ice_gain = np.clip(_snow_new - 10.0, 0.0, None).astype(np.float32, copy=False)
            _ice_melt = (
                np.clip(_T_air_c, 0.0, None).astype(np.float32, copy=False)
                * (float(pp.ice_melt_degree_day_mm) * 1e-3)
            )
            _land_ice_mb = _land_ice_prev + (_ice_gain - _ice_melt) * float(days)
            _land_ice_mb = np.where(
                _land_snow, np.clip(_land_ice_mb, 0.0, None), 0.0
            ).astype(np.float32, copy=False)

            # Flow: substepped for CFL stability, same convention as
            # eddy_heat_flux_coeff/abyssal_overturning_coeff (their r_limit
            # is 0.4 for a 1-D meridional-only stencil; halved here since
            # this is a full 4-neighbor 2-D stencil, stable over a
            # narrower range of the same r = k*H*dt).
            _flow_k = float(pp.ice_flow_diffusivity)
            if _flow_k > 0.0 and np.any(_land_ice_mb > 0.0):
                _h_max = float(np.max(_land_ice_mb))
                _ice_r_limit = 0.2
                _max_ice_sub = 60
                _n_ice_sub = max(1, int(np.ceil(_flow_k * _h_max * float(days) / _ice_r_limit)))
                if _n_ice_sub > _max_ice_sub:
                    # Bound worst-case per-step cost by capping substep count
                    # *and* shrinking the effective diffusivity to match, so
                    # r = k_eff*h_max*dt_sub stays exactly at the stability
                    # limit regardless of thickness/dt. An earlier version
                    # capped substep count alone without touching k, which
                    # silently violated the CFL condition it was meant to
                    # enforce -- caught on a real-terrain check seeding a
                    # 2000 m Antarctic-scale reservoir, which overflowed to
                    # NaN within one MONTHLY (dt=30.44) step at 512x1024.
                    _flow_k_eff = _flow_k * (_max_ice_sub / _n_ice_sub)
                    _n_ice_sub = _max_ice_sub
                else:
                    _flow_k_eff = _flow_k
                _dt_ice_sub = float(days) / _n_ice_sub
                _land_ice_flowed = _land_ice_mb
                for _ in range(_n_ice_sub):
                    _land_ice_flowed = _land_ice_flow_step(
                        _land_ice_flowed, _land_snow, _flow_k_eff, _dt_ice_sub
                    )
            else:
                _land_ice_flowed = _land_ice_mb

            land_ice_thickness_new = np.clip(
                _land_ice_flowed, 0.0, float(pp.land_ice_max_thickness_m)
            ).astype(np.float32, copy=False)

            # Eustatic sea-level diagnostic: land-ice volume (water-
            # equivalent, so no density conversion needed) spread over the
            # ocean's area. land_ice_thickness starts every run's history
            # at zero, so this is cumulative sea-level *change* since
            # dynamics were enabled, not an absolute sea level.
            _lat_rad_full = (0.5 - (np.arange(H, dtype=np.float64) + 0.5) / H) * np.pi
            _w_full = np.cos(_lat_rad_full)
            _w_sum = float(np.sum(_w_full)) + 1e-12
            _total_area_m2 = 4.0 * np.pi * (float(pp.radius_m) ** 2)
            _ice_vol_m3 = float(
                np.sum(np.mean(land_ice_thickness_new, axis=1) * _w_full) / _w_sum
            ) * _total_area_m2
            _ocean_frac = float(
                np.sum(np.mean((~_land_snow).astype(np.float64), axis=1) * _w_full) / _w_sum
            )
            _ocean_area_m2 = max(_ocean_frac * _total_area_m2, 1.0)
            sea_level_change_m_new = float(-_ice_vol_m3 / _ocean_area_m2)
        else:
            land_ice_thickness_new = _land_ice_prev
            sea_level_change_m_new = state.sea_level_change_m
    else:
        snow_depth_new = _snow_prev
        land_ice_thickness_new = (
            state.land_ice_thickness if state.land_ice_thickness is not None
            else np.zeros((H, W), dtype=np.float32)
        )
        sea_level_change_m_new = state.sea_level_change_m

    # Debug logging if requested
    if debug_log:
        if T_full is not None:
            T_stats = {
                'min': float(np.min(T_full)),
                'mean': float(np.mean(T_full)),
                'max': float(np.max(T_full)),
                'p25': float(np.percentile(T_full, 25)),
                'p50': float(np.percentile(T_full, 50)),
                'p75': float(np.percentile(T_full, 75)),
            }
            LOG.info(f"[Simulation Day {new_day:.1f}] T: min={T_stats['min']:.1f}K ({T_stats['min']-273.15:.1f}°C), "
                     f"mean={T_stats['mean']:.1f}K ({T_stats['mean']-273.15:.1f}°C), "
                     f"max={T_stats['max']:.1f}K ({T_stats['max']-273.15:.1f}°C), "
                     f"median={T_stats['p50']:.1f}K ({T_stats['p50']-273.15:.1f}°C)")
            
            # Additional diagnostics: temperature by latitude bands
            H_full = T_full.shape[0]
            eq_idx = H_full // 2
            arctic_idx = int(H_full * 0.15)  # ~66°N
            tropics_idx = int(H_full * 0.4)  # ~23°N
            T_arctic = np.mean(T_full[arctic_idx, :])
            T_tropics = np.mean(T_full[tropics_idx, :])
            T_equator = np.mean(T_full[eq_idx, :])
            LOG.info(f"  By latitude: Arctic(66°N)={float(T_arctic):.1f}K ({float(T_arctic-273.15):.1f}°C), "
                     f"Tropics(23°N)={float(T_tropics):.1f}K ({float(T_tropics-273.15):.1f}°C), "
                     f"Equator={float(T_equator):.1f}K ({float(T_equator-273.15):.1f}°C)")
            
            # Temperature vs elevation analysis
            if state.elevation is not None:
                high_elev_mask = state.elevation > 0.4  # High altitude areas
                if np.any(high_elev_mask):
                    T_high_elev = T_full[high_elev_mask]
                    LOG.info(f"  High altitude (>0.4 elev): T_mean={float(np.mean(T_high_elev)):.1f}K ({float(np.mean(T_high_elev)-273.15):.1f}°C), "
                             f"T_max={float(np.max(T_high_elev)):.1f}K ({float(np.max(T_high_elev)-273.15):.1f}°C)")

    # ------------------------------------------------------
    # Carbon Cycle (Phase 3)
    # ------------------------------------------------------
    if enable_carbon_cycle:
        from carbon_cycle import (
            wetland_ch4_emissions as _wetland_ch4,
            permafrost_thaw_step as _pfc_thaw,
            ch4_oxidation_step as _ch4_oxidize,
            permafrost_init as _pfc_init,
            wildfire_dynamics as _wildfire,
            compute_biome_type as _compute_biome_type,
        )
        from masks import get_masks as _get_masks_cc

        _sea_cc, _land_cc = _get_masks_cc(state.elevation)
        _P_for_carbon = P_full if P_full is not None else np.ones_like(T_air_full) * 3.0

        # --- Slow carbon-cycle bundle: biome classification, wildfire, permafrost
        # thaw, wetland CH4. See CARBON_SLOW_UPDATE_INTERVAL_DAYS above for why.
        _cs = _CARBON_SLOW_CACHE
        _cs_key = (H, W)
        _elapsed_carbon = new_total_days - _cs["last_update_day"]
        # Implausibly large gap (>60d) means a stale cross-run cache (e.g. a
        # previous simulation in this process, or a loaded save with very
        # different total_days) rather than real elapsed simulated time —
        # treat it like a fresh start (dt=days) instead of lump-applying years
        # of accumulated flux in one call.
        _is_first_or_reset = (
            _cs["last_update_day"] <= -9000.0
            or _cs["key"] != _cs_key
            or abs(_elapsed_carbon) > 60.0
        )
        _do_carbon_slow = _is_first_or_reset or abs(_elapsed_carbon) >= CARBON_SLOW_UPDATE_INTERVAL_DAYS
        if _do_carbon_slow:
            _carbon_dt = float(days) if _is_first_or_reset else float(abs(_elapsed_carbon))
            cached_biome = _compute_biome_type(T_air_full, _P_for_carbon, _land_cc)
            _cs["biome"] = cached_biome
            _cs["key"] = _cs_key
            _cs["last_update_day"] = new_total_days
        else:
            cached_biome = _cs["biome"]
            _carbon_dt = 0.0  # unused unless _do_carbon_slow

        # Rolling wind-speed EMA feeding ocean_co2_flux's piston velocity (Jul 2026
        # fix): Wanninkhof's k∝u² is calibrated for time-averaged wind, not the
        # instantaneous per-step value used previously (see
        # carbon_cycle.ocean_co2_flux docstring).
        _wind_speed_now = np.sqrt(u_full**2 + v_full**2).astype(np.float32, copy=False)
        if state.wind_speed_avg is None:
            wind_speed_avg_new = _wind_speed_now.copy()
        else:
            _alpha = np.clip(days / max(pp.co2_wind_averaging_days, 1e-6), 0.0, 1.0)
            wind_speed_avg_new = ((1.0 - _alpha) * state.wind_speed_avg
                                   + _alpha * _wind_speed_now).astype(np.float32, copy=False)

        # Create temporary state for carbon cycle computation
        temp_state_for_carbon = PlanetState(
            day_of_year=new_day,
            total_days=new_total_days,
            elevation=state.elevation,
            temperature=T_air_full,  # vegetation NPP responds to air temperature
            wind_u=u_full,
            wind_v=v_full,
            precipitation=P_full,
            co2_atmosphere=state.co2_atmosphere,
            co2_ocean=state.co2_ocean,
            vegetation_biomass=state.vegetation_biomass,
            wind_speed_avg=wind_speed_avg_new,
        )

        # Evolve the per-step half of the carbon cycle: ocean CO2 exchange +
        # vegetation NPP/growth (fast-responding; stays per-step every mode).
        co2_atm_new, co2_ocean_new, biomass_new, co2_forcing_result = carbon_cycle_step(
            temp_state_for_carbon, days, biome=cached_biome
        )

        # CO2 greenhouse feedback is now applied to T_base (equilibrium temperature) above,
        # not added to final temperature here. This prevents runaway warming.

        # Initialize permafrost on first step
        _pfc = state.permafrost_carbon
        if _pfc is None and T_air_full is not None:
            _pfc = _pfc_init(state.elevation, T_air_full)

        ch4_ppb = state.ch4_atmosphere
        pfc_new = _pfc

        if _do_carbon_slow:
            # Wildfire: applies _carbon_dt days worth of fire risk in one lump
            # (moved out of carbon_cycle_step so it can share this cache).
            biomass_new, co2_from_fire = _wildfire(
                biomass_new, T_air_full, _P_for_carbon, state.soil_moisture, _carbon_dt
            )
            co2_atm_new = float(np.clip(co2_atm_new + co2_from_fire, 100.0, 10000.0))

            if _pfc is not None and T_full is not None:
                pfc_new, d_co2_pfc, d_ch4_pfc = _pfc_thaw(_pfc, T_full, snow_depth_new, _carbon_dt)
                ch4_ppb += d_ch4_pfc
                co2_atm_new += d_co2_pfc

            # Wetland emissions
            if T_full is not None:
                ch4_ppb += _wetland_ch4(T_full, state.soil_moisture, _land_cc, _carbon_dt)

        # Background natural CH4 source balancing OH oxidation at the planetary
        # baseline (see carbon_cycle.ch4_natural_source): without it CH4 decayed
        # from 1900 ppb toward zero over multi-decade runs (τ=9yr), injecting a
        # spurious ~-1 W/m² forcing drift no modeled source could offset.
        if pp.ch4_baseline_ppb > 0.0:
            from carbon_cycle import ch4_natural_source as _ch4_source
            ch4_ppb += _ch4_source(pp.ch4_baseline_ppb, float(days))

        # Atmospheric oxidation (9-yr lifetime) — cheap scalar op, stays per-step.
        ch4_atm_new = float(np.clip(_ch4_oxidize(ch4_ppb, float(days)), 100.0, 50_000.0))

        if debug_log:
            LOG.info(f"Carbon cycle: CO2={co2_atm_new:.1f} ppm, forcing={co2_forcing_result:.2f} W/m², CH4={ch4_atm_new:.0f} ppb, "
                     f"slow_update={_do_carbon_slow}")
    else:
        co2_atm_new = state.co2_atmosphere
        co2_ocean_new = state.co2_ocean
        biomass_new = state.vegetation_biomass
        ch4_atm_new = state.ch4_atmosphere
        pfc_new = state.permafrost_carbon
        wind_speed_avg_new = state.wind_speed_avg

    # SESAM stage P6b (docs/SESAM_GAP_ANALYSIS.md Sec7): replace the legacy
    # air-temperature evolution and row-target precipitation allocator with
    # the (A40)/(A42)/(A44) column-energy/water closure, gated and default
    # off. Runs last, right before state assembly, so it overrides only
    # T_air/humidity/precipitation/the new SESAM water reservoir -- every
    # other quantity above (ocean, ice, land, biome, carbon, clouds, the
    # legacy skin temperature T_full itself) is computed exactly as before
    # and this branch only reads it. See sesam_coupling.py's module
    # docstring for the three documented bridges (wind/EKE/diabatic source)
    # this first live-coupling sub-stage relies on.
    sesam_qq_next = state.sesam_column_water_mm
    sesam_ht_next = state.sesam_tropopause_height_m
    if (
        pp.enable_sesam_column_closure
        and T_air_full is not None
        and T_full is not None
        and humidity_next is not None
        and u_full is not None
        and v_full is not None
        and float(days) > 0.0
    ):
        _sesam_sea_mask, _sesam_land_mask = get_masks(state.elevation, use_cache=True)
        _sesam_elevation_m = np.clip(np.asarray(state.elevation, dtype=np.float64), 0.0, 1.0) * (
            float(pp.max_elevation_km) * 1000.0
        )
        _sesam_ta = (
            state.air_temperature if state.air_temperature is not None else T_air_full
        )
        _sesam_ra = state.humidity if state.humidity is not None else humidity_next
        _sesam_qq = state.sesam_column_water_mm

        # SESAM stage P6c: when the P2/P3 dynamics gate is also on, replace
        # bridges 1-2 (legacy wind, uniform-EKE placeholder) with SESAM's own
        # zonal-only SLP/wind (P2) and local-steady-state EKE (P3), computed
        # once for the whole outer step -- see sesam_wind_coupling.py's
        # module docstring for why this is recomputed per outer call rather
        # than per inner substep (SLP/wind/EKE track the slowly-varying
        # skin/air temperature, the same simplification the legacy
        # precipitation substep loop already makes for its own inner loop).
        _sesam_wind_u, _sesam_wind_v, _sesam_eke = u_full, v_full, None
        if pp.enable_sesam_dynamics:
            _sesam_ice_mask = (
                np.asarray(state.ice_cover) > 0.5 if state.ice_cover is not None else None
            )
            _sesam_dynamics_step = sesam_wind_and_eke_step(
                air_temperature_k=_sesam_ta,
                skin_temperature_k=T_full,
                relative_humidity=_sesam_ra,
                elevation_m=_sesam_elevation_m,
                land_mask=_sesam_land_mask,
                ice_mask=_sesam_ice_mask,
                surface_pressure_pa=float(pp.surface_pressure_pa),
                radius_m=float(pp.radius_m),
                gravity=float(pp.surface_gravity),
                omega=float(pp.omega),
            )
            _sesam_wind_u = _sesam_dynamics_step.wind_u_m_s
            _sesam_wind_v = _sesam_dynamics_step.wind_v_m_s
            _sesam_eke = _sesam_dynamics_step.eke_m2_s2

        # SESAM stage P6d: when the P5 radiation gate is also on, replace
        # bridge 3 (the (T*-Ta)/1-day bulk relaxation) with SESAM's own
        # (A69)-(A117)/(A10) SWa/LWa split. Unlike P6c's wind/EKE, this is
        # recomputed every 1-day substep, not once per outer step: a real
        # instability was found and root-caused during this stage's own
        # development (see sesam_radiation_coupling.py's module docstring) --
        # holding SWa/LWa fixed across a multi-day MONTHLY call removes the
        # Stefan-Boltzmann negative feedback (LW emission rising with Ta) that
        # keeps a radiative source term self-limiting, so a stale forcing
        # applied for many consecutive 1-day substeps overshoots exactly the
        # way the pre-fix diabatic bridge did (docs/SESAM_GAP_ANALYSIS.md
        # Sec7 P6b's own overshoot bug), just via a different mechanism.
        _sesam_ice_mask_rad = None
        if pp.enable_sesam_radiation:
            _sesam_ice_mask_rad = (
                np.asarray(state.ice_cover) > 0.5 if state.ice_cover is not None else None
            )
            _sesam_h, _sesam_w = _sesam_elevation_m.shape
            _sesam_lat_deg = 90.0 - (np.arange(_sesam_h) + 0.5) * 180.0 / _sesam_h
            _sesam_lat_rad = np.deg2rad(_sesam_lat_deg)[:, None] * np.ones((1, _sesam_w))

        _sesam_n_sub = max(1, int(round(float(days) / _SESAM_COLUMN_CLOSURE_SUBSTEP_DAYS)))
        _sesam_sub_dt = float(days) / _sesam_n_sub
        _sesam_p_accum = None
        _sesam_ht = state.sesam_tropopause_height_m
        for _sesam_sub_i in range(_sesam_n_sub):
            _sesam_swa, _sesam_lwa = None, None
            if pp.enable_sesam_radiation:
                _sesam_sub_day = float(state.day_of_year) + _sesam_sub_i * _sesam_sub_dt
                _sesam_declination = pp.solar_declination(_sesam_sub_day)
                _sesam_insolation = pp.daily_mean_insolation(_sesam_lat_rad, _sesam_sub_day)
                _sesam_radiation = sesam_radiation_step(
                    air_temperature_k=_sesam_ta,
                    skin_temperature_k=T_full,
                    relative_humidity=_sesam_ra,
                    cloud_fraction=cloud_full if cloud_full is not None else np.zeros_like(_sesam_ta),
                    column_water_mm=_sesam_qq,
                    elevation_m=_sesam_elevation_m,
                    land_mask=_sesam_land_mask,
                    ice_mask=_sesam_ice_mask_rad,
                    snow_depth=state.snow_depth,
                    tropopause_height_m=_sesam_ht,
                    surface_pressure_pa=float(pp.surface_pressure_pa),
                    gravity=float(pp.surface_gravity),
                    co2_ppm=float(co2_atm_new),
                    day_of_year=_sesam_sub_day,
                    itcz_seasonal_response=float(pp.itcz_seasonal_response),
                    solar_declination_rad=float(_sesam_declination),
                    daily_mean_insolation_w_m2=_sesam_insolation,
                    dt_days=_sesam_sub_dt,
                )
                _sesam_swa = _sesam_radiation.sw_absorbed_w_m2
                _sesam_lwa = _sesam_radiation.lw_net_w_m2
                _sesam_ht = _sesam_radiation.tropopause_height_m

            _sesam_step = sesam_column_closure_step(
                air_temperature_k=_sesam_ta,
                skin_temperature_k=T_full,
                relative_humidity=_sesam_ra,
                column_water_mm=_sesam_qq,
                wind_u_m_s=_sesam_wind_u, wind_v_m_s=_sesam_wind_v,
                elevation_m=_sesam_elevation_m, land_mask=_sesam_land_mask,
                surface_pressure_pa=float(pp.surface_pressure_pa),
                radius_m=float(pp.radius_m),
                dt_days=_sesam_sub_dt,
                eke_m2_s2=_sesam_eke,
                sw_absorbed_w_m2=_sesam_swa,
                lw_net_w_m2=_sesam_lwa,
                gravity_m_s2=float(pp.surface_gravity),
            )
            _sesam_ta = _sesam_step.air_temperature_k
            _sesam_ra = _sesam_step.relative_humidity
            _sesam_qq = _sesam_step.column_water_mm
            _sesam_p_accum = (
                _sesam_step.precipitation_mm_day.astype(np.float32)
                if _sesam_p_accum is None
                else _sesam_p_accum + _sesam_step.precipitation_mm_day
            )
        T_air_full = _sesam_ta
        humidity_next = _sesam_ra
        P_full = (_sesam_p_accum / _sesam_n_sub).astype(np.float32, copy=False)
        sesam_qq_next = _sesam_qq
        sesam_ht_next = _sesam_ht

    new_state = PlanetState(
        day_of_year=new_day,
        total_days=new_total_days,
        elevation=state.elevation,
        temperature=T_full,         # T_sst: sea surface / land surface temperature
        air_temperature=T_air_full, # T_air: 2m air temperature
        wind_u=u_full,
        wind_v=v_full,
        precipitation=P_full,
        humidity=humidity_next,
        sesam_column_water_mm=sesam_qq_next,
        sesam_tropopause_height_m=sesam_ht_next,
        soil_moisture=soil_next,
        soil_moisture_deep=soil_deep_next,
        cloud_cover=cloud_full,
        cloud_water=cloud_water_full,
        atmospheric_condensate=condensate_next,
        precipitating_hydrometeors=hydrometeors_next,
        land_deep_temperature=land_deep_full,
        boundary_layer_temperature=boundary_layer_full,
        boundary_layer_interface_temperature=boundary_interface_full,
        midlevel_temperature=midlevel_temperature_next,
        midlevel_humidity=midlevel_humidity_next,
        upperlevel_temperature=upperlevel_temperature_next,
        grey_optical_depth=(
            radiative_optical_depth_full
            if pp.enable_pressure_defined_radiative_temperature_profile
            else state.grey_optical_depth
        ),
        upperlevel_humidity=upperlevel_humidity_next,
        snow_depth=snow_depth_new,
        ice_cover=ice_full,
        co2_atmosphere=co2_atm_new,
        co2_ocean=co2_ocean_new,
        vegetation_biomass=biomass_new,
        wind_speed_avg=wind_speed_avg_new,
        # Phase 1: Climate averaging and stable biomes
        climate_temp_avg=temp_avg,
        climate_precip_avg=precip_avg,
        climate_sample_days=sample_days,
        biome_type=biome_new,
        biome_last_update_day=biome_last_update,
        # Monthly statistics and Köppen classification
        monthly_temp=monthly_temp,
        monthly_precip=monthly_precip,
        monthly_sample_count=monthly_sample_count,
        koppen_type=koppen_new,
        ice_sheet_age=ice_sheet_age_new,
        # Feature 3: salinity
        salinity=salinity_new,
        # Feature 4: CH4 / permafrost
        ch4_atmosphere=ch4_atm_new,
        permafrost_carbon=pfc_new,
        # Feature 5: deep ocean
        T_deep_ocean=T_deep_full,
        # Feature 6: sea ice thickness
        ice_thickness=ice_thick_full,
        # Feature 7: jet stream dynamics
        jet_index_nh=jet_index_nh_new,
        jet_index_sh=jet_index_sh_new,
        jet_block_lon_nh=jet_block_lon_nh_new,
        jet_block_days_left_nh=jet_block_days_left_nh_new,
        jet_block_total_days_nh=jet_block_total_nh_new,
        jet_block_lon_sh=jet_block_lon_sh_new,
        jet_block_days_left_sh=jet_block_days_left_sh_new,
        jet_block_total_days_sh=jet_block_total_sh_new,
        # Feature 8: 1.5-layer atmosphere upper-level wind
        wind_u_aloft=u2_full,
        wind_v_aloft=v2_full,
        # Native pressure-column middle-level circulation (gated; old saves
        # retain ``None`` while the experimental path is disabled).
        midlevel_wind_u=midlevel_wind_u_full,
        midlevel_wind_v=midlevel_wind_v_full,
        omega_lower_mid_pa_s=omega_lower_mid_full,
        omega_mid_upper_pa_s=omega_mid_upper_full,
        # Section 17: the three-level path's own independent upper-level wind
        # (never fed by, nor feeding back into, the shared jet-stream kernel
        # above). ``None`` while the three-level gate has never been active,
        # matching the midlevel-wind precedent's old-save/gate-off behavior.
        upperlevel_wind_u=upperlevel_wind_u_full,
        upperlevel_wind_v=upperlevel_wind_v_full,
        pressure_moisture_condensation_mm_day=(
            None
            if _precipitation_diagnostics is None
            or float(days) <= 0.0
            or "pressure_moisture_cloud_created_mm" not in _precipitation_diagnostics
            else np.asarray(
                _precipitation_diagnostics["pressure_moisture_cloud_created_mm"],
                dtype=np.float32,
            ) / max(float(days), 1e-12)
        ),
        pressure_overturning_heating_w_m2=pressure_overturning_heating_next,
        pressure_coordinate_heat_convergence_w_m2=pressure_coordinate_heat_convergence_next,
        # The hydrostatic-sigma state is inert until its deepest experimental
        # gate owns a complete transition. Preserve it verbatim meanwhile so
        # save/load and gate-off stepping cannot silently discard it.
        **_sigma_state_next,
        # Preserve the scenario baseline; `pp` may be a time-varying effective
        # Milankovitch snapshot and must not become the next cycle's baseline.
        planet_params=base_pp,
        surface_water_mm=surface_water_mm_new,
        river_discharge_mm_day=river_discharge_new,
        runoff_to_ocean_mm_day=runoff_to_ocean_new,
        land_ice_thickness=land_ice_thickness_new,
        sea_level_change_m=sea_level_change_m_new,
    )

    # Return state and components (empty dict if not tracking)
    return new_state, temp_components if temp_components else {}


def simulate_multiple_steps(
    initial_state: PlanetState,
    total_days: float,
    step_days: float = 1.0,
    **kwargs,
) -> tuple[list[PlanetState], list[dict]]:
    """Simulate multiple steps, returning intermediate states.

    Args:
        initial_state: Starting state
        total_days: Total simulation time
        step_days: Time per step
        **kwargs: Passed to simulate_step

    Returns:
        List of states at each step (including initial)
    """
    return run_multiple_steps(
        initial_state,
        total_days,
        step_days,
        step_function=simulate_step,
        step_kwargs=kwargs,
    )


def initialize_coupled_grey_profile(state, planet_params):
    """Return ``state`` with the mid/upper radiative profile re-initialized.

    The coupled two-layer grey gate inherits its first mid/upper temperatures
    from the dry-adiabatic diagnostic profile, which is far from the grey
    budget's own equilibrium (measured 2026-08-16: +15.2 K area-mean, max
    +60 K at the mid level for the 32x64 handoff state).  The day-1 grey
    gains then shock the column and the diabatic-omega feedback destroys the
    static stability it divides by.  Coupled-grey spin-up/evaluation
    protocols should apply this once, at the handoff from the diagnostic
    spin-up to the coupled integration; it is an explicit initialization
    step, not a per-step adjustment, and has no effect on any other gate
    configuration.
    """
    pp = planet_params
    if not bool(pp.enable_coupled_two_layer_grey_radiation):
        raise ValueError("grey profile initialization requires the coupled grey gate")
    missing = [
        name
        for name, value in (
            ("grey_optical_depth", state.grey_optical_depth),
            ("midlevel_temperature", state.midlevel_temperature),
            ("upperlevel_temperature", state.upperlevel_temperature),
            ("air_temperature", state.air_temperature),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "grey profile initialization requires a diagnostic spin-up state: "
            + ", ".join(missing)
        )
    profile = grey_radiative_convective_equilibrium_temperatures(
        np.asarray(state.temperature, dtype=np.float64),
        np.asarray(state.air_temperature, dtype=np.float64),
        np.asarray(state.grey_optical_depth, dtype=np.float64),
        float(pp.two_layer_pressure_depth_pa),
        float(pp.three_level_mid_upper_pressure_depth_pa),
        gas_constant_dry_air_j_kg_k=float(pp.gas_constant_dry),
        cp_dry_air_j_kg_k=float(pp.cp_dry),
    )
    return state._replace(
        midlevel_temperature=profile.midlevel_temperature_k.astype(np.float32),
        upperlevel_temperature=profile.upperlevel_temperature_k.astype(np.float32),
    )


def create_initial_state(
    elevation: np.ndarray,
    day_of_year: float = 80.0,
    **kwargs,
) -> PlanetState:
    """Create initial planet state from elevation map.

    Args:
        elevation: (H, W) terrain elevation [0,1]
        day_of_year: Starting day (0-365.2422)
        **kwargs: Passed to simulate_step for initial computation

    Returns:
        Initialized state with all fields computed
    """
    return initialize_state(
        elevation,
        day_of_year,
        step_function=simulate_step,
        step_kwargs=kwargs,
    )


def _evolve_temperature(
    T_prev: np.ndarray,
    T_base: np.ndarray,
    elevation: np.ndarray,
    Hc: int,
    Wc: int,
    block_size: int,
    H: int,
    W: int,
    day_of_year: float,
    days: float,
    *,
    T_air_prev: np.ndarray | None = None,
    wind_u: np.ndarray | None = None,
    wind_v: np.ndarray | None = None,
    land_sea_contrast: float = 0.0,
    thermal_diffusion: float = 0.04,
    T_base_land: np.ndarray | None = None,
    ice_cover: np.ndarray | None = None,
    ocean_transport_coeff: float = 0.5,
    ocean_exchange_coeff: float = 0.03,
    ocean_exchange_inertia: float = 0.0,
    epsilon_equator: float = 0.72,
    epsilon_pole: float = 0.50,
    ice_albedo_strength: float = 1.0,
    humidity: np.ndarray | None = None,
    soil_moisture: np.ndarray | None = None,
    soil_moisture_deep: np.ndarray | None = None,
    track_components: bool = False,
    precipitation: np.ndarray | None = None,
    vegetation_biomass: np.ndarray | None = None,
    biome: np.ndarray | None = None,
    koppen_type: np.ndarray | None = None,
    planet_params: PlanetParams | None = None,
    elev_c: np.ndarray | None = None,
    snow_depth: np.ndarray | None = None,
    feedback_flags: dict[str, bool] | None = None,
    total_days: float | None = None,  # monotonic sim time for the ocean-update cache
    prev_cloud_cover: np.ndarray | None = None,  # Feature 1: cloud persistence
    T_deep_ocean: np.ndarray | None = None,       # Feature 5: deep ocean layer
    ice_thickness: np.ndarray | None = None,      # Feature 6: thickness-dependent albedo
    prev_cloud_water: np.ndarray | None = None,   # Feature: prognostic cloud water
    midlevel_condensate: np.ndarray | None = None,
    land_deep_temperature: np.ndarray | None = None,
    boundary_layer_temperature: np.ndarray | None = None,
    boundary_layer_interface_temperature: np.ndarray | None = None,
    radiative_midlevel_temperature: np.ndarray | None = None,
    radiative_upperlevel_temperature: np.ndarray | None = None,
    radiative_optical_depth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray | None, np.ndarray | None]:
    """Evolve temperature with FULL physics: Radiation, Advection, Latent Heat.

    Physics upgrades (Items 1-15):
    - Cloud-radiation feedback (Albedo + Greenhouse)
    - Snow/Ice albedo feedback
    - Latent heat of phase changes (Evap/Condensation)
    - Sensible heat flux
    - Longwave radiation emission
    - Surface heat capacity variations
    """
    # Validate expected shapes
    assert T_prev.shape == (Hc, Wc)
    
    # 1. Prepare Surface Properties
    # Downsample elevation (skip if caller already provides coarse elev)
    if elev_c is None:
        elev_pad = np.pad(elevation.astype(np.float32, copy=False), ((0, Hc*block_size - H), (0, Wc*block_size - W)), mode="edge")
        elev_c = elev_pad.reshape(Hc, block_size, Wc, block_size).mean(axis=(1, 3))
    # Land/Sea Masks — elev_c is a transient coarse array; skip cache to avoid stale hits
    sea_mask, land_mask = get_masks(elev_c, use_cache=False)
    land_fraction = land_mask.astype(np.float32) # Simplified for now
    
    _pp = planet_params if planet_params is not None else EARTH

    # 2. Wind Field (Prognostic or Diagnostic)
    if wind_u is None or wind_v is None:
        u, v = generate_wind_field(
            Hc,
            Wc,
            day_of_year=day_of_year,
            block_size=1,
            elevation=elev_c,
            planet_params=_pp,
        )
    else:
        u, v = wind_u, wind_v
        
    dt_sec = days * 86400.0

    # --- Two prognostic fields ---
    # T_sst: sea surface temperature (ocean) / land surface temperature.
    #        Driven by radiative balance, ocean transport, and land-surface physics.
    #        NOT advected by wind — the atmosphere blows over it, not with it.
    # T_air: 2-metre air temperature.
    #        Advected by wind, diffused, and coupled to T_sst through surface exchange.
    T_sst = T_prev.copy().astype(np.float32, copy=False)
    T_air = (T_air_prev.copy() if T_air_prev is not None else T_prev.copy()).astype(np.float32, copy=False)
    land_deep_out: np.ndarray | None = None
    boundary_layer_out: np.ndarray | None = boundary_layer_temperature
    boundary_interface_out: np.ndarray | None = boundary_layer_interface_temperature
    boundary_exchange_gain = None
    boundary_interface_upper_gain = None
    boundary_horizontal_convergence = None
    boundary_continuity_exchange = None
    boundary_transport_substeps = None
    boundary_horizontal_area_mean = None

    # === PASS 1: T_air dynamics — advection, diffusion, surface exchange ===
    lat_1d = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi
    cos_lat = np.cos(lat_1d).clip(0.05, 1.0)
    dx_lat = (2.0 * np.pi * _pp.radius_m * cos_lat / Wc).astype(np.float32, copy=False)
    dy = float(np.pi * _pp.radius_m / Hc)
    dx_2d = dx_lat[:, None]
    u_cfl = np.clip(np.abs(u) * dt_sec / dx_2d, 0.0, 0.5).astype(np.float32, copy=False)
    v_cfl = np.clip(np.abs(v) * dt_sec / dy, 0.0, 0.5).astype(np.float32, copy=False)

    T_before_advection = T_air.copy()  # for component tracking

    # Advect T_air with wind (atmosphere moves horizontally)
    if NUMBA_AVAILABLE:
        T_air = _advect_temperature_x_numba(T_air, u.astype(np.float32, copy=False), u_cfl)
        T_air = _advect_temperature_y_numba(T_air, v.astype(np.float32, copy=False), v_cfl)
    else:
        T_east = np.roll(T_air, -1, axis=1)
        T_west = np.roll(T_air, 1, axis=1)
        T_x = np.where(u >= 0, T_west, T_east)
        T_air = T_air + u_cfl * (T_x - T_air)
        T_north = np.roll(T_air, -1, axis=0)
        T_south = np.roll(T_air, 1, axis=0)
        T_y = np.where(v >= 0, T_south, T_north)
        T_air = T_air + v_cfl * (T_y - T_air)

    T_after_advection = T_air.copy()

    # Diffuse T_air (atmospheric mixing).
    # Explicit Laplacian diffusion is only stable for r = coeff*1.2*days below
    # ~0.5 per application (same forward-difference CFL bound handled for the
    # ocean eddy flux below); production substeps (days ≤ 7) are fine at the
    # 0.04 default, but direct calls with large `days` need sub-stepping.
    T_before_diffusion = T_air.copy()
    _r_diff = float(thermal_diffusion) * 1.2 * float(days)
    _n_diff_sub = max(1, int(np.ceil(_r_diff / 0.4)))
    _days_diff_sub = float(days) / _n_diff_sub
    for _ in range(_n_diff_sub):
        if NUMBA_AVAILABLE:
            T_air = _apply_diffusion_numba(T_air, float(thermal_diffusion), _days_diff_sub, iterations=2)
        else:
            for _ in range(2):
                T_pad = np.pad(T_air, ((1, 1), (0, 0)), mode="edge")
                c = T_pad[1:-1, :]
                n = T_pad[0:-2, :]
                s = T_pad[2:, :]
                e = np.roll(c, -1, axis=1)
                w = np.roll(c, 1, axis=1)
                T_lap = n + s + e + w - 4.0 * c
                T_air = T_air + thermal_diffusion * 1.2 * np.clip(T_lap, -30.0, 30.0) * _days_diff_sub

    # Capture this before surface exchange or any radiative/surface physics.
    # The old end-of-step component subtraction included all downstream terms
    # and therefore was not a usable atmospheric-transport diagnostic.
    _free_transport_closure_delta_k = np.zeros_like(T_air, dtype=np.float32)
    if (
        _pp.enable_force_restore_land
        and _pp.enable_force_restore_boundary_layer
        and _pp.enable_boundary_layer_capacity_aware_free_air_transport
    ):
        from boundary_layer import (
            close_free_air_transport_energy,
            mixed_layer_pressure_thickness,
            overlying_layer_pressure_thickness,
        )

        _transport_column_capacity = (
            float(_pp.surface_pressure_pa) / float(_pp.surface_gravity)
            * float(_pp.cp_dry)
        )
        _transport_delta_p = mixed_layer_pressure_thickness(
            surface_pressure_pa=float(_pp.surface_pressure_pa),
            gravity_m_s2=float(_pp.surface_gravity),
            gas_constant_j_kg_k=float(_pp.gas_constant_dry),
            reference_temperature_k=float(_pp.boundary_layer_reference_temperature_k),
            mixed_layer_depth_m=float(_pp.boundary_layer_mixed_depth_m),
        )
        _transport_boundary_capacity = (
            _transport_delta_p / float(_pp.surface_gravity) * float(_pp.cp_dry)
        )
        _transport_interface_delta_p = 0.0
        if _pp.enable_boundary_layer_interface_reservoir:
            _transport_interface_delta_p = overlying_layer_pressure_thickness(
                surface_pressure_pa=float(_pp.surface_pressure_pa),
                gravity_m_s2=float(_pp.surface_gravity),
                gas_constant_j_kg_k=float(_pp.gas_constant_dry),
                reference_temperature_k=float(_pp.boundary_layer_reference_temperature_k),
                layer_base_m=float(_pp.boundary_layer_mixed_depth_m),
                layer_depth_m=float(_pp.boundary_layer_mixed_depth_m),
            )
        _transport_interface_capacity = (
            _transport_interface_delta_p
            / float(_pp.surface_gravity)
            * float(_pp.cp_dry)
        )
        _transport_free_capacity = np.where(
            land_mask,
            _transport_column_capacity
            - _transport_boundary_capacity
            - _transport_interface_capacity,
            _transport_column_capacity,
        )
        _raw_T_after_diffusion = T_air
        T_air = close_free_air_transport_energy(
            T_before_advection,
            T_air,
            _transport_free_capacity,
            lat_1d,
        )
        _free_transport_closure_delta_k = T_air - _raw_T_after_diffusion
    T_after_diffusion = T_air.copy()
    resolved_transport_delta_k = T_after_diffusion - T_before_advection

    # T_air relaxes toward surface temperature (T_sst).
    # Over ocean: ~4-day time constant (efficient sensible heat flux at ocean surface).
    # Over land: ~2-day time constant (land surface heats/cools overlying air quickly).
    # Fraction capped at 0.5 so relaxation is stable for any dt (no overshoot).
    #
    # RESTORED 2026-07-25 after this block's removal in d8631cb was measured to be
    # a regression: without it the air column decouples from the surface and NH
    # polar air runs ~25 degC too warm on the standard 64x128 spinup fixture
    # (measured +25.4 degC vs +0.5 degC with this block restored). The
    # equal-and-opposite exchange added below does NOT subsume this -- that term
    # conserves heat between the two layers, but nothing else ties the air column
    # to the surface it sits on. See overnight/FINDINGS.md (2026-07-25).
    _conservative_land_air = bool(
        _pp.enable_force_restore_land
        and _pp.enable_force_restore_conservative_land_air_exchange
    )
    _boundary_layer_active = bool(
        _pp.enable_force_restore_land and _pp.enable_force_restore_boundary_layer
    )
    _boundary_interface_active = bool(
        _boundary_layer_active and _pp.enable_boundary_layer_interface_reservoir
    )
    if (
        _boundary_interface_active
        and (
            boundary_interface_out is None
            or np.asarray(boundary_interface_out).shape != T_air.shape
        )
    ):
        boundary_interface_out = T_air.copy()
    _boundary_cloud_temperature_active = bool(
        _boundary_layer_active
        and _pp.enable_boundary_layer_near_surface_cloud_temperature
    )
    _boundary_cloud_memory_active = bool(
        _boundary_layer_active
        and _pp.enable_boundary_layer_split_invariant_cloud_memory
    )
    if (
        _boundary_cloud_temperature_active
        and (
            boundary_layer_out is None
            or np.asarray(boundary_layer_out).shape != T_air.shape
        )
    ):
        boundary_layer_out = T_air.copy()
    _capacity_aware_airsea = bool(
        _boundary_layer_active
        and _pp.enable_boundary_layer_capacity_aware_airsea_exchange
    )
    _capacity_mld = (
        _ocean_mixed_layer_depth(
            np.repeat(lat_1d[:, None], Wc, axis=1), day_of_year, _pp
        )
        if _capacity_aware_airsea
        else None
    )
    _land_air_relaxation = 0.0 if (_conservative_land_air or _boundary_layer_active) else 0.50
    k_air_surface = np.where(sea_mask, 0.25, _land_air_relaxation).astype(
        np.float32, copy=False
    )
    _air_frac = np.minimum(k_air_surface * float(days), 0.5).astype(np.float32, copy=False)
    _surface_air_relaxation_delta_k = _air_frac * (T_sst - T_air)
    T_air = (T_air + _surface_air_relaxation_delta_k).astype(np.float32, copy=False)
    _column_air_capacity = (
        float(_pp.surface_pressure_pa) / float(_pp.surface_gravity)
        * float(_pp.cp_dry)
    )
    _ocean_capacity = (
        4.186e6 * _capacity_mld
        if _capacity_mld is not None
        else np.ones_like(T_air, dtype=np.float32)
    )
    _ocean_air_relaxation_ocean_delta_k = np.zeros_like(T_air, dtype=np.float32)
    if _capacity_aware_airsea:
        _ocean_air_relaxation_ocean_delta_k = np.where(
            sea_mask,
            -_column_air_capacity * _surface_air_relaxation_delta_k / _ocean_capacity,
            0.0,
        ).astype(np.float32, copy=False)
        T_sst = (T_sst + _ocean_air_relaxation_ocean_delta_k).astype(
            np.float32, copy=False
        )
    if dt_sec > 0.0:
        _ocean_air_relaxation_gain_w_m2 = np.where(
            sea_mask,
            _column_air_capacity * _surface_air_relaxation_delta_k / dt_sec,
            0.0,
        )
        _ocean_air_relaxation_ocean_gain_w_m2 = np.where(
            sea_mask,
            _ocean_capacity * _ocean_air_relaxation_ocean_delta_k / dt_sec,
            0.0,
        )
    else:
        _ocean_air_relaxation_gain_w_m2 = np.zeros_like(T_air, dtype=np.float32)
        _ocean_air_relaxation_ocean_gain_w_m2 = np.zeros_like(T_air, dtype=np.float32)
    _ocean_air_relaxation_residual_w_m2 = (
        _ocean_air_relaxation_gain_w_m2
        + _ocean_air_relaxation_ocean_gain_w_m2
    )

    _boundary_transport_active = bool(
        _boundary_layer_active
        and _pp.enable_boundary_layer_horizontal_transport
        and dt_sec > 0.0
    )
    if _boundary_transport_active:
        from boundary_layer import (
            mixed_layer_pressure_thickness,
            overlying_layer_pressure_thickness,
            transport_boundary_layer_energy,
        )

        if boundary_layer_out is None or np.asarray(boundary_layer_out).shape != T_air.shape:
            boundary_layer_out = T_air.copy()
        _boundary_pressure_pa = mixed_layer_pressure_thickness(
            surface_pressure_pa=float(_pp.surface_pressure_pa),
            gravity_m_s2=float(_pp.surface_gravity),
            gas_constant_j_kg_k=float(_pp.gas_constant_dry),
            reference_temperature_k=float(_pp.boundary_layer_reference_temperature_k),
            mixed_layer_depth_m=float(_pp.boundary_layer_mixed_depth_m),
        )
        transport_step = transport_boundary_layer_energy(
            boundary_layer_out,
            T_air,
            u,
            v,
            pressure_thickness_pa=_boundary_pressure_pa,
            surface_pressure_pa=float(_pp.surface_pressure_pa),
            cp_j_kg_k=float(_pp.cp_dry),
            gravity_m_s2=float(_pp.surface_gravity),
            radius_m=float(_pp.radius_m),
            dt_seconds=dt_sec,
            active_mask=land_mask,
            additional_reserved_pressure_pa=(
                overlying_layer_pressure_thickness(
                    surface_pressure_pa=float(_pp.surface_pressure_pa),
                    gravity_m_s2=float(_pp.surface_gravity),
                    gas_constant_j_kg_k=float(_pp.gas_constant_dry),
                    reference_temperature_k=float(_pp.boundary_layer_reference_temperature_k),
                    layer_base_m=float(_pp.boundary_layer_mixed_depth_m),
                    layer_depth_m=float(_pp.boundary_layer_mixed_depth_m),
                )
                if _boundary_interface_active
                else 0.0
            ),
        )
        boundary_layer_out = transport_step.boundary_temperature
        T_air = transport_step.free_temperature
        boundary_horizontal_convergence = transport_step.horizontal_convergence_w_m2
        boundary_continuity_exchange = transport_step.continuity_exchange_gain_w_m2
        boundary_transport_substeps = transport_step.substeps
        _bl_lat_edges = np.linspace(np.pi / 2.0, -np.pi / 2.0, Hc + 1)
        _bl_area_rows = np.sin(_bl_lat_edges[:-1]) - np.sin(_bl_lat_edges[1:])
        boundary_horizontal_area_mean = float(
            np.sum(
                (
                    boundary_horizontal_convergence
                    + transport_step.free_horizontal_convergence_w_m2
                ) * _bl_area_rows[:, None]
            )
            / (Wc * np.sum(_bl_area_rows))
        )

    # --- Radiative Balance (Physics Item 1, 2, 10, 12) ---
    # Incoming Solar (S_in) - Albedo (A)
    # A depends on: Land/Ocean, Snow/Ice, Cloud
    
    # Approx Latitude
    lat = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi
    lat_2d = np.repeat(lat[:, None], Wc, axis=1)
    
    # === PASS 2: T_sst dynamics — radiation, relaxation, ocean transport ===

    _fb_t = feedback_flags or {}
    # Snow cover — use physically-tracked snow depth when available.
    # Full cover at ≥0.1 m SWE (≈0.3 m fresh snow); falls off linearly below.
    # Fallback to temperature-derived estimate when snow_depth not yet tracked.
    if snow_depth is not None:
        snow_cover = np.clip(snow_depth / 0.1, 0.0, 1.0).astype(np.float32, copy=False)
    else:
        snow_cover = np.clip((273.15 - T_sst) / 10.0, 0.0, 1.0)
    if not _fb_t.get('snow_albedo', True):
        snow_cover = np.zeros_like(T_sst, dtype=np.float32)
    sea_ice = np.zeros_like(T_sst, dtype=np.float32) if ice_cover is None else np.clip(ice_cover.astype(np.float32, copy=False), 0.0, 1.0)
    sea_ice = np.where(sea_mask, sea_ice, 0.0)
    if ice_albedo_strength != 1.0:
        sea_ice = np.clip(sea_ice * float(ice_albedo_strength), 0.0, 1.0)
    if not _fb_t.get('ice_albedo', True):
        sea_ice = np.zeros_like(T_sst, dtype=np.float32)

    # Cloud cover — humidity lives in the atmosphere, so use T_air for Clausius-Clapeyron
    _cloud_thermodynamic_temperature = (
        np.where(land_mask, boundary_layer_out, T_air)
        if _boundary_cloud_temperature_active
        else T_air
    )
    Tc = np.clip(_cloud_thermodynamic_temperature - 273.15, -60.0, 60.0)
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))
    qsat = np.clip(0.622 * es / (_pp.surface_pressure_pa / 100.0), 1e-6, 0.035).astype(np.float32, copy=False)
    if humidity is not None:
        q = np.clip(humidity.astype(np.float32, copy=False), 0.0, qsat)
    else:
        temp_norm = np.clip(
            (_cloud_thermodynamic_temperature - 255.0) / 45.0, 0.0, 1.0
        )
        base_q = np.where(sea_mask, 0.012, 0.008).astype(np.float32, copy=False)
        q = base_q * (0.5 + 0.7 * temp_norm)
    rh = np.clip(q / qsat, 0.0, 1.5)
    if _pp.spherical_metric_clouds:
        lat_cloud = (
            0.5 - (np.arange(Hc, dtype=np.float64) + 0.5) / float(Hc)
        ) * np.pi
        div = flux_divergence_spherical(
            np.ones_like(u, dtype=np.float32),
            u,
            v,
            lat_cloud,
            radius_m=_pp.radius_m,
        )
        ascent = _normalize_positive_driver(-div)
        subsidence = _normalize_positive_driver(div)
    else:
        # Legacy flat-index operator. Row indices increase southward while v
        # is northward-positive, so physical meridional derivative is -d/drow.
        div = (
            0.5 * (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1))
            - np.gradient(v, axis=0)
        )
        ascent = np.clip(-div, 0.0, None)
        subsidence = np.clip(div, 0.0, None)
        ascent = ascent / (np.mean(ascent) + 1e-6)
        subsidence = subsidence / (np.mean(subsidence) + 1e-6)
    gx = 0.5 * (np.roll(elev_c, -1, axis=1) - np.roll(elev_c, 1, axis=1))
    # Physical northward terrain slope; row index increases toward the south.
    gy = -np.gradient(elev_c, axis=0)
    orog = np.clip(gx * u + gy * v, 0.0, None)
    orog = orog / (np.mean(orog) + 1e-6)
    rh_core = np.clip((rh - 0.65) * 2.0, 0.0, 1.0)
    ascent_term = np.clip(0.6 + 0.6 * ascent, 0.0, 1.4)
    cloud_fraction = rh_core * ascent_term
    cloud_fraction = np.clip(cloud_fraction + 0.25 * rh_core * orog, 0.0, 1.0)
    cloud_fraction = np.clip(cloud_fraction * (1.0 - 0.6 * np.clip(subsidence, 0.0, 1.0)), 0.0, 1.0)
    _cloud_fraction_instantaneous = cloud_fraction.copy()

    # Feature 1: cloud temporal persistence (~3-day memory).
    # Blend freshly-diagnosed cloud_fraction toward the previous-step value so
    # clouds don't jump discontinuously between steps.
    if prev_cloud_cover is not None and _fb_t.get('cloud_feedback', True):
        tau_cloud_days = 3.0
        if _boundary_cloud_memory_active:
            _daily_alpha = min(1.0 / tau_cloud_days, 0.5)
            alpha = float(1.0 - (1.0 - _daily_alpha) ** float(days))
        else:
            alpha = float(np.clip(days / tau_cloud_days, 0.0, 0.5))
        cloud_fraction = (
            alpha * cloud_fraction + (1.0 - alpha) * prev_cloud_cover.astype(np.float32, copy=False)
        )
        cloud_fraction = cloud_fraction.astype(np.float32, copy=False)
    _cloud_fraction_after_persistence = cloud_fraction.copy()

    # Cloud <-> precipitation feedback: heavy rain rains a cloud out, so it shouldn't
    # persist at full cover into the next step. `precipitation` is last step's
    # coarsened rain rate [mm/day]; it was previously accepted here but unused,
    # leaving cloud_fraction and precipitation diagnosed independently even though
    # they share the same RH/ascent drivers (a gap noted in PLAN.md). 20 mm/day is a
    # heavy rain rate; deplete_frac saturates at 30% so persistent stratiform cloud
    # sheets don't get wiped out by their own drizzle (kept gentle deliberately —
    # this is a secondary coupling, not the primary cloud-cover driver).
    if precipitation is not None and _fb_t.get('cloud_feedback', True):
        _daily_rain_deplete = np.clip(
            precipitation.astype(np.float32, copy=False) / 20.0, 0.0, 0.30
        )
        if _boundary_cloud_memory_active:
            rain_deplete = 1.0 - (1.0 - _daily_rain_deplete) ** float(days)
        else:
            rain_deplete = _daily_rain_deplete
        cloud_fraction = np.clip(cloud_fraction * (1.0 - rain_deplete), 0.0, 1.0).astype(np.float32, copy=False)
    else:
        rain_deplete = np.zeros_like(cloud_fraction)
    _cloud_fraction_after_rainout = cloud_fraction.copy()

    # Feature: prognostic cloud water (Jul 2026). A real liquid-water mass
    # budget layered on top of the RH-diagnosed cloud_fraction above -- gives
    # clouds memory of their own condensed water instead of being
    # re-diagnosed from scratch every step. Condensation source scales with
    # the same rh_core/ascent_term driving the diagnostic; sinks are rain-out
    # (reuses rain_deplete), evaporation into dry air (1-rh), and a baseline
    # droplet-settling/entrainment sink (see below). Blended into
    # cloud_fraction via cloud_water_feedback (0.0 = pure diagnostic,
    # bit-identical to before); cloud_water_new is still tracked at 0.0 so a
    # later run enabling the feedback doesn't cold-start from zero memory.
    #
    # Calibration note: an earlier version without the baseline sink let
    # cloud_water grow unboundedly in perpetually-humid, non-raining cells
    # (both other sinks vanish there), saturating at its hard clip ceiling
    # and making the feedback inflate mean cloud cover (measured 0.17->0.56
    # at w=0.5) rather than smooth it. With the baseline sink added and
    # cw_ref recalibrated to the measured equilibrium magnitude, real-terrain
    # verification (saves/earth.pkl, 50-day continuation, discarding the
    # first 20 days as spin-up transient) shows day-to-day cloud_cover
    # variance actually decreasing with blend weight as intended: std of
    # day-to-day change 0.00098 (w=0) -> 0.00082 (w=0.5) -> 0.00075 (w=1.0),
    # ~23% smoother at full feedback, with mean cloud cover drifting down
    # only modestly (0.171->0.157).
    _k_cond = 0.02        # kg/kg per day, condensation source strength
    _k_rain_sink = 1.0    # 1/day, rain-out sink rate at full rain_deplete
    _k_evap_sink = 0.5    # 1/day, evaporation sink rate into fully dry air
    # Baseline sink (droplet settling/entrainment) so cloud_water can't grow
    # unboundedly in perpetually-humid, non-raining cells where the other two
    # sinks both vanish -- without this, cloud_water only stops climbing at
    # its hard clip ceiling below, an artificial saturation rather than a
    # real equilibrium. Reuses the same 3-day timescale as this function's
    # own cloud-persistence blend (tau_cloud_days above) for consistency.
    _k_base_sink = 1.0 / 3.0  # 1/day
    _cw_ref = 0.08        # kg/kg reference cloud water <-> cloud_fraction=1
    _S_cond = _k_cond * rh_core * ascent_term
    _sink_rate = _k_base_sink + _k_rain_sink * rain_deplete + _k_evap_sink * (1.0 - rh)
    # Cold-start: a genuinely fresh state has no cloud_water history. Seeding it
    # at 0.0 (as if driest-possible) makes the very first several days' blended
    # cloud_fraction crater toward 0 regardless of the actual diagnostic value,
    # since cloud_water_new needs several sink-timescales to climb from zero to
    # its equilibrium -- measured to collapse a 5-day fresh-start mean cloud
    # fraction from ~0.25 (diagnostic-only) to 0.075 at cloud_water_feedback=0.5.
    # Seed instead from the *current* diagnosed cloud_fraction (backing out the
    # cloud_water value that would reproduce it via the cw_ref scaling below) so
    # the first blended step is consistent with the diagnostic, not a cold pipe.
    _prev_cw = (prev_cloud_water.astype(np.float32, copy=False) if prev_cloud_water is not None
                else cloud_fraction * _cw_ref)
    # Exact solution of dcw/dt = S_cond - sink_rate*cw (not a forward-Euler-style
    # `prev*exp(-sink*dt) + S_cond*dt`, which only approximates this ODE for
    # small dt). At MONTHLY/ANNUAL cadence (dt ~ 30 days) with sink_rate of a
    # few per day, sink_rate*dt >> 1, so the decay term for the *old* value
    # correctly vanishes but the source term `S_cond*dt` was still being added
    # un-decayed -- growing linearly with dt instead of saturating at the
    # steady state S_cond/sink_rate. Measured: a 60yr MONTHLY real-terrain
    # spinup drove mean cloud_cover to 0.59 (w=0.5) / 0.79 (w=1.0) from a 0.25
    # baseline, reproducing the exact runaway this feature's calibration note
    # above says was already fixed -- that check only ran a short DAILY-cadence
    # continuation, where sink_rate*dt << 1 and the bug is invisible. The exact
    # solution below reduces to the original formula in that same small-dt
    # limit (verified: both are cw0*(1-sink*dt) + S_cond*dt to first order) but
    # stays correctly bounded at any cadence, since cw_eq is itself bounded
    # (~0.084 kg/kg at the highest physically-reachable S_cond/lowest
    # sink_rate, essentially cw_ref) rather than growing with dt.
    _cw_eq = _S_cond / _sink_rate
    cloud_water_new = np.clip(
        _cw_eq + (_prev_cw - _cw_eq) * np.exp(-_sink_rate * days), 0.0, 5.0 * _cw_ref
    ).astype(np.float32, copy=False)
    _w_cw = float(_pp.cloud_water_feedback)
    if _w_cw > 0.0:
        cloud_fraction = np.clip(
            (1.0 - _w_cw) * cloud_fraction + _w_cw * np.clip(cloud_water_new / _cw_ref, 0.0, 1.0),
            0.0, 1.0
        ).astype(np.float32, copy=False)

    # The experimental vertical closure carries suspended condensate in its
    # midlevel reservoir.  Feed that *actual persistent mass* into cloud
    # fraction before radiative terms are diagnosed, rather than asking an
    # unrelated RH proxy to stand in for anvil/stratiform cloud.  It remains
    # gated, so default cloud behavior is exactly unchanged.
    if (
        bool(_pp.enable_prognostic_column_water)
        and bool(_pp.enable_stability_aware_condensation)
        and bool(_pp.enable_two_layer_convective_adjustment)
        and midlevel_condensate is not None
    ):
        _midlevel_condensate = np.asarray(midlevel_condensate, dtype=np.float32)
        if _midlevel_condensate.shape != cloud_fraction.shape:
            raise ValueError("midlevel_condensate must match temperature grid")
        # Bulk condensate contains both suspended cloud water and falling
        # hydrometeors.  When the experimental partition is active, only the
        # bounded suspended portion participates in optical cloud cover; the
        # remaining mass stays in the same conserved fallout reservoir.
        if bool(_pp.enable_cloud_precipitating_condensate_partition):
            _midlevel_condensate = np.minimum(
                _midlevel_condensate,
                float(max(_pp.cloud_optical_condensate_cap_q, 1e-8)),
            )
        _condensate_cloud = np.clip(
            _midlevel_condensate / float(_pp.two_layer_cloud_reference_q),
            0.0,
            1.0,
        )
        _condensate_cloud = float(_pp.two_layer_cloud_radiative_weight) * _condensate_cloud
        cloud_fraction = np.clip(
            1.0 - (1.0 - cloud_fraction) * (1.0 - _condensate_cloud),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

    # Albedo (with vegetation feedback - Phase 4)
    # Ocean: 0.06, Sea Ice: 0.75, Snow: 0.8, Cloud: 0.5
    # Land albedo now depends on vegetation/biome type

    # Compute biome-based vegetation albedo (if biomass field exists)
    # Phase 1 improvement: Use stable biomes from long-term climate averages (not daily weather)
    # Köppen classification provides more detailed albedo values
    # Feature 6: thickness-dependent ocean albedo.
    # At h≥0.5m (thick ice): same as old formula — no regression.
    # At h<0.5m (thin/new ice): lower albedo prevents summer ice-albedo runaway.
    # alpha_ice(h) = 0.06 + 0.59 * min(h / 0.5, 1.0)  [0.06 open water → 0.65 thick ice]
    if ice_thickness is not None:
        _alpha_ice = (0.06 + 0.59 * np.minimum(ice_thickness / 0.5, 1.0)).astype(np.float32, copy=False)
        _alpha_sea = ((1.0 - sea_ice) * 0.06 + sea_ice * _alpha_ice).astype(np.float32, copy=False)
    else:
        _alpha_sea = (0.06 * (1.0 - sea_ice) + 0.65 * sea_ice).astype(np.float32, copy=False)

    if vegetation_biomass is not None and biome is not None and _fb_t.get('vegetation_albedo', True):
        albedo_veg = vegetation_albedo(biome, base_land_albedo=0.2, koppen_type=koppen_type)
        albedo_sfc = np.where(sea_mask, _alpha_sea, albedo_veg)
    else:
        albedo_sfc = np.where(sea_mask, _alpha_sea, 0.2)

    # Snow albedo overrides vegetation (snow is brighter)
    snow_cover_land = snow_cover * land_mask.astype(np.float32)
    albedo_sfc = np.where(land_mask, albedo_sfc * (1.0 - snow_cover_land) + 0.8 * snow_cover_land, albedo_sfc)

    # Total albedo including clouds
    albedo_total = albedo_sfc * (1 - cloud_fraction) + 0.5 * cloud_fraction
    
    # Insolation Q (Daily mean) - use proper astronomical calculation
    _pp = planet_params if planet_params is not None else EARTH
    decl = _pp.solar_declination(day_of_year)
    S0 = _pp.effective_solar_constant(day_of_year)

    # Clamp to avoid domain errors in polar regions
    lat_safe = np.clip(lat_2d, -np.pi/2 + 1e-6, np.pi/2 - 1e-6)
    cos_h = np.clip(-np.tan(lat_safe) * np.tan(decl), -1.0, 1.0)
    h = np.arccos(cos_h) # hour angle radians (0 to pi)
    h = np.where(cos_h <= -1.0, np.pi, h)  # 24h daylight
    h = np.where(cos_h >= 1.0, 0.0, h)     # polar night

    Q = S0 * (1.0/np.pi) * (h * np.sin(lat_safe)*np.sin(decl) + np.cos(lat_safe)*np.cos(decl)*np.sin(h))
    Q = np.maximum(0.0, Q)
    
    S_absorbed = np.maximum(0.0, Q * (1.0 - albedo_total) + _pp.aerosol_forcing_w_m2)
    _S_absorbed_without_cloud = np.maximum(
        0.0, Q * (1.0 - albedo_sfc) + _pp.aerosol_forcing_w_m2
    )
    _cloud_shortwave_forcing = S_absorbed - _S_absorbed_without_cloud
    
    # Outgoing Longwave (L_out) = sigma * T^4 * epsilon
    # Greenhouse effect reduces OLR - use latitude-dependent epsilon (like temperature.py)
    abs_lat_deg = np.rad2deg(np.abs(lat_2d))
    epsilon_equator = float(epsilon_equator)  # Increased from 0.75 to match temperature.py and warm global mean
    epsilon_pole = float(epsilon_pole)     # Increased from 0.50 to reduce polar extremes
    lat_factor = np.cos(np.deg2rad(abs_lat_deg))  # 1.0 at equator, 0.0 at poles
    epsilon = epsilon_pole + (epsilon_equator - epsilon_pole) * lat_factor
    _epsilon_without_cloud = np.asarray(epsilon).copy()
    
    # Feature 1: cloud greenhouse on OLR.
    # High clouds (cold tops) trap outgoing longwave; low/warm clouds do not.
    # T_air proxy: colder air column → higher cloud tops → stronger LW trapping.
    if _fb_t.get('cloud_feedback', True) and _pp.cloud_greenhouse_factor > 0.0:
        cloud_high_weight = np.clip((265.0 - T_air) / 20.0, 0.0, 1.0).astype(np.float32, copy=False)
        epsilon_cloud_ghg = _pp.cloud_greenhouse_factor * cloud_fraction * cloud_high_weight
        epsilon = np.clip(epsilon - epsilon_cloud_ghg, 0.30, 0.95).astype(np.float32, copy=False)

    # Feature 2: water vapour greenhouse on OLR (applied after cloud term).
    if _fb_t.get('water_vapor_feedback', True) and _pp.wv_greenhouse_factor > 0.0 and humidity is not None:
        # rh already computed above; higher RH → more WV → lower effective epsilon
        wv_reduction = _pp.wv_greenhouse_factor * np.clip(rh - 0.5, 0.0, 1.0).astype(np.float32, copy=False)
        epsilon = np.clip(epsilon - wv_reduction, 0.30, 0.95).astype(np.float32, copy=False)
        _epsilon_without_cloud = np.clip(
            _epsilon_without_cloud - wv_reduction, 0.30, 0.95
        ).astype(np.float32, copy=False)

    sigma = STEFAN_BOLTZMANN
    # Longwave emission is from the surface (T_sst drives outgoing radiation)
    _radiative_surface_temperature = T_sst.copy()
    _radiative_air_temperature = T_air.copy()
    L_out = epsilon * sigma * (_radiative_surface_temperature ** 4)
    _effective_radiating_temperature = np.power(
        np.maximum(L_out, 0.0) / sigma, 0.25
    )
    _L_out_without_cloud = _epsilon_without_cloud * sigma * (T_sst ** 4)
    _cloud_longwave_forcing = _L_out_without_cloud - L_out
    _cloud_net_radiative_forcing = (
        _cloud_shortwave_forcing + _cloud_longwave_forcing
    )

    R_net = S_absorbed - L_out  # W/m²

    _grey = None
    _optical_closure = None
    _middle_emission_temperature = None
    _upper_emission_temperature = None
    _coupled_middle_temperature = None
    _coupled_upper_temperature = None
    _grey_air_temperature_increment = None
    if _pp.enable_pressure_defined_radiative_temperature_profile:
        _diagnosed_profile = pressure_defined_temperature_profile(
            _radiative_air_temperature,
            float(_pp.surface_pressure_pa),
            float(_pp.two_layer_pressure_depth_pa),
            float(_pp.three_level_mid_upper_pressure_depth_pa),
            gas_constant_dry_air_j_kg_k=float(_pp.gas_constant_dry),
            cp_dry_air_j_kg_k=float(_pp.cp_dry),
        )
        _middle_source = (
            _diagnosed_profile.midlevel_temperature_k
            if radiative_midlevel_temperature is None
            else radiative_midlevel_temperature
        )
        _upper_source = (
            _diagnosed_profile.upperlevel_temperature_k
            if radiative_upperlevel_temperature is None
            else radiative_upperlevel_temperature
        )
        _middle_emission_temperature = resolved_midlevel_emission_temperature(
            _middle_source, expected_shape=_radiative_surface_temperature.shape
        )
        _upper_emission_temperature = resolved_midlevel_emission_temperature(
            _upper_source, expected_shape=_radiative_surface_temperature.shape
        )
        if (
            _pp.enable_coupled_two_layer_grey_radiation
            and radiative_optical_depth is not None
        ):
            _persisted_optical_depth = np.asarray(
                radiative_optical_depth, dtype=np.float64
            )
            if _persisted_optical_depth.shape != _radiative_surface_temperature.shape:
                raise ValueError("persisted grey optical depth has an unexpected shape")
            _split_emissivity = pressure_split_emissivities_from_optical_depth(
                _persisted_optical_depth,
                float(_pp.two_layer_pressure_depth_pa),
                float(_pp.three_level_mid_upper_pressure_depth_pa),
            )
            _optical_closure = TwoLayerGreyOpticalClosure(
                _persisted_optical_depth,
                _split_emissivity.midlevel_emissivity,
                _split_emissivity.upperlevel_emissivity,
                _persisted_optical_depth >= 64.0,
                np.zeros_like(_persisted_optical_depth),
            )
        else:
            _optical_closure = two_layer_optical_depth_for_target_olr(
                _radiative_surface_temperature,
                _middle_emission_temperature,
                _upper_emission_temperature,
                L_out,
                float(_pp.two_layer_pressure_depth_pa),
                float(_pp.three_level_mid_upper_pressure_depth_pa),
                allow_opaque_limit=True,
            )
        _grey = two_layer_grey_radiation(
            _radiative_surface_temperature,
            _middle_emission_temperature,
            _upper_emission_temperature,
            S_absorbed,
            _optical_closure.midlevel_emissivity,
            _optical_closure.upperlevel_emissivity,
        )
        _optical_closure = _optical_closure._replace(
            target_olr_residual_w_m2=_grey.outgoing_longwave_w_m2 - L_out
        )
        # In coupled mode the grey atmospheric gains are applied later by the
        # closed three-level moist-static-energy column. Applying them to the
        # host air temperature here would create a second energy owner.

    # Radiative equilibrium temperature for the surface
    T_eq_rad = equilibrium_temperature_k(S_absorbed, epsilon, sigma=sigma)
    T_eq_rad = np.clip(T_eq_rad, 150.0, 350.0)
    
    # CRITICAL FIX: Blend radiation equilibrium with base temperature
    # T_base comes from temperature_kelvin_for_lat which has proper polar cooling physics
    # We should trust it more, especially at poles where radiation-only calculation fails
    # Use a mostly base-driven blend, but let radiation pull more strongly than before
    # so broad imposed warmth in T_base does not dominate the equilibrium.
    T_eq = 0.90 * T_base + 0.10 * T_eq_rad

    # --- Orographic cooling (lapse rate) ---
    # Previously missing: high terrain never cooled as a function of altitude.
    # Apply to equilibrium temperature so radiation relaxes toward a colder state aloft.
    # Lapse rate and the elevation->altitude ceiling both come from PlanetParams
    # (Earth defaults 6.5 K/km and 8.848 km reproduce the previous hardcoded
    # constants exactly) -- see ACCURACY_AUDIT.md C3.
    lapse_rate = float(_pp.lapse_rate_k_per_km)  # K/km
    alt_km = elevation_to_alt_km(elev_c, max_elevation_km=float(_pp.max_elevation_km))
    T_eq = T_eq - lapse_rate * alt_km

    # --- Ocean Temperature Bounds (SST cap + freeze-point floor) ---
    # These bounds parameterize Earth's liquid-water ocean physics; suppress for dry planets.
    if _pp.has_liquid_water_ocean:
        # Upper cap (SST max ~29°C = 302K): evaporative cooling prevents ocean from exceeding
        # this in practice. The radiative equilibrium formula ignores latent heat and would
        # otherwise push subtropical SSTs to 43-48°C.  Earth's tropical zonal-mean SST is
        # 28-29°C; 302K matches this while preventing runaway tropical heating.
        T_eq = np.where(sea_mask, np.minimum(T_eq, 302.0), T_eq)
        # Latitude-dependent T_eq ocean floor — prevents ice-albedo runaway at mid-latitudes
        # while still allowing polar sea ice:
        #
        # At mid-latitudes (|lat| ≤ 60°): floor = 271K (above ice_freeze_temp=269.9K).
        #   Without this, ice forming at 55°N (albedo→0.65) drives T_eq_rad→140K, so
        #   T_eq = 0.9×278.75 + 0.1×140 = 265.9K < freeze → ice-albedo runaway to 52°N.
        #   With floor=271K the equilibrium always pulls T_sst above freezing → ice melts. ✓
        #
        # At high latitudes (|lat| ≥ 75°): floor = 266K (below freeze_temp).
        #   Allows genuine Arctic/Antarctic sea ice to form and persist. ✓
        #
        # Linear ramp between 60° and 75° so there is no sharp boundary.
        _abs_lat_floor = np.abs(np.rad2deg(lat_2d))
        _ramp = np.clip((_abs_lat_floor - 60.0) / 15.0, 0.0, 1.0)   # 0 at 60°, 1 at 75°+
        T_eq_floor = (271.0 * (1.0 - _ramp) + 266.0 * _ramp).astype(np.float32, copy=False)
        T_eq = np.where(sea_mask, np.maximum(T_eq, T_eq_floor), T_eq)

    # Relaxation rate k (1/days) based on mixed-layer depth
    # Real oceans have latitude-dependent mixed layer depth:
    #   Tropics: ~30-50m (thin thermocline, trade winds)
    #   Mid-latitudes: ~50-150m (seasonal deepening)
    #   High latitudes: ~200-500m (deep convective mixing in winter)
    # Deeper mixed layers = more thermal inertia = slower response to forcing
    abs_lat_1d = np.abs(np.rad2deg(lat))  # lat computed at line 948
    abs_lat_2d_relax = np.repeat(abs_lat_1d[:, None], Wc, axis=1)
    _mld_trop = float(_pp.mixed_layer_depth_tropical_m)
    _mld_polar = float(_pp.mixed_layer_depth_polar_m)
    mld = _mld_trop + (_mld_polar - _mld_trop) * (abs_lat_2d_relax / 90.0) ** 1.5
    # Seasonal polar MLD reduction: Arctic/Antarctic meltwater halocline creates a
    # shallow warm layer (~20-30m) in summer, allowing rapid surface warming → ice melt.
    # Without this, the ~186m polar MLD gives a 93-day thermal time constant — too slow
    # for summer T to reach ice_melt_temp=260K during the 90-day Arctic summer.
    _summer_solstice_day = (172.0 / 365.2422) * float(_pp.orbital_period_days)
    _gamma_mld = 2.0 * np.pi * (float(day_of_year) - _summer_solstice_day) / float(_pp.orbital_period_days)
    _nh_summer = float(0.5 * (1.0 + np.cos(_gamma_mld)))   # 1.0 at NH summer, 0 at NH winter
    _sh_summer = float(0.5 * (1.0 - np.cos(_gamma_mld)))   # 1.0 at SH summer (day ~355)
    _hemi_summer = np.where(lat_2d >= 0, _nh_summer, _sh_summer)  # (Hc, Wc)
    _polar_ramp = np.clip((abs_lat_2d_relax - 55.0) / 30.0, 0.0, 1.0)  # 0 at 55°, 1 at 85°+
    # Up to 50% MLD reduction at poles in polar summer → ~94m → 47-day time constant
    # 80% was too aggressive: T85N reached +15°C in summer (unrealistic).
    # 50% allows meaningful summer warming (3 time constants over 90-day Arctic summer)
    # without overshooting. Target: summer T85N near 0°C, not +15°C.
    mld = mld * (1.0 - 0.50 * _polar_ramp * _hemi_summer)
    k_ocean = np.clip(1.0 / (mld * 0.5), 0.005, 0.07)  # 14-200 day time constants

    # Ice insulates T_sst against cooling (ice-ocean decoupling in winter) but not warming
    cooling_direction = T_eq < T_sst
    ice_insulation = np.where(
        cooling_direction,
        1.0 - 0.7 * sea_ice,
        1.0,
    )
    k_relax = np.where(sea_mask, k_ocean * ice_insulation, 0.1)

    # Relax T_sst toward radiative+transport equilibrium.
    # Fraction capped at 0.5 for unconditional stability at any dt (large-step modes
    # like MONTHLY/ANNUAL use dt=6-7 days where k_relax*dt can exceed 1 without this).
    _sst_frac = np.minimum(k_relax * float(days), 0.5).astype(np.float32, copy=False)
    if _pp.enable_coupled_two_layer_grey_radiation:
        assert _grey is not None
        _surface_capacity = np.where(
            sea_mask,
            4.186e6 * mld,
            float(_pp.land_surface_heat_capacity_j_m2_k),
        )
        _direct_surface_mask = (
            sea_mask if _pp.enable_force_restore_land else np.ones_like(sea_mask)
        )
        T_sst = (
            T_sst
            + np.where(
                _direct_surface_mask,
                _grey.surface_gain_w_m2 * dt_sec / _surface_capacity,
                0.0,
            )
        ).astype(np.float32, copy=False)
    else:
        T_sst = (T_sst + _sst_frac * (T_eq - T_sst)).astype(np.float32, copy=False)
    
    # --- Evaporation: bulk aerodynamic formula, applied to T_sst ---
    # Ocean evaporation depends on SST (surface saturation) and near-surface air humidity.
    if wind_u is not None and wind_v is not None and humidity is not None:
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        # Saturation humidity at SST (the surface provides moisture to the air)
        T_c_sst = np.clip(T_sst - 273.15, -60.0, 60.0)
        es_sst = 6.112 * np.exp(17.67 * T_c_sst / (T_c_sst + 243.5))
        qsat_sst = np.clip(0.622 * es_sst / (_pp.surface_pressure_pa / 100.0), 1e-6, 0.035)
        deficit = np.maximum(0.0, qsat_sst - humidity)
        C_D = np.where(sea_mask, 1.5e-3, 0.5e-3)
        E = C_D * wind_speed * deficit * 1000.0
        E = np.clip(E, 0.0, 20.0)
        evap_cooling = E * 2.5 * float(days)
        # Enhanced evaporative cooling for hot SSTs
        hot_ocean_excess = np.where(sea_mask & (T_sst > 303.0), T_sst - 303.0, 0.0)
        evap_cooling = evap_cooling + 0.3 * hot_ocean_excess * float(days)
        T_sst = (T_sst - evap_cooling).astype(np.float32, copy=False)
    else:
        base_evap = np.where(sea_mask, 0.01 * (T_sst - 270.0), 0.0)
        hot_evap = np.where((T_sst > 303.0) & sea_mask, 0.3 * (T_sst - 303.0), 0.0)
        evap_cooling = np.maximum(0.0, base_evap + hot_evap)
        T_sst = (T_sst - evap_cooling * float(days)).astype(np.float32, copy=False)
    
    # --- Land surface blend toward seasonal baseline ---
    if T_base_land is not None and not bool(_pp.enable_force_restore_land):
        T_base_land = T_base_land - lapse_rate * alt_km
        # This is the land's *only* thermal inertia, and 0.2 is a fraction per
        # CALL rather than per day -- so its effective time constant is set by
        # whatever step length the caller happens to use. Over a 12-day span
        # this keeps 0.800 of the prior temperature integrated as one step and
        # 0.069 integrated as twelve, from identical physics. Same defect class
        # as the monsoon mask's fixed cell count: a physical scale expressed in
        # units of the discretisation.
        #
        # `land_thermal_inertia_days > 0` converts it to `1 - exp(-dt/tau)`,
        # which makes *this term* split-invariant exactly. It does NOT make
        # `simulate_step` step-length invariant end to end -- measured, the
        # 12-way-split land discrepancy goes 4.34 K -> 5.44 K, because the
        # residual is dominated by other terms that scale linearly in `days`.
        # 0.0 keeps the historical constant and is an exact no-op.
        _tau_land = float(_pp.land_thermal_inertia_days)
        if _tau_land > 0.0:
            _land_rate = float(1.0 - np.exp(-float(days) / _tau_land))
        else:
            _land_rate = 0.2
        land_blend = np.where(land_mask, _land_rate, 0.0)
        T_sst = ((1.0 - land_blend) * T_sst + land_blend * T_base_land).astype(np.float32, copy=False)

    # --- Gated prognostic land surface-energy closure ---
    # The baseline branch above supplies the calibrated seasonal large-scale
    # forcing. This optional tendency adds the missing local closure: net
    # radiation is partitioned between sensible and latent heat, then stored in
    # a finite active land layer. It distinguishes dry/hot land from humid or
    # windy land at the same latitude without changing the default path.
    if bool(_pp.enable_land_surface_energy) and not bool(_pp.enable_force_restore_land):
        if wind_u is not None and wind_v is not None and humidity is not None:
            wind_speed_land = np.sqrt(wind_u**2 + wind_v**2)
            T_c_land = np.clip(T_sst - 273.15, -60.0, 60.0)
            es_land = 6.112 * np.exp(17.67 * T_c_land / (T_c_land + 243.5))
            qsat_land = np.clip(
                0.622 * es_land / (_pp.surface_pressure_pa / 100.0), 1e-6, 0.035
            )
            vapor_deficit = np.maximum(0.0, qsat_land - humidity)
            rho_air = 1.2
            bulk_exchange = 1.3e-3
            sensible_flux = (
                rho_air * float(_pp.cp_dry) * bulk_exchange * wind_speed_land * (T_sst - T_air)
            )
            latent_flux = (
                rho_air * 2.5e6 * bulk_exchange * wind_speed_land * vapor_deficit
            )
            net_land_flux = R_net - sensible_flux - latent_flux
            heat_capacity = max(float(_pp.land_surface_heat_capacity_j_m2_k), 1.0e4)
            land_tendency = np.clip(
                net_land_flux * float(days) * 86400.0 / heat_capacity,
                -6.0,
                6.0,
            )
            T_sst = (
                T_sst
                + land_mask.astype(np.float32)
                * float(_pp.land_surface_energy_strength)
                * land_tendency
            ).astype(np.float32, copy=False)

    # --- Gated force-restore / Penman--Monteith replacement path ---
    # This path is intentionally mutually exclusive with the legacy seasonal
    # baseline blend and cap.  It is an A/B-able replacement, not another term
    # calibrated around mechanisms it is meant to retire.
    if bool(_pp.enable_force_restore_land) and float(days) > 0.0:
        from land_surface import force_restore_penman_monteith

        soil_for_land = (
            np.full_like(T_sst, 0.55, dtype=np.float32)
            if soil_moisture is None else np.clip(soil_moisture, 0.0, 1.0)
        )
        # The slow/root-zone bucket (`state.soil_moisture_deep`) was previously
        # not threaded into this call at all, so the deep reservoir's own
        # thermal properties and the dry-resistance term never saw it -- see
        # `land_surface.force_restore_penman_monteith`'s moisture-scaled heat
        # capacity and root-zone resistance blend. `None` here falls back to
        # the shallow bucket inside that function, reproducing the old
        # behaviour exactly when the deep bucket is unavailable (e.g. before
        # it has spun up).
        soil_deep_for_land = (
            None if soil_moisture_deep is None else np.clip(soil_moisture_deep, 0.0, 1.0)
        )
        wind_for_land = np.sqrt(u * u + v * v)
        land_net_forcing = (
            _grey.surface_gain_w_m2
            if _pp.enable_coupled_two_layer_grey_radiation
            else R_net
        )
        resolved_heat_convergence = None
        resolved_heat_convergence_area_mean = None
        if bool(_pp.enable_force_restore_atmospheric_heat_convergence):
            from atmospheric_heat_transport import (
                close_global_heat_convergence,
                temperature_transport_to_heat_convergence,
            )

            resolved_heat_convergence = temperature_transport_to_heat_convergence(
                resolved_transport_delta_k,
                surface_pressure_pa=float(_pp.surface_pressure_pa),
                cp_j_kg_k=float(_pp.cp_dry),
                gravity_m_s2=float(_pp.surface_gravity),
                dt_seconds=dt_sec,
            )
            resolved_heat_convergence = close_global_heat_convergence(
                resolved_heat_convergence, lat_1d
            )
            _heat_weights = np.cos(lat_1d.astype(np.float64))[:, None]
            resolved_heat_convergence_area_mean = float(
                np.sum(resolved_heat_convergence.astype(np.float64) * _heat_weights)
                / (resolved_heat_convergence.shape[1] * np.sum(_heat_weights))
            )
            # The mixed-layer experiment leaves resolved convergence in the
            # transported free atmosphere.  Reapplying the diagnosed transport
            # increment at the surface would count the same energy twice.
            if not _boundary_layer_active:
                land_net_forcing = R_net + resolved_heat_convergence

        if _boundary_layer_active:
            if boundary_layer_out is None or np.asarray(boundary_layer_out).shape != T_air.shape:
                boundary_layer_out = T_air.copy()
            air_for_land = np.asarray(boundary_layer_out, dtype=np.float32)
        else:
            air_for_land = T_air

        land_step = force_restore_penman_monteith(
            T_sst, air_for_land, land_deep_temperature, soil_for_land, wind_for_land,
            land_net_forcing, land_mask, dt_days=float(days),
            surface_heat_capacity_j_m2_k=float(_pp.land_surface_heat_capacity_j_m2_k),
            deep_heat_capacity_j_m2_k=float(_pp.land_deep_heat_capacity_j_m2_k),
            restore_days=float(_pp.land_force_restore_days),
            surface_resistance_min_s_m=float(_pp.land_surface_resistance_min_s_m),
            surface_resistance_dry_s_m=float(_pp.land_surface_resistance_dry_s_m),
            soil_moisture_deep=soil_deep_for_land,
        )
        T_sst = land_step.temperature
        land_deep_out = land_step.deep_temperature
        conservative_sensible_gain = None
        boundary_exchange_gain = None
        boundary_interface_upper_gain = None
        if _boundary_layer_active:
            if _boundary_interface_active:
                from boundary_layer import step_boundary_layer_interface_energy

                boundary_step = step_boundary_layer_interface_energy(
                    boundary_layer_out,
                    boundary_interface_out,
                    T_air,
                    land_step.sensible_heat_w_m2,
                    land_mask,
                    surface_pressure_pa=float(_pp.surface_pressure_pa),
                    cp_j_kg_k=float(_pp.cp_dry),
                    gravity_m_s2=float(_pp.surface_gravity),
                    gas_constant_j_kg_k=float(_pp.gas_constant_dry),
                    reference_temperature_k=float(_pp.boundary_layer_reference_temperature_k),
                    mixed_layer_depth_m=float(_pp.boundary_layer_mixed_depth_m),
                    entrainment_velocity_m_s=float(
                        _pp.boundary_layer_entrainment_velocity_m_s
                    ),
                    dt_seconds=dt_sec,
                    wind_speed_m_s=wind_for_land,
                )
                boundary_interface_out = boundary_step.interface_temperature
                boundary_interface_upper_gain = (
                    boundary_step.interface_free_gain_w_m2
                )
            else:
                from boundary_layer import step_boundary_layer_energy

                boundary_step = step_boundary_layer_energy(
                    boundary_layer_out,
                    T_air,
                    land_step.sensible_heat_w_m2,
                    land_mask,
                    surface_pressure_pa=float(_pp.surface_pressure_pa),
                    cp_j_kg_k=float(_pp.cp_dry),
                    gravity_m_s2=float(_pp.surface_gravity),
                    gas_constant_j_kg_k=float(_pp.gas_constant_dry),
                    reference_temperature_k=float(_pp.boundary_layer_reference_temperature_k),
                    mixed_layer_depth_m=float(_pp.boundary_layer_mixed_depth_m),
                    entrainment_velocity_m_s=float(
                        _pp.boundary_layer_entrainment_velocity_m_s
                    ),
                    dt_seconds=dt_sec,
                    wind_speed_m_s=wind_for_land,
                    stability_dependent_exchange=bool(
                        _pp.enable_boundary_layer_stability_dependent_exchange
                    ),
                    exchange_mask=land_mask,
                )
            boundary_layer_out = boundary_step.boundary_temperature
            T_air = boundary_step.free_temperature
            conservative_sensible_gain = boundary_step.surface_gain_w_m2
            boundary_exchange_gain = boundary_step.exchange_gain_w_m2
        elif _conservative_land_air:
            from atmospheric_heat_transport import apply_sensible_heat_to_atmospheric_column

            T_air, conservative_sensible_gain = apply_sensible_heat_to_atmospheric_column(
                T_air,
                land_step.sensible_heat_w_m2,
                land_mask,
                surface_pressure_pa=float(_pp.surface_pressure_pa),
                cp_j_kg_k=float(_pp.cp_dry),
                gravity_m_s2=float(_pp.surface_gravity),
                dt_seconds=dt_sec,
            )

    # --- Equal-and-opposite air–surface sensible heat exchange (OCEAN ONLY) ---
    # The land branch of `k_couple` was 0.25 until 2026-07-25. Combined with
    # H_surf=0.35 that relaxed the *land surface* toward the air at ~0.71/day --
    # a coupling direction that did not exist before d8631cb (the pre-d8631cb
    # term was `np.where(sea_mask, ..., 0.0)`, i.e. ocean-only by construction).
    # Measured effect of the land branch on the standard 64x128 spinup fixture:
    #   land 0.25 -> global mean 292.6 K, equator-pole gradient 18.3 K, ice 0.022
    #   land 0.0  -> global mean 289.4 K, equator-pole gradient 32.3 K, ice 0.189
    # i.e. it alone caused a +3.2 K global warm bias, halved the meridional
    # gradient, and removed ~90% of NH sea ice, breaking 9 physics tests that
    # were invisible to `pytest -m "not slow"`. Over land the surface has almost
    # no heat capacity and is driven by radiation plus the seasonal-baseline
    # blend above; the air follows it via the relaxation term earlier in this
    # function, not the reverse. Ocean keeps the equal-and-opposite form, which
    # is physically right for a mixed layer with real heat capacity (the ocean
    # multiplier is measurement-neutral: 1.0 and 4.0 give identical results).
    # See overnight/FINDINGS.md (2026-07-25) for the full bisect.
    H_air = 1.0
    if _capacity_aware_airsea:
        H_surf = np.where(
            sea_mask, (4.186e6 * mld) / _column_air_capacity, 0.35
        ).astype(np.float32, copy=False)
    else:
        H_surf = np.where(
            sea_mask, np.clip(mld / 50.0, 0.3, 4.0), 0.35
        ).astype(np.float32, copy=False)
    k_couple = np.where(sea_mask, float(_pp.k_airsea) * 4.0, 0.0).astype(np.float32, copy=False)
    dT_exchange = k_couple * (T_sst - T_air) * float(days)
    dT_exchange = np.clip(dT_exchange, -5.0, 5.0).astype(np.float32, copy=False)
    _airsea_air_delta_k = dT_exchange / H_air
    _airsea_ocean_delta_k = -dT_exchange / H_surf
    T_air = (T_air + dT_exchange / H_air).astype(np.float32, copy=False)
    T_sst = (T_sst - dT_exchange / H_surf).astype(np.float32, copy=False)
    if dt_sec > 0.0:
        _airsea_air_capacity = (
            float(_pp.surface_pressure_pa) / float(_pp.surface_gravity)
            * float(_pp.cp_dry)
        )
        _airsea_ocean_capacity = 4.186e6 * mld
        _airsea_atmospheric_gain_w_m2 = np.where(
            sea_mask, _airsea_air_capacity * _airsea_air_delta_k / dt_sec, 0.0
        )
        _airsea_ocean_gain_w_m2 = np.where(
            sea_mask, _airsea_ocean_capacity * _airsea_ocean_delta_k / dt_sec, 0.0
        )
        _airsea_physical_residual_w_m2 = (
            _airsea_atmospheric_gain_w_m2 + _airsea_ocean_gain_w_m2
        )
    else:
        _airsea_atmospheric_gain_w_m2 = np.zeros_like(T_air, dtype=np.float32)
        _airsea_ocean_gain_w_m2 = np.zeros_like(T_air, dtype=np.float32)
        _airsea_physical_residual_w_m2 = np.zeros_like(T_air, dtype=np.float32)

    # --- Feature 5: Deep ocean heat uptake (heat-capacity weighted) ---
    # Mixed layer and abyss exchange equal heat fluxes; abyssal ΔT is scaled by
    # the mixed/deep heat-capacity ratio so total ocean heat content is conserved.
    T_deep_out: np.ndarray | None = T_deep_ocean
    if T_deep_ocean is not None and _pp.has_liquid_water_ocean:
        k_deep = float(_pp.deep_ocean_exchange_rate)  # ~9.1e-5 /day
        cap_ratio = float(_pp.deep_ocean_heat_capacity_ratio)
        dT_mixed = k_deep * (T_sst - T_deep_ocean) * float(days)
        dT_mixed = np.clip(dT_mixed, -0.5, 0.5).astype(np.float32, copy=False)
        ocean_f = sea_mask.astype(np.float32)
        T_sst = (T_sst - dT_mixed * ocean_f).astype(np.float32, copy=False)
        T_deep_out = (T_deep_ocean + dT_mixed * cap_ratio * ocean_f).astype(np.float32, copy=False)

    # --- Abyssal overturning (meridional mixing of the deep-ocean layer) ---
    # See PlanetParams.abyssal_overturning_coeff docstring: this model's deep
    # ocean otherwise has zero lateral transport at all, unlike the real
    # global overturning conveyor. Same Laplacian-diffusion-with-substepping
    # pattern as Feature 7 below, but applied globally (real overturning
    # isn't confined to mid-latitudes the way baroclinic eddies are) and only
    # where liquid ocean exists.
    _abyssal_k = float(_pp.abyssal_overturning_coeff)
    if _abyssal_k > 0.0 and T_deep_out is not None and _pp.has_liquid_water_ocean:
        _ocean_f_ab = sea_mask.astype(np.float32)
        _abyssal_r_limit = 0.4
        _n_abyssal_sub = max(1, int(np.ceil(_abyssal_k * float(days) / _abyssal_r_limit)))
        _dt_abyssal_sub = float(days) / _n_abyssal_sub
        for _ in range(_n_abyssal_sub):
            _T_deep_lap_y = np.zeros_like(T_deep_out)
            _T_deep_lap_y[1:-1, :] = T_deep_out[:-2, :] - 2.0 * T_deep_out[1:-1, :] + T_deep_out[2:, :]
            _T_deep_lap_y[0, :] = T_deep_out[1, :] - T_deep_out[0, :]
            _T_deep_lap_y[-1, :] = T_deep_out[-2, :] - T_deep_out[-1, :]
            _T_deep_lap_y = np.clip(_T_deep_lap_y, -20.0, 20.0).astype(np.float32, copy=False)
            T_deep_out = (
                T_deep_out + _abyssal_k * _T_deep_lap_y * _ocean_f_ab * _dt_abyssal_sub
            ).astype(np.float32, copy=False)

    # --- Feature 7: Meridional eddy heat flux ---
    # Baroclinic eddies and storm tracks transport heat poleward proportional
    # to the meridional temperature gradient.  Parameterised as Laplacian
    # diffusion in the meridional direction only, weighted to 20–70° latitudes.
    _eddy_k = float(_pp.eddy_heat_flux_coeff)
    if _eddy_k > 0.0 and _fb_t.get('eddy_heat_flux', True):
        _abs_lat_1d = np.abs(np.rad2deg(lat_1d))
        _eddy_lat = np.clip(1.0 - ((_abs_lat_1d - 45.0) / 25.0) ** 2, 0.0, 1.0).astype(np.float32, copy=False)
        # Explicit-Euler Laplacian diffusion is only stable for r = eddy_k * dt_sub
        # below ~0.5 (standard forward-difference diffusion CFL bound); beyond that
        # a single big step overshoots and amplifies grid-scale noise instead of
        # smoothing the gradient -- the same large-dt failure mode fixed elsewhere
        # via sub-stepping (atmosphere.py's 8-substep wind integration,
        # _generate_precipitation_substepped). At the default coeff (0.006) this
        # was already stable even at MONTHLY dt=30 (r=0.18), which is why it went
        # unnoticed; test_eddy_heat_flux.py's coeff=0.05 stress-test (used to get
        # a detectable 2-year signal) pushes r to 1.5 at dt=30 and was the actual
        # cause of test_eddy_flux_reduces_gradient's small negative delta -- not a
        # genuine physics conflict with ocean_transport, despite ocean_transport
        # amplifying the resulting grid-scale noise into a measurable signal.
        _eddy_r_limit = 0.4
        _n_eddy_sub = max(1, int(np.ceil(_eddy_k * float(days) / _eddy_r_limit)))
        _dt_eddy_sub = float(days) / _n_eddy_sub
        for _ in range(_n_eddy_sub):
            _T_lap_y = np.zeros_like(T_sst)
            _T_lap_y[1:-1, :] = T_sst[:-2, :] - 2.0 * T_sst[1:-1, :] + T_sst[2:, :]
            _T_lap_y[0, :]     = T_sst[1, :]  - T_sst[0, :]
            _T_lap_y[-1, :]    = T_sst[-2, :] - T_sst[-1, :]
            _T_lap_y = np.clip(_T_lap_y, -20.0, 20.0).astype(np.float32, copy=False)
            T_sst = (T_sst + _eddy_k * _T_lap_y * _eddy_lat[:, None] * _dt_eddy_sub).astype(np.float32, copy=False)

    # --- Ocean Transport ---
    # NOTE (2026-07-03): this block was originally written as a 30-day cache
    # ("ocean decorrelation time"), but the cache key included round(day_of_year)
    # so it never actually hit — the transport recomputed every step for the
    # entire calibrated life of the model. Honoring the 30-day reuse turned out
    # to change climate measurably (a stale ΔT applied for 30 days weakens the
    # seasonal response; the high-obliquity seasonal-range gate fails at 1.07x
    # vs its 1.1x bar) while the measured saving is only ~1 ms/step at
    # production resolution (Numba-free NumPy path). Per-step recompute is
    # therefore the intended, calibrated behavior; the "cache" is kept solely
    # as a state carrier for feedback-flag zeroing and mode bookkeeping.
    OCEAN_UPDATE_INTERVAL_DAYS = 0.0  # recompute every step (see NOTE above)
    _oc = _OCEAN_ADJ_CACHE
    _oc_key = (Hc, Wc, round(float(days), 3))
    _t_now = float(total_days) if total_days is not None else float(day_of_year)
    days_since_ocean = _t_now - float(_oc.get("last_update_day", -9999.0))
    if not _fb_t.get('ocean_transport', True):
        T_ocean_adj = np.zeros((Hc, Wc), dtype=np.float32)
        _oc["adj"] = T_ocean_adj
        _oc["key"] = _oc_key
        _oc["last_update_day"] = _t_now
    elif (
        _oc.get("adj") is None
        or _oc.get("key") != _oc_key
        or days_since_ocean < 0.0  # time went backwards → new run/loaded save
        or days_since_ocean >= OCEAN_UPDATE_INTERVAL_DAYS
    ):
        T_ocean_adj = calculate_ocean_heat_transport(
            T_sst, elev_c, Hc, Wc, day_of_year, days,
            transport_coefficient=float(ocean_transport_coeff),
            exchange_coefficient=float(ocean_exchange_coeff),
            exchange_inertia=float(ocean_exchange_inertia),
            prev_T=T_prev,
            ice_cover=sea_ice,
            T_equilibrium=T_eq,
        )
        T_ocean_adj = np.clip(T_ocean_adj, -10.0, 10.0)

        # Ekman wind-driven advection: shifts surface water 90° from wind (Coriolis).
        # The increment is scaled to ONE `days`-long step (like the transport
        # term above) because the cached T_ocean_adj is re-applied every step
        # until the next 30-day refresh. The old code scaled it to the full
        # 30-day window AND (with the broken cache) re-applied it every day —
        # a ~30x amplification of the intended Ekman heat shift.
        if _pp.has_liquid_water_ocean and _pp.ekman_strength > 0.0:
            u_ek, v_ek = compute_ekman_transport(
                u, v, elev_c,
                ekman_coefficient=0.03 * float(_pp.ekman_strength),
                rotation_direction=float(getattr(_pp, "rotation_direction", 1.0)),
            )
            # Upwind advection of T_sst by Ekman currents over one step.
            # Zonal grid spacing shrinks with cos(lat) on an equirectangular
            # grid; using the equatorial dx everywhere under-shifted zonal
            # Ekman advection at mid/high latitudes.
            dy_m = (np.pi / Hc) * float(_pp.radius_m)
            dx_m = (2.0 * np.pi / Wc) * float(_pp.radius_m) * cos_lat[:, None]
            dt_ek = float(days)
            shift_x = np.clip(u_ek * dt_ek * 86400.0 / dx_m, -0.5, 0.5)
            shift_y = np.clip(v_ek * dt_ek * 86400.0 / dy_m, -0.5, 0.5)
            T_ek = T_sst
            # Simple upwind: dT ≈ -u·∂T/∂x − v·∂T/∂y (finite difference)
            dT_dx = 0.5 * (np.roll(T_ek, -1, axis=1) - np.roll(T_ek, 1, axis=1))  # central diff, periodic x
            dT_dy = np.zeros_like(T_ek)
            dT_dy[1:-1, :] = 0.5 * (T_ek[:-2, :] - T_ek[2:, :])  # northward derivative (row 0 = north pole)
            ekman_adj = np.clip(-(shift_x * dT_dx + shift_y * dT_dy), -1.5, 1.5)
            _ocean_mask, _ = get_masks(elev_c)
            T_ocean_adj = T_ocean_adj + ekman_adj * _ocean_mask.astype(np.float32)

        # 2D barotropic gyre currents (Jul 2026): purely additive alongside the
        # zonal-mean transport and Ekman deflection above, never replacing
        # them -- gated independently of ekman_strength so it can be enabled
        # (or not) on its own. See ocean.compute_gyre_currents and
        # PlanetParams.ocean_gyre_strength docstrings for the full mechanism
        # and risk notes; highest-risk/most-structural item of the Jul 2026
        # backlog session, hence the conservative 0.0 default.
        if _pp.has_liquid_water_ocean and _pp.ocean_gyre_strength > 0.0:
            u_gy, v_gy = compute_gyre_currents(u, v, elev_c)
            gyre_scale = float(_pp.ocean_gyre_strength)
            dy_m_gy = (np.pi / Hc) * float(_pp.radius_m)
            dx_m_gy = (2.0 * np.pi / Wc) * float(_pp.radius_m) * cos_lat[:, None]
            dt_gy = float(days)
            shift_x_gy = np.clip(gyre_scale * u_gy * dt_gy * 86400.0 / dx_m_gy, -0.5, 0.5)
            shift_y_gy = np.clip(gyre_scale * v_gy * dt_gy * 86400.0 / dy_m_gy, -0.5, 0.5)
            T_gy = T_sst
            dT_dx_gy = 0.5 * (np.roll(T_gy, -1, axis=1) - np.roll(T_gy, 1, axis=1))
            dT_dy_gy = np.zeros_like(T_gy)
            dT_dy_gy[1:-1, :] = 0.5 * (T_gy[:-2, :] - T_gy[2:, :])
            gyre_adj = np.clip(-(shift_x_gy * dT_dx_gy + shift_y_gy * dT_dy_gy), -1.5, 1.5)
            _ocean_mask_gy, _ = get_masks(elev_c)
            T_ocean_adj = T_ocean_adj + gyre_adj * _ocean_mask_gy.astype(np.float32)

        _oc["adj"] = T_ocean_adj
        _oc["key"] = _oc_key
        _oc["last_update_day"] = _t_now
    else:
        T_ocean_adj = _oc["adj"]
    T_sst = (T_sst + T_ocean_adj).astype(np.float32, copy=False)

    # --- Hadley/Subsidence parameterization (applied to T_sst surface) ---
    lat_deg = np.rad2deg(np.abs(lat_2d))
    subsidence = 0.10 * np.exp(-((lat_deg - 30.0)/10.0)**2) * float(days)
    T_sst = (T_sst + subsidence).astype(np.float32, copy=False)

    # --- Final clamping ---
    _T_air_before_final_clamp = T_air.copy()
    _boundary_before_final_clamp = (
        None if boundary_layer_out is None else np.asarray(boundary_layer_out).copy()
    )
    # T_sst: ocean surface / land surface (200K–323K)
    T_sst = np.clip(T_sst, 200.0, 323.0)
    # T_air: free air — allow slightly wider range (180K–340K)
    T_air = np.clip(T_air, 180.0, 340.0)
    if boundary_layer_out is not None:
        boundary_layer_out = np.clip(boundary_layer_out, 180.0, 340.0).astype(
            np.float32, copy=False
        )
    if boundary_interface_out is not None:
        boundary_interface_out = np.clip(
            boundary_interface_out, 180.0, 340.0
        ).astype(np.float32, copy=False)

    # Track component contributions
    components = {}
    if _coupled_middle_temperature is not None:
        components["_radiative_midlevel_temperature"] = (
            _coupled_middle_temperature.astype(np.float32, copy=False)
        )
    if _coupled_upper_temperature is not None:
        components["_radiative_upperlevel_temperature"] = (
            _coupled_upper_temperature.astype(np.float32, copy=False)
        )
    if _optical_closure is not None:
        components["_radiative_optical_depth"] = (
            _optical_closure.total_optical_depth.astype(np.float32, copy=False)
        )
    if _pp.enable_coupled_two_layer_grey_radiation and _grey is not None:
        components["_grey_midlevel_gain_w_m2"] = (
            _grey.midlevel_gain_w_m2.astype(np.float32, copy=False)
        )
        components["_grey_upperlevel_gain_w_m2"] = (
            _grey.upperlevel_gain_w_m2.astype(np.float32, copy=False)
        )
    if land_deep_out is not None:
        # Private transport channel; simulate_step removes it before exposing
        # public temperature diagnostics, preserving this function's API.
        components["_land_deep_temperature"] = land_deep_out
        if track_components:
            components["land_latent_heat_w_m2"] = land_step.latent_heat_w_m2
            components["land_sensible_heat_w_m2"] = land_step.sensible_heat_w_m2
            if conservative_sensible_gain is not None:
                components[
                    "land_air_sensible_atmospheric_gain_w_m2"
                ] = conservative_sensible_gain
                components[
                    "land_air_sensible_exchange_closure_w_m2"
                ] = conservative_sensible_gain - np.where(
                    land_mask, land_step.sensible_heat_w_m2, 0.0
                )
            if boundary_exchange_gain is not None:
                components["boundary_layer_exchange_gain_w_m2"] = boundary_exchange_gain
                if _boundary_interface_active:
                    components["boundary_layer_interface_exchange_gain_w_m2"] = (
                        -boundary_exchange_gain + boundary_interface_upper_gain
                    )
                    components["free_air_exchange_gain_w_m2"] = (
                        -boundary_interface_upper_gain
                    )
                else:
                    components["free_air_exchange_gain_w_m2"] = (
                        -boundary_exchange_gain
                    )
                components["boundary_layer_pressure_thickness_pa"] = (
                    boundary_step.pressure_thickness_pa
                )
                components["boundary_layer_effective_entrainment_velocity_m_s"] = (
                    boundary_step.effective_entrainment_velocity_m_s
                )
                components["boundary_layer_bulk_richardson_number"] = (
                    boundary_step.bulk_richardson_number
                )
                if _boundary_interface_active:
                    components["boundary_layer_interface_pressure_thickness_pa"] = (
                        boundary_step.interface_pressure_thickness_pa
                    )
                    components["boundary_layer_interface_free_gain_w_m2"] = (
                        boundary_interface_upper_gain
                    )
                    components[
                        "boundary_layer_mechanical_entrainment_velocity_m_s"
                    ] = boundary_step.mechanical_entrainment_velocity_m_s
                    components[
                        "boundary_layer_convective_entrainment_velocity_m_s"
                    ] = boundary_step.convective_entrainment_velocity_m_s
                    components["boundary_layer_surface_buoyancy_flux_m2_s3"] = (
                        boundary_step.surface_buoyancy_flux_m2_s3
                    )
                components["boundary_layer_horizontal_transport"] = "omitted"
                components["resolved_heat_convergence_destination"] = "free_atmosphere"
                if boundary_horizontal_convergence is not None:
                    components["boundary_layer_horizontal_heat_convergence_w_m2"] = (
                        boundary_horizontal_convergence
                    )
                    components["boundary_layer_continuity_exchange_gain_w_m2"] = (
                        boundary_continuity_exchange
                    )
                    components["free_air_horizontal_heat_convergence_w_m2"] = (
                        transport_step.free_horizontal_convergence_w_m2
                    )
                    components["free_air_continuity_exchange_gain_w_m2"] = (
                        -boundary_continuity_exchange
                    )
                    components["boundary_layer_transport_substeps"] = (
                        boundary_transport_substeps
                    )
                    components[
                        "boundary_layer_horizontal_convergence_area_mean_w_m2"
                    ] = boundary_horizontal_area_mean
                    components["boundary_layer_horizontal_transport"] = "flux_form"
            if resolved_heat_convergence is not None:
                components["atmospheric_heat_convergence_w_m2"] = resolved_heat_convergence
                components[
                    "atmospheric_heat_convergence_applied_grid_area_mean_w_m2"
                ] = resolved_heat_convergence_area_mean
    if boundary_layer_out is not None:
        # Private channel threaded by the outer solver and omitted from public
        # component diagnostics after it has been copied into PlanetState.
        components["_boundary_layer_temperature"] = boundary_layer_out
    if boundary_interface_out is not None:
        components["_boundary_layer_interface_temperature"] = (
            boundary_interface_out
        )
    if track_components:
        _energy_lat_edges = np.linspace(np.pi / 2.0, -np.pi / 2.0, Hc + 1)
        _energy_area_rows = np.sin(_energy_lat_edges[:-1]) - np.sin(_energy_lat_edges[1:])
        _energy_denom = Wc * np.sum(_energy_area_rows)
        def _energy_area_mean(field: np.ndarray) -> float:
            return float(np.sum(np.asarray(field) * _energy_area_rows[:, None]) / _energy_denom)

        def _masked_energy_area_mean(field: np.ndarray, mask: np.ndarray) -> float:
            weighted_mask = np.asarray(mask) * _energy_area_rows[:, None]
            denominator = float(np.sum(weighted_mask))
            if denominator <= 0.0:
                return float("nan")
            return float(np.sum(np.asarray(field) * weighted_mask) / denominator)

        _diagnostic_free_capacity = np.full_like(
            T_air, _column_air_capacity, dtype=np.float64
        )
        if _boundary_layer_active and boundary_exchange_gain is not None:
            _diagnostic_boundary_capacity = (
                np.asarray(boundary_step.pressure_thickness_pa, dtype=np.float64)
                / float(_pp.surface_gravity)
                * float(_pp.cp_dry)
            )
            _diagnostic_free_capacity = np.where(
                land_mask,
                _column_air_capacity - _diagnostic_boundary_capacity,
                _column_air_capacity,
            )
            if _boundary_interface_active:
                _diagnostic_interface_capacity = (
                    float(boundary_step.interface_pressure_thickness_pa)
                    / float(_pp.surface_gravity)
                    * float(_pp.cp_dry)
                )
                _diagnostic_free_capacity = np.where(
                    land_mask,
                    _diagnostic_free_capacity - _diagnostic_interface_capacity,
                    _diagnostic_free_capacity,
                )
        _free_advection_gain = (
            _diagnostic_free_capacity * (T_after_advection - T_before_advection)
            / dt_sec
            if dt_sec > 0.0
            else np.zeros_like(T_air, dtype=np.float64)
        )
        _free_diffusion_gain = (
            _diagnostic_free_capacity * (T_after_diffusion - T_before_diffusion)
            / dt_sec
            if dt_sec > 0.0
            else np.zeros_like(T_air, dtype=np.float64)
        )
        _free_clamp_gain = (
            _diagnostic_free_capacity * (T_air - _T_air_before_final_clamp) / dt_sec
            if dt_sec > 0.0
            else np.zeros_like(T_air, dtype=np.float64)
        )
        _boundary_clamp_gain = np.zeros_like(T_air, dtype=np.float64)
        if (
            dt_sec > 0.0
            and _boundary_before_final_clamp is not None
            and boundary_layer_out is not None
            and _boundary_layer_active
            and boundary_exchange_gain is not None
        ):
            _boundary_clamp_gain = np.where(
                land_mask,
                _diagnostic_boundary_capacity
                * (boundary_layer_out - _boundary_before_final_clamp)
                / dt_sec,
                0.0,
            )

        components['advection'] = T_after_advection - T_before_advection
        components['diffusion'] = T_after_diffusion - T_before_diffusion
        components['free_air_advection_gain_w_m2'] = _free_advection_gain
        components['free_air_diffusion_gain_w_m2'] = _free_diffusion_gain
        components['free_air_transport_closure_delta_k'] = (
            _free_transport_closure_delta_k
        )
        components['free_air_final_clamp_gain_w_m2'] = _free_clamp_gain
        components['boundary_layer_final_clamp_gain_w_m2'] = _boundary_clamp_gain
        components['free_air_advection_gain_area_mean_w_m2'] = _energy_area_mean(
            _free_advection_gain
        )
        components['free_air_diffusion_gain_area_mean_w_m2'] = _energy_area_mean(
            _free_diffusion_gain
        )
        components['free_air_final_clamp_gain_area_mean_w_m2'] = _energy_area_mean(
            _free_clamp_gain
        )
        components['boundary_layer_final_clamp_gain_area_mean_w_m2'] = _energy_area_mean(
            _boundary_clamp_gain
        )
        _surface_air_relaxation_gain = (
            _column_air_capacity * _surface_air_relaxation_delta_k / dt_sec
            if dt_sec > 0.0
            else np.zeros_like(T_air, dtype=np.float64)
        )
        components['surface_air_relaxation_gain_w_m2'] = (
            _surface_air_relaxation_gain
        )
        components['surface_air_relaxation_gain_area_mean_w_m2'] = (
            _energy_area_mean(_surface_air_relaxation_gain)
        )
        if boundary_exchange_gain is not None:
            components['boundary_layer_surface_sensible_gain_area_mean_w_m2'] = (
                _energy_area_mean(
                    np.where(land_mask, land_step.sensible_heat_w_m2, 0.0)
                )
            )
            components['boundary_layer_exchange_gain_land_mean_w_m2'] = (
                _masked_energy_area_mean(boundary_exchange_gain, land_mask)
            )
            components['boundary_layer_effective_entrainment_land_mean_m_s'] = (
                _masked_energy_area_mean(
                    boundary_step.effective_entrainment_velocity_m_s, land_mask
                )
            )
            if _boundary_interface_active:
                components[
                    'boundary_layer_mechanical_entrainment_land_mean_m_s'
                ] = _masked_energy_area_mean(
                    boundary_step.mechanical_entrainment_velocity_m_s,
                    land_mask,
                )
                components[
                    'boundary_layer_convective_entrainment_land_mean_m_s'
                ] = _masked_energy_area_mean(
                    boundary_step.convective_entrainment_velocity_m_s,
                    land_mask,
                )
                components['boundary_layer_surface_buoyancy_flux_land_mean_m2_s3'] = (
                    _masked_energy_area_mean(
                        boundary_step.surface_buoyancy_flux_m2_s3,
                        land_mask,
                    )
                )
            components['boundary_layer_bulk_ri_land_mean'] = (
                _masked_energy_area_mean(
                    boundary_step.bulk_richardson_number, land_mask
                )
            )
            components['boundary_layer_stable_area_fraction'] = (
                _masked_energy_area_mean(
                    boundary_step.bulk_richardson_number > 0.0, land_mask
                )
            )
            components['boundary_layer_strongly_stable_area_fraction'] = (
                _masked_energy_area_mean(
                    boundary_step.bulk_richardson_number > 0.2, land_mask
                )
            )
            _inversion_top_temperature = (
                boundary_interface_out if _boundary_interface_active else T_air
            )
            _post_exchange_inversion = np.asarray(
                _inversion_top_temperature
            ) - np.asarray(boundary_layer_out)
            components['boundary_layer_post_exchange_inversion_land_mean_k'] = (
                _masked_energy_area_mean(_post_exchange_inversion, land_mask)
            )
            components[
                'boundary_layer_post_exchange_abs_inversion_land_mean_k'
            ] = _masked_energy_area_mean(
                np.abs(_post_exchange_inversion), land_mask
            )
        components['radiation'] = k_relax * (T_eq - T_sst) * float(days)
        components['evaporation'] = -evap_cooling
        components['ocean_transport'] = T_ocean_adj
        components['airsea_atmospheric_gain_w_m2'] = _airsea_atmospheric_gain_w_m2
        components['airsea_ocean_gain_w_m2'] = _airsea_ocean_gain_w_m2
        components['airsea_physical_energy_residual_w_m2'] = _airsea_physical_residual_w_m2
        components['airsea_atmospheric_gain_area_mean_w_m2'] = _energy_area_mean(
            _airsea_atmospheric_gain_w_m2
        )
        components['airsea_ocean_gain_area_mean_w_m2'] = _energy_area_mean(
            _airsea_ocean_gain_w_m2
        )
        components['airsea_physical_energy_residual_area_mean_w_m2'] = _energy_area_mean(
            _airsea_physical_residual_w_m2
        )
        components['ocean_air_relaxation_atmospheric_gain_w_m2'] = (
            _ocean_air_relaxation_gain_w_m2
        )
        components['ocean_air_relaxation_atmospheric_gain_area_mean_w_m2'] = (
            _energy_area_mean(_ocean_air_relaxation_gain_w_m2)
        )
        components['ocean_air_relaxation_ocean_gain_w_m2'] = (
            _ocean_air_relaxation_ocean_gain_w_m2
        )
        components['ocean_air_relaxation_physical_energy_residual_w_m2'] = (
            _ocean_air_relaxation_residual_w_m2
        )
        components['ocean_air_relaxation_ocean_gain_area_mean_w_m2'] = (
            _energy_area_mean(_ocean_air_relaxation_ocean_gain_w_m2)
        )
        components['ocean_air_relaxation_physical_energy_residual_area_mean_w_m2'] = (
            _energy_area_mean(_ocean_air_relaxation_residual_w_m2)
        )
        components['subsidence'] = subsidence
        components['equilibrium_temp'] = T_eq
        components['cloud_thermodynamic_temperature_k'] = (
            _cloud_thermodynamic_temperature
        )
        components['cloud_saturation_specific_humidity'] = qsat
        components['cloud_relative_humidity'] = rh
        components['cloud_temperature_source'] = (
            "boundary_layer_over_land"
            if _boundary_cloud_temperature_active
            else "free_atmosphere"
        )
        components['cloud_specific_humidity_area_mean'] = _energy_area_mean(q)
        components['cloud_saturation_specific_humidity_area_mean'] = (
            _energy_area_mean(qsat)
        )
        components['cloud_relative_humidity_area_mean'] = _energy_area_mean(rh)
        components['cloud_rh_above_065_area_fraction'] = _energy_area_mean(
            rh >= 0.65
        )
        components['land_cloud_relative_humidity_area_mean'] = (
            _masked_energy_area_mean(rh, land_mask)
        )
        components['ocean_cloud_relative_humidity_area_mean'] = (
            _masked_energy_area_mean(rh, sea_mask)
        )
        components['cloud_rh_core_area_mean'] = _energy_area_mean(rh_core)
        components['cloud_ascent_term_area_mean'] = _energy_area_mean(ascent_term)
        components['cloud_subsidence_area_mean'] = _energy_area_mean(subsidence)
        components['cloud_orographic_driver_area_mean'] = _energy_area_mean(orog)
        components['cloud_instantaneous_area_mean'] = _energy_area_mean(
            _cloud_fraction_instantaneous
        )
        components['cloud_after_persistence_area_mean'] = _energy_area_mean(
            _cloud_fraction_after_persistence
        )
        components['cloud_after_rainout_area_mean'] = _energy_area_mean(
            _cloud_fraction_after_rainout
        )
        components['cloud_final_area_mean'] = _energy_area_mean(cloud_fraction)
        components['net_radiation'] = R_net
        components['net_radiation_area_mean_w_m2'] = _energy_area_mean(R_net)
        components['cloud_shortwave_forcing_w_m2'] = _cloud_shortwave_forcing
        components['cloud_longwave_forcing_w_m2'] = _cloud_longwave_forcing
        components['cloud_net_radiative_forcing_w_m2'] = (
            _cloud_net_radiative_forcing
        )
        components['cloud_shortwave_forcing_area_mean_w_m2'] = _energy_area_mean(
            _cloud_shortwave_forcing
        )
        components['cloud_longwave_forcing_area_mean_w_m2'] = _energy_area_mean(
            _cloud_longwave_forcing
        )
        components['cloud_net_radiative_forcing_area_mean_w_m2'] = _energy_area_mean(
            _cloud_net_radiative_forcing
        )
        components['S_absorbed'] = S_absorbed
        components['L_out'] = L_out
        components['effective_surface_longwave_emissivity'] = epsilon
        components['radiative_surface_temperature_k'] = _radiative_surface_temperature
        components['absorbed_shortwave_area_mean_w_m2'] = _energy_area_mean(
            S_absorbed
        )
        components['outgoing_longwave_area_mean_w_m2'] = _energy_area_mean(
            L_out
        )
        components['effective_radiating_temperature_area_mean_k'] = (
            _energy_area_mean(_effective_radiating_temperature)
        )
        components['free_air_minus_effective_radiating_temperature_area_mean_k'] = (
            _energy_area_mean(T_air - _effective_radiating_temperature)
        )
        if _pp.enable_pressure_defined_radiative_temperature_profile:
            assert _middle_emission_temperature is not None
            assert _upper_emission_temperature is not None
            assert _optical_closure is not None
            assert _grey is not None
            components.update({
                'grey_radiation_mode': (
                    'two_layer_coupled'
                    if _pp.enable_coupled_two_layer_grey_radiation
                    else 'two_layer_diagnostic_only'
                ),
                'grey_emission_temperature_source': 'resolved_pressure_midlevel_and_upperlevel',
                'grey_emission_temperature_k': _middle_emission_temperature,
                'grey_upperlevel_emission_temperature_k': _upper_emission_temperature,
                'grey_total_optical_depth': _optical_closure.total_optical_depth,
                'grey_midlevel_emissivity': _optical_closure.midlevel_emissivity,
                'grey_upperlevel_emissivity': _optical_closure.upperlevel_emissivity,
                'grey_opaque_limited': _optical_closure.opaque_limited,
                'grey_target_olr_residual_w_m2': _optical_closure.target_olr_residual_w_m2,
                'grey_surface_gain_w_m2': _grey.surface_gain_w_m2,
                'grey_midlevel_gain_w_m2': _grey.midlevel_gain_w_m2,
                'grey_upperlevel_gain_w_m2': _grey.upperlevel_gain_w_m2,
                'grey_atmospheric_gain_w_m2': _grey.midlevel_gain_w_m2 + _grey.upperlevel_gain_w_m2,
                'grey_downward_longwave_at_surface_w_m2': _grey.downward_longwave_at_surface_w_m2,
                'grey_outgoing_longwave_w_m2': _grey.outgoing_longwave_w_m2,
                'grey_toa_net_radiation_w_m2': _grey.toa_net_radiation_w_m2,
                'grey_emission_temperature_area_mean_k': _energy_area_mean(_middle_emission_temperature),
                'grey_free_air_minus_emission_temperature_area_mean_k': _energy_area_mean(
                    _radiative_air_temperature - _middle_emission_temperature
                ),
                'grey_upperlevel_temperature_area_mean_k': _energy_area_mean(_upper_emission_temperature),
                'grey_total_optical_depth_area_mean': _energy_area_mean(_optical_closure.total_optical_depth),
                'grey_opaque_limited_area_fraction': _energy_area_mean(_optical_closure.opaque_limited),
                'grey_target_olr_residual_area_mean_w_m2': _energy_area_mean(
                    _optical_closure.target_olr_residual_w_m2
                ),
                'grey_surface_gain_area_mean_w_m2': _energy_area_mean(_grey.surface_gain_w_m2),
                'grey_midlevel_gain_area_mean_w_m2': _energy_area_mean(_grey.midlevel_gain_w_m2),
                'grey_upperlevel_gain_area_mean_w_m2': _energy_area_mean(_grey.upperlevel_gain_w_m2),
                'grey_atmospheric_gain_area_mean_w_m2': _energy_area_mean(
                    _grey.midlevel_gain_w_m2 + _grey.upperlevel_gain_w_m2
                ),
                'grey_downward_longwave_at_surface_area_mean_w_m2': _energy_area_mean(
                    _grey.downward_longwave_at_surface_w_m2
                ),
                'grey_outgoing_longwave_area_mean_w_m2': _energy_area_mean(_grey.outgoing_longwave_w_m2),
                'grey_toa_net_radiation_area_mean_w_m2': _energy_area_mean(_grey.toa_net_radiation_w_m2),
            })
            if _grey_air_temperature_increment is not None:
                components['grey_air_temperature_increment_k'] = (
                    _grey_air_temperature_increment
                )
                components['grey_air_temperature_increment_area_mean_k'] = (
                    _energy_area_mean(_grey_air_temperature_increment)
                )
        def _summ(field: np.ndarray) -> dict:
            return {"mean": float(np.mean(field)), "min": float(np.min(field)), "max": float(np.max(field))}
        components["toa"] = {
            "S_absorbed": _summ(S_absorbed),
            "L_out": _summ(L_out),
            "R_net": _summ(R_net),
            "albedo_mean": float(np.mean(albedo_total)),
            "cloud_mean": float(np.mean(cloud_fraction)),
            "epsilon_mean": float(np.mean(epsilon)),
        }

    return T_sst.astype(np.float32), T_air.astype(np.float32), cloud_fraction.astype(np.float32), snow_cover.astype(np.float32), components, T_deep_out, cloud_water_new


# ============================================================================
# Simulation cache management
# ============================================================================

def clear_simulation_caches() -> None:
    """Reset module-level simulation caches (call on load/preset/new sim)."""
    from masks import clear_all_caches
    from temperature import clear_temperature_cache

    _RELAX_CACHE.clear()
    _RELAX_CACHE.update({"key": None, "u": None, "v": None})
    clear_grid_caches()
    _OCEAN_ADJ_CACHE.clear()
    _OCEAN_ADJ_CACHE.update({"adj": None, "last_update_day": -9999.0})
    _CARBON_SLOW_CACHE.clear()
    _CARBON_SLOW_CACHE.update({"key": None, "last_update_day": -9999.0, "biome": None})
    _MARITIME_CACHE.clear()
    _MARITIME_CACHE.update({"key": None, "field": None})
    # `_maritime_proximity_coarse` has its OWN cache, and it was missing here
    # (2026-08-05) -- both are keyed on `id(elevation)` plus a 512-sample strided
    # fingerprint, so both can take a stale hit when a new elevation array lands
    # on a reused id and differs only in cells the stride skips. That is exactly
    # what this function exists to prevent on load/preset/new-sim. It surfaced as
    # an order-dependent test failure (`test_maritime_transport.py::
    # test_coarse_block_mean_is_over_land_cells_only` reading 0.0 for a lone land
    # cell at flat index 1572, which a stride of 16 never samples).
    _MARITIME_COARSE_CACHE.clear()
    _MARITIME_COARSE_CACHE.update({"key": None, "field": None})
    clear_all_caches()
    clear_temperature_cache()
