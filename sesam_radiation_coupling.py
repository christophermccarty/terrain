"""SESAM stage P6d: live coupling of the (A69)-(A117)/(A10) radiation chain
(``sesam_shortwave.py``/``sesam_longwave.py``/``sesam_tropopause.py``, stage
P5) into the prognostic simulation loop, replacing ``sesam_coupling.py``'s
bridge-3 diabatic-source placeholder (the bulk ``(T*-Ta)/1 day`` skin-
temperature relaxation P6b used as a stand-in for a real SWa/LWa split).

docs/SESAM_GAP_ANALYSIS.md Sec7 P6. Gated by
``PlanetParams.enable_sesam_radiation`` (default False, the same gate P5's
own diagnostic-only kernels have used since they were built) *in addition
to* ``enable_sesam_column_closure`` -- both must be on for this module to
run; column closure alone (with or without P6c's wind/EKE gate) still uses
P6b's bulk-relaxation bridge unchanged when this gate is off.

Mirrors the exact, already-validated chain ``scripts/diagnose_sesam_toa.py``
already ran against a saved state for stage P5's own TOA exit-gate
measurement (2026-08-19), adapted to read from live evolving ``PlanetState``
fields and the *current* day of year (a real running simulation has an
actual date, unlike that script's annual-mean comparison against Table 1):

1. P1 vertical structure (``sesam_vertical.compute_vertical_structure``) on
   the same stratosphere-reaching 10-level grid the TOA exit gate used --
   P6c's own vertical-structure call only reaches 12 km (tropospheric,
   sufficient for wind/EKE but not for a genuine top-of-atmosphere OLR).
2. (A6) cloud geometry: fraction reused from the live ``cloud_cover`` field
   (real, already-produced data, not fabricated -- the same reuse
   ``diagnose_sesam_toa.py`` already established), top height/optical
   thickness from SESAM's own (A67)/(A68). Column water path uses the real
   live ``sesam_column_water_mm`` (``Qq``, stage P4's own prognostic state)
   when P6b has already initialized it, falling back to the same
   ``qa*2000`` approximation ``sesam_coupling.py`` uses for its own
   lazy-init case otherwise -- a genuine improvement over
   ``diagnose_sesam_toa.py``'s flat placeholder, which had no live
   prognostic water field to read from a standalone saved-state script.
3. (A69)-(A105) shortwave (``sesam_shortwave.shortwave_radiation``) and
   (A106)-(A117) longwave (``sesam_longwave.longwave_radiation``), reduced
   to the *atmosphere-absorbed* SWa/LWa fields (A40)'s diabatic source
   needs. This reduction is not itself a printed equation number -- the
   paper's (A40) source terms are column-net quantities, so this module
   derives them from the printed TOA/surface fluxes the same way any
   radiative-transfer atmosphere-heating budget is derived:
   ``SWa = (TOA down - TOA up) - surface absorbed`` and
   ``LWa = (surface upward - surface downward) - (TOA upward - TOA
   downward)`` (the second term is the OLR net, TOA downward LW being ~0 by
   the (A106) boundary condition ``sesam_longwave.py`` already proves).
4. (A10) tropopause height (``sesam_tropopause.advance_tropopause_height``),
   persisted across calls in the new ``PlanetState.sesam_tropopause_height_m``
   field (lazily initialized to the 12 km guess every prior SESAM stage has
   used; old saves unaffected -- the same convention ``sesam_column_water_mm``
   already established). Integrated internally at a 1-day cadence for
   whatever ``dt_days`` span the caller passes (a no-op single iteration for
   the now-standard 1-day-substep call described in point 5 below, but still
   correct if a caller ever passes a larger span directly): ``Rstr,net``'s
   two flux-profile inputs are computed once per call and held fixed while
   the tropopause height itself is Euler-integrated in 1-day increments
   (cheap array arithmetic, unlike the full longwave transmission-matrix
   solve this function's caller already recomputes every 1-day substep -- a
   different, unrelated cost). The ITCZ latitude driving (A11)'s shape
   function reuses this project's own existing seasonal-ITCZ convention
   (``pp.itcz_seasonal_response * pp.solar_declination(day)``, the same
   formula ``atmosphere.py``'s precipitation window already uses) rather
   than ``diagnose_sesam_toa.py``'s static annual-mean 0.0 -- a live
   coupling has an actual date to work from, that script did not.

**Four documented placeholders, not fabricated physics** (same "reuse a
real field, don't invent one" discipline the P4/P5 diagnostic scripts and
P6b/P6c's own bridges already use):
1. Surface albedo -- flat per-surface-type constants (ocean 0.08, land
   0.20, ice/thick-snow 0.6), identical to ``diagnose_sesam_toa.py``'s own
   documented placeholder. This project's real per-cell ``albedo_sfc``
   field is computed deep inside ``_evolve_temperature`` in ``simulate.py``
   and not exposed outside that function; threading it out is a larger
   legacy-path change than this stage's own scope (replacing the
   diabatic-source bridge), left as a documented follow-up.
2. Aerosol optical thickness/imaginary refractive index -- zero
   (clean-atmosphere placeholder, ``shortwave_radiation``'s own documented
   convention for a caller with no aerosol field).
3. ``w700_mean`` (mean-cell 700 hPa vertical velocity, feeds A67 cloud-top
   height) -- zero, matching stage P1's own documented placeholder.
4. Ozone -- ``sesam_tropopause.standard_ozone_mixing_ratio_profile_kgkg``'s
   constant-global climatology (see that module's own docstring).

5. **Recomputed every 1-day SESAM column-closure substep, NOT once per
   outer ``simulate_step`` call** -- unlike P6c's wind/EKE (which this
   module's caller does hold fixed for the whole outer step, since wind/EKE
   only scale advection/diffusion and are not themselves a source term). A
   real instability was found and root-caused during this stage's own
   development: an earlier version held SWa/LWa fixed across the whole
   outer step (mirroring P6c's own convention), which stayed numerically
   fine for single-day DAILY calls but drove ``air_temperature`` to NaN
   within 2-3 MONTHLY (30-day) calls on a real 16x32 smoke run. The cause is
   not a lack of substepping in the *column-energy* integration (that part
   was already correctly substepped at 1-day cadence, per P6b's own fix) --
   it is that holding the *diabatic source itself* fixed for 30 consecutive
   1-day Euler steps removes the Stefan-Boltzmann negative feedback (LW
   emission rising with Ta) that keeps a radiative forcing self-limiting in
   the real equations: a fixed SWa/LWa is a constant heating *rate*, not a
   heating rate that responds to the temperature it is heating, so the same
   overshoot mechanics as P6b's own diabatic-bridge bug
   (docs/SESAM_GAP_ANALYSIS.md Sec7 P6b) reappear via a different route.
   The fix is architectural, not a damping term (per this project's own
   ``docs/VERTICAL_THERMODYNAMIC_CLOSURE.md`` precedent against ad-hoc
   stabilizers): the caller (``simulate.py``) now calls this function once
   per 1-day column-closure substep, using that substep's own current
   ``air_temperature_k``, so each day's SW/LW genuinely responds to that
   day's Ta before the next day's heating is computed -- expensive (a full
   longwave transmission-matrix solve per simulated day, not per outer
   call), but correct, and this project's own precedent (P6b's fix) already
   established that correctness here takes priority over the cheaper
   cadence P6c's non-source-term wind/EKE could safely use. A cheaper
   cadence remains a documented performance follow-up for the P6e
   full-chain sanity run if this proves prohibitive at the 512x1024
   headline grid, not something to trade away silently here.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

import sesam_longwave as slw
import sesam_radiation as sr
import sesam_shortwave as ssw
import sesam_tropopause as stp
import sesam_vertical as sv
from sesam_thermo import t2m_diagnostic
from sesam_vertical import saturation_specific_humidity
from sesam_wind import _latitude_rad

# Same stratosphere-reaching level grid stage P5's own TOA exit-gate script
# used (scripts/diagnose_sesam_toa.py) -- P6c's tropospheric-only 6-level
# grid is not enough for a genuine outgoing-longwave measurement.
_LEVELS_M = np.array(
    [0.0, 1500.0, 3000.0, 5500.0, 9000.0, 12000.0, 16000.0, 20000.0, 25000.0, 30000.0]
)
_TROPOPAUSE_HEIGHT_DEFAULT_M = 12000.0
_SCALE_HEIGHT_REFERENCE_TEMP_K = 288.0
_H_PBL_M = 1500.0  # sesam_reference TABLE_A6_CLOUDS entry, A67's cloud-base assumption
_HADLEY_WIDTH_DEG = 30.0  # diagnose_sesam_toa.py's own (A11) shape-function width
_QQ_SCALE_KG_M2_PER_KGKG = 2000.0  # sesam_coupling.py's own lazy-init fallback scale
_TROPOPAUSE_SUBSTEP_DAYS = 1.0


def _surface_albedo(surface_kind: np.ndarray, ice_mask: np.ndarray | None, snow_depth: np.ndarray | None) -> np.ndarray:
    """Flat per-surface-type placeholder, identical to
    ``diagnose_sesam_toa.py``'s own (module docstring placeholder 1)."""
    alb = np.where(surface_kind == 0, 0.08, np.where(surface_kind == 2, 0.6, 0.20))
    bright = np.zeros(alb.shape, dtype=bool)
    if ice_mask is not None:
        bright |= np.asarray(ice_mask, dtype=bool)
    if snow_depth is not None:
        bright |= np.asarray(snow_depth, dtype=np.float64) > 0.1
    return np.where(bright, np.maximum(alb, 0.6), alb)


class SesamRadiationStep(NamedTuple):
    sw_absorbed_w_m2: np.ndarray
    lw_net_w_m2: np.ndarray
    tropopause_height_m: np.ndarray
    diagnostics: dict


def sesam_radiation_step(
    *,
    air_temperature_k: np.ndarray,
    skin_temperature_k: np.ndarray,
    relative_humidity: np.ndarray,
    cloud_fraction: np.ndarray,
    column_water_mm: np.ndarray | None,
    elevation_m: np.ndarray,
    land_mask: np.ndarray,
    ice_mask: np.ndarray | None,
    snow_depth: np.ndarray | None,
    tropopause_height_m: np.ndarray | None,
    surface_pressure_pa: float,
    gravity: float,
    co2_ppm: float,
    day_of_year: float,
    itcz_seasonal_response: float,
    solar_declination_rad: float,
    daily_mean_insolation_w_m2: np.ndarray,
    dt_days: float,
) -> SesamRadiationStep:
    """One P5 (A69)-(A117)/(A10) radiation evaluation, live-state-shaped.

    Returns the atmosphere-absorbed ``SWa``/``LWa`` fields
    ``sesam_thermo.diabatic_heating_rate_k_day`` needs, plus the updated
    tropopause height (persist the returned field across calls, the same
    convention ``sesam_column_water_mm`` already established for ``Qq``).
    Mirrors ``scripts/diagnose_sesam_toa.py``'s chain -- see module
    docstring for the four documented placeholders and the tropopause
    substepping rationale.
    """
    ta = np.asarray(air_temperature_k, dtype=np.float64)
    if ta.ndim != 2:
        raise ValueError("air_temperature_k must be a 2-D (H, W) field")
    h, w = ta.shape
    skin = np.asarray(skin_temperature_k, dtype=np.float64)
    ra = np.clip(np.asarray(relative_humidity, dtype=np.float64), 0.0, 1.0)
    fcld = np.clip(np.asarray(cloud_fraction, dtype=np.float64), 0.0, 1.0)
    elev = np.asarray(elevation_m, dtype=np.float64)
    land = np.asarray(land_mask, dtype=bool)
    p0 = float(surface_pressure_pa)
    p0_field = np.full((h, w), p0)

    lat_rad = _latitude_rad(h)[:, None] * np.ones((1, w))

    surface_kind = np.where(land, 1, 0).astype(np.int64)
    if ice_mask is not None:
        surface_kind = np.where(np.asarray(ice_mask, dtype=bool), 2, surface_kind)

    ht_prev = (
        np.full((h, w), _TROPOPAUSE_HEIGHT_DEFAULT_M)
        if tropopause_height_m is None
        else np.asarray(tropopause_height_m, dtype=np.float64)
    )

    qa_kg_kg = ra * saturation_specific_humidity(ta, p0_field)
    structure = sv.compute_vertical_structure(
        _LEVELS_M,
        near_surface_air_temp_k=ta, skin_temp_k=skin, surface_kind=surface_kind,
        near_surface_specific_humidity_kgkg=qa_kg_kg, surface_elevation_m=elev,
        tropopause_height_m=ht_prev,
        p0_pa=p0, gravity=float(gravity), reference_temp_k=_SCALE_HEIGHT_REFERENCE_TEMP_K,
    )
    surface_pressure_field_pa = p0 * np.exp(
        -elev / sv.height_scale(_SCALE_HEIGHT_REFERENCE_TEMP_K, gravity=float(gravity))
    )

    column_water_kg_m2 = (
        qa_kg_kg * _QQ_SCALE_KG_M2_PER_KGKG
        if column_water_mm is None
        else np.asarray(column_water_mm, dtype=np.float64)
    )

    cloud_top_m = sr.cloud_top_height_m(ht_prev, np.zeros((h, w)))
    cloud_base_m = elev + _H_PBL_M
    cloud_top_m = np.maximum(cloud_top_m, cloud_base_m + 500.0)
    t2m = t2m_diagnostic(ta, skin)
    cloud_optical_thickness = sr.cloud_optical_thickness(t2m, fcld, column_water_kg_m2)

    surface_albedo = _surface_albedo(surface_kind, ice_mask, snow_depth)

    delta = float(solar_declination_rad)
    cos_zenith_noon = np.clip(
        np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta), 1e-3, 1.0
    )
    toa_insolation = np.asarray(daily_mean_insolation_w_m2, dtype=np.float64)
    if toa_insolation.shape != (h, w):
        toa_insolation = toa_insolation * np.ones((h, w))

    sw = ssw.shortwave_radiation(
        incoming_toa_w_m2=toa_insolation, cos_zenith=cos_zenith_noon,
        cloud_fraction=fcld, cloud_top_height_m=cloud_top_m,
        cloud_optical_thickness=cloud_optical_thickness,
        cloud_geometric_thickness_m=cloud_top_m - cloud_base_m,
        column_water_kg_m2=column_water_kg_m2,
        humidity_scale_height_m=sv.rh_scale_height(sv.tropical_weight(np.abs(lat_rad)), 0.0),
        surface_albedo_vu=surface_albedo, surface_albedo_ir=surface_albedo,
        aerosol_optical_thickness=np.zeros((h, w)), aerosol_imaginary_refractive_index=np.zeros((h, w)),
    )
    surface_absorbed_sw = sw.surface_downward_w_m2 * (1.0 - surface_albedo)
    sw_absorbed_w_m2 = (toa_insolation - sw.toa_upward_w_m2) - surface_absorbed_sw

    ozone_profile = stp.standard_ozone_mixing_ratio_profile_kgkg(_LEVELS_M, structure.air_density_kg_m3)
    lw = slw.longwave_radiation(
        temperature_profile_k=structure.temperature_k,
        specific_humidity_profile_kg_kg=structure.specific_humidity_kgkg,
        ozone_mixing_ratio_profile_kg_kg=ozone_profile,
        pressure_profile_pa=structure.pressure_pa,
        air_density_profile_kg_m3=structure.air_density_kg_m3,
        levels_m=_LEVELS_M, surface_pressure_pa=surface_pressure_field_pa,
        surface_skin_temp_k=skin, co2_ppm=float(co2_ppm), gravity=float(gravity),
        cloud_fraction=fcld, cloud_base_height_m=cloud_base_m,
        cloud_top_height_m=cloud_top_m, cloud_optical_thickness=cloud_optical_thickness,
    )
    n = _LEVELS_M.size
    surface_net_up_lw = lw.upward_w_m2[0] - lw.downward_w_m2[0]
    toa_net_up_lw = lw.upward_w_m2[n - 1] - lw.downward_w_m2[n - 1]
    lw_net_w_m2 = surface_net_up_lw - toa_net_up_lw

    itcz_lat_rad = float(itcz_seasonal_response) * delta
    hadley_width_rad = np.deg2rad(_HADLEY_WIDTH_DEG)
    n_sub = max(1, int(round(float(dt_days) / _TROPOPAUSE_SUBSTEP_DAYS)))
    sub_dt = float(dt_days) / n_sub
    ht = ht_prev
    trop = None
    for _ in range(n_sub):
        trop = stp.advance_tropopause_height(
            ht, lat_rad, itcz_lat_rad, hadley_width_rad,
            lw.downward_w_m2, lw.upward_w_m2, _LEVELS_M, toa_insolation,
            dt_days=sub_dt, surface_elevation_m=elev,
        )
        ht = trop.tropopause_height_m

    diagnostics = {
        "toa_solar_up_mean_w_m2": float(np.mean(sw.toa_upward_w_m2)),
        "outgoing_longwave_mean_w_m2": float(np.mean(lw.upward_w_m2[n - 1])),
        "sw_absorbed_mean_w_m2": float(np.mean(sw_absorbed_w_m2)),
        "lw_net_mean_w_m2": float(np.mean(lw_net_w_m2)),
        "tropopause_height_mean_m": float(np.mean(ht)),
    }

    return SesamRadiationStep(
        sw_absorbed_w_m2=sw_absorbed_w_m2.astype(np.float32),
        lw_net_w_m2=lw_net_w_m2.astype(np.float32),
        tropopause_height_m=ht.astype(np.float32),
        diagnostics=diagnostics,
    )


__all__ = ["SesamRadiationStep", "sesam_radiation_step"]
