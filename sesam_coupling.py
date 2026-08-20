"""SESAM stage P6b: live coupling of the (A40)/(A42)/(A44) column-energy/water
closure (``sesam_thermo.py``, stage P4) into the prognostic simulation loop.

docs/SESAM_GAP_ANALYSIS.md Sec7 P6. Gated by ``PlanetParams.enable_sesam_column_closure``
(default False); ``simulate_step`` calls ``sesam_column_closure_step`` only
when that gate is True, overriding the legacy row-target-allocator
precipitation and the legacy near-surface-air-temperature update with
SESAM's own (A40)/(A42)/(A44) closure -- the mechanism-replacement this whole
SESAM adoption (Sec1/Sec4) was motivated by. Everything else in the step
(ocean, sea ice, land surface, biome, carbon cycle, cloud fraction, the
legacy *skin* temperature and its own T_eq/radiation) is untouched and
continues to run exactly as before; only air temperature, near-surface
relative humidity and precipitation are overridden, and only when the gate
is on.

Three documented bridges, not fabricated physics -- the same "reuse a real
field from elsewhere, don't invent one" discipline already used twice by the
P4/P5 diagnostic scripts (``diagnose_sesam_toa.py``'s saved ``cloud_cover``
reuse, ``diagnose_sesam_thermo.py``'s evaporation proxy), each flagged for
replacement by its own later P6 sub-stage:

1. **Wind** (stage P6c will replace this): this project's own already-computed
   surface ``wind_u``/``wind_v`` drives SESAM's advective transport. SESAM's
   own (A16)-(A27) wind (``sesam_wind.py``) is not wired into the live loop.
2. **EKE** (stage P6c will replace this): a spatially uniform placeholder at
   stage P3's own validated global-mean local-steady-state value (245.9
   m^2/s^2 DJF / 232.8 m^2/s^2 JJA at the real 512x1024 saved state,
   docs/SESAM_GAP_ANALYSIS.md Sec7 P3 2026-08-18 measurement), not the real
   prognostic K field (``sesam_synoptic.py``, not wired into the live loop).
3. **Diabatic source** (stage P6d, ``sesam_radiation_coupling.py``, replaces
   this when ``PlanetParams.enable_sesam_radiation`` is also on): (A40)
   needs SWa+LWa+Le*Pw+Ls*Ps+SH. When the caller supplies real
   ``sw_absorbed_w_m2``/``lw_net_w_m2`` fields (P6d's own
   ``sesam_radiation_step``, SESAM's (A69)-(A117) radiative split), this
   module assembles the full (A40) source via
   ``sesam_thermo.diabatic_heating_rate_k_day``: Pw/Ps come from splitting
   this step's own (A44) precipitation output by the same air-temperature
   rain/snow ramp ``simulate.py``'s legacy snow-pack model already uses
   (1 at <=-3 degC, 0 at >=2 degC), and SH reuses the exact bulk-aerodynamic
   formula ``land_surface.py`` already computes for its own sensible-heat
   term (``rho0 * cp_air * 1.3e-3 * |wind| * (T* - Ta)``), the same
   1.3e-3 bulk transfer coefficient this module's own evaporation proxy
   already uses below. When the caller omits those fields (P6d's gate off,
   the default), this module falls back to treating the legacy model's own
   newly-computed skin temperature ``T*`` (already resolved by the
   supported ocean/land/radiation code every step) as a bulk sensible-
   heat-style driver of Ta: ``(T* - Ta) / relaxation_days``. This fallback
   is not an invented mechanism either -- it is (A41)'s own
   T2m=(Ta+T*)/2 relationship read as a forcing, and (A40)'s own SH term is
   proportional to T*-Ta in a standard bulk formulation; it is simply a
   coarser stand-in for the real SW/LW/latent/SH split P6d now supplies
   when its own gate is on.

Grid geometry reuses ``sesam_synoptic.spherical_transport_geometry``/
``zonal_center_spacing_m`` directly (built fresh each call from H/W/radius,
the same convention every ``scripts/diagnose_sesam_*.py`` script already
uses) rather than threading a new geometry cache through ``simulate.py``.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from sesam_synoptic import spherical_transport_geometry, zonal_center_spacing_m
from sesam_thermo import (
    column_heat_capacity_j_m2_k,
    diabatic_heating_rate_k_day,
    evolve_column_energy,
    evolve_column_water_vapor,
)
from sesam_vertical import saturation_specific_humidity

# Stage P3's own validated global-mean local-steady-state EKE
# (docs/SESAM_GAP_ANALYSIS.md Sec7 P3, 2026-08-18 measurement, DJF/JJA mean of
# 245.9/232.8 m^2/s^2 on the real 512x1024 saved state) -- a documented
# placeholder for the real prognostic K field, see module docstring bridge 2.
SESAM_EKE_PLACEHOLDER_M2_S2 = 240.0

# Bulk sensible-heat-style relaxation timescale for the T*->Ta bridge (module
# docstring bridge 3): 1 day, the same order as this project's other
# near-surface air/skin coupling timescales elsewhere in simulate.py.
SESAM_BRIDGE_RELAXATION_DAYS = 1.0

# Same lower-layer water-mass scale atmosphere.py's own gated
# enable_prognostic_column_water path already uses for its Qq<->qa
# conversion (docs/SESAM_GAP_ANALYSIS.md Sec7 P5 TOA-gate entry) -- not a new
# convention invented here.
_QQ_SCALE_KG_M2_PER_KGKG = 2000.0

# Reference near-surface temperature for the scalar air-density conversion
# sesam_thermo.slope_convergence_mm_day requires (its rho0 parameter is
# type-hinted float, not a per-cell field). Standard-atmosphere sea-level
# reference, consistent with sesam_reference.py's own physical-constant
# conventions -- a documented simplification, not a per-planet fit.
_RHO0_REFERENCE_TEMP_K = 288.15
_RD_J_KG_K = 287.0  # dry-air gas constant, this project's standing convention

# Bulk-aerodynamic transfer coefficient -- the same 1.3e-3 constant this
# module's own evaporation proxy already uses below and land_surface.py's
# `ra = 1.0 / (1.3e-3 * wind)` sensible-heat resistance already uses.
_BULK_TRANSFER_COEFFICIENT = 1.3e-3
_CP_AIR_J_KG_K = 1004.0  # standing project convention, see sesam_thermo.py note 2

# Reference gravity for the P6d diabatic source's column heat capacity when
# a caller does not pass gravity_m_s2 explicitly -- standard-atmosphere
# sea-level value, the same "documented scalar simplification" convention
# as _RHO0_REFERENCE_TEMP_K above. Callers on non-Earth planets should pass
# their own PlanetParams.surface_gravity.
_GRAVITY_DEFAULT_M_S2 = 9.81

# Legacy snow/rain-phase ramp, identical to simulate.py's own snow-pack
# model (1 at <=-3 degC, 0 at >=2 degC) -- reused here for (A40)'s Pw/Ps
# split rather than inventing a second convention.
def _snow_fraction(air_temperature_k: np.ndarray) -> np.ndarray:
    t_air_c = np.asarray(air_temperature_k, dtype=np.float64) - 273.15
    return np.clip((-t_air_c + 2.0) / 5.0, 0.0, 1.0)


class SesamColumnClosureStep(NamedTuple):
    air_temperature_k: np.ndarray
    relative_humidity: np.ndarray
    precipitation_mm_day: np.ndarray
    column_water_mm: np.ndarray
    diagnostics: dict


def _slope_magnitude(elevation_m: np.ndarray, dx_m: np.ndarray, dy_m: float) -> np.ndarray:
    """Centered-difference ``|grad zs|``, periodic in longitude, one-sided at
    the poles (the finite-volume grid's own zero-flux polar convention)."""
    d_zs_dx = (np.roll(elevation_m, -1, axis=1) - np.roll(elevation_m, 1, axis=1)) / (2.0 * dx_m)
    d_zs_dy = np.zeros_like(elevation_m)
    d_zs_dy[1:-1] = (elevation_m[:-2] - elevation_m[2:]) / (2.0 * dy_m)
    d_zs_dy[0] = (elevation_m[0] - elevation_m[1]) / dy_m
    d_zs_dy[-1] = (elevation_m[-2] - elevation_m[-1]) / dy_m
    return np.sqrt(d_zs_dx ** 2 + d_zs_dy ** 2)


def sesam_column_closure_step(
    *,
    air_temperature_k: np.ndarray,
    skin_temperature_k: np.ndarray,
    relative_humidity: np.ndarray,
    column_water_mm: np.ndarray | None,
    wind_u_m_s: np.ndarray,
    wind_v_m_s: np.ndarray,
    elevation_m: np.ndarray,
    land_mask: np.ndarray,
    surface_pressure_pa: float,
    radius_m: float,
    dt_days: float,
    eke_m2_s2: np.ndarray | None = None,
    implicit_zonal_diffusion: bool = False,
    sw_absorbed_w_m2: np.ndarray | None = None,
    lw_net_w_m2: np.ndarray | None = None,
    gravity_m_s2: float = _GRAVITY_DEFAULT_M_S2,
    total_wind_speed_m_s: np.ndarray | None = None,
) -> SesamColumnClosureStep:
    """One SESAM (A40)/(A42)/(A44) column-closure step, live-state-shaped.

    Replaces the caller's legacy air-temperature-evolution and precipitation
    outputs; every other field (ocean, ice, land, biome, carbon, clouds, the
    legacy skin temperature itself) is untouched by this function -- it reads
    ``skin_temperature_k`` as a driver, never writes it. See module docstring
    for the three documented bridges (wind, EKE, diabatic source) this first
    live-coupling stage uses in place of SESAM's own not-yet-wired P2/P3/P5
    outputs.

    ``eke_m2_s2``: stage P6c (``sesam_wind_coupling.py``) supplies the real
    P3 local-steady-state field here once ``PlanetParams.enable_sesam_dynamics``
    is also on; ``None`` (the default) keeps bridge 2's spatially uniform
    placeholder (``SESAM_EKE_PLACEHOLDER_M2_S2``) for P6b-only callers.

    ``sw_absorbed_w_m2``/``lw_net_w_m2``: stage P6d (``sesam_radiation_coupling.py``)
    supplies SESAM's own atmosphere-absorbed SWa/LWa fields here once
    ``PlanetParams.enable_sesam_radiation`` is also on; ``None`` (the default,
    both must be supplied together) keeps bridge 3's ``(T*-Ta)/relaxation_days``
    fallback for P6b/P6c-only callers. When supplied, the water step runs
    first (it does not depend on the energy step's output) so its own (A44)
    precipitation can feed (A40)'s Le*Pw+Ls*Ps terms.

    ``total_wind_speed_m_s``: SESAM's own (A58) ``Us = sqrt(us^2+vs^2+Usyn^2)``
    -- the zonal-cell wind combined with P3's synoptic/storm-track gustiness
    term, computed by ``sesam_wind_coupling.sesam_wind_and_eke_step`` as
    ``SesamWindAndEke.total_wind_m_s`` once ``enable_sesam_dynamics`` is on.
    Drives only the bulk-aerodynamic evaporation and sensible-heat *wind
    speed* terms below -- both are nonlinear functions of instantaneous wind
    speed, for which (A58)'s total is the physically appropriate driver (see
    docs/SESAM_GAP_ANALYSIS.md Sec7 P2, 2026-08-20 follow-up: NCEP's true
    wspd climatology, fetched and measured directly, runs ~1.6-1.8x the
    mean-*vector*-wind figure this project had been comparing SESAM's
    zonal-only wind against, and SESAM's own (A58) total lines up with the
    true climatology far better than the zonal component alone does).
    ``wind_u_m_s``/``wind_v_m_s`` continue to drive advection unchanged --
    the correct field for transporting a tracer, not the flux formulas'
    wind-speed term. ``None`` (the default) falls back to
    ``sqrt(wind_u_m_s^2+wind_v_m_s^2)``, reproducing this function's
    previous behaviour exactly for P6b/P6c-only callers.
    """
    ta0 = np.asarray(air_temperature_k, dtype=np.float64)
    if ta0.ndim != 2:
        raise ValueError("air_temperature_k must be a 2-D (H, W) field")
    h, w = ta0.shape
    tstar = np.asarray(skin_temperature_k, dtype=np.float64)
    ra = np.clip(np.asarray(relative_humidity, dtype=np.float64), 0.0, 1.0)
    land = np.asarray(land_mask, dtype=bool)
    elev = np.asarray(elevation_m, dtype=np.float64)
    p0 = float(surface_pressure_pa)

    area, x_face, y_face = spherical_transport_geometry(h, w, float(radius_m))
    dx_m = zonal_center_spacing_m(h, w, float(radius_m))
    dy_m = float(radius_m) * (np.pi / h)

    p0_field = np.full((h, w), p0)
    qa_kg_kg = ra * saturation_specific_humidity(ta0, p0_field)
    if column_water_mm is None:
        qq0 = qa_kg_kg * _QQ_SCALE_KG_M2_PER_KGKG
    else:
        qq0 = np.asarray(column_water_mm, dtype=np.float64)

    eke = (
        np.full((h, w), SESAM_EKE_PLACEHOLDER_M2_S2)
        if eke_m2_s2 is None
        else np.asarray(eke_m2_s2, dtype=np.float64)
    )

    # Evaporation proxy: bulk-aerodynamic estimate, the same documented
    # placeholder diagnose_sesam_thermo.py already uses -- this project's
    # live evaporation code lives inside generate_precipitation/atmosphere.py
    # and is not exposed as a standalone field this module can read.
    # Wind speed driving the bulk formulas below is (A58)'s total (zonal +
    # synoptic gustiness) when the caller supplies it, not the bare zonal
    # magnitude -- see this function's own docstring, `total_wind_speed_m_s`.
    wind_speed = (
        np.sqrt(np.asarray(wind_u_m_s, dtype=np.float64) ** 2
                + np.asarray(wind_v_m_s, dtype=np.float64) ** 2)
        if total_wind_speed_m_s is None
        else np.asarray(total_wind_speed_m_s, dtype=np.float64)
    )
    qsat_skin = saturation_specific_humidity(tstar, p0_field)
    rho0 = p0 / (_RD_J_KG_K * _RHO0_REFERENCE_TEMP_K)
    evap_mm_day = np.maximum(qsat_skin - qa_kg_kg, 0.0) * wind_speed * _BULK_TRANSFER_COEFFICIENT * rho0 * 86400.0

    slope = _slope_magnitude(elev, dx_m, dy_m)

    # Water step first: it depends only on ta0 (via qa_kg_kg/evap above), not
    # on the energy step's output, so running it here lets its own (A44)
    # precipitation feed the real diabatic source below when P6d supplies one.
    water = evolve_column_water_vapor(
        qq0, evap_mm_day, wind_u_m_s, wind_v_m_s, eke,
        ra, qa_kg_kg, slope, land,
        dx_m=dx_m, dy_m=dy_m, dt_days=float(dt_days),
        cell_area_m2=area, x_face_length_m=x_face, y_face_length_m=y_face,
        rho0_kg_m3=rho0, implicit_zonal_diffusion=implicit_zonal_diffusion,
    )

    if sw_absorbed_w_m2 is not None and lw_net_w_m2 is not None:
        swa = np.asarray(sw_absorbed_w_m2, dtype=np.float64)
        lwa = np.asarray(lw_net_w_m2, dtype=np.float64)
        snow_frac = _snow_fraction(ta0)
        rainfall_mm_day = water.precipitation_mm_day.astype(np.float64) * (1.0 - snow_frac)
        snowfall_mm_day = water.precipitation_mm_day.astype(np.float64) * snow_frac
        sensible_heat_w_m2 = rho0 * _CP_AIR_J_KG_K * _BULK_TRANSFER_COEFFICIENT * wind_speed * (tstar - ta0)
        cv = column_heat_capacity_j_m2_k(p0, float(gravity_m_s2))
        diabatic_k_day = diabatic_heating_rate_k_day(
            swa, lwa, rainfall_mm_day, snowfall_mm_day, sensible_heat_w_m2, cv,
        )
    else:
        diabatic_k_day = (tstar - ta0) / SESAM_BRIDGE_RELAXATION_DAYS

    energy = evolve_column_energy(
        ta0, diabatic_k_day, wind_u_m_s, wind_v_m_s, eke,
        dx_m=dx_m, dy_m=dy_m, dt_days=float(dt_days),
        cell_area_m2=area, x_face_length_m=x_face, y_face_length_m=y_face,
        implicit_zonal_diffusion=implicit_zonal_diffusion,
    )
    ta_next = np.clip(energy.temperature_k.astype(np.float64), 150.0, 350.0)

    qa_next_kg_kg = water.water_mm.astype(np.float64) / _QQ_SCALE_KG_M2_PER_KGKG
    qsat_ta_next = saturation_specific_humidity(ta_next, p0_field)
    ra_next = np.clip(qa_next_kg_kg / np.maximum(qsat_ta_next, 1e-8), 0.0, 1.0)

    diagnostics = {
        "energy_relative_residual": energy.relative_residual,
        "water_relative_residual": water.relative_residual,
        "convergence_mm_day": water.convergence_mm_day,
        "slope_convergence_mm_day": water.slope_convergence_mm_day,
    }

    return SesamColumnClosureStep(
        air_temperature_k=ta_next.astype(np.float32),
        relative_humidity=ra_next.astype(np.float32),
        precipitation_mm_day=water.precipitation_mm_day.astype(np.float32),
        column_water_mm=water.water_mm.astype(np.float32),
        diagnostics=diagnostics,
    )


__all__ = [
    "SESAM_EKE_PLACEHOLDER_M2_S2",
    "SESAM_BRIDGE_RELAXATION_DAYS",
    "SesamColumnClosureStep",
    "sesam_column_closure_step",
]
