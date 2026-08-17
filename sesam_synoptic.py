"""SESAM dynamics: synoptic processes / eddy kinetic energy — Appendix A5.

Pure, read-only kernels for the CLIMBER-X/SESAM synoptic (EKE) closure,
transcribed from Appendix A5 of

    Willeit, M., Ganopolski, A., Robinson, A., and Edwards, N. R.:
    GMD 15, 5905–5948 (2022), https://doi.org/10.5194/gmd-15-5905-2022
    (CC-BY 4.0; equations cited below as (A#) against that paper).

This is SESAM stage P3 (docs/SESAM_GAP_ANALYSIS.md §7): the vertically
integrated eddy kinetic energy K, an Eady-baroclinicity-driven production,
drag dissipation, macroturbulent advection and diffusion, and the derived
synoptic quantities that close the surface-wind and transport diagnostics of
stage P2:

- (A53) production from the Eady baroclinic growth rate and the
  Brunt–Väisälä frequency (A54);
- (A55) dissipation ∝ (c3syn + c4syn·CD)·K^1.5;
- (A50)/(A51) macroturbulent diffusion coefficients AT = c5syn·√K and
  Aq = c6syn·K;
- (A56)/(A57) synoptic surface wind Usyn = c7syn·ε·cosα·√K and 700 hPa
  vertical velocity wsyn = c8syn·√K;
- (A58) total surface wind including the synoptic gustiness term, and
  (A59)/(A60) wind stress.

This closes the missing storminess component of the P2 surface wind (the
exit-gate measurement in §7 showed the P2 surface wind lacked it), and K is
the prognostic variable that future stages transport by advection/diffusion
per (A52).

**Verification notes** (checked 2026-08-16 against the article's HTML MathML
and, read-only per §5 licensing policy, the CLIMBER-X Fortran `synop.f90`
and namelist):

1.  **(A53) production uses the full Eady shear.** The paper prints
    ``PK = c1syn + c2syn·(f/N)·(∂u/∂z)``; the reference implementation uses
    ``f/N·√((∂u/∂z)² + (∂v/∂z)²)`` with ``f = 2Ω·|sinϕ|`` and both shear
    components — the Eady baroclinicity of Hoskins and Valdes (1990) that
    the accompanying text describes. This module implements the full form.
2.  **(A54) Brunt–Väisälä frequency.** The paper prints
    ``N = √((g/θ)·(∂θ/∂z))``; the reference implementation evaluates
    ``N² = (g/T)·ΔT/Δz`` between the 850 and 500 hPa levels (a finite
    difference with T the level temperature, not potential temperature).
    The general profile kernel uses the printed potential-temperature form;
    the `eke_production` default uses the reference's 850–500 hPa
    finite-difference proxy for consistency with the validated model.
3.  **Dissipation floor.** The reference clamps the EKE state to ≥ 1
    (m² s⁻²) after each step; the dissipation/production kernels here allow
    an explicit ``eke_floor`` parameter (default 1.0, matching the
    reference) so transient/production equations stay well behaved.
4.  **Topography damping of production** (reference safeguard, optional):
    multiplied by ``(1 − c8syn_ele·zsa/3000)`` where `c8syn_ele` is the
    namelist's `c_syn_8` (0 in the default namelist). The pack's ``c8syn``
    is the (A57) vertical-velocity coefficient and is unrelated; the
    topography-damping factor is exposed as an explicit optional parameter
    since it is not printed in (A53).
5.  **Derived synoptic quantities** (A56)/(A57) use ``ε`` and ``cosα``
    from the stage-P2 cross-isobar solve; ``Usyn`` is floored at the
    reference namelist's ``synsurmin`` = 1 m s⁻¹.

Grids follow the P1/P2 convention: 2-D fields ``(H, W)``, rows north-to-south
on cell centres; vertical profiles ``(N, H, W)`` on absolute-height levels.
Planetary constants are explicit inputs — never hardcoded Earth literals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sesam_reference import value as _param

# ---------------------------------------------------------------------------
# A5 parameter registry (single source: sesam_reference.py)
# ---------------------------------------------------------------------------


def _a5_defaults() -> dict[str, float]:
    """All A5 constants from the published pack."""
    return {
        "c1syn": _param("A5_synoptic", "c1syn"),
        "c2syn": _param("A5_synoptic", "c2syn"),
        "c3syn": _param("A5_synoptic", "c3syn"),
        "c4syn": _param("A5_synoptic", "c4syn"),
        "c5syn": _param("A5_synoptic", "c5syn"),
        "c6syn": _param("A5_synoptic", "c6syn"),
        "c7syn": _param("A5_synoptic", "c7syn"),
        "c8syn": _param("A5_synoptic", "c8syn"),
    }


# Reference-implementation safeguards / namelist defaults (module docstring
# notes 3–5).
_SYN_SURFACE_WIND_MIN_M_S = 1.0  # synsurmin
_OCEAN_WIND_MIN_M_S = 5.0  # windmin (ocean total wind floor)
_EKE_FLOOR_M2_S2 = 1.0  # reference clamps EKE >= 1


def _check_2d(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D (H, W) field")
    return arr


# ---------------------------------------------------------------------------
# (A54) Brunt–Väisälä frequency
# ---------------------------------------------------------------------------


def brunt_vaisala_frequency(
    potential_temperature_k: np.ndarray | None = None,
    temperature_k: np.ndarray | None = None,
    levels_m: np.ndarray | None = None,
    *,
    gravity: float,
) -> np.ndarray:
    """(A54) ``N = √((g/θ)·(∂θ/∂z))`` per column.

    With ``potential_temperature_k`` supplied as a ``(N, H, W)`` profile on
    ``levels_m`` (absolute, surface-first), returns ``(H, W)`` using the
    printed potential-temperature form.  This kernel is kept separate from
    :func:`eke_production`, whose default uses the reference implementation's
    850–500 hPa finite-difference proxy.
    """
    if potential_temperature_k is None or levels_m is None or temperature_k is not None:
        raise ValueError(
            "brunt_vaisala_frequency needs the potential-temperature profile and levels"
        )
    theta = np.asarray(potential_temperature_k, dtype=np.float64)
    levels = np.asarray(levels_m, dtype=np.float64)
    if theta.ndim != 3 or levels.ndim != 1 or levels.size != theta.shape[0]:
        raise ValueError("potential_temperature_k (N,H,W) and levels_m (N,) must match")
    dz = levels[-1] - levels[0]
    if dz <= 0.0:
        raise ValueError("levels_m must increase (surface-first)")
    dtheta_dz = (theta[-1] - theta[0]) / dz
    theta_mid = 0.5 * (theta[0] + theta[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        n2 = float(gravity) / theta_mid * dtheta_dz
    return np.sqrt(np.maximum(n2, 0.0))


def _vertical_shear(
    u_lo: np.ndarray,
    u_hi: np.ndarray,
    v_lo: np.ndarray,
    v_hi: np.ndarray,
    dz_m: float,
    latitude_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Meridional-consistent vertical shear between two levels (m s⁻¹ / m).

    The reference implementation weights the meridional shear by cosϕ
    (a grid-geometry factor for the 3-D wind on the T grid).  ``u/v`` pairs
    are the lower and upper level winds.
    """
    dudz = (u_hi - u_lo) / dz_m
    dvdz = (v_hi - v_lo) / dz_m * np.cos(latitude_rad)[:, None]
    return dudz, dvdz


def eady_growth_rate(
    u_wind_k: np.ndarray,
    v_wind_k: np.ndarray,
    potential_temperature_k: np.ndarray,
    *,
    levels_m: np.ndarray,
    pressure_pa: np.ndarray,
    gravity: float,
    omega: float,
    latitude_rad: np.ndarray,
    shear_lo_pa: float = 85000.0,
    shear_hi_pa: float = 50000.0,
) -> np.ndarray:
    """Eady baroclinic growth rate ``(f/N)·√((∂u/∂z)² + (∂v/∂z)²)``.

    ``(u_wind_k, v_wind_k, potential_temperature_k)`` are ``(N, H, W)``
    profiles on ``levels_m`` with per-column ``pressure_pa``.  The shear and
    Brunt–Väisälä frequency are evaluated between the 850 and 500 hPa levels
    (reference implementation): the vertical differences of u/v/θ between
    those two pressure surfaces divided by the geometric height of the layer
    in an exponential reference atmosphere, ``Δz = Ha·ln(p850/p500)`` with
    ``Ha = Rd·reference_temp/g``.  Potential temperature is used for the
    Brunt–Väisälä frequency (N² = (g/θ)·Δθ/Δz > 0 for a stably stratified
    troposphere) — the reference implementation's ``tp`` field is potential
    temperature.
    """
    # Reference-implementation Brunt–Vaisala proxy N^2 = (g/theta) dtheta/dz.
    th_lo = scalar_at_pressure(potential_temperature_k, pressure_pa, shear_lo_pa)
    th_hi = scalar_at_pressure(potential_temperature_k, pressure_pa, shear_hi_pa)
    u_lo = scalar_at_pressure(u_wind_k, pressure_pa, shear_lo_pa)
    u_hi = scalar_at_pressure(u_wind_k, pressure_pa, shear_hi_pa)
    v_lo = scalar_at_pressure(v_wind_k, pressure_pa, shear_lo_pa)
    v_hi = scalar_at_pressure(v_wind_k, pressure_pa, shear_hi_pa)

    # Height of the (lo..hi) layer in the exponential reference atmosphere.
    ha = (287.0 * 288.0) / float(gravity)  # Rd * T_ref / g
    dz = float(ha) * np.log(shear_lo_pa / shear_hi_pa)
    th_mid = 0.5 * (th_lo + th_hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        nfreq = np.sqrt(gravity / th_mid * (th_hi - th_lo) / dz)
    dudz, dvdz = _vertical_shear(u_lo, u_hi, v_lo, v_hi, dz, latitude_rad)
    f_abs = 2.0 * float(omega) * np.abs(np.sin(latitude_rad))[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(nfreq > 0.0, f_abs / nfreq * np.sqrt(dudz**2 + dvdz**2), 0.0)


# ---------------------------------------------------------------------------
# (A53) production, (A55) dissipation, (A50)/(A51) diffusion
# ---------------------------------------------------------------------------


def eke_production(
    eady_rate: np.ndarray,
    *,
    surface_elevation_m: np.ndarray | None = None,
    topography_damping_depth_m: float = 3000.0,
    topography_damping_coeff: float = 0.0,
    a5: dict[str, float] | None = None,
) -> np.ndarray:
    """(A53) ``PK = c1syn + c2syn·(f/N)·√((∂u/∂z)² + (∂v/∂z)²)·Fz_syn``.

    ``eady_rate`` is the Eady growth rate from :func:`eady_growth_rate`.  The
    optional topography damping ``(1 − c_top·zsa/3000)`` follows the reference
    implementation (default `topography_damping_coeff` = 0, i.e. inert, per
    the default namelist `c_syn_8 = 0`); it is deliberately not folded into
    the pack's ``c8syn`` (which is the (A57) vertical-velocity coefficient).
    """
    params = a5 or _a5_defaults()
    rate = np.asarray(eady_rate, dtype=np.float64)
    prod = params["c1syn"] + params["c2syn"] * rate
    if surface_elevation_m is not None and topography_damping_coeff != 0.0:
        zsa = _check_2d("surface_elevation_m", surface_elevation_m)
        prod = prod * (1.0 - topography_damping_coeff * zsa / topography_damping_depth_m)
    return np.maximum(prod, 0.0)


def eke_dissipation_coefficient(
    drag_coefficient: np.ndarray,
    a5: dict[str, float] | None = None,
) -> np.ndarray:
    """(A55) ``(c3syn + c4syn·CD)`` — the K^1.5 dissipation coefficient."""
    params = a5 or _a5_defaults()
    return params["c3syn"] + params["c4syn"] * _check_2d("drag_coefficient", drag_coefficient)


def eke_dissipation(
    eke: np.ndarray,
    drag_coefficient: np.ndarray,
    *,
    eke_floor: float = _EKE_FLOOR_M2_S2,
    a5: dict[str, float] | None = None,
) -> np.ndarray:
    """(A55) ``DK = (c3syn + c4syn·CD)·K^(3/2)``."""
    k = np.maximum(np.asarray(eke, dtype=np.float64), eke_floor)
    return eke_dissipation_coefficient(drag_coefficient, a5) * k ** (3.0 / 2.0)


def horizontal_diffusion_coefficient(
    eke: np.ndarray,
    *,
    eke_floor: float = _EKE_FLOOR_M2_S2,
    a5: dict[str, float] | None = None,
) -> np.ndarray:
    """(A50) ``AT = c5syn·√K`` — macroturbulent heat diffusivity [m² s⁻¹]."""
    params = a5 or _a5_defaults()
    k = np.maximum(np.asarray(eke, dtype=np.float64), eke_floor)
    return params["c5syn"] * np.sqrt(k)


def moisture_diffusion_coefficient(
    eke: np.ndarray,
    *,
    a5: dict[str, float] | None = None,
) -> np.ndarray:
    """(A51) ``Aq = c6syn·K`` — macroturbulent moisture diffusivity [s]."""
    params = a5 or _a5_defaults()
    return params["c6syn"] * np.maximum(np.asarray(eke, dtype=np.float64), 0.0)


# ---------------------------------------------------------------------------
# (A56) synoptic wind, (A57) vertical velocity, (A58) total wind, (A59/60) stress
# ---------------------------------------------------------------------------


def synoptic_surface_wind(
    eke: np.ndarray,
    epsilon: np.ndarray,
    cos_alpha: np.ndarray,
    *,
    eke_floor: float = _EKE_FLOOR_M2_S2,
    synsur_min_m_s: float = _SYN_SURFACE_WIND_MIN_M_S,
    a5: dict[str, float] | None = None,
) -> np.ndarray:
    """(A56) ``Usyn = c7syn·ε·cosα·√K`` — the synoptic gustiness component.

    ``epsilon`` and ``cos_alpha`` come from the stage-P2 cross-isobar solve
    (:func:`sesam_wind.cross_isobar_angle`).  Floored at ``synsur_min_m_s``
    (reference namelist ``synsurmin`` = 1 m s⁻¹).
    """
    params = a5 or _a5_defaults()
    k = np.maximum(np.asarray(eke, dtype=np.float64), eke_floor)
    usyn = (
        params["c7syn"]
        * np.asarray(epsilon, dtype=np.float64)
        * np.asarray(cos_alpha, dtype=np.float64)
        * np.sqrt(k)
    )
    return np.maximum(usyn, synsur_min_m_s)


def synoptic_vertical_velocity(
    eke: np.ndarray,
    *,
    eke_floor: float = _EKE_FLOOR_M2_S2,
    a5: dict[str, float] | None = None,
) -> np.ndarray:
    """(A57) ``wsyn = c8syn·√K`` — synoptic vertical velocity at 700 hPa."""
    params = a5 or _a5_defaults()
    k = np.maximum(np.asarray(eke, dtype=np.float64), eke_floor)
    return params["c8syn"] * np.sqrt(k)


def total_wind_magnitude(
    surface_u_m_s: np.ndarray,
    surface_v_m_s: np.ndarray,
    synoptic_u_m_s: np.ndarray,
    *,
    surface_elevation_km: np.ndarray | None = None,
    elevation_wind_scale_m_s_per_m: float = 0.0,
    ocean_mask: np.ndarray | None = None,
    ocean_wind_min_m_s: float = _OCEAN_WIND_MIN_M_S,
) -> np.ndarray:
    """(A58) ``Us = √(us² + vs² + Usyn²)`` (+ option elevation/ocean floors).

    ``synoptic_u_m_s`` is the (A56) magnitude.  The elevation term
    ``c_wind_ele·zs`` and the ocean wind floor follow the reference
    implementation (both 0/inert by default per the namelist).
    """
    us = np.asarray(surface_u_m_s, dtype=np.float64)
    vs = np.asarray(surface_v_m_s, dtype=np.float64)
    usyn = np.asarray(synoptic_u_m_s, dtype=np.float64)
    wind = np.sqrt(us**2 + vs**2 + usyn**2)
    if surface_elevation_km is not None and elevation_wind_scale_m_s_per_m != 0.0:
        wind = wind + elevation_wind_scale_m_s_per_m * np.asarray(surface_elevation_km, dtype=np.float64)
    if ocean_mask is not None:
        ocean = np.asarray(ocean_mask, dtype=bool)
        wind = np.where(ocean, np.maximum(wind, ocean_wind_min_m_s), wind)
    return wind


def wind_stress(
    surface_u_m_s: np.ndarray,
    surface_v_m_s: np.ndarray,
    total_wind_m_s: np.ndarray,
    drag_coefficient: np.ndarray,
    rho0_kg_m3: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(A59)/(A60) ``τλ = CD·ρ0·u·Us``, ``τϕ = CD·ρ0·v·Us``."""
    cd = _check_2d("drag_coefficient", drag_coefficient)
    taux = cd * float(rho0_kg_m3) * np.asarray(surface_u_m_s, dtype=np.float64) * total_wind_m_s
    tauy = cd * float(rho0_kg_m3) * np.asarray(surface_v_m_s, dtype=np.float64) * total_wind_m_s
    return taux, tauy


# ---------------------------------------------------------------------------
# Steady state and helpers
# ---------------------------------------------------------------------------


def eke_steady_state(
    production: np.ndarray,
    drag_coefficient: np.ndarray,
    *,
    eke_floor: float = _EKE_FLOOR_M2_S2,
    a5: dict[str, float] | None = None,
) -> np.ndarray:
    """Diagnostic equilibrium EKE where production = dissipation.

    ``PK = (c3syn + c4syn·CD)·K^(3/2)`` ⇒
    ``K_eq = (PK / (c3syn + c4syn·CD))^(2/3)``.  This gives the EKE the
    model would equilibrate to locally from the current baroclinicity alone
    (no advection/diffusion transport), the natural first diagnostic: with
    only the local production/dissipation balance it captures the
    storm-track amplitude.  Returns K floored at ``eke_floor``.
    """
    coeff = eke_dissipation_coefficient(drag_coefficient, a5)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.maximum(np.asarray(production, dtype=np.float64) / coeff, 0.0) ** (2.0 / 3.0)
    return np.maximum(np.nan_to_num(k, nan=eke_floor), eke_floor)


def eke_relaxation_tendency(
    eke: np.ndarray,
    production: np.ndarray,
    drag_coefficient: np.ndarray,
    *,
    eke_floor: float = _EKE_FLOOR_M2_S2,
    a5: dict[str, float] | None = None,
) -> np.ndarray:
    """(A52) local production−dissipation tendency ``PK − DK`` [m² s⁻³]."""
    return np.asarray(production, dtype=np.float64) - eke_dissipation(
        eke, drag_coefficient, eke_floor=eke_floor, a5=a5
    )


def scalar_at_pressure(
    profile: np.ndarray,
    pressure_pa: np.ndarray,
    target_pa: float,
) -> np.ndarray:
    """Interpolate a ``(N, H, W)`` profile to a pressure surface, per column.

    Clamps to the nearest level outside the column's pressure range.
    """
    p = np.asarray(profile, dtype=np.float64)
    pz = np.asarray(pressure_pa, dtype=np.float64)
    if p.ndim != 3 or pz.shape != p.shape:
        raise ValueError("profile and pressure_pa must share a 3-D shape")
    n, h, w = p.shape
    out = np.empty((h, w))
    target = float(target_pa)
    for j in range(h):
        for i in range(w):
            col_p = pz[:, j, i]
            col_v = p[:, j, i]
            if target >= col_p[0]:
                out[j, i] = col_v[0]
            elif target <= col_p[-1]:
                out[j, i] = col_v[-1]
            else:
                out[j, i] = np.interp(target, col_p[::-1], col_v[::-1])
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SesamSynoptic:
    """Assembled synoptic/EKE diagnostic fields, all ``(H, W)`` (north-to-south)."""

    eddy_kinetic_energy_m2_s2: np.ndarray
    eke_steady_state_m2_s2: np.ndarray
    production_m2_s3: np.ndarray
    dissipation_m2_s3: np.ndarray
    local_tendency_m2_s3: np.ndarray
    eady_growth_rate: np.ndarray
    brunt_vaisala_frequency: np.ndarray
    diffusion_coefficient_heat_m2_s: np.ndarray
    diffusion_coefficient_moisture_s: np.ndarray
    synoptic_surface_wind_m_s: np.ndarray
    synoptic_vertical_velocity_m_s: np.ndarray
    total_wind_m_s: np.ndarray
    wind_stress_u_pa: np.ndarray
    wind_stress_v_pa: np.ndarray


def compute_synoptic(
    *,
    potential_temperature_k: np.ndarray,
    u_wind_z: np.ndarray,
    v_wind_z: np.ndarray,
    pressure_z: np.ndarray,
    levels_m: np.ndarray,
    surface_u_m_s: np.ndarray,
    surface_v_m_s: np.ndarray,
    surface_elevation_m: np.ndarray,
    surface_kind: np.ndarray,
    drag_coefficient: np.ndarray,
    epsilon: np.ndarray,
    cos_alpha: np.ndarray,
    gravity: float,
    omega: float,
    rho0_kg_m3: float,
    a5: dict[str, float] | None = None,
) -> SesamSynoptic:
    """Assemble the synoptic/EKE diagnostics for a model state.

    ``potential_temperature_k``/``u_wind_z``/``v_wind_z``/``pressure_z`` are
    ``(N, H, W)`` profiles on ``levels_m`` (from `sesam_vertical`);
    ``surface_u_m_s``/``surface_v_m_s`` are the stage-P2 surface wind;
    ``epsilon``/``cos_alpha``/``sin_alpha`` come from the stage-P2
    cross-isobar solve; ``drag_coefficient`` from
    ``sesam_wind.drag_coefficient``.  ``surface_kind`` follows P1's encoding
    (0 ocean, 1 land, 2 ice).

    ``eddy_kinetic_energy_m2_s2`` is the *steady-state* EKE from the local
    production/dissipation balance (transport of K is stage P4 and is not
    included here).
    """
    eady = eady_growth_rate(
        u_wind_z,
        v_wind_z,
        potential_temperature_k,
        levels_m=levels_m,
        pressure_pa=pressure_z,
        gravity=gravity,
        omega=omega,
        latitude_rad=_latitude_rad(potential_temperature_k.shape[1]),
    )
    production = eke_production(eady, surface_elevation_m=surface_elevation_m, a5=a5)
    keq = eke_steady_state(production, drag_coefficient, a5=a5)
    k = keq
    dissipation = eke_dissipation(k, drag_coefficient, a5=a5)
    tendency = eke_relaxation_tendency(k, production, drag_coefficient, a5=a5)
    nfreq = _nfreq_from_gradient(potential_temperature_k, pressure_z, gravity)
    at = horizontal_diffusion_coefficient(k, a5=a5)
    aq = moisture_diffusion_coefficient(k, a5=a5)
    usyn = synoptic_surface_wind(k, epsilon, cos_alpha, a5=a5)
    wsyn = synoptic_vertical_velocity(k, a5=a5)

    # Total wind and stress (A58)-(A60); ocean cells floored at 5 m/s.
    ocean_mask = np.asarray(surface_kind) == 0
    total_wind = total_wind_magnitude(surface_u_m_s, surface_v_m_s, usyn, ocean_mask=ocean_mask)
    taux, tauy = wind_stress(surface_u_m_s, surface_v_m_s, total_wind, drag_coefficient, rho0_kg_m3)

    return SesamSynoptic(
        eddy_kinetic_energy_m2_s2=k,
        eke_steady_state_m2_s2=keq,
        production_m2_s3=production,
        dissipation_m2_s3=dissipation,
        local_tendency_m2_s3=tendency,
        eady_growth_rate=eady,
        brunt_vaisala_frequency=nfreq,
        diffusion_coefficient_heat_m2_s=at,
        diffusion_coefficient_moisture_s=aq,
        synoptic_surface_wind_m_s=usyn,
        synoptic_vertical_velocity_m_s=wsyn,
        total_wind_m_s=total_wind,
        wind_stress_u_pa=taux,
        wind_stress_v_pa=tauy,
    )


def _nfreq_from_gradient(
    potential_temperature_k: np.ndarray,
    pressure_z: np.ndarray,
    gravity: float,
) -> np.ndarray:
    """Brunt–Väisälä frequency from the 850–500 hPa potential-temperature field."""
    th_lo = scalar_at_pressure(potential_temperature_k, pressure_z, 85000.0)
    th_hi = scalar_at_pressure(potential_temperature_k, pressure_z, 50000.0)
    ha = (287.0 * 288.0) / float(gravity)
    dz = float(ha) * np.log(85000.0 / 50000.0)
    th_mid = 0.5 * (th_lo + th_hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.maximum(gravity / th_mid * (th_hi - th_lo) / dz, 0.0))


def _latitude_rad(h: int) -> np.ndarray:
    return (0.5 - (np.arange(h, dtype=np.float64) + 0.5) / h) * np.pi


__all__ = [
    "SesamSynoptic",
    "brunt_vaisala_frequency",
    "compute_synoptic",
    "eady_growth_rate",
    "eke_dissipation",
    "eke_dissipation_coefficient",
    "eke_production",
    "eke_relaxation_tendency",
    "eke_steady_state",
    "horizontal_diffusion_coefficient",
    "moisture_diffusion_coefficient",
    "scalar_at_pressure",
    "synoptic_surface_wind",
    "synoptic_vertical_velocity",
    "total_wind_magnitude",
    "wind_stress",
]
