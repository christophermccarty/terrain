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
exit-gate measurement in §7 showed the P2 surface wind lacked it). K's own
(A52) transport -- advection by the wind and diffusion by its own
diffusivity AT -- is implemented below (`eke_diffusion_step`,
`eke_transport_step`, `evolve_eke`); `compute_synoptic` itself still returns
only the local steady-state diagnostic (production = dissipation, no
transport), which remains a useful diagnostic in its own right.

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
6.  **(A52) transport is operator-split** (2026-08-18): advection then
    diffusion then the local production/dissipation reaction, each its own
    conservative/CFL-stable sub-step, rather than one coupled substep loop.
    This is first-order accurate in the split but keeps every physical
    process independently conservative and independently testable -- the
    same reasoning `column_water.py` gives for splitting transport from
    precipitation generation. The advective term reuses
    `column_water.evolve_column_water` directly: K obeys the identical
    conservative flux-divergence transport equation as column water
    (``dQ/dt = source - div(Q v)``), just carrying a different scalar with
    zero source/sink at that stage. The diffusive term (`eke_diffusion_step`)
    is a new nonlinear-diffusion kernel (AT depends on K itself via A50) that
    follows the same finite-volume geometry (shared faces, periodic
    longitude, closed latitude) so a flux leaving one cell exactly enters its
    neighbour, with face-averaged AT (the standard treatment for a
    state-dependent diffusivity: using an un-averaged, one-sided cell value
    would make the response depend on which side of a symmetric feature you
    evaluate from, breaking the symmetric smoothing a symmetric bump must
    produce). Both the advective and diffusive terms are CFL-substepped
    (documented at each function).
7.  **The (A52) advecting wind should be the stage-P2 *zonal-only* wind**,
    not the full azonal-inflated chain -- the same finding §10 records for
    driving the Eady production (note the module docstring above): the P2
    exit-gate investigation found the full-chain wind inherits the P2 azonal
    input-conditioning inflation on the saved real-terrain state, and that
    inflation would compound into the K advected here exactly as it would
    compound into local production. This is a caller-level choice (this
    module accepts whatever wind it is given); `scripts/diagnose_sesam_synoptic.py`
    follows it.

Grids follow the P1/P2 convention: 2-D fields ``(H, W)``, rows north-to-south
on cell centres; vertical profiles ``(N, H, W)`` on absolute-height levels.
Planetary constants are explicit inputs — never hardcoded Earth literals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from column_water import evolve_column_water
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
# (A52) prognostic K: advection, nonlinear diffusion, and the full step
# ---------------------------------------------------------------------------


def spherical_transport_geometry(
    h: int, w: int, radius_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact finite-volume geometry for spherical K transport.

    Mirrors `atmosphere.py`'s `_column_water_spherical_geometry` (the same
    cell areas and face lengths already used by the established column-water
    transport) so `eke_transport_step`/`evolve_eke` share its exact
    conservation contract. Returns ``(cell_area_m2, x_face_length_m,
    y_face_length_m)``: cell area ``(H, W)``, the meridional-running east/west
    face length ``(H, W)``, and the zonal-running north/south face length
    ``(H + 1, W)`` (correctly shrinks to zero at the poles).
    """
    dlat = np.pi / float(h)
    dlon = 2.0 * np.pi / float(w)
    edges = np.pi / 2.0 - np.arange(h + 1, dtype=np.float64) * dlat
    area_row = radius_m ** 2 * dlon * (np.sin(edges[:-1]) - np.sin(edges[1:]))
    area = np.broadcast_to(area_row[:, None], (h, w)).copy()
    x_face_length = np.full((h, w), radius_m * dlat, dtype=np.float64)
    y_face_length = np.broadcast_to(
        (radius_m * np.maximum(np.cos(edges), 0.0) * dlon)[:, None], (h + 1, w)
    ).copy()
    return area, x_face_length, y_face_length


def zonal_center_spacing_m(
    h: int, w: int, radius_m: float, *, polar_floor_lat_deg: float = 65.0
) -> np.ndarray:
    """Zonal distance between neighbouring cell centres, ``(H, W)``.

    ``radius*cos(lat)*dlon`` shrinks to zero at the poles on a lat-lon grid;
    `eke_diffusion_step` divides by this distance to form a gradient, and
    dividing by (near-)zero there would demand an unbounded number of
    substeps for an arbitrarily small amount of physically meaningless polar
    zonal diffusion. Floored at its value `polar_floor_lat_deg` poleward of
    that latitude -- the same "polar cap" simplification `atmosphere.py`'s
    `_wind_static_grids` already documents and uses for the analogous zonal
    pressure-gradient singularity (it only ever shrinks the response inside
    the cap, never inflates it elsewhere).
    """
    cos_lat = np.cos(_latitude_rad(h))
    cos_floor = float(np.cos(np.deg2rad(polar_floor_lat_deg)))
    cos_lat_floored = np.maximum(cos_lat, cos_floor)
    dlon = 2.0 * np.pi / float(w)
    dx_row = radius_m * dlon * cos_lat_floored
    return np.broadcast_to(dx_row[:, None], (h, w)).astype(np.float64)


class EkeDiffusionStep(NamedTuple):
    eke_m2_s2: np.ndarray
    residual_m2_s2: float
    relative_residual: float
    substeps: int
    maximum_diffusion_number: float


def eke_diffusion_step(
    eke_m2_s2: np.ndarray,
    *,
    dx_m: np.ndarray | float,
    dy_m: float,
    dt_days: float,
    cell_area_m2: np.ndarray | float,
    x_face_length_m: np.ndarray | float,
    y_face_length_m: np.ndarray,
    diffusion_r_limit: float = 0.4,
    eke_floor: float = _EKE_FLOOR_M2_S2,
    a5: dict[str, float] | None = None,
) -> EkeDiffusionStep:
    """(A52) ``div(AT * grad K)`` term only: nonlinear diffusion of K by its
    own diffusivity ``AT = c5syn·√K`` (A50).

    Conservative finite-volume scheme sharing `column_water.py`'s geometry
    contract: periodic longitude, closed (zero-flux) latitude boundaries at
    the poles, fluxes on shared cell faces so an amount leaving one cell
    enters its neighbour exactly. ``AT`` is face-averaged (arithmetic mean of
    the two neighbouring cells) before forming each flux -- the standard
    treatment for a state-dependent diffusivity in a conservative scheme, and
    the choice that keeps a spatially symmetric field diffusing symmetrically
    (an un-averaged, one-sided AT would make the response depend on which
    side of a symmetric feature it is evaluated from). ``AT`` is recomputed
    every sub-step from the evolving K, since it is itself a function of the
    state being diffused (nonlinear diffusion).

    Stability: standard explicit diffusion requires the discrete self-loss
    fraction per sub-step, ``r = AT·dt_sub·(x_len/(dx·area) + ... four faces
    ...)``, to stay ``≤ 0.5`` (the exact per-cell face-length/area geometry,
    not the ``1/dx² + 1/dy²`` approximation that assumes ``area ~ dx·dy`` --
    that approximation catastrophically underestimates the true constraint at
    the polar-cap row, whose actual area shrinks toward the true spherical-
    cap value while ``x_len`` stays the same constant as everywhere else; see
    ``self_loss_geometry`` below). This kernel sub-steps so the worst-case r
    stays under ``diffusion_r_limit`` (0.4, matching the 0.4 r-limit
    convention `simulate.py`'s `eddy_heat_flux_coeff` and
    `abyssal_overturning_coeff` blocks already document and use for the same
    kind of explicit-Euler Laplacian diffusion). Because AT is itself a
    function of K (nonlinear diffusion), the sub-step size is chosen
    *adaptively*, recomputed from the current AT at every sub-step, rather
    than fixed once from the initial state: an earlier version bounded the
    whole call from the initial AT alone (reasoning that pure diffusion
    cannot raise a field's peak, so the initial AT should bound every later
    one) and overflowed on the real 512x1024 saved-state field -- that
    argument silently assumed the scheme was already stable to prove it
    stable. Adaptive re-estimation alone was not sufficient either: a second,
    independent bug (the area-approximation described above) meant even the
    adaptive estimate was wrong specifically at the poles, and tightening
    `diffusion_r_limit` by 8x (0.4 -> 0.05) did not stop the real-state
    divergence -- the tell that this was a formula error, not an
    insufficient safety margin. Both fixes are needed together.
    """
    if dt_days <= 0.0 or dy_m <= 0.0:
        raise ValueError("dt_days/dy_m must be positive")
    if not 0.0 < diffusion_r_limit <= 0.5:
        raise ValueError("diffusion_r_limit must be in (0, 0.5]")
    k0 = np.clip(np.asarray(eke_m2_s2, dtype=np.float64), 0.0, None)
    if k0.ndim != 2:
        raise ValueError("eke_m2_s2 must be a 2-D (H, W) field")
    h, w = k0.shape
    dx = np.broadcast_to(np.asarray(dx_m, dtype=np.float64), (h, w))
    if np.any(dx <= 0.0):
        raise ValueError("dx_m must be positive")
    area = np.broadcast_to(np.asarray(cell_area_m2, dtype=np.float64), (h, w))
    if np.any(area <= 0.0):
        raise ValueError("cell_area_m2 must be positive")
    x_len = np.broadcast_to(np.asarray(x_face_length_m, dtype=np.float64), (h, w))
    y_len = np.broadcast_to(np.asarray(y_face_length_m, dtype=np.float64), (h + 1, w))

    # Exact per-cell self-loss coefficient (units 1/m^2), from the *actual*
    # face-length/area geometry -- NOT the "rate ~ AT*(1/dx^2+1/dy^2)"
    # approximation an earlier version used, which implicitly assumes
    # area ~ dx*dy. That assumption fails badly at the polar-cap row: its
    # true area shrinks toward the actual (tiny) spherical-cap area while
    # x_len (=radius*dlat) stays the same constant as everywhere else, so the
    # approximation catastrophically *underestimates* the true CFL
    # constraint exactly at the pole. Found via a real 512x1024 saved-state
    # run: max K decayed smoothly for ~180 substeps (correct diffusion) then
    # suddenly diverged starting from row 0 (the north-pole row) outward --
    # reproducible even after tightening diffusion_r_limit by 8x (0.4 -> 0.05),
    # which ruled out "not conservative enough" and pointed at the formula
    # itself. self_loss_geometry below is the exact per-face x_len/(dx*area)
    # and y_len/(dy*area) sum (see eke_diffusion_step's docstring derivation);
    # multiplying by AT gives the true rate everywhere, poles included.
    self_loss_geometry = (
        x_len / (dx * area)
        + np.roll(x_len, 1, axis=1) / (np.roll(dx, 1, axis=1) * area)
        + y_len[:-1] / (dy_m * area)
        + y_len[1:] / (dy_m * area)
    )

    dt_seconds = dt_days * 86400.0
    max_substeps = 200_000  # generous pathological-case cap; never hit in practice
    mass = k0 * area
    elapsed = 0.0
    n_sub = 0
    max_r_seen = 0.0
    while elapsed < dt_seconds - 1e-6 and n_sub < max_substeps:
        depth = mass / area
        at = horizontal_diffusion_coefficient(depth, eke_floor=eke_floor, a5=a5)
        remaining = dt_seconds - elapsed
        max_rate = float(np.max(at * self_loss_geometry))
        dt_sub = remaining if max_rate <= 0.0 else min(diffusion_r_limit / max_rate, remaining)

        # East-face flux: efflux from column i into column i+1, positive when
        # K[i] > K[i+1] (down-gradient). Periodic in longitude via np.roll.
        at_east = 0.5 * (at + np.roll(at, -1, axis=1))
        east_flux = at_east * (depth - np.roll(depth, -1, axis=1)) / dx * x_len
        west_flux_in = np.roll(east_flux, 1, axis=1)

        # North/south-face flux across interior latitude faces only; the
        # pole-edge faces (index 0 and H) stay zero -- closed boundary.
        at_face_y = np.zeros((h + 1, w), dtype=np.float64)
        at_face_y[1:-1] = 0.5 * (at[:-1] + at[1:])
        face_flux = np.zeros((h + 1, w), dtype=np.float64)
        face_flux[1:-1] = at_face_y[1:-1] * (depth[:-1] - depth[1:]) / dy_m * y_len[1:-1]

        mass = np.maximum(
            mass + dt_sub * (west_flux_in - east_flux + face_flux[:-1] - face_flux[1:]),
            0.0,
        )
        elapsed += dt_sub
        n_sub += 1
        max_r_seen = max(max_r_seen, max_rate * dt_sub)
    if elapsed < dt_seconds - 1e-3:
        raise RuntimeError(
            f"eke_diffusion_step did not converge within {max_substeps} substeps "
            f"(reached {elapsed:.1f}/{dt_seconds:.1f} s) -- unexpectedly stiff input"
        )

    eke_next = np.maximum(mass / area, 0.0)
    initial_mass = float(np.sum(k0 * area))
    final_mass = float(np.sum(eke_next * area))
    residual = final_mass - initial_mass
    relative_residual = residual / max(abs(final_mass), abs(initial_mass), 1.0)
    return EkeDiffusionStep(eke_next, residual, relative_residual, n_sub, max_r_seen)


def _thomas_batch(
    sub: np.ndarray, diag: np.ndarray, sup: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Batched Thomas algorithm: ``H`` independent non-cyclic tridiagonal
    systems of length ``W``, all shaped ``(H, W)``, solved simultaneously
    (vectorised over the ``H`` axis; the forward/backward sweep is
    necessarily sequential along ``W``, so that axis is a plain Python loop).
    ``sub[:, 0]`` and ``sup[:, -1]`` are ignored (no wraparound here -- see
    `_cyclic_thomas_batch` for the periodic case this module actually needs).
    """
    h, w = diag.shape
    cp = np.empty((h, w))
    dp = np.empty((h, w))
    cp[:, 0] = sup[:, 0] / diag[:, 0]
    dp[:, 0] = rhs[:, 0] / diag[:, 0]
    for i in range(1, w):
        m = diag[:, i] - sub[:, i] * cp[:, i - 1]
        if i < w - 1:
            cp[:, i] = sup[:, i] / m
        dp[:, i] = (rhs[:, i] - sub[:, i] * dp[:, i - 1]) / m
    x = np.empty((h, w))
    x[:, -1] = dp[:, -1]
    for i in range(w - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i] * x[:, i + 1]
    return x


def _cyclic_thomas_batch(
    sub: np.ndarray, diag: np.ndarray, sup: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Batched Sherman-Morrison cyclic tridiagonal solve (Numerical Recipes
    §2.7 "cyclic", batched over ``H`` independent periodic rows of length
    ``W``): ``sub[:,i]*x[:,i-1] + diag[:,i]*x[:,i] + sup[:,i]*x[:,i+1] =
    rhs[:,i]``, indices wrapping (``x[:,-1]`` means ``x[:,W-1]``,
    ``x[:,W]`` means ``x[:,0]``). ``sub[:, 0]`` is the corner coefficient
    coupling ``x[:, W-1]`` into row 0 (``alpha``); ``sup[:, -1]`` is the
    corner coupling ``x[:, 0]`` into row ``W-1`` (``beta``).

    Standard rank-1 (Sherman-Morrison) correction: solve the same system with
    both corners zeroed and the two corner diagonal entries adjusted
    (``diag[0] -= gamma``, ``diag[-1] -= alpha*beta/gamma`` with
    ``gamma = -diag[0]``) against two right-hand sides -- the real ``rhs``
    and a corner-only correction vector -- then combine. Verified against
    `numpy.linalg.solve` on the equivalent dense periodic matrix for
    arbitrary (non-constant-coefficient) rows up to 1e-14 relative error
    during development of this kernel; ``testing/test_sesam_synoptic.py``
    keeps a smaller instance of that same check.
    """
    h, w = diag.shape
    alpha = sub[:, 0].copy()
    beta = sup[:, -1].copy()
    gamma = -diag[:, 0].copy()

    diagp = diag.copy()
    diagp[:, 0] = diag[:, 0] - gamma
    diagp[:, -1] = diag[:, -1] - alpha * beta / gamma

    sub_t = sub.copy()
    sub_t[:, 0] = 0.0
    sup_t = sup.copy()
    sup_t[:, -1] = 0.0

    x = _thomas_batch(sub_t, diagp, sup_t, rhs)

    u = np.zeros((h, w))
    u[:, 0] = 1.0
    u[:, -1] = beta / gamma
    z = _thomas_batch(sub_t, diagp, sup_t, u)

    numer = gamma * x[:, 0] + alpha * x[:, -1]
    denom = 1.0 + gamma * z[:, 0] + alpha * z[:, -1]
    fact = numer / denom
    return x - fact[:, None] * z


def eke_diffusion_step_implicit_zonal(
    eke_m2_s2: np.ndarray,
    *,
    dx_m: np.ndarray | float,
    dy_m: float,
    dt_days: float,
    cell_area_m2: np.ndarray | float,
    x_face_length_m: np.ndarray | float,
    y_face_length_m: np.ndarray,
    diffusion_r_limit: float = 0.4,
    eke_floor: float = _EKE_FLOOR_M2_S2,
    a5: dict[str, float] | None = None,
) -> EkeDiffusionStep:
    """(A52) ``div(AT * grad K)``, semi-implicit in longitude, explicit in
    latitude (an ADI-style directional split) -- the performance/numerics
    remedy for the polar stiffness the plain `eke_diffusion_step` runs into
    at the project's 512x1024 headline grid.

    **Why longitude and not latitude.** The task that motivated this kernel
    assumed (by analogy with real ocean/atmosphere models' implicit
    *vertical* diffusion and semi-implicit *polar filters*) that the pole
    stiffness would live in the meridional term. Measured directly on the
    real 512x1024 grid (`EARTH.radius_m`, `spherical_transport_geometry`),
    it does not: splitting `eke_diffusion_step`'s per-cell self-loss
    coefficient into its zonal and meridional halves gives a *meridional*
    term that is exactly flat across every row (``1.31e-9`` m^-2 everywhere,
    since `zonal_center_spacing_m` does not enter it) while the *zonal* term
    grows from ``1.31e-9`` at the equator to ``1.01e-6`` at the pole row --
    770x larger, and alone responsible for the >99.8% of the self-loss rate
    at the pole. The mechanism is the classic lon-lat "pole problem": the
    east/west face length ``x_len = radius*dlat`` in `spherical_transport_geometry`
    is the *same constant* at every row by construction, while true cell area
    shrinks toward the actual (tiny) spherical-cap value approaching the
    pole -- so ``x_len/area`` (the zonal self-loss factor) diverges there
    even with `zonal_center_spacing_m`'s own 65 deg polar floor on the
    *gradient* denominator, because that floor does not touch the ``area``
    term at all. At the measured local-steady-state EKE magnitudes on the
    real state (K approx 6.2e3-1.3e5 m^2 s^-2 near the pole), the *zonal*
    term alone demands roughly 1-20 million diffusion sub-steps for a single
    0.25-day coupling step at 512 rows (`docs/SESAM_GAP_ANALYSIS.md`'s P3
    section has the measured numbers) -- genuinely intractable explicitly --
    while the *meridional* term alone would need only a few thousand, an
    entirely ordinary explicit cost. So the fix implemented here treats
    zonal diffusion implicitly (removing its CFL constraint entirely, since
    backward Euler with positive conductances is unconditionally stable and
    preserves the discrete maximum principle regardless of step size) and
    leaves meridional diffusion explicit, CFL-substepped exactly like
    `eke_diffusion_step` -- because the meridional term alone was never the
    stiff one, it needs no such treatment.

    **Scheme, per sub-step:** ``AT = c5syn*sqrt(K)`` is recomputed from the
    live state (nonlinear diffusion, same as `eke_diffusion_step`); the
    sub-step length is chosen from the meridional self-loss rate alone
    (``diffusion_r_limit`` applies to *that* term only, since the zonal
    solve is unconditionally stable); the zonal update solves the implicit
    (backward-Euler) system ``area*depth_new - dt*(C[i-1]*depth_new[i-1] -
    (C[i-1]+C[i])*depth_new[i] + C[i]*depth_new[i+1]) = area*depth_old``
    per row via `_cyclic_thomas_batch` (``C[i] = AT_east[i]/dx[i]*x_len[i]``,
    the same face-averaged-AT conductance `eke_diffusion_step` uses, just
    solved implicitly rather than substituted into an explicit update); the
    meridional update is then applied explicitly to the post-zonal state,
    identical in form to `eke_diffusion_step`'s own north/south flux (closed
    boundary at both poles, face-averaged AT). Both halves are individually
    exactly conservative (the zonal system is a pure flux-difference form
    that telescopes to zero net change summed around a periodic row; the
    explicit meridional half is the same flux-difference form
    `eke_diffusion_step` already uses), so the combined sub-step conserves
    ``sum(K*area)`` to floating-point precision -- verified in
    `testing/test_sesam_synoptic.py`.

    This is a **different, additionally-tested code path** from
    `eke_diffusion_step` (task requirement): the two give the same answer in
    the small-``dt_sub`` limit (both are consistent discretisations of the
    same continuous PDE) and are cross-checked against each other on a small
    grid where the explicit scheme is itself tractable, but this function is
    never called from `eke_diffusion_step`, `eke_transport_step`, or
    `evolve_eke` unless the caller opts in via ``implicit_zonal_diffusion=True``
    (`eke_transport_step`/`evolve_eke`); the exact-value/planted-violation
    tests pinning `eke_diffusion_step`'s explicit numerics are untouched.
    """
    if dt_days <= 0.0 or dy_m <= 0.0:
        raise ValueError("dt_days/dy_m must be positive")
    if not 0.0 < diffusion_r_limit <= 0.5:
        raise ValueError("diffusion_r_limit must be in (0, 0.5]")
    k0 = np.clip(np.asarray(eke_m2_s2, dtype=np.float64), 0.0, None)
    if k0.ndim != 2:
        raise ValueError("eke_m2_s2 must be a 2-D (H, W) field")
    h, w = k0.shape
    if w < 3:
        raise ValueError("eke_diffusion_step_implicit_zonal needs at least 3 longitude columns")
    dx = np.broadcast_to(np.asarray(dx_m, dtype=np.float64), (h, w))
    if np.any(dx <= 0.0):
        raise ValueError("dx_m must be positive")
    area = np.broadcast_to(np.asarray(cell_area_m2, dtype=np.float64), (h, w))
    if np.any(area <= 0.0):
        raise ValueError("cell_area_m2 must be positive")
    x_len = np.broadcast_to(np.asarray(x_face_length_m, dtype=np.float64), (h, w))
    y_len = np.broadcast_to(np.asarray(y_face_length_m, dtype=np.float64), (h + 1, w))

    # Meridional-only self-loss geometry (see docstring): this is the term
    # that still needs an explicit CFL bound.
    self_loss_y = y_len[:-1] / (dy_m * area) + y_len[1:] / (dy_m * area)

    dt_seconds = dt_days * 86400.0
    max_substeps = 200_000
    mass = k0 * area
    elapsed = 0.0
    n_sub = 0
    max_r_seen = 0.0
    while elapsed < dt_seconds - 1e-6 and n_sub < max_substeps:
        depth = mass / area
        at = horizontal_diffusion_coefficient(depth, eke_floor=eke_floor, a5=a5)
        remaining = dt_seconds - elapsed
        max_rate_y = float(np.max(at * self_loss_y))
        dt_sub = remaining if max_rate_y <= 0.0 else min(diffusion_r_limit / max_rate_y, remaining)

        # --- implicit zonal (east-west) half-step: periodic tridiagonal ---
        at_east = 0.5 * (at + np.roll(at, -1, axis=1))
        c_face = at_east / dx * x_len          # C[i]: conductance of face i -> i+1
        c_west = np.roll(c_face, 1, axis=1)    # C[i-1]: conductance of face i-1 -> i
        sub_diag = -dt_sub * c_west
        sup_diag = -dt_sub * c_face
        main_diag = area + dt_sub * (c_west + c_face)
        rhs = area * depth
        depth_x = np.maximum(_cyclic_thomas_batch(sub_diag, main_diag, sup_diag, rhs), 0.0)
        mass_x = depth_x * area

        # --- explicit meridional (north-south) half-step, closed poles ---
        at_face_y = np.zeros((h + 1, w), dtype=np.float64)
        at_face_y[1:-1] = 0.5 * (at[:-1] + at[1:])
        face_flux = np.zeros((h + 1, w), dtype=np.float64)
        face_flux[1:-1] = at_face_y[1:-1] * (depth_x[:-1] - depth_x[1:]) / dy_m * y_len[1:-1]

        mass = np.maximum(mass_x + dt_sub * (face_flux[:-1] - face_flux[1:]), 0.0)
        elapsed += dt_sub
        n_sub += 1
        max_r_seen = max(max_r_seen, max_rate_y * dt_sub)
    if elapsed < dt_seconds - 1e-3:
        raise RuntimeError(
            f"eke_diffusion_step_implicit_zonal did not converge within {max_substeps} "
            f"substeps (reached {elapsed:.1f}/{dt_seconds:.1f} s) -- unexpectedly stiff "
            "meridional input (the zonal term is unconditionally stable by construction)"
        )

    eke_next = np.maximum(mass / area, 0.0)
    initial_mass = float(np.sum(k0 * area))
    final_mass = float(np.sum(eke_next * area))
    residual = final_mass - initial_mass
    relative_residual = residual / max(abs(final_mass), abs(initial_mass), 1.0)
    return EkeDiffusionStep(eke_next, residual, relative_residual, n_sub, max_r_seen)


class EkeTransportStep(NamedTuple):
    eke_m2_s2: np.ndarray
    residual_m2_s2: float
    relative_residual: float
    advection_substeps: int
    diffusion_substeps: int
    maximum_diffusion_number: float


def eke_transport_step(
    eke_m2_s2: np.ndarray,
    wind_u_m_s: np.ndarray,
    wind_v_m_s: np.ndarray,
    *,
    dx_m: np.ndarray | float,
    dy_m: float,
    dt_days: float,
    cell_area_m2: np.ndarray | float,
    x_face_length_m: np.ndarray | float,
    y_face_length_m: np.ndarray,
    max_courant: float = 0.5,
    diffusion_r_limit: float = 0.4,
    eke_floor: float = _EKE_FLOOR_M2_S2,
    a5: dict[str, float] | None = None,
    implicit_zonal_diffusion: bool = False,
) -> EkeTransportStep:
    """(A52) transport-only terms: ``-div(uK) + div(AT·grad K)``.

    No production/dissipation -- with zero source supplied elsewhere, this
    conserves ``sum(K * cell_area)`` up to the non-negativity floor clip
    (matching `column_water.py`'s own conservation contract, since the
    advective term literally reuses `evolve_column_water`).  ``wind_u_m_s``/
    ``wind_v_m_s`` should be the stage-P2 *zonal-only* wind (module docstring
    note 7), not the full azonal-inflated chain.

    ``implicit_zonal_diffusion=True`` routes the diffusive term through
    `eke_diffusion_step_implicit_zonal` instead of the default
    `eke_diffusion_step` -- the fix for the polar zonal-diffusion stiffness
    documented on that function; needed for a tractable run at the 512x1024
    headline grid, optional (and numerically inert as a choice, not a
    physics change) at coarser grids where the explicit scheme is already
    affordable.
    """
    k0 = np.asarray(eke_m2_s2, dtype=np.float64)
    zeros = np.zeros_like(k0)
    adv = evolve_column_water(
        k0, zeros, zeros, wind_u_m_s, wind_v_m_s,
        dx_m=dx_m, dy_m=dy_m, dt_days=dt_days,
        cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
        y_face_length_m=y_face_length_m, max_courant=max_courant,
    )
    diffusion_fn = eke_diffusion_step_implicit_zonal if implicit_zonal_diffusion else eke_diffusion_step
    diff = diffusion_fn(
        adv.water_mm, dx_m=dx_m, dy_m=dy_m, dt_days=dt_days,
        cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
        y_face_length_m=y_face_length_m, diffusion_r_limit=diffusion_r_limit,
        eke_floor=eke_floor, a5=a5,
    )
    residual = float(adv.residual_mm) + float(diff.residual_m2_s2)
    area = np.broadcast_to(np.asarray(cell_area_m2, dtype=np.float64), k0.shape)
    denom = max(abs(float(np.sum(diff.eke_m2_s2 * area))), abs(float(np.sum(k0 * area))), 1.0)
    return EkeTransportStep(
        diff.eke_m2_s2, residual, residual / denom,
        int(adv.substeps), int(diff.substeps), float(diff.maximum_diffusion_number),
    )


def _eke_reaction_step(
    eke_m2_s2: np.ndarray,
    production_m2_s3: np.ndarray,
    drag_coefficient: np.ndarray,
    dt_days: float,
    *,
    eke_floor: float,
    a5: dict[str, float] | None,
    r_limit: float = 0.4,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Local ``dK/dt = PK - DK(K)`` reaction, explicit-Euler sub-stepped.

    ``DK = (c3syn + c4syn·CD)·K^1.5`` (A55) is a monotonically increasing,
    convex sink with no other state coupling (unlike diffusion's AT, this
    cell's K does not depend on its neighbours): the ODE relaxes toward
    ``K_eq = (PK/coeff)^(2/3)`` without overshoot, and since the sink only
    gets *stronger* as K grows, its steepest slope over the whole trajectory
    is at ``max(K0, K_eq)`` -- unlike the diffusion case above, bounding the
    sub-step size from that one upfront value is not circular here, because
    there is no cross-cell feedback that could push K past it mid-step.
    Linearising the sink at that bound (``d(DK)/dK = 1.5·coeff·√K``) gives the
    same ``rate·dt_sub ≤ r_limit`` stability bookkeeping used by the transport
    sub-steps above.  Returns ``(K_final, substeps, time-mean DK applied)``.
    """
    params = a5 or _a5_defaults()
    coeff = eke_dissipation_coefficient(drag_coefficient, a5=params)
    k = np.clip(np.asarray(eke_m2_s2, dtype=np.float64), 0.0, None)
    prod = np.asarray(production_m2_s3, dtype=np.float64)
    dt_seconds = dt_days * 86400.0
    k_bound = np.maximum(k, eke_steady_state(prod, drag_coefficient, eke_floor=eke_floor, a5=params))
    local_rate = 1.5 * coeff * np.sqrt(np.maximum(k_bound, eke_floor))
    n_sub = max(1, int(np.ceil(float(np.max(local_rate)) * dt_seconds / r_limit)))
    dt_sub = dt_seconds / n_sub
    dissipation_accum = np.zeros_like(k)
    for _ in range(n_sub):
        dk = eke_dissipation(k, drag_coefficient, eke_floor=eke_floor, a5=params)
        dissipation_accum = dissipation_accum + dk * dt_sub
        k = np.maximum(k + dt_sub * (prod - dk), 0.0)
    return k, n_sub, dissipation_accum / dt_seconds


class EkeStep(NamedTuple):
    eke_m2_s2: np.ndarray
    production_m2_s3: np.ndarray
    dissipation_m2_s3: np.ndarray
    transport_residual_m2_s2: float
    advection_substeps: int
    diffusion_substeps: int
    reaction_substeps: int


def evolve_eke(
    eke_m2_s2: np.ndarray,
    production_m2_s3: np.ndarray,
    drag_coefficient: np.ndarray,
    wind_u_m_s: np.ndarray,
    wind_v_m_s: np.ndarray,
    *,
    dx_m: np.ndarray | float,
    dy_m: float,
    dt_days: float,
    cell_area_m2: np.ndarray | float,
    x_face_length_m: np.ndarray | float,
    y_face_length_m: np.ndarray,
    max_courant: float = 0.5,
    diffusion_r_limit: float = 0.4,
    reaction_r_limit: float = 0.4,
    eke_floor: float = _EKE_FLOOR_M2_S2,
    a5: dict[str, float] | None = None,
    implicit_zonal_diffusion: bool = False,
) -> EkeStep:
    """Full (A52) prognostic step: ``dK/dt = -div(uK) + div(AT·grad K) + PK - DK``.

    Operator-split (module docstring note 6): transport (`eke_transport_step`
    -- advection then diffusion, each conservative and CFL-substepped) then
    the local production/dissipation reaction (`_eke_reaction_step`).
    ``production_m2_s3`` is the externally supplied (A53) production (e.g.
    from `eke_production`); ``drag_coefficient`` drives the (A55) dissipation
    of the evolving K. ``wind_u_m_s``/``wind_v_m_s`` should be the stage-P2
    *zonal-only* wind (note 7). The returned K is floored at ``eke_floor``
    (the reference implementation's own state floor, module docstring note 3)
    -- `eke_transport_step` itself has no such floor, only non-negativity, so
    a pure-transport conservation check should call it directly rather than
    `evolve_eke`.

    ``implicit_zonal_diffusion`` passes through to `eke_transport_step` --
    see `eke_diffusion_step_implicit_zonal` for what it changes and why.
    """
    transport = eke_transport_step(
        eke_m2_s2, wind_u_m_s, wind_v_m_s,
        dx_m=dx_m, dy_m=dy_m, dt_days=dt_days,
        cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
        y_face_length_m=y_face_length_m, max_courant=max_courant,
        diffusion_r_limit=diffusion_r_limit, eke_floor=eke_floor, a5=a5,
        implicit_zonal_diffusion=implicit_zonal_diffusion,
    )
    k_final, n_reaction, dissipation_mean = _eke_reaction_step(
        transport.eke_m2_s2, production_m2_s3, drag_coefficient, dt_days,
        eke_floor=eke_floor, a5=a5, r_limit=reaction_r_limit,
    )
    return EkeStep(
        np.maximum(k_final, eke_floor),
        np.asarray(production_m2_s3, dtype=np.float64),
        dissipation_mean,
        transport.residual_m2_s2,
        transport.advection_substeps,
        transport.diffusion_substeps,
        n_reaction,
    )


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
    production/dissipation balance only -- no advection/diffusion transport.
    (A52) transport is implemented separately (`eke_diffusion_step`,
    `eke_transport_step`, `evolve_eke`, below); this function is unchanged by
    that addition and remains a useful diagnostic of the local closure alone.
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
    "EkeDiffusionStep",
    "EkeStep",
    "EkeTransportStep",
    "SesamSynoptic",
    "brunt_vaisala_frequency",
    "compute_synoptic",
    "eady_growth_rate",
    "eke_diffusion_step",
    "eke_diffusion_step_implicit_zonal",
    "eke_dissipation",
    "eke_dissipation_coefficient",
    "eke_production",
    "eke_relaxation_tendency",
    "eke_steady_state",
    "eke_transport_step",
    "evolve_eke",
    "horizontal_diffusion_coefficient",
    "moisture_diffusion_coefficient",
    "scalar_at_pressure",
    "spherical_transport_geometry",
    "synoptic_surface_wind",
    "synoptic_vertical_velocity",
    "total_wind_magnitude",
    "wind_stress",
    "zonal_center_spacing_m",
]
