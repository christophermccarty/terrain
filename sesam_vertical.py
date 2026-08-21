"""SESAM diagnostic vertical structure — Appendix A1 of Willeit et al. (2022).

Pure, read-only kernels for the CLIMBER-X/SESAM 2.5-D atmosphere's universal
vertical profiles, transcribed from Appendix A1 of

    Willeit, M., Ganopolski, A., Robinson, A., and Edwards, N. R.:
    GMD 15, 5905–5948 (2022), https://doi.org/10.5194/gmd-15-5905-2022
    (CC-BY 4.0; equations cited below as (A#) against that paper).

The supported PlanetSim pipeline never calls these functions (the
``PlanetParams.enable_sesam_vertical_structure`` gate is off by default), so
this module has zero default-path climate impact by construction.  It exists
so SESAM stage P1 (docs/SESAM_GAP_ANALYSIS.md §7) is evaluable as a pure
kernel, and so the A2/A5/P5 stages that need temperature, humidity and
tropopause structure (Eady growth rate, thermal wind, LW radiation levels)
have a single proven source.

**Documented scope decisions** (kept faithful to the paper, flagged where a
later stage supplies a currently-missing input):

1. Coherent per-column inputs are 2-D fields ``(H, W)`` and heights are
   **absolute** metres above sea level, as in the paper (surface is at
   ``surface_elevation_m``).  ``T(z)`` follows (A5) exactly piecewise.
2. The tropopause ``HT`` is a **required input** to the profile builders.
   (A10) makes it a time-dependent function of the stratospheric radiative
   residual ``Rstr,net``, which needs the A8 longwave/ozone radiation not
   scheduled until stage P5.  P1 supplies the (A10) tendency and the (A11)
   dynamical shape ``S`` as separate kernels; closing ``HT`` from radiation is
   P5's job.
3. The RH scale height (A14) needs ``w700`` (EKE stage P3) and the
   meridional-cell coordinate ``φ`` from (A31) (dynamics stage P2).  Both are
   accepted as optional inputs; callers without them pass zeros / latitude as
   documented placeholders.  Nothing here silently fabricates them.
4. ``p0`` (mean surface pressure) and the scale-height reference temperature
   ``T0`` are explicit inputs — never hardcoded Earth literals — so the module
   is planet-general.  ``surface_kind`` is encoded 0=ocean, 1=land, 2=ice per
   the (A9) near-surface lapse-rate branches.
5. qsat follows the (A15) ice/water partition (saturation over ice below
   −15 °C, over water above 0 °C, linearly weighted between).  The Magnus
   coefficients match the project's ``land_surface.py`` water term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sesam_reference import value as _param

# ---------------------------------------------------------------------------
# A1 parameter registry (single source: sesam_reference.py)
# ---------------------------------------------------------------------------


def _a1_defaults() -> dict[str, float]:
    """All A1 constants from the published Table A1 pack."""
    return {
        "c1_Gamma": _param("A1_vertical_structure", "c1_Gamma"),
        "c2_Gamma": _param("A1_vertical_structure", "c2_Gamma"),
        "c3_Gamma": _param("A1_vertical_structure", "c3_Gamma"),
        "c4_Gamma": _param("A1_vertical_structure", "c4_Gamma"),
        "c5_Gamma": _param("A1_vertical_structure", "c5_Gamma"),
        "c6_Gamma": _param("A1_vertical_structure", "c6_Gamma"),
        "H_Gamma_s": _param("A1_vertical_structure", "H_Gamma_s"),
        "H_Gamma_t": _param("A1_vertical_structure", "H_Gamma_t"),
        "c1r": _param("A1_vertical_structure", "c1r"),
        "c2r": _param("A1_vertical_structure", "c2r"),
        "c3r": _param("A1_vertical_structure", "c3r"),
        "c4r": _param("A1_vertical_structure", "c4r"),
        "c5r": _param("A1_vertical_structure", "c5r"),
        "r_st": _param("A1_vertical_structure", "r_st"),
        "c1tp": _param("A1_vertical_structure", "c1tp"),
        "c2tp": _param("A1_vertical_structure", "c2tp"),
        "c3tp": _param("A1_vertical_structure", "c3tp"),
    }


# Published (A9) caps on the near-surface lapse rate, in K m^-1.
_GAMMA_S_CAP_OCEAN = 7.5e-3
_GAMMA_S_CAP_LAND_ICE = 10.0e-3

# Numerics guard on the *integrated* (A5) temperature profile T(z), not on
# (A9) itself -- see docs/SESAM_GAP_ANALYSIS.md Sec7 P6d/P6e, 2026-08-20
# follow-up. (A9)'s cold-land inversion term is deliberately unbounded per
# the published paper (verified against the source text directly, not a
# transcription gap), so it stays untouched here. But under live coupling,
# a live Ta can transiently drift far enough from T* that integrating that
# unbounded slope over even the ~1.5 km near-surface layer produces a T(z)
# point in the hundreds of Kelvin -- not a paper edge case, a downstream
# numerics failure: `sesam_longwave.longwave_radiation`'s transmission
# matrix and `saturation_specific_humidity`'s Clausius-Clapeyron form are
# not designed to accept such values and produce physically meaningless
# (and further destabilizing) output when they do. This clip bounds the
# *result* T(z) is compared/fed downstream with, the same [150, 350] K
# sanity range `sesam_coupling.py` already clips prognostic Ta to
# (`ta_next = np.clip(..., 150.0, 350.0)`) -- an existing project
# convention, not a new one. Investigated at length before adding this
# (docs/SESAM_GAP_ANALYSIS.md Sec7 P6d/P6e 2026-08-20 entries): the
# wind/dryness causal chain was falsified, finer time-sub-stepping was
# tested and found insufficient even with the real T*-Ta feedback active,
# and CLIMBER-X's own reference architecture was read directly and found to
# match this project's constants and formulas -- the underlying Ta-T* drift
# is a genuine, currently-unresolved coupling gap, not a bug elsewhere; this
# clip only keeps the radiative-transfer numerics from amplifying it into a
# harder crash while that gap stays open.
_PROFILE_CLIP_MIN_K = 150.0
_PROFILE_CLIP_MAX_K = 350.0

# Magnus saturation-curve constants (specific-heat-consistent water term is
# identical to land_surface.py; the ice term is the standard over-ice curve).
_DEFAULT_RD = 287.0
_DEFAULT_CP = 1005.0
_EPS = 0.622  # R_dry / R_vap

_T_ICE_C = -15.0
_T_0_C = 0.0


# ---------------------------------------------------------------------------
# Small algebra helpers
# ---------------------------------------------------------------------------


def _as_2d(name: str, value, h: int, w: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    expected = (h, w)
    if arr.shape == expected:
        return arr
    if arr.ndim == 0:
        return np.full(expected, float(arr))
    raise ValueError(f"{name} must have shape {expected}, got {arr.shape}")


def _level_grid(levels_m: np.ndarray, h: int, w: int) -> np.ndarray:
    """Broadcast the 1-D level axis to an ``(N, H, W)`` absolute-height grid."""
    levels = np.asarray(levels_m, dtype=np.float64)
    if levels.ndim != 1:
        raise ValueError("levels_m must be 1-D")
    return np.broadcast_to(levels[:, None, None], (levels.size, h, w))


def _surface_kind_2d(surface_kind, h: int, w: int) -> np.ndarray:
    arr = np.asarray(surface_kind)
    if arr.shape != (h, w):
        raise ValueError(
            f"surface_kind must have shape {(h, w)}, got {arr.shape}"
        )
    if np.any(arr < 0) or np.any(arr > 2):
        raise ValueError("surface_kind codes must be 0 (ocean), 1 (land), 2 (ice)")
    return arr.astype(np.int64)


def height_scale(
    reference_temp_k: float,
    *,
    gravity: float,
    gas_constant: float = _DEFAULT_RD,
) -> float:
    """(A1) pressure/density scale height ``Ha = Rd·T0/g``, in metres."""
    return gas_constant * reference_temp_k / gravity


def dry_adiabatic_lapse(
    *, gravity: float, specific_heat: float = _DEFAULT_CP
) -> float:
    """(A12) ``Γd = g/cp`` in K m^-1."""
    return gravity / specific_heat


def pressure_profile(
    levels_m: np.ndarray,
    p0_pa: np.ndarray | float,
    ha_m: float,
) -> np.ndarray:
    """(A1) ``p(z) = p0·exp(−z/Ha)`` on the level grid.

    A scalar ``p0_pa`` yields shape ``(N, 1, 1)`` (broadcastable against the
    ``(N, H, W)`` profile arrays); a 2-D ``p0_pa`` of shape ``(H, W)`` yields
    ``(N, H, W)``.  Valid for all heights (including below the surface),
    matching the paper's exponential reference atmosphere.
    """
    levels = np.asarray(levels_m, dtype=np.float64)
    if levels.ndim != 1:
        raise ValueError("levels_m must be a 1-D array of heights in metres")
    p0 = np.asarray(p0_pa, dtype=np.float64)
    exponent = np.exp(-levels / ha_m)
    if p0.ndim == 0:
        return (p0 * exponent)[:, None, None]
    if p0.ndim == 2:
        return exponent[:, None, None] * p0[None, :, :]
    raise ValueError("p0_pa must be a scalar or a 2-D (H, W) field")


def air_density_profile(
    pressure_pa: np.ndarray,
    reference_temp_k: float,
    gas_constant: float = _DEFAULT_RD,
) -> np.ndarray:
    """(A3)/(A4) air density on the level grid: ``rho(z) = rho0*exp(-z/Ha)``,
    ``rho0 = p0/(Rd*T0)``.  Since (A1)'s pressure profile shares the exact
    same exponential (``p(z)/p0 = exp(-z/Ha) = rho(z)/rho0``), this reduces
    to ``rho(z) = p(z)/(Rd*T0)`` exactly -- not a separate approximation,
    just (A1)+(A3)+(A4) combined algebraically.  Added for stage P5 (the LW
    absorber-mass integrals need a density profile that P1's original A1
    kernels never had to materialise on their own).
    """
    p = np.asarray(pressure_pa, dtype=np.float64)
    return p / (gas_constant * reference_temp_k)


# ---------------------------------------------------------------------------
# (A15) saturation specific humidity with ice/water partition
# ---------------------------------------------------------------------------


def _es_magnus(tc_c: np.ndarray, a: float, b: float) -> np.ndarray:
    """Saturation vapour pressure (Pa) over water (17.67/243.5) or ice."""
    return 611.2 * np.exp(a * tc_c / (tc_c + b))


def saturation_specific_humidity(
    temperature_k: np.ndarray,
    pressure_pa: np.ndarray,
    *,
    t_ice_c: float = _T_ICE_C,
    t_0_c: float = _T_0_C,
) -> np.ndarray:
    """(A15) ``qsat(T, p)`` with the paper's ice/water partition.

    Below ``t_ice_c`` (−15 °C) saturation is over ice; above ``t_0_c`` (0 °C)
    it is over water; between them the two values are linearly weighted by
    temperature.  ``qsat = ε·e/(p − (1−ε)·e)`` with the ground truth values
    from the Magnus curves.
    """
    t = np.asarray(temperature_k, dtype=np.float64)
    p = np.asarray(pressure_pa, dtype=np.float64)
    if t.shape != p.shape:
        raise ValueError("temperature_k and pressure_pa must share a shape")
    tc = t - 273.15
    es_w = _es_magnus(tc, 17.67, 243.5)
    es_i = _es_magnus(tc, 22.46, 272.62)
    t_ice = t_ice_c + 273.15
    t_0 = t_0_c + 273.15

    q_w = _EPS * es_w / np.maximum(p - (1.0 - _EPS) * es_w, 1.0)
    q_i = _EPS * es_i / np.maximum(p - (1.0 - _EPS) * es_i, 1.0)
    weight = np.clip((t - t_ice) / max(t_0 - t_ice, 1e-6), 0.0, 1.0)
    qsat = weight * q_w + (1.0 - weight) * q_i
    return np.maximum(qsat, 0.0)


# ---------------------------------------------------------------------------
# (A7)-(A9) lapse rate
# ---------------------------------------------------------------------------


def free_troposphere_lapse(
    near_surface_rh_kgkg: np.ndarray,
    a1: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(A7)-(A8): free-troposphere base/ramp lapse rates ``Γb``, ``Γt``.

    ``Γb = c1Γ − c2Γ·qa`` and ``Γt = Γb − c2Γ·qa + c3Γ``.  Both vary only with
    near-surface humidity ``qa``.
    """
    params = a1 or _a1_defaults()
    qa = np.asarray(near_surface_rh_kgkg, dtype=np.float64)
    c1, c2, c3 = params["c1_Gamma"], params["c2_Gamma"], params["c3_Gamma"]
    gamma_b = c1 - c2 * qa
    gamma_t = gamma_b - c2 * qa + c3
    return gamma_b, gamma_t


def near_surface_lapse(
    near_surface_air_temp_k: np.ndarray,
    skin_temp_k: np.ndarray,
    surface_kind: np.ndarray,
    a1: dict[str, float] | None = None,
) -> np.ndarray:
    """(A9) stability-dependent near-surface lapse rate ``Γs`` (K m^-1).

    Branches: ocean ``c4Γ·max(0, Ta−T⋆)``; warm land ``c5Γ·(Ta−T⋆)``; cold
    (inverted) land ``c6Γ·(Ta−T⋆)``; ice ``c5Γ·(Ta−T⋆)``.  Upper-bounded by
    7.5e-3 (ocean) / 10e-3 (land, ice) K m^-1; negative values (inversions)
    are allowed.
    """
    params = a1 or _a1_defaults()
    ta = np.asarray(near_surface_air_temp_k, dtype=np.float64)
    ts = np.asarray(skin_temp_k, dtype=np.float64)
    kind = _surface_kind_2d(surface_kind, *ta.shape)
    if not (ta.shape == ts.shape):
        raise ValueError("near_surface_air_temp_k and skin_temp_k must share a shape")

    c4, c5, c6 = params["c4_Gamma"], params["c5_Gamma"], params["c6_Gamma"]
    delta = ta - ts
    ocean = kind == 0
    ice = kind == 2
    warm_land = (kind == 1) & (delta > 0.0)
    cold_land = (kind == 1) & (delta <= 0.0)

    gamma_s = np.zeros_like(delta)
    gamma_s = np.where(ocean, c4 * np.maximum(delta, 0.0), gamma_s)
    gamma_s = np.where(ice, c5 * delta, gamma_s)
    gamma_s = np.where(warm_land, c5 * delta, gamma_s)
    gamma_s = np.where(cold_land, c6 * delta, gamma_s)

    cap = np.where(ocean, _GAMMA_S_CAP_OCEAN, _GAMMA_S_CAP_LAND_ICE)
    return np.minimum(gamma_s, cap)


def full_lapse_rate(
    z_abs: np.ndarray,
    near_surface_air_temp_k: np.ndarray,
    skin_temp_k: np.ndarray,
    surface_kind: np.ndarray,
    near_surface_rh_kgkg: np.ndarray,
    surface_elevation_m: np.ndarray,
    tropopause_height_m: np.ndarray,
    a1: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(A6) piecewise lapse rate ``Γ(z)`` over an ``(N, H, W)`` level grid.

    ``z_abs`` is the absolute-height level grid (from ``np.broadcast_to`` of
    ``levels_m``).  Returns ``(gamma_z, gamma_s, gamma_b, gamma_t)`` matching
    the (A9)/(A7)/(A8) fields used to assemble the temperature profile.
    """
    params = a1 or _a1_defaults()
    z = np.asarray(z_abs, dtype=np.float64)
    if z.ndim != 3:
        raise ValueError("z_abs must be a 3-D (N, H, W) absolute-height grid")
    hs = params["H_Gamma_s"]
    ht_ramp = params["H_Gamma_t"]
    zs = np.asarray(surface_elevation_m, dtype=np.float64)
    ht = np.asarray(tropopause_height_m, dtype=np.float64)
    if zs.shape != ht.shape:
        raise ValueError("surface_elevation_m and tropopause_height_m must share a shape")
    if z.shape[1:] != zs.shape:
        raise ValueError("z_abs trailing axes must match the 2-D surface fields")

    gamma_s = near_surface_lapse(
        near_surface_air_temp_k, skin_temp_k, surface_kind, a1=params
    )
    gamma_b, gamma_t = free_troposphere_lapse(near_surface_rh_kgkg, a1=params)

    # Guard degenerate columns (tropopause at/below surface height).
    hs_eff = np.clip(hs, 0.0, np.maximum(ht - zs, 0.0))  # (H, W)
    b1 = zs + hs_eff                                # near-surface top
    b2 = ht                                        # tropopause
    zz = np.maximum(z, zs[None, :, :])

    in_surface = zz <= b1[None, :, :]
    in_ramp = (zz > b1[None, :, :]) & (zz <= b2[None, :, :])
    rampslope = gamma_b + (gamma_t - gamma_b) * (zz / ht_ramp)
    gamma_z = np.where(
        in_surface,
        gamma_s[None, :, :],
        np.where(in_ramp, rampslope, np.zeros_like(zz)),
    )
    return gamma_z, gamma_s, gamma_b, gamma_t


# ---------------------------------------------------------------------------
# (A5) temperature profile
# ---------------------------------------------------------------------------


def temperature_profile(
    levels_m: np.ndarray,
    near_surface_air_temp_k: np.ndarray,
    skin_temp_k: np.ndarray,
    surface_kind: np.ndarray,
    near_surface_rh_kgkg: np.ndarray,
    surface_elevation_m: np.ndarray,
    tropopause_height_m: np.ndarray,
    a1: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(A5) ``T(z) = Ta − ∫Γ dz`` on the level grid.

    The integral is analytic and piecewise for the (A6) lapse-rate form, so
    the result does not depend on level spacing.  Levels at or below the
    surface evaluate to ``Ta``.  Returns ``(T_z, gamma_z, T_surface)``.
    """
    params = a1 or _a1_defaults()
    levels = np.asarray(levels_m, dtype=np.float64)
    if levels.ndim != 1:
        raise ValueError("levels_m must be 1-D")
    ta = np.asarray(near_surface_air_temp_k, dtype=np.float64)
    if ta.ndim != 2:
        raise ValueError("near_surface_air_temp_k must be 2-D (H, W)")
    h, w = ta.shape
    zs = _as_2d("surface_elevation_m", surface_elevation_m, h, w)
    ht = _as_2d("tropopause_height_m", tropopause_height_m, h, w)
    hs = params["H_Gamma_s"]
    ht_ramp = params["H_Gamma_t"]

    zgrid = _level_grid(levels, h, w)          # (N, H, W)
    zz = np.maximum(zgrid, zs[None, :, :])
    zz_q = np.ascontiguousarray(zz)

    hs_eff = np.clip(hs, 0.0, np.maximum(ht - zs, 0.0))
    b1 = zs + hs_eff
    b2 = ht

    gamma_s = near_surface_lapse(ta, skin_temp_k, surface_kind, a1=params)
    gamma_b, gamma_t = free_troposphere_lapse(near_surface_rh_kgkg, a1=params)
    slope = (gamma_t - gamma_b) / ht_ramp

    # Temperature drop integration for each zone (exact antiderivatives).
    drop_surface = gamma_s * (zz_q - zs[None, :, :])                     # z <= b1
    drop_ramp = (
        gamma_s * hs_eff[None, :, :]
        + gamma_b * (zz_q - b1[None, :, :])
        + 0.5 * slope * (zz_q ** 2 - b1[None, :, :] ** 2)
    )                                                                     # b1 < z <= b2
    drop_above = (
        gamma_s * hs_eff[None, :, :]
        + gamma_b * (b2[None, :, :] - b1[None, :, :])
        + 0.5 * slope * (b2[None, :, :] ** 2 - b1[None, :, :] ** 2)
    )                                                                     # z > b2 (isothermal)

    drop = np.where(
        zz_q <= b1[None, :, :],
        drop_surface,
        np.where(zz_q <= b2[None, :, :], drop_ramp, drop_above),
    )
    t_z = ta[None, :, :] - drop

    gamma_z, _, _, _ = full_lapse_rate(
        zgrid,
        ta,
        skin_temp_k,
        surface_kind,
        near_surface_rh_kgkg,
        zs,
        ht,
        a1=params,
    )
    return t_z, gamma_z, ta


def potential_temperature_profile(
    temperature_z: np.ndarray,
    levels_m: np.ndarray,
    *,
    gravity: float,
    specific_heat: float = _DEFAULT_CP,
) -> np.ndarray:
    """(A12) ``θ(z) = T(z) + Γd·z`` with ``Γd = g/cp``.

    Note: SESAM uses this linear definition (dry-adiabatic reference added at
    absolute height), not the Poisson ``θ = T(p0/p)^(Rd/cp)`` form.
    """
    gamma_d = dry_adiabatic_lapse(gravity=gravity, specific_heat=specific_heat)
    z3 = _level_grid(levels_m, *temperature_z.shape[1:])
    return temperature_z + gamma_d * z3


# ---------------------------------------------------------------------------
# (A13)-(A14) relative humidity profile
# ---------------------------------------------------------------------------


def rh_scale_height(
    ftrop: np.ndarray,
    w700_m_s: np.ndarray | float,
    a1: dict[str, float] | None = None,
) -> np.ndarray:
    """(A14) ``Hr = ftrop·c1r·exp(c2r·w700) + (1−ftrop)·c1r·c3r``.

    ``ftrop = 1 − sin⁸φ`` (see :func:`tropical_weight`); ``w700`` is the
    700 hPa vertical velocity in m s^-1 (EKE stage P3 supplies it; a caller
    without it passes zeros, giving the extratropical-limited form).
    """
    params = a1 or _a1_defaults()
    f = np.asarray(ftrop, dtype=np.float64)
    w = np.asarray(w700_m_s, dtype=np.float64)
    c1r, c2r, c3r = params["c1r"], params["c2r"], params["c3r"]
    if w.ndim == 0:
        w = np.broadcast_to(w, f.shape)
    if w.shape != f.shape:
        raise ValueError("w700_m_s and ftrop must share a shape")
    return f * c1r * np.exp(c2r * w) + (1.0 - f) * c1r * c3r


def tropical_weight(phi_mmc_rad: np.ndarray) -> np.ndarray:
    """(A14 note) ``ftrop = 1 − sin⁸φ``.

    ``φ`` is the meridional-cell coordinate (A31), supplied by dynamics stage
    P2.  Until then callers may pass latitude radians as a documented
    placeholder (see module docstring, item 3).
    """
    phi = np.asarray(phi_mmc_rad, dtype=np.float64)
    return 1.0 - np.sin(phi) ** 8


def relative_humidity_profile(
    z_abs: np.ndarray,
    near_surface_rh: np.ndarray,
    surface_elevation_m: np.ndarray,
    tropopause_height_m: np.ndarray,
    rh_scale_height_m: np.ndarray,
    a1: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(A13) relative humidity profile on the ``(N, H, W)`` grid.

    ``r(z) = ra`` inside the PBL (``zpbl = zs + c5r``), exponential decay
    ``ra·exp(−(z−zpbl)/Hr)`` up to ``zs + c4r``, constant at the decayed value
    above that up to the tropopause, and ``r_st`` in the stratosphere.
    ``near_surface_rh`` is the near-surface relative humidity ``ra``.
    Returns ``(r_z, ra)`` with ``ra`` the broadcast surface value array.
    """
    params = a1 or _a1_defaults()
    z = np.asarray(z_abs, dtype=np.float64)
    if z.ndim != 3:
        raise ValueError("z_abs must be a 3-D (N, H, W) absolute-height grid")
    zs = np.asarray(surface_elevation_m, dtype=np.float64)
    ht = np.asarray(tropopause_height_m, dtype=np.float64)
    hr = np.asarray(rh_scale_height_m, dtype=np.float64)
    if not (zs.shape == ht.shape == hr.shape):
        raise ValueError("2-D inputs must share shape (H, W)")
    ra = _as_2d("near_surface_rh", near_surface_rh, *z.shape[1:])
    if z.shape[1:] != zs.shape:
        raise ValueError("z_abs trailing axes must match the 2-D surface fields")
    c4r, c5r, rst = params["c4r"], params["c5r"], params["r_st"]

    zpbl = zs + c5r
    z_c4 = zs + c4r
    zz = np.maximum(z, zs[None, :, :])

    in_pbl = zz <= zpbl[None, :, :]
    in_exp = (zz > zpbl[None, :, :]) & (zz <= z_c4[None, :, :])
    in_low_trop = (zz > z_c4[None, :, :]) & (zz <= ht[None, :, :])
    in_strat = zz > ht[None, :, :]

    decay_pbl_top = np.exp(-(z_c4 - zpbl) / np.maximum(hr, 1e-6))
    r_z = np.where(
        in_pbl,
        ra[None, :, :],
        np.where(
            in_exp,
            ra[None, :, :] * np.exp(-(zz - zpbl[None, :, :]) / np.maximum(hr[None, :, :], 1e-6)),
            np.where(in_low_trop, ra[None, :, :] * decay_pbl_top[None, :, :], rst),
        ),
    )
    return r_z, ra


# ---------------------------------------------------------------------------
# (A10)-(A11) tropopause rate and dynamical shape
# ---------------------------------------------------------------------------


def tropopause_shape_s(
    latitude_rad: np.ndarray,
    itcz_latitude_rad: np.ndarray | float,
    hadley_width_rad: np.ndarray | float,
    a1: dict[str, float] | None = None,
) -> np.ndarray:
    """(A11) dynamical contribution ``S`` to the tropopause tendency.

    ``S = c2tp·(1 − c3tp·(1 − sin⁸[0.85·(φ − φITCZ)/(0.5·ΔφHad)]))``.
    Peaks near the ITCZ and deepens (larger S) within the Hadley cell.
    """
    params = a1 or _a1_defaults()
    phi = np.asarray(latitude_rad, dtype=np.float64)
    phi_it = np.asarray(itcz_latitude_rad, dtype=np.float64)
    dhad = np.asarray(hadley_width_rad, dtype=np.float64)
    if phi_it.ndim == 0:
        phi_it = np.broadcast_to(phi_it, phi.shape)
    if dhad.ndim == 0:
        dhad = np.broadcast_to(dhad, phi.shape)
    if not (phi.shape == phi_it.shape == dhad.shape):
        raise ValueError("latitude/itcz/hadley inputs must share a shape")
    x = 0.85 * (phi - phi_it) / np.maximum(0.5 * dhad, 1e-6)
    sin8 = np.sin(x) ** 8
    c2tp, c3tp = params["c2tp"], params["c3tp"]
    return c2tp * (1.0 - c3tp * (1.0 - sin8))


def tropopause_tendency(
    r_strat_net_w_m2: np.ndarray,
    tropopause_shape_s: np.ndarray | float,
    a1: dict[str, float] | None = None,
) -> np.ndarray:
    """(A10) tropopause-height tendency ``−c1tp·(Rstr,net + S)``.

    ``Rstr,net`` (stratospheric longwave balance + shortwave absorbed by ozone)
    is supplied by the radiation stage (P5), not yet computed here.  The unit
    of ``c1tp`` (100 m³ W⁻¹) implies a per-timestep rate in the source model;
    see the flag in ``sesam_reference`` — do not assume the returned values are
    metres per second until the P5 closure fixes the folding.  This function is
    provided so the (A10) form is present and testable before P5.
    """
    params = a1 or _a1_defaults()
    r = np.asarray(r_strat_net_w_m2, dtype=np.float64)
    s = np.asarray(tropopause_shape_s, dtype=np.float64)
    if s.ndim == 0:
        s = np.broadcast_to(s, r.shape)
    if r.shape != s.shape:
        raise ValueError("r_strat_net_w_m2 and tropopause_shape_s must share a shape")
    return -params["c1tp"] * (r + s)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerticalStructure:
    """Evaluated A1 profiles on a fixed level grid, all shape ``(N, H, W)``.

    ``tropopause_shape_s`` is ``(H, W)`` when it could be computed, else None;
    ``tropopause_tendency`` likewise uses the caller's stratospheric residual.
    """

    levels_m: np.ndarray
    temperature_k: np.ndarray
    lapse_rate_k_per_m: np.ndarray
    relative_humidity: np.ndarray
    specific_humidity_kgkg: np.ndarray
    potential_temperature_k: np.ndarray
    pressure_pa: np.ndarray
    air_density_kg_m3: np.ndarray
    near_surface_rh: np.ndarray
    tropopause_tendency: np.ndarray | None = None
    tropopause_shape_s: np.ndarray | None = None


def compute_vertical_structure(
    levels_m: np.ndarray,
    *,
    near_surface_air_temp_k: np.ndarray,
    skin_temp_k: np.ndarray,
    surface_kind: np.ndarray,
    near_surface_specific_humidity_kgkg: np.ndarray,
    surface_elevation_m: np.ndarray,
    tropopause_height_m: np.ndarray,
    p0_pa: np.ndarray | float,
    gravity: float,
    reference_temp_k: float,
    gas_constant: float = _DEFAULT_RD,
    specific_heat: float = _DEFAULT_CP,
    itcz_latitude_rad: np.ndarray | float | None = None,
    hadley_width_rad: np.ndarray | float | None = None,
    tropical_weight_field: np.ndarray | None = None,
    w700_m_s: np.ndarray | float = 0.0,
    r_strat_net_w_m2: np.ndarray | None = None,
    a1: dict[str, float] | None = None,
) -> VerticalStructure:
    """Assemble the full A1 profile set on ``levels_m``.

    Arguments mirror the paper's A1 inputs; all 2-D fields share shape
    ``(H, W)``.  ``tropopause_height_m`` is a required input until the P5
    radiation stage closes it (module docstring item 2).  If
    ``itcz_latitude_rad`` and ``hadley_width_rad`` are given, the (A11) shape
    is computed and (if ``r_strat_net_w_m2`` is also given) the (A10) tendency.

    ``tropical_weight_field`` overrides the default ``ftrop = 1−sin⁸φ``,
    providing the P2 (A31) coordinate when available (placeholder: latitude).
    """
    params = a1 or _a1_defaults()
    levels = np.asarray(levels_m, dtype=np.float64)
    if levels.ndim != 1:
        raise ValueError("levels_m must be 1-D")

    ta = np.asarray(near_surface_air_temp_k, dtype=np.float64)
    if ta.ndim != 2:
        raise ValueError("near_surface_air_temp_k must be 2-D (H, W)")
    h, w = ta.shape
    _as_2d("skin_temp_k", skin_temp_k, h, w)
    _surface_kind_2d(surface_kind, h, w)
    qa = _as_2d("near_surface_specific_humidity_kgkg",
                near_surface_specific_humidity_kgkg, h, w)
    zs = _as_2d("surface_elevation_m", surface_elevation_m, h, w)
    ht = _as_2d("tropopause_height_m", tropopause_height_m, h, w)

    ha = height_scale(reference_temp_k, gravity=gravity, gas_constant=gas_constant)
    p_z = pressure_profile(levels, p0_pa, ha)
    p_z = np.broadcast_to(p_z, (levels.size, h, w))  # (N, H, W) for q_z

    # Surface pressure for ra = qa / qsat(Ta, ps) — (A1) at z = zs.
    ps = p0_pa * np.exp(-zs / ha)
    qsat_ta = saturation_specific_humidity(ta, ps)
    ra = np.clip(qa / np.maximum(qsat_ta, 1e-12), 0.0, 1.0)

    t_z, gamma_z, _ = temperature_profile(
        levels, ta, skin_temp_k, surface_kind, qa, zs, ht, a1=params
    )
    # Numerics guard, not a physics change -- see _PROFILE_CLIP_MIN_K/MAX_K's
    # own module-level docstring. Clipped here, before theta_z/q_z below are
    # derived from t_z, so every downstream consumer (both this function's
    # own potential-temperature/humidity outputs and every caller's use of
    # temperature_k) sees a consistent, bounded profile -- not a t_z clipped
    # after the fact while theta_z/q_z still reflect the unbounded value.
    t_z = np.clip(t_z, _PROFILE_CLIP_MIN_K, _PROFILE_CLIP_MAX_K)
    theta_z = potential_temperature_profile(
        t_z, levels, gravity=gravity, specific_heat=specific_heat
    )

    if tropical_weight_field is None:
        latitude = np.arccos(np.clip(
            _row_latitude_cos(h, w),
            0.0, 1.0,
        ))  # placeholder φ ≈ |latitude| (absolute value via cos parity)
        ftrop = tropical_weight(latitude)
    else:
        ftrop = np.asarray(tropical_weight_field, dtype=np.float64)
        if ftrop.shape != (h, w):
            raise ValueError("tropical_weight_field must have shape (H, W)")
    hr = rh_scale_height(ftrop, w700_m_s, a1=params)
    zgrid3 = _level_grid(levels, h, w)
    r_z, ra_out = relative_humidity_profile(
        zgrid3,
        ra,
        zs,
        ht,
        hr,
        a1=params,
    )
    q_z = r_z * saturation_specific_humidity(t_z, p_z)

    trop_s = None
    trop_d = None
    if itcz_latitude_rad is not None and hadley_width_rad is not None:
        latitude_rad = _row_latitude_rad(h, w)
        trop_s = tropopause_shape_s(
            latitude_rad, itcz_latitude_rad, hadley_width_rad, a1=params
        )
        if r_strat_net_w_m2 is not None:
            r_strat = _as_2d(
                "r_strat_net_w_m2", r_strat_net_w_m2, h, w
            )
            trop_d = tropopause_tendency(r_strat, trop_s, a1=params)

    return VerticalStructure(
        levels_m=levels,
        temperature_k=t_z,
        lapse_rate_k_per_m=gamma_z,
        relative_humidity=r_z,
        specific_humidity_kgkg=q_z,
        potential_temperature_k=theta_z,
        pressure_pa=p_z,
        air_density_kg_m3=air_density_profile(p_z, reference_temp_k, gas_constant),
        near_surface_rh=ra_out,
        tropopause_tendency=trop_d,
        tropopause_shape_s=trop_s,
    )


def _row_latitude_cos(h: int, w: int) -> np.ndarray:
    """cos(|latitude|) per row (0 at the poles, 1 at the equator)."""
    lat = (0.5 - (np.arange(h, dtype=np.float64) + 0.5) / h) * np.pi
    return np.repeat(np.cos(np.abs(lat))[:, None], w, axis=1)


def _row_latitude_rad(h: int, w: int) -> np.ndarray:
    """Signed latitude in radians per row, north-positive."""
    lat = (0.5 - (np.arange(h, dtype=np.float64) + 0.5) / h) * np.pi
    return np.repeat(lat[:, None], w, axis=1)


__all__ = [
    "VerticalStructure",
    "air_density_profile",
    "compute_vertical_structure",
    "dry_adiabatic_lapse",
    "free_troposphere_lapse",
    "full_lapse_rate",
    "height_scale",
    "near_surface_lapse",
    "potential_temperature_profile",
    "pressure_profile",
    "relative_humidity_profile",
    "rh_scale_height",
    "saturation_specific_humidity",
    "temperature_profile",
    "tropical_weight",
    "tropopause_shape_s",
    "tropopause_tendency",
]