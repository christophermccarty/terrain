"""SESAM dynamics: 3-D wind assembly — Appendix A2 of Willeit et al. (2022).

Pure, read-only kernels for the CLIMBER-X/SESAM 2.5-D atmosphere's wind
field, transcribed from Appendix A2 of

    Willeit, M., Ganopolski, A., Robinson, A., and Edwards, N. R.:
    GMD 15, 5905–5948 (2022), https://doi.org/10.5194/gmd-15-5905-2022
    (CC-BY 4.0; equations cited below as (A#) against that paper).

This is SESAM stage P2's second sub-deliverable (docs/SESAM_GAP_ANALYSIS.md
§7): the wind assembly (A16)–(A27).  It consumes the SLP field from
``sesam_dynamics.py`` (first sub-deliverable) and the 3-D temperature
structure from ``sesam_vertical.py`` (stage P1), and produces the surface
wind (Taylor model + katabatic), the 3-D geostrophic + ageostrophic wind,
and the 500 hPa zonal wind that closes the Charney–Eliassen input of the SLP
stage.  It replaces what the supported path prescribes (the 3-cell wind
targets in ``atmosphere.py``) with wind *derived* from the model's own SLP
and temperature gradients — diagnostically, behind the same default-off gate
(``PlanetParams.enable_sesam_dynamics``), with zero default-path climate
impact by construction.

**Verification notes** (equation semantics checked 2026-08-16 against the
article's HTML MathML and cross-checked against the GPL-3.0 CLIMBER-X
Fortran, which per docs/SESAM_GAP_ANALYSIS.md §5 may be *read* to
cross-check semantics but not copied or translated; all code here is written
from the CC-BY paper):

1.  **ε = √(1 − sin 2α)** in (A21)/(A24)/(A25): the printed ``sin2α`` is the
    double angle sin(2α), and ε is computed from |α| (the drag closure is
    symmetric; α itself is signed, positive in the NH).  Verified against the
    reference implementation.
2.  **The (A21) solve needs no EKE input.** With the paper's Us ≈ √(2K) and
    the reference implementation's PBL viscosity tied to the EKE (Kv = K),
    the K cancels and the closure reduces to
    ``sinα/√(1 − sin 2α) = CD/√|f|``, solved by bisection on [0, π/4] and
    clamped to α ∈ [0.05, 0.5] rad (reference namelist ``acbar_max``).
3.  **sinα·cosα enters (A19)/(A20) as a positive magnitude** with |f| (the
    ageostrophic PBL flow crosses isobars toward low pressure in both
    hemispheres); α is signed only in the Taylor rotation (A24)–(A25).
    Matches the convention used in ``sesam_dynamics.zonal_slp_anomaly``.
4.  **Coriolis floors** follow the paper text: |f| ≥ 3e-5 s⁻¹ in the
    geostrophic/thermal-wind relations (A17)–(A18), |f| ≥ 1e-5 s⁻¹ in the
    ageostrophic relations (A19)–(A21) (the reference namelist uses 1e-5 for
    both; the paper's two-floor form is implemented).
5.  **Thermal-wind damping** (reference-implementation safeguard, adopted and
    documented, not in the printed equations): the shear profile is
    multiplied by ``min(1, c_uter_eq·sin²ϕ)·min(1, c_uter_pol·cos²ϕ)`` with
    c_uter_eq = 5, c_uter_pol = 3 (namelist defaults), suppressing the
    singular geostrophic frame near the equator and the grid singularity at
    the poles.  Pass ``damping=None`` to disable.
6.  **Ageostrophic vertical profile** (paper text: the PBL ageostrophic wind
    "is compensated in the upper troposphere in order to conserve mass in
    the atmospheric column"): the surface ageostrophic wind applies through
    the PBL (top at σ_pbl(ϕ) = 0.85 − 0.05·cos²ϕ, reference namelist
    pblp/pble) and is compensated by a uniform counter-flow spread over a
    σ-layer of depth 0.2 (namelist ``dpc``) immediately below the
    tropopause, so the σ-integrated ageostrophic mass flux vanishes exactly.
7.  **Katabatic (A26)–(A27)**: the slope magnitude is *inside* the radical,
    ``uk = √(g·h/CD·(T2m−T*)/T2m·|slope_x|)·sign(−slope_x)`` (verified
    against the reference implementation), gated on the inversion condition
    T2m > T* (else zero), with h = 100 m (paper prose).
8.  **Reference-implementation features deliberately not ported** (grid
    artifacts or time-stepping, not paper equations): staggered-grid
    interpolations, spatial smoothing of α/CD, time relaxation of the
    katabatic wind, and the advective-mass-flux closure (that is stage P4's
    transport machinery). Polar damping of the surface wind is a distinct
    decision -- see note 9: it is ported, independently derived rather than
    translated from the reference's specific technique.
9.  **Polar cosϕ singularity in vg(0)/ua (2026-08-17, P2 exit-gate fix).**
    (A17)-(A20) as printed divide by ``cosϕ``, which the paper's own text
    only excuses at the *equator* (the ``|f|`` floor's stated rationale,
    "the geostrophic approximation is not valid close to the equator").  The
    identical breakdown holds at the poles: on this module's discrete grid
    ``∂psl/∂λ`` does not vanish at the pole row the way the continuous field
    would, so the printed ``1/cosϕ`` diverges there -- measured to reach
    thousands of m/s within one grid row of a pole, a distinct defect from
    the (A38) fix in ``sesam_dynamics.py``. :func:`surface_geostrophic_wind`
    and :func:`ageostrophic_surface_wind` apply :func:`thermal_wind_shear`'s
    own ``min(1, c_pol·cos²ϕ)`` envelope (module docstring note 5) to
    ``vg(0)`` and ``ua`` -- the same documented safeguard already accepted
    into this module for the identical singularity in the shear integral,
    not a new mechanism and not the reference implementation's own
    "pole-half damping" (which remains unported, note 8).
11. **Equatorial breakdown of ug(0)/ua too, not just the pole (2026-08-17).**
    Fixing notes 7/9/10 in ``sesam_dynamics.py`` still left the P2 exit gate
    failing at specific real mid-to-low latitudes (~15-40°): a physically
    ordinary, smooth meridional SLP gradient (~1.4 hPa per grid row, matched
    to a measured real-data location) drove ``ug(0)`` to 100+ m/s. The
    paper's own equatorial ``|f|`` floor (module docstring note 4) bounds
    |f| away from exactly zero but does not scale down the response as
    latitude approaches it -- a floored-but-still-small ``f`` still divides
    an ordinary gradient into an extraordinary wind. Note 9's fix only
    covered ``vg(0)``/``ua``'s pole ``cosϕ`` term; ``ug(0)`` and ``va`` had
    no analogous protection at either extreme. Both
    :func:`surface_geostrophic_wind` and :func:`ageostrophic_surface_wind`
    now apply the *full* ``min(1, c_eq·sin²ϕ)·min(1, c_pol·cos²ϕ)`` envelope
    (both factors, not just the polar one) to *all four* surface components
    (``ug0``, ``vg0``, ``ua``, ``va``) via the shared
    :func:`_geostrophic_frame_damping` helper -- exactly
    :func:`thermal_wind_shear`'s pre-existing damping, now applied
    consistently across the whole (A17)-(A20) surface-plus-shear family
    instead of only the shear half. The ``polar_damping`` parameter was
    renamed ``damping`` (now a ``(c_eq, c_pol)`` tuple, matching
    :func:`thermal_wind_shear`'s signature) in both functions;
    :func:`compute_wind` exposes it as ``surface_damping`` (independent of
    ``thermal_wind_damping``); ``None`` still disables it.

    **Measured sensitivity, not tuned (2026-08-17).** ``c_eq`` reuses
    :func:`thermal_wind_shear`'s namelist value (5.0) as a principled
    starting point, not a value chosen to fit this measurement. A direct
    sweep on the saved 512x1024 state (`docs/SESAM_GAP_ANALYSIS.md`'s
    2026-08-17 P2 entry) found the exit-gate correlation/RMSE response to
    ``c_eq`` is real but *non-monotonic and season-inconsistent* --
    ``c_eq=2`` scores best for DJF correlation while JJA prefers weaker
    values and neither is uniformly best on RMSE. That pattern -- a
    parameter whose "best" value depends on which season's snapshot is
    scored -- is the signature of a genuine constant-calibration question,
    not a discrete defect, and is exactly what
    `docs/SESAM_GAP_ANALYSIS.md` §6's bounded P6 calibration window exists
    for. The shared 5.0/3.0 default is kept rather than hand-picked from
    this sweep.

Grids follow the P1/P2 convention: 2-D fields are ``(H, W)``, rows run
north-to-south on cell centres, the vertical dimension is ``(N, H, W)`` on
absolute-height levels from ``sesam_vertical``.  Planetary constants are
explicit inputs — never hardcoded Earth literals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Paper-text and reference-implementation constants (module docstring notes)
# ---------------------------------------------------------------------------

_Z_REF_M = 100.0  # (A22) prose reference height
_KARMAN = 0.4  # von Karman constant
_CD_OCEAN = 1.3e-3  # reference namelist cd0_ocn
_ORO_ROUGHNESS_FACTOR = 0.004  # (A23)

_ALPHA_MIN_RAD = 0.05  # reference namelist clamps
_ALPHA_MAX_RAD = 0.5
_ALPHA_BISECTION_ITERATIONS = 10

_F_GEO_FLOOR_S = 3.0e-5  # paper text: |f| floor in the geostrophic relations
_F_AGEO_FLOOR_S = 1.0e-5  # paper text: |f| floor in the ageostrophic relations

_C_UTER_EQ = 5.0  # reference namelist: equatorial thermal-wind damping
_C_UTER_POL = 3.0  # reference namelist: polar thermal-wind damping


def _geostrophic_frame_damping(
    lat: np.ndarray, c_eq: float, c_pol: float
) -> np.ndarray:
    """``min(1, c_eq·sin²ϕ)·min(1, c_pol·cos²ϕ)`` (module docstring notes 5/9/11).

    Shared by :func:`thermal_wind_shear` (where it was already applied) and
    :func:`surface_geostrophic_wind`/:func:`ageostrophic_surface_wind`
    (note 11): the same geostrophic-frame breakdown the paper's own equatorial
    ``|f|`` floor addresses, and the pole ``cosϕ`` singularity note 9 found,
    is a property of the (A17)-(A20) frame as a whole, not specific to the
    shear integral.
    """
    return np.minimum(1.0, float(c_eq) * np.sin(lat) ** 2) * np.minimum(
        1.0, float(c_pol) * np.cos(lat) ** 2
    )

_PBL_TOP_POLE = 0.85  # reference namelist pblp (σ of PBL top at poles)
_PBL_TOP_EQ = 0.8  # reference namelist pble (σ of PBL top at equator)
_COMPENSATION_DEPTH_SIGMA = 0.2  # reference namelist dpc

_KATABATIC_H_M = 100.0  # (A26) prose: surface-layer thickness h = 100 m

_P500_PA = 50000.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _latitude_rad(h: int) -> np.ndarray:
    """Signed cell-centre latitude in radians, north-to-south rows."""
    return (0.5 - (np.arange(h, dtype=np.float64) + 0.5) / h) * np.pi


def _check_2d(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D (H, W) field")
    return arr


def horizontal_gradient(
    field_pa_or_k: np.ndarray,
    latitude_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical horizontal gradient of a 2-D field, per radian.

    Returns ``(d/dϕ, d/dλ)`` with meridional differences centred in the
    interior and one-sided at the pole rows, and zonal differences centred
    with periodic longitude.  Units: field-units per radian; divide by
    ``Re`` (and ``cosϕ`` for λ) for physical gradients.
    """
    f = _check_2d("field", field_pa_or_k)
    lat = np.asarray(latitude_rad, dtype=np.float64)
    if lat.shape != (f.shape[0],):
        raise ValueError("latitude_rad must be (H,) matching field rows")
    dphi = np.diff(lat)  # negative (rows run north-to-south)
    d_dphi = np.empty_like(f)
    # Centred interior: (f[j-1] - f[j+1]) / (lat[j-1] - lat[j+1]).
    d_dphi[1:-1] = (f[:-2] - f[2:]) / (-dphi[:-1] - dphi[1:])[:, None]
    # One-sided at the pole rows.
    d_dphi[0] = (f[0] - f[1]) / (-dphi[0])
    d_dphi[-1] = (f[-2] - f[-1]) / (-dphi[-1])
    # Periodic zonal difference per radian of longitude.
    dlam = 2.0 * np.pi / f.shape[1]
    d_dlam = 0.5 * (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / dlam
    return d_dphi, d_dlam


# ---------------------------------------------------------------------------
# (A22)-(A23) drag coefficient, (A21) cross-isobar angle
# ---------------------------------------------------------------------------


def oro_roughness_m(sigma_oro_m: np.ndarray) -> np.ndarray:
    """(A23) ``zoro = 0.004·σoro`` from sub-grid orography standard deviation."""
    return _ORO_ROUGHNESS_FACTOR * np.asarray(sigma_oro_m, dtype=np.float64)


def drag_coefficient(
    roughness_m: np.ndarray,
    oro_roughness: np.ndarray,
    surface_kind: np.ndarray,
    *,
    z_ref_m: float = _Z_REF_M,
    karman: float = _KARMAN,
    cd_ocean: float = _CD_OCEAN,
) -> np.ndarray:
    """(A22) neutral drag coefficient ``CD = (κ/ln(z_ref/(z0+zoro)))²``.

    ``surface_kind`` follows the P1 encoding (0=ocean, 1=land, 2=ice sheet).
    Ocean cells use the reference namelist's constant ``cd0_ocn``; land and
    ice-sheet cells use the logarithmic form with the caller's roughness
    length ``z0`` and the (A23) orographic roughness.  Sea ice has no
    distinct kind in this encoding (its cells take the caller's ocean/ice
    assignment).  ``z0+zoro`` is floored at 1e-4 m and capped below
    ``z_ref`` so the logarithm stays positive and finite.
    """
    z0 = _check_2d("roughness_m", roughness_m)
    zoro = _check_2d("oro_roughness", oro_roughness)
    kind = np.asarray(surface_kind)
    if kind.shape != z0.shape:
        raise ValueError("surface_kind must match the roughness fields")
    rough_eff = np.clip(z0 + zoro, 1e-4, 0.5 * z_ref_m)
    cd_land = (karman / np.log(z_ref_m / rough_eff)) ** 2
    return np.where(kind == 0, cd_ocean, cd_land)


def cross_isobar_angle(
    cd: np.ndarray,
    latitude_rad: np.ndarray,
    *,
    omega: float,
    coriolis_floor_s: float = _F_AGEO_FLOOR_S,
    alpha_limits: tuple[float, float] = (_ALPHA_MIN_RAD, _ALPHA_MAX_RAD),
) -> dict[str, np.ndarray]:
    """(A21) cross-isobar angle α from the drag-closure bisection solve.

    Solves ``sinα/√(1 − sin 2α) = CD/√|f|`` for |α| on [0, π/4] (module
    docstring notes 1–2: the solve is K-independent), then clamps to
    ``alpha_limits`` and signs α by hemisphere (positive NH).  Returns a dict
    with ``alpha_rad`` (signed), ``sin_alpha``, ``cos_alpha``,
    ``sin_cos_alpha`` (positive magnitude), and ``epsilon``
    (``√(1 − sin 2|α|)``).
    """
    cd_arr = _check_2d("cd", cd)
    lat = np.asarray(latitude_rad, dtype=np.float64)
    if lat.shape != (cd_arr.shape[0],):
        raise ValueError("latitude_rad must be (H,) matching cd rows")
    f_abs = np.maximum(2.0 * float(omega) * np.abs(np.sin(lat)), coriolis_floor_s)
    rhs = cd_arr / np.sqrt(f_abs)[:, None]

    lo = np.zeros_like(rhs)
    hi = np.full_like(rhs, np.pi / 4.0)
    for _ in range(_ALPHA_BISECTION_ITERATIONS):
        mid = 0.5 * (lo + hi)
        val = np.sin(mid) / np.sqrt(np.maximum(1.0 - np.sin(2.0 * mid), 1e-12))
        too_big = val > rhs
        hi = np.where(too_big, mid, hi)
        lo = np.where(too_big, lo, mid)
    alpha_abs = np.clip(0.5 * (lo + hi), alpha_limits[0], alpha_limits[1])
    sign = np.where(lat < 0.0, -1.0, 1.0)[:, None]
    alpha = alpha_abs * sign
    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)
    return {
        "alpha_rad": alpha,
        "sin_alpha": sin_a,
        "cos_alpha": cos_a,
        "sin_cos_alpha": np.sin(alpha_abs) * np.cos(alpha_abs),
        "epsilon": np.sqrt(np.maximum(1.0 - np.sin(2.0 * alpha_abs), 0.0)),
    }


# ---------------------------------------------------------------------------
# (A17)-(A18) surface geostrophic wind and thermal-wind shear
# ---------------------------------------------------------------------------


def surface_geostrophic_wind(
    dpsl_dphi: np.ndarray,
    dpsl_dlam: np.ndarray,
    latitude_rad: np.ndarray,
    *,
    radius_m: float,
    omega: float,
    rho0_kg_m3: float,
    coriolis_floor_s: float = _F_GEO_FLOOR_S,
    damping: tuple[float, float] | None = (_C_UTER_EQ, _C_UTER_POL),
) -> tuple[np.ndarray, np.ndarray]:
    """(A17)-(A18) at z = 0: geostrophic wind from the SLP gradient.

    ``ug(0) = −(1/(ρ0·f·Re))·∂psl/∂ϕ``,
    ``vg(0) = (1/(ρ0·f·Re·cosϕ))·∂psl/∂λ`` with ``f = 2Ω·sinϕ``
    sign-preserving and |f| floored at 3e-5 s⁻¹ (paper text).

    ``vg(0)`` as printed is singular at the poles (``cosϕ → 0`` with
    ``∂psl/∂λ`` generally nonzero on this module's discrete grid, unlike the
    true continuous field), and both components amplify without bound as
    ``|f| → coriolis_floor_s`` near the equator even with the paper's own
    hard floor (module docstring note 11): a smooth, physically ordinary
    ``∂psl/∂ϕ`` at low latitude is enough to drive |ug(0)| to 100+ m/s once
    divided by the floored-but-still-small ``f``. Module docstring note 8
    already flags a pole-only version of this as the reference
    implementation's own "pole-half damping" -- not ported there because it
    is a reference-specific numerical technique, not a paper equation.
    ``damping`` applies the same ``min(1, c_eq·sin²ϕ)·min(1, c_pol·cos²ϕ)``
    envelope :func:`thermal_wind_shear` already uses for the identical
    breakdown in the shear integral (module docstring note 5) to *both*
    surface components: an independently-derived numerical safeguard for the
    same well-known breakdown of the geostrophic approximation the paper
    text itself invokes for the equatorial ``|f|`` floor -- not a
    translation of the reference's technique. Pass ``damping=None`` to
    disable (the pre-2026-08-17 behaviour).
    """
    dp_dphi = _check_2d("dpsl_dphi", dpsl_dphi)
    dp_dlam = _check_2d("dpsl_dlam", dpsl_dlam)
    if dp_dphi.shape != dp_dlam.shape:
        raise ValueError("SLP gradient components must share a shape")
    lat = np.asarray(latitude_rad, dtype=np.float64)
    if lat.shape != (dp_dphi.shape[0],):
        raise ValueError("latitude_rad must be (H,) matching gradient rows")
    f = 2.0 * float(omega) * np.sin(lat)
    f = np.sign(f) * np.maximum(np.abs(f), coriolis_floor_s)
    cos_lat = np.cos(lat)
    cos_lat_safe = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)
    ug0 = -dp_dphi / (float(rho0_kg_m3) * f[:, None] * float(radius_m))
    vg0 = dp_dlam / (float(rho0_kg_m3) * f[:, None] * float(radius_m) * cos_lat_safe[:, None])
    if damping is not None:
        c_eq, c_pol = damping
        damp = _geostrophic_frame_damping(lat, c_eq, c_pol)
        ug0 = ug0 * damp[:, None]
        vg0 = vg0 * damp[:, None]
    return ug0, vg0


def thermal_wind_shear(
    temperature_z: np.ndarray,
    levels_m: np.ndarray,
    latitude_rad: np.ndarray,
    *,
    radius_m: float,
    omega: float,
    reference_temp_k: float,
    gravity: float,
    coriolis_floor_s: float = _F_GEO_FLOOR_S,
    damping: tuple[float, float] | None = (_C_UTER_EQ, _C_UTER_POL),
) -> tuple[np.ndarray, np.ndarray]:
    """(A17)-(A18) integral terms: thermal-wind shear above the surface.

    ``ug_shear(z) = −∫₀^z (g/(T0·f·Re))·∂T/∂ϕ dz``,
    ``vg_shear(z) = +∫₀^z (g/(T0·f·Re·cosϕ))·∂T/∂λ dz`` integrated as
    cumulative trapezoids on the ``(N, H, W)`` level grid (level 0 must be
    the surface, where the shear is zero).  The shear is multiplied by the
    equatorial/polar damping of module docstring note 5 (disable with
    ``damping=None``).
    """
    t_z = np.asarray(temperature_z, dtype=np.float64)
    if t_z.ndim != 3:
        raise ValueError("temperature_z must be a 3-D (N, H, W) field")
    levels = np.asarray(levels_m, dtype=np.float64)
    if levels.ndim != 1 or levels.size != t_z.shape[0]:
        raise ValueError("levels_m must be a 1-D level axis matching temperature_z")
    lat = np.asarray(latitude_rad, dtype=np.float64)
    if lat.shape != (t_z.shape[1],):
        raise ValueError("latitude_rad must be (H,) matching temperature rows")
    f = 2.0 * float(omega) * np.sin(lat)
    f = np.sign(f) * np.maximum(np.abs(f), coriolis_floor_s)
    cos_lat = np.cos(lat)

    n = levels.size
    shear_u = np.zeros_like(t_z)
    shear_v = np.zeros_like(t_z)
    coef_u = float(gravity) / (float(reference_temp_k) * f * float(radius_m))
    coef_v = coef_u / cos_lat
    for k in range(1, n):
        dphi_lo, dlam_lo = horizontal_gradient(t_z[k - 1], lat)
        dphi_hi, dlam_hi = horizontal_gradient(t_z[k], lat)
        dz = levels[k] - levels[k - 1]
        integ_u = -0.5 * (dphi_lo + dphi_hi)  # trapezoid of -dT/dphi
        integ_v = 0.5 * (dlam_lo + dlam_hi)  # trapezoid of +dT/dlam
        shear_u[k] = shear_u[k - 1] + coef_u[:, None] * integ_u * dz
        shear_v[k] = shear_v[k - 1] + coef_v[:, None] * integ_v * dz

    if damping is not None:
        c_eq, c_pol = damping
        damp = _geostrophic_frame_damping(lat, c_eq, c_pol)
        shear_u = shear_u * damp[None, :, None]
        shear_v = shear_v * damp[None, :, None]
    return shear_u, shear_v


# ---------------------------------------------------------------------------
# (A19)-(A20) ageostrophic PBL wind and its vertical profile
# ---------------------------------------------------------------------------


def ageostrophic_surface_wind(
    dpsl_dphi: np.ndarray,
    dpsl_dlam: np.ndarray,
    latitude_rad: np.ndarray,
    sin_cos_alpha: np.ndarray,
    *,
    radius_m: float,
    omega: float,
    rho0_kg_m3: float,
    coriolis_floor_s: float = _F_AGEO_FLOOR_S,
    damping: tuple[float, float] | None = (_C_UTER_EQ, _C_UTER_POL),
) -> tuple[np.ndarray, np.ndarray]:
    """(A19)-(A20): ageostrophic PBL wind from SLP and the cross-isobar angle.

    ``ua = −(sinα·cosα)/(ρ0·|f|·Re·cosϕ)·∂psl/∂λ``,
    ``va = −(sinα·cosα)/(ρ0·|f|·Re)·∂psl/∂ϕ`` with |f| floored at 1e-5 s⁻¹
    and ``sinα·cosα`` a positive magnitude (module docstring note 3): flow
    crosses isobars toward low pressure in both hemispheres.

    ``ua``/``va`` share ``ug(0)``/``vg(0)``'s pole ``cosϕ`` singularity and
    equatorial breakdown (see :func:`surface_geostrophic_wind`, module
    docstring note 11) -- more so, since the ageostrophic ``|f|`` floor
    (1e-5) is smaller than the geostrophic one (3e-5). ``damping`` applies
    the same ``min(1, c_eq·sin²ϕ)·min(1, c_pol·cos²ϕ)`` envelope to both
    components. Pass ``damping=None`` to disable (the pre-2026-08-17
    behaviour).
    """
    dp_dphi = _check_2d("dpsl_dphi", dpsl_dphi)
    dp_dlam = _check_2d("dpsl_dlam", dpsl_dlam)
    if dp_dphi.shape != dp_dlam.shape:
        raise ValueError("SLP gradient components must share a shape")
    lat = np.asarray(latitude_rad, dtype=np.float64)
    if lat.shape != (dp_dphi.shape[0],):
        raise ValueError("latitude_rad must be (H,) matching gradient rows")
    scab = _check_2d("sin_cos_alpha", sin_cos_alpha)
    if scab.shape != dp_dphi.shape:
        raise ValueError("sin_cos_alpha must match the gradient shape")
    f_abs = np.maximum(2.0 * float(omega) * np.abs(np.sin(lat)), coriolis_floor_s)
    cos_lat = np.cos(lat)
    cos_lat_safe = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)
    denom = float(rho0_kg_m3) * f_abs * float(radius_m)
    ua = -scab * dp_dlam / (denom * cos_lat_safe)[:, None]
    va = -scab * dp_dphi / denom[:, None]
    if damping is not None:
        c_eq, c_pol = damping
        damp = _geostrophic_frame_damping(lat, c_eq, c_pol)
        ua = ua * damp[:, None]
        va = va * damp[:, None]
    return ua, va


def pbl_top_sigma(latitude_rad: np.ndarray) -> np.ndarray:
    """PBL-top pressure fraction ``σ_pbl(ϕ) = 0.85 − 0.05·cos²ϕ`` (note 6)."""
    lat = np.asarray(latitude_rad, dtype=np.float64)
    return _PBL_TOP_POLE - (_PBL_TOP_POLE - _PBL_TOP_EQ) * np.cos(lat) ** 2


def ageostrophic_profile(
    ua_surface: np.ndarray,
    va_surface: np.ndarray,
    sigma_levels: np.ndarray,
    latitude_rad: np.ndarray,
    sigma_tropopause: np.ndarray | float,
    *,
    compensation_depth_sigma: float = _COMPENSATION_DEPTH_SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Vertical profile of the ageostrophic wind (module docstring note 6).

    The surface ageostrophic wind applies uniformly through the PBL
    (``σ > σ_pbl(ϕ)``) and is compensated by a uniform counter-flow spread
    over the σ-layer ``[σ_trop, σ_trop + compensation_depth]`` immediately
    below the tropopause, with magnitude chosen so the σ-integrated
    ageostrophic mass flux of every column vanishes exactly:

    ``∫ ua dσ = ua_sfc·(1 − σ_pbl) + ua_comp·depth = 0``.

    ``sigma_levels`` is the 1-D σ = p/p0 axis of the ``(N, H, W)`` grid
    (values in (0, 1], surface first).  Returns ``(ua_z, va_z)``.
    """
    ua = _check_2d("ua_surface", ua_surface)
    va = _check_2d("va_surface", va_surface)
    if ua.shape != va.shape:
        raise ValueError("surface ageostrophic components must share a shape")
    sigma = np.asarray(sigma_levels, dtype=np.float64)
    if sigma.ndim != 1:
        raise ValueError("sigma_levels must be 1-D")
    lat = np.asarray(latitude_rad, dtype=np.float64)
    if lat.shape != (ua.shape[0],):
        raise ValueError("latitude_rad must be (H,) matching the surface fields")
    sig_trop = np.asarray(sigma_tropopause, dtype=np.float64)
    if sig_trop.ndim == 0:
        sig_trop = np.full(ua.shape, float(sig_trop))
    if sig_trop.shape != ua.shape:
        raise ValueError("sigma_tropopause must be a scalar or a 2-D field")

    sig_pbl = pbl_top_sigma(lat)  # (H,)
    depth = float(compensation_depth_sigma)
    comp_mag = -(1.0 - sig_pbl)[:, None] / depth  # per-σ counter-flow factor

    ua_z = np.zeros((sigma.size, *ua.shape))
    va_z = np.zeros_like(ua_z)
    in_pbl = sigma[:, None] > sig_pbl[None, :]  # (N, H)
    in_comp = (sigma[:, None, None] > sig_trop[None, :, :]) & (
        sigma[:, None, None] <= sig_trop[None, :, :] + depth
    )
    ua_z = np.where(in_pbl[:, :, None], ua[None, :, :], 0.0)
    va_z = np.where(in_pbl[:, :, None], va[None, :, :], 0.0)
    ua_z = ua_z + np.where(in_comp, ua[None, :, :] * comp_mag[None, :, :], 0.0)
    va_z = va_z + np.where(in_comp, va[None, :, :] * comp_mag[None, :, :], 0.0)
    return ua_z, va_z


# ---------------------------------------------------------------------------
# (A26)-(A27) katabatic wind, (A24)-(A25) Taylor surface wind
# ---------------------------------------------------------------------------


def katabatic_wind(
    t2m_k: np.ndarray,
    skin_temp_k: np.ndarray,
    cd: np.ndarray,
    dzs_dlam: np.ndarray,
    dzs_dphi: np.ndarray,
    latitude_rad: np.ndarray,
    *,
    radius_m: float,
    gravity: float,
    layer_depth_m: float = _KATABATIC_H_M,
) -> tuple[np.ndarray, np.ndarray]:
    """(A26)-(A27) katabatic wind from buoyancy–friction balance (note 7).

    ``uk = √(g·h/CD·(T2m−T*)/T2m·|slope_x|)·sign(−slope_x)`` (slope inside
    the radical), ``vk`` likewise with ``slope_y``; both gated on the
    inversion condition ``T2m > T*`` (cold surface over a slope), else zero.
    Slopes are dimensionless: ``slope_x = (1/(Re·cosϕ))·∂zs/∂λ``,
    ``slope_y = (1/Re)·∂zs/∂ϕ``.
    """
    t2m = _check_2d("t2m_k", t2m_k)
    tskin = _check_2d("skin_temp_k", skin_temp_k)
    cd_arr = _check_2d("cd", cd)
    dz_lam = _check_2d("dzs_dlam", dzs_dlam)
    dz_phi = _check_2d("dzs_dphi", dzs_dphi)
    if not (t2m.shape == tskin.shape == cd_arr.shape == dz_lam.shape == dz_phi.shape):
        raise ValueError("all katabatic inputs must share a 2-D shape")
    lat = np.asarray(latitude_rad, dtype=np.float64)
    if lat.shape != (t2m.shape[0],):
        raise ValueError("latitude_rad must be (H,) matching the input rows")

    slope_x = dz_lam / (float(radius_m) * np.cos(lat))[:, None]
    slope_y = dz_phi / float(radius_m)
    delta_t = t2m - tskin
    factor = np.where(
        delta_t > 0.0,
        float(gravity) * float(layer_depth_m) / np.maximum(cd_arr, 1e-8)
        * delta_t / np.maximum(t2m, 1.0),
        0.0,
    )
    uk = np.sqrt(factor * np.abs(slope_x)) * np.sign(-slope_x)
    vk = np.sqrt(factor * np.abs(slope_y)) * np.sign(-slope_y)
    return uk, vk


def taylor_surface_wind(
    ug0: np.ndarray,
    vg0: np.ndarray,
    sin_alpha: np.ndarray,
    cos_alpha: np.ndarray,
    epsilon: np.ndarray,
    uk: np.ndarray | float = 0.0,
    vk: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """(A24)-(A25) near-surface wind: rotated geostrophic wind + katabatic.

    ``us = ε·(ug(0)·cosα − vg(0)·sinα) + uk``,
    ``vs = ε·(vg(0)·cosα + ug(0)·sinα) + vk`` with signed ``sinα``
    (positive NH) and ``ε = √(1 − sin 2|α|)`` from :func:`cross_isobar_angle`.
    """
    ug = _check_2d("ug0", ug0)
    vg = _check_2d("vg0", vg0)
    sa = _check_2d("sin_alpha", sin_alpha)
    ca = _check_2d("cos_alpha", cos_alpha)
    eps = _check_2d("epsilon", epsilon)
    if not (ug.shape == vg.shape == sa.shape == ca.shape == eps.shape):
        raise ValueError("Taylor-wind inputs must share a 2-D shape")
    us = eps * (ug * ca - vg * sa) + np.asarray(uk, dtype=np.float64)
    vs = eps * (vg * ca + ug * sa) + np.asarray(vk, dtype=np.float64)
    return us, vs


# ---------------------------------------------------------------------------
# (A16) assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SesamWind:
    """Assembled SESAM wind field (A16).

    2-D fields are ``(H, W)``; vertical fields are ``(N, H, W)`` on the
    caller's level grid.  ``u500_pa_zonal_m_s`` is the zonal-mean 500 hPa
    zonal wind ``(H,)`` consumed by the Charney–Eliassen term of
    ``sesam_dynamics``.
    """

    surface_u_m_s: np.ndarray
    surface_v_m_s: np.ndarray
    geostrophic_surface_u_m_s: np.ndarray
    geostrophic_surface_v_m_s: np.ndarray
    ageostrophic_surface_u_m_s: np.ndarray
    ageostrophic_surface_v_m_s: np.ndarray
    katabatic_u_m_s: np.ndarray
    katabatic_v_m_s: np.ndarray
    u_z_m_s: np.ndarray
    v_z_m_s: np.ndarray
    geostrophic_u_z_m_s: np.ndarray
    geostrophic_v_z_m_s: np.ndarray
    ageostrophic_u_z_m_s: np.ndarray
    ageostrophic_v_z_m_s: np.ndarray
    u500_pa_zonal_m_s: np.ndarray
    alpha_rad: np.ndarray
    sin_cos_alpha: np.ndarray
    drag_coefficient: np.ndarray


def u_at_pressure(
    u_z: np.ndarray,
    pressure_z: np.ndarray,
    target_pa: float,
) -> np.ndarray:
    """Interpolate a wind profile to a pressure surface, per column.

    ``u_z`` and ``pressure_z`` share shape ``(N, H, W)`` with pressure
    decreasing along N.  Levels above the column top clamp to the top level.
    """
    u = np.asarray(u_z, dtype=np.float64)
    p = np.asarray(pressure_z, dtype=np.float64)
    if u.shape != p.shape or u.ndim != 3:
        raise ValueError("u_z and pressure_z must share a 3-D shape")
    n, h, w = u.shape
    out = np.empty((h, w))
    target = float(target_pa)
    for j in range(h):
        for i in range(w):
            col_p = p[:, j, i]
            col_u = u[:, j, i]
            if target >= col_p[0]:
                out[j, i] = col_u[0]
            elif target <= col_p[-1]:
                out[j, i] = col_u[-1]
            else:
                out[j, i] = np.interp(target, col_p[::-1], col_u[::-1])
    return out


def compute_wind(
    *,
    slp_pa: np.ndarray,
    temperature_z: np.ndarray,
    levels_m: np.ndarray,
    pressure_z: np.ndarray,
    skin_temp_k: np.ndarray,
    t2m_k: np.ndarray,
    surface_elevation_m: np.ndarray,
    surface_kind: np.ndarray,
    roughness_m: np.ndarray,
    sigma_oro_m: np.ndarray | None = None,
    tropopause_sigma: np.ndarray | float = 0.2,
    gravity: float,
    radius_m: float,
    omega: float,
    rho0_kg_m3: float,
    reference_temp_k: float,
    thermal_wind_damping: tuple[float, float] | None = (_C_UTER_EQ, _C_UTER_POL),
    surface_damping: tuple[float, float] | None = (_C_UTER_EQ, _C_UTER_POL),
) -> SesamWind:
    """Assemble the full (A16) wind field from SLP and 3-D temperature.

    ``slp_pa`` comes from ``sesam_dynamics.compute_slp``;
    ``temperature_z``/``pressure_z`` on ``levels_m`` from
    ``sesam_vertical.compute_vertical_structure``.  ``t2m_k`` is the
    near-surface (2 m) air temperature used by the katabatic term;
    ``sigma_oro_m`` is the sub-grid orography standard deviation feeding
    (A23) (pass ``None`` for zero orographic roughness);
    ``tropopause_sigma`` is the σ = p/p0 of the tropopause for the
    ageostrophic compensation layer (scalar or 2-D). ``surface_damping``
    (module docstring note 11) is the ``(c_eq, c_pol)`` pair passed to
    :func:`surface_geostrophic_wind`/:func:`ageostrophic_surface_wind`,
    independent of ``thermal_wind_damping``.
    """
    slp = _check_2d("slp_pa", slp_pa)
    t_z = np.asarray(temperature_z, dtype=np.float64)
    p_z = np.asarray(pressure_z, dtype=np.float64)
    if t_z.ndim != 3 or p_z.shape != t_z.shape:
        raise ValueError("temperature_z and pressure_z must share a 3-D shape")
    if t_z.shape[1:] != slp.shape:
        raise ValueError("temperature_z trailing axes must match slp_pa")
    h, w = slp.shape
    lat = _latitude_rad(h)
    levels = np.asarray(levels_m, dtype=np.float64)

    kind = np.asarray(surface_kind)
    zs = _check_2d("surface_elevation_m", surface_elevation_m)
    z0 = _check_2d("roughness_m", roughness_m)
    zoro = (
        oro_roughness_m(sigma_oro_m)
        if sigma_oro_m is not None
        else np.zeros((h, w))
    )
    cd = drag_coefficient(z0, zoro, kind)
    angle = cross_isobar_angle(cd, lat, omega=omega)

    dp_dphi, dp_dlam = horizontal_gradient(slp, lat)
    ug0, vg0 = surface_geostrophic_wind(
        dp_dphi, dp_dlam, lat, radius_m=radius_m, omega=omega, rho0_kg_m3=rho0_kg_m3,
        damping=surface_damping,
    )
    shear_u, shear_v = thermal_wind_shear(
        t_z,
        levels,
        lat,
        radius_m=radius_m,
        omega=omega,
        reference_temp_k=reference_temp_k,
        gravity=gravity,
        damping=thermal_wind_damping,
    )
    ug_z = ug0[None, :, :] + shear_u
    vg_z = vg0[None, :, :] + shear_v

    ua_sfc, va_sfc = ageostrophic_surface_wind(
        dp_dphi,
        dp_dlam,
        lat,
        angle["sin_cos_alpha"],
        radius_m=radius_m,
        omega=omega,
        rho0_kg_m3=rho0_kg_m3,
        damping=surface_damping,
    )
    # σ levels of the input grid, from the surface column-mean pressure
    # profile (the exponential reference profile is uniform to first order).
    sigma_levels = (p_z[:, :, :] / p_z[0:1, :, :]).mean(axis=(1, 2))
    ua_z, va_z = ageostrophic_profile(
        ua_sfc, va_sfc, sigma_levels, lat, tropopause_sigma
    )

    dzs_dphi, dzs_dlam = horizontal_gradient(zs, lat)
    uk, vk = katabatic_wind(
        t2m_k,
        skin_temp_k,
        cd,
        dzs_dlam,
        dzs_dphi,
        lat,
        radius_m=radius_m,
        gravity=gravity,
    )
    us, vs = taylor_surface_wind(
        ug0,
        vg0,
        angle["sin_alpha"],
        angle["cos_alpha"],
        angle["epsilon"],
        uk,
        vk,
    )

    u_z = ug_z + ua_z
    v_z = vg_z + va_z
    u500 = u_at_pressure(u_z, p_z, _P500_PA)
    u500_zonal = np.mean(u500, axis=1)

    return SesamWind(
        surface_u_m_s=us,
        surface_v_m_s=vs,
        geostrophic_surface_u_m_s=ug0,
        geostrophic_surface_v_m_s=vg0,
        ageostrophic_surface_u_m_s=ua_sfc,
        ageostrophic_surface_v_m_s=va_sfc,
        katabatic_u_m_s=uk,
        katabatic_v_m_s=vk,
        u_z_m_s=u_z,
        v_z_m_s=v_z,
        geostrophic_u_z_m_s=ug_z,
        geostrophic_v_z_m_s=vg_z,
        ageostrophic_u_z_m_s=ua_z,
        ageostrophic_v_z_m_s=va_z,
        u500_pa_zonal_m_s=u500_zonal,
        alpha_rad=angle["alpha_rad"],
        sin_cos_alpha=angle["sin_cos_alpha"],
        drag_coefficient=cd,
    )


__all__ = [
    "SesamWind",
    "ageostrophic_profile",
    "ageostrophic_surface_wind",
    "compute_wind",
    "cross_isobar_angle",
    "drag_coefficient",
    "horizontal_gradient",
    "katabatic_wind",
    "oro_roughness_m",
    "pbl_top_sigma",
    "surface_geostrophic_wind",
    "taylor_surface_wind",
    "thermal_wind_shear",
    "u_at_pressure",
]
