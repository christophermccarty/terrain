"""SESAM dynamics: sea-level pressure reconstruction — Appendix A2 of Willeit et al. (2022).

Pure, read-only kernels for the CLIMBER-X/SESAM 2.5-D atmosphere's sea-level
pressure (SLP) construction, transcribed from Appendix A2 of

    Willeit, M., Ganopolski, A., Robinson, A., and Edwards, N. R.:
    GMD 15, 5905–5948 (2022), https://doi.org/10.5194/gmd-15-5905-2022
    (CC-BY 4.0; equations cited below as (A#) against that paper).

This is SESAM stage P2's first sub-deliverable (docs/SESAM_GAP_ANALYSIS.md §7):
the SLP construction — zonal SLP from mean-meridional-cell physics (A29–A35),
azonal thermal SLP (A37), and the Charney–Eliassen topographic term (A38–A39)
— assembled per (A28)/(A36).  The 3-D wind assembly ((A16)–(A27): geostrophic
+ thermal wind, Taylor-model surface wind, katabatic) is P2's second
sub-deliverable and is not in this module.

The supported PlanetSim pipeline never calls these functions (the
``PlanetParams.enable_sesam_dynamics`` gate is off by default and nothing is
wired into ``simulate.py``), so this module has zero default-path climate
impact by construction.

**Verification notes** (equation semantics checked 2026-08-16 against the
article's HTML MathML — which preserves the fraction/cases structure that
PDF text extraction flattens — and cross-checked against the GPL-3.0
CLIMBER-X Fortran, which per docs/SESAM_GAP_ANALYSIS.md §5 may be *read* to
cross-check semantics but not copied or translated; all code here is written
from the CC-BY paper):

1.  **(A31) operand grouping.** The meridional-cell coordinate is
    ``φ = 6·Dhad·(ϕ − φITCZ/(c1mmc·(ϕ − φITCZ)² + 1))`` — the rational factor
    applies to ``φITCZ`` only (HTML MathML ``mfenced``/``mfrac`` structure;
    matches the reference implementation).
2.  **(A33) is a fraction.** ``Dhad = c3mmc/(T_trp − c4mmc)``, not the
    flattened product the PDF text layer suggests.  The reciprocal form is
    dimensionally consistent (Dhad is a dimensionless width scale ≈ 1 at the
    present tropical temperature) and gives the stated behaviour: tropical
    warming shrinks Dhad, which moves the ``φ = ±π`` Hadley edges poleward,
    i.e. "Hadley cells are expanding with warming".  The reference
    implementation clamps the scale to [0.5, 1.5] and floors ``T_trp`` at
    ``c4mmc + 50 K``; both safeguards are adopted here and documented.
3.  **(A34) operand order (the one real correction).** As printed,
    ``ΔT₁ = T̄(φ₁) − max(T̄(φ))`` and ``ΔTᵢ = T̄(φᵢ) − T̄(φᵢ₋₁)`` are negative
    on Earth in all three cells; fed through (A30) ``v̄a = −Cᵢ·ΔTᵢ·Fz·sinφ``
    and (A29) that yields *poleward* surface flow in all six cell branches
    (both hemispheres) — a divergent, physically impossible mean meridional
    circulation that would put SLP highs on the equator and lows at 30°.
    The circulation-consistent ordering, and the one the reference
    implementation evaluates, is the reverse difference at the same fixed
    latitudes: Hadley ``max(0, max(T̄) − T̄(±π/6))``, Ferrel
    ``T̄(±π/6) − T̄(±π/3)``, polar ``T̄(±π/3) − T̄(±π/2)`` — positive on Earth,
    giving equatorward Hadley/polar surface flow and poleward Ferrel flow,
    i.e. subtropical highs, subpolar lows and the ITCZ trough out of (A29).
    This module implements the circulation-correct ordering;
    ``testing/test_sesam_dynamics.py`` pins the sign of every cell branch in
    both hemispheres and plants the printed ordering to show it reverses the
    circulation.  Gradient latitudes are the paper's fixed
    ``|φ₁| = π/6``, ``|φ₂| = π/3``, ``|φ₃| = π/2``.
4.  **(A39) glyph.** The printed ``p*sl,O = 9fρ(500 hPa)`` is the
    streamfunction Ψ mis-OCR'd: ``p*sl,O = ρ(500 hPa)·f·Ψ`` (confirmed by the
    reference implementation, which also uses ``|f|`` in the conversion).
    Directly confirmed 2026-08-17 against the article PDF (Appendix A2,
    p. 44): the printed equation is ``p∗sl,O = Ψfρ(500hPa)``.
5.  **Cross-isobar factor.** (A29) divides by the zonal mean of
    ``sinα·cosα``.  α is solved from the drag closure (A21) by the wind
    assembly (P2's second sub-deliverable, ``sesam_wind.py``).  Until then
    ``sin_cos_alpha_bar`` is a **required caller input** — nothing here
    fabricates it.  Convention (verified against the reference implementation
    and the (A20) physics, 2026-08-16): ``sinα·cosα`` enters (A19)/(A20)/
    (A29) as a positive *magnitude* together with ``|f|`` — the ageostrophic
    PBL flow must cross isobars toward low pressure in both hemispheres,
    which an unsigned factor with ``|f|`` achieves and a signed factor with
    signed ``f`` reproduces only when both carry the hemisphere sign (the two
    forms are then algebraically identical).  This module uses the unsigned
    ``|f|``/magnitude form, which is unambiguous for array input.  The zonal
    SLP *pattern* is insensitive to a uniform-magnitude placeholder; its
    amplitude scales with 1/|sinαcosα|.
6.  **Reference-implementation features deliberately not ported** (they are
    grid/stability artifacts or tuning, not paper equations): staggered-grid
    moving averages, the reference implementation's own spatial-smoothing
    filters, the polar and equatorial azonal damping factors, and the time
    relaxation of the azonal SLP (this module is a diagnostic, not a
    time-stepped state). This is a distinct decision from note 7 below: no
    Fortran technique or constant is used anywhere in this module.
7.  **Resolution matching (2026-08-17, P2 exit-gate fix).** The P2 exit-gate
    measurement (`docs/SESAM_GAP_ANALYSIS.md` §7) found the full azonal chain
    scoring *worse* than the incumbent generator at PlanetSim's native
    512x1024 grid, while the zonal-only chain already beat it: (A37)'s
    ~232 Pa/K coefficient turns the saved state's full-resolution sea-level
    temperature anomalies into implausible local SLP swings, and (A38)-(A39)
    respond to full-resolution 8848 m terrain the same way. This is not a
    numerical instability; the SESAM constants (`H0_slp`, `c5mmc`, the
    Charney-Eliassen wave geometry) come from a paper validated at a native
    5 deg x 5 deg grid (`docs/SESAM_GAP_ANALYSIS.md` §8's own risk note), and
    a 5 deg grid cell's temperature/terrain variance is intrinsically smaller
    than a 0.35 deg cell's -- feeding sub-grid variance the equations were
    never calibrated against is a resolution-domain mismatch, not a
    grid-stability artifact of the kind note 6 excludes.  :func:`resolution_matched_field`
    box-averages to a grid matched to that native resolution (default 36
    rows, i.e. ~5 deg, derived from the caller's grid, never a hardcoded
    Earth literal) and bilinearly regrids back (periodic in longitude,
    edge-clamped at the poles) so the SLP gradients the wind assembly
    differentiates stay smooth. It is a generic, symmetric box+bilinear
    regrid -- a standard numerical technique applied identically at every
    latitude/longitude, written independently of the reference
    implementation, not a translation of its `nsmooth_*` filters (which
    remain unported per note 6; those are staggered-grid, non-periodic, and
    tuned to the reference's own timestep). :func:`compute_slp` applies it to
    the 2-D inputs of the azonal thermal and orographic terms only -- the
    zonal cell-physics path (A29)-(A35) already collapses to a 1-D zonal-mean
    profile and is unaffected. **Measured (2026-08-17) to be insufficient on
    its own** -- see note 9.
8.  **The ftrop coordinate.** The (A14) tropical weight consumed by
    ``sesam_vertical.tropical_weight`` is ``1 − sin⁸(fi)`` with
    ``fi = clamp(c_hrs·(ϕ − had_fi)/(0.5·had_width), ±π/2)`` where
    ``had_fi``/``had_width`` are the Hadley centre/width diagnosed from the
    cell coordinate here (:func:`hadley_geometry`).  ``c_hrs = 0.7`` is from
    the CLIMBER-X namelist (the paper's (A14) prints ``ftrop = 1 − sin⁸φ``
    without defining φ's construction; the same rescaled-latitude structure
    with 0.85 = asin(0.1^(1/8)) appears in the printed (A11)).  This replaces
    P1's documented latitude placeholder.
9.  **(A38) denominator: erroneous extra `u` factor (2026-08-17, the real P2
    exit-gate fix).**  The printed (A38) is the barotropic vorticity equation
    ``u·∂ζ/∂λ + β·v + ζ/τe = −(f/H_T)·0.4·u·∂zs/∂λ`` (confirmed 2026-08-17
    against the article PDF, Appendix A2 p. 43-44 -- the paper states the
    equation and that it is solved by Fourier expansion per latitudinal belt
    using FFTW3, but does not print a closed-form spectral solution).
    Substituting a single Fourier/meridional mode (``ζ = −Kn²Ψ̂``,
    ``v = i·kzn·Ψ̂``) and collecting terms gives
    ``Ψ̂ₙ·[u·Kn² − β − i·Kn²/(τe·kzn)] = (f/H_T)·0.4·u·ẑsₙ`` -- matching
    this function's own documented formula.  The *code*, however, computed
    ``denom_imag = Kn²/(τe·kzn·u)`` -- an extra ``u`` in the denominator not
    present in the derivation or the docstring above it.  Two independent
    checks confirm the extra factor is wrong: (a) dimensional analysis --
    ``denom_real`` has units s⁻¹m⁻¹ (matching ``β``); the erroneous form has
    units m⁻², the corrected form s⁻¹m⁻¹; (b) measurement -- at realistic
    mid-latitude u500 (10-30 m/s) the erroneous damping term is ~1e-13,
    roughly four orders of magnitude too small to bound the response near
    the resonance ``u·Kn² = β`` that exists for some low integer n at any
    positive u, giving |response| spikes above 1e5 and, propagated through
    (A39) and the wind assembly's SLP gradient, the P2 exit-gate's measured
    100+ m/s local winds.  With the fix, the same sweep stays in the
    1e3-1e4 range at every tested (lat, n, u) -- bounded, no resonance.
    ``testing/test_sesam_dynamics.py`` pins the corrected formula and plants
    the erroneous one to show it reproduces the resonance spike.  This was
    the dominant P2 exit-gate defect; note 7's resolution matching remains a
    real, independently-justified secondary improvement (it also damps the
    (A37) thermal term, which this fix does not touch).
10. **`u500(ϕ)` row-to-row noise, not the 2-D fields (2026-08-17).**  After
    notes 7 and 9, the exit gate was still failing at specific real
    latitudes (Sea-of-Japan-adjacent rows near 35°N) despite the 2-D SLP
    inputs there being smooth: ``dp/dλ`` was small (~2e3 Pa/rad) but
    ``dp/dϕ`` was enormous (~1.8e5 Pa/rad), i.e. a meridional, not zonal,
    problem. Tracing it: (A38) is solved *independently per row* with no
    coupling between adjacent latitudes, so it has no mechanism to keep its
    output smooth in ϕ even when every 2-D input is. The row-to-row driver
    turned out to be ``u500`` itself -- the zonal-mean 500 hPa wind closing
    the two-pass SLP↔wind loop -- which is derived from real, ungridded
    saved-state temperature/SLP fields and genuinely oscillates row to row
    (measured: -28, -16, -15, -18, -1, +13, +26 m/s across seven adjacent
    rows). Combined with the ``+0.1 m/s`` westerly-only floor (module
    docstring note: "resonance is only supported for westerly flow"), a
    sign flip between rows means the response jumps from near-zero (u
    floored) to its full unfloored value within 1-2 grid rows -- a sharp,
    physically spurious meridional gradient with nothing to do with
    resolution or the (A38) fix above. Fixing notes 7/9 could not remove
    this: it is a defect in the *driving profile*, not the kernel.
    :func:`resolution_matched_profile` (the 1-D latitude analogue of note
    7's 2-D regrid -- same box+linear-interp mechanism, no longitude to
    wrap) smooths ``u500`` to the same ~5° scale before :func:`compute_slp`
    passes it to :func:`charney_eliassen_slp`; verified to turn the same
    seven-row oscillation into a monotonic profile and the corresponding
    orographic SLP from an oscillating ±15 hPa/row jump into a smooth
    ramp. Applies only when ``resolution_match_rows`` is not None (the
    default), consistent with notes 7/9.

Grids follow the P1 convention: 2-D fields are ``(H, W)``, rows run
north-to-south on cell centres, zonal means are ``(H,)``.  Planetary
constants (gravity, radius, rotation rate, reference density/pressure/
temperature) are explicit inputs — never hardcoded Earth literals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sesam_reference import value as _param
from sim_grid import _coarsen as _block_mean_coarsen

# ---------------------------------------------------------------------------
# A2 parameter registry (single source: sesam_reference.py)
# ---------------------------------------------------------------------------


def _a2_defaults() -> dict[str, float]:
    """All A2 constants from the published Table A2 pack."""
    return {
        "C1_cell": _param("A2_dynamics", "C1_cell"),
        "C2_cell": _param("A2_dynamics", "C2_cell"),
        "C3_cell": _param("A2_dynamics", "C3_cell"),
        "c1mmc": _param("A2_dynamics", "c1mmc"),
        "c2mmc": _param("A2_dynamics", "c2mmc"),
        "c3mmc": _param("A2_dynamics", "c3mmc"),
        "c4mmc": _param("A2_dynamics", "c4mmc"),
        "c5mmc": _param("A2_dynamics", "c5mmc"),
        "H0_slp": _param("A2_dynamics", "H0_slp"),
        "tau_e": _param("A2_dynamics", "tau_e"),
    }


_DEFAULT_RD = 287.0

# Safeguards read from the reference implementation (module docstring notes
# 2, 4 and 5); each is a documented numerical/physical guard, not a fit.
_HADLEY_SCALE_MIN = 0.5
_HADLEY_SCALE_MAX = 1.5
_T_TRP_FLOOR_OFFSET_K = 50.0
_U500_FLOOR_M_S = 0.1

# Paper-text constants.
_F_AGEO_FLOOR_S = 1.0e-5  # |f| floor in the ageostrophic relations (paper A2 text)
_SEA_LEVEL_LAPSE_K_M = 6.5e-3  # (A37) prose: 6.5 K km^-1 sea-level reduction
_C_HRS = 0.7  # ftrop coordinate factor (module docstring note 7)

# Numerical guard on |sinα·cosα| in the (A29) denominator: below this the
# integrand is dominated by placeholder noise (physical values are ~0.4 for
# α ≈ 25–35°).
_SIN_COS_ALPHA_FLOOR = 0.05

# Module docstring note 9: SESAM's validated native grid is 5°x5° (180°/5°
# rows). Not an Earth literal -- it is the paper's own resolution, applied to
# whatever grid the caller passes.
_RESOLUTION_MATCH_TARGET_ROWS = 36

_SECONDS_PER_DAY = 86400.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _latitude_rad(h: int) -> np.ndarray:
    """Signed cell-centre latitude in radians, north-to-south rows."""
    return (0.5 - (np.arange(h, dtype=np.float64) + 0.5) / h) * np.pi


def _zonal_mean(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("expected a 2-D (H, W) field")
    return np.mean(arr, axis=1)


def _cos_weights(latitude_rad: np.ndarray) -> np.ndarray:
    return np.cos(np.asarray(latitude_rad, dtype=np.float64))


def _cos_weighted_mean(values: np.ndarray, latitude_rad: np.ndarray) -> float:
    w = _cos_weights(latitude_rad)
    return float(np.sum(values * w) / np.sum(w))


def _as_zonal(name: str, value, h: int) -> np.ndarray:
    """Accept a (H,) zonal field, a (H, W) field (zonal-meaned), or a scalar."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(h, float(arr))
    if arr.ndim == 1 and arr.shape == (h,):
        return arr
    if arr.ndim == 2 and arr.shape[1] != 0 and arr.shape[0] == h:
        return np.mean(arr, axis=1)
    raise ValueError(f"{name} must be a scalar, (H,) or (H, W) with H={h}; got {arr.shape}")


def _interp_at_latitudes(latitude_rad: np.ndarray, values: np.ndarray, targets) -> np.ndarray:
    """Linear interpolation of a north-to-south row profile to target latitudes."""
    lat = np.asarray(latitude_rad, dtype=np.float64)
    val = np.asarray(values, dtype=np.float64)
    # np.interp requires ascending x; rows are north-to-south (descending lat).
    return np.interp(np.asarray(targets, dtype=np.float64), lat[::-1], val[::-1])


# ---------------------------------------------------------------------------
# Resolution matching (module docstring note 9) -- P2 exit-gate fix
# ---------------------------------------------------------------------------


def _coarse_block_size(h: int, target_rows: int) -> int:
    if target_rows <= 0:
        raise ValueError("target_rows must be positive")
    return max(1, round(h / float(target_rows)))


def _bilinear_refine(coarse: np.ndarray, h: int, w: int) -> np.ndarray:
    """Separable bilinear regrid of a coarse (Hc, Wc) field back to (h, w).

    Latitude (rows) is edge-clamped (no wrap at the poles); longitude
    (columns) wraps periodically, matching the grid's spherical topology.
    Both axes interpolate on normalized cell-centre coordinates so the result
    is independent of the absolute grid size.
    """
    hc, wc = coarse.shape
    if (hc, wc) == (h, w):
        return coarse.astype(np.float64, copy=True)

    src_row = (np.arange(hc, dtype=np.float64) + 0.5) / hc
    dst_row = (np.arange(h, dtype=np.float64) + 0.5) / h
    row_stage = np.empty((h, wc), dtype=np.float64)
    for j in range(wc):
        row_stage[:, j] = np.interp(dst_row, src_row, coarse[:, j])

    src_col = (np.arange(wc, dtype=np.float64) + 0.5) / wc
    dst_col = (np.arange(w, dtype=np.float64) + 0.5) / w
    ext_col = np.concatenate(([src_col[-1] - 1.0], src_col, [src_col[0] + 1.0]))
    ext_vals = np.empty((h, wc + 2), dtype=np.float64)
    ext_vals[:, 1:-1] = row_stage
    ext_vals[:, 0] = row_stage[:, -1]
    ext_vals[:, -1] = row_stage[:, 0]
    out = np.empty((h, w), dtype=np.float64)
    for i in range(h):
        out[i, :] = np.interp(dst_col, ext_col, ext_vals[i, :])
    return out


def resolution_matched_field(
    field_2d: np.ndarray,
    *,
    target_rows: int = _RESOLUTION_MATCH_TARGET_ROWS,
) -> np.ndarray:
    """Low-pass a 2-D field to SESAM's native ~5° validated resolution.

    Box-averages to a coarse grid with roughly ``target_rows`` latitude rows
    (derived from the caller's own grid height -- never a hardcoded Earth
    literal) and bilinearly regrids back to the input's shape (module
    docstring note 9). A no-op (returns a copy) when the input already has
    ``target_rows`` rows or fewer.  Pure and deterministic; carries no
    reference-implementation code (module docstring note 6/9 distinction).
    """
    arr = np.asarray(field_2d, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("field_2d must be a 2-D (H, W) field")
    h, w = arr.shape
    bs = _coarse_block_size(h, target_rows)
    if bs <= 1:
        return arr.copy()
    hc = max(1, -(-h // bs))  # ceil division: _coarsen requires Hc*bs >= h
    wc = max(1, -(-w // bs))
    coarse = _block_mean_coarsen(arr.astype(np.float32), hc, wc, bs).astype(np.float64)
    return _bilinear_refine(coarse, h, w)


def resolution_matched_profile(
    profile_1d: np.ndarray,
    *,
    target_rows: int = _RESOLUTION_MATCH_TARGET_ROWS,
) -> np.ndarray:
    """1-D latitude analogue of :func:`resolution_matched_field` (note 10).

    Box-averages a ``(H,)`` zonal profile to ``target_rows`` latitude bands
    and linearly regrids back (edge-clamped, no longitude to wrap). Used for
    ``u500`` -- see module docstring note 10 for why a per-row-independent
    Charney-Eliassen solve needs a smooth ``u500(ϕ)`` input regardless of
    how smooth the 2-D fields already are.
    """
    arr = np.asarray(profile_1d, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("profile_1d must be a 1-D (H,) array")
    h = arr.shape[0]
    bs = _coarse_block_size(h, target_rows)
    if bs <= 1:
        return arr.copy()
    hc = max(1, -(-h // bs))
    padded = np.pad(arr, (0, hc * bs - h), mode="edge")
    coarse = padded.reshape(hc, bs).mean(axis=1)
    src = (np.arange(hc, dtype=np.float64) + 0.5) / hc
    dst = (np.arange(h, dtype=np.float64) + 0.5) / h
    return np.interp(dst, src, coarse)


# ---------------------------------------------------------------------------
# (A32) ITCZ position, (A33) Hadley width scale, (A31) cell coordinate
# ---------------------------------------------------------------------------


def itcz_latitude(
    t_nh_k: np.ndarray | float,
    t_sh_k: np.ndarray | float,
    a2: dict[str, float] | None = None,
) -> np.ndarray:
    """(A32) ``φITCZ = c2mmc·(T_NH − T_SH)`` in radians.

    ``T_NH``/``T_SH`` are the cosine-weighted hemispheric-mean zonal-mean
    sea-level temperatures (see :func:`hemispheric_mean_temperatures`).
    """
    params = a2 or _a2_defaults()
    t_nh = np.asarray(t_nh_k, dtype=np.float64)
    t_sh = np.asarray(t_sh_k, dtype=np.float64)
    return params["c2mmc"] * (t_nh - t_sh)


def hadley_width_scale(
    tropical_mean_temp_k: np.ndarray | float,
    a2: dict[str, float] | None = None,
    *,
    scale_limits: tuple[float, float] = (_HADLEY_SCALE_MIN, _HADLEY_SCALE_MAX),
) -> np.ndarray:
    """(A33) ``Dhad = c3mmc/(T_trp − c4mmc)``, clamped to ``scale_limits``.

    The fraction form is verified against the article MathML (module
    docstring note 2).  ``T_trp`` is floored at ``c4mmc + 50 K`` before the
    division (reference-implementation safeguard against a non-positive
    denominator); the clamp bounds the scale to [0.5, 1.5].
    """
    params = a2 or _a2_defaults()
    t_trp = np.asarray(tropical_mean_temp_k, dtype=np.float64)
    t_eff = np.maximum(t_trp, params["c4mmc"] + _T_TRP_FLOOR_OFFSET_K)
    scale = params["c3mmc"] / (t_eff - params["c4mmc"])
    lo, hi = scale_limits
    return np.clip(scale, lo, hi)


def cell_coordinate(
    latitude_rad: np.ndarray,
    itcz_latitude_rad: np.ndarray | float,
    hadley_width_scale_value: np.ndarray | float,
    a2: dict[str, float] | None = None,
) -> np.ndarray:
    """(A31) ``φ = 6·Dhad·(ϕ − φITCZ/(c1mmc·(ϕ − φITCZ)² + 1))``.

    The rational factor applies to ``φITCZ`` only (module docstring note 1).
    The coordinate spans [−3π·Dhad·…, +3π·Dhad·…]: 0 at the ITCZ, ±π at the
    Hadley–Ferrel edges, ±2π at the Ferrel–polar edges, ±3π at the poles.
    """
    params = a2 or _a2_defaults()
    phi = np.asarray(latitude_rad, dtype=np.float64)
    itcz = np.asarray(itcz_latitude_rad, dtype=np.float64)
    dhad = np.asarray(hadley_width_scale_value, dtype=np.float64)
    delta = phi - itcz
    return 6.0 * dhad * (phi - itcz / (params["c1mmc"] * delta**2 + 1.0))


def hadley_geometry(
    latitude_rad: np.ndarray,
    phi_mmc_rad: np.ndarray,
) -> dict[str, float]:
    """Hadley-cell edges and geometry from the (A31) cell coordinate.

    Returns the northern/southern Hadley–Ferrel boundary latitudes (where the
    coordinate crosses ±π, linearly interpolated between rows), the cell
    centre ``hadley_centre_rad`` (mean of the two edges; the reference
    implementation's ``had_fi``) and the full width ``hadley_width_rad``
    (northern minus southern edge).  Raises ``ValueError`` if either crossing
    is not bracketed (a climate too distorted for the cell decomposition).
    """
    lat = np.asarray(latitude_rad, dtype=np.float64)
    phi = np.asarray(phi_mmc_rad, dtype=np.float64)
    if lat.shape != phi.shape or lat.ndim != 1:
        raise ValueError("latitude_rad and phi_mmc_rad must share a 1-D shape")

    def crossing(target: float) -> float:
        d = phi - target
        sign_change = np.flatnonzero(d[:-1] * d[1:] < 0.0)
        if sign_change.size == 0:
            raise ValueError(
                f"cell coordinate does not cross {target / np.pi:.3g}·π; "
                "the Hadley-cell boundary is not bracketed"
            )
        j = sign_change[0]
        frac = d[j] / (d[j] - d[j + 1])
        return float(lat[j] + frac * (lat[j + 1] - lat[j]))

    edge_nh = crossing(np.pi)
    edge_sh = crossing(-np.pi)
    return {
        "hadley_edge_nh_rad": edge_nh,
        "hadley_edge_sh_rad": edge_sh,
        "hadley_centre_rad": 0.5 * (edge_nh + edge_sh),
        "hadley_width_rad": edge_nh - edge_sh,
    }


def tropical_weight_from_hadley(
    latitude_rad: np.ndarray,
    hadley_centre_rad: np.ndarray | float,
    hadley_width_rad: np.ndarray | float,
    *,
    c_hrs: float = _C_HRS,
) -> np.ndarray:
    """(A14) tropical weight ``1 − sin⁸(fi)`` on the Hadley-scaled latitude.

    ``fi = clamp(c_hrs·(ϕ − had_fi)/(0.5·had_width), ±π/2)`` — the meridional
    coordinate the (A14) RH scale height actually uses (module docstring note
    7), replacing the latitude placeholder P1 documented.  Feed the result to
    ``sesam_vertical.tropical_weight``'s callers (or use it directly as
    ``tropical_weight_field``); it is 1 at the Hadley centre and 0 poleward
    of ±π/2 in the scaled coordinate.
    """
    lat = np.asarray(latitude_rad, dtype=np.float64)
    centre = np.asarray(hadley_centre_rad, dtype=np.float64)
    width = np.asarray(hadley_width_rad, dtype=np.float64)
    fi = c_hrs * (lat - centre) / np.maximum(0.5 * width, 1e-6)
    fi = np.clip(fi, -0.5 * np.pi, 0.5 * np.pi)
    return 1.0 - np.sin(fi) ** 8


# ---------------------------------------------------------------------------
# (A34) cell temperature gradients, (A35) topography factor
# ---------------------------------------------------------------------------


def hemispheric_mean_temperatures(
    latitude_rad: np.ndarray,
    zonal_mean_temp_k: np.ndarray,
) -> tuple[float, float]:
    """Cosine-weighted hemispheric means ``(T_NH, T_SH)`` of a zonal profile."""
    lat = np.asarray(latitude_rad, dtype=np.float64)
    t = np.asarray(zonal_mean_temp_k, dtype=np.float64)
    if lat.shape != t.shape or lat.ndim != 1:
        raise ValueError("latitude_rad and zonal_mean_temp_k must share a 1-D shape")
    w = _cos_weights(lat)
    nh = lat > 0.0
    sh = lat < 0.0
    t_nh = float(np.sum(t[nh] * w[nh]) / np.sum(w[nh]))
    t_sh = float(np.sum(t[sh] * w[sh]) / np.sum(w[sh]))
    return t_nh, t_sh


def cell_temperature_gradients(
    latitude_rad: np.ndarray,
    zonal_mean_temp_k: np.ndarray,
) -> dict[str, np.ndarray]:
    """(A34) Hadley/Ferrel/polar temperature gradients, both hemispheres.

    Evaluates the zonal-mean sea-level temperature profile at the fixed
    latitudes ±π/6, ±π/3, ±π/2 (linear interpolation) and returns
    ``{"nh": (ΔT₁, ΔT₂, ΔT₃), "sh": (...), "t_max_k": …}`` with the
    circulation-correct ordering (module docstring note 3):

    - Hadley ``ΔT₁ = max(0, max(T̄) − T̄(±π/6))`` where ``max(T̄)`` is the
      global zonal-mean maximum as printed (tropical on Earth-like
      climates); the reference implementation instead takes the maximum over
      the tropical band only, where the ``max(0, ·)`` floor is meaningful —
      with the paper's global maximum the floor can never bind and is kept
      as documented insurance;
    - Ferrel ``ΔT₂ = T̄(±π/6) − T̄(±π/3)``;
    - polar ``ΔT₃ = T̄(±π/3) − T̄(±π/2)`` (the pole value is the northernmost /
      southernmost row).
    """
    lat = np.asarray(latitude_rad, dtype=np.float64)
    t = np.asarray(zonal_mean_temp_k, dtype=np.float64)
    if lat.shape != t.shape or lat.ndim != 1:
        raise ValueError("latitude_rad and zonal_mean_temp_k must share a 1-D shape")
    t_max = float(np.max(t))
    t_n = _interp_at_latitudes(lat, t, [np.pi / 6.0, np.pi / 3.0, np.pi / 2.0])
    t_s = _interp_at_latitudes(lat, t, [-np.pi / 6.0, -np.pi / 3.0, -np.pi / 2.0])
    dt_nh = np.array([
        max(0.0, t_max - t_n[0]),
        t_n[0] - t_n[1],
        t_n[1] - t_n[2],
    ])
    dt_sh = np.array([
        max(0.0, t_max - t_s[0]),
        t_s[0] - t_s[1],
        t_s[1] - t_s[2],
    ])
    return {"nh": dt_nh, "sh": dt_sh, "t_max_k": t_max}


def topography_factor(
    zonal_mean_elevation_m: np.ndarray,
    a2: dict[str, float] | None = None,
) -> np.ndarray:
    """(A35) ``Fz = 1 − z̄s/c5mmc``, clamped to [0, 1].

    The clamp (reference-implementation safeguard) keeps high plateaus from
    reversing the cell forcing and below-sea-level basins from amplifying it.
    """
    params = a2 or _a2_defaults()
    zs = np.asarray(zonal_mean_elevation_m, dtype=np.float64)
    return np.clip(1.0 - zs / params["c5mmc"], 0.0, 1.0)


# ---------------------------------------------------------------------------
# (A30) mean meridional overturning wind, (A29) zonal SLP
# ---------------------------------------------------------------------------


def mean_overturning_wind(
    phi_mmc_rad: np.ndarray,
    dt_nh: np.ndarray,
    dt_sh: np.ndarray,
    fz: np.ndarray,
    a2: dict[str, float] | None = None,
) -> np.ndarray:
    """(A30) ``v̄a(φ) = −Cᵢ·ΔTᵢʲ·Fz(φ)·sinφ`` on the (A31) cell coordinate.

    ``dt_nh``/``dt_sh`` are the per-cell gradients ``(ΔT₁, ΔT₂, ΔT₃)`` from
    :func:`cell_temperature_gradients`; the hemisphere for each row is chosen
    by the sign of the cell coordinate.  Cells are the coordinate bands
    ``(i−1)π < |φ| < iπ``; outside |φ| = 3π (possible under the widest
    Hadley-scale clamp) the parameterization is inactive and ``v̄a = 0``.
    """
    params = a2 or _a2_defaults()
    phi = np.asarray(phi_mmc_rad, dtype=np.float64)
    fz_arr = np.asarray(fz, dtype=np.float64)
    if fz_arr.shape != phi.shape:
        raise ValueError("fz and phi_mmc_rad must share a shape")
    dt_nh = np.asarray(dt_nh, dtype=np.float64)
    dt_sh = np.asarray(dt_sh, dtype=np.float64)
    if dt_nh.shape != (3,) or dt_sh.shape != (3,):
        raise ValueError("dt_nh and dt_sh must be (3,) per-cell gradients")
    cell_strength = np.array(
        [params["C1_cell"], params["C2_cell"], params["C3_cell"]]
    )

    abs_phi = np.abs(phi)
    cell = np.minimum((abs_phi / np.pi).astype(np.int64), 2)
    active = abs_phi < 3.0 * np.pi
    c_i = cell_strength[cell]
    dt_i = np.where(phi >= 0.0, dt_nh[cell], dt_sh[cell])
    va = -c_i * dt_i * fz_arr * np.sin(phi)
    return np.where(active, va, 0.0)


def zonal_slp_anomaly(
    latitude_rad: np.ndarray,
    overturning_wind_m_s: np.ndarray,
    sin_cos_alpha_bar: np.ndarray | float,
    *,
    radius_m: float,
    omega: float,
    rho0_kg_m3: float,
    coriolis_floor_s: float = _F_AGEO_FLOOR_S,
    sin_cos_alpha_floor: float = _SIN_COS_ALPHA_FLOOR,
) -> np.ndarray:
    """(A29) integrate ``∂p̄sl/∂ϕ = −v̄a·ρ0·|f|·Re/(sinα·cosᾱ)`` pole to pole.

    Convention (module docstring note 5): ``|f| = 2Ω·|sinϕ|`` floored at
    ``coriolis_floor_s`` (the paper's 1e-5 s⁻¹ ageostrophic floor, so the
    ITCZ trough stays finite at the equator) and ``sin_cos_alpha_bar`` is a
    positive *magnitude* (a scalar input is broadcast as a uniform
    magnitude).  The trapezoidal integral is returned as an anomaly with
    zero cosine-weighted global mean (atmospheric-mass conservation; the
    absolute ``p0`` offset is restored at assembly).
    """
    lat = np.asarray(latitude_rad, dtype=np.float64)
    va = np.asarray(overturning_wind_m_s, dtype=np.float64)
    if lat.shape != va.shape or lat.ndim != 1:
        raise ValueError("latitude_rad and overturning_wind_m_s must share a 1-D shape")
    scab = np.asarray(sin_cos_alpha_bar, dtype=np.float64)
    if scab.ndim == 0:
        scab = np.full(lat.shape, float(scab))
    if scab.shape != lat.shape:
        raise ValueError("sin_cos_alpha_bar must be a scalar or a (H,) zonal field")

    f = np.maximum(2.0 * float(omega) * np.abs(np.sin(lat)), coriolis_floor_s)
    scab = np.maximum(np.abs(scab), sin_cos_alpha_floor)

    dp_dphi = -va * float(rho0_kg_m3) * f * float(radius_m) / scab
    dphi = np.diff(lat)  # negative: rows run north-to-south
    increments = 0.5 * (dp_dphi[:-1] + dp_dphi[1:]) * dphi
    anomaly = np.concatenate(([0.0], np.cumsum(increments)))
    return anomaly - _cos_weighted_mean(anomaly, lat)


# ---------------------------------------------------------------------------
# (A37) azonal thermal SLP
# ---------------------------------------------------------------------------


def sea_level_temperature(
    skin_temp_k: np.ndarray,
    surface_elevation_m: np.ndarray,
    *,
    lapse_k_m: float = _SEA_LEVEL_LAPSE_K_M,
) -> np.ndarray:
    """Skin temperature reduced to sea level, ``T_sl = T_skin + Γ·zs``.

    The (A37) prose's constant 6.5 K km⁻¹ reduction.  ``zs`` may be negative
    (ocean basins); the reduction is linear in ``zs``.
    """
    t = np.asarray(skin_temp_k, dtype=np.float64)
    zs = np.asarray(surface_elevation_m, dtype=np.float64)
    if t.shape != zs.shape:
        raise ValueError("skin_temp_k and surface_elevation_m must share a shape")
    return t + lapse_k_m * zs


def thermal_azonal_slp(
    sea_level_temp_k: np.ndarray,
    *,
    gravity: float,
    p0_pa: float,
    reference_temp_k: float,
    a2: dict[str, float] | None = None,
    gas_constant: float = _DEFAULT_RD,
) -> np.ndarray:
    """(A37) ``p*sl,T = −(g·p0·H0)/(2·Rd·T0²)·T*sl``.

    ``T*sl`` is the azonal anomaly of the sea-level-reduced skin temperature
    (zonal mean removed per row — this is the missing monsoon/maritime
    driver: warm surface → thermal low).  The result's zonal mean is
    re-zeroed numerically.
    """
    params = a2 or _a2_defaults()
    tsl = np.asarray(sea_level_temp_k, dtype=np.float64)
    if tsl.ndim != 2:
        raise ValueError("sea_level_temp_k must be a 2-D (H, W) field")
    tsl_star = tsl - np.mean(tsl, axis=1, keepdims=True)
    coefficient = (
        float(gravity) * float(p0_pa) * params["H0_slp"]
        / (2.0 * float(gas_constant) * float(reference_temp_k) ** 2)
    )
    out = -coefficient * tsl_star
    return out - np.mean(out, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# (A38)-(A39) Charney–Eliassen topographic stationary waves
# ---------------------------------------------------------------------------


def charney_eliassen_slp(
    topography_m: np.ndarray,
    latitude_rad: np.ndarray,
    u500_m_s: np.ndarray,
    tropopause_height_m: np.ndarray,
    *,
    radius_m: float,
    omega: float,
    rho0_kg_m3: float,
    p0_pa: float,
    a2: dict[str, float] | None = None,
    meridional_half_wavelength_m: float | None = None,
    p500_pa: float = 50000.0,
) -> np.ndarray:
    """(A38)–(A39) orographic azonal SLP from forced topographic Rossby waves.

    Solves the linearized barotropic vorticity equation
    ``u·∂ζ/∂λ + β·v + ζ/τe = −(f/H_T)·0.4·u·∂zs/∂λ`` independently per
    latitude row by zonal Fourier expansion of the topography.  With
    ``k₀ = 1/(Re·cosϕ)``, zonal wavenumbers ``n·k₀``, meridional wavenumber
    ``m = π/L`` (paper: ``L`` = 35° latitudinal half-wavelength, the default),
    and ``Kn² = (n·k₀)² + m²``, the streamfunction response is

    ``Ψ̂ₙ = (f/H_T)·0.4·u·ẑsₙ / (u·Kn² − β − i·Kn²/(τe·n·k₀))``  (n ≥ 1)

    derived from the printed (A38) (module docstring note 9; the paper gives
    the PDE and says it is solved by per-row Fourier expansion, not a closed
    form) and ``p*sl,O = ρ(500 hPa)·|f|·Ψ`` (A39), with
    ``ρ(500) = ρ0·p500/p0`` in the exponential reference atmosphere.  ``u``
    is the zonal-mean 500 hPa zonal wind floored at +0.1 m s⁻¹
    (reference-implementation safeguard: stationary-wave resonance is only
    supported for westerly flow); ``H_T`` is the zonal-mean tropopause
    height.  The zonal-mean (n = 0) response is zero by construction, so the
    result is purely azonal.  The ``τe`` term is the only real damping in the
    denominator away from ``u·Kn² = β``; get its ``kzn`` power wrong (module
    docstring note 9) and the response is unbounded at whatever integer n
    happens to sit near that resonance for the given ``u``, ``β``.
    """
    params = a2 or _a2_defaults()
    zs = np.asarray(topography_m, dtype=np.float64)
    if zs.ndim != 2:
        raise ValueError("topography_m must be a 2-D (H, W) field")
    h, w = zs.shape
    lat = np.asarray(latitude_rad, dtype=np.float64)
    if lat.shape != (h,):
        raise ValueError("latitude_rad must be (H,) matching topography rows")
    u500 = _as_zonal("u500_m_s", u500_m_s, h)
    htrop = _as_zonal("tropopause_height_m", tropopause_height_m, h)

    tau_s = params["tau_e"] * _SECONDS_PER_DAY
    if meridional_half_wavelength_m is None:
        meridional_half_wavelength_m = float(radius_m) * np.radians(35.0)
    m_wave = np.pi / meridional_half_wavelength_m
    u = np.maximum(u500, _U500_FLOOR_M_S)
    rho500 = float(rho0_kg_m3) * (p500_pa / float(p0_pa))

    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    k0 = 1.0 / (float(radius_m) * cos_lat)
    beta = 2.0 * float(omega) * cos_lat / float(radius_m)
    f = 2.0 * float(omega) * sin_lat

    n = np.arange(w // 2 + 1, dtype=np.float64)
    out = np.zeros((h, w), dtype=np.float64)
    for j in range(h):
        kzn = n * k0[j]
        kn2 = kzn**2 + m_wave**2
        denom_real = u[j] * kn2 - beta[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            denom_imag = kn2 / (tau_s * kzn)
        # n = 0 has no azonal content; zero its response (also removes 0/0).
        denom_imag[0] = 1.0
        response = np.where(
            n > 0,
            (f[j] / htrop[j]) * 0.4 * u[j] / (denom_real - 1j * denom_imag),
            0.0,
        )
        psi = np.fft.irfft(response * np.fft.rfft(zs[j]), w)
        psi -= np.mean(psi)
        out[j] = rho500 * abs(f[j]) * psi
    return out


# ---------------------------------------------------------------------------
# (A28)/(A36) assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SesamSlp:
    """Reconstructed sea-level pressure and its (A2) components/diagnostics.

    ``slp_pa`` is the full (A28) field ``p̄sl + p*sl,T + p*sl,O`` with the
    cosine-weighted global mean restored to ``p0``.  Zonal components are
    ``(H,)``; azonal components and the full field are ``(H, W)``.
    """

    slp_pa: np.ndarray
    zonal_slp_anomaly_pa: np.ndarray
    thermal_azonal_slp_pa: np.ndarray
    orographic_azonal_slp_pa: np.ndarray
    overturning_wind_m_s: np.ndarray
    cell_coordinate_rad: np.ndarray
    itcz_latitude_rad: float
    hadley_edge_nh_rad: float
    hadley_edge_sh_rad: float
    hadley_centre_rad: float
    hadley_width_rad: float
    hadley_width_scale: float
    cell_gradients_nh_k: np.ndarray
    cell_gradients_sh_k: np.ndarray
    t_nh_k: float
    t_sh_k: float
    t_trp_k: float
    p0_pa: float


def compute_slp(
    *,
    skin_temp_k: np.ndarray,
    surface_elevation_m: np.ndarray,
    sin_cos_alpha_bar: np.ndarray | float,
    gravity: float,
    radius_m: float,
    omega: float,
    p0_pa: float,
    reference_temp_k: float,
    rho0_kg_m3: float | None = None,
    gas_constant: float = _DEFAULT_RD,
    u500_m_s: np.ndarray | None = None,
    tropopause_height_m: np.ndarray | None = None,
    a2: dict[str, float] | None = None,
    resolution_match_rows: int | None = _RESOLUTION_MATCH_TARGET_ROWS,
) -> SesamSlp:
    """Assemble the full (A28) sea-level pressure from 2-D input fields.

    Inputs are the skin temperature and surface elevation ``(H, W)`` plus the
    planet constants; ``sin_cos_alpha_bar`` is required (module docstring
    note 5).  When ``u500_m_s`` (zonal-mean 500 hPa wind) and
    ``tropopause_height_m`` are supplied the Charney–Eliassen term (A38–A39)
    is included; otherwise it is zero (documented skip, not a placeholder).

    ``rho0_kg_m3`` defaults to ``p0/(Rd·reference_temp_k)`` — the ideal-gas
    surface density of the reference atmosphere, not a tuned constant.

    The tropical temperature ``T_trp`` entering (A33) needs the Hadley edges,
    which need ``Dhad``, which needs ``T_trp``: this is closed by two fixed
    passes (provisional ±π/6 tropical band, then the band between the
    detected edges).  The reference implementation uses the previous day's
    edges instead; the two agree at equilibrium.

    ``resolution_match_rows`` (module docstring note 7) applies
    :func:`resolution_matched_field` to the 2-D inputs of the azonal thermal
    (A37) and orographic (A38–A39) terms before evaluating them, and (module
    docstring note 10) :func:`resolution_matched_profile` to ``u500_m_s``
    before it drives the per-row-independent (A38) solve, matching all three
    to SESAM's validated ~5° grid; pass ``None`` to evaluate the raw
    full-resolution/unsmoothed inputs (the pre-2026-08-17 behaviour, kept for
    A/B comparison). The zonal cell-physics path is unaffected either way.
    """
    params = a2 or _a2_defaults()
    t_skin = np.asarray(skin_temp_k, dtype=np.float64)
    zs = np.asarray(surface_elevation_m, dtype=np.float64)
    if t_skin.shape != zs.shape or t_skin.ndim != 2:
        raise ValueError("skin_temp_k and surface_elevation_m must share a 2-D shape")
    h, w = t_skin.shape
    lat = _latitude_rad(h)
    if rho0_kg_m3 is None:
        rho0_kg_m3 = float(p0_pa) / (float(gas_constant) * float(reference_temp_k))

    tsl = sea_level_temperature(t_skin, zs)
    tsl_z = _zonal_mean(tsl)
    zs_z = _zonal_mean(zs)

    t_nh, t_sh = hemispheric_mean_temperatures(lat, tsl_z)
    phi_itcz = float(itcz_latitude(t_nh, t_sh, params))

    # Two-pass tropical temperature / Hadley geometry closure (docstring).
    edges: dict[str, float] | None = None
    t_trp = 0.0
    dhad = 0.0
    phi_mmc = np.zeros(h)
    for pass_index in range(2):
        if edges is None:
            band = np.abs(lat) <= np.pi / 6.0
        else:
            band = (lat <= edges["hadley_edge_nh_rad"]) & (
                lat >= edges["hadley_edge_sh_rad"]
            )
        if not np.any(band):
            raise ValueError("tropical band is empty; cannot close the Hadley geometry")
        w_cos = _cos_weights(lat[band])
        t_trp = float(np.sum(tsl_z[band] * w_cos) / np.sum(w_cos))
        dhad = float(hadley_width_scale(t_trp, params))
        phi_mmc = cell_coordinate(lat, phi_itcz, dhad, params)
        edges = hadley_geometry(lat, phi_mmc)

    gradients = cell_temperature_gradients(lat, tsl_z)
    fz = topography_factor(zs_z, params)
    va = mean_overturning_wind(phi_mmc, gradients["nh"], gradients["sh"], fz, params)

    # Scalar sin_cos_alpha_bar reaches zonal_slp_anomaly as a uniform
    # magnitude; arrays are zonal-meaned (module docstring note 5).
    if np.asarray(sin_cos_alpha_bar).ndim == 0:
        scab_input: np.ndarray | float = float(np.asarray(sin_cos_alpha_bar))
    else:
        scab_input = _as_zonal("sin_cos_alpha_bar", sin_cos_alpha_bar, h)
    p_zonal = zonal_slp_anomaly(
        lat,
        va,
        scab_input,
        radius_m=radius_m,
        omega=omega,
        rho0_kg_m3=rho0_kg_m3,
    )

    tsl_for_thermal = (
        resolution_matched_field(tsl, target_rows=resolution_match_rows)
        if resolution_match_rows is not None
        else tsl
    )
    p_thermal = thermal_azonal_slp(
        tsl_for_thermal,
        gravity=gravity,
        p0_pa=p0_pa,
        reference_temp_k=reference_temp_k,
        a2=params,
        gas_constant=gas_constant,
    )

    if u500_m_s is not None:
        if tropopause_height_m is None:
            raise ValueError(
                "tropopause_height_m is required when u500_m_s enables the "
                "Charney–Eliassen term (no fabricated tropopause)"
            )
        zs_for_oro = (
            resolution_matched_field(zs, target_rows=resolution_match_rows)
            if resolution_match_rows is not None
            else zs
        )
        u500_for_oro = (
            resolution_matched_profile(
                _as_zonal("u500_m_s", u500_m_s, h), target_rows=resolution_match_rows
            )
            if resolution_match_rows is not None
            else u500_m_s
        )
        p_oro = charney_eliassen_slp(
            zs_for_oro,
            lat,
            u500_for_oro,
            tropopause_height_m,
            radius_m=radius_m,
            omega=omega,
            rho0_kg_m3=rho0_kg_m3,
            p0_pa=p0_pa,
            a2=params,
        )
    else:
        p_oro = np.zeros((h, w), dtype=np.float64)

    slp = p_zonal[:, None] + p_thermal + p_oro
    # Atmospheric-mass restoration (reference implementation): the global
    # cosine-weighted mean of the full SLP is the reference pressure p0.
    cos_2d = np.broadcast_to(_cos_weights(lat)[:, None], (h, w))
    global_mean = float(np.sum(slp * cos_2d) / np.sum(cos_2d))
    slp = slp + (float(p0_pa) - global_mean)

    return SesamSlp(
        slp_pa=slp,
        zonal_slp_anomaly_pa=p_zonal,
        thermal_azonal_slp_pa=p_thermal,
        orographic_azonal_slp_pa=p_oro,
        overturning_wind_m_s=va,
        cell_coordinate_rad=phi_mmc,
        itcz_latitude_rad=phi_itcz,
        hadley_edge_nh_rad=edges["hadley_edge_nh_rad"],
        hadley_edge_sh_rad=edges["hadley_edge_sh_rad"],
        hadley_centre_rad=edges["hadley_centre_rad"],
        hadley_width_rad=edges["hadley_width_rad"],
        hadley_width_scale=dhad,
        cell_gradients_nh_k=gradients["nh"],
        cell_gradients_sh_k=gradients["sh"],
        t_nh_k=t_nh,
        t_sh_k=t_sh,
        t_trp_k=t_trp,
        p0_pa=float(p0_pa),
    )


# ---------------------------------------------------------------------------
# Diagnostics: zonal-profile scorecard
# ---------------------------------------------------------------------------


def zonal_slp_extrema(
    latitude_rad: np.ndarray,
    zonal_slp_anomaly_pa: np.ndarray,
    itcz_latitude_rad: float,
) -> dict[str, dict[str, float]]:
    """Zonal-mean SLP profile extrema — the SLP-side jet/Hadley scorecard.

    For each hemisphere: the subtropical-high position/strength (maximum of
    the zonal anomaly between the ITCZ and 45°), the subpolar-low
    position/strength (minimum between 45° and 75°), and the ITCZ-trough
    value (minimum within ±10° of the ITCZ).  These are diagnostics, not
    paper equations; bands are in radians of latitude.
    """
    lat = np.asarray(latitude_rad, dtype=np.float64)
    p = np.asarray(zonal_slp_anomaly_pa, dtype=np.float64)
    if lat.shape != p.shape or lat.ndim != 1:
        raise ValueError("latitude_rad and zonal_slp_anomaly_pa must share a 1-D shape")

    def band_extremum(
        lo_deg: float, hi_deg: float, want_max: bool, sign: float
    ) -> dict[str, float]:
        mask = (
            (np.abs(lat) >= np.radians(lo_deg))
            & (np.abs(lat) <= np.radians(hi_deg))
            & (lat * sign > 0.0)
        )
        if not np.any(mask):
            return {"latitude_deg": float("nan"), "anomaly_pa": float("nan")}
        rows = np.flatnonzero(mask)
        pick = rows[np.argmax(p[rows])] if want_max else rows[np.argmin(p[rows])]
        return {"latitude_deg": float(np.degrees(lat[pick])), "anomaly_pa": float(p[pick])}

    def itcz_trough() -> dict[str, float]:
        band = np.abs(lat - itcz_latitude_rad) <= np.radians(10.0)
        if not np.any(band):
            return {"latitude_deg": float("nan"), "anomaly_pa": float("nan")}
        rows = np.flatnonzero(band)
        pick = rows[np.argmin(p[rows])]
        return {"latitude_deg": float(np.degrees(lat[pick])), "anomaly_pa": float(p[pick])}

    return {
        "nh": {
            "subtropical_high": band_extremum(5.0, 45.0, True, 1.0),
            "subpolar_low": band_extremum(45.0, 75.0, False, 1.0),
        },
        "sh": {
            "subtropical_high": band_extremum(5.0, 45.0, True, -1.0),
            "subpolar_low": band_extremum(45.0, 75.0, False, -1.0),
        },
        "itcz_trough": itcz_trough(),
    }


__all__ = [
    "SesamSlp",
    "cell_coordinate",
    "cell_temperature_gradients",
    "charney_eliassen_slp",
    "compute_slp",
    "hadley_geometry",
    "hadley_width_scale",
    "hemispheric_mean_temperatures",
    "itcz_latitude",
    "mean_overturning_wind",
    "resolution_matched_field",
    "resolution_matched_profile",
    "sea_level_temperature",
    "thermal_azonal_slp",
    "topography_factor",
    "tropical_weight_from_hadley",
    "zonal_slp_anomaly",
    "zonal_slp_extrema",
]
