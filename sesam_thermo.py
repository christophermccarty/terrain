"""SESAM column energy and water closure -- Appendix A3/A4, stage P4.

Pure, read-only kernels for the CLIMBER-X/SESAM prognostic column-energy and
column-water budgets, transcribed from Appendix A3/A4 of

    Willeit, M., Ganopolski, A., Robinson, A., and Edwards, N. R.:
    GMD 15, 5905-5948 (2022), https://doi.org/10.5194/gmd-15-5905-2022
    (CC-BY 4.0; equations cited below as (A#) against that paper).

This is SESAM stage P4 (docs/SESAM_GAP_ANALYSIS.md Sec7): the prognostic column
energy ``QT``/``Ta`` (A40), the near-surface diagnostics ``T2m`` (A41) and
``q2m`` (A43), the prognostic column water ``Qq``/``qa`` (A42, already built
as ``column_water.py``), and the (A44)/(A45) precipitation closure that
**bypasses the row-target allocator** the supported path still uses. Like
every SESAM stage, this module has zero default-path climate impact:
``simulate.py`` never imports it while ``PlanetParams.enable_sesam_column_closure``
stays False (the reservation stage P1-P3 already established for their own
gates).

**Equations verified against the article PDF (2026-08-18)**, the same
preprint mirror (``gmd-2022-56-manuscript-version2.pdf``, pages 44-47) whose
appendix text caught the real (A38) bug during stage P2 -- this is the first
time A40-A45 have been read directly rather than summarised, so the forms
below are transcribed from the literal printed text, not reconstructed from
the gap-analysis dossier's paraphrase:

1. **(A40)** ``dQT/dt = -div(u T) - div(u' T') + cv^-1 (SWa + LWa + Le Pw +
   Ls Ps + SH)``, with ``QT`` formally ``integral_zs^HTOA T dz`` (a column
   heat content) but the paper's own following sentence states plainly "the
   energy balance equation is solved for Ta" -- i.e. the model's actual
   prognostic state is the near-surface air temperature, not a separately
   tracked heat-content integral. This module follows the paper's own
   practice: the prognostic scalar is ``Ta`` (Kelvin), and the diabatic
   source (SW+LW+latent+sensible, W/m^2) is divided by an explicit *column
   heat capacity* ``cv`` (J/(m^2 K)) to get a direct K/s tendency --
   dimensionally the same operation the paper's cv^-1 division performs, just
   without QT's otherwise-unexplained K*m integral units getting in the way.
2. **cv is not a published constant.** Table A3 in ``sesam_reference.py``
   documents this: "the paper publishes no separate parameter table for the
   A3 column-energy equation beyond physical constants (cv, Le, Ls)". This
   module computes a column heat capacity from the hydrostatic column mass
   per unit area (``p0/g``) and an explicit constant-volume specific heat
   ``cv_specific_j_kg_k`` -- a standard atmospheric-physics relation, not a
   fit -- defaulting to 717.0 J/(kg K) (``cp - Rd`` using this project's own
   ``cp_air=1004.0``/``Rd=287.0`` convention, already used verbatim in
   ``land_surface.py`` and ``atmosphere.py``, so the new module's constants
   agree with the rest of the codebase rather than introducing a third set).
   ``Le``/``Ls`` default to 2.5e6/2.834e6 J/kg, the same ``Le=2.5e6`` value
   ``atmosphere.py`` already uses for its own latent-heating terms
   (``atmosphere.py:3671,4955,5021,5073``); ``Ls`` is the standard
   ``Le + Lf`` (Lf = 3.34e5 J/kg fusion) sublimation value, not separately
   used elsewhere in this project yet.
3. **(A44) precipitation, transcribed exactly as printed** (this differs in
   operand grouping from the gap-analysis dossier's earlier paraphrase
   "moisture convergence past 95% RH"): ``P = max(0, C + Cslope + E) *
   (ra/ramax) + Qq*ra/tau_p`` [land only for the second term]. The *first*
   term is **not** a hard threshold at ``ra > ramax`` -- it is a *continuous*
   efficiency ``ra/ramax`` applied to the *entire* gross convergence-plus-
   evaporation term at every ``ra``, reaching 100% conversion (all
   convergent water rains out immediately) only when ``ra`` reaches its
   ceiling ``ramax``. The second term is the land-only turnover
   ``Qq*ra/tau_p`` the dossier's paraphrase already had right. ``C`` is
   "moisture convergence into the atmospheric column by advection and
   diffusion" (A44 prose) -- this module measures it directly from
   ``column_water.evolve_column_water``'s own ``transport_tendency_mm_day``
   output on a zero-source diagnostic pass plus the (A45) diffusion
   tendency, rather than re-deriving the divergence operator a second time.
4. **(A45)** ``Cslope = c_slope_p * sqrt(K) * |grad zs| * rho0 * qa`` --
   confirmed to exactly match the pre-existing ``sesam_reference.py``
   transcription (no correction needed here, unlike A38/A31/A34 in stage
   P2). Units as printed give a mass flux (kg/(m^2 s)); this module converts
   to mm/day by the same ``kg/m^2 == mm water`` identity ``column_water.py``
   already uses throughout, multiplying by 86400. **``qa`` here is near-
   surface specific humidity (kg/kg), a genuinely different quantity from
   ``Qq`` (the column-integrated water depth in mm that is (A42)'s own
   prognostic state)** -- an early version of ``evolve_column_water_vapor``
   passed ``Qq`` into this slot by mistake, inflating Cslope by roughly the
   mm-to-kg/kg magnitude ratio (~1000x); caught by
   ``testing/test_sesam_thermo.py``'s physical-bounds check on a synthetic
   random field before this module ever touched the real saved state. Fixed
   by giving ``evolve_column_water_vapor`` an explicit, separate
   ``near_surface_specific_humidity_kg_kg`` parameter rather than reusing
   ``qa_water_mm`` for both roles.
5. **(A46)-(A51) macroturbulent diffusion of heat and moisture uses the
   *same* AT/Aq diffusivities already built for K's own (A52) transport in
   stage P3** (``sesam_synoptic.horizontal_diffusion_coefficient``,
   ``sesam_synoptic.moisture_diffusion_coefficient``) -- confirmed directly
   from the PDF: (A46)-(A49) are literally "AT"/"Aq" applied to grad(T) and
   grad(q), the identical symbols (A50)/(A51) define and P3 already computes
   from K. This is a real finding, not an assumption: it means Ta/qa
   diffusion in this module is not a new closure, it is the *same* closure
   applied to a different scalar, and this module deliberately does not
   recompute AT/Aq -- it imports P3's functions directly.
6. **T2m (A41)** ``= (Ta + T*)/2`` where ``T*`` is skin temperature; used for
   the surface sensible-heat flux (paper text, not re-derived here -- the
   existing surface-flux code already computes SH from *some* near-surface
   temperature, and this module documents the formula so a future coupling
   stage can supply the SESAM-consistent T2m to it instead).
7. **q2m (A43)** ``= r2m * qsat(T2m)``, ``r2m = (ra + r*)/2``, ``r* =
   qa/qsat(T*)`` -- transcribed exactly as printed; ``qsat`` reuses
   ``sesam_vertical.saturation_specific_humidity`` (the same A15 ice/water
   partition), not a second implementation.
8. **Two-pass convergence/precipitation closure**, the same architectural
   pattern stage P2 already used for the SLP<->wind mutual dependency
   (``docs/SESAM_GAP_ANALYSIS.md`` Sec7 P2, "two-pass closure"): ``C`` (A44's
   moisture-convergence input) is measured from a diagnostic zero-source
   transport pass on the *pre-precipitation* water field, then the real
   ``evolve_column_water`` call applies the resulting ``P`` as the actual
   source. This is a documented approximation, not a bug: the two calls'
   transport differs by however much a nonzero source perturbs the
   substep-by-substep water depth (the substep *count* is identical -- see
   ``column_water.evolve_column_water``, whose CFL sub-stepping depends only
   on the wind field, not the source), which is the same order of
   approximation P2's own two-pass SLP/wind closure already accepted.

Grids follow the P1-P3 convention: 2-D fields ``(H, W)``, rows north-to-south
on cell centres. Planetary constants are explicit inputs -- never hardcoded
Earth literals -- except the cv/Le/Ls physical constants above, which the
paper leaves unpublished and which this module treats exactly as
``land_surface.py``/``atmosphere.py`` already treat them elsewhere in this
codebase (documented implementation choices, not paper transcriptions).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from column_water import evolve_column_water
from sesam_reference import value as _param
from sesam_synoptic import (
    horizontal_diffusion_coefficient,
    moisture_diffusion_coefficient,
)
from sesam_synoptic import _cyclic_thomas_batch
from sesam_vertical import saturation_specific_humidity

# ---------------------------------------------------------------------------
# Physical constants not published by the paper (see module docstring note 2)
# ---------------------------------------------------------------------------

# cp - Rd with this project's existing land_surface.py/atmosphere.py values.
_CV_SPECIFIC_DEFAULT_J_KG_K = 717.0
_LE_DEFAULT_J_KG = 2.5e6  # matches atmosphere.py's existing Le usage
_LF_J_KG = 3.34e5  # latent heat of fusion, standard value
_LS_DEFAULT_J_KG = _LE_DEFAULT_J_KG + _LF_J_KG


def _a4_defaults() -> dict[str, float]:
    """(A44)/(A45) constants from the published Table A4 pack."""
    return {
        "tau_p": _param("A4_hydrology", "tau_p"),
        "ra_max": _param("A4_hydrology", "ra_max"),
        "c_slope_p": _param("A4_hydrology", "c_slope_p"),
    }


def _check_2d(name: str, value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D (H, W) field")
    return arr


# ---------------------------------------------------------------------------
# (A41)/(A43) near-surface diagnostics
# ---------------------------------------------------------------------------


def t2m_diagnostic(ta_k: np.ndarray, skin_temp_k: np.ndarray) -> np.ndarray:
    """(A41) ``T2m = (Ta + T*)/2``."""
    ta = _check_2d("ta_k", ta_k)
    tstar = _check_2d("skin_temp_k", skin_temp_k)
    if ta.shape != tstar.shape:
        raise ValueError("ta_k and skin_temp_k must share a shape")
    return 0.5 * (ta + tstar)


def surface_relative_humidity_star(
    qa_kg_kg: np.ndarray, skin_temp_k: np.ndarray, pressure_pa: np.ndarray
) -> np.ndarray:
    """``r* = qa / qsat(T*)`` -- the skin-temperature-referenced RH used by
    (A43)'s ``r2m`` blend. Not itself a printed equation number; the paper's
    text names it ``r*`` without a separate (A#) tag between (A43) and its
    definition sentence.
    """
    qa = _check_2d("qa_kg_kg", qa_kg_kg)
    tstar = _check_2d("skin_temp_k", skin_temp_k)
    p = np.broadcast_to(_check_2d("pressure_pa", pressure_pa), qa.shape)
    qsat_star = saturation_specific_humidity(tstar, p)
    return qa / np.maximum(qsat_star, 1e-8)


def q2m_diagnostic(
    near_surface_rh: np.ndarray,
    r_star: np.ndarray,
    ta_k: np.ndarray,
    skin_temp_k: np.ndarray,
    pressure_pa: np.ndarray,
) -> np.ndarray:
    """(A43) ``q2m = r2m * qsat(T2m)``, ``r2m = (ra + r*)/2``."""
    ra = _check_2d("near_surface_rh", near_surface_rh)
    rstar = _check_2d("r_star", r_star)
    if ra.shape != rstar.shape:
        raise ValueError("near_surface_rh and r_star must share a shape")
    r2m = 0.5 * (ra + rstar)
    t2m = t2m_diagnostic(ta_k, skin_temp_k)
    p = np.broadcast_to(_check_2d("pressure_pa", pressure_pa), t2m.shape)
    return r2m * saturation_specific_humidity(t2m, p)


# ---------------------------------------------------------------------------
# (A40) column heat capacity and diabatic source
# ---------------------------------------------------------------------------


def column_heat_capacity_j_m2_k(
    p0_pa: np.ndarray | float,
    gravity: float,
    cv_specific_j_kg_k: float = _CV_SPECIFIC_DEFAULT_J_KG_K,
) -> np.ndarray | float:
    """Column heat capacity ``cv = (p0/g) * cv_specific`` [J m^-2 K^-1].

    ``p0/g`` is the standard hydrostatic column dry-air mass per unit area;
    multiplying by a constant-volume specific heat gives the capacity that
    converts a W/m^2 diabatic flux into a Kelvin-per-second tendency for the
    whole column -- see module docstring notes 1-2 for why this project
    tracks ``Ta`` directly rather than the paper's formal ``QT`` integral.
    """
    p0 = np.asarray(p0_pa, dtype=np.float64) if not np.isscalar(p0_pa) else float(p0_pa)
    return p0 / float(gravity) * float(cv_specific_j_kg_k)


def diabatic_heating_rate_k_day(
    sw_absorbed_w_m2: np.ndarray,
    lw_net_w_m2: np.ndarray,
    rainfall_mm_day: np.ndarray,
    snowfall_mm_day: np.ndarray,
    sensible_heat_w_m2: np.ndarray,
    column_heat_capacity_j_m2_k: np.ndarray | float,
    *,
    latent_heat_vaporization_j_kg: float = _LE_DEFAULT_J_KG,
    latent_heat_sublimation_j_kg: float = _LS_DEFAULT_J_KG,
) -> np.ndarray:
    """(A40) source assembly: ``cv^-1 * (SWa + LWa + Le*Pw + Ls*Ps + SH)``.

    ``rainfall_mm_day``/``snowfall_mm_day`` are converted to mass flux
    (kg m^-2 s^-1) via the same ``1 mm water == 1 kg/m^2`` identity
    ``column_water.py`` uses throughout, then to a heating rate (W/m^2) via
    the latent heats, before dividing by the column heat capacity and
    rescaling seconds to days.
    """
    sw = _check_2d("sw_absorbed_w_m2", sw_absorbed_w_m2)
    lw = _check_2d("lw_net_w_m2", lw_net_w_m2)
    pw = _check_2d("rainfall_mm_day", rainfall_mm_day)
    ps = _check_2d("snowfall_mm_day", snowfall_mm_day)
    sh = _check_2d("sensible_heat_w_m2", sensible_heat_w_m2)
    if not (sw.shape == lw.shape == pw.shape == ps.shape == sh.shape):
        raise ValueError("all (A40) source fields must share a shape")
    cv = np.broadcast_to(np.asarray(column_heat_capacity_j_m2_k, dtype=np.float64), sw.shape)
    if np.any(cv <= 0.0):
        raise ValueError("column_heat_capacity_j_m2_k must be positive")
    pw_flux = pw / 86400.0  # mm/day -> kg m^-2 s^-1
    ps_flux = ps / 86400.0
    net_w_m2 = (
        sw + lw
        + latent_heat_vaporization_j_kg * pw_flux
        + latent_heat_sublimation_j_kg * ps_flux
        + sh
    )
    return (net_w_m2 / cv) * 86400.0  # K/s -> K/day


# ---------------------------------------------------------------------------
# Shared conservative linear diffusion (fixed, externally supplied diffusivity)
# ---------------------------------------------------------------------------


class _DiffusionStep(NamedTuple):
    field: np.ndarray
    residual: float
    relative_residual: float
    substeps: int
    maximum_diffusion_number: float


def _linear_diffusion_step(
    field: np.ndarray,
    diffusivity_m2_s: np.ndarray,
    *,
    dx_m: np.ndarray | float,
    dy_m: float,
    dt_days: float,
    cell_area_m2: np.ndarray | float,
    x_face_length_m: np.ndarray | float,
    y_face_length_m: np.ndarray,
    r_limit: float = 0.4,
    nonnegative: bool = True,
) -> _DiffusionStep:
    """Conservative finite-volume diffusion of ``field`` by a *given*
    (externally computed, not self-derived) diffusivity field.

    Reuses the exact per-cell face-length/area self-loss geometry
    ``sesam_synoptic.eke_diffusion_step`` derived and unit-tested against the
    real 512x1024 polar stiffness bug (module docstring note 5): the
    ``x_len/(dx*area) + y_len/(dy*area)`` sum, not the ``1/dx^2+1/dy^2``
    approximation that catastrophically underestimates the true CFL
    constraint at the polar-cap row. Unlike ``eke_diffusion_step``, the
    diffusivity here does **not** depend on the field being diffused (AT/Aq
    depend on K, not on Ta/qa), so this is genuinely *linear* diffusion: the
    stability bound and substep count are computed once up front rather than
    adaptively re-estimated each substep (no circularity to guard against).
    Diffusivity is still face-averaged before forming each flux -- the same
    conservative, symmetry-preserving treatment ``eke_diffusion_step`` uses
    for a state-dependent diffusivity, applied here to a state-*independent*
    one for consistency and so a spatially uniform diffusivity reduces to the
    textbook constant-coefficient scheme.
    """
    if dt_days <= 0.0 or dy_m <= 0.0:
        raise ValueError("dt_days/dy_m must be positive")
    if not 0.0 < r_limit <= 0.5:
        raise ValueError("r_limit must be in (0, 0.5]")
    f0 = np.asarray(field, dtype=np.float64)
    if nonnegative:
        f0 = np.clip(f0, 0.0, None)
    if f0.ndim != 2:
        raise ValueError("field must be a 2-D (H, W) array")
    h, w = f0.shape
    dx = np.broadcast_to(np.asarray(dx_m, dtype=np.float64), (h, w))
    if np.any(dx <= 0.0):
        raise ValueError("dx_m must be positive")
    area = np.broadcast_to(np.asarray(cell_area_m2, dtype=np.float64), (h, w))
    if np.any(area <= 0.0):
        raise ValueError("cell_area_m2 must be positive")
    x_len = np.broadcast_to(np.asarray(x_face_length_m, dtype=np.float64), (h, w))
    y_len = np.broadcast_to(np.asarray(y_face_length_m, dtype=np.float64), (h + 1, w))
    at = np.broadcast_to(np.asarray(diffusivity_m2_s, dtype=np.float64), (h, w))
    if np.any(at < 0.0):
        raise ValueError("diffusivity_m2_s must be non-negative")

    self_loss_geometry = (
        x_len / (dx * area)
        + np.roll(x_len, 1, axis=1) / (np.roll(dx, 1, axis=1) * area)
        + y_len[:-1] / (dy_m * area)
        + y_len[1:] / (dy_m * area)
    )
    dt_seconds = dt_days * 86400.0
    max_rate = float(np.max(at * self_loss_geometry))
    n_sub = 1 if max_rate <= 0.0 else max(1, int(np.ceil(max_rate * dt_seconds / r_limit)))
    max_substeps = 200_000  # same generous pathological-case cap as eke_diffusion_step
    if n_sub > max_substeps:
        raise RuntimeError(
            f"_linear_diffusion_step would need {n_sub} substeps (cap {max_substeps}) -- "
            "unexpectedly stiff diffusivity/geometry for the explicit scheme; use "
            "_linear_diffusion_step_implicit_zonal instead"
        )
    dt_sub = dt_seconds / n_sub

    at_east = 0.5 * (at + np.roll(at, -1, axis=1))
    at_face_y = np.zeros((h + 1, w), dtype=np.float64)
    at_face_y[1:-1] = 0.5 * (at[:-1] + at[1:])

    mass = f0 * area
    for _ in range(n_sub):
        depth = mass / area
        east_flux = at_east * (depth - np.roll(depth, -1, axis=1)) / dx * x_len
        west_flux_in = np.roll(east_flux, 1, axis=1)
        face_flux = np.zeros((h + 1, w), dtype=np.float64)
        face_flux[1:-1] = at_face_y[1:-1] * (depth[:-1] - depth[1:]) / dy_m * y_len[1:-1]
        mass = mass + dt_sub * (west_flux_in - east_flux + face_flux[:-1] - face_flux[1:])
        if nonnegative:
            mass = np.maximum(mass, 0.0)

    f_next = mass / area
    if nonnegative:
        f_next = np.maximum(f_next, 0.0)
    initial_total = float(np.sum(f0 * area))
    final_total = float(np.sum(f_next * area))
    residual = final_total - initial_total
    relative_residual = residual / max(abs(final_total), abs(initial_total), 1.0)
    return _DiffusionStep(f_next, residual, relative_residual, n_sub, max_rate * dt_sub)


def _linear_diffusion_step_implicit_zonal(
    field: np.ndarray,
    diffusivity_m2_s: np.ndarray,
    *,
    dx_m: np.ndarray | float,
    dy_m: float,
    dt_days: float,
    cell_area_m2: np.ndarray | float,
    x_face_length_m: np.ndarray | float,
    y_face_length_m: np.ndarray,
    r_limit: float = 0.4,
    nonnegative: bool = True,
) -> _DiffusionStep:
    """ADI-style directional split of ``_linear_diffusion_step``: implicit
    (backward-Euler, unconditionally stable) in longitude, explicit
    (CFL-substepped) in latitude -- the same polar-stiffness remedy stage P3
    built for K's own (A52) diffusion (``sesam_synoptic.eke_diffusion_step_implicit_zonal``),
    needed here for the identical reason: AT/Aq inherit K's own magnitude
    (module docstring note 5), so diffusing Ta/qa by them at this project's
    512x1024 headline grid hits the same ``x_len/area`` polar divergence
    P3 already measured and fixed, not a new problem to re-diagnose.

    Because the diffusivity here is *fixed* for the whole call (not
    self-referential the way K's own AT is), the zonal conductances need
    only be built once; unlike P3's per-substep AT recompute, this function's
    substep loop exists solely to bound the *explicit meridional* half, which
    is unconditionally reused unchanged from ``_linear_diffusion_step``'s own
    meridional flux form. Reuses ``sesam_synoptic._cyclic_thomas_batch``
    directly rather than re-implementing the periodic tridiagonal solve --
    the same algorithm, cross-checked to ~1e-14 against a dense periodic
    solve during P3's own development, so a second independent
    implementation here would just be a second place the same bug could
    exist undetected.
    """
    if dt_days <= 0.0 or dy_m <= 0.0:
        raise ValueError("dt_days/dy_m must be positive")
    if not 0.0 < r_limit <= 0.5:
        raise ValueError("r_limit must be in (0, 0.5]")
    f0 = np.asarray(field, dtype=np.float64)
    if nonnegative:
        f0 = np.clip(f0, 0.0, None)
    if f0.ndim != 2:
        raise ValueError("field must be a 2-D (H, W) array")
    h, w = f0.shape
    if w < 3:
        raise ValueError("implicit-zonal diffusion needs at least 3 longitude columns")
    dx = np.broadcast_to(np.asarray(dx_m, dtype=np.float64), (h, w))
    if np.any(dx <= 0.0):
        raise ValueError("dx_m must be positive")
    area = np.broadcast_to(np.asarray(cell_area_m2, dtype=np.float64), (h, w))
    if np.any(area <= 0.0):
        raise ValueError("cell_area_m2 must be positive")
    x_len = np.broadcast_to(np.asarray(x_face_length_m, dtype=np.float64), (h, w))
    y_len = np.broadcast_to(np.asarray(y_face_length_m, dtype=np.float64), (h + 1, w))
    at = np.broadcast_to(np.asarray(diffusivity_m2_s, dtype=np.float64), (h, w))
    if np.any(at < 0.0):
        raise ValueError("diffusivity_m2_s must be non-negative")

    # Meridional-only self-loss geometry bounds the explicit half's substeps.
    self_loss_y = y_len[:-1] / (dy_m * area) + y_len[1:] / (dy_m * area)
    dt_seconds = dt_days * 86400.0
    max_rate_y = float(np.max(at * self_loss_y))
    n_sub = 1 if max_rate_y <= 0.0 else max(1, int(np.ceil(max_rate_y * dt_seconds / r_limit)))
    max_substeps = 200_000
    if n_sub > max_substeps:
        raise RuntimeError(
            f"_linear_diffusion_step_implicit_zonal would need {n_sub} meridional substeps "
            f"(cap {max_substeps}) -- unexpectedly stiff meridional input (the zonal term is "
            "unconditionally stable by construction)"
        )
    dt_sub = dt_seconds / n_sub

    # Zonal conductances are fixed for the whole call (AT does not depend on
    # the diffused field here), so build them once outside the substep loop.
    at_east = 0.5 * (at + np.roll(at, -1, axis=1))
    c_face = at_east / dx * x_len
    c_west = np.roll(c_face, 1, axis=1)
    at_face_y = np.zeros((h + 1, w), dtype=np.float64)
    at_face_y[1:-1] = 0.5 * (at[:-1] + at[1:])

    mass = f0 * area
    for _ in range(n_sub):
        depth = mass / area
        sub_diag = -dt_sub * c_west
        sup_diag = -dt_sub * c_face
        main_diag = area + dt_sub * (c_west + c_face)
        rhs = area * depth
        depth_x = _cyclic_thomas_batch(sub_diag, main_diag, sup_diag, rhs)
        if nonnegative:
            depth_x = np.maximum(depth_x, 0.0)
        mass_x = depth_x * area

        face_flux = np.zeros((h + 1, w), dtype=np.float64)
        face_flux[1:-1] = at_face_y[1:-1] * (depth_x[:-1] - depth_x[1:]) / dy_m * y_len[1:-1]
        mass = mass_x + dt_sub * (face_flux[:-1] - face_flux[1:])
        if nonnegative:
            mass = np.maximum(mass, 0.0)

    f_next = mass / area
    if nonnegative:
        f_next = np.maximum(f_next, 0.0)
    initial_total = float(np.sum(f0 * area))
    final_total = float(np.sum(f_next * area))
    residual = final_total - initial_total
    relative_residual = residual / max(abs(final_total), abs(initial_total), 1.0)
    return _DiffusionStep(f_next, residual, relative_residual, n_sub, max_rate_y * dt_sub)


# ---------------------------------------------------------------------------
# (A40) column energy evolution
# ---------------------------------------------------------------------------


class ColumnEnergyStep(NamedTuple):
    temperature_k: np.ndarray
    residual_k: float
    relative_residual: float
    advection_substeps: int
    diffusion_substeps: int
    maximum_diffusion_number: float


def evolve_column_energy(
    ta_k: np.ndarray,
    diabatic_heating_k_day: np.ndarray,
    wind_u_m_s: np.ndarray,
    wind_v_m_s: np.ndarray,
    eke_m2_s2: np.ndarray,
    *,
    dx_m: np.ndarray | float,
    dy_m: float,
    dt_days: float,
    cell_area_m2: np.ndarray | float,
    x_face_length_m: np.ndarray | float,
    y_face_length_m: np.ndarray,
    max_courant: float = 0.5,
    diffusion_r_limit: float = 0.4,
    a5: dict[str, float] | None = None,
    implicit_zonal_diffusion: bool = False,
) -> ColumnEnergyStep:
    """(A40), operator-split: advection + diabatic source, then macroturbulent
    diffusion -- the same two-piece split stage P3 used for K's (A52)
    transport (``eke_transport_step``), applied here to a column-energy
    budget that genuinely has a nonzero source at this stage (unlike K's
    transport-only sub-deliverable, which deliberately isolated pure
    transport with zero source; see module docstring note 1 for why
    ``Ta`` in Kelvin is a safe drop-in for ``column_water.evolve_column_water``'s
    ``water_mm`` slot: always positive, so the function's own non-negativity
    clip is a no-op for any physically possible temperature).

    The advective step reuses ``column_water.evolve_column_water`` directly,
    with ``diabatic_heating_k_day`` supplied through its ``evaporation_mm_day``
    source slot (misnamed for this use, but algebraically the exact ``E - P``
    additive source term the function already implements; ``precipitation_mm_day``
    is passed as zero here since it is not this scalar's actual precipitation).
    The diffusive step uses stage P3's own AT (``horizontal_diffusion_coefficient``,
    (A50)) -- the literal same diffusivity the paper's (A46)/(A49) print for
    heat, not a new one (module docstring note 5). ``implicit_zonal_diffusion=True``
    routes the diffusive step through ``_linear_diffusion_step_implicit_zonal``,
    needed for a tractable run at the 512x1024 headline grid (see that
    function's docstring); optional and numerically inert as a *choice* at
    coarser grids where the explicit scheme is already affordable.
    """
    ta0 = _check_2d("ta_k", ta_k)
    heating = _check_2d("diabatic_heating_k_day", diabatic_heating_k_day)
    if ta0.shape != heating.shape:
        raise ValueError("ta_k and diabatic_heating_k_day must share a shape")
    zeros = np.zeros_like(ta0)
    adv = evolve_column_water(
        ta0, heating, zeros, wind_u_m_s, wind_v_m_s,
        dx_m=dx_m, dy_m=dy_m, dt_days=dt_days,
        cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
        y_face_length_m=y_face_length_m, max_courant=max_courant,
    )
    at = horizontal_diffusion_coefficient(np.asarray(eke_m2_s2, dtype=np.float64), a5=a5)
    diffusion_fn = (
        _linear_diffusion_step_implicit_zonal if implicit_zonal_diffusion else _linear_diffusion_step
    )
    diff = diffusion_fn(
        adv.water_mm, at,
        dx_m=dx_m, dy_m=dy_m, dt_days=dt_days,
        cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
        y_face_length_m=y_face_length_m, r_limit=diffusion_r_limit,
        nonnegative=False,
    )
    residual = float(adv.residual_mm) + float(diff.residual)
    area = np.broadcast_to(np.asarray(cell_area_m2, dtype=np.float64), ta0.shape)
    denom = max(abs(float(np.sum(diff.field * area))), abs(float(np.sum(ta0 * area))), 1.0)
    return ColumnEnergyStep(
        diff.field.astype(np.float32), residual, residual / denom,
        int(adv.substeps), int(diff.substeps), float(diff.maximum_diffusion_number),
    )


# ---------------------------------------------------------------------------
# (A44)/(A45) precipitation and (A42) column water evolution
# ---------------------------------------------------------------------------


def slope_convergence_mm_day(
    eke_m2_s2: np.ndarray,
    slope_magnitude: np.ndarray,
    qa_kg_kg: np.ndarray,
    rho0_kg_m3: float,
    a4: dict[str, float] | None = None,
) -> np.ndarray:
    """(A45) ``Cslope = c_slope_p * sqrt(K) * |grad zs| * rho0 * qa``.

    Printed units give a mass flux (kg m^-2 s^-1); converted to mm/day via
    the ``1 mm water == 1 kg/m^2`` identity ``column_water.py`` uses
    throughout (multiply by 86400).
    """
    params = a4 or _a4_defaults()
    k = np.maximum(_check_2d("eke_m2_s2", eke_m2_s2), 0.0)
    slope = np.abs(_check_2d("slope_magnitude", slope_magnitude))
    qa = _check_2d("qa_kg_kg", qa_kg_kg)
    if not (k.shape == slope.shape == qa.shape):
        raise ValueError("eke_m2_s2, slope_magnitude, and qa_kg_kg must share a shape")
    flux_kg_m2_s = params["c_slope_p"] * np.sqrt(k) * slope * float(rho0_kg_m3) * qa
    return flux_kg_m2_s * 86400.0


class ConvergenceMeasurement(NamedTuple):
    convergence_mm_day: np.ndarray
    diffusion_substeps: int
    water_after_diffusion_mm: np.ndarray


def moisture_convergence_mm_day(
    qa_water_mm: np.ndarray,
    wind_u_m_s: np.ndarray,
    wind_v_m_s: np.ndarray,
    eke_m2_s2: np.ndarray,
    *,
    dx_m: np.ndarray | float,
    dy_m: float,
    dt_days: float,
    cell_area_m2: np.ndarray | float,
    x_face_length_m: np.ndarray | float,
    y_face_length_m: np.ndarray,
    max_courant: float = 0.5,
    diffusion_r_limit: float = 0.4,
    a5: dict[str, float] | None = None,
    implicit_zonal_diffusion: bool = False,
) -> ConvergenceMeasurement:
    """(A44)'s ``C``: "moisture convergence into the atmospheric column by
    advection and diffusion", measured directly rather than re-derived.

    Diffuses ``qa_water_mm`` by stage P3's own Aq (module docstring note 5),
    then runs a zero-source ``column_water.evolve_column_water`` pass on the
    diffused field and reads its ``transport_tendency_mm_day`` output
    (positive = net convergence) for the advective piece. The two rates are
    summed (paper text: "by advection and diffusion", one combined ``C``),
    reported per day.  This is a diagnostic measurement, not itself a state
    update -- see module docstring note 8 for how the caller (`evolve_column_water_vapor`)
    uses it. ``implicit_zonal_diffusion=True`` routes the diffusion step
    through ``_linear_diffusion_step_implicit_zonal`` (needed for the
    512x1024 headline grid; see that function's docstring).
    """
    q0 = _check_2d("qa_water_mm", qa_water_mm)
    aq = moisture_diffusion_coefficient(np.asarray(eke_m2_s2, dtype=np.float64), a5=a5)
    diffusion_fn = (
        _linear_diffusion_step_implicit_zonal if implicit_zonal_diffusion else _linear_diffusion_step
    )
    diff = diffusion_fn(
        q0, aq,
        dx_m=dx_m, dy_m=dy_m, dt_days=dt_days,
        cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
        y_face_length_m=y_face_length_m, r_limit=diffusion_r_limit,
        nonnegative=True,
    )
    diffusion_rate_mm_day = (diff.field - q0) / dt_days
    zeros = np.zeros_like(q0)
    adv = evolve_column_water(
        diff.field, zeros, zeros, wind_u_m_s, wind_v_m_s,
        dx_m=dx_m, dy_m=dy_m, dt_days=dt_days,
        cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
        y_face_length_m=y_face_length_m, max_courant=max_courant,
    )
    total = diffusion_rate_mm_day + adv.transport_tendency_mm_day
    return ConvergenceMeasurement(total, int(diff.substeps), diff.field)


def precipitation_rate_mm_day(
    convergence_mm_day: np.ndarray,
    slope_convergence_mm_day: np.ndarray,
    evaporation_mm_day: np.ndarray,
    near_surface_rh: np.ndarray,
    column_water_mm: np.ndarray,
    *,
    land_mask: np.ndarray | None = None,
    a4: dict[str, float] | None = None,
) -> np.ndarray:
    """(A44) ``P = max(0, C + Cslope + E) * (ra/ramax) + [land] Qq*ra/tau_p``.

    The first term is a *continuous* efficiency (``ra/ramax``, clamped to
    [0, 1] since ``ra`` can exceed ``ramax`` -- an explicit, documented safety
    clamp, not a printed one; see module docstring note 3), not a hard
    threshold: it scales the *entire* gross convergence-plus-evaporation
    term, not just an excess above a cutoff. The second (land turnover) term
    is applied only where ``land_mask`` is truthy (defaults to everywhere if
    omitted, i.e. no land restriction -- callers with a real land/ocean mask
    should always pass it, since the paper explicitly restricts this term to
    land).
    """
    params = a4 or _a4_defaults()
    c = _check_2d("convergence_mm_day", convergence_mm_day)
    cslope = _check_2d("slope_convergence_mm_day", slope_convergence_mm_day)
    e = _check_2d("evaporation_mm_day", evaporation_mm_day)
    ra = _check_2d("near_surface_rh", near_surface_rh)
    qq = _check_2d("column_water_mm", column_water_mm)
    if not (c.shape == cslope.shape == e.shape == ra.shape == qq.shape):
        raise ValueError("all (A44) input fields must share a shape")
    land = (
        np.ones_like(c, dtype=np.float64)
        if land_mask is None
        else np.asarray(land_mask, dtype=np.float64).astype(bool).astype(np.float64)
    )
    if land.shape != c.shape:
        raise ValueError("land_mask must share the field shape")
    efficiency = np.clip(ra / params["ra_max"], 0.0, 1.0)
    gross_term = np.maximum(c + cslope + e, 0.0) * efficiency
    turnover_term = land * qq * ra / params["tau_p"]
    return gross_term + turnover_term


class ColumnWaterVaporStep(NamedTuple):
    water_mm: np.ndarray
    precipitation_mm_day: np.ndarray
    convergence_mm_day: np.ndarray
    slope_convergence_mm_day: np.ndarray
    residual_mm: float
    relative_residual: float
    diffusion_substeps: int
    advection_substeps: int


def evolve_column_water_vapor(
    qa_water_mm: np.ndarray,
    evaporation_mm_day: np.ndarray,
    wind_u_m_s: np.ndarray,
    wind_v_m_s: np.ndarray,
    eke_m2_s2: np.ndarray,
    near_surface_rh: np.ndarray,
    near_surface_specific_humidity_kg_kg: np.ndarray,
    slope_magnitude: np.ndarray,
    land_mask: np.ndarray,
    *,
    dx_m: np.ndarray | float,
    dy_m: float,
    dt_days: float,
    cell_area_m2: np.ndarray | float,
    x_face_length_m: np.ndarray | float,
    y_face_length_m: np.ndarray,
    rho0_kg_m3: float,
    max_courant: float = 0.5,
    diffusion_r_limit: float = 0.4,
    a4: dict[str, float] | None = None,
    a5: dict[str, float] | None = None,
    implicit_zonal_diffusion: bool = False,
) -> ColumnWaterVaporStep:
    """(A42)/(A44) full column-water step: diffuse, measure convergence,
    generate precipitation, then advance the real state with the resulting
    ``P`` -- the two-pass closure documented in module docstring note 8.

    This is the mechanism that bypasses the supported path's row-target
    precipitation allocator: ``P`` here comes entirely from (A44)'s local
    moisture-budget residual, never from a prescribed ``target_row_mm_day``
    profile. ``implicit_zonal_diffusion=True`` is forwarded to
    ``moisture_convergence_mm_day`` (needed for the 512x1024 headline grid).

    ``qa_water_mm`` is the column-integrated water *depth* (``Qq``, A42's
    prognostic state, in the same mm units as ``column_water.py`` throughout)
    while ``near_surface_specific_humidity_kg_kg`` is the *distinct* (A45)
    input ``qa`` (near-surface specific humidity, kg/kg) -- conflating the two
    was a real bug caught by ``testing/test_sesam_thermo.py``'s physical-
    bounds check (an earlier version passed the mm-scale column depth into
    (A45)'s kg/kg slot, inflating Cslope by roughly the water-depth-to-
    specific-humidity ratio, ~1000x).
    """
    q0 = _check_2d("qa_water_mm", qa_water_mm)
    e = _check_2d("evaporation_mm_day", evaporation_mm_day)
    ra = _check_2d("near_surface_rh", near_surface_rh)
    qa = _check_2d("near_surface_specific_humidity_kg_kg", near_surface_specific_humidity_kg_kg)
    conv = moisture_convergence_mm_day(
        q0, wind_u_m_s, wind_v_m_s, eke_m2_s2,
        dx_m=dx_m, dy_m=dy_m, dt_days=dt_days,
        cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
        y_face_length_m=y_face_length_m, max_courant=max_courant,
        diffusion_r_limit=diffusion_r_limit, a5=a5,
        implicit_zonal_diffusion=implicit_zonal_diffusion,
    )
    cslope = slope_convergence_mm_day(eke_m2_s2, slope_magnitude, qa, rho0_kg_m3, a4=a4)
    p = precipitation_rate_mm_day(
        conv.convergence_mm_day, cslope, e, ra, conv.water_after_diffusion_mm,
        land_mask=land_mask, a4=a4,
    )
    final = evolve_column_water(
        conv.water_after_diffusion_mm, e, p, wind_u_m_s, wind_v_m_s,
        dx_m=dx_m, dy_m=dy_m, dt_days=dt_days,
        cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
        y_face_length_m=y_face_length_m, max_courant=max_courant,
    )
    return ColumnWaterVaporStep(
        final.water_mm, p, conv.convergence_mm_day, cslope,
        final.residual_mm, final.relative_residual,
        conv.diffusion_substeps, final.substeps,
    )


__all__ = [
    "ColumnEnergyStep",
    "ColumnWaterVaporStep",
    "ConvergenceMeasurement",
    "column_heat_capacity_j_m2_k",
    "diabatic_heating_rate_k_day",
    "evolve_column_energy",
    "evolve_column_water_vapor",
    "moisture_convergence_mm_day",
    "precipitation_rate_mm_day",
    "q2m_diagnostic",
    "slope_convergence_mm_day",
    "surface_relative_humidity_star",
    "t2m_diagnostic",
]
