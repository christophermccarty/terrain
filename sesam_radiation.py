"""SESAM cloud scheme -- Appendix A6 of Willeit et al. (2022).

Pure, read-only kernels for the CLIMBER-X/SESAM single-effective-cloud-layer
scheme, transcribed from Appendix A6 of

    Willeit, M., Ganopolski, A., Robinson, A., and Edwards, N. R.:
    GMD 15, 5905-5948 (2022), https://doi.org/10.5194/gmd-15-5905-2022
    (CC-BY 4.0; equations cited below as (A#) against that paper).

This is stage P5's first sub-deliverable (docs/SESAM_GAP_ANALYSIS.md section
7): cloud fraction, top height and optical thickness feed both the (A69)-
(A105) shortwave delta-Eddington scheme and the (A106)-(A117) longwave
two-stream scheme, neither of which is built yet. The supported PlanetSim
pipeline never calls these functions (the ``PlanetParams.enable_sesam_radiation``
gate is off by default), so this module has zero default-path climate impact
by construction.

**Equations verified directly against the source PDF** (2026-08-19, rendered
at 500 dpi via PyMuPDF since poppler's pdftoppm is unavailable here -- see
``sesam_thermo.py``'s module docstring for the same workaround) and, where
the paper's compressed notation left the functional form ambiguous, cross-
checked (read-only, per the section 5 licensing policy) against the running
reference implementation's ``src/atm/clouds.f90`` and ``src/atm/time_step.f90``
at github.com/cxesmc/climber-x. Findings recorded here so no later stage
re-derives them (mirrors the section 10 equation-semantics log style):

1. **``r*`` (the paper's un-numbered symbol in (A65)) is the skin-temperature-
   referenced relative humidity already built at stage P4**:
   ``sesam_thermo.surface_relative_humidity_star(qa, T*, p) = qa / qsat(T*)``.
   The Fortran computes the same ratio per surface type as
   ``min(qsat(T*,n), qa) / qsat(T*,n)`` (a saturation-safe form of the same
   quantity) and area-averages over surface-type fractions; this module
   reuses P4's already-tested single-column version rather than duplicating
   a per-surface-type average, matching the abstraction level P4 already
   established for the same symbol (its ``r2m`` blend uses the identical
   ``ra``/``r*`` pair). Callers must clip the result to ``<= 1`` themselves
   if supersaturated columns are possible; (A65) below does this internally.
2. **(A65)'s "when r* > ra" is descriptive, not a hard branch.** The paper's
   prose reads as a conditional, but the reference implementation always
   evaluates a symmetric ramp of ``(r* - ra)`` clipped to ``+/-c6cld`` --
   continuous through the crossover, with the crossover point sitting at the
   *midpoint* of the ramp rather than a discontinuity. This is the same
   "prose sounds like a threshold, the real formula is continuous" pattern
   already found once for (A44) at stage P4; implemented as the continuous
   form here for the same reason (a numerical model wants the differentiable
   version, and it is what the published Table A5 constants were actually
   tuned against).
3. **(A65) reuses (A62)'s exponent ``c4cld`` on ``ra``**, exactly as the
   paper's equation prints it (``ra^{c4cld}`` in both). The live reference
   repository's namelist has since decoupled these into two independently
   tuned exponents and drifted several other cloud constants away from the
   published Table A5 values (e.g. its ``c_cld_5`` default is 0.75 against
   the paper's printed 0.5). Per the section 6 calibration-window policy,
   this module follows the *paper's published Table A5 values only* --
   already transcribed in ``sesam_reference.py`` -- never the live repo's
   since-retuned namelist defaults; the Fortran is read here only to
   disambiguate functional form, not to import updated constants.
4. **Deliberately not carried over from the reference implementation**:
   the temporal exponential-relaxation smoothing (``0.1*new + 0.9*old``)
   applied to cloud fraction/height/thickness every timestep, the spatial
   smoothing pass on the low-cloud term, the ``cld_min``/``cld_max=0.95``
   clip, and the (A67) cloud-top-height physical bounds
   (``[zs+2500, htrop-1000]``). None of these are published Table A5
   constants; they are integration-time robustness choices for a stateful
   timestep loop, out of scope for this stateless kernel module (matching
   every prior SESAM stage's pure-function convention). A future wiring
   stage may add them explicitly. This module clips only where the paper's
   own algebra demands it (cloud fraction to [0, 1] as a definitional
   bound, not a tuned cap).
5. ``w700_mean`` (the (A63)/(A67) *mean* vertical velocity at 700 hPa, from
   the mean meridional circulation) and ``sigma_oro_m`` (sub-grid orography
   standard deviation, (A64)) are accepted as external inputs, following the
   same documented-placeholder convention P1 established for ``w700`` in its
   own RH scale height and P2's ``compute_wind`` established for
   ``sigma_oro_m``. Neither is fabricated here.
6. ``wsyn`` (700 hPa synoptic vertical velocity, (A57)) and ``Us`` (total
   surface wind, (A58)) are not recomputed: they are stage P3/P4's
   ``sesam_synoptic.synoptic_vertical_velocity`` and
   ``sesam_synoptic.total_wind_magnitude`` outputs, passed straight through.
"""

from __future__ import annotations

import numpy as np

from sesam_reference import value as _param


def _a6_defaults() -> dict[str, float]:
    """All A6 (clouds) constants from the published Table A5 pack."""
    return {
        "c1cld": _param("A6_clouds", "c1cld"),
        "c2cld": _param("A6_clouds", "c2cld"),
        "c3cld": _param("A6_clouds", "c3cld"),
        "c4cld": _param("A6_clouds", "c4cld"),
        "c5cld": _param("A6_clouds", "c5cld"),
        "c6cld": _param("A6_clouds", "c6cld"),
        "c7cld": _param("A6_clouds", "c7cld"),
        "c_weff": _param("A6_clouds", "c_weff"),
        "c_woro": _param("A6_clouds", "c_woro"),
        "H_pbl": _param("A6_clouds", "H_pbl"),
        "c1hcld": _param("A6_clouds", "c1hcld"),
        "c2hcld": _param("A6_clouds", "c2hcld"),
        "c3hcld": _param("A6_clouds", "c3hcld"),
        "c1tau": _param("A6_clouds", "c1tau"),
        "c2tau": _param("A6_clouds", "c2tau"),
        "c3tau": _param("A6_clouds", "c3tau"),
        "c4tau": _param("A6_clouds", "c4tau"),
    }


_EPS = 1e-12


def _check_2d(name: str, value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D (H, W) field")
    return arr


def _matching(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{name_a} and {name_b} must share a shape")


# ---------------------------------------------------------------------------
# (A63)-(A64) effective cloud-level vertical velocity
# ---------------------------------------------------------------------------


def orographic_vertical_velocity(
    surface_wind_m_s: np.ndarray,
    sigma_oro_m: np.ndarray,
    a6: dict[str, float] | None = None,
) -> np.ndarray:
    """(A64) ``woro = c_woro * Us * sigma_oro``."""
    params = a6 or _a6_defaults()
    us = _check_2d("surface_wind_m_s", surface_wind_m_s)
    sigma = _check_2d("sigma_oro_m", sigma_oro_m)
    _matching("surface_wind_m_s", us, "sigma_oro_m", sigma)
    return params["c_woro"] * us * sigma


def effective_cloud_vertical_velocity(
    w700_mean_m_s: np.ndarray,
    synoptic_vertical_velocity_m_s: np.ndarray,
    orographic_vertical_velocity_m_s: np.ndarray,
    a6: dict[str, float] | None = None,
) -> np.ndarray:
    """(A63) ``weff = w(700hPa) + c_weff * (wsyn + woro)``."""
    params = a6 or _a6_defaults()
    w700 = _check_2d("w700_mean_m_s", w700_mean_m_s)
    wsyn = _check_2d("synoptic_vertical_velocity_m_s", synoptic_vertical_velocity_m_s)
    woro = _check_2d("orographic_vertical_velocity_m_s", orographic_vertical_velocity_m_s)
    _matching("w700_mean_m_s", w700, "synoptic_vertical_velocity_m_s", wsyn)
    _matching("w700_mean_m_s", w700, "orographic_vertical_velocity_m_s", woro)
    return w700 + params["c_weff"] * (wsyn + woro)


# ---------------------------------------------------------------------------
# (A61)-(A62), (A65)-(A66) cloud fraction
# ---------------------------------------------------------------------------


def humidity_cloud_fraction(
    near_surface_rh: np.ndarray,
    weff_m_s: np.ndarray,
    a6: dict[str, float] | None = None,
) -> np.ndarray:
    """(A62) ``f_cld^r = (c1cld + c2cld*tanh(c3cld*weff)) * ra^c4cld``."""
    params = a6 or _a6_defaults()
    ra = np.clip(_check_2d("near_surface_rh", near_surface_rh), 0.0, 1.0)
    weff = _check_2d("weff_m_s", weff_m_s)
    _matching("near_surface_rh", ra, "weff_m_s", weff)
    envelope = params["c1cld"] + params["c2cld"] * np.tanh(params["c3cld"] * weff)
    return envelope * np.power(ra, params["c4cld"])


def freezedry_factor(qa_kg_kg: np.ndarray, a6: dict[str, float] | None = None) -> np.ndarray:
    """(A66) ``f_freezedry = clip(0.1 + 0.9*qa/c7cld, 0, 1)``."""
    params = a6 or _a6_defaults()
    qa = _check_2d("qa_kg_kg", qa_kg_kg)
    raw = 0.1 + 0.9 * qa / max(params["c7cld"], _EPS)
    return np.clip(raw, 0.0, 1.0)


def inversion_low_cloud_fraction(
    near_surface_rh: np.ndarray,
    skin_referenced_rh: np.ndarray,
    qa_kg_kg: np.ndarray,
    a6: dict[str, float] | None = None,
) -> np.ndarray:
    """(A65) inversion/low-cloud fraction ``f_cld^low``.

    ``f_cld^low = c5cld * f_freezedry * (dr + c6cld)/(2*c6cld) * ra^c4cld``,
    ``dr = clip(r* - ra, -c6cld, c6cld)`` -- see module docstring note 2 for
    why this is the always-applied continuous ramp rather than a hard branch
    on ``r* > ra``.
    """
    params = a6 or _a6_defaults()
    ra = np.clip(_check_2d("near_surface_rh", near_surface_rh), 0.0, 1.0)
    rstar = np.clip(_check_2d("skin_referenced_rh", skin_referenced_rh), 0.0, 1.0)
    qa = _check_2d("qa_kg_kg", qa_kg_kg)
    _matching("near_surface_rh", ra, "skin_referenced_rh", rstar)
    _matching("near_surface_rh", ra, "qa_kg_kg", qa)
    f_freezedry = freezedry_factor(qa, a6=params)
    c6 = params["c6cld"]
    dr = np.clip(rstar - ra, -c6, c6)
    fr = f_freezedry * (dr + c6) / (2.0 * c6 + _EPS)
    return params["c5cld"] * fr * np.power(ra, params["c4cld"])


def total_cloud_fraction(
    humidity_cloud_fraction_value: np.ndarray,
    inversion_low_cloud_fraction_value: np.ndarray,
) -> np.ndarray:
    """(A61) ``fcld = 1 - (1-f_cld^r)*(1-f_cld^low)``, clipped to [0, 1].

    The clip is a definitional bound on a fraction, not a tuned cap -- see
    module docstring note 4 for why the reference implementation's separate
    ``cld_min``/``cld_max=0.95`` calibration clip is *not* applied here.
    """
    f_r = _check_2d("humidity_cloud_fraction_value", humidity_cloud_fraction_value)
    f_low = _check_2d("inversion_low_cloud_fraction_value", inversion_low_cloud_fraction_value)
    _matching(
        "humidity_cloud_fraction_value", f_r,
        "inversion_low_cloud_fraction_value", f_low,
    )
    fcld = 1.0 - (1.0 - f_r) * (1.0 - f_low)
    return np.clip(fcld, 0.0, 1.0)


# ---------------------------------------------------------------------------
# (A67) cloud top height
# ---------------------------------------------------------------------------


def cloud_top_height_m(
    tropopause_height_m: np.ndarray,
    w700_mean_m_s: np.ndarray,
    a6: dict[str, float] | None = None,
) -> np.ndarray:
    """(A67) ``Hcld = c1hcld + c2hcld*HT*(1 + c3hcld*w(700hPa))``.

    Cloud base is separately assumed to coincide with the PBL top (``H_pbl``,
    a fixed constant per the paper -- exposed as ``a6["H_pbl"]``, not a
    function here since it takes no inputs). No physical bounds are applied
    (module docstring note 4); ``Hcld`` can exceed ``HT`` or fall below the
    surface for pathological inputs and callers needing a bounded value
    should clip explicitly.
    """
    params = a6 or _a6_defaults()
    ht = _check_2d("tropopause_height_m", tropopause_height_m)
    w700 = _check_2d("w700_mean_m_s", w700_mean_m_s)
    _matching("tropopause_height_m", ht, "w700_mean_m_s", w700)
    return params["c1hcld"] + params["c2hcld"] * ht * (1.0 + params["c3hcld"] * w700)


# ---------------------------------------------------------------------------
# (A68) cloud optical thickness
# ---------------------------------------------------------------------------


def cloud_optical_thickness(
    t2m_k: np.ndarray,
    cloud_fraction: np.ndarray,
    column_water_kg_m2: np.ndarray,
    a6: dict[str, float] | None = None,
    t0_k: float = 273.15,
) -> np.ndarray:
    """(A68) ``tau_cld = c3tau*[1+tanh(-(T2m-T0-c1tau)/c2tau)]*(fcld*Qq)^c4tau``.

    Matches the reference implementation's use of the grid-mean ``T2m``
    ((A41) diagnostic, ``sesam_thermo.t2m_diagnostic``) for "surface air
    temperature", not the per-surface-type skin temperature. The sulfate-
    aerosol indirect-effect modifier the paper mentions afterward is out of
    scope (no aerosol field exists in PlanetSim); ``ftemp`` is capped at 1
    matching the reference implementation, since (A68) as printed has no
    such cap and can otherwise slightly exceed the ``[0, 2]`` range of
    ``1+tanh(...)`` under floating-point edge cases.
    """
    params = a6 or _a6_defaults()
    t2m = _check_2d("t2m_k", t2m_k)
    fcld = np.clip(_check_2d("cloud_fraction", cloud_fraction), 0.0, 1.0)
    qq = np.maximum(_check_2d("column_water_kg_m2", column_water_kg_m2), 0.0)
    _matching("t2m_k", t2m, "cloud_fraction", fcld)
    _matching("t2m_k", t2m, "column_water_kg_m2", qq)
    tcldm = t2m - t0_k - params["c1tau"]
    ftemp = np.minimum(1.0 + np.tanh(-tcldm / params["c2tau"]), 1.0)
    return params["c3tau"] * ftemp * np.power(fcld * qq, params["c4tau"])


__all__ = [
    "orographic_vertical_velocity",
    "effective_cloud_vertical_velocity",
    "humidity_cloud_fraction",
    "freezedry_factor",
    "inversion_low_cloud_fraction",
    "total_cloud_fraction",
    "cloud_top_height_m",
    "cloud_optical_thickness",
]
