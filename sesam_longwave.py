"""SESAM longwave radiation -- Appendix A8 of Willeit et al. (2022).

Pure, read-only kernels for the CLIMBER-X/SESAM 15-level two-stream longwave
scheme, transcribed from Appendix A8 of

    Willeit, M., Ganopolski, A., Robinson, A., and Edwards, N. R.:
    GMD 15, 5905-5948 (2022), https://doi.org/10.5194/gmd-15-5905-2022
    (CC-BY 4.0; equations cited below as (A#) against that paper).

This is stage P5's third sub-deliverable (docs/SESAM_GAP_ANALYSIS.md section
7). The supported PlanetSim pipeline never calls these functions (the
``PlanetParams.enable_sesam_radiation`` gate is off by default), so this
module has zero default-path climate impact by construction.

**Equations verified directly against the source PDF** (2026-08-19, 280-600
dpi PyMuPDF renders) and, for the absorber-mass-path closure the GMD paper
states only as an abstract integral (A114)-(A116) with no worked
discretization, against a second, independently-citable published source
identified with the user's help: **Petoukhov, Ganopolski & Claussen (2003),
"POTSDAM -- A Set of Atmosphere Statistical-Dynamical Models: Theoretical
Background", PIK Report No. 81, ISSN 1436-0179** -- the direct scientific
ancestor of SESAM (POTSDAM-2 is CLIMBER-2's atmosphere module; SESAM is its
CLIMBER-X descendant). This is a different, and stronger, situation than the
notational disambiguations used elsewhere in P5: the GMD paper's own
transmission-function forms (A108)-(A112) are confirmed the direct algebraic
descendants of PIK-81's equations (6.4)-(6.8) (same functional families,
refined constants -- e.g. GMD's 3-term water-vapour sum vs. PIK-81's 1-term
form), and PIK-81 sec. 6.1.2 states in prose that its absorber-mass integral
(its eq. 6.1, the same integral as (A114)-(A116)) is evaluated "on the
assumption that the vertical profiles are quasi-exponential for pressure and
air density" -- i.e. the closure this module uses is independently published,
not merely inferred from the reference Fortran implementation.

**Findings recorded here so no later stage re-derives them**:

1. **The absorber-mass integrals (A114)-(A116) use stage P1's own (A1)
   exponential reference pressure profile** (``sesam_vertical.pressure_profile``,
   ``p(z) = p0*exp(-z/Ha)``) as the quasi-exponential density/pressure closure
   PIK-81 sec. 6.1.2 describes -- not a separately invented approximation.
   For a well-mixed gas (CO2) this makes the mass-path integral *exact* given
   that reference profile, not merely approximate (see
   :func:`co2_mass_path_g_cm2`). Water vapour and ozone are integrated
   numerically (trapezoidal, cumulative over the level grid) against their
   own real profiles rather than a further exponential shortcut, since P1
   already supplies a real (non-exponential-shortcut) humidity profile and
   this module has no equivalent shortcut to offer for ozone.
2. **(A111)'s printed exponents beta1^CO2 and beta2^CO2 are the same symbol**
   (Table A8 transcribes only one ``betaco2``), confirmed by PIK-81's older
   version of this formula (its eq. 6.7), which uses one ``betaCO2`` in both
   the numerator and denominator terms. Implemented here with the single
   Table A8 value applied to both.
3. **The Fortran reference implementation's comment labels for PIK-81
   equation numbers are off by one for CO2/ozone** (it cites "(6.6)" for the
   formula that is actually PIK-81's (6.7), and "(6.7)" for what is actually
   (6.8)) -- resolved by reading PIK-81 directly rather than trusting the
   code comments; the *constants and structure* used here are the GMD 2022
   Table A8 values, not PIK-81's (which have since been retuned across two
   model generations), per the section 6 calibration-window policy.
4. **CO2's ppm-to-column-mass conversion is a standard atmospheric-chemistry
   calculation** (mass mixing ratio via the CO2/air molar mass ratio, times
   the hydrostatic column mass ``p0/g``) but is **not independently verified
   against the paper's exact "cm" absorber-mass convention** the way the
   water-vapour g/cm^2 <-> cm liquid-depth identity is (that one is a
   dimensional identity, not a convention choice). Flagged here the same way
   ``sesam_reference.flagged_transcriptions()`` flags unverified constants --
   treat absolute CO2 transmission magnitudes as provisional until checked
   against a real greenhouse-effect-decomposition benchmark
   (``TABLE_MAIN_LW_ABSORBERS`` in ``sesam_reference.py``).
5. **Ozone needs a real climatology, unlike the shortwave stage** (docstring
   of ``sesam_shortwave.py`` found the opposite there). This module accepts
   an already-computed ozone mass-mixing-ratio profile as an external input
   (module docstring convention shared with every prior SESAM stage) rather
   than fabricating one; the ozone-climatology format (constant vs. zonal
   climatology vs. a real prescribed dataset) is still an open decision.
6. **(A106)/(A107) are continuous integrals with no worked discretization in
   the paper.** Discretized here as a direct Riemann-Stieltjes sum over the
   level grid (not sourced from PIK-81 or the Fortran): the paper's own
   boundary conditions confirm the choice is correct -- the discretized
   ``F_down`` vanishes at the top level (no downward LW source above the
   atmosphere) and the discretized ``F_up`` at the surface level reduces
   exactly to ``Bs`` (the surface's own blackbody emission), both checked as
   tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sesam_reference import value as _param

_STEFAN_BOLTZMANN = 5.670374419e-8  # W m^-2 K^-4
_CO2_MOLAR_MASS_OVER_AIR = 44.01 / 28.97  # dimensionless


def _a8_defaults() -> dict[str, float]:
    """All A8 (longwave) constants from the published Table A8 pack."""
    return {
        "beta0": _param("A8_longwave", "beta0"),
        "a1wv_lw": _param("A8_longwave", "a1wv_lw"),
        "a2wv_lw": _param("A8_longwave", "a2wv_lw"),
        "a3wv_lw": _param("A8_longwave", "a3wv_lw"),
        "beta1wv": _param("A8_longwave", "beta1wv"),
        "beta2wv": _param("A8_longwave", "beta2wv"),
        "beta3wv": _param("A8_longwave", "beta3wv"),
        "k_wv": _param("A8_longwave", "k_wv"),
        "a0co2": _param("A8_longwave", "a0co2"),
        "a1co2": _param("A8_longwave", "a1co2"),
        "beta_co2": _param("A8_longwave", "beta_co2"),
        "k_co2": _param("A8_longwave", "k_co2"),
        "a0o3": _param("A8_longwave", "a0o3"),
        "beta_o3": _param("A8_longwave", "beta_o3"),
        "k_o3": _param("A8_longwave", "k_o3"),
    }


def _check_2d(name: str, value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D (H, W) field")
    return arr


def _check_profile(name: str, value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be a 3-D (N, H, W) level-profile field")
    return arr


def _matching(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{name_a} and {name_b} must share a shape")


def blackbody_emission_w_m2(temperature_k: np.ndarray) -> np.ndarray:
    """``B(z) = sigma*T(z)^4`` (used for both (A106)/(A107))."""
    t = np.asarray(temperature_k, dtype=np.float64)
    return _STEFAN_BOLTZMANN * np.power(t, 4)


# ---------------------------------------------------------------------------
# (A108)-(A113) transmission functions
# ---------------------------------------------------------------------------


def water_vapor_lw_transmission(
    mass_path_g_cm2: np.ndarray, a8: dict[str, float] | None = None
) -> np.ndarray:
    """(A110) water-vapour LW transmission.

    ``Dwv = 1 / (1 + a1*(b0*M)^b1 + a2*(b0*M)^b2 + a3*(b0*M)^b3)``.
    """
    params = a8 or _a8_defaults()
    m = np.maximum(_check_2d("mass_path_g_cm2", mass_path_g_cm2), 0.0)
    x = params["beta0"] * m
    denom = (
        1.0
        + params["a1wv_lw"] * np.power(x, params["beta1wv"])
        + params["a2wv_lw"] * np.power(x, params["beta2wv"])
        + params["a3wv_lw"] * np.power(x, params["beta3wv"])
    )
    return 1.0 / denom


def co2_lw_transmission(
    mass_path_g_cm2: np.ndarray, a8: dict[str, float] | None = None
) -> np.ndarray:
    """(A111) CO2 LW transmission (docstring note 2: one beta_co2 used in
    both terms, matching PIK-81's older single-exponent form).

    ``DCO2 = (1 - min(0.2, 0.1*(M/1000)^2)) * (1 + a0*a1*(b0*M)^b) / (1 + a0*(b0*M)^b)``.
    The ``min(0.2, ...)`` cap is not in the printed equation but guards the
    same numerical runaway the reference implementation caps (module
    docstring note 3 territory: a robustness detail, not a retuned
    constant).
    """
    params = a8 or _a8_defaults()
    m = np.maximum(_check_2d("mass_path_g_cm2", mass_path_g_cm2), 0.0)
    x = params["beta0"] * m
    forcing_increase = 1.0 - np.minimum(0.2, 0.1 * np.power(m / 1000.0, 2))
    power_term = np.power(x, params["beta_co2"])
    ratio = (1.0 + params["a0co2"] * params["a1co2"] * power_term) / (
        1.0 + params["a0co2"] * power_term
    )
    return forcing_increase * ratio


def ozone_lw_transmission(
    mass_path_g_cm2: np.ndarray, a8: dict[str, float] | None = None
) -> np.ndarray:
    """(A112) ``DO3 = 1 - a_O3*M_O3^beta_O3``, clipped to [0, 1]."""
    params = a8 or _a8_defaults()
    m = np.maximum(_check_2d("mass_path_g_cm2", mass_path_g_cm2), 0.0)
    return np.clip(1.0 - params["a0o3"] * np.power(m, params["beta_o3"]), 0.0, 1.0)


def cloud_lw_transmission(
    level_gap_m: np.ndarray,
    cloud_geometric_thickness_m: np.ndarray,
    cloud_optical_thickness: np.ndarray,
    inside_cloud: np.ndarray,
) -> np.ndarray:
    """(A113) ``Dcld = exp(-|z-z'|/(Htop-Hbase) * tau_cld)``, 1.0 outside
    cloud layers (``inside_cloud`` is a boolean mask over the level-pair
    gap, true only when both endpoints fall within the cloud layer)."""
    dz = np.abs(_check_2d("level_gap_m", level_gap_m))
    thickness = np.maximum(
        _check_2d("cloud_geometric_thickness_m", cloud_geometric_thickness_m), 1.0
    )
    tau = np.maximum(_check_2d("cloud_optical_thickness", cloud_optical_thickness), 0.0)
    inside = np.asarray(inside_cloud, dtype=bool)
    _matching("level_gap_m", dz, "cloud_geometric_thickness_m", thickness)
    _matching("level_gap_m", dz, "cloud_optical_thickness", tau)
    decay = np.exp(-dz / thickness * tau)
    return np.where(inside, decay, 1.0)


def combined_transmission(
    water_transmission: np.ndarray,
    co2_transmission: np.ndarray,
    ozone_transmission: np.ndarray,
    cloud_transmission: np.ndarray | float = 1.0,
) -> np.ndarray:
    """(A108)/(A109): ``D^cs = Dwv*DCO2*DO3`` (cloud_transmission=1), or
    ``D^cld = Dwv*DCO2*DO3*Dcld``."""
    dwv = _check_2d("water_transmission", water_transmission)
    dco2 = _check_2d("co2_transmission", co2_transmission)
    do3 = _check_2d("ozone_transmission", ozone_transmission)
    _matching("water_transmission", dwv, "co2_transmission", dco2)
    _matching("water_transmission", dwv, "ozone_transmission", do3)
    return dwv * dco2 * do3 * cloud_transmission


# ---------------------------------------------------------------------------
# (A114)-(A116) absorber mass paths
# ---------------------------------------------------------------------------


def water_vapor_mass_path_g_cm2(
    specific_humidity_profile_kg_kg: np.ndarray,
    pressure_profile_pa: np.ndarray,
    air_density_profile_kg_m3: np.ndarray,
    levels_m: np.ndarray,
    surface_pressure_pa: np.ndarray | float,
    level_index_1: int,
    level_index_2: int,
    a8: dict[str, float] | None = None,
) -> np.ndarray:
    """(A114) water-vapour mass path between two levels of the grid.

    ``M_wv(z1,z2) = integral q(z)*rho(z)*(p(z)/p(0))^k_wv dz``, evaluated by
    trapezoidal quadrature over the real (A13)/(A15) profile P1 supplies
    (docstring note 1) rather than a further exponential shortcut, then
    converted g/m^2 -> g/cm^2 (the same identity as
    ``sesam_shortwave.column_water_path_g_cm2``, since precipitable-water
    depth in cm is numerically the column mass in g/cm^2).
    """
    params = a8 or _a8_defaults()
    q = _check_profile("specific_humidity_profile_kg_kg", specific_humidity_profile_kg_kg)
    p = _check_profile("pressure_profile_pa", pressure_profile_pa)
    rho = _check_profile("air_density_profile_kg_m3", air_density_profile_kg_m3)
    levels = np.asarray(levels_m, dtype=np.float64)
    if levels.ndim != 1 or levels.shape[0] != q.shape[0]:
        raise ValueError("levels_m must be 1-D and match the profile's level axis")
    _matching("specific_humidity_profile_kg_kg", q, "pressure_profile_pa", p)
    _matching("specific_humidity_profile_kg_kg", q, "air_density_profile_kg_m3", rho)
    p0 = np.broadcast_to(_check_2d("surface_pressure_pa", surface_pressure_pa), q.shape[1:])

    lo, hi = sorted((int(level_index_1), int(level_index_2)))
    weight = q * rho * np.power(p / p0[None, :, :], params["k_wv"])
    integral = np.zeros(q.shape[1:], dtype=np.float64)
    for k in range(lo, hi):
        integral += 0.5 * (weight[k] + weight[k + 1]) * abs(levels[k + 1] - levels[k])
    return integral * 0.1  # kg/m^2 -> g/cm^2


def co2_mass_path_g_cm2(
    co2_ppm: float,
    pressure_ratio_1: np.ndarray,
    pressure_ratio_2: np.ndarray,
    surface_pressure_pa: np.ndarray | float,
    gravity: float,
    a8: dict[str, float] | None = None,
) -> np.ndarray:
    """(A115) CO2 mass path between two levels, exact under P1's (A1)
    exponential reference pressure profile (docstring note 1): well-mixed
    CO2 makes the (A115) integral analytically closed-form given
    ``p(z)/p0 = exp(-z/Ha)`` --

    ``M_CO2(z1,z2) = chi_co2 * (p0/g) / (k_co2+1) * [(p1/p0)^(k_co2+1) - (p2/p0)^(k_co2+1)]``

    where ``chi_co2`` is the CO2 mass mixing ratio. Docstring note 4: the
    ppm-to-mass-mixing-ratio conversion is a standard calculation but is
    *not* verified against the paper's own "cm" absorber-mass convention --
    treat absolute magnitudes from this function as provisional.
    """
    params = a8 or _a8_defaults()
    r1 = _check_2d("pressure_ratio_1", pressure_ratio_1)
    r2 = _check_2d("pressure_ratio_2", pressure_ratio_2)
    p0 = np.broadcast_to(_check_2d("surface_pressure_pa", surface_pressure_pa), r1.shape)
    _matching("pressure_ratio_1", r1, "pressure_ratio_2", r2)
    chi_co2 = float(co2_ppm) * 1e-6 * _CO2_MOLAR_MASS_OVER_AIR
    k1 = params["k_co2"] + 1.0
    column_kg_m2 = (
        chi_co2 * (p0 / float(gravity)) / k1 * (np.power(r1, k1) - np.power(r2, k1))
    )
    return np.abs(column_kg_m2) * 0.1  # kg/m^2 -> g/cm^2


def ozone_mass_path_g_cm2(
    ozone_mixing_ratio_profile_kg_kg: np.ndarray,
    pressure_profile_pa: np.ndarray,
    air_density_profile_kg_m3: np.ndarray,
    levels_m: np.ndarray,
    surface_pressure_pa: np.ndarray | float,
    level_index_1: int,
    level_index_2: int,
    a8: dict[str, float] | None = None,
) -> np.ndarray:
    """(A116) ozone mass path, same trapezoidal structure as
    :func:`water_vapor_mass_path_g_cm2` (docstring note 1/5: ozone has no
    exponential-profile shortcut here, and its mixing-ratio profile is an
    external input pending the ozone-climatology decision)."""
    params = a8 or _a8_defaults()
    o3 = _check_profile("ozone_mixing_ratio_profile_kg_kg", ozone_mixing_ratio_profile_kg_kg)
    p = _check_profile("pressure_profile_pa", pressure_profile_pa)
    rho = _check_profile("air_density_profile_kg_m3", air_density_profile_kg_m3)
    levels = np.asarray(levels_m, dtype=np.float64)
    if levels.ndim != 1 or levels.shape[0] != o3.shape[0]:
        raise ValueError("levels_m must be 1-D and match the profile's level axis")
    _matching("ozone_mixing_ratio_profile_kg_kg", o3, "pressure_profile_pa", p)
    _matching("ozone_mixing_ratio_profile_kg_kg", o3, "air_density_profile_kg_m3", rho)
    p0 = np.broadcast_to(_check_2d("surface_pressure_pa", surface_pressure_pa), o3.shape[1:])

    lo, hi = sorted((int(level_index_1), int(level_index_2)))
    weight = o3 * rho * np.power(p / p0[None, :, :], params["k_o3"])
    integral = np.zeros(o3.shape[1:], dtype=np.float64)
    for k in range(lo, hi):
        integral += 0.5 * (weight[k] + weight[k + 1]) * abs(levels[k + 1] - levels[k])
    return integral * 0.1  # kg/m^2 -> g/cm^2


# ---------------------------------------------------------------------------
# (A106)/(A107) flux assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LongwaveRadiation:
    """Assembled (A106)/(A107) fluxes at every level, ``(N, H, W)``."""

    downward_w_m2: np.ndarray
    upward_w_m2: np.ndarray


def longwave_flux_profile(
    blackbody_profile_w_m2: np.ndarray,
    surface_blackbody_w_m2: np.ndarray,
    transmission_matrix: np.ndarray,
) -> LongwaveRadiation:
    """(A106)/(A107) discretized as a Riemann-Stieltjes sum (module
    docstring note 6) over an ``N``-level grid, level 0 = surface, level
    N-1 = top of atmosphere.

    ``transmission_matrix`` is ``(N, N, H, W)``, symmetric, unit diagonal:
    ``transmission_matrix[i, j]`` is the ``D^cs``/``D^cld`` transmission
    between level ``i`` and level ``j`` (from :func:`combined_transmission`,
    accumulated over the levels between them).
    """
    b = _check_profile("blackbody_profile_w_m2", blackbody_profile_w_m2)
    bs = _check_2d("surface_blackbody_w_m2", surface_blackbody_w_m2)
    d = np.asarray(transmission_matrix, dtype=np.float64)
    n = b.shape[0]
    if d.shape != (n, n) + b.shape[1:]:
        raise ValueError(
            f"transmission_matrix must have shape ({n}, {n}, H, W) matching the level profile"
        )

    down = np.zeros_like(b)
    up = np.zeros_like(b)

    for level in range(n):
        # (A106): F_down(level) = B(level) - B(top)*D(level,top) + sum_{k=level}^{N-2} D(level,k+1)*(B(k+1)-B(k))
        acc = b[level] - b[n - 1] * d[level, n - 1]
        for k in range(level, n - 1):
            acc += d[level, k + 1] * (b[k + 1] - b[k])
        down[level] = acc

        # (A107): F_up(level) = B(level) + [Bs-B(0)]*D(0,level) - sum_{k=0}^{level-1} D(k,level)*(B(k+1)-B(k))
        acc = b[level] + (bs - b[0]) * d[0, level]
        for k in range(0, level):
            acc -= d[k, level] * (b[k + 1] - b[k])
        up[level] = acc

    return LongwaveRadiation(downward_w_m2=down, upward_w_m2=up)


def sky_combine(
    clear_sky_value: np.ndarray, cloudy_value: np.ndarray, cloud_fraction: np.ndarray
) -> np.ndarray:
    """(A106)/(A107) cloud-fraction weighting: ``fcld*cloudy + (1-fcld)*clear``."""
    cs = np.asarray(clear_sky_value, dtype=np.float64)
    cld = np.asarray(cloudy_value, dtype=np.float64)
    f = np.clip(np.asarray(cloud_fraction, dtype=np.float64), 0.0, 1.0)
    return f * cld + (1.0 - f) * cs


__all__ = [
    "blackbody_emission_w_m2",
    "water_vapor_lw_transmission",
    "co2_lw_transmission",
    "ozone_lw_transmission",
    "cloud_lw_transmission",
    "combined_transmission",
    "water_vapor_mass_path_g_cm2",
    "co2_mass_path_g_cm2",
    "ozone_mass_path_g_cm2",
    "LongwaveRadiation",
    "longwave_flux_profile",
    "sky_combine",
]
