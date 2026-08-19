"""SESAM (A10) tropopause radiative closure -- the last piece of stage P5.

Stage P1 (``sesam_vertical.py``) already implements (A10)/(A11)'s algebraic
form -- ``tropopause_tendency`` and ``tropopause_shape_s`` -- but left
``Rstr,net`` (the stratospheric net-radiation residual) as an accepted
external input, since computing it needs the A8 longwave scheme that wasn't
built until this stage (docs/SESAM_GAP_ANALYSIS.md section 7, P1 docstring
item 2). This module is the combinator that closes that gap: it wires
``sesam_longwave.py``'s assembled fluxes and ``sesam_shortwave.py``'s TOA
insolation into ``Rstr,net``, then advances ``HT`` (the tropopause height)
by one timestep. Also ships the constant-global ozone climatology the (A8)
longwave module has needed as an open decision since its own P5 sub-
deliverable (docs/SESAM_GAP_ANALYSIS.md line ~868).

The supported PlanetSim pipeline never calls these functions (the same
``PlanetParams.enable_sesam_radiation`` gate as the rest of P5's radiation
modules; this module adds no new gate), so it has zero default-path climate
impact by construction.

**Rstr,net closure -- read-only disambiguation, not a new scientific method**:
The paper states (A10)'s ``Rstr,net`` only in prose -- "the balance of
longwave radiation and the shortwave radiation absorbed by ozone" (page 5926)
-- with no worked formula for computing it from the model's own fields. This
is the same class of gap P5's other sub-deliverables hit (the (A114)-(A116)
absorber-mass discretization, (A97)'s sign, (A87)/(A88)'s band labels): the
paper defines the physical quantity but not its numerical construction from
the model state, so the reference implementation
(github.com/cxesmc/climber-x, read-only per docs/SESAM_GAP_ANALYSIS.md
section 5) is the disambiguation source, same discipline as those. Its
``src/atm/atm_model.f90`` (lines ~170-176) computes:

    rb_str = lwr_top - lwr_tro + frac_vu * (1 - ITF_O3) * swr_dw_top

where ``lwr_top``/``lwr_tro`` are net-downward LW flux (``F_down - F_up``) at
the model top and at the tropopause level respectively (confirmed by reading
``src/atm/lwr.f90``'s own ``flwr_top``/``flwr_tro`` definitions: ``flwr_top =
-F_up(top)`` and, since ``F_down(top) = 0`` by the (A106) boundary condition
``sesam_longwave.py`` already proves as a test, this is exactly
``net(top) = F_down(top) - F_up(top)``; ``flwr_tro = F_down(tropopause) -
F_up(tropopause)`` directly). This module reproduces that structure using
:func:`sesam_longwave.longwave_radiation`'s own flux-profile output rather
than re-deriving it.

**One deliberate deviation from the Fortran, flagged**: its ozone-absorption
term hardcodes ``0.02`` with the code comment ``"0.02=1-ITF_O3"`` -- but the
*published* Table A5 constant is ``I_O3,vu = 0.96`` (``1 - 0.96 = 0.04``,
not 0.02), the same constant ``sesam_shortwave.I_O3_VU`` already uses
elsewhere in P5. The Fortran's magic number cannot be traced to that
published constant and is not independently cited anywhere in the source
tree. Per the section 6 calibration-window policy (do not follow a live
reference-repo constant that has drifted from the published paper), this
module uses ``sesam_shortwave.I_O3_VU`` (0.96) instead of the Fortran's 0.02.

**c1tp's per-day folding -- resolved this stage**: ``sesam_reference.py``
flagged ``c1tp`` (100 m^3 W^-1) as unresolved because dimensional analysis
alone leaves an implicit time unit ambiguous (``c1tp * Rstr,net`` computes to
metres, not metres/time, so *some* per-timestep convention is baked into the
printed value). The paper's own Conclusions section states outright: "the
use of a daily time step for most processes" (page 5924) -- CLIMBER-X's
native timestep is one day. Combined with there being no other timestep
convention anywhere else in the paper's SESAM description, this is read as
confirming ``c1tp``'s value already assumes a 1-day step: the raw
``tropopause_tendency`` output is metres of tropopause-height change *per
simulated day*, scaled by ``dt_days`` here exactly as every other SESAM
stage's own ``dt_days`` substep convention already works
(``sesam_thermo.py``'s (A40)/(A44) stepping, ``diagnose_sesam_thermo.py``'s
``--dt-days``).

**Ozone climatology -- the "constant" branch of the three-way decision**
(constant / zonal climatology / real prescribed dataset, raised at P5's
scoping and left open through both the shortwave and longwave sub-
deliverables): SESAM's own real input is a prescribed 3-D time-varying
CMIP6-ensemble-mean field (``sesam_shortwave.py`` docstring finding 4) that
this project has no access to and would have to fabricate wholesale to
adopt -- out of scope for closing P5. The Fortran reference confirms there is
no simpler built-in fallback either (``atm%o3`` is allocated but never
assigned inside ``atm_model.f90``; it is populated from an external boundary
module this repo does not have). :func:`standard_ozone_mixing_ratio_profile_kgkg`
is therefore an explicit engineering placeholder, not sourced from the paper
or the Fortran: a Gaussian ozone layer (textbook stratospheric-ozone shape,
not this project's invention) centred at 25 km with a 7 km half-width,
normalised to a 300 Dobson-Unit column (a standard, uncontroversial global-
mean figure) via the DU definition ``1 DU = 2.1415e-5 kg O3 m^-2`` (0.01 mm
STP ozone; standard atmospheric-chemistry conversion, not fitted to this
project's data). It is global and time-invariant -- no latitude or seasonal
dependence -- flagged here exactly like ``sesam_reference.flagged_transcriptions()``
flags provisional constants elsewhere in P5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import sesam_vertical as _sv
from sesam_shortwave import FRAC_VU, I_O3_VU

# Reference-Fortran grid constant (github.com/cxesmc/climber-x,
# src/atm/lwr.f90: ``real(wp), parameter :: z_atm = 30.e3_wp``) -- a plain
# model-top height, not a scientific method, so citing it for grid
# configuration is the same disambiguation class already used for frac_vu/mu0
# in sesam_shortwave.py.
MODEL_TOP_M = 30000.0

# Standard DU -> kg m^-2 conversion (0.01 mm STP ozone at standard T/p).
_DU_TO_KG_M2 = 2.1415e-5


def _trapezoid(values: np.ndarray, x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Manual trapezoidal quadrature (``numpy.trapz``/``trapezoid`` moved
    across numpy versions; every other SESAM module already sums 0.5*(a+b)*dz
    by hand for the same reason -- see sesam_longwave.py's mass-path loops)."""
    v = np.moveaxis(values, axis, 0)
    dx = np.diff(np.asarray(x, dtype=np.float64))
    seg = 0.5 * (v[1:] + v[:-1]) * dx.reshape((-1,) + (1,) * (v.ndim - 1))
    return np.sum(seg, axis=0)


def standard_ozone_mixing_ratio_profile_kgkg(
    levels_m: np.ndarray,
    air_density_profile_kg_m3: np.ndarray,
    total_column_du: float = 300.0,
    peak_height_m: float = 25000.0,
    half_width_m: float = 7000.0,
) -> np.ndarray:
    """Constant-global ozone mixing-ratio profile (module docstring's
    "ozone climatology" section) -- a Gaussian layer in height, normalised
    by trapezoidal quadrature so its column mass path matches
    ``total_column_du`` Dobson units exactly, then converted to a mixing
    ratio via the supplied density profile (``mmr = rho_o3 / rho_air``).

    Shape ``(N, H, W)`` if ``air_density_profile_kg_m3`` is, else ``(N,)``.
    """
    levels = np.asarray(levels_m, dtype=np.float64)
    if levels.ndim != 1:
        raise ValueError("levels_m must be 1-D")
    rho = np.asarray(air_density_profile_kg_m3, dtype=np.float64)
    shape_z = (levels.shape[0],) + rho.shape[1:] if rho.ndim > 1 else (levels.shape[0],)
    z = levels.reshape((-1,) + (1,) * (len(shape_z) - 1)) if len(shape_z) > 1 else levels
    unnormalised = np.exp(-0.5 * ((z - peak_height_m) / half_width_m) ** 2)
    unnormalised = np.broadcast_to(unnormalised, shape_z).astype(np.float64)

    column_kg_m2 = _trapezoid(unnormalised, levels, axis=0)
    target_kg_m2 = total_column_du * _DU_TO_KG_M2
    scale = np.where(column_kg_m2 > 0.0, target_kg_m2 / np.maximum(column_kg_m2, 1e-30), 0.0)
    rho_o3 = unnormalised * scale[None, ...] if unnormalised.ndim > 1 else unnormalised * scale

    rho_safe = np.maximum(rho, 1e-6)
    return rho_o3 / rho_safe


def interpolate_level_profile(
    levels_m: np.ndarray,
    profile_n_h_w: np.ndarray,
    height_h_w: np.ndarray,
) -> np.ndarray:
    """Linear interpolation of an ``(N, H, W)`` level profile to a per-column
    height field ``(H, W)`` (e.g. the tropopause), needed because
    ``sesam_longwave``'s flux profile lives on a fixed shared level grid
    while the tropopause height varies per column. Heights outside the grid
    are clamped to the nearest edge level rather than extrapolated.
    """
    levels = np.asarray(levels_m, dtype=np.float64)
    if levels.ndim != 1:
        raise ValueError("levels_m must be 1-D")
    profile = np.asarray(profile_n_h_w, dtype=np.float64)
    if profile.shape[0] != levels.shape[0]:
        raise ValueError("profile's level axis must match levels_m")
    height = np.asarray(height_h_w, dtype=np.float64)
    if height.shape != profile.shape[1:]:
        raise ValueError("height_h_w must match the profile's (H, W) shape")

    h_clamped = np.clip(height, levels[0], levels[-1])
    idx_hi = np.clip(np.searchsorted(levels, h_clamped, side="right"), 1, levels.shape[0] - 1)
    idx_lo = idx_hi - 1
    z_lo = levels[idx_lo]
    z_hi = levels[idx_hi]
    weight = np.where(z_hi > z_lo, (h_clamped - z_lo) / np.maximum(z_hi - z_lo, 1e-12), 0.0)

    flat = profile.reshape(profile.shape[0], -1)
    idx_lo_flat = idx_lo.reshape(-1)
    idx_hi_flat = idx_hi.reshape(-1)
    cols = np.arange(flat.shape[1])
    lo_vals = flat[idx_lo_flat, cols].reshape(height.shape)
    hi_vals = flat[idx_hi_flat, cols].reshape(height.shape)
    return lo_vals + weight * (hi_vals - lo_vals)


def stratospheric_net_radiative_residual(
    longwave_downward_w_m2: np.ndarray,
    longwave_upward_w_m2: np.ndarray,
    levels_m: np.ndarray,
    tropopause_height_m: np.ndarray,
    toa_incoming_shortwave_w_m2: np.ndarray,
) -> np.ndarray:
    """``Rstr,net`` (module docstring): the net radiative convergence of the
    stratosphere -- net LW flux at the model top minus net LW flux at the
    tropopause, plus the shortwave absorbed by stratospheric ozone.

    ``longwave_downward_w_m2``/``longwave_upward_w_m2`` are
    :func:`sesam_longwave.longwave_radiation`'s full ``(N, H, W)`` sky-
    combined flux profiles (level ``N-1`` = model top, matching
    ``levels_m``'s own top entry -- must reach into the stratosphere, see
    that module's docstring finding 7).
    """
    down = np.asarray(longwave_downward_w_m2, dtype=np.float64)
    up = np.asarray(longwave_upward_w_m2, dtype=np.float64)
    if down.shape != up.shape:
        raise ValueError("longwave_downward_w_m2 and longwave_upward_w_m2 must share a shape")
    net = down - up
    net_top = net[-1]
    net_tropopause = interpolate_level_profile(levels_m, net, tropopause_height_m)
    lw_term = net_top - net_tropopause
    sw_term = FRAC_VU * (1.0 - I_O3_VU) * np.asarray(toa_incoming_shortwave_w_m2, dtype=np.float64)
    return lw_term + sw_term


@dataclass(frozen=True)
class TropopauseUpdate:
    tropopause_height_m: np.ndarray
    tendency_m_per_day: np.ndarray
    shape_s: np.ndarray
    r_strat_net_w_m2: np.ndarray


def advance_tropopause_height(
    tropopause_height_m: np.ndarray,
    latitude_rad: np.ndarray,
    itcz_latitude_rad: np.ndarray | float,
    hadley_width_rad: np.ndarray | float,
    longwave_downward_w_m2: np.ndarray,
    longwave_upward_w_m2: np.ndarray,
    levels_m: np.ndarray,
    toa_incoming_shortwave_w_m2: np.ndarray,
    dt_days: float,
    surface_elevation_m: np.ndarray,
    a1: dict[str, float] | None = None,
    min_thickness_m: float = 3000.0,
) -> TropopauseUpdate:
    """Full (A10)/(A11) closure: computes ``Rstr,net`` (this module) and
    ``S`` (``sesam_vertical.tropopause_shape_s``), gets the tendency
    (``sesam_vertical.tropopause_tendency``, confirmed metres/day per module
    docstring), and integrates one ``dt_days`` step.

    ``tropopause_height_m`` is clamped to ``[surface_elevation_m +
    min_thickness_m, MODEL_TOP_M]`` afterwards -- a basic domain-validity
    bound on a newly-prognostic field (the paper gives no upper/lower limit
    for (A10) itself), unrelated to the omega-cap prohibition in
    ``docs/VERTICAL_THERMODYNAMIC_CLOSURE.md`` (a different closure family
    entirely -- vertical velocity, not tropopause height).
    """
    ht = np.asarray(tropopause_height_m, dtype=np.float64)
    r_strat = stratospheric_net_radiative_residual(
        longwave_downward_w_m2, longwave_upward_w_m2, levels_m, ht, toa_incoming_shortwave_w_m2,
    )
    shape_s = _sv.tropopause_shape_s(latitude_rad, itcz_latitude_rad, hadley_width_rad, a1=a1)
    tendency = _sv.tropopause_tendency(r_strat, shape_s, a1=a1)
    updated = ht + tendency * float(dt_days)
    zs = np.asarray(surface_elevation_m, dtype=np.float64)
    lower_bound = zs + min_thickness_m
    updated = np.clip(updated, lower_bound, MODEL_TOP_M)
    return TropopauseUpdate(
        tropopause_height_m=updated,
        tendency_m_per_day=tendency,
        shape_s=shape_s,
        r_strat_net_w_m2=r_strat,
    )


__all__ = [
    "MODEL_TOP_M",
    "standard_ozone_mixing_ratio_profile_kgkg",
    "interpolate_level_profile",
    "stratospheric_net_radiative_residual",
    "TropopauseUpdate",
    "advance_tropopause_height",
]
