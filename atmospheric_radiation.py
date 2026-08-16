"""Pure conservative surface--atmosphere longwave energy kernels."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


STEFAN_BOLTZMANN = 5.670374419e-8


class GreyRadiationStep(NamedTuple):
    surface_gain_w_m2: np.ndarray
    atmospheric_gain_w_m2: np.ndarray
    outgoing_longwave_w_m2: np.ndarray
    toa_net_radiation_w_m2: np.ndarray


class PressureDefinedTemperatureProfile(NamedTuple):
    """Dry-adiabatic temperatures at two pressure-defined atmospheric levels."""

    midlevel_pressure_pa: np.ndarray
    upperlevel_pressure_pa: np.ndarray
    midlevel_temperature_k: np.ndarray
    upperlevel_temperature_k: np.ndarray


class TwoLayerGreyOpticalClosure(NamedTuple):
    total_optical_depth: np.ndarray
    midlevel_emissivity: np.ndarray
    upperlevel_emissivity: np.ndarray
    opaque_limited: np.ndarray
    target_olr_residual_w_m2: np.ndarray


class PressureSplitGreyEmissivity(NamedTuple):
    midlevel_emissivity: np.ndarray
    upperlevel_emissivity: np.ndarray


class TwoLayerGreyRadiationStep(NamedTuple):
    surface_gain_w_m2: np.ndarray
    midlevel_gain_w_m2: np.ndarray
    upperlevel_gain_w_m2: np.ndarray
    lower_downward_emission_w_m2: np.ndarray
    lower_upward_emission_w_m2: np.ndarray
    upper_downward_emission_w_m2: np.ndarray
    upper_upward_emission_w_m2: np.ndarray
    downward_longwave_at_surface_w_m2: np.ndarray
    outgoing_longwave_w_m2: np.ndarray
    toa_net_radiation_w_m2: np.ndarray


def pressure_defined_temperature_profile(
    lower_temperature_k: np.ndarray,
    surface_pressure_pa: np.ndarray | float,
    lower_mid_pressure_depth_pa: np.ndarray | float,
    mid_upper_pressure_depth_pa: np.ndarray | float,
    *,
    gas_constant_dry_air_j_kg_k: float = 287.05,
    cp_dry_air_j_kg_k: float = 1004.0,
) -> PressureDefinedTemperatureProfile:
    """Return a pressure-coordinate dry-adiabatic temperature profile.

    The supplied lower temperature is anchored at surface pressure.  Midlevel
    and upper-level pressures are defined by explicit pressure thicknesses,
    and their temperatures follow conservation of dry potential temperature,
    ``T(p) = T_s * (p / p_s) ** (R_d / c_p)``.  The relation therefore has no
    height-based lapse offset or calibrated emission-temperature parameter.

    Scalars may be used for pressure and are broadcast over the temperature
    field.  Returned fields always have the lower-temperature shape.
    """
    lower = np.asarray(lower_temperature_k, dtype=np.float64)
    surface_pressure = np.asarray(surface_pressure_pa, dtype=np.float64)
    lower_mid_depth = np.asarray(lower_mid_pressure_depth_pa, dtype=np.float64)
    mid_upper_depth = np.asarray(mid_upper_pressure_depth_pa, dtype=np.float64)
    try:
        lower, surface_pressure, lower_mid_depth, mid_upper_depth = np.broadcast_arrays(
            lower, surface_pressure, lower_mid_depth, mid_upper_depth
        )
    except ValueError as exc:
        raise ValueError("pressure-profile fields must be broadcast-compatible") from exc

    constants = (gas_constant_dry_air_j_kg_k, cp_dry_air_j_kg_k)
    if not all(np.isfinite(value) and value > 0.0 for value in constants):
        raise ValueError("dry-air gas constant and heat capacity must be finite and positive")
    if gas_constant_dry_air_j_kg_k >= cp_dry_air_j_kg_k:
        raise ValueError("dry-air gas constant must be smaller than heat capacity")
    if not all(
        np.all(np.isfinite(field))
        for field in (lower, surface_pressure, lower_mid_depth, mid_upper_depth)
    ):
        raise ValueError("pressure-profile inputs must be finite")
    if np.any(lower <= 0.0) or np.any(surface_pressure <= 0.0):
        raise ValueError("lower temperature and surface pressure must be positive")
    if np.any(lower_mid_depth <= 0.0) or np.any(mid_upper_depth <= 0.0):
        raise ValueError("pressure-layer depths must be positive")

    midlevel_pressure = surface_pressure - lower_mid_depth
    upperlevel_pressure = midlevel_pressure - mid_upper_depth
    if np.any(upperlevel_pressure <= 0.0):
        raise ValueError("pressure-layer depths must leave positive upper-level pressure")

    exponent = gas_constant_dry_air_j_kg_k / cp_dry_air_j_kg_k
    midlevel_temperature = lower * (midlevel_pressure / surface_pressure) ** exponent
    upperlevel_temperature = lower * (upperlevel_pressure / surface_pressure) ** exponent
    return PressureDefinedTemperatureProfile(
        midlevel_pressure.copy(),
        upperlevel_pressure.copy(),
        midlevel_temperature.copy(),
        upperlevel_temperature.copy(),
    )


def effective_radiating_temperature(
    outgoing_longwave_w_m2: np.ndarray,
    *,
    sigma_w_m2_k4: float = STEFAN_BOLTZMANN,
) -> np.ndarray:
    """Return the blackbody temperature corresponding to TOA longwave flux."""
    outgoing = np.asarray(outgoing_longwave_w_m2, dtype=np.float64)
    if sigma_w_m2_k4 <= 0.0 or not np.isfinite(sigma_w_m2_k4):
        raise ValueError("Stefan--Boltzmann constant must be finite and positive")
    if not np.all(np.isfinite(outgoing)) or np.any(outgoing < 0.0):
        raise ValueError("outgoing longwave must be finite and non-negative")
    return (outgoing / sigma_w_m2_k4) ** 0.25


def resolved_midlevel_emission_temperature(
    midlevel_temperature_k: np.ndarray | None,
    *,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Validate and return a resolved pressure-midlevel emission temperature.

    There is deliberately no lower/free-air fallback.  Callers that have not
    activated or supplied a midlevel state fail closed instead of silently
    turning a near-surface temperature into atmospheric longwave emission.
    """
    if midlevel_temperature_k is None:
        raise ValueError("atmospheric emission requires resolved midlevel temperature")
    temperature = np.asarray(midlevel_temperature_k, dtype=np.float64)
    if expected_shape is not None and temperature.shape != expected_shape:
        raise ValueError("midlevel emission temperature has an unexpected shape")
    if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError(
            "midlevel emission temperature must be finite and positive "
            f"(min_k={float(np.nanmin(temperature)):.6g})"
        )
    return temperature


def atmospheric_emissivity_for_target_olr(
    surface_temperature_k: np.ndarray,
    atmospheric_emission_temperature_k: np.ndarray,
    target_outgoing_longwave_w_m2: np.ndarray,
    *,
    sigma_w_m2_k4: float = STEFAN_BOLTZMANN,
) -> np.ndarray:
    """Diagnose the bounded grey emissivity that reproduces a target OLR.

    For the one-layer grey contract,
    ``OLR = (1 - epsilon) sigma Ts^4 + epsilon sigma Ta^4``.  Solving this
    equation makes the legacy/resolved TOA flux an explicit boundary condition
    while retaining the pressure-defined atmospheric temperature.  No fitted
    offset or emissivity calibration is introduced.

    A target outside the interval spanned by surface and atmospheric blackbody
    emission is not representable by a bounded grey emissivity and fails
    closed.  In an isothermal column OLR is independent of emissivity; the
    transparent representative is returned only when the target equals that
    unique flux to numerical tolerance.
    """
    surface = np.asarray(surface_temperature_k, dtype=np.float64)
    atmosphere = np.asarray(atmospheric_emission_temperature_k, dtype=np.float64)
    target = np.asarray(target_outgoing_longwave_w_m2, dtype=np.float64)
    if not (surface.shape == atmosphere.shape == target.shape):
        raise ValueError("emissivity-closure fields must share a shape")
    if sigma_w_m2_k4 <= 0.0 or not np.isfinite(sigma_w_m2_k4):
        raise ValueError("Stefan--Boltzmann constant must be finite and positive")
    if not all(np.all(np.isfinite(field)) for field in (surface, atmosphere, target)):
        raise ValueError("emissivity-closure inputs must be finite")
    if np.any(surface <= 0.0) or np.any(atmosphere <= 0.0) or np.any(target < 0.0):
        raise ValueError("temperatures must be positive and target OLR non-negative")

    surface_emission = sigma_w_m2_k4 * surface**4
    atmospheric_emission = sigma_w_m2_k4 * atmosphere**4
    lower = np.minimum(surface_emission, atmospheric_emission)
    upper = np.maximum(surface_emission, atmospheric_emission)
    tolerance = 1.0e-12 * np.maximum(np.maximum(np.abs(lower), np.abs(upper)), 1.0)
    if np.any(target < lower - tolerance) or np.any(target > upper + tolerance):
        raise ValueError("target OLR is outside the bounded grey-atmosphere range")

    denominator = surface_emission - atmospheric_emission
    isothermal = np.abs(denominator) <= tolerance
    if np.any(isothermal & (np.abs(target - surface_emission) > tolerance)):
        raise ValueError("isothermal grey column has a unique outgoing longwave flux")
    emissivity = np.zeros_like(surface_emission)
    np.divide(
        surface_emission - target,
        denominator,
        out=emissivity,
        where=~isothermal,
    )
    return np.clip(emissivity, 0.0, 1.0)


def two_layer_grey_radiation(
    surface_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    absorbed_shortwave_w_m2: np.ndarray,
    midlevel_emissivity: np.ndarray,
    upperlevel_emissivity: np.ndarray,
    *,
    sigma_w_m2_k4: float = STEFAN_BOLTZMANN,
) -> TwoLayerGreyRadiationStep:
    """Return a conservative two-atmospheric-layer grey radiative budget."""
    fields = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            surface_temperature_k,
            midlevel_temperature_k,
            upperlevel_temperature_k,
            absorbed_shortwave_w_m2,
            midlevel_emissivity,
            upperlevel_emissivity,
        )
    )
    surface, middle, upper, shortwave, middle_e, upper_e = fields
    if any(field.shape != surface.shape for field in fields[1:]):
        raise ValueError("two-layer grey-radiation fields must share a shape")
    if sigma_w_m2_k4 <= 0.0 or not np.isfinite(sigma_w_m2_k4):
        raise ValueError("Stefan--Boltzmann constant must be finite and positive")
    if any(not np.all(np.isfinite(field)) for field in fields):
        raise ValueError("two-layer grey-radiation inputs must be finite")
    if any(np.any(field <= 0.0) for field in (surface, middle, upper)):
        raise ValueError("radiating temperatures must be positive")
    if np.any(shortwave < 0.0) or any(
        np.any((field < 0.0) | (field > 1.0)) for field in (middle_e, upper_e)
    ):
        raise ValueError("shortwave and emissivities are outside physical bounds")

    surface_emission = sigma_w_m2_k4 * surface**4
    middle_blackbody = sigma_w_m2_k4 * middle**4
    upper_blackbody = sigma_w_m2_k4 * upper**4
    middle_t = 1.0 - middle_e
    upper_t = 1.0 - upper_e
    middle_up = middle_e * middle_blackbody
    middle_down = middle_up
    upper_up = upper_e * upper_blackbody
    upper_down = upper_up

    upward_into_upper = middle_t * surface_emission + middle_up
    downward_at_surface = middle_down + middle_t * upper_down
    outgoing = upper_t * upward_into_upper + upper_up
    surface_gain = shortwave + downward_at_surface - surface_emission
    middle_gain = (
        middle_e * surface_emission
        + middle_e * upper_down
        - middle_up
        - middle_down
    )
    upper_gain = upper_e * upward_into_upper - upper_up - upper_down
    toa_net = shortwave - outgoing
    return TwoLayerGreyRadiationStep(
        surface_gain,
        middle_gain,
        upper_gain,
        middle_down,
        middle_up,
        upper_down,
        upper_up,
        downward_at_surface,
        outgoing,
        toa_net,
    )


def pressure_split_emissivities_from_optical_depth(
    total_optical_depth: np.ndarray,
    lower_mid_pressure_depth_pa: np.ndarray | float,
    mid_upper_pressure_depth_pa: np.ndarray | float,
) -> PressureSplitGreyEmissivity:
    """Split a finite total optical depth in proportion to pressure mass."""
    optical_depth = np.asarray(total_optical_depth, dtype=np.float64)
    lower_depth = np.asarray(lower_mid_pressure_depth_pa, dtype=np.float64)
    upper_depth = np.asarray(mid_upper_pressure_depth_pa, dtype=np.float64)
    try:
        optical_depth, lower_depth, upper_depth = np.broadcast_arrays(
            optical_depth, lower_depth, upper_depth
        )
    except ValueError as exc:
        raise ValueError("optical depth and pressure depths must be broadcast-compatible") from exc
    if any(
        not np.all(np.isfinite(field))
        for field in (optical_depth, lower_depth, upper_depth)
    ):
        raise ValueError("optical depth and pressure depths must be finite")
    if np.any(optical_depth < 0.0) or np.any(lower_depth <= 0.0) or np.any(upper_depth <= 0.0):
        raise ValueError("optical depth must be non-negative and pressure depths positive")
    middle_fraction = lower_depth / (lower_depth + upper_depth)
    upper_fraction = 1.0 - middle_fraction
    return PressureSplitGreyEmissivity(
        -np.expm1(-optical_depth * middle_fraction),
        -np.expm1(-optical_depth * upper_fraction),
    )


class GreyEquilibriumProfile(NamedTuple):
    """Adiabat-limited grey radiative-convective equilibrium profile."""

    midlevel_temperature_k: np.ndarray
    upperlevel_temperature_k: np.ndarray
    adiabatic_limited_midlevel: np.ndarray
    adiabatic_limited_upperlevel: np.ndarray


def grey_radiative_convective_equilibrium_temperatures(
    surface_temperature_k: np.ndarray,
    lower_temperature_k: np.ndarray,
    total_optical_depth: np.ndarray,
    lower_mid_pressure_depth_pa: np.ndarray | float,
    mid_upper_pressure_depth_pa: np.ndarray | float,
    *,
    gas_constant_dry_air_j_kg_k: float = 287.05,
    cp_dry_air_j_kg_k: float = 1004.0,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
    sigma_w_m2_k4: float = STEFAN_BOLTZMANN,
) -> GreyEquilibriumProfile:
    """Return the adiabat-limited grey radiative-convective equilibrium.

    Sets the model's own two-layer grey layer gains to zero
    (``two_layer_grey_radiation``): with ``B = sigma T**4`` and surface
    emission ``E_s``, the equilibrium solves ``2 B_m = E_s + eps_u B_u`` and
    ``2 B_u = (1 - eps_m) E_s + eps_m B_m``, which is linear in the blackbody
    fluxes and therefore exact per cell.  Absorbed shortwave passes through
    both atmospheric layers and does not enter their gains, so it is not an
    input.

    Pure radiative equilibrium is super-adiabatic in the lower troposphere
    (measured 2026-08-16: its lower-mid potential-temperature gradient is
    negative over ~17% of area for the 32x64 handoff state), which would feed
    a near-zero or negative denominator to the diabatic-omega diagnostic.
    The profile is therefore limited from below by the dry adiabat anchored
    on the resolved lower temperature, evaluated at the same layer-centre
    pressures the omega diagnostic uses (the fixed 0.40/0.35/0.25
    pressure-mass partition, a contract of the three-level closure family).
    This is an initialization-time convective constraint -- the standard
    radiative-convective equilibrium -- not a runtime cap, floor, or damping
    term on any tendency.

    The solve is sequential: the mid level is clamped first, then the upper
    level's zero-gain point is re-solved against the *actual* (possibly
    clamped) mid-level blackbody flux, ``2 B_u = (1 - eps_m) E_s + eps_m
    B_m_actual``, before applying its own adiabatic floor.  An unclamped
    upper level therefore has exactly zero grey gain even when the mid level
    is clamped.  Cells where a floor binds are reported in the returned
    masks; there the profile keeps non-negative static stability by
    construction while a bounded residual grey gain remains.

    Screening note (2026-08-16, docs/VERTICAL_THERMODYNAMIC_CLOSURE.md): as a
    handoff initialization for the coupled grey gate this profile removes the
    day-1 grey-gain shock, but clamped rows sit exactly at neutral stability
    -- the singular point of the diabatic-omega diagnostic -- and the coupled
    column collapses regardless.  It is retained as a measured equilibrium
    diagnostic, not as an admissible initialization.
    """
    surface = np.asarray(surface_temperature_k, dtype=np.float64)
    lower = np.asarray(lower_temperature_k, dtype=np.float64)
    optical_depth = np.asarray(total_optical_depth, dtype=np.float64)
    if surface.shape != lower.shape or surface.shape != optical_depth.shape:
        raise ValueError("surface, lower, and optical-depth fields must share a shape")
    if not all(
        np.all(np.isfinite(field)) for field in (surface, lower, optical_depth)
    ):
        raise ValueError("equilibrium inputs must be finite")
    if np.any(surface <= 0.0) or np.any(lower <= 0.0):
        raise ValueError("surface and lower temperatures must be positive")
    if np.any(optical_depth < 0.0):
        raise ValueError("optical depth must be non-negative")
    if gas_constant_dry_air_j_kg_k <= 0.0 or cp_dry_air_j_kg_k <= 0.0:
        raise ValueError("dry-air gas constant and heat capacity must be positive")
    if gas_constant_dry_air_j_kg_k >= cp_dry_air_j_kg_k:
        raise ValueError("dry-air gas constant must be smaller than heat capacity")
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(np.sum(fractions), 1.0):
        raise ValueError("layer mass fractions must be three positive values summing to one")

    split = pressure_split_emissivities_from_optical_depth(
        optical_depth, lower_mid_pressure_depth_pa, mid_upper_pressure_depth_pa
    )
    eps_m = split.midlevel_emissivity
    eps_u = split.upperlevel_emissivity
    surface_emission = sigma_w_m2_k4 * surface**4
    middle_blackbody = (
        surface_emission * (2.0 + eps_u - eps_u * eps_m) / (4.0 - eps_m * eps_u)
    )
    middle_equilibrium = (middle_blackbody / sigma_w_m2_k4) ** 0.25

    # Layer-centre pressures of the omega diagnostic, as fractions of surface
    # pressure; only their ratios enter, so surface pressure itself cancels.
    edges = np.array((1.0, 1.0 - fractions[0], fractions[2], 0.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    kappa = gas_constant_dry_air_j_kg_k / cp_dry_air_j_kg_k
    middle_floor = lower * (centers[1] / centers[0]) ** kappa
    midlevel_temperature = np.maximum(middle_equilibrium, middle_floor)

    # Re-solve the upper level's zero-gain point against the actual mid-level
    # blackbody flux so an unclamped upper level has zero gain even when the
    # mid level's floor bound.
    actual_middle_blackbody = sigma_w_m2_k4 * midlevel_temperature**4
    upper_blackbody = (
        (1.0 - eps_m) * surface_emission + eps_m * actual_middle_blackbody
    ) / 2.0
    upper_equilibrium = (upper_blackbody / sigma_w_m2_k4) ** 0.25
    upper_floor = midlevel_temperature * (centers[2] / centers[1]) ** kappa
    upperlevel_temperature = np.maximum(upper_equilibrium, upper_floor)
    return GreyEquilibriumProfile(
        midlevel_temperature,
        upperlevel_temperature,
        middle_equilibrium < middle_floor,
        upper_equilibrium < upper_floor,
    )


def two_layer_optical_depth_for_target_olr(
    surface_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    target_outgoing_longwave_w_m2: np.ndarray,
    lower_mid_pressure_depth_pa: np.ndarray | float,
    mid_upper_pressure_depth_pa: np.ndarray | float,
    *,
    sigma_w_m2_k4: float = STEFAN_BOLTZMANN,
    allow_opaque_limit: bool = False,
) -> TwoLayerGreyOpticalClosure:
    """Solve pressure-split grey optical depth for an explicit target OLR.

    Total optical depth is the sole diagnosed opacity.  It is divided between
    the resolved midlevel and upper-level emitters in direct proportion to the
    two supplied pressure thicknesses.  With temperature decreasing upward,
    OLR varies monotonically from surface blackbody emission at zero optical
    depth toward upper-level blackbody emission in the opaque limit.
    """
    surface = np.asarray(surface_temperature_k, dtype=np.float64)
    middle = np.asarray(midlevel_temperature_k, dtype=np.float64)
    upper = np.asarray(upperlevel_temperature_k, dtype=np.float64)
    target = np.asarray(target_outgoing_longwave_w_m2, dtype=np.float64)
    lower_depth = np.asarray(lower_mid_pressure_depth_pa, dtype=np.float64)
    upper_depth = np.asarray(mid_upper_pressure_depth_pa, dtype=np.float64)
    try:
        surface, middle, upper, target, lower_depth, upper_depth = np.broadcast_arrays(
            surface, middle, upper, target, lower_depth, upper_depth
        )
    except ValueError as exc:
        raise ValueError("optical-depth closure fields must be broadcast-compatible") from exc
    if sigma_w_m2_k4 <= 0.0 or not np.isfinite(sigma_w_m2_k4):
        raise ValueError("Stefan--Boltzmann constant must be finite and positive")
    if any(
        not np.all(np.isfinite(field))
        for field in (surface, middle, upper, target, lower_depth, upper_depth)
    ):
        raise ValueError("optical-depth closure inputs must be finite")
    if any(np.any(field <= 0.0) for field in (surface, middle, upper)):
        raise ValueError("radiating temperatures must be positive")
    if np.any(target < 0.0) or np.any(lower_depth <= 0.0) or np.any(upper_depth <= 0.0):
        raise ValueError("target OLR must be non-negative and pressure depths positive")
    if np.any(middle < upper):
        raise ValueError("two-layer optical closure requires upper level no warmer than midlevel")

    surface_flux = sigma_w_m2_k4 * surface**4
    middle_flux = sigma_w_m2_k4 * middle**4
    upper_flux = sigma_w_m2_k4 * upper**4
    tolerance = 1.0e-11 * np.maximum(surface_flux, 1.0)
    too_high = target > surface_flux + tolerance
    too_low = target < upper_flux - tolerance
    if np.any(too_high) or (np.any(too_low) and not allow_opaque_limit):
        raise ValueError(
            "target OLR is outside the pressure-profile grey range "
            f"(above_surface={int(np.count_nonzero(too_high))}, "
            f"below_upper={int(np.count_nonzero(too_low))}, "
            f"max_below_upper_w_m2={float(np.max(np.where(too_low, upper_flux - target, 0.0))):.6g})"
        )

    effective_target = np.maximum(target, upper_flux) if allow_opaque_limit else target
    middle_fraction = lower_depth / (lower_depth + upper_depth)
    upper_fraction = 1.0 - middle_fraction

    def _olr(optical_depth: np.ndarray) -> np.ndarray:
        middle_t = np.exp(-optical_depth * middle_fraction)
        upper_t = np.exp(-optical_depth * upper_fraction)
        return (
            upper_t
            * (middle_t * surface_flux + (1.0 - middle_t) * middle_flux)
            + (1.0 - upper_t) * upper_flux
        )

    low = np.zeros_like(surface_flux)
    high = np.full_like(surface_flux, 64.0)
    if np.any(_olr(high) > effective_target + tolerance):
        raise ValueError("target OLR requires effectively infinite optical depth")
    for _ in range(64):
        middle_tau = 0.5 * (low + high)
        trial = _olr(middle_tau)
        low = np.where(trial > effective_target, middle_tau, low)
        high = np.where(trial > effective_target, high, middle_tau)
    total = 0.5 * (low + high)
    middle_emissivity = -np.expm1(-total * middle_fraction)
    upper_emissivity = -np.expm1(-total * upper_fraction)
    if allow_opaque_limit:
        middle_emissivity = np.where(too_low, 1.0, middle_emissivity)
        upper_emissivity = np.where(too_low, 1.0, upper_emissivity)
    achieved_olr = (
        (1.0 - upper_emissivity)
        * (
            (1.0 - middle_emissivity) * surface_flux
            + middle_emissivity * middle_flux
        )
        + upper_emissivity * upper_flux
    )
    return TwoLayerGreyOpticalClosure(
        total,
        middle_emissivity,
        upper_emissivity,
        too_low.copy(),
        achieved_olr - target,
    )


def grey_surface_atmosphere_radiation(
    surface_temperature_k: np.ndarray,
    atmospheric_emission_temperature_k: np.ndarray,
    absorbed_shortwave_w_m2: np.ndarray,
    atmospheric_longwave_emissivity: np.ndarray,
    *,
    sigma_w_m2_k4: float = STEFAN_BOLTZMANN,
) -> GreyRadiationStep:
    """Return a conservative one-layer grey radiative budget.

    Shortwave is absorbed at the surface. The atmosphere absorbs and emits
    longwave with emissivity ``epsilon``; its upward and downward emissions are
    equal. The unabsorbed surface window reaches space directly. Consequently
    surface gain + atmospheric gain equals absorbed shortwave minus TOA OLR
    exactly. An explicit atmospheric emission temperature is required by the
    contract and is intentionally not inferred from near-surface temperature.
    """
    surface = np.asarray(surface_temperature_k, dtype=np.float64)
    atmosphere = np.asarray(atmospheric_emission_temperature_k, dtype=np.float64)
    shortwave = np.asarray(absorbed_shortwave_w_m2, dtype=np.float64)
    emissivity = np.asarray(atmospheric_longwave_emissivity, dtype=np.float64)
    if not (
        surface.shape == atmosphere.shape == shortwave.shape == emissivity.shape
    ):
        raise ValueError("grey-radiation fields must share a shape")
    if sigma_w_m2_k4 <= 0.0 or not np.isfinite(sigma_w_m2_k4):
        raise ValueError("Stefan--Boltzmann constant must be finite and positive")
    if not all(
        np.all(np.isfinite(field))
        for field in (surface, atmosphere, shortwave, emissivity)
    ):
        raise ValueError("grey-radiation inputs must be finite")
    if np.any(surface <= 0.0) or np.any(atmosphere <= 0.0):
        raise ValueError("radiating temperatures must be positive")
    if np.any(shortwave < 0.0) or np.any((emissivity < 0.0) | (emissivity > 1.0)):
        raise ValueError("shortwave and emissivity are outside physical bounds")

    surface_emission = sigma_w_m2_k4 * surface**4
    atmospheric_emission = sigma_w_m2_k4 * atmosphere**4
    surface_gain = (
        shortwave + emissivity * atmospheric_emission - surface_emission
    )
    atmospheric_gain = emissivity * (
        surface_emission - 2.0 * atmospheric_emission
    )
    outgoing = (
        (1.0 - emissivity) * surface_emission
        + emissivity * atmospheric_emission
    )
    toa_net = shortwave - outgoing
    return GreyRadiationStep(
        surface_gain, atmospheric_gain, outgoing, toa_net
    )
