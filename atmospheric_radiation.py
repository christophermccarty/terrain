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
        raise ValueError("midlevel emission temperature must be finite and positive")
    return temperature


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
