"""Conservative primitives for PlanetSim's native pressure-column path.

The legacy atmosphere carries one humidity field plus optional two-layer
adjustments.  This module is deliberately independent of precipitation and
wind generation: it evolves three vapor reservoirs using two pressure-
coordinate interface velocities supplied by the circulation code.  Keeping
the vertical exchange in a small pure kernel makes its water budget testable
before it is coupled to cloud microphysics or radiation.

``evolve_closed_three_level_thermodynamic_column`` is deliberately a separate
kernel from the older experimental path below.  The older path represents its
three humidity values as pieces of one scalar and restores temperature toward
a prescribed lapse profile; it is useful for backwards-compatible experiments,
but it is not a finite-volume thermodynamic closure.  The new kernel instead
uses mixing ratios in explicit pressure-mass layers and transports moist static
energy with the same pressure-coordinate mass flux as water.  Radiation is an
explicit source and condensation is an internal vapour-to-thermal conversion.
Keeping that contract pure and small lets the integration layer be tested
before it is allowed to affect the calibrated climate path.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from column_water import evolve_column_water


class PressureColumnStep(NamedTuple):
    """One conservative three-level column update."""

    lower_humidity: np.ndarray
    midlevel_humidity: np.ndarray
    upperlevel_humidity: np.ndarray
    midlevel_temperature: np.ndarray
    upperlevel_temperature: np.ndarray
    omega_lower_mid_pa_s: np.ndarray
    omega_mid_upper_pa_s: np.ndarray
    relative_water_residual: float


class ClosedThermodynamicColumnStep(NamedTuple):
    """One finite-volume lower/mid/upper thermodynamic column update.

    Humidity values are layer mixing ratios [kg kg-1].  Temperatures are
    recovered from conserved moist static energy [J kg-1].  The residuals are
    column-integrated quantities: water in kg m-2 and energy in J m-2.
    """

    lower_humidity: np.ndarray
    midlevel_humidity: np.ndarray
    upperlevel_humidity: np.ndarray
    lower_temperature: np.ndarray
    midlevel_temperature: np.ndarray
    upperlevel_temperature: np.ndarray
    lower_mid_mass_flux_kg_m2_s: np.ndarray
    mid_upper_mass_flux_kg_m2_s: np.ndarray
    water_residual_kg_m2: float
    moist_static_energy_residual_j_m2: float
    radiative_energy_input_j_m2: float


class ClosedColumnHorizontalMSEStep(NamedTuple):
    """Three-layer temperatures after conservative horizontal MSE transport."""

    lower_temperature: np.ndarray
    midlevel_temperature: np.ndarray
    upperlevel_temperature: np.ndarray
    energy_residual_j: float
    relative_energy_residual: float


_GRAVITY_M_S2 = 9.80665
_DRY_AIR_HEAT_CAPACITY_J_KG_K = 1004.0
_LATENT_HEAT_VAPORIZATION_J_KG = 2.5e6


def transport_closed_three_level_mse(
    lower_humidity_before: np.ndarray,
    midlevel_humidity_before: np.ndarray,
    upperlevel_humidity_before: np.ndarray,
    lower_humidity_after: np.ndarray,
    midlevel_humidity_after: np.ndarray,
    upperlevel_humidity_after: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    lower_wind_u_m_s: np.ndarray,
    lower_wind_v_m_s: np.ndarray,
    midlevel_wind_u_m_s: np.ndarray,
    midlevel_wind_v_m_s: np.ndarray,
    upperlevel_wind_u_m_s: np.ndarray,
    upperlevel_wind_v_m_s: np.ndarray,
    *,
    lower_vapour_source_kg_kg_day: np.ndarray,
    dt_days: float,
    dx_m: np.ndarray | float,
    dy_m: float,
    cell_area_m2: np.ndarray | float,
    x_face_length_m: np.ndarray | float,
    y_face_length_m: np.ndarray | float,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
    surface_pressure_pa: float = 101_325.0,
    layer_heights_m: tuple[float, float, float] = (0.0, 3500.0, 8000.0),
) -> ClosedColumnHorizontalMSEStep:
    """Transport pressure-layer MSE on the same faces as its vapour.

    The pressure-coordinate moisture closure already transports each layer's
    vapour with the diagnosed shared winds. This companion operation carries
    layer MSE contents on the identical finite-volume faces, then recovers
    temperatures using those transported humidities. Surface evaporation
    imports its latent energy into the lower atmospheric layer; no relaxation,
    amplitude scale, or energy clipping is introduced.
    """
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(
        float(np.sum(fractions)), 1.0, atol=1e-12
    ):
        raise ValueError("layer_mass_fractions must be three positive values summing to one")
    if surface_pressure_pa <= 0.0 or dt_days <= 0.0:
        raise ValueError("surface_pressure_pa and dt_days must be positive")
    heights = np.asarray(layer_heights_m, dtype=np.float64)
    if heights.shape != (3,) or not np.all(np.isfinite(heights)):
        raise ValueError("layer_heights_m must be three finite values")
    raw = (
        lower_humidity_before, midlevel_humidity_before, upperlevel_humidity_before,
        lower_humidity_after, midlevel_humidity_after, upperlevel_humidity_after,
        lower_temperature_k, midlevel_temperature_k, upperlevel_temperature_k,
        lower_wind_u_m_s, lower_wind_v_m_s, midlevel_wind_u_m_s,
        midlevel_wind_v_m_s, upperlevel_wind_u_m_s, upperlevel_wind_v_m_s,
        lower_vapour_source_kg_kg_day,
    )
    fields = tuple(np.asarray(value, dtype=np.float64) for value in raw)
    shape = fields[0].shape
    if len(shape) != 2 or any(value.shape != shape for value in fields):
        raise ValueError("MSE transport fields must share a two-dimensional shape")
    if any(not np.all(np.isfinite(value)) for value in fields):
        raise ValueError("MSE transport fields must be finite")
    if any(np.any(value < 0.0) for value in fields[:6]) or np.any(fields[-1] < 0.0):
        raise ValueError("MSE transport humidities and vapour source must be non-negative")

    humidity_before = fields[:3]
    humidity_after = fields[3:6]
    temperatures = fields[6:9]
    winds = ((fields[9], fields[10]), (fields[11], fields[12]), (fields[13], fields[14]))
    layer_mass = fractions * (float(surface_pressure_pa) / _GRAVITY_M_S2)
    energy_residual = 0.0
    expected_energy = 0.0
    transported_energy: list[np.ndarray] = []
    for index in range(3):
        mse = (
            _DRY_AIR_HEAT_CAPACITY_J_KG_K * temperatures[index]
            + _GRAVITY_M_S2 * heights[index]
            + _LATENT_HEAT_VAPORIZATION_J_KG * humidity_before[index]
        )
        energy_content = layer_mass[index] * mse
        source = (
            layer_mass[index] * _LATENT_HEAT_VAPORIZATION_J_KG * fields[-1]
            if index == 0 else np.zeros_like(energy_content)
        )
        step = evolve_column_water(
            energy_content, source, np.zeros_like(energy_content), *winds[index],
            dx_m=dx_m, dy_m=dy_m, dt_days=dt_days, cell_area_m2=cell_area_m2,
            x_face_length_m=x_face_length_m, y_face_length_m=y_face_length_m,
        )
        transported_energy.append(np.asarray(step.water_mm, dtype=np.float64))
        energy_residual += float(step.residual_mm)
        expected_energy += float(np.sum((energy_content + dt_days * source) * np.asarray(cell_area_m2)))

    output_temperature = tuple(
        (transported_energy[index] / layer_mass[index]
         - _GRAVITY_M_S2 * heights[index]
         - _LATENT_HEAT_VAPORIZATION_J_KG * humidity_after[index])
        / _DRY_AIR_HEAT_CAPACITY_J_KG_K
        for index in range(3)
    )
    return ClosedColumnHorizontalMSEStep(
        *(value.astype(np.float32) for value in output_temperature),
        energy_residual,
        energy_residual / max(abs(expected_energy), 1.0),
    )


def _as_column_field(
    value: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    *,
    shape: tuple[int, ...],
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return three finite fields, accepting ``None`` as a zero source."""
    if value is None:
        zero = np.zeros(shape, dtype=np.float64)
        return zero, zero.copy(), zero.copy()
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a three-element tuple of layer fields")
    fields = tuple(np.asarray(item, dtype=np.float64) for item in value)
    if any(item.shape != shape for item in fields):
        raise ValueError(f"{name} fields must match the column shape")
    if any(not np.all(np.isfinite(item)) for item in fields):
        raise ValueError(f"{name} fields must be finite")
    return fields


def _transfer_layer_content(
    humidity: list[np.ndarray],
    moist_static_energy: list[np.ndarray],
    layer_mass_kg_m2: np.ndarray,
    *,
    donor: int,
    receiver: int,
    transfer_mass_kg_m2: np.ndarray,
) -> None:
    """Move donor air composition across one interface without a hidden source."""
    # Layer dry-air masses are held fixed by the supplied horizontal mass
    # convergence.  The moving parcel changes tracer and energy concentrations,
    # rather than changing a layer's diagnosed pressure thickness.
    fraction = np.clip(transfer_mass_kg_m2 / layer_mass_kg_m2[donor], 0.0, 1.0)
    humidity_flux = humidity[donor] * fraction
    energy_flux = moist_static_energy[donor] * fraction
    humidity[donor] = humidity[donor] - humidity_flux
    moist_static_energy[donor] = moist_static_energy[donor] - energy_flux
    receive_ratio = layer_mass_kg_m2[donor] / layer_mass_kg_m2[receiver]
    humidity[receiver] = humidity[receiver] + humidity_flux * receive_ratio
    moist_static_energy[receiver] = (
        moist_static_energy[receiver] + energy_flux * receive_ratio
    )


def evolve_closed_three_level_thermodynamic_column(
    lower_humidity: np.ndarray,
    midlevel_humidity: np.ndarray,
    upperlevel_humidity: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    omega_lower_mid_pa_s: np.ndarray,
    omega_mid_upper_pa_s: np.ndarray,
    *,
    dt_seconds: float,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
    surface_pressure_pa: float = 101_325.0,
    layer_heights_m: tuple[float, float, float] = (0.0, 3500.0, 8000.0),
    radiative_flux_w_m2: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    condensed_specific_humidity: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> ClosedThermodynamicColumnStep:
    """Advance a closed finite-volume vertical column.

    Positive ``omega`` is downward.  Each interface transfers the donor air's
    humidity and moist static energy with mass flux ``abs(omega) / g``.  The
    diagnosed pressure-layer masses remain fixed: their compensating horizontal
    convergence is outside this one-column operator and must be closed by the
    circulation component.  ``radiative_flux_w_m2`` is the *only* external
    energy source.  ``condensed_specific_humidity`` removes vapour while
    retaining moist static energy, so latent heat is released exactly once.

    This function intentionally has no lapse-profile relaxation, temperature
    clipping, precipitation sink, or empirical tuning term.  Callers must
    account for any exported condensate or rainfall in the water/energy budget
    they couple to it.
    """
    if dt_seconds <= 0.0 or not np.isfinite(dt_seconds):
        raise ValueError("dt_seconds must be finite and positive")
    if surface_pressure_pa <= 0.0 or not np.isfinite(surface_pressure_pa):
        raise ValueError("surface_pressure_pa must be finite and positive")
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(
        float(np.sum(fractions)), 1.0, atol=1e-12
    ):
        raise ValueError("layer_mass_fractions must be three positive values summing to one")
    heights = np.asarray(layer_heights_m, dtype=np.float64)
    if heights.shape != (3,) or not np.all(np.isfinite(heights)) or np.any(np.diff(heights) < 0.0):
        raise ValueError("layer_heights_m must be finite and non-decreasing")

    raw_fields = (
        lower_humidity, midlevel_humidity, upperlevel_humidity,
        lower_temperature_k, midlevel_temperature_k, upperlevel_temperature_k,
        omega_lower_mid_pa_s, omega_mid_upper_pa_s,
    )
    fields = tuple(np.asarray(item, dtype=np.float64) for item in raw_fields)
    shape = fields[0].shape
    if not shape or any(item.shape != shape for item in fields):
        raise ValueError("all closed-column fields must share a non-scalar shape")
    if any(not np.all(np.isfinite(item)) for item in fields):
        raise ValueError("closed-column fields must be finite")
    if any(np.any(item < 0.0) for item in fields[:3]):
        raise ValueError("layer humidities must be non-negative")

    radiation = _as_column_field(
        radiative_flux_w_m2, shape=shape, name="radiative_flux_w_m2"
    )
    condensed = _as_column_field(
        condensed_specific_humidity, shape=shape, name="condensed_specific_humidity"
    )
    humidity = [item.copy() for item in fields[:3]]
    temperature = fields[3:6]
    if any(np.any(condensed[index] > humidity[index]) for index in range(3)):
        raise ValueError("condensed_specific_humidity cannot exceed layer vapour")

    layer_mass = fractions * (float(surface_pressure_pa) / _GRAVITY_M_S2)
    geopotential = _GRAVITY_M_S2 * heights
    moist_static_energy = [
        _DRY_AIR_HEAT_CAPACITY_J_KG_K * temperature[index]
        + geopotential[index]
        + _LATENT_HEAT_VAPORIZATION_J_KG * humidity[index]
        for index in range(3)
    ]
    water_before = sum(
        float(np.sum(layer_mass[index] * humidity[index], dtype=np.float64))
        for index in range(3)
    )
    energy_before = sum(
        float(np.sum(layer_mass[index] * moist_static_energy[index], dtype=np.float64))
        for index in range(3)
    )

    omega_lower_mid, omega_mid_upper = fields[6:8]
    lower_mid_flux = np.abs(omega_lower_mid) / _GRAVITY_M_S2
    mid_upper_flux = np.abs(omega_mid_upper) / _GRAVITY_M_S2
    lower_mid_transfer = lower_mid_flux * dt_seconds
    mid_upper_transfer = mid_upper_flux * dt_seconds
    # Apply interfaces sequentially.  The second interface sees the middle
    # layer after the first transfer, exactly as a finite-volume update does.
    up_lower_mid = omega_lower_mid < 0.0
    if np.any(up_lower_mid):
        _transfer_layer_content(
            humidity, moist_static_energy, layer_mass, donor=0, receiver=1,
            transfer_mass_kg_m2=np.where(up_lower_mid, lower_mid_transfer, 0.0),
        )
    if np.any(~up_lower_mid):
        _transfer_layer_content(
            humidity, moist_static_energy, layer_mass, donor=1, receiver=0,
            transfer_mass_kg_m2=np.where(~up_lower_mid, lower_mid_transfer, 0.0),
        )
    up_mid_upper = omega_mid_upper < 0.0
    if np.any(up_mid_upper):
        _transfer_layer_content(
            humidity, moist_static_energy, layer_mass, donor=1, receiver=2,
            transfer_mass_kg_m2=np.where(up_mid_upper, mid_upper_transfer, 0.0),
        )
    if np.any(~up_mid_upper):
        _transfer_layer_content(
            humidity, moist_static_energy, layer_mass, donor=2, receiver=1,
            transfer_mass_kg_m2=np.where(~up_mid_upper, mid_upper_transfer, 0.0),
        )

    radiative_input = 0.0
    for index in range(3):
        moist_static_energy[index] = moist_static_energy[index] + (
            radiation[index] * dt_seconds / layer_mass[index]
        )
        radiative_input += float(np.sum(radiation[index] * dt_seconds, dtype=np.float64))
        # With moist static energy held fixed, this vapour decrease appears as
        # the corresponding latent warming.  Condensate remains in the column
        # for the reported water budget; a later fallout operator must export it
        # explicitly rather than silently treating it as lost here.
        humidity[index] = humidity[index] - condensed[index]

    output_temperature = [
        (moist_static_energy[index] - geopotential[index]
         - _LATENT_HEAT_VAPORIZATION_J_KG * humidity[index])
        / _DRY_AIR_HEAT_CAPACITY_J_KG_K
        for index in range(3)
    ]
    water_after = sum(
        float(np.sum(layer_mass[index] * (humidity[index] + condensed[index]), dtype=np.float64))
        for index in range(3)
    )
    energy_after = sum(
        float(np.sum(layer_mass[index] * moist_static_energy[index], dtype=np.float64))
        for index in range(3)
    )
    return ClosedThermodynamicColumnStep(
        *(item.astype(np.float32) for item in humidity),
        *(item.astype(np.float32) for item in output_temperature),
        lower_mid_flux.astype(np.float32),
        mid_upper_flux.astype(np.float32),
        water_after - water_before,
        energy_after - energy_before - radiative_input,
        radiative_input,
    )


def _partition_total_humidity(
    total: np.ndarray,
    midlevel: np.ndarray | None,
    upperlevel: np.ndarray | None,
    *,
    midlevel_fraction: float,
    upperlevel_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return non-negative lower/mid/upper reservoirs summing to ``total``."""
    total = np.clip(np.asarray(total, dtype=np.float64), 0.0, None)
    mid = (
        total * float(midlevel_fraction)
        if midlevel is None
        else np.clip(np.asarray(midlevel, dtype=np.float64), 0.0, None)
    )
    mid = np.minimum(mid, total)
    remaining = total - mid
    upper = (
        total * float(upperlevel_fraction)
        if upperlevel is None
        else np.clip(np.asarray(upperlevel, dtype=np.float64), 0.0, None)
    )
    upper = np.minimum(upper, remaining)
    return total - mid - upper, mid, upper


def _exchange_fraction(omega_pa_s: np.ndarray, dt_days: float, scale_pa_s: float, days: float) -> np.ndarray:
    strength = np.clip(np.abs(omega_pa_s) / scale_pa_s, 0.0, 6.0)
    return 1.0 - np.exp(-dt_days * strength / days)


def evolve_three_level_column(
    total_humidity: np.ndarray,
    lower_temperature_k: np.ndarray,
    lower_divergence_s: np.ndarray,
    midlevel_divergence_s: np.ndarray,
    upperlevel_divergence_s: np.ndarray,
    *,
    midlevel_humidity: np.ndarray | None = None,
    upperlevel_humidity: np.ndarray | None = None,
    midlevel_temperature_k: np.ndarray | None = None,
    upperlevel_temperature_k: np.ndarray | None = None,
    dt_days: float,
    lower_mid_pressure_depth_pa: float = 35_000.0,
    mid_upper_pressure_depth_pa: float = 30_000.0,
    vertical_velocity_scale_pa_s: float = 0.05,
    exchange_days: float = 2.0,
    midlevel_fraction: float = 0.25,
    upperlevel_fraction: float = 0.15,
    midlevel_height_m: float = 3500.0,
    upperlevel_height_m: float = 8000.0,
    lapse_rate_k_m: float = 6.5e-3,
    thermal_relaxation_days: float = 10.0,
    use_flux_form_exchange: bool = False,
    enforce_column_mass_closure: bool = False,
    lower_mass_fraction: float = 0.40,
    midlevel_mass_fraction: float = 0.35,
    upperlevel_mass_fraction: float = 0.25,
) -> PressureColumnStep:
    """Evolve a three-level column using pressure-coordinate continuity.

    Positive omega denotes downward pressure motion. At each interface,
    ``omega = 0.5 * dp * (div_below - div_above)`` is a centred discrete form
    of ``d omega / dp = -div(V)``. Negative omega moves vapor upward; positive
    omega moves it downward. Transfers are donor-limited and internal, so the
    combined vapor column is conserved to roundoff.
    """
    if dt_days <= 0.0 or lower_mid_pressure_depth_pa <= 0.0 or mid_upper_pressure_depth_pa <= 0.0:
        raise ValueError("dt_days and pressure depths must be positive")
    if vertical_velocity_scale_pa_s <= 0.0 or exchange_days <= 0.0 or thermal_relaxation_days <= 0.0:
        raise ValueError("vertical velocity scale and time scales must be positive")
    if not 0.0 <= midlevel_fraction <= 1.0 or not 0.0 <= upperlevel_fraction <= 1.0:
        raise ValueError("initial layer fractions must lie in [0, 1]")

    total = np.asarray(total_humidity, dtype=np.float64)
    lower_t = np.asarray(lower_temperature_k, dtype=np.float64)
    lower_div = np.asarray(lower_divergence_s, dtype=np.float64)
    mid_div = np.asarray(midlevel_divergence_s, dtype=np.float64)
    upper_div = np.asarray(upperlevel_divergence_s, dtype=np.float64)
    if not (
        total.shape == lower_t.shape == lower_div.shape == mid_div.shape == upper_div.shape
    ):
        raise ValueError("all column inputs must share one shape")

    lower_q, mid_q, upper_q = _partition_total_humidity(
        total,
        midlevel_humidity,
        upperlevel_humidity,
        midlevel_fraction=midlevel_fraction,
        upperlevel_fraction=upperlevel_fraction,
    )
    before = lower_q + mid_q + upper_q

    if enforce_column_mass_closure:
        weights = np.asarray(
            (lower_mass_fraction, midlevel_mass_fraction, upperlevel_mass_fraction), dtype=np.float64
        )
        if np.any(weights <= 0.0):
            raise ValueError("pressure-layer mass fractions must be positive")
        upper_div = -(weights[0] * lower_div + weights[1] * mid_div) / weights[2]

    omega_lower_mid = 0.5 * lower_mid_pressure_depth_pa * (lower_div - mid_div)
    omega_mid_upper = 0.5 * mid_upper_pressure_depth_pa * (mid_div - upper_div)

    lower_mid_fraction = (
        np.clip(
            np.abs(omega_lower_mid) * dt_days * 86400.0 / lower_mid_pressure_depth_pa,
            0.0, 1.0,
        )
        if use_flux_form_exchange
        else _exchange_fraction(
            omega_lower_mid, dt_days, vertical_velocity_scale_pa_s, exchange_days
        )
    )
    lower_to_mid = np.where(omega_lower_mid < 0.0, lower_q * lower_mid_fraction, 0.0)
    mid_to_lower = np.where(omega_lower_mid > 0.0, mid_q * lower_mid_fraction, 0.0)
    lower_q = lower_q - lower_to_mid + mid_to_lower
    mid_q = mid_q + lower_to_mid - mid_to_lower

    mid_upper_fraction = (
        np.clip(
            np.abs(omega_mid_upper) * dt_days * 86400.0 / mid_upper_pressure_depth_pa,
            0.0, 1.0,
        )
        if use_flux_form_exchange
        else _exchange_fraction(
            omega_mid_upper, dt_days, vertical_velocity_scale_pa_s, exchange_days
        )
    )
    mid_to_upper = np.where(omega_mid_upper < 0.0, mid_q * mid_upper_fraction, 0.0)
    upper_to_mid = np.where(omega_mid_upper > 0.0, upper_q * mid_upper_fraction, 0.0)
    mid_q = mid_q - mid_to_upper + upper_to_mid
    upper_q = upper_q + mid_to_upper - upper_to_mid

    mid_ref = lower_t - lapse_rate_k_m * midlevel_height_m
    upper_ref = lower_t - lapse_rate_k_m * upperlevel_height_m
    relax = 1.0 - np.exp(-dt_days / thermal_relaxation_days)
    mid_t_in = mid_ref if midlevel_temperature_k is None else np.asarray(midlevel_temperature_k, dtype=np.float64)
    upper_t_in = upper_ref if upperlevel_temperature_k is None else np.asarray(upperlevel_temperature_k, dtype=np.float64)
    if mid_t_in.shape != total.shape or upper_t_in.shape != total.shape:
        raise ValueError("layer temperatures must match humidity shape")
    mid_t = np.clip(mid_t_in + relax * (mid_ref - mid_t_in), 150.0, 350.0)
    upper_t = np.clip(upper_t_in + relax * (upper_ref - upper_t_in), 150.0, 350.0)

    after = lower_q + mid_q + upper_q
    scale = max(float(np.sum(np.abs(before), dtype=np.float64)), 1.0)
    residual = float(np.sum(after - before, dtype=np.float64)) / scale
    return PressureColumnStep(
        lower_q.astype(np.float32),
        mid_q.astype(np.float32),
        upper_q.astype(np.float32),
        mid_t.astype(np.float32),
        upper_t.astype(np.float32),
        omega_lower_mid.astype(np.float32),
        omega_mid_upper.astype(np.float32),
        residual,
    )
