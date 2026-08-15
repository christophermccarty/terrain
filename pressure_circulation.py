"""Horizontally resolved divergent-wind closure for a pressure column.

The prognostic wind solver carries rotational and divergent components together.
For an optional three-level pressure-column experiment, this module diagnoses the
layer-weighted horizontal mass-flux residual and applies only the minimum
diagnostic divergent correction needed to the upper level.  The correction is
constructed in longitude Fourier space plus a zonally symmetric meridional
flux, so its cost scales linearly with horizontal cell count rather than with a
slow global relaxation solve.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, NamedTuple

import numpy as np

try:
    from numba import njit  # pyright: ignore[reportMissingImports]
    _NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    _NUMBA_AVAILABLE = False


_STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8
_MOMENTUM_WAVE_FRACTION = 0.5
"""Maximum resolved PGF speed increment as a fraction of gravity-wave speed."""


def _cosine_area_balanced_zonal_anomaly(field: np.ndarray) -> np.ndarray:
    """Return the zonal, globally mass-balanced component used by circulation."""
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 * values.shape[0]:
        raise ValueError("circulation anomaly fields must use a two-dimensional 2:1 grid")
    if not np.all(np.isfinite(values)):
        raise ValueError("circulation anomaly fields must be finite")
    h, _ = values.shape
    latitude = 0.5 * np.pi - (np.arange(h) + 0.5) * np.pi / h
    weights = np.cos(latitude)[:, None]
    zonal = np.mean(values, axis=1, keepdims=True)
    balanced = zonal - np.sum(zonal * weights) / np.sum(weights)
    return np.broadcast_to(balanced, values.shape).copy()


def column_mse_storage_tendency_w_m2(
    lower_humidity_before: np.ndarray,
    midlevel_humidity_before: np.ndarray,
    upperlevel_humidity_before: np.ndarray,
    lower_temperature_before_k: np.ndarray,
    midlevel_temperature_before_k: np.ndarray,
    upperlevel_temperature_before_k: np.ndarray,
    lower_humidity_after: np.ndarray,
    midlevel_humidity_after: np.ndarray,
    upperlevel_humidity_after: np.ndarray,
    lower_temperature_after_k: np.ndarray,
    midlevel_temperature_after_k: np.ndarray,
    upperlevel_temperature_after_k: np.ndarray,
    *,
    dt_seconds: float,
    surface_pressure_pa: float,
    gravity_m_s2: float = 9.80665,
    cp_dry_j_kg_k: float = 1004.0,
    latent_heat_j_kg: float = 2.5e6,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
    layer_heights_m: tuple[float, float, float] = (0.0, 3500.0, 8000.0),
) -> np.ndarray:
    """Return the transient three-level atmospheric MSE storage [W m-2].

    This is the energy analogue of the pressure-column water storage term. It
    includes dry-static, geopotential, and vapour latent energy in the three
    resolved air layers. Phase conversion is internal: the closed column's
    temperature response carries its compensating sensible-heat storage.
    """
    if dt_seconds <= 0.0 or surface_pressure_pa <= 0.0:
        raise ValueError("dt_seconds and surface_pressure_pa must be positive")
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    heights = np.asarray(layer_heights_m, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(float(np.sum(fractions)), 1.0):
        raise ValueError("layer_mass_fractions must be three positive values summing to one")
    if heights.shape != (3,) or not np.all(np.isfinite(heights)):
        raise ValueError("layer_heights_m must be three finite values")
    fields = tuple(np.asarray(value, dtype=np.float64) for value in (
        lower_humidity_before, midlevel_humidity_before, upperlevel_humidity_before,
        lower_temperature_before_k, midlevel_temperature_before_k, upperlevel_temperature_before_k,
        lower_humidity_after, midlevel_humidity_after, upperlevel_humidity_after,
        lower_temperature_after_k, midlevel_temperature_after_k, upperlevel_temperature_after_k,
    ))
    shape = fields[0].shape
    if len(shape) != 2 or shape[1] != 2 * shape[0] or any(value.shape != shape for value in fields):
        raise ValueError("MSE storage fields must share a two-dimensional 2:1 grid")
    if any(not np.all(np.isfinite(value)) for value in fields) or any(np.any(value < 0.0) for value in fields[:3] + fields[6:9]):
        raise ValueError("MSE storage fields must be finite with non-negative humidity")
    mass = fractions * float(surface_pressure_pa) / float(gravity_m_s2)
    before = 0.0
    after = 0.0
    for layer in range(3):
        before += mass[layer] * (
            float(cp_dry_j_kg_k) * fields[3 + layer]
            + float(gravity_m_s2) * heights[layer]
            + float(latent_heat_j_kg) * fields[layer]
        )
        after += mass[layer] * (
            float(cp_dry_j_kg_k) * fields[9 + layer]
            + float(gravity_m_s2) * heights[layer]
            + float(latent_heat_j_kg) * fields[6 + layer]
        )
    return ((after - before) / float(dt_seconds)).astype(np.float32)


class HorizontalMassClosure(NamedTuple):
    upper_u_correction: np.ndarray
    upper_v_correction: np.ndarray
    residual_before_s: float
    residual_after_s: float
    equatorial_throughflow_before_m_s: float
    equatorial_throughflow_after_m_s: float


class DiabaticInterfaceMassFlux(NamedTuple):
    """Zonal-mean, mass-consistent pressure-interface circulation diagnosis."""

    omega_lower_mid_pa_s: np.ndarray
    omega_mid_upper_pa_s: np.ndarray
    lower_divergence_s: np.ndarray
    midlevel_divergence_s: np.ndarray
    upperlevel_divergence_s: np.ndarray
    latent_heating_w_m2: np.ndarray
    lower_mid_vertical_courant_max: float
    mid_upper_vertical_courant_max: float


class PressureCoordinateCirculation(NamedTuple):
    """One mass-consistent three-level horizontal/vertical circulation state."""

    lower_u: np.ndarray
    lower_v: np.ndarray
    midlevel_u: np.ndarray
    midlevel_v: np.ndarray
    upperlevel_u: np.ndarray
    upperlevel_v: np.ndarray
    interface_mass_flux: DiabaticInterfaceMassFlux


class LargeScaleHeatingStep(NamedTuple):
    """Radiatively adjusted latent-heating anomaly for diagnosed overturning."""

    heating_w_m2: np.ndarray
    radiative_adjustment_time_s: np.ndarray


class MSEConstrainedPressureCoordinateCirculation(NamedTuple):
    """Mass-closed circulation diagnosed from a zonal MSE transport budget."""

    circulation: PressureCoordinateCirculation
    diabatic_forcing_w_m2: np.ndarray
    meridional_mse_transport_w_m: np.ndarray
    energy_closure_residual_w: float
    lower_upper_mse_contrast_j_kg: np.ndarray


class ThreeBranchMSEConstrainedPressureCoordinateCirculation(NamedTuple):
    """Three-branch MSE/mass closure plus its implied layer forcing."""

    circulation: PressureCoordinateCirculation
    diabatic_forcing_w_m2: np.ndarray
    meridional_mse_transport_w_m: np.ndarray
    energy_closure_residual_w: float
    mse_variance_j2_kg2: np.ndarray
    lower_diabatic_deposition_w_m2: np.ndarray
    midlevel_diabatic_deposition_w_m2: np.ndarray
    upperlevel_diabatic_deposition_w_m2: np.ndarray


class ThreeLevelZonalMomentumStep(NamedTuple):
    """Pressure-level horizontal momentum after one split conservative step."""

    lower_u: np.ndarray
    lower_v: np.ndarray
    midlevel_u: np.ndarray
    midlevel_v: np.ndarray
    upperlevel_u: np.ndarray
    upperlevel_v: np.ndarray
    lower_meridional_pressure_gradient_m_s2: np.ndarray
    midlevel_meridional_pressure_gradient_m_s2: np.ndarray
    upperlevel_meridional_pressure_gradient_m_s2: np.ndarray
    vertical_courant_max: float


class JointPressureColumnStep(NamedTuple):
    """One adaptive joint momentum/MSE/vertical-exchange integration."""

    lower_humidity: np.ndarray
    midlevel_humidity: np.ndarray
    upperlevel_humidity: np.ndarray
    lower_temperature: np.ndarray
    midlevel_temperature: np.ndarray
    upperlevel_temperature: np.ndarray
    lower_u: np.ndarray
    lower_v: np.ndarray
    midlevel_u: np.ndarray
    midlevel_v: np.ndarray
    upperlevel_u: np.ndarray
    upperlevel_v: np.ndarray
    interface_mass_flux: DiabaticInterfaceMassFlux
    substeps: int
    vertical_courant_max: float
    water_residual_kg_m2: float
    moist_static_energy_residual_j_m2: float


class JointPressureColumnRuntimeStep(NamedTuple):
    """One complete pressure-column transition used by the climate adapter.

    This packages the horizontal MSE leg, adaptive joint circulation/vertical
    integration, and the ensuing phase conversion.  Keeping those operations
    together prevents the runtime path from applying either vertical exchange
    or latent heating a second time.
    """

    joint: JointPressureColumnStep
    lower_condensed_specific_humidity: np.ndarray
    midlevel_condensed_specific_humidity: np.ndarray
    upperlevel_condensed_specific_humidity: np.ndarray
    horizontal_mse_energy_residual_j: float
    horizontal_mse_relative_energy_residual: float
    phase_water_residual_kg_m2: float
    phase_moist_static_energy_residual_j_m2: float


class JointPressureColumnCouplingResidual(NamedTuple):
    """Residual of one simultaneous water/MSE-transport pressure-column candidate."""

    runtime: JointPressureColumnRuntimeStep
    cloud_condensate_kg_m2: np.ndarray
    precipitating_hydrometeors_kg_m2: np.ndarray
    fallout_kg_m2: np.ndarray
    diagnosed_water_forcing_kg_m2_s: np.ndarray
    diagnosed_heating_w_m2: np.ndarray
    water_forcing_residual_kg_m2_s: np.ndarray
    heating_residual_w_m2: np.ndarray


class JointPressureColumnCouplingSolve(NamedTuple):
    """A converged simultaneous zonal water/MSE-transport candidate."""

    heating_w_m2: np.ndarray
    water_forcing_kg_m2_s: np.ndarray
    residual: JointPressureColumnCouplingResidual
    iterations: int


class JointPressureColumnSimultaneousRuntimeStep(NamedTuple):
    """One converged pressure-column transition including water reservoirs."""

    runtime: JointPressureColumnRuntimeStep
    cloud_condensate_kg_m2: np.ndarray
    precipitating_hydrometeors_kg_m2: np.ndarray
    fallout_kg_m2: np.ndarray
    heating_w_m2: np.ndarray
    water_forcing_kg_m2_s: np.ndarray
    iterations: int
    cloud_transport_relative_residual: float
    hydrometeor_transport_relative_residual: float | None


class PrognosticPressureCoordinateTransportStep(NamedTuple):
    """One explicit three-level horizontal pressure-coordinate transition.

    This is intentionally a state evolution, not a diagnostic branch solve.
    Layer vapour and moist static energy are transported on the same
    finite-volume faces; the only external water input is surface evaporation
    into the lower layer.  Pressure-level winds then receive their independent
    hydrostatic-pressure-gradient/Coriolis update with zero interface exchange.
    Vertical transport and phase conversion remain separate future operators,
    so this primitive cannot silently turn an instantaneous water or MSE
    residual into an overturning velocity.
    """

    lower_humidity: np.ndarray
    midlevel_humidity: np.ndarray
    upperlevel_humidity: np.ndarray
    lower_temperature: np.ndarray
    midlevel_temperature: np.ndarray
    upperlevel_temperature: np.ndarray
    lower_u: np.ndarray
    lower_v: np.ndarray
    midlevel_u: np.ndarray
    midlevel_v: np.ndarray
    upperlevel_u: np.ndarray
    upperlevel_v: np.ndarray
    water_relative_residual: float
    moist_static_energy_relative_residual: float
    horizontal_substeps: int
    horizontal_courant_max: float


class PrognosticPressureLayerMassStep(NamedTuple):
    """One conservative pressure-thickness and interface-flux transition."""

    lower_pressure_depth_pa: np.ndarray
    midlevel_pressure_depth_pa: np.ndarray
    upperlevel_pressure_depth_pa: np.ndarray
    lower_mid_interface_mass_flux_kg_m2_s: np.ndarray
    mid_upper_interface_mass_flux_kg_m2_s: np.ndarray
    relative_mass_residual: float
    horizontal_substeps: int
    horizontal_courant_max: float
    vertical_substeps: int
    vertical_courant_max: float


class VariableMassPressureCoordinateTransportStep(NamedTuple):
    """Conservative variable-pressure three-layer water/MSE transition."""

    lower_pressure_depth_pa: np.ndarray
    midlevel_pressure_depth_pa: np.ndarray
    upperlevel_pressure_depth_pa: np.ndarray
    lower_humidity: np.ndarray
    midlevel_humidity: np.ndarray
    upperlevel_humidity: np.ndarray
    lower_temperature: np.ndarray
    midlevel_temperature: np.ndarray
    upperlevel_temperature: np.ndarray
    lower_mid_interface_mass_flux_kg_m2_s: np.ndarray
    mid_upper_interface_mass_flux_kg_m2_s: np.ndarray
    water_relative_residual: float
    moist_static_energy_relative_residual: float
    horizontal_substeps: int
    horizontal_courant_max: float
    vertical_substeps: int
    vertical_courant_max: float


class HydrostaticSigmaContinuityStep(NamedTuple):
    """Interface fluxes implied by hydrostatic sigma-coordinate continuity."""

    lower_pressure_mass_tendency_kg_m2_s: np.ndarray
    midlevel_pressure_mass_tendency_kg_m2_s: np.ndarray
    upperlevel_pressure_mass_tendency_kg_m2_s: np.ndarray
    lower_mid_interface_mass_flux_kg_m2_s: np.ndarray
    mid_upper_interface_mass_flux_kg_m2_s: np.ndarray
    surface_pressure_tendency_pa_s: np.ndarray
    relative_continuity_residual: float
    horizontal_substeps: int
    horizontal_courant_max: float


class HydrostaticSigmaMassMomentumStep(NamedTuple):
    """One jointly CFL-controlled horizontal sigma mass/momentum transition."""

    lower_pressure_depth_pa: np.ndarray
    midlevel_pressure_depth_pa: np.ndarray
    upperlevel_pressure_depth_pa: np.ndarray
    lower_u: np.ndarray
    lower_v: np.ndarray
    midlevel_u: np.ndarray
    midlevel_v: np.ndarray
    upperlevel_u: np.ndarray
    upperlevel_v: np.ndarray
    lower_mid_interface_mass_flux_kg_m2_s: np.ndarray
    mid_upper_interface_mass_flux_kg_m2_s: np.ndarray
    relative_mass_residual: float
    horizontal_momentum_relative_residual: float
    substeps: int
    horizontal_courant_max: float
    vertical_courant_max: float
    lower_humidity: np.ndarray | None = None
    midlevel_humidity: np.ndarray | None = None
    upperlevel_humidity: np.ndarray | None = None
    lower_temperature: np.ndarray | None = None
    midlevel_temperature: np.ndarray | None = None
    upperlevel_temperature: np.ndarray | None = None
    lower_cloud_condensate_kg_m2: np.ndarray | None = None
    midlevel_cloud_condensate_kg_m2: np.ndarray | None = None
    upperlevel_cloud_condensate_kg_m2: np.ndarray | None = None
    lower_hydrometeors_kg_m2: np.ndarray | None = None
    midlevel_hydrometeors_kg_m2: np.ndarray | None = None
    upperlevel_hydrometeors_kg_m2: np.ndarray | None = None
    water_relative_residual: float | None = None
    moist_static_energy_relative_residual: float | None = None
    horizontal_mse_convergence_w_m2: np.ndarray | None = None


class HydrostaticSigmaPressureCoordinateTransportStep(NamedTuple):
    """One fully continuity-closed hydrostatic sigma transport transition."""

    transport: VariableMassPressureCoordinateTransportStep
    continuity: HydrostaticSigmaContinuityStep


class HydrostaticSigmaPhaseReservoirStep(NamedTuple):
    """Hydrostatic sigma transition with layer-resolved phase reservoirs."""

    transport: HydrostaticSigmaPressureCoordinateTransportStep
    lower_cloud_condensate_kg_m2: np.ndarray
    midlevel_cloud_condensate_kg_m2: np.ndarray
    upperlevel_cloud_condensate_kg_m2: np.ndarray
    lower_hydrometeors_kg_m2: np.ndarray
    midlevel_hydrometeors_kg_m2: np.ndarray
    upperlevel_hydrometeors_kg_m2: np.ndarray
    fallout_kg_m2: np.ndarray
    water_relative_residual: float
    moist_static_energy_relative_residual: float
    momentum: object


class VariableMassPressureMomentumStep(NamedTuple):
    """Pressure-level momentum after hydrostatic forcing and mass exchange."""

    lower_u: np.ndarray
    lower_v: np.ndarray
    midlevel_u: np.ndarray
    midlevel_v: np.ndarray
    upperlevel_u: np.ndarray
    upperlevel_v: np.ndarray
    lower_meridional_pressure_gradient_m_s2: np.ndarray
    midlevel_meridional_pressure_gradient_m_s2: np.ndarray
    upperlevel_meridional_pressure_gradient_m_s2: np.ndarray
    vertical_courant_max: float
    horizontal_momentum_relative_residual: float


def diabatic_interface_mass_flux(
    precipitation_mm_day: np.ndarray | None,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    *,
    dt_seconds: float,
    surface_pressure_pa: float,
    lower_mid_pressure_depth_pa: float,
    mid_upper_pressure_depth_pa: float,
    gravity_m_s2: float = 9.80665,
    cp_dry_j_kg_k: float = 1004.0,
    latent_heat_j_kg: float = 2.5e6,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
) -> DiabaticInterfaceMassFlux:
    """Diagnose a vertically closed large-scale omega from diabatic heating.

    The existing raw independent layer winds contain divergent modes that imply
    pressure-interface mass transfer of tens of complete layers per day.  This
    operator deliberately does not cap those omegas.  It bypasses them and
    derives the large-scale overturning from the preceding raw-column
    precipitation: ``Lv * P`` is deposited uniformly per unit free-troposphere
    mass, then balanced against each interface's resolved static stability.

    Precipitation and temperatures are zonally averaged before the diagnosis.
    That is intentional: a three-level climate column represents the Hadley/
    seasonal mean overturning, whereas individual grid-cell convective plumes
    belong to the condensate closure.  The returned layer divergences solve the
    same centred continuity relations as the production omega calculation and
    have zero 0.40/0.35/0.25 mass-weighted column divergence at every cell.
    """
    if dt_seconds <= 0.0 or surface_pressure_pa <= 0.0:
        raise ValueError("dt_seconds and surface_pressure_pa must be positive")
    if lower_mid_pressure_depth_pa <= 0.0 or mid_upper_pressure_depth_pa <= 0.0:
        raise ValueError("pressure depths must be positive")
    if gravity_m_s2 <= 0.0 or cp_dry_j_kg_k <= 0.0 or latent_heat_j_kg <= 0.0:
        raise ValueError("physical constants must be positive")
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(np.sum(fractions), 1.0):
        raise ValueError("layer mass fractions must be three positive values summing to one")
    lower = np.asarray(lower_temperature_k, dtype=np.float64)
    middle = np.asarray(midlevel_temperature_k, dtype=np.float64)
    upper = np.asarray(upperlevel_temperature_k, dtype=np.float64)
    if lower.ndim != 2 or lower.shape[1] != 2 * lower.shape[0] or middle.shape != lower.shape or upper.shape != lower.shape:
        raise ValueError("temperature layers must share a two-dimensional 2:1 grid")
    if precipitation_mm_day is None:
        precipitation = np.zeros_like(lower)
    else:
        precipitation = np.asarray(precipitation_mm_day, dtype=np.float64)
        if precipitation.shape != lower.shape:
            raise ValueError("precipitation must match the temperature grid")
        if not np.all(np.isfinite(precipitation)):
            raise ValueError("precipitation must be finite")

    # kg m-2 day-1 is numerically equal to mm day-1.  The planetwide mean
    # latent release is balanced by the host radiation/surface step; only its
    # cosine-area-weighted anomaly can drive a closed large-scale overturning.
    # Without that removal every latitude ascends, which cannot be represented
    # by a horizontal circulation and leads to arbitrary polar return flow.
    zonal_precipitation = np.maximum(np.mean(precipitation, axis=1, keepdims=True), 0.0)
    cos_lat = np.cos(np.radians(90.0 - (np.arange(lower.shape[0]) + 0.5) * 180.0 / lower.shape[0]))[:, None]
    mean_precipitation = np.sum(zonal_precipitation * cos_lat) / np.sum(cos_lat)
    latent_flux = float(latent_heat_j_kg) * (
        zonal_precipitation - mean_precipitation
    ) / 86400.0
    return diabatic_interface_mass_flux_from_heating(
        np.broadcast_to(latent_flux, lower.shape), lower, middle, upper,
        dt_seconds=dt_seconds, surface_pressure_pa=surface_pressure_pa,
        lower_mid_pressure_depth_pa=lower_mid_pressure_depth_pa,
        mid_upper_pressure_depth_pa=mid_upper_pressure_depth_pa,
        gravity_m_s2=gravity_m_s2, cp_dry_j_kg_k=cp_dry_j_kg_k,
        layer_mass_fractions=layer_mass_fractions,
    )


def diabatic_interface_mass_flux_from_heating(
    large_scale_heating_w_m2: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    *,
    dt_seconds: float,
    surface_pressure_pa: float,
    lower_mid_pressure_depth_pa: float,
    mid_upper_pressure_depth_pa: float,
    gravity_m_s2: float = 9.80665,
    cp_dry_j_kg_k: float = 1004.0,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
) -> DiabaticInterfaceMassFlux:
    """Diagnose closed interface mass flux from a signed heating anomaly."""
    lower = np.asarray(lower_temperature_k, dtype=np.float64)
    middle = np.asarray(midlevel_temperature_k, dtype=np.float64)
    upper = np.asarray(upperlevel_temperature_k, dtype=np.float64)
    heating = np.asarray(large_scale_heating_w_m2, dtype=np.float64)
    if lower.ndim != 2 or lower.shape[1] != 2 * lower.shape[0] or middle.shape != lower.shape or upper.shape != lower.shape:
        raise ValueError("temperature layers must share a two-dimensional 2:1 grid")
    if heating.shape != lower.shape or not np.all(np.isfinite(heating)):
        raise ValueError("large-scale heating must be finite and match the temperature grid")
    if dt_seconds <= 0.0 or surface_pressure_pa <= 0.0 or lower_mid_pressure_depth_pa <= 0.0 or mid_upper_pressure_depth_pa <= 0.0:
        raise ValueError("time, pressure, and pressure depths must be positive")
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(np.sum(fractions), 1.0):
        raise ValueError("layer mass fractions must be three positive values summing to one")
    # The forcing is a large-scale zonal anomaly.  Re-average here so callers
    # cannot accidentally turn local plume noise into a global circulation.
    latent_flux = np.broadcast_to(np.mean(heating, axis=1, keepdims=True), lower.shape).copy()
    column_mass = float(surface_pressure_pa) / float(gravity_m_s2)
    free_troposphere_mass = column_mass * (fractions[1] + fractions[2])
    heating_k_s = latent_flux / (float(cp_dry_j_kg_k) * free_troposphere_mass)
    lower_zonal = np.mean(lower, axis=1, keepdims=True)
    middle_zonal = np.mean(middle, axis=1, keepdims=True)
    upper_zonal = np.mean(upper, axis=1, keepdims=True)
    # Static stability is the vertical *potential-temperature* gradient, not
    # a raw temperature difference. A normally stratified troposphere cools
    # with height, so raw T differences can approach zero even while theta
    # increases strongly upward. Using raw T produced a singular omega during
    # phase-heating events. Layer-centre pressures follow the documented
    # 0.40/0.35/0.25 pressure-mass partition; no stability floor or omega cap
    # is introduced.
    layer_edges = float(surface_pressure_pa) * np.array(
        (1.0, 1.0 - fractions[0], fractions[2], 0.0), dtype=np.float64
    )
    layer_pressures = 0.5 * (layer_edges[:-1] + layer_edges[1:])
    kappa = 287.05 / float(cp_dry_j_kg_k)
    reference_pressure = float(surface_pressure_pa)
    lower_theta = lower_zonal * (reference_pressure / layer_pressures[0]) ** kappa
    middle_theta = middle_zonal * (reference_pressure / layer_pressures[1]) ** kappa
    upper_theta = upper_zonal * (reference_pressure / layer_pressures[2]) ** kappa
    lower_mid_stability = (middle_theta - lower_theta) / float(lower_mid_pressure_depth_pa)
    mid_upper_stability = (upper_theta - middle_theta) / float(mid_upper_pressure_depth_pa)
    # An unstably stratified interface is a microphysical convective-adjustment
    # problem, not a hydrostatic large-scale omega diagnostic.  Leave that
    # branch at zero instead of hiding an arbitrary stability floor here.
    omega_lower_mid = np.divide(
        -heating_k_s, lower_mid_stability,
        out=np.zeros_like(heating_k_s), where=lower_mid_stability > 0.0,
    )
    omega_mid_upper = np.divide(
        -heating_k_s, mid_upper_stability,
        out=np.zeros_like(heating_k_s), where=mid_upper_stability > 0.0,
    )
    omega_lower_mid = np.broadcast_to(omega_lower_mid, lower.shape).copy()
    omega_mid_upper = np.broadcast_to(omega_mid_upper, lower.shape).copy()

    # Invert the centred continuity relations used by the pressure column:
    # omega_lm=0.5*dp_lm*(d_lower-d_mid), omega_mu=0.5*dp_mu*(d_mid-d_upper),
    # plus weighted column divergence exactly equal to zero.
    lower_mid_difference = 2.0 * omega_lower_mid / float(lower_mid_pressure_depth_pa)
    mid_upper_difference = 2.0 * omega_mid_upper / float(mid_upper_pressure_depth_pa)
    mid_divergence = -fractions[0] * lower_mid_difference + fractions[2] * mid_upper_difference
    lower_divergence = mid_divergence + lower_mid_difference
    upper_divergence = mid_divergence - mid_upper_difference
    return DiabaticInterfaceMassFlux(
        omega_lower_mid.astype(np.float32),
        omega_mid_upper.astype(np.float32),
        lower_divergence.astype(np.float32),
        mid_divergence.astype(np.float32),
        upper_divergence.astype(np.float32),
        np.broadcast_to(latent_flux, lower.shape).astype(np.float32),
        float(np.max(np.abs(omega_lower_mid)) * dt_seconds / float(lower_mid_pressure_depth_pa)),
        float(np.max(np.abs(omega_mid_upper)) * dt_seconds / float(mid_upper_pressure_depth_pa)),
    )


def evolve_large_scale_heating_reservoir(
    previous_heating_w_m2: np.ndarray | None,
    condensation_mm_day: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    *,
    dt_seconds: float,
    surface_pressure_pa: float,
    gravity_m_s2: float = 9.80665,
    cp_dry_j_kg_k: float = 1004.0,
    latent_heat_j_kg: float = 2.5e6,
) -> LargeScaleHeatingStep:
    """Low-pass large-scale latent heating using a derived radiative timescale.

    The reservoir stores only cosine-area-balanced zonal anomalies.  Its
    e-folding time is free-tropospheric heat capacity divided by linearized
    thermal emission (``4 sigma T^3``), so it introduces no configurable
    damping or circulation-strength coefficient.
    """
    lower, middle, upper, condensation = (
        np.asarray(value, dtype=np.float64)
        for value in (lower_temperature_k, midlevel_temperature_k, upperlevel_temperature_k, condensation_mm_day)
    )
    if lower.ndim != 2 or lower.shape[1] != 2 * lower.shape[0] or any(value.shape != lower.shape for value in (middle, upper, condensation)):
        raise ValueError("heating-reservoir fields must share a two-dimensional 2:1 grid")
    if dt_seconds <= 0.0 or surface_pressure_pa <= 0.0 or gravity_m_s2 <= 0.0 or cp_dry_j_kg_k <= 0.0:
        raise ValueError("heating-reservoir constants must be positive")
    previous = np.zeros_like(lower) if previous_heating_w_m2 is None else np.asarray(previous_heating_w_m2, dtype=np.float64)
    if previous.shape != lower.shape or not np.all(np.isfinite(previous)):
        raise ValueError("previous heating must be finite and match the temperature grid")
    zonal_condensation = np.maximum(np.mean(condensation, axis=1, keepdims=True), 0.0)
    cos_lat = np.cos(np.radians(90.0 - (np.arange(lower.shape[0]) + 0.5) * 180.0 / lower.shape[0]))[:, None]
    mean_condensation = np.sum(zonal_condensation * cos_lat) / np.sum(cos_lat)
    forcing = float(latent_heat_j_kg) * (zonal_condensation - mean_condensation) / 86400.0
    free_temperature = 0.35 * np.mean(middle, axis=1, keepdims=True) + 0.25 * np.mean(upper, axis=1, keepdims=True)
    free_mass = float(surface_pressure_pa) / float(gravity_m_s2) * 0.60
    radiative_stiffness = 4.0 * _STEFAN_BOLTZMANN_W_M2_K4 * np.maximum(free_temperature, 180.0) ** 3
    adjustment_time = free_mass * float(cp_dry_j_kg_k) / radiative_stiffness
    decay = np.exp(-float(dt_seconds) / adjustment_time)
    updated = decay * np.mean(previous, axis=1, keepdims=True) + (1.0 - decay) * forcing
    return LargeScaleHeatingStep(
        np.broadcast_to(updated, lower.shape).astype(np.float32),
        np.broadcast_to(adjustment_time, lower.shape).astype(np.float32),
    )


def evolve_three_level_zonal_momentum(
    lower_u_m_s: np.ndarray,
    lower_v_m_s: np.ndarray,
    midlevel_u_m_s: np.ndarray,
    midlevel_v_m_s: np.ndarray,
    upperlevel_u_m_s: np.ndarray,
    upperlevel_v_m_s: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    omega_lower_mid_pa_s: np.ndarray,
    omega_mid_upper_pa_s: np.ndarray,
    *,
    dt_seconds: float,
    radius_m: float,
    sidereal_day_hours: float,
    surface_pressure_pa: float,
    gravity_m_s2: float = 9.80665,
    gas_constant_dry_j_kg_k: float = 287.05,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
) -> ThreeLevelZonalMomentumStep:
    """Advance pressure-level momentum with PGF, Coriolis, and omega exchange.

    Hydrostatic layer thickness supplies the zonal-mean meridional pressure
    gradient.  Coriolis rotation under that constant-in-step acceleration is
    integrated analytically, so the update has no near-equatorial ``1/f``
    singularity or explicit inertial-step stability limit.  The two interface
    mass transfers then exchange both horizontal momentum components
    conservatively; integration is subdivided for the diagnosed vertical CFL
    rather than clipping a physical interface flux.
    """
    if dt_seconds <= 0.0 or radius_m <= 0.0 or sidereal_day_hours <= 0.0 or surface_pressure_pa <= 0.0:
        raise ValueError("time, radius, rotation period, and pressure must be positive")
    if gravity_m_s2 <= 0.0 or gas_constant_dry_j_kg_k <= 0.0:
        raise ValueError("gravity and gas constant must be positive")
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(float(np.sum(fractions)), 1.0):
        raise ValueError("layer_mass_fractions must be three positive values summing to one")
    raw = (
        lower_u_m_s, lower_v_m_s, midlevel_u_m_s, midlevel_v_m_s,
        upperlevel_u_m_s, upperlevel_v_m_s, lower_temperature_k,
        midlevel_temperature_k, upperlevel_temperature_k,
        omega_lower_mid_pa_s, omega_mid_upper_pa_s,
    )
    fields = tuple(np.asarray(value, dtype=np.float64) for value in raw)
    shape = fields[0].shape
    if len(shape) != 2 or shape[1] != 2 * shape[0] or any(value.shape != shape for value in fields):
        raise ValueError("momentum fields must share a two-dimensional 2:1 grid")
    if any(not np.all(np.isfinite(value)) for value in fields):
        raise ValueError("momentum fields must be finite")
    h, _ = shape
    layer_pressure = float(surface_pressure_pa) * np.array(
        (0.80, 0.425, 0.125), dtype=np.float64
    )
    temperatures = fields[6:9]
    dy_m = float(radius_m) * np.pi / h
    pressure_gradient: list[np.ndarray] = []
    for temperature, pressure in zip(temperatures, layer_pressure, strict=True):
        geopotential = (
            float(gas_constant_dry_j_kg_k)
            * np.mean(temperature, axis=1)
            * np.log(float(surface_pressure_pa) / pressure)
        )
        # Rows run north-to-south, hence d/dy_north=-d/drow.  The pressure
        # acceleration is -dPhi/dy_north = +dPhi/drow.
        acceleration = np.gradient(geopotential, dy_m, edge_order=1)
        pressure_gradient.append(np.broadcast_to(acceleration[:, None], shape).copy())
    latitude = 0.5 * np.pi - (np.arange(h) + 0.5) * np.pi / h
    coriolis = 4.0 * np.pi / (float(sidereal_day_hours) * 3600.0) * np.sin(latitude)
    theta = coriolis[:, None] * float(dt_seconds)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    small = np.abs(theta) < 1.0e-5
    sin_over_f = np.where(
        small,
        float(dt_seconds) * (1.0 - theta**2 / 6.0),
        sine / coriolis[:, None],
    )
    one_minus_cos_over_f = np.where(
        small,
        0.5 * float(dt_seconds) * theta,
        (1.0 - cosine) / coriolis[:, None],
    )
    velocity = [fields[0].copy(), fields[1].copy(), fields[2].copy(), fields[3].copy(), fields[4].copy(), fields[5].copy()]
    for level in range(3):
        u, v = velocity[2 * level], velocity[2 * level + 1]
        acceleration = pressure_gradient[level]
        velocity[2 * level] = cosine * u + sine * v + one_minus_cos_over_f * acceleration
        velocity[2 * level + 1] = -sine * u + cosine * v + sin_over_f * acceleration

    layer_mass = fractions * (float(surface_pressure_pa) / float(gravity_m_s2))
    omega_lm, omega_mu = fields[9:11]
    courant_lm = np.max(np.abs(omega_lm) * float(dt_seconds) / (float(gravity_m_s2) * np.minimum(layer_mass[0], layer_mass[1])))
    courant_mu = np.max(np.abs(omega_mu) * float(dt_seconds) / (float(gravity_m_s2) * np.minimum(layer_mass[1], layer_mass[2])))
    vertical_courant = float(max(courant_lm, courant_mu))
    substeps = max(1, int(np.ceil(vertical_courant / 0.25)))
    transfer_lm = np.abs(omega_lm) * float(dt_seconds) / (float(gravity_m_s2) * substeps)
    transfer_mu = np.abs(omega_mu) * float(dt_seconds) / (float(gravity_m_s2) * substeps)

    def exchange(donor: int, receiver: int, transfer: np.ndarray, mask: np.ndarray) -> None:
        fraction = np.where(mask, transfer / layer_mass[donor], 0.0)
        if np.any(fraction > 1.0 + 1e-12):
            raise RuntimeError("vertical momentum substepping failed its donor-mass CFL bound")
        for component in (0, 1):
            donor_values = velocity[2 * donor + component]
            receiver_values = velocity[2 * receiver + component]
            content = donor_values * fraction
            velocity[2 * donor + component] = donor_values - content
            velocity[2 * receiver + component] = receiver_values + content * layer_mass[donor] / layer_mass[receiver]

    for _ in range(substeps):
        exchange(0, 1, transfer_lm, omega_lm < 0.0)
        exchange(1, 0, transfer_lm, omega_lm >= 0.0)
        exchange(1, 2, transfer_mu, omega_mu < 0.0)
        exchange(2, 1, transfer_mu, omega_mu >= 0.0)
    return ThreeLevelZonalMomentumStep(
        *(item.astype(np.float32) for item in velocity),
        *(item.astype(np.float32) for item in pressure_gradient), vertical_courant,
    )


def evolve_prognostic_pressure_coordinate_transport(
    lower_humidity: np.ndarray,
    midlevel_humidity: np.ndarray,
    upperlevel_humidity: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    lower_u_m_s: np.ndarray,
    lower_v_m_s: np.ndarray,
    midlevel_u_m_s: np.ndarray,
    midlevel_v_m_s: np.ndarray,
    upperlevel_u_m_s: np.ndarray,
    upperlevel_v_m_s: np.ndarray,
    *,
    lower_surface_vapour_source_kg_m2_s: np.ndarray,
    dt_seconds: float,
    radius_m: float,
    sidereal_day_hours: float,
    surface_pressure_pa: float,
    dx_m: np.ndarray | float,
    dy_m: float,
    cell_area_m2: np.ndarray | float,
    x_face_length_m: np.ndarray | float,
    y_face_length_m: np.ndarray | float,
    gravity_m_s2: float = 9.80665,
    gas_constant_dry_j_kg_k: float = 287.05,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
    layer_heights_m: tuple[float, float, float] = (0.0, 3500.0, 8000.0),
) -> PrognosticPressureCoordinateTransportStep:
    """Evolve horizontal water/MSE and pressure-level momentum explicitly.

    The old Phase-2 candidates inferred branch winds from instantaneous MSE
    and water constraints.  This replacement advances those conserved fields
    *from* the existing pressure-level state instead.  The finite-volume water
    updates choose their own donor-cell CFL subdivision; the corresponding MSE
    transport uses the identical faces and lower-layer latent source.  There
    is no diagnostic overturning solve, damping, cap, branch selector, or
    prescribed physical inner timestep.

    The function is intentionally limited to horizontal evolution.  Its next
    companion must provide a prognostic vertical mass-flux state, then couple
    phase conversion and the condensate reservoirs within the same transition.
    Keeping that omission explicit is preferable to reintroducing the rejected
    MSE/water residual-to-omega diagnosis through a hidden shortcut.
    """
    if dt_seconds <= 0.0 or radius_m <= 0.0 or sidereal_day_hours <= 0.0:
        raise ValueError("dt_seconds, radius_m, and sidereal_day_hours must be positive")
    if surface_pressure_pa <= 0.0 or gravity_m_s2 <= 0.0:
        raise ValueError("surface_pressure_pa and gravity_m_s2 must be positive")
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(float(np.sum(fractions)), 1.0):
        raise ValueError("layer_mass_fractions must be three positive values summing to one")
    raw = (
        lower_humidity, midlevel_humidity, upperlevel_humidity,
        lower_temperature_k, midlevel_temperature_k, upperlevel_temperature_k,
        lower_u_m_s, lower_v_m_s, midlevel_u_m_s, midlevel_v_m_s,
        upperlevel_u_m_s, upperlevel_v_m_s, lower_surface_vapour_source_kg_m2_s,
    )
    fields = tuple(np.asarray(value, dtype=np.float64) for value in raw)
    shape = fields[0].shape
    if len(shape) != 2 or shape[1] != 2 * shape[0] or any(value.shape != shape for value in fields):
        raise ValueError("prognostic pressure-coordinate fields must share a two-dimensional 2:1 grid")
    if any(not np.all(np.isfinite(value)) for value in fields):
        raise ValueError("prognostic pressure-coordinate fields must be finite")
    if any(np.any(value < 0.0) for value in fields[:3]) or np.any(fields[-1] < 0.0):
        raise ValueError("humidities and lower surface vapour source must be non-negative")

    from column_water import evolve_column_water
    from pressure_column import transport_closed_three_level_mse

    layer_mass = fractions * (float(surface_pressure_pa) / float(gravity_m_s2))
    humidity = fields[:3]
    winds = ((fields[6], fields[7]), (fields[8], fields[9]), (fields[10], fields[11]))
    source = fields[12]
    water_steps = []
    humidity_after: list[np.ndarray] = []
    for index in range(3):
        water_step = evolve_column_water(
            layer_mass[index] * humidity[index],
            source * 86400.0 if index == 0 else np.zeros(shape), np.zeros(shape),
            *winds[index], dx_m=dx_m, dy_m=dy_m, dt_days=float(dt_seconds) / 86400.0,
            cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
            y_face_length_m=y_face_length_m,
        )
        water_steps.append(water_step)
        humidity_after.append(np.asarray(water_step.water_mm, dtype=np.float64) / layer_mass[index])

    # The horizontal MSE step receives exactly the humidity transition just
    # evolved above, guaranteeing that its latent term and the vapour transport
    # use one set of finite-volume faces and one lower-boundary source.
    horizontal_mse = transport_closed_three_level_mse(
        humidity[0], humidity[1], humidity[2],
        humidity_after[0], humidity_after[1], humidity_after[2],
        fields[3], fields[4], fields[5],
        *winds[0], *winds[1], *winds[2],
        lower_vapour_source_kg_kg_day=source * 86400.0 / layer_mass[0],
        dt_days=float(dt_seconds) / 86400.0, dx_m=dx_m, dy_m=dy_m,
        cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
        y_face_length_m=y_face_length_m, layer_mass_fractions=layer_mass_fractions,
        surface_pressure_pa=surface_pressure_pa, layer_heights_m=layer_heights_m,
    )
    zeros = np.zeros(shape, dtype=np.float64)
    momentum = evolve_three_level_zonal_momentum(
        *winds[0], *winds[1], *winds[2],
        horizontal_mse.lower_temperature, horizontal_mse.midlevel_temperature,
        horizontal_mse.upperlevel_temperature, zeros, zeros,
        dt_seconds=dt_seconds, radius_m=radius_m, sidereal_day_hours=sidereal_day_hours,
        surface_pressure_pa=surface_pressure_pa, gravity_m_s2=gravity_m_s2,
        gas_constant_dry_j_kg_k=gas_constant_dry_j_kg_k,
        layer_mass_fractions=layer_mass_fractions,
    )
    return PrognosticPressureCoordinateTransportStep(
        *(value.astype(np.float32) for value in humidity_after),
        horizontal_mse.lower_temperature, horizontal_mse.midlevel_temperature,
        horizontal_mse.upperlevel_temperature,
        momentum.lower_u, momentum.lower_v, momentum.midlevel_u, momentum.midlevel_v,
        momentum.upperlevel_u, momentum.upperlevel_v,
        max(step.relative_residual for step in water_steps),
        horizontal_mse.relative_energy_residual,
        max(step.substeps for step in water_steps),
        max(step.maximum_outgoing_courant for step in water_steps),
    )


def evolve_prognostic_pressure_layer_mass(
    lower_pressure_depth_pa: np.ndarray,
    midlevel_pressure_depth_pa: np.ndarray,
    upperlevel_pressure_depth_pa: np.ndarray,
    lower_u_m_s: np.ndarray,
    lower_v_m_s: np.ndarray,
    midlevel_u_m_s: np.ndarray,
    midlevel_v_m_s: np.ndarray,
    upperlevel_u_m_s: np.ndarray,
    upperlevel_v_m_s: np.ndarray,
    lower_mid_interface_mass_flux_kg_m2_s: np.ndarray,
    mid_upper_interface_mass_flux_kg_m2_s: np.ndarray,
    *,
    lower_mid_interface_mass_flux_tendency_kg_m2_s2: np.ndarray | None = None,
    mid_upper_interface_mass_flux_tendency_kg_m2_s2: np.ndarray | None = None,
    dt_seconds: float,
    gravity_m_s2: float,
    dx_m: np.ndarray | float,
    dy_m: float,
    cell_area_m2: np.ndarray | float,
    x_face_length_m: np.ndarray | float,
    y_face_length_m: np.ndarray | float,
    max_vertical_courant: float = 0.25,
) -> PrognosticPressureLayerMassStep:
    """Advance pressure thickness under horizontal and stateful vertical flux.

    Pressure-layer mass is ``dp / g``.  Each layer is first transported on
    the resolved finite-volume horizontal faces.  The two interface mass fluxes
    are then advanced from their *own* supplied tendencies with a midpoint
    update and exchange mass between adjacent layers.  Thus continuity evolves
    pressure thickness without ever inverting an instantaneous humidity or MSE
    residual into omega.  A future vertical-momentum equation must supply the
    two tendencies; this primitive only defines their conservative state
    boundary and rejects an unstable transfer rather than clipping it.

    Positive interface flux is upward.  The adaptive vertical subdivision is
    derived solely from the combined donor-layer Courant fraction, including
    both fluxes when they drain the middle layer.
    """
    if dt_seconds <= 0.0 or gravity_m_s2 <= 0.0 or not 0.0 < max_vertical_courant <= 1.0:
        raise ValueError("dt_seconds/gravity_m_s2 must be positive and max_vertical_courant in (0, 1]")
    raw = (
        lower_pressure_depth_pa, midlevel_pressure_depth_pa, upperlevel_pressure_depth_pa,
        lower_u_m_s, lower_v_m_s, midlevel_u_m_s, midlevel_v_m_s,
        upperlevel_u_m_s, upperlevel_v_m_s,
        lower_mid_interface_mass_flux_kg_m2_s, mid_upper_interface_mass_flux_kg_m2_s,
    )
    fields = tuple(np.asarray(value, dtype=np.float64) for value in raw)
    shape = fields[0].shape
    if len(shape) != 2 or shape[1] != 2 * shape[0] or any(value.shape != shape for value in fields):
        raise ValueError("pressure-layer mass fields must share a two-dimensional 2:1 grid")
    if any(not np.all(np.isfinite(value)) for value in fields):
        raise ValueError("pressure-layer mass fields must be finite")
    if any(np.any(value <= 0.0) for value in fields[:3]):
        raise ValueError("pressure-layer depths must be positive")
    tendency_lm = np.zeros(shape, dtype=np.float64) if lower_mid_interface_mass_flux_tendency_kg_m2_s2 is None else np.asarray(lower_mid_interface_mass_flux_tendency_kg_m2_s2, dtype=np.float64)
    tendency_mu = np.zeros(shape, dtype=np.float64) if mid_upper_interface_mass_flux_tendency_kg_m2_s2 is None else np.asarray(mid_upper_interface_mass_flux_tendency_kg_m2_s2, dtype=np.float64)
    if tendency_lm.shape != shape or tendency_mu.shape != shape or not np.all(np.isfinite(tendency_lm)) or not np.all(np.isfinite(tendency_mu)):
        raise ValueError("interface mass-flux tendencies must be finite and match pressure layers")

    from column_water import evolve_column_water

    mass_steps = []
    mass: list[np.ndarray] = []
    for pressure, u, v in ((fields[0], fields[3], fields[4]), (fields[1], fields[5], fields[6]), (fields[2], fields[7], fields[8])):
        step = evolve_column_water(
            pressure / float(gravity_m_s2), np.zeros(shape), np.zeros(shape), u, v,
            dx_m=dx_m, dy_m=dy_m, dt_days=float(dt_seconds) / 86400.0,
            cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
            y_face_length_m=y_face_length_m,
        )
        mass_steps.append(step)
        mass.append(np.asarray(step.water_mm, dtype=np.float64))

    flux_lm_next = fields[9] + float(dt_seconds) * tendency_lm
    flux_mu_next = fields[10] + float(dt_seconds) * tendency_mu
    flux_lm = 0.5 * (fields[9] + flux_lm_next)
    flux_mu = 0.5 * (fields[10] + flux_mu_next)
    # The full-step combined donor fraction fixes an adaptive, unit-derived
    # vertical cadence.  It is a stability condition, not a physical cap.
    outgoing_lower = np.maximum(flux_lm, 0.0)
    outgoing_middle = np.maximum(-flux_lm, 0.0) + np.maximum(flux_mu, 0.0)
    outgoing_upper = np.maximum(-flux_mu, 0.0)
    # Layer masses change during exchange, especially when both interfaces
    # drain the middle layer.  Recompute the donor limit after every transfer;
    # choosing all substeps from the initial mass would violate the stated CFL
    # bound late in such a transition.
    remaining = float(dt_seconds)
    vertical_substeps = 0
    maximum_vertical_courant = 0.0
    while remaining > 0.0:
        donor_rate = np.maximum.reduce((
            outgoing_lower / mass[0], outgoing_middle / mass[1], outgoing_upper / mass[2],
        ))
        rate_max = float(np.max(donor_rate))
        dt_sub = remaining if rate_max == 0.0 else min(remaining, max_vertical_courant / rate_max)
        if not np.isfinite(dt_sub) or dt_sub <= 0.0:
            raise RuntimeError("pressure-layer mass CFL policy produced an invalid substep")
        donor_fraction = donor_rate * dt_sub
        maximum_vertical_courant = max(maximum_vertical_courant, float(np.max(donor_fraction)))
        if np.any(donor_fraction > 1.0 + 1e-12):
            raise RuntimeError("pressure-layer mass CFL subdivision failed its donor bound")
        transferred_lm = flux_lm * dt_sub
        transferred_mu = flux_mu * dt_sub
        mass[0] = mass[0] - transferred_lm
        mass[1] = mass[1] + transferred_lm - transferred_mu
        mass[2] = mass[2] + transferred_mu
        remaining -= dt_sub
        vertical_substeps += 1
    if any(np.any(value < -1e-10) for value in mass):
        raise RuntimeError("pressure-layer mass transition produced a negative layer")
    mass = [np.maximum(value, 0.0) for value in mass]
    area = np.broadcast_to(np.asarray(cell_area_m2, dtype=np.float64), shape)
    expected_total = sum(float(np.sum(np.asarray(step.water_mm, dtype=np.float64) * area)) for step in mass_steps)
    actual_total = sum(float(np.sum(value * area)) for value in mass)
    residual = actual_total - expected_total
    return PrognosticPressureLayerMassStep(
        *(value.astype(np.float32) * float(gravity_m_s2) for value in mass),
        flux_lm_next.astype(np.float32), flux_mu_next.astype(np.float32),
        residual / max(abs(expected_total), abs(actual_total), 1.0),
        max(step.substeps for step in mass_steps),
        max(step.maximum_outgoing_courant for step in mass_steps),
        vertical_substeps, maximum_vertical_courant,
    )


def diagnose_hydrostatic_sigma_continuity(
    lower_pressure_depth_pa: np.ndarray,
    midlevel_pressure_depth_pa: np.ndarray,
    upperlevel_pressure_depth_pa: np.ndarray,
    lower_u_m_s: np.ndarray,
    lower_v_m_s: np.ndarray,
    midlevel_u_m_s: np.ndarray,
    midlevel_v_m_s: np.ndarray,
    upperlevel_u_m_s: np.ndarray,
    upperlevel_v_m_s: np.ndarray,
    *,
    dt_seconds: float,
    gravity_m_s2: float,
    dx_m: np.ndarray | float,
    dy_m: float,
    cell_area_m2: np.ndarray | float,
    x_face_length_m: np.ndarray | float,
    y_face_length_m: np.ndarray | float,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
) -> HydrostaticSigmaContinuityStep:
    """Diagnose interface flux from horizontal mass continuity in sigma layers.

    A hydrostatic pressure-coordinate model does not need an arbitrary vertical
    acceleration law for omega.  In a three-layer sigma coordinate the layer
    masses are fixed fractions of the *prognostic total* column pressure.  The
    three independently resolved horizontal mass tendencies therefore give a
    unique pair of interface mass fluxes:

    ``F_lm = H_l - a_l H_total`` and ``F_mu = a_u H_total - H_u``.

    Here ``H`` is the finite-volume horizontal pressure-mass tendency and a
    positive ``F`` is upward.  The middle-layer equation then closes exactly.
    This derives vertical transport from pressure continuity alone; it makes no
    reference to water, MSE, latent heating, or a target circulation speed.
    """
    if dt_seconds <= 0.0 or gravity_m_s2 <= 0.0:
        raise ValueError("dt_seconds and gravity_m_s2 must be positive")
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(float(np.sum(fractions)), 1.0):
        raise ValueError("layer_mass_fractions must be three positive values summing to one")
    raw = (
        lower_pressure_depth_pa, midlevel_pressure_depth_pa, upperlevel_pressure_depth_pa,
        lower_u_m_s, lower_v_m_s, midlevel_u_m_s, midlevel_v_m_s,
        upperlevel_u_m_s, upperlevel_v_m_s,
    )
    fields = tuple(np.asarray(value, dtype=np.float64) for value in raw)
    shape = fields[0].shape
    if len(shape) != 2 or shape[1] != 2 * shape[0] or any(value.shape != shape for value in fields):
        raise ValueError("sigma-continuity fields must share a two-dimensional 2:1 grid")
    if any(not np.all(np.isfinite(value)) for value in fields) or any(np.any(value <= 0.0) for value in fields[:3]):
        raise ValueError("sigma-continuity pressures must be positive and fields finite")
    column_pressure = fields[0] + fields[1] + fields[2]
    for index, pressure in enumerate(fields[:3]):
        if not np.allclose(pressure, fractions[index] * column_pressure, rtol=2e-6, atol=2e-4):
            raise ValueError("sigma-continuity pressure layers must retain their declared mass fractions")

    from column_water import evolve_column_water

    tendency: list[np.ndarray] = []
    steps = []
    for pressure, u, v in ((fields[0], fields[3], fields[4]), (fields[1], fields[5], fields[6]), (fields[2], fields[7], fields[8])):
        mass_before = pressure / float(gravity_m_s2)
        step = evolve_column_water(
            mass_before, np.zeros(shape), np.zeros(shape), u, v,
            dx_m=dx_m, dy_m=dy_m, dt_days=float(dt_seconds) / 86400.0,
            cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
            y_face_length_m=y_face_length_m,
        )
        steps.append(step)
        tendency.append((np.asarray(step.water_mm, dtype=np.float64) - mass_before) / float(dt_seconds))
    total_tendency = tendency[0] + tendency[1] + tendency[2]
    sigma_tendency = [fractions[index] * total_tendency for index in range(3)]
    flux_lm = tendency[0] - sigma_tendency[0]
    flux_mu = sigma_tendency[2] - tendency[2]
    middle_residual = sigma_tendency[1] - (tendency[1] + flux_lm - flux_mu)
    scale = max(float(np.max(np.abs(total_tendency))), 1e-16)
    return HydrostaticSigmaContinuityStep(
        *(value.astype(np.float32) for value in sigma_tendency),
        flux_lm.astype(np.float32), flux_mu.astype(np.float32),
        (float(gravity_m_s2) * total_tendency).astype(np.float32),
        float(np.max(np.abs(middle_residual))) / scale,
        max(step.substeps for step in steps),
        max(step.maximum_outgoing_courant for step in steps),
    )


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _evolve_hydrostatic_sigma_mass_momentum_numba(
        state, area, x_length, y_length, source_water, dt_seconds,
        gravity, gas_constant, cp_dry, latent_heat, heights, fractions,
        dy_m, sidereal_day_hours, max_horizontal_courant,
        max_vertical_courant,
    ):
        """Compiled finite-volume core for the sigma mass/momentum carrier.

        ``state[layer, component]`` contains dry pressure mass, two momentum
        components, vapour, MSE, cloud condensate, and hydrometeors.  It is
        deliberately a direct donor-cell translation of the NumPy transition
        below: all contents use the same face velocity and the same vertical
        donor parcel at every CFL-limited substep.
        """
        nlayer, ncomponent, h, w = state.shape
        tendency = np.empty_like(state)
        trial = np.empty_like(state)
        exchanged = np.empty_like(state)
        row_temperature = np.empty((nlayer, h), dtype=np.float64)
        row_gradient = np.empty((nlayer, h), dtype=np.float64)
        interface_lm = np.zeros((h, w), dtype=np.float64)
        interface_mu = np.zeros((h, w), dtype=np.float64)
        remaining = dt_seconds
        substeps = 0
        horizontal_courant_max = 0.0
        vertical_courant_max = 0.0
        momentum_residual = 0.0
        day_seconds = sidereal_day_hours * 3600.0

        while remaining > 1.0e-9:
            horizontal_rate = 0.0
            maximum_speed = 0.0
            for layer in range(nlayer):
                for i in range(h):
                    for j in range(w):
                        mass = state[layer, 0, i, j]
                        u = state[layer, 1, i, j] / mass
                        v = state[layer, 2, i, j] / mass
                        speed = np.sqrt(u * u + v * v)
                        if speed > maximum_speed:
                            maximum_speed = speed
                        jp = 0 if j == w - 1 else j + 1
                        jm = w - 1 if j == 0 else j - 1
                        ue = 0.5 * (u + state[layer, 1, i, jp] / state[layer, 0, i, jp])
                        uw = 0.5 * (state[layer, 1, i, jm] / state[layer, 0, i, jm] + u)
                        north = 0.0
                        south = 0.0
                        if i > 0:
                            north = 0.5 * (state[layer, 2, i - 1, j] / state[layer, 0, i - 1, j] + v)
                        if i < h - 1:
                            south = 0.5 * (v + state[layer, 2, i + 1, j] / state[layer, 0, i + 1, j])
                        rate = (
                            max(ue, 0.0) * x_length[i, j]
                            + max(-uw, 0.0) * x_length[i, j]
                            + max(north, 0.0) * y_length[i, j]
                            + max(-south, 0.0) * y_length[i + 1, j]
                        ) / area[i, j]
                        if rate > horizontal_rate:
                            horizontal_rate = rate
            gravity_wave_speed = np.sqrt(gravity * max(heights[2] - heights[0], 1.0))
            if maximum_speed >= gravity_wave_speed:
                return state, interface_lm, interface_mu, -1, horizontal_courant_max, vertical_courant_max, momentum_residual
            dt_sub = remaining
            if horizontal_rate > 0.0:
                dt_sub = min(dt_sub, max_horizontal_courant / horizontal_rate)

            # Face flux divergence for every conserved component.  The face
            # orientation and donor selection match ``horizontal_operator``.
            for layer in range(nlayer):
                for component in range(ncomponent):
                    for i in range(h):
                        for j in range(w):
                            jp = 0 if j == w - 1 else j + 1
                            jm = w - 1 if j == 0 else j - 1
                            mass = state[layer, 0, i, j]
                            u = state[layer, 1, i, j] / mass
                            v = state[layer, 2, i, j] / mass
                            ue = 0.5 * (u + state[layer, 1, i, jp] / state[layer, 0, i, jp])
                            uw = 0.5 * (state[layer, 1, i, jm] / state[layer, 0, i, jm] + u)
                            east = ue * (state[layer, component, i, j] if ue >= 0.0 else state[layer, component, i, jp]) * x_length[i, j]
                            west = uw * (state[layer, component, i, jm] if uw >= 0.0 else state[layer, component, i, j]) * x_length[i, j]
                            north = 0.0
                            south = 0.0
                            if i > 0:
                                vn = 0.5 * (state[layer, 2, i - 1, j] / state[layer, 0, i - 1, j] + v)
                                north = vn * (state[layer, component, i, j] if vn >= 0.0 else state[layer, component, i - 1, j]) * y_length[i, j]
                            if i < h - 1:
                                vs = 0.5 * (v + state[layer, 2, i + 1, j] / state[layer, 0, i + 1, j])
                                south = vs * (state[layer, component, i + 1, j] if vs >= 0.0 else state[layer, component, i, j]) * y_length[i + 1, j]
                            tendency[layer, component, i, j] = (west - east + south - north) / area[i, j]

            # Find the common timestep that also satisfies simultaneous
            # lower/middle and middle/upper donor fractions.
            vertical_fraction = 0.0
            while True:
                vertical_fraction = 0.0
                for i in range(h):
                    for j in range(w):
                        m0 = state[0, 0, i, j] + dt_sub * tendency[0, 0, i, j]
                        m1 = state[1, 0, i, j] + dt_sub * tendency[1, 0, i, j]
                        m2 = state[2, 0, i, j] + dt_sub * tendency[2, 0, i, j]
                        if m0 <= 0.0 or m1 <= 0.0 or m2 <= 0.0:
                            return state, interface_lm, interface_mu, -2, horizontal_courant_max, vertical_courant_max, momentum_residual
                        total = m0 + m1 + m2
                        lm = m0 - fractions[0] * total
                        mu = fractions[2] * total - m2
                        f0 = max(lm, 0.0) / m0
                        f1 = (max(-lm, 0.0) + max(mu, 0.0)) / m1
                        f2 = max(-mu, 0.0) / m2
                        if f0 > vertical_fraction:
                            vertical_fraction = f0
                        if f1 > vertical_fraction:
                            vertical_fraction = f1
                        if f2 > vertical_fraction:
                            vertical_fraction = f2
                if vertical_fraction <= max_vertical_courant + 1.0e-14:
                    break
                dt_sub *= max_vertical_courant / vertical_fraction
                if dt_sub <= 0.0 or not np.isfinite(dt_sub):
                    return state, interface_lm, interface_mu, -3, horizontal_courant_max, vertical_courant_max, momentum_residual

            before_u = 0.0
            before_v = 0.0
            before_u_scale = 0.0
            before_v_scale = 0.0
            for layer in range(nlayer):
                for i in range(h):
                    for j in range(w):
                        before_u += state[layer, 1, i, j] * area[i, j]
                        before_v += state[layer, 2, i, j] * area[i, j]
                        before_u_scale += abs(state[layer, 1, i, j]) * area[i, j]
                        before_v_scale += abs(state[layer, 2, i, j]) * area[i, j]
                        for component in range(ncomponent):
                            trial[layer, component, i, j] = state[layer, component, i, j] + dt_sub * tendency[layer, component, i, j]
                        if layer == 0:
                            trial[layer, 3, i, j] += dt_sub * source_water[i, j]
                            trial[layer, 4, i, j] += dt_sub * latent_heat * source_water[i, j]
            for layer in range(nlayer):
                for component in range(ncomponent):
                    for i in range(h):
                        for j in range(w):
                            exchanged[layer, component, i, j] = trial[layer, component, i, j]
            for i in range(h):
                for j in range(w):
                    m0 = trial[0, 0, i, j]
                    m1 = trial[1, 0, i, j]
                    m2 = trial[2, 0, i, j]
                    total = m0 + m1 + m2
                    lm = m0 - fractions[0] * total
                    mu = fractions[2] * total - m2
                    interface_lm[i, j] += lm
                    interface_mu[i, j] += mu
                    exchanged[0, 0, i, j] = fractions[0] * total
                    exchanged[1, 0, i, j] = fractions[1] * total
                    exchanged[2, 0, i, j] = fractions[2] * total
                    for component in range(1, ncomponent):
                        if lm >= 0.0:
                            parcel = trial[0, component, i, j] / m0 * lm
                            exchanged[0, component, i, j] -= parcel
                            exchanged[1, component, i, j] += parcel
                        else:
                            parcel = trial[1, component, i, j] / m1 * (-lm)
                            exchanged[0, component, i, j] += parcel
                            exchanged[1, component, i, j] -= parcel
                        if mu >= 0.0:
                            parcel = trial[1, component, i, j] / m1 * mu
                            exchanged[1, component, i, j] -= parcel
                            exchanged[2, component, i, j] += parcel
                        else:
                            parcel = trial[2, component, i, j] / m2 * (-mu)
                            exchanged[1, component, i, j] += parcel
                            exchanged[2, component, i, j] -= parcel
            after_u = 0.0
            after_v = 0.0
            for layer in range(nlayer):
                for i in range(h):
                    for j in range(w):
                        after_u += exchanged[layer, 1, i, j] * area[i, j]
                        after_v += exchanged[layer, 2, i, j] * area[i, j]
            momentum_residual = max(momentum_residual, abs(after_u - before_u) / max(before_u_scale, 1.0), abs(after_v - before_v) / max(before_v_scale, 1.0))

            # Hydrostatic PGF and analytic Coriolis rotation are evaluated on
            # the same post-exchange pressure/momentum state as the NumPy
            # carrier, before it becomes the next donor state.
            for layer in range(nlayer):
                for i in range(h):
                    row_sum = 0.0
                    for j in range(w):
                        mass = exchanged[layer, 0, i, j]
                        q = exchanged[layer, 3, i, j] / mass
                        temperature = (exchanged[layer, 4, i, j] / mass - gravity * heights[layer] - latent_heat * q) / cp_dry
                        total_pressure = gravity * (exchanged[0, 0, i, j] + exchanged[1, 0, i, j] + exchanged[2, 0, i, j])
                        if layer == 0:
                            center_pressure = total_pressure - 0.5 * gravity * exchanged[0, 0, i, j]
                        elif layer == 1:
                            center_pressure = gravity * exchanged[2, 0, i, j] + 0.5 * gravity * exchanged[1, 0, i, j]
                        else:
                            center_pressure = 0.5 * gravity * exchanged[2, 0, i, j]
                        row_sum += gas_constant * temperature * np.log(total_pressure / center_pressure)
                    row_temperature[layer, i] = row_sum / w
                for i in range(h):
                    if i == 0:
                        row_gradient[layer, i] = (row_temperature[layer, 1] - row_temperature[layer, 0]) / dy_m
                    elif i == h - 1:
                        row_gradient[layer, i] = (row_temperature[layer, h - 1] - row_temperature[layer, h - 2]) / dy_m
                    else:
                        row_gradient[layer, i] = (row_temperature[layer, i + 1] - row_temperature[layer, i - 1]) / (2.0 * dy_m)
            for layer in range(nlayer):
                for i in range(h):
                    latitude = 0.5 * np.pi - (i + 0.5) * np.pi / h
                    coriolis = 4.0 * np.pi / day_seconds * np.sin(latitude)
                    theta = coriolis * dt_sub
                    cosine = np.cos(theta)
                    sine = np.sin(theta)
                    if abs(theta) < 1.0e-5:
                        sin_over_f = dt_sub * (1.0 - theta * theta / 6.0)
                        one_minus_cos_over_f = 0.5 * dt_sub * theta
                    else:
                        sin_over_f = sine / coriolis
                        one_minus_cos_over_f = (1.0 - cosine) / coriolis
                    for j in range(w):
                        mass = exchanged[layer, 0, i, j]
                        u = exchanged[layer, 1, i, j] / mass
                        v = exchanged[layer, 2, i, j] / mass
                        next_u = cosine * u + sine * v + one_minus_cos_over_f * row_gradient[layer, i]
                        next_v = -sine * u + cosine * v + sin_over_f * row_gradient[layer, i]
                        exchanged[layer, 1, i, j] = mass * next_u
                        exchanged[layer, 2, i, j] = mass * next_v
            for layer in range(nlayer):
                for component in range(ncomponent):
                    for i in range(h):
                        for j in range(w):
                            state[layer, component, i, j] = exchanged[layer, component, i, j]
            horizontal_courant_max = max(horizontal_courant_max, horizontal_rate * dt_sub)
            vertical_courant_max = max(vertical_courant_max, vertical_fraction)
            remaining -= dt_sub
            substeps += 1
        return state, interface_lm / dt_seconds, interface_mu / dt_seconds, substeps, horizontal_courant_max, vertical_courant_max, momentum_residual


def evolve_hydrostatic_sigma_mass_momentum(
    lower_pressure_depth_pa: np.ndarray,
    midlevel_pressure_depth_pa: np.ndarray,
    upperlevel_pressure_depth_pa: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    lower_u_m_s: np.ndarray,
    lower_v_m_s: np.ndarray,
    midlevel_u_m_s: np.ndarray,
    midlevel_v_m_s: np.ndarray,
    upperlevel_u_m_s: np.ndarray,
    upperlevel_v_m_s: np.ndarray,
    *,
    dt_seconds: float,
    radius_m: float,
    sidereal_day_hours: float,
    gravity_m_s2: float,
    gas_constant_dry_j_kg_k: float = 287.05,
    dx_m: np.ndarray | float = 1.0,
    dy_m: float = 1.0,
    cell_area_m2: np.ndarray | float = 1.0,
    x_face_length_m: np.ndarray | float = 1.0,
    y_face_length_m: np.ndarray | float = 1.0,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
    max_horizontal_courant: float = 0.9,
    max_vertical_courant: float = 0.25,
    lower_humidity: np.ndarray | None = None,
    midlevel_humidity: np.ndarray | None = None,
    upperlevel_humidity: np.ndarray | None = None,
    lower_surface_vapour_source_kg_m2_s: np.ndarray | None = None,
    lower_cloud_condensate_kg_m2: np.ndarray | None = None,
    midlevel_cloud_condensate_kg_m2: np.ndarray | None = None,
    upperlevel_cloud_condensate_kg_m2: np.ndarray | None = None,
    lower_hydrometeors_kg_m2: np.ndarray | None = None,
    midlevel_hydrometeors_kg_m2: np.ndarray | None = None,
    upperlevel_hydrometeors_kg_m2: np.ndarray | None = None,
    cp_dry_j_kg_k: float = 1004.0,
    latent_heat_j_kg: float = 2.5e6,
    layer_heights_m: tuple[float, float, float] = (0.0, 3500.0, 8000.0),
) -> HydrostaticSigmaMassMomentumStep:
    """Jointly evolve horizontal layer mass and momentum in a sigma column.

    The former hydrostatic-sigma candidate transported pressure mass with a
    wind fixed for an entire host step and only then advanced momentum.  That
    split allowed the inherited carrier to evacuate a layer before its own
    pressure-gradient/Coriolis response could act.  This primitive moves layer
    mass and both momentum components through the *same* donor-cell faces at
    every adaptive substep.  It then performs the continuity-required vertical
    redistribution using simultaneous donor parcels, restoring the declared
    sigma fractions exactly before applying the resolved hydrostatic
    pressure-gradient/Coriolis force.

    When humidity is supplied for all three layers, vapour and moist-static
    energy become carried inventories; the optional cloud/hydrometeor fields
    join those same parcels. Temperature then follows the transported MSE,
    rather than remaining a split force boundary. There is no wind cap,
    damping, mass floor, fallback path, or prescribed timestep. Both
    horizontal and vertical donor fractions determine the adaptive substep
    size, and an invalid state raises rather than clipping.
    """
    if dt_seconds <= 0.0 or radius_m <= 0.0 or sidereal_day_hours <= 0.0 or gravity_m_s2 <= 0.0 or gas_constant_dry_j_kg_k <= 0.0:
        raise ValueError("time, geometry, and physical constants must be positive")
    if dy_m <= 0.0 or not 0.0 < max_horizontal_courant <= 1.0 or not 0.0 < max_vertical_courant <= 1.0:
        raise ValueError("Courant limits must be in (0, 1] and dy_m must be positive")
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(float(np.sum(fractions)), 1.0):
        raise ValueError("layer_mass_fractions must be three positive values summing to one")
    raw = (
        lower_pressure_depth_pa, midlevel_pressure_depth_pa, upperlevel_pressure_depth_pa,
        lower_temperature_k, midlevel_temperature_k, upperlevel_temperature_k,
        lower_u_m_s, lower_v_m_s, midlevel_u_m_s, midlevel_v_m_s,
        upperlevel_u_m_s, upperlevel_v_m_s,
    )
    fields = tuple(np.asarray(value, dtype=np.float64) for value in raw)
    shape = fields[0].shape
    if len(shape) != 2 or shape[1] != 2 * shape[0] or any(value.shape != shape for value in fields):
        raise ValueError("hydrostatic-sigma mass/momentum fields must share a two-dimensional 2:1 grid")
    if any(not np.all(np.isfinite(value)) for value in fields) or any(np.any(value <= 0.0) for value in fields[:3]):
        raise ValueError("pressure depths must be positive and all fields finite")
    area = np.broadcast_to(np.asarray(cell_area_m2, dtype=np.float64), shape)
    x_length = np.broadcast_to(np.asarray(x_face_length_m, dtype=np.float64), shape)
    y_length = np.broadcast_to(np.asarray(y_face_length_m, dtype=np.float64), (shape[0] + 1, shape[1]))
    if np.any(area <= 0.0) or np.any(x_length <= 0.0) or np.any(y_length < 0.0):
        raise ValueError("cell area and face lengths must be non-negative with positive cell/x-face area")
    pressure_initial = fields[:3]
    pressure_total = sum(pressure_initial)
    if any(not np.allclose(pressure_initial[index], fractions[index] * pressure_total, rtol=2e-6, atol=2e-4) for index in range(3)):
        raise ValueError("hydrostatic-sigma pressure layers must retain their declared mass fractions")
    humidity_inputs = (lower_humidity, midlevel_humidity, upperlevel_humidity)
    tracer_active = all(value is not None for value in humidity_inputs)
    if tracer_active != (lower_surface_vapour_source_kg_m2_s is not None):
        raise ValueError("humidity and lower_surface_vapour_source_kg_m2_s must be supplied together")
    if not tracer_active and any(value is not None for value in humidity_inputs):
        raise ValueError("supply humidity for all three layers or for none")
    reservoir_inputs = (
        lower_cloud_condensate_kg_m2, midlevel_cloud_condensate_kg_m2, upperlevel_cloud_condensate_kg_m2,
        lower_hydrometeors_kg_m2, midlevel_hydrometeors_kg_m2, upperlevel_hydrometeors_kg_m2,
    )
    reservoirs_active = all(value is not None for value in reservoir_inputs)
    if reservoirs_active and not tracer_active:
        raise ValueError("cloud/hydrometeor transport requires the carried humidity/MSE state")
    if not reservoirs_active and any(value is not None for value in reservoir_inputs):
        raise ValueError("supply all six cloud/hydrometeor reservoirs or none")
    if tracer_active and (cp_dry_j_kg_k <= 0.0 or latent_heat_j_kg <= 0.0):
        raise ValueError("thermodynamic constants must be positive")
    heights = np.asarray(layer_heights_m, dtype=np.float64)
    if heights.shape != (3,) or not np.all(np.isfinite(heights)):
        raise ValueError("layer_heights_m must be three finite values")
    hydrostatic_gravity_wave_speed = np.sqrt(
        float(gravity_m_s2) * max(float(np.max(heights) - np.min(heights)), 1.0)
    )

    # The vectorized finite-volume path below is faster for the large grids
    # used by the runtime gate.  Retain the compiled reference for explicitly
    # requested low-Courant kernel checks, where its allocation-free state
    # layout is useful without penalising the production transport loop.
    if _NUMBA_AVAILABLE and tracer_active and max_horizontal_courant <= 0.25:
        compiled = np.empty((3, 7, *shape), dtype=np.float64)
        for index in range(3):
            compiled[index, 0] = pressure_initial[index] / float(gravity_m_s2)
            compiled[index, 1] = compiled[index, 0] * fields[6 + 2 * index]
            compiled[index, 2] = compiled[index, 0] * fields[7 + 2 * index]
            compiled[index, 3] = compiled[index, 0] * np.asarray(humidity_inputs[index], dtype=np.float64)
            compiled[index, 4] = compiled[index, 0] * (
                float(cp_dry_j_kg_k) * fields[3 + index]
                + float(gravity_m_s2) * heights[index]
                + float(latent_heat_j_kg) * np.asarray(humidity_inputs[index], dtype=np.float64)
            )
        if reservoirs_active:
            for index, value in enumerate(reservoir_inputs[:3]):
                compiled[index, 5] = np.asarray(value, dtype=np.float64)
            for index, value in enumerate(reservoir_inputs[3:]):
                compiled[index, 6] = np.asarray(value, dtype=np.float64)
        else:
            compiled[:, 5:] = 0.0
        source = np.asarray(lower_surface_vapour_source_kg_m2_s, dtype=np.float64)
        initial_mass = float(np.sum(np.sum(compiled[:, 0], axis=0) * area))
        initial_water = float(np.sum(np.sum(compiled[:, 3], axis=0) * area))
        initial_energy = float(np.sum(np.sum(compiled[:, 4], axis=0) * area))
        result, interface_lm, interface_mu, substeps, horizontal_courant, vertical_courant, momentum_residual = (
            _evolve_hydrostatic_sigma_mass_momentum_numba(
                compiled, area, x_length, y_length, source, float(dt_seconds),
                float(gravity_m_s2), float(gas_constant_dry_j_kg_k),
                float(cp_dry_j_kg_k), float(latent_heat_j_kg), heights,
                fractions, float(dy_m), float(sidereal_day_hours),
                float(max_horizontal_courant), float(max_vertical_courant),
            )
        )
        if substeps == -1:
            raise RuntimeError(
                "coupled sigma carrier exceeded its hydrostatic gravity-wave speed; "
                "the pressure-coordinate approximation is no longer admissible"
            )
        if substeps == -2:
            raise RuntimeError("coupled horizontal transport exhausted a sigma-layer mass")
        if substeps == -3:
            raise RuntimeError("coupled sigma CFL policy produced an invalid substep")
        final_mass = float(np.sum(np.sum(result[:, 0], axis=0) * area))
        actual_water = float(np.sum(np.sum(result[:, 3], axis=0) * area))
        actual_energy = float(np.sum(np.sum(result[:, 4], axis=0) * area))
        expected_water = initial_water + float(dt_seconds) * float(np.sum(source * area))
        expected_energy = initial_energy + float(dt_seconds) * float(latent_heat_j_kg) * float(np.sum(source * area))
        water_residual = (actual_water - expected_water) / max(abs(actual_water), abs(expected_water), 1.0)
        mse_residual = (actual_energy - expected_energy) / max(abs(actual_energy), abs(expected_energy), 1.0)
        humidity_next = [result[index, 3] / result[index, 0] for index in range(3)]
        temperature_next = [
            (result[index, 4] / result[index, 0] - float(gravity_m_s2) * heights[index]
             - float(latent_heat_j_kg) * humidity_next[index]) / float(cp_dry_j_kg_k)
            for index in range(3)
        ]
        return HydrostaticSigmaMassMomentumStep(
            *(result[index, 0].astype(np.float32) * float(gravity_m_s2) for index in range(3)),
            *(result[index, component].astype(np.float32) / result[index, 0].astype(np.float32) for index in range(3) for component in (1, 2)),
            interface_lm.astype(np.float32), interface_mu.astype(np.float32),
            (final_mass - initial_mass) / max(abs(initial_mass), abs(final_mass), 1.0),
            float(momentum_residual), int(substeps), float(horizontal_courant), float(vertical_courant),
            *(value.astype(np.float32) for value in humidity_next),
            *(value.astype(np.float32) for value in temperature_next),
            *(result[index, 5].astype(np.float32) for index in range(3)) if reservoirs_active else (None, None, None),
            *(result[index, 6].astype(np.float32) for index in range(3)) if reservoirs_active else (None, None, None),
            water_relative_residual=float(water_residual),
            moist_static_energy_relative_residual=float(mse_residual),
            horizontal_mse_convergence_w_m2=None,
        )

    mass = [pressure / float(gravity_m_s2) for pressure in pressure_initial]
    momentum_u = [mass[index] * fields[6 + 2 * index] for index in range(3)]
    momentum_v = [mass[index] * fields[7 + 2 * index] for index in range(3)]
    temperatures = fields[3:6]
    inventories: dict[str, list[np.ndarray]] = {}
    sources: dict[str, list[np.ndarray]] = {}
    inventory_initial: dict[str, float] = {}
    if tracer_active:
        humidity = [np.asarray(value, dtype=np.float64) for value in humidity_inputs]
        source = np.asarray(lower_surface_vapour_source_kg_m2_s, dtype=np.float64)
        if any(value.shape != shape or not np.all(np.isfinite(value)) or np.any(value < 0.0) for value in humidity + [source]):
            raise ValueError("humidity and surface vapour source must be finite, non-negative, and match the pressure state")
        inventories["water"] = [mass[index] * humidity[index] for index in range(3)]
        inventories["energy"] = [mass[index] * (
            float(cp_dry_j_kg_k) * temperatures[index]
            + float(gravity_m_s2) * heights[index]
            + float(latent_heat_j_kg) * humidity[index]
        ) for index in range(3)]
        zeros = np.zeros(shape, dtype=np.float64)
        sources["water"] = [source, zeros, zeros]
        sources["energy"] = [float(latent_heat_j_kg) * source, zeros, zeros]
        if reservoirs_active:
            clouds = [np.asarray(value, dtype=np.float64) for value in reservoir_inputs[:3]]
            hydrometeors = [np.asarray(value, dtype=np.float64) for value in reservoir_inputs[3:]]
            if any(value.shape != shape or not np.all(np.isfinite(value)) or np.any(value < 0.0) for value in clouds + hydrometeors):
                raise ValueError("cloud/hydrometeor reservoirs must be finite, non-negative, and match the pressure state")
            inventories["cloud"] = clouds
            inventories["hydrometeors"] = hydrometeors
            sources["cloud"] = [zeros, zeros, zeros]
            sources["hydrometeors"] = [zeros, zeros, zeros]
        inventory_initial = {
            name: sum(float(np.sum(value * area)) for value in values)
            for name, values in inventories.items()
        }
    initial_mass = sum(float(np.sum(value * area)) for value in mass)
    interface_lm_amount = np.zeros(shape, dtype=np.float64)
    interface_mu_amount = np.zeros(shape, dtype=np.float64)
    horizontal_mse_convergence_j_m2 = np.zeros(shape, dtype=np.float64)
    residual_momentum = 0.0
    remaining = float(dt_seconds)
    substeps = 0
    horizontal_courant_max = 0.0
    vertical_courant_max = 0.0
    h = shape[0]
    latitude = 0.5 * np.pi - (np.arange(h) + 0.5) * np.pi / h
    coriolis = 4.0 * np.pi / (float(sidereal_day_hours) * 3600.0) * np.sin(latitude)

    def horizontal_operator(u: np.ndarray, v: np.ndarray) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
        u_east = 0.5 * (u + np.roll(u, -1, axis=1))
        v_north = np.zeros((h + 1, shape[1]), dtype=np.float64)
        v_north[1:-1] = 0.5 * (v[:-1] + v[1:])
        outbound = (
            np.maximum(u_east, 0.0) * x_length
            + np.maximum(-np.roll(u_east, 1, axis=1), 0.0) * x_length
            + np.maximum(v_north[:-1], 0.0) * y_length[:-1]
            + np.maximum(-v_north[1:], 0.0) * y_length[1:]
        ) / area

        def tendency(field: np.ndarray) -> np.ndarray:
            # `field` may carry leading inventory axes.  Applying every
            # conserved quantity through one shared face construction keeps
            # the donor map identical while avoiding one geometry rebuild per
            # mass/momentum/tracer inventory.
            east = np.where(
                u_east >= 0.0, u_east * field, u_east * np.roll(field, -1, axis=-1)
            ) * x_length
            north = np.zeros(field.shape[:-2] + (h + 1, shape[1]), dtype=np.float64)
            north[..., 1:-1, :] = np.where(
                v_north[1:-1] >= 0.0,
                v_north[1:-1] * field[..., 1:, :],
                v_north[1:-1] * field[..., :-1, :],
            ) * y_length[1:-1]
            return (np.roll(east, 1, axis=-1) - east + north[..., 1:, :] - north[..., :-1, :]) / area

        return tendency, outbound

    while remaining > 0.0:
        winds_u = [momentum_u[index] / mass[index] for index in range(3)]
        winds_v = [momentum_v[index] / mass[index] for index in range(3)]
        maximum_speed = max(float(np.max(np.hypot(winds_u[index], winds_v[index]))) for index in range(3))
        if maximum_speed >= hydrostatic_gravity_wave_speed:
            raise RuntimeError(
                "coupled sigma carrier exceeded its hydrostatic gravity-wave speed; "
                f"maximum={maximum_speed:.3f} m/s, limit={hydrostatic_gravity_wave_speed:.3f} m/s; "
                "the pressure-coordinate approximation is no longer admissible"
            )
        # Momentum needs its own stability bound: a quiet column can have no
        # advective CFL signal at all while a hydrostatic pressure gradient
        # would accelerate it across the gravity-wave scale in one coarse host
        # step.  This is derived from the resolved force and state, not a
        # prescribed physical inner timestep or a wind limiter.
        temperatures_for_cfl = temperatures
        if tracer_active:
            humidity_for_cfl = [inventories["water"][index] / mass[index] for index in range(3)]
            temperatures_for_cfl = [
                (inventories["energy"][index] / mass[index]
                 - float(gravity_m_s2) * heights[index]
                 - float(latent_heat_j_kg) * humidity_for_cfl[index]) / float(cp_dry_j_kg_k)
                for index in range(3)
            ]
        total_pressure_for_cfl = sum(value * float(gravity_m_s2) for value in mass)
        center_pressure_for_cfl = (
            total_pressure_for_cfl - 0.5 * mass[0] * float(gravity_m_s2),
            mass[2] * float(gravity_m_s2) + 0.5 * mass[1] * float(gravity_m_s2),
            0.5 * mass[2] * float(gravity_m_s2),
        )
        maximum_pressure_gradient = 0.0
        for temperature, center_pressure in zip(temperatures_for_cfl, center_pressure_for_cfl, strict=True):
            geopotential = float(gas_constant_dry_j_kg_k) * np.mean(
                temperature * np.log(total_pressure_for_cfl / center_pressure), axis=1
            )
            maximum_pressure_gradient = max(
                maximum_pressure_gradient,
                float(np.max(np.abs(np.gradient(geopotential, float(dy_m), edge_order=1)))),
            )
        momentum_dt_limit = np.inf if maximum_pressure_gradient == 0.0 else (
            float(_MOMENTUM_WAVE_FRACTION)
            * (hydrostatic_gravity_wave_speed - maximum_speed)
            / maximum_pressure_gradient
        )
        operators = [horizontal_operator(winds_u[index], winds_v[index]) for index in range(3)]
        inventory_names = tuple(inventories)
        stacked_state = [
            np.stack(
                (mass[index], momentum_u[index], momentum_v[index], *(inventories[name][index] for name in inventory_names))
            )
            for index in range(3)
        ]
        stacked_tendency = [operators[index][0](stacked_state[index]) for index in range(3)]
        mass_tendency = [stacked_tendency[index][0] for index in range(3)]
        outbound = [operators[index][1] for index in range(3)]
        horizontal_rate = max(float(np.max(value)) for value in outbound)
        dt_sub = min(
            remaining,
            momentum_dt_limit,
            np.inf if horizontal_rate == 0.0 else max_horizontal_courant / horizontal_rate,
        )
        # The sigma-restoring exchange is computed from the same trial
        # horizontal mass update.  If it is the tighter donor constraint,
        # shorten the same substep before any state is committed.
        while True:
            trial_mass = [mass[index] + dt_sub * mass_tendency[index] for index in range(3)]
            if any(np.any(value <= 0.0) for value in trial_mass):
                raise RuntimeError("coupled horizontal transport exhausted a sigma-layer mass")
            total = sum(trial_mass)
            target = [fractions[index] * total for index in range(3)]
            amount_lm = trial_mass[0] - target[0]
            amount_mu = target[2] - trial_mass[2]
            outgoing = (
                np.maximum(amount_lm, 0.0),
                np.maximum(-amount_lm, 0.0) + np.maximum(amount_mu, 0.0),
                np.maximum(-amount_mu, 0.0),
            )
            vertical_fraction = max(float(np.max(outgoing[index] / trial_mass[index])) for index in range(3))
            if vertical_fraction <= max_vertical_courant + 1e-14:
                break
            dt_sub *= max_vertical_courant / vertical_fraction
            if not np.isfinite(dt_sub) or dt_sub <= 0.0:
                raise RuntimeError("coupled sigma CFL policy produced an invalid substep")

        if tracer_active:
            energy_component = 3 + inventory_names.index("energy")
            for index in range(3):
                horizontal_mse_convergence_j_m2 += dt_sub * stacked_tendency[index][energy_component]

        momentum_before_transport = [sum(float(np.sum(value * area)) for value in component) for component in (momentum_u, momentum_v)]
        momentum_scale = [sum(float(np.sum(np.abs(value) * area)) for value in component) for component in (momentum_u, momentum_v)]
        trial_u = [momentum_u[index] + dt_sub * stacked_tendency[index][1] for index in range(3)]
        trial_v = [momentum_v[index] + dt_sub * stacked_tendency[index][2] for index in range(3)]
        trial_inventories = {
            name: [
                values[index]
                + dt_sub * (
                    # The face velocities and donor direction are shared with
                    # pressure mass and momentum in this exact substep.
                    stacked_tendency[index][3 + inventory_names.index(name)]
                    + sources[name][index]
                )
                for index in range(3)
            ]
            for name, values in inventories.items()
        }
        if any(np.any(value < -1e-10) for values in trial_inventories.values() for value in values):
            raise RuntimeError("coupled sigma transport produced a negative carried inventory")
        for values in trial_inventories.values():
            for index in range(3):
                values[index] = np.maximum(values[index], 0.0)
        donor_u = [trial_u[index] / trial_mass[index] for index in range(3)]
        donor_v = [trial_v[index] / trial_mass[index] for index in range(3)]
        for amount, lower, upper in ((amount_lm, 0, 1), (amount_mu, 1, 2)):
            upward = amount >= 0.0
            # There are only two interfaces.  Construct their parcel contents
            # from the common pre-exchange state, so simultaneous middle-layer
            # outflow cannot borrow mass from the other exchange in this step.
            parcel_u = np.where(upward, donor_u[lower], donor_u[upper]) * np.abs(amount)
            parcel_v = np.where(upward, donor_v[lower], donor_v[upper]) * np.abs(amount)
            trial_u[lower] = np.where(upward, trial_u[lower] - parcel_u, trial_u[lower] + parcel_u)
            trial_u[upper] = np.where(upward, trial_u[upper] + parcel_u, trial_u[upper] - parcel_u)
            trial_v[lower] = np.where(upward, trial_v[lower] - parcel_v, trial_v[lower] + parcel_v)
            trial_v[upper] = np.where(upward, trial_v[upper] + parcel_v, trial_v[upper] - parcel_v)
            for values in trial_inventories.values():
                donor_content = np.where(upward, values[lower] / trial_mass[lower], values[upper] / trial_mass[upper])
                parcel_content = donor_content * np.abs(amount)
                values[lower] = np.where(upward, values[lower] - parcel_content, values[lower] + parcel_content)
                values[upper] = np.where(upward, values[upper] + parcel_content, values[upper] - parcel_content)
        transport_momentum_after = [sum(float(np.sum(value * area)) for value in component) for component in (trial_u, trial_v)]
        residual_momentum = max(
            residual_momentum,
            *(abs(transport_momentum_after[index] - momentum_before_transport[index]) / max(momentum_scale[index], 1.0) for index in (0, 1)),
        )
        mass = target
        inventories = trial_inventories
        interface_lm_amount += amount_lm
        interface_mu_amount += amount_mu

        pressures = [mass[index] * float(gravity_m_s2) for index in range(3)]
        total_pressure = sum(pressures)
        centers = (total_pressure - 0.5 * pressures[0], pressures[2] + 0.5 * pressures[1], 0.5 * pressures[2])
        gradients = []
        temperatures_for_force = temperatures
        if tracer_active:
            humidity_for_force = [inventories["water"][index] / mass[index] for index in range(3)]
            temperatures_for_force = [
                (inventories["energy"][index] / mass[index]
                 - float(gravity_m_s2) * heights[index]
                 - float(latent_heat_j_kg) * humidity_for_force[index]) / float(cp_dry_j_kg_k)
                for index in range(3)
            ]
        for temperature, center_pressure in zip(temperatures_for_force, centers, strict=True):
            geopotential = float(gas_constant_dry_j_kg_k) * np.mean(temperature * np.log(total_pressure / center_pressure), axis=1)
            gradients.append(np.broadcast_to(np.gradient(geopotential, float(dy_m), edge_order=1)[:, None], shape))
        theta = coriolis[:, None] * dt_sub
        cosine, sine = np.cos(theta), np.sin(theta)
        small = np.abs(theta) < 1e-5
        sin_over_f = np.where(small, dt_sub * (1.0 - theta**2 / 6.0), sine / coriolis[:, None])
        one_minus_cos_over_f = np.where(small, 0.5 * dt_sub * theta, (1.0 - cosine) / coriolis[:, None])
        for index in range(3):
            u = trial_u[index] / mass[index]
            v = trial_v[index] / mass[index]
            next_u = cosine * u + sine * v + one_minus_cos_over_f * gradients[index]
            next_v = -sine * u + cosine * v + sin_over_f * gradients[index]
            momentum_u[index] = mass[index] * next_u
            momentum_v[index] = mass[index] * next_v
        horizontal_courant_max = max(horizontal_courant_max, horizontal_rate * dt_sub)
        vertical_courant_max = max(vertical_courant_max, vertical_fraction)
        remaining -= dt_sub
        substeps += 1

    final_mass = sum(float(np.sum(value * area)) for value in mass)
    humidity_next = temperature_next = cloud_next = hydrometeor_next = None
    water_relative_residual = mse_relative_residual = None
    if tracer_active:
        humidity_next = [inventories["water"][index] / mass[index] for index in range(3)]
        temperature_next = [
            (inventories["energy"][index] / mass[index]
             - float(gravity_m_s2) * heights[index]
             - float(latent_heat_j_kg) * humidity_next[index]) / float(cp_dry_j_kg_k)
            for index in range(3)
        ]
        expected_water = inventory_initial["water"] + float(dt_seconds) * float(np.sum(sources["water"][0] * area))
        expected_energy = inventory_initial["energy"] + float(dt_seconds) * float(np.sum(sources["energy"][0] * area))
        actual_water = sum(float(np.sum(value * area)) for value in inventories["water"])
        actual_energy = sum(float(np.sum(value * area)) for value in inventories["energy"])
        water_relative_residual = (actual_water - expected_water) / max(abs(actual_water), abs(expected_water), 1.0)
        mse_relative_residual = (actual_energy - expected_energy) / max(abs(actual_energy), abs(expected_energy), 1.0)
        if reservoirs_active:
            cloud_next = inventories["cloud"]
            hydrometeor_next = inventories["hydrometeors"]
    return HydrostaticSigmaMassMomentumStep(
        *(value.astype(np.float32) * float(gravity_m_s2) for value in mass),
        (momentum_u[0] / mass[0]).astype(np.float32), (momentum_v[0] / mass[0]).astype(np.float32),
        (momentum_u[1] / mass[1]).astype(np.float32), (momentum_v[1] / mass[1]).astype(np.float32),
        (momentum_u[2] / mass[2]).astype(np.float32), (momentum_v[2] / mass[2]).astype(np.float32),
        (interface_lm_amount / float(dt_seconds)).astype(np.float32),
        (interface_mu_amount / float(dt_seconds)).astype(np.float32),
        (final_mass - initial_mass) / max(abs(initial_mass), abs(final_mass), 1.0),
        residual_momentum,
        substeps, horizontal_courant_max, vertical_courant_max,
        lower_humidity=None if humidity_next is None else humidity_next[0].astype(np.float32),
        midlevel_humidity=None if humidity_next is None else humidity_next[1].astype(np.float32),
        upperlevel_humidity=None if humidity_next is None else humidity_next[2].astype(np.float32),
        lower_temperature=None if temperature_next is None else temperature_next[0].astype(np.float32),
        midlevel_temperature=None if temperature_next is None else temperature_next[1].astype(np.float32),
        upperlevel_temperature=None if temperature_next is None else temperature_next[2].astype(np.float32),
        lower_cloud_condensate_kg_m2=None if cloud_next is None else cloud_next[0].astype(np.float32),
        midlevel_cloud_condensate_kg_m2=None if cloud_next is None else cloud_next[1].astype(np.float32),
        upperlevel_cloud_condensate_kg_m2=None if cloud_next is None else cloud_next[2].astype(np.float32),
        lower_hydrometeors_kg_m2=None if hydrometeor_next is None else hydrometeor_next[0].astype(np.float32),
        midlevel_hydrometeors_kg_m2=None if hydrometeor_next is None else hydrometeor_next[1].astype(np.float32),
        upperlevel_hydrometeors_kg_m2=None if hydrometeor_next is None else hydrometeor_next[2].astype(np.float32),
        water_relative_residual=water_relative_residual,
        moist_static_energy_relative_residual=mse_relative_residual,
        horizontal_mse_convergence_w_m2=(
            None if not tracer_active
            else (horizontal_mse_convergence_j_m2 / float(dt_seconds)).astype(np.float32)
        ),
    )


def evolve_variable_mass_pressure_coordinate_transport(
    lower_pressure_depth_pa: np.ndarray,
    midlevel_pressure_depth_pa: np.ndarray,
    upperlevel_pressure_depth_pa: np.ndarray,
    lower_humidity: np.ndarray,
    midlevel_humidity: np.ndarray,
    upperlevel_humidity: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    lower_u_m_s: np.ndarray,
    lower_v_m_s: np.ndarray,
    midlevel_u_m_s: np.ndarray,
    midlevel_v_m_s: np.ndarray,
    upperlevel_u_m_s: np.ndarray,
    upperlevel_v_m_s: np.ndarray,
    lower_mid_interface_mass_flux_kg_m2_s: np.ndarray,
    mid_upper_interface_mass_flux_kg_m2_s: np.ndarray,
    *,
    lower_surface_vapour_source_kg_m2_s: np.ndarray,
    lower_mid_interface_mass_flux_tendency_kg_m2_s2: np.ndarray | None = None,
    mid_upper_interface_mass_flux_tendency_kg_m2_s2: np.ndarray | None = None,
    dt_seconds: float,
    gravity_m_s2: float,
    cp_dry_j_kg_k: float = 1004.0,
    latent_heat_j_kg: float = 2.5e6,
    layer_heights_m: tuple[float, float, float] = (0.0, 3500.0, 8000.0),
    dx_m: np.ndarray | float = 1.0,
    dy_m: float = 1.0,
    cell_area_m2: np.ndarray | float = 1.0,
    x_face_length_m: np.ndarray | float = 1.0,
    y_face_length_m: np.ndarray | float = 1.0,
    max_vertical_courant: float = 0.25,
) -> VariableMassPressureCoordinateTransportStep:
    """Advance variable pressure mass, vapour, and MSE in one transition.

    All three conserved contents are first transported on the same horizontal
    faces: dry pressure mass, total vapour mass, and moist-static energy. The
    signed interface mass-flux states then move a donor parcel's mass, vapour,
    and MSE together. Surface vapour supply enters only the lower layer and
    brings its matching latent energy. This makes changing pressure thickness
    an explicit part of the water/energy state, rather than applying a
    fixed-mass tracer exchange after a separate circulation diagnosis.

    Interface-flux tendencies are an explicit input state boundary. They are
    not inferred from instantaneous MSE, water, condensation, or a target.
    Radiation, phase conversion, condensate reservoirs, and momentum remain
    outside this pure transport primitive until they can share its transition.
    """
    if dt_seconds <= 0.0 or gravity_m_s2 <= 0.0 or cp_dry_j_kg_k <= 0.0 or latent_heat_j_kg <= 0.0:
        raise ValueError("time and physical constants must be positive")
    if not 0.0 < max_vertical_courant <= 1.0:
        raise ValueError("max_vertical_courant must be in (0, 1]")
    raw = (
        lower_pressure_depth_pa, midlevel_pressure_depth_pa, upperlevel_pressure_depth_pa,
        lower_humidity, midlevel_humidity, upperlevel_humidity,
        lower_temperature_k, midlevel_temperature_k, upperlevel_temperature_k,
        lower_u_m_s, lower_v_m_s, midlevel_u_m_s, midlevel_v_m_s,
        upperlevel_u_m_s, upperlevel_v_m_s,
        lower_mid_interface_mass_flux_kg_m2_s, mid_upper_interface_mass_flux_kg_m2_s,
        lower_surface_vapour_source_kg_m2_s,
    )
    fields = tuple(np.asarray(value, dtype=np.float64) for value in raw)
    shape = fields[0].shape
    if len(shape) != 2 or shape[1] != 2 * shape[0] or any(value.shape != shape for value in fields):
        raise ValueError("variable-mass transport fields must share a two-dimensional 2:1 grid")
    if any(not np.all(np.isfinite(value)) for value in fields):
        raise ValueError("variable-mass transport fields must be finite")
    if any(np.any(value <= 0.0) for value in fields[:3]) or any(np.any(value < 0.0) for value in fields[3:6]) or np.any(fields[-1] < 0.0):
        raise ValueError("pressure depths must be positive; humidity and surface source non-negative")
    heights = np.asarray(layer_heights_m, dtype=np.float64)
    if heights.shape != (3,) or not np.all(np.isfinite(heights)):
        raise ValueError("layer_heights_m must be three finite values")
    tendency_lm = np.zeros(shape, dtype=np.float64) if lower_mid_interface_mass_flux_tendency_kg_m2_s2 is None else np.asarray(lower_mid_interface_mass_flux_tendency_kg_m2_s2, dtype=np.float64)
    tendency_mu = np.zeros(shape, dtype=np.float64) if mid_upper_interface_mass_flux_tendency_kg_m2_s2 is None else np.asarray(mid_upper_interface_mass_flux_tendency_kg_m2_s2, dtype=np.float64)
    if tendency_lm.shape != shape or tendency_mu.shape != shape or not np.all(np.isfinite(tendency_lm)) or not np.all(np.isfinite(tendency_mu)):
        raise ValueError("interface mass-flux tendencies must be finite and match the state")

    from column_water import evolve_column_water

    pressures, humidities, temperatures = fields[:3], fields[3:6], fields[6:9]
    winds = ((fields[9], fields[10]), (fields[11], fields[12]), (fields[13], fields[14]))
    mass: list[np.ndarray] = []
    water: list[np.ndarray] = []
    energy: list[np.ndarray] = []
    mass_steps = []
    water_steps = []
    energy_steps = []
    source = fields[17]
    for index in range(3):
        initial_mass = pressures[index] / float(gravity_m_s2)
        initial_water = initial_mass * humidities[index]
        initial_energy = initial_mass * (
            float(cp_dry_j_kg_k) * temperatures[index]
            + float(gravity_m_s2) * heights[index]
            + float(latent_heat_j_kg) * humidities[index]
        )
        source_water_day = source * 86400.0 if index == 0 else np.zeros(shape)
        source_energy_day = float(latent_heat_j_kg) * source_water_day if index == 0 else np.zeros(shape)
        kwargs = dict(dx_m=dx_m, dy_m=dy_m, dt_days=float(dt_seconds) / 86400.0,
                      cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
                      y_face_length_m=y_face_length_m)
        mass_step = evolve_column_water(initial_mass, np.zeros(shape), np.zeros(shape), *winds[index], **kwargs)
        water_step = evolve_column_water(initial_water, source_water_day, np.zeros(shape), *winds[index], **kwargs)
        energy_step = evolve_column_water(initial_energy, source_energy_day, np.zeros(shape), *winds[index], **kwargs)
        mass_steps.append(mass_step)
        water_steps.append(water_step)
        energy_steps.append(energy_step)
        mass.append(np.asarray(mass_step.water_mm, dtype=np.float64))
        water.append(np.asarray(water_step.water_mm, dtype=np.float64))
        energy.append(np.asarray(energy_step.water_mm, dtype=np.float64))

    if any(np.any(value <= 0.0) for value in mass):
        raise RuntimeError(
            "horizontal pressure transport exhausted a layer before vertical exchange; "
            "the supplied wind state is not an admissible hydrostatic sigma transition"
        )

    flux_lm_next = fields[15] + float(dt_seconds) * tendency_lm
    flux_mu_next = fields[16] + float(dt_seconds) * tendency_mu
    flux_lm = 0.5 * (fields[15] + flux_lm_next)
    flux_mu = 0.5 * (fields[16] + flux_mu_next)
    # Interface fluxes are held fixed over this pure transition.  Before
    # entering the CFL loop, reject an externally supplied flow that would
    # exhaust a layer during the requested interval.  Continuing with
    # ever-smaller substeps in that case is not stabilisation; it merely
    # approaches the physical singularity without reaching a valid state.
    vertical_mass_tendency = (-flux_lm, flux_lm - flux_mu, flux_mu)
    if any(np.any(mass[index] + float(dt_seconds) * vertical_mass_tendency[index] <= 0.0) for index in range(3)):
        raise RuntimeError("variable-mass pressure transport would exhaust a layer during the requested transition")

    def transfer(interface_flux: np.ndarray, lower: int, upper: int, elapsed_s: float) -> None:
        upward = interface_flux >= 0.0
        # Mixed-sign fields require cellwise donor properties.
        donor_mass = np.where(upward, mass[lower], mass[upper])
        donor_water = np.where(upward, water[lower], water[upper])
        donor_energy = np.where(upward, energy[lower], energy[upper])
        amount = np.abs(interface_flux) * elapsed_s
        fraction = amount / donor_mass
        if np.any(fraction > 1.0 + 1e-12):
            raise RuntimeError("variable-mass vertical transport exceeded its donor CFL bound")
        moved_water = donor_water * fraction
        moved_energy = donor_energy * fraction
        delta_mass = interface_flux * elapsed_s
        mass[lower] -= delta_mass
        mass[upper] += delta_mass
        water[lower] = np.where(upward, water[lower] - moved_water, water[lower] + moved_water)
        water[upper] = np.where(upward, water[upper] + moved_water, water[upper] - moved_water)
        energy[lower] = np.where(upward, energy[lower] - moved_energy, energy[lower] + moved_energy)
        energy[upper] = np.where(upward, energy[upper] + moved_energy, energy[upper] - moved_energy)

    remaining = float(dt_seconds)
    vertical_substeps = 0
    maximum_vertical_courant = 0.0
    while remaining > 0.0:
        outgoing_lower = np.maximum(flux_lm, 0.0)
        outgoing_middle = np.maximum(-flux_lm, 0.0) + np.maximum(flux_mu, 0.0)
        outgoing_upper = np.maximum(-flux_mu, 0.0)
        rate = np.maximum.reduce((outgoing_lower / mass[0], outgoing_middle / mass[1], outgoing_upper / mass[2]))
        rate_max = float(np.max(rate))
        dt_sub = remaining if rate_max == 0.0 else min(remaining, max_vertical_courant / rate_max)
        if not np.isfinite(dt_sub) or dt_sub <= 0.0:
            raise RuntimeError("variable-mass transport CFL policy produced an invalid substep")
        maximum_vertical_courant = max(maximum_vertical_courant, float(np.max(rate * dt_sub)))
        transfer(flux_lm, 0, 1, dt_sub)
        transfer(flux_mu, 1, 2, dt_sub)
        remaining -= dt_sub
        vertical_substeps += 1

    if any(np.any(value < -1e-10) for values in (mass, water) for value in values):
        raise RuntimeError("variable-mass transport produced a negative conserved inventory")
    mass = [np.maximum(value, 0.0) for value in mass]
    water = [np.maximum(value, 0.0) for value in water]
    humidity_next = [water[index] / mass[index] for index in range(3)]
    temperature_next = [
        (energy[index] / mass[index] - float(gravity_m_s2) * heights[index] - float(latent_heat_j_kg) * humidity_next[index]) / float(cp_dry_j_kg_k)
        for index in range(3)
    ]
    area = np.broadcast_to(np.asarray(cell_area_m2, dtype=np.float64), shape)
    expected_water = sum(float(np.sum(np.asarray(step.water_mm, dtype=np.float64) * area)) for step in water_steps)
    expected_energy = sum(float(np.sum(np.asarray(step.water_mm, dtype=np.float64) * area)) for step in energy_steps)
    actual_water = sum(float(np.sum(value * area)) for value in water)
    actual_energy = sum(float(np.sum(value * area)) for value in energy)
    return VariableMassPressureCoordinateTransportStep(
        *(value.astype(np.float32) * float(gravity_m_s2) for value in mass),
        *(value.astype(np.float32) for value in humidity_next),
        *(value.astype(np.float32) for value in temperature_next),
        flux_lm_next.astype(np.float32), flux_mu_next.astype(np.float32),
        (actual_water - expected_water) / max(abs(actual_water), abs(expected_water), 1.0),
        (actual_energy - expected_energy) / max(abs(actual_energy), abs(expected_energy), 1.0),
        max(step.substeps for step in mass_steps + water_steps + energy_steps),
        max(step.maximum_outgoing_courant for step in mass_steps + water_steps + energy_steps),
        vertical_substeps, maximum_vertical_courant,
    )


def evolve_hydrostatic_sigma_pressure_coordinate_transport(
    lower_pressure_depth_pa: np.ndarray,
    midlevel_pressure_depth_pa: np.ndarray,
    upperlevel_pressure_depth_pa: np.ndarray,
    lower_humidity: np.ndarray,
    midlevel_humidity: np.ndarray,
    upperlevel_humidity: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    lower_u_m_s: np.ndarray,
    lower_v_m_s: np.ndarray,
    midlevel_u_m_s: np.ndarray,
    midlevel_v_m_s: np.ndarray,
    upperlevel_u_m_s: np.ndarray,
    upperlevel_v_m_s: np.ndarray,
    *,
    lower_surface_vapour_source_kg_m2_s: np.ndarray,
    dt_seconds: float,
    gravity_m_s2: float,
    cp_dry_j_kg_k: float = 1004.0,
    latent_heat_j_kg: float = 2.5e6,
    layer_heights_m: tuple[float, float, float] = (0.0, 3500.0, 8000.0),
    dx_m: np.ndarray | float = 1.0,
    dy_m: float = 1.0,
    cell_area_m2: np.ndarray | float = 1.0,
    x_face_length_m: np.ndarray | float = 1.0,
    y_face_length_m: np.ndarray | float = 1.0,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
    max_vertical_courant: float = 0.25,
) -> HydrostaticSigmaPressureCoordinateTransportStep:
    """Advance the hydrostatic sigma state with continuity-derived omega.

    This is the first closed pressure-coordinate transport composition. It
    derives both interface fluxes from the same resolved horizontal mass
    transport that changes surface pressure, then moves mass, water, and MSE
    using those fluxes in one finite-volume transition. No external omega or
    interface acceleration is accepted, so callers cannot bypass the
    hydrostatic continuity contract with a lagged phase or heating residual.
    """
    continuity = diagnose_hydrostatic_sigma_continuity(
        lower_pressure_depth_pa, midlevel_pressure_depth_pa, upperlevel_pressure_depth_pa,
        lower_u_m_s, lower_v_m_s, midlevel_u_m_s, midlevel_v_m_s,
        upperlevel_u_m_s, upperlevel_v_m_s,
        dt_seconds=dt_seconds, gravity_m_s2=gravity_m_s2,
        dx_m=dx_m, dy_m=dy_m, cell_area_m2=cell_area_m2,
        x_face_length_m=x_face_length_m, y_face_length_m=y_face_length_m,
        layer_mass_fractions=layer_mass_fractions,
    )
    transport = evolve_variable_mass_pressure_coordinate_transport(
        lower_pressure_depth_pa, midlevel_pressure_depth_pa, upperlevel_pressure_depth_pa,
        lower_humidity, midlevel_humidity, upperlevel_humidity,
        lower_temperature_k, midlevel_temperature_k, upperlevel_temperature_k,
        lower_u_m_s, lower_v_m_s, midlevel_u_m_s, midlevel_v_m_s,
        upperlevel_u_m_s, upperlevel_v_m_s,
        continuity.lower_mid_interface_mass_flux_kg_m2_s,
        continuity.mid_upper_interface_mass_flux_kg_m2_s,
        lower_surface_vapour_source_kg_m2_s=lower_surface_vapour_source_kg_m2_s,
        dt_seconds=dt_seconds, gravity_m_s2=gravity_m_s2,
        cp_dry_j_kg_k=cp_dry_j_kg_k, latent_heat_j_kg=latent_heat_j_kg,
        layer_heights_m=layer_heights_m, dx_m=dx_m, dy_m=dy_m,
        cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
        y_face_length_m=y_face_length_m, max_vertical_courant=max_vertical_courant,
    )
    return HydrostaticSigmaPressureCoordinateTransportStep(transport, continuity)


def evolve_hydrostatic_sigma_phase_reservoir_transport(
    lower_pressure_depth_pa: np.ndarray, midlevel_pressure_depth_pa: np.ndarray, upperlevel_pressure_depth_pa: np.ndarray,
    lower_humidity: np.ndarray, midlevel_humidity: np.ndarray, upperlevel_humidity: np.ndarray,
    lower_temperature_k: np.ndarray, midlevel_temperature_k: np.ndarray, upperlevel_temperature_k: np.ndarray,
    lower_u_m_s: np.ndarray, lower_v_m_s: np.ndarray, midlevel_u_m_s: np.ndarray, midlevel_v_m_s: np.ndarray,
    upperlevel_u_m_s: np.ndarray, upperlevel_v_m_s: np.ndarray,
    lower_cloud_condensate_kg_m2: np.ndarray, midlevel_cloud_condensate_kg_m2: np.ndarray, upperlevel_cloud_condensate_kg_m2: np.ndarray,
    lower_hydrometeors_kg_m2: np.ndarray, midlevel_hydrometeors_kg_m2: np.ndarray, upperlevel_hydrometeors_kg_m2: np.ndarray,
    *, lower_surface_vapour_source_kg_m2_s: np.ndarray, dt_seconds: float, gravity_m_s2: float, radius_m: float, sidereal_day_hours: float,
    critical_relative_humidity: float, autoconversion_timescale_days: float, fallout_timescale_days: float,
    cloud_retention_kg_m2: float, cp_dry_j_kg_k: float = 1004.0, latent_heat_j_kg: float = 2.5e6,
    layer_heights_m: tuple[float, float, float] = (0.0, 3500.0, 8000.0),
    dx_m: np.ndarray | float = 1.0, dy_m: float = 1.0, cell_area_m2: np.ndarray | float = 1.0,
    x_face_length_m: np.ndarray | float = 1.0, y_face_length_m: np.ndarray | float = 1.0,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
    max_vertical_courant: float = 0.25,
) -> HydrostaticSigmaPhaseReservoirStep:
    """Own transport, phase conversion, and layer-resolved fallout once."""
    if not 0.0 < critical_relative_humidity < 1.0:
        raise ValueError("critical_relative_humidity must be in (0, 1)")
    carrier = evolve_hydrostatic_sigma_mass_momentum(
        lower_pressure_depth_pa, midlevel_pressure_depth_pa, upperlevel_pressure_depth_pa,
        lower_temperature_k, midlevel_temperature_k, upperlevel_temperature_k,
        lower_u_m_s, lower_v_m_s, midlevel_u_m_s, midlevel_v_m_s, upperlevel_u_m_s, upperlevel_v_m_s,
        dt_seconds=dt_seconds, radius_m=radius_m, sidereal_day_hours=sidereal_day_hours,
        gravity_m_s2=gravity_m_s2, cp_dry_j_kg_k=cp_dry_j_kg_k, latent_heat_j_kg=latent_heat_j_kg,
        layer_heights_m=layer_heights_m, lower_humidity=lower_humidity,
        midlevel_humidity=midlevel_humidity, upperlevel_humidity=upperlevel_humidity,
        lower_surface_vapour_source_kg_m2_s=lower_surface_vapour_source_kg_m2_s,
        lower_cloud_condensate_kg_m2=lower_cloud_condensate_kg_m2,
        midlevel_cloud_condensate_kg_m2=midlevel_cloud_condensate_kg_m2,
        upperlevel_cloud_condensate_kg_m2=upperlevel_cloud_condensate_kg_m2,
        lower_hydrometeors_kg_m2=lower_hydrometeors_kg_m2,
        midlevel_hydrometeors_kg_m2=midlevel_hydrometeors_kg_m2,
        upperlevel_hydrometeors_kg_m2=upperlevel_hydrometeors_kg_m2,
        dx_m=dx_m, dy_m=dy_m, cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
        y_face_length_m=y_face_length_m, layer_mass_fractions=layer_mass_fractions,
        max_vertical_courant=max_vertical_courant,
    )
    from condensate import evolve_pressure_condensate_reservoirs

    initial_pressures = tuple(np.asarray(value, dtype=np.float64) for value in (
        lower_pressure_depth_pa, midlevel_pressure_depth_pa, upperlevel_pressure_depth_pa,
    ))
    final_pressures = (
        carrier.lower_pressure_depth_pa, carrier.midlevel_pressure_depth_pa,
        carrier.upperlevel_pressure_depth_pa,
    )
    pressure_tendency = tuple(
        (np.asarray(final_pressures[index], dtype=np.float64) - initial_pressures[index])
        / float(gravity_m_s2) / float(dt_seconds)
        for index in range(3)
    )
    transport = VariableMassPressureCoordinateTransportStep(
        *final_pressures, carrier.lower_humidity, carrier.midlevel_humidity,
        carrier.upperlevel_humidity, carrier.lower_temperature,
        carrier.midlevel_temperature, carrier.upperlevel_temperature,
        carrier.lower_mid_interface_mass_flux_kg_m2_s,
        carrier.mid_upper_interface_mass_flux_kg_m2_s,
        float(carrier.water_relative_residual),
        float(carrier.moist_static_energy_relative_residual), carrier.substeps,
        carrier.horizontal_courant_max, carrier.substeps, carrier.vertical_courant_max,
    )
    continuity = HydrostaticSigmaContinuityStep(
        *pressure_tendency, carrier.lower_mid_interface_mass_flux_kg_m2_s,
        carrier.mid_upper_interface_mass_flux_kg_m2_s,
        (sum(np.asarray(value, dtype=np.float64) for value in final_pressures)
         - sum(initial_pressures)) / float(dt_seconds),
        0.0, carrier.substeps, carrier.horizontal_courant_max,
    )
    base = HydrostaticSigmaPressureCoordinateTransportStep(transport, continuity)
    state = transport
    pressures = (state.lower_pressure_depth_pa, state.midlevel_pressure_depth_pa, state.upperlevel_pressure_depth_pa)
    humidity = [np.asarray(state.lower_humidity, dtype=np.float64), np.asarray(state.midlevel_humidity, dtype=np.float64), np.asarray(state.upperlevel_humidity, dtype=np.float64)]
    temperature = [np.asarray(state.lower_temperature, dtype=np.float64), np.asarray(state.midlevel_temperature, dtype=np.float64), np.asarray(state.upperlevel_temperature, dtype=np.float64)]
    shape = humidity[0].shape
    reservoirs = tuple(np.asarray(value, dtype=np.float64) for value in (
        carrier.lower_cloud_condensate_kg_m2, carrier.midlevel_cloud_condensate_kg_m2,
        carrier.upperlevel_cloud_condensate_kg_m2, carrier.lower_hydrometeors_kg_m2,
        carrier.midlevel_hydrometeors_kg_m2, carrier.upperlevel_hydrometeors_kg_m2,
    ))
    if any(value.shape != shape or not np.all(np.isfinite(value)) or np.any(value < 0.0) for value in reservoirs):
        raise ValueError("layer condensate reservoirs must be finite, non-negative, and match the pressure state")
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(float(np.sum(fractions)), 1.0):
        raise ValueError("layer_mass_fractions must be three positive values summing to one")
    heights = np.asarray(layer_heights_m, dtype=np.float64)
    total_pressure = sum(np.asarray(value, dtype=np.float64) for value in pressures)
    cloud_next, hydro_next, fallout_parts = [], [], []
    phase_water_residual = 0.0
    for index in range(3):
        mass = np.asarray(pressures[index], dtype=np.float64) / float(gravity_m_s2)
        local_hpa = total_pressure * np.exp(-heights[index] / 8000.0) / 100.0
        tc = np.clip(temperature[index] - 273.15, -60.0, 60.0)
        es_hpa = 6.112 * np.exp(17.67 * tc / (tc + 243.5))
        qsat = np.clip(0.622 * es_hpa / local_hpa, 0.0, 0.035)
        lm = np.asarray(state.lower_mid_interface_mass_flux_kg_m2_s, dtype=np.float64)
        mu = np.asarray(state.mid_upper_interface_mass_flux_kg_m2_s, dtype=np.float64)
        ascent_rate = ((np.maximum(lm, 0.0), np.maximum.reduce((lm, mu, np.zeros(shape))), np.maximum(mu, 0.0))[index] / mass)
        activation = 1.0 - np.exp(-ascent_rate * float(dt_seconds))
        condensed_q = np.minimum(humidity[index], np.maximum(humidity[index] - critical_relative_humidity * qsat, 0.0) * activation + np.maximum(humidity[index] - qsat, 0.0))
        condensed_mass = mass * condensed_q
        energy = mass * (float(cp_dry_j_kg_k) * temperature[index] + float(gravity_m_s2) * heights[index] + float(latent_heat_j_kg) * humidity[index])
        humidity[index] -= condensed_q
        temperature[index] = (energy / mass - float(gravity_m_s2) * heights[index] - float(latent_heat_j_kg) * humidity[index]) / float(cp_dry_j_kg_k)
        reservoir = evolve_pressure_condensate_reservoirs(
            reservoirs[index], reservoirs[3 + index], condensed_mass,
            dt_days=dt_seconds / 86400.0, autoconversion_timescale_days=autoconversion_timescale_days,
            fallout_timescale_days=fallout_timescale_days,
            cloud_retention_kg_m2=float(cloud_retention_kg_m2) * fractions[index],
        )
        cloud_next.append(reservoir.cloud_condensate_kg_m2)
        hydro_next.append(reservoir.precipitating_hydrometeors_kg_m2)
        fallout_parts.append(reservoir.fallout_kg_m2)
    fallout = sum(np.asarray(value, dtype=np.float64) for value in fallout_parts)
    phased_transport = state._replace(
        lower_humidity=humidity[0].astype(np.float32), midlevel_humidity=humidity[1].astype(np.float32),
        upperlevel_humidity=humidity[2].astype(np.float32), lower_temperature=temperature[0].astype(np.float32),
        midlevel_temperature=temperature[1].astype(np.float32), upperlevel_temperature=temperature[2].astype(np.float32),
    )
    phased_base = base._replace(transport=phased_transport)
    return HydrostaticSigmaPhaseReservoirStep(
        phased_base, *(np.asarray(value, dtype=np.float32) for value in cloud_next), *(np.asarray(value, dtype=np.float32) for value in hydro_next), fallout.astype(np.float32),
        max(abs(state.water_relative_residual), phase_water_residual), state.moist_static_energy_relative_residual, carrier,
    )


def evolve_variable_mass_pressure_momentum(
    lower_pressure_depth_pa: np.ndarray, midlevel_pressure_depth_pa: np.ndarray, upperlevel_pressure_depth_pa: np.ndarray,
    lower_u_m_s: np.ndarray, lower_v_m_s: np.ndarray, midlevel_u_m_s: np.ndarray, midlevel_v_m_s: np.ndarray,
    upperlevel_u_m_s: np.ndarray, upperlevel_v_m_s: np.ndarray,
    lower_temperature_k: np.ndarray, midlevel_temperature_k: np.ndarray, upperlevel_temperature_k: np.ndarray,
    lower_mid_interface_mass_flux_kg_m2_s: np.ndarray, mid_upper_interface_mass_flux_kg_m2_s: np.ndarray,
    *, dt_seconds: float, radius_m: float, sidereal_day_hours: float, gravity_m_s2: float = 9.80665,
    gas_constant_dry_j_kg_k: float = 287.05, max_vertical_courant: float = 0.25,
) -> VariableMassPressureMomentumStep:
    """Advance pressure-level winds with variable-mass interface exchange.

    Hydrostatic layer-center geopotential supplies the meridional pressure
    gradient. Coriolis rotation is analytic, including at the equator. The
    continuity-derived signed interface mass flux then transfers horizontal
    momentum with its donor air under an adaptive combined-donor CFL bound.
    No Rayleigh term, speed cap, fixed layer mass, or prescribed substep is
    introduced.
    """
    if dt_seconds <= 0.0 or radius_m <= 0.0 or sidereal_day_hours <= 0.0 or gravity_m_s2 <= 0.0 or gas_constant_dry_j_kg_k <= 0.0:
        raise ValueError("time, geometry, and physical constants must be positive")
    if not 0.0 < max_vertical_courant <= 1.0:
        raise ValueError("max_vertical_courant must be in (0, 1]")
    raw = (lower_pressure_depth_pa, midlevel_pressure_depth_pa, upperlevel_pressure_depth_pa,
           lower_u_m_s, lower_v_m_s, midlevel_u_m_s, midlevel_v_m_s, upperlevel_u_m_s, upperlevel_v_m_s,
           lower_temperature_k, midlevel_temperature_k, upperlevel_temperature_k,
           lower_mid_interface_mass_flux_kg_m2_s, mid_upper_interface_mass_flux_kg_m2_s)
    fields = tuple(np.asarray(value, dtype=np.float64) for value in raw)
    shape = fields[0].shape
    if len(shape) != 2 or shape[1] != 2 * shape[0] or any(value.shape != shape for value in fields):
        raise ValueError("variable-mass momentum fields must share a two-dimensional 2:1 grid")
    if any(not np.all(np.isfinite(value)) for value in fields) or any(np.any(value <= 0.0) for value in fields[:3]):
        raise ValueError("variable-mass momentum fields must be finite with positive pressure depths")
    h, _ = shape
    pressure_total = fields[0] + fields[1] + fields[2]
    centers = (pressure_total - 0.5 * fields[0], pressure_total - fields[0] - 0.5 * fields[1], 0.5 * fields[2])
    dy_m = float(radius_m) * np.pi / h
    gradients = []
    for temperature, center_pressure in zip(fields[9:12], centers, strict=True):
        geopotential = float(gas_constant_dry_j_kg_k) * np.mean(temperature * np.log(pressure_total / center_pressure), axis=1)
        gradients.append(np.broadcast_to(np.gradient(geopotential, dy_m, edge_order=1)[:, None], shape).copy())
    latitude = 0.5 * np.pi - (np.arange(h) + 0.5) * np.pi / h
    coriolis = 4.0 * np.pi / (float(sidereal_day_hours) * 3600.0) * np.sin(latitude)
    theta = coriolis[:, None] * float(dt_seconds)
    cosine, sine = np.cos(theta), np.sin(theta)
    small = np.abs(theta) < 1e-5
    sin_over_f = np.where(small, float(dt_seconds) * (1.0 - theta**2 / 6.0), sine / coriolis[:, None])
    one_minus_cos_over_f = np.where(small, 0.5 * float(dt_seconds) * theta, (1.0 - cosine) / coriolis[:, None])
    velocity = [fields[3].copy(), fields[4].copy(), fields[5].copy(), fields[6].copy(), fields[7].copy(), fields[8].copy()]
    for index, gradient in enumerate(gradients):
        u, v = velocity[2 * index], velocity[2 * index + 1]
        velocity[2 * index] = cosine * u + sine * v + one_minus_cos_over_f * gradient
        velocity[2 * index + 1] = -sine * u + cosine * v + sin_over_f * gradient
    mass = [fields[index].copy() / float(gravity_m_s2) for index in range(3)]
    flux_lm, flux_mu = fields[12], fields[13]
    vertical_mass_tendency = (-flux_lm, flux_lm - flux_mu, flux_mu)
    if any(np.any(mass[index] + float(dt_seconds) * vertical_mass_tendency[index] <= 0.0) for index in range(3)):
        raise RuntimeError("variable-mass momentum exchange would exhaust a layer during the requested transition")
    remaining, maximum_courant = float(dt_seconds), 0.0
    # Momentum is externally forced by the hydrostatic PGF/Coriolis leg.  The
    # following residual measures only the interface exchange, whose initial
    # component inventories are retained separately.
    before_exchange = [sum(mass[index] * velocity[2 * index + component] for index in range(3)) for component in (0, 1)]
    while remaining > 0.0:
        out0, out1, out2 = np.maximum(flux_lm, 0.0), np.maximum(-flux_lm, 0.0) + np.maximum(flux_mu, 0.0), np.maximum(-flux_mu, 0.0)
        rate = np.maximum.reduce((out0 / mass[0], out1 / mass[1], out2 / mass[2]))
        rate_max = float(np.max(rate))
        dt_sub = remaining if rate_max == 0.0 else min(remaining, max_vertical_courant / rate_max)
        maximum_courant = max(maximum_courant, float(np.max(rate * dt_sub)))
        for flux, lower, upper in ((flux_lm, 0, 1), (flux_mu, 1, 2)):
            upward = flux >= 0.0
            donor_mass = np.where(upward, mass[lower], mass[upper])
            amount = np.abs(flux) * dt_sub
            if np.any(amount > donor_mass * (1.0 + 1e-12)):
                raise RuntimeError("variable-mass momentum exchange exceeded its donor CFL bound")
            for component in (0, 1):
                donor_value = np.where(upward, velocity[2 * lower + component], velocity[2 * upper + component])
                content = amount * donor_value
                lower_content = mass[lower] * velocity[2 * lower + component]
                upper_content = mass[upper] * velocity[2 * upper + component]
                lower_content = np.where(upward, lower_content - content, lower_content + content)
                upper_content = np.where(upward, upper_content + content, upper_content - content)
                velocity[2 * lower + component] = lower_content / (mass[lower] - flux * dt_sub)
                velocity[2 * upper + component] = upper_content / (mass[upper] + flux * dt_sub)
            mass[lower] -= flux * dt_sub
            mass[upper] += flux * dt_sub
        remaining -= dt_sub
    after_exchange = [sum(mass[index] * velocity[2 * index + component] for index in range(3)) for component in (0, 1)]
    exchange_residual = max(abs(float(np.sum(after_exchange[i] - before_exchange[i]))) / max(abs(float(np.sum(before_exchange[i]))), 1.0) for i in (0, 1))
    return VariableMassPressureMomentumStep(
        *(value.astype(np.float32) for value in velocity), *(value.astype(np.float32) for value in gradients), maximum_courant, exchange_residual,
    )


def evolve_joint_mse_momentum_pressure_column(
    large_scale_heating_w_m2: np.ndarray,
    lower_humidity: np.ndarray,
    midlevel_humidity: np.ndarray,
    upperlevel_humidity: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    lower_u_m_s: np.ndarray,
    lower_v_m_s: np.ndarray,
    midlevel_u_m_s: np.ndarray,
    midlevel_v_m_s: np.ndarray,
    upperlevel_u_m_s: np.ndarray,
    upperlevel_v_m_s: np.ndarray,
    *,
    dt_seconds: float,
    radius_m: float,
    sidereal_day_hours: float,
    surface_pressure_pa: float,
    lower_mid_pressure_depth_pa: float,
    mid_upper_pressure_depth_pa: float,
    gravity_m_s2: float = 9.80665,
    cp_dry_j_kg_k: float = 1004.0,
    gas_constant_dry_j_kg_k: float = 287.05,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
    layer_heights_m: tuple[float, float, float] = (0.0, 3500.0, 8000.0),
    max_vertical_courant: float = 0.25,
    column_water_forcing_kg_m2_s: np.ndarray | None = None,
) -> JointPressureColumnStep:
    """Jointly substep the MSE circulation, vertical column, and momentum.

    Every substep starts by re-diagnosing the three-branch MSE/momentum
    circulation from the current state. The same interface flux is then used
    for a half momentum step, conservative vertical MSE exchange, and a second
    half momentum step. The next circulation is diagnosed from that updated
    state. Adaptive substep length is derived solely from the interface CFL;
    there is no omega cap, wind relaxation, or prescribed inner timestep.

    Horizontal MSE transport remains the caller's responsibility. This kernel
    closes the *simultaneous vertical* MSE/momentum response, providing the
    runtime adapter a single state-transition contract before horizontal
    transport and phase deposition are folded into the same cadence.
    """
    if dt_seconds <= 0.0 or not 0.0 < max_vertical_courant <= 1.0:
        raise ValueError("dt_seconds must be positive and max_vertical_courant in (0, 1]")
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(float(np.sum(fractions)), 1.0):
        raise ValueError("layer_mass_fractions must be three positive values summing to one")
    fields = tuple(np.asarray(value, dtype=np.float64) for value in (
        large_scale_heating_w_m2, lower_humidity, midlevel_humidity, upperlevel_humidity,
        lower_temperature_k, midlevel_temperature_k, upperlevel_temperature_k,
        lower_u_m_s, lower_v_m_s, midlevel_u_m_s, midlevel_v_m_s,
        upperlevel_u_m_s, upperlevel_v_m_s,
    ))
    shape = fields[0].shape
    if len(shape) != 2 or shape[1] != 2 * shape[0] or any(value.shape != shape for value in fields):
        raise ValueError("joint pressure-column fields must share a two-dimensional 2:1 grid")
    if any(not np.all(np.isfinite(value)) for value in fields):
        raise ValueError("joint pressure-column fields must be finite")
    if any(np.any(value < 0.0) for value in fields[1:4]):
        raise ValueError("joint pressure-column humidities must be non-negative")
    heating, q0, q1, q2, t0, t1, t2, u0, v0, u1, v1, u2, v2 = (value.copy() for value in fields)
    layer_mass = fractions * (float(surface_pressure_pa) / float(gravity_m_s2))
    remaining = float(dt_seconds)
    substeps = 0
    maximum_courant = 0.0
    water_residual = 0.0
    energy_residual = 0.0
    final_interface: DiabaticInterfaceMassFlux | None = None
    from pressure_column import evolve_closed_three_level_thermodynamic_column

    while remaining > 0.0:
        circulation_kwargs = dict(
            dt_seconds=remaining, radius_m=radius_m, surface_pressure_pa=surface_pressure_pa,
            lower_mid_pressure_depth_pa=lower_mid_pressure_depth_pa,
            mid_upper_pressure_depth_pa=mid_upper_pressure_depth_pa,
            gravity_m_s2=gravity_m_s2, cp_dry_j_kg_k=cp_dry_j_kg_k,
            layer_mass_fractions=layer_mass_fractions, layer_heights_m=layer_heights_m,
        )
        diagnosed = (
            water_constrained_three_branch_mse_pressure_coordinate_circulation(
                heating, column_water_forcing_kg_m2_s, t0, t1, t2, q0, q1, q2,
                u0, u1, u2, **circulation_kwargs,
            )
            if column_water_forcing_kg_m2_s is not None
            else momentum_constrained_three_branch_mse_pressure_coordinate_circulation(
                heating, t0, t1, t2, q0, q1, q2, u0, u1, u2,
                **circulation_kwargs,
            )
        )
        interface = diagnosed.circulation.interface_mass_flux
        omega_peak = max(
            float(np.max(np.abs(interface.omega_lower_mid_pa_s))),
            float(np.max(np.abs(interface.omega_mid_upper_pa_s))),
        )
        # The pressure-layer donor mass sets the physical transfer time. The
        # two interfaces have distinct donor pairs, so use the stricter one.
        if omega_peak == 0.0:
            dt_sub = remaining
        else:
            transfer_rate = max(
                float(np.max(np.abs(interface.omega_lower_mid_pa_s))) / float(gravity_m_s2) / min(layer_mass[0], layer_mass[1]),
                float(np.max(np.abs(interface.omega_mid_upper_pa_s))) / float(gravity_m_s2) / min(layer_mass[1], layer_mass[2]),
            )
            dt_sub = min(remaining, float(max_vertical_courant) / transfer_rate)
        if not np.isfinite(dt_sub) or dt_sub <= 0.0:
            raise RuntimeError("joint pressure-column CFL policy produced an invalid substep")
        courant = max(
            float(np.max(np.abs(interface.omega_lower_mid_pa_s))) * dt_sub / float(gravity_m_s2) / min(layer_mass[0], layer_mass[1]),
            float(np.max(np.abs(interface.omega_mid_upper_pa_s))) * dt_sub / float(gravity_m_s2) / min(layer_mass[1], layer_mass[2]),
        )
        maximum_courant = max(maximum_courant, courant)
        first_momentum = evolve_three_level_zonal_momentum(
            u0, v0, u1, v1, u2, v2, t0, t1, t2,
            interface.omega_lower_mid_pa_s, interface.omega_mid_upper_pa_s,
            dt_seconds=0.5 * dt_sub, radius_m=radius_m,
            sidereal_day_hours=sidereal_day_hours, surface_pressure_pa=surface_pressure_pa,
            gravity_m_s2=gravity_m_s2, gas_constant_dry_j_kg_k=gas_constant_dry_j_kg_k,
            layer_mass_fractions=layer_mass_fractions,
        )
        column = evolve_closed_three_level_thermodynamic_column(
            q0, q1, q2, t0, t1, t2,
            interface.omega_lower_mid_pa_s, interface.omega_mid_upper_pa_s,
            dt_seconds=dt_sub, surface_pressure_pa=surface_pressure_pa,
            layer_mass_fractions=layer_mass_fractions, layer_heights_m=layer_heights_m,
        )
        second_momentum = evolve_three_level_zonal_momentum(
            first_momentum.lower_u, first_momentum.lower_v,
            first_momentum.midlevel_u, first_momentum.midlevel_v,
            first_momentum.upperlevel_u, first_momentum.upperlevel_v,
            column.lower_temperature, column.midlevel_temperature, column.upperlevel_temperature,
            interface.omega_lower_mid_pa_s, interface.omega_mid_upper_pa_s,
            dt_seconds=0.5 * dt_sub, radius_m=radius_m,
            sidereal_day_hours=sidereal_day_hours, surface_pressure_pa=surface_pressure_pa,
            gravity_m_s2=gravity_m_s2, gas_constant_dry_j_kg_k=gas_constant_dry_j_kg_k,
            layer_mass_fractions=layer_mass_fractions,
        )
        q0, q1, q2 = column.lower_humidity, column.midlevel_humidity, column.upperlevel_humidity
        t0, t1, t2 = column.lower_temperature, column.midlevel_temperature, column.upperlevel_temperature
        u0, v0, u1, v1, u2, v2 = (
            second_momentum.lower_u, second_momentum.lower_v,
            second_momentum.midlevel_u, second_momentum.midlevel_v,
            second_momentum.upperlevel_u, second_momentum.upperlevel_v,
        )
        water_residual += column.water_residual_kg_m2
        energy_residual += column.moist_static_energy_residual_j_m2
        final_interface = interface
        remaining -= dt_sub
        substeps += 1
    assert final_interface is not None
    return JointPressureColumnStep(
        q0.astype(np.float32), q1.astype(np.float32), q2.astype(np.float32),
        t0.astype(np.float32), t1.astype(np.float32), t2.astype(np.float32),
        u0.astype(np.float32), v0.astype(np.float32), u1.astype(np.float32), v1.astype(np.float32),
        u2.astype(np.float32), v2.astype(np.float32), final_interface, substeps,
        maximum_courant, water_residual, energy_residual,
    )


def evolve_joint_mse_momentum_pressure_column_runtime(
    large_scale_heating_w_m2: np.ndarray,
    lower_humidity_before_horizontal: np.ndarray,
    midlevel_humidity_before_horizontal: np.ndarray,
    upperlevel_humidity_before_horizontal: np.ndarray,
    lower_humidity_after_horizontal: np.ndarray,
    midlevel_humidity_after_horizontal: np.ndarray,
    upperlevel_humidity_after_horizontal: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    lower_u_m_s: np.ndarray,
    lower_v_m_s: np.ndarray,
    midlevel_u_m_s: np.ndarray,
    midlevel_v_m_s: np.ndarray,
    upperlevel_u_m_s: np.ndarray,
    upperlevel_v_m_s: np.ndarray,
    *,
    lower_vapour_source_kg_kg_day: np.ndarray,
    column_water_forcing_kg_m2_s: np.ndarray | None = None,
    dt_seconds: float,
    radius_m: float,
    sidereal_day_hours: float,
    surface_pressure_pa: float,
    lower_mid_pressure_depth_pa: float,
    mid_upper_pressure_depth_pa: float,
    dx_m: np.ndarray | float,
    dy_m: float,
    cell_area_m2: np.ndarray | float,
    x_face_length_m: np.ndarray | float,
    y_face_length_m: np.ndarray | float,
    critical_relative_humidity: float,
    layer_heights_m: tuple[float, float, float] = (0.0, 3500.0, 8000.0),
    gravity_m_s2: float = 9.80665,
    cp_dry_j_kg_k: float = 1004.0,
    gas_constant_dry_j_kg_k: float = 287.05,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
    max_vertical_courant: float = 0.25,
) -> JointPressureColumnRuntimeStep:
    """Apply horizontal MSE, joint vertical dynamics, and phase heating once.

    The caller supplies the three humidity fields before and after its
    conservative horizontal vapour transport.  This mirrors that transport for
    MSE, advances the resulting column with the sole adaptive interface-CFL
    policy, then converts the diagnosed pressure-layer condensate to heat in
    the same state transition.  Rainout remains a reservoir operation in the
    host microphysics; it must not reapply this phase conversion.
    """
    if dt_seconds <= 0.0 or not 0.0 < critical_relative_humidity < 1.0:
        raise ValueError("dt_seconds must be positive and critical_relative_humidity in (0, 1)")
    from pressure_column import (
        evolve_closed_three_level_thermodynamic_column,
        transport_closed_three_level_mse,
    )

    horizontal = transport_closed_three_level_mse(
        lower_humidity_before_horizontal, midlevel_humidity_before_horizontal,
        upperlevel_humidity_before_horizontal, lower_humidity_after_horizontal,
        midlevel_humidity_after_horizontal, upperlevel_humidity_after_horizontal,
        lower_temperature_k, midlevel_temperature_k, upperlevel_temperature_k,
        lower_u_m_s, lower_v_m_s, midlevel_u_m_s, midlevel_v_m_s,
        upperlevel_u_m_s, upperlevel_v_m_s,
        lower_vapour_source_kg_kg_day=lower_vapour_source_kg_kg_day,
        dt_days=dt_seconds / 86400.0, dx_m=dx_m, dy_m=dy_m,
        cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
        y_face_length_m=y_face_length_m, layer_mass_fractions=layer_mass_fractions,
        surface_pressure_pa=surface_pressure_pa, layer_heights_m=layer_heights_m,
    )
    joint = evolve_joint_mse_momentum_pressure_column(
        large_scale_heating_w_m2,
        lower_humidity_after_horizontal, midlevel_humidity_after_horizontal,
        upperlevel_humidity_after_horizontal,
        horizontal.lower_temperature, horizontal.midlevel_temperature,
        horizontal.upperlevel_temperature,
        lower_u_m_s, lower_v_m_s, midlevel_u_m_s, midlevel_v_m_s,
        upperlevel_u_m_s, upperlevel_v_m_s,
        dt_seconds=dt_seconds, radius_m=radius_m,
        sidereal_day_hours=sidereal_day_hours,
        surface_pressure_pa=surface_pressure_pa,
        lower_mid_pressure_depth_pa=lower_mid_pressure_depth_pa,
        mid_upper_pressure_depth_pa=mid_upper_pressure_depth_pa,
        gravity_m_s2=gravity_m_s2, cp_dry_j_kg_k=cp_dry_j_kg_k,
        gas_constant_dry_j_kg_k=gas_constant_dry_j_kg_k,
        layer_mass_fractions=layer_mass_fractions, layer_heights_m=layer_heights_m,
        max_vertical_courant=max_vertical_courant,
        column_water_forcing_kg_m2_s=column_water_forcing_kg_m2_s,
    )
    heights = np.asarray(layer_heights_m, dtype=np.float64)
    pressure_hpa = float(surface_pressure_pa) / 100.0

    def saturation(temperature_k: np.ndarray, height_m: float) -> np.ndarray:
        local_pressure_hpa = pressure_hpa * np.exp(-height_m / 8000.0)
        tc = np.clip(np.asarray(temperature_k, dtype=np.float64) - 273.15, -60.0, 60.0)
        es = 6.112 * np.exp(17.67 * tc / (tc + 243.5))
        return np.clip(0.622 * es / local_pressure_hpa, 0.0, 0.035)

    lower_qsat = saturation(joint.lower_temperature, heights[0])
    mid_qsat = saturation(joint.midlevel_temperature, heights[1])
    upper_qsat = saturation(joint.upperlevel_temperature, heights[2])
    omega_lower_mid = joint.interface_mass_flux.omega_lower_mid_pa_s
    omega_mid_upper = joint.interface_mass_flux.omega_mid_upper_pa_s
    lower_activation = 1.0 - np.exp(
        -np.maximum(-omega_lower_mid, 0.0) * dt_seconds / lower_mid_pressure_depth_pa
    )
    mid_activation = 1.0 - np.exp(-np.maximum.reduce((
        np.maximum(-omega_lower_mid, 0.0) / lower_mid_pressure_depth_pa,
        np.maximum(-omega_mid_upper, 0.0) / mid_upper_pressure_depth_pa,
    )) * dt_seconds)
    upper_activation = 1.0 - np.exp(
        -np.maximum(-omega_mid_upper, 0.0) * dt_seconds / mid_upper_pressure_depth_pa
    )
    condensed = (
        np.minimum(joint.lower_humidity, np.maximum(
            joint.lower_humidity - critical_relative_humidity * lower_qsat, 0.0
        ) * lower_activation + np.maximum(joint.lower_humidity - lower_qsat, 0.0)),
        np.minimum(joint.midlevel_humidity, np.maximum(
            joint.midlevel_humidity - critical_relative_humidity * mid_qsat, 0.0
        ) * mid_activation + np.maximum(joint.midlevel_humidity - mid_qsat, 0.0)),
        np.minimum(joint.upperlevel_humidity, np.maximum(
            joint.upperlevel_humidity - critical_relative_humidity * upper_qsat, 0.0
        ) * upper_activation + np.maximum(joint.upperlevel_humidity - upper_qsat, 0.0)),
    )
    phase = evolve_closed_three_level_thermodynamic_column(
        joint.lower_humidity, joint.midlevel_humidity, joint.upperlevel_humidity,
        joint.lower_temperature, joint.midlevel_temperature, joint.upperlevel_temperature,
        np.zeros_like(joint.lower_humidity), np.zeros_like(joint.lower_humidity),
        dt_seconds=dt_seconds, surface_pressure_pa=surface_pressure_pa,
        layer_mass_fractions=layer_mass_fractions, layer_heights_m=layer_heights_m,
        condensed_specific_humidity=condensed,
    )
    phased_joint = JointPressureColumnStep(
        phase.lower_humidity, phase.midlevel_humidity, phase.upperlevel_humidity,
        phase.lower_temperature, phase.midlevel_temperature, phase.upperlevel_temperature,
        joint.lower_u, joint.lower_v, joint.midlevel_u, joint.midlevel_v,
        joint.upperlevel_u, joint.upperlevel_v, joint.interface_mass_flux,
        joint.substeps, joint.vertical_courant_max,
        joint.water_residual_kg_m2 + phase.water_residual_kg_m2,
        joint.moist_static_energy_residual_j_m2 + phase.moist_static_energy_residual_j_m2,
    )
    return JointPressureColumnRuntimeStep(
        phased_joint, *(item.astype(np.float32) for item in condensed),
        horizontal.energy_residual_j, horizontal.relative_energy_residual,
        phase.water_residual_kg_m2, phase.moist_static_energy_residual_j_m2,
    )


def diagnose_joint_pressure_column_coupling_residual(
    candidate_heating_w_m2: np.ndarray,
    candidate_water_forcing_kg_m2_s: np.ndarray,
    previous_heating_w_m2: np.ndarray,
    surface_source_kg_m2: np.ndarray,
    cloud_condensate_kg_m2: np.ndarray,
    precipitating_hydrometeors_kg_m2: np.ndarray,
    lower_humidity_before_horizontal: np.ndarray,
    midlevel_humidity_before_horizontal: np.ndarray,
    upperlevel_humidity_before_horizontal: np.ndarray,
    lower_humidity_after_horizontal: np.ndarray,
    midlevel_humidity_after_horizontal: np.ndarray,
    upperlevel_humidity_after_horizontal: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    lower_u_m_s: np.ndarray,
    lower_v_m_s: np.ndarray,
    midlevel_u_m_s: np.ndarray,
    midlevel_v_m_s: np.ndarray,
    upperlevel_u_m_s: np.ndarray,
    upperlevel_v_m_s: np.ndarray,
    *,
    lower_vapour_source_kg_kg_day: np.ndarray,
    dt_seconds: float,
    radius_m: float,
    sidereal_day_hours: float,
    surface_pressure_pa: float,
    lower_mid_pressure_depth_pa: float,
    mid_upper_pressure_depth_pa: float,
    dx_m: np.ndarray | float,
    dy_m: float,
    cell_area_m2: np.ndarray | float,
    x_face_length_m: np.ndarray | float,
    y_face_length_m: np.ndarray | float,
    critical_relative_humidity: float,
    cloud_retention_kg_m2: float,
    autoconversion_timescale_days: float,
    fallout_timescale_days: float,
    layer_heights_m: tuple[float, float, float] = (0.0, 3500.0, 8000.0),
    gravity_m_s2: float = 9.80665,
    cp_dry_j_kg_k: float = 1004.0,
    gas_constant_dry_j_kg_k: float = 287.05,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
    max_vertical_courant: float = 0.25,
) -> JointPressureColumnCouplingResidual:
    """Evaluate, but do not solve, the simultaneous water/heating equations.

    A valid coupled transition requires its candidate large-scale heating and
    external column-water forcing to equal the values diagnosed from its own
    phase conversion and cloud/hydrometeor reservoirs.
    """
    if dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be positive")
    from condensate import (
        column_water_forcing_from_budget,
        evolve_pressure_condensate_reservoirs,
    )

    runtime = evolve_joint_mse_momentum_pressure_column_runtime(
        candidate_heating_w_m2,
        lower_humidity_before_horizontal, midlevel_humidity_before_horizontal,
        upperlevel_humidity_before_horizontal, lower_humidity_after_horizontal,
        midlevel_humidity_after_horizontal, upperlevel_humidity_after_horizontal,
        lower_temperature_k, midlevel_temperature_k, upperlevel_temperature_k,
        lower_u_m_s, lower_v_m_s, midlevel_u_m_s, midlevel_v_m_s,
        upperlevel_u_m_s, upperlevel_v_m_s,
        lower_vapour_source_kg_kg_day=lower_vapour_source_kg_kg_day,
        column_water_forcing_kg_m2_s=candidate_water_forcing_kg_m2_s,
        dt_seconds=dt_seconds, radius_m=radius_m,
        sidereal_day_hours=sidereal_day_hours, surface_pressure_pa=surface_pressure_pa,
        lower_mid_pressure_depth_pa=lower_mid_pressure_depth_pa,
        mid_upper_pressure_depth_pa=mid_upper_pressure_depth_pa, dx_m=dx_m,
        dy_m=dy_m, cell_area_m2=cell_area_m2, x_face_length_m=x_face_length_m,
        y_face_length_m=y_face_length_m,
        critical_relative_humidity=critical_relative_humidity,
        layer_heights_m=layer_heights_m, gravity_m_s2=gravity_m_s2,
        cp_dry_j_kg_k=cp_dry_j_kg_k, gas_constant_dry_j_kg_k=gas_constant_dry_j_kg_k,
        layer_mass_fractions=layer_mass_fractions,
        max_vertical_courant=max_vertical_courant,
    )
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(float(np.sum(fractions)), 1.0):
        raise ValueError("layer_mass_fractions must be three positive values summing to one")
    layer_mass = fractions * float(surface_pressure_pa) / float(gravity_m_s2)
    condensed_kg_m2 = (
        layer_mass[0] * runtime.lower_condensed_specific_humidity
        + layer_mass[1] * runtime.midlevel_condensed_specific_humidity
        + layer_mass[2] * runtime.upperlevel_condensed_specific_humidity
    )
    reservoir = evolve_pressure_condensate_reservoirs(
        cloud_condensate_kg_m2, precipitating_hydrometeors_kg_m2,
        condensed_kg_m2, dt_days=dt_seconds / 86400.0,
        autoconversion_timescale_days=autoconversion_timescale_days,
        fallout_timescale_days=fallout_timescale_days,
        cloud_retention_kg_m2=cloud_retention_kg_m2,
    )
    # The three pressure vapour layers plus the two persistent condensed-water
    # reservoirs are the complete atmospheric-water store.  The circulation
    # constraint is its *transient* horizontal convergence, not merely its
    # external boundary source.  Retain the boundary helper above as a useful
    # public diagnostic; the simultaneous residual must include storage.
    water_before = (
        layer_mass[0] * np.asarray(lower_humidity_before_horizontal, dtype=np.float64)
        + layer_mass[1] * np.asarray(midlevel_humidity_before_horizontal, dtype=np.float64)
        + layer_mass[2] * np.asarray(upperlevel_humidity_before_horizontal, dtype=np.float64)
        + np.asarray(cloud_condensate_kg_m2, dtype=np.float64)
        + np.asarray(precipitating_hydrometeors_kg_m2, dtype=np.float64)
    )
    water_after = (
        layer_mass[0] * runtime.joint.lower_humidity
        + layer_mass[1] * runtime.joint.midlevel_humidity
        + layer_mass[2] * runtime.joint.upperlevel_humidity
        + reservoir.cloud_condensate_kg_m2
        + reservoir.precipitating_hydrometeors_kg_m2
    )
    diagnosed_water = column_water_forcing_from_budget(
        surface_source_kg_m2, reservoir.fallout_kg_m2, water_before, water_after,
        dt_seconds=dt_seconds,
    )
    heating = evolve_large_scale_heating_reservoir(
        previous_heating_w_m2, condensed_kg_m2 / (dt_seconds / 86400.0),
        runtime.joint.lower_temperature, runtime.joint.midlevel_temperature,
        runtime.joint.upperlevel_temperature, dt_seconds=dt_seconds,
        surface_pressure_pa=surface_pressure_pa, gravity_m_s2=gravity_m_s2,
        cp_dry_j_kg_k=cp_dry_j_kg_k,
    ).heating_w_m2
    candidate_water = np.asarray(candidate_water_forcing_kg_m2_s, dtype=np.float64)
    candidate_heating = np.asarray(candidate_heating_w_m2, dtype=np.float64)
    if candidate_water.shape != diagnosed_water.shape or candidate_heating.shape != heating.shape:
        raise ValueError("coupling candidates must match the diagnosed pressure-column grid")
    return JointPressureColumnCouplingResidual(
        runtime, reservoir.cloud_condensate_kg_m2,
        reservoir.precipitating_hydrometeors_kg_m2, reservoir.fallout_kg_m2,
        diagnosed_water, heating,
        # The circulation operator itself removes the global water/energy
        # source and all longitude structure.  Those components belong to the
        # host atmospheric budget and cannot be solved by a closed meridional
        # circulation.  Compare precisely the anomaly seen by that operator;
        # comparing raw source-minus-fallout would manufacture an impossible
        # nonlinear equation whenever a global-mean boundary flux is present.
        (_cosine_area_balanced_zonal_anomaly(diagnosed_water)
         - _cosine_area_balanced_zonal_anomaly(candidate_water)).astype(np.float32),
        (_cosine_area_balanced_zonal_anomaly(heating)
         - _cosine_area_balanced_zonal_anomaly(candidate_heating)).astype(np.float32),
    )


def converge_joint_pressure_column_coupling(
    evaluate: Callable[[np.ndarray, np.ndarray], JointPressureColumnCouplingResidual],
    initial_heating_w_m2: np.ndarray,
    initial_water_forcing_kg_m2_s: np.ndarray,
) -> JointPressureColumnCouplingSolve:
    """Converge the exact simultaneous candidate map without relaxation.

    ``evaluate`` must return ``diagnosed - candidate`` residuals from
    :func:`diagnose_joint_pressure_column_coupling_residual`.  Each update is
    therefore the full contemporaneously diagnosed state, not a damped blend:

    ``candidate_next = candidate + (diagnosed - candidate)``.

    The convergence criterion is tied only to float32 output precision.  The
    iteration guard scales with the number of independent latitude anomalies
    and raises on nonconvergence; it is not a physical inner timestep or an
    accepted partially converged state.
    """
    heating = np.asarray(initial_heating_w_m2, dtype=np.float64).copy()
    water = np.asarray(initial_water_forcing_kg_m2_s, dtype=np.float64).copy()
    if heating.ndim != 2 or heating.shape[1] != 2 * heating.shape[0] or water.shape != heating.shape:
        raise ValueError("coupling candidates must share a two-dimensional 2:1 grid")
    if not np.all(np.isfinite(heating)) or not np.all(np.isfinite(water)):
        raise ValueError("coupling candidates must be finite")
    # A closed meridional circulation owns only its balanced zonal anomaly.
    # Canonicalising the unknowns removes the unconstrained global-mean/null
    # mode, so the fixed point cannot depend on an arbitrary initial offset.
    heating = _cosine_area_balanced_zonal_anomaly(heating)
    water = _cosine_area_balanced_zonal_anomaly(water)
    precision = np.finfo(np.float32).eps
    maximum_iterations = 8 * heating.shape[0]
    for iteration in range(1, maximum_iterations + 1):
        residual = evaluate(heating, water)
        diagnosed_heating = np.asarray(residual.diagnosed_heating_w_m2, dtype=np.float64)
        diagnosed_water = np.asarray(residual.diagnosed_water_forcing_kg_m2_s, dtype=np.float64)
        heat_error = np.asarray(residual.heating_residual_w_m2, dtype=np.float64)
        water_error = np.asarray(residual.water_forcing_residual_kg_m2_s, dtype=np.float64)
        if not all(np.all(np.isfinite(value)) for value in (
            diagnosed_heating, diagnosed_water, heat_error, water_error,
        )):
            raise RuntimeError("joint pressure-column coupling produced a non-finite residual")
        heat_scale = max(float(np.max(np.abs(heating))), float(np.max(np.abs(diagnosed_heating))), 1.0)
        water_scale = max(float(np.max(np.abs(water))), float(np.max(np.abs(diagnosed_water))), np.finfo(np.float32).tiny)
        if (
            float(np.max(np.abs(heat_error))) <= 64.0 * precision * heat_scale
            and float(np.max(np.abs(water_error))) <= 64.0 * precision * water_scale
        ):
            return JointPressureColumnCouplingSolve(
                heating.astype(np.float32), water.astype(np.float32), residual, iteration,
            )
        # This is exactly the diagnosed simultaneous candidate.  In
        # particular, no under-relaxation, clipping, or physical damping term
        # is inserted between the two states.
        heating = heating + heat_error
        water = water + water_error
    raise RuntimeError(
        "joint pressure-column coupling did not reach float32-consistent "
        f"water/heating closure within {maximum_iterations} exact-map updates"
    )


def evolve_simultaneous_joint_pressure_column_runtime(
    previous_heating_w_m2: np.ndarray,
    surface_source_kg_m2: np.ndarray,
    cloud_condensate_kg_m2: np.ndarray,
    precipitating_hydrometeors_kg_m2: np.ndarray,
    lower_humidity_before_horizontal: np.ndarray,
    midlevel_humidity_before_horizontal: np.ndarray,
    upperlevel_humidity_before_horizontal: np.ndarray,
    lower_humidity_after_horizontal: np.ndarray,
    midlevel_humidity_after_horizontal: np.ndarray,
    upperlevel_humidity_after_horizontal: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    lower_u_m_s: np.ndarray,
    lower_v_m_s: np.ndarray,
    midlevel_u_m_s: np.ndarray,
    midlevel_v_m_s: np.ndarray,
    upperlevel_u_m_s: np.ndarray,
    upperlevel_v_m_s: np.ndarray,
    *,
    lower_vapour_source_kg_kg_day: np.ndarray,
    dt_seconds: float,
    radius_m: float,
    sidereal_day_hours: float,
    surface_pressure_pa: float,
    lower_mid_pressure_depth_pa: float,
    mid_upper_pressure_depth_pa: float,
    dx_m: np.ndarray | float,
    dy_m: float,
    cell_area_m2: np.ndarray | float,
    x_face_length_m: np.ndarray | float,
    y_face_length_m: np.ndarray | float,
    critical_relative_humidity: float,
    cloud_retention_kg_m2: float,
    autoconversion_timescale_days: float,
    fallout_timescale_days: float,
    reservoir_transport_u_m_s: np.ndarray,
    reservoir_transport_v_m_s: np.ndarray,
    cloud_transport_scale: float,
    transport_hydrometeors: bool,
    layer_heights_m: tuple[float, float, float] = (0.0, 3500.0, 8000.0),
    gravity_m_s2: float = 9.80665,
    cp_dry_j_kg_k: float = 1004.0,
    gas_constant_dry_j_kg_k: float = 287.05,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
    max_vertical_courant: float = 0.25,
) -> JointPressureColumnSimultaneousRuntimeStep:
    """Solve phase, reservoirs, water circulation, and heating as one step.

    Horizontal vapour/MSE transport is still supplied through the before/after
    humidity fields, just as for the existing runtime adapter.  In contrast to
    that adapter, this transition owns the contemporaneous cloud/hydrometeor
    mass, fallout, external-water convergence, and large-scale heating state.
    It returns only a fully converged state; no lagged precipitation field or
    partially converged nonlinear iterate can escape to the host.
    """
    previous = np.asarray(previous_heating_w_m2, dtype=np.float64)
    if previous.ndim != 2 or previous.shape[1] != 2 * previous.shape[0]:
        raise ValueError("previous heating must use a two-dimensional 2:1 grid")

    from condensate import transport_pressure_condensate_reservoirs
    transported = transport_pressure_condensate_reservoirs(
        cloud_condensate_kg_m2, precipitating_hydrometeors_kg_m2,
        reservoir_transport_u_m_s, reservoir_transport_v_m_s,
        dt_days=dt_seconds / 86400.0,
        fallout_timescale_days=fallout_timescale_days,
        cloud_transport_scale=cloud_transport_scale,
        transport_hydrometeors=transport_hydrometeors,
        dx_m=dx_m, dy_m=dy_m, cell_area_m2=cell_area_m2,
        x_face_length_m=x_face_length_m, y_face_length_m=y_face_length_m,
    )

    def evaluate(
        candidate_heating: np.ndarray,
        candidate_water: np.ndarray,
    ) -> JointPressureColumnCouplingResidual:
        return diagnose_joint_pressure_column_coupling_residual(
            candidate_heating, candidate_water, previous, surface_source_kg_m2,
            transported.cloud_condensate_kg_m2,
            transported.precipitating_hydrometeors_kg_m2,
            lower_humidity_before_horizontal, midlevel_humidity_before_horizontal,
            upperlevel_humidity_before_horizontal, lower_humidity_after_horizontal,
            midlevel_humidity_after_horizontal, upperlevel_humidity_after_horizontal,
            lower_temperature_k, midlevel_temperature_k, upperlevel_temperature_k,
            lower_u_m_s, lower_v_m_s, midlevel_u_m_s, midlevel_v_m_s,
            upperlevel_u_m_s, upperlevel_v_m_s,
            lower_vapour_source_kg_kg_day=lower_vapour_source_kg_kg_day,
            dt_seconds=dt_seconds, radius_m=radius_m,
            sidereal_day_hours=sidereal_day_hours,
            surface_pressure_pa=surface_pressure_pa,
            lower_mid_pressure_depth_pa=lower_mid_pressure_depth_pa,
            mid_upper_pressure_depth_pa=mid_upper_pressure_depth_pa,
            dx_m=dx_m, dy_m=dy_m, cell_area_m2=cell_area_m2,
            x_face_length_m=x_face_length_m, y_face_length_m=y_face_length_m,
            critical_relative_humidity=critical_relative_humidity,
            cloud_retention_kg_m2=cloud_retention_kg_m2,
            autoconversion_timescale_days=autoconversion_timescale_days,
            fallout_timescale_days=fallout_timescale_days,
            layer_heights_m=layer_heights_m, gravity_m_s2=gravity_m_s2,
            cp_dry_j_kg_k=cp_dry_j_kg_k,
            gas_constant_dry_j_kg_k=gas_constant_dry_j_kg_k,
            layer_mass_fractions=layer_mass_fractions,
            max_vertical_courant=max_vertical_courant,
        )

    solved = converge_joint_pressure_column_coupling(
        evaluate, previous, np.zeros_like(previous),
    )
    residual = solved.residual
    return JointPressureColumnSimultaneousRuntimeStep(
        residual.runtime, residual.cloud_condensate_kg_m2,
        residual.precipitating_hydrometeors_kg_m2, residual.fallout_kg_m2,
        solved.heating_w_m2, solved.water_forcing_kg_m2_s, solved.iterations,
        transported.cloud_relative_residual,
        transported.hydrometeor_relative_residual,
    )


def mse_constrained_pressure_coordinate_circulation(
    large_scale_heating_w_m2: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    lower_humidity: np.ndarray,
    midlevel_humidity: np.ndarray,
    upperlevel_humidity: np.ndarray,
    lower_zonal_wind_m_s: np.ndarray,
    midlevel_zonal_wind_m_s: np.ndarray,
    upperlevel_zonal_wind_m_s: np.ndarray,
    *,
    dt_seconds: float,
    radius_m: float,
    surface_pressure_pa: float,
    lower_mid_pressure_depth_pa: float,
    mid_upper_pressure_depth_pa: float,
    gravity_m_s2: float = 9.80665,
    cp_dry_j_kg_k: float = 1004.0,
    latent_heat_j_kg: float = 2.5e6,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
    layer_heights_m: tuple[float, float, float] = (0.0, 3500.0, 8000.0),
) -> MSEConstrainedPressureCoordinateCirculation:
    """Diagnose a mass-closed overturning from diabatic MSE export.

    The zonal, cosine-area-balanced diabatic forcing fixes the meridional MSE
    transport needed across every latitude circle.  A lower/upper pressure
    branch carries equal and opposite dry-air mass; its resolved signed MSE
    contrast then fixes the branch mass flux. Thus the same calculation determines
    horizontal energy transport, layer divergences, and interface omega.

    The middle branch retains its longitude-mean zonal wind but has no
    divergent meridional component in this first two-branch closure. This is
    a deliberately explicit model choice, not a hidden damping or speed
    scalar. The branch direction follows the resolved signed MSE contrast;
    only a vanishing contrast is inadmissible, because it has no finite
    two-branch mass-flux solution and must not be repaired with a floor.
    """
    if radius_m <= 0.0 or dt_seconds <= 0.0 or surface_pressure_pa <= 0.0:
        raise ValueError("time, radius, and surface pressure must be positive")
    if lower_mid_pressure_depth_pa <= 0.0 or mid_upper_pressure_depth_pa <= 0.0:
        raise ValueError("pressure depths must be positive")
    if gravity_m_s2 <= 0.0 or cp_dry_j_kg_k <= 0.0 or latent_heat_j_kg <= 0.0:
        raise ValueError("physical constants must be positive")
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(
        float(np.sum(fractions)), 1.0, atol=1e-12
    ):
        raise ValueError("layer_mass_fractions must be three positive values summing to one")
    heights = np.asarray(layer_heights_m, dtype=np.float64)
    if heights.shape != (3,) or not np.all(np.isfinite(heights)):
        raise ValueError("layer_heights_m must be three finite values")
    raw = (
        large_scale_heating_w_m2, lower_temperature_k, midlevel_temperature_k,
        upperlevel_temperature_k, lower_humidity, midlevel_humidity,
        upperlevel_humidity, lower_zonal_wind_m_s, midlevel_zonal_wind_m_s,
        upperlevel_zonal_wind_m_s,
    )
    fields = tuple(np.asarray(value, dtype=np.float64) for value in raw)
    shape = fields[0].shape
    if len(shape) != 2 or shape[1] != 2 * shape[0] or any(value.shape != shape for value in fields):
        raise ValueError("MSE-constrained fields must share a two-dimensional 2:1 grid")
    if any(not np.all(np.isfinite(value)) for value in fields):
        raise ValueError("MSE-constrained fields must be finite")
    if any(np.any(value < 0.0) for value in fields[4:7]):
        raise ValueError("MSE-constrained humidities must be non-negative")

    forcing_input, lower_t, middle_t, upper_t, lower_q, middle_q, upper_q, lower_u, middle_u, upper_u = fields
    h, w = shape
    latitude = 0.5 * np.pi - (np.arange(h) + 0.5) * np.pi / h
    cos_lat = np.cos(latitude)
    weights = cos_lat[:, None]
    forcing_zonal = np.mean(forcing_input, axis=1, keepdims=True)
    forcing = forcing_zonal - np.sum(forcing_zonal * weights) / np.sum(weights)
    # Remove only floating-point residue from exact uniform forcing. This is
    # not a physical threshold: its scale tracks machine roundoff of the input
    # magnitude, and preserves every resolved forcing anomaly.
    forcing[np.abs(forcing) <= np.finfo(np.float64).eps * max(float(np.max(np.abs(forcing_input))), 1.0) * 64.0] = 0.0
    forcing = np.broadcast_to(forcing, shape).copy()

    layer_masses = fractions * (float(surface_pressure_pa) / float(gravity_m_s2))
    mse_lower = float(cp_dry_j_kg_k) * lower_t + float(gravity_m_s2) * heights[0] + float(latent_heat_j_kg) * lower_q
    mse_upper = float(cp_dry_j_kg_k) * upper_t + float(gravity_m_s2) * heights[2] + float(latent_heat_j_kg) * upper_q
    contrast = np.mean(mse_lower - mse_upper, axis=1, keepdims=True)
    contrast_roundoff = np.finfo(np.float64).eps * max(
        float(np.max(np.abs(mse_lower))), float(np.max(np.abs(mse_upper))), 1.0
    ) * 64.0
    if np.any(np.abs(contrast) <= contrast_roundoff):
        raise ValueError("lower-minus-upper MSE contrast must not vanish for a two-branch solution")

    # Integrate the zonal-ring energy budget from the North Pole.  `transport`
    # is northward MSE flux per unit latitude-circle length [W m-1]; pole faces
    # are exactly zero, and the returned residual exposes any forcing imbalance.
    edge_latitude = np.linspace(0.5 * np.pi, -0.5 * np.pi, h + 1)
    ring_area = 2.0 * np.pi * float(radius_m) ** 2 * (
        np.sin(edge_latitude[:-1]) - np.sin(edge_latitude[1:])
    )
    forcing_row = forcing[:, 0]
    cumulative_w = -np.concatenate(([0.0], np.cumsum(forcing_row * ring_area)))
    edge_cos = np.cos(edge_latitude)
    transport_face = np.zeros(h + 1, dtype=np.float64)
    transport_face[1:-1] = cumulative_w[1:-1] / (
        2.0 * np.pi * float(radius_m) * edge_cos[1:-1]
    )
    transport_row = 0.5 * (transport_face[:-1] + transport_face[1:])
    transport = np.broadcast_to(transport_row[:, None], shape).copy()

    # `m_lower * v_lower * (h_lower - h_upper)` is the total two-branch
    # northward MSE flux.  The upper velocity follows from mass closure.
    lower_v = transport / (layer_masses[0] * contrast)
    upper_v = -layer_masses[0] / layer_masses[2] * lower_v
    middle_v = np.zeros_like(lower_v)
    lower_v = np.broadcast_to(lower_v, shape).copy()
    middle_v = np.broadcast_to(middle_v, shape).copy()
    upper_v = np.broadcast_to(upper_v, shape).copy()
    lower_u_out = np.broadcast_to(np.mean(lower_u, axis=1, keepdims=True), shape).copy()
    middle_u_out = np.broadcast_to(np.mean(middle_u, axis=1, keepdims=True), shape).copy()
    upper_u_out = np.broadcast_to(np.mean(upper_u, axis=1, keepdims=True), shape).copy()

    lower_divergence = spherical_divergence(lower_u_out, lower_v, radius_m)
    middle_divergence = np.zeros_like(lower_divergence)
    upper_divergence = spherical_divergence(upper_u_out, upper_v, radius_m)
    mass_residual = (
        fractions[0] * lower_divergence
        + fractions[1] * middle_divergence
        + fractions[2] * upper_divergence
    )
    if not np.allclose(mass_residual, 0.0, atol=5e-13):
        raise RuntimeError("MSE-constrained two-branch mass closure failed")
    omega_lower_mid = 0.5 * float(lower_mid_pressure_depth_pa) * (
        lower_divergence - middle_divergence
    )
    omega_mid_upper = 0.5 * float(mid_upper_pressure_depth_pa) * (
        middle_divergence - upper_divergence
    )
    interface = DiabaticInterfaceMassFlux(
        omega_lower_mid.astype(np.float32), omega_mid_upper.astype(np.float32),
        lower_divergence.astype(np.float32), middle_divergence.astype(np.float32),
        upper_divergence.astype(np.float32), forcing.astype(np.float32),
        float(np.max(np.abs(omega_lower_mid)) * dt_seconds / float(lower_mid_pressure_depth_pa)),
        float(np.max(np.abs(omega_mid_upper)) * dt_seconds / float(mid_upper_pressure_depth_pa)),
    )
    circulation = PressureCoordinateCirculation(
        lower_u_out.astype(np.float32), lower_v.astype(np.float32),
        middle_u_out.astype(np.float32), middle_v.astype(np.float32),
        upper_u_out.astype(np.float32), upper_v.astype(np.float32), interface,
    )
    return MSEConstrainedPressureCoordinateCirculation(
        circulation, forcing.astype(np.float32), transport.astype(np.float32),
        float(cumulative_w[-1]), np.broadcast_to(contrast, shape).astype(np.float32),
    )


def three_branch_mse_constrained_pressure_coordinate_circulation(
    large_scale_heating_w_m2: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    lower_humidity: np.ndarray,
    midlevel_humidity: np.ndarray,
    upperlevel_humidity: np.ndarray,
    lower_zonal_wind_m_s: np.ndarray,
    midlevel_zonal_wind_m_s: np.ndarray,
    upperlevel_zonal_wind_m_s: np.ndarray,
    *,
    dt_seconds: float,
    radius_m: float,
    surface_pressure_pa: float,
    lower_mid_pressure_depth_pa: float,
    mid_upper_pressure_depth_pa: float,
    gravity_m_s2: float = 9.80665,
    cp_dry_j_kg_k: float = 1004.0,
    latent_heat_j_kg: float = 2.5e6,
    layer_mass_fractions: tuple[float, float, float] = (0.40, 0.35, 0.25),
    layer_heights_m: tuple[float, float, float] = (0.0, 3500.0, 8000.0),
    zonal_momentum_constraint: bool = False,
    column_water_forcing_kg_m2_s: np.ndarray | None = None,
) -> ThreeBranchMSEConstrainedPressureCoordinateCirculation:
    """Jointly diagnose all three pressure branches from the MSE budget.

    At each latitude, the layer velocities minimize mass-flux norm subject to
    exact zero column mass flux and the MSE transport demanded by the zonal
    diabatic forcing. ``zonal_momentum_constraint=True`` instead uses the
    resolved pressure-level zonal winds as the third constraint; the public
    momentum wrapper below exposes that physically stricter variant.

    Layerwise horizontal MSE export is closed with both diagnosed interface
    energy fluxes. The returned ``*_diabatic_deposition`` fields are therefore
    the heating distribution required for a layerwise steady MSE budget. They
    are diagnostics only here: runtime phase conversion remains the sole heat
    tendency owner until it can consume this decomposition without double use.
    """
    if radius_m <= 0.0 or dt_seconds <= 0.0 or surface_pressure_pa <= 0.0:
        raise ValueError("time, radius, and surface pressure must be positive")
    if lower_mid_pressure_depth_pa <= 0.0 or mid_upper_pressure_depth_pa <= 0.0:
        raise ValueError("pressure depths must be positive")
    if gravity_m_s2 <= 0.0 or cp_dry_j_kg_k <= 0.0 or latent_heat_j_kg <= 0.0:
        raise ValueError("physical constants must be positive")
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0.0) or not np.isclose(
        float(np.sum(fractions)), 1.0, atol=1e-12
    ):
        raise ValueError("layer_mass_fractions must be three positive values summing to one")
    heights = np.asarray(layer_heights_m, dtype=np.float64)
    if heights.shape != (3,) or not np.all(np.isfinite(heights)):
        raise ValueError("layer_heights_m must be three finite values")
    water_forcing = (
        None if column_water_forcing_kg_m2_s is None
        else np.asarray(column_water_forcing_kg_m2_s, dtype=np.float64)
    )
    raw = (
        large_scale_heating_w_m2, lower_temperature_k, midlevel_temperature_k,
        upperlevel_temperature_k, lower_humidity, midlevel_humidity,
        upperlevel_humidity, lower_zonal_wind_m_s, midlevel_zonal_wind_m_s,
        upperlevel_zonal_wind_m_s,
    )
    fields = tuple(np.asarray(value, dtype=np.float64) for value in raw)
    shape = fields[0].shape
    if len(shape) != 2 or shape[1] != 2 * shape[0] or any(value.shape != shape for value in fields):
        raise ValueError("three-branch MSE fields must share a two-dimensional 2:1 grid")
    if any(not np.all(np.isfinite(value)) for value in fields):
        raise ValueError("three-branch MSE fields must be finite")
    if any(np.any(value < 0.0) for value in fields[4:7]):
        raise ValueError("three-branch MSE humidities must be non-negative")
    if water_forcing is not None and (
        water_forcing.shape != shape or not np.all(np.isfinite(water_forcing))
    ):
        raise ValueError("column_water_forcing_kg_m2_s must be finite and match the three-branch grid")
    if zonal_momentum_constraint and water_forcing is not None:
        raise ValueError("choose either a zonal-momentum or column-water third constraint")

    forcing_input, lower_t, middle_t, upper_t, lower_q, middle_q, upper_q, lower_u, middle_u, upper_u = fields
    h, w = shape
    latitude = 0.5 * np.pi - (np.arange(h) + 0.5) * np.pi / h
    weights = np.cos(latitude)[:, None]
    forcing_zonal = np.mean(forcing_input, axis=1, keepdims=True)
    forcing = forcing_zonal - np.sum(forcing_zonal * weights) / np.sum(weights)
    forcing[np.abs(forcing) <= np.finfo(np.float64).eps * max(float(np.max(np.abs(forcing_input))), 1.0) * 64.0] = 0.0
    forcing = np.broadcast_to(forcing, shape).copy()
    layer_masses = fractions * (float(surface_pressure_pa) / float(gravity_m_s2))
    mse = np.stack((
        float(cp_dry_j_kg_k) * lower_t + float(gravity_m_s2) * heights[0] + float(latent_heat_j_kg) * lower_q,
        float(cp_dry_j_kg_k) * middle_t + float(gravity_m_s2) * heights[1] + float(latent_heat_j_kg) * middle_q,
        float(cp_dry_j_kg_k) * upper_t + float(gravity_m_s2) * heights[2] + float(latent_heat_j_kg) * upper_q,
    ))
    mse_zonal = np.mean(mse, axis=2)
    mse_mean = np.sum(layer_masses[:, None] * mse_zonal, axis=0) / np.sum(layer_masses)
    mse_anomaly = mse_zonal - mse_mean
    mse_variance = np.sum(layer_masses[:, None] * mse_anomaly**2, axis=0)
    variance_roundoff = np.finfo(np.float64).eps * max(float(np.max(np.abs(mse_zonal))), 1.0) ** 2 * np.sum(layer_masses) * 64.0
    if np.any(mse_variance <= variance_roundoff):
        raise ValueError("three-branch MSE variance must not vanish")
    layer_u_zonal = np.stack((
        np.mean(lower_u, axis=1), np.mean(middle_u, axis=1), np.mean(upper_u, axis=1),
    ))

    edge_latitude = np.linspace(0.5 * np.pi, -0.5 * np.pi, h + 1)
    ring_area = 2.0 * np.pi * float(radius_m) ** 2 * (
        np.sin(edge_latitude[:-1]) - np.sin(edge_latitude[1:])
    )
    cumulative_w = -np.concatenate(([0.0], np.cumsum(forcing[:, 0] * ring_area)))
    edge_cos = np.cos(edge_latitude)
    transport_face = np.zeros(h + 1, dtype=np.float64)
    transport_face[1:-1] = cumulative_w[1:-1] / (
        2.0 * np.pi * float(radius_m) * edge_cos[1:-1]
    )
    transport_row = 0.5 * (transport_face[:-1] + transport_face[1:])
    constraint_condition_max = 1.0
    if water_forcing is not None:
        water_forcing_zonal = np.mean(water_forcing, axis=1, keepdims=True)
        water_forcing_zonal = water_forcing_zonal - (
            np.sum(water_forcing_zonal * weights) / np.sum(weights)
        )
        water_cumulative_kg_s = -np.concatenate((
            [0.0], np.cumsum(water_forcing_zonal[:, 0] * ring_area),
        ))
        water_transport_face = np.zeros(h + 1, dtype=np.float64)
        water_transport_face[1:-1] = water_cumulative_kg_s[1:-1] / (
            2.0 * np.pi * float(radius_m) * edge_cos[1:-1]
        )
        water_transport_row = 0.5 * (
            water_transport_face[:-1] + water_transport_face[1:]
        )
        # Eliminate the upper branch with exact pressure-mass continuity, then
        # solve the independent MSE and vapour-budget transports for the lower
        # two branch velocities.  This is an explicit conservation constraint,
        # not a tunable branch weight or a fallback to the rejected two-branch
        # solution.
        mse_coefficients = np.stack((
            layer_masses[0] * (mse_zonal[0] - mse_zonal[2]),
            layer_masses[1] * (mse_zonal[1] - mse_zonal[2]),
        ), axis=1)
        water_coefficients = np.stack((
            layer_masses[0] * (np.mean(lower_q, axis=1) - np.mean(upper_q, axis=1)),
            layer_masses[1] * (np.mean(middle_q, axis=1) - np.mean(upper_q, axis=1)),
        ), axis=1)
        constraint_matrix = np.stack((mse_coefficients, water_coefficients), axis=1)
        constraint_condition_max = float(np.max(np.linalg.cond(constraint_matrix)))
        constraint_rank = np.linalg.matrix_rank(constraint_matrix)
        active_transport = (transport_row != 0.0) | (water_transport_row != 0.0)
        if np.any(constraint_rank[active_transport] < 2):
            raise ValueError(
                "MSE and column-water constraints are linearly dependent for an active transport"
            )
        try:
            solved_velocity = np.linalg.solve(
                constraint_matrix,
                np.stack((transport_row, water_transport_row), axis=1)[:, :, None],
            )[:, :, 0]
        except np.linalg.LinAlgError as error:
            raise ValueError("MSE and column-water constraints are linearly dependent") from error
        layer_v_row = np.stack((
            solved_velocity[:, 0], solved_velocity[:, 1],
            -(layer_masses[0] * solved_velocity[:, 0]
              + layer_masses[1] * solved_velocity[:, 1]) / layer_masses[2],
        ))
    elif zonal_momentum_constraint and np.any(transport_row != 0.0):
        # Solve only the two independent physical constraints.  Eliminate the
        # upper branch algebraically from exact pressure-mass continuity:
        #
        #   m0*v0 + m1*v1 + m2*v2 = 0.
        #
        # The prior normalized 3x3 solve treated this identity as a third
        # floating-point equation.  Under a strongly evolved monthly shear it
        # could leave a tiny velocity residual which the discrete spherical
        # divergence amplified enough to trip the strict mass-closure check.
        # Eliminating v2 first makes the mass constraint structural, while the
        # remaining 2x2 system still solves the same MSE and zonal-momentum
        # budgets with no regularization, fallback branch, or strength scalar.
        mse_coefficients = np.stack((
            layer_masses[0] * (mse_zonal[0] - mse_zonal[2]),
            layer_masses[1] * (mse_zonal[1] - mse_zonal[2]),
        ), axis=1)
        momentum_coefficients = np.stack((
            layer_masses[0] * (layer_u_zonal[0] - layer_u_zonal[2]),
            layer_masses[1] * (layer_u_zonal[1] - layer_u_zonal[2]),
        ), axis=1)
        constraint_matrix = np.stack((mse_coefficients, momentum_coefficients), axis=1)
        constraint_condition_max = float(np.max(np.linalg.cond(constraint_matrix)))
        transport_response = np.zeros((3, h), dtype=np.float64)
        active_transport = transport_row != 0.0
        total_mass = float(np.sum(layer_masses))
        u_mean = np.sum(layer_masses[:, None] * layer_u_zonal, axis=0) / total_mass
        u_variance = np.sum(layer_masses[:, None] * (layer_u_zonal - u_mean)**2, axis=0)
        u_roundoff = np.finfo(np.float64).eps * max(
            float(np.max(np.abs(layer_u_zonal))), 1.0
        ) ** 2 * total_mass * 64.0
        if np.any(u_variance[active_transport] <= u_roundoff):
            raise ValueError("vertical zonal-wind variance must not vanish for momentum closure")
        try:
            solved_velocity = np.linalg.solve(
                constraint_matrix[active_transport],
                np.stack((np.ones(np.count_nonzero(active_transport)),
                          np.zeros(np.count_nonzero(active_transport))), axis=1)[:, :, None],
            )[:, :, 0]
        except np.linalg.LinAlgError as error:
            raise ValueError("MSE and zonal-momentum constraints are linearly dependent") from error
        transport_response[0, active_transport] = solved_velocity[:, 0]
        transport_response[1, active_transport] = solved_velocity[:, 1]
        transport_response[2] = -(
            layer_masses[0] * transport_response[0]
            + layer_masses[1] * transport_response[1]
        ) / layer_masses[2]
    else:
        transport_response = mse_anomaly / mse_variance
    if water_forcing is None:
        layer_v_row = transport_row[None, :] * transport_response
    layer_v = np.broadcast_to(layer_v_row[:, :, None], (3, h, w)).copy()
    layer_u = np.stack((
        np.broadcast_to(layer_u_zonal[0, :, None], shape),
        np.broadcast_to(layer_u_zonal[1, :, None], shape),
        np.broadcast_to(layer_u_zonal[2, :, None], shape),
    )).copy()
    divergence = np.stack([
        spherical_divergence(layer_u[index], layer_v[index], radius_m)
        for index in range(3)
    ])
    mass_residual = np.sum(fractions[:, None, None] * divergence, axis=0)
    if not np.allclose(mass_residual, 0.0, atol=5e-13):
        raise RuntimeError(
            "three-branch MSE mass closure failed "
            f"(max residual {float(np.max(np.abs(mass_residual))):.3e} s-1; "
            f"max divergence {float(np.max(np.abs(divergence))):.3e} s-1; "
            f"max branch speed {float(np.max(np.abs(layer_v))):.3e} m s-1; "
            f"max constraint condition {constraint_condition_max:.3e})"
        )
    omega_lower_mid = 0.5 * float(lower_mid_pressure_depth_pa) * (divergence[0] - divergence[1])
    omega_mid_upper = 0.5 * float(mid_upper_pressure_depth_pa) * (divergence[1] - divergence[2])

    # Decompose the exact face transport among layers. The coefficient sums to
    # one at every row, so its finite-volume horizontal divergences sum to the
    # imposed diabatic forcing exactly, not merely in a global mean.
    layer_flux_face = np.zeros((3, h + 1), dtype=np.float64)
    if water_forcing is not None:
        # Cell-centre branch velocities close the interface-mass diagnosis.
        # They cannot, however, be used to *partition* a finite-volume face
        # flux by interpolation: a pair of neighbouring centre transports can
        # cancel exactly at a face even while the integrated forcing requires
        # a nonzero face flux.  Re-solve the same two physical constraints at
        # each face using face-interpolated MSE and vapour.  This is the unique
        # mass-closed face transport and preserves both budgets directly;
        # unlike normalising the raw shares, it has no singular zero-share
        # case and introduces no branch weight.
        mse_face = 0.5 * (mse_zonal[:, :-1] + mse_zonal[:, 1:])
        vapour_zonal = np.stack((
            np.mean(lower_q, axis=1), np.mean(middle_q, axis=1),
            np.mean(upper_q, axis=1),
        ))
        vapour_face = 0.5 * (vapour_zonal[:, :-1] + vapour_zonal[:, 1:])
        mse_face_coefficients = np.stack((
            layer_masses[0] * (mse_face[0] - mse_face[2]),
            layer_masses[1] * (mse_face[1] - mse_face[2]),
        ), axis=1)
        water_face_coefficients = np.stack((
            layer_masses[0] * (vapour_face[0] - vapour_face[2]),
            layer_masses[1] * (vapour_face[1] - vapour_face[2]),
        ), axis=1)
        face_constraint_matrix = np.stack((
            mse_face_coefficients, water_face_coefficients,
        ), axis=1)
        face_rank = np.linalg.matrix_rank(face_constraint_matrix)
        active_face_transport = (
            (transport_face[1:-1] != 0.0)
            | (water_transport_face[1:-1] != 0.0)
        )
        if np.any(face_rank[active_face_transport] < 2):
            raise ValueError(
                "MSE and column-water face constraints are linearly dependent for an active transport"
            )
        try:
            face_velocity = np.linalg.solve(
                face_constraint_matrix,
                np.stack((transport_face[1:-1], water_transport_face[1:-1]), axis=1)[:, :, None],
            )[:, :, 0]
        except np.linalg.LinAlgError as error:
            raise ValueError("MSE and column-water face constraints are linearly dependent") from error
        face_layer_velocity = np.stack((
            face_velocity[:, 0], face_velocity[:, 1],
            -(layer_masses[0] * face_velocity[:, 0]
              + layer_masses[1] * face_velocity[:, 1]) / layer_masses[2],
        ))
        layer_flux_face[:, 1:-1] = layer_masses[:, None] * mse_face * face_layer_velocity
        if not np.allclose(
            np.sum(layer_flux_face[:, 1:-1], axis=0), transport_face[1:-1], atol=2e-10,
        ):
            raise RuntimeError("column-water face MSE closure failed")
    else:
        layer_flux_fraction = layer_masses[:, None] * mse_zonal * transport_response
        layer_flux_fraction_face = 0.5 * (
            layer_flux_fraction[:, :-1] + layer_flux_fraction[:, 1:]
        )
        layer_flux_face[:, 1:-1] = layer_flux_fraction_face * transport_face[None, 1:-1]
    layer_power_face = 2.0 * np.pi * float(radius_m) * edge_cos[None, :] * layer_flux_face
    horizontal_export = (layer_power_face[:, :-1] - layer_power_face[:, 1:]) / ring_area[None, :]
    lower_mid_energy_flux = -omega_lower_mid[:, 0] / float(gravity_m_s2)
    lower_mid_energy_flux *= np.where(lower_mid_energy_flux >= 0.0, mse_zonal[0], mse_zonal[1])
    mid_upper_energy_flux = -omega_mid_upper[:, 0] / float(gravity_m_s2)
    mid_upper_energy_flux *= np.where(mid_upper_energy_flux >= 0.0, mse_zonal[1], mse_zonal[2])
    vertical_convergence = np.stack((
        -lower_mid_energy_flux,
        lower_mid_energy_flux - mid_upper_energy_flux,
        mid_upper_energy_flux,
    ))
    deposition = horizontal_export - vertical_convergence
    layer_energy_residual = np.sum(deposition, axis=0) - forcing[:, 0]
    energy_roundoff = (
        np.finfo(np.float64).eps
        * max(
            float(np.max(np.abs(horizontal_export))),
            float(np.max(np.abs(vertical_convergence))),
            float(np.max(np.abs(forcing))),
            1.0,
        )
        * 256.0
    )
    if not np.allclose(layer_energy_residual, 0.0, atol=energy_roundoff):
        raise RuntimeError(
            "three-branch MSE layer-energy closure failed "
            f"(max residual {float(np.max(np.abs(layer_energy_residual))):.3e} W m-2; "
            f"roundoff allowance {energy_roundoff:.3e} W m-2; "
            f"max forcing {float(np.max(np.abs(forcing))):.3e} W m-2)"
        )

    interface = DiabaticInterfaceMassFlux(
        omega_lower_mid.astype(np.float32), omega_mid_upper.astype(np.float32),
        divergence[0].astype(np.float32), divergence[1].astype(np.float32), divergence[2].astype(np.float32),
        forcing.astype(np.float32),
        float(np.max(np.abs(omega_lower_mid)) * dt_seconds / float(lower_mid_pressure_depth_pa)),
        float(np.max(np.abs(omega_mid_upper)) * dt_seconds / float(mid_upper_pressure_depth_pa)),
    )
    circulation = PressureCoordinateCirculation(
        layer_u[0].astype(np.float32), layer_v[0].astype(np.float32),
        layer_u[1].astype(np.float32), layer_v[1].astype(np.float32),
        layer_u[2].astype(np.float32), layer_v[2].astype(np.float32), interface,
    )
    return ThreeBranchMSEConstrainedPressureCoordinateCirculation(
        circulation, forcing.astype(np.float32),
        np.broadcast_to(transport_row[:, None], shape).astype(np.float32),
        float(cumulative_w[-1]), np.broadcast_to(mse_variance[:, None], shape).astype(np.float32),
        np.broadcast_to(deposition[0, :, None], shape).astype(np.float32),
        np.broadcast_to(deposition[1, :, None], shape).astype(np.float32),
        np.broadcast_to(deposition[2, :, None], shape).astype(np.float32),
    )


def momentum_constrained_three_branch_mse_pressure_coordinate_circulation(
    large_scale_heating_w_m2: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    lower_humidity: np.ndarray,
    midlevel_humidity: np.ndarray,
    upperlevel_humidity: np.ndarray,
    lower_zonal_wind_m_s: np.ndarray,
    midlevel_zonal_wind_m_s: np.ndarray,
    upperlevel_zonal_wind_m_s: np.ndarray,
    **kwargs: object,
) -> ThreeBranchMSEConstrainedPressureCoordinateCirculation:
    """Solve three branches with exact mass, MSE, and zonal momentum budgets.

    At one latitude the planetary component of absolute angular momentum is a
    constant times the column mass flux and therefore vanishes under exact mass
    closure. The remaining condition is zero mass-weighted transport of the
    resolved pressure-level zonal momentum. Its vertical shear supplies the
    missing momentum/thermal-wind branch constraint without a new scalar.
    """
    return three_branch_mse_constrained_pressure_coordinate_circulation(
        large_scale_heating_w_m2, lower_temperature_k, midlevel_temperature_k,
        upperlevel_temperature_k, lower_humidity, midlevel_humidity,
        upperlevel_humidity, lower_zonal_wind_m_s, midlevel_zonal_wind_m_s,
        upperlevel_zonal_wind_m_s, zonal_momentum_constraint=True, **kwargs,
    )


def water_constrained_three_branch_mse_pressure_coordinate_circulation(
    large_scale_heating_w_m2: np.ndarray,
    column_water_forcing_kg_m2_s: np.ndarray,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    lower_humidity: np.ndarray,
    midlevel_humidity: np.ndarray,
    upperlevel_humidity: np.ndarray,
    lower_zonal_wind_m_s: np.ndarray,
    midlevel_zonal_wind_m_s: np.ndarray,
    upperlevel_zonal_wind_m_s: np.ndarray,
    **kwargs: object,
) -> ThreeBranchMSEConstrainedPressureCoordinateCirculation:
    """Solve exact mass, MSE, and column-water transport budgets together.

    The water forcing is the zonal anomaly of the resolved atmospheric column
    source/sink [kg m-2 s-1]. Its meridional integral fixes vapour transport,
    selecting the third three-branch degree of freedom independently of the
    MSE budget and without treating a nearly barotropic zonal-wind profile as
    an externally imposed transport constraint.
    """
    return three_branch_mse_constrained_pressure_coordinate_circulation(
        large_scale_heating_w_m2, lower_temperature_k, midlevel_temperature_k,
        upperlevel_temperature_k, lower_humidity, midlevel_humidity,
        upperlevel_humidity, lower_zonal_wind_m_s, midlevel_zonal_wind_m_s,
        upperlevel_zonal_wind_m_s,
        column_water_forcing_kg_m2_s=column_water_forcing_kg_m2_s, **kwargs,
    )


def shared_pressure_coordinate_circulation(
    precipitation_mm_day: np.ndarray | None,
    lower_temperature_k: np.ndarray,
    midlevel_temperature_k: np.ndarray,
    upperlevel_temperature_k: np.ndarray,
    lower_zonal_wind_m_s: np.ndarray,
    midlevel_zonal_wind_m_s: np.ndarray,
    upperlevel_zonal_wind_m_s: np.ndarray,
    *,
    dt_seconds: float,
    radius_m: float,
    surface_pressure_pa: float,
    lower_mid_pressure_depth_pa: float,
    mid_upper_pressure_depth_pa: float,
    gravity_m_s2: float = 9.80665,
    cp_dry_j_kg_k: float = 1004.0,
    large_scale_heating_w_m2: np.ndarray | None = None,
) -> PressureCoordinateCirculation:
    """Return winds and omega from one pressure-coordinate continuity solve.

    The zonal component keeps only each raw wind's longitude mean, which is a
    non-divergent zonal flow on this grid.  Meridional winds are then the
    minimum-norm fields whose production spherical divergence matches the
    diabatic interface solve exactly.  Thus the horizontal energy carrier and
    the vertical water/energy exchange are no longer two incompatible systems.
    """
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    fields = tuple(np.asarray(value, dtype=np.float64) for value in (
        lower_temperature_k, midlevel_temperature_k, upperlevel_temperature_k,
        lower_zonal_wind_m_s, midlevel_zonal_wind_m_s, upperlevel_zonal_wind_m_s,
    ))
    shape = fields[0].shape
    if any(value.shape != shape for value in fields) or len(shape) != 2 or shape[1] != 2 * shape[0]:
        raise ValueError("all pressure-coordinate fields must share a two-dimensional 2:1 grid")
    if large_scale_heating_w_m2 is None:
        interface = diabatic_interface_mass_flux(
            precipitation_mm_day, fields[0], fields[1], fields[2],
            dt_seconds=dt_seconds, surface_pressure_pa=surface_pressure_pa,
            lower_mid_pressure_depth_pa=lower_mid_pressure_depth_pa,
            mid_upper_pressure_depth_pa=mid_upper_pressure_depth_pa,
            gravity_m_s2=gravity_m_s2, cp_dry_j_kg_k=cp_dry_j_kg_k,
        )
    else:
        interface = diabatic_interface_mass_flux_from_heating(
            large_scale_heating_w_m2, fields[0], fields[1], fields[2],
            dt_seconds=dt_seconds, surface_pressure_pa=surface_pressure_pa,
            lower_mid_pressure_depth_pa=lower_mid_pressure_depth_pa,
            mid_upper_pressure_depth_pa=mid_upper_pressure_depth_pa,
            gravity_m_s2=gravity_m_s2, cp_dry_j_kg_k=cp_dry_j_kg_k,
        )
    latitude = np.radians(90.0 - (np.arange(shape[0]) + 0.5) * 180.0 / shape[0])
    cos_lat = np.cos(latitude)

    def layer(zonal_wind: np.ndarray, divergence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.broadcast_to(np.mean(zonal_wind, axis=1, keepdims=True), shape).copy()
        v_1d = _zonal_mean_meridional_wind(divergence[:, 0], radius_m, cos_lat)
        v = np.broadcast_to(v_1d[:, None], shape).copy()
        return u.astype(np.float32), v.astype(np.float32)

    lower_u, lower_v = layer(fields[3], interface.lower_divergence_s)
    mid_u, mid_v = layer(fields[4], interface.midlevel_divergence_s)
    upper_u, upper_v = layer(fields[5], interface.upperlevel_divergence_s)
    return PressureCoordinateCirculation(
        lower_u, lower_v, mid_u, mid_v, upper_u, upper_v, interface,
    )


def spherical_divergence(u: np.ndarray, v: np.ndarray, radius_m: float) -> np.ndarray:
    """Production-compatible signed divergence on a north-to-south 2:1 grid.

    This is deliberately the unit-mass form of
    :func:`atmosphere.flux_divergence_spherical`: the pressure-column exchange
    must close the same discrete operator that diagnoses its interface omega.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    h, w = u.shape
    lat = np.radians(90.0 - (np.arange(h) + 0.5) * 180.0 / h)
    cos = np.maximum(np.cos(lat), 1.0e-3)[:, None]
    dphi, dlon = np.pi / h, 2.0 * np.pi / w
    du = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dlon)
    vcos = v * cos
    dvcos = np.empty_like(vcos)
    dvcos[1:-1] = 0.5 * (vcos[2:] - vcos[:-2])
    dvcos[0] = vcos[1] - vcos[0]
    dvcos[-1] = vcos[-1] - vcos[-2]
    # Grid rows run north to south, while latitude increases northward.
    dvcos /= -dphi
    return (du + dvcos) / (float(radius_m) * cos)


def smooth_spherical_scalar(
    field: np.ndarray, *, strength: float, passes: int = 1,
) -> np.ndarray:
    """Apply a conservative, scale-selective filter on a regular spherical grid.

    The filter is for diagnosed divergence, where unresolved one- and two-cell
    noise otherwise enters pressure-interface omega directly.  Each pass
    preserves the cosine-area-weighted global mean exactly, so it cannot create
    a net column mass source or sink.
    """
    if passes < 0 or not 0.0 <= strength <= 1.0:
        raise ValueError("passes must be non-negative and strength must lie in [0, 1]")
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 2 * values.shape[0]:
        raise ValueError("field must use a two-dimensional 2:1 global grid")
    h = values.shape[0]
    weights = np.cos(np.radians(90.0 - (np.arange(h) + 0.5) * 180.0 / h))[:, None]
    for _ in range(passes):
        north = np.vstack((values[:1], values[:-1]))
        south = np.vstack((values[1:], values[-1:]))
        neighbour_mean = 0.25 * (np.roll(values, 1, 1) + np.roll(values, -1, 1) + north + south)
        filtered = (1.0 - strength) * values + strength * neighbour_mean
        # Correct roundoff and boundary-stencil asymmetry in the only global
        # moment pressure continuity requires us to retain.
        filtered += (
            np.sum(values * weights) - np.sum(filtered * weights)
        ) / np.sum(np.broadcast_to(weights, values.shape))
        values = filtered
    return values


def balanced_thermal_wind_u(
    lower_u: np.ndarray,
    temperature_k: np.ndarray,
    *,
    radius_m: float,
    sidereal_day_hours: float,
    surface_pressure_pa: float,
    upper_pressure_pa: float,
    hadley_edge_deg: float,
    gas_constant_dry: float = 287.05,
) -> np.ndarray:
    """Return upper-level zonal wind implied by zonal-mean thermal-wind shear.

    The hydrostatic thermal-wind relation gives
    ``u_upper - u_lower = -(R/f) log(p_s/p_u) dT/dy``.  It is diagnosed from
    the zonal mean, which prevents terrain-scale temperature noise from being
    converted into an unphysical free-tropospheric jet.  A smooth equatorial
    taper suppresses the singular geostrophic approximation inside the Hadley
    cell rather than relying on an arbitrary finite ``f`` there.
    """
    lower_u = np.asarray(lower_u, dtype=np.float64)
    temperature_k = np.asarray(temperature_k, dtype=np.float64)
    if lower_u.shape != temperature_k.shape or lower_u.ndim != 2:
        raise ValueError("lower wind and temperature must share one two-dimensional shape")
    if radius_m <= 0.0 or sidereal_day_hours <= 0.0:
        raise ValueError("radius and sidereal day must be positive")
    if not 0.0 < upper_pressure_pa < surface_pressure_pa:
        raise ValueError("upper pressure must lie between zero and surface pressure")
    if hadley_edge_deg <= 0.0:
        raise ValueError("Hadley edge must be positive")
    h, _ = lower_u.shape
    latitude = np.radians(90.0 - (np.arange(h) + 0.5) * 180.0 / h)
    dy_m = float(radius_m) * np.pi / h
    zonal_temperature = np.mean(temperature_k, axis=1)
    dtemp_dy = -np.gradient(zonal_temperature, dy_m, edge_order=1)
    omega = 2.0 * np.pi / (float(sidereal_day_hours) * 3600.0)
    coriolis = 2.0 * omega * np.sin(latitude)
    edge_rad = np.radians(float(hadley_edge_deg))
    f_edge = max(2.0 * omega * np.sin(edge_rad), 1.0e-8)
    # The taper goes rapidly to zero inside the direct circulation but does
    # not impose a discontinuity at its edge.
    taper = 1.0 - np.exp(-((np.abs(latitude) / edge_rad) ** 4))
    f_safe = np.copysign(np.maximum(np.abs(coriolis), f_edge), coriolis)
    shear = (
        -float(gas_constant_dry)
        * np.log(float(surface_pressure_pa) / float(upper_pressure_pa))
        * dtemp_dy
        / f_safe
        * taper
    )
    return (lower_u + shear[:, None]).astype(np.float32)


@lru_cache(maxsize=8)
def _meridional_gradient_pseudoinverse(height: int) -> np.ndarray:
    """Cache the fixed north-to-south derivative stencil's pseudoinverse."""
    if height < 2:
        raise ValueError("at least two latitude rows are required")
    dphi = np.pi / height
    h = height
    gradient = np.zeros((h, h), dtype=np.float64)
    gradient[0, 0], gradient[0, 1] = -1.0 / dphi, 1.0 / dphi
    gradient[-1, -2], gradient[-1, -1] = -1.0 / dphi, 1.0 / dphi
    row = np.arange(1, h - 1)
    gradient[row, row - 1] = -0.5 / dphi
    gradient[row, row + 1] = 0.5 / dphi
    # The constant null mode is immaterial.  The pseudoinverse selects its
    # minimum-norm representative while retaining the exact production stencil.
    return np.linalg.pinv(gradient)


def _zonal_mean_meridional_wind(
    target_divergence_s: np.ndarray, radius_m: float, cos_lat: np.ndarray,
) -> np.ndarray:
    """Return v whose discrete meridional divergence matches a zonal mean target."""
    vcos = _meridional_gradient_pseudoinverse(target_divergence_s.size) @ (
        -target_divergence_s * float(radius_m) * cos_lat
    )
    return vcos / cos_lat


def _equatorial_column_throughflow(
    lower_v: np.ndarray,
    mid_v: np.ndarray,
    upper_v: np.ndarray,
    cos_lat: np.ndarray,
) -> float:
    """Return the column-integrated zonal-mean ``v cos(latitude)`` at the equator.

    A zonally symmetric constant of ``v cos(latitude)`` belongs to the null
    space of the spherical-divergence stencil.  It is invisible to a pressure
    solve but represents an impossible pole-to-pole atmospheric throughflow.
    The equatorial pair is the least singular place to identify that gauge
    mode on PlanetSim's even-height grid.
    """
    column_vcos = np.mean(
        (0.40 * lower_v + 0.35 * mid_v + 0.25 * upper_v) * cos_lat[:, None],
        axis=1,
    )
    h = column_vcos.size
    return float(np.mean(column_vcos[(h // 2 - 1):(h // 2 + 1)]))


def close_upper_mass_flux(
    lower_u: np.ndarray, lower_v: np.ndarray, mid_u: np.ndarray, mid_v: np.ndarray,
    upper_u: np.ndarray, upper_v: np.ndarray, *, radius_m: float,
    strength: float = 1.0, max_speed_m_s: float = 12.0,
    throughflow_max_speed_m_s: float = 80.0,
) -> HorizontalMassClosure:
    """Return a bounded upper-level correction that closes weighted divergence.

    The lower, middle, and upper layers carry weights 0.40, 0.35, and 0.25.
    Zonal Fourier inversion exactly removes every resolved non-zonal divergence
    mode under the centred stencil.  A small least-squares solve handles the
    zonal-mean meridional flux; no arbitrary number of global relaxation
    iterations is required.
    """
    if max_speed_m_s <= 0.0 or throughflow_max_speed_m_s <= 0.0:
        raise ValueError("mass-flux correction speed limits must be positive")
    lower_u = np.asarray(lower_u, dtype=np.float64)
    h, w = lower_u.shape
    if any(np.asarray(field).shape != (h, w) for field in (lower_v, mid_u, mid_v, upper_u, upper_v)):
        raise ValueError("all layer winds must have the same two-dimensional shape")
    dl = spherical_divergence(lower_u, lower_v, radius_m)
    dm = spherical_divergence(mid_u, mid_v, radius_m)
    du = spherical_divergence(upper_u, upper_v, radius_m)
    residual = 0.40 * dl + 0.35 * dm + 0.25 * du
    target = -float(strength) * residual / 0.25

    lat = np.radians(90.0 - (np.arange(h) + 0.5) * 180.0 / h)
    cos = np.cos(lat)
    dlon = 2.0 * np.pi / w
    zonal_mean = np.mean(target, axis=1)
    eddy = target - zonal_mean[:, None]
    mode = np.fft.fftfreq(w, d=1.0 / w)
    eigenvalue = 1j * np.sin(2.0 * np.pi * mode / w) / (dlon * float(radius_m) * cos[:, None])
    correction_u_hat = np.zeros((h, w), dtype=np.complex128)
    usable = np.abs(eigenvalue) > 1.0e-16
    correction_u_hat[usable] = np.fft.fft(eddy, axis=1)[usable] / eigenvalue[usable]
    correction_u = np.fft.ifft(correction_u_hat, axis=1).real
    correction_v = _zonal_mean_meridional_wind(zonal_mean, radius_m, cos)[:, None]
    correction_v = np.broadcast_to(correction_v, (h, w)).copy()

    speed = np.hypot(correction_u, correction_v)
    limiter = np.minimum(1.0, float(max_speed_m_s) / np.maximum(speed, 1e-12))
    correction_u *= limiter
    correction_v *= limiter
    # The divergence inversion cannot determine its zonally symmetric
    # v*cos(latitude) null mode. Remove the column-integrated component with
    # a separately bounded upper-wind correction; otherwise a spurious net
    # pole-to-pole transport can survive despite a locally closed divergence.
    throughflow_before = _equatorial_column_throughflow(
        np.asarray(lower_v, dtype=np.float64), np.asarray(mid_v, dtype=np.float64),
        np.asarray(upper_v, dtype=np.float64) + correction_v, cos,
    )
    throughflow_correction_v = -throughflow_before / (0.25 * np.maximum(cos, 1.0e-3))
    correction_v += np.clip(
        throughflow_correction_v,
        -float(throughflow_max_speed_m_s),
        float(throughflow_max_speed_m_s),
    )[:, None]
    throughflow_after = _equatorial_column_throughflow(
        np.asarray(lower_v, dtype=np.float64), np.asarray(mid_v, dtype=np.float64),
        np.asarray(upper_v, dtype=np.float64) + correction_v, cos,
    )
    after = 0.40 * dl + 0.35 * dm + 0.25 * spherical_divergence(
        np.asarray(upper_u, dtype=np.float64) + correction_u,
        np.asarray(upper_v, dtype=np.float64) + correction_v,
        radius_m,
    )
    return HorizontalMassClosure(
        correction_u.astype(np.float32),
        correction_v.astype(np.float32),
        float(np.sqrt(np.mean(residual**2))),
        float(np.sqrt(np.mean(after**2))),
        throughflow_before,
        throughflow_after,
    )
