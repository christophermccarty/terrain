"""Pure dry atmospheric mixed-layer energy kernels.

The boundary layer is a hydrostatic slab of geometric depth ``h``.  Its
pressure thickness is diagnosed from an isothermal hydrostatic atmosphere,

    delta_p = p_s * (1 - exp(-g h / (R_d T_ref))),

and its heat capacity per unit area is ``(delta_p / g) * cp``.  Entrainment
with the free atmosphere is represented by a mass flux ``rho_ref * w_e``.
The resulting two-reservoir exchange is integrated analytically, making it
equal-and-opposite and stable for any positive timestep.

Horizontal transport is deliberately absent: the host model transports the
free-atmosphere temperature, while this local mixed layer receives surface
sensible heat and exchanges energy with that transported reservoir.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class BoundaryLayerStep(NamedTuple):
    boundary_temperature: np.ndarray
    free_temperature: np.ndarray
    surface_gain_w_m2: np.ndarray
    exchange_gain_w_m2: np.ndarray
    pressure_thickness_pa: float
    effective_entrainment_velocity_m_s: np.ndarray
    bulk_richardson_number: np.ndarray


def mixed_layer_pressure_thickness(
    *,
    surface_pressure_pa: float,
    gravity_m_s2: float,
    gas_constant_j_kg_k: float,
    reference_temperature_k: float,
    mixed_layer_depth_m: float,
) -> float:
    """Return hydrostatic pressure thickness of an isothermal mixed layer."""
    values = (
        surface_pressure_pa,
        gravity_m_s2,
        gas_constant_j_kg_k,
        reference_temperature_k,
        mixed_layer_depth_m,
    )
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("mixed-layer thermodynamic inputs must be finite and positive")
    exponent = -gravity_m_s2 * mixed_layer_depth_m / (
        gas_constant_j_kg_k * reference_temperature_k
    )
    return float(surface_pressure_pa * -np.expm1(exponent))


def step_boundary_layer_energy(
    boundary_temperature_k: np.ndarray,
    free_temperature_k: np.ndarray,
    upward_surface_sensible_w_m2: np.ndarray,
    land_mask: np.ndarray,
    *,
    surface_pressure_pa: float,
    cp_j_kg_k: float,
    gravity_m_s2: float,
    gas_constant_j_kg_k: float,
    reference_temperature_k: float,
    mixed_layer_depth_m: float,
    entrainment_velocity_m_s: float,
    dt_seconds: float,
    wind_speed_m_s: np.ndarray | None = None,
    stability_dependent_exchange: bool = False,
) -> BoundaryLayerStep:
    """Apply surface heating and conservative mixed-layer entrainment.

    Positive sensible heat is upward: it is already a loss in the land
    surface kernel and is therefore a gain here.  Exchange gain is positive
    into the boundary layer and negative into the free atmosphere.
    """
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be finite and positive")
    if not np.isfinite(cp_j_kg_k) or cp_j_kg_k <= 0.0:
        raise ValueError("cp_j_kg_k must be finite and positive")
    if not np.isfinite(entrainment_velocity_m_s) or entrainment_velocity_m_s < 0.0:
        raise ValueError("entrainment_velocity_m_s must be finite and non-negative")

    boundary = np.asarray(boundary_temperature_k, dtype=np.float64)
    free = np.asarray(free_temperature_k, dtype=np.float64)
    sensible = np.asarray(upward_surface_sensible_w_m2, dtype=np.float64)
    land = np.asarray(land_mask, dtype=bool)
    if boundary.shape != free.shape or boundary.shape != sensible.shape or boundary.shape != land.shape:
        raise ValueError("temperatures, sensible heat, and land mask must share a shape")
    if not np.all(np.isfinite(boundary)) or not np.all(np.isfinite(free)) or not np.all(np.isfinite(sensible)):
        raise ValueError("boundary-layer energy inputs must be finite")

    delta_p = mixed_layer_pressure_thickness(
        surface_pressure_pa=surface_pressure_pa,
        gravity_m_s2=gravity_m_s2,
        gas_constant_j_kg_k=gas_constant_j_kg_k,
        reference_temperature_k=reference_temperature_k,
        mixed_layer_depth_m=mixed_layer_depth_m,
    )
    if delta_p >= surface_pressure_pa:
        raise ValueError("mixed layer must be thinner than the atmospheric column")
    boundary_capacity = delta_p * cp_j_kg_k / gravity_m_s2
    free_capacity = (surface_pressure_pa - delta_p) * cp_j_kg_k / gravity_m_s2
    surface_gain = np.where(land, sensible, 0.0)
    boundary_heated = boundary + surface_gain * dt_seconds / boundary_capacity

    # A first-order bulk Richardson closure prevents warm free air from being
    # mixed downward too rapidly through a stable nocturnal/winter inversion.
    # Ri_b = g h (T_free - T_bl) / (T_ref U^2).  The mechanical mixing limit
    # follows the stable Businger--Dyer form kappa*u_star/(1 + 5 Ri_b), with
    # u_star diagnosed from the neutral logarithmic wind law over representative
    # open-land roughness z0=0.1 m.  The specified entrainment velocity remains
    # the neutral/convective upper bound; no fitted relaxation time is added.
    bulk_ri = np.zeros_like(boundary_heated)
    effective_velocity = np.full_like(boundary_heated, entrainment_velocity_m_s)
    if stability_dependent_exchange:
        if wind_speed_m_s is None:
            raise ValueError("wind_speed_m_s is required for stability-dependent exchange")
        wind = np.asarray(wind_speed_m_s, dtype=np.float64)
        if wind.shape != boundary.shape or not np.all(np.isfinite(wind)):
            raise ValueError("wind speed must be finite and share the temperature shape")
        shear_squared = np.maximum(wind * wind, 0.25**2)
        bulk_ri = (
            gravity_m_s2 * mixed_layer_depth_m * (free - boundary_heated)
            / (reference_temperature_k * shear_squared)
        )
        von_karman = 0.4
        roughness_m = 0.1
        friction_velocity = von_karman * np.maximum(wind, 0.25) / np.log(
            mixed_layer_depth_m / roughness_m
        )
        mechanical_limit = (
            von_karman * friction_velocity
            / (1.0 + 5.0 * np.maximum(bulk_ri, 0.0))
        )
        effective_velocity = np.minimum(effective_velocity, mechanical_limit)

    # Isothermal reference density and entrainment heat conductance [W m-2 K-1].
    rho_ref = surface_pressure_pa / (gas_constant_j_kg_k * reference_temperature_k)
    conductance = rho_ref * cp_j_kg_k * effective_velocity
    decay = np.exp(
        -conductance * (1.0 / boundary_capacity + 1.0 / free_capacity) * dt_seconds
    )
    equilibrium = (
        boundary_capacity * boundary_heated + free_capacity * free
    ) / (boundary_capacity + free_capacity)
    boundary_exchanged = equilibrium + (boundary_heated - equilibrium) * decay
    free_exchanged = equilibrium + (free - equilibrium) * decay

    boundary_out = np.where(land, boundary_exchanged, boundary)
    free_out = np.where(land, free_exchanged, free)
    exchange_gain = np.where(
        land,
        boundary_capacity * (boundary_exchanged - boundary_heated) / dt_seconds,
        0.0,
    )
    return BoundaryLayerStep(
        boundary_out,
        free_out,
        surface_gain,
        exchange_gain,
        delta_p,
        effective_velocity,
        bulk_ri,
    )
