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


class BoundaryLayerTransportStep(NamedTuple):
    boundary_temperature: np.ndarray
    free_temperature: np.ndarray
    horizontal_convergence_w_m2: np.ndarray
    continuity_exchange_gain_w_m2: np.ndarray
    substeps: int


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
    exchange_mask: np.ndarray | None = None,
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
    exchange = land if exchange_mask is None else np.asarray(exchange_mask, dtype=bool)
    if boundary.shape != free.shape or boundary.shape != sensible.shape or boundary.shape != land.shape:
        raise ValueError("temperatures, sensible heat, and land mask must share a shape")
    if exchange.shape != boundary.shape:
        raise ValueError("exchange mask must share the temperature shape")
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

    boundary_out = np.where(exchange, boundary_exchanged, boundary_heated)
    free_out = np.where(exchange, free_exchanged, free)
    exchange_gain = np.where(
        exchange,
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


def transport_boundary_layer_energy(
    boundary_temperature_k: np.ndarray,
    free_temperature_k: np.ndarray,
    wind_u_m_s: np.ndarray,
    wind_v_m_s: np.ndarray,
    *,
    pressure_thickness_pa: float,
    surface_pressure_pa: float,
    cp_j_kg_k: float,
    gravity_m_s2: float,
    radius_m: float,
    dt_seconds: float,
) -> BoundaryLayerTransportStep:
    """Flux-form horizontal transport with conservative mass closure.

    Donor-cell face fluxes transport mixed-layer mass and enthalpy on the
    latitude/longitude finite-volume grid.  Because the prescribed host wind
    is divergent while mixed-layer pressure thickness is fixed, each substep
    entrains a local mass deficit from the free atmosphere or detrains a mass
    surplus back to it.  That continuity exchange is equal-and-opposite in
    energy and makes a spatially uniform equilibrium an exact fixed point.
    """
    boundary = np.asarray(boundary_temperature_k, dtype=np.float64)
    free = np.asarray(free_temperature_k, dtype=np.float64)
    u = np.asarray(wind_u_m_s, dtype=np.float64)
    v = np.asarray(wind_v_m_s, dtype=np.float64)
    if boundary.ndim != 2 or free.shape != boundary.shape or u.shape != boundary.shape or v.shape != boundary.shape:
        raise ValueError("transport fields must be 2-D with identical shapes")
    if not all(np.all(np.isfinite(field)) for field in (boundary, free, u, v)):
        raise ValueError("transport fields must be finite")
    scalars = (
        pressure_thickness_pa, surface_pressure_pa, cp_j_kg_k,
        gravity_m_s2, radius_m, dt_seconds,
    )
    if not all(np.isfinite(value) and value > 0.0 for value in scalars):
        raise ValueError("transport thermodynamic and grid inputs must be positive")
    if pressure_thickness_pa >= surface_pressure_pa:
        raise ValueError("boundary-layer pressure thickness must be below surface pressure")

    height, width = boundary.shape
    dlat = np.pi / height
    dlon = 2.0 * np.pi / width
    lat_edges = np.linspace(np.pi / 2.0, -np.pi / 2.0, height + 1)
    area_row = radius_m**2 * dlon * (
        np.sin(lat_edges[:-1]) - np.sin(lat_edges[1:])
    )
    area = np.broadcast_to(area_row[:, None], boundary.shape)
    layer_mass_area = pressure_thickness_pa / gravity_m_s2
    free_mass_area = (surface_pressure_pa - pressure_thickness_pa) / gravity_m_s2
    target_mass = layer_mass_area * area
    target_free_mass = free_mass_area * area

    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    dx = radius_m * dlon * np.maximum(np.cos(lat_centers), 1e-6)
    dy = radius_m * dlat
    cfl = np.max(np.abs(u) * dt_seconds / dx[:, None] + np.abs(v) * dt_seconds / dy)
    substeps = max(1, int(np.ceil(cfl / 0.45)))
    sub_dt = dt_seconds / substeps
    horizontal_energy = np.zeros_like(boundary)
    continuity_energy = np.zeros_like(boundary)

    for _ in range(substeps):
        mass = target_mass.copy()
        heat = target_mass * cp_j_kg_k * boundary
        free_mass = target_free_mass.copy()
        free_heat = target_free_mass * cp_j_kg_k * free

        # East face of every cell; longitude is periodic.
        u_face = 0.5 * (u + np.roll(u, -1, axis=1))
        moved = np.abs(u_face) * sub_dt * layer_mass_area * (radius_m * dlat)
        donor_t = np.where(u_face >= 0.0, boundary, np.roll(boundary, -1, axis=1))
        signed_mass = np.where(u_face >= 0.0, moved, -moved)
        signed_heat = signed_mass * cp_j_kg_k * donor_t
        mass -= signed_mass
        mass += np.roll(signed_mass, 1, axis=1)
        heat -= signed_heat
        heat += np.roll(signed_heat, 1, axis=1)
        free_signed_mass = -signed_mass
        free_donor_t = np.where(
            free_signed_mass >= 0.0, free, np.roll(free, -1, axis=1)
        )
        free_signed_heat = free_signed_mass * cp_j_kg_k * free_donor_t
        free_mass -= free_signed_mass
        free_mass += np.roll(free_signed_mass, 1, axis=1)
        free_heat -= free_signed_heat
        free_heat += np.roll(free_signed_heat, 1, axis=1)

        # South face between adjacent latitude rows. Positive v is northward.
        if height > 1:
            v_face = 0.5 * (v[:-1, :] + v[1:, :])
            face_length = radius_m * dlon * np.cos(lat_edges[1:-1])[:, None]
            moved = np.abs(v_face) * sub_dt * layer_mass_area * face_length
            donor_t = np.where(v_face < 0.0, boundary[:-1, :], boundary[1:, :])
            # Positive signed flux is from the north row to the south row.
            signed_mass = np.where(v_face < 0.0, moved, -moved)
            signed_heat = signed_mass * cp_j_kg_k * donor_t
            mass[:-1, :] -= signed_mass
            mass[1:, :] += signed_mass
            heat[:-1, :] -= signed_heat
            heat[1:, :] += signed_heat
            free_signed_mass = -signed_mass
            free_donor_t = np.where(
                free_signed_mass >= 0.0, free[:-1, :], free[1:, :]
            )
            free_signed_heat = free_signed_mass * cp_j_kg_k * free_donor_t
            free_mass[:-1, :] -= free_signed_mass
            free_mass[1:, :] += free_signed_mass
            free_heat[:-1, :] -= free_signed_heat
            free_heat[1:, :] += free_signed_heat

        horizontal_energy += heat - target_mass * cp_j_kg_k * boundary
        advected_temperature = heat / (mass * cp_j_kg_k)
        advected_free_temperature = free_heat / (free_mass * cp_j_kg_k)
        mass_deficit = target_mass - mass
        # The equal-opposite return flow makes the free-layer mass anomaly the
        # negative of the mixed-layer anomaly. Deficits entrain advected free
        # air; surpluses detrain advected mixed-layer air.
        closure_temperature = np.where(
            mass_deficit >= 0.0, advected_free_temperature, advected_temperature
        )
        closure_heat = mass_deficit * cp_j_kg_k * closure_temperature
        heat += closure_heat
        free_heat -= closure_heat
        continuity_energy += closure_heat
        boundary = heat / (target_mass * cp_j_kg_k)
        free = free_heat / (target_free_mass * cp_j_kg_k)

    return BoundaryLayerTransportStep(
        boundary,
        free,
        horizontal_energy / (area * dt_seconds),
        continuity_energy / (area * dt_seconds),
        substeps,
    )
