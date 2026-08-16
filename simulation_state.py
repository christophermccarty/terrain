"""Stable simulation state and time-scale data contracts."""
from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import numpy as np

from planet_params import PlanetParams


class TimeScaleMode(Enum):
    """Time integration strategy selected by UI and headless runners."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ANNUAL = "annual"


class PlanetState(NamedTuple):
    """Immutable snapshot exchanged between physics, UI, and persistence."""

    day_of_year: float
    elevation: np.ndarray
    total_days: float = 0.0
    temperature: np.ndarray | None = None
    air_temperature: np.ndarray | None = None
    wind_u: np.ndarray | None = None
    wind_v: np.ndarray | None = None
    precipitation: np.ndarray | None = None
    humidity: np.ndarray | None = None
    soil_moisture: np.ndarray | None = None
    cloud_cover: np.ndarray | None = None
    cloud_water: np.ndarray | None = None
    snow_depth: np.ndarray | None = None
    ice_cover: np.ndarray | None = None
    co2_atmosphere: float = 400.0
    co2_ocean: np.ndarray | None = None
    vegetation_biomass: np.ndarray | None = None
    climate_temp_avg: np.ndarray | None = None
    climate_precip_avg: np.ndarray | None = None
    climate_sample_days: float = 0.0
    biome_type: np.ndarray | None = None
    biome_last_update_day: float = 0.0
    monthly_temp: np.ndarray | None = None
    monthly_precip: np.ndarray | None = None
    monthly_sample_count: np.ndarray | None = None
    koppen_type: np.ndarray | None = None
    ice_sheet_age: np.ndarray | None = None
    salinity: np.ndarray | None = None
    ch4_atmosphere: float = 1900.0
    permafrost_carbon: np.ndarray | None = None
    T_deep_ocean: np.ndarray | None = None
    ice_thickness: np.ndarray | None = None
    jet_index_nh: float = 0.0
    jet_index_sh: float = 0.0
    jet_block_lon_nh: float = -1.0
    jet_block_days_left_nh: float = 0.0
    jet_block_total_days_nh: float = 0.0
    jet_block_lon_sh: float = -1.0
    jet_block_days_left_sh: float = 0.0
    jet_block_total_days_sh: float = 0.0
    wind_u_aloft: np.ndarray | None = None
    wind_v_aloft: np.ndarray | None = None
    planet_params: PlanetParams | None = None
    soil_moisture_deep: np.ndarray | None = None
    wind_speed_avg: np.ndarray | None = None
    surface_water_mm: np.ndarray | None = None
    river_discharge_mm_day: np.ndarray | None = None
    runoff_to_ocean_mm_day: np.ndarray | None = None
    land_ice_thickness: np.ndarray | None = None
    sea_level_change_m: float = 0.0
    atmospheric_condensate: np.ndarray | None = None
    precipitating_hydrometeors: np.ndarray | None = None
    land_deep_temperature: np.ndarray | None = None
    boundary_layer_temperature: np.ndarray | None = None
    boundary_layer_interface_temperature: np.ndarray | None = None
    midlevel_temperature: np.ndarray | None = None
    midlevel_humidity: np.ndarray | None = None
    upperlevel_temperature: np.ndarray | None = None
    grey_optical_depth: np.ndarray | None = None
    upperlevel_humidity: np.ndarray | None = None
    midlevel_wind_u: np.ndarray | None = None
    midlevel_wind_v: np.ndarray | None = None
    omega_lower_mid_pa_s: np.ndarray | None = None
    omega_mid_upper_pa_s: np.ndarray | None = None
    upperlevel_wind_u: np.ndarray | None = None
    upperlevel_wind_v: np.ndarray | None = None
    pressure_moisture_condensation_mm_day: np.ndarray | None = None
    pressure_overturning_heating_w_m2: np.ndarray | None = None
    pressure_coordinate_heat_convergence_w_m2: np.ndarray | None = None
    lower_pressure_depth_pa: np.ndarray | None = None
    midlevel_pressure_depth_pa: np.ndarray | None = None
    upperlevel_pressure_depth_pa: np.ndarray | None = None
    lower_pressure_cloud_condensate: np.ndarray | None = None
    midlevel_pressure_cloud_condensate: np.ndarray | None = None
    upperlevel_pressure_cloud_condensate: np.ndarray | None = None
    lower_pressure_hydrometeors: np.ndarray | None = None
    midlevel_pressure_hydrometeors: np.ndarray | None = None
    upperlevel_pressure_hydrometeors: np.ndarray | None = None
