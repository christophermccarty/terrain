"""
Climate averaging and stable biome classification module.

This module implements long-term climate averaging (10-year windows) and
biome classification with hysteresis to prevent daily oscillations.

Phase 1 of planet simulation improvements - addresses the critical issue
where biomes were changing daily due to instantaneous weather fluctuations.
"""

import numpy as np
from typing import NamedTuple


def update_climate_averages(
    state,  # PlanetState
    dt_days: float,
    window_days: float = 3650.0,  # 10 years × 365 days
) -> tuple[np.ndarray, np.ndarray, float]:
    """Update exponential moving average for climate variables.

    Uses the formula: MA_new = (1-α)*MA_old + α*X_current
    where α = dt / window is the smoothing factor.

    This approach provides a running average that gradually incorporates
    new data while maintaining memory of past conditions - essential for
    identifying long-term climate patterns vs. short-term weather.

    Args:
        state: Current PlanetState with temperature, precipitation fields
        dt_days: Time step size in days
        window_days: Averaging window period (default 10 years = 3650 days)

    Returns:
        Tuple of (temp_avg, precip_avg, updated_sample_count):
            - temp_avg: (H,W) 10-year average temperature [K]
            - precip_avg: (H,W) 10-year average precipitation [mm/day]
            - updated_sample_count: Total days in average
    """
    alpha = dt_days / window_days  # Smoothing factor

    # Handle initialization: if temperature/precip not yet initialized, return None
    if state.temperature is None or state.precipitation is None:
        return (None, None, 0.0)

    if state.climate_temp_avg is None or state.climate_precip_avg is None:
        # Initialize from current state (first timestep with valid data)
        return (
            state.temperature.copy(),
            state.precipitation.copy(),
            dt_days
        )

    # Update exponential moving averages
    temp_avg = (1.0 - alpha) * state.climate_temp_avg + alpha * state.temperature
    precip_avg = (1.0 - alpha) * state.climate_precip_avg + alpha * state.precipitation

    return temp_avg, precip_avg, state.climate_sample_days + dt_days


def compute_stable_biomes(
    temp_avg: np.ndarray,
    precip_avg: np.ndarray,
    land_mask: np.ndarray,
    prev_biome: np.ndarray | None = None,
    hysteresis_temp: float = 2.0,       # ±2K temperature buffer
    hysteresis_precip: float = 100.0,   # ±100 mm/yr precipitation buffer
) -> np.ndarray:
    """Classify biomes with hysteresis to prevent boundary oscillations.

    Uses a modified Whittaker biome diagram approach based on long-term
    climate averages (not instantaneous weather). Hysteresis prevents
    flip-flopping at classification boundaries.

    Standard Whittaker thresholds:
        - Ocean/Ice (0): land_mask ≤ 0.5
        - Tundra (4): T < 0°C
        - Desert (1): T ≥ 0°C, P < 250 mm/yr
        - Grassland (2): T ≥ 0°C, 250 ≤ P < 1000 mm/yr
        - Forest (3): T ≥ 0°C, P ≥ 1000 mm/yr

    Hysteresis mechanism:
        Once in biome X, you need to cross threshold ± buffer to switch.
        Examples:
            - Forest (P=1050) → needs P < 900 to become grassland (not just <1000)
            - Grassland (P=950) → needs P > 1100 to become forest (not just >1000)
            - Tundra (T=-1°C) → needs T > +2°C to switch (not just >0°C)

        This prevents oscillation where a location at exactly 0°C or 1000 mm/yr
        would flip categories every timestep due to small fluctuations.

    Args:
        temp_avg: (H,W) long-term average temperature [K]
        precip_avg: (H,W) long-term average precipitation [mm/day]
        land_mask: (H,W) 1.0=land, 0.0=ocean
        prev_biome: (H,W) previous biome classification (None for first call)
        hysteresis_temp: Temperature hysteresis buffer [K]
        hysteresis_precip: Precipitation hysteresis buffer [mm/yr]

    Returns:
        (H,W) int32 array of biome codes:
            0 = Ocean/Ice
            1 = Desert
            2 = Grassland
            3 = Forest
            4 = Tundra
    """
    H, W = temp_avg.shape

    # Convert to classification-friendly units
    T_celsius = temp_avg - 273.15  # Kelvin → Celsius
    P_annual = precip_avg * 365.0  # mm/day → mm/year

    # Initialize biome array
    biome = np.zeros((H, W), dtype=np.int32)

    # Land mask (ocean = 0)
    land = land_mask > 0.5

    if prev_biome is None:
        # First classification: use standard thresholds without hysteresis
        _classify_biomes_standard(biome, T_celsius, P_annual, land)
    else:
        # With hysteresis: harder to leave current biome
        _classify_biomes_with_hysteresis(
            biome, T_celsius, P_annual, land, prev_biome,
            hysteresis_temp, hysteresis_precip
        )

    return biome


def _classify_biomes_standard(
    biome: np.ndarray,
    T_celsius: np.ndarray,
    P_annual: np.ndarray,
    land: np.ndarray
) -> None:
    """Standard Whittaker classification (no hysteresis).

    Modifies biome array in-place.
    """
    # Priority order: Tundra → Desert → Grassland → Forest
    # (earlier conditions take precedence)

    # Tundra: Cold climates (below freezing average)
    tundra_mask = land & (T_celsius < 0.0)
    biome[tundra_mask] = 4

    # Desert: Warm but very dry
    desert_mask = land & (T_celsius >= 0.0) & (P_annual < 250.0)
    biome[desert_mask] = 1

    # Grassland: Moderate precipitation
    grassland_mask = land & (T_celsius >= 0.0) & (P_annual >= 250.0) & (P_annual < 1000.0)
    biome[grassland_mask] = 2

    # Forest: High precipitation
    forest_mask = land & (T_celsius >= 0.0) & (P_annual >= 1000.0)
    biome[forest_mask] = 3


def _classify_biomes_with_hysteresis(
    biome: np.ndarray,
    T_celsius: np.ndarray,
    P_annual: np.ndarray,
    land: np.ndarray,
    prev_biome: np.ndarray,
    h_temp: float,
    h_precip: float
) -> None:
    """Whittaker classification with hysteresis (prevents oscillation).

    Modifies biome array in-place.
    """
    # Start with previous biome classification
    biome[:] = prev_biome

    # Define transition masks with hysteresis buffers

    # ===== TUNDRA TRANSITIONS (0°C ± h_temp) =====
    # Tundra → Other: need T > +h_temp to leave tundra
    leaving_tundra = (prev_biome == 4) & (T_celsius > h_temp) & land

    # Other → Tundra: need T < -h_temp to enter tundra
    entering_tundra = (prev_biome != 4) & (T_celsius < -h_temp) & land

    # Apply tundra transitions first (high priority)
    biome[entering_tundra] = 4

    # For cells leaving tundra, reclassify based on precipitation
    warm_land = leaving_tundra & (T_celsius >= 0.0)
    biome[warm_land & (P_annual < 250.0)] = 1  # Desert
    biome[warm_land & (P_annual >= 250.0) & (P_annual < 1000.0)] = 2  # Grassland
    biome[warm_land & (P_annual >= 1000.0)] = 3  # Forest

    # ===== DESERT ↔ GRASSLAND TRANSITIONS (250 mm/yr ± h_precip) =====
    # Desert → Grassland: need P > 250 + h_precip
    desert_to_grassland = (prev_biome == 1) & (P_annual > 250.0 + h_precip) & land & (T_celsius >= 0.0)

    # Grassland → Desert: need P < 250 - h_precip
    grassland_to_desert = (prev_biome == 2) & (P_annual < 250.0 - h_precip) & land & (T_celsius >= 0.0)

    biome[desert_to_grassland] = 2
    biome[grassland_to_desert] = 1

    # ===== GRASSLAND ↔ FOREST TRANSITIONS (1000 mm/yr ± h_precip) =====
    # Grassland → Forest: need P > 1000 + h_precip
    grassland_to_forest = (prev_biome == 2) & (P_annual > 1000.0 + h_precip) & land & (T_celsius >= 0.0)

    # Forest → Grassland: need P < 1000 - h_precip
    forest_to_grassland = (prev_biome == 3) & (P_annual < 1000.0 - h_precip) & land & (T_celsius >= 0.0)

    biome[grassland_to_forest] = 3
    biome[forest_to_grassland] = 2

    # ===== DESERT ↔ FOREST (direct transition, rare but possible) =====
    # Desert → Forest: need P > 1000 + h_precip (skip grassland if very wet)
    desert_to_forest = (prev_biome == 1) & (P_annual > 1000.0 + h_precip) & land & (T_celsius >= 0.0)

    # Forest → Desert: need P < 250 - h_precip (skip grassland if very dry)
    forest_to_desert = (prev_biome == 3) & (P_annual < 250.0 - h_precip) & land & (T_celsius >= 0.0)

    biome[desert_to_forest] = 3
    biome[forest_to_desert] = 1

    # Ocean cells always stay ocean (0)
    biome[~land] = 0


def get_biome_name(biome_code: int) -> str:
    """Convert biome code to human-readable name.

    Args:
        biome_code: Integer biome code (0-4)

    Returns:
        Biome name string
    """
    names = {
        0: "Ocean/Ice",
        1: "Desert",
        2: "Grassland",
        3: "Forest",
        4: "Tundra"
    }
    return names.get(biome_code, "Unknown")


def get_biome_statistics(biome: np.ndarray, land_mask: np.ndarray) -> dict:
    """Calculate biome coverage statistics.

    Args:
        biome: (H,W) biome classification array
        land_mask: (H,W) land mask (1=land, 0=ocean)

    Returns:
        Dictionary with biome fractions and counts
    """
    land = land_mask > 0.5
    total_land_cells = np.sum(land)

    if total_land_cells == 0:
        return {name: 0.0 for name in ["desert", "grassland", "forest", "tundra"]}

    # Count each biome type on land
    desert_count = np.sum((biome == 1) & land)
    grassland_count = np.sum((biome == 2) & land)
    forest_count = np.sum((biome == 3) & land)
    tundra_count = np.sum((biome == 4) & land)

    return {
        "desert_fraction": desert_count / total_land_cells,
        "grassland_fraction": grassland_count / total_land_cells,
        "forest_fraction": forest_count / total_land_cells,
        "tundra_fraction": tundra_count / total_land_cells,
        "desert_cells": int(desert_count),
        "grassland_cells": int(grassland_count),
        "forest_cells": int(forest_count),
        "tundra_cells": int(tundra_count),
        "total_land_cells": int(total_land_cells),
    }
