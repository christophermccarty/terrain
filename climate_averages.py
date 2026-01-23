"""
Climate averaging and Köppen climate classification module.

This module implements:
- Long-term climate averaging (10-year windows)
- Monthly climate statistics for seasonality detection
- Köppen climate classification with 20 climate types
- Legacy biome classification with hysteresis

Phase 1 of planet simulation improvements - addresses the critical issue
where biomes were changing daily due to instantaneous weather fluctuations.

Updated to support Köppen classification for more realistic climate zones.
"""

import numpy as np
from typing import NamedTuple


# =============================================================================
# Köppen Climate Classification Constants
# =============================================================================

# Köppen climate type codes (0-19)
KOPPEN_OCEAN = 0
KOPPEN_AF = 1   # Tropical Rainforest
KOPPEN_AM = 2   # Tropical Monsoon
KOPPEN_AW = 3   # Tropical Savanna
KOPPEN_BWH = 4  # Hot Desert
KOPPEN_BWK = 5  # Cold Desert
KOPPEN_BSH = 6  # Hot Steppe
KOPPEN_BSK = 7  # Cold Steppe
KOPPEN_CFA = 8  # Humid Subtropical
KOPPEN_CFB = 9  # Oceanic
KOPPEN_CFC = 10 # Subpolar Oceanic
KOPPEN_CSA = 11 # Mediterranean Hot Summer
KOPPEN_CSB = 12 # Mediterranean Warm Summer
KOPPEN_CWA = 13 # Subtropical Monsoon
KOPPEN_DFA = 14 # Hot Continental
KOPPEN_DFB = 15 # Warm Continental
KOPPEN_DFC = 16 # Subarctic
KOPPEN_DWD = 17 # Extreme Continental
KOPPEN_ET = 18  # Tundra
KOPPEN_EF = 19  # Ice Cap

# Köppen type names
KOPPEN_NAMES = {
    0: "Ocean",
    1: "Af - Tropical Rainforest",
    2: "Am - Tropical Monsoon",
    3: "Aw - Tropical Savanna",
    4: "BWh - Hot Desert",
    5: "BWk - Cold Desert",
    6: "BSh - Hot Steppe",
    7: "BSk - Cold Steppe",
    8: "Cfa - Humid Subtropical",
    9: "Cfb - Oceanic",
    10: "Cfc - Subpolar Oceanic",
    11: "Csa - Mediterranean Hot",
    12: "Csb - Mediterranean Warm",
    13: "Cwa - Subtropical Monsoon",
    14: "Dfa - Hot Continental",
    15: "Dfb - Warm Continental",
    16: "Dfc - Subarctic",
    17: "Dwd - Extreme Continental",
    18: "ET - Tundra",
    19: "EF - Ice Cap",
}

# Köppen type colors for visualization (RGB 0-1)
KOPPEN_COLORS = np.array([
    [0.10, 0.20, 0.40],  # 0: Ocean - dark blue
    [0.00, 0.30, 0.00],  # 1: Af - dark green (tropical rainforest)
    [0.00, 0.50, 0.30],  # 2: Am - teal green (tropical monsoon)
    [0.60, 0.80, 0.40],  # 3: Aw - yellow-green (savanna)
    [1.00, 0.50, 0.30],  # 4: BWh - orange-red (hot desert)
    [0.90, 0.70, 0.50],  # 5: BWk - tan (cold desert)
    [1.00, 0.80, 0.50],  # 6: BSh - light orange (hot steppe)
    [0.85, 0.75, 0.55],  # 7: BSk - khaki (cold steppe)
    [0.80, 1.00, 0.40],  # 8: Cfa - bright green (humid subtropical)
    [0.40, 0.80, 0.40],  # 9: Cfb - medium green (oceanic)
    [0.50, 0.70, 0.60],  # 10: Cfc - gray-green (subpolar oceanic)
    [1.00, 1.00, 0.00],  # 11: Csa - yellow (mediterranean hot)
    [0.80, 0.80, 0.00],  # 12: Csb - olive (mediterranean warm)
    [0.60, 0.90, 0.40],  # 13: Cwa - lime (subtropical monsoon)
    [0.40, 1.00, 0.80],  # 14: Dfa - cyan (hot continental)
    [0.30, 0.70, 0.70],  # 15: Dfb - teal (warm continental)
    [0.50, 0.60, 0.70],  # 16: Dfc - steel blue (subarctic)
    [0.60, 0.50, 0.70],  # 17: Dwd - purple-gray (extreme continental)
    [0.75, 0.78, 0.82],  # 18: ET - light gray (tundra)
    [1.00, 1.00, 1.00],  # 19: EF - white (ice cap)
], dtype=np.float32)

# Köppen albedo values (surface reflectivity)
KOPPEN_ALBEDO = {
    0: 0.06,   # Ocean
    1: 0.12,   # Af - Tropical Rainforest (dark canopy)
    2: 0.14,   # Am - Tropical Monsoon
    3: 0.18,   # Aw - Tropical Savanna (mixed)
    4: 0.35,   # BWh - Hot Desert (bright sand)
    5: 0.30,   # BWk - Cold Desert (rocky)
    6: 0.28,   # BSh - Hot Steppe (sparse vegetation)
    7: 0.25,   # BSk - Cold Steppe
    8: 0.15,   # Cfa - Humid Subtropical
    9: 0.16,   # Cfb - Oceanic (deciduous)
    10: 0.18,  # Cfc - Subpolar Oceanic
    11: 0.22,  # Csa - Mediterranean (shrubs)
    12: 0.20,  # Csb - Mediterranean
    13: 0.15,  # Cwa - Subtropical Monsoon
    14: 0.14,  # Dfa - Hot Continental
    15: 0.15,  # Dfb - Warm Continental
    16: 0.18,  # Dfc - Subarctic (boreal forest)
    17: 0.20,  # Dwd - Extreme Continental
    18: 0.25,  # ET - Tundra
    19: 0.80,  # EF - Ice Cap
}

# Mapping from Köppen to legacy 5-biome system for backward compatibility
KOPPEN_TO_LEGACY_BIOME = {
    0: 0,   # Ocean -> Ocean
    1: 3,   # Af -> Forest
    2: 3,   # Am -> Forest
    3: 2,   # Aw -> Grassland
    4: 1,   # BWh -> Desert
    5: 1,   # BWk -> Desert
    6: 1,   # BSh -> Desert
    7: 2,   # BSk -> Grassland
    8: 3,   # Cfa -> Forest
    9: 3,   # Cfb -> Forest
    10: 3,  # Cfc -> Forest
    11: 2,  # Csa -> Grassland
    12: 2,  # Csb -> Grassland
    13: 3,  # Cwa -> Forest
    14: 3,  # Dfa -> Forest
    15: 3,  # Dfb -> Forest
    16: 4,  # Dfc -> Tundra (taiga)
    17: 4,  # Dwd -> Tundra
    18: 4,  # ET -> Tundra
    19: 4,  # EF -> Tundra (ice)
}


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


def update_monthly_statistics(
    state,  # PlanetState
    dt_days: float,
    window_years: float = 1.0,  # 1-year rolling average per month
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update per-month rolling averages for Köppen classification.

    Köppen climate classification requires monthly temperature and precipitation
    data to detect seasonality (e.g., dry summers for Mediterranean climates).
    This function accumulates data into 12 monthly bins using exponential moving
    averages.

    Args:
        state: Current PlanetState with temperature, precipitation, day_of_year
        dt_days: Time step size in days
        window_years: Averaging window in years per month (default 1 year)

    Returns:
        Tuple of (monthly_temp, monthly_precip, monthly_sample_count):
            - monthly_temp: (12, H, W) monthly mean temperature [K]
            - monthly_precip: (12, H, W) monthly mean precipitation [mm/day]
            - monthly_sample_count: (12,) sample count per month
    """
    # Handle uninitialized state
    if state.temperature is None or state.precipitation is None:
        return (None, None, None)

    H, W = state.temperature.shape

    # Determine current month (0-11) from day_of_year
    day_of_year = state.day_of_year if state.day_of_year is not None else 80.0
    month = int(day_of_year / 30.44) % 12

    # Initialize if needed
    if state.monthly_temp is None or state.monthly_precip is None:
        monthly_temp = np.zeros((12, H, W), dtype=np.float32)
        monthly_precip = np.zeros((12, H, W), dtype=np.float32)
        monthly_sample_count = np.zeros(12, dtype=np.float32)
        # Initialize all months with current values for faster spin-up
        for m in range(12):
            monthly_temp[m] = state.temperature.astype(np.float32)
            monthly_precip[m] = state.precipitation.astype(np.float32)
    else:
        monthly_temp = state.monthly_temp.copy()
        monthly_precip = state.monthly_precip.copy()
        monthly_sample_count = state.monthly_sample_count.copy()

    # EMA smoothing factor (~30 samples per year = 1 per day in each month)
    # With window_years=1, we average over ~30 samples per month
    alpha = dt_days / (window_years * 30.44)
    alpha = min(alpha, 0.1)  # Cap for stability

    # Update current month's statistics
    monthly_temp[month] = (1.0 - alpha) * monthly_temp[month] + alpha * state.temperature.astype(np.float32)
    monthly_precip[month] = (1.0 - alpha) * monthly_precip[month] + alpha * state.precipitation.astype(np.float32)
    monthly_sample_count[month] += dt_days

    return monthly_temp, monthly_precip, monthly_sample_count


def classify_koppen(
    monthly_temp: np.ndarray,      # (12, H, W) monthly temps [K]
    monthly_precip: np.ndarray,    # (12, H, W) monthly precip [mm/day]
    land_mask: np.ndarray,         # (H, W) boolean or float
) -> np.ndarray:
    """Classify Köppen climate type for each grid cell.

    Uses the standard Köppen-Geiger climate classification system with
    5 main groups (A, B, C, D, E) and 20 subtypes based on temperature
    and precipitation seasonality.

    Args:
        monthly_temp: (12, H, W) monthly mean temperatures [K]
        monthly_precip: (12, H, W) monthly mean precipitation [mm/day]
        land_mask: (H, W) land mask (1=land, 0=ocean)

    Returns:
        (H, W) int32 array of Köppen codes (0-19)
    """
    H, W = land_mask.shape
    koppen = np.zeros((H, W), dtype=np.int32)

    # Convert to useful derived quantities
    T_celsius = monthly_temp - 273.15  # (12, H, W)
    P_monthly_mm = monthly_precip * 30.44  # mm/day -> mm/month

    # Key temperature metrics
    T_annual_mean = T_celsius.mean(axis=0)  # (H, W)
    T_warmest = T_celsius.max(axis=0)       # Hottest month (Thot)
    T_coldest = T_celsius.min(axis=0)       # Coldest month (Tcold)
    months_above_10 = (T_celsius > 10.0).sum(axis=0)  # Count of months > 10°C

    # Key precipitation metrics
    P_annual = P_monthly_mm.sum(axis=0)     # Total annual precipitation [mm/year]
    P_driest = P_monthly_mm.min(axis=0)     # Driest month [mm]
    P_wettest = P_monthly_mm.max(axis=0)    # Wettest month [mm]

    # Determine summer/winter by hemisphere (latitude)
    # Row 0 = North Pole, Row H-1 = South Pole
    lat_idx = np.arange(H)[:, None] / H  # 0 at top (north), 1 at bottom (south)
    is_southern = lat_idx > 0.5  # Southern hemisphere

    # Summer months: Apr-Sep (months 3-8) for NH, Oct-Mar (months 9-11, 0-2) for SH
    # Winter months: Oct-Mar for NH, Apr-Sep for SH
    summer_mask_nh = np.array([False, False, False, True, True, True, True, True, True, False, False, False])
    summer_mask_sh = ~summer_mask_nh

    # Calculate summer and winter precipitation
    P_summer = np.zeros((H, W), dtype=np.float32)
    P_winter = np.zeros((H, W), dtype=np.float32)
    P_summer_driest = np.full((H, W), np.inf, dtype=np.float32)
    P_winter_driest = np.full((H, W), np.inf, dtype=np.float32)
    P_summer_wettest = np.zeros((H, W), dtype=np.float32)
    P_winter_wettest = np.zeros((H, W), dtype=np.float32)

    for m in range(12):
        P_m = P_monthly_mm[m]
        # Northern hemisphere
        if summer_mask_nh[m]:
            P_summer = np.where(~is_southern, P_summer + P_m, P_summer)
            P_summer_driest = np.where(~is_southern, np.minimum(P_summer_driest, P_m), P_summer_driest)
            P_summer_wettest = np.where(~is_southern, np.maximum(P_summer_wettest, P_m), P_summer_wettest)
        else:
            P_winter = np.where(~is_southern, P_winter + P_m, P_winter)
            P_winter_driest = np.where(~is_southern, np.minimum(P_winter_driest, P_m), P_winter_driest)
            P_winter_wettest = np.where(~is_southern, np.maximum(P_winter_wettest, P_m), P_winter_wettest)
        # Southern hemisphere (reversed seasons)
        if summer_mask_sh[m]:
            P_summer = np.where(is_southern, P_summer + P_m, P_summer)
            P_summer_driest = np.where(is_southern, np.minimum(P_summer_driest, P_m), P_summer_driest)
            P_summer_wettest = np.where(is_southern, np.maximum(P_summer_wettest, P_m), P_summer_wettest)
        else:
            P_winter = np.where(is_southern, P_winter + P_m, P_winter)
            P_winter_driest = np.where(is_southern, np.minimum(P_winter_driest, P_m), P_winter_driest)
            P_winter_wettest = np.where(is_southern, np.maximum(P_winter_wettest, P_m), P_winter_wettest)

    # Land mask as boolean
    land = land_mask > 0.5

    # ==========================================================================
    # B Climate Aridity Threshold (must check before other warm climates)
    # ==========================================================================
    # Threshold depends on precipitation seasonality
    pct_summer = P_summer / (P_annual + 1e-6)

    # R = aridity threshold
    # If >=70% of rain in summer: R = 20*T + 280
    # If 30-70% in summer: R = 20*T + 140
    # If <30% in summer (winter dominant): R = 20*T
    P_threshold = np.where(
        pct_summer >= 0.7,
        20.0 * T_annual_mean + 280.0,
        np.where(
            pct_summer >= 0.3,
            20.0 * T_annual_mean + 140.0,
            20.0 * T_annual_mean
        )
    )

    # ==========================================================================
    # Classification Logic (in priority order)
    # ==========================================================================

    # Ocean cells (code 0)
    koppen[~land] = KOPPEN_OCEAN

    # ----- E Climates (Polar) - check first -----
    # EF: Ice Cap - warmest month < 0°C
    is_EF = land & (T_warmest < 0.0)
    koppen[is_EF] = KOPPEN_EF

    # ET: Tundra - warmest month 0-10°C
    is_ET = land & (T_warmest >= 0.0) & (T_warmest < 10.0)
    koppen[is_ET] = KOPPEN_ET

    # ----- B Climates (Arid) - check second -----
    # Only for cells with warmest month >= 10°C (not polar)
    is_warm_enough = (T_warmest >= 10.0)

    # BW (Desert): P_annual < 0.5 * threshold
    is_BW = land & is_warm_enough & (P_annual < 0.5 * P_threshold)
    # BS (Steppe): 0.5 * threshold <= P_annual < threshold
    is_BS = land & is_warm_enough & (P_annual >= 0.5 * P_threshold) & (P_annual < P_threshold)

    # Subdivide by temperature (h=hot, k=cold based on mean annual temp)
    koppen[is_BW & (T_annual_mean >= 18.0)] = KOPPEN_BWH   # Hot Desert
    koppen[is_BW & (T_annual_mean < 18.0)] = KOPPEN_BWK    # Cold Desert
    koppen[is_BS & (T_annual_mean >= 18.0)] = KOPPEN_BSH   # Hot Steppe
    koppen[is_BS & (T_annual_mean < 18.0)] = KOPPEN_BSK    # Cold Steppe

    is_B = is_BW | is_BS

    # ----- A Climates (Tropical) -----
    # Coldest month >= 18°C, not arid
    is_A = land & (T_coldest >= 18.0) & ~is_B & is_warm_enough

    # Af: Tropical Rainforest - driest month >= 60mm
    is_Af = is_A & (P_driest >= 60.0)
    koppen[is_Af] = KOPPEN_AF

    # Am: Tropical Monsoon - driest month < 60mm but >= (100 - P_annual/25)
    monsoon_threshold = 100.0 - P_annual / 25.0
    is_Am = is_A & (P_driest < 60.0) & (P_driest >= monsoon_threshold)
    koppen[is_Am] = KOPPEN_AM

    # Aw: Tropical Savanna - driest month < monsoon threshold
    is_Aw = is_A & ~is_Af & ~is_Am
    koppen[is_Aw] = KOPPEN_AW

    # ----- C Climates (Temperate/Mesothermal) -----
    # Coldest month > 0°C and < 18°C, warmest month >= 10°C, not arid
    is_C = land & (T_coldest > 0.0) & (T_coldest < 18.0) & (T_warmest >= 10.0) & ~is_B

    # Dry season classification (s=dry summer, w=dry winter, f=no dry season)
    # s: Driest summer month < 40mm AND < 1/3 of wettest winter month
    is_Cs = is_C & (P_summer_driest < 40.0) & (P_summer_driest < P_winter_wettest / 3.0)
    # w: Driest winter month < 1/10 of wettest summer month
    is_Cw = is_C & ~is_Cs & (P_winter_driest < P_summer_wettest / 10.0)
    # f: No dry season
    is_Cf = is_C & ~is_Cs & ~is_Cw

    # Temperature subclassification (a=hot summer, b=warm summer, c=cold summer)
    # a: Warmest month >= 22°C
    # b: Warmest month < 22°C, but >= 4 months >= 10°C
    # c: 1-3 months >= 10°C

    # Cfa, Cfb, Cfc
    koppen[is_Cf & (T_warmest >= 22.0)] = KOPPEN_CFA
    koppen[is_Cf & (T_warmest < 22.0) & (months_above_10 >= 4)] = KOPPEN_CFB
    koppen[is_Cf & (T_warmest < 22.0) & (months_above_10 < 4)] = KOPPEN_CFC

    # Csa, Csb (Mediterranean)
    koppen[is_Cs & (T_warmest >= 22.0)] = KOPPEN_CSA
    koppen[is_Cs & (T_warmest < 22.0)] = KOPPEN_CSB

    # Cwa (Subtropical Monsoon)
    koppen[is_Cw & (T_warmest >= 22.0)] = KOPPEN_CWA
    koppen[is_Cw & (T_warmest < 22.0)] = KOPPEN_CFB  # Treat as Cfb if not hot

    # ----- D Climates (Continental/Microthermal) -----
    # Coldest month <= 0°C (or < -3°C in strict definition), warmest month >= 10°C, not arid
    is_D = land & (T_coldest <= 0.0) & (T_warmest >= 10.0) & ~is_B & ~is_ET & ~is_EF

    # Dry season classification
    is_Ds = is_D & (P_summer_driest < 40.0) & (P_summer_driest < P_winter_wettest / 3.0)
    is_Dw = is_D & ~is_Ds & (P_winter_driest < P_summer_wettest / 10.0)
    is_Df = is_D & ~is_Ds & ~is_Dw

    # Extreme cold (d): Coldest month < -38°C
    is_extreme_cold = T_coldest < -38.0

    # Dfa, Dfb, Dfc, Dfd
    koppen[is_Df & (T_warmest >= 22.0)] = KOPPEN_DFA
    koppen[is_Df & (T_warmest < 22.0) & (months_above_10 >= 4)] = KOPPEN_DFB
    koppen[is_Df & (T_warmest < 22.0) & (months_above_10 < 4) & ~is_extreme_cold] = KOPPEN_DFC
    koppen[is_Df & is_extreme_cold] = KOPPEN_DWD

    # Dsa, Dsb, Dsc (rare - dry summer continental)
    koppen[is_Ds & (T_warmest >= 22.0)] = KOPPEN_DFA  # Treat as Dfa
    koppen[is_Ds & (T_warmest < 22.0)] = KOPPEN_DFB   # Treat as Dfb

    # Dwa, Dwb, Dwc, Dwd (dry winter continental - e.g., Manchuria)
    koppen[is_Dw & (T_warmest >= 22.0)] = KOPPEN_DFA
    koppen[is_Dw & (T_warmest < 22.0) & (months_above_10 >= 4)] = KOPPEN_DFB
    koppen[is_Dw & (T_warmest < 22.0) & (months_above_10 < 4) & ~is_extreme_cold] = KOPPEN_DFC
    koppen[is_Dw & is_extreme_cold] = KOPPEN_DWD

    return koppen


def koppen_to_legacy_biome(koppen_type: np.ndarray) -> np.ndarray:
    """Convert Köppen classification to legacy 5-biome system.

    Args:
        koppen_type: (H, W) Köppen codes (0-19)

    Returns:
        (H, W) legacy biome codes (0-4)
    """
    # Vectorized lookup
    legacy = np.zeros_like(koppen_type, dtype=np.int32)
    for koppen_code, biome_code in KOPPEN_TO_LEGACY_BIOME.items():
        legacy[koppen_type == koppen_code] = biome_code
    return legacy


def get_koppen_name(koppen_code: int) -> str:
    """Get human-readable name for Köppen code."""
    return KOPPEN_NAMES.get(koppen_code, "Unknown")


def get_koppen_albedo(koppen_type: np.ndarray) -> np.ndarray:
    """Get albedo values for Köppen climate types.

    Args:
        koppen_type: (H, W) Köppen codes (0-19)

    Returns:
        (H, W) albedo values
    """
    albedo = np.zeros_like(koppen_type, dtype=np.float32)
    for koppen_code, albedo_val in KOPPEN_ALBEDO.items():
        albedo[koppen_type == koppen_code] = albedo_val
    return albedo


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
