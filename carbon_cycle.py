"""Carbon cycle physics for PlanetSim.

Implements SimEarth-inspired carbon cycle including:
- Ocean CO2 solubility pump (temperature-dependent)
- Vegetation carbon sequestration (biome-based)
- CO2-temperature greenhouse feedback
- Atmospheric CO2 evolution

Physical basis:
- Henry's Law for ocean CO2 solubility
- Simple NPP model for vegetation uptake
- Logarithmic radiative forcing: ΔF = 5.35 * ln(C/C₀)

CO2 unit conventions
--------------------
``co2_atmosphere``  – global mean atmospheric CO2 [ppm by volume].
``co2_ocean``       – dissolved CO2 as *ppm-equivalent*: the atmospheric
                      concentration that would be in Henry's-Law equilibrium
                      with the local ocean at its temperature.  This is NOT
                      a volumetric ocean concentration but a dimensionless
                      ratio relative to atmospheric ppm.

The conversion from a ppm-equivalent ocean flux to an atmospheric ppm
change uses the known ocean / atmosphere carbon partition ratio and Earth's
atmospheric carbon mass:

    1 ppm CO2 in the atmosphere  ≈ 2.13 GtC   (IPCC AR6)
    Global ocean area            ≈ 3.61 × 10¹⁴ m²
    Ocean mixed-layer depth      ≈ 100 m
    Ocean ppm-eq / atm ratio     ≈ 1 / OCEAN_ATM_RATIO  ≈ 1/50

``OCEAN_ATM_TRANSFER`` is calibrated so that at pre-industrial steady state
the net annual ocean flux is near-zero, and at 415 ppm it removes
~2.5 GtC yr⁻¹ ≈ 1.17 ppm yr⁻¹ – matching IPCC observations.
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Physical conversion constants
# ---------------------------------------------------------------------------
# 1 ppm CO2 in atmosphere = 2.13 GtC  (IPCC AR6 Table 5.1)
PPM_PER_GTC = 1.0 / 2.13               # ppm / GtC

# Fraction of the ocean ppm-equivalent flux that translates to an
# atmospheric ppm change.  Calibrated: at a mean ocean outgassing
# of 1 ppm-eq·m/day across the global ocean, ≈ 2.5 ppm/yr enters
# the atmosphere.  Derived from ocean/atmosphere reservoir ratio:
# C_ocean_dissolved ~ 50 × C_atmosphere at steady state.
OCEAN_ATM_TRANSFER = 3.5e-3            # dimensionless scale factor

# Ocean area fraction for normalization (weighted average over Earth)
OCEAN_AREA_FRACTION = 0.71

# Try to import Numba for acceleration
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    NUMBA_AVAILABLE = False


# ==============================================================================
# Physical Constants
# ==============================================================================

# Preindustrial CO2 concentration [ppm]
CO2_PREINDUSTRIAL = 280.0

# Current (2020) CO2 concentration [ppm]
CO2_CURRENT = 415.0

# Ocean CO2 solubility reference [mol/L at 15°C]
CO2_SOLUBILITY_REF = 3.4e-2

# Temperature coefficient for Henry's Law [1/K]
# K_H(T) = K_H(T_ref) * exp(d_soln_H/R * (1/T - 1/T_ref))
HENRY_TEMP_COEFF = 0.024

# Vegetation carbon density ranges [kg C/m²]
CARBON_FOREST_MAX = 15.0  # Dense tropical forest
CARBON_GRASSLAND_MAX = 3.0  # Temperate grassland
CARBON_DESERT_MIN = 0.1  # Sparse desert vegetation

# Net Primary Productivity (NPP) rates [kg C/m²/year]
NPP_FOREST_MAX = 1.2  # Tropical rainforest
NPP_GRASSLAND_MAX = 0.6  # Temperate grassland
NPP_DESERT_MIN = 0.05  # Sparse desert

# Respiration rate as fraction of biomass [1/year]
RESPIRATION_RATE = 0.08  # ~8% per year

# CO2 fertilization effect (beta factor)
# NPP increases with CO2: NPP(C) = NPP₀ * (1 + beta * ln(C/C₀))
CO2_FERTILIZATION_BETA = 0.6

# Radiative forcing coefficient [W/m² per doubling]
# Standard value from IPCC: ΔF = 5.35 * ln(C/C₀)
RADIATIVE_FORCING_COEFF = 5.35

# Climate sensitivity parameter [K per W/m²]
# Converts radiative forcing to temperature change
# λ = ΔT / ΔF, typical range 0.5-1.2 K/(W/m²)
CLIMATE_SENSITIVITY = 0.8


# ==============================================================================
# Ocean CO2 Solubility (Henry's Law)
# ==============================================================================

def ocean_co2_solubility(temperature: np.ndarray, co2_atm: float) -> np.ndarray:
    """Compute equilibrium dissolved CO2 in ocean using Henry's Law.

    Parameters:
    -----------
    temperature : np.ndarray (H, W)
        Sea surface temperature [K]
    co2_atm : float
        Atmospheric CO2 concentration [ppm]

    Returns:
    --------
    co2_ocean_eq : np.ndarray (H, W)
        Equilibrium dissolved CO2 [ppm equivalent]

    Physics:
    --------
    Henry's Law: C_aq = K_H * p_CO2
    Temperature dependence: K_H(T) ∝ exp(-ΔH/R * (1/T - 1/T_ref))

    Colder water dissolves more CO2 (polar sink).
    Warmer water releases CO2 (tropical source).
    """
    # Reference temperature (288 K = 15°C)
    T_ref = 288.0

    # Temperature-dependent solubility (Henry's Law)
    # Solubility decreases with temperature
    temp_factor = np.exp(-HENRY_TEMP_COEFF * (temperature - T_ref))

    # Equilibrium dissolved CO2 (proportional to atmospheric partial pressure)
    co2_ocean_eq = co2_atm * temp_factor

    return co2_ocean_eq.astype(np.float32)


def ocean_co2_flux(
    co2_ocean: np.ndarray,
    co2_ocean_eq: np.ndarray,
    wind_speed: np.ndarray,
    sea_mask: np.ndarray,
    dt_days: float,
) -> tuple[np.ndarray, float]:
    """Compute air-sea CO2 flux and atmospheric CO2 change.

    Parameters:
    -----------
    co2_ocean : np.ndarray (H, W)
        Current dissolved CO2 in ocean [ppm equivalent]
    co2_ocean_eq : np.ndarray (H, W)
        Equilibrium dissolved CO2 [ppm equivalent]
    wind_speed : np.ndarray (H, W)
        Surface wind speed [m/s]
    sea_mask : np.ndarray (H, W)
        Ocean mask (1 = ocean, 0 = land)
    dt_days : float
        Time step [days]

    Returns:
    --------
    co2_ocean_new : np.ndarray (H, W)
        Updated dissolved CO2 [ppm equivalent]
    d_co2_atm : float
        Change in atmospheric CO2 [ppm]

    Physics:
    --------
    Gas exchange rate proportional to wind speed and (C_eq - C_actual).
    Piston velocity: k = 0.31 * u² (Wanninkhof 1992)
    """
    # Gas transfer velocity (piston velocity) [m/day]
    # k ∝ u² (quadratic wind speed dependence)
    # Clip wind speed to prevent overflow
    wind_speed_safe = np.clip(wind_speed, 0.0, 50.0)  # Max 50 m/s (hurricane-force)
    k_transfer = 0.31 * wind_speed_safe**2 * 24.0  # Convert to per day
    k_transfer = np.clip(k_transfer, 0.0, 100.0)  # Reasonable upper limit

    # CO2 flux from ocean to atmosphere (positive = outgassing)
    # F = k * (C_ocean - C_eq)
    co2_diff = np.clip(co2_ocean - co2_ocean_eq, -1000.0, 1000.0)  # Limit differences
    flux = k_transfer * co2_diff * dt_days

    # Clip flux to prevent extreme values
    flux = np.clip(flux, -500.0, 500.0)  # Limit to ±500 ppm/timestep

    # Apply only over ocean
    flux = flux * sea_mask

    # Update ocean CO2 (relax toward equilibrium)
    co2_ocean_new = np.clip(co2_ocean - flux, 50.0, 5000.0)  # Keep in reasonable range

    # Conservation: CO2 lost from ocean enters atmosphere.
    # Use cos-lat area-weighted mean over ocean cells only, then scale by
    # the calibrated OCEAN_ATM_TRANSFER factor.  This avoids the previous
    # bug of averaging over land (where flux=0), which diluted the signal by
    # the land fraction and then multiplied back by ocean_area_fraction —
    # effectively squaring the ocean fraction — and used a magic * 0.5 factor.
    H, W = co2_ocean.shape
    lat_rad = np.linspace(-np.pi / 2.0, np.pi / 2.0, H)
    cos_lat = np.cos(lat_rad).reshape(-1, 1)          # (H, 1) — broadcast to (H, W)
    ocean_weights = sea_mask * cos_lat                 # area weight, ocean cells only
    total_weight = float(np.sum(ocean_weights)) + 1e-10
    ocean_flux_mean = float(np.sum(flux * ocean_weights) / total_weight)
    d_co2_atm = ocean_flux_mean * OCEAN_ATM_TRANSFER

    return co2_ocean_new.astype(np.float32), float(d_co2_atm)


# ==============================================================================
# Vegetation Carbon Dynamics
# ==============================================================================

def vegetation_albedo(biome: np.ndarray, base_land_albedo: float = 0.2,
                      koppen_type: np.ndarray | None = None) -> np.ndarray:
    """Compute surface albedo based on vegetation/biome type.

    Parameters:
    -----------
    biome : np.ndarray (H, W)
        Legacy biome type index (0-4) for backward compatibility
    base_land_albedo : float
        Base albedo for bare land (default: 0.2)
    koppen_type : np.ndarray (H, W), optional
        Köppen climate classification (0-19). If provided, uses Köppen-based
        albedo values which are more detailed than legacy biomes.

    Returns:
    --------
    albedo : np.ndarray (H, W)
        Surface albedo [0-1]

    Legacy biome albedo values (used when koppen_type is None):
    -----------------------------------------------------------
    - Ocean/Ice (0): 0.06 (handled separately with ice fraction)
    - Desert (1): 0.30 (bright sand/rock)
    - Grassland (2): 0.20 (moderate)
    - Forest (3): 0.12 (dark canopy absorbs more solar)
    - Tundra (4): 0.25 (lichens, sparse vegetation)

    Köppen albedo values (used when koppen_type is provided):
    ---------------------------------------------------------
    - Af/Am (Tropical rainforest/monsoon): 0.12-0.14 (dark canopy)
    - Aw (Tropical savanna): 0.18 (mixed)
    - BWh/BWk (Desert): 0.30-0.35 (bright sand/rock)
    - BSh/BSk (Steppe): 0.25-0.28 (sparse vegetation)
    - Cfa-Cwa (Temperate): 0.15-0.22 (forest/shrub)
    - Dfa-Dfc (Continental): 0.14-0.18 (boreal forest)
    - ET (Tundra): 0.25
    - EF (Ice Cap): 0.80

    Physics:
    --------
    Forests have lower albedo than grasslands/deserts due to:
    1. Dark canopy structure (multiple scattering layers)
    2. Shadows between trees
    3. Lower reflectance of chlorophyll-rich leaves

    This creates a positive feedback: forests warm their environment
    (lower albedo → more absorbed solar → warmer).
    """
    # Use Köppen-based albedo if available
    if koppen_type is not None:
        from climate_averages import get_koppen_albedo
        return get_koppen_albedo(koppen_type)

    # Legacy biome-based albedo
    albedo = np.full_like(biome, base_land_albedo, dtype=np.float32)

    # Biome-specific albedo
    albedo[biome == 0] = 0.06  # Ocean (fallback, usually masked)
    albedo[biome == 1] = 0.30  # Desert (bright)
    albedo[biome == 2] = 0.20  # Grassland (moderate)
    albedo[biome == 3] = 0.12  # Forest (dark)
    albedo[biome == 4] = 0.25  # Tundra (sparse vegetation)

    return albedo


def compute_biome_type(
    temperature: np.ndarray,
    precipitation: np.ndarray,
    land_mask: np.ndarray,
) -> np.ndarray:
    """Classify biomes based on temperature and precipitation.

    Parameters:
    -----------
    temperature : np.ndarray (H, W)
        Surface temperature [K]
    precipitation : np.ndarray (H, W)
        Precipitation rate [mm/day]
    land_mask : np.ndarray (H, W)
        Land mask (1 = land, 0 = ocean)

    Returns:
    --------
    biome : np.ndarray (H, W)
        Biome type index:
        0 = Ocean/Ice
        1 = Desert (<250 mm/year precip)
        2 = Grassland (250-1000 mm/year)
        3 = Forest (>1000 mm/year, T > 273K)
        4 = Tundra (T < 273K)

    Note:
    -----
    This is a simplified Whittaker biome classification.
    Real biomes depend on seasonality, soil, etc.
    """
    T_celsius = temperature - 273.15
    P_annual = precipitation * 365.0  # mm/year

    # Initialize as ocean/ice
    biome = np.zeros_like(temperature, dtype=np.int32)

    # Land biomes
    land = land_mask > 0.5

    # Tundra: cold lands
    tundra = land & (T_celsius < 0)
    biome[tundra] = 4

    # Desert: dry lands
    desert = land & (T_celsius >= 0) & (P_annual < 250)
    biome[desert] = 1

    # Grassland: moderate precipitation
    grassland = land & (T_celsius >= 0) & (P_annual >= 250) & (P_annual < 1000)
    biome[grassland] = 2

    # Forest: high precipitation and warm
    forest = land & (T_celsius >= 0) & (P_annual >= 1000)
    biome[forest] = 3

    return biome


def vegetation_npp(
    biome: np.ndarray,
    temperature: np.ndarray,
    precipitation: np.ndarray,
    co2_atm: float,
) -> np.ndarray:
    """Compute Net Primary Productivity (NPP) for vegetation.

    Parameters:
    -----------
    biome : np.ndarray (H, W)
        Biome type index (from compute_biome_type)
    temperature : np.ndarray (H, W)
        Surface temperature [K]
    precipitation : np.ndarray (H, W)
        Precipitation rate [mm/day]
    co2_atm : float
        Atmospheric CO2 concentration [ppm]

    Returns:
    --------
    npp : np.ndarray (H, W)
        Net Primary Productivity [kg C/m²/day]

    Physics:
    --------
    NPP depends on:
    1. Biome type (potential productivity)
    2. Temperature (growing season warmth)
    3. Water availability (precipitation)
    4. CO2 fertilization effect
    """
    H, W = biome.shape
    npp = np.zeros((H, W), dtype=np.float32)

    # CO2 fertilization factor: NPP increases with CO2
    # β-factor model: NPP(C) = NPP₀ * (1 + β * ln(C/C₀))
    # Ensure CO2 is positive before taking log
    co2_safe = max(co2_atm, 100.0)  # Minimum 100 ppm to avoid log issues
    co2_factor = 1.0 + CO2_FERTILIZATION_BETA * np.log(co2_safe / CO2_PREINDUSTRIAL)
    co2_factor = np.clip(co2_factor, 0.5, 2.0)  # Reasonable bounds

    # Temperature factor (optimal around 20-25°C)
    T_celsius = temperature - 273.15
    temp_factor = np.clip((T_celsius + 10) / 30.0, 0.0, 1.0)

    # Water factor (from precipitation)
    P_annual = precipitation * 365.0
    water_factor = np.clip(P_annual / 1000.0, 0.0, 1.0)

    # Biome-specific NPP
    # Desert (biome=1)
    npp[biome == 1] = NPP_DESERT_MIN

    # Grassland (biome=2)
    npp[biome == 2] = NPP_GRASSLAND_MAX * temp_factor[biome == 2] * water_factor[biome == 2]

    # Forest (biome=3)
    npp[biome == 3] = NPP_FOREST_MAX * temp_factor[biome == 3] * water_factor[biome == 3]

    # Tundra (biome=4)
    npp[biome == 4] = NPP_DESERT_MIN * 0.5

    # Apply CO2 fertilization
    npp = npp * co2_factor

    # Convert from annual to daily
    npp = npp / 365.0

    return npp


def vegetation_carbon_balance(
    biomass: np.ndarray,
    npp: np.ndarray,
    temperature: np.ndarray,
    dt_days: float,
) -> tuple[np.ndarray, float]:
    """Update vegetation biomass and compute CO2 flux.

    Parameters:
    -----------
    biomass : np.ndarray (H, W)
        Current vegetation biomass [kg C/m²]
    npp : np.ndarray (H, W)
        Net Primary Productivity [kg C/m²/day]
    temperature : np.ndarray (H, W)
        Surface temperature [K]
    dt_days : float
        Time step [days]

    Returns:
    --------
    biomass_new : np.ndarray (H, W)
        Updated biomass [kg C/m²]
    d_co2_atm : float
        Net CO2 flux to atmosphere [ppm]
        (negative = uptake, positive = release)

    Physics:
    --------
    dC/dt = NPP - Respiration
    Respiration increases with temperature (Q10 = 2)
    """
    # Respiration rate (temperature-dependent)
    # Q10 rule: rate doubles every 10°C
    T_celsius = np.clip(temperature - 273.15, -50.0, 50.0)  # Reasonable temperature range
    T_ref = 15.0  # Reference temperature
    Q10 = 2.0
    # Clip exponent to prevent overflow
    temp_diff = np.clip((T_celsius - T_ref) / 10.0, -5.0, 5.0)
    respiration_factor = np.clip(Q10 ** temp_diff, 0.01, 100.0)
    respiration_rate = RESPIRATION_RATE * respiration_factor / 365.0  # Convert to daily

    # Carbon balance
    respiration = biomass * respiration_rate
    # Clip NPP and respiration to prevent overflow
    npp_safe = np.clip(npp, 0.0, 10.0)  # Max 10 kg C/m²/day
    respiration_safe = np.clip(respiration, 0.0, 10.0)
    d_biomass = (npp_safe - respiration_safe) * dt_days

    # Update biomass (with bounds)
    biomass_new = np.clip(biomass + d_biomass, 0.0, CARBON_FOREST_MAX)

    # Net CO2 flux (negative = uptake by vegetation)
    net_flux = -(npp_safe - respiration_safe)

    # Global mean flux (land area weighted)
    # Convert kg C/m²/day to ppm change
    # Rough conversion: 1 Gt C ≈ 0.47 ppm
    # Global land area ≈ 1.5e8 km²
    land_fraction = np.sum(biomass > 0) / biomass.size
    mean_flux = np.mean(net_flux[biomass > 0]) if np.any(biomass > 0) else 0.0
    d_co2_atm = mean_flux * land_fraction * dt_days * 0.001  # Scaling factor

    return biomass_new.astype(np.float32), float(d_co2_atm)


def wildfire_dynamics(
    biomass: np.ndarray,
    temperature: np.ndarray,
    precipitation: np.ndarray,
    soil_moisture: np.ndarray | None,
    dt_days: float,
    fire_threshold_temp: float = 298.15,  # 25°C
    fire_threshold_dryness: float = 0.3,   # Soil moisture < 0.3
    fire_consumption_rate: float = 0.2,    # Fraction of biomass burned
) -> tuple[np.ndarray, float]:
    """Simulate wildfire dynamics with CO2 release.

    Parameters:
    -----------
    biomass : np.ndarray (H, W)
        Current vegetation biomass [kg C/m²]
    temperature : np.ndarray (H, W)
        Surface temperature [K]
    precipitation : np.ndarray (H, W)
        Precipitation rate [mm/day]
    soil_moisture : np.ndarray (H, W) | None
        Soil moisture fraction [0-1]
    dt_days : float
        Time step [days]
    fire_threshold_temp : float
        Minimum temperature for fire risk [K] (default: 298K = 25°C)
    fire_threshold_dryness : float
        Maximum soil moisture for fire risk [0-1] (default: 0.3)
    fire_consumption_rate : float
        Fraction of biomass consumed per fire event (default: 0.2 = 20%)

    Returns:
    --------
    biomass_new : np.ndarray (H, W)
        Biomass after fires [kg C/m²]
    co2_released : float
        CO2 released to atmosphere [ppm]

    Physics:
    --------
    Wildfire probability depends on:
    1. Fuel availability (biomass > threshold)
    2. Temperature (hot = higher ignition risk)
    3. Dryness (low soil moisture = higher flammability)

    Fires consume vegetation and release CO2, creating a positive
    feedback during warming/drying events.

    SimEarth inspiration: Fires are a natural disturbance that can
    dramatically alter climate by releasing stored carbon.
    """
    # Fire risk factors
    T_celsius = temperature - 273.15

    # Hot temperature increases fire risk
    temp_risk = np.clip((T_celsius - (fire_threshold_temp - 273.15)) / 20.0, 0.0, 1.0)

    # Dry conditions increase fire risk
    if soil_moisture is not None:
        dryness_risk = np.clip((fire_threshold_dryness - soil_moisture) / fire_threshold_dryness, 0.0, 1.0)
    else:
        # Fallback: use inverse of precipitation as dryness proxy
        P_normalized = np.clip(precipitation / 5.0, 0.0, 1.0)  # 5 mm/day = wet
        dryness_risk = 1.0 - P_normalized

    # Fuel availability (need biomass to burn)
    fuel_available = np.clip(biomass / 5.0, 0.0, 1.0)  # Normalize: 5 kg C/m² = full fuel

    # Combined fire risk (all factors must be present)
    fire_risk = temp_risk * dryness_risk * fuel_available

    # Fire probability per day (scaled by timestep)
    # Base probability: 0.01/day at maximum risk (1% chance per day)
    fire_prob_per_day = 0.01 * fire_risk
    fire_prob = np.clip(fire_prob_per_day * dt_days, 0.0, 0.5)  # Cap at 50% per timestep

    # Stochastic fire occurrence (use deterministic approximation for stability)
    # In a real implementation, could use np.random.random() < fire_prob
    # For now, use fire_prob as fractional burn area
    burn_fraction = fire_prob * fire_consumption_rate

    # Biomass consumed by fire
    biomass_burned = biomass * burn_fraction
    biomass_new = biomass - biomass_burned

    # CO2 released (all burned carbon goes to atmosphere)
    # Convert kg C/m² to ppm (rough approximation)
    land_fraction = np.sum(biomass > 0) / biomass.size if np.sum(biomass > 0) > 0 else 0.0
    mean_burned = np.mean(biomass_burned[biomass > 0]) if np.any(biomass > 0) else 0.0
    co2_released = mean_burned * land_fraction * dt_days * 0.01  # Scaling factor

    return biomass_new.astype(np.float32), float(co2_released)


# ==============================================================================
# CO2-Temperature Feedback
# ==============================================================================

def co2_radiative_forcing(co2_current: float, co2_reference: float = CO2_PREINDUSTRIAL) -> float:
    """Compute radiative forcing from CO2 change.

    Parameters:
    -----------
    co2_current : float
        Current CO2 concentration [ppm]
    co2_reference : float
        Reference CO2 concentration [ppm] (default: preindustrial 280 ppm)

    Returns:
    --------
    forcing : float
        Radiative forcing [W/m²]

    Physics:
    --------
    IPCC formula: ΔF = 5.35 * ln(C/C₀)
    Doubling CO2 (280→560 ppm) gives ΔF ≈ 3.7 W/m²
    """
    # Ensure both values are positive before taking log
    co2_current_safe = max(co2_current, 100.0)  # Minimum 100 ppm
    co2_reference_safe = max(co2_reference, 100.0)
    forcing = RADIATIVE_FORCING_COEFF * np.log(co2_current_safe / co2_reference_safe)
    return float(forcing)


def co2_temperature_response(forcing: float, climate_sensitivity: float = CLIMATE_SENSITIVITY) -> float:
    """Compute equilibrium temperature change from radiative forcing.

    Parameters:
    -----------
    forcing : float
        Radiative forcing [W/m²]
    climate_sensitivity : float
        Climate sensitivity parameter [K per W/m²]
        (default: 0.8 K/(W/m²), gives ECS ≈ 3K for 2×CO2)

    Returns:
    --------
    dT : float
        Equilibrium temperature change [K]

    Physics:
    --------
    ΔT = λ * ΔF
    where λ is climate sensitivity parameter

    Equilibrium Climate Sensitivity (ECS) for CO2 doubling:
    ECS = λ * 3.7 W/m² ≈ 3K (IPCC best estimate: 2.5-4K)
    """
    dT = climate_sensitivity * forcing
    return float(dT)


# ==============================================================================
# Main Carbon Cycle Step
# ==============================================================================

def carbon_cycle_step(
    state,  # PlanetState
    dt_days: float,
) -> tuple:
    """Evolve carbon cycle for one time step.

    Parameters:
    -----------
    state : PlanetState
        Current simulation state
    dt_days : float
        Time step [days]

    Returns:
    --------
    co2_atm_new : float
        Updated atmospheric CO2 [ppm]
    co2_ocean_new : np.ndarray (H, W)
        Updated ocean dissolved CO2 [ppm equivalent]
    biomass_new : np.ndarray (H, W)
        Updated vegetation biomass [kg C/m²]
    co2_forcing : float
        Current radiative forcing from CO2 [W/m²]

    Notes:
    ------
    This implements a simple box model:
    - Atmosphere (well-mixed, single value)
    - Ocean (spatially varying, temperature-dependent solubility)
    - Vegetation (spatially varying, climate-dependent growth)
    """
    # Get current state
    co2_atm = state.co2_atmosphere
    co2_ocean = state.co2_ocean if state.co2_ocean is not None else None
    biomass = state.vegetation_biomass if state.vegetation_biomass is not None else None
    temperature = state.temperature
    precipitation = state.precipitation if state.precipitation is not None else np.ones_like(temperature) * 3.0

    # Initialize if needed
    if co2_ocean is None:
        # Start at equilibrium with preindustrial CO2
        co2_ocean = ocean_co2_solubility(temperature, CO2_PREINDUSTRIAL)

    if biomass is None:
        # Initialize biomass based on biome
        land_mask = (state.elevation > 0.02).astype(np.float32)
        biome = compute_biome_type(temperature, precipitation, land_mask)
        biomass = np.where(biome == 3, CARBON_FOREST_MAX * 0.7, 0.0)  # 70% of max for forests
        biomass = np.where(biome == 2, CARBON_GRASSLAND_MAX * 0.5, biomass)  # 50% for grassland
        biomass = biomass.astype(np.float32)

    # Ocean CO2 exchange
    wind_speed = np.sqrt(state.wind_u**2 + state.wind_v**2) if state.wind_u is not None else np.ones_like(temperature) * 5.0
    sea_mask = (state.elevation <= 0.02).astype(np.float32)
    co2_ocean_eq = ocean_co2_solubility(temperature, co2_atm)
    co2_ocean_new, d_co2_ocean = ocean_co2_flux(co2_ocean, co2_ocean_eq, wind_speed, sea_mask, dt_days)

    # Vegetation carbon dynamics
    land_mask = (state.elevation > 0.02).astype(np.float32)
    biome = compute_biome_type(temperature, precipitation, land_mask)
    npp = vegetation_npp(biome, temperature, precipitation, co2_atm)
    biomass_new, d_co2_veg = vegetation_carbon_balance(biomass, npp, temperature, dt_days)

    # Wildfire dynamics (Phase 4: Biosphere)
    soil_moisture = state.soil_moisture if hasattr(state, 'soil_moisture') and state.soil_moisture is not None else None
    biomass_new, co2_from_fire = wildfire_dynamics(
        biomass_new, temperature, precipitation, soil_moisture, dt_days
    )

    # Update atmospheric CO2 (vegetation uptake + ocean exchange + fires)
    co2_atm_new = co2_atm + d_co2_ocean + d_co2_veg + co2_from_fire

    # Ensure CO2 stays within physically reasonable bounds
    co2_atm_new = np.clip(co2_atm_new, 100.0, 10000.0)  # 100-10000 ppm range

    # Compute radiative forcing
    co2_forcing = co2_radiative_forcing(co2_atm_new, CO2_PREINDUSTRIAL)

    return co2_atm_new, co2_ocean_new, biomass_new, co2_forcing
