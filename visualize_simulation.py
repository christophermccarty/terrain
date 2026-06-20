"""Visualization tool for simulation data exports.

Run this script to create plots from exported CSV/JSON data.
Requires: matplotlib, pandas (install with: pip install matplotlib pandas)
"""

import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Any

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    print("Error: matplotlib and pandas are required.")
    print("Install with: pip install matplotlib pandas")
    sys.exit(1)


def load_data(filename: str) -> List[Dict[str, Any]]:
    """Load simulation data from CSV or JSON."""
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    
    if filename.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)


def convert_to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convert specified columns to numeric, handling JSON strings."""
    for col in cols:
        if col in df.columns:
            # Try direct conversion first
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def plot_temperature_evolution(data: List[Dict[str, Any]]):
    """Plot temperature evolution over time."""
    df = pd.DataFrame(data)
    
    # Convert string values to float
    numeric_cols = ['day_of_year', 'total_days', 'global_mean_temp', 'T_equator', 
                     'T_pole_north', 'T_pole_south', 'T_min', 'T_max', 'T_std',
                     'gradient_north', 'gradient_south']
    df = convert_to_numeric(df, numeric_cols)
    
    # Use total_days if available, otherwise day_of_year
    time_col = 'total_days' if 'total_days' in df.columns else 'day_of_year'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Temperature Evolution Over Time', fontsize=16)
    
    # Global mean temperature
    ax = axes[0, 0]
    ax.plot(df[time_col], df['global_mean_temp'] - 273.15, 'b-', label='Global Mean')
    ax.axhline(y=15.0, color='r', linestyle='--', label='Earth Reference (15°C)')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Global Mean Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Equator vs Poles
    ax = axes[0, 1]
    ax.plot(df[time_col], df['T_equator'] - 273.15, 'r-', label='Equator')
    ax.plot(df[time_col], df['T_pole_north'] - 273.15, 'b-', label='North Pole')
    ax.plot(df[time_col], df['T_pole_south'] - 273.15, 'g-', label='South Pole')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature by Latitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temperature range
    ax = axes[1, 0]
    ax.plot(df[time_col], df['T_max'] - 273.15, 'r-', label='Max')
    ax.plot(df[time_col], df['T_min'] - 273.15, 'b-', label='Min')
    ax.fill_between(df[time_col], 
                     df['T_min'] - 273.15, 
                     df['T_max'] - 273.15, 
                     alpha=0.2)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Equator-Pole gradient
    ax = axes[1, 1]
    if 'gradient_north' in df.columns and not df['gradient_north'].isna().all():
        ax.plot(df[time_col], df['gradient_north'], 'b-', label='North')
    if 'gradient_south' in df.columns and not df['gradient_south'].isna().all():
        ax.plot(df[time_col], df['gradient_south'], 'g-', label='South')
    ax.axhline(y=45.0, color='r', linestyle='--', label='Earth Reference (~45K)')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Temperature Gradient (K)')
    ax.set_title('Equator-Pole Temperature Gradient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_latitude_profile(data: List[Dict[str, Any]], day_of_year: float = None):
    """Plot temperature profile by latitude for a specific day."""
    # Check if we have total_days or day_of_year
    has_total_days = any('total_days' in d for d in data)
    
    if day_of_year is None:
        # Use latest day
        if has_total_days:
            latest = max(data, key=lambda x: float(x.get('total_days', 0)))
            day_of_year = float(latest.get('day_of_year', 0))
        else:
            latest = max(data, key=lambda x: float(x.get('day_of_year', 0)))
            day_of_year = float(latest['day_of_year'])
    
    # Find closest day
    if has_total_days:
        # Match by day_of_year but prefer entries with total_days
        closest = min(data, key=lambda x: abs(float(x.get('day_of_year', 0)) - day_of_year))
    else:
        closest = min(data, key=lambda x: abs(float(x.get('day_of_year', 0)) - day_of_year))
    
    if 'latitudes' not in closest or 'zonal_means' not in closest:
        print("Error: Latitude profile data not available in this export.")
        return None
    
    # Parse JSON strings if needed (from CSV export)
    # Handle both string (from CSV) and list/array (from JSON) formats
    try:
        if isinstance(closest['latitudes'], str):
            latitudes = json.loads(closest['latitudes'])
        elif isinstance(closest['latitudes'], list):
            latitudes = closest['latitudes']
        else:
            latitudes = list(closest['latitudes'])
    except (json.JSONDecodeError, TypeError):
        print(f"Warning: Could not parse latitudes data. Using empty list.")
        return None
    
    try:
        if isinstance(closest['zonal_means'], str):
            zonal_means = json.loads(closest['zonal_means'])
        elif isinstance(closest['zonal_means'], list):
            zonal_means = closest['zonal_means']
        else:
            zonal_means = list(closest['zonal_means'])
    except (json.JSONDecodeError, TypeError):
        print(f"Warning: Could not parse zonal_means data. Using empty list.")
        return None
    
    if len(latitudes) == 0 or len(zonal_means) == 0:
        print("Error: Empty latitude profile data.")
        return None
    
    # Convert latitudes from radians to degrees if needed
    import math
    if latitudes and abs(latitudes[0]) > 2:  # Likely already in degrees
        lat_deg = latitudes
    else:  # Likely in radians
        lat_deg = [math.degrees(lat) for lat in latitudes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lat_deg, [t - 273.15 for t in zonal_means], 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Latitude (degrees)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'Temperature Profile by Latitude (Day {day_of_year:.0f})')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-90, 90)
    
    return fig


def plot_precipitation_evolution(data: List[Dict[str, Any]]):
    """Plot precipitation evolution over time."""
    df = pd.DataFrame(data)
    numeric_cols = ['day_of_year', 'total_days', 'mean_precip', 'precip_max', 'precip_min',
                     'precip_equator', 'precip_pole_north', 'precip_pole_south']
    df = convert_to_numeric(df, numeric_cols)
    time_col = 'total_days' if 'total_days' in df.columns else 'day_of_year'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Precipitation Evolution Over Time', fontsize=16)
    
    # Global mean precipitation
    ax = axes[0, 0]
    if 'mean_precip' in df.columns:
        ax.plot(df[time_col], df['mean_precip'], 'b-', label='Global Mean')
        ax.axhline(y=2.7, color='r', linestyle='--', label='Earth Reference (2.7 mm/day)')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Precipitation (mm/day)')
    ax.set_title('Global Mean Precipitation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precipitation by latitude
    ax = axes[0, 1]
    if 'precip_equator' in df.columns:
        ax.plot(df[time_col], df['precip_equator'], 'r-', label='Equator')
    if 'precip_pole_north' in df.columns:
        ax.plot(df[time_col], df['precip_pole_north'], 'b-', label='North Pole')
    if 'precip_pole_south' in df.columns:
        ax.plot(df[time_col], df['precip_pole_south'], 'g-', label='South Pole')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Precipitation (mm/day)')
    ax.set_title('Precipitation by Latitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precipitation range
    ax = axes[1, 0]
    if 'precip_max' in df.columns and 'precip_min' in df.columns:
        ax.plot(df[time_col], df['precip_max'], 'r-', label='Max')
        ax.plot(df[time_col], df['precip_min'], 'b-', label='Min')
        ax.fill_between(df[time_col], df['precip_min'], df['precip_max'], alpha=0.2)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Precipitation (mm/day)')
    ax.set_title('Precipitation Range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precipitation std
    ax = axes[1, 1]
    if 'precip_std' in df.columns:
        ax.plot(df[time_col], df['precip_std'], 'g-', label='Std Dev')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Precipitation Std Dev (mm/day)')
    ax.set_title('Precipitation Variability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_wind_clouds_humidity(data: List[Dict[str, Any]]):
    """Plot wind, cloud, and humidity evolution."""
    df = pd.DataFrame(data)
    numeric_cols = ['day_of_year', 'total_days', 'wind_mean', 'wind_max', 'wind_u_zonal_mean', 'wind_v_merid_mean',
                     'cloud_mean', 'cloud_max', 'cloud_min', 'cloud_equator', 'cloud_pole_north', 'cloud_pole_south',
                     'humidity_mean', 'humidity_max', 'humidity_min', 'albedo_mean']
    df = convert_to_numeric(df, numeric_cols)
    time_col = 'total_days' if 'total_days' in df.columns else 'day_of_year'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Wind, Clouds, and Humidity Evolution', fontsize=16)
    
    # Wind speed
    ax = axes[0, 0]
    if 'wind_mean' in df.columns:
        ax.plot(df[time_col], df['wind_mean'], 'b-', label='Mean Wind Speed')
    if 'wind_max' in df.columns:
        ax.plot(df[time_col], df['wind_max'], 'r--', label='Max Wind Speed', alpha=0.7)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.set_title('Wind Speed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Wind components
    ax = axes[0, 1]
    if 'wind_u_zonal_mean' in df.columns:
        ax.plot(df[time_col], df['wind_u_zonal_mean'], 'b-', label='Zonal (E-W)')
    if 'wind_v_merid_mean' in df.columns:
        ax.plot(df[time_col], df['wind_v_merid_mean'], 'g-', label='Meridional (N-S)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Wind Component (m/s)')
    ax.set_title('Wind Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cloud cover
    ax = axes[1, 0]
    if 'cloud_mean' in df.columns:
        ax.plot(df[time_col], df['cloud_mean'], 'b-', label='Global Mean')
    if 'cloud_equator' in df.columns:
        ax.plot(df[time_col], df['cloud_equator'], 'r-', label='Equator', alpha=0.7)
    if 'cloud_pole_north' in df.columns:
        ax.plot(df[time_col], df['cloud_pole_north'], 'g-', label='North Pole', alpha=0.7)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Cloud Cover Fraction')
    ax.set_title('Cloud Cover')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Humidity and Albedo
    ax = axes[1, 1]
    ax2 = ax.twinx()
    if 'humidity_mean' in df.columns:
        ax.plot(df[time_col], df['humidity_mean'], 'b-', label='Humidity Mean')
    if 'albedo_mean' in df.columns:
        ax2.plot(df[time_col], df['albedo_mean'], 'r-', label='Albedo Mean')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Humidity (kg/kg)', color='b')
    ax2.set_ylabel('Albedo', color='r')
    ax.set_title('Humidity and Albedo')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_snow_ice(data: List[Dict[str, Any]]):
    """Plot snow and ice evolution."""
    df = pd.DataFrame(data)
    numeric_cols = ['day_of_year', 'total_days', 'snow_mean', 'snow_max', 'snow_cover_fraction',
                     'ice_mean', 'ice_max', 'ice_cover_fraction']
    df = convert_to_numeric(df, numeric_cols)
    time_col = 'total_days' if 'total_days' in df.columns else 'day_of_year'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Snow and Ice Evolution', fontsize=16)
    
    # Snow depth
    ax = axes[0, 0]
    if 'snow_mean' in df.columns:
        ax.plot(df[time_col], df['snow_mean'], 'b-', label='Mean Snow Depth')
    if 'snow_max' in df.columns:
        ax.plot(df[time_col], df['snow_max'], 'r--', label='Max Snow Depth', alpha=0.7)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Snow Depth (m)')
    ax.set_title('Snow Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Snow cover fraction
    ax = axes[0, 1]
    if 'snow_cover_fraction' in df.columns:
        ax.plot(df[time_col], df['snow_cover_fraction'], 'b-', label='Snow Cover')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Cover Fraction')
    ax.set_title('Snow Cover Fraction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Ice cover
    ax = axes[1, 0]
    if 'ice_mean' in df.columns:
        ax.plot(df[time_col], df['ice_mean'], 'b-', label='Mean Ice Cover')
    if 'ice_max' in df.columns:
        ax.plot(df[time_col], df['ice_max'], 'r--', label='Max Ice Cover', alpha=0.7)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Ice Cover Fraction')
    ax.set_title('Ice Cover')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Ice cover fraction
    ax = axes[1, 1]
    if 'ice_cover_fraction' in df.columns:
        ax.plot(df[time_col], df['ice_cover_fraction'], 'b-', label='Ice Cover')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Cover Fraction')
    ax.set_title('Ice Cover Fraction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig


def plot_component_contributions(data: List[Dict[str, Any]]):
    """Plot temperature component contributions if available."""
    # Check if we have component data
    sample = data[0] if data else {}
    component_cols = [k for k in sample.keys() if any(x in k for x in ['advection', 'diffusion', 'radiation', 
                                                                        'evaporation', 'ocean_transport', 'subsidence',
                                                                        'equilibrium_temp', 'net_radiation'])]
    
    if not component_cols:
        return None
    
    df = pd.DataFrame(data)
    numeric_cols = ['day_of_year', 'total_days'] + component_cols
    df = convert_to_numeric(df, numeric_cols)
    time_col = 'total_days' if 'total_days' in df.columns else 'day_of_year'
    
    # Group by component type (mean, min, max, std)
    components = {}
    for col in component_cols:
        parts = col.split('_')
        if len(parts) >= 2:
            comp_name = '_'.join(parts[:-1])
            stat = parts[-1]
            if comp_name not in components:
                components[comp_name] = {}
            components[comp_name][stat] = col
    
    if not components:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Temperature Component Contributions', fontsize=16)
    
    # Mean contributions
    ax = axes[0, 0]
    for comp_name, stats in components.items():
        if 'mean' in stats:
            ax.plot(df[time_col], df[stats['mean']], label=comp_name.replace('_', ' ').title())
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Temperature Change (K)')
    ax.set_title('Mean Component Contributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Net radiation
    ax = axes[0, 1]
    if 'net_radiation' in df.columns:
        ax.plot(df[time_col], df['net_radiation'], 'r-', label='Net Radiation')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Net Radiation (W/m²)')
    ax.set_title('Net Radiation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Equilibrium temperature
    ax = axes[1, 0]
    if 'equilibrium_temp_mean' in df.columns:
        ax.plot(df[time_col], df['equilibrium_temp_mean'] - 273.15, 'b-', label='Equilibrium Temp')
    if 'global_mean_temp' in df.columns:
        ax.plot(df[time_col], df['global_mean_temp'] - 273.15, 'r--', label='Actual Temp', alpha=0.7)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Equilibrium vs Actual Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Component std (variability)
    ax = axes[1, 1]
    for comp_name, stats in components.items():
        if 'std' in stats and comp_name != 'net_radiation':
            ax.plot(df[time_col], df[stats['std']], label=comp_name.replace('_', ' ').title(), alpha=0.7)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Std Dev (K)')
    ax.set_title('Component Variability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_combined_plot(data: List[Dict[str, Any]], day_of_year: float = None):
    """Create a single figure with all plots combined."""
    # Determine which plots are available
    sample = data[0] if data else {}
    has_components = any('advection' in k or 'diffusion' in k or 'radiation' in k for k in sample.keys())
    has_latitude = any('latitudes' in d for d in data)
    
    # Calculate grid size: we'll have multiple sections
    # Section 1: Temperature (2x2 = 4)
    # Section 2: Precipitation (2x2 = 4)
    # Section 3: Wind/Clouds/Humidity (2x2 = 4)
    # Section 4: Snow/Ice (2x2 = 4)
    # Section 5: Components (2x2 = 4, if available)
    # Section 6: Latitude profile (1x1 = 1, if available)
    
    n_sections = 4  # Always: temp, precip, wind/clouds, snow/ice
    if has_components:
        n_sections += 1
    if has_latitude:
        n_sections += 1
    
    # Create a large figure with subplots arranged in a grid
    # Each section is 2x2 except latitude which is 1x1
    # Arrange in columns: 2 columns, multiple rows
    rows_per_section = 2
    cols = 2
    total_rows = n_sections * rows_per_section + (1 if has_latitude else 0)
    
    fig = plt.figure(figsize=(20, total_rows * 4))
    gs = fig.add_gridspec(total_rows, cols, hspace=0.4, wspace=0.3)
    
    plot_idx = 0
    
    # Section 1: Temperature Evolution
    print("Creating temperature evolution plots...")
    df = pd.DataFrame(data)
    numeric_cols = ['day_of_year', 'total_days', 'global_mean_temp', 'T_equator', 
                     'T_pole_north', 'T_pole_south', 'T_min', 'T_max', 'T_std',
                     'gradient_north', 'gradient_south']
    df = convert_to_numeric(df, numeric_cols)
    time_col = 'total_days' if 'total_days' in df.columns else 'day_of_year'
    
    ax = fig.add_subplot(gs[plot_idx, 0])
    ax.plot(df[time_col], df['global_mean_temp'] - 273.15, 'b-', label='Global Mean')
    ax.axhline(y=15.0, color='r', linestyle='--', label='Earth Reference (15°C)')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Global Mean Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[plot_idx, 1])
    ax.plot(df[time_col], df['T_equator'] - 273.15, 'r-', label='Equator')
    ax.plot(df[time_col], df['T_pole_north'] - 273.15, 'b-', label='North Pole')
    ax.plot(df[time_col], df['T_pole_south'] - 273.15, 'g-', label='South Pole')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature by Latitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_idx += 1
    
    ax = fig.add_subplot(gs[plot_idx, 0])
    ax.plot(df[time_col], df['T_max'] - 273.15, 'r-', label='Max')
    ax.plot(df[time_col], df['T_min'] - 273.15, 'b-', label='Min')
    ax.fill_between(df[time_col], df['T_min'] - 273.15, df['T_max'] - 273.15, alpha=0.2)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[plot_idx, 1])
    if 'gradient_north' in df.columns and not df['gradient_north'].isna().all():
        ax.plot(df[time_col], df['gradient_north'], 'b-', label='North')
    if 'gradient_south' in df.columns and not df['gradient_south'].isna().all():
        ax.plot(df[time_col], df['gradient_south'], 'g-', label='South')
    ax.axhline(y=45.0, color='r', linestyle='--', label='Earth Reference (~45K)')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Temperature Gradient (K)')
    ax.set_title('Equator-Pole Temperature Gradient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_idx += 1
    
    # Section 2: Precipitation
    print("Creating precipitation plots...")
    numeric_cols = ['day_of_year', 'total_days', 'mean_precip', 'precip_max', 'precip_min',
                     'precip_equator', 'precip_pole_north', 'precip_pole_south']
    df = convert_to_numeric(df, numeric_cols)
    
    ax = fig.add_subplot(gs[plot_idx, 0])
    if 'mean_precip' in df.columns:
        ax.plot(df[time_col], df['mean_precip'], 'b-', label='Global Mean')
        ax.axhline(y=2.7, color='r', linestyle='--', label='Earth Reference (2.7 mm/day)')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Precipitation (mm/day)')
    ax.set_title('Global Mean Precipitation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[plot_idx, 1])
    if 'precip_equator' in df.columns:
        ax.plot(df[time_col], df['precip_equator'], 'r-', label='Equator')
    if 'precip_pole_north' in df.columns:
        ax.plot(df[time_col], df['precip_pole_north'], 'b-', label='North Pole')
    if 'precip_pole_south' in df.columns:
        ax.plot(df[time_col], df['precip_pole_south'], 'g-', label='South Pole')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Precipitation (mm/day)')
    ax.set_title('Precipitation by Latitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_idx += 1
    
    ax = fig.add_subplot(gs[plot_idx, 0])
    if 'precip_max' in df.columns and 'precip_min' in df.columns:
        ax.plot(df[time_col], df['precip_max'], 'r-', label='Max')
        ax.plot(df[time_col], df['precip_min'], 'b-', label='Min')
        ax.fill_between(df[time_col], df['precip_min'], df['precip_max'], alpha=0.2)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Precipitation (mm/day)')
    ax.set_title('Precipitation Range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[plot_idx, 1])
    if 'precip_std' in df.columns:
        ax.plot(df[time_col], df['precip_std'], 'g-', label='Std Dev')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Precipitation Std Dev (mm/day)')
    ax.set_title('Precipitation Variability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_idx += 1
    
    # Section 3: Wind, Clouds, Humidity
    print("Creating wind/clouds/humidity plots...")
    numeric_cols = ['day_of_year', 'total_days', 'wind_mean', 'wind_max', 'wind_u_zonal_mean', 'wind_v_merid_mean',
                     'cloud_mean', 'cloud_max', 'cloud_min', 'cloud_equator', 'cloud_pole_north', 'cloud_pole_south',
                     'humidity_mean', 'humidity_max', 'humidity_min', 'albedo_mean']
    df = convert_to_numeric(df, numeric_cols)
    
    ax = fig.add_subplot(gs[plot_idx, 0])
    if 'wind_mean' in df.columns:
        ax.plot(df[time_col], df['wind_mean'], 'b-', label='Mean Wind Speed')
    if 'wind_max' in df.columns:
        ax.plot(df[time_col], df['wind_max'], 'r--', label='Max Wind Speed', alpha=0.7)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.set_title('Wind Speed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[plot_idx, 1])
    if 'wind_u_zonal_mean' in df.columns:
        ax.plot(df[time_col], df['wind_u_zonal_mean'], 'b-', label='Zonal (E-W)')
    if 'wind_v_merid_mean' in df.columns:
        ax.plot(df[time_col], df['wind_v_merid_mean'], 'g-', label='Meridional (N-S)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Wind Component (m/s)')
    ax.set_title('Wind Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_idx += 1
    
    ax = fig.add_subplot(gs[plot_idx, 0])
    if 'cloud_mean' in df.columns:
        ax.plot(df[time_col], df['cloud_mean'], 'b-', label='Global Mean')
    if 'cloud_equator' in df.columns:
        ax.plot(df[time_col], df['cloud_equator'], 'r-', label='Equator', alpha=0.7)
    if 'cloud_pole_north' in df.columns:
        ax.plot(df[time_col], df['cloud_pole_north'], 'g-', label='North Pole', alpha=0.7)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Cloud Cover Fraction')
    ax.set_title('Cloud Cover')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    ax = fig.add_subplot(gs[plot_idx, 1])
    ax2 = ax.twinx()
    if 'humidity_mean' in df.columns:
        ax.plot(df[time_col], df['humidity_mean'], 'b-', label='Humidity Mean')
    if 'albedo_mean' in df.columns:
        ax2.plot(df[time_col], df['albedo_mean'], 'r-', label='Albedo Mean')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Humidity (kg/kg)', color='b')
    ax2.set_ylabel('Albedo', color='r')
    ax.set_title('Humidity and Albedo')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plot_idx += 1
    
    # Section 4: Snow and Ice
    print("Creating snow/ice plots...")
    numeric_cols = ['day_of_year', 'total_days', 'snow_mean', 'snow_max', 'snow_cover_fraction',
                     'ice_mean', 'ice_max', 'ice_cover_fraction']
    df = convert_to_numeric(df, numeric_cols)
    
    ax = fig.add_subplot(gs[plot_idx, 0])
    if 'snow_mean' in df.columns:
        ax.plot(df[time_col], df['snow_mean'], 'b-', label='Mean Snow Depth')
    if 'snow_max' in df.columns:
        ax.plot(df[time_col], df['snow_max'], 'r--', label='Max Snow Depth', alpha=0.7)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Snow Depth (m)')
    ax.set_title('Snow Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[plot_idx, 1])
    if 'snow_cover_fraction' in df.columns:
        ax.plot(df[time_col], df['snow_cover_fraction'], 'b-', label='Snow Cover')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Cover Fraction')
    ax.set_title('Snow Cover Fraction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plot_idx += 1
    
    ax = fig.add_subplot(gs[plot_idx, 0])
    if 'ice_mean' in df.columns:
        ax.plot(df[time_col], df['ice_mean'], 'b-', label='Mean Ice Cover')
    if 'ice_max' in df.columns:
        ax.plot(df[time_col], df['ice_max'], 'r--', label='Max Ice Cover', alpha=0.7)
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Ice Cover Fraction')
    ax.set_title('Ice Cover')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    ax = fig.add_subplot(gs[plot_idx, 1])
    if 'ice_cover_fraction' in df.columns:
        ax.plot(df[time_col], df['ice_cover_fraction'], 'b-', label='Ice Cover')
    ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
    ax.set_ylabel('Cover Fraction')
    ax.set_title('Ice Cover Fraction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plot_idx += 1
    
    # Section 5: Component Contributions (if available)
    if has_components:
        print("Creating component contribution plots...")
        component_cols = [k for k in sample.keys() if any(x in k for x in ['advection', 'diffusion', 'radiation', 
                                                                            'evaporation', 'ocean_transport', 'subsidence',
                                                                            'equilibrium_temp', 'net_radiation'])]
        numeric_cols = ['day_of_year', 'total_days'] + component_cols
        df = convert_to_numeric(df, numeric_cols)
        
        components = {}
        for col in component_cols:
            parts = col.split('_')
            if len(parts) >= 2:
                comp_name = '_'.join(parts[:-1])
                stat = parts[-1]
                if comp_name not in components:
                    components[comp_name] = {}
                components[comp_name][stat] = col
        
        ax = fig.add_subplot(gs[plot_idx, 0])
        for comp_name, stats in components.items():
            if 'mean' in stats:
                ax.plot(df[time_col], df[stats['mean']], label=comp_name.replace('_', ' ').title())
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
        ax.set_ylabel('Temperature Change (K)')
        ax.set_title('Mean Component Contributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = fig.add_subplot(gs[plot_idx, 1])
        if 'net_radiation' in df.columns:
            ax.plot(df[time_col], df['net_radiation'], 'r-', label='Net Radiation')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
        ax.set_ylabel('Net Radiation (W/m²)')
        ax.set_title('Net Radiation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
        
        ax = fig.add_subplot(gs[plot_idx, 0])
        if 'equilibrium_temp_mean' in df.columns:
            ax.plot(df[time_col], df['equilibrium_temp_mean'] - 273.15, 'b-', label='Equilibrium Temp')
        if 'global_mean_temp' in df.columns:
            ax.plot(df[time_col], df['global_mean_temp'] - 273.15, 'r--', label='Actual Temp', alpha=0.7)
        ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Equilibrium vs Actual Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = fig.add_subplot(gs[plot_idx, 1])
        for comp_name, stats in components.items():
            if 'std' in stats and comp_name != 'net_radiation':
                ax.plot(df[time_col], df[stats['std']], label=comp_name.replace('_', ' ').title(), alpha=0.7)
        ax.set_xlabel('Time (days)' if time_col == 'total_days' else 'Day of Year')
        ax.set_ylabel('Std Dev (K)')
        ax.set_title('Component Variability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Section 6: Latitude Profile (if available)
    if has_latitude:
        print("Creating latitude profile plot...")
        if day_of_year is None:
            latest = max(data, key=lambda x: float(x.get('total_days' if 'total_days' in x else 'day_of_year', 0)))
            day_of_year = float(latest.get('day_of_year', 0))
        
        closest = min(data, key=lambda x: abs(float(x.get('day_of_year', 0)) - day_of_year))
        
        if 'latitudes' in closest and 'zonal_means' in closest:
            try:
                if isinstance(closest['latitudes'], str):
                    latitudes = json.loads(closest['latitudes'])
                elif isinstance(closest['latitudes'], list):
                    latitudes = closest['latitudes']
                else:
                    latitudes = list(closest['latitudes'])
                
                if isinstance(closest['zonal_means'], str):
                    zonal_means = json.loads(closest['zonal_means'])
                elif isinstance(closest['zonal_means'], list):
                    zonal_means = closest['zonal_means']
                else:
                    zonal_means = list(closest['zonal_means'])
                
                import math
                if latitudes and abs(latitudes[0]) > 2:
                    lat_deg = latitudes
                else:
                    lat_deg = [math.degrees(lat) for lat in latitudes]
                
                # Span both columns for latitude profile
                ax = fig.add_subplot(gs[plot_idx, :])
                ax.plot(lat_deg, [t - 273.15 for t in zonal_means], 'b-', linewidth=2)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax.set_xlabel('Latitude (degrees)')
                ax.set_ylabel('Temperature (°C)')
                ax.set_title(f'Temperature Profile by Latitude (Day {day_of_year:.0f})')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-90, 90)
            except Exception as e:
                print(f"Warning: Could not create latitude profile: {e}")
    
    return fig


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_simulation.py <data_file.csv|json> [day_of_year] [output.png]")
        print("\nExample:")
        print("  python visualize_simulation.py simulation_data_20241124_120000.csv")
        print("  python visualize_simulation.py simulation_data_20241124_120000.csv 180 output.png")
        sys.exit(1)
    
    filename = sys.argv[1]
    day_of_year = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].replace('.', '').isdigit() else None
    output_file = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].endswith('.png') else None
    
    if output_file is None and day_of_year is None and len(sys.argv) > 2:
        # Check if second arg is output file
        if sys.argv[2].endswith('.png'):
            output_file = sys.argv[2]
            day_of_year = None
    
    print(f"Loading data from {filename}...")
    data = load_data(filename)
    print(f"Loaded {len(data)} time steps")
    
    # Create combined plot
    print("Creating combined visualization...")
    fig = create_combined_plot(data, day_of_year)
    
    # Save to PNG
    if output_file is None:
        input_path = Path(filename)
        output_file = input_path.stem + "_visualization.png"
    
    print(f"Saving to {output_file}...")
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    
    plt.close(fig)


if __name__ == "__main__":
    main()

