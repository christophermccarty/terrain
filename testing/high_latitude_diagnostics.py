"""High-latitude temperature diagnostics for planet simulation.

Compares simulation output against real-world Arctic/Antarctic temperature data
to identify sources of temperature discrepancies. Focuses on land vs ocean differences
and seasonal variations at high latitudes.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json

# Real-world high-latitude temperature data (annual means and seasonal extremes)
# Sources: NOAA, NASA GISS, Antarctic/Arctic research stations
REAL_WORLD_HIGH_LAT_DATA = {
    # Arctic land stations (60-90°N)
    "arctic_land": {
        # Latitude: (annual_mean_C, winter_mean_C, summer_mean_C, winter_min_C, summer_max_C)
        60: (-1.0, -15.0, 13.0, -35.0, 25.0),    # Anchorage, AK / Oslo area
        65: (-5.0, -20.0, 10.0, -40.0, 22.0),    # Fairbanks, AK / Arkhangelsk
        70: (-10.0, -25.0, 5.0, -45.0, 15.0),    # Murmansk / Barrow, AK
        75: (-15.0, -30.0, 2.0, -50.0, 10.0),    # Svalbard / North Slope
        80: (-20.0, -35.0, 0.0, -55.0, 5.0),     # Alert, Nunavut / Nord, Greenland
        85: (-30.0, -42.0, -5.0, -60.0, 2.0),    # High Arctic (sparse data)
        90: (-35.0, -45.0, -10.0, -65.0, 0.0),   # North Pole (ice cap)
    },
    # Arctic ocean (60-90°N) - moderated by water
    "arctic_ocean": {
        60: (2.0, -5.0, 10.0, -20.0, 18.0),      # Norwegian Sea / Bering Sea
        65: (0.0, -8.0, 8.0, -25.0, 15.0),       # Barents Sea / North Atlantic
        70: (-5.0, -15.0, 5.0, -30.0, 12.0),     # Arctic Ocean margins
        75: (-10.0, -20.0, 2.0, -35.0, 8.0),     # Arctic Ocean ice edge
        80: (-15.0, -25.0, -2.0, -40.0, 5.0),    # Central Arctic Ocean
        85: (-20.0, -30.0, -8.0, -45.0, 0.0),    # High Arctic Ocean
        90: (-25.0, -35.0, -12.0, -50.0, -5.0),  # North Pole (ocean under ice)
    },
    # Antarctic land (ice sheet) (60-90°S)
    "antarctic_land": {
        60: (-1.0, -8.0, 5.0, -25.0, 12.0),      # South Georgia, sub-Antarctic islands
        65: (-10.0, -18.0, -2.0, -35.0, 8.0),    # Antarctic Peninsula
        70: (-20.0, -30.0, -10.0, -45.0, 0.0),   # East Antarctic coast
        75: (-30.0, -40.0, -20.0, -55.0, -8.0),  # Antarctic plateau edge
        80: (-40.0, -50.0, -30.0, -65.0, -18.0), # Antarctic plateau interior
        85: (-50.0, -60.0, -40.0, -75.0, -28.0), # South Pole region
        90: (-55.0, -65.0, -45.0, -80.0, -32.0), # South Pole (Vostok, Dome Fuji)
    },
    # Southern Ocean (60-90°S) - moderates coastal temperatures
    "antarctic_ocean": {
        60: (2.0, -2.0, 7.0, -10.0, 15.0),       # Southern Ocean (Drake Passage)
        65: (-2.0, -8.0, 4.0, -20.0, 10.0),      # Weddell Sea / Ross Sea
        70: (-10.0, -18.0, -2.0, -30.0, 5.0),    # Antarctic coastal waters
        75: (-15.0, -25.0, -8.0, -38.0, 0.0),    # Under ice shelf
        80: (-20.0, -30.0, -12.0, -45.0, -5.0),  # (sparse ocean data)
        85: (-25.0, -35.0, -18.0, -50.0, -10.0), # (sparse ocean data)
        90: (-30.0, -40.0, -22.0, -55.0, -15.0), # (extrapolated)
    },
}


def analyze_high_latitude_temperatures(
    state,
    latitude_threshold: float = 60.0,
    land_elevation_threshold: float = 0.02,
) -> Dict[str, Any]:
    """Analyze temperature differences between simulation and real-world data at high latitudes.

    Args:
        state: PlanetState with temperature and elevation data
        latitude_threshold: Latitude (degrees) above which to analyze (default 60°)
        land_elevation_threshold: Elevation threshold for land classification (default 0.02)

    Returns:
        Dictionary with diagnostic results including:
        - comparison: Band-by-band comparison with real-world data
        - land_ocean_diff: Difference in temperature behavior between land and ocean
        - extremes: Maximum/minimum temperatures by latitude band
        - summary: Overall statistics
    """
    if state.temperature is None or state.elevation is None:
        return {"error": "Temperature or elevation data not available"}

    T = state.temperature
    elev = state.elevation
    H, W = T.shape

    # Calculate latitude for each row (degrees, north positive)
    lat_deg = 90.0 - (np.arange(H, dtype=np.float32) + 0.5) * 180.0 / H

    # Determine hemisphere and season from day_of_year
    day = state.day_of_year
    # Northern summer: days 80-264, Southern summer: days 264-445 (or 0-80)
    is_nh_summer = 80 <= day < 264
    season = "summer" if is_nh_summer else "winter"

    # Create land/ocean masks
    land_mask = elev > land_elevation_threshold
    ocean_mask = ~land_mask

    # Analyze both hemispheres
    results = {
        "northern_hemisphere": _analyze_hemisphere(
            T, elev, lat_deg, land_mask, ocean_mask,
            latitude_threshold, "north", season
        ),
        "southern_hemisphere": _analyze_hemisphere(
            T, elev, lat_deg, land_mask, ocean_mask,
            latitude_threshold, "south", season
        ),
        "day_of_year": day,
        "season": season,
    }

    # Add summary statistics
    results["summary"] = _compute_summary_stats(results)

    return results


def _analyze_hemisphere(
    T: np.ndarray,
    elev: np.ndarray,
    lat_deg: np.ndarray,
    land_mask: np.ndarray,
    ocean_mask: np.ndarray,
    lat_threshold: float,
    hemisphere: str,
    season: str,
) -> Dict[str, Any]:
    """Analyze high-latitude temperatures for one hemisphere."""

    # Select hemisphere
    if hemisphere == "north":
        hemi_mask = lat_deg >= lat_threshold
        ref_land = REAL_WORLD_HIGH_LAT_DATA["arctic_land"]
        ref_ocean = REAL_WORLD_HIGH_LAT_DATA["arctic_ocean"]
        hemi_name = "Arctic"
    else:
        hemi_mask = lat_deg <= -lat_threshold
        ref_land = REAL_WORLD_HIGH_LAT_DATA["antarctic_land"]
        ref_ocean = REAL_WORLD_HIGH_LAT_DATA["antarctic_ocean"]
        hemi_name = "Antarctic"

    if not np.any(hemi_mask):
        return {"error": f"No data above {lat_threshold}° in {hemisphere} hemisphere"}

    # Get 2D latitude array for broadcasting (repeat across longitude)
    H, W = T.shape
    lat_2d = np.repeat(lat_deg[:, np.newaxis], W, axis=1)

    # Analyze by 5-degree latitude bands
    bands = []
    for lat_center in [60, 65, 70, 75, 80, 85, 90]:
        if hemisphere == "north":
            if lat_center < lat_threshold:
                continue
            band_mask = (lat_2d >= lat_threshold) & (np.abs(lat_2d - lat_center) < 2.5)
        else:
            if lat_center < lat_threshold:
                continue
            band_mask = (lat_2d <= -lat_threshold) & (np.abs(lat_2d + lat_center) < 2.5)

        if not np.any(band_mask):
            continue

        # Get simulation temperatures for land and ocean
        land_temps = T[band_mask & land_mask]
        ocean_temps = T[band_mask & ocean_mask]

        # Convert to Celsius
        sim_land_mean = float(np.mean(land_temps) - 273.15) if land_temps.size > 0 else None
        sim_ocean_mean = float(np.mean(ocean_temps) - 273.15) if ocean_temps.size > 0 else None
        sim_land_min = float(np.min(land_temps) - 273.15) if land_temps.size > 0 else None
        sim_land_max = float(np.max(land_temps) - 273.15) if land_temps.size > 0 else None
        sim_ocean_min = float(np.min(ocean_temps) - 273.15) if ocean_temps.size > 0 else None
        sim_ocean_max = float(np.max(ocean_temps) - 273.15) if ocean_temps.size > 0 else None

        # Get real-world reference data
        real_land = ref_land.get(lat_center, None)
        real_ocean = ref_ocean.get(lat_center, None)

        # Extract appropriate seasonal value from real-world data
        # Format: (annual_mean, winter_mean, summer_mean, winter_min, summer_max)
        if real_land:
            real_land_annual = real_land[0]
            real_land_seasonal = real_land[2] if season == "summer" else real_land[1]
            real_land_extreme_min = real_land[3]
            real_land_extreme_max = real_land[4]
        else:
            real_land_annual = real_land_seasonal = real_land_extreme_min = real_land_extreme_max = None

        if real_ocean:
            real_ocean_annual = real_ocean[0]
            real_ocean_seasonal = real_ocean[2] if season == "summer" else real_ocean[1]
            real_ocean_extreme_min = real_ocean[3]
            real_ocean_extreme_max = real_ocean[4]
        else:
            real_ocean_annual = real_ocean_seasonal = real_ocean_extreme_min = real_ocean_extreme_max = None

        # Calculate differences
        land_diff_annual = sim_land_mean - real_land_annual if (sim_land_mean is not None and real_land_annual is not None) else None
        land_diff_seasonal = sim_land_mean - real_land_seasonal if (sim_land_mean is not None and real_land_seasonal is not None) else None
        ocean_diff_annual = sim_ocean_mean - real_ocean_annual if (sim_ocean_mean is not None and real_ocean_annual is not None) else None
        ocean_diff_seasonal = sim_ocean_mean - real_ocean_seasonal if (sim_ocean_mean is not None and real_ocean_seasonal is not None) else None

        # Check for unrealistic extremes
        land_too_cold = (sim_land_min is not None and real_land_extreme_min is not None and
                         sim_land_min < real_land_extreme_min - 20.0)  # More than 20°C colder
        land_too_hot = (sim_land_max is not None and real_land_extreme_max is not None and
                        sim_land_max > real_land_extreme_max + 20.0)   # More than 20°C hotter
        ocean_too_cold = (sim_ocean_min is not None and real_ocean_extreme_min is not None and
                          sim_ocean_min < real_ocean_extreme_min - 10.0)
        ocean_too_hot = (sim_ocean_max is not None and real_ocean_extreme_max is not None and
                         sim_ocean_max > real_ocean_extreme_max + 10.0)

        band_result = {
            "latitude": lat_center * (1 if hemisphere == "north" else -1),
            "land_cell_count": int(np.sum(band_mask & land_mask)),
            "ocean_cell_count": int(np.sum(band_mask & ocean_mask)),
            "simulation": {
                "land_mean_c": round(sim_land_mean, 1) if sim_land_mean is not None else None,
                "land_min_c": round(sim_land_min, 1) if sim_land_min is not None else None,
                "land_max_c": round(sim_land_max, 1) if sim_land_max is not None else None,
                "ocean_mean_c": round(sim_ocean_mean, 1) if sim_ocean_mean is not None else None,
                "ocean_min_c": round(sim_ocean_min, 1) if sim_ocean_min is not None else None,
                "ocean_max_c": round(sim_ocean_max, 1) if sim_ocean_max is not None else None,
            },
            "real_world": {
                "land_annual_c": real_land_annual,
                "land_seasonal_c": real_land_seasonal,
                "land_extreme_min_c": real_land_extreme_min,
                "land_extreme_max_c": real_land_extreme_max,
                "ocean_annual_c": real_ocean_annual,
                "ocean_seasonal_c": real_ocean_seasonal,
                "ocean_extreme_min_c": real_ocean_extreme_min,
                "ocean_extreme_max_c": real_ocean_extreme_max,
            },
            "difference": {
                "land_vs_annual_c": round(land_diff_annual, 1) if land_diff_annual is not None else None,
                "land_vs_seasonal_c": round(land_diff_seasonal, 1) if land_diff_seasonal is not None else None,
                "ocean_vs_annual_c": round(ocean_diff_annual, 1) if ocean_diff_annual is not None else None,
                "ocean_vs_seasonal_c": round(ocean_diff_seasonal, 1) if ocean_diff_seasonal is not None else None,
            },
            "flags": {
                "land_too_cold": land_too_cold,
                "land_too_hot": land_too_hot,
                "ocean_too_cold": ocean_too_cold,
                "ocean_too_hot": ocean_too_hot,
            }
        }

        bands.append(band_result)

    return {
        "hemisphere": hemi_name,
        "bands": bands,
    }


def _compute_summary_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute overall summary statistics across both hemispheres."""

    all_land_diffs = []
    all_ocean_diffs = []
    flags_summary = {
        "land_too_cold_count": 0,
        "land_too_hot_count": 0,
        "ocean_too_cold_count": 0,
        "ocean_too_hot_count": 0,
    }

    for hemi_key in ["northern_hemisphere", "southern_hemisphere"]:
        hemi = results.get(hemi_key, {})
        for band in hemi.get("bands", []):
            # Collect differences
            diff = band.get("difference", {})
            if diff.get("land_vs_seasonal_c") is not None:
                all_land_diffs.append(diff["land_vs_seasonal_c"])
            if diff.get("ocean_vs_seasonal_c") is not None:
                all_ocean_diffs.append(diff["ocean_vs_seasonal_c"])

            # Count flags
            flags = band.get("flags", {})
            if flags.get("land_too_cold"):
                flags_summary["land_too_cold_count"] += 1
            if flags.get("land_too_hot"):
                flags_summary["land_too_hot_count"] += 1
            if flags.get("ocean_too_cold"):
                flags_summary["ocean_too_cold_count"] += 1
            if flags.get("ocean_too_hot"):
                flags_summary["ocean_too_hot_count"] += 1

    return {
        "mean_land_bias_c": round(np.mean(all_land_diffs), 1) if all_land_diffs else None,
        "max_land_bias_c": round(np.max(np.abs(all_land_diffs)), 1) if all_land_diffs else None,
        "mean_ocean_bias_c": round(np.mean(all_ocean_diffs), 1) if all_ocean_diffs else None,
        "max_ocean_bias_c": round(np.max(np.abs(all_ocean_diffs)), 1) if all_ocean_diffs else None,
        "flags": flags_summary,
    }


def print_high_latitude_report(results: Dict[str, Any]) -> None:
    """Print formatted high-latitude diagnostic report."""

    if "error" in results:
        print(f"\n❌ Error: {results['error']}\n")
        return

    print("\n" + "="*90)
    print(f"HIGH-LATITUDE TEMPERATURE DIAGNOSTIC REPORT")
    print(f"Day of Year: {results['day_of_year']:.1f} ({results['season'].upper()})")
    print("="*90)

    # Print each hemisphere
    for hemi_key in ["northern_hemisphere", "southern_hemisphere"]:
        hemi = results.get(hemi_key, {})
        if "error" in hemi:
            print(f"\n{hemi_key.replace('_', ' ').title()}: {hemi['error']}")
            continue

        print(f"\n{hemi['hemisphere'].upper()} (>{abs(results.get('northern_hemisphere', {}).get('bands', [{}])[0].get('latitude', 60))}°)")
        print("-"*90)
        print(f"{'Lat':>4} | {'Type':>5} | {'Cells':>6} | {'Sim°C':>7} | {'Real°C':>7} | "
              f"{'Diff':>7} | {'Min°C':>7} | {'Max°C':>7} | {'Flags':>12}")
        print("-"*90)

        for band in hemi.get("bands", []):
            lat = band["latitude"]

            # Land row
            land_cells = band["land_cell_count"]
            sim_land = band["simulation"]["land_mean_c"]
            real_land = band["real_world"]["land_seasonal_c"]
            diff_land = band["difference"]["land_vs_seasonal_c"]
            min_land = band["simulation"]["land_min_c"]
            max_land = band["simulation"]["land_max_c"]
            flags_land = []
            if band["flags"]["land_too_cold"]:
                flags_land.append("TOO_COLD")
            if band["flags"]["land_too_hot"]:
                flags_land.append("TOO_HOT")

            if land_cells > 0:
                print(f"{lat:>4}° | {'LAND':>5} | {land_cells:>6} | "
                      f"{sim_land:>7.1f} | {real_land:>7.1f} | "
                      f"{diff_land:>+7.1f} | {min_land:>7.1f} | {max_land:>7.1f} | "
                      f"{','.join(flags_land):>12}")

            # Ocean row
            ocean_cells = band["ocean_cell_count"]
            sim_ocean = band["simulation"]["ocean_mean_c"]
            real_ocean = band["real_world"]["ocean_seasonal_c"]
            diff_ocean = band["difference"]["ocean_vs_seasonal_c"]
            min_ocean = band["simulation"]["ocean_min_c"]
            max_ocean = band["simulation"]["ocean_max_c"]
            flags_ocean = []
            if band["flags"]["ocean_too_cold"]:
                flags_ocean.append("TOO_COLD")
            if band["flags"]["ocean_too_hot"]:
                flags_ocean.append("TOO_HOT")

            if ocean_cells > 0:
                print(f"{lat:>4}° | {'OCEAN':>5} | {ocean_cells:>6} | "
                      f"{sim_ocean:>7.1f} | {real_ocean:>7.1f} | "
                      f"{diff_ocean:>+7.1f} | {min_ocean:>7.1f} | {max_ocean:>7.1f} | "
                      f"{','.join(flags_ocean):>12}")

    # Summary
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    summary = results.get("summary", {})

    land_bias = summary.get('mean_land_bias_c')
    max_land = summary.get('max_land_bias_c')
    ocean_bias = summary.get('mean_ocean_bias_c')
    max_ocean = summary.get('max_ocean_bias_c')

    land_str = f"{land_bias:>6.1f}" if land_bias is not None else "   N/A"
    max_land_str = f"{max_land:.1f}" if max_land is not None else "N/A"
    ocean_str = f"{ocean_bias:>6.1f}" if ocean_bias is not None else "   N/A"
    max_ocean_str = f"{max_ocean:.1f}" if max_ocean is not None else "N/A"

    print(f"Mean land bias:   {land_str} °C  (max: {max_land_str} °C)")
    print(f"Mean ocean bias:  {ocean_str} °C  (max: {max_ocean_str} °C)")

    flags = summary.get("flags", {})
    print(f"\nExtreme Temperature Flags:")
    print(f"  Land too cold:  {flags.get('land_too_cold_count', 0)} bands")
    print(f"  Land too hot:   {flags.get('land_too_hot_count', 0)} bands")
    print(f"  Ocean too cold: {flags.get('ocean_too_cold_count', 0)} bands")
    print(f"  Ocean too hot:  {flags.get('ocean_too_hot_count', 0)} bands")
    print("="*90 + "\n")


def export_high_latitude_comparison(results: Dict[str, Any], filename: str = None) -> str:
    """Export high-latitude comparison to JSON file.

    Args:
        results: Results from analyze_high_latitude_temperatures()
        filename: Output filename (auto-generated if None)

    Returns:
        Path to exported file
    """
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = Path(__file__).resolve().parent / "results" / "high_latitude" / f"high_latitude_diagnostic_{timestamp}.json"

    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    return str(filepath)


if __name__ == "__main__":
    # Example usage (requires running simulation)
    print("High-latitude diagnostics module loaded.")
    print("Usage:")
    print("  from high_latitude_diagnostics import analyze_high_latitude_temperatures, print_high_latitude_report")
    print("  results = analyze_high_latitude_temperatures(state)")
    print("  print_high_latitude_report(results)")
