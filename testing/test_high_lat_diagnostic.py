"""Test script for high-latitude temperature diagnostics.

Run this after starting a simulation to compare high-latitude temperatures
with real-world data and identify sources of temperature extremes.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from terrain import ensure_elevation
from simulate import create_initial_state, simulate_multiple_steps
from high_latitude_diagnostics import (
    analyze_high_latitude_temperatures,
    print_high_latitude_report,
    export_high_latitude_comparison
)
import numpy as np

def run_diagnostic_test(days_to_simulate: int = 180, size: int = 512):
    """Run a quick simulation and analyze high-latitude temperatures.

    Args:
        days_to_simulate: Number of days to simulate (default: 180 = ~6 months)
        size: Map size (default: 512)
    """
    print(f"Initializing {size}x{size} simulation...")

    # Generate or load elevation
    elevation = ensure_elevation(size, seed=42, octaves=4, freq=1.2, lac=2.0, gain=0.5)

    # Create initial state (start at spring equinox, day 80)
    print("Creating initial state (day 80 - spring equinox)...")
    state = create_initial_state(elevation, day_of_year=80.0, wind_block_size=8)

    # Run simulation for specified days
    print(f"Simulating {days_to_simulate} days (this may take a minute)...")
    states, diagnostics = simulate_multiple_steps(state, total_days=days_to_simulate, step_days=1.0)
    final_state = states[-1]

    print(f"\nFinal state: Day {final_state.day_of_year:.1f}")
    print(f"Total days simulated: {final_state.total_days:.1f}")

    # Analyze high-latitude temperatures
    print("\nAnalyzing high-latitude temperatures...")
    results = analyze_high_latitude_temperatures(final_state, latitude_threshold=60.0)

    # Print report
    print_high_latitude_report(results)

    # Export to JSON for further analysis
    output_file = export_high_latitude_comparison(results)
    print(f"\n[OK] Results exported to: {output_file}")

    # Print some quick insights
    print("\n" + "="*90)
    print("QUICK INSIGHTS")
    print("="*90)

    summary = results.get("summary", {})

    # Temperature bias
    land_bias = summary.get("mean_land_bias_c")
    ocean_bias = summary.get("mean_ocean_bias_c")

    if land_bias is not None and ocean_bias is not None:
        if abs(land_bias) > 10:
            print(f"[!] LARGE LAND BIAS: {land_bias:+.1f}C")
            if land_bias < -5:
                print("  -> Land temperatures are TOO COLD in high latitudes")
                print("  -> Check: polar_cooling_scale, T_min floor, heat transport")
            elif land_bias > 5:
                print("  -> Land temperatures are TOO HOT in high latitudes")
                print("  -> Check: albedo, greenhouse effect, elevation corrections")

        if abs(ocean_bias) > 10:
            print(f"[!] LARGE OCEAN BIAS: {ocean_bias:+.1f}C")
            if ocean_bias < -5:
                print("  -> Ocean temperatures are TOO COLD in high latitudes")
                print("  -> Check: ocean heat transport, sea ice modeling")
            elif ocean_bias > 5:
                print("  -> Ocean temperatures are TOO HOT in high latitudes")

    # Land-ocean disparity
    if land_bias is not None and ocean_bias is not None:
        disparity = abs(land_bias - ocean_bias)
        if disparity > 15:
            print(f"\n[!] LARGE LAND-OCEAN DISPARITY: {disparity:.1f}C difference")
            print("  -> Land and ocean have very different temperature biases")
            print("  -> This suggests issues with:")
            print("     - Thermal inertia differences")
            print("     - Land-specific cooling mechanisms (evaporation, snow albedo)")
            print("     - Ocean heat transport")

    # Extreme flags
    flags = summary.get("flags", {})
    total_flags = sum(flags.values())
    if total_flags > 0:
        print(f"\n[!] {total_flags} EXTREME TEMPERATURE FLAGS detected")
        print("  -> See detailed report above for specific latitude bands")

    # Specific guidance based on patterns
    print("\n" + "="*90)
    print("POSSIBLE CAUSES & FIXES")
    print("="*90)

    # Analyze patterns in the data
    nh_bands = results.get("northern_hemisphere", {}).get("bands", [])
    sh_bands = results.get("southern_hemisphere", {}).get("bands", [])

    # Check for systematic cold bias at highest latitudes
    high_lat_land_diffs = []
    for band in nh_bands + sh_bands:
        if abs(band["latitude"]) >= 70 and band["land_cell_count"] > 0:
            diff = band["difference"].get("land_vs_seasonal_c")
            if diff is not None:
                high_lat_land_diffs.append(diff)

    if high_lat_land_diffs:
        mean_high_lat_diff = np.mean(high_lat_land_diffs)
        if mean_high_lat_diff < -15:
            print("\n1. EXTREME POLAR COLD BIAS (>70° latitude)")
            print("   Likely causes:")
            print("   → T_min floor too low (temperature.py:545)")
            print("   → Excessive polar cooling (F_latent, F_convective in temperature.py:489, 519)")
            print("   → Insufficient heat transport to poles")
            print("   Suggested fixes:")
            print("   → Increase T_min from 240K to 245-250K")
            print("   → Reduce polar_cooling_scale from 0.8 to 0.5-0.6")
            print("   → Increase heat diffusion/transport strength")
        elif mean_high_lat_diff > 15:
            print("\n1. EXTREME POLAR WARM BIAS (>70° latitude)")
            print("   Likely causes:")
            print("   → Excessive greenhouse effect at poles (epsilon_pole in temperature.py:440)")
            print("   → Insufficient polar cooling mechanisms")
            print("   → Too much heat transport to poles")

    # Check for land-specific issues
    land_cold_flags = flags.get("land_too_cold_count", 0)
    if land_cold_flags > 2:
        print("\n2. LAND-SPECIFIC COLD EXTREMES")
        print("   Likely causes:")
        print("   → Elevation lapse rate too strong (temperature.py:232)")
        print("   → Missing land thermal inertia (should moderate extremes)")
        print("   → Snow/ice albedo feedback too strong")
        print("   Suggested fixes:")
        print("   → Check elevation_to_alt_km() mapping")
        print("   → Add soil/land heat capacity to simulation")

    print("\n" + "="*90)
    print("\nTo investigate further:")
    print("  1. Check the temperature.py file, especially:")
    print("     - Line 440: epsilon_pole (greenhouse effect at poles)")
    print("     - Line 489: F_latent_max (latent heat cooling)")
    print("     - Line 519: convection_efficiency (convective cooling)")
    print("     - Line 545: T_min (temperature floor)")
    print("  2. Run this diagnostic at different times of year (vary days_to_simulate)")
    print("  3. Compare 'winter' vs 'summer' results to check seasonal behavior")
    print("="*90 + "\n")

    return results, final_state


if __name__ == "__main__":
    # Run diagnostic test
    # You can adjust these parameters:
    # - days_to_simulate: 90 (spring), 180 (summer), 270 (fall), 360 (winter)
    # - size: 256 (faster), 512 (standard), 1024 (detailed)

    print("="*90)
    print("HIGH-LATITUDE TEMPERATURE DIAGNOSTIC TEST")
    print("="*90)
    print("\nThis will:")
    print("  1. Create a simulation")
    print("  2. Run for ~6 months")
    print("  3. Compare high-latitude temperatures with real-world data")
    print("  4. Identify sources of temperature extremes")
    print("\n" + "="*90 + "\n")

    results, state = run_diagnostic_test(days_to_simulate=180, size=512)
