"""Test ocean temperature fixes.

This script runs the simulation at different times of year and compares
ocean temperatures with real-world Sea Surface Temperature (SST) data.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from simulate import create_initial_state, simulate_step
from terrain import ensure_elevation

# Real-world SST reference data (approximate averages)
# Format: (latitude, summer_temp_C, winter_temp_C)
REAL_WORLD_SST = {
    # Equatorial
    0: (27, 27),    # Equator: ~27C year-round
    # Subtropics (warmest oceans)
    10: (28, 27),   # 10N/S: 27-28C
    20: (28, 26),   # 20N/S: 26-28C (warm pools up to 30C)
    30: (26, 20),   # 30N/S: 20-26C
    # Mid-latitudes
    40: (20, 12),   # 40N/S: 12-20C
    50: (14, 6),    # 50N/S: 6-14C
    # High latitudes
    60: (8, 2),     # 60N/S: 2-8C
    70: (4, -1),    # 70N/S: -1 to 4C
}


def analyze_ocean_temps(state, day_of_year):
    """Analyze ocean temperatures at different latitudes."""
    T = state.temperature
    elev = state.elevation
    Hc, Wc = T.shape

    # Create latitude array
    lat_rad = np.linspace(np.pi/2, -np.pi/2, Hc)
    lat_deg = np.rad2deg(lat_rad)

    # Ocean mask (elevation < 0.2 for procedural terrain)
    sea_mask = elev < 0.2

    print(f"\n--- Ocean Temperature Analysis (Day {day_of_year}) ---\n")

    # Determine season
    is_nh_summer = 80 < day_of_year < 264

    results = []
    for target_lat in [0, 10, 20, 25, 30, 40, 50, 60]:
        # Find rows near target latitude (both hemispheres)
        for sign in [1, -1]:
            lat_target = sign * target_lat
            lat_idx = np.argmin(np.abs(lat_deg - lat_target))
            row_lat = lat_deg[lat_idx]

            # Get ocean pixels in this row
            ocean_mask_row = sea_mask[lat_idx, :]
            if not np.any(ocean_mask_row):
                continue

            ocean_temps = T[lat_idx, ocean_mask_row]
            mean_T_C = np.mean(ocean_temps) - 273.15
            max_T_C = np.max(ocean_temps) - 273.15
            min_T_C = np.min(ocean_temps) - 273.15

            # Get reference SST
            ref_lat = abs(target_lat)
            if ref_lat in REAL_WORLD_SST:
                summer_ref, winter_ref = REAL_WORLD_SST[ref_lat]
                # Determine expected based on hemisphere and season
                if lat_target >= 0:  # Northern hemisphere
                    expected = summer_ref if is_nh_summer else winter_ref
                else:  # Southern hemisphere
                    expected = winter_ref if is_nh_summer else summer_ref
                bias = mean_T_C - expected
            else:
                expected = None
                bias = None

            results.append({
                'lat': lat_target,
                'mean': mean_T_C,
                'max': max_T_C,
                'min': min_T_C,
                'expected': expected,
                'bias': bias,
            })

            # Print result
            lat_str = f"{abs(lat_target):3.0f}{'N' if lat_target >= 0 else 'S'}"
            if expected is not None:
                bias_str = f"{bias:+5.1f}C"
                status = "[OK]" if abs(bias) < 5 else "[!]" if abs(bias) < 10 else "[!!!]"
            else:
                bias_str = "  N/A"
                status = ""

            print(f"  {lat_str}: Mean={mean_T_C:5.1f}C, Max={max_T_C:5.1f}C, Min={min_T_C:5.1f}C | Expected={expected if expected else 'N/A':>4}, Bias={bias_str} {status}")

    # Summary statistics
    print(f"\n--- Summary ---")
    ocean_all = T[sea_mask]
    ocean_all_C = ocean_all - 273.15
    print(f"  Global ocean mean: {np.mean(ocean_all_C):.1f}C")
    print(f"  Global ocean max:  {np.max(ocean_all_C):.1f}C")
    print(f"  Global ocean min:  {np.min(ocean_all_C):.1f}C")
    print(f"  Pct > 32C (real max): {100 * np.sum(ocean_all_C > 32) / len(ocean_all_C):.1f}%")
    print(f"  Pct > 35C (too hot):  {100 * np.sum(ocean_all_C > 35) / len(ocean_all_C):.1f}%")
    print(f"  Pct > 40C (extreme):  {100 * np.sum(ocean_all_C > 40) / len(ocean_all_C):.1f}%")

    return results


def main():
    print("=" * 60)
    print("Ocean Temperature Fix Verification")
    print("=" * 60)

    # Generate elevation
    print("\nGenerating terrain...")
    elevation = ensure_elevation(size=256, seed=42)

    # Test at different times of year
    test_days = [
        (172, "Northern Summer Solstice"),
        (356, "Southern Summer Solstice"),
        (80, "Northern Spring Equinox"),
        (264, "Northern Fall Equinox"),
    ]

    # First, run a 2-year spinup to reach equilibrium
    print(f"\nRunning 2-year spinup to reach thermal equilibrium...")
    state = create_initial_state(elevation, day_of_year=0.0)
    spinup_days = 730  # 2 years
    current_day = 0
    step_days = 10
    while current_day < spinup_days:
        days_to_run = min(step_days, spinup_days - current_day)
        state, _ = simulate_step(state, days=float(days_to_run))
        current_day += days_to_run
        if current_day % 100 == 0:
            print(f"  Spinup day {current_day}/{spinup_days}")
    print("Spinup complete!")

    for target_day, season_name in test_days:
        print(f"\n{'='*60}")
        print(f"Simulating to Day {target_day} ({season_name})")
        print("=" * 60)

        # Run from end of spinup to target day of year
        # First run to start of year
        target_total_day = spinup_days + target_day
        while state.total_days < target_total_day:
            days_to_run = min(step_days, target_total_day - state.total_days)
            state, _ = simulate_step(state, days=float(days_to_run))

        # Analyze ocean temperatures
        results = analyze_ocean_temps(state, target_day)

    print(f"\n{'='*60}")
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
