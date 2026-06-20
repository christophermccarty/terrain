"""quick_diagnosis.py — Fast diagnostic tool for iterative physics tuning.

Instead of running a 1-year benchmark (slow) after every parameter change, use
this script to:

1. Print the T_base ocean profile INSTANTLY (no simulation) -> tells you if your
   temperature TARGETS are correctly calibrated vs Earth SST.

2. Run a configurable short simulation (default 90 days) and report key metrics:
   - NH/SH ice extent at monthly snapshots
   - Temperature at key diagnostic latitudes (55°N, 65°N, 75°N, 85°N)
   - Whether temperatures are converging toward (or away from) Earth references

Usage:
    python testing/quick_diagnosis.py                    # 90-day default
    python testing/quick_diagnosis.py --days 180         # 6-month run
    python testing/quick_diagnosis.py --tbase-only       # just print T_base table

The script starts from a fresh initial state (day 80 = spring equinox) so results
are deterministic and comparable across parameter changes.  It uses a small 60×120
grid for speed (~10–30s for 90 days depending on hardware).

Efficient tuning workflow:
1. Make a parameter change
2. Run `python testing/quick_diagnosis.py --tbase-only` (< 1s) to see T_base shift
3. Run `python testing/quick_diagnosis.py` (10-30s) to see how simulation responds
4. Only run the full 1-year benchmark when a 90-day run looks promising
"""
from __future__ import annotations

import sys
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _make_earth_elevation(H: int = 60, W: int = 120) -> np.ndarray:
    """Generate a simple procedural Earth-like elevation field (land fraction ~30%)."""
    rng = np.random.default_rng(42)  # fixed seed for reproducibility
    lat = (0.5 - (np.arange(H) + 0.5) / H) * np.pi  # radians, +pi/2 at top
    lon = np.linspace(0, 2 * np.pi, W, endpoint=False)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")

    # Large-scale continent pattern (roughly Earth-like)
    land = (
        0.40 * np.cos(1.0 * LON) * np.cos(0.8 * LAT)
        + 0.25 * np.cos(2.1 * LON + 0.5) * np.cos(1.5 * LAT)
        + 0.15 * np.sin(3.0 * LON + 1.2) * np.cos(2.0 * LAT)
        + 0.10 * rng.standard_normal((H, W))
    )
    # Normalise to [0,1] with ~30% land
    land = (land - np.percentile(land, 70)) / (np.max(land) - np.min(land) + 1e-9)
    land = np.clip(land, 0.0, 1.0).astype(np.float32)
    return land


def _lat_rows(H: int) -> np.ndarray:
    return (0.5 - (np.arange(H) + 0.5) / H) * 180.0


def _zonal_T_at(T: np.ndarray, lat_c: float) -> float:
    H = T.shape[0]
    lats = _lat_rows(H)
    idx = int(np.argmin(np.abs(lats - lat_c)))
    return float(np.mean(T[idx, :])) - 273.15


# ---------------------------------------------------------------------------
# T_base profile (no simulation needed)
# ---------------------------------------------------------------------------

def print_tbase_profile() -> None:
    from diagnostics import compute_t_base_profile, print_t_base_report
    data = compute_t_base_profile()
    print_t_base_report(data)

    # Also compute what effect the new T_min (215K) has vs old (200K)
    from temperature import temperature_kelvin_for_lat
    from planet_params import EARTH
    print("  [T_min sensitivity]  How much did raising T_min 200->215K change T_lat_annual_mean?")
    lats_deg = [85, 75, 65, 55]
    changes = {}
    for ld in lats_deg:
        lr = np.array([np.deg2rad(float(ld))], dtype=np.float32)
        t = float(temperature_kelvin_for_lat(lr, day_of_year=80, polar_cooling_scale=0.3)[0])
        # Estimate what the value would have been with T_min=200 (if it's 215, it might have been clipped)
        if abs(t - 215.0) < 0.5:
            # currently at floor -> was also at floor before (or lower); old value was 200
            old_t = 200.0
        else:
            old_t = t  # not at floor, no change
        changes[ld] = (old_t, t, t - old_t)
    for ld, (old, new, delta) in changes.items():
        if abs(delta) > 0.1:
            print(f"    {ld:+4}°N: T_lat was {old:.1f}K -> now {new:.1f}K  (D={delta:+.1f}K -> DT_base={delta:+.1f}K)")
        else:
            print(f"    {ld:+4}°N: not at floor, T_lat unchanged ({new:.1f}K)")
    print()


# ---------------------------------------------------------------------------
# Short simulation + diagnostics
# ---------------------------------------------------------------------------

def run_quick_benchmark(days: int = 90, grid_H: int = 60, grid_W: int = 120) -> None:
    from simulate import simulate_step, create_initial_state
    from diagnostics import ClimateDiagnostics

    print(f"\n=== QUICK BENCHMARK: {days}-day run, {grid_H}×{grid_W} grid ===\n")

    elev = _make_earth_elevation(grid_H, grid_W)
    state = create_initial_state(elev, day_of_year=80.0)

    diag = ClimateDiagnostics(track_history=False)

    MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    monthly_nh: list[float] = []
    monthly_sh: list[float] = []
    monthly_T55: list[float] = []
    monthly_T65: list[float] = []
    monthly_T75: list[float] = []
    monthly_T85: list[float] = []

    elapsed_days = 0
    month_i = 0
    t0 = time.time()

    while elapsed_days < days:
        mdays = MONTH_DAYS[month_i % 12]
        run_days = min(mdays, days - elapsed_days)

        current = state
        for _ in range(run_days):
            current, _ = simulate_step(current, days=1.0)
        state = current
        elapsed_days += run_days

        snap = diag.analyze_snapshot(state)
        monthly_nh.append(snap.get("ice_frac_nh", 0.0))
        monthly_sh.append(snap.get("ice_frac_sh", 0.0))

        T = state.temperature
        if T is not None:
            monthly_T55.append(_zonal_T_at(T, 55.0))
            monthly_T65.append(_zonal_T_at(T, 65.0))
            monthly_T75.append(_zonal_T_at(T, 75.0))
            monthly_T85.append(_zonal_T_at(T, 85.0))

        month_i += 1
        if elapsed_days >= days:
            break

    elapsed = time.time() - t0
    print(f"  Simulation completed in {elapsed:.1f}s ({elapsed/days*365:.0f}s/year equivalent)\n")

    MNAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    print(f"{'Month':>5} | {'NHice%':>7} | {'SHice%':>7} | {'T55N':>7} | {'T65N':>7} | {'T75N':>7} | {'T85N':>7}")
    print("-" * 66)
    for i in range(len(monthly_nh)):
        ni = monthly_nh[i] * 100
        si = monthly_sh[i] * 100
        t55 = monthly_T55[i] if i < len(monthly_T55) else float("nan")
        t65 = monthly_T65[i] if i < len(monthly_T65) else float("nan")
        t75 = monthly_T75[i] if i < len(monthly_T75) else float("nan")
        t85 = monthly_T85[i] if i < len(monthly_T85) else float("nan")
        print(f"{MNAMES[i%12]:>5} | {ni:>7.1f} | {si:>7.1f} | {t55:>+7.1f} | {t65:>+7.1f} | {t75:>+7.1f} | {t85:>+7.1f}")
    print("-" * 66)

    # Final snapshot
    snap = diag.analyze_snapshot(state)
    gmt = snap["global_mean_temp"]
    print(f"\n  Global mean T: {gmt:.1f}K ({gmt-273.15:.1f}°C)  [Earth: 288K / 15°C]")
    print(f"  NH gradient:  {snap['gradient_north']:.1f}K  [Earth: 45-60K]")
    print(f"  SH gradient:  {snap['gradient_south']:.1f}K  [Earth: 45-60K]")

    if monthly_nh:
        print(f"  NH ice (final month): {monthly_nh[-1]*100:.1f}%  [Earth: ~5%]")
    if monthly_T65:
        T65_final = monthly_T65[-1]
        print(f"  T65N (final month): {T65_final:+.1f}°C  [Earth: ~-2°C]")
        if T65_final < -20:
            print("    ! T65N still very cold — check T_base at 65N or AMOC bonus")
        elif T65_final > 10:
            print("    ! T65N too warm — check AMOC bonus (may be too large)")
        else:
            print("    ok T65N in reasonable range")
    if monthly_T75:
        T75_final = monthly_T75[-1]
        print(f"  T75N (final month): {T75_final:+.1f}°C  [Earth: ~-12°C]")
    print()

    # Key parameter summary
    from simulate import simulate_step as _ss
    import inspect
    sig = inspect.signature(_ss)
    key_params = ["ice_freeze_temp", "ice_melt_temp", "ice_albedo_strength",
                  "heat_transport_coeff", "thermal_diffusion", "polar_cooling_scale"]
    print("  Key simulation parameters (defaults):")
    for p in key_params:
        default = sig.parameters[p].default if p in sig.parameters else "N/A"
        print(f"    {p}: {default}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fast climate diagnostic tool")
    parser.add_argument("--days", type=int, default=90,
                        help="Number of simulation days (default: 90)")
    parser.add_argument("--tbase-only", action="store_true",
                        help="Only print T_base profile, skip simulation")
    parser.add_argument("--H", type=int, default=60, help="Grid height (default: 60)")
    parser.add_argument("--W", type=int, default=120, help="Grid width (default: 120)")
    args = parser.parse_args()

    print_tbase_profile()

    if not args.tbase_only:
        run_quick_benchmark(days=args.days, grid_H=args.H, grid_W=args.W)


if __name__ == "__main__":
    main()
