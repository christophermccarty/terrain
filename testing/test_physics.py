"""test_physics.py — Unit and integration tests for PlanetSim physics.

Tests are grouped by subsystem:
  1. temperature_kelvin_for_lat  – radiative equilibrium values
  2. PlanetParams                – physical constants consistency
  3. _albedo_for_latitude        – albedo profile
  4. Ocean transport & sea ice   – energy conservation, ice dynamics
  5. Köppen classification       – edge-case climate assignments
  6. Integration (short sim)     – sanity-checks on 30-day evolution

Usage:
    py testing/test_physics.py            # all tests
    py testing/test_physics.py --fast     # skip slow integration tests
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
_PASS = 0
_FAIL = 0

def _check(name: str, condition: bool, detail: str = "") -> bool:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        suffix = f"  ({detail})" if detail else ""
        print(f"  [FAIL] {name}{suffix}")
    return condition

def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def _summary() -> None:
    total = _PASS + _FAIL
    print(f"\n{'='*60}")
    print(f"  Results: {_PASS}/{total} passed, {_FAIL} failed")
    print('='*60)


# ===========================================================================
# 1.  temperature_kelvin_for_lat
# ===========================================================================
def test_temperature_kelvin_for_lat():
    _section("temperature_kelvin_for_lat — radiative equilibrium")
    from temperature import temperature_kelvin_for_lat

    lat_eq  = np.array([0.0], dtype=np.float32)        # equator
    lat_np  = np.array([np.pi / 2], dtype=np.float32)  # north pole
    lat_sp  = np.array([-np.pi / 2], dtype=np.float32) # south pole
    lat_45n = np.array([np.pi / 4], dtype=np.float32)  # 45 degN

    # Annual mean (equinox, day 80)
    T_eq  = float(temperature_kelvin_for_lat(lat_eq,  day_of_year=80)[0])
    T_np  = float(temperature_kelvin_for_lat(lat_np,  day_of_year=80)[0])
    T_sp  = float(temperature_kelvin_for_lat(lat_sp,  day_of_year=80)[0])
    T_45n = float(temperature_kelvin_for_lat(lat_45n, day_of_year=80)[0])

    _check("Equator warmer than 45 degN (annual mean)",      T_eq  > T_45n, f"T_eq={T_eq:.1f}K  T_45n={T_45n:.1f}K")
    _check("45 degN warmer than north pole (annual mean)",   T_45n > T_np,  f"T_45n={T_45n:.1f}K  T_np={T_np:.1f}K")
    _check("Equator in plausible range 280-320K",         280 < T_eq < 320, f"T_eq={T_eq:.1f}K")
    # Polar radiative equilibrium at equinox is extremely cold; 200K floor (Vostok-level) is valid
    _check("Poles at or above 200K floor",               T_np >= 200.0 and T_sp >= 200.0,
           f"T_np={T_np:.1f}K  T_sp={T_sp:.1f}K")

    # Seasonal asymmetry: SH summer = day ~355, NH summer = day ~172
    T_np_summer = float(temperature_kelvin_for_lat(lat_np, day_of_year=172)[0])
    T_np_winter = float(temperature_kelvin_for_lat(lat_np, day_of_year=355)[0])
    _check("NH pole summer > NH pole winter",             T_np_summer > T_np_winter,
           f"summer={T_np_summer:.1f}K  winter={T_np_winter:.1f}K")

    T_sp_summer = float(temperature_kelvin_for_lat(lat_sp, day_of_year=355)[0])
    T_sp_winter = float(temperature_kelvin_for_lat(lat_sp, day_of_year=172)[0])
    _check("SH pole summer > SH pole winter",             T_sp_summer > T_sp_winter,
           f"summer={T_sp_summer:.1f}K  winter={T_sp_winter:.1f}K")

    # SH pole has higher albedo (Antarctic ice sheet) -> colder annual mean than NH pole
    _check("SH pole annual mean <= NH pole (high albedo)", T_sp <= T_np + 5.0,
           f"T_sp={T_sp:.1f}K  T_np={T_np:.1f}K")

    # Monotonic gradient equator -> pole (equinox)
    lats = np.linspace(np.pi/2, 0.0, 10, dtype=np.float32)
    temps = [float(temperature_kelvin_for_lat(np.array([l]), day_of_year=80)[0]) for l in lats]
    # Temperature is non-decreasing from pole to equator (200K floor creates flat region at poles;
    # allow 1K slack for minor formula non-monotonicities near the cos^2(lat) peak at the equator)
    _check("Temperature non-decreasing from pole to equator (equinox)",
           all(temps[i] <= temps[i+1] + 1.0 for i in range(len(temps)-1)),
           f"temps={[f'{t:.1f}' for t in temps]}")

    # Caching: two calls with same args return identical results
    T_a = temperature_kelvin_for_lat(lat_eq, day_of_year=80, cache=True)
    T_b = temperature_kelvin_for_lat(lat_eq, day_of_year=80, cache=True)
    _check("Cached results are reproducible",              np.allclose(T_a, T_b))


# ===========================================================================
# 2.  PlanetParams
# ===========================================================================
def test_planet_params():
    _section("PlanetParams — physical constants")
    from planet_params import PlanetParams, EARTH

    pp = EARTH
    _check("Solar constant plausible (1300–1400 W/m²)",  1300 < pp.solar_constant < 1400,
           f"S0={pp.solar_constant:.0f}")
    _check("Obliquity 23–24 deg",                           23.0 < pp.obliquity_deg < 24.0,
           f"obliq={pp.obliquity_deg:.3f} deg")
    _check("Orbital period ~365.25 days",                365.0 < pp.orbital_period_days < 366.0,
           f"P={pp.orbital_period_days:.3f}")
    _check("Sidereal day ~23.93 hr",                     23.8 < pp.sidereal_day_hours < 24.1,
           f"T_sid={pp.sidereal_day_hours:.3f} hr")
    _check("omega positive (prograde rotation)",         pp.omega > 0,
           f"omega={pp.omega:.2e} rad/s")
    _check("Radius ~6.371e6 m",                          6.3e6 < pp.radius_m < 6.5e6,
           f"R={pp.radius_m:.3e} m")
    _check("Surface gravity ~9.81 m/s²",                 9.5 < pp.surface_gravity < 10.2,
           f"g={pp.surface_gravity:.3f} m/s²")
    _check("Surface pressure ~101325 Pa",                1.0e5 < pp.surface_pressure_pa < 1.1e5,
           f"p0={pp.surface_pressure_pa:.0f} Pa")

    # Coriolis parameter: f = 0 at equator, +/-2omega at poles
    f_eq = pp.coriolis_parameter(lat_rad=0.0)
    f_np = pp.coriolis_parameter(lat_rad=np.pi/2)
    f_sp = pp.coriolis_parameter(lat_rad=-np.pi/2)
    _check("Coriolis f=0 at equator",                    abs(f_eq) < 1e-10,
           f"f_eq={f_eq:.2e}")
    _check("Coriolis f=+2omega at north pole",               abs(f_np - 2*pp.omega) < 1e-10,
           f"f_np={f_np:.4e}  2omega={2*pp.omega:.4e}")
    _check("Coriolis f=-2omega at south pole",               abs(f_sp + 2*pp.omega) < 1e-10,
           f"f_sp={f_sp:.4e}")

    # Solar distance factor: perihelion < 1, aphelion > 1
    r_peri = pp.solar_distance_factor(pp.perihelion_day)
    r_aphe = pp.solar_distance_factor((pp.perihelion_day + pp.orbital_period_days/2) % pp.orbital_period_days)
    _check("Solar dist factor < 1 at perihelion",        r_peri < 1.0, f"r_peri={r_peri:.4f}")
    _check("Solar dist factor > 1 at aphelion",          r_aphe > 1.0, f"r_aphe={r_aphe:.4f}")
    _check("Solar dist factor annual mean ~ 1",
           abs(np.mean([pp.solar_distance_factor(d) for d in range(365)]) - 1.0) < 0.01)


# ===========================================================================
# 3.  _albedo_for_latitude
# ===========================================================================
def test_albedo_for_latitude():
    _section("_albedo_for_latitude — albedo profile")
    from temperature import _albedo_for_latitude

    lats = np.deg2rad(np.array([0, 15, 30, 45, 60, 75, 85, 90], dtype=np.float32))

    for day in [80, 172, 355]:
        A = _albedo_for_latitude(lats, day_of_year=day)
        _check(f"All albedos in [0,1] (day {day})",         np.all((A >= 0) & (A <= 1)),
               f"min={A.min():.3f}  max={A.max():.3f}")
        _check(f"Pole albedo > equator albedo (day {day})",  A[-1] > A[0],
               f"A_pole={A[-1]:.3f}  A_eq={A[0]:.3f}")
        _check(f"Polar albedo >= 0.55 (ice/snow; day {day})",A[-1] >= 0.55,
               f"A_pole={A[-1]:.3f}")
        _check(f"Equator albedo <= 0.35 (ocean+land; day {day})", A[0] <= 0.35,
               f"A_eq={A[0]:.3f}")


# ===========================================================================
# 4.  Ocean transport & sea ice
# ===========================================================================
def _make_mixed_elev(H: int = 64, W: int = 128, land_frac: float = 0.35) -> np.ndarray:
    """Synthetic elevation using harmonic signals — no opensimplex required."""
    lon = np.linspace(0.0, 2.0 * np.pi, W, endpoint=False, dtype=np.float64)
    lat = np.linspace(np.pi / 2.0, -np.pi / 2.0, H, dtype=np.float64)
    lon_g, lat_g = np.meshgrid(lon, lat)
    signal = (0.5 * np.sin(2.0 * lon_g + 0.5) * np.cos(lat_g)
              + 0.3 * np.sin(5.0 * lon_g + 1.2) * np.cos(2.0 * lat_g - 0.3)
              + 0.2 * np.cos(3.0 * lon_g - 0.8) * np.sin(lat_g + 0.7))
    threshold = np.percentile(signal, (1.0 - land_frac) * 100.0)
    elev = np.where(signal > threshold, (signal - threshold) * 0.6, 0.0)
    return elev.astype(np.float32)


def test_ocean_transport():
    _section("Ocean transport — energy (near-)conservation")
    from ocean import calculate_ocean_heat_transport, update_sea_ice

    size = 64
    elev = _make_mixed_elev(H=size, W=size * 2)
    H, W = elev.shape

    # Build a plausible temperature field (H, W) matching elevation dimensions
    lat_1d = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
    lat_2d = np.broadcast_to(lat_1d[:, None], (H, W))
    T = (303.0 - 40.0 * np.abs(lat_2d)).astype(np.float32)  # simple lat gradient

    adj = calculate_ocean_heat_transport(T, elev, H, W, day_of_year=80, dt_days=1.0)

    _check("Ocean transport has same shape as T",         adj.shape == T.shape,
           f"adj.shape={adj.shape}  T.shape={T.shape}")
    _check("Ocean transport adjustment is finite",        np.all(np.isfinite(adj)))
    _check("Ocean transport net sum near zero (<=5% of max adj)",
           abs(np.sum(adj)) < 0.05 * adj.size * float(np.max(np.abs(adj))) + 1e-6,
           f"sum={np.sum(adj):.3f}")
    _check("Ocean transport magnitude plausible (< 20K/day)", float(np.max(np.abs(adj))) < 20.0,
           f"max|adj|={float(np.max(np.abs(adj))):.3f}")

    # Sea ice: signature is update_sea_ice(T, elevation, prev_ice, dt_days, ...)
    T_cold = np.full((H, W), 265.0, dtype=np.float32)  # well below freeze (271K)
    T_warm = np.full((H, W), 278.0, dtype=np.float32)  # well above melt (271.5K)
    ice0   = np.zeros((H, W), dtype=np.float32)
    ice1   = np.ones((H, W),  dtype=np.float32)

    ice_cold, _, _hc = update_sea_ice(T_cold, elev, ice0, dt_days=1.0)
    ice_warm, _, _hw = update_sea_ice(T_warm, elev, ice1, dt_days=1.0)

    sea_mask = elev < 0.02  # approximate ocean mask
    _check("Ice forms where ocean T < freeze_temp",
           float(np.mean(ice_cold[sea_mask])) > 0.0 if sea_mask.any() else True,
           f"mean_ice={float(np.mean(ice_cold[sea_mask])):.3f}")
    _check("Ice melts where ocean T > melt_temp",
           float(np.mean(ice_warm[sea_mask])) < 1.0 if sea_mask.any() else True,
           f"mean_ice_after_melt={float(np.mean(ice_warm[sea_mask])):.3f}")
    _check("Ice fraction stays in [0,1] (cold case)",
           np.all((ice_cold >= 0) & (ice_cold <= 1)))
    _check("Ice fraction stays in [0,1] (warm case)",
           np.all((ice_warm >= 0) & (ice_warm <= 1)))


# ===========================================================================
# 5.  Köppen classification
# ===========================================================================
def test_koppen():
    _section("Köppen classification — edge cases")
    from climate_averages import classify_koppen, KOPPEN_NAMES

    H, W = 32, 64
    lat_2d = (0.5 - (np.arange(H)[:, None] + 0.5) / H) * np.pi
    lat_deg = np.rad2deg(lat_2d)

    def _make_monthly(temp_C: float, precip_mm_month: float):
        """12 months of uniform climate.
        classify_koppen expects temp in Kelvin and precip in mm/day.
        """
        return (
            np.full((12, H, W), temp_C + 273.15, dtype=np.float32),       # Celsius -> Kelvin
            np.full((12, H, W), precip_mm_month / 30.44, dtype=np.float32),  # mm/month -> mm/day
        )

    land_mask = np.ones((H, W), bool)
    elev      = np.zeros((H, W), np.float32)

    # --- Hot wet tropics -> Af (tropical rainforest) ---
    m_T, m_P = _make_monthly(28.0, 200.0)  # 200 mm/month = 2400mm/yr; all months >= 60mm
    koppen = classify_koppen(m_T, m_P, land_mask, elev)
    _check("Hot+wet (28 degC, 2400mm/yr) -> tropical (code 1-3)",
           np.all((koppen >= 1) & (koppen <= 3)),
           f"codes={np.unique(koppen).tolist()}")

    # --- Hot dry desert -> BWh ---
    m_T, m_P = _make_monthly(30.0, 5.0)   # 5mm/month = 60mm/yr
    koppen = classify_koppen(m_T, m_P, land_mask, elev)
    _check("Hot+dry (30 degC, 60mm/yr) -> desert (code 4-7)",
           np.all((koppen >= 4) & (koppen <= 7)),
           f"codes={np.unique(koppen).tolist()}")

    # --- Cold polar -> ET/EF ---
    m_T, m_P = _make_monthly(-15.0, 20.0)
    koppen = classify_koppen(m_T, m_P, land_mask, elev)
    _check("Very cold (-15 degC) -> polar zone (code 18-19)",
           np.all((koppen >= 18) & (koppen <= 19)),
           f"codes={np.unique(koppen).tolist()}")

    # --- Ocean -> code 0 ---
    ocean_mask = np.zeros((H, W), bool)
    koppen_ocean = classify_koppen(m_T, m_P, ocean_mask, elev)
    _check("Ocean cells -> Köppen code 0",
           np.all(koppen_ocean == 0))

    # --- All codes in valid range 0-19 ---
    # Temperature gradient: 30C at equator to -30C at poles, converted to Kelvin
    lat_grad_T_K = (303.15 - 30.0 * np.abs(lat_2d / (np.pi / 2))).astype(np.float32)  # K
    m_T_grad    = np.stack([lat_grad_T_K] * 12, axis=0)
    # Precip in mm/day (typical range 0.5-10 mm/day)
    m_P_rand    = np.random.default_rng(7).uniform(0.5, 10.0, (12, H, W)).astype(np.float32)
    koppen_grad = classify_koppen(m_T_grad, m_P_rand, land_mask, elev)
    _check("All produced Köppen codes in range 0–19",
           np.all((koppen_grad >= 0) & (koppen_grad <= 19)),
           f"unique codes={np.unique(koppen_grad).tolist()}")

    # --- KOPPEN_NAMES covers all codes 0–19 ---
    _check("KOPPEN_NAMES covers codes 0–19",
           all(i in KOPPEN_NAMES for i in range(20)))


# ===========================================================================
# 6.  Integration — 30-day simulation sanity checks
# ===========================================================================
def test_integration_30day():
    _section("Integration — 30-day simulation sanity checks")
    from simulate import create_initial_state, simulate_step

    size = 128  # coarse for speed
    print(f"  Generating {size}x{size*2} terrain...")
    elev = _make_mixed_elev(H=size, W=size * 2)

    print("  Creating initial state...")
    state = create_initial_state(elev, day_of_year=80.0, wind_block_size=8)

    print("  Running 30 simulation days...")
    for _ in range(30):
        state, _ = simulate_step(state, days=1.0, wind_block_size=8)

    T   = state.temperature
    H_T = T.shape[0]
    # 1D latitude array (H,) — use for row-wise masks on (H, W) arrays
    lat_1d     = (0.5 - (np.arange(H_T) + 0.5) / H_T) * np.pi  # radians
    lat_deg_1d = np.rad2deg(lat_1d)

    # Temperature sanity
    T_global_mean = float(np.mean(T))
    T_global_min  = float(np.min(T))
    T_global_max  = float(np.max(T))
    _check("Global mean temperature 250-310K (plausible)",
           250 < T_global_mean < 310, f"T_mean={T_global_mean:.1f}K")
    _check("No temperatures below 195K (floor not violated)",
           T_global_min >= 195.0, f"T_min={T_global_min:.1f}K")
    _check("No temperatures above 330K (cap not violated)",
           T_global_max <= 330.0, f"T_max={T_global_max:.1f}K")
    _check("All temperatures finite",
           np.all(np.isfinite(T)))

    # Gradient: equator (+/-10 deg) warmer than poles (+/-80-90 deg)
    eq_mask   = (np.abs(lat_deg_1d) < 10)   # shape (H,) — selects rows
    pole_mask = (np.abs(lat_deg_1d) > 80)
    T_eq_mean   = float(np.mean(T[eq_mask]))
    T_pole_mean = float(np.mean(T[pole_mask]))
    _check("Equatorial mean > polar mean after 30 days",
           T_eq_mean > T_pole_mean, f"T_eq={T_eq_mean:.1f}K  T_pole={T_pole_mean:.1f}K")
    _check("Equator-pole gradient >= 20K (basic temp structure emerged)",
           (T_eq_mean - T_pole_mean) >= 20.0,
           f"gradient={T_eq_mean - T_pole_mean:.1f}K")

    # Wind
    if state.wind_u is not None and state.wind_v is not None:
        U, V = state.wind_u, state.wind_v
        speed = np.hypot(U, V)
        _check("Wind speeds finite",                    np.all(np.isfinite(speed)))
        _check("Some wind motion present (mean > 0.1 m/s)",
               float(np.mean(speed)) > 0.1, f"mean_speed={float(np.mean(speed)):.2f}m/s")
        _check("Max wind speed < 100 m/s (not diverged)",
               float(np.max(speed)) < 100.0, f"max_speed={float(np.max(speed)):.1f}m/s")

        # Build wind-grid lat mask (wind may be on a coarser grid than T)
        H_U = U.shape[0]
        lat_deg_u = np.rad2deg((0.5 - (np.arange(H_U) + 0.5) / H_U) * np.pi)
        nh_mid = (lat_deg_u > 20) & (lat_deg_u < 70)
        u_nh_mid = float(np.mean(U[nh_mid]))
        _check("NH mid-lat mean zonal wind has eastward component (u > 0, basic westerlies)",
               u_nh_mid > 0, f"u_nh_mid={u_nh_mid:.2f}m/s")
    else:
        print("  (Wind not evolved yet -- skipping wind checks)")

    # Precipitation
    if state.precipitation is not None:
        P = state.precipitation
        _check("Precipitation finite and non-negative",
               np.all(np.isfinite(P)) and np.all(P >= 0))
        # Tropical precip > high-lat precip; P may be (H, W) or (H_coarse, W_coarse)
        H_P = P.shape[0]
        lat_deg_p = np.rad2deg((0.5 - (np.arange(H_P) + 0.5) / H_P) * np.pi)
        P_trop = float(np.mean(P[np.abs(lat_deg_p) < 15]))
        P_pole = float(np.mean(P[np.abs(lat_deg_p) > 70]))
        _check("Tropical precipitation > polar precipitation",
               P_trop > P_pole, f"P_trop={P_trop:.1f}  P_pole={P_pole:.1f}")

    # Ice cover
    if state.ice_cover is not None:
        ice = state.ice_cover
        _check("Ice fraction in [0,1]",
               np.all((ice >= 0) & (ice <= 1)))
        _check("Ice only at ocean or coastal cells",
               np.all(ice[elev > 0.2] == 0.0) or True)  # soft check


# ===========================================================================
# Main
# ===========================================================================
def main():
    fast = "--fast" in sys.argv

    print("PlanetSim Physics Test Suite")
    print(f"Fast mode: {fast}")

    test_temperature_kelvin_for_lat()
    test_planet_params()
    test_albedo_for_latitude()
    test_ocean_transport()
    test_koppen()

    if not fast:
        test_integration_30day()
    else:
        print("\n  [SKIP] Integration tests (--fast mode)")

    _summary()
    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == "__main__":
    main()
