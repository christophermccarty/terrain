"""Climate diagnostics and benchmarking tool.

Calculates physical metrics from the simulation state to compare against
Earth reference values. Useful for tuning physics parameters.

Now includes time-series tracking and per-component analysis.
"""

import numpy as np
from typing import Dict, Any, List, Optional
import json
import csv
from pathlib import Path
from datetime import datetime

# Earth Reference Values (Approximate)
EARTH_REF = {
    "global_mean_temp": 288.0,  # K (15°C)
    "equator_temp": 300.0,      # K (27°C)
    "pole_temp_winter": 240.0,  # K (-33°C)
    "pole_temp_summer": 273.0,  # K (0°C)
    "pole_n_annual": 239.0,     # K (~-34°C, annual mean NH pole)
    "pole_s_annual": 228.0,     # K (~-45°C, annual mean SH pole — colder due to ice sheet)
    "equator_pole_gradient": 45.0,   # K minimum (Earth range ~45-60K)
    "equator_pole_gradient_max": 60.0, # K maximum
    "mean_precip": 2.7,         # mm/day
    "albedo_global": 0.30,      # unitless
    # Surface wind speed references (annual zonal mean, m/s, positive = westerly)
    "wind_doldrums_speed": 2.5,   # 0-5°, weak convergence zone
    "wind_trades_speed": 6.5,     # 5-20°, trade winds (easterly, ~5-8 m/s)
    "wind_subtrop_speed": 3.5,    # 20-30°, horse latitudes (weaker)
    "wind_midlat_speed": 8.0,     # 30-60°, storm track westerlies (~6-10 m/s)
    "wind_polar_speed": 4.0,      # 60-90°, polar easterlies (~3-5 m/s)
    "wind_trades_u": -6.5,        # easterly (negative u)
    "wind_midlat_u": +7.0,        # westerly (positive u)
    "wind_polar_u": -3.5,         # easterly (negative u)
    # Sea ice extent references (annual mean)
    "ice_frac_nh": 0.05,          # NH ice cover fraction (~5% of NH cells, annual mean)
    "ice_frac_sh": 0.05,          # SH ice cover fraction (~5% of SH cells, annual mean)
    "ice_edge_n": 72.0,           # NH equatorward ice edge (°N, annual mean ~72°N)
    "ice_edge_s": 62.0,           # SH equatorward ice edge (|°S|, annual mean ~62°S)
}


def _jet_latitudes_from_profiles(
    lat_deg: np.ndarray,
    zspeed: np.ndarray,
    zu: np.ndarray,
) -> tuple[float, float, float, float]:
    """Locate jet cores from mid-lat westerlies instead of raw polar speed maxima."""
    lat_deg = np.asarray(lat_deg, dtype=np.float32)
    zspeed = np.asarray(zspeed, dtype=np.float32)
    zu = np.asarray(zu, dtype=np.float32)
    if lat_deg.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    westerly_core = np.clip(zu, 0.0, None)
    for _ in range(2):
        pad = np.pad(westerly_core, 1, mode="edge")
        westerly_core = (0.25 * pad[:-2] + 0.50 * pad[1:-1] + 0.25 * pad[2:]).astype(np.float32)

    midlat_window = ((np.abs(lat_deg) >= 15.0) & (np.abs(lat_deg) <= 75.0)).astype(np.float32)
    jet_metric = westerly_core * midlat_window
    if float(np.max(jet_metric)) <= 1e-6:
        jet_metric = zspeed * midlat_window

    n_mask = lat_deg > 0.0
    s_mask = lat_deg < 0.0
    jet_lat_n = float(lat_deg[np.argmax(np.where(n_mask, jet_metric, -1e9))]) if np.any(n_mask) else 0.0
    jet_lat_s = float(lat_deg[np.argmax(np.where(s_mask, jet_metric, -1e9))]) if np.any(s_mask) else 0.0
    jet_speed_n = float(np.max(np.where(n_mask, jet_metric, 0.0))) if np.any(n_mask) else 0.0
    jet_speed_s = float(np.max(np.where(s_mask, jet_metric, 0.0))) if np.any(s_mask) else 0.0
    return jet_lat_n, jet_lat_s, jet_speed_n, jet_speed_s


class ClimateDiagnostics:
    def __init__(self, track_history: bool = True):
        self.history: List[Dict[str, Any]] = []
        self.track_history = track_history
        self.component_history: List[Dict[str, Any]] = []  # Per-component contributions
        self.total_days: float = 0.0  # Track total simulation time (doesn't wrap)

    @staticmethod
    def _lat_deg_for_H(H: int) -> np.ndarray:
        """Row-centered latitudes in degrees (north positive)."""
        lat = (0.5 - (np.arange(int(H), dtype=np.float32) + 0.5) / float(H)) * np.pi
        return np.rad2deg(lat).astype(np.float32)

    @staticmethod
    def _band_mean(weights: np.ndarray, x_by_lat: np.ndarray, lat_deg: np.ndarray, *, lo: float, hi: float, hemi: str | None = None) -> float:
        """Area-weighted mean of a zonal-mean profile within a latitude band."""
        if hemi == "N":
            m = (lat_deg >= lo) & (lat_deg < hi)
        elif hemi == "S":
            m = (lat_deg <= -lo) & (lat_deg > -hi)
        else:
            m = (np.abs(lat_deg) >= lo) & (np.abs(lat_deg) < hi)
        if not np.any(m):
            return 0.0
        w = weights[m]
        w = w / (float(np.sum(w)) + 1e-12)
        return float(np.sum(x_by_lat[m] * w))

    @staticmethod
    def _expect_sign(name: str, value: float, *, sign: int, tol: float = 0.0) -> Dict[str, Any]:
        """Return pass/fail for sign expectation (sign: +1 or -1)."""
        s = 1 if value >= 0 else -1
        ok = (abs(value) <= tol) or (s == int(np.sign(sign)))
        return {"name": name, "value": float(value), "expect": ">=0" if sign > 0 else "<=0", "ok": bool(ok)}

    def analyze_snapshot(self, state) -> Dict[str, Any]:
        """Analyze a single simulation state with comprehensive diagnostics."""
        if state.temperature is None:
            return {}

        # Use air temperature (2m) for surface climate diagnostics when available.
        # Fall back to state.temperature (T_sst) for old saves that lack the field.
        T = state.air_temperature if state.air_temperature is not None else state.temperature
        H, W = T.shape
        
        # Latitude array (radians)
        lat = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
        # Area weights (cosine of latitude) for global averages
        weights = np.cos(lat)
        weights /= np.sum(weights)
        
        # Zonal Means (Average across longitude)
        T_zonal = np.mean(T, axis=1)
        
        # 1. Global Mean Temperature (Area Weighted)
        global_mean_T = np.sum(T_zonal * weights)
        
        # 2. Equator vs Pole Temps
        eq_idx = H // 2
        pole_n_idx = 0
        pole_s_idx = -1
        
        T_equator = T_zonal[eq_idx]
        T_pole_n = T_zonal[pole_n_idx]
        T_pole_s = T_zonal[pole_s_idx]
        
        # 3. Gradients
        grad_n = T_equator - T_pole_n
        grad_s = T_equator - T_pole_s
        
        # 4. Extremes
        T_min = np.min(T)
        T_max = np.max(T)
        T_std = np.std(T)
        
        # 5. Precipitation Statistics
        if state.precipitation is not None:
            P = state.precipitation
            precip_mean = np.sum(np.mean(P, axis=1) * weights)
            precip_max = np.max(P)
            precip_min = np.min(P)
            precip_std = np.std(P)
            # Precipitation by latitude
            P_zonal = np.mean(P, axis=1)
            precip_equator = P_zonal[eq_idx]
            precip_pole_n = P_zonal[pole_n_idx]
            precip_pole_s = P_zonal[pole_s_idx]
        else:
            precip_mean = precip_max = precip_min = precip_std = 0.0
            precip_equator = precip_pole_n = precip_pole_s = 0.0
        
        # 6. Wind Statistics
        if state.wind_u is not None and state.wind_v is not None:
            u = state.wind_u
            v = state.wind_v
            wind_speed = np.sqrt(u**2 + v**2)
            wind_mean = np.sum(np.mean(wind_speed, axis=1) * weights)
            wind_max = np.max(wind_speed)
            # Zonal (east-west) and meridional (north-south) components
            u_zonal_mean = np.sum(np.mean(u, axis=1) * weights)
            v_merid_mean = np.sum(np.mean(v, axis=1) * weights)

            # Extra wind diagnostics for tuning (Earth-like circulation targets)
            lat_deg = self._lat_deg_for_H(H)
            abs_lat = np.abs(lat_deg)
            zspeed = np.mean(wind_speed, axis=1)  # zonal-mean speed by latitude
            zu = np.mean(u, axis=1)  # zonal-mean u by latitude (sign matters: easterly vs westerly)
            zv = np.mean(v, axis=1)  # zonal-mean v by latitude (sign matters: equatorward vs poleward)

            def _band_mean(lo: float, hi: float) -> float:
                m = (abs_lat >= lo) & (abs_lat < hi)
                if not np.any(m):
                    return 0.0
                w = weights[m]
                w = w / (np.sum(w) + 1e-12)
                return float(np.sum(zspeed[m] * w))

            wind_eq_mean = _band_mean(0.0, 5.0)          # doldrums target: low
            wind_trade_mean = _band_mean(5.0, 20.0)      # trades target: higher
            wind_midlat_mean = _band_mean(30.0, 60.0)    # storm track target: higher/variable
            wind_subtrop_mean = _band_mean(20.0, 35.0)   # horse latitudes target: relatively lower
            wind_polar_mean = _band_mean(60.0, 90.0)     # polar cell surface winds (often easterly)
            wind_trade_ratio = float(wind_trade_mean / (wind_eq_mean + 1e-6))

            def _band_mean_signed(arr: np.ndarray, lo: float, hi: float, *, hemi: str | None = None) -> float:
                if hemi == "N":
                    m = (lat_deg >= lo) & (lat_deg < hi)
                elif hemi == "S":
                    m = (lat_deg <= -lo) & (lat_deg > -hi)
                else:
                    m = (abs_lat >= lo) & (abs_lat < hi)
                if not np.any(m):
                    return 0.0
                w = weights[m]
                w = w / (np.sum(w) + 1e-12)
                return float(np.sum(arr[m] * w))

            # Zonal wind signs by regime (Earth-like targets):
            # - Trades (5-20): easterly => negative u
            # - Mid-lats (30-60): westerly => positive u
            # - Polar (60-90): easterly => negative u
            wind_u_trade_mean = _band_mean_signed(zu, 5.0, 20.0)
            wind_u_midlat_mean = _band_mean_signed(zu, 30.0, 60.0)
            wind_u_polar_mean = _band_mean_signed(zu, 60.0, 90.0)

            # Meridional return flow / convergence diagnostics:
            # Hadley near-surface flow should be equatorward in both hemispheres.
            wind_v_hadley_n_mean = _band_mean_signed(zv, 5.0, 25.0, hemi="N")  # expect negative
            wind_v_hadley_s_mean = _band_mean_signed(zv, 5.0, 25.0, hemi="S")  # expect positive
            # Polar easterlies flow equatorward from poles.
            wind_v_polar_n_mean = _band_mean_signed(zv, 60.0, 85.0, hemi="N")  # expect negative
            wind_v_polar_s_mean = _band_mean_signed(zv, 60.0, 85.0, hemi="S")  # expect positive

            # Simple ITCZ convergence proxy near equator (zonal-mean dv/dy should be negative near 0°).
            # Use derivative w.r.t. latitude degrees (lat_deg runs north→south).
            dv_dlat = np.gradient(zv, lat_deg)  # per degree
            eq_idx = int(np.argmin(np.abs(lat_deg)))
            lo_i = max(0, eq_idx - 2)
            hi_i = min(len(dv_dlat), eq_idx + 3)
            wind_itcz_conv = float(-np.mean(dv_dlat[lo_i:hi_i]))  # larger => stronger convergence

            wind_jet_lat_n, wind_jet_lat_s, wind_jet_speed_n, wind_jet_speed_s = (
                _jet_latitudes_from_profiles(lat_deg, zspeed, zu)
            )

            # Eddy kinetic energy (variance around zonal mean)
            u_prime = u - np.mean(u, axis=1, keepdims=True)
            v_prime = v - np.mean(v, axis=1, keepdims=True)
            wind_eke_mean = float(0.5 * np.mean(u_prime**2 + v_prime**2))

            # Divergence/vorticity RMS (periodic in lon)
            dudx = 0.5 * (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1))
            dvdx = 0.5 * (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1))
            dudy = np.gradient(u, axis=0)
            dvdy = np.gradient(v, axis=0)
            wind_div_rms = float(np.sqrt(np.mean((dudx + dvdy) ** 2)))
            wind_vort_rms = float(np.sqrt(np.mean((dvdx - dudy) ** 2)))

            # Composite circulation score (higher is better / more Earth-like).
            # Targets:
            # - Weak equatorial winds (doldrums), stronger trades off-equator
            # - Trades easterly, mid-lats westerly, polar easterly
            # - Subtropical "horse latitudes" relatively calmer than trades
            # - Positive ITCZ convergence proxy (our definition)
            # - Avoid extreme divergence / runaway eddy energy
            def _tanh(x: float) -> float:
                return float(np.tanh(float(x)))

            score = 0.0
            score += 2.0 * _tanh((wind_trade_ratio - 1.0) / 0.35)          # trades > equator
            score += 1.0 * _tanh((4.0 - wind_eq_mean) / 3.0)               # penalize windy equator
            score += 0.8 * _tanh((wind_trade_mean - wind_subtrop_mean) / 2.0)  # horse lats calmer than trades

            score += 0.8 * _tanh((-wind_u_trade_mean) / 4.0)               # easterly trades (u<0)
            score += 1.8 * _tanh((wind_u_midlat_mean) / 4.0)               # westerly mid-lats (u>0) — emphasize
            score += 0.5 * _tanh((-wind_u_polar_mean) / 3.0)               # polar easterlies (u<0)

            score += 0.6 * _tanh(wind_itcz_conv / 0.10)                    # stronger ITCZ convergence (proxy)

            score += 0.3 * _tanh((wind_eke_mean - 6.0) / 6.0)              # some eddies
            score -= 0.6 * _tanh((wind_div_rms - 0.8) / 0.6)               # too divergent is bad
            score -= 0.3 * _tanh((wind_eke_mean - 30.0) / 10.0)            # too chaotic is bad
        else:
            wind_mean = wind_max = u_zonal_mean = v_merid_mean = 0.0
            wind_eq_mean = wind_trade_mean = wind_midlat_mean = wind_trade_ratio = 0.0
            wind_subtrop_mean = wind_polar_mean = 0.0
            wind_u_trade_mean = wind_u_midlat_mean = wind_u_polar_mean = 0.0
            wind_v_hadley_n_mean = wind_v_hadley_s_mean = 0.0
            wind_v_polar_n_mean = wind_v_polar_s_mean = 0.0
            wind_itcz_conv = 0.0
            wind_jet_lat_n = wind_jet_lat_s = wind_jet_speed_n = wind_jet_speed_s = 0.0
            wind_eke_mean = wind_div_rms = wind_vort_rms = 0.0
            score = 0.0
        
        # 7. Cloud Cover Statistics
        if state.cloud_cover is not None:
            C = state.cloud_cover
            cloud_mean = np.sum(np.mean(C, axis=1) * weights)
            cloud_max = np.max(C)
            cloud_min = np.min(C)
            C_zonal = np.mean(C, axis=1)
            cloud_equator = C_zonal[eq_idx]
            cloud_pole_n = C_zonal[pole_n_idx]
            cloud_pole_s = C_zonal[pole_s_idx]
        else:
            cloud_mean = cloud_max = cloud_min = 0.0
            cloud_equator = cloud_pole_n = cloud_pole_s = 0.0
        
        # 8. Humidity Statistics
        if state.humidity is not None:
            H_hum = state.humidity
            humidity_mean = np.sum(np.mean(H_hum, axis=1) * weights)
            humidity_max = np.max(H_hum)
            humidity_min = np.min(H_hum)
        else:
            humidity_mean = humidity_max = humidity_min = 0.0

        # 8b. Soil Moisture Statistics
        if state.soil_moisture is not None:
            S_m = state.soil_moisture
            soil_mean = np.sum(np.mean(S_m, axis=1) * weights)
            soil_max = np.max(S_m)
            soil_min = np.min(S_m)
        else:
            soil_mean = soil_max = soil_min = 0.0
        
        # 9. Snow/Ice Statistics
        if state.snow_depth is not None:
            S = state.snow_depth
            snow_mean = np.sum(np.mean(S, axis=1) * weights)
            snow_max = np.max(S)
            snow_cover_fraction = np.sum(S > 0.01) / (H * W)  # Fraction of surface with >1cm snow
        else:
            snow_mean = snow_max = snow_cover_fraction = 0.0
        
        if state.ice_cover is not None:
            I = state.ice_cover
            ice_mean = np.sum(np.mean(I, axis=1) * weights)
            ice_max = np.max(I)
            ice_cover_fraction = np.sum(I > 0.01) / (H * W)  # Fraction of surface with ice
            # Per-hemisphere fractions and equatorward ice-edge latitudes
            lat_deg_1d = np.rad2deg(lat)          # shape (H,), positive = north
            I_zonal = np.mean(I, axis=1)          # zonal-mean ice fraction, shape (H,)
            ICE_THRESH = 0.1                       # zonal row counts as "iced" above this
            nh_mask = lat_deg_1d > 0.0
            sh_mask = lat_deg_1d < 0.0
            ice_frac_nh = float(np.mean(I_zonal[nh_mask] > ICE_THRESH)) if np.any(nh_mask) else 0.0
            ice_frac_sh = float(np.mean(I_zonal[sh_mask] > ICE_THRESH)) if np.any(sh_mask) else 0.0
            # Equatorward ice edge: most equatorward latitude with significant ice
            nh_ice_lats = lat_deg_1d[nh_mask & (I_zonal > ICE_THRESH)]
            sh_ice_lats = lat_deg_1d[sh_mask & (I_zonal > ICE_THRESH)]
            ice_edge_n = float(np.min(nh_ice_lats)) if len(nh_ice_lats) > 0 else 90.0
            ice_edge_s = float(np.max(sh_ice_lats)) if len(sh_ice_lats) > 0 else -90.0
        else:
            ice_mean = ice_max = ice_cover_fraction = 0.0
            ice_frac_nh = ice_frac_sh = 0.0
            ice_edge_n = 90.0     # no ice → report as all-clear at pole
            ice_edge_s = -90.0
        
        # 10. Albedo Estimate (simplified)
        # Based on snow/ice and cloud cover
        if state.snow_depth is not None and state.cloud_cover is not None:
            snow_albedo = np.clip(state.snow_depth * 0.8, 0.0, 0.8)  # Snow increases albedo
            cloud_albedo = state.cloud_cover * 0.5  # Clouds increase albedo
            base_albedo = np.where(state.elevation > 0.5, 0.2, 0.06)  # Land vs ocean
            albedo_est = base_albedo * (1 - snow_albedo) + snow_albedo + cloud_albedo * 0.3
            albedo_est = np.clip(albedo_est, 0.0, 1.0)
            albedo_mean = np.sum(np.mean(albedo_est, axis=1) * weights)
        else:
            albedo_mean = 0.3  # Default Earth-like
        
        return {
            # Temperature metrics
            "global_mean_temp": global_mean_T,
            "T_equator": T_equator,
            "T_pole_north": T_pole_n,
            "T_pole_south": T_pole_s,
            "gradient_north": grad_n,
            "gradient_south": grad_s,
            "T_min": T_min,
            "T_max": T_max,
            "T_std": T_std,
            
            # Precipitation metrics
            "mean_precip": precip_mean,
            "precip_max": precip_max,
            "precip_min": precip_min,
            "precip_std": precip_std,
            "precip_equator": precip_equator,
            "precip_pole_north": precip_pole_n,
            "precip_pole_south": precip_pole_s,
            
            # Wind metrics
            "wind_mean": wind_mean,
            "wind_max": wind_max,
            "wind_u_zonal_mean": u_zonal_mean,
            "wind_v_merid_mean": v_merid_mean,
            "wind_eq_mean": wind_eq_mean,
            "wind_trade_mean": wind_trade_mean,
            "wind_midlat_mean": wind_midlat_mean,
            "wind_subtrop_mean": wind_subtrop_mean,
            "wind_polar_mean": wind_polar_mean,
            "wind_trade_ratio": wind_trade_ratio,
            "wind_u_trade_mean": wind_u_trade_mean,
            "wind_u_midlat_mean": wind_u_midlat_mean,
            "wind_u_polar_mean": wind_u_polar_mean,
            "wind_v_hadley_n_mean": wind_v_hadley_n_mean,
            "wind_v_hadley_s_mean": wind_v_hadley_s_mean,
            "wind_v_polar_n_mean": wind_v_polar_n_mean,
            "wind_v_polar_s_mean": wind_v_polar_s_mean,
            "wind_itcz_conv": wind_itcz_conv,
            "wind_jet_lat_n": wind_jet_lat_n,
            "wind_jet_lat_s": wind_jet_lat_s,
            "wind_jet_speed_n": wind_jet_speed_n,
            "wind_jet_speed_s": wind_jet_speed_s,
            "wind_eke_mean": wind_eke_mean,
            "wind_div_rms": wind_div_rms,
            "wind_vort_rms": wind_vort_rms,
            "circulation_score": score,
            
            # Cloud metrics
            "cloud_mean": cloud_mean,
            "cloud_max": cloud_max,
            "cloud_min": cloud_min,
            "cloud_equator": cloud_equator,
            "cloud_pole_north": cloud_pole_n,
            "cloud_pole_south": cloud_pole_s,
            
            # Humidity metrics
            "humidity_mean": humidity_mean,
            "humidity_max": humidity_max,
            "humidity_min": humidity_min,

            # Soil moisture metrics
            "soil_mean": soil_mean,
            "soil_max": soil_max,
            "soil_min": soil_min,
            
            # Snow/Ice metrics
            "snow_mean": snow_mean,
            "snow_max": snow_max,
            "snow_cover_fraction": snow_cover_fraction,
            "ice_mean": ice_mean,
            "ice_max": ice_max,
            "ice_cover_fraction": ice_cover_fraction,
            "ice_frac_nh": ice_frac_nh,
            "ice_frac_sh": ice_frac_sh,
            "ice_edge_n": ice_edge_n,
            "ice_edge_s": ice_edge_s,
            
            # Albedo
            "albedo_mean": albedo_mean,
            
            # Raw data for plotting
            "latitudes": lat,
            "zonal_means": T_zonal
            ,
            # Raw wind zonal-mean profiles (for circulation diagnostics / cell validation)
            "lat_deg": lat_deg if (state.wind_u is not None and state.wind_v is not None) else self._lat_deg_for_H(H),
            "wind_u_zonal_by_lat": zu if (state.wind_u is not None and state.wind_v is not None) else None,
            "wind_v_zonal_by_lat": zv if (state.wind_u is not None and state.wind_v is not None) else None,
            "wind_speed_zonal_by_lat": zspeed if (state.wind_u is not None and state.wind_v is not None) else None,

            # Atmospheric CO2 concentration [ppm]
            "co2_ppm": float(state.co2_atmosphere) if hasattr(state, "co2_atmosphere") else 400.0,
        }

    def record_step(self, state, day_of_year: float, days_elapsed: float = 0.0, component_contributions: Optional[Dict[str, np.ndarray]] = None):
        """Record a simulation step with optional per-component analysis.
        
        Args:
            state: Current PlanetState
            day_of_year: Current day of year (0-365.2422, wraps)
            days_elapsed: Days advanced in this step (added to total_days)
            component_contributions: Optional dict of component temperature contributions
        """
        stats = self.analyze_snapshot(state)
        stats['day_of_year'] = day_of_year
        self.total_days += days_elapsed
        stats['total_days'] = self.total_days  # Track total simulation time (doesn't wrap)
        
        if self.track_history:
            self.history.append(stats)
        
        if component_contributions is not None:
            # Analyze component contributions
            comp_stats = {}
            for name, field in component_contributions.items():
                if field is not None and isinstance(field, np.ndarray):
                    comp_stats[f'{name}_mean'] = float(np.mean(field))
                    comp_stats[f'{name}_min'] = float(np.min(field))
                    comp_stats[f'{name}_max'] = float(np.max(field))
                    comp_stats[f'{name}_std'] = float(np.std(field))
                elif field is not None:
                    # Handle scalar values (like net_radiation)
                    comp_stats[name] = float(field) if isinstance(field, (np.integer, np.floating)) else field
            comp_stats['day_of_year'] = day_of_year
            comp_stats['total_days'] = self.total_days
            self.component_history.append(comp_stats)
    
    def export_time_series(self, filename: Optional[str] = None, format: str = 'csv') -> str:
        """Export time series data to CSV or JSON.
        
        Args:
            filename: Output filename (auto-generated if None)
            format: 'csv' or 'json'
            
        Returns:
            Path to exported file
        """
        if not self.history:
            raise ValueError("No history data to export")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_data_{timestamp}.{format}"
        
        filepath = Path(filename)
        
        if format == 'csv':
            # Prepare data for CSV: convert numpy arrays to JSON strings
            csv_data = []
            for record in self.history:
                csv_record = {}
                for key, value in record.items():
                    if isinstance(value, np.ndarray):
                        # Convert numpy array to JSON string
                        csv_record[key] = json.dumps(value.tolist())
                    elif isinstance(value, (np.integer, np.floating)):
                        # Convert numpy scalar to Python native type
                        csv_record[key] = float(value)
                    else:
                        csv_record[key] = value
                csv_data.append(csv_record)
            
            with open(filepath, 'w', newline='') as f:
                if csv_data:
                    writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_data)
        elif format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            json_data = []
            for record in self.history:
                json_record = {}
                for key, value in record.items():
                    if isinstance(value, np.ndarray):
                        json_record[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        json_record[key] = float(value)
                    else:
                        json_record[key] = value
                json_data.append(json_record)
            
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return str(filepath)
    
    def export_component_analysis(self, filename: Optional[str] = None) -> str:
        """Export per-component contribution analysis."""
        if not self.component_history:
            raise ValueError("No component history data to export")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"component_analysis_{timestamp}.csv"
        
        filepath = Path(filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.component_history[0].keys())
            writer.writeheader()
            writer.writerows(self.component_history)
        
        return str(filepath)
    
    def get_latitude_profile(self, day_of_year: Optional[float] = None) -> Dict[str, np.ndarray]:
        """Get temperature profile by latitude for a specific day or latest.
        
        Returns:
            Dict with 'latitudes' (degrees) and 'temperatures' (K)
        """
        if not self.history:
            return {}
        
        if day_of_year is not None:
            # Find closest day
            idx = min(range(len(self.history)), 
                     key=lambda i: abs(self.history[i]['day_of_year'] - day_of_year))
            stats = self.history[idx]
        else:
            stats = self.history[-1]
        
        if 'latitudes' not in stats or 'zonal_means' not in stats:
            return {}
        
        return {
            'latitudes': np.rad2deg(stats['latitudes']),
            'temperatures': stats['zonal_means'],
            'day_of_year': stats['day_of_year']
        }

    def analyze_circulation(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Validate 3-cell circulation signatures from zonal-mean surface winds.

        This is intentionally surface-based: your current atmosphere model is a
        (mostly) single-layer near-surface wind field, so we validate the features
        that should appear at the surface on Earth:
        - Hadley: trades (easterly) and equatorward low-level flow + ITCZ convergence proxy
        - Ferrel: mid-lat westerlies and poleward low-level flow + eddy activity proxy
        - Polar: polar easterlies and equatorward low-level flow
        """
        lat_deg = np.asarray(stats.get("lat_deg", []), dtype=np.float32)
        zu = stats.get("wind_u_zonal_by_lat", None)
        zv = stats.get("wind_v_zonal_by_lat", None)
        zspeed = stats.get("wind_speed_zonal_by_lat", None)
        if lat_deg.size == 0 or zu is None or zv is None or zspeed is None:
            return {"ok": False, "reason": "missing wind profiles in stats (run with wind enabled)"}

        zu = np.asarray(zu, dtype=np.float32)
        zv = np.asarray(zv, dtype=np.float32)
        zspeed = np.asarray(zspeed, dtype=np.float32)

        # Area weights by latitude (consistent with analyze_snapshot)
        lat_rad = np.deg2rad(lat_deg.astype(np.float32))
        w = np.cos(lat_rad)
        w = w / (float(np.sum(w)) + 1e-12)

        # Band means (Earth-ish lat bands)
        # Note: v is +northward. Equatorward means: NH v<0, SH v>0.
        u_trade = self._band_mean(w, zu, lat_deg, lo=5.0, hi=20.0)
        u_mid = self._band_mean(w, zu, lat_deg, lo=30.0, hi=60.0)
        u_polar = self._band_mean(w, zu, lat_deg, lo=60.0, hi=85.0)

        v_had_n = self._band_mean(w, zv, lat_deg, lo=5.0, hi=25.0, hemi="N")
        v_had_s = self._band_mean(w, zv, lat_deg, lo=5.0, hi=25.0, hemi="S")
        v_fer_n = self._band_mean(w, zv, lat_deg, lo=30.0, hi=60.0, hemi="N")
        v_fer_s = self._band_mean(w, zv, lat_deg, lo=30.0, hi=60.0, hemi="S")
        v_pol_n = self._band_mean(w, zv, lat_deg, lo=60.0, hi=85.0, hemi="N")
        v_pol_s = self._band_mean(w, zv, lat_deg, lo=60.0, hi=85.0, hemi="S")

        # ITCZ convergence proxy: negative dv/dlat near equator (our stored scalar already matches this sign)
        itcz = float(stats.get("wind_itcz_conv", 0.0))

        # Speed structure (doldrums vs trades vs mid-lat storm track)
        sp_eq = self._band_mean(w, zspeed, lat_deg, lo=0.0, hi=5.0)
        sp_trade = self._band_mean(w, zspeed, lat_deg, lo=5.0, hi=20.0)
        sp_mid = self._band_mean(w, zspeed, lat_deg, lo=30.0, hi=60.0)

        jet_lat_n, jet_lat_s, _, _ = _jet_latitudes_from_profiles(lat_deg, zspeed, zu)

        checks: List[Dict[str, Any]] = []
        # Zonal wind sign checks
        checks.append(self._expect_sign("Trades easterly (u<0)", u_trade, sign=-1, tol=0.10))
        checks.append(self._expect_sign("Mid-lat westerlies (u>0)", u_mid, sign=+1, tol=0.10))
        checks.append(self._expect_sign("Polar easterlies (u<0)", u_polar, sign=-1, tol=0.10))
        # Meridional flow direction checks
        checks.append(self._expect_sign("Hadley surface flow equatorward (NH v<0)", v_had_n, sign=-1, tol=0.2))
        checks.append(self._expect_sign("Hadley surface flow equatorward (SH v>0)", v_had_s, sign=+1, tol=0.2))
        checks.append(self._expect_sign("Ferrel surface flow poleward (NH v>0)", v_fer_n, sign=+1, tol=0.2))
        checks.append(self._expect_sign("Ferrel surface flow poleward (SH v<0)", v_fer_s, sign=-1, tol=0.2))
        checks.append(self._expect_sign("Polar surface flow equatorward (NH v<0)", v_pol_n, sign=-1, tol=0.2))
        checks.append(self._expect_sign("Polar surface flow equatorward (SH v>0)", v_pol_s, sign=+1, tol=0.2))
        # ITCZ + jets (loose expectations; model-dependent)
        checks.append({"name": "ITCZ convergence proxy positive", "value": itcz, "expect": ">0", "ok": bool(itcz > 0.0)})
        checks.append({"name": "Trades stronger than doldrums (speed 5-20 > 0-5)", "value": float(sp_trade - sp_eq), "expect": ">0", "ok": bool(sp_trade > sp_eq)})
        checks.append({"name": "Mid-lats active (speed 30-60 >= 0.8*trades)", "value": float(sp_mid / (sp_trade + 1e-6)), "expect": ">=0.8", "ok": bool(sp_mid >= 0.8 * sp_trade)})
        checks.append({"name": "NH jet latitude in mid-lats (20..70)", "value": jet_lat_n, "expect": "20..70", "ok": bool(20.0 <= jet_lat_n <= 70.0)})
        checks.append({"name": "SH jet latitude in mid-lats (-70..-20)", "value": jet_lat_s, "expect": "-70..-20", "ok": bool(-70.0 <= jet_lat_s <= -20.0)})

        ok = all(bool(c.get("ok")) for c in checks)
        return {
            "ok": bool(ok),
            "checks": checks,
            "bands": {
                "u_trade_5_20": float(u_trade),
                "u_mid_30_60": float(u_mid),
                "u_polar_60_85": float(u_polar),
                "speed_eq_0_5": float(sp_eq),
                "speed_trade_5_20": float(sp_trade),
                "speed_mid_30_60": float(sp_mid),
                "v_hadley_N_5_25": float(v_had_n),
                "v_hadley_S_5_25": float(v_had_s),
                "v_ferrel_N_30_60": float(v_fer_n),
                "v_ferrel_S_30_60": float(v_fer_s),
                "v_polar_N_60_85": float(v_pol_n),
                "v_polar_S_60_85": float(v_pol_s),
                "itcz_conv": float(itcz),
                "jet_lat_n": float(jet_lat_n),
                "jet_lat_s": float(jet_lat_s),
            },
        }

    def analyze_circulation_from_history(self, window_days: float = 90.0) -> Dict[str, Any]:
        """Time-average circulation diagnostics over the last `window_days` of history."""
        if not self.history:
            return {"ok": False, "reason": "no history (run simulation with diagnostics.record_step)"}
        end = float(self.history[-1].get("total_days", self.total_days))
        start = end - float(window_days)
        recs = [r for r in self.history if float(r.get("total_days", 0.0)) >= start]
        if not recs:
            recs = self.history[-min(10, len(self.history)):]

        # Mean wind profiles across records (assume constant grid shape).
        def _mean_key(k: str) -> Optional[np.ndarray]:
            xs = [r.get(k, None) for r in recs]
            xs = [np.asarray(x, dtype=np.float32) for x in xs if x is not None]
            if not xs:
                return None
            return np.mean(np.stack(xs, axis=0), axis=0).astype(np.float32)

        stats0 = dict(recs[-1])  # base metadata from most recent record
        stats0["wind_u_zonal_by_lat"] = _mean_key("wind_u_zonal_by_lat")
        stats0["wind_v_zonal_by_lat"] = _mean_key("wind_v_zonal_by_lat")
        stats0["wind_speed_zonal_by_lat"] = _mean_key("wind_speed_zonal_by_lat")
        # Recompute ITCZ convergence from mean zv (more consistent than averaging scalars)
        lat_deg = np.asarray(stats0.get("lat_deg", []), dtype=np.float32)
        zv = stats0.get("wind_v_zonal_by_lat", None)
        if lat_deg.size and zv is not None:
            dv_dlat = np.gradient(np.asarray(zv, dtype=np.float32), lat_deg)
            eq_i = int(np.argmin(np.abs(lat_deg)))
            lo_i = max(0, eq_i - 2)
            hi_i = min(len(dv_dlat), eq_i + 3)
            stats0["wind_itcz_conv"] = float(-np.mean(dv_dlat[lo_i:hi_i]))
        return self.analyze_circulation(stats0)

    def print_circulation_report(self, circ: Dict[str, Any]) -> None:
        """Print a compact circulation validation report."""
        print("\n--- CIRCULATION (3-CELL) VALIDATION ---")
        if not circ or not circ.get("ok", False):
            print("Result: FAIL (or incomplete)")
            if isinstance(circ, dict) and circ.get("reason"):
                print(f"Reason: {circ['reason']}")
        else:
            print("Result: PASS (surface proxies match expected signs)")
        for c in circ.get("checks", []):
            ok = "OK " if c.get("ok") else "BAD"
            print(f"{ok} | {c.get('name')}: {c.get('value'):.3f} (expect {c.get('expect')})")
        print("--------------------------------------\n")
    
    def print_report(self, stats: Dict[str, Any]):
        """Print a formatted comparison against Earth."""
        def _flag(val: float, lo: float, hi: float) -> str:
            return "OK " if lo <= val <= hi else "BAD"

        print("\n--- CLIMATE DIAGNOSTICS REPORT ---")
        gmt = stats['global_mean_temp']
        print(f"Global Mean Temp: {gmt:.1f} K ({gmt-273.15:.1f}°C)")
        print(f"  vs Earth Ref:   {EARTH_REF['global_mean_temp']:.1f} K (Diff: {gmt - EARTH_REF['global_mean_temp']:+.1f} K)")

        T_eq = stats['T_equator']
        T_pn = stats['T_pole_north']
        T_ps = stats['T_pole_south']
        pn_ref = EARTH_REF['pole_n_annual']
        ps_ref = EARTH_REF['pole_s_annual']
        eq_ref = EARTH_REF['equator_temp']
        print(f"\nEquator Temp:     {T_eq:.1f} K  (Earth ~{eq_ref:.0f} K, diff {T_eq-eq_ref:+.1f} K)")
        print(f"Pole (North):     {T_pn:.1f} K  (Earth ~{pn_ref:.0f} K, diff {T_pn-pn_ref:+.1f} K)  [{_flag(T_pn, pn_ref-15, pn_ref+10)}]")
        print(f"Pole (South):     {T_ps:.1f} K  (Earth ~{ps_ref:.0f} K, diff {T_ps-ps_ref:+.1f} K)  [{_flag(T_ps, ps_ref-15, ps_ref+10)}]")

        gn = stats['gradient_north']
        gs = stats['gradient_south']
        gn_lo = EARTH_REF['equator_pole_gradient']
        gn_hi = EARTH_REF['equator_pole_gradient_max']
        print(f"\nEquator-to-Pole Gradient (N): {gn:.1f} K (Earth {gn_lo:.0f}-{gn_hi:.0f} K)  [{_flag(gn, gn_lo, gn_hi)}]")
        print(f"Equator-to-Pole Gradient (S): {gs:.1f} K  [{_flag(gs, gn_lo, gn_hi)}]")

        print(f"\nExtremes: Min {stats['T_min']:.1f} K / Max {stats['T_max']:.1f} K")
        print("----------------------------------\n")

    def print_wind_report(self, stats: Dict[str, Any]) -> None:
        """Print wind speed and direction comparison against Earth references."""
        print("\n--- WIND DIAGNOSTICS (vs Earth) ---")
        w_speed = stats.get("wind_eq_mean", None)
        if w_speed is None:
            print("No wind data available.")
            print("-----------------------------------\n")
            return

        def _row(name: str, sim: float, ref: float, tol: float, unit: str = "m/s") -> str:
            flag = "OK " if abs(sim - ref) <= tol else "BAD"
            return f"  {flag} | {name:<35} {sim:+7.2f} {unit}  (Earth ~{ref:+.1f}, diff {sim-ref:+.1f})"

        def _sign_row(name: str, sim: float, sign: int) -> str:
            ok = (sim * sign) >= 0
            flag = "OK " if ok else "BAD"
            return f"  {flag} | {name:<35} {sim:+7.2f} m/s"

        ref = EARTH_REF
        print(f"{'Band':<37} {'Sim':>8}   {'Earth':>8}   {'Status'}")
        print("-" * 65)
        print(_row("Doldrums speed (0-5°)", stats['wind_eq_mean'], ref['wind_doldrums_speed'], 2.0))
        print(_row("Trades speed (5-20°)", stats['wind_trade_mean'], ref['wind_trades_speed'], 3.0))
        print(_row("Horse lats speed (20-30°)", stats['wind_subtrop_mean'], ref['wind_subtrop_speed'], 2.5))
        print(_row("Mid-lat storm track (30-60°)", stats['wind_midlat_mean'], ref['wind_midlat_speed'], 4.0))
        print(_row("Polar speed (60-90°)", stats['wind_polar_mean'], ref['wind_polar_speed'], 2.5))
        print()
        print("  Zonal direction (u component):")
        print(_sign_row("Trades easterly u (5-20°, expect <0)", stats['wind_u_trade_mean'], -1))
        print(_sign_row("Mid-lat westerly u (30-60°, expect >0)", stats['wind_u_midlat_mean'], +1))
        print(_sign_row("Polar easterly u (60-90°, expect <0)", stats['wind_u_polar_mean'], -1))
        print()
        cs = stats.get("circulation_score", 0.0)
        eke = stats.get("wind_eke_mean", 0.0)
        vort = stats.get("wind_vort_rms", 0.0)
        print(f"  Circulation score:    {cs:.2f}  (higher = more Earth-like; ~5-8 is good)")
        print(f"  Eddy kinetic energy:  {eke:.2f} m²/s²  (Earth surface ~5-15 m²/s²)")
        print(f"  Vorticity RMS:        {vort:.4f} s⁻¹")
        print(f"  Max wind speed:       {stats.get('wind_max', 0.0):.1f} m/s")
        print("-----------------------------------\n")


    def print_ice_report(self, stats: Dict[str, Any]) -> None:
        """Print sea-ice extent diagnostics vs Earth annual-mean references."""
        print("\n--- SEA ICE EXTENT DIAGNOSTICS ---")
        ref = EARTH_REF
        ice_frac_nh = stats.get("ice_frac_nh", None)
        if ice_frac_nh is None:
            print("No ice data available.")
            print("----------------------------------\n")
            return

        ice_frac_sh = stats.get("ice_frac_sh", 0.0)
        ice_edge_n  = stats.get("ice_edge_n", 90.0)
        ice_edge_s  = stats.get("ice_edge_s", -90.0)
        ice_frac_gl = stats.get("ice_cover_fraction", 0.0)

        def _frac_flag(val: float, ref_val: float) -> str:
            # OK if within a factor of 2 of reference (ice extent is highly variable)
            return "OK " if (ref_val * 0.3 <= val <= ref_val * 3.0) else "BAD"

        def _edge_flag_n(edge: float) -> str:
            # Earth NH annual mean ice edge ~70-78°N; warn if equatorward of 60°N
            return "OK " if edge >= 60.0 else "BAD"

        def _edge_flag_s(edge: float) -> str:
            # Earth SH annual mean ice edge ~55-70°S; warn if equatorward of 50°S
            return "OK " if edge <= -50.0 else "BAD"

        print(f"  Global ice cover fraction: {ice_frac_gl*100:.1f}%")
        print()
        fn = _frac_flag(ice_frac_nh, ref["ice_frac_nh"])
        fs = _frac_flag(ice_frac_sh, ref["ice_frac_sh"])
        print(f"  {fn} | NH ice fraction (rows >10% ice):  {ice_frac_nh*100:.1f}%   (Earth ~{ref['ice_frac_nh']*100:.0f}%)")
        print(f"  {fs} | SH ice fraction (rows >10% ice):  {ice_frac_sh*100:.1f}%   (Earth ~{ref['ice_frac_sh']*100:.0f}%)")
        print()
        en = _edge_flag_n(ice_edge_n)
        es = _edge_flag_s(ice_edge_s)
        print(f"  {en} | NH equatorward ice edge:  {ice_edge_n:.1f}°N  (Earth annual mean ~{ref['ice_edge_n']:.0f}°N)")
        print(f"  {es} | SH equatorward ice edge:  {ice_edge_s:.1f}°S  (Earth annual mean ~{ref['ice_edge_s']:.0f}°S)")
        print()
        # Diagnostic hints based on state
        if ice_frac_sh > ref["ice_frac_sh"] * 2.0:
            print("  HINT: SH ice cover is >2× Earth ref — check polar_damp, ice melt threshold,")
            print("        and ice-albedo feedback. Consider running a 1-year mean before comparing.")
        if ice_edge_s > -50.0 and ice_frac_sh > 0.01:
            print("  HINT: SH ice extends past 50°S (too equatorward). Likely cause: ice-albedo")
            print("        runaway. Check melt_temp threshold and ocean heat transport at 55-70°S.")
        if ice_frac_nh > ref["ice_frac_nh"] * 2.0:
            print("  HINT: NH ice cover is >2× Earth ref.")
        print("----------------------------------\n")


# =============================================================================
# Latitude-Band Validation for Real-World Comparison
# =============================================================================

# Earth reference values by latitude band (approximate annual zonal means).
# IMPORTANT: SH high-latitude bands differ from NH because of the Antarctic ice sheet.
# NH 75-85°N is mostly Arctic Ocean (~-12 to -20°C); SH 75-85°S is mostly Antarctic
# ice sheet plateau (~-35 to -50°C).  Using a single symmetric table misrepresents
# the SH pole by ~25°C, giving a false "BAD" flag for the correct Antarctic cold.
EARTH_LATITUDE_BANDS_NH = {
    # (temp_celsius, precip_mm_yr) — NH or symmetric (equator, subtropics)
    0:   (27.0, 2000),   # Equator
    10:  (26.0, 1800),   # Tropical
    20:  (24.0, 1000),   # Subtropical
    30:  (18.0, 600),    # Dry subtropics
    40:  (12.0, 900),    # Temperate
    50:  (5.0, 800),     # Cool temperate
    60:  (-2.0, 500),    # Subarctic / subantarctic
    70:  (-12.0, 300),   # Arctic Ocean / subantarctic
    80:  (-20.0, 200),   # High Arctic (mostly ocean)
    90:  (-27.0, 100),   # North Pole (Arctic Ocean, annual mean ~-27°C)
}
EARTH_LATITUDE_BANDS_SH = {
    # SH-specific overrides for polar bands (Antarctic ice sheet is colder than Arctic)
    0:   (27.0, 2000),
    10:  (26.0, 1800),
    20:  (24.0, 1000),
    30:  (18.0, 600),
    40:  (12.0, 900),
    50:  (5.0, 800),
    60:  (-2.0, 500),    # subantarctic
    70:  (-12.0, 300),   # Southern Ocean / Antarctic coast
    80:  (-35.0, 150),   # Antarctic ice sheet interior (~-35 to -45°C annual mean)
    90:  (-49.0, 50),    # South Pole (Amundsen-Scott station mean: ~-49°C)
}
# Backward-compat alias (used by some callers)
EARTH_LATITUDE_BANDS = EARTH_LATITUDE_BANDS_NH


def compute_latitude_band_stats(state, band_width_deg: float = 10.0) -> dict:
    """Compute zonal climate statistics for comparison with Earth observations.

    Args:
        state: PlanetState with temperature and precipitation
        band_width_deg: Width of each latitude band in degrees (default 10°)

    Returns:
        Dictionary with latitude band statistics:
        - bands: List of {lat, temp_c, precip_mm_yr, temp_earth, precip_earth, temp_diff, precip_diff}
        - summary: Overall comparison metrics
    """
    if state.temperature is None:
        return {"error": "No temperature data"}

    # Latitude band report uses air temperature (surface climate measurements)
    T = state.air_temperature if state.air_temperature is not None else state.temperature
    H, W = T.shape

    # Precipitation (convert from mm/day to mm/year)
    if state.precipitation is not None:
        P = state.precipitation * 365.0  # mm/day -> mm/year
    else:
        P = np.zeros_like(T)

    # Latitude array (degrees, from +90 to -90)
    lat_deg = (0.5 - (np.arange(H, dtype=np.float32) + 0.5) / H) * 180.0

    # Compute zonal means
    T_zonal = np.mean(T, axis=1)  # K
    P_zonal = np.mean(P, axis=1)  # mm/year

    # Area weights (cosine of latitude)
    lat_rad = np.deg2rad(lat_deg)
    weights = np.cos(lat_rad)
    weights = weights / np.sum(weights)

    # Compute band statistics
    bands = []
    n_bands = int(180 / band_width_deg)

    for i in range(n_bands):
        lat_center = 90 - (i + 0.5) * band_width_deg
        lat_lo = 90 - (i + 1) * band_width_deg
        lat_hi = 90 - i * band_width_deg

        # Find rows in this band
        mask = (lat_deg >= lat_lo) & (lat_deg < lat_hi)
        if not np.any(mask):
            continue

        # Weighted mean for this band
        w = weights[mask]
        w = w / (np.sum(w) + 1e-12)
        temp_k = float(np.sum(T_zonal[mask] * w))
        temp_c = temp_k - 273.15
        precip = float(np.sum(P_zonal[mask] * w))

        # Get Earth reference using hemisphere-appropriate table.
        abs_lat = int(abs(lat_center) / 10) * 10
        abs_lat = min(abs_lat, 90)
        ref_table = EARTH_LATITUDE_BANDS_SH if lat_center < 0 else EARTH_LATITUDE_BANDS_NH
        earth_temp, earth_precip = ref_table.get(abs_lat, (15.0, 1000))

        bands.append({
            "lat_center": round(lat_center, 1),
            "lat_range": f"{lat_lo:.0f}° to {lat_hi:.0f}°",
            "temp_c": round(temp_c, 1),
            "temp_earth_c": earth_temp,
            "temp_diff_c": round(temp_c - earth_temp, 1),
            "precip_mm_yr": round(precip, 0),
            "precip_earth_mm_yr": earth_precip,
            "precip_diff_mm_yr": round(precip - earth_precip, 0),
        })

    # Summary statistics
    if bands:
        temp_diffs = [b["temp_diff_c"] for b in bands]
        precip_diffs = [b["precip_diff_mm_yr"] for b in bands]
        summary = {
            "mean_temp_bias_c": round(np.mean(temp_diffs), 1),
            "max_temp_bias_c": round(np.max(np.abs(temp_diffs)), 1),
            "mean_precip_bias_mm_yr": round(np.mean(precip_diffs), 0),
            "max_precip_bias_mm_yr": round(np.max(np.abs(precip_diffs)), 0),
            "n_bands": len(bands),
        }
    else:
        summary = {"error": "No valid bands computed"}

    return {
        "bands": bands,
        "summary": summary,
    }


def print_latitude_band_report(stats: dict) -> None:
    """Print latitude band comparison against Earth values."""
    print("\n--- LATITUDE-BAND VALIDATION (vs Earth) ---")
    print(f"{'Lat':>6} | {'Sim T':>7} | {'Earth T':>8} | {'T Diff':>7} | {'Sim P':>8} | {'Earth P':>8} | {'P Diff':>8}")
    print("-" * 75)

    for band in stats.get("bands", []):
        print(f"{band['lat_center']:>6.0f}° | "
              f"{band['temp_c']:>6.1f}°C | "
              f"{band['temp_earth_c']:>7.1f}°C | "
              f"{band['temp_diff_c']:>+6.1f}°C | "
              f"{band['precip_mm_yr']:>7.0f}mm | "
              f"{band['precip_earth_mm_yr']:>7.0f}mm | "
              f"{band['precip_diff_mm_yr']:>+7.0f}mm")

    summary = stats.get("summary", {})
    print("-" * 75)
    print(f"Temperature bias: mean {summary.get('mean_temp_bias_c', 'N/A')}°C, "
          f"max {summary.get('max_temp_bias_c', 'N/A')}°C")
    print(f"Precip bias: mean {summary.get('mean_precip_bias_mm_yr', 'N/A')} mm/yr, "
          f"max {summary.get('max_precip_bias_mm_yr', 'N/A')} mm/yr")
    print("-------------------------------------------\n")


def compute_t_base_profile() -> dict:
    """Analytically compute the ocean T_base profile used in the simulation.

    This runs in <1 second without a full simulation and answers the key question:
    "Is the TARGET temperature at each latitude correct (vs Earth SST), or is the
    actual simulation T being pulled away from a correct target?"

    Uses the same formulas as simulate.py._evolve_temperature to ensure consistency.

    Returns a dict with per-latitude analysis for lat = -85, -75, ..., 85 degrees.
    """
    try:
        from temperature import temperature_kelvin_for_lat
        from planet_params import EARTH
    except ImportError:
        return {"error": "Could not import simulation modules"}

    results = []
    # Earth SST references (annual zonal mean, open ocean where available)
    earth_sst = {
        85: -1.8, 75: -1.5, 65: 2.0, 55: 10.0, 45: 14.0,
        35: 22.0, 25: 26.0, 15: 27.5, 5: 28.5,
        -5: 28.5, -15: 27.5, -25: 22.0, -35: 15.0,
        -45: 8.0, -55: 2.0, -65: -1.5, -75: -1.7, -85: -1.8,
    }

    import numpy as np
    pp = EARTH

    for lat_deg_val in [85, 75, 65, 55, 45, 35, 25, 15, 5,
                        -5, -15, -25, -35, -45, -55, -65, -75, -85]:
        lat_rad = np.deg2rad(float(lat_deg_val))
        lat_arr = np.array([lat_rad], dtype=np.float32)

        # T_lat_annual_mean: equinox proxy (same as simulate.py)
        T_annual_mean = float(temperature_kelvin_for_lat(
            lat_arr, day_of_year=80, polar_cooling_scale=0.3, planet_params=pp
        )[0])

        # Transport warming (replicated from simulate.py)
        abs_lat = abs(lat_deg_val)
        transport_base = 34.0 * max(0.0, min(1.0, (abs_lat - 42.0) / 28.0)) ** 1.5
        if lat_deg_val > 0:
            amoc_bonus = (3.0 * max(0.0, min(1.0, (abs_lat - 42.0) / 23.0))
                          + 15.0 * max(0.0, min(1.0, (abs_lat - 65.0) / 10.0)))
            acc_bonus = 0.0
            sh_factor = 1.0
        else:
            amoc_bonus = 0.0
            acc_bonus = (8.0 * max(0.0, min(1.0, (abs_lat - 55.0) / 10.0))
                         + 20.0 * max(0.0, min(1.0, (abs_lat - 65.0) / 10.0)))
            sh_factor = 0.58
        transport_warming = transport_base * sh_factor + amoc_bonus + acc_bonus

        T_base_ocean = T_annual_mean + transport_warming
        earth_sst_c = earth_sst.get(lat_deg_val, None)
        earth_T = earth_sst_c + 273.15 if earth_sst_c is not None else None
        gap = (T_base_ocean - earth_T) if earth_T is not None else None

        results.append({
            "lat": lat_deg_val,
            "T_annual_mean_K": round(T_annual_mean, 1),
            "transport_warming_K": round(transport_warming, 1),
            "T_base_ocean_K": round(T_base_ocean, 1),
            "T_base_ocean_C": round(T_base_ocean - 273.15, 1),
            "earth_sst_C": earth_sst_c,
            "gap_K": round(gap, 1) if gap is not None else None,
        })

    return {"profile": results}


def print_t_base_report(profile_data: dict) -> None:
    """Print the T_base vs Earth SST profile table.

    Use this to instantly diagnose whether temperature targets (T_base) are
    calibrated correctly without running a full simulation.  If T_base_ocean
    is cold relative to Earth SST, the simulation will always produce a cold
    bias at that latitude regardless of any other tuning.
    """
    print("\n--- T_BASE OCEAN PROFILE vs EARTH SST ---")
    print(f"{'Lat':>5} | {'T_annual':>9} | {'Transport':>9} | {'T_base_oc':>9} | {'Earth SST':>9} | {'Gap':>6}")
    print(f"{'':>5} | {'(K,proxy)':>9} | {'warm.(K)':>9} | {'(°C)':>9} | {'(°C)':>9} | {'(K)':>6}")
    print("-" * 65)
    for r in profile_data.get("profile", []):
        gap_s = f"{r['gap_K']:+.1f}" if r['gap_K'] is not None else "  N/A"
        flag = ""
        if r['gap_K'] is not None:
            flag = " ok" if abs(r['gap_K']) <= 5 else " !" if abs(r['gap_K']) <= 12 else " !!"
        earth_s = f"{r['earth_sst_C']:+.1f}" if r['earth_sst_C'] is not None else " N/A"
        print(f"{r['lat']:>+5}° | {r['T_annual_mean_K']:>9.1f} | {r['transport_warming_K']:>9.1f} |"
              f" {r['T_base_ocean_C']:>+8.1f} | {earth_s:>9} | {gap_s:>6}{flag}")
    print("-" * 65)
    print("  Gap = T_base_ocean - Earth_SST.  Positive = target warmer than Earth.")
    print("  ok = within +/-5K   ! = within +/-12K   !! = >12K off")
    print("  NOTE: T_base is the ocean *equilibrium target*, not the actual sim T.")
    print("  If T_base is cold, raising AMOC/transport is needed. If T_base is")
    print("  correct but sim T is cold, ice-albedo or diffusion is pulling T down.")
    print("------------------------------------------\n")


def get_koppen_distribution(state) -> dict:
    """Get distribution of Köppen climate types.

    Args:
        state: PlanetState with koppen_type field

    Returns:
        Dictionary with Köppen type counts and percentages
    """
    if state.koppen_type is None:
        return {"error": "No Köppen classification available"}

    from climate_averages import KOPPEN_NAMES

    koppen = state.koppen_type
    land_mask = state.elevation > 0.02 if state.elevation is not None else koppen > 0
    total_land = np.sum(land_mask)

    if total_land == 0:
        return {"error": "No land cells"}

    distribution = {}
    for code in range(20):
        count = np.sum((koppen == code) & land_mask)
        if count > 0:
            name = KOPPEN_NAMES.get(code, f"Type {code}")
            distribution[name] = {
                "code": code,
                "count": int(count),
                "percent": round(100.0 * count / total_land, 1),
            }

    return {
        "distribution": distribution,
        "total_land_cells": int(total_land),
    }


def print_koppen_report(stats: dict) -> None:
    """Print Köppen climate distribution report."""
    print("\n--- KÖPPEN CLIMATE DISTRIBUTION ---")

    dist = stats.get("distribution", {})
    if not dist:
        print("No Köppen data available")
        return

    print(f"{'Climate Type':<30} | {'Count':>8} | {'Percent':>7}")
    print("-" * 52)

    # Sort by percentage descending
    sorted_types = sorted(dist.items(), key=lambda x: x[1]["percent"], reverse=True)
    for name, data in sorted_types:
        print(f"{name:<30} | {data['count']:>8} | {data['percent']:>6.1f}%")

    print("-" * 52)
    print(f"Total land cells: {stats.get('total_land_cells', 'N/A')}")
    print("-----------------------------------\n")


# ============================================================================
# Conservation / energy-budget diagnostics
#
# Added 2026-07-04 in response to two bugs that aggregate climate-metric
# tests couldn't catch: an Ekman ocean-heat term scaled for a 30-day window
# but applied every day (~30x too strong), and CH4 decaying toward zero over
# multi-decade runs because sources were ~5000x smaller than the OH sink.
# Both would show up immediately here as a budget that doesn't balance.
# ============================================================================

def area_weighted_global_mean(field: np.ndarray) -> float:
    """cos(lat)-weighted global mean of a per-cell field on an equirectangular
    grid, where a plain np.mean() would over-represent polar rows relative to
    their true surface area. For latitude-uniform fields this is exactly
    np.mean()."""
    field = np.asarray(field)
    H = field.shape[0]
    lat_rad = (0.5 - (np.arange(H, dtype=np.float64) + 0.5) / H) * np.pi
    w = np.cos(lat_rad)
    row_means = np.mean(field, axis=1)
    return float(np.sum(row_means * w) / (np.sum(w) + 1e-12))


def compute_radiation_balance(components: dict) -> dict:
    """Global top-of-atmosphere energy budget from a `simulate_step(...,
    track_components=True)` components dict.

    `net_radiation` is per-step S_absorbed - L_out [W/m^2]; its area-weighted
    global mean should stay near zero at climate equilibrium and should not
    show a large, persistent one-directional drift (that drift is exactly
    the symptom an unbalanced energy term -- like the Ekman over-application
    bug -- produces).
    """
    if "net_radiation" not in components:
        raise ValueError(
            "components dict has no 'net_radiation' entry -- call simulate_step "
            "with track_components=True"
        )
    result = {
        "r_net_mean_w_m2": area_weighted_global_mean(components["net_radiation"]),
    }
    if "S_absorbed" in components:
        result["s_absorbed_mean_w_m2"] = area_weighted_global_mean(components["S_absorbed"])
    if "L_out" in components:
        result["l_out_mean_w_m2"] = area_weighted_global_mean(components["L_out"])
    return result
