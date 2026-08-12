"""Grid-aware diagnostics for PlanetSim's layer-aware circulation path."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pressure_circulation import spherical_divergence


def latitude_centres_deg(height: int) -> np.ndarray:
    if height < 2:
        raise ValueError("at least two latitude rows are required")
    return 90.0 - (np.arange(height, dtype=np.float64) + 0.5) * 180.0 / height


def jet_latitudes_deg(zonal_wind_m_s: np.ndarray, *, tropical_edge_deg: float = 15.0) -> dict[str, float]:
    """Latitude of each hemisphere's strongest zonal-mean westerly jet."""
    wind = np.asarray(zonal_wind_m_s, dtype=np.float64)
    if wind.ndim != 2 or wind.shape[1] != 2 * wind.shape[0]:
        raise ValueError("zonal wind must use a 2:1 global grid")
    lat = latitude_centres_deg(wind.shape[0])
    mean_u = np.mean(wind, axis=1)

    def peak(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        rows = np.flatnonzero(mask)
        return float(lat[rows[np.argmax(mean_u[rows])]])

    return {"nh_deg": peak(lat >= tropical_edge_deg), "sh_deg": peak(lat <= -tropical_edge_deg)}


def jet_core_properties(
    zonal_wind_m_s: np.ndarray, *, tropical_edge_deg: float = 15.0
) -> dict[str, dict[str, float]]:
    """Return each hemisphere's zonal-mean westerly jet position and strength."""
    wind = np.asarray(zonal_wind_m_s, dtype=np.float64)
    if wind.ndim != 2 or wind.shape[1] != 2 * wind.shape[0]:
        raise ValueError("zonal wind must use a 2:1 global grid")
    lat = latitude_centres_deg(wind.shape[0])
    mean_u = np.mean(wind, axis=1)

    def core(mask: np.ndarray) -> dict[str, float]:
        if not np.any(mask):
            return {"latitude_deg": float("nan"), "speed_m_s": float("nan")}
        rows = np.flatnonzero(mask)
        row = rows[np.argmax(mean_u[rows])]
        return {"latitude_deg": float(lat[row]), "speed_m_s": float(mean_u[row])}

    return {
        "nh": core(lat >= tropical_edge_deg),
        "sh": core(lat <= -tropical_edge_deg),
    }


def seasonal_jet_scorecard(
    lower_zonal_wind_samples: Sequence[np.ndarray],
    *,
    upper_zonal_wind_samples: Sequence[np.ndarray] | None = None,
    tropical_edge_deg: float = 15.0,
) -> dict[str, Any]:
    """Summarize seasonal jet migration and NH/SH core asymmetry.

    Samples are successive model states from one evaluation period.  This is a
    model-internal placement diagnostic, not a comparison to near-surface NCEP
    wind speed: that product cannot identify an upper-tropospheric jet core.
    """
    if not lower_zonal_wind_samples:
        raise ValueError("at least one lower-wind sample is required")

    def summarize(samples: Sequence[np.ndarray]) -> dict[str, Any]:
        cores = [jet_core_properties(sample, tropical_edge_deg=tropical_edge_deg) for sample in samples]

        def hemisphere(name: str) -> dict[str, float]:
            latitude = np.asarray([entry[name]["latitude_deg"] for entry in cores])
            speed = np.asarray([entry[name]["speed_m_s"] for entry in cores])
            return {
                "mean_latitude_deg": float(np.mean(latitude)),
                "seasonal_latitude_span_deg": float(np.max(latitude) - np.min(latitude)),
                "mean_core_speed_m_s": float(np.mean(speed)),
            }

        nh = hemisphere("nh")
        sh = hemisphere("sh")
        return {
            "nh": nh,
            "sh": sh,
            "hemispheric": {
                "absolute_latitude_difference_deg": float(
                    abs(abs(nh["mean_latitude_deg"]) - abs(sh["mean_latitude_deg"]))
                ),
                "core_speed_ratio_sh_to_nh": float(
                    sh["mean_core_speed_m_s"] / (nh["mean_core_speed_m_s"] + 1e-12)
                ),
                "core_speed_difference_nh_minus_sh_m_s": float(
                    nh["mean_core_speed_m_s"] - sh["mean_core_speed_m_s"]
                ),
            },
        }

    result: dict[str, Any] = {
        "sample_count": len(lower_zonal_wind_samples),
        "lower": summarize(lower_zonal_wind_samples),
    }
    if upper_zonal_wind_samples is not None:
        if len(upper_zonal_wind_samples) != len(lower_zonal_wind_samples):
            raise ValueError("upper-wind samples must match lower-wind sample count")
        result["upper"] = summarize(upper_zonal_wind_samples)
    return result


def hadley_edges_deg(meridional_wind_m_s: np.ndarray, *, tropical_edge_deg: float = 5.0) -> dict[str, float]:
    """First poleward sign reversal of zonal-mean meridional flow in each hemisphere."""
    wind = np.asarray(meridional_wind_m_s, dtype=np.float64)
    if wind.ndim != 2 or wind.shape[1] != 2 * wind.shape[0]:
        raise ValueError("meridional wind must use a 2:1 global grid")
    lat = latitude_centres_deg(wind.shape[0])
    mean_v = np.mean(wind, axis=1)

    def edge(sign: float) -> float:
        rows = np.flatnonzero((lat * sign) >= tropical_edge_deg)
        if rows.size < 2:
            return float("nan")
        # Rows are north-to-south: order them from equator toward the pole.
        rows = rows[np.argsort(np.abs(lat[rows]))]
        values = mean_v[rows]
        changes = np.flatnonzero(values[:-1] * values[1:] <= 0.0)
        if not changes.size:
            return float("nan")
        i = changes[0]
        a, b = values[i], values[i + 1]
        fraction = 0.5 if a == b else abs(a) / (abs(a) + abs(b))
        return float(abs(lat[rows[i]]) + fraction * (abs(lat[rows[i + 1]]) - abs(lat[rows[i]])))

    return {"nh_deg": edge(1.0), "sh_deg": edge(-1.0)}


def meridional_transport_diagnostics(
    temperature_k: np.ndarray,
    humidity: np.ndarray,
    meridional_wind_m_s: np.ndarray,
    *,
    radius_m: float,
    surface_pressure_pa: float,
    gravity_m_s2: float,
    cp_dry_j_kg_k: float,
    latent_heat_j_kg: float = 2.5e6,
    midlevel_temperature_k: np.ndarray | None = None,
    midlevel_humidity: np.ndarray | None = None,
    midlevel_meridional_wind_m_s: np.ndarray | None = None,
    upperlevel_temperature_k: np.ndarray | None = None,
    upperlevel_humidity: np.ndarray | None = None,
    upperlevel_meridional_wind_m_s: np.ndarray | None = None,
    layer_mass_fractions: tuple[float, float, float] = (1.0, 0.0, 0.0),
    lower_geopotential_m2_s2: np.ndarray | None = None,
    midlevel_geopotential_m2_s2: np.ndarray | None = None,
    upperlevel_geopotential_m2_s2: np.ndarray | None = None,
) -> dict[str, float]:
    """Return column-scaled zonal-mean moisture and energy transport metrics.

    Dry-static-energy transport is ``cp*T`` alone unless a per-layer
    geopotential (``g*z``, from ``balanced_dynamics.pressure_level_geopotential``
    or equivalent) is supplied, in which case it becomes the full dry static
    energy ``cp*T + g*z``.  Geopotential is optional per layer independently --
    a caller with only a lower-layer estimate still improves that one term.
    """
    temperature = np.asarray(temperature_k, dtype=np.float64)
    q = np.asarray(humidity, dtype=np.float64)
    v = np.asarray(meridional_wind_m_s, dtype=np.float64)
    if temperature.shape != q.shape or temperature.shape != v.shape:
        raise ValueError("temperature, humidity, and meridional wind must share one shape")
    if temperature.ndim != 2 or temperature.shape[1] != 2 * temperature.shape[0]:
        raise ValueError("transport fields must use a 2:1 global grid")
    if radius_m <= 0.0 or surface_pressure_pa <= 0.0 or gravity_m_s2 <= 0.0:
        raise ValueError("planetary radius, pressure, and gravity must be positive")
    h = temperature.shape[0]
    lat = np.radians(latitude_centres_deg(h))
    fractions = np.asarray(layer_mass_fractions, dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions < 0.0) or not np.any(fractions > 0.0):
        raise ValueError("layer mass fractions must be three non-negative values with positive sum")
    optional = (
        midlevel_temperature_k, midlevel_humidity, midlevel_meridional_wind_m_s,
        upperlevel_temperature_k, upperlevel_humidity, upperlevel_meridional_wind_m_s,
    )
    if any(value is not None for value in optional) and any(value is None for value in optional):
        raise ValueError("middle and upper transport inputs must be supplied together")
    def _geopotential(value: np.ndarray | None) -> np.ndarray:
        if value is None:
            return np.zeros_like(temperature)
        gz = np.asarray(value, dtype=np.float64)
        if gz.shape != temperature.shape:
            raise ValueError("geopotential fields must match lower-layer shape")
        return gz

    # A scorecard can be requested on a legacy one-level state while callers
    # still carry the native three-level default fractions.  In that case the
    # supplied lower field represents the full column, not merely its nominal
    # lower-layer share.
    layers = [(
        temperature, q, v,
        1.0 if all(value is None for value in optional) else fractions[0],
        _geopotential(lower_geopotential_m2_s2),
    )]
    if all(value is not None for value in optional):
        mid_t, mid_q, mid_v, upper_t, upper_q, upper_v = (
            np.asarray(value, dtype=np.float64) for value in optional
        )
        if any(value.shape != temperature.shape for value in (mid_t, mid_q, mid_v, upper_t, upper_q, upper_v)):
            raise ValueError("all transport-layer fields must match lower-layer shape")
        layers.extend((
            (mid_t, mid_q, mid_v, fractions[1], _geopotential(midlevel_geopotential_m2_s2)),
            (upper_t, upper_q, upper_v, fractions[2], _geopotential(upperlevel_geopotential_m2_s2)),
        ))
    column_mass = float(surface_pressure_pa) / float(gravity_m_s2)
    moisture_flux = sum(
        fraction * column_mass * np.mean(layer_q * layer_v, axis=1)
        for _, layer_q, layer_v, fraction, _gz in layers
    )
    dry_static_flux = sum(
        fraction * column_mass
        * np.mean((float(cp_dry_j_kg_k) * layer_t + layer_gz) * layer_v, axis=1)
        for layer_t, _, layer_v, fraction, layer_gz in layers
    )
    latent_flux = sum(
        fraction * column_mass * float(latent_heat_j_kg) * np.mean(layer_q * layer_v, axis=1)
        for _, layer_q, layer_v, fraction, _gz in layers
    )
    circumference = 2.0 * np.pi * float(radius_m) * np.cos(lat)
    dry_transport_pw = dry_static_flux * circumference / 1.0e15
    latent_transport_pw = latent_flux * circumference / 1.0e15
    total_transport_pw = dry_transport_pw + latent_transport_pw
    equatorial = np.argsort(np.abs(lat))[:2]
    return {
        "cross_equatorial_moisture_flux_kg_m_s": float(np.mean(moisture_flux[equatorial])),
        "cross_equatorial_dry_static_energy_transport_pw": float(np.mean(dry_transport_pw[equatorial])),
        "cross_equatorial_latent_energy_transport_pw": float(np.mean(latent_transport_pw[equatorial])),
        "cross_equatorial_total_energy_transport_pw": float(np.mean(total_transport_pw[equatorial])),
        "peak_northward_energy_transport_pw": float(np.max(total_transport_pw)),
        "peak_southward_energy_transport_pw": float(np.min(total_transport_pw)),
    }


def circulation_scorecard(
    lower_u: np.ndarray,
    lower_v: np.ndarray,
    mid_u: np.ndarray | None = None,
    mid_v: np.ndarray | None = None,
    upper_u: np.ndarray | None = None,
    upper_v: np.ndarray | None = None,
    omega_lower_mid_pa_s: np.ndarray | None = None,
    omega_mid_upper_pa_s: np.ndarray | None = None,
    temperature_k: np.ndarray | None = None,
    humidity: np.ndarray | None = None,
    radius_m: float | None = None,
    surface_pressure_pa: float | None = None,
    gravity_m_s2: float | None = None,
    cp_dry_j_kg_k: float | None = None,
    midlevel_temperature_k: np.ndarray | None = None,
    midlevel_humidity: np.ndarray | None = None,
    upperlevel_temperature_k: np.ndarray | None = None,
    upperlevel_humidity: np.ndarray | None = None,
    layer_mass_fractions: tuple[float, float, float] = (1.0, 0.0, 0.0),
    lower_geopotential_m2_s2: np.ndarray | None = None,
    midlevel_geopotential_m2_s2: np.ndarray | None = None,
    upperlevel_geopotential_m2_s2: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return compact, serializable circulation observables for validation."""
    lower_u = np.asarray(lower_u, dtype=np.float64)
    lower_v = np.asarray(lower_v, dtype=np.float64)
    if lower_u.shape != lower_v.shape:
        raise ValueError("lower winds must share one shape")
    result: dict[str, Any] = {
        "jet_latitude_deg": jet_latitudes_deg(lower_u),
        "hadley_edge_deg": hadley_edges_deg(lower_v),
    }
    if radius_m is not None:
        divergence = spherical_divergence(lower_u, lower_v, float(radius_m))
        result["horizontal_divergence_s"] = {
            "mean": float(np.mean(divergence)),
            "rms": float(np.sqrt(np.mean(divergence**2))),
        }
    transport_args = (
        temperature_k, humidity, radius_m, surface_pressure_pa,
        gravity_m_s2, cp_dry_j_kg_k,
    )
    if all(value is not None for value in transport_args):
        three_level_transport = all(value is not None for value in (
            midlevel_temperature_k, midlevel_humidity, mid_v,
            upperlevel_temperature_k, upperlevel_humidity, upper_v,
        ))
        result["meridional_transport"] = meridional_transport_diagnostics(
            np.asarray(temperature_k), np.asarray(humidity), lower_v,
            radius_m=float(radius_m),
            surface_pressure_pa=float(surface_pressure_pa),
            gravity_m_s2=float(gravity_m_s2),
            cp_dry_j_kg_k=float(cp_dry_j_kg_k),
            midlevel_temperature_k=(midlevel_temperature_k if three_level_transport else None),
            midlevel_humidity=(midlevel_humidity if three_level_transport else None),
            midlevel_meridional_wind_m_s=(mid_v if three_level_transport else None),
            upperlevel_temperature_k=(upperlevel_temperature_k if three_level_transport else None),
            upperlevel_humidity=(upperlevel_humidity if three_level_transport else None),
            upperlevel_meridional_wind_m_s=(upper_v if three_level_transport else None),
            layer_mass_fractions=(layer_mass_fractions if three_level_transport else (1.0, 0.0, 0.0)),
            lower_geopotential_m2_s2=lower_geopotential_m2_s2,
            midlevel_geopotential_m2_s2=(midlevel_geopotential_m2_s2 if three_level_transport else None),
            upperlevel_geopotential_m2_s2=(upperlevel_geopotential_m2_s2 if three_level_transport else None),
        )
    for name, field in (("omega_lower_mid_pa_s", omega_lower_mid_pa_s), ("omega_mid_upper_pa_s", omega_mid_upper_pa_s)):
        if field is not None:
            values = np.asarray(field, dtype=np.float64)
            if values.shape != lower_u.shape:
                raise ValueError(f"{name} shape does not match lower winds")
            result[name] = {"mean": float(np.mean(values)), "rms": float(np.sqrt(np.mean(values**2)))}
    if mid_u is not None and mid_v is not None:
        result["midlevel_jet_latitude_deg"] = jet_latitudes_deg(np.asarray(mid_u))
        result["midlevel_hadley_edge_deg"] = hadley_edges_deg(np.asarray(mid_v))
    if upper_u is not None and upper_v is not None:
        result["upper_jet_latitude_deg"] = jet_latitudes_deg(np.asarray(upper_u))
        result["upper_hadley_edge_deg"] = hadley_edges_deg(np.asarray(upper_v))
    return result
