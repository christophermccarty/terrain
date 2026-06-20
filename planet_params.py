"""Planet-level physical parameters.

All simulation constants that differ between planets live here.  Pass a
``PlanetParams`` instance (or the ``EARTH`` singleton) through to any
function that previously hard-coded Earth-specific values (S0, obliquity,
Ω, R, surface pressure, etc.).  Default values are calibrated for Earth.

Usage
-----
from planet_params import EARTH, PlanetParams

# Earth simulation (default)
state, _ = simulate_step(state, days=1.0, planet_params=EARTH)

# Mars-like simulation
mars = PlanetParams(
    solar_constant=590.0,
    obliquity_deg=25.19,
    orbital_period_days=687.0,
    sidereal_day_hours=24.623,
    radius_m=3.3895e6,
    surface_gravity=3.71,
    surface_pressure_pa=610.0,
    obliquity_deg=25.19,
)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PlanetParams:
    """Physical constants for a simulated planet.  All SI unless noted."""

    # ------------------------------------------------------------------ #
    # Stellar / orbital
    # ------------------------------------------------------------------ #
    solar_constant: float = 1361.0
    """TOA insolation at the reference (mean) orbital distance [W/m²]."""

    obliquity_deg: float = 23.44
    """Axial tilt [degrees]."""

    orbital_period_days: float = 365.2422
    """Length of one orbit around the host star [days]."""

    eccentricity: float = 0.0167
    """Orbital eccentricity (0 = circular)."""

    perihelion_day: float = 3.0
    """Day of year when planet is closest to star (perihelion).
    Earth: ~Jan 3, i.e. day 3.  Irrelevant for circular orbits."""

    # ------------------------------------------------------------------ #
    # Rotation
    # ------------------------------------------------------------------ #
    sidereal_day_hours: float = 23.9345
    """Length of one sidereal (stellar) day [hours]."""

    # ------------------------------------------------------------------ #
    # Size / gravity
    # ------------------------------------------------------------------ #
    radius_m: float = 6.371e6
    """Mean planetary radius [m]."""

    surface_gravity: float = 9.81
    """Surface gravitational acceleration [m/s²]."""

    # ------------------------------------------------------------------ #
    # Atmosphere
    # ------------------------------------------------------------------ #
    surface_pressure_pa: float = 101_325.0
    """Mean surface pressure [Pa]."""

    mean_molar_mass: float = 0.029
    """Mean molar mass of the atmosphere [kg/mol]  (dry air ≈ 0.029)."""

    gas_constant_dry: float = 287.0
    """Specific gas constant for dry atmosphere [J/(kg·K)]."""

    cp_dry: float = 1004.0
    """Specific heat at constant pressure [J/(kg·K)]."""

    # ------------------------------------------------------------------ #
    # Effective single-layer greenhouse parameters
    # These are used in the temperature baseline calculation and will be
    # gradually superseded by the prognostic CO2 radiative forcing path.
    # ------------------------------------------------------------------ #
    epsilon_equator: float = 0.68
    """Effective longwave emissivity at the equator.
    Reduced from 0.78: tropical T_annual_mean was 308K (+6K over Earth 302K).
    0.68 cools equator by ~6K, mid-lats by ~3-4K, poles unaffected (epsilon_pole unchanged).
    """

    epsilon_pole: float = 0.70
    """Effective longwave emissivity at the poles."""

    # ------------------------------------------------------------------ #
    # Aerosol / volcanic forcing
    # ------------------------------------------------------------------ #
    aerosol_optical_depth: float = 0.0
    """Stratospheric aerosol optical depth (AOD).
    0 = clear sky; ~0.1 typical Pinatubo forcing."""

    # ------------------------------------------------------------------ #
    # Derived convenience properties
    # ------------------------------------------------------------------ #

    @property
    def omega(self) -> float:
        """Planetary rotation rate [rad/s]."""
        return 2.0 * math.pi / (self.sidereal_day_hours * 3600.0)

    @property
    def obliquity_rad(self) -> float:
        """Axial tilt [radians]."""
        return math.radians(self.obliquity_deg)

    @property
    def surface_area_m2(self) -> float:
        """Total surface area [m²]."""
        return 4.0 * math.pi * self.radius_m ** 2

    @property
    def aerosol_forcing_w_m2(self) -> float:
        """Shortwave radiative forcing from stratospheric aerosols [W/m²].

        Uses the Lacis et al. approximation:  ΔF ≈ −25 × AOD.
        Typical values: background ≈ 0, Pinatubo 1991 ≈ −4 W/m².
        """
        return -25.0 * self.aerosol_optical_depth

    def solar_distance_factor(self, day_of_year: float) -> float:
        """Ratio of actual to mean Sun–planet distance at the given day.

        Returns ``r/a`` where ``a`` is the semi-major axis.
        TOA insolation scales as ``1 / factor²``.
        Uses the first-order Kepler expansion (accurate to ~1 % for e < 0.2).
        """
        M = 2.0 * math.pi * (day_of_year - self.perihelion_day) / self.orbital_period_days
        nu = M + 2.0 * self.eccentricity * math.sin(M)
        e = self.eccentricity
        return (1.0 - e * e) / (1.0 + e * math.cos(nu))

    def effective_solar_constant(self, day_of_year: float) -> float:
        """Solar constant corrected for orbital distance [W/m²]."""
        d = self.solar_distance_factor(day_of_year)
        return self.solar_constant / (d * d)

    def daily_mean_insolation(
        self,
        lat_rad: np.ndarray,
        day_of_year: float,
    ) -> np.ndarray:
        """Daily-mean TOA insolation Q(φ, day) [W/m²].

        Generalised version of ``temperature._daily_mean_insolation_Q`` that
        uses ``self`` (S0, obliquity, orbital period, eccentricity).
        Handles polar day/night and the exact poles correctly.

        Args:
            lat_rad: Latitude(s) in radians (scalar or array).
            day_of_year: Day of year (float; supports fractional days).

        Returns:
            Array of the same shape as ``lat_rad``, float32.
        """
        lat = np.asarray(lat_rad, dtype=np.float64)
        S0 = self.effective_solar_constant(day_of_year)
        obliq = self.obliquity_rad
        gamma = 2.0 * math.pi * (float(day_of_year) - 80.0) / self.orbital_period_days
        delta = math.asin(math.sin(obliq) * math.sin(gamma))  # solar declination

        lat_safe = np.clip(lat, -math.pi / 2 + 1e-9, math.pi / 2 - 1e-9)
        cosH0 = -np.tan(lat_safe) * math.tan(delta)
        H0 = np.arccos(np.clip(cosH0, -1.0, 1.0))
        H0 = np.where(cosH0 <= -1.0, math.pi, H0)   # 24-h day
        H0 = np.where(cosH0 >= 1.0, 0.0, H0)          # polar night

        Q = (S0 / math.pi) * (
            H0 * np.sin(lat_safe) * math.sin(delta)
            + np.cos(lat_safe) * math.cos(delta) * np.sin(H0)
        )

        # Exact pole corrections
        pole_mask = np.abs(np.abs(lat) - math.pi / 2) < 1e-6
        if np.any(pole_mask):
            Q_pole = np.zeros_like(lat)
            np_mask = pole_mask & (lat > 0)
            Q_pole[np_mask] = S0 * max(0.0, math.sin(delta))
            sp_mask = pole_mask & (lat < 0)
            Q_pole[sp_mask] = S0 * max(0.0, -math.sin(delta))
            Q = np.where(pole_mask, Q_pole, Q)

        return np.maximum(0.0, Q).astype(np.float32)

    def coriolis_parameter(self, lat_rad: np.ndarray) -> np.ndarray:
        """Coriolis parameter f = 2Ω sin(φ) [rad/s]."""
        return (2.0 * self.omega * np.sin(np.asarray(lat_rad, dtype=np.float32))).astype(np.float32)


# ---------------------------------------------------------------------------
# Singleton: Earth with present-day orbital parameters
# ---------------------------------------------------------------------------
EARTH = PlanetParams()
