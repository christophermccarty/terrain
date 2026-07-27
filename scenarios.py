"""Curated, reproducible planet/climate starting scenarios."""
from __future__ import annotations

from dataclasses import dataclass, replace

from planet_params import EARTH, MARS, PlanetParams


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    planet_params: PlanetParams


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("Earth", "Modern Earth calibration baseline.", EARTH),
    Scenario("Mars", "Thin, cold, dry Mars-like atmosphere.", MARS),
    Scenario(
        "High CO₂ Earth",
        "A warm forcing experiment starting at four times preindustrial CO₂.",
        replace(EARTH, co2_initial_ppm=1120.0),
    ),
    Scenario(
        "Snowball Earth",
        "Low stellar flux and low CO₂ favor global glaciation.",
        replace(
            EARTH,
            solar_constant=1225.0,
            co2_initial_ppm=180.0,
            co2_baseline_ppm=180.0,
        ),
    ),
    Scenario(
        "High Obliquity Earth",
        "A 60° axial tilt produces extreme high-latitude seasons.",
        replace(EARTH, obliquity_deg=60.0),
    ),
    Scenario(
        "Eccentric Earth",
        "A high-eccentricity orbit emphasizes perihelion/aphelion forcing.",
        replace(EARTH, eccentricity=0.20, perihelion_day=3.0),
    ),
    Scenario(
        "Milankovitch Earth",
        "Accelerated orbital cycles make long-period forcing visible.",
        replace(
            EARTH,
            enable_milankovitch_cycles=True,
            milankovitch_time_scale=10_000.0,
        ),
    ),
    Scenario(
        "Hydrological Earth",
        "Experimental runoff, lake storage, and river-routing diagnostics.",
        replace(EARTH, enable_surface_hydrology=True),
    ),
    Scenario(
        "Slow-Rotating Earth",
        "A ten-day sidereal rotation tests circulation scaling.",
        replace(EARTH, sidereal_day_hours=240.0),
    ),
)

SCENARIO_BY_NAME: dict[str, Scenario] = {scenario.name: scenario for scenario in SCENARIOS}


def scenario_planet_params(name: str) -> PlanetParams:
    """Return immutable parameters for a named scenario."""
    try:
        return SCENARIO_BY_NAME[name].planet_params
    except KeyError as exc:
        raise ValueError(f"Unknown scenario {name!r}") from exc
