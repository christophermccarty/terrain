"""PlanetSim parameter optimization backend.

Headless simulation runner + scoring function + search strategies.

Quick start
-----------
from optimizer.headless import run_simulation
from optimizer.scoring import ClimateScore, EARTH_REFERENCE
from planet_params import EARTH

state, metrics = run_simulation(EARTH, spinup_years=1.0, eval_years=0.5)
score = ClimateScore(EARTH_REFERENCE).score(metrics)
print(f"Earth score: {score:.1f}/100")
"""
from optimizer.scoring import ClimateMetrics, ReferenceClimate, ClimateScore, EARTH_REFERENCE
from optimizer.headless import run_simulation

__all__ = [
    "ClimateMetrics",
    "ReferenceClimate",
    "ClimateScore",
    "EARTH_REFERENCE",
    "run_simulation",
]
