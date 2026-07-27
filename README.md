# PlanetSim

PlanetSim is an interactive, Earth-calibrated climate simulator with a Tkinter
GUI and a headless experiment/optimization path. It models a 1.5-layer
atmosphere, precipitation and soil moisture, ocean heat transport and salinity,
sea ice, carbon and methane feedbacks, and Köppen climate zones on a global
latitude/longitude grid.

The model is intended for climate-system exploration rather than operational
forecasting or GCM-grade prediction. Earth is the primary validation target;
Mars and other planet configurations exercise the generalization architecture.

## Setup

Python 3.12 is the currently tested interpreter.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Always run the project with its virtual-environment interpreter:

```powershell
.\.venv\Scripts\python.exe main.py
```

The bundled Earth DEM is resolved relative to the repository. If it is absent,
the GUI falls back to deterministic procedural terrain.

## Tests

```powershell
# Routine verification
.\.venv\Scripts\python.exe -m pytest testing -m "not slow"

# Complete suite, including long climate integrations
.\.venv\Scripts\python.exe -m pytest testing

# One test module
.\.venv\Scripts\python.exe -m pytest testing/test_generalize_time_orbit.py -q
```

The suite currently collects 429 tests. Slow tests cover multi-decadal climate
drift, conservation, circulation, seasonal behavior, and reanalysis anchors.

## Headless runs and benchmarks

`optimizer/headless.py` is the deterministic, GUI-free simulation entry point
used by tests and parameter searches. Useful scripts include:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_headless.py
.\.venv\Scripts\python.exe scripts\profile_simulate_step.py
.\.venv\Scripts\python.exe scripts\check_real_terrain_koppen.py --help
.\.venv\Scripts\python.exe scripts\run_real_terrain_validation.py --compare
```

At the 512×1024 production grid, DAILY and WEEKLY modes prioritize fidelity
and are substantially slower than the long-run target. MONTHLY and ANNUAL are
the practical modes for long spinups, but intentionally approximate some fast
weather processes.

## Architecture

- `simulate.py` — state definition, time-step orchestration, persistence
- `atmosphere.py` — wind, storms, humidity, and precipitation
- `temperature.py` — insolation and radiative temperature baseline
- `ocean.py` — ocean transport, currents, salinity, and sea ice
- `hydrology.py` — optional runoff, lake storage, and river routing
- `carbon_cycle.py` — CO₂, vegetation, permafrost, and methane
- `climate_averages.py` — monthly climatology, Köppen, and biomes
- `planet_params.py` — per-planet physical and calibrated parameters
- `scenarios.py` / `orbital_cycles.py` — curated experiments and Milankovitch forcing
- `simulation_state.py` / `simulation_runner.py` — shared state contract and orchestration
- `sim_grid.py` — coarsening utilities and static-terrain cache
- `state_persistence.py` — safe NPZ persistence and legacy-save migration
- `gui_worker.py` / `gui_view_cache.py` — threaded ownership and view-layer caches
- `simulate.py` / `main.py` — numerical integration and Tk application shell
- `diagnostics.py` — climate metrics, budgets, and data export
- `optimizer/` — headless runner, scoring, sweeps, and Bayesian search
- `testing/` — analytic, regression, integration, and long-horizon tests

## Project documentation

- `PLAN.md` — historical development log
- `ROADMAP.md` — long-horizon physics and engineering ideas
- `FEATURES.md` — prioritized experience and system features
- `PLAN_PHYSICS_FIXES.md` — recent measured physics decisions
- `docs/FINDINGS_SUMMARY.md` — durable findings promoted from local experiments
- `docs/ARCHITECTURE.md` — module boundaries and compatibility contracts
- `docs/REAL_TERRAIN_VALIDATION.md` — regional/zonal benchmark workflow

## Save-file security

New saves use the versioned `.npz` format, which stores arrays plus JSON metadata
and is loaded with pickle disabled. Legacy `.pkl` saves remain readable for
migration, but **only open a pickle you created yourself or obtained from a
fully trusted source**: unpickling an untrusted file can execute arbitrary code.
Save files are intentionally excluded from git.

## Validation philosophy

Physics changes should be independently gated and measured. A default change
requires more than a green golden-state test: compare analytic expectations,
zonal bands, regional behavior, conservation budgets, long-run drift, and
DAILY/coarse-mode agreement. Fixture regeneration must include an explanation
of the observed climate changes.
