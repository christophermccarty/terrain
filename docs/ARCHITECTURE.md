# Architecture boundaries

PlanetSim keeps compatibility entry points in `simulate.py` and `main.py`, but
the state, orchestration, persistence, and worker contracts are independent of
the large physics and Tkinter implementations.

## Simulation boundary

- `simulation_state.py` defines `PlanetState` and `TimeScaleMode`, the data
  contracts shared by physics, headless runs, persistence, and the GUI.
- `simulation_runner.py` owns multi-step orchestration and zero-day state
  initialization. Physics is injected as a callable, so orchestration tests do
  not need to execute the climate model.
- `sim_grid.py` owns block coarsening and the immutable terrain-coarsening
  cache.
- `state_persistence.py` owns safe NPZ and legacy pickle migration.
- `simulate.py` owns numerical integration and cache invalidation. It re-exports
  the historical state and persistence names so existing scripts and saves
  remain compatible.

Physics modules must not import `main.py` or `gui_worker.py`. Persistence must
not import the numerical integrator.

## GUI boundary

- `gui_worker.py` owns threaded state mutation, pause/snapshot synchronization,
  atomic latest-frame delivery, and error delivery. It has no Tkinter imports.
- `gui_view_cache.py` owns GUI-only wind, ocean-current, and precipitation
  overlay caches and invalidates them as one unit.
- `main.py` owns Tk widgets, event bindings, rendering, and application
  lifecycle. Its `SimulationThread` compatibility facade injects the numerical
  step function into `SimulationWorker`.

Tk callbacks should communicate with the worker through its public pause,
snapshot, resume, and queue interfaces rather than mutating an in-flight state.

## Compatibility and verification

`simulate.PlanetState`, `simulate.TimeScaleMode`, `simulate.save_state`, and
`simulate.load_state` remain supported imports. Likewise,
`main.SimulationThread` remains available.

Behavior-preserving refactors are checked with:

- the golden-state fixture for numerical identity;
- NPZ and legacy-pickle round trips;
- headless/threaded parity and state-ownership tests;
- the real-terrain baseline for aggregate climate identity.
