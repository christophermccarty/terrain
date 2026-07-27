# Durable findings summary

This tracked document preserves conclusions that were originally established by
local, generated, or overnight experiments. Raw logs and large result artifacts
remain intentionally ignored. Detailed measurement tables for the most recent
wind work are tracked in `PLAN_PHYSICS_FIXES.md`.

Last updated: 2026-07-26.

## Rules learned from prior investigations

1. Compare time averages, not isolated snapshots. Seasonal phase aliasing can
   make two valid samples look like centuries of climate drift.
2. Validate on both synthetic fixtures and real terrain. Either one alone can
   hide important regressions.
3. Do not widen a test threshold until the change has been reproduced and
   attributed numerically.
4. A golden state detects behavior changes; it does not prove realism.
5. Regional boxes and zonal means must be considered together. Optimizing one
   can make the other worse.
6. Keep refactors, physics changes, calibration changes, and fixture
   regeneration in separate changes.

## Confirmed fixes

### Air–surface coupling regression

A July 2026 rewrite warmed the global mean by roughly 4 K, collapsed the
equator-to-pole gradient, and nearly eliminated Northern Hemisphere sea ice.
The regression was isolated to `_evolve_temperature`:

- the land branch incorrectly applied ocean-style equal-and-opposite surface
  coupling;
- the atmospheric relaxation toward the underlying surface had been removed.

The shipped correction keeps equal-and-opposite mixed-layer exchange over ocean
only and restores atmospheric relaxation toward the surface. The previously
blanket-slow Earth benchmark and polar-balance tests now run in the routine
suite so this class of regression cannot remain hidden.

### Mid-latitude continental precipitation

The dry US Midwest was not moisture-limited. It carried substantially more
humidity than the Canadian Prairies or Central Europe but had almost no ascent
because the prescribed meridional circulation placed strong divergence over
38–45°N.

`ferrel_v_centre_deg` separates the meridional-cell center from the zonal-jet
center. `ferrel_v_land_shift_deg` then permits an additional land-only shift so
continental convergence can move without over-wetting the ocean-dominated
40–50°N zonal mean. The current defaults are 44° and −4° respectively.

### Spherical moisture-convergence operator

The legacy precipitation kernel omits spherical metric factors and leaves pole
rows at zero. A closed-form tested spherical operator now exists behind
`spherical_metric_precip`, but remains disabled because enabling it increases
high-latitude precipitation while worsening an independent precipitation
rescale-saturation problem. Fix the target/rescale mechanism before changing
this default.

## Open correctness issues

### Calendar aliasing

ANNUAL mode advances 52 × 7 = 364 days per nominal Earth year. The 1.2422-day
phase slip produces an approximately 294-year false oscillation in seasonal
metrics. MONTHLY mode has the same class of problem at 30 days per cycle.
Substep policy must be derived from `PlanetParams.orbital_period_days`.

### Carbon-cycle flag coupling

`enable_carbon_cycle=False` also disables CO₂/CH₄ radiative forcing. It does not
mean “hold concentrations fixed.” Fixed-concentration experiments must keep
radiative forcing active while suppressing only reservoir evolution.

### Diagnostic-wind cache key

`_RELAX_CACHE` includes `pgf_continentality_amp` but omits
`ferrel_v_centre_deg` and `ferrel_v_land_shift_deg`. Interleaved parameter
experiments at the same simulated time can reuse stale wind targets.

## Open structural physics gaps

### Precipitation rescale saturation

Raw precipitation production is too low, and the row/global multiplier
disproportionately amplifies naturally dry subtropical bands. This compresses
the contrast between deserts and mid-latitudes and leaves little headroom for
the correct spherical convergence operator. The next fix belongs in the target
mechanism, not another scalar driver adjustment.

### Time-scale divergence

MONTHLY and DAILY modes still disagree materially in polar land fraction,
aridity, humidity, and winter temperature. Temperature prognostic substepping
overshoots the DAILY winter target, while prognostic wind substepping can
degrade desert precipitation. Neither gate should be enabled by default without
a new structural correction.

### Soil-moisture bistability

Real-terrain 45–65°N soil moisture can remain pinned near its hard floor. Once
surface and deep stores dry sufficiently, reduced evaporation and precipitation
reinforce the dry state. Existing synthetic long-run guards do not fully expose
the real-terrain saturation.

### Deep ocean

The deep layer exchanges vertically with the local surface but has no
overturning or lateral abyssal transport. Its effective equilibration time is
roughly 2,200 years, and its eventual target is close to area-weighted surface
ocean temperature (~25°C), not the observed 1–4°C abyss. Changing only the
exchange rate makes the endpoint less realistic.

### Clouds and climate sensitivity

Global cloud fraction remains far below Earth observations, and the measured
model equilibrium climate sensitivity is around 1.8 K. These are model-realism
gaps, not current numerical-instability failures. Revisit them after the
precipitation target and spherical-operator work to avoid tuning around known
structural errors.

### Regional mechanisms

- Atacama remains too wet because cold-current/coastal-fog desert physics is
  absent.
- Salinity uses a temperature-derived evaporation proxy rather than the actual
  freshwater flux.
- Land ice has age/hysteresis but no mass balance, thickness, flow, or sea-level
  coupling.
- Land water has no lateral runoff, rivers, or lakes.

## Current validation status

The fast and slow suites contain analytic operators, sign tests, wiring checks,
golden-state regression, Earth benchmarks, conservation checks, and multi-year
drift guards. The largest remaining validation gaps are:

- automated real-terrain regional scoring;
- gridded ERA5/CRU spatial correlation and RMSE;
- production-resolution fidelity/performance checks;
- GUI smoke and accessibility coverage;
- an explicit guard against orbital-cycle phase slip.

When this summary conflicts with a newer measured execution log in
`PLAN_PHYSICS_FIXES.md`, the newer tracked measurement takes precedence.
