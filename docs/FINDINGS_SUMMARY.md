# Durable findings summary

This tracked document preserves conclusions that were originally established by
local, generated, or overnight experiments. Raw logs and large result artifacts
remain intentionally ignored. Detailed measurement tables for the most recent
wind work are tracked in `PLAN_PHYSICS_FIXES.md`.

Last updated: 2026-08-01.

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

### High-latitude soil desiccation

Land soil moisture at 45-65°N/S sat pinned at its 0.05 floor for 84-100% of
cells across every longitude, uniformly across desert and continental-interior
boxes alike — a genuine stable attractor under current physics, confirmed by
continuing a real-terrain save several years and observing no recovery. The
single-layer soil bucket is documented as bistable with no stable middle
ground, and the balance landed on the collapsed branch almost everywhere at
this latitude.

Fixed by enabling `PlanetParams.soil_deep_gain_rate` (0.0 → 0.0005), a
deep-soil-layer knob shipped inert in an earlier session after being found
ineffective — before a later, separate fix (gating desert `land_evap` by
`subsidence_suppression`) gave raw precipitation the desert-vs-continental
differentiation the deep layer needed to inherit. Real-terrain A/B: `soil_deep`
now separates desert boxes (0.05-0.25) from continental-interior boxes
(0.30-0.44) for the first time, with every named box moving modestly toward
its Earth precipitation target and no reordering. See `PLAN_PHYSICS_FIXES.md`
for the full measurement table.

### Half-resolution precipitation shortcut

At the production resolution (512x1024), the automatic half-resolution
precipitation path (used for H>=256) leaves zonal-band climate structure
essentially unaffected but is a major contributor to soil-moisture
floor-pinning: 27% of global land sits at the 0.05 floor at half-resolution
versus ~0% at full resolution, and named regional boxes come out
systematically 4-33% drier. Full resolution costs 2.8x wall time. Rather
than change the default, a GUI toggle ("Full-res precipitation") now lets
users opt into full resolution when regional/soil-moisture realism matters
more than speed; the automatic half-resolution default is unchanged. See
`PLAN_PHYSICS_FIXES.md` for the full measurement table.

### Mid-latitude continental precipitation

The dry US Midwest was not moisture-limited. It carried substantially more
humidity than the Canadian Prairies or Central Europe but had almost no ascent
because the prescribed meridional circulation placed strong divergence over
38–45°N.

`ferrel_v_centre_deg` separates the meridional-cell center from the zonal-jet
center. `ferrel_v_land_shift_deg` then permits an additional land-only shift so
continental convergence can move without over-wetting the ocean-dominated
40–50°N zonal mean. The current defaults are 44° and −4° respectively.

### Spherical moisture-convergence operator (shipped as default, 2026-07-26)

The legacy precipitation kernel omitted spherical metric factors and left pole
rows at zero. `spherical_metric_precip` and `spherical_metric_clouds` are now
both `True` by default (`planet_params.py`) — a dimensionally-incompatible
epsilon in the normalization that had nearly erased the corrected signal was
found and fixed alongside the flip. The small real-terrain reference-error
cost (~2-3%) was accepted to remove the coordinate error and support
high-obliquity/polar worlds; it still compounds the separate, still-open
precipitation rescale-saturation problem below rather than fixing it. See
`PLAN_PHYSICS_FIXES.md`'s 2026-07-26 "spherical precipitation and cloud
metrics shipped" entry.

### Calendar aliasing (fixed 2026-07-26)

ANNUAL mode used to advance 52 × 7 = 364 days per nominal Earth year, a
1.2422-day phase slip that produced an approximately 294-year false
oscillation in seasonal metrics; MONTHLY mode had the same class of problem
at 30 days per cycle. Fixed by centralizing the substep schedule into
`time_policy.py`'s `substeps_for_mode`/`cycle_days`, both derived from
`PlanetParams.orbital_period_days` directly (commit `dfaf41c`). Verified
2026-08-01 via `testing/test_generalize_time_orbit.py`; see
`docs/ACCURACY_AUDIT.md` C2.

### Diagnostic-wind cache key (fixed)

`_RELAX_CACHE`'s key in `simulate.py` now includes `pgf_continentality_amp`,
`ferrel_v_centre_deg`, and `ferrel_v_land_shift_deg` together, so interleaved
parameter experiments at the same simulated time no longer reuse a stale wind
target from before this fix.

### Carbon-cycle flag coupling (fixed)

`enable_carbon_cycle=False` used to also disable CO₂/CH₄ radiative forcing,
silently breaking any fixed-concentration experiment that expected "hold
concentrations fixed, keep radiative physics active." `simulate_step` now takes
an independent `apply_greenhouse_forcing: bool = True` parameter, and the
greenhouse-forcing gate reads `apply_greenhouse_forcing`, not
`enable_carbon_cycle` (`simulate.py:788`). See
`testing/test_co2_feedback.py::test_fixed_co2_keeps_greenhouse_forcing_active`.

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

- Atacama remains too wet (~207 mm/yr vs. a <50 target). A diagnostic
  west-coast/dry-belt suppression gate (`coastal_upwelling_fog_strength`,
  default 0.5) now exists and gave a real, isolated improvement
  (123→102 mm/yr), but it is a proxy gate, not real simulated ocean
  upwelling — no SST-coupled cold-current physics exists. See
  `docs/ACCURACY_AUDIT.md` A1/D3.
- Salinity uses a temperature-derived evaporation proxy by default. A real
  freshwater-flux pathway (river runoff into `evolve_salinity`) exists behind
  `PlanetParams.enable_surface_hydrology` (default `False`), so this is only
  true while that flag is off.
- Land ice now has a real mass-balance/thickness/flow mechanism and a
  `sea_level_change_m` diagnostic behind `PlanetParams.enable_land_ice_dynamics`
  (default `False`), not just age/hysteresis — but still has no albedo
  coupling, no calving→salinity coupling, and no mask/coastline feedback. See
  `docs/ACCURACY_AUDIT.md` E2.
- Land water can now route lateral runoff/rivers/lakes behind
  `PlanetParams.enable_surface_hydrology` (default `False`) — three real bugs
  were fixed, but the router has no channel capacity/velocity concept, so
  continent-scale basins can pool to unbounded depth (capped only by a 50m
  safety valve). See `docs/ACCURACY_AUDIT.md` E1.

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
