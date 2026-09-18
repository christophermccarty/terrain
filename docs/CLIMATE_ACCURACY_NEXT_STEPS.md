# Climate accuracy diagnosis and next steps

Status: current decision document, 2026-08-21.

## Decision

Stop trying to obtain the remaining Earth accuracy by tuning the supported
1.5-layer climate path. Freeze it as the playable Earth baseline and build a
second, independently validated climate core behind the existing state/runner
boundary. The new core should begin as a coarse moist primitive-equation or
equivalent mass-conserving dynamical model, not as another collection of
regional corrections.

The supported path is not simply missing one parameter. It combines useful
physics with prescribed climatological structure. That has produced a good,
fast baseline, but it also creates a ceiling: replacing any one empirical
support exposes several other missing closures at once, so a physically better
component often scores worse before the coupled system is complete.

## Evidence from the current baseline

- The tracked compact contract reproduces and passes: Köppen group/class
  accuracy 0.674/0.388, global precipitation 3.00 mm/day, and global
  temperature 17.96 C.
- The matched 128x256, five-year spin-up/five-year evaluation is the more useful
  accuracy measurement: CRU monthly temperature RMSE is 5.73 C with +1.61 C
  bias; precipitation log-RMSE is 1.46 and log-correlation only 0.459;
  Köppen group/class accuracy is 0.709/0.422.
- Global cloud fraction is 0.131 in that longer run. Satellite observations
  put typical Earth cloud cover near 0.67. The cloud deficit is large enough
  to corrupt both reflected shortwave and outgoing longwave radiation, which
  the SESAM TOA experiment independently exposed.
- The supported circulation diagnostic reports peak lower-layer energy
  transports of hundreds of petawatts. This is not a valid full-column Earth
  transport estimate because the supported model lacks compensating
  thermodynamic layers. Earth total poleward transport peaks near 5.7 PW. The
  result is therefore a missing-closure signal, not a scalar calibration target.
- The project's offline ExoPlaSim reference is worse on temperature and Köppen
  at T21, but its annual precipitation log-correlation is 0.744 versus roughly
  0.46 for PlanetSim. That is strong evidence that resolved three-dimensional
  circulation, rather than another rain multiplier, is the remaining rainfall
  lever.
- SESAM P0-P6 is complete, wired, and boundedly calibrated. Its best full-chain
  compact state still loses to the supported baseline (0.582/0.282 Köppen,
  5.00 mm/day precipitation, 282.8 K). The experiment is valuable evidence:
  porting formulas without their native coupled state, time stepping, and
  calibration context does not create a climate core.

## Structural ceiling in the supported path

The following mechanisms are appropriate for a fast game model but prevent
the residual errors from being independently fixable:

1. Precipitation is allocated toward a prescribed Earth zonal profile in
   `atmosphere.py`. Local moisture production is budget-bounded, but the final
   field is not solely the result of prognostic column water, convergence,
   condensation, and fallout.
2. Surface equilibrium temperature is driven 90% by a latitude-based calibrated
   temperature target, with polar flux corrections, floors, SST caps, and an
   ocean temperature floor. Land then receives a constant per-call seasonal
   blend unless an experimental replacement is enabled.
3. Wind combines prognostic pieces with prescribed circulation cells, waves,
   storms, and relaxation. There is no supported vertically closed mass,
   momentum, and moist-static-energy circulation.
4. Clouds are mostly diagnostic and far too sparse. Cloud condensate, cloud
   fraction, optical depth, precipitation, and radiation are not one conserved
   coupled closure in the supported path.
5. The ocean carries useful mixed-layer and gyre parameterizations, but not a
   dynamically consistent atmosphere-ocean momentum and energy exchange.

This explains the recurring result in the experiment logs: a local physical
replacement removes a compensating empirical term and loses the joint gate.
The gate is doing its job; the candidate is incomplete at the system level.

## Systems that are actually missing

Priority order for modern-Earth climate accuracy:

1. A conservative atmospheric dynamical core with at least 3-5 vertical
   layers, prognostic pressure/mass and horizontal momentum, hydrostatic
   balance, and resolved vertical mass continuity.
2. A single coupled column closure for water vapour, cloud condensate,
   precipitation/fallout, latent heating, and moist convection.
3. A boundary layer and surface-energy budget with finite heat capacities,
   conservative sensible/latent exchange, and no latitude-temperature target.
4. Cloud optical properties connected to shortwave and longwave radiation,
   validated against TOA and surface energy-budget data as well as cloud
   fraction.
5. Only after 1-4: atmosphere-ocean stress, coastal upwelling, mixed-layer
   entrainment, and a bounded overturning/deep-ocean model. These matter for
   Atacama, coastal East Asia, and long perturbation experiments, but the
   current atmosphere cannot transmit their signal reliably.

Hydrology, dynamic ice sheets, vegetation, impactors, and carbon-cycle features
can continue against the frozen playable baseline. They should not be used to
repair modern-Earth atmospheric skill.

## Build plan

### Phase A - freeze and instrument (1-2 weeks)

- Name the current default `legacy_earth_v1` and stop default-changing climate
  sweeps.
- Add one versioned validation manifest that identifies the terrain, reference
  products, periods, grid, spin-up, code revision, and tolerances.
- Add ERA5/CERES-style targets for SLP, 850/500/250 hPa winds and temperature,
  column water, cloud fraction, TOA reflected shortwave, OLR, surface latent and
  sensible heat, and meridional energy transport.
- Make the 128x256 5+5-year report the climate promotion scorecard. Keep the
  64x128 1+1-year report as a regression/smoke gate only.

### Phase B - prove the new core on idealized cases (4-8 weeks)

- Start at 32x64 or T21 with 3-5 sigma/pressure layers and a short internal
  timestep. Do not start at the GUI's 512x1024 display grid.
- Pass Held-Suarez dry dynamics, a gray-radiation moist aquaplanet, solid-body
  advection, hydrostatic resting atmosphere, mass/energy conservation, and
  timestep/grid-convergence tests before adding Earth terrain.
- Use the stored ExoPlaSim fields as an independent circulation oracle, not as
  values to nudge toward.

### Phase C - add the coupled moist column (4-8 weeks)

- Implement large-scale saturation adjustment first; add a simple
  Betts-Miller-like convection relaxation only if needed.
- Couple condensate, latent heating, cloud fraction/optical depth, and two-stream
  radiation atomically. No prescribed precipitation target is allowed in this
  branch.
- Admit the branch on budgets and idealized behavior before asking it to beat
  the heavily calibrated legacy Köppen score.

### Phase D - Earth boundary conditions and calibration (4-8 weeks)

- Add real terrain, slab ocean, sea ice, and force-restore land with conservative
  flux exchange.
- Calibrate a small set of global physical/closure parameters with held-out
  validation: tune on selected years or regions, evaluate on the rest.
- Require joint improvement in T/P maps, circulation, clouds, and energy budget.
  Köppen is the final emergent score, not the optimization target.

### Phase E - choose product integration

- If the new core clears the long gate, make it the scientific backend and keep
  the legacy core as the fast interactive mode.
- If it does not, integrate ExoPlaSim or another maintained intermediate GCM as
  an optional offline/high-fidelity backend instead of restarting scalar tuning.

## Validation changes needed

- Do not optimize primarily against Köppen. It is a thresholded summary of T/P;
  optimizing it directly encourages compensating errors near class boundaries.
- Score all 30 reference classes or report group accuracy as primary. The
  current 19-class vocabulary folds multiple reference classes, so fine-class
  accuracy has a built-in interpretation problem.
- Use the Beck et al. numeric GeoTIFF plus confidence layer rather than a
  palette-decoded PNG, and down-weight uncertain boundary cells.
- Report area-weighted bias, RMSE, correlation, seasonal amplitude/phase, and
  distributional extremes separately. Never collapse promotion to one composite
  scalar.
- Replace the rule that every named region must be non-regressing by a declared
  Pareto/tolerance policy. Otherwise harmless noise or a tiny local trade blocks
  architectural improvements while the incumbent retains large compensating
  errors.
- Separate regression, scientific admission, and product promotion. A new core
  may pass conservation and idealized admission while still being below the
  legacy Earth calibration; that is progress, not a reason to abandon it.

## Why this is the shortest route to new features

The legacy baseline is already good enough for interactive scenarios and can
be feature-frozen. A parallel climate core removes the need to make every
physical experiment immediately preserve a decade of empirical calibration.
It also produces the state variables needed for volcanic aerosol forcing,
impact dust, greenhouse perturbations, orbital experiments, and non-Earth
atmospheres without adding another Earth-specific correction for each case.

Primary external references:

- Willeit et al. (2022), CLIMBER-X/SESAM model description:
  https://gmd.copernicus.org/articles/15/5905/2022/
- Frierson, Held, and Zurita-Gotor (2006), gray-radiation moist GCM:
  https://doi.org/10.1175/JAS3753.1
- Paradise et al. (2022), ExoPlaSim:
  https://doi.org/10.1093/mnras/stac172
- Beck et al. (2023), 1991-2020 Köppen-Geiger v2 maps:
  https://doi.org/10.1038/s41597-023-02549-6
- Trenberth and Fasullo (2017), observed meridional heat transport:
  https://doi.org/10.1002/2016GL072475
- CERES EBAF global radiation-budget products:
  https://ceres.larc.nasa.gov/Data/
