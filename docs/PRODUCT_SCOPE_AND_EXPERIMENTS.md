# Product scope and experiment policy

PlanetSim's supported product is an interactive, Earth-calibrated climate
simulator with a deterministic CPU headless path for validation and bounded
parameter searches. It is a climate-system exploration tool, not an
operational forecasting system or a general-circulation model.

## Supported baseline

The supported configuration is the default `EARTH` parameter set, evaluated
against the regression contract in `CURRENT_BASELINE.md`. The GUI, save/load
format, CPU headless runner, documented time-scale modes, and the routine test
suite are part of that supported baseline.

Earth is the only calibrated planet. Mars and other parameter combinations are
architecture and stability exercises: they must remain numerically valid, but
they do not carry an Earth-equivalent accuracy claim. The optional JAX/GPU
screening model and ExoPlaSim exchange workflow are research tools; neither is
a replacement for the CPU model or its real-terrain promotion gates.

## What does not enter the default product automatically

A mechanism stays experimental when it is incomplete, has not cleared the
real-terrain promotion gates, materially worsens the baseline, or has not been
measured at a relevant duration and resolution. Experimental code may be
useful, tested, and documented; default-off is a deliberate product decision,
not an indication that it is disposable.

Promoting an experiment requires all of the following:

1. Unit and directional tests for the mechanism, including a default-off
   no-op test where applicable.
2. A bounded 64x128 screen with a recorded baseline and no compensating
   regression hidden by an aggregate score.
3. A matched real-terrain 128x256, five-year spin-up/five-year evaluation
   comparison that meets the acceptance gate.
4. Routine and slow-suite verification, long-run drift checks where a slow
   reservoir is involved, and DAILY/coarse-mode comparison where cadence can
   affect the result.
5. An update to `CURRENT_BASELINE.md`, relevant user documentation, and this
   matrix. A golden-state regeneration must state the intentional mechanism
   and observed change.

The promotion scorecard also rejects a candidate that improves global map
metrics by worsening a tracked named-region target error or the composite
regional reference error. Those safeguards apply whenever real-terrain
diagnostics are present; generic unit-test reports remain intentionally small.

An experiment is retired when its premise is disproven, its behavior remains
net-negative after a bounded investigation, or its maintenance cost is not
justified by an intended product capability. Retired configuration names remain
readable only through an explicit compatibility migration, never as active
controls for new users.

## Compatibility-only settings

`simulate_step` still accepts `ocean_exchange_floor`, `ocean_exchange_span`,
and `latent_cooling_coeff` so legacy scripts and saved configurations keep
loading. They are confirmed no-ops, are excluded from all current optimizer
configuration, and emit a `DeprecationWarning` if a caller supplies a
non-default value. They must not be surfaced in new UI or configuration work.

## Default-off gate matrix

Each name below is a `PlanetParams` boolean that defaults to `False`. Nested
gates require their parent path; enabling a child alone does not make a new
default behavior.

| Area | Gates | Status and promotion/retirement condition |
|---|---|---|
| Orbital experiments | `enable_milankovitch_cycles` | Keep experimental. The forcing is wired, but there is no dedicated multi-millennial experiment harness or calibrated ice-age response. Promote only with that harness plus stable long-run ice/ocean validation. |
| Land and seasonal ocean | `enable_force_restore_land`, `derive_ocean_seasonal_lag` | Keep experimental. The force-restore replacement has not cleared regional CRU/Köppen gates; the derived ocean lag is unit-tested but lacks real-terrain validation. Each must independently clear the promotion protocol. |
| Humidity transport | `humidity_advection_cfl` | Keep experimental until the Courant-number path is compared across grid sizes and time scales against the calibrated fixed-divisor path. |
| Surface hydrology | `enable_surface_hydrology` | Do not promote. D8 routing works, but no channel capacity/velocity permits unphysical basin pooling. Redesign around capacity and lateral spill before recalibration. |
| Condensate and convection | `enable_prognostic_condensate`, `enable_stability_aware_condensation`, `enable_two_layer_convective_adjustment`, `enable_cloud_precipitating_condensate_partition`, `enable_separate_precipitating_hydrometeors`, `enable_hydrometeor_transport`, `enable_simplified_betts_miller_convection` | Keep as one experimental closure family. It cannot replace the default rainfall path until it closes water/energy budgets and improves real-terrain precipitation and cloud skill together. |
| Conserved column water | `enable_prognostic_column_water`, `enable_energy_limited_evaporation`, `enable_humidity_dependent_downwelling_longwave`, `column_water_use_bulk_condensate_rainfall`, `enable_pressure_coordinate_moisture_closure`, `enable_prognostic_overturning_heat_reservoir`, `enable_pressure_coordinate_mse_transport` | **Retain the pressure-mass budget plus derived heat reservoir; do not promote.** A unit audit corrected cloud mass being read as mixing ratio by radiation. The corrected stable 64x128 screen fails climate (0.530 mm/day global rain, 0.965 polar rain, cloud 0.434, error 0.870; omega RMS 0.00161/0.00175). The subsequent separate-cloud/hydrometeor path passes mass/footprint tests. Its raw-T stability singularity is repaired using potential temperature, but the full compact screen remains rejected (1.336 mm/day global rain, 4.670 polar rain, cloud 0.539, error 0.955; omega RMS 0.0411/0.0234). Directly transporting MSE with those shared winds is also rejected (0.481 mm/day global rain; +28.6/-94.7 PW peak transport): the next solve must constrain circulation with MSE, rather than use a coefficient or cap. |
| Diagnosed overturning | `enable_two_level_thermally_direct_overturning`, `two_level_thermally_direct_overturning_speed_m_s`, `enable_three_level_pressure_column`, `enable_closed_three_level_thermodynamics`, `enable_diabatic_interface_mass_flux`, `enable_shared_pressure_coordinate_circulation`, `enable_three_level_horizontal_mass_flux_closure`, `enforce_three_level_mass_closure`, `enable_native_balanced_pressure_dynamics`, `enable_native_balanced_diabatic_overturning`, `enable_native_balanced_moist_static_energy_overturning`, `native_balanced_mse_use_toa_radiative_target`, `enable_three_level_flux_form_exchange` | **Retain the shared-circulation foundation; redesign its moisture closure.** It derives meridional winds, divergence, omega, and transport from one latent-heating/static-stability solve, with no strength/damping scalar. Its 64x128 screen repairs omega (0.00198/0.000688 Pa/s) and transport (-9.66 PW) at 42.3 s, but fails climate (0.954 mm/day rain; KÃ¶ppen 0.501/0.287; error 0.696). Do not restore raw divergent winds or tune its speed; build circulation-coupled evaporation/condensate/fallout production next. |
| Land ice | `enable_land_ice_dynamics` | Do not promote. Terrain-slope flow, albedo, calving/freshwater, coastline feedback, and multi-century calibration are incomplete. |

## Current experimental-family dispositions

This is the explicit outcome of the priority-4 review. "Retain" means the
experiment remains a valid, bounded research capability and needs the stated
validation work. "Redesign" means further scalar tuning is not informative:
the missing behavior is architectural. No family is retired in this pass; each
still represents a useful, testable physical direction or compatibility-safe
research capability.

| Family | Disposition | Rationale and next admissible work |
|---|---|---|
| Orbital forcing | **Retain experimental** | The forcing is wired. Build a dedicated multi-millennial harness and validate ice/ocean stability before making an Earth-accuracy claim. |
| Force-restore land | **Redesign** | The local two-reservoir/Penman--Monteith closure is valid, but it lacks diagnosed atmospheric heat convergence and decisively fails the CRU/Koppen gate. Do not re-sweep local parameters. |
| Derived ocean seasonal lag | **Retain experimental** | The derivation is unit-tested but needs an independent matched real-terrain climate comparison. |
| CFL humidity transport | **Retain experimental** | Its numerical premise is valid, but it needs grid-size and time-scale comparison with the calibrated fixed-divisor path. |
| Surface hydrology | **Redesign** | D8 routing lacks channel capacity/velocity and lateral spill, allowing unphysical basin pooling. Add those physical constraints before recalibration. |
| Condensate/convection closure | **Redesign** | The individual gates are not independently promotable; the closure must jointly close water/energy budgets and improve real-terrain precipitation and clouds. |
| Conserved column water | **Redesign** | It replaces the empirical rainfall correction, so a local skill gain cannot validate it. The complete moisture closure must pass the normal real-terrain gate. |
| Two-/three-level overturning | **Redesign** | The normal path lacks upper thermodynamic/radiative closure for a diagnosed overturning strength. The best three-level result also worsened whole-climate skill at roughly 47x cost. |
| Land ice | **Redesign** | Flow, albedo, calving/freshwater, coastline feedback, and multi-century calibration are incomplete. |

## Inert numeric trials

The following default-zero mechanisms are also experiments and must not be
silently enabled: `temperature_amoc_scale`, `abyssal_overturning_coeff`,
`cloud_water_feedback`, `sst_land_coupling_strength`,
`sst_land_target_weight`, `orographic_upwind_footprint_km`,
`orographic_spillover_km`, `precip_land_shape_weight`,
`drybelt_seasonal_equatorward_fraction`, `storm_track_seasonal_response`,
`precip_raw_shape_weight`, `moisture_budget_tropical_cap_boost`, and
`pgf_continentality_amp`. Their detailed evidence remains in
`ACCURACY_AUDIT.md`, `FEATURES.md`, and `PRIOR_ART_IMPLEMENTATION_PLAN.md`.

## Other disabled or zero-valued controls

This inventory prevents a numeric zero from becoming an undocumented second
kind of default-off feature. It is intentionally separate from the promotion
matrix: some entries are scenario inputs or automatic-policy sentinels, rather
than candidate Earth defaults.

| Area | Controls | Current disposition |
|---|---|---|
| Scenario forcing | `aerosol_optical_depth` | Valid clear-sky Earth baseline (`0.0`). A positive value is an explicit volcanic/aerosol scenario, not a missing baseline feature. |
| Land-seasonal refinements | `land_transport_deficit_k`, `land_thermal_inertia_days`, `land_transport_seasonality`, `land_cap_softness_k`, `evap_cooling_amplitude` | Keep inert. These have individually shown useful directional behavior but no configuration has cleared the cross-resolution climate gates. Do not revisit them as independent scalar sweeps; priority 3 established that the credible follow-up is a redesigned force-restore path with atmospheric heat convergence. |
| Moisture transport | `moisture_advection_scale` | Keep inert. The added long-range transport is implemented and tested, but has no net validated Earth improvement over the calibrated path. |
| Three-level tuning companions | `three_level_divergence_filter_strength`, `three_level_divergence_filter_passes`, `three_level_balanced_thermal_wind_relaxation`, `native_balanced_pressure_relaxation`, `native_balanced_ageostrophic_timescale_hours`, `native_balanced_overturning_speed_m_s`, `three_level_diabatic_ascent_scale` | Inactive because their parent three-level gates are off. They are not independent user controls and must not be tuned outside that family’s promotion protocol. |
| Column-water tuning companion | `evaporation_downwelling_longwave_w_m2` | Inactive because `enable_energy_limited_evaporation` is off; evaluate only with the complete conserved-column-water family. |
| Cadence overrides | `precip_substep_days`, `temperature_substep_days` | Valid automatic-policy sentinels (`0.0` chooses the calibrated built-in cadence). Positive values are opt-in accuracy/performance experiments, pending long DAILY-versus-coarse validation. |

## Working order

No new experimental mechanism should be added while the supported baseline has
an unresolved higher-priority accuracy issue. Work proceeds in this order:

1. Regional precipitation/moisture budget.
2. Jet placement and hemispheric circulation asymmetry.
3. Land seasonal temperature bias and the temperature ceiling.
4. A deliberate promote/redesign/retire decision for each experimental family.

This document is the concise current policy; `PLAN.md`,
`IMPLEMENTATION_PLAN.md`, and `PLAN_PHYSICS.md` remain historical records.
The dependency-ordered remaining implementation work is maintained in
`docs/REMAINING_WORK_PLAN.md`.

## Current priority-1 disposition

The active `monsoon_east_margin_exemption=3.0` is retained. On 2026-08-11,
compact screens at 4.0 and 5.0 improved the eastern-margin regional means, but
the required matched 128×256, five-year CRU comparison rejected 5.0: regional
reference error improved 6.2%, group accuracy improved 0.15 points, and SE US
entered its target range, but precipitation log-RMSE worsened by 0.0081 and
class accuracy fell by 0.03 points. The 4.0 compact CRU screen also worsened
precipitation log-RMSE and group accuracy. Raising this scalar is not a valid
route to promote the remaining East China/S Japan precipitation improvement.

Atacama’s dry-coast residual, the eastern-margin monsoon deficits, and any
coarse-grid Central Europe discrepancy need separate, physically scoped work;
they must not be combined into another global dry-belt exemption.

The 2026-08-11 regional moisture-budget decomposition confirms that these are
not one shared missing transport term. In the compact seasonal baseline,
Atacama produces 0.076 mm/day of raw rain and ends at 0.176 mm/day because the
allocator adds 0.100 mm/day; East China produces 0.855 and ends at 1.823
because the allocator adds 0.967; S Japan is 2.307 to 2.620 (+0.313); and
Central Europe is the opposite case, 7.648 to 2.623 (-5.025). The first three
need distinct coastal/monsoon moisture pathways, while Central Europe needs a
mid-latitude rain-production/track diagnosis. There is no honest shared scalar
repair.

The existing cold-side SST target-share pathway was re-screened at its recorded
physical strength (`sst_land_target_weight=1.5`) against the current 64x128,
one-year CRU gate. It was rejected before long validation: precipitation
log-RMSE worsened by 0.00007, Köppen class accuracy fell by 0.07 points, and
the Atacama, Central Europe, East China, and S Japan target errors all
increased. It remains implemented but inert. The diagnosis priority is thereby
complete: retain the baseline, do not add a regional exemption, and proceed to
the separate jet-placement and hemispheric-asymmetry priority.

The extracted two-level thermally direct overturning experiment was compact-
screened on 2026-08-11 at 0.1–1.0 m/s. It is real, two-layer mass-conserving,
and improves CRU precipitation skill modestly, but its best regional composite
result (0.5 m/s) still lowers Köppen class accuracy by 0.009 points and worsens
the Central Europe, S Japan, and SE US target errors. It remains default-off;
do not advance it to long-run validation until a physically diagnosed strength
or structure resolves that trade-off.

The read-only baseline diagnostic added after that screen establishes why a
diagnosed strength is not yet justified in the normal 1.5-layer path: its
seasonal mean has tropical latent heating of about 124 W/m² and both wind
layers, but no upper-layer temperature/humidity or radiative-tendency state.
Its assumed 40% lower / 60% upper meridional mass-flux residual is about
23 m/s RMS. The existing upper wind is therefore a jet layer, not a diagnosed
compensating overturning branch. A physically diagnosed two-layer overturning
requires an upper thermodynamic reservoir and radiative tendency/closure;
adding another speed parameter would just repeat the rejected tuning path.

The 2026-08-12 seasonal regional pathway dossier confirms that no remaining
priority-1 scalar is admissible. Atacama has offshore southeast-Pacific source
flow (-0.69 m/s), negative lower moisture-flux convergence (-1.34e-8 q/s), no
ascent, and only a -0.057 K upwind SST anomaly; the existing cold-SST target
path was already rejected. East China and South Japan have negative physical
moisture-flux convergence in every sampled season and source-to-land flow of
the wrong sign through most of the year; their allocator corrections are
therefore not a monsoon mechanism. Central Europe instead has positive
convergence but 7.65 mm/day raw rain (10.70 in JJA) followed by a -5.03
(-8.45 in JJA) allocator correction from the static storm-track/ascent path.
The sole existing seasonal storm-track control at 0.3 increased that JJA raw
rain to 10.80 mm/day, worsened precipitation log-RMSE by 0.00209, and regressed
Atacama and S Japan target errors. It is rejected.

No priority-1 target, SST-weight, or storm-window scalar should be retuned.
The eastern-margin/coastal requirements hand off to the vertical
thermodynamic/circulation architecture, and Central Europe needs a diagnosed
transient rain-production path that uses those states rather than a latitude
window. The supported baseline remains unchanged.

The Phase-2 closed thermodynamic-column kernel is now implemented in
`pressure_column.py`, with its explicit state/budget contract in
`VERTICAL_THERMODYNAMIC_CLOSURE.md`. It is a pure, unconnected experimental
kernel rather than an enabled feature: runtime coupling awaits an auditable
radiative/surface/precipitation source adapter, so the supported baseline and
every default-off experimental switch remain unchanged.

## Current priority-2 disposition

Seasonal jet diagnostics are now emitted by the real-terrain report rather
than reading one final-state snapshot. At the supported 64x128, one-year
baseline, lower cores average 43.8 degrees N / 45.2 degrees S at 6.72 / 6.75
m/s, so there is no material lower-level hemispheric strength asymmetry at
that resolution. Upper cores are persistently equatorward at 26.7 degrees N /
28.6 degrees S, with 22.4 / 24.0 m/s cores; the modest 7.4% stronger SH core
is not evidence for a separate asymmetry repair.

The one directly relevant active-shape screen, widening
`wind_upper_hadley_edge_deg` from 24 to 30 degrees, did not move either
seasonal upper-jet latitude and reduced their core strengths. Despite a tiny
precipitation log-RMSE gain (0.00022), it reduced both Köppen group and class
accuracy and regressed Central Europe and East China target errors. It is
rejected before long validation. The persistent placement error is therefore
the known 1.5-layer thermal-gradient/momentum-structure limitation, not a
missing width value. Do not tune the upper PGF or Hadley-edge scalars further;
the supported baseline remains unchanged and the next priority is land
seasonal temperature bias and its ceiling.

## Current priority-3 disposition

The supported land-temperature path remains the calibrated seasonal-amplitude,
continentality, transport, and small surface-energy closure. It is not a
complete surface-energy model: its latitude-only ceiling is still a numerical
surrogate for unresolved land and atmospheric heat transport. The active
configuration is nevertheless materially better than the legacy undamped path
on the regression-gated temperature and Koppen measures, so removing or
retuning the ceiling by itself is not justified.

The physically motivated replacement is present as the default-off
`enable_force_restore_land` path. It replaces, rather than stacks on, the
legacy seasonal blend and ceiling with a two-reservoir force-restore surface
and moisture-dependent Penman--Monteith partition. Its local annual-cycle
shape is more physical, but it does not yet represent the atmospheric
eddy/storm-track heat convergence that the supported path currently supplies.

A representative current CRU A/B at 64x128 (one year spin-up plus one year
evaluation, 30-day restore time, 12 MJ m-2 K-1 deep capacity, and 2000 s m-1
dry resistance) remains decisively outside the promotion gate: temperature
RMSE is 6.279 -> 7.679 C, Koppen group accuracy 0.674 -> 0.603, and class
accuracy 0.388 -> 0.248. Precipitation log-RMSE improves 1.406 -> 1.363, but
that single gain cannot compensate for the temperature and biome regressions.
The wider published-range screen is recorded in
`docs/PRIOR_ART_IMPLEMENTATION_PLAN.md`.

Priority 3 is therefore complete as a supported-baseline decision: retain the
current default and keep force-restore experimental. Do not perform another
local parameter sweep or introduce a softer/global ceiling. A future promotion
attempt needs a separately designed, diagnosed atmospheric heat-convergence
term in the replacement architecture, with a physically defined forcing and
the same CRU/Koppen gates; copying the legacy latitude trapezoids would only
move the old surrogate into the new branch. The next work priority is the
explicit promote/redesign/retire decision for each remaining experimental
family.
