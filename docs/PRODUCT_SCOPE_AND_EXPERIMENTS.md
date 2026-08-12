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
| Conserved column water | `enable_prognostic_column_water`, `enable_energy_limited_evaporation`, `enable_humidity_dependent_downwelling_longwave`, `column_water_use_bulk_condensate_rainfall` | Keep experimental. These gates bypass or replace empirical rainfall correction, so promotion requires a complete validated moisture closure, not a local metric improvement. |
| Diagnosed overturning | `enable_two_level_thermally_direct_overturning`, `two_level_thermally_direct_overturning_speed_m_s`, `enable_three_level_pressure_column`, `enforce_three_level_mass_closure`, `enable_three_level_horizontal_mass_flux_closure`, `enable_native_balanced_pressure_dynamics`, `enable_native_balanced_diabatic_overturning`, `enable_native_balanced_moist_static_energy_overturning`, `native_balanced_mse_use_toa_radiative_target`, `enable_three_level_flux_form_exchange` | Keep experimental. The new two-level path reuses the existing mass-conserving thermal structure with the normal surface/upper wind states; it needs process and real-terrain validation. The best three-level configuration improved transport but made the full climate score worse and cost about 47x more. |
| Land ice | `enable_land_ice_dynamics` | Do not promote. Terrain-slope flow, albedo, calving/freshwater, coastline feedback, and multi-century calibration are incomplete. |

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
| Land-seasonal refinements | `land_transport_deficit_k`, `land_thermal_inertia_days`, `land_transport_seasonality`, `land_cap_softness_k`, `evap_cooling_amplitude` | Keep inert. These have individually shown useful directional behavior but no configuration has cleared the cross-resolution climate gates. Revisit together only after the supported precipitation and land-temperature priorities. |
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

The extracted two-level thermally direct overturning experiment was compact-
screened on 2026-08-11 at 0.1–1.0 m/s. It is real, two-layer mass-conserving,
and improves CRU precipitation skill modestly, but its best regional composite
result (0.5 m/s) still lowers Köppen class accuracy by 0.009 points and worsens
the Central Europe, S Japan, and SE US target errors. It remains default-off;
do not advance it to long-run validation until a physically diagnosed strength
or structure resolves that trade-off.
