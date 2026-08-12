# Remaining development plan

This is the active implementation plan after the supported-baseline audit.
It replaces open-ended parameter exploration with a sequence of bounded
architecture and validation workstreams. `PRODUCT_SCOPE_AND_EXPERIMENTS.md`
remains the authoritative list of supported and experimental controls; this
document states the order in which their unresolved capabilities should be
addressed.

## Operating rules

1. Preserve the default Earth baseline until a candidate passes the existing
   compact and matched long real-terrain gates. A better aggregate score never
   compensates for a regression in Köppen skill or a named regional target.
2. Start each workstream with a read-only diagnostic that measures the proposed
   physical pathway at the stage where it acts. Do not infer a mechanism from a
   downstream field that may be clipped, relaxed, or allocator-corrected.
3. Treat a failed bounded physical range as a conclusion about that mechanism,
   not an invitation to widen the same scalar sweep. The next attempt must add
   a missing state, conservation law, or spatial pathway.
4. A new experimental state or closure needs unit conservation tests,
   default-off/no-op coverage, deterministic save/load coverage, and a compact
   CRU screen before it can use a five-year 128x256 benchmark.
5. Keep one structural climate candidate active at a time. Independent,
   low-cost validation-only experiments may run between structural candidates,
   but must not change the supported default opportunistically.

## Phase 0 — baseline stewardship (ongoing)

**Purpose:** keep the calibrated product trustworthy while research branches
remain default-off.

**Work:** maintain the real-terrain report, named-region diagnostics, seasonal
jet scorecard, land-cycle metrics, and experimental-scope inventory. Refresh a
reference report only when an intentionally promoted default changes.

**Exit condition:** continuous work, rather than a one-time feature. Every
later phase must finish with routine tests, the relevant slow tests, and a
recorded promotion/rejection decision.

## Phase 1 — regional precipitation pathways (next implementation priority)

**Problem:** the allocator hides three different upstream failures. Atacama is
a dry-coast pathway, East China/South Japan are eastern-margin monsoon and
marine-moisture pathways, and Central Europe is an overproductive mid-latitude
rain/storm-track pathway. Their raw-to-final budget signs differ, so a shared
regional multiplier is invalid.

**Step 1: seasonal pathway dossier.** For each region, add only the
diagnostics needed to answer the causal question:

- Atacama: coastal lower-air humidity, onshore/offshore flow, adjacent SST,
  inversion/low-cloud proxy, evaporation, raw rain, and allocator adjustment.
- East China and South Japan: seasonal land-sea thermal contrast, lower-level
  moisture flux convergence, western-Pacific source humidity, ascent, raw
  rain, and allocator adjustment.
- Central Europe: seasonal transient/storm-track proxy, moisture convergence,
  ascent, raw rain, and the allocator's downward correction.

The reference is the seasonal pattern and the raw/final decomposition, not an
annual regional total alone.

**Step 2: implement one pathway at a time.** Choose the smallest physical
extension supported by the dossier: a coastal boundary-layer/cloud pathway for
Atacama, a diagnosed onshore/convergence pathway for the eastern margin, or a
mid-latitude transient-rain production pathway for Central Europe. The change
must alter raw production or transport in the relevant season; it may not be a
new geographic exemption or a stronger allocator target.

**Admission gate:** the compact CRU run must move the affected region's
*raw* and final precipitation in the intended direction, reduce allocator
dependence there, preserve all other named regions, and pass the standard
Köppen/temperature/precipitation gates. Only then run the matched 128x256,
five-year comparison.

**Stop condition:** if the diagnostic shows that the needed convergence or
vertical structure cannot be represented by the normal 1.5-layer circulation,
do not tune precipitation. Hand the requirement to Phase 2.

### Phase-1 first-pass result (2026-08-12)

The seasonal pathway dossier is now in the real-terrain report as
`metrics.seasonal_regional_moisture_budget`; it records the native raw/final
rain decomposition together with an explicitly labelled spherical lower
specific-humidity-flux divergence and four fixed ocean-source controls. The
current 64x128, one-year CRU baseline rules out another target-side or
geographic scalar repair:

- **Atacama** has negative physical moisture-flux convergence
  (-1.34e-8 q/s), mean flow away from its southeast-Pacific source (-0.69 m/s),
  no ascent, and only a -0.057 K mean upwind SST anomaly. The existing cold-SST
  target-share path was already rejected; it cannot represent a cold marine
  boundary layer/inversion when the modeled source flow is offshore.
- **East China** has negative physical moisture-flux convergence in every
  sampled season (-0.92 to -2.04e-8 q/s) and source-to-land flow of the wrong
  sign in DJF/MAM/JJA. Its allocator adds 0.65-1.11 mm/day, so the final rain is
  not evidence for a functioning monsoon pathway.
- **South Japan** also has negative convergence in every season and strongly
  source-away mean flow (-2.15 to -2.61 m/s). Its raw summer burst is then
  allocator-reduced by 1.64 mm/day while winter is allocator-filled, which is a
  seasonal circulation/vertical-motion defect rather than a missing annual
  moisture scalar.
- **Central Europe** has positive mean convergence but raw production of
  7.65 mm/day (10.70 in JJA), followed by a -5.03 (-8.45 in JJA) allocator
  correction. The static storm-track window and its derived ascent/convection
  drivers are the amplifier, not a lack of local vapour. Re-screening the only
  existing physical scalar, `storm_track_seasonal_response=0.3`, reduced its
  mean window but increased Central Europe's JJA raw rain to 10.80 mm/day;
  precipitation log-RMSE worsened by 0.00209 and Atacama/South Japan regional
  errors regressed. It is rejected.

**Decision:** Phase 1 has completed its diagnostic and bounded-existing-control
work with no promotable supported-baseline change. Eastern-margin monsoon and
coastal-inversion pathways require the upper thermodynamic/circulation closure
in Phase 2. Central Europe requires a diagnosed transient storm/rain-production
path rather than another latitude-window scalar; design it against Phase 2's
mass, humidity, and vertical-motion states. Do not add regional exemptions or
run another `storm_track_seasonal_response`/SST-weight sweep.

## Phase 2 — circulation and vertical thermodynamic closure

**Problem:** upper-jet latitude and thermally direct overturning cannot be
physically diagnosed in the normal path because it has upper wind but no upper
temperature, humidity, or radiative-tendency reservoir. The existing
two-/three-level experiments demonstrate useful kernels but not a promotable
whole-climate closure.

**Step 1: architecture design before code.** Specify layer masses/pressures,
prognostic upper temperature and humidity, radiative and latent-heating terms,
vertical exchange, horizontal mass closure, and the exact energy and water
budgets. State which existing experimental state is reused and which legacy
paths are bypassed; avoid a second parallel wind model.

**Step 2: kernel proof.** Implement the smallest closed vertical column with
tests for mass conservation, water conservation, non-spurious energy sources,
stable timestep behavior, save/load, and default-off bit identity. Add
diagnostics for layer temperatures/humidity, radiative tendencies, vertical
mass residual, cross-equatorial transport, and seasonal jet cores.

**Step 3: climate admission.** Screen the complete closure, not isolated
companions, at 64x128. Advance only a candidate that improves the diagnosed
jet/overturning target while preserving the existing climate gate. Validate
the survivor at 128x256 for five years and compare cost against the supported
path.

**Stop condition:** an upper-wind-only or fixed-speed result is not evidence
for promotion. If the closed column cannot meet the compact gate, retain it as
experimental and do not carry its tuning scalars into the supported model.

### Phase-2 architecture and kernel result (2026-08-12)

The existing three-level experiment was inspected before any new tuning. Its
best full-resolution candidate reduced the net cross-equatorial transport to
-3.44 PW, but lowered Köppen group/class skill (0.7088/0.4223 to
0.7015/0.3780), made the composite reference error 4.26x worse, and cost
46.6x the supported runtime. It remains rejected as a promotion candidate;
the issue is structural, not an untried upper-wind or mass-closure value.

`docs/VERTICAL_THERMODYNAMIC_CLOSURE.md` now fixes the replacement contract.
`pressure_column.evolve_closed_three_level_thermodynamic_column` implements
the first, pure finite-volume proof: mass-weighted water and moist static
energy move together under diagnosed interface omega; radiation is an explicit
energy input; and condensation conserves moist static energy while converting
latent energy to temperature. It deliberately does **not** contain the old
lapse-rate relaxation, temperature clipping, rain sink, or a target term.
Unit tests cover conservation, latent conversion, source accounting, timestep
subdivision, and invalid inputs.

The kernel is now wired behind the default-off
`enable_closed_three_level_thermodynamics` gate. The resolved host temperature
solver remains the explicit radiative/surface split step; the adapter then
applies mass-weighted vertical exchange and phase conversion, routes the lower
temperature into `PlanetState.air_temperature`, and emits water/MSE residuals.
The old three-level lapse relaxation and fractional latent heating are bypassed
within this gate only.

The mandatory 64x128, one-year spin-up/one-year evaluation screen rejects this
first complete candidate before any long run: global air temperature collapsed
to 164.36 K, precipitation to 0.245 mm/day, reference error was 4.429, and
interface omega RMS was 14.58/29.99 Pa/s. Those values violate the vertical
mass-flux CFL condition by transferring more than one pressure layer in a
daily step. This is a resolved-wind/interface-mass-flux architecture failure,
not an exchange or damping value to tune. **Phase 2 is therefore retained as
experimental and redesign-required:** the next admissible work is a vertically
consistent circulation/mass-flux derivation with an explicit CFL bound, not a
long benchmark or another closure scalar sweep.

The first such derivation, `enable_diabatic_interface_mass_flux`, is now also
complete and rejected as a whole-climate candidate. It derives zonal-mean
omega from prior raw-column latent heating and resolved static stability, then
substeps the conserved finite-volume transport to a 0.25 layer-Courant bound.
At 64x128 it removes the raw-wind vertical pathology (omega RMS
14.58/29.99 -> 0.0248/0.0229 Pa/s) and prevents the temperature collapse, but
still produces only 1.34 mm/day global rain, 0.571/0.317 KÃ¶ppen group/class
accuracy, 0.545 reference error, and 2177 PW cross-equatorial transport.

**Next Phase-2 architecture:** do not tune this branch. Replace the separate
raw lower/middle/upper wind fields plus independently diagnosed omega with one
shared pressure-coordinate mass/energy circulation solve. It must emit winds,
layer divergence, omega, and horizontal moist-static-energy transport from one
continuity-constrained state, then clear unit conservation/CFL tests before it
returns to a compact CRU screen.

## Phase 3 — land temperature replacement with diagnosed heat convergence

**Dependency:** Phase 2 must first provide a diagnosed atmospheric heat-
convergence term or an equivalent validated forcing. The force-restore branch
already proves that local radiative/turbulent/conductive physics alone is not
enough.

**Work:** feed a physically defined atmospheric heat-convergence tendency into
the two-reservoir force-restore land path. Retire the latitude-only land cap
and legacy seasonal transport terms only inside that A/B-able replacement;
never copy their latitude trapezoids into the new branch. Preserve the
soil-moisture-dependent heat capacity and Penman--Monteith partition already
implemented there.

**Admission gate:** improve CRU temperature RMSE without loss of Köppen group
or class accuracy, improve the anchor-free warmest/coldest-month thresholds,
reduce plateau/ceiling dependence, and retain regional precipitation skill.
Pass 64x128 before the 128x256 five-year promotion run.

**Stop condition:** a prettier annual-cycle shape without climate skill is not
enough. If the heat-convergence coupling regresses the climate gate, keep the
legacy land path supported and record the replacement's missing process.

## Phase 4 — surface-water and land-ice foundations

**Dependency:** land ice does not become a climate feature until surface
hydrology has a bounded, physical routing destination.

**Step 1: hydrology redesign.** Extend D8 routing with channel/storage
capacity, flow velocity or residence time, and lateral spill/outflow. Demonstrate
area-weighted water conservation, bounded lake depth, and stable multi-year
basin behavior before coupling it more strongly to climate.

**Step 2: land-ice redesign.** Build on the bounded water path with a defined
mass balance, terrain-slope flow, albedo evolution, calving/freshwater export,
coastline interaction, and a multi-century test harness. Start as an offline
or default-off experiment; do not use it to patch modern-Earth temperature.

**Admission gate:** conservation and stability first, then multi-century
response to controlled temperature forcing, then a calibrated Earth or
paleo-climate target. No short CRU result can promote either feature.

## Phase 5 — long-horizon orbital capability

**Dependency:** stable ice/ocean and a dedicated long-run harness from Phase
4. The orbital forcing itself is already wired.

**Work:** add reproducible multi-millennial scenarios, checkpointing,
drift diagnostics, orbital phase validation, and ice/ocean response metrics.
Validate that a fixed-forcing control remains stable before interpreting a
Milankovitch response.

**Admission gate:** numerical stability, bounded energy/carbon/ice behavior,
and a documented qualitative response to controlled orbital forcing. This is a
research capability, not a prerequisite for the supported modern-Earth
baseline.

## Validation-only queue

These items are useful but must not interrupt a structural phase:

| Item | Required decision evidence |
|---|---|
| Derived ocean seasonal lag | Matched real-terrain A/B showing no climate regression and a measurable phase-error improvement. |
| CFL humidity advection | DAILY/WEEKLY/MONTHLY and multi-grid comparison demonstrating a convergence or stability benefit over the calibrated fixed-divisor path. |
| Inert land-shape controls | No independent sweeps. Revisit only as part of Phase 3's replacement architecture. |
| Default-off condensate, column-water, and overturning companions | Evaluate only as part of their complete closure family, never as independent user-facing tuning controls. |

## Decision record required at every phase boundary

For every candidate, record the exact configuration, baseline report identity,
grid, duration, wall cost, compact and long-gate results, affected regional
pathway metrics, conservation diagnostics, and one of: **promoted**,
**retained experimental**, **redesign required**, or **retired**. Update
`PRODUCT_SCOPE_AND_EXPERIMENTS.md`, `CURRENT_BASELINE.md` when defaults change,
and the relevant test/readme documentation in the same change.
