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

That shared solve is now implemented as
`enable_shared_pressure_coordinate_circulation`. At 64x128 it is the first
candidate to make the wind/omega/energy contract coherent: omega RMS is
0.00198/0.000688 Pa/s, cross-equatorial energy transport is -9.66 PW, peak
transport +62.9/-74.6 PW, and runtime 42.3 seconds. It is nevertheless
rejected as a complete climate candidate because global precipitation is only
0.954 mm/day, KÃ¶ppen group/class is 0.501/0.287, and reference error 0.696.

**Next Phase-2 architecture:** retain the shared circulation; do not raise
its speed or reintroduce raw divergent winds. Build an explicit
circulation-coupled moisture-source/condensate/rainfall pathway that restores
raw production through a physical surface-evaporation, ascent, and fallout
budget. Validate its water/MSE export terms against this shared circulation
before another compact CRU run; geographic targets and allocator repair remain
out of scope.

The pressure-mass source/condensate/fallout branch now closes water to 1.34e-7
relative residual at 64x128, but is rejected as a coupled climate candidate:
30 days produces 54.6 mm/day rain peaks and 1.27 Pa/s lower-interface omega.
Its direct condensation-to-next-step omega handoff lacks a large-scale
adjustment timescale.

**Revised next Phase-2 architecture:** retain the shared circulation and
pressure-mass water budget, but add a prognostic large-scale overturning/heat
reservoir with explicit adjustment and compensating energy export. Do not tune
fallout, RH, or circulation strength to suppress the oscillation.

That reservoir is now implemented with a derived ~54.4-day radiative
free-troposphere adjustment time. The required phase audit corrected an
important unit boundary: pressure-mass cloud water had been interpreted by
radiation as a mixing ratio. The corrected 64x128 one-year spin-up/one-year
evaluation is stable but rejects the candidate (0.530 mm/day global rain,
0.965 polar rain, 0.434 cloud fraction, 0.870 reference error; omega RMS
0.00161/0.00175 Pa/s). Do not restore the unit error or tune cloud/rain
coefficients. The separate pressure-mass suspended-cloud and
precipitating-hydrometeor reservoirs are implemented, nested default-off:
cloud excess autoconverts into an independently transported, falling mass
reservoir, so direct cloud fallout is no longer used in that branch. Its mass
and distinct-footprint regressions pass, but the unchanged 64x128 candidate
initially hit a raw-temperature static-stability singularity on evaluation day
15 (905,002 W m-2 stored heating; 26.9 s/day). Replacing that nonphysical raw-T
denominator with the derived potential-temperature gradient makes the compact
screen stable, but the full 64x128 one-year spin-up/one-year evaluation still
fails: 1.336 mm/day global rain, 4.670 polar rain, 0.539 cloud, 0.955 reference
error, and omega RMS 0.0411/0.0234 Pa/s. Do not tune conversion or fallout
values. The next structural redesign is an explicit large-scale heating export
closure; its purpose is to constrain the response to stored condensation
without an arbitrary heating or omega cap.

The first default-off energy-export attempt,
`enable_pressure_coordinate_mse_transport`, now carries each pressure layer's
MSE on the identical finite-volume faces as its vapour and accounts for lower
evaporation's latent-energy input. Its conservation regression passes, but its
64x128 one-year spin-up/one-year evaluation rejects direct use with the
existing shared winds: 0.481 mm/day global precipitation, 0.207 polar
precipitation, 0.549 cloud, 0.898 reference error, and +28.6/-94.7 PW peak
northward/southward energy transport. Retain the kernel default-off; do not
tune it. The required Phase-2 redesign is now narrower: an MSE-constrained
circulation solve must diagnose transport and overturning together, rather
than applying MSE export to winds diagnosed solely from latent heating.

The first joint-budget implementation,
`enable_mse_constrained_pressure_circulation`, diagnoses the required zonal
MSE export first, then uses the resolved lower/upper MSE contrast to determine
equal-and-opposite pressure-branch mass fluxes, winds, and omega. Its analytic
tests close mass, give no response to uniform forcing, export an equatorial
heating anomaly poleward, and reject a vanishing branch contrast without a
floor. The 64x128 one-year screen is stable and reduces the direct-export
pathology from +28.6/-94.7 to +9.42/-6.00 PW, but rejects climate:
0.688 mm/day global precipitation, 0.208 polar precipitation, 0.567 cloud,
0.663 reference error, 0.486/0.271 KÃ¶ppen group/class accuracy, and
0.000454/0.00410 Pa/s omega RMS. Retain it default-off; do not tune it.
The next structural change is a three-branch MSE/vertical-exchange solve that
derives the middle-layer transport and phase-heating deposition with both
interface mass fluxes.

That three-branch operator is now implemented behind
`enable_three_branch_mse_pressure_circulation`. It chooses the unique
minimum-mass-flux solution subject to mass and MSE transport constraints, then
diagnoses layerwise diabatic deposition after both interface energy exchanges.
Its kernel tests close the three-layer forcing exactly and reject zero MSE
variance. The 64x128 one-year screen is stable, but rejects the candidate:
0.428 mm/day global precipitation, 0.071 polar precipitation, 0.571 cloud,
0.754 reference error, 0.483/0.274 KÃ¶ppen group/class accuracy,
0.00102/0.0108 Pa/s omega RMS, and +0.745/-0.160 PW peak transport. The
minimum-norm branch choice under-transports energy; retain it default-off and
do not add a branch-weight scalar. The next structural constraint must be a
momentum/thermal-wind relation for the vertical branch shape, solved together
with the existing exact MSE and mass budgets.

The first algebraic momentum constraint is implemented behind
`enable_momentum_constrained_three_branch_mse_circulation`: exact column mass
closure removes the planetary angular-momentum term, leaving zero
mass-weighted transport of the resolved vertically sheared zonal momentum as
the third constraint. Its unit tests close mass/MSE/momentum and reject
barotropic winds. The 64x128 one-year candidate is stable but rejected:
0.497 mm/day global precipitation, 0.075 polar precipitation, 0.578 cloud,
0.883 reference error, 0.465/0.260 KÃ¶ppen group/class accuracy,
0.00179/0.01499 Pa/s omega RMS, and a 69.4 PW northern energy-transport peak.
This identifies the exact missing state: current layer winds are not a closed
pressure-coordinate momentum/thermal-wind solution. Retain the diagnostic
default-off; do not cap or tune it. The next replacement is a prognostic
three-level zonal-momentum solve with pressure-gradient, Coriolis, and vertical
momentum-exchange terms, whose diagnosed shear may then enter the MSE closure.

The pure three-level pressure-momentum update is now implemented behind
`enable_prognostic_pressure_coordinate_momentum`: hydrostatic pressure
gradients, analytic Coriolis rotation, and conservative prior-interface
momentum exchange evolve layer winds without a damping scalar. Its tests pass,
but its current monthly split coupling is rejected before climate scoring. At
64x128 its first two 30.44-day cycles take 9.03/8.43 s; the second produces
0.00319/0.00987 Pa/s interface omega, and the third cannot complete within the
two-minute runner limit because the downstream closed-column substep policy is
violated. Do not cap or relax it. The next redesign must jointly discretize
pressure-level momentum, MSE, and vertical transport under one CFL policy;
sequential monthly operators are not an admissible closure.

The replacement's pure state-transition contract is now implemented as
`evolve_joint_mse_momentum_pressure_column`. It adaptively substeps the
three-branch MSE/momentum diagnosis, conservative vertical MSE exchange, and
Strang-split pressure-level momentum under one interface-CFL policy. Its tests
close water/MSE, preserve a uniform zero state, and bound every coupled
substep's vertical Courant number. It is deliberately not wired yet: the
current atmosphere adapter still applies horizontal MSE transport and phase
conversion outside the vertical operator, so partial wiring would double-apply
vertical exchange. Next build the one-call adapter that owns horizontal MSE,
phase deposition, and this joint kernel together; then run the standard
64x128 screen. Do not add a monthly timestep, CFL, damping, or omega scalar.

That adapter is now wired under the deepest momentum/MSE gate and disables the
legacy split transition and its daily wrapper there. Its focused runtime and
conservation regressions pass. The branch solve now eliminates one velocity
algebraically so mass closure is structural. The required 64x128 screen still
fails in its second 6.09-day monthly substep: near-degenerate MSE and momentum
constraints (condition number 2.64e9) demand a 3.30e9 m/s branch speed. Its
absolute mass residual is roundoff relative to that unphysical divergence, not
the cause. It has no climate score and remains rejected. Keep it default-off;
replace the nearly dependent third constraint with a physically independent
one, without introducing a cap, damping, fallback branch, or prescribed joint
inner timestep.

The pure replacement is now available as a column-water-constrained
three-branch diagnostic: it closes pressure mass structurally, then solves the
MSE and column-vapour transport budgets together. It is intentionally not
wired into the monthly adapter. A partial attempt using surface supply minus
prior precipitation is physically inconsistent whenever water forcing is
nonzero but the current MSE export is zero. The next bounded build is one
simultaneous transition for evaporation, condensation/cloud storage, fallout,
and the water-constrained circulation; do not substitute a lagged rainfall
field, branch weight, cap, damping, or inner timestep.

Its finite-volume MSE layer deposition now solves the MSE/water constraints
directly at each latitude face, avoiding an invalid centre-transport share
normalisation when water transport is zero. A pure undamped fixed-point probe
that fed its newly diagnosed vapour condensation back as the water sink is
also rejected: it produced a 6.85e-4 kg m-2 s-1 sink and a 94.6 m s-1 middle
branch in the analytic 80 W m-2 case. The still-required transition must solve
vapour, cloud, hydrometeors/fallout, and the contemporaneous latent-heating
reservoir as one coupled prognostic system; do not conceal this failure with
relaxation or an iteration-derived circulation cap.

The existing pressure reservoir code has been factored into a single pure
cloud/hydrometeor transition plus an explicitly unit-safe external water term:
only surface supply and fallout form the circulation forcing, while phase
change stays internal to total atmospheric water. The next build must consume
those contracts in one nonlinear transition that also updates the heat
reservoir; merely passing either term through the current adapter remains
partial wiring.

The pure candidate residual is now available and projects both water and heat
onto exactly the zonal, global-mean-free component the closed circulation can
constrain. Its analytic regression confirms uniform boundary terms are not
mistaken for circulation constraints. A scaled 8x16 nonlinear probe cannot
reach a root (maximum scaled residual 0.89; standard root methods also fail
to establish one), so no nonlinear solver is wired or represented as a
completed closure. The remaining physics task is to
replace the one-way omega-based phase activation with a prognostic simultaneous
phase/reservoir/heating relation that admits the two exact circulation budgets.

That relation is now available as a pure simultaneous adapter. Its water
constraint includes the actual total-atmospheric-water storage tendency, which
makes the exact undamped candidate map converge in the analytic case; it also
owns cloud and hydrometeor horizontal transport before the reservoir update.
It is not yet wired into the monthly gate: run and assess the 64x128 screen
first, retaining failure-on-nonconvergence and avoiding a relaxed iterate,
cap, damping coefficient, or prescribed physical inner timestep.

## Simultaneous pressure-column admission goals and decision tree (2026-08-13)

**Goal 1 — Atomic runtime ownership.** Replace the nested joint-runtime
adapter and the pressure moisture phase/reservoir block together, behind one
new default-off gate. The simultaneous call must own horizontal MSE, adaptive
vertical exchange, pressure momentum, phase conversion, cloud/hydrometeor
transport, autoconversion, fallout, external-water forcing, and the heating
reservoir. The host must consume its returned winds, three thermodynamic
layers, reservoirs, fallout/precipitation, and heating state exactly once.

* If a legacy operation would also run under that gate, restructure the block
  until there is one owner; do not insert a compensating subtraction.
* If the adapter cannot receive an input already owned by the host (for
  example the surface source or prior heating state), expose that input in the
  adapter contract with physical units and add an identity/closure regression.
* If the atomic replacement changes a gate-off regression, reject the edit and
  restore gate-off behavior before proceeding.

**Goal 2 — Runtime invariants and compact admission.** Add runtime tests for
one-call ownership, converged coupling residuals, no double reservoir
transport, water/MSE closure, and persisted heating/winds/reservoirs. Then run
the 64x128 monthly cadence used by the existing joint adapter screen.

* If coupling fails to converge, reject the candidate and diagnose the exact
  residual/state causing it; do not return a partial iterate or add relaxation.
* If fields become non-finite, violate shared interface CFL, or fail water/MSE
  closure, reject the candidate and repair only an identified conservation or
  ownership defect. Do not add caps, damping, fallback branches, or a physical
  inner timestep.
* If the compact path is finite and closed but climate metrics reject it,
  retain the infrastructure default-off, record the measured failure, and move
  to the next missing physical constraint rather than tune the rejected one.
* If compact validation is finite, closed, and improves the defined climate
  admission measures, run the standard longer screen before considering any
  promotion. It remains default-off until that screen passes.

**Goal 3 — Post-admission route.**

* If the simultaneous closure is rejected for physical skill after valid
  integration, next derive the missing physical relation from the diagnosed
  failure (for example a phase-equilibrium or radiation coupling), with a pure
  kernel and regression first.
* If it is admitted through the long screen, document the exact gate family,
  add persistence/serialization coverage, and only then consider whether it
  can replace an older experimental path. Default behavior is not changed by
  admission alone.

**Admission result — rejected before compact climate scoring.** The atomic
runtime wiring was exercised at 12x24 for one day and rejected when an exact
nonlinear candidate produced a 2.91e17 MSE/water constraint condition number,
9.73e9 m s-1 branch, and 1.71e4 s-1 divergence. The 64x128 screen is not run:
the smaller runtime failure already violates Goal 2's finite/closed condition.
The gate wiring is removed; retain the pure simultaneous adapter and its
contracts. The next goal is to identify a third branch constraint that remains
independent when MSE and transient water transports approach degeneracy, rather
than tuning or bounding the rejected system.

**Next derived constraint — transient MSE storage.** The water solve already
uses ``source - fallout - d(total water)/dt``. The MSE circulation still uses
an instantaneous heating balance, so its next pure residual must instead use
``diagnosed diabatic heating - d(three-level MSE)/dt`` as the candidate
meridional MSE-transport forcing. `column_mse_storage_tendency_w_m2` now
defines and tests that storage term. Decision tree: if the revised exact map
converges while preserving finite rank, rebuild the simultaneous pure adapter
with separate heating-state and transport-forcing unknowns; if rank remains
deficient, reject that closure with the incompatibility diagnostic and move to
a prognostic pressure-coordinate transport formulation rather than adding a
branch selector.

**Transient-MSE result — rejected.** Feeding the storage correction into the
exact simultaneous map drives the analytic 8x16 candidate into an active
rank-deficient MSE/water constraint before convergence, with temperatures and
transport forcing diverging to nonphysical values. The residual integration is
therefore removed; the pure storage diagnostic remains. Both the original and
transient-MSE simultaneous formulations are rejected before 64x128 climate
scoring. The next architecture is a genuinely prognostic pressure-coordinate
transport evolution, in which MSE and water tendencies are evolved rather than
used as simultaneous diagnostic branch constraints.

### Prognostic transport foundation (2026-08-13)

`evolve_prognostic_pressure_coordinate_transport` is the first pure primitive
for that replacement. It advances the three pressure-layer vapour inventories
by conservative finite-volume horizontal fluxes, transports each layer's MSE
on the identical faces (including the lower-boundary latent-energy input), and
then advances the three wind levels with the existing hydrostatic
pressure-gradient/Coriolis momentum operator. The water and MSE fields are
therefore tendencies of a state, never residuals inverted into branch speeds.
Its donor-cell transport subdivision is derived from the resolved horizontal
CFL and reports the actual number of substeps and maximum per-substep Courant
number; it introduces no cap, damping, branch selector, or prescribed physical
inner timestep. Focused regressions verify water/MSE conservation, correct
surface-source/latent-energy accounting, non-negative water, and CFL
subdivision.

It is deliberately not runtime-wired yet: it has zero interface mass flux and
does not claim to provide phase conversion, cloud/hydrometeor evolution, or a
prognostic vertical-momentum relation. The next bounded kernel must add a
prognostic pressure-layer mass/interface-flux state that closes horizontal and
vertical continuity without deriving omega from an instantaneous MSE or water
constraint. Only after that kernel transports the existing condensate
reservoirs and phase heating in the same transition may a new default-off
atmosphere gate be considered.

`evolve_prognostic_pressure_layer_mass` now supplies that pressure-coordinate
state boundary. It transports each layer's pressure mass (`dp / g`) on the
resolved horizontal faces and advances both signed interface mass-flux states
from externally supplied acceleration tendencies using a midpoint update.
Positive interface flux is upward; its conservative exchange preserves total
column mass exactly. The vertical cadence is recomputed from the *current*
combined donor-layer Courant fraction after each exchange, which is required
when both interfaces drain the middle layer. Its focused tests cover a resting
identity, conservative two-interface transfer, interface-flux-state evolution,
and the adaptive combined-donor CFL condition. It deliberately contains no
vertical-force law yet: the next kernel must derive the interface-flux
tendencies from a prognostic pressure-coordinate vertical-momentum relation,
then carry water, MSE, phase conversion, and condensate reservoirs with its
changing pressure-layer masses in the same state transition.

`evolve_variable_mass_pressure_coordinate_transport` now carries the first
three of those inventories together: horizontal pressure mass, vapour mass,
and MSE use the same resolved faces, and each signed interface flux transfers
the donor parcel's mass, vapour, and MSE together. Lower surface supply imports
the matching latent energy. The primitive therefore closes water and MSE while
pressure-layer thickness changes, rather than pairing a variable mass solve
with a fixed-mass tracer update. It retains interface-flux tendencies as an
explicit state boundary and does not infer them from water/MSE residuals.
Focused regressions verify conservative two-interface transport and exact
surface-water/latent-energy accounting.

The remaining circulation task is to replace that external interface-flux
tendency boundary with a hydrostatic continuity closure based on the evolving
pressure-layer mass and resolved horizontal momentum state. Do not add a
nonhydrostatic vertical acceleration coefficient merely to manufacture an
omega prognostic: in the intended hydrostatic pressure-coordinate system,
interface flux must be constrained by pressure continuity. Once that closure
is independently derived, add phase conversion and condensate/hydrometeor
reservoir transport to this exact variable-mass transition before a runtime
gate is reconsidered.

`diagnose_hydrostatic_sigma_continuity` now closes the missing hydrostatic
relation. The three layer masses are declared sigma fractions of the evolving
total column pressure. Their finite-volume horizontal mass tendencies uniquely
determine the two interface fluxes; the resulting middle-layer continuity
residual is tested directly. `evolve_hydrostatic_sigma_pressure_coordinate_transport`
composes that diagnosis with the variable-mass water/MSE transition, so one
pure call keeps the pressure partition, mass, water, and MSE closure together.
It accepts no externally diagnosed omega or interface acceleration. The first
tests verify the analytic lower-only convergence case, the exact middle-layer
continuity relation, maintained 0.40/0.35/0.25 pressure partition, and water/
MSE closure.

The next bounded build is phase conversion and suspended-cloud/precipitating-
hydrometeor transport inside that same hydrostatic sigma transition. It must
move condensate on the corresponding layer winds and use the contemporaneous
pressure masses for conversion, fallout, water, and latent-energy accounting.
Only then add the pressure-level momentum update under the same transport
cadence; do not revive a separate omega state, a heating-derived interface
flux, or a prescribed vertical acceleration.

`evolve_hydrostatic_sigma_phase_reservoir_transport` now makes that transition
atomic for all currently represented water phases. It transports three
layer-resolved cloud and hydrometeor reservoirs on their corresponding layer
winds, converts activated/supersaturated vapour with a pressure-adjusted
saturation calculation, retains MSE as sensible heat during phase conversion,
then autoconverts and sediments each layer independently. Surface vapour,
vapour reservoirs, cloud, hydrometeors, and returned fallout therefore have
one owner. A regression caught and corrected an early ownership defect in
which condensate was added to a reservoir without returning the matching
vapour-depleted state; the final test closes total water including fallout.

The remaining kernel work before runtime admission is to carry the
pressure-level horizontal momentum update in this exact transition, using the
same continuity-derived interface flux and variable pressure masses. Then add
the new state to persistence and a deeply nested default-off atmosphere gate;
do not mix it with the older diagnostic pressure branch.

`evolve_variable_mass_pressure_momentum` now supplies the remaining momentum
leg: hydrostatic layer-center pressure gradients and analytic Coriolis rotation
advance each pressure-level wind, and the same signed sigma-continuity fluxes
transfer horizontal momentum with donor air under the current combined-donor
CFL bound. The phase/reservoir transition owns this momentum update and returns
its three winds with the thermodynamic state. A focused regression confirms
interface exchange conserves both horizontal momentum components when external
pressure-gradient/Coriolis forcing is disabled.

The pure column contract is now complete enough for bounded runtime admission:
add the pressure-layer masses and six layer-reservoir fields to state
persistence, place one deeply nested default-off gate around this call, and
prove gate-off identity and one-call ownership before the first compact screen.

The pressure-layer masses and six layer-resolved reservoir arrays are now
explicit `PlanetState` fields and round-trip through the safe NPZ persistence
format. The new `enable_hydrostatic_sigma_pressure_coordinate_transport` flag
is default-off, and ordinary simulation steps preserve this inactive state
verbatim rather than dropping it. Its runtime gate now performs one atomic
transition: it passes the persisted layer state to the hydrostatic-sigma
transport/phase/momentum closure, adopts its temperatures and winds, persists
its pressure and phase reservoirs, and bypasses the old lagged heating
reservoir. A full enabled one-day 8x16 transition is finite, while a direct
gate-off regression proves normal supported stepping keeps every experimental
array bit-identical. The variable-mass transport and momentum primitives also
reject a requested fixed-flux transition that would physically exhaust a
layer, rather than silently taking an unbounded sequence of smaller CFL
substeps. Its first 64x128 MONTHLY compact spin-up is rejected before scoring:
the legacy-provided horizontal wind transports an entire sigma-layer mass out
of at least one cell before the closure can form its vertical donor exchange.
The next Phase-2 constraint is therefore a simultaneous mass/wind evolution
that establishes an admissible horizontal pressure carrier before hydrostatic
vertical exchange. No Earth baseline behavior has been changed.

`evolve_hydrostatic_sigma_mass_momentum` now supplies that carrier as a pure
kernel. At each adaptive substep it advects every layer's pressure mass, both
momentum components, vapour, MSE, cloud water, and hydrometeors through
identical donor faces; it derives the two continuity exchanges from that same
horizontal update, performs those parcels simultaneously, restores the sigma
partition exactly, and then applies the hydrostatic
pressure-gradient/Coriolis force. Its coupled horizontal and vertical donor
fractions set the timestep; the horizontal donor fraction is bounded by a
strict 0.9 positivity margin (so roundoff cannot empty a donor layer) and the
simultaneous vertical donor fraction by 0.25. It has no wind cap, damping,
mass floor, fallback, or prescribed
cadence. A 30-day divergent-wind regression keeps every layer finite,
preserves mass/water/MSE, and verifies both Courant bounds. A speed
that exceeds the hydrostatic gravity-wave scale is rejected rather than
clipped. A derived pressure-gradient momentum-CFL bound now limits each
resolved acceleration increment to half of the *remaining* gravity-wave-speed
headroom; it is a state-derived numerical stability condition, not a wind cap
or prescribed physical cadence. The atomic hydrostatic-sigma runtime gate now
uses this carrier, rather than the older split transport/phase/momentum path.
Its three-layer horizontal MSE-flux convergence is emitted as the
area-balanced persisted diagnostic
`pressure_coordinate_heat_convergence_w_m2`; this is the atmospheric
heat-convergence term Phase 3 requires.

**Runtime admission result (2026-08-14): rejected, redesign required.** The
first real-terrain 64x128 MONTHLY screen without the momentum bound crossed
the gravity-wave limit at 283.276 m/s against a 280.143 m/s limit. The
headroom-bound version remains finite, but the standard one-year spin-up plus
one-year evaluation protocol had consumed more than 37 CPU-minutes without
finishing, because the MONTHLY policy invokes the atomic carrier five times
per month and each call resolves the physical pressure-gradient and advective
CFL constraints. Do not remove the bound, reintroduce damping, or use a cap
to obtain a score. The next Phase-2 architecture is a conservative
semi-implicit or otherwise fast gravity-wave/mass-continuity solve that
remains finite under the same pressure-gradient contract; only then rerun the
complete climate gate.

A simulation-step gate-off regression now also seeds every persisted
hydrostatic-sigma field and proves that the normal supported step retains each
array exactly. This protects the forthcoming atomic bypass from both ordinary
state loss and a false no-op claim.

**Candidate 1 — compiled explicit carrier (2026-08-14): rejected.** The
existing finite-volume carrier was moved wholesale into its compiled path,
including the same pressure-gradient/gravity-wave-headroom timestep bound as
the reference transition. Its 53 focused pressure-circulation regressions
passed, but a realistic 64x128 six-day enabled transition exceeded 60 seconds
(the baseline vectorised transition takes about 12.6 seconds after warm-up).
The scalar compiled adaptive loop cannot amortise the many headroom-limited
substeps. The edit was reverted. This rules out a direct implementation-only
port: the next candidate must reduce the stiffness structurally while retaining
conservative pressure/mass continuity and the hydrostatic admissibility test.

**Candidate 2 — semi-implicit barotropic pressure response (2026-08-14):
rejected.** A pure implicit gravity-wave pressure/continuity kernel passed its
resting-column and coarse-step finite regressions and was wired behind a new
default-off sub-gate. The first 64x128 six-day enabled transition completed in
0.12 seconds, but its resolved carrier reached 289.5 m/s against the same
280.1 m/s hydrostatic gravity-wave limit. The method removes the runtime
stiffness but not the underlying unbalanced acceleration, so it is not a
valid Phase-2 carrier. Revert the sub-gate and retain only the documented
result; Candidate 3 must supply an actual balanced momentum/pressure relation,
not merely an unconditionally stable pressure solve.

**Candidate 3 — zonal-mean anelastic barotropic projection (2026-08-14):
rejected.** The pure projection exactly removed each latitude's zonal-mean
column meridional mass export while preserving vertical shear, and its focused
regression suite passed. The first 64x128 six-day transition stayed admissible
(52.5 m/s peak upper wind), but the required 30-day enabled sequence exceeded
60 seconds. It does not relieve the carrier's adaptive transport cost, so the
runtime integration was reverted. Candidate 4 must change the transport
algorithm itself, not merely constrain one component of its input wind.

**Candidate 4 — conservative large-Courant moving-frame remap (2026-08-14):
rejected.** An arbitrary-Courant periodic remap preserved positivity and mass,
and the complete carrier retained all 55 focused regressions when the same
remap was applied to mass, momentum, water, MSE, cloud, and hydrometeors. In
the 64x128 30-day runtime gate, day 6 took 19.1 seconds and day 12 reached 34.6
seconds; day 18 did not finish before the 60-second limit. Removing uniform
zonal translation does not remove the residual adaptive stiffness. The runtime
wiring was reverted. Candidate 5 may optimize the full residual carrier only
if it preserves the identical force and CFL contracts.

**Candidate 5 — compiled carrier with in-substep anelastic projection
(2026-08-14): rejected.** The compiled finite-volume path projected the
zonal-mean column mass-flux mode before every donor transition and transported
all seven conserved components together. All 55 focused regressions passed,
but the 64x128 runtime gate took 48.3 seconds for its first six-day transition
and did not complete day 12 before 60 seconds. Compiling the residual carrier
therefore does not make the adaptive architecture admissible. The runtime edit
was reverted. Five bounded redesign attempts have now rejected compilation,
implicit pressure response, anelastic projection, large-Courant moving-frame
remapping, and their compiled combination. Phase 2 remains redesign-required.

**Equivalent-forcing investigation (2026-08-14): Phase 3 dependency met at
compact admission scale.** Three independent alternatives were evaluated
before relaxing the dependency on the rejected native pressure-column runtime:

1. An offline seasonal-storage inversion produced a bounded field (land
   monthly standard deviation 2.23 W m-2, 5th/95th percentiles -3.68/+3.66
   W m-2), but its annual mean is zero by construction and its point-month
   correlation with resolved supported transport was only 0.014. A fuller
   CRU-target surface-energy residual was much worse: +409 W m-2 mean, 660
   W m-2 standard deviation, and values up to 4.04 kW m-2. It conflates model
   state error with missing transport and is rejected as a forcing; retain it
   only as an offline falsification diagnostic.
2. ExoPlaSim 3.4.2 exposes the required monthly energy fields: net top
   radiation (`ntr`, code 261) and surface downward heat flux (`hfns`, code
   263). With its downward-positive convention, atmospheric horizontal heat
   convergence follows the auditable budget `storage - ntr + hfns`. The
   retained T21 reference contains only `tas` and `pr`, however, and the local
   Windows probe cannot build ExoPlaSim; a Linux rerun is required before this
   route can be numerically validated. It remains an independent future
   provider/falsification route, not a current runtime dependency.
3. The supported single-layer atmosphere can provide an equivalent forcing
   without the rejected pressure runtime. The exact temperature increment
   immediately after its active horizontal advection and diffusion operators
   is converted with `(p_s/g) cp / dt`. The prior public component diagnostic
   was unsuitable because it subtracted from the final end-of-step air
   temperature and therefore included downstream surface physics. The new
   diagnostic is captured at the operator boundary and its area-weighted
   global mean is removed, enforcing the closed-sphere requirement that
   horizontal heat convergence integrate to zero.

The third route is wired behind
`enable_force_restore_atmospheric_heat_convergence`, which is default-off and
requires `enable_force_restore_land`. On the bundled 64x128 Earth DEM with one
year spin-up and one year evaluation against CRU TS v4.10, the exact current
force-restore control and the globally closed forcing candidate measured:

| Metric | Force-restore control | Closed resolved forcing |
|---|---:|---:|
| land monthly temperature RMSE (C) | 7.940 | 7.857 |
| land monthly temperature correlation | 0.868 | 0.876 |
| precipitation monthly RMSE (mm/day) | 2.626 | 2.614 |
| precipitation annual log correlation | 0.518 | 0.522 |
| Koppen group accuracy | 0.583 | 0.608 |
| Koppen class accuracy | 0.229 | 0.267 |
| coldest-month threshold accuracy | 0.743 | 0.770 |
| warmest-month threshold accuracy | 0.369 | 0.412 |

The candidate improves every stated compact admission metric, so it is
**retained experimental** and Phase 3 may start. Its +2.70 C monthly bias and
7.86 C RMSE remain worse than the supported legacy-land baseline (6.28 C
tracked compact RMSE), so this is not default promotion. The next Phase 3 work
is regional/seasonal error diagnosis and plateau-dependence measurement before
the 128x256 five-year gate.

## Phase 3 — land temperature replacement with diagnosed heat convergence

**Dependency:** met at compact scale by the globally closed resolved-transport
forcing above. Phase 2 remains independently redesign-required; it is no
longer on Phase 3's critical path. The force-restore branch already proves that
local radiative/turbulent/conductive physics alone is not enough.

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

### Phase 3 implementation pass (2026-08-15): compact matrix complete, promotion rejected

The Phase 3 work is now reproducible through
`scripts/run_phase3_land_validation.py`. It generates fresh supported and
force-restore controls, runs five bounded physical candidates, records the
resolved forcing on every evaluation step, adds per-month named-region errors,
and evaluates both the mechanism-relative land gate and a strict promotion
gate against the supported baseline. The strict gate includes CRU temperature,
precipitation, Koppen group/class skill, warmest/coldest-month thresholds,
annual-cycle shape, a present forcing diagnostic, and applied-grid global
energy closure.

The 64x128, one-year spin-up plus one-year evaluation matrix produced:

| Candidate | CRU T RMSE (C) | T correlation | Koppen group/class | cycle error |
|---|---:|---:|---:|---:|
| supported legacy land | **6.276** | **0.930** | **0.674 / 0.389** | 5.269 |
| force-restore, no convergence | 7.940 | 0.868 | 0.583 / 0.229 | 1.320 |
| resolved convergence, 30 d | 7.857 | 0.876 | 0.608 / 0.267 | 1.852 |
| resolved convergence, 15 d | 7.874 | 0.875 | 0.612 / 0.265 | 1.841 |
| resolved convergence, 60 d | 7.825 | 0.877 | 0.605 / 0.264 | 1.795 |
| 6 MJ m-2 K-1 deep store | 7.828 | 0.876 | 0.606 / 0.265 | 1.818 |
| 1000 s m-1 dry resistance | **7.793** | 0.875 | 0.603 / 0.262 | **1.757** |

All five convergence candidates improve temperature RMSE and both Koppen
accuracies relative to the uncoupled force-restore control, confirming that
the diagnosed forcing adds useful information. None passes against the actual
supported baseline: even the best RMSE remains 1.52 C worse, group accuracy is
7.1 percentage points lower, class accuracy is 12.7 points lower, and the
warmest/coldest threshold accuracies remain lower. Precipitation log-RMSE is
better than supported for every candidate, but every candidate slightly
regresses the named East China and Southeast US precipitation-target errors.
Plateau/cycle shape is much less clamp-like, but those isolated wins do not
satisfy the stop condition.

The forcing itself is finite and closed. Across 60 evaluation samples at
64x128, its RMS is about 19 W m-2, 5th/95th percentiles are about -36.5/+31.5
W m-2, maximum magnitude is 117 W m-2, and the maximum absolute applied-grid
global mean is below 2.5e-7 W m-2. A separate 32x64 run remains closed below
3.1e-7 W m-2 with 16.3 W m-2 RMS and 94 W m-2 maximum magnitude. A one-day
versus two-half-day test preserves the spatial pattern (correlation 0.977) but
has 21% relative RMS disagreement, consistent with the supported temperature
operator's already-documented step-partition sensitivity. This is acceptable
for retaining the diagnostic experimentally, but not for promotion.

Regional/monthly attribution identifies the missing process. In the best-RMSE
candidate, seasonal-range error is +19.1 C in Central Europe, +12.5 C in South
Japan, +12.2 C in Atacama, and +9.3 C in the Southeast US. Central Europe's
monthly RMSE rises from the supported 1.65 C to 7.59 C; South Japan rises from
9.07 C to 13.14 C. Changing deep storage, restore time, or dry resistance does
not repair this coherent continental/coastal over-amplitude. The replacement
still lacks a conservative land-air boundary exchange and/or resolved maritime
thermal moderation; it is not missing another local scalar value.

**Disposition:** retain the globally closed forcing and evaluation
infrastructure as default-off experimental Phase 3 work. Keep the legacy land
path supported. Do not run the 128x256 five-year promotion checkpoint because
zero of five candidates passed the required compact gate. The next structural
candidate must address conservative land-air/maritime energy exchange, not
repeat this bounded local parameter matrix or restore a latitude cap.

### Conservative land-air exchange screen (2026-08-15): physically closed, rejected

The next structural candidate replaced the force-restore branch's empirical
two-day land-air relaxation with an explicit boundary transfer. The positive-
upward Penman--Monteith sensible flux already subtracted from the surface was
added to the represented atmospheric column using `(p_s/g) cp`; the exchange
residual is zero to floating-point precision. Ocean relaxation and every
supported default remain unchanged, and the candidate is independently gated
by `enable_force_restore_conservative_land_air_exchange`.

The required 32x64 directional screen rejected it before 64x128:

| Metric | resolved forcing control | conservative exchange |
|---|---:|---:|
| CRU temperature RMSE (C) | **7.179** | 7.899 |
| temperature correlation | **0.872** | 0.843 |
| precipitation log-RMSE | 1.301 | **1.299** |
| Koppen group accuracy | **0.547** | 0.543 |
| Koppen class accuracy | **0.238** | 0.231 |
| coldest-month accuracy | **0.695** | 0.615 |
| warmest-month accuracy | 0.481 | **0.503** |
| land-cycle error | **1.482** | 2.594 |

The mechanism does provide the expected regional damping: Central Europe's
seasonal-range error falls from +9.30 C to +5.32 C, the Southeast US from
+11.33 C to +7.20 C, and Atacama from +13.64 C to +12.74 C. But it warms the
air climate, worsens overall spatial skill and cold thresholds, and moves East
China from a near-correct +0.46 C range error to -4.73 C. Regional
precipitation-target errors also regress in the Canadian Prairies, Central
Europe, East China, and the US Midwest.

**Decision:** retain the pure exchange kernel and gate as diagnostic
infrastructure, but reject this full-column coupling. Do not run 64x128 or the
five-year checkpoint. The result exposes a representation conflict: the single
supported `T_air` field is used as near-surface/2 m air in turbulent fluxes but
as a full atmospheric column in transport energy. A promotable conservative
boundary exchange now requires a distinct boundary-layer heat reservoir (with
a defined mass/depth and conservative exchange with the free atmosphere), not
a fractional-strength retune of this failed coupling.

### Boundary-layer reservoir candidate screen (2026-08-16): rejected

The distinct boundary-layer reservoir required by the previous decision was
built (horizontal transport, interface reservoir, capacity-aware air-sea and
free-air exchange, near-surface cloud temperature, split-invariant cloud
memory) and screened with legacy radiation -- deliberately without the
coupled-grey/closed-three-level family, whose diabatic-omega instability is
an independent, separately tracked blocker
(`docs/VERTICAL_THERMODYNAMIC_CLOSURE.md`). The formal 32x64 one-year
spin-up/one-year evaluation CRU screen
(`scripts/run_phase3_land_air_screen.py`, candidate vs the same resolved
forcing control as the table above) rejects it on 8 of 11 gates:

| Metric | resolved forcing control | boundary-layer reservoir |
|---|---:|---:|
| CRU temperature RMSE (C) | **7.179** | 8.283 |
| precipitation log-RMSE | **1.301** | 1.344 |
| precipitation log-correlation | **0.478** | 0.454 |
| Koppen group accuracy | **0.547** | 0.496 |
| Koppen class accuracy | **0.238** | 0.221 |
| coldest-month accuracy | 0.695 | **0.818** |
| warmest-month accuracy | **0.481** | 0.189 |
| land-cycle error | **1.482** | 2.667 |

The heat-convergence forcing remains diagnosed and globally area-closed
(<= 1e-3 W m-2). The failure pattern is a systematic warm bias: the
corrected ablation matrix shows every boundary-layer variant 6-7.5 K hotter
globally than the control with worse Koppen and cycle error, and the formal
screen confirms the bias is worst in the warm season (warmest-month accuracy
-29 pp) while cold-season thresholds improve (+12 pp). The reservoir as
constructed insulates the lower atmosphere instead of exchanging
conservatively with it.

**Decision:** reject the boundary-layer reservoir candidate at 32x64; do not
run 64x128. The force-restore-plus-convergence direction has now been
rejected at every documented structural level: uncoupled, convergent
full-column, conservative full-column, and distinct boundary-layer reservoir.
The remaining documented atmospheric dependency is the diabatic-omega
operator redesign shared with Phase 2
(`docs/VERTICAL_THERMODYNAMIC_CLOSURE.md`, 2026-08-16). Before any further
Phase 3 candidate construction, this phase needs an explicit re-scoping
decision: either invest in that shared operator redesign, or redefine the
phase's acceptance target around what the supported single-layer atmosphere
can conservatively provide.

### Active goal (2026-08-16): diabatic-omega stability reformulation (shared Phase 2/3 dependency)

**Re-scoping decision:** invest in the shared operator redesign. Phase 3
candidate construction is paused until the operator question is settled.

**Goal:** replace the diabatic-omega operator's dry static-stability
denominator with a reformulated stability measure that remains bounded under
the grey budget's own equilibrium tendency -- with no stability floor, omega
cap, or damping term -- so that the coupled conservative atmospheric branch
(closed three-level column + coupled grey budget) integrates stably at
compact resolution. This is the one remaining documented atmospheric
dependency jointly blocking Phase 2's moisture closure and any conservative
Phase 3 branch: the grey budget's radiative-equilibrium tendency drives dry
static stability toward zero, the current operator is singular exactly there,
and 2026-08-16 screening proved no initialization can hold the state away
from that singularity.

**Admission criteria:**

1. Read-only boundedness diagnostic first (operating rule 2): the
   reformulated stability, measured along the collapsing coupled trajectory,
   stays positive and bounded at the rows/times where the heating acts.
2. Pure kernel with regressions: identical to the current operator in the
   dry-stable limit; bounded omega as dry stability approaches zero under
   realistic humidity gradients; zero mass-weighted column divergence; no
   new tunable strength or damping scalar.
3. The coupled-grey handoff trace
   (`scripts/diagnose_coupled_grey_handoff.py`) shows no mid/upper
   temperature collapse and bounded Courant numbers over 14 days.
4. The 32x64 column-energy audit's grey admission block passes, or any
   residual failure is attributable to the independently tracked OLR
   optical-depth calibration gap.
5. Routine suite green; the decision is recorded either way in
   `docs/VERTICAL_THERMODYNAMIC_CLOSURE.md`.

If criterion 1 fails -- i.e. no humidity-informed stability measure stays
bounded where the forcing acts either -- the fallback re-scoping is to close
Phase 3 around the resolved-forcing control as its answer, and this goal is
recorded as rejected with its measurements.

**Criterion-1 outcome (2026-08-16):** humidity-informed stability
denominators (`theta_e`, moist-static-energy forms) are eliminated with
measurements -- the singular rows are moisture-free subpolar rows where both
candidates are numerically identical to the dry operator and go equally
negative (`docs/VERTICAL_THERMODYNAMIC_CLOSURE.md`). The follow-on
water-budget omega measurement (same day, same doc) eliminates the
moisture-contrast inversion as well: the tropical rows diverge there
(Courant 51-152), the coupled collapse destroys the moisture contrast just
as it destroys the temperature contrast, and its implied heat transport
over-closes the heating by a median factor of ~5. Every diagnostic
`forcing / state-contrast` inversion is therefore singular by construction
under the coupled evolution, and the diagnostic-omega reformulation goal is
**rejected with measurements**. The two remaining formulations are separate,
larger programs -- prognostic pressure-coordinate momentum (previously
rejected for runtime cost) and the simultaneous phase/reservoir closure
(previously rejected for constraint degeneracy) -- and neither is an omega
reformulation. Phase 3 accordingly closes with no land-temperature
replacement: the supported land scheme remains the supported product, the
force-restore/boundary-layer branch is retained default-off with its full
rejection ledger, and the diagnosed heat-convergence instruments and dataset
are the phase's deliverable. Phase 2's blocker is restated precisely by this
goal's evidence: any future circulation operator must carry prognostic
transport or a simultaneous phase/reservoir transition.

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
