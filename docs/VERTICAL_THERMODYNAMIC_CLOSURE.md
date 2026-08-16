# Vertical circulation and thermodynamic-closure contract

Status: **experimental atomic runtime gate implemented; compact admission
rejected, redesign required** (2026-08-13).

This document is the contract for Phase 2 of `REMAINING_WORK_PLAN.md`. It
separates the physical closure that must exist from the older three-level
experiment, which is useful research scaffolding but is not itself a closed
atmospheric column.

## Why a new kernel is needed

The current default climate has a resolved lower air/surface temperature and
an always-on upper wind, but no corresponding upper temperature, humidity, or
radiative-energy reservoir. The opt-in three-level path added those state
arrays and diagnosed pressure-coordinate interface motion, yet it still
partitions one humidity scalar as if the layers had equal mass and restores
mid/upper temperatures toward a prescribed lapse profile. That relaxation is
an unlabelled heat source/sink, and its humidity transfer is not a
finite-volume moisture budget. It must not be tuned into a replacement for the
supported climate path.

## Canonical column state

The closed operator uses the existing lower/mid/upper state concept, with
fixed pressure-mass fractions of 0.40, 0.35, and 0.25. At the standard surface
pressure this gives a total atmospheric column mass of `ps / g`; a layer's
stored water is `m_layer * q_layer`, not the unweighted sum of its mixing
ratios. The representative heights are 0 m, 3.5 km, and 8 km.

For each layer it carries:

- specific humidity `q` [kg kg-1];
- temperature `T` [K]; and
- moist static energy `h = cp*T + g*z + Lv*q` [J kg-1].

The existing `PlanetState` already has the mid/upper humidity and temperature
arrays, the two interface omegas, and independent middle/upper winds. A
runtime adapter must use those arrays; it must not introduce another wind state
or alter the supported `wind_u_aloft`/`wind_v_aloft` jet kernel.

## Finite-volume update

`pressure_column.evolve_closed_three_level_thermodynamic_column` is a pure
operator. It accepts the two diagnosed interface velocities, where positive
omega is downward. At each interface it moves a donor air mass
`abs(omega) * dt / g`; the donor's `q` and `h` move together. Fixed layer
masses mean the matching horizontal mass convergence is deliberately outside
this one-column operator and must be closed by the circulation closure.

There are only two allowed energy terms:

1. Layer radiative fluxes [W m-2], supplied explicitly and reported as the
   corresponding column energy input; and
2. Condensation, supplied as a vapour mixing-ratio removal. It leaves moist
   static energy unchanged, so `Lv*dq` becomes thermal energy exactly once.

The operator has no lapse-profile relaxation, temperature clipping,
precipitation removal, target rainfall, or empirical heat term. Rainfall,
falling condensate, surface enthalpy fluxes, and any horizontal energy flux
remain external budget terms and must be passed through a coupled adapter
before the operator can affect a climate run.

## Required diagnostics and closure checks

Every runtime candidate must emit, at least as evaluation diagnostics:

- layer `T`, `q`, pressure mass, and radiative flux/tendency;
- interface omega and the mass-weighted horizontal divergence residual;
- vapour, suspended-condensate, precipitation, and total-water budgets;
- moist-static-energy residual after explicit radiative, surface, and exported
  precipitation terms; and
- cross-equatorial moist-static-energy transport and seasonal jet cores.

The present kernel returns water and energy residuals plus its radiative input.
Unit tests prove internal vertical transport, latent conversion, source
accounting, timestep subdivision, and invalid-source rejection. Those tests
are a proof of the operator, not evidence that a coupled climate is valid.

## Runtime admission sequence

The next implementation change is a single adapter, default-off, which must:

1. derive the layer radiative/surface/precipitation source terms from existing
   resolved fields without double-counting the lower air-temperature update;
2. route the returned lower temperature into `PlanetState.air_temperature` and
   the returned mid/upper arrays into their existing state fields;
3. retire the old three-level lapse relaxation and fractional latent-heating
   additions **inside that new gate only**;
4. preserve bit identity when the gate is false, persist state deterministically,
   and report the complete budget; then
5. pass a 64x128 compact CRU screen before any long benchmark.

No existing three-level, overturning, or wind scalar is promotable merely
because this kernel exists. The prior full benchmark rejected that family for
overall climate skill and cost; this contract is the required architectural
repair before a new candidate can be evaluated.

## First runtime-adapter verdict (2026-08-12): redesign required

The adapter was coupled behind the new default-off
`enable_closed_three_level_thermodynamics` gate and evaluated once at 64x128,
with one year of spin-up and one year of evaluation. It used the complete
existing prerequisite family (prognostic column water, condensate,
stability-aware convection, two-layer adjustment, and the three-level pressure
column), with no new tuning scalar or changed supported default.

It fails catastrophically rather than marginally: global air temperature was
164.36 K, precipitation 0.245 mm/day, and the composite reference-error score
4.429. The diagnosed lower-mid and mid-upper omega RMS values were 14.58 and
29.99 Pa/s. At daily microphysics cadence, either value transfers more than a
full pressure layer (`abs(omega) * dt / dp > 1`) in one update; the
finite-volume donor limit preserves non-negative water, but cannot make that
resolved vertical mass flux physically valid. This is why a numerically
conservative operator can still produce a climate collapse.

**Decision:** retain the kernel and adapter as default-off research code, but
do not run a long benchmark or tune its exchange, wind, heating, or damping
scalars. The prerequisite is now explicit: derive vertically consistent winds
and interface mass fluxes with a documented CFL/stability bound *before*
coupling thermodynamics. A cap applied only to omega after the fact would
discard the diagnosed circulation and merely create another unclosed tuning
path; it is not an admissible repair.

## Diabatic-interface branch verdict (2026-08-12): mechanism isolated, not admitted

The first redesign increment is implemented behind the nested default-off
`enable_diabatic_interface_mass_flux` gate. It does not reuse raw divergence
from the independently evolved lower/middle/upper winds. Instead, it derives a
zonal-mean large-scale latent-heating flux from the preceding raw-column
precipitation, distributes that heating uniformly per free-tropospheric mass,
and balances it against resolved lower-mid and mid-upper static stability. The
three implied layer divergences satisfy the 0.40/0.35/0.25 column constraint
exactly. If a daily flux has Courant number above 0.25, the same flux is
conservatively integrated in enough inner updates; it is not capped.

Its 64x128, one-year spin-up/one-year evaluation screen confirms the intended
isolation: omega RMS falls from 14.58/29.99 Pa/s in the raw-wind closure to
0.0248/0.0229 Pa/s, and the global temperature remains 295.51 K rather than
collapsing. The candidate nevertheless fails every climate admission metric:
precipitation is 1.34 mm/day, KÃ¶ppen group/class accuracy is 0.571/0.317, and
the composite reference-error score is 0.545. It also retains 2177 PW
cross-equatorial total energy transport.

This separates two defects that must not be conflated: the raw layer-wind
divergence was invalid as a vertical mass-flux source, while the raw horizontal
wind field is independently invalid as an energy-transport carrier. The new
omega path repairs the former but intentionally does not rewrite those winds,
so it cannot promote the complete circulation. **Do not tune the diabatic
heating profile or vertical substep bound.** The next architecture must derive
one shared horizontal mass/energy circulation whose wind and omega branches
come from the same pressure-coordinate continuity solve.

## Shared pressure-coordinate solve verdict (2026-08-12): transport repaired, climate rejected

`enable_shared_pressure_coordinate_circulation` implements that one shared
solve. The raw longitude-varying zonal winds are reduced to their
non-divergent longitude means. Each layer's meridional wind is reconstructed
from the same latent-heating/static-stability divergence used to diagnose the
two interface omegas. Thus winds, divergence, omega, and transport finally
share one pressure-coordinate continuity contract. There is no circulation
strength, damping, or omega-cap parameter.

The 64x128 one-year spin-up/one-year evaluation is a real transport success:
omega RMS is 0.00198/0.000688 Pa/s and cross-equatorial total energy transport
is -9.66 PW (previously 2177 PW with repaired omega but unrelated winds). Peak
transport is also reduced to +62.9/-74.6 PW. Runtime is 42.3 seconds, far
below the prior multi-minute experimental configurations.

It still fails the climate gate: global precipitation is 0.954 mm/day,
KÃ¶ppen group/class accuracy is 0.501/0.287, and reference error is 0.696.
The coherent slow overturning no longer supplies the transient/convergent
moisture production that the old empirical precipitation path implicitly
relied on. **Decision: retain as the valid circulation foundation, but reject
it as a complete climate candidate.** Do not strengthen the circulation. The
next missing state is an explicit moisture source/condensate-rainfall pathway
coupled to this resolved large-scale circulation; it must be designed against
the new raw precipitation deficit rather than restoring allocator targets.

## Pressure-mass moisture closure verdict (2026-08-12): budget repaired, coupled dynamics rejected

`enable_pressure_coordinate_moisture_closure` puts surface evaporation, three
layer vapour reservoirs, persistent cloud water, and fallout into pressure-mass
units. Its focused regression verifies the source -> vapour -> cloud -> fallout
path; a 64x128 30-day screen closes global water to 1.34e-7 relative residual.

The coupled candidate is rejected: rainfall peaks at 54.6 mm/day and the
lower-interface omega at 1.27 Pa/s. Condensation anomalies feed directly into
next-step diagnosed omega, which then concentrates vapour and creates the next
anomaly. Conservative substeps retain water, but cannot resolve that missing
large-scale adjustment timescale. Do not tune fallout, RH, or circulation
strength. Retain the budget kernel default-off; next build a prognostic
large-scale overturning/heat reservoir with compensating energy export.

## Prognostic overturning-heating verdict (2026-08-12): stable, retained, not promoted

`enable_prognostic_overturning_heat_reservoir` supplies that missing
adjustment. It stores only the zonal, cosine-area-balanced condensation-heating
anomaly and relaxes it on a derived free-tropospheric radiative time of about
54.4 days (heat capacity divided by `4 sigma T^3`). The adjusted heating drives
the same shared pressure-coordinate omega solve. There is no user-selected
timescale, damping, or circulation-strength control.

The first 64x128 screen appeared to improve precipitation to 2.558 mm/day, but
the subsequent phase audit found that result invalid: the stored cloud mass
[kg m-2] had been handed to cloud radiation as a mixing ratio. That made a few
mm of water optically opaque. The boundary now converts cloud mass through the
midlevel pressure mass before radiation sees it.

The corrected one-year spin-up/one-year evaluation remains numerically stable,
but fails climate admission: global precipitation is 0.530 mm/day, polar
precipitation 0.965 mm/day, cloud fraction 0.434, reference error 0.870, and
omega RMS 0.00161/0.00175 Pa/s. **Decision:** retain the circulation, water,
and unit contracts as default-off infrastructure, but reject the climate
configuration. Do not restore the unit error or tune cloud/rain coefficients.
The next missing state is a pressure-mass separation of suspended cloud water
from precipitating hydrometeors, including their distinct transport and
residence paths.

## Separate cloud/hydrometeor pressure-mass path (2026-08-12): implemented, compact screen rejected

The nested default-off `enable_separate_precipitating_hydrometeors` path now
works with the pressure-coordinate moisture closure. `atmospheric_condensate`
is suspended cloud water [kg m-2]; `precipitating_hydrometeors` is a second,
also pressure-mass, falling reservoir. Newly condensed water joins the cloud
reservoir, cloud excess autoconverts over the existing autoconversion time,
and only hydrometeors sediment over the fallout time. Suspended cloud water
uses its cloud-layer transport path, while hydrometeors may be carried only
for their aloft residence before fallout.

The focused regression closes a 4 + 2 kg m-2 cloud/hydrometeor inventory
through autoconversion and fallout, and verifies that surface precipitation is
the hydrometeor fallout—not a direct cloud-water sink. A second controlled
transport regression verifies that suspended cloud mass is conserved over the
full cloud-layer transport step while hydrometeors travel only for their aloft
residence and then sediment, giving them different horizontal footprints.

The initial unchanged-coefficient 64x128 screen exposed a diagnostic
singularity: on evaluation day 15 its stored heating jumped from order 500 W
m-2 to 905,002 W m-2 because the interface solve divided heating by raw
temperature difference. A raw-T-neutral layer is not necessarily
neutrally stratified: its potential temperature rises strongly with height.
The coupling now diagnoses interface stability from the derived three-layer
potential-temperature gradient. This is a unit/physics correction, not a
stability floor or omega cap; genuinely potential-temperature-unstable layers
still have no resolved large-scale omega diagnosis.

The corrected 64x128 one-year spin-up/one-year evaluation is computationally
stable (42.3 s) but fails climate admission: global precipitation is 1.336
mm/day, polar precipitation 4.670 mm/day, cloud fraction 0.539, reference
error 0.955, KÃ¶ppen group/class accuracy 0.529/0.252, and lower/mid-upper
omega RMS 0.0411/0.0234 Pa/s. The final mean suspended-cloud and
hydrometeor masses are 3.475 and 0.336 kg m-2. **Decision:** retain the
potential-stability and separate-reservoir contracts as default-off
infrastructure, but reject the complete candidate. Do not tune
autoconversion, fallout, or cloud retention. The remaining defect is a
large-scale heating amplitude/phase response, which needs an energy-export
closure rather than microphysical compensation.

## Shared-wind MSE export screen (2026-08-12): transport contract retained, coupling rejected

`enable_pressure_coordinate_mse_transport` is a nested default-off companion
to the pressure-mass moisture closure. It transports each fixed pressure
layer's moist-static-energy content on exactly the conservative horizontal
faces already used for that layer's vapour, then recovers temperature from the
transported MSE and humidity. Lower-layer evaporation supplies the matching
latent-energy flux. The pure regression confirms transport conservation and
that a vapour anomaly carries its latent energy without changing a uniform
temperature tracer.

The unchanged 64x128 one-year spin-up/one-year evaluation is stable (44.1 s),
but rejects the coupled candidate: global precipitation is 0.481 mm/day,
polar precipitation 0.207 mm/day, cloud fraction 0.549, reference error
0.898, KÃ¶ppen group/class accuracy 0.500/0.273, and lower/mid-upper omega RMS
0.00463/0.00102 Pa/s. Its implied peak northward/southward MSE transports are
28.6/-94.7 PW, far beyond an admissible planetary energy transport. **Decision:**
retain the conservative MSE-transport kernel as default-off infrastructure,
but reject its direct coupling to the existing diagnosed winds. Do not tune
wind, radiative, autoconversion, or fallout coefficients. The next closure
must diagnose the circulation from the simultaneously conserved MSE transport
and diabatic forcing, so the transport itself constrains overturning strength.

## MSE-constrained pressure circulation screen (2026-08-13): kernel retained, two-branch candidate rejected

`enable_mse_constrained_pressure_circulation` is the first joint solve. It
balances the zonal, cosine-area-balanced diabatic forcing by a meridional MSE
transport at every latitude circle. A lower/upper pressure branch carries
equal and opposite dry-air mass, and its *resolved signed* MSE contrast fixes
the branch mass flux; its divergent winds, pressure-interface omega, and MSE
transport therefore follow from one calculation. The middle layer remains a
non-divergent branch in this deliberately limited first formulation. There is
no speed coefficient, relaxation, omega cap, or MSE-contrast floor. A truly
vanishing branch contrast is rejected as having no finite two-branch solution.

Analytic regressions verify: uniform diabatic forcing produces no circulation;
an equatorial heating anomaly produces poleward MSE export in each hemisphere;
the layer-weighted mass divergence is zero; and a vanishing MSE contrast is
rejected. The 64x128 one-year spin-up/one-year evaluation is stable (44.5 s)
and removes the previous +28.6/-94.7 PW transport pathology, reaching
+9.42/-6.00 PW. It nevertheless rejects the climate candidate: global
precipitation is 0.688 mm/day, polar precipitation 0.208 mm/day, cloud
fraction 0.567, reference error 0.663, KÃ¶ppen group/class accuracy
0.486/0.271, and lower/mid-upper omega RMS 0.000454/0.00410 Pa/s.
**Decision:** retain the pure joint-budget solver as default-off
infrastructure, but do not promote or tune the two-branch runtime coupling.
The next redesign is a three-branch MSE/vertical-exchange solve: it must derive
the middle-layer MSE flux and phase-heating deposition with the two interface
mass fluxes, rather than treating the middle layer as horizontally passive.

## Three-branch MSE/vertical-exchange screen (2026-08-13): closure retained, minimum-norm branch selection rejected

`enable_three_branch_mse_pressure_circulation` extends the joint MSE solve to
all three pressure layers. At each latitude it selects the unique
minimum-mass-flux velocity vector that has zero column mass flux and carries
the required MSE export. It then diagnoses both interface mass fluxes and the
layerwise diabatic deposition needed to balance horizontal MSE export plus
vertical MSE exchange. The phase-deposition fields remain diagnostic: applying
them to the existing phase-conversion temperature update now would double-count
latent heat.

The pure regressions verify three-layer mass closure, exact recovery of the
column diabatic forcing from the three deposition fields, a nonzero middle
branch, and rejection of a vertically uniform MSE state. The 64x128 one-year
spin-up/one-year evaluation is stable (38.9 s), but rejects the runtime
candidate: global precipitation is 0.428 mm/day, polar precipitation 0.071
mm/day, cloud fraction 0.571, reference error 0.754, KÃ¶ppen group/class
accuracy 0.483/0.274, and lower/mid-upper omega RMS 0.00102/0.0108 Pa/s. Peak
northward/southward energy transports fall to +0.745/-0.160 PW, showing that
minimum mass flux is not a physically sufficient branch-selection principle.
**Decision:** retain the three-branch budget/deposition diagnostics as
default-off infrastructure, but reject the coupled candidate. Do not tune a
branch weight or restore the two-branch solution. The next required constraint
is a momentum/thermal-wind relation that selects the vertical branch structure
while the MSE and mass budgets remain exact.

## Momentum-constrained three-branch screen (2026-08-13): algebra retained, resolved-wind constraint rejected

`enable_momentum_constrained_three_branch_mse_circulation` replaces the
minimum-norm branch choice with zero mass-weighted transport of resolved
pressure-level zonal momentum. At a fixed latitude the planetary
angular-momentum term cancels under exact column mass closure, leaving this
third linear constraint alongside mass and MSE export. Its analytic regression
closes all three budgets and rejects barotropic (vertically unsheared) winds.

The 64x128 one-year spin-up/one-year evaluation is computationally stable
(41.6 s) but rejects the candidate: global precipitation is 0.497 mm/day,
polar precipitation 0.075 mm/day, cloud fraction 0.578, reference error
0.883, KÃ¶ppen group/class accuracy 0.465/0.260, and lower/mid-upper omega RMS
0.00179/0.01499 Pa/s. The northern peak energy transport explodes to 69.4 PW
(southern peak -2.75 PW). **Decision:** retain the algebraic budget test as
default-off infrastructure, but reject runtime use of the present resolved
zonal winds as a momentum constraint. They are not themselves a prognostic,
pressure-coordinate thermal-wind/momentum balance. Do not cap transport or
tune a branch weight. The next architecture must evolve pressure-level zonal
momentum with explicit pressure-gradient, Coriolis, and vertical momentum
exchange terms before its shear can constrain the MSE circulation.

## Prognostic pressure-momentum integration (2026-08-13): kernel retained, monthly coupling inadmissible

`enable_prognostic_pressure_coordinate_momentum` advances all three pressure
wind levels before branch selection. The pure operator obtains each layer's
meridional pressure-gradient acceleration from hydrostatic pressure thickness,
integrates Coriolis rotation analytically (including the finite equatorial
limit), and transfers both horizontal momentum components conservatively across
the prior-step pressure interfaces. Its regressions verify exact split-step
agreement under fixed forcing and mass-weighted horizontal-momentum conservation
under vertical exchange.

The daily small-grid path is finite, but the required 64x128 MONTHLY compact
integration is not admissible. Its first two monthly cycles take 9.03 and 8.43
seconds; after cycle two, lower/mid-upper interface omega already reaches
0.00319/0.00987 Pa/s. The third cycle enters the existing conservative
closed-column substep path and does not complete within the two-minute runner
limit. This is a true coarse-cadence/CFL failure, not a candidate climate
result. **Decision:** retain the pure momentum operator default-off, but reject
its current split runtime coupling before climate scoring. Do not cap omega or
relax the winds. The remaining missing design is a jointly time-discretized
pressure-column momentum/MSE/vertical-transport solve, with one shared CFL
policy, rather than sequential monthly momentum and thermodynamic operators.

## Joint pressure-column time-integration kernel (2026-08-13): pure contract complete

`pressure_circulation.evolve_joint_mse_momentum_pressure_column` now provides
one adaptive state transition for the three-branch MSE/momentum diagnosis,
conservative vertical MSE exchange, and pressure-level momentum update. Each
substep re-diagnoses circulation from its current state, applies half momentum,
one full vertical MSE exchange, and the remaining half momentum using the same
interface flux. The substep duration is derived from that interface's shared
vertical Courant number; it neither clips omega nor prescribes an inner
timestep. Regressions verify its zero-state identity, finite coupled response,
water/MSE closure, and the shared CFL bound.

The runtime adapter now moves horizontal MSE transport, joint CFL integration,
and phase deposition into one transition under the deepest momentum/MSE gate;
the legacy pressure-column split is bypassed there. Its focused regressions
pass, including runtime wind persistence and the absence of the old separate
momentum step.

The legacy daily wrapper is also bypassed at this gate, so the host monthly
step calls the adapter once and its interface-CFL policy is the only inner
discretization. The first mass-closure report exposed that the previous
normalized 3x3 branch solve accumulated a tiny continuity residual. The solve
now eliminates its third branch algebraically, so pressure-mass continuity is
structural and its discrete-divergence regression passes.

The 64x128 compact path still fails admission in its second 6.09-day monthly
substep, but for the physically relevant reason: the simultaneous MSE and
zonal-momentum constraints become near-degenerate (condition number
2.64e9), demanding a 3.30e9 m s-1 branch speed and 8.77e3 s-1 divergence.
The reported absolute mass residual (1.36e-12 s-1) is only floating-point
roundoff relative to that divergence; it is not the cause. This produces no
climate metric and remains rejected. Retain the adapter default-off; do not
add an omega cap, damping term, fallback branch, or prescribed joint inner
timestep. A subsequent redesign must supply a physically independent third
branch constraint rather than forcing exact momentum transport through a
nearly barotropic MSE state.

## Column-water-constrained three-branch diagnostic (2026-08-13)

`water_constrained_three_branch_mse_pressure_coordinate_circulation` now
supplies that independent pure constraint: after structural pressure-mass
closure, it solves the required MSE and column-vapour transports together.
Its water transport is diagnosed from the cosine-area-balanced atmospheric
column source/sink, rather than from a resolved zonal-wind shear. The focused
regression verifies discrete mass closure, finite interface omega, and a
nonzero middle branch.

An attempted partial runtime coupling used surface vapour supply minus prior
precipitation as that source/sink. It is inadmissible: surface supply can be
nonzero while the simultaneously prescribed MSE export is zero, which forces
large compensating branches. The next implementation must therefore put
surface evaporation, ascent/condensation, cloud storage, and fallout in the
same state transition before this water constraint is allowed into the monthly
path. Do not use lagged precipitation, a branch weight, a cap, or damping to
manufacture that closure.

The diagnostic's layerwise MSE deposition now solves its same two constraints
at pressure-latitude faces, using face-interpolated MSE and vapour states.
Interpolating the already-solved cell-centre branch transports can cancel at a
face even when the finite-volume MSE budget requires a nonzero transport. The
face solve preserves both exact transports and has a regression for nonzero
MSE transport with zero column-water transport; it does not introduce a
normalising branch share.

An undamped pure-runtime fixed-point probe was also rejected, not wired: with
the existing 80 W m-2 analytic heating case, its first diagnosed phase sink
was 6.85e-4 kg m-2 s-1 and the next water-constrained solve required a 94.6
m s-1 middle branch. Thus merely iterating the current condensation activation
does not make the transition simultaneous; it amplifies a vapour-only sink
that excludes contemporaneous cloud storage, hydrometeor conversion/fallout,
and the matching large-scale latent-heating update. The next closure must
solve those prognostic reservoirs and heating together, rather than applying
this fixed point or a relaxation to it.

`evolve_pressure_condensate_reservoirs` is now the sole pressure-path
cloud/hydrometeor conversion and fallout transition, and
`column_water_forcing_from_boundary_fluxes` defines its circulation input as
``(surface source - fallout) / elapsed seconds`` in kg m-2 s-1. Phase change
does not enter that forcing because it is internal to total atmospheric water.
These pure contracts are tested and are intentionally not yet passed into the
runtime water branch until their fallout and latent-heating diagnostics are
solved with the same circulation state.

`diagnose_joint_pressure_column_coupling_residual` now evaluates precisely
that candidate transition and returns both residuals. The comparison projects
both candidate and diagnosed fields onto the cosine-area-balanced zonal
anomaly, because a closed meridional circulation cannot determine their global
mean or longitude-dependent component. A regression verifies that adding a
uniform heating or water boundary term leaves the residual unchanged. A scaled
8x16 nonlinear least-squares probe (heating scale 100 W m-2; water scale
1e-4 kg m-2 s-1) terminates with maximum scaled residual 0.89 rather than a
root. Bounded attempts with standard hybrid, Levenberg--Marquardt, and Broyden
root methods likewise do not establish a finite root (the Broyden update
becomes singular). It is therefore not a usable solver or runtime gate: the
current phase-activation/reservoir equations do not yet supply a simultaneous
closure.

The missing transient storage term is now included: the water constraint is
``surface source - fallout - d(vapour + cloud + hydrometeors)/dt``, not a
steady ``source - fallout`` approximation. With that correction, the exact
undamped candidate map converges in the analytic case to float32-consistent
water/heating residuals. The solver canonicalises its unknowns to balanced
zonal anomalies, updates by the full diagnosed state (never a relaxed blend),
and raises rather than returning a partial iterate. The resulting
`evolve_simultaneous_joint_pressure_column_runtime` also owns the established
cloud and hydrometeor horizontal transport leg before reservoir conversion.
Its pure regressions pass; it is still deliberately outside the monthly
atmosphere gate pending the required 64x128 admission screen.

The first atomic admission attempt was rejected at the smaller 12x24,
one-day runtime screen, before a 64x128 climate run was warranted. A nonlinear
candidate reached a water/MSE constraint condition number of 2.91e17 and
demanded a 9.73e9 m s-1 branch, producing 1.71e4 s-1 divergence. This is a
genuine degeneracy of the simultaneous transport constraints, not a shared-CFL
failure or an accepted partially converged state. The default-off atmosphere
gate was removed; the pure adapter, transient storage budget, reservoir
transport contract, and strict failure diagnostics remain retained. No cap,
damping, fallback branch, or prescribed inner timestep was introduced.

The analogous missing MSE-storage term is now factored as
`column_mse_storage_tendency_w_m2`: pressure-layer dry-static, geopotential,
and vapour latent storage divided by elapsed time. The next residual must keep
the heating reservoir as its own diagnosed state and use ``heating - storage``
for the MSE transport constraint. It must be evaluated as a pure exact map
before any runtime admission attempt.

That exact-map experiment is now rejected: on the analytic 8x16 case it enters
the active rank-deficient MSE/water constraint before convergence and produces
nonphysical thermodynamic and forcing excursions. The integration was removed,
while the storage diagnostic remains tested. This confirms that neither a
steady nor a transient diagnostic MSE/water branch system has an admissible
simultaneous runtime solution; the next required replacement is prognostic
pressure-coordinate transport, not another diagnostic branch selection.

## Hydrostatic-sigma runtime admission (2026-08-13): redesign required

`enable_hydrostatic_sigma_pressure_coordinate_transport` now owns a default-off
atomic transition through the precipitation adapter. It persists the three
layer pressure depths plus cloud and hydrometeor reservoirs, returns the
closure's temperature and momentum state, and bypasses the legacy lagged
overturning-heating update. Gate-off state identity, persistence, pure-kernel
conservation/CFL checks, and a finite one-day 8x16 enabled transition are
covered by focused regressions.

The required 64x128 one-year spin-up/one-year evaluation MONTHLY screen was
started with the full prerequisite family enabled and rejected during its first
spin-up cycle, before climate scores were meaningful. Horizontal transport of
the inherited legacy wind state exhausted a pressure layer in at least one
cell before the vertical donor exchange. The kernel now raises that explicit
admissibility error rather than dividing by an empty donor, taking an unbounded
CFL sequence, capping the carrier, or reverting to a legacy path.

**Decision:** retain the gate and pure closure as default-off research code;
do not tune a wind scale, mass floor, damping, cap, fallback, or prescribed
inner timestep. The next required Phase-2 replacement is a simultaneous
horizontal pressure-mass and momentum transition that supplies an admissible
carrier to the hydrostatic vertical/phase closure. The supported Earth
baseline remains unchanged.

`evolve_hydrostatic_sigma_mass_momentum` now provides the resulting pure
carrier, including vapour/MSE/cloud/hydrometeor inventories on the identical
adaptive donor parcels. Its controlled 30-day regression closes the carried
inventories and remains finite. A 64x128 one-month runtime-coupling probe did
not finish within 60 seconds, so this carrier is deliberately not substituted
into the atmosphere gate: runtime admission requires an equivalently
conservative implementation with bounded compact-screen cost, not relaxed
Courant control or a capped wind.

## Coupled two-layer grey radiation admission (2026-08-15): handoff instability isolated, redesign required

`enable_coupled_two_layer_grey_radiation` replaces the legacy radiative
tendency with a conservative pressure-defined two-layer grey budget, routing
its midlevel/upperlevel gains through the closed three-level MSE column
(`evolve_closed_three_level_thermodynamic_column`) rather than a second,
independent atmospheric energy owner. The column-internal residual is exact
(`grey_column_conservation_residual_w_m2` is float64-roundoff-level, ~1e-14)
and 125 related tests pass, but the 32x64 one-year spin-up/one-year evaluation
screen (`scripts/audit_boundary_layer_column_energy.py`) rejects the candidate
on temperature shock (+17.3 K air, +15.0 K surface) and TOA imbalance
(-39.2 W m-2).

A new full-system (atmosphere + land/ocean skin + land-deep + deep-ocean)
energy-conservation check was added to the audit script alongside the
existing column-internal one, comparing total storage change against the
time-integrated `grey_toa_net_radiation_w_m2` -- the only flux that should
cross the system boundary. It exposes a genuine, un-owned ~105 W m-2 mean
annual residual, dominated by the surface/ocean reservoir. A flag-knockout
sweep and a reverted trial (removing the legacy `T_eq` feed into
`calculate_ocean_heat_transport`'s exchange term, which made the residual
*worse* rather than better) both rule out the capacity-aware air-sea/free-air
exchange, boundary-layer horizontal transport, the interface reservoir, and
that legacy exchange term as the primary driver -- each is independently
conservative or a small, steady bias, not this magnitude.

Per-step tracing over the first several days after the handoff finds the
actual mechanism: `diabatic_interface_mass_flux`'s precipitation-anomaly path
computes omega by dividing a heating anomaly by resolved potential-temperature
static stability with no floor (by design -- see the diabatic-interface
verdict above). On the coupled-grey handoff specifically, the freshly
bootstrapped mid/upper-level temperatures start close to neutral stability at
some cells, and the resulting omega feeds back through next-day's
precipitation anomaly: the diagnosed vertical Courant number roughly doubles
each day (measured 0.069 -> 0.804 -> 1.733 -> 3.48 over four days). The
existing CFL substep mechanism keeps each individual transfer conservative and
bounded, so the closed column's own energy ledger stays exact even as this
happens, but the compounding vertical mixing still drives mid/upper
temperatures to non-physical values within days (T_air up to 556 K, T_mid
down to -27 K measured), which is what the full-system audit reads as a large,
sign-flipping implicit source.

A stability-floor patch to the omega denominator was implemented and measured
as numerically effective (day-4 Courant number bounded at ~0.4-0.7 instead of
3.48; admission-screen air shock fell from +17.3 K to +3.4 K, implicit source
from 105 to 79 W m-2, all 112 pressure-circulation/pressure-column/prior-art
tests still passing) -- and then reverted unapplied. It contradicts this
document's own repeated, explicit verdicts on this exact closure family
("Do not add an omega cap, damping term..."; "A cap applied only to omega
after the fact would discard the diagnosed circulation and merely create
another unclosed tuning path; it is not an admissible repair."). Patching the
symptom would also leave the remaining TOA/OLR-target mismatch (independently
present before this instability was even triggered) unaddressed.

**Decision:** retain the coupled grey-radiation gate, the closed-column
adapter, and the new full-system energy-audit instrument as default-off
research code; do not add a stability floor, omega cap, or damping term to
`diabatic_interface_mass_flux_from_heating`. The next required replacement is
a coupled-grey-aware spin-up/initialization for the mid/upper-level
temperature and optical-depth state (so the handoff does not start near
neutral stability), or a structurally independent bound on the diagnosed
overturning consistent with the rest of this closure family's redesign
requirement -- not a post-hoc cap on this specific instance. The TOA/OLR
target mismatch (`grey_target_olr_residual_w_m2` ~+11-13 W m-2 even before
this instability compounds it) is a separate, still-open calibration gap in
the grey column's optical-depth solve and must be resolved independently of
the handoff-stability fix.

## Handoff pathway diagnostic (2026-08-16): trigger isolated; near-neutral state develops, not inherited

`scripts/diagnose_coupled_grey_handoff.py` reproduces the admission screen's
exact warmup (12 MONTHLY cycles at 32x64, closed-column family off, profile
gate on), records the handoff state, then traces 14 coupled DAILY steps with
an exact recomputation of the live diabatic-omega branch, the zonal
potential-temperature stability at both interfaces, the closed-form grey
radiative-equilibrium profile implied by the persisted optical depth, and the
audit script's full-system energy ledger. The instrument is validated against
the earlier ad hoc trace: it reproduces the day-4 midlevel collapse to
-26.9 K (prior trace: -27 K).

Measured findings (`temp/coupled_grey_handoff_32x64.json`):

1. **The handoff state is not near-neutral.** At handoff the zonal stability
   is uniformly positive (lower-mid 5.6e-4 K/Pa, mid-upper 2.2e-3 K/Pa; zero
   area below 1e-4 K/Pa) and the implied Courant numbers are benign
   (0.085/0.025). The earlier "starts close to neutral stability" description
   is refuted at the zonal scale the omega solve actually uses: the
   near-neutral state *develops* during the first one to two coupled days
   (lower-mid p5 stability is -2.4e-4 K/Pa after day 1 and stays negative).
2. **The trigger is the grey-gain shock from a mis-initialized mid level.**
   The dry-adiabatic bootstrap mid level is +15.2 K area-mean (max +60 K)
   warmer than the grey radiative equilibrium implied by the persisted
   optical depth (equilibrium mid 249.4 K vs bootstrap 264.6 K; upper level
   is nearly consistent at -1.2 K mean). Day-1 grey gains extract -40 W m-2
   from the mid level and deposit +31/+22 W m-2 in the upper/surface levels;
   the omega/vertical-exchange feedback then executes the collapse (T_mid
   range 90-300 K on day 2, -27-313 K on day 4; post-day-1 implied Courant
   1.12). The daily implicit source oscillates between roughly -1.6 and
   +2.9 kW m-2 and accumulates to +27 W m-2 by day 14.
3. **Pure grey radiative equilibrium is not itself an admissible
   initialization.** Its lower-mid stability is below 1e-4 K/Pa over ~23% of
   area (negative over ~17%) -- radiative equilibrium is super-adiabatic in
   the lower troposphere, which is precisely what convective adjustment
   exists to repair. A grey-aware initialization must therefore be
   radiative-*convective*: the grey equilibrium profile limited by the
   adiabat, matching the family's already-required
   `enable_two_layer_convective_adjustment`.
4. **The OLR-target gap compounds with the collapse.**
   `grey_target_olr_residual_w_m2` grows from -9.3 W m-2 on day 1 to
   -29.9 W m-2 by day 4, so its independent calibration fix must be evaluated
   against a stable integration, not this one.

**Next bounded build:** a pure initialization kernel that, at the first
coupled step, sets mid/upper temperatures to the adiabat-limited grey
radiative-convective equilibrium from the persisted optical depth and current
surface state, with regressions verifying zero grey layer gains and
non-negative zonal stability at handoff. Whether an ongoing structurally
independent bound on the diagnosed overturning remains necessary is an open
question until that initialization is screened: days 2-14 show the stability
denominator collapsing even after the initial shock, so the initialization
alone may not be sufficient.

## Adiabat-limited equilibrium initialization screen (2026-08-16): rejected; instability is a property of the coupled operator, not the handoff state

The bounded build above was implemented and screened.
`grey_radiative_convective_equilibrium_temperatures` (atmospheric_radiation.py)
solves the model's own two-layer grey gains for zero exactly (linear in the
blackbody fluxes), limited from below by the dry adiabat at the omega
diagnostic's layer centres, with the upper level re-solved against the actual
clamped mid level; `initialize_coupled_grey_profile` (simulate.py) applies it
once at the warmup-to-coupled handoff. Unit regressions verify zero layer
gains for unclamped cells, exact adiabatic floors for clamped cells, and
elementwise theta monotonicity. The diagnostic script's `--initialize` mode
screens it at 32x64 (`temp/coupled_grey_handoff_initialized.json`).

Measured outcome:

1. **The kernel works as designed.** Day-1 grey gains fall from
   -40.3/+30.7/+21.7 W m-2 (mid/upper/surface, uninitialized) to
   -5.9/-1.9/+14.6 W m-2, and the initialized profile has non-negative zonal
   stability at every row. The grey-gain trigger identified by the handoff
   diagnostic is removed.
2. **The adiabatic limiter places ~15% of zonal rows exactly on the neutral
   singularity.** Clamped rows sit at theta_mid == theta_lower by
   construction, so the omega denominator there is ~3e-7 K/Pa and the
   handoff-state recompute gives a lower-mid Courant maximum of 124.6
   (uninitialized: 0.085). Because the closed column's substeps preserve the
   diagnosed flux exactly (atmosphere.py: quarter-layer substepping), that
   omega is applied in full -- the affected columns are homogenized many
   times over on day 1.
3. **The column collapses anyway.** T_mid reaches -34 K on day 3, the
   lower-mid Courant spikes to 18.1 on day 5, and the lower-mid p5 stability
   is negative from day 1 onward -- the same qualitative trajectory as the
   uninitialized run. The daily implicit source still oscillates at kW m-2
   amplitude.

**Decision:** the adiabat-limited equilibrium initialization is rejected as
the handoff repair, and -- more importantly -- the screening result closes
the initialization question itself. Both a healthy-stability handoff
(uninitialized: 5.6e-4 K/Pa everywhere) and a zero-grey-gain handoff
(initialized) reach the same collapsing state within one to two days, because
the grey budget's own equilibrium tendency drives the dry static stability
toward zero and the diabatic-omega operator is singular exactly there. No
choice of initial state can keep the handoff away from that singularity for
more than about a day. The remaining admissible branch from the 2026-08-15
decision is therefore the operative one: a structurally independent bound on
the diagnosed overturning -- i.e. a reformulation of the omega diagnostic
whose stability measure is bounded under the grey budget's equilibrium
tendency (for example a gross moist stability that accounts for the latent
feedback the precipitation-anomaly forcing represents), not a floor, cap, or
damping term on the current operator. The kernel, helper, and regressions are
retained as default-off research code; the TOA/OLR optical-depth calibration
gap remains an independent open item.

## Criterion-1 boundedness diagnostic (2026-08-16): humidity-informed stability denominators eliminated

Following the goal recorded in `docs/REMAINING_WORK_PLAN.md` (2026-08-16:
diabatic-omega stability reformulation), the handoff diagnostic now measures
candidate reformulated stability measures along both collapsing 32x64 coupled
trajectories (uninitialized and grey-initialized, 14 days each): the current
dry `d(theta)/dp`, an equivalent-potential-temperature form
`d(theta_e)/dp` with `theta_e = theta exp(L q / cp T)`, and a cp-normalized
moist-static-energy form `d(theta + (L/cp) q)/dp`, all on the operator's own
zonal layer-centre structure, plus the omega/Courant each would diagnose from
the live heating.

Measured outcome (`temp/coupled_grey_handoff_32x64.json`,
`temp/coupled_grey_handoff_initialized.json`):

1. **At the rows that drive the singularity, the humidity candidates are
   indistinguishable from dry.** The minimum-stability rows are the dry
   subpolar rows, where the resolved mid-level humidity is effectively zero
   (zonal q_mid as low as 1e-18 kg/kg). There `theta_e` and `mse_cp`
   reproduce the dry value to within a few percent (e.g. day 4
   uninitialized: dry -1.79e-2, theta_e -1.91e-2, mse_cp -1.90e-2 K/Pa).
   The moisture correction is material only at tropical rows, which are not
   the singular ones.
2. **All three candidates go firmly negative at the same rows** (minima
   around -1e-2 K/Pa throughout the trajectory), so the collapse is not a
   near-zero approach any of them reclassifies; the Courant explosions come
   from other rows with small *positive* stability, and the candidate
   recompute reproduces the same spikes there (initialized day 3: dry 0.65,
   theta_e 1.53, mse_cp 11.1).
3. **Criterion 1 therefore fails for humidity-informed denominators:** no
   humidity-based stability measure stays positive and bounded at the
   rows/times where the heating acts, because there is no moisture there to
   inform it. This direction is eliminated with measurements.
4. **The remaining admissible formulations, with evidence:**
   (a) *Water-budget / moisture-convergence omega* -- the singular rows carry
   zonal precipitation of only 0.03-0.3 mm/day and heating anomalies around
   -3e-5 K/s, so an omega diagnosed from the column water budget
   (`omega ~ P g / (q_lower - q_mid)`) is order 1e-2 Pa/s there (Courant
   ~0.03) -- bounded by construction where the dry operator produces Courant
   ~60. Its own edge (q-contrast -> 0 in a homogenized column) must be
   measured next; this is also the documented Phase 2 moisture-closure
   direction ("circulation-coupled evaporation/condensate/fallout
   production").
   (b) *Storage/self-consistency partition* -- the singular-row heating
   anomaly (~2.4 K/day equivalent tendency) would erase the row's own theta
   contrast within the step, so the steady-balance assumption the operator
   encodes is internally inconsistent there; partitioning the heating between
   circulation and storage is the transient route, partially explored and
   rejected for the MSE/water branch system on 2026-08-13 and thus requiring
   a fresh formulation, not a revival.

Next diagnostic: water-budget omega boundedness along the same trajectories,
including its q-contrast edge and its implied heat-transport consistency with
the closed column.

## Water-budget omega diagnostic (2026-08-16): eliminated; diagnostic inversion is singular by construction

The water-budget form was measured on both 14-day trajectories:
`omega = -g * P_anomaly / (q_below - q_above)` per interface, zonal,
cosine-area-balanced anomaly exactly as in the current operator, with the
non-positive-contrast branch at zero.

Measured outcome:

1. **The same singularity, relocated.** The tropical rows carry large
   precipitation anomalies with modest resolved moisture contrast, giving
   Courant maxima of 51 (uninitialized day 1), 12 (day 6), and 152
   (initialized day 13). The calm polar rows (Courant ~0.03) are exactly
   where the dry operator explodes, and vice versa: the two diagnostics are
   complementary in where they diverge.
2. **The coupled collapse destroys the moisture contrast too.** The lower-mid
   q contrast is non-positive at 3-28% of rows along the trajectory (the
   homogenized column has q_mid >= q_lower), so the water budget's own
   denominator is driven through zero by the same state evolution.
3. **The heat budget does not close on it.** The implied heat transport
   `omega * dry_stability` exceeds the live heating by a median factor of ~5
   (consistency ratio median ~0.2), so even where bounded the water-budget
   omega would thermally destabilize the column it is meant to steady.

**Structural conclusion.** With this measurement, every *diagnostic*
inversion of the form `omega = forcing / (state contrast)` stands
eliminated: dry static stability (current), humidity-informed stability
(theta_e, MSE forms -- identical to dry at the moisture-free singular rows),
and water-budget moisture contrast (this section). The coupled grey/closed
column evolution drives *any* contrast through zero, so no state-contrast
denominator can bound the operator. The two remaining formulations are the
documented larger programs, not omega reformulations: prognostic
pressure-coordinate momentum transport (rejected 2026-08-13/15 for runtime
cost and spin-up layer exhaustion, with five runtime redesigns already
rejected) and the simultaneous phase/reservoir closure (rejected 2026-08-13
for constraint degeneracy, condition number 2.91e17). The diagnostic-omega
reformulation goal is therefore recorded as **rejected with measurements**;
its evidence defines the requirement any future operator must meet: the
transport must be a prognostic state or the phase/reservoir transition must
be simultaneous, because no diagnostic division survives the coupled state
evolution.
