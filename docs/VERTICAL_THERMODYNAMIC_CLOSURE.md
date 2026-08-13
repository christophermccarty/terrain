# Vertical circulation and thermodynamic-closure contract

Status: **experimental kernel complete; runtime coupling not yet admitted**
(2026-08-12).

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
