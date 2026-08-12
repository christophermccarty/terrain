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
