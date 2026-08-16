"""Planet-level physical parameters.

All simulation constants that differ between planets live here.  Pass a
``PlanetParams`` instance (or the ``EARTH`` singleton) through to any
function that previously hard-coded Earth-specific values (S0, obliquity,
Ω, R, surface pressure, etc.).  Default values are calibrated for Earth.

Usage
-----
from planet_params import EARTH, PlanetParams

# Earth simulation (default)
state, _ = simulate_step(state, days=1.0, planet_params=EARTH)

# Mars-like simulation
mars = PlanetParams(
    solar_constant=589.0,
    obliquity_deg=25.19,
    orbital_period_days=686.97,
    sidereal_day_hours=24.623,
    radius_m=3.3895e6,
    surface_gravity=3.71,
    surface_pressure_pa=636.0,
)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PlanetParams:
    """Physical constants for a simulated planet.  All SI unless noted."""

    # ------------------------------------------------------------------ #
    # Stellar / orbital
    # ------------------------------------------------------------------ #
    solar_constant: float = 1361.0
    """TOA insolation at the reference (mean) orbital distance [W/m²]."""

    obliquity_deg: float = 23.44
    """Axial tilt [degrees]."""

    orbital_period_days: float = 365.2422
    """Length of one orbit around the host star [days]."""

    eccentricity: float = 0.0167
    """Orbital eccentricity (0 = circular)."""

    perihelion_day: float = 3.0
    """Day of year when planet is closest to star (perihelion).
    Earth: ~Jan 3, i.e. day 3.  Irrelevant for circular orbits."""

    vernal_equinox_day: float = 80.0
    """Orbital day of northern vernal equinox.

    This is the zero phase for seasonal declination.  It is intentionally
    planet-relative rather than an Earth day-number convention.
    """

    enable_milankovitch_cycles: bool = False
    """Evolve obliquity, eccentricity, and precession over simulated time."""

    milankovitch_time_scale: float = 1.0
    """Orbital-cycle years elapsed per simulated orbital year.

    Use 1 for real pacing or a larger value (for example 10,000) to make
    multi-millennial cycles observable during an interactive run.
    """

    milankovitch_obliquity_amplitude_deg: float = 1.2
    """Sinusoidal obliquity amplitude around `obliquity_deg`."""

    milankovitch_eccentricity_amplitude: float = 0.02
    """Sinusoidal eccentricity amplitude around `eccentricity`."""

    # ------------------------------------------------------------------ #
    # Rotation
    # ------------------------------------------------------------------ #
    sidereal_day_hours: float = 23.9345
    """Length of one sidereal (stellar) day [hours]."""

    # ------------------------------------------------------------------ #
    # Size / gravity
    # ------------------------------------------------------------------ #
    radius_m: float = 6.371e6
    """Mean planetary radius [m]."""

    surface_gravity: float = 9.81
    """Surface gravitational acceleration [m/s²]."""

    max_elevation_km: float = 8.848
    """Highest terrain elevation [km] that normalized elevation `1.0` maps to
    (ACCURACY_AUDIT.md C3). Earth's default is Everest (8.848 km).

    Normalized elevation arrays are `[0, 1]`; converting them to a real altitude
    (needed for lapse-rate cooling and for Koppen elevation corrections) requires
    knowing what `1.0` means in metres. That ceiling was previously hardcoded as
    `8848.0` in four separate places across three modules, so loading Mars
    terrain silently rescaled Olympus Mons (~21.9 km, 2.5x Earth's max) down into
    Earth's height range. Threaded through
    `temperature.elevation_to_alt_km` (both its loaded-heightmap and procedural
    branches) and `climate_averages.classify_koppen` (both its elevation-delta and
    legacy full-elevation branches). The Earth default reproduces every previous
    hardcoded value exactly, so this is an exact no-op for Earth."""

    lapse_rate_k_per_km: float = 6.5
    """Environmental temperature lapse rate [K per km of altitude]
    (ACCURACY_AUDIT.md C3). Earth's default is the standard 6.5 K/km.

    Previously hardcoded as `6.5` at five call sites across `simulate.py`,
    `temperature.py` and `climate_averages.py`, making Mars's real ~2.5 K/km
    (lower gravity + CO2 atmosphere, so a much weaker vertical temperature
    gradient) unreachable -- Mars terrain got Earth's cooling-per-km verbatim.
    The Earth default is an exact no-op."""

    # ------------------------------------------------------------------ #
    # Atmosphere
    # ------------------------------------------------------------------ #
    surface_pressure_pa: float = 101_325.0
    """Mean surface pressure [Pa]."""

    mean_molar_mass: float = 0.029
    """Mean molar mass of the atmosphere [kg/mol]  (dry air ≈ 0.029)."""

    gas_constant_dry: float = 287.0
    """Specific gas constant for dry atmosphere [J/(kg·K)]."""

    cp_dry: float = 1004.0
    """Specific heat at constant pressure [J/(kg·K)]."""

    enable_pressure_defined_radiative_temperature_profile: bool = False
    """Persist dry-adiabatic mid/upper temperatures for radiation diagnostics.

    This state is defined from the explicit pressure thicknesses used by the
    native column and conserves dry potential temperature. It is independent
    of the deeper convection gates, default-off, and does not by itself alter
    the active radiative budget.
    """

    enable_coupled_two_layer_grey_radiation: bool = False
    """Replace the legacy radiative tendency with the conservative grey budget.

    Requires ``enable_pressure_defined_radiative_temperature_profile``. The
    summed atmospheric gain updates the existing full-column air-energy owner;
    midlevel and upper-level states retain the pressure-weighted differential
    response without adding duplicate atmospheric mass. The surface gain enters
    the existing land/ocean thermal stores. Default off until admission passes.
    """

    # ------------------------------------------------------------------ #
    # Effective single-layer greenhouse parameters
    # These are used in the temperature baseline calculation and will be
    # gradually superseded by the prognostic CO2 radiative forcing path.
    # ------------------------------------------------------------------ #
    epsilon_equator: float = 0.68
    """Effective longwave emissivity at the equator.
    Reduced from 0.78: tropical T_annual_mean was 308K (+6K over Earth 302K).
    0.68 cools equator by ~6K, mid-lats by ~3-4K, poles unaffected (epsilon_pole unchanged).
    """

    epsilon_pole: float = 0.70
    """Effective longwave emissivity at the poles."""

    # ------------------------------------------------------------------ #
    # Aerosol / volcanic forcing
    # ------------------------------------------------------------------ #
    aerosol_optical_depth: float = 0.0
    """Stratospheric aerosol optical depth (AOD).
    0 = clear sky; ~0.1 typical Pinatubo forcing."""

    # ------------------------------------------------------------------ #
    # Surface / ocean
    # ------------------------------------------------------------------ #
    ocean_fraction: float = 0.71
    """Fraction of the surface covered by liquid-water ocean [0–1].
    Earth ≈ 0.71.  Dry planets (Mars) = 0.  Used to scale ocean heat
    capacity, evaporation rates, and Ekman transport coefficients."""

    has_liquid_water_ocean: bool = True
    """Whether the planet has a stable liquid-water ocean.
    When False, ocean heat transport and sea-ice dynamics are suppressed."""

    rotation_direction: int = 1
    """Prograde (+1) or retrograde (-1) rotation relative to the orbit.
    Flips the sign of the Coriolis parameter.  Earth = +1, Venus = −1."""

    # ------------------------------------------------------------------ #
    # Carbon / atmosphere composition
    # ------------------------------------------------------------------ #
    co2_baseline_ppm: float = 280.0
    """Pre-industrial CO2 reference concentration [ppm].  Used as C₀ in
    the radiative forcing formula ΔF = 5.35 ln(C/C₀).  Only meaningful
    for N₂-dominated atmospheres (Earth-like); set to 1 for CO₂-dominated
    atmospheres (Mars) where a different forcing model should be used."""

    co2_initial_ppm: float = 415.0
    """Initial atmospheric CO2 concentration at simulation start [ppm]."""

    bg_n2_frac: float = 0.7808
    """Background N2 volume fraction of the *dry, non-CO2/CH4* atmosphere.
    Fixed (not simulated) -- this model tracks CO2/CH4 as prognostic trace
    gases but treats N2/O2/Ar as an inert bulk background. Earth values are
    real dry-air composition; Mars is CO2-dominated so its bulk background
    is instead almost entirely CO2 (see MARS override below and
    `atmosphere_composition()`, which handles the two cases differently
    rather than trying to force one formula to cover both)."""

    bg_o2_frac: float = 0.2095
    """Background O2 volume fraction (dry, non-CO2/CH4 air).  See bg_n2_frac."""

    bg_ar_frac: float = 0.0093
    """Background Ar volume fraction (dry, non-CO2/CH4 air).  See bg_n2_frac."""

    # ------------------------------------------------------------------ #
    # Climate feedback / tunable physics
    # These constants can be swept by the optimizer or varied per-planet.
    # ------------------------------------------------------------------ #
    co2_climate_feedback: float = 0.8
    """Climate sensitivity parameter λ [K per W/m²] — Planck response only.
    Water-vapour amplification is now handled explicitly via wv_greenhouse_factor,
    which reduces epsilon when RH is high.  Together they reproduce Earth ECS ~3K.
    The pre-WV value was 1.4 (implicit WV included); lowering to 0.8 avoids
    double-counting.  MARS keeps 0.8 (no liquid water, no WV feedback)."""

    thermal_diffusivity: float = 0.04
    """Lateral atmospheric heat diffusion coefficient [K·day⁻¹ per K·cell⁻²].
    Controls pole-equator temperature gradient; higher = warmer poles."""

    polar_cooling_scale: float = 0.3
    """Polar latent-heat loss strength [dimensionless, 0–1].
    Scales the peak latent heat flux in the polar energy budget.
    Lower values allow more polar warming; higher values sharpen the
    equator-to-pole gradient."""

    land_transport_maritime_decay: float = 1.0
    """Continentality contrast on the land heat-transport bonus, in units of
    *fraction of the winter bonus per standard deviation of continentality*.
    0.0 is an exact no-op; positive values move part of each row's bonus from its
    continental interior to its maritime margins, **preserving the row's land
    mean**.

    **The defect this addresses** (2026-08-04, audit C1b). All three transport
    trapezoids (`_atm_land_transport_1d`, `_midlat_storm_bonus_1d`,
    `_handoff_bonus_1d`) are pure functions of |latitude|, so every land cell in
    a row gets the identical winter bonus and the model has no maritime
    moderation gradient at all. C1b previously read this as a single-signed
    "25-45N winters are 8-10 K too warm", measured against
    `EARTH_LAND_CYCLE_REFERENCE`'s mid-continental station anchors. Those anchors
    do not describe the population that metric averages over (see its own note),
    and the anchor-free `koppen_temperature_thresholds` shows the real defect has
    **both signs inside one latitude zone**: at 40-50N, 22% of reference land is
    too warm and 15% too cold, split by continentality --

    - 35-45N reference-**D** (continental) land: **94.8%** has a coldest month
      above the -3 C its Koppen class requires;
    - 45-55N reference-**C** (maritime) land: **99.5%** is below it.

    No latitude-only term can fix an error whose sign flips within a row, which
    is why five previous knobs aimed at the band mean all traded one error for
    another.

    Mean-preserving by construction: the zonal-mean winter level was calibrated
    by C1's `_handoff_bonus_1d` work and the trapezoids' own tuning, and this
    knob exists to add contrast, not to reopen that. The anomaly is divided by
    its own land-area-weighted spread, so this number keeps its meaning when the
    shape knobs (`land_transport_maritime_km`, `land_transport_upwind_ratio`)
    change the field. Applied with a winter weight -- see `simulate.py` -- since
    maritime moderation is a winter effect and applying it year-round warms
    maritime *summers*, which is the wrong sign.

    **History: this shipped inert at 0.0 on 2026-08-04 as a measured negative
    result, and that result was wrong.** It was rejected because "35-45N
    reference-D land is not more continental than reference-C land -- maritime
    proximity 0.312 vs 0.310". Those two numbers are reproducible, and they are
    a population artifact: they were measured by re-deriving the Köppen
    reference *at the 32x64 coarse grid*, which leaves 35 C cells and 15 D cells
    for the entire band, while the metric they were aimed at
    (`koppen_temperature_thresholds`) scores the fine grid. On the population
    actually scored, the two separate by **0.84 sd**. Two fixes then made that
    separation reachable by the physics -- computing the proximity field at
    native resolution (`simulate._maritime_proximity_coarse`) and making it
    anisotropic (`land_transport_upwind_ratio`) -- and the sign of the result
    flipped. Swept at 128x256, winter-gated, `upwind_ratio=32`:

    | decay | H10 group | H10 class | kappa | Tcold  | Twarm  | 40-50N |
    |-------|-----------|-----------|-------|--------|--------|--------|
    | 0.0   | 0.7082    | 0.4223    | 0.6303| 0.8646 | 0.7138 | 0.6286 |
    | 0.7   | 0.7179    | 0.4242    | 0.6427| 0.8843 | 0.7147 | 0.7581 |
    | **1.0** | **0.7176** | **0.4252** | **0.6424** | **0.8875** | **0.7142** | **0.7825** |
    | 1.4   | 0.7170    | 0.4245    | 0.6417| 0.8902 | 0.7141 | 0.8041 |

    1.0 sits at the group-accuracy/kappa knee; past it the coldest-month score
    keeps creeping up while group accuracy turns over, and the strength stops
    being physically interpretable (a cell 1 sd from its row's mean
    continentality already gets its whole winter bonus doubled or removed).

    **Confirmed at three resolutions** (process note 14), baseline -> shipped:

    | grid    | H10 group       | Tcold           | 40-50N zone     | Twarm  |
    |---------|-----------------|-----------------|-----------------|--------|
    | 64x128  | 0.6864 -> 0.6968| 0.8750 -> 0.8977| 0.5715 -> 0.7498| +0.0000|
    | 128x256 | 0.7082 -> 0.7176| 0.8646 -> 0.8875| 0.6286 -> 0.7825| +0.0004|
    | 256x512 | 0.7049 -> 0.7114| 0.8646 -> 0.8837| 0.6437 -> 0.7794| +0.0017|

    The gain is contrast, not level: reference-C and reference-D coldest-month
    accuracy improve *simultaneously* (128x256: C 0.7352 -> 0.7719, D 0.8693 ->
    0.9087), which a shift in the band mean cannot do.

    **Real, accepted cost**: `reference_error_score` 0.1508 -> 0.1526 at
    128x256 (+0.0058 at 64x128), traceable entirely to the 40-50N and 50-60N
    zonal-mean temperature bias rising 0.10 K and 0.18 K. The mechanism is
    mean-preserving in the *forcing*, but land temperature is a nonlinear
    function of it (evapotranspiration cooling and `_land_cap_1d` both compress
    the warm side), so redistributing the forcing does not preserve the mean
    temperature exactly. Both bands were already warm-biased, so this adds to an
    existing error rather than creating a new one -- and per process note 10,
    `reference_error_score` carries no biome information at all.

    Band x reference-group coldest-month agreement, 128x256, baseline ->
    shipped (this cross-tab is the view the defect was localized with; the
    metric reports `by_zone` and `by_reference_group` separately):

    | band   | grp | accuracy        | model coldest month |
    |--------|-----|-----------------|---------------------|
    | 25-35N | A   | 0.125 -> 0.502  | 17.11 -> 18.01 C    |
    | 25-35N | D   | 0.092 -> 0.229  |  4.21 ->  1.55 C    |
    | 35-45N | C   | 0.998 -> 0.990  |  4.37 ->  5.79 C    |
    | 35-45N | D   | 0.050 -> 0.433  |  1.17 -> -1.50 C    |
    | 45-55N | C   | 0.005 -> 0.292  | -8.65 -> -4.56 C    |
    | 45-55N | D   | 0.983 -> 0.967  |-10.44 ->-12.37 C    |
    | 55-65N | C   | 0.011 -> 0.011  |-14.03 -> -9.93 C    |

    The two headline defects both improve sharply: 35-45N reference-D goes from
    95.0% to 56.7% too warm, and 45-55N reference-C from 99.5% to 70.8% too
    cold. That second row is also **C1's long-standing mid-latitude winter cold
    residual**, which no latitude-only term had moved.

    **Still open, and neither defect is closed.** 57% of 35-45N reference-D land
    is still too warm and 71% of 45-55N reference-C still too cold. 55-65N
    reference-C is the one place the mechanism does not reach at all: it warms
    those cells 4 K (-14.03 -> -9.93) and their accuracy does not move, because
    they need -3 C. Coastal Norway and southern Alaska are ~1 cell wide at this
    resolution, so the row's land is overwhelmingly Siberian and Canadian and a
    row-mean-preserving redistribution has almost nothing to give them. The
    contrast is now present and correctly signed; it is not yet large enough,
    and at 55-65N the limit is geometric rather than a matter of strength."""

    land_transport_maritime_km: float = 1200.0
    """e-folding distance [km] for `land_transport_maritime_decay`'s maritime
    proximity field. Inert while that knob is 0.0.

    1200 km is roughly the depth to which winter maritime air masses moderate
    real continents -- the Cfb/Dfb transition across Europe sits near it, and it
    is the scale at which January isotherms turn from zonal to meridional over
    Eurasia and North America. Distances are physical, not in grid cells, so the
    field is resolution-invariant (the trap `atmosphere.py`'s monsoon inland mask
    had to be fixed for once already)."""

    land_transport_upwind_ratio: float = 32.0
    """Anisotropy of the maritime-proximity field: how much further ocean to a
    cell's **west** reaches than ocean in any other direction. 1.0 is isotropic.
    Inert while `land_transport_maritime_decay` is 0.0.

    Midlatitude flow is westerly in both hemispheres, so the ocean that moderates
    a winter continent is the one upwind of it. An isotropic distance-to-water
    field cannot express that: New York (Dfa, coldest month -3 C) has open water
    100 km east and 4000 km of continent west, while Lisbon (Csb, +11 C) at the
    same latitude has the ocean upwind -- yet both sit ~100 km from the sea.

    Measured on the population `koppen_temperature_thresholds` actually scores
    (128x256 reference-classified land), as the separation between 35-45N
    reference-C and reference-D land in pooled standard deviations:

    | field                                    | 35-45N | 45-55N |
    |------------------------------------------|--------|--------|
    | isotropic, coarse any-land mask (was)    | 0.37   | 0.74   |
    | isotropic, native resolution             | 0.83   | 0.99   |
    | **upwind x32, native resolution (shipped)** | **1.02** | **1.21** |
    | upwind x32, uncoarsened 128x256 (ceiling)| 1.00   | 1.21   |

    The shipped coarse field matches the uncoarsened one, so once the mask and
    the anisotropy are right the coarse forcing grid costs nothing here.

    **This is the knob that decides whether the mechanism helps at all.** Held
    at 1.0 (isotropic) with everything else shipped, H10 group accuracy is
    *below* baseline at every strength (0.7053-0.7065 against 0.7082) --
    reproducing the 2026-08-04 negative result. It only turns positive once the
    field is anisotropic, and improves monotonically to a plateau at ~24-32:

    | upwind | 1.0    | 4.0    | 8.0    | 16.0   | 24.0   | **32.0** | 48.0   |
    |--------|--------|--------|--------|--------|--------|--------|--------|
    | group  | 0.7053 | 0.7095 | 0.7135 | 0.7157 | 0.7167 | **0.7166** | 0.7171 |

    (measured ungated at decay 1.0; the winter-gated defaults peak the same
    way.) At 32 the upwind e-folding is ~38400 km on Earth, comparable to the
    circumference -- that is the finding, not a fitting artifact: **westward
    ocean influence should not decay on any scale smaller than a continent.**
    An exponential with the 1200 km isotropic length saturates 2000 km inland
    and then reports Warsaw, Moscow and Novosibirsk as equally continental,
    when Earth's January isotherms keep falling across all of it. Values past
    ~32 are on the plateau; the exact number is not critical.

    See `simulate._maritime_proximity` for the implementation and
    `_maritime_proximity_coarse` for why the mask matters as much as the
    anisotropy does."""

    land_transport_deficit_k: float = 0.0
    """Width [K] of the temperature-deficit gate on land heat transport in
    `simulate._evolve_temperature`. 0.0 disables the gate and reproduces the flat
    trapezoids bit-for-bit; a positive value scales them by
    `clip((273.15 - T_pre) / land_transport_deficit_k, 0, 1)`.

    **Why this exists rather than another seasonal knob** (2026-08-03, audit
    C1b). `land_transport_seasonality` below was built on the correct diagnosis
    -- the three trapezoids are sized for winter and applied year-round -- but on
    the wrong quantity. Measured offline at 41.4 deg, the forcing reaching
    `_land_cap_1d` has an **annual mean of 32 C against Earth's ~10 C**, and:

    - a first-order relaxation preserves the annual mean *exactly*, so no amount
      of land thermal inertia can touch it (swept: at tau = 120 days the summer
      target is still 48 C and the clamp still binds on 59% of the cycle);
    - `land_transport_seasonality` also preserves it, because its `summer_signal`
      averages to zero over an orbit -- it redistributes winter/summer without
      lowering the mean.

    So both existing knobs are amplitude-side levers aimed at a **level** error,
    which is why sweeping them improved shape and share metrics while H10 bounded
    accuracy fell: they were trading one error for another. A deficit gate is the
    first lever here that changes the mean.

    It is also the physically-shaped one. Eddy heat flux scales with the
    meridional temperature gradient *and damps it*, so the mechanism is
    self-limiting -- full strength into a cold winter continent, nothing into a
    warm summer one -- with no prescribed seasonal schedule at all. Gating on the
    cell's own pre-bonus temperature reproduces that, and does it **per cell**
    rather than per latitude row, so a cold continental interior draws more
    transport than a mild maritime cell at the same latitude. That is the right
    sign for continentality, which the latitude-only trapezoids cannot express.

    The reference is the freezing point (`simulate._LAND_TRANSPORT_DEFICIT_REF_K`)
    rather than a fitted constant, so this parameter controls only the gate's
    width. Measured mean bonus, flat vs gated at D = 20 K: 30 deg 10.8 -> 1.2 K,
    41.4 deg 26.2 -> 8.8 K, 50 deg 23.4 -> 10.2 K, 60 deg 18.9 -> 10.0 K -- the
    winter peak is preserved everywhere the trapezoids were calibrated on it
    (C1's -37 C coldest-month fix), while the summer contribution goes to zero."""

    land_transport_deficit_gain: float = 1.0
    """Multiplier on the deficit-gated land heat transport. Inert unless
    `land_transport_deficit_k > 0`; 1.0 keeps the trapezoids' calibrated
    magnitudes.

    This is the affordance the gate exists to create, and it is meaningless
    without it. Each of the three trapezoids was capped by the same constraint --
    raising it pushed the *summer* target past anything physical and forced
    `_land_cap_1d` to clamp harder -- so their peaks are lower bounds set by a
    side effect, not tuned optima. Once the gate makes summer transport
    identically zero, the winter magnitude can be raised on its own merits.

    That matters because two independent instruments agree the model's 45-55 deg
    winter is still too cold: the land-cycle metric reads its coldest month 4 K
    below Earth's, and H10's group shares read C 4.0pp under and D 3.3pp over --
    the two largest group errors, near equal-and-opposite, which is what a
    coldest-month bias across the C/D boundary looks like."""

    land_seasonal_amplitude: float = 0.75
    """Fraction of the radiative seasonal swing the land surface actually feels
    [dimensionless]. `1.0` is an exact no-op reproducing the historical
    behaviour; smaller values contract the land forcing's seasonal anomaly
    toward its own annual mean.

    **This is the term the land branch never had.** `temperature_kelvin_for_lat`
    returns instantaneous local radiative equilibrium -- no surface heat
    capacity, no dynamical damping. Its annual half-range at 41.4 deg is ~81 K
    against Earth land's ~28 K. The ocean branch of `_evolve_temperature` has
    damped its own swing since the model's early days (`ocean_seasonal_frac`,
    the identical mean-preserving form); land ran with an implicit thermal
    inertia of exactly zero.

    Everything else in the land stack is a one-sided patch on that one error:
    the three transport trapezoids lift a -33 C radiative winter, and
    `_land_cap_1d` cuts a +41 C radiative summer. They do not cancel, because
    the trapezoids are added in all twelve months while the cap only subtracts
    in summer -- which is how the forcing ends up with an annual mean 21 K too
    warm and a flat top across seven months (audit C1b's square wave).

    Applied as `mean + k * (instantaneous - mean)`, so it is **exactly
    mean-preserving**: it cannot move the annual-mean level that the trapezoids,
    the cap and C1's handoff work were all calibrated against, only the swing
    about it. That is the property four earlier C1b knobs lacked -- they were
    amplitude-side levers aimed at a level error and could only trade one for
    the other. This one is the reverse, and is meant to be set together with
    `land_transport_gain`, which takes the level error out once the swing no
    longer needs patching.

    Calibrated jointly with that knob on the tracked benchmark; see
    `docs/ACCURACY_AUDIT.md` C1b for the 2-D sweep."""

    land_seasonal_amplitude_maritime: float = 0.45
    """Continentality contrast on `land_seasonal_amplitude`, in units of
    fraction of the row's damping per standard deviation of continentality.
    `0.0` is an exact no-op; positive values damp maritime land's seasonal
    cycle further and let continental land keep more of its swing.

    `land_seasonal_amplitude` sets the zonal *level* of the damping; this sets
    its contrast within a row, and the contrast is where the physical content
    is. A maritime climate **is** a damped seasonal cycle -- that is the
    definition of one, not a downstream consequence -- so the continentality
    field `land_transport_maritime_decay` already builds (upwind-weighted
    maritime proximity) is the correct modulator, shared rather than rebuilt.

    **This is the term that mechanism's winter gate exists to work around.**
    `land_transport_maritime_decay` scales an additive *bonus*, so applied
    year-round it warms maritime summers -- the wrong sign -- and had to be
    restricted to the winter half. It therefore cannot reach the model's
    maritime-*summer* error at all, which is large: at 128x256, 81.5% of
    -50:-40 and 48.7% of -40:-30 reference-C land (Chile, New Zealand,
    Tasmania) has a warmest month above the 22 C its class requires. An
    amplitude damping has the right sign in both seasons from one term -- warmer
    maritime winters, cooler maritime summers, annual mean untouched -- so it
    needs no seasonal gate.

    Shares `_maritime_transport_factor`'s row-mean preservation, global spread
    normalization and [-1, 1] bound, so amplitude stays in [0, 2x] and this knob
    keeps its meaning when the field's shape knobs move."""

    land_transport_gain: float = 0.5
    """Uniform multiplier on the three land heat-transport trapezoids in
    `simulate._evolve_temperature` [dimensionless]. `1.0` is an exact no-op.

    Scales `_atm_land_transport_1d`, `_midlat_storm_bonus_1d` and
    `_handoff_bonus_1d` together, so the latitude *shape* C1's handoff work
    calibrated is preserved exactly and only the magnitude moves. Unlike
    `land_transport_deficit_gain` this is unconditional -- that one is inert
    without its gate.

    Meaningless on its own, and deliberately so: all three trapezoids were sized
    against an **undamped** radiative winter (27 K at 41 deg to lift a -33 C
    January, 22 K at 65 deg to keep Antarctic winter off the 200 K floor).
    `land_seasonal_amplitude` shrinks that deficit by construction, at which
    point the calibrated peaks are an overshoot applied in all twelve months --
    the annual-mean half of C1b's defect. Lowering this is what removes it."""

    land_thermal_inertia_days: float = 0.0
    """Land surface relaxation time constant [days] toward the seasonal baseline
    `T_base_land`, in `simulate._evolve_temperature`. 0.0 keeps the historical
    fixed blend and is an exact no-op.

    **The historical blend is a fraction per CALL, not per day**: `land_blend =
    0.2` regardless of how long the step is. This is the land's only thermal
    inertia, so the effective time constant -- and with it the entire amplitude
    of the land annual cycle -- is set by whatever step length the caller
    happens to use. Over a 12-day span it keeps **0.800** of the prior
    temperature integrated as one step against **0.069** integrated as twelve,
    from identical physics. Same defect class as the monsoon inland mask's fixed
    20-cell reach: a physical scale expressed in units of the discretisation.

    A positive value uses `1 - exp(-dt/tau)`, so *this term* becomes split
    invariant exactly: at tau = 27 days a 12-day span retains 0.6412 of the prior
    temperature whether it is integrated in one step or twelve, against 0.800 vs
    0.069 for the fixed blend.

    **It does not make `simulate_step` step-length invariant end to end, and that
    was measured rather than assumed.** On the same 12-day split the mean land
    temperature discrepancy is 4.34 K with the fixed blend and **5.44 K** at
    tau = 27 -- it goes *up*, because the residual is dominated by other terms
    that scale linearly in `days` (advection, diffusion, the evaporation budget)
    and changing this rate changes how much of that shows through. So this is a
    correctness fix for one term, not a remedy for the MONTHLY-vs-DAILY
    differences the wind and precip substep gates address. See
    `testing/test_land_seasonal_cycle.py`, which pins both halves."""

    enable_land_surface_energy: bool = True
    """Enable the calibrated prognostic land surface-energy tendency.

    The existing land branch relaxes toward a prescribed seasonal baseline.
    This closure additionally integrates net radiation minus bulk sensible and
    latent heat fluxes through a finite land heat capacity.  The deliberately
    small Earth-default strength cleared the compact and 128x256 five-year CRU
    gates: temperature RMSE improved while precipitation and both Köppen
    accuracy measures were preserved.  Set this to ``False`` for a strict
    legacy-path A/B comparison.
    """

    land_surface_heat_capacity_j_m2_k: float = 1_500_000.0
    """Effective active land-layer heat capacity [J m-2 K-1] when enabled."""

    land_surface_energy_strength: float = 0.001
    """Dimensionless multiplier for the calibrated land-energy tendency.

    0.001 is the largest compact-CRU-screened strength that improved
    temperature RMSE without regressing either Köppen accuracy metric; it
    also passed the 128x256 five-year promotion check.  Larger strengths cool
    the land further but move cells across Köppen boundaries in the wrong
    direction, so remain opt-in experiment values.
    """

    enable_force_restore_land: bool = False
    """Use the gated two-reservoir force-restore land replacement path.

    Unlike ``enable_land_surface_energy``, this branch does *not* blend land
    toward the legacy latitude-only seasonal baseline and therefore does not
    apply the historical seasonal-amplitude stack or ``_land_cap_1d`` to the
    land surface.  It is deliberately off until it passes regional CRU and
    Köppen promotion gates.
    """

    enable_force_restore_atmospheric_heat_convergence: bool = False
    """Feed resolved atmospheric heat convergence to force-restore land.

    This experimental Phase 3 gate diagnoses the exact temperature increment
    produced by the supported atmosphere's horizontal advection and diffusion
    operators, converts that increment with the represented column heat
    capacity, and supplies it as a surface-energy forcing.  It has no effect
    unless ``enable_force_restore_land`` is also true.
    """

    enable_force_restore_conservative_land_air_exchange: bool = False
    """Use explicit equal-and-opposite sensible exchange in force-restore land.

    When enabled with ``enable_force_restore_land``, the replacement branch
    disables its empirical land-side air-temperature relaxation and instead
    adds the Penman--Monteith surface sensible flux to the represented
    atmospheric column using ``(p_s/g) cp``. Ocean relaxation is unchanged.
    """

    enable_force_restore_boundary_layer: bool = False
    """Use a distinct prognostic land atmospheric mixed layer.

    This experimental gate requires ``enable_force_restore_land``.  Penman--
    Monteith uses the mixed-layer temperature, surface sensible heat enters
    that reservoir, and conservative entrainment exchanges energy with the
    horizontally transported free atmosphere.  Horizontal mixed-layer
    transport is controlled independently for fixed-versus-transported A/Bs.
    """

    enable_boundary_layer_horizontal_transport: bool = False
    """Advect mixed-layer mass and heat with conservative finite-volume fluxes.

    Divergence of the prescribed wind is closed by conservative entrainment or
    detrainment with the free atmosphere, keeping the hydrostatic layer pressure
    thickness fixed.  This gate requires the force-restore boundary layer.
    """

    enable_boundary_layer_capacity_aware_airsea_exchange: bool = False
    """Close ocean/atmosphere heat exchange with physical heat capacities.

    This experimental gate requires the force-restore boundary layer. It keeps
    the existing atmospheric temperature tendencies over ocean but applies
    their equal-and-opposite energy to the slab ocean using its physical mixed-
    layer heat capacity. The supported/default coupling is unchanged when off.
    """

    enable_boundary_layer_capacity_aware_free_air_transport: bool = False
    """Conserve resolved free-air transport with its residual heat capacity.

    This experimental gate requires the force-restore boundary layer. It
    removes the impossible global source or sink from the combined supported
    advection/diffusion update after accounting for the mixed-layer mass split.
    """

    enable_boundary_layer_near_surface_cloud_temperature: bool = False
    """Use mixed-layer temperature for land surface-pressure cloud RH.

    This experimental gate requires the force-restore boundary layer. It only
    changes the low-cloud saturation diagnostic over land; column precipitation
    and high-cloud temperature weighting continue to use free-atmosphere state.
    """

    enable_boundary_layer_split_invariant_cloud_memory: bool = False
    """Make cloud persistence and rainout invariant to mixed-layer substeps.

    This experimental gate requires the force-restore boundary layer. Daily
    survival fractions are exponentiated by the substep duration, preventing
    four six-hour calls from applying four full days of cloud rainout.
    """

    boundary_layer_mixed_depth_m: float = 1_000.0
    """Dry mixed-layer geometric depth [m], used to derive pressure mass."""

    boundary_layer_reference_temperature_k: float = 288.15
    """Isothermal hydrostatic reference temperature [K] for layer pressure mass.

    Used only to convert geometric boundary-layer depths to pressure
    thicknesses.  It is a planet parameter so non-Earth configurations do not
    inherit Earth's mean surface temperature.
    """

    boundary_layer_entrainment_velocity_m_s: float = 0.005
    """Mixed-layer/free-air entrainment velocity [m s-1].

    The exchange conductance is ``rho * cp * w_e``; this is a physical mass-
    exchange velocity rather than an unconstrained temperature relaxation.
    """

    enable_boundary_layer_stability_dependent_exchange: bool = False
    """Suppress mixed-layer entrainment across stable inversions.

    The experimental closure diagnoses the bulk Richardson number and limits
    entrainment with the stable Businger--Dyer ``1 + 5 Ri`` form and a friction
    velocity from the logarithmic wind law.  It preserves fixed entrainment as
    an explicit control and has no effect outside the boundary-layer branch.
    """

    enable_boundary_layer_interface_reservoir: bool = False
    """Use a prognostic inversion-top layer for stability-dependent exchange.

    The interface layer occupies the hydrostatic slab immediately above the
    mixed layer with the same geometric depth. It therefore adds no independent
    thickness calibration parameter and conserves energy with both adjacent
    atmospheric reservoirs. Requires the force-restore boundary layer.
    """

    land_force_restore_days: float = 30.0
    """Surface-to-deep-soil restore time scale [days] in the gated land path."""

    land_deep_heat_capacity_j_m2_k: float = 12_000_000.0
    """Heat capacity [J m-2 K-1] of the gated deep-soil reservoir."""

    land_surface_resistance_min_s_m: float = 70.0
    """Wet-surface resistance [s m-1] for Penman--Monteith evaporation."""

    land_surface_resistance_dry_s_m: float = 2_000.0
    """Additional dry-soil resistance [s m-1] in the gated land path."""

    land_transport_seasonality: float = 0.0
    """Seasonal modulation of atmospheric heat transport into land
    [dimensionless, 0-1]. `0.0` is an exact no-op reproducing the previous
    season-independent behaviour; `1.0` means the bonus doubles at the local
    winter solstice and falls to zero at the local summer solstice.

    `simulate._evolve_temperature` warms land with three latitude-only
    trapezoids (`_atm_land_transport_1d`, `_midlat_storm_bonus_1d`,
    `_handoff_bonus_1d`) that stand in for eddy/storm-track/Ferrel heat
    transport. Every one of them is motivated in its own comment by a *winter*
    deficit ("Antarctic winter equilibrium", "winter cyclones", "coldest-month
    means of -37 to -39 C") -- yet all three are applied at full strength year
    round, with no `day_of_year` dependence at all. That is not what the real
    mechanism does: eddy heat flux scales with the meridional temperature
    gradient, which is roughly 2-3x larger in winter than in summer, so real
    transport into midlatitude land is strongly winter-weighted.

    Applying a constant is what forces `_land_cap_1d` to bind as hard as it
    does, and that clamp -- not the cap's own existence -- is what squares off
    the land seasonal cycle. Traced stage by stage at 41.4 N (zonal, soil 0.55):

        radiative only      -33.0 -12.6   9.2  26.3  36.9  40.9 ... (a sinusoid)
        + transport bonus    -6.7  13.6  35.4  52.5  63.2  67.1 ... (+26 K flat)
        + evap cooling       -6.7  13.6  35.4  45.3  45.3  43.7 ...
        + land cap           -6.7  13.6  27.9  27.9  27.9  27.9 ... (7 months
                                                                     identical)

    The bonus is sized for winter but added in summer too, lifting the summer
    target to a physically impossible 67 C; the evapotranspiration cooling below
    removes part of it (to ~44 C) but nowhere near enough, so `_land_cap_1d`
    hard-clamps seven consecutive months to exactly its own ceiling value. The
    cap binds on **55.7% of (month, row) pairs at 25-50 deg** -- its docstring
    calling it a "rarely-binding safety net" is measurably wrong -- and its
    25.7-27.9 C range at those latitudes is exactly the 26-29 C window that
    42-45% of midlatitude land's warmest month falls into.

    Consequence: land at 25-50 deg spends 7.0-8.0 months above its own annual
    mean (a sinusoid gives 6.0; ocean at the same latitudes gives 6.3). That
    shape makes Koppen's Cfc unreachable (needs <4 months >10 C) and pushes
    maritime midlatitude cells out of the C group into D -- a direct cause of
    the model emitting zero Csa/Csb/Cfc.

    This scales the three trapezoids by `1 - land_transport_seasonality *
    summer_signal`, where `summer_signal` runs +1 at the local summer solstice
    to -1 at the local winter solstice (the same solar-declination signal the
    evapotranspiration cooling uses). Cutting the *summer* bonus is the operative
    half: it lowers the summer target back under the cap so the clamp stops
    binding and the underlying sinusoid shows through. Raising the winter half
    is a bonus that also narrows the amplitude excess -- but note that a
    winter-only variant of this term was measured first and moved squareness
    only 7.00 -> 6.99, so the winter trough is *not* where the square wave comes
    from. Do not re-derive this the other way round.

    See ACCURACY_AUDIT.md C1 (mid-latitude winter cold bias) and the
    missing-Koppen-classes root cause."""

    land_cap_softness_k: float = 0.0
    """Softening width [K] for `simulate._evolve_temperature`'s summer land
    temperature ceiling `_land_cap_1d`. `0.0` is an exact no-op reproducing the
    previous hard `np.minimum` clamp.

    The hard clamp is what physically writes the plateau into the land seasonal
    cycle: because `np.minimum` maps every overshooting month onto the *same*
    ceiling value, seven consecutive months at 41 N come out bit-identical (see
    `land_transport_seasonality`). This replaces it with a smooth soft-min,

        y = c - w * log(1 + exp((c - x) / w))

    which is `y ~ x` well below the ceiling, asymptotes to `c` well above it, and
    is *strictly monotonic* in `x` throughout -- so months that differ before the
    clamp still differ after it, and the annual cycle keeps its ordering and
    curvature instead of being flat-topped. `w` sets how far below the ceiling
    the softening starts to matter.

    Note this alone cannot rescue a large overshoot: the soft-min still converges
    to the ceiling, so a month sitting 16 K above it lands within 0.02 K of the
    hard-clamped value. It only preserves shape once the overshoot has been
    brought down to the same order as `w` -- which is what
    `land_transport_seasonality` and `evap_cooling_strength` are for. The three
    are meant to be set together.

    **That precondition is now met for the first time** (measured 2026-08-05,
    audit C1b-EVAP). The 2026-08-05 amplitude work collapsed the forcing's peak
    overshoot against this ceiling from **+11.54 K to -0.19 K** at 35-45N (and
    +5.21 -> +1.08 at 25-35N), so the regime this knob has always required now
    exists. Swept there it improves `cycle_error_score`, both Koppen threshold
    accuracies, and produces the project's first nonzero Cfc -- but costs H10
    group accuracy above w ~= 2, because the soft-min sits strictly *below* the
    hard clamp (`y(c) = c - w*log 2`) and so carries a systematic cool bias of
    order `w`. Anyone taking this up should compensate that level shift rather
    than sweep the width alone."""

    evap_cooling_strength: float = 1.0
    """Multiplier on `simulate._evolve_temperature`'s evapotranspiration cooling
    coefficient (`_EVAP_COOL_COEFF_MAX`, 0.85) [dimensionless]. `1.0` is an exact
    no-op.

    Raising this deepens the contraction of summer land temperature toward the
    `evap_cooling_threshold_k` reference, lowering the summer target so the
    `_land_cap_1d` ceiling stops binding. The mechanism stays soil-moisture-aware,
    so deserts (low `soil_moisture`) are affected far less than moist continental
    interiors -- which is both the physically correct behaviour and the reason
    this cannot on its own flatten the desert side of the problem.

    **This knob does nothing until it is large, and the reason is absorption**
    (measured 2026-08-05, audit C1b-EVAP). `_land_cap_1d` is applied on the very
    next line, and `min(T - cooling, cap)` is independent of `cooling` while the
    result is still above the cap -- so **99.4% of this term never reaches the
    output**, and values between 0.0 and ~1.4 are indistinguishable end to end.
    Only once the contraction pulls the forcing *below* the cap does the knob
    become live, which is why its response is a threshold rather than a ramp.
    Do not read a null result here as the mechanism being weak.

    **Above ~1.18 it used to invert.** The removed fraction
    `season * evap_cooling_coeff * strength * soil` is now clipped to [0, 1] in
    `simulate._evap_cooling_fraction`; before that guard, exceeding 1 removed more
    than the whole excess and a hotter cell came out colder than a cooler one.
    Sweeps of this knob recorded before 2026-08-05 that went past ~1.18 were
    measuring that inversion, not a stronger contraction.

    Retains the property that the cooling only removes energy above the
    threshold, so it can never drive land below that reference."""

    evap_cooling_season_width: float = 1.0
    """Width of the seasonal ramp gating evapotranspiration cooling
    [dimensionless, (0, 1]]. `1.0` is an exact no-op reproducing the previous
    behaviour.

    `simulate._evolve_temperature`'s evapotranspiration cooling is
    `summer_factor * 0.85 * soil * max(T - 290, 0)`, where `summer_factor` runs
    0 -> 1 from equinox to local summer solstice. Rearranged, that is a
    contraction of land temperature toward 290 K:

        T <- 290 + (1 - 0.85 * soil * summer_factor) * (T - 290)

    The contraction *strength* therefore rises and falls with the season while
    the quantity being contracted, `(T - 290)`, already peaks in summer. Cooling
    thus carries two powers of the seasonal cycle and bites hardest exactly at
    the peak -- a differential peak-flattening operator. A contraction with a
    *constant* factor is linear and so shape-preserving (it rescales amplitude
    above the threshold but leaves a sinusoid a sinusoid); only a
    seasonally-varying factor squares off the top of the cycle. Measured: this
    is what puts 42-45% of midlatitude land's warmest month into a single
    26-29 C window and holds land at ~7.0-8.0 months above its own annual mean
    against ocean's 6.3 (a sinusoid gives 6.0). Raising winter temperatures does
    *not* fix it -- a full sweep of `land_winter_transport_boost` moved squareness
    only 7.00 -> 6.99 -- because the defect is in the summer half.

    This divides `summer_factor` by the field before clipping back to [0, 1], so
    the gate saturates once the season is `evap_cooling_season_width` of the way
    to the solstice and is flat from there on. Cooling then acts as a
    near-constant contraction across the warm season -- still exactly zero in
    winter (the `max(T - 290, 0)` term and the ramp both vanish), still
    soil-moisture-aware, so deserts stay hot and the mechanism from
    `evapotranspiration-cooling-fix` is preserved -- while no longer flattening
    the peak it was only ever meant to lower. Smaller values saturate earlier
    and are more shape-preserving; the cost is a cooler shoulder season, so this
    trades against the summer land-warmer-than-ocean sign `_land_cap_1d`
    protects and should not be pushed to 0."""

    evap_cooling_amplitude: float = 0.0
    """Evapotranspirative damping of the land seasonal amplitude, in units of
    fraction of amplitude per standard deviation of soil moisture
    [dimensionless]. `0.0` is an exact no-op.

    The shape-only sibling of `evap_cooling_strength`, and built because that
    knob is a *level* mechanism (audit C1b-EVAP). Contracting land toward a
    fixed `evap_cooling_threshold_k` subtracts from a cell's annual mean, which
    is what the mid-latitudes want and what the wet tropics emphatically do not:
    strengthening it costs 2.05pp of Koppen group accuracy in 0:10 and 1.08pp in
    -20:-10 (cooling rainforest land under the 18 C A-boundary) while gaining
    2.92pp on the US Midwest.

    Damping the *amplitude* by the same soil-moisture field is the same physics
    with the level removed: latent heat flux buffers a moist surface's seasonal
    swing and leaves a dry one's unbuffered. It is exactly mean-preserving in
    time, so it cannot move any cell's annual mean, and a cell with no seasonal
    cycle has no amplitude to damp -- making it inert by construction precisely
    where the threshold form does its damage.

    Applied through the same row-mean-preserving, globally spread-normalized,
    `[-1, 1]`-bounded factor as `land_seasonal_amplitude_maritime`, so
    `land_seasonal_amplitude` keeps meaning the row's mean damping and this knob
    only adds within-row contrast. Soil moisture is a genuinely different
    discriminator from continentality rather than a proxy for it -- the Sahel is
    dry without being continental."""

    evap_cooling_threshold_k: float = 290.0
    """Reference temperature the evapotranspiration cooling contracts land
    toward, in `simulate._evolve_temperature`'s forcing [K]. `290.0` reproduces
    the historical hardcoded constant exactly.

    **This is the term's reach, and it was the reason the mechanism could not
    touch the model's largest remaining temperature defect** (measured
    2026-08-05, audit C1b-EVAP). The cooling is `... * max(T - threshold, 0)`,
    so at 290 K (16.85 C) it is identically zero on any land whose forcing stays
    below that -- which is all of the sub-polar and Southern-Hemisphere
    mid-latitude land where the warmest-month error is concentrated. Scored
    against the Koppen reference's own bounds at 128x256, warmest month is the
    model's weak metric (0.760 against coldest month's 0.900) and the error is
    two-thirds one-signed: 17.3% of scorable land too warm against 6.7% too
    cold, rising to **39.1% too warm over reference-E land** and 0.76-0.99 of
    the -70:-30 zones. None of that land ever reaches 290 K, so no setting of
    `evap_cooling_strength` could reach it either.

    Real evapotranspiration has no activation temperature; it proceeds wherever
    energy and soil moisture are both available, which is exactly the regime
    those zones are in. The threshold is a tuning constant from the term's
    original subtropical purpose, not a physical bound, and lowering it extends
    the mechanism to the land that needs it while leaving the moisture gate
    (which is the part that keeps deserts hot) intact.

    Also a hardcoded Earth constant on a non-Earth code path (audit C3): 290 K
    means nothing on Mars, where it sits ~80 K above any surface temperature the
    model produces and silently disables the term."""

    evap_cooling_coeff: float = 0.85
    """Peak fraction of the above-threshold excess removed per unit soil
    moisture per step, in `simulate._evolve_temperature` [dimensionless, >=0].
    `0.85` reproduces the historical hardcoded `_EVAP_COOL_COEFF_MAX` exactly.

    The applied contraction is
    `clip(season * evap_cooling_coeff * evap_cooling_strength * soil, 0, 1)`.
    **The clip is a correctness guard, not a tuning choice** (added 2026-08-05):
    without it the product exceeds 1 once
    `evap_cooling_coeff * evap_cooling_strength > 1/soil`, at which point the
    term removes more than the whole excess and *inverts* -- a cell hotter than
    the threshold comes out colder than one at it, and the mapping stops being
    monotonic in temperature. At the shipped defaults the product peaks at 0.85
    so the clip never engages, but `evap_cooling_strength` above ~1.18 crossed
    that line silently, which made the knob's own documented sweep range
    partly meaningless. Audit process note 19's pattern: the bound and the
    invariant next to it have to be checked together."""

    ocean_transport_coeff: float = 0.3
    """Poleward ocean heat flux scale [dimensionless].
    Multiplier on the parameterised meridional ocean heat transport.
    0 = no transport (pure radiative equilibrium), 1 = maximum."""

    amoc_cutoff_lat: float = 80.0
    """Latitude above which the AMOC bonus tapers to zero [°N].
    Fixes the known NH pole over-warming artifact: the AMOC 18 K bonus
    was previously applied uniformly to 90°N.  Taper width is 10°, so
    the bonus reaches zero at (amoc_cutoff_lat + 10)°N.
    Default 80° reduces the 18 K Arctic bonus to 9 K at 75°N and 0 K at
    90°N, without disrupting the sub-polar warmth that anchors the ITCZ."""

    ice_albedo_strength: float = 0.30
    """Ice-albedo feedback magnitude [dimensionless, 0–1].
    0 = no ice-albedo effect; 1 = full sea-ice albedo contrast applied.
    Default 0.30 weakens runaway glaciation while preserving the signal."""

    pgf_continentality_amp: float = 0.0
    """Continental-interior amplification of `evolve_wind`'s thermal pressure
    term [dimensionless, >=0]. Real continental interiors develop stronger
    seasonal thermal lows/highs than coastal land or islands because they're
    far from the ocean's moderating heat capacity -- the same temperature
    anomaly should produce a proportionally larger pressure response inland.
    Locally scales the existing `-pgf_temp_scale * (T-273.15)/30` term by
    `(1 + pgf_continentality_amp * continentality)`, where `continentality`
    (masks.get_continentality) is a `[0,1]` distance-from-coast proxy, 0 at
    coast/ocean. Deliberately still anomaly-following (scales with the
    already-tuned T field, zero-inert when T is at its reference value) --
    NOT a flat additive land-sea bonus, which was tried and reverted
    (evolve_wind's own docstring/comments) because it gave Antarctica (all
    land, always cold) a permanent artificial high and crashed the SH pole
    via runaway katabatic/ice-albedo feedback. A cold interior still gets a
    *high* here, just a more strongly and correctly signed one (the Siberian
    High is real, not a bug), so it shouldn't reproduce that failure mode --
    but it touches the same high-latitude pressure/ice-feedback machinery
    that caused it, so re-verify ice-sensitivity tests after changing this.
    Default 0.0 (exact no-op) until calibrated against
    `scripts/check_real_terrain_koppen.py --wind-diagnostics`
    (known-physics-gaps.md item 3b)."""

    wind_terrain_pgf_scale: float = 1.0
    """Multiplier for the resolved-terrain pressure-gradient forcing in the
    prognostic surface-wind solver. ``1.0`` preserves the historical 900-Pa
    scale exactly; values below one are a gated diagnostic for excessive local
    terrain deflection overwhelming the Earth-like midlatitude westerly target.
    This does not change terrain height, drag, or precipitation directly."""

    # ------------------------------------------------------------------ #
    # Ocean circulation — AMOC / ACC bonus magnitudes
    # ------------------------------------------------------------------ #
    amoc_bonus_near: float = 3.0
    """AMOC warming at the start of the NH sub-polar ramp (42–65°N) [K].
    Combined with amoc_bonus_far this produces the Gulf-Stream / thermohaline
    warming that keeps the North Atlantic 10–18 K warmer than radiative equilibrium."""

    amoc_bonus_far: float = 9.0
    """AMOC peak warming at 75°N+ [K].
    Reduced from 15 K to 9 K to partially close the known NH gradient gap
    (model runs 22 K, target 40–65 K) without disrupting the sub-polar warmth
    that anchors the ITCZ.  Use the amoc sweep script to find the optimal value."""

    acc_bonus_near: float = 8.0
    """ACC warming at the start of the SH sub-polar ramp (55–65°S) [K].
    Represents CDW upwelling warming south of the Polar Front."""

    acc_bonus_far: float = 20.0
    """ACC peak warming at 75°S+ [K]."""

    # ------------------------------------------------------------------ #
    # Ocean–atmosphere coupling
    # ------------------------------------------------------------------ #
    k_airsea: float = 0.001
    """Air-sea sensible heat exchange coefficient [day⁻¹].
    Controls how fast T_sst is pulled toward T_air.  At 0.001/day the
    ocean thermal timescale is ~1000 days (~3 yr), consistent with the
    mixed-layer depth of 50–100 m.  Values above ~0.003 drive T_sst below
    the freezing point at 55°N in winter, triggering ice-albedo runaway."""

    ocean_lag_days: float = 50.0
    """Ocean thermal lag [days].  SST responds to insolation with this delay,
    representing the heat capacity of the oceanic mixed layer.  Earth ≈ 50 days
    (phase lag of ~1.5 months).  Scaled by orbital period for non-Earth planets."""

    # ------------------------------------------------------------------ #
    # Latitude-dependent mixed-layer depth (Tier 1 item 6, FEATURES.md)
    # ------------------------------------------------------------------ #
    mixed_layer_depth_tropical_m: float = 30.0
    """Ocean mixed-layer depth at the equator [m] -- shallow tropical
    thermocline maintained by persistent trade-wind stratification. Also the
    base value `_evolve_temperature`'s T_sst relaxation step uses for its
    latitude-dependent thermal-inertia ramp; this field replaces what used to
    be a hardcoded literal there, so the Earth default reproduces prior
    behavior exactly."""

    mixed_layer_depth_polar_m: float = 200.0
    """Ocean mixed-layer depth at the poles [m] -- deep winter convective
    mixing. Interpolated with `mixed_layer_depth_tropical_m` by
    `(|lat|/90)**1.5`. Earth default reproduces the prior hardcoded ramp
    (30 + 170*(|lat|/90)**1.5) exactly."""

    derive_ocean_seasonal_lag: bool = False
    """If True, `ocean_seasonal_frac` (how much of the radiative seasonal
    swing reaches SST -- see simulate.py's `_ocean_seasonal_fraction`) is
    computed from `mixed_layer_depth_*_m` via the standard slab-ocean thermal
    relaxation response (ΔT/ΔT_rad = 1/sqrt(1+(2πτ/P)²), τ = ρ·cp·h/λ)
    instead of the legacy hand-tuned per-latitude polynomial.  Default False:
    the derived path is real and unit-tested but has not yet been checked
    against the real-terrain regression baseline -- flip on for a calibration
    pass, not by default, same convention as enable_surface_hydrology /
    enable_land_ice_dynamics."""

    ocean_thermal_relaxation_coefficient: float = 3.0
    """Effective radiative-restoring coefficient λ [W/m²/K], used only when
    `derive_ocean_seasonal_lag=True`. Order-of-magnitude Planck-response
    value (4σT³ at T≈255K); free parameter for a future calibration pass."""

    ekman_strength: float = 0.3
    """Scaling factor for Ekman wind-driven ocean current advection [0–1].
    0 = Ekman transport disabled; 1 = full wind-to-current scaling (3% of wind speed).
    At 0.3, coastal upwelling introduces realistic SST gradients at continental margins.
    Gated by has_liquid_water_ocean."""

    ocean_gyre_strength: float = 1.0
    """Scaling factor for the 2D barotropic gyre current contribution to ocean
    heat transport (`ocean.compute_gyre_currents`) [0-1, >=0 in principle].
    0 = disabled (exact no-op: skips the whole block, matching `ekman_strength`'s
    own guard pattern). Purely additive alongside (never replacing) the existing
    1D zonal-mean transport (`calculate_ocean_heat_transport`) and Ekman
    deflection (`ekman_strength`) -- gives ocean currents real east-west (gyre)
    structure that those two mechanisms can't produce on their own.

    **Calibrated 2026-08-02 (ACCURACY_AUDIT.md D5), default 0.0 -> 1.0.** It had
    shipped inert with no calibration history, flagged as the highest-risk item
    of the Jul 2026 backlog session, with two specific things to check first.
    Both were checked and both came back clean:

    - *"Check the SST pattern for plausible western-boundary-current structure
      rather than noise."* It is coherent structure, not noise: the grid-scale
      residual of the gyre-induced SST anomaly is 0.10 of its total standard
      deviation (a value near/above 1.0 would indicate grid-scale noise). The
      anomaly reproduces the correct gyre dipole in every region checked --
      warming on western boundaries (Gulf Stream +0.38 K, Kuroshio +0.14 K) and
      cooling on eastern boundaries (Canary -0.36 K, California -0.22 K,
      Benguela -0.54 K, Humboldt -0.25 K).
    - *"Re-verify guard-rail tests before raising this default."* Full suite
      green; `reference_error_score` improves 0.3226 -> 0.3209 on the tracked
      64x128 benchmark (and 0.3195 combined with the `land_west` sign fix below).

    The improvement is physically located rather than a global-cooling artifact:
    decomposing the score shows the precipitation half is flat (0.2811 ->
    0.2817) and essentially all the gain is in the temperature half,
    concentrated in the two ocean-dominated Southern-Hemisphere bands (0-10S
    0.048 -> 0.036, 40-50S 0.557 -> 0.524) while the large Northern-Hemisphere
    mid-latitude biases -- which are land/AMOC-driven (D2), not gyre-driven --
    barely move. Returns saturate around 1.0 (values to 3.0 keep improving the
    score only marginally, and this solve has no natural physical amplitude --
    it is clipped to +-0.5 m/s -- so 1.0, the unscaled solve, is preferred over
    chasing a metric that cannot distinguish "more correct" from "cools an
    already-too-warm model").

    **Known limitation, measured**: the eastern-boundary cooling above is real
    ocean physics that D3 lists as entirely missing, but it does *not* propagate
    to adjacent land precipitation -- Atacama precipitation is unchanged (146
    mm/yr) at every gyre strength from 0.0 to 3.0. This directly answers the
    doubt D3 itself raised ("it was never established that per-cell SST
    anomalies propagate to land climate at all in this model's simplified
    atmosphere"): they do not, at least not through this pathway. Enabling gyres
    is therefore *not* a route to fixing Atacama (A1's remaining desert
    residual).

    Caveat unchanged: the underlying streamfunction solve is periodic-in-x/
    DFT-in-y, tolerating no true meridional boundary condition -- same caveat as
    its atmosphere.py usage."""

    ferrel_v_centre_deg: float = 44.0
    """Latitude centre [deg] of the mid-latitude lobe of the prescribed
    *meridional* 3-cell surface profile -- `v_surface` in `generate_wind_field`
    (MONTHLY/ANNUAL) and `v_target` in `evolve_wind` (DAILY/WEEKLY). Both paths
    read this so every time-scale places the dry belt at the same latitude.

    **Changed 48.0 -> 44.0 on 2026-07-25** after root-causing the long-standing
    US Midwest dryness. Measured effects at 44 on 30yr real terrain: Sahara
    361->299, Kalahari 355->273, Atacama 161->154, Canadian Prairies 452->480,
    US Midwest 300->400, Central Europe 451->494 -- all six boxes toward their
    Earth values at once, and US Midwest clears the driest desert box for the
    first time. The zonal-mean divergence peak moves 38-45N -> 30-38N (Earth
    ~25-30N).

    Why 44 and not 40, which scores better on those six boxes (total error 621 vs
    962): 44 is the only tested value that ALSO improves the independent
    ERA5/CRU zonal-band fit (61.8% -> 60.2% mean relative error), whereas 42 and
    40 degrade it (64.7%, 67.9%) by over-wetting the 40-50N band, which is
    already too wet in the zonal mean even though its *land* is too dry. The
    six-box metric alone is not trustworthy for tuning -- it is dominated by US
    Midwest still being ~330mm short, so it rewards more shift indefinitely.
    **Resolved 2026-07-26**: see `ferrel_v_land_shift_deg` below, which decouples
    land from ocean instead of moving this shared centre further.

    Decoupled from `u_surface`'s mid-latitude centre (fixed at 48 deg, which sets
    the surface westerly/jet latitude) on 2026-07-25. Before that both used the
    same constant, so the latitude at which the zonal-mean flow switches from
    diverging to converging could not be adjusted without also moving the jet.

    That crossing latitude decides whether a mid-latitude continent sits inside
    the subtropical dry belt. Measured: the model crosses at ~48 deg N vs Earth's
    ~40 deg N, placing the entire 38-45N band on the diverging side -- the root
    cause of the long-standing US Midwest dryness (its divergence is 85%
    zonal-mean, so local fixes cannot work). The analytic crossing moves ~1:1
    with this value: 48->46.4, 44->42.6, 42->40.7, 40->38.8.

    Default is 44.0 as of 2026-07-25 (see above); passing 48.0 reproduces the
    original pre-fix behaviour exactly. See PLAN_PHYSICS_FIXES.md and
    overnight/FINDINGS.md.

    **2026-07-26**: `evolve_wind`'s v-relaxation *strength* weight (`a_v_row`)
    now also uses this centre (via `w_mid_v`), closing the "known residual"
    left open when this field first shipped -- previously that strength weight
    stayed jet-centred (48 deg) even after the *target* it relaxes toward moved
    to this field's centre, so the two peaked at different latitudes in
    DAILY/WEEKLY mode. Bit-identical whenever this value equals 48.0."""

    ferrel_v_land_shift_deg: float = -4.0
    """Additional equatorward offset [deg] applied to `ferrel_v_centre_deg`
    *only over land cells*, blended by land fraction so ocean cells keep the
    unshifted centre. `0.0` is an exact no-op: land and ocean use the same
    centre, identical to the shared-centre behaviour above.

    **Added 2026-07-26 to address the "40-50N land/ocean partition" ceiling**
    flagged in `ferrel_v_centre_deg`'s own docstring: that field is a single
    *zonal-mean* correction applied identically at every longitude in a row
    (`vc = vc + (v_surface[:, None] - vc_zm) * v_nudge`, an add-the-same-delta-
    everywhere nudge), so pushing its centre from 44 toward 40 to fix
    under-wet continental interiors (US Midwest, Canadian Prairies, Central
    Europe) necessarily pushes the mostly-ocean 40-50N zonal mean the same
    amount -- which is *already* a good fit to ERA5/CRU, so it gets over-wet.
    There was no way to fix the land without also moving the (already-correct)
    ocean, because the correction could not tell land and ocean apart.

    This field lets it: `w_mid_v`'s centre becomes
    `ferrel_v_centre_deg + ferrel_v_land_shift_deg * land_fraction` instead of
    a single scalar, using the same land mask (`masks.get_masks`) the rest of
    the model already derives from `elevation`. A negative value moves the
    land-only crossing further equatorward (toward Earth's ~40N) while the
    ocean-heavy zonal mean stays anchored near the reanalysis-validated 44.
    This is also physically motivated, not just a numerical trick: real
    continental summer thermal lows (the Asian and North American monsoon
    troughs being the strongest examples) pull the moisture-convergence
    latitude further poleward over land than the oceanic Ferrel cell sits at
    over the open ocean at the same latitude -- exactly the asymmetry a
    single shared centre cannot represent.

    Applied identically in `generate_wind_field` (MONTHLY/ANNUAL) and
    `evolve_wind` (DAILY/WEEKLY) so every time-scale still agrees on where the
    dry belt sits, the same invariant `ferrel_v_centre_deg` itself maintains.

    **Calibrated to -4.0 on 2026-07-26** via a 10yr real-terrain sweep on
    `saves/earth.pkl` (ferrel_v_centre_deg pinned at 44). All six boxes
    improve or hold vs. the shift=0 baseline, while the ERA5/CRU zonal-band
    fit costs under 1pp (vs. 6-7pp for the old uniform 44->40 shift this
    field replaces):

    | shift | Sahara | Kalahari | Atacama | Can.Prairies | US Midwest | Cent.Europe | zonal fit err |
    |---|---|---|---|---|---|---|---|
    | 0 | 303 | 279 | 154 | 478 | 414 | 496 | 56.7% |
    | **-4** | **211** | **166** | 142 | 451 | **602** | 504 | 57.3% |
    | -8 | 161 | 119 | 136 | 445 | 782 | 499 | 57.5% |
    | -12 | 159 | 137 | 143 | 388 | 913 | 466 | 57.0% |
    | -16 | 267 (^) | 283 (^) | 166 | 344 | 918 | 418 | 55.6%* |
    | Earth | <200 | <200 | <50 | 400-500 | 800-1000 | ~650 | -- |

    -4 was chosen over the box/zonal-fit-preferred -8 (and -12, which lands US
    Midwest almost exactly in its target range) after a *separate* 60yr
    MONTHLY spinup on the synthetic `mixed_elev` test fixture
    (`testing/test_climate_drift.py::test_nh_midlat_soil_moisture_not_floored`,
    a regression guard for the soil-moisture desiccation-spiral bug) surfaced
    a real risk the six real-terrain boxes never sampled: 45-65N land soil
    moisture (none of the six boxes reach north of 55N) declined monotonically
    and substantially with shift magnitude -- 0.302 (shift=0) -> 0.157 (-4,
    barely above the test's 0.15 line) -> 0.110 (-8, below it, approaching the
    hard 0.05 floor that triggers the spiral). Real terrain couldn't confirm
    or rule out the same tendency: `saves/earth.pkl`'s own 45-65N band is
    *already* pinned at the 0.05 floor at every tested shift including 0.0,
    for a separate, pre-existing reason unrelated to this field, which
    saturates the signal. Given the uncertainty, -4 was chosen as the largest
    shift with real margin above that guard's threshold on the one fixture
    that could show the effect at all -- user-confirmed choice (shown this
    exact table plus the soil-moisture numbers) over -8 and over leaving this
    gated at 0.0. -16 is worse than the table looks even ignoring the soil
    -moisture finding: its zonal-band fit *improves*, but Sahara and Kalahari
    both reverse direction there (the land-centre Gaussian window has drifted
    far enough to start overlapping the trade-wind window), so that number
    isn't trustworthy -- a reminder that no single metric here is sufficient,
    the same lesson `ferrel_v_centre_deg`'s own docstring already draws.
    Atacama remains far off target regardless of this field -- a separate,
    known gap (coastal fog/cold-current desert mechanism, not modelled).
    See PLAN_PHYSICS_FIXES.md for the fuller sweep log, including the -8
    numbers this field was calibrated away from and why."""

    spherical_metric_precip: bool = True
    """Use the true spherical metric in the moisture-flux convergence driver.

    `False` keeps `_moisture_convergence_numba`, which takes raw index
    differences: the zonal term is under-weighted by 1/cos(phi) (x2 at 60 deg,
    x3.9 at 75 deg), the meridional term lacks the cos(phi) flux weighting for
    converging meridians, and both pole rows are left identically zero.
    `True` routes the driver through `atmosphere.flux_divergence_spherical`,
    which implements
        div(F) = 1/(a cos phi) [ dFx/dlambda + d(Fy cos phi)/dphi ]
    and includes the pole rows. Verified against closed-form solutions in
    `testing/test_spherical_metric.py`.

    Opt-in because `u_scale`/`v_scale` and the `target_mean_mm_day` rescale were
    all calibrated against the metric-free kernel, so enabling this shifts the
    precipitation distribution poleward and will need recalibration -- that shift
    is the intended effect, not a regression. Low urgency at Earth obliquity
    (ROADMAP Theme 1); structural for high-obliquity or polar-precipitation
    worlds. See PLAN_PHYSICS_FIXES.md."""

    moisture_budget_precip_rescale: bool = True
    """Use bounded moisture-budget precipitation targeting instead of the
    legacy multiplicative row rescale.

    The bounded strategy treats the zonal target as aspirational, allocates
    added rain preferentially to existing condensation systems, and refuses to
    exceed local atmospheric-moisture/rainout capacity. Disable this flag only
    for legacy-regression experiments."""

    spherical_metric_clouds: bool = True
    """Use spherical wind divergence for cloud ascent/subsidence diagnostics.

    The cloud path in `simulate._evolve_temperature` historically used the
    same flat index-space derivative class as the legacy precipitation kernel.
    `True` computes divergence through `atmosphere.flux_divergence_spherical`
    with a unit scalar field. It is independently gated from
    `spherical_metric_precip` so each redistribution can still be measured
    separately. Disable either flag only for legacy-regression experiments."""

    moisture_advection_scale: float = 0.0
    """Blend weight [0-1] for an additional longer-range moisture transport term
    in `atmosphere.generate_precipitation` (`_advect_scalar_flux_eulerian`),
    layered on top of (not replacing) the existing short-range donor-cell blend
    -- 0.0 = original behavior exactly.

    History: the original transport implementation (single-jump semi-
    Lagrangian, pre-2026-07) monotonically *dried out* mid-latitude
    continental-interior land at any positive blend, across three variants
    (moisture-transport-investigation-2026-07 memory) -- traced to that jump
    diluting even the *ocean source* cells at real MONTHLY-mode substep dt
    (coastal RH 100%->66% at scale 0.7; moisture-advection-jump-dilution-
    2026-07 memory), not a property of transport itself. Replaced with a
    CFL-safe Eulerian upwind scheme (`_advect_scalar_flux_eulerian`, many
    small substeps instead of one ~5000km jump) that holds ocean RH steady
    (moisture-flux-eulerian-fix-2026-07 memory).

    With that fix, a re-swept real-terrain check (moisture-advection-scale-
    real-terrain-sweep-2026-07 memory) shows the opposite failure mode: it now
    *wets* both continental interior (Canadian Prairies/Central Europe improve
    correctly relative to Sahara) and Southern Hemisphere deserts (Kalahari
    167->350, Atacama 57->136 mm/yr at scale 0->1 -- overshoot, undesirable).
    Gating the long-range contribution's blend weight by the same
    `subsidence_suppression` that already gates `land_evap` (added same
    session) reduces but does not eliminate this overshoot (Kalahari
    167->306, Atacama 57->122 at scale 0->1 with the gate). US Midwest barely
    responds either way -- its bottleneck is a genuinely weak/divergent
    wind-derived `ascent` signal at that latitude band, not moisture transport
    or evaporation (same memory's wind-diagnostics check).

    Second gate (Jul 2026 follow-up): additionally damping the effective
    blend by `(1.0 - 0.5*drybelt_window)` -- deserts sit at drybelt_window's
    peak, continental-interior boxes near zero -- cut desert overshoot
    substantially further without eroding continental gains. Re-measured on
    the same real-terrain boxes at scale=0.7 (instantaneous, 2nd half of a
    1yr MONTHLY continuation): Sahara 155->157, Kalahari 117->163, Atacama
    74->109 (all three now stay under the 200 mm/yr desert target at every
    scale up to 1.0), vs. Canadian Prairies 138->243 and Central Europe
    99->236 -- both continental boxes now clearly exceed all three deserts,
    a clean ranking flip that wasn't achieved before. US Midwest still lags
    (73->127, still under Sahara) -- unchanged, confirms this is genuinely
    the separate wind-model gap, not a moisture-gate tuning issue. Default
    stays 0.0: this result is promising enough to be a real candidate for
    enabling by default, but flipping it is a deliberate separate decision
    (would need golden-state/climate-drift/ECS bound re-baselining across
    the whole test suite, and a longer multi-year check that continental
    interior's 10yr EMA -- which barely moves within 1yr -- actually reaches
    the 350-450 mm/yr target over time), not bundled into this fix."""

    humidity_advection_cfl: bool = False
    """Use a real Courant-number-based velocity scale for the short-range
    donor-cell humidity advection in `atmosphere.generate_precipitation`,
    instead of the fixed `|u|/20`, `|v|/12` divisors (tuned constants,
    independent of dt or grid spacing). When True, the scale becomes
    `|u|*dt_seconds/dx` / `|v|*dt_seconds/dy` (clipped to [0,1]), matching the
    real Courant number the long-range `_advect_scalar_flux_eulerian` term
    already computes and substeps against. Default False reproduces the
    original constants exactly (bit-identical output)."""

    # ------------------------------------------------------------------ #
    # 2-layer soil moisture bucket (Jul 2026 desiccation-bistability fix)
    # ------------------------------------------------------------------ #
    # The single-layer bucket's gain/drain balance is genuinely bistable under one
    # global gain constant -- soil either saturates near 1.0 everywhere or collapses
    # to its 0.05 floor, with no stable middle ground (measured directly while
    # calibrating; see atmosphere.generate_precipitation's soil-update comment and
    # known-physics-gaps.md). These fields add a slow deep/root-zone reservoir
    # alongside the existing fast surface layer (`PlanetState.soil_moisture`) so each
    # region's long-term precip/evap balance can settle at its own differentiated
    # equilibrium instead of being pinned to one of two global attractors.
    #
    # NOTE on design: an earlier version gated the deep layer's input on surface
    # moisture exceeding a field-capacity threshold (real-soil-physics-style
    # percolation). Measured directly on real terrain that this doesn't work: the
    # surface layer is *already* pinned at its 0.05 floor (the exact bug being
    # fixed) and can never climb back above a field-capacity threshold on its own,
    # so percolation never triggers and the deep layer just decays to zero with no
    # input -- a chicken-and-egg trap. Replaced with a direct precipitation-fed gain
    # for the deep layer (independent of the surface layer's own state), which
    # sidesteps the trap entirely: the deep layer's equilibrium reflects each
    # region's real long-run precip rate directly, decoupled from whichever branch
    # the bistable surface layer happens to be on.
    soil_deep_gain_rate: float = 0.0005
    """Fraction of precipitation that feeds the deep layer directly per day
    [1/day], independent of the surface layer's own state (see NOTE above).

    Was 0.0 (exact no-op) through 2026-07: a controlled real-terrain
    comparison at that time found the deep layer amplified whatever desert-
    vs-continental-interior differentiation already existed in its input
    rather than creating it, because desert land_evap wasn't yet gated by
    subsidence/dry-belt suppression -- so raw precipitation itself barely
    differentiated desert from continental interior (both were governed by
    the same locally-saturating humidity), and the deep layer just amplified
    that flat signal (pushing Sahara to ~354 mm/yr without helping
    continental interior). See known-physics-gaps.md item 3 UPDATE 2.

    That precondition changed: `desert_evapotranspiration_fix` (UPDATE 4 in
    the same doc) gated `land_evap` by `subsidence_suppression`, which made
    raw precipitation itself reliably differentiate desert from continental
    interior (measured on `saves/earth.pkl`, 10yr real-terrain, instantaneous
    2nd-half: Sahara/Kalahari/Atacama 110-217 mm/yr vs. Canadian Prairies/US
    Midwest/Central Europe 392-694 mm/yr). Since this knob feeds the deep
    layer directly from precipitation, it now inherits that differentiation
    instead of erasing it -- confirmed at 0.0005: `soil_deep` settles at
    ~0.05-0.25 for desert boxes vs. ~0.30-0.44 for continental-interior boxes
    (Atacama at 0.25 is the one overlap, a known separate gap -- see its
    memory note below), materially above the single-layer surface bucket's
    universal 0.05 floor pin (see NOTE above) for every box alike. Named-box
    precip moved modestly in the right direction (desert boxes +5-25 mm/yr,
    within measurement noise of their <200 target; continental-interior
    +15-80 mm/yr, closing part of the remaining gap to Earth targets) with
    no reordering of any box relative to any other. This was the root-cause
    precondition the pre-2026-07 investigation was missing, not a case where
    the deep-layer mechanism itself needed redesigning.

    Atacama (0.25) not clearing Kalahari (0.09-0.14) is expected, not a
    regression: the model has no coastal-fog/cold-current desert mechanism
    (known-physics-gaps.md), so Atacama has never cleanly separated from
    continental-interior boxes on any metric tried, at any prior calibration
    step, in this project's history."""

    soil_deep_drain_rate: float = 0.002
    """Slow baseflow/groundwater drain rate for the deep layer [1/day]. Unlike
    percolation, this *is* a sink (water leaves the system). ~500-day (1.4yr)
    e-folding time by default -- deliberately much slower than the surface layer's
    day-to-week response, so the deep reservoir carries real multi-year memory rather
    than snapping to a new equilibrium within one spinup like the single-layer bucket
    did."""

    soil_deep_evap_weight: float = 0.5
    """Efficiency of deep-layer moisture at supporting evaporation, relative to
    surface moisture [0-1]. `land_evap`'s soil factor becomes
    `0.35 + 0.65*max(soil_surface, soil_deep_evap_weight*soil_deep)` -- deep moisture
    can "rescue" evaporation via root uptake when the surface is dry, at reduced
    efficiency, without needing to dominate when the surface is already adequately
    moist. 0.5 is a starting point pending calibration against real terrain."""

    enable_surface_hydrology: bool = False
    """Enable experimental runoff, lake storage, and D8 river routing.

    Disabled by default until multi-decadal real-terrain water budgets and
    precipitation feedbacks are calibrated. When enabled, the model records
    `surface_water_mm`, `river_discharge_mm_day`, and
    `runoff_to_ocean_mm_day` in PlanetState."""

    runoff_soil_threshold: float = 0.3
    """`soil_moisture_deep` saturation above which precipitation generates
    runoff. Checks the deep/root-zone layer, not the fast surface bucket --
    the surface layer sits chronically pinned near its 0.05 floor across
    nearly all real terrain (see the 2026-07 high-latitude soil-desiccation
    fix), so a surface-only threshold near 0.75 never triggers in practice
    (confirmed: bit-exactly zero runoff/river_discharge/surface_water_mm on a
    10yr real-terrain continuation prior to this fix). 0.3 sits around the
    75th percentile of `soil_moisture_deep` measured on that same real-terrain
    continuation (p50~0.12, p75~0.33, p90~0.78), so roughly the wettest
    quarter of land generates runoff -- continental interior and tropics, not
    deserts."""

    runoff_fraction: float = 0.35
    """Maximum fraction of precipitation converted to runoff at saturation."""

    river_routing_passes: int = 8
    """Number of vectorized D8 routing passes per climate step."""

    river_routing_fraction: float = 0.55
    """Fraction of available water sent downhill during each routing pass."""

    sst_land_coupling_strength: float = 0.0
    """Strength of the SST -> adjacent-land precipitation coupling, per kelvin
    of upwind ocean temperature anomaly. `0.0` is an exact no-op.

    Multiplies `subsidence_suppression` by `1 + strength * upwind_sst_anomaly`
    over land, so water that is cold *for its latitude* dries the land it blows
    onto and warm water wets it. See `atmosphere._upwind_sst_anomaly` for the
    field and `generate_precipitation` for where it is applied.

    **Audit D3's named blocker.** D5 built real eastern-boundary upwelling (a
    wind-stress-curl gyre solve producing Benguela -0.54 K, Canary -0.36 K,
    Humboldt -0.25 K against gyres-off) and measured Atacama precipitation
    unchanged across gyre strengths 0.0 to 3.0. The conclusion recorded there is
    that the missing mechanism is not the ocean cooling but the coupling, and
    that the coupling must be built and shown to transmit *before* any further
    ocean physics -- otherwise the null result just repeats. This is that term.

    Applied to `subsidence_suppression` specifically because that is the one
    pathway already demonstrated to reach land precipitation in this model
    (`coastal_upwelling_fog_strength`, the geometric proxy this supersedes,
    moved Atacama 123 -> 102 mm/yr through it). Ocean evaporation already
    responds to SST through `qsat`, and the moisture budget's row rescale
    removes it again -- process note 9."""

    sst_land_target_weight: float = 0.0
    """Target-side half of the SST -> land precipitation coupling, per kelvin of
    upwind ocean temperature anomaly. `0.0` is an exact no-op.

    Scales each land cell's share of its row's precipitation target by
    `1 + weight * min(tanh(anomaly / 2 K), 0)`, renormalized per row, so it is
    exactly row-mean-preserving: it redistributes rainfall within a latitude
    circle away from land fed by anomalously cold water, and cannot move a zonal
    total the model is calibrated against.

    **Cold side only, and saturating.** Two things the sweep forced:

    - The warm half is a *double count*. Ocean evaporation already responds to
      SST through `qsat`, and `monsoon_east_margin_exemption` is calibrated at
      3.0 for exactly the warm western-boundary-current margins (SE US, East
      China, S Japan) a symmetric form would boost again -- measured, that drove
      S Japan to 2809 mm/yr against a 1100-2200 target at 256x512. The
      suppressing half has no counterpart anywhere in the model: a marine
      inversion capping convection is why Atacama and the Namib are hyper-arid
      *on a coastline*.
    - `tanh` rather than linear-in-kelvin. The model's Kuroshio anomaly reaches
      +1.9 K where the land median is 0.35 K, so a linear response bounded only
      by a clip let one outlier take the whole bound. The 2 K reference is a
      fixed constant, not a knob -- it describes how far an SST anomaly can shift
      a boundary layer, which is not a per-planet tuning freedom.

    Note one indirect effect that is *not* a warm-side boost: a warm patch lifts
    its row's ocean mean, pushing other ocean cells negative and so raising the
    fed land's share slightly after renormalization (+0.1% measured). That is
    the row-relative reference, not this term's sign.

    **This is the half that survives.** Its companion
    `sst_land_coupling_strength` gates `subsidence_suppression`, which sits
    upstream of the moisture budget's rescale -- the deficit fill simply tops
    the cell back up toward a target that never heard about the anomaly, and the
    measured effect on every benchmark metric is under 0.1pp at any strength.
    Process note 9's rule (check which side of the rescale a mechanism sits on)
    applied to ocean coupling, exactly as A5-OROG had to apply it to orography.
    """

    enable_prognostic_condensate: bool = False
    """Enable the gated bulk vapor-to-condensate-to-rain closure.

    When enabled, resolved ascent and relative humidity transfer a conserved
    column-water mixing ratio from vapor into a persistent condensate reservoir;
    fallout then becomes precipitation.  The current target allocator remains
    active during calibration, so this switch supports a clean A/B measurement
    before the allocator is retired.  ``False`` preserves the established
    rainfall path exactly.
    """

    enable_stability_aware_condensation: bool = False
    """Use the gated CAPE-plus-resolved-ascent condensation closure.

    This replaces the simple relative-humidity/ascent activation inside the
    existing condensate reservoir with a one-column moist-stability estimate:
    a parcel lifts dry-adiabatically to its LCL, then moist-adiabatically to a
    fixed lower-tropospheric reference height.  Positive CAPE-like buoyancy
    and resolved convergence must both be present before vapor above the
    critical RH relaxes into condensate.  Requires
    ``enable_prognostic_condensate``; default ``False`` preserves all current
    precipitation behavior.
    """

    stability_condensation_critical_rh: float = 0.70
    """Reference RH for stability-aware condensation (SBM-like 70% default)."""

    stability_condensation_reference_height_m: float = 3500.0
    """Parcel-evaluation height [m] for the one-column CAPE proxy."""

    stability_condensation_cape_scale_j_kg: float = 50.0
    """CAPE-like e-folding scale [J kg-1] for stability-aware activation."""

    enable_two_layer_convective_adjustment: bool = False
    """Persist a mid-tropospheric thermal layer for stability-aware convection.

    With the stability-aware condensate gate, this carries a 3.5-km
    environmental temperature between calls.  It relaxes toward the resolved
    lower-tropospheric lapse profile and receives latent heating from local
    condensation, giving buoyancy a physical memory without creating a second
    water reservoir.  Requires ``enable_stability_aware_condensation`` and is
    off by default.
    """

    two_layer_midlevel_relaxation_days: float = 10.0
    """Radiative/dynamical relaxation time [days] for midlevel temperature."""

    two_layer_upper_humidity_fraction: float = 0.25
    """Initial fraction [0-1] of conserved vapor assigned to the upper layer."""

    two_layer_entrainment_days: float = 2.0
    """Resolved-ascent exchange time [days] between lower and upper vapor."""

    two_layer_vertical_mixing_days: float = 5.0
    """Thermodynamic exchange time [days] between midlevel and resolved air."""

    two_layer_upper_mass_fraction: float = 0.25
    """Mass fraction [0-1] represented by the midlevel thermodynamic layer."""

    two_layer_cloud_radiative_weight: float = 0.50
    """Cloud-fraction weight [0-1] diagnosed from suspended midlevel condensate."""

    two_layer_cloud_reference_q: float = 0.004
    """Midlevel condensate mixing ratio corresponding to unit cloud fraction."""

    two_layer_pressure_depth_pa: float = 35_000.0
    """Pressure thickness [Pa] between the surface and midlevel reservoir."""

    two_layer_vertical_velocity_scale_pa_s: float = 0.05
    """Pressure-vertical-velocity scale [Pa s-1] for layer exchange activation."""

    enable_three_level_pressure_column: bool = False
    """Enable the experimental native lower/mid/upper pressure-column state."""

    enable_closed_three_level_thermodynamics: bool = False
    """Use the finite-volume three-level water/thermodynamic closure.

    Requires the existing prognostic-column, stability-convection, two-layer,
    and three-level pressure-column gates.  It replaces that path's unweighted
    humidity exchange and lapse-profile temperature relaxation with a
    mass-weighted moist-static-energy update.  The host temperature solver is
    the explicit resolved radiative/surface step; this operator applies only
    vertical exchange and phase conversion afterwards.  Experimental and off
    by default pending a compact climate gate.
    """

    enable_diabatic_interface_mass_flux: bool = False
    """Derive closed-column interface omega from prior raw-column latent heating.

    Nested inside ``enable_closed_three_level_thermodynamics``.  This uses the
    zonal-mean, allocator-free precipitation heating and resolved layer static
    stability to derive a mass-consistent large-scale pressure circulation,
    replacing the invalid omega diagnosed from independently evolved raw wind
    divergences.  It has no amplitude scalar: a vertical-Courant violation is
    an admission failure, not a value to clip.
    """

    enable_shared_pressure_coordinate_circulation: bool = False
    """Use one diabatic pressure-coordinate solve for winds and interface mass flux.

    Requires ``enable_diabatic_interface_mass_flux``. The longitude-mean
    zonal wind and the meridional wind reconstructed from the same closed layer
    divergences become the precipitation/transport circulation. This is the
    next experimental replacement for independent raw layer winds carrying
    incompatible mass and energy transport; it has no strength or damping
    scalar and remains default-off pending compact climate validation.
    """

    enable_pressure_coordinate_moisture_closure: bool = False
    """Use a pressure-mass water budget with the shared circulation experiment.

    Requires the closed three-level thermodynamics and shared pressure-coordinate
    circulation gates.  Surface evaporation enters the lower layer in kg m-2;
    each layer's actual interface ascent can convert vapour to a suspended-cloud
    reservoir [kg m-2], and explicit fallout is the only non-numerical
    precipitation sink.  With ``enable_separate_precipitating_hydrometeors``,
    cloud water autoconverts into a second pressure-mass hydrometeor reservoir;
    only that reservoir sediments. It replaces the older mixed-q bulk
    condensate accounting only inside this nested experimental family.  No
    precipitation target, geographic exception, strength, or damping control is
    introduced.  ``False`` preserves the prior experimental path exactly.
    """

    enable_prognostic_overturning_heat_reservoir: bool = False
    """Buffer condensation heating before it diagnoses shared overturning.

    Requires the pressure-coordinate moisture closure.  It stores the
    cosine-area-balanced, zonal latent-heating anomaly and relaxes it on the
    free-tropospheric radiative timescale derived from heat capacity and
    ``4 sigma T^3``.  The stored anomaly, rather than one step's condensation,
    drives the mass-consistent omega solve.  This adds no user tuning scalar.
    """

    enable_pressure_coordinate_mse_transport: bool = False
    """Transport three-layer moist static energy with shared pressure winds.

    Requires the pressure-coordinate moisture closure. Each layer's MSE is
    carried on the same conservative faces as its vapour, and lower-layer
    evaporation imports its latent energy. This closes the otherwise missing
    large-scale energy-export path without a circulation or damping scalar.
    Experimental and off by default pending compact climate validation.
    """

    enable_mse_constrained_pressure_circulation: bool = False
    """Diagnose pressure-column overturning from the MSE export budget.

    Requires the prognostic overturning-heating reservoir and pressure-MSE
    transport. A mass-closed lower/upper branch carries the zonal,
    area-balanced diabatic forcing, with its mass flux fixed by resolved MSE
    contrast. No wind-strength, damping, or omega-cap control is present.
    Experimental and off by default pending compact climate validation.
    """

    enable_three_branch_mse_pressure_circulation: bool = False
    """Use all three pressure branches in the MSE-constrained circulation.

    Requires ``enable_mse_constrained_pressure_circulation``. The unique
    minimum-mass-flux solution uses each layer's MSE departure from the
    mass-weighted column mean, while diagnosed interface fluxes close its
    layerwise MSE deposition. The deposition is diagnostic until phase heating
    can use it without double counting. Experimental and off by default.
    """

    enable_momentum_constrained_three_branch_mse_circulation: bool = False
    """Constrain the three-branch MSE solution with zonal momentum transport.

    Requires ``enable_three_branch_mse_pressure_circulation``. It replaces the
    minimum-mass-flux branch choice with zero mass-weighted transport of the
    resolved vertically sheared zonal momentum. The planetary angular-momentum
    term cancels under exact local mass closure. Experimental and off by
    default pending compact climate validation.
    """

    enable_prognostic_pressure_coordinate_momentum: bool = False
    """Evolve the pressure-level wind shear before MSE branch selection.

    Requires the momentum-constrained three-branch MSE closure. Hydrostatic
    layer pressure gradients, exact Coriolis rotation, and conservative prior-
    step interface momentum exchange evolve all three pressure winds. This
    supplies a prognostic shear state rather than using raw wind diagnostics.
    Experimental and off by default pending compact climate validation.
    """

    enable_hydrostatic_sigma_pressure_coordinate_transport: bool = False
    """Enable the complete experimental hydrostatic sigma-column transition.

    Requires the closed three-level pressure-column family.  The deeply nested
    gate replaces its pressure mass, vapour/MSE, layer-resolved cloud and
    hydrometeor reservoirs, continuity-derived interface flux, and
    pressure-level momentum as one state transition.  It is default-off until
    gate-off, persistence, compact, and long admission checks pass.
    """

    enforce_three_level_mass_closure: bool = False
    """Close mass-weighted lower/mid/upper divergence before diagnosing omega."""

    enable_three_level_horizontal_mass_flux_closure: bool = False
    """Apply a bounded resolved divergent-wind closure to the upper pressure level.

    This experimental gate corrects the layer-weighted horizontal mass-flux
    residual after all three momentum levels evolve.  It is distinct from the
    local algebraic omega closure above and remains opt-in pending climate
    validation.
    """

    three_level_horizontal_mass_flux_strength: float = 1.0
    """Fraction [0-1] of diagnosed three-level mass-flux residual corrected per step."""

    three_level_horizontal_mass_flux_max_speed_m_s: float = 12.0
    """Maximum magnitude [m s-1] of one horizontal mass-flux correction step."""
    three_level_horizontal_mass_flux_throughflow_max_speed_m_s: float = 80.0
    """Upper-wind bound [m s-1] for removing the divergence-free column throughflow mode."""

    three_level_divergence_filter_strength: float = 0.0
    """Weight [0-1] of conservative small-scale filtering before omega diagnosis."""

    three_level_divergence_filter_passes: int = 0
    """Number of divergence-filter passes for the experimental pressure column."""

    three_level_balanced_thermal_wind_relaxation: float = 0.0
    """Per-step [0-1] relaxation of upper wind toward resolved thermal-wind shear."""

    three_level_thermal_wind_upper_pressure_pa: float = 30_000.0
    """Pressure [Pa] assigned to the balanced upper thermal-wind target."""

    enable_native_balanced_pressure_dynamics: bool = False
    """Blend pressure-level winds toward native hydrostatic geostrophic balance."""

    native_balanced_pressure_relaxation: float = 0.0
    """Per-step [0-1] relaxation of middle/upper winds toward balanced targets."""

    native_balanced_ageostrophic_timescale_hours: float = 0.0
    """Cross-isobar adjustment timescale [h]; zero keeps the target geostrophic."""

    native_balanced_mid_pressure_pa: float = 65_000.0
    """Pressure [Pa] assigned to the native balanced middle-wind target."""
    native_balanced_surface_pressure_pa: float = 90_000.0
    """Pressure [Pa] assigned to the native balanced lower-wind target."""
    native_balanced_overturning_speed_m_s: float = 0.0
    """Lower-branch speed [m s-1] for the thermally centred native Hadley closure; zero disables it."""
    enable_native_balanced_diabatic_overturning: bool = False
    """Diagnose native Hadley overturning strength from persistent midlevel diabatic heating."""
    native_balanced_diabatic_overturning_max_speed_m_s: float = 2.0
    """Physical cap [m s-1] on the diagnosed native diabatic lower branch."""

    enable_native_balanced_moist_static_energy_overturning: bool = False
    """Diagnose native Hadley overturning from a full lower/mid moist-static-energy budget.

    Supersedes ``enable_native_balanced_diabatic_overturning`` when both are
    enabled.  Instead of inferring heating solely from the persistent midlevel
    temperature anomaly, this sums two independently diagnosed terms before
    converting to pressure velocity: latent heating from actual resolved
    condensation (the previous step's precipitation field, falling back to the
    midlevel anomaly memory when unavailable) and a resolved radiative/thermal
    heating rate toward the model's own seasonal equilibrium-temperature
    target.  Both remain zonal-mean, tropical-band scalars; the mass-conserving
    three-level structure is still supplied by ``thermally_direct_overturning``.
    """
    native_balanced_mse_overturning_max_speed_m_s: float = 2.0
    """Physical cap [m s-1] on the moist-static-energy-diagnosed native lower branch."""
    native_balanced_mse_radiative_relaxation_days: float = 10.0
    """Relaxation time [days] converting the resolved thermal-equilibrium anomaly into a heating rate."""
    native_balanced_mse_use_toa_radiative_target: bool = False
    """Use a genuine top-of-atmosphere radiative-equilibrium target instead of the ocean-transport target.

    When False (default), the radiative term relaxes toward
    ``_compute_T_base_ocean_full()`` -- the same seasonal target used for the
    SST field, which bakes in AMOC/ACC bonuses, hemisphere asymmetry, and
    ocean thermal lag.  When True, it relaxes toward
    ``temperature.temperature_kelvin_for_lat`` evaluated directly at the
    current day, with no transport terms.  This isolates whether the MSE
    closure's skill gain (PRIOR_ART_IMPLEMENTATION_PLAN.md Section 10) is
    genuine radiative-heating physics or borrowed transport asymmetry.
    """

    three_level_upper_humidity_fraction: float = 0.15
    """Initial fraction [0-1] of total vapor placed in the upper reservoir."""

    three_level_mid_upper_pressure_depth_pa: float = 30_000.0
    """Pressure thickness [Pa] between the mid and upper reservoirs."""

    three_level_upper_height_m: float = 8000.0
    """Reference height [m] of the upper temperature and humidity reservoir."""

    three_level_mid_wind_pgf_fraction: float = 0.55
    """Fraction of upper-level thermal pressure forcing used for middle wind."""

    three_level_mid_wind_damping: float = 0.08
    """Rayleigh damping [day-1] of the prognostic middle-wind level."""

    three_level_mid_wind_relaxation: float = 0.10
    """Per-step relaxation of middle wind toward adjacent resolved wind levels."""

    three_level_upper_wind_pgf_fraction: float = 1.0
    """Fraction of the shared upper-level thermal pressure forcing used for the
    three-level path's own, independent upper wind (``PlanetState.upperlevel_wind_u/v``).

    PRIOR_ART_IMPLEMENTATION_PLAN.md Section 16 found that the three-level
    experimental path's excess cross-equatorial transport traces back to
    `state.wind_u_aloft`/`wind_v_aloft` -- the same always-on, default-on
    "1.5-layer atmosphere" jet-stream kernel (`atmosphere.evolve_wind_aloft`,
    called unconditionally every step regardless of any experimental gate)
    that `wind_upper_pgf_amp`/`wind_upper_damping` were extensively calibrated
    against for real jet-latitude/speed skill. The three-level path's balanced-
    pressure blend, thermal-wind relaxation, `thermally_direct_overturning`'s
    upper branch, and `close_upper_mass_flux`'s correction were all applying
    directly to that shared, already-validated field -- so any attempt to tame
    its magnitude for the experimental path risked silently regressing the
    default jet stream, since both consumers shared the exact same array and
    constants. Section 17 decouples them: the three-level path now evolves its
    own independent upper wind, using the same `evolve_wind_aloft` physics but
    through this fraction (multiplying `wind_upper_pgf_amp`) and its own
    `three_level_upper_wind_damping`, so it can be tuned freely without ever
    touching the shared kernel. `1.0` reproduces the shared kernel's full PGF
    amplitude as the starting point (the three-level additions previously
    never scaled the raw forcing term itself, only blended/added onto its
    result), mirroring how `three_level_mid_wind_pgf_fraction` already gives
    the independent middle level its own fraction (0.55) of this same term.
    """

    three_level_upper_wind_damping: float = 0.08
    """Rayleigh damping [day-1] of the three-level path's independent upper
    wind (see `three_level_upper_wind_pgf_fraction` docstring for the full
    rationale). Deliberately starts higher than the shared kernel's
    `wind_upper_damping=0.05`: Section 16 measured the shared kernel's raw
    meridional wind at 5-30x the ~1-3 m/s literature Hadley-cell value, and
    both the project's own precedent (the independent middle level already
    uses 0.08 against the shared level's 0.05) and that measurement point the
    same direction -- stronger damping tames excess magnitude without
    eliminating the underlying thermal-wind mechanism entirely. This matches
    `three_level_mid_wind_damping` exactly as the most directly precedented
    starting value; Section 17's own damping sweep
    (`scripts/diagnose_resolved_wind_magnitude.py`) reports whether a
    stronger value is warranted for this level specifically.
    """

    enable_prognostic_column_water: bool = False
    """Use the experimental raw conserved-column-water precipitation path.

    When enabled, precipitation is removed only from the local prognostic
    humidity reservoir and bypasses both imposed zonal row-target rescale
    variants.  The existing humidity transport, evaporation, and condensation
    machinery remain in place; the gate makes its local water sink observable
    without target-rainfall correction.  It is intentionally ``False`` by
    default until long climate validation establishes a better complete closure.
    """

    enable_energy_limited_evaporation: bool = False
    """Limit raw-column evaporation by available surface radiative energy.

    The legacy precipitation path uses empirical target correction and is left
    bit-identical.  The opt-in conserved-column path instead needs an explicit
    surface energy bound: an unlimited humidity-deficit flux can otherwise
    recycle substantially more water than a terrestrial surface can supply.
    """

    evaporation_surface_shortwave_transmissivity: float = 0.50
    """Atmospheric transmission of daily-mean shortwave used by the experimental evaporation energy cap."""

    evaporation_latent_energy_fraction: float = 0.75
    """Share [0-1] of available surface shortwave permitted to become latent heat in the experimental cap."""

    evaporation_downwelling_longwave_w_m2: float = 0.0
    """Additional downwelling longwave energy [W m-2] available to experimental energy-limited evaporation."""

    enable_humidity_dependent_downwelling_longwave: bool = False
    """Diagnose the experimental longwave increment from lower-air humidity and cloud instead of a spatial constant."""

    evaporation_longwave_clear_sky_emissivity_floor: float = 0.70
    """Lower bound for Brutsaert clear-sky longwave emissivity in the experimental surface-energy closure."""

    evaporation_longwave_cloud_emissivity_weight: float = 0.50
    """Fraction of the remaining sky-emissivity deficit filled by cloud fraction in the experimental closure."""

    evaporation_longwave_reference_emissivity: float = 0.80
    """Dry-reference emissivity already implicit in the surface-energy approximation; only excess back-radiation is added."""

    enable_cloud_precipitating_condensate_partition: bool = False
    """Diagnose excess bulk condensate as precipitating, not optically active cloud.

    The existing condensate reservoir conserves cloud water and hydrometeors
    together.  This experimental radiation split caps the cloud-water portion
    while retaining the excess as precipitating mass for the unchanged fallout
    budget.  It is a transition toward separate cloud/precipitating reservoirs.
    """

    cloud_optical_condensate_cap_q: float = 0.001
    """Maximum optically active cloud-condensate mixing ratio in the partitioned bulk reservoir."""

    condensate_autoconversion_timescale_days: float = 0.25
    """Cloud-water-to-precipitating-hydrometeor conversion time in the separated-reservoir path."""

    enable_separate_precipitating_hydrometeors: bool = False
    """Persist a distinct precipitating-hydrometeor reservoir in experimental closures.

    In the pressure-coordinate closure both this reservoir and suspended cloud
    water are pressure masses [kg m-2].  Cloud water autoconverts above the
    retained optical-cloud amount; only hydrometeors sediment.
    """

    enable_hydrometeor_transport: bool = False
    """Advect persisted precipitating hydrometeors with the resolved cloud-layer wind before sedimentation."""

    enable_simplified_betts_miller_convection: bool = False
    """Use an opt-in resolved-ascent Betts--Miller humidity relaxation instead of the slab-CAPE condensation closure."""

    betts_miller_relaxation_hours: float = 2.0
    """Convective humidity-adjustment timescale [hours] in the simplified Betts--Miller closure."""

    betts_miller_target_relative_humidity: float = 0.85
    """Boundary-layer reference relative humidity approached by convectively ascending air in the simplified Betts--Miller closure."""

    betts_miller_midlevel_target_relative_humidity: float = 0.70
    """Mid-tropospheric reference RH in the simplified Betts--Miller vertical target profile."""

    betts_miller_upper_target_relative_humidity: float = 0.50
    """Upper-tropospheric reference RH in the simplified Betts--Miller vertical target profile."""

    three_level_diabatic_ascent_scale: float = 0.0
    """Resolved tropical diabatic-ascent contribution for the experimental pressure column; zero preserves divergence-only ascent."""

    enable_three_level_flux_form_exchange: bool = False
    """Use finite-volume pressure-flux exchange |omega| dt / dp in the native three-level column."""

    column_water_use_bulk_condensate_rainfall: bool = False
    """With both prognostic-water gates enabled, omit empirical vapor rainout.

    This selects the structural vapor -> bulk condensate -> fallout closure:
    resolved ascent and saturation form condensate, and only condensate fallout
    (plus mandatory supersaturation adjustment) reaches the surface.  It is a
    nested experimental switch so the conservative transport migration can be
    evaluated separately from replacing the established empirical rain chain.
    """

    condensate_condensation_timescale_days: float = 0.5
    """Bulk-condensation relaxation time [days] for active ascending air."""

    condensate_fallout_timescale_days: float = 1.0
    """Bulk condensate fallout time [days]; shorter values rain sooner."""

    condensate_transport_scale: float = 1.0
    """Blend [0-1] of CFL-safe horizontal transport for suspended condensate.

    This only applies while ``enable_prognostic_condensate`` is enabled.
    ``1.0`` carries condensate with the resolved wind before fallout; ``0.0``
    is the local-reservoir ablation used to isolate the transport contribution.
    """

    sst_land_coupling_km: float = 600.0
    """E-folding upwind fetch [km] for `sst_land_coupling_strength`'s SST
    anomaly field. A physical distance rather than a cell count, so the
    mechanism is resolution-invariant -- the trap the monsoon inland mask (fixed
    20 cells: 7 deg at 1024 columns, 56 deg at 128) and `_maritime_proximity`
    (fixed 512-pass cap) each hit once. Inert while
    `sst_land_coupling_strength` is 0.0."""

    coastal_upwelling_fog_strength: float = 0.5
    """Strength [0-1] of coastal-fog/cold-current desert suppression (see
    `atmosphere.generate_precipitation`'s comment for the mechanism and why
    it's a diagnostic gate rather than real ocean-upwelling physics). Applies
    an additional multiplicative suppression to `subsidence_suppression`
    (and therefore both `land_evap` and `precip_potential`) at west-coast
    land cells (coast + ~2 cells inland, decaying) within the subtropical
    dry-belt latitude window (`DRYBELT_CENTER_DEG` ~28 deg) -- targets
    Atacama/Namib/Baja California/Western-Sahara-analogue deserts
    specifically, the model's long-standing "no coastal-fog mechanism" gap.

    Calibrated 2026-07 on real terrain (`saves/earth.pkl`, 10yr, instantaneous
    2nd-half mm/yr): Atacama 123 (strength=0) -> 116 (0.3) -> 111 (0.5) -> 106
    (0.7) -> 102 (0.9), monotonic and controlled. Sahara/Kalahari/Canadian
    Prairies/US Midwest/Central Europe all stay within measurement noise
    (<3%) across the whole range -- the coastal+drybelt-latitude gating is
    spatially selective, as intended. **A real, honest, partial win, not a
    fix**: even at strength=1.0 this would not get Atacama near its <50
    mm/yr Earth target (the box's land cells only average ~0.28 coastal-mask
    weight, and the mechanism competes with the row-mean-preserving desert
    redistribution downstream) -- matches this project's repeated prior
    finding that Atacama has never cleanly separated from other deserts on
    any metric tried. 0.5 shipped as a moderate default given the
    monotonic, side-effect-free improvement; 0.0 remains an exact no-op for
    anyone who wants the pre-2026-07 behavior."""

    monsoon_east_margin_exemption: float = 3.0
    """Strength of the exemption that reduces `drybelt_window`'s flat
    subtropical dry-belt latitude penalty at east-coast/monsoon land cells
    (SE US, East China, S Japan -- real Cfa/humid-subtropical climates that
    sit inside the same ~20-36 deg latitude window as Sahara/Kalahari/
    Atacama). See `atmosphere.generate_precipitation`'s `monsoon_margin_factor`
    and `_monsoon_recover` comments for the full mechanism. 0.0 is an exact
    no-op (pre-2026-07-30 behavior). Not capped at 1.0 like
    `coastal_upwelling_fog_strength` -- unlike that gate, this one only ever
    *removes* a penalty (the result is still clipped into
    `subsidence_suppression`'s normal [0.08, 1.0] range), so there's no
    "over-application" failure mode the same way, and the real-terrain sweep
    below found continued (if diminishing) benefit past 1.0.

    Root cause (2026-07-30, real-terrain audit following the user's own
    reference-Koppen-map comparison): `drybelt_window` is pure |latitude| --
    it cannot distinguish real subsiding-air deserts (western continental
    margins, under the eastern lobe of a subtropical high: Sahara, Mexico/SW
    US, Kalahari/Namib, Atacama) from eastern continental margins at the same
    latitude, which in reality escape the high via warm-current/monsoon
    moisture pump (Gulf Stream -> SE US, Kuroshio -> East China/S Japan).
    Measured directly on `saves/test.npz` (512x1024): `subsidence_suppression`
    was 0.14-0.28 for SE US/East China/S Japan, AT OR BELOW Sahara's own 0.24
    -- the model was suppressing real Cfa climates as hard as an actual
    desert. 5-year real-terrain continuation, named-box Koppen breakdown
    (baseline -> this default):
        SE US:       94% BSh -> 30% BSh / 61% Cfa
        East China:  99% BSh -> 39% BSh / 35% Cfa / 19% Af / 7% Aw
        S Japan:     87% BSh -> 100% Cfa
    Sahara/Kalahari/Atacama/US Midwest/Central Europe all held within
    measurement noise (Kalahari's own BSh share drifted 91%->84%, its closest
    real east coast -- Mozambique Channel -- sits within this mechanism's
    inland decay reach of the box's eastern edge; accepted as a minor,
    directionally-correct cost, the same category of trade-off
    `coastal_upwelling_fog_strength` already accepted for Atacama).

    Swept 0.0/1.0/1.5/1.6/2.5: SE US and S Japan are already decisively fixed
    by 1.0-1.5; East China only reaches a plurality (not majority) Cfa at any
    strength tried, and pushing to 2.5 started overshooting it into Af
    (tropical rainforest, 31%) while introducing the first hint of
    continental-interior bleed (US Midwest showed 4% BSh, was 0% through
    1.6). 1.5 sits before that collateral cost appears -- a real, measured
    partial win for East China specifically, full for S Japan, majority for
    SE US, not a complete fix for all three.

    **RE-CALIBRATED 2026-08-02, 1.5 -> 3.0. Neither documented failure mode
    above reproduces.** Re-tested per ACCURACY_AUDIT.md process note 7 after the
    Koppen area-weighting fix, on the tracked deterministic 64x128 benchmark
    (with `itcz_seasonal_response` at its own new 0.4):

        exempt   groupMAE  tropMAE  refErr  rescale  SE US  E.China  S.Japan  Sahara  Kalahari  Midwest
        1.5       2.265     0.715   0.318    2.659    725     497      630      199     159       976
        2.0       2.164     0.768   0.311    2.370    803     563      774      199     162       977
        2.5       2.055     0.893   0.302    2.138    879     628      911      197     166       978
        3.0       1.969     0.986   0.297    2.008    929     665      993      196     170       985
        3.25      1.848     1.016   0.296    1.977    943     680     1012      197     172       985
        3.5       1.831     1.031   0.294    1.950    956     695     1031      198     174       985
        7.0       2.250     1.299   0.278    1.667   1136     895     1273      202     201       982

    All three monsoon boxes are chronically *far below* target (SE US 1100-1500,
    East China 1300-1800, S Japan 1600-2200), which is why raising this moves
    `reference_error_score` so much. Directly checked for the two collateral
    effects the 1.5 cap was protecting against, and both are absent: East China
    shows **no Af at any strength through 7.0** (it converts BWh -> Cfa, the
    correct direction for a real Cfa climate), and US Midwest's composition is
    byte-stable at Cfa 75% / Cfb 12% / Dfb 12% throughout -- **no BSh bleed at
    all**. The earlier readings came from a 512x1024 5yr run whose Koppen state
    was a lagging 10yr EMA; the fresh-spinup benchmark isolates the mechanism.

    **The real binding constraint is different, and it is the deserts.** Sahara
    holds BWh 82% / BSh 16% through 3.0 (that fringe is from
    `itcz_seasonal_response`, not this knob -- see there) but Kalahari breaks at
    3.25 (BWh 100% -> 92%, BSh 8%) and Sahara's own precip crosses its 200 mm/yr
    ceiling by 5.0. 3.0 is the last value with Kalahari still 100% BWh. Note
    `reference_error_score` keeps falling past that point and is *not* a valid
    guide here: `target_error_fraction` scores only distance *outside* a target
    interval, so it cannot see a desert degrading while it remains under 200.

    At 3.0 every headline metric improves at once rather than trading:
    group MAE 2.246 -> 1.969, tropical MAE 1.369 -> 0.986, refErr 0.3217 ->
    0.2969, and the moisture-budget rescale factor 2.463 -> 2.008 (18% less
    synthetic fill -- the A5 structural metric). Per-box Koppen: S Japan reaches
    **100% Cfa**, SE US 75% Cfa, East China a 42% Cfa plurality (its first, and
    the residual A4 flagged as unclosed). Real-terrain confirmation
    (512x1024, `saves/earth.pkl`, 730d MONTHLY, seasonally balanced, jointly
    with `itcz_seasonal_response` 0.7 -> 0.4): SE US 772 -> 945, East China
    576 -> 780, S Japan 1138 -> 1256 mm/yr, US Midwest and Central Europe flat
    (741 -> 735, 506 -> 508), deserts wetter but in range (Sahara 124 -> 150,
    Kalahari 131 -> 147, Atacama 100 -> 109)."""

    itcz_zonal_smooth_deg: float = 8.0
    """Longitude-direction Gaussian smoothing [degrees] applied to
    `subsidence_suppression` right after its own local Laplacian pass, before
    it gates `precip_potential`/`land_evap` or feeds the desert/continental
    redistribution weight (`atmosphere._zonal_gaussian_smooth`).

    Root cause (2026-07-29, real-terrain transect audit): adjacent longitude
    columns only 5 degrees apart can show completely different ITCZ shapes at
    the same latitude band -- e.g. one an unbroken rainforest belt 18N-6.5S,
    the other a dry notch straddling the equator flanked by two wet peaks --
    even though real terrain/elevation is nearly identical between them (this
    is deep Congo-basin rainforest either way). Traced to the wind model's own
    divergence field: near the equator, Coriolis-based damping vanishes
    (f -> 0) and the deterministic Rossby-mode standing waves
    (`ROSSBY_MODES`, wavenumbers 3/5/7 -- spatial half-periods of
    60/36/25.7 degrees) are the least-damped forcing left, so their
    interference pattern dominates the local divergence signal instead of any
    real geography. Averaged into the 10yr EMA climatology via
    `subsidence_suppression`'s desert/continental redistribution, this bakes
    an essentially arbitrary synoptic-noise pattern into a supposedly
    persistent biome map: tropical savanna (Aw/Am) reads as ~5-6x
    underrepresented and hot steppe (BSh) ~2x overrepresented on real terrain
    (see test-npz-koppen-audit-2026-07-29 memory), because land that should
    grade smoothly from rainforest to savanna instead either stays wet enough
    to read as rainforest or falls straight into steppe with no transition.

    Applied only to the shared `subsidence_suppression` array (not a
    downstream copy) so both the direct precip_potential/land_evap
    suppression and the redistribution weight see the same smoothed signal --
    splitting them risked the two consumers disagreeing on where the ITCZ
    sits. Applied *after* the existing `+0.15*laplacian(...)` pass (a ~1-2
    cell local smooth) and *before* the coastal-fog gate (a deliberately
    narrow ~2-3 cell west-coast feature that a wide zonal smoothing would
    wash out if applied after it).

    0.0 is an exact no-op (`_zonal_gaussian_smooth` returns `field` unchanged
    below half a grid cell of sigma).

    Calibrated 2026-07-29 on real terrain (`saves/test.npz`, 512x1024,
    5yr MONTHLY continuation from a 23.8yr base, instantaneous 2nd-half /
    10yr EMA): swept 0/4/8/12/16 degrees. The 15E/20E transect discontinuity
    (the actual reported bug) is fully resolved by 8 degrees -- both columns
    track each other closely at every latitude instead of one reading as an
    unbroken rainforest block and the other as a dry notch flanked by two wet
    peaks -- with diminishing returns beyond that (named-box precip is nearly
    flat from 8 to 16 degrees, confirming 8 is past the knee of the curve,
    not an arbitrary stopping point). Koppen breakdown moved
    cleanly toward Earth's real values in the same run: arid_pct
    22.1% -> 14.6% (real ~19-20%), tropical_pct (Af/Am/Aw) 8.9% -> 12.3%
    (real ~20%, was measured separately at 3.4% on the pre-fix 23.8yr save in
    test-npz-koppen-audit-2026-07-29 -- a large relative improvement, though
    still short of Earth's share).

    **Correction 2026-08-02: that Koppen sentence is void, though the 8.0 value
    survives.** Those figures were called "area-weighted" but were plain cell
    counts (the bias fixed 2026-08-02 -- see
    `real_terrain_validation._koppen_land_percentages`), and correcting them
    *inverts the sign of that argument*: area-weighted, arid land is ~29% against
    Earth's 26.4%, so the model is over-arid and "22.1% -> 14.6%, real ~19-20%"
    was wrong in both value and direction. Re-swept 0/4/8/12/16 on the tracked
    64x128 benchmark under the corrected metric: raising sigma does keep lowering
    the arid share (group MAE 2.246 at 8.0 -> 2.148 at 12 -> 2.105 at 16), but
    only by trading away tropical accuracy (tropical MAE 1.369 -> 1.763 -> 1.994)
    and `reference_error_score` (0.322 -> 0.324 -> 0.326), while pulling Canadian
    Prairies further above its 400-500 target (530 -> 558 -> 577). **8.0 is kept
    on its original and still-valid justification -- it is the sigma that
    resolves the 15E/20E transect discontinuity, a structural bug -- not on the
    arid-fraction number, which should not be cited for it again.**

    Sahara and the continental-interior boxes
    (Canadian Prairies/US Midwest/Central Europe) are essentially unaffected
    (<3% change). **Real, honest cost**: Atacama gets substantially wetter
    (195 -> 448 mm/yr in the same run) -- its own aridity signature is a
    narrow coastal strip that this same longitude-direction smoothing
    competes with, and per coastal_upwelling_fog_strength's own docstring
    Atacama has never cleanly separated from other deserts on any metric
    tried in this project's history, so this asymmetric trade-off (clear win
    on the reported tropical-belt bug, a real cost to an already-fragile,
    already-off-target desert box) was accepted rather than searching for a
    sigma that protects Atacama, which the 4-16 degree sweep did not find."""

    subsidence_divergence_regime_gate: float = 1.0
    """Strength [0-1] of regime correction for the zonal-background part of
    the wind-divergence signal used by `subsidence_suppression`.

    Root cause (2026-08-01, ACCURACY_AUDIT.md A5/B1 follow-up): the wind
    model's own zonal-mean divergence field peaks at ~38-45 deg N/S instead
    of Earth's ~25-30 deg subtropical high (B1) -- real continental-interior
    latitudes (US Midwest's own box, 38-45N) sit right in that displaced
    peak. `_subsidence_norm_early` reads raw local wind divergence with no
    latitude-regime awareness at all, so it registers Midwest's own spurious
    divergence as if it were real desert subsidence: measured directly
    (`saves/earth.pkl`, 512x1024, 2yr MONTHLY real-terrain continuation),
    US Midwest's `subsidence_suppression` averages 0.196 -- nearly as
    suppressed as the Sahara's own 0.446 mean -- even though Midwest sits far
    outside `drybelt_window`'s ~28 deg peak and should see almost none of
    this gating. Since `subsidence_suppression` gates both `land_evap` and
    `precip_potential` (and shapes the desert/continental `cell_weight`
    redistribution target), this directly fights A3's continental-interior
    shortfall at the same mechanism relied on to fix A1's desert-too-wet gap.

    The first version multiplied the entire local divergence signal by a
    latitude window.  Its gate=1 sweep made Sahara 30% wetter because it also
    erased useful local desert subsidence.  The shipped version decomposes
    divergence into the zonal row mean plus the cell-local anomaly, attenuates
    only the contaminated zonal background outside the true dry belt, and
    preserves local information everywhere.  It is paired with the widened
    subtropical regime and wet-regime raw-conversion calibration documented in
    ACCURACY_AUDIT.md A5/B1.  On the 512x1024 two-year seasonal validation this
    package changes Sahara/Kalahari/Atacama from 586/509/419 to 131/130/102
    mm/yr, Midwest from 480 to 709, and raw rescale from 5.462x to 2.064x.

    0.0 retains the zonal background everywhere; 1.0 applies the full regime
    correction and is the calibrated Earth default."""

    precip_raw_conversion_gain: float = 4.5
    """Peak extra gain applied to raw pre-rescale `precip_potential` in
    *wet* regimes, in `atmosphere.generate_precipitation`'s
    `_raw_conversion_gain` (which evaluates to
    `1.0 + precip_raw_conversion_gain * _raw_conversion_affinity`, so the
    effective multiplier runs 1.0x in the fully dry subtropical regime up to
    5.5x in fully wet regimes at this default).

    This is the A5 fix's central mechanism: raw production was chronically
    ~5.5x below `target_mean_mm_day`, forcing the moisture-budget rescale to
    invent most of the world's rainfall, and calibrating the conversion to the
    atmospheric residence-time deficit in wet regimes -- while deliberately NOT
    amplifying the genuinely low-production dry belt, which is what makes
    deserts deserts -- is what cut the global rescale factor from 5.46x to
    ~2.0x. See ACCURACY_AUDIT.md A5.

    **Why this field exists (added 2026-08-02, ACCURACY_AUDIT.md process note
    6): 4.5 is exactly the value that was already hardcoded, so this is a
    bit-identical no-op change.** The mechanism shipped 2026-08-01 with no
    `PlanetParams` gate at all -- the only large-effect lever in this file to do
    so, breaking the project's own convention that a new mechanism ships behind
    a knob whose neutral value reproduces the prior behaviour. It then broke two
    tests through an interaction nobody predicted (it pushes
    `remove_frac_prerescale` into its 0.85 ceiling in strongly-orographic rows,
    clipping the windward/leeward differential), and diagnosing that needed a
    git bisect precisely because there was no way to ablate it in place.
    `0.0` now restores pre-A5 raw production exactly (gain == 1.0 everywhere)
    and is the ablation to reach for when a precipitation regression is
    suspected of routing through this mechanism.

    Not a tuning knob: 4.5 is calibrated and lowering it re-opens A5's
    structural deficit. It is a diagnostic gate."""

    orographic_uplift_clip: float = 2.0
    """Ceiling applied to the normalized orographic-uplift term `orog` in
    `atmosphere.generate_precipitation`. 2.0 is the value that was hardcoded
    before 2026-08-02, so that default is an exact no-op.

    **Why this is the lever for windward/leeward contrast.** ACCURACY_AUDIT.md
    A5 flagged that `orog`'s within-run percentile normalization
    (`orog / percentile(orog, 90)`) makes the term exactly scale-invariant --
    100m/300m/1000m-per-row ramps give byte-identical results -- and proposed
    making it sensitive to absolute terrain relief instead. Investigated
    2026-08-02, and that framing turns out to be the wrong target for Earth:
    scale-invariance is a *cross-planet* defect, and on a fixed DEM any global
    relief gain is algebraically just a retune of the 0.20 `orog` coefficient in
    `precip_potential` -- it cannot change the spatial contrast at all. (It is
    also not cleanly fixable: measured across 64x128 -> 512x1024, Earth's
    resolved land slope p90 rises 3.3x, 0.0050 -> 0.0165 m/m, while relief per
    grid cell falls 2.4x the other way, so no absolute normalization is
    simultaneously resolution-invariant and relief-sensitive. Topography is
    self-affine; this is information loss at coarse resolution, not an
    implementation choice.)

    The ceiling is the spatially real constraint, and it binds far harder than
    its innocuous look suggests. Measured on `saves/earth.pkl` (512x1024) with
    the real simulated wind field:

        clip   % of all land truncated   % of steepest-5% land truncated
        2.0            19.96                        87.7
        3.0            14.60                          --
        4.0            11.50                          --
        6.0             7.88                          --

    The steepest 5% of land has a mean *pre-clip* `orog` of 11.3 against a
    ceiling of 2.0 -- roughly 80% of the orographic signal was being discarded
    exactly in the Andes/Himalaya/Cascades, which is precisely where A5 hoped a
    fix would show up. The truncation is also resolution-dependent in a way the
    tracked benchmark hides: 9.5% of land at 64x128 versus 20% at 512x1024, so
    the fixture systematically under-represents the problem.

    **Shipped at 2.0 (no-op) because raising it alone is a measured null
    result, and the ablation identified the real binding constraint.** A/B'd on
    real terrain (`saves/earth.pkl`, 512x1024, 730d MONTHLY, seasonally
    balanced), windward-vs-leeward annual precipitation across four ranges,
    clip 2.0 -> 4.0:

        range        W/L ratio 2.0   W/L ratio 4.0   Earth (real)
        Cascades          1.18            1.20         ~3-5x
        S Andes           1.14            1.15         large
        Himalaya          6.32            6.53         large
        Sierra Nevada     1.11            1.10         ~3-4x

    Restoring ~80% of the truncated mountain signal moves real orographic
    contrast by under 2%. Single-step debug-field ablation at the Cascades pair
    shows exactly where it goes, and it is three compounding stages, not one:

      1. **`remove_frac` saturation absorbs it.** Windward cells at the 0.85
         ceiling go from 8.9% (clip 2.0) to **90.0%** (clip 4.0). The extra
         uplift is converted straight into a ceiling hit, so final precipitation
         is *identical* to three digits either way. A5 suspected this
         interaction; this measures it.
      2. **`orog` barely differentiates the pair to begin with**: its own
         windward/leeward ratio is only **1.05**, and because it carries just
         0.20 of the six-term `precip_potential` sum, that sum's ratio is
         **0.93** -- windward comes out *lower* than leeward, the wrong sign,
         swamped by the humidity/convective/ascent terms.
      3. **The row rescale renormalizes what survives**: `zonal_rescale_factor`
         for the Cascades rows is **0.241** (they over-produce against their row
         target, `precip_target_achieved_fraction` 1.018, and get scaled down
         ~4x), the process-note-9 effect -- a raw-production-side change at these
         latitudes is absorbed by a compensating change in the fill.

    So the orographic gap is real and large (model ~1.2x vs Earth's ~3-5x at the
    Cascades, newly quantified here), but **neither the percentile normalization
    A5 proposed nor this ceiling is what binds it.** A genuine fix has to raise
    `orog`'s weight/sharpness relative to the other five terms *and* lift the
    0.85 `remove_frac` cap *and* survive the row rescale -- three coupled
    changes, which is a calibration session of its own, not a one-line change.
    This field stays as the ablation handle for that work.

    **Update 2026-08-02**: stage 2 above ("`orog` barely differentiates the pair
    to begin with") was root-caused, and it was not a property of the formula --
    it was the normalizer feeding this ceiling. See
    `orographic_normalizer_land_only`, which makes this ceiling meaningful for
    the first time."""

    orographic_normalizer_land_only: bool = True
    """Normalize the orographic-uplift term `orog` by the 90th percentile of its
    **land** cells instead of the whole grid. `False` reproduces the historical
    behavior exactly and is the current default.

    **The bug this exists to fix (found 2026-08-02, ACCURACY_AUDIT.md A5).**
    `atmosphere.generate_precipitation` normalizes with
    `orog / (np.percentile(orog, 90.0) + 1e-6)`, but the line immediately above it
    is `orog = land_f * orog`, which sets every ocean cell to exactly 0.0. On
    Earth's DEM ~66% of cells are ocean, so that "90th percentile" is really the
    ~70th percentile of the land distribution, and the normalizer comes out
    **3.8x too small** (0.0101 vs 0.0382 on `saves/earth.pkl`). Every land value
    is inflated by that factor, and `orographic_uplift_clip` then truncates
    **20.0% of all land and 100% of the steepest 5%**.

    The consequence is not a magnitude error, it is **total loss of the
    directional signal**. Traced along the Cascades transect at 46.6N, the raw
    upslope term behaves exactly as intended -- 0.41 climbing to the crest,
    dropping to *precisely* 0.0000 on the lee side -- but after normalization and
    clipping, `orog` reads a saturated **2.000 on both flanks**. The term that
    exists solely to distinguish windward from leeward was reporting the same
    number for both. Same picture on the S Andes transect.

    This is why A5's earlier ablation found raising the ceiling to be a null
    result and concluded `orog`'s own W/L ratio was "only 1.05": with a
    3.8x-inflated input, raising a ceiling from 2.0 to 4.0 just moves where the
    flattening happens. The ceiling was never the binding constraint; the
    normalizer feeding it was.

    Setting this `True` drops clip truncation from 20.0% of land to 5.6% and
    restores real contrast in the term itself. Measured `orog` windward/leeward
    ratios on `saves/earth.pkl`, land-only normalizer with the clip relaxed:

        range           global/2.0   land/2.0   land/4.0   land/inf   Earth
        Cascades             1.05       1.16       1.17       1.35     3-6x
        Sierra Nevada        0.94       0.83       0.90       1.70     2-5x
        S Andes              1.03       1.11       1.32       2.11    5-15x
        Southern Alps        1.06       1.45       1.41       2.23    4-12x
        Scandinavia          1.39       1.49       1.49       1.49     2-4x
        Himalaya             0.80       1.00       1.25       1.49    5-20x

    Note this is the *signal*, measured before any downstream absorption. See
    `scripts/check_orographic_contrast.py`, which reports the ratio at each
    pipeline stage, and A5 for what the end-to-end effect turned out to be."""

    orographic_upwind_footprint_km: float = 0.0
    """E-folding length of the **upstream** footprint given to the orographic
    uplift term, in kilometres. 0.0 = exact no-op (the pointwise term that
    existed before 2026-08-03).

    **The defect this addresses is shape, and it is the only one A5-OROG left
    open.** That session repaired the orographic *signal* (the
    `orographic_normalizer_land_only` bug) and measured every pointwise ceiling
    in the pipeline -- `orographic_uplift_clip`, `precip_potential_ceiling`,
    `precip_rain_out_ceiling`, `precip_orographic_weight`. At the crest the fix
    was large (Cascades windward/leeward 1.03 -> 1.93), but at **box scale --
    the only measure comparable to Earth's published 3-6x ratios -- the mean
    across six ranges was 0.96 before and 0.96 after, i.e. unchanged.** The
    reason is geometric rather than a missing gain: `clip(gx*u + gy*v, 0, None)`
    on a real DEM is a 1-2 cell spike sitting on the crest, so a box mean over
    4-6 cells dilutes a signal that is correct exactly where it exists and
    absent everywhere else in the box. No ceiling can fix that -- they all act
    pointwise. A5-OROG named this parameter's mechanism as the next lever.

    The physics: air begins ascending well upstream of a barrier (upstream
    blocking and deceleration), so uplift is not confined to the cells where the
    resolved slope is steepest. A cell therefore samples the uplift at points
    *downwind* of itself, which places a crest's signal onto the windward flank
    ahead of it. See `atmosphere._smear_along_wind`, and
    `orographic_spillover_km` for the opposite-direction companion.

    **Specified in kilometres, not cells, deliberately.** The equivalent
    cell-count formulation would mean a different physical distance at every
    resolution -- exactly the bug A5 had to fix in the monsoon inland mask,
    whose fixed 20-cell reach meant 7 deg at 1024 columns and 56 deg in the
    128-column fixture.

    **Shipped at 0.0 (inert) because the measured gain is real but small, and it
    is paid for elsewhere.** The mechanism works: it is the first thing tried
    that moves box-scale windward/leeward contrast at all, and the attenuation
    chain shows why -- on `saves/earth.pkl` at 400 km the S Andes `orog` ratio
    goes 1.11 -> 3.14 and now survives *intact* through `precip_potential`
    (3.15) and `remove_frac` (3.09), so the four pointwise ceilings really are no
    longer binding. Measured on the 256x512 fresh-spinup benchmark against the
    annual-mean contrast metric (`metrics.orographic_contrast`):

        footprint_km   mean W/L   grpAcc   clsAcc   refErr   Atacama
        0 (default)      1.567    0.7024   0.4343   0.1290     57
        100              1.628    0.7031   0.4350   0.1309     58
        200              1.649    0.7029   0.4356   0.1357     62
        400              1.646    0.7026   0.4345   0.1399     66

    That is +5% of contrast against Earth's 3-6x, for `reference_error_score`
    +0.007 and Atacama drifting back up toward the <50 target it had just come
    under. Saturates by ~200 km. The H10 accuracies move by less than a tenth of
    a percent either way, i.e. within the noise that process note 14 says a
    single grid cannot resolve.

    **What remains binding is the moisture budget's deficit fill, not the uplift
    signal.** The same S Andes pair whose `remove_frac` ratio reaches 3.09 comes
    out at 1.37 in final precipitation: `precip_orographic_shape_weight` blends
    raw shape into the target but its gate saturates at
    `orog >= orographic_uplift_clip`, so a broader footprint does not widen it
    proportionally. Raising that weight to 2.2 alongside a 200 km footprint is a
    genuinely large regional win -- mean W/L 1.567 -> **1.799**, Atacama
    **57 -> 39.5 mm/yr, under its <50 target for the first time**, Sahara
    164 -> 148, `reference_error_score` 0.129 -> 0.123 -- but it reproduces
    A5-LEAD's verdict on that knob exactly (group accuracy 0.7024 -> 0.6983,
    class 0.4343 -> 0.4291 at 256x512, matching A5-LEAD's own 0.6985 for
    weight 2.2), so the trade is the shape weight's, not this parameter's.
    Whoever revisits that trade should reach for this footprint too; it is the
    piece that makes the raw signal worth propagating."""

    orographic_spillover_km: float = 0.0
    """E-folding length of the **downstream** spillover of the orographic uplift
    term, in kilometres. 0.0 = exact no-op.

    The companion to `orographic_upwind_footprint_km`, in the other direction:
    condensate formed in ascent is carried some distance downstream before it
    reaches the ground, so a cell also samples the uplift at points *upwind* of
    itself. Physically real, but it works against windward/leeward contrast by
    construction (it moves signal toward the lee), which is why it is separately
    gated rather than folded into one symmetric smoothing length."""

    precip_rain_out_ceiling: float = 0.85
    """Maximum fraction of a cell's moisture column that can rain out in one
    step, in `atmosphere.generate_precipitation`. 0.85 is the value hardcoded
    before 2026-08-02, so that default is an exact no-op.

    This is stage 1 of A5's three absorption stages, and it is real: measured on
    `saves/earth.pkl`, the **entire S Andes windward slope** (-75.1 through
    -72.2) sits pinned at 0.85, so no upstream orographic gain can raise its
    precipitation -- it is already stripping the maximum permitted fraction of
    its column. Globally 4.7% of land is pinned, but the pinning is concentrated
    precisely on the wettest windward flanks.

    The original 0.85 has a real justification that any change has to respect:
    at dt=6 an uncapped removal fraction clips to 1.0, stripping the column
    completely, leaving `humidity_next` ~= 0 everywhere and erasing the spatial
    humidity gradients the next substep needs. Raising this trades windward
    dynamic range against that stability margin -- it is not free headroom."""

    precip_orographic_weight: float = 0.20
    """Weight of the orographic-uplift term `orog` in `precip_potential`'s
    six-term sum. 0.20 is the historical hardcoded value and an exact no-op.

    A5's stage 2. The other five terms are rescaled so the total weight stays at
    its historical 1.10, making this a knob on orography's *relative* share
    rather than on the magnitude of `precip_potential` -- magnitude alone is
    absorbed by the moisture-budget rescale, so only the relative share can move
    spatial contrast.

    The stage is real: even with `orographic_normalizer_land_only` restoring the
    signal, `orog`'s windward/leeward ratio at the Cascades (1.16) is diluted to
    0.92 by the time it is summed with the humidity/convective/ascent terms,
    which carry 0.90 of the weight between them and are not orographically
    organized. Raising this alone is not sufficient, though -- see
    `precip_orographic_shape_weight` for the stage that actually binds."""

    precip_potential_ceiling: float = 3.0
    """Ceiling on `precip_potential` in `atmosphere.generate_precipitation`,
    applied inside its smoothing loop and again after the raw-conversion gain.
    3.0 is the historical hardcoded value and an exact no-op.

    **A5's fourth absorption stage, which the audit's original three-stage
    analysis missed.** It is invisible to a magnitude check and only shows up in a
    contrast one: with the orographic normalizer fixed, `precip_potential`'s
    windward/leeward ratio saturates at **~2.9 whether the incoming `orog` ratio
    is 3.0 or 11.7**, because the windward cell is pinned at this ceiling while
    the leeward cell is not. Raising `precip_orographic_weight` against a bound
    ceiling then *lowers* the ratio, since it can only lift the unpinned leeward
    side -- which is why that knob measured backwards before this was found."""

    precip_orographic_shape_weight: float = 1.0
    """Blend raw production shape into the moisture-budget *target* where
    orography is active, 0.0 = exact no-op.

    **A5's stage 3, and the one that actually binds.** The moisture-budget
    rescale is a deficit-filling mechanism: it tops each cell up toward a target,
    so wherever the fill supplies most of a row's rain -- which is nearly
    everywhere, `global_rescale_factor` averages ~2.0 even after A5 -- it
    actively *erases* whatever spatial contrast raw production created. Measured
    on `saves/earth.pkl`: the S Andes pair reaches a `precip_potential`
    windward/leeward ratio of 1.69 and comes out at **0.88 in final P**. Not
    merely damped -- inverted, because the leeward flank under-produces against
    its target and is therefore handed more synthetic fill than the windward one.

    This is process note 9 ("check which side of the rescale a proposed mechanism
    sits on") applied to orography: the target is the lever, so an orographic
    mechanism that only touches raw production cannot work no matter how strong
    it is. `precip_raw_shape_weight` already implements exactly the needed blend,
    but is gated by `itcz_window` and therefore cannot reach a mid-latitude
    mountain range at all. This parameter applies the same blend gated by the
    orographic signal instead, so it acts only where orography is doing something
    and leaves the hard-won desert/continental mid-latitude weighting alone.

    Mean-preserving by construction (renormalized into `cell_weight` exactly like
    `_desert_factor`), so row totals are unaffected."""

    precip_land_shape_weight: float = 0.0
    """Blend raw production shape into the moisture-budget *target* over **all**
    land, at every latitude, ungated. 0.0 = exact no-op (the default).

    The third and broadest gating of the one shared `_raw_shape` blend, alongside
    `precip_raw_shape_weight` (gated by `itcz_window`) and
    `precip_orographic_shape_weight` (gated by the orographic signal). All three
    compose multiplicatively and each is an exact no-op at 0.0.

    **Why this exists as its own knob, and why it is expected to stay at 0.0**
    (2026-08-02, audit A5-LEAD). A5-OROG deferred a "large lead": with the
    orographic normalizer bug left in place its gate covers 87.8% of land at an
    area-weighted mean of 0.36 -- not an orographic mechanism at all -- and run
    that way the tracked 64x128 benchmark beat the shipped configuration on
    `reference_error_score` *and* both H10 accuracies. That lead identified
    itself as "effectively the ungated raw-shape target blend", and asked for a
    session testing it honestly as a land-wide mechanism with its own gate. This
    parameter is that mechanism. **The test refuted the lead.**

    Swept, it degrades every bounded metric monotonically (group accuracy 0.6855
    -> 0.6723 -> 0.6577 at 0.0/0.45/1.0, class accuracy and share MAE likewise)
    while `reference_error_score` improves to a minimum at 0.45 -- process note
    10 exactly. At 1.0 it reproduces the **2026-07-31 rejection** of the ungated
    `precip_raw_shape_weight` almost exactly: `arid_pct` 27.5 -> 33.2% and US
    Midwest 900 -> 399 mm/yr, the two effects that session gated that mechanism
    to the ITCZ to avoid. So that rejection stands, contrary to the lead's claim
    that it "does not reproduce" -- it does not reproduce for the *buggy
    orographic* gate, which is terrain-weighted, but it does for a uniform
    land-wide blend. The two are not the same mechanism.

    Kept rather than reverted (process note 2) so the next session to reach for
    "blend raw production shape into the target over all land" finds a tested
    mechanism and the audit's sweep table instead of rebuilding it."""

    itcz_seasonal_response: float = 0.4
    """Fraction [0-1] of the full solar-declination swing (`solar_declination`)
    that the ITCZ's *center latitude* (not its width) tracks seasonally in
    `atmosphere.generate_precipitation`'s `itcz_window`. 0.0 is an exact no-op:
    the belt stays pinned to the equator year-round, the behavior at every
    prior default.

    Root cause (2026-07-30, real-terrain-vs-reference-Koppen-map audit,
    direct follow-up to the same-day handoff-gap/monsoon-margin sessions):
    `itcz_window` was `exp(-(|lat|/ITCZ_HALF_WIDTH_DEG)**2)` -- a pure
    function of `|latitude|`, with zero `day_of_year` dependence despite
    `day_of_year` already being threaded through the whole call correctly.
    In reality the ITCZ is the ascending branch of the Hadley circulation
    tracking the sub-solar latitude (damped by ocean thermal inertia, not the
    full ~23.4 deg swing) -- it sits over a given savanna latitude for part of
    the year (wet season) and displaces to the other hemisphere the rest of
    the year (dry season). A static, equator-locked ITCZ instead rains on
    every latitude within its Gaussian footprint essentially every month,
    which the Koppen classifier reads as "no dry month" -- Af (rainforest,
    requires driest month >= 60mm) instead of Aw (savanna, requires a real dry
    season). Measured directly on `saves/test.npz` (512x1024, 10yr MONTHLY
    continuation): a lon=10E transect held every sampled latitude from 14N to
    -2S at a 80-170mm driest-month floor -- no latitude in that whole band
    ever showed a real dry season. Area-weighted land Koppen breakdown: Af
    21.5% of land (Earth ~6-7%), Aw+Am 2.4% (Earth ~18-20%) -- worse than any
    previously-recorded measurement of this same known symptom (see
    known-physics-gaps.md item 3b and the itcz-global-rescale-coupling/
    itcz-rossby-noise memories), because none of those sessions had found
    this specific mechanism: every prior fix addressed the belt's *width*/
    *noise*, not that it never moves at all.

    Implementation: `itcz_center_deg = itcz_seasonal_response *
    degrees(pp.solar_declination(day_of_year))`, then `itcz_window` is
    `exp(-(((lat_deg - itcz_center_deg)/ITCZ_HALF_WIDTH_DEG)**2))` using
    signed (not absolute) latitude. Generalizes across obliquity via
    `solar_declination` (works for Mars too, though Mars's precipitation
    pathway is already near-inert with `has_liquid_water_ocean=False`, so
    this knob makes no meaningful difference there either way -- not
    special-cased, same treatment as every other Earth-calibrated precip/wind
    parameter already inherited by `MARS`, e.g. `ferrel_v_land_shift_deg`/
    `coastal_upwelling_fog_strength`). `storm_window`/
    `drybelt_window` are untouched (subtropical/storm-track features this
    session did not investigate for their own seasonal migration).

    **Calibration (2026-07-30, `saves/test.npz`, 512x1024)**: swept 0.0/0.4/
    0.7/1.0 over a 5yr MONTHLY continuation (area-weighted land Koppen,
    named-box precip via `climate_precip_avg`):

        response   Af      Aw     Sahara  Kalahari  Atacama  Can.Prairies  US Midwest  Cent.Europe
        0.0        21.71%  2.52%  707     524       423      603           727         679
        0.4        21.17%  2.89%  703     523       422      602           727         679
        0.7        20.36%  3.50%  698     520       420      602           727         679
        1.0        20.29%  3.50%  --      --        --       --            --          --

    (**The sweep above is cell-count-biased and its "0.7" conclusion no longer
    holds -- see the 2026-08-02 recalibration at the end of this docstring.**)

    Monotonic in the right direction (Af down, Aw up) at every tested value,
    essentially zero cost to every desert/continental-interior box (<2%
    change, well within run-to-run noise), and saturates between 0.7 and 1.0
    -- 0.7 sits right at that knee. Also confirmed directly at the
    *instantaneous* (non-EMA) level, not just via the slow-converging 10yr
    Koppen average: a 12-month probe at lon=10E, response=0.7 vs 0.0, shows a
    real (if modest) seasonal cycle appearing where none existed before --
    e.g. 14N driest-month precip 78->70mm/month, 10N 125->117mm/month, with
    the wet-season peak simultaneously rising (10N max 132->141mm/month).

    **Honest limitation, not a full fix**: this real, measured, zero-cost
    improvement is far short of Earth's Af~6-7%/Aw~18-20% targets, and the
    driest-month values above mostly don't cross the 60mm Af/Aw threshold
    even at response=1.0. Root cause of the shortfall (read from code, not
    yet independently isolated by a controlled A/B this session): this
    project's own already-documented moisture-budget global rescale
    (`known-physics-gaps.md` item 3b's "per-cell not per-row" limitation,
    `itcz-global-rescale-coupling-2026-07` memory) aggressively fills any
    row's precipitation deficit toward its target-mean profile every step --
    exactly the kind of month-to-month dip this fix now introduces at
    savanna latitudes is a prime target for that same deficit-filling to
    partially refill. Fixing that structural rescale mechanism (or excluding
    a genuine seasonal ITCZ dip from being treated as a "deficit" worth
    filling) is the next lever, and was already flagged in prior sessions as
    a larger, currently-blocked undertaking -- this fix does not attempt it,
    but does correct the specific, previously-undiscovered bug of the ITCZ
    never moving at all, which is unambiguously wrong physics on any tilted
    planet regardless of how much of the area-fraction gap it closes alone.

    **RE-CALIBRATED 2026-08-02, 0.7 -> 0.4.** The 2026-07-30 sweep above chose
    0.7 as "the knee" from Af/Aw shares computed as plain cell counts (the
    measurement bug fixed 2026-08-02 -- see
    `real_terrain_validation._koppen_land_percentages` and ACCURACY_AUDIT.md
    process note 7, which says in as many words to re-run the sweeps that
    justified a cap after a structural change). Re-swept against the corrected
    area-weighted metric on the tracked deterministic 64x128 benchmark, with
    `itcz_seasonal_target_response` held at its own 2026-08-02 value of 2.0:

        response   Af      Am     Aw     tropMAE  groupMAE  refErr  Sahara  Midwest
        0.0        21.86   0.10    1.40   9.455     2.315    0.316    201     974
        0.4         5.89   4.88   11.16   0.715     2.265    0.318    199     976
        0.7 (was)   6.82   6.22    8.94   1.369     2.246    0.322    192     982
        0.85       10.60   5.96    5.48   3.693     2.245    0.320    190     987
        1.0        14.51   2.67    5.21   4.877     2.351    0.317    189     996
        (Earth: Af ~6.0%, Am ~4.0%, Aw ~10.0% of land area; tropMAE is the mean
         absolute error against those three.)

    The response is strongly non-monotonic under the corrected metric -- the old
    unweighted numbers had read it as saturating past 0.7, which it does not --
    and 0.4 roughly halves the tropical error (1.37pp -> 0.72pp) while also
    *lowering* `reference_error_score` (0.3217 -> 0.3178), i.e. it partially
    repays the zonal-magnitude cost `itcz_seasonal_target_response`'s own
    docstring records as an accepted trade. A 2D grid over
    (`itcz_seasonal_response`, `itcz_seasonal_target_response`) confirmed the two
    knobs are coupled and that the optimum region is broad and shallow
    (0.3-0.45 x 2.0-2.3, tropMAE 0.63-0.72); 0.4 with the target knob left at its
    already-calibrated 2.0 was preferred over chasing the 0.64 minimum, so only
    one parameter moves and attribution stays clean.

    **0.4 is also the more physically defensible value**, which is why this is
    not just metric-fitting: Earth's zonal-mean ITCZ migrates roughly +/-5-8
    degrees against a 23.44-degree declination swing (ratio ~0.25-0.35, heavily
    damped by ocean thermal inertia), so 0.7 was an over-migration that happened
    to score well on a biased statistic.

    **Real, accepted cost**: the deeper seasonal cycle wets the deserts slightly
    (benchmark Sahara 192 -> 199 mm/yr, real-terrain 512x1024 124 -> 150,
    Kalahari 131 -> 147 -- all still inside their <200 target). Verified on real
    terrain (512x1024, `saves/earth.pkl`, 730d MONTHLY, seasonally balanced,
    jointly with `monsoon_east_margin_exemption` 1.5 -> 3.0): area-weighted arid
    land 30.6% -> 27.4% (Earth 26.4%) and tropical 17.7% -> 18.6% (Earth 19.0%).

    On the 64x128 benchmark this also puts a BSh fringe on the Sahara box
    (BWh 98% -> 82%). **That fringe is a coarse-grid artifact, not a real cost**:
    re-run at 128x256 the same change leaves the Sahara at BWh 98% (BSh 1%) and
    the Kalahari at BWh 100%, with US Midwest / Central Europe / Atacama
    compositions byte-identical. Worth remembering as an instance of the general
    rule -- a 64x128 named box holds only a handful of land cells, so a single
    reclassified cell reads as a double-digit percentage swing."""

    drybelt_seasonal_response: float = 0.25
    """Fraction [0-1] of the full solar-declination swing that the **subtropical
    dry belt's** poleward edge tracks seasonally, in
    `atmosphere.generate_precipitation`'s `drybelt_window` /
    `drybelt_regime_window`. 0.0 is an exact no-op (the static belt every default
    before 2026-08-03 used).

    **This is the third instance of the same bug class, and the one that carries
    Mediterranean climate.** `itcz_window` was found in 2026-07-30 to be a pure
    function of `|latitude|` with zero `day_of_year` dependence, and
    `_zonal_precip_target_profile` the same on 2026-07-31 (see
    `itcz_seasonal_response` / `itcz_seasonal_target_response`, whose docstrings
    both explicitly record that `storm_window`/`drybelt_window` were left
    untouched as "features this session did not investigate"). They are the same
    defect: on a tilted planet the subtropical high migrates with the sun, and a
    belt pinned to a fixed `|latitude|` cannot produce a summer-dry climate.

    **Why it is the specific blocker for Csa/Csb** (H10-DONE, ACCURACY_AUDIT.md,
    and the `missing-koppen-classes-rootcause` diagnosis it prompted): the model
    emitted *zero* Mediterranean cells. `climate_averages.classify_koppen`
    reaches Cs only when `P_summer_driest < P_winter_wettest / 3`, and land
    precipitation in this model peaks in local *summer* essentially everywhere,
    because land precip tracks land surface temperature through evaporation and
    nothing opposes it. Real Mediterranean climate is precisely the opposite
    forcing winning: the subtropical high sits over 35-42 deg in summer and
    suppresses convection outright, then retreats equatorward in winter and lets
    the storm track in. With a static belt that seasonal alternation does not
    exist anywhere in the code, so the ratio is not merely too weak -- it is
    inverted, and no classifier threshold change can recover the class.

    **Hemisphere-antisymmetric, unlike the ITCZ knob.** `itcz_window` is built on
    signed latitude around a single centre, so one shift moves the whole belt.
    This belt is symmetric about the equator (`abs_lat_deg`), and its two halves
    migrate in *opposite* directions at any given time: in NH summer the NH belt
    moves poleward while the SH belt moves equatorward. Implemented as
    `abs_lat_deg - k * declination_deg * sign(lat_deg)`, which reproduces that
    with one term and feeds the Gaussian and the wide 16-34 deg regime window
    identically.

    Physically, Earth's subtropical ridge migrates ~5-10 deg against the 23.44
    deg declination swing, i.e. a ratio of ~0.25-0.4 -- the same range
    `itcz_seasonal_response` was independently calibrated into. 0.25 is a 5.9 deg
    migration, at the conservative end of that.

    **Calibration (2026-08-03), with `drybelt_seasonal_equatorward_fraction=0.0`
    throughout.** Swept on the 128x256 fresh-spinup benchmark; `Csa` is the
    area-weighted land share of the class the model previously could not emit
    (Earth 1.94%):

        response  grpAcc  clsAcc  kappa   shareMAE   Csa    Sahara  Midwest
        0.00      0.7031  0.4194  0.6236    2.18    0.01     129      912
        0.15      0.7062  0.4227  0.6275    2.15    0.68     131      855
        0.20      0.7068  0.4224  0.6283    2.13    1.11     131      827
        0.25      0.7082  0.4223  0.6303    2.05    1.58     131      811
        0.30      0.7078  0.4200  0.6298    1.99    1.87     132      788

    0.25 is both the group-accuracy/kappa optimum and the largest value that
    leaves **US Midwest inside its 800-1000 target** at that resolution; 0.30
    buys another 0.3pp of Csa and pushes it out.

    **Confirmed at three resolutions** (audit process note 14 -- a bounded-metric
    gain read only at 64x128 has reversed sign at higher resolution before):

        grid      grpAcc            clsAcc            kappa             Csa
        64x128    0.6855 -> 0.6864  0.3884 -> 0.3943  0.6021 -> 0.6033  0.11 -> 0.97
        128x256   0.7031 -> 0.7082  0.4194 -> 0.4223  0.6236 -> 0.6303  0.01 -> 1.58
        256x512   0.7024 -> 0.7049  0.4343 -> 0.4335  0.6223 -> 0.6259  0.00 -> 1.93

    Group accuracy and kappa improve at all three; class accuracy improves at two
    and is flat at the third. Deserts are untouched everywhere (Sahara 164->165 /
    129->131 / 164->166; Atacama 65->64 / 61->60 / 57->56).

    **Real, accepted cost: the continental-interior boxes, and it grows with
    resolution.** US Midwest 900->841 (64x128), 912->811 (128x256), 878->786
    (256x512) and Central Europe 791->785 / 568->554 / 581->548 -- the last two
    dip just under their 550 floor. The monsoon boxes also give back (S Japan
    1115->979 at 64x128, 2073->1875 at 256x512). Both are the same mechanism
    seen from the other side: a *zonally uniform* belt that advances poleward in
    summer necessarily suppresses summer rain over the 35-45 deg eastern margins
    too, where real monsoon inflow keeps it. That is the identical tension
    `monsoon_east_margin_exemption` exists for (audit A4), and the honest next
    lever for it. `reference_error_score` registers the cost as +0.0025 at
    64x128 and +0.0006 at 256x512.

    **A measured regional artifact worth knowing about before re-deriving it.**
    Of six real Mediterranean regions the mechanism fixes five (Greece, coastal
    California, central Chile, Cape Town, SW Australia all develop a proper
    winter-wet cycle -- e.g. California goes from a flat 22-36 mm/month all year
    to 70 in January and 4.8 in July). **Iberia inverts instead**, developing a
    285 mm/month June peak. Root-caused, not left as a puzzle: it is
    `monsoon_east_margin_exemption`, which sees the Mediterranean Sea to Iberia's
    east and exempts it from the dry belt. Setting that knob to 0.0 restores
    Iberia's proper cycle (Jan 85, Jun 38, Nov 103) and confirms the cause. The
    spike is that large because the moisture budget's redistribution is
    mean-preserving within a row: once the seasonal belt suppresses the rest of
    37-41N, an unsuppressed strip inherits nearly the whole row's synthetic fill.
    The exemption itself is calibrated at 3.0 for SE US/East China/S Japan and
    was deliberately not disturbed here."""

    drybelt_seasonal_equatorward_fraction: float = 0.0
    """How much of the dry belt's poleward-edge seasonal migration its
    *equatorward* edge follows, in `atmosphere.generate_precipitation`. 1.0 is a
    rigid translation (the whole belt slides); **0.0, the shipped default**, pins
    the equatorward edge so the belt widens into the summer hemisphere instead.
    Inert whenever `drybelt_seasonal_response` is 0.0.

    **Why the rigid translation is wrong, measured rather than assumed.** At
    `drybelt_seasonal_response=0.3` with this at 1.0, the belt leaves 15-22N for
    half the year and takes `subsidence_suppression` with it: the Sahara box goes
    **129 -> 223 mm/yr**, straight through its <200 target, and US Midwest 912 ->
    782 out of its own range. The audit's A1 names the desert boxes as the
    binding constraint on any tropical/monsoon tuning, and a rigid belt spends
    all of that budget at once.

    The physics says the same thing: the Hadley cell's descending branch is
    bounded on its equatorward side by the ITCZ, which migrates only ~5-8 deg,
    while its poleward edge advances much further into the summer hemisphere. The
    belt therefore *widens* seasonally rather than sliding, which is exactly what
    keeps the Sahara dry year-round on the real Earth while still giving 35-42 deg
    a summer under the ridge and a winter under the storm track.

    Pinning the edge costs nothing that the rigid version bought: at response 0.3
    the two score group accuracy 0.7078 (pinned) vs 0.7048 (rigid) and class
    accuracy 0.4200 vs 0.4125, i.e. the pinned form is better on both, while
    Sahara stays at 132 instead of 223 mm/yr.

    Implemented as a single latitude-dependent shift so the Gaussian and the
    wide regime window inherit it together. A shift that varies with latitude can
    fold the coordinate onto itself and give the belt a spurious second peak, so
    the transition widens with the shift magnitude to keep the warp's slope below
    1 for any response in [0, 1]; at the shipped 0.25 that widening is inactive.
    See `testing/test_seasonal_belts.py::test_belt_never_develops_a_second_peak`,
    which checks past every shipped value because these knobs exist to be
    swept."""

    storm_track_seasonal_response: float = 0.0
    """As `drybelt_seasonal_response`, but for `storm_window` (the mid-latitude
    storm-track latitude weight, centred at `STORM_TRACK_CENTER_DEG`). 0.0 is an
    exact no-op.

    Separately gated rather than folded into one knob because the two belts feed
    different terms and carry independent calibration history: `drybelt_window`
    gates `subsidence_suppression`, `land_evap` and the desert redistribution
    (i.e. it is load-bearing for A1's desert targets), while `storm_window` only
    adds mid-latitude weight to the humidity/convergence/ascent drivers. Keeping
    them separable is what makes an ablation attribute a change to one or the
    other.

    Both migrate together on Earth (the storm track rides the poleward edge of
    the Ferrel cell), so a physically coherent configuration moves this roughly
    in step with `drybelt_seasonal_response`.

    **Ships at 0.0 on measurement, not by omission.** At 0.3 on the 128x256
    benchmark it leaves every Koppen skill metric and every desert unmoved (group
    accuracy 0.7031 -> 0.7030, Sahara 129.40 -> 129.38, arid share 26.86 ->
    26.88, Csa unchanged at 0.01%). Its one visible effect is Central Europe
    568 -> 608 mm/yr, still inside its 550-750 target -- real, but unrelated to
    the Mediterranean gap this pair of knobs was built for, so it is left for a
    session with a reason to move it. The dry belt carries the entire effect."""

    itcz_seasonal_target_response: float = 2.0
    """Strength [0-1ish] of the fix for `itcz_seasonal_response`'s own flagged
    "honest limitation" (see that field's docstring): closes the gap where the
    moisture-budget/zonal-rescale target in `atmosphere.generate_precipitation`
    (`target_row_mm_day`, from `_zonal_precip_target_profile`) was a pure |lat|
    shape with **zero `day_of_year` dependence**, so every month it pulled a
    savanna-latitude row back toward the SAME flat annual-mean target regardless
    of season -- exactly refilling the seasonal dry-season dip
    `itcz_seasonal_response` introduces into the raw precip signal, before it
    ever reaches Koppen classification. 0.0 is an exact no-op (prior behavior).

    Implementation: `target_row_mm_day *= 1 + k*(itcz_window(day) -
    itcz_window_annual_mean)`, where `itcz_window_annual_mean` (see
    `atmosphere._itcz_window_annual_mean`) is the true time-average of
    `itcz_window` over a full seasonal cycle, computed once per (grid
    resolution, `itcz_seasonal_response`, `obliquity_deg`) and cached. Because
    the modulation's own time-average is exactly 1.0 by construction, each
    row's calibrated *annual-mean* target -- and therefore every existing
    desert/continental/latitude-band calibration built on it -- is unaffected
    in the long run; only the within-year distribution moves, which is
    specifically what lets a real dry season register as "near its own target"
    instead of "in deficit" and stop being force-filled. Naturally near-inert
    at `itcz_seasonal_response=0.0` (itcz_window is then time-invariant, always
    equal to its own mean) or far from the tropics (both the window and its
    mean shrink together there).

    Only meaningful when `itcz_seasonal_response > 0.0` (gated in code; the
    modulation is a hard no-op otherwise regardless of this value).

    **Calibration (2026-07-31, `saves/test.npz`, 512x1024, 5yr MONTHLY
    continuation, area-weighted land Koppen split + named-box precip via
    `climate_precip_avg`)**:

        k     Af      Am     Aw     arid_pct  Sahara  Kalahari  Atacama  Can.Prairies  US Midwest  Cent.Europe
        0.0   11.24%  0.00%  2.00%  12.3%     612     489       444      606(EMA 605)  711(695)    679(671)
        0.3   11.70%  0.01%  1.68%  --        611     489       446      --            --          --
        0.6   10.85%  0.04%  2.63%  --        610     489       447      --            --          --
        1.0    7.98%  0.72%  4.97%  12.5%     555(610) 509(508) 454(475) 606(EMA)      695(EMA)    671(EMA)
        1.5    3.40%  2.47%  6.98%  13.7%     539(608) 517(508) 455(475) 581(EMA)      696(EMA)    671(EMA)
        2.0    0.27%  2.10%  9.64%  15.3%     522(606) 524(509) 453(473) 581(EMA)      696(EMA)    670(EMA)

    (desert-box figures are `instantaneous, 2nd half` with the `10yr EMA` in
    parens, matching `scripts/check_real_terrain_koppen.py`'s own convention.)

    Monotonic and substantial in the right direction through the whole swept
    range (Af down, Aw+Am up, total A-climate land-fraction roughly conserved
    at k<=1.0 as intended -- cells are converting Af->Aw/Am within the tropics,
    not leaving it), but **arid_pct started climbing measurably past k=1.0**
    (12.3%->12.5%->13.7%->15.3% at k=0/1.0/1.5/2.0) while every named
    desert/continental EMA box stayed flat through k=1.0 and Kalahari/Atacama
    drifted up (~4-7%) beyond it -- read at the time as an overshoot where an
    excessively deep dry season pushes savanna-adjacent cells over Koppen's
    aridity-index threshold into steppe (BSh) instead of stopping at Aw. On
    that basis k=1.0 was chosen as "the knee".

    **RE-CALIBRATED 2026-08-01 to 1.7** (deterministic 64x128 benchmark,
    `real_terrain_validation.RealTerrainValidationConfig()` defaults, fresh
    spinup -- the tracked instrument, not the 10yr-EMA-lagged save above):

        k     Af%     Aw+Am%   arid%   Sahara  Kalahari  Atacama  Midwest  refErr
        1.0   11.05   2.25     17.67   195     157       146      979      0.3185
        1.5    7.79   5.07     18.15   194     156       146      981      0.3213
        1.6    7.01   5.73     18.23   193     156       146      981      0.3219
        1.7    6.07   6.65     18.23   193     156       146      981      0.3226
        2.0    3.95   8.71     18.29   192     155       146      981      0.3245

    **The k>1.0 blocker above no longer applies, and re-testing it was the
    point.** That 2026-07-31 reading rejected k>1.0 because arid_pct rose *while
    desert boxes drifted wetter* -- i.e. the extra aridity was being manufactured
    at the tropical margin rather than in real deserts. Post-A5 (see
    `docs/ACCURACY_AUDIT.md` A5, which fixed raw production and made
    `subsidence_suppression` regime-aware) that failure mode is simply absent:
    across this whole sweep Sahara/Kalahari/Atacama are flat-to-slightly-*drier*
    and every continental box is flat, while arid_pct moves 17.67%->18.23%
    *toward* Earth's ~19-20% rather than past it. A5 also moved arid_pct's
    starting point from ~12% up to ~17.7%, which is why the old sweep's
    "climbing arid" signal read as overshoot then and reads as convergence now.

    1.7 puts **Af at 6.07%, inside Earth's ~6-7%** (from 11.05%, a longstanding
    ~1.7x over-representation) and nearly triples Aw+Am (2.25%->6.65%), with no
    measurable desert/continental cost on the benchmark *or* on real terrain
    (512x1024, 2yr MONTHLY seasonally-balanced: Sahara 131->126, Midwest
    709->705, Central Europe 502->502, S Japan 1138->1138 -- all within noise).

    **Real, accepted cost**: `reference_error_score` 0.3185->0.3226 (+1.3%),
    driven entirely by the 10-20N zonal band's precip ratio (0.768->0.714). That
    band is the savanna belt itself, and deepening its dry season genuinely lowers
    its annual mean. Note the modulation stops being exactly mean-preserving above
    k~1.0 because the `clip(0.05)` floor truncates the dry-season trough while the
    wet-season boost stays moisture-budget-limited. This is a deliberate trade of
    a zonal *magnitude* metric (which contains no biome information -- refErr is
    only zonal temperature bias + precip ratio) for a large, on-target improvement
    in Koppen biome classification against the project's designated reference map.
    Revisit if zonal precip magnitude ever becomes the priority over biome shape.

    **RE-CALIBRATED AGAIN 2026-08-02, 1.7 -> 2.0, after fixing a measurement
    bug.** Everything above (and the "1.7" conclusion) was calibrated against
    Köppen shares computed as *plain cell counts*, which on an equirectangular
    grid over-weight the poles severely -- see
    `real_terrain_validation._koppen_land_percentages`, now area-weighted. Under
    the corrected metric the picture changes materially, because tropical cells
    had been systematically under-counted:

        k     Af      Am      Aw     MAE vs Earth   refErr
        1.7   10.70%  4.23%   7.10%      2.61pp     0.3195
        2.0    6.82%  6.22%   8.94%      1.37pp     0.3217
        2.2    4.84%  6.13%  10.97%      1.42pp     0.3233
        2.4    3.14%  6.43%  12.28%      2.52pp     0.3248
        (Earth: Af ~6.0%, Am ~4.0%, Aw ~10.0% of land area)

    2.0 minimises the error against Earth's real Af/Am/Aw split. Note the old
    unweighted metric had reported k=2.0 as "overshooting Af to 3.95%, below
    Earth's 6-7%" -- area-weighted, k=2.0 puts Af at 6.82%, essentially on
    target. The previous 1.7 stopping point was an artifact of that bias.
    Deserts and continental boxes stay flat across this range (Sahara 193->192,
    US Midwest 980->982), and `reference_error_score` rises 0.3195->0.3217 --
    the same zonal-magnitude-vs-biome-accuracy trade documented above, now with
    a trustworthy biome metric on the other side of it.

    **Honest limitation, unchanged**: this knob redistributes *within* the
    tropical band (A total stays ~22% across the whole sweep); it cannot grow
    the band. That is fine -- area-weighted, A total is 22.0% against Earth's
    19.0%, i.e. the band was never actually too small. The old "Aw+Am is 2-4x
    under-represented" framing was itself a product of the counting bug.

    **EVERY CALIBRATION ABOVE IS INVALID (2026-08-05).** All three sweeps read
    Af/Am/Aw off runs of two simulated years or less (the tracked benchmark's
    `spinup_years=1.0` + `evaluation_years=1.0`, or a short MONTHLY
    continuation). `climate_averages.update_monthly_statistics` used to *blend*
    its spin-up seed -- all 12 bins set to one instantaneous field, i.e. a
    zero-amplitude annual cycle -- into the monthly bins, leaving
    `(1-alpha)**n` of it behind. At `window_years=1.0` that is ~13.5% still
    present after two years, and being flat it landed almost entirely on the
    bin that should have been the year's driest: worth ~+23 mm/month in the
    deep tropics, enough by itself to carry Amazon/Congo/Borneo over the 60 mm
    Af threshold. Measured at 256x512, same params, only spin-up changed:

        spin-up        Af      driest-month median   wet/dry ratio
        1yr + 1yr    6.02%          29.4 mm              6.1
        5yr + 1yr    3.82%           9.2 mm             21.9
        (24yr save)  2.61%           8.2 mm             24

    So "k=2.0 puts Af at 6.82%, essentially on target" was measuring the seed,
    not the climate. The seed is now discarded on each bin's first real sample
    (see that function), which makes even the 64x128 benchmark show the true
    signal, and `itcz_seasonal_target_min_fraction` fixes the structural flaw
    this masked. `k` itself is retained at 2.0 -- re-swept on the corrected
    instrument it remains the best value, see that field's own table."""

    itcz_seasonal_target_min_fraction: float = 0.10
    """Floor on `itcz_seasonal_target_response`'s dry-season trough, as a
    fraction of each row's own annual-mean precipitation target. `0.0` disables
    the floor and restores the unbounded-additive behaviour (the 2026-08-05 bug
    described below); `1.0` would suppress the modulation entirely.

    **The flaw this fixes.** That knob's modulation is additive,
    `1 + k*(itcz_window(day) - itcz_window_annual_mean)`, whose seasonal trough
    is `1 - k*(mean - window_min)`. That is unbounded below, and goes negative
    for any row with `itcz_window_annual_mean > 1/k`. At `k=2.0` the threshold
    is `mean > 0.5`, which at `itcz_seasonal_response=0.4` and
    `ITCZ_HALF_WIDTH_DEG=10` is **every row within ~9 deg of the equator** --
    precisely the tropical rainforest band. Those rows were held up only by the
    bare `clip(0.05)` in `atmosphere.generate_precipitation`, i.e. a 95%
    dry-season shutoff of the target. Because the multiplicative swing peaks
    where the *window mean* is large rather than where the window *changes*
    most, the damage was concentrated in the deep tropics and left the savanna
    latitudes this knob exists to serve barely affected -- the shape was
    inverted. Measured on `saves/test.npz` (512x1024, 24yr), the model's
    wet/dry precipitation ratio tracked the predicted target swing 1:1:

        lat     measured wet/dry    predicted target swing
        0 deg          3.4                  3.7
        3 deg         13.9                 14.8
        5 deg         36.6                 35.1  <- clip(0.05) binding
        8 deg         36.1                 38.4  <- clip(0.05) binding
        10 deg        14.4                 15.6
        15 deg         3.2                  3.8
        20 deg         1.9                  1.8

    Earth's equatorial wet/dry ratio is 1.5-5, and driest-month precipitation in
    the rainforest cores is 60-150 mm; the model was producing 4-30 mm. Since Af
    vs Aw is decided purely on "driest month >= 60 mm", Af collapsed.

    **Implementation.** `k` is capped per row at `(1 - f)/itcz_window_mean`,
    which bounds the trough at `f + (1-f)*window_min/window_mean >= f`. This
    binds only on rows where the additive form was already being clipped
    (`window_mean <= (1-f)/k`); the savanna belt keeps the full `k` and its full
    dry season, so the fix is targeted rather than a global weakening.
    Physically it says the migrating ITCZ delivers only part of deep-tropical
    rainfall -- the rest is year-round local convection and moisture recycling,
    which a displaced ITCZ does not switch off -- while at savanna latitudes
    essentially all rainfall is ITCZ-delivered and a displaced belt really does
    mean a dry season.

    **It also restores exact mean-preservation, which `k=2.0` had silently
    broken.** Because the cap is time-independent (it divides by the window's
    *annual mean*), the modulation's own time-average is still exactly 1.0, so
    every annual-mean zonal/desert/continental calibration built on
    `target_row_mm_day` is genuinely untouched -- which is why the sweep above
    moves `reference_error_score` so little. The old form could not claim that:
    its own docstring notes the `clip(0.05)` floor "stops being exactly
    mean-preserving above k~1.0". Verified numerically over a dense sampling of
    a full orbit at H=64/256/512, `f=0.10`, `k=2.0`:

        quantity                              old form        capped form
        raw trough minimum                     -0.010          +0.145
        rows demanding NEGATIVE precip         2 / 6 / 10       none
        (row, time) samples hitting the clip      0.6%          none
        max |time-mean(modulation) - 1|         5.8e-3          5.4e-14
        rows capped (i.e. affected at all)         --        |lat| <= ~10 deg

    The old floor was not a safety net that happened to catch an edge case: it
    was load-bearing, and it was carrying the rainforest band.

    **Calibration (2026-08-05, `k` held at 2.0, seed fix in place -- so Af/Am/Aw
    read off these runs are trustworthy for the first time). Swept at two
    resolutions because the defect this fixes was originally introduced by
    trusting 64x128 alone; `dryMed`/`ratio` are tropical-land (|lat|<=10)
    driest-month median [mm] and wet/dry ratio, against Earth's ~60-150 mm and
    1.5-5. Earth shares: Af 6.0%, Am 4.0%, Aw 10.0%.**

    Tracked 64x128 benchmark (`RealTerrainValidationConfig()` defaults, ~14 s):

        f      Af      Am     Aw    tropMAE  dryMed  ratio  cKappa  gMAE   refErr
        0.0    5.12   2.98  11.90    1.27     10.3   22.3   0.3214  3.79   0.2118
        0.10   7.38   3.51   9.41    0.82     27.9    5.6   0.3348  3.68   0.2116
        0.15   7.87   4.04   8.43    1.16     33.5    4.8   0.3305  3.68   0.2116
        0.25   9.34   3.74   7.56    2.01     43.4    3.7   0.3285  3.56   0.2116
        0.35  11.91   1.95   7.17    3.60     51.0    3.0   0.3335  3.42   0.2115
        0.50  13.81   1.12   6.29    4.80     63.0    2.3   0.3324  3.32   0.2111

    256x512, `block_size=3` (nearer the 512x1024 production grid, ~90 s):

        f      Af      Am     Aw    tropMAE  dryMed  ratio  cKappa  gMAE   refErr
        0.0    3.95   1.86  13.25    2.48      9.1   21.6   0.3843  2.31   0.1364
        0.10   6.32   3.35   9.55    0.47     37.9    4.6   0.3852  2.25   0.1376
        0.15   7.08   3.37   8.81    0.97     42.6    4.0   0.3862  2.23   0.1381
        0.20   7.79   3.57   7.98    1.41     46.9    3.5   0.3861  2.21   0.1382
        0.25   8.76   3.38   7.28    2.03     51.8    3.1   0.3876  2.17   0.1387
        0.35  11.08   2.60   5.89    3.53     61.1    2.5   0.3861  2.12   0.1384

    **0.10 minimises the Af/Am/Aw error at BOTH resolutions** (tropMAE 0.82 and
    0.47) and is the `class_kappa` optimum at 64x128, which is why it ships
    rather than a value read off one grid. Higher `f` keeps improving the
    group-share and `class_kappa` metrics marginally at 256x512, but overshoots
    Af hard (11% at f=0.35, against Earth's 6%) and drains Aw below target --
    the deep tropics stop having any dry season at all. `reference_error_score`
    costs +0.0012 (0.1364 -> 0.1376) at 256x512 and *improves* slightly at
    64x128; either way this is far cheaper than the +0.0022 zonal-magnitude cost
    `itcz_seasonal_target_response` itself had to accept.

    Deserts and continental boxes are flat across the entire sweep at both
    resolutions (64x128: Sahara 173 mm/yr, Kalahari 138-139, Atacama 64, US
    Midwest 937, Central Europe 923-924, SE US 986, S Japan 960 -- i.e. within
    1 mm/yr at every `f`). That is structural, not luck: the cap only
    ever *raises* a dry-season trough, and only on rows where `itcz_window`'s
    annual mean is large, so it cannot reach a subtropical desert.

    **Long-run behaviour, which is the actual point.** At 256x512 with 5yr spinup
    + 1yr eval (seed-free and soil/ocean-equilibrated), f=0.0 gives Af 3.82% /
    dryMed 9.2 mm / ratio 21.9 and f=0.10 gives Af 6.06% / 37.2 mm / 4.7. More
    directly: continuing `saves/test.npz` itself (512x1024, the production grid,
    already 24yr old and fully equilibrated on the *broken* physics) forward under
    the fix, the band grows back in and **stabilises** rather than decaying --

        state          Af     Am     Aw    tropMAE  dryMed  ratio  Amazon  Congo
        as saved      2.61   1.75  11.16    2.27      8.2   23.9    17      45
        +1 yr         4.27   3.15  10.55    1.04     27.5    6.9    50      70
        +2 yr         5.59   3.16   9.88    0.46     34.8    5.3    67      80
        +3 yr         6.10   3.17   9.50    0.48     37.5    4.9    73      84
        +4 yr         6.32   3.15   9.33    0.61     38.4    4.8    76      85
        Earth         6.00   4.00  10.00     --    60-150  1.5-5    60      80

    -- i.e. the exact reverse of the reported symptom (a rainforest belt that
    appeared and then dissolved). Convergence takes ~3 years because the monthly
    bins must re-equilibrate at `window_years=1.0`; the Köppen shares land on
    6.32 / 3.15 / 9.33 against Earth's 6.0 / 4.0 / 10.0, and the wet/dry ratio
    lands inside Earth's 1.5-5 band.

    **Remaining known gaps, NOT addressed here** (both are the separate
    arid-overrepresentation defect -- BWh is 17.8% of land against Earth's ~9.5%
    -- rather than this seasonality one, and both are *annual-total* shortfalls
    that no seasonal-distribution knob can reach): Borneo's driest month is 58 mm
    against ~150, New Guinea's 31 against ~120, and southern Congo's 36 against
    ~55. Borneo's annual total is ~900 mm/yr against a real ~3000. BWh also
    drifts up slightly across the continuation (16.98% -> 17.79%) as tropical
    cells sharpen, which is a real if small cost."""

    precip_raw_shape_weight: float = 0.0
    """Strength [0-1] of blending `precip_potential`'s own row-relative raw
    shape into `atmosphere.generate_precipitation`'s per-cell target weight
    (`cell_weight`/`target_cell_weight`), on top of the existing
    `subsidence_suppression`-based desert/continental redistribution.

    Direct follow-up to `itcz_seasonal_target_response`'s own flagged
    limitation (see its docstring): real-terrain measurement
    (2026-07-31, `saves/test.npz`) found `global_rescale_factor` averaging
    ~5x on a 5yr MONTHLY continuation -- i.e. the moisture-budget
    aspirational-fill mechanism supplies most of a typical row's final rain,
    not raw local physics, so *where* that fill lands is what actually
    determines the spatial pattern. The existing `cell_weight` only shapes
    that via `subsidence_suppression`, which is ~uniformly near 1.0 across
    the whole ITCZ zone (deserts are defined by subsidence; the deep tropics
    aren't), making `cell_weight` a near-exact no-op inside the tropics --
    exactly where the Af (rainforest) vs. Aw/Am (savanna/monsoon) distinction
    needs real per-cell differentiation and currently has none.

    `precip_potential` (this cell's own pre-rescale raw signal) has already
    been independently verified to correctly rank wet vs. dry areas
    (`known-physics-gaps.md` item 3b UPDATE 3: continental interior
    consistently 2.5-3x desert every year, pre-rescale) and, unlike
    `subsidence_suppression`, varies meaningfully *within* the tropics too
    (e.g. a real Congo-basin convergence hotspot vs. a savanna-margin cell at
    the same latitude). Blending in its row-relative shape gives the
    aspirational fill a genuine per-cell target everywhere, not only in the
    subtropics.

    Implementation: `_desert_factor *= (1 - w + w*raw_shape)`, then
    renormalized to row-mean 1.0 exactly like the existing desert-factor-only
    path -- mean-preserving by construction, so row totals (and therefore the
    orographic-uplift/cloud-cover tests a purely per-row target would break)
    are unaffected regardless of `w`. `0.0` is an exact no-op (recovers the
    prior `cell_weight` byte-for-byte).

    `raw_shape` is normalized against the row's LAND-ONLY mean
    (`precip_potential` weighted by `land_f`), not the whole row, and is only
    ever applied to land cells (ocean cells stay at a neutral factor of 1.0
    in the blend, same as `_desert_factor` alone). **An earlier version of
    this fix normalized against the whole row (land+ocean) and applied to
    every cell -- WRONG, found via real-terrain measurement**:
    `precip_potential` runs far higher over open ocean than land almost
    everywhere (abundant moisture, no dry-belt/subsidence gating), so a
    whole-row mean read every land cell as "below average" purely for
    sharing a row with ocean, letting ocean bid weight-share away from land
    entirely rather than reshaping *within* land. Measured directly at
    weight=0.3 on `saves/test.npz`: Canadian Prairies/US Midwest/Central
    Europe collapsed ~65-75% and Kalahari/Atacama got *wetter*, both
    backwards. The land-only-mean, land-only-applied version fixes this by
    construction: ocean's share of every row's target is provably unchanged
    for any `w`.

    Additionally gated by `itcz_window` (`effective_w = w * itcz_window[row]`)
    -- the blend is intended to fill the gap where `subsidence_suppression`
    is a no-op (inside the ITCZ), not to touch mid-latitudes where it's
    already doing its job. Confirmed necessary by real-terrain measurement:
    an earlier ungated version (flat `w` at every latitude) degraded
    Canadian Prairies/US Midwest by 18-34% even though those boxes sit far
    outside the true tropics -- `precip_potential`'s land-relative shape at
    those latitudes evidently still favors *other* cells in the same row
    over the Prairies/Midwest specifically, actively fighting the
    already-validated mid-latitude desert/continental ranking work
    (`ferrel_v_land_shift_deg`, `moisture-budget-desert-ceiling-fix`).
    Gating by `itcz_window` confines the effect to where it was actually
    aimed and fully protects those boxes (confirmed flat within noise at
    w=0.6/1.0 after gating).

    **HONEST RESULT (2026-07-31, `saves/test.npz`, 512x1024, 5yr MONTHLY
    continuation, area-weighted land Koppen): this mechanism does not achieve
    its goal, even gated, and should not be enabled.**

        w      Af      Am     Aw     arid(BW+BS)  Sahara  Kalahari  Atacama  Prairies  Midwest  Cent.Eur
        0.0    14.52%  1.30%  8.81%  19.53%       571     493       437      462       610      532
        0.6    14.52%  1.22%  8.19%  20.63%       546     478       436      467       611      533
        1.0    13.74%  1.40%  7.85%  21.73%       529     471       433      467       612      532

    Gating successfully protects continental interior (Prairies/Midwest/
    Central Europe all flat within noise) and desert boxes improve modestly
    (Sahara -7%, Kalahari -4%) -- but **Aw moves the WRONG direction**
    (8.81%->7.85%, a ~11% *decrease*, not the increase this whole line of
    work was chasing) while **arid_pct climbs** (19.53%->21.73%), entirely
    within the gated (tropical) zone since continental interior is provably
    untouched. Root cause (consistent with, though not independently
    isolated cell-by-cell from, `climate_averages.classify_koppen`'s own
    logic): Koppen's B-climate aridity threshold `P_threshold` scales with
    `pct_summer` concentration (jumping from `20*T+140` to `20*T+280` once a
    cell's rain is >=70% one-season-concentrated) -- redistributing an
    already latitude-band-scarce raw signal (recall `global_rescale_factor`
    averages ~5x; most of a row's rain is synthetic fill, not raw physics)
    pushes already-marginal cells' *totals* down at the same time their
    seasonality (via `itcz_seasonal_target_response`) is rising, so they
    cross into full BW/BS aridity rather than landing in the in-between Aw
    band this project has been trying to grow. This is the third,
    independently-discovered failure mode against the same underlying
    "per-row vs. per-cell target resolution" gap named in
    `known-physics-gaps.md` item 3b (after the reverted flat-multiplier
    desert-suppression attempt and the reverted moisture-transport
    strengthening attempts) -- redistribution mechanisms in this codebase
    keep hitting the same wall: with raw production this scarce, any
    within-row reshaping mostly moves cells *between* Aw and full aridity,
    not from Af into Aw. **Shipped as tested, wired, but inert infrastructure
    (default `0.0`, verified exact no-op) following this project's
    established convention (`moisture_advection_scale`,
    `cloud_water_feedback`, `abyssal_overturning_coeff`) for a real,
    correctly-implemented mechanism that measured net-negative for its
    intended purpose.** A future attempt at this exact gap would need to
    raise raw tropical-belt production itself (not just reshape a scarce
    signal) -- already attempted and failed via 4+ different mechanisms in
    `known-physics-gaps.md` item 3b UPDATES 3-9 (moisture transport, soil
    bucket, evapotranspiration gating, wind convergence)."""

    moisture_budget_tropical_cap_boost: float = 0.0
    """[0-1ish] Raises `_moisture_budget_precip_rescale`'s per-step removal caps
    (`max_total_removal_fraction`/`max_added_removal_fraction`, normally the
    flat constants 0.85/0.15 everywhere) specifically inside the ITCZ, gated by
    `itcz_window` exactly like `precip_raw_shape_weight`. `0.0` is an exact
    no-op (both caps stay their flat scalar default, byte-for-byte).

    Motivated by a direct measurement this session (2026-08-01, `saves/earth.pkl`,
    512x1024, 2yr MONTHLY real-terrain, `debug_fields["precip_rescale_capacity_limited"]`
    zonal-mean): the moisture-budget rescale is capacity-limited (pinned at its
    cap, still short of `target_row_mm_day`) at *every* sampled latitude, not
    just the dry/cold ones -- including the deep tropics, which only reach
    ~0.85-0.94 of target despite sitting on the planet's most abundant ocean
    moisture supply. This reframes `precip_raw_shape_weight`'s and A5's
    documented "raw production is too scarce" finding: it is not merely that
    raw physics under-produces relative to a target that's otherwise
    reachable -- the *aspirational-fill mechanism itself* is capped below what
    even the tropics' own (comparatively abundant) moisture reserve could
    supply.

    This differs from the previously-reverted flat/global cap raise (see A5 in
    `docs/ACCURACY_AUDIT.md`: "made desert/continental ranking worse --
    draining continental interior's q faster than land_evap replenishes it").
    That attempt raised the cap *everywhere*, including the already-fragile
    mid-latitude/dry-belt rows where q has little margin to spare. This knob
    only ever raises the cap where `itcz_window` is nonzero, leaving every
    other regime's cap at exactly 0.85/0.15 -- deliberately narrower in scope
    than the earlier attempt, to test whether the previous failure mode was
    caused by the *global* reach of that fix rather than by cap-raising in
    general.

    Implementation: `_moisture_budget_precip_rescale` now accepts either a
    scalar or a per-row `np.ndarray` for both cap parameters (scalar path is
    byte-identical to the prior behavior; existing unit tests call it with
    scalar defaults and are unaffected). `generate_precipitation` builds
    `max_total_removal_fraction_row = clip(0.85 + boost*0.10*itcz_window, ..., 0.95)`
    and `max_added_removal_fraction_row = clip(0.15 + boost*0.15*itcz_window, ..., 0.30)`
    when `moisture_budget_precip_rescale` is enabled."""

    surface_water_cap_mm: float = 50_000.0
    """Hard ceiling [mm] on `surface_water_mm` per cell (50 m -- deeper than
    any real lake, a safety backstop not a physical mechanism). This compact
    D8 router has no channel-capacity concept: a real-terrain test found a
    continent-scale drainage basin (Amazon/Congo-like) funneling into a
    single grid cell grew past 1.9 KILOMETERS of area-averaged depth over 10
    years, essentially linearly, and neither more routing passes (8->16) nor
    a near-total per-pass routing fraction (0.55->0.99) meaningfully reduced
    it -- ruling out "not enough passes to traverse the basin" as the cause.
    `lake_evap_mm_day` alone cannot bound this either: it is utterly
    negligible next to a continent's real discharge concentrated in one
    cell with no lateral spreading or channel-velocity limit. This cap is a
    blunt but necessary backstop until (if ever) a real flow-accumulation-
    aware channel capacity is implemented; excess is simply discarded (not
    conserved), so it is not a substitute for fixing the underlying
    limitation, only a guard against literal km-scale numbers reaching the
    UI/save files."""

    lake_evap_mm_day: float = 4.0
    """Open-water evaporation demand [mm/day] applied to `surface_water_mm`
    wherever it is > 0, scaled by a simple temperature factor. Without this,
    standing water has no sink at all: a real-terrain test found flat,
    near-equatorial river-delta terrain (Amazon/Congo-scale drainage basins,
    D8's strict "must have a strictly lower neighbor" rule finding no exit)
    accumulating to 611 METERS of area-averaged depth after just 12 years of
    continuation -- an unbounded runaway, not a real lake reaching
    equilibrium. Real lakes and floodplains lose water to evaporation; this
    is that term. 4.0 mm/day is a representative open-water pan-evaporation
    magnitude (roughly 1.5 m/year at full strength), scaled down (never to
    zero -- lakes still evaporate somewhat even when cool) in cold climates
    via `clip((T-273.15)/20, 0.1, 2.0)`."""

    # ------------------------------------------------------------------ #
    # Cloud radiative feedback (Feature 1)
    # ------------------------------------------------------------------ #
    cloud_greenhouse_factor: float = 0.12
    """Strength of high-cloud OLR trapping [dimensionless].
    High clouds (cold tops, T_air < 265K) reduce effective epsilon by this fraction
    times cloud fraction.  0.12 gives tropical cloud LW CRE ≈ +6–10 W/m².
    Set 0.0 to disable cloud greenhouse effect."""

    cloud_water_feedback: float = 0.0
    """Blend weight [0-1] for prognostic cloud water (`PlanetState.cloud_water`)
    feeding back into `cloud_fraction`, layered on top of the existing
    RH-diagnosed value -- 0.0 = pure diagnostic, bit-identical to before.
    `cloud_water` accrues a real condensation/rain-out/evaporation/baseline-
    settling mass budget every step regardless of this weight (so enabling
    it later doesn't cold-start from zero memory), but only feeds back into
    `cloud_fraction` (and therefore albedo/greenhouse) when > 0.

    First calibrated Jul 2026 against a short real-terrain check (std of
    day-to-day cloud_cover change dropping ~23% from w=0 to w=1, mean cloud
    cover drifting down only modestly, 0.171->0.157) -- but default was kept
    at 0.0 pending the multi-decade climate-drift/ECS re-validation a
    default flip would warrant.

    That re-validation (2026-07-27) found two real bugs in the mechanism,
    both now fixed, and then a calibration reason the default should stay
    0.0 regardless.

    Bug 1: `simulate._evolve_temperature`'s cloud-water update was a
    forward-Euler-style `prev*exp(-sink*dt) + S_cond*dt`, correct only for
    small dt. At MONTHLY/ANNUAL cadence (dt ~30d) with sink_rate*dt >> 1, the
    source term `S_cond*dt` grows linearly with dt instead of saturating at
    the true steady state -- a 60yr MONTHLY synthetic spinup drove mean
    cloud_cover to 0.59 (w=0.5) / 0.79 (w=1.0) from a 0.25 baseline,
    reproducing the exact runaway an earlier session thought it had already
    fixed (that check only ran a short DAILY-cadence continuation, where the
    bug is invisible). Fixed by replacing the update with the exact solution
    of the underlying ODE (dcw/dt = S_cond - sink_rate*cw), which reduces to
    the original formula in the small-dt limit but stays correctly bounded at
    any cadence. Re-ran the 60yr sweep after the fix: mean cloud_cover
    0.252->0.222->0.206->0.178 for w=0/0.3/0.5/1.0, smooth and bounded: no
    runaway. ECS unaffected: the 50yr-ANNUAL 2xCO2 pair
    (`test_ecs_equilibrium_magnitude`'s own config) gives dT=1.769/1.778/
    1.778/1.786 K for the same four weights, under 1% spread.

    Bug 2: a fresh state cold-starts `cloud_water` at 0.0, so the first
    several days' blended `cloud_fraction` crater toward 0 while
    `cloud_water` climbs from zero to its equilibrium -- measured to collapse
    `test_cloud_feedback.py`'s 5-day fresh-start mean cloud fraction from
    ~0.124 (w=0) to 0.075 at w=0.5. Fixed by seeding `cloud_water` from the
    current diagnostic `cloud_fraction` (via the `cw_ref` scaling) whenever
    `prev_cloud_water` is None, instead of zero.

    With both bugs fixed, re-measured whether a nonzero default is actually
    a net improvement -- it is not. Even after the cold-start fix, mean
    cloud_cover on the same 5-day fixture still declines monotonically with
    w (0.124 -> 0.118 -> 0.112 -> 0.108 -> ~0.10 for w=0/0.1/0.2/0.3/0.5).
    `test_cloud_feedback.py`'s own long-standing comment already flags this
    model's cloud fraction as a KNOWN GAP -- ~0.16 is roughly 4x below
    Earth's observed ~0.67 global mean -- and *any* nonzero blend weight
    measurably worsens that existing low bias, even at w=0.1. The
    smoothing benefit (real, and unaffected by any of this) doesn't outweigh
    making an already-documented realism gap worse. **Default stays 0.0**:
    both mechanism bugs are fixed and the infrastructure is tested and ready,
    but this is the same "no scoped win available" conclusion this project
    reached before for `moisture_advection_scale` and (pre-fix)
    `soil_deep_gain_rate` -- correct now, not yet worth turning on."""

    # ------------------------------------------------------------------ #
    # Water vapor greenhouse (Feature 2)
    # ------------------------------------------------------------------ #
    wv_greenhouse_factor: float = 0.10
    """Strength of water-vapor epsilon reduction [dimensionless].
    Higher RH → lower epsilon (stronger greenhouse trapping).
    0.10 restores ~0.6 K/(W/m²) amplification when co2_climate_feedback is the
    Planck-only value (0.8).  Set 0.0 to disable explicit WV feedback."""

    # ------------------------------------------------------------------ #
    # Salinity / AMOC freshwater (Feature 3)
    # ------------------------------------------------------------------ #
    salinity_reference_psu: float = 35.0
    """Global mean ocean salinity [PSU].  Used as the restoring target for deep
    mixing and as the baseline for computing North Atlantic salinity anomalies."""

    salinity_amoc_scale: float = 1.0
    """Sensitivity of amoc_factor to North Atlantic salinity anomaly [dimensionless].
    1.0 → +1 PSU anomaly multiplies amoc_factor by 1.15; −2 PSU by ~0.55.
    0.0 disables salinity–AMOC coupling."""

    temperature_amoc_reference_k: float = 277.15
    """Reference North Atlantic (50-75N, ocean cells) surface temperature [K]
    used as the neutral point for the temperature-density AMOC term -- 4 degC,
    the temperature of maximum density for fresh water, physically relevant to
    deep convective sinking. Colder-than-reference water is denser and
    strengthens AMOC; warmer water is less dense and weakens it -- same
    phenomenological-gain convention as salinity_reference_psu/
    salinity_amoc_scale above, not a full seawater equation of state."""

    temperature_amoc_scale: float = 0.0
    """Sensitivity of amoc_factor to North Atlantic surface temperature
    anomaly [dimensionless]. 0.0 (default) disables the term. Unlike the
    salinity anomaly -- pinned near its reference by evolve_salinity's
    explicit 2-year restoring tendency -- ocean temperature has no equivalent
    restoring force, so the anomaly's typical magnitude against a fixed
    reference hasn't been checked against the real-terrain baseline yet. Real
    and unit-tested; needs a calibration pass before defaulting on
    (FEATURES.md item 5)."""

    # ------------------------------------------------------------------ #
    # CH4 / permafrost carbon (Feature 4)
    # ------------------------------------------------------------------ #
    ch4_baseline_ppb: float = 700.0
    """Pre-industrial CH4 reference concentration [ppb].  Used as M₀ in the
    IPCC AR6 forcing formula ΔF = 0.036*(sqrt(M)−sqrt(M₀))."""

    ch4_initial_ppb: float = 1900.0
    """Initial atmospheric CH4 at simulation start [ppb].  Modern ≈ 1900 ppb."""

    # ------------------------------------------------------------------ #
    # Deep ocean 2-layer (Feature 5)
    # ------------------------------------------------------------------ #
    deep_ocean_exchange_rate: float = 9.13e-5
    """Heat exchange rate between mixed layer and deep ocean [1/day].

    τ = 1/rate ≈ 10957 days ≈ 30 yr is the *mixed layer's* damping timescale.
    It is NOT how fast the deep layer approaches equilibrium -- see
    `deep_ocean_heat_capacity_ratio` below, which makes that ~74x slower."""

    deep_ocean_heat_capacity_ratio: float = 50.0 / 3700.0
    """Mixed-layer / abyssal heat-capacity ratio (≈ 50 m / 3700 m).

    Scales the abyssal ΔT for a given heat flux (correct: the same flux into a
    ~74x larger reservoir produces 1/74 the temperature change), so the deep
    ocean's own equilibration timescale is
        τ_eff = 1 / (deep_ocean_exchange_rate * this) ≈ **2219 years**.
    Measured 2026-07-25 on a 500-year real-terrain run: 3.41 K of a 16.8 K
    surface-deep gap closed in 498 years, implying τ ≈ 2195 yr -- a 1% match.

    Two consequences worth knowing before using deep-ocean temperature as a
    validation target (see overnight/FINDINGS.md):
      * Deep-ocean T cannot reach observational values in any practical run; it
        is a structurally transient metric, not an equilibrium one.
      * The equilibrium it is *heading toward* is ~25 degC, not Earth's 1-4 degC,
        because the deep layer exchanges heat only vertically with the cell above
        it -- there is no overturning or lateral abyssal transport, so each cell
        relaxes toward its own local surface temperature. Raising the exchange
        rate makes the deep ocean reach that wrong target sooner, not become more
        realistic; a physical abyss needs overturning, not a rate tweak."""

    deep_ocean_depth_m: float = 3700.0
    """Mean abyssal ocean depth [m].  Used only for diagnostic OHC calculations."""

    co2_wind_averaging_days: float = 30.0
    """EMA window [days] for the wind speed fed into ocean_co2_flux's piston
    velocity (Jul 2026 fix). Wanninkhof's k∝u² is calibrated for time-averaged
    wind; feeding it the instantaneous per-step value inflates mean(k) via
    Jensen's inequality whenever wind has day-to-day variance. See
    carbon_cycle.ocean_co2_flux docstring for the full explanation."""

    # ------------------------------------------------------------------ #
    # Eddy meridional heat flux (Feature 7)
    # ------------------------------------------------------------------ #
    eddy_heat_flux_coeff: float = 0.006
    """Meridional eddy heat flux coefficient [K/day per K/cell²].
    Represents baroclinic eddy transport by mid-latitude storm tracks (20–70°).
    Applied as meridional Laplacian diffusion on T_sst, weighted by a
    storm-track window peaked at 45°.  0.006 adds ~0.5 K of mid-latitude
    warming per year of spinup relative to a run with the coefficient at 0.
    Set 0.0 to disable."""

    # ------------------------------------------------------------------ #
    # Abyssal overturning (Phase 5 canvas item)
    # ------------------------------------------------------------------ #
    abyssal_overturning_coeff: float = 0.0
    """Meridional eddy-diffusion coefficient [K/day per K/cell²] applied to
    `T_deep_ocean`, representing the real global overturning conveyor
    (North Atlantic/Southern Ocean deep-water formation spreading and mixing
    abyssal temperature worldwide) that this model otherwise completely
    lacks: `deep_ocean_exchange_rate` only exchanges each ocean cell's deep
    layer *vertically* with its own local mixed layer -- there is no lateral
    transport between deep-ocean cells at different latitudes at all. Real
    deep ocean is remarkably globally uniform (~2-4C) precisely because of
    this overturning; a purely-local-vertical model instead lets deep
    temperature slowly drift toward each cell's own surface climate
    (measured: tau_eff ~2219yr, equilibrium ~25C rather than ~2-4C -- see
    known-physics-gaps.md). Applied as the same Laplacian-diffusion-with-
    substepping pattern as `eddy_heat_flux_coeff`, but globally (not
    storm-track-windowed) and only where liquid ocean exists. 0.0 (default)
    is an exact no-op pending real-terrain/long-run calibration."""

    # ------------------------------------------------------------------ #
    # Land ice mass balance, thickness, and flow (Phase 5 canvas item)
    # ------------------------------------------------------------------ #
    enable_land_ice_dynamics: bool = False
    """Enable prognostic land-ice thickness: mass balance (accumulation minus
    degree-day ablation), a simplified shallow-ice-approximation flow, and a
    derived eustatic sea-level diagnostic (`PlanetState.sea_level_change_m`).

    Before this, land ice was only `ice_sheet_age`, a Koppen-EF-classification
    counter with no mass, thickness, or flow at all -- and the snow-depth
    bucket (`snow_depth`, meters SWE) silently discarded any accumulation
    beyond its 10 m cap. When enabled, that discarded overflow instead feeds
    `PlanetState.land_ice_thickness` (also meters water-equivalent --
    deliberately the same convention as `snow_depth`, to avoid introducing a
    free ice-density parameter into an already-simplified single-layer
    model). Disabled by default pending real-terrain calibration, matching
    this project's convention for new structural mechanisms
    (`abyssal_overturning_coeff`, pre-fix `soil_deep_gain_rate`); the numeric
    fields below have real, reasoned defaults regardless, so re-enabling
    doesn't require re-deriving them from scratch.

    Deliberately NOT coupled to albedo or `evolve_salinity` this pass (unlike
    `ice_sheet_age`'s 0.80 albedo and the hydrology feature's ocean-runoff
    coupling) -- kept out to keep this addition self-contained; both are
    natural follow-ups once real-terrain thickness/flow behavior has been
    checked. Flow also does not follow terrain slope: `elevation` is a
    normalised [0, 1] field with no single canonical meters conversion
    anywhere in this codebase (see ROADMAP.md's "max_elevation_km hardcoded
    four different ways, three different formulas" item) -- adding a fifth
    conversion would compound a known, already-flagged gap rather than fix
    it, so flow here is plain thickness-gradient diffusion (spreads existing
    ice domes outward toward their margins) rather than true downhill
    transport."""

    ice_melt_degree_day_mm: float = 6.0
    """Degree-day ablation factor [mm w.e./degC/day] for exposed land ice,
    applied above 0 degC air temperature. Distinct from (and higher than)
    `snow_depth`'s fixed 3.0 mm/degC/day melt factor: bare glacier ice is
    darker and denser than fresh snow and empirically melts faster once
    exposed (real-world degree-day factors: ~2-4 for fresh snow, ~5-8 for
    bare ice) -- this uses a representative mid-range value."""

    ice_flow_diffusivity: float = 2.0e-3
    """Flow strength [1/day per m of local thickness per cell^2] for land
    ice. Applied as a mass-conservative flux-form diffusion of thickness
    itself (not ice-surface elevation -- see `enable_land_ice_dynamics` for
    why), with a per-cell diffusivity of `this * local_thickness` so thick
    ice spreads faster than thin ice: a one-parameter linearisation of the
    real Glen's-law H^(n+2) dependence (real n~3 gives H^5; this uses H^1
    for numerical simplicity and stability), the same kind of deliberate
    diffusive proxy `eddy_heat_flux_coeff`/`abyssal_overturning_coeff`
    already use for their own transport processes. Substepped for CFL
    stability the same way. Ice that diffuses into an adjacent ocean cell is
    discarded from the land reservoir (a simplified calving proxy) rather
    than credited to `evolve_salinity`'s freshwater input -- consistent with
    this feature's scope boundary above -- but that mass loss is still
    correctly reflected in the sea-level diagnostic, since it reads current
    total land-ice volume directly rather than tracking flux history. 0.0
    disables flow, leaving pure local mass balance."""

    land_ice_max_thickness_m: float = 4000.0
    """Hard ceiling [m w.e.] on `land_ice_thickness` (~Antarctica's real
    ~4.8 km maximum, rounded down for the water-equivalent convention used
    here) -- a safety backstop bounding both physically-implausible
    unbounded growth and the flow diffusion's substep count, not a physical
    mechanism. Mirrors `surface_water_cap_mm`'s role for the hydrology
    feature."""

    # ------------------------------------------------------------------ #
    # Discrete moving storm systems
    # ------------------------------------------------------------------ #
    storm_pressure_amp_pa: float = 110.0
    """Peak pressure-anomaly amplitude of discrete moving mid-latitude storm
    cyclones [Pa], a deterministic function of simulated time
    (evolve_wind's storm-track parameterisation, atmosphere._storm_pressure_anomaly).
    Individual storms vary ±30% around this value, spin up/mature/decay over
    ~9 days, and translate eastward-and-poleward through the 35–55° storm-
    track band in both hemispheres. Comparable to 2-3x the existing Rossby-
    wave term (30–60 Pa) but well below the thermal (~450 Pa) and terrain
    (~900 Pa) PGF terms, so storms read as embedded transients rather than a
    dominant/discontinuous signal. Set 0.0 to disable."""

    trade_wave_pressure_amp_pa: float = 65.0
    """Peak pressure-anomaly amplitude of discrete moving trade-wind/subtropical
    wave disturbances [Pa] (real-world analogue: easterly waves), a deterministic
    function of simulated time (atmosphere._storm_pressure_anomaly, second
    population). Individual waves vary ±30% around this value, spin up/mature/
    decay over ~5 days, and translate westward through the 12–32° band in both
    hemispheres, matching the trade easterlies. Weaker and shorter-lived than
    storm_pressure_amp_pa's mid-latitude cyclones, matching real easterly waves'
    smaller/faster character, and covers the latitude band those storms don't
    reach. Set 0.0 to disable."""

    # ------------------------------------------------------------------ #
    # Jet stream dynamics: persistent meander index + blocking events
    # ------------------------------------------------------------------ #
    jet_meander_tau_days: float = 10.0
    """AR1 relaxation timescale [days] for the per-hemisphere jet meander/
    waviness index (atmosphere._update_jet_index). Shorter = index tracks its
    thermal-gradient-derived target more tightly; longer = more inertia."""

    jet_meander_noise_amp: float = 0.35
    """Stochastic forcing scale (per sqrt(day)) on the jet meander index. Drawn
    from a deterministic hashed RNG seeded by simulated time, so identical
    time_days always reproduces identical noise (same reproducibility contract
    as ROSSBY_MODES/_storm_pressure_anomaly)."""

    jet_gradient_ref_k: float = 40.0
    """Reference pole-equator temperature gradient [K] used to compute the jet
    index's target: gradients weaker than this push the index positive
    (wavier/more blocked jet), stronger gradients push it negative (fast,
    zonal jet) — a simplified Arctic-amplification-weakens-the-jet coupling to
    the model's existing ice/polar-cooling physics."""

    jet_lat_shift_per_index: float = 6.0
    """Degrees of latitude the jet core (MID_LAT_JET_CENTER_DEG) shifts per
    unit of jet index, independently per hemisphere."""

    jet_speed_scale_per_index: float = 0.25
    """Fractional change to U_TARGET_MIDLAT per unit of jet index (positive
    index = wavier/slower jet, so this is typically applied as a reduction)."""

    jet_wave_amp_scale_per_index: float = 0.5
    """Fractional boost to the Rossby-wave (ROSSBY_MODES) amplitude per unit of
    positive jet index — a wavier jet state should produce larger-amplitude
    meanders, not just a latitude shift."""

    jet_block_trigger_rate_per_day: float = 0.015
    """Base daily probability of a new blocking-ridge event when the jet index
    is elevated (atmosphere._update_jet_blocking). Scaled up by how far the
    index sits above its trigger threshold."""

    jet_block_duration_range_days: tuple[float, float] = (10.0, 40.0)
    """Range of durations [days] drawn for a triggered blocking event —
    matches real-world persistent ridge/trough lifetimes (weeks, not days)."""

    jet_block_pressure_amp_pa: float = 180.0
    """Peak pressure amplitude [Pa] of an active blocking ridge
    (atmosphere._blocking_ridge_pressure_anomaly). Positive (a ridge is a
    high), larger than storm_pressure_amp_pa since a block is a single
    persistent quasi-stationary feature rather than an embedded transient."""

    jet_block_radius_km: float = 3200.0
    """Spatial footprint [km] of an active blocking ridge — much larger than
    an individual storm (STORM_RADIUS_KM), matching the synoptic scale of a
    real blocking high."""

    # ------------------------------------------------------------------ #
    # 1.5-layer atmosphere: prognostic upper-level wind (atmosphere.evolve_wind_aloft)
    # ------------------------------------------------------------------ #
    wind_upper_pgf_amp: float = 90.0
    """Amplitude of the upper-level thermal pressure-gradient term
    (atmosphere.evolve_wind_aloft) -- opposite sign convention from the
    surface's thermal PGF (see evolve_wind_aloft's docstring: a warm column
    is thicker, so upper-level pressure is relatively higher over warm
    regions, inverted from the surface's "cold = high" pattern), and a
    larger amplitude than the surface term since real meridional
    temperature/pressure gradients strengthen with altitude up to jet
    level. Originally calibrated to just 8.0 (120-day mixed-terrain spinup
    at 32x64) so the layer merely came out "stronger than the surface" --
    but a real-world jet-stream diagnostic comparison (weekly zonal-mean
    profile over a full year, see jet-stream-vs-real-world memory) found
    that value produced only a ~2-8 m/s subtropical ridge, 5-10x weaker
    than Earth's actual 30-50 m/s jet cores; recalibrated to 40.0 at the
    time, bringing the NH jet-band core to ~24-29 m/s. A follow-up session
    (see jet-latitude-fix memory) found the jet was sitting ~10-15 deg
    equatorward of Earth's real position because of `wind_upper_pgf_amp`
    alone -- fixing that required widening `wind_upper_hadley_edge_deg`
    (below), which reduces the achievable core magnitude for a given amp
    (a wider suppression footprint means less domain-wide momentum builds
    up anywhere). Recalibrated again to 90.0 to restore a ~21-22 m/s core
    at the new, better-positioned latitude -- the ceiling is now ~21.7 m/s
    (amp beyond ~100 gives no further gain, per a direct sweep) rather than
    the old 24-29 m/s target, a real trade-off of position for magnitude."""

    wind_upper_hadley_edge_deg: float = 24.0
    """Gaussian half-width [deg] of the extra equatorial-suppression window
    applied only in atmosphere.evolve_wind_aloft (kept separate from the
    surface layer's `eq_window`, sigma=12 deg, tuned for a different reason:
    surface Ekman/frictional damping in the deep tropics). Real subtropical
    jets sit at the Hadley cell's poleward edge (~25-30 deg) because within
    the cell's footprint, direct meridional overturning -- not modeled by
    this layer's pure thermal-wind balance -- dominates over geostrophic
    dynamics; only beyond the cell edge does the free thermal-wind response
    this layer actually simulates take over. A full-year weekly-sampled
    zonal-mean diagnostic (see jet-latitude-fix memory) found the emergent
    jet peaking at ~18 deg in both hemispheres with the old, narrow (12 deg)
    window -- not because dT/dy itself peaked there (in the SH it actually
    peaked at a realistic ~46 deg), but because the thermal-wind response is
    dominated by the model's meridional temperature profile's ratio to the
    Coriolis parameter (which grows with latitude), and that profile is too
    gently-sloped across the whole subtropical band for the response to beat
    a 1/f decay -- so the emergent peak just tracked wherever the (too
    narrow) equatorial damping stopped suppressing it. Widening to 24 deg
    reshapes the response into a broad ~15-30 deg plateau (SH argmax moves
    cleanly to ~29.5 deg; NH is a near-flat tie across the same band) --
    much closer to Earth's real subtropical/polar-front jet position.
    Widening further (36+) over-suppresses and just clips the same
    monotonic-decay curve at a later point without adding real structure."""

    wind_upper_damping: float = 0.05
    """Rayleigh-friction rate [1/day] for the upper-level wind layer — much
    weaker than the surface's quadratic/terrain-enhanced drag
    (wind_drag_base/wind_drag_elev_scale), since the upper troposphere is
    nearly frictionless compared to the boundary layer."""

    wind_cell_relax_days: float = 3.0
    """Timescale [days] of the surface 3-cell circulation scaffold.

    ``atmosphere.evolve_wind`` combines resolved pressure-gradient, Coriolis,
    friction, and upper-layer momentum tendencies with a weak relaxation toward
    an Earth-like Hadley/Ferrel/polar surface pattern. A one-layer surface
    model cannot generate that complete overturning pattern by itself, so this
    is an explicit calibrated scaffold rather than an emergent circulation.

    The historical ``simulate_step`` default was 3.0 days. Keeping it here
    makes the value part of reproducible ``PlanetParams`` validation and allows
    real-terrain screens to override it without a hidden integrator argument.
    Smaller values strengthen the scaffold and must clear circulation,
    precipitation, drift, and cadence gates before any default change.
    """

    enable_two_level_thermally_direct_overturning: bool = False
    """Apply the existing thermally centred overturning primitive to the
    normal 1.5-layer wind state. The surface lower branch is paired with the
    compensating upper return branch, so the two represented layers have zero
    mass-weighted meridional flow. Experimental; default-off."""

    two_level_thermally_direct_overturning_speed_m_s: float = 0.0
    """Lower-branch speed [m/s] for the opt-in two-level overturning path.
    Zero keeps the gate inert even when enabled."""

    wind_prognostic_substep_days: float = 1.0
    """Run the real prognostic `evolve_wind`/`evolve_wind_aloft` integration
    (in inner chunks of this many days) instead of the cached diagnostic
    `generate_wind_field` snapshot during MONTHLY/ANNUAL (`update_wind=False`)
    substeps. `0.0` disables this and is an exact no-op -- MONTHLY/ANNUAL
    behave like the pre-2026-07-28 default (fast, diagnostic-wind-only) at
    today's speed. DAILY/WEEKLY (`update_wind=True`) already run the real
    prognostic wind every step and are unaffected by this field either way.

    Why this is now the default (2026-07-28, razor-sharp-biome-line session):
    `generate_wind_field`'s diagnostic wind is a smooth, mostly-analytic
    snapshot with almost no real per-cell terrain-driven divergence at a
    given latitude (measured: `subsidence_suppression` longitude std ~0.011
    across a real-terrain band that should show real heterogeneity, versus
    ~0.36 -- a 33x difference -- once this substepping is enabled). Without
    it, no amount of terrain-aware precip-rescale logic (see
    `_moisture_budget_precip_rescale`'s `target_cell_weight`) has any
    longitude signal to work with, producing a Koppen rainforest/desert
    boundary that sits at the *exact same latitude* for every column
    regardless of real terrain -- confirmed both as the mechanism and as the
    fix: enabling this at `1.0` moved the measured transition-latitude spread
    across a real-terrain longitude band from a single fixed value to 7.6-24.8
    degrees (std 2.6 deg) over a 12-month averaged run. Measured compute cost
    at 512x1024/`wind_block_size=8` was much smaller than originally feared
    (~6%, 27.7s -> 29.4s for 3 MONTHLY cycles) -- benchmark again at your own
    grid size if this matters for a latency-sensitive context (optimizer
    sweeps, long spinups), since `evolve_wind`'s internal physics is tuned
    around ~1-day steps and cost could scale differently elsewhere. Set to
    `0.0` to restore the old fast/diagnostic-only MONTHLY/ANNUAL behavior.

    This was originally a deliberate re-opening of a previously-resolved
    design question (PLAN.md Open Question 1: "cached relaxation target ...
    chosen for speed"), shipped opt-in in an earlier session and evaluated
    there against a different, broader biome-map-divergence metric (found
    "not the dominant driver" for that metric specifically -- see
    wind-prognostic-substep-gate-2026-07 memory) -- that finding is about a
    different question than this one (longitude variation of a specific
    latitude-band boundary) and doesn't contradict it. See
    razor-sharp-biome-line-precip-target-smoothing-2026-07-28 memory for the
    full investigation and validation."""

    precip_substep_days: float = 0.0
    """Override `simulate.py`'s `_PRECIP_SUBSTEP_DAYS` (1.0) chunk
    size used by `_generate_precipitation_substepped` to split a large
    outer `dt_days` into repeated `atmosphere.generate_precipitation` calls.
    `0.0` uses the calibrated module default; a positive value replaces it.
    Set `8.0` to reproduce the former one-call cadence for 6/7-day chunks.

    Why this exists: `generate_precipitation` has two independent, hardcoded
    per-call caps -- evaporation replenishment capped at a 1.5-day
    equivalent (`dt_evap = min(dt, 1.5)`) and the rain-out fraction capped
    at a 2.0-day equivalent (`remove_frac`'s `min(dt, 2.0)`) -- that throttle
    the *absolute* moisture cycled through a single call regardless of how
    many days it spans, then divide that capped total by the *full*
    (uncapped) `dt` to report a mm/day rate. A single 6-7 day MONTHLY/ANNUAL
    call therefore produces roughly the same absolute rain as a ~1.5-2 day
    call would, silently deflating the reported rate. The existing 8.0-day
    substep threshold doesn't fix this in practice: real GUI MONTHLY/ANNUAL
    dispatch (`main.py`'s `[(6.0, False)]*5` / `[(7.0, False)]*52`) never
    exceeds it (6.0, 7.0 <= 8.0), so `_generate_precipitation_substepped`
    never actually splits in real usage -- every real MONTHLY/ANNUAL
    precipitation call hit both caps directly before the default changed.
    A 32x64, 0.5-orbit convergence run reduced MONTHLY mean-precipitation
    error versus DAILY from 1.554 to 0.148 mm/day at 1-day cadence. This mirrors
    `wind_prognostic_substep_days`'s "reuse-already-tuned-physics-via-finer-
    substep" shape, applied to precipitation's rain-out mechanic instead of
    wind. See `atmosphere.generate_precipitation`'s `dt_evap`/`remove_frac`
    comments (atmosphere.py) for the caps themselves.

    `generate_precipitation`'s own `dt = max(dt_days, 1.0)` floors the
    per-call timestep at 1 day -- setting this below `1.0` buys no extra
    fidelity, only extra call overhead. `1.0` is the natural target (true
    per-day cadence, matches how many inner calls a 6/7-day outer step would
    need to reconstruct a real daily sequence). Has a real, non-trivial
    performance cost: profiled at roughly a 1.65-1.7x total per-step
    slowdown at MONTHLY speed (precipitation's own per-call cost is ~12% of
    total step cost at the default 1-call-per-outer-step, and scales
    ~linearly with the number of inner calls) -- benchmark before relying on
    it in any latency-sensitive context (optimizer sweeps, long spinups)."""

    temperature_substep_days: float = 0.0
    """Opt-in: split a large outer `dt_days` into repeated
    `simulate._evolve_temperature` calls of ~this many days each, instead of
    one call spanning the whole outer step. `0.0` (default) is an exact
    no-op. Inner calls advance all prognostic temperature fields and rebuild
    land/ocean seasonal targets for each fractional date, avoiding the former
    stale-`T_base_land` overshoot.

    The default remains off pending longer calibrated DAILY-vs-coarse
    convergence runs. Precipitation cadence is independently configurable
    with `precip_substep_days` and is likewise not enabled by default."""

    # ------------------------------------------------------------------ #
    # Derived convenience properties
    # ------------------------------------------------------------------ #

    @property
    def omega(self) -> float:
        """Planetary rotation rate [rad/s]."""
        return 2.0 * math.pi / (self.sidereal_day_hours * 3600.0)

    @property
    def obliquity_rad(self) -> float:
        """Axial tilt [radians]."""
        return math.radians(self.obliquity_deg)

    @property
    def surface_area_m2(self) -> float:
        """Total surface area [m²]."""
        return 4.0 * math.pi * self.radius_m ** 2

    @property
    def aerosol_forcing_w_m2(self) -> float:
        """Shortwave radiative forcing from stratospheric aerosols [W/m²].

        Uses the Lacis et al. approximation:  ΔF ≈ −25 × AOD.
        Typical values: background ≈ 0, Pinatubo 1991 ≈ −4 W/m².
        """
        return -25.0 * self.aerosol_optical_depth

    @property
    def reference_air_density(self) -> float:
        """Reference surface-air density [kg/m³] at 288.15 K."""
        return self.surface_pressure_pa / (self.gas_constant_dry * 288.15)

    def equinox_phase(self, day_of_year: float) -> float:
        """Wrapped orbital phase since northern vernal equinox [0, 2π)."""
        return (
            2.0 * math.pi
            * (
                (float(day_of_year) - self.vernal_equinox_day)
                % self.orbital_period_days
            )
            / self.orbital_period_days
        )

    def solar_declination(self, day_of_year: float) -> float:
        """Solar declination [radians] for a fractional orbital day."""
        return math.asin(
            math.sin(self.obliquity_rad) * math.sin(self.equinox_phase(day_of_year))
        )

    def solar_distance_factor(self, day_of_year: float) -> float:
        """Ratio of actual to mean Sun–planet distance at the given day.

        Returns ``r/a`` where ``a`` is the semi-major axis.
        TOA insolation scales as ``1 / factor²``.
        Solves Kepler's equation with bounded Newton iterations, then returns
        the exact elliptic-orbit radius ``r/a = 1 - e cos(E)``.
        """
        e = self.eccentricity
        if not 0.0 <= e < 1.0:
            raise ValueError("eccentricity must satisfy 0 <= e < 1")
        M = 2.0 * math.pi * (
            float(day_of_year) - self.perihelion_day
        ) / self.orbital_period_days
        M = (M + math.pi) % (2.0 * math.pi) - math.pi
        E = M if e < 0.8 else math.copysign(math.pi, M if M != 0.0 else 1.0)
        for _ in range(12):
            residual = E - e * math.sin(E) - M
            derivative = 1.0 - e * math.cos(E)
            step = max(-1.0, min(1.0, residual / derivative))
            E -= step
            if abs(step) < 1e-13:
                break
        return 1.0 - e * math.cos(E)

    def effective_solar_constant(self, day_of_year: float) -> float:
        """Solar constant corrected for orbital distance [W/m²]."""
        d = self.solar_distance_factor(day_of_year)
        return self.solar_constant / (d * d)

    def daily_mean_insolation(
        self,
        lat_rad: np.ndarray,
        day_of_year: float,
    ) -> np.ndarray:
        """Daily-mean TOA insolation Q(φ, day) [W/m²].

        Generalised version of ``temperature._daily_mean_insolation_Q`` that
        uses ``self`` (S0, obliquity, orbital period, eccentricity).
        Handles polar day/night and the exact poles correctly.

        Args:
            lat_rad: Latitude(s) in radians (scalar or array).
            day_of_year: Day of year (float; supports fractional days).

        Returns:
            Array of the same shape as ``lat_rad``, float32.
        """
        lat = np.asarray(lat_rad, dtype=np.float64)
        S0 = self.effective_solar_constant(day_of_year)
        delta = self.solar_declination(day_of_year)

        lat_safe = np.clip(lat, -math.pi / 2 + 1e-9, math.pi / 2 - 1e-9)
        cosH0 = -np.tan(lat_safe) * math.tan(delta)
        H0 = np.arccos(np.clip(cosH0, -1.0, 1.0))
        H0 = np.where(cosH0 <= -1.0, math.pi, H0)   # 24-h day
        H0 = np.where(cosH0 >= 1.0, 0.0, H0)          # polar night

        Q = (S0 / math.pi) * (
            H0 * np.sin(lat_safe) * math.sin(delta)
            + np.cos(lat_safe) * math.cos(delta) * np.sin(H0)
        )

        # Exact pole corrections
        pole_mask = np.abs(np.abs(lat) - math.pi / 2) < 1e-6
        if np.any(pole_mask):
            Q_pole = np.zeros_like(lat)
            np_mask = pole_mask & (lat > 0)
            Q_pole[np_mask] = S0 * max(0.0, math.sin(delta))
            sp_mask = pole_mask & (lat < 0)
            Q_pole[sp_mask] = S0 * max(0.0, -math.sin(delta))
            Q = np.where(pole_mask, Q_pole, Q)

        return np.maximum(0.0, Q).astype(np.float32)

    def coriolis_parameter(self, lat_rad: np.ndarray) -> np.ndarray:
        """Coriolis parameter f = 2Ω sin(φ) [rad/s].
        Sign flipped for retrograde rotators (rotation_direction = -1)."""
        return (
            2.0 * self.omega * float(self.rotation_direction)
            * np.sin(np.asarray(lat_rad, dtype=np.float32))
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# Singleton: Earth with present-day orbital parameters
# ---------------------------------------------------------------------------
EARTH = PlanetParams()

# ---------------------------------------------------------------------------
# Singleton: Mars — present-day orbital / physical parameters
# ---------------------------------------------------------------------------
# References:
#   Solar constant at Mars: 1361 / 1.524² ≈ 589 W/m²
#   Perihelion: Ls=250° ≈ day 477 of the Martian year (southern summer)
#   Surface pressure: ~636 Pa (global mean, varies ±10% with season/dust)
#   Atmosphere: 95% CO2, ~3% N2/Ar trace → mean molar mass ≈ 0.0435 kg/mol
#   Epsilon near-blackbody: thin atmosphere, modest CO2 greenhouse bands (~5 K effect)
#   has_liquid_water_ocean=False → ocean transport and sea-ice suppressed in simulate.py
MARS = PlanetParams(
    solar_constant=589.0,
    obliquity_deg=25.19,
    orbital_period_days=686.97,
    eccentricity=0.0934,
    perihelion_day=477.0,
    vernal_equinox_day=0.0,
    sidereal_day_hours=24.623,
    radius_m=3.3895e6,
    surface_gravity=3.71,
    # Olympus Mons, ~21.9 km -- 2.5x Earth's ceiling. Before this was
    # parameterized (ACCURACY_AUDIT.md C3) Mars terrain was silently rescaled
    # into Earth's 8.848 km range.
    max_elevation_km=21.9,
    # ~2.5 K/km: lower gravity and a CO2 atmosphere give Mars a much weaker
    # vertical temperature gradient than Earth's 6.5 K/km.
    lapse_rate_k_per_km=2.5,
    surface_pressure_pa=636.0,
    mean_molar_mass=0.0435,
    gas_constant_dry=191.0,     # R_univ / M_CO2 = 8314 / 44
    cp_dry=735.0,
    epsilon_equator=0.90,       # Near-blackbody; thin CO2 greenhouse adds ~5 K
    epsilon_pole=0.95,
    aerosol_optical_depth=0.0,
    ocean_fraction=0.0,
    has_liquid_water_ocean=False,
    rotation_direction=1,
    co2_baseline_ppm=1.0,       # CO2-dominated atmosphere; Earth formula not applicable
    co2_initial_ppm=1.0,
    co2_climate_feedback=0.8,   # No water-vapour amplification on dry Mars
    wv_greenhouse_factor=0.0,   # Negligible water vapour on Mars
    # Real Mars trace-gas composition (bulk is CO2, ~95.3%, filled in as the
    # remainder by atmosphere_composition() -- co2_baseline_ppm/co2_initial_ppm
    # above are greenhouse-formula placeholders, not real composition numbers).
    bg_n2_frac=0.0189,
    bg_o2_frac=0.00145,
    bg_ar_frac=0.0193,
    cloud_greenhouse_factor=0.0,  # No liquid water clouds
    ch4_baseline_ppb=0.0,
    ch4_initial_ppb=0.0,
    storm_pressure_amp_pa=40.0,  # Much thinner atmosphere plausibly weakens baroclinic transients
    trade_wave_pressure_amp_pa=25.0,  # Same reasoning, scaled with storm_pressure_amp_pa
    jet_meander_noise_amp=0.15,  # Weaker baroclinicity on a thin, dry atmosphere
    jet_block_pressure_amp_pa=60.0,  # Same reasoning, scaled with storm_pressure_amp_pa
    wind_upper_pgf_amp=54.0,  # Thin CO2 atmosphere: weaker vertical strengthening of gradients (kept at 0.6x Earth's default through both recalibrations)
    wind_upper_damping=0.08,  # Slightly more damped: thinner atmosphere, less inertia aloft
    # wind_upper_hadley_edge_deg left at the Earth default (24.0): no Mars-specific
    # jet-latitude diagnostic has been run, and the Hadley-cell-edge reasoning behind
    # the value isn't obviously Earth-specific.
    #
    # Land seasonal damping: reasoned from first principles, NOT measured -- no
    # Mars seasonal-cycle diagnostic exists. Earth's 0.75 represents surface heat
    # capacity plus the atmospheric heat transport that responds to a temperature
    # anomaly and damps it. Mars has ~0.6% of Earth's atmospheric mass and no
    # ocean, so both terms are small and its surface tracks radiative equilibrium
    # far more closely -- which is the whole reason Mars has such large seasonal
    # and diurnal swings. Inheriting Earth's damping would suppress exactly the
    # behaviour that distinguishes the planet, so this is one of the few places
    # where carrying the Earth calibration over is affirmatively wrong rather
    # than merely unvalidated (contrast the note above).
    land_seasonal_amplitude=1.0,
    # The default land-energy closure was calibrated only against Earth's CRU
    # benchmark.  Mars has no liquid-water surface exchange, so retaining this
    # Earth-specific tendency would be an unvalidated source of damping.
    enable_land_surface_energy=False,
    # land_transport_gain is left inherited at 0.5: it scales atmospheric heat
    # transport into land, and a thin atmosphere transports less, so if anything
    # Earth's reduced value is closer for Mars than the old 1.0 was.
    # land_seasonal_amplitude_maritime needs no override -- with ocean_fraction=0
    # the maritime-proximity field has no spread and `_maritime_transport_factor`
    # returns exactly 1.0 by its own guard.
)
