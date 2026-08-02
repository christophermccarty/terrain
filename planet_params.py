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

    monsoon_east_margin_exemption: float = 1.5
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
    SE US, not a complete fix for all three."""

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
    not an arbitrary stopping point). Area-weighted Koppen breakdown moved
    cleanly toward Earth's real values in the same run: arid_pct
    22.1% -> 14.6% (real ~19-20%), tropical_pct (Af/Am/Aw) 8.9% -> 12.3%
    (real ~20%, was measured separately at 3.4% on the pre-fix 23.8yr save in
    test-npz-koppen-audit-2026-07-29 -- a large relative improvement, though
    still short of Earth's share). Sahara and the continental-interior boxes
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

    itcz_seasonal_response: float = 0.7
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
    planet regardless of how much of the area-fraction gap it closes alone."""

    itcz_seasonal_target_response: float = 1.7
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

    **Honest limitation, unchanged**: Aw+Am is still well short of Earth's
    ~18-20% even at 1.7 -- this knob converts Af->Aw/Am within an
    already-too-small tropical band; it cannot grow the band itself. The
    remaining gap is the moisture-budget rescale's other structural limits (see
    `docs/ACCURACY_AUDIT.md` A2/A5), not further tuning of this knob: pushing to
    2.0 overshoots Af to 3.95% (below Earth) while Aw+Am only reaches 8.71%."""

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
)
