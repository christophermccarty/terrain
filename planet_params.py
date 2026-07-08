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
    or evaporation (same memory's wind-diagnostics check). Default stays 0.0:
    not yet clean enough to enable, but the remaining gap is now narrow and
    well-characterized rather than a fundamental mechanism failure."""

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
    soil_deep_gain_rate: float = 0.0
    """Fraction of precipitation that feeds the deep layer directly per day
    [1/day], independent of the surface layer's own state (see NOTE above).
    Default kept at 0.0 (exact no-op, matching moisture_advection_scale /
    pgf_continentality_amp's convention): calibrated directly against real
    terrain (2026-07) and found the deep layer amplifies whatever desert-vs-
    continental-interior precip differentiation already exists in its input,
    rather than creating it -- a controlled 20yr real-terrain comparison
    (deep layer on vs off) showed no measurable net effect at conservative
    gain rates, and pushing the gain rate up 20x made a desert box (Sahara)
    *wetter* (up to ~354 mm/yr, above its own realism target) without
    reliably helping continental interior. The real bottleneck is upstream --
    desert vs. continental-interior precipitation isn't reliably
    differentiated by the model in the first place (see known-physics-gaps.md)
    -- fixing that is a precondition for this knob to be useful, not
    something this knob can produce on its own. Left wired and tested as
    infrastructure for whoever picks that up."""

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

    # ------------------------------------------------------------------ #
    # Cloud radiative feedback (Feature 1)
    # ------------------------------------------------------------------ #
    cloud_greenhouse_factor: float = 0.12
    """Strength of high-cloud OLR trapping [dimensionless].
    High clouds (cold tops, T_air < 265K) reduce effective epsilon by this fraction
    times cloud fraction.  0.12 gives tropical cloud LW CRE ≈ +6–10 W/m².
    Set 0.0 to disable cloud greenhouse effect."""

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
    τ = 1/rate ≈ 10957 days ≈ 30 yr.  Slows surface warming to realistic TCR."""

    deep_ocean_depth_m: float = 3700.0
    """Mean abyssal ocean depth [m].  Used only for diagnostic OHC calculations."""

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

    def solar_distance_factor(self, day_of_year: float) -> float:
        """Ratio of actual to mean Sun–planet distance at the given day.

        Returns ``r/a`` where ``a`` is the semi-major axis.
        TOA insolation scales as ``1 / factor²``.
        Uses the first-order Kepler expansion (accurate to ~1 % for e < 0.2).
        """
        M = 2.0 * math.pi * (day_of_year - self.perihelion_day) / self.orbital_period_days
        nu = M + 2.0 * self.eccentricity * math.sin(M)
        e = self.eccentricity
        return (1.0 - e * e) / (1.0 + e * math.cos(nu))

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
        obliq = self.obliquity_rad
        gamma = 2.0 * math.pi * (float(day_of_year) - 80.0) / self.orbital_period_days
        delta = math.asin(math.sin(obliq) * math.sin(gamma))  # solar declination

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
    sidereal_day_hours=24.623,
    radius_m=3.3895e6,
    surface_gravity=3.71,
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
