# Prior-art implementation plan

This document converts `EXTERNAL_PRIOR_ART.md` into staged, promotion-gated
work.  A published formulation is a source of physics and validation, not an
automatic default for PlanetSim.

## 1. Orographic precipitation: Smith--Barstad linear theory

**Status: implemented offline; not coupled.** `orographic_linear.py` provides
the NumPy FFT transfer function and
`scripts/check_linear_orographic_theory.py` scores it on the bundled DEM.

The 256x512 fixed-10-m/s-westerly screen is deliberately not a promotion:
S Andes (3.45x) and Scandinavia (3.52x) show a usable footprint, while several
pairs miss their Earth targets.  The subsequent local-tile/model-wind screen
improves Sierra Nevada to 2.34x (its 2-5x target) and S Andes to 3.83x, but
still misses the remaining pairs.  Before coupling it to precipitation:

1. Run each pair in a local padded tile using a seasonally sampled, local mean
   model wind and stable/moist-stability diagnostics.
2. Require a raw W/L contrast of at least the pair's Earth lower bound in a
   majority of sampled seasons.
3. Couple only its positive, water-limited anomaly to the new column-water
   path.  Do not inject it upstream of the prescribed row target and expect it
   to survive rescaling.

The first seasonal local-wind/stability screen reinforces that caution: no
single tested stability (0.004, 0.006, or 0.008 s-1) recovers the western
mountain pairs, while the Himalaya overshoots its target (22.65-36.39x).
This is evidence to diagnose seasonal wind direction/persistence and the
background-rain coupling, not evidence to tune a universal stability constant.
The accompanying direction diagnostic confirms an upstream circulation issue:
the Cascades and Sierra Nevada tiles have strongly westward annual mean flow
(-16.1 and -21.9 m s-1 eastward component, respectively), despite the
eastward-positive convention and their expected midlatitude westerlies.

The -16.1/-21.9 m s-1 figures reproduce exactly on the current tree (256x512,
one spin-up year, default `--moist-stabilities`), so this is not a stale
number. Before treating it as a circulation bug, three diagnostic-artifact
explanations were ruled out. Area weighting is not the cause: a cos-latitude
area-weighted tile mean gives -16.15/-21.88 m s-1, statistically identical to
the unweighted -16.11/-21.90 the script reports, because these boxes only span
4 degrees of latitude. A short or seasonally biased sampling window is not the
cause either: the seasonal screen already samples all 12 months after a full
spin-up year, and the annual-mean-vector-to-mean-speed persistence ratio is
0.93 (Cascades) and 0.97 (Sierra Nevada) -- the westward flow is present in
essentially every month, not an artifact of averaging a reversing signal. A
sign-convention mismatch is not the cause either: `atmosphere.py` documents
`u` as eastward-positive throughout (matching `U_TARGET_MIDLAT = 11.5`, the
model's own westerly-jet target), and a direct zonal-mean check -- averaging
`state.wind_u` across all 512 longitudes at the same latitude bands, from the
same run -- gives +7.06 m s-1 at 44.5-48.5N (Cascades) and +3.73 m s-1 at
36-40N (Sierra Nevada). The model's own large-scale circulation has correctly
signed midlatitude westerlies at these exact latitudes; only the local
coastal-mountain tile reads westward. This also falsifies the "short window
catching a seasonal reversal" hypothesis a second way: there is no reversal to
catch, the discrepancy is a standing local-vs-zonal-mean split, not a temporal
sampling issue.

That local/zonal-mean split is a real, localized circulation feature, not a
bug in the tile extraction, and it is produced by `atmosphere.evolve_wind`'s
static terrain pressure-gradient term, `p_terrain = pgf_terrain_scale *
elevation` (effective scale 900 Pa, `simulate.py`'s
`wind_pgf_terrain_scale=900.0` times `PlanetParams.wind_terrain_pgf_scale=1.0`
by default), decomposed non-invasively via `evolve_wind`'s existing
`debug_fields` hook (a monkeypatch on `simulate.evolve_wind` capturing each
month's final substep; no production code changed). At the Cascades tile this
term's annual mean is +111.1 Pa over the windward+leeward cells and rises
almost monotonically west to east -- 8, 43, 100, 170, 206, 214 Pa at
representative columns from the open Pacific to the Rockies foothills -- so
`dp/dx > 0` across nearly the whole 17-degree-wide tile, not just at the
resolved crest. Because `pgf_u = -(1/rho) dp/dx`, that sustained positive
gradient forces a sustained *westward* wind spanning the tile, matching the
observed column profile of `wind_u` (near 0 far offshore, a trough of -36.5 m
s-1 just off the coast at -124.8 degrees, staying negative through the
interior, only turning positive again past -117 degrees). Sierra Nevada shows
the same shape at larger amplitude (terrain term annual mean +157.6 Pa, column
profile 0, 42, 105, 158, 235 Pa). A competing, physically sensible thermal
term is present at both ranges (mean -60.0 Pa at Cascades, -156.3 Pa at Sierra
Nevada, decreasing eastward -- i.e. correctly implying a cooler ocean and
warmer continental interior, which would drive onshore/eastward flow) but is
outweighed rather than absent: its west-to-east span is roughly half the
terrain term's at Sierra Nevada and less than half at Cascades. A back-of-
envelope geostrophic estimate using the measured `dp/dx` and `f` at 46.5N
(`f = 2 Omega sin(46.5 deg) = 1.06e-4 s-1`, `rho = 1.225 kg/m3`) gives a
properly rotated wind of only about 2.7 m s-1 for the steepest Cascades
segment -- roughly an order of magnitude below the 16-22 m s-1 the model
actually produces -- consistent with this single-layer solver settling into a
friction-dominated, down-gradient response rather than a geostrophically
turned along-contour jet at this grid scale.

The same mechanism explains the Himalaya's independent 22.65-36.39x overshoot,
just projected onto `v` instead of `u` because that pair's boxes are oriented
north-south. The terrain term's row profile (north to south, 39.0N to 19.3N)
rises from about 50 Pa near the Ganges-plain edge (25.7N) to a plateau above
600 Pa across roughly 32-37N, on the Tibetan-plateau side, before falling back off the
tile's north edge; its annual-mean pair-cell value is +389.8 Pa, by far the
largest of any pair tested. The resulting sustained `dp/dy` drives the
already-reported extreme -34.0 m s-1 `v` anomaly, which directly inflates the
Smith-Barstad orographic response past the pair's real-world bound rather than
merely mis-orienting it.

This was confirmed, not just inferred, using the model's own existing
`wind_terrain_pgf_scale` gate (`PlanetParams`, default 1.0, already unit-
tested in `testing/test_prior_art_kernels.py` to be a no-op at that default;
no code changed here). Re-running the seasonal screen at
`--terrain-pgf-scale 0.0` flips both western tiles to the physically correct
eastward sign and moves two of the doc's open problems substantially toward
their targets:

| Range | scale=1.0 (default) wind (u,v) | scale=1.0 W/L @0.004/0.006/0.008 | scale=0.0 wind (u,v) | scale=0.0 W/L @0.004/0.006/0.008 | Earth target |
|---|---|---|---|---|---|
| Cascades | (-16.1, 18.6) | 1.06 / 1.13 / 1.20 | (6.5, 18.1) | 0.89 / 0.88 / 0.86 | 3-6x |
| Sierra Nevada | (-21.9, -13.9) | 0.34 / 0.35 / 0.36 | (1.3, 13.2) | 1.24 / 1.27 / 1.30 | 2-5x |
| S Andes | (1.1, -10.3) | 1.76 / 1.82 / 1.86 | (3.6, -16.4) | 1.74 / 1.75 / 1.74 | 5-15x |
| Southern Alps | (7.5, -6.5) | 1.29 / 1.20 / 1.11 | (4.1, -4.7) | 1.37 / 1.35 / 1.32 | 4-12x |
| Scandinavia | (-6.0, -9.7) | 0.90 / 1.03 / 1.13 | (9.8, -4.9) | 3.92 / 4.03 / 4.06 | **2-4x (passes)** |
| Himalaya | (-4.0, -34.0) | 22.65 / 32.04 / 36.39 | (-2.9, -22.5) | 15.01 / 18.15 / 20.03 | **5-20x (passes/near)** |

Disabling the terrain term newly brings Scandinavia inside its target at every
tested stability and brings the Himalaya inside or to the edge of its target
(only the 0.008 s-1 row, at 20.03x, sits marginally outside). Cascades, Sierra
Nevada, S Andes, and Southern Alps still miss their targets even with the
correct wind sign, so disabling this term is necessary but not sufficient for
this section's promotion bar; the remaining shortfall on those four pairs is a
separate open question (magnitude/persistence of the approach wind, or the
transfer function's own conversion/fallout timescales, not diagnosed further
here).

The response to this parameter is not a clean monotonic dial, which matters
for what to do next. Sweeping intermediate scales at stability 0.006 s-1 shows
Cascades' `u` flip back negative and Himalaya's overshoot get *worse*, not
better, at intermediate values before both partially recover at scale 1.0:

| scale | Cascades (u, W/L) | Sierra Nevada (u, W/L) | Himalaya (v, W/L) |
|---|---|---|---|
| 0.00 | 6.5, 0.88 | 1.3, 1.27 | -22.5, 18.15 |
| 0.25 | 3.7, 0.95 | -6.5, 1.14 | -32.1, 28.18 |
| 0.50 | -7.1, 1.01 | -12.7, 0.68 | -35.3, 34.95 |
| 0.75 | -9.5, 0.95 | -17.4, 0.51 | -36.3, 36.07 |
| 1.00 | -16.1, 1.13 | -21.9, 0.35 | -34.0, 32.04 |

This non-monotonicity is expected once the term is understood as feeding a
one-year prognostic spin-up (jet meander index, blocking ridges, sea ice) that
is itself chaotic/history-dependent, not a static offset added to a fixed
wind field -- changing the scale changes the whole trajectory, not just its
endpoint. Only the two tested endpoints (0 and 1) are therefore evidenced;
an intermediate constant is not a validated fallback.

Because `wind_terrain_pgf_scale` is a global wind-model parameter (every grid
cell, every step, feeding temperature/precipitation/Koppen everywhere, not an
orographic-screen-only knob), whether disabling or redesigning it is a net
win cannot be answered by the six-pair screen alone. A compact check using
this project's standard harness (`scripts/run_real_terrain_validation.py`,
64x128, one spin-up year plus one evaluation year -- the same configuration
Section 2 cites) is informative and, unusually for this document, positive
across the board: group Koppen accuracy rises from 0.6737 to 0.7086, class
accuracy from 0.3885 to 0.4049, group kappa from 0.587 to 0.632, and
group-share MAE nearly halves (3.68 to 1.77 points), while global
precipitation is essentially unchanged (2.996 to 2.991 mm/day) and every
named-region accuracy in `region_group_accuracy` (Sahara, Kalahari, Atacama,
Canadian Prairies, US Midwest, Central Europe, SE US, East China, S Japan) is
numerically identical between the two runs. The gain is concentrated at the
equator (the -10:0 and 0:10 zonal bands jump from 0.673/0.730 to 0.906/0.862)
and in the arid/tropical land-percent split moving toward more realistic
values (arid 30.16% to 24.67%, tropical 20.30% to 24.29%). Both runs are
exactly reproducible (bit-identical on repeat, and the default run still
matches the tracked baseline via `--compare`), so this is not run-to-run
noise.

This is exactly the situation the ground rules for this task anticipate as a
judgment call rather than something to resolve unilaterally: a single compact,
short-duration screen is this project's own standing disqualifier for
promotion (see Sections 5-7, 10-11), and `wind_terrain_pgf_scale` was reached
by a diagnostic investigation into six orographic tiles, not by the
document's usual bounded-sweep-then-scale-up protocol. `wind_terrain_pgf_scale`
therefore remains at its default of 1.0, `orographic_linear.py` remains
uncoupled, and no other default changed. The concrete next step is to run
`wind_terrain_pgf_scale=0.0` through this project's full promotion protocol
(a bounded sweep, then the 128x256 five-year-spinup/five-year-evaluation CRU
benchmark) purely as a global wind-model candidate, independent of the
orographic work -- and, given the observed non-monotonicity, to also consider
redesigning the term itself (e.g. proportional to local relief/slope
magnitude with synoptic-scale smoothing, rather than a raw absolute-elevation
pressure wedge applied everywhere) rather than treating 0.0 as the only
alternative to 1.0. Only after that global question is settled does it make
sense to return to the four pairs (Cascades, Sierra Nevada, S Andes, Southern
Alps) that still miss their targets even with the term disabled.

## 2. Land: force-restore plus Penman--Monteith

**Status: gated replacement implemented; rejected by the first short A/B.**
`land_surface.py` supplies a two-reservoir force-restore step and a
moisture-dependent Penman--Monteith energy partition.  With
`enable_force_restore_land=true`, `simulate.py` bypasses the legacy land
seasonal-baseline blend and latitude-only cap; it persists a deep-soil
temperature state instead of stacking another correction onto that path.

At 64x128 with one year of spin-up and evaluation against CRU, the first
untuned run changed temperature RMSE from 6.28 to 7.92 C and Köppen
group/class accuracy from 0.674/0.388 to 0.587/0.235.  It improved the
temperature bias (+2.98 to +2.04 C) and precipitation log-RMSE (1.406 to
1.373), but fails the joint promotion gate.  A four-point bounded sweep over
15/60-day restore time and 1000/2000 s m-1 dry resistance also produced zero
accepted candidates (temperature RMSE 7.86-7.91 C; group skill 0.581-0.583).
It remains off.

Next steps are to calibrate physical ranges, not restore the old cap:

1. Feed the existing shallow/deep soil bucket explicitly into resistance and
   thermal inertia (the current initial range is conservative). The first
   energy diagnostic shows the deep reservoir remains much warmer than the
   surface in Canadian Prairies, Central Europe, and the US Midwest after the
   short run, so reservoir initialization/spin-up is a prerequisite to
   interpreting its restore-time sweep. Initializing an unknown deep reservoir
   from the surface rather than the legacy air state was implemented; its
   one-year A/B effect is negligible, so it fixes activation physics but is not
   the cause of the rejected climate result.
2. Sweep only published-meaningful ranges for deep heat capacity, restore time,
   and wet/dry resistance at 64x128, then validate the winner at 128x256 for
   five years plus five years.
3. Promote only if temperature RMSE improves without a loss of Köppen group
   or class accuracy under `climate_acceptance.evaluate_land_candidate`.

## 3. Precipitation: prognostic column water

**Status: finite-volume transport and an opt-in raw path implemented; rejected
by the first short CRU A/B.** `column_water.py` advances
`dW/dt = E - P - div(Wv)` with shared-face fluxes, exact spherical cell areas,
periodic longitude, closed poles, and a CFL substep limiter.  When
`enable_prognostic_column_water=true`, humidity is transported in that
column-water representation, precipitation removes local vapor, and both
row-target rescale variants are bypassed.  Supersaturation is explicitly
rained out, rather than being lost in a humidity clip.

At 64x128 with one year of spin-up/evaluation against CRU, the conservative
raw path reaches group/class KÃ¶ppen accuracy of 0.632/0.306, versus the
established compact baseline's 0.674/0.388.  The previous non-conservative
raw transport screen was effectively identical (0.633/0.303).  This is a
clear rejection of promotion, not a cue to tune the raw rainout conversion:
removing the prescribed target reveals a missing calibrated physical closure.
The default remains the established path.

The migration sequence is:

1. The condensate transport leg now uses the same finite-volume geometry when
   both gates are enabled, and an area-weighted vapor + condensate + rainout
   closure diagnostic spans the whole call.  Next, make that residual a formal
   validation threshold over the full sampled climate, not only a snapshot.
2. `column_water_use_bulk_condensate_rainfall` now supplies an isolated
   vapor -> condensate -> fallout experiment tied to resolved ascent and
   saturation.  Its first short CRU screen gives group/class skill 0.592/0.294
   (baseline 0.674/0.388) with 48% arid land.  A bounded two-point calibration
   (0.15/0.5-day condensation, 0.5-day fallout, full transport) accepts zero
   candidates: both worsen precipitation log-RMSE by about 0.71 and lower both
   KÃ¶ppen scores.  The next change must therefore be a consequential
   microphysics/design improvement (for example, a stability-aware resolved
   ascent/condensation trigger), not another timescale sweep.
   That trigger is now implemented as `enable_stability_aware_condensation`:
   a parcel lifts dry-adiabatically to its LCL, then moist-adiabatically to
   3.5 km; its CAPE-like buoyancy and resolved convergence jointly gate a
   conservative 70%-RH condensation relaxation.  Its first compact screen is
   also rejected (group/class 0.569/0.290; 1.39 mm/day global precipitation).
   A four-point 60/70%-RH × 25/100-J-kg-1 CAPE-scale sweep accepts zero
   candidates; its best precipitation log-RMSE regression is +0.73 and group
   skill loss is -0.07.  Therefore the one-layer stability proxy is useful,
   tested infrastructure but insufficient as the physical closure: the next
   consequential step is a vertically resolved convective adjustment (or a
   calibrated two-layer moist-static-energy model), not further threshold
   tuning.
   The first two-layer thermal-memory implementation is now also present:
   `enable_two_layer_convective_adjustment` persists a mid-tropospheric
   temperature, relaxes it toward the resolved 3.5-km lapse profile, and adds
   latent heating from condensation while retaining the single conserved water
   column.  It runs and preserves state correctly, but its first compact CRU
   A/B is decisively rejected (0.63 mm/day global precipitation; KÃ¶ppen
   group/class 0.483/0.224).  The result is too far from the baseline to
   justify a parameter sweep.  A viable next version would need explicit
   upper-layer moisture/enthalpy transport and coupling back to the resolved
   air-temperature dynamics, not merely a longer midlevel thermal memory.
   The first conservative increment of that work is now implemented but also
   rejected: with all three existing gates plus
   `enable_two_layer_convective_adjustment`, `midlevel_humidity` is a
   transported partition of the *same* total column water (not an extra water
   source). Surface evaporation enters the lower partition, the lower and
   upper partitions use their respective resolved winds, and resolved ascent
   entrains lower vapor into the upper partition. A dedicated unit test
   verifies both transport conservation and the full vapor/condensate/rainout
   column budget. In making that test pass, a pre-existing raw-column error
   was also fixed: saturation limiting may not be passed to the transport
   kernel as negative evaporation, because explicit supersaturation rainout
   owns that sink.

   This remains only a moisture-transport increment, not a complete
   moist-static-energy closure: upper vapor does not yet feed a resolved
   upper-layer phase-change or temperature tendency. Its one-year 64x128 CRU
   A/B is decisively worse (0.036 mm/day global precipitation, 3.188
   precipitation log-RMSE, 0.017 precipitation correlation, and KÃ¶ppen
   group/class 0.423/0.233; baseline 1.406, 0.463, and 0.674/0.388). It
   therefore receives no calibration sweep and remains opt-in. The evidence
   says the next consequential design must be a genuinely coupled vertical
   thermodynamic model with upper-layer condensation/detrainment and energy
   feedback, rather than further tuning of this split-column proxy.

   That active-layer increment is now implemented and evaluated. The upper
   layer has pressure-adjusted saturation, conservatively condenses moisture
   above the stability RH threshold (with explicit supersaturation removal),
   retains non-fallen condensate, and falls out on the existing condensate
   time-scale. Resolved convergence entrains lower vapor upward while the
   divergent branch detrains it back down. A mass-weighted exchange transfers
   the persistent midlevel temperature anomaly into the resolved air state and
   removes the companion anomaly from the midlevel reservoir, so latent heating
   is no longer diagnostically isolated. All of this remains nested behind the
   existing experimental gates.

   Its one-year 64x128 CRU result is a substantial recovery from the passive
   split-column screen but still a rejection: 2.553 mm/day global
   precipitation, 2.838 precipitation log-RMSE, -0.094 precipitation
   correlation, and KÃ¶ppen group/class 0.512/0.282. The compact baseline is
   1.406, 0.463, and 0.674/0.388 respectively. Temperature RMSE improves
   slightly (5.948 C versus 6.279 C), but the precipitation and biome gates
   all fail. This does not warrant tuning the exchange constants: the remaining
   missing process is vertical structure beyond one upper reservoir (including
   detrainment into cloud layers and its radiative/large-scale circulation
   coupling), not a choice among nearby two-layer time-scales.

   A further midlevel-cloud increment was tested rather than assumed: the
   suspended condensate reservoir now follows the resolved upper wind and is
   converted into an additional persistent cloud-fraction contribution before
   radiation is evaluated. This closes the obvious inconsistency in which an
   upper cloud reservoir was transported at the surface and had no radiative
   signature. It passes dedicated transport-layer and cloud-feedback tests,
   but its compact CRU screen is marginally worse than the active two-layer
   result (group/class 0.507/0.279, precipitation log-RMSE 2.894, correlation
   -0.124). It is rejected without a weight sweep. The remaining problem is
   not missing cloud visibility; it is the lack of vertically resolved
   large-scale circulation and cloud detrainment structure that determines
   where the condensate forms in the first place.

   The first bounded pressure-coordinate circulation increment is now present:
   spherical horizontal divergence is diagnosed independently at the lower and
   upper wind levels, and their centred continuity difference produces a
   midlevel pressure velocity. Its sign drives conservative entrainment or
   detrainment and gates upper-layer condensation. This is a genuine
   mass-continuity relation rather than the former independent surface-ascent
   and descent proxies. The short CRU result is the best in this experimental
   family (group/class 0.521/0.297; precipitation log-RMSE 2.757,
   correlation -0.073), but it remains well below the baseline and therefore
   is rejected. A same-protocol ablation with its condensate-radiative weight
   set to zero is effectively unchanged (0.521/0.297), isolating the remaining
   defect to circulation/condensation placement rather than cloud radiation.
   The next valid step is a prognostic, vertically resolved circulation model
   (or adoption of an external dynamical core); no local closure parameter can
   restore the missing spatial precipitation correlation.

   The native path has now started with an opt-in three-reservoir pressure
   column (`pressure_column.py`): lower, 3.5-km midlevel, and 8-km upper vapor
   and temperature states; two pressure-coordinate continuity interfaces; and
   conservative vertical donor transfers. Both mid and upper reservoirs have
   pressure-adjusted saturation/phase-change paths, and latent heat from both
   exchanges with the resolved air state. Unit tests cover partition
   conservation, both interface signs, state persistence, upper
   supersaturation/rainout, and upper thermal feedback.

   The first uncalibrated three-level CRU A/B substantially improves group
   structure (0.550 group accuracy; log-RMSE 2.263), and coupling upper thermal
   anomalies to resolved air improves it further to 0.601 group / 0.311 class
   accuracy, with group-share MAE 5.23 percentage points. It is nonetheless
   rejected: global precipitation is 6.18 mm/day, precipitation log-RMSE is
   2.155, and correlation only 0.082 (baseline: 1.406 and 0.463). Reducing the
   initial upper-vapor share from 15% to 5% is effectively inert after spin-up,
   confirming that the excess is the column's dynamical equilibrium, not its
   initialization.

   The next native increment is now implemented: `PlanetState` persists an
   independent middle-level `u/v` wind, evolved by the free-tropospheric
   momentum kernel with intermediate thermal forcing, stronger Rayleigh
   damping, and weak relaxation to the adjacent resolved levels. Its spherical
   divergence is passed directly to both pressure interfaces; it is no longer
   inferred as a surface/upper interpolation. The compact CRU screen improves
   the three-level result to 0.638 group / 0.328 class accuracy, a 3.57-point
   group-share MAE, 1.953 monthly precipitation log-RMSE, and 0.192 monthly
   log-correlation. This is a meaningful recovery over the interpolated
   three-level column (0.601 / 0.311, 2.155, 0.082), but remains rejected
   against the baseline (0.674 / 0.388, 1.406, 0.463): global precipitation
   rises further to 6.61 mm/day versus 3.00 mm/day. The native pressure-column
   architecture is therefore physically more complete but must stay opt-in
   while its resolved circulation/rainout equilibrium is corrected.

   A source-decomposition diagnostic then found that the apparent rainout
   defect was mostly a hidden final saturation safeguard: in a stationary
   three-level sample, about three fifths of the rain came from direct
   `q - qsat` vapor removal, bypassing both condensate residence time and
   latent heating. `stability_aware_condensation` now transfers unavoidable
   supersaturation into the conserved condensate reservoir before fallout.
   This improves the compact screen to 0.658 group accuracy, 3.09-point
   group-share MAE, 5.97 C temperature RMSE, and 1.806 precipitation
   log-RMSE, but a one-day fallout lifetime makes cloud fraction unrealistically
   high (0.459). A six-hour fallout screen restores cloud fraction to 0.170
   and reaches 0.691 group accuracy, but leaves global precipitation at
   7.03 mm/day and distorts group shares; neither is promotable.

   The same diagnostic identified an unbounded humidity-deficit evaporation
   flux as the remaining global-water source. An opt-in daily-mean radiative
   energy cap is now applied to raw conserved-column evaporation, with explicit
   atmospheric transmission, surface albedo, cloud attenuation, and latent
   heat share. It reduces global precipitation to 2.75 mm/day (baseline 3.00),
   confirming the source diagnosis. The first six-hour-fallout combination is
   too dry/cloud-poor (cloud 0.057; group accuracy 0.632); the one-day lifetime
   restores cloud (0.208) but collapses tropical rainfall (group accuracy
   0.536). These are calibration-direction results, not default changes: the
   remaining native work is a layer-aware cloud/precipitating-condensate split
   and an energy closure that includes longwave/sensible heat, rather than a
   single global shortwave cap.

   A first, mass-preserving radiative transition is now available behind
   `enable_cloud_precipitating_condensate_partition`: only a capped suspended
   cloud-water portion of the existing bulk reservoir affects cloud optical
   cover; excess mass remains in the unchanged fallout budget. It restores
   the energy-limited one-day-fallout screen to realistic global precipitation
   (2.63 mm/day) and cloud fraction (0.148), but still collapses the tropical
   group (overall group accuracy 0.535). This rejects a radiative-only split:
   the next implementation must persist separate cloud and precipitating
   hydrometeor reservoirs with their own conversion, transport, and sedimentation
   tendencies, rather than merely changing the radiation diagnostic.

   That separate-reservoir kernel is now implemented and persisted behind
   `enable_separate_precipitating_hydrometeors`: cloud condensate is retained
   up to a physical cloud-water threshold, excess autoconverts into an
   independently stored hydrometeor reservoir, and only that reservoir falls
   out. Both full and coarsened precipitation paths carry the state, and the
   whole-column closure includes both reservoirs. Its first CRU screen closes
   global precipitation well (2.69 mm/day) but remains under-cloudy (0.112)
   and weak in the tropics (0.593 group accuracy, 9.31-point group-share MAE).
   The architecture is retained; the next bounded experiment varies only
   cloud retention and autoconversion before touching circulation again.

   That bounded retention experiment (0.002 retained cloud mixing ratio and
   0.5-day autoconversion) restores cloud to 0.197 but worsens group accuracy
   to 0.537 and leaves only 2.86% tropical land. The original separate-reservoir
   setting remains the less-bad physical closure (0.593 group accuracy,
   2.69 mm/day). This rejects cloud-residence tuning as the main defect: the
   remaining priority is the placement of tropical ascent/subsidence in the
   resolved three-level circulation, not another microphysics sweep.

   A first opt-in migrating diabatic-ascent contribution was tested at 0.5
   relative to normalized horizontal convergence. It preserves the corrected
   global water cycle (2.69 mm/day) but slightly lowers group accuracy to
   0.588, so a uniform ITCZ ascent floor is rejected. The deficit instead lies
   in how ascent couples to instability and condensation, not simply in the
   magnitude of an imposed tropical vertical velocity.

   Halving the CAPE activation scale from 50 to 25 J/kg is likewise neutral
   (0.593 group accuracy, 2.68 mm/day). This rejects a condensation-threshold
   explanation and points back to the pressure-interface mass-exchange
   formulation itself as the next diagnostic target.

   Replacing the empirical vertical exchange timescale with a conservative
   finite-volume pressure flux, `|omega| dt / dp`, is the first material
   circulation recovery: the separated-reservoir screen rises from 0.593 to
   0.612 group accuracy and improves group-share MAE from 9.31 to 8.51 points
   while holding global precipitation at 2.67 mm/day. It remains below the
   baseline, but this establishes flux-form exchange as the active native path;
   the next bounded search is middle-level momentum forcing, which directly
   determines the two interface divergences.

   Middle-wind forcing has a shallow optimum: 80% of the upper-level thermal
   forcing improves the flux-form result to 0.615 group accuracy and 8.44-point
   share MAE; 110% reverses that gain (0.606). The 80% setting is retained for
   the next diagnosis, which separates tropical moisture supply from horizontal
   and vertical transport rather than extending this momentum sweep.

   The surface energy cap initially omitted atmospheric downwelling longwave.
   Adding an opt-in 15 W/m² term supplies this missing physical energy and is
   the best native candidate so far: 2.85 mm/day global precipitation, 0.616
   group accuracy, and much stronger ±10° tropical skill. The next bounded
   screen increases only this longwave term to establish whether the recovered
   tropical moisture supply has already reached its useful range.

   A 30 W/m² upper bracket exactly restores the 3.00 mm/day compact global
   precipitation magnitude, but group accuracy falls to 0.636. The 25 W/m²
   result is therefore retained as the local screen winner. A new gated
   Brutsaert humidity/emissivity plus cloud adjustment now supplies the next
   experiment: it adds only longwave back-radiation above the dry-sky energy
   already implicit in the cap, rather than treating the 25 W/m² diagnostic
   constant as a physical default.

   The CRU-scored emissivity-only trial is rejected (0.610 group accuracy,
   2.005 precipitation log-RMSE, and 2.73 mm/day): its humidity/cloud contrast
   is physically preferable to a constant but too weak and too localized to
   restore the moisture supply. The next isolated missing tendency is
   hydrometeor advection during their finite sedimentation lifetime. The new
   default-off conservative transport gate uses the resolved middle/cloud-layer
   wind for at most one fallout timescale, avoiding both stationary rainfall
   and the unphysical full-month displacement of falling rain.

   That CRU transport trial modestly improves precipitation log-RMSE (1.960 to
   1.909) but worsens group/class skill to 0.626/0.283 and group-share MAE to
   7.37 points, so it is rejected as a standalone improvement. The next
   structural experiment is an opt-in, conservative simplified Betts--Miller
   humidity relaxation on resolved ascending columns (2-hour target-RH
   adjustment), rather than another transport-distance sweep.

   Its first CRU screen is rejected without a parameter sweep: temperature
   RMSE improves slightly (6.14 to 6.00 C), but tropical land collapses to
   0.98%, group accuracy to 0.515, and precipitation log-RMSE to 2.751. This
   humidity-only half is intentionally retained as a gated research kernel,
   not promoted as "Betts--Miller": a complete implementation requires the
   paired moist-temperature relaxation and an independent aquaplanet
   circulation target. The next work therefore instruments the latter before
   any RH/timescale calibration can disguise the missing dynamics.

   Replacing the uniform target with a physically ordered 85%/70%/50%
   lower/mid/upper RH profile does not rescue the formulation: the compact CRU
   screen falls further to 0.507 group accuracy, 0.48% tropical land, and
   2.37 mm/day. This ends the native local-closure sweep. The 25 W/m²
   flux-form candidate remains the best complete native experiment, but its
   1.960 precipitation log-RMSE and 0.188 log-correlation versus the
   baseline's 1.406/0.463 establish that the remaining defect is horizontal
   circulation, not a missing fallout, radiation, or convective timescale.

   At 25 W/m² the same candidate reaches 2.97 mm/day, essentially the compact
   baseline's 3.00 mm/day global water-cycle magnitude, while Köppen group
   accuracy rises to 0.644 (class accuracy 0.300). This is the best complete
   three-level physical path to date, though it remains below the default
   baseline's 0.674/0.388 group/class score and has a 6.61-point group-share
   MAE. The longwave term is therefore a diagnostic proxy for the missing
   atmospheric back-radiation closure, not yet a default parameter; bracket it
   locally, then replace its constant value with a humidity/cloud-dependent
   emissivity formulation if the improvement persists.
3. Add area-weighted global and regional budget tolerances to the validation
   contract; use them to reject any clipping or untracked sink/source.
4. Reintroduce calibration, if still needed, only as a weak diagnosed
   relaxation; remove it after the physical closure clears the CRU gate.

## Validation contract

## 4. Pressure-column mass closure diagnostic

The native three-level path now exposes a circulation scorecard (surface,
middle, and upper jet latitude; Hadley-edge sign reversal; and pressure-
interface omega statistics). A default-off mass-weighted divergence closure was
added before diagnosing the two pressure-interface velocities. Its first compact
32x64 CRU screen is rejected: global precipitation rose to 6.13 mm/day,
monthly precipitation log-RMSE to 2.282, and log-correlation fell to 0.072
(baseline 1.305/0.484 at that grid). Köppen group agreement also fell from
0.641 to 0.597. The closure gate remains off.

This distinguishes a constraint from a solution: enforcing column-integrated
continuity only in the vertical exchange leaves the diagnosed upper jet and
Hadley flow uncorrected, then turns their residual into excessive ascent. The
next acceptable increment must evolve horizontally divergent wind/mass flux
with a compatible pressure solve, not redistribute the existing divergence
into the upper layer algebraically.

## 5. Resolved horizontal mass-flux closure

That next increment is now available behind
`enable_three_level_horizontal_mass_flux_closure` (default `False`).  After
the lower, independent middle, and upper winds have evolved, it diagnoses
their 0.40/0.35/0.25 mass-weighted divergence residual and applies a bounded
divergent correction only to the upper wind.  The non-zonal part is inverted
directly in longitude Fourier modes and the zonal-mean part by a small
meridional least-squares solve.  This is a resolved horizontal transport
closure, not an algebraic reassignment of the divergence used to diagnose
omega.  Its divergence operator is regression-tested to be numerically
identical to the production `flux_divergence_spherical` operator, including
the north-to-south latitude-row sign convention.

The first production-compatible compact (32x64, one-year spin-up plus one
year evaluation) CRU comparison for the retained three-level physical
candidate improved composite reference error from 0.826 to 0.560, Köppen
group agreement from 0.492 to 0.528, group-share MAE from 10.00 to 9.11
points, and reduced polar precipitation from 10.60 to 9.64 mm/day while
holding global precipitation near 2.73 mm/day.  A 64x128 one-year screen was
stable: 0.425 composite reference error, 0.642/0.299 Köppen group/class
agreement, 6.36-point group-share MAE, and 2.95 mm/day global precipitation.
The upper jets remain too equatorward (~+/-30 degrees), and the upper Hadley
edge is incomplete in the Northern Hemisphere, so this is retained as an
experimental closure rather than promoted to Earth defaults.

`PlanetState` now retains both diagnosed pressure-interface velocities
(`omega_lower_mid_pa_s` and `omega_mid_upper_pa_s`), and the real-terrain
report includes their mean/RMS alongside lower/middle/upper jet and Hadley
diagnostics.  The remaining validation gate is the existing longer 128x256
multi-year CRU benchmark, followed by diagnosis of the upper-level momentum
and Hadley-cell geometry if the compact-resolution improvement does not
survive.

## 6. Final high-resolution closure decision

The 128x256 one-year screen reached 0.664/0.344 Koppen group/class agreement
and 4.90-point group-share MAE, but also revealed excessive interface-omega
RMS (32.6/28.5 Pa s-1) and 3.22 mm/day global precipitation. A conservative
one-pass divergence filter reduced omega but degraded the compact composite
score (0.560 to 0.909), so it remains a default-off rejected diagnostic.

The prescribed 128x256 five-year spin-up plus five-year evaluation completed
at 0.667/0.351 group/class agreement, 5.36-point group-share MAE, 3.18
mm/day global precipitation, and 0.688 composite reference error. The
closure is numerically stable and benefits the precipitation/Koppen pattern,
but is not promotable: the Northern Hadley edge contracts to about 21 degrees,
the upper Northern jet remains near 20 degrees, and pressure-interface omega
remains too large (30.6/26.8 Pa s-1). Earth defaults therefore remain
unchanged.

The next native step requires a consequential momentum-design decision:
either add a balanced pressure-level geostrophic/thermal-wind solver with
scale-aware divergence damping, or use the optional external-dycore workflow
strictly as the circulation provider while PlanetSim supplies surface and
hydrology physics. Repeating the same horizontal closure is not warranted.

## 7. Follow-up balanced-target and transport diagnostics

The external prior-art review motivated a default-off balanced thermal-wind
upper-wind target. It derives zonal-mean vertical shear from the resolved
meridional temperature gradient and hydrostatic pressure interval, with a
smooth Hadley-cell taper, rather than treating a finite Coriolis parameter at
the equator as a tunable numerical fix. At 0.10 per-step relaxation it did not
move the compact upper jet and degraded the composite CRU score to 0.845. At
0.50 it moved the upper jets to about +/-42 degrees, but degraded the score
further to 0.917. The target is therefore retained as a testable default-off
kernel and rejected as a standalone climate solution.

The circulation report now explicitly includes lower-level horizontal
divergence RMS and column-scaled cross-equatorial moisture, latent-energy, and
dry-static-energy transport. Together with persisted lower-mid and mid-upper
omega, middle/upper jet latitude, and Hadley edge, this makes the next
momentum experiment falsifiable against the external-dycore precipitation
reference rather than relying only on precipitation and Koppen scores.

The direct native-versus-ExoPlaSim monthly comparison is now available in
`external_dycore.score_native_against_external_dycore`. On the 32x64 retained
closure candidate, its annual log-precipitation correlation against the 30-year
T21 external reference is -0.044 (annual log-RMSE 1.567). This is a hard
falsification signal: the external reference itself reaches 0.744 against CRU,
so the native candidate's apparently improved compact Koppen score does not
represent correct large-scale precipitation placement. Do not promote the
closure or continue a local parameter sweep; the next work must supply a
genuinely balanced large-scale circulation or deliberately adopt the external
circulation field as an offline provider.

Default Earth behavior is unchanged by every gate above.  Unit/persistence
coverage lives in `testing/test_prior_art_kernels.py` and
`testing/test_land_surface_energy.py`.  No experimental route becomes the
default from a short-run result; it must first clear the stored 128x256,
five-year-spinup/five-year-evaluation CRU baseline.

## 8. Native balanced three-level core (experimental)

The native path now has three explicitly separated pieces, all default-off and
with no ExoPlaSim runtime dependency: hydrostatic pressure-level geopotential
and geostrophic/ageostrophic wind targets at lower, middle, and upper levels;
a projection that removes the divergence-free zonal-mean column throughflow
left unconstrained by the spherical divergence inversion; and a thermally
centred, three-level mass-conserving overturning anomaly.  The overturning
cell centre is diagnosed from the model's tropical zonal-mean temperature
maximum rather than being fixed to an Earth latitude.  Its lower branch
converges on that thermal equator and its middle/upper return branches cancel
the lower mass flux exactly at every latitude.

The new full-column diagnostics exposed why this was necessary.  The retained
horizontal-closure candidate carried roughly 630 PW of spurious
cross-equatorial dry-static transport.  The null-mode projection removes that
unphysical pole-to-pole mass throughflow.  A fully applied balanced state
reduces interface-omega RMS to 0.024/0.559 Pa s-1 and makes cross-equatorial
transport effectively zero, but is too dry (0.55 mm/day) because geostrophy
alone has no direct overturning.  Adding a 0.5 m s-1 thermally direct lower
branch raises compact CRU precipitation log-correlation to 0.271 (from 0.100
for pure balance) while retaining small lower-interface omega, but remains too
dry (0.73 mm/day), has -7.0 PW cross-equatorial energy transport, and reaches
only 0.432/0.236 KÃ¶ppen group/class agreement.  It is therefore not promoted.

The same one-year 32x64 screen improves annual precipitation log-correlation
against the offline T21 ExoPlaSim reference from the earlier retained closure's
-0.044 to +0.191.  Monthly correlation remains approximately zero, so this is
evidence that the circulation direction is better, not evidence that the
native core is complete.  The next native increment must close the resolved
three-level moist-static-energy transport (including pressure/geopotential
work) and derive the overturning amplitude from diabatic heating rather than a
fixed branch-speed experiment.  Earth defaults and standalone PlanetSim
behavior remain unchanged.

## 9. Diabatic-strength overturning diagnostic

The fixed-speed overturning experiment now has a default-off replacement:
`enable_native_balanced_diabatic_overturning`.  It converts the persistent
midlevel latent-heating temperature anomaly above its lapse-rate reference
into a heating rate, pressure velocity using resolved static stability, and a
meridional lower-branch speed through continuity over the thermally diagnosed
Hadley width.  The resulting velocity is capped only as a numerical physical
bound; it has no prescribed Earth circulation-speed target.

The first 32x64 one-year spin-up/evaluation screen is stable but rejected for
promotion: 0.66 mm/day global precipitation, 0.214 CRU monthly precipitation
log-correlation, 1.826 log-RMSE, and -3.80 PW cross-equatorial dry-plus-latent
transport.  This is better constrained than pure geostrophy, but weaker than
the explicit 0.5 m/s diagnostic overturning.  The remaining implementation
task is to use a full moist-static-energy tendency (including geopotential
energy and resolved radiative heating) rather than only the relaxed midlevel
temperature anomaly.  Default Earth behaviour remains unchanged.

## 10. Full moist-static-energy overturning budget

That remaining task is now implemented behind a new, separately gated closure:
`enable_native_balanced_moist_static_energy_overturning`
(`moist_static_energy_overturning_speed` in `balanced_dynamics.py`).  It sums
two independently diagnosed heating terms before converting to pressure
velocity, rather than reusing only the persistent midlevel-anomaly memory:

1. **Latent heating from actual resolved condensation.**  The previous step's
   precipitation field converts directly to `L * P / (cp * column mass)`,
   falling back to the old midlevel-anomaly proxy only when no precipitation
   field is available (e.g. before the column-water path spins up).
2. **A resolved radiative/thermal heating rate.**  The lower-layer temperature
   relaxes toward the model's own seasonal radiative-plus-transport
   equilibrium target (the same `_compute_T_base_ocean_full()` reused
   elsewhere in the native-balanced wind block) on a new
   `native_balanced_mse_radiative_relaxation_days` timescale.  This is a
   genuinely resolved, latitude/day-of-year/planet-parameter-dependent target,
   not a fixed constant.

The mass-conserving three-level structure is unchanged: the diagnosed scalar
speed still feeds `thermally_direct_overturning`, whose lower/middle/upper
branches cancel exactly regardless of how the speed was diagnosed.  The
meridional-transport diagnostic was separately extended so dry-static-energy
transport is `cp*T + g*z` (via `pressure_level_geopotential`) rather than
`cp*T` alone, wired through `circulation_scorecard` and into
`real_terrain_validation.summarize_real_terrain_climate`.

The 32x64 one-year spin-up/evaluation screen, run three ways against the same
retained candidate to isolate each term's contribution:

| Variant | Global precip (mm/day) | CRU log-corr | CRU log-RMSE | Köppen group/class | Composite ref. error | Cross-eq. total transport (PW) | ExoPlaSim log-corr |
|---|---|---|---|---|---|---|---|
| `enable_native_balanced_diabatic_overturning` (baseline) | 0.66 | 0.214 | 1.826 | 0.432/0.233 | 0.682 | -1.91 | 0.138 |
| MSE closure, latent term only (`native_balanced_mse_radiative_relaxation_days=1e6`, radiative term nulled) | 0.70 | 0.245 | 1.770 | 0.435/0.236 | 0.666 | -2.74 | 0.170 |
| MSE closure, latent + radiative (default 10-day radiative relaxation) | 1.08 | **0.362** | **1.523** | **0.462/0.240** | **0.580** | **-10.93** | **0.256** |

Replacing the anomaly-memory proxy with actual condensation-driven latent
heating (the latent-only row) is a clean, modest, well-behaved improvement on
every axis, including cross-equatorial transport (-1.91 to -2.74 PW, the same
order of magnitude as the baseline).  Adding the resolved radiative term is
where the picture splits: it produces the best CRU correlation, log-RMSE,
Köppen skill, composite error, and external ExoPlaSim correlation of this
entire experimental family (Sections 8-10 combined) -- but it also nearly
quintuples cross-equatorial energy transport, from -2.74 PW to -10.93 PW.
That magnitude is worse than the -7.0 PW that already sank the fixed-speed
0.5 m/s experiment in Section 8, so by this project's established gate (a
short-run skill gain is not evidence on its own; the full-column energy
diagnostic must also pass) the combined closure is **not promoted**, despite
being the strongest compact skill result to date.

The ablation isolates the mechanism: `_compute_T_base_ocean_full()` was built
to give the *surface* field realistic hemispheric asymmetry (AMOC/ACC bonus
terms, an explicit `_sh_factor` hemisphere scaling, seasonal lag) — useful as
a temperature target, but reused here as a *heating-rate* target it hands the
overturning kernel a persistent, already-asymmetric forcing to amplify.  Part
of the correlation gain is therefore plausibly "encoding the answer" (the
same real-world asymmetry the transport parameterization already assumes)
rather than emergent physics, echoing earlier retracted results in this
project's history where a pattern-matching gain masked an unphysical
mechanism.  The next candidate radiative term should be a genuine top-of-
atmosphere equilibrium target (`temperature.temperature_kelvin_for_lat`
directly, without the ocean-transport bonus terms) so the ablation can
distinguish real radiative-heating skill from borrowed transport asymmetry
before any further resolution/duration scale-up.  Both overturning gates and
the geopotential-energy diagnostic extension remain default-off; Earth
defaults and standalone PlanetSim behavior are unchanged.  Unit coverage adds
five `moist_static_energy_overturning_speed` tests (zero-heating, bounded
response, additive heating-budget conservation, mass-conserving three-level
feed-through, default-off) and two `meridional_transport_diagnostics`
geopotential tests to `testing/test_balanced_dynamics.py` and
`testing/test_circulation_diagnostics.py`.

## 11. TOA-only radiative target: the borrowed-asymmetry hypothesis is falsified

The proposed follow-up is now implemented: `native_balanced_mse_use_toa_radiative_target`
(default `False`) swaps the MSE closure's radiative-heating term from
`_compute_T_base_ocean_full()` (AMOC/ACC bonuses, hemisphere `_sh_factor`
scaling, ocean seasonal lag) to a new `_compute_T_toa_equilibrium_full()` --
`temperature.temperature_kelvin_for_lat` evaluated directly at the current,
unlagged day, plus the same CO2 offset applied everywhere else, and nothing
else. Two unit tests cover it (`testing/test_balanced_dynamics.py`): the gate
defaults off, and toggling it changes the diagnosed overturning speed (a
functional wiring test -- at the default 2 m/s speed cap both targets
saturate to the same bound on a small toy grid, so that test raises the cap
to confirm the *uncapped* diagnosed speed actually differs, not just that the
gate is plumbed through).

A four-way 32x64, one-year spin-up/one-year evaluation screen then compared
the ocean-transport and TOA-only targets, the latter at 10/30/60-day
relaxation. **Caveat on comparability:** Section 10's exact background
configuration (the full "retained candidate" accumulated informally across
Sections 5-9: horizontal mass-flux closure, balanced pressure dynamics,
energy-limited evaporation, specific relaxation/strength constants) was never
committed as a reusable script or saved report, so it cannot be reproduced
bit-for-bit. This screen instead uses its own explicit, reproducible
configuration -- the column-water/condensate/three-level pipeline already
established in `test_balanced_dynamics.py`'s MSE-gate tests, plus
`enable_three_level_horizontal_mass_flux_closure` (its null-mode projection
removes the ~600+ PW spurious pole-to-pole throughflow that otherwise
dominates the transport diagnostic) and `enable_energy_limited_evaporation`
with `evaporation_downwelling_longwave_w_m2=25.0` (Section 3's "best complete
native experiment" constant, needed to keep the water cycle physically
bounded near 2.6 mm/day instead of 6+ mm/day). Absolute numbers below are
therefore not directly comparable row-for-row against Section 10's table; the
ocean-vs-TOA comparison is valid because all four rows share this one
configuration.

| Variant | Global precip (mm/day) | Köppen group/class | Composite ref. error | Cross-eq. total transport (PW) | ExoPlaSim annual log-corr |
|---|---|---|---|---|---|
| Ocean-transport target, 10-day relaxation | 2.61 | 0.545/0.227 | 1.112 | -25.10 | 0.181 |
| TOA-only target, 10-day relaxation | 2.62 | 0.557/0.227 | 0.988 | -22.44 | 0.173 |
| TOA-only target, 30-day relaxation | 2.62 | 0.542/0.230 | 0.849 | -30.86 | 0.199 |
| TOA-only target, 60-day relaxation | 2.62 | 0.541/0.230 | 0.712 | -18.16 | 0.181 |

Every metric is essentially flat across all four rows -- Köppen group accuracy
spans only 0.541-0.557, class accuracy 0.227-0.230, ExoPlaSim correlation
0.173-0.199 -- and cross-equatorial transport stays in the same unphysical
-18 to -31 PW band regardless of which radiative target is used or how far
its relaxation timescale is detuned. This is a materially different outcome
than Section 10's hypothesis predicted: removing the ocean-transport
asymmetry from the target does not reduce the transport blowup in any
consistent direction, and neither does weakening the heating rate 6x (10 to
60 days). **The borrowed-hemispheric-asymmetry explanation is therefore
falsified as the primary driver.** The pathology is not really about which
temperature field the radiative term chases; something else in this
configuration -- most likely the interaction between
`thermally_direct_overturning`'s fixed 0.40/0.35/0.25 mass-conserving branch
structure and the horizontal mass-flux closure's own divergence correction --
is generating the excess transport independent of the diabatic-heating
source. The next diagnostic step should isolate that interaction directly
(e.g. run the same screen with the overturning closure's heating pinned to
exactly zero, isolating whatever `enable_three_level_horizontal_mass_flux_closure`
alone contributes to cross-equatorial transport under this configuration)
rather than continuing to vary the MSE closure's heating term. Both radiative-
target gates remain default-off; Earth defaults and standalone PlanetSim
behavior are unchanged.

## 12. The mass-flux closure alone reproduces most of the transport excess

That diagnostic is now run, via `scripts/screen_mass_flux_transport_isolation.py`,
across three variants of the exact shared 32x64, one-year spin-up/one-year
evaluation configuration from Section 11 (column-water/condensate/three-level
pipeline, `enable_three_level_horizontal_mass_flux_closure`, and
`enable_energy_limited_evaporation` at 25 W/m²). The first variant reproduces
Section 11's "ocean-transport target, 10-day relaxation" row inside this
script's own harness: -25.10 PW cross-equatorial transport, 1.112 composite
reference error, and 0.545/0.227 Köppen group/class, matching Section 11's
table to three significant figures. This confirms the harness is
apples-to-apples with Section 11 before trusting the two new rows.

The second variant pins the overturning closure's heating to exactly zero --
both `enable_native_balanced_moist_static_energy_overturning` and
`enable_native_balanced_diabatic_overturning` left off, so
`thermally_direct_overturning` is never invoked at all (its call site is
gated on `_overturning_speed > 0.0`) -- while leaving the mass-flux closure
on. Cross-equatorial total transport is **-19.77 PW**, still squarely inside
Section 11's -18 to -31 PW unphysical band, and only about 5 PW less in
magnitude than the full ocean-target/10-day row (-25.10 PW). **This falsifies
the hypothesis proposed at the end of Section 11**: the excess transport is
not primarily an interaction between `thermally_direct_overturning`'s branch
structure and the mass-flux closure's divergence correction, because nearly
the same magnitude of transport survives with the overturning branches never
invoked. The mass-flux closure's own divergence correction, applied to the
resolved three-level winds, is the dominant source on its own, independent of
any diabatic-heating term.

The third variant turns the mass-flux closure off as well (heating still
pinned to zero). Cross-equatorial transport explodes to **+1184 PW** --
consistent with, and roughly double, the "~600+ PW spurious pole-to-pole
throughflow" figure Section 11 cited for its own configuration, and opposite
in sign. So the closure's null-mode projection is doing essential,
large-magnitude work (cutting the unconstrained throughflow by around 98%),
but its own bounded correction still leaves a residual an order of magnitude
too large for a believable transport budget -- and that residual is not
attributable to the diabatic-heating term at all.

| Variant | Cross-eq. total transport (PW) | Cross-eq. dry-static (PW) | Cross-eq. latent (PW) | Köppen group/class | Composite ref. error |
|---|---|---|---|---|---|
| Neither closure (mass-flux off, heating zero) | +1183.95 | +1174.81 | +9.13 | 0.479/0.257 | 0.689 |
| Mass-flux closure only, heating pinned to zero | -19.77 | -18.09 | -1.68 | 0.565/0.230 | 0.777 |
| Mass-flux closure + ocean-target overturning, 10-day (= Section 11 row 1) | -25.10 | -28.14 | +3.04 | 0.545/0.227 | 1.112 |

The next diagnostic must therefore look inside the mass-flux closure and the
raw pre-closure three-level divergence it corrects, not vary the overturning
closure any further: measure the resolved lower/middle/upper wind divergence
field before `close_upper_mass_flux` acts on it -- its zonal-mean and
non-zonal parts separately, since the closure inverts them differently
(direct Fourier-mode inversion in longitude for the non-zonal part, a small
meridional least-squares solve for the zonal mean) -- to see whether the
~20 PW residual is concentrated in one wind level, one spatial mode, or
spread evenly. That would point to a specific correction (for example a
stronger zonal-mean throughflow constraint, or a bounded correction cap that
is currently too permissive) rather than motivating another whole-closure
ablation. Results are archived at
`scripts/mass_flux_transport_isolation_result.json`. Nothing in Sections
5-11's default-off gates changed; Earth defaults and standalone PlanetSim
behavior are unchanged.

## 13. The eddy-correction speed cap, not the closure's design, is the fixable culprit

Non-invasively instrumenting `close_upper_mass_flux`'s own before/after
diagnostics (`scripts/diagnose_mass_flux_closure_throughflow.py`, via a
`simulate.close_upper_mass_flux` monkeypatch -- no production code changed)
during the Section 12 zero-heating configuration answers the "one level/mode,
or spread evenly" question directly, and the closure's two corrections turn
out to behave completely differently.

The zonal-mean null-mode throughflow correction is exact on every one of 121
calls across the run: `equatorial_throughflow_after_m_s` is ~1e-16
(floating-point zero) every time, and never comes within 1% of its 80 m/s
cap. That mechanism is not the source of the residual.

The eddy (non-zonal, Fourier-inverted) correction is a different story.
Despite the code comment's claim that the inversion "exactly removes every
resolved non-zonal divergence mode," the mean divergence-residual RMS across
the run only falls from 9.29e-5 to 8.74e-5 s-1 at the current default
`three_level_horizontal_mass_flux_max_speed_m_s=12.0` -- a 6.0% reduction.
Sweeping that cap in isolation (12/40/120/1000 m/s, same zero-heating
configuration otherwise) shows the inversion itself is not broken; it is
being clipped:

| Cap (m/s) | Residual closure | Cross-eq. transport (PW) | Köppen group/class |
|---|---|---|---|
| 12 (current default) | 6.0% | -19.77 | 0.565/0.230 |
| 40 | 20.6% | -17.09 | 0.614/0.250 |
| 120 | 53.0% | -14.10 | 0.660/0.285 |
| 1000 (near-unlimited) | 86.2% | -20.89 | **0.674/0.282** |

Two findings here, not one. First, 12 m/s is far too tight for a bound meant
to represent synoptic-scale ageostrophic/divergent upper winds (tens of m/s
is physically ordinary); raising it substantially and monotonically improves
both the residual closure and Köppen group/class skill, with the 1000 m/s
row's 0.674 group accuracy landing on the untouched compact baseline almost
exactly. Second, cross-equatorial transport is *not* monotonic in the cap: it
improves from -19.77 to -14.10 PW through 120 m/s, then worsens back to
-20.89 PW at 1000. Fully closing the local, per-level divergence residual is
not the same thing as zeroing net meridional energy transport -- a
divergence-free correction at the 3-level weighted sum can still carry a
non-trivial meridional flux wherever it is spatially correlated with the
temperature/moisture field, and an unbounded correction is free to do that
more aggressively than a moderately capped one.

This reframes the open question from "why does the closure leave a transport
residual" into a calibration problem with real structure:
`three_level_horizontal_mass_flux_max_speed_m_s` is not a physically
motivated bound at its current default, but the best-looking point in this
compact scan (120 m/s) is not itself a promotion candidate -- it is one point
in a coarse four-value scan on the zero-heating configuration, not the full
retained candidate with overturning heating restored, and per this project's
standing rule a short-run gain is not evidence on its own. The next step is a
proper bounded sweep of this cap (for example 20/40/60/80/120 m/s) against
the full CRU/Köppen/transport gate with the overturning closure back on (both
the ocean- and TOA-radiative-target variants from Section 11), followed by
the standard 128x256 five-year validation for any surviving candidate, rather
than treating 120 m/s or 1000 m/s as a default change from this diagnostic
alone. Results are archived at
`scripts/mass_flux_closure_throughflow_diagnostic.json`. No default changed;
both closures remain default-off.
