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
document's usual bounded-sweep-then-scale-up protocol. That protocol was
therefore run to completion.

A bounded sweep at 64x128 (0.0, 0.05, 0.1, 0.25, versus the default 1.0)
brackets the endpoint tightly, per the non-monotonicity already found above.
Group Koppen accuracy is essentially flat and near its best across 0.0-0.1
(0.7086, 0.7077, 0.7076) before degrading at 0.25 (0.6965) and 1.0 (0.6738);
class accuracy, kappa, and group-share MAE follow the same shape. The
six-pair orographic screen (256x512, stability 0.006 s-1) confirms this
near-zero neighborhood is comparatively well-behaved, unlike the wild swings
already documented between 0.25 and 0.75: at 0.05 Cascades is (10.3, 17.5)
m s-1 (W/L 0.98), Sierra Nevada (1.5, 10.1) (W/L 1.19), Himalaya (-3.3, -25.4)
(W/L 20.98); at 0.1, Cascades (9.7, 14.7) (W/L 1.07), Sierra Nevada (1.5, 6.6)
(W/L 1.21), Himalaya (-1.0, -27.1) (W/L 21.49). Wind direction stays correctly
signed and Himalaya stays close to its target across the whole 0.0-0.1 band,
so 0.0 -- already measured in full above -- remains the clear best endpoint
and no nearby value is worth separately carrying through the five-year gate.

The bounded sweep did, however, surface one metric this document had not yet
checked here: `reference_error_score`, the same area-weighted composite of
regional precipitation-target error and zonal temperature/precipitation bias
used to gate promotion in Sections 5-13. It moves the *opposite* direction
from Koppen skill -- 0.2117 (scale 1.0) to 0.2154 (0.25) to 0.2170 (0.1) to
0.2176 (0.05) to 0.2181 (0.0) -- monotonically *worsening* as the terrain term
is reduced, a small (about 3%) but directionally consistent regression that
the first compact check in this section did not surface because it only
inspected `region_group_accuracy` (identical across scales) and not
`regional_target_error_fraction` (which is not).

Because this was a mixed rather than a clean-sweep result, the full protocol
was still run rather than skipped, exactly matching the standing rule that a
compact screen -- however positive on Koppen skill alone -- is never
sufficient. No pre-existing tracked baseline exists at 128x256/five-plus-five
years (`docs/CURRENT_BASELINE.md` pins the tracked baseline at 64x128,
one-plus-one years), so scale 1.0 and scale 0.0 were both run at matched
128x256, five-spin-up-year/five-evaluation-year configuration for a direct,
apples-to-apples comparison (each deterministic; wall time 283s and 190s
respectively):

| Metric | scale=1.0 (default) | scale=0.0 |
|---|---|---|
| Koppen group accuracy | 0.7088 | **0.7195** |
| Koppen class accuracy | 0.4223 | **0.4258** |
| Group kappa | 0.6304 | **0.6443** |
| Group-share MAE (pp) | 2.765 | **1.742** |
| Global precipitation (mm/day) | 2.9611 | 2.9582 |
| `reference_error_score` | **0.1720** | 0.1757 |

At full resolution/duration the Koppen-skill gain not only survives, it grows
relative to the compact screen, and global precipitation stays a near-wash.
`reference_error_score` reproduces the same small, consistent regression seen
in the compact sweep (0.1720 to 0.1757, about 2.1%), and at this scale it is
traceable to specific regions rather than being diffuse: Atacama's
target-error fraction worsens from 0.432 to 0.525 (precipitation rises 71.6
to 76.2 mm/year -- a real desert getting measurably wetter, plausibly because
the terrain-locked high over the Andes was reinforcing its rain-shadow dryness
for reasons unrelated to the six orographic pairs), and US Midwest's
`region_group_accuracy` softens from 0.765 to 0.735. The zonal breakdown shows
the same pattern geographically: the 30-40N, 40-50N, and 50-60N group-accuracy
bands -- the exact midlatitude belt the six orographic pairs sit in -- are
each flat-to-slightly-worse at scale 0.0 (0.639 to 0.609, 0.587 to 0.584,
0.814 to 0.756), so the large net Koppen gain is earned entirely outside that
belt (concentrated at the equator, consistent with the compact screen).

This does not clear the full gate cleanly. Four of six tracked axes favor
scale 0.0 by a clear margin, one (global precipitation) is a wash, and one
(`reference_error_score`) regresses by a small but consistent amount at both
resolutions, concentrated in a named desert region and the same midlatitude
belt this investigation started from. Per the task's own standing rule, a
mixed long-run result is a genuine judgment call, not something to resolve by
picking the axis that looks best. `wind_terrain_pgf_scale` therefore remains
at its default of 1.0, `orographic_linear.py` remains uncoupled, and no other
default changed; this section proposes -- rather than makes -- the change, for
a human to weigh: is a large, broad Koppen-classification gain worth a small,
regionally-concentrated regression in absolute precipitation-target accuracy,
in a parameter that touches every cell of every simulation? If the answer is
yes, `wind_terrain_pgf_scale=0.0` is fully validated by this section's own
protocol and ready to flip. If the answer is no, or is "fix the regression
first," the concrete next step is to understand why Atacama and the western
midlatitude belt respond the opposite way from the equator -- most plausibly
by redesigning the term to be proportional to local relief/slope magnitude
with synoptic-scale smoothing (Andes rain-shadow forcing preserved, Cascades/
Sierra Nevada's currently-uniform-with-elevation forcing removed) rather than
a raw absolute-elevation pressure wedge applied everywhere -- before returning
to the four orographic pairs (Cascades, Sierra Nevada, S Andes, Southern Alps)
that still miss their W/L targets even with the term fully disabled.

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

Step 1 is now implemented, and it changes real physical terms rather than
only the activation-state fix from the previous session. Both reservoirs'
heat capacity is scaled by their own soil wetness
(`land_surface._moisture_scaled_heat_capacity`), using the dry-to-saturated
volumetric soil heat-capacity ratio tabulated in standard boundary-layer
climatology references (~1.3 MJ m-3 K-1 dry mineral soil to ~3.0 MJ m-3 K-1
water-saturated soil, e.g. Oke, *Boundary Layer Climates*, 2nd ed., Table
2.1) -- about a 2.3x range from dry to saturated -- centred so that the
existing calibrated `land_surface_heat_capacity_j_m2_k` /
`land_deep_heat_capacity_j_m2_k` defaults are reproduced exactly at 0.55
wetness (`simulate.py`'s own no-history land-moisture fallback), so a run
without soil-moisture history is bit-for-bit unaffected. The Penman--Monteith
dry-surface resistance term now uses a root-zone-weighted blend of the fast
shallow bucket and the slow deep/root-zone bucket (70/30 shallow/deep, per
Jackson et al. 1996's finding that most fine-root biomass across biomes
concentrates in roughly the top 30 cm with a real minority extending deeper)
rather than the shallow bucket alone. `soil_moisture_deep=None` falls back to
the shallow bucket for both new terms, so every existing caller and unit test
is unaffected; four new tests in `testing/test_prior_art_kernels.py` cover the
none-fallback identity, the deep-reservoir damping effect, and the root-zone
resistance blend directly.

Re-running the regional energy diagnostic
(`scripts/diagnose_force_restore_land.py`) after this change shows the wiring
works in the intended direction but only partially closes the gap it was
built to close. The deep-minus-surface annual-mean excess in the three flagged
regions shrinks -- Canadian Prairies 11.18 to 10.19 C, Central Europe 6.29 to
4.94 C, US Midwest 7.10 to 5.52 C -- but does not go to zero within one year of
spin-up. An isolated CRU screen with the wiring enabled and every parameter
held at its previous default (30-day restore, 12 MJ m-2 K-1 deep capacity,
2000 s m-1 dry resistance) is, within run-to-run noise, unchanged from the
pre-wiring result: temperature RMSE 7.94 C versus the earlier 7.92 C, group/
class accuracy 0.583/0.229 versus 0.587/0.235. This confirms directly, not
just by inference, that the reservoir-physics fix is a real but insufficient
prerequisite -- exactly as step 1 predicted for the *initialization* half of
this same diagnosis, now shown to also hold for the *resistance/heat-capacity*
half.

Step 2's sweep is also now run, over literature-bracketed points rather than
the previous session's narrower ad hoc pair. Deep heat capacity is bracketed
using the same ~1.3-3.0 MJ m-3 K-1 volumetric range applied to a ~4 m
representative deep-reservoir depth (matching the total column depth of
existing land-surface schemes, e.g. CLM's ~3.8 m default column or ECMWF
HTESSEL's ~2.89 m), giving a ~5.2-12.0 MJ m-2 K-1 physical bracket; the
existing 12 MJ m-2 K-1 default sits at its wet/deep end, so {5, 8, 12} MJ m-2
K-1 was swept. Dry resistance is bracketed against bare-soil/stressed-canopy
resistance literature (FAO-56's 70 s m-1 well-watered reference, rising into
the low thousands of s m-1 under water stress in bare-soil-evaporation
parameterizations such as van de Griend & Owe 1994), so {1000, 2000, 3000}
s m-1 was swept. Restore time reused the previously established {15, 30} day
pair, since the prior four-point screen had already shown it to be the
flattest of the three axes. All 2x3x3 = 18 candidates were run at 64x128,
one year spin-up plus one year evaluation, against the same CRU baseline
(`scripts/sweep_force_restore_land.py`, archived at
`testing/reference_data/force_restore_screen_v2_64x128.json`):

| Metric | Baseline | Sweep range (18 candidates) | Best-group candidate |
|---|---|---|---|
| Temperature RMSE (C) | 6.279 | 7.833 - 7.963 | 7.944 |
| Köppen group accuracy | 0.674 | 0.570 - 0.589 | 0.589 |
| Köppen class accuracy | 0.388 | 0.224 - 0.236 | 0.236 |
| Precipitation log-RMSE | 1.406 | 1.370 - 1.372 | 1.371 |

Zero of the 18 candidates are accepted. Every one sits in essentially the same
rejected band the pre-wiring, narrower sweep already found (temperature RMSE
7.86-7.91 C; group skill 0.581-0.583) -- widening the parameter range to its
full physically defensible bracket, on top of the reservoir-physics fix from
step 1, does not move the result meaningfully in any direction. The
best-group-accuracy point (8 MJ m-2 K-1 deep capacity, 15-day restore, 3000
s m-1 dry resistance) still trails the baseline by 0.085 group and 0.152 class
accuracy while its temperature RMSE is 1.67 C worse. This is decisive enough,
across a properly bracketed physical range, to treat as a rejection of these
three parameters as the binding defect, not as grounds for a still-wider or
finer-grained re-sweep.

The land-seasonal-cycle diagnostic (`_land_seasonal_cycle_metrics`,
`_koppen_temperature_thresholds`) clarifies what is actually wrong. The
force-restore candidate's shape score is *better* than the legacy path's --
`cycle_error_score` falls from 5.269 to 1.355, and squareness lands at
6.00-6.08 months in every band versus the legacy path's 6.75-7.12 (a pure
sinusoid scores 6.00; `EARTH_LAND_SQUARENESS` is 6.2) -- so the replacement
closure's local physics genuinely produces a more physically realistic annual
cycle *shape* than the heavily hand-tuned legacy blend. But its amplitude and
level are wrong relative to where CRU-anchored calibration put the legacy
path: seasonal amplitude is 30-70% larger in every band (e.g. 45-55N: 38.5 K
versus 30.2 K), and the anchor-free Köppen threshold accuracies collapse
(coldest-month 0.900 to 0.743, warmest-month 0.677 to 0.369). The legacy path
carries several terms with no equivalent in the force-restore closure --
`land_seasonal_amplitude`, `_atm_land_transport_1d`, `_midlat_storm_bonus_1d`,
`_handoff_bonus_1d`, and `_land_cap_1d` -- that exist specifically to import
eddy/storm-track atmospheric heat transport and damp the summer plateau; the
force-restore path integrates only local radiative, turbulent, and
conductive terms. That is the more likely source of the still-large RMSE and
Köppen regression than any of the three parameters just swept.

Per this project's standing rule, a candidate does not proceed to the
128x256, five-year benchmark on a short-run result, and doubly does not when
the short-run result is a decisive, range-spanning rejection rather than a
promising-but-unproven one (contrast e.g. Section 5's compact candidates,
which improved on the baseline before earning their long run). None of the 18
step-2 candidates meet that bar, so no candidate was taken to the 128x256
benchmark this session. For the record, the current codebase's 128x256,
five-year spin-up plus five-year evaluation baseline (force-restore off) was
regenerated to confirm it is unchanged by every change in this session:
temperature RMSE 5.727 C, Köppen group/class accuracy 0.709/0.422 --
bit-for-bit identical to the previously archived
`testing/reference_data/baseline_128x256_5y_cru.json`, despite the intervening
`PlanetParams` field additions -- so it remains available unmodified for any
future force-restore candidate that does clear a compact screen.
`enable_force_restore_land` therefore stays off; nothing in `EARTH`'s defaults
changed this session.

The next consequential step is accordingly not another local-parameter sweep:
it is giving the force-restore path an explicit atmospheric heat-transport
term (or an equivalent seasonal-amplitude calibration), since local
radiative-turbulent-conductive physics alone -- now confirmed correctly wired
to both soil reservoirs and swept across its full literature-defensible
range -- tops out roughly 1.6 C and 0.09-0.10 Köppen group accuracy short of
the legacy path's heavily calibrated result. Only after that structural gap
is addressed does a further local-parameter sweep become diagnostic again.

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

## 14. Restoring overturning heating decouples the cap from both transport and composite score

Section 13's cap scan intentionally used zero overturning heating to isolate
the closure's own behavior. `scripts/sweep_mass_flux_cap_with_overturning.py`
re-runs the same five cap values (20/40/60/80/120 m/s) with
`enable_native_balanced_moist_static_energy_overturning` restored, at its
10-day relaxation, crossed with both the ocean-transport and TOA-only
radiative targets from Section 11 -- ten 32x64 one-year-spinup/one-year-eval
runs in total:

| Cap (m/s) | Ocean: Köppen group/class | Ocean: composite ref. error | Ocean: transport (PW) | TOA: Köppen group/class | TOA: composite ref. error | TOA: transport (PW) |
|---|---|---|---|---|---|---|
| 20 | 0.548/0.218 | 1.362 | -18.62 | 0.566/0.246 | 0.818 | -20.80 |
| 40 | 0.611/0.231 | 0.847 | -23.65 | 0.606/0.241 | 0.984 | -13.98 |
| 60 | 0.640/0.266 | 0.914 | -15.93 | 0.623/0.252 | **0.745** | -22.57 |
| 80 | 0.632/0.257 | 1.050 | -15.05 | 0.649/0.275 | 1.071 | -20.24 |
| 120 | 0.658/0.284 | 1.883 | -24.21 | **0.659/0.268** | 0.936 | -15.34 |

Two things carry over from Section 13's zero-heating scan, and one does not.
Köppen group accuracy still rises with the cap for both radiative targets
(TOA rises on every single step, 0.566 to 0.659; ocean rises on every step
except a small dip at 80), so the "12 m/s is too tight a bound" conclusion is
confirmed with heating present, not an artifact of the zero-heating setup.

What does *not* carry over is any usable relationship for the other two
metrics. Cross-equatorial transport stays in the same -14 to -24 PW
unphysical band regardless of cap or target, exactly as before, but here it
is not even the same non-monotonic-with-a-visible-minimum shape Section 13
showed (best at 120, worst at 1000) -- it is closer to noise, with the best
and worst values both appearing at different caps for the two targets (worst
ocean value at 120, worst TOA value at 60). The composite reference-error
score is a third, independent story again: it does not track the cap at all
(worst ocean value is also at 120 -- 1.883, nearly double every other row in
the table -- while the single best composite score in this entire
experimental family across Sections 11-14, 0.745, appears at TOA/60, a
middling point for both Köppen skill and transport). No cap value is a joint
winner across group accuracy, composite error, and transport at once; this
is a genuine multi-objective trade-off between three metrics that move
independently of one shared underlying variable, not a single dial. Nothing
in this table clears the untouched compact baseline on all axes
simultaneously, so nothing here is a promotion candidate.

This sharpens, rather than resolves, the open question from Section 13.
There, closing the *closure's own* local divergence residual (measured
directly via its `residual_before_s`/`residual_after_s` diagnostic) tracked
the cap cleanly and monotonically (6% to 86% closure). Whether that same
clean relationship survives once overturning heating is present, or whether
heating's own contribution to the pre-closure divergence field changes what
the correction is actually closing, has not yet been measured directly in
this run -- Section 13's instrumentation was zero-heating only. The next
diagnostic step is to rerun that same `simulate.close_upper_mass_flux`
recording wrapper (not a new mechanism, the one already built for Section 13)
across a couple of cap values on one of this section's heated configurations,
to check directly whether `residual_after_s` still closes cleanly with cap
when heating is present. If it does, that would confirm the disconnect
between "the closure works as intended" and "transport/composite score don't
visibly improve" is real and structural -- i.e. locally closing divergence is
provably not sufficient to control net meridional transport, and the next
target for investigation becomes the resolved wind field feeding into the
closure (the lower/middle/upper winds themselves, and specifically whatever
`thermally_direct_overturning` and the balanced-pressure wind target
contribute to them before the closure ever runs) rather than the closure's
own correction. Results are archived at
`scripts/mass_flux_cap_overturning_sweep_result.json`. No default changed;
all four gates involved (`enable_three_level_horizontal_mass_flux_closure`,
`enable_native_balanced_moist_static_energy_overturning`,
`native_balanced_mse_use_toa_radiative_target`, and the cap itself) remain at
their existing defaults.

## 15. The mass closure and the transport pathology are provably different problems

The direct check Section 14 called for is now done:
`scripts/diagnose_mass_flux_closure_with_overturning.py` reruns the same
`simulate.close_upper_mass_flux` recording wrapper from Section 13, this time
on a heated (TOA target, 10-day relaxation) configuration, at cap values
20/60/120 m/s.

| Cap (m/s) | Residual before (s-1) | Residual closure | Transport (PW) |
|---|---|---|---|
| 20 | 9.37e-5 | 10.4% | -20.80 |
| 60 | 9.48e-5 | 30.0% | -22.57 |
| 120 | 9.36e-5 | 52.4% | -15.34 |

Two confirmations. First, `residual_before_s` is essentially unchanged by
heating (9.36-9.48e-5 s-1 here versus 9.29-9.44e-5 s-1 in Section 13's
zero-heating scan) -- the overturning branches' contribution to the
pre-closure divergence field is small compared to whatever already dominates
it. Second, at the one cap value tested in both scans, closure fraction is
essentially identical whether heating is present or not: 52.4% here at 120
m/s versus 53.0% in Section 13's zero-heating run. (The transport and
Köppen/precip numbers at each cap also reproduce Section 14's TOA row
exactly, e.g. -20.80 PW at 20 m/s in both, which is a useful sanity check on
top of the substantive result.) The closure's own local divergence-closing
behavior is therefore provably independent of whether any diabatic heating
is driving the circulation -- it is governed entirely by the cap and the
pre-existing divergence field, not by the heating source this whole
experimental family (Sections 8-14) has been varying.

This is worth stating precisely, because it changes what the "residual
transport" finding actually means. A closure that drives
`equatorial_throughflow_after_m_s` to exactly zero is enforcing zero *net
column mass flux* at the equator -- and a real Hadley circulation is
*supposed* to have exactly that property while still carrying a large net
*energy* transport: warm air rises and moves poleward aloft, sinks, and
returns equatorward near the surface, with the three levels' mass fluxes
cancelling by construction while their energy-per-unit-mass differs because
temperature and geopotential differ by level. So driving the local,
per-level mass-divergence residual toward zero was never mechanically
guaranteed to shrink meridional *energy* transport, and Section 14's finding
that the two move independently under the cap is the expected consequence of
that, not a mysterious decoupling. The real anomaly is not "the closure
doesn't reduce transport" -- it is that this experimental family's transport
values (Section 11: -18 to -31 PW; Section 12-15: -14 to -25 PW) sit roughly
3-5x above Earth's actual peak meridional Hadley-cell energy transport
(observationally on the order of 5-6 PW), and do not scale in any consistent
direction with either the heating source (Section 11) or the closure's own
cap (Sections 13-15).

That reframes the next diagnostic away from "tune the mass-flux closure
further" (this family of experiments has now been tested from three
different angles -- heating source in Section 11, heating magnitude in
Section 12, and the closure's own cap in Sections 13-15 -- without finding a
lever that reliably shrinks transport) and toward the resolved wind fields
themselves. The next step is to measure the raw lower/middle/upper
meridional wind magnitude and pattern directly from a heated run's final
`PlanetState` (`state.wind_v`, `state.midlevel_wind_v`, `state.wind_v_aloft`
-- no closure or transport-diagnostic code needed, just read the state
returned by `run_real_terrain_validation`) and compare it against realistic
Hadley-cell wind speeds by level, to check whether the branch speeds coming
out of `thermally_direct_overturning` and/or the balanced-pressure wind
target are simply too large for this configuration -- a much more direct
target than continuing to vary the mass-flux closure that Sections 13-15 have
now shown is not the mechanism. No default changed.

## 16. The excess wind is the shared, already-calibrated jet-stream kernel, not an experimental artifact

`scripts/diagnose_resolved_wind_magnitude.py` now runs two variants: the full
Section 15 heated configuration, and a `kernel_only` variant with none of
Sections 8-15's overturning/mass-flux-closure gates enabled at all (just the
column-water/condensate/two-layer/three-level-pressure-column prerequisites).

| Variant | Lower-v mean/max/RMS (m/s) | Mid-v mean/max/RMS (m/s) | Upper-v mean/max/RMS (m/s) | Cross-eq. transport (PW) |
|---|---|---|---|---|
| `kernel_only` | 0.57 / 0.96 / 0.63 | 6.90 / 13.66 / 32.07 | 24.14 / 45.71 / 29.45 | +1162.52 |
| `full_heated` (Section 15 config) | 1.02 / 1.54 / 1.10 | 10.96 / 20.35 / 31.01 | 15.84 / 37.53 / 21.48 | -22.44 |

The excess is not only present without any of Sections 8-15's additions, it
is *larger* for the upper level (24.14 vs 15.84 m/s mean, 45.71 vs 37.53 m/s
max) and comparable for the middle level. `kernel_only`'s +1162.52 PW also
reproduces the same order-of-magnitude spurious pole-to-pole throughflow
Section 12 found whenever the mass-flux closure is disabled (there +1183.95
PW), this time with zero contribution from any diabatic-heating source --
confirming again that the closure's null-mode projection, not the heating
term, is what keeps the full configuration's transport in the -14 to -31 PW
band rather than four figures.

Tracing where `state.wind_v_aloft` (this diagnostic's "upper") actually comes
from in `simulate.py` produces the important qualifier: `u2_full`/`v2_full`
are first assigned at line 2384 by `_evolve_wind_substepped`, which calls
`evolve_wind_aloft` (`atmosphere.py`) **unconditionally, on every step,
regardless of any three-level or experimental gate** -- this is the same
"1.5-layer atmosphere" aloft wind that has been the project's default,
always-on jet-stream/baroclinic-jet model since the jet-stream feature work
(see `docs/PRIOR_ART_IMPLEMENTATION_PLAN.md`-adjacent project history: pgf_amp
was raised from 8 to 40 and subsequently to its current default of 90.0
across that earlier, extensively validated jet-latitude calibration effort).
The three-level/overturning/closure machinery in Sections 8-15 only *adds to*
this same base array afterward (the balanced-pressure blend, the thermal-wind
relaxation, `thermally_direct_overturning`'s upper branch, and the mass-flux
closure's correction, in that order) -- it never replaces it. The
experimental midlevel wind reuses the identical `evolve_wind_aloft` function
at a reduced amplitude (`three_level_mid_wind_pgf_fraction=0.55`, giving an
effective PGF amplitude of 49.5 against the upper level's 90.0, with its own
stronger damping of 0.08 against the upper level's 0.05).

This reframes the finding once more, and turns it into a genuine fork rather
than a next sweep. `wind_upper_pgf_amp` and `wind_upper_damping` are not
uncalibrated constants -- they are the same parameters behind the project's
already-validated, default-on jet-stream latitude/speed behavior, tuned
against that target in earlier, separate work. Everything measured in this
section is that kernel's *meridional* (v) component, which was never itself
a target of that calibration (the jet-stream work optimized zonal jet
latitude and speed, i.e. u) and is not exercised by any default-on
diagnostic -- the transport/Köppen impact only becomes visible once the
three-level experimental path and its `circulation_scorecard` transport
diagnostic are enabled. Turning `wind_upper_pgf_amp`/`wind_upper_damping` down
to fix the experimental path's meridional-wind magnitude therefore risks
regressing the already-validated default jet-stream behavior, since both
paths share the exact same function and constants.

This is a genuine design fork, not a numeric sweep, and per this project's
practice of flagging real judgment calls rather than guessing, it is left
open here rather than picked: (a) decouple the meridional PGF/damping term
from the zonal one with a new parameter specific to the experimental
three-level path (real new surface area, needs its own design and, ideally,
a re-validation that it does not change the default jet-stream's zonal
behavior at all), or (b) accept the shared kernel's meridional component as a
known limitation and constrain the transport diagnostic/overturning
calibration to be robust to it, rather than re-deriving the shared kernel.
Sections 8-15's individual findings about the mass-flux closure's and
overturning heating's own behavior remain valid characterizations of those
mechanisms; what changes is the conclusion that any of them could reach a
physically small transport residual on their own -- none of them were ever
operating on a realistically scaled base wind field. No default changed;
results are archived at `scripts/resolved_wind_magnitude_diagnostic.json`.

## 17. Decoupling the upper wind: a real transport improvement, still short of promotable

Section 16 left a genuine fork rather than picking a side. This session picks
option (a): the three-level path now evolves its own, genuinely independent
upper-level wind, `PlanetState.upperlevel_wind_u/v`, instead of continuing to
build on `state.wind_u_aloft`/`wind_v_aloft` -- the shared, always-on
"1.5-layer atmosphere" jet-stream kernel that `wind_upper_pgf_amp`/
`wind_upper_damping` are separately calibrated against.

**What changed.** A new substep wrapper, `_evolve_upper_wind_substepped`
(`simulate.py`), calls the identical `evolve_wind_aloft` physics used by both
the shared kernel and the existing independent middle level, but with its own
new `PlanetParams` fields: `three_level_upper_wind_pgf_fraction` (a
multiplier on `wind_upper_pgf_amp`, default `1.0` -- reproducing the shared
kernel's full amplitude as the starting point, since the three-level
additions never previously scaled the raw forcing, only blended/added onto
its result) and `three_level_upper_wind_damping` (default `0.08`, matching
the independent middle level's own precedent exactly, versus the shared
level's `0.05`). All four three-level-only additions that previously wrote
directly to `u2_full`/`v2_full` -- the balanced-pressure blend
(`enable_native_balanced_pressure_dynamics`), the thermal-wind relaxation
(`three_level_balanced_thermal_wind_relaxation`), `thermally_direct_
overturning`'s upper branch, and `close_upper_mass_flux`'s correction -- are
redirected onto the new state, along with the "upper" wind fed into
`generate_precipitation`'s pressure-column vapor transport and mid-upper
interface omega (the field is seeded as a copy of `u2_full`/`v2_full` on
first use, then evolves independently every step after). The middle level's
own reference/relaxation target (`0.5 * (u_full + upper)`) now averages
against the new independent state instead of the shared kernel, since it was
always meant to represent "the three-level column's own upper level," not
the jet-stream kernel specifically.

**Regression coverage.** Five new tests in `testing/test_prior_art_kernels.py`
cover state persistence (present when the gate is active, absent otherwise),
that the new substep wrapper produces genuinely different output than the
shared field (proving it is not an accidental alias), and -- the load-bearing
test -- that `wind_u_aloft`/`wind_v_aloft` are `np.testing.assert_array_equal`
bit-identical whether every one of the four redirected additions is off/zero
or aggressively on, both with the three-level gate on and with it off, with a
positive control confirming those same knobs do visibly perturb the
*independent* state (so the bit-identical result is not vacuous). Two
pre-existing tests (`test_balanced_dynamics.py`,
`test_pressure_circulation.py`) asserted the old coupled behavior directly
("the gate changes `wind_u_aloft`") and were updated to assert the new,
intentional decoupling instead. The full `pytest testing/ -q` suite passes
except the one pre-existing, unrelated failure noted below.

**Measurement 1: damping alone is not the fix.** Holding
`three_level_upper_wind_pgf_fraction=1.0` and sweeping only
`three_level_upper_wind_damping` (0.05/0.08/0.15/0.25/0.40) on the full
Section-11-style heated 32x64 one-year spin-up/one-year evaluation
configuration (`scripts/screen_decoupled_upper_wind.py`, both the
ocean-transport and TOA-only radiative targets from Section 11), transport
stays noisy in the same unphysical band the whole family has shown since
Section 11, and does not move in a consistent direction with damping:

| Damping | Ocean: Köppen group/class | Ocean: composite ref. error | Ocean: transport (PW) | TOA: Köppen group/class | TOA: composite ref. error | TOA: transport (PW) |
|---|---|---|---|---|---|---|
| 0.05 | 0.562/0.241 | 0.860 | -17.17 | 0.582/0.237 | 0.693 | -15.17 |
| 0.08 (prior default) | 0.568/0.226 | 0.873 | -27.86 | 0.572/0.219 | 0.960 | -23.46 |
| 0.15 | 0.568/0.232 | 0.694 | -19.97 | 0.570/0.238 | 1.299 | -20.73 |
| 0.25 | 0.583/0.229 | 0.870 | -16.46 | 0.576/0.234 | 0.665 | -22.85 |
| 0.40 | 0.563/0.231 | 0.788 | -16.33 | 0.570/0.243 | 0.658 | -15.71 |

The raw meridional-wind magnitude (`scripts/diagnose_resolved_wind_magnitude.py`,
extended to report both `state.wind_v_aloft` and the new
`state.upperlevel_wind_v`) tells the same story: at the TOA-target
configuration, mean/max/RMS zonal-mean tropical `|v|` is 12.71/33.49/31.06 m/s
at damping 0.05, *worsens* to 18.18/43.22/34.79 m/s at the 0.08 default, then
16.32/37.15/33.22 at 0.15 and 17.77/36.93/33.58 at 0.25 -- non-monotonic and
never within an order of magnitude of the ~1-3 m/s literature Hadley-cell
target. Per `evolve_wind_aloft`'s own mechanism (Euler PGF forcing, then an
exact Coriolis rotation, then weak Rayleigh damping each substep), this is
not surprising in hindsight: damping removes momentum only *after* each
substep's rotation has already redistributed the forcing, so it fights the
same rotation-driven dynamics that make the response noisy rather than
addressing the forcing magnitude directly.

**Measurement 2: PGF fraction is the more direct, and more effective,
lever.** Holding damping fixed at 0.08 (the middle-level precedent) and
sweeping `three_level_upper_wind_pgf_fraction` instead
(0.1/0.25/0.4/0.55/0.7/1.0, 0.55 mirroring the middle level's own fraction)
on the identical configuration:

| PGF fraction | Ocean: Köppen group/class | Ocean: composite ref. error | Ocean: transport (PW) | TOA: Köppen group/class | TOA: composite ref. error | TOA: transport (PW) |
|---|---|---|---|---|---|---|
| 0.1 | **0.627/0.268** | 0.933 | **-14.65** | **0.639/0.284** | 1.111 | **-9.36** |
| 0.25 | 0.598/0.258 | 1.106 | -11.67 | 0.613/0.252 | 0.988 | -16.66 |
| 0.4 | 0.586/0.229 | 0.947 | -21.77 | 0.594/0.235 | 0.853 | -18.69 |
| 0.55 | 0.582/0.245 | 0.562 | -28.77 | 0.584/0.237 | 0.704 | -9.23 |
| 0.7 | 0.572/0.230 | 1.076 | -11.88 | 0.587/0.235 | 0.745 | -23.15 |
| 1.0 (= damping-sweep 0.08 row) | 0.568/0.226 | 0.873 | -27.86 | 0.572/0.219 | 0.960 | -23.46 |

`0.1` is not monotonic with the rest of the sweep (0.25-0.7 scatter worse on
transport than either 0.1 or the 1.0 baseline), but it is consistently the
best point across *both* independently-run radiative targets simultaneously
-- Köppen group accuracy (0.627 ocean / 0.639 TOA) and transport magnitude
(-14.65 / **-9.36** PW) both improve together at 0.1, which is a real signal
given the rest of this experimental family's history of metrics moving
independently of each other and of any single dial (Sections 13-15). The TOA
row's -9.36 PW is, to date, the closest any configuration in this entire
Sections-8-17 experimental family has come to Earth's real ~5-6 PW peak
Hadley-cell transport magnitude -- previous families ranged -14 to -31 PW.
The raw wind-magnitude diagnostic corroborates it directly: at
`pgf_fraction=0.1`, mean/max/RMS tropical `|v|` is 12.21/**14.94**/12.79 m/s
-- the *max* in particular drops to barely a third of the 1.0-fraction row's
43.22 m/s, even though the zonal-mean value (12.21 m/s) is still 4-12x the
literature target. `pgf_fraction=0.55` reaches a similarly good TOA transport
(-9.23 PW) but with a much larger max (29.00 m/s) and no group-accuracy gain,
so 0.1 is the cleaner of the two candidate points.

The shared kernel's own upper wind is confirmed unaffected by any of this, as
designed: `state.wind_v_aloft`'s mean/max across the whole pgf_fraction sweep
stays at 23.99-24.18 / 45.36-45.60 m/s regardless of which value the
*independent* state's fraction takes (kernel_only: 24.04/45.60; fraction=0.1:
23.998/45.36; fraction=0.55: 24.10/45.44) -- the small residual variation
between rows is ordinary run-to-run trajectory noise from the differently
evolving decoupled state feeding back into shared fields like temperature,
not a leak in the decoupling itself.

| Variant | Upper-v (decoupled) mean/max/RMS (m/s) | Upper-v (shared kernel) mean/max/RMS (m/s) | Cross-eq. transport (PW) |
|---|---|---|---|
| `kernel_only` (no closures, fraction=1.0, damping=0.08) | 11.83 / 25.67 / 35.49 | 24.04 / 45.60 / 29.25 | +840.17 |
| `full_heated`, fraction=1.0, damping=0.08 (prior default) | 18.18 / 43.22 / 34.79 | 24.18 / 45.56 / 29.78 | -23.46 |
| `full_heated`, fraction=1.0, damping=0.05 | 12.71 / 33.49 / 31.06 | -- | -15.17 |
| `full_heated`, fraction=0.1, damping=0.08 | **12.21 / 14.94 / 12.79** | 24.00 / 45.36 / 29.45 | **-9.36** |
| `full_heated`, fraction=0.55, damping=0.08 | 12.13 / 29.00 / 27.51 | 24.10 / 45.44 / 29.73 | -9.23 |

**Confirmed.** (1) The decoupling itself is complete and regression-tested:
the shared jet-stream kernel is bit-identical regardless of any three-level
setting, both with the gate on and off. (2) Damping alone, at any tested
value, does not move transport in a useful or consistent direction -- this
negative result closes that branch of the design space. (3) PGF fraction is
a materially more effective lever, and `pgf_fraction=0.1`/`damping=0.08` is a
genuine, reproducible improvement over the prior coupled behavior on this
compact screen: transport improves from the -14 to -31 PW band that has
defined this whole family since Section 11 to -9.36 PW (TOA target), and
Köppen group accuracy improves alongside it (0.639 vs 0.572-0.674 range seen
elsewhere in this family) rather than trading off against it, at both
radiative targets independently.

**Not confirmed / next step.** This is one compact 32x64, one-year
spin-up/one-year evaluation screen at two points in a six-value sweep that is
not itself monotonic -- per this project's standing rule, it is not a
promotion candidate on this evidence alone, and no default has changed
(`three_level_upper_wind_pgf_fraction` stays at `1.0`, all three three-level
gates stay off). The next step is a proper bounded sweep around 0.1
(for example 0.05/0.1/0.15/0.2) crossed with a finer damping grid near 0.08,
to check whether the fraction=0.1 point is a real local optimum or the edge
of a coarse six-value grid, followed by the standard 128x256 five-year
spin-up/five-year evaluation CRU benchmark for whatever survives that finer
screen. If that clears the joint temperature/Köppen/transport gate, it would
be the first candidate in the Sections-8-17 family to seriously approach
Earth's real cross-equatorial transport magnitude while also improving
compact Köppen skill, rather than trading one for the other. Results are
archived at `scripts/resolved_wind_magnitude_diagnostic.json` and
`scripts/decoupled_upper_wind_screen_result.json`; the sweep scripts
(`scripts/diagnose_resolved_wind_magnitude.py`,
`scripts/screen_decoupled_upper_wind.py`) are reusable for the finer follow-up
sweep.

## 18. The finer joint sweep: transport is noise at this protocol, group accuracy is not

Section 17's own next step, run: `scripts/screen_decoupled_upper_wind_fine.py`
crosses `three_level_upper_wind_pgf_fraction` (0.05/0.1/0.15/0.2, bracketing
the 0.1 candidate) with `three_level_upper_wind_damping` (0.04/0.06/0.08/0.1/
0.12, a finer grid around the 0.08 precedent) jointly, at both radiative
targets, on the identical 32x64 one-year spin-up/one-year evaluation
protocol -- 40 runs total. The single `frac=0.1`/`damp=0.08`/TOA point was
first re-verified in isolation and reproduced Section 17's exact -9.36 PW
bit-for-bit, confirming the harness itself is deterministic and correct.

**Result: cross-equatorial transport does not survive being crossed.** Within
each fixed fraction, transport ranges over 5-11 PW of within-row spread as
damping alone varies (e.g. ocean/frac=0.05: -1.09 to -26.48 PW across just
five damping values), and neither axis is monotonic once the other is held
fixed -- full per-row means/stdevs:

| Fraction | Ocean transport mean/stdev (PW) | TOA transport mean/stdev (PW) |
|---|---|---|
| 0.05 | -14.74 / 10.35 | -17.47 / 2.53 |
| 0.1 | -15.02 / 4.37 | -16.85 / 5.88 |
| 0.15 | -20.89 / 5.92 | -13.22 / 6.88 |
| 0.2 | -12.45 / 5.43 | -18.51 / 6.04 |

The grid's overall spread (mean -15.8 PW ocean / -16.5 PW TOA, pstdev 7.6 /
5.9 PW across all 20 points per target) is roughly as wide as the entire
-14 to -31 PW band this metric has occupied since Section 11, and the
individual-row stdevs are comparable in size to the differences *between*
row means. Section 17's -9.36 PW at `frac=0.1`/`damp=0.08` was a real,
reproducible output of that exact configuration, but the surrounding grid
gives no evidence it sits in a real minimum rather than being one favorable
draw among many similarly-sized swings -- the same class of error this
project's testing-methodology lesson (`known-physics-gaps.md`) already warns
about, here appearing as parameter-space noise rather than time-axis noise:
a single 32x64/1yr+1yr point is not resolved enough to certify a location in
this two-parameter space, only to sample it.

**What is not noise: Köppen group accuracy tracks `pgf_fraction` cleanly and
monotonically, independent of damping**, at both targets:

| Fraction | Ocean group accuracy (mean over damping) | TOA group accuracy (mean over damping) |
|---|---|---|
| 0.05 | 0.6478 | 0.6495 |
| 0.1 | 0.6345 | 0.6320 |
| 0.15 | 0.6225 | 0.6263 |
| 0.2 | 0.6124 | 0.6118 |

This extends Section 17's coarse-sweep finding (accuracy improving as
fraction fell from 1.0 toward 0.1) below 0.1 as well -- 0.05 is the best
point in this finer grid on this axis, at both targets, with damping barely
perturbing it (row stdevs on group accuracy, not tabulated above, stayed
under 0.01 throughout, versus transport's 2.5-10 PW row stdevs). Group
accuracy is evidently governed almost entirely by the forcing magnitude
(`pgf_fraction`) and not by how it is damped, which is a clean, physically
sensible result independent of the transport metric's noise.

**How to apply.** Per this project's standing rule that a short-run gain is
not evidence, `frac=0.1`/`damp=0.08` is *not* a promotion candidate on this
evidence -- the finer sweep just called for by Section 17 found that its
apparent transport win does not generalize to neighboring points in the same
grid it was drawn from. No default changed (`three_level_upper_wind_pgf_
fraction` stays `1.0`, all three three-level gates stay off). Two real
options going forward, not picked here: (a) treat cross-equatorial transport
as unreliable at this compact protocol and stop trying to select a
`pgf_fraction`/damping point by it -- select by group accuracy alone (which
*is* stable here, and argues for something at or below 0.05, lower than
anything tested in Section 17) and treat transport as a diagnostic to
re-check only at the full 128x256 five-year benchmark, where longer
averaging may suppress the noise this compact protocol cannot; or (b) before
trusting any transport number at this protocol again, first characterize the
metric's own run-to-run noise floor directly -- e.g. reruns of the identical
`frac=0.1`/`damp=0.08` configuration with only the terrain-independent RNG
seed (if any) or a trivially perturbed initial condition varied, to learn
whether 5-10 PW of swing is intrinsic to a 32x64/1yr+1yr evaluation window
regardless of which parameter moved. Results archived at
`scripts/decoupled_upper_wind_fine_screen_result.json`; sweep script is
`scripts/screen_decoupled_upper_wind_fine.py`.
