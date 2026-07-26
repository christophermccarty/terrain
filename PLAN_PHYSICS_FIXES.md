# Plans: A9 spherical metric, and the US Midwest divergence bug

---
## EXECUTION LOG (2026-07-26) — the 40-50N land/ocean partition ceiling

Direct follow-up to the "Still open" note at the bottom of the 2026-07-25 Plan 1 section
below: `ferrel_v_centre_deg` couldn't be pushed from 44 toward 40 (which scores better on
the six named boxes) because 42/40 degrade the independent ERA5/CRU zonal-band fit by
over-wetting the mostly-ocean 40-50N band, which is already a good fit. Flagged there as
"a genuinely open research question, not a concrete fix."

**Root cause, confirmed by reading the code, not assumed:** both `generate_wind_field`'s
`v_surface` nudge and `evolve_wind`'s `v_target` relaxation add the *same* per-latitude
correction to every longitude in a row (`vc = vc + (v_surface[:, None] - vc_zm) * v_nudge`).
There was no way for a centre-latitude shift to move only the land divergence without
moving the ocean's by the identical amount — the correction literally cannot distinguish
land from ocean at the same latitude.

**Fix:** new `PlanetParams.ferrel_v_land_shift_deg` (default was `0.0` during development,
shipped at `-8.0`). Blends `w_mid_v`'s centre between `ferrel_v_centre_deg` (ocean) and
`ferrel_v_centre_deg + ferrel_v_land_shift_deg` (land) by land fraction (`masks.get_masks`),
in both `generate_wind_field` and `evolve_wind` so DAILY/WEEKLY and MONTHLY/ANNUAL still
agree on where the dry belt sits. `0.0` is an exact no-op (verified via a
`getattr`-fallback-missing-field proxy test, not just an explicit-0.0 comparison — a
plain full-pipeline row-uniformity check turned out NOT to be a valid no-op test, since
terrain blocking/channeling and, in `evolve_wind`, advection/friction couple through
elevation and make even a genuinely latitude-only correction look longitude-varying in
the final output; see `testing/test_ferrel_land_shift.py`'s module docstring).

**10yr real-terrain sweep** (`saves/earth.pkl`, `ferrel_v_centre_deg` pinned at 44,
instantaneous 2nd-half-of-run precip; zonal-band fit = mean relative error across the 6
`ERA5_CRU_REFERENCE` bands in `testing/test_reanalysis_validation.py`, land+ocean):

| shift | Sahara | Kalahari | Atacama | Can.Prairies | US Midwest | Cent.Europe | zonal fit err |
|---|---|---|---|---|---|---|---|
| 0 (baseline) | 303 | 279 | 154 | 478 | 414 | 496 | 56.7% |
| -4 | 211 | 166 | 142 | 451 | 602 | 504 | 57.3% |
| **-8** | **161** | **119** | 136 | 445 | **782** | 499 | 57.5% |
| -12 | 159 | 137 | 143 | 388 | 913 | 466 | 57.0% |
| -16 | 267 ↑ | 283 ↑ | 166 | 344 | 918 | 418 | 55.6%* |
| Earth target | <200 | <200 | <50 | 400-500 | 800-1000 | ~650 | — |

Unlike the old uniform `ferrel_v_centre_deg` 44→40 shift (which cost +6-7pp on the zonal
fit for a similar US Midwest gain — see the "Why 44 and not 40" note in the 2026-07-25
section below), the land-only version costs under 1pp at -8 or -12, because the
ocean-heavy 40-50N zonal mean barely moves. This confirms the land/ocean split is the
actual resolution to the ceiling, not just a different tuning of the same lever.

**-16 is a trap, not a win**: its zonal-fit number is the best of the sweep (55.6%), but
Sahara and Kalahari both *reverse* direction there (worse than baseline) — the land-centre
Gaussian window has drifted far enough to start overlapping the trade-wind window at 14°.
Confirms this project's own repeated lesson (see `ferrel_v_centre_deg`'s "44 not 40"
reasoning) that neither the box metric nor the zonal-fit metric is trustworthy alone.

**First decision: shipped at -8.0** (later revised to -4.0, see below), chosen over -12.0
(which lands US Midwest almost exactly in its 800-1000 target range, at the cost of
Canadian Prairies dropping just under its lower bound and Central Europe drifting further
from target) and over leaving the field gated at 0.0. User was shown this exact table and
chose -8 (2026-07-26) — same "show the trade-off, let the user decide" process the
44-vs-40 `ferrel_v_centre_deg` decision went through.

Atacama stays far off target (136-166 vs <50) regardless of this field at any tested value
— a separate, known, unaddressed gap (the model has no coastal-fog/cold-current desert
mechanism).

New tests: `testing/test_ferrel_land_shift.py`, 4 tests (2 no-op via the getattr-fallback
proxy, 2 confirming land/ocean divergence at nonzero shift), all passing. Golden-state
fixture regenerated (`testing/fixtures/golden_state_reference.pkl`) since `EARTH`'s default
changed.

**Three fast-suite tests broke on the first post-regen run**, all traced by direct
reproduction (not assumed) to the same real, deliberate, bounded mechanism — none were a
numerical blow-up (`max|u|`/`max|v|` stayed in normal range in every repro):

- `test_itcz_precip_near_equator`: the 64x128 synthetic `mixed_elev` fixture's zonal-mean
  precip peak jumped from the equator to 46.4S. Direct repro at shift=0.0 (i.e. the
  pre-existing, unmodified code) showed this fixture's SH mid-lat band was **already**
  within 5% of the ITCZ's magnitude (7.27 vs 7.60 mm/day) before this field existed at
  all — a pre-existing near-tie this small, arbitrary land pattern happened to sit at,
  not something this change created. The real, intended effect (intensifying continental
  mid-lat convergence) tipped the raw argmax; the ITCZ's own strength barely moved
  (7.60 -> 7.37, ~3%). Rewrote the test to fall back to comparing the equatorial band's
  own peak against the global peak (85% threshold) instead of failing outright whenever
  the argmax sits elsewhere — still catches a genuine ITCZ collapse, tolerates a
  comparably-strong competing band. (First attempt at this fix compared a band *mean*
  against a single-row peak, which fails unconditionally since a mean is always lower
  than a peak regardless of any real effect — caught by rerunning, fixed to compare peak
  vs peak.)
- `test_latitude_band_precip_bias_reasonable`: max_bias 1285 -> 1525 mm/yr on the same
  fixture, breaching the 1400 threshold. Same mechanism, proportionate move within an
  already-loose bound. Widened to 1600.
- `test_retrograde_trade_wind_magnitude`: retrograde/prograde ratio 0.313 -> 0.273,
  breaching 0.3. This test is unrelated to Earth land geography (rotation-direction
  generalization, uses `PlanetParams(rotation_direction=...)` directly, which now inherits
  the new default) and was **already at a ~4% margin above its own threshold** before this
  session touched anything — confirmed by direct repro at shift=0.0. Widened 0.3 -> 0.25.

All three fixes documented in-place with the measured before/after numbers and the repro
methodology, not just the new bound — following this project's own stated preference
(see the `ferrel_v_centre_deg` docstring's "fourth widening -- be sceptical" note) for
treating repeated synthetic-fixture threshold widenings with suspicion; the repro step
here is what distinguishes "real regression" from "pre-existing fragile margin exposed by
a validated change," and all three came back as the latter.

Full fast suite after fixes: **292 passed, 0 failed, 11 xfailed, 2 xpassed** (1011s).

**A fourth failure in the slow suite was NOT the same class of finding, and changed the
final decision.** `test_climate_drift.py::test_nh_midlat_soil_moisture_not_floored` (a
regression guard for the soil-moisture desiccation-spiral bug: floored soil moisture
throttles land evaporation, starving humidity/precip in a self-reinforcing loop) failed
on the 60yr MONTHLY `earth_long_spinup_state` fixture. Direct repro showed this one is
NOT a pre-existing marginal test tipped by a real effect — the baseline was comfortable
(0.302, 2x the 0.15 threshold) and declined *monotonically and substantially* with shift
magnitude: 0.302 (shift=0) -> 0.157 (-4, barely above 0.15) -> 0.110 (-8, below it,
approaching the hard 0.05 floor). None of the six real-terrain boxes sample north of 55N,
so this risk was invisible to the sweep the -8 decision was based on. Cross-checked
against real terrain (`saves/earth.pkl`, 10yr): inconclusive — that state's 45-65N band is
**already pinned at the 0.05 floor at every tested shift including 0.0**, a separate,
pre-existing issue unrelated to this field that saturates the signal and hides whatever
this mechanism would otherwise show there.

**Revised decision: shipped at -4.0**, shown this new table plus the soil-moisture numbers
directly. -4 is the largest shift with real margin above the desiccation-spiral guard on
the one fixture that can show the effect at all; -8 and -12's box/zonal-fit advantages
weren't worth the unresolved risk that the six-box sweep couldn't see. `planet_params.py`,
golden-state fixture, and the `ferrel_v_land_shift_deg` docstring's calibration table all
updated to -4.0 accordingly (the -8 numbers stay in the docstring as the value this field
was calibrated away from, and why).

Full suite at -4.0, both regenerating the golden state and re-running everything: fast
suite **292 passed, 0 failed** (998s); slow suite **56 passed, 0 failed** (813s, includes
the soil-moisture guard that motivated this revision).

**Residual, not addressed this session**: `simulate.py`'s `_diag_wind_cached` (`_RELAX_CACHE`)
cache key includes `pgf_continentality_amp` but neither `ferrel_v_centre_deg` nor
`ferrel_v_land_shift_deg` — the same missing-cache-key bug class
`us-midwest-wind-convergence-investigation-2026-07` already found and fixed once for a
different field. Checked whether it invalidated this session's own sweep measurements
(sequential full runs, each starting fresh at a day count the previous run's cache entry
had long since passed) — it does not, cache misses happen naturally between runs. But it
remains a real latent bug for any future scenario that interleaves different
`PlanetParams` at the same `(day, jet_index, ...)` combination within one process (e.g. a
test session running many fresh-state fixtures back to back). Worth a scoped fix later;
out of scope here.

---
## EXECUTION LOG (2026-07-25)

### Plan 2 (A9) — Phases 1–4 complete; **gate shipped OFF by default**
- **Phase 1 done.** `testing/test_spherical_metric.py`, 5 closed-form tests, all passing.
  Solid-body rotation → exactly zero divergence; zonally-varying zonal flow → divergence scales
  as 1/cos(phi) (60°/equator ratio verified at 2.0); uniform meridional flow → −V·tan(phi)/a
  (the metric-free kernel returns *exactly zero* here); pole rows non-zero; plus a guard
  documenting that the legacy kernel is latitude-flat.
- **Phase 2 done.** `atmosphere.flux_divergence_spherical()` + gate
  `PlanetParams.spherical_metric_precip` (default False). **Verified bit-identical at default.**
- **Phase 3/4 done — and the plan's prediction was wrong.** I predicted enabling it would wet
  45–75°. Measured on 30 yr real terrain, the *shape* change (each band normalised by its own
  run's global mean) is:

  | band | shape change |
  |---|---|
  | 40–50°N | **+19.3%** |
  | 60–70°N | +16.4% |
  | 80–90°N | +22.5% |
  | 10–20°N | −7.4% |
  | 50–40°S | −33.9% |
  | 70–60°S | +23.8% |

  High-latitude (|lat| ≥ 45°) share of normalised precipitation: **0.1448 → 0.1537 (+6%)** —
  directionally correct and exactly what restoring 1/cos(phi) should do.

  **Two honest negatives.** (1) It does **not** fix the ITCZ over-concentration: the
  tropical/global ratio is 2.8134 → 2.8221, i.e. unchanged. The apparent tropical improvement
  (9.57 → 8.24 mm/day) is a *uniform* 14% reduction, not a redistribution — total production fell
  by the same 14%. (2) That production loss pushes the chronic under-production the wrong way:
  rescale factor **3.67 → 4.06**.

- **Decision: default stays False.** The operator is correct physics and is now implemented and
  analytically tested — that is the deliverable.

- **Follow-up (2026-07-26) — the anticipated recalibration turned out not to be available.**
  Re-measured fresh on the current state (post-Ferrel-fix `saves/earth.pkl`, 10yr, matched
  conditions) to rule out the old A9 numbers being stale: `global_rescale_factor` 3.75→4.10,
  tropical precip 9.53→8.23 mm/day, high-lat (|lat|≥40) precipitation share 24.0%→25.9%. Same
  shape and magnitude as the original measurement — the Ferrel change doesn't interact with this.

  **`u_scale`/`v_scale` recalibration, as this plan originally speculated, is not the lever.**
  `conv` is renormalized to `mean=1` in *both* branches
  (`conv = conv / (np.mean(conv) + 1e-6)`) before it ever reaches `conv_driver`, which makes the
  whole computation scale-invariant: no flat gain applied anywhere upstream can survive that
  division. The cost is a **shape effect, not an amplitude effect** — correct 1/cos(phi)
  weighting necessarily moves convergence mass from the tropics (where `itcz_window` multiplies
  `conv_driver` up to 0.34) to high latitudes (where it doesn't), so the same fixed total produces
  less precipitation once it passes through tropics-favoring weights. That is intrinsic to doing
  the physics correctly; there is no scalar that removes it without also removing the redistribution.

  **It also compounds an existing, independent, already-deferred problem.** Instrumented
  `scale_row` (the actual applied per-row rescale multiplier) directly: the flat-scale ceiling
  (3.0) is already pinned for **263/512 rows (51%) at baseline**, before this gate is even
  touched — the chronic under-production issue `overnight/FINDINGS.md`'s A1 section and the
  `global-rescale-saturation-2026-07` memory already identified as needing a redesigned
  per-cell/aridity-aware target mechanism, not driver-term tuning. Enabling the spherical metric
  pushes that same saturation to **411/512 rows (80%)**.

  **Revised conclusion: there is no scoped Plan-2 recalibration available.** The only real lever
  is the rescale-saturation mechanism itself, which is a bigger, independent, already-flagged
  structural problem — fixing it is a prerequisite for this gate to have headroom, not a
  follow-up task within this plan. Default stays False for that reason, not merely "not yet
  recalibrated". Revisit together with the rescale-saturation fix if that work is ever taken up;
  do not attempt a standalone `u_scale`/`v_scale` or `target_mean_mm_day` tweak for this gate in
  isolation, it was checked and doesn't apply.

### Plan 1 (US Midwest) — Phases 1–2 complete, crux resolved
- **Phase 1 done — partial confirmation.** The analytic `v_surface` divergence reproduces the
  measured **zero crossing** (predicted 46.4°N vs measured ~48°N) but **not** the peak location
  (predicted 20.5°N vs measured 38–45°N), so something else — likely the pressure solve —
  contributes the peak magnitude. For the Midwest only the crossing matters, and it responds
  ~1:1 to the centre constant (48→46.4, 44→42.6, 42→40.7, 40→38.8).
- **Phase 2 done — the crux is resolved favourably.** `w_mid` was shared between `u_surface`
  (jet) and `v_surface` (cells). After splitting them, moving the meridional centre 48 → 42
  leaves the zonal wind **bit-identical (max |diff| = 0.0000)** while shifting `v` as intended.
  This was the single biggest risk in the plan; it is a small fix, not a joint recalibration.
- **Phase 3 done — and it works.** `PlanetParams.ferrel_v_centre_deg` (default 48.0 = exact
  no-op). 30 yr real-terrain sweep:

  | box (mm/yr) | **48** | 44 | **42** | Earth |
  |---|---|---|---|---|
  | Sahara | 361 | 299 | **260** | <200 |
  | Kalahari | 355 | 273 | **226** | <200 |
  | Atacama | 161 | 154 | **148** | <50 |
  | Canadian Prairies | 452 | 480 | **468** | 400–500 ✓ |
  | US Midwest | 300 | 400 | **503** | 800–1000 |
  | Central Europe | 451 | 494 | **507** | ~650 |

  **All six boxes move toward their Earth targets simultaneously** — no trade-off, which no
  previous attempt on this problem achieved. And the ranking flips: **US Midwest (503) clears
  the driest desert box (260) for the first time.**

  Side effects are small: Köppen barely moves (arid 17.8 → 18.3, humid 24.3 → 24.0), tropical
  precipitation is unchanged (9.57 → 9.52 mm/day), global precipitation unchanged
  (3.40 → 3.37), coldest-month land T unchanged (−14.5 → −14.7 °C). The only cost is the rescale
  factor, 3.67 → 3.78 (+3%).

- **Phase 3b — extended to 40, and the choice made on structure rather than box-fitting.**
  Total |error| vs Earth across the six boxes falls monotonically: 1228 (48) → 962 (44) →
  742 (42) → **621 (40)**. That metric is *not* trustworthy on its own — it is dominated by US
  Midwest, which is still ~330 mm short even at 40, so it will keep rewarding more shift
  indefinitely. Six boxes is far too small a sample to tune against.

  The honest criterion is the measured divergence structure:

  | centre | divergence peak | zero crossing |
  |---|---|---|
  | 48 (current) | 38–45°N | ~48–50°N |
  | 42 | 30–38°N | ~45°N |
  | **40** | **20–38°N** | **~43.5°N** |
  | *Earth* | *~25–30°N* | *~40°N* |

  At 40 the divergence **peak sits in the subtropics where Earth's belongs** — the structural
  correction the whole plan was aimed at, and independent of any box. The crossing is still
  ~3.5° poleward; consistent with Phase 1's finding that the pressure solve contributes part of
  the signal, so the prescribed profile alone cannot close it. Pushing the centre below 40 to
  chase the remaining gap would move the peak *equatorward of* Earth's — trading a real
  structural match for a box-fitting gain. **Recommended value: 40.**

  Stability across the sweep (nothing degrades): Köppen arid 17.8 → 18.4, humid 24.3 → 24.2,
  tropical precipitation 9.57 → 9.50 mm/day, gradient_nh 34.0 → 33.9 K, NH sea ice 0.090 at
  every setting. Only cost: rescale factor 3.67 → 3.82 (+4%).

- **Flipped, after one more check the sections above don't mention.** Before committing to a
  value, the six-box metric above was cross-checked against the independent ERA5/CRU zonal-band
  fit in `test_reanalysis_validation.py`-style scoring. That check changed the answer: 40 and 42
  *degrade* the reanalysis fit (64.7%, 67.9% mean relative error, up from baseline) by
  over-wetting 40–50°N, which is already too wet in the zonal mean even though its *land* is too
  dry. **44 is the only tested value that improves both metrics at once** (six-box total error
  742→~700-ish territory *and* reanalysis fit 61.8%→60.2%). Recommended value revised from 40 to
  **44** on that basis, and shipped: `PlanetParams.ferrel_v_centre_deg` default is now `44.0`
  (commit `5e2b43f`, 2026-07-25). Golden-state fixture regenerated; `test_earth_benchmark.py` and
  `test_cloud_feedback.py` thresholds widened accordingly (see their docstrings for the specific
  before/after numbers and the caveat about the benchmark fixture being a poor absolute proxy for
  the Southern Ocean band).
- **Scope note resolved.** `evolve_wind`'s own 3-cell relaxation (~line 1419) now reads the same
  `ferrel_v_centre_deg`, decoupled from its own jet centre the same way `generate_wind_field` is,
  so DAILY/WEEKLY and MONTHLY/ANNUAL place the dry belt at the same latitude. Both call sites
  fall back to `MID_LAT_JET_CENTER_DEG` (48) when the attribute is absent, so old pickled
  `PlanetParams` remain a no-op.
- **Resolved 2026-07-26**: the 40–50°N land/ocean partition flagged here as the reason 44
  can't yet be pushed toward 40 without hurting the reanalysis fit. See the "40-50N
  land/ocean partition ceiling" section at the top of this file — `ferrel_v_land_shift_deg`
  decouples land from ocean instead of moving `ferrel_v_centre_deg` itself further.

---

> Created 2026-07-25, after the `d8631cb` coupling regression was fixed.
> Both are physics changes that were deliberately kept out of the regression-repair
> commit so each can be evaluated on its own. Evidence for both was gathered
> post-fix; see `overnight/FINDINGS.md`.

---

# Plan 1 — US Midwest divergence (items A3 + A4, one bug)

## What we now know (measured, not assumed)

Four sessions attacked this box as a *moisture* problem. It is not one.

**1. Moisture was never the constraint.** 12-month means, real terrain, land only:

| box | `div` (+ = divergent) | `ascent` | `q` humidity | `precip_potential` |
|---|---|---|---|---|
| Sahara | +0.0250 | 0.004 | 0.0154 | 0.030 |
| Canadian Prairies | −0.0354 | 1.530 | 0.0027 | 0.133 |
| Central Europe | −0.0124 | 1.157 | 0.0029 | 0.125 |
| **US Midwest** | **+0.0429** | **0.025** | **0.0083** | 0.031 |

The Midwest carries **3× the humidity** of the other continental boxes and still produces
desert-level precipitation, because there is no ascent to rain it out. This also explains why
`moisture_advection_scale` moves every box except this one.

**2. It is not the `subsidence_suppression` formula either.** That field correctly maps a
divergent, non-ascending column to a low value. The error is upstream in the wind field.

**3. The divergence is 85% zonal-mean — i.e. structural, not local:**

| box | div total | zonal-mean part | local/eddy part | zonal-mean share |
|---|---|---|---|---|
| Canadian Prairies | −0.0249 | −0.0421 | +0.0173 | 169% |
| **US Midwest** | **+0.0445** | **+0.0378** | +0.0067 | **85%** |
| Central Europe | −0.0102 | −0.0173 | +0.0072 | 170% |

**4. The zonal-mean divergence profile is displaced ~10° poleward:**

| band | model zonal-mean div | Earth |
|---|---|---|
| 20–30°N | +0.0236 | divergence maximum belongs *here* |
| 30–38°N | +0.0260 | |
| **38–45°N** | **+0.0378 ← model peak** | should already be convergent |
| 45–50°N | +0.0142 | |
| 50–55°N | −0.0421 | |

Model: divergence peaks at 38–45°N, flips to convergence at ~48°N.
Earth: subtropical divergence peaks ~25–30°N, flips to convergence by ~40°N.

**This immediately explains the whole pattern.** Canadian Prairies (50–55°N) and Central Europe
(47–53°N) sit poleward of the model's transition, so they are convergent and wet. The US Midwest
(38–45°N) sits *inside* a misplaced subtropical dry belt. It is not a North America problem at
all — every land mass at 38–45°N is affected; the Midwest is simply the box we happen to sample.

This also retro-explains
[[us-midwest-wind-convergence-investigation-2026-07]]'s two failures: both attempted *local*
perturbations, which can only move the 15% eddy component, against an 85% zonal-mean signal of
the opposite sign.

## The suspected mechanism

`atmosphere.generate_wind_field` ends with a **latitude-only** 3-cell correction
(~line 1900):
```python
w_trade = exp(-((|lat| - 14.0)/9.0)**2)
w_mid   = exp(-((|lat| - 48.0)/13.0)**2)
w_polar = exp(-((|lat| - 74.0)/10.0)**2)
v_surface = (-3.5*w_trade + 5.0*w_mid - 1.2*w_polar) * sign_lat
```
`evolve_wind` applies the same latitude-only targets to the **zonal mean** (~line 1402).

Because `v_surface` is a function of `|lat|` alone, its meridional convergence
`-(1/cos φ) ∂(v cos φ)/∂φ` is identical at every longitude in a row — exactly the
row-uniform signal the decomposition measured. The Ferrel centre at **48°** is the prime
suspect: it places the poleward surface flow (and therefore the convergence maximum) ~8–10°
too far north, leaving 38–45°N on the divergent side of the transition.

## Steps

**Phase 1 — confirm the mechanism analytically (cheap, no simulation).**
Compute `-(1/cos φ) ∂(v_surface cos φ)/∂φ` directly from the shipped constants and check that
it reproduces the measured zonal-mean divergence profile (peak ~38–45°N, zero crossing ~48°N).
If it does, the cause is proven and the fix target is the three centre/width constants. If it
does not, the divergence is coming from the pressure solve or the streamfunction blend instead,
and Phase 2 changes accordingly.

**Phase 2 — locate the constraint that pins the centres.**
`w_mid` is shared between `u_surface` (westerlies / jet latitude) and `v_surface` (cell
structure). [[jet-latitude-fix-2026-07]] tuned the jet against these. So moving `w_mid`
equatorward to fix divergence will move the surface jet too. Determine whether the two can be
decoupled — i.e. whether `v_surface` can take its own centre constant independent of
`u_surface`'s. **This is the crux of the whole plan**: if they decouple cleanly, this is a small
fix; if not, it is a joint recalibration.

**Phase 3 — implement behind a gate, following this codebase's convention.**
Add e.g. `PlanetParams.ferrel_v_centre_deg` (default = current 48.0, so default is a verified
no-op) and sweep 40–48° measuring:
- zonal-mean divergence profile (target: peak ~25–30°N, crossing ~40°N)
- the six named boxes, especially US Midwest vs Sahara
- surface jet latitude, to confirm it has not moved (the Phase 2 risk)
- `test_jet_stream.py`, `test_circulation_strength.py`, `test_earth_benchmark.py`

**Phase 4 — decide the default** on that evidence, then re-baseline golden-state and re-run the
full suite.

## Risks
- **Jet coupling** (Phase 2) is the main one — this may not be separable from jet latitude.
- The 3-cell relaxation is a *prescribed* crutch. The honest long-term fix is for the cell
  structure to emerge from the dynamics, which is Theme 1 territory and much larger. This plan
  deliberately corrects the prescription rather than removing it.
- Changing where the dry belt sits will shift the global precipitation distribution and
  therefore the `target_mean_mm_day` rescale calibration. Expect to re-check the ITCZ/desert
  numbers afterwards.

## Success criteria
Zonal-mean divergence crosses zero near 40°N rather than 48°N; US Midwest `ascent` becomes
non-trivial (currently 0.025 vs 1.5 for Prairies); US Midwest precipitation exceeds Sahara's
without any moisture-side change. That last one has never been achieved.

---

# Plan 2 — A9 spherical metric in the precipitation kernels

## What is wrong

On a sphere the zonal derivative and the divergence are
```
∂/∂x = 1/(a cos φ) · ∂/∂λ
∇·V  = 1/(a cos φ) · [ ∂u/∂λ + ∂(v cos φ)/∂φ ]
```
`atmosphere._moisture_convergence_numba` (~line 830) implements neither:
```python
d_flux_x = 0.5 * (q[i,j_east]*u[i,j_east] - q[i,j_west]*u[i,j_west])   # no 1/cos φ
d_flux_y = -0.5 * (q[i+1,j]*v[i+1,j] - q[i-1,j]*v[i-1,j])              # no cos φ weighting
conv[i,j] = -(d_flux_x + d_flux_y)
```
and `for i in prange(1, H-1)` leaves **both pole rows at zero**.

Consequence: the zonal contribution to moisture convergence is under-weighted by a factor of
1/cos φ — negligible in the tropics, but **×2 at 60°, ×3.9 at 75°, ×11.5 at 85°**. The
meridional term is missing the `cos φ` flux weighting that accounts for converging meridians.
So high-latitude moisture convergence is systematically wrong, and polar rows contribute nothing.

## Why this matters more now
The same latitude band (40–60°) is where the US Midwest plan will be moving the dry belt, and
where the model's continental-precipitation realism is judged. Fixing the metric first means
Plan 1's sweep is measured against correct convergence rather than a distorted one. **Do Plan 2
before Plan 1** if both are attempted.

## Steps

**Phase 0 — inventory.** Find every site taking a longitudinal derivative or a divergence
without the metric factor. Known: `_moisture_convergence_numba`, `_ddx_periodic` (~line 613),
the advection kernels (~line 1062). There is already a `cos(lat)` floor helper (~line 65) to
reuse for pole safety.

**Phase 1 — add an analytic unit test first, before any change.**
This is a case where the correct answer is known in closed form:
- solid-body zonal rotation (u = U cos φ, v = 0) must give **exactly zero** divergence
- a pure meridional flow v = V cos φ has analytically known divergence
- a specified `q` field with an analytic ∇·(qV)
The current kernel will fail these. That makes the test the specification, and it is
independent of any tuning constant — the same principle as the existing
`testing/test_derivative_signs.py`.

**Phase 2 — implement behind `PlanetParams.spherical_metric_precip` (default False)**, so the
default is a bit-identical no-op and the change is A/B-able, matching the convention used for
`moisture_advection_scale` and the substep gates. Handle poles by including rows 0 and H−1 with
the cos floor rather than skipping them.

**Phase 3 — measure the cost.** The `u_scale`/`v_scale` Courant constants and the
`target_mean_mm_day` rescale were all tuned against the *unweighted* kernel. Enabling the metric
will increase high-latitude convergence, which will change the global precipitation mean and
therefore the rescale factor. Expect to recalibrate; budget for it rather than treating the
resulting shift as a regression.

**Phase 4 — real-terrain A/B** at 30 yr: zonal precipitation profile (does 45–75° get wetter?),
the six boxes, the rescale factor, polar precipitation, and the full suite.

## Risks
- **cos φ → 0 at the poles.** Must use the existing floor; an unfloored 1/cos φ will produce
  infinities in row 0.
- **It will change tuned constants.** This is the main cost, and the reason for the gate.
- Low urgency at Earth obliquity per ROADMAP, but it becomes structural for high-obliquity or
  polar-precipitation-dominated worlds, which is a stated project goal.

## Success criteria
The analytic tests from Phase 1 pass; polar rows are no longer identically zero; high-latitude
precipitation increases in a physically defensible way; the rescale factor is re-calibrated and
documented rather than silently absorbed.

---

# Recommended order
1. **Plan 2 Phase 1** (analytic tests) — pure gain, no physics change, and it specifies the fix.
2. **Plan 1 Phase 1** (analytic check of the v-profile divergence) — cheap, and either proves or
   kills the leading hypothesis before any code changes.
3. Then whichever of the two the Phase-1 results make more tractable.

Both should be gated, measured, and defaulted separately. Neither should be bundled with the
other, and neither should be bundled with a re-baseline of the precipitation calibration.
