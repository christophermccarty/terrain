# Simulation Accuracy Audit — Master List

> Compiled 2026-08-01. Reference standard: `Koppen_classification_world_map_1991-2020_-3C_borderless.png`
> (repo root), the user's designated ground truth for biome/climate comparison — see the
> `koppen-reference-map` memory note for the color key.
>
> This document consolidates ~40 sessions of physics investigation (memory notes,
> `known-physics-gaps.md`'s living list, `ROADMAP.md`, `FEATURES.md`, `PLAN_PHYSICS_FIXES.md`,
> `docs/FINDINGS_SUMMARY.md`) into one master audit: every element of the simulation currently
> known to diverge from real-world conditions, what controls it, how to test it, what's already
> been tried, and what's missing outright. It supersedes no other document — `known-physics-gaps.md`
> and `PLAN_PHYSICS_FIXES.md` remain the detailed session-by-session logs; this is the rolled-up
> index. **Re-verify against current code before trusting any specific number below** — this
> project's own history shows numbers drift session to session; grep the cited file/param first.

**Severity legend**: 🔴 open, actively wrong · 🟡 partially fixed, real residual gap · 🟢 investigated
and correctly left as-is (not a bug) · ⬛ missing system entirely, no code exists

**Baseline numbers cited below** are from `testing/fixtures/real_terrain_validation_baseline.json`
(64×128 compact fixture, MONTHLY, 1yr spinup + 1yr eval) unless noted "real-terrain" (512×1024
Earth DEM, longer runs, less reproducible but higher-fidelity). Targets are from
`regional_validation.py`'s `EARTH_PRECIP_REGIONS`.

---

## A. Precipitation spatial pattern (the dominant, most-worked cluster)

### A1. 🟡 Deserts still too wet — Sahara/Kalahari now fixed, Atacama residual only
**Symptom (largely resolved 2026-08-01, see A5)**: Sahara/Kalahari/Atacama used to all render
wetter than Köppen's BWh/BWk cores. Atacama in particular has never cleanly separated from other
deserts on any metric tried, and still hasn't.
**Current numbers, independently re-verified 2026-08-01** (`saves/earth.pkl`, 512×1024, 2yr MONTHLY,
seasonally-balanced 2nd-half average — see A5's sampling-window caveat): Sahara **131 mm/yr**
(target <200 — **now in range**), Kalahari **130 mm/yr** (target <200 — **now in range**), Atacama
**102 mm/yr** (target <50 — still ~2x over, the sole residual). This is the A5 regime-architecture
fix's direct effect, verified by reproducing its own claimed numbers independently rather than
trusting the write-up alone (this project's own established practice — see process note 3).
**Controlling variables**: `atmosphere.py`'s `subsidence_suppression` (wind-divergence-derived
aridity proxy — gates both `land_evap` and `precip_potential`), `drybelt_window`/`drybelt_regime_window`
(the latter a smooth 16–34° core with shoulders, added 2026-08-01 — see A5), `PlanetParams.
coastal_upwelling_fog_strength` (0.5, diagnostic west-coast gate for Atacama specifically),
`desert_redistribution_weight`/`cell_weight`, `subsidence_divergence_regime_gate` (now **1.0**
default, zonal-background/local-anomaly divergence decomposition — see A5/B1), `_div_pos_norm_ref`
cap (0.02).
**How to test**: `scripts/check_real_terrain_koppen.py --wind-diagnostics --days 730` (use 730+, not
the 365 default — see A5's sampling-window caveat) for named-box precip; `scripts/
run_real_terrain_validation.py --compare` for the tracked composite score.
**What's been tried** (all documented, don't repeat blind): flat desert-suppression multiplier
(reverted — fought the zonal-rescale calibration, too blunt spatially); moisture-transport
strengthening (reverted — monotonically wets deserts *more*, not less, at any transport strength);
mean-preserving `subsidence_suppression`-based redistribution (shipped, `k=0.9`, real improvement
but plateaued until the A5 fix); coastal-fog gate for Atacama specifically (shipped at strength 0.5,
Atacama 123→102 mm/yr in isolation, real but small); the A5 regime-architecture fix (shipped
2026-08-01 — closed Sahara/Kalahari, did not move Atacama, whose own aridity signature — a narrow
coastal strip rather than a broad subsiding interior — was never well-served by any
latitude/divergence-based mechanism tried so far, per B3's own note on the same tension).
**Recommended next lever for the Atacama residual — sharpened 2026-08-02 by D5**: Atacama's own
diagnostics (see A5's real-terrain wind-diagnostics table) show `div=1.70–1.79`, an outlier an order
of magnitude above every other desert box — it's already about as suppressed as that mechanism can
make it. This entry previously named "real SST-coupled upwelling physics" as the honest fix; **that
is now known to be insufficient on its own.** D5's gyre calibration produced exactly that upwelling
signal (Humboldt −0.25 K SST off the Atacama coast) and Atacama's precipitation did not move at all,
at any gyre strength from 0.0 to 3.0. The blocker is not the absence of cold water — it's that
**per-cell SST anomalies don't propagate into land precipitation in this model's atmosphere**. Any
future Atacama attempt should target that coupling pathway first and verify it transmits *before*
building more ocean physics, or it will reproduce D5's null result.

### A2. 🟢 Tropical rainforest over-extent / savanna under-extent (Af vs Aw/Am) — resolved; the headline gap was largely a measurement bug
> **⚠️ READ THIS BEFORE TRUSTING ANY KÖPPEN PERCENTAGE ELSEWHERE IN THIS DOCUMENT.**
> Every Köppen land-share figure recorded here before 2026-08-02 — in A1, A2, A4, A5, B3, and the
> memory notes they cite — was computed as a **plain cell count on an equirectangular grid**, with no
> cos(latitude) area weighting. That systematically over-weights polar cells (a cell at 85° covers
> ~8.7% the area of an equatorial one) and under-weights tropical ones. Corrected, the model is
> **~4× more accurate** than those numbers implied (mean absolute error vs Earth's real group shares:
> **8.7pp → 2.2pp**), with no physics change whatsoever:
>
> | group | cell-count (old) | **area-weighted** | Earth |
> |---|---|---|---|
> | A tropical | 12.7% | **22.0%** | 19.0% |
> | B arid | 18.3% | **28.9%** | 26.4% |
> | C temperate | 9.0% | 12.3% | 13.4% |
> | D continental | 21.7% | 21.1% | 24.6% |
> | E polar | 38.3% | **15.6%** | 16.6% |
>
> Because Köppen shares are a closed budget, the +21.7pp of phantom polar land was being subtracted
> from every other group — which is precisely how the tropics appeared "2–4× under-represented" for
> multiple sessions. **Fixed 2026-08-02** in both tracked measurement sites
> (`real_terrain_validation._koppen_land_percentages` and
> `scripts/check_real_terrain_koppen.py::_koppen_breakdown`).

**Resolution**: the tropical band was never too small — area-weighted, A totals **22.0%** against
Earth's 19.0% (slightly *over*, not 2–4× under). What was genuinely wrong was the *split within* it,
and that is now calibrated: `PlanetParams.itcz_seasonal_target_response` **1.0 → 1.7 → 2.0**.

| | Af | Am | Aw | MAE vs Earth |
|---|---|---|---|---|
| k=1.0 (start of 2026-08-01) | 19.27% | 0.24% | 3.57% | — |
| k=1.7 (calibrated on the biased metric) | 10.70% | 4.23% | 7.10% | 2.61pp |
| **k=2.0 (shipped)** | **6.82%** | 6.22% | **8.94%** | **1.37pp** |
| Earth | 6.0% | 4.0% | 10.0% | — |

**The bias directly caused a mis-calibration, which is worth remembering**: the unweighted metric
reported k=2.0 as "overshooting Af down to 3.95%, below Earth's 6–7%", which is why 1.7 was chosen
earlier the same day. Area-weighted, k=2.0 puts Af at 6.82% — essentially on target. Deserts and
continental boxes are flat across the range (Sahara 193→192, US Midwest 980→982); `reference_error_score`
rises 0.3195→0.3217, the same zonal-magnitude-vs-biome-accuracy trade as before, but now with a
trustworthy biome metric on the other side of it.

**Also ruled out along the way** (so it isn't re-investigated): the hypothesis that the A band was
capped by Köppen's coldest-month ≥18°C temperature gate. It is not — pushing the coldest month up by
a hypothetical +5°C raises *potential* A from 13.1% to only 14.5% (cell-count basis), because the
newly-warm land is almost entirely arid. Actual A was already capturing ~97% of available warm
non-arid land. The gate is not the constraint.

<details>
<summary>Superseded 2026-08-01 writeup (kept for the diagnosis trail — its numbers are cell-count-biased)</summary>

**STATUS UPDATE 2026-08-01 (later same day)**: the regression documented below was **fixed**, and Af
is now *inside* Earth's range for the first time in the project's history. Fix shipped:
`PlanetParams.itcz_seasonal_target_response` **1.0 → 1.7**.

**Why this value was available now but not before** (the interesting part — this knob had already
been swept to 2.0 on 2026-07-31 and deliberately capped at 1.0): that earlier sweep rejected k>1.0
because `arid_pct` climbed *while the named desert boxes drifted wetter*, i.e. the extra aridity was
being manufactured at the tropical margin instead of in real deserts. Post-A5 that failure mode is
simply absent — across a fresh 1.0→2.0 sweep the desert boxes are flat-to-slightly-**drier**
(Sahara 195→193, Kalahari 157→156, Atacama 146) and every continental box is flat, while `arid_pct`
moves 17.67%→18.23%, i.e. *toward* Earth's ~19–20% rather than past it. A5 had also lifted
`arid_pct`'s starting point from ~12% to ~17.7%, which is exactly why the same signal read as
"overshoot" then and reads as "convergence" now. **Re-testing a previously-rejected parameter after
a major upstream change was the whole win here** — the parameter did not change, the constraint did.

**Measured (deterministic 64×128 benchmark, fresh spinup — the tracked instrument, not the
EMA-lagged save)**:

| k | Af% (→6–7) | Aw+Am% (→18–20) | arid% (→19–20) | Sahara | Midwest | refErr |
|---|---|---|---|---|---|---|
| 1.0 (was) | 11.05 | 2.25 | 17.67 | 195 | 979 | 0.3185 |
| **1.7 (now)** | **6.07** | **6.65** | **18.23** | 193 | 981 | 0.3226 |
| 2.0 | 3.95 | 8.71 | 18.29 | 192 | 981 | 0.3245 |

Confirmed on real terrain (512×1024, 2yr MONTHLY, seasonally balanced): named boxes unchanged —
Sahara 131→126, Midwest 709→705, Central Europe 502→502, S Japan 1138→1138, all within noise.
**Real, accepted cost**: `reference_error_score` 0.3185→0.3226 (+1.3%), traceable entirely to the
10–20°N zonal band's precip ratio (0.768→0.714) — that band *is* the savanna belt, and deepening its
dry season genuinely lowers its annual mean. (The modulation also stops being exactly mean-preserving
above k≈1.0: its `clip(0.05)` floor truncates the dry-season trough while the wet-season boost stays
moisture-budget-limited.) Deliberate trade of a zonal *magnitude* metric that contains no biome
information (refErr = zonal temperature bias + precip ratio only) for a large, on-target improvement
in biome classification against the project's designated Köppen reference map.
**Residual keeping this 🟡 rather than 🟢**: Aw+Am is 6.65% vs Earth's ~18–20%. This knob converts
Af→Aw/Am *within* an already-too-small tropical band (total tropical land 13.30%→12.66% across the
sweep vs Earth's ~24–27%); it cannot grow the band itself. Pushing to 2.0 overshoots Af to 3.95%
(now *below* Earth) while Aw+Am only reaches 8.71% — so this specific lever is exhausted at 1.7.
**Next lever for the residual**: the tropical band's total extent, not its internal Af/Aw split.

**A negative result worth not repeating** (tried and reverted this session): the hypothesis that
`_raw_conversion_gain` should migrate seasonally (it is built from `drybelt_regime_window`, a pure
|latitude| shape with zero `day_of_year` dependence — the same gap class already fixed twice here)
was implemented and swept 0.0→1.3. It changes final precipitation by **~0.05%** (savanna-band wettest
month 4.325→4.327 mm/day) and moves no Köppen cell at all. Reason, measured: at savanna latitudes the
moisture-budget rescale pins output to `target_row_mm_day`, so a raw-production-side change is
absorbed almost entirely by a compensating change in synthetic fill. **Corollary worth remembering:
at these latitudes the *target* is the lever, not raw production** — which is precisely why the
target-side knob above worked. Reverted rather than shipped inert (unlike other 0.0-default params
here, it provably does nothing for its stated purpose, so it would be misleading dead code).

---

<details>
<summary>Original regression writeup (superseded by the fix above, kept for the diagnosis trail)</summary>

### A2 (historical). 🔴 Tropical rainforest over-extent / savanna under-extent — WORSENED by the A5 fix
**Symptom**: Sub-Saharan Africa and other tropical land render as near-unbroken Af (rainforest)
with almost no savanna transition, vs. real Köppen's broad Aw/Am bands flanking a narrower Af core.
**Re-measured 2026-08-01, and it's a regression, not the improvement A5's own writeup assumed**:
added the Af/Am/Aw split to `scripts/check_real_terrain_koppen.py`'s `_koppen_breakdown` (previously
only reported combined `tropical_pct`, per this section's own long-standing complaint) and directly
bisected the tracked deterministic 64×128 benchmark (`real_terrain_validation.py`'s
`RealTerrainValidationConfig()` defaults — 1yr spinup + 1yr eval, matching
`testing/fixtures/real_terrain_validation_baseline.json`) against commit `4bdc79a` (immediately
pre-A5) vs current:

| | pre-A5 | post-A5 | Earth |
|---|---|---|---|
| Af | 8.82% | 11.05% | ~6–7% |
| Am | 1.48% | 0.22% | — |
| Aw | 6.68% | 2.03% | — |
| **Aw+Am combined** | **8.16%** | **2.25%** | **~18–20%** |

Af moved *further* from target (worse), and Aw+Am — the specific under-represented category this
item tracks — collapsed to barely a quarter of its pre-A5 share. **Mechanism**: A5's
`_raw_conversion_gain` sharpens the wet/dry contrast by boosting all non-dry-belt raw production
uniformly (up to 5.5×) while deliberately leaving dry-belt production untouched — this is exactly
right for A1 (creates real desert-vs-non-desert contrast) but wrong for A2, which needs a *smoother*
wet→dry gradient (savanna as a genuine transitional category) rather than a sharper wet/dry cliff.
The same mechanism that fixed one gap widened the other — **these two items are now in direct,
measured tension**, not independently solvable by the same lever.
**Controlling variables**: `PlanetParams.itcz_seasonal_response` (0.7), `itcz_seasonal_target_response`
(1.0), `itcz_zonal_smooth_deg` (8.0), `precip_raw_shape_weight` (0.0, inert), and now
`atmosphere.generate_precipitation`'s `_raw_conversion_gain`/`drybelt_regime_window` (see A5) — the
newly-identified direct cause of the regression.
**How to test**: `scripts/check_real_terrain_koppen.py`'s `_koppen_breakdown` now reports
`af_pct`/`am_pct`/`aw_pct` individually (added 2026-08-01); or the deterministic 64×128 benchmark via
`real_terrain_validation.run_real_terrain_validation()`, reading `state.koppen_type` directly.
**What's been tried**: static-ITCZ fix (real, ~1pp gain, pre-A5), target-deficit fix (real, larger
gain pre-A5, Af 11%→8%, Aw 2%→5%), per-cell raw-shape redistribution (implemented correctly, found 2
real bugs, but net-negative), and now A5's regime-gain fix (real gain for A1, net-negative here).
**Root cause, now precisely characterized**: the wall isn't just "raw production is too scarce"
(A5 addressed that) — it's that raising raw production *uniformly* across the whole non-dry-belt
range doesn't create an intermediate savanna regime, it just makes rainforest-adjacent land wetter
and steppe-adjacent land drier, both pulling *away* from the Aw classification band in the middle.
**Recommended next lever** (proposed here, then **tested and disproved** the same session — see the
negative result in the current A2 entry above): gating `_raw_conversion_gain` by seasonality was the
obvious-looking fix, but raw-production-side changes are absorbed by the moisture-budget rescale at
these latitudes. The target side (`itcz_seasonal_target_response`) is what actually worked.

</details>

</details>

### A3. 🟡 Continental interior still short of target (US Midwest specifically)
**Symptom**: US Midwest chronically the hardest of the six named boxes to bring up to Earth values.
**Current numbers, independently re-verified 2026-08-01** (post-A5-fix, same run as A1): Canadian
Prairies **437 mm/yr** (target 400–500 — back in range after the A5 fix); Central Europe **502**
(target 550–750 — still under range); US Midwest **709** (target 800–1000, ~11–30% short — better
than the pre-fix level, but not fully closed).
**Correction (2026-08-01, same-session self-correction)**: this document previously claimed Central
Europe was "a new small regression" from A5, citing 639 as its pre-fix value. That 639 figure was
stale (from an unrelated, non-comparable earlier session's measurement), not a clean same-methodology
baseline. A direct, same-save/same-code bisection against commit `4bdc79a` (immediately pre-A5) shows
Central Europe was **463 mm/yr before A5**, i.e. A5 genuinely *improved* it (463→502, +8.4%) — still
short of its 550 floor, but a real gain, not a regression. Always bisect against the immediately-prior
commit on the identical save/config when checking whether a fix helped or hurt a specific box; a
number carried over from a different session's measurement is not a valid baseline (see process
note 6's sibling lesson about this exact mistake).
**Controlling variables**: `ferrel_v_centre_deg` (44.0, down from 48.0), `ferrel_v_land_shift_deg`
(-4.0, decouples land Ferrel-cell center from ocean's), `PlanetParams` land-temperature bonus terms
in `simulate.py` (`_midlat_storm_bonus_1d`, `_atm_land_transport_1d`, `_handoff_bonus_1d`), the
land-cap taper (`_land_cap_1d`).
**How to test**: `scripts/check_real_terrain_koppen.py --wind-diagnostics`, specifically the
`div`/`ascent` fields at 38–45°N.
**What's been tried**: ageostrophic cross-isobar scaling by continentality (reverted — amplifies
whatever flow the pressure pattern already has, doesn't reliably push toward convergence);
pressure-field deepening (reverted — moved div/ascent the wrong direction even at ~2.7 hPa
depression); land-only Ferrel centre shift (shipped, real gain, but capped at -4.0 rather than the
box-preferred -8.0 because -8.0 tips a real 45–65°N soil-moisture desiccation-spiral guard).
**Root cause, found later than the box-level fixes above (2026-07-25, `part1-overnight-batch`)**:
this is **not fundamentally a Midwest-local problem**. The model's zonal-mean divergence profile
peaks at 38–45°N vs Earth's ~25–30°N — the entire subtropical dry belt is displaced ~10° poleward,
and 85% of the Midwest's own divergence is literally the zonal-mean signal, not a local anomaly.
Every land mass at 38–45°N is affected; the Midwest is just the box that happens to get sampled.
This retroactively explains why two separate local-perturbation attempts both failed — they were
each fighting an 85%-weight zonal-mean signal with a ~15%-weight local term of the wrong sign.
**Recommended next lever**: the prime suspect named in that session is the latitude-only 3-cell
`w_mid` profile in `generate_wind_field` (centered at 48°) — this is a structural
shape-of-the-meridional-circulation issue, not a magnitude-tuning one, and per the ferrel-land-shift
work, the *land-only* decoupling approach (rather than a uniform shift) is the mechanism that's
actually worked so far without collateral damage. Worth checking whether the ocean-side crossing
latitude can also be independently corrected the same way the land side was, since ROADMAP's own
"2D barotropic gyres" item would let ocean circulation shape emerge from basin topology instead of
a single global latitude constant.

### A4. 🟢 SE US / East China / S Japan monsoon-margin misclassification — mostly fixed
**Symptom (was)**: all three regions rendered as BSh hot steppe despite being real Köppen Cfa
(humid subtropical) — they sit in the same latitude band as the true deserts but escape the
subtropical high via Gulf Stream/Kuroshio monsoon moisture in reality.
**Fix shipped**: `PlanetParams.monsoon_east_margin_exemption` (1.5) — an ocean-to-the-east adjacency
gate decayed ~20 cells inland, applied post-hoc after the ITCZ zonal smoothing (placement mattered:
baking it in pre-smoothing measured almost no effect).
**Residual**: S Japan fully fixed (100% Cfa); SE US majority-fixed (61% Cfa); East China only
reaches a plurality (35% Cfa / 39% BSh / 19% Af, never a clean majority — pushing the exemption
strength further starts overshooting into Af and bleeding into US Midwest). Köppen classification
shares themselves haven't been re-measured post-A5-fix (10yr EMA lags too much to reflect it yet
regardless — see A5's sampling note).
**Magnitude update, independently re-verified 2026-08-01** (post-A5-fix, same save/run as A1/A3;
pre-A5 baseline confirmed via clean same-save bisection against commit `4bdc79a`, not a stale
cross-session number — see A3's correction note for why that distinction matters): SE US
**745→772 mm/yr** (modest real gain), S Japan **840→1138** (now clears the 1100–2200 target range
entirely), East China **745→578** (a real, confirmed drop, not an artifact) — a mixed, not uniformly
positive, side effect of A5's wet-regime recalibration on the monsoon boxes specifically.
**East China investigated 2026-08-01**: not a simple saturation or single-formula cause like the
orographic-test bug — `precip_target_achieved_fraction` for East China's latitude rows only drops
marginally post-A5 (~0.94→~0.92), far too small on its own to explain a 22% final drop, while
`zonal_rescale_factor` needed for those rows rises (~1.4–1.9 → ~1.6–2.2). This means the effect
compounds through the moisture-budget's step-to-step feedback (soil moisture → land_evap → humidity)
over the 2-year measurement window rather than being visible in any single step's snapshot — a
genuinely different, harder-to-isolate cause than the two bugs already fixed in A5's own entry.
**Not yet root-caused**; flagged for a future session rather than force-diagnosed here, since single-step
debug-field snapshots (the diagnostic method that worked for the orographic/cloud bugs) don't expose it.
**Real, accepted cost**: Kalahari's own BSh share drifted 91%→84% (the exemption's inland decay
reach overlaps southern Africa's Indian Ocean/Mozambique-Channel coast).

### A5. 🟡 Structural root cause underlying A1–A4: chronic raw-production deficit — largely fixed
**This is the single most important item in this document.** Every named regional/biome gap above
(desert dryness, Af/Aw ratio, continental shortfall, monsoon-margin magnitude) traces back to the
same upstream mechanism, confirmed independently from at least four different investigative angles
across many sessions. It remained open through the first 2026-08-01 pass: two more angles were
tried below and neither closed it, though one produced the quantified link between A1 and A3/B1
that enabled the successful follow-up above.

**Follow-up fix shipped 2026-08-01.** The successful change is the deferred
regional architecture, not another flat multiplier or redistribution-only pass:

- Divergence is decomposed into its zonal-row background and cell-local anomaly. Only the
  contaminated zonal component is gated away outside the true subtropical regime; local
  subsidence remains intact.
- The dry regime is represented by a smooth 16–34° core with shoulders rather than the old
  narrow 28° Gaussian alone. This covers real desert boxes without extending into the Midwest.
- Raw precipitation conversion is calibrated to the atmospheric residence-time deficit in wet
  regimes (5.5×), while the genuinely low-production dry belt is deliberately not amplified.
- The 38–45° land band gets a mean-preserving share of its row target, preventing ocean cells at
  the same latitude from consuming the correction. The monsoon inland mask was also made
  resolution-invariant; its former fixed 20-cell reach meant 7° at 1024 columns but 56° in the
  tracked 128-column fixture.

**Seasonally balanced 512×1024 result** (`saves/earth.pkl`, 730 days MONTHLY, second year),
independently reproduced 2026-08-01: `global_rescale_factor` **5.462→2.064** (62% less synthetic
fill); Sahara **586→131**, Kalahari **509→130**, Atacama **419→102**, and US Midwest **480→709
mm/yr**. The three general desert boxes now clear their `<200` target (Atacama remains above its
special `<50` target), while the Midwest deficit is cut by roughly half without drying the Prairies
(439) or creating a zonal precipitation increase. On the deterministic 64×128 benchmark, reference
error improves **0.457→0.318**, arid land is **17.7%** (Earth ~19–20%), Sahara/Kalahari are
**195/157**, and the Midwest is **979 mm/yr** (in target).
> **Caveat added 2026-08-02**: the "arid land is 17.7% (Earth ~19–20%)" figure here is a *cell-count*
> percentage and is biased — see A2's banner. Area-weighted, arid land is **28.9%** against Earth's
> **26.4%**, i.e. the model is slightly *over*-arid rather than slightly under. This **inverts the
> sign** of that particular piece of evidence; it does not affect A5's precipitation numbers (mm/yr
> box means and `global_rescale_factor` are unaffected by the Köppen counting bug), so A5's core
> conclusion stands.

Residual: the real-terrain Midwest result
is still ~11% below the bottom of its target and the remaining 2.08× rescale is concentrated by
design in regimes whose raw production should be low; this is a large structural closure, not a
claim that all precipitation calibration is finished.

**Two real bugs found and fixed 2026-08-01 in this fix's own test coverage** (the commit that
shipped the numbers above landed with 3 failing tests; investigated and resolved, not just silenced
— see process note 6): `_raw_conversion_gain` (the wet-regime raw-production multiplier above,
up to 5.5×, unconditional — unlike every other lever in this document it shipped with no
`PlanetParams` gate) pushes `remove_frac_prerescale` to its pre-existing 0.85 ceiling specifically in
high-elevation/strongly-orographic latitude rows, clipping away most of the local windward-vs-leeward
orographic differential exactly where it should be largest, while leaving unrelated
wind-direction-dependent moisture-convergence effects in low-elevation rows unclipped — confirmed by
direct ablation (forcing the gain to 1.0 restores both the old orographic-uplift-exceeds-rain-shadow
invariant and the pre-fix cloud-fraction level). Both `orog` and `rain_shadow_suppression`
themselves were directly verified to remain correctly signed throughout — the underlying meridional
`gy` sign convention this project fixed in an earlier session is intact; only the aggregate
`P.mean()` metric that used to proxy for it is now legitimately confounded by this new regime
mechanism. Fixed by making `testing/test_derivative_signs.py`'s orographic regression test check
`orog`/`rain_shadow_suppression` directly (immune to the confound) rather than final precipitation,
and by moving `testing/test_cloud_feedback.py`'s cloud-fraction floor 0.12→0.10 (physically coherent
— less raw production gets "saved" for the moisture-budget's targeted fill and more gets stripped
out directly as rain, leaving less residual humidity for cloud formation — following this test's own
established "the floor moves, not the physics" practice for prior deliberate recalibrations). The
golden-state fixture (`testing/fixtures/golden_state_reference.pkl`) was regenerated against the
now-verified-intentional physics change, per its own docstring's instructions.
**Not yet fixed, flagged for a future session**: the `remove_frac`-saturation interaction itself is a
real, if narrow, structural finding independent of this specific test — `orog`'s own within-run
percentile normalization (`orog / percentile(orog, 90)`) means it provides *zero* net
windward-vs-leeward differentiation on any spatially-uniform-slope terrain regardless of how steep
that slope is (directly verified: 100m/300m/1000m-per-row ramps give byte-identical results), so the
entire orographic effect in practice rests on `rain_shadow_suppression`'s much smaller ~20% uniform
suppression term fighting against whatever else is happening in a given region. This was already true
before A5; A5's gain just made the interaction visible by amplifying both sides enough to flip a
previously-thin margin. A genuine fix (making `orog` differentiate on absolute terrain-relief
magnitude, not just relative-within-run shape) is out of scope for this session but would likely
improve real-terrain windward/leeward contrast (Cascades/Sierra Nevada rain shadows, Andes,
Himalaya/Tibetan-Plateau lee deserts) beyond what this synthetic test alone can reveal.
**The mechanism**: `atmosphere.py`'s raw, pre-rescale `precip_potential` chronically produces far
below `target_mean_mm_day` — the moisture-budget rescale (`_moisture_budget_precip_rescale`)
compensates with a per-row multiplier (`global_rescale_factor`) that averages **~5.47x** on real
terrain (re-measured 2026-08-01, `saves/earth.pkl`, 2yr MONTHLY continuation averaged over a full
seasonal cycle — see the process note on sampling below; earlier sessions measured 3.0–5.3x
depending on exact code state and sampling window — always well above 1.0). Because the fill
mechanism supplies the majority of a typical row's final rain, **where the fill mechanism places
its synthetic addition determines the spatial pattern almost as much as raw physics does.**
**Controlling variables**: `PlanetParams.target_mean_mm_day`, the `_moisture_budget_precip_rescale`
function's `max_total_removal_fraction`/`max_added_removal_fraction` caps (0.85/0.15), `cell_weight`/
`target_cell_weight` (built from `subsidence_suppression`, which is a near-exact no-op *inside* the
ITCZ — deserts are a subsidence phenomenon, the deep tropics structurally aren't, so there was never
a real per-cell differentiation mechanism operating within the tropics until `precip_raw_shape_weight`
was tried and found net-negative).
**How to test**: instrument `debug_fields["global_rescale_factor"]`/`zonal_rescale_factor` directly
(cheap — no simulation stepping needed via `check_real_terrain_koppen.py`'s own wind-diagnostic
helper) and check what fraction of rows sit pinned at whatever the current ceiling is. **Sampling
caveat found this session**: `check_real_terrain_koppen.py`'s "instantaneous, 2nd half" window is
literally half of `--days`, not a seasonally-balanced sample — at the default `--days 365` it's only
~6 months, which can catch different named boxes at different points in their own seasonal cycle and
produce a spurious ranking (see the ruled-out lead below). Always use `--days 730` (or greater, in
multiples of 365) so the "2nd half" spans one full 12-month cycle before trusting any single-box
comparison.
**What's been tried and failed** (do not re-attempt without a new angle): raising the rescale
ceiling / rescaling `precip_potential` pre-clip instead of post-clip (made desert/continental
ranking *worse* — draining continental interior's `q` faster than `land_evap` replenishes it);
strengthening moisture transport (monotonically dries continental interior — the RH-threshold
convective trigger consumes moisture locally before it can travel); redistribution-only mechanisms
at every granularity tried (row-level, per-cell-in-tropics) — all trade cells *between* categories
(Aw↔BS, desert↔continental) rather than genuinely raising output.
**Earlier angles tried and characterized, 2026-08-01** (historical results before the successful
follow-up architecture above):
- A false lead first: a single-window (`--days 365`) sample showed Sahara's raw pre-rescale
  `precip_potential` (0.0532) ranking *above* US Midwest's (0.0437) — seemingly contradicting this
  document's own claim that raw physics ranks wet/dry areas correctly. Re-measured with a
  seasonally-balanced 2-year sample (see sampling caveat above): the ranking is correct after all
  (Sahara 0.0481 < Midwest 0.0547) — this was purely a seasonal-sampling artifact, not a regression.
  Recorded here so a future session doesn't re-chase the same false lead.
- `moisture_budget_tropical_cap_boost` (raises `_moisture_budget_precip_rescale`'s removal-fraction
  caps inside the ITCZ only, gated by `itcz_window`): swept 0.0/0.5/1.0 on real terrain — **no
  measurable effect at any strength** (every named box and `global_rescale_factor` stayed within
  measurement noise of baseline). The moisture-budget fill is not actually capacity-limited by these
  caps in practice; the finer, already-existing per-cell `target_cell_weight` ceiling is the binding
  constraint, and raising the coarser row-level cap on top of it does nothing. Confirms in code what
  was previously only inferred: this specific lever is a dead end, not just untried.
- The first `subsidence_divergence_regime_gate` implementation (see B1 below for the mechanism and
  full sweep table): a real,
  measurable effect, but a straight trade-off, not a fix — it meaningfully helps US Midwest reach its
  target at the direct cost of making every desert box measurably wetter (Sahara +30% at full
  strength, already 2.9x over target before this change). Same failure shape as every previously-tried
  lever in this section: moved the same finite "how much rain, and where" budget between regimes
  rather than raising raw output. It initially shipped inert; the follow-up replaced its
  whole-signal gate with the zonal-background decomposition and calibrated that version at 1.0.
**Why this is hard**: `spherical-metric-recalibration-blocked` found that `conv` (the convergence
driver) is renormalized to mean=1 before use, making the whole computation scale-invariant at that
point — there is no flat gain anywhere upstream that can raise the total without also being
divided back out. Any real fix has to change the *distribution*, not find a missing scalar.
**Recommended next lever** (unattempted, per the itcz-global-rescale-coupling session's own
unfinished list): a genuinely regional/zonal-regime rescale (separate scale factors for
tropical / subtropical-dry / mid-latitude / polar regimes) instead of one row-level or one global
scalar, so fixing one band's over/under-production doesn't mechanically leak into every other band
via a shared multiplier. This is explicitly the larger, not-yet-attempted architecture change that
every session so far has deferred as "bigger scope." **Narrowed down this session**: this session's
own `subsidence_divergence_regime_gate` experiment shows the row-level-vs-cell-level distinction
matters less than which *underlying signal* (`subsidence_suppression`) the redistribution reads —
the real leverage point is fixing that signal's own regime-accuracy (see B1), not building another
layer of caps/weights on top of it.

---

## B. Atmospheric circulation

### B1. 🟡 Subtropical dry belt displaced poleward — precipitation impact fixed, circulation residual
Restated here because it's a circulation-model gap, not just a precip-formula gap: the zonal-mean
divergence-to-convergence crossing sits at ~48°N (partially improved to ~42–46°N via
`ferrel_v_centre_deg`/`ferrel_v_land_shift_deg`) vs Earth's ~40°N. **Test**: zonal-mean `div` profile
from `check_real_terrain_koppen.py --wind-diagnostics`, compare crossing latitude directly against
an Earth reanalysis reference (not yet automated — see H's "gridded validation" gap).

**2026-08-01 follow-up:** the precipitation-side contamination is now corrected by decomposing
`div = zonal_mean(div) + local_anomaly` and gating only the first term outside the 16–34° dry
regime. This is the cell-level refinement requested below: it retains genuine local subsidence,
unlike the failed whole-signal latitude gate. Together with stronger upstream dry-regime
suppression and wet-regime raw production, the full-resolution result is Sahara/Kalahari/Atacama
131/130/102 and Midwest 709 mm/yr. The underlying diagnostic wind crossing is still not an
automated reanalysis match, so B1 remains yellow rather than green; its damaging precipitation
coupling is no longer open.
**New, precisely-quantified evidence (2026-08-01)**: this displacement doesn't just move where the
dry belt *sits* — it actively contaminates `atmosphere.py`'s `subsidence_suppression` signal (the
same one A1 relies on to suppress desert precip), because that signal's raw-divergence term has no
latitude-regime awareness. Measured directly on `saves/earth.pkl` (512×1024, 2yr MONTHLY real-terrain):
US Midwest's own `subsidence_suppression` averages **0.196** — nearly as suppressed as the Sahara's
own **0.446** mean — even though Midwest sits far outside `drybelt_window`'s ~28° peak and should see
almost none of this gating. The first `PlanetParams.subsidence_divergence_regime_gate` implementation
was built at a 0.0 default and swept by multiplying the whole raw-divergence signal by
`drybelt_window`:

| `subsidence_divergence_regime_gate` | Sahara | Kalahari | Atacama | Canadian Prairies | US Midwest | Central Europe |
|---|---|---|---|---|---|---|
| 0.0 (baseline) | 586 | 509 | 419 | 369 | 480 | 467 |
| 0.5 | 670 | 522 | 426 | 444 | 670 | 547 |
| 1.0 | 759 | 556 | 447 | 492 | **814** (in target range) | 608 |

(mm/yr, instantaneous 2nd-half of a 2yr real-terrain run; desert targets all <200–500, US Midwest
target 800–1000.) At full strength Midwest reaches its target range, but every desert gets
meaningfully *worse* in the same run (Sahara already 2.9x over target at baseline, +30% further at
gate=1.0) — a real, characterized trade-off, not a fix, so this shipped at the inert 0.0 default. The
mechanism is real and worth another session's attention: the gate as built is too coarse (a smooth
function of `drybelt_window` alone), and doesn't distinguish "this cell is at the true dry-belt
latitude" from "this cell merely shares a latitude *band* with the dry belt" — real deserts span a
range of `drybelt_window` values themselves (Sahara's own `subsidence_suppression` was measured
ranging 0.166–1.000 across the box), so any latitude-only gate relieves part of the real desert along
with the false-positive continental interior. A cell-level (not latitude-only) discriminator between
"genuinely locally subsiding" and "merely at a contaminated latitude" is the more promising refinement,
not yet attempted.

### B2. 🟡 Jet stream equatorward of real position, NH/SH asymmetry
**Symptom**: prognostic upper-level jet (`evolve_wind_aloft`) settles at ~18–30°N / ~18–41°S vs.
Earth's ~25–40° subtropical/polar-front bands; SH core measured notably weaker than NH (~9–18 vs
~24–29 m/s) in at least one snapshot, not resolved as real vs. sampling noise.
**Controlling variables**: `PlanetParams.wind_upper_hadley_edge_deg` (24°, widened from 12° to fix
an earlier position bug), `wind_upper_pgf_amp` (40.0, recalibrated from 8.0 for realistic m/s
magnitude).
**How to test**: a full-year weekly-sampled zonal-mean `wind_u_aloft`/`wind_v_aloft` profile — the
existing test suite only checks a single 2-year snapshot at one band, which is too narrow to see
either the position or the asymmetry issue (this was true before the polar-singularity and
equatorial-suppression bugs were found and fixed; both were invisible to the narrower test).
**What's fixed already**: a polar PGF singularity (cos(lat) floored at 65°) and a missing
equatorial-suppression term (false "jet" locking onto 7–15° latitude) were both found and fixed —
see `jet-stream-vs-real-world`. What remains is a genuine physics-shape gap: real jets form where
the meridional temperature gradient itself peaks (~30–50°N on Earth); this model's simulated T
gradient peaks more equatorward, and reshaping that would need touching how `p_anom` derives from
T(lat) curvature — bigger scope than the numerical-stability pass that fixed the other two bugs.
**Not yet done**: no seasonal-migration check (does the model jet actually shift latitude with the
seasons the ~10–15° Earth's does?).

### B3. 🟢 ITCZ row-to-row/zonal shape inconsistency — fixed
Was: two longitude columns 5° apart (e.g. 15°E vs 20°E through Africa) showing completely different
ITCZ shapes at the same latitude (one an unbroken rainforest block, the other a dry equatorial notch
with double wet peaks) — caused by Rossby-wave standing-pattern noise (wavenumbers 3/5/7) dominating
equatorial divergence where Coriolis damping vanishes (f→0). Fixed via `itcz_zonal_smooth_deg=8.0`
(longitude-only Gaussian smoothing on `subsidence_suppression`). **Real, accepted cost**: Atacama got
wetter (195→448 mm/yr in one measurement) since this smoothing competes with Atacama's narrow
coastal-strip aridity signature — feeds directly into A1's open desert-dryness gap.

### B4. ⬛ No diurnal cycle
No time-of-day temperature modulation exists anywhere in the codebase (confirmed: no `diurnal`/
`tidal_lock` references in `planet_params.py` or `temperature.py`). Real convective precipitation
timing (afternoon thunderstorms) and continental diurnal temperature range are both unmodeled.
**Recommended scope**: DAILY-mode-only (skip at weekly-and-coarser per ROADMAP's own guidance), a
cheap sinusoidal `T_air` modulation over land. On the roadmap since the project's early sessions,
never attempted.

---

## C. Temperature

### C1. 🟡 Mid-latitude winter cold bias — largely fixed, real residual remains
**Symptom (was)**: 45–55°N coldest-month temps of -37 to -39°C (Europe/Russia/Canadian Prairies
misclassified Dwd extreme-continental instead of Dfb). Root cause: a real trough between two
independently-tuned heat-transport bonus terms in `simulate.py`, each deliberately capped so it
wouldn't disturb the other's already-validated range, leaving a gap right at 45–55°N.
**Fix shipped**: new `_handoff_bonus_1d` term (peaks 20K at ~50–52°, zero outside 44–66°).
**Residual**: post-fix coldest-month values land at -21 to -26°C — no longer *impossible*, but still
colder than the real -12 to -20°C range depending on maritime/continental exposure. **Not closed,
just brought into physical plausibility.**
**Controlling variables**: `simulate.py`'s `_midlat_storm_bonus_1d`, `_atm_land_transport_1d`,
`_handoff_bonus_1d` (the sum of all three across latitude is what matters, not any one in
isolation — this is exactly the class of bug that hid for multiple sessions because each term was
validated independently).
**How to test**: box-averaged (not single-pixel) sampling at named cities (Berlin, Moscow, Winnipeg,
Novosibirsk, Kiev) via `monthly_temp`, compared against the coldest DAILY-mode reference, not just
the 10yr Köppen EMA (which lags and can hide an already-fixed bug behind stale classification — this
exact confusion happened once already, see `static-itcz-seasonal-fix`'s "false alarm" note).

### C2. 🟢 Calendar aliasing — ANNUAL/MONTHLY seasonal-phase drift — fixed, verified 2026-08-01
**Was**: the old `optimizer/headless.py`'s `_SUBSTEPS[ANNUAL]` advanced exactly 364.0 days per
nominal year vs. `orbital_period_days=365.2422` — a 1.2422-day/year phase slip that completed a
full false seasonal oscillation every ~294 years, producing a spurious multi-century oscillation in
any long ANNUAL diagnostic (previously mistaken for genuine climate drift at least once — a
reported "+1.56 K fixed-CO2 control drift" was actually two samples at opposite seasonal phases).
**Fix** (already shipped, commit `dfaf41c`, 2026-07-26 — this document's 🔴 status was stale): the
substep schedule was centralized into `time_policy.py`'s `substeps_for_mode`/`cycle_days`, both
now derived from `planet_params.orbital_period_days` directly instead of a hardcoded day count —
`MONTHLY` = `period / (12 * 5)` per substep × 5, `ANNUAL` = `period / 52` per substep × 52 — so one
ANNUAL cycle is the *exact* orbital period (`365.2422` for Earth) with zero residual phase slip by
construction, for any planet preset. Both the GUI (`gui_worker.py`) and headless runner
(`optimizer/headless.py`) consume this same shared function, so there's no separate code path left
with the old fixed-364 assumption. (One unrelated, harmless `364.0` remains in
`scripts/profile_simulate_step.py` — a CPU-profiling script that runs a handful of steps per
invocation and never accumulates day_of_year across simulated years, so it cannot exhibit the
aliasing artifact; not in scope.)
**Verified 2026-08-01**: `testing/test_generalize_time_orbit.py::test_slow_mode_cycles_match_planet_orbit_exactly`
(both EARTH and MARS presets) and `::test_daily_and_weekly_cycles_remain_literal_days` — both pass,
directly asserting `cycle_days(ANNUAL, pp) == pp.orbital_period_days` and
`12 * cycle_days(MONTHLY, pp) == pp.orbital_period_days` to `1e-10` absolute tolerance. No further
action needed.

### C3. 🟢 Hardcoded Earth constants block Mars/exoplanet realism — elevation + lapse rate FIXED 2026-08-02
Not wrong *for Earth*, but actively wrong the moment a non-Earth `PlanetParams` preset is used.
**Both headline constants are now parameterized — see the fix note at the end of this entry.**
The inventory below is the pre-fix state, kept because it names the exact sites:
- **Max terrain elevation** (8848m/Everest) hardcoded in **four separate places with at least three
  different formulas**: `temperature.elevation_to_alt_km` (piecewise, two branches),
  `climate_averages.compute_biome_type` (two sites, a different simpler linear formula),
  `main.py` (a third power-curve formula), `terrain.py` (comments/lookup assuming the same
  ceiling). Loading Mars terrain today silently rescales Olympus Mons (~21.9km, 2.5x Earth's max)
  down to Earth's height range.
- **Lapse rate** (6.5 K/km) hardcoded in **five call sites** across `simulate.py`,
  `temperature.py`, `climate_averages.py`. Mars's real lapse rate (~2.5 K/km, lower gravity + CO2
  atmosphere) is currently unreachable — Mars terrain gets Earth's cooling-per-km verbatim.
- **0.622 epsilon** (Rd/Rv, water-vapor physics) hardcoded in four sites — low priority in practice
  since `MARS.has_liquid_water_ocean=False` already gates essentially all humidity physics off for
  the one non-Earth preset that currently exists; inert until a water-bearing exoplanet preset
  exists to exercise it.
**FIXED 2026-08-02.** `PlanetParams.max_elevation_km` (Earth 8.848, **Mars 21.9** — Olympus Mons)
and `PlanetParams.lapse_rate_k_per_km` (Earth 6.5, **Mars 2.5**) added and threaded through every
cited site:
- `temperature.elevation_to_alt_km` — both the loaded-heightmap and procedural branches (the loaded
  branch's `8748.0` is now derived as `max_elevation_m - 100.0`, the 100 m consumed by its own
  linear lowland segment).
- `temperature.generate_temperature_overlay` — now takes an optional `planet_params`.
- `simulate.py`'s orographic-cooling block — reads both from `_pp`.
- `climate_averages.classify_koppen` — both its elevation-delta and legacy full-elevation branches,
  as scalar kwargs matching that module's existing `orbital_period_days` convention; `simulate.py`
  passes them from `pp`.
- `main.py`'s hover-readout altitude formula — sourced from the active preset.
**Verified**: the Earth path is **bit-identical** (`np.array_equal`) to the old hardcoded formulas on
both branches of `elevation_to_alt_km`, so this is an exact no-op for Earth; Mars terrain now reaches
a real 21.90 km instead of being silently rescaled to Earth's 8.85 km ceiling (2.48x).
**Regression guard**: `testing/test_no_hardcoded_earth_constants.py` — which already existed for this
exact class of bug — was extended with patterns for `8848`/`8748`/`8.848`/`6.5`, so these constants
cannot be reintroduced silently. That test's `_code_lines` helper was also taught to skip
triple-quoted strings: docstrings legitimately *describe* these values in prose, and without that the
guard would have forced an ever-growing allowlist of documentation lines and become noise. The
extended guard was self-tested (planted violations are caught; it is not passing by having stopped
looking).
**Residual, unchanged**: the **0.622 epsilon** (Rd/Rv) is still hardcoded at four sites — deliberately
left, since `MARS.has_liquid_water_ocean=False` gates essentially all humidity physics off for the
only non-Earth preset that exists, making it inert until a water-bearing exoplanet preset exists to
exercise it. `terrain.py`'s references are comments/lookup prose, not computation.

---

## D. Ocean

### D1. 🟢 Deep ocean equilibrium target is unphysical — investigated, understood, not a "bug" to patch blindly
**Symptom**: deep-ocean temperature (`T_deep_ocean`) drifts toward ~25°C rather than the real
abyssal ~1–4°C, on an effective timescale of **τ_eff ≈ 2219 years** (measured 2195yr on a 500yr
real-terrain run, 1% match to the analytic value).
**Root cause**: the deep layer exchanges heat *only vertically*, with the mixed layer directly above
it, at the same grid cell — there is no overturning or lateral abyssal transport. It relaxes toward
each column's *own local* surface temperature, not toward a globally-mixed value. Real Earth's
abyssal ocean is uniformly cold specifically because of the actual overturning conveyor (North
Atlantic + Southern Ocean deep-water formation); this model has no analogue.
**Why not to naively "fix" it**: raising `deep_ocean_exchange_rate` makes the model reach its *wrong*
target (~25°C) faster, not a more realistic one — this was directly measured and inverts an earlier
session's mistaken conclusion. A 100-year real-terrain extension showed steady, unflattened linear
warming (-2.14°C → -1.47°C, no curvature after 100yr / 4.75 nominal time constants) — consistent
with genuine multi-century transient physics, not a stuck/frozen field.
**What exists but is inert**: `PlanetParams.abyssal_overturning_coeff` (0.0 default) — a meridional
Laplacian-diffusion mechanism mirroring `eddy_heat_flux_coeff`'s pattern, wired and tested (confirmed
it produces a different, coefficient-dependent state) but **never validated against the actual
physics question** (does it pull the equilibrium back toward ~2–4°C?) because that requires a
multi-century run, out of scope for any session so far.
**Recommended next step**: a dedicated long-run (multi-century, small grid for speed) calibration
session tracking `T_deep_ocean`'s zonal-mean profile and equilibrium value at several
`abyssal_overturning_coeff` values — same template as the existing ECS-sensitivity experiments.

### D2. 🟡 AMOC is semi-prescribed, not fully prognostic
`amoc_bonus_near`/`amoc_bonus_far` (3.0K/9.0K) are fixed magnitude constants representing the
Gulf-Stream/thermohaline warming bonus; `salinity_amoc_scale` (1.0) does let `amoc_factor` respond
to a real, prognostic North Atlantic salinity anomaly (a +1 PSU anomaly multiplies amoc_factor by
1.15) — so this is **partially** done, contrary to ROADMAP's "AMOC is currently just a scale factor"
framing, which is stale. What's still missing: AMOC strength doesn't respond to *temperature*
density contributions (salinity-only), and the base bonus magnitudes themselves are prescribed
constants rather than derived from an actual overturning-strength calculation — no freshwater-hosing
or bistability/collapse experiments are possible without that. `gradient_nh` (NH pole-equator
gradient) also still runs ~22K at 60×120 vs. a 40–65K target, partly an AMOC over-warming artifact
at higher resolution (lower resolutions have a separate, unrelated synthetic-terrain artifact).

### D3. 🟡 No real cold-current/coastal-upwelling physics — SST half now EXISTS (D5); the missing link is SST→land coupling
**Substantially revised 2026-08-02 by D5's calibration pass.** The old framing — "eastern-boundary
cooling was never implemented, only the symmetric western-boundary warming exists" — is **no longer
accurate**. Enabling `ocean_gyre_strength` (now default 1.0, see D5) produces real, coherent
eastern-boundary cooling as an emergent consequence of the wind-stress-curl gyre solve, in every
region this mechanism is supposed to explain: **Benguela −0.54 K, Canary −0.36 K, Humboldt −0.25 K,
California −0.22 K** (SST anomaly vs. gyres-off, on the tracked 64×128 benchmark), alongside the
matching western-boundary warming. That is genuine cold-current physics, not a proxy gate.

**But the gap this entry exists for is not closed — it has moved, and is now precisely located.**
The upwelling-side SST signal does **not** reach land precipitation: Atacama is unchanged at
146 mm/yr across the *entire* gyre-strength range 0.0→3.0. This settles the exact doubt this entry
previously raised as untested ("it was never established that per-cell SST anomalies propagate to
land climate at all in this model's simplified atmosphere") — **they do not**, at least not through
this pathway.
**So the real missing mechanism is SST→adjacent-land climate coupling, not the ocean cooling itself.**
That is a materially different (and better-defined) target than "build upwelling physics", which is
what this entry used to recommend and which would now be redundant work.
`coastal_upwelling_fog_strength` remains a diagnostic proxy gate and remains the only thing actually
affecting Atacama's precipitation — see A1.

### D4. 🟢 Ocean CO2 exchange uses instantaneous, not time-averaged, wind — fixed, verified 2026-08-01
**Was**: `carbon_cycle.py`'s `ocean_co2_flux` computed piston velocity as `k ∝ wind_speed²`
(Wanninkhof 1992), a formula calibrated for *time-averaged* wind but fed the instantaneous per-step
value. Added wind variance (from the jet-stream meander/blocking mechanism) raised mean(k) via
Jensen's inequality, artificially speeding convergence toward the ocean-atmosphere CO2
quasi-equilibrium (a test bound already had to be widened 30→40 ppm/50yr to accommodate this).
**Fix** (already shipped, commit `c1e28da` "Time-average wind speed feeding ocean CO2 piston
velocity" — this document's 🔴 status was stale): `simulation_state.py` carries `wind_speed_avg`,
a rolling EMA over `PlanetParams.co2_wind_averaging_days` (default 30d) maintained each step in
`simulate.py`; `carbon_cycle_step` now feeds that to `ocean_co2_flux` instead of the instantaneous
value (falls back to instantaneous only when `wind_speed_avg is None`, e.g. very first step).
**Verified 2026-08-01**: re-ran `testing/test_conservation.py::test_co2_budget_near_steady_state`
(50yr, no anthropogenic emissions) — passes, drift is +16.1 ppm from 280 ppm preindustrial start,
well inside the tightened ±25 ppm bound (down from the pre-fix +33 ppm that had forced the bound to
be widened to 40 in the first place). No further action needed.

### D5. 🟢 2D barotropic gyre solver — calibrated and enabled 2026-08-02; found a real sign bug in the mechanism it was meant to replace
**Was**: `ocean.compute_gyre_currents` (a real streamfunction solve from wind-stress curl, reusing
the FFT Poisson machinery in `atmosphere.py`) existed but was gated behind
`ocean_gyre_strength=0.0`, contributing nothing to climate — western-boundary-current warming came
only from the older `land_west` heuristic in `calculate_ocean_heat_transport`.

**Shipped**: `PlanetParams.ocean_gyre_strength` **0.0 → 1.0**, plus a **sign-bug fix in
`land_west`** (see below). Both of the docstring's own stated pre-conditions were checked first and
both came back clean:
- *Structure, not noise*: the gyre-induced SST anomaly's grid-scale residual is **0.10** of its total
  standard deviation (≥1.0 would mean grid-scale noise), and it reproduces the correct gyre dipole
  everywhere checked — western boundaries warm (Gulf Stream +0.38 K, Kuroshio +0.14 K), eastern
  boundaries cool (Benguela −0.54 K, Canary −0.36 K, Humboldt −0.25 K, California −0.22 K).
- *Guard-rails*: full suite green; `reference_error_score` 0.3226 → 0.3209 (gyre alone), 0.3195 with
  the `land_west` fix.

**The improvement is physically located, not a global-cooling artifact** — decomposing the score
shows the precipitation half is flat (0.2811 → 0.2817) and essentially all the gain is temperature,
concentrated in the two ocean-dominated SH bands (0–10S 0.048 → 0.036, 40–50S 0.557 → 0.524) while
the large NH mid-latitude biases (land/AMOC-driven, D2) barely move. Returns saturate near 1.0, and
since the solve has no natural physical amplitude (clipped ±0.5 m/s), 1.0 — the unscaled solve — was
preferred over chasing a metric that can't distinguish "more correct" from "cools an already-too-warm
model" (values to 3.0 keep improving it marginally).

**Real bug found in `land_west` while answering "can the gyre replace it?"**: `western_enhancement`
is a 1.0–1.5× multiplier that was applied to `T_adjustment` unconditionally — but `T_adjustment` is a
**signed** quantity, and at real WBC cells it is predominantly *negative* (measured on the bundled
64×128 DEM: **73.6% of the 159 WBC cells negative**, mean −0.0138 K). Multiplying a negative by 1.5
makes it more negative, so the heuristic was **amplifying cooling at exactly the Gulf-Stream/Kuroshio
cells it was written to warm**. Two independent lines of evidence agreed: the sign census, and the
fact that simply deleting the multiplier *improved* both `reference_error_score` and Gulf-Stream SST.
Fixed by amplifying only the positive/warming component. Full matrix on the tracked benchmark:

| `land_west` | gyre | refErr | Gulf Stream ΔT |
|---|---|---|---|
| orig (shipped) | 0.0 | 0.3226 | +0.000 |
| orig | 1.0 | 0.3209 | +0.377 |
| off | 1.0 | **0.3186** | +0.686 |
| **fixed (shipped)** | **1.0** | 0.3195 | **+0.851** |

Deleting the multiplier outright scores marginally best on refErr, but discards a real
physically-motivated mechanism to do it; the sign fix is strictly better than the old behaviour on
*both* the aggregate score and the mechanism's own intended regional signal, so it was preferred.
**Real-terrain confirmation** (512×1024, 2yr MONTHLY, seasonally balanced): all named boxes flat
except **US Midwest 705 → 747 mm/yr** — an unlooked-for gain on A3's chronically-short box.

**Important negative result, and it answers a question D3 explicitly raised**: the eastern-boundary
cooling above is exactly the physics D3 lists as entirely missing — but it does **not** propagate to
adjacent land precipitation. Atacama precipitation is unchanged (146 mm/yr on the benchmark) at
*every* gyre strength from 0.0 to 3.0. D3's own doubt ("it was never established that per-cell SST
anomalies propagate to land climate at all in this model's simplified atmosphere") is now settled:
**they do not, at least not through this pathway.** Enabling gyres is therefore not a route to fixing
Atacama (A1's remaining desert residual) — that still needs a mechanism that couples SST to land
precipitation, which is the actual missing link, not the upwelling itself.

---

## E. Land hydrology & cryosphere

### E1. ⬛/🟡 No lateral surface water by default; channel-capacity gap when enabled
`PlanetParams.enable_surface_hydrology` defaults `False`. When investigated and calibrated
(2026-07-27), three real bugs were found and fixed (salinity never consumed river runoff; the
runoff trigger checked the chronically-floored surface soil bucket instead of the deep layer;
standing water had no evaporative sink) — but a **fourth, deeper structural limitation** remains:
`hydrology.py`'s D8 router has **no channel capacity or flow-velocity concept at all**. A
continent-scale basin (Amazon/Congo-scale) funneling its full real-world discharge into one grid
cell with no lateral spreading and no capacity limit can produce area-averaged depths in the
hundreds of thousands of mm (measured: one cell hit 611m, growing linearly, not asymptoting, over
a 10yr continuation) — evaporation at a few mm/day is utterly negligible against that. A
`surface_water_cap_mm` (50m) hard ceiling was added as a **safety valve, not a fix** — it bounds the
absurdity but doesn't make large-basin behavior realistic. **This is why the feature stays off by
default**, not because the routing itself is wrong.
**Recommended next step** (real scope, not a tuning pass): actual channel hydraulics — a
flow-accumulation-weighted capacity/velocity term, so discharge that exceeds a cell's physical
channel capacity spills laterally instead of pooling to unbounded depth.

### E2. 🟡 Land ice: mass balance/flow exist, but with four named real gaps, default off
`PlanetParams.enable_land_ice_dynamics` (default `False`) adds a real mass-balance/thickness/flow
mechanism (real-terrain tested, numerically stable, spatially selective) plus a `sea_level_change_m`
diagnostic. Four gaps, all documented as deliberate deferrals rather than oversights:
1. **Flow doesn't follow terrain slope** — diffuses thickness only, not ice-surface elevation,
   because elevation has no single canonical meters conversion (same root gap as C3's elevation
   hardcoding).
2. **No albedo coupling** — `land_ice_thickness` doesn't feed surface albedo the way the existing
   `ice_sheet_age` EF-threshold mechanism does.
3. **No calving→salinity coupling** — ice lost at a coastline is discarded, not credited as ocean
   freshwater input (unlike hydrology's `runoff`, which does reach `evolve_salinity`).
4. **No mask/coastline feedback** — `sea_level_change_m` is computed correctly but doesn't shift
   `masks.get_masks`'s land/sea split (that function has ~54 call sites across 22 files — judged too
   large a change to thread through safely in one session).
Also: `ice_flow_diffusivity` (2.0e-3 default) is **uncalibrated** — a seeded 2000m Antarctic-scale
test reservoir lost ~50% of its volume in 10 simulated years via flow-driven marginal loss, almost
certainly too fast for a stable ice sheet.

### E3. 🟡 Soil-moisture bistability (structural, partially mitigated)
The single-layer soil bucket is "genuinely bistable" by its own in-code documentation — no stable
middle ground, it either saturates near 1.0 or collapses to its 0.05 floor. A 2-layer bucket
(`soil_deep_gain_rate=0.0005`, enabled since 2026-07-26) gives desert-vs-continental differentiation
in the deep reservoir for the first time (0.05–0.25 desert vs 0.30–0.44 continental interior), but
this fixes the soil moisture *state*, not the underlying bistability of the *surface* layer, which
still pins near its floor across most real terrain at 45–65°N regardless of desert/continental
distinction. `testing/test_climate_drift.py`'s guards catch regressions here but don't fully expose
the real-terrain-scale saturation (`docs/FINDINGS_SUMMARY.md`'s own open-issues list flags this).

---

## F. Clouds & climate sensitivity

### F1. 🟡 Global cloud fraction below observations; prognostic cloud water net-negative
Cloud fraction is re-diagnosed from RH every step rather than carried with memory. A real
prognostic cloud-water budget (`cloud_water_feedback` param, condensation/precipitation/evaporation
ODE, forward-Euler-runaway bug fixed) was built and calibrated — but **any nonzero weight worsens
the model's already-known cloud-fraction-too-low bias**, so it correctly ships at `0.0` (inert).
This is one of several items where the "fix" was real, tested, and still net-negative for the
actual goal — same pattern as A2's `precip_raw_shape_weight`.

### F2. 🟢 Climate sensitivity: model ECS≈TCR, both below real Earth's ~3°C — understood, not a bug per se
Measured **ECS = 1.77 ± 0.06 K** post-regression-fix (400yr fixed-CO2 pair, 60×120), equilibrated by
~yr 50–100 — i.e. ECS and TCR are nearly equal in this model, because the deep ocean is decoupled at
τ_eff ≈ 2219 yr (D1): it neither restrains nor amplifies the surface response within a run. Real
Earth's TCR (~1.8K) and ECS (~3°C) differ because deep-ocean heat uptake progressively unmasks more
warming over centuries — this model's deep ocean does eventually warm (D1), just via the wrong
mechanism (local relaxation, not overturning), so it doesn't produce the same TCR/ECS gap. **Not
necessarily a bug to fix in isolation** — it's a direct, logical consequence of D1, and fixing D1
properly (real overturning) would be the correct way to also fix this, not a separate lever.

---

## G. Carbon cycle

### G1. 🟡 Toy permafrost/wetland flux constants deliberately below physical magnitude
Permafrost ppm-per-kgC and wetland CH4 ppb conversions are intentionally conservative (orders of
magnitude below real values) from early calibration. Since CH4 now has a baseline-balancing natural
source (closing an earlier equilibrium gap), these could likely be raised toward realistic
magnitudes with the optimizer verifying stability — flagged as a real, tractable next step in
ROADMAP, not yet attempted.

### G2. ⬛ Single well-mixed ocean carbon reservoir (no 2-box solubility pump)
A single ocean CO2 reservoir exists; real century-scale carbon drawdown depends on a
surface/deep split with a solubility pump. Absent entirely. Would make ECS-adjacent experiments
more meaningful once D1 (deep-ocean overturning) is also addressed, since the two share the same
underlying "no vertical/lateral ocean structure below the mixed layer" gap.

### G3. 🟢 `enable_carbon_cycle=False` also disables greenhouse forcing — a real trap, not a bug
Confirmed: `simulate.py` gates `co2_temp_offset` on the same flag used to disable reservoir
evolution. Anyone using this flag expecting "hold concentrations fixed, keep radiative physics
otherwise intact" is silently getting a different experiment than intended. Not itself inaccurate
physics, but a documented footgun worth fixing at the API level (split into two flags) if it keeps
tripping up experiments.

---

## H. Missing systems entirely (explicit call-outs, per the request)

These are not "inaccurate" in the sense of producing a wrong number — no code path exists at all.

1. **Diurnal cycle** (B4) — no time-of-day temperature variation anywhere.
2. **True channel hydraulics** (E1) — rivers/lakes exist but have no capacity/velocity concept.
3. **Sea-level ↔ land/sea mask feedback** (E2.4) — computed sea-level change never moves the
   coastline.
4. **Ice-albedo coupling for dynamic ice sheets** (E2.2) and **calving→ocean freshwater** (E2.3).
5. **Tidally-locked / non-Earth orbital regime support** — no substellar-point insolation model;
   `PlanetParams` has obliquity/eccentricity wired through insolation, but locked rotation (day
   length → ∞) is unimplemented despite the temperature LUT machinery reportedly mostly supporting
   it structurally.
6. **Non-water condensable cycles** (CO2 snow/frost on Mars, CH4 on Titan-like bodies) — latent
   heat/precip constants are partially parameterized but no alternate condensable cycle exists as a
   real code path.
7. **Random planet generator** (`PlanetParams.random(seed, archetype=...)`) — doesn't exist; only
   `EARTH` and `MARS` singletons.
8. **Scenario/event system** — no way to trigger a volcanic eruption, CO2 ramp, asteroid impact, or
   terrain edit at runtime, despite the aerosol/volcanic-forcing field and `masks.invalidate()`
   plumbing already existing to support exactly this.
9. **Milankovitch scenario runner** — obliquity/eccentricity are wired through insolation but no
   scripted orbital-parameter sweep exists to demonstrate spontaneous glaciation.
10. **Gridded (ERA5/CRU) map-correlation validation** — current validation is 9 named boxes +
    6 zonal reference bands + a handful of Köppen area fractions. No spatial correlation/RMSE
    against an actual gridded reference product exists, meaning *regional* errors (a desert core,
    a continental interior) that a zonal mean would average away can silently persist — precisely
    the class of error most of section A describes. This is arguably the highest-leverage missing
    piece of *infrastructure* (not physics) in the whole project: nearly every session above had to
    re-derive its own named-box measurement from scratch because no single scored map-comparison
    exists yet.
11. **Real SST-coupled ocean-current climate feedback** — ocean current *fields* are computed
    (climatological gyres + Ekman transport) but only for GUI visualization; they don't feed back
    into temperature, salinity, or precipitation (D3, D5 describe the two real physics mechanisms
    that exist instead as stand-ins).
12. **Prognostic AMOC responding to temperature** (only salinity is coupled today, D2).
13. **GUI test coverage** — `main.py` (~116KB) has no automated test coverage at all; not a physics
    inaccuracy, but a real risk multiplier for undetected regressions in the biome/temperature/wind
    view layers that this whole audit relies on for visual comparison against the Köppen reference.

---

## I. Cross-cutting process notes (read before starting new physics work here)

1. **A5 was the single biggest recurring wall** (the chronic raw-production deficit forced a
   ~5.5x rescale multiplier; the regime fix reduced it to 2.06x). At least eight independently-motivated fixes across different
   subsystems (moisture transport, soil bucket, evapotranspiration, wind convergence, raw-shape
   redistribution, flat desert suppression, ITCZ-only removal-cap raise, subsidence-signal regime
   gating) have all been defeated by this same structural cause. **Any new session on desert
   dryness, Aw/Af ratio, or continental shortfall should read A5 first** — a redistribution-only
   mechanism cannot fix any of these; raw production had to rise and the rescale architecture had
   to become regime-aware. The successful follow-up did both and corrected
   `subsidence_suppression`'s regime accuracy; use that as the new baseline rather than reviving
   the failed flat/redistribution-only attempts. **It was not a free win**: the same wet/dry
   contrast-sharpening that fixed A1 (deserts) directly *worsened* A2 (savanna transition) —
   confirmed by measurement, not theoretical (Aw+Am land share dropped from 8.16% to 2.25%). That
   specific regression was subsequently fixed via `itcz_seasonal_target_response` 1.0→1.7 (see A2),
   but the general lesson stands: a fix in section A should be measured against *all* of A1–A4, not
   only its own target box.
7. **Re-test previously-rejected parameters after any major upstream change — the constraint may
   have moved, not the parameter** (2026-08-01, the session's highest-leverage-per-effort win).
   `itcz_seasonal_target_response` had been swept to 2.0 on 2026-07-31 and deliberately capped at
   1.0, with a documented reason: past 1.0, `arid_pct` climbed *while desert boxes drifted wetter*
   (aridity manufactured at the tropical margin instead of in real deserts). After A5 landed, an
   identical sweep showed that failure mode simply gone — deserts now stay flat-to-drier and
   `arid_pct` moves *toward* Earth's target instead of past it, because A5 had raised `arid_pct`'s
   starting point from ~12% to ~17.7%. The same measurement read as "overshoot" before and
   "convergence" after. Nothing about the parameter changed; the regime it operated in did.
   **Practical rule**: after a structural fix, re-run the sweeps that earlier sessions used to
   justify a cap — the docstring recording *why* a value was capped is exactly the list of
   hypotheses worth re-testing, and re-running a cached sweep is far cheaper than inventing a new
   mechanism. (Corollary: this is why those "why we capped it here" notes are worth writing.)
8. **Validate the *metric* before spending sessions optimizing against it** (2026-08-02 — the
   single highest-leverage finding of this whole audit's history). Köppen land-share percentages were
   computed as plain cell counts on an equirectangular grid, with no cos(latitude) area weighting, in
   *both* tracked measurement sites. That inflated polar land to 38.3% (Earth ~16.6%) and, since the
   shares are a closed budget, deflated every other group — which is the entire origin of the
   long-standing "tropics are 2–4x under-represented" belief that multiple sessions tried to fix with
   precipitation physics. Correcting the weighting moved mean absolute error against Earth from
   **8.7pp to 2.2pp with zero physics change**, and revealed that the tropical band (22.0% vs Earth's
   19.0%) was never too small at all. It also showed a same-day calibration had stopped at the wrong
   value because the biased metric mis-reported the optimum. **Practical rule**: when a metric has
   driven several sessions of work without the gap closing, audit the metric's own construction —
   especially any per-cell statistic on a lat/lon grid, where a missing `cos(lat)` is invisible,
   plausible-looking, and systematically wrong in a fixed direction. Cheap to check, and it can
   invalidate more accumulated conclusions than any physics fix.
9. **At savanna/tropical latitudes the rescale *target* is the lever, not raw production**
   (2026-08-01, measured): a raw-production-side change there is absorbed almost entirely by a
   compensating change in the moisture-budget's synthetic fill — an attempted seasonal modulation of
   `_raw_conversion_gain` moved final precipitation by ~0.05% and reclassified zero cells, while a
   target-side change of comparable intent moved Af from 11.05% to 6.07%. Check which side of the
   rescale a proposed mechanism sits on before building it.
2. **Several "fixes" are real, tested, and shipped — but deliberately inert (default off/0.0)**
   because they measured net-negative, no-effect, or an unresolved trade-off for their intended
   purpose: `moisture_advection_scale`, `precip_raw_shape_weight`, `cloud_water_feedback`,
   `abyssal_overturning_coeff`, `pgf_continentality_amp`, `enable_surface_hydrology`,
   `enable_land_ice_dynamics`, `moisture_budget_tropical_cap_boost` (no measurable effect at any
   strength). The old whole-signal version of `subsidence_divergence_regime_gate` belonged in this
   list; the shipped zonal-background implementation is materially different and defaults to 1.0.
   Check current defaults in `planet_params.py` before assuming any of these is active — several
   are excellent starting infrastructure for a future session with a different strategy, not
   oversights.
3. **Testing-methodology lesson, repeated across multiple sessions**: a single-day or single-month
   snapshot has been mistaken for climatology at least twice, producing false "regression" reports
   later retracted. Always use a 12-month (or full seasonal cycle) average, and record `day_of_year`
   with any long-run sample to rule out the calendar-aliasing artifact (C2). A third instance
   (2026-08-01): when checking whether a fix helped or hurt a specific named box, compare against a
   **same-save, same-code, freshly-bisected** pre-fix measurement, not a number carried over from an
   earlier session's write-up — Central Europe was briefly, incorrectly flagged as "regressed by A5"
   in this document because its cited pre-fix baseline (639) came from an unrelated, non-comparable
   earlier measurement; a clean bisect against the immediately-prior commit on the identical save
   showed A5 actually improved it (463→502). `git worktree add --detach <commit>` against the same
   `saves/*.pkl` is cheap and removes this whole class of false positive/negative.
4. **Real-terrain vs synthetic-fixture tension**: several precip fixes were tuned against a
   synthetic test fixture's spatial uniformity (e.g. the orographic test's meridional-only elevation
   ramp) and had to be deliberately gated (row-heterogeneity-based blending) so they wouldn't be
   diluted to protect an edge case that never occurs on real terrain. When calibrating anything in
   section A, check both the fast-suite synthetic fixtures *and* a real-terrain run — either alone
   has hidden a real regression before.
5. **This document vs. other project docs**: `known-physics-gaps.md` (memory) and
   `PLAN_PHYSICS_FIXES.md` (repo) remain the detailed, chronological session logs — this file is the
   rolled-up index and should be refreshed (not appended-to indefinitely) as major items close or
   new ones are found, mirroring how `test-suite-status.md` is maintained as a living snapshot.
6. **A large physics recalibration's headline numbers being real doesn't mean it's clean — always run
   the full test suite before trusting a "fixed" claim, including this document's own** (2026-08-01):
   the A5 regime-architecture fix landed with genuinely reproducible, valuable real-terrain numbers
   *and* 3 failing tests. One was benign (golden-state fixture needing regeneration after a deliberate
   change, exactly as its own docstring anticipates). The other two were real: a new, large-effect
   mechanism (`_raw_conversion_gain`, up to 5.5×) shipped completely unconditional — no `PlanetParams`
   gate at all — breaking this project's own consistent convention (every other lever in this
   document ships behind a 0.0/False-default no-op specifically so it can be validated and reverted
   in isolation). Bisecting against recent commits (not just reasoning about the diff) was what
   actually pinned the exact cause in minutes rather than hours. **New large-effect mechanisms should
   still ship gated even when the calibrated value is believed correct** — the gate itself is what
   makes an interaction bug like this one cheaply diagnosable via ablation instead of a bisect.
