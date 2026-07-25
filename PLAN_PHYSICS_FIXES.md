# Plans: A9 spherical metric, and the US Midwest divergence bug

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
  analytically tested — that is the deliverable. Enabling it requires exactly the recalibration
  this plan anticipated (`u_scale`/`v_scale`, `target_mean_mm_day`), which is its own piece of
  work and should not be bundled here. **Follow-up**: recalibrate with the metric on, then
  re-evaluate the default.

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

- **Not flipped.** The default remains 48.0. Changing it materially alters the precipitation
  field, so it needs golden-state regeneration and an explicit decision — not something to do
  unilaterally at the end of a session. Everything needed to make that call is above.
- **Scope note**: the split currently applies only to `generate_wind_field` (the diagnostic wind
  used in MONTHLY/ANNUAL). `evolve_wind`'s own 3-cell relaxation has a separate `w_mid` left
  untouched, so DAILY/WEEKLY are unaffected. If the sweep favours a change, that path needs the
  same treatment for consistency.

---

> Created 2026-07-25, after the `aa4b127` coupling regression was fixed.
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
