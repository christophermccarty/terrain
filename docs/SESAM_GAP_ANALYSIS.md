# SESAM gap analysis — adopting the published statistical–dynamical atmosphere design

Status: **analysis only, no code changes**. Adopted direction: "Option A" from the
2026-08-16 map-gap review — replace the hand-derived Phase-2/3 closure series with a
bounded port of the published, validated SESAM design (the atmosphere of CLIMBER-X),
under an explicit calibration-window policy defined in §6 below.

This document is the term-by-term gap analysis promised by that decision. It maps
every SESAM mechanism onto PlanetSim's current modules, marks what exists, what is
partially present, what is missing, and what PlanetSim currently does in a way that
is *anti-aligned* with the published design. It ends with the staged adoption plan
and its admission/stop gates.

## Sources

- **Published design**: Willeit, Ganopolski, Robinson & Edwards, *The Earth system
  model CLIMBER-X v1.0 – Part 1*, Geosci. Model Dev. 15, 5905–5948 (2022),
  <https://doi.org/10.5194/gmd-15-5905-2022> (CC-BY 4.0). Equations below are cited
  as (A1)–(A117), matching the paper's appendix numbering.
- **Reference implementation**: <https://github.com/cxesmc/climber-x> (GPL-3.0,
  Fortran). See §5 for the licensing constraint on its use.
- **PlanetSim inventory**: read directly from the current code (file:line citations
  below are current as of 2026-08-16).
- **Measured motivation**: `saves/test.npz` (512×1024, 16.8 yr) scored against the
  bundled Köppen-Geiger reference via `koppen_reference.score_koppen_map` /
  `score_temperature_thresholds`.

## 1. What the measurement says the gaps are

Headline skill of the saved state: Köppen group 0.706 (κ 0.625), class 0.436;
29.4% of scored land is in the wrong group. Global means are fine (T 15.0 °C,
P 2.78 mm/day); cloud fraction 0.134 vs Earth ~0.67 is the known outlier. The
wrong-group area decomposes as (share of all scored land):

| Error | Share | Reads as |
|---|---|---|
| C→B | 5.3% | temperate classified arid |
| B→C | 4.1% | desert margins too wet |
| A→B | 3.8% | tropical land too dry |
| B→D | 3.7% | arid land too cold in winter (Patagonia) |
| C→D | 3.5% | maritime temperate → continental (Europe) |
| B→A | 2.5% | deserts too warm in winter |
| D→E | 2.4% | continental summers too cold |

Two geographic failures dominate the *systematic* error, and they are one
mechanism: **the model has no advective maritime heat supply to land**.

- Band −60:−40 (Patagonia, Tasmania, NZ, sub-Antarctic): group accuracy 0.03–0.04;
  model coldest-month mean **−10.6 °C** against a real maritime value near +3 °C.
  The model labels 89% of the band Dfa/Dfb; the reference is Cfb/Cfc/BSk/ET.
- Europe (44–58°N, −5–25°E): model 69% Dfb against a reference 82% Cfb; coldest
  month −5.9 °C; Central Europe region group accuracy 0.05.

PlanetSim's shipped continentality mechanisms
(`land_transport_maritime_decay`, `land_seasonal_amplitude_maritime`) are
row-mean-preserving redistributions of a *latitude-shaped* forcing. In rows where
essentially all land is maritime (40–60°S, coastal Norway/Alaska) there is no
continental anomaly to redistribute, so they cannot fix these regions
(`docs/ACCURACY_AUDIT.md` C1b-2026-08-05 calls this the "geometric limit"). The
Phase-3 rejection ledger (2026-08-15/16, `docs/REMAINING_WORK_PLAN.md`) reached the
same structural conclusion from the inside: the missing piece is a conservative
land–air/maritime energy exchange, and the single `T_air` field cannot serve as
both 2 m air and transport column.

The remaining large errors are the Phase-1 precipitation-pathway set (East China /
S Japan monsoon under-production, Central Europe over-production, Atacama coastal
desert), whose dossier already concluded they need real vertical structure and
monsoon circulations rather than more scalar tuning.

## 2. The SESAM design in one view

SESAM is "2.5-D": **four 2-D prognostic fields**, all vertical structure
diagnostic:

| Variable | Meaning | Governed by |
|---|---|---|
| `Ta` | near-surface air temperature | column energy eq. (A40) |
| `qa` | near-surface specific humidity | column water eq. (A42) |
| `K` | vertically integrated eddy kinetic energy | EKE eq. (A52) |
| dust | (aerosol; out of scope for us) | — |

From these it *derives*: 3-D `T(z)`, `r(z)`, `q(z)`, `θ(z)` (A1 profiles), SLP
(zonal-mean part from cell-circulation physics + azonal thermal part from skin-T
anomaly + Charney–Eliassen topographic waves) (A2), full 3-D wind `u(z) = ug + ua`
(geostrophic + thermal wind + Taylor-model ageostrophic + katabatic), clouds
(A6), precipitation (A44), and all radiation (A7/A8).

The architectural properties that matter for us:

1. **Nothing is prescribed by latitude.** ITCZ position `= c2·(T_NH − T_SH)` (A32),
   Hadley width `= c3·(T_trp − c4)` (A33), cell strengths ∝ zonal-mean temperature
   gradients at fixed latitudes (A34), storminess ∝ Eady baroclinicity via EKE
   (A53). Every zonal structure PlanetSim hardcodes, SESAM derives.
2. **Precipitation is a water-budget residual**, not a target: rain is whatever
   converges past 95% near-surface RH plus a land turnover term `Qq·ra/τp` (A44).
3. **Maritime moderation is structural**: the azonal thermal SLP component (A37)
   turns ocean–land temperature contrast into onshore flow; the full 3-D wind
   advects `θ` and `q`; per-surface-type fluxes share one atmospheric column, so
   ocean heat reaches land air by advection and exchange, not by a redistribution
   knob.
4. **EKE is prognostic**: baroclinic production, drag dissipation `∝ K^1.5`,
   diffusivities `AT = c5·√K`, `Aq = c6·K` (A50–51). Storm tracks emerge and
   respond to the model's own baroclinicity.
5. **Cost**: 5°×5° grid, 1-day coupling step, ~2 h internal atmosphere substep,
   ~10⁴ yr/day on 16 cores. This is the same complexity tier PlanetSim already
   occupies.

## 3. Term-by-term mapping

Legend: ✅ present and equivalent · ◐ partially present · ❌ missing ·
⚠️ present-but-anti-aligned (does the opposite job or prescribes what SESAM derives)

### 3.1 Vertical structure (A1)

| SESAM term | PlanetSim status | Where |
|---|---|---|
| `p(z) = p0·exp(−z/Ha)` diagnostic | ◐ pressure-column experiments have pressure levels, but the supported path has no analytic profile | `pressure_column.py` (gated) |
| Piecewise lapse: stability-dependent near-surface (1500 m), linear-ramp mid, isothermal stratosphere (A6–A9) | ❌ | — |
| Tropopause from stratospheric radiative equilibrium (A10–A11) | ❌ | — |
| RH profile: PBL constant → exponential decay scale height `Hr` modulated by `w700` → constant aloft (A13–A14) | ❌ (humidity exists only at surface; gated mid/upper levels) | `planet_params.py` gates |
| `q(z) = r(z)·qsat(T(z),p(z))` (A15) | ◐ `qsat` used everywhere; no profile | — |

### 3.2 Dynamics (A2)

| SESAM term | PlanetSim status | Where |
|---|---|---|
| SLP = zonal + azonal construction | ❌ SLP is not a state; pressure anomalies are synthesized inside the wind generator | `atmosphere.py:1797–1845` |
| Zonal SLP from cell physics: `v̄a(ϕ) = −Ci·ΔTij·Fz(ϕ)·sinφ`, cells driven by zonal-mean T gradients at π/6, π/3, π/2, ITCZ from inter-hemispheric T contrast, Hadley width from tropical T (A30–A35) | ⚠️ **prescribed 3-cell targets instead**: fixed centers/widths (14°/48°/74°) and U/V target velocities, with zonal-mean relaxation toward them | `atmosphere.py:131–146`, `:2554–2607`, `:1889–1998` |
| Azonal thermal SLP `∝ T_sl*` (A37) | ❌ (the missing monsoon/maritime driver) | — |
| Charney–Eliassen topographic Rossby waves per latitude belt (A38–A39) | ❌ (terrain PGF gate exists experimentally) | `wind_terrain_pgf_scale` |
| Geostrophic + thermal wind (A17–A18) | ◐ geostrophic from synthetic pressure + thermal-wind jet mixing; Taylor ageostrophic + katabatic absent | `atmosphere.py:2451–2531` |
| Taylor surface wind + cross-isobar angle solved from drag (A21–A25) | ◐ fixed 3% cross-isobar factor | `atmosphere.py:2514–2520` |
| Katabatic wind (A26–A27) | ❌ | — |

### 3.3 Thermodynamics (A3)

| SESAM term | PlanetSim status | Where |
|---|---|---|
| Prognostic column energy `QT`; `∂QT/∂t = −∇·∫ρuθ dz + (SW+LW+Le·Pw+Ls·Ps+SH)/cv` (A40) | ⚠️ PlanetSim carries `T_sst` (not advected) and `T_air` (advected + diffused + relaxed to surface) separately; neither is a column energy | `simulate.py:4682–4875` |
| Advection of 3-D `θ` by the full 3-D wind | ❌ (T_air advected as a 2-D passive field) | `simulate.py:4701–4724` |
| Macroturbulent diffusion of heat with `AT ∝ √K` (A46–A50) | ◐ Laplacian diffusion with fixed `thermal_diffusivity=0.04` + fixed-window eddy flux `eddy_heat_flux_coeff=0.006`, 20–70° | `simulate.py:4727–4748`, `:5803–5832` |
| Sources: SW, LW, latent heat from P, sensible heat (A40) | ◐ all present term-by-term but not as a closed column budget | `simulate.py:5204–5499` |
| `T2m = (Ta + T⋆)/2`; per-macro-surface-type fluxes share one column | ❌ single surface T per cell; ocean-only equal-and-opposite exchange; land–air exchange is a one-way relaxation | `simulate.py:5731–5746`, `:4869–4875` |

### 3.4 Hydrology (A4)

| SESAM term | PlanetSim status | Where |
|---|---|---|
| Prognostic column water `∂Qq/∂t = −∇·∫ρuq dz + E − P` (A42) | ✅ **already built** (finite-volume, spherical, CFL-substepped) but default-off and was rejected at screen because nothing replaced what the row target was hiding | `column_water.py`, gate `enable_prognostic_column_water` |
| Advection of 3-D `q` by the full wind | ◐ 2-D humidity advection + diffusion; column-water transport when gated | `atmosphere.py:3959–3981`, `:3862–3958` |
| Rain = moisture convergence past 95% RH + land turnover `Qq·ra/τp` (A44) | ⚠️ **row-target allocator instead**: every latitude row is rescaled toward a prescribed `target_row_mm_day` profile (10 Earth breakpoints, smoothed) — the mechanism `EXTERNAL_PRIOR_ART.md` item 4 and the Phase-1 dossier both identify as the wall | `atmosphere.py:176–233`, `:5410–5478`, `:5736–5807` |
| Slope convergence `Cslope ∝ √K·|∇zs|·ρ0·qa` (A45) | ◐ orographic uplift exists (Smith–Barstad gated experiment + hand-rolled kernels) | `orographic_linear.py`, `atmosphere.py:4414–4495` |

### 3.5 Synoptic processes (A5)

| SESAM term | PlanetSim status | Where |
|---|---|---|
| Prognostic EKE `∂K/∂t = −∇·(uK) + ∇·(AT∇K) + PK − DK` (A52) | ❌ no EKE state | — |
| Production ∝ Eady growth rate from 850–500 hPa shear/stability (A53–A54) | ❌ (requires the A1 vertical profiles to exist) | — |
| Dissipation `(c3 + c4·CD)·K^1.5` (A55) | ❌ | — |
| Synoptic wind `Usyn ∝ √K` entering fluxes and cloud (A56–A58) | ⚠️ fixed storm-track window (48°±15°) + prescribed wave pressure anomalies | `atmosphere.py:3105`, `:2413–2445` |

### 3.6 Clouds & radiation (A6–A8)

| SESAM term | PlanetSim status | Where |
|---|---|---|
| Single effective cloud layer from RH + vertical velocity + inversion-strength low cloud (A61–A66) | ⚠️ cloud fraction 0.134 vs ~0.67 observed; no inversion-low-cloud term (this is the documented Atacama/stratocumulus gap) | `atmosphere.py` cloud drivers; FINDINGS "Clouds and climate sensitivity" |
| Optical thickness from `fcld·Qq` and T (A68) | ❌ | — |
| SW: delta-Eddington 2-band with water-vapour/aerosol/ozone/cloud transmissions (A69–A105) | ❌ `Q(1−albedo)` with albedo by latitude/surface type | `simulate.py:5204–5232`, `temperature.py:286–329` |
| LW: 15-level two-stream with `Dwv·DCO2·DO3·Dcld` transmission products, pressure-weighted absorber masses, Etminan CO2-equivalence (A106–A117) | ⚠️ one-layer grey `ε(φ,cloud,WV)·σT⁴` inline; two-layer grey and humidity-dependent DLW exist only as rejected/gated experiments | `simulate.py:5281–5295`, `atmospheric_radiation.py` |

### 3.7 What SESAM does NOT have (and we keep ours)

Ocean (GOLDSTEIN 3-D frictional–geostrophic), sea ice (SISIM), land surface
(PALADYN) are separate CLIMBER-X components. PlanetSim's `ocean.py`, sea ice,
hydrology, land ice, and carbon cycle are out of scope for this adoption; the
coupling contract (per-surface-type fluxes, wind stress to ocean, runoff to
ocean/salinity) is preserved.

## 4. The gap, stated plainly

1. **The four prescribed scaffolds are the gap.** The 3-cell wind targets, the
   land transport trapezoids, the AMOC/ACC/ocean transport-warming profile, and
   the precipitation row-target allocator are all *latitude-shaped Earth answers
   baked in as forcing*. SESAM's value is not its constants; it is that the
   corresponding quantities are **derived from the model's own temperature
   contrasts and baroclinicity**. Every dead end in the Phase-1/2/3 ledgers is a
   place where one of these scaffolds fights a correction.
2. **Two missing states block everything else.** Without diagnostic vertical
   structure (A1) there is no Eady growth rate (A53), no thermal wind (A17–18),
   no LW levels (A8). Without prognostic EKE there is no storm-track dynamics, no
   `√K` synoptic wind, no physically-responsive diffusion.
3. **The single-`T_air` representation conflict** (Phase-3's final finding) is
   resolved *by design* in SESAM: `Ta` is a column-derived near-surface air
   temperature, `T2m = (Ta+T⋆)/2` serves fluxes, and surface temperatures live in
   per-surface-type models. We do not need to invent the split.
4. **Two required pieces already exist here**: the conservative column-water
   transport (`column_water.py`) is exactly the flux geometry (A42) needs, and the
   force-restore/Penman–Monteith land branch (`land_surface.py`) is PALADYN-tier
   surface physics. Their rejections measured the missing *companions* (no real
   circulation behind the water; no real air column above the land), not their
   own kernels.

## 5. Parameter values and licensing

- The paper's equation **constants live in Tables A1–A8**, which the HTML text
  renders as XLSX attachments under the article DOI. **P0 is complete
  (2026-08-16):** all 12 table files were fetched and transcribed into
  `sesam_reference.py` — a versioned, read-only parameter pack with per-entry
  units, equation references, and source-XLSX SHA-256 provenance, guarded by
  `testing/test_sesam_reference.py`. Entries whose units were ambiguous in the
  XLSX/HTML conversion carry an explicit `transcription_note` and must be
  checked against the paper PDF at the stage where they are first used
  (`sesam_reference.flagged_transcriptions()` lists them). Table A3
  (thermodynamics) publishes no parameter table; its constants are physical
  constants and the ~2 h internal substep, noted in the pack.
- CLIMBER-X source is **GPL-3.0 Fortran**. PlanetSim carries no LICENSE file and
  must not vendor or translate that code. Permitted use: the paper (CC-BY 4.0)
  equations, the published table values, and *reading* the Fortran namelist
  defaults to cross-check constants. All Python here is written from the paper.
  If any doubt arises, constants enter `planet_params.py` as named, documented
  parameters with the paper citation — which is this project's existing practice.

## 6. Calibration-window policy (the meta-decision)

The prior-art ledger's repeated failure mode is structural: a published mechanism
is screened *uncalibrated* against an incumbent with ~40 sessions of co-fitted
knobs, and the gates correctly reject the regression. To adopt a *design* rather
than a knob, the promotion rules need one bounded exception:

1. The SESAM branch develops behind `enable_sesam_atmosphere` (default False).
   The supported baseline and all its gates are untouched; no existing metric may
   regress on the default path.
2. When the branch is feature-complete (§7 stage P4), it gets **one bounded
   calibration window**: a documented sweep over only the SESAM paper's own
   constants (Table A values) — not legacy knobs — at 64×128. The judgement
   happens **after** the sweep, against the full standard gate set (Köppen
   group/class, temperature thresholds, CRU T/P, named regions, conservation).
3. If the calibrated branch still loses, it is rejected with measurements like
   any other candidate and stays default-off. The window is spent, not extended;
   a second window requires a new decision record.
4. The window never includes geography-specific exemptions or per-region
   multipliers — the discipline that kept the legacy path honest applies inside
   the window too.

## 7. Staged adoption plan

Each stage is default-off, pure-kernel-first, unit-tested (conservation, sign,
planted-violation), and gated per the repo's operating rules. Stages are ordered
so each is evaluable without the next.

- **P0 — Parameter pack** ✅ **complete (2026-08-16)**: Tables A1–A8 plus the
  main-text validation budgets and the GOLDSTEIN/SISIM appendix constants are
  transcribed in `sesam_reference.py` with provenance and per-entry equation
  references. Every equation in §3 now has named constants.
- **P1 — Diagnostic vertical structure** ✅ **complete (2026-08-16)**: A1
  kernels implemented in `sesam_vertical.py` as pure functions — height scale
  and reference pressure (A1-A4), piecewise lapse + stability-dependent
  near-surface layer with the published caps (A6-A9), analytic `T(z)` integral
  (A5), RH profile with `Hr` from tropical weight and `w700` (A13-A14), qsat
  ice/water partition (A15), potential temperature (A12), and the tropopause
  dynamical shape + tendency (A10-A11). The tropopause is a required input
  until P5 closes it from radiation; `w700` and the P2 coordinate are optional
  inputs with documented placeholders. Gated `enable_sesam_vertical_structure`
  (default False, **not wired into the supported path**), guarded by
  `testing/test_sesam_vertical.py` (15 tests incl. an analytic piecewise-lapse
  cross-check). Profile shapes are validated qualitatively against the paper's
  form; zero climate impact, verified by the unchanged routine suite.
- **P2 — SLP and wind reconstruction** (A2): zonal SLP from cell physics
  (A30–A35), azonal thermal SLP (A37), Charney–Eliassen term (A38), geostrophic +
  thermal wind + Taylor + katabatic assembly. Run *diagnostically alongside* the
  existing winds first; score SLP vs ERA patterns (the Phase-1 dossier's
  circulation diagnostics are the comparison harness). *Exit*: DJF/JJA SLP and
  surface-wind patterns beat the current prescribed-cell generator on the
  jet/Hadley scorecard before any coupling.
- **P3 — EKE and synoptic transport** (A5): prognostic K, Eady production from
  P1 profiles, drag dissipation, `AT/Aq` diffusivities, synoptic wind. *Exit*:
  storm-track placement responds to baroclinicity; macroturbulent heat/moisture
  fluxes replace the fixed-window eddy term in the branch.
- **P4 — Column energy and water closure** (A3/A4): prognostic `QT`, `Qq` (reuse
  `column_water.py` flux machinery), 95% RH + land turnover precipitation,
  per-surface-type fluxes with `T2m`. This is the stage that **bypasses the
  row-target allocator** inside the branch. *Exit*: water/energy conservation
  residuals at the `column_water.py` standard; raw global P in [0.5, 5] mm/day
  without any target.
- **P5 — Radiation upgrade** (A7/A8): only after P1 gives it real profiles.
  Two-band SW + 15-level two-stream LW. *Exit*: TOA fluxes vs the paper's
  validation figures; ECS moves off the structural ~1.8 K floor.
- **P6 — The calibration window** (§6), then the standard 128×256 five-year
  promotion checkpoint.

**Stop conditions** (per operating rule 3): a stage that fails its exit gate after
its *own* constants' bounded sweep is a conclusion about that mechanism, and the
branch pauses — no widening into legacy-knob sweeps, no per-region patches.

## 8. Risks and honest costs

- **Equator**: SESAM's geostrophic frame breaks down near the equator (paper's
  stated main limitation); its `|f|` floors (3e-5 / 1e-5 s⁻¹) and trade-regime
  treatment must be ported with the same care, or the tropical belt that currently
  scores 0.85+ regresses.
- **Resolution**: SESAM is validated at 5°×5°. PlanetSim runs 2.8° (64×128) and
  finer; constants tuned at 5° are not guaranteed to transfer. The calibration
  window exists for exactly this.
- **Speed**: SESAM's cost claim (~10⁴ yr/day, 16 cores, Fortran) maps to roughly
  interactive-friendly budgets in Python only if the per-step kernels stay
  vectorized; P1–P5 each need a runtime measurement against the current baseline
  before proceeding (the project already gates on this).
- **Validation trap**: the Köppen/CRU contract stays the judge — no redefining
  targets to fit the branch. The ExoPlaSim T21 reference remains the independent
  falsification instrument for circulation (docs/EXTERNAL_DYCORE_WORKFLOW.md).
- **The window discipline is the risk**: if the calibrated branch still loses,
  this whole effort lands as another documented rejection. That outcome is
  acceptable and is priced into the decision.

## 9. Deliberately not adopted

Diurnal cycle (SESAM has none), dust cycle, aerosol indirect effects, ozone
chemistry, GOLDSTEIN/SISIM/PALADYN internals, ice-sheet coupling (SICOPOLIS/Yelmo)
— all either out of scope or already covered by PlanetSim components. The
`koppenpasta`-style "external reference run" path (Option B) remains available
unchanged if this branch fails.
