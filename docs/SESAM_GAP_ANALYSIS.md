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
| SLP = zonal + azonal construction | ◐ diagnostic kernels exist (default-off, P2 first sub-deliverable); still not a state in the supported path | `sesam_dynamics.py` |
| Zonal SLP from cell physics: `v̄a(ϕ) = −Ci·ΔTij·Fz(ϕ)·sinφ`, cells driven by zonal-mean T gradients at π/6, π/3, π/2, ITCZ from inter-hemispheric T contrast, Hadley width from tropical T (A30–A35) | ⚠️ **prescribed 3-cell targets still in the supported path**: fixed centers/widths (14°/48°/74°) and U/V target velocities, with zonal-mean relaxation toward them; ◐ derived-cell kernels exist diagnostically | `atmosphere.py:131–146`, `:2554–2607`, `:1889–1998`; `sesam_dynamics.py` |
| Azonal thermal SLP `∝ T_sl*` (A37) | ◐ diagnostic kernel exists (default-off); still absent from the supported path (the missing monsoon/maritime driver) | `sesam_dynamics.py` |
| Charney–Eliassen topographic Rossby waves per latitude belt (A38–A39) | ◐ diagnostic kernel exists (default-off; terrain PGF gate exists experimentally) | `sesam_dynamics.py`, `wind_terrain_pgf_scale` |
| Geostrophic + thermal wind (A17–A18) | ◐ supported path: geostrophic from synthetic pressure + thermal-wind jet mixing; ◐ SESAM diagnostic kernels exist (default-off) | `atmosphere.py:2451–2531`; `sesam_wind.py` |
| Taylor surface wind + cross-isobar angle solved from drag (A21–A25) | ◐ supported path: fixed 3% cross-isobar factor; ◐ SESAM diagnostic kernels with the (A21) bisection solve exist (default-off) | `atmosphere.py:2514–2520`; `sesam_wind.py` |
| Katabatic wind (A26–A27) | ◐ diagnostic kernel exists (default-off) | `sesam_wind.py` |

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
| Prognostic EKE `∂K/∂t = −∇·(uK) + ∇·(AT∇K) + PK − DK` (A52) | ◐ production/dissipation/diffusivity/synoptic-wind kernels exist (default-off, local steady state); the (A52) advection/diffusion transport of K is stage P4 | `sesam_synoptic.py` |
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
  - **First sub-deliverable complete (2026-08-16): the SLP construction**
    (A28–A39) as pure functions in `sesam_dynamics.py` — (A31) cell coordinate,
    (A32) ITCZ position, (A33) Hadley width scale, (A34) cell temperature
    gradients, (A35) topography factor, (A30) mean overturning wind, (A29)
    zonal-SLP integral, (A37) azonal thermal SLP, (A38)–(A39) Charney–Eliassen
    topographic term, (A28)/(A36) assembly with mass restoration, plus the
    zonal-extrema scorecard. Gated `enable_sesam_dynamics` (default False,
    **not wired into the supported path**), guarded by
    `testing/test_sesam_dynamics.py` (23 tests: hand-equation checks, all-six-
    branch circulation signs, planted (A34) sign violation, mass conservation,
    machine-precision Charney–Eliassen cross-check, equator stability).
    `scripts/build_ncep_slp_reference.py` builds the NCEP/NCAR 1991–2020 SLP
    climatology (Pa units verified from the file attribute), and
    `scripts/diagnose_sesam_slp.py` runs the reconstruction diagnostically on
    a saved state for DJF/JJA and scores it against NCEP. The wind assembly
    (A16–A27) is the remaining sub-deliverable. Equation-semantics findings
    from this stage are recorded in §10 below.
  - **Second sub-deliverable complete (2026-08-16): the 3-D wind assembly**
    (A16)–(A27) as pure functions in `sesam_wind.py` — (A22)–(A23) drag
    coefficient, (A21) cross-isobar bisection solve, spherical SLP gradients,
    (A17)–(A18) surface geostrophic wind and thermal-wind shear, (A19)–(A20)
    ageostrophic PBL wind, the mass-conserving ageostrophic vertical profile,
    (A26)–(A27) katabatic wind, (A24)–(A25) Taylor surface wind, and the
    (A16) assembly with the 500 hPa wind closing the Charney–Eliassen input
    of the SLP stage (two-pass closure, replacing the first sub-deliverable's
    documented `sin_cos_alpha_bar` and `u500` placeholders). Guarded by
    `testing/test_sesam_wind.py` (16 tests, incl. the analytic-gradient and
    hand-solve checks that caught a real d/dλ scaling bug during
    development). `scripts/diagnose_sesam_wind.py` runs the full chain
    (SLP → wind) diagnostically on a saved state for DJF/JJA and scores it
    head-to-head against the prescribed-cell generator and NCEP/NCAR wind.
  - **Exit-gate measurement (2026-08-16, saved 512×1024 state): NOT passed —
    with a decisive decomposition.** Full-chain surface-wind speed vs NCEP:
    pattern correlation −0.28 (DJF) / −0.19 (JJA), RMSE 73 / 113 m s⁻¹ — the
    prescribed generator holds +0.34 / +0.23 and 2.8 / 3.6 m s⁻¹. But with
    the azonal SLP terms removed, the *zonal-only* chain **beats the
    generator**: +0.55 / +0.46 correlation, 2.9 / 3.7 m s⁻¹ RMSE, mean
    surface speed 2.2–2.4 m s⁻¹ (NCEP 3.9–4.0), SH jet −56°/8.7 m s⁻¹ vs
    NCEP −51°/8.2. The catastrophic full-chain failure is entirely the
    **azonal channel amplifying the saved state's sharp regional fields**:
    (A37)'s 232 Pa K⁻¹ converts the state's ±35…+51 K sea-level-temperature
    anomalies (ice sheets, high plateaus, sharp coastlines) into ±118 hPa
    SLP, and (A38)–(A39) respond to full-resolution 8848 m terrain with
    ±100–170 hPa — whose gradients then drive 100+ m s⁻¹ local geostrophic
    winds, further amplified by the |f| = 3e-5 s⁻¹ tropical floor. The
    cell-physics core is sound; the azonal closures need conditioned inputs
    (SESAM-native smoothing like the reference implementation's own
    `nsmooth_*` filters, or a state spun up with the SESAM closures active)
    before any coupling or calibration sweep is admissible. Also measured:
    the SH-summer Hadley cell in the saved state is too weak for the cell
    physics (ΔT = 0.28 K in DJF) — a state property, not a kernel error.
  - *First honest SLP-only baseline (2026-08-16, first sub-deliverable,
    documented placeholders for sin α·cos α, u500 and H_T):* DJF full-field
    pattern correlation vs NCEP 0.49 (RMSE 11.2 hPa), DJF zonal-mean profile
    0.78; JJA 0.29 / 0.31. The NH-winter subtropical high is reproduced at
    the right latitude and strength; the summer-hemisphere cells and the
    model state's polar gradients are the weak points. The SLP-stage
    placeholders were subsequently closed by the wind assembly (the (A21)
    solve and the two-pass u500 closure), and the full exit-gate measurement
    above supersedes this baseline for the completed P2.
- **P3 — EKE and synoptic transport** (A5): prognostic K, Eady production from
  P1 profiles, drag dissipation, `AT/Aq` diffusivities, synoptic wind. *Exit*:
  storm-track placement responds to baroclinicity; macroturbulent heat/moisture
  fluxes replace the fixed-window eddy term in the branch.
  - **Sub-deliverable complete (2026-08-16): the EKE closure** (A50)–(A60) as
    pure functions in `sesam_synoptic.py` — (A54) Brunt–Väisälä frequency,
    (A53) Eady-baroclinicity production, (A55) drag dissipation, (A50)/(A51)
    macroturbulent diffusion coefficients, (A56)/(A57) synoptic surface wind
    and 700 hPa vertical velocity, (A58) total surface wind, (A59)/(A60) wind
    stress, and the diagnostic steady-state EKE (local production/dissipation
    balance; transport of K is stage P4). Guarded by
    `testing/test_sesam_synoptic.py` (12 tests).
    `scripts/diagnose_sesam_synoptic.py` runs it on the saved state for
    DJF/JJA. The stage-P2 missing storminess component is now closed: total
    surface wind ≈ zonal-cell wind (2.2–2.4 m s⁻¹) + synoptic gustiness
    (≈7 m s⁻¹) ≈ 8 m s⁻¹, EKE mean ≈ 240 m² s⁻² (p50 ~160–190, p90 ~480),
    storm track ≈ 49°, wind stress ≈ 0.06 Pa, macroturbulent heat
    diffusivity ≈ 3.1e6 m² s⁻¹ — all Earth-realistic. Two methodological
    findings are recorded in §10: the EKE must be driven by the *zonal-only*
    P2 wind (the full-chain wind inherits the P2 azonal input-conditioning
    inflation), and the (A54) Brunt–Väisälä frequency must use *potential*
    temperature.
  - *Constant correction (2026-08-16, §5-mandated cross-check):* the
    published supplement transcribes `c2syn` = 1.6e4, but the reference
    namelist `c_syn_2` = 1.6e2 is authoritative — 1.6e4 gives an absurd ~5e3
    m² s⁻² equilibrium EKE / ~30 m s⁻¹ synoptic wind; 1.6e2 gives the
    validated ~2e2 m² s⁻² / ~7 m s⁻¹. `c5syn` also corrected to the namelist
    `c_syn_5` = 2.3e5. Both recorded in `sesam_reference.py` notes.
  - *Exit gate status:* the storm-track placement in this diagnostic responds
    to baroclinicity as required (peak EKE at the jet's latitude, zero at the
    equator). The *transport* sub-deliverable (macroturbulent heat/moisture
    fluxes replacing the fixed-window eddy term) is stage P4, so the full P3
    exit gate is not yet closed — this sub-deliverable covers the closure and
    its diagnostics, not the transport wiring.
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

## 10. Equation-semantics verification log

Findings from checking appendix equations against the article's HTML MathML
(which preserves fraction/cases structure that the PDF text layer flattens)
and — per the §5 licensing policy, read-only — against the CLIMBER-X Fortran
semantics. Recorded as they are verified, so no later stage re-derives them.

- **(A31) operand grouping** (verified 2026-08-16): the cell coordinate is
  `φ = 6·Dhad·(ϕ − φITCZ/(c1mmc·(ϕ − φITCZ)² + 1))`; the rational factor
  applies to `φITCZ` only, not to the whole `(ϕ − φITCZ)` difference.
- **(A33) is a fraction**: `Dhad = c3mmc/(T_trp − c4mmc)`, not a product.
  Dimensionally consistent (Dhad ≈ 1 at present tropical temperatures) and
  gives the paper's stated behaviour (warming → smaller Dhad → `φ = ±π`
  crossings move poleward → Hadley cells expand). Reference-implementation
  safeguards adopted: scale clamped to [0.5, 1.5], `T_trp` floored at
  `c4mmc + 50 K`. The `sesam_reference.py` c3mmc/c4mmc notes were corrected
  2026-08-16.
- **(A34) operand order (the one real correction)**: as printed, the
  gradients are negative on Earth in all three cells and, fed through
  (A29)–(A30), yield poleward surface flow in all six cell branches — a
  divergent, physically impossible circulation with SLP highs on the equator.
  The circulation-correct ordering, and the one the reference implementation
  evaluates, is the reverse difference at the same fixed latitudes: Hadley
  `max(0, max(T̄) − T̄(±π/6))`, Ferrel `T̄(±π/6) − T̄(±π/3)`, polar
  `T̄(±π/3) − T̄(±π/2)`. This gives equatorward Hadley/polar and poleward
  Ferrel surface flow in both hemispheres, hence subtropical highs, subpolar
  lows and the ITCZ trough out of (A29). `sesam_dynamics.py` implements the
  corrected ordering; `test_sesam_dynamics.py` pins every branch's sign and
  plants the printed ordering to show it reverses the circulation.
- **(A39) glyph**: the printed `p*sl,O = 9fρ(500 hPa)` is the streamfunction
  Ψ mis-OCR'd: `p*sl,O = ρ(500 hPa)·f·Ψ` (with `|f|` in the conversion).
- **The (A14) ftrop coordinate** (P1 placeholder resolution): ftrop is
  `1 − sin⁸(fi)` with `fi = clamp(c_hrs·(ϕ − had_fi)/(0.5·had_width), ±π/2)`,
  `c_hrs = 0.7` (CLIMBER-X namelist `c_hrs_6`; the paper prints ftrop without
  defining φ's construction; the same structure with 0.85 = asin(0.1^{1/8})
  is printed in (A11)). `had_fi`/`had_width` are the Hadley centre/width
  diagnosed from the (A31) cell coordinate
  (`sesam_dynamics.hadley_geometry`), replacing P1's latitude placeholder via
  `sesam_dynamics.tropical_weight_from_hadley`.
- **Reference-implementation features deliberately not ported** (grid
  artifacts/tuning, not paper equations): staggered-grid moving averages,
  azonal-SLP spatial smoothing, polar/equatorial azonal damping factors, and
  azonal-SLP time relaxation (the P2 kernels are diagnostic, not time-stepped).
- **NCEP SLP reference**: the NOAA PSL `slp.mon.ltm.1991-2020.nc` file is in
  millibars; `scripts/build_ncep_slp_reference.py` converts to Pa explicitly
  (verified against the file's `units` attribute, per the ExoPlaSim
  precipitation-unit lesson in `docs/EXTERNAL_DYCORE_WORKFLOW.md`).

### Wind-assembly stage (verified 2026-08-16)

- **ε = √(1 − sin 2α)** in (A21)/(A24)/(A25): the printed ``sin2α`` is the
  double angle sin(2α), and ε is computed from |α| (the drag closure is
  hemisphere-symmetric; α itself is signed, positive NH, used only in the
  Taylor rotation). Verified against the reference implementation.
- **The (A21) cross-isobar solve needs no EKE input**: with the paper's
  `Us ≈ √(2K)` and the reference implementation's PBL viscosity tied to the
  EKE (Kv = K), K cancels and the closure reduces to
  `sinα/√(1 − sin 2α) = CD/√|f|`, solved by bisection on [0, π/4] and
  clamped to α ∈ [0.05, 0.5] rad (namelist `acbar_max`).
- **sinα·cosα enters (A19)/(A20)/(A29) as a positive magnitude** with |f|
  (flow crosses isobars toward low pressure in both hemispheres). The
  `sesam_dynamics` zonal-SLP convention was corrected to this form on
  2026-08-16 (algebraically identical to the earlier signed form for the
  scalar path, but unambiguous for array input).
- **Coriolis floors**: paper text |f| ≥ 3e-5 s⁻¹ (geostrophic) and
  |f| ≥ 1e-5 s⁻¹ (ageostrophic); the reference namelist uses 1e-5 for both.
  The paper's two-floor form is implemented.
- **Thermal-wind damping** (reference safeguard, adopted, not printed):
  shear × `min(1, c_uter_eq·sin²ϕ)·min(1, c_uter_pol·cos²ϕ)` with
  c_uter_eq = 5, c_uter_pol = 3.
- **Ageostrophic profile**: surface value uniform through the PBL
  (σ_pbl(ϕ) = 0.85 − 0.05·cos²ϕ, namelist pblp/pble), compensated by a
  uniform counter-flow over a σ = 0.2 layer below the tropopause (namelist
  dpc), exact column mass balance.
- **Katabatic (A26)–(A27)**: slope magnitude is *inside* the radical
  (`uk = √(g·h/CD·(T2m−T*)/T2m·|slope|)·sign(−slope)`), gated on the
  inversion condition T2m > T*, h = 100 m (paper prose).
- **Input conditioning is the P2 blocker, not the closures** (measured,
  see §7 P2): (A37)'s 232 Pa K⁻¹ and (A38)–(A39) respond linearly to the
  saved state's ±35…+51 K sea-level-temperature anomalies and full-
  resolution terrain, producing ±118/±170 hPa azonal SLP and 100+ m s⁻¹
  winds; the zonal-only chain on the same state is sane and beats the
  prescribed generator. Also: the (A9) near-surface lapse Γ = (Ta−T*)/zpbl
  is catastrophically sensitive to season-mismatched Ta/T* inputs (a
  January Ta with a July skin produced a −50 K/km inverted profile in the
  diagnostic driver) — diagnostic drivers must use season-consistent
  Ta/T* pairs.

### Synoptic/EKE stage (verified 2026-08-16)

- **`c2syn` transcription corrected** (§5-mandated cross-check at first use):
  the published supplement (t06) transcribes `c2syn` = 1.6e4, but the
  CLIMBER-X reference namelist `c_syn_2` = 1.6e2. With 1.6e4 the local
  equilibrium EKE is ~5e3 m² s⁻² and the synoptic surface wind ~30 m s⁻¹ —
  physically absurd; with 1.6e2 the diagnostic gives ~2e2 m² s⁻² and
  ~7 m s⁻¹, matching validated CLIMBER-X behaviour. `c5syn` also corrected
  to the namelist `c_syn_5` = 2.3e5. Recorded in `sesam_reference.py`.
- **(A54) Brunt–Väisälä frequency uses *potential* temperature.** With
  temperature differences, `N² = (g/T)·ΔT/Δz` is negative in the stably
  stratified troposphere (T decreases upward) and the frequency comes out
  zero/NaN. The reference implementation's `tp` field is potential
  temperature; `N² = (g/θ)·Δθ/Δz` is positive. Implemented accordingly.
- **(A53) production uses the full Eady shear**: `f/N·√((∂u/∂z)² + (∂v/∂z)²)`
  (both components), not just the printed `(f/N)·(∂u/∂z)` — the Hoskins and
  Valdes (1990) form the accompanying text specifies.
- **EKE diagnostic must be driven by the zonal-only P2 wind.** The full-chain
  P2 3-D wind inherits the P2 azonal input-conditioning inflation (see
  above); driving the Eady production with it compounds the artefact
  (a ~5e3 m² s⁻² EKE). Driven by the zonal-only wind (the sane ~2.4 m s⁻¹
  circulation), the EKE is realistic (~2e2 m² s⁻²) and the storminess term
  correctly *adds* to it. This is a methodological finding, not a kernel
  correction.
