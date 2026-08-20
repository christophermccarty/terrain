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
2. When the branch is feature-complete (§7 stage P5), it gets **one bounded
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
  - **Exit-gate re-investigation (2026-08-17/18): four structural bugs found
    and fixed; exit gate massively improved but still NOT passed, and the
    remainder is now measured to be a calibration question, not a further
    bug.** The 2026-08-16 verdict's diagnosis ("azonal channel amplifying the
    saved state's sharp regional fields") named resolution as the cause.
    That turned out to be a minor contributor; the investigation instead
    found and fixed, in order:
    1. **(A38) denominator, erroneous extra `u` factor.** Fetching the
       article PDF directly (a smaller preprint mirror,
       `gmd-2022-56-manuscript-version2.pdf`, pages 43-44 -- the final
       typeset PDF and full HTML/XML both exceed this session's fetch-size
       limits before reaching the appendix) and re-deriving the (A38)
       spectral solution from the literally printed vorticity equation
       reproduced this module's own documented formula
       (`u·Kn² − β − i·Kn²/(τe·kzn)`), but the *code* computed
       `Kn²/(τe·kzn·u)` -- an extra `u` absent from both the derivation and
       the function's own docstring one line above it. Confirmed wrong
       dimensionally (m⁻² vs. every other term's s⁻¹m⁻¹) and by measurement
       (a (lat, n, u500) sweep found |response| spiking above 1e5 at the
       stationary-wavenumber resonance `u·Kn² = β` that exists for some
       integer n at any positive u -- not an edge case). Fixed
       (`sesam_dynamics.py`, module docstring note 9); real-terrain
       orographic SLP fell from ±97-170 hPa to ±18-20 hPa.
    2. **Pole `cosϕ` singularity in `vg(0)`/`ua`.** (A17)-(A20) divide by
       `cosϕ`, excused by the paper's text only at the equator; the same
       breakdown holds at the poles on this module's discrete grid, measured
       at >6500 m/s within one row of a pole. Fixed (`sesam_wind.py`, note
       9) with `thermal_wind_shear`'s own already-accepted
       `min(1, c_pol·cos²ϕ)` envelope.
    3. **`u500(ϕ)` row-to-row noise.** With 1-2 fixed, the exit gate still
       failed at real mid-latitudes despite smooth 2-D SLP inputs: `dp/dλ`
       was small but `dp/dϕ` was ~90x larger. (A38) is solved independently
       per row with no coupling between adjacent latitudes, so nothing kept
       it smooth in ϕ even when every input was -- and its driver, `u500`
       (closing the two-pass SLP<->wind loop from real, ungridded saved-state
       fields), genuinely oscillates row to row (measured: -28, -16, -15,
       -18, -1, +13, +26 m/s across seven adjacent rows), which the
       westerly-only `+0.1 m/s` floor turns into a near-discontinuous
       regime switch on every sign flip. Fixed (`sesam_dynamics.py`, note
       10) with `resolution_matched_profile`, a new 1-D latitude analogue of
       note 7's 2-D regrid, applied to `u500` before it drives the per-row
       solve.
    4. **Equatorial breakdown of `ug(0)`/`va`, not just the pole.** With 1-3
       fixed, the gate still failed at real low-to-mid latitudes (~15-40°)
       from an *already-smooth* SLP gradient (~1.4 hPa/row, a genuine,
       non-noisy feature) divided by a small-but-not-floored `f`. Fix 2 only
       covered `vg(0)`/`ua`'s `cosϕ` term; `ug(0)` and `va` had no analogous
       protection. Fixed (`sesam_wind.py`, note 11) by applying the *full*
       `min(1, c_eq·sin²ϕ)·min(1, c_pol·cos²ϕ)` envelope (both factors, not
       just the polar one) to all four surface components, via a new shared
       `_geostrophic_frame_damping` helper reused by `thermal_wind_shear`
       (unchanged behaviour there) -- i.e. the same damping the shear
       integral already had, now applied to the surface terms too.
       `compute_wind` exposes it as a new `surface_damping` parameter.

    All four fixes are guarded by new tests (planted-bug /
    planted-undamped-blowup style, matching the existing (A34) precedent):
    `testing/test_sesam_dynamics.py` (11 new tests) and
    `testing/test_sesam_wind.py` (7 new tests); full SESAM suite 109/109,
    full project suite 999 passed/1 skipped/20 xfailed/2 xpassed (no
    regressions -- both modules stay behind `enable_sesam_dynamics`, default
    off, unreferenced by `simulate.py`).

    **Combined exit-gate result** (saved 512×1024 state, all four fixes):
    full-chain surface-wind speed vs NCEP pattern correlation −0.011 (DJF) /
    −0.121 (JJA), RMSE 6.74 / 6.82 m s⁻¹ -- RMSE improved **10.8x / 16.6x**
    from the 2026-08-16 baseline (73/113 m s⁻¹) and mean surface speed
    (5.46/5.05 m s⁻¹) is now close to NCEP's own (3.9-4.0). The generator
    (+0.34/+0.23, 2.8/3.6 m s⁻¹) still wins on correlation, though DJF is now
    within 0.35 correlation points of it (JJA remains further off).

    **The remainder is measured to be a calibration question, not a further
    bug.** Two checks close this out: (a) clipping the top 1-10% of
    remaining wind-speed outliers *does not* improve correlation (it stays
    flat or worsens slightly across p90-p99 clip thresholds), meaning the
    residual gap is a broad pattern mismatch, not a few hot cells; (b) the
    equatorial-damping strength `c_eq` (reused at 5.0 from
    `thermal_wind_shear`'s existing namelist value, not chosen for this
    metric) was swept from 0.5 to 5 on the same state: the correlation
    response is real but **non-monotonic and season-inconsistent** -- `c_eq
    ≈ 2` scores best for DJF while JJA prefers weaker values and neither is
    uniformly best on RMSE. A parameter whose "best" value depends on which
    season's snapshot is scored is the signature of a genuine
    constant-calibration question, not a discrete defect -- exactly what §6's
    bounded P6 calibration window exists for. The default was left at the
    principled 5.0/3.0 value rather than hand-picked from this sweep, per
    §6/§7's own discipline against per-instance constant fitting. Separately
    (state-specific, already noted 2026-08-16): the saved state's own
    SH-summer Hadley cell is too weak for the cell physics, a state
    property unrelated to any of the four fixes.

    **Verdict: P2's kernels are now bug-free as far as this investigation can
    determine; the exit gate is not passed and is not expected to pass via
    further debugging.** All four fixes are real, dimensionally/textually
    verified corrections (not tuned knobs) and stay in unconditionally; do
    not revert them. Closing the remaining gap requires either (i) the P6
    calibration window (after P4), scoring a bounded sweep of SESAM's own
    constants against the full standard gate set rather than this one
    state's wind correlation, or (ii) validating against additional
    states/seasons to separate genuine constant miscalibration from this
    particular saved state's own known biases. Neither is admissible as a
    P2-stage action per the project's own staged discipline; P2 does not
    re-open for further hand-tuning until one of those conditions is met.
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
  - *Exit gate status (superseded below):* the storm-track placement in this
    diagnostic responds to baroclinicity as required (peak EKE at the jet's
    latitude, zero at the equator). The *transport* sub-deliverable
    (macroturbulent heat/moisture fluxes replacing the fixed-window eddy
    term) is stage P4, so the full P3 exit gate is not yet closed — this
    sub-deliverable covers the closure and its diagnostics, not the
    transport wiring.
  - **Second sub-deliverable complete (2026-08-18): the (A52) prognostic K
    transport itself** — `eke_diffusion_step` (nonlinear diffusion of K by
    its own `AT = c5syn·√K`), `eke_transport_step` (adds advection, reusing
    `column_water.evolve_column_water`'s exact finite-volume geometry for
    the advective term per the task's own instruction — K obeys the
    identical conservative flux-divergence transport equation as column
    water), and `evolve_eke` (the full `dK/dt = -div(uK) + div(AT·grad K) +
    PK - DK`, operator-split: transport then the local production/
    dissipation reaction). All in `sesam_synoptic.py`, CFL-substepped,
    default-off (`enable_sesam_dynamics`), unwired from the supported path.
    Guarded by 8 new tests in `testing/test_sesam_synoptic.py` (20 total):
    a hand-computable single-substep diffusion value, two independent
    planted-violation tests (omitting AT face-averaging desymmetrises a
    symmetric bump's response; omitting CFL sub-stepping lets one Euler step
    blow past the discrete maximum principle and go negative), a
    pure-transport conservation check (zero production/dissipation conserves
    `sum(K·area)` to float32 precision, matching `column_water.py`'s own
    contract), and two physical-sanity checks (diffusion smooths a bump
    while leaving its centroid essentially unmoved; advection by a uniform
    wind translates a bump's centroid by the expected `u·dt` distance,
    verified via circular-mean tracking on the periodic grid).
  - **Two real bugs found and fixed while verifying against the real
    512x1024 saved state** (both dimensionally/logically real defects, not
    calibration or a one-state artifact — see §10 for the full writeup):
    1. The diffusion sub-step count was originally computed *once* from the
       initial AT, reasoning that pure diffusion cannot raise a field's peak
       so the initial state bounds every later one. That argument silently
       assumed the scheme was already stable to prove it stable, and
       overflowed on the real state. Fixed with per-substep adaptive
       re-estimation from the live K.
    2. Adaptive re-estimation alone did not fix it: the CFL rate formula
       (`AT·(1/dx²+1/dy²)`) implicitly assumes cell area `~ dx·dy`, which
       fails specifically at the polar-cap row, whose true area shrinks
       toward the actual (tiny) spherical-cap value while the face length
       stays the same constant as everywhere else — silently understating
       the true constraint exactly at the pole. Confirmed as a formula bug,
       not an insufficient safety margin, by tightening `diffusion_r_limit`
       8x (0.4 → 0.05) on the real state without effect. Fixed by computing
       the exact per-cell self-loss coefficient from the true face-length/
       area geometry instead of the `dx`/`dy` approximation.
  - **Measured (DJF, saved 512x1024 real-terrain state):** local
    steady-state-only EKE (no transport, prior sub-deliverable's number):
    mean 245.9 m² s⁻², max 6201.6 m² s⁻², storm track 49.39°N. Running the
    full prognostic closure (advection + diffusion + production/dissipation)
    forward to convergence (15 iterations of a 0.25-day coupling step = 3.75
    days simulated, 13.6 s wall time) gives: EKE mean 278.2 m² s⁻² (+13%),
    max 2813.1 m² s⁻² (−55%, transport smooths the sharp local peak, exactly
    the expected qualitative effect), storm track 49.92°N (+0.5°, i.e.
    essentially unchanged). **Storm-track-vs-baroclinicity check: passes**
    with transport active, holistically re-verifying the local-only
    sub-deliverable's finding. Resulting macroturbulent heat diffusivity:
    AT mean 3.60e6 m² s⁻¹, max 1.22e7 m² s⁻¹ — consistent with (slightly
    above) the already-reported local-only ~3.1e6 m² s⁻¹; **transport does
    not change the order-of-magnitude conclusion.** The incumbent
    `eddy_heat_flux_coeff = 0.006` fixed-window term (`simulate.py` ~line
    5803) is algebraically equivalent to standard diffusion with
    `D_eff = coeff·dy²` (its Laplacian update is unnormalised by `dy²`);
    at this project's headline 512-row grid spacing, `D_eff ≈ 106 m² s⁻¹` —
    **the transport-equilibrated SESAM AT is ~34,000x larger.** The
    incumbent's `D_eff` is inherently grid-resolution-dependent (scales with
    `dy²`) while SESAM's AT is a physical quantity independent of grid
    resolution — itself one more instance of the architectural gap §2 and
    §4 describe.
  - **Resolution caveat (an honest limitation, not a further bug):** the
    prognostic-equilibrium measurement above ran on a 128x256 block-averaged
    downsample of the real saved state, not the full 512x1024 headline grid.
    The area/geometry fix above is confirmed correct and resolution-general
    (unit-tested from 3x4 up to 32x64, and the real-state run showed no
    divergence before being stopped for time at full resolution), but at
    512 rows the now-*correctly*-enforced polar-cell CFL constraint demands
    many thousands of diffusion sub-steps near the poles, making a full
    512x1024 confirmation run impractically slow within this session's time
    budget. This is a performance/tractability limitation, not a
    correctness question, and 128x256 is a precedented resolution for this
    kind of screen (matches this project's existing 64x128 compact-screen
    convention and the §6 P6 calibration window). A full-resolution
    confirmation run (or a performance pass — e.g. vectorising the substep
    loop, or a coarser default coupling step tuned for high-resolution
    poles) is flagged as follow-up, not attempted here.
  - **Full-resolution confirmation (2026-08-18, supersedes the 128x256-transport
    measurement above and closes the resolution caveat):** the performance
    blocker was root-caused, not worked around. Direct measurement on the
    real 512x1024 grid (`spherical_transport_geometry`/`zonal_center_spacing_m`)
    shows the polar CFL stiffness the caveat above describes lives almost
    entirely in the **zonal** (east-west) diffusion self-loss term, not the
    meridional (north-south) one: splitting `eke_diffusion_step`'s exact
    self-loss geometry into its two directional halves, the meridional term
    is flat across every latitude row (≈1.31e-9 m⁻², resolution-driven but
    not pole-specific) while the zonal term grows from ≈1.31e-9 at the
    equator to ≈1.01e-6 at the pole row — 770x larger, and alone responsible
    for >99.8% of the pole's self-loss rate. Mechanism: `x_len =
    radius·dlat` (the east/west face length) is the *same constant* at
    every row by construction, while true cell area shrinks toward the
    actual spherical-cap value at the pole, so `x_len/area` diverges there —
    the classic lon-lat "pole problem" real circulation models solve with
    implicit zonal treatment or Fourier polar filters, not with implicit
    *meridional* diffusion (the initial hypothesis going into this
    investigation, based on the ocean/atmosphere-model implicit-vertical-
    diffusion analogy — measurement overturned it). At the diagnosed
    near-pole EKE magnitudes (K ≈ 6.2e3–1.3e5 m² s⁻²), the zonal term alone
    demanded on the order of 1–20 million diffusion sub-steps for a single
    0.25-day coupling step at 512 rows — genuinely intractable explicitly,
    confirming the caveat's "impractically slow" was a real stiffness fact,
    not a fluke.
  - **Remedy implemented: `eke_diffusion_step_implicit_zonal`
    (`sesam_synoptic.py`), a new, separately-tested code path** — an
    ADI-style directional split: the stiff zonal direction is solved
    implicitly (backward Euler, unconditionally stable for positive
    conductances, so no CFL constraint at all) via a periodic (cyclic)
    tridiagonal solve per row, batched over all rows
    (`_cyclic_thomas_batch`, the standard Sherman-Morrison/Numerical-Recipes
    §2.7 cyclic algorithm, cross-checked during development against
    `numpy.linalg.solve` on the equivalent dense matrix for arbitrary
    non-constant-coefficient rows to machine precision); the mild
    meridional direction stays explicit and CFL-substepped exactly as
    `eke_diffusion_step` already does, since it was never the stiff one.
    Same face-averaged-AT conservative finite-volume geometry, same
    nonlinear re-evaluation of AT from the live K each sub-step. This is
    the standard textbook remedy for CFL-limited explicit diffusion on a
    grid converging toward a point — a choice of numerical integrator, not
    a physics shortcut, cap, or damping (`docs/VERTICAL_THERMODYNAMIC_CLOSURE.md`
    precedent). It is opt-in (`implicit_zonal_diffusion=True` on
    `eke_transport_step`/`evolve_eke`) and additive: the original explicit
    `eke_diffusion_step` and its exact-value/planted-violation tests in
    `testing/test_sesam_synoptic.py` are untouched and still pass unchanged.
    Verified before use: on the 24x48 tractable-grid case both schemes are
    already validated on, implicit-zonal lands within 5% (peak) / 15%
    (neighbours) of explicit after the same simulated time (expected — two
    consistent time-discretisations of the same PDE, not identical since
    backward- vs forward-Euler differ); on a synthetic small grid that
    reproduces the real pole's exact area/face-length ratio, the plain
    explicit kernel correctly raises `RuntimeError` (>200,000 sub-steps,
    infeasible) while the implicit kernel completes in <60 sub-steps,
    stays finite, non-negative, mass-conserving, and respects the discrete
    maximum principle. 4 new tests added (24 total in the file).
  - **Full 512x1024 measurement, both seasons, converged:** run via
    `scripts/diagnose_sesam_synoptic.py --block-size 1` (now the default —
    a `--block-size`/`--explicit-diffusion` pair of flags still exposes the
    old fast coarse-screen path for iteration). Both seasons ran the
    identical local closure and 0.25-day-coupled transport loop as the
    128x256 measurement above, just at native resolution throughout, with
    `implicit_zonal_diffusion=True`.
    - **DJF:** local steady state mean 245.9 m² s⁻², max 6201.6 m² s⁻²,
      storm track 49.39°N — identical to the 128x256 entry's local numbers,
      as expected (that number was already computed at native resolution;
      only the *transport* step was previously downsampled). Prognostic
      transport converged in 16 iterations (4.0 days simulated, 1731.5 s
      wall time): mean 275.9 m² s⁻² (+12.2%), max 5621.9 m² s⁻² (−9.3%),
      storm track 49.39°N (**0.0° shift** — the peak EKE row is literally
      unchanged by transport). AT mean 3.566e6 m² s⁻¹ — **33,603x** the
      incumbent's `D_eff ≈ 106 m² s⁻¹` effective diffusivity at this grid.
    - **JJA:** local steady state mean 232.8 m² s⁻², max 6208.2 m² s⁻²,
      storm track 49.39°N. Prognostic transport converged in 16 iterations
      (4.0 days, 1706.0 s wall time): mean 262.0 m² s⁻² (+12.5%), max
      5632.8 m² s⁻² (−9.3%), storm track 49.39°N (0.0° shift). AT mean
      3.475e6 m² s⁻¹ — **32,747x** the incumbent.
  - **Agreement and divergence with the 128x256 screen.** The
    exit-gate-relevant conclusions are robustly **confirmed**: mean EKE
    increase from transport (+12–13%) and the AT-vs-incumbent order of
    magnitude (~33,000–34,000x both resolutions) match closely, and the
    storm-track-tracks-baroclinicity check passes even more cleanly at
    native resolution (exactly 0.0° shift vs the coarse screen's +0.5°).
    But the two **disagree substantially on how much transport smooths the
    local peak**: −55% at 128x256 (6201.6 → 2813.1) vs only **−9.3%** at
    native 512x1024 (6201.6 → 5621.9). This is a genuine, real finding, not
    a bug in either measurement: the 128x256 screen's transport step ran on
    a block-averaged *downsample* of the already-computed local K field, so
    its own averaging pre-smooths the sharp local peak before the transport
    loop ever starts, on top of whatever the transport loop itself then
    does at the coarser grid — the coarse screen was never measuring "how
    much does transport smooth the true peak," it was measuring that
    question on an input that had already lost most of the peak's sharpness
    to block-averaging. The full-resolution run is the first measurement of
    the actual quantity the exit gate cares about, and it shows the true
    peak survives transport much more intact than the coarse screen implied
    — worth carrying forward as a caution against trusting a coarse-grid
    transport magnitude even when the coarse grid's *order-of-magnitude*
    and *qualitative* conclusions (both true here) hold up.
  - **Exit gate verdict: closed.** The mechanism-replacement question the
    exit gate asks — does a derived, baroclinicity-responsive macroturbulent
    diffusivity now exist that could structurally replace the fixed-window
    incumbent — is answered yes, with a real, converged, real-terrain-derived
    measurement **at the project's actual headline 512x1024 grid, both
    seasons**: AT is ~4.5 orders of magnitude larger than the incumbent's
    effective diffusivity, and storm-track placement is exactly unchanged by
    transport (0.0° shift both seasons), the cleanest possible pass of the
    storm-track-vs-baroclinicity check. The resolution caveat that kept the
    prior entry at "substantially advanced, not unconditionally closed" is
    resolved by genuine numerical-methods engineering (the implicit-zonal
    remedy above), not by approximation, a time-budget trade, or accepting
    the coarser number — the full run was performed and is the number
    recorded here. What remains explicitly out of scope, unchanged from the
    prior verdict: wiring the branch into the supported climate path and any
    resulting climate-impact comparison, which is P6's job, not P3's.
- **P4 — Column energy and water closure** (A3/A4) ✅ **exit gate passed
  (2026-08-18)**: prognostic `QT`/`Ta` (A40), `Qq`/`qa` (A42, reusing
  `column_water.py` flux machinery), the (A41)/(A43) `T2m`/`q2m` near-surface
  diagnostics, and the (A44)/(A45) precipitation closure (moisture
  convergence scaled by a continuous relative-humidity efficiency, plus a
  land-only turnover term) as pure functions in `sesam_thermo.py`. This is
  the stage that **bypasses the row-target allocator**: `P` in this branch
  never touches `target_row_mm_day`, it is entirely a local moisture-budget
  residual. Gated `enable_sesam_column_closure` (default False, **not wired
  into the supported path**), guarded by `testing/test_sesam_thermo.py`
  (29 tests: hand-computed values for every equation, planted-violation
  tests for the shared diffusion primitive's face-averaging and CFL
  sub-stepping, and conservation checks matching `column_water.py`'s own
  contract). `scripts/diagnose_sesam_thermo.py` runs the full chain
  (P1 vertical structure → P2 zonal-only SLP/wind → P3 local-steady-state
  EKE → P4 column energy/water) diagnostically on the saved state for
  DJF/JJA.
  - **Equations verified against the article PDF directly** (2026-08-18,
    same preprint mirror as the P2 §10 A38 finding, pages 44-47): this is
    the first time A40-A45 were read from the source rather than
    paraphrased. One real correction to this document's own earlier
    shorthand: the gap-analysis dossier's §1/§7 paraphrase "moisture
    convergence past 95% RH" undersold (A44)'s actual form — it is not a
    hard threshold at `ra > ramax`, but a *continuous* efficiency `ra/ramax`
    applied to the entire gross convergence-plus-evaporation term at every
    `ra`, reaching 100% conversion only when `ra` reaches its ceiling. Full
    verification log in `sesam_thermo.py`'s module docstring (8 numbered
    notes) — cv/Le/Ls are confirmed genuinely unpublished (Table A3 prints
    no parameter table beyond physical constants), so this module adopts
    this project's own existing `cp=1004.0`/`Rd=287.0`/`Le=2.5e6` convention
    (`land_surface.py`, `atmosphere.py`) rather than a fourth independent
    set of constants.
  - **(A46)-(A51) finding**: confirmed directly from the PDF that the
    macroturbulent diffusivities `AT`/`Aq` used for Ta/qa transport are the
    *literal same* `AT = c5syn·√K` / `Aq = c6syn·K` stage P3 already built
    for K's own (A52) transport — not a new closure applied to a different
    scalar by analogy, but the identical symbols. `sesam_thermo.py` imports
    P3's `horizontal_diffusion_coefficient`/`moisture_diffusion_coefficient`
    directly rather than recomputing them.
  - **Polar zonal-diffusion stiffness recurs here, pre-empted rather than
    re-discovered**: since AT/Aq inherit K's own magnitude (~1e6-1e7 range),
    diffusing Ta/qa by them at the 512x1024 headline grid hits the identical
    `x_len/area` polar divergence P3's 2026-08-18 entry found and fixed for
    K's own diffusion. Rather than wait to hit it on a real run,
    `sesam_thermo.py` ships a generic `_linear_diffusion_step_implicit_zonal`
    from the start, reusing P3's already-validated `_cyclic_thomas_batch`
    cyclic tridiagonal solve directly (not a second implementation of the
    same algorithm). Simpler than P3's own implicit-zonal kernel in one
    respect: because AT/Aq here are *externally supplied* (computed once
    from K, not self-referential the way K's own AT is), the substep count
    and zonal conductances are built once per call rather than adaptively
    re-estimated — there is no nonlinear self-consistency loop to guard
    against, only the ordinary CFL bound on the (never pole-stiff)
    meridional half.
  - **Two real bugs found and fixed during development** (both caught by
    `testing/test_sesam_thermo.py` before this module ever touched the real
    saved state, matching the "verify before use" discipline P3's own bug
    hunts established): (1) the new diffusion primitive initially had no
    substep-count safety cap, unlike `sesam_synoptic.eke_diffusion_step`'s
    own `max_substeps=200_000` — a pathological-geometry test case hung the
    test run indefinitely instead of raising `RuntimeError`; fixed by
    porting the same cap to both the explicit and implicit-zonal variants.
    (2) `evolve_column_water_vapor` passed the column-integrated water depth
    `Qq` (mm) into (A45)'s `qa` slot, which needs near-surface *specific
    humidity* (kg/kg) — a distinct physical quantity the function never
    separately received. This inflated the slope-convergence term by
    roughly the mm-to-kg/kg magnitude ratio (~1000x) and was caught by a
    synthetic-field physical-bounds sanity test (global P computed at
    2345 mm/day against an intended "generously above Earth's mean" ceiling
    of 50). Fixed by adding an explicit, separate
    `near_surface_specific_humidity_kg_kg` parameter; the diagnostic
    script's own call site had the same conflation (passing raw specific
    humidity into the *relative*-humidity `near_surface_rh` slot) and was
    fixed alongside by diagnosing `ra = qa/qsat(Ta, p0)` explicitly.
  - **Diagnostic placeholders** (documented, not fabricated physics,
    following the same policy P2/P3's drivers already established): no
    evaporation field is saved, so the script computes a standard bulk-
    aerodynamic estimate `E = rho0·Ch·|wind|·max(qsat(Tskin)-qa, 0)`
    (`Ch=1.3e-3`, the saved `wind_speed_avg` field) — not the project's
    real evaporation code, which the script does not attempt to extract
    from `simulate.py`/`atmosphere.py`. `Qq` is diagnosed from the saved
    specific-humidity field via `qa · 2000 kg/m²`, the *same* lower-layer
    water-mass scale `atmosphere.py`'s own gated `enable_prognostic_column_water`
    path already uses (not a new, inconsistent conversion). EKE is P3's
    local steady state, not the full prognostic-transport-to-equilibrium
    loop (P3's own exit gate already closed that question separately).
  - **Energy-closure scope, an honest limitation**: (A40)'s diabatic source
    needs atmosphere-absorbed SW and net atmosphere LW, which do not exist
    as separable fields until stage P5 (radiation) replaces the current
    single-layer grey scheme. Rather than fabricate a SWa/LWa split from
    fields the model does not track, the real-state measurement reports the
    column-energy kernel's conservation contract under zero external
    forcing only (pure advection + diffusion) — the same scope P3's own
    "pure-transport conservation check" used for K. The nonzero-source
    algebra (the actual point of A40) is fully covered by hand-computed
    unit tests instead. Wiring a real SWa/LWa/SH assembly is P5's/P6's job,
    not P4's.
  - **Exit-gate measurement, full 512x1024 resolution, both seasons
    (2026-08-18)**: global precipitation lands inside the target band with
    no target ever supplied — **DJF 2.70 mm/day, JJA 3.00 mm/day** — both
    close to the incumbent row-target allocator's own output on the same
    state (DJF 2.80, JJA 2.91 mm/day), a reassuring cross-check that the
    two very differently-derived mechanisms agree in aggregate even though
    P4's `P` is a pure local residual with no prescribed profile anywhere in
    its call chain. Conservation: water relative residual 5.4e-7 (DJF) /
    6.1e-6 (JJA); energy (pure-transport) relative residual 2.2e-16 (DJF) /
    0.0 (JJA) — both at or below `column_water.py`'s own established
    precision standard. A coarse 64x128 screen (run first, to catch bugs
    cheaply) gave consistent numbers (DJF 1.98, JJA 2.23 mm/day, exact
    water conservation), confirming the result is not a resolution
    artifact. Full test suite: no regressions (see below).
  - **What remains explicitly out of scope**, unchanged in spirit from P2/P3's
    own admission language: wiring this branch's `P`/`Ta` into the supported
    climate path, per-surface-type flux sharing beyond the `T2m`/`q2m`
    diagnostic formulas themselves (a real coupling question the SST-land
    D3 precedent already flagged as its own separate effort), and the real
    SWa/LWa energy-budget measurement that needs P5. All are P5/P6's job,
    not P4's — P4's own stated exit gate (conservation + raw global P range,
    no target) is unambiguously met.
- **P5 — Radiation upgrade** (A6/A7/A8): only after P1 gives it real profiles.
  Cloud scheme, then two-band SW + 15-level two-stream LW. *Exit*: TOA fluxes
  vs the paper's validation figures; ECS moves off the structural ~1.8 K floor.
  - **First sub-deliverable complete (2026-08-19): the cloud scheme** (A61)-
    (A68) as pure functions in `sesam_radiation.py` — (A62) humidity/vertical-
    velocity cloud fraction, (A63)-(A64) effective cloud-level vertical
    velocity (mean 700 hPa + synoptic + orographic terms), (A65)-(A66)
    inversion/low-cloud fraction with freeze-dry factor, (A61) combination,
    (A67) cloud top height, (A68) cloud optical thickness. Gated
    `enable_sesam_radiation` (default False, **not wired into the supported
    path**), guarded by `testing/test_sesam_radiation.py`. Equations verified
    against the source PDF (500 dpi PyMuPDF render, poppler unavailable) and,
    where the paper's notation was ambiguous, against the reference
    implementation's `src/atm/clouds.f90`/`time_step.f90` (read-only, per
    section 5). Two findings recorded in the module docstring: (A65)'s un-
    numbered `r*` symbol is exactly stage P4's `surface_relative_humidity_star`
    (no new physics needed), and (A65)'s "when r*>ra" phrasing describes a
    continuous ramp in the reference implementation, not a hard branch —
    the same "prose reads as a threshold, the real formula is continuous"
    pattern already found once for (A44) at P4. Follows the published Table
    A5 constants only; the live reference repo's namelist has since drifted
    from them (e.g. `c_cld_5=0.75` there vs the paper's printed 0.5) and is
    not a citable source under the section 6 calibration-window policy.
  - **Second sub-deliverable complete (2026-08-19): shortwave radiation**
    (A69)-(A105) as pure functions in `sesam_shortwave.py` — (A79)-(A80)
    atmospheric scattering albedo, (A81)-(A82) cloud albedo, (A87)-(A89)
    water-vapour/aerosol transmission, (A94)-(A105) the shared absorber-mass-
    path structure (water and aerosol are the literal same formula), (A75)-
    (A78) planetary albedo and (A83)-(A86) surface transmission (both via one
    shared two-stream adding-method combinator), and the (A69)-(A74) band/sky
    combination, assembled end to end by `shortwave_radiation()`. Guarded by
    `testing/test_sesam_shortwave.py` (26 tests). Verified against the source
    PDF (500-600 dpi renders) and, for two genuine paper transcription
    errors, against `src/atm/swr.f90`/`constants.f90` (read-only, per section
    5): (A87)/(A88)'s visible/IR band labels are swapped in the published
    PDF (water vapour absorbs in near-IR, not visible/UV — corrected to match
    physics and the reference code), and (A97)'s cloud-thickness term has a
    sign error (printed `e^{+Dcld/Hq}`, unbounded; corrected to the reference
    implementation's bounded `e^{-Dcld/Hq}`, consistent with the paper's own
    neighbouring `f_exp` definitions). Confirmed no ozone climatology is
    needed anywhere in the shortwave scheme — (A90)/(A91) are fixed constants
    (0.96/1), not a field — so the ozone-climatology design question raised
    at P5's scoping applies only to the (A106)-(A117) longwave stage.
  - **Third sub-deliverable complete (2026-08-19): longwave radiation**
    (A108)-(A116) as pure functions in `sesam_longwave.py` — (A110)-(A112)
    water-vapour/CO2/ozone transmission, (A113) cloud transmission (only
    inside cloud layers), (A108)/(A109) combination, (A114)-(A116) absorber
    mass paths, and the (A106)/(A107) flux-profile assembly discretized as a
    Riemann-Stieltjes sum over the level grid. Gated `enable_sesam_radiation`
    (same gate as clouds/shortwave; default False, **not wired into the
    supported path**), guarded by `testing/test_sesam_longwave.py` (26
    tests) including two structural boundary-condition checks derived
    directly from (A106)/(A107) themselves (discretized downward flux
    vanishes at the top of atmosphere; discretized upward flux at the
    surface level equals the surface's own blackbody emission exactly) — a
    stronger correctness guarantee than a hand-value check alone, since both
    hold for *any* transmission matrix. The absorber-mass integrals
    (A114)-(A116), which the GMD paper states only as an abstract integral
    with no worked discretization, use the exponential reference pressure
    profile stage P1 already provides ((A1), exact for well-mixed CO2, since
    P1's own profile literally is that exponential; trapezoidal quadrature
    against P1's real humidity profile for water vapour). This closure
    method is independently verified against a second citable source located
    with the user's direct help: **Petoukhov, Ganopolski & Claussen (2003),
    PIK Report No. 81** — SESAM's direct scientific ancestor (POTSDAM-2 is
    CLIMBER-2's atmosphere module) — whose §6.1.2 confirms in prose that its
    own absorber-mass integral (the same integral as A114-A116) is evaluated
    "on the assumption that the vertical profiles are quasi-exponential for
    pressure and air density." This resolved a real scope question raised
    mid-implementation: the reference Fortran's absorber-mass closure
    initially looked sourced only from an inaccessible internal report methodology
    rather than the published paper, which would have sat closer to the
    project's "written from the paper" licensing line than any prior P5
    disambiguation; the PIK report — found by the user, not fetchable by
    Claude directly — closes that gap with a real citation. One open item
    remains flagged in the module docstring (not blocking): CO2's ppm-to-
    column-mass conversion is a standard atmospheric-chemistry calculation
    but not verified against the paper's specific "cm" absorber-mass
    convention the way the water-vapour identity is. Ozone is confirmed to
    need a real climatology here (unlike shortwave) and remains an accepted
    external input — the constant/zonal-climatology/real-dataset decision is
    still open.

  - **Fourth sub-deliverable and exit gate complete (2026-08-19): the (A10)
    tropopause radiative closure, the ozone-climatology decision, and the
    TOA-flux exit-gate measurement — P5 is now closed end-to-end.**
    `sesam_tropopause.py` closes P1's input-starved `tropopause_tendency`
    with a real `Rstr,net`: a read-only disambiguation from the reference
    Fortran's `atm_model.f90` `rb_str` (the paper states `Rstr,net` only in
    prose — "the balance of longwave radiation and the shortwave radiation
    absorbed by ozone" — with no worked formula), reproduced as
    `net_LW(top) − net_LW(tropopause) + frac_vu·(1−I_O3,vu)·SW_down,top`,
    using the *published* Table A5 `I_O3,vu=0.96` rather than the Fortran's
    own uncited, undocumented `0.02` constant (per the section 6
    calibration-window policy). `c1tp`'s previously-flagged per-day unit
    folding is resolved from the paper's own Conclusions text ("the use of a
    daily time step for most processes", page 5924): `tropopause_tendency`'s
    output is metres/day, scaled by `dt_days` like every other SESAM stage.
    The ozone-climatology decision lands on the "constant" branch of the
    three-way choice raised at P5's scoping: a Gaussian stratospheric layer
    (25 km peak, 7 km half-width — textbook shape, not fitted) normalised to
    a 300 DU global column via the standard DU→kg/m² conversion — SESAM's
    real input (a prescribed 3-D time-varying CMIP6 field) is out of reach
    without external data this project does not have, and the reference
    Fortran confirms there is no simpler built-in fallback either (`atm%o3`
    is allocated but populated from an external boundary module this repo
    lacks). `sesam_longwave.py` also gained the full `longwave_radiation()`
    assembly (clear/cloudy `(N,N,H,W)` transmission matrices, sky-combined —
    the piece every other P5 sub-deliverable needed but none had built), and
    `sesam_vertical.py` gained the small (A3)/(A4) `air_density_profile` LW
    absorber-mass integrals need. 12 new tests in
    `testing/test_sesam_tropopause.py`, 3 more in `testing/test_sesam_longwave.py`
    for the new assembly; full repo suite re-confirmed clean.

    **Exit-gate measurement (2026-08-19, real 512×1024 saved state, annual
    mean): NOT passed at a tight tolerance, with a decisive decomposition**
    (`scripts/diagnose_sesam_toa.py`, scored against Table 1's own CLIMBER-X
    and Wild et al. 2013 observational-mean columns, both now in
    `sesam_reference.TABLE_MAIN_ENERGY_BUDGET`). TOA solar-down is exact
    (340.3 vs 340.2/340.0 W/m², as expected — pure orbital geometry,
    independent of SESAM). TOA solar-up is under-predicted by 41%  (60.1 vs
    102.2/100.0 W/m²), TOA thermal-up (OLR) over-predicted by 16-17% (275.1
    vs 237.6/239.0 W/m²) — both trace to one shared, identified cause: this
    diagnostic reuses the *saved state's own* `cloud_cover` field (real data,
    not fabricated, but out of P5's own scope to fix — clouds were already
    measured at sub-deliverable 1) to avoid regridding the P2/P3 wind/EKE
    chain onto the stratosphere-reaching level grid OLR needs, and that
    field's actual global mean is ~9.8% versus the ~60-70% observed
    climatology — too little cloud to reflect enough shortwave or trap
    enough longwave. The radiative-transfer machinery this stage was built
    to close (A69-A105, A106-A117) is exercised end-to-end and produces
    values in the right physical regime, not a decomposition failure of its
    own; the input-cloud-fraction gap is the dominant, identified driver,
    consistent with the "no target ever supplied, no calibration window
    spent yet" framing P4's own exit gate used. The (A10) closure was
    exercised (not iterated to equilibrium, same "local closure" scope P3's
    EKE exit gate used): `Rstr,net` averaged −17.8 W/m² (net stratospheric
    radiative loss under this state) and the tropopause moved from the 12 km
    initial guess to a ~13.2 km one-step update, the physically correct
    sign per (A10)'s own `−c1tp·(Rstr,net+S)` form.
- **P6 — The calibration window** (§6), then the standard 128×256 five-year
  promotion checkpoint. **Started 2026-08-19.** §6 point 2's trigger
  ("when the branch is feature-complete, §7 stage P4") was itself stale --
  written before P5 (radiation) existed as its own stage -- and is corrected
  to stage P5 as part of starting this stage; P5's 2026-08-19 completion is
  what actually unblocks P6.

  Wiring five independent, unit-tested-but-never-live kernel stages (P1-P5)
  into the single ~9000-line `simulate_step()` is itself comparable in size
  to building them, so P6's own wiring is sub-staged the same way P1-P5 were,
  rather than touched all at once:
  - P6b: column energy/water closure (P4) first -- replaces the row-target
    precipitation allocator, the mechanism-replacement §1/§4 identify as the
    actual motivation for this whole adoption.
  - P6c: SLP/wind (P2, zonal-only, matching P2's own admitted verdict) + EKE
    transport (P3), replacing the bridge wind/EKE placeholders P6b uses.
  - P6d: radiation (P5), replacing the bridge diabatic-source placeholder.
  - P6e: full-chain sanity run, then the §6 bounded Table-A constant sweep
    against the full standard gate set.

  - **P6b sub-deliverable complete (2026-08-19): the column-closure branch is
    now live-wired, gated, and verified inert on the default path -- not yet
    verified as a climate improvement.** New `sesam_coupling.py` adapts
    `PlanetState`'s live fields to `sesam_thermo.py`'s (stage P4) pure
    kernels and is called from `simulate_step()` (`simulate.py`, immediately
    before final state assembly) only when
    `PlanetParams.enable_sesam_column_closure` is True; a new
    `PlanetState.sesam_column_water_mm` field carries the (A42) prognostic
    column-water depth `Qq`, defaulting to `None` (lazy-initialized on first
    use, old saves unaffected, matching every other gated field's existing
    convention). Verified byte-identical-in-effect on the default path
    (`state.sesam_column_water_mm` stays `None`, `sesam_coupling.py` is never
    called) by both a manual A/B smoke run and a new
    `test_simulate_step_default_path_unaffected_by_sesam_gate` regression
    test; full SESAM test suite 249 passed/1 skipped (`testing/ -k sesam`),
    plus 7 new tests in `testing/test_sesam_coupling.py`.

    **Three documented bridges** (module docstring, same "reuse a real field
    from elsewhere, don't fabricate" discipline P4/P5's own diagnostic
    scripts already used twice), each earmarked for replacement by its own
    later P6 sub-stage: (1) wind -- the supported model's own already-computed
    `wind_u`/`wind_v` drives SESAM's advection, not P2's own (not wired until
    P6c); (2) EKE -- a spatially uniform placeholder at P3's own validated
    global-mean local-steady-state value (240 m²/s², P3's 2026-08-18
    measurement), not the real prognostic `K` field (not wired until P6c);
    (3) diabatic source -- (A40) needs a real SWa/LWa split P5 hasn't
    supplied to the live loop yet (only P5's *diagnostic* scripts have it),
    so this stage instead drives Ta with a bulk sensible-heat-style relaxation
    toward the legacy model's own already-computed skin temperature `T*`
    (`(T* − Ta) / 1 day`) -- (A41)'s own `T2m=(Ta+T*)/2` relationship read as
    a forcing, not an invented mechanism, but a real approximation nonetheless
    (not wired to the real split until P6d).

    **Real bug found and fixed (2026-08-19): multi-day outer steps applied
    the diabatic bridge's 1-day relaxation *rate* for the whole span in one
    Euler call, with no substepping.** A 10-day DAILY-mode smoke run at
    32×64 stayed finite (no NaN/Inf) but showed a small, persistent number of
    cells pinned at Ta's defensive ±[150, 350] K clip bounds near the equator
    every day. Root-causing this (rather than papering over it with a damping
    term, per this project's stop-condition discipline and the
    `docs/VERTICAL_THERMODYNAMIC_CLOSURE.md` precedent) escalated fast once
    tested at the actual §6 target resolution: a real 64×128/MONTHLY
    (`days=30` per `simulate_step` call) `real_terrain_validation.py` run
    showed `air_temperature` going **globally uniform at exactly the 350 K
    clip bound one month, then exactly 150 K the next**, oscillating every
    subsequent call -- and even the legacy skin temperature field
    (`state.temperature`, which this branch never writes) collapsed to an
    unphysical ~250 K global mean over the 1-year spinup, via the ice-albedo/
    radiation feedback the corrupted air temperature drove in the untouched
    downstream code. The cause: `sesam_coupling.sesam_column_closure_step`'s
    diabatic bridge (module docstring bridge 3, `(T*-Ta)/1 day`) is a
    K/day *rate*, but `simulate_step` was calling it once per outer step with
    `dt_days=days` directly -- exactly the outer-step-vs-per-call-physics-
    timescale mismatch `_PRECIP_SUBSTEP_DAYS` already exists to prevent for
    precipitation (a 30-day call applies a 1-day-calibrated rate for 30 days
    in one shot, overshooting by ~30x). **Fixed** by substepping the SESAM
    call at 1-day cadence inside `simulate_step`
    (`_SESAM_COLUMN_CLOSURE_SUBSTEP_DAYS`), mirroring the pre-existing
    `_generate_precipitation_substepped`/`_evolve_wind_substepped` idiom
    exactly rather than inventing a new one. New regression test
    `test_simulate_step_multiday_call_does_not_saturate_clip_bounds`
    (`testing/test_sesam_coupling.py`, 8 tests total now); full SESAM suite
    250 passed/1 skipped after the fix.

    **A second, smaller finding survives the fix, isolated but deferred to
    P6c rather than chased further here.** Feeding `evolve_column_energy` the
    real post-spinup legacy `wind_u`/`wind_v` with zero diabatic source
    produced 264/2048 cells at the hot clip bound after 10 days; the same
    wind field with only the diabatic term active (zero wind) produced *zero*
    hot cells and stayed in a sane 230-303 K range throughout -- isolating
    the cause to bridge 1 (the legacy, non-SESAM-native wind field), not
    bridge 3 (the diabatic relaxation) or a substep/CFL discretization defect
    (ruled out separately: sweeping the wind-only case's substep size from
    1.0 down to 0.05 days did not resolve it, and if anything mildly
    worsened it -- consistent with genuine concentration under a persistently
    convergent velocity field, the same *real, not-a-bug* behaviour P2's
    "azonal channel amplifying the saved state's sharp regional fields" and
    P3's "pure advection can legitimately concentrate a field far above its
    initial peak" findings already documented for their own bridge inputs).
    This is exactly what P6c (replacing the legacy-wind bridge with SESAM's
    own P2/P3-native wind/EKE) exists to fix, so it is not chased further
    inside P6b.

    **First real measurement, full 64×128/MONTHLY, 1yr spinup + 1yr eval
    (`real_terrain_validation.py` defaults, post-fix):** global mean
    temperature 289.96 K (gate off: 291.11 K) and ocean temperature 292.67 K
    (gate off: 292.76 K) -- both close, confirming the catastrophic collapse
    is gone and the branch is now numerically well-behaved at the target
    resolution. But **Köppen group accuracy drops to 0.480 (κ 0.350) from the
    gate-off baseline's 0.674 (κ 0.587)**, class accuracy 0.215 from 0.389,
    and polar precipitation comes out 6x too high (2.29 vs the gate-off run's
    0.38 mm/day, both vs the same reference). **This is not a surprising or
    blocking result -- it is the expected state of a branch still running on
    all three documented placeholder bridges** (legacy wind including the
    just-isolated convergence-zone mismatch, P3's uniform-EKE placeholder,
    and the crude skin-temperature diabatic relaxation), not yet SESAM's own
    native wind/EKE/radiation. No promotion claim is made here; this is the
    first honest baseline number for what P6c and P6d need to improve on, in
    the same spirit as P2's "not passed, decisive decomposition" and P4's
    "no target ever supplied" framings.

  - **P6c sub-deliverable complete (2026-08-19): SESAM's own P2 (zonal-only
    SLP/wind) and P3 (local-steady-state EKE) are now live-wired, replacing
    P6b's bridges 1-2 -- numerically stable, but a real climate-impact
    finding, not yet an improvement.** New `sesam_wind_coupling.py` mirrors
    the already-validated `scripts/diagnose_sesam_wind.py`/
    `scripts/diagnose_sesam_synoptic.py` chain exactly (P1 vertical structure
    -> two-pass SLP<->wind closure -> zonal-only extraction -> local-
    steady-state EKE via `sesam_synoptic.compute_synoptic`), sourced from
    live `PlanetState` fields instead of a saved `.npz`. Gated on
    `PlanetParams.enable_sesam_dynamics` *in addition to*
    `enable_sesam_column_closure` -- column closure alone is unchanged and
    still uses P6b's bridges. Recomputed once per outer `simulate_step` call
    (not per column-closure inner substep), the same "hold the slower field
    fixed across faster substeps" simplification the legacy precipitation
    substep loop already makes. `sesam_coupling.sesam_column_closure_step`
    gained an optional `eke_m2_s2` parameter (defaults to P6b's placeholder
    when omitted, so P6b-only callers are unchanged) to accept the real P3
    field. 6 new tests (`testing/test_sesam_wind_coupling.py`); full SESAM
    suite 256 passed/1 skipped.

    **Explicit scope narrowing, not the full P3 prognostic K transport.**
    P3's own (A52) advection+diffusion transport of K needs a persistent
    per-cell state field and a short coupling cadence to stay numerically
    sane (Sec7 P3, 2026-08-18) -- real, validated machinery, but P3 itself
    delivered it as a *second* sub-deliverable after first shipping the
    local-steady-state closure alone. This live-coupling debut mirrors that
    same staging; full prognostic K transport is a documented follow-up, not
    attempted here.

    **First real measurement, full 64×128/MONTHLY, 1yr spinup + 1yr eval,
    both gates on:** global temperature 289.72 K (P6b-only: 289.96 K,
    gate-off baseline: 291.11 K) -- close, confirms P6c stays numerically
    well-behaved at the target resolution, same as P6b. Köppen group
    accuracy is essentially unchanged from P6b-only (0.477 vs 0.480, both
    well below the 0.674 baseline) but **precipitation collapses further**:
    global mean 0.64 mm/day (P6b-only: 1.99, gate-off: 3.00 -- Earth's real
    mean is ~2.7-3.0), tropical Köppen land coverage falls to 0.9% (P6b-only:
    17.7%, gate-off: 20.3%) while arid coverage more than doubles to 56.1%
    (P6b-only: 23.2%, gate-off: 30.2%). **The cause is not a new bug --
    it is P2's own already-documented exit-gate finding surfacing as a
    downstream climate effect**: the zonal-only SESAM wind's own validated
    mean surface speed (~2.2-2.4 m/s, Sec7 P2 2026-08-16/17/18 entries) is
    genuinely weaker than the legacy generator's (and NCEP's own ~3.9-4.0
    m/s), so the (A44) moisture-convergence term this branch's precipitation
    depends on has less advective convergence to work with -- P2's wind
    speed shortfall was already known and documented as unresolved; this is
    the first time it has been shown to propagate into the hydrological
    cycle rather than just a wind-pattern-correlation score. **No promotion
    claim is made; this is not chased further inside P6c** -- it is either a
    genuine input for the §6 calibration window (SESAM's own Table A
    convergence-efficiency/turnover constants may partly compensate for a
    weaker but more physically-derived wind) or evidence that P2's zonal-only
    wind itself needs revisiting before P6 can close, a judgement call for
    the §6 sweep itself rather than a P6c-stage action per this project's own
    staged discipline.

  - **P6d sub-deliverable built and wired (2026-08-19): SESAM's own (A69)-(A117)
    shortwave/longwave radiation and (A10) tropopause closure now replace
    P6b's bridge 3 (the ``(T*-Ta)/1-day`` bulk relaxation) -- but this stage's
    own first real multi-day exercise surfaced a genuine, unresolved
    numerical fragility that keeps it short of a clean pass, reported here
    rather than papered over.** New ``sesam_radiation_coupling.py`` mirrors
    ``scripts/diagnose_sesam_toa.py``'s already-validated chain (P1 vertical
    structure on the stratosphere-reaching 10-level grid, (A6) cloud
    geometry, (A69)-(A105) shortwave, (A106)-(A117) longwave, (A10)
    tropopause), sourced from live ``PlanetState`` fields and the *current*
    day of year rather than an annual mean. ``sesam_coupling.py``'s
    ``sesam_column_closure_step`` gained optional
    ``sw_absorbed_w_m2``/``lw_net_w_m2`` parameters: when both are supplied
    it assembles the real (A40) source via
    ``sesam_thermo.diabatic_heating_rate_k_day`` (Pw/Ps split from this
    step's own (A44) precipitation by the same air-temperature rain/snow
    ramp ``simulate.py``'s legacy snow-pack model already uses; SH reuses
    ``land_surface.py``'s own bulk-aerodynamic sensible-heat formula), and
    the water step now runs before the energy step (a pure reordering --
    water never depended on the energy step's output -- needed so the real
    source can consume this step's own precipitation). Gated on
    ``PlanetParams.enable_sesam_radiation`` *in addition to*
    ``enable_sesam_column_closure``; 15 new tests
    (``testing/test_sesam_radiation_coupling.py``,
    ``testing/test_sesam_coupling.py``); full SESAM suite 266 passed/1
    skipped.

    **A real cadence bug found and fixed during development, the same class
    as P6b's own multi-day overshoot but via a different mechanism.** An
    initial version held SWa/LWa fixed for the whole outer ``simulate_step``
    call (mirroring P6c's own wind/EKE convention, justified there because
    wind/EKE only scale advection/diffusion rather than acting as a source
    term). This stayed numerically fine for single-day DAILY calls but drove
    ``air_temperature`` to NaN within 2-3 MONTHLY (30-day) calls on a real
    16x32 smoke run: holding a *diabatic source* fixed for many consecutive
    1-day Euler substeps removes the Stefan-Boltzmann negative feedback (LW
    emission rising with Ta) that keeps a radiative forcing self-limiting,
    so a stale forcing overshoots the same way P6b's own un-substepped
    bridge did. **Fixed** by recomputing the full radiation chain every
    1-day column-closure substep instead of once per outer step -- expensive
    (a full longwave transmission-matrix solve per simulated day, not per
    outer call), left as a documented performance follow-up for P6e rather
    than traded away silently.

    **A second, deeper finding survives that fix, unresolved and not chased
    further inside P6d per this project's own stop-condition discipline.**
    Even with per-substep radiative recompute, a sustained multi-day
    MONTHLY run still diverges (confirmed via a direct, isolated repro,
    ``testing/test_sesam_vertical.py``'s
    ``test_near_surface_lapse_large_cold_land_contrast_produces_unphysical_profile``):
    stage P1's (A9) cold-land near-surface lapse rate
    (``sesam_vertical.near_surface_lapse``) has a **deliberately unbounded**
    inversion term (``c6_Gamma * (Ta - T*)``, already covered by
    ``test_near_surface_lapse_branches_and_caps``'s own
    "not lower-capped" assertion -- paper-faithful behaviour per that
    branch's docstring, not a transcription gap). When the diabatic-source
    bridge/radiation feedback loop lets Ta drift tens of Kelvin from the
    legacy skin temperature T* (observed: a 24 K gap alone produces a
    temperature-profile point above 600 K when integrated over the ~1.5 km
    near-surface layer), this formula's own unbounded design -- never
    exercised under live iteration before, since every prior SESAM stage
    used either smooth synthetic test fields or a single diagnostic
    measurement against an already-near-equilibrium saved state -- blows up
    the whole vertical structure, cascading into the longwave scheme's
    transmission matrix and producing LW fluxes many orders of magnitude
    past anything physical. **This is not fixed here.** It is a paper-
    faithful P1 formula's genuine edge case under a coupling architecture
    that can (for now) let Ta wander further from T* than the formula was
    evidently designed to tolerate -- altering already-tested P1 physics to
    add a symmetric bound is a plausible fix but not one to make
    unilaterally without verifying against the source paper text, per this
    project's own section 5/6 discipline (read-only reference-repo policy,
    no silent physics changes). Left as an explicit decision point for
    whoever picks up P6e: verify against the paper and bound
    ``near_surface_lapse`` symmetrically, add a coupling-side constraint
    that keeps Ta closer to T*, or accept this as a bounded-scope limitation
    the calibration window must design around.

  - **P6e (full-chain sanity run) attempted 2026-08-19: NOT passed, and decisively
    so -- the P6d finding above is confirmed to be far more severe than its own
    write-up suggested, not a new root cause.** All three gates
    (``enable_sesam_column_closure``/``enable_sesam_dynamics``/``enable_sesam_radiation``)
    together at the standard ``real_terrain_validation.py`` config (64x128,
    MONTHLY, 1yr spinup + 1yr eval): ``air_temperature`` collapses to the 150 K
    clip floor almost everywhere (mean 153.3 K, vs a sane ``state.temperature``
    skin mean of 281.4 K) within the **first simulated MONTHLY step (30 days)**,
    not over "many days" as P6d's own smoke-run framing implied -- the smaller
    16x32 test grid used there happened to collapse more gradually than the
    real DEM's actual elevation/latitude distribution does at the target
    resolution. By the third outer step (cycle 2 of the spinup), the now-
    thoroughly-corrupted temperature field feeds into stage P2's own
    ``sesam_dynamics.hadley_geometry`` (called every outer step by
    ``sesam_wind_coupling.py`` when ``enable_sesam_dynamics`` is on) and raises
    a hard ``ValueError`` ("the Hadley-cell boundary is not bracketed") --
    P2's own deliberate guard for "a climate too distorted for the cell
    decomposition" (its docstring's own words), tripped by exactly the
    distortion P6d's ``near_surface_lapse`` finding produces. **This is the
    same root cause as the P6d entry above, not a second bug**: confirmed by
    an isolated control run (P6b+P6c only, radiation gate off, same config)
    completing all 12 spinup cycles cleanly in 172.7s with no exception --
    the divergence and the downstream Hadley-geometry crash both require
    ``enable_sesam_radiation`` to be on.

    **Consequence for this stage's own remaining scope**: the §6 bounded
    Table-A constant sweep this stage exists to run cannot proceed on top of
    a full chain that does not survive its first simulated month. Per this
    project's own stop-condition discipline (below), this stage pauses here
    rather than widening into a workaround; the P6d entry's own three listed
    options (verify against the paper and bound ``near_surface_lapse``
    symmetrically, add a coupling-side Ta-drift constraint, or design the
    calibration window around this as a hard limitation) remain the explicit
    decision point, now with the added information that the failure is not a
    rare, slow-drift edge case but the default outcome of the standard
    64x128/MONTHLY target config within one simulated month.

  - **Follow-up investigation (2026-08-20), no code changed.** Two questions
    raised by the P6e entry above are now resolved:

    1. **Is (A9) actually a paper transcription bug, leaving a clamp as a
       legitimate fix?** No. The published equation was read directly from
       the CLIMBER-X GMD paper's Appendix A (`10.5194/gmd-15-5905-2022`,
       page 5926): `near_surface_lapse`'s four branches and both upper-only
       caps (7.5e-3 ocean / 10e-3 land+ice) match verbatim, and the paper's
       own text states the unbounded cold-land inversion is intentional
       ("Eq. (A9) allows SESAM to reproduce near-surface inversions which
       are important for surface climate"). This removes "verify against
       the paper and bound `near_surface_lapse`" from the P6d entry's list
       of options with full confidence — it is now a confirmed paper
       deviation, not an open question, and should not be revisited without
       new information.
    2. **Why does `Ta-T*` open a 24 K+ gap in the first place?** The
       regression repro (`test_near_surface_lapse_large_cold_land_contrast_produces_unphysical_profile`)
       uses `qa=0.0005 kg/kg` — a near-desert-dry column — which lines up
       with this same stage's own P6c finding above: the zonal-only SESAM
       wind's ~2.2-2.4 m/s mean (vs. the legacy generator's ~3.9-4.0 m/s)
       starves (A44)'s moisture convergence, collapsing global mean
       precipitation to 0.64 mm/day and expanding arid land coverage to
       56.1%. A dry column has weak water-vapour greenhouse trapping, so its
       near-surface air radiatively cools much faster than a moist column's
       under clear skies — the same desert-nighttime-crash mechanism this
       project has hit and fixed repeatedly in the legacy model (e.g. the
       land-cap thermal-low and desert-evapotranspiration fixes, both
       pre-SESAM). **This alone is not sufficient to blow up the profile**:
       the P6e control run (P6b+P6c only, same dry land, same weak wind,
       radiation gate off) completes all 12 spinup cycles cleanly. What
       changes when P6d's gate is added is not the land's dryness but the
       *coupling*: P6b/c's bulk `(T*-Ta)/1-day` bridge is a strong daily
       pull of Ta back toward T*, which structurally caps how far the two
       can separate regardless of how dry the column is. P6d's real SWa/LWa
       assembly removes that artificial pull, so a dry column's genuinely
       larger radiative swing is now free to open the gap (A9) was not
       designed to tolerate. **Conclusion: the P6d/P6e divergence is
       downstream of the already-documented, already-deferred P2 wind-speed
       shortfall, not a defect newly introduced by P6d's own radiation
       wiring.** This reframes the decision point: a fix aimed only at the
       Ta/T* coupling (options (c)/the profile-handoff bound) would be
       patching a symptom of P2's wind shortfall rather than addressing it,
       whereas revisiting P2's zonal-only wind speed (already flagged in
       the P6c entry as "a judgement call for the §6 sweep") might close
       this gap at its actual source. No fix has been applied — this is
       presented as a decision point, per this stage's own discipline
       against silent physics changes.

    3. **Follow-up (2026-08-20): why is the live wind ~2.2-2.4 m/s specifically,
       vs NCEP's 3.9-4.0?** Traced to a concrete, narrow gap, not a deeper flaw
       in P2's zonal-cell physics. This same stage's own P3 sub-deliverable
       (2026-08-16 entry above) already documented that SESAM's Earth-realistic
       "total surface wind" is `Us = sqrt(us^2 + vs^2 + Usyn^2)` (A58) — the
       zonal-cell wind *plus* the synoptic/storm-track gustiness term `Usyn`
       (A56, ≈7 m/s) — not the zonal-cell wind alone (that entry's own worked
       total: "zonal-cell wind (2.2-2.4) + synoptic gustiness (≈7) ≈ 8 m/s").
       `sesam_synoptic.compute_synoptic` already computes this exact quantity
       and returns it as `SesamSynoptic.total_wind_m_s` — but
       `sesam_wind_coupling.sesam_wind_and_eke_step` (P6c) discards it: its
       `SesamWindAndEke` return only carries `wind_zonal.surface_u_m_s`/
       `surface_v_m_s` (the pre-synoptic zonal components) plus
       `eke_m2_s2`, never `total_wind_m_s`. Everything downstream that reads
       `wind_u_m_s`/`wind_v_m_s` — (A44) moisture-convergence advection in
       `sesam_coupling.py`, and the bulk-aerodynamic evaporation/sensible-heat
       wind-speed terms in `sesam_coupling.py`/`sesam_radiation_coupling.py`
       (`wind_speed = sqrt(wind_u**2+wind_v**2)`) — therefore only ever sees
       the ~2.2-2.4 m/s zonal component, never the synoptic contribution.

       **This likely conflates two physically distinct uses that should not
       share one wind field.** Moisture *advection* should plausibly stay on
       the mean (zonal) flow — that is the correct field for transporting a
       tracer — but bulk-aerodynamic flux formulas (evaporation, sensible
       heat) are standard nonlinear functions of *wind speed*, where
       real-world/NCEP climatologies (themselves time-means of instantaneous
       speed, which does not cancel eddies the way a time-mean vector wind
       does) are physically closer to SESAM's own (A58) total than to its
       zonal-only component. Using the zonal-only magnitude for those bulk
       formulas would systematically understate evaporation and sensible
       heat versus what SESAM's own equations intend.

       **Resolved (2026-08-20): the "NCEP 3.9-4.0 m/s" figure used throughout
       is a mean-*vector*-wind speed, not the true mean wind-speed
       climatology, and comparing against the real product changes the
       verdict substantially.** `scripts/build_ncep_wind_reference.py`/
       `diagnose_sesam_wind.py` compute `sqrt(mean(u)^2 + mean(v)^2)` from
       NOAA's `uwnd.sig995.mon.ltm`/`vwnd.sig995.mon.ltm` — the *speed of the
       time-mean vector wind*, which is always ≤ the true mean of
       instantaneous speed (transient/storm-track eddies largely cancel in a
       component-wise time mean before the magnitude is taken). NOAA
       separately publishes the actual scalar-speed climatology,
       `wspd.sig995.mon.ltm` — its own metadata literally states "Long Term
       Mean Monthly Mean Wind Speed... from daily wind speed (from daily
       vector winds)," i.e. computed from daily |wind| before time-averaging,
       not from time-averaged components. This project had never downloaded
       it. Fetched and compared directly (area-weighted global mean, same
       grid handling as the existing reference builder):

       | | `sqrt(mean_u^2+mean_v^2)` (current comparison basis) | true `wspd` |
       |---|---|---|
       | DJF | 4.093 m/s | 6.536 m/s |
       | JJA | 4.152 m/s | 6.578 m/s |
       | ANN | 3.623 m/s | 6.513 m/s |

       The true wind-speed climatology is **~1.6-1.8x** the vector-mean
       figure this project had been treating as "NCEP wind speed." This
       resolves the inconsistency cleanly, in favor of the P3 reading:
       SESAM's zonal-only wind (2.2-2.4 m/s) is correctly compared against
       the *vector-mean* NCEP figure (3.9-4.2 m/s) — that comparison stays
       valid, and the ~40-45% shortfall there is real (a genuine, separate,
       already-documented P2 calibration question). But SESAM's own (A58)
       *total* wind (zonal + `Usyn`, ≈7.4-8 m/s per the P3 entry's worked
       quadrature sum) should be judged against the *true* `wspd` climatology
       (~6.5 m/s annual, ~6.5-6.6 seasonal) — and by that comparison, SESAM's
       total wind is already in the right ballpark, if anything a bit strong
       (~15-25% high), not "40% low." **This reframes the fix priority**: the
       dominant lever on the P6c precipitation collapse / P6d Ta-T* blowup
       chain is very likely wiring the already-computed but currently-discarded
       `SesamSynoptic.total_wind_m_s` into the bulk-aerodynamic evaporation/
       sensible-heat wind-speed terms (`sesam_coupling.py`,
       `sesam_radiation_coupling.py`) — roughly a 3x increase in the wind
       speed driving those formulas, now measurement-backed rather than
       speculative. The zonal-only wind should stay driving moisture
       *advection* (the correct field for transporting a tracer, and its own
       remaining ~40% shortfall is a separate, smaller, already-tracked
       question). (Raw file:
       `testing/reference_data/ncep_ncar_raw/wspd.sig995.mon.ltm.1991-2020.nc`,
       gitignored per this project's existing reference-data convention, not
       committed.)

    4. **Implemented (2026-08-20) and does NOT resolve the P6e blowup — a
       real, tested fix kept in the codebase, but a negative result for this
       specific chain.** `SesamSynoptic.total_wind_m_s` is now threaded
       through: `sesam_wind_coupling.SesamWindAndEke` gained a
       `total_wind_m_s` field (`sesam_wind_coupling.py`), and
       `sesam_column_closure_step` gained an optional
       `total_wind_speed_m_s` parameter (`sesam_coupling.py`) that drives
       only the bulk-aerodynamic evaporation/sensible-heat wind-speed terms
       when supplied, defaulting to the previous zonal-magnitude behaviour
       when omitted (`simulate.py`'s P6c call site now passes it whenever
       `enable_sesam_dynamics` is on). 2 new regression tests
       (`testing/test_sesam_coupling.py`) plus 1 extended test
       (`testing/test_sesam_wind_coupling.py`); full SESAM suite 268
       passed/1 skipped, no regressions. Verified live on the real DEM: mean
       total wind (6.37 m/s) is genuinely ~3.7x the zonal-only mean
       (1.73 m/s) the bulk formulas previously saw — the wiring is real, not
       a no-op.

       **Re-ran the exact P6e full-chain sanity check (standard
       64x128/MONTHLY, all three gates on) with this fix in place: it still
       fails at essentially the same speed and severity.** Cycle 0 (the
       first simulated MONTHLY cycle) already clips 63.6% of
       `air_temperature` to the 150 K floor; cycle 2 trips the same
       `sesam_dynamics.hadley_geometry` "Hadley-cell boundary is not
       bracketed" guard as the original P6e finding. **This falsifies the
       specific causal chain hypothesized in the 2026-08-20 part-1 follow-up
       above** (P2 wind shortfall → starved evaporation → dry land → weak
       greenhouse trapping → fast radiative Ta crash → (A9) blowup): if that
       chain were the dominant driver, a ~3.7x increase in the wind-speed
       term driving evaporation should have measurably slowed or prevented
       the collapse within the first simulated month, and it did not. The
       collapse happens too fast (within the first ~30 simulated days,
       cold-start) for a precipitation/soil-moisture-mediated pathway to
       have had time to act — pointing instead toward something in the
       first-step transient itself (e.g. an initial Ta/T* mismatch at cold
       start, before any climate has equilibrated) or a mechanism inside the
       P6d coupling not explained by upstream wind/moisture starvation.
       **The wind fix is kept** (it corrects a real, independently
       measurement-backed physics gap, is fully tested, and is inert on
       every path except the still-default-off `enable_sesam_dynamics`
       gate) but does not by itself unblock P6e. The original P6d/P6e
       decision point (verify/bound `near_surface_lapse`'s coupling
       exposure, bound the Ta-T* drift, or bound the profile handoff — see
       the P6d entry above) stays open; root-causing what actually happens
       cycle-by-cycle within that first simulated month (day 1 through day
       30) is the logical next diagnostic step, not yet done.

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
- **(A38) denominator: erroneous extra `u` factor, found and fixed
  (2026-08-17).** The article does not print a closed-form spectral solution
  for (A38) -- only the PDE and "solved... by Fourier expansion... using
  FFTW3" (confirmed by direct PDF read, pages 43-44 of
  `gmd-2022-56-manuscript-version2.pdf`; the final typeset PDF is 14 MB,
  over this session's fetch cap, and the HTML/XML full text both truncate
  before the appendix). Re-deriving the spectral solution from the literal
  printed equation (`u∂ζ/∂λ + βv + ζ/τe = -(f/HT)·0.4·u∂zs/∂λ`, a single
  Fourier/meridional mode, `ζ=-Kn²Ψ̂`, `v=i·kzn·Ψ̂`) gives
  `Ψ̂ₙ = (f/HT)·0.4·u·ẑsₙ / (u·Kn² − β − i·Kn²/(τe·kzn))` -- exactly
  `sesam_dynamics.charney_eliassen_slp`'s own docstring formula. The
  *implementation*, however, computed `Kn²/(τe·kzn·u)`: an extra `u` absent
  from the derivation and the docstring one line above it. Confirmed wrong
  two ways: dimensionally (the erroneous term is m⁻², every other term in
  the denominator is s⁻¹m⁻¹) and by measurement (a (lat, n, u500) sweep at
  realistic values shows the erroneous damping is ~1e-13, four orders of
  magnitude too weak to bound the resonance `u·Kn² = β` that exists for
  some low integer n at any positive u -- the actual mechanism behind the
  2026-08-16 exit-gate's ">100 m/s local winds"). Fixed by removing the
  extra `u`; real-terrain orographic SLP range fell from ±97-170 hPa to
  ±18-20 hPa (physically plausible; real stationary waves are ~10-20 hPa).
  The accompanying hand-Fourier test (`test_charney_eliassen_matches_hand_
  fourier_solution`) had the same extra `u` baked into its own hand
  computation and passed regardless -- it was verifying self-consistency
  with the bug, not correctness against the source; fixed alongside, plus a
  new dedicated resonance-boundedness regression.
- **`u500(ϕ)` row-to-row noise driving spurious meridional SLP gradients,
  found and fixed (2026-08-17).** Once the two bugs above were fixed, the
  exit gate still failed at specific real latitudes despite every 2-D input
  (thermal, orographic, zonal) being smooth there: `dp/dλ` was ~2e3 Pa/rad
  but `dp/dϕ` was ~1.8e5 Pa/rad -- a meridional, not zonal, problem. (A38)
  is solved independently per latitude row with no coupling between
  adjacent rows, so a smooth 2-D input does not guarantee a smooth output
  in ϕ; the actual driver was `u500` itself (the zonal-mean 500 hPa wind
  closing the two-pass SLP<->wind loop), measured oscillating -28, -16,
  -15, -18, -1, +13, +26 m/s across seven adjacent rows of the real
  saved-state DJF diagnostic. Combined with the westerly-only `+0.1 m/s`
  floor, each sign flip in that noise produced a near-discontinuous jump
  between a floored (near-zero) and unfloored (full-amplitude) response.
  Fixed with `sesam_dynamics.resolution_matched_profile` (a new 1-D
  latitude-only analogue of the existing 2-D `resolution_matched_field`,
  same box+linear-interp mechanism, no longitude to wrap) applied to `u500`
  before `compute_slp` passes it to `charney_eliassen_slp`; verified to turn
  the measured seven-row oscillation into a monotonic profile and the
  corresponding orographic SLP from an oscillating jump into a smooth ramp.

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
- **Pole `cosϕ` singularity in `vg(0)`/`ua`, found and fixed (2026-08-17).**
  (A17)-(A20) as printed divide by `cosϕ`; the paper's text excuses this
  only at the equator (the stated rationale for the `|f|` floor above). The
  same breakdown holds at the poles, where this module's discrete azonal
  SLP does not vanish in longitude the way the continuous field would --
  measured at >6500 m/s within one grid row of the pole in the DJF
  diagnostic, a second and independent defect from the (A38) fix above.
  Fixed by applying `thermal_wind_shear`'s own already-accepted
  `min(1, c_pol·cos²ϕ)` envelope to `vg(0)` and `ua` (`sesam_wind.py`) --
  the identical safeguard already used for the identical singularity in the
  shear integral, not a new mechanism and not the reference implementation's
  own unported "pole-half damping".
- **Equatorial breakdown of `ug(0)`/`va`, found and fixed (2026-08-17).**
  With the (A38) and pole fixes above in place, the exit gate still failed
  at real low-to-mid latitudes (~15-40°) driven by an *already-smooth* SLP
  gradient (~1.4 hPa/row, a genuine feature, not noise) divided by a small
  but not-floored `f` -- the paper's hard `|f|` floor (3e-5 geostrophic,
  1e-5 ageostrophic) bounds the denominator away from exactly zero but does
  not scale the response down as latitude approaches it, so a floored-but-
  still-small `f` still turns an ordinary gradient into 100+ m/s. The pole
  fix above only covered `vg(0)`/`ua`'s `cosϕ` term; `ug(0)` and `va` had no
  analogous protection at either singularity. Fixed by applying the *full*
  `min(1, c_eq·sin²ϕ)·min(1, c_pol·cos²ϕ)` envelope (both factors) to all
  four surface components via a new shared `_geostrophic_frame_damping`
  helper, also now used by `thermal_wind_shear` (behaviour there
  unchanged). `compute_wind` gained a `surface_damping` parameter,
  independent of `thermal_wind_damping`.
- **`c_eq` sensitivity swept, not tuned (2026-08-17).** A direct sweep of
  the equatorial-damping strength (0.5 to 5) on the saved 512x1024 DJF/JJA
  state found the exit-gate correlation/RMSE response is real but
  non-monotonic and season-inconsistent: `c_eq≈2` scores best for DJF while
  JJA prefers weaker values, and neither is uniformly best on RMSE. That
  pattern is the signature of a genuine constant-calibration question (§6),
  not a discrete defect worth chasing further at the P2 stage -- the shared
  5.0/3.0 value (`thermal_wind_shear`'s existing namelist constant) is kept
  as the principled default rather than hand-picked from this sweep.
- **Exit-gate result after all four fixes (2026-08-17/18):** saved-state
  (512x1024) full-chain surface wind vs NCEP: pattern correlation -0.011
  (DJF) / -0.121 (JJA), RMSE 6.74 / 6.82 m/s -- down from 73/113 m/s
  pre-fix (10.8x/16.6x), mean surface speed 5.46/5.05 m/s (NCEP 3.9-4.0).
  Still short of the generator's +0.34/+0.23, 2.8/3.6 m/s. Clipping the
  worst 1-10% of remaining outliers does not improve correlation (flat or
  slightly worse), so the gap is a broad pattern mismatch, not a few hot
  cells -- consistent with the `c_eq` finding above pointing at calibration
  rather than a fifth bug. See §7 P2's 2026-08-17/18 entry for the full
  verdict and admission status.

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
- **(A52) K-transport substep-count circularity, found and fixed
  (2026-08-18).** The first working version of `eke_diffusion_step` computed
  its sub-step count once, from the *initial* AT field, reasoning that pure
  diffusion with zero-flux boundaries and no source cannot raise a field's
  peak, so the initial state should bound every later sub-step's AT too.
  That argument is circular -- it assumes the discrete scheme is already
  stable in order to conclude it stays stable -- and a single
  marginally-insufficient sub-step early in a long chain (before AT could
  relax) is enough to seed a self-reinforcing oscillation, since a larger K
  in the next sub-step means a larger AT there too (AT depends on K).
  Confirmed on the real 512x1024 saved state: `RuntimeWarning: overflow
  encountered in multiply` inside `eke_diffusion_step`. Fixed by recomputing
  the required sub-step size from the *live* K at every sub-step (an
  adaptive `while` loop bounded by `diffusion_r_limit`, replacing the
  original fixed `n_sub` computed once up front).
- **(A52) K-transport polar-cell area/geometry bug, found and fixed
  (2026-08-18), a second and independent defect from the one above.**
  Adaptive re-estimation alone did not stop the real-state divergence: the
  CFL rate formula, `AT·(1/dx² + 1/dy²)`, implicitly assumes each cell's area
  is `~ dx·dy` (a flat-grid approximation). This fails specifically at the
  polar-cap row: the true cell area there is the actual (small) spherical-
  cap area, which shrinks toward zero much faster than the approximation
  assumes, while the face length used in the flux formula (`x_len =
  radius·dlat`) stays the same constant it is at every other row --
  understating the true self-loss fraction, and hence the true required
  sub-step restriction, specifically and only at the pole. Traced on the
  real state by logging the per-substep field maximum: it decayed smoothly
  (correct diffusion) for ~180 sub-steps, then suddenly diverged starting
  from row 0 (the north-pole row) outward. Confirmed as a formula error
  rather than an insufficient stability margin by tightening
  `diffusion_r_limit` 8x (0.4 → 0.05) on the same real state: the divergence
  persisted unchanged, which a genuinely marginal-CFL case would not do.
  Fixed by replacing the `1/dx² + 1/dy²` approximation with the exact
  per-cell self-loss coefficient derived from the actual face-length/area
  geometry (`x_len/(dx·area)` summed over all four faces, using the true
  `x_face_length_m`/`y_face_length_m`/`cell_area_m2` inputs rather than a
  `dx`/`dy` proxy). This is a real, general defect -- it would affect any
  spherical grid at any resolution close enough to the pole for the
  flat-grid approximation to break down, not an artifact of this one saved
  state -- and, incidentally, ~2x more conservative even on a *uniform*
  grid far from any pole (the exact formula sums four independent face
  terms; the approximation implicitly halved them), which shifted this
  stage's hand-value unit test's exact numbers (documented in
  `testing/test_sesam_synoptic.py`, re-derived and re-verified after the
  fix, not merely adjusted to make the test pass).
- **Advective transport reuses `column_water.evolve_column_water` directly**,
  not a re-implementation of donor-cell advection: K obeys the identical
  `dQ/dt = source − div(Qv)` conservative flux-divergence equation as column
  water, just carrying a different scalar with zero source/sink at that
  stage. This is a deliberate design choice (module docstring note 6), not
  merely convenience -- it means the advective term inherits
  `column_water.py`'s own extensive test coverage rather than introducing a
  second, independently-fallible implementation of the same numerics.
- **Pure advection can legitimately concentrate K far above its initial
  peak with no numerical error at all** -- a finding from tuning the
  diagnostic driver's coupling step, not a kernel bug. Measured directly:
  0.1 day of advection alone (zero diffusion, zero source) on the real DJF
  zonal-only wind field raised the field's max by ~45% (6202 → 9018);
  extrapolated to a naively large 5-day operator-split coupling step, the
  same real convergence zone piles K up to 131,112 (21x the pre-advection
  max) before diffusion ever gets a chance to relax it, which then demands
  an impractical number of diffusion sub-steps to smooth within that same
  call. This is the correct behaviour of the continuous advection equation
  under a persistently convergent velocity field with no counteracting sink
  in that sub-phase -- not a defect in `evolve_column_water` or in
  `eke_diffusion_step`. The fix is a diagnostic-driver choice, not a kernel
  change: `scripts/diagnose_sesam_synoptic.py`'s prognostic-transport driver
  uses a short 0.25-day coupling step (interleaving advection and diffusion
  frequently, the same reason real coupled climate models use a short
  coupling timestep between separately time-stepped processes) rather than
  a long one.
- **The 512x1024 polar diffusion stiffness lives in the zonal term, not the
  meridional one (found 2026-08-18) — overturns the initial hypothesis.**
  Going into the full-resolution performance investigation, the working
  assumption (by analogy with real ocean/atmosphere models' implicit
  *vertical* diffusion and semi-implicit *polar filters*) was that the pole
  stiffness would live in the meridional (north-south) direction. Direct
  measurement on the real grid overturned that: splitting
  `eke_diffusion_step`'s exact self-loss geometry into its zonal and
  meridional halves, the meridional term is flat across every latitude row
  (≈1.31e-9 m⁻² everywhere — a resolution-driven cost, present at every row
  equally, not a pole-specific one) while the zonal term alone grows from
  ≈1.31e-9 at the equator to ≈1.01e-6 at the pole row (770x), and is alone
  responsible for >99.8% of the pole's self-loss rate. The mechanism: the
  east/west face length `x_len = radius·dlat` in `spherical_transport_geometry`
  is the *same constant* at every row by construction (it does not shrink
  toward the pole), while true cell area shrinks toward the actual
  (vanishingly small) spherical-cap value there — so `x_len/area` (the
  zonal self-loss factor) diverges specifically at the pole, exactly the
  classic lon-lat-grid "pole problem" that motivates real circulation
  models' zonal-direction remedies (implicit zonal diffusion or Fourier
  polar filters), not a meridional one. This matters beyond just this one
  kernel: any future SESAM (or general lon-lat-grid) explicit horizontal
  diffusion/advection kernel at this project's headline 512-row resolution
  should expect the *zonal* direction to be the one that goes stiff near
  the poles, not the meridional one, unless its own face-length convention
  differs from `spherical_transport_geometry`'s.
- **Implicit zonal / explicit meridional (ADI-style) diffusion implemented
  as a new, separately-tested code path: `eke_diffusion_step_implicit_zonal`
  (`sesam_synoptic.py`, 2026-08-18).** Backward-Euler (unconditionally
  stable for positive conductances, so no CFL restriction at all) solved
  per row via a periodic cyclic tridiagonal system, batched over every row
  simultaneously (`_cyclic_thomas_batch`); the standard Sherman-Morrison
  reduction (Numerical Recipes §2.7 "cyclic"): zero the two corner
  coefficients, correct the two corner diagonal entries
  (`diag[0] -= gamma`, `diag[-1] -= alpha·beta/gamma` with
  `gamma = -diag[0]`), solve the resulting plain (non-cyclic) tridiagonal
  system twice via a batched Thomas algorithm (`_thomas_batch`) against the
  real right-hand side and a corner-correction vector, then combine via the
  rank-1 correction factor. Cross-checked during development against
  `numpy.linalg.solve` on the equivalent dense periodic matrix for
  arbitrary (non-constant-coefficient) rows to ~1e-14 relative error; a
  smaller instance of that same check is a permanent test
  (`test_cyclic_thomas_batch_matches_dense_solve`). The meridional direction
  stays exactly as `eke_diffusion_step` already does it (explicit,
  CFL-substepped, face-averaged AT) since it was never the stiff term. Kept
  fully separate from the validated explicit `eke_diffusion_step` (task
  requirement and good practice regardless): opt-in via
  `implicit_zonal_diffusion=True` on `eke_transport_step`/`evolve_eke`,
  the original function and its exact-value/planted-violation tests
  untouched. This removed the need for on the order of 1–20 million
  diffusion sub-steps (the explicit scheme's real, measured requirement at
  the diagnosed near-pole EKE magnitudes) down to ~530–540 sub-steps per
  0.25-day coupling step at full 512x1024 resolution — the meridional CFL
  bound that remains is an ordinary, resolution-driven cost, not a
  pole-specific one, per the finding above.
