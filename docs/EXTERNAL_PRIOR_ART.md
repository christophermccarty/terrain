# External Prior Art — What Already Exists That We Could Borrow or Replace

> Compiled 2026-08-05. Companion to `docs/ACCURACY_AUDIT.md`, which is the internal audit of what's
> wrong; this is the external survey of **what has already been solved elsewhere**, mapped onto the
> audit's open items.
>
> Ordering is by leverage against currently-open, currently-blocked audit items — not by how
> impressive the external work is. Several very famous models appear only in Tier 3 because they
> don't fit this project's interactive/desktop constraint.
>
> Every claim about *our* code below cites a file/line; every external claim cites a paper or repo.
> Re-verify both before acting, per process note 3.

---

## Executive summary

Four of this project's longest-running walls are hand-rolled versions of mechanisms that have a
standard, published, validated formulation:

| Audit item | What we built | What already exists |
|---|---|---|
| A5-OROG / A5-FOOTPRINT (box-scale W/L 1.57 vs Earth 3–6) | `_smear_along_wind` + two km-scale kernels + four ceilings | **Smith & Barstad (2004) linear theory** — one FFT, gives the upwind footprint *and* leeward spillover from two physical timescales |
| C1b (land seasonal amplitude/squareness) | `land_seasonal_amplitude=0.75`, `land_seasonal_amplitude_maritime`, `land_transport_gain` | **Force-restore** (Bhumralkar 1975 / Deardorff 1978) — 2 equations, gives damping + phase lag + moisture-dependent contrast |
| C1b / `_land_cap_1d` (latitude-only 301 K clamp, binds 45% of land-months) | a moisture-blind clamp that caps the Sahara at 27.9 °C | **Surface energy balance / Bowen ratio** (Penman-Monteith) — wet surfaces self-cap, deserts don't |
| A5 (the `zonal_rescale_factor` architecture that has defeated 8 fixes) | per-row multiplicative nudge to a prescribed `target_row_mm_day` | **Prognostic column water budget** ∂W/∂t = E − P − ∇·(W**v**) with a relaxation closure; see **CLIMBER-X/SESAM** for this exact complexity tier |

Below, each in detail, then instruments, then whole-model peers, then an explicit "not worth
borrowing" list.

---

## Tier 1 — Direct replacements for mechanisms that are currently failing

### 1. Orographic precipitation → Smith & Barstad (2004) linear theory

**Replaces**: `atmosphere.py:1389` `_smear_along_wind`, `orographic_upwind_footprint_km`,
`orographic_spillover_km`, `orographic_uplift_clip`, `precip_potential_ceiling`,
`precip_rain_out_ceiling`, `precip_orographic_shape_weight` — the whole A5-OROG/A5-FOOTPRINT
apparatus.

A5-FOOTPRINT's own conclusion was: *"give the uplift signal an upwind footprint — an
upstream-integrated parcel trajectory, or advection of the uplift/condensate signal along the wind."*
That is, almost verbatim, the Smith–Barstad model, which has been the standard tool for this since
2004.

**What it is.** Two vertically-integrated steady-state advection equations — one for cloud water
density, one for hydrometeor density — solved in Fourier space, giving a single closed-form transfer
function from terrain to precipitation:

```
P̂(k,l) = Ĉw · i σ ĥ(k,l) / [ (1 − i m Hw)(1 + i σ τc)(1 + i σ τf) ]
```

with `σ = k·U + l·V` the intrinsic frequency and `m` the vertical wavenumber set by the moist
stability `Nm`. The two denominators that matter for us:

- **τc** (cloud conversion time, ~1000 s) — condensate doesn't fall where it forms.
- **τf** (hydrometeor fallout time, ~1000 s) — it advects downwind while falling.

Together with `U·τ` as a length scale, those produce exactly the two features we hand-built as
independent tuned kernels: a **broad windward-flank footprint** (air ascends well upstream of the
crest) and **leeward spillover**. They come from one physical parameterization rather than two knobs
that can be set inconsistently. The `1 − i m Hw` term additionally represents airflow *dynamics* —
low-level blocking and flow-over-vs-around — which we have no representation of at all.

**Why it should move the metric we can't move.** Published applications routinely reproduce
box-scale windward/leeward ratios in the 3–10× range for the Southern Alps (NZ), Olympics, Andes and
Cascades — the same targets in `regional_validation.OROGRAPHIC_PAIRS` that we currently miss at
1.57. The reason ours is a 1–2 cell crest spike and theirs is a flank-wide pattern is precisely the
τc/τf advection our pointwise formulation lacks.

**Availability**: [`fastscape-lem/orographic-precipitation`](https://github.com/fastscape-lem/orographic-precipitation)
— **MIT**, NumPy-only, Python ≥3.9, small enough to vendor rather than depend on. Also a
[QGIS plugin](https://github.com/pism/LinearTheoryOrographicPrecipitation) by the PISM group (Aschwanden
& Khrulev) if a second reference implementation is wanted for cross-checking.
Paper: [Smith & Barstad, *J. Atmos. Sci.* 61, 1377 (2004)](https://journals.ametsoc.org/view/journals/atsc/61/12/1520-0469_2004_061_1377_altoop_2.0.co_2.xml).

**Integration cost — read before starting.** This is not free:

- The LT model is **steady-state and linear**, assuming *uniform* upstream wind/stability across the
  FFT domain. A global 512×1024 grid has no single background wind. Realistic approach: apply it
  per-basin or per-tile with the local mean wind, or per-latitude-band, and accept edge effects; the
  literature applies it to single ranges, not to whole planets.
- It returns a precipitation **rate** with its own magnitude calibration (`Cw`, the uplift
  sensitivity factor, from the moist adiabatic lapse rate and a reference density). Whether that
  magnitude survives our moisture budget is the same stage-3 question A5-FOOTPRINT already answered
  negatively for the hand-rolled version — **the deficit-filling rescale will erase LT's contrast
  exactly as it erased ours**. So item 4 below is arguably a prerequisite for getting full value
  here, though the crest-scale and shape gains should show up regardless.
- FFT over the full grid is O(N log N) and cheap; this is not a performance concern.

**Suggested first test**: run LT offline over the bundled DEM with a fixed 10 m/s westerly and score
it directly against `metrics.orographic_contrast` at 256×512, *before* wiring it into
`generate_precipitation`. If the offline field doesn't show 3–6× box-scale contrast, the integration
isn't worth attempting.

---

### 2. Land thermal inertia → force-restore (Bhumralkar 1975; Deardorff 1978)

**Replaces**: the C1b-2026-08-05 stack — `land_seasonal_amplitude` (1.0→0.75),
`land_transport_gain` (1.0→0.5), `land_seasonal_amplitude_maritime` (0.0→0.45).

C1b-2026-08-05's finding was that `temperature_kelvin_for_lat` returns *instantaneous radiative
equilibrium* — a ~81 K annual half-range at 41° against Earth land's ~28 K — and that **the land
branch never had a thermal-inertia term at all**, while the ocean branch has damped its own swing
since early on. The fix shipped was a mean-preserving amplitude multiplier of 0.75.

Force-restore is the canonical cheap solution to exactly that, and it's what essentially every
intermediate-complexity land surface used before multi-layer soil models:

```
∂Ts/∂t = (C1 / (ρc·d1))·G  −  C2·(Ts − T2)/τ        (force term + restore term)
∂T2/∂t = G / (ρc·d2)                                  (deep reservoir, τ = 1 day or 1 year)
```

**Why it's better than the multiplier we shipped**, not just more principled:

1. It produces the **phase lag** as well as the amplitude damping. A pure amplitude multiplier
   damps the swing but keeps the peak on the solstice; real land peaks ~3–4 weeks late. Our audit
   hasn't measured seasonal phase error at all — this is a defect we currently can't see.
2. Thermal inertia is parameterized **as a function of volumetric soil moisture** (dry-soil inertia
   + water inertia). Wet soil = high inertia = damped swing; desert = low inertia = huge swing.
   That is a physically-derived version of the maritime/continental amplitude contrast we hand-built
   as `land_seasonal_amplitude_maritime` — and it would be *moisture*-based rather than
   *upwind-fetch-geometry*-based, which is the correct discriminator and would work on arbitrary
   planets where our fetch heuristic is meaningless.
3. It couples to the **2-layer soil bucket we already have** (`soil-moisture-2layer-bucket`), which
   is currently calibrated near-inert. This would give that module a real job.
4. Force-restore's original purpose is the **diurnal** harmonic. Adopting it also closes audit item
   **B4 / H1 (no diurnal cycle)** essentially for free, since the same two equations run at both
   frequencies — it's a matter of which τ you pick.

**References**: Deardorff, *JGR* 83, 1889 (1978); [Hu & Islam, *WRR* 31 (1995)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/95WR01650);
[Ren & Xue, improved force-restore, *J. Appl. Meteor.* ](https://twister.caps.ou.edu/papers/RX_JAM.pdf).
No package needed — it's two prognostic equations and a thermal-inertia lookup, ~40 lines.

---

### 3. `_land_cap_1d` → a surface energy balance with a Bowen ratio

**Replaces**: `simulate.py:1219`

```python
_land_cap_1d = 301.0 - 15.0 * np.clip((_abs_lat_deg_land - 45.0) / 15.0, 0.0, 1.0)
```

C1b-EVAP established this binds on **45.1% of all land-months**, is latitude-only and
moisture-blind, and caps the Sahara at 27.9 °C against a real ~35 °C July mean. It also absorbs
99.4% of the evapotranspiration cooling term, making that term inert.

The physical mechanism it stands in for is well known: a surface with available water self-limits
its temperature because rising temperature raises the saturation deficit and hence latent flux,
while a dry surface has no such brake and partitions nearly all net radiation into sensible heat.
That's the **Bowen ratio** / evaporative fraction, and the standard closed form is
**Penman–Monteith** (FAO-56):

```
λE = [ Δ(Rn − G) + ρa·cp·(es − ea)/ra ] / [ Δ + γ(1 + rs/ra) ]
Ts  = Ta + ra·(Rn − G − λE) / (ρa·cp)
```

with `rs` the surface resistance driven by our existing soil moisture.

**Why this specifically unblocks two named audit items:**

- **The Sahara/Midwest conflict.** No latitude profile can cap the Midwest at ~28 °C while letting
  the Sahara reach 35 °C — they're at similar latitudes. An evaporative-fraction cap does it
  automatically: the Midwest is wet, the Sahara has `rs → ∞`. This is the single cleanest argument
  for the replacement.
- **Csb and Cfc (H10-DONE, ~1.05% of land the model structurally cannot emit).** Both need a
  *warmest month below 22 °C / 10 °C* in maritime NW Europe and the Pacific NW — places with high
  evaporative fraction and cool, cloudy summers. C1b-EVAP confirmed Cfc **never fires** at 128×256
  under any route tried. A wet-surface latent cap is the mechanism that produces those summers, and
  a latitude clamp is structurally incapable of it.

C1b-EVAP also notes the precondition is now met for the first time: the 35–45°N band-mean peak
overshoot is −0.19 K (was +11.54 K). The clamp is no longer hiding a large error, so replacing it is
newly low-risk.

**Availability**: [`pyet`](https://github.com/pyet-org/pyet) ([GMD 17, 7083, 2024](https://gmd.copernicus.org/articles/17/7083/2024/))
implements 18 PET methods vectorized over `xarray.DataArray`, benchmarked against FAO-56. Realistically
we'd **vendor the FAO-56 Penman-Monteith formula (~30 lines)** rather than take an xarray dependency
into a Numba-heavy codebase — but pyet is the reference to check our implementation against, and its
benchmark tables are the validation.

---

### 4. The `zonal_rescale_factor` architecture → a prognostic column water budget

**Replaces**: `atmosphere.py:4362-4373` — the per-row multiplicative nudge toward a prescribed
`target_row_mm_day`, and everything downstream of it.

This is the largest and riskiest item, and it is the one process note 1 identifies as *"the single
biggest recurring wall,"* having defeated at least eight independently-motivated fixes. A5-FOOTPRINT
localized the residual precisely: *"the entire remaining loss is the deficit fill erasing it."*

The reason it erases contrast is structural, not a tuning error: precipitation is **prescribed**
zonally rather than **conserved**. Any mechanism that creates a local anomaly is renormalized away
because the row must hit its target. This is why redistribution-only fixes provably cannot work, and
why the S Andes pair reaches a `remove_frac` ratio of 3.09 and comes out at 1.37.

**The standard formulation.** Carry column water `W` as a prognostic field:

```
∂W/∂t = E − P − ∇·(W **v**)
```

and close `P` with either:

- **Large-scale condensation**: `P = max(0, q − RHcrit·qsat)/τ` — trivially cheap, τ ~ hours.
- **Simplified Betts–Miller** (Frierson 2007): relax temperature and humidity in convectively
  unstable columns toward a moist adiabat at ~70% RH over a fixed ~2 h timescale, with separate deep
  and shallow (non-precipitating) branches. Frierson found SBM makes the tropics *quieter and much
  less resolution-dependent* than moist-adjustment or large-scale-condensation alternatives — which
  is directly relevant to process note 14's recurring "result reverses sign at higher resolution"
  problem.
  [Frierson, *JAS* 64, 1959 (2007)](https://journals.ametsoc.org/view/journals/atsc/64/6/jas3935.1.xml);
  [Isca's implementation docs](https://execlim.github.io/Isca/latest/html/modules/convection_simple_betts_miller.html)
  (GPL, but readable as a spec).

The zonal mean then **emerges** and becomes a *validation metric* rather than an input. That's a
qualitative change in what `reference_error_score` means — it stops being partly self-fulfilling.

**The existence proof at our exact complexity tier: CLIMBER-X / SESAM.**
[Willeit et al., *GMD* 15, 5905 (2022)](https://gmd.copernicus.org/articles/15/5905/2022/) —
open-source, and the closest published architecture to what PlanetSim is trying to be:

- 2.5-D **statistical-dynamical** atmosphere (not a GCM), 5°×5°, on a regular lat-lon grid.
- Horizontal wind split into **geostrophic** (from SLP, extended aloft by thermal wind) and
  **ageostrophic** (Taylor model, cross-isobar angle) components. Compare our `evolve_wind` /
  `evolve_wind_aloft`, which reach a similar place by prognostic momentum plus a prescribed 3-cell
  relaxation.
- **Prognostic total column water**, with precipitation triggered above 95% near-surface RH plus a
  land term with a turnover time inversely proportional to RH.
- Synoptic eddies via **macroturbulent diffusion with a prognostic eddy-kinetic-energy equation**
  (production ∝ baroclinicity, dissipation ∝ surface drag). This is the principled replacement for
  our prescribed storm/Rossby/meander-index/blocking machinery — the eddy transport *intensity*
  becomes a derived quantity that responds to the baroclinicity the model actually has.
- Tropospheric temperature as a quadratic in height, RH decaying exponentially in the lower
  troposphere — i.e. it gets vertical structure without vertical levels.
- Performance: **~10,000 simulated years/day on 16 CPUs**, coupled to a 3-D ocean.

Whatever we do about the moisture budget, SESAM's paper is the single most useful document to read
before doing it, because it is a validated design for the exact tradeoff this project is making.

---

## Tier 2 — Instruments, reference data, and theory that would derive knobs we currently tune

### 5. Per-cell T/P validation data (closes H10 gap #10)

`scripts/build_cru_ts_reference.py` is built and committed, and CRU TS v4.10 scoring is now
integrated into both `real_terrain_validation.py` and the optimizer (`monthly_climatology_path`
on `run_simulation`) — see `docs/MONTHLY_CLIMATOLOGY_REFERENCE.md` and `FEATURES.md` §4. Notes
below are kept for reference on the source data landscape:

- **CHELSA** ([chelsa-climate.org](https://www.chelsa-climate.org/), 1 km monthly climatologies,
  1979–2013) is specifically documented as having **better orographic precipitation** than WorldClim
  — directly relevant to A5-OROG, where the model's contrast is being scored against Earth. Its
  precipitation algorithm uses wind fields, exposure and boundary-layer height, so it resolves the
  windward/leeward pattern we're trying to reproduce. Note this cuts both ways: it's a *statistical
  downscaling of ERA-Interim*, not observations, so it's a strong pattern reference and a weaker
  absolute-magnitude one.
- **WorldClim 2** is station-interpolated: strong where stations are dense (Europe, N America),
  weak in the tropics, high mountains and the Arctic — which is where several of our named boxes
  are. Prefer CHELSA for mountains, WorldClim for the Midwest/Central Europe boxes.
- **ERA5** monthly means are the right choice if we want a *closed energy/water budget* reference
  rather than a land-only climatology, and the only option over ocean.
- Format warning: CHELSA ships 12 GeoTIFFs per variable, scaled int16/uint16, ~80–150 MB each.
  Regrid once into a compact `.npz` at our benchmark resolutions, the way `koppen_reference.py`
  already does.

### 6. Köppen reference as data rather than a decoded PNG

`koppen_reference.py` decodes `Koppen_classification_world_map_1991-2020_-3C_borderless.png` by
palette lookup. That works (H10-DONE) but is fragile by construction — an unknown colour raises, and
one legend colour had to be identified by geography.

[Beck et al., *Sci. Data* 5:180214 (2018)](https://www.nature.com/articles/sdata2018214) and the
[2023 v2 update](https://www.nature.com/articles/s41597-023-02549-6) publish the same product as
**1-km GeoTIFFs, uint8 class codes, with per-cell confidence maps**, free at
[gloh2o.org/koppen](https://www.gloh2o.org/koppen/). Switching would remove the palette-decode risk
entirely and — more usefully — the **confidence maps let us down-weight cells where the reference
itself is uncertain**, which is exactly the right treatment for the high-latitude and mountain
regions where H10's per-zone accuracy is worst.

### 7. ITCZ position from energetics (would derive `itcz_seasonal_response`)

A2 records an empirical finding: *"0.4 is also the physically better value — Earth's zonal-mean ITCZ
migrates ~±5–8° against a 23.44° declination swing (ratio ~0.25–0.35), so 0.7 was an
over-migration."*

That ratio is not a coincidence to be tuned — it's predicted. The **energy flux equator** framework
(Kang et al. 2008; [Bischoff & Schneider 2014](https://web.gps.caltech.edu/~bordoni/docs/ITCZ_seasonal_james.pdf);
Adam et al. 2016) places the ITCZ where the column-integrated meridional energy flux vanishes, giving

```
φ_ITCZ ≈ −(1/a) · F₀ / NEI₀
```

with `F₀` the cross-equatorial atmospheric energy transport and `NEI₀` the net energy input near the
equator. We have both quantities available. Deriving the ITCZ latitude rather than prescribing its
migration amplitude would:

- remove `itcz_seasonal_response` and `itcz_seasonal_target_response` as free parameters (A2 spent
  multiple sessions and a 2-D grid search on their coupling);
- make the ITCZ respond correctly to **hemispheric asymmetry**, which is the actual reason Earth's
  ITCZ sits north of the equator — something no `|latitude|`-shaped window can represent, and a
  standing structural gap;
- generalize to other planets for free.

### 8. Hadley cell edge theory (B1: dry belt displaced ~10° poleward)

B1/A3's root cause is that the model's zonal-mean divergence peaks at 38–45°N vs Earth's ~25–30°N,
with 85% of the Midwest's divergence being the zonal-mean signal. `generate_wind_field`'s `w_mid`
profile is a latitude-only 3-cell shape centred at 48°.

The descending edge has a predictive theory — **Kang & Lu**: the cell terminates at the lowest
latitude where the upper-branch zonal wind becomes baroclinically unstable, under a uniform
cell-mean Rossby number. The ascending edge follows from **supercriticality**. Both are combined in
[Hill, Bordoni & Mitchell, *JAS* 79 (2022)](https://journals.ametsoc.org/view/journals/atsc/79/10/JAS-D-21-0328.1.xml)
into a scheme that captures the annual cycle of both edges in an idealized aquaplanet GCM.

Practical use: even without adopting the full theory, it gives a **physically-derived target
latitude** for `w_mid`'s centre and `ferrel_v_centre_deg`, and one that moves seasonally and depends
on planetary rotation rate — which is what Theme 3 (planet generalization) needs and a hardcoded
44°/48° cannot provide.

### 9. Moist EBM — an independent cross-check on the zonal profile

If the full moisture-budget rewrite (item 4) is deferred, the next-best thing is to stop
hand-fitting `target_row_mm_day` and instead **derive** it. A diffusive moist EBM transports moist
static energy meridionally and yields zonal-mean T *and* P−E from a handful of parameters.

- [Hwang & Frierson 2010]; [Siler, Roe & Armour, *J. Climate* 31, 18 (2018)](https://journals.ametsoc.org/view/journals/clim/31/18/jcli-d-18-0081.1.xml)
- Code: [`dbonan/energy-balance-models`](https://github.com/dbonan/energy-balance-models) (moist EBM
  + mixed layer + sea ice), [`hgpeterson/mebm`](https://github.com/hgpeterson/mebm) (ITCZ-focused).
- Also useful as a **falsification instrument**: run the MEBM with our insolation and albedo, and
  any large disagreement in the zonal T profile localizes an energy-budget error in our model rather
  than a precipitation error.

### 10. `climlab` — the highest-value single dependency

[climlab](https://github.com/climlab/climlab) (Brian Rose; [JOSS](https://joss.theoj.org/) / GMD),
pure Python since the Fortran was split into `climlab-rrtmg`, `climlab-cam3-radiation`,
`climlab-emanuel-convection` — all pip-installable **binary wheels**, no compiler.

Three pieces are individually worth taking even if we adopt nothing else:

- **`climlab.solar.insolation`** — daily-mean insolation for arbitrary obliquity, eccentricity and
  longitude of perihelion. This closes audit item **H9 (Milankovitch scenario runner)** on its own
  and is a validated replacement for our own orbital insolation path (`orbital_cycles.py`).
- **RRTMG** — real correlated-k band radiation, replacing `temperature.py`'s one-layer gray
  `epsilon_atm`. Relevant to **F2** (model ECS ≈ TCR, both below Earth's ~3 °C): a gray one-layer
  scheme cannot produce a realistic water-vapour or CO₂ forcing curve, so our climate sensitivity is
  structurally constrained. Cost: RRTMG needs vertical profiles we don't have — this is only worth
  it if we ever add levels, or as an **offline calibration** to fit `epsilon_atm(CO2, q)` against
  RRTMG columns, which would be cheap and immediately useful.
- Diffusion solvers and EBMs for item 9.

### 11. Sea ice → Winton (2000) 3-layer, not Semtner 0-layer

If `ocean.update_sea_ice` (`ocean.py:32`) is currently a 0-layer/equilibrium scheme, note the
documented consequence: the zero-layer model has **no ice heat capacity**, and comparisons show it
alters ice-albedo hysteresis to the point that the waterbelt/Snowball hysteresis present under
0-layer *disappears* under 3-layer. That matters specifically for the Milankovitch/glaciation
scenarios in H9 and E2. [Winton, *JTECH* 17, 525 (2000)](https://journals.ametsoc.org/view/journals/atot/17/4/1520-0426_2000_017_0525_artlsi_2_0_co_2.xml)
— an efficient non-iterative implicit scheme, designed for exactly this use.

### 12. Prognostic AMOC → Stommel (1961) two-box

**D2** wants AMOC to respond to temperature as well as salinity. Stommel's two-box model is
literally two ODEs in ΔT and ΔS with a flow `q = k(αΔT − βΔS)`, and it produces the bistability and
collapse bifurcation needed for freshwater-hosing experiments. Modern extensions add an advective
delay for the Arctic salinity anomaly and yield self-sustained multidecadal oscillation
([Wei & Zhang, *GRL* 2022](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022GL099800)) —
which would give the model genuine internal variability, something it has none of today.

---

## Tier 3 — Whole-model peers (context and calibration targets, not replacements)

These are what we'd be reinventing if the goal were a GCM. It isn't — none of them run at
interactive speed with a live GUI — but they are the right things to **validate against** and to
borrow parameterizations from.

- **[ExoPlaSim](https://github.com/alphaparrot/ExoPlaSim)** (Paradise et al.; ASCL 2108.021;
  [docs](https://exoplasim.readthedocs.io/)) — **the closest existing thing to this project's stated
  end goal.** PlaSim (T21 spectral GCM, Fraedrich et al.) plus a Python API, extended for
  tidally-locked rotation, arbitrary surface pressure, non-solar host stars, super-Earths, dynamic
  orography via a glacier module, and carbon-silicate weathering. Directly addresses audit items
  **H5** (tidally locked), **H6** (non-Earth condensables, partially), **H7** (random planets) and
  **H9**. Not interactive — hours per simulated year — but it is the obvious source of *reference
  runs* to calibrate our fast model against for non-Earth configurations, where we currently have no
  ground truth at all.
- **[koppenpasta](https://github.com/hersfeldtn/koppenpasta)** + the
  [Worldbuilding Pasta](https://worldbuildingpasta.blogspot.com/2021/11/an-apple-pie-from-scratch-part-vi.html)
  blog — an existing community doing precisely our workflow: heightmap → ExoPlaSim → Köppen /
  Holdridge / Trewartha maps. Two things worth taking: `im2sra_n.py` (greyscale heightmap → model
  topography) as a spec for arbitrary-terrain import, and the **"Pasta" bioclimate classification**,
  purpose-designed as a Köppen alternative for climates Köppen's Earth-fitted thresholds don't
  describe. That's a real gap for Theme 3 — Köppen's 18 °C and 10 °C thresholds are Earth-biome
  empirics and are meaningless on a Mars or a tidally-locked world.
- **[Isca](https://github.com/ExeClim/Isca)** ([GMD 11, 843, 2018](https://gmd.copernicus.org/articles/11/843/2018/))
  — GFDL FMS dycore with a Python configuration layer, spanning Held-Suarez → grey radiation →
  multi-band moist with topography, explicitly built for other planets. Best use for us: its
  parameterization modules are clean, documented specifications (the SBM convection docs in item 4
  are an example), and its **Held-Suarez** and **Frierson aquaplanet** configurations are the
  standard falsifiable circulation benchmarks. Running a Frierson aquaplanet and comparing our
  zonal-mean jet latitude, Hadley edge and precipitation profile would be a much stronger test of
  B1/B2 than anything we currently have.
- **[SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl)**
  ([JOSS 2024](https://joss.theoj.org/papers/10.21105/joss.06323)) — spectral GCM at ~500 simulated
  years/day at 400 km resolution, differentiable, GPU-capable, and explicitly designed for
  *interactivity and extensibility*. It is the only project with the same performance-and-interaction
  philosophy as PlanetSim while having a real dynamical core. Wrong language for us, but its design
  choices (spherical-harmonic transforms, everything-swappable components) are worth reading, and it
  is the standing answer to "could a real dycore ever be fast enough?" — apparently yes.
- **[climt / sympl](https://github.com/CliMT/climt)** ([GMD 11, 3781, 2018](https://gmd.copernicus.org/articles/11/3781/2018/))
  — componentized Python modelling framework, GFS dycore + RRTMG + Emanuel convection, binary
  wheels. Overlaps climlab; climlab is the better fit for our EBM/insolation needs, climt if we ever
  want a real dycore in Python.

---

## Explicitly not worth borrowing

Saying so to save the time of looking:

- **Full ESMs** — CESM, MITgcm, WRF, MOM6. Wrong scale by three orders of magnitude in both compute
  and setup complexity. Nothing extractable at our tier except formulations already available in
  simpler form above.
- **The hobbyist/worldbuilding tier** — [WorldEngine](https://github.com/esampson/worldengine),
  [genworldvoronoi](https://github.com/Flokey82/genworldvoronoi),
  [C_ClWxSim](https://github.com/RosesHaveThorns/C_ClWxSim), and the various procedural planet
  generators. **All are substantially less sophisticated than PlanetSim already** — typically
  latitude bands plus a rain-shadow heuristic plus Holdridge life zones. There is nothing to take.
  The one exception is the *ExoPlaSim*-based tooling in Tier 3, which is a different tier of work.
- **BIOME4 / LPJ-GUESS** — `carbon_cycle.py` already does NPP and biome classification. BIOME4 is
  diagnostic-only Fortran; LPJ is a much larger commitment than the biome fidelity gap justifies.
  **Holdridge life zones** are the exception worth a mention: biotemperature + annual precipitation +
  PET ratio, ~20 lines, and they'd give a **second classification axis that doesn't depend on
  Earth-fitted thresholds** — useful for Theme 3, and free once item 3 gives us a real PET.

---

## Suggested order, and the honest risk

Ordered by (value × independence) ÷ risk. The key constraint is that items 2–4 replace mechanisms
with ~40 sessions of calibration built around them, so each will move many tracked metrics at once.

1. **Reference data (items 5, 6)** — no physics risk, and it improves the instrument every later
   claim will be judged by. Do this first regardless. Partly underway already.
2. **Smith–Barstad LT, offline first (item 1)** — self-contained, scored against an existing metric
   (`metrics.orographic_contrast`), and cheap to abandon if the offline field doesn't show 3–6×.
   The one Tier-1 item that can be evaluated *before* integrating.
3. **Force-restore + surface energy balance (items 2, 3) together** — they share the soil-moisture
   coupling and both target C1b, `_land_cap_1d`, and the Csb/Cfc vocabulary gap. Doing them
   separately means calibrating each against the other's fudge. This is the highest-confidence real
   physics win: it replaces three tuned amplitude knobs and one unphysical clamp with two published
   mechanisms, and it closes the diurnal-cycle gap as a side effect.
4. **The moisture budget (item 4)** — highest value, highest risk, and read the CLIMBER-X/SESAM
   paper before starting. Note that item 1's full value probably depends on this, per
   A5-FOOTPRINT's own finding that the deficit fill erases whatever contrast raw production creates.
5. **Theory-derived knobs (items 7, 8)** — do after 4, since both concern quantities the moisture
   budget rewrite would change anyway.

**The risk worth stating plainly**: each of items 2–4 will regress tracked metrics on first
implementation, because the surrounding calibration was fitted to the mechanism being removed. Process
note 14 (results reversing sign across resolution) and process note 7 (re-test previously-rejected
parameters after upstream changes) both apply with unusual force here — after any of these lands, the
correct move is a **full re-sweep of the knobs the old mechanism was compensating for**, not a
judgement on the new mechanism from its first-run score.
