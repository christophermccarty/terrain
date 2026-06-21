# PlanetSim — Physics Depth Plan

> Created: 2026-06-20

Two sequential efforts. Effort 1 completes first with passing tests before Effort 2 begins.

---

## Effort 1 — Obliquity & Planet Physics Depth

### Problem Summary

Three physics gaps remain for non-Earth planets:

1. **AMOC/ACC hardcoded to Earth scale** — the 18 K AMOC peak and 28 K ACC peak don't scale with rotation rate or ocean coverage. A slow rotator (weak Coriolis → weak western boundary currents → weak AMOC) gets the same NH warming bonus as Earth.
2. **Obliquity seasonal cap** — `ocean_seasonal_frac` is clipped to 0.45 regardless of obliquity, suppressing the larger polar seasonal swings that high-obliquity planets should show.
3. **Ekman transport never called in simulate_step** — `compute_ekman_transport` exists in ocean.py but is not wired into the heat-transport path.

---

### Phase 1A — AMOC/ACC Rotation & Ocean-Fraction Scaling

**File:** `simulate.py`, function `_evolve_temperature`, ~line 604

Add two scale factors immediately before the transport block:

```python
_EARTH_OMEGA = 7.2921e-5  # rad/s  (2π / 23.9345 h)
_rotation_scale = float(np.clip((pp.omega / _EARTH_OMEGA) ** 0.4, 0.05, 2.0))
_ocean_frac_scale = float(pp.ocean_fraction / 0.71)
# AMOC requires prograde rotation for western boundary currents
_amoc_scale = _ocean_scale * _rotation_scale * _ocean_frac_scale * float(pp.rotation_direction > 0)
_acc_scale  = _ocean_scale * _ocean_frac_scale  # ACC is wind-driven, less rotation-sensitive
```

Replace the three existing uses of `_ocean_scale` in the transport block:

| Before | After |
|--------|-------|
| `_transport_base = _ocean_scale * 34.0 * …` | `_transport_base = _acc_scale * 34.0 * …` |
| `_amoc_bonus = _ocean_scale * amoc_factor * …` | `_amoc_bonus = _amoc_scale * amoc_factor * …` |
| `_acc_bonus = _ocean_scale * acc_factor * …` | `_acc_bonus = _acc_scale * acc_factor * …` |

Apply identically to the full-resolution versions (`_transport_base_full`, `_amoc_bonus_full`, `_acc_bonus_full`).

**Physical basis:**  
- AMOC scales ~ω^0.4: at 0.1× Earth rotation → 40% AMOC strength. Coriolis force drives the western boundary intensification that concentrates AMOC heat delivery; halving ω weakens it more than linearly.  
- Ocean fraction: a planet with 30% ocean has proportionally less thermohaline reservoir.  
- Retrograde rotators: Coriolis deflects to the east (opposite), so western boundary currents form on the eastern side. No Atlantic-style AMOC in the NH. Suppress by multiplying by `float(rotation_direction > 0)`.

**Earth validation:** for Earth, `_rotation_scale = 1.0`, `_ocean_frac_scale = 1.0`, `rotation_direction = 1` → multipliers are identical to today. Zero change to Earth calibration.

---

### Phase 1B — Obliquity-Scaled Seasonal Cap

**File:** `simulate.py`, function `_evolve_temperature`, ~line 643

Current (two locations — coarse and full-resolution):
```python
ocean_seasonal_frac = np.clip(ocean_seasonal_frac, 0.03, 0.45)
```

Replace with:
```python
_seasonal_cap = float(min(0.45 * obliq_factor, 0.85))
ocean_seasonal_frac = np.clip(ocean_seasonal_frac, 0.03, _seasonal_cap)
```

Effect by obliquity:
| Obliquity | obliq_factor | Old cap | New cap |
|-----------|-------------|---------|---------|
| 23.44° (Earth) | 1.00 | 0.45 | 0.45 |
| 45° | 1.39 | 0.45 | 0.62 |
| 60° | 1.60 | 0.45 | 0.72 |
| 90° | 2.00 (capped) | 0.45 | 0.85 |

**Earth validation:** `obliq_factor = 1.0` → new cap = 0.45 = old cap. Zero change.

---

### Phase 1C — Ekman Transport Wiring (lower priority)

**File:** `ocean.py` → `calculate_ocean_heat_transport`; `simulate.py` call site

Add `wind_u: np.ndarray | None = None, wind_v: np.ndarray | None = None` parameters to `calculate_ocean_heat_transport`. Inside the function, if wind fields are provided, compute Ekman divergence and add a small correction to `T_adjustment`:

```python
if wind_u is not None and wind_v is not None:
    u_e, v_e = compute_ekman_transport(wind_u, wind_v, elevation)
    # Ekman divergence → upwelling (cold) or downwelling (warm)
    div_e = _ddx_ekman(u_e) + np.gradient(v_e, axis=0)  # ∂u/∂x + ∂v/∂y
    ekman_T_adj = -pp_ocean_frac_scale * 0.5 * div_e * dt_days
    T_adjustment += np.clip(ekman_T_adj * is_ocean, -1.0, 1.0)
```

In `simulate.py`, pass `wind_u=u_full, wind_v=v_full` to `calculate_ocean_heat_transport` at the existing call site. Scale the Ekman coefficient by `pp.surface_pressure_pa / 101325`.

This is optional for Effort 1 validation — implement only if 1A+1B pass without instability.

---

### Phase 1D — Tests

New file: `testing/test_planet_physics.py`

| Test | What it checks | How |
|------|---------------|-----|
| `test_slow_rotator_weaker_nh_warming` | 10× slower rotation → NH 70–80° polar T at least 4 K colder than Earth | 0.5yr MONTHLY spinup, compare polar T_base |
| `test_retrograde_no_amoc_asymmetry` | rotation_direction=-1 → NH/SH polar T difference < 5 K (no AMOC asymmetry) | 0.5yr spinup |
| `test_low_ocean_fraction_steeper_gradient` | ocean_fraction=0.3 → NH gradient at least 3 K larger than Earth | 0.8yr spinup |
| `test_high_obliq_seasonal_range` (promoted from xfail) | obliquity=45° seasonal amplitude > Earth amplitude × 1.2 | 1yr DAILY via headless |

After implementing 1A+1B, attempt to remove `@pytest.mark.xfail` from `test_high_obliquity_larger_seasonal_range` in `testing/test_planet_params.py`. If it passes, promote to a hard assert.

---

### Phase 1E — Validation Checklist

1. Full test suite: `python -m pytest testing/ -q` — must stay at ≥ 130 passed, ≤ 14 xfailed.
2. Optimizer single run: `python optimizer/runner.py --mode single --H 60 --W 120 --spinup-years 2 --eval-years 1` — Earth score must stay ≥ previous baseline.
3. Mars runs without NaN: `python -m pytest testing/test_planet_generalization.py -v`.

---

## Effort 2 — Biomes & Long-run Dynamics

Begins after Effort 1 tests pass.

### Problem Summary

The ANNUAL time-scale mode exists but has never been stress-tested for multi-decade or multi-century runs. Biomes (Köppen + vegetation) respond to climate but the feedback chain hasn't been validated over long runs. There is no test that demonstrates a stable equilibrium over 50+ simulated years.

---

### Phase 2A — ANNUAL Stability Baseline

**File:** `testing/test_annual_stability.py` (new)

Run Earth ANNUAL for 50 simulated years (headless, 32×64) and verify:
- No NaN/Inf at any step
- Global mean T drift < 5 K over the 50 years
- CO2 stays in [200, 700] ppm
- Ice fraction stays in [0.01, 0.40]

This is the stability proof that gates the rest of Effort 2. If it fails with current ANNUAL physics, fix the instabilities before proceeding.

**Implementation note:** `optimizer/headless.py` needs a `run_long_simulation(years, ...)` helper that runs ANNUAL-only with minimal overhead (no eval_snapshots, just final state + per-year diagnostics).

---

### Phase 2B — Biome Response to Climate Shift

**File:** `testing/test_biome_response.py` (new)

Two-phase run (both at 32×64, ANNUAL):
1. **Baseline:** 20yr Earth → record ice fraction + biome distribution
2. **Perturbed:** same state, solar_constant → 1600 W/m², run 20yr more → record

Assertions:
- NH ice fraction decreases after perturbation
- Fraction of cells classified as tropical (Af/Am/Aw) increases
- No NaN in either phase

This validates that the biome + climate feedback chain responds correctly to a large forcing over a multi-decade run.

---

### Phase 2C — CO2 Drawdown Feedback

**File:** `testing/test_co2_feedback.py` (new)

Start at 600 ppm CO2, run Earth ANNUAL for 30 years.
- CO2 should draw down toward equilibrium (vegetation + ocean uptake)
- T should be higher than Earth baseline (greenhouse warming)
- Verify the CO2 drawdown rate is in a physically plausible range (1–5 ppm/yr)

Start at 200 ppm CO2, run 30 years.
- CO2 should rise (respiration > uptake at low CO2)
- T should be lower
- Ice fraction higher

---

### Phase 2D — Long Headless Helper

**File:** `optimizer/headless.py`

Add:
```python
def run_long_simulation(
    planet_params: PlanetParams,
    years: int,
    *,
    H: int = 32,
    W: int = 64,
    elevation: np.ndarray | None = None,
    spinup_years: float = 2.0,
    sample_every: int = 10,   # record diagnostics every N years
) -> list[dict]:
    """Run ANNUAL-mode simulation for many years. Returns per-sample diagnostic dicts."""
```

This runner doesn't call into `run_simulation` (which uses a fixed eval structure) — it drives ANNUAL sub-steps directly in a tight loop. Target: 100 simulated years at 32×64 in < 20 seconds.

---

### Phase 2E — Ice Age Proof-of-Concept (stretch goal)

Not a test — an interactive scenario:

1. Start Earth with `co2_initial_ppm=200.0, obliquity_deg=22.0` (Milankovitch minimum)
2. Run `run_long_simulation(years=500, H=32, W=64)`
3. Track: T, CO2, ice fraction per decade
4. Expected: sustained cold state with high ice fraction

If the model produces a clear glacial state distinct from modern Earth, this demonstrates that the feedback chain is working at long timescales. Document results in a notebook or script under `experiments/`.

---

## Execution Order

```
1A (AMOC scaling)
    ↓
1B (obliquity cap)
    ↓  
Test Phase 1D → validate → confirm Earth calibration stable
    ↓
1C (Ekman, optional)
    ↓
2A (ANNUAL stability baseline) — if not stable, fix ANNUAL path first
    ↓
2B + 2C + 2D (parallel — all use headless ANNUAL)
    ↓
2E (proof-of-concept, optional)
```

Total expected implementation: ~8–10 focused edits across simulate.py, ocean.py, and 3 new test files.
