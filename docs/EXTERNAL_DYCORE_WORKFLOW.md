# External-dycore reference workflow

PlanetSim's interactive solver remains self-contained.  This workflow runs
ExoPlaSim offline as a dynamical reference and returns its monthly temperature
and precipitation fields to the existing CRU/Köppen validation contract. It is
not an unscored replacement or a runtime dependency.

## Exchange contract

`external_dycore.export_exoplasim_request()` writes a safe NPZ request with:

- area-conservatively regridded topography in metres and fractional land mask;
- north-to-south, [-180, 180) coordinates;
- gravity, radius, pressure, stellar flux, obliquity, year length, and rotation;
- resolution/layer/spin-up/evaluation choices in versioned JSON metadata.

The Linux wrapper writes ExoPlaSim's land (`kcode 172`) and geopotential
topography (`kcode 129`) files using ExoPlaSim's own writer. It passes the
writer's actual suffixed filenames to the model and discovers the regular
archive in ExoPlaSim's finalized output directory. Spin-up fields are discarded
while the dynamical restart is retained; every requested evaluation year is
postprocessed to near-surface air temperature `tas` and `pr`, then explicitly averaged before normalization. It then normalizes
the postprocessed `ts`/`pr` fields to `MonthlyClimatology`, including an
explicit precipitation-unit conversion. That artifact can be scored against
CRU using the same functions as PlanetSim's native validation.

## First Earth reference

Create the request from the bundled DEM (the command can be kept in a small
one-off Python driver until the first environment is provisioned):

```python
from external_dycore import export_exoplasim_request
from planet_params import EARTH
from real_terrain_validation import load_bundled_earth_dem

export_exoplasim_request(
    load_bundled_earth_dem(64, 128), EARTH,
    "testing/reference_data/exoplasim_earth_t21_request.npz",
)
```

Inside a Linux environment with ExoPlaSim and its GNU compiler prerequisites:

```bash
python scripts/run_exoplasim_reference.py \
  --request testing/reference_data/exoplasim_earth_t21_request.npz \
  --workdir testing/reference_data/exoplasim_earth_t21_run \
  --output testing/reference_data/exoplasim_earth_t21_monthly.npz
```

For a multi-process run, first install a standard MPI implementation, cache
the desired executable once (as the account that maintains the environment),
then run the simulation as an unprivileged account:

```bash
python scripts/run_exoplasim_reference.py --build-only --ncpus 4 \
  --request testing/reference_data/exoplasim_earth_t21_request.npz \
  --workdir /tmp/planetsim_exoplasim_build
```

On the current Debian/WSL toolchain, rerun `exoplasim.sysconfigure()` after
installing MPI. If the generated MPI compiler options enable
`-ffpe-trap=invalid,zero,overflow`, remove that *trap* option and rebuild the
MPI executable: OpenMPI's own topology initialization can otherwise fault
before PlaSim begins. This is an environment-compatibility adjustment, not a
PlanetSim parameter or physics change; retain `-O3` and record the compiler
options next to every promoted reference.

Before treating any output as a calibration target, inspect the actual field
keys and units in the raw archive. Current ExoPlaSim NPZ metadata labels `pr`
as `m s-1`; the runner defaults to that conversion. Use an explicit override
only when a different installed version reports different units.

If a completed run retained snapshots with `tas`, reprocess it without another
GCM integration:

```bash
python scripts/canonicalize_external_dycore_reference.py \
  --archive-glob '.../snapshots/MOST_SNAP.*.npz' \
  --request testing/reference_data/exoplasim_earth_t21_request.npz \
  --output testing/reference_data/exoplasim_earth_t21_monthly.npz
```

Score the normalized artifact without importing ExoPlaSim into the PlanetSim
runtime:

```bash
python scripts/score_external_dycore_reference.py \
  --monthly testing/reference_data/exoplasim_earth_t21_monthly.npz \
  --output testing/reference_data/exoplasim_earth_t21_cru.json
```

The score regrids CRU to the external model grid, applies the same
area-weighted temperature and precipitation metrics, and derives a Köppen map
from the external fields. It deliberately does not apply PlanetSim's terrain
lapse adjustment a second time.

The normalized artifact records the external-engine version, process count,
actual evaluation-year count, and averaging rule, so a score retains the
execution context that produced it.

## Field conventions verified against ExoPlaSim NPZ metadata

- Use `tas` (near-surface air temperature) for comparison with CRU air
  temperature; `ts` is surface skin temperature and is not interchangeable.
- Use `m s-1` for `pr` in the current ExoPlaSim NPZ output. Converting it as
  `kg m-2 s-1` underestimates precipitation by a factor of 1,000.
- Preserve snapshots when possible: they make it possible to recover `tas`
  after a compact `ts`/`pr` regular archive has already been written.

The first 20-year spin-up plus 30-year T21 Earth reference completed with
credible land precipitation after these conventions were corrected (annual
log-precipitation correlation 0.744 against CRU; Köppen-group agreement
0.608). Its near-surface temperature pattern remains too continental/polar at
this coarse external configuration, so it is a precipitation/circulation
reference rather than a field-by-field temperature calibration target.

At the same 32x64 grid, the native one-year CRU benchmark has lower
temperature RMSE (7.13 C versus 19.32 C) and stronger Köppen-group agreement
(0.641 versus 0.608), while the external case has stronger annual
precipitation log-correlation (0.744 versus 0.464). Therefore do not couple
the external temperatures or classes directly into PlanetSim; use its
large-scale precipitation placement as an independent falsification signal.

## Promotion rules

1. Preserve the request, exported ExoPlaSim configuration, raw output, and
   normalized monthly artifact together.
2. Score the normalized artifact against the local CRU TS v4.10 reference.
3. First use it to falsify PlanetSim's zonal-mean jet, Hadley-edge, and
   precipitation placement; do not tune a field-by-field correction from it.
4. Promote a coupling only when it improves the stored CRU/Köppen contract at
   compact resolution and then survives the 128x256 five-year checkpoint.
