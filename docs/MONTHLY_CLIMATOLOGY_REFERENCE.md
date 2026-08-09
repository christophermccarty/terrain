# Monthly climatology reference contract

`monthly_climatology.py` provides the optional, provider-neutral monthly map
validation used by `scripts/run_real_terrain_validation.py`.

The recommended initial source is **CRU TS v4.10**: it supplies 0.5 degree
global-land monthly `tmp` (mean temperature) and `pre` (precipitation total)
for the exact 1991-2020 period used by the Köppen reference. It is available
under ODbL/DbCL; retain the required attribution to the Climatic Research Unit,
University of East Anglia, and the share-alike conditions for derived
databases. Build the local reference with:

```powershell
.\.venv\Scripts\python.exe scripts\build_cru_ts_reference.py
```

The raw downloads and derived reference are reproducible local artifacts under
`testing/reference_data/`, which is ignored by git. The builder records every
official input URL in the NPZ metadata.

The reference must cover a documented 12-month climatological period and be
stored as a safe NPZ file. Its required arrays are `temperature_k` and
`precipitation_mm_day`, both shaped `(12, H, W)` on a north-to-south,
[-180, 180) regular 2:1 global grid. `land_fraction` is optional and excludes
mixed coastal/reference-ocean cells from land skill metrics. `metadata_json`
must record at least `source`, `period`, and the source's redistribution
licence.

Run it with:

```powershell
.\.venv\Scripts\python.exe scripts\run_real_terrain_validation.py `
  --monthly-climatology path\to\climatology.npz
```

The input is area-conservatively regridded to the simulation grid. The report
contains monthly temperature bias/RMSE/correlation and raw/log precipitation
skill. The logarithmic precipitation metrics use a documented 0.05 mm/day
floor so arid-region errors remain measurable without allowing zeros to make a
score undefined.

## Initial measured baseline

At the tracked 64x128, one-year spin-up/one-year evaluation configuration, the
first CRU TS v4.10 comparison measured:

- Land monthly temperature: +2.98 C bias, 6.28 C RMSE, 0.930 correlation.
- Land monthly precipitation: -0.109 mm/day bias, 2.65 mm/day RMSE, 0.361
  correlation; log-space RMSE 1.406 and correlation 0.463.

This is a measurement baseline, not an acceptance threshold. It establishes a
clear next priority: correct the warm land-temperature bias with the existing
land surface-energy/thermal-inertia mechanisms before treating individual
Koppen or rainfall-map symptoms. Re-run the command above after every
intentional physics change and retain the generated JSON report with the run.

At the first 128x256, five-year spin-up/five-year evaluation checkpoint, the
baseline measured 5.73 C temperature RMSE, 1.61 C temperature bias, 1.460
precipitation log-RMSE, and Köppen group/class accuracy of 0.709/0.422. The
regional temperature report identifies Atacama, southern Japan, SE US, and
continental interiors as the principal warm-bias targets; this is the promotion
baseline for any future land-energy change.

## Gated condensate-closure experiment

`PlanetParams.enable_prognostic_condensate` activates a new bulk,
mass-conserving vapor-to-condensate-to-rain reservoir. It is intentionally
off by default while it is calibrated against this reference. The first
uncoupled trial double-counted its rainfall alongside the allocator and was
rejected. The corrected closure lets condensate satisfy the row target before
additional vapor rain is allocated, and carries suspended condensate with the
resolved wind. Its best first trial (6-day condensation timescale) reduced
precipitation log-RMSE from 1.406 to 1.401, but reduced Koppen group accuracy
from 0.674 to 0.666. The mechanism therefore remains an A/B pathway; it must
improve both rainfall skill and classification before it can displace the
existing target allocator.

Do not commit a source dataset until its licence permits redistribution. Commit
the provenance metadata and the preprocessing script alongside any derived
fixture; the source period should match the 1991-2020 Köppen reference whenever
possible.

## Wind reference (NCEP/NCAR Reanalysis 1)

CRU TS publishes no wind variable at all (verified by listing its actual server
directory: only `cld, dtr, frs, pet, pre, tmn, tmp, tmx, vap, wet` exist), so
wind uses a second, independent provider: **NCEP/NCAR Reanalysis 1**, a global
land+ocean model reanalysis, anonymously downloadable from NOAA PSL (no API
key). Build it with:

```powershell
.\.venv\Scripts\python.exe scripts\build_ncep_wind_reference.py
```

This fetches the official `uwnd.sig995.mon.ltm.1991-2020.nc` /
`vwnd.sig995.mon.ltm.1991-2020.nc` files (near-surface sigma-0.995 monthly
long-term means, the same 1991-2020 period as the CRU reference though a
different native grid: 2.5deg, pole-inclusive) and derives
`wind_speed_ms = sqrt(uwnd^2 + vwnd^2)`. The 73-row pole-to-pole point grid is
averaged down to 72 cell-center rows (adjacent-pair mean) to match
`MonthlyClimatology`'s cell-center 2:1 grid convention -- see the builder's own
comments for why this differs from CRU's grid, which needed no such
adjustment. Requires `h5py` (NCEP's files are HDF5/NetCDF4; the NetCDF3-only
reader already used for CRU can't open them).

Unlike temperature/precipitation, wind is scored **globally** (no land mask --
wind is physically meaningful over ocean) and as an **annual mean only** on the
model side, not true month-by-month (no wind reference here has genuine
monthly resolution end-to-end yet) -- both documented in
`monthly_climatology.score_monthly_climatology`'s own docstring. Use
`--wind-climatology path\to\ncep_ncar_wind_1991_2020.npz` on
`scripts\run_real_terrain_validation.py`, independently of
`--monthly-climatology` (different providers; either or both may be given).

**Initial measured baseline** (64x128, one-year spin-up/one-year evaluation,
real bundled Earth DEM): -1.10 m/s bias, 2.73 m/s RMSE, 0.151 correlation. Weak
correlation is expected and not itself evidence of a bug -- the model's
single-layer, largely-diagnostic wind field is far cruder than a full
reanalysis; `testing/test_reanalysis_validation.py`'s wind gate bounds are
deliberately loose for exactly this reason (catch a real pipeline break, e.g.
a longitude misalignment, not enforce realism the model was never designed to
reach).
