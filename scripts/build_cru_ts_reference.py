"""Build PlanetSim's local CRU TS v4.10 1991-2020 monthly reference.

CRU TS is selected for the first gridded validation reference because it
provides the exact calendar period and the two fields PlanetSim needs over
global land on a regular 0.5 degree grid.  The source data are published by
the Climatic Research Unit, University of East Anglia under ODbL/DbCL terms.

The raw downloads and generated reference are intentionally local artifacts:
they are reproducible from this script but are not committed automatically.
See docs/MONTHLY_CLIMATOLOGY_REFERENCE.md for attribution and use notes.
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlretrieve

import numpy as np
from scipy.io import netcdf_file

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monthly_climatology import MonthlyClimatology, save_monthly_climatology  # noqa: E402


CRU_VERSION = "4.10"
CRU_RELEASE = "cruts.2604091129.v4.10"
CRU_BASE_URL = f"https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_{CRU_VERSION}/{CRU_RELEASE}"
YEARS = ((1991, 2000), (2001, 2010), (2011, 2020))
MONTH_DAYS_1991_2020 = np.array(
    [31.0, 28.266666666666666, 31.0, 30.0, 31.0, 30.0,
     31.0, 31.0, 30.0, 31.0, 30.0, 31.0],
    dtype=np.float64,
)


def _source_filename(variable: str, start_year: int, end_year: int) -> str:
    return f"cru_ts{CRU_VERSION}.{start_year}.{end_year}.{variable}.dat.nc.gz"


def _download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        print(f"Downloading {url}")
        urlretrieve(url, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_gz_netcdf(path: Path, variable: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a 120-month field in north-to-south order and its valid mask."""
    with tempfile.TemporaryDirectory(prefix="planetsim-cru-") as directory:
        extracted = Path(directory) / path.with_suffix("").name
        with gzip.open(path, "rb") as source, extracted.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        with netcdf_file(extracted, "r", mmap=False) as dataset:
            if variable not in dataset.variables:
                raise ValueError(f"{path.name} has no {variable!r} variable")
            values = np.asarray(dataset.variables[variable][:], dtype=np.float64)
            latitudes = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
            missing = getattr(dataset.variables[variable], "missing_value", None)
    if values.ndim != 3 or values.shape[0] != 120:
        raise ValueError(f"{path.name} should contain 120 monthly grids, got {values.shape}")
    if missing is not None:
        values[np.isclose(values, float(missing))] = np.nan
    values[values <= -1e10] = np.nan
    if latitudes[0] < latitudes[-1]:
        values = values[:, ::-1, :]
    valid = np.all(np.isfinite(values), axis=0)
    return values, valid


def _nanmean_without_empty_warnings(values: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    """Average valid values while leaving all-missing cells as NaN."""
    finite = np.isfinite(values)
    totals = np.nansum(values, axis=axis)
    counts = np.sum(finite, axis=axis)
    return np.divide(
        totals,
        counts,
        out=np.full_like(totals, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def build_reference(download_dir: Path, output: Path) -> Path:
    """Download the six decade-variable files and average their 1991-2020 months."""
    decade_fields: dict[str, list[np.ndarray]] = {"tmp": [], "pre": []}
    valid_land: np.ndarray | None = None
    source_urls: list[str] = []
    for variable in decade_fields:
        for start_year, end_year in YEARS:
            filename = _source_filename(variable, start_year, end_year)
            url = f"{CRU_BASE_URL}/{variable}/{filename}"
            path = download_dir / filename
            _download(url, path)
            values, valid = _read_gz_netcdf(path, variable)
            decade_fields[variable].append(values)
            valid_land = valid if valid_land is None else valid_land & valid
            source_urls.append(url)

    if valid_land is None:
        raise RuntimeError("no CRU TS fields were loaded")
    temperature_c = np.concatenate(decade_fields["tmp"], axis=0).reshape(3, 10, 12, 360, 720)
    precipitation_mm_month = np.concatenate(decade_fields["pre"], axis=0).reshape(3, 10, 12, 360, 720)
    temperature_k = _nanmean_without_empty_warnings(temperature_c, axis=(0, 1)) + 273.15
    precipitation_mm_day = (
        _nanmean_without_empty_warnings(precipitation_mm_month, axis=(0, 1))
        / MONTH_DAYS_1991_2020[:, None, None]
    )

    # Ocean and Antarctic cells are intentionally unscored. Replace their
    # missing payload with harmless finite values because MonthlyClimatology
    # rejects NaNs to catch accidentally unmasked source data.
    temperature_k = np.where(valid_land[None, :, :], temperature_k, 273.15)
    precipitation_mm_day = np.where(valid_land[None, :, :], precipitation_mm_day, 0.0)
    reference = MonthlyClimatology(
        temperature_k=temperature_k,
        precipitation_mm_day=precipitation_mm_day,
        land_fraction=valid_land.astype(np.float32),
        metadata={
            "source": "CRU TS v4.10, Climatic Research Unit, University of East Anglia",
            "period": "1991-2020",
            "license": "Open Database License (ODbL) with Database Contents License (DbCL)",
            "grid": "0.5 degree regular global land grid; north-to-south rows",
            "temperature_variable": "tmp: monthly average daily mean temperature (degC)",
            "precipitation_variable": "pre: monthly precipitation total (mm/month)",
            "source_urls": source_urls,
            "builder": "scripts/build_cru_ts_reference.py",
        },
    )
    return save_monthly_climatology(reference, output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the local CRU TS 1991-2020 monthly reference.")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=ROOT / "testing" / "reference_data" / "cru_ts_v4.10_raw",
        help="Cache directory for the six official compressed NetCDF downloads.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "testing" / "reference_data" / "cru_ts_v4.10_1991_2020.npz",
        help="Output MonthlyClimatology NPZ path.",
    )
    args = parser.parse_args()
    output = build_reference(args.download_dir, args.output)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
