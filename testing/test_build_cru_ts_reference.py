from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

from scripts.build_cru_ts_reference import (
    MONTH_DAYS_1991_2020,
    _read_gz_netcdf,
    _nanmean_without_empty_warnings,
    _source_filename,
)


def test_cru_reference_builder_uses_exact_period_chunks_and_calendar_days():
    assert _source_filename("tmp", 1991, 2000) == "cru_ts4.10.1991.2000.tmp.dat.nc.gz"
    assert _source_filename("pre", 2011, 2020) == "cru_ts4.10.2011.2020.pre.dat.nc.gz"
    # 1991-2020 contains eight leap Februaries (1992..2020).
    assert MONTH_DAYS_1991_2020[1] == (28.0 * 22.0 + 29.0 * 8.0) / 30.0


def test_cru_reader_flips_south_to_north_input_and_marks_missing_cells(tmp_path):
    raw_path = tmp_path / "sample.nc"
    with netcdf_file(raw_path, "w") as dataset:
        dataset.createDimension("time", 120)
        dataset.createDimension("lat", 2)
        dataset.createDimension("lon", 4)
        lat = dataset.createVariable("lat", "f4", ("lat",))
        lat[:] = [-45.0, 45.0]  # CRU-style ascending rows
        tmp = dataset.createVariable("tmp", "f4", ("time", "lat", "lon"))
        tmp.missing_value = -9999.0
        values = np.zeros((120, 2, 4), dtype=np.float32)
        values[:, 0, :] = 10.0
        values[:, 1, :] = 20.0
        values[:, :, 3] = -9999.0
        tmp[:] = values
    gz_path = Path(str(raw_path) + ".gz")
    with raw_path.open("rb") as source, gzip.open(gz_path, "wb") as destination:
        destination.write(source.read())

    values, valid = _read_gz_netcdf(gz_path, "tmp")
    assert np.all(values[:, 0, :3] == 20.0)
    assert np.all(values[:, 1, :3] == 10.0)
    assert np.all(valid[:, :3])
    assert not np.any(valid[:, 3])


def test_nanmean_preserves_fully_missing_cells_without_warning():
    values = np.array([[[1.0, np.nan]], [[3.0, np.nan]]])

    result = _nanmean_without_empty_warnings(values, axis=(0,))

    assert result[0, 0] == 2.0
    assert np.isnan(result[0, 1])
