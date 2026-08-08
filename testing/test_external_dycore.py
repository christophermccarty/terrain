from __future__ import annotations

import numpy as np

from external_dycore import (
    ExoPlaSimRequest,
    canonicalize_exoplasim_output,
    export_exoplasim_request,
    load_exoplasim_request,
    score_external_dycore_against_cru,
    score_native_against_external_dycore,
    average_exoplasim_archives,
)
from monthly_climatology import MonthlyClimatology, load_monthly_climatology
from planet_params import EARTH


def test_exoplasim_exchange_request_regrids_and_records_planet(tmp_path):
    elevation = np.zeros((64, 128), dtype=np.float32)
    elevation[20:40, 40:90] = 0.5
    request_path = export_exoplasim_request(
        elevation,
        EARTH,
        tmp_path / "request.npz",
        request=ExoPlaSimRequest(spinup_years=2, evaluation_years=3),
    )
    arrays, metadata = load_exoplasim_request(request_path)
    assert arrays["topography_m"].shape == (32, 64)
    assert arrays["land_fraction"].shape == (32, 64)
    assert metadata["engine"] == "ExoPlaSim"
    assert metadata["planet"]["surface_gravity_m_s2"] == EARTH.surface_gravity


def test_exoplasim_output_is_canonicalized_to_monthly_contract(tmp_path):
    elevation = np.zeros((32, 64), dtype=np.float32)
    request_path = export_exoplasim_request(elevation, EARTH, tmp_path / "request.npz")
    raw_path = tmp_path / "raw.npz"
    np.savez_compressed(
        raw_path,
        ts=np.full((24, 32, 64), 280.0, dtype=np.float32),
        pr=np.full((24, 32, 64), 1.0e-5, dtype=np.float32),
    )
    output_path = canonicalize_exoplasim_output(
        raw_path,
        request_path,
        tmp_path / "monthly.npz",
        runner_provenance={"ncpus": 4},
    )
    result = load_monthly_climatology(output_path)
    assert output_path.exists()
    assert result.temperature_k.shape == (12, 32, 64)
    assert np.allclose(result.temperature_k, 280.0)
    assert np.allclose(result.precipitation_mm_day, 0.864)
    assert result.metadata["engine_request"]["engine"] == "ExoPlaSim"
    assert result.metadata["external_runner"]["ncpus"] == 4


def test_exoplasim_finalize_directory_selects_regular_archive(tmp_path):
    elevation = np.zeros((32, 64), dtype=np.float32)
    request_path = export_exoplasim_request(elevation, EARTH, tmp_path / "request.npz")
    raw_dir = tmp_path / "finalized"
    raw_dir.mkdir()
    np.savez_compressed(
        raw_dir / "planetsim_reference.npz",
        ts=np.full((12, 32, 64), 280.0, dtype=np.float32),
        pr=np.full((12, 32, 64), 1.0e-5, dtype=np.float32),
    )
    np.savez_compressed(raw_dir / "planetsim_reference_metadata.npz", ignored=np.asarray([1]))
    np.savez_compressed(raw_dir / "planetsim_reference_snapshot.npz", ignored=np.asarray([1]))
    output_path = canonicalize_exoplasim_output(raw_dir, request_path, tmp_path / "monthly.npz")
    result = load_monthly_climatology(output_path)
    assert result.metadata["raw_archive"] == "planetsim_reference.npz"


def test_external_score_regrids_cru_without_reclassifying_when_no_land_mask():
    model = MonthlyClimatology(
        temperature_k=np.full((12, 32, 64), 280.0),
        precipitation_mm_day=np.full((12, 32, 64), 2.0),
        metadata={"schema_version": 1, "source": "external test", "period": "1-year"},
    )
    cru = MonthlyClimatology(
        temperature_k=np.full((12, 64, 128), 281.0),
        precipitation_mm_day=np.full((12, 64, 128), 3.0),
        land_fraction=np.ones((64, 128)),
        metadata={"schema_version": 1, "source": "CRU test", "period": "1991-2020"},
    )
    report = score_external_dycore_against_cru(model, cru)
    assert report["model_grid"] == {"height": 32, "width": 64}
    assert report["koppen_map_skill"] == {}
    assert report["monthly_climatology"]["temperature_c"]["monthly_bias"] == -1.0


def test_average_exoplasim_archives_reduces_each_snapshot_year_before_mean(tmp_path):
    paths = []
    for year, value in enumerate((280.0, 284.0)):
        path = tmp_path / f"year_{year}.npz"
        np.savez_compressed(
            path,
            tas=np.full((24, 32, 64), value, dtype=np.float32),
            pr=np.full((24, 32, 64), (year + 1) * 1e-6, dtype=np.float32),
        )
        paths.append(path)
    temperature, precipitation, years = average_exoplasim_archives(
        paths, temperature_key="tas", precipitation_key="pr"
    )
    assert years == 2
    assert temperature.shape == (12, 32, 64)
    assert np.allclose(temperature, 282.0)
    assert np.allclose(precipitation, 1.5e-6)


def test_native_external_score_regrids_reference_to_native_grid():
    native = MonthlyClimatology(
        temperature_k=np.full((12, 4, 8), 280.0),
        precipitation_mm_day=np.full((12, 4, 8), 3.0),
        land_fraction=np.ones((4, 8)),
        metadata={"source": "native test", "period": "one year"},
    )
    external = MonthlyClimatology(
        temperature_k=np.full((12, 2, 4), 280.0),
        precipitation_mm_day=np.full((12, 2, 4), 3.0),
        land_fraction=np.ones((2, 4)),
        metadata={"source": "external test", "period": "one year"},
    )
    report = score_native_against_external_dycore(native, external)
    assert report["native_grid"] == {"height": 4, "width": 8}
    assert report["monthly_climatology"]["temperature_c"]["monthly_rmse"] == 0.0
