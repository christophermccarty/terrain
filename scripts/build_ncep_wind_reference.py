"""Build PlanetSim's local NCEP/NCAR Reanalysis 1 wind climatology.

CRU TS (this project's existing land T/P reference) publishes no wind
variable at all -- verified by listing its actual server directory, not
assumed. NCEP/NCAR Reanalysis 1 is used instead for wind specifically: it is
a global (land+ocean) model reanalysis, anonymously downloadable from NOAA
PSL with no API key, and publishes a ready-made 1991-2020 monthly long-term
mean (climatology) at sigma level 0.995 (near-surface) -- the same period as
the CRU reference, so both can anchor the same "1991-2020" period label even
though they are different products with different native grids
(2.5deg/T62-derived here vs CRU's 0.5deg).

Unlike CRU, this is not a land-only station product: wind is physically
meaningful over ocean too, so no land_fraction is attached, and
monthly_climatology.score_monthly_climatology scores wind_speed_ms globally
(plain cos(lat) weights), independent of any land mask.

The raw downloads and generated reference are intentionally local artifacts,
matching build_cru_ts_reference.py's convention: reproducible from this
script but not committed automatically. See
docs/MONTHLY_CLIMATOLOGY_REFERENCE.md for attribution and use notes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from urllib.request import urlretrieve

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monthly_climatology import MonthlyClimatology, save_monthly_climatology  # noqa: E402


NCEP_BASE_URL = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/surface"
PERIOD = "1991-2020"
_VARIABLES = ("uwnd", "vwnd")


def _source_filename(variable: str) -> str:
    return f"{variable}.sig995.mon.ltm.{PERIOD}.nc"


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


def _read_wind_component(path: Path, variable: str) -> np.ndarray:
    """Return a (12, H, W) component in north-to-south, [-180, 180) order."""
    with h5py.File(path, "r") as f:
        raw = np.asarray(f[variable][:], dtype=np.float64)
        attrs = f[variable].attrs
        scale = float(np.asarray(attrs.get("scale_factor", 1.0)).reshape(-1)[0])
        offset = float(np.asarray(attrs.get("add_offset", 0.0)).reshape(-1)[0])
        missing = attrs.get("missing_value")
        lat = np.asarray(f["lat"][:], dtype=np.float64)
        lon = np.asarray(f["lon"][:], dtype=np.float64)
    if raw.shape != (12, lat.size, lon.size):
        raise ValueError(f"{path.name}: unexpected shape {raw.shape}")
    values = raw * scale + offset
    if missing is not None:
        missing_value = float(np.asarray(missing).reshape(-1)[0])
        values[np.isclose(raw, missing_value)] = np.nan

    # lat: NCEP already runs 90 -> -90 (north-to-south), matching the
    # MonthlyClimatology convention -- no flip needed, but verify rather
    # than assume, since a silently-wrong orientation would corrupt every
    # downstream score without raising an error.
    if not (lat[0] > lat[-1]):
        raise ValueError(f"{path.name}: expected north-to-south lat, got {lat[0]} -> {lat[-1]}")

    # lon: NCEP runs 0..357.5 (0..360 convention); MonthlyClimatology wants
    # [-180, 180). A 0-anchored 2.5deg global grid splits exactly in half at
    # the antimeridian, so a roll by half the array length reorders it to
    # -180..177.5 without interpolation.
    if not np.allclose(lon, np.arange(lon.size) * (360.0 / lon.size)):
        raise ValueError(f"{path.name}: unexpected longitude grid {lon[:3]}...{lon[-3:]}")
    half = lon.size // 2
    values = np.roll(values, shift=half, axis=2)

    # NCEP's native grid samples AT both poles (73 rows for 2.5deg over
    # 180deg: a "grid-point" convention), not MonthlyClimatology's assumed
    # cell-center convention (H rows evenly dividing 180deg with no pole
    # point, matching CRU's own grid and what regrid_monthly_climatology's
    # edge math expects). Longitude has no such mismatch -- it's periodic,
    # so N point samples over 360deg already behave as N cell centers.
    # Collapse the 73 pole-to-pole point rows into 72 band-center rows by
    # averaging each adjacent pair (e.g. the 90N and 87.5N samples become
    # the representative value for the 88.75N band) -- a standard, honest
    # downsampling for a 2.5deg product, not a precision loss beyond what
    # this reference's coarse native resolution already implies.
    values = 0.5 * (values[:, :-1, :] + values[:, 1:, :])

    if np.any(np.isnan(values)):
        raise ValueError(
            f"{path.name}: reanalysis wind has missing cells -- expected full "
            "global coverage from a model product, not a land-station one"
        )
    return values


def build_reference(download_dir: Path, output: Path) -> Path:
    components: dict[str, np.ndarray] = {}
    source_urls: list[str] = []
    for variable in _VARIABLES:
        filename = _source_filename(variable)
        url = f"{NCEP_BASE_URL}/{filename}"
        path = download_dir / filename
        _download(url, path)
        components[variable] = _read_wind_component(path, variable)
        source_urls.append(url)

    wind_speed_ms = np.sqrt(components["uwnd"] ** 2 + components["vwnd"] ** 2).astype(np.float32)
    reference = MonthlyClimatology(
        wind_speed_ms=wind_speed_ms,
        metadata={
            "source": "NCEP/NCAR Reanalysis 1, NOAA Physical Sciences Laboratory",
            "period": PERIOD,
            "license": "NOAA PSL public domain / open data (US Government work)",
            "grid": "2.5 degree regular global grid (land+ocean); north-to-south rows",
            "wind_variable": (
                "sqrt(uwnd^2 + vwnd^2) at sigma level 0.995 (near-surface), "
                "each a monthly long-term mean (climatology)"
            ),
            "source_urls": source_urls,
            "builder": "scripts/build_ncep_wind_reference.py",
        },
    )
    return save_monthly_climatology(reference, output)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the local NCEP/NCAR Reanalysis 1 1991-2020 monthly wind reference."
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=ROOT / "testing" / "reference_data" / "ncep_ncar_raw",
        help="Cache directory for the two official NetCDF4 downloads.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "testing" / "reference_data" / "ncep_ncar_wind_1991_2020.npz",
        help="Output MonthlyClimatology NPZ path.",
    )
    args = parser.parse_args()
    output = build_reference(args.download_dir, args.output)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
