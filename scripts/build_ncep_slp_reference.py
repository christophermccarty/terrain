"""Build PlanetSim's local NCEP/NCAR Reanalysis 1 sea-level-pressure climatology.

Companion to build_ncep_wind_reference.py: same source (NCEP/NCAR Reanalysis
1, NOAA PSL, anonymously downloadable, no API key), same period (1991-2020
monthly long-term mean), same 2.5deg global grid handling. SLP is the one
circulation field SESAM stage P2's diagnostic SLP reconstruction
(`sesam_dynamics.py`, Appendix A2 of Willeit et al. 2022) can be scored
against directly -- the P2 exit gate compares reconstructed DJF/JJA SLP
patterns against reanalysis (docs/SESAM_GAP_ANALYSIS.md section 7), and this
is the reanalysis SLP product already used for the project's wind reference.

The raw download and generated reference are intentionally local artifacts,
matching build_ncep_wind_reference.py's convention: reproducible from this
script but not committed automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


NCEP_BASE_URL = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/surface"
PERIOD = "1991-2020"
VARIABLE = "slp"
SCHEMA_VERSION = 1


def _source_filename() -> str:
    return f"{VARIABLE}.mon.ltm.{PERIOD}.nc"


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


def _read_slp(path: Path) -> np.ndarray:
    """Return the (12, H, W) SLP field in Pa, north-to-south, [-180, 180)."""
    with h5py.File(path, "r") as f:
        raw = np.asarray(f[VARIABLE][:], dtype=np.float64)
        attrs = f[VARIABLE].attrs
        scale = float(np.asarray(attrs.get("scale_factor", 1.0)).reshape(-1)[0])
        offset = float(np.asarray(attrs.get("add_offset", 0.0)).reshape(-1)[0])
        missing = attrs.get("missing_value")
        units = attrs.get("units")
        lat = np.asarray(f["lat"][:], dtype=np.float64)
        lon = np.asarray(f["lon"][:], dtype=np.float64)
    if raw.shape != (12, lat.size, lon.size):
        raise ValueError(f"{path.name}: unexpected shape {raw.shape}")
    # Explicit unit conversion, verified from the file's own attribute (the
    # ExoPlaSim workflow's recorded lesson: never assume reference units).
    if isinstance(units, np.ndarray):
        units_text = units.tobytes().decode("ascii", "replace").strip("\x00").lower()
    elif isinstance(units, (bytes, np.bytes_)):
        units_text = bytes(units).decode("ascii", "replace").lower()
    else:
        units_text = str(units or "").lower()
    if units_text not in ("millibars", "mb", "hpa"):
        raise ValueError(f"{path.name}: expected millibar SLP units, got {units_text!r}")
    values = (raw * scale + offset) * 100.0  # hPa -> Pa
    if missing is not None:
        missing_value = float(np.asarray(missing).reshape(-1)[0])
        values[np.isclose(raw, missing_value)] = np.nan

    # lat: verified north-to-south, matching MonthlyClimatology (see the wind
    # builder for why this must be verified rather than assumed).
    if not (lat[0] > lat[-1]):
        raise ValueError(f"{path.name}: expected north-to-south lat, got {lat[0]} -> {lat[-1]}")

    # lon: 0..357.5 -> roll to [-180, 180) (periodic grid, no interpolation).
    if not np.allclose(lon, np.arange(lon.size) * (360.0 / lon.size)):
        raise ValueError(f"{path.name}: unexpected longitude grid {lon[:3]}...{lon[-3:]}")
    values = np.roll(values, shift=lon.size // 2, axis=2)

    # Collapse the 73 pole-to-pole point rows into 72 band-center rows by
    # adjacent-pair averaging (same convention as the wind builder).
    values = 0.5 * (values[:, :-1, :] + values[:, 1:, :])

    if np.any(np.isnan(values)):
        raise ValueError(
            f"{path.name}: reanalysis SLP has missing cells -- expected full "
            "global coverage from a model product"
        )
    return values


def build_reference(download_dir: Path, output: Path) -> Path:
    filename = _source_filename()
    url = f"{NCEP_BASE_URL}/{filename}"
    path = download_dir / filename
    _download(url, path)
    slp_pa = _read_slp(path).astype(np.float32)

    # Plain NPZ, not MonthlyClimatology: that container's fixed fields cover
    # T/P/wind for the supported scoring contract, and a diagnostic-only SLP
    # reference does not belong in it. Same safety conventions: no pickle,
    # metadata_json provenance, versioned schema.
    metadata = {
        "source": "NCEP/NCAR Reanalysis 1, NOAA Physical Sciences Laboratory",
        "period": PERIOD,
        "license": "NOAA PSL public domain / open data (US Government work)",
        "grid": "2.5 degree regular global grid (land+ocean); north-to-south rows",
        "slp_variable": (
            "sea-level pressure [Pa] (source file in millibars, x100 conversion "
            "applied here), monthly long-term mean (climatology)"
        ),
        "source_urls": [url],
        "builder": "scripts/build_ncep_slp_reference.py",
        "schema_version": SCHEMA_VERSION,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        slp_pa=slp_pa,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the local NCEP/NCAR Reanalysis 1 1991-2020 monthly SLP reference."
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=ROOT / "testing" / "reference_data" / "ncep_ncar_raw",
        help="Cache directory for the official NetCDF4 download.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "testing" / "reference_data" / "ncep_ncar_slp_1991_2020.npz",
        help="Output MonthlyClimatology NPZ path.",
    )
    args = parser.parse_args()
    output = build_reference(args.download_dir, args.output)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
