"""Run an ExoPlaSim offline reference from a PlanetSim exchange request.

Run this script *inside a Linux environment containing ExoPlaSim*; it is not
imported by PlanetSim's interactive runtime.  It writes boundary maps through
ExoPlaSim's own SRA writer, runs a spin-up followed by an evaluation segment,
and normalizes the postprocessed NPZ result to PlanetSim's monthly-climatology
contract for direct CRU scoring.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_dycore import canonicalize_exoplasim_output, load_exoplasim_request  # noqa: E402


def _remove_spinup_products(model_workdir: Path) -> None:
    """Discard postprocessed spin-up fields but retain PlaSim's restart state."""
    for path in model_workdir.glob("MOST*.npz"):
        path.unlink()
    for path in model_workdir.glob("MOST_DIAG.*"):
        path.unlink()
    snapshots = model_workdir / "snapshots"
    if snapshots.exists():
        shutil.rmtree(snapshots)


def _evaluation_mean(model_workdir: Path, temperature_key: str, precipitation_key: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean the twelve-month archives from every retained evaluation year."""
    archives = sorted(
        path
        for path in model_workdir.glob("MOST*.npz")
        if not path.name.endswith("_metadata.npz")
    )
    if not archives:
        raise RuntimeError("ExoPlaSim did not retain any evaluation archives")
    temperature_sum: np.ndarray | None = None
    precipitation_sum: np.ndarray | None = None
    for path in archives:
        with np.load(path, allow_pickle=False) as archive:
            missing = [key for key in (temperature_key, precipitation_key) if key not in archive.files]
            if missing:
                raise RuntimeError(f"{path.name} lacks requested output field(s): {', '.join(missing)}")
            temperature = np.asarray(archive[temperature_key], dtype=np.float64)
            precipitation = np.asarray(archive[precipitation_key], dtype=np.float64)
        if temperature_sum is None:
            temperature_sum = np.zeros_like(temperature)
            precipitation_sum = np.zeros_like(precipitation)
        if temperature.shape != temperature_sum.shape or precipitation.shape != precipitation_sum.shape:
            raise RuntimeError("ExoPlaSim evaluation archives have inconsistent shapes")
        temperature_sum += temperature
        precipitation_sum += precipitation
    assert temperature_sum is not None and precipitation_sum is not None
    count = len(archives)
    return temperature_sum / count, precipitation_sum / count, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ncpus", type=int, default=1)
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="compile/cache the requested ExoPlaSim executable without running a climate integration",
    )
    parser.add_argument("--temperature-key", default="tas", help="near-surface air temperature")
    parser.add_argument("--precipitation-key", default="pr")
    parser.add_argument(
        "--precipitation-units",
        choices=("mm_day", "kg_m2_s", "m_s"),
        default="m_s",
        help="units of ExoPlaSim `pr` (current NPZ metadata reports m_s)",
    )
    args = parser.parse_args()
    if args.ncpus < 1:
        parser.error("--ncpus must be positive")
    try:
        import exoplasim as exo
        from exoplasim.randomcontinents import writeSRA
    except ImportError as exc:
        raise SystemExit(
            "ExoPlaSim is required in this Linux environment; install its documented "
            "Python and GNU compiler prerequisites before running this wrapper."
        ) from exc

    arrays, metadata = load_exoplasim_request(args.request)
    request = metadata["request"]
    planet = metadata["planet"]
    if args.build_only:
        build_workdir = args.workdir.resolve() / "build_only"
        build_workdir.parent.mkdir(parents=True, exist_ok=True)
        model = exo.Model(
            resolution=request["resolution"],
            layers=int(request["layers"]),
            ncpus=args.ncpus,
            workdir=str(build_workdir),
            modelname="planetsim_reference_build",
            outputtype=".npz",
        )
        print(model.executable)
        return 0
    if args.output is None:
        parser.error("--output is required unless --build-only is set")
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    h, w = arrays["topography_m"].shape
    # ``writeSRA`` appends ``_surf_####.sra`` itself; pass stems and configure
    # the exact resulting files (not the stems) into PlaSim.
    landmap_stem = workdir / "planetsim_landmask"
    topomap_stem = workdir / "planetsim_topography"
    landmap = Path(f"{landmap_stem}_surf_0172.sra")
    topomap = Path(f"{topomap_stem}_surf_0129.sra")
    # ExoPlaSim's topographic boundary condition is surface geopotential.
    writeSRA(str(landmap_stem), 172, (arrays["land_fraction"] >= 0.5).astype(float), h, w)
    writeSRA(
        str(topomap_stem),
        129,
        arrays["topography_m"] * float(planet["surface_gravity_m_s2"]),
        h,
        w,
    )
    model = exo.Earthlike(
        resolution=request["resolution"],
        layers=int(request["layers"]),
        ncpus=args.ncpus,
        workdir=str(workdir / "model"),
        modelname="planetsim_reference",
        outputtype=".npz",
    )
    # Retain only the exact fields required by the bridge.  This makes a
    # genuine multi-year climatology inexpensive to preserve and avoids the
    # accidental "last year only" interpretation of Model.finalize().
    model.cfgpostprocessor(variables=[args.temperature_key, args.precipitation_key], times=12)
    model.configure(
        flux=float(planet["solar_constant_w_m2"]),
        gravity=float(planet["surface_gravity_m_s2"]),
        radius=float(planet["radius_m"]) / 6_371_000.0,
        pressure=float(planet["surface_pressure_pa"]) / 100_000.0,
        rotationperiod=float(planet["sidereal_day_hours"]) / 24.0,
        year=float(planet["orbital_period_days"]),
        obliquity=float(planet["obliquity_deg"]),
        fixedorbit=True,
        landmap=str(landmap),
        topomap=str(topomap),
    )
    model.exportcfg()
    if int(request["spinup_years"]):
        model.run(years=int(request["spinup_years"]), crashifbroken=True)
        _remove_spinup_products(Path(model.workdir))
    model.run(years=int(request["evaluation_years"]), crashifbroken=True)
    temperature, precipitation, actual_evaluation_years = _evaluation_mean(
        Path(model.workdir), args.temperature_key, args.precipitation_key
    )
    raw_output = workdir / "planetsim_exoplasim_raw"
    model.finalize(str(raw_output), allyears=True)
    mean_archive = raw_output / "planetsim_reference_eval_mean.npz"
    np.savez_compressed(
        mean_archive,
        **{args.temperature_key: temperature.astype(np.float32), args.precipitation_key: precipitation.astype(np.float32)},
        evaluation_years=np.asarray(actual_evaluation_years, dtype=np.int32),
    )
    canonicalize_exoplasim_output(
        mean_archive,
        args.request,
        args.output,
        temperature_key=args.temperature_key,
        precipitation_key=args.precipitation_key,
        precipitation_units=args.precipitation_units,
        runner_provenance={
            "engine_version": str(getattr(exo, "__version__", "unknown")),
            "ncpus": args.ncpus,
            "evaluation_years_actual": actual_evaluation_years,
            "evaluation_mean": "arithmetic mean of per-year monthly fields",
        },
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
