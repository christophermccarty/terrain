"""Diagnose the SESAM stage-P3 synoptic/EKE closure on a saved PlanetSim state.

Diagnostic-only companion to `sesam_synoptic.py` (docs/SESAM_GAP_ANALYSIS.md
§7, stage P3). It runs the Appendix-A5 EKE closure on a saved state for
DJF/JJA: the Eady-baroclinicity production, the drag dissipation, and the
resulting diagnostic steady-state EKE (local production/dissipation balance;
transport of K is stage P4). It reports the EKE field statistics and storm
track, the synoptic surface-wind component, and the *total* surface wind —
the stage-P2 exit gate showed the P2 surface wind lacked this synoptic
gustiness term, so P3 closes that missing piece.

The vertical structure (T, θ, p profiles), the surface wind, the drag
coefficient, and the cross-isobar angle come from the stage-P1/P2 chain
(`scripts/diagnose_sesam_wind.py`). Placeholder policy matches that driver
(documented, not fabricated physics).

Nothing here touches the supported climate path; the script only reads the
save and writes a JSON report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import sesam_synoptic as ss  # noqa: E402
import sesam_vertical as sv  # noqa: E402
import sesam_wind as sw  # noqa: E402
from masks import get_masks  # noqa: E402
from planet_params import EARTH  # noqa: E402
from diagnose_sesam_wind import _run_chain, _sigma_oro_estimate, _seasonal_mean  # noqa: E402
from diagnose_sesam_slp import _lat_centers  # noqa: E402

SEASONS = {"DJF": (11, 0, 1), "JJA": (5, 6, 7)}
_LEVELS_M = np.array([0.0, 1500.0, 3000.0, 5500.0, 9000.0, 12000.0])
_TROPOPAUSE_M = 12000.0
_P0_PA = float(EARTH.surface_pressure_pa)


def _load_state(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files if not name.startswith("__")}


def _storm_track_latitude_deg(eke: np.ndarray, lat_deg: np.ndarray) -> float:
    return float(lat_deg[int(np.argmax(eke.mean(axis=1)))])


def _vertical_structure(skin, t2m, qa, elevation_m, surface_kind):
    h, w = elevation_m.shape
    return sv.compute_vertical_structure(
        _LEVELS_M,
        near_surface_air_temp_k=t2m,
        skin_temp_k=skin,
        surface_kind=surface_kind,
        near_surface_specific_humidity_kgkg=qa,
        surface_elevation_m=elevation_m,
        tropopause_height_m=np.full((h, w), _TROPOPAUSE_M),
        p0_pa=_P0_PA,
        gravity=float(EARTH.surface_gravity),
        reference_temp_k=288.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=Path, default=ROOT / "saves" / "test.npz")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    state = _load_state(args.state)
    monthly_temp = state["monthly_temp"].astype(np.float64)
    elevation_m = np.clip(state["elevation"].astype(np.float64), 0.0, 1.0) * (
        float(EARTH.max_elevation_km) * 1000.0
    )
    sea, land = get_masks(state["elevation"], use_cache=False)
    surface_kind = np.where(np.asarray(land) > 0.5, 1, 0).astype(np.int64)
    if "land_ice_thickness" in state:
        surface_kind = np.where(
            np.asarray(state["land_ice_thickness"]) > 0.0, 2, surface_kind
        ).astype(np.int64)
    z0 = np.where(surface_kind == 0, 0.0, 0.1)
    zoro = sw.oro_roughness_m(_sigma_oro_estimate(elevation_m))
    t2m_inst = state["air_temperature"].astype(np.float64)
    skin_inst = state["temperature"].astype(np.float64)
    ta_minus_skin = t2m_inst - skin_inst
    qa = np.clip(state["humidity"].astype(np.float64), 0.0, 1.0)
    h, w = elevation_m.shape
    lat = sw._latitude_rad(h)
    lat_deg = _lat_centers(h)
    cd = sw.drag_coefficient(z0, zoro, surface_kind)
    angle = sw.cross_isobar_angle(cd, lat, omega=float(EARTH.omega))

    report: dict = {
        "state": str(args.state),
        "grid": [h, w],
        "placeholders": {
            "t2m": "skin_scenario + (Ta_inst - skin_inst); exact for the instantaneous scenario",
            "humidity": "saved instantaneous humidity",
            "tropopause_m": _TROPOPAUSE_M,
            "EKE": "local production/dissipation steady state (transport of K is P4)",
        },
        "seasons": {},
    }

    for season, months in SEASONS.items():
        skin = _seasonal_mean(monthly_temp, months)
        t2m = skin + ta_minus_skin
        # P2 chain: SLP + 3-D/full surface wind.
        slp, wind, wind_zonal, closure = _run_chain(
            skin, t2m, qa, elevation_m, surface_kind, z0, zoro
        )
        # P1 vertical structure for the potential-temperature/pressure profiles.
        structure = _vertical_structure(skin, t2m, qa, elevation_m, surface_kind)

        t0 = time.perf_counter()
        # Drive the EKE closure with the zonal-only P2 wind (the sane
        # circulation): the full-chain wind inherits the stage-P2 azonal
        # input-conditioning inflation, which would compound into EKE.
        syn = ss.compute_synoptic(
            potential_temperature_k=structure.potential_temperature_k,
            u_wind_z=wind_zonal.u_z_m_s,
            v_wind_z=wind_zonal.v_z_m_s,
            pressure_z=structure.pressure_pa,
            levels_m=_LEVELS_M,
            surface_u_m_s=wind_zonal.surface_u_m_s,
            surface_v_m_s=wind_zonal.surface_v_m_s,
            surface_elevation_m=elevation_m,
            surface_kind=surface_kind,
            drag_coefficient=cd,
            epsilon=angle["epsilon"],
            cos_alpha=angle["cos_alpha"],
            gravity=float(EARTH.surface_gravity),
            omega=float(EARTH.omega),
            rho0_kg_m3=_P0_PA / (287.0 * 273.15),
        )
        elapsed = time.perf_counter() - t0

        zonal_eke = syn.eddy_kinetic_energy_m2_s2.mean(axis=1)
        report["seasons"][season] = {
            "runtime_s": round(elapsed, 2),
            "eke_mean_m2_s2": float(np.mean(syn.eddy_kinetic_energy_m2_s2)),
            "eke_percentiles_m2_s2": {
                str(p): float(np.percentile(syn.eddy_kinetic_energy_m2_s2, p))
                for p in (50, 90, 99)
            },
            "storm_track_latitude_deg": _storm_track_latitude_deg(
                syn.eddy_kinetic_energy_m2_s2, lat_deg
            ),
            "brunt_vaisala_band_mean": {
                f"{int(lo)}": float(
                    syn.brunt_vaisala_frequency[(lat_deg >= lo) & (lat_deg < hi)].mean()
                )
                for lo, hi in [(20, 40), (40, 60), (-60, -40), (-40, -20)]
            },
            "production_mean_m2_s3": float(np.mean(syn.production_m2_s3)),
            "dissipation_mean_m2_s3": float(np.mean(syn.dissipation_m2_s3)),
            "diffusion_heat_mean_m2_s": float(np.mean(syn.diffusion_coefficient_heat_m2_s)),
            "synoptic_surface_wind_mean_m_s": float(np.mean(syn.synoptic_surface_wind_m_s)),
            "surface_wind_mean_m_s": float(
                np.mean(np.sqrt(wind_zonal.surface_u_m_s**2 + wind_zonal.surface_v_m_s**2))
            ),
            "total_wind_mean_m_s": float(np.mean(syn.total_wind_m_s)),
            "wind_stress_mean_pa": float(
                np.mean(np.sqrt(syn.wind_stress_u_pa**2 + syn.wind_stress_v_pa**2))
            ),
            "eke_zonal_profile": {
                str(int(lat_deg[j])): float(zonal_eke[j])
                for j in np.linspace(5, h - 6, 12, dtype=int)
            },
        }

    text = json.dumps(report, indent=1)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
