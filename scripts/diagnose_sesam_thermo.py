"""Diagnose the SESAM stage-P4 column energy/water closure on a saved state.

Diagnostic-only companion to `sesam_thermo.py` (docs/SESAM_GAP_ANALYSIS.md
Sec7, stage P4). This is the stage whose exit gate is stated concretely:
"water/energy conservation residuals at the column_water.py standard; raw
global P in [0.5, 5] mm/day without any target" -- i.e. does the (A44)
precipitation formula, run on the real saved state with *no* row-target
allocator anywhere in the call chain, land in a physically sane range on its
own.

Chain (reusing prior stages exactly as they left it, no re-derivation):

1. P1 vertical structure + P2 SLP/wind (`diagnose_sesam_wind._run_chain`),
   zonal-only wind -- module docstring note 7 of `sesam_synoptic.py` and the
   P3 driver's own choice: the full azonal-inflated chain would compound
   into any transport this script runs exactly as it would compound into
   local EKE production, so this script inherits that same documented
   choice rather than re-litigating it.
2. P3 local steady-state EKE (`sesam_synoptic.compute_synoptic`) from that
   wind -- *not* the full prognostic (A52) transport-to-equilibrium loop
   (`diagnose_sesam_synoptic.py`'s multi-hundred-second-per-season run):
   P3's own docstring already calls the local closure "a useful diagnostic
   in its own right", and this script's job is P4's water/energy closure,
   not re-measuring P3's own exit gate a second time.
3. P4 column-water step (`sesam_thermo.evolve_column_water_vapor`): near-
   surface specific humidity (saved `humidity` field, kg/kg) converted to a
   column-water depth via the *same* `water_scale=2000.0 kg/m^2` convention
   `atmosphere.py`'s own gated `enable_prognostic_column_water` path already
   uses for its lower-layer water mass (not a new, inconsistent conversion
   invented for this script).
4. P4 column-energy step (`sesam_thermo.evolve_column_energy`): reported
   only as a *pure-transport* conservation check (zero external diabatic
   forcing) -- see "Energy-closure scope" below for why a full real-state
   SWa/LWa forcing is not attempted here.

Evaporation placeholder (documented, not fabricated): the saved state has no
evaporation field, so this script computes a standard bulk-aerodynamic
estimate ``E = rho0 * Ch * |wind| * max(qsat(T_skin) - qa, 0)`` using the
saved ``wind_speed_avg`` field (the model's own recorded surface wind, not
the P2 reconstruction, which the P2 exit gate found still short of NCEP) and
a standard bulk transfer coefficient Ch=1.3e-3. This is *not* the project's
actual evaporation code (buried across a 6000-line `simulate.py`/`atmosphere.py`
land/ocean split this script does not attempt to extract) -- it exists only
to give the (A44) precipitation formula a physically-ordered E term, which
the equation genuinely needs as an input, not to be a validated evaporation
climatology.

Energy-closure scope (an honest limitation, not a hidden defect): (A40)'s
diabatic source needs atmosphere-absorbed SW and net atmosphere LW, which do
not exist as separable fields until stage P5 (radiation) is built -- the
supported model's radiation is a single-layer grey scheme that never
separates "absorbed by atmosphere" from "absorbed at surface"
(docs/SESAM_GAP_ANALYSIS.md Sec3.6). Rather than fabricate a SWa/LWa split
from fields the model does not track, this script measures the column-energy
kernel's conservation contract under zero external forcing (pure advection +
diffusion), which is exactly the same scope P3's own "pure-transport
conservation check" used for K, and reports it honestly as a kernel-level
check, not a real-state energy-budget validation. `testing/test_sesam_thermo.py`
already covers the nonzero-source algebra with hand-computed values.

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
import sesam_thermo as st  # noqa: E402
import sesam_vertical as sv  # noqa: E402
import sesam_wind as sw  # noqa: E402
from masks import get_masks  # noqa: E402
from planet_params import EARTH  # noqa: E402
from diagnose_sesam_wind import _run_chain, _sigma_oro_estimate, _seasonal_mean  # noqa: E402
from diagnose_sesam_slp import _lat_centers  # noqa: E402
from sim_grid import _coarsen as _block_mean_coarsen  # noqa: E402

SEASONS = {"DJF": (11, 0, 1), "JJA": (5, 6, 7)}
_LEVELS_M = np.array([0.0, 1500.0, 3000.0, 5500.0, 9000.0, 12000.0])
_TROPOPAUSE_M = 12000.0
_P0_PA = float(EARTH.surface_pressure_pa)
_RD = 287.0
_REFERENCE_TEMP_K = 273.15
_WATER_SCALE_KG_M2 = 2000.0  # matches atmosphere.py's _lower_water_scale fallback
_BULK_TRANSFER_COEFF = 1.3e-3  # standard ocean/land bulk aerodynamic Ch


def _load_state(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files if not name.startswith("__")}


def _coarsen_state(state: dict[str, np.ndarray], block_size: int) -> dict[str, np.ndarray]:
    if block_size <= 1:
        return state
    h, w = state["elevation"].shape
    if h % block_size != 0 or w % block_size != 0:
        raise ValueError(f"grid {h}x{w} is not evenly divisible by block_size={block_size}")
    hc, wc = h // block_size, w // block_size
    out: dict[str, np.ndarray] = {}
    for key, arr in state.items():
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape == (h, w):
            out[key] = _block_mean_coarsen(arr.astype(np.float32), hc, wc, block_size).astype(np.float64)
        elif arr.ndim == 3 and arr.shape[-2:] == (h, w):
            out[key] = np.stack(
                [_block_mean_coarsen(arr[k].astype(np.float32), hc, wc, block_size).astype(np.float64)
                 for k in range(arr.shape[0])]
            )
        else:
            out[key] = arr
    return out


def _slope_magnitude(elevation_m: np.ndarray) -> np.ndarray:
    """``|grad zs|`` (dimensionless), the (A45) slope input.

    Reuses `sesam_wind.horizontal_gradient` (per-radian spherical gradient)
    directly rather than a new finite-difference implementation; the zonal
    denominator's `cos(lat)` is floored at 65 deg poleward exactly like
    `sesam_synoptic.zonal_center_spacing_m` already does for the analogous
    polar singularity, so this script does not need a second polar-floor
    convention.
    """
    h, w = elevation_m.shape
    lat = sw._latitude_rad(h)
    d_dphi, d_dlam = sw.horizontal_gradient(elevation_m, lat)
    cos_lat = np.maximum(np.cos(lat), np.cos(np.deg2rad(65.0)))[:, None]
    dzdy = d_dphi / float(EARTH.radius_m)
    dzdx = d_dlam / (float(EARTH.radius_m) * cos_lat)
    return np.sqrt(dzdx**2 + dzdy**2)


def _evaporation_mm_day(
    skin_k: np.ndarray, qa_kg_kg: np.ndarray, wind_speed_m_s: np.ndarray, rho0: float
) -> np.ndarray:
    qsat_skin = sv.saturation_specific_humidity(skin_k, np.full_like(skin_k, _P0_PA))
    deficit = np.maximum(qsat_skin - qa_kg_kg, 0.0)
    flux_kg_m2_s = rho0 * _BULK_TRANSFER_COEFF * wind_speed_m_s * deficit
    return flux_kg_m2_s * 86400.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=Path, default=ROOT / "saves" / "test.npz")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--block-size", type=int, default=1,
        help="Block-average downsample factor applied before running the chain "
             "(1 = native resolution, the default).",
    )
    parser.add_argument(
        "--explicit-diffusion", action="store_true",
        help="Use the plain explicit diffusion kernels instead of the implicit-zonal "
             "remedy (only tractable at coarse --block-size; see "
             "sesam_thermo._linear_diffusion_step_implicit_zonal's docstring).",
    )
    parser.add_argument("--dt-days", type=float, default=0.25)
    args = parser.parse_args()

    state = _load_state(args.state)
    state = _coarsen_state(state, args.block_size)
    monthly_temp = state["monthly_temp"].astype(np.float64)
    elevation_m = np.clip(state["elevation"].astype(np.float64), 0.0, 1.0) * (
        float(EARTH.max_elevation_km) * 1000.0
    )
    sea, land = get_masks(state["elevation"], use_cache=False)
    land_mask = (np.asarray(land) > 0.5).astype(np.float64)
    surface_kind = np.where(land_mask > 0.5, 1, 0).astype(np.int64)
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
    wind_speed = np.maximum(state["wind_speed_avg"].astype(np.float64), 0.0)
    h, w = elevation_m.shape
    lat_deg = _lat_centers(h)
    rho0 = _P0_PA / (_RD * _REFERENCE_TEMP_K)
    slope_mag = _slope_magnitude(elevation_m)

    area, xlen, ylen = ss.spherical_transport_geometry(h, w, float(EARTH.radius_m))
    dx = ss.zonal_center_spacing_m(h, w, float(EARTH.radius_m))
    dy = float(EARTH.radius_m) * (np.pi / h)

    report: dict = {
        "state": str(args.state),
        "grid": [h, w],
        "block_size": args.block_size,
        "diffusion_scheme": "explicit" if args.explicit_diffusion else "implicit_zonal",
        "placeholders": {
            "evaporation": f"bulk aerodynamic proxy, Ch={_BULK_TRANSFER_COEFF} "
                            "(see module docstring; not the project's real evaporation code)",
            "column_water_scale_kg_m2": _WATER_SCALE_KG_M2,
            "eke": "P3 local steady state (production=dissipation), not the full "
                   "prognostic-transport-to-equilibrium loop",
            "column_energy": "pure-transport conservation check only (zero external "
                              "diabatic forcing) -- see module docstring 'Energy-closure scope'",
            "near_surface_rh": "diagnosed as qa/qsat(Ta, p0), not a saved field",
        },
        "seasons": {},
    }

    for season, months in SEASONS.items():
        skin = _seasonal_mean(monthly_temp, months)
        t2m = skin + ta_minus_skin
        t0 = time.perf_counter()
        slp, wind, wind_zonal, closure = _run_chain(
            skin, t2m, qa, elevation_m, surface_kind, z0, zoro
        )
        structure = sv.compute_vertical_structure(
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
        cd = sw.drag_coefficient(z0, zoro, surface_kind)
        lat = sw._latitude_rad(h)
        angle = sw.cross_isobar_angle(cd, lat, omega=float(EARTH.omega))
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
            rho0_kg_m3=rho0,
        )
        k_eke = syn.eddy_kinetic_energy_m2_s2

        evap = _evaporation_mm_day(skin, qa, wind_speed, rho0)
        qa_water_mm = qa * _WATER_SCALE_KG_M2
        # (A44)'s "ra" is *relative* humidity, distinct from the saved
        # "humidity" field itself (specific humidity, kg/kg -- see
        # atmosphere.py's own field docstring). Diagnosed here as
        # qa/qsat(Ta, p0), the same ratio sesam_thermo.surface_relative_humidity_star
        # uses for the skin-referenced r*, just anchored to Ta instead of T*.
        qsat_t2m = sv.saturation_specific_humidity(t2m, np.full_like(t2m, _P0_PA))
        ra = np.clip(qa / np.maximum(qsat_t2m, 1e-8), 0.0, 1.2)

        print(f"  [{season}] running P4 column-water step ({h}x{w})...", flush=True)
        t1 = time.perf_counter()
        water_step = st.evolve_column_water_vapor(
            qa_water_mm, evap,
            wind_zonal.surface_u_m_s, wind_zonal.surface_v_m_s,
            k_eke, ra, qa, slope_mag, land_mask,
            dx_m=dx, dy_m=dy, dt_days=args.dt_days,
            cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
            rho0_kg_m3=rho0,
            implicit_zonal_diffusion=not args.explicit_diffusion,
        )
        water_elapsed = time.perf_counter() - t1

        print(f"  [{season}] running P4 column-energy pure-transport check...", flush=True)
        t2 = time.perf_counter()
        zero_heating = np.zeros((h, w))
        energy_step = st.evolve_column_energy(
            t2m, zero_heating,
            wind_zonal.surface_u_m_s, wind_zonal.surface_v_m_s, k_eke,
            dx_m=dx, dy_m=dy, dt_days=args.dt_days,
            cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
            implicit_zonal_diffusion=not args.explicit_diffusion,
        )
        energy_elapsed = time.perf_counter() - t2

        global_p = float(np.sum(water_step.precipitation_mm_day * area) / np.sum(area))
        global_e = float(np.sum(evap * area) / np.sum(area))
        incumbent_p = float(np.sum(state["monthly_precip"][list(months)].mean(axis=0) * area) / np.sum(area)) \
            if "monthly_precip" in state else None

        report["seasons"][season] = {
            "runtime_s": {
                "chain": round(t1 - t0, 2),
                "water_step": round(water_elapsed, 2),
                "energy_step": round(energy_elapsed, 2),
            },
            "eke_mean_m2_s2": float(np.mean(k_eke)),
            "evaporation_global_mean_mm_day": global_e,
            "column_water": {
                "global_water_mean_mm": float(np.sum(water_step.water_mm.astype(np.float64) * area) / np.sum(area)),
                "global_precipitation_mm_day": global_p,
                "precipitation_in_target_range_0p5_5": bool(0.5 <= global_p <= 5.0),
                "incumbent_row_target_precipitation_mm_day": incumbent_p,
                "precipitation_percentiles_mm_day": {
                    str(p): float(np.percentile(water_step.precipitation_mm_day, p)) for p in (10, 50, 90, 99)
                },
                "convergence_mean_mm_day": float(np.mean(water_step.convergence_mm_day)),
                "slope_convergence_mean_mm_day": float(np.mean(water_step.slope_convergence_mm_day)),
                "residual_mm": water_step.residual_mm,
                "relative_residual": water_step.relative_residual,
                "advection_substeps": water_step.advection_substeps,
                "diffusion_substeps": water_step.diffusion_substeps,
            },
            "column_energy_pure_transport": {
                "temperature_mean_k_before": float(np.mean(t2m)),
                "temperature_mean_k_after": float(np.mean(energy_step.temperature_k)),
                "residual_k": energy_step.residual_k,
                "relative_residual": energy_step.relative_residual,
                "advection_substeps": energy_step.advection_substeps,
                "diffusion_substeps": energy_step.diffusion_substeps,
            },
        }

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
            print(f"  [{season}] partial report written to {args.output}", flush=True)

    text = json.dumps(report, indent=1)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
