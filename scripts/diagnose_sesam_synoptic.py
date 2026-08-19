"""Diagnose the SESAM stage-P3 synoptic/EKE closure on a saved PlanetSim state.

Diagnostic-only companion to `sesam_synoptic.py` (docs/SESAM_GAP_ANALYSIS.md
§7, stage P3). It runs the Appendix-A5 EKE closure on a saved state for
DJF/JJA: the Eady-baroclinicity production, the drag dissipation, and the
resulting diagnostic steady-state EKE (local production/dissipation balance).
It reports the EKE field statistics and storm track, the synoptic
surface-wind component, and the *total* surface wind — the stage-P2 exit
gate showed the P2 surface wind lacked this synoptic gustiness term, so P3
closes that missing piece.

It then runs the full (A52) prognostic closure -- transport (advection by
the zonal-only P2 wind, nonlinear diffusion by AT) plus the same
production/dissipation -- forward from that local steady state toward its
own equilibrium (`evolve_eke`, `sesam_synoptic.py`), and reports whether
storm-track placement still tracks baroclinicity once transport is active,
and how the resulting AT/Aq compare to the incumbent fixed-window eddy term
(`eddy_heat_flux_coeff`, `simulate.py` ~line 5803) at the same grid.

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
from sim_grid import _coarsen as _block_mean_coarsen  # noqa: E402

SEASONS = {"DJF": (11, 0, 1), "JJA": (5, 6, 7)}
_LEVELS_M = np.array([0.0, 1500.0, 3000.0, 5500.0, 9000.0, 12000.0])
_TROPOPAUSE_M = 12000.0
_P0_PA = float(EARTH.surface_pressure_pa)


def _load_state(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files if not name.startswith("__")}


def _coarsen_state(state: dict[str, np.ndarray], block_size: int) -> dict[str, np.ndarray]:
    """Block-average every ``(H, W)``/``(N, H, W)`` field in ``state`` down by
    ``block_size`` (reusing `sim_grid._coarsen`, the same block-mean utility
    `sesam_dynamics.py`'s resolution-matching already uses). Fast-iteration
    path only -- the default is ``block_size=1`` (a no-op) now that
    `sesam_synoptic.eke_diffusion_step_implicit_zonal` makes the native
    512x1024 grid itself tractable (docs/SESAM_GAP_ANALYSIS.md P3,
    2026-08-18 follow-up entry).
    """
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


def _incumbent_effective_diffusivity_m2_s(eddy_heat_flux_coeff_per_day: float, dy_m: float) -> float:
    """Convert the incumbent `eddy_heat_flux_coeff` to a diffusivity [m^2/s].

    `simulate.py`'s Feature-7 block (~line 5803-5832) updates
    ``T_sst += eddy_k * T_lap_y * eddy_lat * dt_sub`` where ``T_lap_y`` is the
    *unnormalised* discrete second difference
    ``T[i-1] - 2*T[i] + T[i+1]`` (no division by ``dy**2``). For a smooth
    field that finite difference approximates ``d2T/dy2 * dy**2``, so the
    block is equivalent to a standard diffusion step
    ``dT/dt = D_eff * d2T/dy2`` with ``D_eff = eddy_k * dy**2`` (``eddy_k`` in
    1/day here, so ``D_eff`` comes out in m^2/day; converted to m^2/s to
    match `sesam_synoptic`'s AT units).
    """
    d_eff_m2_day = float(eddy_heat_flux_coeff_per_day) * float(dy_m) ** 2
    return d_eff_m2_day / 86400.0


def _run_prognostic_transport(
    *,
    k0: np.ndarray,
    production: np.ndarray,
    drag_coefficient: np.ndarray,
    wind_zonal,
    lat_deg: np.ndarray,
    radius_m: float,
    dt_days: float = 0.25,
    max_iterations: int = 200,
    convergence_tol: float = 1e-4,
    implicit_zonal_diffusion: bool = True,
    verbose: bool = False,
) -> dict:
    """Run the full (A52) closure forward from `k0` toward its own equilibrium.

    Frozen forcing (production/dissipation inputs held at their local
    steady-state values for this season) -- this measures what transport
    alone does to the local closure's answer, holding baroclinicity fixed,
    which is the direct question the P3 exit gate asks (does the picture
    still track baroclinicity once transport is active). Advection uses the
    stage-P2 *zonal-only* surface wind (`wind_zonal.surface_u_m_s`/
    `surface_v_m_s`), per the module docstring's note 7 / the §10 finding
    that the full azonal-inflated chain would compound into the transported
    field exactly as it would compound into local production.

    ``dt_days`` defaults to a short 0.25-day coupling step, not a larger one,
    for a real reason found while building this driver: `evolve_eke`
    operator-splits advection before diffusion (module docstring note 6), so
    within *one* call the entire ``dt_days`` worth of advection runs before
    diffusion gets any chance to relax it. This state's zonal wind has a real
    (physical, not a bug) convergence zone; measured directly, 0.1 day of
    advection alone already raised the field's max by ~45% (6202 -> 9018)
    with zero smoothing counteracting it yet. At ``dt_days=5`` that compounds
    into an extreme, numerically stiff spike (measured: max K 131,112 after
    one 5-day advection sub-phase, 21x the pre-advection max) that then
    demands an impractical number of diffusion sub-steps to relax within the
    same call. A short coupling step interleaves advection and diffusion
    frequently enough that neither ever runs far ahead of the other -- the
    same reason real coupled climate models use a short coupling timestep
    between separately time-stepped processes, not a kernel defect in either
    `eke_diffusion_step` or the reused `evolve_column_water`.

    ``implicit_zonal_diffusion`` (default ``True``) routes the diffusive
    term through `sesam_synoptic.eke_diffusion_step_implicit_zonal` instead
    of the plain explicit `eke_diffusion_step`: at the grid's native
    512x1024 resolution, the plain explicit scheme's zonal (east-west)
    diffusion term is pole-stiff enough to need on the order of 1-20 million
    substeps for a single one of these 0.25-day calls (measured directly;
    see the P3 2026-08-18 gap-analysis entry), which is what made a full-
    resolution run impractical in the prior session. Pass ``False`` to force
    the original explicit path -- only tractable at coarse ``--block-size``.
    """
    h, w = k0.shape
    area, xlen, ylen = ss.spherical_transport_geometry(h, w, radius_m)
    dx = ss.zonal_center_spacing_m(h, w, radius_m)
    dy = radius_m * (np.pi / h)
    u = np.asarray(wind_zonal.surface_u_m_s, dtype=np.float64)
    v = np.asarray(wind_zonal.surface_v_m_s, dtype=np.float64)

    k = np.asarray(k0, dtype=np.float64)
    mean_history = [float(np.mean(k))]
    iterations_run = 0
    for iterations_run in range(1, max_iterations + 1):
        step = ss.evolve_eke(
            k, production, drag_coefficient, u, v,
            dx_m=dx, dy_m=dy, dt_days=dt_days,
            cell_area_m2=area, x_face_length_m=xlen, y_face_length_m=ylen,
            implicit_zonal_diffusion=implicit_zonal_diffusion,
        )
        new_mean = float(np.mean(step.eke_m2_s2))
        mean_history.append(new_mean)
        rel_change = abs(new_mean - mean_history[-2]) / max(abs(new_mean), 1.0)
        k = step.eke_m2_s2
        if verbose:
            print(
                f"    [prognostic] iter {iterations_run:3d}  mean K={new_mean:9.2f}  "
                f"max K={float(np.max(k)):10.2f}  rel_change={rel_change:.2e}  "
                f"diff_substeps={step.diffusion_substeps}  adv_substeps={step.advection_substeps}",
                flush=True,
            )
        if rel_change < convergence_tol:
            break

    at_final = ss.horizontal_diffusion_coefficient(k)
    aq_final = ss.moisture_diffusion_coefficient(k)
    zonal_k = k.mean(axis=1)
    return {
        "diffusion_scheme": "implicit_zonal" if implicit_zonal_diffusion else "explicit",
        "iterations_run": iterations_run,
        "converged": bool(rel_change < convergence_tol),
        "days_simulated": iterations_run * dt_days,
        "mean_history_m2_s2": mean_history,
        "eke_mean_m2_s2": float(np.mean(k)),
        "eke_percentiles_m2_s2": {str(p): float(np.percentile(k, p)) for p in (50, 90, 99)},
        "storm_track_latitude_deg": _storm_track_latitude_deg(k, lat_deg),
        "diffusion_heat_mean_m2_s": float(np.mean(at_final)),
        "diffusion_moisture_mean_s": float(np.mean(aq_final)),
        "diffusion_heat_zonal_profile": {
            str(int(lat_deg[j])): float(
                ss.horizontal_diffusion_coefficient(np.array([[zonal_k[j]]]))[0, 0]
            )
            for j in np.linspace(5, h - 6, 12, dtype=int)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=Path, default=ROOT / "saves" / "test.npz")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--block-size", type=int, default=1,
        help="Block-average downsample factor applied to the loaded state before running the "
             "chain (1 = native resolution, the default -- e.g. saves/test.npz is natively "
             "512x1024, so block-size=4 reproduces the earlier 128x256 fast-iteration screen).",
    )
    parser.add_argument(
        "--explicit-diffusion", action="store_true",
        help="Use the plain explicit eke_diffusion_step for the (A52) prognostic transport "
             "instead of eke_diffusion_step_implicit_zonal (the default). Only tractable at "
             "coarse --block-size -- see sesam_synoptic.eke_diffusion_step_implicit_zonal's "
             "docstring for why the explicit scheme is impractical at native resolution.",
    )
    parser.add_argument(
        "--verbose-prognostic", action="store_true",
        help="Print per-iteration progress of the (A52) prognostic transport loop.",
    )
    args = parser.parse_args()

    state = _load_state(args.state)
    state = _coarsen_state(state, args.block_size)
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
        "block_size": args.block_size,
        "prognostic_diffusion_scheme": "explicit" if args.explicit_diffusion else "implicit_zonal",
        "placeholders": {
            "t2m": "skin_scenario + (Ta_inst - skin_inst); exact for the instantaneous scenario",
            "humidity": "saved instantaneous humidity",
            "tropopause_m": _TROPOPAUSE_M,
            "EKE": "local steady state feeds the (A52) prognostic transport run as its initial condition",
        },
        "incumbent_eddy_heat_flux_coeff": {
            "value_per_day": 0.006,
            "note": "simulate.py PlanetParams.eddy_heat_flux_coeff default; Feature 7 block ~line 5803",
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

        # --- (A52) prognostic transport, run forward toward equilibrium ---
        print(f"  [{season}] starting prognostic transport ({h}x{w}, "
              f"diffusion={'explicit' if args.explicit_diffusion else 'implicit_zonal'})...", flush=True)
        t1 = time.perf_counter()
        prognostic = _run_prognostic_transport(
            k0=syn.eddy_kinetic_energy_m2_s2,
            production=syn.production_m2_s3,
            drag_coefficient=cd,
            wind_zonal=wind_zonal,
            lat_deg=lat_deg,
            radius_m=float(EARTH.radius_m),
            implicit_zonal_diffusion=not args.explicit_diffusion,
            verbose=args.verbose_prognostic,
        )
        prognostic["runtime_s"] = round(time.perf_counter() - t1, 2)
        print(f"  [{season}] prognostic transport done in {prognostic['runtime_s']:.1f}s "
              f"({prognostic['iterations_run']} iterations, converged={prognostic['converged']})", flush=True)
        dy_m = float(EARTH.radius_m) * (np.pi / h)
        incumbent_d_eff = _incumbent_effective_diffusivity_m2_s(0.006, dy_m)
        prognostic["incumbent_comparison"] = {
            "incumbent_effective_diffusivity_m2_s": incumbent_d_eff,
            "sesam_AT_over_incumbent_ratio": prognostic["diffusion_heat_mean_m2_s"] / incumbent_d_eff,
            "local_steady_state_AT_m2_s": float(np.mean(syn.diffusion_coefficient_heat_m2_s)),
            "AT_change_from_transport_pct": 100.0 * (
                prognostic["diffusion_heat_mean_m2_s"] / float(np.mean(syn.diffusion_coefficient_heat_m2_s)) - 1.0
            ),
        }
        prognostic["storm_track_shift_from_local_deg"] = (
            prognostic["storm_track_latitude_deg"]
            - report["seasons"][season]["storm_track_latitude_deg"]
        )
        report["seasons"][season]["prognostic_transport"] = prognostic

        # Persist after every season completes, not only at the very end --
        # the (A52) prognostic transport loop is the slow part of this
        # script (tens of minutes per season at native 512x1024 resolution),
        # so a single-write-at-the-end policy loses an entire completed
        # season's result to any interruption between seasons. Cheap
        # (JSON dump of a small report dict) relative to the run itself.
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
