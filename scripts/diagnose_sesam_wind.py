"""Diagnose the SESAM stage-P2 wind assembly against the prescribed-cell generator.

Diagnostic-only companion to `sesam_wind.py` (docs/SESAM_GAP_ANALYSIS.md §7,
stage P2 second sub-deliverable). It runs the full Appendix-A2 chain on a
saved PlanetSim state — SLP from `sesam_dynamics.py`, wind from
`sesam_wind.py`, vertical structure from `sesam_vertical.py` — for DJF, JJA,
and the instantaneous saved fields, then:

1. scores the reconstructed surface wind speed against the NCEP/NCAR
   Reanalysis 1 1991-2020 wind climatology (raw uwnd/vwnd components, read
   with the same grid handling as `scripts/build_ncep_wind_reference.py`);
2. runs the jet/Hadley circulation scorecard
   (`circulation_diagnostics.jet_core_properties` / `hadley_edges_deg`)
   head-to-head: SESAM-reconstructed winds vs the saved state's own
   prescribed-cell winds vs NCEP — the P2 exit-gate comparison; and
3. reports the placeholder closures: the SLP stage's `sin_cos_alpha_bar`
   and Charney–Eliassen `u500` are now computed (two-pass closure: SLP
   without the orographic term -> wind -> u500 -> SLP with the term),
   replacing the first sub-deliverable's documented placeholders.

Placeholder policy for this driver (documented, not fabricated physics):
tropopause 12 km uniform (P5 closes it); the save stores no monthly
air-temperature or humidity means, so (a) humidity is the saved instantaneous
field for all scenarios and (b) each scenario's near-surface air temperature
is anchored as ``Ta = skin_scenario + (Ta_instantaneous − skin_instantaneous)``
— the (A9) near-surface lapse Γ = (Ta − T*)/zpbl is catastrophically
sensitive to a season-mismatched Ta/T* pair (a January Ta with a July skin
yields a −50 K/km super-adiabatic inverted profile, measured on this save),
so the instantaneous Ta − T* difference is carried as the scenario's; the
instantaneous scenario itself is exact. Roughness z0 = 0.1 m over land (the
project's boundary-layer value), σ_oro estimated as the ~2.5° block standard
deviation of elevation; the reference temperature for the scale height is
288 K.

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

import sesam_dynamics as sd  # noqa: E402
import sesam_vertical as sv  # noqa: E402
import sesam_wind as sw  # noqa: E402
from circulation_diagnostics import hadley_edges_deg, jet_core_properties  # noqa: E402
from masks import get_masks  # noqa: E402
from planet_params import EARTH  # noqa: E402
from build_ncep_wind_reference import _read_wind_component  # noqa: E402
from diagnose_sesam_slp import (  # noqa: E402
    _area_weighted_correlation,
    _downsample_to_reference,
    _lat_centers,
    _zonal_correlation,
)

SEASONS = {"DJF": (11, 0, 1), "JJA": (5, 6, 7)}
_LEVELS_M = np.array([0.0, 1500.0, 3000.0, 5500.0, 9000.0, 12000.0])
_TROPOPAUSE_M = 12000.0
_REFERENCE_TEMP_K = 273.15
_SCALE_HEIGHT_REF_K = 288.0
_ROUGHNESS_LAND_M = 0.1
_P0_PA = float(EARTH.surface_pressure_pa)


def _load_state(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files if not name.startswith("__")}


def _seasonal_mean(monthly: np.ndarray, months: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(monthly[list(months)], dtype=np.float64).mean(axis=0)


def _sigma_oro_estimate(elevation_m: np.ndarray, block: int = 8) -> np.ndarray:
    """~2.5deg block standard deviation of elevation (A23 input estimate)."""
    h, w = elevation_m.shape
    padded_h = (h // block) * block
    padded_w = (w // block) * block
    z = elevation_m[:padded_h, :padded_w]
    blocks = z.reshape(padded_h // block, block, padded_w // block, block)
    std = blocks.std(axis=(1, 3))
    # Upsample the block std back to the full grid (nearest).
    out = np.repeat(np.repeat(std, block, axis=0), block, axis=1)
    full = np.zeros_like(elevation_m)
    full[:padded_h, :padded_w] = out
    full[:, padded_w:] = full[:, :1]
    full[padded_h:, :] = full[:1, :]
    return full


def _run_chain(
    skin: np.ndarray,
    t2m: np.ndarray,
    qa: np.ndarray,
    elevation_m: np.ndarray,
    surface_kind: np.ndarray,
    z0: np.ndarray,
    zoro: np.ndarray,
) -> tuple[sd.SesamSlp, sw.SesamWind, sw.SesamWind, dict]:
    """Two-pass SLP<->wind closure: SLP -> wind -> u500 -> SLP(+CE) -> wind."""
    h, w = skin.shape
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
        reference_temp_k=_SCALE_HEIGHT_REF_K,
    )
    cd = sw.drag_coefficient(z0, zoro, surface_kind)
    lat = sw._latitude_rad(h)
    angle = sw.cross_isobar_angle(cd, lat, omega=float(EARTH.omega))
    scab_zonal = angle["sin_cos_alpha"].mean(axis=1)

    def slp_with(u500: np.ndarray | None) -> sd.SesamSlp:
        return sd.compute_slp(
            skin_temp_k=skin,
            surface_elevation_m=elevation_m,
            sin_cos_alpha_bar=scab_zonal,
            gravity=float(EARTH.surface_gravity),
            radius_m=float(EARTH.radius_m),
            omega=float(EARTH.omega),
            p0_pa=_P0_PA,
            reference_temp_k=_REFERENCE_TEMP_K,
            u500_m_s=u500,
            tropopause_height_m=np.full(h, _TROPOPAUSE_M) if u500 is not None else None,
        )

    sigma_trop = float(structure.pressure_pa[-1, 0, 0] / structure.pressure_pa[0, 0, 0])

    def wind_for(slp: np.ndarray) -> sw.SesamWind:
        return sw.compute_wind(
            slp_pa=slp,
            temperature_z=structure.temperature_k,
            levels_m=_LEVELS_M,
            pressure_z=structure.pressure_pa,
            skin_temp_k=skin,
            t2m_k=t2m,
            surface_elevation_m=elevation_m,
            surface_kind=surface_kind,
            roughness_m=z0,
            sigma_oro_m=zoro / sw._ORO_ROUGHNESS_FACTOR,
            tropopause_sigma=sigma_trop,
            gravity=float(EARTH.surface_gravity),
            radius_m=float(EARTH.radius_m),
            omega=float(EARTH.omega),
            rho0_kg_m3=_P0_PA / (287.0 * _REFERENCE_TEMP_K),
            reference_temp_k=_REFERENCE_TEMP_K,
        )

    slp1 = slp_with(None)
    wind1 = wind_for(slp1.slp_pa)
    slp2 = slp_with(wind1.u500_pa_zonal_m_s)
    wind2 = wind_for(slp2.slp_pa)
    # Zonal-only decomposition: same wind chain driven by the SLP field with
    # the azonal parts removed (p0 + zonal anomaly). Separates the zonal
    # cell-physics channel from the azonal (A37)/(A39) channel.
    slp_zonal_only = slp2.p0_pa + slp2.zonal_slp_anomaly_pa[:, None] * np.ones((h, w))
    wind_zonal = wind_for(slp_zonal_only)
    closure = {
        "u500_pass1_rms_m_s": float(np.sqrt(np.mean(wind1.u500_pa_zonal_m_s**2))),
        "u500_final_rms_m_s": float(np.sqrt(np.mean(wind2.u500_pa_zonal_m_s**2))),
        "slp_orographic_range_hpa": [
            float(slp2.orographic_azonal_slp_pa.min() / 100.0),
            float(slp2.orographic_azonal_slp_pa.max() / 100.0),
        ],
        "sin_cos_alpha_zonal_mean_range": [
            float(scab_zonal.min()),
            float(scab_zonal.max()),
        ],
        "zonal_only_surface_speed_mean_m_s": float(
            np.mean(np.sqrt(wind_zonal.surface_u_m_s**2 + wind_zonal.surface_v_m_s**2))
        ),
        "zonal_only_scorecard": _scorecard_report(wind_zonal.surface_u_m_s, wind_zonal.surface_v_m_s),
    }
    return slp2, wind2, wind_zonal, closure


def _scorecard_report(u: np.ndarray, v: np.ndarray) -> dict:
    return {
        "jet_core": jet_core_properties(u),
        "hadley_edge_deg": hadley_edges_deg(v),
        "mean_surface_speed_m_s": float(np.mean(np.sqrt(u**2 + v**2))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=Path, default=ROOT / "saves" / "test.npz")
    parser.add_argument(
        "--ncep-wind-raw",
        type=Path,
        default=ROOT / "testing" / "reference_data" / "ncep_ncar_raw",
    )
    parser.add_argument(
        "--ncep-slp",
        type=Path,
        default=ROOT / "testing" / "reference_data" / "ncep_ncar_slp_1991_2020.npz",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    state = _load_state(args.state)
    monthly_temp = state["monthly_temp"].astype(np.float64)
    elevation_m = np.clip(state["elevation"].astype(np.float64), 0.0, 1.0) * (
        float(EARTH.max_elevation_km) * 1000.0
    )
    sea, land = get_masks(state["elevation"], use_cache=False)
    h, w = elevation_m.shape
    surface_kind = np.where(np.asarray(land) > 0.5, 1, 0).astype(np.int64)
    if "land_ice_thickness" in state:
        surface_kind = np.where(
            np.asarray(state["land_ice_thickness"]) > 0.0, 2, surface_kind
        ).astype(np.int64)
    z0 = np.where(surface_kind == 0, 0.0, _ROUGHNESS_LAND_M)
    zoro = sw.oro_roughness_m(_sigma_oro_estimate(elevation_m))
    t2m_inst = state["air_temperature"].astype(np.float64)
    skin_inst = state["temperature"].astype(np.float64)
    ta_minus_skin = t2m_inst - skin_inst
    qa = np.clip(state["humidity"].astype(np.float64), 0.0, 1.0)
    lat_deg = _lat_centers(h)

    scenarios: dict[str, np.ndarray] = {
        "DJF": _seasonal_mean(monthly_temp, SEASONS["DJF"]),
        "JJA": _seasonal_mean(monthly_temp, SEASONS["JJA"]),
        "instantaneous": skin_inst,
    }

    report: dict = {
        "state": str(args.state),
        "grid": [h, w],
        "placeholders": {
            "tropopause_m": _TROPOPAUSE_M,
            "t2m": "skin_scenario + (Ta_inst - skin_inst); exact for the instantaneous scenario",
            "humidity": "saved instantaneous humidity for all scenarios",
            "roughness_land_m": _ROUGHNESS_LAND_M,
            "sigma_oro": "~2.5deg block std of elevation",
            "scale_height_reference_k": _SCALE_HEIGHT_REF_K,
        },
        "scenarios": {},
        "head_to_head": {},
    }

    sesam_fields: dict[str, dict] = {}
    timings: dict[str, float] = {}
    for name, skin in scenarios.items():
        t2m = skin + ta_minus_skin
        t0 = time.perf_counter()
        slp, wind, wind_zonal, closure = _run_chain(skin, t2m, qa, elevation_m, surface_kind, z0, zoro)
        timings[name] = time.perf_counter() - t0
        sesam_fields[name] = {"slp": slp, "wind": wind, "wind_zonal": wind_zonal}
        speed = np.sqrt(wind.surface_u_m_s**2 + wind.surface_v_m_s**2)
        report["scenarios"][name] = {
            "closure": closure,
            "surface_speed_mean_m_s": float(np.mean(speed)),
            "surface_speed_percentiles_m_s": {
                str(p): float(np.percentile(speed, p)) for p in (50, 75, 90, 99)
            },
            "slp_thermal_azonal_range_hpa": [
                float(slp.thermal_azonal_slp_pa.min() / 100.0),
                float(slp.thermal_azonal_slp_pa.max() / 100.0),
            ],
            "slp_zonal_anomaly_range_hpa": [
                float(slp.zonal_slp_anomaly_pa.min() / 100.0),
                float(slp.zonal_slp_anomaly_pa.max() / 100.0),
            ],
            "sesam_scorecard": _scorecard_report(wind.surface_u_m_s, wind.surface_v_m_s),
        }
    report["chain_runtime_s_at_state_grid"] = {k: round(v, 2) for k, v in timings.items()}

    # NCEP wind components (raw NetCDF, same grid handling as the builder).
    ncep_u_path = args.ncep_wind_raw / "uwnd.sig995.mon.ltm.1991-2020.nc"
    ncep_v_path = args.ncep_wind_raw / "vwnd.sig995.mon.ltm.1991-2020.nc"
    ncep: dict[str, dict[str, np.ndarray]] | None = None
    if ncep_u_path.exists() and ncep_v_path.exists():
        u_all = _read_wind_component(ncep_u_path, "uwnd")
        v_all = _read_wind_component(ncep_v_path, "vwnd")
        ncep = {}
        for season, months in SEASONS.items():
            ncep[season] = {
                "u": _seasonal_mean(u_all, months),
                "v": _seasonal_mean(v_all, months),
            }
        ref_h, ref_w = ncep["DJF"]["u"].shape
        for season in SEASONS:
            nu, nv = ncep[season]["u"], ncep[season]["v"]
            ncep_speed = np.sqrt(nu**2 + nv**2)
            wind = sesam_fields[season]["wind"]
            sesam_speed = np.sqrt(wind.surface_u_m_s**2 + wind.surface_v_m_s**2)
            sesam_on_ref = _downsample_to_reference(sesam_speed, ref_h, ref_w)
            ref_lat = _lat_centers(ref_h)
            saved_speed = np.sqrt(
                state["wind_u"].astype(np.float64) ** 2 + state["wind_v"].astype(np.float64) ** 2
            )
            saved_on_ref = _downsample_to_reference(saved_speed, ref_h, ref_w)
            wind_zonal = sesam_fields[season]["wind_zonal"]
            zonal_speed = np.sqrt(wind_zonal.surface_u_m_s**2 + wind_zonal.surface_v_m_s**2)
            zonal_on_ref = _downsample_to_reference(zonal_speed, ref_h, ref_w)
            report["head_to_head"][season] = {
                "ncep_scorecard": _scorecard_report(nu, nv),
                "sesam_scorecard": report["scenarios"][season]["sesam_scorecard"],
                "sesam_vs_ncep_speed": {
                    "pattern_correlation": _area_weighted_correlation(sesam_on_ref, ncep_speed, ref_lat),
                    "rmse_m_s": float(np.sqrt(np.mean((sesam_on_ref - ncep_speed) ** 2))),
                },
                "sesam_zonal_only_vs_ncep_speed": {
                    "pattern_correlation": _area_weighted_correlation(zonal_on_ref, ncep_speed, ref_lat),
                    "rmse_m_s": float(np.sqrt(np.mean((zonal_on_ref - ncep_speed) ** 2))),
                },
                "saved_generator_scorecard": _scorecard_report(
                    state["wind_u"].astype(np.float64), state["wind_v"].astype(np.float64)
                ),
                "saved_generator_vs_ncep_speed": {
                    "pattern_correlation": _area_weighted_correlation(saved_on_ref, ncep_speed, ref_lat),
                    "rmse_m_s": float(np.sqrt(np.mean((saved_on_ref - ncep_speed) ** 2))),
                    "note": "saved winds are instantaneous (day ~19, ~DJF), not a seasonal mean",
                },
            }
            # Zonal-mean zonal wind profile comparison (the jet structure).
            sesam_u_zonal = wind.surface_u_m_s.mean(axis=1)
            sesam_u_ref = _downsample_to_reference(wind.surface_u_m_s, ref_h, ref_w).mean(axis=1)
            ncep_u_zonal = nu.mean(axis=1)
            report["head_to_head"][season]["zonal_surface_u_correlation"] = _zonal_correlation(
                sesam_u_ref, ncep_u_zonal, ref_lat
            )
        # Instantaneous comparison vs NCEP DJF (save is day ~19).
        wind_inst = sesam_fields["instantaneous"]["wind"]
        inst_speed = np.sqrt(wind_inst.surface_u_m_s**2 + wind_inst.surface_v_m_s**2)
        ref_lat = _lat_centers(ref_h)
        ncep_djf_speed = np.sqrt(ncep["DJF"]["u"] ** 2 + ncep["DJF"]["v"] ** 2)
        report["head_to_head"]["instantaneous"] = {
            "sesam_vs_ncep_djf_speed": {
                "pattern_correlation": _area_weighted_correlation(
                    _downsample_to_reference(inst_speed, ref_h, ref_w), ncep_djf_speed, ref_lat
                ),
            },
            "sesam_vs_saved_speed_correlation": _area_weighted_correlation(
                inst_speed, saved_speed, lat_deg
            ),
        }
    else:
        report["head_to_head"] = {"skipped": "NCEP raw wind components not found"}

    # u500 zonal profile vs the saved upper-level wind (model-internal check).
    if "wind_u_aloft" in state:
        saved_aloft_zonal = state["wind_u_aloft"].astype(np.float64).mean(axis=1)
        for season in SEASONS:
            u500 = sesam_fields[season]["wind"].u500_pa_zonal_m_s
            report["scenarios"][season]["u500_zonal_profile_vs_saved_aloft"] = {
                "correlation": _zonal_correlation(u500, saved_aloft_zonal, lat_deg),
                "sesam_u500_at_45n_m_s": float(u500[np.argmin(np.abs(lat_deg - 45.0))]),
                "saved_aloft_at_45n_m_s": float(saved_aloft_zonal[np.argmin(np.abs(lat_deg - 45.0))]),
            }

    text = json.dumps(report, indent=1)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
