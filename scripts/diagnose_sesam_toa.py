"""SESAM P5 exit gate: TOA-flux measurement against the paper's Table 1.

Diagnostic-only companion to `sesam_radiation.py` (A6 clouds),
`sesam_shortwave.py` (A7), `sesam_longwave.py` (A8) and `sesam_tropopause.py`
(A10) -- docs/SESAM_GAP_ANALYSIS.md section 7, stage P5's third and final
open item ("the TOA-flux exit-gate measurement against the paper's
validation figures"). Runs the full assembled radiation chain on a real
saved state and compares global annual-mean TOA solar-down/up/net and
outgoing-longwave against Table 1 of Willeit et al. (2022) -- transcribed as
``sesam_reference.TABLE_MAIN_ENERGY_BUDGET`` (both the paper's own
CLIMBER-X column and the independent Wild et al. 2013 observational mean).

**Deliberately narrower scope than a full P2/P3 dynamics run** (an honest
simplification, documented, not a hidden shortcut): computing SESAM's own
(A61)-(A68) cloud fraction/geometry needs ``wsyn``/``Us`` from the P2/P3
wind+EKE chain, which would require regridding that chain onto the extended
stratosphere-reaching level grid this script needs for a genuine top-of-
atmosphere OLR (P1-P4's diagnostic scripts all use a troposphere-only grid
stopping at the tropopause, since none of them needed the stratosphere).
Rather than force that regrid, this script uses the saved state's own
``cloud_cover`` field (real, already-produced-by-the-supported-model data,
not fabricated) for cloud *fraction*, while still exercising SESAM's own
(A67)/(A68) formulas for cloud *top height* and *optical thickness* (both
need only tropopause height, T2m and column water -- no wind chain). This
validates the radiative-transfer machinery this stage was actually built to
close (A69-A105, A106-A117, A10) without re-deriving P2/P3's already-
measured (and separately gated) circulation.

**Placeholders, documented not fabricated** (same discipline as
`diagnose_sesam_thermo.py`'s evaporation proxy):
- Surface albedo: flat per-surface-type constants (ocean 0.08, land 0.20,
  ice/thick-snow 0.6) -- the saved state has no per-cell albedo field.
  Same value used for both shortwave bands (sesam_shortwave.py's own
  documented simplification for the common case with no per-band data).
- Aerosol optical thickness/imaginary refractive index: zero (clean-
  atmosphere placeholder, the same convention `shortwave_radiation`'s own
  docstring establishes for a caller with no aerosol field).
- ``w700_mean`` (mean-cell 700 hPa vertical velocity, feeds A67 cloud-top
  height and A63 weff): zero, matching stage P1's own documented placeholder
  for the same symbol.
- Ozone: `sesam_tropopause.standard_ozone_mixing_ratio_profile_kgkg`'s
  constant-global climatology (see that module's docstring for the "constant
  vs zonal vs real-dataset" decision this represents).
- Cloud base height: ``zs + H_pbl`` (the A67 docstring's own stated
  assumption that cloud base coincides with the PBL top).
- Cosine solar zenith: the local-noon value
  ``sin(lat)sin(delta)+cos(lat)cos(delta)`` (a standard daily-mean-EMIC
  representative sun angle for the optical-path terms), paired with
  ``EARTH.daily_mean_insolation`` (already day/night-averaged) for the flux
  magnitude -- the same two-part decomposition CLIMBER-X's own daily
  timestep design uses (docs/SESAM_GAP_ANALYSIS.md's paper-Conclusions
  citation for the "daily time step" finding in `sesam_tropopause.py`).

The tropopause closure (A10) is measured but not iterated to equilibrium: a
single `sesam_tropopause.advance_tropopause_height` step from the initial
12 km guess is reported (tendency sign/magnitude), the same "local closure,
not a converged prognostic loop" scope P3's own exit gate already used for
EKE.

Nothing here touches the supported climate path; the script only reads the
save and writes a JSON report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import sesam_longwave as slw  # noqa: E402
import sesam_radiation as sr  # noqa: E402
import sesam_reference as sref  # noqa: E402
import sesam_shortwave as ssw  # noqa: E402
import sesam_tropopause as stp  # noqa: E402
import sesam_vertical as sv  # noqa: E402
from carbon_cycle import CO2_CURRENT  # noqa: E402
from masks import get_masks  # noqa: E402
from planet_params import EARTH  # noqa: E402
from diagnose_sesam_slp import _lat_centers  # noqa: E402
from sim_grid import _coarsen as _block_mean_coarsen  # noqa: E402

_LEVELS_M = np.array(
    [0.0, 1500.0, 3000.0, 5500.0, 9000.0, 12000.0, 16000.0, 20000.0, 25000.0, 30000.0]
)
_TROPOPAUSE_M = 12000.0
_P0_PA = float(EARTH.surface_pressure_pa)
_H_PBL_M = 1500.0  # sesam_reference TABLE_A6_CLOUDS entry, A67's cloud-base assumption
_MID_MONTH_DAYS = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]


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


def _surface_albedo(surface_kind: np.ndarray, ice_cover: np.ndarray, snow_depth: np.ndarray) -> np.ndarray:
    alb = np.where(surface_kind == 0, 0.08, np.where(surface_kind == 2, 0.6, 0.20))
    bright = (np.asarray(ice_cover) > 0.5) | (np.asarray(snow_depth) > 0.1)
    return np.where(bright, np.maximum(alb, 0.6), alb)


def _annual_mean_insolation(lat_rad: np.ndarray) -> np.ndarray:
    q = np.zeros_like(lat_rad, dtype=np.float64)
    for day in _MID_MONTH_DAYS:
        q += EARTH.daily_mean_insolation(lat_rad, float(day))
    return q / len(_MID_MONTH_DAYS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=Path, default=ROOT / "saves" / "test.npz")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--block-size", type=int, default=1,
        help="Block-average downsample factor applied before running the chain "
             "(1 = native resolution, the default).",
    )
    args = parser.parse_args()

    state = _load_state(args.state)
    state = _coarsen_state(state, args.block_size)

    elevation_m = np.clip(state["elevation"].astype(np.float64), 0.0, 1.0) * (
        float(EARTH.max_elevation_km) * 1000.0
    )
    _sea, land = get_masks(state["elevation"], use_cache=False)
    land_mask = (np.asarray(land) > 0.5).astype(np.float64)
    surface_kind = np.where(land_mask > 0.5, 1, 0).astype(np.int64)
    if "land_ice_thickness" in state:
        surface_kind = np.where(np.asarray(state["land_ice_thickness"]) > 0.0, 2, surface_kind).astype(np.int64)

    h, w = elevation_m.shape
    lat_deg = _lat_centers(h)
    lat_rad = np.deg2rad(lat_deg)[:, None] * np.ones((1, w))

    skin = state["monthly_temp"].astype(np.float64).mean(axis=0)
    t2m_inst = state["air_temperature"].astype(np.float64)
    skin_inst = state["temperature"].astype(np.float64)
    t2m = skin + (t2m_inst - skin_inst)
    qa = np.clip(state["humidity"].astype(np.float64), 0.0, 1.0)
    cloud_fraction = np.clip(state["cloud_cover"].astype(np.float64), 0.0, 1.0)

    print(f"[toa] vertical structure on {h}x{w}, {_LEVELS_M.size} levels to {_LEVELS_M[-1]/1000:.0f} km", flush=True)
    structure = sv.compute_vertical_structure(
        _LEVELS_M,
        near_surface_air_temp_k=t2m, skin_temp_k=skin, surface_kind=surface_kind,
        near_surface_specific_humidity_kgkg=qa, surface_elevation_m=elevation_m,
        tropopause_height_m=np.full((h, w), _TROPOPAUSE_M),
        p0_pa=_P0_PA, gravity=float(EARTH.surface_gravity), reference_temp_k=288.0,
    )
    surface_pressure_pa = _P0_PA * np.exp(-elevation_m / sv.height_scale(288.0, gravity=float(EARTH.surface_gravity)))

    ftrop = sv.tropical_weight(np.abs(lat_rad))
    hr_m = sv.rh_scale_height(ftrop, 0.0)  # w700_mean placeholder = 0, module docstring
    column_water_kg_m2 = qa * 2000.0  # atmosphere.py's own water_scale convention

    # --- (A6) cloud geometry (fraction reused from the save; see module docstring) ---
    cloud_top_m = sr.cloud_top_height_m(np.full((h, w), _TROPOPAUSE_M), np.zeros((h, w)))
    cloud_base_m = elevation_m + _H_PBL_M
    cloud_top_m = np.maximum(cloud_top_m, cloud_base_m + 500.0)
    cloud_optical_thickness = sr.cloud_optical_thickness(t2m, cloud_fraction, column_water_kg_m2)

    surface_albedo = _surface_albedo(surface_kind, state.get("ice_cover", np.zeros((h, w))), state.get("snow_depth", np.zeros((h, w))))

    delta_annual = np.mean([EARTH.solar_declination(float(d)) for d in _MID_MONTH_DAYS])
    cos_zenith_noon = np.clip(np.sin(lat_rad) * np.sin(delta_annual) + np.cos(lat_rad) * np.cos(delta_annual), 1e-3, 1.0)
    toa_insolation = _annual_mean_insolation(lat_rad)

    print("[toa] running (A69)-(A105) shortwave...", flush=True)
    sw = ssw.shortwave_radiation(
        incoming_toa_w_m2=toa_insolation, cos_zenith=cos_zenith_noon,
        cloud_fraction=cloud_fraction, cloud_top_height_m=cloud_top_m,
        cloud_optical_thickness=cloud_optical_thickness,
        cloud_geometric_thickness_m=cloud_top_m - cloud_base_m,
        column_water_kg_m2=column_water_kg_m2, humidity_scale_height_m=hr_m,
        surface_albedo_vu=surface_albedo, surface_albedo_ir=surface_albedo,
        aerosol_optical_thickness=np.zeros((h, w)), aerosol_imaginary_refractive_index=np.zeros((h, w)),
    )

    print("[toa] building ozone climatology...", flush=True)
    ozone_profile = stp.standard_ozone_mixing_ratio_profile_kgkg(_LEVELS_M, structure.air_density_kg_m3)

    print(f"[toa] running (A106)-(A117) longwave ({_LEVELS_M.size}x{_LEVELS_M.size} transmission matrix)...", flush=True)
    lw = slw.longwave_radiation(
        temperature_profile_k=structure.temperature_k,
        specific_humidity_profile_kg_kg=structure.specific_humidity_kgkg,
        ozone_mixing_ratio_profile_kg_kg=ozone_profile,
        pressure_profile_pa=structure.pressure_pa,
        air_density_profile_kg_m3=structure.air_density_kg_m3,
        levels_m=_LEVELS_M, surface_pressure_pa=surface_pressure_pa,
        surface_skin_temp_k=skin, co2_ppm=float(CO2_CURRENT), gravity=float(EARTH.surface_gravity),
        cloud_fraction=cloud_fraction, cloud_base_height_m=cloud_base_m,
        cloud_top_height_m=cloud_top_m, cloud_optical_thickness=cloud_optical_thickness,
    )

    print("[toa] closing (A10)/(A11) tropopause tendency...", flush=True)
    itcz_lat = 0.0
    hadley_width = np.deg2rad(30.0)
    trop = stp.advance_tropopause_height(
        np.full((h, w), _TROPOPAUSE_M), lat_rad, itcz_lat, hadley_width,
        lw.downward_w_m2, lw.upward_w_m2, _LEVELS_M, toa_insolation,
        dt_days=1.0, surface_elevation_m=elevation_m,
    )

    area = np.cos(lat_rad)
    area_sum = float(np.sum(area))

    def gmean(field: np.ndarray) -> float:
        return float(np.sum(field * area) / area_sum)

    toa_solar_down = gmean(toa_insolation)
    toa_solar_up = gmean(sw.toa_upward_w_m2)
    toa_solar_net = toa_solar_down - toa_solar_up
    toa_thermal_up = gmean(lw.outgoing_longwave_w_m2)

    budget = sref.table("main_energy_budget")["entries"]
    targets = {
        name: (sim, budget[name]["value"], budget[f"{name}_obs_mean"]["value"])
        for name, sim in (
            ("toa_solar_down", toa_solar_down),
            ("toa_solar_up", toa_solar_up),
            ("toa_solar_net", toa_solar_net),
            ("toa_thermal_up", toa_thermal_up),
        )
    }

    report: dict = {
        "state": str(args.state),
        "grid": [h, w],
        "block_size": args.block_size,
        "levels_m": _LEVELS_M.tolist(),
        "placeholders": {
            "cloud_fraction": "reused from the saved state's own cloud_cover field",
            "cloud_top_height_and_optical_thickness": "SESAM (A67)/(A68), w700_mean=0",
            "surface_albedo": "flat per-surface-type constants (ocean 0.08, land 0.20, ice 0.6)",
            "aerosol": "zero (clean-atmosphere placeholder)",
            "ozone": "sesam_tropopause constant-global climatology (300 DU, Gaussian layer)",
            "co2_ppm": float(CO2_CURRENT),
            "insolation": "annual mean of 12 mid-month EARTH.daily_mean_insolation calls",
        },
        "toa_flux_w_m2": {
            name: {
                "simulated": round(sim, 2),
                "climberx_paper_table1": climberx,
                "observed_mean_table1": obs,
                "diff_vs_climberx": round(sim - climberx, 2),
                "diff_vs_observed": round(sim - obs, 2),
                "pct_diff_vs_climberx": round(100.0 * (sim - climberx) / climberx, 1),
            }
            for name, (sim, climberx, obs) in targets.items()
        },
        "tropopause_closure": {
            "initial_height_m": _TROPOPAUSE_M,
            "r_strat_net_mean_w_m2": gmean(trop.r_strat_net_w_m2),
            "shape_s_mean_w_m2": gmean(trop.shape_s),
            "tendency_mean_m_per_day": gmean(trop.tendency_m_per_day),
            "updated_height_mean_m": gmean(trop.tropopause_height_m),
        },
        "diagnostics": {
            "surface_downward_sw_mean_w_m2": gmean(sw.surface_downward_w_m2),
            "cloud_fraction_mean": gmean(cloud_fraction),
            "cloud_top_height_mean_m": gmean(cloud_top_m),
            "column_water_mean_kg_m2": gmean(column_water_kg_m2),
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
