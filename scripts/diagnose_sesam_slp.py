"""Diagnose the SESAM stage-P2 SLP reconstruction on a saved PlanetSim state.

Diagnostic-only companion to `sesam_dynamics.py` (docs/SESAM_GAP_ANALYSIS.md
§7, stage P2 first sub-deliverable). It loads a saved state NPZ, builds the
DJF/JJA skin-temperature and elevation fields, runs the Appendix-A2 SLP
reconstruction (zonal cell-physics SLP + azonal thermal SLP + the
Charney–Eliassen topographic term), reports the cell geometry and zonal
extrema, and scores the reconstructed DJF/JJA SLP against the NCEP/NCAR
Reanalysis 1 SLP climatology when that reference has been built
(`scripts/build_ncep_slp_reference.py`).

Placeholder policy (documented in sesam_dynamics.py's module docstring):

- ``sin_cos_alpha_bar``: uniform magnitude |sin α cos α| = sin(30°)·cos(30°)
  signed by hemisphere, until the P2 wind assembly solves (A21). The zonal
  SLP *pattern* is insensitive to this; its amplitude scales with 1/|·|.
- ``u500``: the zonal mean of the saved state's upper-level wind
  (`wind_u_aloft`), until the P2 wind assembly derives the 500 hPa wind from
  the thermal-wind relation (A17).
- ``H_T``: uniform 12 km tropopause, until P1/P5 supply the closed field.

Nothing here touches the supported climate path; the script only reads the
save and writes a JSON report.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sesam_dynamics as sd  # noqa: E402
from planet_params import EARTH  # noqa: E402

SEASONS = {"DJF": (11, 0, 1), "JJA": (5, 6, 7)}
_REFERENCE_TEMP_K = 273.15
_SIN_COS_ALPHA_PLACEHOLDER = float(np.sin(np.radians(30.0)) * np.cos(np.radians(30.0)))
_TROPOPAUSE_PLACEHOLDER_M = 12000.0


def _load_state(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files if not name.startswith("__")}


def _seasonal_mean(monthly: np.ndarray, months: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(monthly[list(months)], dtype=np.float64).mean(axis=0)


def _lat_centers(h: int) -> np.ndarray:
    return 90.0 - (np.arange(h) + 0.5) * 180.0 / h


def _lon_centers(w: int) -> np.ndarray:
    return -180.0 + (np.arange(w) + 0.5) * 360.0 / w


def _downsample_to_reference(
    field: np.ndarray, ref_h: int, ref_w: int
) -> np.ndarray:
    """Area-aware separable box-average of a model field to the reference grid.

    Model and reference grids are both regular north-to-south, [-180, 180)
    cell-centred grids. Rows are cosine-weighted; columns uniform.
    """
    h, w = field.shape
    model_lat = np.radians(_lat_centers(h))
    ref_lat = np.radians(_lat_centers(ref_h))
    model_lon = _lon_centers(w)
    ref_lon = _lon_centers(ref_w)

    # Row mapping with cosine weights.
    row_edges = np.zeros((ref_h, 2))
    centers = ref_lat
    dlat = np.pi / ref_h
    row_edges[:, 0] = centers + 0.5 * dlat  # northern edge
    row_edges[:, 1] = centers - 0.5 * dlat  # southern edge
    b = np.zeros((ref_h, h))
    for j in range(ref_h):
        mask = (model_lat <= row_edges[j, 0]) & (model_lat > row_edges[j, 1])
        if not np.any(mask):
            # No model row centre in this box: take the nearest row.
            mask = np.zeros(h, dtype=bool)
            mask[np.argmin(np.abs(model_lat - centers[j]))] = True
        weights = np.cos(model_lat) * mask
        b[j] = weights / np.sum(weights)

    dlon = 360.0 / ref_w
    a = np.zeros((ref_w, w))
    for k in range(ref_w):
        west = ref_lon[k] - 0.5 * dlon
        east = ref_lon[k] + 0.5 * dlon
        mask = (model_lon >= west) & (model_lon < east)
        if not np.any(mask):
            mask = np.zeros(w, dtype=bool)
            mask[np.argmin(np.abs(model_lon - ref_lon[k]))] = True
        a[k] = mask / np.sum(mask)
    return b @ field @ a.T


def _area_weighted_correlation(x: np.ndarray, y: np.ndarray, lat_deg: np.ndarray) -> float:
    w = np.cos(np.radians(lat_deg))[:, None] * np.ones_like(x)
    wsum = np.sum(w)
    xm = np.sum(x * w) / wsum
    ym = np.sum(y * w) / wsum
    xa = x - xm
    ya = y - ym
    denom = np.sqrt(np.sum(xa**2 * w) * np.sum(ya**2 * w))
    return float(np.sum(xa * ya * w) / denom) if denom > 0 else float("nan")


def _zonal_correlation(x: np.ndarray, y: np.ndarray, lat_deg: np.ndarray) -> float:
    """Cosine-weighted correlation of two (H,) zonal-mean profiles."""
    w = np.cos(np.radians(lat_deg))
    xm = np.sum(x * w) / np.sum(w)
    ym = np.sum(y * w) / np.sum(w)
    xa = x - xm
    ya = y - ym
    denom = np.sqrt(np.sum(xa**2 * w) * np.sum(ya**2 * w))
    return float(np.sum(xa * ya * w) / denom) if denom > 0 else float("nan")


def _season_report(result: "sd.SesamSlp", lat_deg: np.ndarray) -> dict:
    lat = np.radians(lat_deg)
    extrema = sd.zonal_slp_extrema(lat, result.zonal_slp_anomaly_pa, result.itcz_latitude_rad)

    def va_at(deg: float) -> float:
        return float(result.overturning_wind_m_s[np.argmin(np.abs(lat_deg - deg))])

    w2d = np.cos(lat)[:, None]
    global_mean = float(
        np.sum(result.slp_pa * w2d) / np.sum(w2d * np.ones_like(result.slp_pa))
    )
    return {
        "itcz_latitude_deg": float(np.degrees(result.itcz_latitude_rad)),
        "hadley_edge_nh_deg": float(np.degrees(result.hadley_edge_nh_rad)),
        "hadley_edge_sh_deg": float(np.degrees(result.hadley_edge_sh_rad)),
        "hadley_width_deg": float(np.degrees(result.hadley_width_rad)),
        "hadley_width_scale": float(result.hadley_width_scale),
        "cell_gradients_nh_k": [float(v) for v in result.cell_gradients_nh_k],
        "cell_gradients_sh_k": [float(v) for v in result.cell_gradients_sh_k],
        "t_nh_k": result.t_nh_k,
        "t_sh_k": result.t_sh_k,
        "t_trp_k": result.t_trp_k,
        "overturning_wind_m_s": {
            "+15": va_at(15.0), "+45": va_at(45.0), "+75": va_at(75.0),
            "-15": va_at(-15.0), "-45": va_at(-45.0), "-75": va_at(-75.0),
        },
        "zonal_slp_extrema": extrema,
        "zonal_anomaly_range_hpa": [
            float(np.min(result.zonal_slp_anomaly_pa) / 100.0),
            float(np.max(result.zonal_slp_anomaly_pa) / 100.0),
        ],
        "thermal_azonal_range_hpa": [
            float(np.min(result.thermal_azonal_slp_pa) / 100.0),
            float(np.max(result.thermal_azonal_slp_pa) / 100.0),
        ],
        "orographic_azonal_range_hpa": [
            float(np.min(result.orographic_azonal_slp_pa) / 100.0),
            float(np.max(result.orographic_azonal_slp_pa) / 100.0),
        ],
        "global_mean_slp_pa": global_mean,
        "mass_residual_pa": global_mean - result.p0_pa,
    }


def _score_against_reference(
    model_slp: np.ndarray, ref_slp: np.ndarray, lat_deg: np.ndarray
) -> dict:
    ref_h, ref_w = ref_slp.shape
    ref_lat = _lat_centers(ref_h)
    model_on_ref = _downsample_to_reference(model_slp, ref_h, ref_w)
    w = np.cos(np.radians(ref_lat))[:, None]
    wsum = np.sum(w * np.ones_like(ref_slp))

    def stats(a: np.ndarray, b: np.ndarray) -> dict:
        a_mean = np.sum(a * w) / wsum
        b_mean = np.sum(b * w) / wsum
        aa = a - a_mean
        bb = b - b_mean
        return {
            "pattern_correlation": _area_weighted_correlation(a, b, ref_lat),
            "rmse_hpa": float(np.sqrt(np.sum((aa - bb) ** 2 * w) / wsum) / 100.0),
        }

    full = stats(model_on_ref, ref_slp)
    zonal_model = model_on_ref.mean(axis=1)
    zonal_ref = ref_slp.mean(axis=1)
    zonal_corr = _zonal_correlation(zonal_model, zonal_ref, ref_lat)
    cos_lat = np.cos(np.radians(ref_lat))
    zonal_ref_anomaly = zonal_ref - np.sum(zonal_ref * cos_lat) / np.sum(cos_lat)
    zonal_model_anomaly = zonal_model - np.sum(zonal_model * cos_lat) / np.sum(cos_lat)
    ref_extrema = sd.zonal_slp_extrema(np.radians(ref_lat), zonal_ref_anomaly, 0.0)
    model_extrema = sd.zonal_slp_extrema(np.radians(ref_lat), zonal_model_anomaly, 0.0)
    return {
        "reference_grid": [ref_h, ref_w],
        "full_field": full,
        "zonal_mean_profile_correlation": zonal_corr,
        "ncep_zonal_extrema": ref_extrema,
        "model_zonal_extrema_on_reference_grid": model_extrema,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=Path, default=ROOT / "saves" / "test.npz")
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "testing" / "reference_data" / "ncep_ncar_slp_1991_2020.npz",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    state = _load_state(args.state)
    monthly_temp = state["monthly_temp"].astype(np.float64)
    elevation = state["elevation"].astype(np.float64)
    h, w = elevation.shape
    lat_deg = _lat_centers(h)
    u500 = (
        np.asarray(state["wind_u_aloft"], dtype=np.float64).mean(axis=1)
        if "wind_u_aloft" in state
        else None
    )

    report: dict = {
        "state": str(args.state),
        "grid": [h, w],
        "placeholders": {
            "sin_cos_alpha_bar_abs": _SIN_COS_ALPHA_PLACEHOLDER,
            "u500": "zonal mean of saved wind_u_aloft" if u500 is not None else "unavailable (Charney–Eliassen term skipped)",
            "tropopause_m": _TROPOPAUSE_PLACEHOLDER_M,
        },
        "seasons": {},
        "ncep_score": {},
    }

    timings = []
    results: dict[str, sd.SesamSlp] = {}
    for season, months in SEASONS.items():
        skin = _seasonal_mean(monthly_temp, months)
        t0 = time.perf_counter()
        result = sd.compute_slp(
            skin_temp_k=skin,
            surface_elevation_m=elevation,
            sin_cos_alpha_bar=_SIN_COS_ALPHA_PLACEHOLDER,
            gravity=float(EARTH.surface_gravity),
            radius_m=float(EARTH.radius_m),
            omega=float(EARTH.omega),
            p0_pa=float(EARTH.surface_pressure_pa),
            reference_temp_k=_REFERENCE_TEMP_K,
            u500_m_s=u500,
            tropopause_height_m=(
                np.full(h, _TROPOPAUSE_PLACEHOLDER_M) if u500 is not None else None
            ),
        )
        timings.append(time.perf_counter() - t0)
        results[season] = result
        report["seasons"][season] = _season_report(result, lat_deg)

    report["kernel_runtime_ms_per_call_at_state_grid"] = [
        round(1000.0 * t, 2) for t in timings
    ]

    if args.reference.exists():
        with np.load(args.reference, allow_pickle=False) as archive:
            ref_monthly = archive["slp_pa"].astype(np.float64)
        for season, months in SEASONS.items():
            ref_season = _seasonal_mean(ref_monthly, months)
            report["ncep_score"][season] = _score_against_reference(
                results[season].slp_pa, ref_season, lat_deg
            )
    else:
        report["ncep_score"] = {
            "skipped": f"reference not built: run scripts/build_ncep_slp_reference.py ({args.reference})"
        }

    text = json.dumps(report, indent=1)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
