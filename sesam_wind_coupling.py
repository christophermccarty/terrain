"""SESAM stage P6c: live coupling of the P2 (SLP/wind) and P3 (EKE) closures
into the prognostic simulation loop, replacing P6b's own documented legacy-
wind and placeholder-EKE bridges (`sesam_coupling.py` bridges 1-2).

docs/SESAM_GAP_ANALYSIS.md Sec7 P6. Gated by
``PlanetParams.enable_sesam_dynamics`` (default False, the same gate P2/P3's
own diagnostic-only kernels have used since they were built) *in addition to*
``enable_sesam_column_closure`` -- both must be on for this module to run;
column closure alone still uses P6b's bridges unchanged.

This mirrors the exact, already-validated chain
``scripts/diagnose_sesam_wind.py``/``scripts/diagnose_sesam_synoptic.py``
already run against saved states, adapted to read from live evolving
`PlanetState` fields instead of a saved ``.npz``:

1. P1 vertical structure (`sesam_vertical.compute_vertical_structure`).
2. P2 two-pass SLP<->wind closure (`sesam_dynamics.compute_slp` /
   `sesam_wind.compute_wind`), then the **zonal-only** wind extraction
   (dropping the azonal thermal/Charney-Eliassen terms) -- per
   docs/SESAM_GAP_ANALYSIS.md's own P2 exit-gate verdict: the full azonal
   chain amplifies this project's own sharp regional fields into
   100+ m/s local winds, while the zonal-only chain is sane and already
   beats the legacy prescribed-cell generator on pattern correlation.
3. P3 local-steady-state EKE (`sesam_synoptic.compute_synoptic`), driven by
   the zonal-only wind -- per P3's own documented finding that the full
   azonal-inflated wind would compound the same artefact into EKE.
   `compute_synoptic` also returns (A58)'s `total_wind_m_s` (zonal wind
   combined with the synoptic gustiness term `Usyn`), exposed here as
   `SesamWindAndEke.total_wind_m_s` for callers that need a bulk-flux wind
   speed rather than the mean advecting wind (docs/SESAM_GAP_ANALYSIS.md
   Sec7 P2, 2026-08-20 follow-up).

**Scope decision, not yet the full P3 prognostic K transport.** P3's own
(A52) advection+diffusion transport of K needs a persistent per-cell state
field and a short (~0.25-day) coupling cadence to stay numerically sane
(docs/SESAM_GAP_ANALYSIS.md Sec7 P3, 2026-08-18 entries) -- real, validated
machinery, but a second sub-stage of scope P3 itself delivered separately
from its own first (local-steady-state) sub-deliverable. This module mirrors
that same staging for its live-coupling debut: local-steady-state EKE first,
prognostic K transport as a documented follow-up, not attempted here.

**Recomputed once per outer `simulate_step` call, not per SESAM column-
closure substep.** SLP/wind/EKE are driven by the (slowly varying) skin and
air temperature fields, which this module treats as fixed for the whole
outer step -- the same simplification the legacy
`_generate_precipitation_substepped` already makes for its own per-day
inner loop (temperature held fixed across precipitation substeps).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

import sesam_dynamics as sd
import sesam_synoptic as ss
import sesam_vertical as sv
import sesam_wind as sw
from sesam_vertical import saturation_specific_humidity

# P1's own documented placeholder tropopause height (P5's real (A10) closure
# is not wired into the live loop yet) -- the same value every
# scripts/diagnose_sesam_*.py driver uses.
_LEVELS_M = np.array([0.0, 1500.0, 3000.0, 5500.0, 9000.0, 12000.0])
_TROPOPAUSE_HEIGHT_M = 12000.0
_SCALE_HEIGHT_REFERENCE_TEMP_K = 288.0
_SLP_REFERENCE_TEMP_K = 273.15
_ROUGHNESS_LAND_M = 0.1
_OROGRAPHY_BLOCK_CELLS = 8  # ~2.5 deg at this project's headline 512-row grid


def _sigma_oro_estimate(elevation_m: np.ndarray, block: int = _OROGRAPHY_BLOCK_CELLS) -> np.ndarray:
    """Sub-grid orography standard deviation ((A23) input), block-estimated.

    Identical method to ``scripts/diagnose_sesam_wind.py``'s own
    ``_sigma_oro_estimate`` (nearest-nnpsample nearest-neighbour blowup of a
    block standard deviation) -- copied rather than imported since it is a
    small, self-contained numpy routine and the scripts/ package is not a
    dependency of the live simulation path.
    """
    h, w = elevation_m.shape
    padded_h = (h // block) * block
    padded_w = (w // block) * block
    if padded_h == 0 or padded_w == 0:
        return np.zeros_like(elevation_m)
    z = elevation_m[:padded_h, :padded_w]
    blocks = z.reshape(padded_h // block, block, padded_w // block, block)
    std = blocks.std(axis=(1, 3))
    out = np.repeat(np.repeat(std, block, axis=0), block, axis=1)
    full = np.zeros_like(elevation_m)
    full[:padded_h, :padded_w] = out
    full[:, padded_w:] = full[:, :1]
    full[padded_h:, :] = full[:1, :]
    return full


class SesamWindAndEke(NamedTuple):
    wind_u_m_s: np.ndarray
    wind_v_m_s: np.ndarray
    eke_m2_s2: np.ndarray
    total_wind_m_s: np.ndarray
    diagnostics: dict


def sesam_wind_and_eke_step(
    *,
    air_temperature_k: np.ndarray,
    skin_temperature_k: np.ndarray,
    relative_humidity: np.ndarray,
    elevation_m: np.ndarray,
    land_mask: np.ndarray,
    ice_mask: np.ndarray | None,
    surface_pressure_pa: float,
    radius_m: float,
    gravity: float,
    omega: float,
) -> SesamWindAndEke:
    """One P2 (zonal-only SLP/wind) + P3 (local-steady-state EKE) evaluation.

    Mirrors ``scripts/diagnose_sesam_wind.py``'s ``_run_chain`` (the
    two-pass SLP<->wind closure and zonal-only extraction) and
    ``scripts/diagnose_sesam_synoptic.py``'s ``compute_synoptic`` call
    exactly, sourcing fields from live state instead of a saved ``.npz``.
    """
    h, w = np.asarray(skin_temperature_k).shape
    skin = np.asarray(skin_temperature_k, dtype=np.float64)
    ta = np.asarray(air_temperature_k, dtype=np.float64)
    ra = np.clip(np.asarray(relative_humidity, dtype=np.float64), 0.0, 1.0)
    elev = np.asarray(elevation_m, dtype=np.float64)
    land = np.asarray(land_mask, dtype=bool)
    p0 = float(surface_pressure_pa)
    p0_field = np.full((h, w), p0)
    qa = ra * saturation_specific_humidity(ta, p0_field)

    surface_kind = np.where(land, 1, 0).astype(np.int64)
    if ice_mask is not None:
        surface_kind = np.where(np.asarray(ice_mask, dtype=bool), 2, surface_kind)

    z0 = np.where(surface_kind == 0, 0.0, _ROUGHNESS_LAND_M)
    zoro = sw.oro_roughness_m(_sigma_oro_estimate(elev))

    structure = sv.compute_vertical_structure(
        _LEVELS_M,
        near_surface_air_temp_k=ta,
        skin_temp_k=skin,
        surface_kind=surface_kind,
        near_surface_specific_humidity_kgkg=qa,
        surface_elevation_m=elev,
        tropopause_height_m=np.full((h, w), _TROPOPAUSE_HEIGHT_M),
        p0_pa=p0,
        gravity=gravity,
        reference_temp_k=_SCALE_HEIGHT_REFERENCE_TEMP_K,
    )
    cd = sw.drag_coefficient(z0, zoro, surface_kind)
    lat = sw._latitude_rad(h)
    angle = sw.cross_isobar_angle(cd, lat, omega=omega)
    scab_zonal = angle["sin_cos_alpha"].mean(axis=1)

    def slp_with(u500: np.ndarray | None) -> sd.SesamSlp:
        return sd.compute_slp(
            skin_temp_k=skin,
            surface_elevation_m=elev,
            sin_cos_alpha_bar=scab_zonal,
            gravity=gravity,
            radius_m=radius_m,
            omega=omega,
            p0_pa=p0,
            reference_temp_k=_SLP_REFERENCE_TEMP_K,
            u500_m_s=u500,
            tropopause_height_m=np.full(h, _TROPOPAUSE_HEIGHT_M) if u500 is not None else None,
        )

    sigma_trop = float(structure.pressure_pa[-1, 0, 0] / structure.pressure_pa[0, 0, 0])

    def wind_for(slp: np.ndarray) -> sw.SesamWind:
        return sw.compute_wind(
            slp_pa=slp,
            temperature_z=structure.temperature_k,
            levels_m=_LEVELS_M,
            pressure_z=structure.pressure_pa,
            skin_temp_k=skin,
            t2m_k=ta,
            surface_elevation_m=elev,
            surface_kind=surface_kind,
            roughness_m=z0,
            sigma_oro_m=zoro / sw._ORO_ROUGHNESS_FACTOR,
            tropopause_sigma=sigma_trop,
            gravity=gravity,
            radius_m=radius_m,
            omega=omega,
            rho0_kg_m3=p0 / (287.0 * _SLP_REFERENCE_TEMP_K),
            reference_temp_k=_SLP_REFERENCE_TEMP_K,
        )

    slp1 = slp_with(None)
    wind1 = wind_for(slp1.slp_pa)
    slp2 = slp_with(wind1.u500_pa_zonal_m_s)
    # Zonal-only decomposition (docs/SESAM_GAP_ANALYSIS.md Sec7 P2 exit-gate
    # verdict): drop the azonal thermal/Charney-Eliassen terms, keep only the
    # mean-cell-physics zonal SLP anomaly, before computing the final wind.
    slp_zonal_only = slp2.p0_pa + slp2.zonal_slp_anomaly_pa[:, None] * np.ones((h, w))
    wind_zonal = wind_for(slp_zonal_only)

    syn = ss.compute_synoptic(
        potential_temperature_k=structure.potential_temperature_k,
        u_wind_z=wind_zonal.u_z_m_s,
        v_wind_z=wind_zonal.v_z_m_s,
        pressure_z=structure.pressure_pa,
        levels_m=_LEVELS_M,
        surface_u_m_s=wind_zonal.surface_u_m_s,
        surface_v_m_s=wind_zonal.surface_v_m_s,
        surface_elevation_m=elev,
        surface_kind=surface_kind,
        drag_coefficient=cd,
        epsilon=angle["epsilon"],
        cos_alpha=angle["cos_alpha"],
        gravity=gravity,
        omega=omega,
        rho0_kg_m3=p0 / (287.0 * _SLP_REFERENCE_TEMP_K),
    )

    diagnostics = {
        "mean_surface_speed_m_s": float(np.mean(np.sqrt(
            wind_zonal.surface_u_m_s ** 2 + wind_zonal.surface_v_m_s ** 2
        ))),
        "eke_mean_m2_s2": float(np.mean(syn.eddy_kinetic_energy_m2_s2)),
        "storm_track_latitude_deg": float(
            (90.0 - (np.arange(h) + 0.5) * 180.0 / h)[
                int(np.argmax(syn.eddy_kinetic_energy_m2_s2.mean(axis=1)))
            ]
        ),
    }

    return SesamWindAndEke(
        wind_u_m_s=wind_zonal.surface_u_m_s.astype(np.float32),
        wind_v_m_s=wind_zonal.surface_v_m_s.astype(np.float32),
        eke_m2_s2=syn.eddy_kinetic_energy_m2_s2.astype(np.float32),
        total_wind_m_s=syn.total_wind_m_s.astype(np.float32),
        diagnostics=diagnostics,
    )


__all__ = ["SesamWindAndEke", "sesam_wind_and_eke_step"]
