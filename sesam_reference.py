"""SESAM (CLIMBER-X) reference parameter pack — versioned, read-only.

Transcribed 2026-08-16 from the published parameter tables of

    Willeit, M., Ganopolski, A., Robinson, A., and Edwards, N. R.:
    *The Earth system model CLIMBER-X v1.0 – Part 1: Climate model
    description and validation*, Geosci. Model Dev., 15, 5905–5948,
    https://doi.org/10.5194/gmd-15-5905-2022 (CC-BY 4.0).

The paper's HTML renders its parameter tables only as XLSX attachments
("Download XLSX"); this module is the project's single source of truth for
those constants so the SESAM adoption stages in ``docs/SESAM_GAP_ANALYSIS.md``
never re-derive or hand-copy them per call site.

**Licensing note.** The equations are taken from the CC-BY 4.0 paper. The
CLIMBER-X *code* (https://github.com/cxesmc/climber-x) is GPL-3.0 Fortran and
must not be copied or translated; consulting its namelist defaults to
cross-check these numbers is fine. Nothing here is GPL-sourced.

**Transcription caution.** A few table units involve superscript mangling in
the source XLSX or read ambiguously against the HTML equation text
(``transcription_note`` fields). Before *using* any entry in code, check the
equation cited in its ``equations`` field against the paper PDF. Values
themselves are numeric transcribes of the printed table cells, no fits.
"""
from __future__ import annotations

from typing import Any

PROVENANCE: dict[str, Any] = {
    "citation": (
        "Willeit, M., Ganopolski, A., Robinson, A., and Edwards, N. R.: "
        "The Earth system model CLIMBER-X v1.0 – Part 1: Climate model "
        "description and validation, Geosci. Model Dev., 15, 5905–5948, "
        "https://doi.org/10.5194/gmd-15-5905-2022, 2022."
    ),
    "license": "CC-BY 4.0",
    "fetched": "2026-08-16",
    "url_pattern": (
        "https://gmd.copernicus.org/articles/15/5905/2022/"
        "gmd-15-5905-2022-{table_id}.xlsx"
    ),
    "source_sha256": {
        "t01": "A80325C0F18C3A2E96019B56134841B860BE64599B8A9BFC2B49474854418819",
        "t02": "3D17313DE56E8B77E01267C7D58F45E2024BAE8CAE1A45AC224A7843E5B7D80F",
        "t03": "1B771C0DE3573520F226FF9D84EA3EAF9C225FE5A6C629382DACA93CEA418D0C",
        "t04": "7422DBC9FF57236B99632D4919F8875274D1C081BC58F95E539079B629BFEAF0",
        "t05": "4D5F90F1A3F23D9F93EFDA49EC85375408F2A14C72327D22FC6F123D53CA25DF",
        "t06": "5418602F76CDEA332F6D09F2F064A97194D5D48DF649B1A413E6DE54E64138C7",
        "t07": "3D750E39B6E68214620C19170DC920E0D636E2E1836F91F8F024D0DAE6C83F72",
        "t08": "E5A95C5E342E37B053050E8BFE93C7BE2B0AB2AFBAEE13ACC06174F44C8F9CDF",
        "t09": "402DE6DD9AE33077744BB0AF180F6B9E1FF6C1D5423BE07DE508122CBBE8CEAA",
        "t10": "4FB84CFF12E5A8F9777C84CA6596B4C9AAAB0177A66B282B1B21EE9E10B51D26",
        "t11": "AFBEA04A9F7E587B2BE3A7E20F84CC873D4137760EFBC4BFD0C23D3AD19F87E2",
        "t12": "6DA7A53196655BD610C2FB22119EABAB526C6BDEB37CD1127F9F31274EC5A240",
    },
}


def _p(
    value: float,
    unit: str,
    symbol: str,
    equations: tuple[str, ...] = (),
    note: str = "",
    documentation: str = "",
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "value": float(value),
        "unit": unit,
        "symbol": symbol,
        "equations": equations,
    }
    if documentation:
        entry["note"] = documentation
    if note:
        entry["transcription_note"] = note
    return entry


# ---------------------------------------------------------------------------
# Appendix parameter tables (paper symbols kept in ``symbol``; keys are ASCII)
# ---------------------------------------------------------------------------

TABLE_A1_VERTICAL_STRUCTURE: dict[str, Any] = {
    "source_xlsx": "t03",
    "entries": {
        # --- lapse-rate / temperature profile (A5-A9) ---
        "c1_Gamma": _p(3.8e-3, "K m^-1", "c1Γ", ("A7",)),
        "c2_Gamma": _p(0.02, "K m^-1", "c2Γ", ("A7", "A8")),
        "c3_Gamma": _p(6.0e-3, "K m^-1", "c3Γ", ("A8",)),
        "c4_Gamma": _p(
            5.0e-3, "m^-1", "c4Γ", ("A9",),
            documentation="A9 read from the paper PDF (2026-08-16) is "
            "linear: Γs = c4Γ·max(0, Ta−T*) over ocean, so the unit is m^-1",
        ),
        "c5_Gamma": _p(2.0e-3, "m^-1", "c5Γ", ("A9",),
            documentation="A9 confirmed linear from the PDF: Γs = "
            "c5Γ·(Ta−T*) for warm land and ice"),
        "c6_Gamma": _p(10.0e-3, "m^-1", "c6Γ", ("A9",),
            documentation="A9 confirmed linear from the PDF: Γs = "
            "c6Γ·(Ta−T*) for cold (inverted) land"),
        "H_Gamma_s": _p(1500.0, "m", "HΓ,s", ("A6",)),
        "H_Gamma_t": _p(15000.0, "m", "HΓ,t", ("A6",)),
        # --- relative-humidity profile (A13-A14) ---
        "c1r": _p(2500.0, "m", "c1r", ("A14",)),
        "c2r": _p(200.0, "s m^-1", "c2r", ("A14",)),
        "c3r": _p(2.4, "-", "c3r", ("A14",)),
        "c4r": _p(3000.0, "m", "c4r", ("A13",)),
        "c5r": _p(1000.0, "m", "c5r", ("A13",),
            note="defines PBL top: zpbl = zs + c5r"),
        "r_st": _p(0.05, "-", "rst", ("A13",),
            note="uniform stratospheric relative humidity"),
        # --- tropopause (A10-A11) ---
        "c1tp": _p(100.0, "m^3 W^-1", "c1tp", ("A10",),
            note="unit printed 'm3 W−1'; per-day vs per-second folding must be "
            "checked against the PDF"),
        "c2tp": _p(18.0, "W m^-2", "c2tp", ("A11",)),
        "c3tp": _p(1.0, "-", "c3tp", ("A11",)),
    },
}

TABLE_A2_DYNAMICS: dict[str, Any] = {
    "source_xlsx": "t04",
    "entries": {
        # --- mean meridional circulation (A30-A35) ---
        "C1_cell": _p(0.3, "-", "C1", ("A30",),
            note="Hadley-cell strength coefficient (i=1)"),
        "C2_cell": _p(0.05, "-", "C2", ("A30",),
            note="Ferrel-cell strength coefficient (i=2)"),
        "C3_cell": _p(0.005, "-", "C3", ("A30",),
            note="polar-cell strength coefficient (i=3)"),
        "c1mmc": _p(5.0, "-", "c1mmc", ("A31",)),
        "c2mmc": _p(0.017, "K^-1", "c2mmc", ("A32",),
            note="ITCZ latitude = c2mmc·(T_NH − T_SH)"),
        "c3mmc": _p(90.0, "K", "c3mmc", ("A33",),
            note="Hadley width = c3mmc·(T_trp − c4mmc)"),
        "c4mmc": _p(200.0, "K", "c4mmc", ("A33",)),
        "c5mmc": _p(750.0, "m", "c5mmc", ("A35",),
            note="topography factor Fz = mean(1 − zs/c5mmc)"),
        # --- azonal sea-level pressure (A37-A38) ---
        "H0_slp": _p(10000.0, "m", "H0", ("A37",),
            note="thermal SLP: psl,T* = −(g·p0·H0/(2·Rd·T0²))·Tsl*"),
        "tau_e": _p(5.0, "d", "τe", ("A38",),
            note="Charney–Eliassen wave damping time"),
    },
}

TABLE_A3_THERMODYNAMICS: dict[str, Any] = {
    "source_xlsx": None,
    "entries": {},
    "note": (
        "The paper publishes no separate parameter table for the A3 column-"
        "energy equation beyond physical constants (cv, Le, Ls). The internal "
        "substep is stated in prose as ~2 h."
    ),
}

TABLE_A4_HYDROLOGY: dict[str, Any] = {
    "source_xlsx": "t05",
    "entries": {
        "tau_p": _p(50.0, "d", "τp", ("A44",),
            note="land precipitation turnover: P += Qq·ra/τp"),
        "ra_max": _p(0.95, "-", "ramax", ("A44",),
            note="critical near-surface RH; all excess converged water rains "
            "out at ra = ramax"),
        "c_slope_p": _p(0.005, "-", "cslopep", ("A45",),
            note="slope-convergence precipitation term Cslope = cslopep·√K·"
            "|∇zs|·ρ0·qa"),
    },
}

TABLE_A5_SYNOPTIC: dict[str, Any] = {
    "source_xlsx": "t06",
    "entries": {
        "c1syn": _p(1.0e-4, "m^2 s^-3", "c1syn", ("A53",),
            note="EKE production baseline term"),
        "c2syn": _p(1.6e4, "m^2 s^-2", "c2syn", ("A53",),
            note="EKE production ∝ Eady growth rate (f/N)·|∂u/∂z|"),
        "c3syn": _p(8.0e-7, "m^-1", "c3syn", ("A55",)),
        "c4syn": _p(1.0e-4, "m^-1", "c4syn", ("A55",),
            note="EKE dissipation (c3syn + c4syn·CD)·K^1.5"),
        "c5syn": _p(2.0e5, "m", "c5syn", ("A50",),
            note="heat diffusivity AT = c5syn·√K"),
        "c6syn": _p(2.0e4, "s", "c6syn", ("A51",),
            note="moisture diffusivity Aq = c6syn·K"),
        "c7syn": _p(0.7, "-", "c7syn", ("A56",),
            note="synoptic surface wind Usyn = c7syn·ε·cosα·√K"),
        "c8syn": _p(1.0e-3, "-", "c8syn", ("A57",),
            note="synoptic 700 hPa vertical velocity wsyn = c8syn·√K"),
    },
}

TABLE_A6_CLOUDS: dict[str, Any] = {
    "source_xlsx": "t07",
    "entries": {
        # --- cloud fraction (A61-A66) ---
        "c1cld": _p(0.47, "-", "c1cld", ("A62",)),
        "c2cld": _p(0.5, "-", "c2cld", ("A62",)),
        "c3cld": _p(200.0, "s m^-1", "c3cld", ("A62", "A63")),
        "c4cld": _p(1.5, "-", "c4cld", ("A62", "A65")),
        "c5cld": _p(0.5, "-", "c5cld", ("A65",),
            note="inversion low-cloud term (the Atacama/stratocumulus class "
            "of cloud PlanetSim lacks)"),
        "c6cld": _p(0.1, "-", "c6cld", ("A65",)),
        "c7cld": _p(0.003, "kg kg^-1", "c7cld", ("A66",),
            note="freeze-dry factor scale (Vavrus & Waliser 2008)"),
        "c_weff": _p(0.25, "-", "cweff", ("A63",)),
        "c_woro": _p(1.0e-5, "m^-1", "cworo", ("A64",),
            note="orographic vertical velocity woro = cworo·Us·σoro"),
        # --- cloud top height (A67) ---
        "H_pbl": _p(1500.0, "m", "Hpbl", ("A67",),
            note="cloud base = PBL top"),
        "c1hcld": _p(2000.0, "m", "c1hcld", ("A67",)),
        "c2hcld": _p(0.27, "-", "c2hcld", ("A67",)),
        "c3hcld": _p(200.0, "s m^-1", "c3hcld", ("A67",)),
        # --- cloud optical thickness (A68) ---
        "c1tau": _p(5.0, "K", "c1τ", ("A68",)),
        "c2tau": _p(30.0, "K", "c2τ", ("A68",)),
        "c3tau": _p(2.0, "-", "c3τ", ("A68",)),
        "c4tau": _p(0.5, "-", "c4τ", ("A68",)),
    },
}

TABLE_A7_SHORTWAVE: dict[str, Any] = {
    "source_xlsx": "t08",
    "entries": {
        "r_sct": _p(0.17, "-", "rsct", ("A79", "A80")),
        "g_cld": _p(0.14, "-", "gcld", ("A81", "A82")),
        "p1": _p(-1.97, "-", "p1", ("A79", "A80")),
        "p2": _p(0.82, "-", "p2", ("A79", "A80")),
        "p3": _p(0.35, "-", "p3", ("A81", "A82")),
        "p4": _p(0.67, "-", "p4", ("A81", "A82")),
        "alpha1": _p(7.73e-2, "-", "α1", ("A79", "A80")),
        "alpha2": _p(2.39e-2, "-", "α2", ("A79", "A80")),
        "alpha3": _p(1.51e2, "-", "α3", ("A79", "A80")),
        "gamma1aer": _p(2.75, "-", "γ1aer", ("A88",)),
        "gamma2aer": _p(0.636, "-", "γ2aer", ("A88",)),
        "D_cld": _p(1000.0, "m", "Dcld", ("A93",),
            note="cloud geometric depth used in SW absorber-mass attenuation"),
        "a1wv_sw": _p(0.174, "-", "a1wv", ("A87",)),
        "a2wv_sw": _p(
            0.826, "-", "a2wv", ("A87",),
            note="printed as the expression '1−a1wv'; value stored resolved",
        ),
        "b1wv_sw": _p(6.27, "-", "b1wv", ("A87",)),
        "b2wv_sw": _p(0.0267, "-", "b2wv", ("A87",)),
    },
}

TABLE_A8_LONGWAVE: dict[str, Any] = {
    "source_xlsx": "t09",
    "entries": {
        "beta0": _p(1.66, "-", "β0", ("A110", "A111")),
        "a1wv_lw": _p(1.5, "-", "a1wv", ("A110",)),
        "a2wv_lw": _p(0.1, "-", "a2wv", ("A110",)),
        "a3wv_lw": _p(0.01, "-", "a3wv", ("A110",)),
        "beta1wv": _p(0.42, "-", "β1wv", ("A110",)),
        "beta2wv": _p(1.5, "-", "β2wv", ("A110",)),
        "beta3wv": _p(3.0, "-", "β3wv", ("A110",)),
        "k_wv": _p(1.0, "-", "kwv", ("A114",),
            note="pressure weighting of water-vapour absorber mass"),
        "a0co2": _p(0.247, "-", "a0CO2", ("A111",)),
        "a1co2": _p(0.755, "-", "a1CO2", ("A111",)),
        "beta_co2": _p(0.45, "-", "βCO2", ("A111",)),
        "k_co2": _p(0.8, "-", "kCO2", ("A115",)),
        "a0o3": _p(8.246, "-", "aO3", ("A112",)),
        "beta_o3": _p(0.539, "-", "βO3", ("A112",)),
        "k_o3": _p(0.6, "-", "kO3", ("A116",)),
    },
}

# ---------------------------------------------------------------------------
# Non-SESAM appendix tables, transcribed for later ocean/sea-ice cross-checks.
# Out of scope for the SESAM adoption stages (docs/SESAM_GAP_ANALYSIS.md §9).
# ---------------------------------------------------------------------------

TABLE_MAIN_ENERGY_BUDGET: dict[str, Any] = {
    "source_xlsx": "t01",
    "scope": "validation_reference",
    "note": "CLIMBER-X simulated vs observed global energy budget components "
    "(W m^-2); Table-3-class validation data, not parameters.",
    "entries": {
        "toa_solar_down": _p(340.2, "W m^-2", "SW↓,TOA"),
        "toa_solar_up": _p(102.2, "W m^-2", "SW↑,TOA"),
        "toa_solar_net": _p(238.1, "W m^-2", "SWnet,TOA"),
        "toa_thermal_up": _p(237.6, "W m^-2", "LW↑,TOA"),
        "atm_solar_net": _p(72.6, "W m^-2", "SWnet,atm"),
        "atm_thermal_net": _p(-177.1, "W m^-2", "LWnet,atm"),
        "sfc_solar_down": _p(192.0, "W m^-2", "SW↓,sfc"),
        "sfc_solar_up": _p(26.5, "W m^-2", "SW↑,sfc"),
        "sfc_solar_net": _p(165.5, "W m^-2", "SWnet,sfc"),
        "sfc_thermal_down": _p(338.2, "W m^-2", "LW↓,sfc"),
        "sfc_thermal_up": _p(398.8, "W m^-2", "LW↑,sfc"),
        "sfc_thermal_net": _p(-60.5, "W m^-2", "LWnet,sfc"),
        "sfc_net_radiation": _p(105.0, "W m^-2", "Rnet"),
        "sfc_latent_heat": _p(82.6, "W m^-2", "LE"),
        "sfc_sensible_heat": _p(21.2, "W m^-2", "SH"),
    },
}

TABLE_MAIN_WATER_BUDGET: dict[str, Any] = {
    "source_xlsx": "t02",
    "scope": "validation_reference",
    "note": "CLIMBER-X vs observed global water budget (km^3 d^-1 per the "
    "paper's table heading).",
    "entries": {
        "precipitation_global": _p(531.0, "10^3 km^3 yr^-1", "P"),
        "precipitation_land": _p(123.0, "10^3 km^3 yr^-1", "P_land"),
        "precipitation_ocean": _p(409.0, "10^3 km^3 yr^-1", "P_ocean"),
        "evaporation_global": _p(531.0, "10^3 km^3 yr^-1", "E"),
        "evaporation_land": _p(78.0, "10^3 km^3 yr^-1", "E_land"),
        "evaporation_ocean": _p(453.0, "10^3 km^3 yr^-1", "E_ocean"),
        "runoff": _p(41.0, "10^3 km^3 yr^-1", "R"),
    },
}

TABLE_MAIN_LW_ABSORBERS: dict[str, Any] = {
    "source_xlsx": "t10",
    "scope": "validation_reference",
    "note": "Longwave greenhouse separation in W m^-2 (Schmidt et al. 2010 "
    "comparison); used to sanity-check a future A8 implementation.",
    "entries": {
        "gh_co2": _p(21.2, "W m^-2", "CO2"),
        "gh_h2o_vapour": _p(52.2, "W m^-2", "H2O"),
        "gh_o3": _p(4.3, "W m^-2", "O3"),
        "gh_clouds": _p(19.6, "W m^-2", "clouds"),
        "gh_h2o_plus_co2": _p(84.3, "W m^-2", "H2O+CO2"),
    },
}

TABLE_B_GOLDSTEIN: dict[str, Any] = {
    "source_xlsx": "t11",
    "scope": "reference_only",
    "entries": {
        "KI": _p(1500.0, "m^2 s^-1", "KI"),
        "kappa_isopycnal": _p(1500.0, "m^2 s^-1", "κ"),
        "KD_min": _p(1.0e-5, "m^2 s^-1", "KDmin"),
        "KD_max": _p(1.5e-4, "m^2 s^-1", "KDmax"),
        "z_ref_ocean": _p(1000.0, "m", "zref"),
        "mu0": _p(4.0, "d^-1", "μ0"),
    },
}

TABLE_C_SISIM: dict[str, Any] = {
    "source_xlsx": "t12",
    "scope": "reference_only",
    "entries": {
        "emissivity_ice": _p(0.99, "-", "ε"),
        "lambda_snow": _p(0.3, "W m^-1 K^-1", "λsnow"),
        "lambda_ice": _p(2.2, "W m^-1 K^-1", "λice"),
        "Ch": _p(0.0058, "-", "Ch"),
        "u_star": _p(0.01, "m s^-1", "u⋆"),
        "h0": _p(0.5, "m", "h0"),
        "Cdw": _p(3.24e-3, "-", "Cdw"),
    },
}

SESAM_TABLES: dict[str, dict[str, Any]] = {
    "A1_vertical_structure": TABLE_A1_VERTICAL_STRUCTURE,
    "A2_dynamics": TABLE_A2_DYNAMICS,
    "A3_thermodynamics": TABLE_A3_THERMODYNAMICS,
    "A4_hydrology": TABLE_A4_HYDROLOGY,
    "A5_synoptic": TABLE_A5_SYNOPTIC,
    "A6_clouds": TABLE_A6_CLOUDS,
    "A7_shortwave": TABLE_A7_SHORTWAVE,
    "A8_longwave": TABLE_A8_LONGWAVE,
    "main_energy_budget": TABLE_MAIN_ENERGY_BUDGET,
    "main_water_budget": TABLE_MAIN_WATER_BUDGET,
    "main_lw_absorbers": TABLE_MAIN_LW_ABSORBERS,
    "B_goldstein": TABLE_B_GOLDSTEIN,
    "C_sisim": TABLE_C_SISIM,
}

# Tables the SESAM adoption stages (docs/SESAM_GAP_ANALYSIS.md §7) depend on.
ATMOSPHERE_TABLES: tuple[str, ...] = (
    "A1_vertical_structure",
    "A2_dynamics",
    "A4_hydrology",
    "A5_synoptic",
    "A6_clouds",
    "A7_shortwave",
    "A8_longwave",
)


def table(name: str) -> dict[str, Any]:
    """Return the named table dict (defensive copy of its entries)."""
    if name not in SESAM_TABLES:
        raise KeyError(
            f"unknown SESAM table {name!r}; available: {sorted(SESAM_TABLES)}"
        )
    tab = SESAM_TABLES[name]
    return {**tab, "entries": {k: dict(v) for k, v in tab["entries"].items()}}


def value(table_name: str, key: str) -> float:
    """Return one parameter's numeric value, e.g. ``value('A4_hydrology', 'ra_max')``."""
    entries = SESAM_TABLES[table_name]["entries"]
    if key not in entries:
        raise KeyError(
            f"{table_name} has no parameter {key!r}; available: {sorted(entries)}"
        )
    return float(entries[key]["value"])


def flagged_transcriptions() -> dict[str, list[str]]:
    """Parameters whose units/forms carry a verify-against-the-PDF note.

    c4Γ/c5Γ/c6Γ were cleared 2026-08-16 when the A9 equation was read from
    the paper PDF and confirmed linear; c1tp remains unresolved pending the
    P5 radiation work that uses it (the per-day folding of its ``m^3 W^-1``
    unit is not fixed by the A10 text alone).
    """
    out: dict[str, list[str]] = {}
    for name, tab in SESAM_TABLES.items():
        flagged = [
            key for key, entry in tab["entries"].items()
            if "transcription_note" in entry
        ]
        if flagged:
            out[name] = flagged
    return out
