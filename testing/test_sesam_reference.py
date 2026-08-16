"""Tests for the SESAM reference parameter pack (`sesam_reference.py`).

These guard the transcription layer only: table completeness, provenance, and
spot values against the published Willeit et al. (2022) tables. They do not
exercise any physics — the pack is read-only reference data.
"""
from __future__ import annotations

import math

import pytest

import sesam_reference as sr


def test_all_appendix_tables_present():
    for name in sr.ATMOSPHERE_TABLES:
        assert name in sr.SESAM_TABLES, name
    # A3 publishes no parameter table; it is present but intentionally empty.
    assert sr.SESAM_TABLES["A3_thermodynamics"]["entries"] == {}
    assert "no separate parameter table" in (
        sr.SESAM_TABLES["A3_thermodynamics"]["note"]
    )


@pytest.mark.parametrize(
    "table_name,key,expected",
    [
        ("A1_vertical_structure", "H_Gamma_s", 1500.0),
        ("A1_vertical_structure", "r_st", 0.05),
        ("A2_dynamics", "C1_cell", 0.3),
        ("A2_dynamics", "C2_cell", 0.05),
        ("A2_dynamics", "C3_cell", 0.005),
        ("A2_dynamics", "c2mmc", 0.017),   # ITCZ latitude per K of T_NH−T_SH
        ("A2_dynamics", "c5mmc", 750.0),
        ("A4_hydrology", "ra_max", 0.95),  # the 95% RH precipitation threshold
        ("A4_hydrology", "tau_p", 50.0),   # land turnover time, days
        ("A5_synoptic", "c5syn", 2.0e5),   # AT = c5syn·√K
        ("A5_synoptic", "c6syn", 2.0e4),   # Aq = c6syn·K
        ("A6_clouds", "H_pbl", 1500.0),
        ("A6_clouds", "c4cld", 1.5),
        ("A7_shortwave", "a1wv_sw", 0.174),
        ("A7_shortwave", "a2wv_sw", 0.826),  # printed as '1−a1wv'; stored resolved
        ("A8_longwave", "beta0", 1.66),
        ("A8_longwave", "k_co2", 0.8),
    ],
)
def test_spot_values(table_name, key, expected):
    assert sr.value(table_name, key) == pytest.approx(expected, rel=1e-12)


def test_every_entry_is_well_formed():
    for name in sr.ATMOSPHERE_TABLES:
        for key, entry in sr.table(name)["entries"].items():
            assert isinstance(entry["value"], float), (name, key)
            assert math.isfinite(entry["value"]), (name, key)
            assert entry["unit"], (name, key)
            assert entry["symbol"], (name, key)


def test_provenance_is_complete():
    prov = sr.PROVENANCE
    assert "10.5194/gmd-15-5905-2022" in prov["citation"]
    assert prov["license"] == "CC-BY 4.0"
    assert len(prov["source_sha256"]) == 12
    for table_id, digest in prov["source_sha256"].items():
        assert len(digest) == 64, table_id


def test_flagged_transcriptions_are_explicit():
    flagged = sr.flagged_transcriptions()
    # c4Γ/c5Γ/c6Γ were resolved (linear, m^-1) from the paper PDF on
    # 2026-08-16 and are no longer flagged. Only c1tp remains unresolved
    # (per-day folding of the A10 tendency), pending the P5 radiation work.
    assert "c4_Gamma" not in flagged.get("A1_vertical_structure", [])
    assert "c5_Gamma" not in flagged.get("A1_vertical_structure", [])
    assert "c6_Gamma" not in flagged.get("A1_vertical_structure", [])
    assert "c1tp" in flagged["A1_vertical_structure"]


def test_accessor_errors_name_available_keys():
    with pytest.raises(KeyError, match="available"):
        sr.table("not_a_table")
    with pytest.raises(KeyError, match="available"):
        sr.value("A4_hydrology", "not_a_parameter")


def test_table_accessor_returns_a_copy():
    tab = sr.table("A4_hydrology")
    tab["entries"]["ra_max"]["value"] = -1.0
    assert sr.value("A4_hydrology", "ra_max") == pytest.approx(0.95)
