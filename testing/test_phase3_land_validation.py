from __future__ import annotations

from scripts.run_phase3_land_validation import phase3_candidate_overrides


def test_phase3_matrix_is_bounded_and_keeps_the_forcing_contract_fixed():
    candidates = phase3_candidate_overrides()
    assert len(candidates) == 5
    assert len({name for name, _ in candidates}) == 5
    for _, overrides in candidates:
        assert overrides["enable_force_restore_land"] is True
        assert overrides["enable_force_restore_atmospheric_heat_convergence"] is True
