from scripts.sweep_condensate_closure import candidate_overrides, parse_values


def test_sweep_values_are_positive_and_deterministic():
    assert parse_values("3, 6") == (3.0, 6.0)
    rows = list(candidate_overrides((3.0,), (1.0, 2.0), (0.0, 1.0)))
    assert len(rows) == 4
    assert rows[0]["enable_prognostic_condensate"] is True
    assert rows[-1]["condensate_fallout_timescale_days"] == 2.0


def test_sweep_rejects_nonpositive_values():
    try:
        parse_values("0")
    except Exception as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("nonphysical sweep values must be rejected")


def test_raw_column_sweep_enables_the_complete_experimental_closure():
    candidate = next(candidate_overrides((1.0,), (1.0,), (1.0,), raw_column_water=True))
    assert candidate["enable_prognostic_column_water"] is True
    assert candidate["column_water_use_bulk_condensate_rainfall"] is True


def test_stability_aware_sweep_carries_the_requested_trigger_parameters():
    candidate = next(
        candidate_overrides(
            (1.0,), (1.0,), (1.0,), stability_aware=True,
            stability_critical_rh=0.6, stability_cape_scale_j_kg=25.0,
        )
    )
    assert candidate["enable_stability_aware_condensation"] is True
    assert candidate["stability_condensation_critical_rh"] == 0.6
