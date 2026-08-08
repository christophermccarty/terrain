"""Comparable CRU/Köppen scorecards and phase-specific acceptance decisions."""
from __future__ import annotations

from typing import Any, Mapping


SCORECARD_KEYS = (
    "temperature_rmse_c",
    "precipitation_log_rmse",
    "precipitation_log_correlation",
    "koppen_group_accuracy",
    "koppen_class_accuracy",
)


def scorecard(report: Mapping[str, Any]) -> dict[str, float]:
    """Extract the small, comparable objective set from a validation report."""
    metrics = report["metrics"]
    monthly = metrics.get("monthly_climatology") or {}
    temperature = monthly.get("temperature_c") or {}
    precipitation = monthly.get("precipitation_mm_day") or {}
    koppen = metrics.get("koppen_map_skill") or {}
    values = {
        "temperature_rmse_c": temperature.get("monthly_rmse"),
        "precipitation_log_rmse": precipitation.get("monthly_log_rmse"),
        "precipitation_log_correlation": precipitation.get("monthly_log_correlation"),
        "koppen_group_accuracy": koppen.get("group_accuracy"),
        "koppen_class_accuracy": koppen.get("class_accuracy"),
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError("report lacks acceptance metrics: " + ", ".join(missing))
    return {name: float(value) for name, value in values.items()}


def evaluate_precipitation_candidate(
    candidate_report: Mapping[str, Any], baseline_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the short-run precipitation admission gate.

    A candidate must improve log-space precipitation error while preserving both
    Köppen skill measures. Temperature is held to a deliberately small 0.10 C
    RMSE regression allowance because this phase is not a land-energy change.
    The individual gates are retained so a rejected candidate remains useful
    evidence instead of a bare pass/fail.
    """
    candidate = scorecard(candidate_report)
    baseline = scorecard(baseline_report)
    deltas = {name: candidate[name] - baseline[name] for name in SCORECARD_KEYS}
    gates = {
        "precipitation_log_rmse_improves": deltas["precipitation_log_rmse"] < 0.0,
        "precipitation_log_correlation_preserved": deltas[
            "precipitation_log_correlation"
        ] >= -0.005,
        "koppen_group_preserved": deltas["koppen_group_accuracy"] >= 0.0,
        "koppen_class_preserved": deltas["koppen_class_accuracy"] >= 0.0,
        "temperature_rmse_preserved": deltas["temperature_rmse_c"] <= 0.10,
    }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": deltas,
        "gates": gates,
        "accepted": all(gates.values()),
    }


def evaluate_land_candidate(
    candidate_report: Mapping[str, Any], baseline_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Admission gate for a land-energy change screened against CRU."""
    candidate = scorecard(candidate_report)
    baseline = scorecard(baseline_report)
    deltas = {name: candidate[name] - baseline[name] for name in SCORECARD_KEYS}
    gates = {
        "temperature_rmse_improves": deltas["temperature_rmse_c"] < 0.0,
        "precipitation_log_rmse_preserved": deltas["precipitation_log_rmse"] <= 0.02,
        "koppen_group_preserved": deltas["koppen_group_accuracy"] >= 0.0,
        "koppen_class_preserved": deltas["koppen_class_accuracy"] >= 0.0,
    }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": deltas,
        "gates": gates,
        "accepted": all(gates.values()),
    }
