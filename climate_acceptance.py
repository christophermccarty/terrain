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


def _regional_target_errors(report: Mapping[str, Any]) -> dict[str, float]:
    """Return comparable named-region target errors when a report provides them."""
    values = (report.get("metrics") or {}).get("regional_target_error_fraction") or {}
    return {
        str(name): float(value)
        for name, value in values.items()
        if value is not None
    }


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
    candidate_metrics = candidate_report["metrics"]
    baseline_metrics = baseline_report["metrics"]
    candidate_regional = _regional_target_errors(candidate_report)
    baseline_regional = _regional_target_errors(baseline_report)
    regional_names = sorted(set(candidate_regional) | set(baseline_regional))
    regional_deltas = {
        name: candidate_regional.get(name, float("inf")) - baseline_regional.get(name, float("inf"))
        for name in regional_names
    }
    regional_checked = bool(regional_names)
    regional_nonregressing = (
        all(delta <= 1e-12 for delta in regional_deltas.values())
        if regional_checked
        else True
    )
    baseline_reference_error = baseline_metrics.get("reference_error_score")
    candidate_reference_error = candidate_metrics.get("reference_error_score")
    reference_error_checked = (
        baseline_reference_error is not None and candidate_reference_error is not None
    )
    reference_error_nonregressing = (
        float(candidate_reference_error) <= float(baseline_reference_error) + 1e-12
        if reference_error_checked
        else True
    )
    gates = {
        "precipitation_log_rmse_improves": deltas["precipitation_log_rmse"] < 0.0,
        "precipitation_log_correlation_preserved": deltas[
            "precipitation_log_correlation"
        ] >= -0.005,
        "koppen_group_preserved": deltas["koppen_group_accuracy"] >= 0.0,
        "koppen_class_preserved": deltas["koppen_class_accuracy"] >= 0.0,
        "temperature_rmse_preserved": deltas["temperature_rmse_c"] <= 0.10,
        # Map-level scores can improve while moving a named regional target in
        # the wrong direction. Real-terrain reports provide these diagnostics;
        # generic/unit reports need not.
        "regional_target_errors_nonregressing": regional_nonregressing,
        "reference_error_score_nonregressing": reference_error_nonregressing,
    }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": deltas,
        "regional_target_error_delta": regional_deltas,
        "regional_target_errors_checked": regional_checked,
        "reference_error_score_checked": reference_error_checked,
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


def evaluate_phase3_candidate(
    candidate_report: Mapping[str, Any], baseline_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Strict Phase 3 replacement gate against the supported land baseline."""
    decision = evaluate_land_candidate(candidate_report, baseline_report)
    candidate_metrics = candidate_report["metrics"]
    baseline_metrics = baseline_report["metrics"]
    candidate_thresholds = candidate_metrics.get("koppen_temperature_thresholds") or {}
    baseline_thresholds = baseline_metrics.get("koppen_temperature_thresholds") or {}
    candidate_cycle = candidate_metrics.get("land_seasonal_cycle") or {}
    baseline_cycle = baseline_metrics.get("land_seasonal_cycle") or {}
    convergence = candidate_metrics.get("atmospheric_heat_convergence") or {}
    candidate_regional = _regional_target_errors(candidate_report)
    baseline_regional = _regional_target_errors(baseline_report)
    regional_names = sorted(set(candidate_regional) | set(baseline_regional))
    regional_deltas = {
        name: candidate_regional.get(name, float("inf"))
        - baseline_regional.get(name, float("inf"))
        for name in regional_names
    }

    threshold_delta: dict[str, float] = {}
    for key in ("coldest_month", "warmest_month"):
        candidate_value = (candidate_thresholds.get(key) or {}).get("accuracy")
        baseline_value = (baseline_thresholds.get(key) or {}).get("accuracy")
        if candidate_value is None or baseline_value is None:
            raise ValueError(f"report lacks Phase 3 threshold metric {key}")
        threshold_delta[key] = float(candidate_value) - float(baseline_value)
    candidate_cycle_error = candidate_cycle.get("cycle_error_score")
    baseline_cycle_error = baseline_cycle.get("cycle_error_score")
    if candidate_cycle_error is None or baseline_cycle_error is None:
        raise ValueError("report lacks Phase 3 land-cycle metric")

    extra_gates = {
        "coldest_month_threshold_preserved": threshold_delta["coldest_month"] >= 0.0,
        "warmest_month_threshold_preserved": threshold_delta["warmest_month"] >= 0.0,
        "land_cycle_shape_improves": float(candidate_cycle_error) < float(baseline_cycle_error),
        "precipitation_log_correlation_preserved": (
            decision["delta"]["precipitation_log_correlation"] >= -0.005
        ),
        "regional_precipitation_skill_preserved": (
            all(delta <= 1e-12 for delta in regional_deltas.values())
            if regional_names else True
        ),
        "heat_convergence_diagnosed": bool(convergence),
        "heat_convergence_globally_closed": (
            bool(convergence)
            and float(convergence.get("max_abs_global_area_mean_w_m2", float("inf"))) <= 1e-3
        ),
    }
    gates = {**decision["gates"], **extra_gates}
    return {
        **decision,
        "threshold_accuracy_delta": threshold_delta,
        "land_cycle_error_delta": float(candidate_cycle_error) - float(baseline_cycle_error),
        "regional_target_error_delta": regional_deltas,
        "regional_target_errors_checked": bool(regional_names),
        "gates": gates,
        "accepted": all(gates.values()),
    }
