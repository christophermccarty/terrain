from __future__ import annotations

from scripts.score_external_dycore_reference import _json_safe


def test_json_safe_replaces_nonfinite_metrics_with_null():
    result = _json_safe({"finite": 1.0, "undefined": float("nan"), "nested": [float("inf")]})
    assert result == {"finite": 1.0, "undefined": None, "nested": [None]}
