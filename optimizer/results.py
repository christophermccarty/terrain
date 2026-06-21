"""Save, load, and summarise sweep result tables.

Results are stored as CSV (for easy inspection / Excel import) with a
companion JSON sidecar for metadata (param space, run config, timestamp).

Usage
-----
from optimizer.results import save_results, load_results, top_n, summarize

save_results(df_or_list, "results/sweep_2026.csv", metadata={"n_samples": 50})
df = load_results("results/sweep_2026.csv")
print(top_n(df, 10))
print(summarize(df))
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(
    results: "list[dict] | pd.DataFrame",
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write results to CSV + JSON sidecar.

    Parameters
    ----------
    results:
        List of result dicts or a pandas DataFrame.
    path:
        Output CSV path. A companion ``.meta.json`` file is written alongside.
    metadata:
        Optional extra info (param space, run config) stored in the sidecar.

    Returns
    -------
    Path to the written CSV file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if _PANDAS and isinstance(results, __import__("pandas").DataFrame):
        rows = results.to_dict(orient="records")
    else:
        rows = list(results)  # type: ignore[arg-type]

    if not rows:
        path.write_text("")
        return path

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Sidecar
    meta = {
        "created_at": datetime.utcnow().isoformat(),
        "n_results": len(rows),
        **(metadata or {}),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    return path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_results(path: str | Path) -> "list[dict] | pd.DataFrame":
    """Load a results CSV. Returns DataFrame if pandas is available."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    if _PANDAS:
        import pandas as pd  # noqa: PLC0415
        return pd.read_csv(path)

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Coerce numeric strings
    for row in rows:
        for k, v in row.items():
            try:
                row[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def top_n(
    results: "list[dict] | pd.DataFrame",
    n: int = 10,
    score_col: str = "score",
) -> "list[dict] | pd.DataFrame":
    """Return the top-n results sorted by score descending."""
    if _PANDAS and isinstance(results, __import__("pandas").DataFrame):
        return results.nlargest(n, score_col)

    rows = list(results)  # type: ignore[arg-type]
    rows.sort(key=lambda r: float(r.get(score_col, -1.0)), reverse=True)
    return rows[:n]


def summarize(
    results: "list[dict] | pd.DataFrame",
    score_col: str = "score",
) -> dict[str, Any]:
    """Return summary statistics for the result table."""
    if _PANDAS and isinstance(results, __import__("pandas").DataFrame):
        scores = results[score_col].dropna().values.astype(float)
    else:
        scores = np.array([float(r.get(score_col, float("nan"))) for r in results])  # type: ignore[arg-type]
        scores = scores[~np.isnan(scores)]

    if len(scores) == 0:
        return {"n": 0}

    return {
        "n": int(len(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "p25": float(np.percentile(scores, 25)),
        "median": float(np.median(scores)),
        "p75": float(np.percentile(scores, 75)),
        "max": float(np.max(scores)),
        "n_above_65": int(np.sum(scores >= 65)),
        "n_above_80": int(np.sum(scores >= 80)),
    }


def load_metadata(path: str | Path) -> dict[str, Any]:
    """Load the JSON sidecar produced by save_results."""
    meta_path = Path(path).with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def merge_csvs(paths: list[str | Path], output: str | Path) -> "list[dict] | pd.DataFrame":
    """Merge multiple result CSVs into one, removing duplicates by trial_id."""
    all_rows: list[dict] = []
    seen_ids: set[str] = set()
    for p in paths:
        rows = load_results(p)
        if _PANDAS and isinstance(rows, __import__("pandas").DataFrame):
            rows = rows.to_dict(orient="records")
        for row in rows:
            tid = str(row.get("trial_id", id(row)))
            if tid not in seen_ids:
                all_rows.append(row)
                seen_ids.add(tid)

    all_rows.sort(key=lambda r: float(r.get("score", -1.0)), reverse=True)
    save_results(all_rows, output)
    if _PANDAS:
        import pandas as pd  # noqa: PLC0415
        return pd.DataFrame(all_rows)
    return all_rows
