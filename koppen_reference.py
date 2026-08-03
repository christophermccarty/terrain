"""Gridded Köppen-Geiger reference map: decoding, regridding, and map skill scoring.

This closes ``docs/ACCURACY_AUDIT.md`` H10 -- the audit's own "arguably the
highest-leverage missing piece of *infrastructure* (not physics) in the whole
project".  Before this module, every validation signal in the project was a
**spatial aggregate**: 9 named boxes, 6 zonal bands, and a handful of global
Köppen area fractions.  All three average over exactly the errors section A of
the audit is about -- a desert core that is wet in the right latitude band, or a
continental interior that is dry while its row mean is fine, is invisible to a
zonal mean and to a global share.

The reference product is
``Koppen_classification_world_map_1991-2020_-3C_borderless.png`` at the repo
root -- the user's designated ground truth (see the ``koppen-reference-map``
memory note).  It is a 3600x1800 (0.1 deg) equirectangular raster in the
standard 31-colour Beck et al. legend, spanning the full globe
(-180..180 lon, 90..-90 lat).  Verified by direct spot check: Amazon -> Af,
Sahara -> BWh, Chicago -> Dfa, London -> Cfb, Moscow -> Dfb, Atacama -> BWk,
Greenland/Antarctica -> EF, open ocean -> white.

**What this is not**: it is a *classification* reference, not a gridded
temperature/precipitation product.  Per-cell T/P RMSE against ERA5/CRU still
needs a licensed gridded pack (see ``EARTH_ZONAL_REFERENCE`` in
``real_terrain_validation.py``, which remains the only T/P anchor).  What a
classification map *does* give is a per-cell verdict that is sensitive to
regional pattern error, which is the specific blind spot H10 names.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from climate_averages import (
    KOPPEN_AF,
    KOPPEN_AM,
    KOPPEN_AW,
    KOPPEN_BSH,
    KOPPEN_BSK,
    KOPPEN_BWH,
    KOPPEN_BWK,
    KOPPEN_CFA,
    KOPPEN_CFB,
    KOPPEN_CFC,
    KOPPEN_CSA,
    KOPPEN_CSB,
    KOPPEN_CWA,
    KOPPEN_DFA,
    KOPPEN_DFB,
    KOPPEN_DFC,
    KOPPEN_DWD,
    KOPPEN_EF,
    KOPPEN_ET,
    KOPPEN_NAMES,
    KOPPEN_OCEAN,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_PATH = (
    ROOT / "Koppen_classification_world_map_1991-2020_-3C_borderless.png"
)

# Number of distinct codes in the project's own Köppen vocabulary (0 = ocean).
N_KOPPEN_CODES = 20

# ---------------------------------------------------------------------------
# Reference palette
# ---------------------------------------------------------------------------
# Exact RGB triples as they appear in the bundled PNG (palette mode, no
# antialiasing -- there are exactly 31 distinct colours, so an exact lookup is
# correct and any deviation means the file was swapped).  These are the standard
# Beck et al. (2018) Köppen-Geiger legend colours; a few differ by +-1 from the
# nominal published values (e.g. 254 vs 255) purely from the palette encoding.
REFERENCE_PALETTE: dict[tuple[int, int, int], str] = {
    (0, 0, 254): "Af",
    (0, 119, 255): "Am",
    (70, 169, 250): "Aw",
    (127, 201, 255): "As",
    (254, 0, 0): "BWh",
    (254, 150, 149): "BWk",
    (245, 163, 1): "BSh",
    (255, 219, 99): "BSk",
    (255, 255, 0): "Csa",
    (198, 199, 0): "Csb",
    (150, 150, 0): "Csc",
    (150, 255, 150): "Cwa",
    (99, 199, 100): "Cwb",
    (50, 150, 51): "Cwc",
    (198, 255, 78): "Cfa",
    (102, 255, 51): "Cfb",
    (51, 199, 1): "Cfc",
    (255, 0, 254): "Dsa",
    (198, 0, 199): "Dsb",
    (150, 50, 149): "Dsc",
    (150, 100, 150): "Dsd",
    (171, 177, 255): "Dwa",
    (90, 119, 219): "Dwb",
    (76, 81, 181): "Dwc",
    (50, 0, 135): "Dwd",
    (0, 255, 255): "Dfa",
    (56, 199, 255): "Dfb",
    (0, 126, 125): "Dfc",
    (0, 69, 94): "Dfd",
    (178, 178, 178): "ET",
    (104, 104, 104): "EF",
    (255, 255, 255): "Ocean",  # ocean / no-data
}

# The reference distinguishes 30 land classes; this project's classifier emits
# 19.  Folding is therefore required, and is done toward the nearest class the
# model can actually express.  Every fold preserves the Köppen *group* letter
# (A/B/C/D/E) and the thermal sub-letter (a/b/c/d) wherever the model has one,
# so group-level scoring -- the primary metric below -- is completely unaffected
# by these choices.  Only the finer class-level score depends on them.
#
#   As   -> Aw   model has no dry-summer savanna variant
#   Csc  -> Csb  no cold-summer Mediterranean
#   Cwb/Cwc -> Cwa   no highland dry-winter variants
#   Ds*  -> Df*  no dry-summer continental; folded by thermal sub-letter
#   Dwa/Dwb/Dwc -> Dfa/Dfb/Dfc   model keeps only Dwd from the Dw family
#   Dsd/Dfd -> Dwd   model's Dwd is its single "extreme continental" bucket
REFERENCE_TO_MODEL_CODE: dict[str, int] = {
    "Af": KOPPEN_AF,
    "Am": KOPPEN_AM,
    "Aw": KOPPEN_AW,
    "As": KOPPEN_AW,
    "BWh": KOPPEN_BWH,
    "BWk": KOPPEN_BWK,
    "BSh": KOPPEN_BSH,
    "BSk": KOPPEN_BSK,
    "Csa": KOPPEN_CSA,
    "Csb": KOPPEN_CSB,
    "Csc": KOPPEN_CSB,
    "Cwa": KOPPEN_CWA,
    "Cwb": KOPPEN_CWA,
    "Cwc": KOPPEN_CWA,
    "Cfa": KOPPEN_CFA,
    "Cfb": KOPPEN_CFB,
    "Cfc": KOPPEN_CFC,
    "Dsa": KOPPEN_DFA,
    "Dsb": KOPPEN_DFB,
    "Dsc": KOPPEN_DFC,
    "Dsd": KOPPEN_DWD,
    "Dwa": KOPPEN_DFA,
    "Dwb": KOPPEN_DFB,
    "Dwc": KOPPEN_DFC,
    "Dwd": KOPPEN_DWD,
    "Dfa": KOPPEN_DFA,
    "Dfb": KOPPEN_DFB,
    "Dfc": KOPPEN_DFC,
    "Dfd": KOPPEN_DWD,
    "ET": KOPPEN_ET,
    "EF": KOPPEN_EF,
    "Ocean": KOPPEN_OCEAN,
}

# Group letter per model code.  0 = ocean/none, 1..5 = A,B,C,D,E.
GROUP_NONE, GROUP_A, GROUP_B, GROUP_C, GROUP_D, GROUP_E = range(6)
GROUP_LABELS = ("none", "A", "B", "C", "D", "E")
GROUP_DESCRIPTIONS = {
    "A": "tropical",
    "B": "arid",
    "C": "temperate",
    "D": "continental",
    "E": "polar",
}

_CODE_TO_GROUP = np.zeros(N_KOPPEN_CODES, dtype=np.int8)
for _code, _name in KOPPEN_NAMES.items():
    if _code == KOPPEN_OCEAN:
        continue
    _CODE_TO_GROUP[_code] = {"A": GROUP_A, "B": GROUP_B, "C": GROUP_C,
                             "D": GROUP_D, "E": GROUP_E}[_name[0]]


def koppen_group(codes: np.ndarray) -> np.ndarray:
    """Map model Köppen codes to group indices (0 none, 1..5 = A..E)."""
    values = np.asarray(codes, dtype=np.int64)
    if values.size and (values.min() < 0 or values.max() >= N_KOPPEN_CODES):
        raise ValueError("Köppen codes out of range for the project vocabulary")
    return _CODE_TO_GROUP[values]


def short_class_name(code: int) -> str:
    """``8`` -> ``'Cfa'`` (the ``KOPPEN_NAMES`` prefix before the description)."""
    return KOPPEN_NAMES.get(int(code), "?").split(" - ")[0]


# ---------------------------------------------------------------------------
# Decoding and regridding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceGrid:
    """Reference Köppen classes regridded onto an ``H x W`` model grid.

    ``codes`` holds the majority *land* class per cell in the project's own code
    space (``KOPPEN_OCEAN`` where the cell has no land at all).  ``land_fraction``
    is the fraction of the underlying 0.1 deg reference pixels in that cell that
    are land, which lets a caller decide how much land a cell needs before it is
    worth scoring (coastal cells are genuinely ambiguous at coarse resolution).
    """

    codes: np.ndarray
    land_fraction: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.codes.shape  # type: ignore[return-value]


@lru_cache(maxsize=2)
def _decode_reference_pixels(path: Path) -> np.ndarray:
    """Return the raw reference raster as model Köppen codes (source resolution)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Köppen reference map not found: {path}. This is the repo-root PNG "
            "designated as the project's climate ground truth."
        )
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"reference map must be RGB, got shape {rgb.shape}")
    height, width = rgb.shape[:2]
    if width != 2 * height:
        raise ValueError(
            f"reference map must be 2:1 equirectangular, got {width}x{height}"
        )

    # Exact palette lookup via a packed 24-bit key.  The source is a palette PNG
    # with no antialiasing, so every pixel must match a legend entry exactly; an
    # unknown colour means the file was replaced with a differently-styled map
    # and silently misclassifying it would poison every downstream metric.
    packed = (
        rgb[..., 0].astype(np.int32) << 16
        | rgb[..., 1].astype(np.int32) << 8
        | rgb[..., 2].astype(np.int32)
    )
    lookup = np.full(1 << 24, -1, dtype=np.int16)
    for (r, g, b), label in REFERENCE_PALETTE.items():
        lookup[(r << 16) | (g << 8) | b] = REFERENCE_TO_MODEL_CODE[label]

    codes = lookup[packed]
    if np.any(codes < 0):
        bad = np.unique(packed[codes < 0])[:8]
        rendered = ", ".join(
            f"#{int(value):06x}" for value in bad
        )
        raise ValueError(
            f"reference map contains {int(np.sum(codes < 0))} pixels in "
            f"{np.unique(packed[codes < 0]).size} unrecognized colours "
            f"(first: {rendered}); REFERENCE_PALETTE needs updating"
        )
    return codes.astype(np.int8)


@lru_cache(maxsize=8)
def _reference_grid_cached(
    height: int, width: int, path_str: str
) -> tuple[np.ndarray, np.ndarray]:
    source = _decode_reference_pixels(Path(path_str))
    src_h, src_w = source.shape

    # Both source and target are equirectangular over the same domain, so a
    # target cell is exactly the union of the source pixels whose centres fall
    # inside it.  This holds for any H/W, integer ratio or not, which matters:
    # the tracked benchmark is 64x128 (1800/64 = 28.125) and the GUI grid is
    # 512x1024 (1800/512 = 3.515625).
    rows = np.minimum(
        ((np.arange(src_h) + 0.5) * height / src_h).astype(np.int64), height - 1
    )
    cols = np.minimum(
        ((np.arange(src_w) + 0.5) * width / src_w).astype(np.int64), width - 1
    )
    cell = (rows[:, None] * width + cols[None, :]).ravel()

    flat = cell * N_KOPPEN_CODES + source.ravel().astype(np.int64)
    counts = np.bincount(
        flat, minlength=height * width * N_KOPPEN_CODES
    ).reshape(height, width, N_KOPPEN_CODES)

    total = counts.sum(axis=2)
    land_counts = counts[..., 1:]
    land_total = land_counts.sum(axis=2)
    # Majority vote among *land* pixels only: a cell that is 70% ocean but has a
    # real coastal strip should still be scored against that strip's climate,
    # since the model classifies the whole cell as land whenever its DEM says so.
    codes = np.where(land_total > 0, land_counts.argmax(axis=2) + 1, KOPPEN_OCEAN)
    land_fraction = np.divide(
        land_total, np.maximum(total, 1), dtype=np.float64
    ).astype(np.float32)
    return codes.astype(np.int8), land_fraction


def load_reference_grid(
    height: int,
    width: int,
    *,
    path: str | Path = DEFAULT_REFERENCE_PATH,
) -> ReferenceGrid:
    """Regrid the bundled Köppen reference onto an ``height x width`` model grid."""
    if height < 8 or width != 2 * height:
        raise ValueError(
            f"target grid must be 2:1 equirectangular and at least 8x16, "
            f"got {width}x{height}"
        )
    codes, land_fraction = _reference_grid_cached(
        int(height), int(width), str(Path(path))
    )
    # Hand out copies: the cache holds the canonical arrays and callers should
    # not be able to mutate them for everyone else.
    return ReferenceGrid(codes=codes.copy(), land_fraction=land_fraction.copy())


# ---------------------------------------------------------------------------
# Skill scoring
# ---------------------------------------------------------------------------


def _area_weights(height: int, width: int) -> np.ndarray:
    """cos(latitude) cell-area weights.

    Every share/score in this module is area-weighted.  The audit's process
    note 8 exists because a missing cos(lat) on an equirectangular grid inflated
    polar land from ~16% to 38% and drove several sessions of misdirected
    physics work -- a per-cell skill score has exactly the same exposure.
    """
    lat = (0.5 - (np.arange(height, dtype=np.float64) + 0.5) / height) * np.pi
    return np.repeat(np.cos(lat)[:, None], width, axis=1)


def _weighted_shares(
    labels: np.ndarray, weights: np.ndarray, n_labels: int
) -> np.ndarray:
    totals = np.bincount(labels, weights=weights, minlength=n_labels)
    grand = totals.sum()
    return totals / grand if grand > 0 else totals


def _cohen_kappa(confusion: np.ndarray) -> float:
    """Chance-corrected agreement.

    Necessary rather than decorative: land is dominated by a few big classes, so
    a degenerate model that painted everything B would still score a respectable
    raw accuracy.  Kappa removes that floor (0 = chance, 1 = perfect).
    """
    total = confusion.sum()
    if total <= 0:
        return float("nan")
    observed = np.trace(confusion) / total
    expected = float(
        np.sum(confusion.sum(axis=0) * confusion.sum(axis=1)) / (total * total)
    )
    if np.isclose(expected, 1.0):
        return float("nan")
    return float((observed - expected) / (1.0 - expected))


def _per_label_scores(
    confusion: np.ndarray, labels: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    total = confusion.sum()
    for index, name in enumerate(labels):
        predicted = confusion[:, index].sum()   # model said this
        actual = confusion[index, :].sum()      # reference says this
        hit = confusion[index, index]
        precision = float(hit / predicted) if predicted > 0 else float("nan")
        recall = float(hit / actual) if actual > 0 else float("nan")
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = float("nan")
        out[name] = {
            "model_share_pct": float(100.0 * predicted / total) if total else float("nan"),
            "reference_share_pct": float(100.0 * actual / total) if total else float("nan"),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return out


def score_koppen_map(
    model_codes: np.ndarray,
    *,
    land_mask: np.ndarray | None = None,
    min_reference_land_fraction: float = 0.0,
    reference: ReferenceGrid | None = None,
    reference_path: str | Path = DEFAULT_REFERENCE_PATH,
    regions: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    """Score a model Köppen field against the gridded reference map.

    Scoring is restricted to cells that are land in *both* the model and the
    reference.  The model's coastline comes from its DEM and the reference's
    from a 0.1 deg product, so they disagree on a real fringe of cells; those are
    excluded and reported as ``coastline_mismatch`` rather than silently counted
    as classification errors.

    ``min_reference_land_fraction`` sets how much of a cell must be land in the
    reference before it is scored (0.5 = majority-land cells only).  The default
    of 0.0 scores every cell with any reference land at all, which keeps 98.4% of
    the model's land in view on the tracked 64x128 grid.  **The headline group
    score is insensitive to this choice** -- measured on that grid, group accuracy
    varies only 0.687-0.694 and kappa 0.596-0.611 across thresholds 0.0 to 0.75,
    while the fraction of model land excluded rises from 1.6% to 33.9%.  Class
    accuracy does move (0.388 -> 0.433), because the cells a high threshold drops
    are coastal ones whose exact class is hardest to get right; 0.0 is therefore
    the conservative reading as well as the most complete one.
    """
    codes = np.asarray(model_codes)
    if codes.ndim != 2:
        raise ValueError(f"model_codes must be 2-D, got shape {codes.shape}")
    height, width = codes.shape
    if reference is None:
        reference = load_reference_grid(height, width, path=reference_path)
    if reference.shape != codes.shape:
        raise ValueError(
            f"reference grid {reference.shape} does not match model grid {codes.shape}"
        )

    weights = _area_weights(height, width)
    # The ``!= KOPPEN_OCEAN`` term is not redundant with the threshold: a
    # threshold of 0.0 is a legitimate request ("score every cell with any land
    # at all") and would otherwise admit pure-ocean cells, which carry no
    # reference class.
    reference_land = (reference.land_fraction >= float(min_reference_land_fraction)) & (
        reference.codes != KOPPEN_OCEAN
    )
    model_land = (
        np.asarray(land_mask, dtype=bool)
        if land_mask is not None
        else codes != KOPPEN_OCEAN
    )
    model_land = model_land & (codes != KOPPEN_OCEAN)
    scored = reference_land & model_land

    total_land_weight = float(weights[reference_land | model_land].sum())
    result: dict[str, Any] = {
        "grid": {"height": height, "width": width},
        "min_reference_land_fraction": float(min_reference_land_fraction),
        "coastline_mismatch": {
            "model_land_not_reference_pct": (
                100.0 * float(weights[model_land & ~reference_land].sum())
                / total_land_weight
                if total_land_weight > 0 else float("nan")
            ),
            "reference_land_not_model_pct": (
                100.0 * float(weights[reference_land & ~model_land].sum())
                / total_land_weight
                if total_land_weight > 0 else float("nan")
            ),
        },
        "scored_cells": int(np.count_nonzero(scored)),
    }
    if not np.any(scored):
        result["error"] = "no cells are land in both the model and the reference"
        return result

    weight_values = weights[scored]
    model_flat = codes[scored].astype(np.int64)
    reference_flat = reference.codes[scored].astype(np.int64)

    # --- group level (A/B/C/D/E): immune to the vocabulary folding above ---
    model_groups = koppen_group(model_flat)
    reference_groups = koppen_group(reference_flat)
    group_confusion = np.bincount(
        reference_groups * 6 + model_groups, weights=weight_values, minlength=36
    ).reshape(6, 6)[1:, 1:]  # drop the "none" row/column
    group_labels = GROUP_LABELS[1:]

    group_shares_model = _weighted_shares(model_groups - 1, weight_values, 5)
    group_shares_reference = _weighted_shares(reference_groups - 1, weight_values, 5)

    result["group"] = {
        "accuracy": float(np.trace(group_confusion) / group_confusion.sum()),
        "kappa": _cohen_kappa(group_confusion),
        "share_mae_pp": float(
            np.mean(np.abs(group_shares_model - group_shares_reference)) * 100.0
        ),
        "per_class": _per_label_scores(group_confusion, group_labels),
        "confusion": {
            "labels": list(group_labels),
            "rows_are_reference": True,
            "matrix": group_confusion.tolist(),
        },
    }

    # --- class level (Af/BWh/Dfb/...): depends on REFERENCE_TO_MODEL_CODE ---
    class_confusion = np.bincount(
        reference_flat * N_KOPPEN_CODES + model_flat,
        weights=weight_values,
        minlength=N_KOPPEN_CODES * N_KOPPEN_CODES,
    ).reshape(N_KOPPEN_CODES, N_KOPPEN_CODES)[1:, 1:]
    class_labels = tuple(short_class_name(code) for code in range(1, N_KOPPEN_CODES))
    result["class"] = {
        "accuracy": float(np.trace(class_confusion) / class_confusion.sum()),
        "kappa": _cohen_kappa(class_confusion),
        "per_class": _per_label_scores(class_confusion, class_labels),
    }

    # --- where the errors are: zonal and per named region ---
    correct = np.zeros((height, width), dtype=bool)
    correct[scored] = model_groups == reference_groups
    lat = (0.5 - (np.arange(height, dtype=np.float64) + 0.5) / height) * 180.0
    bands: dict[str, float] = {}
    for low in range(-90, 90, 10):
        rows = (lat >= low) & (lat < low + 10)
        band = scored & rows[:, None]
        if not np.any(band):
            continue
        bands[f"{low}:{low + 10}"] = float(
            weights[band & correct].sum() / weights[band].sum()
        )
    result["group_accuracy_by_zone"] = bands

    if regions is None:
        from regional_validation import EARTH_PRECIP_REGIONS

        regions = EARTH_PRECIP_REGIONS
    from regional_validation import region_mask

    per_region: dict[str, Any] = {}
    for region in regions:
        selected = region_mask((height, width), region, cell_mask=scored)
        if not np.any(selected):
            per_region[region.name] = None
            continue
        region_weight = weights[selected]
        model_here = codes[selected].astype(np.int64)
        reference_here = reference.codes[selected].astype(np.int64)
        dominant_model = int(
            np.bincount(model_here, weights=region_weight,
                        minlength=N_KOPPEN_CODES).argmax()
        )
        dominant_reference = int(
            np.bincount(reference_here, weights=region_weight,
                        minlength=N_KOPPEN_CODES).argmax()
        )
        per_region[region.name] = {
            "group_accuracy": float(
                weights[selected & correct].sum() / region_weight.sum()
            ),
            "class_accuracy": float(
                region_weight[model_here == reference_here].sum()
                / region_weight.sum()
            ),
            "model_dominant": short_class_name(dominant_model),
            "reference_dominant": short_class_name(dominant_reference),
        }
    result["per_region"] = per_region
    return result


def earth_group_shares(
    height: int = 360,
    width: int = 720,
    *,
    min_reference_land_fraction: float = 0.5,
    path: str | Path = DEFAULT_REFERENCE_PATH,
) -> dict[str, float]:
    """Area-weighted A/B/C/D/E land shares of the *reference map itself*.

    Supersedes the hand-entered "Earth 19.0/26.4/13.4/24.6/16.6" constants that
    audit item A2 compares against -- those came from general knowledge, whereas
    these are measured from the designated ground-truth product on the same
    area-weighted basis the model is scored on.
    """
    reference = load_reference_grid(height, width, path=path)
    land = reference.land_fraction >= float(min_reference_land_fraction)
    weights = _area_weights(height, width)[land]
    groups = koppen_group(reference.codes[land].astype(np.int64))
    shares = _weighted_shares(groups - 1, weights, 5) * 100.0
    return {label: float(value) for label, value in zip(GROUP_LABELS[1:], shares)}
