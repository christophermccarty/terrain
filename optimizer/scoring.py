"""Climate scoring: compare simulation output against a reference climate.

ClimateScore computes a 0–100 score by independently evaluating each metric
against a target range. No single metric can dominate — each is capped at its
own weight and only contributes proportionally.

Usage
-----
from optimizer.scoring import ClimateScore, EARTH_REFERENCE, ClimateMetrics

score_fn = ClimateScore(EARTH_REFERENCE)
score = score_fn.score(metrics)          # float 0-100
breakdown = score_fn.breakdown(metrics)  # dict of per-metric contributions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# ClimateMetrics — output of headless.run_simulation
# ---------------------------------------------------------------------------

@dataclass
class ClimateMetrics:
    """Summary climate metrics extracted from a simulation evaluation period."""

    global_mean_t: float = 0.0
    """Global area-weighted mean surface temperature [K]."""

    gradient_nh: float = 0.0
    """Equator-to-NH-pole temperature gradient [K]."""

    gradient_sh: float = 0.0
    """Equator-to-SH-pole temperature gradient [K]."""

    ice_frac_nh: float = 0.0
    """Fraction of NH latitude rows with zonal-mean ice > 10% [0-1]."""

    ice_frac_sh: float = 0.0
    """Fraction of SH latitude rows with zonal-mean ice > 10% [0-1]."""

    mean_precip: float = 0.0
    """Global area-weighted mean precipitation [mm/day]."""

    wind_trade_mean: float = 0.0
    """Area-weighted mean wind speed in trade-wind band (5–20°) [m/s]."""

    wind_midlat_mean: float = 0.0
    """Area-weighted mean wind speed in mid-latitude band (30–60°) [m/s]."""

    wind_itcz_conv: float = 0.0
    """Equatorial ITCZ convergence proxy (positive = converging)."""

    seasonal_amplitude_nh: float = 0.0
    """Peak-to-peak NH mid-latitude (40–60°N) temperature range over 1 year [K]."""

    circulation_score: float = 0.0
    """Pre-computed diagnostics circulation score (from diagnostics.py)."""

    has_nan: bool = False
    """True if the simulation produced any NaN values."""

    has_inf: bool = False
    """True if the simulation produced any Inf values."""


# ---------------------------------------------------------------------------
# ReferenceClimate — configurable target
# ---------------------------------------------------------------------------

@dataclass
class ReferenceClimate:
    """Climate targets and scoring weights for one planet type.

    Each metric entry is a tuple (lo, hi, weight) where:
    - lo / hi define the target range (full score inside the range)
    - weight is the maximum contribution to the total score
    - penalty_width is the extra margin over which the score decays to zero
      (calculated as (hi - lo) * penalty_factor for range metrics, or
       explicit width for point targets)
    """

    global_mean_t: tuple[float, float, float] = (286.0, 290.0, 2.0)
    gradient_nh: tuple[float, float, float] = (40.0, 65.0, 1.5)
    gradient_sh: tuple[float, float, float] = (38.0, 62.0, 1.0)
    ice_frac_nh: tuple[float, float, float] = (0.02, 0.10, 1.5)
    ice_frac_sh: tuple[float, float, float] = (0.03, 0.12, 1.0)
    mean_precip: tuple[float, float, float] = (2.2, 3.2, 0.5)
    wind_trade_mean: tuple[float, float, float] = (4.0, 9.0, 1.0)
    wind_midlat_mean: tuple[float, float, float] = (5.0, 11.0, 1.0)
    wind_itcz_conv: tuple[float, float, float] = (0.02, 10.0, 1.0)
    seasonal_amplitude_nh: tuple[float, float, float] = (28.0, 55.0, 1.0)

    penalty_factor: float = 3.0
    """Width of the decay zone as a multiple of the target range width.
    Larger = more gradual penalty; smaller = sharper cliff."""


EARTH_REFERENCE = ReferenceClimate()
"""Earth-calibrated reference targets (default values of ReferenceClimate)."""


# ---------------------------------------------------------------------------
# ClimateScore
# ---------------------------------------------------------------------------

def _smooth_range(value: float, lo: float, hi: float, penalty_width: float) -> float:
    """Return 1.0 inside [lo, hi], linearly decaying to 0.0 at lo±penalty_width."""
    if lo <= value <= hi:
        return 1.0
    elif value < lo:
        return max(0.0, 1.0 - (lo - value) / (penalty_width + 1e-9))
    else:
        return max(0.0, 1.0 - (value - hi) / (penalty_width + 1e-9))


class ClimateScore:
    """Compute a 0–100 climate realism score.

    Each metric is scored independently and capped at its own weight so that
    no single metric dominates. If NaN or Inf is detected in the state, the
    overall score is 0.
    """

    def __init__(self, reference: ReferenceClimate = EARTH_REFERENCE) -> None:
        self.reference = reference
        self._metrics: list[tuple[str, float]] = self._build_metric_list()
        self._total_weight: float = sum(w for _, w in self._metrics)

    def _build_metric_list(self) -> list[tuple[str, float]]:
        ref = self.reference
        return [
            ("global_mean_t", ref.global_mean_t[2]),
            ("gradient_nh", ref.gradient_nh[2]),
            ("gradient_sh", ref.gradient_sh[2]),
            ("ice_frac_nh", ref.ice_frac_nh[2]),
            ("ice_frac_sh", ref.ice_frac_sh[2]),
            ("mean_precip", ref.mean_precip[2]),
            ("wind_trade_mean", ref.wind_trade_mean[2]),
            ("wind_midlat_mean", ref.wind_midlat_mean[2]),
            ("wind_itcz_conv", ref.wind_itcz_conv[2]),
            ("seasonal_amplitude_nh", ref.seasonal_amplitude_nh[2]),
        ]

    def _component(self, name: str, value: float) -> float:
        """Score for one metric: 0..weight."""
        ref = self.reference
        spec: tuple[float, float, float] = getattr(ref, name)
        lo, hi, weight = spec
        width = (hi - lo) * ref.penalty_factor
        return weight * _smooth_range(value, lo, hi, max(width, 1e-9))

    def breakdown(self, metrics: ClimateMetrics) -> dict[str, float]:
        """Per-metric contribution (0..weight for each), plus 'total' (0..100)."""
        if metrics.has_nan or metrics.has_inf:
            result = {name: 0.0 for name, _ in self._metrics}
            result["total"] = 0.0
            return result

        result: dict[str, float] = {}
        raw_sum = 0.0
        for name, _weight in self._metrics:
            val = getattr(metrics, name)
            c = self._component(name, float(val))
            result[name] = c
            raw_sum += c

        result["total"] = 100.0 * raw_sum / (self._total_weight + 1e-9)
        return result

    def score(self, metrics: ClimateMetrics) -> float:
        """Overall score in [0, 100]."""
        return self.breakdown(metrics)["total"]
