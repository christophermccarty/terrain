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

    cru_temp_correlation: float = 0.0
    """Area-weighted monthly land-temperature correlation vs CRU TS v4.10 [-1..1].
    Only populated when run_simulation is given monthly_climatology_path AND run on
    real (not synthetic) terrain -- see that function's docstring. 0.0 (its default)
    combined with ReferenceClimate's default weight=0.0 is an exact no-op."""

    cru_temp_rmse: float = 0.0
    """Area-weighted monthly land-temperature RMSE vs CRU TS v4.10 [K]. See
    cru_temp_correlation for when this is actually populated vs. left at its inert
    default."""

    cru_precip_log_correlation: float = 0.0
    """Area-weighted monthly land log-precipitation correlation vs CRU TS v4.10
    [-1..1]. Log-space matches monthly_climatology.score_monthly_climatology's own
    convention (arid-region errors stay measurable). See cru_temp_correlation for
    when this is actually populated."""

    cru_precip_log_rmse: float = 0.0
    """Area-weighted monthly land log-precipitation RMSE vs CRU TS v4.10. See
    cru_temp_correlation for when this is actually populated."""

    ncep_wind_correlation: float = 0.0
    """Global-area-weighted annual-mean wind-speed correlation vs NCEP/NCAR
    Reanalysis 1 [-1..1] (CRU publishes no wind variable at all -- a
    different provider). Global, not land-only (wind is meaningful over
    ocean too), and annual-mean-only on the model side, not true monthly --
    see run_simulation's monthly_climatology_path docstring and
    monthly_climatology.score_monthly_climatology's own docstring for why.
    0.0 default + ReferenceClimate's default weight=0.0 is an exact no-op."""

    ncep_wind_rmse: float = 0.0
    """Global-area-weighted annual-mean wind-speed RMSE vs NCEP/NCAR
    Reanalysis 1 [m/s]. See ncep_wind_correlation for when this is actually
    populated vs. left at its inert default."""

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

    # CRU TS v4.10 real-reanalysis map-correlation targets. Default weight 0.0:
    # an exact no-op for every existing sweep/config, matching this project's
    # standing convention for real-but-not-yet-promoted mechanisms (see e.g.
    # PlanetParams.moisture_advection_scale). Set weight > 0 (and pass
    # run_simulation a real elevation + monthly_climatology_path) to fold
    # real-terrain CRU accuracy into the score. Bounds carry headroom over the
    # measured current 64x128/1yr baseline (temp 6.28C RMSE/0.930 correlation;
    # precip 1.406 log-RMSE/0.463 log-correlation -- see
    # docs/MONTHLY_CLIMATOLOGY_REFERENCE.md), the same headroom convention as
    # testing/test_reanalysis_validation.py's CRU regression gates.
    cru_temp_correlation: tuple[float, float, float] = (0.88, 1.0, 0.0)
    cru_temp_rmse: tuple[float, float, float] = (0.0, 7.5, 0.0)
    cru_precip_log_correlation: tuple[float, float, float] = (0.35, 1.0, 0.0)
    cru_precip_log_rmse: tuple[float, float, float] = (0.0, 1.6, 0.0)

    # NCEP/NCAR Reanalysis 1 wind targets (CRU has no wind variable -- see
    # ClimateMetrics.ncep_wind_correlation). Global, annual-mean-only, and
    # loose by design: the model's wind is far cruder than the T/P physics,
    # so these bounds match testing/test_reanalysis_validation.py's
    # regression-gate looseness (catch a real pipeline break, not enforce
    # unreached realism), not the tighter headroom convention used above.
    ncep_wind_correlation: tuple[float, float, float] = (0.0, 1.0, 0.0)
    ncep_wind_rmse: tuple[float, float, float] = (0.0, 6.0, 0.0)

    penalty_factor: float = 3.0
    """Width of the decay zone as a multiple of the target range width.
    Larger = more gradual penalty; smaller = sharper cliff."""


EARTH_REFERENCE = ReferenceClimate()
"""Earth-calibrated reference targets (default values of ReferenceClimate)."""

WIND_SCREENING_REFERENCE = ReferenceClimate(
    global_mean_t=(286.0, 290.0, 0.0),
    gradient_nh=(40.0, 65.0, 0.0),
    gradient_sh=(38.0, 62.0, 0.0),
    ice_frac_nh=(0.02, 0.10, 0.0),
    ice_frac_sh=(0.03, 0.12, 0.0),
    mean_precip=(2.2, 3.2, 0.0),
    wind_trade_mean=(2.519, 3.987, 1.0),
    wind_midlat_mean=(3.196, 5.753, 1.0),
    wind_itcz_conv=(-0.001, 0.070, 1.0),
    seasonal_amplitude_nh=(28.0, 55.0, 0.0),
)
"""EARTH_REFERENCE, restricted and rebanded for optimizer/jax_screening.py's
GPU screening model -- NOT a general-purpose reference, and CPU-backend
sweeps have no need for it.

SUPERSEDED as gpu_random_search's default: once the screening model gained
a wind-speed-driven evaporative-cooling term (see jax_screening.py's
docstring), scoring against the plain EARTH_REFERENCE reached Spearman
0.708 against the real CPU model -- well above this reference's own 0.338
on the same 20-config validation set. gpu_random_search now defaults to
EARTH_REFERENCE. This reference is kept only as a documented alternative
(e.g. if a future change regresses the temperature coupling and the
full-reference score stops correlating well again), not because it
performs better today. History below, for that scenario:

Two problems needed fixing, found in that order:

1. Temperature/ice metrics carry a validated-WRONG signal. The screening
   model's wind mechanism matches the real CPU model well (Spearman
   0.985-0.995 on wind_trade_mean, optimizer/configs/sweep_wind.json's
   parameter family), but its temperature response to wind does not
   (global_mean_t -0.37, gradient_nh/sh -0.73/-0.74) -- so
   global_mean_t/gradient_nh/gradient_sh/ice_frac_nh/ice_frac_sh/
   seasonal_amplitude_nh are zeroed, same as this project's existing
   default-0.0-weight convention for the inert CRU/NCEP fields above.
2. EARTH_REFERENCE's [lo, hi] bands were sized for a 10-component score
   where ties get broken by the other 6 metrics; restricted to the 4
   wind/precip metrics left after (1), those wide bands gave a FLAT 1.0 to
   a large, correlated fraction of configs simultaneously (6 of 20
   validation configs landed on an exact tied ceiling of 100.0), which
   destroyed rank information even though the underlying raw metrics still
   correlated fine. Two fixes: mean_precip's weight is zeroed too -- its
   real-model values for this parameter family cluster in a ~0.05-wide
   band (2.99-3.04 mm/day) regardless of the swept params, i.e. it carries
   no discriminating signal here, unlike gradient/ice_frac's problem of
   carrying a wrong signal. wind_trade_mean/wind_midlat_mean/
   wind_itcz_conv's bands are narrowed to the interquartile range (25th-
   75th percentile) of the real CPU model's own output on 20 latin-
   hypercube-sampled sweep_wind.json configs -- a standard, non-tuned-to-
   maximize-one-metric statistical choice (narrower percentile windows
   scored marginally higher on that same 20-config sample, up to ~0.44,
   but turned the score increasingly binary/step-like doing it -- a sign
   of overfitting to a 20-point sample rather than a real improvement).

Net result on that 20-config validation set: overall-score Spearman
correlation against the CPU model went from -0.117 (worse than not
restricting the reference at all) to +0.337 -- a real fix for the outright
-broken ceiling-tie bug, but still short of a validated bar for pointing a
real sweep at this backend (wind_trade_mean alone independently reaches
0.985-0.995; the composite score doesn't get close to that). See project
memory gpu-sweep-screening-phase4-anticorrelated-2026-08-10 for the full
investigation and everything already tried."""


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
            ("cru_temp_correlation", ref.cru_temp_correlation[2]),
            ("cru_temp_rmse", ref.cru_temp_rmse[2]),
            ("cru_precip_log_correlation", ref.cru_precip_log_correlation[2]),
            ("cru_precip_log_rmse", ref.cru_precip_log_rmse[2]),
            ("ncep_wind_correlation", ref.ncep_wind_correlation[2]),
            ("ncep_wind_rmse", ref.ncep_wind_rmse[2]),
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
