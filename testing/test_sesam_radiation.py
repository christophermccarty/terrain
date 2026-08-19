"""Tests for SESAM stage P5 sub-deliverable 1 -- cloud scheme (`sesam_radiation.py`).

Equations transcribed from Appendix A6 of Willeit et al. (2022) (GMD 15,
5905-5948), verified directly against the article PDF (2026-08-19; see the
module docstring). These tests guard: (A62) humidity/vertical-velocity cloud
fraction, (A63)-(A64) effective cloud-level vertical velocity, (A65)-(A66)
inversion/low-cloud fraction and its freeze-dry factor (including the clipped-
ramp behavior that replaces the paper's "when r*>ra" prose), (A61) the total-
cloud-fraction combination, (A67) cloud top height, and (A68) cloud optical
thickness.

The supported climate path never calls this module
(``PlanetParams.enable_sesam_radiation`` is False), so these tests exercise a
default-off diagnostic only.
"""
from __future__ import annotations

import numpy as np
import pytest

import sesam_radiation as sr
from planet_params import PlanetParams

A6 = sr._a6_defaults()


def test_sesam_radiation_gate_defaults_off():
    assert PlanetParams().enable_sesam_radiation is False


# ---------------------------------------------------------------------------
# (A63)-(A64) effective cloud-level vertical velocity
# ---------------------------------------------------------------------------


def test_orographic_vertical_velocity_hand_value():
    us = np.array([[10.0]])
    sigma = np.array([[100.0]])
    woro = sr.orographic_vertical_velocity(us, sigma)
    assert woro == pytest.approx(A6["c_woro"] * 10.0 * 100.0)


def test_orographic_vertical_velocity_shape_mismatch_raises():
    with pytest.raises(ValueError):
        sr.orographic_vertical_velocity(np.zeros((2, 2)), np.zeros((2, 3)))


def test_effective_cloud_vertical_velocity_hand_value():
    w700 = np.array([[0.01]])
    wsyn = np.array([[0.02]])
    woro = np.array([[0.01]])
    weff = sr.effective_cloud_vertical_velocity(w700, wsyn, woro)
    expected = 0.01 + A6["c_weff"] * (0.02 + 0.01)
    assert weff == pytest.approx(expected)


def test_effective_cloud_vertical_velocity_zero_synoptic_and_oro_is_identity():
    w700 = np.array([[0.037]])
    weff = sr.effective_cloud_vertical_velocity(w700, np.zeros((1, 1)), np.zeros((1, 1)))
    assert weff == pytest.approx(0.037)


# ---------------------------------------------------------------------------
# (A62) humidity/vertical-velocity cloud fraction
# ---------------------------------------------------------------------------


def test_humidity_cloud_fraction_hand_value_zero_weff():
    ra = np.array([[0.8]])
    f = sr.humidity_cloud_fraction(ra, np.zeros((1, 1)))
    expected = A6["c1cld"] * 0.8 ** A6["c4cld"]
    assert f == pytest.approx(expected, rel=1e-9)


def test_humidity_cloud_fraction_hand_value_nonzero_weff():
    ra = np.array([[0.8]])
    weff = np.array([[0.005]])
    f = sr.humidity_cloud_fraction(ra, weff)
    envelope = A6["c1cld"] + A6["c2cld"] * np.tanh(A6["c3cld"] * 0.005)
    expected = envelope * 0.8 ** A6["c4cld"]
    assert f == pytest.approx(expected, rel=1e-9)


def test_humidity_cloud_fraction_increases_with_updraft():
    """Physical sign check: rising motion (positive weff) must not suppress cloud."""
    ra = np.full((3, 3), 0.6)
    f_calm = sr.humidity_cloud_fraction(ra, np.zeros((3, 3)))
    f_rising = sr.humidity_cloud_fraction(ra, np.full((3, 3), 0.02))
    assert np.all(f_rising >= f_calm)


def test_humidity_cloud_fraction_clips_rh_outside_unit_interval():
    f_over = sr.humidity_cloud_fraction(np.array([[1.5]]), np.zeros((1, 1)))
    f_at_one = sr.humidity_cloud_fraction(np.array([[1.0]]), np.zeros((1, 1)))
    assert f_over == pytest.approx(f_at_one)


# ---------------------------------------------------------------------------
# (A65)-(A66) inversion/low-cloud fraction
# ---------------------------------------------------------------------------


def test_freezedry_factor_hand_values():
    assert sr.freezedry_factor(np.array([[0.0]])) == pytest.approx(0.1)
    assert sr.freezedry_factor(np.array([[A6["c7cld"]]])) == pytest.approx(1.0)
    # Saturates at 1.0 well before qa reaches c7cld * 10.
    assert sr.freezedry_factor(np.array([[A6["c7cld"] * 10.0]])) == pytest.approx(1.0)


def test_inversion_low_cloud_fraction_hand_value_at_crossover():
    """r* == ra (dr=0) sits at the *midpoint* of the ramp, not zero -- see
    module docstring note 2: the paper's "when r*>ra" is descriptive."""
    ra = np.array([[0.5]])
    rstar = np.array([[0.5]])
    qa = np.array([[A6["c7cld"]]])  # f_freezedry == 1.0
    f_low = sr.inversion_low_cloud_fraction(ra, rstar, qa)
    expected = A6["c5cld"] * 0.5 * 0.5 ** A6["c4cld"]
    assert f_low == pytest.approx(expected, rel=1e-9)
    assert f_low > 0.0


def test_inversion_low_cloud_fraction_ramp_is_clipped_not_unbounded():
    """Planted-violation guard: an unclipped (dr+c6)/(2c6) would give a
    materially different (larger) value once r*-ra exceeds c6cld. This
    confirms the clip in the implementation is actually active."""
    ra = np.array([[0.5]])
    rstar = np.array([[0.9]])  # dr = 0.4, clipped to c6cld = 0.1
    qa = np.array([[A6["c7cld"]]])
    f_low = sr.inversion_low_cloud_fraction(ra, rstar, qa)

    c6 = A6["c6cld"]
    clipped_fr = 1.0 * (c6 + c6) / (2.0 * c6)  # == 1.0, the ramp's ceiling
    unclipped_fr_if_bug = 1.0 * (0.4 + c6) / (2.0 * c6)  # == 2.5, would be a bug
    expected = A6["c5cld"] * clipped_fr * 0.5 ** A6["c4cld"]

    assert f_low == pytest.approx(expected, rel=1e-9)
    assert unclipped_fr_if_bug > clipped_fr  # sanity: the two really do differ
    assert f_low < A6["c5cld"] * unclipped_fr_if_bug * 0.5 ** A6["c4cld"]


def test_inversion_low_cloud_fraction_symmetric_clip_floors_at_zero_fr():
    """Strongly stable case (r* << ra) clips to the ramp's floor, fr == 0."""
    ra = np.array([[0.9]])
    rstar = np.array([[0.1]])  # dr = -0.8, clipped to -c6cld
    qa = np.array([[A6["c7cld"]]])
    f_low = sr.inversion_low_cloud_fraction(ra, rstar, qa)
    assert f_low == pytest.approx(0.0, abs=1e-12)


def test_inversion_low_cloud_fraction_shape_mismatch_raises():
    with pytest.raises(ValueError):
        sr.inversion_low_cloud_fraction(
            np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 3))
        )


# ---------------------------------------------------------------------------
# (A61) total cloud fraction
# ---------------------------------------------------------------------------


def test_total_cloud_fraction_hand_value():
    fcld = sr.total_cloud_fraction(np.array([[0.3]]), np.array([[0.2]]))
    assert fcld == pytest.approx(1.0 - 0.7 * 0.8)


def test_total_cloud_fraction_identity_with_zero_low_cloud():
    f_r = np.array([[0.37]])
    fcld = sr.total_cloud_fraction(f_r, np.zeros((1, 1)))
    assert fcld == pytest.approx(0.37)


def test_total_cloud_fraction_identity_with_zero_humidity_cloud():
    f_low = np.array([[0.22]])
    fcld = sr.total_cloud_fraction(np.zeros((1, 1)), f_low)
    assert fcld == pytest.approx(0.22)


def test_total_cloud_fraction_stays_in_unit_interval_for_out_of_range_inputs():
    """Bounds-sanity check (per the P4 lesson: unit-conflation/out-of-range
    upstream bugs are caught by bounds tests, not hand-value tests alone)."""
    fcld = sr.total_cloud_fraction(np.array([[-0.5]]), np.array([[-0.5]]))
    assert 0.0 <= float(fcld[0, 0]) <= 1.0


def test_total_cloud_fraction_never_exceeds_unit_interval_over_grid_sweep():
    rng = np.random.default_rng(0)
    f_r = rng.uniform(0.0, 1.0, size=(20, 20))
    f_low = rng.uniform(0.0, 1.0, size=(20, 20))
    fcld = sr.total_cloud_fraction(f_r, f_low)
    assert np.all(fcld >= 0.0) and np.all(fcld <= 1.0)


# ---------------------------------------------------------------------------
# (A67) cloud top height
# ---------------------------------------------------------------------------


def test_cloud_top_height_hand_value_zero_w700():
    ht = np.array([[12000.0]])
    hcld = sr.cloud_top_height_m(ht, np.zeros((1, 1)))
    expected = A6["c1hcld"] + A6["c2hcld"] * 12000.0
    assert hcld == pytest.approx(expected)


def test_cloud_top_height_hand_value_nonzero_w700():
    ht = np.array([[12000.0]])
    w700 = np.array([[0.01]])
    hcld = sr.cloud_top_height_m(ht, w700)
    expected = A6["c1hcld"] + A6["c2hcld"] * 12000.0 * (1.0 + A6["c3hcld"] * 0.01)
    assert hcld == pytest.approx(expected)


# ---------------------------------------------------------------------------
# (A68) cloud optical thickness
# ---------------------------------------------------------------------------


def test_cloud_optical_thickness_hand_value_at_threshold_temperature():
    t2m = np.array([[273.15 + A6["c1tau"]]])  # tcldm == 0 -> ftemp == 1
    fcld = np.array([[0.5]])
    qq = np.array([[4.0]])
    tau = sr.cloud_optical_thickness(t2m, fcld, qq)
    expected = A6["c3tau"] * 1.0 * (0.5 * 4.0) ** A6["c4tau"]
    assert tau == pytest.approx(expected, rel=1e-9)


def test_cloud_optical_thickness_ftemp_capped_at_one_for_very_cold_air():
    """Without the cap, 1+tanh(...) can approach 2 for very cold T2m -- the
    reference implementation caps ftemp at 1; guard that this cap is active."""
    t2m = np.array([[150.0]])  # far below T0 + c1tau
    fcld = np.array([[0.5]])
    qq = np.array([[4.0]])
    tau = sr.cloud_optical_thickness(t2m, fcld, qq)
    capped_expected = A6["c3tau"] * 1.0 * (0.5 * 4.0) ** A6["c4tau"]
    uncapped_would_be = A6["c3tau"] * 2.0 * (0.5 * 4.0) ** A6["c4tau"]
    assert tau == pytest.approx(capped_expected, rel=1e-6)
    assert tau < uncapped_would_be


def test_cloud_optical_thickness_nonnegative_over_grid_sweep():
    rng = np.random.default_rng(1)
    t2m = rng.uniform(180.0, 320.0, size=(15, 15))
    fcld = rng.uniform(0.0, 1.0, size=(15, 15))
    qq = rng.uniform(0.0, 60.0, size=(15, 15))
    tau = sr.cloud_optical_thickness(t2m, fcld, qq)
    assert np.all(np.isfinite(tau))
    assert np.all(tau >= 0.0)


def test_cloud_optical_thickness_shape_mismatch_raises():
    with pytest.raises(ValueError):
        sr.cloud_optical_thickness(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 3)))
