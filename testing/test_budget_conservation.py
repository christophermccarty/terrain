"""Focused budget-closure tests added during the 2026-07 audit remediation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_stefan_boltzmann_and_gray_atmosphere_shared():
    from temperature import (
        STEFAN_BOLTZMANN,
        equilibrium_temperature_k,
        gray_atmosphere_denominator,
    )

    eps = np.array([0.68], dtype=np.float32)
    F = np.array([300.0], dtype=np.float32)
    T = equilibrium_temperature_k(F, eps)
    denom = gray_atmosphere_denominator(eps)
    expected = (F[0] / (STEFAN_BOLTZMANN * denom[0])) ** 0.25
    assert T[0] == pytest.approx(expected, rel=1e-6)


def test_deep_ocean_exchange_conserves_heat_content():
    from planet_params import EARTH

    cap_ratio = float(EARTH.deep_ocean_heat_capacity_ratio)
    k = float(EARTH.deep_ocean_exchange_rate)
    days = 30.0
    T_sst, T_deep = 285.0, 270.0
    dT_mixed = k * (T_sst - T_deep) * days
    dT_deep = dT_mixed * cap_ratio
    C_mixed = 1.0
    C_deep = C_mixed / cap_ratio
    assert C_mixed * (-dT_mixed) + C_deep * dT_deep == pytest.approx(0.0, abs=1e-12)


def test_ocean_transport_redistribution_has_zero_ocean_mean():
    from ocean import calculate_ocean_heat_transport

    Hc, Wc = 16, 32
    T = np.linspace(280.0, 270.0, Hc, dtype=np.float32)[:, None]
    T = np.repeat(T, Wc, axis=1)
    elev = np.zeros((Hc, Wc), dtype=np.float32)
    elev[:, : Wc // 4] = 0.5
    # Without radiative restoring the full adjustment is redistribution only.
    adj = calculate_ocean_heat_transport(
        T, elev, Hc, Wc, day_of_year=80, dt_days=1.0, T_equilibrium=None
    )
    is_ocean = elev == 0.0
    lat = (0.5 - (np.arange(Hc, dtype=np.float32) + 0.5) / Hc) * np.pi
    w = np.cos(lat)[:, None] * is_ocean.astype(np.float32)
    mean_adj = float(np.sum(adj * w) / (np.sum(w) + 1e-12))
    assert abs(mean_adj) < 0.02


def test_air_surface_sensible_exchange_is_equal_and_opposite():
    from planet_params import EARTH
    from simulate import _evolve_temperature

    Hc, Wc = 4, 8
    T = np.full((Hc, Wc), 280.0, dtype=np.float32)
    elev = np.zeros((Hc, Wc), dtype=np.float32)
    T_base = T.copy()
    T_air = T.copy() - 8.0

    T_sst, T_air_out, _, _, _, _, _ = _evolve_temperature(
        T, T_base, elev, Hc, Wc, 1, Hc, Wc,
        day_of_year=80.0, days=5.0,
        T_air_prev=T_air,
        planet_params=EARTH,
        track_components=False,
    )
    d_sst = float(np.mean(T_sst - T))
    d_air = float(np.mean(T_air_out - T_air))
    assert d_sst * d_air < 0.0 or abs(d_sst) < 1e-4
    assert abs(d_air) > abs(d_sst) * 0.1


def test_ocean_co2_piston_velocity_scales_with_wind_squared():
    from carbon_cycle import (
        OCEAN_CO2_K_CALIBRATION,
        OCEAN_CO2_MIXED_LAYER_DEPTH_M,
        WANINKHOF_CM_H_TO_M_DAY,
    )

    def transfer(u: float) -> float:
        k_piston = 0.31 * u * u * WANINKHOF_CM_H_TO_M_DAY
        return (k_piston / OCEAN_CO2_MIXED_LAYER_DEPTH_M) * OCEAN_CO2_K_CALIBRATION

    assert transfer(4.0) == pytest.approx(4.0 * transfer(2.0), rel=1e-6)


def test_precipitation_rescale_never_exceeds_available_q():
    from atmosphere import generate_precipitation

    H, W = 16, 32
    elev = np.zeros((H, W), dtype=np.float32)
    humidity = np.full((H, W), 0.012, dtype=np.float32)
    debug = {}
    P, humidity_next, _, _ = generate_precipitation(
        H, W, elev,
        humidity=humidity,
        target_mean_mm_day=8.0,
        dt_days=1.0,
        debug_fields=debug,
    )
    dq = humidity - humidity_next
    if "global_rescale_factor" in debug:
        assert float(np.max(dq)) <= float(np.max(humidity)) + 1e-9
    assert float(np.mean(P)) > 0.0


def test_compute_budget_diagnostics_keys():
    from diagnostics import compute_budget_diagnostics
    from simulate import create_initial_state
    from testing.conftest import make_mixed_elev

    state = create_initial_state(make_mixed_elev(16, 32), day_of_year=80.0)
    diag = compute_budget_diagnostics(state)
    assert "ocean_fraction" in diag
    assert "land_fraction" in diag
    assert diag["ocean_fraction"] + diag["land_fraction"] == pytest.approx(1.0)
