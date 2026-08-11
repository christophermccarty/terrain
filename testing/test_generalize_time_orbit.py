from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from climate_averages import update_climate_averages, update_monthly_statistics
from planet_params import EARTH, MARS, PlanetParams
from temperature import temperature_kelvin_for_lat


@pytest.mark.parametrize("pp", [EARTH, MARS])
def test_planet_relative_equinox_and_solstice(pp):
    assert pp.solar_declination(pp.vernal_equinox_day) == pytest.approx(0.0, abs=1e-12)
    northern_solstice = pp.vernal_equinox_day + pp.orbital_period_days / 4.0
    assert pp.solar_declination(northern_solstice) == pytest.approx(
        pp.obliquity_rad, abs=1e-12
    )


def test_fractional_days_are_preserved_in_seasonal_temperature():
    lat = np.array([np.deg2rad(45.0)], dtype=np.float32)
    t0 = temperature_kelvin_for_lat(
        lat, EARTH.vernal_equinox_day, planet_params=EARTH, cache=True
    )
    t1 = temperature_kelvin_for_lat(
        lat, EARTH.vernal_equinox_day + 0.5, planet_params=EARTH, cache=True
    )
    assert not np.array_equal(t0, t1)


@pytest.mark.parametrize("pp", [EARTH, MARS])
def test_kepler_solver_hits_perihelion_and_aphelion(pp):
    assert pp.solar_distance_factor(pp.perihelion_day) == pytest.approx(
        1.0 - pp.eccentricity, rel=0.0, abs=1e-12
    )
    aphelion = pp.perihelion_day + pp.orbital_period_days / 2.0
    assert pp.solar_distance_factor(aphelion) == pytest.approx(
        1.0 + pp.eccentricity, rel=0.0, abs=1e-12
    )


def test_orbital_period_controls_month_bins_and_ema_windows():
    period = MARS.orbital_period_days
    state = SimpleNamespace(
        temperature=np.full((2, 2), 250.0, dtype=np.float32),
        precipitation=np.full((2, 2), 1.0, dtype=np.float32),
        climate_temp_avg=np.full((2, 2), 200.0, dtype=np.float32),
        climate_precip_avg=np.zeros((2, 2), dtype=np.float32),
        climate_sample_days=0.0,
        day_of_year=period * 6.5 / 12.0,
        monthly_temp=None,
        monthly_precip=None,
        monthly_sample_count=None,
    )
    temp_avg, _, _ = update_climate_averages(
        state, period, orbital_period_days=period, window_years=10.0
    )
    assert temp_avg[0, 0] == pytest.approx(205.0)

    _, _, counts = update_monthly_statistics(
        state, 1.0, orbital_period_days=period
    )
    assert counts[6] == pytest.approx(1.0)
    assert np.count_nonzero(counts) == 1


def test_reference_air_density_is_planet_derived():
    assert EARTH.reference_air_density == pytest.approx(1.225, rel=2e-3)
    assert MARS.reference_air_density == pytest.approx(
        MARS.surface_pressure_pa / (MARS.gas_constant_dry * 288.15)
    )
    assert MARS.reference_air_density < 0.02


def test_mars_pgf_uses_thin_atmosphere_density():
    from atmosphere import evolve_wind

    shape = (8, 16)
    u0 = np.zeros(shape, dtype=np.float32)
    v0 = np.zeros(shape, dtype=np.float32)
    temperature = np.repeat(
        np.linspace(240.0, 300.0, shape[0], dtype=np.float32)[:, None],
        shape[1],
        axis=1,
    )
    elevation = np.zeros(shape, dtype=np.float32)

    def speed(pp: PlanetParams) -> float:
        u, v = evolve_wind(
            u0,
            v0,
            temperature,
            pressure=None,
            elevation=elevation,
            dt_days=1e-5,
            damping=1.0,
            drag_base=0.0,
            drag_elev_scale=0.0,
            cell_relax_days=0.0,
            planet_params=pp,
        )
        return float(np.mean(np.hypot(u, v)))

    assert speed(MARS) > 20.0 * speed(EARTH)


def test_temperature_substeps_rebuild_bases_at_each_inner_date(monkeypatch):
    import simulate

    seen = []

    def fake_evolve(T_prev, T_base, *args, day_of_year, T_base_land, **kwargs):
        seen.append((day_of_year, float(T_base[0, 0]), float(T_base_land[0, 0])))
        zeros = np.zeros_like(T_prev)
        return T_prev, T_prev, zeros, zeros, {}, None, None

    monkeypatch.setattr(simulate, "_evolve_temperature", fake_evolve)
    field = np.zeros((1, 1), dtype=np.float32)

    def bases(day):
        return (
            np.full((1, 1), day, dtype=np.float32),
            np.full((1, 1), -day, dtype=np.float32),
        )

    simulate._evolve_temperature_substepped(
        field,
        field,
        field,
        1,
        1,
        1,
        1,
        1,
        day_of_year=16.0,
        days=6.0,
        substep_days=1.0,
        temperature_bases_for_day=bases,
        T_base_land=field,
    )
    assert seen == [(d, d, -d) for d in np.arange(11.0, 17.0)]


def test_precipitation_substeps_advance_fractional_dates(monkeypatch):
    import simulate

    seen = []

    def fake_precip(H, W, elev, *, day_of_year, humidity, soil_moisture,
                    soil_moisture_deep, condensate, precipitating_hydrometeors,
                    midlevel_temperature, midlevel_humidity,
                    upperlevel_temperature, upperlevel_humidity, **kwargs):
        seen.append(day_of_year)
        zeros = np.zeros((H, W), dtype=np.float32)
        # Order matches generate_precipitation's real return contract when
        # every return_* flag is set (see the unpack in
        # _generate_precipitation_substepped's substep loop).
        return (
            zeros, humidity, soil_moisture, soil_moisture_deep, condensate,
            midlevel_temperature, midlevel_humidity,
            upperlevel_temperature, upperlevel_humidity,
            precipitating_hydrometeors,
        )

    monkeypatch.setattr(simulate, "generate_precipitation", fake_precip)
    field = np.zeros((1, 1), dtype=np.float32)
    simulate._generate_precipitation_substepped(
        1,
        1,
        field,
        temperature=field,
        wind_u=field,
        wind_v=field,
        wind_u_aloft=field,
        wind_v_aloft=field,
        wind_u_midlevel=field,
        wind_v_midlevel=field,
        humidity=field,
        soil_moisture=field,
        soil_moisture_deep=field,
        condensate=field,
        precipitating_hydrometeors=field,
        midlevel_temperature=field,
        midlevel_humidity=field,
        upperlevel_temperature=field,
        upperlevel_humidity=field,
        cloud_fraction=field,
        day_of_year=16.0,
        dt_days=6.0,
        planet_params=EARTH,
    )
    assert seen == pytest.approx(np.arange(11.0, 17.0))


@pytest.mark.parametrize("pp", [EARTH, MARS])
def test_slow_mode_cycles_match_planet_orbit_exactly(pp):
    from simulate import TimeScaleMode
    from time_policy import cycle_days, substeps_for_mode

    annual = substeps_for_mode(TimeScaleMode.ANNUAL, pp)
    monthly = substeps_for_mode(TimeScaleMode.MONTHLY, pp)

    assert len(annual) == 52
    assert len(monthly) == 5
    assert cycle_days(TimeScaleMode.ANNUAL, pp) == pytest.approx(
        pp.orbital_period_days, rel=0.0, abs=1e-10
    )
    assert 12.0 * cycle_days(TimeScaleMode.MONTHLY, pp) == pytest.approx(
        pp.orbital_period_days, rel=0.0, abs=1e-10
    )
    assert all(not update_wind for _, update_wind in annual + monthly)


def test_daily_and_weekly_cycles_remain_literal_days():
    from simulate import TimeScaleMode
    from time_policy import cycle_days

    assert cycle_days(TimeScaleMode.DAILY, MARS) == pytest.approx(1.0)
    assert cycle_days(TimeScaleMode.WEEKLY, MARS) == pytest.approx(7.0)


def test_ocean_current_seasonality_uses_planet_orbit(monkeypatch):
    import ocean

    def constant_currents(H, W, **_kwargs):
        return (
            np.ones((H, W), dtype=np.float32),
            np.zeros((H, W), dtype=np.float32),
        )

    monkeypatch.setattr(ocean, "get_major_ocean_currents", constant_currents)
    elevation = np.zeros((4, 8), dtype=np.float32)

    earth_u, _ = ocean.generate_ocean_currents(
        elevation,
        day_of_year=EARTH.orbital_period_days / 4.0,
        time_days=0.0,
        orbital_period_days=EARTH.orbital_period_days,
    )
    mars_u, _ = ocean.generate_ocean_currents(
        elevation,
        day_of_year=MARS.orbital_period_days / 4.0,
        time_days=0.0,
        orbital_period_days=MARS.orbital_period_days,
    )

    np.testing.assert_allclose(earth_u, mars_u, atol=1e-6, rtol=0.0)
