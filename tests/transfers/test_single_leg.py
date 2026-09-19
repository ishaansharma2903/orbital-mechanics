"""
tests/transfers/test_single_leg.py

Unit and benchmark validation tests for single-leg patched-conic transfers.
"""
import pytest
import numpy as np

from orbital_mechanics.constants import (
    MU_SUN,
    MU_EARTH,
    MU_MARS,
    R_EARTH,
    R_MARS,
    AU,
)
from orbital_mechanics.core.state import State
from orbital_mechanics.transfers.single_leg import solve_single_leg, SingleLegTransfer


def test_solve_single_leg_mars_2020():
    """
    Validate Earth-to-Mars transfer for Mars 2020 Perseverance mission
    (Launch: 2020-07-30, Arrival: 2021-02-18).

    NASA published mission design parameters:
    - TOF: ~203 days
    - C3: ~14.5 km^2/s^2
    - v_inf arrival: ~2.5 - 2.6 km/s
    """
    dep_epoch = "2020-07-30"
    arr_epoch = "2021-02-18"

    transfer = solve_single_leg(
        departure_body="earth",
        arrival_body="mars",
        departure_epoch=dep_epoch,
        arrival_epoch=arr_epoch,
        prograde=True,
    )

    assert isinstance(transfer, SingleLegTransfer)
    assert abs(transfer.tof_days - 203.0) < 1.0

    # C3 and hyperbolic excess velocities
    assert 14.0 < transfer.c3 < 15.0
    assert 3.7 < transfer.v_inf_dep_mag < 3.9
    assert 2.4 < transfer.v_inf_arr_mag < 2.7

    # Validate delta-v calculations for circular parking orbits
    # LEO: 200 km altitude
    r_park_earth = R_EARTH + 200.0
    # Mars orbit: 250 km altitude
    r_park_mars = R_MARS + 250.0

    dv_dep = transfer.departure_delta_v(r_park=r_park_earth, mu_body=MU_EARTH)
    dv_arr = transfer.arrival_delta_v(r_park=r_park_mars, mu_body=MU_MARS)
    dv_total = transfer.total_delta_v(
        r_park_dep=r_park_earth,
        mu_dep=MU_EARTH,
        r_park_arr=r_park_mars,
        mu_arr=MU_MARS,
    )

    # Typical Earth escape burn from LEO is ~3.6 - 3.9 km/s
    assert 3.5 < dv_dep < 4.0
    # Mars orbit insertion burn is ~2.0 - 2.4 km/s
    assert 2.0 < dv_arr < 2.4
    assert abs(dv_total - (dv_dep + dv_arr)) < 1e-10


def test_solve_single_leg_energy_conservation():
    """
    Verify that the solved transfer orbit conserves energy and angular momentum
    between departure and arrival.
    """
    transfer = solve_single_leg(
        departure_body="earth",
        arrival_body="mars",
        departure_epoch="2020-07-30",
        arrival_epoch="2021-02-18",
    )

    oe_dep = transfer.departure_state_craft.to_orbit_elements(mu=MU_SUN)
    oe_arr = transfer.arrival_state_craft.to_orbit_elements(mu=MU_SUN)

    # Semi-major axis and eccentricity must be identical at departure and arrival
    assert abs((oe_dep.a - oe_arr.a) / oe_dep.a) < 1e-5
    assert abs(oe_dep.e - oe_arr.e) < 1e-5
    assert abs(oe_dep.i - oe_arr.i) < 1e-5


def test_solve_single_leg_invalid_dates():
    """Verify that arrival date before departure date raises ValueError."""
    with pytest.raises(ValueError, match="strictly after"):
        solve_single_leg(
            departure_body="earth",
            arrival_body="mars",
            departure_epoch="2021-02-18",
            arrival_epoch="2020-07-30",
        )


def test_solve_single_leg_custom_states():
    """Verify solve_single_leg when custom State objects are provided directly."""
    # Synthetic circular coplanar orbits (Earth ~ 1 AU, Mars ~ 1.5 AU)
    r1_dist = AU
    r2_dist = 1.5 * AU
    theta_arr = 2.0 * np.pi / 3.0  # 120 degrees

    r1 = np.array([r1_dist, 0.0, 0.0])
    v1 = np.array([0.0, np.sqrt(MU_SUN / r1_dist), 0.0])
    s1 = State(position=r1, velocity=v1)

    r2 = np.array([r2_dist * np.cos(theta_arr), r2_dist * np.sin(theta_arr), 0.0])
    v2 = np.array([-np.sqrt(MU_SUN / r2_dist) * np.sin(theta_arr), np.sqrt(MU_SUN / r2_dist) * np.cos(theta_arr), 0.0])
    s2 = State(position=r2, velocity=v2)

    tof_days = 150.0

    transfer = solve_single_leg(
        departure_body=s1,
        arrival_body=s2,
        departure_epoch=2451545.0,
        arrival_epoch=2451545.0 + tof_days,
        prograde=True,
    )

    assert transfer.departure_body == "custom"
    assert transfer.arrival_body == "custom"
    assert abs(transfer.tof_days - 150.0) < 1e-8
    np.testing.assert_allclose(transfer.departure_state_craft.position, r1)
    np.testing.assert_allclose(transfer.arrival_state_craft.position, r2)
    assert transfer.c3 > 0.0
    assert abs(transfer.c3 - transfer.v_inf_dep_mag ** 2) < 1e-10

    # Verify transfer orbit physical invariants
    oe_dep = transfer.departure_state_craft.to_orbit_elements(mu=MU_SUN)
    oe_arr = transfer.arrival_state_craft.to_orbit_elements(mu=MU_SUN)
    assert abs((oe_dep.a - oe_arr.a) / oe_dep.a) < 1e-5
    assert abs(oe_dep.e - oe_arr.e) < 1e-5
