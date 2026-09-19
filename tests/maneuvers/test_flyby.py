"""
tests/maneuvers/test_flyby.py

Unit and validation tests for hyperbolic gravity-assist (flyby) kinematics.
"""
import pytest
import numpy as np

from orbital_mechanics.constants import (
    MU_EARTH,
    MU_SUN,
    MU_VENUS,
    R_EARTH,
    R_VENUS,
    AU,
)
from orbital_mechanics.maneuvers.flyby import (
    FlybyResult,
    solve_flyby,
    compute_flyby_turn_angle,
    compute_max_turn_angle,
    compute_flyby_periapsis,
)


def test_curtis_example_8_3_jupiter_flyby():
    """
    Validate hyperbolic flyby kinematics against Curtis (4th ed.) Example 8.3:
    Spacecraft flyby of Jupiter with v_inf = 10.0 km/s and rp = 200,000 km.
    """
    mu_jup = 1.26686e8  # km^3/s^2
    r_jup = 71492.0     # km
    v_inf = 10.0        # km/s
    rp_expected = 200000.0

    e_expected = 1.0 + (rp_expected * v_inf ** 2) / mu_jup
    delta_expected = 2.0 * np.arcsin(1.0 / e_expected)

    # Compute periapsis from turn angle
    rp_calc = compute_flyby_periapsis(v_inf=v_inf, turn_angle=delta_expected, mu_body=mu_jup)
    assert abs(rp_calc - rp_expected) < 1e-6

    # Maximum turn angle at 1000 km safe altitude
    max_delta = compute_max_turn_angle(v_inf=v_inf, mu_body=mu_jup, r_body=r_jup, h_safe=1000.0)
    assert max_delta > delta_expected


def test_solve_flyby_unpowered_coplanar():
    """
    Validate solve_flyby with an unpowered deflection around Earth.
    """
    v_body = np.array([0.0, 29.78, 0.0])  # Earth heliocentric velocity ~29.78 km/s
    v_inf_mag = 4.0                       # 4 km/s excess speed

    # Inbound v_inf along +x
    v_inf_in = np.array([v_inf_mag, 0.0, 0.0])
    v_in = v_body + v_inf_in

    # Outbound v_inf turned by 60 degrees in xy-plane
    theta = np.radians(60.0)
    v_inf_out = np.array([v_inf_mag * np.cos(theta), v_inf_mag * np.sin(theta), 0.0])
    v_out = v_body + v_inf_out

    result = solve_flyby(
        v_in=v_in,
        v_out=v_out,
        v_body=v_body,
        mu_body=MU_EARTH,
        r_body=R_EARTH,
        h_safe=200.0,
    )

    assert isinstance(result, FlybyResult)
    assert abs(result.v_inf_in_mag - v_inf_mag) < 1e-10
    assert abs(result.v_inf_out_mag - v_inf_mag) < 1e-10
    assert abs(np.degrees(result.turn_angle) - 60.0) < 1e-5

    # Check periapsis radius: e = 1 / sin(30 deg) = 2.0
    # rp = mu / v_inf^2 * (e - 1) = 398600.4418 / 16.0 * 1.0 = 24912.5 km
    rp_expected = (MU_EARTH / (v_inf_mag ** 2)) * (2.0 - 1.0)
    assert abs(result.rp - rp_expected) < 1e-4
    assert result.hp == result.rp - R_EARTH
    assert result.is_feasible is True
    assert result.delta_v_peri == 0.0

    # Heliocentric delta-v magnitude = 2 * v_inf * sin(delta / 2) = 2 * 4 * sin(30 deg) = 4.0 km/s
    assert abs(result.delta_v_flyby_mag - 4.0) < 1e-10


def test_solve_flyby_infeasible_low_altitude():
    """Verify is_feasible is False when required periapsis penetrates below safe altitude."""
    v_body = np.array([0.0, 35.0, 0.0])
    v_inf_mag = 10.0  # High speed means tight turn requires subsurface periapsis

    # 150 degree turn angle
    delta = np.radians(150.0)
    v_inf_in = np.array([v_inf_mag, 0.0, 0.0])
    v_inf_out = np.array([v_inf_mag * np.cos(delta), v_inf_mag * np.sin(delta), 0.0])

    result = solve_flyby(
        v_in=v_body + v_inf_in,
        v_out=v_body + v_inf_out,
        v_body=v_body,
        mu_body=MU_VENUS,
        r_body=R_VENUS,
        h_safe=300.0,
    )

    assert result.hp < 300.0
    assert result.is_feasible is False


def test_solve_flyby_powered_energy_mismatch():
    """Verify powered flyby calculation when inbound and outbound v_inf magnitudes differ."""
    v_body = np.array([0.0, 29.8, 0.0])
    v_in = v_body + np.array([3.0, 0.0, 0.0])   # v_inf_in = 3.0 km/s
    v_out = v_body + np.array([0.0, 5.0, 0.0])  # v_inf_out = 5.0 km/s

    result = solve_flyby(
        v_in=v_in,
        v_out=v_out,
        v_body=v_body,
        mu_body=MU_EARTH,
        r_body=R_EARTH,
        h_safe=200.0,
    )

    assert result.is_feasible is False  # unpowered is infeasible due to energy mismatch
    assert result.delta_v_peri > 0.0    # requires impulsive burn at periapsis


def test_flyby_straight_through():
    """Verify straight-through flyby (zero turn angle) yields infinite rp and zero delta_v."""
    v_body = np.array([0.0, 30.0, 0.0])
    v_inf = np.array([4.0, 0.0, 0.0])

    result = solve_flyby(
        v_in=v_body + v_inf,
        v_out=v_body + v_inf,
        v_body=v_body,
        mu_body=MU_EARTH,
        r_body=R_EARTH,
    )

    assert result.turn_angle == 0.0
    assert result.rp == np.inf
    assert result.hp == np.inf
    assert result.delta_v_flyby_mag == 0.0
    assert result.is_feasible is True


def test_flyby_input_validation():
    """Verify input validation errors for invalid vectors, angles, and speeds."""
    # Invalid vector shape
    with pytest.raises(ValueError, match="length-3"):
        solve_flyby([1, 2], [3, 4, 5], [6, 7, 8], MU_EARTH, R_EARTH)

    # Near zero v_inf
    with pytest.raises(ValueError, match="near zero"):
        solve_flyby([10, 0, 0], [10, 0, 0], [10, 0, 0], MU_EARTH, R_EARTH)

    # Invalid turn angle
    with pytest.raises(ValueError, match="between 0 and pi"):
        compute_flyby_periapsis(v_inf=5.0, turn_angle=-0.1, mu_body=MU_EARTH)
    with pytest.raises(ValueError, match="between 0 and pi"):
        compute_flyby_periapsis(v_inf=5.0, turn_angle=3.5, mu_body=MU_EARTH)

    # Non-positive v_inf or mu
    with pytest.raises(ValueError, match="strictly positive"):
        compute_max_turn_angle(v_inf=-1.0, mu_body=MU_EARTH, r_body=R_EARTH)
    with pytest.raises(ValueError, match="strictly positive"):
        compute_max_turn_angle(v_inf=5.0, mu_body=-100.0, r_body=R_EARTH)
    with pytest.raises(ValueError, match="Minimum periapsis radius"):
        compute_max_turn_angle(v_inf=5.0, mu_body=MU_EARTH, r_body=-7000.0, h_safe=100.0)
    with pytest.raises(ValueError, match="strictly positive"):
        compute_flyby_periapsis(v_inf=-2.0, turn_angle=0.5, mu_body=MU_EARTH)

    # Near zero v_inf for compute_flyby_turn_angle
    with pytest.raises(ValueError, match="cannot be near zero"):
        compute_flyby_turn_angle([0, 0, 0], [1, 0, 0])
