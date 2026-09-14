"""
tests/maneuvers/test_lambert.py

Validation suite for orbital_mechanics.maneuvers.lambert.
"""
import numpy as np
import pytest

from orbital_mechanics import (
    State,
    OrbitalElements,
    elements_to_state,
    solve_lambert,
    lambert,
    MU_EARTH,
    MU_SUN,
)


# ---------------------------------------------------------------------------
# Textbook Reference Case 1: Curtis (Example 5.2) - Elliptic Geocentric
# r1 = [5000, 10000, 2100] km
# r2 = [-14600, 2500, 7000] km
# tof = 3600 s (1 hour), mu = 398600 km^3/s^2
# Expected:
#   v1 = [-5.9925, 1.9254, 3.2456] km/s
#   v2 = [-3.3125, -4.1966, -0.3853] km/s
# ---------------------------------------------------------------------------
def test_curtis_example_5_2_elliptic():
    r1 = np.array([5000.0, 10000.0, 2100.0])
    r2 = np.array([-14600.0, 2500.0, 7000.0])
    tof = 3600.0
    mu = 398600.0

    v1, v2 = solve_lambert(r1, r2, tof, mu=mu, prograde=True)

    expected_v1 = np.array([-5.9925, 1.9254, 3.2456])
    expected_v2 = np.array([-3.3125, -4.1966, -0.3853])

    np.testing.assert_allclose(v1, expected_v1, atol=1e-3)
    np.testing.assert_allclose(v2, expected_v2, atol=1e-3)

    # State orbital elements check
    s1 = State(r1, v1)
    elems1 = s1.to_orbit_elements(mu=mu)
    assert elems1.e < 1.0  # Must be an elliptic transfer orbit
    assert elems1.a > 0.0


# ---------------------------------------------------------------------------
# Textbook Reference Case 2: Curtis (Example 5.3) - Hyperbolic Geocentric
# Sighting 1: altitude = 267000 km -> r1 = 273378 km
# Sighting 2: altitude = 140000 km -> r2 = 146378 km
# Delta theta = 5 deg, Delta t = 48600 s (13.5 hours), mu = 398600 km^3/s^2
# Expected:
#   Hyperbolic orbit (e > 1), perigee radius rp ~ 6538.2 km, perigee altitude ~ 160.2 km
# ---------------------------------------------------------------------------
def test_curtis_example_5_3_hyperbolic():
    r1_mag = 273378.0
    r2_mag = 146378.0
    dtheta = np.radians(5.0)
    tof = 48600.0
    mu = 398600.0

    r1 = np.array([r1_mag, 0.0, 0.0])
    r2 = np.array([r2_mag * np.cos(dtheta), r2_mag * np.sin(dtheta), 0.0])

    v1, v2 = solve_lambert(r1, r2, tof, mu=mu, prograde=True)

    s1 = State(r1, v1)
    elems = s1.to_orbit_elements(mu=mu)

    # Must be hyperbolic
    assert elems.e > 1.0
    assert elems.e == pytest.approx(1.0506, abs=1e-3)
    assert elems.a < 0.0

    # Perigee radius rp = a * (1 - e)
    rp = elems.a * (1.0 - elems.e)
    assert rp == pytest.approx(6538.2, abs=1.0)
    # Altitude above Earth (R_EARTH = 6378 km)
    perigee_altitude = rp - 6378.0
    assert perigee_altitude == pytest.approx(160.2, abs=1.0)


# ---------------------------------------------------------------------------
# Physical Conservation Laws
# ---------------------------------------------------------------------------
def test_conservation_of_specific_energy_and_angular_momentum():
    r1 = np.array([5000.0, 10000.0, 2100.0])
    r2 = np.array([-14600.0, 2500.0, 7000.0])
    tof = 3600.0
    mu = MU_EARTH

    v1, v2 = solve_lambert(r1, r2, tof, mu=mu, prograde=True)

    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    # Specific energy: E = v^2/2 - mu/r
    energy1 = 0.5 * v1_norm ** 2 - mu / r1_norm
    energy2 = 0.5 * v2_norm ** 2 - mu / r2_norm
    assert energy1 == pytest.approx(energy2, rel=1e-7)

    # Angular momentum vector: h = r x v
    h1 = np.cross(r1, v1)
    h2 = np.cross(r2, v2)
    np.testing.assert_allclose(h1, h2, rtol=1e-7)

    # Planar alignment: position and velocity vectors must be orthogonal to h
    h_unit = h1 / np.linalg.norm(h1)
    assert abs(np.dot(r1, h_unit)) == pytest.approx(0.0, abs=1e-7)
    assert abs(np.dot(r2, h_unit)) == pytest.approx(0.0, abs=1e-7)
    assert abs(np.dot(v1, h_unit)) == pytest.approx(0.0, abs=1e-7)
    assert abs(np.dot(v2, h_unit)) == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# Round-Trip Test: Propagated Keplerian state -> Lambert recovery
# ---------------------------------------------------------------------------
def test_round_trip_keplerian_orbit():
    # Known elliptic inclined orbit
    original_elems = OrbitalElements(
        a=15000.0,
        e=0.35,
        i=np.radians(28.5),
        raan=np.radians(45.0),
        arg_periapsis=np.radians(30.0),
        true_anomaly=np.radians(20.0),
    )
    s1 = elements_to_state(original_elems, mu=MU_EARTH)

    # Propagate forward by true anomaly delta = 60 deg
    delta_nu = np.radians(60.0)
    nu2 = original_elems.true_anomaly + delta_nu
    elems2 = original_elems._replace(true_anomaly=nu2)
    s2 = elements_to_state(elems2, mu=MU_EARTH)

    # Calculate exact time of flight using Kepler's equation
    e = original_elems.e
    a = original_elems.a
    E1 = 2.0 * np.arctan(np.sqrt((1.0 - e) / (1.0 + e)) * np.tan(original_elems.true_anomaly / 2.0))
    E2 = 2.0 * np.arctan(np.sqrt((1.0 - e) / (1.0 + e)) * np.tan(nu2 / 2.0))
    M1 = E1 - e * np.sin(E1)
    M2 = E2 - e * np.sin(E2)
    n = np.sqrt(MU_EARTH / a ** 3)
    tof = (M2 - M1) / n

    # Solve Lambert
    v1_rec, v2_rec = solve_lambert(s1.position, s2.position, tof, mu=MU_EARTH, prograde=True)

    np.testing.assert_allclose(v1_rec, s1.velocity, rtol=1e-6)
    np.testing.assert_allclose(v2_rec, s2.velocity, rtol=1e-6)


# ---------------------------------------------------------------------------
# Direction of Motion: Prograde vs Retrograde & Short-way vs Long-way
# ---------------------------------------------------------------------------
def test_prograde_vs_retrograde_geometry():
    r1 = np.array([7000.0, 0.0, 0.0])
    r2 = np.array([0.0, 8000.0, 0.0])
    tof = 2500.0
    mu = MU_EARTH

    # Prograde transfer: h_z > 0
    v1_pro, v2_pro = solve_lambert(r1, r2, tof, mu=mu, prograde=True)
    h_pro = np.cross(r1, v1_pro)
    assert h_pro[2] > 0.0

    # Retrograde transfer: h_z < 0
    v1_ret, v2_ret = solve_lambert(r1, r2, tof, mu=mu, prograde=False)
    h_ret = np.cross(r1, v1_ret)
    assert h_ret[2] < 0.0


def test_explicit_short_way_and_long_way():
    r1 = np.array([1.0e8, 0.0, 0.0])
    r2 = np.array([0.0, 1.2e8, 0.0])
    tof_short = 50.0 * 86400.0  # 50 days
    tof_long = 200.0 * 86400.0  # 200 days
    mu = MU_SUN

    v1_short, _ = solve_lambert(r1, r2, tof_short, mu=mu, short_way=True)
    v1_long, _ = solve_lambert(r1, r2, tof_long, mu=mu, short_way=False)

    # Different trajectories must produce different departure velocities
    assert not np.allclose(v1_short, v1_long)


# ---------------------------------------------------------------------------
# Heliocentric Interplanetary Transfer (Earth to Mars scale)
# ---------------------------------------------------------------------------
def test_heliocentric_transfer():
    # 1 AU departure, 1.524 AU arrival, 90 deg transfer angle
    r1 = np.array([1.4959787e8, 0.0, 0.0])
    r2 = np.array([0.0, 2.279e8, 0.0])
    tof = 180.0 * 86400.0  # 180 days

    v1, v2 = solve_lambert(r1, r2, tof, mu=MU_SUN, prograde=True)

    # Heliocentric speeds are on the order of ~20 to ~40 km/s
    speed1 = np.linalg.norm(v1)
    speed2 = np.linalg.norm(v2)
    assert 20.0 < speed1 < 50.0
    assert 10.0 < speed2 < 40.0


# ---------------------------------------------------------------------------
# Input Validation and Edge Cases
# ---------------------------------------------------------------------------
def test_rejects_non_positive_tof_or_mu():
    r1 = np.array([7000.0, 0.0, 0.0])
    r2 = np.array([0.0, 8000.0, 0.0])

    with pytest.raises(ValueError, match="Time of flight must be strictly positive"):
        solve_lambert(r1, r2, tof=-100.0, mu=MU_EARTH)

    with pytest.raises(ValueError, match="Time of flight must be strictly positive"):
        solve_lambert(r1, r2, tof=0.0, mu=MU_EARTH)

    with pytest.raises(ValueError, match="Gravitational parameter mu must be strictly positive"):
        solve_lambert(r1, r2, tof=1000.0, mu=0.0)


def test_rejects_degenerate_position_vector():
    with pytest.raises(ValueError, match="Position magnitude ~0"):
        solve_lambert([0.0, 0.0, 0.0], [7000.0, 0.0, 0.0], tof=1000.0)

    with pytest.raises(ValueError, match="Position magnitude ~0"):
        solve_lambert([7000.0, 0.0, 0.0], [0.0, 0.0, 0.0], tof=1000.0)


def test_rejects_invalid_vector_dimensions():
    with pytest.raises(ValueError, match="3-dimensional"):
        solve_lambert([7000.0, 0.0], [0.0, 8000.0, 0.0], tof=1000.0)

    with pytest.raises(ValueError, match="3-dimensional"):
        solve_lambert([7000.0, 0.0, 0.0], [0.0, 8000.0, 0.0, 1.0], tof=1000.0)


def test_rejects_collinear_or_180_deg_transfer():
    r1 = np.array([7000.0, 0.0, 0.0])
    r2 = np.array([-7000.0, 0.0, 0.0])  # Exactly 180 deg
    with pytest.raises(ValueError, match="180-degree transfer"):
        solve_lambert(r1, r2, tof=3000.0, mu=MU_EARTH)


def test_extreme_hyperbolic_transfer():
    # Extremely fast transfer requiring negative-y correction
    r1 = np.array([7000.0, 0.0, 0.0])
    r2 = np.array([0.0, 7000.0, 0.0])
    tof = 50.0  # 50 seconds for 90-degree geocentric chord
    mu = MU_EARTH

    v1, v2 = solve_lambert(r1, r2, tof, mu=mu, prograde=True)
    s1 = State(r1, v1)
    elems = s1.to_orbit_elements(mu=mu)
    assert elems.e > 1.0  # Strong hyperbola


def test_raises_runtime_error_when_max_iter_exceeded():
    r1 = np.array([5000.0, 10000.0, 2100.0])
    r2 = np.array([-14600.0, 2500.0, 7000.0])
    with pytest.raises(RuntimeError, match="failed to converge after 1 iterations"):
        solve_lambert(r1, r2, tof=3600.0, mu=MU_EARTH, max_iter=1)


def test_convenience_alias_is_identical():
    assert lambert is solve_lambert


