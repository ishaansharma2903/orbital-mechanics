"""
tests/core/test_state.py

Test suite for orbital_mechanics.core.state.State.
"""
import numpy as np
import pytest

from orbital_mechanics import State, OrbitalElements, elements_to_state, MU_EARTH

ATOL_ANGLE = 1e-6   # rad
ATOL_LEN = 1e-6      # km / (km/s)


# ---------------------------------------------------------------------------
# Known reference case (Curtis, Example 4.3)
# r = [-6045, -3490, 2500] km, v = [-3.457, 6.618, 2.533] km/s
# Expected: h=58310 km^2/s, e=0.1712, i=153.2 deg, RAAN=255.3 deg,
#           omega=20.07 deg, theta=28.45 deg
# ---------------------------------------------------------------------------
def test_known_reference_case():
    s = State(position=[-6045, -3490, 2500], velocity=[-3.457, 6.618, 2.533])
    elems = s.to_orbit_elements()

    h_vec = np.cross(s.position, s.velocity)
    assert np.linalg.norm(h_vec) == pytest.approx(58310, rel=1e-3)

    assert elems.e == pytest.approx(0.1712, abs=1e-3)
    assert np.degrees(elems.i) == pytest.approx(153.2, abs=1e-1)
    assert np.degrees(elems.raan) == pytest.approx(255.3, abs=1e-1)
    assert np.degrees(elems.arg_periapsis) == pytest.approx(20.07, abs=1e-1)
    assert np.degrees(elems.true_anomaly) == pytest.approx(28.45, abs=1e-1)


# ---------------------------------------------------------------------------
# Round-trip: elements -> state -> elements, across representative geometries
# ---------------------------------------------------------------------------
ROUND_TRIP_CASES = {
    "circular_inclined": OrbitalElements(
        a=7000, e=0.0, i=np.radians(51.6), raan=np.radians(120),
        arg_periapsis=0.0, true_anomaly=np.radians(45),
    ),
    "elliptical_inclined": OrbitalElements(
        a=26600, e=0.74, i=np.radians(63.4), raan=np.radians(200),
        arg_periapsis=np.radians(270), true_anomaly=np.radians(100),
    ),
    "equatorial_elliptical": OrbitalElements(
        a=42164, e=0.3, i=0.0, raan=0.0,
        arg_periapsis=np.radians(50), true_anomaly=np.radians(10),
    ),
    "polar_circular": OrbitalElements(
        a=7500, e=0.0, i=np.radians(90), raan=np.radians(30),
        arg_periapsis=0.0, true_anomaly=np.radians(200),
    ),
    "circular_equatorial": OrbitalElements(
        a=7000, e=0.0, i=0.0, raan=0.0, arg_periapsis=0.0,
        true_anomaly=np.radians(75),
    ),
    "high_eccentricity": OrbitalElements(
        a=24500, e=0.95, i=np.radians(28.5), raan=np.radians(80),
        arg_periapsis=np.radians(15), true_anomaly=np.radians(5),
    ),
}


@pytest.mark.parametrize("name", ROUND_TRIP_CASES.keys())
def test_round_trip_elements(name):
    original = ROUND_TRIP_CASES[name]
    s = elements_to_state(original)
    recovered = s.to_orbit_elements()

    assert recovered.a == pytest.approx(original.a, rel=1e-6)
    assert recovered.e == pytest.approx(original.e, abs=1e-8)
    assert recovered.i == pytest.approx(original.i, abs=ATOL_ANGLE)

    # RAAN is only meaningful for inclined orbits; skip for equatorial cases
    if original.i > 1e-6:
        assert recovered.raan % (2 * np.pi) == pytest.approx(
            original.raan % (2 * np.pi), abs=ATOL_ANGLE
        )

    # Argument of periapsis requires both a defined node vector (inclined
    # orbit) and a defined eccentricity vector (eccentric orbit); for
    # equatorial orbits the module conventionally returns 0.0 regardless of
    # the "true" periapsis direction, so skip the comparison in that case.
    if original.e > 1e-6 and original.i > 1e-6:
        assert recovered.arg_periapsis % (2 * np.pi) == pytest.approx(
            original.arg_periapsis % (2 * np.pi), abs=ATOL_ANGLE
        )

    if original.e > 1e-6:
        assert recovered.true_anomaly % (2 * np.pi) == pytest.approx(
            original.true_anomaly % (2 * np.pi), abs=ATOL_ANGLE
        )


# ---------------------------------------------------------------------------
# Perifocal frame transform
# ---------------------------------------------------------------------------
def test_perifocal_frame_zeroes_z_component():
    s = State(position=[-6045, -3490, 2500], velocity=[-3.457, 6.618, 2.533])
    pf = s.to_perifocal_frame()
    assert pf.position[2] == pytest.approx(0.0, abs=1e-8)
    assert pf.velocity[2] == pytest.approx(0.0, abs=1e-8)


def test_perifocal_frame_preserves_radius_and_speed():
    s = State(position=[-6045, -3490, 2500], velocity=[-3.457, 6.618, 2.533])
    pf = s.to_perifocal_frame()
    assert np.linalg.norm(pf.position) == pytest.approx(np.linalg.norm(s.position), rel=1e-9)
    assert np.linalg.norm(pf.velocity) == pytest.approx(np.linalg.norm(s.velocity), rel=1e-9)


# ---------------------------------------------------------------------------
# Physical sanity checks
# ---------------------------------------------------------------------------
def test_specific_energy_matches_semimajor_axis():
    s = State(position=[-6045, -3490, 2500], velocity=[-3.457, 6.618, 2.533])
    elems = s.to_orbit_elements()
    r = np.linalg.norm(s.position)
    v = np.linalg.norm(s.velocity)
    energy = v ** 2 / 2 - MU_EARTH / r
    a_from_energy = -MU_EARTH / (2 * energy)
    assert elems.a == pytest.approx(a_from_energy, rel=1e-6)


def test_angular_momentum_magnitude_matches_pqw_construction():
    elems = OrbitalElements(
        a=8000, e=0.2, i=np.radians(45), raan=np.radians(10),
        arg_periapsis=np.radians(30), true_anomaly=np.radians(60),
    )
    s = elements_to_state(elems)
    h_expected = np.sqrt(MU_EARTH * elems.a * (1 - elems.e ** 2))
    h_actual = np.linalg.norm(np.cross(s.position, s.velocity))
    assert h_actual == pytest.approx(h_expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Input validation / degenerate states
# ---------------------------------------------------------------------------
def test_rejects_wrong_shape_vectors():
    with pytest.raises(ValueError):
        State(position=[1, 2], velocity=[1, 2, 3])
    with pytest.raises(ValueError):
        State(position=[1, 2, 3], velocity=[1, 2])


def test_accepts_list_and_tuple_input_and_casts_to_float_array():
    s = State(position=(7000, 0, 0), velocity=[0, 7.5, 0])
    assert isinstance(s.position, np.ndarray)
    assert s.position.dtype == float
    assert isinstance(s.velocity, np.ndarray)


def test_default_state_is_zero_vectors():
    s = State()
    np.testing.assert_array_equal(s.position, [0, 0, 0])
    np.testing.assert_array_equal(s.velocity, [0, 0, 0])


def test_rejects_zero_position():
    s = State(position=[0, 0, 0], velocity=[1, 0, 0])
    with pytest.raises(ValueError):
        s.to_orbit_elements()


def test_rejects_zero_angular_momentum_rectilinear_trajectory():
    # Purely radial velocity => h = r x v = 0 (degenerate rectilinear orbit)
    s = State(position=[7000, 0, 0], velocity=[1.0, 0, 0])
    with pytest.raises(ValueError):
        s.to_orbit_elements()


def test_state_is_immutable():
    s = State(position=[7000, 0, 0], velocity=[0, 7.5, 0])
    with pytest.raises(Exception):
        s.position = np.array([0, 0, 0])


# ---------------------------------------------------------------------------
# Undefined-angle conventions (circular / equatorial edge cases)
# ---------------------------------------------------------------------------
def test_circular_orbit_true_anomaly_and_arg_periapsis_default_to_zero():
    s = State(position=[7000, 0, 0], velocity=[0, np.sqrt(MU_EARTH / 7000), 0])
    elems = s.to_orbit_elements()
    assert elems.e == pytest.approx(0.0, abs=1e-8)
    assert elems.arg_periapsis == 0.0
    assert elems.true_anomaly == 0.0


def test_equatorial_orbit_raan_defaults_to_zero():
    s = State(position=[7000, 0, 0], velocity=[0, 8.0, 0])
    elems = s.to_orbit_elements()
    assert elems.i == pytest.approx(0.0, abs=1e-8)
    assert elems.raan == 0.0
