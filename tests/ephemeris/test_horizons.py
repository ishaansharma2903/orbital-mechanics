"""
tests/ephemeris/test_horizons.py

Unit and validation tests for JPL Horizons ephemeris retrieval.
"""
import datetime
import pytest
import numpy as np
from astropy.time import Time

from polaris.constants import AU
from polaris.core.state import State
from polaris.ephemeris.horizons import (
    get_body_state,
    parse_epoch,
    clear_ephemeris_cache,
    _resolve_body_id,
    _resolve_center_id,
    BODY_MAP,
    CENTER_MAP,
)


def test_parse_epoch_variants():
    """Verify parse_epoch correctly parses all supported date formats to JD."""
    # J2000 epoch JD
    jd_j2000 = 2451545.0
    assert parse_epoch(2451545.0) == 2451545.0
    assert parse_epoch(2451545) == 2451545.0
    assert parse_epoch("2451545.0") == 2451545.0

    # ISO string
    assert abs(parse_epoch("2000-01-01 12:00:00") - jd_j2000) < 1e-6
    assert abs(parse_epoch("2004-08-03") - 2453220.5) < 1e-6

    # datetime
    dt = datetime.datetime(2000, 1, 1, 12, 0, 0)
    assert abs(parse_epoch(dt) - jd_j2000) < 1e-6

    d = datetime.date(2004, 8, 3)
    assert abs(parse_epoch(d) - 2453220.5) < 1e-6

    # astropy Time
    t = Time("2000-01-01 12:00:00")
    assert abs(parse_epoch(t) - jd_j2000) < 1e-6


def test_parse_epoch_invalid():
    """Verify parse_epoch raises TypeError for unsupported types."""
    with pytest.raises(TypeError):
        parse_epoch([2020, 1, 1])


def test_resolve_body_id():
    """Verify body name resolution to NAIF / Horizons target IDs."""
    assert _resolve_body_id("earth") == "399"
    assert _resolve_body_id("Earth") == "399"
    assert _resolve_body_id(" VENUS ") == "299"
    assert _resolve_body_id("mercury") == "199"
    assert _resolve_body_id("mars") == "499"
    assert _resolve_body_id("sun") == "10"
    assert _resolve_body_id(399) == "399"
    assert _resolve_body_id("custom_asteroid") == "custom_asteroid"


def test_resolve_center_id():
    """Verify center observer resolution to Horizons location syntax."""
    assert _resolve_center_id("sun") == "@10"
    assert _resolve_center_id("Sun") == "@10"
    assert _resolve_center_id("ssb") == "@0"
    assert _resolve_center_id("earth") == "@399"
    assert _resolve_center_id("@10") == "@10"
    assert _resolve_center_id(10) == "@10"
    assert _resolve_center_id("0") == "@0"


def test_get_body_state_earth_j2000():
    """
    Validate Earth's heliocentric position and velocity at J2000.0 epoch
    against physical astronomical invariants.
    """
    state = get_body_state("earth", 2451545.0)
    assert isinstance(state, State)
    assert state.position.shape == (3,)
    assert state.velocity.shape == (3,)

    # Distance from Sun: Earth is near perihelion (~0.983 AU) in early January
    r_mag = np.linalg.norm(state.position)
    assert 0.98 * AU < r_mag < 1.02 * AU

    # Heliocentric orbital speed: ~29.3 - 30.3 km/s
    v_mag = np.linalg.norm(state.velocity)
    assert 29.0 < v_mag < 31.0

    # In ecliptic coordinates, z and vz components for Earth should be nearly zero
    assert abs(state.position[2]) < 2000.0  # < 2000 km out of ecliptic plane
    assert abs(state.velocity[2]) < 0.01    # < 10 m/s z-velocity


def test_get_body_state_planets():
    """Verify ephemeris retrieval for multiple planetary bodies."""
    epoch = "2004-08-03"  # MESSENGER launch epoch

    for body in ["mercury", "venus", "earth", "mars"]:
        st = get_body_state(body, epoch)
        r_mag = np.linalg.norm(st.position)
        v_mag = np.linalg.norm(st.velocity)

        if body == "mercury":
            assert 0.30 * AU < r_mag < 0.48 * AU
            assert 38.0 < v_mag < 60.0
        elif body == "venus":
            assert 0.71 * AU < r_mag < 0.73 * AU
            assert 34.0 < v_mag < 36.0
        elif body == "earth":
            assert 0.98 * AU < r_mag < 1.02 * AU
            assert 29.0 < v_mag < 31.0
        elif body == "mars":
            assert 1.38 * AU < r_mag < 1.67 * AU
            assert 21.0 < v_mag < 27.0


def test_ephemeris_caching():
    """Verify caching prevents re-fetching identical query parameters."""
    clear_ephemeris_cache()
    # First call
    st1 = get_body_state("earth", 2451545.0)
    # Second call (hits cache)
    st2 = get_body_state("earth", 2451545.0)
    np.testing.assert_array_equal(st1.position, st2.position)
    np.testing.assert_array_equal(st1.velocity, st2.velocity)
    clear_ephemeris_cache()


def test_get_body_state_mock(monkeypatch):
    """Verify get_body_state with mocked Horizons response for offline reliability."""
    from astropy.table import Table

    mock_table = Table({
        "x": [1.0],
        "y": [0.0],
        "z": [0.0],
        "vx": [0.0],
        "vy": [0.0172],
        "vz": [0.0],
    })

    class MockHorizons:
        def __init__(self, *args, **kwargs):
            pass
        def vectors(self, *args, **kwargs):
            return mock_table

    monkeypatch.setattr("polaris.ephemeris.horizons.Horizons", MockHorizons)
    clear_ephemeris_cache()

    st = get_body_state("earth", "2020-01-01")
    np.testing.assert_allclose(st.position, [AU, 0.0, 0.0])
    clear_ephemeris_cache()


def test_get_body_state_empty_vectors_raises(monkeypatch):
    """Verify get_body_state raises ValueError when Horizons returns empty results."""
    from astropy.table import Table

    mock_table = Table({"x": [], "y": [], "z": [], "vx": [], "vy": [], "vz": []})

    class MockEmptyHorizons:
        def __init__(self, *args, **kwargs):
            pass
        def vectors(self, *args, **kwargs):
            return mock_table

    monkeypatch.setattr("polaris.ephemeris.horizons.Horizons", MockEmptyHorizons)
    clear_ephemeris_cache()

    with pytest.raises(ValueError, match="No ephemeris data"):
        get_body_state("unknown_body", "2020-01-01")
    clear_ephemeris_cache()
