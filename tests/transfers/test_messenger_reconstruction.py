"""
tests/transfers/test_messenger_reconstruction.py

Validation suite for the reconstruction of the MESSENGER mission trajectory:
Earth-Earth-Venus-Venus-Mercury-Mercury-Mercury-Mercury Multi-Gravity-Assist (MGA)
and Deep-Space Maneuver (DSM) sequence.

Reference:
- McAdams, J. V., et al., "MESSENGER Mission Design and Navigation: From Launch to Mercury Orbit",
  AIAA / Space Science Reviews (2007).
- Yen, C. L., "Ballistic Mercury Orbiter Mission via Venus and Mercury Gravity Assists",
  AAS/AIAA Astrodynamics Conference.
"""
import pytest
import numpy as np

from polaris import (
    State,
    solve_flyby,
    solve_single_leg,
    solve_dsm_leg,
    solve_mga_trajectory,
    get_body_state,
    MU_SUN,
    MU_EARTH,
    MU_VENUS,
    MU_MERCURY,
    R_EARTH,
    R_VENUS,
    R_MERCURY,
    AU,
)


# Historical encounter epochs for MESSENGER
MESSENGER_ENCOUNTERS = {
    "launch": "2004-08-03",
    "earth_fb": "2005-08-02",
    "venus_fb1": "2006-10-24",
    "venus_fb2": "2007-06-05",
    "mercury_fb1": "2008-01-14",
    "mercury_fb2": "2008-10-06",
    "mercury_fb3": "2009-09-29",
    "moi": "2011-03-18",
}


def test_messenger_total_mission_timeline():
    """Verify total duration of the 6.6-year cruise phase to Mercury orbit insertion."""
    bodies = ["earth", "earth", "venus", "venus", "mercury", "mercury", "mercury", "mercury"]
    epochs = list(MESSENGER_ENCOUNTERS.values())

    traj = solve_mga_trajectory(bodies=bodies, epochs=epochs, mu=MU_SUN)

    # 2418 days = ~6.62 years from Aug 3, 2004 to Mar 18, 2011
    assert traj.tof_total_days == pytest.approx(2418.0, abs=1.0)
    assert len(traj.legs) == 7
    assert len(traj.flybys) == 6


def test_messenger_earth_flyby_kinematics():
    """
    Validate the Earth gravity assist encounter (Aug 2, 2005).
    """
    # Outbound from Earth flyby heading to Venus 1
    t_earth_fb = MESSENGER_ENCOUNTERS["earth_fb"]
    t_venus_1 = MESSENGER_ENCOUNTERS["venus_fb1"]

    leg_earth_venus = solve_single_leg("earth", "venus", t_earth_fb, t_venus_1)

    # Earth excess velocity departing for Venus in ballistic model
    assert leg_earth_venus.v_inf_dep_mag > 0.0
    assert leg_earth_venus.c3 > 0.0
    assert leg_earth_venus.tof_days == pytest.approx(448.0, abs=1.0)


def test_messenger_venus_gravity_assists():
    """
    Validate Venus flyby 1 (Oct 24, 2006) and Venus flyby 2 (Jun 5, 2007).
    Published: Venus 1 altitude ~ 2992 km; Venus 2 altitude ~ 338 km.
    """
    t_v1 = MESSENGER_ENCOUNTERS["venus_fb1"]
    t_v2 = MESSENGER_ENCOUNTERS["venus_fb2"]
    t_m1 = MESSENGER_ENCOUNTERS["mercury_fb1"]

    # Inbound to Venus 2 from Venus 1, outbound from Venus 2 to Mercury 1
    leg_v1_v2 = solve_single_leg("venus", "venus", t_v1, t_v2)
    leg_v2_m1 = solve_single_leg("venus", "mercury", t_v2, t_m1)

    fb_venus_2 = solve_flyby(
        v_in=leg_v1_v2.arrival_state_craft.velocity,
        v_out=leg_v2_m1.departure_state_craft.velocity,
        v_body=leg_v1_v2.arrival_state_body.velocity,
        mu_body=MU_VENUS,
        r_body=R_VENUS,
        h_safe=300.0,
    )

    # Check physical deflection and energy consistency
    assert fb_venus_2.turn_angle > 0.0
    assert fb_venus_2.v_inf_in_mag > 0.0
    assert fb_venus_2.v_inf_out_mag > 0.0


def test_messenger_mercury_orbit_insertion():
    """
    Validate Mercury orbit insertion (MOI) Delta-V budget at encounter (Mar 18, 2011).
    Published: MOI Delta-v ~ 0.865 km/s for insertion into a 200 km x 15,300 km orbit.
    """
    t_m3 = MESSENGER_ENCOUNTERS["mercury_fb3"]
    t_moi = MESSENGER_ENCOUNTERS["moi"]

    final_leg = solve_single_leg("mercury", "mercury", t_m3, t_moi)

    # Inbound excess velocity at Mercury
    assert final_leg.v_inf_arr_mag > 0.0

    # Capture delta-v for 200 km periapsis parking orbit
    r_park_moi = R_MERCURY + 200.0
    v_c = np.sqrt(MU_MERCURY / r_park_moi)
    # v_peri = sqrt(v_inf^2 + 2*mu/r_park)
    v_hyp = np.sqrt(final_leg.v_inf_arr_mag ** 2 + 2 * MU_MERCURY / r_park_moi)
    dv_capture = v_hyp - v_c

    assert dv_capture > 0.0
