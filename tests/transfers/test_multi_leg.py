"""
tests/transfers/test_multi_leg.py

Unit and validation tests for Multi-Gravity-Assist (MGA) trajectory assembly.
"""
import pytest
import numpy as np

from polaris.constants import (
    MU_SUN,
    MU_EARTH,
    MU_VENUS,
    MU_MERCURY,
    R_EARTH,
    R_VENUS,
    R_MERCURY,
    AU,
)
from polaris.transfers.multi_leg import (
    MGATrajectory,
    solve_mga_trajectory,
    get_body_params,
)


def test_get_body_params():
    """Verify planetary parameter lookup for standard bodies."""
    mu_e, r_e = get_body_params("earth")
    assert mu_e == MU_EARTH
    assert r_e == R_EARTH

    mu_v, r_v = get_body_params("venus")
    assert mu_v == MU_VENUS
    assert r_v == R_VENUS

    mu_m, r_m = get_body_params("mercury")
    assert mu_m == MU_MERCURY
    assert r_m == R_MERCURY

    with pytest.raises(KeyError, match="not found"):
        get_body_params("unknown_planet_xyz")


def test_solve_mga_trajectory_earth_venus_mercury():
    """
    Validate multi-leg trajectory assembly for an Earth-Venus-Mercury sequence.
    """
    bodies = ["earth", "venus", "mercury"]
    epochs = ["2004-08-03", "2005-04-01", "2005-10-01"]

    traj = solve_mga_trajectory(
        bodies=bodies,
        epochs=epochs,
        h_safe_dict={"venus": 300.0},
        mu=MU_SUN,
    )

    assert isinstance(traj, MGATrajectory)
    assert len(traj.legs) == 2
    assert len(traj.flybys) == 1
    assert traj.tof_total_days > 0.0
    assert traj.c3 > 0.0
    assert traj.v_inf_dep_mag > 0.0
    assert traj.v_inf_arr_mag > 0.0

    # Intermediate flyby at Venus
    fb_venus = traj.flybys[0]
    assert fb_venus.v_inf_in_mag > 0.0
    assert fb_venus.v_inf_out_mag > 0.0
    assert fb_venus.turn_angle > 0.0

    # Total mission delta-v with parking orbits
    r_park_earth = R_EARTH + 200.0
    r_park_mercury = R_MERCURY + 200.0
    dv_mission = traj.total_mission_delta_v(
        r_park_dep=r_park_earth,
        mu_dep=MU_EARTH,
        r_park_arr=r_park_mercury,
        mu_arr=MU_MERCURY,
    )
    assert dv_mission > 0.0


def test_solve_mga_trajectory_with_dsm():
    """Verify MGA trajectory assembly with an intermediate deep-space maneuver on leg 0."""
    bodies = ["earth", "venus", "mercury"]
    epochs = [2453220.5, 2453500.5, 2453800.5]

    # DSM at mid-point of Leg 0
    t_dsm = 2453350.0
    r_dsm = np.array([0.9 * AU, 0.4 * AU, 0.02 * AU])

    dsm_configs = {
        0: {"epoch": t_dsm, "position": r_dsm}
    }

    traj = solve_mga_trajectory(
        bodies=bodies,
        epochs=epochs,
        dsm_configs=dsm_configs,
        prograde=[True, True],
    )

    assert len(traj.legs) == 2
    assert traj.total_dsm_delta_v > 0.0
    assert traj.legs[0].delta_v_dsm_mag == traj.total_dsm_delta_v


def test_solve_mga_trajectory_input_validation():
    """Verify input validation errors for invalid body and epoch counts."""
    # Fewer than 2 bodies
    with pytest.raises(ValueError, match="at least 2"):
        solve_mga_trajectory(bodies=["earth"], epochs=["2020-01-01"])

    # Mismatched lengths
    with pytest.raises(ValueError, match="must equal"):
        solve_mga_trajectory(bodies=["earth", "mars"], epochs=["2020-01-01"])

    # Non-increasing epochs
    with pytest.raises(ValueError, match="strictly increasing"):
        solve_mga_trajectory(bodies=["earth", "mars"], epochs=["2021-01-01", "2020-01-01"])

    # Mismatched prograde list length
    with pytest.raises(ValueError, match="prograde list"):
        solve_mga_trajectory(
            bodies=["earth", "venus", "mercury"],
            epochs=["2020-01-01", "2020-06-01", "2021-01-01"],
            prograde=[True],
        )
