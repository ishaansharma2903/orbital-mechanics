"""
tests/maneuvers/test_dsm.py

Unit and validation tests for deep-space maneuver (DSM) transfers.
"""
import pytest
import numpy as np

from polaris.constants import MU_SUN, AU
from polaris.core.state import State
from polaris.maneuvers.dsm import DSMTransfer, solve_dsm_leg
from polaris.maneuvers.lambert import solve_lambert


def test_solve_dsm_leg_broken_plane():
    """
    Validate DSM calculation for a broken-plane interplanetary trajectory
    connecting two heliocentric position vectors with an intermediate maneuver point.
    """
    r1 = np.array([AU, 0.0, 0.0])
    # Out of ecliptic plane DSM position
    r_dsm = np.array([1.2 * AU * np.cos(np.radians(45.0)), 1.2 * AU * np.sin(np.radians(45.0)), 0.05 * AU])
    r2 = np.array([0.0, 1.524 * AU, 0.0])

    t1 = 2451545.0
    t_dsm = 2451545.0 + 50.0   # 50 days to DSM
    t2 = 2451545.0 + 120.0     # 70 days from DSM to arrival

    transfer = solve_dsm_leg(
        r1=r1,
        r2=r2,
        r_dsm=r_dsm,
        t1=t1,
        t_dsm=t_dsm,
        t2=t2,
        mu=MU_SUN,
    )

    assert isinstance(transfer, DSMTransfer)
    assert abs(transfer.tof_leg1_days - 50.0) < 1e-8
    assert abs(transfer.tof_leg2_days - 70.0) < 1e-8
    assert abs(transfer.total_tof_days - 120.0) < 1e-8

    np.testing.assert_allclose(transfer.departure_state_craft.position, r1)
    np.testing.assert_allclose(transfer.dsm_state_in.position, r_dsm)
    np.testing.assert_allclose(transfer.dsm_state_out.position, r_dsm)
    np.testing.assert_allclose(transfer.arrival_state_craft.position, r2)

    # Verify vector delta-v equals velocity difference across DSM point
    delta_v_expected = transfer.dsm_state_out.velocity - transfer.dsm_state_in.velocity
    np.testing.assert_allclose(transfer.delta_v_dsm, delta_v_expected)
    assert abs(transfer.delta_v_dsm_mag - np.linalg.norm(delta_v_expected)) < 1e-10
    assert transfer.delta_v_dsm_mag > 0.0

    # Verify physical invariants: orbital energy on Leg 1
    oe_leg1_dep = transfer.departure_state_craft.to_orbit_elements(mu=MU_SUN)
    oe_leg1_dsm = transfer.dsm_state_in.to_orbit_elements(mu=MU_SUN)
    assert abs((oe_leg1_dep.a - oe_leg1_dsm.a) / oe_leg1_dep.a) < 1e-5
    assert abs(oe_leg1_dep.e - oe_leg1_dsm.e) < 1e-5

    # Verify physical invariants: orbital energy on Leg 2
    oe_leg2_dsm = transfer.dsm_state_out.to_orbit_elements(mu=MU_SUN)
    oe_leg2_arr = transfer.arrival_state_craft.to_orbit_elements(mu=MU_SUN)
    assert abs((oe_leg2_dsm.a - oe_leg2_arr.a) / oe_leg2_dsm.a) < 1e-5
    assert abs(oe_leg2_dsm.e - oe_leg2_arr.e) < 1e-5


def test_solve_dsm_leg_with_body_velocities():
    """Verify v_inf and C3 calculations when departure/arrival velocities are provided."""
    r1 = np.array([AU, 0.0, 0.0])
    v_body1 = np.array([0.0, np.sqrt(MU_SUN / AU), 0.0])

    r_dsm = np.array([1.1 * AU, 0.5 * AU, 0.0])
    r2 = np.array([0.0, 1.4 * AU, 0.0])
    v_body2 = np.array([-np.sqrt(MU_SUN / (1.4 * AU)), 0.0, 0.0])

    s1 = State(position=r1, velocity=v_body1)
    s2 = State(position=r2, velocity=v_body2)

    transfer = solve_dsm_leg(
        r1=s1,
        r2=s2,
        r_dsm=r_dsm,
        t1="2025-01-01",
        t_dsm="2025-03-01",
        t2="2025-06-01",
        v_dep_body=s1,
        v_arr_body=s2,
    )

    assert transfer.v_inf_dep is not None
    assert transfer.v_inf_arr is not None
    assert transfer.c3 is not None
    assert abs(transfer.c3 - transfer.v_inf_dep_mag ** 2) < 1e-10
    np.testing.assert_allclose(
        transfer.v_inf_dep,
        transfer.departure_state_craft.velocity - v_body1,
    )
    np.testing.assert_allclose(
        transfer.v_inf_arr,
        transfer.arrival_state_craft.velocity - v_body2,
    )


def test_solve_dsm_leg_invalid_epoch_order():
    """Verify ValueError is raised if t_dsm is not strictly between t1 and t2."""
    r = np.array([AU, 0.0, 0.0])
    with pytest.raises(ValueError, match="strictly between"):
        solve_dsm_leg(r, r, r, t1=2451545.0, t_dsm=2451540.0, t2=2451550.0)

    with pytest.raises(ValueError, match="strictly between"):
        solve_dsm_leg(r, r, r, t1=2451545.0, t_dsm=2451555.0, t2=2451550.0)
