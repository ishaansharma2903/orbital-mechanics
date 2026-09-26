"""
tests/bridges/test_gmat.py

Unit tests for the GMAT (General Mission Analysis Tool) script export bridge.
"""
import os
import pytest
import numpy as np

from polaris import (
    get_body_state,
    solve_single_leg,
    solve_dsm_leg,
    solve_mga_trajectory,
    export_to_gmat_script,
    MU_SUN,
    AU,
)


def test_export_single_leg_to_gmat():
    """Verify GMAT script generation for a single-leg interplanetary transfer."""
    transfer = solve_single_leg("earth", "mars", "2020-07-30", "2021-02-18")
    script = export_to_gmat_script(transfer, spacecraft_name="Perseverance")

    assert "Create CoordinateSystem SunICRF;" in script
    assert "Create Spacecraft Perseverance;" in script
    assert "GMAT Perseverance.DateFormat = UTCGregorian;" in script
    assert "30 Jul 2020" in script
    assert "GMAT Perseverance.CoordinateSystem = SunICRF;" in script
    assert "GMAT Perseverance.DisplayStateType = Cartesian;"
    assert "Create Propagator HeliousProp;" in script
    assert "BeginMissionSequence;" in script
    assert "Propagate HeliousProp(Perseverance) {Perseverance.ElapsedDays =" in script


def test_export_dsm_transfer_to_gmat():
    """Verify GMAT script generation for a transfer with a deep-space maneuver."""
    t_dep = "2004-08-03"
    t_arr = "2005-08-02"
    t_dsm = "2004-12-01"
    r_dsm = np.array([1.1 * AU, 0.2 * AU, 0.01 * AU])

    state_dep = get_body_state("earth", t_dep)
    state_arr = get_body_state("earth", t_arr)
    transfer = solve_dsm_leg(
        r1=state_dep.position,
        r2=state_arr.position,
        r_dsm=r_dsm,
        t1=t_dep,
        t_dsm=t_dsm,
        t2=t_arr,
        v_dep_body=state_dep.velocity,
        v_arr_body=state_arr.velocity,
    )
    script = export_to_gmat_script(transfer, spacecraft_name="MESSENGER_DSM")

    assert "Create Spacecraft MESSENGER_DSM;" in script
    assert "Create ImpulsiveBurn DSM_Burn;" in script
    assert "GMAT DSM_Burn.CoordinateSystem = SunICRF;" in script
    assert "GMAT DSM_Burn.Element1 =" in script
    assert "BeginMissionSequence;" in script
    assert "Maneuver DSM_Burn(MESSENGER_DSM);" in script


def test_export_mga_trajectory_to_gmat(tmp_path):
    """Verify GMAT script export for a multi-leg MGA trajectory written to file."""
    bodies = ["earth", "venus", "mercury"]
    epochs = ["2004-08-03", "2005-04-01", "2005-10-01"]

    traj = solve_mga_trajectory(bodies, epochs)
    out_file = tmp_path / "mga_mission.script"

    script = export_to_gmat_script(traj, output_path=str(out_file), spacecraft_name="ProbeMGA")

    assert os.path.exists(out_file)
    with open(out_file, "r") as f:
        content = f.read()

    assert content == script
    assert "Create Spacecraft ProbeMGA;" in content
    assert "BeginMissionSequence;" in content
    # Should contain propagation for each leg
    assert content.count("Propagate HeliousProp(ProbeMGA)") == 2


def test_export_mga_trajectory_with_dsm_and_powered_flyby():
    """Verify GMAT script export for an MGA trajectory with DSM and powered flyby burns."""
    bodies = ["earth", "venus", "mercury"]
    epochs = [2453220.5, 2453500.5, 2453800.5]
    t_dsm = 2453350.0
    r_dsm = np.array([0.9 * AU, 0.4 * AU, 0.02 * AU])

    dsm_configs = {
        0: {"epoch": t_dsm, "position": r_dsm}
    }

    traj = solve_mga_trajectory(
        bodies=bodies,
        epochs=epochs,
        dsm_configs=dsm_configs,
        h_safe_dict={"venus": 10000.0},  # Forces a powered flyby burn
    )

    script = export_to_gmat_script(traj, spacecraft_name="MessengerTest")
    assert "Create ImpulsiveBurn DSM_Leg_0;" in script
    assert "Maneuver DSM_Leg_0(MessengerTest);" in script
    assert "Create ImpulsiveBurn Powered_Flyby_Burn_1;" in script
    assert "Maneuver Powered_Flyby_Burn_1(MessengerTest);" in script


def test_export_invalid_type_raises():
    """Verify TypeError when unsupported object is passed."""
    with pytest.raises(TypeError, match="Unsupported trajectory type"):
        export_to_gmat_script("invalid_object")  # type: ignore

