"""
maneuvers/dsm.py

Deep-space maneuver (DSM) trajectory arc modeling.
Splits an interplanetary leg into two Lambert arcs intersecting at a mid-course
position and epoch, calculating the required impulsive Delta-v vector and magnitude.
"""
from dataclasses import dataclass
from typing import Optional, Union
import datetime
import numpy as np
from astropy.time import Time

from orbital_mechanics.constants import MU_SUN, DAY_TO_SEC
from orbital_mechanics.core.state import State
from orbital_mechanics.ephemeris.horizons import parse_epoch
from orbital_mechanics.maneuvers.lambert import solve_lambert


@dataclass(frozen=True)
class DSMTransfer:
    """
    Result of a deep-space maneuver (DSM) two-subleg trajectory solve.

    Units:
        positions: km
        velocities / delta-v: km/s
        epochs / tof: Julian Date / days
        C3: km^2/s^2
    """
    departure_epoch: float          # Julian Date (days)
    dsm_epoch: float                # Julian Date (days)
    arrival_epoch: float            # Julian Date (days)
    tof_leg1_days: float            # Time of flight from departure to DSM (days)
    tof_leg2_days: float            # Time of flight from DSM to arrival (days)
    total_tof_days: float           # Total transfer duration (days)
    departure_state_craft: State    # Spacecraft state at departure
    dsm_state_in: State             # Spacecraft state arriving at DSM point (pre-burn)
    dsm_state_out: State            # Spacecraft state departing DSM point (post-burn)
    arrival_state_craft: State      # Spacecraft state at arrival
    delta_v_dsm: np.ndarray         # Impulsive DSM Delta-v vector (km/s)
    delta_v_dsm_mag: float          # Magnitude of DSM Delta-v (km/s)
    v_inf_dep: Optional[np.ndarray] = None  # Departure hyperbolic excess velocity vector (km/s)
    v_inf_dep_mag: Optional[float] = None   # Departure excess speed (km/s)
    v_inf_arr: Optional[np.ndarray] = None  # Arrival hyperbolic excess velocity vector (km/s)
    v_inf_arr_mag: Optional[float] = None   # Arrival excess speed (km/s)
    c3: Optional[float] = None              # Characteristic launch energy (km^2/s^2)
    mu: float = MU_SUN


def solve_dsm_leg(
    r1: Union[np.ndarray, State],
    r2: Union[np.ndarray, State],
    r_dsm: Union[np.ndarray, State],
    t1: Union[float, int, str, datetime.date, datetime.datetime, Time],
    t_dsm: Union[float, int, str, datetime.date, datetime.datetime, Time],
    t2: Union[float, int, str, datetime.date, datetime.datetime, Time],
    v_dep_body: Optional[Union[np.ndarray, State]] = None,
    v_arr_body: Optional[Union[np.ndarray, State]] = None,
    prograde1: bool = True,
    prograde2: bool = True,
    mu: float = MU_SUN,
) -> DSMTransfer:
    """
    Solve a single interplanetary leg with an intermediate deep-space maneuver.

    Parameters:
        r1: Departure position vector (km) or State.
        r2: Arrival position vector (km) or State.
        r_dsm: Mid-course DSM position vector (km) or State.
        t1: Departure epoch (JD float, ISO string, or datetime).
        t_dsm: DSM epoch (JD float, ISO string, or datetime).
        t2: Arrival epoch (JD float, ISO string, or datetime).
        v_dep_body: Optional departure body velocity vector (km/s) or State for v_inf/C3.
        v_arr_body: Optional arrival body velocity vector (km/s) or State for v_inf.
        prograde1: Direction for Leg 1 (default: True).
        prograde2: Direction for Leg 2 (default: True).
        mu: Gravitational parameter of central body (default: MU_SUN).

    Returns:
        DSMTransfer: Complete DSM solution with states, Delta-v, and excess velocities.
    """
    jd1 = parse_epoch(t1)
    jd_dsm = parse_epoch(t_dsm)
    jd2 = parse_epoch(t2)

    if not (jd1 < jd_dsm < jd2):
        raise ValueError(
            f"DSM epoch ({jd_dsm:.3f}) must be strictly between departure ({jd1:.3f}) "
            f"and arrival ({jd2:.3f}) epochs."
        )

    tof1_days = jd_dsm - jd1
    tof2_days = jd2 - jd_dsm
    total_tof_days = jd2 - jd1

    tof1_sec = tof1_days * DAY_TO_SEC
    tof2_sec = tof2_days * DAY_TO_SEC

    # Extract position vectors
    r1_vec = r1.position if isinstance(r1, State) else np.asarray(r1, dtype=float)
    r2_vec = r2.position if isinstance(r2, State) else np.asarray(r2, dtype=float)
    rdsm_vec = r_dsm.position if isinstance(r_dsm, State) else np.asarray(r_dsm, dtype=float)

    # Solve Lambert for Leg 1: r1 -> r_dsm
    v1_craft, v_dsm_in = solve_lambert(
        r1=r1_vec,
        r2=rdsm_vec,
        tof=tof1_sec,
        prograde=prograde1,
        mu=mu,
    )

    # Solve Lambert for Leg 2: r_dsm -> r2
    v_dsm_out, v2_craft = solve_lambert(
        r1=rdsm_vec,
        r2=r2_vec,
        tof=tof2_sec,
        prograde=prograde2,
        mu=mu,
    )

    # Compute DSM impulsive Delta-v
    delta_v_dsm = v_dsm_out - v_dsm_in
    delta_v_dsm_mag = float(np.linalg.norm(delta_v_dsm))

    # Construct spacecraft states
    s_dep_craft = State(position=r1_vec, velocity=v1_craft)
    s_dsm_in = State(position=rdsm_vec, velocity=v_dsm_in)
    s_dsm_out = State(position=rdsm_vec, velocity=v_dsm_out)
    s_arr_craft = State(position=r2_vec, velocity=v2_craft)

    # Optional departure excess velocity and C3
    v_inf_dep = None
    v_inf_dep_mag = None
    c3 = None
    if v_dep_body is not None:
        v_body_vec = v_dep_body.velocity if isinstance(v_dep_body, State) else np.asarray(v_dep_body, dtype=float)
        v_inf_dep = v1_craft - v_body_vec
        v_inf_dep_mag = float(np.linalg.norm(v_inf_dep))
        c3 = float(v_inf_dep_mag ** 2)

    # Optional arrival excess velocity
    v_inf_arr = None
    v_inf_arr_mag = None
    if v_arr_body is not None:
        v_arr_vec = v_arr_body.velocity if isinstance(v_arr_body, State) else np.asarray(v_arr_body, dtype=float)
        v_inf_arr = v2_craft - v_arr_vec
        v_inf_arr_mag = float(np.linalg.norm(v_inf_arr))

    return DSMTransfer(
        departure_epoch=jd1,
        dsm_epoch=jd_dsm,
        arrival_epoch=jd2,
        tof_leg1_days=tof1_days,
        tof_leg2_days=tof2_days,
        total_tof_days=total_tof_days,
        departure_state_craft=s_dep_craft,
        dsm_state_in=s_dsm_in,
        dsm_state_out=s_dsm_out,
        arrival_state_craft=s_arr_craft,
        delta_v_dsm=delta_v_dsm,
        delta_v_dsm_mag=delta_v_dsm_mag,
        v_inf_dep=v_inf_dep,
        v_inf_dep_mag=v_inf_dep_mag,
        v_inf_arr=v_inf_arr,
        v_inf_arr_mag=v_inf_arr_mag,
        c3=c3,
        mu=mu,
    )
