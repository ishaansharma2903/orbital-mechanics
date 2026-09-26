"""
transfers/multi_leg.py

Multi-gravity-assist (MGA) trajectory assembly and Delta-v budgeting.
Chains multiple Lambert / DSM legs and planetary gravity assists into a unified
mission trajectory, evaluating intermediate flyby kinematics and total Delta-v.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import datetime
import numpy as np
from astropy.time import Time

from orbital_mechanics.constants import (
    MU_SUN,
    MU_EARTH,
    MU_VENUS,
    MU_MERCURY,
    MU_MARS,
    R_EARTH,
    R_VENUS,
    R_MERCURY,
    R_MARS,
    DAY_TO_SEC,
)
from orbital_mechanics.core.state import State
from orbital_mechanics.ephemeris.horizons import parse_epoch, get_body_state
from orbital_mechanics.maneuvers.dsm import DSMTransfer, solve_dsm_leg
from orbital_mechanics.maneuvers.flyby import FlybyResult, solve_flyby
from orbital_mechanics.transfers.single_leg import SingleLegTransfer, solve_single_leg

# Standard planetary body physical properties (mu, equatorial_radius)
PLANETARY_PARAMS: Dict[str, Tuple[float, float]] = {
    "mercury": (MU_MERCURY, R_MERCURY),
    "199": (MU_MERCURY, R_MERCURY),
    "1": (MU_MERCURY, R_MERCURY),
    "venus": (MU_VENUS, R_VENUS),
    "299": (MU_VENUS, R_VENUS),
    "2": (MU_VENUS, R_VENUS),
    "earth": (MU_EARTH, R_EARTH),
    "399": (MU_EARTH, R_EARTH),
    "3": (MU_EARTH, R_EARTH),
    "mars": (MU_MARS, R_MARS),
    "499": (MU_MARS, R_MARS),
    "4": (MU_MARS, R_MARS),
    "jupiter": (1.26686534e8, 71492.0),
    "599": (1.26686534e8, 71492.0),
    "5": (1.26686534e8, 71492.0),
    "saturn": (3.7931187e7, 60268.0),
    "699": (3.7931187e7, 60268.0),
    "6": (3.7931187e7, 60268.0),
}


def get_body_params(body: Union[str, int]) -> Tuple[float, float]:
    """
    Retrieve standard gravitational parameter (km^3/s^2) and radius (km)
    for a planetary body.
    """
    body_key = str(body).strip().lower()
    if body_key in PLANETARY_PARAMS:
        return PLANETARY_PARAMS[body_key]
    raise KeyError(
        f"Body '{body}' not found in default planetary parameters. "
        "Please provide explicit mu and radius."
    )


@dataclass(frozen=True)
class MGATrajectory:
    """
    Complete Multi-Gravity-Assist (MGA) trajectory solution.

    Units:
        velocities / delta-v: km/s
        epochs / tof: Julian Date / days
        C3: km^2/s^2
    """
    bodies: List[str]
    epochs: List[float]                                     # Encounter epochs as Julian Dates
    tof_total_days: float                                   # Total mission duration from launch to final arrival
    legs: List[Union[SingleLegTransfer, DSMTransfer]]       # Leg-by-leg transfer solutions
    flybys: List[FlybyResult]                               # Flyby results at intermediate encounters
    c3: float                                               # Characteristic launch energy (km^2/s^2)
    v_inf_dep_mag: float                                    # Departure hyperbolic excess speed (km/s)
    v_inf_arr_mag: float                                    # Final arrival hyperbolic excess speed (km/s)
    total_dsm_delta_v: float                                # Sum of all deep-space maneuver impulses (km/s)
    total_flyby_delta_v: float                             # Sum of all powered flyby periapsis burns (km/s)
    is_feasible: bool                                       # True if all intermediate flybys are altitude-safe
    mu: float = MU_SUN

    def total_mission_delta_v(
        self,
        r_park_dep: Optional[float] = None,
        mu_dep: Optional[float] = None,
        r_park_arr: Optional[float] = None,
        mu_arr: Optional[float] = None,
    ) -> float:
        """
        Calculate total mission Delta-v budget including optional departure ejection,
        intermediate DSMs, powered flybys, and destination orbit insertion.

        Parameters:
            r_park_dep: Optional departure parking orbit radius (km).
            mu_dep: Optional departure body mu (km^3/s^2).
            r_park_arr: Optional arrival parking orbit radius (km).
            mu_arr: Optional arrival body mu (km^3/s^2).

        Returns:
            float: Total mission impulsive Delta-v (km/s).
        """
        dv_total = self.total_dsm_delta_v + self.total_flyby_delta_v

        if r_park_dep is not None and mu_dep is not None:
            v_circ = np.sqrt(mu_dep / r_park_dep)
            v_hyp = np.sqrt(self.v_inf_dep_mag ** 2 + 2.0 * mu_dep / r_park_dep)
            dv_total += float(v_hyp - v_circ)

        if r_park_arr is not None and mu_arr is not None:
            v_circ = np.sqrt(mu_arr / r_park_arr)
            v_hyp = np.sqrt(self.v_inf_arr_mag ** 2 + 2.0 * mu_arr / r_park_arr)
            dv_total += float(v_hyp - v_circ)

        return dv_total


def solve_mga_trajectory(
    bodies: List[str],
    epochs: List[Union[float, int, str, datetime.date, datetime.datetime, Time]],
    dsm_configs: Optional[Dict[int, Dict[str, Union[float, str, np.ndarray]]]] = None,
    h_safe_dict: Optional[Dict[str, float]] = None,
    prograde: Union[bool, List[bool]] = True,
    mu: float = MU_SUN,
) -> MGATrajectory:
    """
    Assemble and solve an end-to-end Multi-Gravity-Assist (MGA) trajectory.

    Parameters:
        bodies: List of N encounter body names (e.g. ['earth', 'venus', 'venus', 'mercury']).
        epochs: List of N encounter epochs (JD float, ISO string, datetime, or Time).
        dsm_configs: Optional dict mapping leg index (0 to N-2) to DSM parameters:
                     {'epoch': <epoch>, 'position': np.ndarray}.
        h_safe_dict: Optional dict mapping body names to minimum safe altitudes (km).
        prograde: Transfer direction boolean or list of booleans per leg.
        mu: Gravitational parameter of central body (default: MU_SUN).

    Returns:
        MGATrajectory: Complete MGA solution with all legs, flybys, and Delta-v breakdown.
    """
    n_encounters = len(bodies)
    if n_encounters < 2:
        raise ValueError("MGA trajectory requires at least 2 encounter bodies.")

    if len(epochs) != n_encounters:
        raise ValueError(
            f"Number of epochs ({len(epochs)}) must equal number of bodies ({n_encounters})."
        )

    parsed_epochs = [parse_epoch(ep) for ep in epochs]

    # Verify chronological epoch progression
    for i in range(len(parsed_epochs) - 1):
        if parsed_epochs[i + 1] <= parsed_epochs[i]:
            raise ValueError(
                f"Encounter epochs must be strictly increasing. "
                f"Epoch[{i}]={parsed_epochs[i]:.3f} >= Epoch[{i+1}]={parsed_epochs[i+1]:.3f}"
            )

    n_legs = n_encounters - 1
    if isinstance(prograde, bool):
        prograde_list = [prograde] * n_legs
    else:
        if len(prograde) != n_legs:
            raise ValueError(f"Length of prograde list ({len(prograde)}) must match number of legs ({n_legs}).")
        prograde_list = prograde

    dsm_map = dsm_configs if dsm_configs is not None else {}
    h_safe_map = h_safe_dict if h_safe_dict is not None else {}

    legs: List[Union[SingleLegTransfer, DSMTransfer]] = []

    # Solve all interplanetary legs
    for k in range(n_legs):
        body_dep = bodies[k]
        body_arr = bodies[k + 1]
        t_dep = parsed_epochs[k]
        t_arr = parsed_epochs[k + 1]
        prog = prograde_list[k]

        s_dep = get_body_state(body=body_dep, epoch=t_dep)
        s_arr = get_body_state(body=body_arr, epoch=t_arr)

        if k in dsm_map:
            dsm_info = dsm_map[k]
            t_dsm = dsm_info["epoch"]
            r_dsm = dsm_info["position"]
            leg_transfer = solve_dsm_leg(
                r1=s_dep,
                r2=s_arr,
                r_dsm=r_dsm,
                t1=t_dep,
                t_dsm=t_dsm,
                t2=t_arr,
                v_dep_body=s_dep,
                v_arr_body=s_arr,
                prograde1=prog,
                prograde2=prog,
                mu=mu,
            )
        else:
            leg_transfer = solve_single_leg(
                departure_body=s_dep,
                arrival_body=s_arr,
                departure_epoch=t_dep,
                arrival_epoch=t_arr,
                prograde=prog,
                mu=mu,
            )
        legs.append(leg_transfer)

    # Evaluate intermediate flybys (at bodies 1 to N-2)
    flybys: List[FlybyResult] = []
    for k in range(1, n_encounters - 1):
        flyby_body = bodies[k]
        t_fb = parsed_epochs[k]
        mu_body, r_body = get_body_params(flyby_body)
        h_safe = h_safe_map.get(flyby_body.lower(), 100.0)

        # Inbound velocity from previous leg
        v_in = legs[k - 1].arrival_state_craft.velocity

        # Outbound velocity for next leg
        v_out = legs[k].departure_state_craft.velocity

        # Planet heliocentric velocity
        s_planet = get_body_state(body=flyby_body, epoch=t_fb)
        v_body = s_planet.velocity

        fb_result = solve_flyby(
            v_in=v_in,
            v_out=v_out,
            v_body=v_body,
            mu_body=mu_body,
            r_body=r_body,
            h_safe=h_safe,
        )
        flybys.append(fb_result)

    # Calculate overall metrics
    first_leg = legs[0]
    last_leg = legs[-1]

    c3 = first_leg.c3 if first_leg.c3 is not None else float(np.linalg.norm(first_leg.v_inf_dep) ** 2)
    v_inf_dep_mag = first_leg.v_inf_dep_mag if first_leg.v_inf_dep_mag is not None else float(np.linalg.norm(first_leg.v_inf_dep))
    v_inf_arr_mag = last_leg.v_inf_arr_mag if last_leg.v_inf_arr_mag is not None else float(np.linalg.norm(last_leg.v_inf_arr))

    total_dsm_delta_v = float(sum(
        leg.delta_v_dsm_mag for leg in legs if isinstance(leg, DSMTransfer)
    ))
    total_flyby_delta_v = float(sum(fb.delta_v_peri for fb in flybys))
    is_feasible = all(fb.is_feasible for fb in flybys) if flybys else True
    tof_total_days = parsed_epochs[-1] - parsed_epochs[0]

    return MGATrajectory(
        bodies=[str(b) for b in bodies],
        epochs=parsed_epochs,
        tof_total_days=tof_total_days,
        legs=legs,
        flybys=flybys,
        c3=c3,
        v_inf_dep_mag=v_inf_dep_mag,
        v_inf_arr_mag=v_inf_arr_mag,
        total_dsm_delta_v=total_dsm_delta_v,
        total_flyby_delta_v=total_flyby_delta_v,
        is_feasible=is_feasible,
        mu=mu,
    )
