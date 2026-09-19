"""
transfers/single_leg.py

Patched-conic single-leg interplanetary transfer solver.
Combines planetary ephemeris (JPL Horizons) with Lambert two-body transfer
to compute trajectory state vectors, hyperbolic excess velocities (v_inf),
characteristic launch energy (C3), and impulsive delta-v budgets.
"""
from dataclasses import dataclass
from typing import Union, Optional
import datetime
import numpy as np
from astropy.time import Time

from orbital_mechanics.constants import MU_SUN, DAY_TO_SEC
from orbital_mechanics.core.state import State
from orbital_mechanics.ephemeris.horizons import get_body_state, parse_epoch
from orbital_mechanics.maneuvers.lambert import solve_lambert


@dataclass(frozen=True)
class SingleLegTransfer:
    """
    Result of a single-leg patched-conic interplanetary transfer.

    All states and velocities are in the heliocentric ecliptic frame.
    Units:
        distance: km
        velocity: km/s
        epochs & tof: Julian Date / days
        C3: km^2/s^2
    """
    departure_body: str
    arrival_body: str
    departure_epoch: float      # Julian Date (days)
    arrival_epoch: float        # Julian Date (days)
    tof_days: float             # Time of flight (days)
    departure_state_body: State
    arrival_state_body: State
    departure_state_craft: State
    arrival_state_craft: State
    v_inf_dep: np.ndarray       # Departure hyperbolic excess velocity vector (km/s)
    v_inf_arr: np.ndarray       # Arrival hyperbolic excess velocity vector (km/s)
    c3: float                   # Characteristic energy = ||v_inf_dep||^2 (km^2/s^2)
    v_inf_dep_mag: float        # Magnitude of v_inf_dep (km/s)
    v_inf_arr_mag: float        # Magnitude of v_inf_arr (km/s)
    mu: float = MU_SUN

    def departure_delta_v(self, r_park: float, mu_body: float) -> float:
        """
        Compute the departure burn Delta-v from a circular parking orbit.

        Parameters:
            r_park: Parking orbit radius from body center (km).
            mu_body: Gravitational parameter of departure body (km^3/s^2).

        Returns:
            float: Impulsive Delta-v (km/s).
        """
        v_circ = np.sqrt(mu_body / r_park)
        v_hyp = np.sqrt(self.v_inf_dep_mag ** 2 + 2.0 * mu_body / r_park)
        return float(v_hyp - v_circ)

    def arrival_delta_v(self, r_park: float, mu_body: float) -> float:
        """
        Compute the orbit-insertion capture burn Delta-v into a circular parking orbit.

        Parameters:
            r_park: Parking orbit radius from body center (km).
            mu_body: Gravitational parameter of arrival body (km^3/s^2).

        Returns:
            float: Impulsive Delta-v (km/s).
        """
        v_circ = np.sqrt(mu_body / r_park)
        v_hyp = np.sqrt(self.v_inf_arr_mag ** 2 + 2.0 * mu_body / r_park)
        return float(v_hyp - v_circ)

    def total_delta_v(
        self,
        r_park_dep: float,
        mu_dep: float,
        r_park_arr: float,
        mu_arr: float,
    ) -> float:
        """
        Compute total mission Delta-v (departure ejection + arrival capture).

        Parameters:
            r_park_dep: Departure parking orbit radius (km).
            mu_dep: Gravitational parameter of departure body (km^3/s^2).
            r_park_arr: Arrival parking orbit radius (km).
            mu_arr: Gravitational parameter of arrival body (km^3/s^2).

        Returns:
            float: Total impulsive Delta-v (km/s).
        """
        dv_dep = self.departure_delta_v(r_park=r_park_dep, mu_body=mu_dep)
        dv_arr = self.arrival_delta_v(r_park=r_park_arr, mu_body=mu_arr)
        return dv_dep + dv_arr


def solve_single_leg(
    departure_body: Union[str, int, State],
    arrival_body: Union[str, int, State],
    departure_epoch: Union[float, int, str, datetime.date, datetime.datetime, Time],
    arrival_epoch: Union[float, int, str, datetime.date, datetime.datetime, Time],
    prograde: bool = True,
    mu: float = MU_SUN,
    center: Union[str, int] = "sun",
    refplane: str = "ecliptic",
) -> SingleLegTransfer:
    """
    Solve a single-leg patched-conic interplanetary transfer between two bodies.

    Parameters:
        departure_body: Departure body name, ID, or custom State.
        arrival_body: Arrival body name, ID, or custom State.
        departure_epoch: Departure epoch (JD float, ISO string, datetime, or Time).
        arrival_epoch: Arrival epoch (JD float, ISO string, datetime, or Time).
        prograde: Transfer trajectory direction (default: True).
        mu: Gravitational parameter of central body (default: MU_SUN).
        center: Observer coordinate origin for ephemeris retrieval (default: 'sun').
        refplane: Reference plane for ephemeris retrieval (default: 'ecliptic').

    Returns:
        SingleLegTransfer: Complete transfer solution with states, v_inf, and C3.
    """
    jd_dep = parse_epoch(departure_epoch)
    jd_arr = parse_epoch(arrival_epoch)
    tof_days = jd_arr - jd_dep

    if tof_days <= 0.0:
        raise ValueError(
            f"Arrival epoch ({jd_arr:.3f}) must be strictly after departure epoch ({jd_dep:.3f})."
        )

    tof_sec = tof_days * DAY_TO_SEC

    # Get departure body state
    if isinstance(departure_body, State):
        s_dep_body = departure_body
        dep_name = "custom"
    else:
        s_dep_body = get_body_state(
            body=departure_body,
            epoch=jd_dep,
            center=center,
            refplane=refplane,
        )
        dep_name = str(departure_body)

    # Get arrival body state
    if isinstance(arrival_body, State):
        s_arr_body = arrival_body
        arr_name = "custom"
    else:
        s_arr_body = get_body_state(
            body=arrival_body,
            epoch=jd_arr,
            center=center,
            refplane=refplane,
        )
        arr_name = str(arrival_body)

    # Solve Lambert transfer
    v1_craft, v2_craft = solve_lambert(
        r1=s_dep_body.position,
        r2=s_arr_body.position,
        tof=tof_sec,
        prograde=prograde,
        mu=mu,
    )

    s_dep_craft = State(position=s_dep_body.position, velocity=v1_craft)
    s_arr_craft = State(position=s_arr_body.position, velocity=v2_craft)

    # Compute hyperbolic excess velocities
    v_inf_dep = v1_craft - s_dep_body.velocity
    v_inf_dep_mag = float(np.linalg.norm(v_inf_dep))
    c3 = float(v_inf_dep_mag ** 2)

    v_inf_arr = v2_craft - s_arr_body.velocity
    v_inf_arr_mag = float(np.linalg.norm(v_inf_arr))

    return SingleLegTransfer(
        departure_body=dep_name,
        arrival_body=arr_name,
        departure_epoch=jd_dep,
        arrival_epoch=jd_arr,
        tof_days=tof_days,
        departure_state_body=s_dep_body,
        arrival_state_body=s_arr_body,
        departure_state_craft=s_dep_craft,
        arrival_state_craft=s_arr_craft,
        v_inf_dep=v_inf_dep,
        v_inf_arr=v_inf_arr,
        c3=c3,
        v_inf_dep_mag=v_inf_dep_mag,
        v_inf_arr_mag=v_inf_arr_mag,
        mu=mu,
    )
