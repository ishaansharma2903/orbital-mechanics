"""
transfers/porkchop.py

Launch window porkchop plot grid generation for interplanetary transfers.
Evaluates 2D parameter spaces of departure and arrival dates, calculating
characteristic launch energy (C3), arrival hyperbolic excess speeds,
times of flight, and total mission Delta-v.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import datetime
import numpy as np
from astropy.time import Time

from orbital_mechanics.constants import MU_SUN, DAY_TO_SEC
from orbital_mechanics.ephemeris.horizons import parse_epoch, get_body_state, get_body_states_batch
from orbital_mechanics.maneuvers.lambert import solve_lambert


@dataclass(frozen=True)
class PorkchopResult:
    """
    2D grid evaluation result for interplanetary launch windows.

    Arrays are 2D grids of shape (len(dep_epochs), len(arr_epochs)).
    Units:
        epochs / tof: Julian Date / days
        C3: km^2/s^2
        velocities / delta-v: km/s
    """
    departure_body: str
    arrival_body: str
    dep_epochs: np.ndarray          # 1D array of departure Julian Dates
    arr_epochs: np.ndarray          # 1D array of arrival Julian Dates
    tof_days: np.ndarray            # 2D grid of times of flight (days)
    c3: np.ndarray                  # 2D grid of characteristic launch energy (km^2/s^2)
    v_inf_dep: np.ndarray           # 2D grid of departure hyperbolic excess speeds (km/s)
    v_inf_arr: np.ndarray           # 2D grid of arrival hyperbolic excess speeds (km/s)
    total_delta_v: np.ndarray       # 2D grid of total mission Delta-v (km/s)

    def min_c3(self) -> Tuple[float, float, float]:
        """
        Find the minimum C3 opportunity in the launch window grid.

        Returns:
            Tuple[float, float, float]: (min_c3_value, best_dep_jd, best_arr_jd).
        """
        min_idx = np.unravel_index(np.nanargmin(self.c3), self.c3.shape)
        return (
            float(self.c3[min_idx]),
            float(self.dep_epochs[min_idx[0]]),
            float(self.arr_epochs[min_idx[1]]),
        )

    def min_delta_v(self) -> Tuple[float, float, float]:
        """
        Find the minimum total Delta-v opportunity in the launch window grid.

        Returns:
            Tuple[float, float, float]: (min_delta_v_value, best_dep_jd, best_arr_jd).
        """
        min_idx = np.unravel_index(np.nanargmin(self.total_delta_v), self.total_delta_v.shape)
        return (
            float(self.total_delta_v[min_idx]),
            float(self.dep_epochs[min_idx[0]]),
            float(self.arr_epochs[min_idx[1]]),
        )


def generate_porkchop(
    departure_body: Union[str, int],
    arrival_body: Union[str, int],
    dep_epoch_range: Tuple[Union[float, str, datetime.date], Union[float, str, datetime.date]],
    arr_epoch_range: Tuple[Union[float, str, datetime.date], Union[float, str, datetime.date]],
    num_dep: int = 30,
    num_arr: int = 30,
    r_park_dep: Optional[float] = None,
    mu_dep: Optional[float] = None,
    r_park_arr: Optional[float] = None,
    mu_arr: Optional[float] = None,
    prograde: bool = True,
    mu: float = MU_SUN,
) -> PorkchopResult:
    """
    Generate a 2D porkchop plot grid across departure and arrival epoch ranges.

    Parameters:
        departure_body: Departure body name or NAIF/Horizons ID.
        arrival_body: Arrival body name or NAIF/Horizons ID.
        dep_epoch_range: (start_dep_epoch, end_dep_epoch).
        arr_epoch_range: (start_arr_epoch, end_arr_epoch).
        num_dep: Number of departure date grid samples (default: 30).
        num_arr: Number of arrival date grid samples (default: 30).
        r_park_dep: Optional circular parking orbit radius at departure (km).
        mu_dep: Optional departure body gravitational parameter (km^3/s^2).
        r_park_arr: Optional circular parking orbit radius at arrival (km).
        mu_arr: Optional arrival body gravitational parameter (km^3/s^2).
        prograde: Transfer trajectory direction (default: True).
        mu: Gravitational parameter of central body (default: MU_SUN).

    Returns:
        PorkchopResult: 2D grid dataset containing C3, v_inf, TOF, and Delta-v.
    """
    dep_start = parse_epoch(dep_epoch_range[0])
    dep_end = parse_epoch(dep_epoch_range[1])
    arr_start = parse_epoch(arr_epoch_range[0])
    arr_end = parse_epoch(arr_epoch_range[1])

    if dep_end <= dep_start:
        raise ValueError("dep_epoch_range end must be strictly after start.")
    if arr_end <= arr_start:
        raise ValueError("arr_epoch_range end must be strictly after start.")
    if num_dep < 2 or num_arr < 2:
        raise ValueError("num_dep and num_arr must be at least 2.")

    dep_epochs = np.linspace(dep_start, dep_end, num_dep)
    arr_epochs = np.linspace(arr_start, arr_end, num_arr)

    # Pre-fetch ephemeris states for each grid date in single batch queries
    dep_states = get_body_states_batch(departure_body, dep_epochs)
    arr_states = get_body_states_batch(arrival_body, arr_epochs)

    tof_grid = np.zeros((num_dep, num_arr))
    c3_grid = np.full((num_dep, num_arr), np.nan)
    v_inf_dep_grid = np.full((num_dep, num_arr), np.nan)
    v_inf_arr_grid = np.full((num_dep, num_arr), np.nan)
    dv_total_grid = np.full((num_dep, num_arr), np.nan)

    for i, (t_dep, s_dep) in enumerate(zip(dep_epochs, dep_states)):
        for j, (t_arr, s_arr) in enumerate(zip(arr_epochs, arr_states)):
            dt_days = t_arr - t_dep
            tof_grid[i, j] = dt_days

            if dt_days <= 0.0:
                continue

            dt_sec = dt_days * DAY_TO_SEC
            try:
                v1_craft, v2_craft = solve_lambert(
                    r1=s_dep.position,
                    r2=s_arr.position,
                    tof=dt_sec,
                    prograde=prograde,
                    mu=mu,
                )
            except (ValueError, RuntimeError):
                # Collinear or non-converging transfer geometry
                continue

            v_inf_dep_vec = v1_craft - s_dep.velocity
            v_inf_arr_vec = v2_craft - s_arr.velocity

            v_inf_dep_mag = float(np.linalg.norm(v_inf_dep_vec))
            v_inf_arr_mag = float(np.linalg.norm(v_inf_arr_vec))
            c3_val = v_inf_dep_mag ** 2

            v_inf_dep_grid[i, j] = v_inf_dep_mag
            v_inf_arr_grid[i, j] = v_inf_arr_mag
            c3_grid[i, j] = c3_val

            # Compute mission Delta-v
            if r_park_dep is not None and mu_dep is not None:
                v_circ_dep = np.sqrt(mu_dep / r_park_dep)
                dv_dep = float(np.sqrt(v_inf_dep_mag ** 2 + 2.0 * mu_dep / r_park_dep) - v_circ_dep)
            else:
                dv_dep = v_inf_dep_mag

            if r_park_arr is not None and mu_arr is not None:
                v_circ_arr = np.sqrt(mu_arr / r_park_arr)
                dv_arr = float(np.sqrt(v_inf_arr_mag ** 2 + 2.0 * mu_arr / r_park_arr) - v_circ_arr)
            else:
                dv_arr = v_inf_arr_mag

            dv_total_grid[i, j] = dv_dep + dv_arr

    return PorkchopResult(
        departure_body=str(departure_body),
        arrival_body=str(arrival_body),
        dep_epochs=dep_epochs,
        arr_epochs=arr_epochs,
        tof_days=tof_grid,
        c3=c3_grid,
        v_inf_dep=v_inf_dep_grid,
        v_inf_arr=v_inf_arr_grid,
        total_delta_v=dv_total_grid,
    )
