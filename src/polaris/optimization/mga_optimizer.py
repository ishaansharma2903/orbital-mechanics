"""
optimization/mga_optimizer.py

Optimization layer for multi-gravity-assist (MGA) trajectories.
Refines launch and flyby encounter epochs to minimize total mission Delta-v
and powered gravity-assist penalties using numerical optimization.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import datetime
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from astropy.time import Time

from polaris.constants import MU_SUN
from polaris.core.state import State
from polaris.ephemeris.horizons import parse_epoch, get_body_states_batch
from polaris.maneuvers.dsm import solve_dsm_leg, DSMTransfer
from polaris.maneuvers.flyby import solve_flyby
from polaris.transfers.single_leg import solve_single_leg
from polaris.transfers.multi_leg import (
    MGATrajectory,
    solve_mga_trajectory,
    get_body_params,
)


@dataclass(frozen=True)
class MGAOptimizationResult:
    """
    Result of an MGA encounter epoch optimization.

    Units:
        epochs / tof: Julian Date / days
        delta-v: km/s
    """
    optimized_trajectory: MGATrajectory
    initial_epochs: List[float]
    optimized_epochs: List[float]
    initial_total_delta_v: float
    optimized_total_delta_v: float
    success: bool
    message: str
    n_evaluations: int


def _build_ephemeris_splines(
    bodies: List[str],
    epochs_center: List[float],
    search_radius_days: float = 60.0,
    step_days: float = 2.0,
) -> Dict[str, Tuple[CubicSpline, CubicSpline]]:
    """
    Pre-fetch planetary state vectors across the optimization window and build
    fast cubic splines for rapid, network-free evaluation during optimization loops.
    """
    splines: Dict[str, Tuple[CubicSpline, CubicSpline]] = {}
    for body, ep_c in zip(bodies, epochs_center):
        body_key = str(body).strip().lower()
        if body_key in splines:
            continue
        grid_epochs = np.arange(
            ep_c - search_radius_days,
            ep_c + search_radius_days + step_days,
            step_days,
        )
        states = get_body_states_batch(body=body, epochs=grid_epochs)
        pos_arr = np.array([s.position for s in states])
        vel_arr = np.array([s.velocity for s in states])
        cs_pos = CubicSpline(grid_epochs, pos_arr)
        cs_vel = CubicSpline(grid_epochs, vel_arr)
        splines[body_key] = (cs_pos, cs_vel)
    return splines


def optimize_mga_epochs(
    bodies: List[str],
    initial_epochs: List[Union[float, int, str, datetime.date, datetime.datetime, Time]],
    epoch_bounds: Optional[List[Tuple[float, float]]] = None,
    dsm_configs: Optional[Dict[int, Dict[str, Union[float, str, np.ndarray]]]] = None,
    h_safe_dict: Optional[Dict[str, float]] = None,
    r_park_dep: Optional[float] = None,
    mu_dep: Optional[float] = None,
    r_park_arr: Optional[float] = None,
    mu_arr: Optional[float] = None,
    prograde: Union[bool, List[bool]] = True,
    mu: float = MU_SUN,
    method: str = "Nelder-Mead",
    max_iter: int = 100,
    tol: float = 1e-3,
) -> MGAOptimizationResult:
    """
    Numerically optimize encounter epochs for an MGA trajectory to minimize total Delta-v.

    Pre-samples planetary ephemerides into high-precision cubic splines for ultra-fast
    in-memory convergence, then performs final exact verification against Horizons.

    Parameters:
        bodies: List of N encounter body names.
        initial_epochs: Initial guess for encounter epochs (JD float, ISO string, etc.).
        epoch_bounds: Optional list of (min_jd, max_jd) bounds for each encounter epoch.
        dsm_configs: Optional DSM configuration dictionary.
        h_safe_dict: Optional minimum safe altitudes dictionary.
        r_park_dep: Optional departure parking orbit radius (km).
        mu_dep: Optional departure body gravitational parameter (km^3/s^2).
        r_park_arr: Optional arrival parking orbit radius (km).
        mu_arr: Optional arrival body gravitational parameter (km^3/s^2).
        prograde: Transfer trajectory direction.
        mu: Central body gravitational parameter (default: MU_SUN).
        method: Optimization algorithm for scipy.optimize.minimize (default: 'Nelder-Mead').
        max_iter: Maximum optimizer iterations (default: 100).
        tol: Relative termination tolerance.

    Returns:
        MGAOptimizationResult: Optimized trajectory, epochs, and performance comparison.
    """
    x0 = np.array([parse_epoch(ep) for ep in initial_epochs], dtype=float)
    n_encounters = len(x0)
    n_legs = n_encounters - 1

    # Initial trajectory evaluation (exact)
    initial_traj = solve_mga_trajectory(
        bodies=bodies,
        epochs=list(x0),
        dsm_configs=dsm_configs,
        h_safe_dict=h_safe_dict,
        prograde=prograde,
        mu=mu,
    )
    initial_dv = initial_traj.total_mission_delta_v(
        r_park_dep=r_park_dep,
        mu_dep=mu_dep,
        r_park_arr=r_park_arr,
        mu_arr=mu_arr,
    )

    # Determine search window radius for ephemeris pre-fetching
    if epoch_bounds is not None:
        radius = max(
            abs(x0[i] - epoch_bounds[i][0]) + 10.0 for i in range(n_encounters)
        )
    else:
        radius = 60.0

    splines = _build_ephemeris_splines(
        bodies=bodies,
        epochs_center=list(x0),
        search_radius_days=radius,
        step_days=2.0,
    )

    h_safe_map = h_safe_dict if h_safe_dict is not None else {}
    prograde_list = [prograde] * n_legs if isinstance(prograde, bool) else prograde
    dsm_map = dsm_configs if dsm_configs is not None else {}

    def get_spline_state(body_name: str, epoch_jd: float) -> State:
        cs_p, cs_v = splines[body_name.lower()]
        return State(position=cs_p(epoch_jd), velocity=cs_v(epoch_jd))

    def fast_eval_mga(x: np.ndarray) -> float:
        # Check strict chronological order
        for i in range(len(x) - 1):
            if x[i + 1] <= x[i] + 1.0:
                return 1e6 + (x[i] - x[i + 1] + 1.0) * 1e4

        try:
            # Solve legs using spline-interpolated states
            legs = []
            for k in range(n_legs):
                s_dep = get_spline_state(bodies[k], x[k])
                s_arr = get_spline_state(bodies[k + 1], x[k + 1])
                prog = prograde_list[k]

                if k in dsm_map:
                    dsm_info = dsm_map[k]
                    leg_res = solve_dsm_leg(
                        r1=s_dep,
                        r2=s_arr,
                        r_dsm=dsm_info["position"],
                        t1=x[k],
                        t_dsm=dsm_info["epoch"],
                        t2=x[k + 1],
                        v_dep_body=s_dep,
                        v_arr_body=s_arr,
                        prograde1=prog,
                        prograde2=prog,
                        mu=mu,
                    )
                else:
                    leg_res = solve_single_leg(
                        departure_body=s_dep,
                        arrival_body=s_arr,
                        departure_epoch=x[k],
                        arrival_epoch=x[k + 1],
                        prograde=prog,
                        mu=mu,
                    )
                legs.append(leg_res)

            # Evaluate intermediate flybys
            penalty = 0.0
            flyby_dv = 0.0
            for k in range(1, n_encounters - 1):
                flyby_body = bodies[k]
                mu_body, r_body = get_body_params(flyby_body)
                h_safe = h_safe_map.get(flyby_body.lower(), 100.0)

                v_in = legs[k - 1].arrival_state_craft.velocity
                v_out = legs[k].departure_state_craft.velocity
                v_body = get_spline_state(flyby_body, x[k]).velocity

                fb_res = solve_flyby(
                    v_in=v_in,
                    v_out=v_out,
                    v_body=v_body,
                    mu_body=mu_body,
                    r_body=r_body,
                    h_safe=h_safe,
                )
                flyby_dv += fb_res.delta_v_peri
                if fb_res.hp < h_safe:
                    penalty += (h_safe - fb_res.hp) * 10.0

            dsm_dv = sum(leg.delta_v_dsm_mag for leg in legs if isinstance(leg, DSMTransfer))

            # Mission Delta-v
            v_inf_dep = legs[0].v_inf_dep_mag if legs[0].v_inf_dep_mag is not None else float(np.linalg.norm(legs[0].v_inf_dep))
            v_inf_arr = legs[-1].v_inf_arr_mag if legs[-1].v_inf_arr_mag is not None else float(np.linalg.norm(legs[-1].v_inf_arr))

            dv_total = dsm_dv + flyby_dv
            if r_park_dep is not None and mu_dep is not None:
                v_circ = np.sqrt(mu_dep / r_park_dep)
                dv_total += float(np.sqrt(v_inf_dep ** 2 + 2.0 * mu_dep / r_park_dep) - v_circ)
            else:
                dv_total += v_inf_dep

            if r_park_arr is not None and mu_arr is not None:
                v_circ = np.sqrt(mu_arr / r_park_arr)
                dv_total += float(np.sqrt(v_inf_arr ** 2 + 2.0 * mu_arr / r_park_arr) - v_circ)
            else:
                dv_total += v_inf_arr

            return dv_total + penalty
        except (ValueError, RuntimeError):
            return 1e7

    # Run optimizer in-memory
    opt_res = minimize(
        fun=fast_eval_mga,
        x0=x0,
        method=method,
        bounds=epoch_bounds,
        options={"maxiter": max_iter},
        tol=tol,
    )

    opt_epochs = list(opt_res.x)

    # Final exact evaluation via Horizons
    opt_traj = solve_mga_trajectory(
        bodies=bodies,
        epochs=opt_epochs,
        dsm_configs=dsm_configs,
        h_safe_dict=h_safe_dict,
        prograde=prograde,
        mu=mu,
    )
    opt_dv = opt_traj.total_mission_delta_v(
        r_park_dep=r_park_dep,
        mu_dep=mu_dep,
        r_park_arr=r_park_arr,
        mu_arr=mu_arr,
    )

    return MGAOptimizationResult(
        optimized_trajectory=opt_traj,
        initial_epochs=list(x0),
        optimized_epochs=opt_epochs,
        initial_total_delta_v=initial_dv,
        optimized_total_delta_v=opt_dv,
        success=bool(opt_res.success),
        message=str(opt_res.message),
        n_evaluations=int(opt_res.nfev),
    )
