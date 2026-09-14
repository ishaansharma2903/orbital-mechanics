"""
maneuvers/lambert.py

Lambert problem solver using the universal-variable formulation (Vallado Algorithm 58).

Given two position vectors and time of flight, determines the unique Keplerian
orbit connecting them and returns the departure and arrival velocity vectors.
Works across elliptic, parabolic, and hyperbolic trajectories.
"""
from typing import Optional
import numpy as np

from orbital_mechanics.constants import EPSILON, MU_SUN


def _stumpff_c2(psi: float) -> float:
    """
    Second Stumpff function c2(psi).

    c2(psi) = (1 - cos(sqrt(psi))) / psi       for psi > 0
    c2(psi) = (cosh(sqrt(-psi)) - 1) / (-psi)   for psi < 0
    c2(0)   = 1/2
    """
    if psi > 1e-6:
        return (1.0 - np.cos(np.sqrt(psi))) / psi
    elif psi < -1e-6:
        return (np.cosh(np.sqrt(-psi)) - 1.0) / (-psi)
    else:
        # Taylor series expansion around psi = 0
        return 0.5 - psi / 24.0 + (psi ** 2) / 720.0 - (psi ** 3) / 40320.0


def _stumpff_c3(psi: float) -> float:
    """
    Third Stumpff function c3(psi).

    c3(psi) = (sqrt(psi) - sin(sqrt(psi))) / (psi^(3/2))      for psi > 0
    c3(psi) = (sinh(sqrt(-psi)) - sqrt(-psi)) / ((-psi)^(3/2)) for psi < 0
    c3(0)   = 1/6
    """
    if psi > 1e-6:
        sqrt_psi = np.sqrt(psi)
        return (sqrt_psi - np.sin(sqrt_psi)) / (psi * sqrt_psi)
    elif psi < -1e-6:
        sqrt_neg = np.sqrt(-psi)
        return (np.sinh(sqrt_neg) - sqrt_neg) / (-psi * sqrt_neg)
    else:
        # Taylor series expansion around psi = 0
        return 1.0 / 6.0 - psi / 120.0 + (psi ** 2) / 5040.0 - (psi ** 3) / 362880.0


def _c2_dot(psi: float, c2: float, c3: float) -> float:
    """Derivative of c2(psi) with respect to psi."""
    if abs(psi) > 1e-5:
        return 0.5 / psi * (1.0 - psi * c3 - 2.0 * c2)
    else:
        return -1.0 / 24.0 + 2.0 * psi / 720.0 - 3.0 * (psi ** 2) / 40320.0


def _c3_dot(psi: float, c2: float, c3: float) -> float:
    """Derivative of c3(psi) with respect to psi."""
    if abs(psi) > 1e-5:
        return 0.5 / psi * (c2 - 3.0 * c3)
    else:
        return -1.0 / 120.0 + 2.0 * psi / 5040.0 - 3.0 * (psi ** 2) / 362880.0


def solve_lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float = MU_SUN,
    prograde: bool = True,
    short_way: Optional[bool] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve Lambert's problem using universal variables (Vallado Algorithm 58).

    Given two position vectors and a time of flight, computes the initial (v1)
    and final (v2) velocity vectors of the transfer orbit.

    Parameters
    ----------
    r1 : array-like, shape (3,)
        Initial position vector in km.
    r2 : array-like, shape (3,)
        Final position vector in km.
    tof : float
        Time of flight in seconds. Must be strictly positive.
    mu : float, optional
        Gravitational parameter of the central body in km^3/s^2. Defaults to MU_SUN.
    prograde : bool, optional
        If True, the transfer orbit is prograde (inclination < 90 deg relative
        to the fundamental plane). If False, retrograde. Ignored if `short_way`
        is explicitly provided. Defaults to True.
    short_way : bool, optional
        If True, forces the transfer angle to be less than 180 deg (Type I).
        If False, forces the transfer angle to be greater than 180 deg (Type II).
        If None (default), chosen automatically based on `prograde` and the
        orientation of r1 x r2.
    max_iter : int, optional
        Maximum iterations for the solver. Defaults to 100.
    tol : float, optional
        Relative tolerance on time-of-flight convergence. Defaults to 1e-8.

    Returns
    -------
    v1 : np.ndarray, shape (3,)
        Initial velocity vector at r1 in km/s.
    v2 : np.ndarray, shape (3,)
        Final velocity vector at r2 in km/s.

    Raises
    ------
    ValueError
        If inputs have invalid shapes, non-positive tof/mu, degenerate radius,
        or collinear/180-degree transfer geometry.
    RuntimeError
        If the solver fails to converge within `max_iter` iterations.
    """
    r1_vec = np.asarray(r1, dtype=float)
    r2_vec = np.asarray(r2, dtype=float)

    if r1_vec.shape != (3,) or r2_vec.shape != (3,):
        raise ValueError("r1 and r2 must be 3-dimensional vectors.")

    r1_norm = float(np.linalg.norm(r1_vec))
    r2_norm = float(np.linalg.norm(r2_vec))

    if r1_norm < EPSILON or r2_norm < EPSILON:
        raise ValueError("Position magnitude ~0; state is degenerate.")

    if tof <= 0.0:
        raise ValueError("Time of flight must be strictly positive.")

    if mu <= 0.0:
        raise ValueError("Gravitational parameter mu must be strictly positive.")

    cross_r1r2 = np.cross(r1_vec, r2_vec)
    cross_norm = float(np.linalg.norm(cross_r1r2))

    # Angle between r1 and r2 using atan2 for numerical robustness
    cos_theta0 = np.clip(np.dot(r1_vec, r2_vec) / (r1_norm * r2_norm), -1.0, 1.0)
    sin_theta0 = cross_norm / (r1_norm * r2_norm)
    theta0 = np.arctan2(sin_theta0, cos_theta0)

    # Determine transfer angle dnu
    if short_way is not None:
        dnu = theta0 if short_way else (2.0 * np.pi - theta0)
    else:
        # Default prograde/retrograde convention from cross product z-component
        if prograde:
            dnu = theta0 if cross_r1r2[2] >= 0.0 else (2.0 * np.pi - theta0)
        else:
            dnu = theta0 if cross_r1r2[2] < 0.0 else (2.0 * np.pi - theta0)

    # Singularity check: 180 deg or 0 deg transfer
    if np.isclose(dnu, np.pi, atol=1e-5) or np.isclose(dnu, 0.0, atol=1e-5) or np.isclose(dnu, 2.0 * np.pi, atol=1e-5):
        raise ValueError("Collinear or 180-degree transfer: transfer orbital plane is undefined.")

    # Geometric parameter A
    A = np.sin(dnu) * np.sqrt((r1_norm * r2_norm) / (1.0 - np.cos(dnu)))

    # Initial bisection / iteration bounds for psi
    psi = 0.0
    psi_low = -4.0 * np.pi ** 2
    psi_up = 4.0 * np.pi ** 2

    sqrt_mu = np.sqrt(mu)

    for _ in range(max_iter):
        c2 = _stumpff_c2(psi)
        c3 = _stumpff_c3(psi)

        # Auxiliary variable y(psi)
        y = r1_norm + r2_norm + A * (psi * c3 - 1.0) / np.sqrt(c2)

        # Negative-y correction when A > 0 (Vallado Algorithm 58 adjustment)
        if A > 0.0:
            while y < 0.0:
                psi_low = psi
                psi = 0.8 * (1.0 / c3) * (1.0 - (r1_norm + r2_norm) * np.sqrt(c2) / A)
                c2 = _stumpff_c2(psi)
                c3 = _stumpff_c3(psi)
                y = r1_norm + r2_norm + A * (psi * c3 - 1.0) / np.sqrt(c2)

        x = np.sqrt(y / c2)
        tof_calc = (x ** 3 * c3 + A * np.sqrt(y)) / sqrt_mu

        if abs((tof_calc - tof) / tof) < tol:
            break

        # Newton-Raphson derivative dt/dpsi
        c2d = _c2_dot(psi, c2, c3)
        c3d = _c3_dot(psi, c2, c3)
        dtdpsi = (
            x ** 3 * (c3d - 1.5 * c3 * c2d / c2)
            + 0.125 * A * (3.0 * c3 * np.sqrt(y) / c2 + A / x)
        ) / sqrt_mu

        # Update bisection bracket
        if tof_calc <= tof:
            psi_low = psi
        else:
            psi_up = psi

        # Newton step
        psi_next = psi - (tof_calc - tof) / dtdpsi

        # If Newton step stays strictly within the bracket, accept it;
        # otherwise, fall back to bisection for guaranteed convergence.
        if psi_low < psi_next < psi_up:
            psi = psi_next
        else:
            psi = 0.5 * (psi_up + psi_low)
    else:
        raise RuntimeError(
            f"Lambert solver failed to converge after {max_iter} iterations "
            f"(final tof error: {abs(tof_calc - tof):.3e} s)."
        )

    # Lagrange coefficients
    f = 1.0 - y / r1_norm
    g = A * np.sqrt(y / mu)
    g_dot = 1.0 - y / r2_norm

    v1 = (r2_vec - f * r1_vec) / g
    v2 = (g_dot * r2_vec - r1_vec) / g

    return v1, v2


# Convenience alias
lambert = solve_lambert
