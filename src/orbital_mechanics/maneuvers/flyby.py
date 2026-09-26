"""
maneuvers/flyby.py

Hyperbolic gravity-assist (flyby) kinematics and validation.
Calculates inbound/outbound hyperbolic excess velocity vectors, turning angles,
required periapsis radii, and checks feasibility against minimum safe altitudes.
"""
from dataclasses import dataclass
import numpy as np

from orbital_mechanics.constants import EPSILON


@dataclass(frozen=True)
class FlybyResult:
    """
    Result of a gravity-assist (flyby) trajectory analysis.

    All vectors are in the inertial / ecliptic frame.
    Units:
        velocities: km/s
        radii / altitudes: km
        angles: rad
    """
    v_inf_in: np.ndarray        # Inbound hyperbolic excess velocity vector (km/s)
    v_inf_out: np.ndarray       # Outbound hyperbolic excess velocity vector (km/s)
    v_inf_in_mag: float         # Magnitude of v_inf_in (km/s)
    v_inf_out_mag: float        # Magnitude of v_inf_out (km/s)
    turn_angle: float           # Hyperbolic deflection / turning angle (rad)
    max_turn_angle: float       # Maximum possible deflection at minimum safe altitude (rad)
    rp: float                   # Required flyby periapsis radius from body center (km)
    hp: float                   # Required flyby periapsis altitude above body surface (km)
    is_feasible: bool           # True if hp >= h_safe and energy is conserved within tolerance
    delta_v_flyby: np.ndarray   # Heliocentric velocity change imparted by flyby (km/s)
    delta_v_flyby_mag: float    # Magnitude of heliocentric delta-v (km/s)
    delta_v_peri: float         # Impulsive delta-v at periapsis if powered (km/s, 0 if unpowered)
    eccentricity: float         # Hyperbolic orbit eccentricity


def compute_flyby_turn_angle(v_inf_in: np.ndarray, v_inf_out: np.ndarray) -> float:
    """
    Compute the turning / bending angle between inbound and outbound v_inf vectors.

    Parameters:
        v_inf_in: Inbound hyperbolic excess velocity vector (km/s).
        v_inf_out: Outbound hyperbolic excess velocity vector (km/s).

    Returns:
        float: Turn angle in radians in [0, pi].
    """
    v_in = np.asarray(v_inf_in, dtype=float)
    v_out = np.asarray(v_inf_out, dtype=float)
    norm_in = np.linalg.norm(v_in)
    norm_out = np.linalg.norm(v_out)

    if norm_in < EPSILON or norm_out < EPSILON:
        raise ValueError("v_inf magnitude cannot be near zero for a hyperbolic flyby.")

    cos_delta = np.clip(np.dot(v_in, v_out) / (norm_in * norm_out), -1.0, 1.0)
    cross_norm = np.linalg.norm(np.cross(v_in, v_out))
    sin_delta = np.clip(cross_norm / (norm_in * norm_out), 0.0, 1.0)

    return float(np.arctan2(sin_delta, cos_delta))


def compute_max_turn_angle(
    v_inf: float,
    mu_body: float,
    r_body: float,
    h_safe: float = 100.0,
) -> float:
    """
    Compute maximum achievable turning angle at minimum safe periapsis altitude.

    Parameters:
        v_inf: Hyperbolic excess speed (km/s).
        mu_body: Gravitational parameter of flyby body (km^3/s^2).
        r_body: Equatorial radius of flyby body (km).
        h_safe: Minimum safe clearance altitude above surface (km, default 100 km).

    Returns:
        float: Maximum deflection angle in radians.
    """
    if v_inf <= 0.0:
        raise ValueError("v_inf must be strictly positive.")
    if mu_body <= 0.0:
        raise ValueError("mu_body must be strictly positive.")

    rp_min = r_body + h_safe
    if rp_min <= 0.0:
        raise ValueError("Minimum periapsis radius must be positive.")

    e_min = 1.0 + (rp_min * v_inf ** 2) / mu_body
    return float(2.0 * np.arcsin(1.0 / e_min))


def compute_flyby_periapsis(
    v_inf: float,
    turn_angle: float,
    mu_body: float,
) -> float:
    """
    Compute required periapsis radius for a given turn angle and excess speed.

    Parameters:
        v_inf: Hyperbolic excess speed (km/s).
        turn_angle: Turn angle (rad).
        mu_body: Gravitational parameter of flyby body (km^3/s^2).

    Returns:
        float: Periapsis radius rp (km).
    """
    if turn_angle <= 0.0 or turn_angle >= np.pi:
        raise ValueError("turn_angle must be strictly between 0 and pi radians.")
    if v_inf <= 0.0:
        raise ValueError("v_inf must be strictly positive.")

    sin_half = np.sin(0.5 * turn_angle)
    e = 1.0 / sin_half
    rp = (mu_body / (v_inf ** 2)) * (e - 1.0)
    return float(rp)


def solve_flyby(
    v_in: np.ndarray,
    v_out: np.ndarray,
    v_body: np.ndarray,
    mu_body: float,
    r_body: float,
    h_safe: float = 100.0,
    tol: float = 1e-3,
) -> FlybyResult:
    """
    Analyze and solve a gravity-assist (flyby) maneuver.

    Parameters:
        v_in: Spacecraft inbound heliocentric velocity vector (km/s).
        v_out: Spacecraft outbound heliocentric velocity vector (km/s).
        v_body: Flyby body heliocentric velocity vector (km/s).
        mu_body: Gravitational parameter of flyby body (km^3/s^2).
        r_body: Physical radius of flyby body (km).
        h_safe: Minimum safe clearance altitude above surface (km, default 100 km).
        tol: Relative tolerance for inbound/outbound v_inf matching for unpowered flybys.

    Returns:
        FlybyResult: Complete flyby kinematics, periapsis altitude, feasibility, and delta-v.
    """
    v_in_vec = np.asarray(v_in, dtype=float)
    v_out_vec = np.asarray(v_out, dtype=float)
    v_body_vec = np.asarray(v_body, dtype=float)

    if v_in_vec.shape != (3,) or v_out_vec.shape != (3,) or v_body_vec.shape != (3,):
        raise ValueError("Velocity vectors must all be length-3.")

    # Hyperbolic excess velocities relative to planet
    v_inf_in = v_in_vec - v_body_vec
    v_inf_out = v_out_vec - v_body_vec

    v_inf_in_mag = float(np.linalg.norm(v_inf_in))
    v_inf_out_mag = float(np.linalg.norm(v_inf_out))

    if v_inf_in_mag < EPSILON or v_inf_out_mag < EPSILON:
        raise ValueError("v_inf magnitude is near zero; flyby trajectory is undefined.")

    # Turning angle
    delta = compute_flyby_turn_angle(v_inf_in, v_inf_out)

    # Average v_inf for kinematics
    v_inf_avg = 0.5 * (v_inf_in_mag + v_inf_out_mag)

    # Maximum turn angle at safe altitude
    max_delta = compute_max_turn_angle(
        v_inf=v_inf_avg,
        mu_body=mu_body,
        r_body=r_body,
        h_safe=h_safe,
    )

    # Heliocentric delta-v imparted
    delta_v_flyby = v_out_vec - v_in_vec
    delta_v_flyby_mag = float(np.linalg.norm(delta_v_flyby))

    if delta < EPSILON:
        # Straight through (0 turn angle)
        rp = np.inf
        hp = np.inf
        e = np.inf
        is_feasible = True
        delta_v_peri = 0.0
    else:
        e = 1.0 / np.sin(0.5 * delta)
        rp = compute_flyby_periapsis(v_inf=v_inf_avg, turn_angle=delta, mu_body=mu_body)
        hp = rp - r_body

        # Check unpowered energy matching
        energy_matched = abs(v_inf_out_mag - v_inf_in_mag) / max(v_inf_in_mag, v_inf_out_mag) <= tol
        altitude_safe = hp >= h_safe
        is_feasible = energy_matched and altitude_safe

        # Powered periapsis burn if energy mismatch or turning angle exceeds unpowered capability
        if not energy_matched:
            # Collinear periapsis impulse
            v_p_in = np.sqrt(v_inf_in_mag ** 2 + 2.0 * mu_body / max(rp, r_body + h_safe))
            v_p_out = np.sqrt(v_inf_out_mag ** 2 + 2.0 * mu_body / max(rp, r_body + h_safe))
            delta_v_peri = float(abs(v_p_out - v_p_in))
        else:
            delta_v_peri = 0.0

    return FlybyResult(
        v_inf_in=v_inf_in,
        v_inf_out=v_inf_out,
        v_inf_in_mag=v_inf_in_mag,
        v_inf_out_mag=v_inf_out_mag,
        turn_angle=delta,
        max_turn_angle=max_delta,
        rp=rp,
        hp=hp,
        is_feasible=is_feasible,
        delta_v_flyby=delta_v_flyby,
        delta_v_flyby_mag=delta_v_flyby_mag,
        delta_v_peri=delta_v_peri,
        eccentricity=e,
    )
