"""
core/state.py

Cartesian state <-> classical orbital elements conversion.

Note: OrbitalElements, State, and elements_to_state are kept together in
one file for now since they're tightly coupled (elements_to_state needs
to construct a State, and State.to_orbit_elements needs to return
OrbitalElements) -- splitting them into separate files today would just
create a circular import between them for no real benefit. Split this out
once it actually grows unwieldy, not before.
"""
from dataclasses import dataclass, field
from typing import NamedTuple
import numpy as np

from orbital_mechanics.constants import MU_EARTH, EPSILON


class OrbitalElements(NamedTuple):
    a: float               # semi-major axis, km
    e: float                # eccentricity
    i: float                 # inclination, rad
    raan: float                # right ascension of ascending node, rad
    arg_periapsis: float         # argument of periapsis, rad
    true_anomaly: float            # true anomaly, rad


@dataclass(frozen=True)
class State:
    """Cartesian orbital state (position & velocity) in an inertial frame (e.g. ECI)."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        pos = np.asarray(self.position, dtype=float)
        vel = np.asarray(self.velocity, dtype=float)
        if pos.shape != (3,) or vel.shape != (3,):
            raise ValueError("position and velocity must be length-3 vectors")
        # bypass frozen dataclass's immutability for validated assignment
        object.__setattr__(self, "position", pos)
        object.__setattr__(self, "velocity", vel)

    def to_orbit_elements(self, mu: float = MU_EARTH) -> OrbitalElements:
        r_vec, v_vec = self.position, self.velocity
        r, v = np.linalg.norm(r_vec), np.linalg.norm(v_vec)
        if r < EPSILON:
            raise ValueError("Position magnitude ~0; state is degenerate.")

        v_r = np.dot(r_vec, v_vec) / r

        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        if h < EPSILON:
            raise ValueError("Angular momentum ~0; rectilinear trajectory has no classical elements.")

        # Semi-major axis (vis-viva); guard the parabolic case
        energy_term = 2 / r - v ** 2 / mu
        a = np.inf if abs(energy_term) < EPSILON else 1 / energy_term

        # Inclination
        i = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

        # Node vector
        N_vec = np.cross([0, 0, 1], h_vec)
        N = np.linalg.norm(N_vec)

        # RAAN
        raan = 0.0 if N < EPSILON else np.arctan2(N_vec[1], N_vec[0]) % (2 * np.pi)

        # Eccentricity vector
        e_vec = (np.cross(v_vec, h_vec) - mu * r_vec / r) / mu
        e = np.linalg.norm(e_vec)

        # Argument of periapsis
        if N < EPSILON or e < EPSILON:
            arg_periapsis = 0.0
        else:
            # cross(N, e) is parallel to h_hat (= h_vec / h) with magnitude
            # N*e*sin(omega); dividing by h (not just using h_vec) is required
            # so the sin- and cos-arguments to atan2 are consistently scaled.
            arg_periapsis = np.arctan2(
                np.dot(np.cross(N_vec, e_vec), h_vec) / h,
                np.dot(N_vec, e_vec)
            ) % (2 * np.pi)

        # True anomaly
        if e < EPSILON:
            true_anomaly = 0.0
        else:
            true_anomaly = np.arctan2(r * v_r * h, mu * np.dot(e_vec, r_vec)) % (2 * np.pi)

        return OrbitalElements(a, e, i, raan, arg_periapsis, true_anomaly)

    def to_perifocal_frame(self, mu: float = MU_EARTH) -> "State":
        _, _, i, raan, arg_periapsis, _ = self.to_orbit_elements(mu)
        Q = self._eci_to_perifocal_matrix(raan, i, arg_periapsis)
        return State(position=Q @ self.position, velocity=Q @ self.velocity)

    @staticmethod
    def _eci_to_perifocal_matrix(raan: float, i: float, arg_periapsis: float) -> np.ndarray:
        """Rotation matrix from ECI to the perifocal (PQW) frame."""
        cO, sO = np.cos(raan), np.sin(raan)
        ci, si = np.cos(i), np.sin(i)
        cw, sw = np.cos(arg_periapsis), np.sin(arg_periapsis)
        return np.array([
            [cw * cO - sw * sO * ci,  cw * sO + sw * cO * ci,  sw * si],
            [-sw * cO - cw * sO * ci, -sw * sO + cw * cO * ci, cw * si],
            [sO * si,                 -cO * si,                ci],
        ])


def elements_to_state(elements: OrbitalElements, mu: float = MU_EARTH) -> State:
    """
    Construct a Cartesian State from classical orbital elements.
    Used for round-trip testing and for cases (e.g. Lambert-solver results)
    where elements need to be turned back into a propagatable state.
    """
    a, e, i, raan, arg_periapsis, theta = elements
    p = a * (1 - e ** 2)
    r = p / (1 + e * np.cos(theta))

    # Position & velocity in the perifocal (PQW) frame
    r_pf = r * np.array([np.cos(theta), np.sin(theta), 0.0])
    v_pf = (mu / p) ** 0.5 * np.array([-np.sin(theta), e + np.cos(theta), 0.0])

    # Perifocal -> ECI is the transpose of the ECI -> perifocal matrix
    Q = State._eci_to_perifocal_matrix(raan, i, arg_periapsis).T
    return State(position=Q @ r_pf, velocity=Q @ v_pf)
