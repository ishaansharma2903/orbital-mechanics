"""
orbital_mechanics

A Python library for orbital mechanics, mission design, and trajectory
optimization.
"""
from orbital_mechanics.core.state import State, OrbitalElements, elements_to_state
from orbital_mechanics.maneuvers.lambert import solve_lambert, lambert
from orbital_mechanics.constants import MU_EARTH, MU_SUN, MU_MARS, AU

__version__ = "0.1.0"

__all__ = [
    "State",
    "OrbitalElements",
    "elements_to_state",
    "solve_lambert",
    "lambert",
    "MU_EARTH",
    "MU_SUN",
    "MU_MARS",
    "AU",
]

