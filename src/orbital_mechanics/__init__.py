"""
orbital_mechanics

A Python library for orbital mechanics, mission design, and trajectory
optimization.
"""
from orbital_mechanics.core.state import State, OrbitalElements, elements_to_state
from orbital_mechanics.maneuvers.lambert import solve_lambert, lambert
from orbital_mechanics.maneuvers.flyby import (
    FlybyResult,
    solve_flyby,
    compute_flyby_turn_angle,
    compute_max_turn_angle,
    compute_flyby_periapsis,
)
from orbital_mechanics.maneuvers.dsm import (
    DSMTransfer,
    solve_dsm_leg,
)
from orbital_mechanics.ephemeris.horizons import get_body_state, parse_epoch, clear_ephemeris_cache
from orbital_mechanics.transfers.single_leg import SingleLegTransfer, solve_single_leg
from orbital_mechanics.constants import (
    MU_EARTH,
    MU_SUN,
    MU_MARS,
    MU_VENUS,
    MU_MERCURY,
    R_EARTH,
    R_MARS,
    R_VENUS,
    R_MERCURY,
    R_SUN,
    AU,
    DAY_TO_SEC,
)

__version__ = "0.1.0"

__all__ = [
    "State",
    "OrbitalElements",
    "elements_to_state",
    "solve_lambert",
    "lambert",
    "FlybyResult",
    "solve_flyby",
    "compute_flyby_turn_angle",
    "compute_max_turn_angle",
    "compute_flyby_periapsis",
    "DSMTransfer",
    "solve_dsm_leg",
    "get_body_state",
    "parse_epoch",
    "clear_ephemeris_cache",
    "SingleLegTransfer",
    "solve_single_leg",
    "MU_EARTH",
    "MU_SUN",
    "MU_MARS",
    "MU_VENUS",
    "MU_MERCURY",
    "R_EARTH",
    "R_MARS",
    "R_VENUS",
    "R_MERCURY",
    "R_SUN",
    "AU",
    "DAY_TO_SEC",
]

