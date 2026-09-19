"""
maneuvers

Orbital maneuvers and trajectory design methods.
"""
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

__all__ = [
    "solve_lambert",
    "lambert",
    "FlybyResult",
    "solve_flyby",
    "compute_flyby_turn_angle",
    "compute_max_turn_angle",
    "compute_flyby_periapsis",
    "DSMTransfer",
    "solve_dsm_leg",
]
