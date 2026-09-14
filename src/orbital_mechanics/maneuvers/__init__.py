"""
maneuvers

Orbital maneuvers and trajectory design methods.
"""
from orbital_mechanics.maneuvers.lambert import solve_lambert, lambert

__all__ = [
    "solve_lambert",
    "lambert",
]
