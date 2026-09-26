"""
tests/optimization/test_optimizer.py

Unit and validation tests for MGA trajectory optimization.
"""
import pytest
import numpy as np

from orbital_mechanics.constants import (
    MU_EARTH,
    MU_MARS,
    R_EARTH,
    R_MARS,
)
from orbital_mechanics.optimization.mga_optimizer import (
    MGAOptimizationResult,
    optimize_mga_epochs,
)


def test_optimize_mga_epochs_earth_mars():
    """
    Validate numerical optimization of launch and arrival dates for an Earth-Mars transfer.
    """
    bodies = ["earth", "mars"]
    # Slightly perturbed guess away from optimal 2020 window
    initial_epochs = ["2020-07-10", "2021-03-01"]

    result = optimize_mga_epochs(
        bodies=bodies,
        initial_epochs=initial_epochs,
        r_park_dep=R_EARTH + 200.0,
        mu_dep=MU_EARTH,
        r_park_arr=R_MARS + 250.0,
        mu_arr=MU_MARS,
        max_iter=30,
    )

    assert isinstance(result, MGAOptimizationResult)
    assert len(result.optimized_epochs) == 2
    assert result.optimized_total_delta_v <= result.initial_total_delta_v + 1e-4
    assert result.optimized_trajectory.tof_total_days > 0.0


def test_optimize_mga_epochs_penalty_and_bounds():
    """Verify optimization handles chronological ordering constraint violations gracefully."""
    bodies = ["earth", "venus", "mercury"]
    initial_epochs = [2453220.5, 2453450.5, 2453700.5]

    bounds = [
        (2453200.0, 2453250.0),
        (2453400.0, 2453500.0),
        (2453650.0, 2453750.0),
    ]

    result = optimize_mga_epochs(
        bodies=bodies,
        initial_epochs=initial_epochs,
        epoch_bounds=bounds,
        max_iter=15,
    )

    assert isinstance(result, MGAOptimizationResult)
    assert len(result.optimized_trajectory.flybys) == 1
