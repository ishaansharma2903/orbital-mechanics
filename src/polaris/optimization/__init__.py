"""
optimization

Trajectory optimization and parameter search modules.
"""
from polaris.optimization.mga_optimizer import (
    MGAOptimizationResult,
    optimize_mga_epochs,
)

__all__ = [
    "MGAOptimizationResult",
    "optimize_mga_epochs",
]
