"""
Physics-Informed Neural Networks (PINNs) for solving PDEs.
"""

__version__ = "0.1.0"

from .core import PINN
from .equations import (
    HelmholtzSolver,
    HeatSolver1D,
    HeatSolver2D,
    BurgersSolver1D
)

__all__ = [
    "PINN",
    "HelmholtzSolver",
    "HeatSolver1D",
    "HeatSolver2D",
    "BurgersSolver1D"
]
