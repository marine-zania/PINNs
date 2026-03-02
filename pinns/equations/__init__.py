"""Base class and specific PDE solvers."""

from .base import PDESolver
from .helmholtz import HelmholtzSolver
from .heat import HeatSolver1D, HeatSolver2D
from .burgers import BurgersSolver1D

__all__ = [
    "PDESolver",
    "HelmholtzSolver",
    "HeatSolver1D",
    "HeatSolver2D",
    "BurgersSolver1D"
]
