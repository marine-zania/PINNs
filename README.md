# Physics-Informed Neural Networks (PINNs)

A comprehensive, modular framework for solving PDEs using Physics-Informed Neural Networks.

## Overview

This repository contains implementations of PINNs for various PDEs:
- **1D Burgers' Equation**
- **1D Heat Equation**
- **2D Heat Equation**
- **2D Helmholtz Equation** (Rectangular Waveguide)

## Project Structure

```
pinns/
├── core/                 # Core PINN model and architecture
│   ├── model.py         # Base PINN class
│   └── __init__.py
├── equations/            # PDE-specific solvers
│   ├── base.py          # Abstract base solver class
│   ├── helmholtz.py     # Helmholtz equation solver
│   ├── heat.py          # Heat equation solver
│   ├── burgers.py       # Burgers' equation solver
│   └── __init__.py
├── utils/                # Utility functions
│   ├── data.py          # Data generation and sampling
│   ├── derivatives.py   # Automatic differentiation utilities
│   ├── training.py      # Training loop utilities
│   └── __init__.py
└── visualization/        # Plotting and visualization
    ├── __init__.py      # Plotting utilities
    └── __init__.py

examples/                  # Example scripts for each equation type
├── helmholtz_2d.py
├── heat_equation_1d.py
├── heat_equation_2d.py
└── burgers_equation_1d.py
```

## Features

- **Modular Architecture**: Clean separation of concerns (models, solvers, utilities, visualization)
- **Reusable Components**: PDE-agnostic core that works with any equation
- **Automatic Differentiation**: Built-in utilities for computing derivatives
- **Flexible Training**: Generic trainer supporting custom loss functions
- **Multiple Equations**: Examples for 1D/2D PDEs with different boundary conditions
- **Built-in Visualization**: Plotting utilities for 1D and 2D solutions

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Helmholtz Equation)

```python
import torch
from pinns.core import PINN
from pinns.equations.helmholtz import HelmholtzSolver
from pinns.utils.data import create_domain_points, create_boundary_points

# Create model
model = PINN(input_dim=2, hidden_dim=50, hidden_layers=3)

# Create solver
solver = HelmholtzSolver(model, k2_value=10.0)

# Generate training data
X, Y = create_domain_points((0, 1), (0, 1), nx=50, ny=50)
X_b, Y_b = create_boundary_points((0, 1), (0, 1), n_points=30)

# Train
solver.train(X, Y, X_b, Y_b, epochs=5000, lr=1e-3)
```

### Run Full Examples

```bash
# Helmholtz equation (2D rectangular waveguide)
python examples/helmholtz_2d.py

# Heat equation (1D and 2D)
python examples/heat_equation_1d.py
python examples/heat_equation_2d.py

# Burgers' equation (1D nonlinear)
python examples/burgers_equation_1d.py
```

## Key Components

### Core Model (`pinns.core.PINN`)

Fully-connected feedforward neural network with:
- Configurable hidden dimensions and layers
- Xavier uniform weight initialization
- Multiple activation functions (Tanh, ReLU, SiLU)

### Equation Solvers (`pinns.equations.*`)

Inherit from base `PDESolver` class and implement:
- `pde_loss()`: PDE residual
- `boundary_loss()`: Boundary condition enforcement
- `train()`: Full training loop

### Utilities

- **Derivatives** (`pinns.utils.derivatives`): Gradient, Laplacian, higher-order derivatives
- **Data** (`pinns.utils.data`): Domain and boundary point sampling
- **Training** (`pinns.utils.training`): Generic training utilities
- **Visualization** (`pinns.visualization`): 1D/2D plotting functions

## Supported PDEs

| PDE | Form | Domain | BC |
|-----|------|--------|-----|
| Helmholtz | ∇²u + k²u = 0 | 2D Rectangle | Dirichlet |
| Heat | ∂u/∂t = α∇²u | 1D/2D | Dirichlet |
| Burgers | ∂u/∂t + u∂u/∂x = ν∂²u/∂x² | 1D | Dirichlet |

## Performance Tips

1. **Normalize inputs and outputs** to [0, 1] for better training stability
2. **Use appropriate network size**: Start with 3-4 hidden layers, 50-100 neurons
3. **Adaptive learning rates**: Use schedulers for complex problems
4. **Anchor points**: Add soft constraints at known solution points
5. **Loss weighting**: Balance PDE, BC, and data terms appropriately

## Contributing

Contributions welcome! Areas for expansion:
- Additional PDEs (Navier-Stokes, Schrodinger, etc.)
- More complex geometries
- Physics-based loss regularization
- Uncertainty quantification
- Inverse problems

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems.
- https://github.com/maziarraissi/PINNs

## License

MIT

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Status**: Framework in active development. More equation types and examples coming soon!

