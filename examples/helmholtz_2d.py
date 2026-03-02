"""
Example: 2D Helmholtz equation for rectangular waveguide modes.

Solves: ∇²Hₖ + k²Hₖ = 0 with Dirichlet boundary conditions.
Demonstrates the modular structure for solving PDEs with PINNs.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from pinns.core import PINN
from pinns.equations.helmholtz import HelmholtzSolver
from pinns.utils.data import create_domain_points, create_boundary_points


# Domain parameters
A = 19.05  # Width (mm)
B = 9.525  # Height (mm)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def analytical_k2(m, n, a=A, b=B):
    """Compute analytical k² for TEmn mode."""
    return (m * np.pi / a) ** 2 + (n * np.pi / b) ** 2


def get_mode_anchors(m, n, a=A, b=B):
    """
    Generate anchor points mimicking the field pattern.
    
    For TEmn mode, Hₖ has m peaks in x and n peaks in y.
    """
    coords = []
    values = []
    for i in range(m):
        for j in range(n):
            xi = (i + 0.5) * a / m
            yj = (j + 0.5) * b / n
            val = (-1) ** (i + j)
            coords.append((xi, yj))
            values.append(val)
    return coords, values


def plot_solution(model, m, n, k2_val, a=A, b=B, ax=None):
    """Plot the trained solution."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.linspace(0, a, 100)
    y = np.linspace(0, b, 100)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    X_torch = torch.tensor(X.ravel(), dtype=torch.float32, device=DEVICE).view(-1, 1)
    Y_torch = torch.tensor(Y.ravel(), dtype=torch.float32, device=DEVICE).view(-1, 1)
    
    with torch.no_grad():
        Hz = model(X_torch, Y_torch).cpu().numpy().reshape(100, 100)
    
    Hz_norm = Hz / (np.abs(Hz).max() + 1e-10)
    cf = ax.contourf(x, y, Hz_norm.T, levels=100, cmap='jet', vmin=-1, vmax=1)
    ax.set_title(f"TE({m},{n}) mode, $k^2={k2_val:.3f}$", fontsize=12)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    
    return ax, cf


def main():
    """Train and visualize multiple Helmholtz modes."""
    
    # Configuration
    modes = [(1, 1), (2, 1), (3, 1), (1, 2), (4, 1), (2, 2), (3, 2), (5, 1)]
    
    models = []
    k2_values = []
    
    print("=" * 60)
    print("Training PINNs for 2D Helmholtz Equation (Rectangular Waveguide)")
    print("=" * 60)
    
    for m, n in modes:
        print(f"\n[TE({m},{n})] Training...")
        
        # Compute analytical k²
        k2_val = analytical_k2(m, n)
        
        # Create model
        model = PINN(input_dim=2, hidden_dim=50, hidden_layers=3, output_dim=1)
        
        # Create solver
        solver = HelmholtzSolver(model, k2_value=k2_val, device=DEVICE)
        
        # Generate training data
        X_interior, Y_interior = create_domain_points(
            (0, A), (0, B), nx=80, ny=40, device=DEVICE
        )
        X_boundary, Y_boundary = create_boundary_points(
            (0, A), (0, B), n_points=40, device=DEVICE
        )
        
        # Get anchor points
        anchor_coords, anchor_values = get_mode_anchors(m, n)
        
        # Train
        solver.train(
            X_interior, Y_interior,
            X_boundary, Y_boundary,
            anchor_coords=anchor_coords,
            anchor_values=anchor_values,
            anchor_weight=0.1,
            lr=1e-3,
            epochs=5000,
            verbose=False
        )
        
        models.append(model)
        k2_values.append(k2_val)
        print(f"[TE({m},{n})] Training complete! k²={k2_val:.4f}")
    
    # Plot results
    print("\n" + "=" * 60)
    print("Plotting results...")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (model, (m, n), k2_val) in enumerate(zip(models, modes, k2_values)):
        plot_solution(model, m, n, k2_val, ax=axes[idx])
    
    fig.suptitle("2D Helmholtz Equation - First 8 TEmn Modes", fontsize=16, y=1.00)
    fig.tight_layout()
    plt.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
