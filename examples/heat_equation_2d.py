"""Example: 2D Heat Equation

Solves: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
with Dirichlet boundary conditions and initial condition.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from pinns.core import PINN
from pinns.equations.heat import HeatSolver2D


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    """Train and visualize 2D heat equation solution."""
    
    print("=" * 60)
    print("Training PINN for 2D Heat Equation")
    print("=" * 60)
    
    # Configuration
    alpha = 1.0  # Diffusion coefficient
    
    # Create model
    model = PINN(input_dim=3, hidden_dim=50, hidden_layers=3, output_dim=1)
    solver = HeatSolver2D(model, alpha=alpha, device=DEVICE)
    
    # Generate training data
    # Interior points
    x_int = torch.rand(1000, 1, device=DEVICE)
    y_int = torch.rand(1000, 1, device=DEVICE)
    t_int = torch.rand(1000, 1, device=DEVICE)
    
    # Boundary points: u = 0 on all boundaries
    n_b = 200
    
    # For each boundary: sample along edge and random time
    t_b = torch.rand(4*n_b, 1, device=DEVICE)
    
    # x=0 and x=1 edges
    x_b_left = torch.zeros(n_b, 1, device=DEVICE)
    x_b_right = torch.ones(n_b, 1, device=DEVICE)
    y_b_left_right = torch.rand(n_b, 1, device=DEVICE)
    
    # y=0 and y=1 edges
    y_b_bottom = torch.zeros(n_b, 1, device=DEVICE)
    y_b_top = torch.ones(n_b, 1, device=DEVICE)
    x_b_bottom_top = torch.rand(n_b, 1, device=DEVICE)
    
    x_b = torch.cat([x_b_left, x_b_right, x_b_bottom_top, x_b_bottom_top], dim=0)
    y_b = torch.cat([y_b_left_right, y_b_left_right, y_b_bottom, y_b_top], dim=0)
    u_b = torch.zeros(4*n_b, 1, device=DEVICE)
    
    # Initial condition: u(x, y, 0) = sin(πx) * sin(πy)
    n_i = 200
    x_i = torch.rand(n_i, 1, device=DEVICE)
    y_i = torch.rand(n_i, 1, device=DEVICE)
    t_i = torch.zeros(n_i, 1, device=DEVICE)
    u_i = torch.sin(np.pi * x_i) * torch.sin(np.pi * y_i)
    
    # Train
    print("\nTraining...")
    model, losses = solver.train(
        x_int, y_int, t_int,
        x_b, y_b, t_b, u_b,
        x_i, y_i, t_i, u_i,
        lr=1e-3,
        epochs=5000,
        verbose=False
    )
    
    print("Training complete!")
    
    # Plot solution at different times
    t_vals = [0.0, 0.005, 0.01]
    fig, axes = plt.subplots(1, len(t_vals) + 1, figsize=(15, 4))
    
    n_plot = 100
    x_plot = np.linspace(0, 1, n_plot)
    y_plot = np.linspace(0, 1, n_plot)
    X, Y = np.meshgrid(x_plot, y_plot, indexing='ij')
    
    # Plots 1-3: Solution snapshots
    for idx, t_val in enumerate(t_vals):
        ax = axes[idx]
        
        X_flat = torch.tensor(X.ravel(), dtype=torch.float32, device=DEVICE).view(-1, 1)
        Y_flat = torch.tensor(Y.ravel(), dtype=torch.float32, device=DEVICE).view(-1, 1)
        T_flat = torch.full_like(X_flat, t_val)
        
        with torch.no_grad():
            U = model(X_flat, Y_flat, T_flat).cpu().numpy().reshape(n_plot, n_plot)
        
        im = ax.contourf(x_plot, y_plot, U.T, levels=20, cmap='jet')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"u(x,y,t={t_val})")
        fig.colorbar(im, ax=ax)
    
    # Plot 4: Loss history
    ax = axes[-1]
    ax.semilogy(losses, linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
