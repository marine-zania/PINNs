"""Example: 1D Heat Equation

Solves: ∂u/∂t = α∂²u/∂x²
with Dirichlet boundary conditions and initial condition.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from pinns.core import PINN
from pinns.equations.heat import HeatSolver1D


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    """Train and visualize 1D heat equation solution."""
    
    print("=" * 60)
    print("Training PINN for 1D Heat Equation")
    print("=" * 60)
    
    # Configuration
    alpha = 1.0 / 16.0  # Diffusion coefficient
    domain = (0, 4)     # x ∈ [0, 4]
    time_range = (0, 1) # t ∈ [0, 1]
    
    # Create model
    model = PINN(input_dim=2, hidden_dim=64, hidden_layers=3, output_dim=1)
    solver = HeatSolver1D(model, alpha=alpha, device=DEVICE)
    
    # Generate training data
    # Interior points
    x_int = torch.rand(10000, 1, device=DEVICE) * 4.0
    t_int = torch.rand(10000, 1, device=DEVICE)
    
    # Boundary conditions: u(0, t) = 0, u(4, t) = 8
    t_b = torch.linspace(0, 1, 100).view(-1, 1).to(DEVICE)
    x_b0 = torch.zeros_like(t_b)
    x_b4 = 4 * torch.ones_like(t_b)
    u_b0 = torch.zeros_like(t_b)
    u_b4 = 8 * torch.ones_like(t_b)
    
    # Combine boundaries
    x_b = torch.cat([x_b0, x_b4], dim=0)
    t_b_full = torch.cat([t_b, t_b], dim=0)
    u_b = torch.cat([u_b0, u_b4], dim=0)
    
    # Initial condition: u(x, 0) = 0.5 * x * (8 - x)
    x_i = torch.linspace(0, 4, 100).view(-1, 1).to(DEVICE)
    t_i = torch.zeros_like(x_i)
    u_i = 0.5 * x_i * (8 - x_i)
    
    # Train
    print("\nTraining...")
    model, losses = solver.train(
        x_int, t_int,
        x_b, t_b_full, u_b,
        x_i, t_i, u_i,
        lr=1e-3,
        epochs=5000,
        verbose=False
    )
    
    print("Training complete!")
    
    # Plot solution
    x_plot = np.linspace(0, 4, 200)
    t_plot = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Solution at different times
    ax = axes[0]
    for t_val in t_plot:
        x_tensor = torch.tensor(x_plot, dtype=torch.float32, device=DEVICE).view(-1, 1)
        t_tensor = torch.full_like(x_tensor, t_val)
        
        with torch.no_grad():
            u_pred = model(x_tensor, t_tensor).cpu().numpy().flatten()
        
        ax.plot(x_plot, u_pred, label=f"t={t_val:.2f}", linewidth=2)
    
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("u(x, t)", fontsize=12)
    ax.set_title("1D Heat Equation Solution", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss history
    ax = axes[1]
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
