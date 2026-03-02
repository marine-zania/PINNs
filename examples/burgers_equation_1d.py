"""Example: 1D Burgers Equation

Solves: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²  (nonlinear conservation law)
with Dirichlet boundary conditions and initial condition.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from pinns.core import PINN
from pinns.equations.burgers import BurgersSolver1D


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    """Train and visualize 1D Burgers equation solution."""
    
    print("=" * 60)
    print("Training PINN for 1D Burgers Equation")
    print("=" * 60)
    
    # Configuration
    nu = 0.01 / np.pi  # Viscosity
    domain = (-1, 1)   # x ∈ [-1, 1]
    time_range = (0, 1) # t ∈ [0, 1]
    
    # Create model
    model = PINN(input_dim=2, hidden_dim=50, hidden_layers=3, output_dim=1)
    solver = BurgersSolver1D(model, nu=nu, device=DEVICE)
    
    # Generate training data
    # Interior points (collocation points)
    x_int = torch.rand(10000, 1, device=DEVICE) * 2 - 1   # [-1, 1]
    t_int = torch.rand(10000, 1, device=DEVICE)
    
    # Boundary conditions: u(-1, t) = 0, u(1, t) = 0
    t_b = torch.rand(2000, 1, device=DEVICE)
    x_b_left = torch.full((1000, 1), -1.0, device=DEVICE)
    x_b_right = torch.full((1000, 1), 1.0, device=DEVICE)
    x_b = torch.cat([x_b_left, x_b_right], dim=0)
    t_b_full = torch.cat([t_b[:1000], t_b[1000:]], dim=0)
    u_b = torch.zeros_like(x_b)  # u = 0 on boundaries
    
    # Initial condition: u(x, 0) = -sin(πx)
    x_i = torch.rand(2000, 1, device=DEVICE) * 2 - 1
    t_i = torch.zeros_like(x_i)
    u_i = -torch.sin(np.pi * x_i)
    
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
    x_plot = np.linspace(-1, 1, 256)
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
    ax.set_title("1D Burgers Equation Solution", fontsize=12)
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
