"""Burgers equation solver."""

import torch
import numpy as np
import torch.optim as optim

from ..core import PINN
from ..utils.derivatives import gradient
from .base import PDESolver


class BurgersSolver1D(PDESolver):
    """
    Solver for 1D Burgers equation:
    ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
    
    Nonlinear conservation law with diffusion.
    """
    
    def __init__(self, model, nu=0.01/np.pi, device='cpu'):
        """
        Initialize 1D Burgers solver.
        
        Args:
            model: PINN model instance
            nu: Viscosity coefficient
            device: torch device
        """
        super().__init__(model, device)
        self.nu = nu
    
    def pde_loss(self, x, t):
        """
        Compute PDE residual loss for Burgers equation.
        
        Args:
            x, t: Coordinate tensors (requires grad)
        
        Returns:
            PDE loss (mean squared residual)
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.model(x, t)
        
        # ∂u/∂t
        u_t = gradient(u, t)
        
        # ∂u/∂x
        u_x = gradient(u, x)
        
        # ∂²u/∂x²
        u_xx = gradient(u_x, x)
        
        # PDE: ∂u/∂t + u∂u/∂x - ν∂²u/∂x² = 0
        residual = u_t + u * u_x - self.nu * u_xx
        
        return torch.mean(residual ** 2)
    
    def boundary_loss(self, x_b, t_b, u_b_values):
        """
        Compute boundary condition loss.
        
        Args:
            x_b, t_b: Boundary coordinate tensors
            u_b_values: Expected values at boundaries
        
        Returns:
            Boundary loss
        """
        u_b_pred = self.model(x_b, t_b)
        return torch.mean((u_b_pred - u_b_values) ** 2)
    
    def initial_loss(self, x_i, t_i, u_i_values):
        """
        Compute initial condition loss.
        
        Args:
            x_i, t_i: Initial condition coordinates
            u_i_values: Expected values at t=0
        
        Returns:
            Initial condition loss
        """
        u_i_pred = self.model(x_i, t_i)
        return torch.mean((u_i_pred - u_i_values) ** 2)
    
    def total_loss(
        self, x, t, x_b, t_b, u_b_values,
        x_i, t_i, u_i_values
    ):
        """
        Compute weighted total loss.
        """
        loss_pde = self.pde_loss(x, t)
        loss_bc = self.boundary_loss(x_b, t_b, u_b_values)
        loss_ic = self.initial_loss(x_i, t_i, u_i_values)
        
        return loss_pde + loss_bc + loss_ic
    
    def train(
        self,
        x, t, x_b, t_b, u_b_values,
        x_i, t_i, u_i_values,
        lr=1e-3,
        epochs=5000,
        verbose=True,
        print_every=1000
    ):
        """Train the 1D Burgers solver."""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        losses = []
        
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            loss = self.total_loss(
                x, t, x_b, t_b, u_b_values,
                x_i, t_i, u_i_values
            )
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and epoch % print_every == 0:
                print(f"Epoch {epoch:>5d}  Loss={loss.item():.3e}")
        
        return self.model, losses
