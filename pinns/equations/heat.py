"""Heat equation solver (1D and 2D)."""

import torch
import numpy as np
import torch.optim as optim

from ..core import PINN
from ..utils.derivatives import gradient
from .base import PDESolver


class HeatSolver1D(PDESolver):
    """
    Solver for 1D Heat equation:
    ∂u/∂t = α∂²u/∂x²
    """
    
    def __init__(self, model, alpha=1/16, device='cpu'):
        """
        Initialize 1D Heat solver.
        
        Args:
            model: PINN model instance
            alpha: Diffusion coefficient
            device: torch device
        """
        super().__init__(model, device)
        self.alpha = alpha
    
    def pde_loss(self, x, t):
        """
        Compute PDE residual loss for heat equation.
        
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
        
        # PDE: ∂u/∂t - α∂²u/∂x² = 0
        residual = u_t - self.alpha * u_xx
        
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
        """Train the 1D heat solver."""
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


class HeatSolver2D(PDESolver):
    """
    Solver for 2D Heat equation:
    ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
    """
    
    def __init__(self, model, alpha=1.0, device='cpu'):
        """
        Initialize 2D Heat solver.
        
        Args:
            model: PINN model instance
            alpha: Diffusion coefficient
            device: torch device
        """
        super().__init__(model, device)
        self.alpha = alpha
    
    def pde_loss(self, x, y, t):
        """
        Compute PDE residual loss for 2D heat equation.
        
        Args:
            x, y, t: Coordinate tensors (requires grad)
        
        Returns:
            PDE loss (mean squared residual)
        """
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.model(x, y, t)
        
        # ∂u/∂t
        u_t = gradient(u, t)
        
        # ∂u/∂x, ∂u/∂y
        u_x = gradient(u, x)
        u_y = gradient(u, y)
        
        # ∂²u/∂x², ∂²u/∂y²
        u_xx = gradient(u_x, x)
        u_yy = gradient(u_y, y)
        
        # PDE: ∂u/∂t - α(∂²u/∂x² + ∂²u/∂y²) = 0
        residual = u_t - self.alpha * (u_xx + u_yy)
        
        return torch.mean(residual ** 2)
    
    def boundary_loss(self, x_b, y_b, t_b, u_b_values):
        """
        Compute boundary condition loss.
        
        Args:
            x_b, y_b, t_b: Boundary coordinate tensors
            u_b_values: Expected values at boundaries
        
        Returns:
            Boundary loss
        """
        u_b_pred = self.model(x_b, y_b, t_b)
        return torch.mean((u_b_pred - u_b_values) ** 2)
    
    def initial_loss(self, x_i, y_i, t_i, u_i_values):
        """
        Compute initial condition loss.
        
        Args:
            x_i, y_i, t_i: Initial condition coordinates
            u_i_values: Expected values at t=0
        
        Returns:
            Initial condition loss
        """
        u_i_pred = self.model(x_i, y_i, t_i)
        return torch.mean((u_i_pred - u_i_values) ** 2)
    
    def total_loss(
        self, x, y, t, x_b, y_b, t_b, u_b_values,
        x_i, y_i, t_i, u_i_values
    ):
        """
        Compute weighted total loss.
        """
        loss_pde = self.pde_loss(x, y, t)
        loss_bc = self.boundary_loss(x_b, y_b, t_b, u_b_values)
        loss_ic = self.initial_loss(x_i, y_i, t_i, u_i_values)
        
        return loss_pde + loss_bc + loss_ic
    
    def train(
        self,
        x, y, t, x_b, y_b, t_b, u_b_values,
        x_i, y_i, t_i, u_i_values,
        lr=1e-3,
        epochs=5000,
        verbose=True,
        print_every=1000
    ):
        """Train the 2D heat solver."""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        losses = []
        
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            loss = self.total_loss(
                x, y, t, x_b, y_b, t_b, u_b_values,
                x_i, y_i, t_i, u_i_values
            )
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and epoch % print_every == 0:
                print(f"Epoch {epoch:>5d}  Loss={loss.item():.3e}")
        
        return self.model, losses
