"""Helmholtz equation solver."""

import torch
import numpy as np
import torch.optim as optim

from ..core import PINN
from ..utils.derivatives import laplacian
from ..utils.training import train_pinn
from .base import PDESolver


class HelmholtzSolver(PDESolver):
    """
    Solver for 2D Helmholtz equation:
    ∇²u + k²u = 0
    with Dirichlet boundary conditions (u = 0 on boundary)
    """
    
    def __init__(self, model, k2_value, device='cpu'):
        """
        Initialize Helmholtz solver.
        
        Args:
            model: PINN model instance
            k2_value: k² parameter for the equation
            device: torch device
        """
        super().__init__(model, device)
        self.k2 = k2_value
    
    def pde_loss(self, x, y):
        """
        Compute PDE residual loss.
        
        Args:
            x, y: Coordinate tensors
        
        Returns:
            PDE loss (mean squared residual)
        """
        u = self.model(x, y)
        lap_u = laplacian(u, x, y)
        residual = lap_u + self.k2 * u
        return torch.mean(residual ** 2)
    
    def boundary_loss(self, x_b, y_b):
        """
        Compute boundary condition loss (Dirichlet: u = 0 on boundary).
        
        Args:
            x_b, y_b: Boundary coordinate tensors
        
        Returns:
            Boundary loss
        """
        u_b = self.model(x_b, y_b)
        return torch.mean(u_b ** 2)
    
    def anchor_loss(self, anchor_coords, anchor_values):
        """
        Compute loss for anchor points (soft constraints).
        
        Args:
            anchor_coords: List of (x, y) tuples
            anchor_values: List of expected values
        
        Returns:
            Anchor loss
        """
        total = 0.0
        for (xi, yi), vi in zip(anchor_coords, anchor_values):
            xa = torch.tensor([[xi]], device=self.device, dtype=torch.float32, requires_grad=True)
            ya = torch.tensor([[yi]], device=self.device, dtype=torch.float32, requires_grad=True)
            u_pred = self.model(xa, ya)
            total += (u_pred - vi) ** 2
        return total / len(anchor_coords) if anchor_coords else torch.tensor(0.0)
    
    def total_loss(self, x, y, x_b, y_b, anchor_coords=None, anchor_values=None, anchor_weight=0.1):
        """
        Compute weighted total loss.
        
        Args:
            x, y: Interior domain points
            x_b, y_b: Boundary points
            anchor_coords: Optional anchor point coordinates
            anchor_values: Optional anchor point values
            anchor_weight: Weight for anchor loss
        
        Returns:
            Total loss
        """
        loss_pde = self.pde_loss(x, y)
        loss_bc = self.boundary_loss(x_b, y_b)
        loss_anchor = 0.0
        
        if anchor_coords is not None and anchor_values is not None:
            loss_anchor = anchor_weight * self.anchor_loss(anchor_coords, anchor_values)
        
        return loss_pde + loss_bc + loss_anchor
    
    def train(
        self,
        x, y, x_b, y_b,
        anchor_coords=None,
        anchor_values=None,
        anchor_weight=0.1,
        lr=1e-3,
        epochs=5000,
        verbose=True,
        print_every=1000
    ):
        """
        Train the Helmholtz solver.
        
        Args:
            x, y: Interior domain points
            x_b, y_b: Boundary points
            anchor_coords: Optional anchor coordinates
            anchor_values: Optional anchor values
            anchor_weight: Weight for anchor loss
            lr: Learning rate
            epochs: Number of epochs
            verbose: Print updates
            print_every: Print frequency
        
        Returns:
            Trained model and losses
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        losses = []
        
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            loss = self.total_loss(
                x, y, x_b, y_b,
                anchor_coords, anchor_values,
                anchor_weight
            )
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and epoch % print_every == 0:
                print(f"Epoch {epoch:>5d}  Loss={loss.item():.3e}")
        
        return self.model, losses
