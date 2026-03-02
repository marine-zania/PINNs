"""Automatic differentiation utilities for computing derivatives."""

import torch


def gradient(u, x, create_graph=True):
    """
    Compute gradient du/dx.
    
    Args:
        u: Output tensor
        x: Input tensor (requires grad)
        create_graph: Whether to create computation graph for higher derivatives
    
    Returns:
        Gradient tensor
    """
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=create_graph,
        retain_graph=True
    )[0]


def laplacian(u, x, y):
    """
    Compute Laplacian ∇²u = ∂²u/∂x² + ∂²u/∂y².
    
    Args:
        u: Output tensor
        x: X coordinate tensor (requires grad)
        y: Y coordinate tensor (requires grad)
    
    Returns:
        Laplacian tensor
    """
    du_dx = gradient(u, x)
    du_dy = gradient(u, y)
    
    d2u_dx2 = gradient(du_dx, x)
    d2u_dy2 = gradient(du_dy, y)
    
    return d2u_dx2 + d2u_dy2


def laplacian_3d(u, x, y, z):
    """
    Compute 3D Laplacian ∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z².
    
    Args:
        u: Output tensor
        x, y, z: Coordinate tensors (requires grad)
    
    Returns:
        Laplacian tensor
    """
    du_dx = gradient(u, x)
    du_dy = gradient(u, y)
    du_dz = gradient(u, z)
    
    d2u_dx2 = gradient(du_dx, x)
    d2u_dy2 = gradient(du_dy, y)
    d2u_dz2 = gradient(du_dz, z)
    
    return d2u_dx2 + d2u_dy2 + d2u_dz2
