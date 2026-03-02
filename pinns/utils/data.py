"""Data generation utilities for domain and boundary sampling."""

import numpy as np
import torch


def create_domain_points(x_range, y_range, nx, ny, device='cpu'):
    """
    Create uniformly sampled interior domain points.
    
    Args:
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        nx: Number of points in x
        ny: Number of points in y
        device: torch device
    
    Returns:
        (X, Y): Tensors of shape (nx*ny, 1) with requires_grad=True
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    X = torch.tensor(XX.ravel(), dtype=torch.float32, device=device).view(-1, 1)
    Y = torch.tensor(YY.ravel(), dtype=torch.float32, device=device).view(-1, 1)
    
    X.requires_grad_(True)
    Y.requires_grad_(True)
    
    return X, Y


def create_boundary_points(x_range, y_range, n_points, device='cpu'):
    """
    Create uniformly sampled boundary points.
    
    Args:
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        n_points: Number of points per boundary edge
        device: torch device
    
    Returns:
        (X_b, Y_b): Boundary point tensors with requires_grad=True
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    ys = np.linspace(y_min, y_max, n_points)
    xs = np.linspace(x_min, x_max, n_points)
    
    # Four edges: x=x_min, x=x_max, y=y_min, y=y_max
    x_min_edge = np.full(n_points, x_min)
    x_max_edge = np.full(n_points, x_max)
    y_min_edge = np.full(n_points, y_min)
    y_max_edge = np.full(n_points, y_max)
    
    X_b = np.concatenate([x_min_edge, x_max_edge, xs, xs])
    Y_b = np.concatenate([ys, ys, y_min_edge, y_max_edge])
    
    X_b = torch.tensor(X_b, dtype=torch.float32, device=device).view(-1, 1)
    Y_b = torch.tensor(Y_b, dtype=torch.float32, device=device).view(-1, 1)
    
    X_b.requires_grad_(True)
    Y_b.requires_grad_(True)
    
    return X_b, Y_b


def create_1d_points(x_range, n_points, device='cpu'):
    """
    Create 1D domain points.
    
    Args:
        x_range: (x_min, x_max)
        n_points: Number of points
        device: torch device
    
    Returns:
        X: Tensor of shape (n_points, 1) with requires_grad=True
    """
    x_min, x_max = x_range
    xs = np.linspace(x_min, x_max, n_points)
    X = torch.tensor(xs, dtype=torch.float32, device=device).view(-1, 1)
    X.requires_grad_(True)
    return X
