"""Visualization utilities for PINNs."""

import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_1d_solution(model, x_range, n_points=200, device='cpu', ax=None, title=None):
    """
    Plot 1D PINN solution.
    
    Args:
        model: PINN model
        x_range: (x_min, x_max)
        n_points: Number of plot points
        device: torch device
        ax: matplotlib axis (creates new if None)
        title: Plot title
    
    Returns:
        ax: matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    x_min, x_max = x_range
    x_plot = np.linspace(x_min, x_max, n_points)
    x_torch = torch.tensor(x_plot, dtype=torch.float32, device=device).view(-1, 1)
    
    with torch.no_grad():
        u_plot = model(x_torch).cpu().numpy()
    
    ax.plot(x_plot, u_plot, 'b-', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_2d_solution(model, x_range, y_range, n_points=100, device='cpu', ax=None, title=None, cmap='jet'):
    """
    Plot 2D PINN solution as contourf.
    
    Args:
        model: PINN model
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        n_points: Resolution
        device: torch device
        ax: matplotlib axis
        title: Plot title
        cmap: Colormap
    
    Returns:
        ax, contourf handle
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    x = np.linspace(x_min, x_max, n_points)
    y = np.linspace(y_min, y_max, n_points)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    X_torch = torch.tensor(X.ravel(), dtype=torch.float32, device=device).view(-1, 1)
    Y_torch = torch.tensor(Y.ravel(), dtype=torch.float32, device=device).view(-1, 1)
    
    with torch.no_grad():
        U = model(X_torch, Y_torch).cpu().numpy().reshape(n_points, n_points)
    
    cf = ax.contourf(x, y, U.T, levels=100, cmap=cmap)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if title:
        ax.set_title(title)
    
    return ax, cf


def plot_modes_grid(models, x_range, y_range, mode_list, analytical_k2_vals, device='cpu'):
    """
    Plot multiple modes in a grid (e.g., for Helmholtz equation).
    
    Args:
        models: List of trained PINN models
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        mode_list: List of (m, n) mode tuples
        analytical_k2_vals: List of k² values
        device: torch device
    
    Returns:
        fig: matplotlib figure
    """
    n_modes = len(models)
    n_cols = 4
    n_rows = (n_modes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    n_plot = 100
    
    x = np.linspace(x_min, x_max, n_plot)
    y = np.linspace(y_min, y_max, n_plot)
    X, Y = np.meshgrid(x, y, indexing='ij')
    X_torch = torch.tensor(X.ravel(), dtype=torch.float32, device=device).view(-1, 1)
    Y_torch = torch.tensor(Y.ravel(), dtype=torch.float32, device=device).view(-1, 1)
    
    for idx, (model, (m, n), k2_val) in enumerate(zip(models, mode_list, analytical_k2_vals)):
        ax = axes[idx]
        
        with torch.no_grad():
            U = model(X_torch, Y_torch).cpu().numpy().reshape(n_plot, n_plot)
        
        U_norm = U / (np.abs(U).max() + 1e-10)
        cf = ax.contourf(x, y, U_norm.T, levels=100, cmap='jet', vmin=-1, vmax=1)
        ax.set_title(f"TE({m},{n}), $k^2={k2_val:.3f}$", fontsize=10)
        ax.set_xlabel('x' if idx // n_cols == n_rows - 1 else '')
        ax.set_ylabel('y' if idx % n_cols == 0 else '')
    
    # Hide unused subplots
    for idx in range(n_modes, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle("PINN Solutions", fontsize=14, y=1.00)
    fig.tight_layout()
    
    return fig
