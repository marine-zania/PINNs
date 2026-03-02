"""Training utilities for PINNs."""

import torch
import torch.optim as optim


def train_pinn(
    model,
    loss_fn,
    optimizer=None,
    lr=1e-3,
    epochs=5000,
    verbose=True,
    print_every=1000
):
    """
    Generic PINN training loop.
    
    Args:
        model: PINN model instance
        loss_fn: Loss function that takes model and returns scalar loss
        optimizer: torch optimizer (default: Adam)
        lr: Learning rate
        epochs: Number of epochs
        verbose: Print loss updates
        print_every: Print frequency
    
    Returns:
        model: Trained model
        losses: List of total losses per epoch
    """
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        loss = loss_fn(model)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if verbose and epoch % print_every == 0:
            print(f"Epoch {epoch:>5d}  Loss={loss.item():.3e}")
    
    return model, losses
