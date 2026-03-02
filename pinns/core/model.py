"""Base PINN neural network model."""

import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) model.
    
    A fully-connected feedforward neural network with configurable architecture
    for solving physics-based problems.
    
    Args:
        input_dim (int): Input dimension (default: 2 for 2D PDEs)
        hidden_dim (int): Number of neurons in hidden layers (default: 50)
        hidden_layers (int): Number of hidden layers (default: 3)
        output_dim (int): Output dimension (default: 1)
        activation (str): Activation function type (default: 'tanh')
    """
    
    def __init__(
        self,
        input_dim=2,
        hidden_dim=50,
        hidden_layers=3,
        output_dim=1,
        activation='tanh'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        
        # Build network
        layers = []
        in_dim = input_dim
        
        for i in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self._activation_fn(activation))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    @staticmethod
    def _activation_fn(name):
        """Get activation function."""
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'sin': nn.SiLU(),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]
    
    @staticmethod
    def _init_weights(m):
        """Xavier uniform initialization."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, *inputs):
        """
        Forward pass.
        
        Args:
            *inputs: Variable number of input tensors (e.g., x, y for 2D)
        
        Returns:
            Output tensor
        """
        # Concatenate all inputs
        if len(inputs) == 1:
            xy = inputs[0]
        else:
            xy = torch.cat(inputs, dim=1)
        
        return self.net(xy)
