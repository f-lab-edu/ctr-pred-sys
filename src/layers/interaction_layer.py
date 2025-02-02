import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple


class FMInteractionLayer(nn.Module):
    """
    Factorization Machine (FM) Interaction Layer for modeling feature interactions.
    """

    def __init__(self, input_dim: int) -> None:
        """
        Initialize the FMInteractionLayer.

        Parameters:
            input_dim (int): The number of input features.
        """
        super(FMInteractionLayer, self).__init__()

        self.first_order = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Perform the forward pass.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            tuple: First-order interactions (torch.Tensor) and second-order interactions (torch.Tensor).
        """
        first_order = self.first_order(x)
        second_order = torch.sum(x, dim=1, keepdim=True)

        return first_order, second_order


class CrossInteractionLayer(nn.Module):
    """
    Cross Interaction Layer for feature crossing.
    """

    def __init__(self, input_dim: int, num_layers: int) -> None:
        """
        Initialize the CrossInteractionLayer.

        Parameters:
            input_dim (int): The number of input features.
            num_layers (int): The number of cross-interaction layers.
        """
        super(CrossInteractionLayer, self).__init__()

        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(num_layers)]
        )
        self.weight = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim)) for _ in range(num_layers)]
        )
        self.bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform the forward pass.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: The output tensor after applying cross-interaction layers.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x * self.weight[i]) + self.bias[i] + x
        return x
