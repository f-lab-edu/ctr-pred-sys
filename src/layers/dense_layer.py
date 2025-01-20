import torch
import torch.nn as nn

from torch import Tensor


class DenseLayer(nn.Module):
    """
    A fully connected (dense) layer.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2) -> None:
        """
        Initialize the DenseLayer.

        Parameters:
            in_dim (int): The number of input features.
            out_dim (int): The number of output features.
            dropout (float, optional): The dropout rate for regularization. Default is 0.2.
        """
        super(DenseLayer, self).__init__()
        layers = [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform the forward pass.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the dense layer.
        """
        return self.fc(x)
