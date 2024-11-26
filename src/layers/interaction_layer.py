import torch
import torch.nn as nn


class FMInteractionLayer(nn.Module):
    def __init__(self, input_dim):
        super(FMInteractionLayer, self).__init__()

        self.first_order = nn.Linear(input_dim, 1)

    def forward(self, x):
        first_order = self.first_order(x)
        second_order = torch.sum(x, dim=1, keepdim=True)

        return first_order, second_order


class CrossInteractionLayer(nn.Module):
    def __init__(self, input_dim, num_layers):
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

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x * self.weight[i]) + self.bias[i] + x
        return x