import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super(DenseLayer, self).__init__()
        layers = [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)
