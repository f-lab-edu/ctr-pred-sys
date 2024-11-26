import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, num_features, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(input_dim, embedding_dim) for input_dim in num_features]
        )

    def forward(self, x):
        return torch.cat(
            [self.embeddings[i](x[:, i]) for i in range(len(self.embeddings))], dim=1
        )
