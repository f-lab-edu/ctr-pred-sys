import torch
import torch.nn as nn

from src.layers.embedding_layer import EmbeddingLayer
from src.layers.dense_layer import DenseLayer


class BaseCTRModel(nn.Module):
    def __init__(
        self, num_numeric, categorical_dims, embedding_dim, hidden_dims, dropout
    ):
        super(BaseCTRModel, self).__init__()
        self.embeddings = EmbeddingLayer(categorical_dims, embedding_dim)

        # Deep Network
        input_dim = num_numeric + len(categorical_dims) * embedding_dim
        self.deep_layers = nn.Sequential(
            DenseLayer(input_dim, hidden_dims[0], dropout),
            DenseLayer(hidden_dims[0], hidden_dims[1], dropout),
            nn.Linear(hidden_dims[1], 1)            
        )
    
    def forward_embeddings(self, numeric, categorical):
        # Embedding Lookup
        categorical_embeddings = self.embeddings.forward(categorical)
        return torch.cat([numeric, categorical_embeddings], dim=1)
