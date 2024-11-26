import torch
import torch.nn as nn

from src.models.base_model import BaseCTRModel
from src.layers.interaction_layer import CrossInteractionLayer


class DCNv2(BaseCTRModel):
    def __init__(self, num_numeric, categorical_dims, embedding_dim, deep_dims, num_cross_layers, dropout):
        super(DCNv2, self).__init__(num_numeric, categorical_dims, embedding_dim, deep_dims, dropout)

        # Cross Network
        cross_input_dim = num_numeric + len(categorical_dims) * embedding_dim
        self.cross_network = CrossInteractionLayer(cross_input_dim, num_cross_layers)

    def forward(self, numeric, categorical):
        # Common Embedding and Feature Concatenation
        x = self.forward_embeddings(numeric, categorical)

        # Cross Network
        cross_x = self.cross_network.forward(x)

        # Deep Part
        deep_output = self.deep_layers(x)

        # Combine Cross and Deep outputs
        output = torch.sigmoid(cross_x.sum(dim=1) + deep_output.squeeze(1))
        
        return output