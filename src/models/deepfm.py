import torch
import torch.nn as nn

from src.models.base_model import BaseCTRModel
from src.layers.interaction_layer import FMInteractionLayer
from src.utils.load import load_yaml




class DeepFM(BaseCTRModel):
    def __init__(self, num_numeric, categorical_dims, embedding_dim, deep_dims, dropout):
        super(DeepFM, self).__init__(num_numeric, categorical_dims, embedding_dim, deep_dims, dropout)

        # FM Part (First-order and Second-order interactions)
        fm_input_dim = num_numeric + len(categorical_dims) * embedding_dim
        self.fm_network = FMInteractionLayer(fm_input_dim)

    def forward(self, numeric, categorical):
        # Common Embedding and Feature Concatenation
        x = self.forward_embeddings(numeric, categorical)

        # FM Part
        fm_first_order, fm_second_order = self.fm_network.forward(x)

        # Deep Part
        deep_output = self.deep_layers(x)

        # Combine FM and Deep outputs
        output = fm_first_order + fm_second_order + deep_output
        return torch.sigmoid(output).squeeze(1)
