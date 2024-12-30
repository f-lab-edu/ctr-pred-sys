import torch
import torch.nn as nn

from src.models.base_model import BaseCTRModel
from src.layers.interaction_layer import FMInteractionLayer
from src.utils.load import load_yaml




class DeepFM(BaseCTRModel):
    """
    A DeepFM model for Click-Through Rate (CTR) prediction.
    """

    def __init__(self, num_numeric, categorical_dims, embedding_dim, deep_dims, dropout):
        """
        Initialize the DeepFM model.

        Parameters:
            num_numeric (int): The number of numeric features.
            categorical_dims (list[int]): A list containing the sizes of categorical feature dimensions.
            embedding_dim (int): The dimension of the embedding vectors for categorical features.
            deep_dims (list[int]): A list containing the sizes of hidden layers in the deep network.
            dropout (float): The dropout rate for regularization in the deep layers.
        """
        super(DeepFM, self).__init__(num_numeric, categorical_dims, embedding_dim, deep_dims, dropout)

        # FM Part (First-order and Second-order interactions)
        fm_input_dim = num_numeric + len(categorical_dims) * embedding_dim
        self.fm_network = FMInteractionLayer(fm_input_dim)

    def forward(self, numeric, categorical):
        """
        Forward pass for the DeepFM model.

        Parameters:
            numeric (torch.Tensor): A tensor of numeric features.
            categorical (torch.Tensor): A tensor of categorical feature indices.

        Returns:
            torch.Tensor: A tensor containing the predicted probabilities.
        """
        # Common Embedding and Feature Concatenation
        x = self.forward_embeddings(numeric, categorical)

        # FM Part
        fm_first_order, fm_second_order = self.fm_network.forward(x)

        # Deep Part
        deep_output = self.deep_layers(x)

        # Combine FM and Deep outputs
        output = fm_first_order + fm_second_order + deep_output
        return torch.sigmoid(output).squeeze(1)
