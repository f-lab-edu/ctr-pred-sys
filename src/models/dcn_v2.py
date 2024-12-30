import torch
import torch.nn as nn

from src.models.base_model import BaseCTRModel
from src.layers.interaction_layer import CrossInteractionLayer


class DCNv2(BaseCTRModel):
    """
    A deep and cross network (DCNv2) for Click-Through Rate (CTR) prediction.
    """
    
    def __init__(self, num_numeric, categorical_dims, embedding_dim, deep_dims, num_cross_layers, dropout):
        """
        Initialize the DCNv2 model.

        Parameters:
            num_numeric (int): The number of numeric features.
            categorical_dims (list[int]): A list containing the sizes of categorical feature dimensions.
            embedding_dim (int): The dimension of the embedding vectors for categorical features.
            deep_dims (list[int]): A list containing the sizes of hidden layers in the deep network.
            num_cross_layers (int): The number of cross-interaction layers in the Cross Network.
            dropout (float): The dropout rate for regularization in the deep layers.
        """
        super(DCNv2, self).__init__(num_numeric, categorical_dims, embedding_dim, deep_dims, dropout)

        # Cross Network
        cross_input_dim = num_numeric + len(categorical_dims) * embedding_dim
        self.cross_network = CrossInteractionLayer(cross_input_dim, num_cross_layers)

    def forward(self, numeric, categorical):
        """
        Forward pass for the DCNv2 model.

        Parameters:
            numeric (torch.Tensor): A tensor of numeric features.
            categorical (torch.Tensor): A tensor of categorical feature indices.

        Returns:
            torch.Tensor: A tensor containing the predicted probabilities.
        """
        # Common Embedding and Feature Concatenation
        x = self.forward_embeddings(numeric, categorical)

        # Cross Network
        cross_x = self.cross_network.forward(x)

        # Deep Part
        deep_output = self.deep_layers(x)

        # Combine Cross and Deep outputs
        output = torch.sigmoid(cross_x.sum(dim=1) + deep_output.squeeze(1))
        
        return output