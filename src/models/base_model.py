import torch
import torch.nn as nn

from src.layers.embedding_layer import EmbeddingLayer
from src.layers.dense_layer import DenseLayer


class BaseCTRModel(nn.Module):
    """
    A base model for Click-Through Rate (CTR) prediction using deep learning.
    """

    def __init__(
        self, num_numeric, categorical_dims, embedding_dim, hidden_dims, dropout
    ):
        """
        Initialize the BaseCTRModel.
        
        Parameters:
            num_numeric (int): The number of numeric features.
            categorical_dims (list[int]): A list containing the sizes of categorical feature dimensions.
            embedding_dim (int): The dimension of the embedding vectors for categorical features.
            hidden_dims (list[int]): A list containing the sizes of hidden layers in the deep network.
            dropout (float): The dropout rate for regularization in the dense layers.
        """
        super(BaseCTRModel, self).__init__()
        self.embeddings = EmbeddingLayer(categorical_dims, embedding_dim)

        input_dim = num_numeric + len(categorical_dims) * embedding_dim
        self.deep_layers = nn.Sequential(
            DenseLayer(input_dim, hidden_dims[0], dropout),
            DenseLayer(hidden_dims[0], hidden_dims[1], dropout),
            nn.Linear(hidden_dims[1], 1)            
        )
    
    def forward_embeddings(self, numeric, categorical):
        """
        Forward pass for embeddings.

        Parameters:
            numeric (torch.Tensor): A tensor of numeric features.
            categorical (torch.Tensor): A tensor of categorical feature indices.

        Returns:
            torch.Tensor: A concatenated tensor of numeric and embedded categorical features.
        """
        categorical_embeddings = self.embeddings.forward(categorical)
        return torch.cat([numeric, categorical_embeddings], dim=1)
