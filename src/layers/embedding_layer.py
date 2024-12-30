import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    An embedding layer.
    """

    def __init__(self, num_features, embedding_dim):
        """
        Initialize the EmbeddingLayer.

        Parameters:
            num_features (list[int]): A list where each element represents the size
                                       of the vocabulary for a categorical feature.
            embedding_dim (int): The dimension of the embedding vector for each feature.
        """
        super(EmbeddingLayer, self).__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(input_dim, embedding_dim) for input_dim in num_features]
        )

    def forward(self, x):
        """
        Perform the forward pass.

        Embeds each categorical feature and concatenates the embeddings.

        Parameters:
            x (torch.Tensor): A tensor of categorical feature indices with shape
                              (batch_size, num_features).

        Returns:
            torch.Tensor: A concatenated tensor of embeddings with shape
                          (batch_size, num_features * embedding_dim).
        """
        return torch.cat(
            [self.embeddings[i](x[:, i]) for i in range(len(self.embeddings))], dim=1
        )
