import torch
import torch.nn as nn

from torch import Tensor


class GeneWeightedAnnotation(nn.Module):
    """Two-layer classifier with learnable per-gene attention

    Parameters
    ----------
    n_genes : int
        Number of genes used for classification
    n_classes : int
        Number of cell type classes
    hidden_dim : int
        Number of hidden features in gene weighting layer
    dropout : float
        Dropout probability applied to input features
    """

    def __init__(
        self,
        n_genes: int,
        n_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.5
    ):
        super(GeneWeightedAnnotation, self).__init__()

        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("Dropout must be 0.0 <= p <= 1.0")

        self.features = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_genes, n_genes),
            nn.ReLU(inplace=True),
            nn.Linear(n_genes, n_genes),
        )

        self.attention = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_genes, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_genes)
        )

        self.head = nn.Linear(n_genes, n_classes)

    def forward(self, x: Tensor, return_weights: bool = False):
        x = self.features(x)
        a = self.attention(x)
        a = torch.sigmoid(a)
        z = a * x

        if return_weights:
            return self.head(z).squeeze(1), a

        return self.head(z).squeeze(1)
