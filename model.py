import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class BaselineGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout: float = 0.1):
        super().__init__()
        # cached=True: the graph is static/transductive, so the normalized
        # adjacency is computed once on the first forward and reused (exact).
        self.conv1 = GCNConv(in_channels, hidden_dim1, cached=True)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2, cached=True)
        self.conv3 = GCNConv(hidden_dim2, 1, cached=True)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        return x.squeeze(-1)  # [N]


from torch_geometric.nn import ChebConv

class ChebGCN(nn.Module):
    """
    Same 3-layer structure as BaselineGCN, but with ChebConv (K-order
    Chebyshev spectral filters) in place of GCNConv, so each layer sees a
    K-hop neighborhood instead of just 1-hop. Otherwise identical: same
    forward(x, edge_index) -> Tensor[N] interface, so it plugs into
    train.py's train() with zero changes.
    """
    def __init__(self, in_channels: int, hidden_dim1: int = 64, hidden_dim2: int = 32, K: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden_dim1, K=K)
        self.conv2 = ChebConv(hidden_dim1, hidden_dim2, K=K)
        self.conv3 = ChebConv(hidden_dim2, 1, K=K)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        return x.squeeze(-1)  # [N]