import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class BaselineGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, 1)
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