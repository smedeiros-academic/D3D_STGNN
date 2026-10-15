import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU

class GConvGRUModel(nn.Module):
    def __init__(self, in_feats, dropout: float = 0.1):
        super().__init__()
        self.gru1 = GConvGRU(in_feats, 32, K=3)
        self.gru2 = GConvGRU(32, 16, K=3)
        self.linear = nn.Linear(16, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, h1=None, h2=None):
        # x:  [N, F] at the current timestep.
        # h1, h2: hidden states of the two GConvGRU cells from the previous
        #   timestep (None on the first step -> GConvGRU initialises zeros).
        # The recurrent state is threaded explicitly by the caller's rollout
        # loop, so we return the updated (h1, h2) alongside the prediction.
        h1 = self.gru1(x, edge_index, H=h1)          # [N, 32]
        z = F.relu(h1)
        z = F.dropout(z, p=self.dropout, training=self.training)
        h2 = self.gru2(z, edge_index, H=h2)          # [N, 16]
        z = F.relu(h2)
        z = F.dropout(z, p=self.dropout, training=self.training)
        out = self.linear(z)                         # [N, 1]
        return out.squeeze(-1), h1, h2               # ([N], [N,32], [N,16])