"""
dataset.py

PyTorch Dataset for Delft3D GNN erosion prediction.

Expected files
--------------
features.npz
    X : (T, N, F)
    Y : (T, N)

edge_index.npy
    (2, E)

node_index.npy
    (N, 2)

meta.json
"""

import json
import numpy as np
import torch

from torch.utils.data import Dataset

class Delft3DDataset(Dataset):
    """
    Dataset returning one graph snapshot per timestep.
    Returns
    -------
    sample : dict
        x            Node features (N,F)
        y            Targets (N,)
        edge_index   Graph connectivity
        pos          Node coordinates
        timestep     Integer timestep
    """
    def __init__(
        self,
        feature_file,
        edge_file,
        node_file,
        meta_file,
        normalize=True,
        device=None,
    ):
        ########## Load files

        data = np.load(feature_file)

        self.X = data["X"].astype(np.float32)
        self.Y = data["Y"].astype(np.float32)

        self.edge_index = np.load(edge_file).astype(np.int64)

        self.pos = np.load(node_file).astype(np.float32)

        with open(meta_file, "r") as f:
            self.meta = json.load(f)

        ########## Dataset dimensions

        self.T = self.X.shape[0]
        self.N = self.X.shape[1]
        self.F = self.X.shape[2]


        ########## Normalize features

        self.normalize = normalize

        if normalize:

            self.mean = self.X.mean(axis=(0, 1), keepdims=True)
            self.std = self.X.std(axis=(0, 1), keepdims=True)
            self.std[self.std == 0] = 1.0
            self.X = (self.X - self.mean) / self.std

        ########## Convert to tensors

        self.X = torch.from_numpy(self.X)
        self.Y = torch.from_numpy(self.Y)
        self.edge_index = torch.from_numpy(self.edge_index)
        self.pos = torch.from_numpy(self.pos)

        if device is not None:

            self.X = self.X.to(device)
            self.Y = self.Y.to(device)
            self.edge_index = self.edge_index.to(device)
            self.pos = self.pos.to(device)

    ########## Required Dataset functions

    def __len__(self):
        return self.T
    def __getitem__(self, idx):

        sample = {
            "x": self.X[idx],
            "y": self.Y[idx],
            "edge_index": self.edge_index,
            "pos": self.pos,
            "timestep": idx
        }
        return sample

    ########## Convenience functions

    @property
    def feature_names(self):
        return self.meta["features"]
    def get_graph(self, idx):
        """
        Returns graph components separately.
        """
        return (
            self.X[idx],
            self.edge_index,
            self.Y[idx],
            self.pos
        )

    def summary(self):

        print("--------------- Dataset Summary ---------------")
        print(f"Timesteps : {self.T}")
        print(f"Nodes     : {self.N:,}")
        print(f"Features  : {self.F}")
        print(f"Edges     : {self.edge_index.shape[1]:,}")
        print("Feature names:")
        for f in self.feature_names:
            print(f"   - {f}")
        print("-----------------------------------------------")