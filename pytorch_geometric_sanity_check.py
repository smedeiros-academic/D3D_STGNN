#!/usr/bin/env python3

import argparse
import sys

# Optional: switch to a non-interactive backend if running headless
parser = argparse.ArgumentParser(description="PyTorch Geometric sanity check")
parser.add_argument("--no-plot", action="store_true", help="Skip graph visualization")
args = parser.parse_args()

import torch
import numpy as np

print("PyTorch version:", torch.__version__)
try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    print("PyTorch Geometric version:", torch_geometric.__version__)
except Exception as e:
    print("ERROR: Could not import torch_geometric:", e)
    sys.exit(1)

# Tiny graph: 3 nodes in a triangle
x = torch.tensor([[1.0], [2.0], [3.0]])
edge_index = torch.tensor([[0, 1, 2],
                           [1, 2, 0]], dtype=torch.long)

data = Data(x=x, edge_index=edge_index)

# Run a simple Graph Convolution
conv = GCNConv(in_channels=1, out_channels=2)
out = conv(data.x, data.edge_index)

print("Input node features:\n", data.x)
print("Output node embeddings (shape={}):\n".format(tuple(out.shape)), out)

if not args.no_plot:
    try:
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.Graph()
        edges = edge_index.numpy().T.tolist()
        G.add_edges_from(edges)

        plt.figure(figsize=(4, 4))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray",
                node_size=800, font_size=12, font_weight="bold")
        labels = {i: f"{i}\nfeat={x[i].item():.1f}" for i in range(len(x))}
        nx.draw_networkx_labels(G, pos, labels=labels)
        plt.title("Sanity Check Graph (3 nodes, triangle)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plotting skipped due to error (use --no-plot to suppress):", e)
