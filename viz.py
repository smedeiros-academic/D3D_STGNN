"""
viz.py

Plotting and full-field inference helpers shared by the temporal-split and node-split workflows in the notebook.
"""

import copy

import numpy as np
import torch
import matplotlib.pyplot as plt

from gconvgru import GConvGRUModel


def plot_loss_curve(history, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs_range, history["train_loss"], label="Train Loss")
    ax.plot(epochs_range, history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_parity(results, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(results["y_true"], results["y_pred"], s=4, alpha=0.4)
    lims = [
        min(results["y_true"].min(), results["y_pred"].min()),
        max(results["y_true"].max(), results["y_pred"].max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="Ideal (y = x)")
    ax.set_xlabel("True Value")
    ax.set_ylabel("Predicted Value")
    ax.set_title(
        f"{title}\n(RMSE={results['rmse']:.4f}, "
        f"MAE={results['mae']:.4f}, NSE={results['nse']:.4f})"
    )
    ax.legend()
    plt.show()


def plot_spatial(results, dataset, timestep_idx=-1, title_prefix=""):
    """
    Side-by-side predicted vs. true Δbed map at one timestep.
    Handles both splits automatically via results['split_mode']:
      - temporal: all N nodes shown, at a held-out test timestep
      - node:     only test nodes shown (rest grayed out), at one timestep
    """
    pos_np = dataset.pos.cpu().numpy().astype(int)
    mmax, nmax = pos_np[:, 0].max() + 1, pos_np[:, 1].max() + 1

    if results["split_mode"] == "temporal":
        n_rows = len(results["test_idx"])
        y_true_2d = results["y_true"].reshape(n_rows, dataset.N)
        y_pred_2d = results["y_pred"].reshape(n_rows, dataset.N)
        node_pos = pos_np
        ti = timestep_idx % n_rows
        actual_t = results["test_idx"][ti]
    else:  # node
        T = len(dataset)
        test_mask_np = results["test_mask"].numpy()
        n_test = int(test_mask_np.sum())
        y_true_2d = results["y_true"].reshape(T, n_test)
        y_pred_2d = results["y_pred"].reshape(T, n_test)
        node_pos = pos_np[test_mask_np]
        ti = timestep_idx % T
        actual_t = ti

    def to_grid(values_1d):
        grid = np.full((mmax, nmax), np.nan, dtype=np.float32)
        grid[node_pos[:, 0], node_pos[:, 1]] = values_1d
        return grid

    pred_grid = to_grid(y_pred_2d[ti])
    true_grid = to_grid(y_true_2d[ti])

    cmap = copy.copy(plt.get_cmap("terrain"))
    cmap.set_bad(color="lightgray")  # NaN / non-test cells
    vmin = min(np.nanmin(pred_grid), np.nanmin(true_grid))
    vmax = max(np.nanmax(pred_grid), np.nanmax(true_grid))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axes[0].imshow(pred_grid, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{title_prefix}Predicted Δbed (Timestep {actual_t})")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(true_grid, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{title_prefix}True Δbed (Timestep {actual_t})")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def predict_full_field(model, dataset, device=None):
    """
    Run the trained model over the FULL graph and return per-node predictions
    for EVERY node at every timestep, shape (T, N).

    The node-split results dict only stores predictions for the ~15% held-out
    test nodes, which are scattered individually across the mesh under the
    random split -> a spatially incoherent map. But the model message-passes
    over the whole graph regardless (the mask only restricts the loss), so it
    can predict every node. This produces the dense field needed for a
    readable spatial map.

    Handles both the recurrent GConvGRUModel (hidden state threaded across
    timesteps in order) and the static BaselineGCN / ChebGCN. Forward-only
    under no_grad, so it is cheap (~1-2 GB, no BPTT graph).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    edge_index = dataset.edge_index.to(device)
    T, N = len(dataset), dataset.N
    preds = np.empty((T, N), dtype=np.float32)

    recurrent = isinstance(model, GConvGRUModel)
    h1 = h2 = None
    with torch.no_grad():
        for t in range(T):
            x = dataset[t]["x"].to(device)
            if recurrent:
                pred, h1, h2 = model(x, edge_index, h1, h2)
            else:
                pred = model(x, edge_index)
            preds[t] = pred.detach().cpu().numpy()
    return preds


def plot_spatial_field(y_pred_full, dataset, timestep_idx=-1, results=None,
                       robust_pct=99.0, cmap_name="RdBu_r", title_prefix=""):
    """
    Full-domain 3-panel Δbed map at one timestep: Predicted | True | Error.

    Δbed is signed (erosion < 0 < deposition) and sparse (most of the seabed
    is static, with strong localized channels), so we use a DIVERGING colormap
    centered at 0 with ROBUST symmetric color limits (the robust_pct-th
    percentile of |value|). Without the robust limit a few ±5 m outliers would
    wash the whole field to near-white. The Predicted and True panels share
    one limit so they are directly comparable; the Error panel gets its own.
    Invalid / dry cells (not in the graph) are drawn light gray.

    Parameters
    ----------
    y_pred_full : np.ndarray (T, N)   -- output of predict_full_field()
    dataset     : Delft3DDataset      -- supplies true Y and node (m,n) positions
    timestep_idx: int   -- which timestep to show (default -1 = last, has
                           signal; t=0,1 are all-zero morphology spin-up)
    results     : dict, optional -- if given, its held-out test RMSE/MAE are
                                    shown in the title (honest metrics on
                                    unseen nodes)
    """
    T = y_pred_full.shape[0]
    ti = timestep_idx % T

    pos = dataset.pos.cpu().numpy().astype(int)
    mmax, nmax = pos[:, 0].max() + 1, pos[:, 1].max() + 1

    pred_vals = y_pred_full[ti]
    true_vals = dataset.Y[ti].cpu().numpy()
    err_vals = pred_vals - true_vals

    def to_grid(vals):
        grid = np.full((mmax, nmax), np.nan, dtype=np.float32)
        grid[pos[:, 0], pos[:, 1]] = vals
        return grid

    pred_grid = to_grid(pred_vals)
    true_grid = to_grid(true_vals)
    err_grid = to_grid(err_vals)

    # Robust symmetric limits (eps guards the all-zero spin-up steps t=0,1).
    eps = 1e-6
    lim = max(float(np.nanpercentile(np.abs(np.concatenate([pred_vals, true_vals])), robust_pct)), eps)
    elim = max(float(np.nanpercentile(np.abs(err_vals), robust_pct)), eps)

    cmap = copy.copy(plt.get_cmap(cmap_name))  # diverging: blue = erosion, red = deposition
    cmap.set_bad(color="lightgray")

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    panels = [
        ("Predicted Δbed", pred_grid, lim),
        ("True Δbed", true_grid, lim),
        ("Error (pred − true)", err_grid, elim),
    ]
    for ax, (name, grid, vlim) in zip(axes, panels):
        im = ax.imshow(grid, cmap=cmap, origin="lower", vmin=-vlim, vmax=vlim)
        ax.set_title(name)
        ax.set_xlabel("n")
        ax.set_ylabel("m")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δbed (m)")

    suptitle = f"{title_prefix}Full-domain Δbed  (timestep {ti})"
    if results is not None and "rmse" in results:
        suptitle += f"   |   held-out test RMSE={results['rmse']:.4f}  MAE={results['mae']:.4f}"
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()
