"""
train.py

Training loop for the Delft3D GNN project.

Any model with signature:
    forward(x: Tensor[N, F], edge_index: Tensor[2, E]) -> Tensor[N]
can be passed to `train()`.

Recurrent models (GConvGRUModel, which returns (pred, h1, h2)) train via the
dedicated loop in train_gconvgru.py instead.
"""

import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv
from torch_geometric.transforms import RandomNodeSplit


########## Metrics

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))

def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else float("nan")


########## Loss

def make_weighted_mse(sigma: float, alpha: float = 0.0, w_cap: float = 50.0):
    """
    MSE with per-node weights  w = 1 + alpha * |y| / sigma  (clamped to w_cap).

    The deltaZ target is ~78% zeros with rare strong erosion/deposition spikes,
    so plain MSE is minimized by predicting ~0 everywhere (zero-collapse).
    Weighting active cells by their magnitude makes them dominate the gradient.

    alpha=0 reproduces plain MSE exactly. sigma should be the std of the
    TRAINING targets only (it is just a loss scale, but computing it on the
    train subset avoids any test leakage).
    """
    sigma = max(float(sigma), 1e-12)

    def criterion(pred, y):
        w = (1.0 + alpha * y.abs() / sigma).clamp(max=w_cap)
        return (w * (pred - y) ** 2).mean()

    return criterion


def collapse_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, active_tau: float = 0.005):
    """
    Metrics that expose zero-collapse, which plain RMSE hides (a model that
    predicts ~0 everywhere gets a flattering RMSE on this mostly-zero target).

    rmse_zero      : RMSE of an all-zeros predictor (the do-nothing baseline)
    skill_vs_zero  : 1 - rmse/rmse_zero  (> 0 means the model beats the baseline)
    std_ratio      : pred std / true std (~1 healthy, <<1 means collapse)
    active_rmse    : RMSE restricted to cells with |y_true| > active_tau
    active_frac    : fraction of cells that are active at that threshold
    """
    r = rmse(y_true, y_pred)
    r0 = rmse(y_true, np.zeros_like(y_true))
    act = np.abs(y_true) > active_tau
    return {
        "rmse_zero":     r0,
        "skill_vs_zero": float(1.0 - r / r0) if r0 > 0 else float("nan"),
        "std_ratio":     float(y_pred.std() / y_true.std()) if y_true.std() > 0 else float("nan"),
        "active_rmse":   rmse(y_true[act], y_pred[act]) if act.any() else float("nan"),
        "active_frac":   float(act.mean()),
    }


########## Utilities

def enable_laplacian_cache(model):
    """
    Memoize each internal ChebConv's normalized Laplacian.

    ChebConv recomputes get_laplacian on every forward (it has no `cached`
    flag, unlike GCNConv) and profiling showed that recompute is ~95% of each
    ChebConv call on this graph. The project's graph is static with
    edge_weight=None, so __norm__ returns an identical (edge_index, norm)
    every call -> compute once and reuse. Exact (validated max output diff
    2.4e-7). No-op for models without ChebConv layers. Benefits ChebGCN here
    and the GConvGRU cells in train_gconvgru.py (6 ChebConvs per cell).
    """
    for _, mod in model.named_modules():
        if isinstance(mod, ChebConv):
            orig = mod.__norm__            # bound method (self baked in)
            store = {}
            def cached_norm(edge_index, num_nodes, edge_weight, normalization,
                            lambda_max, dtype=None, batch=None, _orig=orig, _store=store):
                key = int(num_nodes)
                if key not in _store:
                    _store[key] = _orig(edge_index, num_nodes, edge_weight,
                                        normalization, lambda_max, dtype, batch)
                return _store[key]
            object.__setattr__(mod, "__norm__", cached_norm)
    return model


def temporal_split(total: int, train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Chronological train / val / test split.
    Returns (train_idx, val_idx, test_idx) as lists of integers.
    """
    n_train = int(total * train_frac)
    n_val   = int(total * (train_frac + val_frac))
    return list(range(0, n_train)), list(range(n_train, n_val)), list(range(n_val, total))


def _run_epoch(model, subset, edge_index, criterion, device, optimizer=None, node_mask=None):
    """
    One pass over a dataset subset. Returns (avg_loss, y_true, y_pred).

    node_mask : optional bool Tensor [N]
        If provided, the model still forward-passes over the *full* graph
        (so message passing is unaffected), but the loss and returned
        y_true/y_pred are restricted to the masked nodes. Used for
        node-level (transductive) splitting, where every timestep is seen
        in every phase and only the node subset differs between
        train/val/test.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    loader = DataLoader(subset, batch_size=1, shuffle=training)
    total_loss  = 0.0
    y_true_list = []
    y_pred_list = []

    mask = node_mask.to(device) if node_mask is not None else None

    for batch in loader:
        x  = batch["x"].squeeze(0).to(device)   # [N, F]
        y  = batch["y"].squeeze(0).to(device)   # [N]
        ei = edge_index.to(device)              # [2, E]

        if training:
            optimizer.zero_grad()
            pred_full = model(x, ei)
            pred   = pred_full[mask] if mask is not None else pred_full
            y_eval = y[mask]         if mask is not None else y
            loss = criterion(pred, y_eval)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pred_full = model(x, ei)
                pred   = pred_full[mask] if mask is not None else pred_full
                y_eval = y[mask]         if mask is not None else y
                loss = criterion(pred, y_eval)

        total_loss  += loss.item()
        y_true_list.append(y_eval.detach().cpu().numpy().ravel())
        y_pred_list.append(pred.detach().cpu().numpy().ravel())

    n = max(len(loader), 1)
    return total_loss / n, np.concatenate(y_true_list), np.concatenate(y_pred_list)


def node_random_split(num_nodes: int, val_frac: float = 0.15, test_frac: float = 0.15):
    """
    Randomly split node indices into disjoint train/val/test boolean masks
    using PyG's RandomNodeSplit (transductive setting: the full graph is
    used for message passing in every phase; only the loss/metric mask
    differs between phases).

    split="train_rest" is used rather than split="random" because "random"
    stratifies by class label (data.y), which assumes discrete class
    labels. Our targets are continuous, so "train_rest" (uniform
    torch.randperm over all nodes, remainder goes to train) is the correct
    mode. key=None so the transform doesn't require any label attribute
    to be present at all.

    Returns
    -------
    train_mask, val_mask, test_mask : bool Tensor [num_nodes]
    """
    data = Data(num_nodes=num_nodes)
    transform = RandomNodeSplit(
        split="train_rest", num_val=val_frac, num_test=test_frac, key=None
    )
    data = transform(data)
    return data.train_mask, data.val_mask, data.test_mask


########### Define Function For Jupyter Notebook

def train(
    model: nn.Module,
    dataset,
    *,
    split_mode:      str   = "temporal",
    train_frac:      float = 0.8,
    val_frac:        float = 0.1,
    lr:              float = 1e-3,
    weight_decay:    float = 1e-4,
    epochs:          int   = 100,
    patience:        int   = 10,
    loss_alpha:      float = 0.0,
    scheduler:       str   = None,
    active_tau:      float = 0.005,
    checkpoint_path: str   = "best_model.pt",
    history_path:    str   = None,
    device                 = None,
    verbose:         bool  = True,
    node_split_seed: int   = None,
):
    """
    Train any model with signature  forward(x, edge_index) -> Tensor[N].

    Parameters
    ----------
    model           : nn.Module
    dataset         : Delft3DDataset (or any Dataset whose __getitem__ returns
                      dicts with keys 'x' and 'y', and that has a .edge_index
                      attribute holding the static graph connectivity [2, E])
    split_mode      : "temporal" (default) splits whole timesteps into
                      train/val/test, as before. "node" instead uses every
                      timestep in every phase and splits the nodes into
                      disjoint train/val/test sets (transductive), which is
                      a better fit when there are very few timesteps but a
                      very large graph per timestep.
    train_frac      : "temporal" -> fraction of timesteps used for training.
                      "node" -> fraction of nodes used for training.
    val_frac        : fraction for validation (the remainder goes to test) in
                      either mode.
    lr              : Adam learning rate
    weight_decay    : L2 regularisation strength
    epochs          : maximum training epochs
    patience        : early-stopping patience (epochs without val improvement)
    loss_alpha      : weighted-MSE strength (see make_weighted_mse). 0 = plain
                      MSE (exact previous behavior). When > 0, history loss
                      values are weighted and not comparable to alpha=0 runs;
                      reported RMSE/MAE/etc. remain plain/physical.
    scheduler       : None (fixed lr, previous behavior) or "plateau"
                      (ReduceLROnPlateau on val_loss: factor 0.5, patience 5,
                      min_lr 1e-5 — for when val loss stalls at a fixed lr)
    active_tau      : |y| threshold (m) for the active-cell diagnostics
    checkpoint_path : path to save best model weights (.pt)
    history_path    : if set, write per-epoch metrics as JSON to this path
    device          : torch.device  (auto-detects CUDA when None)
    verbose         : print per-epoch metrics to stdout
    node_split_seed : optional int, sets torch's RNG seed before the node
                      split so it's reproducible (only used when
                      split_mode="node")

    Returns
    -------
    history : dict
        train_loss, val_loss, train_rmse, val_rmse (one value per epoch)
    results : dict
        test_loss, mae, rmse, bias, nse,
              y_true (np.ndarray), y_pred (np.ndarray), test_idx (list),
              split_mode, and (for split_mode="node") train_mask, val_mask,
              test_mask (bool Tensor [N] each)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = enable_laplacian_cache(model.to(device))  # no-op unless ChebConv layers

    edge_index = dataset.edge_index          # [2, E] – static for all timesteps

    assert split_mode in ("temporal", "node"), "split_mode must be 'temporal' or 'node'"

    if split_mode == "temporal":
        T = len(dataset)
        train_idx, val_idx, test_idx = temporal_split(T, train_frac, val_frac)

        train_sub = Subset(dataset, train_idx)
        val_sub   = Subset(dataset, val_idx)
        test_sub  = Subset(dataset, test_idx)

        train_mask = val_mask = test_mask = None

    else:  # split_mode == "node"
        # Every timestep is used in every phase; only the node subset
        # (via boolean masks applied to the loss) differs between them.
        train_sub = val_sub = test_sub = dataset
        test_idx  = list(range(len(dataset)))   # kept for reporting/plotting

        if node_split_seed is not None:
            torch.manual_seed(node_split_seed)

        train_mask, val_mask, test_mask = node_random_split(
            num_nodes=dataset.N,
            val_frac=val_frac,
            test_frac=1.0 - train_frac - val_frac,
        )
        if verbose:
            print(
                f"Node split -> train: {int(train_mask.sum()):,}  "
                f"val: {int(val_mask.sum()):,}  test: {int(test_mask.sum()):,} "
                f"(of {dataset.N:,} nodes)"
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss scale from the TRAINING subset only (avoids test leakage).
    if split_mode == "temporal":
        sigma = float(dataset.Y[train_idx].std())
    else:
        sigma = float(dataset.Y[:, train_mask.to(dataset.Y.device)].std())
    criterion = make_weighted_mse(sigma, loss_alpha)

    assert scheduler in (None, "plateau"), "scheduler must be None or 'plateau'"
    sched = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-5)
        if scheduler == "plateau" else None
    )

    history = {"train_loss": [], "val_loss": [], "train_rmse": [], "val_rmse": []}
    best_val         = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        tr_loss, y_tr, y_tr_pred = _run_epoch(
            model, train_sub, edge_index, criterion, device, optimizer, node_mask=train_mask
        )
        val_loss, y_val, y_val_pred = _run_epoch(
            model, val_sub, edge_index, criterion, device, node_mask=val_mask
        )

        if sched is not None:
            sched.step(val_loss)

        tr_rmse  = rmse(y_tr,  y_tr_pred)
        val_rmse = rmse(y_val, y_val_pred)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(tr_rmse)
        history["val_rmse"].append(val_rmse)

        if verbose:
            msg = (
                f"Epoch {epoch:03d} | "
                f"train_loss={tr_loss:.6f}  val_loss={val_loss:.6f} | "
                f"train_rmse={tr_rmse:.6f}  val_rmse={val_rmse:.6f}"
            )
            if sched is not None:
                msg += f" | lr={optimizer.param_groups[0]['lr']:.2e}"
            print(msg)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print("Early stopping triggered.")
                break

    ########### Evaluate on held-out test set
    chkpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(chkpt["model_state_dict"])

    test_loss, y_test, y_test_pred = _run_epoch(
        model, test_sub, edge_index, criterion, device, node_mask=test_mask
    )

    results = {
        "test_loss":  test_loss,
        "mae":        mae(y_test,  y_test_pred),
        "rmse":       rmse(y_test, y_test_pred),
        "bias":       bias(y_test, y_test_pred),
        "nse":        nse(y_test,  y_test_pred),
        "y_true":     y_test,
        "y_pred":     y_test_pred,
        "test_idx":   test_idx,
        "split_mode": split_mode,
    }
    results.update(collapse_diagnostics(y_test, y_test_pred, active_tau))
    if split_mode == "node":
        results["train_mask"] = train_mask
        results["val_mask"]   = val_mask
        results["test_mask"]  = test_mask

    if verbose:
        print("\nTest results:")
        for k in ("test_loss", "mae", "rmse", "bias", "nse",
                  "rmse_zero", "skill_vs_zero", "std_ratio", "active_rmse", "active_frac"):
            print(f"  {k:<13s}: {results[k]:.6f}")

    if history_path is not None:
        serializable = {k: [float(v) for v in vs] for k, vs in history.items()}
        with open(history_path, "w") as f:
            json.dump(serializable, f, indent=2)

    return history, results