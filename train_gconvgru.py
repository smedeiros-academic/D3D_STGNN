"""
train_gconvgru.py

Dedicated training loop for the recurrent GConvGRUModel (node-split only).

GConvGRU is recurrent: its forward() returns (pred, h1, h2), so it is
structurally incompatible with train.py's train() (which expects a bare
Tensor[N]). This loop threads the hidden state across the T timesteps in
order and does one full-sequence BPTT backward pass per epoch. Output format
mirrors train.py's node-split results, so the same plotting/comparison code
works for both.

Memory
------
The full-graph (N=756,809) unrolled autograd graph over all T timesteps would
need ~21 GB on the training pass, which OOMs an 8 GB GPU. To keep the exact
full-sequence BPTT gradient while fitting in memory, each timestep's forward
is wrapped in torch.utils.checkpoint: its GConvGRU activations are recomputed
during backward instead of stored, cutting peak activation memory to ~1/T
(only the small per-step h1/h2/pred boundaries are retained).
use_reentrant=False is required so gradients still flow through the module
parameters on the first step (where h1=h2=None and x has no grad) and so the
dropout RNG is preserved across the recompute.

Speed (all exact -- the trained model's math is unchanged)
----------------------------------------------------------
1. enable_laplacian_cache (from train.py): get_laplacian was ~95% of every
   ChebConv call and GConvGRU makes 12 identical calls per timestep on this
   static graph. Memorizing it measured ~18% faster (output max diff 2.4e-7).
2. X/Y are moved to the GPU once up front instead of every step of every
   epoch (H2D copies were ~8% of forward CUDA time).
3. val_every: validation costs ~24% of wall time; running it every k-th epoch
   skips most of that. Training gradients are untouched -- only the
   granularity of best-checkpoint selection changes (patience is counted in
   validation events). val_every=1 reproduces the original behavior.

Training quality (added after a 150-epoch zero-collapse diagnosis)
------------------------------------------------------------------
- loss_alpha: plain MSE on this ~78%-zero deltaZ target is minimized by
  predicting ~0 everywhere; a 150-epoch plain-MSE run beat an all-zeros
  predictor by only 0.35% RMSE with output std at 12% of truth. loss_alpha>0
  uses train.make_weighted_mse so active cells dominate the gradient.
  NOTE: history loss values are then weighted (not comparable to alpha=0
  runs); reported RMSE/MAE stay physical.
- scheduler="cosine": this loop takes ONE optimizer step per epoch (single
  accumulated full-BPTT backward), so 150 epochs = 150 Adam steps. A higher
  initial lr with cosine decay (to lr/20) compensates. Cosine (not plateau)
  because validation only runs every val_every epochs.
- train.collapse_diagnostics in the results/printout makes zero-collapse
  visible (skill_vs_zero <= 0 or std_ratio << 1 means the model predicts
  nothing).
"""

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from train import (
    mae, rmse, bias, nse,
    make_weighted_mse, collapse_diagnostics, enable_laplacian_cache,
)


def train_gconvgru_node_split(
    model, dataset, train_mask, val_mask, test_mask,
    lr=1e-3, weight_decay=1e-4, epochs=50, patience=10, val_every=1,
    loss_alpha=0.0, scheduler=None, active_tau=0.005,
    checkpoint_path="best_model_gconvgru.pt", device=None, verbose=True,
):
    """
    Train a GConvGRUModel transductively (node split) with checkpointed
    full-sequence BPTT. See the module docstring for the memory/speed/loss
    rationale.

    Parameters
    ----------
    model            : GConvGRUModel (forward(x, edge_index, h1, h2) ->
                       (pred, h1, h2))
    dataset          : Delft3DDataset
    train_mask, val_mask, test_mask : bool Tensor [N] node masks
                       (e.g. from train.node_random_split)
    lr               : Adam learning rate. Only ONE optimizer step happens per
                       epoch, so consider a larger lr (e.g. 5e-3) with
                       scheduler="cosine".
    weight_decay     : L2 strength. On a near-zero target this nudges
                       predictions toward zero; use 0.0 if collapse is a risk.
    epochs, patience : max epochs / early-stopping patience, where patience
                       counts VALIDATION EVENTS (epochs // val_every), not epochs
    val_every        : run validation every k-th epoch (1 = every epoch)
    loss_alpha       : weighted-MSE strength (0 = plain MSE); see
                       train.make_weighted_mse
    scheduler        : None (fixed lr) or "cosine"
                       (CosineAnnealingLR to lr/20 over `epochs`)
    active_tau       : |y| threshold (m) for the active-cell diagnostics
    checkpoint_path  : where the best (by val loss) weights are saved
    device           : torch.device (auto-detects CUDA when None)
    verbose          : per-epoch prints

    Returns
    -------
    history : dict of per-epoch lists (val entries are NaN on skipped epochs,
              keeping arrays epoch-aligned for plotting)
    results : dict matching train.py's node-split results, plus the
              collapse diagnostics (rmse_zero, skill_vs_zero, std_ratio,
              active_rmse, active_frac)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = enable_laplacian_cache(model.to(device))

    # Loss scale from the TRAINING nodes only (avoids test leakage).
    sigma = float(dataset.Y[:, train_mask.to(dataset.Y.device)].std())
    criterion = make_weighted_mse(sigma, loss_alpha)

    edge_index = dataset.edge_index.to(device)
    train_mask = train_mask.to(device)
    val_mask   = val_mask.to(device)
    test_mask  = test_mask.to(device)

    T = len(dataset)
    # Preload all timesteps to the GPU once (X~181 MB + Y~30 MB, fits easily
    # alongside the ~6.3 GB training peak) instead of re-copying every epoch.
    Xg = [dataset[t]["x"].to(device) for t in range(T)]
    Yg = [dataset[t]["y"].to(device) for t in range(T)]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,
                                 fused=(device.type == "cuda"))

    assert scheduler in (None, "cosine"), "scheduler must be None or 'cosine'"
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 20)
        if scheduler == "cosine" else None
    )

    history = {"train_loss": [], "val_loss": [], "train_rmse": [], "val_rmse": []}
    best_val, patience_counter = float("inf"), 0

    def rollout(mask):
        """Unroll all T timesteps in order, masked loss/metrics at each step."""
        h1 = h2 = None
        total_loss = 0.0
        y_true_list, y_pred_list = [], []
        for t in range(T):
            x, y = Xg[t], Yg[t]
            if torch.is_grad_enabled():
                # Training pass: recompute this step's activations in backward
                # instead of storing them (full-sequence BPTT at ~1/T memory).
                pred, h1, h2 = checkpoint(model, x, edge_index, h1, h2, use_reentrant=False)
            else:
                # Val/test pass runs under no_grad -> nothing to checkpoint.
                pred, h1, h2 = model(x, edge_index, h1, h2)
            total_loss = total_loss + criterion(pred[mask], y[mask])
            y_true_list.append(y[mask].detach().cpu().numpy())
            y_pred_list.append(pred[mask].detach().cpu().numpy())
        return total_loss / T, np.concatenate(y_true_list), np.concatenate(y_pred_list)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss, y_tr, y_tr_pred = rollout(train_mask)
        train_loss.backward()
        optimizer.step()
        if sched is not None:
            sched.step()

        tr_rmse = rmse(y_tr, y_tr_pred)
        history["train_loss"].append(train_loss.item())
        history["train_rmse"].append(tr_rmse)

        validate = (epoch % val_every == 0) or (epoch == epochs)
        if validate:
            model.eval()
            with torch.inference_mode():
                val_loss, y_val, y_val_pred = rollout(val_mask)
            val_rmse = rmse(y_val, y_val_pred)
            history["val_loss"].append(val_loss.item())
            history["val_rmse"].append(val_rmse)
        else:
            # keep history arrays epoch-aligned for plotting
            history["val_loss"].append(float("nan"))
            history["val_rmse"].append(float("nan"))

        if verbose:
            msg = (f"Epoch {epoch:03d} | "
                   f"train_loss={train_loss.item():.6f}  train_rmse={tr_rmse:.6f}")
            if validate:
                msg += f" | val_loss={val_loss.item():.6f}  val_rmse={val_rmse:.6f}"
            if sched is not None:
                msg += f" | lr={optimizer.param_groups[0]['lr']:.2e}"
            print(msg)

        if validate:
            if val_loss.item() < best_val:
                best_val, patience_counter = val_loss.item(), 0
                torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
            else:
                patience_counter += 1   # counted in validation events, not epochs
                if patience_counter >= patience:
                    if verbose:
                        print("Early stopping triggered.")
                    break

    chkpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(chkpt["model_state_dict"])

    model.eval()
    with torch.inference_mode():
        test_loss, y_test, y_test_pred = rollout(test_mask)

    results = {
        "test_loss": test_loss.item(),
        "mae":  mae(y_test,  y_test_pred),
        "rmse": rmse(y_test, y_test_pred),
        "bias": bias(y_test, y_test_pred),
        "nse":  nse(y_test,  y_test_pred),
        "y_true": y_test,
        "y_pred": y_test_pred,
        "test_idx": list(range(T)),
        "split_mode": "node",
        "train_mask": train_mask.cpu(),
        "val_mask": val_mask.cpu(),
        "test_mask": test_mask.cpu(),
    }
    results.update(collapse_diagnostics(y_test, y_test_pred, active_tau))
    if verbose:
        print("\nTest results:")
        for k in ("test_loss", "mae", "rmse", "bias", "nse",
                  "rmse_zero", "skill_vs_zero", "std_ratio", "active_rmse", "active_frac"):
            print(f"  {k:<13s}: {results[k]:.6f}")

    return history, results
