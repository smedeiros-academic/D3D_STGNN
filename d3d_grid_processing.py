#!/usr/bin/env python3
"""
Delft3D-FLOW → ST-GNN Extraction Script
=======================================

Purpose
-------
Extract gridded, time-dependent hydrodynamic and morphodynamic variables
from a Delft3D-FLOW map file (NetCDF preferred; NEFIS supported via conversion)
and package them into machine-learning-ready tensors suitable for training
a Spatiotemporal Graph Neural Network (ST-GNN).

Primary Variables Extracted (if available):
    - ZB    : Bed level
    - S1    : Water level
    - U1    : Depth-averaged x-velocity
    - V1    : Depth-averaged y-velocity
    - TAUB  : Bed shear stress

Output Structure
----------------
The script produces:

1) X tensor: shape (T, N, F)
   T = number of timesteps
   N = number of valid grid cells (graph nodes)
   F = number of features (variables extracted)

2) Y tensor: shape (T, N)
   Contains bed level (ZB). For one-step prediction:
       X[t] → Y[t+1]

3) edge_index.npy:
   Graph connectivity in COO format (2, E) for PyTorch Geometric.

4) node_index.npy:
   Array (N,2) giving (m,n) grid indices for each node.

5) meta.json:
   Metadata describing shapes, variable mappings, etc.

Design Assumptions
------------------
- Structured 2D grid (m,n)
- Time dimension present
- Variables are either cell-centered OR user accepts them as-is
- Mask is derived from valid bed level cells

Important Modeling Notes
------------------------
- If U1/V1 are staggered, interpolation to cell centers may be required.
- If MORFAC was used, timestamps represent hydro-time, not morph-time.
- Masking removes permanently invalid cells.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr


# =============================================================================
# Variable Canonicalization
# =============================================================================

"""
Delft3D variable naming varies across:
- Versions
- Converters
- Case conventions
- NetCDF exporters

We define canonical names and map aliases to improve robustness.
"""

CANONICAL_VARS = ["ZB", "S1", "U1", "V1", "TAUB"]

VAR_ALIASES: Dict[str, List[str]] = {
    "ZB":   ["ZB", "zb", "bedlevel", "bed_level", "BedLevel"],
    "S1":   ["S1", "s1", "waterlevel", "WaterLevel", "eta"],
    "U1":   ["U1", "u1", "U", "ucx"],
    "V1":   ["V1", "v1", "V", "ucy"],
    "TAUB": ["TAUB", "taub", "tau_b", "TAU"],
}


# =============================================================================
# Utility Functions
# =============================================================================

def maybe_convert_nefis(trim_dat: Path, out_nc: Path) -> Path:
    """
    Convert a Delft3D NEFIS trim file to NetCDF using an external converter.

    The conversion step is intentionally explicit because converter availability
    differs across Delft3D installations. This helper supports two modes:

    1. Set D3D_NEFIS_CONVERTER to a command prefix that accepts:
           <input_trim.dat> <output.nc>
       Example:
           export D3D_NEFIS_CONVERTER="python tools/convert_nefis.py"

    2. Install a converter executable named one of:
           d3d-nefis-to-netcdf
           vs_trim2nc
           trim2nc

    Returns:
        Path to the converted NetCDF file.
    """
    trim_dat = trim_dat.expanduser().resolve()
    out_nc = out_nc.expanduser().resolve()
    def_file = trim_dat.with_suffix(".def")

    if not trim_dat.exists():
        raise SystemExit(f"NEFIS data file not found: {trim_dat}")
    if not def_file.exists():
        raise SystemExit(
            f"NEFIS definition file not found for {trim_dat.name}: expected {def_file.name}"
        )
    if out_nc.exists():
        return out_nc

    converter_cmd = os.environ.get("D3D_NEFIS_CONVERTER")
    if converter_cmd:
        cmd = shlex.split(converter_cmd) + [str(trim_dat), str(out_nc)]
    else:
        candidates = ["d3d-nefis-to-netcdf", "vs_trim2nc", "trim2nc"]
        executable = next((name for name in candidates if shutil.which(name)), None)
        if executable is None:
            raise SystemExit(
                "NEFIS conversion was requested, but no converter is configured. "
                "Set D3D_NEFIS_CONVERTER to a converter command or install one of: "
                f"{', '.join(candidates)}"
            )
        cmd = [executable, str(trim_dat), str(out_nc)]

    print(f"Converting NEFIS to NetCDF: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "(no stderr output)"
        raise SystemExit(
            "NEFIS conversion failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {stderr}"
        )
    if not out_nc.exists():
        raise SystemExit(
            "NEFIS converter reported success, but the expected NetCDF output was not created: "
            f"{out_nc}"
        )

    return out_nc

def resolve_input_to_netcdf(input_path: Path, outdir: Path, allow_convert: bool) -> Path:
    """
    Resolve --input into a readable dataset for xarray.

    Accepts:
      - a NetCDF file (*.nc)
      - a Zarr directory (*.zarr)
      - a directory containing either:
          * a NetCDF file (trim*.nc), OR
          * a NEFIS trim*.dat (with matching .def), which we can convert (if allow_convert=True)

    Returns:
      Path to NetCDF file or Zarr store.
    """
    input_path = input_path.expanduser().resolve()

    # 1) Direct NetCDF file
    if input_path.is_file() and input_path.suffix.lower() == ".nc":
        return input_path

    # 2) Direct Zarr store
    if input_path.is_dir() and input_path.suffix.lower() == ".zarr":
        return input_path

    # 3) Directory: search for candidates
    if input_path.is_dir():
        # Prefer NetCDF first
        nc_candidates = sorted(input_path.glob("**/trim-*.nc")) + sorted(input_path.glob("**/*trim*.nc"))
        if nc_candidates:
            return nc_candidates[0]

        # Otherwise look for NEFIS trim
        nefis_candidates = sorted(input_path.glob("**/trim-*.dat")) + sorted(input_path.glob("**/*trim*.dat"))
        if nefis_candidates:
            trim_dat = nefis_candidates[0]
            if not allow_convert:
                raise SystemExit(
                    f"Found NEFIS file {trim_dat} but conversion is disabled. "
                    f"Re-run with --convert or provide a NetCDF .nc."
                )
            out_nc = outdir / (trim_dat.stem + ".nc")
            return maybe_convert_nefis(trim_dat, out_nc)

    # 4) Direct NEFIS file
    if input_path.is_file() and input_path.suffix.lower() == ".dat":
        if not allow_convert:
            raise SystemExit("Input is NEFIS .dat. Re-run with --convert or provide NetCDF.")
        out_nc = outdir / (input_path.stem + ".nc")
        return maybe_convert_nefis(input_path, out_nc)

    raise SystemExit(f"Could not resolve input: {input_path}")


def find_var(ds: xr.Dataset, canonical: str) -> Optional[str]:
    """
    Identify the actual dataset variable name corresponding to a canonical name.

    Strategy:
        1. Exact match
        2. Case-insensitive match
        3. Substring heuristic

    Returns:
        Variable name in dataset if found, else None.
    """
    candidates = VAR_ALIASES.get(canonical, [canonical])

    for c in candidates:
        if c in ds.variables:
            return c

    for v in ds.variables:
        for c in candidates:
            if c.lower() == v.lower():
                return v

    for v in ds.variables:
        for c in candidates:
            if c.lower() in v.lower():
                return v

    return None


def detect_time_dim(da: xr.DataArray) -> Optional[str]:
    """
    Attempt to automatically detect time dimension.

    Looks for dimension names like:
        'time', 't', 'tim', 'nt'

    Returns:
        Name of time dimension or None.
    """
    for d in da.dims:
        if d.lower() in ("time", "t", "tim", "nt"):
            return d
    return None


def detect_mn_dims(da: xr.DataArray, time_dim: str) -> Tuple[str, str]:
    """
    Identify spatial dimensions (m,n).

    Removes time dimension and selects first two remaining dims.
    Assumes structured 2D grid.

    Returns:
        (m_dim, n_dim)
    """
    dims = [d for d in da.dims if d != time_dim]
    if len(dims) < 2:
        raise ValueError("Cannot detect spatial dimensions.")
    return dims[0], dims[1]


def build_valid_mask(zb_da: xr.DataArray, time_dim: str) -> np.ndarray:
    """
    Construct boolean mask of valid computational cells.

    Criteria:
        - Finite bed level at initial timestep
        - Not NaN for all timesteps

    Returns:
        2D boolean mask (m,n)
    """
    zb0 = zb_da.isel({time_dim: 0})
    mask0 = np.isfinite(zb0.values)
    any_valid = np.isfinite(zb_da).any(dim=time_dim).values
    return mask0 & any_valid


def build_edge_index(mask: np.ndarray, diagonals: bool = False):
    """
    Construct graph connectivity from structured grid.

    Each valid cell becomes a node.
    Edges connect neighboring cells.

    Parameters:
        mask : 2D boolean array
        diagonals : include diagonal connections

    Returns:
        edge_index : (2, E)
        node_index : (N,2)
    """
    mmax, nmax = mask.shape
    node_id = -np.ones_like(mask, dtype=int)
    coords = np.argwhere(mask)
    node_id[mask] = np.arange(len(coords))

    neighbors = [(1,0), (-1,0), (0,1), (0,-1)]
    if diagonals:
        neighbors += [(1,1), (1,-1), (-1,1), (-1,-1)]

    src, dst = [], []

    for i, j in coords:
        u = node_id[i,j]
        for di, dj in neighbors:
            ii, jj = i+di, j+dj
            if 0 <= ii < mmax and 0 <= jj < nmax and mask[ii,jj]:
                v = node_id[ii,jj]
                src.append(u)
                dst.append(v)

    edge_index = np.vstack([src, dst])
    return edge_index, coords


# =============================================================================
# Main Extraction Logic
# =============================================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--every", type=int, default=1,
                        help="Temporal downsampling interval")
    parser.add_argument("--float32", action="store_true",
                        help="Store tensors as float32 (recommended)")
    parser.add_argument("--convert", action="store_true",
                    help="If input is NEFIS (.dat/.def) or directory containing NEFIS, attempt conversion to NetCDF.")

    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Open Dataset
    # -------------------------------------------------------------------------

    print("Opening dataset...")
    resolved = resolve_input_to_netcdf(input_path, outdir, allow_convert=args.convert)

    if resolved.is_dir() and resolved.suffix.lower() == ".zarr":
        ds = xr.open_zarr(resolved)
    else:
        ds = xr.open_dataset(resolved)

    # -------------------------------------------------------------------------
    # Variable Mapping
    # -------------------------------------------------------------------------

    var_map = {}
    for v in CANONICAL_VARS:
        found = find_var(ds, v)
        if found:
            var_map[v] = found
        else:
            print(f"Warning: {v} not found.")

    if "ZB" not in var_map:
        raise RuntimeError("ZB required for masking and targets.")

    # -------------------------------------------------------------------------
    # Dimension Detection
    # -------------------------------------------------------------------------

    zb = ds[var_map["ZB"]]
    time_dim = detect_time_dim(zb)
    m_dim, n_dim = detect_mn_dims(zb, time_dim)

    print(f"Detected dimensions: time={time_dim}, m={m_dim}, n={n_dim}")

    # -------------------------------------------------------------------------
    # Downsample Time
    # -------------------------------------------------------------------------

    ds = ds.isel({time_dim: slice(None, None, args.every)})

    # -------------------------------------------------------------------------
    # Mask Construction
    # -------------------------------------------------------------------------

    zb = ds[var_map["ZB"]]
    mask = build_valid_mask(zb, time_dim)
    edge_index, node_index = build_edge_index(mask)

    # Save graph structure
    np.save(outdir / "edge_index.npy", edge_index)
    np.save(outdir / "node_index.npy", node_index)

    # -------------------------------------------------------------------------
    # Feature Tensor Assembly
    # -------------------------------------------------------------------------

    T = ds.dims[time_dim]
    N = node_index.shape[0]
    F = len(var_map)

    dtype = np.float32 if args.float32 else np.float64
    X = np.empty((T, N, F), dtype=dtype)

    print("Extracting features...")

    for f_idx, (canon, actual) in enumerate(var_map.items()):
        da = ds[actual]
        da = da.transpose(time_dim, m_dim, n_dim)
        arr = da.values
        X[:,:,f_idx] = arr[:, mask]

    Y = ds[var_map["ZB"]].transpose(time_dim, m_dim, n_dim).values[:, mask]

    # -------------------------------------------------------------------------
    # Save Outputs
    # -------------------------------------------------------------------------

    np.savez_compressed(outdir / "features.npz", X=X, Y=Y)

    meta = {
        "T": int(T),
        "N": int(N),
        "F": int(F),
        "features": list(var_map.keys()),
        "note": "For 1-step training use X[t] → Y[t+1]"
    }

    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Extraction complete.")
    print(f"Tensor shape: X={X.shape}, Y={Y.shape}")


if __name__ == "__main__":
    main()
