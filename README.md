# Delft3D to Graph-ML Preprocessing Demo

This repository is a lightweight prototype for coastal morphodynamics / hydrodynamics machine-learning workflows.

Its main purpose is to:

- extract Delft3D-FLOW outputs into graph-ready tensors;
- build structured-grid connectivity for use with graph neural network tooling such as [PyTorch Geometric](https://pytorch-geometric.readthedocs.io);
- provide a small notebook with synthetic data generation and visualization;
- verify that the PyTorch Geometric install is working.

It is not currently a full end-to-end ST-GNN training package.

## Project Files

- `d3d_grid_processing.py`
  Main preprocessing script. Reads Delft3D-FLOW outputs, detects relevant variables, constructs a valid-node mask and neighbor graph, and writes ML-ready arrays.

- `stgnn_core.ipynb`
  Notebook for synthetic spatiotemporal data generation, train/test tensor creation, and visualization/export of bed-elevation animations.

- `pytorch_geometric_sanity_check.py`
  Standalone install check for PyTorch and PyTorch Geometric using a tiny triangle graph and a simple `GCNConv`.

- `sanity-check.sh`
  Convenience wrapper for running the PyG sanity check without plotting.

- `environment.yml`
  Conda environment specification for Python, scientific IO, plotting, and a pinned PyTorch baseline.

- `test_bed_elevations.mp4`
  Example animation exported from the notebook.

## What the Preprocessing Script Produces

`d3d_grid_processing.py` extracts canonical Delft3D variables when available:

- `ZB`: bed level
- `S1`: water level
- `U1`: depth-averaged x-velocity
- `V1`: depth-averaged y-velocity
- `TAUB`: bed shear stress

Given an input dataset, it produces:

1. `features.npz`
   Contains:
   - `X` with shape `(T, N, F)`
   - `Y` with shape `(T, N)`

2. `edge_index.npy`
   Graph connectivity in COO format `(2, E)`.

3. `node_index.npy`
   Grid coordinates for each valid node `(N, 2)`.

4. `meta.json`
   Basic metadata including dimensions and selected features.

The graph is built from valid cells on a structured 2D grid using 4-neighbor connectivity by default.

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate d3d-stgnn
```

### 2. Install PyTorch Geometric

`environment.yml` pins a conservative dependency baseline for reproducibility. It installs PyTorch, but PyTorch Geometric is installed separately.

Example CPU-only installation:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install torch-geometric
```

Adjust wheel URLs as needed to match your installed PyTorch version and platform.

## Sanity Check

Run:

```bash
python pytorch_geometric_sanity_check.py
```

Or headless:

```bash
bash sanity-check.sh
```

Expected behavior:

- prints PyTorch and PyTorch Geometric versions;
- runs a small graph convolution on a 3-node graph;
- optionally plots the graph if `--no-plot` is not used.

## Running the Delft3D Preprocessing

Basic example:

```bash
python d3d_grid_processing.py --input /path/to/model/output.nc --outdir ./processed
```

Useful options:

- `--every N`
  Temporal downsampling interval.

- `--float32`
  Store arrays as `float32` instead of `float64`.

- `--convert`
  Attempt NEFIS-to-NetCDF conversion when the input is a Delft3D `.dat/.def` pair or a directory containing one.

The script accepts:

- a NetCDF file (`.nc`);
- a Zarr store (`.zarr`);
- a directory containing Delft3D outputs;
- a NEFIS `.dat` file when `--convert` is enabled.

For NEFIS conversion, configure an external converter command that accepts:

```bash
<input_trim.dat> <output.nc>
```

The script supports either:

```bash
export D3D_NEFIS_CONVERTER="python tools/convert_nefis.py"
```

or an installed executable named one of:

- `d3d-nefis-to-netcdf`
- `vs_trim2nc`
- `trim2nc`

## Running the Notebook

Install a Jupyter kernel if needed:

```bash
python -m ipykernel install --user --name=d3d-stgnn --display-name "Python (d3d-stgnn)"
```

Then launch Jupyter:

```bash
jupyter lab
```

Open `stgnn_core.ipynb`.

Current notebook contents:

- synthetic wind / water / shear / sediment-feature generation on a regular grid;
- synthetic bed-elevation target generation over time;
- train/test tensor preparation;
- animated visualization and export of bed-elevation fields.

## Current Scope and Limitations

- The repository includes graph-ready preprocessing, but does not currently contain a completed ST-GNN training pipeline.
- The notebook is a synthetic-data prototype, not a production training workflow on real Delft3D outputs.
- The notebook is intentionally checked in without cell outputs so the repository stays lightweight and reviewable.
- `d3d_grid_processing.py` assumes a structured 2D grid and uses heuristic variable-name matching.
- If velocity variables are staggered in the source model output, interpolation to cell centers may still be needed before model training.
- NEFIS conversion is optional and depends on external tooling being available and configured.

## Citation

If you use this workflow in research, consider citing:

- PyTorch: Paszke et al., 2019
- PyTorch Geometric: Fey and Lenssen, 2019
