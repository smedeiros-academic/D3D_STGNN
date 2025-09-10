# Spatio-Temporal Graph Neural Network (ST-GNN) Demo

This project demonstrates how to build and train a **Spatio-Temporal Graph Neural Network (ST-GNN)** using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io) and LSTMs.  
The use case is predicting **erosion/accretion (Δz_bed)** at mesh nodes based on time-series forcing (waves, currents, wind).

---

## 📂 Project Files

- `environment.yml`  
  Conda environment specification (pins `numpy<2` for compatibility).  

- `pytorch_geometric_sanity_check.py`  
  Standalone Python script to verify that PyTorch and PyTorch Geometric are installed correctly.  

- `stgnn_core.ipynb`  
  Jupyter Notebook containing the ST-GNN model definition and a dummy training loop.  
  (Sanity check cells have been moved to the standalone script.)

---

## 🔧 Setup Instructions

### 1. Create the Conda Environment
```bash
conda env create -f environment.yml
conda activate stgnn-env
```

### 2. Install PyTorch Geometric (CPU-only example)
Inside the environment, run these one at a time:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install torch-geometric
```

---

## ✅ Sanity Check

Run the standalone script to confirm everything works:

```bash
python pytorch_geometric_sanity_check.py
```

Expected output:
- Prints PyTorch and PyTorch Geometric versions.  
- Runs a GCN on a tiny 3-node triangle graph.  
- Displays a matplotlib visualization (unless you add `--no-plot`).  

---

## 📓 Running the Notebook

Launch Jupyter Lab and open the notebook:

```bash
jupyter lab
```

Then open **`stgnn_core.ipynb`**.  
It includes:
- ST-GNN model definition (GAT + GCN + LSTM + regression head).  
- Dummy time-series dataset on a toy graph.  
- Training loop with loss reporting.  

---

## ⚠️ Notes

- **NumPy**: The environment pins `numpy<2` because PyTorch and PyG wheels are not yet fully compatible with NumPy 2.x.  
- **GPU**: The environment is CPU-only by default. If you run on an NVIDIA GPU with CUDA 11.8, uncomment the `pytorch-cuda=11.8` line in `environment.yml`.  
- **Mesh data**: Replace the dummy data in the notebook with real forcing and bed-change data (e.g., ADCIRC/XBeach outputs).  

---

## 📝 Citation
If you use this workflow in research, you can cite:  
- PyTorch: *Paszke et al., 2019*  
- PyTorch Geometric: *Fey & Lenssen, 2019*  
