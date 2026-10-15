# ---------------------------------------------------------------------------
# torch-geometric-temporal is installed in a SEPARATE manual step after this
# environment is created - do NOT add it back to the pip list above.
#
# Reason: torch-geometric-temporal pulls in torch-scatter/torch-sparse, whose
# setup.py needs `import torch` at build time. pip builds wheels in an
# isolated sandbox that can't see torch even though it's listed right here,
# so the build fails with "ModuleNotFoundError: No module named 'torch'"
# no matter what order these lines are in. The fix is to install
# torch-scatter/torch-sparse from PyG's prebuilt wheel index instead of
# letting pip compile them from source, which requires torch to already be
# installed and requires knowing your exact torch+CUDA build tag.
#
# After `conda env update -n d3d-stgnn -f environment.yml --prune` finishes:
#
#   conda activate d3d-stgnn
#   python -c "import torch; print(torch.__version__)"
#     -> e.g. "2.2.2+cu121" ; the part after the "+" is your CUDA tag
#
#   pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+<CUDA_TAG>.html
#     -> e.g. https://data.pyg.org/whl/torch-2.2.0+cu121.html
#
#   pip install torch-geometric-temporal==0.56.*
# ---------------------------------------------------------------------------