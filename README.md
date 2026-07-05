# ScaleComm

ScaleComm is a scale-aware framework for spatial cell-cell communication inference. It models candidate sender-receiver-LR events on tissue graphs and uses LR-specific multi-scale spatial context to score putative communication events.

This public repository contains only the core ScaleComm code and the interactive web UI. Simulation benchmark contents, generated outputs, manuscript figures, raw data, and local analysis workspaces are intentionally not included.

## Repository Layout

```text
src/spitialself/              Core ScaleComm package
  models.py                   Graph neural network and scale/context models
  model_factory.py            Model construction helper
  evaluation.py, metrics.py   Prediction metric helpers
  fdr.py                      Spatial-null and FDR helper code

apps/spatialself_viewer/      Streamlit web UI for inspecting ScaleComm results
```

The historical import path is `src.spitialself` for compatibility with earlier analyses. The method name used in manuscript text and user-facing material is ScaleComm.

## Installation

Create a Python environment and install the core dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch and PyTorch Geometric builds matching your CUDA/runtime environment.

## Web UI

The Streamlit viewer displays exported ScaleComm result tables on an interactive spatial canvas with LR selection, cell-type filtering, communication edges, sender/receiver overlays, and hover/click inspection.

```bash
pip install -r apps/spatialself_viewer/requirements.txt
streamlit run apps/spatialself_viewer/app.py --server.address 0.0.0.0 --server.port 8501
```

The UI expects result/source tables exported from a ScaleComm run. Set `SCALECOMM_VIEWER_ROOT` to the project root containing those exported tables, or adapt the dataset paths in `apps/spatialself_viewer/app.py` for your own run directory.

## Not Included

This repository deliberately excludes:

- controlled or semi-synthetic simulation benchmark data/content
- raw spatial transcriptomics datasets
- generated `outputs/` folders
- manuscript PDFs, figure exports, and reference images
- local virtual environments
- large binary artifacts such as `.h5ad`, `.pt`, `.npy`, `.png`, `.pdf`, and `.csv` result tables

