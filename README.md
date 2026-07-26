# ScaleComm

ScaleComm is a scale-aware framework for spatial cell-cell communication inference. It models candidate sender-receiver-LR events on tissue graphs and uses LR-specific multi-scale spatial context to score putative communication events.

This repository provides the core ScaleComm implementation, interactive web UI, reviewer-facing toy demo, and public smoke tests.

## Repository Layout

```text
src/spitialself/              Core ScaleComm package
  models.py                   Graph neural network and scale/context models
  model_factory.py            Model construction helper
  evaluation.py, metrics.py   Prediction metric helpers
  fdr.py                      Spatial-null and FDR helper code

apps/spatialself_viewer/      Streamlit web UI for inspecting ScaleComm results
examples/toy_scalecomm_demo.py  Data-free end-to-end smoke demo
tests/test_core.py              Core forward/backward and metric smoke tests
```

The historical import path is `src.spitialself` for compatibility with earlier analyses. The method name used in manuscript text and user-facing material is ScaleComm.

## Installation

ScaleComm has been tested on Ubuntu 24.04 LTS (x86_64) with Python
3.12.7, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2, PyTorch
2.7.1 and PyTorch Geometric 2.7.0. Python 3.10 or newer is required.
The CPU demo does not require non-standard hardware. A CUDA-capable GPU is
recommended, but not required, for large spatial datasets.

Create a clean environment and install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Installation typically takes 3-10 minutes on a normal desktop with a broadband
connection. For GPU training, install the PyTorch and PyTorch Geometric builds
matching your CUDA/runtime environment.

## Minimal Demo

The bundled demo generates a small tissue graph in memory, performs a short
optimization run with the official graph-wavelet scale-mixture model, scores
candidate sender-receiver-LR events and exports the learned scale profiles. It
does not contain or reproduce the manuscript simulation benchmark.

```bash
python examples/toy_scalecomm_demo.py
```

Expected output:

```text
ScaleComm toy demo completed
node_scale_bank: (12, 6, 16)
event_scores: (24,)
lr_scale_profiles: (4, 6)
scale_profile_row_sums: [1.0, 1.0, 1.0, 1.0]
training_loss: 0.662942 -> 0.521395
outputs: toy_event_scores.csv, toy_lr_scale_profiles.csv
```

The CPU demo normally finishes in less than 10 seconds after dependencies are
installed. It writes `toy_event_scores.csv` and
`toy_lr_scale_profiles.csv` in the current directory. The generated labels are
only a software smoke test; they are not part of the manuscript benchmark and
are not evidence of biological performance. Small floating-point differences
can occur across platforms; a successful run has finite event scores, six
scale weights per LR pair that sum to 1, and a lower final training loss.

Run the public smoke tests with:

```bash
python -m unittest discover -s tests -v
```

## Using Your Data

Prepare the following tensors using the same node order throughout:

- `data.x`: an `n_nodes x n_features` floating-point expression/state matrix.
- `data.edge_index`: a `2 x n_graph_edges` spatial graph index.
- `src_idx`, `dst_idx`: sender and receiver node indices for candidate events.
- `lr_ids`: integer LR identifiers in `[0, n_lr)`.
- `dists`: an `n_events x 1` distance tensor.
- `edge_features`: optional event-level prior features when
  `edge_feature_dim > 0`.

Construct the model with
`build_model("spatialself_wavelet_scale", ...)`, call `encode_nodes` once for
the tissue graph, and call `score_edges` for candidate-event batches. The demo
contains a complete minimal example, including optimization with
`binary_cross_entropy_with_logits` and export of event scores and LR scale
profiles. Large datasets should score candidate events in batches.

## Web UI

The Streamlit viewer displays exported ScaleComm result tables on an interactive spatial canvas with LR selection, cell-type filtering, communication edges, sender/receiver overlays, and hover/click inspection.

```bash
pip install -r apps/spatialself_viewer/requirements.txt
streamlit run apps/spatialself_viewer/app.py --server.address 0.0.0.0 --server.port 8501
```

The UI expects result/source tables exported from a ScaleComm run. Set `SCALECOMM_VIEWER_ROOT` to the project root containing those exported tables, or adapt the dataset paths in `apps/spatialself_viewer/app.py` for your own run directory.
