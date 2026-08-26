# ScaleComm

ScaleComm is a scale-aware framework for spatial cell-cell communication inference. It models candidate sender-receiver-LR events on tissue graphs and uses an LR-specific multi-scale graph-diffusion representation to score putative communication events.

This repository provides the core ScaleComm implementation, interactive web UI, reviewer examples, and public smoke tests.

## Repository Layout

```text
src/spitialself/              Core ScaleComm package
  models.py                   Graph neural network and scale/context models
  model_factory.py            Model construction helper
  evaluation.py, metrics.py   Prediction metric helpers
  fdr.py                      Spatial-null and FDR helper code

apps/spatialself_viewer/      Streamlit web UI for inspecting ScaleComm results
examples/toy_scalecomm_demo.py  Data-free end-to-end smoke demo
examples/tonsil_cxcl13_cxcr5/  Paper-linked real-data reproduction
reproducibility/               Paper-linked workflow, manifests and figure resources
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
optimization run with the official multi-scale diffusion scale-mixture model, scores
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

## Paper-Linked Real-Data Reproduction

The compact versioned tonsil example below reproduces the receiver-response and
germinal-center enrichment summary for the CXCL13-CXCR5 axis in section A1.
It starts from a 2,500-receiver manuscript source-data table derived from the
public SpaMosaic human tonsil RNA+ADT resource. It validates an exported result
table; it does not retrain ScaleComm.

```bash
python examples/tonsil_cxcl13_cxcr5/reproduce_validation.py
```

Expected output:

```text
ScaleComm tonsil reproduction completed
receivers: 2500
top_receivers: 300
target_response_lift: 0.806701
mannwhitney_greater_p: 6.690272e-81
spearman_incoming_target: 0.621562
spearman_p_value: 3.403190e-267
gc_fold_enrichment: 2.339520
gc_fisher_greater_p: 4.035281e-69
output: tonsil_cxcl13_cxcr5_reproduction.csv
```

The script separately verifies the one-sided Mann-Whitney test for response
lift, the two-sided Spearman test for receiver-score association, and the
one-sided Fisher exact test for GC enrichment. It writes all reproduced
statistics to `tonsil_cxcl13_cxcr5_reproduction.csv`. Data
provenance and citation information are provided alongside the example.

### End-to-end Tonsil A1 Workflow

For an end-to-end paper-linked workflow from the public SpaMosaic A1 h5ad
object through input preparation, candidate construction, weak supervision,
de novo training, scoring and validation export, see:

```text
reproducibility/tonsil_a1/README.md
```

The detailed step-by-step tutorial is at
`docs/END_TO_END_TONSIL_A1_TUTORIAL.md`.

The workflow is launched as:

```bash
bash reproducibility/tonsil_a1/run_end_to_end.sh /path/to/tonsil_A1_rna_adt.h5ad cuda
```

It uses the fixed A1 study configuration in
`reproducibility/tonsil_a1/config.yaml`, including a 12-NN graph, preparation
seed 17, training seed 7 and 200 epochs. These settings describe the
paper-linked A1 workflow and are not universal defaults for every dataset.

## Reproducibility Resources

The `v0.1.3` tag fixes the reviewer-facing software version. The release
provides the paper-linked SpaMosaic tonsil A1 workflow, dataset-specific
configuration manifests, released source-data tables and figure-reconstruction
scripts for the reported analyses. Commands and expected outputs are listed in
[reproducibility/README.md](reproducibility/README.md). Run the guard for
the compact tonsil validation table with:

```bash
python reproducibility/verify_reported_statistics.py
```

## Using Your Data

For a complete worked example that converts a public h5ad object into these
inputs, use `reproducibility/tonsil_a1/README.md`. The tensor description below
is the lower-level API for users who already have prepared inputs.

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

The paper-linked static Explorer is self-contained: it embeds the exported
SpaMosaic tonsil A1 tables and does not depend on hard-coded local result-table
paths. Launch it from the repository root:

```bash
python3 -m http.server 8502 --bind 0.0.0.0 --directory apps/scalecomm_explorer
```

Then open:

```text
http://localhost:8502/scalecomm_explorer_tonsil_cxcl13_cxcr5.html
```

Its controls, data provenance, screenshots and interpretation boundary are
documented in `apps/scalecomm_explorer/README.md`.

The separate Streamlit viewer is a developer-facing browser for arbitrary
exported ScaleComm result tables. It displays LR selection, cell-type
filtering, communication edges, sender/receiver overlays and hover/click
inspection. Its input-table convention is documented in
`apps/spatialself_viewer/README.md`.

```bash
pip install -r apps/spatialself_viewer/requirements.txt
streamlit run apps/spatialself_viewer/app.py --server.address 0.0.0.0 --server.port 8501
```

The UI expects result/source tables exported from a ScaleComm run. Set `SCALECOMM_VIEWER_ROOT` to the project root containing those exported tables, or adapt the dataset paths in `apps/spatialself_viewer/app.py` for your own run directory.

## License

ScaleComm software is released under the MIT License. See `LICENSE` for terms.
The bundled tonsil example includes data provenance and citation information
in `examples/tonsil_cxcl13_cxcr5/DATA_SOURCE.md`.
