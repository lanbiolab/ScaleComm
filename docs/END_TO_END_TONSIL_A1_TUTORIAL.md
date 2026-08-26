# End-to-End Tutorial: SpaMosaic Tonsil A1

This tutorial reproduces the paper-linked ScaleComm workflow for the
`CXCL13-CXCR5` analysis in SpaMosaic human tonsil section A1. It starts from a
public spatial multi-omic h5ad input and performs a new ScaleComm fit.

The workflow has six stages:

```text
public A1 h5ad
  -> deterministic analysis subset and proxy annotations
  -> standardized expression/cell/LR/candidate tables
  -> expression-derived weak supervision and graph construction
  -> de novo ScaleComm training and event scoring
  -> receiver validation and figure-ready export
```

The tutorial does not create communication ground truth. The resulting event
scores are within-dataset prioritization quantities, not calibrated
probabilities, physical diffusion distances or standalone evidence of causal
communication.

## 1. Create the tested environment

From the repository root:

```bash
conda env create -f reproducibility/environment.yml
conda activate scalecomm-reproduction
pip install -e .
```

The tested configuration is Python 3.12, NumPy 1.26.4, pandas 2.3.3, PyTorch
2.7.1 and PyTorch Geometric 2.7.0. NumPy 2.x is intentionally excluded because
some h5ad-related binary dependencies used by the preparation step are not
compatible with it in the tested environment.

For a CPU-only machine, install a PyTorch build appropriate for the local CPU
environment and pass `cpu` to the workflow command below. A GPU is recommended
for the 2,500-cell, 200-epoch paper-linked configuration.

## 2. Obtain and inspect the public input

Download the SpaMosaic human tonsil A1 RNA+ADT h5ad object from the data record
listed in:

```text
examples/tonsil_cxcl13_cxcr5/DATA_SOURCE.md
```

The preparation script requires these fields:

| h5ad field | Use in this workflow |
| --- | --- |
| `X` | RNA expression values used for model features and LR expression |
| `obs_names` | Stable cell identifiers |
| `var_names` | Gene symbols |
| `obsm['spatial']` | Two-dimensional cell coordinates |
| `obsm['protein_counts']` | ADT measurements used to create reproducible proxy annotations |
| `uns['protein_features']` | ADT feature names when `protein_counts` is array-like |

Before running, confirm that the file can be opened:

```bash
python - <<'PY'
import anndata as ad
adata = ad.read_h5ad('/path/to/tonsil_A1_rna_adt.h5ad', backed='r')
print(adata.shape)
print('spatial' in adata.obsm, 'protein_counts' in adata.obsm)
PY
```

## 3. Run the complete workflow

Run the fixed paper-linked configuration on a CUDA device:

```bash
bash reproducibility/tonsil_a1/run_end_to_end.sh \
  /path/to/tonsil_A1_rna_adt.h5ad cuda
```

For CPU execution:

```bash
bash reproducibility/tonsil_a1/run_end_to_end.sh \
  /path/to/tonsil_A1_rna_adt.h5ad cpu
```

The workflow is deterministic up to normal hardware/library floating-point
variation. Its exact parameters are fixed in
`reproducibility/tonsil_a1/config.yaml`:

| Component | Paper-linked A1 setting |
| --- | --- |
| Selected cells | 2,500 |
| Preparation seed | 17 |
| Cell-level graph | 12-nearest-neighbor graph |
| Diffusion depths | 1, 2, 4, 8 graph propagation steps |
| Training seed | 7 |
| Training epochs | 200 |
| LR panel | 7 documented tonsil axes |
| Weak-supervision strategy | `mechanism_niche_expression_distance` |
| Maximum training events | 12,000 |

These settings reproduce the paper-linked A1 analysis protocol. They are not a
claim that the same K, epoch count, candidate cap or LR panel is optimal for a
different spatial dataset.

## 4. What each stage writes

### Preparation

`prepare_tonsil_a1.py` writes:

```text
reproducibility/tonsil_a1/work/prepared/
  spamosaic_tonsil_A1_rich_targets_subset/
    cells.csv
    expression.csv
    lr_pairs.csv
    candidate_interactions.csv
    target_signatures.csv
    metadata.json
    standardization_summary.csv
```

`cells.csv` contains coordinates and deterministic ADT-marker proxy labels.
`expression.csv` has the selected RNA features. `lr_pairs.csv` gives the seven
LR axes and their documented context/range metadata. `candidate_interactions.csv`
contains candidate sender-receiver-LR events and the expression/distance/context
quantities used by the weak-supervision construction. `target_signatures.csv`
records the receiver-state panels used only for the downstream descriptive
validation step.

The cell-type and niche annotations are reproducible proxy annotations derived
from ADT markers. They are not histology labels and are not CCC ground truth.

### Training and inference

`run_scalecomm.py` reads the five standardized tables. It constructs the 12-NN
graph from `cells.csv`, builds node features from expression, proxy labels and
coordinates, derives the configured weak-supervision labels from the candidate
table, trains the multi-scale diffusion ScaleComm model and scores every
candidate event in batches.

It writes:

```text
reproducibility/tonsil_a1/work/results/a1_rich_scalecomm_e200/
  spamosaic_tonsil_A1_rich_targets_subset__spatialself_wavelet_scale_predictions.csv
  spamosaic_tonsil_A1_rich_targets_subset__spatialself_wavelet_scale_lr_scale_distribution.csv
  spamosaic_tonsil_A1_rich_targets_subset__spatialself_wavelet_scale_metrics.json
```

The historical file identifier `spatialself_wavelet_scale` is retained for
software compatibility. In the manuscript, the representation is described as
multi-scale graph diffusion rather than an exact spectral graph wavelet.

### Receiver validation

`validate_tonsil_a1.py` selects `CXCL13-CXCR5` candidate events, aggregates
incoming event scores by receiver, derives the prespecified receiver-state
summary, selects the top 300 receivers and exports:

```text
  tonsil_a1_cxcl13_cxcr5_receiver_validation.csv
  tonsil_a1_cxcl13_cxcr5_validation_metrics.json
  tonsil_a1_cxcl13_cxcr5_receiver_map.png
```

The metrics file reports the number of receivers, number of top receivers,
target-response contrast, Mann-Whitney P value, receiver-score/target-score
Spearman correlation, germinal-center proxy fold enrichment and Fisher P value.
It is a reproducible association/localization summary, not causal validation.

## 5. Check a completed run

The expected prepared A1 analysis has 2,500 cells, 7 LR pairs and 525,000
candidate events. Check these immediately after the preparation stage:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('reproducibility/tonsil_a1/work/prepared/spamosaic_tonsil_A1_rich_targets_subset/metadata.json')
meta = json.loads(p.read_text())
for key in ['n_cells', 'n_lr_pairs', 'n_candidate_interactions']:
    print(key, meta[key])
PY
```

Reference SHA256 values for the recorded prepared files and the recorded model
outputs are in `reproducibility/manifests/tonsil_a1.yaml`. Prepared-table hashes
should match when the public input release and the tested software environment
match. De novo model prediction hashes can differ slightly across hardware or
library builds; use the exported receiver validation statistics as the primary
cross-platform check.

The compact manuscript receiver table can be checked independently with:

```bash
python reproducibility/verify_reported_statistics.py
```

This checks the table distributed in
`examples/tonsil_cxcl13_cxcr5/`; it does not replace the fresh end-to-end run.

## 6. Inspect results

The output prediction CSV contains one score per candidate event. The
scale-distribution CSV contains the LR-specific normalized mixture over the six
diffusion-derived components. Use the self-contained paper-linked Explorer for
interactive inspection of the released A1 result tables:

```bash
python3 -m http.server 8502 --bind 0.0.0.0 --directory apps/scalecomm_explorer
```

Open:

```text
http://localhost:8502/scalecomm_explorer_tonsil_cxcl13_cxcr5.html
```

The Explorer is read-only and embeds the paper-linked release tables. It does
not rerun the model and does not modify scores.

## 7. Common problems

| Problem | Resolution |
| --- | --- |
| `anndata`/NumPy import error | Create the supplied conda environment; do not use NumPy 2.x. |
| Missing `protein_counts` or `spatial` | Confirm that the selected SpaMosaic RNA+ADT h5ad object, not an RNA-only derivative, was downloaded. |
| Missing LR gene | Verify gene symbols in `var_names`; the released LR panel requires the documented ligand and receptor genes. |
| CUDA unavailable | Use `cpu`; expect a longer 200-epoch training run. |
| Output filename differs | The runner names files from the prepared-directory name. The wrapper discovers the prediction file automatically before validation. |
| Result differs slightly from recorded file | Check the environment, source data version, seeds and config first. Floating-point/model-training variation can change individual scores across hardware. |

## 8. Additional reported analyses

Dataset-specific manifests, released source-data tables and figure
reconstruction scripts for CRC, DSS, NSCLC and the semi-synthetic benchmark are
organized under `reproducibility/`. See `reproducibility/README.md` for the
relevant entry point for each analysis.
