# SpaMosaic Tonsil A1 End-to-End Workflow

This is the paper-linked end-to-end workflow distributed with ScaleComm. It
starts with the public SpaMosaic human tonsil RNA+ADT A1 h5ad object and
recreates the processed analysis inputs, candidate events, expression-derived
weak supervision, a de novo ScaleComm fit and the CXCL13-CXCR5 receiver
validation export.

It is intentionally distinct from
`examples/tonsil_cxcl13_cxcr5/reproduce_validation.py`, which checks a compact
pre-exported receiver table only.

For the step-by-step tutorial, input schema, output descriptions, validation
checks and troubleshooting, read the
[end-to-end Tonsil A1 tutorial](../../docs/END_TO_END_TONSIL_A1_TUTORIAL.md).

## Prerequisites

Install the repository dependencies, then add the h5ad preparation dependency:

```bash
pip install -r requirements.txt
pip install -e .
pip install anndata h5py
```

Download the public SpaMosaic tonsil A1 RNA+ADT h5ad object from the data
record cited in `examples/tonsil_cxcl13_cxcr5/DATA_SOURCE.md`. This workflow
expects an h5ad containing `X`, `obsm['spatial']`, `obsm['protein_counts']` and
`uns['protein_features']`.

## Run

From the repository root:

```bash
bash reproducibility/tonsil_a1/run_end_to_end.sh /path/to/tonsil_A1_rna_adt.h5ad cuda --ui
```

The default training example runs for 1,000 epochs with seed 17. Append
`--paper` to select the 200-epoch configuration with training seed 7. Both
use preparation seed 17 and the same compact-to-rich input construction.
See the [seed comparison](../training_seed_examples.md) for the measured outcomes.

The optional `--ui` flag starts the generic Run Viewer when scoring is complete.
It reads `cells.csv`, `candidate_interactions.csv`, the prediction CSV and the
LR scale-profile CSV created by this run. No manuscript source tables are
required. The terminal prints the local URL; stop the viewer with `Ctrl+C`.
Install the optional viewer dependencies first with
`pip install -r apps/spatialself_viewer/requirements.txt`.

The command writes:

```text
reproducibility/tonsil_a1/work/prepared/spamosaic_tonsil_A1_rich_targets_subset/
  cells.csv
  expression.csv
  lr_pairs.csv
  candidate_interactions.csv
  target_signatures.csv

reproducibility/tonsil_a1/work/results/a1_rich_scalecomm_e1000/
  spamosaic_tonsil_A1_rich_targets_subset__spatialself_wavelet_scale_predictions.csv
  spamosaic_tonsil_A1_rich_targets_subset__spatialself_wavelet_scale_lr_scale_distribution.csv
  spamosaic_tonsil_A1_rich_targets_subset__spatialself_wavelet_scale_metrics.json
  tonsil_a1_cxcl13_cxcr5_receiver_validation.csv
  tonsil_a1_cxcl13_cxcr5_receiver_map.png
```

The run performs de novo optimization. CPU/GPU and library differences can
change individual scores slightly, so exact prediction-file hashes are not a
cross-platform success criterion. The supplied validation script instead
records the receiver count, top-receiver count, target-response contrast,
receiver-target Spearman correlation and germinal-center enrichment.

## Interpretation

The A1 cell-type and niche labels are deterministic ADT-marker proxies created
by `prepare_tonsil_a1.py`; they are not native CCC truth labels. The resulting
scores are used for within-dataset event prioritization. They are not calibrated
probabilities, direct measurements of molecular diffusion or standalone proof
of active communication.
