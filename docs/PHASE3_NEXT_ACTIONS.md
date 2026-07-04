# Phase 3 Next Actions

The Phase 3 dataset entry point is now:

- `configs/experiments/phase3_end_to_end.yaml`
- `scripts/phase3/build_phase3_dataset_manifest.py`
- `outputs/phase3/end_to_end/phase3_dataset_manifest.csv`

The first audit gate is:

- `scripts/phase3/audit_phase3.py`
- `outputs/phase3/end_to_end/audit/phase3_audit_summary.csv`

## Next Execution Block

### 1. Prediction Index

Create a Phase 3 prediction index that reads the dataset manifest and records,
for each dataset and method:

- dataset id;
- method;
- prediction path;
- status;
- method scope;
- runtime path if available;
- baseline adapter notes.

This should reuse existing Phase 2 predictions when possible before launching
new runs.

### 2. Unified Metric Index

Create one metric table that joins the prediction index to:

- ground-truth metrics for controlled and real-background injected datasets;
- control-rank metrics for high-resolution/real validation contexts;
- runtime and memory fields when available.

The table should keep unsupported fields blank rather than forcing every dataset
into the same truth assumptions.

### 3. FDR Attachment Index

Create an FDR index with two tiers:

- smoke: 20 permutations;
- representative: 100 permutations.

The fixed nulls are:

- spatial permutation;
- receiver permutation;
- LR identity permutation.

### 4. Biological Validation Index

Create a validation index covering:

- lymph node;
- tonsil;
- melanoma;
- high-resolution tonsil subset.

Each row should record positive controls, negative controls, region enrichment,
downstream Spearman availability, and representative spatial-map source table
availability.

### 5. First-Pass Manuscript Figure Index

Create a figure-source index for Fig1-Fig8. At this stage, a figure can be
marked as source-ready, placeholder, or missing. The goal is pipeline coverage,
not final visual polish.

## Current Guardrail

Phase 3 is complete only when the end-to-end source table and figure pipeline is
auditable. The current state only completes the dataset-manifest gate.
