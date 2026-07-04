# Formal FDR Module

SpatialSelf's formal FDR entry point is:

```bash
conda run -n Benchmark python scripts/benchmark/run_formal_fdr_panel.py \
  --candidates path/to/candidate_interactions.csv \
  --predictions path/to/predictions.csv \
  --output-dir path/to/fdr_output \
  --n-permutations 100
```

## Fixed Null Models

The formal module uses exactly three nulls:

- `spatial_permutation`
- `receiver_permutation`
- `lr_identity_permutation`

Older exploratory scripts may expose additional nulls for diagnostics, but these
three are the manuscript-facing fixed nulls.

## Standard Outputs

For each null model, the module writes:

- `*__candidate_edge_fdr.csv`
- `*__lr_pair_direct_fdr.csv`
- `*__cell_type_pair_direct_fdr.csv`
- `*__significant_spatial_edges.csv`
- `*__significant_lr_pairs.csv`
- `*__significant_cell_type_pairs.csv`

The panel also writes:

- `goal5_formal_fdr_summary.csv`
- `goal5_formal_fdr_group_summary.csv`
- `goal5_formal_fdr_manifest.json`

## Required Discovery Columns

Candidate-edge FDR tables include:

- `candidate_id`
- `lr_id` when present
- `method`
- `null_model`
- `score`
- `empirical_p_value`
- `bh_fdr`
- `null_mean`
- `null_std`
- `null_q95`
- `n_permutations`

LR-pair and cell-type-pair group tables include group-level empirical p-values
and BH FDR. Significant discovery tables are filtered at the configured
`alpha`, default `0.05`.

## Interpretation Rules

- FDR outputs are post-processing calibration tables for completed prediction
  scores.
- Candidate-edge significance, LR-pair significance, and cell-type-pair
  significance are separate discovery levels and should not be mixed.
- Current Phase 2 evidence supports the FDR output schema and selected formal
  panels, not final 500-permutation manuscript calibration.

