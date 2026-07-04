# Phase 3 End-To-End Manuscript Pipeline

Phase 3 is not a model-optimization phase. Its purpose is to make the whole
manuscript evidence chain executable end to end before further weak-signal,
downstream, or FDR tuning.

## Overall Objective

Build a reproducible pipeline that starts from benchmark datasets, runs the
frozen SpatialSelf representative and retained baselines, evaluates predictions,
attaches FDR and biological validation, and exports manuscript source tables and
first-pass figures.

The target is a complete evidence machine:

> dataset manifest -> predictions -> unified metrics -> FDR -> biological
> validation -> manuscript source tables -> first-pass figures

Results do not need to be final or optimal at this stage. The output should make
weak points visible so they can be optimized later.

## Goal 1: Freeze End-To-End Data Entry

All downstream scripts should read datasets from one manifest instead of
hard-coded historical paths.

Required dataset entries:

- controlled simulator v2;
- real-background semi-synthetic lymph node;
- real-background semi-synthetic tonsil;
- real-background semi-synthetic melanoma;
- high-resolution tonsil subset;
- optional Xenium or Visium HD placeholder entries.

Required manifest columns:

- `dataset_id`;
- `dataset_type`;
- `path`;
- `n_cells`;
- `n_lr_pairs`;
- `n_candidate_edges`;
- `has_ground_truth`;
- `has_injected_controls`;
- `available_labels`;
- `intended_figure_panel`.

## Goal 2: Run Main Model And Baseline Panel

The Phase 3 representative SpatialSelf method is frozen as:

- `spatialself_wavelet_edge_prior`.

Retained baselines:

- `commot`;
- `cellchat`;
- `spatalk`;
- `cellnest`;
- `deeptalk` as result or reproducible status;
- `niches` as result or reproducible status.

The first pass should cover the manifest datasets where the method is
applicable. Outputs must include unified prediction and metric indexes.

## Goal 3: Run Unified Evaluation Layer

For datasets with truth, each dataset/method row should expose:

- AUROC;
- AUPR;
- weak-signal recall;
- empirical FDR;
- high-expression false-positive rate;
- top-k precision.

For datasets without full truth, each dataset/method row should expose:

- positive-control rank;
- negative-control rank;
- high-expression negative top-k;
- downstream Spearman;
- region enrichment;
- FDR availability;
- runtime and memory.

## Goal 4: Run FDR / Permutation Layer

Phase 3 has two FDR tiers:

- smoke: 20 permutations;
- representative: 100 permutations.

The manuscript-facing nulls remain fixed:

- spatial permutation;
- receiver permutation;
- LR identity permutation.

Each FDR panel should produce candidate-edge, LR-pair, cell-type-pair,
significant-discovery, and null-summary outputs.

## Goal 5: Run Biological Validation

This phase should run the validation framework, not claim new biology.

Required validation contexts:

- lymph node;
- tonsil;
- melanoma;
- high-resolution tonsil subset.

Each context should output positive controls, negative controls, top LR ranking,
region enrichment, downstream Spearman, high-expression FP control, method
comparison, and representative spatial-map source tables when available.

## Goal 6: Assemble Manuscript Source Tables And First-Pass Figures

The manuscript scaffold should contain:

- Fig 1: method overview / benchmark schema;
- Fig 2: simulator v2 benchmark;
- Fig 3: real-background benchmark;
- Fig 4: baseline comparison;
- Fig 5: weak signal versus false-positive tradeoff;
- Fig 6: FDR calibration;
- Fig 7: biological validation;
- Fig 8: runtime / scalability.

Each figure must have a source table, plotting-script entry, output path,
QA/audit status, supported claim, and limitation.

## Priority Order

1. Unified dataset manifest.
2. Unified prediction / metric / control-evaluation schema.
3. Main model plus baseline panel.
4. FDR attachment.
5. Biological validation attachment.
6. Manuscript source tables and first-pass figures.
7. Later targeted optimization of weak signal, downstream targets, and
   permutation depth.

## Claim Guardrail

Phase 3 completion means the pipeline is executable and auditable. It does not
mean the manuscript claims are final, that SpatialSelf globally dominates all
baselines, or that 500-permutation FDR calibration is complete.
