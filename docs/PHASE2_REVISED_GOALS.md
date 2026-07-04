# Revised Phase 2 Goals

This revised Phase 2 plan replaces the earlier model-tuning-oriented Phase 2
as the main manuscript evidence plan. The purpose is to turn the current
SpatialSelf signal into a reviewer-resistant evidence chain, not to claim that
SpatialSelf wins every AUROC/AUPR benchmark.

Canonical scope: Phase 2 has six goals. Runtime/scalability is part of Goal 6
manuscript evidence convergence, not a seventh standalone goal. Historical
output folders may keep their old names for reproducibility, but manuscripts,
configs, claim matrices, and audits should map all evidence to the six goals
below.

## Overall Claim

SpatialSelf should be developed and evaluated as a calibrated, LR-aware,
weak-signal-sensitive spatial CCC inference framework. The central claim is:

> SpatialSelf improves the tradeoff between weak-signal recovery and
> high-expression false-positive suppression, with FDR and downstream biological
> validation under realistic spatial transcriptomics backgrounds.

The acceptable claim is specific. We should not claim broad superiority across
all metrics or all baselines unless future evidence supports it.

## Goal 1: Baseline-Matched Biological Validation

Purpose: expand the strongest current lymph-node evidence to additional real
tissue contexts.

Datasets:

- Slide-tags/human tonsil large tile or multiple tiles.
- Melanoma TME.
- Optional Xenium or Visium HD subset if the local data and runtime allow it.

Methods should be compared on the same candidate LR controls:

- SpatialSelf default.
- COMMOT.
- CellChat.
- SpaTalk.
- CellNEST.
- DeepTalk when executable outputs are available.
- NICHES when the R dependency chain is reproducible.

Primary metrics:

- positive-control rank;
- negative-control rank;
- downstream Spearman correlation;
- empirical FDR;
- high-expression false-positive rate.

Target conclusion:

> SpatialSelf ranks biologically plausible controls ahead of high-expression
> negative controls in realistic tissue backgrounds.

## Goal 2: Weak-Signal Claim

Purpose: make the weak-signal result robust enough for the manuscript.

Benchmark requirements:

- ligand and receptor expression should both be low or moderate;
- sender/receiver spatial organization should remain realistic;
- high-expression false-positive LR controls must be present;
- mechanisms must be stratified into contact, short/medium paracrine, and
  longer-range niche-restricted chemokine-like signals.

Primary metrics:

- weak-signal recall;
- AUPR;
- empirical FDR;
- high-expression false-positive rate;
- top-k precision.

Target conclusion:

> SpatialSelf has a better weak-signal versus false-positive tradeoff, rather
> than necessarily the highest weak recall alone.

## Goal 3: Benchmark Simulator v2

Purpose: rebuild the controlled benchmark as a mechanism-layered simulator with
auditable truth tables.

The simulator must include:

- contact-dependent/juxtacrine LR;
- short-to-medium paracrine LR;
- niche-restricted longer-range chemokine-like LR;
- weak signals;
- strong signals;
- high-expression false-positive LR pairs.

Each LR pair must record:

- sender cell type;
- receiver cell type;
- spatial range;
- expression strength;
- background expression;
- weak-signal flag;
- false-positive high-expression-control flag.

Required truth outputs:

- candidate-level true edge table;
- LR-pair-level truth table;
- cell-type-pair-level truth table;
- spatial-niche-level truth table;
- weak/strong/mechanism labels.

Target conclusion:

> The benchmark is no longer reducible to "high expression plus short distance";
> it has mechanism-layered CCC truth.

## Goal 4: Real-Background Semi-Synthetic Benchmark

Purpose: test SpatialSelf under real tissue coordinates and real expression
noise while retaining known injected truth.

Real properties to preserve:

- coordinates;
- cell-type composition;
- expression sparsity;
- library size;
- local density;
- tissue domain structure.

Priority injections:

- lymph node: CCR7/CCL19-like T-zone signal;
- tonsil: FDC-B and Tfh-B signals;
- melanoma: macrophage-T and tumor-immune-boundary signals.

Target conclusion:

> SpatialSelf's advantage is not an artifact of overly artificial simulation;
> it persists under real spatial structure and expression background.

## Goal 5: Formal FDR and Calibration Module

Purpose: make significance testing part of the method rather than a side
analysis script.

Required nulls:

- spatial permutation;
- receiver permutation;
- LR identity permutation.

Required outputs:

- empirical p-value;
- BH FDR;
- significant LR pairs;
- significant cell-type-pair interactions;
- significant spatial edges or candidate groups.

Target conclusion:

> SpatialSelf can report statistically calibrated CCC discoveries, not only
> communication scores.

## Goal 6: Manuscript Evidence and Figure Convergence

Purpose: converge code, source tables, and figures around a compact main story.

Main evidence chain:

1. method overview;
2. simulator v2;
3. real-background semi-synthetic benchmark;
4. baseline comparison;
5. weak-signal and false-positive tradeoff;
6. ablation;
7. FDR/permutation;
8. lymph node, tonsil, and melanoma biological validation;
9. runtime/scalability.

Target conclusion:

> SpatialSelf is a calibrated LR-aware spatial CCC framework with weak-signal,
> false-positive-control, FDR, and downstream-validation evidence.

Runtime/scalability is included here because it supports the final manuscript
claim and figure package. It should not expand the phase into a separate Goal 7.

## Canonical Goal Mapping

| Canonical goal | Main meaning | Current artifact families |
| --- | --- | --- |
| Goal 1 | baseline-matched biological validation | `goal1_*baseline_matched*`, full baseline matched validation outputs |
| Goal 2 | weak-signal and false-positive tradeoff | `goal2_weak_signal_tradeoff`, full baseline panel outputs |
| Goal 3 | mechanism-layered simulator v2 | `goal3_simulator_v2` |
| Goal 4 | real-background semi-synthetic benchmark | `goal4_real_background_v2` |
| Goal 5 | formal FDR/calibration module | `goal5_formal_fdr`, `goal5_fdr_200perm` |
| Goal 6 | manuscript source tables, figures, limitations, runtime/scalability | `goal6_manuscript_assembly`, historical `goal7_runtime_scalability` support tables |

## Priority Order

1. Implement and validate benchmark simulator v2.
2. Implement real-background semi-synthetic benchmark v2.
3. Extend baseline-matched biological validation to tonsil and melanoma.
4. Promote FDR/permutation to a formal reusable module.
5. Run weak-signal-focused benchmark panels.
6. Converge manuscript source tables and figures.

## Current Evidence Constraint

Current comparative evidence supports a specific advantage, not broad dominance.
SpatialSelf is strongest in high-expression false-positive suppression and
downstream/control-ranking validation. COMMOT can still win some AUROC/AUPR or
FDR summaries. Manuscript wording must preserve that distinction.
