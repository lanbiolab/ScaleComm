# Phase 2 Next Actions

This checklist translates the revised six-goal Phase 2 plan into the next
execution block. It is intentionally narrower than the full roadmap: each item
should produce auditable files that can feed the manuscript source tables.

## Priority 1: Freeze Benchmark Schemas

Goal mapping: Goal 3 and Goal 4.

Deliverables:

- controlled simulator v2 schema with required truth columns;
- real-background semi-synthetic schema mapped to the same truth columns;
- validator output for controlled and real-background datasets;
- realism summary for library size, sparsity, local density, nearest-neighbor
  distances, LR expression, and zero inflation.

Current contract artifact:

- `scripts/phase2/validate_goal34_truth_contract.py`
- `outputs/phase2_revised/goal34_truth_contract/goal34_truth_contract_replicate_summary.csv`
- `outputs/phase2_revised/goal34_truth_contract/goal34_truth_contract_dataset_summary.csv`
- `outputs/phase2_revised/goal34_truth_contract/goal34_truth_contract_claims.csv`

Gate:

- No new large baseline panel should start until candidate-level, LR-level,
  cell-type-pair-level, and niche-level truth tables pass validation.

## Priority 2: Formal FDR Representative Panels

Goal mapping: Goal 5.

Deliverables:

- one representative lymph node FDR panel;
- one representative tonsil FDR panel;
- one representative melanoma FDR panel;
- each panel uses `spatial_permutation`, `receiver_permutation`, and
  `lr_identity_permutation`;
- each panel starts at 100 permutations before any 200-500 permutation run.

Current representative artifact:

- `configs/experiments/phase2_revised_goal5_real_background_fdr_100perm.yaml`
- `outputs/phase2_revised/goal5_real_background_fdr_100perm/goal5_real_background_fdr_panel_summary.csv`
- `outputs/phase2_revised/goal5_real_background_fdr_100perm/goal5_real_background_fdr_panel_group_summary.csv`
- `outputs/phase2_revised/goal5_real_background_fdr_100perm/goal5_real_background_fdr_panel_claims.csv`

Gate:

- A 5-permutation run is only a CLI/output-contract smoke test.
- Manuscript calibration wording requires representative fixed-null panels with
  enough permutations to inspect positive controls, negative controls, and null
  p-value behavior.
- The current 100-permutation panel supports representative group-level FDR
  evidence, but candidate-edge discoveries remain absent and final manuscript
  calibration still needs selected 200-500 permutation panels.

## Priority 3: Downstream Validation Upgrade

Goal mapping: Goal 1.

Deliverables:

- ligand-target coverage table for every retained method and tissue;
- baseline-matched downstream Spearman table;
- positive-control and negative-control rank table;
- high-expression false-positive table;
- status table for DeepTalk and NICHES if metric outputs remain unavailable.

Gate:

- Do not claim global biological superiority from selected LR examples.
- A main-text claim needs matched lymph node, tonsil, and melanoma summaries, or
  the limitation must be stated explicitly.

## Priority 4: Weak-Signal Tradeoff Panel

Goal mapping: Goal 2.

Deliverables:

- mechanism-stratified weak-signal metrics for contact, short/medium paracrine,
  and longer-range niche-restricted signals;
- matched metrics for SpatialSelf, COMMOT, CellChat, SpaTalk, CellNEST, and
  available NICHES/DeepTalk outputs or statuses;
- combined source table with weak recall, AUPR, empirical FDR,
  high-expression false-positive rate, and top-k precision.

Gate:

- The intended claim is a tradeoff claim, not "highest weak recall".
- If another method has higher weak recall but also high false-positive rate,
  report that explicitly instead of hiding it.

## Priority 5: Figure and Source-Table Convergence

Goal mapping: Goal 6.

First figure packages:

- simulator v2 mechanism design;
- 18-replicate baseline comparison;
- weak-signal versus high-expression false-positive tradeoff;
- formal FDR calibration and fixed nulls;
- baseline-matched biological validation across lymph node, tonsil, and
  melanoma.

Gate:

- Figures should read source tables from `outputs/phase2_revised`.
- Claim tables must classify each claim as supported, partial, or unsupported.
- Failed or status-only baselines need explicit reason and scope.

## Current Claim Guardrails

Supported or nearly supported:

- SpatialSelf wavelet edge prior is the aggregate frozen-endpoint winner on the
  current 18-replicate tuned panel.
- SpatialSelf shows strong suppression of high-expression false positives.
- The overall evidence supports a weak-signal/false-positive tradeoff framing.

Not supported yet:

- SpatialSelf globally dominates every baseline on every tissue and metric.
- SpatialSelf has the highest weak-signal recall in all benchmark settings.
- Current 5-permutation FDR smoke output is final calibration evidence.
- Selected real-data LR examples alone prove global biological validity.
