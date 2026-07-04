# Main Figure Plan

Phase 5 converts Phase 4 evidence-lock figures into manuscript figures. The target is not to add decorative panels, but to make each figure answer one reviewer-facing question.

## Main Text Figures

### Fig. 1: Method Overview

Primary claim: SpatialSelf combines spatial topology, graph-wavelet propagation, LR identity, expression, distance, self-supervised scoring, and permutation/FDR calibration.

Source evidence: model implementation, Phase 4 evidence index, ablation source tables.

Reviewer concern addressed: the method is not just expression times distance or a generic graph model.

Limitation: schematic figure does not prove performance; it introduces the design logic.

### Fig. 2: Controlled Benchmark

Primary claim: the controlled simulator tests multiple CCC mechanisms rather than only high-expression nearby pairs.

Source evidence: `goal1_controlled_simulator_v2_5rep` summaries.

Reviewer concern addressed: original simulation was too aligned with model assumptions.

Limitation: truth is still injected and synthetic.

### Fig. 3: Real-Background Benchmark

Primary claim: real ST coordinate/expression backgrounds make the semi-synthetic benchmark more realistic than fully artificial simulation.

Source evidence: `goal1_real_background_v2_5rep` summaries.

Reviewer concern addressed: simulation should use real spatial structure and expression noise.

Limitation: injected truth is not native real-tissue ground truth.

### Fig. 4: Baseline Comparison

Primary claim: SpatialSelf supports a metric-specific weak-signal / false-positive tradeoff against retained baselines.

Source evidence: replicated tuned baseline summary and baseline claim guardrails.

Reviewer concern addressed: missing spatial CCC and deep-learning baseline comparison.

Limitation: no global dominance claim; CellNEST remains tuned-small and DeepTalk/NICHES are status methods.

### Fig. 5: Ablation and Weak-Signal Mechanism

Primary claim: LR identity and graph-wavelet modeling are important contributors, and weak-signal claims must be interpreted with FDR/FP guardrails.

Source evidence: Phase 4 formal ablation summary and weak-signal claim table.

Reviewer concern addressed: whether improvement comes from model capacity or pseudo-label bias.

Limitation: current formal ablation is a 10-epoch coverage panel.

### Fig. 6: Permutation/FDR Calibration

Primary claim: SpatialSelf outputs can be calibrated under spatial, receiver, and LR-identity null models.

Source evidence: 500-permutation key lymph-node FDR panel.

Reviewer concern addressed: lack of p-values and FDR control.

Limitation: current strongest evidence is group-level; edge-level discovery power is limited.

### Fig. 7: Lymph Node Validation

Primary claim: lymph-node positive controls and key downstream target correlations support biological plausibility.

Source evidence: Phase 4 downstream key lymph-node panel and Phase 5 real-background control rankings.

Core panels:

- positive versus negative control ranks for lymph node;
- receiver target/downstream correlation for the key lymph-node panel;
- high-expression false-positive rate at q95;
- representative LR controls such as CCL19-CCR7 and CXCL13-CXCR5.

Key numbers from current source table:

- mean positive rank: 2.0;
- mean negative rank: 4.5;
- positive controls in top three: 3;
- negative controls in top three: 0;
- high-expression false-positive rate at q95: 0.000266.

Reviewer concern addressed: real data results should not be limited to selected examples.

Limitation: current downstream result is key lymph-node, not global cross-tissue validation.

### Fig. 8: Tonsil or Melanoma Validation

Primary claim: high-resolution tonsil or melanoma TME validation supports broader biological utility through positive/negative control ranking and niche/boundary biology.

Source evidence: Phase 5 real-validation tissue summary, LR panel, claims table, Phase 3 biological validation index, and Phase 2 revised matched-control summaries.

Core panels:

- tonsil positive/negative control rank distribution;
- melanoma positive/negative control rank distribution;
- high-expression false-positive rate at q95 across lymph node, tonsil, and melanoma;
- LR panel table or heatmap for tonsil GC/T-zone controls and melanoma tumor-immune controls.

Key numbers from current source table:

- tonsil: mean positive rank 2.0, mean negative rank 4.5, positive top-three count 3, negative top-three count 0, high-expression false-positive rate at q95 0.00261;
- melanoma: mean positive rank 2.0, mean negative rank 4.5, positive top-three count 3, negative top-three count 0, high-expression false-positive rate at q95 0.000716;
- receiver-response proxy delta is positive in lymph node, tonsil, and melanoma.

Reviewer concern addressed: high-resolution platform and heterogeneous tissue validation.

Limitation: current evidence is prior-control or semi-synthetic real-background validation; native real-data ground truth is absent.

## Supplementary Figures

- Extended runtime and scalability.
- Extended baseline status and failed method details.
- Extended ablation variants.
- LR embedding UMAP or pathway-category enrichment.
- Hyperparameter sensitivity.
- Extra real-background replicates and tissue-specific source tables.
