# Results Draft

This is a Phase 5 scientific draft. It intentionally uses conservative wording from `docs/CLAIM_BOUNDARY.md`.

## SpatialSelf Models CCC with Spatial Topology and LR Identity

SpatialSelf represents candidate ligand-receptor communication on a spatial cell or spot graph and combines expression-derived features, distance-aware edge information, LR identity, and graph-wavelet propagation. This design is intended to avoid treating all LR pairs as having the same spatial behavior.

Supported claim: SpatialSelf is not a pure expression co-occurrence score or a generic graph model without LR identity.

Source evidence: model design, Phase 4 ablation source tables.

Limitation: the method-design figure is explanatory; performance claims require benchmark and ablation evidence.

## Realistic Simulations Benchmark Weak and Mechanism-Specific CCC Recovery

The Phase 4 benchmark includes contact-dependent, short-to-medium paracrine, long-range niche-restricted, weak-signal, and high-expression false-positive mechanisms. The real-background benchmark further uses real ST-like coordinate and expression backgrounds across lymph node, tonsil, and melanoma settings.

Supported claim: the benchmark is more independent than a fully artificial "high expression plus nearby distance" simulation.

Source evidence: Phase 4 evidence index rows for controlled simulator and real-background benchmark formalization.

Limitation: these datasets contain injected truth; they are not native unlabeled real-tissue ground truth.

## SpatialSelf Improves the Weak-Signal / False-Positive Tradeoff Against Retained Baselines

Across the retained baseline panel, SpatialSelf supports a metric-specific tradeoff interpretation, particularly for weak-signal recovery under false-positive and empirical-FDR guardrails. The retained comparison includes COMMOT, CellChat, SpaTalk, and CellNEST; DeepTalk and NICHES remain status methods.

Supported claim: SpatialSelf supports a weak-signal / false-positive tradeoff in the current benchmark panel.

Source evidence: replicated tuned baseline summary and weak-signal claim table.

Limitation: SpatialSelf is not claimed to be globally best on every metric, tissue, or baseline. SpaTalk is a conservative low-FP baseline in some panels.

## Ablation Studies Show LR Identity and Graph-Wavelet Components Drive Performance

The formal Phase 4 ablation panel compares the full graph-wavelet model with no LR identity, no spatial graph, no distance, no expression, GAT, MLP, and pseudo-label variants. The full model shows higher controlled-benchmark AUPR than the no-LR-identity variant in the current five-replicate coverage panel.

Supported claim: LR identity and graph-wavelet modeling contribute to performance.

Source evidence: Phase 4 formal ablation summary.

Limitation: the current ablation is a 10-epoch formal coverage panel; final effect sizes may need a longer training-budget sweep.

## Permutation Testing Provides FDR-Calibrated Communication Discoveries

SpatialSelf predictions are evaluated with permutation-derived empirical p-values under spatial, receiver, and LR-identity null models. The Phase 4 key lymph-node panel includes 500 permutations and group-level LR and cell-type-pair summaries.

Supported claim: the FDR framework is wired and supports group-level calibration in the key panel.

Source evidence: Phase 4 500-permutation FDR summary and group summary.

Limitation: current edge-level discoveries are absent or limited at the selected threshold; manuscript wording should emphasize group-level FDR.

## SpatialSelf Recovers Region-Specific Communication in Lymph Node

The key lymph-node downstream panel tests positive and high-expression negative LR controls. Positive controls show higher mean receiver target correlation than negative controls in the current curated panel. In the Phase 5 real-background control summary, SpatialSelf ranks the lymph-node positive controls above high-expression negative controls, with mean positive rank 2.0 versus mean negative rank 4.5, three positive controls in the top three, and zero negative controls in the top three. The high-expression false-positive rate at the q95 threshold is 0.000266 in the lymph-node panel.

Supported claim: key lymph-node downstream validation supports biological plausibility.

Source evidence: Phase 4 key lymph-node downstream summary; Phase 5 real-validation tissue summary and LR panel.

Limitation: this does not establish global cross-tissue downstream superiority. The Phase 5 tissue summary is a real-background semi-synthetic control analysis, not native complete ground truth.

## High-Resolution Tonsil and Melanoma Validation Support Broader Biological Utility

High-resolution tonsil and melanoma/TME analyses provide additional evidence through positive/negative control ranking, niche-specific or boundary-relevant LR controls, and baseline-matched validation summaries. In the Phase 5 real-background summaries, SpatialSelf ranks all three positive controls in the top three and no high-expression negative controls in the top three for both tonsil and melanoma. The tonsil panel includes IL21-IL21R, CD40LG-CD40, and CXCL13-CXCR5 controls, with a high-expression false-positive rate of 0.00261 at q95. The melanoma panel includes CXCL9-CXCR3, SPP1-CD44, and PDCD1-CD274 controls, with a high-expression false-positive rate of 0.000716 at q95. Across lymph node, tonsil, and melanoma, the positive-minus-negative receiver-response proxy delta is positive in all three panels.

Supported claim: the workflow extends beyond one lymph-node example and supports positive/negative control separation in tonsil and melanoma real-background panels.

Source evidence: Phase 5 real-validation tissue summary, LR panel, and claims table; Phase 3 biological validation index.

Limitation: these analyses remain prior-control or semi-synthetic real-background validations; native real ST ground truth is unavailable. Tonsil and melanoma use receiver-response proxy rather than curated downstream target validation.
