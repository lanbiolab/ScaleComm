# SpatialSelf Claim Boundary

This document fixes the manuscript claim boundary for Phase 5. Every Results section must map its main statement to a source row in `outputs/phase4/manuscript_evidence_lock/phase4_evidence_index.csv` or the Phase 5 evidence index.

## Core Claim

SpatialSelf supports a calibrated weak-signal and false-positive tradeoff in realistic CCC benchmark settings. The current evidence combines mechanism-aware controlled simulation, real-background semi-synthetic benchmarks, retained baseline comparisons, component ablations, permutation/FDR calibration, and key lymph-node downstream validation.

## Claims We Can Make

1. **Realistic benchmark claim**
   SpatialSelf can be evaluated on controlled and real-background semi-synthetic benchmarks that include contact-dependent, short-to-medium paracrine, long-range niche-restricted, weak-signal, and high-expression false-positive LR mechanisms.

   Evidence: `goal1_formalize_benchmarks` rows in the Phase 4 evidence index.

2. **Tradeoff claim**
   SpatialSelf supports a weak-signal / false-positive tradeoff in the current realistic benchmark panel.

   Evidence: `goal4_weak_signal_evidence` and `goal2_strengthen_baselines`.

   Required wording: report weak recall together with empirical FDR, top-k precision, and high-expression false-positive rate.

3. **Component claim**
   Graph wavelet modeling, LR identity, distance, expression, and pseudo-label design contribute to the observed performance.

   Evidence: `goal6_ablation_lock`.

4. **FDR claim**
   SpatialSelf predictions can be evaluated with permutation-based empirical p-values and BH-adjusted FDR under spatial, receiver, and LR-identity null models.

   Evidence: `goal3_upgrade_fdr`.

   Required wording: current 500-permutation evidence is strongest at LR-group and cell-type-pair-group level in the key lymph-node panel.

5. **Biological validation claim**
   In the key lymph-node validation panel, positive LR controls show higher mean receiver target correlation than negative controls.

   Evidence: `goal5_real_validation` downstream validation row.

6. **Real negative-control claim**
   In current semi-synthetic real-background validation, SpatialSelf ranks injected positive controls above high-expression negative controls across lymph node, tonsil, and melanoma backgrounds.

   Evidence: Phase 5 real-control index built from `goal1_full_baseline_matched_validation_all`.

## Claims We Cannot Make

1. SpatialSelf is the best CCC method across all metrics, datasets, tissues, and baselines.
2. SpatialSelf has the highest raw weak-signal recall in every setting.
3. SpatialSelf always has the lowest false-positive rate; SpaTalk is a conservative low-FP baseline in some panels.
4. Current 500-permutation FDR proves many edge-level significant discoveries.
5. Unlabeled real ST data provide complete ground truth.
6. Current downstream validation proves global cross-tissue biological superiority.
7. Current runtime evidence proves full 50k or 100k high-resolution scalability.

## Results Section Guardrails

Each Results section must include:

- one primary claim;
- one source table or evidence-index row;
- one limitation sentence;
- no unsupported global wording.

Suggested Results sections:

1. SpatialSelf models CCC with spatial topology and LR identity.
2. Realistic simulations benchmark weak and mechanism-specific CCC recovery.
3. SpatialSelf improves the weak-signal / false-positive tradeoff against retained baselines.
4. Ablation studies show LR identity and graph-wavelet components drive performance.
5. Permutation testing provides FDR-calibrated communication discoveries.
6. SpatialSelf recovers region-specific communication in lymph node.
7. High-resolution tonsil and melanoma validation support broader biological utility, with current limitations stated explicitly.

## Figure Guardrails

Main figures should defend claims, not merely display successful examples. Every figure must have:

- a source table;
- a supported claim;
- a limitation;
- an explicit reviewer concern it addresses.

Runtime/scalability should remain supplementary unless a formal 50k or 100k run is generated.
