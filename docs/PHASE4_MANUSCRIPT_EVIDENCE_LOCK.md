# Phase 4 Manuscript Evidence Lock

Phase 3 closed the end-to-end manuscript scaffold. Phase 4 is stricter: it upgrades scaffold outputs into submission-grade evidence and prevents unsupported wording from entering the manuscript.

## Core Claim

SpatialSelf should only be described as supporting a calibrated weak-signal and false-positive tradeoff when the relevant source tables pass Phase 4 evidence gates. The current evidence should not be phrased as global dominance across every metric, tissue, and baseline.

## Goals

1. Formalize the key benchmarks.
   - Controlled simulator v2 must have at least five replicates.
   - Real-background lymph node, tonsil, and melanoma must each have at least five replicates for final benchmark claims.
   - Contact, short/medium paracrine, long-range niche-restricted, weak signal, and high-expression negative-control mechanisms must be present.

2. Strengthen baseline comparison.
   - Keep COMMOT, CellChat, SpaTalk, CellNEST, DeepTalk status, and NICHES status.
   - Do not reintroduce LR-Coexp.
   - Use common LR database, common candidate universe, and explicit failure/status reasons.

3. Upgrade FDR.
   - Smoke FDR is not manuscript-final evidence.
   - A formal panel requires at least 200 permutations.
   - The final key panel target is 500 permutations.
   - The three required nulls are spatial permutation, receiver permutation, and LR identity permutation.

4. Lock weak-signal evidence.
   - Report weak recall beside empirical FDR, top-k precision, and high-expression false-positive rate.
   - Phrase the result as a tradeoff unless SpatialSelf is actually best on raw weak recall.

5. Make real-sample validation credible.
   - Use positive controls, negative controls, region enrichment, downstream Spearman, FDR-significant LR pairs, and baseline comparison.
   - Show global statistics before selected biological examples.

6. Lock ablations.
   - Required ablations: full model, no LR identity, no graph wavelet, no distance, no expression, MLP instead of graph model, GAT/wavelet comparison, and pseudo-label strategy variants.

7. Convert scaffold figures into manuscript figures.
   - Each figure must have a source table, supported claim, limitation, and export QA.
   - Fig8 runtime/scalability remains main-text only if the evidence is strong enough.

## Evidence Tiers

- `missing`: no usable artifact.
- `scaffold`: the Phase 3 pipeline has a table or figure, but the result is not enough for a manuscript claim.
- `representative`: enough to show the mechanism or workflow, but still below final claim strength.
- `formal`: meets the Phase 4 manuscript gate.
- `final`: reaches the target level for the strongest manuscript claim, such as 500-permutation FDR for the key panel.

## Wording Guardrail

Allowed:

> SpatialSelf improves or supports a weak-signal/false-positive tradeoff in selected realistic benchmark settings, with FDR and biological validation evidence provided by source tables.

Not allowed:

> SpatialSelf is the best CCC method across all datasets, all metrics, and all baselines.

