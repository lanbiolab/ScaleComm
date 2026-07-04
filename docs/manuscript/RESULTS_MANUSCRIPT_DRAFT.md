# Results

This Phase 5 Results draft follows the claim boundary in `docs/CLAIM_BOUNDARY.md`. Each section states the figure, the supported claim, the source evidence, and the limitation that must remain visible during manuscript polishing.

## SpatialSelf Models CCC with Spatial Topology and LR Identity

SpatialSelf was designed to score spatial ligand-receptor (LR) communication candidates while keeping three sources of information separate: expression evidence, spatial graph topology, and LR identity. The model constructs candidate sender-receiver edges on a spatial cell or spot graph, computes expression and distance-aware edge features, and propagates information through a graph-wavelet representation. LR identity is represented explicitly so that contact-dependent, short-to-medium paracrine, and longer-range niche-restricted LR pairs are not forced into the same spatial behavior.

This design addresses a key limitation of simple co-expression or distance-filtered scoring: a high ligand and receptor expression pair is not necessarily a plausible communication event if it occurs in the wrong cell-type context or spatial domain. Conversely, weak but spatially structured LR activity may be missed by methods that rely mainly on expression magnitude. The method overview in Fig. 1 should therefore introduce SpatialSelf as a topology- and LR-aware scoring framework, not as a claim of performance by itself.

Supported claim: SpatialSelf is not a pure expression co-occurrence score or a generic graph model without LR identity.

Figure: Fig. 1.

Source evidence: model implementation, Phase 4 ablation source tables, and the Phase 4 evidence index.

Limitation: the method schematic is explanatory. Benchmark, ablation, FDR, and validation results are required before making performance claims.

## Realistic Simulations Benchmark Weak and Mechanism-Specific CCC Recovery

We next formalized the benchmark so that simulated truth is not defined only by high expression and short distance. The Phase 4 controlled simulator includes contact-dependent, short-to-medium paracrine, longer-range niche-restricted chemokine-like, weak-signal, and high-expression false-positive mechanisms. The real-background benchmark uses lymph node, tonsil, and melanoma-like spatial expression backgrounds while injecting known LR controls. This design preserves realistic coordinate structure and expression noise while keeping ground truth available for AUROC, AUPR, weak-signal recall, empirical FDR, and top-k precision.

The controlled benchmark passed the Phase 4 five-replicate gate, and the real-background benchmark covers three tissues with five replicates per tissue. These properties are summarized in Fig. 2 and Fig. 3. This gives the manuscript a defensible benchmark layer: the truth is still injected, but the spatial mechanisms and real-background noise make it harder than an idealized expression-distance simulation.

Supported claim: the benchmark evaluates mechanism-specific CCC recovery and includes weak-signal and high-expression false-positive controls.

Figures: Fig. 2 and Fig. 3.

Source evidence: `goal1_controlled_simulator_v2_5rep` and `goal1_real_background_v2_5rep` summaries in the Phase 4 evidence index.

Limitation: these datasets contain injected truth and should not be described as native real-tissue ground truth.

## SpatialSelf Improves the Weak-Signal / False-Positive Tradeoff Against Retained Baselines

The retained baseline panel compares SpatialSelf with COMMOT, CellChat, SpaTalk, CellNEST, and SpatialSelf variants under a common benchmark evaluation. LR-Coexp is excluded from the Phase 5 retained method list. DeepTalk and NICHES remain status methods rather than formal metric entries. The Phase 4 replicated baseline evidence supports a metric-specific interpretation: SpatialSelf is useful for a weak-signal and high-expression false-positive tradeoff, but the results do not justify a global "best on every metric" claim.

This distinction is important for reviewer-facing wording. SpatialSelf should be described as supporting a calibrated tradeoff when weak-signal recall is interpreted together with empirical FDR, high-expression false-positive rate, and top-k precision. It should not be described as universally outperforming every method on raw weak recall or every false-positive metric. Fig. 4 should therefore show the retained baseline comparison and the claim guardrail together, so the manuscript communicates both the evidence and the boundary.

Supported claim: SpatialSelf supports a weak-signal / false-positive tradeoff in realistic benchmark settings.

Figure: Fig. 4.

Source evidence: replicated tuned baseline summary and weak-signal claim table in the Phase 4 evidence index.

Limitation: SpatialSelf is not globally best on every metric, tissue, or baseline. SpaTalk can behave as a conservative low-FP baseline in some panels.

## Ablation Studies Show LR Identity and Graph-Wavelet Components Drive Performance

The Phase 4 ablation panel tests whether the observed performance comes from the intended model components rather than only from model capacity. The registered comparison includes the full graph-wavelet model, no LR identity, no spatial graph, no distance feature, no expression feature, GAT and MLP alternatives, and pseudo-label variants. In the five-replicate controlled ablation summary, the full SpatialSelf variant has higher mean AUPR than the no-LR-identity variant (0.430839 versus 0.290923).

These results support the interpretation that LR identity and graph-wavelet modeling contribute to the method's behavior. The ablation also helps separate SpatialSelf from a simple "expression plus distance plus GAT" baseline. Fig. 5 should focus on the component-level evidence rather than presenting the model as a black box.

Supported claim: LR identity and graph-wavelet modeling contribute to performance.

Figure: Fig. 5.

Source evidence: Phase 4 formal ablation summary.

Limitation: the current ablation is a 10-epoch formal coverage panel. Final manuscript effect sizes may still benefit from a longer training-budget sweep.

## Permutation Testing Provides FDR-Calibrated Communication Discoveries

SpatialSelf predictions are evaluated with empirical null models rather than being reported only as uncalibrated scores. The Phase 4 FDR module includes spatial permutation, receiver permutation, and LR-identity permutation. The key lymph-node FDR panel was run with 500 permutations and produces edge-level, LR-pair-level, and cell-type-pair-level outputs.

The strongest current claim is group-level calibration in the key lymph-node panel. The Phase 4 evidence index reports 500-permutation final rows and group-level discoveries, including LR-pair and cell-type-pair outputs. Edge-level significant discoveries are limited or absent at the selected threshold, so the main text should avoid implying broad single-edge discovery power. Fig. 6 should make this explicit by showing both the null calibration layer and the group-level discovery layer.

Supported claim: SpatialSelf outputs can be calibrated with permutation-derived p-values and BH-adjusted FDR under three null models.

Figure: Fig. 6.

Source evidence: Phase 4 500-permutation FDR summary and group summary.

Limitation: current 500-permutation evidence is strongest at group level in one key lymph-node panel; it does not prove broad edge-level discovery power.

## SpatialSelf Recovers Region-Specific Communication in Lymph Node

The lymph-node validation combines curated downstream evidence with positive/negative control ranking. In the Phase 4 key lymph-node downstream panel, positive LR controls show higher mean receiver target correlation than high-expression negative controls. In the Phase 5 real-background lymph-node summary, SpatialSelf ranks the positive controls above high-expression negatives: mean positive rank is 2.0, mean negative rank is 4.5, all three positive controls are in the top three, and no negative controls are in the top three. The high-expression false-positive rate at q95 is 0.000266.

The LR control panel includes CCL19-CCR7 and CXCL13-CXCR5 lymph-node controls, matching the expected T-zone and germinal-center biology used in the injection design. Fig. 7 should first show global control-ranking statistics and then selected LR controls. This ordering avoids presenting the result as a hand-picked example.

Supported claim: key lymph-node downstream and control-ranking evidence support biological plausibility.

Figure: Fig. 7.

Source evidence: Phase 4 key lymph-node downstream summary; Phase 5 real-validation tissue summary, LR panel, and Fig7 source index.

Limitation: lymph-node evidence does not establish global cross-tissue downstream superiority. The real-background control analysis uses injected controls, not native complete real-tissue ground truth.

## High-Resolution Tonsil and Melanoma Validation Support Broader Biological Utility

We further evaluated whether the validation logic extends beyond one lymph-node example. In tonsil and melanoma real-background panels, SpatialSelf ranks all three positive controls in the top three and no high-expression negative controls in the top three. The tonsil controls include IL21-IL21R, CD40LG-CD40, and CXCL13-CXCR5, covering Tfh-B-cell and germinal-center-like biology. The melanoma panel includes CXCL9-CXCR3, SPP1-CD44, and PDCD1-CD274, covering chemokine, macrophage/T-cell boundary, and checkpoint-like interactions. The high-expression false-positive rate at q95 is 0.00261 in tonsil and 0.000716 in melanoma. Receiver-response proxy deltas are positive in lymph node, tonsil, and melanoma.

The Phase 5 cross-tissue validation scope table fixes the allowed interpretation. Lymph node reaches the `key_curated_validation` tier because it combines control separation, key curated downstream validation, and group-level FDR outputs. Tonsil and melanoma reach `proxy_plus_controls_plus_fdr`: both show control separation, positive receiver-response proxy, and group-level FDR outputs, but they do not have the same key curated downstream validation as lymph node. This prevents the Results from overstating the cross-tissue downstream conclusion.

Fig. 8 should therefore present tonsil and melanoma as broader biological validation, but not as complete native-real ground truth. The correct interpretation is that SpatialSelf's top-ranked predictions are more consistent with injected/prior biological controls than high-expression negative controls in multiple tissue backgrounds.

Supported claim: SpatialSelf supports positive/negative control separation in tonsil and melanoma real-background panels.

Figure: Fig. 8.

Source evidence: Phase 5 real-validation tissue summary, method summary, LR panel, claims table, cross-tissue validation scope table, and Fig8 source index.

Limitation: tonsil and melanoma currently use receiver-response proxy rather than key curated downstream target validation. Native real ST ground truth is unavailable, so random LR, shuffled LR database, spatially shuffled coordinate, and non-permissive cell-type controls remain useful future strengthening experiments.
