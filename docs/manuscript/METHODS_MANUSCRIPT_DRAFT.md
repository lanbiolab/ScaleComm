# Methods

This Phase 5 Methods draft is written for manuscript assembly. It records the current reproducible workflow and explicitly flags items that still require final polishing or locked manifests.

## SpatialSelf Overview

SpatialSelf infers candidate cell-cell communication (CCC) events from spatial transcriptomics data by scoring ligand-receptor (LR) activity on a spatial graph. Candidate sender-receiver pairs are represented as graph edges. Each candidate contains expression-derived ligand and receptor features, spatial distance features, cell or spot metadata when available, and an explicit LR identity representation. The graph-wavelet component propagates information across local and broader graph neighborhoods, allowing the model to represent communication patterns that are not limited to the immediate K-nearest-neighbor edge alone.

The model is trained with self-supervised pseudo-labels derived from expression and spatial priors. Final scores are interpreted through benchmark truth tables, ablation analyses, and permutation/FDR calibration rather than as uncalibrated biological discoveries.

## Controlled CCC Simulator

The controlled simulator generates spatial CCC benchmark data with known ground truth. It includes four required mechanism classes:

- contact-dependent or juxtacrine interactions;
- short-to-medium paracrine interactions;
- longer-range niche-restricted chemokine-like interactions;
- high-expression false-positive controls.

Each benchmark replicate records expression matrices, spatial coordinates, cell or spot annotations, candidate interactions, truth LR pairs, truth cell-type pairs, truth spatial niches, and edge-level ground truth. Weak-signal controls are included so that methods cannot be evaluated only on strong, high-expression interactions.

The Phase 4 controlled simulator was regenerated with five replicates and passed the formal mechanism-coverage gate. The main benchmark metrics include AUROC, AUPR, weak-signal recall, empirical FDR, high-expression false-positive rate, and top-k precision.

Source artifacts:

- `outputs/phase4/manuscript_evidence_lock/goal1_controlled_simulator_v2_5rep/formal_benchmark_summary.csv`
- `outputs/phase4/manuscript_evidence_lock/goal1_controlled_simulator_v2_5rep/formal_mechanism_summary.csv`
- `outputs/phase4/manuscript_evidence_lock/goal1_controlled_simulator_v2_5rep/manifest.csv`

## Real-Background Semi-Synthetic Benchmark

The real-background semi-synthetic benchmark uses real-like spatial transcriptomics backgrounds while injecting known LR signals. This preserves coordinate structure, cell-type composition, expression sparsity, library-size variation, and local spatial density more realistically than a fully artificial simulator. The current Phase 4 benchmark covers lymph node, tonsil, and melanoma backgrounds with five replicates per tissue.

Injected controls include lymph-node T-zone and germinal-center-like interactions, tonsil Tfh-B-cell and germinal-center controls, and melanoma tumor-immune or macrophage/T-cell boundary controls. High-expression false-positive LR pairs are injected as negative controls. These datasets provide ground truth for benchmarking because the LR controls are injected, but they should not be described as native real-tissue ground truth.

Source artifacts:

- `outputs/phase4/manuscript_evidence_lock/goal1_real_background_v2_5rep/formal_real_background_summary.csv`
- `outputs/phase4/manuscript_evidence_lock/goal1_real_background_v2_5rep/formal_mechanism_summary.csv`
- `outputs/phase4/manuscript_evidence_lock/goal1_real_background_v2_5rep/manifest.csv`

## LR Database and Candidate Universe

All retained methods should be evaluated on a common LR candidate universe whenever their input format allows it. Ligands or receptors absent from the expression matrix are filtered before scoring. Phase 5 now includes a benchmark LR manifest for the controlled and real-background benchmark universes. The controlled simulator contains 8 unique LR pairs across the required mechanism classes, and the real-background benchmark contains 15 unique LR pairs across lymph node, tonsil, and melanoma backgrounds. These LR universes are benchmark-internal injected LR sets with explicit weak-signal and high-expression false-positive controls.

CellNEST adapter exports are also tracked separately. The current adapter manifests contain CellNEST pair-level LR database exports after dataset filtering. These adapter exports confirm what was provided to CellNEST, but they do not replace the benchmark LR universe.

Complex handling is fixed as a Phase 5 scope boundary. The benchmark uses pair-level LR controls without multi-subunit complex modeling. The current benchmark LR manifest therefore should not be described as a complete external LR database with resolved ligand or receptor complexes. This limitation is acceptable for the current controlled and real-background benchmarks because the injected truth is defined at the pair level, but it must remain explicit in the final manuscript.

Current manuscript rule:

- retained formal baselines: COMMOT, CellChat, SpaTalk, CellNEST;
- status methods: DeepTalk and NICHES;
- excluded baseline: LR-Coexp;
- common candidate universe required where method adapters allow it.

Source artifacts:

- `outputs/phase5/manuscript_submission_lock/goal5_methods_support/phase5_lr_database_manifest.csv`
- `outputs/phase5/manuscript_submission_lock/goal5_methods_support/phase5_methods_support_claims.csv`

## Spatial Graph, KNN, and Graph-Wavelet Range Handling

SpatialSelf constructs a spatial graph from cell or spot coordinates. `K_NN` denotes the number of neighbors used for graph construction. This is distinct from pseudo-label sampling parameters such as `K_pos` and `K_neg`. The graph is used as a computational support for spatial propagation, not as a statement that every biological signal is limited to direct nearest neighbors.

The graph-wavelet representation allows information to be aggregated across graph scales. Distance features and range priors remain part of the candidate edge feature set, so short-range contact-dependent and longer-range niche-restricted LR pairs can be treated differently. This design directly addresses the concern that a plain KNN graph would mechanically restrict every LR signal to the same local neighborhood.

Notation to keep in the final Methods:

- `K_NN`: spatial graph neighbor count;
- `K_pos`: pseudo-label positive sample count or ratio;
- `K_neg`: pseudo-label negative sample count or ratio;
- `R_max`: candidate maximum range or range prior when used;
- `tau`: distance bandwidth;
- wavelet scale: graph-wavelet propagation scale.

## Self-Supervised Pseudo-Labels

SpatialSelf uses pseudo-labels to train a communication scoring model when native cell-cell CCC ground truth is unavailable. Pseudo-labels combine expression and spatial priors and are evaluated by ablation against expression-only, distance-only, hard/soft pseudo-label, and graph-model alternatives. The manuscript should distinguish pseudo-label generation from final biological validation: pseudo-labels provide supervision for model fitting, while benchmark truth tables, negative controls, permutation/FDR, and downstream validation test whether predictions are credible.

Weak-signal results must always be reported with false-positive and FDR metrics. The current claim boundary does not allow stating that SpatialSelf has the highest raw weak-signal recall in every setting.

Source artifacts:

- `outputs/phase2_revised/goal2_weak_signal_tradeoff_readiness/goal2_weak_signal_tradeoff_claims.csv`
- `outputs/phase4/manuscript_evidence_lock/goal6_formal_ablation_controlled_5rep/spatialself_ablation_summary.csv`

## Ablation Analysis

The Phase 4 ablation panel includes the full graph-wavelet model and variants without LR identity, without spatial graph, without distance features, without expression features, GAT alternatives, MLP alternatives, and pseudo-label strategy variants. The five-replicate controlled ablation panel showed higher AUPR for the full model than for the no-LR-identity variant. This supports the interpretation that LR identity and graph-wavelet modeling contribute to performance.

Source artifacts:

- `outputs/phase4/manuscript_evidence_lock/goal6_formal_ablation_controlled_5rep/spatialself_ablation_summary.csv`
- `outputs/phase4/manuscript_evidence_lock/goal6_formal_ablation_controlled_5rep/spatialself_ablation_metrics.csv`

## Baseline Comparison

The retained Phase 5 baseline comparison includes COMMOT, CellChat, SpaTalk, and CellNEST, along with SpatialSelf variants. The comparison uses common benchmark datasets and metric definitions. Baseline outputs are evaluated on the same truth tables whenever possible. Methods that cannot be run as formal metric entries should be documented as status methods with explicit reasons, such as unavailable code, incompatible input/output level, non-reproducible dependencies, or inability to produce required pair-level outputs.

The current Phase 5 baseline status table records SpatialSelf edge-prior, SpatialSelf base, COMMOT, CellChat, SpaTalk, and CellNEST as included in the replicated metric panel, each with 18 metric entries in the current summary. DeepTalk and NICHES are status methods: DeepTalk is projection-smoke-tested but requires suitable trained or pretrained LR-specific model outputs for formal inclusion, and NICHES is blocked by local dependency reproducibility issues. LR-Coexp is excluded by the Phase 5 retained-baseline policy.

The current manuscript wording must state that the replicated baseline panel supports metric- and tissue-specific tradeoffs, not global dominance. It must also state when a method output is projected from cell-type-pair level back to candidate edges.

Source artifacts:

- `outputs/phase2_revised/goal2_replicated_tuned_baseline_panel/goal2_replicated_tuned_metric_summary.csv`
- `outputs/phase2_revised/goal2_replicated_tuned_baseline_panel/goal2_replicated_tuned_claims.csv`
- `outputs/phase5/manuscript_submission_lock/goal5_methods_support/phase5_baseline_status_table.csv`
- `outputs/phase5/manuscript_submission_lock/goal5_methods_support/phase5_methods_support_claims.csv`

## Permutation-Based FDR

Permutation testing estimates empirical significance for SpatialSelf predictions. The required null models are:

- spatial permutation: spatial coordinates are permuted while preserving expression;
- receiver permutation: receiver identities are permuted while preserving sender expression and LR identity;
- LR identity permutation: LR labels are permuted while preserving the broader expression and graph structure.

For each candidate, empirical p-values are calculated from the null distribution and adjusted with the Benjamini-Hochberg procedure. Outputs are separated into edge-level, LR-pair-level, and cell-type-pair-level summaries. The Phase 4 key lymph-node panel uses 500 permutations. Current wording should emphasize group-level FDR evidence because edge-level significant discoveries are limited or absent at the selected threshold.

Source artifacts:

- `outputs/phase4/manuscript_evidence_lock/goal3_fdr_500perm_lymph_node/goal5_formal_fdr_summary.csv`
- `outputs/phase4/manuscript_evidence_lock/goal3_fdr_500perm_lymph_node/goal5_formal_fdr_group_summary.csv`

## Real-Sample and Real-Background Validation

Native unlabeled spatial transcriptomics datasets generally do not provide complete edge-level CCC ground truth. Phase 5 therefore uses a layered validation strategy:

- positive LR controls;
- high-expression negative controls;
- receiver-response or downstream target correlation;
- region, niche, or boundary relevance;
- permutation/FDR group summaries;
- baseline-matched ranking comparisons.

The Phase 5 real-validation summary covers lymph node, tonsil, and melanoma real-background panels. SpatialSelf ranks all three positive controls in the top three and no high-expression negative controls in the top three across all three tissues. High-expression false-positive rates at q95 are 0.000266 in lymph node, 0.00261 in tonsil, and 0.000716 in melanoma. Receiver-response proxy deltas are positive in all three tissues.

The key lymph-node downstream panel provides curated downstream support. Phase 5 also records a cross-tissue validation scope table. Lymph node is assigned the `key_curated_validation` tier. Tonsil and melanoma are assigned `proxy_plus_controls_plus_fdr`, meaning they have control separation, receiver-response proxy, and group-level FDR outputs, but not the same key curated downstream validation as lymph node. This limitation should remain explicit.

Source artifacts:

- `outputs/phase5/manuscript_submission_lock/goal2_real_validation/phase5_real_validation_tissue_summary.csv`
- `outputs/phase5/manuscript_submission_lock/goal2_real_validation/phase5_real_validation_method_summary.csv`
- `outputs/phase5/manuscript_submission_lock/goal2_real_validation/phase5_real_validation_lr_panel.csv`
- `outputs/phase5/manuscript_submission_lock/goal2_real_validation/phase5_real_validation_claims.csv`
- `outputs/phase5/manuscript_submission_lock/goal2_cross_tissue_scope/phase5_cross_tissue_validation_scope.csv`
- `outputs/phase5/manuscript_submission_lock/goal2_cross_tissue_scope/phase5_cross_tissue_validation_claims.csv`
- `outputs/phase4/manuscript_evidence_lock/goal5_downstream_key_lymph_node/phase4_downstream_key_lymph_node_summary.csv`

## Figure and Source-Data Reproducibility

Phase 5 tracks each manuscript figure with a source index, supported claim, limitation, and export QA. Fig. 7 and Fig. 8 have Phase 5 real-validation exports in PDF, SVG, and TIFF. Fig. 1-Fig. 6 still rely on Phase 4 evidence-lock figures and require final journal polish before submission.

Source artifacts:

- `outputs/phase5/manuscript_submission_lock/phase5_real_validation_figure_source_index.csv`
- `outputs/phase5/manuscript_submission_lock/phase5_real_validation_figure_qa.csv`

## Reproducibility Commands

Phase 4 evidence index and audit:

```bash
conda run -n Benchmark python scripts/phase4/build_phase4_evidence_index.py --config configs/experiments/phase4_manuscript_evidence_lock.yaml
conda run -n Benchmark python scripts/phase4/audit_phase4.py --config configs/experiments/phase4_manuscript_evidence_lock.yaml
```

Phase 5 real-validation summaries:

```bash
conda run -n Benchmark python scripts/phase5/summarize_phase5_real_validation.py
```

Phase 5 real-validation figures:

```bash
conda run -n Benchmark python scripts/phase5/figures/plot_phase5_real_validation_figures.py --output-dir outputs/phase5/manuscript_submission_lock
```

Phase 5 evidence index and audit:

```bash
conda run -n Benchmark python scripts/phase5/build_phase5_evidence_index.py --config configs/experiments/phase5_submission_manuscript_lock.yaml
conda run -n Benchmark python scripts/phase5/audit_phase5.py --config configs/experiments/phase5_submission_manuscript_lock.yaml
```

Remaining manuscript-polish tasks:

- add native-real random/shuffled negative controls if time permits;
- redraw Fig. 1-Fig. 6 as final journal figures.
