# Phase 2 Goals

Phase 2 turns the Phase 1 benchmark framework into a manuscript-ready SpatialSelf method. The priority is not to add more baselines; it is to fix the two largest Phase 1 weaknesses: weak-signal recovery and overly conservative FDR.

## Goal 1: Weak-Signal Recovery

Purpose: recover low-expression or subtle CCC without increasing false-positive high-expression calls.

Current implementation:

- `scripts/benchmark/run_spatialself.py` now supports `train_sampling_strategy`, `focal_loss_gamma`, and `focal_loss_alpha`.
- `balanced_per_lr` sampling keeps each LR represented in training instead of letting large LR candidate sets dominate.
- The weak-like mining proxy now works per LR and excludes high-expression false-positive LR controls.
- Phase 2 candidate ablations are registered in `scripts/benchmark/run_spatialself_ablation.py`.
- The runner is `scripts/phase2/run_goal1_weak_signal_recovery.py`.

Initial controlled five-replicate evidence:

- Output: `outputs/phase2/goal1_weak_signal_recovery/controlled_five_replicate_initial/weak_signal_recovery_summary.csv`
- Phase 1 controlled weak-recall reference: 0.155
- Phase 1 default-style candidate: weak recall 0.113, AUPR 0.211, FDR 0.829
- Edge-prior ranking: weak recall 0.164, AUPR 0.234, FDR 0.825, high-expression FP rate 0.007
- Focal weak-mining edge-prior: weak recall 0.170, AUPR 0.239, FDR 0.823, high-expression FP rate 0.024

Interpretation: Phase 2 has an initial controlled-benchmark improvement over the Phase 1 weak-recall reference without FDR inflation. The next required test is full-slide semi-synthetic validation, because controlled-only gains are not enough for model selection.

Full-slide semi-synthetic evidence:

- Output: `outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_initial/weak_signal_recovery_summary.csv`
- Full-slide weak-recall reference: 0.040
- Edge-prior ranking at the default 0.95 threshold: weak recall 0.0398, AUPR 0.384, FDR 0.451, high-expression FP rate 0.00012
- Focal weak-mining edge-prior does not transfer: weak recall 0.0126, AUPR 0.328, FDR 0.604, high-expression FP rate 0.0478
- Soft weak-positive mining variants also reduce weak recall and increase FDR on full-slide realistic background.

Operating-point evidence:

- Output: `outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_threshold_tradeoff/threshold_tradeoff_summary.csv`
- Edge-prior ranking at threshold quantile 0.94 reaches weak recall 0.0508.
- FDR is 0.463, an increase of 0.012 versus the default 0.95 threshold.
- High-expression false-positive rate remains low at 0.00032.

LR-pair-balanced operating diagnostic:

- Runner: `scripts/phase2/run_goal1_lr_balanced_threshold.py`
- Output: `outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_lr_balanced_threshold/lr_balanced_threshold_summary.csv`
- This is a post-hoc operating-rule diagnostic from existing full-slide edge-prior predictions, not a newly trained model and not a formal FDR procedure.
- The useful rule is LR-wise quantile 0.90 plus a global score floor at 0.90. It reaches weak recall 0.0871, FDR 0.455, FDR delta 0.0036 versus q0.95, and high-expression false-positive rate 0.00305.
- Pure LR-wise quantile calling is not acceptable: it raises FDR to 0.672 and high-expression false-positive rate to 0.200.
- Controlled cross-replicate stability output: `outputs/phase2/goal1_weak_signal_recovery/controlled_lr_balanced_threshold_stability/lr_balanced_replicate_aggregate.csv`
- The same LR-wise quantile 0.90 plus global floor 0.90 rule does not pass controlled replicate constraints: pass fraction 0/5, mean weak recall 0.178, mean FDR delta 0.037, and mean high-expression false-positive rate 0.028.

Mechanism-specific weak-signal diagnostic:

- Runner: `scripts/phase2/run_goal1_mechanism_weak_signal_summary.py`
- Mechanism-specific threshold runner: `scripts/phase2/run_goal1_mechanism_threshold_tradeoff.py`
- Combined output: `outputs/phase2/goal1_weak_signal_recovery/goal1_mechanism_weak_signal_combined_summary.csv`
- Controlled mechanism output: `outputs/phase2/goal1_weak_signal_recovery/controlled_five_replicate_mechanism_summary/mechanism_weak_signal_summary.csv`
- Full-slide q0.94 mechanism output: `outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_operating_q94_mechanism_summary/mechanism_weak_signal_summary.csv`
- Mechanism-specific threshold output: `outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_mechanism_threshold_tradeoff/mechanism_threshold_tradeoff_summary.csv`
- Mechanism-targeted weak-mining probe output: `outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_mechanism_targeted_probe/weak_signal_recovery_summary.csv`
- This is a post-hoc diagnostic from existing predictions, not a newly trained model.
- In the controlled five-replicate benchmark, edge-prior q0.95 weak recall is 0.461 for contact, 0.383 for short/medium paracrine, and 0.040 for long-range chemokine-like signals.
- In the controlled five-replicate benchmark, focal weak mining improves short/medium paracrine weak recall to 0.552 but reduces contact to 0.291 and long-range to 0.007.
- In the full-slide sparse benchmark, weak positives are long-range only; q0.94 improves long-range weak recall from 0.0398 to 0.0508 for edge-prior ranking.
- Full-slide mechanism-targeted training probes do not transfer: long-range soft weak recall is 0.0031 and non-contact soft weak recall is 0.0078, both below the edge-prior comparator in the probe run.
- A mechanism-specific operating-point diagnostic gives a small long-range gain: long-range q0.92 reaches weak recall 0.0417 with FDR delta 0.0027 and no increase in high-expression false-positive rate. The combined mechanism-specific rule reaches weak recall 0.0417 and FDR 0.450, which is weaker than the global q0.94 and LR-balanced operating points.

Interpretation: the current Phase 2 survivor is not the focal weak-mining model. It is edge-prior ranking plus a calibrated operating rule. The model ranking contains recoverable weak signal on full-slide realistic background, but the default 0.95 call threshold is too conservative for weak-signal recall. LR-wise quantile 0.90 with a global score floor 0.90 is the strongest full-slide post-hoc diagnostic, but it fails controlled cross-replicate FDR/FP constraints. It should not replace the safer global q0.94 operating point in Goal 3 unless future realistic-background replicate validation supports it.

Current Goal 1 conclusion: edge-prior ranking remains the transferable candidate, while focal/targeted weak mining is a controlled-only or non-transfer gain. Mechanism-level diagnostics show that long-range weak-signal recovery is still the main bottleneck, especially under realistic full-slide background. LR-balanced thresholding is useful as a full-slide diagnostic but not stable enough to be promoted; mechanism-specific thresholds remain secondary diagnostics.

## Goal 2: FDR Calibration

Purpose: make empirical p-values useful rather than uniformly over-conservative.

Current implementation:

- `src/spitialself/fdr.py` now supports score permutation, spatial permutation, receiver permutation, LR identity permutation, within-region permutation, and cell-type preserving permutation.
- `scripts/phase2/run_goal2_fdr_calibration.py` writes candidate-edge FDR, LR-pair FDR, cell-type-pair FDR, pathway/module FDR, a calibration summary, a group summary, and a manifest.
- The runner also records the Goal 1 calibrated operating point and adds empirical-null diagnostics (`null_z`, Gaussian-tail p-value/BH-FDR, and histogram local-FDR). These diagnostic columns are not yet the final manuscript discovery rule.

Initial full-slide edge-prior calibration smoke:

- Output: `outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_calibration_initial/fdr_calibration_summary.csv`
- Dataset: full-slide sparse human lymph-node semi-synthetic benchmark with 332,328 candidate interactions.
- The smoke run covers all six Phase 2 null models with 20 permutations.
- Candidate-edge permutation BH remains conservative: minimum empirical p-value is 0.0476 and no edge-level BH discoveries are possible at this permutation count with 332,328 tests.
- The calibrated operating point from Goal 1 is preserved: q=0.94, weak recall 0.0508, realized FDR 0.463, high-expression false-positive rate 0.00032.
- Group-level summaries now track positive-control groups and high-expression negative-control groups. The runner reports both Simes-combined edge p-values and a direct top-tail group permutation test.
- Direct group-level permutation improves positive-control power relative to Simes: in the current 20-permutation smoke, the best LR-pair positive-control BH-FDR is 0.119 under LR-identity permutation, while high-expression negative-control groups are not called significant. This is promising but not yet a significant-discovery result.
- Empirical-null diagnostics show high z-score tails and some Gaussian/local-FDR discoveries, which means the next step is to decide whether a formal empirical-null or group-level procedure can be justified without losing negative-control calibration.

Focused direct group-level calibration:

- Runner: `scripts/phase2/run_goal2_direct_group_grid.py`
- Seed-stability runner: `scripts/phase2/run_goal2_direct_group_seed_stability.py`
- Controlled replicate-stability runner: `scripts/phase2/run_goal2_direct_group_replicate_stability.py`
- 100-permutation grid: `outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_grid/direct_group_grid_summary.csv`
- 500-permutation focused check: `outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_grid_500perm_focused/direct_group_grid_summary.csv`
- Selected-setting seed stability: `outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_seed_stability/direct_group_seed_stability_summary.csv`
- Selected-setting controlled replicate stability: `outputs/phase2/goal2_fdr_calibration/controlled_edge_prior_direct_group_replicate_stability/direct_group_replicate_stability_summary.csv`
- At LR-pair level, score permutation with top 5% group statistic calls 1 positive-control LR pair significant, with positive-control BH-FDR 0.00998 and 0 significant high-expression negative-control groups.
- At LR-pair level, LR-identity permutation with top 10% group statistic calls all 3 positive-control LR pairs significant, with positive-control BH-FDR 0.00333 and 0 significant high-expression negative-control groups.
- Pathway/module-level direct tests also call positive-control modules significant under score and LR-identity nulls.
- Receiver permutation remains much stricter and low power: no positive-control LR-pair or module discoveries in the 500-permutation focused check.
- The selected LR-identity/LR-pair/top-10% setting is stable across 5 random seeds at 200 permutations per seed: 5/5 seeds pass, all 3 positive-control LR groups are significant in every seed, and 0 high-expression negative-control groups are significant.
- The same selected setting is stable across the 5 controlled ground-truth replicates at 100 permutations: 5/5 replicates pass, positive-control significant LR groups range from 4 to 5, and 0 high-expression negative-control groups are significant.

Interpretation: direct group-level permutation is now the leading Phase 2 FDR candidate. The scientifically conservative framing is that SpatialSelf has usable group-level discoveries under score-permutation and LR-identity-permutation nulls, while receiver-permutation should be reported as a stricter null that remains underpowered for this full-slide semi-synthetic setting. The selected setting now has seed-stable single-full-slide support and controlled cross-replicate support. Before manuscript-level FDR claims, it still needs realistic-background replicate/dataset validation and preferably higher permutation counts.

Current Goal 2 conclusion: candidate-edge BH remains too granular/conservative for large candidate tables, but direct group-level permutation now meets the initial positive-control/negative-control criterion on the full-slide semi-synthetic benchmark under score-permutation and LR-identity-permutation nulls. The selected LR-identity/LR-pair/top-10% setting is stable across random seeds on one full-slide replicate and across the 5 controlled ground-truth replicates. The next requirement is broader robustness on realistic-background replicates/datasets and preferably higher permutation counts before treating it as a manuscript-level default.

## Goal 3: Default Model Selection

Purpose: select the Phase 2 default SpatialSelf model using controlled, semi-synthetic, and lymph-node validation evidence.

Candidate models:

- wavelet default
- wavelet edge-prior
- weak-mining wavelet
- focal weak-mining edge-prior
- GAT reference
- hybrid wavelet-GAT if implemented
- no-LR-identity / no-distance / no-expression controls

Current implementation:

- Runner: `scripts/phase2/run_goal3_model_selection.py`
- Output: `outputs/phase2/goal3_model_selection/model_selection_evidence.csv`
- Criteria table: `outputs/phase2/goal3_model_selection/model_selection_criteria.csv`
- Recommendation: `outputs/phase2/goal3_model_selection/model_selection_recommendation.json`

Current model-selection result:

- Recommended Phase 2 default candidate: `wavelet_calibrated_fdr`
- This means `wavelet_mechanism_niche_edge_prior_ranking` plus the global q0.94 calibrated operating point and direct group-level FDR.
- Current model-selection score: 10, after adding selected direct-FDR seed-stability and controlled replicate-stability evidence.
- Controlled weak recall: 0.164, above the 0.155 reference.
- Full-slide weak recall at default q=0.95: 0.0398, near the 0.040 boundary.
- Full-slide global calibrated operating point q=0.94: weak recall 0.0508 with FDR delta 0.012 versus q=0.95 and high-expression FP rate 0.00032.
- Full-slide LR-balanced operating diagnostic: weak recall 0.0871 with FDR delta 0.0036 versus q=0.95 and high-expression FP rate 0.00305, but it is not promoted because controlled cross-replicate stability fails FDR/FP constraints.
- Direct group-level FDR: LR-identity permutation, LR-pair level, top 10% group statistic gives positive-control BH-FDR 0.00333, 3 significant positive-control LR pairs, and 0 significant high-expression negative-control groups.
- Seed-stability evidence: the selected LR-identity/LR-pair/top-10% FDR setting passes in 5/5 random seeds at 200 permutations per seed, with 3 positive-control groups and 0 high-expression negative-control groups significant in every seed.
- Lymph-node validation: positive controls outrank negative controls, negative top-3 count is 0, and mean positive downstream Spearman is 0.765.

Non-selected candidates:

- `wavelet_default` keeps low high-expression false positives and good lymph-node downstream validation, but full-slide weak recall is near zero at q=0.95.
- `wavelet_edge_prior` is the best raw model candidate, but without FDR calibration it lacks the statistical-discovery layer needed for Goal 2.
- `phase2_focal_weak_mining_edge_prior_wavelet` improves controlled weak recall but fails transfer to full-slide realistic background, with FDR 0.604 and high-expression FP rate 0.0478.
- `gat_reference` does not provide a consistent controlled/full-slide advantage.
- no-LR-identity, no-distance, and no-expression controls are covered by the Phase 1 architecture/component ablation and are not competitive enough to be default candidates.
- `hybrid_wavelet_gat` remains unimplemented and should stay a future candidate rather than being treated as missing evidence for the current default.

Interpretation: Goal 3 now has a Phase 2 default candidate. The cautious wording is important: this is the default for the next Phase 2 experiments, not yet the final manuscript default. The FDR component is seed-stable on one full-slide replicate and stable across controlled synthetic replicates, but still lacks realistic-background cross-dataset validation. Goals 4-5 still need to test the default on high-resolution and biological-interpretability workflows.

## Goal 4: High-Resolution Real Data

Purpose: test whether SpatialSelf can support high-resolution ST claims.

Priority datasets:

- Xenium or Visium HD subset
- Slide-tags tonsil
- melanoma TME

Initial tonsil evidence:

- Runner: `scripts/phase2/run_goal4_high_resolution_tonsil_subset.py`
- Output: `outputs/phase2/goal4_high_resolution/tonsil_existing_prediction_subset/`
- Dataset: local human tonsil/Slide-tags-style data with 5,778 cells and 13 cell types.
- The script validates the node order against `HumanTonsil_expression.csv.gz` and `graph_visium.pt`.
- It streams the existing full prediction table with 58,717,428 candidate rows across 924 LR pairs.
- Current runtime evidence: 28.5 seconds wall-clock, 581 MB peak RSS for the stream summary.
- Top cell-type-pair mass is dominated by B germinal center interactions and FDC/B-germinal-center edges, which is structurally plausible for tonsil but not yet a biological validation.
- Five tonsil positive-control LR priors are present in the LR table. Current evidence is mixed: `CXCL13-CXCR5` is the best positive by mean score rank at 305/924 and has a small set of high-scoring localized edges; none of the five positives are in the top decile by mean score.

Interpretation: Goal 4 now has an initial high-resolution wiring and scalability artifact, but it is not yet a manuscript-level result. It uses an existing prediction stream rather than a fresh `wavelet_calibrated_fdr` inference run, and the positive-control evidence is not strong enough to support a high-resolution biological claim. The next required step is to rerun the Phase 2 default model on tonsil or another high-resolution subset and add downstream/pathway validation.

Fresh Phase 2 default subset evidence:

- Runner: `scripts/phase2/run_goal4_fresh_tonsil_subset.py`
- Output: `outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset/`
- Dataset subset: 720 tonsil cells, 10 cell types, 45 LR pairs, and 406,728 candidate edges.
- Model: fresh `spatialself_wavelet` run using the current Phase 2 default candidate settings (`wavelet_calibrated_fdr`: edge-prior features, mechanism/niche pseudo-labeling, ranking loss, q=0.94 operating point).
- Runtime: 12.8 seconds wall-clock, 2.1 seconds for train+inference, 794 MB peak RSS.
- Positive/negative controls: 6 positive tonsil priors and 3 housekeeping high-expression negative controls are included.
- Current biological-control signal: `CXCL13-CXCR5` ranks 4/45 by mean score and is the only positive control in the top decile; all 3 housekeeping negative controls are near the bottom and none are in the top decile.
- Top cell-type pairs at q=0.94 include FDC-FDC, B germinal-center self-interactions, mDC-FDC, T_CD4-FDC, and B germinal-center-FDC edges.

Interpretation: this is the first fresh high-resolution real-data run of the Phase 2 default candidate. It supports wiring, scalability, and limited control enrichment on a tonsil subset. It still does not satisfy the full Goal 4 manuscript standard because it is a subset, lacks downstream target/pathway validation, and is not yet compared with retained baselines on the same real-data subset.

Fresh subset scaling evidence:

- Runner: `scripts/phase2/run_goal4_high_resolution_scaling.py`
- Output: `outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/`
- Current sizes: 720, 1,200, and 1,800 cells.
- Largest subset: 1,800 cells, 45 LR pairs, and 1,641,363 candidate edges.
- Largest runtime/memory: 34.4 seconds wall-clock and about 1.6 GB peak RSS.
- Largest subset controls: one positive control in the top decile and zero housekeeping high-expression negatives in the top decile.

Interpretation: this adds subset-level scaling evidence for the fresh Phase 2 default tonsil workflow. It supports the claim that candidate generation and model inference can be run beyond the initial 720-cell subset, but it is still not full high-resolution evidence for Slide-tags tonsil, Xenium, or Visium HD.

Fresh same-subset baseline comparison:

- Runner: `scripts/phase2/run_goal4_tonsil_baseline_matched.py`
- Output: `outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/`
- Methods in the current run: SpatialSelf calibrated default, COMMOT, CellChat, and SpaTalk.
- Evaluation scope: biological positive/negative control ranking on the same 720-cell tonsil subset, not ground-truth AUROC/AUPR.
- SpatialSelf: 1 positive and 0 housekeeping negative controls in the top decile; rank-validation pass.
- COMMOT: valid nonzero candidate-edge scores but no positive control in the top decile; rank-validation fail.
- CellChat: valid projected cell-type-pair scores but no positive control in the top decile and one housekeeping negative ranks high; rank-validation fail.
- SpaTalk: flat all-zero scores in this setting and is recorded as not claim-eligible.

Interpretation: this is the first same-subset retained-baseline comparison for the fresh high-resolution tonsil workflow. It supports the current SpatialSelf high-resolution control-ranking result, but it is still a single subset and should not be phrased as final broad baseline superiority.

Fresh CCC module diagnostic:

- Runner: `scripts/phase2/run_goal4_tonsil_ccc_modules.py`
- Output: `outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_fresh_subset/`
- Input: fresh 720-cell Goal 4 tonsil subset prediction.
- Current result: 45 LR pairs are grouped into 4 modules.
- Module 3 is a long-range chemokine positive-control module with 3 positive controls, 0 housekeeping negatives, and positive-control enrichment BH-FDR 0.0056.
- Module 4 is a high-expression negative-control module with 3 housekeeping negatives, 0 positives, and housekeeping-negative enrichment BH-FDR 0.00042.

Interpretation: this adds module/control separation evidence for the fresh high-resolution tonsil workflow. It remains a single 720-cell subset diagnostic. Spatial concentration values are weak, so this should not be framed as strong spatial confinement, full high-resolution validation, or real-data ground truth.

n1800 CCC module stress test:

- Runner: `scripts/phase2/run_goal4_tonsil_ccc_modules.py`
- Output: `outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_n1800_subset/`
- Input: 1,800-cell Goal 4 scaling subset.
- Current result: 45 LR pairs are grouped into 4 modules.
- The housekeeping-negative enriched module has 3 housekeeping negatives but also contains `CXCL13-CXCR5` and `CCL21-CCR7`.
- Housekeeping-negative enrichment remains significant at BH-FDR 0.0042, while positive-control module enrichment is not significant (best BH-FDR 0.17-0.32).

Interpretation: this stress test weakens the clean module/control-separation story from the 720-cell subset. It should be used to constrain high-resolution module claims and motivate more stable domain-specific module definitions before manuscript-level biological interpretation.

## Goal 5: Biological Interpretability

Purpose: show that predicted CCC is associated with downstream response and pathway structure, not just spatial colocalization.

Analyses:

- LR-target downstream validation
- LR embedding UMAP/clustering
- pathway category enrichment
- CCC module discovery
- mechanism category analysis

Initial tonsil interpretability evidence:

- Runner: `scripts/phase2/run_goal5_tonsil_interpretability.py`
- Output: `outputs/phase2/goal5_interpretability/tonsil_fresh_subset/`
- Input: fresh Goal 4 tonsil subset prediction from `outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset/`
- The runner reads target genes from the original tonsil expression matrix, not the LR-only subset expression table.
- It tests 45 LR pairs using curated tonsil/immune target signatures, 200 permutation tests, BH-FDR, receiver-cell-type residualized Spearman, expected-receiver subset Spearman, pathway-category summaries, and simple CCC activity modules.
- Current raw downstream result is mixed: positive controls have mean Spearman -0.002, housekeeping negative controls 0.017, and background LR 0.153.
- After receiver-cell-type residualization, positive controls improve to mean residual Spearman 0.087 and housekeeping negatives drop to -0.007, but background LR remain high at 0.140.
- Four positive controls and zero housekeeping negative controls are significant at residual BH-FDR <= 0.10.
- The highest raw downstream correlations are mostly broad background LR with early-response/antigen-presentation signatures; the top residual LR includes `CCL21-CCR7`.

Interpretation: this is useful initial interpretability evidence but not yet a strong manuscript claim. It suggests SpatialSelf positive controls can associate with receiver-side response after controlling for receiver cell type, while also showing that broad immune activation/background LR can dominate target-correlation metrics. The next step is stronger ligand-target priors, larger real-data runs, and baseline-matched downstream validation.

Larger-subset downstream stress test:

- Runner: `scripts/phase2/run_goal5_tonsil_interpretability.py`
- Output: `outputs/phase2/goal5_interpretability/tonsil_n1800_subset/`
- Input: 1,800-cell Goal 4 scaling subset at `outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/n1800/`
- It tests the same 45 LR pairs and target signatures with 200 permutations.
- Current result is mixed/negative for downstream interpretability: positive-control residual mean Spearman is -0.016, housekeeping negative residual mean is 0.009, background LR residual mean is 0.049, and residual significant positive/negative controls are 0/0 at BH-FDR <= 0.10.

Interpretation: the 1,800-cell stress test does not preserve the 720-cell positive-control residual advantage. This should be used to prevent overclaiming and to motivate stronger receiver/domain-specific ligand-target priors before using downstream correlation as a main biological validation.

Initial LR embedding structure evidence:

- Runner: `scripts/phase2/run_goal5_lr_embedding_interpretability.py`
- Output: `outputs/phase2/goal5_interpretability/lr_embedding_tonsil_fresh_subset/`
- Input model: fresh Goal 4 tonsil subset model checkpoint at `outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset/tonsil_fresh_spatialself_wavelet_model.pt`
- The runner extracts learned LR identity embeddings and links them to LR metadata, model activity summaries, downstream residual correlations, PCA/UMAP coordinates, cluster summaries, and cluster enrichment tests.
- Current subset result is weak: mechanism silhouette is -0.081, control-class silhouette is -0.052, and zero cluster enrichments are significant at BH-FDR <= 0.10.
- Embedding norm has only small correlations with spatial range and downstream residual correlation.

Interpretation: this should be recorded as a negative/mixed interpretability diagnostic, not as evidence that LR embeddings recover biological mechanism. For manuscript-level interpretability, this analysis needs a larger LR universe, stronger pathway annotations, baseline/context controls, and preferably replicate-level stability checks.

Initial mechanism/category interpretability evidence:

- Runner: `scripts/phase2/run_goal5_mechanism_category_interpretability.py`
- Output: `outputs/phase2/goal5_interpretability/mechanism_category_tonsil_fresh_subset/`
- The runner combines downstream residual correlations, LR summary statistics, spatial concentration, and LR embedding metadata by mechanism and LR functional category.
- It covers contact, short/medium paracrine, long-range chemokine, high-expression negative, chemokine, adhesion/contact, checkpoint-like, antigen-presentation, TNF/checkpoint, background, and housekeeping categories.
- Current result: positive controls exceed housekeeping negatives after receiver-cell-type residualization by 0.094 mean Spearman.
- Four positive controls and zero housekeeping negatives are significant at residual BH-FDR <= 0.10.
- The strongest mechanism by mean residual correlation is short/medium paracrine, and the strongest category is adhesion/contact.
- The strongest group-level enrichment is still driven by broad background/short-medium paracrine LR, not by a clean positive-control-only category.
- Embedding cluster category purity is 0.467, so LR embedding structure remains weak/mixed.

Interpretation: this adds a traceable mechanism/category diagnostic layer, but it does not solve Goal 5. The cautious claim is that SpatialSelf predictions show receiver-response associations stratified by LR mechanism/category while also revealing broad immune/background correlations that need stronger ligand-target priors and larger datasets.

Goal 5 interpretability synthesis:

- Lymph-node runner: `scripts/phase2/run_goal5_lymph_node_interpretability.py`
- Lymph-node output: `outputs/phase2/goal5_interpretability/lymph_node_real_background/`
- Runner: `scripts/phase2/run_goal5_interpretability_synthesis.py`
- Output: `outputs/phase2/goal5_interpretability/synthesis/`
- Source tables: `interpretability_synthesis_summary.csv`, `interpretability_synthesis_claims.csv`, and `interpretability_synthesis_metrics.csv`
- Current synthesis level: `mixed_initial_support_not_manuscript_ready`
- Lymph-node result: `real_background_injected_control_support`, with 3/3 positive controls downstream-supported, 1/1 weak positive downstream-supported, zero high-expression negatives in the top three ranks, and mean positive downstream Spearman 0.752.
- Support score: 4
- Contradiction score: 5
- Supported initial signals: 720-cell positive controls exceed housekeeping negatives after residualization, the 720-cell CCC module diagnostic has a positive-control module BH-FDR of 0.0056, mechanism/category summaries retain a positive-housekeeping residual difference of 0.094, and lymph-node real-background injected controls support downstream/rank validation.
- Contradictory signals: background LR residual correlations remain stronger than positives, the 1,800-cell downstream stress test has negative positive-housekeeping residual difference, the 1,800-cell module stress test weakens positive-control enrichment, and LR embedding silhouettes are negative with zero significant cluster enrichments.

Interpretation: Goal 5 now has a stronger synthesis artifact because lymph-node real-background injected controls support downstream/rank validation. It should still be used to constrain broad claims rather than promote a finished biological validation story. The current defensible wording is that SpatialSelf has mixed initial evidence linking selected predicted LR activity to receiver response and module structure, with cross-real-background injected-control support, while larger-subset tonsil and embedding diagnostics argue against manuscript-ready interpretability claims.

## Goal 6: Paper-Ready Tables And Figures

Purpose: assemble reproducible source tables and main-figure scripts.

Required figure sources:

- benchmark performance
- weak-signal/FDR tradeoff
- ablation
- FDR calibration
- lymph-node global validation
- high-resolution scalability
- biological module/downstream validation

Initial figure/source-table assembly:

- Runner: `scripts/phase2/run_goal6_figure_assembly.py`
- Lymph-node validation runner: `scripts/phase2/run_goal6_lymph_node_validation.py`
- Lymph-node baseline validation runner: `scripts/phase2/run_goal6_lymph_node_baseline_validation.py`
- Output: `outputs/phase2/goal6_paper_figures/`
- Lymph-node validation output: `outputs/phase2/goal6_lymph_node_validation/full_slide_sparse_wavelet_edge_prior/`
- Lymph-node baseline validation output: `outputs/phase2/goal6_lymph_node_validation/baseline_matched_full_slide_sparse/`
- Figure files: `outputs/phase2/goal6_paper_figures/figures/phase2_overview.png` and `outputs/phase2/goal6_paper_figures/figures/phase2_overview.pdf`
- Manuscript-style panel files are exported under `outputs/phase2/goal6_paper_figures/manuscript_panels/` as PNG/PDF pairs.
- Figure QA files are exported under `outputs/phase2/goal6_paper_figures/figure_qa/`.
- Source tables:
  - `source_tables/goal1_controlled_weak_signal.csv`
  - `source_tables/goal1_mechanism_weak_signal.csv`
  - `source_tables/goal1_full_slide_operating_point.csv`
  - `source_tables/goal2_direct_group_fdr.csv`
  - `source_tables/goal2_direct_group_replicate_stability.csv`
  - `source_tables/goal3_model_selection.csv`
  - `source_tables/goal4_high_resolution_controls.csv`
  - `source_tables/goal4_high_resolution_scaling.csv`
  - `source_tables/goal4_tonsil_baseline_matched.csv`
  - `source_tables/goal4_tonsil_ccc_modules.csv`
  - `source_tables/goal4_tonsil_ccc_modules_n1800.csv`
  - `source_tables/goal5_interpretability.csv`
  - `source_tables/goal5_interpretability_n1800.csv`
  - `source_tables/goal5_mechanism_category.csv`
  - `source_tables/goal5_lymph_node_interpretability.csv`
  - `source_tables/goal5_interpretability_synthesis.csv`
  - `source_tables/goal6_controlled_baseline_comparison.csv`
  - `source_tables/goal6_full_slide_baseline_comparison.csv`
  - `source_tables/goal6_component_ablation.csv`
  - `source_tables/goal6_lymph_node_global_validation.csv`
  - `source_tables/goal6_lymph_node_baseline_matched.csv`
  - `source_tables/goal6_claim_traceability.csv`

The current overview figure has ten panels: retained baseline source, component ablation, controlled weak-signal recall, full-slide operating-point tradeoff, direct group-level FDR, model-selection score, tonsil subset control ranks, interpretability diagnostics, lymph-node real-background injected-control validation, and baseline-matched lymph-node controls. The same evidence is also exported as independent manuscript-style panels, including separate mechanism-specific weak-signal, high-resolution subset-scaling, tonsil same-subset baseline-matched, tonsil CCC module/control separation, and mechanism/category interpretability panels. Goal 6 also exports controlled cross-replicate FDR, n1800 CCC module stress-test, 1,800-cell downstream stress-test, lymph-node Goal 5 interpretability, and Goal 5 synthesis source tables for claim traceability. Automated figure QA now passes file/source checks with 16 PNG files, 16 PDF files, 30 manuscript-panel files, 24 non-empty source tables, and no missing paired exports. These are useful as Phase 2 evidence assembly artifacts, but they are not yet the final paper figure suite. Goal 1 mechanism analysis is a post-hoc diagnostic, Goal 4 is still subset-level, Goal 5 synthesis has lymph-node injected-control support but remains mixed and not manuscript-ready, baseline rows must preserve their evidence levels, the component ablation source is Phase 1 controlled component-level evidence rather than a final Phase 2 default ablation, and the lymph-node sources are regenerated by Phase 2 runners as full-slide sparse real-background injected-control evidence rather than unlabeled real-data ground truth. The current export bundle is PNG/PDF-only, so final submission still needs manual visual review plus SVG/TIFF export and journal-size checks.

## Current Commands

Run Goal 1 initial panel:

```bash
conda run -n Benchmark python scripts/phase2/run_goal1_weak_signal_recovery.py \
  --dataset-root outputs/phase1/exp01_controlled_benchmark/datasets \
  --output-dir outputs/phase2/goal1_weak_signal_recovery/controlled_five_replicate_initial \
  --epochs 6 \
  --max-train-samples 6000 \
  --ablation wavelet_mechanism_niche_lr_prior_lrwise_ranking \
  --ablation wavelet_mechanism_niche_edge_prior_ranking \
  --ablation phase2_weak_mining_edge_prior_wavelet \
  --ablation phase2_focal_weak_mining_edge_prior_wavelet
```

Summarize and audit:

```bash
conda run -n Benchmark python scripts/phase2/summarize_phase2_evidence.py
conda run -n Benchmark python scripts/phase2/audit_phase2.py
```

Build the current Goal 6 overview figure and source tables:

```bash
conda run -n Benchmark python scripts/phase2/run_goal6_figure_assembly.py
```

Run Goal 1 mechanism-specific weak-signal diagnostic:

```bash
conda run -n Benchmark python scripts/phase2/run_goal1_mechanism_weak_signal_summary.py
conda run -n Benchmark python scripts/phase2/run_goal1_mechanism_threshold_tradeoff.py \
  --candidates outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_datasets/human_lymph_node_visium_replicate_00/candidate_interactions.csv \
  --prediction outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_initial/wavelet_mechanism_niche_edge_prior_ranking/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions.csv \
  --label edge_prior_ranking \
  --output-dir outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_mechanism_threshold_tradeoff \
  --target-mechanism long_range_chemokine_niche \
  --target-mechanism short_medium_paracrine
```

Run high-resolution tonsil subset scaling:

```bash
conda run -n Benchmark python scripts/phase2/run_goal4_high_resolution_scaling.py
```

Run same-subset tonsil baseline comparison:

```bash
conda run -n Benchmark python scripts/phase2/run_goal4_tonsil_baseline_matched.py \
  --method spatialself_wavelet_calibrated \
  --method commot \
  --method cellchat \
  --method spatalk \
  --output-dir outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset
```

Run tonsil CCC module diagnostic:

```bash
conda run -n Benchmark python scripts/phase2/run_goal4_tonsil_ccc_modules.py
```

Run Goal 5 mechanism/category interpretability:

```bash
conda run -n Benchmark python scripts/phase2/run_goal5_mechanism_category_interpretability.py
```

Run Goal 5 interpretability synthesis:

```bash
conda run -n Benchmark python scripts/phase2/run_goal5_lymph_node_interpretability.py
conda run -n Benchmark python scripts/phase2/run_goal5_interpretability_synthesis.py
```

Run Goal 6 figure QA:

```bash
conda run -n Benchmark python scripts/phase2/run_goal6_figure_qa.py
```
