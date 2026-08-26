# ScaleComm

ScaleComm is a graph-wavelet framework for ligand-receptor-specific spatial scale and context modeling in spatial cell-cell communication inference. The current codebase contains benchmark generators, baseline adapters, real-data analysis scripts, manuscript figure builders, and the Genome Biology draft workspace.

The historical Python package path remains `src/spitialself/` for compatibility with existing scripts. New manuscript text and figures use the method name ScaleComm.

## Interactive Result Browser

The repository includes a standalone ScaleComm Explorer for the SpaMosaic tonsil A1 `CXCL13-CXCR5` manuscript case study. It is a read-only interactive view of the same result tables used by revised Figure 6 and supports LR selection, cell-type-pair filtering, event inspection, and PNG/CSV export. Launch it from the repository root:

```bash
python3 -m http.server 8502 --bind 0.0.0.0 --directory apps/scalecomm_explorer
```

Open `http://localhost:8502/scalecomm_explorer_tonsil_cxcl13_cxcr5.html`. See [the Explorer README](apps/scalecomm_explorer/README.md) for provenance, controls, exports, and screenshots.

## Current Structure

```text
src/spitialself/                  Shared Python modules
  benchmarks/                     Benchmark data generators and schemas
  models.py                       GraphSAGE/GATv2 LR scoring models
  metrics.py                      Confusion, threshold, and metric helpers

visium_lr_benchmark/              Human lymph node Visium artifacts
high_spatial/                     Human tonsil dataset scripts and artifacts
mouse/                            Mouse hippocampus dataset scripts and artifacts
HumanMelannomaMultiome/           Human melanoma multiome scripts and artifacts
lung_cancer/                      Lung cancer Visium scripts and artifacts
downstreamgene_eva/               LR -> TF -> target-gene downstream analysis

train_visium.py                   Visium self-supervised training entry point
pred_train.py                     Visium full-graph LR inference
spilt_zone.py                     Lymph node region annotation from cell2location
T_zone_*.py, CCR7-CCL19_*.py      T-zone / CCR7-CCL19 case-study plots
docs/PROJECT_STRUCTURE.md         More detailed structure notes
docs/PHASE1_GOALS.md              First-phase experiment plan
docs/PHASE2_GOALS.md              Second-phase model-improvement plan
configs/experiments/              Phase 1 experiment configs
configs/baselines/                Baseline fairness and method policy
configs/datasets/                 Local dataset root registry
scripts/phase1/                   Phase 1 listing/runner entrypoints
scripts/phase2/                   Phase 2 model-improvement/evidence entrypoints
scripts/benchmark/                Benchmark generation, validation, and baseline smoke tests
```

## Main Pipelines

### Phase 1 Benchmark Work

Old benchmark directories were deleted. The replacement benchmarks are generated under `outputs/phase1/` from auditable simulators.

Controlled ground-truth CCC benchmark:

```bash
conda run -n Benchmark python scripts/benchmark/generate_controlled_ccc.py
conda run -n Benchmark python scripts/benchmark/validate_controlled_ccc.py outputs/phase1/exp01_controlled_benchmark/datasets
conda run -n Benchmark python scripts/benchmark/run_spatialself.py outputs/phase1/exp01_controlled_benchmark/datasets --model-name spatialself_wavelet
```

Real-background semi-synthetic benchmark:

```bash
conda run -n Benchmark python scripts/benchmark/generate_semisynthetic_real_background.py
conda run -n Benchmark python scripts/benchmark/validate_semisynthetic_real_background.py outputs/phase1/exp02_semisynthetic_real_background/datasets
conda run -n Benchmark python scripts/benchmark/run_spatialself.py outputs/phase1/exp02_semisynthetic_real_background/datasets --model-name spatialself_wavelet
```

The semi-synthetic generator exports a broad real-expression background by default (`n_background_genes: 2000`, mixed mean/variance/random gene selection). This is needed for downstream target validation; a tiny expression panel can make NicheNet/Reactome target signatures unverifiable even when the LR signal injection is correct.

The current formal semi-synthetic Phase 1 config keeps a manageable `n_spots: 1200` run for baseline/FDR workflows. A stronger full-slide sparse semi-synthetic human lymph node Visium dataset is also available at `outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_datasets/`, using all 4,035 spots with real coordinates, expression sparsity, library size, and cell-type composition. It writes 332,328 candidate interactions, 91,721 positives, 49,538 weak positives, and zero positive labels for high-expression false-positive LR pairs. The realism report at `outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_realism/` compares the output against the source h5ad for the same spots/genes: selected-library-size Pearson 0.999998, cell-sparsity Pearson 0.999991, background gene mean/sparsity Pearson 1.000/1.000, and background Moran's I Pearson 1.000, while injected LR genes show median mean shift 1.949.

For small local smoke tests:

```bash
conda run -n Benchmark python scripts/phase1/run_experiment.py exp01_controlled_benchmark --quick-smoke
conda run -n Benchmark python scripts/phase1/run_experiment.py exp02_semisynthetic_real_background --quick-smoke
conda run -n Benchmark python scripts/phase1/run_experiment.py exp03_baseline_comparison --quick-smoke
conda run -n Benchmark python scripts/phase1/run_experiment.py exp04_model_ablation --quick-smoke
conda run -n Benchmark python scripts/phase1/run_experiment.py exp05_weak_signal_false_positive --quick-smoke
conda run -n Benchmark python scripts/phase1/run_experiment.py exp06_permutation_fdr --quick-smoke
conda run -n Benchmark python scripts/phase1/run_experiment.py exp07_lymph_node_global_validation --quick-smoke
conda run -n Benchmark python scripts/phase1/run_experiment.py exp08_runtime_memory --quick-smoke
```

`scripts/phase1/run_experiment.py` now has executable entries for all eight Phase 1 experiments. The quick-smoke entries reuse existing smoke datasets when present and generate them when absent. They are meant to validate schema, outputs, and environment wiring; main manuscript runs should increase epochs, permutations, replicate counts, and baseline coverage.

Baseline environment smoke test:

```bash
conda run -n Benchmark python scripts/benchmark/smoke_test_baselines.py
```

Phase 1 audit:

```bash
conda run -n Benchmark python scripts/phase1/audit_phase1.py
```

The audit writes `outputs/phase1/phase1_audit/phase1_audit_summary.csv` and `.json`, checking runner/config presence, key output artifacts, retained baseline status, removed-baseline residue, and known remaining gaps.

Phase 1 completion gate:

```bash
conda run -n Benchmark python scripts/phase1/check_phase1_completion.py
```

The completion gate writes `outputs/phase1/phase1_completion_gate/phase1_completion_gate.{csv,json,md}` and verifies the eight user-defined Phase 1 goals against concrete evidence. The current gate has 28/28 passing checks.

Phase 1 evidence matrix:

```bash
conda run -n Benchmark python scripts/phase1/summarize_phase1_evidence.py
```

This writes `outputs/phase1/phase1_evidence_matrix/phase1_evidence_matrix.{csv,json,md}` with one row per Phase 1 goal, the primary evidence artifacts, key numbers, and caveats.

### Phase 2 Model Improvement

Phase 2 starts from the completed Phase 1 benchmark framework and focuses on weak-signal recovery, FDR calibration, model selection, high-resolution validation, interpretability, and paper-ready figures.

Current Goal 1 weak-signal recovery run:

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

The initial five-replicate controlled result is saved at `outputs/phase2/goal1_weak_signal_recovery/controlled_five_replicate_initial/weak_signal_recovery_summary.csv`. The current best controlled weak-signal candidate is `phase2_focal_weak_mining_edge_prior_wavelet`, with mean weak recall 0.170, AUPR 0.239, FDR 0.823, and high-expression false-positive rate 0.024. The simpler `wavelet_mechanism_niche_edge_prior_ranking` also passes the controlled Goal 1 criterion, with mean weak recall 0.164, AUPR 0.234, FDR 0.825, and lower high-expression false-positive rate 0.007.

Full-slide semi-synthetic validation is saved at `outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_initial/weak_signal_recovery_summary.csv`. The focal weak-mining candidate does not transfer to the realistic-background benchmark: weak recall drops to 0.0126 and FDR rises to 0.604. The stable candidate is `wavelet_mechanism_niche_edge_prior_ranking`, which reaches weak recall 0.0398 at the default 0.95 threshold. A threshold tradeoff diagnostic at `outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_threshold_tradeoff/threshold_tradeoff_summary.csv` finds a viable global operating point for the same model at quantile 0.94: weak recall 0.0508, FDR 0.463, FDR delta versus q0.95 of 0.012, and high-expression false-positive rate 0.00032. A stronger post-hoc LR-pair-balanced diagnostic at `outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_lr_balanced_threshold/lr_balanced_threshold_summary.csv` uses LR-wise quantile 0.90 plus global score floor 0.90 and reaches weak recall 0.0871, FDR 0.455, FDR delta 0.0036, and high-expression false-positive rate 0.00305 on the full-slide replicate. However, controlled cross-replicate stability at `outputs/phase2/goal1_weak_signal_recovery/controlled_lr_balanced_threshold_stability/lr_balanced_replicate_aggregate.csv` fails the same constraints: pass fraction 0/5, mean FDR delta 0.037, and mean high-expression FP rate 0.028. Therefore LR-balanced thresholding is retained as a full-slide diagnostic only; the safer current Goal 3 operating point remains global q0.94.

Goal 1 now also has a post-hoc mechanism-specific weak-signal diagnostic:

```bash
conda run -n Benchmark python scripts/phase2/run_goal1_mechanism_weak_signal_summary.py
```

This summarizes existing predictions by contact, short/medium paracrine, and long-range chemokine-like mechanisms under `outputs/phase2/goal1_weak_signal_recovery/goal1_mechanism_weak_signal_combined_summary.csv`. In the controlled five-replicate benchmark, edge-prior ranking recovers contact and short/medium weak positives better than long-range weak positives at q0.95: weak recall is 0.461 for contact, 0.383 for short/medium paracrine, and 0.040 for long-range chemokine-like signals. Focal weak mining improves controlled short/medium recall to 0.552 but reduces contact to 0.291 and long-range to 0.007, and it does not transfer to full-slide realistic background. In the full-slide sparse benchmark, weak positives are long-range only, so the q0.94 operating-point gain is specifically long-range weak recall from 0.0398 to 0.0508. Treat this as a diagnostic of the current bottleneck, not as a new model improvement.

Two follow-up Goal 1 diagnostics are now recorded. `outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_mechanism_targeted_probe/weak_signal_recovery_summary.csv` tests long-range/non-contact weak-positive mining variants on full-slide sparse data; both fail to transfer, with long-range soft weak recall 0.0031 and non-contact soft weak recall 0.0078. `outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_mechanism_threshold_tradeoff/mechanism_threshold_tradeoff_summary.csv` tests a post-hoc mechanism-specific operating point on the existing edge-prior predictions. Long-range q0.92 gives weak recall 0.0417 with FDR delta 0.0027 and unchanged high-expression false-positive rate, while the combined mechanism-specific rule gives weak recall 0.0417 and FDR 0.450. This is a small diagnostic gain, weaker than the global q0.94 and LR-balanced operating points, so it should not replace the current Phase 2 default.

Phase 2 evidence and audit:

```bash
conda run -n Benchmark python scripts/phase2/summarize_phase2_evidence.py
conda run -n Benchmark python scripts/phase2/audit_phase2.py
```

Goal 2 FDR calibration smoke on the current full-slide edge-prior survivor:

```bash
conda run -n Benchmark python scripts/phase2/run_goal2_fdr_calibration.py \
  --candidates outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_datasets/human_lymph_node_visium_replicate_00/candidate_interactions.csv \
  --predictions outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_initial/wavelet_mechanism_niche_edge_prior_ranking/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions.csv \
  --output-dir outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_calibration_initial \
  --n-permutations 20 \
  --operating-quantile 0.94
```

This Goal 2 smoke run covers score, spatial, receiver, LR-identity, within-region, and cell-type-preserving nulls, plus candidate-edge, LR-pair, cell-type-pair, and pathway/module summaries. It is a schema and diagnostic run, not a final significance analysis: with 20 permutations and 332,328 candidate tests, candidate-edge BH remains conservative by construction.
The same runner also writes direct top-tail group permutation tests. In the current smoke, direct LR-pair testing improves positive-control BH-FDR to about 0.119 without significant high-expression negative-control groups, so the next Goal 2 step is a higher-permutation group-level calibration rather than more candidate-edge BH.

Focused direct group-level FDR grid:

```bash
conda run -n Benchmark python scripts/phase2/run_goal2_direct_group_grid.py \
  --candidates outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_datasets/human_lymph_node_visium_replicate_00/candidate_interactions.csv \
  --predictions outputs/phase2/goal1_weak_signal_recovery/full_slide_sparse_initial/wavelet_mechanism_niche_edge_prior_ranking/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions.csv \
  --output-dir outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_grid_500perm_focused \
  --n-permutations 500 \
  --top-fraction 0.05 \
  --top-fraction 0.10 \
  --null-model score_permutation \
  --null-model lr_identity_permutation \
  --null-model receiver_permutation \
  --group-level lr_pair \
  --group-level pathway_module
```

The 500-permutation focused check finds significant positive-control LR-pair/module discoveries under score-permutation and LR-identity-permutation nulls with no significant high-expression negative-control groups. Receiver permutation remains a stricter low-power null and is reported separately.

Selected Goal 2 seed-stability check:

```bash
conda run -n Benchmark python scripts/phase2/run_goal2_direct_group_seed_stability.py
```

This reruns the selected direct group-level setting, `lr_identity_permutation` at LR-pair level with top 10% group statistic, across 5 random seeds with 200 permutations per seed. The output is saved under `outputs/phase2/goal2_fdr_calibration/full_slide_edge_prior_direct_group_seed_stability/`. Current result is seed-stable on this single full-slide replicate: 5/5 seeds pass, all 3 positive-control LR groups are significant in every seed, 0 high-expression negative-control groups are significant, and the negative-control minimum BH-FDR remains 1.0. Treat this as seed-stable single-replicate support, not final manuscript calibration.

Selected Goal 2 controlled replicate-stability check:

```bash
conda run -n Benchmark python scripts/phase2/run_goal2_direct_group_replicate_stability.py
```

This reruns the same selected direct group-level setting across the 5 controlled ground-truth benchmark replicates. The output is saved under `outputs/phase2/goal2_fdr_calibration/controlled_edge_prior_direct_group_replicate_stability/`. Current result: 5/5 controlled replicates pass, positive-control significant LR groups range from 4 to 5, high-expression negative-control significant groups remain 0, and the negative-control minimum BH-FDR remains 1.0. Treat this as controlled cross-replicate support only; the full-slide realistic-background FDR evidence is still single-replicate.

Goal 3 model selection:

```bash
conda run -n Benchmark python scripts/phase2/run_goal3_model_selection.py \
  --output-dir outputs/phase2/goal3_model_selection
```

The current Phase 2 default candidate is `wavelet_calibrated_fdr`: edge-prior wavelet ranking, global q0.94 calibrated operating point, and direct group-level FDR. The current model-selection score is 10 after adding selected-FDR seed-stability and controlled replicate-stability evidence. It is selected because it combines controlled weak-recall improvement, full-slide weak-signal recovery, low high-expression false-positive rate, significant LR-pair positive controls under direct group-level FDR, seed-stable selected FDR behavior on one full-slide replicate, controlled cross-replicate FDR support, and lymph-node downstream/control validation. Treat this as the default for Goals 4-5, not yet as a final manuscript default. LR-balanced thresholding remains a full-slide post-hoc diagnostic because it fails the controlled cross-replicate FDR/FP constraints.

Goal 4 high-resolution tonsil bridge:

```bash
conda run -n Benchmark python scripts/phase2/run_goal4_high_resolution_tonsil_subset.py \
  --output-dir outputs/phase2/goal4_high_resolution/tonsil_existing_prediction_subset \
  --chunksize 1000000 \
  --top-edges 50000 \
  --chunk-top-n 20000
```

This streams the existing human tonsil prediction table rather than retraining the Phase 2 default. The current artifact covers 5,778 cells, 13 cell types, 924 LR pairs, and 58,717,428 candidate prediction rows in about 28.5 seconds with about 581 MB peak RSS. It validates node order against the expression header and graph artifact, writes LR/cell-type-pair/positive-control summaries, and should be treated as Goal 4 wiring plus scalability evidence. The biological positive-control signal is mixed, so a fresh `wavelet_calibrated_fdr` high-resolution inference plus downstream/pathway validation is still required before manuscript claims.

Fresh Goal 4 tonsil subset inference:

```bash
conda run -n Benchmark python scripts/phase2/run_goal4_fresh_tonsil_subset.py \
  --output-dir outputs/phase2/goal4_high_resolution/tonsil_fresh_wavelet_calibrated_subset \
  --n-cells 720 \
  --n-background-lr 36 \
  --epochs 6 \
  --max-train-samples 8000
```

This constructs a benchmark-style real-data replicate from the local tonsil data and trains the current Phase 2 default candidate on the subset. The current run covers 720 cells, 45 LR pairs, and 406,728 candidate edges in about 12.8 seconds wall-clock with about 794 MB peak RSS. It is not a ground-truth benchmark and does not report AUROC/AUPR. The initial biological-control result is useful but still preliminary: `CXCL13-CXCR5` ranks 4/45 and no housekeeping high-expression negative control reaches the top decile. Larger high-resolution tiles, downstream validation, and retained-baseline comparisons are still needed for manuscript claims.

Fresh Goal 4 tonsil subset scaling:

```bash
conda run -n Benchmark python scripts/phase2/run_goal4_high_resolution_scaling.py
```

This repeats the fresh Phase 2 default tonsil workflow at 720, 1,200, and 1,800 cells and writes outputs under `outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/`. The largest current subset has 1,800 cells, 45 LR pairs, and 1,641,363 candidate edges, with 34.4 seconds wall-clock and about 1.6 GB peak RSS. Positive controls retain one top-decile LR and housekeeping high-expression negatives remain outside the top decile. This is subset-level scalability evidence only; it is not a full Slide-tags, Xenium, or Visium HD run.

Fresh Goal 4 tonsil same-subset baseline comparison:

```bash
conda run -n Benchmark python scripts/phase2/run_goal4_tonsil_baseline_matched.py \
  --method spatialself_wavelet_calibrated \
  --method commot \
  --method cellchat \
  --method spatalk \
  --output-dir outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset
```

This compares retained methods on the same 720-cell fresh tonsil subset using biological positive/negative control ranking rather than AUROC/AUPR, because the subset is not a ground-truth benchmark. Current initial result: SpatialSelf has 1 positive and 0 housekeeping negative controls in the top decile and passes the control-rank criterion; COMMOT and CellChat have valid nonzero scores but do not pass; SpaTalk returns flat all-zero scores in this setting and is recorded as not claim-eligible. This is useful same-subset validation support, not final baseline superiority evidence.

Fresh Goal 4 tonsil CCC module diagnostic:

```bash
conda run -n Benchmark python scripts/phase2/run_goal4_tonsil_ccc_modules.py
```

This clusters the fresh 720-cell tonsil subset LR activity profiles into CCC modules and writes results under `outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_fresh_subset/`. The current diagnostic finds 4 modules across 45 LR pairs. Module 3 is a long-range chemokine positive-control module with `CXCL13-CXCR5`, `CXCL12-CXCR4`, and `CCL21-CCR7`, 3 positive controls, 0 housekeeping negatives, and positive-control enrichment BH-FDR 0.0056. Module 4 is a high-expression negative-control module with `ACTB-EGFR`, `ACTB-GAPDH`, and `GAPDH-ITGB1`, 3 housekeeping negatives, 0 positives, and housekeeping-negative enrichment BH-FDR 0.00042. Treat this as module/control separation on one subset; spatial concentration values are weak and this is not full high-resolution validation or real-data ground truth.

Fresh Goal 4 tonsil n1800 CCC module stress test:

```bash
conda run -n Benchmark python scripts/phase2/run_goal4_tonsil_ccc_modules.py \
  --goal4-dir outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/n1800 \
  --output-dir outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_n1800_subset
```

This repeats the CCC module diagnostic on the 1,800-cell tonsil subset. The result weakens the clean 720-cell module interpretation: the significant housekeeping-negative module contains 3 housekeeping negatives but also `CXCL13-CXCR5` and `CCL21-CCR7`, positive-control enrichment is not significant (best BH-FDR 0.17-0.32), and housekeeping-negative enrichment remains significant (BH-FDR 0.0042). Treat this as a larger-subset stress test showing that positive/negative module separation is not yet stable.

Goal 5 tonsil downstream interpretability:

```bash
conda run -n Benchmark python scripts/phase2/run_goal5_tonsil_interpretability.py \
  --output-dir outputs/phase2/goal5_interpretability/tonsil_fresh_subset \
  --n-permutations 200
```

This validates the fresh tonsil subset against receiver-side target signatures read from the original tonsil expression matrix. It writes raw Spearman, permutation p-values, BH-FDR, receiver-cell-type residualized Spearman, pathway summaries, and CCC activity modules. Current evidence is mixed: positive controls have higher residual downstream correlation than housekeeping negatives, with 4 positive and 0 housekeeping negative controls significant at residual BH-FDR <= 0.10, but broad background immune LR still show stronger downstream correlations. Treat this as an initial interpretability diagnostic, not a final biological claim.

Goal 5 larger tonsil downstream stress test:

```bash
conda run -n Benchmark python scripts/phase2/run_goal5_tonsil_interpretability.py \
  --goal4-dir outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/n1800 \
  --output-dir outputs/phase2/goal5_interpretability/tonsil_n1800_subset \
  --n-permutations 200
```

This reruns the same downstream validation on the 1,800-cell tonsil subset from Goal 4 scaling. Current result is mixed/negative for biological interpretability: positive-control residual mean Spearman is -0.016, housekeeping negative residual mean is 0.009, background LR residual mean is 0.049, and no positive controls are significant at residual BH-FDR <= 0.10. This stress test argues against overclaiming from the 720-cell subset and points to the need for stronger receiver/domain-specific ligand-target priors.

Goal 5 LR embedding interpretability:

```bash
conda run -n Benchmark python scripts/phase2/run_goal5_lr_embedding_interpretability.py \
  --output-dir outputs/phase2/goal5_interpretability/lr_embedding_tonsil_fresh_subset
```

This extracts learned LR identity embeddings from the fresh tonsil subset model and links them to LR categories, model scores, downstream residual correlations, PCA/UMAP coordinates, cluster summaries, and cluster enrichment tests. Current evidence is weak: mechanism silhouette is -0.081, control-class silhouette is -0.052, and no cluster enrichment is significant at BH-FDR <= 0.10. Record this as a negative/mixed interpretability diagnostic, not as support for a biological-mechanism claim.

Goal 5 mechanism/category interpretability:

```bash
conda run -n Benchmark python scripts/phase2/run_goal5_mechanism_category_interpretability.py
```

This summarizes downstream correlation, spatial concentration, and LR embedding diagnostics by mechanism and LR functional category. Current result remains mixed: positive controls exceed housekeeping negatives after residualization, with 4 positive and 0 housekeeping negatives significant at residual BH-FDR <= 0.10, but the strongest group-level enrichment is driven by broad background/short-medium paracrine LR rather than clean positive-control categories. Treat this as traceable mechanism-level diagnostic evidence, not proof that LR embeddings recover biological mechanisms.

Goal 5 interpretability synthesis:

```bash
conda run -n Benchmark python scripts/phase2/run_goal5_lymph_node_interpretability.py
conda run -n Benchmark python scripts/phase2/run_goal5_interpretability_synthesis.py
```

The lymph-node interpretability runner summarizes the full-slide sparse human lymph-node real-background injected-control validation into `outputs/phase2/goal5_interpretability/lymph_node_real_background/`. Current result supports all three injected positive controls by downstream response, including the weak `CXCL13-CXCR5` control, and keeps high-expression negatives outside the top three ranks. This is useful cross-real-background support, but still injected-control evidence rather than unlabeled real-data ground truth.

The synthesis runner combines the 720-cell downstream validation, 1,800-cell stress test, CCC module diagnostics, LR embedding diagnostics, mechanism/category summaries, and lymph-node real-background injected-control evidence into `outputs/phase2/goal5_interpretability/synthesis/`. Current synthesis level remains `mixed_initial_support_not_manuscript_ready`: the lymph-node result strengthens injected/control validation, but background LR remain strong in tonsil, LR embedding clustering is weak, and the 1,800-cell tonsil stress test does not preserve positive-control downstream advantage.

Goal 6 overview figure/source-table assembly:

```bash
conda run -n Benchmark python scripts/phase2/run_goal6_lymph_node_validation.py
conda run -n Benchmark python scripts/phase2/run_goal6_lymph_node_baseline_validation.py
conda run -n Benchmark python scripts/phase2/run_goal6_figure_assembly.py
conda run -n Benchmark python scripts/phase2/run_goal6_figure_qa.py
```

This writes reproducible source tables and a ten-panel Phase 2 overview figure under `outputs/phase2/goal6_paper_figures/`. The current figure summarizes retained baseline evidence, component ablation, controlled weak-signal recall, full-slide operating-point tradeoff, direct group-level FDR, model selection, tonsil subset controls, interpretability diagnostics, lymph-node real-background injected controls, and baseline-matched lymph-node controls. Source tables now also include the LR-balanced full-slide operating diagnostic, its controlled-stability failure, and the Goal 5 lymph-node interpretability source. Treat this as an initial figure-source artifact; it is not yet the final paper figure suite because high-resolution evidence remains subset-level and Goal 5 interpretability is still mixed.

The same commands also export independent manuscript-style PNG/PDF panel pairs under `outputs/phase2/goal6_paper_figures/manuscript_panels/`. Each panel is backed by one CSV in `outputs/phase2/goal6_paper_figures/source_tables/`, including `goal1_mechanism_weak_signal.csv` for the mechanism-specific weak-signal diagnostic, `goal2_direct_group_replicate_stability.csv` for controlled cross-replicate FDR support, `goal4_high_resolution_scaling.csv` for the 720/1,200/1,800-cell tonsil subset scaling panel, `goal4_tonsil_baseline_matched.csv` for the same-subset tonsil control-ranking comparison, `goal4_tonsil_ccc_modules.csv` for the tonsil CCC module/control separation panel, `goal5_interpretability_n1800.csv` for the larger-subset downstream stress test, `goal5_mechanism_category.csv` for mechanism/category interpretability diagnostics, `goal5_lymph_node_interpretability.csv` for lymph-node real-background injected-control downstream/rank support, and `goal5_interpretability_synthesis.csv` for the explicit mixed/not-manuscript-ready Goal 5 claim constraint. Baseline source tables and the claim traceability table explicitly mark evidence levels, so adapter-light, projection/status-only, and single-replicate rows are not over-interpreted as final baseline superiority evidence. The component ablation panel is included as Phase 1 controlled component-level evidence, not as the final Phase 2 default-model ablation. The lymph-node panels are regenerated by Phase 2 runners as full-slide sparse real-background injected-control evidence, not unlabeled real-data ground truth.

The figure QA runner writes `outputs/phase2/goal6_paper_figures/figure_qa/`. Current automated QA passes file/source traceability checks: 16 PNG files, 16 PDF files, 30 manuscript-panel files, 24 non-empty source tables, and no missing paired exports. It also records the remaining manuscript-packaging risk: the current bundle is PNG/PDF-only, so final submission still needs manual visual review plus SVG/TIFF export and journal-size checks.

Candidate-level baseline comparison smoke test:

```bash
conda run -n Benchmark python scripts/benchmark/run_baseline_comparison.py \
  outputs/phase1/exp02_semisynthetic_real_background/smoke_datasets \
  --output-dir outputs/phase1/exp03_baseline_comparison/smoke_semisynthetic/combined \
  --method spatialself_wavelet_scale \
  --method spatialself_gat \
  --method commot \
  --method cellchat \
  --method spatalk \
  --method cellnest \
  --method deeptalk \
  --method niches
```

The current formal SpatialSelf baseline is `spatialself_wavelet_scale` (`SpatialSelf-scale`). It keeps the graph-wavelet pyramid, but scores each candidate edge through LR-specific scale-mixture logits, so every LR pair has an inspectable learned spatial-scale distribution. The default benchmark runner uses the mechanism/niche/expression/distance pseudo-label strategy, LR-balanced sampling, edge-prior features, and scale-target/diversity/responsibility losses.

Implemented candidate-level adapters currently include SpatialSelf-scale, SpatialSelf-wavelet, SpatialSelf-GAT, COMMOT, CellChat, SpaTalk, CellNEST, and DeepTalk projection/status. CellChat and SpaTalk are run at cell-type-pair level and then projected back to the benchmark candidate-edge schema so they share the same AUROC/AUPR/FDR evaluation path. SpaTalk also requires receptor-to-target pathway support; synthetic benchmark smoke tests can add minimal receptor-target pathway rows and must report that setting separately from native SpaTalk.

Historical Phase 1 controlled core runs covered 5 controlled replicates with SpatialSelf-wavelet, SpatialSelf-GAT, COMMOT, DeepTalk projection/status, NicheNet downstream-only status, and NICHES optional blocked status. These outputs are retained for traceability, but new benchmark and real-data reruns should use `spatialself_wavelet_scale` as the SpatialSelf baseline.

The first SpatialSelf weak-signal tradeoff check is saved at `outputs/phase1/exp04_model_ablation/full_spatialself_tradeoff_summary.csv`. The `wavelet_lr_capped_q80_weak_reweighted` ablation improves weak-signal recall but worsens AUPR/FDR, so it is an explicit ablation point rather than the new default.

The earlier LR-prior hard-negative ablation is summarized at `outputs/phase1/exp04_model_ablation/full_spatialself_strategy_summary.csv`. It uses curated LR-pair prior/control information to suppress false-positive LR pairs and improves AUROC/AUPR/FDR relative to the first SpatialSelf-wavelet setting, but weak-signal recall remains well below COMMOT.

The current fair focused ablation, saved at `outputs/phase1/exp04_model_ablation/full_fair_focused_ablation/`, removes weak-label leakage during training and compares mechanism-aware pseudo-labeling, leakage-safe weak augmentation, and LR-wise ranking across 5 controlled replicates. The strongest fair variant so far is `wavelet_mechanism_niche_lr_prior_lrwise_ranking` with mean AUROC 0.928, AUPR 0.207, weak-signal recall 0.118, FDR 0.821, and high-expression false-positive call rate 0.015.

The score-calibration diagnostic at `outputs/phase1/exp04_model_ablation/full_fair_score_calibration_nonoracle/` shows that non-oracle post-processing is not enough: raw scores remain the best fair setting among tested transforms, while LR-wise rank and non-oracle mechanism/distance rank reduce AUPR or worsen FDR. The rule-prior transform is strong but oracle-style because it uses benchmark rule-satisfied columns, so it is only an upper-bound diagnostic.

The next model candidate is the edge-prior ranking variant in `outputs/phase1/exp04_model_ablation/full_edge_prior_focused_ablation_epoch10/`. It gives the edge scorer observable expression/mechanism/niche prior features and improves mean AUPR and weak-signal recall over the current fair default with similar FDR in a 5-replicate controlled check. Treat it as a candidate for the next baseline/FDR/real-data rerun, not as a finalized default yet.

`spatialself_wavelet_edge_prior` is now available as a separate method name in the baseline and stress runners. The Phase 1 baseline/stress/FDR smoke commands exercise it without replacing `spatialself_wavelet`, so old and candidate SpatialSelf variants can be compared side by side.

The full controlled edge-prior core comparison is saved at `outputs/phase1/exp03_baseline_comparison/full_controlled_edge_prior_core/`. In that 5-replicate run, edge-prior improves SpatialSelf AUPR and weak-signal recall with similar FDR, but COMMOT still has substantially higher weak-signal recall. The full controlled FDR panel is saved at `outputs/phase1/exp06_permutation_fdr/full_controlled_spatialself_panel/`; it covers default wavelet and edge-prior wavelet across 5 replicates and three null models, and remains conservative with no BH-significant edges/groups under the current score-permutation null.

The retained baseline framework now also runs on the full-slide sparse semi-synthetic lymph-node benchmark at `outputs/phase1/exp03_baseline_comparison/full_slide_sparse_semisynthetic_retained/`. On 4,035 spots and 332,328 candidate interactions, COMMOT has the strongest AUROC/AUPR (0.717/0.480), SpatialSelf-wavelet reaches 0.609/0.395, edge-prior improves weak-signal recall over default (0.0398 vs 0.0008) but increases FDR, and CellNEST reaches weak recall 1.000 with high FDR 0.724. DeepTalk, NicheNet, and NICHES statuses are recorded without treating them as missing runnable candidate-edge scorers.

The lymph-node global validation now has Phase 2 reproducible runners at `scripts/phase2/run_goal6_lymph_node_validation.py` and `scripts/phase2/run_goal6_lymph_node_baseline_validation.py`. On the 4,035-spot full-slide sparse human lymph-node real-background injected-control dataset, `spatialself_wavelet_edge_prior` ranks the 3 injected positive controls ahead of 2 high-expression negative controls, reports mean positive downstream Spearman 0.752, and has BH-FDR 0.0099 for the three tested positive LR controls. The baseline-matched validation compares 7 retained candidate-edge/projection methods under the same injected-control metrics: SpatialSelf edge-prior/default/GAT and SpaTalk pass the rank-control criterion, while COMMOT, CellChat, and CellNEST rank at least one high-expression negative in the top three or fail positive-before-negative ordering. This is still single-replicate injected-control validation, not unlabeled real-data ground truth. The older Phase 1 exp07 outputs remain retained for traceability.

Phase 1 baseline and weak-signal stress runners now expose SpatialSelf ranking/weak-positive parameters. The phase runner defaults SpatialSelf to the current strongest leakage-safe setting: `lr_prior_lr_capped_mechanism_niche_expression_distance` with LR-wise ranking loss weight 0.35 and margin 0.5.

The earlier single-replicate FDR pilot for the LR-prior hard-negative SpatialSelf prediction is saved at `outputs/phase1/exp06_permutation_fdr/full_controlled_replicate00_spatialself_lr_prior/`. The current stronger controlled evidence is the 5-replicate panel at `outputs/phase1/exp06_permutation_fdr/full_controlled_spatialself_panel/`. A realistic-background FDR panel is also saved at `outputs/phase1/exp06_permutation_fdr/full_slide_sparse_semisynthetic_panel/`, covering SpatialSelf-wavelet, edge-prior wavelet, and COMMOT on the 4,035-spot full-slide sparse lymph-node benchmark with 332,328 candidate interactions. Both panels confirm the permutation/FDR schema runs and that the present score-permutation null is conservative; power improvement should target calibration or group/model-rerun nulls.

Standalone SpatialSelf smoke test:

```bash
conda run -n Benchmark python scripts/benchmark/run_spatialself.py \
  outputs/phase1/exp01_controlled_benchmark/smoke_datasets \
  --output-dir outputs/phase1/exp03_baseline_comparison/smoke_controlled/spatialself_wavelet \
  --model-name spatialself_wavelet \
  --epochs 5
```

SpatialSelf pseudo-label ablation smoke test:

```bash
conda run -n Benchmark python scripts/benchmark/run_spatialself_ablation.py \
  outputs/phase1/exp01_controlled_benchmark/smoke_datasets \
  --output-dir outputs/phase1/exp04_model_ablation/smoke_controlled/pseudo_label_check \
  --epochs 5 \
  --max-train-samples 4000 \
  --ablation wavelet_hard_expression_distance \
  --ablation wavelet_soft_expression_distance
```

SpatialSelf component ablation smoke test:

```bash
conda run -n Benchmark python scripts/benchmark/run_spatialself_ablation.py \
  outputs/phase1/exp01_controlled_benchmark/smoke_datasets \
  --output-dir outputs/phase1/exp04_model_ablation/smoke_controlled/component_check \
  --epochs 5 \
  --max-train-samples 4000 \
  --ablation wavelet_hard_expression_distance \
  --ablation wavelet_no_lr_embedding \
  --ablation wavelet_no_distance \
  --ablation wavelet_no_spatial_graph
```

Weak-signal and false-positive stress smoke test:

```bash
conda run -n Benchmark python scripts/benchmark/run_weak_signal_false_positive_dataset.py \
  outputs/phase1/exp01_controlled_benchmark/smoke_datasets \
  --output-dir outputs/phase1/exp05_weak_signal_false_positive/smoke_controlled \
  --method spatialself_wavelet \
  --method spatialself_gat \
  --spatialself-epochs 5 \
  --spatialself-max-train-samples 4000
```

This writes method-level stress metrics and threshold/top-k/LR/mechanism tables under `stress_tables/`. The current smoke results expose a real weakness: the lightweight SpatialSelf pseudo-label settings reduce high-expression false-positive calls but still miss weak signals at the default top-threshold.

LR-capped pseudo-label stress smoke:

```bash
conda run -n Benchmark python scripts/benchmark/run_weak_signal_false_positive_dataset.py \
  outputs/phase1/exp01_controlled_benchmark/smoke_datasets \
  --output-dir outputs/phase1/exp05_weak_signal_false_positive/smoke_controlled_lr_capped \
  --method spatialself_wavelet \
  --spatialself-epochs 5 \
  --spatialself-max-train-samples 4000 \
  --spatialself-pseudo-label-strategy lr_capped_expression_distance
```

Standalone CellChat smoke test:

```bash
conda run -n Benchmark python scripts/benchmark/run_cellchat.py \
  outputs/phase1/exp01_controlled_benchmark/smoke_datasets \
  --output-dir outputs/phase1/exp03_baseline_comparison/smoke_controlled/cellchat \
  --nboot 5
```

Standalone SpaTalk smoke test:

```bash
conda run -n Benchmark python scripts/benchmark/run_spatalk.py \
  outputs/phase1/exp01_controlled_benchmark/smoke_datasets \
  --output-dir outputs/phase1/exp03_baseline_comparison/smoke_controlled/spatalk \
  --per-num 20 \
  --min-pairs 1 \
  --use-synthetic-pathways
```

DeepTalk input preparation smoke test:

```bash
conda run -n Benchmark python scripts/benchmark/prepare_deeptalk_inputs.py \
  outputs/phase1/exp01_controlled_benchmark/smoke_datasets \
  --output-dir outputs/phase1/exp03_baseline_comparison/smoke_controlled/deeptalk_inputs
```

DeepTalk projection/status smoke test:

```bash
conda run -n Benchmark python scripts/benchmark/run_deeptalk.py \
  outputs/phase1/exp01_controlled_benchmark/smoke_datasets \
  --output-dir outputs/phase1/exp03_baseline_comparison/smoke_controlled/deeptalk_projection_model_required_check \
  --deeptalk-input-root outputs/phase1/exp03_baseline_comparison/smoke_controlled/deeptalk_inputs
```

CellNEST input preparation smoke test:

```bash
conda run -n Benchmark python scripts/benchmark/prepare_cellnest_inputs.py \
  outputs/phase1/exp01_controlled_benchmark/smoke_datasets \
  --output-dir outputs/phase1/exp03_baseline_comparison/smoke_controlled/cellnest_inputs
```

CellNEST end-to-end smoke test:

```bash
conda run -n Benchmark python scripts/benchmark/run_cellnest.py \
  outputs/phase1/exp01_controlled_benchmark/smoke_datasets \
  --output-dir outputs/phase1/exp03_baseline_comparison/smoke_controlled/cellnest_adapter_check \
  --num-epoch 2 \
  --hidden 32 \
  --top-percent 100
```

Permutation/FDR post-processing for any prediction table with `candidate_id, score`:

```bash
conda run -n Benchmark python scripts/benchmark/run_permutation_fdr.py \
  --candidates outputs/phase1/exp02_semisynthetic_real_background/smoke_datasets/human_lymph_node_visium_replicate_00/candidate_interactions.csv \
  --predictions outputs/phase1/exp03_baseline_comparison/smoke_semisynthetic/combined/spatialself_wavelet/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions.csv \
  --output-dir outputs/phase1/exp06_permutation_fdr/smoke_spatialself_wavelet_semisynthetic \
  --n-permutations 20
```

The FDR runner writes candidate-edge, LR-pair, and cell-type-pair interaction tables. Group-level p-values use Simes aggregation over candidate edges followed by BH adjustment across groups.
When benchmark ground truth is available, `permutation_fdr_summary.csv` also includes realized FDR/precision/recall, weak-signal recall, high-expression false-positive call rate, positive/negative control minimum p-values, and a p-value uniformity diagnostic. Smoke runs use small permutation counts only to validate the output schema; main results should use at least 1000 permutations.

Runtime/memory smoke profiling:

```bash
conda run -n Benchmark python scripts/benchmark/profile_runtime_memory.py \
  outputs/phase1/exp01_controlled_benchmark/smoke_datasets \
  --output-dir outputs/phase1/exp08_runtime_memory/smoke_controlled \
  --cot-nitermax 50 \
  --cellchat-nboot 5 \
  --method commot \
  --method cellchat \
  --method spatialself_wavelet \
  --method cellnest \
  --method deeptalk
```

Runtime/memory scaling smoke:

```bash
conda run -n Benchmark python scripts/benchmark/run_runtime_scaling.py \
  --quick-smoke \
  --output-dir outputs/phase1/exp08_runtime_memory/smoke_scaling_controlled \
  --spatialself-epochs 3 \
  --spatialself-max-train-samples 2000 \
  --method spatialself_wavelet \
  --method spatialself_gat
```

The profiler records wall-clock time, peak CPU RSS, GPU memory when `nvidia-smi` is available, candidate edges, LR pairs, per-epoch runtime for epoch-based methods, and method status/failure details. The default controlled scaling runner still uses the full-radius/all-pairs candidate generator and is capped at 10k cells. For larger 50k/100k stress tests, use `--candidate-generation sparse_sampled` with `--max-candidates-per-lr` and `--max-candidates-per-sender-lr`.

Sparse candidate generation smoke:

```bash
conda run -n Benchmark python scripts/benchmark/run_runtime_scaling.py \
  --output-dir outputs/phase1/exp08_runtime_memory/sparse_candidate_smoke/runtime \
  --scale 600 \
  --method spatialself_wavelet \
  --replicates 1 \
  --spatialself-epochs 1 \
  --spatialself-max-train-samples 2000 \
  --candidate-generation sparse_sampled \
  --max-candidates-per-lr 12000 \
  --max-candidates-per-sender-lr 32
```

The current full Phase 1 runtime evidence is `outputs/phase1/exp08_runtime_memory/controlled_medium_scaling/runtime_memory_scaling.csv`, covering 500 and 1000 cells with SpatialSelf-wavelet, edge-prior wavelet, SpatialSelf-GAT, and COMMOT. At 1000 cells / 926,654 candidate edges, the measured wall-clock times are about 15.73s for wavelet, 14.98s for edge-prior, 12.87s for GAT, and 75.33s for COMMOT.

The capped sparse SpatialSelf scaling run is `outputs/phase1/exp08_runtime_memory/sparse_spatialself_scaling/runtime_memory_scaling.csv`, covering 1k, 5k, and 10k cells with SpatialSelf-wavelet, edge-prior wavelet, and SpatialSelf-GAT. At 10k cells / 130,223 candidate edges, the measured wall-clock times are about 14.30s for wavelet, 15.63s for edge-prior, and 11.18s for GAT. Treat this as capped-candidate scalability evidence; final 50k/100k claims still need larger sparse or real high-resolution runs.

The SpatialSelf runner now uses KD-tree KNN graph construction instead of an all-pairs distance matrix. On the 50k capped sparse smoke dataset, the original GAT run peaked at 95,874.5 MB before the fix. After the fix, the same 50k / 130,790-edge dataset runs wavelet in 8.05s / 1,268.3 MB, edge-prior wavelet in 8.77s / 1,281.8 MB, and GAT in 10.86s / 4,433.5 MB. The fixed profiles are saved at `outputs/phase1/exp08_runtime_memory/sparse_50k_spatialself_gat_sparse_knn_profile/runtime_memory_profile.csv` and `outputs/phase1/exp08_runtime_memory/sparse_50k_spatialself_wavelet_profile/runtime_memory_profile.csv`.

The capped sparse 100k SpatialSelf smoke is `outputs/phase1/exp08_runtime_memory/sparse_100k_spatialself_smoke/runtime_memory_scaling.csv`. On 100k cells / 130,547 candidate edges, wavelet completes in 10.88s / 2,062.6 MB, edge-prior wavelet in 11.83s / 2,037.5 MB, and GAT in 12.57s / 8,386.8 MB. This is still capped-candidate evidence; higher candidate caps and real high-resolution subsets are needed before a manuscript-scale high-resolution claim.

The 50k higher-cap sparse run is `outputs/phase1/exp08_runtime_memory/sparse_50k_highcap_spatialself/runtime_memory_scaling.csv`, using `max_candidates_per_lr=50000` and `max_candidates_per_sender_lr=48`. It produces 326,963 candidate edges; wavelet completes in 10.76s / 1,353.6 MB, edge-prior wavelet in 10.12s / 1,340.0 MB, and GAT in 11.78s / 4,585.6 MB.

The retained-method full-slide sparse runtime profile is `outputs/phase1/exp08_runtime_memory/full_slide_sparse_semisynthetic_retained_profile/runtime_memory_profile.csv`. It uses the realistic-background 4,035-spot human lymph node benchmark with 332,328 candidate interactions and 5 LR pairs. Completed wall-clock / peak CPU memory values are: SpatialSelf-wavelet 20.86s / 2,537.9 MB, edge-prior wavelet 20.92s / 2,611.5 MB, SpatialSelf-GAT 10.81s / 1,070.4 MB, COMMOT 501.33s / 4,392.7 MB, CellChat 27.64s / 1,117.5 MB, SpaTalk 170.22s / 2,448.1 MB, CellNEST 78.60s / 806.4 MB, and DeepTalk projection/status 43.04s / 988.9 MB. NicheNet is recorded as pending runtime adapter because it is downstream-prior validation rather than candidate-edge scoring; NICHES remains optional blocked.

Lymph-node all-LR region enrichment smoke test:

```bash
conda run -n Benchmark python scripts/benchmark/run_region_enrichment.py \
  --candidates outputs/phase1/exp02_semisynthetic_real_background/smoke_datasets/human_lymph_node_visium_replicate_00/candidate_interactions.csv \
  --predictions outputs/phase1/exp07_lymph_node_global_validation/smoke_phase_runner/prediction_inputs/spatialself_wavelet/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions.csv \
  --cells outputs/phase1/exp02_semisynthetic_real_background/smoke_datasets/human_lymph_node_visium_replicate_00/cells.csv \
  --output-dir outputs/phase1/exp07_lymph_node_global_validation/smoke_region_enrichment_spatialself_wavelet \
  --n-permutations 20
```

Lymph-node downstream target correlation smoke test:

```bash
conda run -n Benchmark python scripts/benchmark/run_downstream_target_correlation.py \
  --candidates outputs/phase1/exp02_semisynthetic_real_background/smoke_datasets/human_lymph_node_visium_replicate_00/candidate_interactions.csv \
  --predictions outputs/phase1/exp07_lymph_node_global_validation/smoke_phase_runner/prediction_inputs/spatialself_wavelet/human_lymph_node_visium_replicate_00__spatialself_wavelet_predictions.csv \
  --expression outputs/phase1/exp02_semisynthetic_real_background/smoke_datasets/human_lymph_node_visium_replicate_00/expression.csv \
  --output-dir outputs/phase1/exp07_lymph_node_global_validation/smoke_downstream_target_spatialself_wavelet \
  --n-permutations 20
```

NicheNet ligand-target prior extraction for downstream validation:

```bash
conda run -n Benchmark python scripts/benchmark/extract_nichenet_targets.py \
  outputs/phase1/exp01_controlled_benchmark/smoke_datasets/replicate_00/lr_pairs.csv \
  --output-dir outputs/phase1/exp07_lymph_node_global_validation/smoke_nichenet_targets_controlled \
  --species human \
  --top-n 20
```

For a NicheNet-only downstream target check, pass the extracted target CSV as `--pathway-csv` and disable the Reactome/TRRUST fallback with empty values for `--reactome-tf-ppr-csv` and `--trrust-dir`.

### Real Visium self-supervised LR scoring

```bash
python train_visium.py
python pred_train.py
```

Dataset-specific graph builders are still kept inside their dataset folders, for example:

```bash
python mouse/struct_grape.py
python HumanMelannomaMultiome/struct_grape.py
python lung_cancer/struct_grape.py
```

These generate `graph_visium.pt`, `lr_table.csv`, and `S_vis.npy`, which are consumed by the training and prediction scripts.

## Refactor Status

Completed:

- Added `src/spitialself/` as the shared-code home.
- Moved reusable model definitions into `src/spitialself/models.py`.
- Moved shared metric helpers into `src/spitialself/metrics.py`.
- Removed old benchmark data/results/scripts so Phase 1 can start from a clean benchmark design.

Still legacy:

- Many dataset scripts duplicate training, prediction, and plotting logic.
- Paths and thresholds are mostly hard-coded inside scripts.
- Data and generated outputs are mixed with executable scripts.

Recommended next step: introduce YAML/JSON configs for dataset roots and convert dataset-specific scripts into thin wrappers around shared `src/spitialself` functions.

## Phase 1 Goal

The first manuscript-focused phase has eight experiments:

1. controlled ground-truth CCC benchmark
2. real-background semi-synthetic benchmark
3. baseline comparison including CellNEST and DeepTalk
4. model component and pseudo-label ablation
5. weak-signal and false-positive stress test
6. permutation p-value and FDR
7. human lymph node global validation
8. runtime and memory benchmark

See `docs/PHASE1_GOALS.md` and list the registry with:

```bash
python scripts/phase1/list_experiments.py
```
