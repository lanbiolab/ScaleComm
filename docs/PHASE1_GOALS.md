# Phase 1 Goals

Phase 1 turns the current SpatialSelf prototype into a defensible method-paper workflow. The goal is not to run every possible analysis, but to close the major review risks first.

## Core Evidence Chain

```text
realistic benchmark
  -> deep-learning baseline comparison
  -> model and pseudo-label ablation
  -> weak-signal and false-positive stress test
  -> permutation p-value and FDR
  -> global real-data validation
  -> runtime and memory benchmark
```

## The Eight Phase 1 Experiments

All eight experiments now have a unified entry point through `scripts/phase1/run_experiment.py`.
Use `--quick-smoke` for small local schema/environment checks before starting manuscript-scale runs:

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

Quick-smoke runs intentionally use reduced epochs, permutation counts, and method sets where needed. They prove wiring and output schemas, not final manuscript-scale performance.

Current evidence matrix:

```bash
conda run -n Benchmark python scripts/phase1/summarize_phase1_evidence.py
```

This writes `outputs/phase1/phase1_evidence_matrix/phase1_evidence_matrix.{csv,json,md}`. Use it as the compact index from the eight Phase 1 goals to primary artifacts, key numbers, and caveats.

### 1. Controlled Ground-Truth CCC Benchmark

Config: `configs/experiments/exp01_controlled_benchmark.yaml`

Purpose: test whether SpatialSelf recovers known CCC mechanisms under controlled truth.

Implementation:

- Generator: `src/spitialself/benchmarks/controlled_ccc.py`
- CLI: `scripts/benchmark/generate_controlled_ccc.py`
- Validator: `scripts/benchmark/validate_controlled_ccc.py`
- Phase runner: `python scripts/phase1/run_experiment.py exp01_controlled_benchmark`
- Smoke runner: `python scripts/phase1/run_experiment.py exp01_controlled_benchmark --quick-smoke`

Must include:

- juxtacrine/contact-dependent LR
- short-to-medium paracrine LR
- longer-range chemokine/niche-specific LR
- weak signals
- high-expression false-positive LR pairs

Primary metrics: AUROC, AUPR, top-k precision, recall, weak-signal recall, empirical FDR, calibration error.

Current full output: `outputs/phase1/exp01_controlled_benchmark/datasets/` contains 5 replicates with 720 cells and 8 LR pairs per replicate. Aggregate candidate/positive counts across the 5 replicates are:

- `juxtacrine_contact`: 33,266 candidates, 1,417 positives, 800 weak positives
- `short_medium_paracrine`: 300,742 candidates, 9,856 positives, 2,070 weak positives
- `long_range_chemokine_niche`: 1,062,664 candidates, 25,610 positives, 5,501 weak positives
- `false_positive_high_expression`: 1,047,210 candidates, 0 positives

Per-LR summaries are saved in `controlled_lr_summary.csv`. The two high-expression control pairs, `ALB_EGFR_false_positive_high_expression` and `ACTB_ITGB1_false_positive_high_expression`, intentionally have 0 positive labels despite large candidate counts.

### 2. Real-Background Semi-Synthetic Benchmark

Config: `configs/experiments/exp02_semisynthetic_real_background.yaml`

Purpose: avoid artificial simulation criticism by injecting known signals into real spatial coordinates and expression background.

Implementation:

- Generator: `src/spitialself/benchmarks/semisynthetic.py`
- CLI: `scripts/benchmark/generate_semisynthetic_real_background.py`
- Validator: `scripts/benchmark/validate_semisynthetic_real_background.py`
- Phase runner: `python scripts/phase1/run_experiment.py exp02_semisynthetic_real_background`
- Smoke runner: `python scripts/phase1/run_experiment.py exp02_semisynthetic_real_background --quick-smoke`

Initial backgrounds:

- human lymph node Visium
- Slide-tags tonsil
- melanoma spatial dataset

Preserve real coordinates, sparsity, library size, local density, and cell-type composition.

Current human lymph node implementation preserves real Visium coordinates, real count background, library size, observed sparsity, cell2location-derived dominant cell type, and region labels from `visium_lr_benchmark/spot_region_from_c2l.csv`. It now exports a broad expression background by default (`n_background_genes: 2000`, `mean_variance_random` selection) instead of a tiny random panel, and supports `required_gene_csvs` for validation-specific target genes. It injects T-zone `CCL19-CCR7`, GC/B-follicle `CXCL13-CXCR5`, mixed-region `SPP1-CD44`, plus high-expression false-positive pairs.

Current full-slide sparse output: `outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_datasets/human_lymph_node_visium_replicate_00/`. This dataset uses all 4,035 real human lymph node Visium spots with 2,010 expression genes, mean observed sparsity 0.518, 4 regions, and 6 broad cell types. It writes 332,328 candidate interactions across 5 LR pairs, with 91,721 positives, 49,538 weak positives, and zero positive labels for high-expression false-positive LR pairs. The mechanism summary covers long-range chemokine-like, short/medium paracrine, and high-expression false-positive controls; all positive injected labels satisfy cell-type, range, and niche rules.

Full-slide realism statistics are saved at `outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_realism/`. The report compares the semi-synthetic expression matrix against the original `Human_Lymph_Node.h5ad` expression for the same spots and output genes. Background genes preserve real distributional structure nearly exactly: selected-library-size Pearson 0.999998, cell-sparsity Pearson 0.999991, background gene mean/sparsity Pearson 1.000/1.000, and background Moran's I Pearson 1.000. LR genes show the intended injection effect, with median mean shift 1.949 and max mean shift 12.783. This supports using the dataset as a realistic-background benchmark rather than a purely artificial simulation.

Previous manageable full output: `outputs/phase1/exp02_semisynthetic_real_background/datasets/human_lymph_node_visium_replicate_00/`. This 1,200-real-spot subset remains useful for baseline/FDR workflows that need faster repeated runs. It has 2,010 expression genes, mean observed sparsity 0.520, 805,526 candidate interactions across 5 LR pairs, 99,101 positives, 92,397 weak positives, and zero positive labels for high-expression false-positive LR pairs.

Current smoke output: `outputs/phase1/exp02_semisynthetic_real_background/smoke_datasets/human_lymph_node_visium_replicate_00/`. The refreshed quick-smoke preserves 300 real Visium spots, 2,010 expression genes, mean observed sparsity 0.519, 4 regions, and 3 broad cell types. It writes 48,575 candidate interactions over 5 LR pairs, with 5,823 positives, 5,321 weak positives, and zero positive labels for high-expression false-positive LR pairs. `scripts/benchmark/validate_semisynthetic_real_background.py` validates the schema and label invariants.

### 3. Baseline Comparison

Config: `configs/experiments/exp03_baseline_comparison.yaml`

Purpose: address missing traditional and deep-learning CCC baselines.

Required methods:

- SpatialSelf
- COMMOT: candidate-level adapter implemented and smoke-tested on Exp01/Exp02 smoke data
- CellChat: cell-type pair adapter implemented and smoke-tested on Exp01/Exp02 smoke data
- SpaTalk: cell-type pair adapter implemented; synthetic-pathway smoke tested on Exp01 smoke data
- NICHES
- CellNEST: preprocess/train/postprocess adapter implemented and smoke-tested on controlled benchmark
- DeepTalk: input conversion and `predict_ccc.txt` projection implemented; LR-specific model execution remains pending

Optional if feasible: CellPhoneDB, CytoSignal, HoloNet, stMLnet.

Fairness policy:

- use shared LR database where possible
- use shared candidate edges where method supports edge-level output
- tune only on validation data
- record explicit failure reasons for methods that cannot run

Current installation smoke status:

- `Benchmark`: COMMOT, DeepTalk, CellNEST CLI, PyTorch/PyG
- `cellchat_env`: CellChat
- `spatalk_env`: SpaTalk
- `nichenet_env`: NicheNet
- NICHES: source cloned, optional blocked by `OmnipathR/xml2/rvest` and `BisRNA` dependency issues

Current benchmark adapter status:

- `scripts/benchmark/run_baseline_comparison.py` runs available candidate-level adapters and writes `baseline_metrics.csv`, `baseline_metrics_summary.csv`, and `baseline_status.csv`.
- `scripts/phase1/run_experiment.py exp03_baseline_comparison --quick-smoke` runs a lightweight controlled benchmark comparison and records NicheNet/NICHES status rows.
- `spatialself_wavelet_edge_prior` is now registered as a separate SpatialSelf candidate method in the shared baseline runner. It uses the same graph-wavelet model family as `spatialself_wavelet`, but passes observable edge-level expression/mechanism/niche priors into the scorer. It is intentionally kept as a separate method name so comparisons do not overwrite the current fair default.
- The current full controlled core comparison is saved at `outputs/phase1/exp03_baseline_comparison/full_controlled_retained_core_fair/`. It covers 5 controlled replicates with current fair SpatialSelf-wavelet, SpatialSelf-GAT, COMMOT, DeepTalk projection/status, NicheNet downstream-only status, and NICHES optional blocked status. In this run, SpatialSelf-wavelet has higher mean AUROC than COMMOT (0.928 vs 0.898) and lower high-expression false-positive call rate (0.015 vs 0.043), but COMMOT still has higher AUPR (0.217 vs 0.207) and much higher weak-signal recall (0.580 vs 0.118). SpatialSelf-GAT is substantially weaker than wavelet on AUROC/AUPR. This makes weak-signal recovery the next model bottleneck before manuscript-scale claims.
- A refreshed full controlled core comparison including the edge-prior candidate is saved at `outputs/phase1/exp03_baseline_comparison/full_controlled_edge_prior_core/`. Across 5 controlled replicates, `spatialself_wavelet_edge_prior` improves over the current `spatialself_wavelet` default on AUPR (0.222 vs 0.207), weak-signal recall (0.155 vs 0.118), and high-expression false-positive call rate (0.008 vs 0.015), with essentially unchanged FDR (0.821) and slightly lower AUROC (0.925 vs 0.928). COMMOT remains much stronger on weak-signal recall (0.578) and lower FDR (0.793), but edge-prior narrows the AUPR gap and improves false-positive suppression.
- A 5-replicate controlled adapter-light comparison for CellChat, SpaTalk, and CellNEST is saved at `outputs/phase1/exp03_baseline_comparison/full_controlled_retained_adapter_light/`. It uses lightweight settings for reproducibility checks (`CellChat nboot=5`, `SpaTalk per_num=20` with synthetic receptor-target support, `CellNEST num_epoch=1`). CellChat is strong on AUROC/AUPR/weak recall in this projected cell-type-level setup (AUROC 0.979, AUPR 0.239, weak recall 1.000) but has top-k precision 0, so ranking calibration needs careful interpretation. SpaTalk and CellNEST are weak under these lightweight settings (AUROC about 0.500 and 0.522, FDR about 0.985). This run is adapter evidence, not final tuned baseline performance.
- A full-slide sparse semi-synthetic baseline comparison is saved at `outputs/phase1/exp03_baseline_comparison/full_slide_sparse_semisynthetic_retained/`. It uses all 4,035 human lymph node spots and 332,328 candidate interactions from the realistic-background benchmark. Runnable retained methods completed for SpatialSelf-wavelet, edge-prior wavelet, SpatialSelf-GAT, COMMOT, CellChat, SpaTalk, and CellNEST; DeepTalk records model-required/projection status, NicheNet remains downstream-only, and NICHES remains optional blocked. On this single full-slide replicate, COMMOT has the strongest AUROC/AUPR (0.717/0.480), SpatialSelf-wavelet has AUROC/AUPR 0.609/0.395, edge-prior improves weak-signal recall over default (0.0398 vs 0.0008) but worsens FDR, and CellNEST reaches weak recall 1.000 with high FDR 0.724. This result makes the weak-signal/FDR tradeoff visible under realistic expression background.
- Implemented benchmark adapters: SpatialSelf-wavelet, SpatialSelf-GAT, COMMOT, CellChat, SpaTalk, CellNEST, DeepTalk projection.
- `scripts/benchmark/run_spatialself.py` trains SpatialSelf variants from pseudo-labels and evaluates against held-out benchmark ground truth only after prediction.
- CellChat is a cell-type-level baseline; the adapter projects `(source cell type, target cell type, LR)` probabilities back to benchmark candidate edges for shared metric calculation.
- SpaTalk is also a cell-type-level baseline; the adapter projects `(sender cell type, receiver cell type, LR)` scores back to benchmark candidate edges. Native SpaTalk requires receptor downstream pathway/TF support, so synthetic benchmark smoke tests can add minimal receptor-target pathway rows and this setting must be reported.
- DeepTalk input conversion is implemented in `scripts/benchmark/prepare_deeptalk_inputs.py`; `scripts/benchmark/run_deeptalk.py` projects existing `predict_ccc.txt` outputs to candidate-level predictions and records `model_required` when LR-specific trained/pretrained models are absent.
- CellNEST adapter is implemented in `scripts/benchmark/run_cellnest.py`; controlled smoke runs preprocess, short training, postprocess, and candidate-level projection.
- NicheNet is routed to downstream validation rather than candidate-edge AUROC because it ranks ligands against receiver target gene sets. `scripts/benchmark/extract_nichenet_targets.py` extracts NicheNet ligand-target priors for all benchmark LR pairs.
- NicheNet-only downstream validation should disable the Reactome/TRRUST fallback in `scripts/benchmark/run_downstream_target_correlation.py`. After widening the semi-synthetic smoke expression background to 2010 genes, NicheNet target overlap is sufficient for the injected `CCL19-CCR7`, `CXCL13-CXCR5`, and `SPP1-CD44` signals to enter downstream target-correlation testing.
- Pending adapter work: full DeepTalk model training/execution for benchmark LR pairs.
- A combined full-adapter smoke check is saved at `outputs/phase1/exp03_baseline_comparison/smoke_controlled/combined_full_adapter_check/`. It runs SpatialSelf-wavelet with the fair LR-wise ranking default, COMMOT, CellChat, SpaTalk, CellNEST, DeepTalk projection/status, NicheNet downstream-only status, and NICHES optional-blocked status through the shared baseline runner. On controlled smoke replicate_00, all runnable adapters completed; DeepTalk correctly reports model-required/projection status, NicheNet is recorded as downstream validation only, and NICHES remains optional blocked. This check validates adapter wiring and output schemas, not manuscript-scale performance.
- The phase-runner baseline smoke now also includes `spatialself_wavelet_edge_prior` at `outputs/phase1/exp03_baseline_comparison/smoke_controlled/phase_runner/`. On smoke replicate_00 it improves AUROC/FDR and false-positive suppression relative to the current default, but has lower weak-signal recall on that single replicate; use the 5-replicate focused ablation above for model selection.
- NICHES remains optional blocked until its R dependencies install reproducibly.

Smoke commands:

- Environment smoke: `python scripts/benchmark/smoke_test_baselines.py`
- Candidate-level comparison smoke: `python scripts/benchmark/run_baseline_comparison.py <dataset_root>`

### 4. Model Component Ablation

Config: `configs/experiments/exp04_model_ablation.yaml`

Purpose: show that performance comes from SpatialSelf design, not only model capacity.

Required ablations:

- SpatialSelf-wavelet versus current GAT model
- no LR embedding
- fixed random LR embedding
- one-hot LR identity
- no spatial graph
- no distance feature
- GCN instead of GAT
- MLP instead of GAT
- heuristic pseudo-label only
- expression and distance without LR identity

Also compare pseudo-label strategies including top-K/bottom-K, quantile, soft labels, hard negatives, and weak-signal reweighting.

The wavelet variant uses multiscale sparse graph diffusion as a graph-wavelet approximation. This lets the benchmark test whether contact, paracrine, and long-range niche signals benefit from explicit multiscale spatial features.

Current implementation:

- `scripts/benchmark/run_spatialself_ablation.py` runs focused SpatialSelf ablations through the same candidate-level metric schema.
- `scripts/phase1/run_experiment.py exp04_model_ablation --quick-smoke` runs a reduced ablation set covering GAT, wavelet, mechanism/niche pseudo-labeling, fair LR-wise ranking, leakage-safe weak augmentation, LR identity variants, no distance, no spatial graph, GCN, and MLP.
- Registered ablations include GAT vs wavelet, no LR embedding, fixed random LR embedding, one-hot LR identity, no distance feature, no spatial graph, no expression features, expression+distance without LR identity, GCN, MLP, and pseudo-label strategies: expression-only, distance-only, expression+distance, soft pseudo-label, hard negatives, and weak-signal reweighting.
- `scripts/benchmark/diagnose_pseudo_labels.py` reports pseudo-label truth composition before training. This exposed that naive LR-balanced pseudo-labeling admits too many high-expression false-positive edges, while LR-capped pseudo-labeling improves pseudo-positive truth rate and weak-signal coverage.
- The first full controlled tradeoff check is saved at `outputs/phase1/exp04_model_ablation/full_spatialself_tradeoff_summary.csv`. Lowering the positive pseudo-label quantile to 0.8 and applying weak-signal reweighting raises mean weak-signal recall from 0.022 to 0.211, but lowers AUPR from 0.114 to 0.069 and increases empirical FDR from 0.880 to 0.915. This earlier run used the benchmark `weak_signal` column as a training proxy and is now treated as a leakage-prone historical diagnostic, not as a fair model result. The current code no longer reads `weak_signal` during training.
- The LR-prior hard-negative ablation is saved in `outputs/phase1/exp04_model_ablation/full_spatialself_strategy_summary.csv`. Using curated LR-pair prior/control information to treat false-positive LR pairs as hard negatives improves SpatialSelf-wavelet mean AUROC from 0.843 to 0.899, AUPR from 0.114 to 0.160, empirical FDR from 0.880 to 0.845, and high-expression false-positive call rate from 0.051 to 0.011. Weak-signal recall improves from 0.022 to 0.106 but still trails COMMOT at 0.585, so weak-signal ranking remains the next model-design bottleneck.
- Current interpretation: the LR-prior hard-negative variant is useful for false-positive suppression, but it is not a final model. The next model-design step should target weak-signal ranking directly rather than only lowering pseudo-label thresholds, because the q80 weak-reweighting check recovered more weak positives at the cost of worse AUPR/FDR.
- New mechanism-aware ablations are registered to avoid treating every LR mechanism as "nearer is better": `wavelet_mechanism_niche_lr_capped`, `wavelet_mechanism_niche_lr_prior`, and `wavelet_mechanism_niche_lr_prior_ranking`. These use LR mechanism/range/niche metadata for pseudo-label scoring, and the ranking variant adds a pairwise margin loss between pseudo-positive and hard-negative edges.
- The first controlled-smoke mechanism/ranking check is saved at `outputs/phase1/exp04_model_ablation/smoke_controlled/mechanism_ranking_check/`. On replicate_00, `wavelet_mechanism_niche_lr_capped` improved AUPR from 0.145 to 0.164 and empirical FDR from 0.844 to 0.836 relative to LR-capped expression-distance, but weak-signal recall stayed at 0.135. Adding LR-prior hard negatives increased AUROC to 0.909 but reduced weak-signal recall to 0.108; the current ranking-loss setting did not recover that loss. Mechanism-aware scoring is therefore worth keeping as an ablation, while ranking loss needs better pair construction before it can be considered a model improvement.
- LR-wise ranking now constructs pairs within each LR: weak-like/rule-consistent pseudo-positives are ranked above high-expression or rule-violating hard negatives. The earlier controlled-smoke check at `outputs/phase1/exp04_model_ablation/smoke_controlled/lrwise_ranking_check/` improved AUROC to 0.927, AUPR to 0.215, and empirical FDR to 0.810, but weak-signal recall remained 0.108. Because the weak-like proxy previously read the benchmark `weak_signal` column, this result should be rerun before being used in any fair comparison.
- Current weak-like training proxy is leakage-safe: it uses pseudo-score rank, expression rank, mechanism distance prior, niche prior, and available rule-like metadata, but never the benchmark `weak_signal` label. New registered ablations are `wavelet_mechanism_niche_weak_augmented` and `wavelet_mechanism_niche_weak_augmented_ranking`, which add weak-like pseudo-positives without using ground truth.
- The current phase-runner ablation smoke at `outputs/phase1/exp04_model_ablation/smoke_controlled/phase_runner/` now includes the fair LR-wise ranking and weak-augmented variants plus the extended LR identity and architecture controls. On replicate_00, `wavelet_mechanism_niche_lr_prior_lrwise_ranking` reaches AUROC 0.925, AUPR 0.208, weak-signal recall 0.108, and FDR 0.821; `wavelet_mechanism_niche_weak_augmented` has weak-signal recall 0.162 with lower AUPR 0.165. The same smoke table now includes `wavelet_fixed_random_lr_embedding`, `wavelet_one_hot_lr_identity`, `gcn_hard_expression_distance`, and `mlp_hard_expression_distance`, so the requested identity/architecture ablation axes are wired into the metric schema.
- The leakage-safe weak-augmented smoke check is saved at `outputs/phase1/exp04_model_ablation/smoke_controlled/fair_weak_augmented_check/`. On replicate_00, `wavelet_mechanism_niche_weak_augmented` improved weak-signal recall from 0.135 to 0.162 and empirical FDR from 0.836 to 0.833 relative to `wavelet_mechanism_niche_lr_capped`, with similar AUPR (0.165 vs 0.164). Combining weak augmentation with LR-wise ranking improved AUPR to 0.208 and FDR to 0.821 but reduced weak-signal recall to 0.054, so weak augmentation and ranking should remain separate ablation axes for now.
- A full controlled fair focused ablation over 5 replicates is saved at `outputs/phase1/exp04_model_ablation/full_fair_focused_ablation/`. The runner now writes `spatialself_ablation_summary.csv` with mean/std/count per ablation. In this full run, `wavelet_mechanism_niche_lr_prior_lrwise_ranking` is the strongest fair SpatialSelf variant so far: mean AUROC 0.928, AUPR 0.207, weak-signal recall 0.118, empirical FDR 0.821, and high-expression false-positive call rate 0.015. `wavelet_mechanism_niche_lr_capped` has mean AUROC 0.879, AUPR 0.185, weak-signal recall 0.014, FDR 0.854, and false-positive call rate 0.044. `wavelet_mechanism_niche_weak_augmented` raises weak-signal recall to 0.037 but lowers AUPR to 0.149 and increases FDR to 0.863, so weak augmentation remains an ablation rather than a default.
- A new edge-prior focused ablation is saved at `outputs/phase1/exp04_model_ablation/full_edge_prior_focused_ablation_epoch10/`. This variant passes observable edge-level priors to the scorer: LR expression strength, LR expression rank, mechanism-distance prior, niche prior, and normalized distance. The best candidate in this diagnostic is `wavelet_mechanism_niche_edge_prior_ranking`, which improves mean AUPR from 0.207 to 0.222 and weak-signal recall from 0.118 to 0.155 relative to the current fair default, while keeping empirical FDR essentially unchanged (0.821) and reducing high-expression false-positive rate from 0.015 to 0.008. Mean AUROC decreases slightly from 0.928 to 0.925. This is the next model candidate to carry into baseline/FDR/real-data reruns, not yet the final default.
- A full score-calibration diagnostic over the same 5 controlled replicates is saved at `outputs/phase1/exp04_model_ablation/full_fair_score_calibration_nonoracle/`. The non-oracle transforms do not fix weak-signal recovery: raw scores remain the best fair setting among tested post-processing options (mean AUROC 0.928, AUPR 0.207, weak-signal recall 0.118, FDR 0.821), while LR-wise rank calibration drops AUPR to 0.114 and weak recall to 0.036, and non-oracle mechanism/distance rank drops AUPR to 0.078 with similar weak recall (0.114) but worse FDR (0.907). The `lr_mechanism_prior_rank` transform reaches AUROC 0.995, AUPR 0.652, and weak recall 0.982, but it uses rule-satisfied columns and is therefore an oracle-style upper bound diagnostic, not a fair SpatialSelf model. The next model change should happen in training-time weak-positive mining or auxiliary objectives, not score post-processing.

### 5. Weak-Signal And False-Positive Stress Test

Config: `configs/experiments/exp05_weak_signal_false_positive.yaml`

Purpose: directly test whether SpatialSelf misses weak CCC or overcalls high-expression artifacts.

Stressors:

- weak/very weak injected signals
- wrong-domain high-expression LR pairs
- non-permissive cell-type pairs
- shuffled spatial pairs
- ligand/receptor dropout

Primary metrics: weak-signal recall, FDR, false-positive suppression, calibration error.

Current implementation:

- `scripts/benchmark/run_weak_signal_false_positive.py` summarizes any `candidate_id, score` prediction table against benchmark truth.
- `scripts/benchmark/run_weak_signal_false_positive_dataset.py` can run lightweight methods and writes method-level metrics plus threshold, top-k, LR-level, and mechanism-level stress tables.
- `scripts/phase1/run_experiment.py exp05_weak_signal_false_positive --quick-smoke` runs SpatialSelf-wavelet and SpatialSelf-GAT on the controlled smoke dataset.
- Current quick-smoke uses the leakage-safe fair SpatialSelf default (`lr_prior_lr_capped_mechanism_niche_expression_distance`, LR-wise ranking loss weight 0.35, margin 0.5) and now includes `spatialself_wavelet_edge_prior` as a separate candidate. On smoke replicate_00, default wavelet has AUROC 0.925, AUPR 0.208, weak-signal recall 0.108, FDR 0.821, and high-expression false-positive call rate 0.026; edge-prior wavelet has AUROC 0.945, AUPR 0.211, weak-signal recall 0.081, FDR 0.780, and high-expression false-positive call rate 0.006; GAT has AUROC 0.840, AUPR 0.083, weak-signal recall 0.000, FDR 0.869, and high-expression false-positive call rate 0.016. This smoke validates wiring and highlights the precision/weak-recall tradeoff on one replicate.
- Mechanism-level stress shows good overall ranking for short-medium paracrine and long-range chemokine mechanisms, but LR-level stress still exposes weak-signal heterogeneity: `JAG1-NOTCH2` weak contact and `TGFB1-TGFBR2` weak paracrine have usable LR-level AUROC, while weak `CXCL13-CXCR5` remains difficult. This keeps weak-signal recall as an active model-design issue.

### 6. Permutation p-value And FDR

Config: `configs/experiments/exp06_permutation_fdr.yaml`

Purpose: add statistical significance to predictions.

Implementation:

- Shared module: `src/spitialself/fdr.py`
- CLI: `scripts/benchmark/run_permutation_fdr.py`
- Input schema: candidate table plus prediction table containing `candidate_id` and `score`
- Output schema: candidate-level `empirical_p_value`, `bh_fdr`, null mean/std/q95, LR-pair FDR table, cell-type-pair FDR table, and summary discovery counts
- When ground truth columns are present, the summary also reports realized FDR, precision, recall, weak-signal recall, high-expression false-positive call rate, positive/negative control minimum p-values, and a KS-style p-value uniformity statistic.

The current implementation is a method-agnostic score-permutation post-processing null. It gives SpatialSelf and baselines the same FDR output format now; stricter model-rerun nulls can reuse the same schema later.

Group-level LR-pair and cell-type-pair p-values use Simes aggregation over candidate edges, followed by BH correction across groups.

Current status:

- `scripts/benchmark/run_permutation_fdr.py` runs all four nulls on the controlled benchmark SpatialSelf LR-capped smoke prediction.
- `scripts/phase1/run_experiment.py exp06_permutation_fdr --quick-smoke` runs three nulls on a controlled SpatialSelf-wavelet smoke prediction as a fast wiring check.
- Each null writes candidate-edge, LR-pair, and cell-type-pair FDR tables.
- With 20 smoke permutations, no BH-significant edge is expected; this smoke validates schema and diagnostics only.
- A full controlled pilot for replicate_00 is saved at `outputs/phase1/exp06_permutation_fdr/full_controlled_replicate00_spatialself_lr_prior/`. It used the LR-prior hard-negative SpatialSelf prediction, 480,013 candidate interactions, three nulls, and 100 permutations. The pilot produced no BH-significant candidate edges, no significant LR-pair groups, and no significant cell-type-pair groups; the minimum raw p-value was 0.0099 and the minimum BH-FDR was approximately 0.987-1.000 across nulls.
- A matched edge-prior candidate FDR pilot for replicate_00 is saved at `outputs/phase1/exp06_permutation_fdr/full_controlled_replicate00_spatialself_edge_prior/`. It used the full controlled edge-prior prediction table, 480,013 candidate interactions, three nulls, and 100 permutations. It also produced no BH-significant candidate edges or groups; minimum raw p-value was 0.0099 and minimum BH-FDR was approximately 0.994-1.000. This confirms the FDR schema works for the candidate model, but also confirms the current score-permutation null remains conservative.
- A full controlled FDR panel is saved at `outputs/phase1/exp06_permutation_fdr/full_controlled_spatialself_panel/`. It covers `spatialself_wavelet` and `spatialself_wavelet_edge_prior` across 5 controlled replicates, three null models (`spatial_permutation`, `receiver_permutation`, `lr_identity_permutation`), and 100 permutations. It writes candidate-edge, LR-pair, and cell-type-pair FDR outputs for 30 method/replicate/null combinations plus method-null summaries. It produced no BH-significant candidate edges or groups, confirming that the current method-agnostic score-permutation null is very conservative for the present SpatialSelf score distribution.
- A realistic-background FDR panel is saved at `outputs/phase1/exp06_permutation_fdr/full_slide_sparse_semisynthetic_panel/`. It uses the 4,035-spot full-slide sparse human lymph node semi-synthetic benchmark with 332,328 candidate interactions. The panel covers `spatialself_wavelet`, `spatialself_wavelet_edge_prior`, and `commot` under spatial, receiver, and LR-identity permutation nulls with 50 permutations. All 3 method runs completed and wrote candidate-edge, LR-pair, and cell-type-pair FDR tables. It also found no BH-significant edges or groups, so this should be presented as a completed realistic-background FDR schema check and evidence of current null conservativeness, not as a powered discovery analysis.
- Interpretation of the full panel: the schema and diagnostics are now in place, but manuscript-level discovery claims will need either stronger model scoring separation, a better calibrated local/group null, higher-powered permutations, or a stricter model-rerun permutation design.
- The phase runner now generates FDR input predictions with the same leakage-safe default as exp03/exp05: `lr_prior_lr_capped_mechanism_niche_expression_distance` plus LR-wise ranking loss weight 0.35 and margin 0.5. Quick-smoke FDR outputs are written under `outputs/phase1/exp06_permutation_fdr/smoke_spatialself_wavelet_fair_phase_runner/`.
- The fair FDR quick-smoke completed on controlled replicate_00 with 12,810 candidates, three null models, and 20 permutations. It wrote candidate-edge, LR-pair, and cell-type-pair FDR tables for spatial, receiver, and LR-identity permutations. As expected for a low-permutation smoke run, it found no BH-significant edges or groups; minimum raw p-value was 0.0476 and minimum BH-FDR was 0.996-1.000.
- The edge-prior candidate FDR quick-smoke is saved at `outputs/phase1/exp06_permutation_fdr/smoke_spatialself_wavelet_edge_prior_phase_runner/`. It completed on the same controlled smoke replicate with 12,810 candidates, three null models, and 20 permutations, and wrote the same candidate-edge, LR-pair, and cell-type-pair FDR schemas. It is a wiring check only; main FDR claims still require higher permutation counts and the chosen default model.

Null models:

- spatial permutation
- receiver permutation
- LR identity permutation
- within-domain permutation

Outputs:

- empirical p-value
- BH-adjusted FDR
- significant LR pairs
- significant cell-type pair interactions
- significant spatial edges

### 7. Human Lymph Node Global Validation

Config: `configs/experiments/exp07_lymph_node_global_validation.yaml`

Purpose: avoid selected-case-only criticism by evaluating all LR pairs before showing CCR7-CCL19.

Required analyses:

- all-LR region enrichment
- downstream target correlation
- FDR
- top pathway enrichment
- spatial concentration
- method ranking across all LR pairs

Current implementation:

- `scripts/benchmark/run_region_enrichment.py` computes all-LR region enrichment and global LR ranking from candidate-level predictions.
- The first smoke test on the lymph-node semi-synthetic benchmark recovers CCL19-CCR7 enrichment in T cell zone as a top region signal.
- `scripts/benchmark/run_downstream_target_correlation.py` correlates LR communication scores with receiver-side target signatures using pathway CSV targets or Reactome receptor-TF plus TRRUST TF-target fallback.
- `scripts/phase1/run_experiment.py exp07_lymph_node_global_validation --quick-smoke` runs both `spatialself_wavelet` and `spatialself_wavelet_edge_prior` prediction, all-LR region enrichment, NicheNet target extraction, and NicheNet-only downstream target correlation on the semi-synthetic lymph-node smoke dataset. Prediction, region-enrichment, LR-ranking, and downstream-correlation outputs are written separately for the two methods.
- The prediction step uses the same leakage-safe SpatialSelf default as exp03/exp05/exp06: `lr_prior_lr_capped_mechanism_niche_expression_distance` plus LR-wise ranking loss weight 0.35 and margin 0.5.
- The fair exp07 quick-smoke completed under `outputs/phase1/exp07_lymph_node_global_validation/smoke_phase_runner/`. Downstream target correlation is strong for injected biological controls under the current default (`CXCL13-CXCR5` Spearman 0.540, `CCL19-CCR7` Spearman 0.535) and stronger under the edge-prior candidate (`CCL19-CCR7` Spearman 0.605, `CXCL13-CXCR5` Spearman 0.591), with smoke empirical p=0.0476 using 20 permutations. Region enrichment and global LR ranking still place high-expression false-positive LR pairs near true signals, so false-positive suppression remains a real-data validation bottleneck.
- The stronger full-slide sparse validation now uses all 4,035 human lymph node Visium spots from `outputs/phase1/exp02_semisynthetic_real_background/full_slide_sparse_datasets/`. Outputs are under `outputs/phase1/exp07_lymph_node_global_validation/full_slide_sparse_region_enrichment/`, `full_slide_sparse_downstream_target/`, and `full_slide_sparse_control_summary/`. Both default wavelet and edge-prior wavelet rank the 3 injected positive controls ahead of the 2 high-expression negative controls. Mean positive downstream Spearman is 0.763 for default and 0.765 for edge-prior, and all 3 positive LR controls have downstream BH-FDR 0.0099 using 100 permutations.
- Pending: broader pathway enrichment across external real datasets.

Case studies after global analysis:

- CCL19-CCR7
- CCL21-CCR7
- CXCL13-CXCR5

### 8. Runtime And Memory Benchmark

Config: `configs/experiments/exp08_runtime_memory.yaml`

Purpose: assess high-resolution scalability.

Scales:

- 1k
- 5k
- 10k
- 50k
- 100k cells/spots

Metrics:

- wall-clock time
- peak CPU memory
- peak GPU memory
- candidate edges
- LR pairs
- runtime per epoch
- total runtime

Current implementation:

- `scripts/benchmark/profile_runtime_memory.py` profiles runnable adapters with `/usr/bin/time -v` and writes `runtime_memory_profile.csv`.
- `scripts/benchmark/run_runtime_scaling.py` generates controlled benchmark scales, profiles selected methods, and writes `runtime_memory_scaling.csv`.
- Runnable now: SpatialSelf-wavelet, SpatialSelf-GAT, COMMOT, CellChat, SpaTalk, CellNEST, and DeepTalk projection/status.
- Pending but represented in the same table: NICHES; NicheNet is treated as downstream-prior validation rather than candidate-edge scoring.
- GPU memory is sampled from `nvidia-smi` when available; otherwise the table records `nvidia_smi_unavailable`.
- The default controlled scaling runner is capped at 10k cells because `candidate_generation: full_radius` materializes an all-pairs distance matrix. The controlled and semi-synthetic generators now also support `candidate_generation: sparse_sampled`, `max_candidates_per_lr`, and `max_candidates_per_sender_lr` for bounded 50k/100k stress tests.
- Runtime profiling now uses the same fair SpatialSelf default as exp03/exp05/exp06/exp07. The current full Phase 1 runtime evidence is saved at `outputs/phase1/exp08_runtime_memory/controlled_medium_scaling/runtime_memory_scaling.csv`. It covers 500 and 1000 cells with SpatialSelf-wavelet, edge-prior wavelet, SpatialSelf-GAT, and COMMOT. At 1000 cells / 926,654 candidate edges, wavelet took 15.73 s and 972.5 MB peak CPU memory, edge-prior took 14.98 s and 964.1 MB, GAT took 12.87 s and 970.7 MB, and COMMOT took 75.33 s and 1211.1 MB.
- A capped sparse SpatialSelf scaling run is saved at `outputs/phase1/exp08_runtime_memory/sparse_spatialself_scaling/runtime_memory_scaling.csv`. It covers 1,000, 5,000, and 10,000 cells with SpatialSelf-wavelet, edge-prior wavelet, and SpatialSelf-GAT. At 10,000 cells / 130,223 candidate edges, wavelet took 14.30 s and 4247.7 MB peak CPU memory, edge-prior took 15.63 s and 4248.1 MB, and GAT took 11.18 s and 4246.2 MB. This is capped-candidate scalability evidence, not a replacement for uncapped high-resolution profiling.
- The SpatialSelf benchmark runner now builds KNN graphs with `scipy.spatial.cKDTree` instead of an all-pairs distance matrix. Before this fix, a 50,000-cell capped sparse GAT smoke run completed but peaked at 95,874.5 MB CPU memory. After the KD-tree KNN fix, the same 50,000-cell / 130,790-edge capped sparse dataset runs all three SpatialSelf variants: wavelet took 8.05 s and 1268.3 MB peak CPU memory, edge-prior wavelet took 8.77 s and 1281.8 MB, and GAT took 10.86 s and 4433.5 MB. The fixed profiles are saved at `outputs/phase1/exp08_runtime_memory/sparse_50k_spatialself_gat_sparse_knn_profile/runtime_memory_profile.csv` and `outputs/phase1/exp08_runtime_memory/sparse_50k_spatialself_wavelet_profile/runtime_memory_profile.csv`.
- The 100,000-cell capped sparse smoke is saved at `outputs/phase1/exp08_runtime_memory/sparse_100k_spatialself_smoke/runtime_memory_scaling.csv`. On 100,000 cells / 130,547 candidate edges, wavelet took 10.88 s and 2062.6 MB peak CPU memory, edge-prior wavelet took 11.83 s and 2037.5 MB, and GAT took 12.57 s and 8386.8 MB. This supports capped-candidate 100k feasibility for SpatialSelf variants, while higher candidate caps and real high-resolution subsets remain necessary for manuscript-scale scalability claims.
- A 50,000-cell higher-cap sparse run is saved at `outputs/phase1/exp08_runtime_memory/sparse_50k_highcap_spatialself/runtime_memory_scaling.csv`. It uses `max_candidates_per_lr: 50000` and `max_candidates_per_sender_lr: 48`, producing 326,963 candidate edges. Wavelet took 10.76 s and 1353.6 MB peak CPU memory, edge-prior wavelet took 10.12 s and 1340.0 MB, and GAT took 11.78 s and 4585.6 MB. This shows candidate-edge scaling remains moderate at roughly 2.5x the previous 50k capped edge count.
- A retained-method full-slide sparse runtime profile is saved at `outputs/phase1/exp08_runtime_memory/full_slide_sparse_semisynthetic_retained_profile/runtime_memory_profile.csv`. It uses the realistic-background 4,035-spot human lymph node benchmark with 332,328 candidate interactions and 5 LR pairs. Completed wall-clock / peak CPU memory values are: SpatialSelf-wavelet 20.86 s / 2537.9 MB, edge-prior wavelet 20.92 s / 2611.5 MB, SpatialSelf-GAT 10.81 s / 1070.4 MB, COMMOT 501.33 s / 4392.7 MB, CellChat 27.64 s / 1117.5 MB, SpaTalk 170.22 s / 2448.1 MB, CellNEST 78.60 s / 806.4 MB, and DeepTalk projection/status 43.04 s / 988.9 MB. NicheNet is recorded as a pending runtime adapter because it is used for downstream-prior validation rather than candidate-edge scoring; NICHES remains optional blocked.
- Sparse candidate generation smoke evidence is saved under `outputs/phase1/exp08_runtime_memory/sparse_candidate_smoke/`. The generated 600-cell controlled candidate table has 44,491 candidate interactions, 3,252 positives, 847 weak positives, all four controlled mechanism categories, and zero positive labels for high-expression false-positive LR pairs. The matching runtime smoke profiles `spatialself_wavelet` on 44,181 candidate edges in 7.44 seconds peak 496.3 MB CPU memory.
- The same sparse candidate path also validates on a 300-spot real-background semi-synthetic smoke dataset with 2,010 expression genes, 12,006 candidate interactions, 2,353 positives, 1,647 weak positives, paracrine/long-range/false-positive mechanism coverage, and zero positive labels for high-expression false-positive LR pairs.
- The quick-smoke scaling table remains available at `outputs/phase1/exp08_runtime_memory/smoke_scaling_controlled/runtime_memory_scaling.csv` for fast wiring checks.

## Review Risks Addressed

| Review risk | Phase 1 experiment |
| --- | --- |
| Simulation aligned with method assumptions | Exp01, Exp02, Exp05 |
| Missing CellNEST/DeepTalk/deep baselines | Exp03 |
| Missing ablation | Exp04 |
| Top-K pseudo-label may miss weak signals | Exp04, Exp05 |
| No p-value/FDR | Exp06 |
| Real data only shows selected interactions | Exp07 |
| No high-resolution scalability evidence | Exp08 |
| False positives from high expression | Exp01, Exp05, Exp06 |

## Implementation Order

1. Implement Exp01 simulator and metrics.
2. Implement Exp02 spike-in on real backgrounds.
3. Add common prediction table schema for SpatialSelf and baselines.
4. Add baseline adapters for COMMOT, CellChat, NICHES, CellNEST, DeepTalk, SpaTalk, and NicheNet downstream validation.
5. Add ablation switches and pseudo-label strategies.
6. Add permutation/FDR module.
7. Add lymph-node global validation workflow.
8. Add runtime/memory profiler.

## Definition Of Done

Phase 1 is done when each experiment has:

- one config file
- one executable runner
- saved prediction tables
- saved metric tables
- saved run metadata
- documented failures for unsupported baselines
- figure-ready summary tables

Current audit command:

```bash
conda run -n Benchmark python scripts/phase1/audit_phase1.py
```

The audit writes `outputs/phase1/phase1_audit/phase1_audit_summary.csv` and `.json`. "Smoke-ready" still means schema and wiring evidence, not manuscript-scale performance evidence; full-scale reruns remain needed before paper claims.

Strict completion gate:

```bash
conda run -n Benchmark python scripts/phase1/check_phase1_completion.py
```

The completion gate writes `outputs/phase1/phase1_completion_gate/phase1_completion_gate.{csv,json,md}`. It verifies the original eight Phase 1 goals directly: controlled mechanism truth, real-background semi-synthetic evidence, retained baselines after removing the expression-distance baseline by user decision, replaceable model structures, ablations, permutation/FDR outputs, full-slide lymph-node global validation, runtime/memory profiling, and repository hygiene. The current gate has 28/28 passing checks.
