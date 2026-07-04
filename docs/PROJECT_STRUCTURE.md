# Project Structure Notes

## Intent

The project has two experiment families:

1. New Phase 1 synthetic/controlled benchmarks generated from auditable simulators.
2. Real spatial transcriptomics datasets trained with self-supervised LR samples.

Both families share the same high-level idea:

```text
expression + spatial coordinates + LR table
  -> spatial graph
  -> node encoder
  -> LR embedding
  -> directed edge scorer
  -> LR-specific communication probability
```

## Stable Source Layer

`src/spitialself/` is the new home for reusable implementation:

- `models.py`: reusable GraphSAGE and GATv2 model classes.
- `metrics.py`: reusable metric and threshold helpers.
- `evaluation.py`: shared candidate-level benchmark evaluation metrics.
- `fdr.py`: method-agnostic permutation p-value and BH-FDR helpers.
- `baselines/commot_adapter.py`: COMMOT adapter that projects OT communication matrices to candidate-level benchmark predictions.
- `baselines/deeptalk_io.py`: DeepTalk input converter with candidate-id mapping files.
- `baselines/cellnest_io.py`: CellNEST h5ad/database/annotation input converter.
- `benchmarks/controlled_ccc.py`: controlled ground-truth CCC simulator.
- `benchmarks/semisynthetic.py`: real-background semi-synthetic CCC generator.

The goal is to move reusable graph building, self-supervised sampling, inference, and plotting utilities here over time.

## Experiment Entrypoints

Phase 1 entrypoints live under `scripts/phase1/`:

- `list_experiments.py`: lists the eight Phase 1 experiments.
- `run_experiment.py`: dry-run capable runner with quick-smoke entries for all eight Phase 1 experiments.

Benchmark utilities live under `scripts/benchmark/`:

- `generate_controlled_ccc.py`: generates controlled CCC benchmark replicates.
- `validate_controlled_ccc.py`: validates the controlled benchmark schema and label invariants.
- `generate_semisynthetic_real_background.py`: injects known LR signals into real ST coordinates and expression background.
- `validate_semisynthetic_real_background.py`: validates semi-synthetic real-background schema and injected-label invariants.
- `run_spatialself.py`: trains SpatialSelf GAT or graph-wavelet variants on pseudo-labels and writes candidate-level predictions.
- `run_spatialself_ablation.py`: runs focused SpatialSelf model/pseudo-label ablations with shared metric output.
- `run_commot.py`: runs COMMOT and writes candidate-level prediction and metric tables.
- `run_cellchat.py`: runs CellChat through R and projects cell-type-pair LR probabilities back to benchmark candidate edges.
- `run_cellchat_replicate.R`: per-replicate CellChat runner used by the Python adapter.
- `run_cellnest.py`: runs CellNEST preprocess/train/postprocess and projects attention scores to benchmark candidate edges.
- `run_baseline_comparison.py`: runs available baseline adapters and records pending/blocked baseline status.
- `prepare_deeptalk_inputs.py`: prepares DeepTalk per-LR prediction directories and candidate maps.
- `run_deeptalk.py`: projects DeepTalk `predict_ccc.txt` outputs to benchmark candidate edges and records missing LR-specific model status.
- `prepare_cellnest_inputs.py`: prepares CellNEST h5ad, LR database, annotation, and preprocess command files.
- `run_permutation_fdr.py`: computes empirical p-values and BH-FDR for any prediction table.
- `profile_runtime_memory.py`: wraps benchmark methods with wall-clock and peak-RSS profiling.
- `run_region_enrichment.py`: computes all-LR region enrichment and global LR ranking from candidate-level predictions.
- `run_downstream_target_correlation.py`: correlates LR communication scores with receiver-side target-gene signatures.
- `smoke_test_baselines.py`: checks installed baseline environments and records optional blocked methods.

Current candidate-level baseline status:

- Formal SpatialSelf baseline: `spatialself_wavelet_scale` (`SpatialSelf-scale`), a scale-aware graph-wavelet model with LR-specific scale-mixture scoring.
- Retained SpatialSelf variants for ablation/history: `spatialself_wavelet`, `spatialself_wavelet_adaptive`, `spatialself_wavelet_edge_prior`, and `spatialself_gat`.
- Implemented external adapters: COMMOT, CellChat, SpaTalk, CellNEST, DeepTalk projection.
- CellChat is naturally a cell-type-level method, so its adapter maps `(source cell type, target cell type, LR)` probabilities back to every matching candidate edge before shared metric/FDR calculation.
- DeepTalk model execution still requires LR-specific trained/pretrained model files; the projection adapter is implemented and smoke-tested.
- Optional blocked: NICHES, pending R dependency resolution on this machine.

Root-level real-data scripts are retained for now because they encode working relative paths:

- `train_visium.py`: trains GATv2 self-supervised Visium model.
- `pred_train.py`: runs full Visium LR inference.

## Data And Artifact Directories

Large real-data directories are intentionally not moved in this refactor:

- `visium_lr_benchmark`
- `high_spatial/visium_lr_benchmark`
- `mouse/visium_lr_benchmark`
- `HumanMelannomaMultiome/melanoma_lr_benchmark`
- `lung_cancer/visium_lr_benchmark_lung`

The old benchmark directories and result artifacts were removed. New benchmark outputs should be created under `outputs/phase1/`.

Manuscript-critical experiments that should be easy to recover later are indexed in `docs/KEY_EXPERIMENTS.md`. The DSS/MERFISH mouse colon IBD time-course experiment is marked there as a central real-data biology case study.

## Legacy Dataset Folders

Dataset folders currently contain both code and results:

- `high_spatial/`
- `mouse/`
- `HumanMelannomaMultiome/`
- `lung_cancer/`

They repeat the same pattern:

```text
struct_grape.py or equivalent graph builder
train.py / train_visium.py
pred.py / pred_train.py
analyze*.py
plot*.py
data/
*_lr_benchmark*/
figures*/
```

## Next Refactor Steps

1. Add `configs/datasets/*.yaml` for paths and thresholds.
2. Extract graph building into `src/spitialself/graph.py`.
3. Extract self-supervised sample construction into `src/spitialself/sampling.py`.
4. Replace dataset-local `train.py` and `pred.py` copies with wrappers.
5. Move plotting scripts into `analysis/` or `scripts/analysis/` after their inputs are config-driven.
