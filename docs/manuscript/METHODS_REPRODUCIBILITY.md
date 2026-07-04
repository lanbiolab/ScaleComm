# Methods and Reproducibility Checklist

This Phase 5 checklist tracks the methods details that must be explicit before submission.

## Simulation and Benchmark Construction

- Controlled simulator v2 must describe contact-dependent, short-to-medium paracrine, long-range niche-restricted, weak-signal, and high-expression false-positive LR mechanisms.
- Real-background semi-synthetic simulation must state which real-like backgrounds are used, which properties are preserved, and how injected truth is added.
- Ground-truth tables must define positive LR pairs, negative LR pairs, true edges, cell-type pairs, spatial niches, and weak-signal labels.

Current evidence: Phase 4 controlled and real-background benchmark summaries.

Remaining polish: add a concise mathematical or algorithmic description for the injection process.

## LR Database and Complex Handling

- State ligand-receptor database source.
- State whether complexes are expanded, collapsed, filtered, or excluded.
- State how ligand/receptor genes absent from the expression matrix are handled.
- State whether all methods use the same candidate LR universe.

Current evidence: baseline configs and method adapters.

Remaining polish: write one locked LR-database manifest for the manuscript.

## Candidate Graph and Range Handling

- Define spatial candidate edges and KNN usage.
- Explain how graph-wavelet propagation allows information from local to broader graph neighborhoods without treating KNN as the final biological range definition.
- Distinguish `K_NN` for graph construction from `K_pos` and `K_neg` for pseudo-label sampling.
- State how distance features and maximum range priors are used.

Current evidence: Phase 4 model and ablation outputs.

Remaining polish: add notation table for `K_NN`, `K_pos`, `K_neg`, `R_max`, wavelet scale, and distance bandwidth.

## Self-Supervised Pseudo-Labels

- Describe pseudo-label score construction.
- Describe hard or soft positive/negative sampling.
- Explain weak-signal guardrails and why raw weak recall is interpreted together with FDR/FP metrics.

Current evidence: weak-signal claim table and ablation panel.

Remaining polish: include pseudo-label sensitivity in supplement if enough source tables are available.

## Permutation and FDR

- Describe spatial permutation, receiver permutation, and LR-identity permutation nulls.
- State empirical p-value calculation.
- State BH-adjusted FDR procedure.
- Separate edge-level, LR-level, and cell-type-pair-level outputs.

Current evidence: Phase 4 500-permutation key lymph-node panel.

Remaining polish: explicitly state that current strongest FDR evidence is group-level in the key panel.

## Baseline Fairness

- State retained baselines: COMMOT, CellChat, SpaTalk, CellNEST.
- State status baselines: DeepTalk and NICHES.
- State excluded baseline: LR-Coexp.
- State common inputs, common LR universe, and candidate-universe alignment where applicable.
- State failure or status reason for any method not included in formal metric tables.

Current evidence: Phase 4 baseline evidence index and baseline claim guardrails.

Remaining polish: create a baseline status table for Supplementary Methods.

## Real-Sample Validation

- State that native unlabeled ST data do not provide complete ground truth.
- Use positive controls, high-expression negative controls, region or niche enrichment, downstream target correlation, and FDR group significance.
- Present global statistics before selected LR examples.
- For Phase 5 real-background validation, report `phase5_real_validation_tissue_summary.csv`, `phase5_real_validation_method_summary.csv`, `phase5_real_validation_lr_panel.csv`, and `phase5_real_validation_claims.csv`.
- Distinguish curated downstream validation in the key lymph-node panel from receiver-response proxy validation in tonsil and melanoma.
- State that high-expression false-positive controls are injected controls and do not replace native-real random/shuffled negative controls.

Current evidence: Phase 4 key lymph-node downstream panel; Phase 2/3 biological validation indices.

Remaining polish: strengthen tonsil and melanoma native-real validation panels with random LR pairs, shuffled LR database, spatially shuffled coordinates, and non-permissive cell-type/domain controls.

## Reproducibility Entry Points

Recommended commands:

```bash
conda run -n Benchmark python scripts/phase4/build_phase4_evidence_index.py --config configs/experiments/phase4_manuscript_evidence_lock.yaml
conda run -n Benchmark python scripts/phase4/audit_phase4.py --config configs/experiments/phase4_manuscript_evidence_lock.yaml
conda run -n Benchmark python scripts/phase5/build_phase5_evidence_index.py --config configs/experiments/phase5_submission_manuscript_lock.yaml
conda run -n Benchmark python scripts/phase5/audit_phase5.py --config configs/experiments/phase5_submission_manuscript_lock.yaml
```

Remaining polish: add a one-command tutorial after Phase 5 validation scripts are finalized.
