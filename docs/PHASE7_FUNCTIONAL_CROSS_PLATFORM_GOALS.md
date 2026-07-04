# Phase 7: Functional and Cross-platform Validation

## Objective

Phase 7 upgrades SpatialSelf from benchmark and ranking evidence to functional and cross-platform validation.

Plain-language goal:

> Move SpatialSelf from "good benchmark metrics and rankings" to "weak or context-specific CCC discoveries with downstream response support, baseline separation, and FDR control across tissue backgrounds and spatial platforms."

## Reviewer Risks Addressed

1. Current biological validation is still partly proxy-based, especially receptor-expression proxy in Phase 6.
2. Deep spatial CCC baselines, especially DeepTalk, can still be challenged by reviewers.
3. Current data platforms may look too restricted to realistic/semi-synthetic benchmark settings.

## Required Goals

### Goal 1: Downstream Target Validation

Upgrade receptor-expression proxy into functional receiver-response validation.

Inputs:

- SpatialSelf, COMMOT, CellChat, SpaTalk, and CellNEST predictions on matched LR candidate universes.
- Candidate interactions with sender/receiver ids, LR ids, cell types, coordinates, and niches.
- Expression matrices.
- Target gene resources from NicheNet, OmniPath, Reactome, KEGG, and/or CellChatDB.

Priority LR panels:

- lymph node: `CCL19/CCL21-CCR7`, `CXCL13-CXCR5`, `CD48-CD244`;
- tonsil: `IL21-IL21R`, `CD40LG-CD40`, `CXCL13-CXCR5`;
- melanoma: `CXCL9/10-CXCR3`, `SPP1-CD44`, `PDCD1-CD274`.

Required outputs:

- `phase7_downstream_target_signature_table.csv`
- `phase7_lr_target_response_correlation.csv`
- `phase7_method_target_response_comparison.csv`
- `phase7_downstream_fdr.csv`
- source tables for any figure panels.

Completion criteria:

- At least one key lymph-node LR and one non-lymph-node LR have target signature scores.
- SpatialSelf incoming LR score is tested against receiver target activity.
- COMMOT, CellChat, SpaTalk, and CellNEST are evaluated on the same LR/cell background where their outputs exist.
- Permutation/FDR support is reported.
- Receptor-expression proxy is clearly separated from target-signature validation.

Supported claim if passed:

> SpatialSelf-ranked weak/context-specific LR activity is associated with receiver downstream target activity in selected tissue backgrounds.

Limitations:

- Target signatures depend on curated priors and are not direct perturbation evidence.
- Absence of a target set for a given LR is not evidence against that LR.

### Goal 2: DeepTalk and Deep Baseline Closure

Close the most obvious deep-learning baseline gap.

Formal baseline target:

- DeepTalk.

Status/failure table target:

- DeepTalk if formal run fails;
- HoloNet;
- stMLnet;
- CytoSignal;
- NICHES.

Required outputs:

- `phase7_deeptalk_formal_run_summary.csv` if DeepTalk runs.
- `phase7_deep_baseline_status_table.csv` regardless of outcome.
- failure logs under `outputs/phase7/goal2_deep_baseline_closure/logs/`.

Required status columns:

- method;
- code availability;
- version/source;
- dependency reproducibility;
- input compatibility;
- output level;
- LR-pair ranking support;
- cell/spot-level support;
- runtime/memory status;
- final status;
- exact failure reason or output path.

Completion criteria:

- DeepTalk is either run on at least one matched benchmark panel or has a reproducible failure/status record.
- HoloNet, stMLnet, CytoSignal, and NICHES have structured status entries.
- The manuscript can explain what was run, what failed, and why.

Supported claim if passed:

> Deep spatial CCC baselines were either formally evaluated or audited with reproducible failure/status reasons.

Limitations:

- A failed or incompatible method should not be described as inferior.

### Goal 3: Benchmark Realism Figure

Quantify whether semi-synthetic benchmarks preserve key ST statistics.

Statistics:

- library size distribution;
- expression sparsity;
- mean-variance relationship;
- nearest-neighbor distance distribution;
- spatial density;
- Moran's I distribution;
- LR expression distribution;
- PCA/UMAP overlap between real and semi-synthetic backgrounds.

Required outputs:

- `phase7_benchmark_realism_metrics.csv`
- `phase7_benchmark_realism_summary.csv`
- `fig_phase7_benchmark_realism.{pdf,svg,tiff}`
- panel-level source tables.

Completion criteria:

- Real and semi-synthetic versions are compared for at least lymph node, tonsil, and melanoma backgrounds if available.
- Each statistic has a source table.
- The figure states which statistics are preserved and which are not.

Supported claim if passed:

> The semi-synthetic benchmark preserves major spatial and expression properties of the real ST backgrounds used for injection.

Limitations:

- Real-background simulation still contains injected truth and should not be described as native real-tissue ground truth.

### Goal 4: Extend SpatialSelf-specific Discovery Beyond Lymph Node

Repeat the Phase 6 discovery logic in tonsil or melanoma.

Strict candidate rule:

- SpatialSelf rank <= 10;
- COMMOT / CellChat / SpaTalk / CellNEST rank > 10 or absent;
- not high-expression trivial;
- true injected/prior-supported LR if the panel is semi-synthetic;
- weak by expression or context-specific by spatial/niche enrichment.

Required validation:

- region/niche enrichment;
- sender-receiver context specificity;
- downstream target validation from Goal 1 or explicit missing-target reason;
- permutation/FDR;
- negative controls;
- figure/source tables.

Required outputs:

- `phase7_cross_background_spatialself_only_candidates.csv`
- `phase7_cross_background_candidate_validation.csv`
- `phase7_cross_background_candidate_fdr.csv`
- at least one figure panel/source table package.

Completion criteria:

- At least one tonsil or melanoma candidate passes the strict screen and validation criteria.
- If no candidate passes, the result is recorded as negative evidence and not replaced with a hand-picked case.

Supported claim if passed:

> SpatialSelf-specific weak/context-specific LR discovery is not limited to the lymph-node expanded benchmark.

Limitations:

- A second benchmark-supported case is stronger than one case but still not a universal native-tissue discovery claim.

### Goal 5: Add Cross-platform Real Data

Add at least one new platform beyond the current real-background panel.

Priority platforms:

- Xenium human lymph node;
- Visium HD or CRC;
- CosMx melanoma;
- MERFISH / seqFISH / Slide-seqV2 as optional high-resolution extensions.

Minimum analysis per selected platform:

- dataset manifest and preprocessing notes;
- cell/spot coordinates and expression matrix;
- cell type or region/niche annotations if available;
- positive/negative LR control panel;
- SpatialSelf predictions;
- retained baseline predictions or status entries;
- region/niche enrichment;
- downstream target response where target sets are available;
- permutation/FDR or a documented reason why FDR is not feasible;
- runtime/memory summary;
- source tables.

Required outputs:

- `phase7_cross_platform_dataset_manifest.csv`
- `phase7_cross_platform_control_panel.csv`
- `phase7_cross_platform_validation_summary.csv`
- `phase7_cross_platform_runtime_summary.csv`

Completion criteria:

- At least one new platform has a complete manifest, predictions, validation summary, and source tables.
- Dataset-specific ground truth limitations are stated explicitly.

Supported claim if passed:

> SpatialSelf can be evaluated on additional spatial platforms and supports biologically plausible CCC profiles beyond the original benchmark backgrounds.

Limitations:

- Without native ground truth, real-platform validation relies on controls, enrichment, target response, and FDR rather than complete truth labels.

## Phase 7 Final Success Claim

If all five goals pass, the manuscript can say:

> SpatialSelf is not only effective in realistic benchmarks, but also prioritizes weak or context-specific CCC with downstream functional support across tissue backgrounds and spatial platforms.

Do not claim:

- SpatialSelf discovers all true CCC in native tissues.
- SpatialSelf is universally best across all methods, datasets, and metrics.
- Target-signature validation is equivalent to perturbation validation.
- Failed baselines are biologically inferior.

## Minimum Viable Phase 7

If time is limited, the minimum viable package is:

1. lymph-node target-signature validation including `CD48-CD244`;
2. DeepTalk formal run or structured failure table;
3. benchmark realism figure for at least lymph node, tonsil, and melanoma;
4. one additional SpatialSelf-specific weak/context candidate in tonsil or melanoma.

