# Phase 2 Execution Roadmap

This roadmap operationalizes the six canonical Phase 2 goals in
`docs/PHASE2_REVISED_GOALS.md`. The phase is not considered complete until the
evidence chain is baseline-matched, auditable, and figure/source-table ready.

## Phase 2 Objective

Build a mechanism-layered, real-background, baseline-matched evidence system
showing whether SpatialSelf's main advantage is the tradeoff between weak-signal
recovery and high-expression false-positive suppression, supported by FDR and
downstream biological validation.

This phase should not claim that SpatialSelf is globally best on every metric
unless completed replicate evidence supports that claim.

## Current Conservative Status

- Simulator v2 scaffolding exists, but the final mechanism-layered benchmark is
  still an active deliverable and should remain the first engineering priority.
- Real-background replicated datasets have been generated for 18 matched
  replicates: lymph node, tonsil, and melanoma, six each.
- The replicated tuned baseline panel has completed 18 replicates across lymph
  node, tonsil, and melanoma. Metric outputs are available for SpatialSelf
  wavelet edge prior, SpatialSelf wavelet, COMMOT, CellChat, SpaTalk, and
  CellNEST, with DeepTalk and NICHES retained as reproducible status-only
  methods when full metric outputs are unavailable.
- A shared Goal3/4 truth-contract validator now checks the controlled simulator
  and 18 real-background replicates against the same candidate-level, LR-level,
  cell-type-pair-level, and niche-level truth schema.
- Frozen endpoint and paired diagnostics on the 18-replicate panel favor the
  SpatialSelf wavelet edge-prior representative in the aggregate endpoint, but
  tissue winners are mixed and paired confidence intervals do not support a
  broad dominance claim against all baselines.
- Formal FDR/permutation has a reusable CLI, fixed-null smoke output, one
  200-permutation tonsil smoke, and a representative three-tissue
  100-permutation real-background panel. These support the FDR schema and
  group-level discovery evidence, but they are still not final 500-permutation
  manuscript calibration.
- Manuscript source-table assembly is current enough for tracking source-ready
  claims, but the manuscript remains not final until stronger FDR calibration,
  downstream validation, and figure exports are complete.

## Goal 1: Baseline-Matched Biological Validation

Question: In real tissue backgrounds, does SpatialSelf rank biological controls
ahead of high-expression negative controls better than the same baselines?

Required datasets:

- lymph node;
- tonsil large tile or multiple tiles;
- melanoma TME;
- optional Xenium or Visium HD subset if feasible.

Required methods:

- SpatialSelf wavelet edge prior;
- SpatialSelf wavelet;
- COMMOT;
- CellChat;
- SpaTalk;
- CellNEST;
- DeepTalk status or results;
- NICHES status or results.

Required metrics:

- positive-control rank;
- negative-control rank;
- downstream Spearman;
- empirical FDR;
- high-expression false-positive rate.

Exit criteria:

- Each dataset has the same positive/negative-control definition across methods.
- Each method has either result tables or a reproducible failure/status record.
- The summary states tissue-specific wins and losses, not only selected cases.

Next actions:

- Strengthen downstream target maps for lymph node, tonsil, and melanoma so the
  validation is not limited to an injected receiver-response proxy.
- Expand or replace single-tile tonsil and melanoma panels with larger tiles or
  multiple tiles when local data allow it.
- Keep results baseline-matched: every added positive or negative control must
  be evaluated for all retained methods or documented as status-only.

## Goal 2: Weak-Signal Claim

Question: Does SpatialSelf improve the weak-signal versus false-positive
tradeoff?

Required benchmark design:

- low/moderate ligand and receptor expression;
- realistic sender/receiver spatial relation;
- explicit high-expression false-positive controls;
- contact, short/medium paracrine, and longer-range niche-restricted signals.

Required metrics:

- weak-signal recall;
- AUPR;
- empirical FDR;
- high-expression false-positive rate;
- top-k precision.

Exit criteria:

- The claim is framed as a tradeoff, not necessarily highest weak recall.
- Results are matched across SpatialSelf, COMMOT, CellChat, CellNEST, SpaTalk,
  and NICHES/status.
- The weak-signal panel is summarized separately from ordinary aggregate AUPR.

Next actions:

- Run a weak-signal-specific panel from simulator v2 and real-background v2
  datasets rather than relying only on aggregate controlled benchmark scores.
- Report weak-signal recall together with high-expression false-positive rate,
  empirical FDR, and top-k precision in the same source table.
- Use wording that SpatialSelf improves the tradeoff only when the paired or
  aggregate evidence supports that statement.

## Goal 3: Benchmark Simulator v2

Question: Is the benchmark mechanism-layered instead of reducible to expression
plus distance?

Required mechanisms:

- contact-dependent or juxtacrine;
- short-to-medium paracrine;
- niche-restricted longer-range chemokine-like.

Required LR annotations:

- sender cell type;
- receiver cell type;
- spatial range;
- expression strength;
- background expression;
- weak-signal flag;
- high-expression false-positive-control flag.

Required truth tables:

- candidate-level true edge table;
- LR-pair-level truth table;
- cell-type-pair-level truth table;
- spatial-niche-level truth table;
- weak/strong/mechanism labels.

Exit criteria:

- Validator confirms all required truth tables and columns.
- Realism report compares simulator outputs against real ST statistics.
- Baseline scripts can consume the simulator output without ad hoc conversion.

Next actions:

- Treat the simulator schema and truth tables as frozen before launching new
  large panels.
- Confirm each mechanism has weak and strong signals plus high-expression false
  positives.
- Add a small contract test that fails if required truth columns disappear.
- Keep `scripts/phase2/validate_goal34_truth_contract.py` passing before
  generating new baseline panels or figure source tables.

## Goal 4: Real-Background Semi-Synthetic Benchmark

Question: Does the method still work under real coordinates and expression
noise?

Required preserved properties:

- coordinates;
- cell-type composition;
- library size;
- sparsity;
- local density;
- tissue domain structure.

Required injected scenarios:

- lymph node CCR7/CCL19-like signal;
- tonsil FDC-B and Tfh-B signal;
- melanoma macrophage-T and tumor-immune-boundary signal.

Exit criteria:

- 18 replicated datasets are generated and validated.
- All datasets include injected truth and high-expression negative controls.
- Matched baseline outputs are completed or have reproducible status records.

Next actions:

- Revalidate the existing 18 replicate datasets against the simulator v2 truth
  schema so downstream scripts can merge controlled and real-background panels.
- Add a realism summary that compares library size, sparsity, local density,
  nearest-neighbor distance, LR expression distribution, and zero inflation.
- Keep the injection mechanisms tissue-specific: lymph node CCR7/CCL19-like,
  tonsil FDC-B/Tfh-B, and melanoma macrophage-T or boundary signals.

## Goal 5: Formal FDR and Calibration Module

Question: Can SpatialSelf report calibrated discoveries rather than scores only?

Required nulls:

- spatial permutation;
- receiver permutation;
- LR identity permutation.

Required outputs:

- empirical p-value;
- BH FDR;
- significant LR pairs;
- significant cell-type-pair interactions;
- significant spatial edges or candidate groups.

Exit criteria:

- FDR code is callable as a reusable module or stable command, not only a one-off
  notebook/script.
- Positive controls, negative controls, and null p-value behavior are summarized.
- Manuscript tables distinguish score ranking from statistically significant
  discoveries.

Next actions:

- Use the completed fixed three-null 100-permutation panel on one representative
  replicate per tissue as the current representative FDR evidence before
  attempting expensive 200-500 permutation panels.
- Promote the CLI contract into a test or validator that checks all standard FDR
  output files and required columns.
- Do not describe the current 5-permutation smoke run as calibration evidence.

## Goal 6: Manuscript Evidence and Figure Convergence

Question: Can the evidence chain be read as a compact methods-paper story?

Main figure/source-table chain:

1. method overview;
2. simulator v2;
3. real-background semi-synthetic benchmark;
4. baseline comparison;
5. weak-signal and false-positive tradeoff;
6. ablation;
7. FDR/permutation;
8. lymph node, tonsil, and melanoma biological validation;
9. runtime/scalability.

Exit criteria:

- Source tables are generated from fixed scripts and configs.
- Figure scripts read source tables instead of hidden intermediate files.
- Claims table records supported, partial, and unsupported claims.
- Failed or status-only baselines are documented with reason and scope.

Next actions:

- Build figures from source tables in `outputs/phase2_revised`, not from hidden
  method-specific intermediate outputs.
- Prioritize five figure packages: simulator v2 design, 18-replicate baseline
  comparison, weak-signal/false-positive tradeoff, formal FDR calibration, and
  baseline-matched biological validation.
- Keep the manuscript claim centered on calibrated, LR-aware CCC inference with
  false-positive suppression and downstream validation rather than a generic
  "new GAT model" story.

## Immediate Execution Order

1. Freeze and validate simulator v2 truth schema, including mechanism, weak
   signal, and high-expression false-positive annotations.
2. Revalidate the existing real-background 18-replicate datasets against the
   same truth schema and realism summaries.
3. Run fixed-null FDR panels at 100 permutations for one representative replicate
   per tissue, then decide whether 200-500 permutation panels are affordable.
4. Strengthen downstream validation with broader ligand-target resources and
   baseline-matched source tables for lymph node, tonsil, and melanoma.
5. Rebuild the weak-signal tradeoff tables from the frozen simulator and
   real-background panels.
6. Generate manuscript figure source tables and first-pass figure exports from
   the fixed outputs.

## Model-Freeze Gate

SpatialSelf should remain frozen as the wavelet edge-prior representative unless
a proposed model change improves at least two of the following without a major
loss in the others:

- frozen endpoint score;
- empirical FDR or formal BH-FDR discovery quality;
- high-expression false-positive suppression;
- downstream target correlation;
- weak-signal recall;
- runtime or memory.

If this gate is not met, engineering effort should go to benchmark quality,
FDR, downstream validation, and figures instead of further baseline redesign.
