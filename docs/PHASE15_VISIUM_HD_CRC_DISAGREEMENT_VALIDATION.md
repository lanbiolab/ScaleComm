# Phase 15: Visium HD CRC Disagreement Validation

## Plain Goal

Use a non-lymph-node real spatial dataset to test whether SpatialSelf can localize a communication axis more consistently with receiver downstream response than COMMOT and CellNEST.

This phase does not treat real tissue as perturbation ground truth. The claim boundary is narrower:

> If methods disagree on the spatial position of an LR signal, the prediction whose receiver hotspot better matches downstream target activation and CRC niche context is more biologically supported.

## Dataset

Primary dataset:

- `outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/visium_hd_crc_p2crc_rich_targets_subset`
- 10x Visium HD human colorectal cancer P2CRC 8 um bins
- Real coordinates, expression, library sparsity, deconvolution labels, and tissue niches
- Not human lymph node

Why this dataset:

- CRC has tumor, stromal, myeloid, immune, endothelial, and epithelial niches.
- Communication axes such as SPP1-CD44, TGFB1-TGFBR2, CXCL12-CXCR4, VEGFA-KDR, and CD274-PDCD1 have interpretable receiver-response programs.
- The task can evaluate spatial localization disagreement, not just LR ranking.

## Methods

Run or reuse matched predictions from:

- SpatialSelf calibrated: current primary method version
- COMMOT: official-style baseline prediction already converted to candidate scores
- CellNEST: official-style baseline prediction already converted to candidate scores

All methods are evaluated on the same candidate IDs and LR universe.

## Fairness Rules

1. Scan all curated positive LR axes before choosing a story case.
2. Include high-expression invalid controls (`ACTB-KRT8`, `COL1A1-EPCAM`) to check false-positive leakage.
3. Use the same receiver bins, target signatures, and hotspot definition for all methods.
4. Do not claim native ground truth in real Visium HD data.
5. A case is only story-ready if ranking/localization, downstream response, context enrichment, and negative-control behavior point in the same direction.

## Metrics

For each method and LR:

- LR rank by mean candidate score
- top-100 and top-1 percent edge concentration
- receiver incoming score map
- Spearman correlation between incoming communication score and receiver target activity
- permutation p-value and BH FDR
- top-N receiver hotspot overlap with target-response hotspot
- active-niche enrichment of predicted hotspot
- negative-control separation

## Candidate Axes

Positive/context axes:

- `SPP1-CD44`: macrophage/stromal/tumor remodeling
- `TGFB1-TGFBR2`: TGF-beta/ECM remodeling
- `CXCL12-CXCR4`: fibroblast/immune chemokine context
- `VEGFA-KDR`: vascular response
- `CD274-PDCD1`: immune checkpoint context

Negative controls:

- `ACTB-KRT8`
- `COL1A1-EPCAM`

## Success Standard

A strong candidate should satisfy most of the following:

- SpatialSelf ranks the LR axis higher than COMMOT and CellNEST, or predicts a clearly different hotspot.
- SpatialSelf hotspot has higher overlap with the receiver target-response hotspot.
- SpatialSelf incoming score has positive downstream correlation with BH FDR <= 0.1.
- SpatialSelf hotspot is enriched in the expected CRC niche.
- SpatialSelf does not also rank high-expression negative controls as strongly.

## Outputs

The first executable scan is:

```bash
python scripts/current/run_visium_hd_crc_disagreement_validation.py
```

Expected output directory:

- `outputs/current_visium_hd_crc_disagreement_validation/`

Key output tables:

- `method_input_status.csv`
- `method_lr_rank_table.csv`
- `downstream_correlation_by_method.csv`
- `hotspot_overlap_with_target_signature.csv`
- `candidate_axis_scan.csv`
- `candidate_axis_recommendations.csv`
- `phase15_visium_hd_crc_disagreement_status.md`

## Figure Direction

If a candidate passes the scan, the later figure should show:

- CRC tissue/niche map
- SpatialSelf, COMMOT, and CellNEST maps for the same LR axis
- receiver target-response hotspot
- quantitative hotspot-target overlap and downstream correlation
- negative-control suppression

The main message should be spatial localization support, not simply "rank first".

## Current First-Pass Result

The first executable scan has been run on the Visium HD CRC P2CRC rich-target subset.

Current strongest candidate:

- `TGFB1-TGFBR2`
- SpatialSelf rank: 2
- COMMOT rank: 4
- CellNEST rank: 4
- SpatialSelf downstream Spearman: 0.243
- Best baseline downstream Spearman: 0.0369
- SpatialSelf BH-FDR: 0.00765
- SpatialSelf hotspot-target Jaccard: 0.0554
- Best baseline hotspot-target Jaccard: 0.0050
- SpatialSelf tumor-stromal-boundary fold: 2.38
- Best baseline tumor-stromal-boundary fold: 0.94

Interpretation:

> In the Visium HD CRC section, TGFB1-TGFBR2 is currently the cleanest disagreement case. SpatialSelf places this axis in a tumor-stromal-boundary pattern that better matches the TGF-beta/ECM receiver response, while COMMOT and CellNEST rank the axis lower and localize the hotspot less consistently with the response signature.

Generated outputs:

- `outputs/current_visium_hd_crc_disagreement_validation/candidate_axis_recommendations.csv`
- `outputs/current_visium_hd_crc_disagreement_validation/downstream_correlation_by_method.csv`
- `outputs/current_visium_hd_crc_disagreement_validation/hotspot_overlap_with_target_signature.csv`
- `outputs/current_visium_hd_crc_disagreement_validation/figures/phase15_visium_hd_crc_tgfb1_tgfbr2_disagreement.pdf`
- `outputs/current_visium_hd_crc_disagreement_validation/figures/phase15_visium_hd_crc_tgfb1_tgfbr2_disagreement.svg`
- `outputs/current_visium_hd_crc_disagreement_validation/figures/phase15_visium_hd_crc_tgfb1_tgfbr2_disagreement.png`
- `outputs/current_visium_hd_crc_disagreement_validation/figures/source_data/`

This is still an observational validation figure. It supports localization consistency, not perturbation-proven causality.

## Revised CellNEST-style Narrative

The preferred story should follow a CellNEST Fig.2-like order, rather than starting with a method-comparison bar chart.

Panel logic:

1. Show the CRC tissue niches and mark the tumor-stromal boundary as the biological context.
2. Within the tumor-stromal boundary receiver region, rank LR axes by SpatialSelf top-edge abundance.
3. Highlight that `TGFB1-TGFBR2` is a leading region-specific axis.
4. Show SpatialSelf edge-score distribution for `TGFB1-TGFBR2` versus other LR/control edges.
5. Show that simple expression-distance prior would favor high-expression negative controls instead, so the axis is not just a high-expression artifact.
6. Map SpatialSelf `TGFB1-TGFBR2` top receivers at progressively stricter thresholds.
7. Map the TGF-beta/ECM receiver-response signature.
8. Quantify receiver-response enrichment in SpatialSelf hotspots.
9. Put COMMOT and CellNEST spatial maps after the biological story, as localization contrasts rather than the opening claim.

Generated CellNEST-style figure:

- `outputs/current_visium_hd_crc_disagreement_validation/figures_cellnest_style/phase15_visium_hd_crc_tgfb1_tgfbr2_cellnest_style_story.pdf`
- `outputs/current_visium_hd_crc_disagreement_validation/figures_cellnest_style/phase15_visium_hd_crc_tgfb1_tgfbr2_cellnest_style_story.svg`
- `outputs/current_visium_hd_crc_disagreement_validation/figures_cellnest_style/phase15_visium_hd_crc_tgfb1_tgfbr2_cellnest_style_story.png`
- `outputs/current_visium_hd_crc_disagreement_validation/figures_cellnest_style/source_data/`

This revised figure should be the starting point for publication polishing. The older matched-evidence figure is still useful as a supplementary quantitative audit, but it should not drive the main biological narrative.
