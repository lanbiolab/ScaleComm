# Key Experiments Index

This file marks manuscript-critical experiments that should be easy to find later.

## Visium HD CRC TGFB1-TGFBR2 Fig5 Experiment

Status: manuscript-critical real-data comparison and downstream-response validation experiment.

Short name to use in discussion: `CRC Fig5`, `Visium HD CRC`, or `TGFB1-TGFBR2 CRC story`.

Scientific role:

- This is the main real-data method-comparison story after the benchmark and DSS time-course figure.
- It tests whether SpatialSelf can prioritize a context-specific CRC communication axis whose spatial localization is more consistent with receiver downstream response than COMMOT or CellNEST.
- The current story focuses on TGFB1-TGFBR2 in CAF/stromal and tumor-stromal boundary context.
- This experiment is one of the main reasons the baseline was changed from the earlier fixed-neighborhood style model to the scale-aware `spatialself_wavelet_scale` model: the Fig5 story needs a method that can learn LR-specific spatial scale and context, not just rank high-expression nearby edges.

Current baseline:

- Formal baseline: `spatialself_wavelet_scale`.
- Display name: `SpatialSelf`.
- Current preferred figure version: `v24`, which replaces the weak v21 response-enrichment boxplot with a target-response hotspot alignment effect-size panel.

Important data/result roots:

- Standardized input:
  `outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/visium_hd_crc_p2crc_rich_targets_subset/`
- New-baseline SpatialSelf prediction:
  `outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_spatialself_scale_v21/spatialself_wavelet_scale/`
- Current validation tables:
  `outputs/current_visium_hd_crc_disagreement_validation_scale_v21/`
- Current preferred figure and source data:
  `outputs/current_visium_hd_crc_disagreement_validation_scale_v21/figures_cellnest_style_g_alignment_effect_v24/`

Key scripts:

- Current preferred plotting script:
  `scripts/current/plot_visium_hd_crc_tgfb1_cellnest_style_story_scale_v24.py`
- Earlier new-baseline plotting versions:
  `scripts/current/plot_visium_hd_crc_tgfb1_cellnest_style_story_scale_v21.py`
  `scripts/current/plot_visium_hd_crc_tgfb1_cellnest_style_story_scale_v22.py`
  `scripts/current/plot_visium_hd_crc_tgfb1_cellnest_style_story_scale_v23.py`

Current v24 figure files:

- Main figure:
  `outputs/current_visium_hd_crc_disagreement_validation_scale_v21/figures_cellnest_style_g_alignment_effect_v24/phase15_visium_hd_crc_tgfb1_tgfbr2_cellnest_style_story.png`
- Vector outputs are in the same directory as `.pdf` and `.svg`.
- Source tables are under:
  `outputs/current_visium_hd_crc_disagreement_validation_scale_v21/figures_cellnest_style_g_alignment_effect_v24/source_data/`
- Figure-specific README:
  `outputs/current_visium_hd_crc_disagreement_validation_scale_v21/figures_cellnest_style_g_alignment_effect_v24/README_fig5_crc_v24.md`

Current story anchors:

- Candidate axis: `TGFB1-TGFBR2`.
- SpatialSelf rank: 2.
- COMMOT rank: 4.
- CellNEST rank: 4.
- SpatialSelf downstream Spearman: 0.1431.
- Best baseline downstream Spearman: 0.03679.
- SpatialSelf BH-FDR: 0.009046.
- Panel g target-response hotspot alignment: SpatialSelf +0.103946 above random, COMMOT +0.001429, CellNEST -0.000245.

Planned Fig5 story:

- Problem setup: in real CRC tissue, biologically meaningful CCC is not simply the closest ligand/receptor coexpression event. A TGF-beta-like axis can be context-specific, spatially patterned, and tied to receiver transcriptional response.
- Method point: SpatialSelf uses LR identity plus graph wavelet multi-scale context and scale-aware gating, so TGFB1-TGFBR2 is allowed to behave differently from short-range or generic coexpression signals.
- Discovery point: in the CRC Visium HD sample, SpatialSelf ranks TGFB1-TGFBR2 high in CAF/stromal and tumor-stromal boundary context, while expression-distance prior alone does not explain the signal.
- Localization point: SpatialSelf hotspots form coherent spatial regions, instead of being only high-expression control regions.
- Independent validation point: downstream TGFB/TGF-beta/ECM response hotspots independently align with SpatialSelf TGFB1-TGFBR2 scores.
- Comparison point: COMMOT and CellNEST can produce spatial signals, but for this axis their localization is less aligned with the downstream-response hotspots.
- Claim boundary: this supports a biologically plausible, downstream-response-consistent CRC communication axis; it is not causal perturbation proof.

Future optimization notes:

- Re-check the TGFB/TGF-beta/ECM target signature gene list and keep a frozen version in source data.
- Consider adding a second CRC axis, such as VEGFA-KDR, if it improves the breadth of the story without diluting TGFB1-TGFBR2.
- Consider replacing panel c or d if they feel too technical; the core panels are tissue context, SpatialSelf localization, downstream response, target-hotspot alignment, and baseline localization.
- Keep panel g as target-response hotspot alignment unless a stronger downstream validation is added.

Interpretation boundary:

- This is a real-data observational validation, not a ground-truth false-positive experiment.
- The defensible claim is that SpatialSelf localization for TGFB1-TGFBR2 is more consistent with receiver downstream response hotspots and CRC niche context than the tested baselines.
- Do not overstate this as proof of causality or perturbation-validated TGF-beta signaling.

## DSS/MERFISH Colitis Time-Course Experiment

Status: manuscript-critical real-data biology experiment.

Short name to use in discussion: `DSS experiment`, `DSS/MERFISH`, or `mouse colon IBD MERFISH`.

Scientific role:

- This is the main time-course biological story for SpatialSelf on mouse colon DSS colitis MERFISH data.
- It follows healthy colon, early DSS response, inflammatory peak, and later repair/remodeling stages.
- The current story focuses on stage-specific communication programs and spatial communication niches, not ground-truth false-positive claims.

Current baseline:

- Formal baseline: `spatialself_wavelet_scale`.
- Display name: `SpatialSelf-scale`.
- Current v21 figure reran the experiment with this scale-aware baseline.

Important data/result roots:

- Input multislice replicates:
  `outputs/current_mouse_colon_ibd_merfish/multislice_replicates/`
- Current new-baseline predictions and summaries:
  `outputs/current_mouse_colon_ibd_merfish/multislice_spatialself_scale_v21/`
- Previous v20 figure and source tables:
  `outputs/current_mouse_colon_ibd_merfish/figures/merfish_dss_cellnest_style_story_v20_j_axis_label/`
- Current v21 figure and source tables:
  `outputs/current_mouse_colon_ibd_merfish/figures/merfish_dss_cellnest_style_story_v21_scale_baseline/`

Key scripts:

- Re-run SpatialSelf on DSS multislice replicates:
  `scripts/current/run_mouse_colon_ibd_merfish_stagewise.py`
- v20 plotting script:
  `scripts/current/plot_mouse_colon_merfish_cellnest_style_story.py`
- v21 plotting script using the new baseline:
  `scripts/current/plot_mouse_colon_merfish_cellnest_style_story_v21.py`

Current v21 figure files:

- Main figure:
  `outputs/current_mouse_colon_ibd_merfish/figures/merfish_dss_cellnest_style_story_v21_scale_baseline/mouse_colon_merfish_cellnest_style_story_v21_scale_baseline.png`
- Vector outputs are in the same directory as `.pdf` and `.svg`.
- Single panels are under:
  `outputs/current_mouse_colon_ibd_merfish/figures/merfish_dss_cellnest_style_story_v21_scale_baseline/panels/`
- Source tables are under:
  `outputs/current_mouse_colon_ibd_merfish/figures/merfish_dss_cellnest_style_story_v21_scale_baseline/source_tables/`

Current story anchors:

- D9 inflammatory communication: `Cxcl10-Cxcr3`.
- D21 repair/remodeling-associated communication niche: currently `Tgfb3-Acvrl1` in the v20/v21 story figure.
- Other important trajectories in the new baseline summary include `Ccl5-Ccr5`, `Cxcl12-Cxcr4`, `Il1b-Il1r2`, and TGF/Wnt-related repair axes.

Current v21 QA notes:

- All 8 multislice replicates completed with `spatialself_wavelet_scale`.
- Main PNG size: 6707 x 8242.
- Exported formats: PNG, PDF, SVG, TIFF.
- SVG has no embedded raster image tags in the main figure.
- Single-panel exports: 10 PNG/PDF/SVG/TIFF panels.
- Source tables: 7 CSV files.

Interpretation boundary:

- This is a real-data observational time-course experiment.
- Do not describe it as having ground-truth false-positive labels.
- The defensible framing is stage-specific CCC rewiring, spatial localization, replicate consistency, and biological plausibility.
- Later optimization can revisit LR axis choice, downstream gene validation, and comparison panels, but this experiment should remain a central manuscript case study.
