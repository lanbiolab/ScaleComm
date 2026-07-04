# Fig5 New Dataset Validation Plan

## Purpose

Fig5 should not repeat the benchmark or ablation story. The main role is to show that SpatialSelf generalizes across additional real spatial platforms and tissue contexts.

Working conclusion to test:

> SpatialSelf can identify context-consistent communication programs across independent spatial platforms, not only in the benchmark, CRC Visium HD, or DSS MERFISH examples.

## Figure Position

- Fig1: method overview.
- Fig2: mechanism benchmark and ablations.
- Fig3: Visium HD CRC context-specific case.
- Fig4: MERFISH DSS time-course communication programs.
- Fig5: cross-platform real-data validation on additional datasets.

## Dataset Gate

A dataset enters Fig5 only if it satisfies most of these criteria:

1. Publicly accessible raw or processed data.
2. Spatial coordinates and expression are available.
3. Cell-type, compartment, region, or niche context is available or can be robustly inferred.
4. The gene panel contains interpretable LR axes.
5. SpatialSelf, COMMOT, and CellNEST can be run or audited on a matched candidate universe.
6. Positive biology and negative/high-expression controls can be defined.
7. At least one validation layer is available:
   - region or niche enrichment;
   - downstream target signature;
   - local contrast / permutation FDR;
   - consistency across samples or sections.

Datasets with negative controls ranking above positives should not be used as primary Fig5 evidence, even if they are useful as supplementary stress tests.

## Current Candidate Ranking

### Primary Candidate A: CosMx NSCLC Lung9 Rep1

- Platform: CosMx SMI.
- Tissue: NSCLC FFPE.
- Source:
  - Bruker/NanoString NSCLC public dataset page: https://brukerspatialbiology.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/nsclc-ffpe-dataset/
  - Direct S3 archive: https://nanostring-public-share.s3.us-west-2.amazonaws.com/SMI-Compressed/Lung9_Rep1/Lung9_Rep1+SMI+Flat+data.tar.gz
- Local status: download started at `data/optional/cosmx_nsclc_lung9_rep1/raw/Lung9_Rep1_SMI_Flat_data.tar.gz`.
- Size: about 1.0 GB.
- Why useful:
  - independent commercial platform;
  - tumor-immune microenvironment;
  - broad panel compared with small Xenium panels;
  - expected axes include CXCL9/10-CXCR3, SPP1-CD44, TGFB-related, checkpoint or macrophage-associated pathways if panel supports them.
- Risk:
  - download is slow;
  - parser needs to be written after archive inspection;
  - cell-type annotation quality must be checked.
- Decision:
  - best Fig5 lead candidate if download and standardization succeed.

### Primary Candidate B: Xenium Human Breast Biomarkers

- Platform: 10x Xenium.
- Tissue: human breast cancer FFPE.
- Source:
  - 10x Genomics dataset page: https://www.10xgenomics.com/datasets/xenium-ffpe-human-breast-biomarkers
  - S1_Top small files:
    - https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S1_Top/Human_Breast_Biomarkers_S1_Top_cell_groups.csv
    - https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S1_Top/Human_Breast_Biomarkers_S1_Top_gene_panel.json
    - https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S1_Top/Human_Breast_Biomarkers_S1_Top_analysis_summary.html
  - S1_Top full expression bundle: https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S1_Top/Human_Breast_Biomarkers_S1_Top_xe_outs.zip
- Local status:
  - `cell_groups.csv`, `gene_panel.json`, and analysis summary downloaded for `Human_Breast_Biomarkers_S1_Top`.
  - Full `xe_outs.zip` is available but is about 3.3 GB.
- Key local probe result:
  - S1_Top has about 200,673 cells.
  - Cell groups include proliferative tumor cells, tumor cells, macrophages, stromal cells, perivascular cells, endothelial cells, T lymphocytes, plasma cells, normal epithelial cells, normal myoepithelial cells, and tumor-adjacent myoepithelial cells.
  - Panel has 280 genes.
  - Confirmed candidate LR-compatible genes include `CXCL12`, `CXCR4`, `CXCL10`, `CCR7`, `CCL5`, `EGFR`, `ERBB2`, `EPCAM`, `PDGFD`, `PDGFRB`, `COL4A1`, and `ITGB1`.
- Why useful:
  - strong tumor/peritumor/stroma/immune context;
  - independent Xenium platform;
  - built-in cell groups reduce annotation burden.
- Risk:
  - key axes such as `TGFB1-TGFBR2`, `SPP1-CD44`, and `PDCD1-CD274` are not all present in the 280-gene panel;
  - full expression package is large.
- Decision:
  - good secondary Fig5 platform candidate, especially for `CXCL12-CXCR4`, `PDGFD-PDGFRB`, or ECM/integrin programs.

### Primary Candidate B2: Xenium Prime Human Breast Cancer FFPE

- Platform: 10x Xenium Prime.
- Tissue: human breast cancer FFPE.
- Source:
  - 10x Genomics dataset page: https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-breast-cancer
  - Gene panel: https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Breast_Cancer_FFPE/Xenium_Prime_Breast_Cancer_FFPE_gene_panel.json
  - Full `xe_outs.zip`: https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Breast_Cancer_FFPE/Xenium_Prime_Breast_Cancer_FFPE_xe_outs.zip
- Local status:
  - gene panel downloaded to `data/optional/xenium_prime_breast_cancer_ffpe/raw/Xenium_Prime_Breast_Cancer_FFPE_gene_panel.json`;
  - full expression bundle is about 12.3 GB and was not downloaded.
- Key local probe result:
  - 5,101 genes.
  - Major LR axes present: `TGFB1-TGFBR2`, `CXCL9/10-CXCR3`, `CXCL12-CXCR4`, `CCL19-CCR7`, `PDCD1-CD274`, `CD86-CD28`, `VEGFA-KDR`, and `PDGFD-PDGFRB`.
- Why useful:
  - much stronger LR coverage than the 280-gene breast biomarker panel;
  - independent Xenium high-plex platform;
  - tumor, immune, vascular, and stromal communication can be tested.
- Risk:
  - full bundle is large;
  - cell-type annotation source needs to be checked from the output bundle or generated from markers.
- Decision:
  - high-value follow-up candidate; do not download full bundle until CosMx status is known.

### Candidate C: SPAMOSAIC / Slide-tags Tonsil

- Platform: high-resolution tonsil RNA+ADT.
- Local standardized inputs already exist:
  - `outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_A1_rich_targets_subset`
  - `outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_D1_rich_targets_subset`
- Candidate axes:
  - `CXCL13-CXCR5`;
  - `CCL21-CCR7`;
  - `CD40LG-CD40`;
  - `IL21-IL21R`;
  - `TNFSF13B-TNFRSF13C`.
- Existing useful result:
  - scaling runs show `CXCL13-CXCR5` ranked first in larger subsets while negative controls stay low.
- Why useful:
  - high-resolution immune tissue;
  - clean known biology;
  - already standardized and runnable.
- Risk:
  - this is closer to lymph-node biology and may feel less "new" if we already use lymph node in Fig4/supplement.
- Decision:
  - reliable backup or supplementary cross-platform immune validation.

### Candidate D: Xenium Human Lymph Node

- Platform: 10x Xenium.
- Local standardized input exists.
- Candidate axes:
  - `CCL19-CCR7`;
  - `CD86-CD28`;
  - `CD70-CD27`;
  - `CD274-PDCD1`.
- Current audit:
  - positive controls are present, but high-expression negatives rank too high across several methods.
- Decision:
  - do not use as primary Fig5 success without improving control panel or local-context validation.

### Candidate E: Lung Cancer Visium Subset

- Platform: 10x Visium.
- Local standardized input and SpatialSelf/COMMOT results exist.
- Candidate axes:
  - `CXCL12-CXCR4`;
  - `SPP1-CD44`;
  - `TGFB1-TGFBR2`;
  - `EGF-EGFR`.
- Current audit:
  - high-expression negative controls rank above positives for both SpatialSelf and COMMOT.
- Decision:
  - not suitable as primary Fig5 evidence in the current form.

### Candidate F: Native Human Melanoma Multiome

- Platform: Slide-tags / 10x Multiome human melanoma lymph-node metastasis.
- Local standardized input and multi-method outputs exist.
- Current audit:
  - claim status is `not_clean_native_real_success`;
  - current control panel should not be used for a primary claim.
- Decision:
  - keep as supplementary stress test or redesign controls before reuse.

## Recommended Fig5 Story

If CosMx NSCLC works:

1. Show CosMx NSCLC tissue/cell-type context.
2. Identify a tumor-immune or macrophage-stroma communication program.
3. Compare SpatialSelf, COMMOT, and CellNEST spatial maps on the same LR axis.
4. Validate with receiver target signature or niche enrichment.
5. Add a compact cross-platform summary including CRC Visium HD, DSS MERFISH, tonsil, and CosMx.

If CosMx download/standardization fails:

1. Use Xenium breast as the main new dataset.
2. Focus on `CXCL12-CXCR4`, `PDGFD-PDGFRB`, or ECM/integrin programs.
3. Use SPAMOSAIC tonsil as the clean supporting dataset.

## Immediate Next Steps

1. Complete or resume CosMx Lung9 Rep1 download.
2. Inspect archive file structure.
3. Write a CosMx flat-data standardizer.
4. Build a small LR panel from genes present in the data.
5. Run SpatialSelf first.
6. Run COMMOT and CellNEST on the matched candidate universe.
7. Screen for:
   - context-enriched LR;
   - negative-control suppression;
   - downstream target or pathway response;
   - visualizable spatial pattern.

## Reproducible Commands

Candidate screen:

```bash
python scripts/current/screen_fig5_candidate_datasets.py
```

Resume CosMx Lung9 Rep1 download:

```bash
mkdir -p data/optional/cosmx_nsclc_lung9_rep1/raw
curl -L --continue-at - \
  -o data/optional/cosmx_nsclc_lung9_rep1/raw/Lung9_Rep1_SMI_Flat_data.tar.gz \
  'https://nanostring-public-share.s3.us-west-2.amazonaws.com/SMI-Compressed/Lung9_Rep1/Lung9_Rep1+SMI+Flat+data.tar.gz'
```

Inspect partial or complete archive:

```bash
tar -tzf data/optional/cosmx_nsclc_lung9_rep1/raw/Lung9_Rep1_SMI_Flat_data.tar.gz | sed -n '1,160p'
```

The partial archive already exposes image and FOV-position entries. Cell-level expression and metadata may appear later in the tar stream, so full archive completion is required before standardization.
