# New Fig5 Dataset Plan: CosMx NSCLC Lung9

## Decision

Use a new real dataset for Fig5, not CRC and not DSS.

Current first-choice dataset:

- Dataset: NanoString/Bruker CosMx SMI human NSCLC, Lung9 Rep1 / patient 9 slice 1.
- Preferred access route: `STexampleData::CosMx_lungCancer()` if R dependencies can be resolved.
- Fallback access route: NanoString public S3 flat-data archive.
- Optional processed source: Figshare `CosMx human NSCLC (clustered)`, which contains CellCharter niche labels.

## Why This Dataset

- It is a different tissue background from CRC and DSS.
- It is a single-cell/subcellular spatial platform, so it is well suited for testing LR-specific spatial scale.
- The tumor microenvironment contains tumor, immune, macrophage, stromal, and vascular contexts.
- Public documentation explicitly discusses cell-type maps, spatial niches, macrophage spatial context, and ligand-receptor analysis.
- It is biologically aligned with the method's new selling point: different LR axes should have different spatial range and context dependence.

## Intended Fig5 Story

Working title:

SpatialSelf learns scale-specific tumor-immune communication programs in single-cell CosMx NSCLC.

Main claim:

In a single-cell CosMx NSCLC tissue, SpatialSelf-scale separates LR axes with different spatial operating scales and local contexts, identifying tumor-immune and macrophage/stromal communication programs that are not explained by generic nearest-neighbor coexpression alone.

## Candidate Biological Axes

Initial axes to test:

- Immune checkpoint / local contact:
  - `CD274-PDCD1`
  - `CD48-CD244`
- Tumor-immune chemokine / longer-range:
  - `CXCL9-CXCR3`
  - `CXCL10-CXCR3`
  - `CCL5-CCR5`
- Macrophage / matrix remodeling:
  - `SPP1-CD44`
  - `TGFB1-TGFBR2`
  - `CSF1-CSF1R`
- Stromal / vascular:
  - `VEGFA-KDR`
  - `CXCL12-CXCR4`

These are candidates, not fixed claims. The final axis should be chosen by data-supported localization, scale profile, and downstream/context validation.

## Proposed Panel Structure

- a: CosMx NSCLC cell-type or niche map.
- b: SpatialSelf-scale LR-specific scale profile for selected short/contact, medium, and longer-range axes.
- c: Spatial map of a local/contact-like tumor-immune LR axis.
- d: Spatial map of a chemokine-like longer-range LR axis.
- e: Spatial map of a macrophage/stromal remodeling LR axis.
- f: Quantification of predicted edge-distance distributions by LR axis.
- g: Downstream or context validation, such as receiver target-response hotspot alignment or niche enrichment.
- h: COMMOT and CellNEST comparison for the selected main axis or scale separation summary.

## Success Criteria

Minimum:

- Standardized input tables: cells, expression, coordinates, cell type or niche labels, LR candidates.
- SpatialSelf-scale predictions on the dataset.
- COMMOT and CellNEST predictions on the same candidate set.
- A selected LR axis with:
  - coherent spatial localization;
  - clear scale/context behavior;
  - biological plausibility in NSCLC;
  - independent support from niche enrichment or downstream response.

Stronger:

- At least three LR axes showing different learned spatial scales.
- A quantitative scale-separation panel.
- A downstream-response alignment panel similar in spirit to CRC Fig5 panel g, but tailored to NSCLC biology.

## Current Download / Dependency Status

- NanoString S3 source was confirmed reachable only when the stale proxy variables are unset.
- Remote file size for `Lung9_Rep1 SMI Flat data.tar.gz`: `1,078,783,664` bytes.
- Local partial file currently exists under:
  `data/optional/cosmx_nsclc_lung9_rep1/raw/Lung9_Rep1_SMI_Flat_data.tar.gz`
- Local size is partial, not complete.
- The direct S3 download is currently very slow from this machine.
- `STexampleData` would be a cleaner route, but R installation is blocked by `SpatialExperiment -> magick -> ImageMagick++` system dependency.

## Next Practical Steps

1. Try a no-proxy `wget -c` or other resumable download for the NanoString S3 file.
2. Try a user-local conda/R environment with ImageMagick available, then install `STexampleData` and load `CosMx_lungCancer`.
3. If both are slow, use the Figshare processed CellCharter niche dataset if a direct download URL can be resolved.
4. Once the data is loaded, build `outputs/current_fig5_cosmx_nsclc_scale_context/` with standardized inputs, prediction outputs, source tables, and figure scripts.

