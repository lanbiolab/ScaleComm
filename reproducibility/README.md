# Reproducibility Resources

This directory organizes the paper-linked workflow, dataset manifests,
released source-data tables and figure-reconstruction scripts.

| Analysis | Resources | Entry point |
| --- | --- | --- |
| SpaMosaic tonsil A1 | Public h5ad preparation, candidate construction, de novo ScaleComm training, validation and figure-ready export | `tonsil_a1/README.md` |
| CRC Visium HD | Released receiver-level source data and figure reconstruction | `manifests/crc.yaml` |
| MERFISH DSS | Released stage-specific source data and figure reconstruction | `manifests/dss.yaml` |
| CosMx NSCLC | Released receiver-level/control source data and figure reconstruction | `manifests/nsclc.yaml` |
| Semi-synthetic benchmark | Released benchmark/control source data and figure reconstruction | `manifests/benchmark.yaml` |

## Release conventions

The Tonsil A1 entry is the public raw-to-result workflow. The CRC, DSS, CosMx
NSCLC and benchmark entries are released-source-data reproductions: their
manifests identify the source dataset, subset, LR axes, reported run settings,
output tables and reconstruction command, but they do not claim to provide
public raw-data retraining workflows. SHA256 values are used where an artifact
is included in the release archive. Paths are relative to the repository root.

`verify_reported_statistics.py` checks the versioned tonsil validation table
against the expected receiver-response and germinal-center statistics. It is a
guard against divergence between the distributed table, the validation script
and reported rounded values.

The tested environment is specified in `environment.yml` (Python 3.12,
NumPy 1.26.4, pandas 2.3.3, PyTorch 2.7.1 and PyG 2.7.0). Create it with:

```bash
conda env create -f reproducibility/environment.yml
conda activate scalecomm-reproduction
```

For the complete paper-linked walkthrough, begin with `tonsil_a1/README.md`.
For all other analyses, run `rebuild_figures.py` from the corresponding
artifact directory to regenerate the released figures from the adjacent source
tables.
