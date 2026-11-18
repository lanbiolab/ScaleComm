# Reproducibility Resources

This directory organizes the paper-linked workflow, dataset manifests,
released source-data tables and figure-reconstruction scripts.

| Analysis | Resources | Entry point |
| --- | --- | --- |
| SpaMosaic tonsil A1 | Public h5ad preparation, candidate construction, de novo ScaleComm training, validation and figure-ready export | `tonsil_a1/README.md` |
| CRC Visium HD | Prepare public P2CRC counts and coordinates, retrain, and compare receiver scores with the released table | [P2CRC workflow](crc_p2/README.md) |
| MERFISH DSS | Dataset manifest, released source data and stage-specific reconstruction | `manifests/dss.yaml` |
| CosMx NSCLC | Dataset manifest, released source data and spatial-control reconstruction | `manifests/nsclc.yaml` |
| Semi-synthetic benchmark | Benchmark source data, reported outputs, evaluation tables and figure reconstruction | `manifests/benchmark.yaml` |

## Release conventions

Each manifest records the source dataset, processed-input identity, fixed model
configuration, random seed, output tables and figure-reconstruction command.
SHA256 values are used where an artifact is included in the release archive.
Paths in manifests are relative to the repository root unless stated otherwise.

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

For the complete paper-linked walkthrough, begin with
`tonsil_a1/README.md`. Dataset-specific manifests provide the commands and
source-data locations for the remaining reported analyses.
