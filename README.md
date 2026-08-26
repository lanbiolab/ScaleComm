# ScaleComm

ScaleComm prioritizes candidate ligand-receptor sender-receiver events in
spatial transcriptomics data using LR-specific multi-scale graph-diffusion
context. The paper-linked resources in this repository provide an executable
SpaMosaic tonsil workflow, released analysis source data, figure reconstruction
scripts and an interactive result Explorer.

## Quick Start

Create the tested environment and install ScaleComm:

```bash
conda env create -f reproducibility/environment.yml
conda activate scalecomm-reproduction
pip install -e .
```

The tested environment uses Python 3.12, NumPy 1.26.4, pandas 2.3.3, PyTorch
2.7.1 and PyTorch Geometric 2.7.0.

## Paper-Linked Tonsil Workflow

The end-to-end SpaMosaic tonsil A1 workflow starts with the public RNA+ADT h5ad
input and performs preprocessing, candidate construction, weak supervision,
de novo ScaleComm training, event scoring, receiver validation and figure-ready
export.

```bash
bash reproducibility/tonsil_a1/run_end_to_end.sh \
  /path/to/tonsil_A1_rna_adt.h5ad cuda
```

Use `cpu` instead of `cuda` where necessary. The fixed paper-linked
configuration uses a 2,500-cell A1 subset, a 12-NN graph, preparation seed 17,
training seed 7 and 200 epochs.

Read the complete tutorial before running the workflow:

- [End-to-end Tonsil A1 tutorial](docs/END_TO_END_TONSIL_A1_TUTORIAL.md)
- [Workflow configuration](reproducibility/tonsil_a1/config.yaml)
- [Workflow output and troubleshooting guide](reproducibility/tonsil_a1/README.md)
- [SpaMosaic source-data record](examples/tonsil_cxcl13_cxcr5/DATA_SOURCE.md)

Check the compact versioned tonsil receiver table with:

```bash
python reproducibility/verify_reported_statistics.py
```

## Reproducing Reported Analyses

`reproducibility/` contains the configuration manifests, released source-data
tables and local figure-reconstruction scripts for the reported CRC, DSS,
CosMx NSCLC and benchmark analyses.

```text
reproducibility/README.md                 Resource index
reproducibility/manifests/                Dataset and benchmark configurations
reproducibility/artifacts/                Source data and figure scripts
```

Run a reconstruction script from its directory, for example:

```bash
cd reproducibility/artifacts/real_data_threshold_free
python rebuild_figures.py
```

## Interactive Explorer

The self-contained ScaleComm Explorer provides an interactive view of the
paper-linked SpaMosaic tonsil A1 result tables, including LR selection,
cell-type-pair filters, event inspection and PNG/CSV export.

```bash
python3 -m http.server 8502 --bind 0.0.0.0 --directory apps/scalecomm_explorer
```

Open `http://localhost:8502/scalecomm_explorer_tonsil_cxcl13_cxcr5.html`.
Controls, screenshots and data provenance are documented in
[apps/scalecomm_explorer/README.md](apps/scalecomm_explorer/README.md).

## Repository Layout

```text
src/spitialself/        Core ScaleComm implementation
reproducibility/        Tutorials, manifests, source data and figure scripts
apps/scalecomm_explorer/ Paper-linked interactive Explorer
examples/               Compact validation table and data provenance
tests/                  Core software smoke tests
```

The historical Python import path remains `src.spitialself`; the user-facing
method name is ScaleComm.

## License

ScaleComm is released under the MIT License. See [LICENSE](LICENSE).
