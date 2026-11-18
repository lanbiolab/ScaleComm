# Reproduce the P2CRC receiver ranking

This workflow starts with the 8-µm Visium HD files from the P2 human colorectal cancer section, trains ScaleComm from a new initialization, scores candidate events and compares TGFB1–TGFBR2 receiver rankings with the public W12 source table.

## Inputs

Download these three files from [GEO sample GSM8594568](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8594568) into one directory. Leave the two Parquet files gzip-compressed. The deposited metadata supplies the annotations used for ROI sampling.

| File | SHA-256 of the file used for verification |
| --- | --- |
| `GSM8594568_P2CRC_filtered_feature_bc_matrix.h5` | `445f02025d60a0aaa5bb2a0d9e4f5a476741d71a73c7d99c2397169cfba11b73` |
| `GSM8594568_P2CRC_Metadata.parquet.gz` | `f9794fb7b075929530579c98881fc9d5d6d3dacc88969e8645f786a2a7ac5ec4` |
| `GSM8594568_P2CRC_tissue_positions.parquet.gz` | `08a5e2c772bf300a324f71381c6aaaaf568796e32e540a0cf460ed0ed48af586` |

## Run

Activate the repository's [reproduction environment](../environment.yml). From the repository root, run:

```bash
bash reproducibility/crc_p2/run_end_to_end.sh /path/to/GSM8594568 /path/to/new-output cuda
```

The second argument is a new output directory. Use a fresh path for each run so that tables from different runs stay together. Training supports `cpu` as the third argument, although the reference replay below used CUDA.

The default example uses ROI preparation seed 33, training seed 17 and 1,000 epochs. Append `--paper` for the original training seed 20260624 and 200 epochs. See the [seed comparison](../training_seed_examples.md). It selects the 6,000-bin annotated ROI, prepares the rich panel of 72 genes and 7 LR pairs, and scores 91,345 candidates with a 12-nearest-neighbour graph. Training uses the common ScaleComm runner, graph diffusion steps 1/2/4/8 and KL coefficient 1.1. The full configuration is in [run_end_to_end.sh](run_end_to_end.sh).

## Outputs and reference check

The `prepared/` directory contains the cells, expression, LR panel, target signatures, candidates and preparation metadata. The `results/` directory contains the full event predictions, LR scale distributions, model metrics and these two comparisons:

- `receiver_comparison.csv` joins receiver IDs one-to-one and lists the retrained and reference mean incoming scores and ranks. It uses only TGFB1–TGFBR2 events.
- `receiver_comparison.json` reports counts, rank correlation, maximum absolute score difference and top-5% receiver-set overlap. Top lists use descending score and break ties by ascending receiver ID.

The reference is the CRC/TGFB1–TGFBR2 subset of the released [receiver table](../artifacts/real_data_threshold_free/source_data/w12_receiver_scores.csv). The report also computes receiver–response Spearman correlation and hotspot ROC AUC across all 6,000 bins using the 13-gene signature and 200 hotspots. Both modes validate cohort sizes. The `--paper` replay check requires complete candidate coverage, identical receiver IDs, 91,345 candidates, 12,864 focal events, 4,280 focal receivers and Spearman rank correlation of at least 0.99. This tolerance is specific to replaying this dataset and configuration.

A fresh RTX 4090 run using Python 3.12.7, PyTorch 2.7.1+cu126, PyG 2.7.0, NumPy 1.26.4, pandas 2.3.3, SciPy 1.15.1, scikit-learn 1.7.2, h5py 3.15.1 and pyarrow 16.1.0 regenerated all five prepared input tables byte-for-byte against the archived run. Across two fresh CUDA replays of the 200-epoch paper configuration, Spearman correlations with the released table over all 4,280 receivers were 0.99946 and 0.99963; maximum absolute score differences were 0.00788 and 0.00947. The top-5% lists shared 203 and 206 of 214 receivers with the reference (Jaccard 0.9022 and 0.9279, respectively). Repeated GPU training can differ numerically even with a fixed seed.
