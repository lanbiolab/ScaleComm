# ScaleComm Interactive Communication Browser

## Run Viewer: inspect any completed standardized run

The generic Run Viewer reads the files produced by a single ScaleComm run:
`cells.csv`, `candidate_interactions.csv`, the prediction CSV and the LR
scale-profile CSV. It does not require manuscript source tables. Start it
directly after training by appending `--ui` to `run_scalecomm.py`, for example:

```bash
python reproducibility/tonsil_a1/run_scalecomm.py \
  reproducibility/tonsil_a1/work/prepared/spamosaic_tonsil_A1_rich_targets_subset \
  --output-dir reproducibility/tonsil_a1/work/results/a1_rich_scalecomm_e200 \
  --device cuda --ui
```

The command starts a local Streamlit server and prints its URL. The viewer
supports LR selection, sender/receiver cell-type filtering, top-fraction
display, spatial event maps, receiver hotspots, scale-profile inspection and
CSV export of the displayed event set. Use `--ui-port 8502` or
`--ui-no-browser` when needed. The `--ui` shortcut requires one prepared
dataset directory; launch each replicate separately.

The paper-linked shell workflow accepts the same shortcut:

```bash
bash reproducibility/tonsil_a1/run_end_to_end.sh /path/to/input.h5ad cuda --ui
```

## Paper-linked browser

The ScaleComm interactive communication browser is a lightweight Streamlit application for exploring exported spatial communication results. It is a companion to the manuscript figures: it reads precomputed source tables and does not retrain or modify the model.

The distributed browser view includes:

- CosMx NSCLC scale/context results, with searchable ligand-receptor (LR) rankings;
- pan, zoom, hover, and PNG export for spatial communication maps;
- LR-specific scale profiles, receiver-response support, ranked candidate edges, and receiver hotspots; and
- CSV downloads for the active LR ranking, candidate edges, and receiver hotspots.

## Quick start

From the repository root:

```bash
python -m pip install -r apps/spatialself_viewer/requirements.txt
streamlit run apps/spatialself_viewer/app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://localhost:8501` in a browser. Use a different free port if `8501` is already occupied.

The browser expects the manuscript source tables distributed with this repository. It currently reads:

```text
outputs/current_fig5_cosmx_nsclc_scale_context/source_tables/
```

## Using the browser

1. Select the distributed **CosMx NSCLC scale/context validation** dataset from the left-side dataset control.
2. In the LR control card, choose a ranking criterion, optionally search by gene symbol, and select an LR pair.
3. Adjust the number of communication lines and show or hide cell-type, hotspot, and line layers.
4. Pan, zoom, hover, or export the spatial map using the Plotly mode bar.
5. Inspect the LR-specific scale profile and receiver-response evidence in the evidence column.
6. Open **Source tables and downloads** to inspect or export the current result tables.

Raw ScaleComm scores are intended for within-dataset prioritization and visualization. They are not calibrated probabilities of communication; candidate events remain hypotheses requiring contextual and orthogonal biological validation.

## Documentation and screenshots

The complete browser guide, source-table descriptions, export behaviour, and screenshot guide are in [docs/USER_GUIDE.md](docs/USER_GUIDE.md). Screenshots used in the guide are stored in [docs/screenshots](docs/screenshots).

## Dependencies

The minimal browser dependencies are listed in `requirements.txt`:

```text
streamlit
plotly
pandas
numpy
```
