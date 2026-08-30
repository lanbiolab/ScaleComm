# ScaleComm Interactive Communication Browser: User Guide

## Purpose

This browser provides an interactive view of precomputed ScaleComm results. It supports inspection of candidate communication events on spatial coordinates, LR-specific diffusion-derived scale profiles, receiver-response evidence, and exportable source tables. It is not a model-training interface and does not alter the underlying result tables.

## Installation and launch

Run the following commands from the repository root:

```bash
python -m pip install -r apps/spatialself_viewer/requirements.txt
streamlit run apps/spatialself_viewer/app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://localhost:8501`. The application can be stopped from the terminal with `Ctrl+C`.

The browser uses repository-relative source tables. Before launching, verify that the data directories listed in the application README are present. A missing source-table directory causes the relevant dataset view to be unavailable rather than triggering a new analysis.

## Distributed view

### CosMx NSCLC scale/context validation

The CosMx view provides a searchable LR ranking and an interactive spatial communication map. The side column contains the selected LR's summary evidence:

- ranking sort criteria: target correlation, target lift, mean score, broad-scale weight, and maximum score;
- selected LR metadata: operational mechanism, diffusion-derived scale class, active niche, and dominant receiver niche;
- learned LR-specific diffusion-derived scale profile;
- receiver-response scatter plot; and
- ranked candidate-edge summary.

The map can show communication lines, receiver hotspots, and cell-type labels. The receiver overlay can be switched between the downstream target signature and receiver activity.

## Typical workflow

1. Select the distributed CosMx NSCLC scale/context validation dataset.
2. Choose an LR axis, either by searching its gene symbols or by changing the sort metric.
3. Select the desired receiver overlay and visible layers.
4. Inspect the spatial map using Plotly pan, scroll zoom, hover, and PNG export controls.
5. Interpret the selected LR alongside its diffusion-derived scale profile and receiver-response evidence.
6. Export the active source tables when further analysis is required.

## Spatial map controls

| Control | Function |
|---|---|
| `Communication lines` | Sets the number of highest-scoring candidate edges drawn on the CosMx map. |
| `Receiver overlay` | Displays either downstream target-signature support or receiver activity. |
| `lines`, `hotspots`, `types` | Toggles communication edges, receiver hotspots, and cell-type labels. |
| Plotly pan/zoom/hover | Supports local inspection of spatial positions and event metadata. |
| Plotly PNG export | Exports the current map view at high resolution. |

## Downloads and source tables

Open **Source tables and downloads** below the map to access:

| Export | Contents |
|---|---|
| Current LR ranking | Selected summary columns for all ranked LR axes in the active dataset. |
| Selected LR candidate edges | Candidate sender-receiver events for the active LR, including score, distance, cell type, and niche metadata when available. |
| Selected LR receiver hotspots | Receiver-level hotspot table for the active LR. |

The `Top edges` and `LR table` tabs render the same content for in-browser inspection before export.

## Interpretation boundary

The browser is an exploration and prioritization interface. Raw event scores support relative ranking and visualization within a dataset. They should not be treated as calibrated posterior probabilities, event-level significance calls, direct molecular diffusion distances, or stand-alone evidence of active communication. Interpretations should consider the spatial map, LR-specific diffusion-derived support profile, receiver evidence, tissue context, and orthogonal biological validation.

## Screenshots

The following screenshots are included with the software release after capture:

| File | What it documents |
|---|---|
| `screenshots/viewer_overview.png` | CosMx dataset selection, LR controls, and interactive communication map. |
| `screenshots/viewer_lr_evidence.png` | Selected-LR scale profile, receiver-response evidence, and status metadata. |
| `screenshots/viewer_downloads.png` | Source-table inspection and CSV download controls. |

When the files are present, they should be inserted below this table in the same order. Capture them at the native browser size without cropping the controls or Plotly mode bar.

## Reproducibility notes

The browser reads exported source tables and is deliberately separated from model fitting. Therefore, it provides a reproducible interface for the result tables supplied with the code release. The model-training, benchmark, and figure-generation workflows remain available in the repository-level documentation and scripts.
