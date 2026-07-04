# SpatialSelf Interactive Communication Viewer

Lightweight Streamlit browser for SpatialSelf result tables. The main view is a CellNEST-style tissue canvas: choose an LR axis on the right, then inspect SpatialSelf communication lines, sender/receiver endpoints, receiver hotspots, and learned scale/context information directly on spatial coordinates.

## Run

```bash
streamlit run apps/spatialself_viewer/app.py --server.address 0.0.0.0 --server.port 8501
```

The first version reads the CosMx NSCLC source tables from:

```text
outputs/current_fig5_cosmx_nsclc_scale_context/source_tables
```

## What It Shows

- One large spatial tissue canvas with zoom, pan, hover, and high-resolution PNG export.
- Right-side LR selector, search, ranking sort, edge-count slider, and layer toggles.
- SpatialSelf communication lines overlaid on cells/spots.
- Sender and receiver endpoints with hover details for score, distance, cell type, and niche.
- Receiver hotspot overlay from receiver activity or downstream target signature.
- LR-specific learned scale profile and receiver-response scatter.
- CSV downloads for rankings, candidate edges, and receiver hotspots.

## Dependencies

The repository environment already contains `streamlit`, `plotly`, and `pandas` on this machine. If needed:

```bash
pip install streamlit plotly pandas numpy
```
