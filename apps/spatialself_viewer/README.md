# ScaleComm Streamlit Viewer

Interactive web UI for inspecting exported ScaleComm communication results. The viewer focuses on one large spatial canvas with LR selection, cell-type filtering, sender/receiver overlays, communication edges, and hover/click inspection.

## Run

```bash
pip install -r apps/spatialself_viewer/requirements.txt
streamlit run apps/spatialself_viewer/app.py --server.address 0.0.0.0 --server.port 8501
```

By default, the app looks for exported source tables under the local project `outputs/` paths used during development. For a clean deployment, either set `SCALECOMM_VIEWER_ROOT` to the project directory containing exported ScaleComm result tables, or edit the dataset paths in `app.py`.

