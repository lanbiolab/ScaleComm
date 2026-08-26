# Released Source Data and Figure Reconstruction

These directories contain the released source-data tables and scripts needed to
recreate the corresponding reported response figures.

Run each figure script from its own directory:

```bash
python rebuild_figures.py
```

The scripts read local `source_data/` files and write vector and raster figures
under `figures/`. Dataset manifests under `../manifests/` record the associated
configuration and source-data provenance.
