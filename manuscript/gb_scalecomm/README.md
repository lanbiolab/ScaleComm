# Genome Biology / Springer Nature LaTeX Draft

This directory is a clean manuscript workspace based on the official Springer Nature LaTeX template, version 3.1, December 2024.

## Files

- `main.tex`: ScaleComm manuscript skeleton.
- `sn-jnl.cls`: Springer Nature article class copied from the official template.
- `sn-vancouver-num.bst` and other `sn-*.bst`: bibliography styles copied from the official template.
- `figures/`: manuscript figures.
- `tables/`: manuscript tables.
- `build/`: local compilation output.

## Compile locally

Use the user-level Tectonic compiler:

```bash
make
```

or directly:

```bash
tectonic -X compile main.tex --outdir build
```

The journal submission system uses pdfLaTeX/TeXLive; this local setup is for drafting and fast checks.
