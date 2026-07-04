#!/usr/bin/env python
"""Audit access to the 10x Xenium human skin melanoma preview dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
BASE_URL = (
    "https://cf.10xgenomics.com/samples/xenium/1.7.0/"
    "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/"
)
PREFIX = "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE"
FILES = [
    f"{PREFIX}_cells.parquet",
    f"{PREFIX}_cell_feature_matrix.h5",
    f"{PREFIX}_gene_panel.json",
    f"{PREFIX}_analysis.zarr.zip",
    f"{PREFIX}_cell_feature_matrix.zarr.zip",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(GOAL_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in FILES:
        url = BASE_URL + name
        probe = subprocess.run(
            ["curl", "-I", "-L", "--max-time", str(args.timeout), "-sS", url],
            text=True,
            capture_output=True,
            check=False,
        )
        rows.append(
            {
                "dataset": "xenium_human_skin_melanoma_preview",
                "file_name": name,
                "url": url,
                "curl_exit_code": int(probe.returncode),
                "download_accessible": bool(probe.returncode == 0 and " 200 " in probe.stdout.splitlines()[0:5]),
                "stdout_head": "\\n".join(probe.stdout.splitlines()[:8]),
                "stderr_head": "\\n".join(probe.stderr.splitlines()[:8]),
            }
        )

    table = pd.DataFrame(rows)
    status_path = output_dir / "xenium_skin_melanoma_access_status.csv"
    table.to_csv(status_path, index=False)
    manifest = {
        "dataset": "xenium_human_skin_melanoma_preview",
        "dataset_page": "https://www.10xgenomics.com/datasets/human-skin-preview-data-xenium-human-skin-gene-expression-panel-add-on-1-standard",
        "provider": "10x Genomics",
        "technology": "Xenium In Situ",
        "reason": "Melanoma FFPE imaging platform candidate for tumor-immune/CXCR3 cross-platform validation.",
        "n_files_probed": int(len(table)),
        "n_files_download_accessible": int(table["download_accessible"].fillna(False).astype(bool).sum()),
        "status_path": str(status_path.relative_to(PROJECT_ROOT)),
        "latest_probe_note": (
            "File URL pattern was identified from the public 10x dataset page, but current environment curl probes "
            "fail with TLS SSL_ERROR_SYSCALL after proxy CONNECT. This is a data-access limitation, not a method result."
        ),
    }
    manifest_path = output_dir / "xenium_skin_melanoma_access_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
