#!/usr/bin/env python
"""Extract NicheNet ligand-target priors for benchmark LR pairs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


R_RUNNER = PROJECT_ROOT / "scripts/benchmark/extract_nichenet_targets.R"
DEFAULT_MOUSE_MATRIX = (
    "/home/xd/y/BenchmarkBaselines/artifacts/NicheNet/cache/"
    "ligand_target_matrix_nsga2r_final_mouse.rds"
)
DEFAULT_HUMAN_MATRIX = (
    "/home/xd/y/BenchmarkBaselines/artifacts/NicheNet/cache/"
    "ligand_target_matrix_human_3260758.rds"
)


def extract_nichenet_targets(
    lr_pairs_path: Path,
    output_dir: Path,
    *,
    ligand_target_matrix: Path,
    conda_env: str = "nichenet_env",
    top_n: int = 200,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{lr_pairs_path.stem}__nichenet_targets.csv"
    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "Rscript",
        str(R_RUNNER),
        str(lr_pairs_path),
        str(ligand_target_matrix),
        str(output_path),
        str(int(top_n)),
    ]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "NicheNet target extraction failed.\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lr_pairs", help="Benchmark lr_pairs.csv path.")
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/exp07_lymph_node_global_validation/nichenet_targets",
    )
    parser.add_argument("--species", choices=["human", "mouse"], default="human")
    parser.add_argument(
        "--ligand-target-matrix",
        default="",
        help="Override NicheNet ligand-target matrix RDS. Defaults to cached human or mouse prior by --species.",
    )
    parser.add_argument("--conda-env", default="nichenet_env")
    parser.add_argument("--top-n", type=int, default=200)
    args = parser.parse_args()

    output_path = extract_nichenet_targets(
        Path(args.lr_pairs),
        Path(args.output_dir),
        ligand_target_matrix=Path(
            args.ligand_target_matrix
            or (DEFAULT_HUMAN_MATRIX if args.species == "human" else DEFAULT_MOUSE_MATRIX)
        ),
        conda_env=args.conda_env,
        top_n=args.top_n,
    )
    table = pd.read_csv(output_path)
    print(
        table[
            [
                "lr_id",
                "ligand",
                "receptor",
                "n_targets",
                "status",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote NicheNet target priors to {output_path}")


if __name__ == "__main__":
    main()
