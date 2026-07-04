#!/usr/bin/env python
"""Run official DeepTalk prediction on prepared benchmark inputs, then project back to candidates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.prepare_deeptalk_inputs import prepare_replicate
from scripts.benchmark.run_deeptalk import run_deeptalk_projection_on_replicate


DEFAULT_DEEPTALK_PYTHON = "/home/Anaconda3/envs/deeptalk_env/bin/python"
DEFAULT_MODEL_FILE = "/home/xd/y/BenchmarkBaselines/DeepTalk/test/single-cell/Sst_Sstr2.model"


def _replicate_dirs(dataset_root: Path) -> list[Path]:
    dirs = sorted(path for path in dataset_root.glob("replicate_*") if path.is_dir())
    dirs += sorted(path for path in dataset_root.glob("*_replicate_*") if path.is_dir())
    return dirs


def _run_official_predict(
    *,
    deeptalk_python: Path,
    model_file: Path,
    replicate_input_dir: Path,
    lr_dir: Path,
    relation: str,
) -> subprocess.CompletedProcess[str]:
    script = """
import DeepTalk_ST as dt
from pathlib import Path
import sys

base = Path(sys.argv[1])
lr = sys.argv[2]
model_file = sys.argv[3]
lr_dir = base / lr
outdir = lr_dir / "output"

dt.run_predict(
    data_name=lr,
    data_path=str(base),
    outdir=str(outdir),
    pretrained_embeddings=str(lr_dir / "data_pca_pre.emd"),
    model_path=str(base),
    is_trained=True,
    model_name=model_file,
)
"""
    return subprocess.run(
        [
            str(deeptalk_python),
            "-c",
            script,
            str(replicate_input_dir / "data"),
            relation,
            str(model_file),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def run_official_deeptalk_on_replicate(
    replicate_dir: Path,
    output_dir: Path,
    *,
    deeptalk_input_root: Path,
    deeptalk_python: Path,
    model_file: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    replicate_input_dir = deeptalk_input_root / replicate_dir.name
    if not replicate_input_dir.exists():
        prepare_replicate(replicate_dir, deeptalk_input_root)

    manifest_path = replicate_input_dir / "deeptalk_input_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    run_rows = []
    for row in manifest.itertuples(index=False):
        relation = str(row.deeptalk_relation)
        lr_dir = Path(str(row.lr_dir))
        predict_path = lr_dir / "predict_ccc.txt"
        if predict_path.exists():
            run_rows.append(
                {
                    "replicate": replicate_dir.name,
                    "lr_id": str(row.lr_id),
                    "deeptalk_relation": relation,
                    "return_code": 0,
                    "predict_exists": True,
                    "predict_path": str(predict_path),
                    "stdout_tail": "reused existing predict_ccc.txt",
                    "stderr_tail": "",
                }
            )
            continue
        result = _run_official_predict(
            deeptalk_python=deeptalk_python,
            model_file=model_file,
            replicate_input_dir=replicate_input_dir,
            lr_dir=lr_dir,
            relation=relation,
        )
        run_rows.append(
            {
                "replicate": replicate_dir.name,
                "lr_id": str(row.lr_id),
                "deeptalk_relation": relation,
                "return_code": int(result.returncode),
                "predict_exists": bool(predict_path.exists()),
                "predict_path": str(predict_path),
                "stdout_tail": "\n".join(result.stdout.strip().splitlines()[-10:]),
                "stderr_tail": "\n".join(result.stderr.strip().splitlines()[-10:]),
            }
        )

    run_table = pd.DataFrame(run_rows)
    run_table.to_csv(output_dir / f"{replicate_dir.name}__deeptalk_official_run_status.csv", index=False)
    with (output_dir / f"{replicate_dir.name}__deeptalk_official_run_status.json").open("w", encoding="utf-8") as handle:
        json.dump(run_rows, handle, indent=2, ensure_ascii=True)

    metrics = run_deeptalk_projection_on_replicate(
        replicate_dir,
        output_dir,
        deeptalk_input_root=deeptalk_input_root,
    )
    metrics["official_model_file"] = str(model_file)
    metrics["deeptalk_python"] = str(deeptalk_python)
    metrics["n_lr_predict_success"] = int(run_table["predict_exists"].sum())
    metrics["n_lr_total"] = int(len(run_table))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Root containing replicate directories.")
    parser.add_argument(
        "--output-dir",
        default="outputs/reference_based_benchmark/deeptalk_official",
    )
    parser.add_argument(
        "--deeptalk-input-root",
        default="",
        help="Root for prepared DeepTalk inputs. Defaults to <output-dir>/deeptalk_inputs",
    )
    parser.add_argument(
        "--deeptalk-python",
        default=DEFAULT_DEEPTALK_PYTHON,
        help="Python executable inside the DeepTalk environment.",
    )
    parser.add_argument(
        "--model-file",
        default=DEFAULT_MODEL_FILE,
        help="Official DeepTalk .model file used for prediction.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    input_root = Path(args.deeptalk_input_root) if args.deeptalk_input_root else (output_dir / "deeptalk_inputs")
    deeptalk_python = Path(args.deeptalk_python)
    model_file = Path(args.model_file)

    replicate_dirs = _replicate_dirs(dataset_root)
    if not replicate_dirs:
        raise SystemExit(f"No replicate directories found under {dataset_root}")

    metrics = []
    for replicate_dir in replicate_dirs:
        metrics.append(
            run_official_deeptalk_on_replicate(
                replicate_dir,
                output_dir,
                deeptalk_input_root=input_root,
                deeptalk_python=deeptalk_python,
                model_file=model_file,
            )
        )

    metrics_table = pd.DataFrame(metrics)
    metrics_table.to_csv(output_dir / "deeptalk_official_metrics.csv", index=False)
    print(
        metrics_table[
            [
                "replicate",
                "status",
                "n_lr_predict_success",
                "n_lr_total",
                "n_projected_candidates",
                "auroc",
                "aupr",
                "empirical_fdr_at_threshold",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote official DeepTalk outputs to {output_dir}")


if __name__ == "__main__":
    main()
