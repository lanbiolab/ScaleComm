#!/usr/bin/env python
"""Run CellNEST on benchmark replicates and project outputs to candidates."""

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

from scripts.benchmark.prepare_cellnest_inputs import prepare_replicate
from src.spitialself.evaluation import evaluate_predictions_auto


CELLNEST_ROOT = Path("/home/xd/y/BenchmarkBaselines/CellNEST")


def run_cellnest_on_replicate(
    replicate_dir: Path,
    output_dir: Path,
    *,
    conda_env: str = "Benchmark",
    num_epoch: int = 2,
    hidden: int = 32,
    seed: int = 1,
    top_percent: int = 100,
    threshold_gene_exp: float = 90.0,
    k: int = 12,
) -> dict:
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    input_root = output_dir / "cellnest_inputs"
    summary = prepare_replicate(replicate_dir, input_root)
    dataset_name = summary["dataset_name"]
    model_name = f"CellNEST_{dataset_name}"

    run_root = output_dir / "cellnest_run"
    embedding_root = run_root / "embedding_data"
    model_root = run_root / "model"
    cellnest_output_root = run_root / "output"
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    preprocess_cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        "data_preprocess_CellNEST.py",
        "--data_name",
        dataset_name,
        "--data_from",
        str(Path(summary["h5ad_path"]).resolve()),
        "--data_type",
        "anndata",
        "--database_path",
        str(Path(summary["database_path"]).resolve()),
        "--data_to",
        str(Path(summary["h5ad_path"]).parent / "input_graph" / dataset_name) + "/",
        "--metadata_to",
        str(Path(summary["h5ad_path"]).parent / "metadata" / dataset_name) + "/",
        "--skip_normalize",
        "1",
        "--filter_min_cell",
        "1",
        "--threshold_gene_exp",
        str(float(threshold_gene_exp)),
        "--distance_measure",
        "knn",
        "--k",
        str(int(k)),
    ]
    if summary.get("annotation_path"):
        preprocess_cmd.extend(
            [
                "--use_celltype",
                "1",
                "--celltype_path",
                str(Path(summary["annotation_path"]).resolve()),
            ]
        )
    _run_command(
        preprocess_cmd,
        cwd=CELLNEST_ROOT,
        log_path=log_dir / f"{replicate_dir.name}__preprocess.log",
    )

    train_cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        "run_CellNEST.py",
        "--data_name",
        dataset_name,
        "--model_name",
        model_name,
        "--run_id",
        "1",
        "--num_epoch",
        str(int(num_epoch)),
        "--hidden",
        str(int(hidden)),
        "--manual_seed",
        "yes",
        "--seed",
        str(int(seed)),
        "--training_data",
        str(Path(summary["h5ad_path"]).parent / "input_graph") + "/",
        "--metadata_to",
        str(Path(summary["h5ad_path"]).parent / "metadata") + "/",
        "--embedding_path",
        str(embedding_root) + "/",
        "--model_path",
        str(model_root) + "/",
    ]
    _run_command(train_cmd, cwd=CELLNEST_ROOT, log_path=log_dir / f"{replicate_dir.name}__train.log")

    postprocess_cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        "output_postprocess_CellNEST.py",
        "--data_name",
        dataset_name,
        "--model_name",
        model_name,
        "--total_runs",
        "1",
        "--embedding_path",
        str(embedding_root) + "/",
        "--metadata_from",
        str(Path(summary["h5ad_path"]).parent / "metadata") + "/",
        "--data_from",
        str(Path(summary["h5ad_path"]).parent / "input_graph") + "/",
        "--output_path",
        str(cellnest_output_root) + "/",
        "--top_percent",
        str(int(top_percent)),
        "--output_all",
        "1",
    ]
    _run_command(
        postprocess_cmd,
        cwd=CELLNEST_ROOT,
        log_path=log_dir / f"{replicate_dir.name}__postprocess.log",
    )

    cellnest_output_path = cellnest_output_root / dataset_name / f"{model_name}_allCCC.csv"
    predictions = project_cellnest_output_to_candidates(candidates, cellnest_output_path)

    prediction_path = output_dir / f"{replicate_dir.name}__cellnest_predictions.csv"
    metrics_path = output_dir / f"{replicate_dir.name}__cellnest_metrics.json"
    predictions.to_csv(prediction_path, index=False)
    metrics = evaluate_predictions_auto(candidates, predictions)
    metrics.update(
        {
            "replicate": replicate_dir.name,
            "method": "cellnest",
            "prediction_path": str(prediction_path),
            "cellnest_output_path": str(cellnest_output_path),
            "num_epoch": int(num_epoch),
            "hidden": int(hidden),
            "seed": int(seed),
            "top_percent": int(top_percent),
            "threshold_gene_exp": float(threshold_gene_exp),
            "k": int(k),
        }
    )
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    return metrics


def project_cellnest_output_to_candidates(
    candidates: pd.DataFrame,
    cellnest_output_path: Path,
) -> pd.DataFrame:
    if not cellnest_output_path.exists():
        raise FileNotFoundError(f"CellNEST output not found: {cellnest_output_path}")
    cellnest = pd.read_csv(cellnest_output_path)
    required = {"from_cell", "to_cell", "ligand", "receptor", "attention_score"}
    missing = required - set(cellnest.columns)
    if missing:
        raise ValueError(f"CellNEST output missing columns: {sorted(missing)}")

    scores = cellnest[
        ["from_cell", "to_cell", "ligand", "receptor", "attention_score"]
    ].rename(
        columns={
            "from_cell": "sender_id",
            "to_cell": "receiver_id",
            "attention_score": "score",
        }
    )
    scores["ligand_key"] = scores["ligand"].astype(str).str.upper()
    scores["receptor_key"] = scores["receptor"].astype(str).str.upper()
    scores = scores.groupby(["sender_id", "receiver_id", "ligand_key", "receptor_key"], as_index=False)[
        "score"
    ].max()

    candidate_keys = candidates[
        ["candidate_id", "sender_id", "receiver_id", "ligand", "receptor"]
    ].copy()
    candidate_keys["ligand_key"] = candidate_keys["ligand"].astype(str).str.upper()
    candidate_keys["receptor_key"] = candidate_keys["receptor"].astype(str).str.upper()
    merged = candidate_keys.merge(
        scores,
        on=["sender_id", "receiver_id", "ligand_key", "receptor_key"],
        how="left",
        validate="many_to_one",
    )
    merged["score"] = merged["score"].fillna(0.0).astype(float)
    return pd.DataFrame(
        {
            "candidate_id": merged["candidate_id"].to_numpy(),
            "method": "cellnest",
            "score": merged["score"].to_numpy(),
        }
    )


def _run_command(command: list[str], *, cwd: Path, log_path: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    log_path.write_text(
        "COMMAND:\n"
        + " ".join(command)
        + "\n\nSTDOUT:\n"
        + completed.stdout
        + "\n\nSTDERR:\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}. Log: {log_path}")


def _replicate_dirs(dataset_root: Path) -> list[Path]:
    dirs = sorted(path for path in dataset_root.glob("replicate_*") if path.is_dir())
    dirs += sorted(path for path in dataset_root.glob("*_replicate_*") if path.is_dir())
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Root containing replicate directories.")
    parser.add_argument(
        "--output-dir",
        default="outputs/phase1/exp03_baseline_comparison/cellnest",
    )
    parser.add_argument("--conda-env", default="Benchmark")
    parser.add_argument("--num-epoch", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--top-percent", type=int, default=100)
    parser.add_argument("--threshold-gene-exp", type=float, default=90.0)
    parser.add_argument("--k", type=int, default=12)
    args = parser.parse_args()

    replicate_dirs = _replicate_dirs(Path(args.dataset_root))
    if not replicate_dirs:
        raise SystemExit(f"No replicate directories found under {args.dataset_root}")

    metrics = [
        run_cellnest_on_replicate(
            replicate_dir,
            Path(args.output_dir),
            conda_env=args.conda_env,
            num_epoch=args.num_epoch,
            hidden=args.hidden,
            seed=args.seed,
            top_percent=args.top_percent,
            threshold_gene_exp=args.threshold_gene_exp,
            k=args.k,
        )
        for replicate_dir in replicate_dirs
    ]
    metrics_table = pd.DataFrame(metrics)
    metrics_table.to_csv(Path(args.output_dir) / "cellnest_metrics.csv", index=False)
    print(
        metrics_table[
            [
                "replicate",
                "method",
                "auroc",
                "aupr",
                "top_k_precision",
                "weak_signal_recall_at_threshold",
                "empirical_fdr_at_threshold",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote CellNEST outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
