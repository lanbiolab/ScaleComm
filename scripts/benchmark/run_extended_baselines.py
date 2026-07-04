#!/usr/bin/env python3
"""Run extended CCC baselines on reference benchmark replicate directories.

These adapters are kept separate from the primary benchmark runner because
several methods require projection from their native output level to the
shared candidate-edge universe.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_cellphonedb_adapter_smoke import build_minimal_db, prepare_inputs
from src.spitialself.evaluation import evaluate_predictions_auto

BENCHMARK_PYTHON = Path("/home/xd/.conda/envs/Benchmark/bin/python")
GIOTTO_RSCRIPT = Path("/home/xd/h/spatialdecon/.conda-envs/giotto44/bin/Rscript")
GIOTTO_R_LIBS = PROJECT_ROOT / "outputs/reference_based_benchmark/native_rerun/giotto44_r_libs"
LOCAL_CDD_ROOT = (
    PROJECT_ROOT
    / "outputs/reference_based_benchmark/native_rerun/adapter_smoke_extra_methods/spaotsc/local_cddlib/root/usr"
)


def run_niches_on_replicate(replicate_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{replicate_dir.name}__niches.log"
    cmd = [
        str(GIOTTO_RSCRIPT),
        str(PROJECT_ROOT / "scripts/benchmark/run_niches_adapter_smoke.R"),
        str(replicate_dir),
        str(output_dir),
    ]
    env = _r_env()
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True)
    log_path.write_text(completed.stdout + "\nSTDERR:\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"NICHES failed for {replicate_dir}; see {log_path}")
    raw = pd.read_csv(output_dir / "niches_candidate_scores.csv")
    predictions = pd.DataFrame(
        {
            "candidate_id": raw["candidate_id"],
            "method": "niches",
            "score": pd.to_numeric(raw["niches_score"], errors="coerce").fillna(0.0),
        }
    )
    return _write_predictions_and_metrics(
        replicate_dir,
        output_dir,
        "niches",
        predictions,
        output_level="candidate_edge_scores_projected_from_CellToCellSpatial",
        projection_policy="official RunNICHES CellToCellSpatial custom LR; k=n_cells-1 for full candidate coverage",
        extra={"log_path": str(log_path)},
    )


def run_cytosignal_on_replicate(replicate_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{replicate_dir.name}__cytosignal.log"
    cmd = [
        str(GIOTTO_RSCRIPT),
        str(PROJECT_ROOT / "scripts/benchmark/run_cytosignal_adapter_smoke.R"),
        str(replicate_dir),
        str(output_dir),
    ]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=_r_env(), text=True, capture_output=True)
    log_path.write_text(completed.stdout + "\nSTDERR:\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"CytoSignal failed for {replicate_dir}; see {log_path}")

    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    score = pd.read_csv(output_dir / "cytosignal_cell_lr_scores.csv", index_col=0)
    score.columns = [str(col).replace(".", "-") for col in score.columns]
    values = []
    for row in candidates.itertuples(index=False):
        lr_key = f"{row.ligand}-{row.receptor}"
        values.append(float(score.loc[row.receiver_id, lr_key]) if lr_key in score.columns and row.receiver_id in score.index else 0.0)
    predictions = pd.DataFrame({"candidate_id": candidates["candidate_id"], "method": "cytosignal", "score": values})
    return _write_predictions_and_metrics(
        replicate_dir,
        output_dir,
        "cytosignal",
        predictions,
        output_level="cell_by_lr_score_projected_to_receiver_candidate_edges",
        projection_policy="CytoSignal cell-level LR score assigned to candidate receiver cell; sender-independent projection",
        extra={"log_path": str(log_path)},
    )


def run_cellphonedb_on_replicate(replicate_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cpdb_zip = build_minimal_db(replicate_dir, output_dir / "minimal_cpdb_inputs")
    counts_path, meta_path = prepare_inputs(replicate_dir, output_dir)
    result_dir = output_dir / "cpdb_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    from cellphonedb.src.core.methods import cpdb_analysis_method

    cpdb_analysis_method.call(
        cpdb_file_path=str(cpdb_zip),
        meta_file_path=str(meta_path),
        counts_file_path=str(counts_path),
        counts_data="gene_name",
        output_path=str(result_dir),
        threshold=0.0,
        score_interactions=True,
        threads=1,
        output_suffix="extended",
    )
    score_path = result_dir / "simple_analysis_interaction_scores_extended.txt"
    if not score_path.exists():
        score_path = sorted(result_dir.glob("*interaction_scores*.txt"))[-1]
    cpdb_scores = pd.read_csv(score_path, sep="\t")
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    predictions = _project_cpdb_scores(candidates, cpdb_scores)
    return _write_predictions_and_metrics(
        replicate_dir,
        output_dir,
        "cellphonedb",
        predictions,
        output_level="cell_type_pair_lr_score_projected_to_candidate_edges",
        projection_policy="CellPhoneDB non-statistical interaction_scores from minimal custom CPDB database projected by ligand-receptor and sender|receiver cell type",
        extra={"cpdb_file_path": str(cpdb_zip), "interaction_scores_path": str(score_path)},
    )


def run_spaotsc_on_replicate(replicate_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = output_dir / f"{replicate_dir.name}__spaotsc_inline_runner.py"
    prediction_path = output_dir / f"{replicate_dir.name}__spaotsc_predictions.csv"
    log_path = output_dir / f"{replicate_dir.name}__spaotsc.log"
    runner.write_text(_spaotsc_runner_code(), encoding="utf-8")
    env = os.environ.copy()
    lib = LOCAL_CDD_ROOT / "lib/x86_64-linux-gnu"
    env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
    cmd = [str(BENCHMARK_PYTHON), str(runner), str(replicate_dir), str(prediction_path)]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True)
    log_path.write_text(completed.stdout + "\nSTDERR:\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"SpaOTsc failed for {replicate_dir}; see {log_path}")
    predictions = pd.read_csv(prediction_path)
    return _write_predictions_and_metrics(
        replicate_dir,
        output_dir,
        "spaotsc",
        predictions,
        output_level="cell_cell_lr_score_matrix_projected_to_candidate_edges",
        projection_policy="SpaOTsc spatial_signaling_scoring run for each LR with effect_range=max coordinate distance; candidate score=sender,receiver matrix entry",
        extra={"log_path": str(log_path)},
        prediction_already_written=True,
    )


def _project_cpdb_scores(candidates: pd.DataFrame, cpdb_scores: pd.DataFrame) -> pd.DataFrame:
    lookup = {}
    meta_cols = {
        "id_cp_interaction",
        "interacting_pair",
        "partner_a",
        "partner_b",
        "gene_a",
        "gene_b",
        "secreted",
        "receptor_a",
        "receptor_b",
        "annotation_strategy",
        "is_integrin",
    }
    pair_cols = [col for col in cpdb_scores.columns if col not in meta_cols]
    for _, row_dict in cpdb_scores.iterrows():
        ligand = str(row_dict.get("gene_a", ""))
        receptor = str(row_dict.get("gene_b", ""))
        for pair_col in pair_cols:
            lookup[(ligand, receptor, pair_col)] = float(row_dict[pair_col])
    values = []
    for row in candidates.itertuples(index=False):
        pair_key = f"{row.sender_cell_type}|{row.receiver_cell_type}"
        values.append(lookup.get((row.ligand, row.receptor, pair_key), 0.0))
    return pd.DataFrame({"candidate_id": candidates["candidate_id"], "method": "cellphonedb", "score": values})


def _write_predictions_and_metrics(
    replicate_dir: Path,
    output_dir: Path,
    method: str,
    predictions: pd.DataFrame,
    *,
    output_level: str,
    projection_policy: str,
    extra: dict | None = None,
    prediction_already_written: bool = False,
) -> dict:
    candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
    prediction_path = output_dir / f"{replicate_dir.name}__{method}_predictions.csv"
    metrics_path = output_dir / f"{replicate_dir.name}__{method}_metrics.json"
    predictions = predictions[["candidate_id", "method", "score"]].copy()
    predictions["score"] = pd.to_numeric(predictions["score"], errors="coerce").fillna(0.0)
    if not prediction_already_written:
        predictions.to_csv(prediction_path, index=False)
    metrics = evaluate_predictions_auto(candidates, predictions)
    metrics.update(
        {
            "replicate": replicate_dir.name,
            "method": method,
            "prediction_path": str(prediction_path),
            "output_level": output_level,
            "projection_policy": projection_policy,
        }
    )
    if extra:
        metrics.update(extra)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    return metrics


def _replicate_dirs(dataset_root: Path) -> list[Path]:
    required_files = {"cells.csv", "expression.csv", "lr_pairs.csv", "candidate_interactions.csv"}
    if dataset_root.is_dir() and required_files.issubset({p.name for p in dataset_root.iterdir() if p.is_file()}):
        return [dataset_root]
    dirs = sorted(dataset_root.glob("replicate_*"))
    dirs += sorted(dataset_root.glob("*_replicate_*"))
    return [path for path in dirs if path.is_dir()]


def _summarize(metrics_table: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "false_positive_high_expression_rate_at_threshold",
        "calibration_error_10bin",
    ]
    available = [col for col in metric_columns if col in metrics_table.columns]
    summary = metrics_table.groupby("method", sort=False)[available].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _r_env() -> dict:
    env = os.environ.copy()
    env["R_LIBS_USER"] = str(GIOTTO_R_LIBS)
    env["PROJ_LIB"] = "/home/xd/h/spatialdecon/.conda-envs/giotto44/share/proj"
    return env


def _spaotsc_runner_code() -> str:
    return r'''
from pathlib import Path
import collections, collections.abc
for name in ["Iterable", "Mapping", "MutableMapping", "Sequence"]:
    if not hasattr(collections, name):
        setattr(collections, name, getattr(collections.abc, name))

import sys
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from spaotsc import SpaOTsc

replicate_dir = Path(sys.argv[1])
prediction_path = Path(sys.argv[2])
expr = pd.read_csv(replicate_dir / "expression.csv").set_index("cell_id")
cells = pd.read_csv(replicate_dir / "cells.csv").set_index("cell_id").loc[expr.index]
lr_pairs = pd.read_csv(replicate_dir / "lr_pairs.csv")
candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
coords = cells[["x", "y"]].to_numpy(float)
dmat = distance_matrix(coords, coords)
effect_range = float(np.nanmax(dmat)) if np.isfinite(dmat).any() else 1.0
spsc = SpaOTsc.spatial_sc(sc_data=expr, is_data=expr, is_pos=coords, is_dmat=dmat, sc_dmat=dmat)
spsc.sc_dmat_spatial = dmat
cell_index = {cell_id: i for i, cell_id in enumerate(expr.index)}
matrix_by_lr = {}
for lr in lr_pairs.itertuples(index=False):
    ligand = str(lr.ligand)
    receptor = str(lr.receptor)
    lr_id = str(lr.lr_id)
    if ligand not in expr.columns or receptor not in expr.columns:
        matrix_by_lr[lr_id] = np.zeros((expr.shape[0], expr.shape[0]), dtype=float)
        continue
    try:
        score = spsc.spatial_signaling_scoring(
            ligand,
            [receptor],
            DSgenes_up=[receptor],
            effect_range=effect_range,
        )
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as exc:
        print(f"SPAOTSC_LR_FAILED {lr_id} {type(exc).__name__}: {exc}")
        score = np.zeros((expr.shape[0], expr.shape[0]), dtype=float)
    matrix_by_lr[lr_id] = score

values = []
for row in candidates.itertuples(index=False):
    mat = matrix_by_lr.get(str(row.lr_id))
    if mat is None:
        values.append(0.0)
        continue
    i = cell_index.get(row.sender_id)
    j = cell_index.get(row.receiver_id)
    values.append(float(mat[i, j]) if i is not None and j is not None else 0.0)
out = pd.DataFrame({"candidate_id": candidates["candidate_id"], "method": "spaotsc", "score": values})
out.to_csv(prediction_path, index=False)
print(f"SPAOTSC_EXTENDED_OK {prediction_path} {len(out)}")
'''


RUNNERS = {
    "niches": run_niches_on_replicate,
    "cytosignal": run_cytosignal_on_replicate,
    "cellphonedb": run_cellphonedb_on_replicate,
    "spaotsc": run_spaotsc_on_replicate,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", action="append", choices=sorted(RUNNERS), dest="methods")
    args = parser.parse_args()

    methods = args.methods or list(RUNNERS)
    replicate_dirs = _replicate_dirs(args.dataset_root)
    if not replicate_dirs:
        raise SystemExit(f"No replicate directories found under {args.dataset_root}")

    metrics = []
    status = []
    for method in methods:
        method_dir = args.output_dir / method
        for replicate_dir in replicate_dirs:
            try:
                metrics.append(RUNNERS[method](replicate_dir, method_dir))
                status.append({"method": method, "replicate": replicate_dir.name, "status": "completed", "detail": ""})
            except Exception as exc:
                status.append({"method": method, "replicate": replicate_dir.name, "status": "failed", "detail": f"{type(exc).__name__}: {exc}"})
                print(f"[FAILED] {method} {replicate_dir.name}: {exc}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_table = pd.DataFrame(metrics)
    status_table = pd.DataFrame(status)
    metrics_table.to_csv(args.output_dir / "extended_baseline_metrics.csv", index=False)
    status_table.to_csv(args.output_dir / "extended_baseline_status.csv", index=False)
    if not metrics_table.empty:
        _summarize(metrics_table).to_csv(args.output_dir / "extended_baseline_metrics_summary.csv", index=False)
        columns = [
            "replicate",
            "method",
            "auroc",
            "aupr",
            "top_k_precision",
            "weak_signal_recall_at_threshold",
            "empirical_fdr_at_threshold",
            "output_level",
        ]
        print(metrics_table[columns].to_string(index=False))
    print(f"Wrote extended baseline outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
