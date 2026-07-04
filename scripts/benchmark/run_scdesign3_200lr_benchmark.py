#!/usr/bin/env python
"""Run the 200-LR scDesign3 mechanism benchmark with registered baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.evaluate_range_mechanism_benchmark import evaluate_range_mechanism
from scripts.benchmark.run_baseline_comparison import run_baselines_on_dataset
from scripts.benchmark.run_cellnest import run_cellnest_on_replicate
from scripts.benchmark.run_deeptalk_official import run_official_deeptalk_on_replicate
from scripts.benchmark.run_extended_baselines import (
    run_cellphonedb_on_replicate,
    run_cytosignal_on_replicate,
    run_niches_on_replicate,
    run_spaotsc_on_replicate,
)


METHOD_ORDER = [
    "SpatialSelf",
    "COMMOT",
    "SpaOTsc",
    "CellChat",
    "CellPhoneDB",
    "NICHES",
    "CellNEST",
    "DeepTalk",
    "CytoSignal",
    "SpaTalk",
]

METHOD_DIR = {
    "SpatialSelf": "spatialself_range_context",
    "COMMOT": "commot",
    "SpaOTsc": "spaotsc",
    "CellChat": "cellchat",
    "CellPhoneDB": "cellphonedb",
    "NICHES": "niches",
    "CellNEST": "cellnest",
    "DeepTalk": "deeptalk",
    "CytoSignal": "cytosignal",
    "SpaTalk": "spatalk",
}

METHOD_FILE = {
    "SpatialSelf": "spatialself_range_context",
    "COMMOT": "commot",
    "SpaOTsc": "spaotsc",
    "CellChat": "cellchat",
    "CellPhoneDB": "cellphonedb",
    "NICHES": "niches",
    "CellNEST": "cellnest",
    "DeepTalk": "deeptalk",
    "CytoSignal": "cytosignal",
    "SpaTalk": "spatalk",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="outputs/reference_based_benchmark/scdesign3_mechanism_200lr/datasets/scdesign3_200lr_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reference_based_benchmark/scdesign3_mechanism_200lr/method_runs",
    )
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--skip-deeptalk-official", action="store_true")
    parser.add_argument("--cellnest-epochs", type=int, default=500)
    parser.add_argument("--cellnest-threshold-gene-exp", type=float, default=60.0)
    parser.add_argument("--cellnest-k", type=int, default=50)
    parser.add_argument("--spatialself-epochs", type=int, default=50)
    parser.add_argument("--commot-nitermax", type=int, default=500)
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    methods = args.methods or METHOD_ORDER

    detail_rows: list[pd.DataFrame] = []
    mechanism_rows: list[pd.DataFrame] = []
    status_rows: list[dict] = []

    for row in manifest.itertuples(index=False):
        dataset_id = str(row.dataset_id)
        tissue = str(getattr(row, "tissue", ""))
        replicate_dir = _project_path(str(row.replicate_dir))
        dataset_out = output_dir / dataset_id
        dataset_out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {dataset_id} ({tissue}) ===", flush=True)

        for method in methods:
            method = _normalize_method(method)
            if method not in METHOD_ORDER:
                status_rows.append(_status(dataset_id, tissue, method, "unknown_method", ""))
                continue
            pred_path = _prediction_path(dataset_out, replicate_dir.name, method)
            metrics_path = _metrics_path(dataset_out, replicate_dir.name, method)
            if args.reuse_existing and pred_path.exists() and metrics_path.exists():
                status_rows.append(_status(dataset_id, tissue, method, "reused_existing", str(pred_path.relative_to(PROJECT_ROOT))))
                continue
            try:
                _run_method(
                    method,
                    replicate_dir,
                    dataset_out,
                    args=args,
                )
                status_rows.append(_status(dataset_id, tissue, method, "completed", str(pred_path.relative_to(PROJECT_ROOT))))
            except Exception as exc:
                status_rows.append(_status(dataset_id, tissue, method, "failed", f"{type(exc).__name__}: {exc}"))
                print(f"[failed] {dataset_id} {method}: {type(exc).__name__}: {exc}", flush=True)

        recovered = _recover_metrics(dataset_out, dataset_id=dataset_id, tissue=tissue)
        if not recovered.empty:
            detail_rows.append(recovered)
        predictions = {
            method: _prediction_path(dataset_out, replicate_dir.name, method)
            for method in METHOD_ORDER
            if _prediction_path(dataset_out, replicate_dir.name, method).exists()
        }
        if predictions:
            tables = evaluate_range_mechanism(
                replicate_dir,
                output_dir / "range_mechanism" / dataset_id,
                predictions=predictions,
            )
            mechanism = tables["mechanism_range_summary"].copy()
            mechanism["dataset_id"] = dataset_id
            mechanism["tissue"] = tissue
            mechanism_rows.append(mechanism)

    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    mechanism_detail = pd.concat(mechanism_rows, ignore_index=True) if mechanism_rows else pd.DataFrame()
    summary = _summary(detail)
    mechanism_summary = _mechanism_summary(mechanism_detail)
    status = pd.DataFrame(status_rows)

    detail.to_csv(output_dir / "scdesign3_200lr_method_detail.csv", index=False)
    summary.to_csv(output_dir / "scdesign3_200lr_method_summary.csv", index=False)
    mechanism_detail.to_csv(output_dir / "scdesign3_200lr_by_mechanism.csv", index=False)
    mechanism_summary.to_csv(output_dir / "scdesign3_200lr_mechanism_summary.csv", index=False)
    status.to_csv(output_dir / "scdesign3_200lr_method_status.csv", index=False)
    (output_dir / "scdesign3_200lr_run_manifest.json").write_text(
        json.dumps(
            {
                "manifest": args.manifest,
                "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                "methods": methods,
                "cellnest": {
                    "epochs": int(args.cellnest_epochs),
                    "threshold_gene_exp": float(args.cellnest_threshold_gene_exp),
                    "k": int(args.cellnest_k),
                },
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\nWrote 200-LR benchmark outputs to {output_dir}")


def _run_method(method: str, replicate_dir: Path, dataset_out: Path, *, args) -> None:
    if method in {"SpatialSelf", "COMMOT", "CellChat", "SpaTalk"}:
        method_arg = {
            "SpatialSelf": "spatialself_range_context",
            "COMMOT": "commot",
            "CellChat": "cellchat",
            "SpaTalk": "spatalk",
        }[method]
        run_baselines_on_dataset(
            replicate_dir,
            dataset_out,
            methods=(method_arg,),
            cot_nitermax=int(args.commot_nitermax),
            cellchat_nboot=10,
            spatialself_epochs=int(args.spatialself_epochs),
            spatialself_k_nn=12,
            spatialself_max_train_samples=12000,
            spatialself_pseudo_label_strategy="mechanism_niche_expression_distance",
            spatialself_ranking_loss_weight=0.0,
            spatialself_ranking_margin=1.0,
            spatialself_feasibility_loss_weight=0.0,
            spatialself_hard_negative_loss_weight=0.05,
            spatialself_context_contrastive_loss_weight=0.0,
            spatialself_context_contrastive_margin=0.5,
            spatialself_context_contrastive_negative_mode="random_receiver",
            spatialself_null_candidate_loss_weight=0.0,
            spatialself_mechanism_guard_loss_weight=0.03,
            spatialself_lr_cap_positive=50,
            spatialself_lr_cap_negative=200,
            spatialself_no_lr_embedding=False,
            spatialself_no_distance=False,
            spatialself_no_spatial_graph=True,
            spatialself_no_expression_features=False,
            cellnest_num_epoch=80,
            cellnest_hidden=32,
            deeptalk_input_root=None,
            spatalk_per_num=100,
            spatalk_min_pairs=1,
            spatalk_use_synthetic_pathways=False,
        )
    elif method == "SpaOTsc":
        run_spaotsc_on_replicate(replicate_dir, dataset_out / "spaotsc")
    elif method == "CellPhoneDB":
        run_cellphonedb_on_replicate(replicate_dir, dataset_out / "cellphonedb")
    elif method == "NICHES":
        run_niches_on_replicate(replicate_dir, dataset_out / "niches")
    elif method == "CytoSignal":
        run_cytosignal_on_replicate(replicate_dir, dataset_out / "cytosignal")
    elif method == "CellNEST":
        run_cellnest_on_replicate(
            replicate_dir,
            dataset_out / "cellnest",
            num_epoch=int(args.cellnest_epochs),
            threshold_gene_exp=float(args.cellnest_threshold_gene_exp),
            k=int(args.cellnest_k),
        )
    elif method == "DeepTalk":
        if args.skip_deeptalk_official:
            raise RuntimeError("DeepTalk official run was skipped by request")
        run_official_deeptalk_on_replicate(
            replicate_dir,
            dataset_out / "deeptalk",
            deeptalk_input_root=dataset_out / "deeptalk" / "deeptalk_inputs",
            deeptalk_python=Path("/home/Anaconda3/envs/deeptalk_env/bin/python"),
            model_file=Path("/home/xd/y/BenchmarkBaselines/DeepTalk/test/single-cell/Sst_Sstr2.model"),
        )


def _prediction_path(dataset_out: Path, replicate_name: str, method: str) -> Path:
    return dataset_out / METHOD_DIR[method] / f"{replicate_name}__{METHOD_FILE[method]}_predictions.csv"


def _metrics_path(dataset_out: Path, replicate_name: str, method: str) -> Path:
    return dataset_out / METHOD_DIR[method] / f"{replicate_name}__{METHOD_FILE[method]}_metrics.json"


def _recover_metrics(dataset_out: Path, *, dataset_id: str, tissue: str) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        metrics_path = _metrics_path(dataset_out, _replicate_name_from_dataset_out(dataset_out), method)
        if not metrics_path.exists():
            for fallback in (dataset_out / METHOD_DIR[method]).glob("*_metrics.json"):
                metrics_path = fallback
                break
        if not metrics_path.exists():
            continue
        try:
            row = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        row["method"] = method
        row["dataset_id"] = dataset_id
        row["tissue"] = tissue
        rows.append(row)
    return pd.DataFrame(rows)


def _replicate_name_from_dataset_out(dataset_out: Path) -> str:
    for method_dir in METHOD_DIR.values():
        directory = dataset_out / method_dir
        if directory.exists():
            for prediction in directory.glob("*_predictions.csv"):
                suffix = f"__{method_dir}_predictions.csv"
                if prediction.name.endswith(suffix):
                    return prediction.name[: -len(suffix)]
    return ""


def _summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    metrics = [
        "auroc",
        "aupr",
        "top_k_precision",
        "weak_signal_recall_at_threshold",
        "empirical_fdr_at_threshold",
        "calibration_error_10bin",
    ]
    available = [column for column in metrics if column in detail.columns]
    summary = detail.groupby(["dataset_id", "tissue", "method"], as_index=False)[available].mean(numeric_only=True)
    summary["_method_order"] = summary["method"].map({method: i for i, method in enumerate(METHOD_ORDER)}).fillna(999)
    return summary.sort_values(["dataset_id", "_method_order"]).drop(columns=["_method_order"])


def _mechanism_summary(mechanism: pd.DataFrame) -> pd.DataFrame:
    if mechanism.empty:
        return pd.DataFrame()
    summary = (
        mechanism.groupby(["dataset_id", "tissue", "method", "range_class"], as_index=False)[
            ["aupr", "top100_precision", "fdr_top5pct", "wrong_range_rate_top5pct", "too_close_long_range_rate_top5pct"]
        ]
        .mean(numeric_only=True)
    )
    summary["_method_order"] = summary["method"].map({method: i for i, method in enumerate(METHOD_ORDER)}).fillna(999)
    return summary.sort_values(["dataset_id", "_method_order", "range_class"]).drop(columns=["_method_order"])


def _status(dataset_id: str, tissue: str, method: str, status: str, detail: str) -> dict:
    return {"dataset_id": dataset_id, "tissue": tissue, "method": method, "status": status, "detail": detail}


def _normalize_method(value: str) -> str:
    value = str(value)
    lookup = {method.lower(): method for method in METHOD_ORDER}
    lookup.update({METHOD_DIR[method].lower(): method for method in METHOD_ORDER})
    return lookup.get(value.lower(), value)


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
