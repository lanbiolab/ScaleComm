#!/usr/bin/env python
"""Audit method comparability and score behavior for the scDesign3 200-LR benchmark."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_USE_BOTTLENECK", "0")
os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
sys.modules["numexpr"] = None
sys.modules["bottleneck"] = None

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "outputs/reference_based_benchmark/scdesign3_mechanism_200lr"
OUT = BENCH / "benchmark_audit"

METHODS = [
    "SpatialSelf",
    "COMMOT",
    "SpaOTsc",
    "CellPhoneDB",
    "CellChat",
    "NICHES",
    "SpaTalk",
    "CellNEST",
    "DeepTalk",
    "CytoSignal",
]

METHOD_DIR = {
    "SpatialSelf": "spatialself_range_context",
    "COMMOT": "commot",
    "SpaOTsc": "spaotsc",
    "CellPhoneDB": "cellphonedb",
    "CellChat": "cellchat",
    "NICHES": "niches",
    "SpaTalk": "spatalk",
    "CellNEST": "cellnest",
    "DeepTalk": "deeptalk",
    "CytoSignal": "cytosignal",
}

METHOD_FILE = {
    "SpatialSelf": "spatialself_range_context",
    "COMMOT": "commot",
    "SpaOTsc": "spaotsc",
    "CellPhoneDB": "cellphonedb",
    "CellChat": "cellchat",
    "NICHES": "niches",
    "SpaTalk": "spatalk",
    "CellNEST": "cellnest",
    "DeepTalk": "deeptalk",
    "CytoSignal": "cytosignal",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = load_dataset_index()
    score_audit = audit_scores(datasets)
    status = audit_status(datasets)
    truth = audit_truth(datasets)
    summary = summarize_interpretation(score_audit, status)

    score_audit.to_csv(OUT / "score_distribution_and_top5_audit.csv", index=False)
    status.to_csv(OUT / "method_run_and_comparability_audit.csv", index=False)
    truth.to_csv(OUT / "truth_design_audit.csv", index=False)
    summary.to_csv(OUT / "method_interpretation_summary.csv", index=False)
    write_report(score_audit, status, truth, summary)
    print(f"Wrote benchmark audit to {OUT}")


def load_dataset_index() -> pd.DataFrame:
    clean = pd.read_csv(BENCH / "datasets/scdesign3_200lr_manifest.csv")
    clean = clean[["dataset_id", "tissue", "replicate_dir"]].copy()
    clean["noise"] = "clean"
    clean["run_root"] = str(BENCH / "method_runs")

    noisy = pd.read_csv(BENCH / "noisy_datasets/noisy_scdesign3_manifest.csv")
    noisy = noisy.rename(columns={"noise_level": "noise"})
    noisy["noise"] = noisy["noise"].str.replace("_noise", "", regex=False)
    noisy["tissue"] = noisy["source_dataset_id"].str.replace("_scdesign3_200lr", "", regex=False).str.replace("lymph", "lymph_node", regex=False)
    noisy = noisy[["dataset_id", "tissue", "replicate_dir", "noise"]].copy()
    noisy["run_root"] = str(BENCH / "noisy_method_runs")
    return pd.concat([clean, noisy], ignore_index=True)


def audit_scores(datasets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in datasets.itertuples(index=False):
        replicate_dir = ROOT / str(ds.replicate_dir)
        if not replicate_dir.exists():
            replicate_dir = Path(str(ds.replicate_dir))
        candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
        truth = candidates[
            [
                "candidate_id",
                "ground_truth",
                "wrong_range_false_positive",
                "false_positive_high_expression",
            ]
        ].copy()
        truth["ground_truth"] = truth["ground_truth"].astype(int)
        n_candidates = len(truth)
        n_positive = int(truth["ground_truth"].sum())
        top_n = max(1, int(np.ceil(n_candidates * 0.05)))
        run_root = ROOT / str(ds.run_root)
        if not run_root.exists():
            run_root = Path(str(ds.run_root))
        for method in METHODS:
            pred_path = prediction_path(run_root, str(ds.dataset_id), replicate_dir.name, method)
            row = {
                "dataset_id": ds.dataset_id,
                "tissue": ds.tissue,
                "noise": ds.noise,
                "method": method,
                "n_candidates": n_candidates,
                "n_positive": n_positive,
                "positive_fraction": n_positive / n_candidates if n_candidates else np.nan,
                "prediction_exists": pred_path.exists(),
            }
            if pred_path.exists():
                pred = pd.read_csv(pred_path)
                merged = truth.merge(pred[["candidate_id", "score"]], on="candidate_id", how="left")
                merged["score"] = merged["score"].fillna(0.0).astype(float)
                scores = merged["score"].to_numpy(float)
                order = np.lexsort((merged["candidate_id"].astype(str).to_numpy(), -scores))
                top = merged.iloc[order[:top_n]].copy()
                cutoff = float(top["score"].iloc[-1])
                row.update(
                    {
                        "score_min": float(np.nanmin(scores)),
                        "score_max": float(np.nanmax(scores)),
                        "score_mean": float(np.nanmean(scores)),
                        "score_std": float(np.nanstd(scores)),
                        "score_q50": float(np.nanquantile(scores, 0.50)),
                        "score_q95": float(np.nanquantile(scores, 0.95)),
                        "score_q99": float(np.nanquantile(scores, 0.99)),
                        "n_unique": int(pd.Series(scores).nunique(dropna=False)),
                        "zero_fraction": float(np.mean(np.isclose(scores, 0.0))),
                        "one_fraction": float(np.mean(np.isclose(scores, 1.0))),
                        "nonzero_fraction": float(np.mean(~np.isclose(scores, 0.0))),
                        "top5_size": top_n,
                        "top5_cutoff_score": cutoff,
                        "threshold_tie_fraction": float(np.mean(np.isclose(scores, cutoff))),
                        "top5_truth_fraction": float(top["ground_truth"].mean()),
                        "top5_negative_fraction": float(1.0 - top["ground_truth"].mean()),
                        "top5_wrong_range_fraction": float(top["wrong_range_false_positive"].mean()),
                        "top5_high_expression_decoy_fraction": float(top["false_positive_high_expression"].mean()),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def prediction_path(run_root: Path, dataset_id: str, replicate_name: str, method: str) -> Path:
    return run_root / dataset_id / METHOD_DIR[method] / f"{replicate_name}__{METHOD_FILE[method]}_predictions.csv"


def audit_status(datasets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in datasets.itertuples(index=False):
        run_root = ROOT / str(ds.run_root)
        if not run_root.exists():
            run_root = Path(str(ds.run_root))
        for method in METHODS:
            method_dir = run_root / str(ds.dataset_id) / METHOD_DIR[method]
            row = {
                "dataset_id": ds.dataset_id,
                "tissue": ds.tissue,
                "noise": ds.noise,
                "method": method,
                "prediction_exists": any(method_dir.glob("*_predictions.csv")),
                "metrics_exists": any(method_dir.glob("*_metrics.json")),
                "native_or_projected_output": output_level(method),
                "main_figure_status": main_figure_status(method),
                "caveat": method_caveat(method),
            }
            if method == "CellNEST":
                metrics = first_json(method_dir.glob("*_metrics.json"))
                out_path = metrics.get("cellnest_output_path", "")
                row.update(
                    {
                        "official_like_parameters": f"epoch={metrics.get('num_epoch')}; k={metrics.get('k')}; threshold_gene_exp={metrics.get('threshold_gene_exp')}",
                        "native_output_rows": count_lines(Path(out_path)) - 1 if out_path else np.nan,
                    }
                )
            elif method == "DeepTalk":
                status = first_csv(method_dir.glob("*_deeptalk_status.csv"))
                official = first_json(method_dir.glob("*_deeptalk_official_run_status.json"))
                row.update(
                    {
                        "official_like_parameters": "fixed pretrained Sst_Sstr2.model",
                        "native_output_rows": len(status) if status is not None else np.nan,
                        "n_lr_predict_success": sum(bool(x.get("predict_exists")) for x in official) if isinstance(official, list) else np.nan,
                        "n_lr_total": len(official) if isinstance(official, list) else np.nan,
                    }
                )
            elif method == "SpaTalk":
                metrics = first_json(method_dir.glob("*_metrics.json"))
                row["official_like_parameters"] = metrics.get("spatalk_status", "") if metrics else ""
            elif method == "CytoSignal":
                smoke = first_csv(method_dir.glob("*smoke_status.csv"))
                row["official_like_parameters"] = "" if smoke is None else "; ".join(smoke.astype(str).iloc[0].tolist()[:7])
            rows.append(row)
    return pd.DataFrame(rows)


def audit_truth(datasets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in datasets.itertuples(index=False):
        replicate_dir = ROOT / str(ds.replicate_dir)
        if not replicate_dir.exists():
            replicate_dir = Path(str(ds.replicate_dir))
        candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
        for mechanism, group in candidates.groupby("mechanism", observed=True):
            rows.append(
                {
                    "dataset_id": ds.dataset_id,
                    "tissue": ds.tissue,
                    "noise": ds.noise,
                    "mechanism": mechanism,
                    "candidate_edges": len(group),
                    "true_edges": int(group["ground_truth"].sum()),
                    "positive_fraction": float(group["ground_truth"].mean()),
                    "wrong_range_candidates": int(group["wrong_range_false_positive"].sum()),
                    "high_expression_decoys": int(group["false_positive_high_expression"].sum()),
                }
            )
    return pd.DataFrame(rows)


def summarize_interpretation(score_audit: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    clean = score_audit[score_audit["noise"].eq("clean")].copy()
    rows = []
    for method, group in clean.groupby("method", observed=True):
        rows.append(
            {
                "method": method,
                "main_figure_status": main_figure_status(method),
                "mean_top5_truth_fraction_clean": group["top5_truth_fraction"].mean(),
                "mean_top5_high_expression_decoy_fraction_clean": group["top5_high_expression_decoy_fraction"].mean(),
                "mean_top5_wrong_range_fraction_clean": group["top5_wrong_range_fraction"].mean(),
                "mean_zero_fraction_clean": group["zero_fraction"].mean(),
                "mean_one_fraction_clean": group["one_fraction"].mean(),
                "mean_n_unique_clean": group["n_unique"].mean(),
                "interpretation": interpretation(method, group),
            }
        )
    return pd.DataFrame(rows)


def output_level(method: str) -> str:
    if method in {"SpatialSelf", "COMMOT", "SpaOTsc", "CellNEST", "DeepTalk"}:
        return "candidate-edge score or projected candidate-edge score"
    if method in {"CellPhoneDB", "CellChat", "NICHES", "SpaTalk"}:
        return "cell-type or LR-level score projected to candidate edges"
    return "cell-level LR score; no native sender-receiver edge ranking"


def main_figure_status(method: str) -> str:
    if method == "CytoSignal":
        return "exclude_from_main_quantitative_baseline"
    if method in {"DeepTalk", "SpaTalk"}:
        return "show_with_status_caveat"
    return "include"


def method_caveat(method: str) -> str:
    return {
        "SpatialSelf": "target method; range/context-aware edge ranking",
        "COMMOT": "official recommended spatial communication workflow; strong continuous baseline",
        "SpaOTsc": "transport score projected to LR candidate edges",
        "CellPhoneDB": "cell-type-pair score projected to candidate edges; limited edge resolution",
        "CellChat": "cell-type-level communication score projected to candidate edges; sparse edge-level scores",
        "NICHES": "cell-to-cell spatial output projected to benchmark candidates; sparse candidate coverage",
        "SpaTalk": "adapter completed but produced non-informative/constant scores in these runs",
        "CellNEST": "official-like preprocess/train/postprocess completed; top calls dominated by high-expression decoys under range-aware truth",
        "DeepTalk": "official prediction completed with a fixed pretrained model, but candidate scores are nearly binary/saturated",
        "CytoSignal": "adapter smoke returned cell-by-LR scores, not comparable candidate-edge ranking",
    }[method]


def interpretation(method: str, group: pd.DataFrame) -> str:
    zero = group["zero_fraction"].mean()
    one = group["one_fraction"].mean()
    high_decoy = group["top5_high_expression_decoy_fraction"].mean()
    truth = group["top5_truth_fraction"].mean()
    if method == "CytoSignal":
        return "Exclude from main comparison: smoke adapter output is not native sender-receiver edge ranking."
    if method == "DeepTalk":
        return "Low discrimination is driven by saturated 0/1 candidate scores after projection; report as status-caveated baseline."
    if method == "CellNEST" and high_decoy > 0.5:
        return "Run completed, but high-expression decoys dominate selected calls; this benchmark stresses range/context specificity beyond expression-rich edges."
    if zero > 0.95:
        return "Sparse or constant candidate-edge scores make top 5% ranking mostly tie-driven; interpret as limited edge-level compatibility."
    if truth >= 0.5:
        return "Informative ranking under the range-aware benchmark."
    return "Moderate or weak range-aware ranking in this benchmark."


def first_json(paths) -> dict | list:
    for path in paths:
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def first_csv(paths) -> pd.DataFrame | None:
    for path in paths:
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def write_report(score_audit: pd.DataFrame, status: pd.DataFrame, truth: pd.DataFrame, summary: pd.DataFrame) -> None:
    clean_summary = summary[summary["method"].ne("CytoSignal")].copy()
    cyto_status = status[status["method"].eq("CytoSignal")]["main_figure_status"].dropna().unique().tolist()
    text = [
        "# scDesign3 200-LR benchmark audit",
        "",
        "## Bottom line",
        "",
        "- CytoSignal should be excluded from the main quantitative benchmark because the available run is a smoke adapter returning cell-by-LR scores, not a native sender-receiver candidate-edge ranking.",
        "- DeepTalk completed official prediction and projection, but the projected candidate scores are nearly binary/saturated; its poor AUROC/AUPR is mainly a score-resolution/comparability issue.",
        "- CellNEST completed official-like preprocess/train/postprocess runs with large allCCC outputs. Its weak top-5% performance is driven by high-expression decoys dominating selected calls under range-aware truth.",
        "- CellChat, NICHES and SpaTalk have sparse or constant projected candidate scores in this edge-level benchmark, so their top-5% rankings can be tie-driven.",
        "",
        "## Main figure status",
        "",
        clean_summary[
            [
                "method",
                "main_figure_status",
                "mean_top5_truth_fraction_clean",
                "mean_top5_high_expression_decoy_fraction_clean",
                "mean_zero_fraction_clean",
                "mean_n_unique_clean",
                "interpretation",
            ]
        ].to_markdown(index=False),
        "",
        f"CytoSignal status values: {cyto_status}",
        "",
        "## Files",
        "",
        "- score_distribution_and_top5_audit.csv",
        "- method_run_and_comparability_audit.csv",
        "- truth_design_audit.csv",
        "- method_interpretation_summary.csv",
    ]
    (OUT / "benchmark_audit_report.md").write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()
