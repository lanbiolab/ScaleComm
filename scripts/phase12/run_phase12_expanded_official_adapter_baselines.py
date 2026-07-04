#!/usr/bin/env python
"""Run official/adapter-light baselines on Phase 12 expanded atlas replicates.

This is a same-universe run on the Phase 12 886/957 LR candidate tables. COMMOT
is a candidate-level adapter. CellChat and SpaTalk are cell-type/LR scores
projected to candidate edges by the existing benchmark adapters. CellNEST is
attempted with a short smoke configuration and recorded as failed if the native
pipeline cannot complete.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_cellchat import run_cellchat_on_replicate
from scripts.benchmark.run_cellnest import run_cellnest_on_replicate
from scripts.benchmark.run_commot import run_commot_on_replicate
from scripts.benchmark.run_spatalk import run_spatalk_on_replicate


PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
EXPANDED_DIR = PHASE12_DIR / "atlas_model_expanded"
REPLICATE_ROOT = EXPANDED_DIR / "replicates"
OUT_DIR = EXPANDED_DIR / "official_adapter_baselines"
TOP_K_VALUES = (10, 20, 50, 100)


METHODS = (
    "commot",
    "cellchat",
    "spatalk",
    "cellnest",
)

METHOD_DETAIL = {
    "commot": "COMMOT candidate-level optimal-transport adapter",
    "cellchat": "CellChat cell-type pair scores projected to candidate edges",
    "spatalk": "SpaTalk LR/cell-type scores projected to candidate edges",
    "cellnest": "CellNEST native short-run output projected to candidate edges",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    statuses: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for replicate_dir in sorted(p for p in REPLICATE_ROOT.iterdir() if p.is_dir()):
        for method in METHODS:
            status, metric, prediction = run_one(replicate_dir, method)
            statuses.append(status)
            if metric:
                metrics.append(metric)
            if prediction is not None and not prediction.empty:
                predictions.append(prediction)

    status_table = pd.DataFrame(statuses)
    metric_table = pd.DataFrame(metrics)
    prediction_table = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    ranking = build_lr_ranking(prediction_table) if not prediction_table.empty else pd.DataFrame()
    topk = summarize_topk(ranking) if not ranking.empty else pd.DataFrame()
    hard = summarize_hard_negative(ranking) if not ranking.empty else pd.DataFrame()
    dataset_summary = summarize_dataset(topk, hard, ranking, status_table)
    method_summary = summarize_method(dataset_summary, status_table)

    status_table.to_csv(OUT_DIR / "phase12_expanded_official_adapter_status.csv", index=False)
    metric_table.to_csv(OUT_DIR / "phase12_expanded_official_adapter_metrics.csv", index=False)
    prediction_table.to_csv(OUT_DIR / "phase12_expanded_official_adapter_candidate_predictions.csv", index=False)
    ranking.to_csv(OUT_DIR / "phase12_expanded_official_adapter_lr_ranking.csv", index=False)
    topk.to_csv(OUT_DIR / "phase12_expanded_official_adapter_topk_summary.csv", index=False)
    hard.to_csv(OUT_DIR / "phase12_expanded_official_adapter_hard_negative.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "phase12_expanded_official_adapter_dataset_summary.csv", index=False)
    method_summary.to_csv(OUT_DIR / "phase12_expanded_official_adapter_method_summary.csv", index=False)
    write_status(status_table, dataset_summary, method_summary)
    write_manifest()
    print(status_table.to_string(index=False))
    print()
    if not method_summary.empty:
        print(method_summary.to_string(index=False))


def run_one(replicate_dir: Path, method: str) -> tuple[dict[str, object], dict[str, object] | None, pd.DataFrame | None]:
    method_dir = OUT_DIR / method
    log_dir = OUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        if method == "commot":
            metric = run_commot_on_replicate(replicate_dir, method_dir, cot_nitermax=300)
        elif method == "cellchat":
            metric = run_cellchat_on_replicate(replicate_dir, method_dir, nboot=5)
        elif method == "spatalk":
            metric = run_spatalk_on_replicate(replicate_dir, method_dir, per_num=20, min_pairs=1, use_synthetic_pathways=False)
        elif method == "cellnest":
            metric = run_cellnest_on_replicate(replicate_dir, method_dir, num_epoch=1, hidden=16, top_percent=100)
        else:
            raise ValueError(f"Unknown method: {method}")
        prediction_path = Path(metric["prediction_path"])
        prediction = pd.read_csv(prediction_path)
        candidates = pd.read_csv(replicate_dir / "candidate_interactions.csv")
        prediction = annotate_prediction(replicate_dir.name, method, candidates, prediction)
        status = status_row(replicate_dir, method, "completed", METHOD_DETAIL[method], prediction_path)
        return status, metric, prediction
    except Exception as exc:  # noqa: BLE001
        log_path = log_dir / f"{replicate_dir.name}__{method}__failure.log"
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        detail = f"{METHOD_DETAIL[method]} failed: {type(exc).__name__}: {str(exc)[:240]}"
        return status_row(replicate_dir, method, "failed", detail, log_path), None, None


def annotate_prediction(replicate_name: str, method: str, candidates: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    merged = candidates.merge(prediction[["candidate_id", "score"]], on="candidate_id", how="inner", validate="one_to_one")
    if merged.empty:
        return pd.DataFrame()
    merged["dataset_key"] = replicate_name
    merged["method"] = method
    merged["method_label"] = method.upper() if method == "commot" else method.capitalize()
    merged["true_lr"] = merged.get("true_lr", merged.get("ground_truth", False)).astype(bool)
    merged["false_positive_high_expression"] = merged["false_positive_high_expression"].astype(bool)
    return merged[
        [
            "dataset_key",
            "method",
            "method_label",
            "candidate_id",
            "lr_id",
            "ligand",
            "receptor",
            "mechanism",
            "score",
            "true_lr",
            "false_positive_high_expression",
            "weak_signal",
            "sender_cell_type",
            "receiver_cell_type",
            "sender_niche",
            "receiver_niche",
            "distance",
            "ligand_expr",
            "receptor_expr",
        ]
    ].copy()


def build_lr_ranking(prediction: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_key, method, lr_id), sub in prediction.groupby(["dataset_key", "method", "lr_id"], sort=False):
        first = sub.iloc[0]
        rows.append(
            {
                "dataset_key": dataset_key,
                "method": method,
                "method_label": first["method_label"],
                "lr_id": lr_id,
                "lr_pair": f"{first['ligand']}-{first['receptor']}",
                "ligand": first["ligand"],
                "receptor": first["receptor"],
                "mechanism": first["mechanism"],
                "score": float(pd.to_numeric(sub["score"], errors="coerce").fillna(0).mean()),
                "q95_score": float(pd.to_numeric(sub["score"], errors="coerce").fillna(0).quantile(0.95)),
                "n_candidates": int(len(sub)),
                "n_nonzero_score": int(pd.to_numeric(sub["score"], errors="coerce").fillna(0).gt(0).sum()),
                "true_lr": bool(sub["true_lr"].astype(bool).any()),
                "false_positive_high_expression": bool(sub["false_positive_high_expression"].astype(bool).any()),
                "weak_signal": bool(sub["weak_signal"].astype(bool).any()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank"] = out.groupby(["dataset_key", "method"])["score"].rank(ascending=False, method="min").astype(int)
    out = annotate_risk(out)
    return out.sort_values(["dataset_key", "method", "rank", "lr_pair"]).reset_index(drop=True)


def annotate_risk(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["receptor_shared_risk"] = False
    out["ligand_shared_risk"] = False
    for (dataset_key, method), sub in out.groupby(["dataset_key", "method"], sort=False):
        true_receptors = set(sub[sub["true_lr"]]["receptor"].astype(str))
        true_ligands = set(sub[sub["true_lr"]]["ligand"].astype(str))
        mask = out["dataset_key"].eq(dataset_key) & out["method"].eq(method)
        out.loc[mask, "receptor_shared_risk"] = (
            out.loc[mask, "receptor"].astype(str).isin(true_receptors) & ~out.loc[mask, "true_lr"].astype(bool)
        )
        out.loc[mask, "ligand_shared_risk"] = (
            out.loc[mask, "ligand"].astype(str).isin(true_ligands) & ~out.loc[mask, "true_lr"].astype(bool)
        )
    out["negative_or_risk"] = out["false_positive_high_expression"] | out["receptor_shared_risk"] | out["ligand_shared_risk"]
    return out


def summarize_topk(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_key, method), sub in ranking.groupby(["dataset_key", "method"], sort=False):
        sub = sub.sort_values("rank")
        for top_k in TOP_K_VALUES:
            top = sub.head(top_k)
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "method": method,
                    "method_label": top["method_label"].iloc[0],
                    "top_k": top_k,
                    "n_lr": int(len(top)),
                    "n_true_lr": int(top["true_lr"].sum()),
                    "n_high_expression_negative": int(top["false_positive_high_expression"].sum()),
                    "n_receptor_shared_risk": int(top["receptor_shared_risk"].sum()),
                    "n_ligand_shared_risk": int(top["ligand_shared_risk"].sum()),
                    "n_any_negative_or_risk": int(top["negative_or_risk"].sum()),
                    "true_lr_fraction": safe_div(top["true_lr"].sum(), len(top)),
                    "negative_or_risk_fraction": safe_div(top["negative_or_risk"].sum(), len(top)),
                    "control_panel_empirical_fdr": safe_div(
                        top["negative_or_risk"].sum(),
                        max(int(top["negative_or_risk"].sum()) + int(top["true_lr"].sum()), 1),
                    ),
                    "top_lr_pairs": ";".join(top["lr_pair"].astype(str).head(12)),
                }
            )
    return pd.DataFrame(rows)


def summarize_hard_negative(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_key, method), sub in ranking.groupby(["dataset_key", "method"], sort=False):
        positives = sub[sub["true_lr"].astype(bool)]
        negatives = sub[sub["false_positive_high_expression"].astype(bool)]
        for pos in positives.itertuples(index=False):
            for negative_type, pool in [
                ("same_receptor", negatives[negatives["receptor"].astype(str).eq(str(pos.receptor))]),
                ("same_ligand", negatives[negatives["ligand"].astype(str).eq(str(pos.ligand))]),
                ("any_high_expression_negative", negatives),
            ]:
                if pool.empty:
                    continue
                neg = pool.sort_values(["rank", "score"], ascending=[True, False]).iloc[0]
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "method": method,
                        "positive_lr_pair": pos.lr_pair,
                        "positive_rank": int(pos.rank),
                        "negative_pair": neg["lr_pair"],
                        "negative_type": negative_type,
                        "negative_rank": int(neg["rank"]),
                        "negative_beats_positive": bool(int(neg["rank"]) < int(pos.rank)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_dataset(topk: pd.DataFrame, hard: pd.DataFrame, ranking: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_key, method), sub in ranking.groupby(["dataset_key", "method"], sort=False):
        top10 = one_row(topk, dataset_key, method, 10)
        top50 = one_row(topk, dataset_key, method, 50)
        hard_sub = hard[hard["dataset_key"].eq(dataset_key) & hard["method"].eq(method)] if not hard.empty else pd.DataFrame()
        status_sub = status[status["dataset_key"].eq(dataset_key) & status["method"].eq(method)]
        rows.append(
            {
                "dataset_key": dataset_key,
                "method": method,
                "method_label": sub["method_label"].iloc[0],
                "run_status": status_sub["status"].iloc[0] if len(status_sub) else "completed",
                "n_lr": int(len(sub)),
                "top10_true_lr": int(top10.get("n_true_lr", 0)),
                "top10_high_expression_negative": int(top10.get("n_high_expression_negative", 0)),
                "top10_negative_or_risk_fraction": float(top10.get("negative_or_risk_fraction", np.nan)),
                "top10_control_panel_empirical_fdr": float(top10.get("control_panel_empirical_fdr", np.nan)),
                "top50_true_lr": int(top50.get("n_true_lr", 0)),
                "top50_high_expression_negative": int(top50.get("n_high_expression_negative", 0)),
                "top50_negative_or_risk_fraction": float(top50.get("negative_or_risk_fraction", np.nan)),
                "hard_negative_best_comparison_beats": int(hard_sub["negative_beats_positive"].sum()) if len(hard_sub) else 0,
                "hard_negative_best_comparisons": int(len(hard_sub)),
                "hard_negative_best_comparison_beat_fraction": safe_div(
                    int(hard_sub["negative_beats_positive"].sum()) if len(hard_sub) else 0,
                    len(hard_sub),
                ),
                "best_true_lr": best_true_lr(sub),
                "best_true_lr_rank": best_true_lr_rank(sub),
                "adapter_baseline_call": classify_dataset(top10, hard_sub),
            }
        )
    failed = status[~status["status"].eq("completed")]
    for row in failed.itertuples(index=False):
        if ((ranking["dataset_key"].eq(row.dataset_key)) & (ranking["method"].eq(row.method))).any():
            continue
        rows.append(
            {
                "dataset_key": row.dataset_key,
                "method": row.method,
                "method_label": str(row.method).upper() if row.method == "commot" else str(row.method).capitalize(),
                "run_status": row.status,
                "n_lr": 0,
                "top10_true_lr": 0,
                "top10_high_expression_negative": 0,
                "top10_negative_or_risk_fraction": np.nan,
                "top10_control_panel_empirical_fdr": np.nan,
                "top50_true_lr": 0,
                "top50_high_expression_negative": 0,
                "top50_negative_or_risk_fraction": np.nan,
                "hard_negative_best_comparison_beats": 0,
                "hard_negative_best_comparisons": 0,
                "hard_negative_best_comparison_beat_fraction": np.nan,
                "best_true_lr": "",
                "best_true_lr_rank": -1,
                "adapter_baseline_call": "failed_or_unavailable",
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_key", "method"]).reset_index(drop=True)


def summarize_method(dataset_summary: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in dataset_summary.groupby("method", sort=False):
        completed = sub[sub["run_status"].eq("completed")]
        rows.append(
            {
                "method": method,
                "method_label": sub["method_label"].iloc[0],
                "n_datasets": int(len(sub)),
                "n_completed_datasets": int(len(completed)),
                "mean_top10_true_lr": float(completed["top10_true_lr"].mean()) if len(completed) else np.nan,
                "mean_top10_high_expression_negative": float(completed["top10_high_expression_negative"].mean()) if len(completed) else np.nan,
                "mean_top10_negative_or_risk_fraction": float(completed["top10_negative_or_risk_fraction"].mean()) if len(completed) else np.nan,
                "mean_hard_negative_beat_fraction": float(completed["hard_negative_best_comparison_beat_fraction"].mean()) if len(completed) else np.nan,
                "summary_call": method_call(sub),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_completed_datasets", "mean_top10_true_lr"], ascending=[False, False])


def status_row(replicate_dir: Path, method: str, status: str, detail: str, evidence_path: Path) -> dict[str, object]:
    return {
        "dataset_key": replicate_dir.name,
        "method": method,
        "status": status,
        "detail": detail,
        "same_universe_phase12_available": status == "completed",
        "usable_in_phase12_expanded_claim": "adapter_light" if status == "completed" else "no",
        "evidence_file": rel(evidence_path),
    }


def classify_dataset(top10: dict[str, object], hard_sub: pd.DataFrame) -> str:
    if not top10:
        return "failed_or_unavailable"
    true = int(top10.get("n_true_lr", 0))
    high = int(top10.get("n_high_expression_negative", 0))
    risk = float(top10.get("negative_or_risk_fraction", np.nan))
    beat = safe_div(int(hard_sub["negative_beats_positive"].sum()) if len(hard_sub) else 0, len(hard_sub))
    if true >= 3 and high == 0 and risk <= 0.25 and beat <= 0.35:
        return "adapter_topk_support"
    if high > 0 or risk >= 0.5:
        return "adapter_contaminated_or_risk_prone"
    return "adapter_mixed_or_low_recall"


def method_call(sub: pd.DataFrame) -> str:
    if not sub["run_status"].eq("completed").any():
        return "failed_or_unavailable"
    completed = sub[sub["run_status"].eq("completed")]
    clean = completed["adapter_baseline_call"].eq("adapter_topk_support").sum()
    if clean == len(completed) and len(completed) == len(sub):
        return "completed_clean_in_available_datasets"
    if completed["adapter_baseline_call"].astype(str).str.contains("contaminated").any():
        return "completed_but_contaminated_or_risk_prone"
    return "completed_adapter_light_mixed"


def one_row(table: pd.DataFrame, dataset_key: str, method: str, top_k: int) -> dict[str, object]:
    sub = table[table["dataset_key"].eq(dataset_key) & table["method"].eq(method) & table["top_k"].eq(top_k)]
    return sub.iloc[0].to_dict() if len(sub) else {}


def best_true_lr(sub: pd.DataFrame) -> str:
    true = sub[sub["true_lr"].astype(bool)].sort_values("rank")
    return str(true.iloc[0]["lr_pair"]) if len(true) else ""


def best_true_lr_rank(sub: pd.DataFrame) -> int:
    true = sub[sub["true_lr"].astype(bool)].sort_values("rank")
    return int(true.iloc[0]["rank"]) if len(true) else -1


def safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if float(denom) else float("nan")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_status(status: pd.DataFrame, dataset: pd.DataFrame, method: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Expanded Official Adapter Baselines",
        "",
        "## Scope",
        "",
        "This run applies available official/adapter-light baselines to the exact Phase 12 expanded candidate universe. Completed methods can be used as adapter-light same-universe evidence; failed methods are recorded with logs.",
        "",
        "## Run Status",
        "",
    ]
    for row in status.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.method}` `{row.status}`: {row.detail}; evidence=`{row.evidence_file}`"
        )
    if not method.empty:
        lines += ["", "## Method Summary", ""]
        for row in method.itertuples(index=False):
            lines.append(
                f"- `{row.method}` `{row.summary_call}`: completed={int(row.n_completed_datasets)}/{int(row.n_datasets)}, "
                f"mean top10 true={fmt(row.mean_top10_true_lr)}, high-neg={fmt(row.mean_top10_high_expression_negative)}, "
                f"risk={fmt(row.mean_top10_negative_or_risk_fraction)}, hard-neg beat={fmt(row.mean_hard_negative_beat_fraction)}"
            )
    if not dataset.empty:
        lines += ["", "## Dataset Summary", ""]
        for row in dataset.itertuples(index=False):
            lines.append(
                f"- `{row.dataset_key}` `{row.method}` `{row.adapter_baseline_call}`: "
                f"status={row.run_status}, top10 true={int(row.top10_true_lr)}, high-neg={int(row.top10_high_expression_negative)}, "
                f"risk={fmt(row.top10_negative_or_risk_fraction)}, best true={row.best_true_lr}#{int(row.best_true_lr_rank)}"
            )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- These are adapter-light same-universe runs, not necessarily fully tuned official pipelines.",
        "- Cell-type-level methods projected to candidate edges must be described as projected adapter results.",
        "- Failed methods should be reported in the failure/status table, not silently omitted.",
    ]
    (OUT_DIR / "phase12_expanded_official_adapter_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "task": "phase12_expanded_official_adapter_baselines",
        "inputs": [
            "outputs/phase12/atlas_model_expanded/replicates/*/cells.csv",
            "outputs/phase12/atlas_model_expanded/replicates/*/expression.csv",
            "outputs/phase12/atlas_model_expanded/replicates/*/lr_pairs.csv",
            "outputs/phase12/atlas_model_expanded/replicates/*/candidate_interactions.csv",
        ],
        "methods": list(METHODS),
        "outputs": [
            "phase12_expanded_official_adapter_status.csv",
            "phase12_expanded_official_adapter_metrics.csv",
            "phase12_expanded_official_adapter_candidate_predictions.csv",
            "phase12_expanded_official_adapter_lr_ranking.csv",
            "phase12_expanded_official_adapter_topk_summary.csv",
            "phase12_expanded_official_adapter_hard_negative.csv",
            "phase12_expanded_official_adapter_dataset_summary.csv",
            "phase12_expanded_official_adapter_method_summary.csv",
            "phase12_expanded_official_adapter_status.md",
        ],
        "claim_boundary": "Same-universe official/adapter-light baseline run; projected adapters are not fully tuned official end-to-end claims.",
    }
    (OUT_DIR / "phase12_expanded_official_adapter_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def fmt(value: object) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.3g}"


if __name__ == "__main__":
    main()
