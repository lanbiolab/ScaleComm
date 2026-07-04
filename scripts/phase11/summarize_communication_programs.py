#!/usr/bin/env python
"""Summarize communication modules as cross-dataset signaling programs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story"
OUT_DIR = BASE_DIR / "communication_program_summary"

DATASETS = {
    "visium_hd_crc": {
        "label": "Visium HD CRC",
        "root": BASE_DIR / "visium_hd_crc_p2crc_subset",
        "summary": "communication_modules_summary.csv",
        "assignments": "communication_modules_lr_assignments.csv",
        "enrichment": "communication_modules_pathway_summary.csv",
        "kind": "crc_pathway",
    },
    "lymph_node_expanded": {
        "label": "Lymph node expanded benchmark",
        "root": BASE_DIR / "lymph_node_phase6_expanded",
        "summary": "lymph_node_modules_summary.csv",
        "assignments": "lymph_node_modules_lr_assignments.csv",
        "enrichment": "lymph_node_modules_mechanism_enrichment.csv",
        "kind": "mechanism",
    },
    "lymph_node_full": {
        "label": "Native human lymph node Visium",
        "root": BASE_DIR / "human_lymph_node_visium_full_subset",
        "summary": "lymph_node_full_modules_summary.csv",
        "assignments": "lymph_node_full_modules_lr_assignments.csv",
        "enrichment": "lymph_node_full_modules_mechanism_enrichment.csv",
        "kind": "mechanism",
    },
}


PROGRAM_FAMILIES = [
    "myeloid-remodeling",
    "chemokine recruitment",
    "angiogenesis",
    "immune checkpoint",
    "ECM/stromal remodeling",
    "contact-dependent immune signaling",
    "inflammatory/paracrine remodeling",
    "high-expression negative",
    "background LR-like decoy",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    enrich_rows = []
    for dataset_key, cfg in DATASETS.items():
        result_dir = cfg["root"] / "results"
        summary_path = result_dir / cfg["summary"]
        assignment_path = result_dir / cfg["assignments"]
        enrichment_path = result_dir / cfg["enrichment"]
        if not summary_path.exists() or not assignment_path.exists():
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": cfg["label"],
                    "status": "missing_module_inputs",
                }
            )
            continue
        summary = pd.read_csv(summary_path)
        assignments = pd.read_csv(assignment_path)
        enrichment = pd.read_csv(enrichment_path) if enrichment_path.exists() else pd.DataFrame()
        rows.extend(summarize_dataset(dataset_key, cfg, summary, assignments, enrichment))
        if not enrichment.empty:
            tmp = enrichment.copy()
            tmp["dataset_key"] = dataset_key
            tmp["dataset"] = cfg["label"]
            enrich_rows.append(tmp)

    programs = pd.DataFrame(rows)
    programs = programs.sort_values(["dataset_key", "method", "module"]).reset_index(drop=True)
    spatialself = programs[programs["method"].eq("spatialself")].copy()
    family_matrix = build_family_matrix(spatialself)
    programs.to_csv(OUT_DIR / "phase11_communication_program_summary.csv", index=False)
    spatialself.to_csv(OUT_DIR / "phase11_spatialself_communication_program_summary.csv", index=False)
    family_matrix.to_csv(OUT_DIR / "phase11_spatialself_program_family_matrix.csv", index=False)
    if enrich_rows:
        pd.concat(enrich_rows, ignore_index=True).to_csv(OUT_DIR / "phase11_communication_program_enrichment_long.csv", index=False)
    manifest = {
        "datasets": {key: cfg["label"] for key, cfg in DATASETS.items()},
        "program_families": PROGRAM_FAMILIES,
        "outputs": [
            "phase11_communication_program_summary.csv",
            "phase11_spatialself_communication_program_summary.csv",
            "phase11_spatialself_program_family_matrix.csv",
            "phase11_communication_program_enrichment_long.csv",
        ],
        "claim_boundary": (
            "Program summaries combine LR activity clustering, mechanism priors/pathway enrichment, top niche activity, "
            "downstream impact and expected-flow support. CRC uses a 7-LR small panel and lymph-node full uses a 9-LR first-pass panel; "
            "these support communication-program framing but are not whole-LR-database atlases."
        ),
    }
    (OUT_DIR / "phase11_communication_program_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(spatialself.to_string(index=False))


def summarize_dataset(
    dataset_key: str,
    cfg: dict[str, object],
    summary: pd.DataFrame,
    assignments: pd.DataFrame,
    enrichment: pd.DataFrame,
) -> list[dict[str, object]]:
    rows = []
    method_modules = summary.copy()
    for row in method_modules.itertuples(index=False):
        method = getattr(row, "method")
        module = getattr(row, "module")
        assign = assignments[(assignments["method"].eq(method)) & (assignments["module"].eq(module))]
        if assign.empty:
            assign = pd.DataFrame()
        families = program_counts(assign)
        top_family, top_count, top_fraction = top_program_family(families)
        pathway_term, pathway_fdr = pathway_info(cfg["kind"], row, enrichment, method, module)
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": cfg["label"],
                "method": method,
                "module": module,
                "status": getattr(row, "status", "module_summary"),
                "n_lr": int(getattr(row, "n_lr", len(assign))) if pd.notna(getattr(row, "n_lr", len(assign))) else int(len(assign)),
                "lr_pairs": normalize_lr_pairs(row, assign),
                "top_program_family": top_family,
                "top_program_family_count": top_count,
                "top_program_family_fraction": top_fraction,
                **{f"n_{clean_family_name(family)}": int(families.get(family, 0)) for family in PROGRAM_FAMILIES},
                "n_true_lr": int(getattr(row, "n_true_lr", assign.get("true_lr", pd.Series(dtype=bool)).sum() if not assign.empty else 0)),
                "n_weak_lr": int(getattr(row, "n_weak_lr", assign.get("weak_signal", pd.Series(dtype=bool)).sum() if not assign.empty else 0)),
                "n_high_expression_negative": int(
                    getattr(
                        row,
                        "n_high_expression_negative",
                        assign.get("false_positive_high_expression", pd.Series(dtype=bool)).sum() if not assign.empty else 0,
                    )
                ),
                "top_niche": getattr(row, "top_niche", ""),
                "top_cell_type": getattr(row, "top_cell_type", ""),
                "top_niche_mean_activity": to_float(getattr(row, "top_niche_mean_activity", np.nan)),
                "mean_downstream_delta_r2": to_float(getattr(row, "mean_downstream_delta_r2", np.nan)),
                "n_impact_lr_fdr_0_1": int(getattr(row, "n_impact_lr_fdr_0_1", 0) or 0),
                "mean_expected_flow_fraction": to_float(getattr(row, "mean_expected_flow_fraction", np.nan)),
                "n_best_flows_expected": int(getattr(row, "n_best_flows_expected", 0) or 0),
                "silhouette_score_method": to_float(getattr(row, "silhouette_score_method", np.nan)),
                "pathway_or_enrichment_term": pathway_term,
                "pathway_or_enrichment_fdr": pathway_fdr,
                "interpretation": interpret_module(dataset_key, top_family, row, assign),
            }
        )
    return rows


def program_counts(assign: pd.DataFrame) -> dict[str, int]:
    counts = {family: 0 for family in PROGRAM_FAMILIES}
    if assign.empty or "program_prior" not in assign:
        return counts
    for value in assign["program_prior"].fillna("unknown").astype(str):
        for family in PROGRAM_FAMILIES:
            if value == family:
                counts[family] += 1
                break
    return counts


def top_program_family(counts: dict[str, int]) -> tuple[str, int, float]:
    total = sum(counts.values())
    if total <= 0:
        return "unknown", 0, np.nan
    family, count = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return family, int(count), float(count / total)


def pathway_info(kind: str, row: object, enrichment: pd.DataFrame, method: str, module: str) -> tuple[str, float]:
    if kind == "crc_pathway":
        term = getattr(row, "best_pathway_term", "")
        fdr = to_float(getattr(row, "best_pathway_fdr", np.nan))
        return str(term), fdr
    if enrichment.empty:
        return "", np.nan
    sub = enrichment[(enrichment["method"].eq(method)) & (enrichment["module"].eq(module))]
    if sub.empty:
        return "", np.nan
    sub = sub.sort_values(["enrichment_ratio", "module_fraction"], ascending=[False, False])
    best = sub.iloc[0]
    return str(best.get("program_prior", "")), to_float(best.get("enrichment_ratio", np.nan))


def normalize_lr_pairs(row: object, assign: pd.DataFrame) -> str:
    for attr in ["lr_pairs", "lr_ids"]:
        value = getattr(row, attr, "")
        if isinstance(value, str) and value:
            return value
    if "lr_pair" in assign:
        return ";".join(assign["lr_pair"].astype(str))
    if "lr_id" in assign:
        return ";".join(assign["lr_id"].astype(str))
    return ""


def clean_family_name(family: str) -> str:
    return (
        family.replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("LR-like", "LR_like")
        .replace("__", "_")
    )


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def interpret_module(dataset_key: str, family: str, row: object, assign: pd.DataFrame) -> str:
    n_neg = int(getattr(row, "n_high_expression_negative", 0) or 0)
    flow = to_float(getattr(row, "mean_expected_flow_fraction", np.nan))
    delta = to_float(getattr(row, "mean_downstream_delta_r2", np.nan))
    if n_neg > 0:
        return f"{family} / negative-control module; interpret as leakage or background stress."
    if family in {"chemokine recruitment", "contact-dependent immune signaling"} and np.isfinite(flow) and flow > 0.4:
        return f"{family} module with spatial-flow support."
    if family in {"myeloid-remodeling", "angiogenesis", "immune checkpoint", "ECM/stromal remodeling"}:
        return f"{family} program localized to {getattr(row, 'top_niche', '')}."
    if np.isfinite(delta) and delta > 0.03:
        return f"{family} module with downstream target-response support."
    return f"{family} module; supportive but limited by panel size."


def build_family_matrix(spatialself: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in spatialself.itertuples(index=False):
        for family in PROGRAM_FAMILIES:
            rows.append(
                {
                    "dataset_key": row.dataset_key,
                    "dataset": row.dataset,
                    "module": row.module,
                    "program_family": family,
                    "n_lr": getattr(row, f"n_{clean_family_name(family)}"),
                    "module_total_lr": row.n_lr,
                    "module_fraction": getattr(row, f"n_{clean_family_name(family)}") / row.n_lr if row.n_lr else np.nan,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
