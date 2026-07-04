#!/usr/bin/env python
"""Consolidate cross-platform biological story evidence for Phase 12."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE7_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
OUT_DIR = PROJECT_ROOT / "outputs/phase12/cross_platform_story"


DATASETS = [
    {
        "dataset_key": "post_xenium_colon",
        "platform": "post-Xenium colon cancer background",
        "method_control": PHASE7_DIR / "post_xenium_human_colon_cancer_method_control_summary.csv",
        "local_contrast": PHASE7_DIR / "post_xenium_colon_local_contrast_fdr_200perm/post_xenium_colon_local_contrast_control_summary.csv",
        "targeted_fdr": PHASE7_DIR / "post_xenium_colon_region_aware_targeted_fdr_200perm/post_xenium_colon_region_aware_targeted_fdr_audit.csv",
        "downstream_claims": PHASE7_DIR / "post_xenium_colon_clean_targets_fully_residual_pathway_downstream/post_xenium_colon_pathway_specific_downstream_claims.csv",
        "spatialself_methods": ["spatialself_calibrated", "spatialself_default"],
    },
    {
        "dataset_key": "spamosaic_tonsil_A1",
        "platform": "SpaMosaic tonsil A1",
        "method_control": PHASE7_DIR / "spamosaic_tonsil_A1_method_control_summary.csv",
        "local_contrast": PHASE7_DIR / "spamosaic_tonsil_A1_local_contrast_fdr_200perm/spamosaic_tonsil_A1_local_contrast_control_summary.csv",
        "targeted_fdr": PHASE7_DIR / "spamosaic_tonsil_A1_targeted_region_fdr_200perm/spamosaic_tonsil_A1_targeted_region_fdr_audit.csv",
        "downstream_claims": PHASE7_DIR / "spamosaic_tonsil_A1_rich_targets_fully_residual_pathway_downstream/spamosaic_tonsil_A1_pathway_specific_downstream_claims.csv",
        "spatialself_methods": ["spatialself_wavelet_edge_prior"],
    },
    {
        "dataset_key": "spamosaic_tonsil_D1",
        "platform": "SpaMosaic tonsil D1",
        "method_control": PHASE7_DIR / "spamosaic_tonsil_D1_method_control_summary.csv",
        "local_contrast": None,
        "targeted_fdr": PHASE7_DIR / "spamosaic_tonsil_D1_targeted_region_fdr_200perm/spamosaic_tonsil_D1_targeted_region_fdr_audit.csv",
        "downstream_claims": PHASE7_DIR / "spamosaic_tonsil_D1_rich_targets_fully_residual_pathway_downstream/spamosaic_tonsil_D1_pathway_specific_downstream_claims.csv",
        "spatialself_methods": ["spatialself_wavelet_edge_prior"],
    },
    {
        "dataset_key": "xenium_human_lymph_node",
        "platform": "Xenium human lymph node",
        "method_control": PHASE7_DIR / "xenium_human_lymph_node_method_control_summary.csv",
        "local_contrast": None,
        "targeted_fdr": PHASE7_DIR / "xenium_graphclust_validation_200perm/xenium_graphclust_control_summary.csv",
        "downstream_claims": PHASE7_DIR / "xenium_rich_targets_graphclust_residual_downstream_200perm/xenium_graphclust_residual_downstream_claims.csv",
        "spatialself_methods": ["spatialself_wavelet_edge_prior"],
    },
    {
        "dataset_key": "native_human_melanoma_multiome",
        "platform": "native human melanoma multiome",
        "method_control": PHASE7_DIR / "native_human_melanoma_multiome/native_melanoma_control_ranking.csv",
        "local_contrast": None,
        "targeted_fdr": PHASE7_DIR / "native_human_melanoma_multiome/native_melanoma_targeted_fdr_audit.csv",
        "downstream_claims": PHASE7_DIR / "native_human_melanoma_multiome/residual_downstream_200perm/native_melanoma_residual_downstream_claims.csv",
        "claim_audit": PHASE7_DIR / "native_human_melanoma_multiome/native_melanoma_claim_audit.csv",
        "spatialself_methods": ["spatialself_wavelet_edge_prior"],
    },
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame([audit_dataset(spec) for spec in DATASETS])
    audit = audit.sort_values(["story_score", "dataset_key"], ascending=[False, True]).reset_index(drop=True)
    audit.to_csv(OUT_DIR / "phase12_cross_platform_story_audit.csv", index=False)
    write_status(audit)
    manifest = {
        "task": "phase12_cross_platform_story_audit",
        "inputs": [rel(path) for spec in DATASETS for path in spec.values() if isinstance(path, Path) and path.exists()],
        "outputs": [
            rel(OUT_DIR / "phase12_cross_platform_story_audit.csv"),
            rel(OUT_DIR / "phase12_cross_platform_story_status.md"),
        ],
        "claim_boundary": "Consolidates existing Phase 7 outputs. It ranks story readiness; it does not create new biological ground truth or perturbation evidence.",
    }
    (OUT_DIR / "phase12_cross_platform_story_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(audit.to_string(index=False))


def audit_dataset(spec: dict[str, object]) -> dict[str, object]:
    method = pick_spatialself_method(read_csv(spec.get("method_control")), spec["spatialself_methods"])
    local = summarize_local_contrast(read_csv(spec.get("local_contrast")))
    targeted = summarize_targeted_fdr(read_csv(spec.get("targeted_fdr")), spec["dataset_key"], spec["spatialself_methods"])
    downstream = summarize_downstream(read_csv(spec.get("downstream_claims")), spec["spatialself_methods"])
    claim = summarize_claim_audit(read_csv(spec.get("claim_audit")), spec["spatialself_methods"])

    row = {
        "dataset_key": spec["dataset_key"],
        "platform": spec["platform"],
        **method,
        **local,
        **targeted,
        **downstream,
        **claim,
    }
    row["story_score"] = score(row)
    row["story_class"] = classify(row)
    row["recommended_use"] = recommend(row)
    return row


def pick_spatialself_method(df: pd.DataFrame, methods: list[str]) -> dict[str, object]:
    if df.empty:
        return {
            "spatialself_method": np.nan,
            "rank_validation_status": "missing",
            "best_positive_control": np.nan,
            "best_positive_rank": np.nan,
            "best_negative_control": np.nan,
            "best_negative_rank": np.nan,
            "positive_top3": np.nan,
            "negative_top3": np.nan,
        }
    sub = df[df["method"].astype(str).isin(methods)].copy()
    if sub.empty:
        sub = df[df["method"].astype(str).str.contains("spatialself", case=False, na=False)].copy()
    if sub.empty:
        return {
            "spatialself_method": np.nan,
            "rank_validation_status": "spatialself_missing",
            "best_positive_control": np.nan,
            "best_positive_rank": np.nan,
            "best_negative_control": np.nan,
            "best_negative_rank": np.nan,
            "positive_top3": np.nan,
            "negative_top3": np.nan,
        }
    if "rank_validation_status" not in sub.columns:
        sub["rank_validation_status"] = sub.apply(infer_rank_status, axis=1)
    sub["priority"] = sub["rank_validation_status"].astype(str).map(
        {
            "positive_prioritized_negative_suppressed": 0,
            "positive_top3_negative_suppressed": 1,
            "negative_control_ranks_as_high_or_higher": 3,
        }
    ).fillna(2)
    sub["positive_rank_numeric"] = pd.to_numeric(sub.get("best_positive_rank"), errors="coerce")
    row = sub.sort_values(["priority", "positive_rank_numeric"]).iloc[0]
    return {
        "spatialself_method": row.get("method"),
        "rank_validation_status": row.get("rank_validation_status", row.get("status", "unknown")),
        "best_positive_control": row.get("best_positive_control", row.get("best_positive")),
        "best_positive_rank": row.get("best_positive_rank"),
        "best_negative_control": row.get("best_negative_control", row.get("best_negative")),
        "best_negative_rank": row.get("best_negative_rank"),
        "positive_top3": row.get("positive_top3"),
        "negative_top3": row.get("negative_top3"),
        "weak_context_lr_rank": row.get("weak_context_lr_rank", np.nan),
    }


def infer_rank_status(row: pd.Series) -> str:
    pos_rank = safe_float(row.get("best_positive_rank"))
    neg_rank = safe_float(row.get("best_negative_rank"))
    pos_top3 = safe_float(row.get("positive_top3"))
    neg_top3 = safe_float(row.get("negative_top3"))
    if np.isfinite(pos_rank) and np.isfinite(neg_rank) and pos_rank < neg_rank and neg_top3 == 0:
        return "positive_prioritized_negative_suppressed"
    if np.isfinite(pos_top3) and np.isfinite(neg_top3) and pos_top3 > 0 and neg_top3 == 0:
        return "positive_top3_negative_suppressed"
    if np.isfinite(pos_rank) and np.isfinite(neg_rank) and neg_rank <= pos_rank:
        return "negative_control_ranks_as_high_or_higher"
    return "mixed_or_low_rank"


def summarize_local_contrast(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {
            "local_contrast_status": "missing",
            "best_local_positive": np.nan,
            "best_local_positive_fdr": np.nan,
            "best_local_negative": np.nan,
            "best_local_negative_fdr": np.nan,
            "n_local_positive_significant": np.nan,
            "n_local_negative_significant": np.nan,
        }
    best = df.sort_values("best_positive_local_contrast_fdr").iloc[0]
    statuses = ";".join(sorted(df["control_status"].dropna().astype(str).unique()))
    return {
        "local_contrast_status": statuses,
        "best_local_positive": best.get("best_positive_control"),
        "best_local_positive_fdr": best.get("best_positive_local_contrast_fdr"),
        "best_local_negative": best.get("best_negative_control"),
        "best_local_negative_fdr": best.get("best_negative_local_contrast_fdr"),
        "n_local_positive_significant": df.get("n_positive_significant", pd.Series(dtype=float)).max(),
        "n_local_negative_significant": df.get("n_negative_significant", pd.Series(dtype=float)).max(),
    }


def summarize_targeted_fdr(df: pd.DataFrame, dataset_key: str, methods: list[str]) -> dict[str, object]:
    if df.empty:
        return {
            "targeted_fdr_status": "missing",
            "n_targeted_positive_significant": np.nan,
            "n_targeted_negative_leakage": np.nan,
            "best_targeted_positive": np.nan,
            "best_targeted_positive_fdr": np.nan,
        }
    if "method" in df.columns:
        df = df[df["method"].astype(str).isin(methods) | df["method"].astype(str).str.contains("spatialself", case=False, na=False)]
    if df.empty:
        return {
            "targeted_fdr_status": "spatialself_missing",
            "n_targeted_positive_significant": np.nan,
            "n_targeted_negative_leakage": np.nan,
            "best_targeted_positive": np.nan,
            "best_targeted_positive_fdr": np.nan,
        }
    if "summary_type" in df.columns:
        fdr = df[df["summary_type"].astype(str).eq("group_fdr")]
        status = ";".join(sorted(fdr["control_status"].dropna().astype(str).unique())) if not fdr.empty else "no_group_fdr"
        pos_sig = pd.to_numeric(fdr.get("n_positive_significant"), errors="coerce").max() if not fdr.empty else np.nan
        neg_sig = pd.to_numeric(fdr.get("n_negative_significant"), errors="coerce").max() if not fdr.empty else np.nan
        best = fdr.sort_values("best_positive_group_bh_fdr").head(1)
        return {
            "targeted_fdr_status": status,
            "n_targeted_positive_significant": pos_sig,
            "n_targeted_negative_leakage": neg_sig,
            "best_targeted_positive": best["best_positive_control"].iloc[0] if len(best) else np.nan,
            "best_targeted_positive_fdr": best["best_positive_group_bh_fdr"].iloc[0] if len(best) else np.nan,
        }
    status_col = next(
        (
            c
            for c in ["targeted_region_fdr_status", "region_aware_fdr_status", "targeted_fdr_status"]
            if c in df.columns
        ),
        None,
    )
    if status_col is None:
        return {
            "targeted_fdr_status": "status_column_missing",
            "n_targeted_positive_significant": np.nan,
            "n_targeted_negative_leakage": np.nan,
            "best_targeted_positive": np.nan,
            "best_targeted_positive_fdr": np.nan,
        }
    pos = df[(~df["false_positive_high_expression"].astype(bool)) & df[status_col].astype(str).str.contains("positive_significant", na=False)]
    neg = df[(df["false_positive_high_expression"].astype(bool)) & df[status_col].astype(str).str.contains("leakage", na=False)]
    fdr_cols = [c for c in df.columns if c.endswith("_bh_fdr") and ("cell_type" in c or "lr_identity" in c or "spatial" in c or "within_region" in c)]
    if len(pos) and fdr_cols:
        pos = pos.assign(best_fdr=pos[fdr_cols].min(axis=1, skipna=True)).sort_values("best_fdr")
    return {
        "targeted_fdr_status": ";".join(sorted(df[status_col].dropna().astype(str).unique())),
        "n_targeted_positive_significant": len(pos),
        "n_targeted_negative_leakage": len(neg),
        "best_targeted_positive": pos["control_name"].iloc[0] if len(pos) else np.nan,
        "best_targeted_positive_fdr": pos["best_fdr"].iloc[0] if len(pos) and "best_fdr" in pos.columns else np.nan,
    }


def summarize_downstream(df: pd.DataFrame, methods: list[str]) -> dict[str, object]:
    if df.empty:
        return {
            "downstream_claim_status": "missing",
            "best_downstream_positive": np.nan,
            "best_downstream_fdr": np.nan,
            "best_downstream_negative": np.nan,
            "best_downstream_negative_fdr": np.nan,
        }
    sub = df[df["method"].astype(str).isin(methods) | df["method"].astype(str).str.contains("spatialself", case=False, na=False)].copy()
    if sub.empty:
        return {
            "downstream_claim_status": "spatialself_missing",
            "best_downstream_positive": np.nan,
            "best_downstream_fdr": np.nan,
            "best_downstream_negative": np.nan,
            "best_downstream_negative_fdr": np.nan,
        }
    row = sub.iloc[0]
    return {
        "downstream_claim_status": row.get("claim_status"),
        "best_downstream_positive": row.get("best_target_pair", row.get("best_positive_lr")),
        "best_downstream_fdr": row.get("best_target_lr_bh_fdr", np.nan),
        "best_downstream_negative": row.get("best_negative_pair", row.get("best_negative_lr")),
        "best_downstream_negative_fdr": row.get("best_negative_bh_fdr", np.nan),
    }


def summarize_claim_audit(df: pd.DataFrame, methods: list[str]) -> dict[str, object]:
    if df.empty:
        return {"external_claim_status": np.nan, "allowed_claim": np.nan}
    sub = df[df["method"].astype(str).isin(methods) | df["method"].astype(str).str.contains("spatialself", case=False, na=False)]
    if sub.empty:
        return {"external_claim_status": "spatialself_missing", "allowed_claim": np.nan}
    row = sub.iloc[0]
    return {"external_claim_status": row.get("claim_status"), "allowed_claim": row.get("allowed_claim")}


def score(row: dict[str, object]) -> float:
    score_value = 0.0
    if row.get("rank_validation_status") in {"positive_prioritized_negative_suppressed", "positive_top3_negative_suppressed"}:
        score_value += 2.0
    if safe_float(row.get("best_positive_rank")) <= 3:
        score_value += 1.0
    if safe_float(row.get("negative_top3")) == 0:
        score_value += 1.0
    if "positive_significant_negative_suppressed" in str(row.get("local_contrast_status")):
        score_value += 2.0
    if safe_float(row.get("n_targeted_positive_significant")) > 0:
        score_value += 1.0
    if safe_float(row.get("n_targeted_negative_leakage")) > 0:
        score_value -= 1.0
    if row.get("downstream_claim_status") and "no_clean" not in str(row.get("downstream_claim_status")):
        score_value += 1.0
    if row.get("external_claim_status") and "not_clean" in str(row.get("external_claim_status")):
        score_value -= 2.0
    return score_value


def classify(row: dict[str, object]) -> str:
    rank_clean = row.get("rank_validation_status") in {"positive_prioritized_negative_suppressed", "positive_top3_negative_suppressed"}
    local_clean = "positive_significant_negative_suppressed" in str(row.get("local_contrast_status"))
    targeted_pos = safe_float(row.get("n_targeted_positive_significant")) > 0
    targeted_leak = safe_float(row.get("n_targeted_negative_leakage")) > 0
    downstream_clean = row.get("downstream_claim_status") and "no_clean" not in str(row.get("downstream_claim_status"))
    if rank_clean and local_clean and downstream_clean and not targeted_leak:
        return "main_story_ready"
    if rank_clean and local_clean and not targeted_leak:
        return "strong_ranking_fdr_downstream_caveat"
    if rank_clean and targeted_pos and targeted_leak:
        return "ranking_support_targeted_fdr_negative_leakage"
    if rank_clean:
        return "ranking_support_only"
    if "not_clean" in str(row.get("external_claim_status")):
        return "not_clean_real_sample_story"
    return "insufficient_or_mixed"


def recommend(row: dict[str, object]) -> str:
    cls = row.get("story_class")
    if cls == "main_story_ready":
        return "Can be used as a main biological story with current evidence."
    if cls == "strong_ranking_fdr_downstream_caveat":
        return "Good second-story candidate if framed with downstream caveat."
    if cls == "ranking_support_targeted_fdr_negative_leakage":
        return "Use as supportive platform validation; disclose targeted negative leakage."
    if cls == "ranking_support_only":
        return "Use only as ranking/platform support unless FDR or downstream improves."
    if cls == "not_clean_real_sample_story":
        return "Do not use as a success story under current controls."
    return "Keep as exploratory or supplementary."


def read_csv(path: object) -> pd.DataFrame:
    if not isinstance(path, Path) or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_float(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_status(audit: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Cross-Platform Story Audit",
        "",
        "## Main Ranking",
        "",
    ]
    for row in audit.itertuples(index=False):
        lines.append(
            f"- **{row.dataset_key}**: `{row.story_class}`; best positive {row.best_positive_control} rank {fmt(row.best_positive_rank)}, "
            f"best negative {row.best_negative_control} rank {fmt(row.best_negative_rank)}, rank status `{row.rank_validation_status}`; "
            f"local contrast `{row.local_contrast_status}`; targeted positives {fmt(row.n_targeted_positive_significant)}, "
            f"targeted negative leakage {fmt(row.n_targeted_negative_leakage)}; downstream `{row.downstream_claim_status}`. {row.recommended_use}"
        )
    lines += [
        "",
        "## Practical Interpretation",
        "",
        "- Post-Xenium colon remains the strongest main case because local-contrast FDR suppresses negatives, but its pathway-specific downstream remains leakage-caveated.",
        "- SpaMosaic tonsil A1 is the best non-CRC second-story candidate among the audited outputs: SpatialSelf ranks positives ahead of negatives and local-contrast FDR suppresses negatives, but pathway-specific downstream is not clean.",
        "- SpaMosaic tonsil D1 supports ranking and cell-type FDR, but lacks the same local-contrast summary in the current files.",
        "- Xenium lymph node is useful as high-resolution platform support under graphclust residual/FDR, but raw ranking still has a negative top hit.",
        "- Native melanoma should not be framed as a success story under the current control panel.",
    ]
    (OUT_DIR / "phase12_cross_platform_story_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: object) -> str:
    val = safe_float(value)
    if not np.isfinite(val):
        return "NA"
    if abs(val) >= 100:
        return f"{val:.0f}"
    if abs(val) >= 10:
        return f"{val:.2f}"
    return f"{val:.3g}"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
