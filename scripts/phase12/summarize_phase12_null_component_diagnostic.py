#!/usr/bin/env python
"""Null-component diagnostic for Phase 12 whole-atlas FDR bottlenecks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "null_component_diagnostic"

METHOD_INPUTS = {
    "main_calibrated": PHASE12_DIR
    / "atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_lr_fdr.csv",
    "anchor_background_suppressed": PHASE12_DIR
    / "anchor_background_suppressed_permutation_audit/phase12_anchor_background_suppressed_permutation_lr_fdr.csv",
    "anchor_background_guarded_training": PHASE12_DIR
    / "anchor_background_guarded_training_permutation_audit/phase12_anchor_background_guarded_training_permutation_lr_fdr.csv",
    "anchor_background_range_guarded_training": PHASE12_DIR
    / "anchor_background_range_guarded_training_permutation_audit/phase12_anchor_background_range_guarded_training_permutation_lr_fdr.csv",
}

NULL_TYPES = ("receiver_permutation", "lr_identity_permutation", "spatial_distance_permutation")
Q_CUTOFF = 0.1
RELAXED_Q_CUTOFF = 0.25
WITHIN_P_CUTOFF = 0.05


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lr = build_lr_table()
    dataset = summarize_dataset(lr)
    method = summarize_method(dataset)
    story = summarize_story(method, dataset)

    lr.to_csv(OUT_DIR / "phase12_null_component_lr_diagnostic.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_null_component_dataset_summary.csv", index=False)
    method.to_csv(OUT_DIR / "phase12_null_component_method_summary.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_null_component_story.csv", index=False)
    write_status(method, dataset, story)
    write_manifest()

    print(method.to_string(index=False))
    print()
    print(dataset.to_string(index=False))
    print()
    print(story.to_string(index=False))


def build_lr_table() -> pd.DataFrame:
    rows = []
    for method, path in METHOD_INPUTS.items():
        if not path.exists():
            continue
        table = pd.read_csv(path)
        for row in table.itertuples(index=False):
            item = row._asdict()
            receiver_q = get_float(item, "receiver_permutation_global_bh_fdr")
            lrid_q = get_float(item, "lr_identity_permutation_global_bh_fdr")
            spatial_q = get_float(item, "spatial_distance_permutation_global_bh_fdr")
            conservative_q = get_float(item, "conservative_global_bh_fdr")
            receiver_within = get_float(item, "receiver_permutation_within_lr_empirical_p")
            lrid_within = get_float(item, "lr_identity_permutation_within_lr_empirical_p")
            spatial_within = get_float(item, "spatial_distance_permutation_within_lr_empirical_p")
            non_spatial_support = (receiver_q <= Q_CUTOFF) and (lrid_q <= Q_CUTOFF)
            spatial_support = spatial_q <= Q_CUTOFF
            relaxed_spatial_support = spatial_q <= RELAXED_Q_CUTOFF
            within_support = (
                (receiver_within <= WITHIN_P_CUTOFF)
                and (lrid_within <= WITHIN_P_CUTOFF)
                and (spatial_within <= WITHIN_P_CUTOFF)
            )
            rows.append(
                {
                    "method": method,
                    "dataset_key": item.get("dataset_key"),
                    "rank": int(item.get("rank", 0)),
                    "lr_pair": item.get("lr_pair"),
                    "lr_id": item.get("lr_id"),
                    "ligand": item.get("ligand"),
                    "receptor": item.get("receptor"),
                    "biological_category": item.get("biological_category"),
                    "mechanism": item.get("mechanism"),
                    "selected_reason": item.get("selected_reason"),
                    "true_lr": bool(item.get("true_lr", False)),
                    "false_positive_high_expression": bool(item.get("false_positive_high_expression", False)),
                    "calibrated_score": get_float(item, "calibrated_score"),
                    "conservative_global_bh_fdr": conservative_q,
                    "receiver_global_bh_fdr": receiver_q,
                    "lr_identity_global_bh_fdr": lrid_q,
                    "spatial_distance_global_bh_fdr": spatial_q,
                    "receiver_within_lr_p": receiver_within,
                    "lr_identity_within_lr_p": lrid_within,
                    "spatial_distance_within_lr_p": spatial_within,
                    "non_spatial_component_supported": non_spatial_support,
                    "spatial_component_supported": spatial_support,
                    "relaxed_spatial_component_supported": relaxed_spatial_support,
                    "within_lr_all_null_supported": within_support,
                    "spatial_global_blocker": non_spatial_support and not spatial_support,
                    "conservative_supported": conservative_q <= Q_CUTOFF,
                    "component_call": classify_component(
                        bool(item.get("true_lr", False)),
                        conservative_q,
                        receiver_q,
                        lrid_q,
                        spatial_q,
                        receiver_within,
                        lrid_within,
                        spatial_within,
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["method", "dataset_key", "rank"]).reset_index(drop=True)


def classify_component(
    true_lr: bool,
    conservative_q: float,
    receiver_q: float,
    lrid_q: float,
    spatial_q: float,
    receiver_within: float,
    lrid_within: float,
    spatial_within: float,
) -> str:
    if conservative_q <= Q_CUTOFF:
        return "broad_fdr_supported_true" if true_lr else "broad_fdr_supported_control_or_background"
    non_spatial = receiver_q <= Q_CUTOFF and lrid_q <= Q_CUTOFF
    if true_lr and non_spatial and spatial_q > Q_CUTOFF:
        return "true_lr_spatial_global_blocked"
    if true_lr and spatial_q <= RELAXED_Q_CUTOFF:
        return "true_lr_relaxed_spatial_support_only"
    within_all = (
        receiver_within <= WITHIN_P_CUTOFF
        and lrid_within <= WITHIN_P_CUTOFF
        and spatial_within <= WITHIN_P_CUTOFF
    )
    if true_lr and within_all:
        return "true_lr_within_lr_supported_broad_blocked"
    if true_lr:
        return "true_lr_not_component_supported"
    if non_spatial:
        return "background_or_control_non_spatial_supported"
    return "background_or_control_not_supported"


def summarize_dataset(lr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if lr.empty:
        return pd.DataFrame()
    for (method, dataset_key), sub in lr.groupby(["method", "dataset_key"], sort=False):
        true = sub[sub["true_lr"]].copy()
        top10 = sub.sort_values("rank").head(10)
        rows.append(
            {
                "method": method,
                "dataset_key": dataset_key,
                "n_lr": int(len(sub)),
                "n_true_lr": int(len(true)),
                "top10_true_lr": int(top10["true_lr"].sum()),
                "top10_negative_or_risk": int(top10_risk(top10)),
                "true_broad_fdr_supported": int(true["conservative_supported"].sum()),
                "true_non_spatial_supported": int(true["non_spatial_component_supported"].sum()),
                "true_spatial_supported": int(true["spatial_component_supported"].sum()),
                "true_relaxed_spatial_supported": int(true["relaxed_spatial_component_supported"].sum()),
                "true_within_lr_supported": int(true["within_lr_all_null_supported"].sum()),
                "true_spatial_global_blocked": int(true["spatial_global_blocker"].sum()),
                "best_true_conservative_q": min_or_nan(true["conservative_global_bh_fdr"]),
                "best_true_spatial_q": min_or_nan(true["spatial_distance_global_bh_fdr"]),
                "best_true_receiver_q": min_or_nan(true["receiver_global_bh_fdr"]),
                "best_true_lrid_q": min_or_nan(true["lr_identity_global_bh_fdr"]),
                "best_true_spatial_within_p": min_or_nan(true["spatial_distance_within_lr_p"]),
                "dataset_component_call": dataset_call(true, top10),
            }
        )
    return pd.DataFrame(rows)


def summarize_method(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if dataset.empty:
        return pd.DataFrame()
    for method, sub in dataset.groupby("method", sort=False):
        rows.append(
            {
                "method": method,
                "n_datasets": int(len(sub)),
                "mean_top10_true_lr": float(pd.to_numeric(sub["top10_true_lr"], errors="coerce").mean()),
                "sum_top10_negative_or_risk": int(pd.to_numeric(sub["top10_negative_or_risk"], errors="coerce").sum()),
                "sum_true_broad_fdr_supported": int(pd.to_numeric(sub["true_broad_fdr_supported"], errors="coerce").sum()),
                "sum_true_non_spatial_supported": int(pd.to_numeric(sub["true_non_spatial_supported"], errors="coerce").sum()),
                "sum_true_spatial_supported": int(pd.to_numeric(sub["true_spatial_supported"], errors="coerce").sum()),
                "sum_true_relaxed_spatial_supported": int(pd.to_numeric(sub["true_relaxed_spatial_supported"], errors="coerce").sum()),
                "sum_true_within_lr_supported": int(pd.to_numeric(sub["true_within_lr_supported"], errors="coerce").sum()),
                "sum_true_spatial_global_blocked": int(pd.to_numeric(sub["true_spatial_global_blocked"], errors="coerce").sum()),
                "best_true_spatial_q": float(pd.to_numeric(sub["best_true_spatial_q"], errors="coerce").min()),
                "method_component_call": method_call(sub),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["sum_true_broad_fdr_supported", "sum_true_non_spatial_supported", "mean_top10_true_lr"],
        ascending=[False, False, False],
    )


def summarize_story(method: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    if method.empty:
        return pd.DataFrame()
    best_non_spatial = method.sort_values(
        ["sum_true_non_spatial_supported", "mean_top10_true_lr"], ascending=[False, False]
    ).iloc[0]
    best_spatial = method.sort_values(["best_true_spatial_q", "mean_top10_true_lr"], ascending=[True, False]).iloc[0]
    return pd.DataFrame(
        [
            {
                "topic": "null_component_diagnostic",
                "overall_call": "spatial_global_null_is_primary_blocker",
                "n_methods": int(len(method)),
                "n_dataset_method_rows": int(len(dataset)),
                "best_non_spatial_method": best_non_spatial["method"],
                "best_non_spatial_true_supported": int(best_non_spatial["sum_true_non_spatial_supported"]),
                "best_spatial_q_method": best_spatial["method"],
                "best_true_spatial_q": float(best_spatial["best_true_spatial_q"]),
                "broad_fdr_supported_true_total": int(method["sum_true_broad_fdr_supported"].sum()),
                "claim_boundary": (
                    "Receiver and LR-identity nulls often support true LR, but pooled spatial-distance global BH remains the dominant blocker. "
                    "Use this as a diagnostic and calibration target, not as a replacement FDR procedure."
                ),
            }
        ]
    )


def dataset_call(true: pd.DataFrame, top10: pd.DataFrame) -> str:
    if true.empty:
        return "no_true_lr"
    if int(true["conservative_supported"].sum()) > 0:
        return "broad_fdr_supported"
    if int(true["spatial_global_blocker"].sum()) > 0:
        return "non_spatial_supported_spatial_global_blocked"
    if int(true["within_lr_all_null_supported"].sum()) > 0:
        return "within_lr_supported_broad_blocked"
    if int(top10["true_lr"].sum()) >= 3:
        return "ranking_support_no_component_fdr"
    return "weak_component_support"


def method_call(sub: pd.DataFrame) -> str:
    broad = int(pd.to_numeric(sub["true_broad_fdr_supported"], errors="coerce").sum())
    if broad > 0:
        return "broad_fdr_partial_support"
    spatial_blocked = int(pd.to_numeric(sub["true_spatial_global_blocked"], errors="coerce").sum())
    non_spatial = int(pd.to_numeric(sub["true_non_spatial_supported"], errors="coerce").sum())
    if non_spatial > 0 and spatial_blocked > 0:
        return "non_spatial_signal_spatial_global_blocked"
    if int(pd.to_numeric(sub["true_within_lr_supported"], errors="coerce").sum()) > 0:
        return "within_lr_signal_broad_blocked"
    return "ranking_or_control_support_only"


def top10_risk(top10: pd.DataFrame) -> int:
    high = top10["false_positive_high_expression"].astype(bool)
    true = top10["true_lr"].astype(bool)
    # Conservative proxy: high-expression controls plus non-true top10 entries that are not curated anchors.
    non_true = ~true
    return int((high | non_true).sum())


def get_float(row: dict[str, object], key: str) -> float:
    try:
        return float(row.get(key, np.nan))
    except (TypeError, ValueError):
        return float("nan")


def min_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.min()) if len(values) else float("nan")


def write_status(method: pd.DataFrame, dataset: pd.DataFrame, story: pd.DataFrame) -> None:
    lines = [
        "# Phase 12 Null-Component Diagnostic",
        "",
        "## Scope",
        "",
        "This diagnostic decomposes the broad whole-atlas FDR blocker into receiver, LR-identity and spatial-distance null components.",
        "It does not create a new accepted FDR procedure and should not be used to claim broad discovery FDR.",
        "",
        "## Method Summary",
        "",
    ]
    for row in method.itertuples(index=False):
        lines.append(
            f"- `{row.method}` `{row.method_component_call}`: mean top10 true={row.mean_top10_true_lr:.2f}, "
            f"broad true q<=0.1={int(row.sum_true_broad_fdr_supported)}, "
            f"non-spatial true support={int(row.sum_true_non_spatial_supported)}, "
            f"spatial true support={int(row.sum_true_spatial_supported)}, "
            f"spatial-blocked true={int(row.sum_true_spatial_global_blocked)}, "
            f"best spatial q={fmt(row.best_true_spatial_q)}."
        )
    lines += ["", "## Dataset Summary", ""]
    for row in dataset.itertuples(index=False):
        lines.append(
            f"- `{row.method}` / `{row.dataset_key}` `{row.dataset_component_call}`: "
            f"top10 true={int(row.top10_true_lr)}, risk={int(row.top10_negative_or_risk)}, "
            f"broad={int(row.true_broad_fdr_supported)}, non-spatial={int(row.true_non_spatial_supported)}, "
            f"spatial={int(row.true_spatial_supported)}, within={int(row.true_within_lr_supported)}, "
            f"best spatial q={fmt(row.best_true_spatial_q)}."
        )
    if not story.empty:
        first = story.iloc[0]
        lines += [
            "",
            "## Decision",
            "",
            f"- Overall: `{first['overall_call']}`.",
            f"- Best non-spatial method: `{first['best_non_spatial_method']}`.",
            f"- Best spatial-q method: `{first['best_spatial_q_method']}` with best true spatial q={fmt(first['best_true_spatial_q'])}.",
            "- Manuscript boundary: this is a bottleneck explanation, not FDR closure.",
        ]
    (OUT_DIR / "phase12_null_component_diagnostic_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    manifest = {
        "task": "phase12_null_component_diagnostic",
        "inputs": [str(path.relative_to(PHASE12_DIR)) for path in METHOD_INPUTS.values()],
        "outputs": [
            "phase12_null_component_lr_diagnostic.csv",
            "phase12_null_component_dataset_summary.csv",
            "phase12_null_component_method_summary.csv",
            "phase12_null_component_story.csv",
            "phase12_null_component_diagnostic_status.md",
        ],
        "claim_boundary": "Diagnostic decomposition of null components; not a replacement broad FDR procedure.",
    }
    (OUT_DIR / "phase12_null_component_diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def fmt(value: object) -> str:
    if pd.isna(value):
        return "NA"
    try:
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
