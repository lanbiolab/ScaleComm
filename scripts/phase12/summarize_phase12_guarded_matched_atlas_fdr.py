#!/usr/bin/env python
"""Compute guarded all-LR FDR using receiver/LR-identity plus matched spatial nulls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
OUT_DIR = PHASE12_DIR / "guarded_matched_atlas_fdr"

MATCHED_PATH = PHASE12_DIR / "matched_spatial_atlas_fdr/phase12_matched_spatial_atlas_fdr_lr.csv"
ELIGIBLE_PATH = PHASE12_DIR / "discovery_eligible_atlas_fdr/phase12_discovery_eligible_atlas_all_lr.csv"


CONFIGS = {
    "guarded_matched_spatial": [
        "receiver_global_empirical_p",
        "lr_identity_global_empirical_p",
        "matched_spatial_empirical_p",
    ],
    "guarded_matched_all_null": [
        "receiver_global_empirical_p",
        "lr_identity_global_empirical_p",
        "matched_all_null_empirical_p",
    ],
    "strict_guarded_matched_all_null": [
        "strict_receiver_global_empirical_p",
        "strict_lr_identity_global_empirical_p",
        "strict_matched_all_null_empirical_p",
    ],
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lr_table = build_lr_table()
    dataset_summary = summarize_datasets(lr_table)
    story = summarize_story(dataset_summary)

    lr_table.to_csv(OUT_DIR / "phase12_guarded_matched_atlas_fdr_lr.csv", index=False)
    dataset_summary.to_csv(OUT_DIR / "phase12_guarded_matched_atlas_fdr_dataset.csv", index=False)
    story.to_csv(OUT_DIR / "phase12_guarded_matched_atlas_fdr_story.csv", index=False)
    write_status(dataset_summary, story)
    manifest = {
        "task": "phase12_guarded_matched_atlas_fdr",
        "inputs": [rel(MATCHED_PATH), rel(ELIGIBLE_PATH)],
        "outputs": [
            rel(OUT_DIR / "phase12_guarded_matched_atlas_fdr_lr.csv"),
            rel(OUT_DIR / "phase12_guarded_matched_atlas_fdr_dataset.csv"),
            rel(OUT_DIR / "phase12_guarded_matched_atlas_fdr_story.csv"),
            rel(OUT_DIR / "phase12_guarded_matched_atlas_fdr_status.md"),
        ],
        "claim_boundary": (
            "Full-LR BH with matched spatial null and receiver/LR-identity guards. "
            "This is not the conservative pooled spatial-distance whole-atlas FDR."
        ),
    }
    (OUT_DIR / "phase12_guarded_matched_atlas_fdr_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(dataset_summary.to_string(index=False))
    print(story.to_string(index=False))


def build_lr_table() -> pd.DataFrame:
    matched = pd.read_csv(MATCHED_PATH)
    flags = pd.read_csv(
        ELIGIBLE_PATH,
        usecols=[
            "dataset_key",
            "lr_id",
            "primary_control_decoy",
            "receptor_shared_decoy",
            "ligand_shared_decoy",
            "decoy_class",
            "discovery_eligible",
        ],
    )
    table = matched.merge(flags, on=["dataset_key", "lr_id"], how="left")
    for col in ["primary_control_decoy", "receptor_shared_decoy", "ligand_shared_decoy", "discovery_eligible"]:
        table[col] = table[col].fillna(False).astype(bool)
    table["decoy_class"] = table["decoy_class"].fillna("unlabeled_background")
    for config_name, p_cols in CONFIGS.items():
        table[f"{config_name}_p"] = table[p_cols].max(axis=1)
        table[f"{config_name}_bh_q"] = np.nan
        for _, idx in table.groupby("dataset_key").groups.items():
            table.loc[idx, f"{config_name}_bh_q"] = bh(table.loc[idx, f"{config_name}_p"].to_numpy())
        table[f"{config_name}_supported_q_0_1"] = table[f"{config_name}_bh_q"] <= 0.1
    table["any_guarded_supported_q_0_1"] = table[
        [f"{name}_supported_q_0_1" for name in CONFIGS]
    ].any(axis=1)
    return table


def summarize_datasets(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_key, group in table.groupby("dataset_key"):
        row: dict[str, object] = {
            "dataset_key": dataset_key,
            "n_lr": len(group),
            "n_true_lr": int(group["true_lr"].sum()),
        }
        for config_name in CONFIGS:
            supported = group[group[f"{config_name}_supported_q_0_1"]]
            row[f"{config_name}_n_supported"] = len(supported)
            row[f"{config_name}_n_true_supported"] = int(supported["true_lr"].sum())
            row[f"{config_name}_n_primary_control_supported"] = int(supported["primary_control_decoy"].sum())
            row[f"{config_name}_n_receptor_shared_supported"] = int(supported["receptor_shared_decoy"].sum())
            row[f"{config_name}_n_ligand_shared_supported"] = int(supported["ligand_shared_decoy"].sum())
            row[f"{config_name}_n_background_supported"] = int(
                ((~supported["true_lr"]) & (~supported["primary_control_decoy"])).sum()
            )
            best_true = supported[supported["true_lr"]].sort_values(f"{config_name}_bh_q")
            row[f"{config_name}_best_true_lr"] = best_true.iloc[0]["lr_pair"] if len(best_true) else "none"
            row[f"{config_name}_best_true_q"] = (
                float(best_true.iloc[0][f"{config_name}_bh_q"]) if len(best_true) else np.nan
            )
        row["guarded_dataset_call"] = dataset_call(row)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_story(summary: pd.DataFrame) -> pd.DataFrame:
    config = "guarded_matched_all_null"
    n_supported_datasets = int((summary[f"{config}_n_true_supported"] > 0).sum())
    total_true = int(summary[f"{config}_n_true_supported"].sum())
    total_control = int(summary[f"{config}_n_primary_control_supported"].sum())
    total_receptor_shared = int(summary[f"{config}_n_receptor_shared_supported"].sum())
    if total_true > 0 and total_control == 0 and total_receptor_shared == 0:
        overall = "matched_guarded_support_for_selected_full_lr_discovery"
    else:
        overall = "matched_guarded_not_supportive_or_contaminated"
    return pd.DataFrame(
        [
            {
                "topic": "guarded_matched_full_lr_fdr",
                "overall_call": overall,
                "n_datasets": len(summary),
                "n_supported_datasets": n_supported_datasets,
                "total_true_supported": total_true,
                "total_primary_control_supported": total_control,
                "total_receptor_shared_supported": total_receptor_shared,
                "claim_boundary": (
                    "This supports a full-LR matched-null guarded discovery layer for selected signals, "
                    "not broad pooled spatial-distance FDR closure."
                ),
            }
        ]
    )


def dataset_call(row: dict[str, object]) -> str:
    true_supported = int(row["guarded_matched_all_null_n_true_supported"])
    control_supported = int(row["guarded_matched_all_null_n_primary_control_supported"])
    receptor_supported = int(row["guarded_matched_all_null_n_receptor_shared_supported"])
    background_supported = int(row["guarded_matched_all_null_n_background_supported"])
    if true_supported > 0 and control_supported == 0 and receptor_supported == 0 and background_supported == 0:
        return "clean_selected_full_lr_guarded_support"
    if true_supported > 0:
        return "guarded_support_with_contamination_caveat"
    return "not_guarded_matched_supported"


def bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = p[order[i]] * n / rank
        prev = min(prev, val)
        q[order[i]] = min(prev, 1.0)
    return q


def write_status(summary: pd.DataFrame, story: pd.DataFrame) -> None:
    overall = story.iloc[0]
    lines = [
        "# Phase 12 Guarded Matched Atlas FDR",
        "",
        "## Overall",
        "",
        f"- Overall call: `{overall.overall_call}`.",
        f"- Supported datasets: {int(overall.n_supported_datasets)}/{int(overall.n_datasets)}.",
        f"- True LR supported: {int(overall.total_true_supported)}.",
        f"- Primary-control supported: {int(overall.total_primary_control_supported)}.",
        f"- Receptor-shared supported: {int(overall.total_receptor_shared_supported)}.",
        "",
        "## Dataset Calls",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` `{row.guarded_dataset_call}`: "
            f"guarded matched-all true={int(row.guarded_matched_all_null_n_true_supported)}, "
            f"primary-control={int(row.guarded_matched_all_null_n_primary_control_supported)}, "
            f"receptor-shared={int(row.guarded_matched_all_null_n_receptor_shared_supported)}, "
            f"background={int(row.guarded_matched_all_null_n_background_supported)}, "
            f"best true={row.guarded_matched_all_null_best_true_lr}, "
            f"best q={fmt_num(row.guarded_matched_all_null_best_true_q)}."
        )
    lines += [
        "",
        "## Manuscript Boundary",
        "",
        "- Safe wording: full-LR guarded matched-null FDR supports selected SPP1-CD44-like discovery in post-Xenium, while lymph node and CRC remain unsupported under this layer.",
        "- Unsafe wording: broad pooled spatial-distance whole-atlas FDR is solved.",
    ]
    (OUT_DIR / "phase12_guarded_matched_atlas_fdr_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def fmt_num(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{val:.3g}"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
