#!/usr/bin/env python
"""Discovery-eligible audit for integrated coordinate-rerun smoke.

This uses the same label-independent eligibility idea as the Phase 12
discovery-eligible atlas audit, but applies it to the integrated guarded spatial
coordinate-rerun FDR table. It asks whether the end-to-end coordinate-null
support survives after removing weakly annotated broad/background LR pairs from
the BH family.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE12_DIR = PROJECT_ROOT / "outputs/phase12"
SMOKE_ROOT = PHASE12_DIR / "integrated_guarded_spatial_coordinate_rerun_smoke"
OUT_DIR = PHASE12_DIR / "integrated_guarded_spatial_coordinate_eligible_audit"
Q_THRESHOLD = 0.10


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fdr_path = best_fdr_path()
    if fdr_path is None:
        raise FileNotFoundError(f"No integrated coordinate FDR files found under {SMOKE_ROOT}")
    table = pd.read_csv(fdr_path)
    audited = build_audit_table(table)
    eligible = audited[audited["coordinate_discovery_eligible"].astype(bool)].copy()
    dataset = summarize_dataset(audited, eligible, fdr_path)
    low_p = summarize_low_p(eligible)

    audited.to_csv(OUT_DIR / "phase12_integrated_coordinate_eligible_all_lr.csv", index=False)
    eligible.to_csv(OUT_DIR / "phase12_integrated_coordinate_eligible_lr.csv", index=False)
    dataset.to_csv(OUT_DIR / "phase12_integrated_coordinate_eligible_dataset.csv", index=False)
    low_p.to_csv(OUT_DIR / "phase12_integrated_coordinate_eligible_low_p_lr.csv", index=False)
    write_status(dataset, low_p)
    write_manifest(fdr_path)

    print(dataset.to_string(index=False))
    print()
    print(low_p.head(40).to_string(index=False))


def best_fdr_path() -> Path | None:
    paths = sorted(SMOKE_ROOT.glob("*/*_integrated_coordinate_lr_fdr.csv"))
    if not paths:
        return None
    rows = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["n_completed_permutations"])
        n_perm = int(pd.to_numeric(frame["n_completed_permutations"], errors="coerce").fillna(0).max())
        rows.append((n_perm, path))
    rows.sort(key=lambda item: (item[0], str(item[1])))
    return rows[-1][1]


def build_audit_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["true_lr"] = out["true_lr"].fillna(False).astype(bool)
    out["false_positive_high_expression"] = out["false_positive_high_expression"].fillna(False).astype(bool)
    out["coordinate_discovery_eligible"] = (
        out["biological_category"].astype(str).ne("other")
        & pd.to_numeric(out["n_nonzero_expr"], errors="coerce").fillna(0).ge(3)
        & pd.to_numeric(out["expr_gate"], errors="coerce").fillna(0).ge(0.5)
        & pd.to_numeric(out["ligand_positive_fraction"], errors="coerce").fillna(0).gt(0)
        & pd.to_numeric(out["receptor_positive_fraction"], errors="coerce").fillna(0).gt(0)
    )
    out["receptor_shared_decoy"] = False
    out["ligand_shared_decoy"] = False
    for dataset_key, sub in out.groupby("dataset_key", sort=False):
        true_receptors = set(sub[sub["true_lr"]]["receptor"].astype(str))
        true_ligands = set(sub[sub["true_lr"]]["ligand"].astype(str))
        mask = out["dataset_key"].eq(dataset_key)
        out.loc[mask, "receptor_shared_decoy"] = (
            out.loc[mask, "receptor"].astype(str).isin(true_receptors)
            & ~out.loc[mask, "true_lr"].astype(bool)
        )
        out.loc[mask, "ligand_shared_decoy"] = (
            out.loc[mask, "ligand"].astype(str).isin(true_ligands)
            & ~out.loc[mask, "true_lr"].astype(bool)
        )
    out["primary_control_decoy"] = (
        out["false_positive_high_expression"] | out["receptor_shared_decoy"] | out["ligand_shared_decoy"]
    )
    out["decoy_class"] = np.select(
        [
            out["true_lr"],
            out["false_positive_high_expression"],
            out["receptor_shared_decoy"],
            out["ligand_shared_decoy"],
        ],
        [
            "prespecified_positive_anchor",
            "high_expression_control",
            "receptor_shared_control",
            "ligand_shared_control",
        ],
        default="unlabeled_background_lr",
    )

    out["eligible_coordinate_bh_fdr"] = np.nan
    out["eligible_strict_coordinate_bh_fdr"] = np.nan
    for _, sub in out.groupby("dataset_key", sort=False):
        idx = sub.index[sub["coordinate_discovery_eligible"].astype(bool)]
        out.loc[idx, "eligible_coordinate_bh_fdr"] = bh_adjust(
            pd.to_numeric(out.loc[idx, "coordinate_rerun_p_value"], errors="coerce").to_numpy(dtype=float)
        )
        out.loc[idx, "eligible_strict_coordinate_bh_fdr"] = bh_adjust(
            pd.to_numeric(out.loc[idx, "strict_coordinate_rerun_p_value"], errors="coerce").to_numpy(dtype=float)
        )
    out["eligible_coordinate_call"] = [classify_lr(row) for row in out.itertuples(index=False)]
    return out.sort_values(["dataset_key", "rank", "lr_pair"]).reset_index(drop=True)


def summarize_dataset(all_lr: pd.DataFrame, eligible: pd.DataFrame, fdr_path: Path) -> pd.DataFrame:
    rows = []
    for dataset_key, all_sub in all_lr.groupby("dataset_key", sort=False):
        sub = eligible[eligible["dataset_key"].eq(dataset_key)].copy()
        q_hits = sub[sub["eligible_coordinate_bh_fdr"].le(Q_THRESHOLD)]
        strict_q_hits = sub[sub["eligible_strict_coordinate_bh_fdr"].le(Q_THRESHOLD)]
        best_true = sub[sub["true_lr"]].sort_values(["eligible_coordinate_bh_fdr", "rank"]).head(1)
        best_control = sub[sub["primary_control_decoy"]].sort_values(["eligible_coordinate_bh_fdr", "rank"]).head(1)
        rows.append(
            {
                "dataset_key": dataset_key,
                "source_fdr_file": rel(fdr_path),
                "n_total_lr": int(len(all_sub)),
                "n_eligible_lr": int(len(sub)),
                "n_eligible_true_lr": int(sub["true_lr"].sum()),
                "n_eligible_primary_control_decoy": int(sub["primary_control_decoy"].sum()),
                "n_eligible_high_expression_control": int(sub["false_positive_high_expression"].sum()),
                "n_eligible_unlabeled_background": int(sub["decoy_class"].eq("unlabeled_background_lr").sum()),
                "eligible_coordinate_q_0_1": int(len(q_hits)),
                "eligible_coordinate_true_q_0_1": int(q_hits["true_lr"].sum()) if len(q_hits) else 0,
                "eligible_coordinate_control_q_0_1": int(q_hits["primary_control_decoy"].sum()) if len(q_hits) else 0,
                "eligible_coordinate_background_q_0_1": int(
                    q_hits["decoy_class"].eq("unlabeled_background_lr").sum()
                )
                if len(q_hits)
                else 0,
                "eligible_strict_coordinate_q_0_1": int(len(strict_q_hits)),
                "eligible_strict_coordinate_true_q_0_1": int(strict_q_hits["true_lr"].sum())
                if len(strict_q_hits)
                else 0,
                "best_true_lr": str(best_true.iloc[0]["lr_pair"]) if len(best_true) else "",
                "best_true_eligible_coordinate_q": float(best_true.iloc[0]["eligible_coordinate_bh_fdr"])
                if len(best_true)
                else np.nan,
                "best_true_coordinate_p": float(best_true.iloc[0]["coordinate_rerun_p_value"])
                if len(best_true)
                else np.nan,
                "best_control_lr": str(best_control.iloc[0]["lr_pair"]) if len(best_control) else "",
                "best_control_eligible_coordinate_q": float(best_control.iloc[0]["eligible_coordinate_bh_fdr"])
                if len(best_control)
                else np.nan,
                "eligible_coordinate_call": classify_dataset(q_hits),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_key")


def summarize_low_p(eligible: pd.DataFrame) -> pd.DataFrame:
    if eligible.empty:
        return pd.DataFrame()
    p_min = float(pd.to_numeric(eligible["coordinate_rerun_p_value"], errors="coerce").dropna().min())
    cols = [
        "dataset_key",
        "rank",
        "lr_pair",
        "true_lr",
        "primary_control_decoy",
        "decoy_class",
        "biological_category",
        "selected_reason",
        "coordinate_rerun_p_value",
        "eligible_coordinate_bh_fdr",
        "calibrated_score",
    ]
    return eligible[eligible["coordinate_rerun_p_value"].le(p_min + 1e-12)][cols].sort_values(
        ["rank", "lr_pair"]
    )


def classify_lr(row: object) -> str:
    eligible = bool(getattr(row, "coordinate_discovery_eligible", False))
    if not eligible:
        return "not_coordinate_discovery_eligible"
    q = float(getattr(row, "eligible_coordinate_bh_fdr", np.nan))
    if not np.isfinite(q) or q > Q_THRESHOLD:
        return "eligible_not_coordinate_fdr_supported"
    if bool(getattr(row, "true_lr", False)):
        return "eligible_true_coordinate_fdr_supported"
    if bool(getattr(row, "primary_control_decoy", False)):
        return "eligible_control_coordinate_fdr_leakage"
    return "eligible_background_coordinate_fdr_leakage"


def classify_dataset(q_hits: pd.DataFrame) -> str:
    if q_hits.empty:
        return "eligible_coordinate_fdr_not_supported"
    true_hits = int(q_hits["true_lr"].sum())
    control_hits = int(q_hits["primary_control_decoy"].sum())
    background_hits = int(q_hits["decoy_class"].eq("unlabeled_background_lr").sum())
    if true_hits > 0 and control_hits == 0 and background_hits == 0:
        return "eligible_coordinate_true_clean"
    if true_hits > 0 and control_hits == 0:
        return "eligible_coordinate_true_with_background_leakage"
    if true_hits > 0:
        return "eligible_coordinate_true_with_control_leakage"
    return "eligible_coordinate_no_true_support"


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    idx = np.where(valid)[0]
    vals = p[idx]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / float(i + 1))
        adjusted[i] = running
    out[idx[order]] = np.clip(adjusted, 0.0, 1.0)
    return out


def write_status(dataset: pd.DataFrame, low_p: pd.DataFrame) -> None:
    row = dataset.iloc[0]
    lines = [
        "# Phase 12 Integrated Coordinate Eligible Audit",
        "",
        "## Result",
        "",
        f"- Dataset: `{row.dataset_key}`",
        f"- Eligible LR: {int(row.n_eligible_lr)}/{int(row.n_total_lr)}",
        f"- Eligible q<=0.1: total={int(row.eligible_coordinate_q_0_1)}, true={int(row.eligible_coordinate_true_q_0_1)}, controls={int(row.eligible_coordinate_control_q_0_1)}, background={int(row.eligible_coordinate_background_q_0_1)}",
        f"- Best true: `{row.best_true_lr}` p={float(row.best_true_coordinate_p):.4g}, eligible q={float(row.best_true_eligible_coordinate_q):.4g}",
        f"- Best control: `{row.best_control_lr}` eligible q={float(row.best_control_eligible_coordinate_q):.4g}",
        f"- Call: `{row.eligible_coordinate_call}`",
        "",
        "## Claim Boundary",
        "",
        "This is an eligibility-restricted coordinate-null audit. It can support a scoped, functionally annotated LR-universe statement, but it does not replace broad whole-atlas FDR.",
    ]
    (OUT_DIR / "phase12_integrated_coordinate_eligible_status.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifest(fdr_path: Path) -> None:
    manifest = {
        "task": "phase12_integrated_coordinate_eligible_audit",
        "input": rel(fdr_path),
        "outputs": {
            "all_lr": rel(OUT_DIR / "phase12_integrated_coordinate_eligible_all_lr.csv"),
            "eligible_lr": rel(OUT_DIR / "phase12_integrated_coordinate_eligible_lr.csv"),
            "dataset": rel(OUT_DIR / "phase12_integrated_coordinate_eligible_dataset.csv"),
            "low_p_lr": rel(OUT_DIR / "phase12_integrated_coordinate_eligible_low_p_lr.csv"),
            "status": rel(OUT_DIR / "phase12_integrated_coordinate_eligible_status.md"),
        },
        "claim_boundary": "Eligibility-restricted coordinate-rerun audit; not broad whole-atlas FDR.",
    }
    (OUT_DIR / "phase12_integrated_coordinate_eligible_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
